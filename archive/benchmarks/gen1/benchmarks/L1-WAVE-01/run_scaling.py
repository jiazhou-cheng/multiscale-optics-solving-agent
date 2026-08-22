"""Run the CHE-15 Chromatix grid, padding, determinism, and CPU benchmark."""

from __future__ import annotations

# Standard-library-only worker startup makes solver import time observable.
import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

SCRIPT_START_NS = time.perf_counter_ns()
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BENCHMARK_ID = "L1-WAVE-01"
PROTOCOL_ID = "M1-BASELINE-CPU-V1"
DEFAULT_GRIDS = (64, 128, 256)
DEFAULT_REPEATS = 7


def _canonical_array_hash(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"output_complex_field")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _power(field: Any, spacing_m: float) -> float:
    import numpy as np

    return float(np.sum(np.abs(field) ** 2, dtype=np.float64) * spacing_m**2)


def _gaussian_metrics(field: Any, spacing_m: float, request: Any) -> dict[str, Any]:
    import numpy as np

    intensity = np.abs(field) ** 2
    total = float(np.sum(intensity, dtype=np.float64))
    coordinate = (np.arange(field.shape[0], dtype=np.float64) - field.shape[0] // 2) * spacing_m
    marginal_y = np.sum(intensity, axis=1, dtype=np.float64)
    marginal_x = np.sum(intensity, axis=0, dtype=np.float64)
    centroid_y_m = float(np.sum(coordinate * marginal_y) / total)
    centroid_x_m = float(np.sum(coordinate * marginal_x) / total)
    radius_y_m = float(
        2.0 * np.sqrt(np.sum((coordinate - centroid_y_m) ** 2 * marginal_y) / total)
    )
    radius_x_m = float(
        2.0 * np.sqrt(np.sum((coordinate - centroid_x_m) ** 2 * marginal_x) / total)
    )
    expected_radius_m = request.waist_m * np.sqrt(1.0 + request.z_over_rayleigh_range**2)
    radius_relative_error = max(
        abs(radius_x_m - expected_radius_m), abs(radius_y_m - expected_radius_m)
    ) / expected_radius_m
    centroid_input_pixels = max(abs(centroid_x_m), abs(centroid_y_m)) / request.spacing_m
    power_in = _power(request_input_field(request), request.spacing_m)
    power_out = _power(field, spacing_m)
    power_relative_error = abs(power_out / power_in - 1.0)
    return {
        "beam_radius_definition": (
            "per-axis D4-sigma radius w=2*sqrt(<(axis-centroid)^2>_intensity); "
            "exactly equals w for intensity exp(-2 r^2/w^2)"
        ),
        "expected_radius_m": float(expected_radius_m),
        "measured_radius_x_m": radius_x_m,
        "measured_radius_y_m": radius_y_m,
        "radius_relative_error": float(radius_relative_error),
        "centroid_x_m": centroid_x_m,
        "centroid_y_m": centroid_y_m,
        "centroid_input_pixels": float(centroid_input_pixels),
        "power_in": power_in,
        "power_out": power_out,
        "power_conservation_relative_error": float(power_relative_error),
    }


def request_input_field(request: Any) -> Any:
    import numpy as np

    coordinate = (
        np.arange(request.n_grid, dtype=np.float64) - request.n_grid // 2
    ) * request.spacing_m
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / request.waist_m**2).astype(np.complex64)


def _worker(args: argparse.Namespace) -> int:
    import_start_ns = time.perf_counter_ns()
    import importlib.metadata

    import numpy as np

    from multiscale_optics_agent.adapters.chromatix_scaling_adapter import (
        ChromatixScalingRequest,
        import_chromatix_scaling_dependencies,
        prepare_chromatix_scaling_session,
    )

    dependencies = import_chromatix_scaling_dependencies()
    import_seconds = (time.perf_counter_ns() - import_start_ns) * 1e-9
    request = ChromatixScalingRequest(n_grid=args.grid_case)
    setup_start_ns = time.perf_counter_ns()
    session = prepare_chromatix_scaling_session(dependencies, request)
    setup_seconds = (time.perf_counter_ns() - setup_start_ns) * 1e-9

    first_start_ns = time.perf_counter_ns()
    first_output = session.propagate()
    compile_plus_execute_seconds = (time.perf_counter_ns() - first_start_ns) * 1e-9
    seconds_through_first_call = (time.perf_counter_ns() - SCRIPT_START_NS) * 1e-9

    steady_samples: list[float] = []
    hashes = [_canonical_array_hash(first_output.field)]
    summaries = [
        _gaussian_metrics(first_output.field, first_output.output_spacing_m[0], request)
    ]
    final_output = first_output
    for _ in range(args.repeats):
        start_ns = time.perf_counter_ns()
        output = session.propagate()
        steady_samples.append((time.perf_counter_ns() - start_ns) * 1e-9)
        hashes.append(_canonical_array_hash(output.field))
        summaries.append(
            _gaussian_metrics(output.field, output.output_spacing_m[0], request)
        )
        final_output = output
    tolerance = {
        "radius": 0.01,
        "centroid_pixels": 0.1,
        "power": 0.001,
    }
    gaussian_pass = all(
        summary["radius_relative_error"] <= tolerance["radius"]
        and summary["centroid_input_pixels"] <= tolerance["centroid_pixels"]
        and summary["power_conservation_relative_error"] <= tolerance["power"]
        for summary in summaries
    )
    determinism_pass = len(set(hashes)) == 1 and all(
        summary == summaries[0] for summary in summaries
    )

    artifact = Path(args.worker_output).resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, final_output.field, allow_pickle=False)
    cpu_devices = [str(device) for device in dependencies.jax.devices("cpu")]
    result = {
        "grid": {
            "input_shape": list(session.input_shape),
            "input_n": request.n_grid,
            "physical_window_m": request.physical_window_m,
            "input_spacing_m": [request.spacing_m, request.spacing_m],
            "output_shape": list(session.expected_output_shape),
            "output_spacing_m": list(final_output.output_spacing_m),
            "axis_order": "(y, x)",
            "origin": "array index n//2 on each axis is coordinate zero",
        },
        "padding": {
            "policy": "auto_transfer",
            "implementation": "chromatix.functional.propagation.compute_padding_transfer",
            "pad_width": session.pad_width,
            "padded_shape": list(session.expected_output_shape),
            "propagated_cells": int(
                session.expected_output_shape[0] * session.expected_output_shape[1]
            ),
            "crop": "none; output_mode='full'",
        },
        "physics": {
            "field": "scalar monochromatic Gaussian complex amplitude",
            "waist_m": request.waist_m,
            "wavelength_m": request.wavelength_m,
            "refractive_index": request.refractive_index,
            "rayleigh_range_m": request.rayleigh_range_m,
            "distance_m": request.distance_m,
            "distance_over_rayleigh_range": request.z_over_rayleigh_range,
            "phasor": "exp(-i omega t); forward spatial factor exp(+i k z)",
            "reference_planes": "input waist at z=0; output at +0.1 z_R",
            "amplitude": "complex amplitude; intensity is abs(u)**2",
            "normalization": "discrete sampled power=sum(abs(u)**2)*dy*dx",
            "polarization": "scalar; no polarization basis",
            "coherence": "fully coherent monochromatic field",
        },
        "timing": {
            "clock": "time.perf_counter_ns with jax.block_until_ready(field_out.u)",
            "import_seconds": import_seconds,
            "setup_seconds": setup_seconds,
            "compile_plus_execute_seconds": compile_plus_execute_seconds,
            "seconds_through_first_call": seconds_through_first_call,
            "steady_measured_repeats": args.repeats,
            "steady_samples_seconds": steady_samples,
            "steady_median_seconds": float(np.median(steady_samples)),
            "steady_minimum_seconds": float(np.min(steady_samples)),
            "steady_p95_seconds": float(np.percentile(steady_samples, 95)),
            "cache_policy": (
                "fresh process and fresh shape for the first call; seven following calls "
                "reuse that process-local JAX compilation cache; first-call timing is never "
                "mixed into steady samples"
            ),
        },
        "throughput": {
            "propagated_output_cells_per_second": int(
                session.expected_output_shape[0] * session.expected_output_shape[1]
            )
            / float(np.median(steady_samples)),
            "denominator_note": (
                "actual full padded output cells for the same fixed Gaussian physics; "
                "input grid size is not used as propagated-cell count"
            ),
        },
        "gaussian_accuracy": {
            **summaries[-1],
            "tolerances": tolerance,
            "all_repeat_metrics": summaries,
            "pass": gaussian_pass,
        },
        "determinism": {
            "policy": "bitwise identical complex field and exact physical summary",
            "field_hashes_first_plus_steady": hashes,
            "field_hashes_identical": len(set(hashes)) == 1,
            "physical_summaries_identical": all(
                summary == summaries[0] for summary in summaries
            ),
            "pass": determinism_pass,
        },
        "peak_memory": {
            "status": "measured",
            "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; Linux KiB multiplied by 1024",
            "bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "runtime_environment": {
            "device": "cpu",
            "cpu_devices": cpu_devices,
            "jax_backend": dependencies.jax.default_backend(),
            "jax_enable_x64": bool(dependencies.jax.config.read("jax_enable_x64")),
            "dtype": "complex64",
            "chromatix_version": dependencies.chromatix_version,
            "chromatix_commit": dependencies.chromatix_commit,
            "jax_version": importlib.metadata.version("jax"),
            "numpy_version": np.__version__,
        },
        "scientific_field_sha256": hashes[-1],
        "field_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "worker_process_seconds": (time.perf_counter_ns() - SCRIPT_START_NS) * 1e-9,
        "forbidden_modules_loaded": sorted(
            name
            for name in sys.modules
            if name == "optiland"
            or name.startswith("optiland.")
            or name.startswith("multiscale_optics_agent.couplers")
        ),
    }
    Path(args.worker_result).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if gaussian_pass and determinism_pass and not result["forbidden_modules_loaded"] else 2


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _environment(cases: list[dict[str, Any]]) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    runtime = cases[0]["runtime_environment"]
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count() or 1,
        "process_cpu_affinity": affinity,
        "thread_counts": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", "unset"),
        },
        "device": "cpu",
        "cpu_devices": runtime["cpu_devices"],
        "dtype": "complex64",
        "jax_backend": runtime["jax_backend"],
        "jax_enable_x64": runtime["jax_enable_x64"],
        "engine_versions": {
            "chromatix": runtime["chromatix_version"],
            "chromatix_commit": runtime["chromatix_commit"],
            "jax": runtime["jax_version"],
            "numpy": runtime["numpy_version"],
        },
    }


def _fingerprint(environment: dict[str, Any]) -> str:
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plot_scaling(path: Path, cases: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [case["padding"]["propagated_cells"] for case in cases]
    first = [case["timing"]["compile_plus_execute_seconds"] for case in cases]
    median = [case["timing"]["steady_median_seconds"] for case in cases]
    p95 = [case["timing"]["steady_p95_seconds"] for case in cases]
    memory = [case["peak_memory"]["bytes"] / 2**20 for case in cases]
    pads = [case["padding"]["pad_width"] for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(cells, first, "o-", label="first compile + execute")
    axes[0].plot(cells, median, "s-", label="steady median")
    axes[0].plot(cells, p95, "^--", label="steady p95")
    axes[0].set(
        xlabel="actual padded propagated cells",
        ylabel="synchronized time [s]",
        title="Chromatix CPU timing",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(cells, memory, "o-", color="tab:green")
    axes[1].set(
        xlabel="actual padded propagated cells",
        ylabel="peak process RSS [MiB]",
        title="Fresh-process peak memory",
    )
    axes[1].grid(True, alpha=0.3)
    for axis in axes:
        for x_value, y_value, pad in zip(
            cells,
            first if axis is axes[0] else memory,
            pads,
            strict=True,
        ):
            axis.annotate(
                f"pad={pad}",
                (x_value, y_value),
                xytext=(4, 5),
                textcoords="offset points",
            )
    fig.suptitle("CHE-15 fixed-Gaussian automatic-padding scaling (raw, unsmoothed)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _current_accuracy_gate(output_dir: Path) -> dict[str, Any]:
    gate_dir = output_dir / "accuracy_gate"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate.py"),
        "--case",
        "all",
        "--output-dir",
        str(gate_dir),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if not (gate_dir / "result.json").is_file():
        raise RuntimeError(
            "current CHE-18 accuracy suite did not emit result.json: "
            + completed.stdout
            + completed.stderr
        )
    result = json.loads((gate_dir / "result.json").read_text())
    return {
        "command": command,
        "returncode": completed.returncode,
        "source_result": "accuracy_gate/result.json",
        "source_result_sha256": _sha256(gate_dir / "result.json"),
        "gated_cases": result.get("accuracy", {}).get("gated_cases", []),
        "blocked_cases": result.get("accuracy", {}).get("blocked_cases", []),
        "pass": completed.returncode == 0 and bool(result.get("accuracy", {}).get("pass")),
    }


def _write_blocked(output_dir: Path, exc: Exception) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "blocked",
        "failure": {
            "code": "L1_WAVE_01_SCALING_FAILED",
            "message": str(exc),
            "stage": "scaling_benchmark_execution",
            "exception_type": type(exc).__name__,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


def _coordinator(args: argparse.Namespace) -> int:
    import yaml

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = output_dir / "complex_fields"
    fields_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    raw_timing: dict[str, Any] = {"clock": "time.perf_counter_ns", "cases": []}
    for grid in args.grids:
        worker_result = output_dir / f"worker_{grid}.json"
        field_artifact = fields_dir / f"gaussian_grid_{grid}.npy"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--grid-case",
            str(grid),
            "--repeats",
            str(args.repeats),
            "--worker-result",
            str(worker_result),
            "--worker-output",
            str(field_artifact),
        ]
        start_ns = time.perf_counter_ns()
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        process_wall_seconds = (time.perf_counter_ns() - start_ns) * 1e-9
        if completed.returncode != 0 or not worker_result.is_file():
            raise RuntimeError(
                f"grid {grid} worker failed with exit {completed.returncode}: "
                + completed.stdout
                + completed.stderr
            )
        case = json.loads(worker_result.read_text())
        worker_result.unlink()
        case["timing"]["process_wall_seconds"] = process_wall_seconds
        case["field_artifact"] = str(field_artifact.relative_to(output_dir))
        cases.append(case)
        raw_timing["cases"].append(
            {
                "grid": grid,
                "process_wall_seconds": process_wall_seconds,
                **case["timing"],
            }
        )

    environment = _environment(cases)
    environment_fingerprint = _fingerprint(environment)
    anomalies: list[dict[str, Any]] = []
    for previous, current in pairwise(cases):
        for metric, previous_value, current_value in (
            (
                "steady_median_seconds",
                previous["timing"]["steady_median_seconds"],
                current["timing"]["steady_median_seconds"],
            ),
            (
                "peak_memory_bytes",
                previous["peak_memory"]["bytes"],
                current["peak_memory"]["bytes"],
            ),
        ):
            if current_value < previous_value:
                anomalies.append(
                    {
                        "metric": metric,
                        "previous_padded_cells": previous["padding"]["propagated_cells"],
                        "current_padded_cells": current["padding"]["propagated_cells"],
                        "previous_value": previous_value,
                        "current_value": current_value,
                        "note": "observed non-monotonic point retained; no smoothing applied",
                    }
                )

    tolerances = yaml.safe_load((SCRIPT_DIR / "scaling_tolerances.yaml").read_text())
    current_accuracy = _current_accuracy_gate(output_dir)
    gaussian_pass = all(case["gaussian_accuracy"]["pass"] for case in cases)
    determinism_pass = all(case["determinism"]["pass"] for case in cases)
    smallest_pass = (
        cases[0]["timing"]["seconds_through_first_call"]
        <= float(tolerances["smallest_case_compile_inclusive_seconds_max"])
    )
    scaling_pass = bool(
        current_accuracy["pass"] and gaussian_pass and determinism_pass and smallest_pass
    )
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "accuracy": {
            "oracle": (
                "CHE-15 fixed-Gaussian D4-sigma radius and discrete power gates, plus the "
                "unchanged current CHE-18 exact/paraxial accuracy bundle"
            ),
            "metrics": {
                "fixed_gaussian_cases_pass": gaussian_pass,
                "current_che18_gate": current_accuracy,
            },
            "tolerances": {
                "source": "tolerances.yaml",
                "gaussian_radius_relative": 0.01,
                "centroid_input_pixels": 0.1,
                "power_conservation_relative": 0.001,
            },
            "pass": current_accuracy["pass"] and gaussian_pass,
        },
        "performance": {
            "warmup_runs": 1,
            "measured_repeats": args.repeats,
            "samples_seconds": cases[0]["timing"]["steady_samples_seconds"],
            "statistics": {
                "median_seconds": cases[0]["timing"]["steady_median_seconds"],
                "minimum_seconds": cases[0]["timing"]["steady_minimum_seconds"],
                "p95_seconds": cases[0]["timing"]["steady_p95_seconds"],
            },
            "peak_memory": cases[0]["peak_memory"],
        },
        "scaling": {
            "grid_sequence": list(args.grids),
            "fixed_physical_case": cases[0]["physics"],
            "window_policy": (
                "physical window fixed at 64 um; spacing=window/n; only grid sampling changes"
            ),
            "cases": cases,
            "environment_fingerprint": environment_fingerprint,
            "regression_envelope": {
                "metric": "synchronized compiled steady-state median seconds",
                "relative_increase_max": float(
                    tolerances[
                        "matching_environment_steady_median_relative_regression_max"
                    ]
                ),
                "matching_environment_fingerprint_required": True,
                "fingerprint": environment_fingerprint,
                "baseline_median_seconds_by_actual_padded_cells": {
                    str(case["padding"]["propagated_cells"]): case["timing"][
                        "steady_median_seconds"
                    ]
                    for case in cases
                },
                "status": "recorded_pending_human_review",
            },
            "compile_cache_policy": (
                "each grid starts in a fresh process; first compile+execute is measured once; "
                "then exactly seven process-local cached calls form the steady samples"
            ),
            "smallest_case_compile_inclusive_under_10_seconds": smallest_pass,
            "gaussian_accuracy_and_power_pass": gaussian_pass,
            "determinism_pass": determinism_pass,
            "anomalies": anomalies,
            "plot_policy": "raw observed points by actual padded cells; no fit or smoothing",
            "pass": scaling_pass,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_timing_samples.json").write_text(
        json.dumps(raw_timing, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(SCRIPT_DIR / "scaling_tolerances.yaml", output_dir / "tolerances.yaml")
    _plot_scaling(output_dir / "scaling.png", cases)
    (output_dir / "README.md").write_text(
        "# CHE-15 Chromatix scaling evidence\n\n"
        "A scalar 532 nm Gaussian with 10 um waist is propagated from its waist plane "
        "to +0.1 Rayleigh ranges. The 64 um physical input window is fixed while grids "
        "64/128/256 refine the spacing. Chromatix automatic padding is measured, not "
        "overridden. Every grid runs in a fresh process: its first synchronized call is "
        "compile-plus-execute, followed by seven synchronized cache-reusing calls. Runtime "
        "and memory use actual padded output cells. The current CHE-18 accuracy bundle is "
        "embedded as an additional gate because its in-review redesign no longer contains "
        "the Gaussian case still required by CHE-15.\n"
    )

    git_commit, dirty = _git_state()
    artifact_files = [
        "result.json",
        "raw_timing_samples.json",
        "tolerances.yaml",
        "scaling.png",
        "README.md",
        *[case["field_artifact"] for case in cases],
    ]
    provenance = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "command": [
            "./run.sh",
            "python",
            "benchmarks/level1/L1-WAVE-01/run_scaling.py",
            "--device",
            "cpu",
            "--output-dir",
            str(args.output_dir),
        ],
        "git_commit": git_commit,
        "dirty_worktree": dirty,
        **environment,
        "seed": 20260811,
        "input_parameters": {
            "grids": list(args.grids),
            "steady_repeats": args.repeats,
            "fixed_physical_case": cases[0]["physics"],
            "physical_window_m": cases[0]["grid"]["physical_window_m"],
            "padding_policy": "auto_transfer",
        },
        "environment_fingerprint": environment_fingerprint,
        "artifact_hashes": {
            filename: _sha256(output_dir / filename) for filename in artifact_files
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "forbidden_modules_loaded": sorted(
            name
            for name in sys.modules
            if name == "optiland"
            or name.startswith("optiland.")
            or name.startswith("multiscale_optics_agent.couplers")
        ),
    }
    if provenance["forbidden_modules_loaded"]:
        raise RuntimeError(
            f"Wave-only scaling loaded forbidden modules: {provenance['forbidden_modules_loaded']}"
        )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pass": scaling_pass,
                "gaussian_accuracy_and_power_pass": gaussian_pass,
                "current_che18_gate_pass": current_accuracy["pass"],
                "determinism_pass": determinism_pass,
                "smallest_compile_inclusive_under_10_seconds": smallest_pass,
                "output_directory": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if scaling_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/L1-WAVE-01/scaling"))
    parser.add_argument("--grids", nargs="+", type=int, default=list(DEFAULT_GRIDS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grid-case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.grid_case is None or args.worker_result is None or args.worker_output is None:
            parser.error("worker mode requires --grid-case, --worker-result, and --worker-output")
        return _worker(args)
    if args.repeats < 5:
        parser.error("CHE-15 requires at least five compiled steady-state repeats")
    if args.grids != sorted(set(args.grids)) or any(value <= 0 for value in args.grids):
        parser.error("grid sizes must be unique positive integers in increasing order")
    try:
        return _coordinator(args)
    except Exception as exc:
        return _write_blocked(args.output_dir.resolve(), exc)


if __name__ == "__main__":
    raise SystemExit(main())
