"""Run the CHE-16 Optiland sampling, determinism, and CPU scaling benchmark."""

from __future__ import annotations

# Keep worker startup imports in the standard library so the worker can measure
# solver/library import time separately from prescription setup and trace time.
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
BENCHMARK_ID = "L1-RAY-01"
PROTOCOL_ID = "M1-BASELINE-CPU-V2"
DEFAULT_SAMPLING = (8, 16, 32, 64)
DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 7


def _canonical_array_hash(arrays: dict[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _physical_summary(trace: Any) -> dict[str, Any]:
    import numpy as np

    arrays = trace.arrays
    direction_norm = np.sqrt(arrays["L"] ** 2 + arrays["M"] ** 2 + arrays["N"] ** 2)
    return {
        "generated_ray_count": trace.generated_ray_count,
        "traced_ray_count": trace.traced_ray_count,
        "surviving_ray_count": trace.surviving_ray_count,
        "invalid_ray_count": trace.invalid_ray_count,
        "vignetted_ray_count": trace.vignetted_ray_count,
        "max_direction_norm_error": float(np.max(np.abs(direction_norm - 1.0))),
        "intensity_sum": float(np.sum(arrays["intensity"])),
        "x_m_min": float(np.min(arrays["x_m"])),
        "x_m_max": float(np.max(arrays["x_m"])),
        "y_m_min": float(np.min(arrays["y_m"])),
        "y_m_max": float(np.max(arrays["y_m"])),
        "z_m_min": float(np.min(arrays["z_m"])),
        "z_m_max": float(np.max(arrays["z_m"])),
    }


def _worker(args: argparse.Namespace) -> int:
    import_start_ns = time.perf_counter_ns()
    import importlib.metadata

    import numpy as np

    from multiscale_optics_agent.adapters.optiland_benchmark_adapter import (
        OptilandScalingRequest,
        import_optiland_scaling_dependencies,
        prepare_optiland_scaling_session,
    )

    dependencies = import_optiland_scaling_dependencies()
    import_seconds = (time.perf_counter_ns() - import_start_ns) * 1e-9

    request = OptilandScalingRequest(requested_sampling=args.sampling)
    setup_start_ns = time.perf_counter_ns()
    session = prepare_optiland_scaling_session(dependencies, request)
    setup_seconds = (time.perf_counter_ns() - setup_start_ns) * 1e-9

    for _ in range(args.warmup_runs):
        session.trace()

    samples_seconds: list[float] = []
    hashes: list[str] = []
    summaries: list[dict[str, Any]] = []
    final_trace = None
    for _ in range(args.repeats):
        start_ns = time.perf_counter_ns()
        final_trace = session.trace()
        samples_seconds.append((time.perf_counter_ns() - start_ns) * 1e-9)
        hashes.append(_canonical_array_hash(final_trace.arrays))
        summaries.append(_physical_summary(final_trace))
    assert final_trace is not None

    output_path = Path(args.worker_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **final_trace.arrays)
    summary = summaries[-1]
    worker_result = {
        "requested_sampling": args.sampling,
        "prescription": "Optiland samples.objectives.ReverseTelephoto",
        "field": {"Hx": request.field_hx, "Hy": request.field_hy},
        "wavelength_um": request.wavelength_um,
        "surface_count": session.surface_count,
        "backend": request.backend,
        "device": request.device,
        "dtype": request.dtype,
        "jax_x64": "not_applicable",
        "counts": {
            "requested_sampling": args.sampling,
            "generated": final_trace.generated_ray_count,
            "traced": final_trace.traced_ray_count,
            "surviving": final_trace.surviving_ray_count,
            "invalid": final_trace.invalid_ray_count,
            "vignetted": final_trace.vignetted_ray_count,
            "pre_generation_rejected": {
                "status": "unsupported",
                "diagnostic": (
                    "Optiland 0.6.0 exposes generated RealRays after pupil distribution "
                    "creation, not a pre-generation rejection count."
                ),
            },
        },
        "timing": {
            "import_seconds": import_seconds,
            "setup_seconds": setup_seconds,
            "warmup_runs": args.warmup_runs,
            "measured_repeats": args.repeats,
            "trace_samples_seconds": samples_seconds,
            "trace_median_seconds": float(np.median(samples_seconds)),
            "trace_minimum_seconds": float(np.min(samples_seconds)),
            "trace_p95_seconds": float(np.percentile(samples_seconds, 95)),
        },
        "peak_memory": {
            "status": "measured",
            "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; Linux KiB multiplied by 1024",
            "bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "throughput": {
            "traced_rays_per_second": (
                final_trace.traced_ray_count / float(np.median(samples_seconds))
            ),
            "surviving_rays_per_second": (
                final_trace.surviving_ray_count / float(np.median(samples_seconds))
            ),
            "denominator_note": "actual traced/surviving ray counts; never requested sampling",
        },
        "determinism": {
            "policy": "bitwise identical canonical scientific arrays and exact summary metrics",
            "repeat_scientific_array_sha256": hashes,
            "repeat_physical_summaries": summaries,
            "array_hashes_identical": len(set(hashes)) == 1,
            "physical_summaries_identical": all(value == summaries[0] for value in summaries),
            "pass": len(set(hashes)) == 1
            and all(value == summaries[0] for value in summaries),
        },
        "physical_summary": summary,
        "scientific_array_sha256": hashes[-1],
        "artifact_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "engine_versions": {
            "optiland": importlib.metadata.version("optiland"),
            "numpy": np.__version__,
        },
        "worker_process_seconds": (time.perf_counter_ns() - SCRIPT_START_NS) * 1e-9,
    }
    Path(args.worker_result).write_text(json.dumps(worker_result, indent=2, sort_keys=True) + "\n")
    return 0


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


def _environment() -> dict[str, Any]:
    import importlib.metadata

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
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
        },
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "engine_versions": {
            "optiland": importlib.metadata.version("optiland"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }


def _environment_fingerprint(environment: dict[str, Any]) -> str:
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plot_scaling(path: Path, cases: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = [case["counts"]["traced"] for case in cases]
    medians = [case["timing"]["trace_median_seconds"] for case in cases]
    p95 = [case["timing"]["trace_p95_seconds"] for case in cases]
    memory_mib = [case["peak_memory"]["bytes"] / 2**20 for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(counts, medians, "o-", label="median trace")
    axes[0].plot(counts, p95, "s--", label="p95 trace")
    axes[0].set(
        xlabel="actual traced ray count",
        ylabel="trace time [s]",
        title="CPU trace scaling",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(counts, memory_mib, "o-", color="tab:green")
    axes[1].set(
        xlabel="actual traced ray count",
        ylabel="peak process RSS [MiB]",
        title="Peak memory (fresh process per case)",
    )
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("CHE-16 Optiland ReverseTelephoto sampling benchmark (no smoothing)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _accuracy_gate(output_dir: Path) -> dict[str, Any]:
    accuracy_dir = output_dir / "accuracy_gate"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_benchmark.py"),
        "--output-dir",
        str(accuracy_dir),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if not (accuracy_dir / "result.json").exists():
        raise RuntimeError(
            "CHE-17 accuracy gate did not emit result.json: "
            + completed.stdout
            + completed.stderr
        )
    result = json.loads((accuracy_dir / "result.json").read_text())
    return {
        "command": command,
        "returncode": completed.returncode,
        "source_result": "accuracy_gate/result.json",
        "source_result_sha256": _sha256(accuracy_dir / "result.json"),
        "case_pass": {
            name: value["pass"]
            for name, value in result.get("accuracy", {}).get("metrics", {}).items()
            if isinstance(value, dict) and "pass" in value
        },
        "pass": completed.returncode == 0 and bool(result.get("accuracy", {}).get("pass")),
    }


def _write_blocked(output_dir: Path, exc: Exception) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "blocked",
        "failure": {
            "code": "L1_RAY_01_SCALING_FAILED",
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
    arrays_dir = output_dir / "scientific_artifacts"
    arrays_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    raw_timing: dict[str, Any] = {"clock": "time.perf_counter_ns", "cases": []}
    for sampling in args.sampling:
        worker_result_path = output_dir / f"worker_{sampling}.json"
        artifact_path = arrays_dir / f"sampling_{sampling}.npz"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--sampling-case",
            str(sampling),
            "--warmup-runs",
            str(args.warmup_runs),
            "--repeats",
            str(args.repeats),
            "--worker-result",
            str(worker_result_path),
            "--worker-output",
            str(artifact_path),
        ]
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        process_wall_seconds = (time.perf_counter_ns() - started) * 1e-9
        if completed.returncode != 0 or not worker_result_path.exists():
            raise RuntimeError(
                f"sampling {sampling} worker failed with exit {completed.returncode}: "
                + completed.stdout
                + completed.stderr
            )
        case = json.loads(worker_result_path.read_text())
        worker_result_path.unlink()
        case["timing"]["process_wall_seconds"] = process_wall_seconds
        case["scientific_artifact"] = str(artifact_path.relative_to(output_dir))
        cases.append(case)
        raw_timing["cases"].append(
            {
                "requested_sampling": sampling,
                "process_wall_seconds": process_wall_seconds,
                **case["timing"],
            }
        )

    environment = _environment()
    fingerprint = _environment_fingerprint(environment)
    anomalies: list[dict[str, Any]] = []
    for previous, current in pairwise(cases):
        for metric_path, previous_value, current_value in (
            (
                "trace_median_seconds",
                previous["timing"]["trace_median_seconds"],
                current["timing"]["trace_median_seconds"],
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
                        "metric": metric_path,
                        "previous_actual_traced_rays": previous["counts"]["traced"],
                        "current_actual_traced_rays": current["counts"]["traced"],
                        "previous_value": previous_value,
                        "current_value": current_value,
                        "note": "observed non-monotonic sample retained; no smoothing applied",
                    }
                )

    tolerance_data = yaml.safe_load((SCRIPT_DIR / "scaling_tolerances.yaml").read_text())
    accuracy_gate = _accuracy_gate(output_dir)
    smallest_pass = (
        cases[0]["timing"]["process_wall_seconds"]
        <= float(tolerance_data["smallest_case_process_wall_seconds_max"])
    )
    determinism_pass = all(case["determinism"]["pass"] for case in cases)
    scaling_pass = bool(accuracy_gate["pass"] and smallest_pass and determinism_pass)
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "accuracy": {
            "oracle": "unchanged CHE-17 Optiland analytic ray benchmark bundle",
            "metrics": accuracy_gate,
            "tolerances": {"source": "accuracy_gate/tolerances.yaml"},
            "pass": accuracy_gate["pass"],
        },
        "performance": {
            "warmup_runs": args.warmup_runs,
            "measured_repeats": args.repeats,
            "samples_seconds": cases[0]["timing"]["trace_samples_seconds"],
            "statistics": {
                "median_seconds": cases[0]["timing"]["trace_median_seconds"],
                "minimum_seconds": cases[0]["timing"]["trace_minimum_seconds"],
                "p95_seconds": cases[0]["timing"]["trace_p95_seconds"],
            },
            "peak_memory": cases[0]["peak_memory"],
        },
        "scaling": {
            "prescription": "Optiland samples.objectives.ReverseTelephoto",
            "fixed_physical_case": {
                "field": {"Hx": 0.0, "Hy": 0.0},
                "wavelength_m": 0.55e-6,
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
            },
            "requested_sampling_sequence": list(args.sampling),
            "cases": cases,
            "environment_fingerprint": fingerprint,
            "regression_envelope": {
                "metric": "steady trace median seconds",
                "relative_increase_max": float(
                    tolerance_data[
                        "matching_environment_median_runtime_relative_regression_max"
                    ]
                ),
                "matching_environment_fingerprint_required": True,
                "fingerprint": fingerprint,
                "baseline_median_seconds_by_actual_traced_ray_count": {
                    str(case["counts"]["traced"]): case["timing"]["trace_median_seconds"]
                    for case in cases
                },
                "status": "recorded_pending_human_review",
            },
            "smallest_case_under_10_seconds": smallest_pass,
            "determinism_pass": determinism_pass,
            "anomalies": anomalies,
            "plot_policy": "raw observed points connected for readability; no fit or smoothing",
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
        "# CHE-16 Optiland scaling evidence\n\n"
        "The fixed ReverseTelephoto prescription is traced at requested sampling "
        f"{list(args.sampling)} on NumPy/CPU/float64. Counts and throughput use actual "
        "generated/traced/valid-surviving rays. Every sampling case runs in a fresh "
        "process so import, prescription setup, trace timing, and peak RSS remain "
        "separable. The embedded `accuracy_gate/` is the unchanged CHE-17 bundle; "
        "accuracy failure overrides performance success. `scaling.png` displays raw "
        "observations against actual traced counts without smoothing.\n"
    )

    git_commit, dirty = _git_state()
    artifact_files = [
        "result.json",
        "raw_timing_samples.json",
        "tolerances.yaml",
        "scaling.png",
        "README.md",
        *[case["scientific_artifact"] for case in cases],
    ]
    provenance = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "command": [
            "./run.sh",
            "python",
            "benchmarks/level1/L1-RAY-01/run_scaling.py",
            "--backend",
            "numpy",
            "--output-dir",
            str(args.output_dir),
        ],
        "git_commit": git_commit,
        "dirty_worktree": dirty,
        **environment,
        "seed": 20260811,
        "input_parameters": {
            "sampling": list(args.sampling),
            "warmup_runs": args.warmup_runs,
            "measured_repeats": args.repeats,
            "fixed_physical_case": result["scaling"]["fixed_physical_case"],
        },
        "environment_fingerprint": fingerprint,
        "artifact_hashes": {
            filename: _sha256(output_dir / filename) for filename in artifact_files
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "forbidden_modules_loaded": sorted(
            name
            for name in sys.modules
            if name == "chromatix"
            or name.startswith("chromatix.")
            or name.startswith("multiscale_optics_agent.couplers")
        ),
    }
    if provenance["forbidden_modules_loaded"]:
        raise RuntimeError(
            "Ray-only scaling benchmark loaded forbidden modules: "
            f"{provenance['forbidden_modules_loaded']}"
        )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pass": scaling_pass,
                "accuracy_pass": accuracy_gate["pass"],
                "determinism_pass": determinism_pass,
                "smallest_case_under_10_seconds": smallest_pass,
                "output_directory": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if scaling_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy",), default="numpy")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/L1-RAY-01/scaling"))
    parser.add_argument("--sampling", nargs="+", type=int, default=list(DEFAULT_SAMPLING))
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sampling-case", dest="sampling", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if (
            args.worker_result is None
            or args.worker_output is None
            or not isinstance(args.sampling, int)
        ):
            parser.error(
                "worker mode requires --sampling-case, --worker-result, and --worker-output"
            )
        return _worker(args)
    if args.warmup_runs < 1 or args.repeats < 5:
        parser.error("CHE-16 requires at least one warmup and five measured repeats")
    if args.sampling != sorted(set(args.sampling)) or any(value <= 0 for value in args.sampling):
        parser.error("sampling values must be unique positive integers in increasing order")
    try:
        return _coordinator(args)
    except Exception as exc:
        return _write_blocked(args.output_dir.resolve(), exc)


if __name__ == "__main__":
    raise SystemExit(main())
