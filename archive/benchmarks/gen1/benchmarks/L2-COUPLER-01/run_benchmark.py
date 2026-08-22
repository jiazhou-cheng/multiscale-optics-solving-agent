#!/usr/bin/env python3
"""L2-COUPLER-01 — bidirectional ray-wave coupler benchmark (CHE-29).

Runs under ``M2-COUPLER-CPU-V1`` and emits the full bundle the protocol
requires. Accuracy gates are evaluated and must pass BEFORE any performance
number is accepted, exactly as in M1: a fast wrong answer is not a result.

Sections:
  accuracy         deterministic gates -- exactness limit, round trips, the
                   curvature bound, and the negative controls
  stochastic       the Monte Carlo evidence -- unbiasedness against the measured
                   standard error, and a fitted convergence exponent
  differentiability what the gradient estimator was measured to do, with the
                   regime attached. Never a promotion.
  performance      steady-state timing, recorded only after accuracy passes

Run:
    ./run.sh python benchmarks/level2/L2-COUPLER-01/run_benchmark.py \\
        --output-dir outputs/M2/coupler
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from multiscale_optics_agent.couplers.cascade import planar_doe_step  # noqa: E402
from multiscale_optics_agent.couplers.contracts import (  # noqa: E402
    ComplexField,
    ReferencePlane,
)
from multiscale_optics_agent.couplers.curvature import (  # noqa: E402
    curvature_direction_error_bound,
    curvature_observability_width,
    measured_tangent_plane_direction_error,
)
from multiscale_optics_agent.couplers.gradient import GradientProblem, characterize  # noqa: E402
from multiscale_optics_agent.couplers.ray_to_wave import (  # noqa: E402
    Perturbation,
    Projection,
    collimated_bundle,
    ray_to_wave,
)
from multiscale_optics_agent.couplers.wave_to_ray import (  # noqa: E402
    SamplingDensity,
    SamplingPerturbation,
    decompose,
    draw_indices,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)
from multiscale_optics_agent.core.provenance import VOLATILE_KEYS, strip_volatile  # noqa: E402

BENCHMARK_ID = "L2-COUPLER-01"
PROTOCOL_ID = "M2-COUPLER-CPU-V1"
SEED = 20260812
WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N_GRID = 16
GRID = (N_GRID, N_GRID)
PITCH = (PITCH_M, PITCH_M)
PLANE = ReferencePlane(name="coupler plane", z_m=0.0)
FLOAT64_EPS = float(np.finfo(np.float64).eps)


# --- fixtures ------------------------------------------------------------------


def multilobed_field() -> ComplexField:
    rng = np.random.default_rng(SEED)
    return ComplexField(
        u=(rng.normal(size=GRID) + 1j * rng.normal(size=GRID)).astype(np.complex128),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )


def concentrated_field() -> ComplexField:
    coords = (np.arange(N_GRID) - N_GRID // 2) * PITCH_M
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return ComplexField(
        u=(np.exp(-(xx**2 + yy**2) / (4 * PITCH_M) ** 2) + 0j).astype(np.complex128),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )


def reconstruct(bundle) -> np.ndarray:
    field, _ = ray_to_wave(
        bundle, grid_shape=GRID, sample_pitch_m=PITCH, projection=Projection.ASM_CONSISTENT
    )
    return field.u


def relative_rms(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.abs(estimate - truth) ** 2)) / np.sqrt(np.mean(np.abs(truth) ** 2))
    )


# --- accuracy ------------------------------------------------------------------


def accuracy_section(tolerances: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    # 1. Exactness limit. Enumerating every propagating bin removes sampling
    #    error entirely, so any disagreement here is a transform defect.
    for label, field in (("multilobed", multilobed_field()), ("concentrated", concentrated_field())):
        spectrum = decompose(field)
        density = sampling_density(spectrum)
        bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)
        rebuilt = reconstruct(bundle)
        error = float(np.max(np.abs(rebuilt - field.u)))
        bound = (
            64.0 * FLOAT64_EPS * float(np.max(np.abs(field.u))) * math.sqrt(spectrum.propagating_count)
        )
        metrics[f"exactness_{label}_max_abs_error"] = error
        metrics[f"exactness_{label}_roundoff_bound"] = bound
        metrics[f"exactness_{label}_pass"] = bool(error <= bound)
        arrays[f"field_{label}"] = field.u
        arrays[f"reconstructed_{label}"] = rebuilt

    # 2. Exact analytic oracle for the ray->wave direction (SI Figure S1c).
    theta = 0.2
    direction = (math.sin(theta), 0.0, math.cos(theta))
    coords = (np.arange(8) - 4) * 4 * PITCH_M
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    collimated = collimated_bundle(
        positions_xy_m=np.column_stack([xx.ravel(), yy.ravel()]),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    reconstructed = reconstruct(collimated)
    y = (np.arange(N_GRID) - N_GRID // 2) * PITCH_M
    gy, gx = np.meshgrid(y, y, indexing="ij")
    wavenumber = 2.0 * math.pi / WAVELENGTH_M
    oracle = collimated.count * np.exp(1j * wavenumber * (direction[0] * gx + direction[1] * gy))
    plane_wave_error = float(np.max(np.abs(reconstructed - oracle)))
    metrics["plane_wave_oracle_max_abs_error"] = plane_wave_error
    metrics["plane_wave_oracle_pass"] = bool(
        plane_wave_error <= tolerances["plane_wave_oracle_max_abs_error"]
    )
    arrays["plane_wave_reconstructed"] = reconstructed
    arrays["plane_wave_oracle"] = oracle

    # 3. Round trips, and proof that a mismatched pairing is detected. Without
    #    the second, the first proves nothing: a shared convention error cancels.
    field = multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)
    consistent = relative_rms(reconstruct(bundle), field.u)
    flipped, _ = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        projection=Projection.ASM_CONSISTENT,
        perturbation=Perturbation(phase_sign=-1),
    )
    mismatched = relative_rms(flipped.u, field.u)
    metrics["round_trip_relative_rms"] = consistent
    metrics["round_trip_mismatched_phase_relative_rms"] = mismatched
    metrics["round_trip_pass"] = bool(
        consistent <= tolerances["round_trip_relative_rms"]
        and mismatched >= tolerances["mismatched_pairing_min_relative_rms"]
    )

    # 4. Cascade: the ray count after a planar DOE is the caller's budget.
    doe_rng = np.random.default_rng(SEED)
    doe = np.exp(1j * doe_rng.uniform(-math.pi, math.pi, size=GRID)).astype(np.complex128)
    launches = np.zeros((4, 2))
    step_rng = np.random.default_rng(SEED)
    first, transmitted, diag_a = planar_doe_step(
        bundle, doe, grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=launches, secondary_count=64, rng=step_rng,
    )
    second, _, diag_b = planar_doe_step(
        first, doe, grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=launches, secondary_count=64, rng=step_rng,
    )
    metrics["cascade_first_ray_count"] = first.count
    metrics["cascade_second_ray_count"] = second.count
    metrics["cascade_count_is_bounded"] = bool(second.count == first.count == 4 * 64)
    metrics["cascade_phase_doe_power_ratio"] = (
        diag_a.transmitted_discrete_power / diag_a.incident_discrete_power
    )
    metrics["cascade_pass"] = bool(
        metrics["cascade_count_is_bounded"]
        and abs(metrics["cascade_phase_doe_power_ratio"] - 1.0)
        <= tolerances["phase_doe_power_ratio_tolerance"]
    )

    # 5. Curvature bound must bound the measurement across the Figure 3c regime.
    curvature_rows = []
    worst_ratio = 0.0
    for radius_lambda in (1_000, 10_000, 100_000):
        for patch_lambda in (50, 100, 200, 400):
            radius_m = radius_lambda * 1e-6
            patch_m = patch_lambda * 1e-6
            measured = measured_tangent_plane_direction_error(
                patch_width_m=patch_m, radius_m=radius_m, wavelength_m=1e-6
            )
            bound = curvature_direction_error_bound(patch_m, radius_m)
            curvature_rows.append(
                {
                    "radius_lambda": radius_lambda,
                    "patch_lambda": patch_lambda,
                    "measured_rad": measured,
                    "bound_rad": bound,
                    "observable": patch_m > curvature_observability_width(1e-6, radius_m),
                    "holds": bool(measured <= bound),
                }
            )
            worst_ratio = max(worst_ratio, measured / bound if bound else 0.0)
    metrics["curvature_cases"] = curvature_rows
    metrics["curvature_worst_measured_over_bound"] = worst_ratio
    metrics["curvature_pass"] = bool(all(row["holds"] for row in curvature_rows))

    # 6. Negative controls. Each must be detected, each with a passing control.
    negatives: dict[str, bool] = {}
    negatives["phase_sign"] = mismatched >= tolerances["mismatched_pairing_min_relative_rms"]

    no_ramp, _ = ray_to_wave(
        collimated, grid_shape=GRID, sample_pitch_m=PITCH,
        perturbation=Perturbation(apply_oblique_ramp=False),
    )
    negatives["oblique_ramp"] = (
        float(np.max(np.abs(no_ramp.u - oracle))) / collimated.count > 1e-3
    )

    transposed, _ = ray_to_wave(
        collimated, grid_shape=GRID, sample_pitch_m=PITCH,
        perturbation=Perturbation(transpose_axes=True),
    )
    negatives["axis_transpose"] = (
        float(np.max(np.abs(transposed.u - oracle))) / collimated.count > 1e-3
    )

    # The missing 1/p weight is a BIAS, so it has to be detected in the ensemble
    # mean against the standard error -- a single-realization RMS comparison does
    # not see it. Run on the concentrated spectrum under p_mag, because under
    # uniform sampling the omitted weight is exactly a constant scale factor and
    # the test would pass for the wrong reason.
    concentrated = concentrated_field()
    concentrated_spectrum = decompose(concentrated)
    magnitude_density = sampling_density(concentrated_spectrum, SamplingDensity.MAGNITUDE)
    truth = float(np.real(np.vdot(concentrated.u, concentrated.u)))
    unweighted_samples = []
    for realization in range(32):
        drawn = draw_indices(magnitude_density, 2048, np.random.default_rng(4000 + realization))
        estimate = reconstruct(
            spectrum_to_rays(
                concentrated_spectrum, drawn, magnitude_density,
                perturbation=SamplingPerturbation(apply_importance_weight=False),
            )
        )
        unweighted_samples.append(float(np.real(np.vdot(concentrated.u, estimate))))
    unweighted_series = np.asarray(unweighted_samples)
    unweighted_bias = float(np.mean(unweighted_series) - truth)
    unweighted_se = float(np.std(unweighted_series, ddof=1) / math.sqrt(unweighted_series.size))
    metrics["importance_weight_omitted_bias"] = unweighted_bias
    metrics["importance_weight_omitted_standard_error"] = unweighted_se
    metrics["importance_weight_omitted_bias_in_standard_errors"] = (
        abs(unweighted_bias) / unweighted_se if unweighted_se > 0 else float("inf")
    )
    negatives["importance_weight"] = (
        abs(unweighted_bias) > tolerances["unbiasedness_k_sigma"] * unweighted_se
    )

    launch_offsets = np.array([[0.0, 0.0], [3e-6, -2e-6], [-4e-6, 1e-6]])
    with_phase = spectrum_to_rays(
        spectrum, enumerate_indices(density), density, launch_positions_xy_m=launch_offsets
    )
    without_phase = spectrum_to_rays(
        spectrum, enumerate_indices(density), density,
        launch_positions_xy_m=launch_offsets,
        perturbation=SamplingPerturbation(apply_launch_phase=False),
    )
    negatives["launch_phase"] = (
        float(np.max(np.abs(reconstruct(with_phase) - reconstruct(without_phase))))
        > 1e-3 * float(np.max(np.abs(field.u)))
    )

    metrics["negative_controls"] = {key: bool(value) for key, value in negatives.items()}
    metrics["negative_controls_detected"] = int(sum(negatives.values()))
    metrics["negative_controls_total"] = len(negatives)
    metrics["negative_controls_pass"] = bool(all(negatives.values()))

    gates = [key for key in metrics if key.endswith("_pass")]
    section = {
        "oracle": (
            "Exact angular-spectrum enumeration; analytic tilted plane wave "
            "(ACS Photonics 2026 SI Figure S1c); SI eq S9 curvature bound"
        ),
        "metrics": metrics,
        "tolerances": tolerances,
        "gates": gates,
        "pass": bool(all(metrics[key] for key in gates)),
    }
    return section, arrays


# --- stochastic ----------------------------------------------------------------


def stochastic_section(tolerances: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    field = multilobed_field()
    spectrum = decompose(field)
    realizations = 32

    # Exactness limit, restated here because the protocol requires it inside the
    # stochastic section: it is what makes every number below interpretable.
    density = sampling_density(spectrum)
    exact = reconstruct(spectrum_to_rays(spectrum, enumerate_indices(density), density))
    exact_error = float(np.max(np.abs(exact - field.u)))
    exact_bound = (
        64.0 * FLOAT64_EPS * float(np.max(np.abs(field.u))) * math.sqrt(spectrum.propagating_count)
    )

    # Unbiasedness on a scalar linear functional of the estimator.
    truth = complex(np.vdot(field.u, field.u))
    samples = []
    for realization in range(realizations):
        indices = draw_indices(density, 2048, np.random.default_rng(9000 + realization))
        estimate = reconstruct(spectrum_to_rays(spectrum, indices, density))
        samples.append(complex(np.vdot(field.u, estimate)))
    series = np.real(np.asarray(samples))
    mean_error = float(np.mean(series) - truth.real)
    standard_error = float(np.std(series, ddof=1) / math.sqrt(series.size))

    # Convergence, fitted over a sweep.
    counts = [256, 512, 1024, 2048, 4096, 8192]
    sweep = []
    for count in counts:
        errors = [
            relative_rms(
                reconstruct(
                    spectrum_to_rays(
                        spectrum, draw_indices(density, count, np.random.default_rng(500 + s)), density
                    )
                ),
                field.u,
            )
            for s in range(16)
        ]
        sweep.append({"count": count, "relative_rms": float(np.mean(errors))})
    exponent = float(
        np.polyfit(np.log([row["count"] for row in sweep]), np.log([row["relative_rms"] for row in sweep]), 1)[0]
    )

    # Variance by sampling density, at matched N, on both spectrum shapes.
    variance = {}
    for label, source in (("multilobed", field), ("concentrated", concentrated_field())):
        local_spectrum = decompose(source)
        variance[label] = {}
        for kind in (SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE):
            local_density = sampling_density(local_spectrum, kind)
            errors = [
                relative_rms(
                    reconstruct(
                        spectrum_to_rays(
                            local_spectrum,
                            draw_indices(local_density, 1024, np.random.default_rng(700 + s)),
                            local_density,
                        )
                    ),
                    source.u,
                )
                for s in range(16)
            ]
            variance[label][str(kind)] = float(np.mean(errors))
        variance[label]["magnitude_advantage"] = (
            variance[label]["p_uni"] / variance[label]["p_mag"]
        )

    section = {
        "seed": SEED,
        "rng_generator": "numpy.random.Generator(PCG64)",
        "realizations": realizations,
        "exactness_limit": {
            "enumerated_all_propagating_bins": True,
            "propagating_modes": spectrum.propagating_count,
            "max_abs_error": exact_error,
            "tolerance": exact_bound,
            "tolerance_basis": "dtype_roundoff_derived",
            "pass": bool(exact_error <= exact_bound),
        },
        "unbiasedness": {
            "functional": "Re<u, U_estimate>",
            "sampled_ray_count": 2048,
            "mean_error": mean_error,
            "standard_error": standard_error,
            "k_sigma": tolerances["unbiasedness_k_sigma"],
            "pass": bool(abs(mean_error) <= tolerances["unbiasedness_k_sigma"] * standard_error),
        },
        "convergence": {
            "sweep": sweep,
            "seeds_per_point": 16,
            "fitted_exponent": exponent,
            "expected_exponent": -0.5,
            "exponent_tolerance": tolerances["convergence_exponent_tolerance"],
            "pass": bool(abs(exponent + 0.5) <= tolerances["convergence_exponent_tolerance"]),
        },
        "variance_by_density": variance,
        "evanescent_power_fraction": spectrum.evanescent_power_fraction,
    }
    section["pass"] = bool(
        section["exactness_limit"]["pass"]
        and section["unbiasedness"]["pass"]
        and section["convergence"]["pass"]
    )
    convergence_artifact = {"sweep": sweep, "fitted_exponent": exponent}
    ensemble_artifact = {
        "realizations": realizations,
        "functional_samples": [float(v) for v in series],
        "mean": float(np.mean(series)),
        "standard_error": standard_error,
        "reference": truth.real,
    }
    return section, {"convergence": convergence_artifact, "ensemble": ensemble_artifact}


# --- differentiability ------------------------------------------------------------


def differentiability_section() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    incident = ComplexField(
        u=(rng.normal(size=GRID) + 1j * rng.normal(size=GRID)).astype(np.complex128),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )
    mask = rng.uniform(-1.0, 1.0, size=GRID)
    window = np.zeros(GRID)
    window[4:12, 4:12] = 1.0
    problem = GradientProblem(
        incident, mask, lambda f: float(np.sum(window * np.abs(f) ** 2)), "quadratic", 20e-6
    )

    detached = characterize(problem, count=2048, realizations=32, detach_density=True)
    live = characterize(problem, count=2048, realizations=32, detach_density=False)

    record = detached.as_dict()
    record["density_detached"] = record.pop("claim")
    record["claim"] = detached.claim
    record["control_density_live"] = {
        "claim": live.claim,
        "bias": live.bias,
        "bias_in_standard_errors": live.bias_in_standard_errors,
    }
    record["promotion"] = (
        "NOT promoted. One parameter, one grid, one wavelength, one objective, no "
        "optimization loop. derivative.verified remains false for both couplers."
    )
    return record


# --- provenance and bundle ----------------------------------------------------------


def provenance(command: list[str], artifact_hashes: dict[str, str], dtype: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = None

    loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"optiland", "chromatix"})
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "coupler_id": ["C_RAY_TO_WAVE", "C_WAVE_TO_RAY"],
        "coupler_direction": "bidirectional",
        "command": command,
        "git_commit": git("rev-parse", "HEAD"),
        "dirty_worktree": bool(git("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count() or 1,
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_counts": {
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "device": "cpu",
        "dtype": dtype,
        "seed": SEED,
        "rng_generator": "numpy.random.Generator(PCG64)",
        "realizations": 32,
        "forbidden_modules_loaded": loaded,
        "engine_versions": {"numpy": np.__version__},
        "input_parameters": {
            "wavelength_m": WAVELENGTH_M,
            "grid_shape": list(GRID),
            "sample_pitch_m": list(PITCH),
        },
        "artifact_hashes": artifact_hashes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_plot(path: Path, convergence: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Two diagnostic panels. result.json stays authoritative; figures are for
    humans, and are never the basis of a pass/fail decision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    counts = [row["count"] for row in convergence["sweep"]]
    errors = [row["relative_rms"] for row in convergence["sweep"]]
    axes[0].loglog(counts, errors, "o-", label="measured")
    reference = errors[0] * (np.asarray(counts) / counts[0]) ** -0.5
    axes[0].loglog(counts, reference, "--", label=r"$N^{-1/2}$")
    axes[0].set_xticks(counts)
    axes[0].set_xticklabels([str(c) for c in counts])
    axes[0].minorticks_off()
    axes[0].set_xlabel("sampled secondary rays $N$")
    axes[0].set_ylabel("relative RMS field error")
    axes[0].set_title(
        f"Monte Carlo convergence\nfitted exponent = {convergence['fitted_exponent']:.4f}"
    )
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    rows = metrics["curvature_cases"]
    for radius_lambda in sorted({row["radius_lambda"] for row in rows}):
        subset = [row for row in rows if row["radius_lambda"] == radius_lambda]
        subset.sort(key=lambda row: row["patch_lambda"])
        patches = [row["patch_lambda"] for row in subset]
        axes[1].loglog(
            patches, [row["bound_rad"] for row in subset], "-",
            label=f"bound, $R={radius_lambda}\\lambda$",
        )
        axes[1].loglog(
            patches, [row["measured_rad"] for row in subset], "o",
            label=f"measured, $R={radius_lambda}\\lambda$",
        )
    axes[1].set_xlabel(r"patch width $D$ [$\lambda$]")
    axes[1].set_ylabel("direction error [rad]")
    axes[1].set_title(r"Curvature bound $\arcsin(D/2R)$ vs measurement")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _corruption_check(output_dir: Path, here: Path) -> dict[str, Any]:
    """Corrupt a copy of the bundle and record what the evaluator actually did.

    Run against a copy so the published bundle is never mutated. The recorded
    ``evaluator_returncode`` is observed, not asserted -- a hardcoded 2 here
    would be fabricated evidence for the one property that makes every other
    hash in the bundle meaningful.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        mutated = Path(scratch) / "bundle"
        shutil.copytree(output_dir, mutated)
        target = mutated / "arrays.npz"
        with target.open("ab") as handle:
            handle.write(b"corrupted")

        completed = subprocess.run(
            [sys.executable, str(here / "evaluate.py"), str(mutated)],
            capture_output=True,
            text=True,
            check=False,
        )
        clean = subprocess.run(
            [sys.executable, str(here / "evaluate.py"), str(output_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

    return {
        "fixture": "arrays.npz",
        "mutation": "append 9 bytes",
        "evaluator_returncode": completed.returncode,
        "evaluator_stdout_head": completed.stdout[:200],
        "clean_bundle_returncode": clean.returncode,
        # Both halves matter: the mutated bundle must be rejected AND the clean
        # bundle must be accepted, or the check is trivially satisfied.
        "detected": bool(completed.returncode == 2 and clean.returncode == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/M2/coupler"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    tolerances = yaml.safe_load((here / "tolerances.yaml").read_text())

    accuracy, arrays = accuracy_section(tolerances)
    stochastic, extra = stochastic_section(tolerances)

    # Accuracy gates before performance. A fast wrong answer is not a result.
    if not (accuracy["pass"] and stochastic["pass"]):
        result = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "failure": {
                "code": "ACCURACY_GATE_FAILED",
                "message": "accuracy or stochastic gates failed; performance not measured",
                "stage": "accuracy",
                "exception_type": None,
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps({"status": "blocked"}))
        return 1

    differentiability = differentiability_section()

    # Performance: two untimed warmups, seven timed repeats, median primary.
    field = multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    indices = draw_indices(density, 4096, np.random.default_rng(SEED))

    def one_round_trip() -> None:
        reconstruct(spectrum_to_rays(spectrum, indices, density))

    for _ in range(2):
        one_round_trip()
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        one_round_trip()
        samples.append((time.perf_counter_ns() - start) / 1e9)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    performance = {
        "warmup_runs": 2,
        "measured_repeats": 7,
        "samples_seconds": samples,
        "statistics": {
            "median_seconds": float(np.median(samples)),
            "minimum_seconds": float(np.min(samples)),
            "p95_seconds": float(np.percentile(samples, 95)),
        },
        "peak_memory": {
            "status": "measured",
            "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "bytes": int(usage.ru_maxrss) * 1024,
        },
    }

    np.savez(output_dir / "arrays.npz", **arrays)
    _write_plot(output_dir / "plot.png", extra["convergence"], accuracy["metrics"])
    (output_dir / "tolerances.yaml").write_text((here / "tolerances.yaml").read_text())
    (output_dir / "convergence.json").write_text(
        json.dumps(extra["convergence"], indent=2, sort_keys=True)
    )
    (output_dir / "ensemble_statistics.json").write_text(
        json.dumps(extra["ensemble"], indent=2, sort_keys=True)
    )
    (output_dir / "README.md").write_text((here / "README.md").read_text())

    # Scientific fingerprint: physics only. Reuses the M1 volatile-key stripping
    # so the M1.8 defect -- wall-clock leaking into a hash and making it track
    # machine load -- cannot recur here.
    projection = {
        "accuracy": strip_volatile(accuracy),
        "stochastic": strip_volatile(stochastic),
        "array_sha256": hashlib.sha256(
            b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))
        ).hexdigest(),
        "tolerances_sha256": hashlib.sha256(
            (here / "tolerances.yaml").read_bytes()
        ).hexdigest(),
        "volatile_keys_stripped": list(VOLATILE_KEYS),
    }
    fingerprint = hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":"), default=float).encode()
    ).hexdigest()

    # Write result.json and provenance.json first, then actually corrupt a copy
    # of the bundle and run the evaluator against it. The observed exit code is
    # recorded; a hardcoded expectation here would be fabricated evidence.
    def write_result(corruption: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "accuracy": accuracy,
            "stochastic": stochastic,
            "differentiability": differentiability,
            "performance": performance,
            "reproducibility": {
                "branch": "coupler",
                "scientific_fingerprint": fingerprint,
                "environment_fingerprint": hashlib.sha256(
                    json.dumps(
                        {"python": platform.python_version(), "numpy": np.__version__},
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "matching_environment_for_performance": True,
                "scientific_projection": {"volatile_keys_stripped": list(VOLATILE_KEYS)},
                "corrupted_fixture_rejection": corruption,
                "pass": bool(corruption["detected"]),
            },
        }
        (output_dir / "result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float)
        )
        return payload

    hashed_names = (
        "result.json", "arrays.npz", "tolerances.yaml", "README.md",
        "convergence.json", "ensemble_statistics.json", "plot.png",
    )

    def write_provenance() -> None:
        hashes = {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in hashed_names
        }
        (output_dir / "provenance.json").write_text(
            json.dumps(
                provenance([sys.executable, *sys.argv], hashes, "complex128"),
                indent=2, sort_keys=True, default=float,
            )
        )

    write_result({"fixture": "arrays.npz", "mutation": "pending", "evaluator_returncode": None, "detected": False})
    write_provenance()

    corruption = _corruption_check(output_dir, here)
    write_result(corruption)
    write_provenance()

    result = json.loads((output_dir / "result.json").read_text())

    print(
        json.dumps(
            {
                "status": "complete",
                "accuracy_pass": accuracy["pass"],
                "stochastic_pass": stochastic["pass"],
                "scientific_fingerprint": fingerprint,
                "differentiability_claim": differentiability["claim"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
