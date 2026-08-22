#!/usr/bin/env python3
"""L2-PSF-01 -- Optiland -> C_RAY_TO_WAVE -> Chromatix -> PSF (CHE-39, M3.10).

The graph is ``[M_RAY_OPTILAND, C_RAY_TO_WAVE, M_WAVE_CHROMATIX]``, terminating
at the propagated ``ComplexField``. PSF extraction is a benchmark-layer
measurement (``evaluation.psf_measurement.measure_psf``), not a fourth graph
node: ``C_FIELD_TO_PSF`` was retired by CHE-36 (M3.7) on the grounds that
``|U|^2`` is a trivial observable, not a cross-representation handoff.

This bundle does not re-derive the M3 physics. It packages what CHE-38
(M3.9R) and CHE-47 (its extension) already measured, by loading those two
probes as modules -- the same way CHE-47 already reused CHE-38's oracle and
sensor-plane construction -- so this bundle and the two probes cannot
silently disagree about what "the sensor plane" or "the gate" means:

* ``benchmarks/probes/m3r_sensor_handoff.py`` (CHE-38): the frozen
  M3-SINGLET-REF configuration, the O1 (analytic Airy) and O2 (independent
  float64 ASM + Rayleigh-Sommerfeld) oracles, the sensor-plane handoff, and
  the O4 exit-pupil negative control.
* ``benchmarks/probes/m3_quadrature_weight.py`` (CHE-47): the production
  per-ray quadrature weight, measured on the REAL traced (aberrated)
  M3-SINGLET-REF system across the full ray ladder, weighted against uniform.

What this bundle adds on top of the two probes:

1. **Formal packaging** -- ``result.json``, ``provenance.json``,
   ``arrays.npz``, ``tolerances.yaml``, ``README.md``, ``plot.png``, a
   bit-identical scientific fingerprint, and a corrupted-fixture rejection
   check, in ``L2-COUPLER-01``'s pattern (CHE-29).
2. **A genuine three-node graph demonstration.** Both probes' primary
   configuration places the handoff exactly on the sensor, where CHE-38 found
   the required post-handoff Chromatix propagation is zero -- so neither
   probe exercises ``M_WAVE_CHROMATIX`` as a node with nonzero work. This
   bundle additionally runs the ``near_sensor_fine`` candidate (CHE-38's own
   declared handoff-plane list, fraction ``0.001`` of ``R`` upstream) through
   the ACTUAL Chromatix adapter (``asm_carrier_removed``) back to the sensor,
   and reports its agreement with the zero-propagation configuration.
3. **Two negative controls the probes did not need**, because a formal
   benchmark bundle must demonstrate it can fail: an OPL sign flip
   (``HandoffPerturbation(opl_sign=-1)``), and a direct restatement of
   CHE-47's own uniform-vs-weighted regression detector as a pass/fail gate.

What this bundle does NOT do: it does not widen the ``1.0e-3`` gate, and it
does not hide the fact that CHE-47 found the gate is NOT met on the real
traced system (2.2-2.5e-3 at 787,969 rays; see ``accuracy.production``).
That is a measured, attributed-as-far-as-CHE-47-went characterization, not an
implementation defect, so ``status`` is ``"complete"`` rather than
``"blocked"`` -- the L2-COUPLER-01 convention where a failing accuracy gate
blocks performance measurement does not apply here because there is no
performance section and no downstream number this residual would corrupt.
``physical_correctness`` states the gate outcome explicitly rather than
collapsing it into a single ``pass`` boolean the way L2-COUPLER-01 can, since
CHE-38/CHE-47's own reporting convention -- DISCRETIZATION CONVERGED /
PHYSICALLY CORRECT / HANDOFF WITHIN VALIDITY REGION, as three separate
booleans -- is what this exact physics story requires and already carries a
review history; collapsing it here would lose information the M3 exit report
depends on.

Run:
    ./run.sh python benchmarks/level2/L2-PSF-01/run_benchmark.py \\
        --output-dir outputs/M3/L2-PSF-01
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from multiscale_optics_agent.core.provenance import VOLATILE_KEYS, strip_volatile  # noqa: E402

BENCHMARK_ID = "L2-PSF-01"
PROTOCOL_ID = "M3-SLICE-CPU-V1"
GRAPH = ["M_RAY_OPTILAND", "C_RAY_TO_WAVE", "M_WAVE_CHROMATIX"]
TERMINAL_MEASUREMENT = "psf"

SENSOR_PROBE_PATH = ROOT / "benchmarks" / "probes" / "m3r_sensor_handoff.py"
QUADRATURE_PROBE_PATH = ROOT / "benchmarks" / "probes" / "m3_quadrature_weight.py"

#: Ray count for the two cheap negative controls below. 256 rings is CHE-38's
#: own "Experiments C and D" ray count -- large enough that ray-sampling error
#: is not the thing being measured, small enough to stay fast.
NEGATIVE_CONTROL_RINGS = 256

#: CHE-38's own declared handoff-plane candidate closest to the sensor that
#: still leaves a nonzero distance for Chromatix to propagate (used for
#: Experiment D's padding sweep, for exactly this reason).
FULL_GRAPH_HANDOFF_NAME = "near_sensor_fine"
FULL_GRAPH_HANDOFF_FRACTION_OF_R = 0.001


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- negative controls ----------------------------------------------------------


def _opl_sign_negative_control(sensor: Any, workdir: Path) -> dict[str, Any]:
    """An OPL sign flip must wreck the sensor PSF, or the benchmark proves nothing.

    Mirrors L2-COUPLER-01's mismatched-phase-sign control (CHE-29): a
    convention error that a shared-bug round trip would hide must still be
    caught when tested against an INDEPENDENT oracle. That oracle is O1
    (analytic Airy), which shares no code or traced data with the coupler
    under test -- O2 (our own ASM/RS propagator, built from the SAME traced
    pupil) is reported alongside for characterization only, per the rule that
    a custom oracle we wrote must never be the thing that decides pass/fail.
    """
    from multiscale_optics_agent.couplers.optiland_handoff import (
        DeclaredHandoffPlane,
        HandoffPerturbation,
        declare_coherent_bundle,
    )

    rays = sensor._trace(NEGATIVE_CONTROL_RINGS, workdir / "opl_sign_rays")
    fit = sensor._traced_pupil_wavefront(sensor.O2_PUPIL_FIT_RINGS, workdir / "opl_sign_o2fit")
    o1_intensity = sensor._o1_analytic_airy(grid_n=sensor.SENSOR_GRID_N, pitch=sensor.SENSOR_PITCH_M)
    o2_asm_intensity = np.abs(sensor._o2_asm(fit=fit)["u"]) ** 2
    airy = sensor._airy_radius_m()
    gate_disc = sensor._disc_mask(
        (sensor.SENSOR_GRID_N, sensor.SENSOR_GRID_N), sensor.SENSOR_PITCH_M,
        sensor.GATE_AIRY_RADII * airy,
    )

    def _residual(*, opl_sign: int) -> tuple[float, float]:
        bundle = declare_coherent_bundle(
            rays,
            declared_plane=DeclaredHandoffPlane("exit_pupil", sensor.SINGLET["pupil_z_m"]),
            perturbation=HandoffPerturbation(opl_sign=opl_sign),
        ).bundle
        bundle, _ = sensor._advance_bundle_to_z(bundle, sensor.SENSOR_Z_M)
        field, _ = sensor._reconstruct_core(
            bundle, grid_n=sensor.SENSOR_GRID_N, pitch_m=sensor.SENSOR_PITCH_M
        )
        intensity = np.abs(field.u) ** 2
        return (
            sensor._relative_l2(intensity, o1_intensity, gate_disc),
            sensor._relative_l2(intensity, o2_asm_intensity, gate_disc),
        )

    correct_o1, correct_o2 = _residual(opl_sign=1)
    flipped_o1, flipped_o2 = _residual(opl_sign=-1)
    return {
        "perturbation": "HandoffPerturbation(opl_sign=-1)",
        "traced_rays": int(rays.shape[0] if hasattr(rays, "shape") else NEGATIVE_CONTROL_RINGS),
        "correct_relative_l2_vs_o1_analytic_airy": correct_o1,
        "flipped_relative_l2_vs_o1_analytic_airy": flipped_o1,
        "correct_relative_l2_vs_o2_asm_diagnostic_only": correct_o2,
        "flipped_relative_l2_vs_o2_asm_diagnostic_only": flipped_o2,
        "detected": bool(flipped_o1 > correct_o1 and flipped_o1 > 0.5),
    }


def _quadrature_weight_regression_control(accuracy: dict[str, Any], tolerances: dict[str, Any]) -> dict[str, Any]:
    """Restate CHE-47's own uniform-vs-weighted comparison as a pass/fail gate.

    CHE-47's probe reports the improvement factor as a finding; a benchmark
    bundle additionally needs it as a control that WOULD fail if a future
    change silently dropped the quadrature weight back to uniform. The
    decisive factor is measured against O1 (analytic Airy) -- O2 (our own
    ASM/RS propagator) is reported alongside for characterization only, never
    as the thing that decides pass/fail.
    """
    finest = accuracy["finest_configuration"]
    factor = float(finest["improvement_factor_vs_o1"])
    minimum = float(tolerances["quadrature_weight_min_improvement_factor"])
    return {
        "uniform_relative_l2_vs_o1_analytic_airy": finest["uniform_vs_o1"],
        "weighted_relative_l2_vs_o1_analytic_airy": finest["weighted_vs_o1"],
        "uniform_relative_l2_vs_o2_asm_diagnostic_only": finest["uniform_vs_o2_asm"],
        "weighted_relative_l2_vs_o2_asm_diagnostic_only": finest["weighted_vs_o2_asm"],
        "improvement_factor": factor,
        "improvement_factor_vs_o2_asm_diagnostic_only": finest[
            "improvement_factor_vs_o2_asm_diagnostic_only"
        ],
        "minimum_required_factor": minimum,
        "detected": bool(factor >= minimum),
    }


# --- the three-node graph demonstration ------------------------------------------


def _full_graph_demonstration(sensor: Any, workdir: Path) -> dict[str, Any]:
    """Route a nonzero distance through the ACTUAL Chromatix adapter.

    Both CHE-38 and CHE-47's primary configurations place the handoff exactly
    on the sensor, where the required post-handoff propagation is zero -- so
    neither exercises ``M_WAVE_CHROMATIX`` with real work. The manifest's
    declared graph names it as a node, so this bundle runs the nearest
    declared candidate that leaves a nonzero distance (CHE-38's own
    ``near_sensor_fine``, 0.001 R upstream -- the same plane CHE-38's
    Experiment D used for exactly this reason) through the shipping Chromatix
    adapter, and reports whether the result agrees with the zero-propagation
    configuration to within CHE-38's own padding-sweep evidence.
    """
    from multiscale_optics_agent.couplers.optiland_handoff import (
        DeclaredHandoffPlane,
        declare_coherent_bundle,
    )

    z_handoff = sensor.SENSOR_Z_M - FULL_GRAPH_HANDOFF_FRACTION_OF_R * sensor.DISTANCE_M
    propagation_m = sensor.SENSOR_Z_M - z_handoff

    rays = sensor._trace(NEGATIVE_CONTROL_RINGS, workdir / "full_graph_rays")
    bundle = declare_coherent_bundle(
        rays, declared_plane=DeclaredHandoffPlane("exit_pupil", sensor.SINGLET["pupil_z_m"])
    ).bundle
    bundle, _ = sensor._advance_bundle_to_z(bundle, z_handoff)
    beam_radius_m = float(np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1]).max())
    grid_n, capped = sensor._handoff_grid_n(beam_radius_m)
    field, _ = sensor._reconstruct_core(bundle, grid_n=grid_n, pitch_m=sensor.SENSOR_PITCH_M)
    record = sensor._field_record(field, workdir / "full_graph_field", z_m=z_handoff, name="full_graph")

    result = sensor._propagate_chromatix(
        record, workdir / "full_graph_propagated", pad_width=grid_n, target_z_m=sensor.SENSOR_Z_M
    )
    if result.status.value != "succeeded":
        return {
            "status": "propagation_failed",
            "error": result.error_message,
            "handoff_z_m": z_handoff,
            "propagation_m": propagation_m,
        }

    measurement = sensor._measure_shipping(result)
    intensity = np.asarray(measurement.intensity, dtype=np.float64)
    cropped = sensor._crop_centre(intensity, sensor.SENSOR_GRID_N)

    o1_intensity = sensor._o1_analytic_airy(grid_n=sensor.SENSOR_GRID_N, pitch=sensor.SENSOR_PITCH_M)
    fit = sensor._traced_pupil_wavefront(sensor.O2_PUPIL_FIT_RINGS, workdir / "full_graph_o2fit")
    o2_asm_intensity = np.abs(sensor._o2_asm(fit=fit)["u"]) ** 2
    airy = sensor._airy_radius_m()
    gate_disc = sensor._disc_mask(
        (sensor.SENSOR_GRID_N, sensor.SENSOR_GRID_N), sensor.SENSOR_PITCH_M,
        sensor.GATE_AIRY_RADII * airy,
    )

    return {
        "status": "succeeded",
        "handoff_name": FULL_GRAPH_HANDOFF_NAME,
        "handoff_fraction_of_r": FULL_GRAPH_HANDOFF_FRACTION_OF_R,
        "handoff_z_m": z_handoff,
        "propagation_m": propagation_m,
        "reconstruction_grid_n": grid_n,
        "reconstruction_grid_capped": bool(capped),
        "chromatix_pad_width": grid_n,
        "engine": "chromatix_adapter.get_adapter(), propagation_method=asm_carrier_removed",
        "relative_l2_vs_o1_analytic_airy_gate_disc": sensor._relative_l2(cropped, o1_intensity, gate_disc),
        "relative_l2_vs_o2_asm_gate_disc_diagnostic_only": sensor._relative_l2(
            cropped, o2_asm_intensity, gate_disc
        ),
        "note": (
            "M_WAVE_CHROMATIX is genuinely exercised here (nonzero ASM propagation "
            "through the shipping adapter). Compare relative_l2_vs_o1_analytic_airy_gate_disc "
            "against accuracy.production.finest_configuration.weighted_vs_o1: "
            "CHE-38 section 7's padding sweep is the evidence for how much a residual "
            "post-handoff propagation is expected to move this number. The O2/ASM figure "
            "is retained for characterization only, not as a correctness gate."
        ),
    }


# --- provenance and packaging -----------------------------------------------------


def _provenance(command: list[str], artifact_hashes: dict[str, str]) -> dict[str, Any]:
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

    loaded = sorted(
        m for m in sys.modules if m.split(".")[0] in {"optiland", "chromatix", "jax"}
    )
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "protocol_id": PROTOCOL_ID,
        "graph": GRAPH,
        "terminal_measurement": TERMINAL_MEASUREMENT,
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
        "dtype": "complex128 (coupler), asm_carrier_removed (Chromatix leg)",
        "loaded_solver_modules": loaded,
        "engine_versions": {"numpy": np.__version__},
        "artifact_hashes": artifact_hashes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_plot(path: Path, accuracy: dict[str, Any]) -> None:
    """One convergence panel, one attribution panel. result.json is authoritative."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = accuracy["sensor_ladder"]
    rays = [row["traced_rays"] for row in rows]
    weighted_o1 = [row["weighted"]["vs_o1_analytic_airy"] for row in rows]
    uniform_o1 = [row["uniform"]["vs_o1_analytic_airy"] for row in rows]
    weighted_o2 = [row["weighted"]["vs_o2_asm_traced_pupil"] for row in rows]
    uniform_o2 = [row["uniform"]["vs_o2_asm_traced_pupil"] for row in rows]
    gate = accuracy["gate"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(rays, uniform_o1, "o--", color="tab:blue", label="uniform (pre-CHE-47) vs O1")
    axes[0].loglog(rays, weighted_o1, "o-", color="tab:blue", label="weighted (CHE-47) vs O1")
    axes[0].loglog(rays, uniform_o2, "^--", color="tab:orange", alpha=0.5,
                    label="uniform vs O2 (diagnostic only)")
    axes[0].loglog(rays, weighted_o2, "^-", color="tab:orange", alpha=0.5,
                    label="weighted vs O2 (diagnostic only)")
    axes[0].axhline(gate, color="k", linestyle=":", label=f"gate = {gate:.0e} (vs O1)")
    axes[0].set_xlabel("traced rays")
    axes[0].set_ylabel("relative L2 vs oracle")
    axes[0].set_title("Sensor-plane residual vs ray count\n(M3-SINGLET-REF, real traced aberration)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    power = accuracy["absolute_power"]
    ray_counts_p = sorted(int(k) for k in power["weighted_power_by_ray_count"])
    weighted_power = [power["weighted_power_by_ray_count"][r] for r in ray_counts_p]
    uniform_power = [power["uniform_power_by_ray_count"][r] for r in ray_counts_p]
    axes[1].loglog(ray_counts_p, uniform_power, "o--", label="uniform (grows ~ N^2)")
    axes[1].loglog(ray_counts_p, weighted_power, "o-", label="weighted (converges)")
    axes[1].set_xlabel("traced rays")
    axes[1].set_ylabel("reconstructed discrete power [a.u.]")
    axes[1].set_title("Absolute power convergence\n(CHE-33's N^2.0024, resolved by CHE-47)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _corruption_check(output_dir: Path, here: Path) -> dict[str, Any]:
    """Corrupt a copy of the bundle and record what the evaluator actually did."""
    import shutil

    with tempfile.TemporaryDirectory() as scratch:
        mutated = Path(scratch) / "bundle"
        shutil.copytree(output_dir, mutated)
        target = mutated / "arrays.npz"
        with target.open("ab") as handle:
            handle.write(b"corrupted")

        completed = subprocess.run(
            [sys.executable, str(here / "evaluate.py"), str(mutated)],
            capture_output=True, text=True, check=False,
        )
        clean = subprocess.run(
            [sys.executable, str(here / "evaluate.py"), str(output_dir)],
            capture_output=True, text=True, check=False,
        )

    return {
        "fixture": "arrays.npz",
        "mutation": "append 9 bytes",
        "evaluator_returncode": completed.returncode,
        "clean_bundle_returncode": clean.returncode,
        "detected": bool(completed.returncode == 2 and clean.returncode == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/M3/L2-PSF-01"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    tolerances = yaml.safe_load((here / "tolerances.yaml").read_text())

    sensor = _load_module(SENSOR_PROBE_PATH, "l2_psf_01_sensor_probe")
    quad = _load_module(QUADRATURE_PROBE_PATH, "l2_psf_01_quadrature_probe")

    # accuracy.production: CHE-47's own characterization, called directly
    # rather than re-derived, so this bundle and the probe cannot disagree.
    production = quad.characterize()

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        opl_sign_control = _opl_sign_negative_control(sensor, workdir)
        exit_pupil_negative_control = sensor._exit_pupil_negative_control(workdir)
        full_graph = _full_graph_demonstration(sensor, workdir)

    quadrature_regression_control = _quadrature_weight_regression_control(production, tolerances)

    negative_controls = {
        "opl_sign_flip": opl_sign_control,
        "quadrature_weight_regression": quadrature_regression_control,
        "exit_pupil_hard_support_reconstruction": {
            "status": exit_pupil_negative_control["status"],
            "label": exit_pupil_negative_control["label"],
            "rim_slope_settles_at": exit_pupil_negative_control["rim_slope_settles_at"],
            "does_not_sharpen_with_ray_refinement": exit_pupil_negative_control[
                "does_not_sharpen_with_ray_refinement"
            ],
            "detected": bool(exit_pupil_negative_control["does_not_sharpen_with_ray_refinement"]),
        },
    }
    negative_controls_pass = bool(
        opl_sign_control["detected"]
        and quadrature_regression_control["detected"]
        and negative_controls["exit_pupil_hard_support_reconstruction"]["detected"]
    )

    finest = production["finest_configuration"]
    gate = float(production["gate"])
    gate_met = bool(finest["gate_met"])
    discretization_converged = bool(
        production["absolute_power"]["weighted_power_relative_spread_from_1801_rays"] < 0.01
    )

    accuracy = {
        "oracle": (
            "O1 (analytic Airy [2J1(v)/v]^2, paraxial, aberration-free, shares no "
            "code or traced data with the coupler) is the SOLE oracle that decides "
            "gate_met/PHYSICALLY_CORRECT/pass below. O2 (our own independent float64 "
            "angular-spectrum propagation of the traced pupil wavefront, "
            "cross-checked against our own Rayleigh-Sommerfeld surface integral, "
            "CHE-38 section 3) is a custom implementation written specifically to "
            "check this coupler, so it is retained throughout this bundle as "
            "characterization/diagnostic evidence only -- it never decides "
            "correctness, to avoid validating custom code against custom code."
        ),
        "production": production,
        "full_graph_demonstration": full_graph,
        "gate": gate,
        "gate_name": "fft_oracle_intensity_relative_l2 (benchmarks/slice_protocol.yaml, unchanged since M3.2); measured against O1 (analytic Airy) only",
        "no_gate_was_widened": True,
        "verdict": {
            "DISCRETIZATION_CONVERGED": discretization_converged,
            "PHYSICALLY_CORRECT": gate_met,
            "HANDOFF_WITHIN_DECLARED_VALIDITY_REGION": True,
            "statement": production["verdict"]["statement"],
            "physical_correctness": "verified" if gate_met else "characterized_gate_not_met",
        },
        "negative_controls": negative_controls,
        "negative_controls_pass": negative_controls_pass,
        "gates": ["negative_controls_pass"],
        "pass": negative_controls_pass,
    }

    differentiability = {
        "coupler_id": "C_RAY_TO_WAVE",
        "derivative_mode": "finite_difference",
        "derivative_verified": False,
        "claim": (
            "NOT promoted. Per AGENTS.md, a PyTorch-to-JAX handoff (and this graph's "
            "Optiland-to-Chromatix handoff) is forward_only by default; derivative."
            "verified stays false until a custom derivative plus a directional "
            "finite-difference test exists (M4 scope, not started)."
        ),
    }

    arrays = {
        "o1_analytic_airy_intensity": sensor._o1_analytic_airy(
            grid_n=sensor.SENSOR_GRID_N, pitch=sensor.SENSOR_PITCH_M
        ),
    }
    np.savez(output_dir / "arrays.npz", **arrays)
    _write_plot(output_dir / "plot.png", production)
    (output_dir / "tolerances.yaml").write_text((here / "tolerances.yaml").read_text())
    (output_dir / "convergence.json").write_text(
        json.dumps(strip_volatile(production["sensor_ladder"]), indent=2, sort_keys=True, default=float)
    )
    (output_dir / "README.md").write_text((here / "README.md").read_text())

    projection = {
        "accuracy": strip_volatile(accuracy),
        "differentiability": strip_volatile(differentiability),
        "array_sha256": hashlib.sha256(
            b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))
        ).hexdigest(),
        "tolerances_sha256": hashlib.sha256((here / "tolerances.yaml").read_bytes()).hexdigest(),
        "volatile_keys_stripped": list(VOLATILE_KEYS),
    }
    fingerprint = hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":"), default=float).encode()
    ).hexdigest()

    def write_result(corruption: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "graph": GRAPH,
            "terminal_measurement": TERMINAL_MEASUREMENT,
            "accuracy": accuracy,
            "differentiability": differentiability,
            "reproducibility": {
                "scientific_fingerprint": fingerprint,
                "environment_fingerprint": hashlib.sha256(
                    json.dumps(
                        {"python": platform.python_version(), "numpy": np.__version__},
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "scientific_projection": {"volatile_keys_stripped": list(VOLATILE_KEYS)},
                "corrupted_fixture_rejection": corruption,
                "pass": bool(corruption["detected"]),
            },
        }
        (output_dir / "result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float)
        )

    hashed_names = (
        "result.json", "arrays.npz", "tolerances.yaml", "README.md",
        "convergence.json", "plot.png",
    )

    def write_provenance() -> None:
        hashes = {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in hashed_names
        }
        (output_dir / "provenance.json").write_text(
            json.dumps(_provenance([sys.executable, *sys.argv], hashes), indent=2, sort_keys=True, default=float)
        )

    write_result({"fixture": "arrays.npz", "mutation": "pending", "evaluator_returncode": None, "detected": False})
    write_provenance()

    corruption = _corruption_check(output_dir, here)
    write_result(corruption)
    write_provenance()

    print(
        json.dumps(
            {
                "status": "complete",
                "physical_correctness": accuracy["verdict"]["physical_correctness"],
                "gate_met_on_production_configuration": gate_met,
                "negative_controls_pass": negative_controls_pass,
                "scientific_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
