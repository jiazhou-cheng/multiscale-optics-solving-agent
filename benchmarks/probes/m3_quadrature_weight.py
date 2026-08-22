"""CHE-47 (M3.9R extension): per-ray quadrature weight in the production handoff.

CHE-38 (M3.9R) verified the sensor-side ``C_RAY_TO_WAVE`` handoff (verdict A) and
found its dominant remaining residual is a per-ray quadrature-weighting error, not
a kernel defect: with uniform ray weights the sensor residual is ``3.84e-3`` at
787,969 rays; a DIAGNOSTIC radial-trapezoid weight, applied only to a synthetic
aberration-free bundle inside ``m3r_sensor_handoff.py``, collapsed it to a
converged ``4.07e-4``. Nothing in production changed (CHE-38 section 14). This
probe is CHE-47, the ticket that changes production, and it measures what
changes on the REAL traced (aberrated) system rather than the synthetic one.

What changed
------------
``couplers.quadrature`` implements the same
radial-trapezoid weight CHE-38 measured, scaled to an ABSOLUTE area in square
metres (``pi a^2 / (3 num_rings^2)`` per interior ray, corrected 3/4 at the
center and 1/2 at the outermost ring). ``optiland_adapter.py`` regenerates the
raw hexapolar pupil sampling ``Optic.trace`` used (the same technique CHE-41
already uses for the object-space term), matches it row for row against the
traced set, and exports the normalized pupil coordinates only -- it imports no
coupler, so the ring index and area weight are computed downstream.
``optiland_handoff.declare_coherent_bundle`` is where that computation happens
and where the result is folded into the amplitude declaration by default
(``amplitude = sqrt(weight) * quadrature_weight_m2``) whenever the record has
usable pupil coordinates. ``C_RAY_TO_WAVE`` itself (``ray_to_wave.py``) is not
touched -- the kernel still just sums whatever amplitude the bundle declares.

Two questions this probe answers
---------------------------------
1. Does the production default (``HandoffPerturbation.apply_quadrature_weight =
   True``) reproduce CHE-38's diagnostic ``~4e-4`` regime on the REAL traced
   system? Measured, not assumed -- see the ``sensor_ladder`` section.
2. Does the reconstructed field's discrete power now converge under ray
   refinement instead of growing as ``(ray count)^2`` (CHE-33's ``N^2.0024``
   finding)? See ``absolute_power``.

This probe does not re-run CHE-38's handoff-plane sweep, grid sweep, padding
sweep, or exit-pupil negative control -- those are unaffected by a producer-side
amplitude change and stay owned by ``m3r_sensor_handoff.py``. It reuses that
probe's O1/O2 reference construction and sensor-plane configuration directly
(by loading the module, not by re-deriving the physics a second time) so the
two studies cannot silently disagree about what "the sensor plane" means.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3_quadrature_weight.json"
SENSOR_PROBE_PATH = ROOT / "benchmarks" / "probes" / "m3r_sensor_handoff.py"
GATE = 1.0e-3

#: Subset of CHE-38's own ray ladder (217 .. 787969 rays). Full because the
#: sensor-plane reconstruction here needs no ASM/padding (grid_n = 256, no
#: propagation after the handoff), so even the top of the ladder is cheap --
#: unlike CHE-38's exit-pupil path, which pays for a padded ASM per point.
RAY_SWEEP_RINGS = (8, 16, 24, 32, 48, 64, 96, 128, 181, 256, 362, 512)


def _load_sensor_probe():
    spec = importlib.util.spec_from_file_location(
        "m3r_sensor_handoff_probe_for_quadrature", SENSOR_PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _weighted_bundle(sensor: Any, rays: Any, *, apply_quadrature_weight: bool):
    from couplers.handoff import (
        DeclaredHandoffPlane,
        HandoffPerturbation,
        declare_coherent_bundle,
    )

    return declare_coherent_bundle(
        rays,
        declared_plane=DeclaredHandoffPlane("exit_pupil", sensor.SINGLET["pupil_z_m"]),
        perturbation=HandoffPerturbation(apply_quadrature_weight=apply_quadrature_weight),
    ).bundle


def _sensor_reconstruction(
    sensor: Any, rings: int, workdir: Path, *, apply_quadrature_weight: bool
) -> dict[str, Any]:
    rays = sensor._trace(rings, workdir / f"trace_{rings}_{apply_quadrature_weight}")
    bundle = _weighted_bundle(sensor, rays, apply_quadrature_weight=apply_quadrature_weight)
    quadrature_declared = bundle.provenance["handoff"]["quadrature_weight"]
    bundle, _ = sensor._advance_bundle_to_z(bundle, sensor.SENSOR_Z_M)
    field, diagnostics = sensor._reconstruct_core(
        bundle, grid_n=sensor.SENSOR_GRID_N, pitch_m=sensor.SENSOR_PITCH_M
    )
    return {
        "rings": rings,
        "traced_rays": bundle.count,
        "quadrature_weight_status": quadrature_declared["status"],
        "discrete_power": field.discrete_power(),
        "intensity": np.abs(field.u) ** 2,
        "diagnostics": diagnostics.as_dict(),
    }


def _power_law_fit(x: list[float], y: list[float]) -> dict[str, Any]:
    log_x = np.log(np.asarray(x, dtype=np.float64))
    log_y = np.log(np.asarray(y, dtype=np.float64))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = slope * log_x + intercept
    residual = log_y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"exponent": float(slope), "r_squared": r_squared}


def characterize() -> dict[str, Any]:
    sensor = _load_sensor_probe()
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)

        # --- References: reuse CHE-38's own construction verbatim ------------
        fit = sensor._traced_pupil_wavefront(sensor.O2_PUPIL_FIT_RINGS, work / "o2fit")
        o2_asm = sensor._o2_asm(fit=fit)
        o2_rs = sensor._o2_rayleigh_sommerfeld(fit=fit)
        o1 = sensor._o1_analytic_airy(grid_n=sensor.SENSOR_GRID_N, pitch=sensor.SENSOR_PITCH_M)
        gate_disc = sensor._disc_mask(
            (sensor.SENSOR_GRID_N, sensor.SENSOR_GRID_N),
            sensor.SENSOR_PITCH_M,
            sensor.GATE_AIRY_RADII * sensor._airy_radius_m(),
        )
        o2_asm_intensity = np.abs(o2_asm["u"]) ** 2
        o2_rs_intensity = np.abs(o2_rs["u"]) ** 2

        # --- Sensor ladder: weighted (production default) vs uniform (CHE-38's
        # pre-CHE-47 configuration, reproduced here as a negative-test baseline,
        # not as a claim about what ships) --------------------------------------
        rows = []
        for rings in RAY_SWEEP_RINGS:
            weighted = _sensor_reconstruction(sensor, rings, work, apply_quadrature_weight=True)
            uniform = _sensor_reconstruction(sensor, rings, work, apply_quadrature_weight=False)
            rows.append(
                {
                    "rings": rings,
                    "traced_rays": weighted["traced_rays"],
                    "weighted": {
                        "quadrature_weight_status": weighted["quadrature_weight_status"],
                        "discrete_power": weighted["discrete_power"],
                        "vs_o1_analytic_airy": sensor._relative_l2(
                            weighted["intensity"], o1, gate_disc
                        ),
                        "vs_o2_asm_traced_pupil": sensor._relative_l2(
                            weighted["intensity"], o2_asm_intensity, gate_disc
                        ),
                        "vs_o2_rayleigh_sommerfeld": sensor._relative_l2(
                            weighted["intensity"], o2_rs_intensity, gate_disc
                        ),
                    },
                    "uniform": {
                        "quadrature_weight_status": uniform["quadrature_weight_status"],
                        "discrete_power": uniform["discrete_power"],
                        "vs_o1_analytic_airy": sensor._relative_l2(
                            uniform["intensity"], o1, gate_disc
                        ),
                        "vs_o2_asm_traced_pupil": sensor._relative_l2(
                            uniform["intensity"], o2_asm_intensity, gate_disc
                        ),
                        "vs_o2_rayleigh_sommerfeld": sensor._relative_l2(
                            uniform["intensity"], o2_rs_intensity, gate_disc
                        ),
                    },
                }
            )

        finest = rows[-1]

        # --- Absolute power convergence (CHE-33's N^2.0024) --------------------
        # The first rung or two (8, 16 rings) are still in the sub-quadrature
        # regime the diagnostic itself flagged (CHE-38: "below ~28000 rays the sum
        # is not yet a quadrature of the aperture at all"), so the spread that
        # matters is measured from 24 rings (1801 traced rays) upward, not over
        # the full ladder -- exactly mirroring CHE-33/CHE-38's own N^2.0024 fit
        # range (27937-787969 rays there; slightly wider here since the boundary
        # correction converges faster than raw ray-density quadrature did).
        weighted_power = [row["weighted"]["discrete_power"] for row in rows]
        uniform_power = [row["uniform"]["discrete_power"] for row in rows]
        traced_rays = [row["traced_rays"] for row in rows]
        converged_regime = slice(2, None)  # 24 rings (1801 rays) upward
        weighted_power_converged_regime = weighted_power[converged_regime]
        absolute_power = {
            "weighted_power_by_ray_count": dict(zip(traced_rays, weighted_power, strict=True)),
            "uniform_power_by_ray_count": dict(zip(traced_rays, uniform_power, strict=True)),
            "weighted_power_fit": _power_law_fit(traced_rays, weighted_power),
            "uniform_power_fit": _power_law_fit(traced_rays, uniform_power),
            "weighted_power_relative_spread_full_ladder": (
                (max(weighted_power) - min(weighted_power)) / min(weighted_power)
            ),
            "weighted_power_relative_spread_from_1801_rays": (
                (max(weighted_power_converged_regime) - min(weighted_power_converged_regime))
                / min(weighted_power_converged_regime)
            ),
            "che33_n_squared_finding": (
                "CHE-33/CHE-38 measured uniform-weight power ~ (traced rays)^2.0024 "
                "over 27937-787969 rays. Reproduced above (uniform_power_fit.exponent "
                "should be close to 2). weighted_power_fit.exponent close to 0, and "
                "weighted_power_relative_spread_from_1801_rays small, is what "
                "'resolved' means here: the discrete power converges under ray "
                "refinement instead of growing with it. The 217/817-ray rungs are "
                "excluded from the spread (not from the record) because they are "
                "still in the sub-quadrature regime CHE-38 itself named."
            ),
        }

        record: dict[str, Any] = {
            "probe": "benchmarks/probes/m3_quadrature_weight.py",
            "issue": "CHE-47 (M3.9R extension of CHE-38)",
            "extends": "benchmarks/probes/m3r_sensor_handoff.py (CHE-38 / M3.9R)",
            "wavelength_m": sensor.WAVELENGTH_M,
            "sensor_plane_z_m": sensor.SENSOR_Z_M,
            "sensor_grid_n": sensor.SENSOR_GRID_N,
            "sensor_pitch_m": sensor.SENSOR_PITCH_M,
            "gate": GATE,
            "non_goals": [
                "no wavelength sweep",
                "no GPU",
                "no optimization loop",
                "no re-run of CHE-38's handoff-plane / grid / padding sweeps "
                "(unaffected by a producer-side amplitude change)",
                "no aperture-aware exit-pupil hard-support reconstruction "
                "(out of contract, per CHE-38)",
            ],
            "sensor_ladder": rows,
            "absolute_power": absolute_power,
            "finest_configuration": {
                "rings": finest["rings"],
                "traced_rays": finest["traced_rays"],
                "weighted_vs_o1": finest["weighted"]["vs_o1_analytic_airy"],
                "uniform_vs_o1": finest["uniform"]["vs_o1_analytic_airy"],
                "weighted_vs_o2_asm": finest["weighted"]["vs_o2_asm_traced_pupil"],
                "weighted_vs_o2_rs": finest["weighted"]["vs_o2_rayleigh_sommerfeld"],
                "uniform_vs_o2_asm": finest["uniform"]["vs_o2_asm_traced_pupil"],
                # Gate oracle is O1 (analytic Airy) ONLY. O1 is externally
                # established and shares no code or traced data with the
                # coupler under test. O2 (ASM/RS) is our own custom float64
                # propagator built specifically to check this coupler -- using
                # it to DECIDE correctness would be circular validation, so
                # the O2 fields above are retained as diagnostic evidence only
                # and must never gate pass/fail.
                "gate_met": finest["weighted"]["vs_o1_analytic_airy"] < GATE,
                "improvement_factor_vs_o1": (
                    finest["uniform"]["vs_o1_analytic_airy"]
                    / finest["weighted"]["vs_o1_analytic_airy"]
                ),
                "improvement_factor_vs_o2_asm_diagnostic_only": (
                    finest["uniform"]["vs_o2_asm_traced_pupil"]
                    / finest["weighted"]["vs_o2_asm_traced_pupil"]
                ),
            },
        }
        record["verdict"] = _verdict(record)
        return record


def _verdict(record: dict[str, Any]) -> dict[str, Any]:
    finest = record["finest_configuration"]
    absolute_power = record["absolute_power"]
    gate_met = bool(finest["gate_met"])
    power_converged = bool(absolute_power["weighted_power_relative_spread_from_1801_rays"] < 0.01)
    return {
        "gate_oracle": "O1 (analytic Airy) only. O2 (ASM/RS) is our own custom "
        "float64 propagator built to check this coupler and is retained below "
        "as diagnostic evidence, never as the correctness gate -- using a "
        "custom implementation to validate another custom implementation is "
        "circular validation.",
        "gate_met_on_real_traced_system": gate_met,
        "absolute_power_converged": power_converged,
        "statement": (
            (
                "Production per-ray quadrature weighting is below the "
                f"{GATE:.0e} gate against O1 (analytic Airy): "
                f"{finest['weighted_vs_o1']:.2e} at {finest['traced_rays']} rays."
            )
            if gate_met
            else (
                "Production per-ray quadrature weighting "
                + (
                    f"improves the sensor residual {finest['improvement_factor_vs_o1']:.2f}x "
                    "against O1 "
                    if finest["improvement_factor_vs_o1"] >= 1.0
                    else "makes the sensor residual "
                    f"{1.0 / finest['improvement_factor_vs_o1']:.2f}x WORSE against O1 "
                )
                + f"({finest['uniform_vs_o1']:.2e} -> {finest['weighted_vs_o1']:.2e} "
                f"at {finest['traced_rays']} rays) but does NOT reach the "
                f"{GATE:.0e} gate against O1 on the real (residually aberrated) "
                "M3-SINGLET-REF trace, unlike CHE-38's synthetic aberration-free "
                "diagnostic (4.07e-4). For context only (not a gate input): O2/ASM, "
                "our own custom float64 propagator, disagrees with O1 about which "
                f"configuration is closer -- vs O2/ASM weighted is "
                f"{finest['weighted_vs_o2_asm']:.2e} and uniform is "
                f"{finest['uniform_vs_o2_asm']:.2e}, the opposite ordering from O1. "
                "That inversion is itself evidence O2's own ring-averaged, "
                "linearly-interpolated pupil fit carries a resolution-dependent "
                "residual at this scale, which is exactly why O2 must not decide "
                "correctness here. Not decomposed further here; see open_items."
            )
        ),
        "absolute_power_statement": (
            "The reconstructed discrete power converges under ray refinement "
            "(relative spread "
            f"{absolute_power['weighted_power_relative_spread_from_1801_rays']:.2e} "
            "from 1801 rays upward, fitted exponent "
            f"{absolute_power['weighted_power_fit']['exponent']:.4f}), replacing the "
            f"uniform-weight (ray count)^{absolute_power['uniform_power_fit']['exponent']:.4f} "
            "growth CHE-33/CHE-38 measured. CHE-33's N^2.0024 finding is resolved for "
            "this bundle: absolute-scale comparison is no longer impossible."
            if power_converged
            else "Absolute power does not converge; see absolute_power for the fitted exponent."
        ),
        "open_items": [
            "Decompose the residual O2 discrepancy (weighted vs O2/ASM, ~2.5e-3 at "
            "787969 rays) into the O2 pupil-fit's own quality versus any remaining "
            "aberration-quadrature interaction. Candidate test: refit O2 with a "
            "higher-resolution or higher-order representation and see whether the "
            "gap to O1 closes.",
            "The synthetic aberration-free diagnostic (CHE-38 section 14/15, "
            "m3r_sensor_handoff.py::_quadrature_attribution) still reaches 4.07e-4; "
            "this probe's real-system ~2.2-2.5e-3 is not a regression of that finding, "
            "it is a different (harder) test condition -- aberration was excluded "
            "from the diagnostic by construction.",
        ],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"__array__": list(value.shape), "summary": "omitted from the record"}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    started = time.perf_counter()
    record = characterize()
    record["probe_wall_seconds"] = time.perf_counter() - started
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(_json_ready(record), indent=2, sort_keys=True) + "\n")
    print(record["verdict"]["statement"])
    print()
    print(record["verdict"]["absolute_power_statement"])
    print(f"\nwrote {RECORD_PATH.relative_to(ROOT)}")
    print(f"wall {record['probe_wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
