"""Evaluate Chromatix scalar and vector propagation against analytic oracles.

One command generates the analytic reference and evaluates every CPU case:

    ./run.sh python benchmarks/level1/L1-WAVE-01/evaluate.py --case gaussian

Cases 1 and 2 drive the accepted CHE-14 baseline
(``ChromatixAdapter.run_standalone``). Case 3 needs Chromatix's vector path,
which that baseline deliberately rejects, so it goes through the narrow
``chromatix_benchmark_adapter``. No Optiland and no coupler is loaded, and
that is asserted at exit.

Progression
-----------
1. **Exact homogeneous primitive.** FFT-bin plane-wave eigenmodes, unpadded.
   The oracle is exact, so the tolerance is *derived* from float32 phase
   round-off rather than chosen.
2. **Ideal signed paraxial focusing.** Rectangular pupil, ideal thin lens,
   three signed tilts. An analytic Fresnel/Fourier oracle sets the physics; an
   independent float64 angular spectrum separates its paraxial model error
   from the solver's implementation error.
3. **High-NA vectorial focusing.** Full vector focal field against an
   independent float64 Richards-Wolf quadrature oracle.

Sampling, padding, reference-quadrature convergence, and negative
perturbations qualify the benchmark; they are not additional physical cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import measurement  # noqa: E402
import oracles  # noqa: E402
from generate_reference import (  # noqa: E402
    build_reference,
    load_config,
    serializable,
)

from multiscale_optics_agent.adapters.base import RunStatus  # noqa: E402
from multiscale_optics_agent.adapters.chromatix_adapter import (  # noqa: E402
    ChromatixAdapter,
    ChromatixWaveRequest,
)
from multiscale_optics_agent.adapters.chromatix_benchmark_adapter import (  # noqa: E402
    high_na_vector_focus,
)

BENCHMARK_ID = "L1-WAVE-01"
PROTOCOL_ID = "M1-BASELINE-CPU-V1"
FORBIDDEN_PREFIXES = ("optiland", "multiscale_optics_agent.couplers")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / BENCHMARK_ID


# ---------------------------------------------------------------------------
# Solver invocation
# ---------------------------------------------------------------------------
def _propagate(
    adapter: ChromatixAdapter,
    *,
    field: np.ndarray,
    pitch_m: float,
    wavelength_m: float,
    refractive_index: float,
    z_m: float,
    pad_width: int,
    output_directory: Path,
    reference_plane: str,
) -> Any:
    padding_policy = "explicit" if pad_width > 0 else "none"
    return adapter.run_standalone(
        ChromatixWaveRequest(
            input_field_array=field.astype(np.complex64),
            wavelength_m=wavelength_m,
            sample_pitch_m=(pitch_m, pitch_m),
            z_m=z_m,
            refractive_index=refractive_index,
            padding_policy=padding_policy,
            pad_width=pad_width if pad_width > 0 else None,
            output_mode="full",
            reference_plane=reference_plane,
            output_directory=output_directory,
        )
    )


def _solver_block(result: Any) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "output_shape": list(result.output_shape or []),
        "pad_width": result.pad_width,
        "padded": result.padded,
        "cropped": result.cropped,
        "resampled": result.summary_metrics["resampled"],
        "output_sample_pitch_m": result.summary_metrics["output_sample_pitch_m"],
        "output_extent_m": result.summary_metrics["output_extent_m"],
        "power_in": result.summary_metrics["power_in"],
        "power_out": result.summary_metrics["power_out"],
        "power_conservation_ratio": result.summary_metrics["power_conservation_ratio"],
        "input_edge_energy_fraction": result.summary_metrics["input_edge_energy_fraction"],
        "output_edge_energy_fraction": result.summary_metrics["output_edge_energy_fraction"],
        "finite_window_interpretation": result.summary_metrics["finite_window_interpretation"],
        "package_version": result.package_version,
        "package_commit": result.package_commit,
        "runtime_seconds": result.runtime_seconds,
    }


def _failed_record(record: dict[str, Any], result: Any) -> dict[str, Any]:
    record["solver"] = {
        "status": "failed",
        "failure": result.failure.model_dump(mode="json") if result.failure else None,
    }
    record["errors"] = {}
    record["gates"] = {"solver_succeeded": False}
    record["pass"] = False
    return record


# ---------------------------------------------------------------------------
# Case 1 -- exact homogeneous primitive
# ---------------------------------------------------------------------------
def evaluate_case1(
    adapter: ChromatixAdapter,
    reference: dict[str, Any],
    tolerances: dict[str, float],
    scratch: Path,
    *,
    perturbation: str | None = None,
) -> dict[str, Any]:
    grid = reference["grid"]
    physics = reference["physics"]
    closed = reference["closed_form"]
    pitch_m = float(grid["pitch_m"])
    z_m = float(reference["planes"]["z_m"])
    input_field = reference["_input_field"]

    name = reference["name"] if perturbation is None else f"{reference['name']}__{perturbation}"
    record: dict[str, Any] = {
        "name": name,
        "base_case": reference["name"],
        "case": "case1",
        "role": "accuracy" if perturbation is None else "perturbation",
        "perturbation": perturbation,
        "grid": grid,
        "mode": reference["mode"],
        "planes": reference["planes"],
        "sampling": reference["sampling"],
    }

    result = _propagate(
        adapter,
        field=input_field,
        pitch_m=pitch_m,
        wavelength_m=float(physics["wavelength_m"]),
        refractive_index=float(physics["refractive_index"]),
        z_m=z_m,
        pad_width=int(grid["pad_width"]),
        output_directory=scratch / name,
        reference_plane=f"input plane z=0; output plane z={z_m:.6e} m",
    )
    if result.status is not RunStatus.SUCCEEDED:
        return _failed_record(record, result)

    measured = np.load(str(result.output_field_path))

    axial = float(closed["axial_wavenumber_rad_per_m"])
    if perturbation == "case1_paraxial_dispersion":
        # Paraxial expansion of the exact sqrt dispersion relation. If Case 1
        # cannot tell these apart it is not testing what it claims to.
        mode = reference["mode"]
        wavelength_m = float(physics["wavelength_m"])
        refractive_index = float(physics["refractive_index"])
        sin_squared = (wavelength_m / refractive_index) ** 2 * (
            mode["frequency_x_per_m"] ** 2 + mode["frequency_y_per_m"] ** 2
        )
        axial = 2.0 * np.pi * refractive_index / wavelength_m * (1.0 - 0.5 * sin_squared)
    expected = oracles.plane_wave_transfer(input_field, axial, z_m)

    phase_error = float(np.max(np.abs(measurement.wrapped_phase_difference(measured, expected))))
    amplitude_error = measurement.amplitude_ratio_error(measured, expected)
    agreement = measurement.complex_field_agreement(measured, expected)
    power_error = abs(float(result.summary_metrics["power_conservation_ratio"]) - 1.0)

    # Round trip: propagate back and recover the launched field exactly.
    round_trip_rms = float("nan")
    if perturbation is None:
        back = _propagate(
            adapter,
            field=measured,
            pitch_m=pitch_m,
            wavelength_m=float(physics["wavelength_m"]),
            refractive_index=float(physics["refractive_index"]),
            z_m=-z_m,
            pad_width=int(grid["pad_width"]),
            output_directory=scratch / f"{name}__roundtrip",
            reference_plane=f"return leg from z={z_m:.6e} m back to z=0",
        )
        if back.status is RunStatus.SUCCEEDED:
            returned = np.load(str(back.output_field_path))
            round_trip_rms = measurement.complex_field_agreement(returned, input_field)[
                "normalized_rms_error"
            ]

    round_off = float(closed["float32_phase_round_off_rad"])
    phase_tolerance = max(
        tolerances["case1_phase_floor_rad"],
        tolerances["case1_phase_round_off_factor"] * round_off,
    )
    rms_tolerance = max(
        tolerances["case1_rms_floor"], tolerances["case1_rms_round_off_factor"] * round_off
    )

    record["solver"] = _solver_block(result)
    record["closed_form"] = closed
    record["errors"] = {
        "phase_error_rad": phase_error,
        "amplitude_relative": amplitude_error,
        "complex_normalized_rms": agreement["normalized_rms_error"],
        "complex_overlap": agreement["overlap"],
        "power_conservation_relative": power_error,
        "round_trip_normalized_rms": round_trip_rms,
    }
    record["derived_tolerances"] = {
        "float32_phase_round_off_rad": round_off,
        "phase_error_rad": phase_tolerance,
        "complex_normalized_rms": rms_tolerance,
    }
    record["gates"] = {
        "solver_succeeded": True,
        "phase_transfer": phase_error <= phase_tolerance,
        "amplitude_unchanged": amplitude_error <= tolerances["case1_amplitude_relative"],
        "complex_field": agreement["normalized_rms_error"] <= rms_tolerance,
        "power_conserved": power_error <= tolerances["case1_power_conservation_relative"],
        "no_padding_applied": int(result.pad_width or 0) == 0 and result.padded is False,
        "shape_unchanged": tuple(result.output_shape or ()) == (grid["n"], grid["n"]),
        "round_trip": (
            True
            if perturbation is not None
            else bool(
                np.isfinite(round_trip_rms)
                and round_trip_rms
                <= max(tolerances["case1_round_trip_rms_floor"], 2.0 * rms_tolerance)
            )
        ),
    }
    record["pass"] = all(record["gates"].values())
    record["_measured_field"] = measured
    record["_expected_field"] = expected
    record["_solver_summary"] = json.loads(Path(str(result.summary_path)).read_text())
    return record


# ---------------------------------------------------------------------------
# Case 2 -- ideal signed paraxial focusing
# ---------------------------------------------------------------------------
def evaluate_case2(
    adapter: ChromatixAdapter,
    reference: dict[str, Any],
    tolerances: dict[str, float],
    scratch: Path,
    *,
    perturbation: str | None = None,
) -> dict[str, Any]:
    grid = reference["grid"]
    pupil = reference["pupil"]
    physics = reference["physics"]
    closed = reference["closed_form"]
    pitch_m = float(grid["pitch_m"])
    pad_width = int(grid["pad_width"])
    focal_length_m = float(pupil["focal_length_m"])
    wavelength_m = float(physics["wavelength_m"])
    refractive_index = float(physics["refractive_index"])

    name = reference["name"] if perturbation is None else f"{reference['name']}__{perturbation}"
    record: dict[str, Any] = {
        "name": name,
        "base_case": reference["name"],
        "case": "case2",
        "role": "accuracy" if perturbation is None else "perturbation",
        "perturbation": perturbation,
        "grid": grid,
        "pupil": pupil,
        "planes": reference["planes"],
        "sampling": reference["sampling"],
    }

    input_field = reference["_input_field"]
    solver_wavelength_m = wavelength_m
    if perturbation == "case2_lens_sign_flip":
        input_field = oracles.thin_lens_pupil_field(
            n=int(grid["n"]),
            pitch_m=pitch_m,
            samples_x=int(pupil["aperture_samples_x"]),
            samples_y=int(pupil["aperture_samples_y"]),
            focal_length_m=-focal_length_m,  # diverging: wrong sign for a focus at +f
            tilt_x_rad=float(pupil["tilt_x_rad"]),
            tilt_y_rad=float(pupil["tilt_y_rad"]),
            wavelength_m=wavelength_m,
            refractive_index=refractive_index,
        )
    if perturbation == "case2_si_scale":
        solver_wavelength_m = 0.532  # micrometre number supplied as if it were metres

    result = _propagate(
        adapter,
        field=input_field,
        pitch_m=pitch_m,
        wavelength_m=solver_wavelength_m,
        refractive_index=refractive_index,
        z_m=focal_length_m,
        pad_width=pad_width,
        output_directory=scratch / name,
        reference_plane=f"pupil plane z=0; focal plane z={focal_length_m:.6e} m",
    )
    if result.status is not RunStatus.SUCCEEDED:
        return _failed_record(record, result)

    measured = np.load(str(result.output_field_path))
    if perturbation == "case2_axis_transpose":
        measured = measured.T

    expected = reference["_expected_field"]
    independent = oracles.numpy_angular_spectrum(
        input_field,
        pitch_m=pitch_m,
        wavelength_m=wavelength_m,
        refractive_index=refractive_index,
        z_m=focal_length_m,
        pad_width=pad_width,
    )

    padded_n = int(grid["padded_n"])
    coordinates = measurement.grid_coordinates_m(padded_n, pitch_m)
    intensity = np.abs(measured) ** 2
    centroid = measurement.intensity_centroid_m(intensity, pitch_m)
    row = int(np.argmin(np.abs(coordinates - closed["centroid_y_m"])))
    column = int(np.argmin(np.abs(coordinates - closed["centroid_x_m"])))
    fwhm_x = measurement.fwhm_m(intensity[row], pitch_m)
    fwhm_y = measurement.fwhm_m(intensity[:, column], pitch_m)
    sidelobe = measurement.first_sidelobe_ratio(intensity[row])

    agreement_fresnel = measurement.complex_field_agreement(measured, expected)
    agreement_independent = measurement.complex_field_agreement(measured, independent)
    power_error = abs(float(result.summary_metrics["power_conservation_ratio"]) - 1.0)

    def relative(value: float, target: float) -> float:
        if not np.isfinite(value) or target == 0.0:
            return float("inf") if not np.isfinite(value) else abs(value)
        return abs(value / target - 1.0)

    on_grid = reference["estimator_on_analytic_field"]
    independent_intensity = np.abs(independent) ** 2
    independent_fwhm_x = measurement.fwhm_m(independent_intensity[row], pitch_m)
    independent_fwhm_y = measurement.fwhm_m(independent_intensity[:, column], pitch_m)

    errors = {
        "centroid_x_input_pixels": abs(centroid["centroid_x_m"] - closed["centroid_x_m"]) / pitch_m,
        "centroid_y_input_pixels": abs(centroid["centroid_y_m"] - closed["centroid_y_m"]) / pitch_m,
        "fwhm_x_relative": relative(fwhm_x, closed["fwhm_x_m"]),
        "fwhm_y_relative": relative(fwhm_y, closed["fwhm_y_m"]),
        "first_sidelobe_relative": relative(sidelobe, closed["first_sidelobe_ratio"]),
        "overlap_fresnel": agreement_fresnel["overlap"],
        "normalized_rms_fresnel": agreement_fresnel["normalized_rms_error"],
        "overlap_independent_asm": agreement_independent["overlap"],
        "power_conservation_relative": power_error,
    }
    record["measured"] = {
        "centroid_x_m": centroid["centroid_x_m"],
        "centroid_y_m": centroid["centroid_y_m"],
        "fwhm_x_m": fwhm_x,
        "fwhm_y_m": fwhm_y,
        "first_sidelobe_ratio": sidelobe,
        "first_null_x_m": measurement.first_null_radius_m(intensity[row], pitch_m),
    }
    # Attribution is reported per axis. The two apertures differ (201 vs 121
    # samples), so the discrete-aperture-versus-continuous-sinc gap differs
    # between them; reporting only x would leave the larger y residual
    # unexplained. The headline components use the worse of the two axes.
    attribution_x = {
        "discretization_and_window": relative(on_grid["fwhm_x_m"], closed["fwhm_x_m"]),
        "paraxial_model": relative(independent_fwhm_x, closed["fwhm_x_m"]),
        "solver_implementation": relative(fwhm_x, independent_fwhm_x),
    }
    attribution_y = {
        "discretization_and_window": relative(on_grid["fwhm_y_m"], closed["fwhm_y_m"]),
        "paraxial_model": relative(independent_fwhm_y, closed["fwhm_y_m"]),
        "solver_implementation": relative(fwhm_y, independent_fwhm_y),
    }
    record["error_attribution"] = {
        **{key: max(attribution_x[key], attribution_y[key]) for key in attribution_x},
        "normalization": power_error,
        "convention": 1.0 - errors["overlap_independent_asm"],
        "per_axis": {"x": attribution_x, "y": attribution_y},
    }
    record["solver"] = _solver_block(result)
    record["closed_form"] = closed
    record["estimator_on_analytic_field"] = on_grid
    record["errors"] = errors
    record["gates"] = {
        "solver_succeeded": True,
        "signed_focal_position_x": (
            errors["centroid_x_input_pixels"] <= tolerances["case2_centroid_input_pixels"]
        ),
        "signed_focal_position_y": (
            errors["centroid_y_input_pixels"] <= tolerances["case2_centroid_input_pixels"]
        ),
        "diffraction_width_x": errors["fwhm_x_relative"] <= tolerances["case2_fwhm_relative"],
        "diffraction_width_y": errors["fwhm_y_relative"] <= tolerances["case2_fwhm_relative"],
        "sidelobe_shape": (
            errors["first_sidelobe_relative"] <= tolerances["case2_sidelobe_relative"]
        ),
        "complex_field_vs_fresnel": (
            errors["overlap_fresnel"] >= tolerances["case2_overlap_fresnel_minimum"]
        ),
        "complex_field_vs_independent_asm": (
            errors["overlap_independent_asm"] >= tolerances["case2_overlap_independent_asm_minimum"]
        ),
        "power_conserved": power_error <= tolerances["case2_power_conservation_relative"],
        "output_window_contains_field": (
            float(result.summary_metrics["output_edge_energy_fraction"])
            <= tolerances["case2_output_edge_energy_fraction_maximum"]
        ),
    }
    record["pass"] = all(record["gates"].values())
    record["_measured_field"] = measured
    record["_expected_field"] = expected
    record["_independent_field"] = independent
    record["_input_field"] = input_field
    record["_solver_summary"] = json.loads(Path(str(result.summary_path)).read_text())
    return record


# ---------------------------------------------------------------------------
# Case 3 -- high-NA vectorial focusing
# ---------------------------------------------------------------------------
def _focus_with_pupil_samples(geometry: dict[str, Any], pupil_samples: int) -> Any:
    """Run one high-NA focus with the aplanatic pupil filling the grid exactly.

    The pupil is made to span the full array because Chromatix's own
    ``zoom_factor`` uses ``field.shape[1] - 1`` -- it implicitly assumes the
    grid width *is* the pupil diameter. Letting the pupil underfill the grid
    would introduce a scale error that is a usage mistake rather than a solver
    property, so this configuration gives the implementation its best case.
    """
    pupil_radius_m = (
        geometry["focal_length_m"] * geometry["numerical_aperture"] / geometry["refractive_index"]
    )
    pupil_pitch_m = 2.0 * pupil_radius_m / pupil_samples
    amplitude, _ = oracles.aplanatic_pupil_amplitude(
        n=pupil_samples,
        pitch_m=pupil_pitch_m,
        focal_length_m=geometry["focal_length_m"],
        numerical_aperture=geometry["numerical_aperture"],
        refractive_index=geometry["refractive_index"],
    )
    return high_na_vector_focus(
        pupil_amplitude_x=amplitude,
        pupil_pitch_m=pupil_pitch_m,
        wavelength_m=geometry["wavelength_m"],
        refractive_index=geometry["refractive_index"],
        numerical_aperture=geometry["numerical_aperture"],
        focal_length_m=geometry["focal_length_m"],
        output_shape=(geometry["output_samples"], geometry["output_samples"]),
        output_pitch_m=geometry["output_pitch_m"],
    )


def evaluate_case3(reference: dict[str, Any], tolerances: dict[str, float]) -> dict[str, Any]:
    geometry = reference["geometry"]
    closed = reference["closed_form"]
    expected = reference["_expected_field"]
    output_pitch_m = geometry["output_pitch_m"]

    # --- the sampling qualification, run first: refine the pupil sampling with
    # every physical parameter fixed. A physical focal field cannot move.
    sweep: list[dict[str, Any]] = []
    for pupil_samples in geometry["pupil_sample_sweep"]:
        focus = _focus_with_pupil_samples(geometry, pupil_samples)
        intensity_z = np.abs(focus.field_xyz[..., 2]) ** 2
        ratios = measurement.vector_component_ratios(focus.field_xyz)
        n = focus.field_xyz.shape[0]
        row, column = np.unravel_index(int(np.argmax(intensity_z)), intensity_z.shape)
        sweep.append(
            {
                "pupil_samples": pupil_samples,
                "pupil_pitch_m": focus.pupil_pitch_m,
                "reported_output_pitch_m": focus.reported_pitch_m,
                "ez_peak_radius_m": float(np.hypot(row - n // 2, column - n // 2) * output_pitch_m),
                "iz_over_ix": ratios["iz_over_ix"],
                "iy_over_ix": ratios["iy_over_ix"],
            }
        )

    def relative_spread(values: list[float]) -> float:
        finite = [v for v in values if np.isfinite(v) and v != 0.0]
        if len(finite) < 2:
            return float("inf")
        return float((max(finite) - min(finite)) / abs(np.mean(finite)))

    scale_spread = relative_spread([entry["ez_peak_radius_m"] for entry in sweep])
    ratio_spread = relative_spread([entry["iz_over_ix"] for entry in sweep])
    scale_stable = scale_spread <= tolerances["case3_scale_stability_relative"]
    ratio_stable = ratio_spread <= tolerances["case3_ratio_stability_relative"]

    # --- the reference configuration, compared against the oracle
    focus = _focus_with_pupil_samples(geometry, geometry["reference_pupil_samples"])
    measured = focus.field_xyz
    agreement = measurement.complex_field_agreement(measured, expected)
    ratios = measurement.vector_component_ratios(measured)
    symmetry = measurement.vector_symmetry_diagnostics(measured)
    intensity_x = np.abs(measured[..., 0]) ** 2
    centre = measured.shape[0] // 2

    quadrature = reference["quadrature_convergence"]
    oracle_converged = (
        quadrature["tail_relative_spread"] <= tolerances["case3_oracle_quadrature_relative"]
    )

    gates = {
        "oracle_quadrature_converged": bool(oracle_converged),
        "solver_scale_stable_under_pupil_refinement": bool(scale_stable),
        "solver_component_ratios_stable_under_pupil_refinement": bool(ratio_stable),
        "vector_overlap": bool(agreement["overlap"] >= tolerances["case3_vector_overlap_minimum"]),
        "iz_over_ix": bool(
            abs(ratios["iz_over_ix"] / closed["iz_over_ix"] - 1.0)
            <= tolerances["case3_component_ratio_relative"]
        ),
        "ez_vanishes_on_axis": bool(
            symmetry["ez_on_axis_over_peak"] <= tolerances["case3_ez_on_axis_maximum"]
        ),
        "ez_x_antisymmetric": bool(
            symmetry["ez_x_antisymmetry_residual"] <= tolerances["case3_ez_antisymmetry_maximum"]
        ),
    }

    blocked = not (scale_stable and ratio_stable)
    return {
        "name": "case3_high_na_vector",
        "case": "case3",
        "role": "accuracy",
        "status": "blocked" if blocked else "complete",
        "geometry": geometry,
        "closed_form": closed,
        "measured": {
            **ratios,
            **symmetry,
            "fwhm_x_m": measurement.fwhm_m(intensity_x[centre], output_pitch_m),
            "fwhm_y_m": measurement.fwhm_m(intensity_x[:, centre], output_pitch_m),
            "reported_output_pitch_m": focus.reported_pitch_m,
            "requested_output_pitch_m": output_pitch_m,
        },
        "errors": {
            "vector_overlap": agreement["overlap"],
            "vector_normalized_rms": agreement["normalized_rms_error"],
            "iz_over_ix_relative": abs(ratios["iz_over_ix"] / closed["iz_over_ix"] - 1.0),
            "ez_on_axis_over_peak": symmetry["ez_on_axis_over_peak"],
            "ez_x_antisymmetry_residual": symmetry["ez_x_antisymmetry_residual"],
        },
        "oracle_quadrature_convergence": quadrature,
        "pupil_sampling_sweep": sweep,
        "sampling_qualification": {
            "ez_peak_radius_relative_spread": scale_spread,
            "iz_over_ix_relative_spread": ratio_spread,
            "scale_tolerance": tolerances["case3_scale_stability_relative"],
            "ratio_tolerance": tolerances["case3_ratio_stability_relative"],
            "scale_stable": bool(scale_stable),
            "ratios_stable": bool(ratio_stable),
        },
        "gates": gates,
        "pass": all(gates.values()),
        "blocked_reason": (
            None
            if not blocked
            else (
                "chromatix.functional.high_na_ff_lens does not produce a "
                "sampling-independent focal field. Refining only the pupil sampling, "
                "with wavelength, NA, focal length, output grid, and output pitch all "
                "fixed, moves the |E_z| ring radius by a relative "
                f"{scale_spread:.2f} and Iz/Ix by {ratio_spread:.2f}. The independent "
                "Richards-Wolf oracle converges to a relative "
                f"{quadrature['tail_relative_spread']:.1e} over the same comparison, so "
                "the non-convergence is in the solver, not the reference. Root cause "
                "read from the pinned source: high_na_ff_lens derives s_z from "
                "field.f_grid * lambda / n (the frequency grid) rather than from the "
                "pupil position grid, so on any physically sampled pupil |s_grid| << 1, "
                "s_z ~ 1, and both the intended 1/cos(theta) obliquity Jacobian and the "
                "exp(i k f cos theta) defocus degenerate to constants; the zoom_factor "
                "that sets the output scale is computed from the same quantity. Until "
                "that is fixed upstream, no absolute focal-field comparison against a "
                "Richards-Wolf oracle is meaningful, so this case is reported as "
                "blocked rather than failed."
            )
        ),
        "_measured_field": measured,
        "_expected_field": expected,
    }


# ---------------------------------------------------------------------------
# Perturbation analysis
# ---------------------------------------------------------------------------
def analyze_perturbations(
    records: list[dict[str, Any]], tolerances: dict[str, float], config: dict[str, Any]
) -> dict[str, Any]:
    margin = tolerances["perturbation_detection_margin"]
    by_name = {record["name"]: record for record in records}
    detected: dict[str, Any] = {}

    for record in records:
        if record["role"] != "perturbation":
            continue
        errors = record["errors"]
        if record["solver"].get("status") != "succeeded":
            observed, threshold, is_detected = None, None, True
            mechanism = "solver returned a structured failure instead of a field"
        elif record["case"] == "case1":
            observed = errors["phase_error_rad"]
            threshold = margin * record["derived_tolerances"]["phase_error_rad"]
            is_detected = observed > threshold
            mechanism = "phase-transfer error exceeded the detection threshold"
        else:
            # Complex overlap is the universal detector for Case 2: a lens sign
            # flip, an axis swap, and an SI-scale error all collapse it, while a
            # centroid or a width alone would miss at least one of them.
            observed = 1.0 - errors["overlap_fresnel"]
            threshold = margin * (1.0 - tolerances["case2_overlap_fresnel_minimum"])
            is_detected = observed > threshold
            mechanism = "1 - complex overlap against the analytic oracle exceeded threshold"

        detected[record["perturbation"]] = {
            "control_case": record["base_case"],
            "observed": observed,
            "detection_threshold": threshold,
            "detected": bool(is_detected),
            "mechanism": mechanism,
            "description": config["perturbations"][record["perturbation"]]["description"].strip(),
        }

    controls = {
        name: by_name[name]["pass"]
        for name in {record["base_case"] for record in records if record["role"] == "perturbation"}
        if name in by_name
    }
    return {
        "perturbations": detected,
        "controls_pass": controls,
        "pass": bool(
            detected
            and all(item["detected"] for item in detected.values())
            and all(controls.values())
        ),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot(
    path: Path,
    case1: list[dict[str, Any]],
    case2: list[dict[str, Any]],
    case3: dict[str, Any],
    perturbations: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    dashed = {"color": "black", "linestyle": "--", "linewidth": 1.0}

    # 1. Case 1: measured phase error against the derived float32 bound.
    accepted1 = [r for r in case1 if r["role"] == "accuracy" and r["errors"]]
    accumulated = [abs(r["closed_form"]["accumulated_phase_rad"]) for r in accepted1]
    axes[0][0].loglog(
        accumulated, [r["errors"]["phase_error_rad"] for r in accepted1], "o", label="measured"
    )
    axes[0][0].loglog(
        accumulated,
        [r["derived_tolerances"]["phase_error_rad"] for r in accepted1],
        "s",
        label="derived float32 bound",
    )
    axes[0][0].set(
        title="Case 1: exact plane-wave transfer\nphase error vs accumulated phase",
        xlabel="|k_z z| [rad]",
        ylabel="max wrapped phase error [rad]",
    )
    axes[0][0].grid(True, alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8)

    # 2. Case 2: signed focal shift, chromatix vs analytic.
    for record in [r for r in case2 if r["role"] == "accuracy"]:
        pitch_m = float(record["grid"]["pitch_m"])
        padded_n = int(record["grid"]["padded_n"])
        coordinates = measurement.grid_coordinates_m(padded_n, pitch_m)
        row = int(np.argmin(np.abs(coordinates - record["closed_form"]["centroid_y_m"])))
        intensity = np.abs(record["_measured_field"][row]) ** 2
        reference = np.abs(record["_expected_field"][row]) ** 2
        label = f"tilt_x={record['pupil']['tilt_x_rad']:+.1e}"
        line = axes[0][1].plot(coordinates * 1e6, intensity / intensity.max(), "-", label=label)[0]
        axes[0][1].plot(
            coordinates * 1e6, reference / reference.max(), "--", color=line.get_color(), alpha=0.6
        )
    axes[0][1].set(
        title="Case 2: signed focal position\nsolid Chromatix, dashed analytic sinc$^2$",
        xlabel="x [um]",
        ylabel="normalized intensity",
        xlim=(-40, 40),
    )
    axes[0][1].grid(True, alpha=0.3)
    axes[0][1].legend(fontsize=8)

    # 3. Case 2 error attribution.
    components = [
        "discretization_and_window",
        "paraxial_model",
        "solver_implementation",
        "normalization",
        "convention",
    ]
    accepted2 = [r for r in case2 if r["role"] == "accuracy"]
    positions = np.arange(len(accepted2))
    floor = 1e-12
    for index, component in enumerate(components):
        axes[0][2].bar(
            positions + (index - 2) * 0.16,
            [max(r["error_attribution"][component], floor) for r in accepted2],
            0.16,
            label=component.replace("_", " "),
        )
    axes[0][2].set_yscale("log")
    axes[0][2].set_xticks(positions)
    axes[0][2].set_xticklabels([f"{r['pupil']['tilt_x_rad']:+.0e}" for r in accepted2], fontsize=8)
    axes[0][2].set(
        title="Case 2 error attribution\n(paraxial model dominates, not the solver)",
        xlabel="tilt_x [rad]",
        ylabel="relative contribution",
        ylim=(floor, 1.0),
    )
    axes[0][2].grid(True, alpha=0.3, axis="y")
    axes[0][2].legend(fontsize=7)

    # 4. Case 3: oracle vector components.
    geometry = case3["geometry"]
    pitch_m = geometry["output_pitch_m"]
    expected = case3["_expected_field"]
    centre = expected.shape[0] // 2
    coordinates = measurement.grid_coordinates_m(expected.shape[0], pitch_m)
    peak = (np.abs(expected[..., 0]) ** 2).max()
    for index, label in enumerate(("|E_x|^2", "|E_y|^2", "|E_z|^2")):
        axes[1][0].semilogy(
            coordinates * 1e9, (np.abs(expected[centre, :, index]) ** 2) / peak, label=label
        )
    axes[1][0].set(
        title=f"Case 3 oracle: Richards-Wolf, NA={geometry['numerical_aperture']}",
        xlabel="x [nm]",
        ylabel="intensity / peak |E_x|^2",
        ylim=(1e-6, 2.0),
    )
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].legend(fontsize=8)

    # 5. Case 3 sampling qualification -- the non-convergence evidence.
    sweep = case3["pupil_sampling_sweep"]
    samples = [entry["pupil_samples"] for entry in sweep]
    axes[1][1].plot(
        samples, [entry["ez_peak_radius_m"] * 1e9 for entry in sweep], "o-", label="Chromatix"
    )
    axes[1][1].axhline(
        case3["closed_form"]["ez_peak_radius_m"] * 1e9,
        label="Richards-Wolf oracle",
        **dashed,
    )
    axes[1][1].set(
        title="Case 3 qualification: |E_z| ring radius\nvs pupil sampling (physics fixed)",
        xlabel="pupil samples N",
        ylabel="|E_z| peak radius [nm]",
    )
    axes[1][1].grid(True, alpha=0.3)
    axes[1][1].legend(fontsize=8)

    # 6. Perturbation detection.
    entries = perturbations["perturbations"]
    labels = list(entries)
    values = [
        entries[name]["observed"] if entries[name]["observed"] is not None else 1.0
        for name in labels
    ]
    thresholds = [
        entries[name]["detection_threshold"] if entries[name]["detection_threshold"] else 1e-12
        for name in labels
    ]
    axes[1][2].bar(range(len(labels)), values, color="tab:red", alpha=0.75, label="observed")
    axes[1][2].plot(range(len(labels)), thresholds, "k_", markersize=22, label="threshold")
    axes[1][2].set_yscale("log")
    axes[1][2].set_xticks(range(len(labels)))
    axes[1][2].set_xticklabels([n.replace("case", "c") for n in labels], rotation=20, fontsize=7)
    axes[1][2].set(title="Deliberate perturbations are rejected", ylabel="detector value")
    axes[1][2].grid(True, alpha=0.3, axis="y")
    axes[1][2].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or None
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


def _write_blocked(output_dir: Path, exc: Exception) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "blocked",
        "failure": {
            "code": "L1_WAVE_01_EXECUTION_FAILED",
            "message": str(exc),
            "stage": "benchmark_execution",
            "exception_type": type(exc).__name__,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("accuracy", "scaling"),
        default="accuracy",
        help="Evaluate the CHE-18 accuracy suite or an existing CHE-15 scaling bundle.",
    )
    parser.add_argument(
        "--case",
        choices=("gaussian", "all"),
        default="all",
        help=(
            "Both values run the complete CPU suite: the exact plane-wave primitive, "
            "signed paraxial focusing, high-NA vectorial focusing, and every "
            "qualification sweep and perturbation. 'gaussian' is retained as an alias "
            "so the CHE-18 verification command keeps working."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.section == "scaling":
        from scaling_evaluate import evaluate_scaling_bundle

        errors = evaluate_scaling_bundle(output_dir)
        print(json.dumps({"section": "scaling", "pass": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 2
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config()
        raw_tolerances = yaml.safe_load((SCRIPT_DIR / "tolerances.yaml").read_text())
        tolerances = {
            key: float(value) for key, value in raw_tolerances.items() if key != "schema_version"
        }
        reference = build_reference(config)
        adapter = ChromatixAdapter()
        scratch_context = tempfile.TemporaryDirectory(prefix="l1-wave-01-")
        scratch = Path(scratch_context.name)

        case1 = [
            evaluate_case1(adapter, record, tolerances, scratch) for record in reference["case1"]
        ]
        case2 = [
            evaluate_case2(adapter, record, tolerances, scratch) for record in reference["case2"]
        ]
        case3 = evaluate_case3(reference["case3"], tolerances)

        # --- perturbations (qualification only) -----------------------------
        case1_by_name = {record["name"]: record for record in reference["case1"]}
        case2_by_name = {record["name"]: record for record in reference["case2"]}
        # Use the largest-angle, longest-distance mode: the paraxial expansion is
        # least wrong at small angles, so a small-angle mode would understate the
        # detector's power.
        hardest_case1 = max(
            reference["case1"],
            key=lambda r: r["mode"]["sin_theta"] * r["planes"]["z_m"],
        )
        tilted_case2 = max(reference["case2"], key=lambda r: abs(float(r["pupil"]["tilt_x_rad"])))
        case1.append(
            evaluate_case1(
                adapter,
                case1_by_name[hardest_case1["name"]],
                tolerances,
                scratch,
                perturbation="case1_paraxial_dispersion",
            )
        )
        for perturbation in (
            "case2_lens_sign_flip",
            "case2_axis_transpose",
            "case2_si_scale",
        ):
            case2.append(
                evaluate_case2(
                    adapter,
                    case2_by_name[tilted_case2["name"]],
                    tolerances,
                    scratch,
                    perturbation=perturbation,
                )
            )

        perturbations = analyze_perturbations(case1 + case2, tolerances, config)

        gated = [r for r in case1 + case2 if r["role"] == "accuracy"]
        accuracy_pass = bool(gated and all(r["pass"] for r in gated) and perturbations["pass"])

        # --- performance ----------------------------------------------------
        def timed_suite() -> None:
            for index, record in enumerate(reference["case2"]):
                _propagate(
                    adapter,
                    field=record["_input_field"],
                    pitch_m=float(record["grid"]["pitch_m"]),
                    wavelength_m=float(record["physics"]["wavelength_m"]),
                    refractive_index=float(record["physics"]["refractive_index"]),
                    z_m=float(record["planes"]["z_m"]),
                    pad_width=int(record["grid"]["pad_width"]),
                    output_directory=scratch / f"timing_{index}",
                    reference_plane="timing repeat",
                )

        for _ in range(2):
            timed_suite()
        samples_seconds: list[float] = []
        for _ in range(7):
            start = time.perf_counter_ns()
            timed_suite()
            samples_seconds.append((time.perf_counter_ns() - start) * 1e-9)

        error_attribution = {
            "method": (
                "Each case reports the residual together with its cause, because a bare "
                "solver-versus-theory number cannot distinguish a wrong solver from an "
                "approximate oracle."
            ),
            "case1": (
                "No attribution is needed or offered: the discrete plane-wave transfer "
                "oracle is exact, so the only admissible error source is floating-point "
                "round-off, and the tolerance is derived from it rather than chosen."
            ),
            "case2_components": {
                "discretization_and_window": (
                    "The benchmark's own FWHM estimator applied to the analytic field on "
                    "the identical padded grid. Isolates estimator, grid centring, and the "
                    "discrete-aperture-versus-continuous-sinc gap."
                ),
                "paraxial_model": (
                    "An independent float64 angular-spectrum propagation compared against "
                    "the Fresnel oracle: the paraxial approximation's own error, measured."
                ),
                "solver_implementation": (
                    "Chromatix compared against that same independent float64 propagation. "
                    "Two exact calculations, so the residual is the solver plus complex64."
                ),
                "normalization": "Discrete windowed power ratio.",
                "convention": (
                    "1 - complex overlap against the independent implementation; collapses "
                    "on a phasor, axis, or scale error."
                ),
            },
            "case2_per_case": {
                r["name"]: r["error_attribution"] for r in case2 if r.get("error_attribution")
            },
            "case3": (
                "Attribution is not attempted because the prerequisite fails: the solver's "
                "focal field is not sampling-independent, so there is no converged quantity "
                "to attribute. See case3.blocked_reason."
            ),
            "oracle_boundary": {
                "independent_analytic_oracle": [
                    "exact discrete plane-wave eigenmode transfer (Case 1)",
                    "Fresnel/Fourier rectangular-pupil focal field (Case 2)",
                    "float64 Richards-Wolf quadrature (Case 3)",
                    "float64 NumPy angular spectrum (attribution only)",
                ],
                "solver_under_test": [
                    "chromatix.functional.asm_propagate via ChromatixAdapter.run_standalone",
                    "chromatix.functional.high_na_ff_lens via chromatix_benchmark_adapter",
                ],
                "never_used_as_oracle": [
                    "any Chromatix-generated value; the recorded propagation_probe snapshot"
                ],
            },
        }

        result = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "status": "complete",
            "accuracy": {
                "oracle": (
                    "exact discrete plane-wave transfer (Case 1); analytic Fresnel/Fourier "
                    "focal field of a rectangular pupil behind an ideal paraxial thin lens "
                    "(Case 2); independent float64 Richards-Wolf quadrature (Case 3)"
                ),
                "metrics": {
                    "case1_exact_primitive": [_public(r) for r in case1],
                    "case2_paraxial_focus": [_public(r) for r in case2],
                    "case3_high_na_vector": _public(case3),
                    "perturbations": perturbations,
                    "conventions": config["conventions"],
                    "error_attribution": error_attribution,
                },
                "tolerances": tolerances,
                "pass": accuracy_pass,
                "gated_cases": ["case1", "case2"],
                "blocked_cases": ["case3"] if case3["status"] == "blocked" else [],
                "blocked_case_note": (
                    "Case 3 is reported but does not gate this result. "
                    "chromatix.functional.high_na_ff_lens fails the sampling "
                    "qualification, so its focal field has no converged value to compare "
                    "against the Richards-Wolf oracle. See "
                    "accuracy.metrics.case3_high_na_vector.blocked_reason."
                )
                if case3["status"] == "blocked"
                else None,
            },
            "performance": {
                "warmup_runs": 2,
                "measured_repeats": 7,
                "samples_seconds": samples_seconds,
                "statistics": {
                    "median_seconds": float(np.median(samples_seconds)),
                    "minimum_seconds": float(np.min(samples_seconds)),
                    "p95_seconds": float(np.percentile(samples_seconds, 95)),
                },
                "peak_memory": {
                    "status": "measured",
                    "method": (
                        "resource.getrusage(RUSAGE_SELF).ru_maxrss; Linux KiB multiplied by 1024"
                    ),
                    "bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                },
            },
        }

        # --- artifacts -------------------------------------------------------
        arrays: dict[str, np.ndarray] = {}
        for record in case1 + case2:
            if "_measured_field" not in record:
                continue
            arrays[f"{record['name']}_output_field"] = record["_measured_field"].astype(
                np.complex64
            )
            if "_independent_field" in record:
                arrays[f"{record['name']}_independent_asm_field"] = record[
                    "_independent_field"
                ].astype(np.complex64)
            if record["case"] == "case2":
                profile = measurement.radial_intensity_profile(
                    np.abs(record["_measured_field"]) ** 2, float(record["grid"]["pitch_m"])
                )
                arrays[f"{record['name']}_radial_radius_m"] = profile["radius_m"]
                arrays[f"{record['name']}_radial_normalized_intensity"] = profile[
                    "normalized_intensity"
                ]
        arrays["case3_output_field_xyz"] = case3["_measured_field"].astype(np.complex64)

        expected_arrays: dict[str, np.ndarray] = {}
        for record in reference["case1"] + reference["case2"]:
            expected_arrays[f"{record['name']}_input_field"] = record["_input_field"].astype(
                np.complex64
            )
            expected_arrays[f"{record['name']}_expected_field"] = record["_expected_field"].astype(
                np.complex64
            )
        expected_arrays["case3_expected_field_xyz"] = reference["case3"]["_expected_field"].astype(
            np.complex64
        )

        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        np.savez(output_dir / "arrays.npz", **arrays)
        np.savez(output_dir / "reference_fields.npz", **expected_arrays)
        (output_dir / "reference.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "benchmark_id": BENCHMARK_ID,
                    "conventions": config["conventions"],
                    "cases": serializable(reference),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (output_dir / "input_config.yaml").write_text(
            (SCRIPT_DIR / "public_config.yaml").read_text()
        )
        (output_dir / "tolerances.yaml").write_text((SCRIPT_DIR / "tolerances.yaml").read_text())
        (output_dir / "error_attribution.json").write_text(
            json.dumps(error_attribution, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "solver_summaries.json").write_text(
            json.dumps(
                {
                    record["name"]: record["_solver_summary"]
                    for record in case1 + case2
                    if "_solver_summary" in record
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (output_dir / "README.md").write_text((SCRIPT_DIR / "README.md").read_text())
        _plot(output_dir / "plot.png", case1, case2, case3, perturbations)
        scratch_context.cleanup()

        hashed_files = [
            "result.json",
            "arrays.npz",
            "plot.png",
            "tolerances.yaml",
            "README.md",
            "reference.json",
            "reference_fields.npz",
            "input_config.yaml",
            "error_attribution.json",
            "solver_summaries.json",
        ]
        git_commit, dirty_worktree = _git_state()
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        loaded_forbidden = sorted(
            name
            for name in sys.modules
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
        )
        import jax

        provenance = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "command": [
                "./run.sh",
                "python",
                "benchmarks/level1/L1-WAVE-01/evaluate.py",
                "--case",
                args.case,
                "--output-dir",
                str(args.output_dir),
            ],
            "git_commit": git_commit,
            "dirty_worktree": dirty_worktree,
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
            "dtype": "complex64",
            "seed": int(config["seed"]),
            "seed_semantics": "recorded; every field is analytic and no RNG is used",
            "engine_versions": {
                "chromatix": importlib.metadata.version("chromatix"),
                "chromatix_commit": case1[0]["solver"].get("package_commit"),
                "jax": jax.__version__,
                "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
            },
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "input_parameters": config,
            "artifact_hashes": {name: _sha256(output_dir / name) for name in hashed_files},
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "timing_note": (
                "samples_seconds covers the three Case 2 focusing runs end to end through "
                "ChromatixAdapter.run_standalone, including writing each run's complex "
                "field artifacts to disk. JAX compilation is excluded by the two warmups."
            ),
            "forbidden_modules_loaded": loaded_forbidden,
        }
        if loaded_forbidden:
            raise RuntimeError(f"Wave-only benchmark loaded forbidden modules: {loaded_forbidden}")
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )

        summary = {
            "status": result["status"],
            "pass": accuracy_pass,
            "output_directory": str(output_dir),
            "case1_exact_primitive": {
                r["name"]: r["pass"] for r in case1 if r["role"] == "accuracy"
            },
            "case2_paraxial_focus": {
                r["name"]: r["pass"] for r in case2 if r["role"] == "accuracy"
            },
            "case3_high_na_vector": {
                "status": case3["status"],
                "oracle_quadrature_converged": case3["gates"]["oracle_quadrature_converged"],
                "solver_scale_stable": case3["sampling_qualification"]["scale_stable"],
                "ez_peak_radius_relative_spread": case3["sampling_qualification"][
                    "ez_peak_radius_relative_spread"
                ],
            },
            "perturbations_detected": {
                name: item["detected"] for name, item in perturbations["perturbations"].items()
            },
            "median_seconds": result["performance"]["statistics"]["median_seconds"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if case3["status"] == "blocked":
            print(
                "\nNOTE: Case 3 is BLOCKED and does not gate this result.\n"
                + str(case3["blocked_reason"]),
                file=sys.stderr,
            )
        return 0 if accuracy_pass else 2
    except Exception as exc:
        return _write_blocked(output_dir, exc)


if __name__ == "__main__":
    raise SystemExit(main())
