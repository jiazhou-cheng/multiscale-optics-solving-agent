"""Generate the complete Optiland-only L1-RAY-01 benchmark bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiscale_optics_agent.adapters.optiland_benchmark_adapter import (
    Edmund45362Prescription,
    OptilandTrace,
    trace_edmund_45362,
    trace_free_space,
    trace_paraxial_thin_lens,
)

MM_TO_M = 1e-3
UM_TO_M = 1e-6
PROTOCOL_ID = "M1-BASELINE-CPU-V2"
BENCHMARK_ID = "L1-RAY-01"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def _absolute_mm_error_to_m(actual_mm: float, expected_mm: float) -> float:
    """Convert an absolute native-mm residual to metres."""
    return abs(actual_mm - expected_mm) * MM_TO_M


def _spherical_sag_mm(rho_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    """Independent positive-radius spherical sag at radial coordinate rho."""
    rho = np.asarray(rho_mm, dtype=np.float64)
    return (rho**2 / radius_mm) / (1.0 + np.sqrt(1.0 - (rho / radius_mm) ** 2))


def _catalog_geometry(
    prescription: Edmund45362Prescription,
    *,
    sample_count: int = 401,
) -> dict[str, np.ndarray]:
    """Return the SI geometry used by both saved evidence and rendering."""
    height_mm = np.linspace(
        -prescription.clear_aperture_radius_mm,
        prescription.clear_aperture_radius_mm,
        sample_count,
        dtype=np.float64,
    )
    front_z_mm = prescription.front_vertex_z_mm + _spherical_sag_mm(
        np.abs(height_mm), prescription.radius_1_mm
    )
    return {
        "catalog_front_surface_height_m": height_mm * MM_TO_M,
        "catalog_front_surface_z_m": front_z_mm * MM_TO_M,
        "catalog_rear_surface_height_m": height_mm * MM_TO_M,
        "catalog_rear_surface_z_m": np.full_like(
            height_mm, prescription.back_vertex_z_mm * MM_TO_M
        ),
        "catalog_clear_aperture_m": np.array([prescription.clear_aperture_mm * MM_TO_M]),
        "catalog_image_plane_z_m": np.array([prescription.image_reference_plane_z_mm * MM_TO_M]),
    }


def _direction_norm_error(outputs: dict[str, np.ndarray]) -> float:
    norm = np.sqrt(outputs["L"] ** 2 + outputs["M"] ** 2 + outputs["N"] ** 2)
    return float(np.max(np.abs(norm - 1.0)))


def _all_finite(trace: OptilandTrace) -> bool:
    return all(np.all(np.isfinite(value)) for value in trace.outputs.values())


def _invalid_ray_count(trace: OptilandTrace) -> int:
    ray_count = trace.outputs["x_mm"].size
    invalid = np.zeros(ray_count, dtype=bool)
    for value in trace.outputs.values():
        array = np.asarray(value)
        if array.shape and array.shape[-1] == ray_count:
            invalid |= np.any(~np.isfinite(array), axis=tuple(range(array.ndim - 1)))
    return int(np.count_nonzero(invalid))


def _free_space_case(config: dict[str, Any], tolerances: dict[str, float]):
    slopes_x = np.array([-0.03, 0.0, 0.02], dtype=np.float64)
    slopes_y = np.array([0.01, -0.025, 0.015], dtype=np.float64)
    L = slopes_x
    M = slopes_y
    N = np.sqrt(1.0 - L**2 - M**2)
    x0 = np.array([-2.0, 0.25, 1.5], dtype=np.float64)
    y0 = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    z0 = np.zeros(3, dtype=np.float64)
    distance_mm = float(config["distance_mm"])
    trace = trace_free_space(
        x_mm=x0,
        y_mm=y0,
        z_mm=z0,
        L=L,
        M=M,
        N=N,
        distance_mm=distance_mm,
        wavelength_um=float(config["wavelength_um"]),
    )

    geometric_path_mm = (distance_mm - z0) / N
    expected = {
        "x_m": (x0 + geometric_path_mm * L) * MM_TO_M,
        "y_m": (y0 + geometric_path_mm * M) * MM_TO_M,
        "z_m": np.full_like(x0, distance_mm * MM_TO_M),
        "L": L,
        "M": M,
        "N": N,
        "geometric_path_m": geometric_path_mm * MM_TO_M,
        "opl_m": geometric_path_mm * MM_TO_M,
    }
    actual_position = np.stack(
        (
            trace.outputs["x_mm"] * MM_TO_M,
            trace.outputs["y_mm"] * MM_TO_M,
            trace.outputs["z_mm"] * MM_TO_M,
        )
    )
    expected_position = np.stack((expected["x_m"], expected["y_m"], expected["z_m"]))
    actual_direction = np.stack((trace.outputs["L"], trace.outputs["M"], trace.outputs["N"]))
    expected_direction = np.stack((L, M, N))
    position_error_m = float(np.max(np.abs(actual_position - expected_position)))
    direction_error = float(np.max(np.abs(actual_direction - expected_direction)))
    path_error_m = float(
        np.max(np.abs(trace.outputs["opd_native_mm"] * MM_TO_M - expected["opl_m"]))
    )
    position_tolerance_m = tolerances["free_space_position_scaled"] * max(
        1.0, distance_mm * MM_TO_M
    )
    path_tolerance_m = tolerances["free_space_path_scaled"] * max(1.0, distance_mm * MM_TO_M)
    norm_error = _direction_norm_error(trace.outputs)
    passed = bool(
        _all_finite(trace)
        and position_error_m <= position_tolerance_m
        and direction_error <= tolerances["free_space_direction_absolute"]
        and path_error_m <= path_tolerance_m
        and norm_error <= tolerances["direction_unit_norm_absolute"]
    )

    # Deliberately wrong conversion: treat mm as m. The same evaluator gates
    # must reject it by many orders of magnitude.
    wrong_position = np.stack((trace.outputs["x_mm"], trace.outputs["y_mm"], trace.outputs["z_mm"]))
    wrong_position_error_m = float(np.max(np.abs(wrong_position - expected_position)))
    perturbation_detected = wrong_position_error_m > position_tolerance_m

    metrics = {
        "finite": _all_finite(trace),
        "ray_count": int(trace.outputs["x_mm"].size),
        "max_position_error_m": position_error_m,
        "max_direction_error": direction_error,
        "max_geometric_path_or_opl_error_m": path_error_m,
        "max_direction_norm_error": norm_error,
        "invalid_ray_count": _invalid_ray_count(trace),
        "vignetted_ray_count": int(np.count_nonzero(trace.outputs["intensity"] <= 0)),
        "pass": passed,
    }
    negative = {
        "perturbation": "incorrect native geometry scale 1.0 m/mm instead of 1e-3 m/mm",
        "wrong_scale_position_error_m": wrong_position_error_m,
        "tolerance_m": position_tolerance_m,
        "detected": bool(perturbation_detected),
    }
    return trace, expected, metrics, negative


def _paraxial_case(config: dict[str, Any], tolerances: dict[str, float]):
    focal_length_mm = float(config["focal_length_mm"])
    heights = np.asarray(config["pupil_heights_mm"], dtype=np.float64)
    slopes = np.asarray(config["launch_slopes_rad"], dtype=np.float64)
    trace = trace_paraxial_thin_lens(
        focal_length_mm=focal_length_mm,
        pupil_heights_mm=heights,
        launch_slopes_rad=slopes,
        wavelength_um=float(config["wavelength_um"]),
    )
    input_slope = np.tan(trace.inputs["launch_slope_rad"])
    expected_x_mm = focal_length_mm * input_slope
    expected_u_after = input_slope - trace.inputs["pupil_height_mm"] / focal_length_mm
    expected = {
        "x_m": expected_x_mm * MM_TO_M,
        "z_m": np.full_like(expected_x_mm, focal_length_mm * MM_TO_M),
        "u_after": expected_u_after,
    }
    intercept_error_m = float(np.max(np.abs(trace.outputs["x_mm"] - expected_x_mm)) * MM_TO_M)
    relative_intercept_error = intercept_error_m / (focal_length_mm * MM_TO_M)
    norm_error = _direction_norm_error(trace.outputs)
    centroids_m = {}
    spot_rms_m = {}
    opd_reference_m = np.empty_like(trace.outputs["opd_native_mm"])
    for slope in slopes:
        mask = trace.inputs["launch_slope_rad"] == slope
        x_m = trace.outputs["x_mm"][mask] * MM_TO_M
        weights = trace.outputs["intensity"][mask]
        centroid = float(np.average(x_m, weights=weights))
        centroids_m[f"{slope:+.5f}"] = centroid
        spot_rms_m[f"{slope:+.5f}"] = float(
            np.sqrt(np.average((x_m - centroid) ** 2, weights=weights))
        )
        chief = np.argmin(np.abs(trace.inputs["pupil_height_mm"][mask]))
        field_opd = trace.outputs["opd_native_mm"][mask]
        opd_reference_m[mask] = (field_opd - field_opd[chief]) * MM_TO_M
    passed = bool(
        _all_finite(trace)
        and relative_intercept_error
        <= tolerances["paraxial_focal_intercept_relative_to_focal_length"]
        and norm_error <= tolerances["direction_unit_norm_absolute"]
    )
    metrics = {
        "finite": _all_finite(trace),
        "ray_count": int(trace.outputs["x_mm"].size),
        "launch_slopes_rad": slopes.tolist(),
        "max_focal_intercept_error_m": intercept_error_m,
        "focal_intercept_relative_error": relative_intercept_error,
        "max_direction_norm_error": norm_error,
        "centroid_m_by_slope": centroids_m,
        "spot_rms_m_by_slope": spot_rms_m,
        "invalid_ray_count": _invalid_ray_count(trace),
        "vignetted_ray_count": int(np.count_nonzero(trace.outputs["intensity"] <= 0)),
        "opd_reference": (
            "within each field, Optiland accumulated opd minus the pupil-height-zero chief ray"
        ),
        "opd_sign": "ray minus chief; positive means larger accumulated optical path",
        "pass": passed,
    }
    trace.outputs["opd_ray_minus_chief_mm"] = opd_reference_m / MM_TO_M
    return trace, expected, metrics


def _schott_n_bk7(wavelength_um: float) -> float:
    # Independent SCHOTT N-BK7 dispersion coefficients, wavelength in um.
    b = (1.03961212, 0.231792344, 1.010469450)
    c = (0.00600069867, 0.0200179144, 103.5606530)
    wavelength_squared = wavelength_um**2
    n_squared = 1.0 + sum(
        coefficient * wavelength_squared / (wavelength_squared - pole)
        for coefficient, pole in zip(b, c, strict=True)
    )
    return math.sqrt(n_squared)


def _catalog_case(
    config: dict[str, Any],
    tolerances: dict[str, float],
    prescription: Edmund45362Prescription,
):
    heights = np.asarray(config["pupil_heights_mm"], dtype=np.float64)
    slopes = np.asarray(config["launch_slopes_rad"], dtype=np.float64)
    wavelength_um = float(config["wavelength_um"])
    trace = trace_edmund_45362(
        prescription=prescription,
        pupil_heights_mm=heights,
        launch_slopes_rad=slopes,
        object_plane_z_mm=float(config["object_plane_z_mm"]),
        wavelength_um=wavelength_um,
    )
    radius_mm = float(trace.metadata["radius_1_mm"])
    thickness_mm = float(trace.metadata["center_thickness_mm"])
    catalog_efl_mm = float(trace.metadata["catalog_efl_mm"])
    catalog_bfl_mm = float(trace.metadata["catalog_bfl_mm"])
    n_reference = _schott_n_bk7(wavelength_um)
    analytic_efl_mm = radius_mm / (n_reference - 1.0)
    analytic_bfl_mm = analytic_efl_mm - thickness_mm / n_reference
    efl_relative_error = abs(analytic_efl_mm - catalog_efl_mm) / catalog_efl_mm
    bfl_relative_error = abs(analytic_bfl_mm - catalog_bfl_mm) / catalog_bfl_mm
    refractive_index_error = abs(float(trace.metadata["optiland_refractive_index"]) - n_reference)

    axial_mask = (trace.inputs["launch_slope_rad"] == 0.0) & (
        trace.inputs["pupil_height_mm"] == 0.0
    )
    axial_index = int(np.flatnonzero(axial_mask)[0])
    object_distance_mm = -float(config["object_plane_z_mm"])
    expected_geometric_path_mm = object_distance_mm + thickness_mm + catalog_bfl_mm
    expected_opl_mm = object_distance_mm + n_reference * thickness_mm + catalog_bfl_mm
    geometric_path_error_m = _absolute_mm_error_to_m(
        float(trace.outputs["geometric_path_mm"][axial_index]), expected_geometric_path_mm
    )
    opl_error_m = _absolute_mm_error_to_m(
        float(trace.outputs["opd_native_mm"][axial_index]), expected_opl_mm
    )
    scaled_opl_tolerance_m = tolerances["catalog_central_opl_scaled"] * max(
        1.0, expected_opl_mm * MM_TO_M
    )

    centroids_m = {}
    spot_rms_m = {}
    expected_centroids_m = {}
    opd_reference_m = np.empty_like(trace.outputs["opd_native_mm"])
    max_centroid_model_relative_error = 0.0
    for slope in slopes:
        mask = trace.inputs["launch_slope_rad"] == slope
        x_m = trace.outputs["x_mm"][mask] * MM_TO_M
        weights = trace.outputs["intensity"][mask]
        centroid = float(np.average(x_m, weights=weights))
        expected_centroid = analytic_efl_mm * math.tan(float(slope)) * MM_TO_M
        key = f"{slope:+.5f}"
        centroids_m[key] = centroid
        expected_centroids_m[key] = expected_centroid
        spot_rms_m[key] = float(np.sqrt(np.average((x_m - centroid) ** 2, weights=weights)))
        max_centroid_model_relative_error = max(
            max_centroid_model_relative_error,
            abs(centroid - expected_centroid) / (catalog_efl_mm * MM_TO_M),
        )
        chief = np.argmin(np.abs(trace.inputs["pupil_height_mm"][mask]))
        field_opd = trace.outputs["opd_native_mm"][mask]
        opd_reference_m[mask] = (field_opd - field_opd[chief]) * MM_TO_M
    on_axis_centroid_m = abs(centroids_m["+0.00000"])
    antisymmetry_error_m = abs(centroids_m["-0.01000"] + centroids_m["+0.01000"])
    norm_error = _direction_norm_error(trace.outputs)
    trace.outputs["opd_ray_minus_chief_mm"] = opd_reference_m / MM_TO_M
    invalid_count = _invalid_ray_count(trace)
    vignetted_count = int(np.count_nonzero(trace.outputs["intensity"] <= 0))

    front_x_mm = trace.outputs["history_x_mm"][1]
    front_y_mm = trace.outputs["history_y_mm"][1]
    front_z_mm = trace.outputs["history_z_mm"][1]
    front_rho_mm = np.sqrt(front_x_mm**2 + front_y_mm**2)
    expected_front_z_mm = prescription.front_vertex_z_mm + _spherical_sag_mm(
        front_rho_mm, prescription.radius_1_mm
    )
    rear_z_mm = trace.outputs["history_z_mm"][2]
    expected_rear_z_mm = np.full_like(rear_z_mm, prescription.back_vertex_z_mm)
    front_sag_error_m = float(np.max(np.abs(front_z_mm - expected_front_z_mm)) * MM_TO_M)
    rear_plane_error_m = float(np.max(np.abs(rear_z_mm - expected_rear_z_mm)) * MM_TO_M)
    surface_shape_pass = bool(
        front_sag_error_m <= tolerances["catalog_front_surface_sag_absolute_m"]
        and rear_plane_error_m <= tolerances["catalog_rear_surface_plane_absolute_m"]
    )

    aperture_heights = np.asarray(config["aperture_test_heights_mm"], dtype=np.float64)
    aperture_trace = trace_edmund_45362(
        prescription=prescription,
        pupil_heights_mm=aperture_heights,
        launch_slopes_rad=np.array([0.0], dtype=np.float64),
        object_plane_z_mm=float(config["object_plane_z_mm"]),
        wavelength_um=wavelength_um,
    )
    aperture_vignetted = aperture_trace.outputs["intensity"] <= 0.0
    aperture_inside = (
        np.abs(aperture_trace.inputs["pupil_height_mm"]) < prescription.clear_aperture_radius_mm
    )
    aperture_outside = ~aperture_inside
    inside_transmitted_count = int(np.count_nonzero(aperture_inside & ~aperture_vignetted))
    inside_vignetted_count = int(np.count_nonzero(aperture_inside & aperture_vignetted))
    outside_transmitted_count = int(np.count_nonzero(aperture_outside & ~aperture_vignetted))
    outside_vignetted_count = int(np.count_nonzero(aperture_outside & aperture_vignetted))
    aperture_classification_pass = bool(
        inside_transmitted_count == int(np.count_nonzero(aperture_inside))
        and inside_vignetted_count == 0
        and outside_transmitted_count == 0
        and outside_vignetted_count == int(np.count_nonzero(aperture_outside))
    )
    passed = bool(
        _all_finite(trace)
        and norm_error <= tolerances["direction_unit_norm_absolute"]
        and refractive_index_error <= tolerances["catalog_refractive_index_absolute"]
        and efl_relative_error <= tolerances["catalog_efl_relative"]
        and bfl_relative_error <= tolerances["catalog_bfl_relative"]
        and geometric_path_error_m <= scaled_opl_tolerance_m
        and opl_error_m <= scaled_opl_tolerance_m
        and on_axis_centroid_m <= tolerances["catalog_on_axis_centroid_absolute_m"]
        and max_centroid_model_relative_error
        <= tolerances["catalog_off_axis_paraxial_centroid_relative"]
        and antisymmetry_error_m <= tolerances["catalog_field_antisymmetry_absolute_m"]
        and invalid_count == 0
        and surface_shape_pass
        and aperture_classification_pass
    )
    expected = {
        "schott_refractive_index": np.array([n_reference]),
        "analytic_efl_m": np.array([analytic_efl_mm * MM_TO_M]),
        "analytic_bfl_m": np.array([analytic_bfl_mm * MM_TO_M]),
        "axial_geometric_path_m": np.array([expected_geometric_path_mm * MM_TO_M]),
        "axial_opl_m": np.array([expected_opl_mm * MM_TO_M]),
        "paraxial_centroid_m_by_field": np.array(
            [expected_centroids_m[f"{slope:+.5f}"] for slope in slopes]
        ),
        "front_surface_z_m": expected_front_z_mm * MM_TO_M,
        "rear_surface_z_m": expected_rear_z_mm * MM_TO_M,
        "aperture_vignetted": aperture_outside,
    }
    metrics = {
        "finite": _all_finite(trace),
        "ray_count": int(trace.outputs["x_mm"].size),
        "launch_slopes_rad": slopes.tolist(),
        "schott_refractive_index": n_reference,
        "optiland_refractive_index": float(trace.metadata["optiland_refractive_index"]),
        "refractive_index_absolute_error": refractive_index_error,
        "analytic_efl_m": analytic_efl_mm * MM_TO_M,
        "catalog_efl_m": catalog_efl_mm * MM_TO_M,
        "efl_relative_error": efl_relative_error,
        "analytic_bfl_m": analytic_bfl_mm * MM_TO_M,
        "catalog_bfl_m": catalog_bfl_mm * MM_TO_M,
        "bfl_relative_error": bfl_relative_error,
        "axial_geometric_path_m": float(trace.outputs["geometric_path_mm"][axial_index]) * MM_TO_M,
        "axial_geometric_path_error_m": geometric_path_error_m,
        "axial_opl_m": float(trace.outputs["opd_native_mm"][axial_index]) * MM_TO_M,
        "axial_opl_error_m": opl_error_m,
        "max_direction_norm_error": norm_error,
        "centroid_m_by_slope": centroids_m,
        "paraxial_expected_centroid_m_by_slope": expected_centroids_m,
        "max_centroid_model_relative_error": max_centroid_model_relative_error,
        "spot_rms_m_by_slope": spot_rms_m,
        "on_axis_centroid_absolute_m": on_axis_centroid_m,
        "field_antisymmetry_error_m": antisymmetry_error_m,
        "invalid_ray_count": invalid_count,
        "vignetted_ray_count": vignetted_count,
        "front_surface_max_sag_error_m": front_sag_error_m,
        "rear_surface_max_plane_error_m": rear_plane_error_m,
        "front_surface_sample_count": int(front_z_mm.size),
        "surface_shape_pass": surface_shape_pass,
        "clear_aperture_m": prescription.clear_aperture_mm * MM_TO_M,
        "inside_aperture_transmitted_count": inside_transmitted_count,
        "inside_aperture_vignetted_count": inside_vignetted_count,
        "outside_aperture_transmitted_count": outside_transmitted_count,
        "outside_aperture_vignetted_count": outside_vignetted_count,
        "aperture_classification_pass": aperture_classification_pass,
        "opd_reference": (
            "within each field, accumulated Optiland opd minus the pupil-height-zero chief ray"
        ),
        "opd_sign": "ray minus chief; positive means larger accumulated optical path",
        "pass": passed,
    }
    return trace, expected, metrics, aperture_trace


def _run_cases(
    config: dict[str, Any],
    tolerances: dict[str, float],
    prescription: Edmund45362Prescription,
):
    free = _free_space_case(config["free_space"], tolerances)
    paraxial = _paraxial_case(config["paraxial_thin_lens"], tolerances)
    catalog = _catalog_case(config["catalog_lens"], tolerances, prescription)
    return free, paraxial, catalog


def _si_arrays(prefix: str, trace: OptilandTrace) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in trace.outputs.items():
        if name.endswith("_mm"):
            arrays[f"{prefix}_{name.removesuffix('_mm')}_m"] = value * MM_TO_M
        elif name.endswith("_um"):
            arrays[f"{prefix}_{name.removesuffix('_um')}_m"] = value * UM_TO_M
        else:
            arrays[f"{prefix}_{name}"] = value
    return arrays


def _si_inputs(prefix: str, trace: OptilandTrace) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in trace.inputs.items():
        if name.endswith("_mm"):
            arrays[f"{prefix}_{name.removesuffix('_mm')}_m"] = value * MM_TO_M
        else:
            arrays[f"{prefix}_{name}"] = value
    return arrays


def _prefixed(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{name}": np.asarray(value) for name, value in arrays.items()}


def _plot_free_space_layout(axis: Any, free: OptilandTrace) -> None:
    for index in range(free.inputs["x_mm"].size):
        axis.plot(
            [free.inputs["z_mm"][index], free.outputs["z_mm"][index]],
            [free.inputs["x_mm"][index], free.outputs["x_mm"][index]],
            marker="o",
        )
    axis.axvline(0.0, color="0.25", linestyle=":", label="input plane")
    axis.axvline(
        float(free.metadata["reference_plane_z_mm"]),
        color="0.25",
        linestyle="--",
        label="output reference plane",
    )
    axis.set(xlabel="axial z [mm]", ylabel="transverse x [mm]")
    axis.grid(True, alpha=0.25)


def _plot_paraxial_layout(axis: Any, paraxial: OptilandTrace) -> None:
    focal_length_mm = float(paraxial.metadata["focal_length_mm"])
    incoming_z_mm = -0.4 * focal_length_mm
    slopes = np.unique(paraxial.inputs["launch_slope_rad"])
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, slopes.size))
    for color, slope in zip(colors, slopes, strict=True):
        mask = paraxial.inputs["launch_slope_rad"] == slope
        for ray_index, column in enumerate(np.flatnonzero(mask)):
            height = paraxial.inputs["pupil_height_mm"][column]
            incoming_x = height + incoming_z_mm * math.tan(float(slope))
            axis.plot(
                [incoming_z_mm, 0.0],
                [incoming_x, height],
                color=color,
                alpha=0.65,
                label=f"slope {slope:+.3f} rad" if ray_index == 0 else None,
            )
            axis.plot(
                [0.0, paraxial.outputs["z_mm"][column]],
                [height, paraxial.outputs["x_mm"][column]],
                color=color,
                alpha=0.65,
            )
    aperture = float(paraxial.metadata["aperture_radius_mm"])
    axis.plot(
        [0.0, 0.0],
        [-1.25 * aperture, 1.25 * aperture],
        color="#7b2cbf",
        linewidth=3,
        label="ideal thin-lens model plane",
    )
    axis.axvline(
        focal_length_mm,
        color="0.2",
        linestyle="--",
        label="focal/image plane",
    )
    axis.set(xlabel="axial z [mm]", ylabel="transverse x [mm]")
    axis.grid(True, alpha=0.25)


def _plot_catalog_geometry(
    axis: Any,
    geometry: dict[str, np.ndarray],
    prescription: Edmund45362Prescription,
    *,
    show_image_plane: bool = True,
) -> None:
    height_mm = geometry["catalog_front_surface_height_m"] / MM_TO_M
    front_z_mm = geometry["catalog_front_surface_z_m"] / MM_TO_M
    rear_z_mm = geometry["catalog_rear_surface_z_m"] / MM_TO_M
    axis.fill_betweenx(
        height_mm,
        front_z_mm,
        rear_z_mm,
        color="#80deea",
        alpha=0.35,
        label=f"{prescription.material} glass",
        zorder=1,
    )
    axis.plot(front_z_mm, height_mm, color="#006064", linewidth=2, label="spherical front")
    axis.plot(rear_z_mm, height_mm, color="#006064", linewidth=2, label="planar rear")
    for edge_index, edge in enumerate((0, -1)):
        axis.plot(
            [front_z_mm[edge], rear_z_mm[edge]],
            [height_mm[edge], height_mm[edge]],
            color="#006064",
            linewidth=2,
            label=(
                f"{prescription.clear_aperture_mm:g} mm clear-aperture edge"
                if edge_index == 0
                else None
            ),
        )
    if show_image_plane:
        axis.axvline(
            float(geometry["catalog_image_plane_z_m"][0] / MM_TO_M),
            color="#6a1b9a",
            linestyle="--",
            linewidth=1.5,
            label="BFL image/reference plane",
        )


def _plot_catalog_rays(axis: Any, catalog: OptilandTrace, *, alpha: float = 0.6) -> None:
    slopes = np.unique(catalog.inputs["launch_slope_rad"])
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, slopes.size))
    for color, slope in zip(colors, slopes, strict=True):
        mask = catalog.inputs["launch_slope_rad"] == slope
        for ray_index, column in enumerate(np.flatnonzero(mask)):
            axis.plot(
                catalog.outputs["history_z_mm"][:, column],
                catalog.outputs["history_x_mm"][:, column],
                color=color,
                alpha=alpha,
                label=f"rays {slope:+.3f} rad" if ray_index == 0 else None,
                zorder=2,
            )


def _plot_overview(
    path: Path,
    free: OptilandTrace,
    paraxial: OptilandTrace,
    catalog: OptilandTrace,
    geometry: dict[str, np.ndarray],
    prescription: Edmund45362Prescription,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    _plot_free_space_layout(axes[0], free)
    axes[0].set_title("1. Homogeneous Free-Space Propagation")
    axes[0].legend(fontsize=7, loc="best")

    _plot_paraxial_layout(axes[1], paraxial)
    axes[1].set_title("2. Ideal Paraxial Thin Lens")
    axes[1].legend(fontsize=7, loc="best")

    _plot_catalog_geometry(axes[2], geometry, prescription)
    _plot_catalog_rays(axes[2], catalog)
    axes[2].set(
        title="3. Edmund #45-362 Plano-Convex Lens",
        xlabel="axial z [mm]",
        ylabel="transverse x [mm]",
    )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=7, loc="best")
    fig.suptitle("L1-RAY-01 — three-case diagnostic overview (metrics are authoritative)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_free_space_diagnostics(
    path: Path,
    free: OptilandTrace,
    expected: dict[str, np.ndarray],
    tolerances: dict[str, float],
) -> None:
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    layout = fig.add_subplot(grid[:, 0])
    residual_length = fig.add_subplot(grid[0, 1])
    residual_direction = fig.add_subplot(grid[1, 1])
    _plot_free_space_layout(layout, free)
    layout.set_title("Exact straight-line ray layout")
    layout.legend(fontsize=8)

    actual_position_m = np.stack(
        [free.outputs[name] * MM_TO_M for name in ("x_mm", "y_mm", "z_mm")], axis=1
    )
    expected_position_m = np.stack([expected[name] for name in ("x_m", "y_m", "z_m")], axis=1)
    position_error_m = np.linalg.norm(actual_position_m - expected_position_m, axis=1)
    actual_direction = np.stack([free.outputs[name] for name in ("L", "M", "N")], axis=1)
    expected_direction = np.stack([expected[name] for name in ("L", "M", "N")], axis=1)
    direction_error = np.linalg.norm(actual_direction - expected_direction, axis=1)
    path_error_m = np.abs(free.outputs["opd_native_mm"] * MM_TO_M - expected["opl_m"])
    ray_index = np.arange(position_error_m.size)
    residual_length.plot(ray_index, position_error_m, "o-", label="position")
    residual_length.plot(ray_index, path_error_m, "s-", label="path / OPL")
    residual_length.axhline(
        tolerances["free_space_position_scaled"],
        color="tab:blue",
        linestyle=":",
        label="position tol.",
    )
    residual_length.axhline(
        tolerances["free_space_path_scaled"],
        color="tab:orange",
        linestyle=":",
        label="path tol.",
    )
    residual_length.set(
        title="Per-ray length residuals",
        xlabel="ray index",
        ylabel="absolute error [m]",
    )
    residual_length.set_ylim(
        -0.05 * tolerances["free_space_position_scaled"],
        1.15
        * max(
            tolerances["free_space_position_scaled"],
            tolerances["free_space_path_scaled"],
        ),
    )
    residual_length.grid(True, alpha=0.25)
    residual_length.legend(fontsize=7)

    residual_direction.plot(ray_index, direction_error, "o-", color="tab:green")
    residual_direction.axhline(
        tolerances["free_space_direction_absolute"], color="0.25", linestyle=":", label="tolerance"
    )
    residual_direction.set(
        title="Per-ray direction residual", xlabel="ray index", ylabel="direction-cosine error [1]"
    )
    residual_direction.set_ylim(
        -0.05 * tolerances["free_space_direction_absolute"],
        1.15 * tolerances["free_space_direction_absolute"],
    )
    residual_direction.grid(True, alpha=0.25)
    residual_direction.legend(fontsize=8)
    fig.suptitle("Free-space diagnostics — Optiland versus closed-form oracle")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_paraxial_diagnostics(
    path: Path,
    paraxial: OptilandTrace,
    expected: dict[str, np.ndarray],
    tolerances: dict[str, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _plot_paraxial_layout(axes[0], paraxial)
    axes[0].set_title("Ideal paraxial ray layout")
    axes[0].legend(fontsize=7)
    slopes = np.unique(paraxial.inputs["launch_slope_rad"])
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, slopes.size))
    residual_mm = (paraxial.outputs["x_mm"] * MM_TO_M - expected["x_m"]) / MM_TO_M
    tolerance_mm = tolerances["paraxial_focal_intercept_relative_to_focal_length"] * float(
        paraxial.metadata["focal_length_mm"]
    )
    for color, slope in zip(colors, slopes, strict=True):
        mask = paraxial.inputs["launch_slope_rad"] == slope
        pupil = paraxial.inputs["pupil_height_mm"][mask]
        axes[1].plot(
            pupil,
            paraxial.outputs["x_mm"][mask],
            "o",
            color=color,
            label=f"Optiland {slope:+.3f} rad",
        )
        axes[1].plot(pupil, expected["x_m"][mask] / MM_TO_M, "--", color=color)
        axes[2].plot(pupil, residual_mm[mask], "o-", color=color, label=f"{slope:+.3f} rad")
    axes[1].set(
        title="Focal intercept: markers vs ABCD lines",
        xlabel="input pupil height [mm]",
        ylabel="image-plane x [mm]",
    )
    axes[2].axhline(tolerance_mm, color="0.25", linestyle=":", label="± tolerance")
    axes[2].axhline(-tolerance_mm, color="0.25", linestyle=":")
    axes[2].set(
        title="Optiland - analytic focal intercept",
        xlabel="input pupil height [mm]",
        ylabel="residual [mm]",
    )
    for axis in axes[1:]:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Paraxial diagnostics — field groups converge independent of pupil height")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_catalog_diagnostics(
    path: Path,
    catalog: OptilandTrace,
    geometry: dict[str, np.ndarray],
    prescription: Edmund45362Prescription,
    metrics: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_catalog_geometry(axes[0, 0], geometry, prescription)
    _plot_catalog_rays(axes[0, 0], catalog)
    axes[0, 0].set(
        title="Full ray layout",
        xlabel="axial z [mm]",
        ylabel="transverse x [mm]",
    )
    axes[0, 0].legend(fontsize=7)

    _plot_catalog_geometry(axes[0, 1], geometry, prescription, show_image_plane=False)
    _plot_catalog_rays(axes[0, 1], catalog, alpha=0.45)
    axes[0, 1].scatter(
        catalog.outputs["history_z_mm"][1:3].ravel(),
        catalog.outputs["history_x_mm"][1:3].ravel(),
        s=8,
        color="#d84315",
        label="ray/surface intersections",
        zorder=3,
    )
    axes[0, 1].set_xlim(-0.35, prescription.back_vertex_z_mm + 0.35)
    axes[0, 1].set_ylim(
        -1.08 * prescription.clear_aperture_radius_mm,
        1.08 * prescription.clear_aperture_radius_mm,
    )
    axes[0, 1].set(
        title="Lens close-up: spherical front + planar rear",
        xlabel="axial z [mm]",
        ylabel="transverse x [mm]",
    )
    axes[0, 1].legend(fontsize=7)

    slopes = np.unique(catalog.inputs["launch_slope_rad"])
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, slopes.size))
    for color, slope in zip(colors, slopes, strict=True):
        key = f"{slope:+.5f}"
        mask = catalog.inputs["launch_slope_rad"] == slope
        pupil = catalog.inputs["pupil_height_mm"][mask]
        axes[1, 0].plot(
            pupil,
            catalog.outputs["x_mm"][mask],
            "o-",
            color=color,
            label=f"Optiland {slope:+.3f} rad",
        )
        axes[1, 0].axhline(
            metrics["paraxial_expected_centroid_m_by_slope"][key] / MM_TO_M,
            color=color,
            linestyle="--",
        )
        axes[1, 1].plot(
            pupil,
            catalog.outputs["opd_ray_minus_chief_mm"][mask] * 1e3,
            "o-",
            color=color,
            label=f"{slope:+.3f} rad",
        )
    axes[1, 0].set(
        title="Image intercept (solid) vs paraxial centroid (dashed)",
        xlabel="input pupil height [mm]",
        ylabel="image-plane x [mm]",
    )
    axes[1, 1].axhline(0.0, color="0.2", linestyle=":", label="chief-ray zero")
    axes[1, 1].set(
        title="Ray-minus-chief OPD",
        xlabel="input pupil height [mm]",
        ylabel="OPD(ray) - OPD(chief) [µm]",
    )
    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    fig.suptitle("Edmund #45-362 spherical plano-convex catalog-lens diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_bundle(
    output_dir: Path,
    free: OptilandTrace,
    free_expected: dict[str, np.ndarray],
    paraxial: OptilandTrace,
    paraxial_expected: dict[str, np.ndarray],
    catalog: OptilandTrace,
    catalog_metrics: dict[str, Any],
    geometry: dict[str, np.ndarray],
    prescription: Edmund45362Prescription,
    tolerances: dict[str, float],
) -> None:
    _plot_overview(output_dir / "plot.png", free, paraxial, catalog, geometry, prescription)
    _plot_free_space_diagnostics(
        output_dir / "free_space_diagnostics.png", free, free_expected, tolerances
    )
    _plot_paraxial_diagnostics(
        output_dir / "paraxial_diagnostics.png", paraxial, paraxial_expected, tolerances
    )
    _plot_catalog_diagnostics(
        output_dir / "catalog_diagnostics.png",
        catalog,
        geometry,
        prescription,
        catalog_metrics,
    )


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
            "code": "L1_RAY_01_EXECUTION_FAILED",
            "message": str(exc),
            "stage": "benchmark_execution",
            "exception_type": type(exc).__name__,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = yaml.safe_load((SCRIPT_DIR / "public_config.yaml").read_text())
        prescription_data = yaml.safe_load((SCRIPT_DIR / "prescription.yaml").read_text())
        prescription = Edmund45362Prescription.from_mapping(prescription_data)
        tolerances = yaml.safe_load((SCRIPT_DIR / "tolerances.yaml").read_text())
        numeric_tolerances = {
            key: float(value) for key, value in tolerances.items() if key != "schema_version"
        }

        for _ in range(2):
            _run_cases(config, numeric_tolerances, prescription)
        samples_seconds = []
        for _ in range(7):
            start = time.perf_counter_ns()
            _run_cases(config, numeric_tolerances, prescription)
            samples_seconds.append((time.perf_counter_ns() - start) * 1e-9)
        free, paraxial, catalog = _run_cases(config, numeric_tolerances, prescription)
        free_trace, free_expected, free_metrics, negative = free
        paraxial_trace, paraxial_expected, paraxial_metrics = paraxial
        catalog_trace, catalog_expected, catalog_metrics, catalog_aperture_trace = catalog
        catalog_geometry = _catalog_geometry(prescription)

        accuracy_pass = bool(
            free_metrics["pass"]
            and paraxial_metrics["pass"]
            and catalog_metrics["pass"]
            and negative["detected"]
        )
        error_attribution = {
            "evidence_classification": {
                "independent_analytic_oracle": [
                    "free-space position, direction, geometric path, and OPL",
                    "paraxial thin-lens focal intercepts",
                    "SCHOTT N-BK7 refractive index and thick-lens EFL/BFL",
                    "catalog-lens axial chief-ray geometric path and OPL",
                    "catalog spherical-front sag and planar-rear intersection",
                    "catalog clear-aperture inside/outside classification",
                ],
                "manufacturer_reference": [
                    "Edmund #45-362 radii, thickness, material, clear aperture, EFL, and BFL"
                ],
                "deterministic_regression_only": [
                    "catalog marginal-ray coordinates/directions",
                    "catalog spot RMS by field",
                    "ray-minus-chief OPD arrays",
                ],
            },
            "solver_numerical": {
                "free_space": "closed-form position/direction/path residuals",
                "paraxial": "residual against the independently evaluated ideal ABCD map",
                "catalog": (
                    "direction normalization, material-index, axial path/OPL, symmetry, "
                    "surface-shape, and aperture-classification residuals"
                ),
            },
            "analytic_or_paraxial_model": {
                "free_space": "none beyond homogeneous-medium assumptions",
                "paraxial": (
                    "physical higher-order angular and aperture terms are excluded "
                    "by the ideal model"
                ),
                "catalog": (
                    "off-axis centroid prediction is paraxial; real spherical "
                    "aberration/coma is expected"
                ),
            },
            "prescription_reference": {
                "catalog": (
                    "Edmund focal length ±1%; center thickness ±0.10 mm; "
                    "SCHOTT reference dispersion"
                ),
            },
            "sampling": {
                "free_space": 3,
                "paraxial": int(paraxial_trace.outputs["x_mm"].size),
                "catalog": int(catalog_trace.outputs["x_mm"].size),
                "catalog_aperture_subtest": int(catalog_aperture_trace.outputs["x_mm"].size),
            },
            "aperture_vignetting": {
                "catalog_clear_aperture_mm": prescription.clear_aperture_mm,
                "catalog_vignetted_ray_count": catalog_metrics["vignetted_ray_count"],
                "dedicated_subtest": {
                    "inside_transmitted_count": catalog_metrics[
                        "inside_aperture_transmitted_count"
                    ],
                    "outside_vignetted_count": catalog_metrics["outside_aperture_vignetted_count"],
                    "pass": catalog_metrics["aperture_classification_pass"],
                },
                "note": (
                    "Optiland flags clipped rays by zero intensity; it does not "
                    "remove them in this path."
                ),
            },
        }
        conventions = {
            "coordinate_frame": "right-handed Cartesian, propagation along +z",
            "axes": "x and y transverse; z axial",
            "direction": "(L,M,N) normalized direction cosines; launch slope u=L/N",
            "native_geometry_unit": "mm",
            "geometry_to_si_scale_m": MM_TO_M,
            "native_wavelength_unit": "um",
            "wavelength_to_si_scale_m": UM_TO_M,
            "optical_path": (
                "Optiland RealRays.opd accumulation; evaluated in SI after 1e-3 conversion"
            ),
            "opd_reference": (
                "within each field, raw accumulated optical path minus pupil-height-zero chief ray"
            ),
            "opd_sign": "ray minus chief; positive means longer accumulated optical path",
            "polarization": "not modeled",
            "coherence": "not applicable to independent geometric rays",
            "normalization": "raw Optiland intensity/weight; centroids explicitly weight by it",
            "reference_planes": {
                "free_space": "z=0.1 m plane",
                "paraxial": "z=f=0.05 m plane",
                "catalog": (
                    f"z={prescription.image_reference_plane_z_mm * MM_TO_M:.5f} m, "
                    f"{prescription.catalog_bfl_mm * MM_TO_M:.5f} m after the plano rear vertex"
                ),
            },
        }
        result = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "protocol_amendment": (
                "V2 adds authoritative surface-shape and clear-aperture evidence while "
                "preserving the three V1 physical cases."
            ),
            "status": "complete",
            "accuracy": {
                "oracle": (
                    "independent free-space geometry, ABCD thin-lens equations, "
                    "SCHOTT dispersion/thick-lens equations, and Edmund Optics "
                    "catalog reference"
                ),
                "metrics": {
                    "free_space": free_metrics,
                    "paraxial_thin_lens": paraxial_metrics,
                    "catalog_lens": catalog_metrics,
                    "convention_negative_test": negative,
                    "conventions": conventions,
                    "error_attribution": error_attribution,
                },
                "tolerances": numeric_tolerances,
                "pass": accuracy_pass,
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

        arrays = {}
        arrays.update(_si_arrays("free_space", free_trace))
        arrays.update(_si_arrays("paraxial", paraxial_trace))
        arrays.update(_si_arrays("catalog", catalog_trace))
        arrays.update(_si_arrays("catalog_aperture", catalog_aperture_trace))
        arrays.update(catalog_geometry)
        input_arrays = {}
        input_arrays.update(_si_inputs("free_space", free_trace))
        input_arrays.update(_si_inputs("paraxial", paraxial_trace))
        input_arrays.update(_si_inputs("catalog", catalog_trace))
        input_arrays.update(_si_inputs("catalog_aperture", catalog_aperture_trace))
        expected_arrays = {}
        expected_arrays.update(_prefixed("free_space", free_expected))
        expected_arrays.update(_prefixed("paraxial", paraxial_expected))
        expected_arrays.update(_prefixed("catalog", catalog_expected))

        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        np.savez(output_dir / "arrays.npz", **arrays)
        np.savez(output_dir / "ray_inputs.npz", **input_arrays)
        np.savez(output_dir / "expected.npz", **expected_arrays)
        (output_dir / "input_config.yaml").write_text(
            (SCRIPT_DIR / "public_config.yaml").read_text()
        )
        (output_dir / "prescription.yaml").write_text(
            (SCRIPT_DIR / "prescription.yaml").read_text()
        )
        (output_dir / "tolerances.yaml").write_text((SCRIPT_DIR / "tolerances.yaml").read_text())
        (output_dir / "error_attribution.json").write_text(
            json.dumps(error_attribution, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "README.md").write_text((SCRIPT_DIR / "README.md").read_text())
        _plot_bundle(
            output_dir,
            free_trace,
            free_expected,
            paraxial_trace,
            paraxial_expected,
            catalog_trace,
            catalog_metrics,
            catalog_geometry,
            prescription,
            numeric_tolerances,
        )

        required_hash_files = [
            "result.json",
            "arrays.npz",
            "plot.png",
            "free_space_diagnostics.png",
            "paraxial_diagnostics.png",
            "catalog_diagnostics.png",
            "tolerances.yaml",
            "README.md",
            "ray_inputs.npz",
            "expected.npz",
            "input_config.yaml",
            "prescription.yaml",
            "error_attribution.json",
        ]
        git_commit, dirty_worktree = _git_state()
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        provenance = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "protocol_id": PROTOCOL_ID,
            "protocol_amendment": (
                "CHE-20: authoritative catalog surface-shape and aperture checks plus "
                "complete three-case visualization evidence"
            ),
            "command": [
                "./run.sh",
                "python",
                "benchmarks/level1/L1-RAY-01/run_benchmark.py",
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
            },
            "device": "cpu",
            "dtype": "float64",
            "seed": int(config["seed"]),
            "engine_versions": {
                "optiland": importlib.metadata.version("optiland"),
                "numpy": np.__version__,
            },
            "input_parameters": config,
            "catalog_prescription": prescription_data,
            "artifact_hashes": {
                filename: _sha256(output_dir / filename) for filename in required_hash_files
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
                "Ray-only benchmark loaded forbidden modules: "
                f"{provenance['forbidden_modules_loaded']}"
            )
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )

        summary = {
            "status": result["status"],
            "pass": accuracy_pass,
            "output_directory": str(output_dir),
            "case_pass": {
                "free_space": free_metrics["pass"],
                "paraxial_thin_lens": paraxial_metrics["pass"],
                "catalog_lens": catalog_metrics["pass"],
                "convention_negative_test": negative["detected"],
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if accuracy_pass else 2
    except Exception as exc:
        return _write_blocked(output_dir, exc)


if __name__ == "__main__":
    raise SystemExit(main())
