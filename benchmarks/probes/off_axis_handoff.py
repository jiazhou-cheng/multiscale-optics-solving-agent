"""CHE-41: verify the off-axis handoff against a purely geometric oracle.

CHE-37 (M3.8) found that the declared pupil OPL carried 0.13% of the tilt an
off-axis field requires, so the reconstructed wave converged 209 um from where the
rays actually go -- and it converged *cleanly*, 0.072 waves peak-to-valley against
its own fitted reference sphere, which is why nothing downstream noticed. CHE-41
declares the off-axis OPL reference to be the **incoming tilted wavefront** and
this probe checks that declaration against an oracle that owes the reconstruction
nothing:

    the reconstructed field's PSF must land at the TRACED CHIEF-RAY INTERSECTION,
    and the fitted reference sphere's centre must be there too.

Both are geometry. Neither is a tolerance, and neither can be satisfied by a
convention that is internally consistent but aimed at the wrong place -- which is
exactly what the superseded declaration was.

Three further things are measured here because the ticket requires them:

* the superseded declaration is re-run through the shipping code's own
  perturbation hook, so the 209 um error is reproduced rather than quoted;
* a geometric RMS spot radius about the chief-ray point, which depends on no
  reference-sphere convention at all, because M3.8's method note is that a
  sphere-centre error masquerades as aberration;
* the axis-transpose negative control, whose detection margin on every frozen
  configuration was 1.0000066 -- it could not be made to fail. A genuinely
  one-axis off-axis PSF is what makes it non-vacuous, and this probe reports the
  margin under CHE-37's own metric as well, because that metric turns out to be
  blind to a transpose for a second, independent reason.

Writes ``benchmarks/probes/records/m3_off_axis_handoff.json``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.provenance import RECORD_PROVENANCE_KEY, record_provenance

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml"
RECORD_PATH = Path(__file__).resolve().parent / "records" / "m3_off_axis_handoff.json"

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1e-6

#: Same hexapolar density M3.8 used, so every number here is comparable to the
#: record it supersedes. CHE-38 owns the convergence study.
NUM_RAYS = 32

#: M3-REVERSE-TELEPHOTO, frozen. The field is Hy = 0.2 for three independent
#: reasons: the frozen pitch still admits a tilt-carrying pupil there (94.6% of
#: the Nyquist limit -- see off_axis_opd_reference's field scan), the PSF is still
#: inside the unpadded observation window (209 um of 232 um), and it is the field
#: CHE-37 measured the defect at.
GEOMETRY = {
    "sample": "ReverseTelephoto",
    "pupil_z_m": 2.1547825721481666e-3,
    "image_z_m": 5.209361469999999e-3,
    "pitch_m": 1.8258157981959995e-06,
    "grid_n": 254,
    "pad_width": 762,
    "na_frozen": 0.07530880176185195,
    "hy": 0.2,
}


def _trace(directory: Path, *, hy: float):
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    result = get_adapter().run(
        ModelRunRequest(
            run_id="che41",
            node_id="lens",
            config={
                "sample": GEOMETRY["sample"],
                "num_rays": NUM_RAYS,
                "wavelength": WAVELENGTH_UM,
                "Hx": 0.0,
                "Hy": hy,
                "handoff_plane": "exit_pupil",
                "output_directory": str(directory),
            },
        )
    )
    return result


def _bundle(rays, *, perturbation=None):
    from couplers.handoff import (
        DeclaredHandoffPlane,
        HandoffPerturbation,
        declare_coherent_bundle,
    )

    return declare_coherent_bundle(
        rays,
        declared_plane=DeclaredHandoffPlane("exit_pupil", GEOMETRY["pupil_z_m"]),
        perturbation=perturbation or HandoffPerturbation(),
    )


def _psf(bundle, directory: Path, *, transpose: bool = False):
    """Pupil field -> Chromatix ASM -> measured PSF, through the shipping calls."""
    from couplers.ray_to_wave import Perturbation, ray_to_wave
    from solvers.base import ModelRunRequest
    from solvers.chromatix.adapter import get_adapter
    from verification.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf_from_record,
    )

    directory.mkdir(parents=True, exist_ok=True)
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(GEOMETRY["grid_n"], GEOMETRY["grid_n"]),
        sample_pitch_m=(GEOMETRY["pitch_m"], GEOMETRY["pitch_m"]),
        perturbation=Perturbation(transpose_axes=True) if transpose else Perturbation(),
    )
    record = field.to_artifact_record(artifact_id="pupil:che41", uri=directory / "pupil.npy")
    record.metadata["z_m"] = GEOMETRY["pupil_z_m"]
    record.metadata["reference_plane"] = "exit_pupil"
    result = get_adapter().run(
        ModelRunRequest(
            run_id="che41",
            node_id="wave",
            inputs={"input_field": record},
            config={
                "propagation": "angular_spectrum",
                "propagation_method": "asm_carrier_removed",
                "target_plane_z_m": GEOMETRY["image_z_m"],
                "pad_width": GEOMETRY["pad_width"],
                "output_dir": str(directory / "wave"),
            },
        )
    )
    if result.status.value != "succeeded":
        return None
    reported = result.diagnostics["output_sample_pitch_m"]
    return measure_psf_from_record(
        result.outputs["output_field"],
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
    )


def _chief_ray_image_point(bundle) -> dict[str, Any]:
    """Where the rays actually go, computed from the bundle and nothing else."""
    positions = np.asarray(bundle.positions_m)
    directions = np.asarray(bundle.directions)
    step = GEOMETRY["image_z_m"] - GEOMETRY["pupil_z_m"]
    x_img = positions[:, 0] + directions[:, 0] * step / directions[:, 2]
    y_img = positions[:, 1] + directions[:, 1] * step / directions[:, 2]
    radius = np.hypot(positions[:, 0], positions[:, 1])
    chief = int(np.argmin(radius))
    centroid = (float(np.mean(x_img)), float(np.mean(y_img)))
    return {
        "chief_ray_row": chief,
        "chief_ray_pupil_radius_m": float(radius[chief]),
        "chief_ray_image_point_m": [float(x_img[chief]), float(y_img[chief])],
        "ray_centroid_image_point_m": list(centroid),
        "chief_ray_minus_centroid_m": [
            float(x_img[chief]) - centroid[0],
            float(y_img[chief]) - centroid[1],
        ],
        "x_img": x_img,
        "y_img": y_img,
    }


def _geometric_spot(geometry_point: dict[str, Any], airy_radius_m: float) -> dict[str, Any]:
    """RMS and maximum ray radius about the chief-ray point, in Airy radii.

    Reference-sphere-free by construction. M3.8's method note is that a laterally
    displaced sphere centre is only approximately a linear ramp and the remainder
    reads as aberration, so a claim about where the light goes needs at least one
    measure that never fits a sphere.
    """
    x0, y0 = geometry_point["chief_ray_image_point_m"]
    radius = np.hypot(geometry_point["x_img"] - x0, geometry_point["y_img"] - y0)
    return {
        "rms_radius_m": float(np.sqrt(np.mean(radius**2))),
        "max_radius_m": float(np.max(radius)),
        "rms_radius_in_airy_radii": float(np.sqrt(np.mean(radius**2)) / airy_radius_m),
        "max_radius_in_airy_radii": float(np.max(radius) / airy_radius_m),
        "airy_radius_m": airy_radius_m,
    }


def _wavefront(bundle, geometry_point: dict[str, Any]) -> dict[str, Any]:
    """The pupil wavefront against the traced chief-ray point, fitted and not."""
    from verification.psf_oracles import pupil_aberration

    observation = (
        geometry_point["chief_ray_image_point_m"][0],
        geometry_point["chief_ray_image_point_m"][1],
        GEOMETRY["image_z_m"],
    )
    at_point = pupil_aberration(
        bundle,
        plane_z_m=GEOMETRY["pupil_z_m"],
        observation_point_m=observation,
        fit_sphere=False,
    )
    fitted = pupil_aberration(
        bundle,
        plane_z_m=GEOMETRY["pupil_z_m"],
        observation_point_m=observation,
        fit_sphere=True,
    )
    centre = fitted.sphere.center_m
    return {
        "against_the_traced_chief_ray_point": at_point.as_dict(),
        "with_the_sphere_centre_fitted": fitted.as_dict(),
        "fitted_centre_m": list(centre),
        "fitted_centre_distance_from_the_chief_ray_point_m": float(
            np.hypot(
                centre[0] - geometry_point["chief_ray_image_point_m"][0],
                centre[1] - geometry_point["chief_ray_image_point_m"][1],
            )
        ),
        "fitted_centre_distance_from_the_axis_m": float(np.hypot(centre[0], centre[1])),
    }


def _psf_position(measurement, geometry_point: dict[str, Any]) -> dict[str, Any]:
    """The oracle: the measured peak against the traced chief-ray intersection."""
    ny, nx = measurement.intensity.shape
    pitch = measurement.sample_pitch_m
    predicted_m = geometry_point["chief_ray_image_point_m"]
    predicted_px = (predicted_m[1] / pitch[0], predicted_m[0] / pitch[1])
    measured_px = (
        measurement.peak_index[0] - ny // 2,
        measurement.peak_index[1] - nx // 2,
    )
    airy_radius_m = 0.6098349456 * WAVELENGTH_M / GEOMETRY["na_frozen"]
    error_m = (
        measured_px[0] * pitch[0] - predicted_m[1],
        measured_px[1] * pitch[1] - predicted_m[0],
    )
    return {
        "predicted_peak_offset_pixels_y_x": list(predicted_px),
        "measured_peak_offset_pixels_y_x": list(measured_px),
        "measured_peak_position_m_y_x": list(measurement.peak_position_m),
        "predicted_peak_position_m_y_x": [predicted_m[1], predicted_m[0]],
        "error_m_y_x": list(error_m),
        "error_pixels_y_x": [error_m[0] / pitch[0], error_m[1] / pitch[1]],
        "error_in_airy_radii": float(np.hypot(*error_m) / airy_radius_m),
        "peak_is_off_axis_in_y_only": bool(measured_px[0] != 0 and measured_px[1] == 0),
        "airy_radius_m": airy_radius_m,
        "airy_radius_in_pixels": airy_radius_m / pitch[0],
        "border_energy_fraction": measurement.border_energy_fraction,
        "note": (
            "the peak is a pixel-quantized estimate of a continuous maximum, so an "
            "error below one pixel is the measurement floor rather than agreement to "
            "that precision. One pixel is 0.41 Airy radii here."
        ),
    }


def _sampling(bundle) -> dict[str, Any]:
    """The pitch a tilt-carrying pupil needs, over this trace's marginal rays."""
    directions = np.asarray(bundle.directions)
    cosine = float(np.max(np.hypot(directions[:, 0], directions[:, 1])))
    admissible = WAVELENGTH_M / (2.0 * cosine)
    return {
        "rule": "pitch <= lambda / (2 * max|transverse direction cosine|)",
        "max_transverse_direction_cosine": cosine,
        "admissible_pitch_m": admissible,
        "frozen_pitch_m": GEOMETRY["pitch_m"],
        "fraction_of_the_admissible_pitch": GEOMETRY["pitch_m"] / admissible,
        "admissible": bool(GEOMETRY["pitch_m"] <= admissible),
        "consequence_of_the_declared_reference": (
            "the tilt is IN the pupil field, so it is sampled, so it constrains the "
            "pitch. That is the price of declaring the incoming wavefront rather than a "
            "chief-ray-referenced frame, and it is paid at this field with 5.4% to "
            "spare. The full field scan (benchmarks/probes/records/optiland/"
            "off_axis_opd_reference.json) puts the limit at Hy ~ 0.25 and the "
            "refinement needed at full field at 2.58x."
        ),
    }


def _profile_residual_about(
    measured: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
    *,
    pitch: tuple[float, float],
    max_radius_m: float,
    center_m: tuple[float, float],
) -> float | None:
    """CHE-37's azimuthally-averaged metric, about a stated centre."""
    from verification.psf_oracles import azimuthal_profile

    radii_m, profile_m = azimuthal_profile(
        measured / float(np.max(measured)),
        sample_pitch_m=pitch,
        center_m=center_m,
        max_radius_m=max_radius_m,
        radial_samples=400,
        azimuthal_samples=256,
    )
    radii_r, profile_r = azimuthal_profile(
        reference / float(np.max(reference)),
        sample_pitch_m=pitch,
        center_m=center_m,
        max_radius_m=max_radius_m,
        radial_samples=400,
        azimuthal_samples=256,
    )
    common = np.interp(radii_m, radii_r, profile_r)
    difference = profile_m - common
    denominator = float(np.linalg.norm(common))
    return float(np.linalg.norm(difference) / denominator) if denominator else None


def _window_residual(
    measured: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
) -> float | None:
    """Relative L2 over the whole window, with no azimuthal average.

    The distinction matters and is the second half of why CHE-37's transpose
    control could not fail: an azimuthal average about the grid centre is
    invariant under an x/y transpose *by construction*, so no configuration --
    off axis or not -- can be detected by it.
    """
    a = measured / float(np.max(measured))
    b = reference / float(np.max(reference))
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denominator) if denominator else None


def _axis_transpose_control(
    bundle, geometry_point: dict[str, Any], directory: Path
) -> dict[str, Any]:
    """Re-run CHE-37's axis-transpose control against a working off-axis PSF."""
    from verification.psf_oracles import airy_psf_on_grid

    unperturbed = _psf(bundle, directory / "unperturbed")
    transposed = _psf(bundle, directory / "transposed", transpose=True)
    if unperturbed is None or transposed is None:
        return {"status": "a configuration failed to run"}

    pitch = unperturbed.sample_pitch_m
    shape = unperturbed.intensity.shape
    x0, y0 = geometry_point["chief_ray_image_point_m"]
    airy_radius_m = 0.6098349456 * WAVELENGTH_M / GEOMETRY["na_frozen"]
    compare_radius = 6.0 * airy_radius_m

    # The oracle is an Airy pattern centred WHERE THE RAYS GO. CHE-37's version
    # centred it on the axis, which is the same blind spot in a different place.
    oracle = airy_psf_on_grid(
        shape=shape,
        sample_pitch_m=pitch,
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=GEOMETRY["na_frozen"],
        center_m=(y0, x0),
    )

    def offsets(measurement):
        return (
            measurement.peak_index[0] - shape[0] // 2,
            measurement.peak_index[1] - shape[1] // 2,
        )

    unperturbed_offset = offsets(unperturbed)
    transposed_offset = offsets(transposed)

    che37_metric = {
        name: _profile_residual_about(
            psf.intensity,
            oracle,
            pitch=pitch,
            max_radius_m=compare_radius,
            center_m=(0.0, 0.0),
        )
        for name, psf in (("unperturbed", unperturbed), ("transposed", transposed))
    }
    recentred_metric = {
        name: _profile_residual_about(
            psf.intensity,
            oracle,
            pitch=pitch,
            max_radius_m=compare_radius,
            center_m=(y0, x0),
        )
        for name, psf in (("unperturbed", unperturbed), ("transposed", transposed))
    }
    window_metric = {
        name: _window_residual(psf.intensity, oracle)
        for name, psf in (("unperturbed", unperturbed), ("transposed", transposed))
    }
    displacement = float(
        np.hypot(
            (transposed_offset[0] - unperturbed_offset[0]) * pitch[0],
            (transposed_offset[1] - unperturbed_offset[1]) * pitch[1],
        )
    )

    return {
        "status": "ran",
        "perturbation": "couplers.ray_to_wave.Perturbation(transpose_axes=True)",
        "unperturbed_peak_offset_pixels_y_x": list(unperturbed_offset),
        "transposed_peak_offset_pixels_y_x": list(transposed_offset),
        "transpose_moves_the_peak_from_y_to_x": bool(
            unperturbed_offset[0] != 0
            and unperturbed_offset[1] == 0
            and transposed_offset[1] != 0
            and transposed_offset[0] == 0
        ),
        "peak_displacement_m": displacement,
        "peak_displacement_in_airy_radii": displacement / airy_radius_m,
        "peak_displacement_pixels": displacement / pitch[0],
        "che37_metric_azimuthal_profile_about_the_grid_centre": {
            **che37_metric,
            "detection_margin": (
                che37_metric["transposed"] / che37_metric["unperturbed"]
                if che37_metric["unperturbed"]
                else None
            ),
            "verdict": (
                "STILL BLIND, and now for a reason that has nothing to do with the "
                "field: an azimuthal average about the grid centre cannot distinguish "
                "a peak at (114, 0) from one at (0, 114). The off-axis field was "
                "necessary to make this control meaningful and is not sufficient -- the "
                "SCORING was the second blind spot, and CHE-37's 1.0000066 margin was "
                "measuring both at once."
            ),
        },
        "che37_metric_recentred_on_the_traced_image_point": {
            **recentred_metric,
            "detection_margin": (
                recentred_metric["transposed"] / recentred_metric["unperturbed"]
                if recentred_metric["unperturbed"]
                else None
            ),
            "detected": bool(
                recentred_metric["unperturbed"]
                and recentred_metric["transposed"] / recentred_metric["unperturbed"] > 10.0
            ),
            "verdict": (
                "CHE-37's metric, unchanged except that both the analytic reference and "
                "the azimuthal average are centred on the traced image point instead of "
                "the grid origin. This is the smallest repair to the SCORING, and it is "
                "the one applied to m3_psf_verification's off_axis_negative_controls, "
                "where it also lifts the OPL-sign, phase-sign, oblique-ramp and "
                "distance-sign margins from within 2.6x of 1.0 to 651x-1400x."
            ),
        },
        "window_metric_no_azimuthal_average": {
            **window_metric,
            "reference": (
                "analytic Airy centred on the traced chief-ray intersection, "
                f"({y0:.6e}, {x0:.6e}) m"
            ),
            "detection_margin": (
                window_metric["transposed"] / window_metric["unperturbed"]
                if window_metric["unperturbed"]
                else None
            ),
            "detected": bool(
                window_metric["unperturbed"]
                and window_metric["transposed"] / window_metric["unperturbed"] > 10.0
            ),
        },
        "why_this_is_now_non_vacuous": (
            "the PSF sits 114 pixels off axis along +y and nowhere along x, so the "
            "configuration is asymmetric in exactly the degree of freedom the control "
            "perturbs, and the perturbed run has to put the peak somewhere the "
            "unperturbed one did not. Under the superseded declaration the PSF was on "
            "axis and no scoring could have helped."
        ),
    }


def characterize() -> dict[str, Any]:
    from couplers.handoff import (
        OPL_REFERENCE_VERSION,
        SUPERSEDED_OPL_REFERENCE,
        HandoffPerturbation,
    )

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    workdir = Path(tempfile.mkdtemp(prefix="che41-off-axis-"))
    out: dict[str, Any] = {
        "probe": "m3_off_axis_handoff",
        "issue": "CHE-41",
        "protocol_id": protocol["protocol_id"],
        "wavelength_m": WAVELENGTH_M,
        "num_rays_requested": NUM_RAYS,
        "configuration": dict(GEOMETRY),
        "opl_reference_version": OPL_REFERENCE_VERSION,
        "superseded_opl_reference": SUPERSEDED_OPL_REFERENCE,
        "declared_choice": {
            "reference": "the incoming tilted wavefront",
            "alternative_rejected": (
                "an explicit chief-ray-referenced frame with the observation window "
                "centred on the chief ray"
            ),
            "reasoning": [
                "It is the completion of opd_native into a physical quantity. Optiland "
                "seeds the accumulator on a plane perpendicular to z; naming the "
                "wavefront as the reference replaces a coordinate-system artifact with "
                "a surface the incoming bundle itself defines, and that surface is the "
                "same object at every field angle.",
                "It keeps the reconstruction in the world frame the rays are traced in, "
                "which is what makes the oracle falsifiable: 'the PSF lands at the "
                "traced chief-ray intersection' is a prediction about a coordinate "
                "nobody chose. Under a chief-ray-centred window the PSF lands at the "
                "window centre BY DECLARATION, and the tilt is no longer under test.",
                "The chief-ray frame cannot be built by removing a linear ramp -- a "
                "lateral shift of the sphere centre is only approximately a ramp, and "
                "the remainder grows with field angle and reads as aberration. That is "
                "the error that reported this system at 1.0 waves RMS in M3.8. Its "
                "exact form needs a reference sphere aimed at a chosen image point, so "
                "the correct and the incorrect implementations differ by a term shaped "
                "like the thing the slice is trying to measure.",
                "The sampling cost is real but is not currently binding: the frozen "
                "pitch admits the tilt at this field with 5.4% to spare, and the limit "
                "(Hy ~ 0.25, 2.58x refinement at full field) is recorded rather than "
                "discovered later. A chief-ray frame would not sample the tilt, which "
                "removes the constraint from the numerics without removing it from the "
                "physics.",
                "It is the more general of the two. A chief-ray-referenced frame can be "
                "layered on top of a wavefront-referenced OPL by subtracting the exact "
                "sphere aimed at the traced chief-ray intersection; the reverse is "
                "impossible, because recovering the tilt needs object-space data that "
                "the exit-pupil export does not carry.",
            ],
        },
    }

    try:
        traced = _trace(workdir / "rays", hy=GEOMETRY["hy"])
        if traced.status.value != "succeeded":
            out["status"] = "trace_failed"
            out["error"] = traced.error_message
            return out
        rays = traced.outputs["rays"]
        out["ray_record"] = {
            "traced_num_rays": rays.metadata["traced_num_rays"],
            "scientific_array_sha256": rays.metadata["scientific_array_sha256"],
            "object_space_reference": rays.metadata["conventions"]["object_space_reference"],
            "opd_reference_surface": rays.metadata["conventions"]["opd_reference_surface"],
            "opd_omits_incoming_wavefront_tilt": rays.metadata["conventions"][
                "opd_omits_incoming_wavefront_tilt"
            ],
            "opd_is_relative_to_chief_ray": rays.metadata["conventions"][
                "opd_is_relative_to_chief_ray"
            ],
        }

        airy_radius_m = 0.6098349456 * WAVELENGTH_M / GEOMETRY["na_frozen"]
        results: dict[str, Any] = {}
        for name, perturbation in (
            ("declared_v2_incoming_wavefront", HandoffPerturbation()),
            (
                "superseded_v1_launch_plane",
                HandoffPerturbation(reference_incoming_wavefront=False),
            ),
        ):
            handoff = _bundle(rays, perturbation=perturbation)
            bundle = handoff.bundle
            point = _chief_ray_image_point(bundle)
            measurement = _psf(bundle, workdir / name)
            entry: dict[str, Any] = {
                "handoff_diagnostics": {
                    key: handoff.diagnostics[key]
                    for key in (
                        "opl_reference_version",
                        "object_space_reference_applied",
                        "object_space_reference_status",
                        "object_space_reference_span_m",
                        "object_space_reference_span_waves",
                        "relative_opl_span_waves",
                        "chief_ray_index",
                        "perturbation",
                    )
                },
                "traced_geometry": {
                    key: value for key, value in point.items() if key not in {"x_img", "y_img"}
                },
                "geometric_spot_about_the_chief_ray_point": _geometric_spot(point, airy_radius_m),
                "pupil_wavefront": _wavefront(bundle, point),
            }
            entry["psf_position"] = (
                _psf_position(measurement, point)
                if measurement is not None
                else {"status": "propagation_failed"}
            )
            # The linear slope CHE-37 reported, recomputed so the two records can be
            # compared directly.
            positions = np.asarray(bundle.positions_m)
            opl = np.asarray(bundle.optical_path_length_m)
            design = np.stack(
                [np.ones_like(positions[:, 1]), positions[:, 0], positions[:, 1]], axis=1
            )
            coefficients, *_ = np.linalg.lstsq(design, opl, rcond=None)
            step = GEOMETRY["image_z_m"] - GEOMETRY["pupil_z_m"]
            entry["declared_opl_linear_slope_y"] = float(coefficients[2])
            entry["slope_required_to_reach_the_chief_ray_point"] = (
                point["chief_ray_image_point_m"][1] / step
            )
            entry["slope_as_fraction_of_required"] = float(
                coefficients[2] / (point["chief_ray_image_point_m"][1] / step)
            )
            # The declared slope reads 0.19% above y_image / R, and that excess is
            # NOT a residual error: y/R is the sphere's slope at the pupil CENTRE,
            # while a least-squares line over the whole pupil is not. Measured
            # rather than argued -- the same fit is run over the exact analytic
            # sphere aimed at the traced chief-ray point, on the same pupil samples.
            x_pupil, y_pupil = positions[:, 0], positions[:, 1]
            ideal = -np.sqrt(
                x_pupil**2 + (y_pupil - point["chief_ray_image_point_m"][1]) ** 2 + step**2
            )
            ideal = ideal - ideal[point["chief_ray_row"]]
            ideal_coefficients, *_ = np.linalg.lstsq(design, ideal, rcond=None)
            entry["slope_cross_check_against_the_exact_sphere"] = {
                "ideal_sphere_least_squares_slope_y": float(ideal_coefficients[2]),
                "ideal_sphere_slope_as_fraction_of_y_over_r": float(
                    ideal_coefficients[2] / (point["chief_ray_image_point_m"][1] / step)
                ),
                "declared_slope_over_ideal_sphere_slope": float(
                    coefficients[2] / ideal_coefficients[2]
                ),
                "declared_minus_ideal_sphere_pv_waves": float(
                    np.ptp(opl - ideal) / bundle.wavelength_m
                ),
                "meaning": (
                    "an exact converging sphere aimed at the traced chief-ray point "
                    "has the same 0.18% least-squares slope excess over y/R, so the "
                    "declared OPL's slope agrees with the exact sphere's to 0.004% "
                    "and the remaining difference between them is the system's own "
                    "aberration, not a reference error. This is why the sphere fit "
                    "rather than the slope is the oracle."
                ),
            }
            results[name] = entry

        out["variants"] = results
        declared = results["declared_v2_incoming_wavefront"]
        superseded = results["superseded_v1_launch_plane"]
        out["oracle_verdict"] = {
            "psf_lands_at_the_traced_chief_ray_intersection": bool(
                declared["psf_position"].get("error_in_airy_radii", 9e9) < 0.5
            ),
            "psf_error_in_airy_radii": declared["psf_position"].get("error_in_airy_radii"),
            "psf_error_pixels_y_x": declared["psf_position"].get("error_pixels_y_x"),
            "sphere_centre_distance_from_the_chief_ray_point_m": declared["pupil_wavefront"][
                "fitted_centre_distance_from_the_chief_ray_point_m"
            ],
            "sphere_centre_improvement_over_the_superseded_declaration": (
                superseded["pupil_wavefront"]["fitted_centre_distance_from_the_chief_ray_point_m"]
                / declared["pupil_wavefront"]["fitted_centre_distance_from_the_chief_ray_point_m"]
            ),
            "pv_waves_against_the_chief_ray_point": declared["pupil_wavefront"][
                "against_the_traced_chief_ray_point"
            ]["peak_to_valley_waves"],
            "superseded_pv_waves_against_the_chief_ray_point": superseded["pupil_wavefront"][
                "against_the_traced_chief_ray_point"
            ]["peak_to_valley_waves"],
            "psf_displacement_between_the_two_declarations_pixels": (
                declared["psf_position"]["measured_peak_offset_pixels_y_x"][0]
                - superseded["psf_position"]["measured_peak_offset_pixels_y_x"][0]
            ),
            "why_the_superseded_case_is_the_negative_control": (
                "it is the shipping code with one term removed through its own "
                "perturbation hook, and it must move the PSF by the full 114 pixels. A "
                "fix whose omission changes nothing is not a fix."
            ),
        }

        bundle = _bundle(rays).bundle
        out["sampling"] = _sampling(bundle)
        out["axis_transpose_control"] = _axis_transpose_control(
            bundle, _chief_ray_image_point(bundle), workdir / "transpose"
        )

        # On-axis invariance, asserted rather than assumed: the same system at
        # Hy = 0 must produce a declared OPL that is bit-identical under both
        # declarations, because the object-space term is a piston there.
        on_axis = _trace(workdir / "rays_on_axis", hy=0.0)
        if on_axis.status.value == "succeeded":
            v2 = _bundle(on_axis.outputs["rays"]).bundle
            v1 = _bundle(
                on_axis.outputs["rays"],
                perturbation=HandoffPerturbation(reference_incoming_wavefront=False),
            ).bundle
            out["on_axis_invariance"] = {
                "sample": GEOMETRY["sample"],
                "hy": 0.0,
                "object_space_reference_span_m": float(
                    np.ptp(np.asarray(v2.provenance["object_space_reference_offset_m"]))
                ),
                "declared_opl_is_bit_identical": bool(
                    np.array_equal(
                        np.asarray(v1.optical_path_length_m),
                        np.asarray(v2.optical_path_length_m),
                    )
                ),
                "max_abs_difference_waves": float(
                    np.max(
                        np.abs(
                            np.asarray(v1.optical_path_length_m)
                            - np.asarray(v2.optical_path_length_m)
                        )
                    )
                    / v2.wavelength_m
                ),
                "meaning": (
                    "on axis the term is a constant, and a constant cannot survive the "
                    "chief-ray subtraction. The declaration change is therefore a no-op "
                    "for every configuration M3.4-M3.8 verified -- not a small change, "
                    "an exactly zero one."
                ),
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return out


def main() -> None:
    record = characterize()
    # CHE-103: stamp the source and environment the record was produced by, so a
    # later tree can tell whether this file still describes it. Built AFTER
    # characterize() so `sys.modules` holds everything the run actually imported.
    record[RECORD_PROVENANCE_KEY] = record_provenance(
        probe="m3_off_axis_handoff",
        root=ROOT,
        extra_sources=[Path(__file__)],
        data_inputs=[PROTOCOL_PATH],
    )
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {RECORD_PATH.relative_to(ROOT)}")
    print(json.dumps(record, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
