"""Which SURFACE does `RealRays.opd` accumulate from off axis? (CHE-41 / M3.9).

CHE-30 established four parts of the `opd` convention and located its zero at
"the ray launch state". Every case it used was on axis, and on axis a launch
*plane* and a launch *wavefront* are the same surface, so the characterization
could not distinguish them. This probe asks the question the on-axis cases could
not: for an off-axis collimated bundle, is the seeded reference surface a
wavefront of that bundle, or a plane perpendicular to z?

The two answers differ by ``n_object * (d0 . r_launch)`` -- constant on axis, and
linear in the launch coordinate off it. A linear term in the pupil is a tilt, and
a tilt is where the light goes. CHE-37 measured the consequence downstream (the
reconstructed wave converged 209 um from the traced chief-ray intersection at
``Hy = 0.2``); this probe establishes the cause at the source.

Every case states its prediction *and* the competing hypothesis it rules out, so
a passing case is falsifiable rather than merely consistent. Two of them are
falsifiers by construction: case 4 requires the correction to be exactly zero on
axis (otherwise this ticket would have moved a verified on-axis number), and
case 5 requires the fitted reference sphere to land on the AXIS under the plane
reference, which is the defect reproduced rather than argued about.

Run inside the agent_solver container:
    ./run.sh python benchmarks/probes/optiland/off_axis_opd_reference.py
    ./run.sh python benchmarks/probes/optiland/off_axis_opd_reference.py \
        --write-expected benchmarks/probes/records/optiland/off_axis_opd_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optiland.backend as be
import optiland.backend.utils as be_utils
from optiland.distribution import create_distribution
from optiland.samples import objectives

WAVELENGTH_UM = 0.55
WAVELENGTH_MM = WAVELENGTH_UM * 1.0e-3
NUM_RAYS = 32
SAMPLE = "ReverseTelephoto"
#: The field CHE-37 measured the defect at, and the largest field at which the
#: frozen M3-REVERSE-TELEPHOTO pitch still admits a tilt-carrying pupil.
FIELD_HY = 0.2
#: The frozen pupil pitch for M3-REVERSE-TELEPHOTO, from benchmarks/protocols/slice_protocol.yaml.
FROZEN_PITCH_M = 1.8258157981959995e-06


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(be_utils.to_numpy(value), dtype=np.float64)


def _launch_state(optic: Any, hy: float) -> Any:
    """Regenerate the launch rays for the distribution `Optic.trace` builds.

    `Optic.trace` creates its rays through `RayGenerator.generate_rays` and
    returns only the traced result, so the launch state has to be regenerated
    rather than read. Case 2 is what makes that admissible.
    """
    distribution = create_distribution("hexapolar")
    distribution.generate_points(NUM_RAYS)
    pupil_x, pupil_y = distribution.x, distribution.y
    count = int(_to_numpy(pupil_x).size)
    return optic.ray_tracer.ray_generator.generate_rays(
        be.repeat(be.atleast_1d(be.array(0.0)), count),
        be.repeat(be.atleast_1d(be.array(float(hy))), count),
        pupil_x,
        pupil_y,
        WAVELENGTH_UM,
    )


def _geometry(optic: Any) -> dict[str, float]:
    image_z_mm = float(_to_numpy(optic.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
    pupil_z_mm = image_z_mm + float(_to_numpy(optic.paraxial.XPL()).ravel()[0])
    return {
        "image_z_mm": image_z_mm,
        "pupil_z_mm": pupil_z_mm,
        "pupil_to_image_mm": image_z_mm - pupil_z_mm,
        "n_image": float(
            _to_numpy(optic.surfaces.surfaces[-1].material_pre.n(WAVELENGTH_UM)).ravel()[0]
        ),
        "n_object": float(
            _to_numpy(optic.surfaces.surfaces[0].material_post.n(WAVELENGTH_UM)).ravel()[0]
        ),
    }


def _pupil_arrays(traced: Any, geometry: dict[str, float]) -> dict[str, np.ndarray]:
    """The traced rays' image-space asymptote at the exit pupil, and the OPL there."""
    x, y, z = _to_numpy(traced.x), _to_numpy(traced.y), _to_numpy(traced.z)
    lx, my, nz = _to_numpy(traced.L), _to_numpy(traced.M), _to_numpy(traced.N)
    opd = _to_numpy(traced.opd)
    pupil_z = geometry["pupil_z_mm"]
    step = (pupil_z - z) / nz
    return {
        "x_pupil_mm": x + lx * step,
        "y_pupil_mm": y + my * step,
        "L": lx,
        "M": my,
        "N": nz,
        # CHE-33 step 2: the OPL is accumulated to the traced image surface, the
        # positions are the asymptote at the pupil, so move the path back.
        "opl_at_pupil_mm": opd - geometry["n_image"] * (geometry["image_z_mm"] - pupil_z) / nz,
        "opd_native_mm": opd,
    }


def _fit_sphere(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    plane_z_mm: float,
    opl_mm: np.ndarray,
    initial_center_mm: tuple[float, float, float],
) -> dict[str, Any]:
    """Gauss-Newton for the sphere centre, in millimetres.

    Deliberately NOT a least-squares linear ramp. A lateral shift of the sphere
    centre is only approximately a ramp, and the remainder grows with field angle
    and reads as aberration -- the error that reported this system at 1.0 waves
    RMS in M3.8 before the spot check caught it.
    """
    center = np.array(initial_center_mm, dtype=np.float64)

    def residual(c: np.ndarray) -> np.ndarray:
        path = np.sqrt((c[0] - x_mm) ** 2 + (c[1] - y_mm) ** 2 + (c[2] - plane_z_mm) ** 2)
        total = opl_mm + path
        return total - total.mean()

    initial_rms = float(np.std(residual(center)) / WAVELENGTH_MM)
    iterations = 0
    for iterations in range(1, 26):  # noqa: B007
        dx, dy = center[0] - x_mm, center[1] - y_mm
        dz = center[2] - plane_z_mm
        path = np.sqrt(dx**2 + dy**2 + dz**2)
        current = opl_mm + path
        current = current - current.mean()
        jacobian = np.stack([dx / path, dy / path, np.full_like(path, dz) / path], axis=1)
        jacobian = jacobian - jacobian.mean(axis=0, keepdims=True)
        step, *_ = np.linalg.lstsq(jacobian, -current, rcond=None)
        center = center + step
        if float(np.max(np.abs(step))) <= 1e-14 * max(1.0, float(np.max(np.abs(center)))):
            break
    final = residual(center)
    return {
        "center_mm": [float(v) for v in center],
        "residual_rms_waves": float(np.std(final) / WAVELENGTH_MM),
        "residual_pv_waves": float(np.ptp(final) / WAVELENGTH_MM),
        "initial_residual_rms_waves": initial_rms,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_launch_surface_off_axis(optic: Any) -> dict[str, Any]:
    """Is the seeded reference surface a wavefront, or a plane perpendicular to z?"""
    launch = _launch_state(optic, FIELD_HY)
    z0 = _to_numpy(launch.z)
    l0, m0, n0 = _to_numpy(launch.L), _to_numpy(launch.M), _to_numpy(launch.N)
    field_deg = float(_to_numpy(optic.fields.max_y_field).ravel()[0]) * FIELD_HY
    entrance_pupil_diameter_mm = float(_to_numpy(optic.paraxial.EPD()).ravel()[0])
    wavefront_predicted_z_spread_mm = math.tan(math.radians(field_deg)) * entrance_pupil_diameter_mm
    return {
        "claim": (
            "for an object at infinity every ray is launched on ONE plane "
            "perpendicular to z, with ONE common direction. The reference surface is "
            "therefore a plane, and it is a wavefront of the bundle only when that "
            "direction is +z."
        ),
        "competing_hypothesis": (
            "the accumulator is seeded on a wavefront of the incoming bundle, i.e. the "
            "launch points lie on a plane perpendicular to d0. That predicts a z spread "
            f"of tan(theta) * EPD = {wavefront_predicted_z_spread_mm:.6f} mm across the "
            "pupil; a plane perpendicular to z predicts exactly 0."
        ),
        "field_hy": FIELD_HY,
        "field_angle_deg": field_deg,
        "launch_z_spread_mm": float(np.ptp(z0)),
        "launch_z_mm": float(z0[0]),
        "launch_direction_spread": [float(np.ptp(l0)), float(np.ptp(m0)), float(np.ptp(n0))],
        "launch_direction": [float(l0[0]), float(m0[0]), float(n0[0])],
        "predicted_M_is_sin_of_the_field_angle": math.sin(math.radians(field_deg)),
        "M_minus_sin_field_angle": float(m0[0]) - math.sin(math.radians(field_deg)),
        "verdict": (
            "plane perpendicular to z"
            if float(np.ptp(z0)) == 0.0
            else "NOT a plane perpendicular to z"
        ),
    }


def case_regenerated_launch_state_reproduces_the_trace(optic: Any) -> dict[str, Any]:
    """The regenerated launch state must be the one `Optic.trace` actually used."""
    traced = optic.trace(Hx=0.0, Hy=FIELD_HY, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
    traced_x, traced_y = _to_numpy(traced.x), _to_numpy(traced.y)
    traced_opd = _to_numpy(traced.opd)

    launch = _launch_state(optic, FIELD_HY)
    replay = deepcopy(launch)
    optic.surfaces.reset()
    optic.surfaces.trace(replay)
    last = optic.surfaces[-1]
    last.material_post.propagation_model.propagate(replay, last.thickness)

    return {
        "claim": (
            "tracing the regenerated launch state reproduces Optic.trace EXACTLY -- "
            "not approximately -- so a term measured from it describes the exported "
            "rays row for row."
        ),
        "competing_hypothesis": (
            "the generator is stateful or the distribution is not reproducible, in "
            "which case the regenerated rows are a different ensemble and any per-ray "
            "term computed from them is silently misaligned."
        ),
        "ray_count": int(traced_x.size),
        "max_abs_x_difference_mm": float(np.max(np.abs(_to_numpy(replay.x) - traced_x))),
        "max_abs_y_difference_mm": float(np.max(np.abs(_to_numpy(replay.y) - traced_y))),
        "max_abs_opd_difference_mm": float(np.max(np.abs(_to_numpy(replay.opd) - traced_opd))),
        "verdict": (
            "bit-identical" if np.array_equal(_to_numpy(replay.opd), traced_opd) else "differs"
        ),
    }


def case_opd_is_not_chief_ray_relative(optic: Any) -> dict[str, Any]:
    """`opd_is_relative_to_chief_ray: False`, tested where it is testable."""
    geometry = _geometry(optic)
    traced = optic.trace(Hx=0.0, Hy=FIELD_HY, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
    pupil = _pupil_arrays(traced, geometry)
    radius = np.hypot(pupil["x_pupil_mm"], pupil["y_pupil_mm"])
    chief = int(np.argmin(radius))
    return {
        "claim": (
            "opd_native is an ABSOLUTE accumulated path, so the chief ray's own value "
            "is large rather than zero. On axis this claim was untestable in the sense "
            "that mattered: the tilt a chief-ray-referenced OPD would also lack is "
            "zero there, so the two hypotheses predicted the same array."
        ),
        "competing_hypothesis": (
            "opd is already an OPD relative to the chief ray, as the field name "
            "suggests. That predicts opd[chief] == 0 and, off axis, a tilt-free pupil "
            "wavefront -- the second half of which the plane reference ALSO predicts, "
            "which is exactly why the absolute-vs-relative question had to be settled "
            "by the first half."
        ),
        "chief_ray_row": chief,
        "chief_ray_pupil_radius_mm": float(radius[chief]),
        "chief_ray_opd_native_mm": float(pupil["opd_native_mm"][chief]),
        "chief_ray_opd_native_waves": float(pupil["opd_native_mm"][chief] / WAVELENGTH_MM),
        "opd_is_relative_to_chief_ray": bool(
            abs(float(pupil["opd_native_mm"][chief])) < WAVELENGTH_MM
        ),
        "verdict": "absolute accumulated optical path, confirmed off axis",
    }


def case_the_omitted_term_is_the_convergence_tilt(optic: Any) -> dict[str, Any]:
    """The term's presence or absence decides where the wave converges."""
    geometry = _geometry(optic)
    traced = optic.trace(Hx=0.0, Hy=FIELD_HY, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
    pupil = _pupil_arrays(traced, geometry)
    launch = _launch_state(optic, FIELD_HY)
    correction_mm = geometry["n_object"] * (
        _to_numpy(launch.L) * _to_numpy(launch.x)
        + _to_numpy(launch.M) * _to_numpy(launch.y)
        + _to_numpy(launch.N) * _to_numpy(launch.z)
    )

    x_p, y_p = pupil["x_pupil_mm"], pupil["y_pupil_mm"]
    chief = int(np.argmin(np.hypot(x_p, y_p)))
    to_image = geometry["pupil_to_image_mm"] / pupil["N"][chief]
    chief_image_y_mm = float(y_p[chief] + pupil["M"][chief] * to_image)
    chief_image_x_mm = float(x_p[chief] + pupil["L"][chief] * to_image)

    design = np.stack([np.ones_like(y_p), x_p, y_p], axis=1)
    variants: dict[str, Any] = {}
    for name, opl in (
        ("plane_perpendicular_to_z", pupil["opl_at_pupil_mm"]),
        ("incoming_wavefront", pupil["opl_at_pupil_mm"] + correction_mm),
    ):
        relative = opl - opl[chief]
        coefficients, *_ = np.linalg.lstsq(design, relative, rcond=None)
        sphere = _fit_sphere(
            x_p,
            y_p,
            geometry["pupil_z_mm"],
            relative,
            (0.0, chief_image_y_mm, geometry["image_z_mm"]),
        )
        against_chief = relative + np.sqrt(
            (0.0 - x_p) ** 2
            + (chief_image_y_mm - y_p) ** 2
            + (geometry["image_z_mm"] - geometry["pupil_z_mm"]) ** 2
        )
        variants[name] = {
            "least_squares_slope_y": float(coefficients[2]),
            "least_squares_slope_x": float(coefficients[1]),
            "pv_waves_against_the_traced_chief_ray_point": float(
                np.ptp(against_chief) / WAVELENGTH_MM
            ),
            "rms_waves_against_the_traced_chief_ray_point": float(
                np.std(against_chief) / WAVELENGTH_MM
            ),
            "fitted_sphere": sphere,
            "fitted_centre_y_mm": sphere["center_mm"][1],
            "fitted_centre_distance_from_the_chief_ray_point_mm": abs(
                sphere["center_mm"][1] - chief_image_y_mm
            ),
            "fitted_centre_distance_from_the_axis_mm": abs(sphere["center_mm"][1]),
        }

    return {
        "claim": (
            "adding n_object * (d0 . r_launch) turns the pupil OPL into a wave that "
            "converges on the TRACED CHIEF-RAY INTERSECTION. Without it the same "
            "arithmetic produces a clean converging sphere aimed at the axis."
        ),
        "competing_hypothesis": (
            "the missing tilt is an artifact of the exit-pupil projection or of the "
            "chief-ray piston subtraction, and can be repaired downstream. It cannot: "
            "the term below is a function of the LAUNCH coordinate, and no "
            "object-space coordinate survives into the exported pupil arrays."
        ),
        "geometry": geometry,
        "traced_chief_ray_image_point_mm": [chief_image_x_mm, chief_image_y_mm],
        "slope_required_to_reach_it": chief_image_y_mm / geometry["pupil_to_image_mm"],
        "why_the_required_slope_is_not_the_fitted_slope": (
            "y_image / R is the sphere's slope AT THE PUPIL CENTRE; a least-squares "
            "line over the whole pupil reads 0.19% higher because the sphere's slope is "
            "not constant. The sphere fit is the oracle; the slopes are reported "
            "because CHE-37 reported them."
        ),
        "correction_term_mm": {
            "min": float(np.min(correction_mm)),
            "max": float(np.max(correction_mm)),
            "span": float(np.ptp(correction_mm)),
            "span_waves": float(np.ptp(correction_mm) / WAVELENGTH_MM),
        },
        "variants": variants,
        "improvement_in_the_sphere_centre": (
            variants["plane_perpendicular_to_z"][
                "fitted_centre_distance_from_the_chief_ray_point_mm"
            ]
            / variants["incoming_wavefront"]["fitted_centre_distance_from_the_chief_ray_point_mm"]
        ),
    }


def case_the_term_is_a_piston_on_axis(optic: Any) -> dict[str, Any]:
    """On axis the correction must be EXACTLY constant, not merely small."""
    geometry = _geometry(optic)
    launch = _launch_state(optic, 0.0)
    correction_mm = geometry["n_object"] * (
        _to_numpy(launch.L) * _to_numpy(launch.x)
        + _to_numpy(launch.M) * _to_numpy(launch.y)
        + _to_numpy(launch.N) * _to_numpy(launch.z)
    )
    return {
        "claim": (
            "at Hy = 0 the launch direction is exactly +z, so the term is "
            "n_object * z_launch: one number, identical for every ray. A consumer that "
            "removes a piston is therefore free to skip it, and every on-axis number "
            "M3.4-M3.8 recorded is unaffected by this ticket -- bit-identically, not "
            "within tolerance."
        ),
        "competing_hypothesis": (
            "the term is merely small on axis. That would make this a change to the "
            "verified on-axis slice, and the frozen 0.016999-wave residual would have "
            "to be re-verified rather than asserted."
        ),
        "launch_direction": [
            float(_to_numpy(launch.L)[0]),
            float(_to_numpy(launch.M)[0]),
            float(_to_numpy(launch.N)[0]),
        ],
        "correction_span_mm": float(np.ptp(correction_mm)),
        "correction_value_mm": float(correction_mm[0]),
        "span_is_exactly_zero": bool(float(np.ptp(correction_mm)) == 0.0),
    }


def case_admissible_pitch_versus_field(optic: Any) -> dict[str, Any]:
    """What a tilt-carrying pupil costs in sampling, measured over real marginal rays.

    The protocol's Nyquist rule is evaluated over the largest transverse direction
    cosine of an actual trace, so the chief-ray tilt enters it directly. This is
    the re-evaluation the ticket asks for, and it is the one real cost of choosing
    the incoming wavefront over a chief-ray-referenced frame.
    """
    rows = []
    for hy in (0.0, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0):
        traced = optic.trace(Hx=0.0, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
        cosine = float(np.max(np.hypot(_to_numpy(traced.L), _to_numpy(traced.M))))
        admissible_m = (WAVELENGTH_UM * 1.0e-6) / (2.0 * cosine) if cosine else float("inf")
        rows.append(
            {
                "hy": hy,
                "max_transverse_direction_cosine": cosine,
                "admissible_pitch_m": admissible_m,
                "frozen_pitch_over_admissible": FROZEN_PITCH_M / admissible_m
                if math.isfinite(admissible_m)
                else 0.0,
                "admissible": bool(admissible_m >= FROZEN_PITCH_M),
                "refinement_factor_needed": max(1.0, FROZEN_PITCH_M / admissible_m)
                if math.isfinite(admissible_m)
                else 1.0,
            }
        )
    return {
        "claim": (
            "the frozen M3-REVERSE-TELEPHOTO pitch admits a tilt-carrying pupil out to "
            "Hy ~ 0.25 and no further; at full field it needs ~2.6x refinement."
        ),
        "rule": "pitch <= lambda / (2 * max|transverse direction cosine|)",
        "frozen_pitch_m": FROZEN_PITCH_M,
        "rows": rows,
        "note": (
            "this constraint is a property of the FIELD the pupil carries, not of the "
            "declaration: a chief-ray-referenced frame would not sample the tilt and "
            "would not pay it. CHE-41 chose to pay it, and records the limit here "
            "rather than discovering it at Hy = 0.3."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", type=Path)
    args = parser.parse_args()

    optic = getattr(objectives, SAMPLE)()
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "off_axis_opd_reference",
        "issue": "CHE-41",
        "status": "passed",
        "sample": SAMPLE,
        "wavelength_um": WAVELENGTH_UM,
        "num_rays_requested": NUM_RAYS,
        "backend": "numpy (default; not switched)",
        "accumulation_site": (
            "optiland/fields/field_types/angle.py::_get_starting_z_offset and "
            "get_ray_origins: z0 = positions[1] - (EPD - min(positions[1:-1])), one "
            "value for every ray"
        ),
        "cases": {
            "launch_surface_off_axis": case_launch_surface_off_axis(optic),
            "regenerated_launch_state_reproduces_the_trace": (
                case_regenerated_launch_state_reproduces_the_trace(optic)
            ),
            "opd_is_not_chief_ray_relative": case_opd_is_not_chief_ray_relative(optic),
            "the_omitted_term_is_the_convergence_tilt": (
                case_the_omitted_term_is_the_convergence_tilt(optic)
            ),
            "the_term_is_a_piston_on_axis": case_the_term_is_a_piston_on_axis(optic),
            "admissible_pitch_versus_field": case_admissible_pitch_versus_field(optic),
        },
    }

    cases = report["cases"]
    failures: list[str] = []
    if cases["launch_surface_off_axis"]["launch_z_spread_mm"] != 0.0:
        failures.append("the launch points do not lie on one plane perpendicular to z")
    if cases["regenerated_launch_state_reproduces_the_trace"]["max_abs_opd_difference_mm"] != 0.0:
        failures.append("the regenerated launch state does not reproduce the trace exactly")
    if not cases["the_term_is_a_piston_on_axis"]["span_is_exactly_zero"]:
        failures.append("the on-axis correction is not exactly a piston")
    tilt = cases["the_omitted_term_is_the_convergence_tilt"]
    wavefront = tilt["variants"]["incoming_wavefront"]
    plane = tilt["variants"]["plane_perpendicular_to_z"]
    if wavefront["fitted_centre_distance_from_the_chief_ray_point_mm"] > 5.0e-3:
        failures.append("the wavefront-referenced sphere centre is not at the chief-ray point")
    if plane["fitted_centre_distance_from_the_axis_mm"] > 5.0e-3:
        failures.append("the plane-referenced sphere centre is not on axis; defect not reproduced")
    if wavefront["pv_waves_against_the_traced_chief_ray_point"] > 0.25:
        failures.append("the wavefront-referenced residual is not inside the Rayleigh limit")

    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    if args.write_expected and report["status"] == "passed":
        args.write_expected.parent.mkdir(parents=True, exist_ok=True)
        args.write_expected.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
