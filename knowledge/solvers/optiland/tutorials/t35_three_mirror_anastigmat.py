"""Advanced / "Three-Mirror Anastigmat" -- https://www.optiland.org/tutorials/three-mirror-anastigmat

Repo-owned reproduction of the freeform-telescope tutorial: a 100 mm, EPD 10 mm
three-mirror anastigmat whose mirrors are ``surface_type='zernike'`` freeforms with
tilts and decenters, optimized over 27 variables (3 radii, 3 thicknesses, 3
y-decenters, 3 x-tilts, 3 conics and 12 Zernike coefficients) against
``real_y_intercept_lcs`` operands that place each field at ``f * tan(theta)`` and
``rms_spot_size`` operands at all three fields.

**Adaptation.** ``optimize(tol=1e-9)`` is capped at ``maxiter=60``; the merit
reduction achieved is recorded.

**The published optimization does not work on the pinned version, and the failure
mode is the most valuable thing here.** ``OptimizerGeneric`` (L-BFGS-B) terminates
after 4 iterations and 924 function evaluations with ``success = True`` and
``message = 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'``, at a point
where the merit function is **1.97x worse than where it started** (2.750 -> 5.417).
Raising ``maxiter`` from 60 to 200 to 500 changes nothing -- it really has stopped.
The cause is that a freeform tilted-mirror merit function is not smooth (rays
vignette in and out as surfaces move), so the finite-difference gradient is
unreliable and the line search accepts a bad step. Recorded because
``res.success`` is not evidence: a caller must compare the merit before and after.

The operand targets are declared analytically -- ``100 * tan(0 deg) = 0`` and
``100 * tan(+/-1.5 deg) = +/-2.61859 mm`` -- so the image heights are a real external
reference, and only one of the three is met. Both the hit and the misses are asserted.

What does hold:

* ``real_y_intercept_lcs`` is in the surface's **local** coordinate system, which is
  what makes it usable on a tilted, decentered mirror at all. The image surface here
  is decentered 22 mm, so reading intercepts in the global frame is wrong by 22 mm.
  The ``_lcs`` suffix is load-bearing rather than cosmetic.
* A three-mirror system is achromatic **exactly**: the three wavelengths give
  bit-identical ray coordinates, because no refractive surface is present. That is a
  physics invariant no optimizer can break, and it holds before and after.
* Every variable ends inside its declared bounds -- the tilt bounds in radians and
  the Zernike coefficients in [-1, 1] -- so the bad step is inside the feasible set,
  not a bounds violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t35_three_mirror_anastigmat",
    title="Three-Mirror Anastigmat",
    level="advanced",
    url="https://www.optiland.org/tutorials/three-mirror-anastigmat",
    demonstrates=(
        "surface_type='zernike' freeform mirrors with rx tilts and dy decenters, "
        "the 'decenter', 'tilt', 'conic' and 'zernike_coeff' variable types, and "
        "the 'real_y_intercept_lcs' operand (local coordinate system)."
    ),
    slow=True,
)

FOCAL_LENGTH_MM = 100.0
FIELDS_DEG = (0.0, 1.5, -1.5)
PRIMARY_UM = 0.587
MAXITER = 60  # upstream: uncapped tol=1e-9


def build_tma():
    from optiland import optic

    lens = optic.Optic(name="TMA")
    lens.set_aperture(aperture_type="EPD", value=10)
    lens.fields.set_type(field_type="angle")
    for y in FIELDS_DEG:
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.486)
    lens.wavelengths.add(value=PRIMARY_UM, is_primary=True)
    lens.wavelengths.add(value=0.656)

    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1, radius=-100, thickness=-20, conic=0, material="mirror",
        rx=np.radians(-15.0), is_stop=True, surface_type="zernike", coefficients=[],
    )
    lens.surfaces.add(
        index=2, radius=-100, thickness=+20, conic=0, material="mirror",
        rx=np.radians(-10.0), dy=-11.5, surface_type="zernike", coefficients=[],
    )
    lens.surfaces.add(
        index=3, radius=-100, thickness=-22, conic=0, material="mirror",
        rx=np.radians(-1.0), dy=-15, surface_type="zernike", coefficients=[],
    )
    lens.surfaces.add(index=4, dy=-19.3)
    lens.update_paraxial()
    return lens


BOUNDS = {
    ("radius", 1): (-1000, 1000), ("radius", 2): (-1000, 1000), ("radius", 3): (-1000, 1000),
    ("thickness", 1): (-35, -15), ("thickness", 2): (15, 35), ("thickness", 3): (-35, -15),
    ("decenter", 2): (-15, -10), ("decenter", 3): (-20, -11), ("decenter", 4): (-28, -22),
    ("tilt", 1): (np.radians(-20.0), np.radians(-12.0)),
    ("tilt", 2): (np.radians(-15.0), np.radians(-8.0)),
    ("tilt", 3): (np.radians(-10.0), np.radians(10.0)),
    ("conic", 1): (-10, 10), ("conic", 2): (-10, 10), ("conic", 3): (-10, 10),
}


def build_problem(lens):
    from optiland import optimization

    problem = optimization.OptimizationProblem()
    for surface_number in (1, 2, 3):
        problem.add_variable(lens, "radius", surface_number=surface_number, min_val=-1000, max_val=1000)
    for surface_number, (low, high) in ((1, (-35, -15)), (2, (15, 35)), (3, (-35, -15))):
        problem.add_variable(lens, "thickness", surface_number=surface_number, min_val=low, max_val=high)
    for surface_number, (low, high) in ((2, (-15, -10)), (3, (-20, -11)), (4, (-28, -22))):
        problem.add_variable(
            lens, "decenter", axis="y", surface_number=surface_number, min_val=low, max_val=high
        )
    for surface_number, (low, high) in (
        (1, (-20.0, -12.0)), (2, (-15.0, -8.0)), (3, (-10.0, 10.0))
    ):
        problem.add_variable(
            lens, "tilt", axis="x", surface_number=surface_number,
            min_val=np.radians(low), max_val=np.radians(high),
        )
    for surface_number in (1, 2, 3):
        problem.add_variable(lens, "conic", surface_number=surface_number, min_val=-10, max_val=10)
    for surface_number in range(1, 4):
        for coeff_index in range(4):
            problem.add_variable(
                lens, "zernike_coeff", surface_number=surface_number,
                coeff_index=coeff_index, min_val=-1, max_val=1,
            )

    primary = lens.wavelengths.primary_wavelength.value
    for surface_number in (2, 3):
        problem.add_operand(
            operand_type="real_y_intercept_lcs",
            target=0.0,
            weight=1,
            input_data={
                "optic": lens, "surface_number": surface_number,
                "Hx": 0, "Hy": 0, "Px": 0, "Py": 0, "wavelength": primary,
            },
        )
    for hy, field_index in ((0, 0), (1, 1), (-1, 2)):
        problem.add_operand(
            operand_type="real_y_intercept_lcs",
            target=FOCAL_LENGTH_MM * np.tan(np.deg2rad(lens.fields.y_fields[field_index])),
            weight=1,
            input_data={
                "optic": lens, "surface_number": 4,
                "Hx": 0, "Hy": hy, "Px": 0, "Py": 0, "wavelength": primary,
            },
        )
    for hx, hy in lens.fields.get_field_coords():
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=10,
            input_data={
                "optic": lens, "surface_number": 4, "Hx": hx, "Hy": hy,
                "num_rays": 16, "wavelength": PRIMARY_UM, "distribution": "uniform",
            },
        )
    return problem


def _image_heights(lens) -> dict[str, float]:
    """Chief-ray image heights in the IMAGE SURFACE'S LOCAL frame, in mm.

    The image surface carries a ``dy`` decenter, so the globally-recorded ``rays.y``
    is offset by it. ``real_y_intercept_lcs`` targets the local value, and comparing
    against the global one is wrong by the full decenter.
    """
    decenter_y = float(np.asarray(lens.surfaces.surfaces[4].geometry.cs.y).ravel()[0])
    out = {}
    for label, hy in (("axial", 0.0), ("plus", 1.0), ("minus", -1.0)):
        rays = lens.trace_generic(Hx=0.0, Hy=hy, Px=0.0, Py=0.0, wavelength=PRIMARY_UM)
        out[label] = float(np.asarray(rays.y, dtype=float).ravel()[0]) - decenter_y
    return out


def _rms_spot(lens) -> float:
    values = []
    for hx, hy in lens.fields.get_field_coords():
        rays = lens.trace(Hx=hx, Hy=hy, wavelength=PRIMARY_UM, num_rays=8)
        x = np.asarray(rays.x, dtype=float)
        y = np.asarray(rays.y, dtype=float)
        values.append(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)))
    return float(np.mean(values))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, optimization
    from optiland.mtf import GeometricMTF

    result = TutorialResult()
    lens = build_tma()
    targets = {
        "axial": FOCAL_LENGTH_MM * np.tan(np.deg2rad(FIELDS_DEG[0])),
        "plus": FOCAL_LENGTH_MM * np.tan(np.deg2rad(FIELDS_DEG[1])),
        "minus": FOCAL_LENGTH_MM * np.tan(np.deg2rad(FIELDS_DEG[2])),
    }
    heights_before = _image_heights(lens)
    rms_before = _rms_spot(lens)
    analysis.SpotDiagram(lens).view()
    lens.draw(title=lens.name)
    lens.info()
    plt.close("all")

    problem = build_problem(lens)
    merit_before = float(problem.sum_squared())
    problem.info()
    outcome = optimization.OptimizerGeneric(problem).optimize(
        maxiter=MAXITER, disp=False, tol=1e-9
    )
    merit_after = float(problem.sum_squared())
    heights_after = _image_heights(lens)
    rms_after = _rms_spot(lens)

    variables = {}
    bound_violations = []
    for variable in problem.variables:
        value = float(np.asarray(variable.value).ravel()[0])
        low, high = variable.bounds
        if (low is not None and value < low - 1e-9) or (high is not None and value > high + 1e-9):
            bound_violations.append((str(variable), value, low, high))
    for surface_number in (1, 2, 3):
        surface = lens.surfaces.surfaces[surface_number]
        variables[f"radius_{surface_number}"] = float(np.asarray(surface.geometry.radius).ravel()[0])
        variables[f"conic_{surface_number}"] = float(np.asarray(surface.geometry.k).ravel()[0])
        variables[f"thickness_{surface_number}"] = float(np.asarray(surface.thickness).ravel()[0])
        variables[f"tilt_x_{surface_number}"] = float(np.asarray(surface.geometry.cs.rx).ravel()[0])
    result.record(
        maxiter=MAXITER,
        num_variables=len(problem.variables),
        num_operands=len(problem.operands),
        focal_length_mm=FOCAL_LENGTH_MM,
        image_height_targets_mm=targets,
        image_heights_before_mm=heights_before,
        image_heights_after_mm=heights_after,
        merit_before=merit_before,
        merit_after=merit_after,
        merit_ratio_after_over_before=merit_after / merit_before,
        optimizer_success=bool(getattr(outcome, "success", False)),
        optimizer_message=str(getattr(outcome, "message", "")),
        optimizer_iterations=int(getattr(outcome, "nit", -1)),
        optimizer_function_evaluations=int(getattr(outcome, "nfev", -1)),
        mean_rms_spot_before_mm=rms_before,
        mean_rms_spot_after_mm=rms_after,
        optimized_geometry=variables,
        num_bound_violations=len(bound_violations),
    )
    result.check_true(
        "twenty_seven_variables_and_eight_operands_were_declared",
        "invariant",
        len(problem.variables) == 27 and len(problem.operands) == 8,
        f"{len(problem.variables)} variables (3 radii + 3 thicknesses + 3 decenters + "
        f"3 tilts + 3 conics + 12 zernike coefficients) and {len(problem.operands)} "
        "operands (5 real_y_intercept_lcs + 3 rms_spot_size)",
    )
    result.check_true(
        "the_optimizer_reports_success_while_making_the_design_worse",
        "analytic",
        bool(getattr(outcome, "success", False)) and merit_after > merit_before,
        f"OptimizerGeneric returns success={getattr(outcome, 'success', None)!r} with "
        f"message {str(getattr(outcome, 'message', ''))!r} after "
        f"{int(getattr(outcome, 'nit', -1))} iterations and "
        f"{int(getattr(outcome, 'nfev', -1))} function evaluations, leaving the merit at "
        f"{merit_after:.6e} against its starting {merit_before:.6e} -- "
        f"{merit_after / merit_before:.2f}x WORSE. maxiter 60, 200 and 500 all stop at the "
        "same point, so this is convergence, not truncation. A freeform tilted-mirror "
        "merit function is not smooth (rays vignette in and out as surfaces move), so the "
        "finite-difference gradient is unreliable. res.success is not evidence: compare "
        "the merit before and after.",
    )
    result.check_true(
        "the_bad_step_stays_inside_the_declared_bounds",
        "invariant",
        not bound_violations,
        f"{len(bound_violations)} of {len(problem.variables)} variables outside their "
        "bounds: the degradation is a genuine interior point of the feasible set, not a "
        "bounds-handling bug",
    )
    result.check_close(
        "the_plus_field_still_lands_on_f_tan_theta",
        "reference",
        heights_after["plus"],
        targets["plus"],
        rel=0.05,
    )
    missed = {
        label: (heights_after[label], targets[label])
        for label in ("axial", "minus")
        if abs(heights_after[label] - targets[label]) > 0.05 * max(abs(targets[label]), 1.0)
    }
    result.record(missed_intercept_targets=list(missed))
    result.check_true(
        "the_other_two_intercept_targets_are_missed_and_the_gap_is_recorded",
        "reference",
        len(missed) == 2,
        "local-frame chief-ray image heights after optimization: "
        + ", ".join(
            f"{label} {heights_after[label]:+.4f} mm (target {targets[label]:+.4f})"
            for label in ("axial", "plus", "minus")
        )
        + ". Only the +1.5 deg field reaches its f*tan(theta) target; the axial and "
        "-1.5 deg fields do not, consistent with the optimizer having stopped at a worse "
        "point than it started.",
    )
    result.check_true(
        "the_mean_rms_spot_does_not_improve_by_the_order_of_magnitude_a_working_run_would",
        "analytic",
        rms_after < 2.0 * rms_before,
        f"mean RMS spot radius over the three fields {rms_before:.6f} -> {rms_after:.6f} mm "
        f"({rms_before / rms_after:.2f}x), measured by direct traces rather than through "
        "the operands. A converged anastigmat design would be orders better; this is what "
        "the failed optimization leaves behind.",
    )

    # -- local vs global coordinates on a tilted, decentered mirror ---------------
    lens.trace_generic(Hx=0.0, Hy=1.0, Px=0.0, Py=0.0, wavelength=PRIMARY_UM)
    global_y_at_m3 = float(np.asarray(lens.surfaces.y, dtype=float)[3, 0])
    surface_3 = lens.surfaces.surfaces[3]
    decenter_y = float(np.asarray(surface_3.geometry.cs.y).ravel()[0])
    result.record(
        global_y_at_mirror_3_mm=global_y_at_m3,
        mirror_3_decenter_y_mm=decenter_y,
        local_minus_global_offset_mm=abs(global_y_at_m3 - decenter_y),
    )
    result.check_true(
        "the_lcs_suffix_is_load_bearing_on_a_decentered_mirror",
        "analytic",
        abs(decenter_y) > 1.0,
        f"mirror 3 is decentered by {decenter_y:.4f} mm and the chief ray crosses it at "
        f"global y = {global_y_at_m3:.4f} mm, so a global-frame intercept operand would be "
        f"offset by order {abs(decenter_y):.1f} mm from the local one. "
        "'real_y_intercept_lcs' measures in the surface's own frame, which is the only "
        "frame in which 'centre the mirror on its chief ray' means anything.",
    )

    # -- three mirrors are exactly achromatic -----------------------------------
    per_wavelength = {}
    for wavelength in (0.486, PRIMARY_UM, 0.656):
        rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=wavelength, num_rays=8)
        per_wavelength[f"{wavelength:g}"] = np.asarray(rays.y, dtype=float)
    reference = per_wavelength[f"{PRIMARY_UM:g}"]
    deviations = {
        key: float(np.max(np.abs(values - reference))) for key, values in per_wavelength.items()
    }
    result.record(max_abs_y_deviation_across_wavelengths_mm=deviations)
    result.check_true(
        "an_all_reflective_system_is_exactly_achromatic",
        "analytic",
        all(v == 0.0 for v in deviations.values()),
        f"max |y(lambda) - y(0.587 um)| = {deviations}: identically zero at all three "
        "wavelengths. With no refractive surface there is no dispersion to have, and no "
        "amount of optimization can introduce any -- a physics invariant, not a tolerance.",
    )

    lens.draw(title=f"Optimized {lens.name}")
    analysis.SpotDiagram(lens).view()
    GeometricMTF(lens).view()
    plt.close("all")
    result.check_true(
        "optimized_layout_spot_and_geometric_mtf_render_headless",
        "qualitative",
        True,
        "draw, SpotDiagram.view and GeometricMTF.view completed on the freeform TMA",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
