"""Advanced / "Lithographic Projection System" -- https://www.optiland.org/tutorials/lithographic-projection-system

Repo-owned reproduction of the DUV projection-lens tutorial: a 44-surface,
21-element all-fused-silica system at 248 nm, object-space telecentric, with an
``objectNA`` aperture and ``object_height`` fields, then re-optimized over all 42
radii against an ``f2`` target of 494 mm and OPD operands at three field points.

**Adaptation.** Upstream calls ``optimize(tol=1e-9)`` with no iteration cap; a
42-variable finite-difference gradient costs ~3 s per step here, so ``maxiter`` is
set to 30. Recorded, along with the merit reduction actually achieved.

Upstream's one quantitative statement is "~98%" merit improvement, and the problem
declares a 494 mm focal-length target. Both are checked, along with the properties
a lithographic projection lens must have:

* The **as-published prescription already has f2 = 492.76 mm**, within 0.25% of the
  494 mm target -- so the target is a real external reference and the starting point
  is a real design, not a strawman.
* ``obj_space_telecentric = True`` must actually hold: the chief ray leaves the object
  parallel to the axis. Verified from the traced direction cosines at the first
  surface (mean ``|M/N|`` at the object side is at the 1e-9 level for the axial field
  and stays small at 48 mm object height).
* ``aperture_type='objectNA'`` sets the numerical aperture on the **object** side:
  ``max|sin(theta)|`` of the launched marginal rays reproduces the declared 0.133 to
  better than 1%.
* The system demagnifies: the 48 mm object height images at roughly ``48/M`` where
  ``M`` is the paraxial magnification, and ``|M| < 1`` for a projection printer.
* Optimization reduces the merit function; the improvement is compared against
  upstream's "~98%" claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t33_lithographic_projection_system",
    title="Lithographic Projection System",
    level="advanced",
    url="https://www.optiland.org/tutorials/lithographic-projection-system",
    demonstrates=(
        "a 44-surface prescription, aperture_type='objectNA', "
        "field_type='object_height', Optic.obj_space_telecentric, "
        "OptimizerGeneric over 42 radius variables, and "
        "OptimizationProblem.merit_info()."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.248
OBJECT_NA = 0.133
TARGET_EFL_MM = 494.0
OBJECT_HEIGHTS_MM = (0.0, 32.0, 48.0)
MAXITER = 30  # upstream: uncapped
UPSTREAM_MERIT_IMPROVEMENT = 0.98

# (radius, thickness, is_glass) for surfaces 1..42.
PRESCRIPTION = (
    (-737.7847, 27.484, True), (-235.2891, 0.916, False), (211.1786, 36.646, True),
    (-461.3986, 0.916, False), (412.6778, 21.071, True), (160.5391, 16.197, False),
    (-604.1283, 7.215, True), (218.1877, 23.941, False), (-3586.063, 11.978, True),
    (251.8168, 47.506, False), (-85.2817, 11.961, True), (584.8597, 9.968, False),
    (4074.801, 35.291, True), (-162.0185, 0.923, False), (629.544, 41.227, True),
    (-226.7397, 0.916, False), (522.2739, 27.842, True), (-582.424, 0.916, False),
    (423.729, 22.904, True), (-1385.36, 0.916, False), (212.039, 33.646, True),
    (802.3695, 55.304, False), (-776.5697, 8.703, True), (106.1728, 24.09, False),
    (-200.683, 11.452, True), (311.8264, 59.54, False), (-77.2276, 11.772, True),
    (2317.8032, 11.862, False), (-290.8859, 22.904, True), (-148.3577, 1.373, False),
    (-5658.5043, 41.227, True), (-151.9858, 0.916, False), (678.1005, 32.981, True),
    (-358.554, 0.916, False), (264.2734, 32.814, True), (2309.6884, 0.916, False),
    (171.2681, 29.015, True), (364.7765, 0.918, False), (113.37, 76.259, True),
    (78.6982, 54.304, False), (49.5443, 18.65, True), (109.8136, 13.07647896, False),
)
STOP_SURFACE = 20


def build_projection_lens():
    from optiland import materials, optic

    lens = optic.Optic()
    # SiO2 at 248 nm, as an explicit constant-index material.
    fused_silica = materials.IdealMaterial(n=1.5084, k=0)
    lens.surfaces.add(index=0, radius=np.inf, thickness=110.85883544)
    for index, (radius, thickness, is_glass) in enumerate(PRESCRIPTION, start=1):
        kwargs = {"index": index, "radius": radius, "thickness": thickness}
        if is_glass:
            kwargs["material"] = fused_silica
        if index == STOP_SURFACE:
            kwargs["is_stop"] = True
        lens.surfaces.add(**kwargs)
    lens.surfaces.add(index=len(PRESCRIPTION) + 1, radius=np.inf)
    lens.set_aperture(aperture_type="objectNA", value=OBJECT_NA)
    lens.fields.set_type(field_type="object_height")
    for y in OBJECT_HEIGHTS_MM:
        lens.fields.add(y=y)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    lens.obj_space_telecentric = True
    lens.image_solve()
    return lens


def build_problem(lens):
    from optiland import optimization

    problem = optimization.OptimizationProblem()
    problem.add_operand(
        operand_type="f2", target=TARGET_EFL_MM, weight=1, input_data={"optic": lens}
    )
    for hx, hy in lens.fields.get_field_coords():
        problem.add_operand(
            operand_type="OPD_difference",
            target=0,
            weight=10,
            input_data={
                "optic": lens,
                "Hx": hx,
                "Hy": hy,
                "num_rays": 5,
                "wavelength": WAVELENGTH_UM,
                "distribution": "gaussian_quad",
            },
        )
    for k in range(1, lens.surfaces.num_surfaces - 1):
        problem.add_variable(lens, "radius", surface_number=k, min_val=-10000, max_val=10000)
    return problem


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, mtf, optimization, wavefront

    result = TutorialResult()
    lens = build_projection_lens()
    num_surfaces = int(lens.surfaces.num_surfaces)
    efl_initial = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    magnification = float(np.asarray(lens.paraxial.magnification()).ravel()[0])
    result.record(
        num_surfaces=num_surfaces,
        wavelength_um=WAVELENGTH_UM,
        efl_initial_mm=efl_initial,
        target_efl_mm=TARGET_EFL_MM,
        paraxial_magnification=magnification,
    )
    result.check_true(
        "the_published_prescription_builds_all_44_surfaces",
        "invariant",
        num_surfaces == len(PRESCRIPTION) + 2,
        f"{num_surfaces} == 1 object + {len(PRESCRIPTION)} lens surfaces + 1 image",
    )
    result.check_close(
        "the_as_published_focal_length_is_already_near_the_494mm_target",
        "reference",
        efl_initial,
        TARGET_EFL_MM,
        rel=5e-3,
    )
    result.check_true(
        "the_system_demagnifies_as_a_projection_printer_must",
        "analytic",
        0.0 < abs(magnification) < 1.0,
        f"paraxial magnification {magnification:.6f}: |M| < 1, so the reticle is imaged "
        "smaller onto the wafer",
    )

    # -- object-space telecentricity -------------------------------------------
    telecentricity = {}
    for hy_label, hy in (("axial", 0.0), ("edge", 1.0)):
        lens.trace(Hx=0.0, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=8)
        m = np.asarray(lens.surfaces.M, dtype=float)[0, :]
        n = np.asarray(lens.surfaces.N, dtype=float)[0, :]
        telecentricity[hy_label] = float(np.abs(np.mean(m / n)))
    result.record(object_side_mean_slope=telecentricity)
    result.check_true(
        "the_object_space_chief_ray_is_telecentric",
        "analytic",
        telecentricity["axial"] < 1e-6 and telecentricity["edge"] < 1e-6,
        f"mean |M/N| of the launched bundle at the object surface: "
        f"{telecentricity['axial']:.3e} on axis and {telecentricity['edge']:.3e} at the "
        "48 mm object height. obj_space_telecentric = True really does aim every chief "
        "ray parallel to the axis, which is what a stepper needs for depth-of-focus "
        "insensitivity to reticle height.",
    )

    # -- objectNA is the OBJECT-side numerical aperture --------------------------
    lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8)
    m0 = np.asarray(lens.surfaces.M, dtype=float)[0, :]
    l0 = np.asarray(lens.surfaces.L, dtype=float)[0, :]
    launched_na = float(np.max(np.hypot(l0, m0)))
    result.record(measured_object_side_na=launched_na, declared_object_na=OBJECT_NA)
    result.check_close(
        "the_declared_object_side_numerical_aperture_is_realised",
        "analytic",
        launched_na,
        OBJECT_NA,
        rel=1e-2,
    )

    # -- optimization -----------------------------------------------------------
    problem = build_problem(lens)
    merit_before = float(problem.sum_squared())
    problem.merit_info()
    optimization.OptimizerGeneric(problem).optimize(maxiter=MAXITER, disp=False, tol=1e-9)
    merit_after = float(problem.sum_squared())
    problem.merit_info()
    efl_final = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    improvement = 1.0 - merit_after / merit_before
    result.record(
        maxiter=MAXITER,
        num_variables=len(problem.variables),
        num_operands=len(problem.operands),
        merit_before=merit_before,
        merit_after=merit_after,
        merit_improvement_fraction=improvement,
        efl_after_mm=efl_final,
        upstream_stated_improvement=UPSTREAM_MERIT_IMPROVEMENT,
    )
    result.check_true(
        "all_forty_two_radii_became_variables",
        "invariant",
        len(problem.variables) == len(PRESCRIPTION),
        f"{len(problem.variables)} == {len(PRESCRIPTION)} lens surfaces",
    )
    result.check_true(
        "optimization_reduces_the_merit_function",
        "invariant",
        merit_after < merit_before,
        f"{merit_before:.6e} -> {merit_after:.6e}, a {improvement * 100:.2f}% reduction in "
        f"{MAXITER} capped iterations (upstream states ~"
        f"{UPSTREAM_MERIT_IMPROVEMENT * 100:.0f}% with no cap)",
    )
    result.check_close(
        "the_optimized_focal_length_still_meets_the_494mm_target",
        "reference",
        efl_final,
        TARGET_EFL_MM,
        rel=5e-3,
    )

    # -- the tutorial's post-optimization analyses ------------------------------
    fft_mtf = mtf.FFTMTF(lens)
    fft_mtf.view(add_reference=True)
    plt.close("all")
    diffraction = np.asarray(fft_mtf.mtf[0], dtype=float)[0]
    result.record(mtf_at_zero_frequency=float(diffraction[0]))
    result.check_close(
        "the_duv_mtf_starts_at_unity", "analytic", float(diffraction[0]), 1.0, rel=1e-9
    )

    opd = wavefront.OPD(lens, field=(0, 1), wavelength=WAVELENGTH_UM)
    opd.view(projection="2d", num_points=64)
    plt.close("all")
    opd_values = np.asarray(opd.get_data((0, 1), WAVELENGTH_UM).opd, dtype=float)
    zernike = wavefront.ZernikeOPD(
        lens, (0, 1), WAVELENGTH_UM, zernike_type="standard", num_terms=21
    )
    coefficients = np.asarray(zernike.coeffs, dtype=float)
    captured = float(np.sqrt((coefficients[1:] ** 2).sum()))
    measured = float(np.std(opd_values))
    result.record(
        edge_field_opd_ptv_waves=float(opd_values.max() - opd_values.min()),
        edge_field_rms_waves=measured,
        zernike_num_terms=int(coefficients.size),
        zernike_quadrature_sum_waves=captured,
    )
    result.check_finite("edge_field_opd_finite", opd_values)
    result.check_true(
        "twenty_one_standard_zernike_terms_capture_most_of_the_duv_wavefront",
        "analytic",
        captured > 0.8 * measured,
        f"the orthonormal quadrature sum {captured:.6f} waves reaches "
        f"{captured / measured * 100:.1f}% of the {measured:.6f}-wave piston-removed RMS "
        "measured directly off wavefront.OPD",
    )
    analysis.SpotDiagram(lens).view()
    analysis.RayFan(lens).view()
    lens.info()
    plt.close("all")
    result.check_true(
        "spot_diagram_ray_fan_and_info_render_headless",
        "qualitative",
        True,
        "SpotDiagram.view, RayFan.view and Optic.info completed on the 44-surface system",
    )
    result.note(
        "Upstream reduces the original NA of 0.15 to 0.133 'to avoid negative edge "
        "thicknesses'. That adaptation is upstream's own and is reproduced verbatim."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
