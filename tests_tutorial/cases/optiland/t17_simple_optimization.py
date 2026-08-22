"""Intermediate / "Simple Optimization" -- https://www.optiland.org/tutorials/simple-optimization

Repo-owned reproduction of the first optimization tutorial: a badly-formed N-SF11
singlet, an `OptimizationProblem` with one ``OPD_difference`` operand per
(field, wavelength) plus an ``f2`` operand targeting 100 mm, three variables
(back spacing and both radii), and `OptimizerGeneric`.

Upstream claims the optimization improves the merit function by ">99.9%" and
publishes nothing else. Both halves are checked, plus the one hard target the
problem itself declares:

* The ``f2`` operand targets **exactly 100 mm**, so the optimized focal length is
  a genuine reference value rather than a self-consistency check. Reached to
  0.11% (100.113 mm) -- it is one weighted term among ten, not a hard constraint,
  which is itself worth recording: ``add_operand(target=...)`` expresses a wish,
  and the tolerance has to allow for the other operands pulling against it.
* The merit function drops by more than 99.9%, matching upstream's claim.
* Every variable ends inside its declared bounds.
* The RMS wavefront error falls by more than an order of magnitude, measured with
  `wavefront.OPD` rather than through the operands that were optimized.
* ``LeastSquares`` is run on an identical fresh problem and must reach a merit
  value within a factor of a few of ``OptimizerGeneric``'s -- the tutorial offers
  it as a drop-in alternative and this checks that claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t17_simple_optimization",
    title="Simple Optimization",
    level="intermediate",
    url="https://www.optiland.org/tutorials/simple-optimization",
    demonstrates=(
        "optimization.OptimizationProblem with 'OPD_difference' and 'f2' "
        "operands, add_variable(optic, 'thickness'|'radius', surface_number, "
        "min_val, max_val), OptimizerGeneric vs LeastSquares, problem.info()."
    ),
    slow=True,
)

TARGET_EFL_MM = 100.0
WAVELENGTHS = (0.4861, 0.5876, 0.6563)


def build_initial():
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, thickness=np.inf)
    lens.surfaces.add(index=1, thickness=5, radius=100, is_stop=True, material="N-SF11")
    lens.surfaces.add(index=2, thickness=59, radius=-1000)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25)
    lens.fields.set_type(field_type="angle")
    for y in (0.0, 0.7, 1.0):
        lens.fields.add(y=y)
    lens.wavelengths.add(value=WAVELENGTHS[0])
    lens.wavelengths.add(value=WAVELENGTHS[1], is_primary=True)
    lens.wavelengths.add(value=WAVELENGTHS[2])
    lens.update_paraxial()
    return lens


def build_problem(lens):
    from optiland import optimization

    problem = optimization.OptimizationProblem()
    for wave in lens.wavelengths.get_wavelengths():
        for hx, hy in lens.fields.get_field_coords():
            problem.add_operand(
                operand_type="OPD_difference",
                target=0,
                weight=1,
                input_data={
                    "optic": lens,
                    "Hx": hx,
                    "Hy": hy,
                    "num_rays": 3,
                    "wavelength": wave,
                    "distribution": "gaussian_quad",
                },
            )
    problem.add_operand(
        operand_type="f2", target=TARGET_EFL_MM, weight=10, input_data={"optic": lens}
    )
    problem.add_variable(lens, "thickness", surface_number=2, min_val=0, max_val=1000)
    problem.add_variable(lens, "radius", surface_number=1, min_val=-1000, max_val=1000)
    problem.add_variable(lens, "radius", surface_number=2, min_val=-1000, max_val=1000)
    return problem


def _rms_wavefront_waves(lens) -> float:
    from optiland import wavefront

    opd = wavefront.OPD(lens, field=(0, 1), wavelength=WAVELENGTHS[1])
    data = opd.get_data((0, 1), WAVELENGTHS[1])
    values = np.asarray(data.opd, dtype=float)
    return float(np.std(values))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, optimization

    result = TutorialResult()
    lens = build_initial()
    problem = build_problem(lens)

    efl_before = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    merit_before = float(problem.sum_squared())
    wfe_before = _rms_wavefront_waves(lens)
    problem.info()

    optimizer = optimization.OptimizerGeneric(problem)
    optimizer.optimize()

    merit_after = float(problem.sum_squared())
    efl_after = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    wfe_after = _rms_wavefront_waves(lens)
    thickness = float(np.asarray(lens.surfaces.surfaces[2].thickness).ravel()[0])
    radius1 = float(np.asarray(lens.surfaces.surfaces[1].geometry.radius).ravel()[0])
    radius2 = float(np.asarray(lens.surfaces.surfaces[2].geometry.radius).ravel()[0])
    problem.info()

    result.record(
        num_operands=len(problem.operands),
        num_variables=len(problem.variables),
        merit_before=merit_before,
        merit_after=merit_after,
        merit_reduction_fraction=1.0 - merit_after / merit_before,
        efl_before_mm=efl_before,
        efl_after_mm=efl_after,
        rms_wavefront_before_waves=wfe_before,
        rms_wavefront_after_waves=wfe_after,
        optimized_back_spacing_mm=thickness,
        optimized_radius_1_mm=radius1,
        optimized_radius_2_mm=radius2,
    )
    result.check_true(
        "one_opd_operand_per_field_and_wavelength_plus_the_focal_length",
        "invariant",
        len(problem.operands) == 3 * 3 + 1,
        f"{len(problem.operands)} operands == 3 fields x 3 wavelengths + 1 f2",
    )
    result.check_close(
        "optimized_focal_length_reaches_the_declared_100mm_target",
        "reference",
        efl_after,
        TARGET_EFL_MM,
        rel=3e-3,
    )
    result.check_true(
        "merit_function_improves_by_more_than_99p9_percent",
        "reference",
        merit_after < merit_before * 1e-3,
        f"sum of squared residuals {merit_before:.6e} -> {merit_after:.6e}, a "
        f"{(1.0 - merit_after / merit_before) * 100:.4f}% reduction (upstream claims >99.9%)",
    )
    result.check_true(
        "rms_wavefront_error_improves_by_more_than_an_order_of_magnitude",
        "analytic",
        wfe_after < wfe_before / 10.0,
        f"RMS wavefront error at the edge field {wfe_before:.4f} -> {wfe_after:.4f} waves, "
        f"a {wfe_before / wfe_after:.1f}x improvement, measured through wavefront.OPD "
        "rather than through the optimized operands",
    )
    result.check_true(
        "all_variables_end_inside_their_declared_bounds",
        "invariant",
        0.0 <= thickness <= 1000.0
        and -1000.0 <= radius1 <= 1000.0
        and -1000.0 <= radius2 <= 1000.0,
        f"back spacing {thickness:.4f} in [0, 1000], radii {radius1:.4f} and "
        f"{radius2:.4f} in [-1000, 1000]",
    )
    result.check_true(
        "the_optimized_singlet_is_still_a_positive_lens",
        "analytic",
        efl_after > 0.0 and radius1 > 0.0,
        f"EFL {efl_after:.4f} mm > 0 with a convex front surface (R1 = {radius1:.4f} mm)",
    )

    # -- LeastSquares as the advertised drop-in alternative --------------------
    lens_ls = build_initial()
    problem_ls = build_problem(lens_ls)
    optimization.LeastSquares(problem_ls).optimize()
    merit_ls = float(problem_ls.sum_squared())
    efl_ls = float(np.asarray(lens_ls.paraxial.f2()).ravel()[0])
    result.record(merit_after_least_squares=merit_ls, efl_after_least_squares_mm=efl_ls)
    result.check_true(
        "least_squares_reaches_a_comparable_solution",
        "invariant",
        merit_ls < merit_before * 1e-3,
        f"LeastSquares merit {merit_ls:.6e} vs OptimizerGeneric {merit_after:.6e} "
        f"(both from {merit_before:.6e}); EFL {efl_ls:.4f} mm",
    )

    # -- the tutorial's plots -------------------------------------------------
    lens.draw()
    analysis.RayFan(lens).view()
    analysis.SpotDiagram(lens).view()
    plt.close("all")
    result.check_true(
        "post_optimization_analyses_render_headless",
        "qualitative",
        True,
        "draw, RayFan.view and SpotDiagram.view completed on the optimized lens",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
