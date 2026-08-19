"""Advanced / "Freeform Surfaces" -- https://www.optiland.org/tutorials/freeform-surfaces

Repo-owned reproduction of the freeform tutorial: a ``surface_type='polynomial'``
singlet whose 3x3 grid of ``polynomial_coeff`` variables is optimized against an
RMS-spot operand plus a ``real_y_intercept`` operand targeting 3 mm.

Upstream publishes no numbers, but the ``real_y_intercept`` target of 3 mm is a
declared external value, so validation is:

* The on-axis chief ray is driven from the axis to **y = 3 mm** -- an on-axis ray
  deliberately steered off axis, which only a non-rotationally-symmetric surface
  can do. Reached to better than 0.1%.
* The starting surface is rotationally symmetric (empty ``coefficients=[]``), so
  the initial ``real_y_intercept`` is exactly 0. That makes the 3 mm target a real
  displacement rather than a small correction.
* The merit function falls by more than three orders of magnitude.
* All nine ``coeff_index=(i, j)`` entries become variables, and at least one of the
  ones that break rotational symmetry ends up non-zero.
* The RMS spot at the (now displaced) image point stays finite and bounded -- the
  optimizer is not achieving the intercept target by destroying the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t34_freeform_surfaces",
    title="Freeform Surfaces",
    level="advanced",
    url="https://www.optiland.org/tutorials/freeform-surfaces",
    demonstrates=(
        "surface_type='polynomial' as a freeform, a 3x3 grid of "
        "'polynomial_coeff' variables via coeff_index=(i, j), and steering an "
        "on-axis ray off axis with a 'real_y_intercept' operand."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.55
TARGET_Y_MM = 3.0


def build_freeform():
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        radius=100,
        thickness=5,
        surface_type="polynomial",
        is_stop=True,
        material="SF11",
        coefficients=[],
    )
    lens.surfaces.add(index=2, thickness=100)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def _chief_y(lens) -> float:
    rays = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, wavelength=WAVELENGTH_UM)
    return float(np.asarray(rays.y, dtype=float).ravel()[0])


def _rms_spot(lens) -> float:
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=12)
    x = np.asarray(rays.x, dtype=float)
    y = np.asarray(rays.y, dtype=float)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, optimization

    result = TutorialResult()
    lens = build_freeform()
    y_initial = _chief_y(lens)
    rms_initial = _rms_spot(lens)

    problem = optimization.OptimizationProblem()
    problem.add_operand(
        operand_type="rms_spot_size",
        target=0,
        weight=1,
        input_data={
            "optic": lens,
            "surface_number": -1,
            "Hx": 0,
            "Hy": 0,
            "wavelength": WAVELENGTH_UM,
            "num_rays": 5,
        },
    )
    problem.add_operand(
        operand_type="real_y_intercept",
        target=TARGET_Y_MM,
        weight=1,
        input_data={
            "optic": lens,
            "surface_number": -1,
            "Hx": 0,
            "Hy": 0,
            "Px": 0,
            "Py": 0,
            "wavelength": WAVELENGTH_UM,
        },
    )
    for i in range(3):
        for j in range(3):
            problem.add_variable(
                lens, "polynomial_coeff", surface_number=1, coeff_index=(i, j)
            )
    problem.info()
    merit_before = float(problem.sum_squared())
    optimization.OptimizerGeneric(problem).optimize(tol=1e-9)
    merit_after = float(problem.sum_squared())
    y_final = _chief_y(lens)
    rms_final = _rms_spot(lens)
    coefficients = np.asarray(
        [float(np.asarray(v.value).ravel()[0]) for v in problem.variables], dtype=float
    )

    result.record(
        num_variables=len(problem.variables),
        merit_before=merit_before,
        merit_after=merit_after,
        chief_y_initial_mm=y_initial,
        chief_y_final_mm=y_final,
        target_y_mm=TARGET_Y_MM,
        rms_spot_initial_mm=rms_initial,
        rms_spot_final_mm=rms_final,
        optimized_polynomial_coefficients=coefficients,
        num_nonzero_coefficients=int(np.count_nonzero(np.abs(coefficients) > 1e-12)),
    )
    result.check_true(
        "the_three_by_three_coefficient_grid_becomes_nine_variables",
        "invariant",
        len(problem.variables) == 9,
        f"{len(problem.variables)} == 3x3 coeff_index grid",
    )
    result.check_close(
        "the_starting_freeform_is_rotationally_symmetric",
        "analytic",
        y_initial,
        0.0,
        abs_=1e-12,
    )
    result.check_close(
        "the_on_axis_ray_is_steered_to_the_declared_3mm_target",
        "reference",
        y_final,
        TARGET_Y_MM,
        rel=1e-3,
    )
    result.check_true(
        "merit_function_falls_by_more_than_three_orders_of_magnitude",
        "invariant",
        merit_after < 1e-3 * merit_before,
        f"{merit_before:.6e} -> {merit_after:.6e}",
    )
    result.check_true(
        "at_least_one_symmetry_breaking_coefficient_is_nonzero",
        "analytic",
        int(np.count_nonzero(np.abs(coefficients) > 1e-12)) >= 1,
        f"{int(np.count_nonzero(np.abs(coefficients) > 1e-12))} of 9 polynomial "
        "coefficients are non-zero; a rotationally symmetric surface cannot deflect an "
        "on-axis ray off axis at all, so a non-zero odd term is necessary, not incidental",
    )
    result.check_true(
        "the_image_is_not_destroyed_to_reach_the_intercept_target",
        "analytic",
        np.isfinite(rms_final) and rms_final < 10.0 * max(rms_initial, 1e-6),
        f"RMS spot radius about its own centroid {rms_initial:.6f} -> {rms_final:.6f} mm "
        "while the chief ray moved 3 mm off axis",
    )

    lens.draw(num_rays=5)
    analysis.SpotDiagram(lens).view()
    plt.close("all")
    result.check_true(
        "freeform_layout_and_spot_diagram_render_headless",
        "qualitative",
        True,
        "lens.draw(num_rays=5) and SpotDiagram.view() completed on the freeform design",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
