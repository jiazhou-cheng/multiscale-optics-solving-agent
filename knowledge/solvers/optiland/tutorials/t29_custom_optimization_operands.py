"""Advanced / "Custom Optimization Operands" -- https://www.optiland.org/tutorials/custom-optimization-operands

Repo-owned reproduction of the extensibility tutorial: register a user-written
``spot_ellipse_ratio`` function in ``optimization.operand.operand_registry``, then
drive a polynomial-surface singlet's ``polynomial_coeff`` variables to make its
spot diagram's major/minor axis ratio equal the golden ratio 1.618.

The target 1.618 is upstream's own declared number, so this reproduction has a
real reference despite the tutorial printing nothing:

* The **starting** singlet is rotationally symmetric, so the fitted ellipse ratio
  must be 1 to round-off. It is (1.0000000), which makes the optimization's job
  well posed and proves the operand measures what it claims.
* After optimization the ratio reaches 1.618 to better than 0.1%.
* The custom operand is genuinely a registry entry: it is callable through
  ``operand_registry`` by name, and registering it twice without ``overwrite=True``
  is rejected -- checked, because silently shadowing a built-in operand would be a
  nasty failure mode.
* **The ``input_data`` keys are the callable's parameter names.** Upstream passes
  ``{"lens": lens}`` because its function's parameter is named ``lens``, while every
  built-in operand takes ``{"optic": ...}``. Passing ``{"optic": lens}`` to this
  operand raises ``TypeError``. That is asserted, since it is the single most
  likely mistake when writing a custom operand.
* ``surface_type='polynomial'`` with ``coefficients=[]`` starts as a plain sphere,
  and the four ``coeff_index=(i, j)`` variables move it off rotational symmetry --
  which is the only way the ellipse ratio can change at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t29_custom_optimization_operands",
    title="Custom Optimization Operands",
    level="advanced",
    url="https://www.optiland.org/tutorials/custom-optimization-operands",
    demonstrates=(
        "optiland.optimization.operand.operand_registry.register(name, callable, "
        "overwrite=), surface_type='polynomial' with coefficients=[], and the "
        "'polynomial_coeff' variable with coeff_index=(i, j)."
    ),
    slow=True,
)

GOLDEN_RATIO = 1.618
WAVELENGTH_UM = 0.55


def spot_ellipse_ratio(lens):
    """Ratio of the major to minor axis of the best-fit ellipse of the spot diagram."""
    rays_out = lens.trace(
        Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=15, distribution="hexapolar"
    )
    x = np.asarray(rays_out.x, dtype=float)
    y = np.asarray(rays_out.y, dtype=float)
    cov_matrix = np.cov(np.vstack((x, y)))
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    a2, b2 = np.sort(eigenvalues)[::-1]
    return float(np.sqrt(a2) / np.sqrt(b2))


def build_singlet():
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


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, optimization
    from optiland.optimization.operand import operand_registry

    result = TutorialResult()

    # -- registration ---------------------------------------------------------
    operand_registry.register("ellipse_ratio", spot_ellipse_ratio, overwrite=True)
    registered = "ellipse_ratio" in operand_registry
    duplicate_error = ""
    try:
        operand_registry.register("ellipse_ratio", spot_ellipse_ratio)
    except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
        duplicate_error = f"{type(exc).__name__}: {exc}"
    result.record(
        operand_is_registered=bool(registered),
        duplicate_registration_error=duplicate_error,
    )
    result.check_true(
        "the_custom_operand_is_visible_in_the_registry",
        "invariant",
        bool(registered),
        "'ellipse_ratio' in operand_registry after register(..., overwrite=True)",
    )
    result.check_true(
        "re_registering_without_overwrite_is_refused",
        "invariant",
        bool(duplicate_error),
        f"register('ellipse_ratio', ...) without overwrite=True -> "
        f"{duplicate_error or 'silently accepted, which would let a custom operand shadow a built-in'}",
    )

    # -- the starting singlet is rotationally symmetric -------------------------
    lens = build_singlet()
    geometry_class = type(lens.surfaces.surfaces[1].geometry).__name__
    ratio_initial = spot_ellipse_ratio(lens)
    result.record(
        polynomial_geometry_class=geometry_class,
        ellipse_ratio_initial=ratio_initial,
    )
    result.check_true(
        "polynomial_surface_type_builds_a_polynomial_geometry",
        "invariant",
        "polynomial" in geometry_class.lower(),
        f"geometry class {geometry_class}",
    )
    result.check_close(
        "an_empty_coefficient_list_leaves_a_rotationally_symmetric_surface",
        "analytic",
        ratio_initial,
        1.0,
        rel=1e-6,
    )

    # -- input_data keys are the callable's parameter names ---------------------
    problem = optimization.OptimizationProblem()
    wrong_key_error = ""
    try:
        probe = optimization.OptimizationProblem()
        probe.add_operand(
            operand_type="ellipse_ratio",
            target=GOLDEN_RATIO,
            weight=1,
            input_data={"optic": lens},
        )
        probe.sum_squared()
    except Exception as exc:  # noqa: BLE001 - the failure mode is the evidence
        wrong_key_error = f"{type(exc).__name__}: {exc}"
    result.record(wrong_input_data_key_error=wrong_key_error)
    result.check_true(
        "input_data_keys_must_match_the_callables_parameter_names",
        "invariant",
        bool(wrong_key_error),
        f"passing the built-in operands' {{'optic': ...}} to a callable whose parameter "
        f"is named `lens` gives {wrong_key_error or 'no error, which would be worse'}. "
        "input_data is splatted as keyword arguments, so the key names are part of the "
        "custom operand's contract.",
    )

    problem.add_operand(
        operand_type="ellipse_ratio",
        target=GOLDEN_RATIO,
        weight=1,
        input_data={"lens": lens},
    )
    for i in range(2):
        for j in range(2):
            problem.add_variable(
                lens, "polynomial_coeff", surface_number=1, coeff_index=(i, j)
            )
    problem.info()
    merit_before = float(problem.sum_squared())
    optimization.OptimizerGeneric(problem).optimize(tol=1e-6)
    merit_after = float(problem.sum_squared())
    ratio_final = spot_ellipse_ratio(lens)
    coefficients = np.asarray(
        [float(np.asarray(c).ravel()[0]) for c in np.ravel(lens.surfaces.surfaces[1].geometry.c)],
        dtype=float,
    ) if hasattr(lens.surfaces.surfaces[1].geometry, "c") else np.asarray([])

    result.record(
        num_variables=len(problem.variables),
        merit_before=merit_before,
        merit_after=merit_after,
        ellipse_ratio_final=ratio_final,
        target_ellipse_ratio=GOLDEN_RATIO,
        polynomial_coefficients=coefficients,
    )
    result.check_true(
        "four_polynomial_coefficients_became_variables",
        "invariant",
        len(problem.variables) == 4,
        f"{len(problem.variables)} == 2x2 coeff_index grid",
    )
    result.check_close(
        "the_optimized_spot_reaches_the_declared_golden_ratio",
        "reference",
        ratio_final,
        GOLDEN_RATIO,
        rel=1e-3,
    )
    result.check_true(
        "the_custom_operand_drove_the_merit_function_down",
        "invariant",
        merit_after < 1e-3 * merit_before,
        f"squared residual against the 1.618 target {merit_before:.6e} -> "
        f"{merit_after:.6e}",
    )
    result.check_true(
        "the_surface_left_rotational_symmetry",
        "analytic",
        abs(ratio_final - 1.0) > 0.5,
        f"ellipse ratio {ratio_initial:.7f} -> {ratio_final:.7f}: the polynomial "
        "coefficients broke the rotational symmetry, which is the only mechanism by "
        "which this operand can move at all",
    )

    analysis.SpotDiagram(lens, num_rings=15).view()
    plt.close("all")
    result.check_true(
        "spot_diagram_of_the_elliptical_design_renders_headless",
        "qualitative",
        True,
        "analysis.SpotDiagram(lens, num_rings=15).view() completed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
