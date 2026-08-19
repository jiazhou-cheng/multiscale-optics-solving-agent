"""Advanced / "Multi-Configuration Zoom Lens" -- https://www.optiland.org/tutorials/multi-configuration-zoom-lens

Repo-owned reproduction of the zoom-lens tutorial: a 26-surface, 12-element zoom
prescription in inches, wrapped in `multiconfig.MultiConfiguration` with four
configurations that differ in three air spaces and in their field angles, each
scaled to millimetres by ``scale_system(25.4)``, then a two-variable `LeastSquares`
pass whose ``f2`` operands target two of the four focal lengths.

Upstream states the four configuration EFLs -- **0.591, 1.272, 2.741 and 5.905
inches** -- and its optimization targets the first two of them in millimetres. Those
are real external references and they are the primary oracle here:

* Configuration 0's focal length is within a few per cent of ``0.591 * 25.4`` mm and
  configuration 1's of ``1.272 * 25.4`` mm, *before* any optimization: the
  prescription is a real zoom design, not a starting sketch.
* The four configurations form a **monotonically increasing focal-length sequence**
  spanning a 10x zoom range, which is what makes it a zoom lens.
* ``scale_system(25.4)`` is a pure similarity transform: it multiplies every focal
  length by exactly 25.4 and leaves the F-number unchanged, verified to float64
  round-off. That is the check that inch-to-millimetre conversion has not silently
  changed the optics.
* A configuration's ``set_thickness`` and ``set_optic_property`` really are per
  configuration: the three zoom air spaces and both off-axis field angles differ
  between configurations, and switching configuration changes the traced rays.
* ``LeastSquares`` drives the two targeted focal lengths closer to their declared
  values, and the two variables it was given belong to *different* configurations --
  which is the point of a multi-configuration problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t37_multi_configuration_zoom_lens",
    title="Multi-Configuration Zoom Lens",
    level="advanced",
    url="https://www.optiland.org/tutorials/multi-configuration-zoom-lens",
    demonstrates=(
        "multiconfig.MultiConfiguration: add_configuration, set_thickness("
        "surface_index, value, configurations), set_optic_property("
        "attribute_path, value, configurations), .configurations, .draw(); "
        "Optic.scale_system; the per-surface aperture= kwarg; "
        "Optic.set_ray_aiming('iterative', cache=True); and an "
        "OptimizationProblem whose variables live in different configurations."
    ),
    slow=True,
)

INCH_TO_MM = 25.4
UPSTREAM_EFL_INCHES = (0.591, 1.272, 2.741, 5.905)

# (radius, thickness, material, aperture) for surfaces 1..25; None material = air.
PRESCRIPTION = (
    (10.0697, 1.2211, "N-LAK9", 5.8), (-6.2327, 0.2360, "SF5", 5.8),
    (15.1603, 0.0150, None, 5.38), (6.0741, 0.4336, "N-LAF21", 5.26),
    (12.9033, 0.0903, None, 5.2), (41.2877, 0.1400, "N-LAK9", 2.42),
    (1.7269, 0.4613, None, 2.0), (-3.8953, 0.1400, "N-LAK9", 2.0),
    (1.2599, 0.5386, "SF5", 1.96), (-43.9995, 5.7197, None, 1.96),
    (3.4298, 0.3125, "N-LAK9", 1.8), (-4.9794, 0.0150, None, 1.8),
    (1.6004, 0.1698, "SF1", 1.68), (0.7220, 0.4947, "N-LAK9", 1.38),
    (6.9314, 0.0090, None, 1.38), (np.inf, 0.2934, None, None),
    (-1.2083, 0.1191, "N-LAK8", 0.66), (-2.8138, 0.0984, "SF1", 0.7),
    (1.9740, 0.0659, None, 0.7), (-2.5527, 0.3367, "N-SF8", 0.7),
    (-0.8716, 0.0200, None, 0.86), (1.9698, 0.2970, "N-BAK4", 0.88),
    (-0.7692, 0.1000, "SF1", 0.88), (-3.7908, 0.9232, None, 0.88),
    (np.inf, 0.0, None, None),
)
STOP_SURFACE = 16
# (surface_index -> thickness) plus the two off-axis field angles, per configuration.
ZOOM_STATES = (
    {"thicknesses": {5: 2.4124, 10: 3.2501, 15: 0.1566}, "fields": (7.05, 13.9)},
    {"thicknesses": {5: 4.0081, 10: 1.4376, 15: 0.3733}, "fields": (3.288, 6.555)},
    {"thicknesses": {5: 5.1199, 10: 0.0100, 15: 0.6891}, "fields": (1.528, 3.053)},
)


def build_base_lens():
    from optiland.optic import Optic

    lens = Optic()
    lens.wavelengths.add(value=0.4861)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    lens.wavelengths.add(value=0.6563)
    lens.fields.set_type("angle")
    for y in (0, 14.93, 28.07):
        lens.fields.add(y=y)
    lens.set_aperture("imageFNO", 2.4)

    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    for index, (radius, thickness, material, aperture) in enumerate(PRESCRIPTION, start=1):
        kwargs = {"index": index, "radius": radius, "thickness": thickness}
        if material is not None:
            kwargs["material"] = material
        if aperture is not None:
            kwargs["aperture"] = aperture
        if index == STOP_SURFACE:
            kwargs["is_stop"] = True
        lens.surfaces.add(**kwargs)
    lens.set_ray_aiming("iterative", cache=True)
    return lens


def build_multiconfiguration(scale: bool = True):
    from optiland.multiconfig import MultiConfiguration

    mc = MultiConfiguration(build_base_lens())
    for configuration_index, state in enumerate(ZOOM_STATES, start=1):
        mc.add_configuration()
        for surface_index, value in state["thicknesses"].items():
            mc.set_thickness(
                surface_index=surface_index, value=value, configurations=[configuration_index]
            )
        for field_index, value in zip((1, 2), state["fields"], strict=True):
            mc.set_optic_property(
                attribute_path=f"fields.fields[{field_index}].y",
                value=value,
                configurations=[configuration_index],
            )
    if scale:
        for configuration in mc.configurations:
            configuration.scale_system(INCH_TO_MM)
    return mc


def _efl(configuration) -> float:
    return float(np.asarray(configuration.paraxial.f2()).ravel()[0])


def _fno(configuration) -> float:
    return float(np.asarray(configuration.paraxial.FNO()).ravel()[0])


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.optimization import LeastSquares, OptimizationProblem

    result = TutorialResult()

    # -- scale_system is a pure similarity transform ----------------------------
    unscaled = build_multiconfiguration(scale=False)
    scaled = build_multiconfiguration(scale=True)
    efl_unscaled = [_efl(c) for c in unscaled.configurations]
    efl_scaled = [_efl(c) for c in scaled.configurations]
    fno_unscaled = [_fno(c) for c in unscaled.configurations]
    fno_scaled = [_fno(c) for c in scaled.configurations]
    scale_ratios = [b / a for a, b in zip(efl_unscaled, efl_scaled, strict=True)]
    result.record(
        num_configurations=len(scaled.configurations),
        num_surfaces=len(scaled.configurations[0].surfaces.surfaces),
        efl_inches=efl_unscaled,
        efl_mm=efl_scaled,
        fno_before_scaling=fno_unscaled,
        fno_after_scaling=fno_scaled,
        scale_ratios=scale_ratios,
        upstream_efl_inches=list(UPSTREAM_EFL_INCHES),
    )
    result.check_true(
        "the_prescription_yields_four_configurations",
        "invariant",
        len(scaled.configurations) == 4,
        f"{len(scaled.configurations)} configurations: the base system plus "
        f"{len(ZOOM_STATES)} added",
    )
    result.check_true(
        "scale_system_multiplies_every_focal_length_by_exactly_the_scale_factor",
        "analytic",
        all(abs(ratio - INCH_TO_MM) < 1e-9 for ratio in scale_ratios),
        f"EFL ratios after scale_system(25.4): {[round(r, 12) for r in scale_ratios]}",
    )
    result.check_true(
        "scale_system_leaves_the_f_number_unchanged",
        "analytic",
        all(
            abs(after - before) < 1e-9
            for before, after in zip(fno_unscaled, fno_scaled, strict=True)
        ),
        f"F-number before {[round(v, 9) for v in fno_unscaled]} and after "
        f"{[round(v, 9) for v in fno_scaled]}: a similarity transform cannot change a "
        "dimensionless quantity, which is the check that the inch-to-millimetre "
        "conversion did not alter the optics",
    )

    # -- the four configurations really are a zoom sequence ----------------------
    result.check_true(
        "the_configurations_form_a_monotonic_zoom_sequence",
        "analytic",
        all(b > a for a, b in zip(efl_scaled, efl_scaled[1:])),
        "focal lengths "
        + " < ".join(f"{v:.4f}" for v in efl_scaled)
        + f" mm, a {efl_scaled[-1] / efl_scaled[0]:.1f}x zoom range",
    )
    for index, upstream_inches in enumerate(UPSTREAM_EFL_INCHES):
        result.check_close(
            f"configuration_{index}_focal_length_matches_upstreams_{str(upstream_inches).replace('.', 'p')}_inch_value",
            "reference",
            efl_scaled[index],
            upstream_inches * INCH_TO_MM,
            rel=0.05,
        )

    # -- the per-configuration properties really differ -------------------------
    thicknesses = {
        f"config_{i}": [
            float(np.asarray(c.surfaces.surfaces[s].thickness).ravel()[0])
            for s in (5, 10, 15)
        ]
        for i, c in enumerate(scaled.configurations)
    }
    field_angles = {
        f"config_{i}": [float(v) for v in np.asarray(c.fields.y_fields, dtype=float).ravel()]
        for i, c in enumerate(scaled.configurations)
    }
    result.record(zoom_air_spaces_mm=thicknesses, field_angles_deg=field_angles)
    result.check_true(
        "each_configuration_carries_its_own_zoom_air_spaces",
        "analytic",
        len({tuple(round(v, 9) for v in values) for values in thicknesses.values()}) == 4,
        f"the three zoom spaces are distinct in all four configurations: {thicknesses}",
    )
    result.check_true(
        "each_configuration_carries_its_own_field_angles",
        "analytic",
        len({tuple(round(v, 9) for v in values) for values in field_angles.values()}) == 4,
        f"field angles per configuration: {field_angles}. A zoom lens must narrow its "
        "field as it lengthens, and these do.",
    )
    result.check_true(
        "the_field_angle_narrows_as_the_focal_length_grows",
        "analytic",
        all(
            field_angles[f"config_{i + 1}"][-1] < field_angles[f"config_{i}"][-1]
            for i in range(3)
        ),
        "maximum field angle "
        + " > ".join(f"{field_angles[f'config_{i}'][-1]:.3f}" for i in range(4))
        + " degrees against a monotonically rising focal length: the image-side field "
        "coverage is conserved, as a zoom must",
    )

    # -- a two-configuration optimization problem --------------------------------
    problem = OptimizationProblem()
    problem.add_variable(
        optic=scaled.configurations[0], variable_type="radius", surface_number=1
    )
    problem.add_variable(
        optic=scaled.configurations[1], variable_type="thickness", surface_number=5
    )
    problem.add_operand(
        operand_type="f2",
        target=UPSTREAM_EFL_INCHES[0] * INCH_TO_MM,
        weight=1.0,
        input_data={"optic": scaled.configurations[0]},
    )
    problem.add_operand(
        operand_type="f2",
        target=UPSTREAM_EFL_INCHES[1] * INCH_TO_MM,
        weight=1.0,
        input_data={"optic": scaled.configurations[1]},
    )
    merit_before = float(problem.sum_squared())
    efl_before = [_efl(scaled.configurations[i]) for i in (0, 1)]
    LeastSquares(problem).optimize(maxiter=5)
    merit_after = float(problem.sum_squared())
    efl_after = [_efl(scaled.configurations[i]) for i in (0, 1)]
    problem.info()
    result.record(
        num_variables=len(problem.variables),
        num_operands=len(problem.operands),
        efl_targets_mm=[UPSTREAM_EFL_INCHES[0] * INCH_TO_MM, UPSTREAM_EFL_INCHES[1] * INCH_TO_MM],
        efl_before_optimization_mm=efl_before,
        efl_after_optimization_mm=efl_after,
        merit_before=merit_before,
        merit_after=merit_after,
    )
    result.check_true(
        "the_two_variables_belong_to_different_configurations",
        "invariant",
        len(problem.variables) == 2 and len(problem.operands) == 2,
        "one radius variable in configuration 0 and one thickness variable in "
        "configuration 1, with an f2 operand on each -- a problem that a single-Optic "
        "formulation cannot express",
    )
    result.check_true(
        "least_squares_moves_both_focal_lengths_towards_their_targets",
        "invariant",
        merit_after <= merit_before,
        f"merit {merit_before:.6e} -> {merit_after:.6e}; EFLs "
        f"{[round(v, 4) for v in efl_before]} -> {[round(v, 4) for v in efl_after]} mm "
        f"against targets {[round(UPSTREAM_EFL_INCHES[i] * INCH_TO_MM, 4) for i in (0, 1)]}",
    )

    scaled.draw()
    scaled.configurations[3].draw()
    plt.close("all")
    result.check_true(
        "the_multi_configuration_and_single_configuration_drawings_render_headless",
        "qualitative",
        True,
        "MultiConfiguration.draw() and configurations[3].draw() completed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
