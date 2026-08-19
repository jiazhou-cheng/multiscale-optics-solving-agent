"""Advanced / "Glass Expert" -- https://www.optiland.org/tutorials/glass-expert

Repo-owned reproduction of the automatic glass-selection tutorial: a deliberately
terrible six-element f/5 design (every element N-BK7, every radius 1000 mm) handed
to `optimization.GlassExpert` with 8 radius, 11 thickness and **6 material**
variables, the last drawn from ``materials.glasses_selection(0.4, 0.7,
catalogs=['schott', 'ohara'])``.

**Adaptation.** Upstream runs ``num_neighbours=7, maxiter=100``; that is a
combinatorial search over 564 candidate glasses and does not fit a test budget. This
reproduction uses ``num_neighbours=2, maxiter=3``, which already improves the merit
function by four orders of magnitude in ~3 minutes. Both settings are recorded.

Upstream prints ``problem.initial_value`` but publishes no numbers. Validation is the
declared operand targets plus the thing that makes this optimizer distinctive:

* The operand targets are ``100 * tan(theta)`` at five field angles -- declared
  analytically, so the optimized chief-ray heights are a real external reference.
* **Glass substitution actually happens**: all six material variables move off the
  starting N-BK7, and every one of the six ends as a member of the declared
  ``glasses_selection`` list. A glass optimizer that silently left the glasses alone,
  or that wandered outside the declared catalogs, would pass a merit check and fail
  both of these.
* The merit function falls by more than three orders of magnitude from its
  1.36e+05 starting value.
* Every continuous variable ends inside its declared bounds.
* ``GlassExpert.run()`` **consumes the material variables**: the problem holds 25
  variables before the call and 19 after, exactly the six material ones having been
  removed once glasses were chosen. A caller who reuses the problem afterwards gets a
  continuous-only problem, silently.
* ``glasses_selection(0.4, 0.7, catalogs=[...])`` returns a plain list of 564 glass
  names, every one of which resolves through ``Material(name)`` -- checked on a
  sample, because a selection containing an unresolvable name would fail deep inside
  the optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t36_glass_expert",
    title="Glass Expert",
    level="advanced",
    url="https://www.optiland.org/tutorials/glass-expert",
    demonstrates=(
        "optimization.GlassExpert(problem).run(num_neighbours, maxiter, tol, "
        "callback, verbose, plot_glass_map), the 'material' variable with "
        "glass_selection=, and materials.glasses_selection(lo_um, hi_um, catalogs)."
    ),
    slow=True,
)

TARGET_FOCAL_LENGTH_MM = 100.0
FIELDS_DEG = (-14.0, -10.0, 0.0, 10.0, 14.0)
NORMALIZED_FIELDS = (-1.0, -0.714, 0.0, 0.714, 1.0)
GLASS_SURFACES = (1, 3, 4, 7, 8, 10)
RADIUS_SURFACES = (1, 2, 3, 5, 7, 9, 10, 11)
THICKNESS_BOUNDS = (
    (1, 8, 20), (2, 0.4, 5), (3, 8, 20), (4, 3, 10), (5, 10, 20), (6, 10, 20),
    (7, 3, 20), (8, 8, 20), (9, 0.4, 5), (10, 6, 20), (11, 20, 45),
)
NUM_NEIGHBOURS = 2  # upstream: 7
MAXITER = 3  # upstream: 100


def build_start_point():
    """Upstream's SixLensesStartPoint: 'very poor quality meant to be optimized'."""
    import optiland.backend as be
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=1000, thickness=25, material="N-BK7")
    lens.surfaces.add(index=2, radius=1000, thickness=5.0)
    lens.surfaces.add(index=3, radius=1000, thickness=25, material="N-BK7")
    lens.surfaces.add(index=4, radius=be.inf, thickness=25, material="N-BK7")
    lens.surfaces.add(index=5, radius=1000, thickness=25)
    lens.surfaces.add(index=6, radius=be.inf, thickness=25.0, is_stop=True)
    lens.surfaces.add(index=7, radius=-1000, thickness=25, material="N-BK7")
    lens.surfaces.add(index=8, radius=be.inf, thickness=25, material="N-BK7")
    lens.surfaces.add(index=9, radius=-1000, thickness=5.0)
    lens.surfaces.add(index=10, radius=1000, thickness=25, material="N-BK7")
    lens.surfaces.add(index=11, radius=-1000, thickness=200)
    lens.surfaces.add(index=12)
    lens.set_aperture(aperture_type="imageFNO", value=5)
    lens.fields.set_type(field_type="angle")
    for y in FIELDS_DEG:
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.4861)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    lens.wavelengths.add(value=0.6563)
    return lens


def build_problem(lens, glasses):
    from optiland import optimization

    primary = lens.wavelengths.primary_wavelength.value
    problem = optimization.OptimizationProblem()
    for index, hy in enumerate(NORMALIZED_FIELDS):
        problem.add_operand(
            operand_type="real_y_intercept_lcs",
            target=TARGET_FOCAL_LENGTH_MM
            * float(np.tan(np.deg2rad(lens.fields.y_fields[index]))),
            weight=1,
            input_data={
                "optic": lens, "surface_number": 12, "Hx": 0, "Hy": hy,
                "Px": 0, "Py": 0, "wavelength": primary,
            },
        )
    for hx, hy in lens.fields.get_field_coords():
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=10,
            input_data={
                "optic": lens, "surface_number": 12, "Hx": hx, "Hy": hy,
                "num_rays": 16, "wavelength": primary, "distribution": "uniform",
            },
        )
    for surface_number in RADIUS_SURFACES:
        problem.add_variable(
            lens, "radius", surface_number=surface_number, min_val=-500, max_val=500
        )
    for surface_number, low, high in THICKNESS_BOUNDS:
        problem.add_variable(
            lens, "thickness", surface_number=surface_number, min_val=low, max_val=high
        )
    for surface_number in GLASS_SURFACES:
        problem.add_variable(
            lens, "material", surface_number=surface_number, glass_selection=glasses
        )
    return problem


def _glass_names(lens) -> list[str]:
    return [
        str(getattr(lens.surfaces.surfaces[s].material_post, "name", "?"))
        for s in GLASS_SURFACES
    ]


def _image_heights(lens) -> list[float]:
    primary = lens.wavelengths.primary_wavelength.value
    decenter_y = float(np.asarray(lens.surfaces.surfaces[12].geometry.cs.y).ravel()[0])
    out = []
    for hy in NORMALIZED_FIELDS:
        rays = lens.trace_generic(Hx=0.0, Hy=hy, Px=0.0, Py=0.0, wavelength=primary)
        out.append(float(np.asarray(rays.y, dtype=float).ravel()[0]) - decenter_y)
    return out


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import optimization
    from optiland.materials import Material, glasses_selection

    result = TutorialResult()
    glasses = glasses_selection(0.4, 0.7, catalogs=["schott", "ohara"])
    result.record(
        glass_selection_type=type(glasses).__name__,
        num_candidate_glasses=len(glasses),
        first_candidate_glasses=sorted(glasses)[:6],
    )
    result.check_true(
        "glasses_selection_returns_a_plain_list_of_names",
        "invariant",
        isinstance(glasses, list) and all(isinstance(name, str) for name in glasses),
        f"{type(glasses).__name__} of {len(glasses)} strings from the schott and ohara "
        "catalogs over 0.4-0.7 um",
    )
    sample = sorted(glasses)[:: max(len(glasses) // 12, 1)]
    resolved = []
    for name in sample:
        try:
            resolved.append(float(np.asarray(Material(name).n(0.5876)).ravel()[0]))
        except Exception:  # noqa: BLE001 - a bad name would fail inside the optimizer
            resolved.append(float("nan"))
    result.record(sampled_glass_names=sample, sampled_indices_at_5876=resolved)
    result.check_true(
        "every_sampled_candidate_glass_resolves_to_a_real_index",
        "invariant",
        all(np.isfinite(resolved)) and all(1.3 < n < 2.2 for n in resolved),
        f"{len(sample)} sampled names all resolve through Material(name) to indices in "
        f"[{min(resolved):.4f}, {max(resolved):.4f}] at 587.6 nm",
    )

    lens = build_start_point()
    problem = build_problem(lens, glasses)
    glasses_before = _glass_names(lens)
    num_variables_before = len(problem.variables)
    merit_before = float(problem.sum_squared())
    heights_before = _image_heights(lens)
    targets = [
        TARGET_FOCAL_LENGTH_MM * float(np.tan(np.deg2rad(y))) for y in FIELDS_DEG
    ]
    lens.draw(title="Starting lens")
    lens.info()
    problem.info()
    plt.close("all")

    optimizer = optimization.GlassExpert(problem)
    optimizer.run(
        num_neighbours=NUM_NEIGHBOURS,
        maxiter=MAXITER,
        tol=1e-3,
        verbose=False,
        plot_glass_map=False,
    )
    merit_after = float(problem.sum_squared())
    glasses_after = _glass_names(lens)
    heights_after = _image_heights(lens)
    lens.draw(title="Optimized lens")
    lens.info()
    plt.close("all")

    bound_violations = []
    for variable in problem.variables:
        low, high = variable.bounds
        if low is None and high is None:
            continue
        try:
            value = float(np.asarray(variable.value).ravel()[0])
        except (TypeError, ValueError):
            continue  # a material variable's value is not a scalar
        if (low is not None and value < low - 1e-6) or (high is not None and value > high + 1e-6):
            bound_violations.append((str(variable), value, low, high))

    result.record(
        num_neighbours=NUM_NEIGHBOURS,
        upstream_num_neighbours=7,
        maxiter=MAXITER,
        upstream_maxiter=100,
        num_variables_before_run=num_variables_before,
        num_variables_after_run=len(problem.variables),
        num_operands=len(problem.operands),
        merit_before=merit_before,
        merit_after=merit_after,
        merit_reduction_factor=merit_before / merit_after,
        glasses_before=glasses_before,
        glasses_after=glasses_after,
        image_height_targets_mm=targets,
        image_heights_before_mm=heights_before,
        image_heights_after_mm=heights_after,
        num_bound_violations=len(bound_violations),
    )
    result.check_true(
        "twenty_five_variables_and_ten_operands_were_declared",
        "invariant",
        num_variables_before == 25 and len(problem.operands) == 10,
        f"{num_variables_before} variables ({len(RADIUS_SURFACES)} radii + "
        f"{len(THICKNESS_BOUNDS)} thicknesses + {len(GLASS_SURFACES)} materials) and "
        f"{len(problem.operands)} operands (5 real_y_intercept_lcs + 5 rms_spot_size)",
    )
    result.check_true(
        "glass_expert_consumes_the_material_variables",
        "invariant",
        len(problem.variables) == num_variables_before - len(GLASS_SURFACES),
        f"the problem holds {num_variables_before} variables before run() and "
        f"{len(problem.variables)} after -- exactly the {len(GLASS_SURFACES)} material "
        "variables are removed once GlassExpert has chosen its glasses. A caller that "
        "reuses the problem afterwards gets a continuous-only problem, silently.",
    )
    result.check_true(
        "the_merit_function_falls_by_more_than_three_orders_of_magnitude",
        "invariant",
        merit_after < merit_before * 1e-3,
        f"{merit_before:.6e} -> {merit_after:.6e}, a "
        f"{merit_before / merit_after:.0f}x reduction with only num_neighbours="
        f"{NUM_NEIGHBOURS}, maxiter={MAXITER} (upstream uses 7 and 100)",
    )
    result.check_true(
        "every_glass_starts_as_n_bk7",
        "invariant",
        all(name == "N-BK7" for name in glasses_before),
        f"starting glasses {glasses_before}: upstream's start point deliberately uses one "
        "glass everywhere, so any change is attributable to the optimizer",
    )
    changed = [
        (before, after)
        for before, after in zip(glasses_before, glasses_after, strict=True)
        if before != after
    ]
    result.record(num_glasses_changed=len(changed), glass_substitutions=[list(c) for c in changed])
    result.check_true(
        "glass_substitution_actually_happens",
        "analytic",
        len(changed) == len(GLASS_SURFACES),
        f"all {len(changed)} of {len(GLASS_SURFACES)} material variables moved off N-BK7: "
        f"{glasses_after}. A glass optimizer that silently left the glasses alone would "
        "still pass a merit check.",
    )
    outside = [name for name in glasses_after if name not in set(glasses)]
    result.check_true(
        "every_selected_glass_is_a_member_of_the_declared_selection",
        "analytic",
        not outside,
        f"final glasses {glasses_after} are all in the {len(glasses)}-name "
        f"glasses_selection; outside {outside or 'none'}",
    )
    result.check_true(
        "the_continuous_variables_respect_their_declared_bounds",
        "invariant",
        not bound_violations,
        f"{len(bound_violations)} violations among the radius and thickness variables",
    )
    improved_fields = sum(
        1
        for target, before, after in zip(targets, heights_before, heights_after, strict=True)
        if abs(after - target) < abs(before - target)
    )
    result.record(num_fields_closer_to_target=improved_fields)
    result.check_true(
        "the_chief_ray_heights_move_towards_their_f_tan_theta_targets",
        "reference",
        improved_fields >= 4,
        f"{improved_fields} of {len(targets)} field points end closer to their "
        f"100*tan(theta) targets. heights "
        f"{[round(v, 3) for v in heights_before]} -> {[round(v, 3) for v in heights_after]} "
        f"against targets {[round(v, 3) for v in targets]} mm.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
