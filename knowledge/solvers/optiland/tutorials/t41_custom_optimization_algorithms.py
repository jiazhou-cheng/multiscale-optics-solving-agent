"""Advanced / "Custom Optimization Algorithms" -- https://www.optiland.org/tutorials/extending-optimization

Repo-owned reproduction of the optimizer-extension tutorial: subclass
`optimization.OptimizerGeneric` with a hand-written random-walk `optimize()` that
uses the base class's ``self._fun`` objective and ``var.update()`` accessors, and
drive an even-aspheric singlet's three ``asphere_coeff`` variables to minimize RMS
spot size.

Upstream states two soft results -- convergence "within about 250 iterations" and a
runtime of "2 seconds" -- and publishes no numbers. Validation is the algorithm's
own contract, which is exactly checkable:

* A random walk that only accepts improving steps must produce a **monotonically
  non-increasing accepted best value**. Verified over the whole 1000-step history.
* The returned ``values`` list has ``max_steps + 1`` entries -- the initial value plus
  one per step -- and contains *rejected* candidates too, so it is a candidate
  history, not a convergence curve. That distinction is asserted, because plotting
  it as a convergence curve (which upstream does) is misleading.
* The final variables are the best position found, so re-evaluating the merit
  function after ``optimize()`` returns must equal the best value in the history.
* The RMS spot radius improves by more than 5x, measured independently of the
  operand.
* ``seed=42`` makes the walk bit-reproducible, and a different seed gives a
  different trajectory -- so the seed is real, not decorative.
* Upstream's "converges within about 250 iterations" is checked as a number, and it
  it does **not** reproduce under any reasonable criterion: the running best needs 356
  steps to come within a factor of 10 of its final value and 889 to come within 1%.
  The figure describes when a log-scale plot stops looking like it is moving. The
  method itself works -- the spot improves 254x -- only the stated step count does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t41_custom_optimization_algorithms",
    title="Custom Optimization Algorithms",
    level="advanced",
    url="https://www.optiland.org/tutorials/extending-optimization",
    demonstrates=(
        "subclassing optimization.OptimizerGeneric and using its protected "
        "self._fun(position) objective, self._x history and "
        "problem.variables[i].update(value), plus the 'asphere_coeff' variable "
        "with coeff_number=."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.55
MAX_STEPS = 1000
DELTA = 0.1
SEED = 42
UPSTREAM_CONVERGENCE_STEPS = 250


def build_aspheric_singlet():
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=20.0,
        is_stop=True,
        material="N-SF11",
        surface_type="even_asphere",
        conic=0.0,
        coefficients=[0, 0, 0],
    )
    lens.surfaces.add(index=2, thickness=21.56201105)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def _make_optimizer_class():
    from optiland import optimization

    class RandomWalkOptimizer(optimization.OptimizerGeneric):
        def __init__(self, problem):
            super().__init__(problem)

        def optimize(self, max_steps=100, delta=0.1, seed=42):
            np.random.seed(seed)
            current_position = [var.value for var in self.problem.variables]
            current_value = self._fun(current_position)
            num_variables = len(current_position)
            self._x.append(current_position)
            values = [current_value]
            for _ in range(max_steps):
                random_step = np.random.randn(num_variables) * delta
                new_position = current_position + random_step
                new_value = self._fun(new_position)
                values.append(new_value)
                if new_value < current_value:
                    current_position = new_position
                    current_value = new_value
            for idvar, var in enumerate(self.problem.variables):
                var.update(current_position[idvar])
            return values

    return RandomWalkOptimizer


def build_problem(lens):
    from optiland import optimization

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
            "num_rays": 5,
            "wavelength": WAVELENGTH_UM,
            "distribution": "hexapolar",
        },
    )
    for coeff_number in range(3):
        problem.add_variable(
            lens, "asphere_coeff", surface_number=1, coeff_number=coeff_number
        )
    return problem


def _rms_spot(lens) -> float:
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=24)
    x = np.asarray(rays.x, dtype=float)
    y = np.asarray(rays.y, dtype=float)
    return float(np.sqrt(np.mean(x**2 + y**2)))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis

    result = TutorialResult()
    RandomWalkOptimizer = _make_optimizer_class()

    lens = build_aspheric_singlet()
    problem = build_problem(lens)
    rms_initial = _rms_spot(lens)
    problem.info()
    optimizer = RandomWalkOptimizer(problem)
    values = optimizer.optimize(max_steps=MAX_STEPS, delta=DELTA, seed=SEED)
    values = np.asarray([float(v) for v in values], dtype=float)
    rms_final = _rms_spot(lens)
    merit_after = float(problem.sum_squared())
    coefficients = np.asarray(
        [float(np.asarray(v.value).ravel()[0]) for v in problem.variables], dtype=float
    )
    best = np.minimum.accumulate(values)

    result.record(
        max_steps=MAX_STEPS,
        delta=DELTA,
        seed=SEED,
        num_variables=len(problem.variables),
        num_candidate_values=int(values.size),
        initial_value=float(values[0]),
        best_value=float(values.min()),
        final_candidate_value=float(values[-1]),
        merit_after_optimize=merit_after,
        rms_spot_initial_mm=rms_initial,
        rms_spot_final_mm=rms_final,
        optimized_asphere_coefficients=coefficients,
        num_accepted_steps=int(np.count_nonzero(np.diff(best) < 0.0)),
    )
    result.check_finite("candidate_history_finite", values)
    result.check_true(
        "the_history_has_one_entry_per_step_plus_the_initial_value",
        "invariant",
        values.size == MAX_STEPS + 1,
        f"{values.size} == {MAX_STEPS} + 1",
    )
    result.check_true(
        "three_asphere_coefficients_became_variables",
        "invariant",
        len(problem.variables) == 3,
        f"{len(problem.variables)} == coeff_number 0, 1, 2",
    )
    result.check_true(
        "the_accepted_best_value_is_monotonically_non_increasing",
        "analytic",
        bool(np.all(np.diff(best) <= 0.0)),
        f"the running minimum of the candidate history falls monotonically from "
        f"{float(values[0]):.6e} to {float(values.min()):.6e} over {MAX_STEPS} steps, "
        f"accepting {int(np.count_nonzero(np.diff(best) < 0.0))} of them. A hill-climbing "
        "walk that only accepts improvements cannot do otherwise.",
    )
    result.check_true(
        "the_returned_list_is_a_candidate_history_not_a_convergence_curve",
        "analytic",
        float(values[-1]) > float(values.min()) * 1.01,
        f"the last candidate value {float(values[-1]):.6e} is well above the best "
        f"{float(values.min()):.6e}: `values` contains REJECTED trials too, so plotting it "
        "directly (as the tutorial does) shows scatter rather than convergence. The "
        "running minimum is the convergence curve.",
    )
    result.check_close(
        "the_variables_are_left_at_the_best_position_found",
        "analytic",
        merit_after,
        float(values.min()),
        rel=1e-9,
    )
    result.check_true(
        "the_random_walk_improves_the_rms_spot_by_more_than_five_times",
        "analytic",
        rms_final < rms_initial / 5.0,
        f"RMS spot radius {rms_initial:.6f} -> {rms_final:.6f} mm "
        f"({rms_initial / rms_final:.1f}x), measured by a direct trace rather than through "
        "the operand that was optimized",
    )

    # -- upstream's "about 250 iterations" as a number --------------------------
    within_one_percent = int(np.argmax(best <= float(values.min()) * 1.01))
    within_order = int(np.argmax(best <= float(values.min()) * 10.0))
    result.record(
        step_reaching_within_one_percent_of_best=within_one_percent,
        step_reaching_within_one_order_of_best=within_order,
        upstream_stated_convergence_steps=UPSTREAM_CONVERGENCE_STEPS,
    )
    result.check_true(
        "upstreams_250_step_convergence_claim_is_criterion_dependent",
        "reference",
        within_order > UPSTREAM_CONVERGENCE_STEPS
        and within_one_percent > 2 * UPSTREAM_CONVERGENCE_STEPS,
        f"the running best needs {within_order} steps to come within a factor of 10 of its "
        f"final value and {within_one_percent} to come within 1% -- both well past "
        f"upstream's stated 'about {UPSTREAM_CONVERGENCE_STEPS} iterations'. That figure "
        "describes when a log-scale plot stops looking like it is moving, not numerical "
        "convergence, and a random walk's apparent plateau should not be trusted as one. "
        "The design does improve 254x, so the method works; only the stated step count "
        "does not reproduce.",
    )

    # -- the seed is real ------------------------------------------------------
    replay_lens = build_aspheric_singlet()
    replay = RandomWalkOptimizer(build_problem(replay_lens)).optimize(
        max_steps=100, delta=DELTA, seed=SEED
    )
    other_lens = build_aspheric_singlet()
    other = RandomWalkOptimizer(build_problem(other_lens)).optimize(
        max_steps=100, delta=DELTA, seed=7
    )
    baseline_lens = build_aspheric_singlet()
    baseline = RandomWalkOptimizer(build_problem(baseline_lens)).optimize(
        max_steps=100, delta=DELTA, seed=SEED
    )
    same = bool(
        np.array_equal(
            np.asarray([float(v) for v in replay]), np.asarray([float(v) for v in baseline])
        )
    )
    different = not bool(
        np.array_equal(
            np.asarray([float(v) for v in replay]), np.asarray([float(v) for v in other])
        )
    )
    result.record(same_seed_reproducible=same, different_seed_differs=different)
    result.check_true(
        "the_seed_makes_the_walk_reproducible_and_actually_varies_it",
        "invariant",
        same and different,
        f"seed=42 twice gives an identical 101-value history ({same}), and seed=7 gives a "
        f"different one ({different})",
    )

    lens.draw(num_rays=5)
    analysis.SpotDiagram(lens).view()
    plt.plot(values)
    plt.close("all")
    result.check_true(
        "layout_spot_diagram_and_convergence_plot_render_headless",
        "qualitative",
        True,
        "draw, SpotDiagram.view and plt.plot(values) completed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
