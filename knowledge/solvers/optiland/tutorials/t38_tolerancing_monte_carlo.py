"""Advanced / "Tolerancing, Monte Carlo" -- https://www.optiland.org/tutorials/tolerancing-monte-carlo

Repo-owned reproduction of the system Monte Carlo tutorial: 24 normally
distributed perturbations on the bundled `CookeTriplet` (x and y tilt at
sigma = 0.01, x and y decenter at sigma = 0.1 mm, on each of its six lens
surfaces), three operands (RMS spot size, OPD difference and real y intercept at
the edge field), and `tolerancing.monte_carlo.MonteCarlo`.

**Adaptations.** Upstream runs ``num_iterations=1000``; this reproduction runs 150
to fit the test budget, and the count is recorded. Upstream's ``DistributionSampler``
calls pass no ``seed``, and that is **not** fixable with ``np.random.seed``: the
sampler builds its own ``be.default_rng(seed)``, so ``seed=None`` is independent of
the global RNG. Each sampler is therefore given an explicit seed, and both facts are
verified below. (This differs from ``DifferentialEvolution`` in t27, which *does*
follow numpy's global RNG.)

Upstream publishes nothing but three plots. The statistics of a zero-mean
perturbation study are nonetheless fully determined, and that is the validation:

* **The nominal system is the best case for the spot size.** Every perturbation has
  zero mean, and a well-corrected triplet sits at a local minimum, so the ensemble
  mean RMS spot must exceed the nominal. Verified, along with the stronger
  statement that the nominal beats the great majority of individual trials.
* **The real-y-intercept distribution is centred on the nominal**, because tilts and
  decenters are symmetric about zero and image height is odd in them at leading
  order. The sample mean sits within a few standard errors of the nominal.
* The three operand columns are all finite over every trial and the frame has one
  row per iteration.
* Only an **explicit per-sampler seed** makes the study replayable to bit equality,
  which is what makes any of these numbers admissible as recorded evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t38_tolerancing_monte_carlo",
    title="Tolerancing, Monte Carlo",
    level="advanced",
    url="https://www.optiland.org/tutorials/tolerancing-monte-carlo",
    demonstrates=(
        "optiland.tolerancing.monte_carlo.MonteCarlo(tolerancing).run("
        "num_iterations), view_histogram/view_cdf/view_heatmap, get_results() -> "
        "DataFrame, and tolerancing.perturbation.DistributionSampler('normal', "
        "loc, scale)."
    ),
    slow=True,
)

NUM_ITERATIONS = 150  # upstream: 1000
SEED = 20260819
TILT_SIGMA_RAD = 0.01
DECENTER_SIGMA_MM = 0.1
WAVELENGTH_UM = 0.55


def build_tolerancing(seeded: bool = True):
    """The tutorial's 24-perturbation study.

    ``seeded=False`` reproduces upstream's calls verbatim (no ``seed`` argument),
    which is NOT replayable: ``DistributionSampler`` builds its own
    ``be.default_rng(seed)``, so ``seed=None`` makes it independent of
    ``np.random.seed``. ``seeded=True`` gives each sampler an explicit seed.
    """
    from optiland.samples.objectives import CookeTriplet
    from optiland.tolerancing.core import Tolerancing
    from optiland.tolerancing.perturbation import DistributionSampler

    optic = CookeTriplet()
    tolerancing = Tolerancing(optic)
    index = 0
    for k in range(1, 7):
        for kind, scale in (("tilt", TILT_SIGMA_RAD), ("decenter", DECENTER_SIGMA_MM)):
            for axis in ("x", "y"):
                sampler_seed = (SEED + index) if seeded else None
                tolerancing.add_perturbation(
                    kind,
                    DistributionSampler("normal", seed=sampler_seed, loc=0, scale=scale),
                    surface_number=k,
                    axis=axis,
                )
                index += 1
    tolerancing.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0,
            "Hy": 1,
            "wavelength": WAVELENGTH_UM,
            "num_rays": 5,
        },
        target=0,
    )
    tolerancing.add_operand(
        "OPD_difference",
        {"optic": optic, "Hx": 0, "Hy": 1, "wavelength": WAVELENGTH_UM, "num_rays": 5},
    )
    tolerancing.add_operand(
        "real_y_intercept",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0,
            "Hy": 1,
            "Px": 0,
            "Py": 0,
            "wavelength": WAVELENGTH_UM,
        },
    )
    return optic, tolerancing


def _nominal_operand_values():
    """The three operand values on the unperturbed CookeTriplet."""
    optic, tolerancing = build_tolerancing()
    return {
        str(operand.operand_type): float(np.asarray(operand.value).ravel()[0])
        for operand in tolerancing.operands
    }


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.tolerancing.monte_carlo import MonteCarlo

    result = TutorialResult()
    nominal = _nominal_operand_values()

    _, tolerancing = build_tolerancing(seeded=True)
    monte_carlo = MonteCarlo(tolerancing)
    monte_carlo.run(num_iterations=NUM_ITERATIONS)
    frame = monte_carlo.get_results()
    monte_carlo.view_histogram(kde=False)
    monte_carlo.view_cdf()
    monte_carlo.view_heatmap(vmin=-0.2, vmax=0.2, figsize=(6, 6))
    plt.close("all")

    columns = list(frame.columns)
    operand_columns = [c for c in columns if c.split(":")[0].isdigit()]
    numeric = np.asarray(frame[operand_columns], dtype=float)
    result.record(
        num_iterations=NUM_ITERATIONS,
        upstream_num_iterations=1000,
        seed=SEED,
        num_perturbations=len(tolerancing.perturbations),
        num_operands=len(tolerancing.operands),
        num_rows=int(len(frame)),
        num_columns=len(columns),
        operand_columns=operand_columns,
        nominal_operand_values=nominal,
    )
    result.check_true(
        "twenty_four_perturbations_and_three_operands_were_declared",
        "invariant",
        len(tolerancing.perturbations) == 24 and len(tolerancing.operands) == 3,
        f"{len(tolerancing.perturbations)} perturbations (6 surfaces x {{tilt x, tilt y, "
        f"decenter x, decenter y}}) and {len(tolerancing.operands)} operands",
    )
    result.check_true(
        "one_row_per_monte_carlo_iteration",
        "invariant",
        len(frame) == NUM_ITERATIONS,
        f"{len(frame)} == {NUM_ITERATIONS}",
    )
    result.check_finite("all_operand_columns_finite", numeric)

    spot_column = next(c for c in operand_columns if "spot" in c.lower())
    intercept_column = next(c for c in operand_columns if "intercept" in c.lower())
    spot = np.asarray(frame[spot_column], dtype=float)
    intercept = np.asarray(frame[intercept_column], dtype=float)
    nominal_spot = nominal["rms_spot_size"]
    nominal_intercept = nominal["real_y_intercept"]
    result.record(
        spot_mean_mm=float(spot.mean()),
        spot_std_mm=float(spot.std(ddof=1)),
        spot_min_mm=float(spot.min()),
        spot_max_mm=float(spot.max()),
        fraction_of_trials_worse_than_nominal_spot=float((spot > nominal_spot).mean()),
        intercept_mean_mm=float(intercept.mean()),
        intercept_std_mm=float(intercept.std(ddof=1)),
    )
    result.check_true(
        "the_nominal_system_is_the_best_case_for_spot_size",
        "analytic",
        float(spot.mean()) > nominal_spot,
        f"ensemble mean RMS spot {float(spot.mean()):.6f} mm exceeds the nominal "
        f"{nominal_spot:.6f} mm. Every perturbation is zero-mean and a corrected triplet "
        "sits at a local minimum, so a symmetric perturbation can only make it worse.",
    )
    result.check_true(
        "almost_every_trial_is_worse_than_nominal",
        "analytic",
        float((spot > nominal_spot).mean()) > 0.9,
        f"{float((spot > nominal_spot).mean()) * 100:.1f}% of {NUM_ITERATIONS} trials have a "
        f"larger RMS spot than the nominal {nominal_spot:.6f} mm -- the stronger form of "
        "the statement above, which a mean alone could hide",
    )
    standard_error = float(intercept.std(ddof=1)) / np.sqrt(intercept.size)
    result.record(intercept_standard_error_mm=standard_error)
    result.check_true(
        "the_image_height_distribution_stays_centred_on_the_nominal",
        "analytic",
        abs(float(intercept.mean()) - nominal_intercept) < 4.0 * standard_error,
        f"sample mean image height {float(intercept.mean()):.6f} mm against the nominal "
        f"{nominal_intercept:.6f} mm, a difference of "
        f"{abs(float(intercept.mean()) - nominal_intercept) / standard_error:.2f} standard "
        f"errors ({standard_error:.6f} mm each). Tilts and decenters are symmetric about "
        "zero and image height is odd in them at leading order, so the distribution must "
        "not be biased -- unlike the spot size, which is even and therefore biased upward.",
    )

    # -- reproducibility -------------------------------------------------------
    def _replay(seeded: bool) -> np.ndarray:
        _, tol = build_tolerancing(seeded=seeded)
        run = MonteCarlo(tol)
        run.run(num_iterations=24)
        return np.asarray(run.get_results()[operand_columns], dtype=float)

    seeded_a, seeded_b = _replay(True), _replay(True)
    np.random.seed(SEED)
    unseeded_a = _replay(False)
    np.random.seed(SEED)
    unseeded_b = _replay(False)
    seeded_identical = bool(np.array_equal(seeded_a, seeded_b))
    unseeded_identical = bool(np.array_equal(unseeded_a, unseeded_b))
    result.record(
        explicit_seed_reproducible=seeded_identical,
        upstream_call_reproducible_under_numpy_seed=unseeded_identical,
    )
    result.check_true(
        "only_an_explicit_sampler_seed_makes_the_study_replayable",
        "invariant",
        seeded_identical and not unseeded_identical,
        "DistributionSampler builds its OWN be.default_rng(seed), so seed=None -- which "
        "is what the tutorial's calls use -- is independent of np.random.seed: two runs "
        f"with np.random.seed(SEED) set first still differ ({unseeded_identical=}). "
        f"Passing seed= to each sampler makes the operand table bit-identical "
        f"({seeded_identical=}), which is what makes these numbers admissible as recorded "
        "evidence. Note this differs from DifferentialEvolution (t27), which DOES follow "
        "numpy's global RNG.",
    )
    result.check_true(
        "histogram_cdf_and_heatmap_render_headless",
        "qualitative",
        True,
        "view_histogram(kde=False), view_cdf() and view_heatmap(vmin=-0.2, vmax=0.2) "
        "completed under the Agg backend",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
