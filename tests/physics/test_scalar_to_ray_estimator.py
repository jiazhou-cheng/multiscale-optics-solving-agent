"""R08.2: the stochastic estimator -- draw rules, variance, and reproducibility.

CHE-190. R08.1 landed the decomposition and the exhaustive enumeration, which has
no sampling error at all. This is the half that does.

Everything here rests on one formula, which is the only thing a reader has to
trust. Writing `pi_m` for the expected number of times bin `m` is drawn by the
whole scheme, unbiasedness requires `w_m = N / pi_m` -- and the three draw rules
differ only in what `pi_m` is. So the tests are organized around that: each rule's
weight is checked against its own `pi`, then all six rule x density combinations
are shown unbiased by ensemble, then the variance each buys is measured against
the closed-form prediction.

A stochastic gate that passes on one seed is not evidence
---------------------------------------------------------
The ticket's own risk. Twenty tests were moved to `slow` in the old tree precisely
because they are Monte-Carlo convergence characterizations, and a single-seed pass
is evidence about one draw. **Every acceptance number in this file is an ensemble
with a standard error or a fitted slope over four decades of ray count**, not a
point. The one thing checked on a single seed is bit-identical reproducibility,
which is a statement about determinism rather than about an estimator.

The convergence fits are marked `slow` for the reason AC6 gives: they were 40 % of
the old suite's cost, and they are deselected by default because they are
expensive, not because they are optional.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import a_random_field, propagating_only

from couplers import (
    DRAW_RULES,
    SAMPLING_DENSITIES,
    predicted_variance_ratio,
    ray_to_scalar,
    scalar_to_ray,
)
from representations import ContractError

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "couplers"

SHAPE = (24, 32)
PITCH_M = (0.40e-6, 0.35e-6)
SMALL = (16, 16)


def a_concentrated_field(shape=SHAPE, pitch=PITCH_M):
    """A Gaussian, whose spectrum is one narrow lobe -- where importance sampling wins.

    The paper reports faster convergence for spectra concentrated in a single lobe
    and comparable rates for multilobed ones, so the two fixtures here are chosen
    to sit at the two ends of that: a Gaussian and a white-noise field.
    """
    from ray_support import a_surface

    from representations import ScalarField

    y = (np.arange(shape[0]) - shape[0] // 2) * pitch[0]
    x = (np.arange(shape[1]) - shape[1] // 2) * pitch[1]
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    u = np.exp(-(grid_x**2 + grid_y**2) / (2.0e-6) ** 2).astype(np.complex128)
    return ScalarField(
        u=u, sample_pitch_m=pitch, wavelength_m=0.55e-6, reference_surface=a_surface("plane")
    )


def reconstruct(rays, *, shape=SHAPE, pitch=PITCH_M):
    return np.asarray(ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=pitch)[0].u)


def squared_errors(field, *, count, density, draw, seeds, shape=SHAPE, pitch=PITCH_M):
    """Per-draw mean squared error of the reconstruction, one entry per seed."""
    truth = propagating_only(field)
    errors = []
    for seed in seeds:
        rays, _ = scalar_to_ray(
            field, count=count, rng=np.random.default_rng(seed), density=density, draw=draw
        )
        errors.append(
            float(np.mean(np.abs(reconstruct(rays, shape=shape, pitch=pitch) - truth) ** 2))
        )
    return np.asarray(errors)


def mean_squared_error(field, **kwargs):
    """Ensemble mean of `squared_errors`."""
    return float(np.mean(squared_errors(field, **kwargs)))


def measured_ratio(field, *, count, seeds, draw="iid", shape=SHAPE, pitch=PITCH_M):
    """`(ratio, standard error)` of uniform MSE over magnitude MSE, from one ensemble.

    The standard error is propagated from both arms, because the ratio of two noisy
    means is what every variance claim in this file rests on and quoting it without
    one would be the point estimate the module docstring warns against.
    """
    arms = {
        density: squared_errors(
            field, count=count, density=density, draw=draw, seeds=seeds,
            shape=shape, pitch=pitch,
        )
        for density in ("uniform", "magnitude")
    }
    means = {k: float(v.mean()) for k, v in arms.items()}
    relative = {
        k: float(v.std(ddof=1) / np.sqrt(v.size)) / means[k] for k, v in arms.items()
    }
    ratio = means["uniform"] / means["magnitude"]
    return ratio, ratio * float(np.hypot(relative["uniform"], relative["magnitude"]))


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("draw", DRAW_RULES)
@pytest.mark.parametrize("density", SAMPLING_DENSITIES)
def test_a_declared_seed_reproduces_the_ensemble_bit_identically(
    density: str, draw: str
) -> None:
    """Criterion 1. Bit-identical, on every rule and density, in every array.

    Not "statistically the same" and not "to round-off": the core is a pure
    function of pre-drawn indices, and the draw comes from an explicitly seeded
    host generator, so determinism is structural rather than engineered. The
    directions, the amplitudes *and* the measure weights are all checked, because
    the weight is where a rule-dependent computation could sneak in
    non-determinism the geometry would not show.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    kwargs = dict(count=128, density=density, draw=draw, seed=99)
    first, first_record = scalar_to_ray(field, rng=np.random.default_rng(99), **kwargs)
    second, _ = scalar_to_ray(field, rng=np.random.default_rng(99), **kwargs)

    assert first_record.seed == 99
    assert first_record.draw == draw
    for name in ("positions_m", "directions", "amplitude", "measure_weight"):
        assert np.array_equal(
            np.asarray(getattr(first, name)), np.asarray(getattr(second, name))
        ), name


def test_two_different_seeds_give_two_different_ensembles() -> None:
    """The negative twin: a determinism test that passes on a constant is not one."""
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    first, _ = scalar_to_ray(field, count=128, rng=np.random.default_rng(1), seed=1)
    second, _ = scalar_to_ray(field, count=128, rng=np.random.default_rng(2), seed=2)
    assert not np.array_equal(np.asarray(first.directions), np.asarray(second.directions))


# ---------------------------------------------------------------------------
# 2. Every rule's weight is `N / pi`
# ---------------------------------------------------------------------------


def independent_density(field, density: str) -> np.ndarray:
    """`q` over the propagating bins, recomputed here from the field.

    Written out from the centred transform rather than read back from the coupler,
    so the weight assertions below compare against something this module derived
    and not against the quantity under test.
    """
    u = np.asarray(field.u)
    ny, nx = u.shape
    dy, dx = field.sample_pitch_m
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u))) / (ny * nx)
    direction_v, direction_u = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(ny, dy)) * field.wavelength_m,
        np.fft.fftshift(np.fft.fftfreq(nx, dx)) * field.wavelength_m,
        indexing="ij",
    )
    amplitudes = spectrum[direction_u**2 + direction_v**2 < 1.0]
    if density == "uniform":
        return np.full(amplitudes.size, 1.0 / amplitudes.size)
    magnitude = np.abs(amplitudes)
    return magnitude / magnitude.sum()


@pytest.mark.parametrize("draw", ["iid", "stratified_cdf"])
@pytest.mark.parametrize("density", SAMPLING_DENSITIES)
def test_the_emitted_weight_is_exactly_one_over_the_density(density: str, draw: str) -> None:
    """`pi_m = N q_m` for both rules, so `w_m = 1 / q_m` for both -- pinned exactly.

    This is the assertion that actually guards `w = N / pi_m`, and it is
    deterministic. An ensemble test cannot do it: a weight scaled by any constant
    `c` scales the estimator's spread by `c` as well, so the z-score of the mean
    saturates at `|truth| / SE` and is blind to `c > 3` however many draws are
    taken. See `test_every_rule_and_density_is_unbiased`.

    `q` is recomputed from the field by `independent_density`, so this compares the
    emitted weight against arithmetic rather than against the coupler.

    Stratification changes *which* bins are drawn, not what a draw is worth -- so
    the two rules must produce the same function of the bin, and both are checked
    against the same `q`.
    """
    field = a_concentrated_field()
    q = independent_density(field, density)
    rays, _ = scalar_to_ray(
        field, count=64, rng=np.random.default_rng(4), density=density, draw=draw
    )
    weights = np.asarray(rays.measure_weight)
    # Recover the bin each ray came from by matching its weight to 1/q, then check
    # the match is exact rather than approximate.
    expected = 1.0 / q
    for weight in weights:
        assert np.min(np.abs(expected - weight)) <= 1e-9 * weight, weight
    if density == "magnitude":
        assert not np.allclose(weights, weights[0])
    else:
        assert np.allclose(weights, float(q.size), rtol=1e-12)


@pytest.mark.parametrize("draw", ["iid", "stratified_cdf"])
def test_a_weight_scaled_by_a_constant_would_fail_that_assertion(draw: str) -> None:
    """The meta-test. The exact pin is only worth having if it can fail.

    Two ways `w = N / pi_m` gets written wrongly and both are multiplicative, which
    is exactly what the ensemble gates cannot see: dropping the `1/p` altogether,
    and applying it twice.
    """
    field = a_concentrated_field()
    q = independent_density(field, "magnitude")
    expected = 1.0 / q
    for factor in (2.0, 0.5, float(q.size)):
        wrong = expected * factor
        assert np.min(np.abs(expected - wrong[0])) > 1e-9 * wrong[0]


def test_the_jittered_grid_rule_cancels_the_density_and_its_weight_says_so() -> None:
    """`pi_m = N / D` whatever `q` is, so `w_m = D` for every ray.

    Equal-*area* strata with one draw each fix the between-stratum allocation to
    uniform, and only the within-stratum choice can follow the density -- so with
    strata far smaller than the scale the density varies on, the scheme degenerates
    to uniform. Measured rather than argued: the weight is the bin count exactly,
    under both densities, and the two reconstructions have identical error.
    """
    field = a_concentrated_field()
    _, modes = scalar_to_ray(field)
    bins = modes.propagating_modes

    for density in SAMPLING_DENSITIES:
        rays, record = scalar_to_ray(
            field,
            count=64,
            rng=np.random.default_rng(4),
            density=density,
            draw="jittered_grid",
        )
        assert np.allclose(np.asarray(rays.measure_weight), float(bins), rtol=1e-12)
        assert record.mean_measure_weight == pytest.approx(float(bins), rel=1e-12)

    seeds = range(4000, 4030)
    uniform = mean_squared_error(
        field, count=64, density="uniform", draw="jittered_grid", seeds=seeds
    )
    magnitude = mean_squared_error(
        field, count=64, density="magnitude", draw="jittered_grid", seeds=seeds
    )
    assert uniform == pytest.approx(magnitude, rel=1e-12)


@pytest.mark.parametrize("draw", DRAW_RULES)
@pytest.mark.parametrize("density", SAMPLING_DENSITIES)
def test_every_rule_and_density_is_unbiased(density: str, draw: str) -> None:
    """The property the weight exists for, checked as an ensemble rather than a point.

    300 independent estimates of the field at the coordinate origin, against the
    analytic value. Measured `|mean - truth| / SE`, all six combinations: 0.80 to
    1.94. The gate is 4 standard errors, which is a claim about the estimator
    rather than about one draw.

    **What this test cannot do, stated so it is not relied on.** A weight scaled by
    a constant `c` scales the spread by `c` too, so `z = (|c-1| / c) * |truth| / SE`
    and it saturates at `|truth| / SE` -- measured 1.88 here. So this gate is blind
    to a weight 2x or 1000x too large, and more draws do not help: `|truth| / SE`
    grows only as `sqrt(n)`, and a 20 %-level test would need about 3e4 draws.
    `test_the_emitted_weight_is_exactly_one_over_the_density` is the deterministic
    pin that covers that, and the variance-ratio and slope gates fail on a scaled
    weight for a different reason -- both collapse toward 1.

    What this test *does* cover is a weight that is wrong as a **function of the
    bin**, which is the shape a mis-derived `pi_m` actually has: `jittered_grid`
    weighted as `1/q` instead of `D`, for instance, biases the mean without
    changing the spread much, and that is what 4 SE catches.
    """
    field = a_random_field(shape=SMALL, sample_pitch_m=PITCH_M, seed=3)
    truth = propagating_only(field)
    origin = (SMALL[0] // 2, SMALL[1] // 2)

    estimates = []
    for seed in range(4000, 4300):
        rays, _ = scalar_to_ray(
            field, count=48, rng=np.random.default_rng(seed), density=density, draw=draw
        )
        estimates.append(complex(reconstruct(rays, shape=SMALL)[origin]))
    sample = np.asarray(estimates)

    standard_error = np.sqrt(
        (sample.real.var(ddof=1) + sample.imag.var(ddof=1)) / sample.size
    )
    z = abs(sample.mean() - truth[origin]) / standard_error
    assert z < 4.0, (density, draw, z)


# ---------------------------------------------------------------------------
# 3. The predicted variance ratio, against measurement
# ---------------------------------------------------------------------------


def test_the_closed_form_is_one_at_uniform_and_the_optimum_at_the_magnitude_density() -> None:
    """The formula, checked against arithmetic before it is checked against a draw.

    `(D sum f^2 - sum f^2) / [(sum m)(sum f^2/m) - sum f^2]` is 1 at `m` uniform by
    cancellation, and Cauchy-Schwarz puts its maximum at `m = f`. The reference
    implementation's second-moment form `D sum f^2 / [(sum m)(sum f^2/m)]` is the
    `sum f^2 -> 0` limit of it, and is reproduced here as that limit so the ported
    quantity stays checkable.
    """
    f = np.array([4.0, 1.0, 1.0, 2.0])
    self_power = float(np.sum(f**2))

    assert predicted_variance_ratio(f, np.ones_like(f)) == pytest.approx(1.0)
    exact = predicted_variance_ratio(f, f)
    assert exact == pytest.approx(
        (f.size * self_power - self_power) / (np.sum(f) ** 2 - self_power)
    )
    # ...and the reference form, as the stated limit: 4 * 22 / 64 = 1.375 against
    # 66 / 42 = 1.5714 here, a 14 % underestimate, which is why the term is kept.
    reference_form = f.size * self_power / np.sum(f) ** 2
    assert reference_form == pytest.approx(1.375, rel=1e-9)
    assert exact == pytest.approx(66.0 / 42.0, rel=1e-12)
    assert reference_form / exact == pytest.approx(0.875, rel=1e-3)

    # Scale-free in the density, which a normalization slip would break.
    assert predicted_variance_ratio(f, 17.0 * f) == pytest.approx(exact)
    # A flat spectrum has nothing to exploit.
    flat = np.ones(7)
    assert predicted_variance_ratio(flat, flat) == pytest.approx(1.0)
    # A single mode with a density on it is exact, so the ratio is infinite rather
    # than an overflow.
    single = np.array([0.0, 3.0, 0.0])
    assert predicted_variance_ratio(single, np.abs(single)) == math.inf


def test_the_prediction_is_for_the_field_and_not_for_one_point() -> None:
    """Which estimand the number describes, pinned because the two differ hugely.

    The field's mean squared error sums each mode coefficient's own variance and
    carries `sum f^2`; the variance of the estimate of the *single number*
    `S = sum U~` carries `|S|^2`. On the Gaussian fixture, whose spectrum is real
    and positive so `S = sum f`, the field ratio is 47 and the point-estimate ratio
    is about 1.7e4 -- the point estimator is very nearly exact under `p_mag` and
    the field's is not. A caller reconstructs the field, so the field is what is
    reported.
    """
    field = a_concentrated_field()
    _, record = scalar_to_ray(
        field, count=16, rng=np.random.default_rng(0), density="magnitude"
    )
    assert record.predicted_variance_ratio == pytest.approx(47.17, rel=0.01)

    q = independent_density(field, "magnitude")
    u = np.asarray(field.u)
    ny, nx = u.shape
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u))) / (ny * nx)
    direction_v, direction_u = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(ny, field.sample_pitch_m[0])) * field.wavelength_m,
        np.fft.fftshift(np.fft.fftfreq(nx, field.sample_pitch_m[1])) * field.wavelength_m,
        indexing="ij",
    )
    a = spectrum[direction_u**2 + direction_v**2 < 1.0]
    mean_square = abs(np.sum(a)) ** 2
    point_ratio = (q.size * np.sum(np.abs(a) ** 2) - mean_square) / (
        np.sum(np.abs(a) ** 2 / q) - mean_square
    )
    assert point_ratio > 1e3, point_ratio


@pytest.mark.parametrize(
    ("name", "predicted", "draws"),
    [("random", 1.2635, 60), ("concentrated", 47.168, 150)],
)
def test_the_predicted_variance_ratio_matches_measurement(
    name: str, predicted: float, draws: int
) -> None:
    """Criterion 2, on two configurations rather than the one it asks for.

    Measured as the ratio of two ensemble-mean field MSEs at 256 modes, with the
    standard error propagated from both arms, and asserted as a **z-score** rather
    than as a percentage -- because the residual is ensemble error and a percentage
    tolerance would be a number chosen to fit it:

    | field | predicted | measured | z |
    | -- | -- | -- | -- |
    | white noise, 24x32, 60 draws | 1.2635 | 1.2520 +- 0.0178 | 0.65 |
    | Gaussian, 24x32, 150 draws | 47.168 | 44.380 +- 2.661 | 1.05 |

    That the residual is ensemble error and nothing else is checkable rather than
    asserted: a separate 40-draw block on the Gaussian
    (`test_stratification_is_a_separate_lever...`) puts the same ratio at 46.9, and
    two independent blocks giving 44.4 and 46.9 for one number is what a 6 %
    standard error looks like. An earlier version of this test attributed the gap to
    a dropped analytic term instead; that was wrong, and the term is now in the
    formula.

    Two configurations because the whole content of the number is that it is
    configuration-dependent: a flat spectrum has nothing for importance sampling to
    exploit and a concentrated one has a great deal, so a test on one field could
    not tell a correct formula from a constant.
    """
    field = (
        a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
        if name == "random"
        else a_concentrated_field()
    )
    _, record = scalar_to_ray(
        field, count=16, rng=np.random.default_rng(0), density="magnitude"
    )
    assert record.predicted_variance_ratio == pytest.approx(predicted, rel=0.01)

    ratio, standard_error = measured_ratio(
        field, count=256, seeds=range(2000, 2000 + draws)
    )
    z = abs(ratio - predicted) / standard_error
    assert z < 3.0, (ratio, standard_error, z)


def test_the_uniform_density_predicts_no_gain_over_itself() -> None:
    """The baseline every ratio is measured against, and it is exactly 1."""
    field = a_concentrated_field()
    _, record = scalar_to_ray(
        field, count=16, rng=np.random.default_rng(0), density="uniform"
    )
    assert record.predicted_variance_ratio == pytest.approx(1.0, rel=1e-12)


def test_the_record_reports_the_reduction_the_scheme_realizes_not_the_one_requested() -> None:
    """`jittered_grid` cancels the density, so the record says 1.0 and not 47.

    A stamped record claiming a 47x variance reduction the run did not get would be
    the same silent wrong answer this module refuses `count=None` with a non-uniform
    density for. The realized figure is what travels.
    """
    field = a_concentrated_field()
    _, jittered = scalar_to_ray(
        field,
        count=32,
        rng=np.random.default_rng(3),
        density="magnitude",
        draw="jittered_grid",
    )
    assert jittered.density == "magnitude"
    assert jittered.draw == "jittered_grid"
    assert jittered.predicted_variance_ratio == pytest.approx(1.0, rel=1e-12)

    _, used = scalar_to_ray(
        field, count=32, rng=np.random.default_rng(3), density="magnitude", draw="iid"
    )
    assert used.predicted_variance_ratio > 40.0


def test_stratification_is_a_separate_lever_and_only_one_of_the_two_gains_is_real() -> None:
    """What each lever buys, with standard errors, and one claim declined.

    Gaussian field, 256 modes, 40 draws. Mean squared error and the standard error
    of that mean:

    | | iid | stratified_cdf |
    | -- | -- | -- |
    | uniform | 1.574e-1 +- 1.4e-2 | 1.283e-1 +- 1.1e-2 |
    | magnitude | 3.356e-3 +- 2.0e-4 | 2.189e-4 +- 6.5e-6 |

    Two gains are real and one is not, and the errors are what say which:

    * importance sampling buys **47x** over uniform i.i.d. -- many standard errors;
    * stratification on top of importance sampling buys a further **15x** -- also
      many;
    * stratification **alone**, under a uniform density, is `1.574e-1` against
      `1.283e-1`: a difference of `2.9e-2 +- 1.8e-2`, which is **1.6 sigma and is
      not claimed**. It is plausible and unmeasured on this configuration, and
      asserting it would be exactly the single-draw reasoning this file is written
      against.

    The composition is also not the product, which is the point of measuring rather
    than multiplying: stratification removes clumping, and a concentrated density
    clumps far more than a flat one does, so it has more to remove.
    """
    field = a_concentrated_field()
    seeds = range(700, 740)
    measured = {
        (density, draw): mean_squared_error(
            field, count=256, density=density, draw=draw, seeds=seeds
        )
        for density in SAMPLING_DENSITIES
        for draw in ("iid", "stratified_cdf")
    }
    assert measured[("uniform", "iid")] / measured[("magnitude", "iid")] > 20.0
    assert (
        measured[("magnitude", "iid")] / measured[("magnitude", "stratified_cdf")] > 5.0
    )
    # The one that is within noise is pinned as *not* claimed, so a later reading of
    # this file cannot mistake silence for evidence.
    uniform_gain = measured[("uniform", "iid")] / measured[("uniform", "stratified_cdf")]
    assert 0.5 < uniform_gain < 3.0, uniform_gain


# ---------------------------------------------------------------------------
# 4. Convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_estimator_converges_at_the_monte_carlo_rate() -> None:
    """Criterion 3, as a fitted slope over four decades rather than a point.

    Mean squared error over 24 draws at 64, 256, 1024 and 4096 modes on the white-
    noise field, fitted in log-log:

    | scheme | 64 | 256 | 1024 | 4096 | slope |
    | -- | -- | -- | -- | -- | -- |
    | uniform, iid | 2.37e1 | 6.18e0 | 1.51e0 | 3.94e-1 | **-0.988** |
    | magnitude, stratified_cdf | 1.73e1 | 3.15e0 | 2.91e-1 | 1.85e-2 | **-1.652** |

    `-1` is the Monte-Carlo rate (`MSE ~ 1/N`, error `~ 1/sqrt(N)`) and the i.i.d.
    arm sits on it. The stratified arm is *faster* than Monte-Carlo, which is what
    stratification buys and is the reason it is worth having: a rate improvement
    does not stop mattering as the ray budget grows, and a constant-factor one
    does.

    The slope is what is asserted, not the values. A gate on the values would pass
    on an estimator that was merely small; a gate on the rate is what fails if
    `pi_m` is wrong in a way that leaves a residual bias, because a biased
    estimator's error stops falling.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    counts = (64, 256, 1024, 4096)
    seeds = range(9000, 9024)

    iid = [
        mean_squared_error(field, count=n, density="uniform", draw="iid", seeds=seeds)
        for n in counts
    ]
    stratified = [
        mean_squared_error(
            field, count=n, density="magnitude", draw="stratified_cdf", seeds=seeds
        )
        for n in counts
    ]

    iid_slope = float(np.polyfit(np.log(counts), np.log(iid), 1)[0])
    stratified_slope = float(np.polyfit(np.log(counts), np.log(stratified), 1)[0])

    assert iid_slope == pytest.approx(-1.0, abs=0.1), iid_slope
    assert stratified_slope < iid_slope - 0.3, (stratified_slope, iid_slope)


# ---------------------------------------------------------------------------
# 5. The record, and what did not land
# ---------------------------------------------------------------------------


def test_the_draw_rule_and_the_density_travel_with_the_ensemble() -> None:
    """Criterion 4: a record can state how its rays were obtained, and what that cost."""
    field = a_concentrated_field()
    _, record = scalar_to_ray(
        field,
        count=32,
        rng=np.random.default_rng(6),
        seed=6,
        density="magnitude",
        draw="stratified_cdf",
    )
    assert record.draw == "stratified_cdf"
    assert record.density == "magnitude"
    assert record.seed == 6
    assert record.predicted_variance_ratio > 1.0
    assert record.mean_measure_weight > 0.0
    assert record.as_dict()["draw"] == "stratified_cdf"

    # An enumeration has no draw rule, and says so rather than reporting the
    # default it ignored.
    _, exhaustive = scalar_to_ray(field, draw="jittered_grid")
    assert exhaustive.draw == "exhaustive"


def test_an_unknown_draw_rule_is_refused() -> None:
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(
            field, count=8, rng=np.random.default_rng(0), draw="sobol"  # type: ignore[arg-type]
        )
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "draw"


def test_no_chunking_framework_landed() -> None:
    """Criterion 5. Chunking is the executor's concern or the caller's.

    Six classes in the reference implementation existed for what is a sampling
    function plus a diagnostics record. Checked as a walk over defined names for
    the reason `test_scalar_to_ray.py` gives: the package docstring says there is
    no chunking framework here, and a substring search would flag that sentence.
    """
    defined = {
        node.name
        for module in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    assert defined, "the walk read nothing, so it cannot fail"
    for banned in ("chunk", "stream", "PositionalAngularSampler", "LaunchGeometry"):
        assert not any(banned in name.lower() for name in defined), banned
    for avoided in (
        "PositionalAngularSampler",
        "PositionPlan",
        "LaunchGeometry",
        "ChunkWorkItem",
        "StreamingResult",
        "StreamingReconstruction",
    ):
        assert avoided not in defined, avoided
