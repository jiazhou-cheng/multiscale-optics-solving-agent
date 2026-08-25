"""Primary-position densities: unbiasedness first, variance second.

`couplers/patch_positions.py` exists to reduce the variance of the patch
estimator by choosing *where* patches are drawn from. A biased estimator with
lower variance would look better on every metric except the one that matters, so
the order here is deliberate and matches CHE-120's acceptance criteria:

1. the shipped uniform draw is reproduced **bitwise**, so the baseline arm of
   every comparison is the estimator that produced the committed records;
2. the weight formula ``lambda_c = P / (D pi_c)`` is checked against the
   *enumerated exact field* -- an oracle, not a self-comparison -- for every
   scheme, with the tolerance being the measured standard error;
3. the support reduction is proved exact rather than assumed: a candidate
   position is dropped only when its window holds no aperture sample at all,
   and that set is verified against `extract_patch` exhaustively;
4. only then, the predicted variance ratio.

The expensive full-scale measurement lives in
`benchmarks/probes/ray_wave/demo3_variance.py`. This file is a fast guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.boundary import ContractError, ReferencePlane
from couplers.patch import (
    CoverageBasis,
    PatchPlan,
    extract_patch,
    patch_secondary_rays,
    plan_patches,
    resolve_pad_px,
)
from couplers.patch_positions import (
    PositionDensity,
    PositionDraw,
    candidate_index_grid,
    plan_positions,
    predicted_variance_ratio,
    spectral_l1_map,
    window_energy_map,
)
from couplers.ray_to_wave import Projection, ray_to_wave

PITCH_M = 6.3e-6
WAVELENGTH_M = 7.0e-7
DOE_PLANE = ReferencePlane(name="doe", z_m=0.0)

#: Small on purpose. The enumerated reference is O(D x modes) rays and this is a
#: fast guard; the cost curve belongs in a probe. See
#: `test_patch_wft.test_enumerating_every_patch_position_is_exact_not_merely_convergent`,
#: which makes the same size argument for the same reason.
SMALL_N = 15
SMALL_PATCH_PX = 5


def _apertured_doe(n: int = SMALL_N, *, seed: int = 20260824) -> np.ndarray:
    """A phase-only mask behind a circular stop -- demo2's and demo3's shape.

    The stop is the whole point: with a full-square mask the window energy is
    nearly flat and every density here collapses onto uniform, so a test on one
    would pass while measuring nothing.
    """
    rng = np.random.default_rng(seed)
    index = (np.arange(n) - n // 2).astype(np.float64)
    yy, xx = np.meshgrid(index, index, indexing="ij")
    aperture = (xx**2 + yy**2) < (n // 2) ** 2
    return np.where(aperture, np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, n))), 0.0)


def _plan_for(centers: np.ndarray, weights: np.ndarray | None) -> PatchPlan:
    dilation = SMALL_PATCH_PX // 2
    rows, _ = candidate_index_grid(
        grid_shape=(SMALL_N, SMALL_N), patch_px=SMALL_PATCH_PX
    )
    return PatchPlan(
        centers_xy_m=centers,
        patch_px=SMALL_PATCH_PX,
        pad_px=resolve_pad_px(
            grid_n=SMALL_N,
            patch_px=SMALL_PATCH_PX,
            pad_factor=1,
            max_center_px=float(SMALL_N // 2 + dilation),
        ),
        coverage=float(rows.size**2 / (SMALL_PATCH_PX * SMALL_PATCH_PX)),
        dilation_px=dilation,
        curvature_bound_rad=0.0,
        center_weights=weights,
    )


def _field_from(centers: np.ndarray, weights: np.ndarray | None, *, secondary, rng):
    rays, diagnostics = patch_secondary_rays(
        _apertured_doe(),
        plan=_plan_for(centers, weights),
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=secondary,
        rng=rng,
    )
    field, _ = ray_to_wave(
        rays,
        grid_shape=(SMALL_N, SMALL_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        plane=DOE_PLANE,
        projection=Projection.ASM_CONSISTENT,
    )
    return np.asarray(field.u), diagnostics.as_dict()


def _enumerated_reference() -> np.ndarray:
    """Every candidate position once, every mode once: the estimator's own mean.

    Not a second implementation of the physics -- it is the *same* estimator with
    both draws enumerated, which is what makes it the right oracle for a
    question about the draw. `test_patch_wft` separately pins this construction
    against an independent float64 ASM, so the chain to an outside reference
    exists and is not re-run here.
    """
    rows, cols = candidate_index_grid(
        grid_shape=(SMALL_N, SMALL_N), patch_px=SMALL_PATCH_PX
    )
    grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")
    centers = np.column_stack([grid_x.ravel() * PITCH_M, grid_y.ravel() * PITCH_M])
    field, _ = _field_from(centers, None, secondary=None, rng=None)
    return field


# ---------------------------------------------------------------------------
# 1. The baseline arm is the shipped estimator, bitwise
# ---------------------------------------------------------------------------

def test_the_uniform_iid_draw_is_bitwise_the_one_plan_patches_makes() -> None:
    """Same generator, same seed, same positions -- and weights exactly 1.

    `plan_patches` draws rows then columns with two `rng.integers` calls. A
    `choice` over the flattened candidate grid would be a correct uniform draw
    and a different stream, and the baseline arm would quietly stop reproducing
    the committed demo3 records it is supposed to be compared against.
    """
    count = 64
    shipped = plan_patches(
        grid_shape=(SMALL_N, SMALL_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        patch_px=SMALL_PATCH_PX,
        pad_factor=1,
        patch_count=count,
        rng=np.random.default_rng(4242),
    )
    positions = plan_positions(
        doe_field=_apertured_doe(),
        patch_px=SMALL_PATCH_PX,
        sample_pitch_m=(PITCH_M, PITCH_M),
        count=count,
        density=PositionDensity.UNIFORM,
        draw=PositionDraw.IID,
        rng=np.random.default_rng(4242),
    )
    assert np.array_equal(positions.centers_xy_m, shipped.centers_xy_m)
    assert np.array_equal(positions.center_weights, np.ones(count))
    assert positions.predicted_variance_ratio == pytest.approx(1.0, abs=1e-12)


def test_a_weight_of_one_is_absent_from_the_default_path_not_a_no_op_in_it() -> None:
    """`center_weights=None` and `center_weights=ones` agree bitwise; 2x doubles.

    The first half is what keeps every committed demo2/demo3 number valid
    without re-measurement. The second half is what says the weight is wired to
    the amplitude at all -- a `None` check that passed because the weight was
    ignored entirely would satisfy the first half alone.
    """
    rng_centers = np.random.default_rng(7)
    centers = plan_positions(
        doe_field=_apertured_doe(),
        patch_px=SMALL_PATCH_PX,
        sample_pitch_m=(PITCH_M, PITCH_M),
        count=16,
        rng=rng_centers,
    ).centers_xy_m

    unweighted, diag_none = _field_from(
        centers, None, secondary=32, rng=np.random.default_rng(11)
    )
    ones, diag_ones = _field_from(
        centers, np.ones(16), secondary=32, rng=np.random.default_rng(11)
    )
    doubled, _ = _field_from(
        centers, np.full(16, 2.0), secondary=32, rng=np.random.default_rng(11)
    )

    assert np.array_equal(unweighted, ones), (
        "multiplying by exactly 1.0 changed the field, so the weight is not "
        "applied where the record says it is"
    )
    assert diag_none["mean_center_weight"] is None
    assert diag_ones["mean_center_weight"] == pytest.approx(1.0)
    assert np.allclose(doubled, 2.0 * unweighted, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. Unbiasedness against the enumerated exact field -- the anti-bias gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("density", "draw"),
    [
        (PositionDensity.UNIFORM, PositionDraw.IID),
        (PositionDensity.UNIFORM, PositionDraw.JITTERED_GRID),
        (PositionDensity.SQRT_WINDOW_ENERGY, PositionDraw.IID),
        (PositionDensity.SQRT_WINDOW_ENERGY, PositionDraw.JITTERED_GRID),
        (PositionDensity.WINDOW_ENERGY, PositionDraw.IID),
        (PositionDensity.UNIFORM, PositionDraw.STRATIFIED_CDF),
        (PositionDensity.SQRT_WINDOW_ENERGY, PositionDraw.STRATIFIED_CDF),
    ],
)
def test_every_scheme_estimates_the_enumerated_field_within_its_own_error(
    density: PositionDensity, draw: PositionDraw
) -> None:
    """The tolerance IS the measured standard error, on one scalar functional.

    Scored on ``<exact, estimate> / <exact, exact>``, whose true value is exactly
    1. One complex scalar rather than 225 pixels on purpose: a per-pixel test at
    this ray count either needs a multiple-comparison correction or silently
    becomes 225 chances to fail, and the quantity a broken importance weight
    breaks is the *scale*, which this scalar is precisely sensitive to.

    A wrong weight -- ``1/(D q)`` instead of ``P/(D pi)``, or the coverage ratio
    folded in twice -- moves this by orders of magnitude, not by sigmas.
    """
    exact = _enumerated_reference()
    denominator = float(np.vdot(exact, exact).real)

    realizations = []
    for seed in range(8):
        rng = np.random.default_rng(20260824 + seed)
        plan = plan_positions(
            doe_field=_apertured_doe(),
            patch_px=SMALL_PATCH_PX,
            sample_pitch_m=(PITCH_M, PITCH_M),
            count=120,
            density=density,
            draw=draw,
            rng=rng,
        )
        field, _ = _field_from(
            plan.centers_xy_m, plan.center_weights, secondary=256, rng=rng
        )
        realizations.append(np.vdot(exact, field) / denominator)

    projections = np.asarray(realizations)
    mean = projections.mean()
    # Standard error of the mean of the real part; the imaginary part is scored
    # the same way because a sign or conjugation error lands there and nowhere
    # else.
    for label, values in (("real", projections.real), ("imag", projections.imag)):
        target = 1.0 if label == "real" else 0.0
        standard_error = values.std(ddof=1) / np.sqrt(values.size)
        sigmas = abs(values.mean() - target) / max(standard_error, 1e-300)
        assert sigmas <= 3.0, (
            f"{density}/{draw} is biased in the {label} part: "
            f"{values.mean():.6f} against {target} at {standard_error:.2e} "
            f"standard error = {sigmas:.2f} sigma over {values.size} seeds "
            f"(mean projection {mean:.6f})"
        )


# ---------------------------------------------------------------------------
# 3. The support reduction is exact, not approximate
# ---------------------------------------------------------------------------

def test_a_dropped_position_is_one_whose_patch_is_identically_zero() -> None:
    """Exhaustive over every candidate position, both directions.

    The importance densities put zero density on positions whose window holds no
    aperture sample. That is only legitimate if those positions contribute
    exactly nothing -- otherwise the estimator is *inconsistent* and no
    reweighting recovers it. So the set is checked against `extract_patch`
    itself rather than against the argument for it.
    """
    doe = _apertured_doe()
    rows, cols = candidate_index_grid(
        grid_shape=doe.shape, patch_px=SMALL_PATCH_PX
    )
    energy = window_energy_map(doe, patch_px=SMALL_PATCH_PX)
    assert energy.shape == (rows.size, cols.size)

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            window = extract_patch(
                doe,
                center_xy_m=(float(col) * PITCH_M, float(row) * PITCH_M),
                patch_px=SMALL_PATCH_PX,
                sample_pitch_m=(PITCH_M, PITCH_M),
            )
            brute = float((np.abs(window) ** 2).sum())
            assert energy[i, j] == pytest.approx(brute, rel=1e-12, abs=1e-300), (
                f"the integral image disagrees with extract_patch at ({row}, {col})"
            )
            assert (energy[i, j] == 0.0) == (not window.any()), (
                f"position ({row}, {col}) has energy {energy[i, j]} but a "
                f"{'non-' if window.any() else ''}empty window"
            )


def test_a_density_with_a_hole_in_the_support_is_refused() -> None:
    """Zero density where a patch contributes: inconsistent, and refused.

    Injected through a doctored `spectral_l1` map because the built-in densities
    cannot produce it -- which is the point. The guard is there for the caller
    who supplies their own map, and a guard that no input can reach is not a
    guard.
    """
    doe = _apertured_doe()
    energy = window_energy_map(doe, patch_px=SMALL_PATCH_PX).ravel()
    doctored = np.sqrt(energy)
    doctored[int(np.argmax(energy))] = 0.0

    with pytest.raises(ContractError, match="inconsistent"):
        plan_positions(
            doe_field=doe,
            patch_px=SMALL_PATCH_PX,
            sample_pitch_m=(PITCH_M, PITCH_M),
            count=8,
            density=PositionDensity.SPECTRAL_L1,
            rng=np.random.default_rng(0),
            spectral_l1=doctored,
        )


def test_the_exact_optimum_needs_its_map_rather_than_guessing_one() -> None:
    with pytest.raises(ContractError, match="spectral_l1_map"):
        plan_positions(
            doe_field=_apertured_doe(),
            patch_px=SMALL_PATCH_PX,
            sample_pitch_m=(PITCH_M, PITCH_M),
            count=8,
            density=PositionDensity.SPECTRAL_L1,
            rng=np.random.default_rng(0),
        )


def test_the_cheap_proxy_tracks_the_exact_l1_norm_it_stands_in_for() -> None:
    """`sqrt(window energy)` against `||U~_c||_1`, on every candidate position.

    The proxy's derivation assumes a quasi-random phase over the filled part of
    the window. This measures how good that is rather than asserting it: the two
    maps are compared by correlation, and the *ratio of predicted variance
    ratios* is what actually matters, because a proxy that mis-ranks positions
    slightly still buys almost all of the available reduction.
    """
    doe = _apertured_doe()
    pad = resolve_pad_px(
        grid_n=SMALL_N,
        patch_px=SMALL_PATCH_PX,
        pad_factor=1,
        max_center_px=float(SMALL_N // 2 + SMALL_PATCH_PX // 2),
    )
    exact = spectral_l1_map(doe, patch_px=SMALL_PATCH_PX, pad_px=pad)
    proxy = np.sqrt(window_energy_map(doe, patch_px=SMALL_PATCH_PX).ravel())

    support = exact > 0
    assert np.array_equal(support, proxy > 0), (
        "the proxy and the exact optimum disagree about which positions "
        "contribute nothing, which is the one thing they must agree on"
    )
    correlation = np.corrcoef(exact[support], proxy[support])[0, 1]
    assert correlation > 0.95, f"proxy correlation {correlation:.4f}"

    # Both are measured against the same weight model -- the exact one -- so the
    # comparison is "how much of the available reduction does the proxy get",
    # not "how much does each predict for itself".
    available = predicted_variance_ratio(exact, exact)
    captured = predicted_variance_ratio(proxy, exact)
    assert captured > 0.9 * available, (
        f"the proxy captures {captured:.4f} of the {available:.4f} available"
    )


# ---------------------------------------------------------------------------
# 4. The prediction, and the declarations that keep it honest
# ---------------------------------------------------------------------------

def test_a_uniform_density_predicts_exactly_no_improvement() -> None:
    model = np.array([1.0, 4.0, 9.0, 0.5])
    assert predicted_variance_ratio(np.ones(4), model) == pytest.approx(1.0)


def test_the_ratio_is_maximal_at_the_weight_model_itself() -> None:
    """Cauchy-Schwarz, numerically: no density beats ``q ~ f``."""
    rng = np.random.default_rng(3)
    model = rng.uniform(0.1, 3.0, size=64)
    best = predicted_variance_ratio(model, model)
    for _ in range(200):
        other = model * rng.uniform(0.2, 5.0, size=64)
        assert predicted_variance_ratio(other, model) <= best + 1e-12


def test_the_jittered_grid_spends_one_draw_per_cell_and_weights_it_back() -> None:
    plan = plan_positions(
        doe_field=_apertured_doe(),
        patch_px=SMALL_PATCH_PX,
        sample_pitch_m=(PITCH_M, PITCH_M),
        count=100,
        draw=PositionDraw.JITTERED_GRID,
        rng=np.random.default_rng(5),
    )
    assert plan.centers_xy_m.shape[0] == 100
    # Under the uniform density every cell has mass, so no draw is skipped and
    # the weights average to 1 up to the cells' unequal counts.
    assert plan.mean_center_weight == pytest.approx(1.0, rel=0.1)
    assert np.unique(plan.centers_xy_m, axis=0).shape[0] > 80, (
        "a stratified draw that lands 20% of its samples on repeated positions "
        "is not stratifying"
    )


def test_equal_mass_strata_leave_the_importance_weights_exactly_alone() -> None:
    """``pi_c = P q_c`` for the CDF draw, so its weights match the i.i.d. ones.

    This is the property that makes stratification composable with importance
    sampling, and the equal-*area* tiling does not have it: there the between-cell
    allocation is uniform whatever the density, which cancels the importance gain.
    Measured at demo3 scale in `demo3_variance.py`; pinned here as an identity.
    """
    doe = _apertured_doe()
    common = dict(
        doe_field=doe,
        patch_px=SMALL_PATCH_PX,
        sample_pitch_m=(PITCH_M, PITCH_M),
        count=200,
        density=PositionDensity.SQRT_WINDOW_ENERGY,
    )
    iid = plan_positions(**common, draw=PositionDraw.IID, rng=np.random.default_rng(1))
    cdf = plan_positions(
        **common, draw=PositionDraw.STRATIFIED_CDF, rng=np.random.default_rng(2)
    )
    # The weight is a function of the position alone under both schemes, so the
    # same position must carry the same weight however it was reached.
    density = np.sqrt(window_energy_map(doe, patch_px=SMALL_PATCH_PX).ravel())
    q = density / density.sum()
    for plan in (iid, cdf):
        rows, cols = candidate_index_grid(
            grid_shape=doe.shape, patch_px=SMALL_PATCH_PX
        )
        row_index = np.round(plan.centers_xy_m[:, 1] / PITCH_M).astype(int) - rows[0]
        col_index = np.round(plan.centers_xy_m[:, 0] / PITCH_M).astype(int) - cols[0]
        flat = row_index * cols.size + col_index
        assert np.allclose(
            plan.center_weights, 1.0 / (density.size * q[flat]), rtol=1e-12
        ), f"{plan.draw_kind} does not carry the 1 / (D q) weight"


def test_weights_without_the_basis_that_declares_them_are_refused() -> None:
    centers = np.zeros((4, 2))
    with pytest.raises(ContractError, match="coverage_basis"):
        plan_patches(
            grid_shape=(SMALL_N, SMALL_N),
            sample_pitch_m=(PITCH_M, PITCH_M),
            patch_px=SMALL_PATCH_PX,
            centers_xy_m=centers,
            coverage_basis=CoverageBasis.UNIFORM_OVER_DILATED_APERTURE,
            center_weights=np.ones(4),
        )
    with pytest.raises(ContractError, match="center_weights"):
        plan_patches(
            grid_shape=(SMALL_N, SMALL_N),
            sample_pitch_m=(PITCH_M, PITCH_M),
            patch_px=SMALL_PATCH_PX,
            centers_xy_m=centers,
            coverage_basis=CoverageBasis.IMPORTANCE_OVER_DILATED_APERTURE,
        )


def test_a_weight_vector_that_does_not_match_the_centres_is_refused() -> None:
    """The batching bug this guard exists for.

    `run_route` slices centres per chunk. A caller that slices the centres and
    forwards the whole weight vector produces a field that is wrong in every
    chunk and plausible overall, so the length is checked at the emitter rather
    than trusted.
    """
    with pytest.raises(ContractError, match="one importance weight per centre"):
        patch_secondary_rays(
            _apertured_doe(),
            plan=_plan_for(np.zeros((4, 2)), np.ones(7)),
            sample_pitch_m=(PITCH_M, PITCH_M),
            wavelength_m=WAVELENGTH_M,
            plane=DOE_PLANE,
            secondary_count=8,
            rng=np.random.default_rng(0),
        )
