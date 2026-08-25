"""Where the primary (patch-centre) positions come from, and what that costs.

The patch estimator draws twice. Secondary **directions** are drawn by spectrum
magnitude -- importance sampling -- while primary **positions** are drawn
uniformly over the dilated aperture. :class:`couplers.patch.CoverageBasis` and
:class:`couplers.cascade.PrimarySampling` both flag that asymmetry as inherited
from the reference implementation rather than argued for. This module supplies
the missing argument, and it turns out to have a closed form.

The variance-optimal density, derived
-------------------------------------
Write the position axis as a finite sum over the ``D`` candidate draw positions.
For any density ``q`` on those positions the unbiased estimator is

    F_hat = (1/P) sum_p lambda_{c_p} g_{c_p},    lambda_c = 1 / (D q_c)

where ``g_c`` is patch ``c``'s contribution and ``lambda`` is the importance
weight that makes ``E[F_hat] = (1/D) sum_c g_c`` for every ``q`` with support
covering ``{g_c != 0}``. Its second moment is ``(1/P) sum_c f_c^2 / q_c`` with
``f_c^2 = E|g_c|^2``, and Cauchy-Schwarz gives the minimum at

    q_c  proportional to  f_c,

with value ``(sum_c f_c)^2 / P``. That is the same optimum, by the same
argument, that makes ``p_mag`` the right density on the direction axis -- so the
asymmetry is not a difference of principle. It is the *same* principle applied
to one axis and not the other.

What ``f_c`` is for this estimator
----------------------------------
Because the direction axis already uses ``p = |U~| / ||U~||_1``, each secondary
ray carries amplitude ``U~[m] / p[m]`` whose **modulus is constant within a
patch** and equal to ``||U~_c||_1``. So the per-ray second moment is
``||U~_c||_1^2`` and

    f_c = ||U~_c||_1,   the L1 norm of patch c's own angular spectrum.

That is exact and it is expensive: one padded transform per candidate position,
``D = 90601`` of them at demo3's geometry. It also has a closed-form
approximation. For a phase-only mask the patch is a quasi-random phase over the
``n_c`` aperture samples that fall inside its window, its spectrum spreads over
all ``M ~ n_pad^2`` modes with ``|U~| ~ sqrt(n_c) / n_pad^2`` each, so

    f_c  ~  sqrt(n_c)  =  sqrt(window energy at c),

which is one integral image over the aperture and costs nothing. Both are
offered: :func:`window_energy_map` is the cheap analytic proxy and
:func:`spectral_l1_map` is the exact quantity the proxy approximates, so the
approximation is checkable rather than assumed.

The size of the prize, and why it is not larger
-----------------------------------------------
``predicted_variance_ratio`` evaluates

    D sum_c f_c^2 / [ (sum_c m_c) (sum_c f_c^2 / m_c) ]

for the density ``m`` in use. It is 1 when ``m`` is uniform and maximal at
``m = f``. For a DOE whose amplitude is **flat inside its aperture** -- demo2's
and demo3's are, both being phase-only masks behind a circular stop -- the only
spatial non-uniformity left is the taper at the rim of the dilated draw region,
so the ratio is a small number rather than the 4.96x that spectral concentration
buys on the direction axis. **That is the answer to the asymmetry question, and
it is a quantitative one**: uniform positions are nearly optimal here *because*
the spatial amplitude is flat, while uniform directions would not be, because
the spectral amplitude is not. Neither statement transfers to a DOE with
structured transmission, and the ratio is reported per configuration rather
than asserted once.

Stratification is the other lever
---------------------------------
Importance sampling changes which positions are drawn; stratification changes
how a fixed number of draws is spread over them. They compose, they are both
unbiased for any ``q``, and neither touches the direction axis. A jittered grid
over the draw square is used rather than a systematic sweep of the density's
CDF: the CDF's ordering is row-major, so a systematic sweep of it stratifies
one axis and leaves the other to chance.

Every scheme here reduces to one formula, which is the only thing a reader has
to trust. Writing ``pi_c`` for the **expected number of times position ``c`` is
drawn** by the whole scheme,

    E[F_hat] = (1/P) sum_c pi_c lambda_c g_c  ==  (1/D) sum_c g_c
      =>  lambda_c = P / (D pi_c).

For i.i.d. draws ``pi_c = P q_c`` and this is the textbook ``1 / (D q_c)``. For
one draw per stratum ``pi_c`` is the *within-stratum* conditional alone, so the
weight carries a factor ``P`` and the stratum's own mass -- which is the whole
content of stratification: every stratum contributes exactly one draw whatever
its mass, and the weight puts the mass back. ``pi`` is the marginal the scheme
actually realizes, not the density it was built from, and ``P`` is the number of
draws actually returned, because that is what the ``one_over_n`` normalization
downstream will divide by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from core.boundary import ContractCode, ContractError

__all__ = [
    "PositionDensity",
    "PositionDraw",
    "PositionPlan",
    "candidate_index_grid",
    "plan_positions",
    "predicted_variance_ratio",
    "spectral_l1_map",
    "window_energy_map",
    "window_sample_count_map",
]


class PositionDensity(StrEnum):
    """Which density the primary positions are drawn from."""

    #: What the reference implementation does, and the baseline every ratio
    #: here is measured against. Weights are exactly ``1.0``.
    UNIFORM = "uniform"
    #: ``q ~ sqrt(window energy)``: the closed-form approximation to the
    #: variance-optimal density for a phase-only mask, from one integral image.
    SQRT_WINDOW_ENERGY = "sqrt_window_energy"
    #: ``q ~ window energy``. Offered because the ``sqrt`` in the optimum comes
    #: from a quasi-random-phase argument that a strongly focusing patch
    #: violates -- there the patch contribution grows like the energy, not its
    #: root -- so the exponent is a measurable rather than a settled question.
    WINDOW_ENERGY = "window_energy"
    #: ``q ~ ||U~_c||_1``, the exact optimum. Needs a precomputed map from
    #: :func:`spectral_l1_map`; there is no cheap route to it.
    SPECTRAL_L1 = "spectral_l1"


class PositionDraw(StrEnum):
    """How the ``P`` draws are spread over the density."""

    #: Independent draws. What the reference implementation does.
    IID = "iid"
    #: One draw per cell of a near-square tiling of the draw region, each cell
    #: sampled from the density restricted to it. Unbiased for any density, and
    #: the weight carries the cell's own mass.
    #:
    #: **Equal-area strata cancel importance sampling**, which is measured rather
    #: than argued: cells have equal *area* and one draw each, so the between-cell
    #: allocation is uniform by construction and only the within-cell choice
    #: follows the density. Since the density varies on the scale of the aperture
    #: rim and the cells are far smaller, almost all of the variation is between
    #: cells and the scheme degenerates to uniform. Kept because that
    #: cancellation is a real property worth having recorded, not because it is
    #: the way to combine the two levers.
    JITTERED_GRID = "jittered_grid"
    #: Equal-**mass** strata: one draw from each of ``P`` equal-probability
    #: intervals of the density's CDF. This is the way to combine stratification
    #: with importance sampling -- ``pi_c = P q_c`` exactly, the same weight as
    #: the i.i.d. importance draw, with the clumping removed.
    STRATIFIED_CDF = "stratified_cdf"


@dataclass(frozen=True)
class PositionPlan:
    """``P`` primary positions, their importance weights, and the audit trail.

    ``center_weights`` is ``lambda_c = P / (D * pi_c)`` and multiplies each
    patch's emitted amplitudes. It is exactly ``1.0`` for every draw of the
    uniform i.i.d. scheme, which is how the default path stays bitwise what it
    was.
    """

    centers_xy_m: np.ndarray[Any, Any]
    center_weights: np.ndarray[Any, Any]
    density_kind: str
    draw_kind: str
    #: ``D``, the number of candidate positions. The coverage correction
    #: ``A_draw / A_patch`` counts these, so a scheme that changes ``D`` changes
    #: the correction and this is the number to check it against.
    draw_positions: int
    #: Candidate positions carrying non-zero density.
    support_positions: int
    #: Total power the density's support accounts for, as a fraction of the
    #: power over all ``D`` candidates. **1.0 is a correctness condition**, not
    #: a diagnostic: a density that is zero where a patch contributes makes the
    #: estimator inconsistent, and no reweighting recovers it.
    support_power_fraction: float
    #: Candidate positions whose window holds no aperture sample at all. Their
    #: patches are identically zero, so excluding them is exact rather than
    #: approximate -- and under the uniform density they are drawn anyway and
    #: spend their whole secondary budget on zero-amplitude rays.
    empty_positions: int
    #: ``D sum f^2 / [(sum m)(sum f^2/m)]``: the variance reduction this
    #: density predicts against uniform, under the ``f`` model named in
    #: ``variance_model``. A prediction to be measured, not a result.
    predicted_variance_ratio: float
    variance_model: str
    #: Mean of ``center_weights``. Under the uniform density it is 1; under an
    #: importance density it is the fraction of ``D`` the support covers, and a
    #: value far from that is a bug in the weights.
    mean_center_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "density_kind": self.density_kind,
            "draw_kind": self.draw_kind,
            "draw_positions": self.draw_positions,
            "support_positions": self.support_positions,
            "support_power_fraction": self.support_power_fraction,
            "empty_positions": self.empty_positions,
            "predicted_variance_ratio": self.predicted_variance_ratio,
            "variance_model": self.variance_model,
            "mean_center_weight": self.mean_center_weight,
            "patch_count": int(np.asarray(self.centers_xy_m).shape[0]),
        }


def candidate_index_grid(
    *, grid_shape: tuple[int, int], patch_px: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The row and column indices :func:`couplers.patch.plan_patches` draws from.

    Duplicating that range would be the fourth half-sample bug in this family,
    so it is derived here once and both the density maps and the draw use it:
    ``-(n // 2 + dilation) .. n // 2 + dilation`` inclusive, on the plane's own
    origin rule, with ``dilation = patch_px // 2``.
    """
    if patch_px <= 0 or patch_px % 2 == 0:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"patch_px={patch_px} must be positive and odd",
            declaration="patch_px",
        )
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dilation = patch_px // 2
    rows = np.arange(-(ny // 2 + dilation), ny // 2 + dilation + 1, dtype=np.int64)
    cols = np.arange(-(nx // 2 + dilation), nx // 2 + dilation + 1, dtype=np.int64)
    return rows, cols


def window_sample_count_map(
    doe_field: np.ndarray[Any, Any], *, patch_px: int
) -> np.ndarray[Any, Any]:
    """How many **non-zero** DOE samples each candidate position's window holds.

    Separate from :func:`window_energy_map`, and the separation is load-bearing.
    The energy is a sum of floats and an empty window's four-corner difference
    comes out at ``+-1e-13`` rather than at zero -- ``|exp(i phi)|`` is not
    exactly 1, so the cumulative sums carry roundoff. A density built from that
    would put a tiny *positive* mass on positions whose patch is identically
    zero, spend draws there, and make ``support_positions`` a number about
    floating point rather than about the aperture.

    This count is a sum of ones, exact in float64 to 2^53, so ``count == 0`` is
    an exact statement about the window and is what decides the support.
    """
    return _box_sum(
        (np.asarray(doe_field) != 0).astype(np.float64), patch_px=patch_px
    )


def window_energy_map(
    doe_field: np.ndarray[Any, Any], *, patch_px: int
) -> np.ndarray[Any, Any]:
    """``sum |U|^2`` over each candidate position's window.

    An integral image rather than an FFT convolution: the sum is over an integer
    box, and a transform-based convolution would add its own roundoff and a
    wraparound rule to a quantity the density is built from.

    Exactly zero wherever the window holds no non-zero sample -- decided by
    :func:`window_sample_count_map` rather than by the float sum, for the reason
    given there. Inside the support the value is the float sum as computed.

    Returned on the ``(len(rows), len(cols))`` candidate grid of
    :func:`candidate_index_grid`, matching ``extract_patch``'s window: rows
    ``[cy - patch_px // 2, cy - patch_px // 2 + patch_px)`` for
    ``cy = row_index + ny // 2``, clipped to the array with zero continuation.
    """
    array = np.asarray(doe_field)
    energy = _box_sum(np.abs(array).astype(np.float64) ** 2, patch_px=patch_px)
    occupied = window_sample_count_map(array, patch_px=patch_px) > 0
    return np.where(occupied, np.maximum(energy, 0.0), 0.0)


def _box_sum(
    values: np.ndarray[Any, Any], *, patch_px: int
) -> np.ndarray[Any, Any]:
    """Sum of ``values`` over every candidate window, by integral image."""
    if values.ndim != 2:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"the DOE field must be 2-D, got shape {values.shape}",
            declaration="doe_field",
        )
    ny, nx = values.shape
    rows, cols = candidate_index_grid(grid_shape=(ny, nx), patch_px=patch_px)
    # Zero-padded integral image, so the clipped windows at the rim need no
    # special case: sum over [y0, y1) x [x0, x1) is the usual four-corner
    # difference once the bounds are clamped into the padded frame.
    integral = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(values, axis=0), axis=1)

    half = patch_px // 2
    y0 = np.clip(rows + ny // 2 - half, 0, ny)
    y1 = np.clip(rows + ny // 2 - half + patch_px, 0, ny)
    x0 = np.clip(cols + nx // 2 - half, 0, nx)
    x1 = np.clip(cols + nx // 2 - half + patch_px, 0, nx)
    return (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )


def spectral_l1_map(
    doe_field: np.ndarray[Any, Any],
    *,
    patch_px: int,
    pad_px: int,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray[Any, Any]:
    """``||U~_c||_1`` for candidate positions ``start:stop``, flattened row-major.

    The exact ``f_c`` of the module docstring, and the reason a proxy exists: it
    is one ``pad_px^2`` transform per candidate position, ``D`` of them, which
    at demo3's geometry is thirty times the transform work of a whole 60 M-ray
    run. Sharded through ``start``/``stop`` so a caller can keep each shard
    inside a command timeout and concatenate.

    Deliberately reuses :func:`couplers.patch.extract_patch` and the emitter's
    own transform normalization rather than reimplementing either. What this
    returns has to be the L1 norm of the spectrum *the emitter will actually
    draw from*, or it is a map of something else.
    """
    from couplers.patch import extract_patch  # circular at module scope

    array = np.asarray(doe_field)
    ny, nx = array.shape
    rows, cols = candidate_index_grid(grid_shape=(ny, nx), patch_px=patch_px)
    total = rows.size * cols.size
    stop = total if stop is None else min(int(stop), total)
    start = int(start)
    if not 0 <= start <= stop:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"shard bounds {start}:{stop} are not a sub-range of 0:{total}",
            declaration="start",
        )

    out = np.zeros(stop - start, dtype=np.float64)
    off = (pad_px - patch_px) // 2
    padded = np.zeros((pad_px, pad_px), dtype=np.complex128)
    for flat in range(start, stop):
        row = rows[flat // cols.size]
        col = cols[flat % cols.size]
        window = extract_patch(
            array,
            center_xy_m=(float(col), float(row)),
            patch_px=patch_px,
            sample_pitch_m=(1.0, 1.0),
        )
        if not window.any():
            continue
        padded[...] = 0.0
        padded[off : off + patch_px, off : off + patch_px] = window
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) / (
            pad_px * pad_px
        )
        out[flat - start] = float(np.abs(spectrum).sum())
    return out


def predicted_variance_ratio(
    density: np.ndarray[Any, Any], weight_model: np.ndarray[Any, Any]
) -> float:
    """``D sum f^2 / [(sum m)(sum f^2 / m)]`` -- uniform's variance over this one.

    Exactly 1 for a uniform ``density`` and maximal at ``density = weight_model``,
    where it becomes ``D sum f^2 / (sum f)^2``. Positions with zero density are
    dropped from the sum, which is only legitimate where ``f`` is also zero --
    checked by the caller through ``support_power_fraction`` rather than here,
    because here the remedy would be to silently return a finite number for an
    inconsistent estimator.
    """
    m = np.asarray(density, dtype=np.float64).ravel()
    f2 = np.asarray(weight_model, dtype=np.float64).ravel() ** 2
    support = m > 0.0
    if not support.any():
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "the position density is zero everywhere",
            declaration="density",
        )
    numerator = float(m.size) * float(f2.sum())
    denominator = float(m.sum()) * float((f2[support] / m[support]).sum())
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _density_map(
    kind: PositionDensity,
    *,
    energy: np.ndarray[Any, Any],
    spectral_l1: np.ndarray[Any, Any] | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], str]:
    """``(density, weight_model, model_note)``, all flattened row-major."""
    if spectral_l1 is not None:
        model = np.asarray(spectral_l1, dtype=np.float64).ravel()
        if model.size != energy.size:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"spectral_l1 has {model.size} entries but the candidate grid has "
                f"{energy.size}",
                declaration="spectral_l1",
            )
        note = "f = ||U~_c||_1, measured exactly by spectral_l1_map"
    else:
        model = np.sqrt(energy)
        note = (
            "f = sqrt(window energy), the quasi-random-phase approximation to "
            "||U~_c||_1. Exact f needs spectral_l1_map."
        )

    if kind is PositionDensity.UNIFORM:
        density = np.ones_like(energy)
    elif kind is PositionDensity.SQRT_WINDOW_ENERGY:
        density = np.sqrt(energy)
    elif kind is PositionDensity.WINDOW_ENERGY:
        density = energy.copy()
    else:
        if spectral_l1 is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "SPECTRAL_L1 needs a precomputed map from spectral_l1_map; there "
                "is no cheap route to ||U~_c||_1",
                declaration="spectral_l1",
                remedy="Pass spectral_l1=, or use SQRT_WINDOW_ENERGY.",
            )
        density = model.copy()
    return density, model, note


def plan_positions(
    *,
    doe_field: np.ndarray[Any, Any],
    patch_px: int,
    sample_pitch_m: tuple[float, float],
    count: int,
    density: PositionDensity = PositionDensity.UNIFORM,
    draw: PositionDraw = PositionDraw.IID,
    rng: np.random.Generator | None = None,
    spectral_l1: np.ndarray[Any, Any] | None = None,
) -> PositionPlan:
    """``count`` primary positions with the weights that keep them unbiased.

    The uniform i.i.d. case is the estimator that shipped: positions drawn with
    ``rng.integers`` over the same inclusive range, weights exactly 1. It is
    routed through the same code as the others so the comparison is between
    densities and not between code paths -- ``tests/test_patch_positions.py``
    pins it against ``plan_patches``' own draw.
    """
    if rng is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "drawing primary positions needs an explicit seeded generator",
            declaration="rng",
        )
    if count <= 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"primary position count must be positive, got {count}",
            declaration="count",
        )
    array = np.asarray(doe_field)
    rows, cols = candidate_index_grid(grid_shape=array.shape, patch_px=patch_px)
    n_rows, n_cols = rows.size, cols.size
    total = n_rows * n_cols

    energy = window_energy_map(array, patch_px=patch_px).ravel()
    density_map, weight_model, model_note = _density_map(
        density, energy=energy, spectral_l1=spectral_l1
    )
    support = density_map > 0.0
    energy_total = float(energy.sum())
    support_power_fraction = (
        1.0 if energy_total <= 0.0 else float(energy[support].sum()) / energy_total
    )
    if support_power_fraction < 1.0:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            (
                "the position density is zero on a candidate whose window holds "
                f"aperture power ({1.0 - support_power_fraction:.3e} of the total); "
                "the estimator would be inconsistent, not merely noisy"
            ),
            declaration="density",
        )

    uniform = density is PositionDensity.UNIFORM
    if uniform and draw is PositionDraw.IID:
        # The shipped draw, bit for bit. `plan_patches` calls `rng.integers`
        # separately on rows then columns; a `choice` over the flattened grid
        # would consume a different stream and the baseline arm of every
        # comparison here would stop reproducing the committed records.
        row_pick = rng.integers(rows[0], rows[-1] + 1, size=count)
        col_pick = rng.integers(cols[0], cols[-1] + 1, size=count)
        flat = (row_pick - rows[0]) * n_cols + (col_pick - cols[0])
        expected_draws = np.full(count, float(count) / float(total))
    elif draw is PositionDraw.IID:
        flat, expected_draws = _draw_iid(density_map, count=count, rng=rng)
    elif draw is PositionDraw.STRATIFIED_CDF:
        flat, expected_draws = _draw_stratified_cdf(
            density_map, count=count, rng=rng
        )
    else:
        flat, expected_draws = _draw_jittered_grid(
            density_map, count=count, shape=(n_rows, n_cols), rng=rng
        )

    # lambda_c = P / (D pi_c), with P the count actually drawn: a stratified
    # scheme may skip a stratum that holds no density, and the `one_over_n`
    # normalization downstream divides by the rays that were really emitted.
    realized = int(flat.size)
    weights = float(realized) / (float(total) * expected_draws)
    centers = np.column_stack(
        [
            cols[flat % n_cols] * float(sample_pitch_m[1]),
            rows[flat // n_cols] * float(sample_pitch_m[0]),
        ]
    ).astype(np.float64)

    return PositionPlan(
        centers_xy_m=centers,
        center_weights=weights.astype(np.float64),
        density_kind=str(density),
        draw_kind=str(draw),
        draw_positions=total,
        support_positions=int(support.sum()),
        support_power_fraction=support_power_fraction,
        empty_positions=int((energy <= 0.0).sum()),
        predicted_variance_ratio=predicted_variance_ratio(density_map, weight_model),
        variance_model=model_note,
        mean_center_weight=float(weights.mean()),
    )


def _draw_iid(
    density_map: np.ndarray[Any, Any], *, count: int, rng: np.random.Generator
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Independent draws, and ``pi_c = P q_c`` for each one."""
    q = density_map / density_map.sum()
    flat = rng.choice(density_map.size, size=count, p=q)
    return flat, float(count) * q[flat]


def _draw_stratified_cdf(
    density_map: np.ndarray[Any, Any], *, count: int, rng: np.random.Generator
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """One draw per equal-probability interval of the CDF.

    ``pi_c = P q_c`` **exactly**, the same as the i.i.d. draw: stratum ``k``
    picks ``c`` with probability ``P`` times the overlap of ``[F(c-1), F(c)]``
    with ``[k/P, (k+1)/P]``, and summing the overlaps over ``k`` recovers
    ``q_c``. So the weights are unchanged and only the clumping is removed --
    which is what makes this composable with an importance density where the
    equal-area tiling is not.

    A position heavier than ``1/P`` is drawn by more than one stratum, which is
    correct rather than a defect: its expected draw count is still ``P q_c``.
    """
    q = density_map / density_map.sum()
    cumulative = np.cumsum(q)
    cumulative[-1] = 1.0
    strata = (np.arange(count) + rng.random(count)) / count
    flat = np.clip(
        np.searchsorted(cumulative, strata, side="left"), 0, q.size - 1
    )
    # `rng.random()` can return exactly 0.0, and stratum 0 then lands on index 0
    # whatever its density -- which under an importance density is a corner
    # position with q = 0, so pi = 0 and the weight is infinite. Probability
    # 2^-53, and a silently infinite weight is not a failure mode worth leaving
    # in a sampler. Snapped onto the support, which is where side="left" puts
    # every other stratum anyway.
    support = np.flatnonzero(q > 0.0)
    off_support = q[flat] <= 0.0
    if off_support.any():
        flat[off_support] = support[
            np.clip(
                np.searchsorted(support, flat[off_support]), 0, support.size - 1
            )
        ]
    return flat, float(count) * q[flat]


def _draw_jittered_grid(
    density_map: np.ndarray[Any, Any],
    *,
    count: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """One draw per cell of a near-square tiling, sampled within the cell.

    ``pi_c`` is the cell's own draw count times its within-cell conditional, so
    a cell holding little of the density still contributes one draw and the
    weight puts the mass back.

    Cells are as square as ``count`` allows, and ``np.linspace`` edges make the
    remainder cells one row or column wider rather than bunching the remainder
    at the far edge -- an equal-count tiling of a 301 x 301 grid into 1300 cells
    does not exist, and the unequal cell counts are carried in the weight rather
    than approximated away.
    """
    n_rows, n_cols = shape
    tiles_y = max(1, int(round(math.sqrt(count * n_rows / n_cols))))
    tiles_x = max(1, count // tiles_y)
    row_edges = np.linspace(0, n_rows, tiles_y + 1).astype(np.int64)
    col_edges = np.linspace(0, n_cols, tiles_x + 1).astype(np.int64)

    grid = density_map.reshape(n_rows, n_cols)
    flat_out: list[int] = []
    marginal_out: list[float] = []
    cells = [
        (row_edges[i], row_edges[i + 1], col_edges[j], col_edges[j + 1])
        for i in range(tiles_y)
        for j in range(tiles_x)
    ]
    # `count` need not equal the cell count. Extra draws go round again over the
    # cells in order, which keeps every cell's draw count within one of every
    # other's; short counts simply stop early. Both stay unbiased because the
    # weight is built from the realized per-slot marginal.
    per_cell = np.zeros(len(cells), dtype=np.int64)
    for slot in range(count):
        per_cell[slot % len(cells)] += 1

    for (y0, y1, x0, x1), draws in zip(cells, per_cell, strict=True):
        if draws == 0:
            continue
        block = grid[y0:y1, x0:x1]
        mass = float(block.sum())
        if mass <= 0.0:
            # A cell with no density contributes nothing to the integral, so it
            # gets no draw and no weight. Spending a draw there would be exact
            # too -- the patch is zero -- but it would be a wasted patch, which
            # is the defect this whole module is about.
            continue
        conditional = (block / mass).ravel()
        picks = rng.choice(conditional.size, size=int(draws), p=conditional)
        rr, cc = np.divmod(picks, x1 - x0)
        flat_out.extend(((y0 + rr) * n_cols + (x0 + cc)).tolist())
        # This cell's slots each draw from `conditional`, so the expected number
        # of times position c is drawn is `draws * conditional[c]` -- not the
        # global density, which is what makes this stratified rather than a
        # relabelled i.i.d. draw.
        marginal_out.extend((float(draws) * conditional[picks]).tolist())

    if not flat_out:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "every cell of the jittered grid was empty of density",
            declaration="draw",
        )
    return np.asarray(flat_out, dtype=np.int64), np.asarray(
        marginal_out, dtype=np.float64
    )
