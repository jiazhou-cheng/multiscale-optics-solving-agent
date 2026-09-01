"""R10.4: the reduced-order model, and the three margins that bound its domain.

CHE-196. No field is formed at all: each incident ray is redirected by a local
grating equation evaluated at its own transverse position and comes out still one
ray.

    k_t^out = n_i k0 d_t^in + m grad_t(phi)(x, y)
    k_n^out = sqrt( (n_t k0)^2 - |k_t^out|^2 )
    opl^out = opl^in + m phi(x, y) / k0

**One operation with three models does not mean three interchangeable models**,
which is this ticket's named risk. This one is a *reduction*: it has a validity
domain the other two do not, the three signed margins are the boundary of that
domain, and crossing any of them is a refusal rather than a result. If the unified
signature let the margins be optional to inspect, the reduction would be invisible
at the call site -- so they are not optional, they are computed on every call, they
are all in the record, and each of them refuses.

Each boundary is tested from **both sides**. A validity predicate exercised only
deep inside its domain is untested, and the interesting failure is always the case
just outside.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import WAVELENGTH_M, a_surface, collimated_bundle

from operators import DiffractiveSurface, diffractive_surface
from operators.diffractive_surface import (
    MODULUS_LOCALITY_TOLERANCE,
    local_gradient_smoothness_margin,
    propagating_order_margin,
    single_order_dominance,
)
from representations import ContractError, ReferenceSurface

SRC = Path(__file__).resolve().parents[2] / "src"
MODULE = SRC / "operators" / "diffractive_surface.py"

GRID = 65
PITCH_M = (0.25e-6, 0.25e-6)
DOE_SURFACE = a_surface("doe")
COLUMN = np.arange(GRID) - GRID // 2


def an_incident_bundle(*, direction=(0.0, 0.0, 1.0), pitch=PITCH_M, surface=DOE_SURFACE):
    rays, _, _ = collimated_bundle(
        shape=(GRID, GRID), sample_pitch_m=pitch, direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    return dataclasses.replace(rays, reference_surface=surface)


def a_ramp(*, period_px: float, sign: int = 1, pitch=PITCH_M, surface=DOE_SURFACE, **kwargs):
    """A blazed linear phase ramp: exactly one diffraction order, analytically."""
    phase = np.tile(sign * 2.0 * math.pi * COLUMN / period_px, (GRID, 1))
    return DiffractiveSurface.from_phase(
        phase, sample_pitch_m=pitch, reference_surface=surface, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. The grating equation, and the order factor CHE-148 settled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period_px", [8, 16, 32, 200])
def test_a_blazed_ramp_deflects_by_exactly_lambda_over_the_period(period_px: int) -> None:
    """Criterion 1, against the grating equation and nothing this repository made.

    A linear phase ramp of period `Lambda` has one order, at
    `d_u = m lambda / Lambda`. Measured to six decimals at four periods spanning a
    factor of 25 -- and exact rather than close, because the complex-difference
    gradient estimator `angle(t[+1] conj(t[-1]))` is exact to round-off for a
    genuine ramp at any pitch: `angle` of a unit-modulus product has no
    approximation in it.
    """
    surface = a_ramp(period_px=period_px)
    period_m = period_px * PITCH_M[1]
    outgoing, record = diffractive_surface(
        an_incident_bundle(), surface=surface, model="generalized_snell"
    )
    direction_u = float(np.mean(np.asarray(outgoing.directions)[:, 0]))
    assert direction_u == pytest.approx(WAVELENGTH_M / period_m, rel=1e-6)
    assert record["model"] == "generalized_snell"
    # No field is formed, so there is nothing for the coupler records to describe.
    assert record["reconstruction"] is None
    assert record["sampling"] is None
    assert record["interior_field_validity"] == []


def test_the_order_multiplies_the_optical_path_as_well_as_the_momentum() -> None:
    """CHE-148, reproduced as the two code-independent arguments that settled it.

    `opl_out = opl_in + m phi / k0`, **not** `+ phi`. Both come from the m-th
    order's local plane-wave factor `exp(i m phi)`: differentiate it and you get
    the `m grad(phi)` in the momentum equation; evaluate it and you get `m phi`.

    1. `exp(i(-1) phi)` and `exp(i(+1)(-phi))` are the *same complex factor*, so
       `(order=-1, t)` and `(order=+1, conj(t))` must return the same bundle. With
       `phi` alone they returned the same direction and **opposite** optical paths,
       which is a contradiction rather than a tolerance.
    2. `order=0` is the undiffracted transmission and picks up no ramp at all. With
       `phi` alone it was handed the whole ramp phase on an undeflected ray.

    Both are checked here, on a shallow ramp so all three orders clear the
    single-order-dominance gate -- see
    `test_a_blazed_ramp_has_no_opposite_order_and_the_predicate_says_so`.
    """
    rays = an_incident_bundle()
    minus_on_t, _ = diffractive_surface(
        rays, surface=a_ramp(period_px=200), model="generalized_snell", order=-1
    )
    plus_on_conjugate, _ = diffractive_surface(
        rays, surface=a_ramp(period_px=200, sign=-1), model="generalized_snell", order=1
    )
    assert np.allclose(
        np.asarray(minus_on_t.directions), np.asarray(plus_on_conjugate.directions)
    )
    assert np.allclose(
        np.asarray(minus_on_t.optical_path_m),
        np.asarray(plus_on_conjugate.optical_path_m),
    )

    undiffracted, _ = diffractive_surface(
        rays, surface=a_ramp(period_px=200), model="generalized_snell", order=0
    )
    assert np.allclose(np.asarray(undiffracted.directions), np.asarray(rays.directions))
    assert np.allclose(
        np.asarray(undiffracted.optical_path_m), np.asarray(rays.optical_path_m)
    )


def test_the_amplitude_carries_only_the_transmission_modulus() -> None:
    """`a_out = a_in |t(x, y)|`, and the phase goes into the optical path instead.

    Splitting them is the same separation `RayBundle` makes everywhere else: a
    modulus is a physical amplitude and a phase is a path, and folding the second
    into the first would make the outgoing bundle's phase unreadable.
    """
    modulus = 0.4
    phase = np.tile(2.0 * math.pi * COLUMN / 200.0, (GRID, 1))
    surface = DiffractiveSurface(
        transmission=(modulus * np.exp(1j * phase)).astype(complex),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    rays = an_incident_bundle()
    outgoing, _ = diffractive_surface(rays, surface=surface, model="generalized_snell")
    assert np.allclose(
        np.abs(np.asarray(outgoing.amplitude)),
        modulus * np.abs(np.asarray(rays.amplitude)),
        rtol=1e-12,
    )


def test_it_is_the_one_model_that_may_run_in_a_medium() -> None:
    """No field is formed, so R09's `n = 1` ramp refusal does not bind it.

    The couplers refuse `medium_index != 1` because their transverse ramp is the
    vacuum form. This model never reaches them -- it goes ray to ray -- and the
    indices enter its own tangential-momentum equation, which carries them
    explicitly. That is a real capability the other two models do not have, and it
    is the reason `transmitted_index` is a field on the surface at all.
    """
    incident_index, transmitted_index = 1.5, 1.0
    in_glass = ReferenceSurface(name="doe", z_m=0.0, medium_index=incident_index)
    surface = a_ramp(period_px=200, surface=in_glass, transmitted_index=transmitted_index)
    rays = an_incident_bundle(surface=in_glass)

    outgoing, record = diffractive_surface(
        rays, surface=surface, model="generalized_snell"
    )
    snell = record["generalized_snell"]
    assert snell["incident_index"] == incident_index
    assert snell["transmitted_index"] == transmitted_index

    # `n_t k0 d_u^out = n_i k0 d_u^in + m dphi/dx`, with the incident ray on axis.
    gradient = 2.0 * math.pi / (200.0 * PITCH_M[1])
    expected = gradient / (transmitted_index * rays.wavenumber)
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        expected, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 2. Predicate 1: the order has to exist
# ---------------------------------------------------------------------------


def test_the_propagating_order_margin_is_the_fractional_form_of_its_limit() -> None:
    """`(n_t k0)^2 - |k_t|^2` over the limit: `> 0` propagates, `< 0` evanescent."""
    wavenumber, index = 1.0e7, 1.0
    limit_sq = (index * wavenumber) ** 2
    margins = propagating_order_margin(
        np.array([0.0, 0.5 * limit_sq, limit_sq, 2.0 * limit_sq]),
        transmitted_index=index,
        wavenumber=wavenumber,
    )
    assert margins[0] == pytest.approx(1.0)
    assert margins[1] == pytest.approx(0.5)
    assert margins[2] == pytest.approx(0.0)
    assert margins[3] < 0.0


@pytest.mark.parametrize(
    ("period_um", "expected_direction", "admitted"),
    [
        (0.60, 0.9167, True),
        (0.56, 0.9821, True),
        (0.5501, 0.99982, True),
        (0.55, 1.0, False),
        (0.50, 1.1, False),
    ],
)
def test_an_evanescent_order_is_refused_on_both_sides_of_the_boundary(
    period_um: float, expected_direction: float, admitted: bool
) -> None:
    """Criterion 3 and 5 for predicate 1: just inside and just outside.

    The boundary is `d_u = 1`, i.e. `Lambda = lambda`. A 50 nm pitch keeps the
    per-sample phase step small enough that the *gradient* is trustworthy right up
    to it, so this isolates predicate 1 rather than tripping predicate 2 first --
    which is what a coarser grid would do, and would have made this test about a
    different predicate than its name.

    | period | expected `d_u` | order margin | |
    | -- | -- | -- | -- |
    | 0.60 um | 0.9167 | +0.160 | admitted |
    | 0.56 um | 0.9821 | **+0.035** | admitted |
    | 0.5501 um | 0.99982 | **+3.6e-4** | admitted |
    | 0.55 um | 1.0000 | **0** | **refused** |
    | 0.50 um | 1.1000 | negative | **refused** |

    The `Lambda = lambda` row is the grazing case and is exact rather than a
    round-off coin flip: `lambda` is 550 nm and the pitch is 50 nm, so the period is
    11 samples exactly and the margin is analytically 0. It refuses because a margin
    is a distance to a boundary and a caller sitting on one has no distance left --
    physically, a grazing order has `k_n = 0` and carries nothing along the axis.
    The row above it is the same configuration 0.1 nm away, and it is admitted.
    """
    pitch = (0.05e-6, 0.05e-6)
    period_px = period_um * 1e-6 / pitch[1]
    surface = a_ramp(period_px=period_px, pitch=pitch)
    rays = an_incident_bundle(pitch=pitch)

    if admitted:
        outgoing, record = diffractive_surface(
            rays, surface=surface, model="generalized_snell"
        )
        assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
            expected_direction, rel=1e-3
        )
        assert record["generalized_snell"]["propagating_order_margin"] > 0.0
    else:
        with pytest.raises(ContractError) as raised:
            diffractive_surface(rays, surface=surface, model="generalized_snell")
        assert raised.value.code == "MISSING_DECLARATION"
        assert raised.value.declaration == "order"
        assert "PROPAGATING_ORDER_EXISTS" in str(raised.value)
        assert "signed margin" in str(raised.value)


# ---------------------------------------------------------------------------
# 3. Predicate 2: the gradient has to be trustworthy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period_px", "expected_margin", "admitted"),
    [(6.0, 0.3333, True), (4.4, 0.0909, True), (4.05, 0.0123, True), (3.95, None, False)],
)
def test_the_smoothness_boundary_is_the_estimators_own_nyquist_limit(
    period_px: float, expected_margin: float | None, admitted: bool
) -> None:
    """Criterion 5 for predicate 2, and the hole this ticket found in it.

    The gradient is a centred difference over **two** samples, so it aliases once
    the per-sample step exceeds `pi/2` -- a period of 4 samples. Measured, at
    250 nm pitch:

    | period | per-sample step | margin | |
    | -- | -- | -- | -- |
    | 6.00 px | 1.047 rad | +0.333 | admitted |
    | 4.40 px | 1.428 rad | +0.091 | admitted |
    | 4.05 px | 1.551 rad | **+0.012** | admitted |
    | 3.95 px | 1.591 rad | negative | **refused** |

    **The reference implementation's version of this predicate had a hole exactly
    where it mattered**, and it is the finding of this ticket. Its raw-step
    sub-check read the *two-sample* step the estimator returns, which is
    `wrap(2s)` -- and that tends to **zero** as the per-sample step `s` approaches
    `pi`. Measured on the old form: margins of +0.54, +0.74, +0.90 and **+0.98** at
    per-sample steps of 2.42, 2.73, 2.99 and 3.11 rad, rising toward maximum
    confidence precisely as the recovered direction cosine collapsed from its true
    `+0.275` to `-0.011`. The predicate was most certain where the answer was most
    wrong. It now reads `2 x` the adjacent-sample step, which is the unwrapped
    two-sample difference, and every case above is refused.
    """
    surface = a_ramp(period_px=period_px)
    rays = an_incident_bundle()
    if admitted:
        _, record = diffractive_surface(rays, surface=surface, model="generalized_snell")
        assert record["generalized_snell"][
            "local_gradient_smoothness_margin"
        ] == pytest.approx(expected_margin, abs=2e-3)
    else:
        with pytest.raises(ContractError) as raised:
            diffractive_surface(rays, surface=surface, model="generalized_snell")
        assert raised.value.declaration == "patch_px"
        assert "LOCAL_GRADIENT_SMOOTHNESS" in str(raised.value)


@pytest.mark.parametrize("period_px", [2.6, 2.3, 2.1, 2.02])
def test_a_uniformly_aliased_grating_is_refused_rather_than_looking_smooth(
    period_px: float,
) -> None:
    """The failure mode the reference's predicate reported as maximum confidence.

    A *uniformly* undersampled grating aliases every stencil tap by the same wrong
    amount, so the curvature sub-check reads it as perfectly smooth -- zero
    curvature, nonsense gradient. Only a check on the estimator's own span catches
    it, and only if that check measures the **unwrapped** span.
    """
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(), surface=a_ramp(period_px=period_px),
            model="generalized_snell",
        )
    assert "LOCAL_GRADIENT_SMOOTHNESS" in str(raised.value)


def test_the_smoothness_margin_is_the_worse_of_its_two_sub_checks() -> None:
    """Either alone misses a real failure, so the predicate is their minimum."""
    scale = 1.0e-6
    # Curvature dominates: a large second derivative with a tiny raw step.
    assert local_gradient_smoothness_margin(
        np.array([math.pi / scale**2]), np.array([0.0]), transverse_scale_m=scale
    )[0] == pytest.approx(0.0, abs=1e-12)
    # The raw step dominates: no curvature at all, but the span is at the limit.
    assert local_gradient_smoothness_margin(
        np.array([0.0]), np.array([math.pi]), transverse_scale_m=scale
    )[0] == pytest.approx(0.0, abs=1e-12)
    # And a smooth, well-sampled surface clears both.
    assert local_gradient_smoothness_margin(
        np.array([0.0]), np.array([0.0]), transverse_scale_m=scale
    )[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Predicate 3: the requested order has to carry the power
# ---------------------------------------------------------------------------


def test_a_single_ramp_concentrates_its_power_in_one_order() -> None:
    """The admitted side of predicate 3: `dominance = 0.83`, margin `+0.66`.

    Not 1.0, and that is the window rather than the surface: a rectangular window
    of `patch_px` samples has a mainlobe holding about 85 % of its power, and the
    dominance disk is one mainlobe wide by construction.
    """
    ramp = np.tile(np.exp(1j * 2.0 * math.pi * COLUMN / 8.0), (GRID, 1))
    dominance, margin = single_order_dominance(
        ramp,
        sample_pitch_m=PITCH_M,
        centre_xy_m=(0.0, 0.0),
        patch_px=9,
        wavelength_m=WAVELENGTH_M,
        target_direction_xy=(WAVELENGTH_M / (8 * PITCH_M[1]), 0.0),
    )
    # The physics is `> 0.5`, from a surface with analytically one order. The 0.832
    # is a recorded characterization of *this* window, not an oracle -- the analytic
    # anchors live in `test_a_binary_grating_cannot_satisfy_dominance_at_any_window`.
    assert dominance > 0.5
    assert dominance == pytest.approx(0.832, abs=0.02)
    assert margin == pytest.approx(2.0 * dominance - 1.0)


@pytest.mark.parametrize(
    ("patch_px", "expected_dominance"), [(5, 0.659), (9, 0.395), (17, 0.051)]
)
def test_the_smooth_to_diffractive_transition_is_where_dominance_crosses_a_half(
    patch_px: int, expected_dominance: float
) -> None:
    """Criterion 4 and 5 for predicate 3: **where this model stops being valid.**

    CHE-146 established that the boundary is the smooth-to-diffractive transition.
    The construction here isolates it: two linear ramps of opposite slope meeting
    at the origin. Each half is *perfectly* smooth -- constant gradient, zero
    curvature -- so predicates 1 and 2 both pass and cannot be what fails. What
    changes is whether the **window** resolves more than one order:

    | patch_px | dominance | margin | |
    | -- | -- | -- | -- |
    | 5 | 0.659 | +0.317 | one order, just |
    | 9 | 0.395 | **-0.209** | two orders |
    | 17 | 0.051 | -0.898 | fully diffractive |

    The margin crosses zero between 5 and 9 samples, which is the transition, and
    the knob is the window rather than the surface -- exactly the right one, since
    "how many orders are there" is only answerable relative to a window.

    Note what this says about the *other* two predicates: for a curved profile
    (a sinusoid, say) the smoothness check fires long before dominance can report,
    because anything curved enough to emit several orders is curved enough to fail
    the local plane-wave picture at any window wide enough to resolve them. So
    dominance's real job is this residual case -- locally smooth, globally
    multi-order -- and a test on a sinusoid would have measured predicate 2 while
    claiming to measure predicate 3.
    """
    left = np.exp(1j * 2.0 * math.pi * COLUMN / 8.0)
    right = np.exp(-1j * 2.0 * math.pi * COLUMN / 8.0)
    bi_ramp = np.tile(np.where(COLUMN < 0, left, right), (GRID, 1))

    dominance, margin = single_order_dominance(
        bi_ramp,
        sample_pitch_m=PITCH_M,
        centre_xy_m=(0.0, 0.0),
        patch_px=patch_px,
        wavelength_m=WAVELENGTH_M,
        target_direction_xy=(0.0, 0.0),
    )
    assert dominance == pytest.approx(expected_dominance, abs=0.02)
    assert margin == pytest.approx(2.0 * dominance - 1.0)
    assert (margin > 0.0) == (patch_px == 5)


def test_a_blazed_ramp_has_no_opposite_order_and_the_predicate_says_so() -> None:
    """Criterion 3 for predicate 3: crossing it refuses, with the margin.

    A blazed ramp puts *all* its power in one order, so the opposite order does not
    exist -- and asking this model for it would return one ray pointing where there
    is nothing. Refused, because a reduced-order model applied where the requested
    order carries no power is returning a direction for a response that is not
    there.

    At period 8 and the default window the two `+-1` targets are 0.55 apart against
    a disk radius of 0.44, so the disk *does* exclude the real peak even though
    `orders_resolved` is `False` -- that flag is a sufficient condition, not a
    necessary one, and this is the margin between the two. At period 16 the same
    request is admitted, which is
    `test_the_default_window_cannot_separate_orders_and_says_so`.
    """
    rays = an_incident_bundle()
    surface = a_ramp(period_px=8)
    diffractive_surface(rays, surface=surface, model="generalized_snell", order=1)

    with pytest.raises(ContractError) as raised:
        diffractive_surface(rays, surface=surface, model="generalized_snell", order=-1)
    assert raised.value.code == "MISSING_DECLARATION"
    assert "SINGLE_ORDER_DOMINANCE" in str(raised.value)
    assert "signed margin" in str(raised.value)
    assert "full_field" in (raised.value.remedy or "")


def test_a_binary_grating_cannot_satisfy_dominance_at_any_window() -> None:
    """The analytic bound, and the one predicate that actually fires on it.

    A binary `pi` grating has Fourier coefficients `2/(m pi)` on odd `m`, so its
    strongest order carries `4/pi^2 = 0.405` of the total -- **below one half by
    arithmetic**, whatever window measures it. So dominance must refuse it, and the
    measured 0.364 at a 9-sample window is on the correct side of a boundary that
    was fixed before this code existed. The `+-1` symmetry is exact for the same
    reason: `t` is real, so its spectrum is Hermitian and the two orders carry
    identical power. Neither statement comes from this repository.

    Reaching that predicate needs the phase read directly, because a `+-1` surface
    has adjacent phase steps of `pi` and the *operator* refuses it one predicate
    earlier -- on smoothness, which is also correct and is asserted below rather
    than left to an `or`.
    """
    column = np.arange(GRID)
    binary = np.tile(np.where((column // 4) % 2 == 0, 1.0, -1.0), (GRID, 1)).astype(complex)
    first_order = WAVELENGTH_M / (8 * PITCH_M[1])
    dominance = {}
    for order in (1, -1):
        dominance[order], margin = single_order_dominance(
            binary,
            sample_pitch_m=PITCH_M,
            centre_xy_m=(0.0, 0.0),
            patch_px=9,
            wavelength_m=WAVELENGTH_M,
            target_direction_xy=(order * first_order, 0.0),
        )
        assert dominance[order] < 0.5, order
        assert margin < 0.0, order
    assert dominance[1] == pytest.approx(dominance[-1], abs=1e-9)
    assert dominance[1] < 4.0 / math.pi**2

    # ...and a blazed ramp, whose power is all in one order, is the other side of
    # the same analytic statement: strongly asymmetric, and above one half.
    blazed = np.tile(np.exp(1j * 2.0 * math.pi * COLUMN / 8.0), (GRID, 1))
    forward, _ = single_order_dominance(
        blazed, sample_pitch_m=PITCH_M, centre_xy_m=(0.0, 0.0), patch_px=9,
        wavelength_m=WAVELENGTH_M, target_direction_xy=(first_order, 0.0),
    )
    backward, _ = single_order_dominance(
        blazed, sample_pitch_m=PITCH_M, centre_xy_m=(0.0, 0.0), patch_px=9,
        wavelength_m=WAVELENGTH_M, target_direction_xy=(-first_order, 0.0),
    )
    assert forward > 0.5 > backward
    assert forward > 20.0 * backward

    surface = DiffractiveSurface(
        transmission=binary, sample_pitch_m=PITCH_M, reference_surface=DOE_SURFACE
    )
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(), surface=surface, model="generalized_snell", patch_px=9
        )
    assert "LOCAL_GRADIENT_SMOOTHNESS" in str(raised.value)


# ---------------------------------------------------------------------------
# 5. All three margins travel, always
# ---------------------------------------------------------------------------


def test_all_three_margins_are_signed_and_always_in_the_record() -> None:
    """Criterion 2, and the answer to this ticket's named risk.

    The whole value of a reduced-order model is that its domain is bounded and the
    bounds are **visible**. All three are computed on every call, all three are
    signed so a caller sees how close it is rather than only whether it crossed,
    and all three are in the record whether or not they were near a boundary.
    """
    _, record = diffractive_surface(
        an_incident_bundle(), surface=a_ramp(period_px=16), model="generalized_snell"
    )
    snell = record["generalized_snell"]
    for name in (
        "propagating_order_margin",
        "local_gradient_smoothness_margin",
        "single_order_dominance_margin",
    ):
        assert isinstance(snell[name], float), name
        assert snell[name] > 0.0, name
    assert snell["order"] == 1
    assert snell["single_order_dominance"] == pytest.approx(
        (snell["single_order_dominance_margin"] + 1.0) / 2.0
    )
    assert "CHE-148" in snell["opl_convention"]

    # And the emitted bundle records the surface it passed through.
    outgoing, _ = diffractive_surface(
        an_incident_bundle(), surface=a_ramp(period_px=16), model="generalized_snell"
    )
    assert "generalized-Snell" in (outgoing.optical_path_reference or "")
    assert "order m=1" in (outgoing.optical_path_reference or "")


def test_a_curved_substrate_is_refused_because_the_tangent_frame_is_position_dependent() -> None:
    """Planar only. The tangential-momentum equation is evaluated in the surface's
    own local frame, and on a curved substrate that frame is position-dependent --
    this model has no way to accept one declared per ray."""
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(),
            surface=a_ramp(period_px=16, radius_m=1.0e-2),
            model="generalized_snell",
        )
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "radius_m"
    assert "position-dependent" in str(raised.value)


def test_an_order_supplied_to_a_model_that_ignores_it_is_refused() -> None:
    """A parameter the named model ignores is a caller believing something the run
    did not do. The pairing is not inferred in either direction."""
    for model in ("full_field", "local_patch"):
        with pytest.raises(ContractError) as raised:
            diffractive_surface(
                an_incident_bundle(),
                surface=a_ramp(period_px=16),
                model=model,  # type: ignore[arg-type]
                order=2,
            )
        assert raised.value.declaration == "order"


@pytest.mark.parametrize("patch_px", [4, 0])
def test_an_even_patch_has_no_transverse_scale_to_declare(patch_px: int) -> None:
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(),
            surface=a_ramp(period_px=16),
            model="generalized_snell",
            patch_px=patch_px,
        )
    assert raised.value.code == "SHAPE_MISMATCH"
    assert raised.value.declaration == "patch_px"


def test_the_avoided_generalized_snell_classes_did_not_land() -> None:
    """Criterion: 0 production classes. `GeneralizedSnellDiagnostics` folds into the
    operator's record, and there is no subclass."""
    defined = {
        node.name
        for module in sorted((SRC / "operators").rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == {"DiffractiveSurface"}
    for avoided in (
        "GeneralizedSnellDiagnostics",
        "GeneralizedSnellParameters",
        "GeneralizedSnellSubclass",
    ):
        assert avoided not in defined, avoided


# ---------------------------------------------------------------------------
# 6. What the review of this ticket found: four ways the branch was wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("incident_u", [0.0, 0.1, 0.3, 0.5])
def test_a_tilted_illumination_does_not_slide_the_dominance_disk_off_the_order(
    incident_u: float,
) -> None:
    """The dominance disk is centred on the order's **kick**, not on `d_out`.

    The spectrum predicate 3 searches is the spectrum of `exp(i phi)`, so an order
    sits in it at its momentum kick `m grad(phi) / k0`. The outgoing direction
    cosine is `(n_i d_in + m grad(phi)/k0) / n_t`, which coincides with the kick
    only at normal incidence in vacuum -- the one configuration the first draft of
    this file tested. Centred on `d_out`, a 30-degree plane wave on an essentially
    single-order grating was refused with the message that the surface emitted
    several orders, which is false: the illumination tilt does not change the order
    content, it only slides the disk off the peak.

    Measured on a 200-px ramp: `d_out` tracks the tilt exactly (`0.011 + d_in`)
    while dominance stays flat at 0.866 across the whole range.
    """
    direction = (incident_u, 0.0, math.sqrt(1.0 - incident_u**2))
    outgoing, record = diffractive_surface(
        an_incident_bundle(direction=direction),
        surface=a_ramp(period_px=200),
        model="generalized_snell",
    )
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        incident_u + WAVELENGTH_M / (200 * PITCH_M[1]), rel=1e-6
    )
    assert record["generalized_snell"]["single_order_dominance"] == pytest.approx(
        0.866, abs=0.01
    )


def test_the_disk_stays_on_the_order_when_the_transmitted_index_is_not_one() -> None:
    """The same error, its other half: `d_out` is divided by `n_t` and the kick is not.

    A narrow window is the case that exposes it -- at the default `patch_px=5` the
    disk is wide enough to cover both the peak and the wrong centre.
    """
    surface = a_ramp(period_px=8, transmitted_index=1.5)
    outgoing, record = diffractive_surface(
        an_incident_bundle(), surface=surface, model="generalized_snell", patch_px=17
    )
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        WAVELENGTH_M / (1.5 * 8 * PITCH_M[1]), rel=1e-6
    )
    assert record["generalized_snell"]["single_order_dominance"] > 0.5


def test_an_amplitude_grating_is_refused_because_this_model_cannot_see_it() -> None:
    """The invalid case the first draft admitted with every margin green.

    A Ronchi grating `t in {0, 1}` diffracts strongly into `+-1`, and `arg(t)` is
    identically zero. This model reads only `arg(t)`, so it returned every requested
    order **undeflected** -- while predicate 3, which transformed `exp(i arg t)` and
    therefore saw a uniform window, reported 0.83 of the power in the requested
    order. Maximum confidence on a result with no physics in it.

    The gate is on the modulus rather than on the spectrum, because a spectral test
    is not robust here: at a small window the extra orders are inside the disk and
    at a large one the split lands within round-off of the 0.5 boundary. "This
    model is blind to amplitude structure" is the honest statement, and it is a
    property of `|t|` that can be read directly.
    """
    column = np.arange(GRID)
    ronchi = np.tile(np.where((column // 4) % 2 == 0, 1.0, 0.0), (GRID, 1)).astype(complex)
    surface = DiffractiveSurface(
        transmission=ronchi, sample_pitch_m=PITCH_M, reference_surface=DOE_SURFACE
    )
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(), surface=surface, model="generalized_snell"
        )
    assert raised.value.declaration == "transmission"
    assert "blind to" in str(raised.value)
    assert "full_field" in (raised.value.remedy or "")


def test_a_gentle_apodization_is_not_amplitude_structure_and_is_admitted() -> None:
    """The other side: the gate refuses amplitude *structure*, not an envelope.

    A 10 % quadratic taper across the whole aperture varies by 2.5 % over the five
    samples the gradient estimator reads -- under the 5 % tolerance -- and the ray
    is deflected by the phase ramp underneath it exactly as if the taper were not
    there. Refusing this would make the model useless on any real apodized DOE.
    """
    taper = 1.0 - 0.1 * np.tile((COLUMN / (GRID // 2)) ** 2, (GRID, 1))
    surface = DiffractiveSurface(
        transmission=taper * np.exp(1j * np.tile(2.0 * math.pi * COLUMN / 200.0, (GRID, 1))),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    outgoing, record = diffractive_surface(
        an_incident_bundle(), surface=surface, model="generalized_snell"
    )
    variation = record["generalized_snell"]["worst_modulus_variation"]
    assert 0.0 < variation < MODULUS_LOCALITY_TOLERANCE
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        WAVELENGTH_M / (200 * PITCH_M[1]), rel=1e-6
    )


def test_a_ray_off_the_surfaces_sampled_extent_is_refused_not_clamped() -> None:
    """Clamping silently extended the DOE to infinity.

    A bundle wider than the transmission grid had every off-aperture ray indexed to
    the edge sample, so it received the edge gradient and the edge modulus and came
    out deflected by structure that is not there -- no refusal, no diagnostic, and a
    record full of positive margins. Zeroing them instead would have been the other
    guess: a DOE patch in an open window transmits outside its own extent. The
    surface declares neither, so this refuses.
    """
    wide = an_incident_bundle(pitch=(4.0 * PITCH_M[0], 4.0 * PITCH_M[1]))
    with pytest.raises(ContractError) as raised:
        diffractive_surface(wide, surface=a_ramp(period_px=200), model="generalized_snell")
    assert raised.value.code == "SHAPE_MISMATCH"
    assert raised.value.declaration == "positions_m"
    assert "sampled extent" in str(raised.value)


def test_a_bundle_declared_somewhere_else_is_refused() -> None:
    """The expectation check the other two models get from the coupler for free.

    `full_field` and `local_patch` pass `surface=` to `couplers.ray_to_scalar`,
    which refuses a bundle declared elsewhere. This branch forms no field and so
    never reaches that check -- and it reads the incident index off the *surface*
    and overwrites each ray's `z` with the surface's, so without its own check a
    bundle declared in air against a surface in glass was given 1.5x the tangential
    momentum and then silently relocated.
    """
    elsewhere = ReferenceSurface(name="elsewhere", z_m=1.0e-3, medium_index=1.0)
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(surface=elsewhere),
            surface=a_ramp(period_px=200),
            model="generalized_snell",
        )
    assert raised.value.code == "FRAME_MISMATCH"
    assert raised.value.declaration == "reference_surface"
    assert "propagate_rays" in (raised.value.remedy or "")


@pytest.mark.parametrize(
    ("period_px", "recovered_u", "true_u"), [(1.2, -0.3667, 1.8333), (1.1, -0.2000, 2.0)]
)
def test_below_two_samples_per_period_the_predicate_is_fooled_and_that_is_the_arrays_limit(
    period_px: float, recovered_u: float, true_u: float
) -> None:
    """**The residue of predicate 2, pinned rather than claimed closed.**

    Every quantity the smoothness margin is built from comes out of `angle(...)`,
    which is wrapped. Correcting `worst_raw_step` to the unwrapped two-sample step
    moved the failure one octave down; it did not remove it. Below two samples per
    period an adjacent step is itself aliased, so it reads small again:

    | period | recovered `d_u` | true `d_u` | smoothness | order | dominance |
    | -- | -- | -- | -- | -- | -- |
    | 1.2 px | -0.367 | **1.833** (evanescent) | +0.333 | +0.866 | 0.871 |
    | 1.1 px | -0.200 | **2.000** (evanescent) | +0.636 | +0.960 | 0.858 |

    All three margins are comfortably positive and the answer is wrong, including
    its sign. Predicate 1 cannot help: it is evaluated on the already-aliased `k_t`.

    This is **irreducible from the array alone**, which is why it is documented
    instead of fixed. A transmission sampled at 1.2 samples per period does not
    contain the surface it came from -- the sampling threw it away before this
    function saw it -- and every estimator reading only that array is in the same
    position. The guarantee in `_local_phase_gradient` is worded to stop at two
    samples per period for that reason, and this test is what makes the wording
    checkable. A caller in this region has to fix its sampling; nothing downstream
    can.
    """
    outgoing, record = diffractive_surface(
        an_incident_bundle(), surface=a_ramp(period_px=period_px), model="generalized_snell"
    )
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        recovered_u, abs=1e-3
    )
    assert WAVELENGTH_M / (period_px * PITCH_M[1]) == pytest.approx(true_u, abs=1e-3)
    assert true_u > 1.0, "the true order is evanescent, so no ray should come out at all"
    snell = record["generalized_snell"]
    assert snell["local_gradient_smoothness_margin"] > 0.0
    assert snell["propagating_order_margin"] > 0.0
    assert snell["single_order_dominance_margin"] > 0.0

    # Two samples per period, the stated edge of the guarantee, is refused.
    with pytest.raises(ContractError):
        diffractive_surface(
            an_incident_bundle(), surface=a_ramp(period_px=2.0), model="generalized_snell"
        )


def test_a_sparse_bundle_cannot_hide_an_amplitude_grating_from_the_modulus_gate() -> None:
    """The second review round's finding: the gate's support was the ray stencil.

    The first version of the modulus gate read the nine taps around each ray. A
    bundle sampled at the grating's own period -- and a ray-side operator's bundle
    has no relationship to the DOE's sample grid, so this is ordinary rather than
    contrived -- put every ray at a bar centre. Every tap read `|t| = 1`, the
    variation came out exactly 0, and finding the gate was written for came
    straight back: an undeflected ray from a grating that diffracts about 40 % of
    its power into `+-1`.

    The gate is now a statement about the **surface**: the largest change in `|t|`
    between adjacent samples anywhere on it, as a fraction of the peak. A verdict
    that depends on where the rays happen to sit is not a property of the surface.
    """
    column = np.arange(GRID)
    bars = np.where(((column - GRID // 2 + 8) // 8) % 2 == 0, 1.0, 0.0)
    surface = DiffractiveSurface(
        transmission=np.tile(bars, (GRID, 1)).astype(complex),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    dense = an_incident_bundle()
    sparse, _, _ = collimated_bundle(
        shape=(5, 5),
        sample_pitch_m=(16 * PITCH_M[0], 16 * PITCH_M[1]),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    sparse = dataclasses.replace(sparse, reference_surface=DOE_SURFACE)
    for rays in (dense, sparse):
        with pytest.raises(ContractError) as raised:
            diffractive_surface(rays, surface=surface, model="generalized_snell")
        assert raised.value.declaration == "transmission"
        assert "1.000 of its peak" in str(raised.value)


@pytest.mark.parametrize("order", [2, -2, 3])
def test_an_order_beyond_the_first_is_outside_a_local_gradient_model(order: int) -> None:
    """A locally linear phase has one fundamental and no harmonics.

    The predicate searched the fundamental's spectrum at `m` times its kick, which
    finds whichever *real* order happens to land in the window -- so on one blazed
    ramp `m = +2` was admitted at dominance 0.837 while `m = -2` was refused, an
    asymmetry with no physics behind it. Harmonic content is exactly the structure
    the smoothness predicate refuses one step earlier, so there is no surface this
    model admits for which `|m| >= 2` means anything.
    """
    with pytest.raises(ContractError) as raised:
        diffractive_surface(
            an_incident_bundle(),
            surface=a_ramp(period_px=16),
            model="generalized_snell",
            order=order,
        )
    assert raised.value.declaration == "order"
    assert "locally linear" in str(raised.value)
    assert "full_field" in (raised.value.remedy or "")


@pytest.mark.parametrize("requested_order", [1, 0, -1])
def test_the_default_window_cannot_separate_orders_and_says_so(requested_order: int) -> None:
    """The disk is one window-resolution wide, and the caller sets the window.

    On a 16-sample ramp at the default `patch_px=5` the disk radius is 0.44 while
    the orders are 0.1375 apart, so **all three of these are admitted** at dominance
    0.71-0.86 -- and only `+1` exists. Predicate 3 is not wrong here so much as
    uninformative: it reports that power lies within one resolution element of the
    requested direction, which at this window is true of every order at once.

    That is not left to the reader. `orders_resolved` is the comparison itself, in
    the record beside the margin, and it is `False` for every one of these.
    `test_a_resolving_window_refuses_the_orders_that_do_not_exist` is the same
    surface with a window that can separate them.
    """
    outgoing, record = diffractive_surface(
        an_incident_bundle(),
        surface=a_ramp(period_px=16),
        model="generalized_snell",
        order=requested_order,
    )
    snell = record["generalized_snell"]
    assert snell["orders_resolved"] is False
    assert snell["dominance_disk_radius"] == pytest.approx(0.44, rel=1e-3)
    assert snell["order_spacing_direction"] == pytest.approx(0.1375, rel=1e-3)
    assert snell["single_order_dominance"] > 0.5
    assert float(np.mean(np.asarray(outgoing.directions)[:, 0])) == pytest.approx(
        requested_order * 0.1375, abs=1e-6
    )


@pytest.mark.parametrize(("patch_px", "surviving"), [(33, 1), (49, 1)])
def test_a_resolving_window_refuses_the_orders_that_do_not_exist(
    patch_px: int, surviving: int
) -> None:
    """The other side, and the remedy the refusal names.

    Same 16-sample ramp, a window wide enough that the disk (0.067 at 33 samples,
    0.045 at 49) fits inside the 0.1375 order spacing. `orders_resolved` becomes
    `True`, `m = +1` survives at dominance 0.81-0.82, and `m = 0` and `m = -1` --
    which a blazed ramp does not have -- are both refused on dominance. The
    smoothness margin is unaffected, because a ramp has no curvature, so widening
    the window costs nothing on this class of surface.
    """
    for order in (1, 0, -1):
        if order == surviving:
            _, record = diffractive_surface(
                an_incident_bundle(), surface=a_ramp(period_px=16),
                model="generalized_snell", order=order, patch_px=patch_px,
            )
            snell = record["generalized_snell"]
            assert snell["orders_resolved"] is True
            assert snell["dominance_disk_radius"] < snell["order_spacing_direction"]
            assert snell["single_order_dominance"] > 0.5
        else:
            with pytest.raises(ContractError) as raised:
                diffractive_surface(
                    an_incident_bundle(), surface=a_ramp(period_px=16),
                    model="generalized_snell", order=order, patch_px=patch_px,
                )
            assert "SINGLE_ORDER_DOMINANCE" in str(raised.value)
