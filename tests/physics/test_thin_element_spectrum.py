"""A known transmission gives a known spectrum: Bessel orders and an NA cutoff.

CHE-211 (R06.6) acceptance criteria 2 and 3. Three closed forms, none of them
another run of the code under test:

===============================  ==============================================  ==========
case                             oracle                                          measured
===============================  ==============================================  ==========
sinusoidal phase grating         ``A_n / A_0 = J_n(m) / J_0(m)``, signed          9.2e-9 abs
binary phase grating, phi = pi   even orders vanish; ``|A_n|/|A_1| =
                                 sin(pi/M) / sin(n pi/M)``                        4e-6 abs
finite aperture vs NA            passes iff ``|f_x| <= NA/lambda``, step at
                                 ``R = f NA / n``                                 exact step
===============================  ==============================================  ==========

The Bessel case is Jacobi--Anger, and it is the primary gate:

    exp(i m sin(2 pi x / Lambda)) = sum_n J_n(m) exp(i 2 pi n x / Lambda)

so a sinusoidal phase grating of depth `m` puts amplitude `J_n(m)` into order `n`,
at position `x_n = n lambda f / (n_med Lambda)`. The Bessel values come from
`scipy.special.jv` -- an independent implementation of a special function, not
repository numerical code, which is the distinction AGENTS.md draws about what may
gate. The same oracle family (`order_coefficients`) gated CHE-144.

**The signed ratio is asserted, not the magnitude.** `J_{-n} = (-1)^n J_n`, so the
odd negative orders come back with the opposite sign; a magnitude-only comparison
would pass a model that mirrored the spectrum. Measured: order -1 at -1.090087
against -1.090087.

Why the binary grating's oracle is the *sampled* closed form
------------------------------------------------------------
The continuum coefficients of a 50%-duty binary phase grating are `1/n` for odd
`n` (relative to the first order) and zero for even `n`. On a grid with `M`
samples per period the exact discrete coefficient is `1 / (M sin(n pi / M))`
instead, so the ratio is `sin(pi/M) / sin(n pi/M)` -- which is `1/n` only in the
limit. At `M = 32` the third order is 0.33766 rather than 0.33333, a 1.3%
difference and a hundred times the float32 floor. Gating on the continuum value
would therefore require a 5% tolerance covering a real, exactly-known
discretization effect; gating on the sampled form keeps the tolerance at the
arithmetic floor and reports the continuum number beside it as the limit it is.
The continuum value is checked too, at the tolerance the discretization actually
justifies.

Cost: seven optical Fourier transforms on a 64 x 256 complex64 grid, all on the
host, well under a second. Nothing here is marked `slow`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import jv

from operators import (
    circular_aperture_amplitude,
    complex_transmission,
    numerical_aperture_radius_m,
)
from representations import ReferenceSurface, ScalarField
from solvers.chromatix import focal_plane_transform
from sources import plane_wave

WAVELENGTH_M = 0.532e-6
FOCAL_LENGTH_M = 20e-3
MEDIUM_INDEX = 1.0

#: 64 x 256 at 0.5 um. Asymmetric in count so a transposed transform cannot hide,
#: and the grating runs along `x` only, so the whole spectrum must land on one row.
SHAPE = (64, 256)
PITCH_M = (0.5e-6, 0.5e-6)

#: Periods of the grating across the window. 8 periods of 256 samples is 32
#: samples per period, which is what makes every diffraction order land **exactly**
#: on an output sample: order `n` sits at `n * PERIODS` samples from the origin,
#: because `x_n / dx_out = n (N dx_in) / Lambda = n * PERIODS`. An off-grid period
#: would spread each order over neighbouring bins and turn a closed form into a
#: fitting exercise.
PERIODS = 8
SAMPLES_PER_PERIOD = SHAPE[1] // PERIODS

#: Eight float32 epsilons (1.19e-7 each) on a ratio normalized to the zeroth
#: order. Derived, not fitted: the FFT's rounding scale is set by the largest
#: term, so the *absolute* residual on `A_n / A_0` is a few epsilons regardless of
#: how small `A_n` is -- which is why the comparison is absolute rather than
#: relative. The worst measured residual is 9.2e-9, two orders under.
ORDER_ABS_TOLERANCE = 1e-6

#: The same bound relaxed by a factor of four for the binary grating, whose ratio
#: is normalized to the *first* order rather than to the largest one. Worst
#: measured residual 4.3e-6 against the continuum value at order 5, which is the
#: discretization and is reported as such below; against the sampled form it is
#: under 1e-6.
BINARY_ABS_TOLERANCE = 4e-6


def _front_focal_plane() -> ReferenceSurface:
    return ReferenceSurface(name="front_focal", z_m=0.0, medium_index=MEDIUM_INDEX)


def _illumination(kx_rad_per_m: float = 0.0) -> ScalarField:
    """A unit plane wave at the front focal plane, tilted in `x` by `kx`."""
    return plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=_front_focal_plane(),
        transverse_wavevector_rad_per_m=(0.0, kx_rad_per_m),
    )


def _to_fourier_plane(field: ScalarField) -> ScalarField:
    return focal_plane_transform(
        field, focal_length_m=FOCAL_LENGTH_M, model={"target_surface": "fourier"}
    )


def _x_coordinate_m() -> np.ndarray:
    return (np.arange(SHAPE[1]) - SHAPE[1] // 2) * PITCH_M[1]


def _along_x(profile: np.ndarray) -> np.ndarray:
    """Broadcast an `x` profile across the whole `(ny, nx)` grid."""
    return np.broadcast_to(profile[None, :], SHAPE).copy()


def _orders(field: ScalarField, count: int) -> dict[int, complex]:
    """The complex amplitude at each analytic order position, `-count .. +count`."""
    u = np.asarray(field.u)
    y0, x0 = SHAPE[0] // 2, SHAPE[1] // 2
    return {n: complex(u[y0, x0 + n * PERIODS]) for n in range(-count, count + 1)}


# ---------------------------------------------------------------------------
# 1. The sinusoidal phase grating: Jacobi-Anger, signed
# ---------------------------------------------------------------------------

#: Modulation depth. 1.5 rad spreads real amplitude over five orders --
#: `J_0..J_4(1.5)` are 0.512, 0.558, 0.232, 0.061, 0.012 -- so the test measures a
#: shape and not a single number, and `J_1 > J_0` makes a model that always puts
#: the most power on axis fail.
DEPTH_RAD = 1.5


def test_a_sinusoidal_phase_grating_gives_the_bessel_orders() -> None:
    """Criterion 2, the primary gate. Positions **and** signed relative amplitudes."""
    period_m = SHAPE[1] * PITCH_M[1] / PERIODS
    phase = _along_x(DEPTH_RAD * np.sin(2.0 * np.pi * _x_coordinate_m() / period_m))

    grating = complex_transmission(_illumination(), phase_rad=phase, target_surface="object")
    spectrum = _to_fourier_plane(grating)

    # The order positions are the analytic ones, expressed in the plane's own
    # declared pitch rather than in bins: x_n = n lambda f / (n_med Lambda).
    dx_out = spectrum.sample_pitch_m[1]
    for n in (1, 3):
        analytic_m = n * WAVELENGTH_M * FOCAL_LENGTH_M / (MEDIUM_INDEX * period_m)
        assert analytic_m == pytest.approx(n * PERIODS * dx_out, rel=1e-12)

    orders = _orders(spectrum, 4)
    reference = jv(0, DEPTH_RAD)
    assert reference != 0.0
    worst = 0.0
    for n, amplitude in orders.items():
        expected = jv(n, DEPTH_RAD) / reference
        residual = abs(amplitude / orders[0] - expected)
        worst = max(worst, residual)
        assert residual < ORDER_ABS_TOLERANCE, (
            f"order {n:+d} came back at {amplitude / orders[0]!r} where "
            f"J_{n}({DEPTH_RAD})/J_0({DEPTH_RAD}) = {expected!r}"
        )
    assert worst < ORDER_ABS_TOLERANCE

    # The sign is the claim a magnitude comparison would miss: J_{-n} = (-1)^n J_n,
    # so the odd negative orders are *negative* and a mirrored spectrum would pass
    # on |A_n| alone.
    assert (orders[-1] / orders[+1]).real == pytest.approx(-1.0, abs=1e-6)
    assert (orders[-2] / orders[+2]).real == pytest.approx(+1.0, abs=1e-6)
    assert jv(1, DEPTH_RAD) > jv(0, DEPTH_RAD), "the case must not be on-axis-dominated"


def test_the_grating_spectrum_lands_only_on_the_order_positions() -> None:
    """The other half of "at the analytic order positions": nothing in between.

    A grating that is periodic on the grid has a line spectrum, so the bins
    between the orders carry nothing but round-off. This is what would fail if the
    period were not commensurate with the window -- the failure mode R06.5's
    criterion 5 characterizes and R06.8 sweeps into.
    """
    period_m = SHAPE[1] * PITCH_M[1] / PERIODS
    phase = _along_x(DEPTH_RAD * np.sin(2.0 * np.pi * _x_coordinate_m() / period_m))
    spectrum = _to_fourier_plane(complex_transmission(_illumination(), phase_rad=phase))

    u = np.asarray(spectrum.u)
    y0, x0 = SHAPE[0] // 2, SHAPE[1] // 2
    on_order = np.zeros(SHAPE, dtype=bool)
    on_order[y0, x0 :: PERIODS] = True
    on_order[y0, x0 :: -PERIODS] = True

    total = float(np.sum(np.abs(u) ** 2))
    off_order = float(np.sum(np.abs(u[~on_order]) ** 2))
    assert off_order / total < 1e-10, (
        "power landed off the analytic order positions, which is what a period "
        "incommensurate with the window looks like"
    )
    # The grating varies in x only, so the whole spectrum is on one row.
    assert float(np.sum(np.abs(u[y0]) ** 2)) / total == pytest.approx(1.0, abs=1e-10)


def test_a_binary_phase_grating_gives_its_known_order_weights() -> None:
    """Criterion 2, second case. Even orders vanish; odd ones follow the sampled
    closed form, with the continuum `1/n` reported as the limit it is."""
    half = SAMPLES_PER_PERIOD // 2
    binary = np.pi * ((np.arange(SHAPE[1]) // half) % 2).astype(np.float64)
    grating = complex_transmission(_illumination(), phase_rad=_along_x(binary))
    orders = _orders(_to_fourier_plane(grating), 5)

    first = abs(orders[+1])
    assert first > 0.0

    # phi = pi: t = 1 - 2 s, and the 50%-duty square wave s has mean 1/2, so the
    # zeroth order is *exactly* cancelled. Nothing else here would catch a duty
    # cycle that is off by one sample.
    assert abs(orders[0]) / first < 1e-5
    for even in (2, 4, -2, -4):
        assert abs(orders[even]) / first < 1e-5

    sampled = {
        n: math.sin(math.pi / SAMPLES_PER_PERIOD)
        / math.sin(n * math.pi / SAMPLES_PER_PERIOD)
        for n in (3, 5)
    }
    for n, expected in sampled.items():
        for order in (n, -n):
            assert abs(abs(orders[order]) / first - expected) < BINARY_ABS_TOLERANCE, (
                f"order {order:+d} came back at {abs(orders[order]) / first} where the "
                f"sampled closed form is {expected}"
            )
        # ...and the continuum value is the limit, at the tolerance the
        # discretization actually justifies rather than at the arithmetic floor.
        assert abs(expected - 1.0 / n) < 0.05 / n
        assert abs(orders[n]) / first == pytest.approx(1.0 / n, abs=0.05 / n)


# ---------------------------------------------------------------------------
# 2. A finite aperture cuts where it is told to
# ---------------------------------------------------------------------------

#: NA 0.1 in air at f = 20 mm: a stop radius of exactly 2.000 mm and a cutoff
#: frequency of 187969.9 /m, which is 24.0602 samples of the Fourier plane's own
#: pitch. The step therefore has to fall between bins 24 and 25 -- a prediction
#: with no adjustable part in it.
NUMERICAL_APERTURE = 0.1


def test_the_aperture_that_realizes_an_na_passes_exactly_the_frequencies_below_it() -> None:
    """Criterion 3, the part that ties the two halves together.

    A single spatial frequency enters as a tilted plane wave, focuses to one
    sample of the Fourier plane, and the stop either passes it or does not. The
    analytic predicate is `|f_x| <= NA / lambda`; the measured step lands between
    the two bins straddling `NA / lambda`, which is where arithmetic says it is and
    not one bin either side.

    The two straddling cases are the gate. The far-inside and far-outside cases
    are there so that a model with the *wrong* cutoff still fails: a radius off by
    the medium index or by `2 pi` moves the step by tens of bins.
    """
    radius_m = numerical_aperture_radius_m(
        NUMERICAL_APERTURE, focal_length_m=FOCAL_LENGTH_M, medium_index=MEDIUM_INDEX
    )
    assert radius_m == pytest.approx(FOCAL_LENGTH_M * NUMERICAL_APERTURE / MEDIUM_INDEX)

    cutoff_per_m = NUMERICAL_APERTURE / WAVELENGTH_M
    window_m = SHAPE[1] * PITCH_M[1]
    cutoff_bin = cutoff_per_m * window_m
    assert 24.0 < cutoff_bin < 25.0, "the case must actually straddle two bins"

    stop_pitch_m = (
        WAVELENGTH_M * FOCAL_LENGTH_M / (MEDIUM_INDEX * SHAPE[0] * PITCH_M[0]),
        WAVELENGTH_M * FOCAL_LENGTH_M / (MEDIUM_INDEX * SHAPE[1] * PITCH_M[1]),
    )
    stop = circular_aperture_amplitude(
        SHAPE, sample_pitch_m=stop_pitch_m, radius_m=radius_m, edge="hard"
    )

    transmitted: dict[int, float] = {}
    for bin_index in (19, 24, 25, 29):
        # An integer bin puts the whole plane wave on one output sample, so the
        # measurement is the predicate and not a windowing artefact.
        kx = 2.0 * math.pi * bin_index / window_m
        spectrum = _to_fourier_plane(_illumination(kx))
        assert spectrum.sample_pitch_m == pytest.approx(stop_pitch_m, rel=1e-12)
        before = spectrum.discrete_power()
        after = complex_transmission(spectrum, amplitude=stop).discrete_power()
        transmitted[bin_index] = after / before

    inside = {index for index in transmitted if index <= cutoff_bin}
    assert inside == {19, 24}
    for index in sorted(transmitted):
        fraction = transmitted[index]
        if index in inside:
            assert fraction == pytest.approx(1.0, rel=1e-6), (
                f"bin {index} is inside the analytic cutoff {cutoff_bin:.4f} and was blocked"
            )
        else:
            assert fraction < 1e-12, (
                f"bin {index} is outside the analytic cutoff {cutoff_bin:.4f} and passed"
            )
