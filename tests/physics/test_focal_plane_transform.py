"""The ideal lens against closed-form Fourier optics, and the two signs in it.

CHE-209 (R06.4) acceptance criteria 1-6. Six closed forms, each computed here
from its definition rather than from another run of the code under test:

===========================  =========================================  ===========
case                         oracle                                     measured
===========================  =========================================  ===========
output sampling              ``lambda f / (n N dx)``, per axis          exact (f64)
a delta transforms           uniform ``|U| = dy dx / (lambda f / n)``   1e-8 rel
its offset becomes a ramp    slope ``-2 pi a n / (lambda f)``           5e-8 rel
a tilt focuses               ``x = f sin(theta)``                       1.5e-7 rel
a slit gives a sinc          nulls at ``m lambda f / (n w)``            < 1e-6 peak
forward then inverse         ``-U_in`` (see below)                      1.5e-7 rel
power                        discrete Parseval: unchanged               exact (f64)
===========================  =========================================  ===========

`tests/solvers/test_chromatix_focal_plane.py` holds the refusals, the pad-width
decision and the registration; this file is the physics.

Two signs, both measured rather than assumed
----------------------------------------------
**The focus lands at ``f sin(theta)``, not ``f tan(theta)``.** A single optical
Fourier transform maps spatial frequency linearly onto position, and a plane wave
at ``theta`` carries ``f_x = n sin(theta) / lambda``, so ``x = lambda f f_x / n =
f sin(theta)`` exactly. At 20 degrees on this grid the two predictions differ by
2.6 output samples, and the test asserts the measurement **rejects**
``f tan(theta)``. The gap is the sine-condition content of the ideal-lens model,
not an implementation error; it is recorded in the operator's `approximation`.

**Forward then inverse returns ``-U_in``, not ``U_in``.** Each leg carries the
textbook ``1 / (i lambda f / n)`` prefactor, so the pair carries ``(1/i)^2 =
-1``: a global pi. Measured at 3.14159274 rad, and the residual against ``-U_in``
is 1.5e-7 where the residual against ``+U_in`` is 2.0 -- i.e. the sign is the
whole difference and nothing else is wrong. The backend's own docstring says the
pair "yields the same ``Field``", which is true of ``|U|^2`` and false of the
amplitude; this project states the sign instead. It is invisible in intensity and
it is exactly the class of piston `'carrier_removed_phase'` exists to flag, which
both legs declare.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from representations import ReferenceSurface, ScalarField
from solvers.chromatix import focal_plane_transform, fourier_plane_pitch_m

WAVELENGTH_M = 0.532e-6
FOCAL_LENGTH_M = 20e-3

#: The complex64 floor. Every residual below is a single transform pair on a
#: 128^2 grid, and 1.5e-7 is what one float32 epsilon per sample accumulates to;
#: nothing here is tolerance-fitted.
COMPLEX64_FLOOR = 1e-6


def a_field(
    u: np.ndarray, pitch_m: tuple[float, float], *, name: str = "front_focal", n: float = 1.0
) -> ScalarField:
    return ScalarField(
        u=u.astype(np.complex64),
        sample_pitch_m=pitch_m,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name=name, z_m=0.0, medium_index=n),
    )


def transform(field: ScalarField, *, f: float = FOCAL_LENGTH_M, **model: object) -> ScalarField:
    return focal_plane_transform(
        field, focal_length_m=f, model={"target_surface": "back_focal", **model}
    )


# ---------------------------------------------------------------------------
# 1. Sampling: the declared pitch is the analytic one, on a doubly odd grid
# ---------------------------------------------------------------------------


def test_the_output_pitch_is_the_analytic_fourier_pitch() -> None:
    """Criterion 1, and R06.3's criterion 1 end to end through the backend.

    48 x 64 samples at 0.30 x 0.25 um: asymmetric in **both** count and pitch, so
    the two axes have different extents and a transposed `(y, x)` cannot pass. The
    expectation is computed from `df = 1/(N dx)` in float64; the backend never
    supplies it, it is only checked against it -- and the value that comes back is
    the float64 declaration, so this is `==` and not `approx`.
    """
    shape, pitch = (48, 64), (0.30e-6, 0.25e-6)
    u = np.zeros(shape)
    u[shape[0] // 2, shape[1] // 2] = 1.0

    out = transform(a_field(u, pitch))

    analytic = tuple(
        WAVELENGTH_M * FOCAL_LENGTH_M / (1.0 * count * step)
        for count, step in zip(shape, pitch, strict=True)
    )
    assert out.sample_pitch_m == fourier_plane_pitch_m(
        pitch, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.0
    )
    assert out.sample_pitch_m == pytest.approx(analytic, rel=1e-15)
    assert out.sample_pitch_m[0] != out.sample_pitch_m[1], "the two axes must differ here"
    assert out.shape == shape


def test_the_transform_declares_the_piston_it_does_not_carry() -> None:
    """Criterion: the removed `exp(i k n 2f)` is in the type, not in prose.

    The backend applies the textbook `1/(i lambda f / n)` prefactor and no `exp`
    factor at all, so the returned phase is relative to a removed piston. `|U|^2`
    cannot see it, which is why it is declared.
    """
    u = np.zeros((32, 32))
    u[16, 16] = 1.0
    out = transform(a_field(u, (0.5e-6, 0.5e-6)))

    assert "carrier_removed_phase" in out.validity
    assert out.reference_surface.name == "back_focal"
    assert out.reference_surface.z_m == pytest.approx(2.0 * FOCAL_LENGTH_M)


# ---------------------------------------------------------------------------
# 2. A point becomes a plane wave, with the analytic slope and prefactor
# ---------------------------------------------------------------------------

GRID = 128
PITCH_M = 0.5e-6


def test_a_delta_becomes_a_uniform_wave_of_analytic_amplitude() -> None:
    """Criterion 2, first half, plus the normalization the power test cannot localize.

    A unit sample transforms to `|U| = dy dx / (lambda f / n)` everywhere: the
    prefactor with no diffraction in the way. It is the sharpest normalization
    check available, because it fixes the *absolute* scale rather than a ratio.
    """
    u = np.zeros((GRID, GRID))
    u[GRID // 2, GRID // 2] = 1.0

    out = transform(a_field(u, (PITCH_M, PITCH_M)))

    amplitude = np.abs(np.asarray(out.u))
    analytic = PITCH_M * PITCH_M / (WAVELENGTH_M * FOCAL_LENGTH_M / 1.0)
    assert amplitude.max() == pytest.approx(analytic, rel=1e-6)
    assert amplitude.min() == pytest.approx(analytic, rel=1e-6), "a delta transforms to a flat |U|"


def test_the_offset_of_the_delta_is_the_slope_of_the_ramp() -> None:
    """Criterion 2, second half. Sign included, and the sign is negative.

    `U_b(x) = C exp(-i 2 pi a x n / (lambda f))` for a delta at `x' = a`: the
    forward kernel's sign under this project's `exp(-i omega t)` / `exp(+i k z)`
    convention. The conjugate reading is reported as the falsifiable twin, and it
    is a factor of -1 away rather than a marginal difference.
    """
    offset_samples = 5
    u = np.zeros((GRID, GRID))
    u[GRID // 2, GRID // 2 + offset_samples] = 1.0

    out = transform(a_field(u, (PITCH_M, PITCH_M)))

    pitch_out = out.sample_pitch_m[1]
    phase = np.unwrap(np.angle(np.asarray(out.u)[GRID // 2, :]))
    measured = float(np.polyfit(np.arange(GRID) * pitch_out, phase, 1)[0])

    a_m = offset_samples * PITCH_M
    analytic = -2.0 * math.pi * a_m / (WAVELENGTH_M * FOCAL_LENGTH_M / 1.0)
    assert measured == pytest.approx(analytic, rel=1e-5)
    assert measured != pytest.approx(-analytic, rel=0.5), (
        "the exp(+i...) reading is the falsifiable twin and must not also pass"
    )


# ---------------------------------------------------------------------------
# 3. A plane wave becomes a point -- at f sin(theta)
# ---------------------------------------------------------------------------


def test_a_tilted_plane_wave_focuses_at_f_sin_theta_and_not_f_tan_theta() -> None:
    """Criterion 3, on one axis only and with both signs, at an angle that separates
    the two candidate oracles.

    20 degrees: `f sin = 6.8404 mm`, `f tan = 7.2794 mm`, output pitch 166.3 um, so
    the predictions are 2.6 samples apart and the measurement can choose. The
    envelope is a super-Gaussian rather than a hard window because a step edge puts
    power past the light cone (measured at 2.2e-2 by the ASM round trip), which
    would broaden the focus for a reason that has nothing to do with the claim.
    """
    theta_rad = math.radians(20.0)
    coordinate = (np.arange(GRID) - GRID // 2) * PITCH_M
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    envelope = np.exp(-(((x**2 + y**2) / (0.4 * GRID * PITCH_M) ** 2) ** 4))

    for sign in (+1.0, -1.0):
        k_x = sign * 2.0 * math.pi * math.sin(theta_rad) / WAVELENGTH_M
        out = transform(a_field(envelope * np.exp(1j * k_x * x), (PITCH_M, PITCH_M)))

        intensity = np.abs(np.asarray(out.u)) ** 2
        row = intensity[GRID // 2, :]
        peak = int(np.argmax(row))
        window = slice(max(0, peak - 6), min(GRID, peak + 7))
        index = np.arange(GRID)[window]
        centroid = float((row[window] * index).sum() / row[window].sum())
        measured_m = (centroid - GRID // 2) * out.sample_pitch_m[1]

        expected_m = sign * FOCAL_LENGTH_M * math.sin(theta_rad)
        rejected_m = sign * FOCAL_LENGTH_M * math.tan(theta_rad)
        assert measured_m == pytest.approx(expected_m, rel=1e-4)
        assert abs(measured_m - rejected_m) > 2.0 * out.sample_pitch_m[1], (
            f"f tan(theta) = {rejected_m} m is the other candidate and must be rejected"
        )

        # A tilt in x must not move the focus in y: that is what a transposition
        # would look like, and it is invisible in a rotationally symmetric case.
        column_peak = int(np.argmax(intensity[:, peak]))
        assert column_peak == GRID // 2


# ---------------------------------------------------------------------------
# 4. A finite aperture gives the analytic sinc
# ---------------------------------------------------------------------------


def test_a_slit_transforms_to_a_sinc_with_analytic_nulls() -> None:
    """Criterion 4. Null positions and the first-sidelobe ratio, both closed form.

    A slit of width `w` transforms to `sinc(x n w / (lambda f))`, whose intensity
    zeros sit at `m lambda f / (n w)` and whose first sidelobe is 0.047180 of the
    peak (the recorded L1-WAVE-01 value, and `sinc^2` evaluated at its own first
    stationary point). The slit is 16 samples wide, which puts the first null at
    exactly 8 output samples -- so the nulls land on grid points and can be read
    without interpolation.
    """
    slit_samples = 16
    u = np.zeros((GRID, GRID))
    u[:, GRID // 2 - slit_samples // 2 : GRID // 2 + slit_samples // 2] = 1.0

    out = transform(a_field(u, (PITCH_M, PITCH_M)))

    intensity = np.abs(np.asarray(out.u)[GRID // 2, :]) ** 2
    peak = intensity[GRID // 2]
    width_m = slit_samples * PITCH_M
    null_m = WAVELENGTH_M * FOCAL_LENGTH_M / (1.0 * width_m)
    null_samples = null_m / out.sample_pitch_m[1]
    assert null_samples == pytest.approx(8.0, rel=1e-12)

    for order in (1, 2, 3):
        index = GRID // 2 + round(order * null_samples)
        assert intensity[index] / peak < 1e-6, f"the sinc's null {order} is not a null"

    first_lobe = intensity[GRID // 2 + 9 : GRID // 2 + 16].max() / peak
    assert first_lobe == pytest.approx(0.047180, abs=2e-3), (
        "the sinc^2 first sidelobe ratio is a closed form and is 0.047180"
    )


# ---------------------------------------------------------------------------
# 5. The pair is unitary up to the pi it carries, and power is conserved
# ---------------------------------------------------------------------------


def a_gaussian() -> ScalarField:
    coordinate = (np.arange(GRID) - GRID // 2) * PITCH_M
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    return a_field(np.exp(-(x**2 + y**2) / (6e-6) ** 2), (PITCH_M, PITCH_M))


def test_forward_then_inverse_returns_the_input_times_minus_one() -> None:
    """Criterion 5, stated as what it actually is.

    The pitch claim is an identity -- the analytic relation inverts itself exactly
    in float64 -- while the field claim is a residual at the complex64 floor
    against `-U_in`. Both readings are asserted, because a test that only checked
    `|U|` would pass under a sign error and a test that only checked the sign
    would pass under a scale error.
    """
    source = a_gaussian()
    forward = transform(source)
    back = focal_plane_transform(
        forward,
        focal_length_m=FOCAL_LENGTH_M,
        model={"target_surface": "front_focal", "direction": "inverse"},
    )

    assert back.sample_pitch_m == source.sample_pitch_m
    assert back.reference_surface.z_m == pytest.approx(0.0, abs=1e-15)

    returned, original = np.asarray(back.u), np.asarray(source.u)
    residual = float(np.linalg.norm(returned + original) / np.linalg.norm(original))
    assert residual < COMPLEX64_FLOOR

    # ... and the twin: it is *not* the input itself, by a factor of exactly -1.
    assert float(np.linalg.norm(returned - original) / np.linalg.norm(original)) > 1.9
    piston = float(np.angle(np.sum(returned * np.conj(original))))
    assert abs(abs(piston) - math.pi) < 1e-5


def test_the_transform_conserves_discrete_power() -> None:
    """Criterion 6. Discrete Parseval, which is the one gate a phase or position
    check cannot substitute for.

    `sum |u|^2 dy dx` is invariant because the backend's `-i dy dx / (lambda f / n)`
    prefactor is exactly the factor that makes the DFT unitary in these units. A
    normalization error is invisible to every other criterion in this file except
    the delta-amplitude one, and invisible to any peak-normalized metric at all.
    """
    source = a_gaussian()
    forward = transform(source)
    assert forward.discrete_power() == pytest.approx(source.discrete_power(), rel=1e-6)

    # Also on a case whose spectrum fills the window rather than concentrating.
    noise_like = a_field(
        np.cos(2.0 * math.pi * np.arange(GRID) / 7.0)[None, :] * np.ones((GRID, 1)),
        (PITCH_M, PITCH_M),
    )
    assert transform(noise_like).discrete_power() == pytest.approx(
        noise_like.discrete_power(), rel=1e-6
    )
