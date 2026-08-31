"""Scalar-wave propagation against analytic closed forms, and the phasor sign.

CHE-184 (R06.2) acceptance criteria 1, 3 and 5. Four closed forms, each with the
tolerance the reference implementation's `B1-WAVE-*` families justified and the
number they recorded:

===========================  ==========================  =========  ==========
case                         oracle                      tolerance  recorded
===========================  ==========================  =========  ==========
plane-wave phase advance     ``k_z z``, exact             1e-2 rad  4.90e-6
Gaussian spreading           ``w0 sqrt(1 + (z/zR)^2)``    2e-2 rel  1.82e-4
tilted-beam walk-off         ``z tan(theta)``, signed     2e-2 rel  9.93e-5
forward/backward round trip  the input; ASM is unitary    1e-5 rel  2.75e-7
===========================  ==========================  =========  ==========

Every tolerance is inherited rather than chosen here, and every case reproduces
its recorded number: 5.0e-6, 1.8e-4 and 2.75e-7 land on the frozen values to two
or three significant figures. The walk-off is 4.3e-7 rather than 9.9e-5 because
the beam is a wider Gaussian than the frozen instance's, which is a different
measurement of the same claim and is *better*, not equal -- so it is reported as
what it is rather than pinned to a number it should not reproduce.

Which oracle decides, and which does not
-----------------------------------------
Every gate below is a closed form: the angular spectrum's own ``k_z``, the
paraxial Gaussian, straight-line geometry, and the unitarity of free-space
propagation. This repository's float64 ASM implementation
(``pre-rewrite-2026-08-30:src/verification/asm_oracle.py``) is deliberately **not**
ported and gates nothing. It would agree with the code it judges, and the
agreement would mean nothing; letting custom numerical code certify numerical
code is circular validation.

The phasor sign, and why the grid is deliberately lopsided
-----------------------------------------------------------
`representations.PHASOR` is ``exp(-i omega t)`` and `SPATIAL_FACTOR` is therefore
``exp(+i k z)``. The conjugate convention is invisible in ``|U|^2`` and reverses
every phase, so it is measured rather than assumed: section 1 propagates a
manufactured travelling wave and compares its advance against ``+k_z z``, with the
``exp(-i k z)`` convention's prediction reported beside it as the falsifiable twin.

That case runs on a grid that is asymmetric in **both** ways -- 192 x 256 samples
at 0.30 x 0.25 um -- and carries its transverse frequency on one axis only, so a
transposed ``(y, x)`` cannot hide inside it: swapping the two pitches moves the
predicted advance by 0.0899 rad, nearly nine times the tolerance, and the test
asserts the measurement rejects that reading. A square grid with equal pitch would pass under
either convention.

Cost: eleven propagations, the largest on a 1024^2 complex64 grid, all on the
host. Nothing here is marked `slow`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from representations import ReferenceSurface, ScalarField
from solvers.chromatix import propagate

WAVELENGTH_M = 0.532e-6
K = 2.0 * math.pi / WAVELENGTH_M

#: Inherited tolerances. Each is the `B1-WAVE-*` family's `Tolerance.threshold`,
#: with the basis that justified it; none was chosen to make a number pass.
PHASE_TOLERANCE_RAD = 1e-2
GAUSSIAN_TOLERANCE = 2e-2
WALKOFF_TOLERANCE = 2e-2
ROUND_TRIP_TOLERANCE = 1e-5


def _field(u: np.ndarray, pitch_m: tuple[float, float]) -> ScalarField:
    """A field on the source plane at `z = 0` in vacuum."""
    return ScalarField(
        u=u,
        sample_pitch_m=pitch_m,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="source", z_m=0.0, medium_index=1.0),
    )


def _asm(field: ScalarField, distance_m: float, *, pad_width: int = 0) -> ScalarField:
    """One absolute-phase propagation, cropped back to the input window.

    `pad_width=0` throughout this module and that is a physical choice, not a
    shortcut: every case here is either exactly periodic on its grid (the plane
    wave, which zero padding would destroy) or confined to a small fraction of a
    window much larger than the distance spreads it. `edge_energy_fraction` is
    what would say otherwise, and the round-trip case checks it.
    """
    return propagate(
        field,
        distance_m=distance_m,
        model={"method": "asm", "pad_width": pad_width, "target_surface": "target"},
    )


def _axes(field: ScalarField) -> tuple[np.ndarray, np.ndarray]:
    """`(y, x)` coordinate vectors in metres, on the field's own `n // 2` origin."""
    y, x = field.coordinates()
    return np.asarray(y), np.asarray(x)


# ---------------------------------------------------------------------------
# 1. The manufactured travelling wave: exp(+i k_z z), on a lopsided grid
# ---------------------------------------------------------------------------

#: 192 x 256 samples at 0.30 x 0.25 um. Both axes differ in count *and* in pitch,
#: which is what makes a transposition detectable at all.
PLANE_SHAPE = (192, 256)
PLANE_PITCH_M = (0.30e-6, 0.25e-6)
PLANE_DISTANCE_M = 20e-6
#: Bin 6 of the x axis, so the wave is exactly periodic on the grid and nothing
#: leaks. Non-zero on purpose: on axis `k_z = k` exactly and a frequency grid
#: scaled by 2*pi would be invisible.
PLANE_BIN = 6


def _plane_wave(transverse_frequency_per_m: float) -> ScalarField:
    """`exp(+2 pi i f x)` -- a tilt in **x** only, so the two axes are not alike."""
    field = _field(np.ones(PLANE_SHAPE, dtype=np.complex64), PLANE_PITCH_M)
    _, x = _axes(field)
    u = np.exp(2j * np.pi * transverse_frequency_per_m * x)[None, :]
    return _field(np.broadcast_to(u, PLANE_SHAPE).astype(np.complex64), PLANE_PITCH_M)


def _uniform_advance(field: ScalarField, distance_m: float) -> float:
    """The phase `arg(U_out / U_in)` of a field that advances uniformly, in radians.

    Averaged over the grid before the angle is taken, so the estimate is the
    field's own mean phasor rather than one pixel's.
    """
    out = _asm(field, distance_m)
    ratio = np.asarray(out.u) / np.asarray(field.u)
    return float(np.angle(np.mean(ratio)))


def _axial_wavenumber(transverse_frequency_per_m: float) -> float:
    """`k_z = sqrt(k^2 - (2 pi f)^2)`, the angular spectrum's own definition."""
    transverse = 2.0 * math.pi * transverse_frequency_per_m
    return math.sqrt(K * K - transverse * transverse)


def _wrapped(radians: float) -> float:
    """`radians` folded into `(-pi, pi]`, for comparing against a phase that wraps."""
    return float(np.angle(np.exp(1j * radians)))


def test_a_propagated_plane_wave_advances_by_plus_k_z_z() -> None:
    """Criterion 3: the manufactured travelling wave, against `exp(+i k_z z)`.

    The absolute advance is 236 rad and wraps, so the comparison is made modulo
    2*pi -- unambiguously, because the float64 prediction and a measurement good
    to ~2e-5 rad cannot be a cycle apart.
    """
    frequency = PLANE_BIN / (PLANE_SHAPE[1] * PLANE_PITCH_M[1])
    measured = _uniform_advance(_plane_wave(frequency), PLANE_DISTANCE_M)
    predicted = _axial_wavenumber(frequency) * PLANE_DISTANCE_M

    residual = abs(_wrapped(measured - predicted))
    assert residual < PHASE_TOLERANCE_RAD, (
        f"a plane wave propagated {PLANE_DISTANCE_M} m advanced by {measured} rad "
        f"(mod 2 pi) where exp(+i k_z z) requires {_wrapped(predicted)}: {residual} rad off"
    )
    # The conjugate convention, as a prediction rather than a mutation: it puts the
    # advance at -k_z z. What a wrapped comparison can see is the residue of 2 k_z z,
    # which here is 0.593 rad -- 59 tolerances, decisive, but a number that depends on
    # where the wrap happens to fall. The off-axis case below is the one whose
    # separation does not.
    assert abs(_wrapped(measured + predicted)) > 50 * PHASE_TOLERANCE_RAD


def test_the_off_axis_advance_matches_k_z_and_rejects_a_transposed_pitch() -> None:
    """The relative advance `(k_z - k) z`, its sign, and the axis-order guard.

    Measured against an on-axis plane wave propagated the same distance, so the
    quantity is `(k_z - k) z` -- small, negative, and not wrapped. Recorded by
    `B1-WAVE-PLANEPHASE-01`: 4.89889e-6 rad of residual against a 1e-2 gate.

    The transposition guard is the second assertion. If `(dy, dx)` were handed to
    the backend the other way round, the ramp's bin would be read as
    `6 / (nx * dy)` and the same measurement would have to agree with a prediction
    0.0899 rad away. It does not, by nearly nine tolerances.
    """
    frequency = PLANE_BIN / (PLANE_SHAPE[1] * PLANE_PITCH_M[1])
    on_axis = _uniform_advance(_plane_wave(0.0), PLANE_DISTANCE_M)
    off_axis = _uniform_advance(_plane_wave(frequency), PLANE_DISTANCE_M)
    measured = _wrapped(off_axis - on_axis)

    predicted = (_axial_wavenumber(frequency) - K) * PLANE_DISTANCE_M
    assert predicted < 0.0, "k_z < k off axis; a positive prediction is a sign error here"
    assert abs(measured - predicted) < PHASE_TOLERANCE_RAD

    transposed_frequency = PLANE_BIN / (PLANE_SHAPE[1] * PLANE_PITCH_M[0])
    transposed = (_axial_wavenumber(transposed_frequency) - K) * PLANE_DISTANCE_M
    assert abs(measured - transposed) > 8 * PHASE_TOLERANCE_RAD, (
        "the measurement cannot tell a (dy, dx) transposition from the declared order, "
        "so this case is not the axis-asymmetric one it claims to be"
    )


# ---------------------------------------------------------------------------
# 2. Gaussian spreading
# ---------------------------------------------------------------------------

GAUSSIAN_GRID = 1024
GAUSSIAN_PITCH_M = 0.25e-6
GAUSSIAN_WAIST_M = 5e-6
GAUSSIAN_DISTANCE_M = 100e-6


def _intensity_radius_m(field: ScalarField) -> float:
    """The `1/e^2` intensity radius from the second moment of the x marginal.

    For `I = exp(-2 x^2 / w^2)` the second moment is `w^2 / 4`, so `w = 2 sigma`.
    A second moment and not a fit: a fit would impose the Gaussian shape the
    propagation is being asked to preserve.
    """
    intensity = np.abs(np.asarray(field.u)) ** 2
    _, x = _axes(field)
    total = float(intensity.sum())
    return 2.0 * math.sqrt(float((intensity * x[None, :] ** 2).sum()) / total)


def test_a_gaussian_beam_spreads_by_w0_sqrt_one_plus_z_over_zr_squared() -> None:
    """Criterion 1, against the paraxial Gaussian, which is exact for this beam.

    Recorded by `B1-WAVE-GAUSS-01`: 1.82072e-4 relative error against a 2e-2 gate,
    at exactly these parameters. The grid is 1024 and not 512 because the family's
    own sampling predicate `z <= N pitch^2 / lambda` is 60.15 um at 512 against a
    100 um propagation -- a metric inside its tolerance while its validity claim
    contradicts itself is not a pass.
    """
    sampling_limit_m = GAUSSIAN_GRID * GAUSSIAN_PITCH_M**2 / WAVELENGTH_M
    assert sampling_limit_m > GAUSSIAN_DISTANCE_M

    coordinate = (np.arange(GAUSSIAN_GRID) - GAUSSIAN_GRID // 2) * GAUSSIAN_PITCH_M
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    amplitude = np.exp(-(x**2 + y**2) / GAUSSIAN_WAIST_M**2).astype(np.complex64)
    source = _field(amplitude, (GAUSSIAN_PITCH_M, GAUSSIAN_PITCH_M))

    rayleigh_m = math.pi * GAUSSIAN_WAIST_M**2 / WAVELENGTH_M
    expected_m = GAUSSIAN_WAIST_M * math.sqrt(1.0 + (GAUSSIAN_DISTANCE_M / rayleigh_m) ** 2)

    measured_m = _intensity_radius_m(_asm(source, GAUSSIAN_DISTANCE_M))
    error = abs(measured_m - expected_m) / expected_m
    assert error < GAUSSIAN_TOLERANCE, f"w(z) = {measured_m} m against {expected_m} m analytic"

    # The negative control the family names: the unpropagated waist is 17% low,
    # so a run that propagated nothing cannot pass this gate.
    unpropagated = abs(_intensity_radius_m(source) - expected_m) / expected_m
    assert unpropagated > GAUSSIAN_TOLERANCE


# ---------------------------------------------------------------------------
# 3. Tilted-beam walk-off, with its sign and its axis
# ---------------------------------------------------------------------------

TILT_GRID = 1024
TILT_PITCH_M = 0.5e-6
TILT_RAD = 0.08726646259971647  # 5 degrees
TILT_DISTANCE_M = 200e-6
TILT_WAIST_M = 30e-6


def _tilted_beam(sign: float) -> ScalarField:
    """A wide Gaussian carrying `exp(+i k sin(theta) y)` -- a tilt in **y** only."""
    coordinate = (np.arange(TILT_GRID) - TILT_GRID // 2) * TILT_PITCH_M
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    envelope = np.exp(-(x**2 + y**2) / TILT_WAIST_M**2)
    ramp = np.exp(1j * K * sign * math.sin(TILT_RAD) * y)
    return _field((envelope * ramp).astype(np.complex64), (TILT_PITCH_M, TILT_PITCH_M))


def _centroid_m(field: ScalarField) -> tuple[float, float]:
    """The `(y, x)` intensity centroid in metres."""
    intensity = np.abs(np.asarray(field.u)) ** 2
    y, x = _axes(field)
    total = float(intensity.sum())
    return (
        float((intensity * y[:, None]).sum()) / total,
        float((intensity * x[None, :]).sum()) / total,
    )


def test_a_tilted_beam_walks_off_by_z_tan_theta_toward_plus_y() -> None:
    """Criterion 1 for a convention claim: the displacement, its sign, and its axis.

    `z tan(theta)` is exact geometry for a collimated beam. The 2e-2 tolerance
    deliberately does **not** separate `z sin(theta)` from `z tan(theta)` -- they
    differ by 0.4% at 5 degrees -- and `B1-WAVE-TILT` said so rather than claiming
    a separation it does not have.

    The axis assertion is the transposition guard's other half: a `+y` tilt must
    move the beam in `+y` and not at all in `x`.
    """
    nyquist_pitch_m = WAVELENGTH_M / (2.0 * math.sin(TILT_RAD))
    sampling_limit_m = TILT_GRID * TILT_PITCH_M**2 / WAVELENGTH_M
    assert nyquist_pitch_m > TILT_PITCH_M and sampling_limit_m > TILT_DISTANCE_M

    expected_m = TILT_DISTANCE_M * math.tan(TILT_RAD)

    source = _tilted_beam(+1.0)
    before_y, before_x = _centroid_m(source)
    after_y, after_x = _centroid_m(_asm(source, TILT_DISTANCE_M))
    walkoff_m = after_y - before_y

    assert abs(walkoff_m / expected_m - 1.0) < WALKOFF_TOLERANCE, (
        f"a +{TILT_RAD} rad tilt walked {walkoff_m} m over {TILT_DISTANCE_M} m where "
        f"z tan(theta) = {expected_m} m"
    )
    assert abs(after_x - before_x) < WALKOFF_TOLERANCE * expected_m, (
        "a tilt in y displaced the beam in x, which is what a transposed (y, x) looks like"
    )

    # The sign is part of the claim: the opposite tilt walks the other way, and a
    # sign inversion is a 2.0 relative error rather than a marginal one.
    reversed_source = _tilted_beam(-1.0)
    reversed_before, _ = _centroid_m(reversed_source)
    reversed_after, _ = _centroid_m(_asm(reversed_source, TILT_DISTANCE_M))
    assert (reversed_after - reversed_before) < 0.0
    assert abs((reversed_after - reversed_before) / expected_m + 1.0) < WALKOFF_TOLERANCE


# ---------------------------------------------------------------------------
# 4. The round trip, and the two things it cannot see
# ---------------------------------------------------------------------------

ROUND_TRIP_GRID = 1024
ROUND_TRIP_PITCH_M = 0.25e-6
ROUND_TRIP_DISTANCE_M = 100e-6
#: `B1-WAVE-FWDBWD-01`'s `aperture_fill_fraction`: the aperture radius as a
#: fraction of the half-window.
ROUND_TRIP_FILL = 0.4


def _aperture(*, hard: bool) -> ScalarField:
    """A circular aperture, soft-edged by default.

    The edge is `exp(-(r/R)^8)` rather than a step, and the difference is physics
    rather than taste: a step edge puts real power past the light cone, and
    evanescent orders decay on the way out and cannot come back, so a hard edge
    sits outside this case's validity domain by construction. `hard=True` builds
    that case on purpose, as the control.
    """
    coordinate = (np.arange(ROUND_TRIP_GRID) - ROUND_TRIP_GRID // 2) * ROUND_TRIP_PITCH_M
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    radius = np.sqrt(x**2 + y**2)
    edge_m = ROUND_TRIP_FILL * ROUND_TRIP_GRID * ROUND_TRIP_PITCH_M / 2.0
    profile = (radius <= edge_m) if hard else np.exp(-((radius / edge_m) ** 8))
    return _field(profile.astype(np.complex64), (ROUND_TRIP_PITCH_M, ROUND_TRIP_PITCH_M))


def _relative_l2(measured: ScalarField, reference: ScalarField) -> float:
    a, b = np.asarray(measured.u), np.asarray(reference.u)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_forward_then_backward_returns_the_input() -> None:
    """Criterion 1 against a conservation law: the ASM transfer function is unitary.

    `H(-z) = conj(H(z))` on every propagating bin, so the product is exactly 1 and
    the reference is the input rather than another computation. Recorded by
    `B1-WAVE-FWDBWD-01`: 2.75199e-7 against a 1e-5 gate -- *below* the single-pass
    complex64 floor of 1.4e-4 (one float32 epsilon per radian of the 1181 rad the
    leg accumulates), because the two legs' phase errors are correlated.
    """
    source = _aperture(hard=False)
    forward = _asm(source, ROUND_TRIP_DISTANCE_M)
    returned = _asm(forward, -ROUND_TRIP_DISTANCE_M)
    assert _relative_l2(returned, source) < ROUND_TRIP_TOLERANCE


def test_the_round_trip_declares_what_it_cannot_see() -> None:
    """Two blind spots, both made to fire rather than left as prose.

    A hard aperture edge puts power past the light cone; those orders decay
    outward and cannot return, so the round trip is no longer the identity and the
    case is out of its own validity domain -- 2.2e-2 against a 1e-5 gate.

    And a convention error *shared* by both legs cancels exactly. The carrier
    phase is the sharp instance of that: propagating out with the carrier removed
    and back with it present leaves the input times `exp(-i k z)`, a piston that
    `|U|^2` cannot see at all. The residual is predicted from the removed carrier
    alone and matched to it, so what fires is the piston and not noise.
    """
    from solvers.chromatix import carrier_phase_rad

    hard = _aperture(hard=True)
    hard_returned = _asm(_asm(hard, ROUND_TRIP_DISTANCE_M), -ROUND_TRIP_DISTANCE_M)
    assert _relative_l2(hard_returned, hard) > 100 * ROUND_TRIP_TOLERANCE

    source = _aperture(hard=False)
    carrier_removed = propagate(
        source,
        distance_m=ROUND_TRIP_DISTANCE_M,
        model={"method": "asm_carrier_removed", "pad_width": 0, "target_surface": "target"},
    )
    mismatched = _asm(carrier_removed, -ROUND_TRIP_DISTANCE_M)

    carrier = carrier_phase_rad(
        wavelength_m=WAVELENGTH_M, distance_m=ROUND_TRIP_DISTANCE_M, refractive_index=1.0
    )
    predicted = abs(complex(np.exp(-1j * carrier)) - 1.0)
    assert predicted > 1000 * ROUND_TRIP_TOLERANCE
    assert _relative_l2(mismatched, source) == pytest.approx(predicted, rel=1e-3)
