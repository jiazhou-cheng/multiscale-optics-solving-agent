"""A declared illumination against geometry, and the factor of 2 pi that decides.

CHE-210 (R06.5) acceptance criteria 1, 2, 3, 5 and 6. Five cases:

==================================  ==========================================  ==========
case                                oracle                                      measured
==================================  ==========================================  ==========
tilt walks off                      ``z tan(theta)``, signed, in both axes       1.9e-5 rel
the axis ratio is the azimuth       ``walk_y / walk_x = tan(phi)``               3.1e-5 rel
rad/m vs cycles/m                   the same number is two angles, 2 pi apart    6.29x
normal incidence is flat            constant phase, exactly                     0.0 rad ptp
its advance is ``k z``              ``exp(+i k z)``                             1.1e-4 rad
grid-commensurate vs off-grid       ``(sin(pi D)/(N sin(pi D/N)))^2``            6 digits
Chromatix cross-check (diagnostic)  the backend's own ``plane_wave``            see below
==================================  ==========================================  ==========

Which oracle decides
--------------------
Straight-line geometry, the angular spectrum's own ``k_z``, and the Dirichlet
kernel of a finite DFT. The Chromatix comparison in section 5 is a **differential
check between two implementations of one formula**, so it is evidence and never
the gate -- criteria 1 to 3 are the closed forms, exactly as the ticket requires.

The illumination-angle unit, which is the whole point of section 2
-------------------------------------------------------------------
One package upstream ships two meanings for one argument name a factor of ``2 pi``
apart: ``kykx`` is angular wavenumber on ``plane_wave`` and spatial frequency in
cycles per length on ``asm_propagate``. A wrong-by-``2 pi`` illumination angle
produces a perfectly well-formed image at the wrong place in the pupil, and in
Fourier ptychography that is the one parameter the whole method sweeps over. So
section 2 does not assert that the two readings agree -- it asserts that they
**disagree measurably**, and names which one this project declares. Silent unit
agreement would be no evidence at all.

Cost: eight ASM propagations on a 768 x 1024 complex64 grid plus three small
source builds, all on the host: 4.3 s measured. Not marked `slow`; if that
changes, the grid is what to shrink.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backends.chromatix import propagate
from operators import complex_transmission
from representations import ReferenceSurface, ScalarField
from sources import plane_wave, transverse_wavevector_from_angle

WAVELENGTH_M = 0.532e-6
K = 2.0 * math.pi / WAVELENGTH_M
MEDIUM_INDEX = 1.0

#: 768 x 1024 at 0.55 x 0.45 um: asymmetric in **both** count and pitch, which is
#: what makes a transposed `(k_y, k_x)` detectable. The two walk-offs differ by
#: `tan(phi)` and the two pitches differ, so a swap changes the answer twice over.
SHAPE = (768, 1024)
PITCH_M = (0.55e-6, 0.45e-6)
DISTANCE_M = 200e-6
WAIST_M = 30e-6

#: 3 degrees at 30 degrees azimuth: both components non-zero, unequal, and neither
#: a special value. `phi = 30` rather than `45` so `k_y != k_x` and a transposition
#: is a 3x error rather than a no-op.
THETA_RAD = math.radians(3.0)
PHI_RAD = math.radians(30.0)

#: Inherited from `tests/physics/test_scalar_wave_propagation.py`, which took them
#: from the `B1-WAVE-*` families' own thresholds. Neither was chosen here.
WALKOFF_TOLERANCE = 2e-2
PHASE_TOLERANCE_RAD = 1e-2


def _surface(name: str = "illumination") -> ReferenceSurface:
    return ReferenceSurface(name=name, z_m=0.0, medium_index=MEDIUM_INDEX)


def _source(transverse_wavevector: tuple[float, float]) -> ScalarField:
    return plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=_surface(),
        transverse_wavevector_rad_per_m=transverse_wavevector,
    )


def _confined(field: ScalarField) -> ScalarField:
    """The source behind a Gaussian envelope, so it has a centroid to track.

    Built with the project's own thin-element operator (R06.6) rather than by hand:
    a plane wave of infinite extent has no measurable displacement, and the
    envelope is exactly `complex_transmission` with the phase factor at its
    identity. `w0 = 30 um` puts the Rayleigh range at 5.3 mm, twenty-six times the
    propagation distance, so diffractive spreading does not move the centroid.
    """
    y, x = (np.asarray(axis) for axis in field.coordinates())
    envelope = np.exp(-((y[:, None] ** 2 + x[None, :] ** 2) / WAIST_M**2))
    return complex_transmission(field, amplitude=envelope)


def _asm(field: ScalarField, distance_m: float) -> ScalarField:
    return propagate(
        field,
        distance_m=distance_m,
        model={"method": "asm", "pad_width": 0, "target_surface": "target"},
    )


def _centroid_m(field: ScalarField) -> tuple[float, float]:
    intensity = np.abs(np.asarray(field.u)) ** 2
    y, x = (np.asarray(axis) for axis in field.coordinates())
    total = intensity.sum()
    return (
        float((intensity.sum(axis=1) * y).sum() / total),
        float((intensity.sum(axis=0) * x).sum() / total),
    )


def _bin_offset(wavevector: float, *, count: int, pitch_m: float) -> float:
    """How far `wavevector` sits from the nearest DFT frequency bin, in bins."""
    spacing = 2.0 * math.pi / (count * pitch_m)
    bins = wavevector / spacing
    return abs(bins - round(bins))


def _sampling_headroom() -> None:
    """The two conditions every case here needs, asserted rather than assumed."""
    for count, pitch in zip(SHAPE, PITCH_M, strict=True):
        assert count * pitch**2 / WAVELENGTH_M > DISTANCE_M, (
            "the ASM transfer function aliases before this distance on this grid"
        )


# ---------------------------------------------------------------------------
# 1. The tilt is the tilt
# ---------------------------------------------------------------------------


def test_a_declared_tilt_walks_off_by_z_tan_theta_in_the_declared_direction() -> None:
    """Criterion 1. The magnitude, both signs, and both axes.

    `z tan(theta)` is exact geometry for a collimated beam. The 2e-2 tolerance is
    inherited and deliberately does **not** separate `z sin(theta)` from
    `z tan(theta)` -- they differ by 0.14% at 3 degrees -- which is why the axis
    *ratio* is asserted separately: `walk_y / walk_x = tan(phi)` is a pure
    geometric statement that a transposed `(k_y, k_x)` fails by a factor of three,
    not marginally.
    """
    _sampling_headroom()
    wavevector = transverse_wavevector_from_angle(
        THETA_RAD, PHI_RAD, wavelength_m=WAVELENGTH_M, medium_index=MEDIUM_INDEX
    )
    expected_m = DISTANCE_M * math.tan(THETA_RAD)

    measured: dict[float, tuple[float, float]] = {}
    for sign in (+1.0, -1.0):
        source = _confined(_source((sign * wavevector[0], sign * wavevector[1])))
        before = _centroid_m(source)
        after = _centroid_m(_asm(source, DISTANCE_M))
        measured[sign] = (after[0] - before[0], after[1] - before[1])

    for sign, (walk_y, walk_x) in measured.items():
        magnitude = math.hypot(walk_y, walk_x)
        assert abs(magnitude / expected_m - 1.0) < WALKOFF_TOLERANCE, (
            f"a {sign:+.0f}x tilt of {THETA_RAD} rad walked {magnitude} m over "
            f"{DISTANCE_M} m where z tan(theta) = {expected_m} m"
        )
        # The azimuth, as a ratio of the two axes. This is the transposition guard.
        assert walk_y / walk_x == pytest.approx(math.tan(PHI_RAD), rel=WALKOFF_TOLERANCE)
        # ...and each component has the sign the declared wavevector does.
        assert math.copysign(1.0, walk_y) == sign
        assert math.copysign(1.0, walk_x) == sign

    # A sign inversion is a factor-of-two relative error, not a marginal one.
    assert measured[+1.0][0] == pytest.approx(-measured[-1.0][0], rel=WALKOFF_TOLERANCE)
    assert measured[+1.0][1] == pytest.approx(-measured[-1.0][1], rel=WALKOFF_TOLERANCE)

    # This tilt is genuinely off the DFT frequency grid on both axes, which is
    # the second half of criterion 5's characterization: a beam confined well
    # inside the window pays no penalty for that. Recorded here rather than
    # asserted twice.
    assert _bin_offset(wavevector[0], count=SHAPE[0], pitch_m=PITCH_M[0]) > 0.1
    assert _bin_offset(wavevector[1], count=SHAPE[1], pitch_m=PITCH_M[1]) > 0.1


# ---------------------------------------------------------------------------
# 2. The unit is the declared unit
# ---------------------------------------------------------------------------


def test_the_wavevector_unit_is_rad_per_metre_and_not_cycles_per_metre() -> None:
    """Criterion 2. The factor of `2 pi`, asserted as a *difference*.

    One value, two readings. Passed as rad/m -- which is what this project
    declares -- it is a 2.598 degree tilt. The same number read as cycles/m and
    handed over unconverted is a 0.413 degree tilt: `2 pi` times smaller in
    `sin(theta)`, and the walk-off it produces is 6.29 times smaller.

    Both readings produce a perfectly well-formed field that walks off by exactly
    `z tan(theta)` for *its own* angle, which is the point: nothing about the
    result looks wrong. The only thing that distinguishes them is which unit was
    declared, so the test asserts the two are far apart and names which one is the
    project's.
    """
    _sampling_headroom()
    wavevector_rad_per_m = transverse_wavevector_from_angle(
        THETA_RAD, PHI_RAD, wavelength_m=WAVELENGTH_M, medium_index=MEDIUM_INDEX
    )[1]
    #: The same tilt expressed as a spatial frequency. `k_x = 2 pi f_x`, so a
    #: caller who has `f_x` and passes it straight through under-tilts by `2 pi`.
    as_cycles_per_m = wavevector_rad_per_m / (2.0 * math.pi)

    walkoff: dict[str, float] = {}
    for label, value in (
        ("rad_per_m", wavevector_rad_per_m),
        ("cycles_per_m_read_as_rad_per_m", as_cycles_per_m),
    ):
        source = _confined(_source((0.0, value)))
        before = _centroid_m(source)
        after = _centroid_m(_asm(source, DISTANCE_M))
        walkoff[label] = after[1] - before[1]

        # Each reading is internally consistent with the angle it implies, which is
        # exactly why the mistake is invisible without a declared unit.
        implied_theta = math.asin(value / (MEDIUM_INDEX * K))
        assert walkoff[label] == pytest.approx(
            DISTANCE_M * math.tan(implied_theta), rel=WALKOFF_TOLERANCE
        )

    ratio = walkoff["rad_per_m"] / walkoff["cycles_per_m_read_as_rad_per_m"]
    # tan() is very nearly linear at these angles, so the walk-off ratio is 2 pi to
    # better than 0.1%. The assertion is that the two are *far apart*: a project
    # that had accidentally agreed on the wrong unit would land at 1.0 here.
    assert ratio == pytest.approx(2.0 * math.pi, rel=1e-2)
    assert ratio > 6.0, "the two unit readings must be measurably different"


# ---------------------------------------------------------------------------
# 3. Normal incidence
# ---------------------------------------------------------------------------


def test_normal_incidence_is_flat_and_advances_by_k_z() -> None:
    """Criterion 3. Constant phase, and `exp(+i k z)` with `k_z = k` on axis.

    The flatness is asserted **exactly**, not to a tolerance: `exp(0j)` is
    `1 + 0j` with no rounding, so every sample of a zero-`k_t` source is the same
    complex number. A tolerance here would admit a half-sample origin error, which
    is a linear ramp -- i.e. a tilt -- of exactly the kind this source exists to
    state deliberately.

    The advance is the closed form R06.2 already gates, restated on a source this
    package built: 236 rad wraps, so the comparison is modulo `2 pi`, and a
    float64 prediction against a measurement good to 1.1e-4 rad cannot be a cycle
    apart.
    """
    _sampling_headroom()
    source = _source((0.0, 0.0))
    u = np.asarray(source.u)
    assert bool(np.all(u == u[0, 0])), "a zero-tilt source must be constant, exactly"
    assert float(np.ptp(np.angle(u))) == 0.0

    out = _asm(source, DISTANCE_M)
    ratio = np.asarray(out.u) / u
    measured = float(np.angle(np.mean(ratio)))
    # k_z = sqrt(k^2 - 0) = k exactly on axis.
    predicted = float(np.angle(np.exp(1j * K * DISTANCE_M)))
    assert abs(measured - predicted) < PHASE_TOLERANCE_RAD

    # The falsifiable twin: the conjugate phasor convention predicts the opposite
    # sign, and 0.76 rad is 76 times the tolerance apart.
    assert abs(measured + predicted) > 20.0 * PHASE_TOLERANCE_RAD


# ---------------------------------------------------------------------------
# 4. Grid-commensurate vs off-grid tilts
# ---------------------------------------------------------------------------

#: The bin the commensurate cases sit on. Non-zero, so a frequency grid scaled by
#: `2 pi` would not be invisible.
CARRIER_BIN = 12


def test_grid_commensurate_and_off_grid_tilts_differ_as_the_dirichlet_kernel_says() -> None:
    """Criterion 5, and the behaviour R06.8 will sweep through.

    A tilt sitting `D` bins from a DFT frequency puts

        (sin(pi D) / (N sin(pi D / N)))^2

    of its energy in the nearest bin -- the Dirichlet kernel of a finite,
    rectangularly windowed DFT, and 1 exactly when `D = 0`. At `D = 0.5` on 1024
    samples that is 0.40528, so **59% of an off-grid plane wave's energy is not in
    the bin its angle names.**

    The consequence, measured and recorded here rather than discovered in R06.8:
    a **window-filling** off-grid tilt is not propagated as a single plane wave at
    all. ASM decomposes it into every DFT bin, each of which advances with its own
    `k_z`, so the result bears no resemblance to `exp(i(k_t . r + k_z z))` --
    residual 1.4, against 2.5e-4 for the commensurate case. That is not a defect
    in ASM: a ramp that does not close on a periodic grid is discontinuous at the
    wrap, and the discontinuity is real content.

    A **confined** beam does not pay this. The walk-off case above carries a tilt
    0.78 and 0.26 bins off grid on the two axes and still measures `z tan(theta)`
    to 1.9e-5, because the Gaussian envelope suppresses the wrap. So the rule for
    R06.8 is: an angle sweep may leave the DFT grid freely as long as the field is
    windowed, and must not when the field fills the window.
    """
    _sampling_headroom()
    count, pitch = SHAPE[1], PITCH_M[1]
    spacing = 2.0 * math.pi / (count * pitch)

    concentration: dict[float, float] = {}
    plane_wave_residual: dict[float, float] = {}
    for offset in (0.0, 0.25, 0.5):
        wavevector = spacing * (CARRIER_BIN + offset)
        source = _source((0.0, wavevector))

        spectrum = np.abs(np.fft.fft(np.asarray(source.u)[0])) ** 2
        concentration[offset] = float(spectrum.max() / spectrum.sum())

        analytic = (
            1.0
            if offset == 0.0
            else (math.sin(math.pi * offset) / (count * math.sin(math.pi * offset / count))) ** 2
        )
        assert concentration[offset] == pytest.approx(analytic, rel=1e-5), (
            f"a tilt {offset} bins off grid concentrated {concentration[offset]} of its "
            f"energy in the peak bin where the Dirichlet kernel says {analytic}"
        )

        # ...and what that costs a propagation of a window-filling field.
        axial = math.sqrt(K * K - wavevector * wavevector)
        predicted = (np.asarray(source.u) * np.exp(1j * axial * DISTANCE_M)).astype(np.complex64)
        propagated = np.asarray(_asm(source, DISTANCE_M).u)
        plane_wave_residual[offset] = float(
            np.max(np.abs(propagated - predicted)) / np.max(np.abs(predicted))
        )

    assert concentration[0.0] == pytest.approx(1.0, rel=1e-6)
    assert concentration[0.5] == pytest.approx((2.0 / math.pi) ** 2, rel=1e-3)
    assert concentration[0.25] > concentration[0.5]

    assert plane_wave_residual[0.0] < 1e-3, (
        "a commensurate tilt is exactly one plane wave on this grid and must propagate as one"
    )
    assert plane_wave_residual[0.5] > 0.5, (
        "an off-grid, window-filling tilt is not a single plane wave on a periodic grid; "
        "if this ever passes, the characterization above is wrong and R06.8's angle sweep "
        "is resting on it"
    )


# ---------------------------------------------------------------------------
# 5. The Chromatix cross-check -- diagnostic, never a gate
# ---------------------------------------------------------------------------


def test_the_backend_agrees_on_the_angle_and_disagrees_on_the_amplitude() -> None:
    """Criterion 6. Two implementations of one formula, and the `power=` cost.

    **This is evidence and not a gate.** It is a differential check between this
    project's arithmetic and the backend's, so its agreement cannot certify either
    -- criteria 1 to 3 above are the closed forms that decide. What it *does* add
    is independent confirmation that `kykx` on the backend's `plane_wave` really is
    angular wavenumber, which is the fact CHE-57 measured and section 2 depends on.

    Two things are reported:

    * with `power=None` the two agree to the complex64 floor, at the same `n // 2`
      grid origin (verified: the backend's `grid` runs `-2, -1, 0, +1` for four
      samples, the same centring this project declares);
    * with the backend's `power=1.0` **default** the amplitude is renormalized to
      `sqrt(1/total)`, a factor this project must not inherit -- it would silently
      make every source's amplitude a function of its grid size.
    """
    import chromatix.functional as cf
    import jax

    jax.config.update("jax_enable_x64", False)

    # A small grid: this is a per-sample amplitude comparison, not a propagation.
    shape = (32, 48)
    pitch_m = (0.30e-6, 0.25e-6)
    wavevector = transverse_wavevector_from_angle(
        THETA_RAD, PHI_RAD, wavelength_m=WAVELENGTH_M, medium_index=MEDIUM_INDEX
    )

    ours = plane_wave(
        shape,
        sample_pitch_m=pitch_m,
        wavelength_m=WAVELENGTH_M,
        reference_surface=_surface(),
        transverse_wavevector_rad_per_m=wavevector,
    )
    unnormalized = np.asarray(
        cf.plane_wave(
            shape,
            [[pitch_m[0], pitch_m[1]]],
            WAVELENGTH_M,
            power=None,
            kykx=wavevector,
        ).u
    ).reshape(shape)
    normalized = np.asarray(
        cf.plane_wave(
            shape,
            [[pitch_m[0], pitch_m[1]]],
            WAVELENGTH_M,
            kykx=wavevector,
        ).u
    ).reshape(shape)

    mine = np.asarray(ours.u)
    agreement = float(np.max(np.abs(mine - unnormalized)))
    assert agreement < 1e-5, (
        f"the two implementations of A exp(i k_t . r) differ by {agreement}, which is "
        "above the complex64 floor and means one of them is not reading kykx as rad/m"
    )

    # The amplitude difference the `power=` convention introduces, reported rather
    # than absorbed. `power=1.0` divides by sqrt(sum |u|^2 dy dx) over the window,
    # so the peak amplitude becomes a function of the grid -- which is exactly the
    # kind of quiet normalization a project with no radiometric scale cannot carry.
    theirs_peak = float(np.max(np.abs(normalized)))
    ours_peak = float(np.max(np.abs(mine)))
    assert ours_peak == pytest.approx(1.0, rel=1e-6)
    assert theirs_peak == pytest.approx(
        1.0 / math.sqrt(shape[0] * shape[1] * pitch_m[0] * pitch_m[1]), rel=1e-3
    )
    assert theirs_peak / ours_peak > 1e4, (
        "the renormalization factor must be large enough here that inheriting it could "
        "not be mistaken for round-off"
    )
