"""A Gaussian beam at its waist as a contract: the radius convention, and two refusals.

CHE-215 (R06.10), item 2.

The assertion that matters most in this file is the one on `rho = w0`. `w0` is the
`1/e` **amplitude** radius, so the amplitude there is exactly `A/e`. The competing
reading `exp(-rho^2 / (2 w0^2))` differs by a factor of `sqrt(2)` in the waist and
produces an entirely plausible-looking beam of the wrong size, with no signature
anywhere downstream -- which is why the convention is measured here and not merely
documented in the module.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from representations import ContractError, Frame, ReferenceSurface, ScalarField
from sources import gaussian_beam, plane_wave, transverse_wavevector_from_angle

WAVELENGTH_M = 0.532e-6

#: Non-square in both count and pitch: a symmetric fixture cannot fail on a
#: transposed `(y, x)`, and `center_m` is a `(y, x)` pair.
SHAPE = (48, 64)
PITCH_M = (0.20e-6, 0.25e-6)
WAIST_M = 3.0e-6

MODULE = Path(__file__).resolve().parents[2] / "src" / "sources" / "gaussian_beam.py"


def a_surface(*, medium_index: float = 1.0) -> ReferenceSurface:
    return ReferenceSurface(name="waist", z_m=0.0, medium_index=medium_index)


def a_beam(**overrides: object) -> ScalarField:
    arguments: dict[str, object] = {
        "sample_pitch_m": PITCH_M,
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": a_surface(),
        "waist_radius_m": WAIST_M,
    }
    arguments.update(overrides)
    shape = arguments.pop("shape", SHAPE)
    return gaussian_beam(shape, **arguments)  # type: ignore[arg-type]


def coordinates(shape: tuple[int, int] = SHAPE, pitch: tuple[float, float] = PITCH_M):
    frame = Frame()
    y = (np.arange(shape[0], dtype=np.float64) - frame.origin_index(shape[0])) * pitch[0]
    x = (np.arange(shape[1], dtype=np.float64) - frame.origin_index(shape[1])) * pitch[1]
    return y, x


# ---------------------------------------------------------------------------
# 1. The radius convention, which is the whole ticket for this source
# ---------------------------------------------------------------------------


def test_the_amplitude_at_the_waist_radius_is_exactly_one_over_e() -> None:
    """`w0` is the `1/e` amplitude radius, hence the `1/e^2` intensity radius.

    Asserted on the *continuous* envelope rather than on a nearest sample, by
    choosing a waist that is an exact multiple of the pitch on both axes, so the
    sample at `rho = w0` exists and no interpolation enters. The competing
    `sqrt(2)` reading would give `exp(-1/2) = 0.607` here instead of `1/e = 0.368`
    -- a 65% error in amplitude that looks like a perfectly good beam.
    """
    pitch = (0.25e-6, 0.25e-6)
    waist = 4.0 * pitch[0]
    beam = a_beam(sample_pitch_m=pitch, waist_radius_m=waist, amplitude=3.0)
    u = np.asarray(beam.u)

    origin_y = Frame().origin_index(SHAPE[0])
    origin_x = Frame().origin_index(SHAPE[1])
    assert abs(u[origin_y, origin_x]) == pytest.approx(3.0, rel=1e-6)

    # Four samples out is exactly rho = w0, on both axes.
    for offset in ((4, 0), (0, 4), (-4, 0), (0, -4)):
        sample = u[origin_y + offset[0], origin_x + offset[1]]
        assert abs(sample) == pytest.approx(3.0 / math.e, rel=1e-6)

    # ...and the intensity there is 1/e^2, which is the catalogue convention.
    assert abs(u[origin_y + 4, origin_x]) ** 2 == pytest.approx(
        9.0 / math.e**2, rel=1e-5
    )


def test_the_envelope_is_the_analytic_gaussian_everywhere() -> None:
    """`exp(-((y - y0)^2 + (x - x0)^2) / w0^2)`, sample for sample.

    Against an independently written expression rather than against a golden
    array: the point is the *formula*, and a stored array would only pin whatever
    the implementation happened to produce.
    """
    beam = a_beam(amplitude=1.7)
    y, x = coordinates()
    expected = 1.7 * np.exp(-((y[:, None]) ** 2 + (x[None, :]) ** 2) / WAIST_M**2)

    assert np.allclose(np.abs(np.asarray(beam.u)), expected, rtol=1e-6, atol=0.0)


def test_a_centered_beam_is_symmetric_on_a_nonsquare_grid() -> None:
    """Symmetry about `Frame.origin_index`, not about `n // 2` rewritten.

    Even counts on both axes here, so the origin sample has a partner at
    `-offset` for every offset within range. The check is per axis on a grid whose
    two axes differ in both count and pitch, so a transposed `(dy, dx)` breaks it.
    """
    beam = a_beam()
    u = np.abs(np.asarray(beam.u))
    origin_y, origin_x = Frame().origin_index(SHAPE[0]), Frame().origin_index(SHAPE[1])

    for offset in (1, 5, 17):
        assert u[origin_y + offset, origin_x] == pytest.approx(
            u[origin_y - offset, origin_x], rel=1e-6
        )
        assert u[origin_y, origin_x + offset] == pytest.approx(
            u[origin_y, origin_x - offset], rel=1e-6
        )

    # The peak is at the frame's origin sample, and the axes are not interchangeable.
    assert np.unravel_index(int(np.argmax(u)), u.shape) == (origin_y, origin_x)
    assert u[origin_y + 4, origin_x] != pytest.approx(u[origin_y, origin_x + 4], rel=1e-3)


def test_the_center_is_a_half_sample_aware_translation() -> None:
    """`center_m` is `(y0, x0)` in metres and moves the envelope by exactly that.

    Including a **half-sample** shift, which is the case a `n // 2` origin cannot
    express and the one that separates a translation from a resample: the peak
    lands between two samples, which must then be equal.
    """
    whole = a_beam(center_m=(4.0 * PITCH_M[0], -3.0 * PITCH_M[1]))
    u = np.abs(np.asarray(whole.u))
    origin_y, origin_x = Frame().origin_index(SHAPE[0]), Frame().origin_index(SHAPE[1])
    assert np.unravel_index(int(np.argmax(u)), u.shape) == (origin_y + 4, origin_x - 3)

    half = np.abs(np.asarray(a_beam(center_m=(0.5 * PITCH_M[0], 0.0)).u))
    assert half[origin_y, origin_x] == pytest.approx(half[origin_y + 1, origin_x], rel=1e-6)

    # The envelope moves; the carrier does not. A translated beam is not a tilt.
    k_t = transverse_wavevector_from_angle(
        0.2, 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    centered = a_beam(transverse_wavevector_rad_per_m=k_t)
    shifted = a_beam(transverse_wavevector_rad_per_m=k_t, center_m=(0.0, 4.0 * PITCH_M[1]))
    ratio = np.asarray(shifted.u) / np.asarray(centered.u)
    assert np.allclose(np.angle(ratio), 0.0, atol=1e-5)


def test_the_amplitude_is_the_peak_and_scales_linearly() -> None:
    """A peak amplitude, unnormalized. No `power=` renormalization is inherited."""
    unit = a_beam()
    doubled = a_beam(amplitude=2.0)
    assert np.allclose(np.asarray(doubled.u), 2.0 * np.asarray(unit.u), rtol=1e-6)
    assert doubled.discrete_power() == pytest.approx(4.0 * unit.discrete_power(), rel=1e-5)


# ---------------------------------------------------------------------------
# 2. The carrier is shared with `plane_wave`, not reimplemented
# ---------------------------------------------------------------------------


def test_the_tilt_phase_agrees_with_plane_wave_sample_for_sample() -> None:
    """The carrier is `plane_wave`'s ramp exactly, once the envelope is divided out.

    A *shared-code consistency* check between two sources in this repository, not
    an independent physical oracle -- both call `sources._grid` for the coordinate
    axes and both write `exp(i(k_y y + k_x x))`. It earns its place anyway because
    what it guards is that the two agree at all: the analytic envelope is asserted
    separately above, so a divergence here localizes to the carrier.
    """
    k_t = transverse_wavevector_from_angle(
        0.35, 0.7, wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    beam = a_beam(transverse_wavevector_rad_per_m=k_t)
    illumination = plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        transverse_wavevector_rad_per_m=k_t,
    )

    y, x = coordinates()
    envelope = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / WAIST_M**2)
    carrier = np.asarray(beam.u) / envelope

    assert np.allclose(carrier, np.asarray(illumination.u), rtol=0.0, atol=1e-5)

    # And a waist far wider than the grid *is* a plane wave, to the cast's own
    # round-off -- the strongest available statement that nothing else differs.
    wide = a_beam(waist_radius_m=1.0, transverse_wavevector_rad_per_m=k_t)
    assert np.allclose(np.asarray(wide.u), np.asarray(illumination.u), atol=2e-5)


def test_the_field_is_complex64_with_a_float64_envelope_and_ramp() -> None:
    """`complex64` storage, float64 accumulation, `validity=frozenset()`.

    `complex128` is not a storage choice the wave path has -- `numerics.negotiate`
    refuses it against the measured chromatix row -- and the float64 accumulation
    is what keeps the single cast honest. The residual against a float64 reference
    is asserted to sit at the *cast's* round-off (~1e-7 relative) rather than at
    float32 arithmetic's (~1e-5 over a 1e3 rad exponent).
    """
    k_t = transverse_wavevector_from_angle(
        0.4, 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    beam = a_beam(shape=(256, 256), transverse_wavevector_rad_per_m=k_t)
    assert np.asarray(beam.u).dtype == np.complex64
    assert beam.validity == frozenset()

    y, x = coordinates((256, 256), PITCH_M)
    reference = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / WAIST_M**2) * np.exp(
        1j * (k_t[0] * y[:, None] + k_t[1] * x[None, :])
    )
    assert np.max(np.abs(np.asarray(beam.u) - reference)) < 1e-6


def test_an_analytic_waist_declares_no_limitation() -> None:
    """`validity == frozenset()`, and no `z` argument exists to invalidate it.

    At the waist the field is a real envelope times the carrier: no curvature, no
    Gouy phase, no `w(z)`. That makes the empty set a *claim* of exactness rather
    than an absence of one. An off-waist Gaussian is paraxial and `ValidityFlag`
    has no token for it, so the function refuses to take a `z` at all rather than
    taking one and mis-declaring itself -- see the module docstring.
    """
    assert a_beam().validity == frozenset()

    with pytest.raises(TypeError):
        a_beam(z_m=1e-3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        a_beam(distance_m=1e-3)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. The refusals
# ---------------------------------------------------------------------------


def test_the_two_carrier_refusals_fire_and_are_the_shared_ones() -> None:
    """`|k_t| > n k0` and `|k_t| > pi/d`, from the same helper `plane_wave` calls.

    Asserted by *comparing the two sources*, which is the only check that catches
    the failure this ticket named as its main risk: one source accepting a geometry
    the other refuses. The messages and codes have to match because there is one
    implementation, so a divergence means someone copied it.
    """
    k0 = 2.0 * math.pi / WAVELENGTH_M
    fine_pitch = (0.05e-6, 0.05e-6)
    evanescent = (0.0, 1.05 * k0)

    with pytest.raises(ContractError) as beam_error:
        a_beam(sample_pitch_m=fine_pitch, transverse_wavevector_rad_per_m=evanescent)
    with pytest.raises(ContractError) as wave_error:
        plane_wave(
            SHAPE,
            sample_pitch_m=fine_pitch,
            wavelength_m=WAVELENGTH_M,
            reference_surface=a_surface(),
            transverse_wavevector_rad_per_m=evanescent,
        )
    assert beam_error.value.code == wave_error.value.code == "REPRESENTATION_INCONSISTENT"
    assert str(beam_error.value) == str(wave_error.value)
    assert "evanescent" in str(beam_error.value)

    # The same k_t is a legal tilt in glass, refused against n k0 and not k0.
    legal = a_beam(
        sample_pitch_m=fine_pitch,
        reference_surface=a_surface(medium_index=1.515),
        transverse_wavevector_rad_per_m=evanescent,
    )
    assert legal.shape == SHAPE

    coarse_pitch = (0.30e-6, 0.28e-6)
    nyquist_y = math.pi / coarse_pitch[0]
    assert nyquist_y < k0, "the grid limit must be the binding one here"
    with pytest.raises(ContractError) as excinfo:
        a_beam(
            sample_pitch_m=coarse_pitch,
            transverse_wavevector_rad_per_m=(1.01 * nyquist_y, 0.0),
        )
    assert "alias" in str(excinfo.value) and "k_y" in str(excinfo.value)


def test_an_unresolved_waist_is_refused_per_axis() -> None:
    """Fewer than two samples across `w0` is a discretization artifact, not a beam.

    Refused the same way an aliased tilt is, and for the same reason: what comes
    back is a plausible beam of whatever size the grid can carry. Per axis, so a
    coarse `y` refuses a waist the same value in `x` would accept.
    """
    with pytest.raises(ContractError) as excinfo:
        a_beam(sample_pitch_m=(1.0e-6, 0.1e-6), waist_radius_m=1.5e-6)
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "dy" in str(excinfo.value) and "waist" in str(excinfo.value)

    with pytest.raises(ContractError) as excinfo:
        a_beam(sample_pitch_m=(0.1e-6, 1.0e-6), waist_radius_m=1.5e-6)
    assert "dx" in str(excinfo.value)

    # Exactly two samples across the waist is the boundary and is permitted.
    marginal = a_beam(sample_pitch_m=(0.5e-6, 0.5e-6), waist_radius_m=1.0e-6)
    assert marginal.shape == SHAPE


def test_a_truncated_grid_is_documented_and_not_refused() -> None:
    """Edge truncation is a modelling choice a caller is allowed to make.

    A grid whose half-extent is only ~0.8 `w0` clips the envelope at ~0.5 of the
    peak amplitude. That rings when it is propagated, which is a real consequence
    -- and it is the caller's to own, unlike an *unresolved* waist, which is not a
    beam at all. The truncated fraction is stated in the module docstring.
    """
    small = a_beam(shape=(16, 16), sample_pitch_m=(0.3e-6, 0.3e-6), waist_radius_m=3.0e-6)
    u = np.abs(np.asarray(small.u))
    assert u.max() == pytest.approx(1.0, rel=1e-6)
    assert u[0, 0] > 0.2, "this fixture is meant to be truncated, not merely wide"


def test_the_scalar_field_contract_refuses_the_rest() -> None:
    """Pitch, wavelength, amplitude and waist through the shared helpers."""
    for bad_pitch in ((0.0, PITCH_M[1]), (-1e-6, PITCH_M[1]), (math.nan, PITCH_M[1])):
        with pytest.raises(ContractError):
            a_beam(sample_pitch_m=bad_pitch)
    for bad in (0.0, -WAVELENGTH_M, math.nan):
        with pytest.raises(ContractError):
            a_beam(wavelength_m=bad)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ContractError):
            a_beam(waist_radius_m=bad)
        with pytest.raises(ContractError):
            a_beam(amplitude=bad)
    for bad_shape in ((0, 8), (-1, 8), (8,), (8, 8, 8)):
        with pytest.raises(ValueError):
            a_beam(shape=bad_shape)
    for bad_center in ((math.nan, 0.0), (0.0, math.inf), (0.0,)):
        with pytest.raises(ValueError):
            a_beam(center_m=bad_center)


def test_the_module_defines_no_class_and_imports_no_backend() -> None:
    """`BUDGETS["sources"] == 0`, and `chromatix.functional.gaussian_beam` is a
    cross-check rather than the implementation -- note its `power=` renormalization
    before comparing anything to it."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    assert [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)] == []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "numpy", "representations", "sources"}
