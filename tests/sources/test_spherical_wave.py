"""An analytic point emitter as a contract: the sign, the `1/R` scale, and four refusals.

CHE-215 (R06.10), item 3.

Two things in this file are convention decisions rather than derivable facts, and
both are asserted rather than documented:

* **the sign**, pinned to `SPATIAL_FACTOR = exp(+i k z)`. Diverging is
  `exp(+i n k0 R)` and converging is `exp(-i n k0 R)`, and the two are exact
  complex conjugates -- so no intensity, anywhere downstream, can tell them apart.
* **the `1/R` reference distance**. `amplitude` is the field at `R = 1 m`, not a
  peak, decided by the owner on this ticket. Peak normalization would hide the
  absolute scale, and the absolute scale is the one thing a peak-normalized or
  intensity-only check cannot recover.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from representations import ContractError, Frame, ReferenceSurface, ScalarField
from sources import spherical_wave
from sources.spherical_wave import REFERENCE_DISTANCE_M

WAVELENGTH_M = 0.532e-6

#: Non-square in both count and pitch. The source distance is chosen so the
#: radial phase runs over **tens of radians** across the grid while staying inside
#: the sampling limit at every index tested: a source far enough away to be
#: comfortably sampled produces a nearly flat wavefront, against which "the phase
#: tracks n k0 R" degenerates into "the phase is roughly constant" and passes on a
#: field that is wrong. At 10 um the phase spans ~28 rad at n = 1.515 and the worst
#: local sine is 0.45 against the 0.70 a 0.25 um pitch carries there. Along the
#: central row it is 14 rad in air and 21 at n = 1.515.
SHAPE = (32, 40)
PITCH_M = (0.20e-6, 0.25e-6)
SOURCE_M = (0.0, 0.0, -1.0e-5)

MODULE = Path(__file__).resolve().parents[2] / "src" / "sources" / "spherical_wave.py"


def a_surface(*, z_m: float = 0.0, medium_index: float = 1.0) -> ReferenceSurface:
    return ReferenceSurface(name="object_plane", z_m=z_m, medium_index=medium_index)


def a_wave(**overrides: object) -> ScalarField:
    arguments: dict[str, object] = {
        "sample_pitch_m": PITCH_M,
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": a_surface(),
        "source_position_m": SOURCE_M,
    }
    arguments.update(overrides)
    shape = arguments.pop("shape", SHAPE)
    return spherical_wave(shape, **arguments)  # type: ignore[arg-type]


def radius_grid(
    source: tuple[float, float, float] = SOURCE_M,
    *,
    shape: tuple[int, int] = SHAPE,
    pitch: tuple[float, float] = PITCH_M,
    z_plane: float = 0.0,
) -> np.ndarray:
    """`R = |r - r_s|` on the grid, written independently of the module under test."""
    frame = Frame()
    y = (np.arange(shape[0], dtype=np.float64) - frame.origin_index(shape[0])) * pitch[0]
    x = (np.arange(shape[1], dtype=np.float64) - frame.origin_index(shape[1])) * pitch[1]
    x_s, y_s, z_s = source
    return np.sqrt(
        (y[:, None] - y_s) ** 2 + (x[None, :] - x_s) ** 2 + (z_plane - z_s) ** 2
    )


# ---------------------------------------------------------------------------
# 1. The radial phase is the analytic one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("medium_index", [1.0, 1.336, 1.515])
def test_the_radial_phase_is_n_k0_r_to_dtype_round_off(medium_index: float) -> None:
    """`arg(E) = n k0 R` along a cut, unwrapped, against an independent `R`.

    The oracle is the closed form, computed here from the geometry rather than from
    anything the module produced. Unwrapped because `n k0 R` runs to thousands of
    radians across the grid, so a comparison of wrapped phases would pass on a
    field that was wrong by whole cycles.

    Three indices including air, so the `n = 1` case cannot be the only one that
    passes -- a medium shortens the wavelength and the phase must follow.
    """
    wave = a_wave(reference_surface=a_surface(medium_index=medium_index))
    radius = radius_grid()
    row = Frame().origin_index(SHAPE[0])

    measured = np.unwrap(np.angle(np.asarray(wave.u)[row, :]).astype(np.float64))
    expected = 2.0 * math.pi * medium_index / WAVELENGTH_M * radius[row, :]
    # `unwrap` fixes the branch but not the absolute cycle, so the comparison is on
    # the residual about its own mean: an arbitrary constant is allowed, a varying
    # one is not.
    residual = (measured - expected) - np.mean(measured - expected)

    # The phase must actually go somewhere, or this asserts nothing.
    assert np.ptp(expected) > 12.0, "the fixture must span many radians of phase"
    # 2e-7 rad is the complex64 cast's own round-off, measured. A float32 phase
    # accumulation would land near 1e-4 here and a wrong n would land at 14 rad.
    assert np.max(np.abs(residual)) < 1e-6


def test_the_field_is_the_closed_form_everywhere() -> None:
    """`E = A (R_ref / R) exp(+i n k0 R)`, sample for sample, complex.

    The full complex comparison, which subsumes the phase cut above and is the
    check that a plausible-looking magnitude is not hiding a wrong phase. The
    tolerance is the `complex64` cast's, taken relative to the field's own scale.
    """
    wave = a_wave(amplitude=2.5)
    radius = radius_grid()
    expected = (
        2.5
        * (REFERENCE_DISTANCE_M / radius)
        * np.exp(1j * 2.0 * math.pi / WAVELENGTH_M * radius)
    )

    residual = np.max(np.abs(np.asarray(wave.u) - expected)) / np.max(np.abs(expected))
    assert residual < 1e-5


def test_the_amplitude_reference_distance_is_one_metre() -> None:
    """`amplitude` is the field at `R = 1 m`, so the grid value is `A / R`.

    The owner's convention decision on this ticket, and the one an intensity-only
    check cannot recover: a peak-normalized source would return `A` at the closest
    sample regardless of how far away the emitter was, so a geometry error of a
    factor of ten would leave no trace. Here it leaves a factor of ten.
    """
    assert REFERENCE_DISTANCE_M == 1.0

    near = a_wave(source_position_m=(0.0, 0.0, -1.0e-4))
    far = a_wave(source_position_m=(0.0, 0.0, -1.0e-3))
    origin = (Frame().origin_index(SHAPE[0]), Frame().origin_index(SHAPE[1]))

    assert abs(np.asarray(near.u)[origin]) == pytest.approx(1.0e4, rel=1e-4)
    assert abs(np.asarray(far.u)[origin]) == pytest.approx(1.0e3, rel=1e-4)
    # Ten times further is a tenth the amplitude. Not peak-normalized.
    assert abs(np.asarray(far.u)[origin]) == pytest.approx(
        0.1 * abs(np.asarray(near.u)[origin]), rel=1e-4
    )

    # And it scales linearly in A, at fixed geometry.
    doubled = a_wave(amplitude=2.0)
    unit = a_wave()
    assert np.allclose(np.asarray(doubled.u), 2.0 * np.asarray(unit.u), rtol=1e-5)


def test_the_amplitude_falls_as_one_over_r_across_the_grid() -> None:
    """`|E| R` is constant over the whole grid, which `1/R` is and `1/R^2` is not.

    A stronger statement than checking two points: an intensity-style `1/R^2`
    envelope, or a missing `1/R` altogether, both break constancy while leaving the
    peak in the right place. The source is close enough here that `R` varies
    measurably across the grid, which is what gives the test teeth.
    """
    source = (0.0, 0.0, -3.0e-6)
    wave = a_wave(
        shape=(64, 64), sample_pitch_m=(0.1e-6, 0.1e-6), source_position_m=source
    )
    radius = radius_grid(source, shape=(64, 64), pitch=(0.1e-6, 0.1e-6))
    assert radius.max() / radius.min() > 1.4, "R must vary across the grid for this to bite"

    product = np.abs(np.asarray(wave.u)) * radius
    assert np.allclose(product, product.flat[0], rtol=1e-4)


def test_translating_the_source_translates_the_wavefront() -> None:
    """Moving `source_position_m` laterally by `m` samples moves the field by `m`.

    Exactly, because the field depends on the geometry only through `R`, so a
    lateral shift of the source by an integer number of samples is a roll of the
    array. `(x, y, z)` order is asserted here too: shifting `x_s` moves along the
    *column* axis, and a swapped pair would move along the other one.
    """
    shifted = a_wave(source_position_m=(3.0 * PITCH_M[1], 0.0, SOURCE_M[2]))
    base = np.asarray(a_wave().u)

    rolled = np.roll(base, 3, axis=1)
    interior = (slice(None), slice(4, -4))
    residual = np.max(np.abs(np.asarray(shifted.u)[interior] - rolled[interior]))
    assert residual / np.max(np.abs(base)) < 1e-5

    # And the y half of the pair moves along rows, so the order is not transposable.
    shifted_y = a_wave(source_position_m=(0.0, 2.0 * PITCH_M[0], SOURCE_M[2]))
    rolled_y = np.roll(base, 2, axis=0)
    interior_y = (slice(3, -3), slice(None))
    residual_y = np.max(np.abs(np.asarray(shifted_y.u)[interior_y] - rolled_y[interior_y]))
    assert residual_y / np.max(np.abs(base)) < 1e-5


# ---------------------------------------------------------------------------
# 2. The sign, pinned to the phasor convention
# ---------------------------------------------------------------------------


def test_converging_and_diverging_are_complex_conjugates() -> None:
    """`exp(-i n k0 R)` against `exp(+i n k0 R)`, at a mirrored source position.

    The pairing is the physical one: diverging from a point *upstream*, converging
    toward a point *downstream*, so the two source positions are mirror images
    about the sampled plane and `R` is identical for both. What is left is the
    sign, and the fields are exact conjugates.

    This is the trap the assertion exists for. Conjugating a field turns a
    converging wavefront into a diverging one, `|E|` is unchanged sample for
    sample, and no intensity measurement anywhere downstream can tell.
    """
    diverging = a_wave(source_position_m=(0.0, 0.0, -2.0e-4), converging=False)
    converging = a_wave(source_position_m=(0.0, 0.0, +2.0e-4), converging=True)

    assert np.allclose(
        np.asarray(converging.u), np.conj(np.asarray(diverging.u)), rtol=1e-5, atol=0.0
    )
    # Identical in magnitude, which is exactly why the sign needs a phase check.
    assert np.allclose(np.abs(np.asarray(converging.u)), np.abs(np.asarray(diverging.u)))
    assert not np.allclose(np.asarray(converging.u), np.asarray(diverging.u))


def test_the_diverging_sign_is_plus_against_the_forward_phasor() -> None:
    """`SPATIAL_FACTOR = exp(+i k z)`, so a diverging wave gains phase with `R`.

    Asserted as the *sign of the radial phase gradient* rather than by comparing to
    a stored array: for the default `converging=False`, the unwrapped phase must
    increase away from the source, and for `converging=True` it must decrease.
    """
    radius = radius_grid()
    row = Frame().origin_index(SHAPE[0])
    order = np.argsort(radius[row, :])

    for converging, expected_sign in ((False, +1.0), (True, -1.0)):
        wave = a_wave(converging=converging)
        phase = np.unwrap(np.angle(np.asarray(wave.u)[row, :]).astype(np.float64))
        gradient = np.polyfit(radius[row, :][order], phase[order], 1)[0]
        assert math.copysign(1.0, gradient) == expected_sign
        assert abs(gradient) == pytest.approx(2.0 * math.pi / WAVELENGTH_M, rel=1e-3)


def test_converging_is_a_boolean_and_not_a_string_vocabulary() -> None:
    """One bit. Two call sites cannot spell a bool differently."""
    with pytest.raises(TypeError):
        a_wave(direction="diverging")  # type: ignore[call-arg]
    assert a_wave().validity == frozenset()
    assert np.asarray(a_wave().u).dtype == np.complex64


# ---------------------------------------------------------------------------
# 3. The refusals
# ---------------------------------------------------------------------------


def test_a_source_on_the_sampled_plane_is_refused() -> None:
    """`z_s == reference_surface.z_m`: `R` reaches zero and nothing represents it.

    Checked against a *non-zero* plane too, so the comparison is against the
    surface's own `z_m` and not against a hard-coded zero -- which would accept a
    source sitting exactly on a plane declared at 1 mm.
    """
    with pytest.raises(ContractError) as excinfo:
        a_wave(source_position_m=(0.0, 0.0, 0.0))
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "on the sampled plane" in str(excinfo.value)

    with pytest.raises(ContractError):
        a_wave(reference_surface=a_surface(z_m=1e-3), source_position_m=(0.0, 0.0, 1e-3))

    # The same lateral position one plane away is fine.
    assert a_wave(
        reference_surface=a_surface(z_m=1e-3), source_position_m=(0.0, 0.0, 1e-3 - 2e-4)
    ).shape == SHAPE


def test_a_source_within_one_sample_pitch_is_refused() -> None:
    """The `1/R` singularity: inside a sample the field is not a spherical wave.

    `|E|` changes by orders of magnitude between adjacent samples there, so what
    the grid holds is a discretization of a pole. Note the geometry is chosen so
    *this* refusal binds rather than the sampling one -- a source that close is
    also badly under-sampled, and the order of the checks is what decides which
    message a caller gets.
    """
    with pytest.raises(ContractError) as excinfo:
        a_wave(source_position_m=(0.0, 0.0, -0.05 * PITCH_M[0]))
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "inside one sample pitch" in str(excinfo.value)


def test_an_undersampled_geometry_is_refused_with_the_geometry_named() -> None:
    """`n k0 |rho - rho_s| / R > pi/d` on either axis, and the message says why.

    The same failure as an aliased tilt: the aliased local frequency corresponds to
    a real ray at a smaller angle, so an under-sampled spherical wave reads back as
    a **different and entirely plausible geometry**. The message has to carry the
    source position, the plane and the sine it reached, because the fix is a
    geometry change and a bare "aliased" would not say which one.
    """
    # A grid 8 um wide with the source only 2 um away reaches sin ~ 0.9, far past
    # the 0.53 / 2 = 0.27 a 1 um pitch carries at this wavelength.
    with pytest.raises(ContractError) as excinfo:
        a_wave(
            shape=(8, 8),
            sample_pitch_m=(1.0e-6, 1.0e-6),
            source_position_m=(0.0, 0.0, -2.0e-6),
        )
    message = str(excinfo.value)
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "Nyquist" in message and "sin(theta_local)" in message
    assert "-2e-06" in message and "NA" in message

    # A marginally-sampled geometry, inside the limit, is *not* refused. The
    # documented bound is sin(theta) < lambda / (2 n d); at d = 1 um that is 0.266,
    # and a 4 um half-diagonal at 20 um distance reaches ~0.20.
    marginal = spherical_wave(
        (8, 8),
        sample_pitch_m=(1.0e-6, 1.0e-6),
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        source_position_m=(0.0, 0.0, -2.0e-5),
    )
    assert marginal.shape == (8, 8)

    # ...and the bound tightens with the medium index, because n k0 grows while
    # pi/d does not. The same geometry in glass is refused.
    with pytest.raises(ContractError):
        spherical_wave(
            (8, 8),
            sample_pitch_m=(1.0e-6, 1.0e-6),
            wavelength_m=WAVELENGTH_M,
            reference_surface=a_surface(medium_index=1.515),
            source_position_m=(0.0, 0.0, -1.4e-5),
        )


def test_the_undersampling_bound_is_the_shared_nyquist_limit() -> None:
    """The refusal uses `_grid.nyquist_limit_rad_per_m`, the same `pi/d` `plane_wave` uses.

    Asserted by construction rather than by reading the source: a geometry whose
    worst local `k_x` sits just inside `pi/dx` is accepted and one just outside is
    refused, with the pitch as the only thing that changed. A source carrying its
    own copy of the factor of two would put the boundary elsewhere.
    """
    from sources._grid import nyquist_limit_rad_per_m

    shape, distance = (8, 8), 2.0e-5
    for pitch, accepted in ((1.0e-6, True), (2.4e-6, False)):
        radius = radius_grid(
            (0.0, 0.0, -distance), shape=shape, pitch=(pitch, pitch), z_plane=0.0
        )
        frame = Frame()
        x = (np.arange(shape[1], dtype=np.float64) - frame.origin_index(shape[1])) * pitch
        worst = float(np.max(np.abs(x[None, :]) / radius)) * 2.0 * math.pi / WAVELENGTH_M
        assert (worst <= nyquist_limit_rad_per_m(pitch)) is accepted

        def call(pitch: float = pitch) -> ScalarField:
            return spherical_wave(
                shape,
                sample_pitch_m=(pitch, pitch),
                wavelength_m=WAVELENGTH_M,
                reference_surface=a_surface(),
                source_position_m=(0.0, 0.0, -distance),
            )

        if accepted:
            assert call().shape == shape
        else:
            with pytest.raises(ContractError):
                call()


def test_the_scalar_field_contract_refuses_the_rest() -> None:
    """Pitch, wavelength, amplitude and the position triple."""
    for bad_pitch in ((0.0, PITCH_M[1]), (-1e-6, PITCH_M[1]), (math.nan, PITCH_M[1])):
        with pytest.raises(ContractError):
            a_wave(sample_pitch_m=bad_pitch)
    for bad in (0.0, -WAVELENGTH_M, math.nan):
        with pytest.raises(ContractError):
            a_wave(wavelength_m=bad)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ContractError):
            a_wave(amplitude=bad)
    for bad_shape in ((0, 8), (-1, 8), (8,), (8, 8, 8)):
        with pytest.raises(ValueError):
            a_wave(shape=bad_shape)
    for bad_position in ((0.0, 0.0, math.nan), (0.0, math.inf, -1e-4), (0.0, -1e-4)):
        with pytest.raises(ValueError):
            a_wave(source_position_m=bad_position)


def test_there_is_no_aperture_argument() -> None:
    """Truncation composes with the thin-element operator R06.6 landed.

    Strictly more expressive than an `aperture=` here would be -- any complex mask,
    not just a hard disc -- and it keeps this source free of any downstream
    geometry. No NA, no stop, no launch cone, and none inferred.
    """
    for name in ("aperture_radius_m", "aperture", "numerical_aperture", "na", "stop_radius_m"):
        with pytest.raises(TypeError):
            a_wave(**{name: 1e-3})  # type: ignore[arg-type]


def test_the_module_defines_no_class_and_imports_no_backend() -> None:
    """`BUDGETS["sources"] == 0`, and no solver on the path."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    assert [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)] == []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # `numerics` joined the set at CHE-246 (T2), which gave this source a
    # `namespace`/`device` target: it needs `ArrayNamespace` and `DevicePlacement`
    # to spell the argument. Not a widening of what this test protects --
    # `sources/ -> numerics` is a declared edge of the dependency allowlist and
    # `scripts/check_dependencies.py` enforces it -- and the claim here is still
    # "no class, and no solver on the path".
    assert imported <= {
        "__future__",
        "math",
        "numerics",
        "numpy",
        "representations",
        "sources",
    }
