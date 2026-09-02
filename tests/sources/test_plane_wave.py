"""The declared illumination as a contract: every convention, and four refusals.

CHE-210 (R06.5) acceptance criteria 4, 7 and the declaration half of the rest.
`tests/physics/test_coherent_sources.py` holds the physics -- the walk-off, the
factor of `2 pi`, the flat normal-incidence phase, the grid-commensurate
characterization and the Chromatix cross-check -- because those need a
propagation and these need nothing.

What this file is really guarding is that **nothing is defaulted**. A source is
the only operation in the tree that creates a representation out of nothing, so
every convention it does not take as an argument is one it invented.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from operations import CATALOG, OperationKind, resolve
from representations import PHASOR, ContractError, Frame, ReferenceSurface, ScalarField
from sources import plane_wave, transverse_wavevector_from_angle

WAVELENGTH_M = 0.532e-6
SHAPE = (48, 64)
PITCH_M = (0.30e-6, 0.25e-6)

MODULE = Path(__file__).resolve().parents[2] / "src" / "sources" / "plane_wave.py"


def a_surface(*, medium_index: float = 1.0) -> ReferenceSurface:
    return ReferenceSurface(name="illumination", z_m=0.0, medium_index=medium_index)


def a_source(**overrides: object) -> ScalarField:
    arguments: dict[str, object] = {
        "sample_pitch_m": PITCH_M,
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": a_surface(),
    }
    arguments.update(overrides)
    shape = arguments.pop("shape", SHAPE)
    return plane_wave(shape, **arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Every convention is declared
# ---------------------------------------------------------------------------


def test_the_source_declares_the_grid_the_caller_gave_it() -> None:
    """Sampling is caller-supplied, with no default and no inference.

    A lopsided grid in both count and pitch, so a transposed `(dy, dx)` would give
    a different extent and cannot pass.
    """
    field = plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
    )
    assert field.shape == SHAPE
    assert field.sample_pitch_m == PITCH_M
    assert field.extent_m == (SHAPE[0] * PITCH_M[0], SHAPE[1] * PITCH_M[1])
    assert field.wavelength_m == WAVELENGTH_M
    assert field.phasor == PHASOR
    assert field.frame == Frame()
    assert (field.pad_width, field.padded) == (0, False)
    assert field.reference_surface == a_surface()

    # There is no signature-level default for the two declarations that matter.
    import inspect

    parameters = inspect.signature(plane_wave).parameters
    for required in ("sample_pitch_m", "wavelength_m", "reference_surface"):
        assert parameters[required].default is inspect.Parameter.empty


def test_the_origin_is_the_frames_own_rule() -> None:
    """`Frame.origin_index`, not a rewritten `n // 2`.

    A half-sample origin shift is a linear phase ramp across the grid -- a tilt --
    which is the one quantity this function exists to state, so the rule is
    delegated rather than restated. Checked on an **odd** axis too, where
    `n // 2` and `(n - 1) / 2` coincide but a `n // 2 - 1` off-by-one would not.
    """
    frame = Frame()
    for shape in ((48, 64), (33, 31)):
        field = plane_wave(
            shape,
            sample_pitch_m=PITCH_M,
            wavelength_m=WAVELENGTH_M,
            reference_surface=a_surface(),
        )
        y, x = (np.asarray(axis) for axis in field.coordinates())
        assert y[frame.origin_index(shape[0])] == 0.0
        assert x[frame.origin_index(shape[1])] == 0.0


def test_the_field_is_complex64_with_a_float64_phase_ramp() -> None:
    """The dtype is the wave path's one storage dtype, and the ramp is not float32.

    Both halves are load-bearing. `complex128` is refused by
    `numerics.negotiate` against the measured chromatix row, so a float64 source
    could not be propagated at all. And accumulating the ramp in float64 before
    the single cast is what keeps the cast honest: `k` is 1.18e7 rad/m, this grid
    puts ~9 rad through the exponent at the corner, and the measured residual
    against a float64 reference is at the complex64 storage floor rather than at
    the float32 *arithmetic* floor.
    """
    kt = transverse_wavevector_from_angle(
        math.radians(2.0), math.radians(20.0), wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    field = plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        transverse_wavevector_rad_per_m=kt,
    )
    assert str(field.u.dtype) == "complex64"

    y, x = (np.asarray(axis).astype(np.float64) for axis in field.coordinates())
    reference = np.exp(1j * (kt[0] * y[:, None] + kt[1] * x[None, :]))
    assert np.max(np.abs(np.asarray(field.u) - reference)) < 1e-6


def test_an_analytic_plane_wave_declares_no_limitation() -> None:
    """`validity=frozenset()` is a claim, and the empty set is how it is made."""
    assert a_source().validity == frozenset()


def test_the_amplitude_is_the_peak_and_nothing_renormalizes_it() -> None:
    """Chromatix's `power=1.0` convention is not inherited.

    `discrete_power` scales as `amplitude^2` and with the window, exactly as
    `sum |u|^2 dy dx` says it should, because there is no radiometric
    normalization anywhere in this tree to make it anything else.
    """
    unit = a_source()
    doubled = a_source(amplitude=2.0)
    assert np.max(np.abs(np.asarray(unit.u))) == pytest.approx(1.0, rel=1e-7)
    assert np.max(np.abs(np.asarray(doubled.u))) == pytest.approx(2.0, rel=1e-7)

    dy, dx = PITCH_M
    assert unit.discrete_power() == pytest.approx(SHAPE[0] * SHAPE[1] * dy * dx, rel=1e-6)
    assert doubled.discrete_power() == pytest.approx(4.0 * unit.discrete_power(), rel=1e-6)


def test_normal_incidence_is_the_same_primitive() -> None:
    """`k_t = (0, 0)` is the default and there is no second function for it.

    Narrowed by CHE-215 (R06.10), which added three more sources: this used to pin
    `sources.__all__` to exactly two names, which asserted the *package's* whole
    surface in a file about one function and failed the moment a second source
    landed for legitimate reasons. The invariant it was really guarding is that
    `plane_wave` has no per-incidence variant, which is what is asserted now --
    `sources/__init__.py` and `tests/sources/test_sources_package.py` own the
    package surface.

    `point_source` stays on the absent list and `gaussian_beam` comes off it. Both
    are deliberate: R06.10 lifted the Gaussian exclusion on the owner's decision,
    while a wave-optics point emitter remains `spherical_wave` with explicit
    geometry rather than a delta on a pixel.
    """
    import sources

    for name in ("normal_plane_wave", "tilted_plane_wave", "point_source"):
        assert not hasattr(sources, name)

    explicit = a_source(transverse_wavevector_rad_per_m=(0.0, 0.0))
    assert np.array_equal(np.asarray(a_source().u), np.asarray(explicit.u))


# ---------------------------------------------------------------------------
# 2. The angle converter
# ---------------------------------------------------------------------------


def test_the_angle_converter_is_the_analytic_relation() -> None:
    """`|k_t| = (2 pi n / lambda) sin(theta)`, `phi` from `+x` toward `+y`.

    The azimuth convention is asserted at the two axes rather than only through
    the magnitude, because a `(k_y, k_x)` returned in the other order has the same
    magnitude and a completely plausible tilt.
    """
    theta = math.radians(12.0)
    for index in (1.0, 1.515):
        magnitude = (2.0 * math.pi * index / WAVELENGTH_M) * math.sin(theta)

        along_x = transverse_wavevector_from_angle(
            theta, 0.0, wavelength_m=WAVELENGTH_M, medium_index=index
        )
        assert along_x[0] == pytest.approx(0.0, abs=1e-9 * magnitude)
        assert along_x[1] == pytest.approx(magnitude, rel=1e-12)

        along_y = transverse_wavevector_from_angle(
            theta, 0.5 * math.pi, wavelength_m=WAVELENGTH_M, medium_index=index
        )
        assert along_y[0] == pytest.approx(magnitude, rel=1e-12)
        assert along_y[1] == pytest.approx(0.0, abs=1e-9 * magnitude)

        diagonal = transverse_wavevector_from_angle(
            theta, 0.25 * math.pi, wavelength_m=WAVELENGTH_M, medium_index=index
        )
        assert math.hypot(*diagonal) == pytest.approx(magnitude, rel=1e-12)
        assert diagonal[0] == pytest.approx(diagonal[1], rel=1e-12)

    # Zero polar angle is normal incidence, for any azimuth.
    assert transverse_wavevector_from_angle(
        0.0, 1.234, wavelength_m=WAVELENGTH_M, medium_index=1.0
    ) == (0.0, 0.0)
    # The medium index is in the magnitude, not decoration: the same angle in glass
    # is a larger transverse wavevector.
    in_air = transverse_wavevector_from_angle(
        theta, 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    in_glass = transverse_wavevector_from_angle(
        theta, 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.515
    )
    assert in_glass[1] == pytest.approx(1.515 * in_air[1], rel=1e-12)


def test_the_angle_converter_refuses_an_ambiguous_or_unusable_argument() -> None:
    """Past `pi/2`, `sin(theta)` decreases again, so a backward-going direction
    would come back as a small forward tilt -- a plausible answer for a wave
    travelling the other way."""
    with pytest.raises(ValueError, match="beyond pi/2"):
        transverse_wavevector_from_angle(
            math.radians(170.0), 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.0
        )
    with pytest.raises(ValueError, match="finite"):
        transverse_wavevector_from_angle(
            math.nan, 0.0, wavelength_m=WAVELENGTH_M, medium_index=1.0
        )
    with pytest.raises(ValueError, match="wavelength_m"):
        transverse_wavevector_from_angle(0.1, 0.0, wavelength_m=0.0, medium_index=1.0)
    with pytest.raises(ValueError, match="medium_index"):
        transverse_wavevector_from_angle(0.1, 0.0, wavelength_m=WAVELENGTH_M, medium_index=-1.0)


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------


def test_an_evanescent_transverse_wavevector_is_refused() -> None:
    """Criterion 4, first refusal. `|k_t| > n k0` is not an illumination angle.

    The field it would build decays along `+z` and would be carried as a
    propagating one -- and the refusal is against `n k0`, not `k0`, so the same
    `k_t` that is evanescent in air is a legal 41-degree tilt in glass.
    """
    k0 = 2.0 * math.pi / WAVELENGTH_M
    fine_pitch = (0.05e-6, 0.05e-6)  # Nyquist is 6.3e7 rad/m, far above k0.

    with pytest.raises(ContractError) as excinfo:
        a_source(
            sample_pitch_m=fine_pitch,
            transverse_wavevector_rad_per_m=(0.0, 1.05 * k0),
        )
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "evanescent" in str(excinfo.value)

    # ...and in a medium of index 1.515 the same value is inside the light cone.
    legal = a_source(
        sample_pitch_m=fine_pitch,
        reference_surface=a_surface(medium_index=1.515),
        transverse_wavevector_rad_per_m=(0.0, 1.05 * k0),
    )
    assert legal.shape == SHAPE

    # Grazing incidence, |k_t| = n k0 exactly, is the boundary and is permitted.
    grazing = a_source(sample_pitch_m=fine_pitch, transverse_wavevector_rad_per_m=(0.0, k0))
    assert grazing.shape == SHAPE


def test_a_transverse_wavevector_past_the_grids_nyquist_is_refused() -> None:
    """Criterion 4, second refusal, and the one with teeth.

    An aliased tilt reads back as a completely different and entirely plausible
    angle, so the check is per axis against `pi / d` -- the two axes have different
    pitches here, and a coarse `y` refuses a `k_y` that the same magnitude in `x`
    would accept.

    **Which of the two refusals binds is a property of the pitch**, and the case
    is chosen to exercise this one rather than the light cone. `pi/d` exceeds
    `n k0` whenever `d < lambda / 2`, so on a grid finer than half a wavelength
    every representable tilt is inside the light cone and the *evanescent* refusal
    is the only one that can fire; on a coarser grid the Nyquist limit binds first.
    This pitch pair is coarser than `lambda/2 = 0.266 um` on both axes for exactly
    that reason.
    """
    coarse_pitch = (0.30e-6, 0.28e-6)
    nyquist_y = math.pi / coarse_pitch[0]
    nyquist_x = math.pi / coarse_pitch[1]
    light_cone = 2.0 * math.pi / WAVELENGTH_M
    assert nyquist_y < nyquist_x < light_cone, "the grid limit must be the binding one here"

    with pytest.raises(ContractError) as excinfo:
        a_source(
            sample_pitch_m=coarse_pitch,
            transverse_wavevector_rad_per_m=(1.01 * nyquist_y, 0.0),
        )
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "alias" in str(excinfo.value) and "k_y" in str(excinfo.value)

    # The same magnitude on the finer axis is legal, which is what makes this a
    # grid refusal and not a second light-cone refusal.
    legal = a_source(
        sample_pitch_m=coarse_pitch, transverse_wavevector_rad_per_m=(0.0, 1.01 * nyquist_y)
    )
    assert legal.shape == SHAPE

    with pytest.raises(ContractError) as excinfo:
        a_source(
            sample_pitch_m=coarse_pitch,
            transverse_wavevector_rad_per_m=(0.0, 1.01 * nyquist_x),
        )
    assert "k_x" in str(excinfo.value)


def test_the_scalar_field_contract_refuses_the_rest() -> None:
    """Criterion 4, third refusal: the pitch, the wavelength and the amplitude go
    through the helpers `ScalarField` itself uses, not through a second copy."""
    for bad_pitch in ((0.0, PITCH_M[1]), (-1e-6, PITCH_M[1]), (math.nan, PITCH_M[1])):
        with pytest.raises(ContractError) as excinfo:
            a_source(sample_pitch_m=bad_pitch)
        assert excinfo.value.code == "UNIT_NOT_SI"
        assert "sample_pitch_m" in str(excinfo.value)

    for bad_wavelength in (0.0, -1e-6, math.inf):
        with pytest.raises(ContractError) as excinfo:
            a_source(wavelength_m=bad_wavelength)
        assert excinfo.value.code == "UNIT_NOT_SI"

    for bad_amplitude in (0.0, -1.0, math.nan):
        with pytest.raises(ContractError) as excinfo:
            a_source(amplitude=bad_amplitude)
        assert excinfo.value.code == "UNIT_NOT_SI"

    for bad_shape in ((0, 64), (48, -1), (48,)):
        with pytest.raises(ValueError, match="shape"):
            a_source(shape=bad_shape)

    with pytest.raises(ValueError, match="finite"):
        a_source(transverse_wavevector_rad_per_m=(math.nan, 0.0))


# ---------------------------------------------------------------------------
# 4. Registration and class delta
# ---------------------------------------------------------------------------


def test_the_source_registers_as_a_source() -> None:
    """Criterion 7. `solver`-kind, because that is what the definition says.

    A source maps a problem statement into a representation, which is
    `docs/architecture_principles.md` section 2's definition of a solver. It has
    no external backend, which is *why* `src/sources/` exists as a package, and
    not a reason to call it something else.

`inputs=()`, because a source consumes no upstream representation. That is
    CHE-222 (R03.5): this record used to declare `input='scalar_field'` -- the
    representation it *produces*, named on both sides, following the precedent
    R05.3 set for the ray solver -- which contradicted the signature, this
    package's docstring and `docs/architecture_principles.md` §2 at once. What the
    function actually needs is in `requires`, and none of it is a representation.

    `capabilities=None` is the honest citation. There is no measured device/dtype
    row for this operation because it imports no backend; citing the chromatix row
    would claim a measurement that was taken about something else.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `sources/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.
    """
    descriptor = next(d for d in CATALOG if d.operation_id == "S_SOURCE_PLANE_WAVE")
    # `SOURCE` since CHE-224 (R15.1). It was `SOLVER` only because the enum had no
    # `SOURCE` member -- which is what made the `S_` prefix ambiguous, since
    # `SO_RAY_LAUNCH_TRACE`'s `S_` stood for solver and this one's for source.
    assert descriptor.kind is OperationKind.SOURCE
    assert descriptor.backend is None, "the project's own arithmetic drives no library"
    assert descriptor.implementation == "sources.plane_wave:plane_wave"
    assert descriptor.derivative == "forward_only"
    assert descriptor.derivative_evidence is None
    assert descriptor.capabilities is None
    assert resolve("S_SOURCE_PLANE_WAVE") is plane_wave


def test_the_other_two_sources_have_records_too() -> None:
    """CHE-221: `gaussian_beam` and `spherical_wave` had no descriptor at all.

    Three sources, three records, three ids -- not one `S_SOURCE` with a mode
    argument. They differ in what they approximate and in what they refuse: the
    Gaussian is exact only at its waist, the spherical wave is refused when its
    local phase gradient outruns the grid, and the plane wave has neither
    condition. That is metadata a caller reads before choosing, so it belongs on
    separate records.
    """
    catalogued = {d.operation_id: d for d in CATALOG}
    for operation_id, implementation in (
        ("S_SOURCE_GAUSSIAN_BEAM", "sources.gaussian_beam:gaussian_beam"),
        ("S_SOURCE_SPHERICAL_WAVE", "sources.spherical_wave:spherical_wave"),
    ):
        descriptor = catalogued[operation_id]
        assert descriptor.kind is OperationKind.SOURCE
        assert descriptor.backend is None, "no source in sources/ drives a library"
        assert descriptor.implementation == implementation
        assert descriptor.capabilities is None, "no source imports a backend"
        assert descriptor.derivative == "forward_only"
        assert descriptor.validity, "each states its own applicability"
    assert len({d.operation_id for d in CATALOG if d.implementation.startswith("sources.")}) == 3


def test_the_module_defines_no_class() -> None:
    """Criterion: 0 production classes.

    An `Illumination` frozen dataclass is the one that has a real argument --
    lambda, `k_t` and the medium index are coupled by `|k_t| <= n k0`, which is
    minimality rule 1. It did not land because the coupling is only *half* the
    validation: the grid's Nyquist limit is the second refusal and it is not
    knowable from a declaration that has no grid. A declaration object would
    therefore validate less than this function does, while adding a type.
    """
    source = MODULE.read_text(encoding="utf-8")
    assert [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)] == []
