"""The thin element as a contract: one implementation, three cases, six refusals.

CHE-211 (R06.6) acceptance criteria 1, 3 (the mask half), 4, 5, 6, 7 and 8.
`tests/physics/test_thin_element_spectrum.py` holds the closed-form spectra --
the Bessel orders of a sinusoidal phase grating, the known weights of a binary
one, and the NA-to-cutoff tie -- because those need the Fourier plane and this
file needs no backend at all.

The organizing claim is criterion 1: an amplitude object, a phase object and a
pupil are the *same operator* with one factor at its identity. Every assertion
below that compares a special case against a directly-constructed reference is
there to make that structural, not stylistic.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from numerics import CHROMATIX_CAPABILITIES
from operations import OperationDescriptor, OperationKind, registry, resolve
from operators import (
    EDGES,
    circular_aperture_amplitude,
    complex_transmission,
    numerical_aperture_radius_m,
)
from representations import ContractError, ReferenceSurface, ScalarField

WAVELENGTH_M = 0.532e-6
SHAPE = (48, 64)
PITCH_M = (0.30e-6, 0.25e-6)

MODULE = Path(__file__).resolve().parents[2] / "src" / "operators" / "transmission.py"


def a_field(**overrides: object) -> ScalarField:
    """A deliberately lopsided field: different counts *and* different pitches.

    A square grid at equal pitch cannot detect a transposed mask, and the mask
    builder takes a shape and a pitch rather than a field, so a transposition is
    exactly the mistake available here.
    """
    rng = np.random.default_rng(20260901)
    u = (rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)).astype(np.complex64)
    fields: dict[str, object] = {
        "u": u,
        "sample_pitch_m": PITCH_M,
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": ReferenceSurface(name="object", z_m=0.0, medium_index=1.0),
    }
    fields.update(overrides)
    return ScalarField(**fields)  # type: ignore[arg-type]


def a_mask(*, radius_m: float = 6e-6, edge: str = "hard") -> np.ndarray:
    return circular_aperture_amplitude(
        SHAPE, sample_pitch_m=PITCH_M, radius_m=radius_m, edge=edge
    )


# ---------------------------------------------------------------------------
# 1. The special cases are the general case
# ---------------------------------------------------------------------------


def test_a_pure_phase_element_is_the_general_operator_with_amplitude_one() -> None:
    """Criterion 1, half one. `amplitude=1` against a directly built reference."""
    field = a_field()
    ramp = np.linspace(-2.0, 2.0, SHAPE[1], dtype=np.float64)
    phase = np.broadcast_to(ramp[None, :], SHAPE).copy()

    out = complex_transmission(field, phase_rad=phase)

    reference = (np.asarray(field.u) * np.exp(1j * phase).astype(np.complex64)).astype(
        np.complex64
    )
    assert np.asarray(out.u) == pytest.approx(reference, rel=1e-6, abs=1e-7)
    # ...and the default is the identity, exactly.
    assert np.array_equal(np.asarray(complex_transmission(field).u), np.asarray(field.u))


def test_a_pure_amplitude_element_is_the_general_operator_with_phase_zero() -> None:
    """Criterion 1, half two. `phase_rad=0` is bit-exact, not merely close.

    `exp(0j)` is `1 + 0j` with no rounding, and a complex multiply by `1 + 0j`
    computes `a*1 - b*0` per component, so an amplitude-only element must
    reproduce `u * A` sample for sample. If this ever needs a tolerance, the
    operator has grown arithmetic it does not need.
    """
    field = a_field()
    mask = a_mask()

    out = complex_transmission(field, amplitude=mask)

    reference = np.asarray(field.u) * mask.astype(np.complex64)
    assert np.array_equal(np.asarray(out.u), reference)


def test_both_factors_together_are_the_same_one_implementation() -> None:
    """Criterion 1, the composition: `A exp(i phi)` equals applying each in turn.

    This is the assertion that makes "one implementation" a property rather than
    a claim about the source: if the general case were a separate code path from
    the two special cases, this is where it would disagree.
    """
    field = a_field()
    mask = a_mask()
    phase = np.full(SHAPE, 0.7)

    together = complex_transmission(field, amplitude=mask, phase_rad=phase)
    in_turn = complex_transmission(
        complex_transmission(field, amplitude=mask), phase_rad=phase
    )
    assert np.asarray(together.u) == pytest.approx(np.asarray(in_turn.u), rel=1e-6, abs=1e-7)


def test_there_is_no_operator_per_element_type() -> None:
    """Criterion 1 and the ticket's named architectural risk.

    The pressure is to add `phase_mask`, `amplitude_mask`, `pupil` and `grating`
    as public operations, each of which looks cheap. A budget records what exists
    and cannot record what was avoided, so the absence is asserted.
    """
    import operators

    for name in ("phase_mask", "amplitude_mask", "pupil", "grating", "thin_element"):
        assert not hasattr(operators, name), (
            f"operators.{name} exists; it is complex_transmission with one factor at its "
            "identity, and shipping it separately means keeping the two consistent forever"
        )
    # `propagate_rays` (CHE-192, R09.2) is the package's second operator and is not
    # an element type: it changes a ray bundle's *state* between two surfaces and
    # touches no transmission. Named here rather than filtered out, so a third
    # operator has to be argued into this list too.
    assert set(operators.__all__) == {
        "EDGES",
        "circular_aperture_amplitude",
        "complex_transmission",
        "numerical_aperture_radius_m",
        "propagate_rays",
    }


# ---------------------------------------------------------------------------
# 2. The mask, and the surface the element acts at
# ---------------------------------------------------------------------------


def test_a_hard_aperture_transmits_exactly_the_samples_inside_the_radius() -> None:
    """Criterion 3, first half. Sample-exact, on the `n // 2` origin.

    The reference grid is built here from the origin rule rather than from the
    builder, so a builder that centred on `(n-1)/2` -- a half-sample shift, i.e. a
    tilt -- would fail.
    """
    radius_m = 6e-6
    y = (np.arange(SHAPE[0]) - SHAPE[0] // 2) * PITCH_M[0]
    x = (np.arange(SHAPE[1]) - SHAPE[1] // 2) * PITCH_M[1]
    inside = np.hypot(y[:, None], x[None, :]) <= radius_m
    assert inside.any() and not inside.all(), "the mask must actually cut something"

    field = a_field()
    out = complex_transmission(field, amplitude=a_mask(radius_m=radius_m))

    assert np.array_equal(np.asarray(out.u)[inside], np.asarray(field.u)[inside])
    assert np.all(np.asarray(out.u)[~inside] == 0)


def test_the_soft_edge_is_a_different_mask_and_the_choice_has_no_default() -> None:
    """Criterion: the edge is a declared parameter.

    Both edges are legitimate physics -- the module docstring carries the 2.75e-7
    vs 2.2e-2 round-trip measurement -- so the two must be measurably different
    and neither may be reachable without saying which.
    """
    hard = a_mask(edge="hard")
    soft = a_mask(edge="soft_r8")
    assert not np.allclose(hard, soft)
    assert soft[SHAPE[0] // 2, SHAPE[1] // 2] == pytest.approx(1.0)
    # exp(-1) at r = R, which is what makes it a *soft* edge rather than a step.
    assert 0.0 < soft[~(hard > 0)].max() < 1.0

    with pytest.raises(TypeError):
        circular_aperture_amplitude(  # type: ignore[call-arg]
            SHAPE, sample_pitch_m=PITCH_M, radius_m=6e-6
        )
    with pytest.raises(ValueError, match="no default"):
        circular_aperture_amplitude(
            SHAPE, sample_pitch_m=PITCH_M, radius_m=6e-6, edge="gaussian"
        )
    assert EDGES == ("hard", "soft_r8")


def test_the_element_does_not_move_the_field() -> None:
    """The approximation, as a property of the returned artifact.

    An infinitely thin element acts at the field's own reference surface: `z_m`,
    the medium index, the pitch and the pad state all come back unchanged, and
    `target_surface` renames the plane without moving it.
    """
    field = a_field(pad_width=3, padded=True)
    out = complex_transmission(field, amplitude=0.5, target_surface="pupil")

    assert out.reference_surface.name == "pupil"
    assert out.reference_surface.z_m == field.reference_surface.z_m
    assert out.reference_surface.medium_index == field.reference_surface.medium_index
    assert out.reference_surface.normal == field.reference_surface.normal
    assert out.sample_pitch_m == field.sample_pitch_m
    assert (out.pad_width, out.padded) == (3, True)
    assert out.frame == field.frame
    # ...and with no target_surface the name is inherited, not defaulted to one.
    assert complex_transmission(field).reference_surface.name == "object"


def test_numerical_aperture_radius_is_the_analytic_stop_radius() -> None:
    """Criterion 3, second half. `R = f NA / n`, in float64, wavelength-free."""
    assert numerical_aperture_radius_m(0.25, focal_length_m=20e-3, medium_index=1.0) == 5e-3
    assert numerical_aperture_radius_m(
        0.6, focal_length_m=8e-3, medium_index=1.33
    ) == pytest.approx(8e-3 * 0.6 / 1.33, rel=1e-15)

    # The index is in the denominator and it is the error the ticket names: the
    # same NA in water gives a stop 1.33x smaller, a 33% radius error that looks
    # like a slightly tighter aperture rather than like a bug.
    assert numerical_aperture_radius_m(
        0.6, focal_length_m=8e-3, medium_index=1.33
    ) != pytest.approx(numerical_aperture_radius_m(0.6, focal_length_m=8e-3, medium_index=1.0))

    with pytest.raises(ValueError, match="exceeds medium_index"):
        numerical_aperture_radius_m(1.4, focal_length_m=8e-3, medium_index=1.0)
    for bad in (0.0, -0.2, math.nan):
        with pytest.raises(ValueError):
            numerical_aperture_radius_m(bad, focal_length_m=8e-3, medium_index=1.0)


# ---------------------------------------------------------------------------
# 3. Energy accounting
# ---------------------------------------------------------------------------


def test_the_power_after_the_element_is_the_sum_of_the_masked_intensity() -> None:
    """Criterion 4, first half: exactly, because an elementwise multiply has no
    numerical excuse. The oracle is `sum |u A|^2 dy dx`, formed here."""
    field = a_field()
    mask = a_mask()
    out = complex_transmission(field, amplitude=mask)

    dy, dx = PITCH_M
    oracle = float(
        np.sum(np.abs(np.asarray(field.u) * mask.astype(np.complex64)) ** 2) * dy * dx
    )
    assert out.discrete_power() == oracle


def test_a_pure_phase_element_conserves_power() -> None:
    """Criterion 4, second half. A phase-only element that loses power is a bug,
    and this is the cheapest place it shows up."""
    field = a_field()
    rng = np.random.default_rng(7)
    phase = rng.uniform(-np.pi, np.pi, SHAPE)

    out = complex_transmission(field, phase_rad=phase)

    # float32 storage: ~1.2e-7 relative per sample, summed in float32 by
    # `discrete_power`, so a few epsilons is the floor and 1e-6 is not fitted.
    assert out.discrete_power() == pytest.approx(field.discrete_power(), rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Backend neutrality, asserted rather than assumed
# ---------------------------------------------------------------------------


def test_the_operator_returns_the_namespace_it_was_given() -> None:
    """Criterion 5. A NumPy field and a JAX field, same operator, same answer.

    The AST and `sys.modules` halves of this criterion live in
    `tests/solvers/test_chromatix_boundary.py`, which walks every module under
    `src/` and now names `operators` in its fresh-interpreter probe. This half is
    about the object a caller receives.
    """
    import jax.numpy as jnp

    field = a_field()
    mask = a_mask()
    phase = np.full(SHAPE, 0.4)

    host = complex_transmission(field, amplitude=mask, phase_rad=phase)
    device = complex_transmission(
        a_field(u=jnp.asarray(np.asarray(field.u))), amplitude=mask, phase_rad=phase
    )

    assert isinstance(host.u, np.ndarray)
    assert type(device.u).__module__.startswith(("jax", "jaxlib"))
    assert str(device.u.dtype) == "complex64", "the field's dtype is not promoted"
    assert np.asarray(device.u) == pytest.approx(np.asarray(host.u), rel=1e-6, abs=1e-7)


# ---------------------------------------------------------------------------
# 5. Refusals
# ---------------------------------------------------------------------------


def test_a_broadcastable_but_wrong_mask_is_refused() -> None:
    """The silent failure this refusal exists for: a `(1, nx)` row broadcasts
    across the field and applies the aperture along one axis only."""
    field = a_field()
    for wrong in (np.ones((1, SHAPE[1])), np.ones((SHAPE[0], 1)), np.ones((SHAPE[1],))):
        with pytest.raises(ContractError) as excinfo:
            complex_transmission(field, amplitude=wrong)
        assert excinfo.value.code == "SHAPE_MISMATCH"
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, phase_rad=np.ones((*SHAPE, 2)))
    assert excinfo.value.code == "SHAPE_MISMATCH"


def test_a_negative_or_non_finite_amplitude_is_refused() -> None:
    """`A` is a real non-negative modulus. A negative entry is a pi phase written
    in the wrong field, and it passes every intensity check downstream."""
    field = a_field()
    negative = a_mask()
    negative[0, 0] = -1.0
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, amplitude=negative)
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "phase_rad" in str(excinfo.value)

    for bad in (np.nan, np.inf):
        with pytest.raises(ContractError) as excinfo:
            complex_transmission(field, amplitude=bad)
        assert excinfo.value.code == "NON_FINITE"
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, phase_rad=np.nan)
    assert excinfo.value.code == "NON_FINITE"


def test_gain_is_refused_unless_it_is_claimed() -> None:
    """A passive thin element cannot amplify, and the opt-in makes gain a stated
    claim rather than an arithmetic accident."""
    field = a_field()
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, amplitude=1.0001)
    assert excinfo.value.code == "REPRESENTATION_INCONSISTENT"
    assert "allow_gain" in str(excinfo.value)

    amplified = complex_transmission(field, amplitude=2.0, allow_gain=True)
    assert amplified.discrete_power() == pytest.approx(4.0 * field.discrete_power(), rel=1e-6)


def test_a_complex_amplitude_is_refused_and_names_phase_rad() -> None:
    """Accepting it would let one physical quantity be specified two ways and
    disagree -- `abs()` in `amplitude` and a different `angle()` in `phase_rad`."""
    field = a_field()
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, amplitude=np.full(SHAPE, 0.5 + 0.5j))
    assert excinfo.value.code == "DTYPE_KIND_MISMATCH"
    assert "phase_rad" in str(excinfo.value)

    with pytest.raises(ContractError) as excinfo:
        complex_transmission(field, phase_rad=0.5j)
    assert excinfo.value.code == "DTYPE_KIND_MISMATCH"


def test_an_empty_target_surface_is_refused() -> None:
    with pytest.raises(ContractError) as excinfo:
        complex_transmission(a_field(), target_surface="  ")
    assert excinfo.value.code == "MISSING_DECLARATION"


# ---------------------------------------------------------------------------
# 6. Validity: inherited unchanged, and `surface_only` is permitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        frozenset(),
        frozenset({"carrier_removed_phase"}),
        frozenset({"no_wavefront_curvature_term", "carrier_removed_phase"}),
    ],
)
def test_validity_is_inherited_unchanged(flags: frozenset[str]) -> None:
    """Multiplying by a mask neither adds nor removes a declared limitation."""
    field = a_field(validity=flags)
    assert complex_transmission(field, amplitude=a_mask()).validity == flags


def test_a_surface_only_field_is_not_refused_here() -> None:
    """The one place the flag's meaning is subtle, so it is stated and tested.

    `surface_only` means the field is valid at its declared reference surface and
    nowhere else. `propagate` and `focal_plane_transform` refuse it because they
    move the field off that surface. A thin element acts *at* the surface and does
    not move it at all, which is the one operation the flag permits -- so refusing
    it here would be wrong, not conservative.
    """
    field = a_field(validity=frozenset({"surface_only"}))
    out = complex_transmission(field, amplitude=a_mask())
    assert out.validity == frozenset({"surface_only"})
    assert out.reference_surface.z_m == field.reference_surface.z_m


# ---------------------------------------------------------------------------
# 7. Registration and class delta
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_registry() -> Iterator[None]:
    """The registry is module-level state, so the isolation belongs in the test."""
    saved = dict(registry._REGISTERED)
    registry._REGISTERED.clear()
    yield
    registry._REGISTERED.clear()
    registry._REGISTERED.update(saved)


def test_the_element_registers_as_a_physical_operator(isolated_registry: None) -> None:
    """Criterion 7. A physical operator, never a coupler.

    The representation on both sides is a `ScalarField` at the same surface and
    nothing is re-described: the *state* changes. A coupler would be the
    opposite, which is the call `docs/architecture_principles.md` section 2 makes
    and the one the retired `C_FIELD_TO_PSF` got wrong in the other direction.

    The descriptor lives here rather than in production for the reason R05.3,
    R06.2 and R06.4 all recorded: `operators/` may not import `operations/` and
    `operations/` may not import `operators/`, so no production registration site
    exists yet. Widening the allowlist to make one fit is not this ticket's
    change -- and note that R06.5's `sources` row *is* an allowlist change, made
    deliberately with the owner, which is what the difference looks like.

    `capabilities=None` is the honest citation: this operator has no measured
    device/dtype row of its own because it runs in whatever namespace the field
    carries. `CHROMATIX_CAPABILITIES` is imported here only to assert that it is
    *not* cited.
    """
    descriptor = registry.register(
        OperationDescriptor(
            operation_id="O_COMPLEX_TRANSMISSION",
            kind=OperationKind.PHYSICAL_OPERATOR,
            input="scalar_field",
            output="scalar_field",
            implementation="operators.transmission:complex_transmission",
            approximation=(
                "an infinitely thin element acting at the field's own reference surface: "
                "U_out = U_in * A * exp(i phi), elementwise. z_m does not advance, no "
                "propagation happens inside it, and there is no thickness, no multiple "
                "scattering, no polarization and no angular dependence of the "
                "transmission -- which is also what makes tilt-as-spectral-shift exact "
                "rather than approximate for this element"
            ),
            validity=(
                "the transmission is sampled on the field's own grid, so a mask with "
                "structure finer than the pitch is aliased rather than resolved",
                "A is a real non-negative modulus bounded by 1 unless gain is claimed",
                "surface_only fields are permitted: the element acts exactly at the "
                "surface where such a field is valid",
            ),
            evidence=(
                "tests/operators/test_transmission.py",
                "tests/physics/test_thin_element_spectrum.py",
            ),
            capabilities=None,
            derivative="forward_only",
        )
    )

    assert descriptor.kind is OperationKind.PHYSICAL_OPERATOR
    assert descriptor.kind is not OperationKind.COUPLER
    assert descriptor.derivative == "forward_only"
    assert descriptor.capabilities is None
    assert descriptor.capabilities != CHROMATIX_CAPABILITIES.component
    assert resolve("O_COMPLEX_TRANSMISSION") is complex_transmission


def test_the_module_defines_no_class() -> None:
    """Criterion 8. Class delta 0: a mask is an array plus the grid it was built
    on, and the grid already lives on the `ScalarField`."""
    source = MODULE.read_text(encoding="utf-8")
    assert [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)] == []
