"""`ScalarField`: explicit pitch, typed validity, and the `n // 2` grid.

CHE-176 (R02.4). Construction refusals are enumerated in
`test_contract_codes.py`; this file is the four acceptance criteria that are
about behaviour rather than about a code -- that a pitch cannot be omitted, that
`validity` is something a consumer branches on, that `coordinates()` puts zero
where the origin rule says it goes, and that no second scalar type exists.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import representations
from numerics import DType
from representations import (
    VALIDITY_FLAGS,
    VALIDITY_NOTES,
    ContractError,
    Frame,
    ReferenceSurface,
    ScalarField,
)

SURFACE = ReferenceSurface(name="image_surface", z_m=0.0, medium_index=1.0)
WAVELENGTH_M = 550e-9


def _field(**overrides: Any) -> ScalarField:
    fields: dict[str, Any] = {
        "u": np.ones((4, 6), dtype=np.complex64),
        "sample_pitch_m": (1.25e-6, 2.5e-6),
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": SURFACE,
    }
    fields.update(overrides)
    return ScalarField(**fields)


# --- acceptance criterion 1: the pitch is explicit ---


def test_a_field_without_a_pitch_cannot_be_constructed() -> None:
    """No default, so this is a TypeError rather than a validation failure.

    A default pitch would be an extent nobody stated, and an extent is what every
    angular quantity downstream is divided by.
    """
    with pytest.raises(TypeError, match="sample_pitch_m"):
        ScalarField(  # type: ignore[call-arg]
            u=np.ones((4, 6), dtype=np.complex64),
            wavelength_m=WAVELENGTH_M,
            reference_surface=SURFACE,
        )


def test_the_pitch_is_a_dy_dx_pair_in_metres() -> None:
    field = _field()
    assert field.sample_pitch_m == (1.25e-6, 2.5e-6)
    assert all(isinstance(value, float) for value in field.sample_pitch_m)


def test_a_single_scalar_pitch_is_not_broadcast() -> None:
    """Anisotropic sampling is common enough that guessing is not safe."""
    with pytest.raises(ContractError, match="SHAPE_MISMATCH"):
        _field(sample_pitch_m=(1e-6,))


def test_the_extent_comes_from_the_pitch_not_from_the_shape_alone() -> None:
    """M1 measured a 256x256 Chromatix input coming back 1756x1756.

    A shape is not an extent. The extent describes the array as it stands,
    padding included, and `pad_width` is what recovers the modelled window.
    """
    field = _field(u=np.ones((256, 256), dtype=np.complex64))
    assert field.extent_m == pytest.approx((256 * 1.25e-6, 256 * 2.5e-6))
    padded = _field(u=np.ones((1756, 1756), dtype=np.complex64), pad_width=750, padded=True)
    assert padded.shape == (1756, 1756)
    assert padded.pad_width == 750
    assert padded.shape[0] - 2 * padded.pad_width == 256


# --- amplitude, not intensity ---


def test_a_real_array_is_refused_as_an_intensity() -> None:
    """`|U|` has already thrown away the phase; no later operation recovers it."""
    with pytest.raises(ContractError) as caught:
        _field(u=np.ones((4, 6), dtype=np.float64))
    assert caught.value.code == "DTYPE_KIND_MISMATCH"
    assert "intensity" in str(caught.value)


def test_a_complex64_field_stays_complex64() -> None:
    """Chromatix has no complex128 path at any device; widening here would lie."""
    assert _field().state.dtype is DType.COMPLEX64


def test_discrete_power_is_relative_and_uses_both_pitches() -> None:
    """`sum |u|^2 dy dx`. Not watts -- there is no radiometric normalization here."""
    field = _field(u=np.ones((4, 6), dtype=np.complex128))
    assert field.discrete_power() == pytest.approx(24 * 1.25e-6 * 2.5e-6)


def test_the_field_is_frozen() -> None:
    field = _field()
    with pytest.raises(dataclasses.FrozenInstanceError):
        field.wavelength_m = 1e-6  # type: ignore[misc]


# --- acceptance criterion 3: the n // 2 grid ---


def test_coordinates_put_zero_at_the_origin_index() -> None:
    ys, xs = _field(u=np.ones((4, 7), dtype=np.complex64)).coordinates()
    frame = Frame()
    assert ys[frame.origin_index(4)] == 0.0
    assert xs[frame.origin_index(7)] == 0.0


def test_coordinates_are_y_then_x_on_a_non_square_field() -> None:
    """A transpose here is invisible in any rotationally symmetric case."""
    ys, xs = _field(u=np.ones((4, 6), dtype=np.complex64)).coordinates()
    assert ys.shape == (4,)
    assert xs.shape == (6,)
    assert float(ys[1] - ys[0]) == pytest.approx(1.25e-6)
    assert float(xs[1] - xs[0]) == pytest.approx(2.5e-6)


def test_coordinates_are_built_in_the_fields_own_real_precision() -> None:
    """A complex64 field must not produce float64 axes for everything to demote."""
    ys, xs = _field().coordinates()
    assert ys.dtype == np.float32
    assert xs.dtype == np.float32
    ys64, _ = _field(u=np.ones((4, 6), dtype=np.complex128)).coordinates()
    assert ys64.dtype == np.float64


def test_the_grid_is_symmetric_only_for_an_odd_axis() -> None:
    """The half-sample asymmetry of `n // 2`, pinned rather than discovered later."""
    ys_even, _ = _field(u=np.ones((4, 6), dtype=np.complex128)).coordinates()
    assert float(ys_even[0]) == pytest.approx(-2 * 1.25e-6)
    assert float(ys_even[-1]) == pytest.approx(1 * 1.25e-6)
    ys_odd, _ = _field(u=np.ones((5, 6), dtype=np.complex128)).coordinates()
    assert float(ys_odd[0]) == pytest.approx(-float(ys_odd[-1]))


# --- acceptance criterion 2: typed validity ---


def test_validity_defaults_to_the_empty_set_which_is_a_claim() -> None:
    field = _field()
    assert field.validity == frozenset()
    assert isinstance(field.validity, frozenset)


def test_a_consumer_can_branch_on_validity() -> None:
    """AC 2. The branch a propagation operator will make."""

    def would_propagate(field: ScalarField) -> bool:
        return "surface_only" not in field.validity

    assert would_propagate(_field())
    assert not would_propagate(_field(validity={"surface_only"}))


def test_the_two_limitations_the_risk_note_names_are_independently_expressible() -> None:
    """The failure mode: expressing one hides the other.

    The no-curvature-term limitation (CHE-50, R07) and the carrier-removed phase
    question (R06) are both properties only a complex comparison detects. If
    `validity` could hold one value, declaring either would silently un-declare
    the other -- which is how the curvature limitation survived three milestones.
    """
    both = _field(validity={"no_wavefront_curvature_term", "carrier_removed_phase"})
    assert "no_wavefront_curvature_term" in both.validity
    assert "carrier_removed_phase" in both.validity

    only_curvature = _field(validity={"no_wavefront_curvature_term"})
    assert "carrier_removed_phase" not in only_curvature.validity


def test_an_unknown_flag_is_refused_rather_than_carried() -> None:
    """This is what makes it typed rather than a provenance string.

    A free-form entry nothing can branch on is the thing the field replaces.
    """
    with pytest.raises(ContractError) as caught:
        _field(validity={"mostly_fine"})
    assert caught.value.code == "UNKNOWN_VALIDITY_FLAG"


def test_validity_accepts_any_iterable_and_normalizes_to_a_frozenset() -> None:
    assert _field(validity=["surface_only", "surface_only"]).validity == frozenset({"surface_only"})


def test_every_flag_carries_its_measured_consequence() -> None:
    """A limitation with no stated cost is a name, not a declaration."""
    assert set(VALIDITY_NOTES) == set(VALIDITY_FLAGS)
    assert "1.2 rad" in VALIDITY_NOTES["no_wavefront_curvature_term"]
    assert "1e-3" in VALIDITY_NOTES["no_wavefront_curvature_term"]


# --- acceptance criterion 5: exactly one scalar representation ---


def test_the_module_defines_exactly_one_class() -> None:
    import ast
    from pathlib import Path

    source = Path(representations.scalar.__file__).read_text(encoding="utf-8")
    classes = [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)]
    assert classes == ["ScalarField"]


@pytest.mark.parametrize("banned", ["PSF", "ComplexField", "WavefrontSamples", "IntensityField"])
def test_there_is_no_second_scalar_type_and_no_psf(banned: str) -> None:
    """PSF is a measurement (R11). Serializability is not what makes a representation."""
    assert not hasattr(representations, banned)
