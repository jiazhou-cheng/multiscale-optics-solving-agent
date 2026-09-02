"""What the focal-plane transform refuses, and what it registers as.

CHE-209 (R06.4) acceptance criteria 7 and 8, plus the refusal half of the
behaviour contract. The closed-form physics is
`tests/physics/test_focal_plane_transform.py`; this file is the boundary.

The pad-width refusal is R06.3's acceptance criterion 6 as well: `N` is in the
denominator of `lambda f / (n N dx)`, so padding an optical Fourier transform
regrids it. The operator declines to choose a meaning for that rather than
picking one silently.
"""

from __future__ import annotations

import numpy as np
import pytest
from chromatix_support import PITCH_M, SHAPE, a_scalar_field

from operations import CATALOG, OperationKind, resolve
from representations import ContractError, ScalarField
from solvers.chromatix import CAPABILITIES, DIRECTIONS, focal_plane_transform

FOCAL_LENGTH_M = 20e-3


def a_model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {"target_surface": "back_focal"}
    model.update(overrides)
    return model


# ---------------------------------------------------------------------------
# 1. The model mapping is checked, and padding is refused with its reason
# ---------------------------------------------------------------------------


def test_a_pad_width_is_refused_because_it_would_regrid_the_fourier_plane() -> None:
    """R06.3 criterion 6 / R06.4's padding decision, as a refusal that fires.

    Padding an ASM propagation changes how much wraparound it suffers. Padding an
    optical FT changes `N`, and `N` sets the output pitch -- so the same argument
    that is a guard band on one operation is a quiet regrid on the other.
    """
    with pytest.raises(ValueError) as caught:
        focal_plane_transform(
            a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model=a_model(pad_width=8)
        )
    message = str(caught.value)
    assert "pad_width" in message
    assert "N" in message and "regrid" in message


def test_an_unrecognized_key_is_refused_rather_than_discarded() -> None:
    """The rule `propagate` set: a silently dropped key is a different run."""
    with pytest.raises(ValueError, match="targt_surface"):
        focal_plane_transform(
            a_scalar_field(),
            focal_length_m=FOCAL_LENGTH_M,
            model={"targt_surface": "back_focal"},
        )
    with pytest.raises(ValueError, match="target_surface"):
        focal_plane_transform(a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model={})
    with pytest.raises(ValueError, match="direction"):
        focal_plane_transform(
            a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model=a_model(direction="reverse")
        )
    with pytest.raises(ValueError, match="target_surface"):
        focal_plane_transform(
            a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model=a_model(target_surface="  ")
        )


@pytest.mark.parametrize("focal_length_m", [0.0, -20e-3, float("inf"), float("nan")])
def test_a_focal_length_that_is_not_a_positive_length_is_refused(focal_length_m: float) -> None:
    """The direction is `model['direction']`, so a signed `f` would be a second,
    disagreeable statement of the same fact."""
    with pytest.raises(ValueError, match="focal_length_m"):
        focal_plane_transform(a_scalar_field(), focal_length_m=focal_length_m, model=a_model())


def test_both_declared_directions_are_accepted() -> None:
    """Two members, and the surface moves the other way for the second one."""
    assert DIRECTIONS == ("forward", "inverse")
    source = a_scalar_field()
    forward = focal_plane_transform(source, focal_length_m=FOCAL_LENGTH_M, model=a_model())
    inverse = focal_plane_transform(
        source, focal_length_m=FOCAL_LENGTH_M, model=a_model(direction="inverse")
    )
    assert forward.reference_surface.z_m == pytest.approx(+2.0 * FOCAL_LENGTH_M)
    assert inverse.reference_surface.z_m == pytest.approx(-2.0 * FOCAL_LENGTH_M)
    assert forward.sample_pitch_m == inverse.sample_pitch_m


# ---------------------------------------------------------------------------
# 2. Declarations the field carries, and what they cost it
# ---------------------------------------------------------------------------


def test_a_surface_only_field_is_refused() -> None:
    """Valid at its own surface and nowhere else: the conjugate focal plane is a
    different physical claim, not a less accurate one."""
    source = a_scalar_field()
    surface_only = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        validity=frozenset({"surface_only"}),
    )
    with pytest.raises(ContractError) as caught:
        focal_plane_transform(surface_only, focal_length_m=FOCAL_LENGTH_M, model=a_model())
    assert caught.value.code == "REPRESENTATION_INCONSISTENT"


def test_a_still_padded_field_is_refused() -> None:
    """A transform mixes every sample into every other one, so the modelled window
    is not recoverable afterwards -- and the pad samples are in the `N` that sets
    the output pitch."""
    source = a_scalar_field()
    padded = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        pad_width=4,
        padded=True,
    )
    with pytest.raises(ContractError) as caught:
        focal_plane_transform(padded, focal_length_m=FOCAL_LENGTH_M, model=a_model())
    assert caught.value.code == "PAD_STATE_UNKNOWN"


def test_a_complex128_field_is_refused_before_the_backend_is_reached() -> None:
    """The same capability refusal `propagate` makes: the backend has one field
    storage dtype and this boundary names the loss rather than absorbing it."""
    with pytest.raises(ValueError) as caught:
        focal_plane_transform(
            a_scalar_field(dtype="complex128"), focal_length_m=FOCAL_LENGTH_M, model=a_model()
        )
    assert getattr(caught.value, "code", None) == "LOSSY_DOWNCAST_REQUIRED"


def test_the_medium_index_is_read_from_the_surface_and_changes_the_sampling() -> None:
    """`n` enters as `lambda / n`, and it is read from the field rather than taken
    as an argument that could disagree with it."""
    vacuum = focal_plane_transform(a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model=a_model())
    immersed = focal_plane_transform(
        a_scalar_field(medium_index=1.33), focal_length_m=FOCAL_LENGTH_M, model=a_model()
    )
    assert immersed.reference_surface.medium_index == 1.33
    assert immersed.sample_pitch_m == pytest.approx(
        tuple(value / 1.33 for value in vacuum.sample_pitch_m), rel=1e-12
    )


def test_the_namespace_and_shape_the_caller_handed_in_come_back() -> None:
    """A NumPy caller gets NumPy back, on the same grid, with the pad state reset
    to what it now means: nothing."""
    out = focal_plane_transform(a_scalar_field(), focal_length_m=FOCAL_LENGTH_M, model=a_model())
    assert isinstance(out.u, np.ndarray)
    assert out.shape == SHAPE
    assert out.pad_width == 0 and out.padded is False
    assert out.sample_pitch_m != PITCH_M, "this is the operation that regrids"
    assert out.frame == a_scalar_field().frame


# ---------------------------------------------------------------------------
# 3. Registration
# ---------------------------------------------------------------------------


def test_the_transform_registers_as_a_physical_operator() -> None:
    """Criterion 7. A physical operator, never a coupler.

    The state at the back focal plane is a different physical state from the one
    at the front focal plane -- the field has been through a lens. A coupler would
    be the opposite: the same state, described differently. That the output is on
    a different *grid* is not what decides it; a regrid is bookkeeping, and this
    operation would still be an operator without one.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `solvers/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.
    """
    descriptor = next(d for d in CATALOG if d.operation_id == "O_FOCAL_PLANE_TRANSFORM")
    assert descriptor.kind is OperationKind.PHYSICAL_OPERATOR
    assert descriptor.kind is not OperationKind.COUPLER
    assert descriptor.derivative == "forward_only"
    assert descriptor.capabilities == CAPABILITIES
    assert resolve("O_FOCAL_PLANE_TRANSFORM") is focal_plane_transform


def test_the_module_defines_no_class() -> None:
    """Criterion 8. Class delta 0: the request is the arguments, the result is the
    return value, and the failures are the two vocabularies that already exist."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "solvers" / "chromatix" / "focal_plane.py"
    ).read_text(encoding="utf-8")
    assert [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)] == []
