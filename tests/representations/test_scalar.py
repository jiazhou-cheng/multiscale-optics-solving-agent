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
    # CHE-227 (R02.5): the same requirement applied to the fourth flag. The two
    # numbers are the pair that makes the flag worth having -- the same grid, a
    # hard edge and a soft one, four orders of magnitude apart.
    assert "2.3e-1" in VALIDITY_NOTES["paraxial"]
    assert "4.9e-6" in VALIDITY_NOTES["paraxial"]


# --- CHE-227 (R02.5): the fourth flag ---


def test_the_vocabulary_is_the_four_flags_that_have_landed() -> None:
    """The ratchet. A flag joins in the change that gives something to declare.

    `paraxial` is CHE-227 (R02.5), and it exists because R06.11's Fresnel
    propagation had nothing true to say about itself:
    `S_SOURCE_GAUSSIAN_BEAM`'s validity already recorded the gap -- "an off-waist
    Gaussian is a paraxial solution and no ValidityFlag says 'paraxial'".
    """
    assert VALIDITY_FLAGS == (
        "surface_only",
        "no_wavefront_curvature_term",
        "carrier_removed_phase",
        "paraxial",
    )


def test_a_paraxial_field_constructs_and_round_trips_the_flag() -> None:
    assert _field(validity={"paraxial"}).validity == frozenset({"paraxial"})


def test_paraxial_is_independent_of_the_other_three() -> None:
    """AC 4. All four are true of one array at once, and none hides another.

    A Fresnel propagation of a ray-reconstructed field is the concrete case: it is
    paraxial, its phase is carrier-removed, and the wavelet sum that produced it
    carried no curvature term. Declaring the union has to leave all of them
    readable, which is the whole reason `validity` is a set.
    """
    every = _field(validity=set(VALIDITY_FLAGS))
    assert every.validity == frozenset(VALIDITY_FLAGS)

    only_paraxial = _field(validity={"paraxial"})
    assert "carrier_removed_phase" not in only_paraxial.validity
    assert "no_wavefront_curvature_term" not in only_paraxial.validity
    assert "surface_only" not in only_paraxial.validity


def _flag_literals(path: Any) -> set[str]:
    """Validity flags named as string *constants* in one module, docstrings stripped.

    The same AST reading `tests/planning/test_graph.py::_string_constants` uses, and
    for the same reason: prose has to be able to name a flag -- `sources/` and
    `operations/catalog.py` both discuss `paraxial` at length -- while a literal in
    the code is a declaration. Quoting style cannot hide one either way.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value in VALIDITY_FLAGS
    }


def test_no_landed_operation_declares_paraxial_yet() -> None:
    """AC 5. R02.5 lands the vocabulary; R06.11 is what makes it reachable.

    A flag no producer sets is a claim nothing can make, so this asserts the
    vocabulary arrived *before* its first user rather than beside it -- and it is
    the assertion R06.11 has to come past and update, which is the point rather
    than brittleness. `test_semantic_types_are_the_boundaries_that_landed` is the
    same ratchet on the other closed vocabulary.

    A *literal*, not a mention: `sources/gaussian_beam.py` and
    `operations/catalog.py` both say the word in prose, and one of them says it to
    record that this flag did not exist.

    And only where a `ScalarField` could be built, because **the word is
    overloaded and the vocabularies are unrelated**: `backends/optiland/launch.py`
    has `AIMING_MODES = ("paraxial", "iterative", "robust")`, the pinned solver's
    ray-aiming setting. It is a literal, it is not a docstring, and it is not a
    validity flag -- that module never imports `ScalarField` and could not declare
    one. Scoping to the importers is what tells the two apart without this test
    having to know about ray aiming.
    """
    from pathlib import Path

    root = Path(representations.scalar.__file__).resolve().parents[1]
    declared_in = Path(representations.scalar.__file__).resolve()
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path)
        and path.resolve() != declared_in
        and "ScalarField" in path.read_text(encoding="utf-8")
        and "paraxial" in _flag_literals(path)
    )
    assert offenders == [], (
        f"{offenders} declare the 'paraxial' flag, but R02.5 lands the vocabulary "
        "alone. The ticket that makes a field declare it updates this test."
    )


def test_the_literal_check_can_fail(tmp_path: Any) -> None:
    """The meta-check: a prose mention must not count and a declaration must.

    Without this the assertion above would pass on an empty reading of every
    module, which is the failure `tests/planning/test_graph.py` records having made
    with the same check.
    """
    from pathlib import Path

    module = Path(tmp_path) / "probe.py"
    for source in ('V = {"paraxial"}\n', "V = {'paraxial'}\n", 'f(validity={"paraxial"})\n'):
        module.write_text(source)
        assert "paraxial" in _flag_literals(module), source
    module.write_text('"""No ValidityFlag says paraxial or \'paraxial\'."""\n')
    assert _flag_literals(module) == set()


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
