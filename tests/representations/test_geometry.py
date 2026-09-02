"""`Frame` and `ReferenceSurface` reject what they claim to reject.

CHE-174 (R02.2). Two data types whose whole value is refusing a declaration, so
the tests are mostly negative: a validator that has never been shown to fail is a
validator nobody has tested.

The positive half is small on purpose -- constructing the canonical frame and a
plausible surface -- with one exception. The `(y, x)` order and the `n // 2`
origin are checked through `field_axis_index` and `origin_index` against
*asymmetric* inputs (a non-square shape, an even and an odd sample count),
because acceptance criterion 3 asks for them asserted rather than commented, and
the errors both conventions guard against -- a transpose, a half-sample shift --
are exactly the ones a square array or a symmetric grid cannot show.
"""

from __future__ import annotations

import dataclasses

import pytest

from representations import (
    AXIS_ORDER,
    HANDEDNESS,
    ORIGIN_RULE,
    PROPAGATION_AXIS,
    Frame,
    ReferenceSurface,
)

# --- Frame ---


def test_the_default_frame_is_the_reused_convention() -> None:
    """The four constants are the pre-rewrite ones, not re-derived here."""
    frame = Frame()
    assert frame.axis_order == AXIS_ORDER == "(y, x)"
    assert frame.handedness == HANDEDNESS == "right-handed"
    assert frame.origin_rule == ORIGIN_RULE == "array index n//2 is coordinate zero"
    assert frame.propagation_axis == PROPAGATION_AXIS == "+z"


def test_stating_the_convention_explicitly_is_accepted() -> None:
    """A caller that spells all four out gets the same frame as the default."""
    assert Frame(AXIS_ORDER, HANDEDNESS, ORIGIN_RULE, PROPAGATION_AXIS) == Frame()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("axis_order", "(x, y)"),
        ("axis_order", "(y, x, z)"),
        ("handedness", "left-handed"),
        ("origin_rule", "array index (n-1)/2 is coordinate zero"),
        ("propagation_axis", "-z"),
        ("propagation_axis", "+x"),
    ],
)
def test_an_inconsistent_frame_is_rejected_at_construction(field: str, value: str) -> None:
    """Every one of the four is checked eagerly, not on first use.

    `(x, y)` and `left-handed` are the two acceptance criterion 1 names. The other
    four are the near misses: a transposed order that is still three axes, the
    other defensible centring, and propagation reversed or along another axis.
    """
    with pytest.raises(ValueError, match=f"Frame.{field}"):
        Frame(**{field: value})


def test_the_frame_is_frozen() -> None:
    """Validated at construction is worth nothing if a field can be reassigned."""
    frame = Frame()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.handedness = "left-handed"  # type: ignore[misc]


def test_the_field_axis_order_is_y_then_x() -> None:
    frame = Frame()
    assert frame.field_axes == ("y", "x")
    assert frame.field_axis_index("y") == 0
    assert frame.field_axis_index("x") == 1


def test_the_axis_order_maps_a_non_square_array_the_right_way_round() -> None:
    """The assertion a square test array cannot make.

    A 3x5 field has three y samples and five x samples. Under a transposed frame
    it would have five y samples, and every rotationally symmetric check in the
    suite would still pass.
    """
    frame = Frame()
    shape = (3, 5)
    assert shape[frame.field_axis_index("y")] == 3
    assert shape[frame.field_axis_index("x")] == 5


def test_an_axis_the_frame_does_not_declare_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a spatial axis"):
        Frame().field_axis_index("z")


@pytest.mark.parametrize(("count", "expected"), [(1, 0), (2, 1), (3, 1), (4, 2), (256, 128)])
def test_the_origin_index_is_the_upper_centre_sample(count: int, expected: int) -> None:
    """`n // 2`, including for even `n` where the other choice is half a sample away."""
    assert Frame().origin_index(count) == expected


@pytest.mark.parametrize("count", [4, 5])
def test_a_grid_built_on_the_origin_rule_has_exact_zero_at_the_origin(count: int) -> None:
    """The rule as arithmetic: the sample at `origin_index` is coordinate zero.

    Both parities, because the odd case is symmetric and would pass under either
    centring convention -- only the even one distinguishes `n // 2` from
    `(n - 1) / 2`.
    """
    frame = Frame()
    pitch_m = 1.25e-6
    origin = frame.origin_index(count)
    coordinates = [(index - origin) * pitch_m for index in range(count)]
    assert coordinates[origin] == 0.0
    assert len(coordinates) == count


def test_a_zero_length_axis_has_no_origin() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        Frame().origin_index(0)


# --- ReferenceSurface ---


def _surface(**overrides: object) -> ReferenceSurface:
    fields: dict[str, object] = {
        "name": "exit_pupil",
        "z_m": -3.2e-3,
        "medium_index": 1.0,
    }
    fields.update(overrides)
    return ReferenceSurface(**fields)  # type: ignore[arg-type]


def test_a_plausible_surface_is_accepted_and_planar() -> None:
    surface = _surface(medium_index=1.5168, normal=(0.0, 0.0, 1.0))
    assert surface.name == "exit_pupil"
    assert surface.z_m == pytest.approx(-3.2e-3)
    assert surface.medium_index == pytest.approx(1.5168)
    assert surface.normal == (0.0, 0.0, 1.0)
    # Planar in content, not only in name: there is no curvature to declare.
    assert {f.name for f in dataclasses.fields(surface)} == {
        "name",
        "z_m",
        "medium_index",
        "normal",
    }


def test_the_medium_index_has_no_default() -> None:
    """It is read from the prescription, never assumed to be 1."""
    with pytest.raises(TypeError, match="medium_index"):
        ReferenceSurface(name="image_surface", z_m=0.0)  # type: ignore[call-arg]


def test_an_unnamed_surface_is_rejected() -> None:
    with pytest.raises(ValueError, match="name is empty"):
        _surface(name="")


@pytest.mark.parametrize("z_m", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_axial_coordinate_is_rejected(z_m: float) -> None:
    with pytest.raises(ValueError, match="z_m"):
        _surface(z_m=z_m)


@pytest.mark.parametrize("medium_index", [0.0, -1.0, -1.5, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_medium_index_is_rejected(medium_index: float) -> None:
    """Acceptance criterion 2, second half.

    Zero and negative are the two ways an index can be unphysical here; NaN is how
    an index arrives when it was computed from a missing dispersion coefficient
    rather than never set at all.
    """
    with pytest.raises(ValueError, match="medium_index"):
        _surface(medium_index=medium_index)


@pytest.mark.parametrize(
    "normal",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0 + 1e-6),
        (float("nan"), 0.0, 1.0),
    ],
)
def test_a_non_unit_normal_is_rejected(normal: tuple[float, float, float]) -> None:
    """Acceptance criterion 2, first half.

    The zero vector, a doubled axis, an unnormalized diagonal, a normal that is
    close but outside tolerance, and a NaN component. The near-unit case is the one
    that matters: it is what an un-renormalized composition of rotations produces.
    """
    with pytest.raises(ValueError, match="normal"):
        _surface(normal=normal)


def test_a_wrong_length_normal_is_rejected() -> None:
    with pytest.raises(ValueError, match="3-vector"):
        _surface(normal=(0.0, 1.0))


def test_a_tilted_unit_normal_is_accepted() -> None:
    """A tilted reference plane is expressible; only a non-unit one is not."""
    root_half = 0.5**0.5
    surface = _surface(normal=(0.0, root_half, root_half))
    assert surface.normal == pytest.approx((0.0, root_half, root_half))


def test_a_normal_within_round_off_of_unit_is_accepted() -> None:
    """The tolerance is a round-off allowance, so it has to actually allow round-off."""
    assert _surface(normal=(0.0, 0.0, 1.0 - 1e-12)).normal[2] == pytest.approx(1.0)


def test_integer_inputs_are_stored_as_floats() -> None:
    """SI quantities are floats; an int index would compare and divide differently."""
    surface = _surface(z_m=0, medium_index=1, normal=(0, 0, 1))
    assert isinstance(surface.z_m, float)
    assert isinstance(surface.medium_index, float)
    assert all(isinstance(value, float) for value in surface.normal)


def test_the_surface_is_frozen() -> None:
    surface = _surface()
    with pytest.raises(dataclasses.FrozenInstanceError):
        surface.z_m = 1.0  # type: ignore[misc]
