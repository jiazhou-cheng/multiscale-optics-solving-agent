"""Contract tests for ``core.optical_assembly`` (CHE-139).

The module under test turns separately specified physical components into one
``OpticalSystemSpec``. Everything worth testing about it is a *convention*:
where a component's thickness stops, which medium follows which surface, what
happens to a radius when a lens is flipped, and where the air gaps land in the
concatenated sequence. Each of those has exactly one right answer and a
plausible wrong one that would still trace, so each is asserted rather than
inspected.

Every test here is solver-free: nothing imports Optiland. The corresponding
*executable* evidence -- that the built ``Optic`` really carries these radii at
these vertex positions -- lives in the benchmark itself, which reads the built
system back and diffs it against ``surface_table`` on every run.
"""

from __future__ import annotations

import pytest

from core.optical_assembly import (
    OPTICAL_COMPONENT_SPEC_VERSION,
    ComponentPlacement,
    ComponentSpec,
    Orientation,
    assemble_optical_system,
    component_surface_span,
    surface_table,
)
from core.optical_system import (
    AirMaterialSpec,
    ApertureSpec,
    EvenAsphereGeometrySpec,
    FieldSpec,
    IdealMaterialSpec,
    MaterialKind,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
    WavelengthSpec,
)

CROWN = IdealMaterialSpec(refractive_index=1.5168)
FLINT = IdealMaterialSpec(refractive_index=1.6727)


def doublet() -> ComponentSpec:
    """A cemented three-surface doublet: two glasses, no internal air."""
    return ComponentSpec(
        name="Doublet",
        geometries=(
            SphericalGeometrySpec(radius_mm=60.0),
            SphericalGeometrySpec(radius_mm=-45.0),
            SphericalGeometrySpec(radius_mm=-130.0),
        ),
        internal_thicknesses_mm=(5.0, 2.0),
        internal_materials=(CROWN, FLINT),
        clear_aperture_mm=22.86,
        surface_comments=("crown front", "cement", "flint back"),
    )


def plano_convex() -> ComponentSpec:
    return ComponentSpec(
        name="PlanoConvex",
        geometries=(SphericalGeometrySpec(radius_mm=50.0), PlaneGeometrySpec()),
        internal_thicknesses_mm=(4.0,),
        internal_materials=(CROWN,),
        clear_aperture_mm=22.86,
    )


def _system(*placements: ComponentPlacement, **kwargs):
    return assemble_optical_system(
        name=kwargs.pop("name", "System"),
        placements=placements,
        aperture=ApertureSpec(value_mm=20.0),
        fields=(FieldSpec(y_deg=0.0), FieldSpec(y_deg=2.0)),
        wavelengths=(WavelengthSpec(value_um=0.589, is_primary=True),),
        **kwargs,
    )


# --- the component/surface convention boundary ------------------------------


def test_component_owns_no_trailing_thickness():
    """The distance after the last vertex comes from the assembly, not the part.

    This is the whole reason ``ComponentSpec`` is shaped the way it is: the same
    doublet must be reusable at any spacing, so its own declaration cannot
    contain one.
    """
    component = doublet()
    assert component.surface_count == 3
    assert component.axial_thickness_mm == 7.0

    near = component.surfaces(trailing_thickness_mm=1.0)
    far = component.surfaces(trailing_thickness_mm=500.0)
    # Only the last surface's thickness differs; the part itself is identical.
    assert [s.thickness_mm for s in near] == [5.0, 2.0, 1.0]
    assert [s.thickness_mm for s in far] == [5.0, 2.0, 500.0]
    assert near[:-1] == far[:-1]


def test_material_is_the_medium_after_the_surface():
    """A lens is glass-carrying surfaces followed by an air-carrying one.

    The sequential convention is easy to get off by one, and off by one here
    means the cement interface has air in it -- a system that traces and is
    wrong.
    """
    surfaces = doublet().surfaces(trailing_thickness_mm=10.0)
    assert surfaces[0].material == CROWN
    assert surfaces[1].material == FLINT
    assert surfaces[2].material.kind is MaterialKind.AIR
    assert isinstance(surfaces[2].material, AirMaterialSpec)


def test_stop_index_names_a_surface_of_the_requested_orientation():
    forward = doublet().surfaces(trailing_thickness_mm=10.0, stop_surface_index=0)
    assert [s.is_stop for s in forward] == [True, False, False]

    reversed_ = doublet().surfaces(
        trailing_thickness_mm=10.0,
        orientation=Orientation.REVERSED,
        stop_surface_index=0,
    )
    # Index 0 of the REVERSED component is the flint face, i.e. still the
    # front-most physical surface -- which is what makes the stop stay put when a
    # lens is flipped in its mount.
    assert [s.is_stop for s in reversed_] == [True, False, False]
    assert reversed_[0].geometry.resolved_radius_mm == pytest.approx(130.0)


def test_trailing_thickness_and_stop_index_are_validated():
    component = doublet()
    with pytest.raises(PrescriptionError) as negative:
        component.surfaces(trailing_thickness_mm=-1.0)
    assert negative.value.code == "COMPONENT_TRAILING_THICKNESS_INVALID"

    with pytest.raises(PrescriptionError) as out_of_range:
        component.surfaces(trailing_thickness_mm=1.0, stop_surface_index=3)
    assert out_of_range.value.code == "COMPONENT_STOP_INDEX_OUT_OF_RANGE"


# --- reversal ----------------------------------------------------------------


def test_reversal_negates_radii_reverses_order_and_keeps_conic():
    component = ComponentSpec(
        name="Conic",
        geometries=(
            SphericalGeometrySpec(radius_mm=60.0, conic=-0.5),
            SphericalGeometrySpec(radius_mm=-45.0, conic=0.25),
            SphericalGeometrySpec(radius_mm=-130.0),
        ),
        internal_thicknesses_mm=(5.0, 2.0),
        internal_materials=(CROWN, FLINT),
    )
    flipped = component.reversed()
    radii = [g.resolved_radius_mm for g in flipped.geometries]
    assert radii == pytest.approx([130.0, 45.0, -60.0])
    # z -> -z maps c -> -c with k untouched, so the conic travels with its
    # surface and is NOT negated.
    assert [g.conic for g in flipped.geometries] == pytest.approx([0.0, 0.25, -0.5])
    assert flipped.internal_thicknesses_mm == (2.0, 5.0)
    assert flipped.internal_materials == (FLINT, CROWN)
    assert flipped.clear_aperture_mm == component.clear_aperture_mm


def test_reversal_is_an_involution_on_the_optics():
    """Flipping twice restores the optical content, not merely the name."""
    component = doublet()
    twice = component.reversed().reversed()
    assert twice.geometries == component.geometries
    assert twice.internal_thicknesses_mm == component.internal_thicknesses_mm
    assert twice.internal_materials == component.internal_materials
    assert twice.surface_comments == component.surface_comments


def test_plane_survives_reversal_as_a_plane():
    flipped = plano_convex().reversed()
    assert isinstance(flipped.geometries[0], PlaneGeometrySpec)
    assert flipped.geometries[1].resolved_radius_mm == pytest.approx(-50.0)
    assert flipped.internal_materials == (CROWN,)


def test_reversing_an_even_asphere_is_refused_not_approximated():
    """The broken twin of the reversal rule.

    Negating a radius is only half of reversing an even asphere -- every
    polynomial coefficient negates too, and that has no verification here. The
    refusal is the point: a silently half-reversed asphere would be a different
    surface with no error.
    """
    component = ComponentSpec(
        name="Asphere",
        geometries=(
            EvenAsphereGeometrySpec(radius_mm=20.0, coefficients=(1.0e-4,)),
            PlaneGeometrySpec(),
        ),
        internal_thicknesses_mm=(3.0,),
        internal_materials=(CROWN,),
    )
    # Forward is fine; only the reversal is refused.
    assert len(component.surfaces(trailing_thickness_mm=5.0)) == 2
    with pytest.raises(PrescriptionError) as refusal:
        component.reversed()
    assert refusal.value.code == "COMPONENT_REVERSAL_UNSUPPORTED_GEOMETRY"
    assert "geometries[0]" in str(refusal.value)


# --- component validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {"geometries": (SphericalGeometrySpec(radius_mm=50.0),),
             "internal_thicknesses_mm": (), "internal_materials": ()},
            "COMPONENT_TOO_FEW_SURFACES",
        ),
        (
            {"internal_thicknesses_mm": (5.0, 2.0, 1.0)},
            "COMPONENT_INTERNAL_LENGTH_MISMATCH",
        ),
        (
            {"internal_materials": (CROWN,)},
            "COMPONENT_INTERNAL_LENGTH_MISMATCH",
        ),
        (
            {"internal_thicknesses_mm": (5.0, 0.0)},
            "COMPONENT_INTERNAL_THICKNESS_INVALID",
        ),
        (
            {"internal_thicknesses_mm": (5.0, -2.0)},
            "COMPONENT_INTERNAL_THICKNESS_INVALID",
        ),
        ({"clear_aperture_mm": 0.0}, "COMPONENT_CLEAR_APERTURE_INVALID"),
        ({"surface_comments": ("only one",)}, "COMPONENT_COMMENT_LENGTH_MISMATCH"),
    ],
)
def test_component_refuses_inconsistent_declarations(kwargs, code):
    base = {
        "name": "Doublet",
        "geometries": (
            SphericalGeometrySpec(radius_mm=60.0),
            SphericalGeometrySpec(radius_mm=-45.0),
            SphericalGeometrySpec(radius_mm=-130.0),
        ),
        "internal_thicknesses_mm": (5.0, 2.0),
        "internal_materials": (CROWN, FLINT),
    }
    with pytest.raises(PrescriptionError) as failure:
        ComponentSpec(**{**base, **kwargs})
    assert failure.value.code == code


def test_component_declares_its_schema_version():
    assert doublet().spec_version == OPTICAL_COMPONENT_SPEC_VERSION
    with pytest.raises(PrescriptionError) as failure:
        ComponentSpec(
            spec_version="optical-component-spec/999",
            name="Doublet",
            geometries=(SphericalGeometrySpec(radius_mm=60.0), PlaneGeometrySpec()),
            internal_thicknesses_mm=(5.0,),
            internal_materials=(CROWN,),
        )
    assert failure.value.code == "COMPONENT_SCHEMA_VERSION_UNSUPPORTED"


def test_air_gap_must_be_finite_and_non_negative():
    with pytest.raises(PrescriptionError) as failure:
        ComponentPlacement(component=doublet(), air_gap_after_mm=-5.0)
    assert failure.value.code == "PLACEMENT_AIR_GAP_INVALID"


# --- assembly ----------------------------------------------------------------


def test_assembly_concatenates_in_order_with_the_gaps_on_the_right_surfaces():
    spec = _system(
        ComponentPlacement(component=doublet(), air_gap_after_mm=50.0),
        ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
    )
    rows = surface_table(spec)
    # A plane reports radius None rather than inf: the schema has no infinite
    # radius, which is exactly why `math.inf` can never reach a JSON record.
    assert [row["radius_mm"] for row in rows] == [60.0, -45.0, -130.0, 50.0, None]
    # The 50 mm gap sits on the doublet's LAST surface and nowhere else; the
    # 30 mm image distance sits on the singlet's last surface.
    assert [row["thickness_mm"] for row in rows] == [5.0, 2.0, 50.0, 4.0, 30.0]
    # Vertices accumulate the same thicknesses: 0, 5, 7, 57.17-like arithmetic.
    assert [row["vertex_z_mm"] for row in rows] == pytest.approx([0.0, 5.0, 7.0, 57.0, 61.0])
    # Air only where the assembly put it, glass everywhere the components did.
    assert [row["material_kind"] for row in rows] == [
        "ideal", "ideal", "air", "ideal", "air",
    ]
    assert [row["is_stop"] for row in rows] == [True, False, False, False, False]


def test_assembly_emits_exactly_one_stop_on_the_named_surface():
    spec = _system(
        ComponentPlacement(component=doublet(), air_gap_after_mm=50.0),
        ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
        stop_component_index=1,
        stop_surface_index=1,
    )
    # OpticalSystemSpec itself refuses zero or several stops, so reaching here at
    # all proves there is exactly one; this pins WHICH.
    assert spec.stop_index == 4
    assert sum(surface.is_stop for surface in spec.surfaces) == 1


def test_reversing_a_placement_changes_the_assembled_sequence_and_the_fingerprint():
    forward = _system(
        ComponentPlacement(component=doublet(), air_gap_after_mm=50.0),
        ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
    )
    reversed_first = _system(
        ComponentPlacement(
            component=doublet(),
            orientation=Orientation.REVERSED,
            air_gap_after_mm=50.0,
        ),
        ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
    )
    forward_radii = [row["radius_mm"] for row in surface_table(forward)]
    reversed_radii = [row["radius_mm"] for row in surface_table(reversed_first)]
    assert forward_radii[:3] == [60.0, -45.0, -130.0]
    assert reversed_radii[:3] == [130.0, 45.0, -60.0]
    forward_z = [row["vertex_z_mm"] for row in surface_table(forward)]
    reversed_z = [row["vertex_z_mm"] for row in surface_table(reversed_first)]
    # The part is 7 mm thick either way, so its outer envelope and everything
    # downstream of it sit at the same place -- a lens turned round in its mount
    # does not move the next lens.
    assert forward_z[2:] == pytest.approx(reversed_z[2:])
    assert forward_z[0] == reversed_z[0] == 0.0
    # The INTERNAL cement interface does move, and must: reversing swaps the
    # 5 mm crown and the 2 mm flint, so the interface is 2 mm in rather than
    # 5 mm in. A reversal that left this at 5 mm would have reordered the
    # surfaces without reordering the glass -- a doublet with its elements
    # swapped, which traces fine and is a different lens.
    assert forward_z[1] == pytest.approx(5.0)
    assert reversed_z[1] == pytest.approx(2.0)
    assert forward.fingerprint() != reversed_first.fingerprint()


def test_assembly_is_deterministic():
    def build():
        return _system(
            ComponentPlacement(component=doublet(), air_gap_after_mm=50.0),
            ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
        )

    assert build().fingerprint() == build().fingerprint()


def test_assembly_refuses_no_components_and_an_out_of_range_stop():
    with pytest.raises(PrescriptionError) as empty:
        _system()
    assert empty.value.code == "ASSEMBLY_NO_COMPONENTS"

    with pytest.raises(PrescriptionError) as stop:
        _system(
            ComponentPlacement(component=doublet(), air_gap_after_mm=10.0),
            stop_component_index=2,
        )
    assert stop.value.code == "ASSEMBLY_STOP_COMPONENT_OUT_OF_RANGE"

    with pytest.raises(PrescriptionError) as surface:
        _system(
            ComponentPlacement(component=doublet(), air_gap_after_mm=10.0),
            stop_surface_index=7,
        )
    assert surface.value.code == "COMPONENT_STOP_INDEX_OUT_OF_RANGE"


def test_component_span_and_surface_table_agree_on_vertex_positions():
    """Two independent readings of the same assembly must coincide.

    ``component_surface_span`` walks the placements; ``surface_table`` walks the
    emitted prescription. They are the pair the benchmark cross-checks, so a
    drift between them has to fail here.
    """
    placements = (
        ComponentPlacement(component=doublet(), air_gap_after_mm=50.0),
        ComponentPlacement(component=plano_convex(), air_gap_after_mm=30.0),
    )
    spans = component_surface_span(placements)
    rows = surface_table(_system(*placements))
    assert [span["surface_count"] for span in spans] == [3, 2]
    for span in spans:
        first, last = span["first_surface_index"], span["last_surface_index"]
        assert rows[first]["vertex_z_mm"] == pytest.approx(span["first_vertex_z_mm"])
        assert rows[last]["vertex_z_mm"] == pytest.approx(span["last_vertex_z_mm"])
        # The gap the assembly declared is the thickness carried by the
        # component's last surface, and by nothing else.
        assert rows[last]["thickness_mm"] == pytest.approx(span["air_gap_after_mm"])


def test_span_reports_the_reversed_component_under_its_reversed_identity():
    spans = component_surface_span(
        (
            ComponentPlacement(
                component=doublet(),
                orientation=Orientation.REVERSED,
                air_gap_after_mm=50.0,
            ),
        )
    )
    assert spans[0]["orientation"] == "reversed"
    assert "reversed" in spans[0]["component_name"]
    assert spans[0]["axial_thickness_mm"] == pytest.approx(7.0)
