"""The commercial-catalog benchmark's anti-fabrication guards, and their broken twins.

`benchmarks/applied/commercial_lens_systems/catalog_sources.py` writes each
component twice on purpose -- once as the manufacturer's published words and once
as the model built from them -- and cross-checks the two at import time. A guard
nobody has watched fail is a guard nobody should trust, so every check here is
paired with a deliberately corrupted component that must be refused.

These tests import the benchmark's definition modules rather than duplicating
them. They are cheap and solver-free: nothing here builds an ``Optic`` or traces
a ray. The executable evidence for the benchmark itself lives in its own records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "applied" / "commercial_lens_systems"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from catalog_sources import (  # noqa: E402
    KBX058,
    KPX094,
    M10X,
    PAC052,
    CatalogComponent,
    catalog_components,
    require_supported,
    resolve_catalog_component,
)

from core.optical_system import (  # noqa: E402
    CatalogMaterialSpec,
    PrescriptionError,
    SphericalGeometrySpec,
)

SUPPORTED = (KPX094, KBX058, PAC052)


def _corrupt(base: CatalogComponent, **changes: object) -> None:
    """Rebuild a catalog component with one field changed. Must raise.

    The parameter is ``base`` rather than ``component`` because ``component`` is
    itself one of the fields these tests corrupt.
    """
    CatalogComponent(**{**base.model_dump(), **changes})


# --- the guards hold for the real components ---------------------------------


def test_every_supported_component_passes_its_own_transcription_check():
    """Importing the module already proved this; asserting it names the property."""
    for component in SUPPORTED:
        assert component.supported
        assert component.refusal is None
        assert component.component is not None
        assert len(component.radius_keys) == component.component.surface_count
        assert len(component.thickness_keys) == len(
            component.component.internal_thicknesses_mm
        )


def test_at_least_three_real_components_are_reconstructed():
    built = [c for c in catalog_components() if c.supported]
    assert len(built) >= 3
    # Distinct real parts from a real vendor, not three views of one lens.
    assert len({c.part_number for c in built}) == len(built)
    assert all(c.vendor.startswith("Newport") for c in built)


def test_every_published_value_names_a_real_source_document():
    for component in catalog_components():
        urls = component.source_urls()
        assert urls, f"{component.part_number} cites no source"
        assert all(url.startswith("https://") for url in urls)
        keys = {source.key for source in component.sources}
        for name, value in component.published.items():
            assert value.source_key in keys, f"{component.part_number}.{name}"
            assert value.verbatim, f"{component.part_number}.{name} has no source text"


def test_the_doublet_has_glass_on_both_sides_of_its_cemented_interface():
    """A cemented doublet with air inside it would trace and be a different lens."""
    doublet = PAC052.component
    assert doublet is not None
    assert doublet.surface_count == 3
    assert [m.kind.value for m in doublet.internal_materials] == ["catalog", "catalog"]
    assert [m.name for m in doublet.internal_materials] == ["N-BK7", "N-SF5"]  # type: ignore[union-attr]


# --- the refused component ---------------------------------------------------


def test_the_unpublished_objective_is_refused_and_carries_no_component():
    assert not M10X.supported
    assert M10X.component is None
    assert M10X.refusal is not None
    assert M10X.refusal.code == "CATALOG_PRESCRIPTION_NOT_PUBLISHED"
    # The refusal is keyed on missing GEOMETRY, not on a missing focal length --
    # the page publishes an EFL and an NA, which is exactly what would have made
    # fabricating a plausible model easy.
    assert "surface_radii" in M10X.refusal.missing_parameters
    assert "element_count" in M10X.refusal.missing_parameters
    assert any("16.5" in entry for entry in M10X.refusal.published_but_insufficient)


def test_asking_for_the_refused_component_raises_its_own_refusal():
    with pytest.raises(PrescriptionError) as failure:
        require_supported("M-10X")
    assert failure.value.code == "CATALOG_PRESCRIPTION_NOT_PUBLISHED"
    # And the alternatives are named, so a caller is not left guessing.
    assert set(failure.value.supported) == {"KPX094", "KBX058", "PAC052"}


def test_an_unknown_part_number_is_refused_with_the_catalog_listed():
    with pytest.raises(PrescriptionError) as failure:
        resolve_catalog_component("LA1131")
    assert failure.value.code == "CATALOG_PART_UNKNOWN"


# --- broken twins: each guard must actually fire -----------------------------


def test_a_mistyped_radius_is_refused():
    """The likeliest way a fabricated optical parameter would enter this file.

    ``published`` says 51.680 mm and the model is built at 51.860 -- two digits
    transposed, three lines apart in the same literal, and a lens that traces
    perfectly.
    """
    assert KPX094.component is not None
    broken = KPX094.component.model_copy(
        update={
            "geometries": (
                SphericalGeometrySpec(radius_mm=51.860),
                KPX094.component.geometries[1],
            )
        }
    )
    with pytest.raises(PrescriptionError) as failure:
        _corrupt(KPX094, component=broken)
    assert failure.value.code == "CATALOG_TRANSCRIPTION_RADIUS_MISMATCH"


def test_a_mistyped_thickness_is_refused():
    assert PAC052.component is not None
    broken = PAC052.component.model_copy(
        update={"internal_thicknesses_mm": (5.0, 2.71)}  # published says 2.17
    )
    with pytest.raises(PrescriptionError) as failure:
        _corrupt(PAC052, component=broken)
    assert failure.value.code == "CATALOG_TRANSCRIPTION_THICKNESS_MISMATCH"


def test_a_substituted_glass_is_refused():
    """N-SF6 for N-SF5: a real Schott glass, a different achromat."""
    assert PAC052.component is not None
    broken = PAC052.component.model_copy(
        update={
            "internal_materials": (
                PAC052.component.internal_materials[0],
                CatalogMaterialSpec(name="N-SF6"),
            )
        }
    )
    with pytest.raises(PrescriptionError) as failure:
        _corrupt(PAC052, component=broken)
    assert failure.value.code == "CATALOG_TRANSCRIPTION_MATERIAL_MISMATCH"


def test_a_curved_surface_claiming_to_be_a_plane_is_refused():
    """KPX094's second face is flat; citing no published radius must mean flat."""
    assert KPX094.component is not None
    broken = KPX094.component.model_copy(
        update={
            "geometries": (
                KPX094.component.geometries[0],
                SphericalGeometrySpec(radius_mm=-500.0),
            )
        }
    )
    with pytest.raises(PrescriptionError) as failure:
        _corrupt(KPX094, component=broken)
    assert failure.value.code == "CATALOG_TRANSCRIPTION_PLANE_EXPECTED"


def test_dropping_the_transcription_keys_is_refused_rather_than_skipping_the_check():
    """The guard must not be disableable by simply not declaring its inputs."""
    with pytest.raises(PrescriptionError) as radii:
        _corrupt(KPX094, radius_keys=())
    assert radii.value.code == "CATALOG_TRANSCRIPTION_RADIUS_KEYS_MISSING"

    with pytest.raises(PrescriptionError) as thicknesses:
        _corrupt(KPX094, thickness_keys=())
    assert thicknesses.value.code == "CATALOG_TRANSCRIPTION_THICKNESS_KEYS_MISSING"


def test_a_component_cannot_be_both_built_and_refused_or_neither():
    with pytest.raises(PrescriptionError) as both:
        _corrupt(KPX094, refusal=M10X.refusal.model_dump())  # type: ignore[union-attr]
    assert both.value.code == "CATALOG_COMPONENT_STATE_AMBIGUOUS"

    with pytest.raises(PrescriptionError) as neither:
        _corrupt(KPX094, component=None)
    assert neither.value.code == "CATALOG_COMPONENT_STATE_AMBIGUOUS"


def test_a_published_value_citing_an_unknown_source_is_refused():
    assert KPX094.component is not None
    published = {
        **{key: value.model_dump() for key, value in KPX094.published.items()},
    }
    published["radius_1_mm"] = {
        **published["radius_1_mm"],
        "source_key": "some_blog_post",
    }
    with pytest.raises(PrescriptionError) as failure:
        _corrupt(KPX094, published=published)
    assert failure.value.code == "CATALOG_PUBLISHED_VALUE_UNSOURCED"


# --- the system definitions --------------------------------------------------


def test_the_benchmark_declares_a_multi_component_system_and_its_control():
    from benchmark_systems import SystemRole, benchmark_systems

    systems = benchmark_systems()
    multi = [s for s in systems if len(s.placements) > 1]
    assert multi, "the benchmark declares no multi-component system"
    controls = [s for s in systems if s.role is SystemRole.NEGATIVE_CONTROL]
    assert controls, "the benchmark declares no negative control"
    for control in controls:
        assert control.control_of in {s.key for s in systems}


def test_the_control_differs_from_its_case_only_in_orientation():
    """A control that also changed the spacing would prove nothing about orientation."""
    from benchmark_systems import resolve_system

    case = resolve_system("S4_PAC052_KBX058_TANDEM")
    control = resolve_system("S5_TANDEM_REVERSED_ACHROMAT")
    assert control.control_of == case.key
    assert case.field_angles_deg == control.field_angles_deg
    assert case.wavelengths_um == control.wavelengths_um
    assert case.primary_wavelength_um == control.primary_wavelength_um
    assert case.image_plane_rule is control.image_plane_rule
    assert case.pupil_rings.value == control.pupil_rings.value
    assert (
        case.entrance_pupil_diameter.value == control.entrance_pupil_diameter.value
    )
    assert case.stop_component_index == control.stop_component_index
    assert case.stop_surface_index == control.stop_surface_index
    assert case.part_numbers == control.part_numbers
    assert [p.air_gap_after_mm for p in case.placements] == [
        p.air_gap_after_mm for p in control.placements
    ]
    # ...and the one thing that does differ.
    assert [p.orientation for p in case.placements] != [
        p.orientation for p in control.placements
    ]


def test_normalized_field_maps_the_declared_angles_onto_optiland_coordinates():
    """``Hy`` is angle / max(angle); an off-by-one here would trace a wrong field."""
    from benchmark_systems import resolve_system

    system = resolve_system("S4_PAC052_KBX058_TANDEM")
    assert system.field_angles_deg == (0.0, 1.0, 2.0, 3.0)
    assert system.normalized_field(0.0) == 0.0
    assert system.normalized_field(3.0) == 1.0
    assert system.normalized_field(1.0) == pytest.approx(1.0 / 3.0)
    with pytest.raises(PrescriptionError) as failure:
        system.normalized_field(2.5)
    assert failure.value.code == "SYSTEM_FIELD_NOT_DECLARED"


def test_every_system_parameter_states_a_source_and_a_basis():
    from benchmark_systems import benchmark_systems

    for system in benchmark_systems():
        assert system.entrance_pupil_diameter.basis
        assert system.pupil_rings.basis
        assert system.field_basis
        assert system.wavelength_basis
        for placement in system.placements[:-1]:
            # Every interior spacing is declared, sourced and justified.
            assert placement.air_gap_after_mm is not None
            assert placement.air_gap_source is not None
            assert placement.air_gap_basis
        # The image distance is never declared as a placement gap.
        assert system.placements[-1].air_gap_after_mm is None
        for comparison in system.catalog_comparisons:
            assert comparison.basis
            assert comparison.catalog_source_url.startswith("https://")


def test_a_catalog_comparison_value_comes_from_the_catalog_not_a_retyped_literal():
    """The comparison reference and the provenance record must be one value."""
    from benchmark_systems import resolve_system

    for key, part in (
        ("S1_KPX094_SINGLET", "KPX094"),
        ("S2_PAC052_ACHROMAT", "PAC052"),
        ("S3_KBX058_BICONVEX", "KBX058"),
    ):
        system = resolve_system(key)
        component = resolve_catalog_component(part)
        for comparison in system.catalog_comparisons:
            assert comparison.catalog_value == component.number(comparison.quantity)
        assert system.entrance_pupil_diameter.value == component.number(
            "entrance_pupil_diameter_mm"
        )


def test_assembling_a_system_puts_the_declared_gap_on_the_declared_surface():
    """The one place the benchmark's assembly meets the canonical prescription."""
    from benchmark_systems import resolve_system

    from core.optical_assembly import surface_table

    system = resolve_system("S4_PAC052_KBX058_TANDEM")
    spec = system.assemble(26.0)
    rows = surface_table(spec)
    assert len(rows) == 5
    # PAC052's three surfaces, then the 50 mm declared gap, then KBX058's two.
    assert [row["radius_mm"] for row in rows] == [
        60.741, -44.710, -133.104, 77.265, -77.265,
    ]
    assert rows[2]["thickness_mm"] == 50.0
    assert rows[4]["thickness_mm"] == 26.0
    assert rows[3]["vertex_z_mm"] == pytest.approx(7.17 + 50.0)
    assert rows[0]["is_stop"] and sum(row["is_stop"] for row in rows) == 1


def test_reversing_the_achromat_reverses_the_assembled_surfaces_and_the_glasses():
    from benchmark_systems import resolve_system

    from core.optical_assembly import surface_table

    control = resolve_system("S5_TANDEM_REVERSED_ACHROMAT")
    rows = surface_table(control.assemble(27.0))
    # Radii negated and reordered: the flint face now meets the light first.
    assert [row["radius_mm"] for row in rows[:3]] == [133.104, 44.710, -60.741]
    # The glasses travel with their elements: flint first, then crown.
    assert [row["material_name"] for row in rows[:3]] == ["N-SF5", "N-BK7", None]
    # The flint element is 2.17 mm and now comes first.
    assert rows[0]["thickness_mm"] == 2.17
    assert rows[1]["thickness_mm"] == 5.0
    # Same mechanical envelope, so KBX058 sits exactly where S4 puts it.
    assert rows[3]["vertex_z_mm"] == pytest.approx(7.17 + 50.0)


def test_no_surface_comment_states_something_a_reversal_would_falsify():
    """A comment travels with its surface, so it may not claim a radius or a side.

    The first version of this benchmark carried "R=+60.741 (faces the infinite
    conjugate)" on the achromat's crown face, and the reversed control's record
    duly showed that text beside a surface built at -60.741 facing the image.
    """
    for component in SUPPORTED:
        assert component.component is not None
        # Part numbers legitimately contain digits, so the check is on the two
        # things a reversal actually changes: a stated radius and a stated side.
        radii = {
            str(abs(geometry.resolved_radius_mm))
            for geometry in component.component.geometries
            if hasattr(geometry, "resolved_radius_mm")
        }
        for comment in component.component.surface_comments:
            assert "R=" not in comment, comment
            assert "infinite conjugate" not in comment, comment
            assert "faces the" not in comment, comment
            for radius in radii:
                assert radius not in comment, comment
