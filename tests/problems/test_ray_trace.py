"""The neutral ray-trace schema: what it accepts, what it refuses, and what it is not.

CHE-156 (R04), split by CHE-218 (R05.7). Acceptance criteria 4 (the schema names
physics and carries no Optiland API concept) and 5 (a small, fixed set of public
classes) are asserted here; criteria 1-3 are in `test_fixtures.py`, which needs
the fixtures to say anything about them.

The refusal tests are the substance. A prescription that is *invalid* fails
loudly at construction; a prescription that is quietly *different* traces
successfully and produces numbers nobody can attribute. Every check below turns a
would-be second kind into the first.

What R05.7 changed here, and what it did not
--------------------------------------------
`RayTraceProblem` became `OpticalSetup` plus `SourceSpec`. Every validator that
still applies is kept and re-pointed at whichever record now owns the field: the
`UNITS` table, material key refusal, radius/curvature mutual exclusion,
stop-index bounds, aspheric-coefficient finiteness, the aperture, the object
distance. Two validators are **gone with the fields they guarded** -- the
field-angle *list* and the wavelength *list*, each with its index bound -- and
`test_neither_record_carries_a_list_of_fields_or_wavelengths` is what keeps them
from coming back: one solve is one field and one wavelength.
"""


from __future__ import annotations

import ast
import dataclasses
import math
import re
from pathlib import Path

import pytest

from problems import ray_trace
from problems.ray_trace import (
    MATERIAL_KINDS,
    UNITS,
    OpticalSetup,
    SourceSpec,
    SurfaceSpec,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def _code_of(path: Path) -> str:
    """The module's source with every docstring and comment removed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def a_setup(**overrides: object) -> OpticalSetup:
    fields: dict[str, object] = {
        "name": "TwoSurfaceTest",
        "surfaces": (
            SurfaceSpec(radius_mm=10.0, thickness_mm=2.0,
                        material={"kind": "ideal", "refractive_index": 1.5}),
            SurfaceSpec(thickness_mm=20.0),
        ),
        "stop_index": 0,
        "entrance_pupil_diameter_mm": 5.0,
        "reference_wavelength_um": 0.55,
    }
    fields.update(overrides)
    return OpticalSetup(**fields)  # type: ignore[arg-type]


def a_source(**overrides: object) -> SourceSpec:
    fields: dict[str, object] = {"wavelength_um": 0.55}
    fields.update(overrides)
    return SourceSpec(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Criterion 4 -- physics, and no solver concept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "concept",
    [
        "optiland",
        "Optic",
        "surface_group",
        "add_surface",
        "SurfaceFactory",
        "GeometryFactory",
        "set_aperture",
        "set_field_type",
        "BaseBackend",
        "samples.objectives",
        # A normalized field coordinate (Hx, Hy) is Optiland's trace API, not a
        # physical field angle. The schema states degrees.
        "Hx",
        "Hy",
    ],
)
def test_the_schema_names_no_solver_concept(concept: str) -> None:
    """No Optiland API concept in the schema body.

    Matched on **word boundaries** rather than as a bare substring, which CHE-218
    forced and which makes the check stricter rather than looser: `OpticalSetup`
    contains the substring `Optic`, and a rule that flagged the project's own
    neutral noun for containing a solver type's name would have to be answered by
    renaming the noun. `\bOptic\b` still catches `Optic()`, `optic.surfaces` and
    `from optiland.optic import Optic`, which is every way the concept could
    actually arrive. `test_the_solver_concept_detector_still_fires` is the
    meta-test.
    """
    # Checked against the *code*, with docstrings and comments stripped: the
    # module docstring has to be able to name a ray tracer in order to say the
    # schema does not depend on one.
    pattern = re.compile(rf"\b{re.escape(concept)}\b")
    assert not pattern.search(_code_of(SRC / "problems" / "ray_trace.py")), (
        f"{concept!r} appears in the schema body. A problem is physical intent; how a "
        "particular ray tracer spells a surface belongs to that solver's adapter."
    )


def test_the_solver_concept_detector_still_fires() -> None:
    """The meta-test for the word-boundary change: it catches, and does not overreach."""
    for source, concept, caught in (
        ("from optiland.optic import Optic\n", "Optic", True),
        ("lens = Optic(name='x')\n", "Optic", True),
        ("optic.surfaces.add(index=0)\n", "add_surface", False),
        ("optic.surfaces.add_surface(1)\n", "add_surface", True),
        ("setup = OpticalSetup(name='x')\n", "Optic", False),
        ("traced = lens.trace(Hx=0.0)\n", "Hx", True),
    ):
        pattern = re.compile(rf"\b{re.escape(concept)}\b")
        assert bool(pattern.search(source)) is caught, (source, concept)


def test_the_declared_units_are_exported_and_complete() -> None:
    assert UNITS == {
        "radius": "mm",
        "curvature": "1/mm",
        "thickness": "mm",
        "object_distance": "mm",
        "entrance_pupil_diameter": "mm",
        "wavelength": "um",
        "field_angle": "deg",
        # CHE-207. The one entry whose unit is index-dependent, which is why the
        # field carries no unit suffix in its name.
        "aspheric_coefficient": "mm**(1 - 2*(i+1)) for aspheric_coefficients[i]",
    }


def test_the_field_names_are_physical() -> None:
    """Surfaces, materials, aperture, source, field, wavelength -- all named."""
    setup_fields = {f.name for f in dataclasses.fields(OpticalSetup)}
    assert {
        "surfaces",
        "entrance_pupil_diameter_mm",
        "stop_index",
        "reference_wavelength_um",
    } <= setup_fields
    source_fields = {f.name for f in dataclasses.fields(SourceSpec)}
    assert {"wavelength_um", "field_angle_deg", "object_distance_mm"} <= source_fields
    surface_fields = {f.name for f in dataclasses.fields(SurfaceSpec)}
    assert {"radius_mm", "curvature_per_mm", "conic", "thickness_mm", "material"} <= surface_fields


# ---------------------------------------------------------------------------
# CHE-218 -- the split, as an independence claim
# ---------------------------------------------------------------------------


def test_a_setup_is_constructible_with_no_illumination_parameter() -> None:
    """Acceptance criterion 1's schema half, and criterion 4's.

    A setup takes no field angle, no object distance and no wavelength to trace
    at. `reference_wavelength_um` is the one wavelength-shaped field and is not
    illumination: it is what this setup's own exit pupil is located at, which
    CHE-218 measured to be a real property of the constructed system. The
    assertion that it is not illumination is the *absence* of the other three.
    """
    setup = a_setup()
    names = {f.name for f in dataclasses.fields(OpticalSetup)}
    assert "field_angle_deg" not in names
    assert "field_angles_deg" not in names
    assert "object_distance_mm" not in names
    assert "wavelength_um" not in names
    assert not hasattr(setup, "object_at_infinity")
    assert setup.reference_wavelength_um == 0.55


def test_a_source_is_constructible_with_no_system_parameter() -> None:
    """Criterion 4's other half: a source needs nothing that belongs to a setup."""
    names = {f.name for f in dataclasses.fields(SourceSpec)}
    for system_field in ("surfaces", "stop_index", "entrance_pupil_diameter_mm", "name"):
        assert system_field not in names
    source = a_source()
    assert source.field_angle_deg == (0.0, 0.0)
    assert source.object_at_infinity is True


def test_each_record_validates_its_own_fields_independently() -> None:
    """Criterion 4: neither validator reaches into the other record.

    A setup with an unusable stop is refused whatever source it might be paired
    with, and a source with an unusable wavelength is refused whatever setup it
    might be paired with. Nothing here constructs the other half.
    """
    with pytest.raises(ValueError, match="optical setup"):
        a_setup(stop_index=9)
    with pytest.raises(ValueError, match="source is not usable"):
        a_source(wavelength_um=-1.0)


def test_neither_record_carries_a_list_of_fields_or_wavelengths() -> None:
    """Acceptance criterion 5, and the guard on the two dropped capabilities.

    One solver invocation is one field and one wavelength. The old record carried
    `field_angles_deg` and `wavelengths_um` plus a `primary_wavelength_index`, and
    the extra entries never reached a trace: the field set existed only to give the
    backend a `max_field` and the wavelength set only to select the primary. A
    plural field here would put a loop back where there is no loop.
    """
    for record in (OpticalSetup, SourceSpec):
        names = {f.name for f in dataclasses.fields(record)}
        assert "field_angles_deg" not in names
        assert "wavelengths_um" not in names
        assert "primary_wavelength_index" not in names
    assert not hasattr(OpticalSetup, "primary_wavelength_um")


def test_a_source_declaring_two_wavelengths_is_refused_not_looped_over() -> None:
    """Acceptance criterion 5. A spectrum is several sources, not one field.

    `wavelength_um` is a scalar, so this is not merely "a tuple fails to be a
    float" -- the refusal names why, because a caller handed a sequence is a
    caller who expected it to be traced at all of them.
    """
    with pytest.raises(ValueError, match="SINGLE-WAVELENGTH"):
        a_source(wavelength_um=(0.4861, 0.6563))
    with pytest.raises(ValueError, match="SINGLE-WAVELENGTH"):
        a_source(wavelength_um=[0.55, 0.65])


def test_a_source_declaring_several_fields_is_refused() -> None:
    """The same rule on the other axis: one field per solve."""
    with pytest.raises(ValueError, match="field_angle_deg"):
        a_source(field_angle_deg=((0.0, 0.0), (0.0, 6.0)))
    with pytest.raises(ValueError, match="field_angle_deg"):
        a_source(field_angle_deg=(0.0,))
    with pytest.raises(ValueError, match="field_angle_deg"):
        a_source(field_angle_deg=(0.0, math.nan))


def test_the_ray_trace_problem_is_gone_not_aliased() -> None:
    """Acceptance criterion 6, at the module that used to export it.

    Not deprecated, not aliased: the tree is pre-cutover and a compatibility
    wrapper is what the clean-slate rule in `AGENTS.md` bans.
    """
    assert not hasattr(ray_trace, "RayTraceProblem")
    assert "RayTraceProblem" not in ray_trace.__all__


# ---------------------------------------------------------------------------
# Criterion 5 -- two public classes
# ---------------------------------------------------------------------------


def test_there_are_three_public_classes_and_no_geometry_hierarchy() -> None:
    """Was two through CHE-217; R05.7's split is the +1, and it is the whole +1."""
    public_classes = sorted(
        name
        for name in ray_trace.__all__
        if isinstance(getattr(ray_trace, name), type) and name != "Material"
    )
    assert public_classes == ["OpticalSetup", "SourceSpec", "SurfaceSpec"]


@pytest.mark.parametrize(
    "absent",
    [
        # The 20 classes of core/optical_system.py, and the assembly layer.
        "PlaneGeometrySpec",
        "SphericalGeometrySpec",
        "EvenAsphereGeometrySpec",
        "RefractiveInteractionSpec",
        "GratingInteractionSpec",
        "AirMaterialSpec",
        "IdealMaterialSpec",
        "CatalogMaterialSpec",
        "ApertureSpec",
        "FieldSpec",
        "WavelengthSpec",
        "OpticalSystemSpec",
        "GeometryKind",
        "InteractionKind",
        "MaterialKind",
        "ApertureKind",
        "FieldKind",
        "PrescriptionError",
        "ComponentSpec",
        "ComponentPlacement",
        "Orientation",
        # And every builder.
        "OpticalSystemBuilder",
        "SystemBuilder",
        "PrescriptionBuilder",
    ],
)
def test_the_avoided_names_do_not_exist(absent: str) -> None:
    assert not hasattr(ray_trace, absent)


# ---------------------------------------------------------------------------
# Geometry: three kinds as field values
# ---------------------------------------------------------------------------


def test_a_surface_with_no_radius_is_a_plane() -> None:
    surface = SurfaceSpec(thickness_mm=1.0)
    assert surface.is_plane
    assert surface.resolved_radius_mm == math.inf


def test_radius_and_curvature_resolve_to_the_same_surface() -> None:
    """The reference implementation's resolution rule, reused unchanged."""
    by_radius = SurfaceSpec(radius_mm=4.0, thickness_mm=1.0)
    by_curvature = SurfaceSpec(curvature_per_mm=0.25, thickness_mm=1.0)
    assert by_radius.resolved_radius_mm == by_curvature.resolved_radius_mm == 4.0
    assert not by_radius.is_plane


def test_giving_both_forms_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SurfaceSpec(radius_mm=4.0, curvature_per_mm=0.25, thickness_mm=1.0)


@pytest.mark.parametrize("radius", [0.0, math.inf, math.nan])
def test_a_radius_that_is_not_a_curved_surface_is_refused(radius: float) -> None:
    with pytest.raises(ValueError, match="curved surface"):
        SurfaceSpec(radius_mm=radius, thickness_mm=1.0)


@pytest.mark.parametrize("curvature", [0.0, math.inf, math.nan])
def test_a_curvature_that_is_not_a_curved_surface_is_refused(curvature: float) -> None:
    with pytest.raises(ValueError, match="curved surface"):
        SurfaceSpec(curvature_per_mm=curvature, thickness_mm=1.0)


def test_a_conic_is_a_field_not_a_class() -> None:
    assert SurfaceSpec(radius_mm=10.0, thickness_mm=1.0).conic == 0.0
    assert SurfaceSpec(radius_mm=10.0, thickness_mm=1.0, conic=-1.0).conic == -1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_thickness_or_conic_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="thickness_mm"):
        SurfaceSpec(thickness_mm=bad)
    with pytest.raises(ValueError, match="conic"):
        SurfaceSpec(thickness_mm=1.0, conic=bad)


# ---------------------------------------------------------------------------
# Even aspheric coefficients (CHE-207)
# ---------------------------------------------------------------------------


def test_the_default_surface_carries_no_aspheric_terms() -> None:
    """Absent is the default, so an existing prescription means what it meant."""
    surface = SurfaceSpec(radius_mm=10.0, thickness_mm=1.0)
    assert surface.aspheric_coefficients == ()
    assert surface.has_aspheric_terms is False


def test_an_all_zero_polynomial_is_still_a_plain_conic() -> None:
    """The invariant the solver's surface-type selection turns on.

    An empty tuple and a tuple of zeros describe the same surface, so both must
    answer `False` -- otherwise a prescription padded with zeros would silently
    build a different geometry class than the same prescription without them.
    """
    for coefficients in ((), (0.0,), (0.0, 0.0, 0.0), (0.0, -0.0)):
        surface = SurfaceSpec(
            radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=coefficients
        )
        assert surface.has_aspheric_terms is False, coefficients
    # ...and one non-zero anywhere in the series is enough, including trailing.
    assert SurfaceSpec(
        radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=(0.0, 0.0, 4e-7)
    ).has_aspheric_terms is True


def test_aspheric_coefficients_are_normalized_to_a_tuple_of_floats() -> None:
    """A list is what a transcribed prescription looks like; a frozen surface may
    not hold one, or the surface is mutable through one of its fields."""
    surface = SurfaceSpec(
        radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=[1e-3, -2, 0]
    )
    assert surface.aspheric_coefficients == (1e-3, -2.0, 0.0)
    assert isinstance(surface.aspheric_coefficients, tuple)
    assert all(isinstance(value, float) for value in surface.aspheric_coefficients)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_aspheric_coefficient_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match=r"aspheric_coefficients\[1\]"):
        SurfaceSpec(radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=(1e-3, bad))


def test_something_that_is_not_a_sequence_of_numbers_is_refused() -> None:
    with pytest.raises(ValueError, match="aspheric_coefficients"):
        SurfaceSpec(radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=4e-7)  # type: ignore[arg-type]


def test_an_aspheric_surface_is_frozen_like_every_other() -> None:
    surface = SurfaceSpec(radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=(1e-3,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        surface.aspheric_coefficients = (2e-3,)  # type: ignore[misc]


def test_an_aspheric_plate_has_a_planar_base_and_is_not_a_plane() -> None:
    """The representation invariant a single `is_plane` would have got wrong.

    A planar-base asphere is a legitimate aspheric plate -- its sag is the
    polynomial alone -- so the two questions "is the conic term flat" and "is this
    surface flat" have different answers and need different names. The solver reads
    the first to choose a base radius and would build a *plane* if it read the
    second.
    """
    plate = SurfaceSpec(thickness_mm=1.0, aspheric_coefficients=(1e-3, -2.5e-5))
    assert plate.has_planar_base is True
    assert plate.has_aspheric_terms is True
    assert plate.is_plane is False
    assert plate.resolved_radius_mm == math.inf

    flat = SurfaceSpec(thickness_mm=1.0)
    assert flat.has_planar_base is True and flat.is_plane is True

    conic = SurfaceSpec(radius_mm=10.0, thickness_mm=1.0, aspheric_coefficients=(1e-3,))
    assert conic.has_planar_base is False and conic.is_plane is False


def test_the_series_convention_is_documented_where_it_can_be_got_wrong() -> None:
    """The `r**2`-start convention is the one thing a reader must not guess.

    Asserted on the docstring because that is the only place a caller transcribing
    `A4, A6, A8` from a paper will look, and passing them without the leading zero
    is a wrong surface that traces successfully.
    """
    assert SurfaceSpec.__doc__ is not None
    assert "starts at `r**2`" in SurfaceSpec.__doc__
    assert "(0.0, A4, A6, A8)" in SurfaceSpec.__doc__


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_the_default_medium_is_air() -> None:
    assert SurfaceSpec(thickness_mm=1.0).material == {"kind": "air"}
    assert MATERIAL_KINDS == ("air", "ideal", "catalog")


def test_an_ideal_and_a_catalog_material_are_accepted() -> None:
    ideal = SurfaceSpec(
        thickness_mm=1.0, material={"kind": "ideal", "refractive_index": 1.5168}
    )
    assert ideal.material["refractive_index"] == 1.5168
    glass = SurfaceSpec(
        thickness_mm=1.0,
        material={
            "kind": "catalog",
            "name": "N-SK10",
            "catalog": None,
            "expected_catalog_file": "glass/schott/N-SK10.yml",
        },
    )
    assert glass.material["name"] == "N-SK10"


def test_a_misspelled_material_key_is_refused() -> None:
    """The measured failure: an unrecognized key is silently filtered by the solver."""
    with pytest.raises(ValueError, match="does not take"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "ideal", "refactive_index": 1.5})


def test_a_material_missing_its_required_key_is_refused() -> None:
    with pytest.raises(ValueError, match="needs"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "ideal"})
    with pytest.raises(ValueError, match="needs"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "catalog"})


def test_an_unknown_material_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="material kind"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "grin"})  # type: ignore[typeddict-item]


def test_air_takes_no_index() -> None:
    with pytest.raises(ValueError, match="does not take"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "air", "refractive_index": 1.0})


@pytest.mark.parametrize("index", [0.0, -1.5, math.nan, math.inf])
def test_an_unphysical_ideal_index_is_refused(index: float) -> None:
    with pytest.raises(ValueError, match="refractive_index"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "ideal", "refractive_index": index})


def test_a_catalog_glass_needs_a_name() -> None:
    with pytest.raises(ValueError, match="glass name"):
        SurfaceSpec(thickness_mm=1.0, material={"kind": "catalog", "name": "  "})


# ---------------------------------------------------------------------------
# The setup
# ---------------------------------------------------------------------------


def test_the_setup_is_frozen() -> None:
    setup = a_setup()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setup.name = "other"  # type: ignore[misc]


def test_the_stop_is_an_index_so_no_stop_and_two_stops_are_unrepresentable() -> None:
    """The reference schema validated `is_stop` flags; one index cannot go wrong."""
    assert not any(hasattr(s, "is_stop") for s in a_setup().surfaces)
    with pytest.raises(ValueError, match="stop_index"):
        a_setup(stop_index=5)
    with pytest.raises(ValueError, match="stop_index"):
        a_setup(stop_index=-1)


def test_an_empty_system_is_refused() -> None:
    with pytest.raises(ValueError, match="surfaces"):
        a_setup(surfaces=())


@pytest.mark.parametrize("epd", [0.0, -1.0, math.nan, math.inf])
def test_an_unphysical_aperture_is_refused(epd: float) -> None:
    with pytest.raises(ValueError, match="entrance_pupil_diameter_mm"):
        a_setup(entrance_pupil_diameter_mm=epd)


@pytest.mark.parametrize("wavelength", [0.0, -0.55, math.nan, math.inf])
def test_an_unphysical_reference_wavelength_is_refused(wavelength: float) -> None:
    """It has no default, so an absent one is a TypeError and a bad one a refusal."""
    with pytest.raises(ValueError, match="reference_wavelength_um"):
        a_setup(reference_wavelength_um=wavelength)


def test_the_reference_wavelength_has_no_default() -> None:
    """A wavelength nobody chose would be an invented convention.

    It decides where the exit pupil is: CHE-218 measured `paraxial.XPL()` moving
    from -3.0545788978518327 mm to -3.0550180932891653 mm on M3-REVERSE-TELEPHOTO
    between 0.5876 and 0.55 um. A default would pick one of those silently.
    """
    with pytest.raises(TypeError):
        OpticalSetup(  # type: ignore[call-arg]
            name="no-reference",
            surfaces=(SurfaceSpec(thickness_mm=1.0),),
            entrance_pupil_diameter_mm=5.0,
            stop_index=0,
        )


def test_an_unnamed_setup_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        a_setup(name="   ")


def test_every_setup_problem_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as caught:
        a_setup(name="", stop_index=9, entrance_pupil_diameter_mm=0.0)
    message = str(caught.value)
    assert "`name`" in message
    assert "stop_index" in message
    assert "entrance_pupil_diameter_mm" in message


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------


def test_the_source_is_frozen() -> None:
    source = a_source()
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.wavelength_um = 0.65  # type: ignore[misc]


@pytest.mark.parametrize("wavelength", [0.0, -0.55, math.nan, math.inf])
def test_an_unphysical_source_wavelength_is_refused(wavelength: float) -> None:
    with pytest.raises(ValueError, match="wavelength_um"):
        a_source(wavelength_um=wavelength)


def test_a_source_wavelength_and_field_are_normalized_to_floats() -> None:
    """An integer degree is what a transcribed field looks like."""
    source = SourceSpec(wavelength_um=1, field_angle_deg=(0, 2))
    assert source.wavelength_um == 1.0 and isinstance(source.wavelength_um, float)
    assert source.field_angle_deg == (0.0, 2.0)
    assert all(isinstance(value, float) for value in source.field_angle_deg)


@pytest.mark.parametrize("distance", [-1.0, 0.0, math.nan, math.inf])
def test_an_unphysical_object_distance_is_refused(distance: float) -> None:
    """Zero joined the list with CHE-207.

    It used to be accepted, and it never meant anything: a point source at zero
    distance sits *on* the first surface, so there is no object space for it to
    radiate through. Refusing it is what makes "finite and positive" the whole of
    the finite-object contract, with no degenerate value inside it.
    """
    with pytest.raises(ValueError, match="object_distance_mm"):
        a_source(object_distance_mm=distance)


def test_a_finite_object_distance_is_a_point_source(recwarn: pytest.WarningsRecorder) -> None:
    """The two source geometries are one field, and both are reachable."""
    at_infinity = a_source(object_distance_mm=None)
    assert at_infinity.object_at_infinity is True
    assert at_infinity.object_distance_mm is None

    point = a_source(object_distance_mm=9.674922600619196)
    assert point.object_at_infinity is False
    assert point.object_distance_mm == 9.674922600619196
    assert not recwarn, "a finite conjugate is a supported case, not a warned-about one"


def test_the_point_source_position_convention_is_documented() -> None:
    """Where the source *is* -- asserted, because a field angle changing meaning
    between the two source geometries is exactly the thing a caller will assume
    rather than read.

    Measured to twelve digits by CHE-207 and re-verified against the solver by
    `tests/physics/test_optiland_finite_conjugate.py`; this asserts the schema says
    the same thing the solver does.
    """
    assert SourceSpec.__doc__ is not None
    assert "(-tan(x_deg) * d, -tan(y_deg) * d, -d)" in SourceSpec.__doc__
    assert "point source" in SourceSpec.__doc__


def test_every_source_problem_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as caught:
        a_source(wavelength_um=-1.0, object_distance_mm=0.0, field_angle_deg=(0.0,))
    message = str(caught.value)
    assert "wavelength_um" in message
    assert "field_angle_deg" in message
    assert "object_distance_mm" in message
