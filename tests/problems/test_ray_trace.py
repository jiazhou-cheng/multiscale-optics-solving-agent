"""The neutral ray-trace schema: what it accepts, what it refuses, and what it is not.

CHE-156 (R04). Acceptance criteria 4 (the schema names physics and carries no
Optiland API concept) and 5 (two public classes) are asserted here; criteria 1-3
are in `test_fixtures.py`, which needs the fixtures to say anything about them.

The refusal tests are the substance. A prescription that is *invalid* fails
loudly at construction; a prescription that is quietly *different* traces
successfully and produces numbers nobody can attribute. Every check below turns a
would-be second kind into the first.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from problems import ray_trace
from problems.ray_trace import MATERIAL_KINDS, UNITS, RayTraceProblem, SurfaceSpec

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


def a_problem(**overrides: object) -> RayTraceProblem:
    fields: dict[str, object] = {
        "name": "TwoSurfaceTest",
        "surfaces": (
            SurfaceSpec(radius_mm=10.0, thickness_mm=2.0,
                        material={"kind": "ideal", "refractive_index": 1.5}),
            SurfaceSpec(thickness_mm=20.0),
        ),
        "stop_index": 0,
        "entrance_pupil_diameter_mm": 5.0,
        "field_angles_deg": ((0.0, 0.0),),
        "wavelengths_um": (0.55,),
    }
    fields.update(overrides)
    return RayTraceProblem(**fields)  # type: ignore[arg-type]


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
    # Checked against the *code*, with docstrings and comments stripped: the
    # module docstring has to be able to name a ray tracer in order to say the
    # schema does not depend on one.
    assert concept not in _code_of(SRC / "problems" / "ray_trace.py"), (
        f"{concept!r} appears in the schema body. A problem is physical intent; how a "
        "particular ray tracer spells a surface belongs to that solver's adapter."
    )


def test_the_declared_units_are_exported_and_complete() -> None:
    assert UNITS == {
        "radius": "mm",
        "curvature": "1/mm",
        "thickness": "mm",
        "object_distance": "mm",
        "entrance_pupil_diameter": "mm",
        "wavelength": "um",
        "field_angle": "deg",
    }


def test_the_field_names_are_physical() -> None:
    """Surfaces, materials, aperture, source, field, wavelength -- all named."""
    problem_fields = {f.name for f in dataclasses.fields(RayTraceProblem)}
    assert {
        "surfaces",
        "entrance_pupil_diameter_mm",
        "field_angles_deg",
        "wavelengths_um",
        "object_distance_mm",
        "stop_index",
    } <= problem_fields
    surface_fields = {f.name for f in dataclasses.fields(SurfaceSpec)}
    assert {"radius_mm", "curvature_per_mm", "conic", "thickness_mm", "material"} <= surface_fields


# ---------------------------------------------------------------------------
# Criterion 5 -- two public classes
# ---------------------------------------------------------------------------


def test_there_are_two_public_classes_and_no_geometry_hierarchy() -> None:
    public_classes = sorted(
        name
        for name in ray_trace.__all__
        if isinstance(getattr(ray_trace, name), type) and name != "Material"
    )
    assert public_classes == ["RayTraceProblem", "SurfaceSpec"]


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
# The problem
# ---------------------------------------------------------------------------


def test_a_valid_problem_exposes_its_derived_reads() -> None:
    problem = a_problem(wavelengths_um=(0.4861, 0.5876), primary_wavelength_index=1)
    assert problem.primary_wavelength_um == 0.5876
    assert problem.object_at_infinity
    assert a_problem(object_distance_mm=100.0).object_at_infinity is False


def test_the_problem_is_frozen() -> None:
    problem = a_problem()
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.name = "other"  # type: ignore[misc]


def test_the_stop_is_an_index_so_no_stop_and_two_stops_are_unrepresentable() -> None:
    """The reference schema validated `is_stop` flags; one index cannot go wrong."""
    assert not any(hasattr(s, "is_stop") for s in a_problem().surfaces)
    with pytest.raises(ValueError, match="stop_index"):
        a_problem(stop_index=5)
    with pytest.raises(ValueError, match="stop_index"):
        a_problem(stop_index=-1)


def test_an_empty_system_is_refused() -> None:
    with pytest.raises(ValueError, match="surfaces"):
        a_problem(surfaces=())


def test_an_empty_source_is_refused() -> None:
    with pytest.raises(ValueError, match="field_angles_deg"):
        a_problem(field_angles_deg=())
    with pytest.raises(ValueError, match="wavelengths_um"):
        a_problem(wavelengths_um=())


def test_a_field_that_is_not_an_angle_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="field_angles_deg"):
        a_problem(field_angles_deg=((0.0,),))
    with pytest.raises(ValueError, match="field_angles_deg"):
        a_problem(field_angles_deg=((0.0, math.nan),))


@pytest.mark.parametrize("wavelength", [0.0, -0.55, math.nan, math.inf])
def test_an_unphysical_wavelength_is_refused(wavelength: float) -> None:
    with pytest.raises(ValueError, match="wavelengths_um"):
        a_problem(wavelengths_um=(wavelength,))


def test_a_primary_wavelength_index_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="primary_wavelength_index"):
        a_problem(wavelengths_um=(0.55,), primary_wavelength_index=1)


@pytest.mark.parametrize("epd", [0.0, -1.0, math.nan, math.inf])
def test_an_unphysical_aperture_is_refused(epd: float) -> None:
    with pytest.raises(ValueError, match="entrance_pupil_diameter_mm"):
        a_problem(entrance_pupil_diameter_mm=epd)


@pytest.mark.parametrize("distance", [-1.0, math.nan, math.inf])
def test_an_unphysical_object_distance_is_refused(distance: float) -> None:
    with pytest.raises(ValueError, match="object_distance_mm"):
        a_problem(object_distance_mm=distance)


def test_an_unnamed_problem_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        a_problem(name="   ")


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as caught:
        a_problem(name="", stop_index=9, wavelengths_um=())
    message = str(caught.value)
    assert "`name`" in message
    assert "stop_index" in message
    assert "wavelengths_um" in message
