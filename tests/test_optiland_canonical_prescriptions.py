"""CHE-56 (PB5): the canonical optical-system schema and the generic builder.

Three things are under test, and they are deliberately separated:

1. **The schema** (``core/optical_system.py``) -- units, versioning,
   validation, and a canonical normalization strong enough to fingerprint.
   No solver is involved.
2. **The builder** (``adapters/optiland_builder.py``) -- one construction path
   for every Phase 3 feature, checked against closed-form geometry and the
   grating equation rather than against itself.
3. **The migration** -- ``M3SingletRef`` and ``ReverseTelephoto`` now come from
   canonical prescriptions, and must reproduce what they were before.

The migration checks use two *independent* oracles rather than the new code's
own output: ``optiland.samples.objectives.ReverseTelephoto`` for the bundled
system (structural equality of ``Optic.to_dict()`` plus element-wise trace
equality), and ``benchmarks/slice_protocol.yaml``'s frozen derived geometry for
the adapter-owned singlet. The bundled sample is kept for exactly this purpose:
it is no longer a construction path, it is an oracle.

Evidence for the two newly admitted construction paths (even asphere, grating)
is recorded in ``knowledge/solvers/optiland/expected/system_construction_probe.json``
by ``knowledge/solvers/optiland/probes/system_construction_probe.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import load_probe_expected

from adapters.base import ModelRunRequest, RunStatus
from adapters.optiland_adapter import OptilandAdapter
from core.errors import UnsupportedCapabilityError
from core.optical_system import (
    OPTICAL_SYSTEM_SPEC_VERSION,
    UNITS,
    AirMaterialSpec,
    ApertureSpec,
    CatalogMaterialSpec,
    EvenAsphereGeometrySpec,
    FieldSpec,
    GratingInteractionSpec,
    IdealMaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)
from registry.prescriptions import (
    M3_SINGLET_REF,
    PHASE3_FEATURE_DEMO,
    PRESCRIPTION_NAMES,
    REVERSE_TELEPHOTO,
    prescription_names,
    resolve_prescription,
)

optiland_backend = pytest.importorskip("optiland.backend")
build_optiland_system = pytest.importorskip(
    "adapters.optiland_builder"
).build_optiland_system
resolve_catalog_material = pytest.importorskip(
    "adapters.optiland_builder"
).resolve_catalog_material
OptilandMaterial = pytest.importorskip("optiland.materials").Material

pytestmark = pytest.mark.optiland

REPO_ROOT = Path(__file__).resolve().parents[1]
SLICE_PROTOCOL = REPO_ROOT / "benchmarks" / "slice_protocol.yaml"

WAVELENGTH_UM = 0.55
#: float64 round-off over a handful of accumulations. Every geometric identity
#: asserted below is exact arithmetic in principle, so this is a dtype budget.
ROUND_OFF_MM = 1e-12


def _minimal_spec(**overrides) -> OpticalSystemSpec:
    """A valid two-surface system, so a rejection test isolates one defect."""
    payload = {
        "name": "minimal",
        "surfaces": (
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=20.0),
                thickness_mm=3.0,
                material=IdealMaterialSpec(refractive_index=1.5),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=38.0),
        ),
        "aperture": ApertureSpec(value_mm=8.0),
        "fields": (FieldSpec(),),
        "wavelengths": (WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    }
    payload.update(overrides)
    return OpticalSystemSpec(**payload)


# ---------------------------------------------------------------------------
# 1. Schema: units, version, normalization
# ---------------------------------------------------------------------------


def test_declared_units_are_the_repository_conventions() -> None:
    """The units are part of the contract, so they are asserted, not narrated.

    Geometry in millimetres and wavelength in micrometres are CHE-12's findings.
    The grating period sharing the *wavelength's* unit rather than the
    geometry's, and the groove orientation being radians, are CHE-56's -- both
    measured in system_construction_probe.py, and both easy to get silently
    wrong by 1000x or by 57x respectively.
    """
    assert UNITS["radius"] == "mm"
    assert UNITS["thickness"] == "mm"
    assert UNITS["object_distance"] == "mm"
    assert UNITS["aperture_epd"] == "mm"
    assert UNITS["curvature"] == "1/mm"
    assert UNITS["wavelength"] == "um"
    assert UNITS["grating_period"] == "um"
    assert UNITS["groove_orientation"] == "rad"
    assert UNITS["angular_field"] == "deg"

    probe = load_probe_expected("optiland", "system_construction_probe")
    assert probe["units"]["grating_period"].startswith("micrometre")
    assert probe["units"]["groove_orientation_angle"].startswith("radian")


def test_schema_version_is_declared_and_enforced() -> None:
    spec = _minimal_spec()
    assert spec.spec_version == OPTICAL_SYSTEM_SPEC_VERSION
    assert spec.canonical_dict()["spec_version"] == OPTICAL_SYSTEM_SPEC_VERSION

    # A round trip through the serialized form is exact.
    assert OpticalSystemSpec.from_dict(spec.canonical_dict()) == spec

    # An unknown version is refused rather than reinterpreted under today's
    # field semantics -- the whole point of carrying the identifier.
    stale = spec.canonical_dict()
    stale["spec_version"] = "optical-system-spec/0"
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_SCHEMA_VERSION_UNSUPPORTED"):
        OpticalSystemSpec.from_dict(stale)

    missing = spec.canonical_dict()
    del missing["spec_version"]
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_SCHEMA_VERSION_UNSUPPORTED"):
        OpticalSystemSpec.from_dict(missing)


def test_canonical_serialization_is_deterministic_and_strict_json() -> None:
    spec = _minimal_spec()

    # Repeated normalization is byte-identical, and key order does not depend on
    # how the mapping was assembled.
    assert spec.canonical_json() == spec.canonical_json()
    assert list(spec.canonical_dict()) == sorted(spec.canonical_dict())
    reparsed = json.loads(spec.canonical_json())
    assert reparsed == spec.canonical_dict()

    # No non-finite float can reach the serialized form: a plane carries no
    # radius at all rather than an infinite one, which is why allow_nan=False
    # holds. (json.dumps would otherwise emit bare `Infinity`, which is not JSON.)
    assert "Infinity" not in spec.canonical_json()
    assert "NaN" not in spec.canonical_json()

    # Fingerprints: equal prescriptions agree, and a single changed number does not.
    assert spec.fingerprint() == _minimal_spec().fingerprint()
    moved = _minimal_spec(
        surfaces=(
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=20.000000001),
                thickness_mm=3.0,
                material=IdealMaterialSpec(refractive_index=1.5),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=38.0),
        )
    )
    assert moved.fingerprint() != spec.fingerprint()


def test_omitted_defaults_normalize_to_the_same_prescription() -> None:
    """An omitted default and an explicitly written one are the same system."""
    terse = SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=1.0)
    verbose = SurfaceSpec(
        geometry=PlaneGeometrySpec(),
        thickness_mm=1.0,
        material=AirMaterialSpec(),
        is_stop=False,
        comment="",
    )
    assert terse == verbose

    a = _minimal_spec(surfaces=(_minimal_spec().surfaces[0], terse))
    b = _minimal_spec(surfaces=(_minimal_spec().surfaces[0], verbose))
    assert a.fingerprint() == b.fingerprint()


def test_unknown_prescription_fields_are_rejected_not_ignored() -> None:
    """The solver silently discards kwargs it does not recognize; we must not.

    ``GeometryFactory.create`` filters ``**kwargs`` down to the fields of the
    config class it selected, so a misspelled or misplaced key produces a
    different optical system with no error at all (recorded in the probe). The
    schema's ``extra='forbid'`` is the only place that is caught.
    """
    probe = load_probe_expected("optiland", "system_construction_probe")
    silent = probe["cases"]["surface_kwargs_are_silently_filtered"]
    assert silent["unknown_kwarg_raised"] is False
    assert silent["geometry_has_coefficients_attribute"] is False

    with pytest.raises(Exception, match=r"extra_forbidden|Extra inputs"):
        SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=1.0, radius_of_curvature=0.05)
    with pytest.raises(Exception, match=r"extra_forbidden|Extra inputs"):
        SphericalGeometrySpec(radius_mm=10.0, coefficients=(1.0,))


# ---------------------------------------------------------------------------
# 2. Validation: every rejection is eager, coded, and specific
# ---------------------------------------------------------------------------


def _spherical_stop(**geometry) -> SurfaceSpec:
    return SurfaceSpec(geometry=SphericalGeometrySpec(**geometry), thickness_mm=1.0, is_stop=True)


@pytest.mark.parametrize(
    ("code", "make"),
    [
        (
            "PRESCRIPTION_NO_SURFACES",
            lambda: _minimal_spec(surfaces=()),
        ),
        (
            "PRESCRIPTION_STOP_AMBIGUOUS",
            lambda: _minimal_spec(
                surfaces=(SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=1.0),)
            ),
        ),
        (
            "PRESCRIPTION_STOP_AMBIGUOUS",
            lambda: _minimal_spec(surfaces=(_spherical_stop(radius_mm=10.0),) * 2),
        ),
        (
            "PRESCRIPTION_NO_FIELDS",
            lambda: _minimal_spec(fields=()),
        ),
        (
            "PRESCRIPTION_NO_WAVELENGTHS",
            lambda: _minimal_spec(wavelengths=()),
        ),
        (
            "PRESCRIPTION_PRIMARY_WAVELENGTH_AMBIGUOUS",
            lambda: _minimal_spec(wavelengths=(WavelengthSpec(value_um=0.55),)),
        ),
        (
            "PRESCRIPTION_PRIMARY_WAVELENGTH_AMBIGUOUS",
            lambda: _minimal_spec(
                wavelengths=(
                    WavelengthSpec(value_um=0.48, is_primary=True),
                    WavelengthSpec(value_um=0.65, is_primary=True),
                )
            ),
        ),
        (
            "PRESCRIPTION_OBJECT_DISTANCE_INVALID",
            lambda: _minimal_spec(object_distance_mm=-1.0),
        ),
        (
            "PRESCRIPTION_OBJECT_DISTANCE_INVALID",
            lambda: _minimal_spec(object_distance_mm=math.inf),
        ),
        (
            "PRESCRIPTION_THICKNESS_NOT_FINITE",
            lambda: SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=math.inf),
        ),
        # A curved surface must actually be curved: an infinite radius or a zero
        # curvature is a plane, and a plane has its own geometry kind, so the
        # ambiguity is refused instead of resolved.
        (
            "PRESCRIPTION_RADIUS_NOT_CURVED",
            lambda: SphericalGeometrySpec(radius_mm=math.inf),
        ),
        (
            "PRESCRIPTION_RADIUS_NOT_CURVED",
            lambda: SphericalGeometrySpec(radius_mm=0.0),
        ),
        (
            "PRESCRIPTION_CURVATURE_NOT_CURVED",
            lambda: SphericalGeometrySpec(curvature_per_mm=0.0),
        ),
        (
            "PRESCRIPTION_RADIUS_UNDERSPECIFIED",
            lambda: SphericalGeometrySpec(),
        ),
        (
            "PRESCRIPTION_RADIUS_UNDERSPECIFIED",
            lambda: SphericalGeometrySpec(radius_mm=10.0, curvature_per_mm=0.1),
        ),
        (
            "PRESCRIPTION_RADIUS_UNDERSPECIFIED",
            lambda: EvenAsphereGeometrySpec(coefficients=(1e-4,)),
        ),
        (
            "PRESCRIPTION_ASPHERE_COEFFICIENT_NOT_FINITE",
            lambda: EvenAsphereGeometrySpec(radius_mm=10.0, coefficients=(1e-4, math.nan)),
        ),
        (
            "PRESCRIPTION_GRATING_PERIOD_NOT_FINITE",
            lambda: GratingInteractionSpec(order=1, period_um=math.inf),
        ),
        (
            "PRESCRIPTION_GROOVE_ORIENTATION_NOT_FINITE",
            lambda: GratingInteractionSpec(order=1, period_um=2.0, groove_orientation_rad=math.nan),
        ),
        (
            "PRESCRIPTION_APERTURE_NOT_FINITE",
            lambda: ApertureSpec(value_mm=math.inf),
        ),
        (
            "PRESCRIPTION_WAVELENGTH_NOT_FINITE",
            lambda: WavelengthSpec(value_um=math.inf),
        ),
        (
            "PRESCRIPTION_FIELD_NOT_FINITE",
            lambda: FieldSpec(y_deg=math.nan),
        ),
        (
            "PRESCRIPTION_INDEX_NOT_FINITE",
            lambda: IdealMaterialSpec(refractive_index=math.inf),
        ),
        # A grating is a geometry class in the pinned solver, so it cannot be
        # ruled on an asphere -- and must not be silently downgraded to its base
        # conic, which would trace happily and be wrong.
        (
            "PRESCRIPTION_GRATING_GEOMETRY_UNSUPPORTED",
            lambda: SurfaceSpec(
                geometry=EvenAsphereGeometrySpec(radius_mm=10.0, coefficients=(1e-4,)),
                thickness_mm=1.0,
                interaction=GratingInteractionSpec(order=1, period_um=2.0),
            ),
        ),
    ],
)
def test_invalid_prescription_is_rejected_with_a_specific_code(code, make) -> None:
    with pytest.raises(PrescriptionError, match=code):
        make()


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"aperture": {"kind": "imageFNO", "value_mm": 4.0}}, "aperture type outside Phase 3"),
        ({"field_kind": "object_height"}, "field type outside Phase 3"),
    ],
)
def test_unsupported_aperture_or_field_type_is_rejected(payload, why) -> None:
    """Optiland supports these; this schema deliberately does not yet.

    Accepting a field or aperture type the builder has never been validated
    against would be a capability claim without evidence, so the closed
    ``Literal`` refuses it at parse time.
    """
    data = _minimal_spec().canonical_dict()
    data.update(payload)
    with pytest.raises(Exception, match=r"literal_error|Input should be"):
        OpticalSystemSpec.from_dict(data)


def test_unsupported_kinds_are_rejected_at_parse_time() -> None:
    data = _minimal_spec().canonical_dict()
    data["surfaces"][0]["geometry"]["kind"] = "toroidal"
    with pytest.raises(Exception, match=r"union_tag_invalid|does not match any"):
        OpticalSystemSpec.from_dict(data)

    data = _minimal_spec().canonical_dict()
    data["surfaces"][0]["material"]["kind"] = "abbe"
    with pytest.raises(Exception, match=r"union_tag_invalid|does not match any"):
        OpticalSystemSpec.from_dict(data)

    data = _minimal_spec().canonical_dict()
    data["surfaces"][0]["interaction"]["kind"] = "phase"
    with pytest.raises(Exception, match=r"union_tag_invalid|does not match any"):
        OpticalSystemSpec.from_dict(data)


def test_builder_refuses_something_that_is_not_a_spec() -> None:
    """A serialized mapping must go through from_dict, so its version is checked."""
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_NOT_A_SPEC"):
        build_optiland_system(_minimal_spec().canonical_dict())


# ---------------------------------------------------------------------------
# 3. Construction: one path, every Phase 3 feature, checked against geometry
# ---------------------------------------------------------------------------


def test_plane_and_spherical_surfaces_build_the_expected_geometry() -> None:
    spec = _minimal_spec()
    optic = build_optiland_system(spec)

    # Object surface, two optical surfaces, image surface -- in that order.
    kinds = [type(surface.geometry).__name__ for surface in optic.surfaces.surfaces]
    assert kinds == ["Plane", "StandardGeometry", "Plane", "Plane"]
    assert float(optic.surfaces.surfaces[1].geometry.radius) == 20.0

    # Axial placement follows the declared thicknesses.
    z = [float(np.asarray(s.geometry.cs.z)) for s in optic.surfaces.surfaces[1:]]
    assert z == pytest.approx([0.0, 3.0, 41.0], abs=ROUND_OFF_MM)


def test_radius_and_curvature_forms_describe_the_same_surface() -> None:
    """Both representations are offered; neither may mean something different."""
    by_radius = build_optiland_system(_minimal_spec())
    by_curvature = build_optiland_system(
        _minimal_spec(
            surfaces=(
                SurfaceSpec(
                    geometry=SphericalGeometrySpec(curvature_per_mm=1.0 / 20.0),
                    thickness_mm=3.0,
                    material=IdealMaterialSpec(refractive_index=1.5),
                    is_stop=True,
                ),
                SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=38.0),
            )
        )
    )
    assert float(by_curvature.surfaces.surfaces[1].geometry.radius) == pytest.approx(
        20.0, abs=ROUND_OFF_MM
    )
    for name in ("x", "y", "L", "M", "N", "opd"):
        first = np.asarray(
            getattr(by_radius.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8), name)
        )
        second = np.asarray(
            getattr(by_curvature.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8), name)
        )
        assert first.shape == second.shape
        assert float(np.max(np.abs(first - second))) <= ROUND_OFF_MM


def test_even_asphere_sag_matches_the_analytic_series() -> None:
    """Sag against an independent evaluation, plus the falsifier that matters.

    The interesting way to get an even asphere wrong is to start the series at
    ``r**4``, as several optical-design conventions do. The probe measures that
    error at 1.1e-2 mm on this surface, four orders of magnitude above the
    agreement asserted here, so this comparison can actually fail.
    """
    radius_mm, conic = 10.0, -0.5
    coefficients = (1.0e-3, -2.5e-5, 4.0e-7)
    spec = _minimal_spec(
        surfaces=(
            SurfaceSpec(
                geometry=EvenAsphereGeometrySpec(
                    radius_mm=radius_mm, conic=conic, coefficients=coefficients
                ),
                thickness_mm=2.0,
                material=IdealMaterialSpec(refractive_index=1.5),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=10.0),
        )
    )
    geometry = build_optiland_system(spec).surfaces.surfaces[1].geometry
    assert type(geometry).__name__ == "EvenAsphere"

    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    observed = np.asarray(geometry.sag(x=r, y=np.zeros_like(r)), dtype=float)

    r2 = r**2
    curvature = 1.0 / radius_mm
    analytic = curvature * r2 / (1.0 + np.sqrt(1.0 - (1.0 + conic) * curvature**2 * r2))
    for index, coefficient in enumerate(coefficients):
        analytic = analytic + coefficient * r2 ** (index + 1)
    assert float(np.max(np.abs(observed - analytic))) <= ROUND_OFF_MM

    probe = load_probe_expected("optiland", "system_construction_probe")
    case = probe["cases"]["even_asphere_sag_matches_analytic"]
    assert case["max_abs_error_mm"] <= ROUND_OFF_MM
    assert case["max_abs_error_if_series_started_at_r4_mm"] > 1e-3
    assert observed.tolist() == pytest.approx(case["observed_sag_mm"], abs=ROUND_OFF_MM)


def test_zero_coefficient_asphere_is_the_sphere_of_the_same_radius() -> None:
    radius_mm = 10.0
    spec = _minimal_spec(
        surfaces=(
            SurfaceSpec(
                geometry=EvenAsphereGeometrySpec(radius_mm=radius_mm, coefficients=()),
                thickness_mm=2.0,
                material=IdealMaterialSpec(refractive_index=1.5),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=10.0),
        )
    )
    geometry = build_optiland_system(spec).surfaces.surfaces[1].geometry
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    observed = np.asarray(geometry.sag(x=r, y=np.zeros_like(r)), dtype=float)
    analytic = radius_mm - np.sqrt(radius_mm**2 - r**2)
    assert float(np.max(np.abs(observed - analytic))) <= 1e-9


def _grating_spec(order: int, period_um: float, groove_rad: float = 0.0) -> OpticalSystemSpec:
    return OpticalSystemSpec(
        name=f"grating-m{order}-d{period_um}",
        surfaces=(
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                interaction=GratingInteractionSpec(
                    order=order, period_um=period_um, groove_orientation_rad=groove_rad
                ),
                thickness_mm=10.0,
                is_stop=True,
            ),
        ),
        aperture=ApertureSpec(value_mm=4.0),
        fields=(FieldSpec(),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )


@pytest.mark.parametrize("period_um", [2.0, 4.0, 8.0])
def test_grating_reproduces_the_grating_equation_in_micrometres(period_um) -> None:
    """``sin(theta_1) = lambda / d`` for a plane grating at normal incidence.

    Sweeping the period is what makes this a units test rather than a constant
    check: reading ``period_um`` as millimetres would put every deviation 1000x
    low, and no single measurement would reveal it.
    """
    optic = build_optiland_system(_grating_spec(order=1, period_um=period_um))
    assert type(optic.surfaces.surfaces[1].geometry).__name__ == "PlaneGrating"
    assert type(optic.surfaces.surfaces[1].interaction_model).__name__ == (
        "DiffractiveInteractionModel"
    )

    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=6)
    sin_theta = np.asarray(rays.M, dtype=float)
    analytic = WAVELENGTH_UM / period_um
    assert float(np.max(np.abs(sin_theta - analytic))) <= 1e-12
    assert float(np.max(np.abs(np.asarray(rays.L, dtype=float)))) <= 1e-12


def test_grating_order_and_groove_orientation_are_honoured() -> None:
    period_um = 2.0

    # Order 0 is undeviated; order -1 is the mirror of order +1.
    zeroth = build_optiland_system(_grating_spec(order=0, period_um=period_um)).trace(
        Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=6
    )
    assert float(np.max(np.abs(np.asarray(zeroth.M, dtype=float)))) <= 1e-12

    minus_one = build_optiland_system(_grating_spec(order=-1, period_um=period_um)).trace(
        Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=6
    )
    assert float(np.mean(np.asarray(minus_one.M, dtype=float))) == pytest.approx(
        -WAVELENGTH_UM / period_um, abs=1e-12
    )

    # The groove orientation is radians, and rotates the dispersion axis: at
    # pi/2 the deviation moves from y into x. Read as degrees, pi/2 would be
    # 1.57 degrees and the deviation would still be essentially in y.
    rotated = build_optiland_system(
        _grating_spec(order=1, period_um=period_um, groove_rad=math.pi / 2.0)
    ).trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=6)
    assert float(np.max(np.abs(np.asarray(rotated.M, dtype=float)))) <= 1e-12
    assert abs(float(np.mean(np.asarray(rotated.L, dtype=float)))) == pytest.approx(
        WAVELENGTH_UM / period_um, abs=1e-12
    )


def test_material_kinds_build_the_expected_optiland_materials() -> None:
    spec = _minimal_spec(
        surfaces=(
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=20.0),
                thickness_mm=3.0,
                material=CatalogMaterialSpec(
                    name="N-BK7", expected_catalog_file="glass/schott/N-BK7.yml"
                ),
                is_stop=True,
            ),
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=2.0,
                material=IdealMaterialSpec(refractive_index=1.5168),
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=30.0),
        )
    )
    optic = build_optiland_system(spec)
    surfaces = optic.surfaces.surfaces

    catalog = surfaces[1].material_post
    assert type(catalog).__name__ == "Material"
    assert catalog.name == "N-BK7"
    # A real dispersion, not a constant: the whole point of a catalog glass.
    assert float(catalog.n(0.486)) > float(catalog.n(0.656))

    ideal = surfaces[2].material_post
    assert type(ideal).__name__ == "IdealMaterial"
    assert float(ideal.n(0.486)) == pytest.approx(1.5168)
    assert float(ideal.n(0.656)) == pytest.approx(1.5168)

    air = surfaces[3].material_post
    assert type(air).__name__ == "IdealMaterial"
    assert float(air.n(WAVELENGTH_UM)) == pytest.approx(1.0)
    assert float(air.k(WAVELENGTH_UM)) == pytest.approx(0.0)


def test_catalog_resolution_is_pinned_not_merely_looked_up() -> None:
    """A bare glass name is a fuzzy query in the pinned solver, so it is guarded.

    Optiland filters the catalog by substring over three columns, ranks what
    survives by Levenshtein distance, and returns the best row. So ``SK15``
    legitimately selects HIKARI while ``N-SK10`` selects SCHOTT, and a name that
    matches nothing exactly still resolves -- to a different glass.
    """
    probe = load_probe_expected("optiland", "system_construction_probe")
    materials = probe["cases"]["catalog_names_resolve_to_one_exact_match"]["materials"]
    assert materials["SK15|None"]["substring_matches"] == 7
    assert materials["SK15|None"]["exact_matches"] == 1
    assert materials["SK15|None"]["resolved_catalog_file"] == "glass/hikari/SK15.yml"
    assert materials["N-SK10|None"]["resolved_catalog_file"] == "glass/schott/N-SK10.yml"

    # Every glass named by a canonical prescription still resolves to the file
    # that prescription recorded.
    for surface in REVERSE_TELEPHOTO.surfaces:
        if isinstance(surface.material, CatalogMaterialSpec):
            material = resolve_catalog_material(surface.material, OptilandMaterial)
            assert str(material.material_data["filename"]) == surface.material.expected_catalog_file

    # An inexact name is refused instead of silently substituted. 'SK1' resolves
    # to SK16 in this catalog.
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_CATALOG_MATERIAL_INEXACT"):
        resolve_catalog_material(CatalogMaterialSpec(name="SK1"), OptilandMaterial)

    # A name that matches nothing is a structured prescription error, not a bare
    # solver ValueError leaking through the boundary.
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_CATALOG_MATERIAL_UNRESOLVED"):
        resolve_catalog_material(CatalogMaterialSpec(name="Not-A-Glass-9Z"), OptilandMaterial)

    # And a recorded file that no longer wins is an error, not a quiet retrace.
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_CATALOG_FILE_MISMATCH"):
        resolve_catalog_material(
            CatalogMaterialSpec(name="N-BK7", expected_catalog_file="glass/hoya/BSC7.yml"),
            OptilandMaterial,
        )


def test_stop_aperture_fields_and_wavelengths_are_configured() -> None:
    spec = _minimal_spec(
        surfaces=(
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=20.0),
                thickness_mm=3.0,
                material=IdealMaterialSpec(refractive_index=1.5),
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=2.0, is_stop=True),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=36.0),
        ),
        aperture=ApertureSpec(value_mm=6.0),
        fields=(FieldSpec(y_deg=0.0), FieldSpec(x_deg=1.5, y_deg=3.0)),
        wavelengths=(
            WavelengthSpec(value_um=0.486),
            WavelengthSpec(value_um=0.55, is_primary=True),
            WavelengthSpec(value_um=0.656),
        ),
    )
    exported = build_optiland_system(spec).to_dict()

    assert exported["aperture"] == {"type": "EPD", "value": 6.0}
    # The object surface serializes as an ObjectSurface and carries no stop flag
    # at all, which is itself part of the contract: only an optical surface can
    # be the stop.
    assert "is_stop" not in exported["surface_group"]["surfaces"][0]
    assert [
        surface["is_stop"] for surface in exported["surface_group"]["surfaces"][1:]
    ] == [False, True, False, False]
    assert spec.stop_index == 1

    assert exported["fields"]["field_definition"] == {"field_type": "AngleField"}
    assert [(f["x"], f["y"]) for f in exported["fields"]["fields"]] == [(0.0, 0.0), (1.5, 3.0)]
    # Vignetting factors and weights are stated by the builder, not inherited.
    assert all(
        (f["vx"], f["vy"], f["weight"]) == (0.0, 0.0, 1.0)
        for f in exported["fields"]["fields"]
    )

    stored = exported["wavelengths"]["wavelengths"]
    assert [w["value"] for w in stored] == [0.486, 0.55, 0.656]
    assert [w["is_primary"] for w in stored] == [False, True, False]
    assert {w["unit"] for w in stored} == {"um"}
    assert spec.primary_wavelength_um == 0.55


def test_object_at_infinity_and_at_a_finite_distance() -> None:
    infinite = build_optiland_system(_minimal_spec())
    assert math.isinf(float(infinite.surfaces.surfaces[0].thickness))

    finite = build_optiland_system(_minimal_spec(object_distance_mm=250.0))
    assert float(finite.surfaces.surfaces[0].thickness) == pytest.approx(250.0)
    # A finite object changes the system, so the two must not trace alike.
    assert float(infinite.paraxial.f2()) == pytest.approx(float(finite.paraxial.f2()))
    near = np.asarray(finite.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8).y)
    far = np.asarray(infinite.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8).y)
    assert float(np.max(np.abs(near - far))) > 1e-6


def test_repeated_construction_is_equivalent() -> None:
    """Same normalized prescription, same constructed system state and trace."""
    for spec in (M3_SINGLET_REF, REVERSE_TELEPHOTO, PHASE3_FEATURE_DEMO):
        first, second = build_optiland_system(spec), build_optiland_system(spec)
        assert first.to_dict() == second.to_dict()
        a = first.trace(Hx=0.0, Hy=0.0, wavelength=spec.primary_wavelength_um, num_rays=8)
        b = second.trace(Hx=0.0, Hy=0.0, wavelength=spec.primary_wavelength_um, num_rays=8)
        for name in ("x", "y", "z", "L", "M", "N", "opd", "i"):
            first_values = np.asarray(getattr(a, name), dtype=float)
            second_values = np.asarray(getattr(b, name), dtype=float)
            assert first_values.shape == second_values.shape
            assert np.array_equal(first_values, second_values)


def test_phase3_feature_demo_exercises_every_supported_category() -> None:
    """One system, defined directly in the schema, covering the Phase 3 table.

    It is not a registered adapter prescription on purpose: it demonstrates the
    builder's supported surface, and has no analytic oracle or frozen protocol
    behind it, so promoting it to a named system would be a validation claim
    nothing here supports.
    """
    assert PHASE3_FEATURE_DEMO.name not in PRESCRIPTION_NAMES

    optic = build_optiland_system(PHASE3_FEATURE_DEMO)
    geometries = [type(s.geometry).__name__ for s in optic.surfaces.surfaces]
    assert geometries == [
        "Plane",
        "EvenAsphere",
        "StandardGeometry",
        "PlaneGrating",
        "Plane",
        "Plane",
    ]
    interactions = [type(s.interaction_model).__name__ for s in optic.surfaces.surfaces[1:]]
    assert interactions.count("DiffractiveInteractionModel") == 1

    materials = [type(s.material_post).__name__ for s in optic.surfaces.surfaces[1:]]
    assert materials[0] == "Material"  # catalog glass
    assert "IdealMaterial" in materials  # ideal index and air

    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=8)
    x, y = np.asarray(rays.x, dtype=float), np.asarray(rays.y, dtype=float)
    assert x.size > 0
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    # The grating disperses in +y for order +1 with grooves at 0 rad. The bundle
    # fills the pupil in both axes, so the signature is in the centroid: y is
    # displaced by the diffraction while x, having no term to displace it,
    # stays on axis by symmetry.
    assert float(np.mean(y)) > 1.0
    assert abs(float(np.mean(x))) <= 1e-9


# ---------------------------------------------------------------------------
# 4. Migration: the two existing systems, against independent oracles
# ---------------------------------------------------------------------------


#: Frozen canonical fingerprints. These are not a physics oracle -- they are a
#: tripwire: an accidental edit to a radius, a thickness, a glass, a field or a
#: wavelength in registry/prescriptions.py changes the digest, and this test says
#: so before a downstream fingerprint moves silently. Changing one of these
#: intentionally means changing the optical system, which needs its own evidence.
FROZEN_FINGERPRINTS = {
    "M3SingletRef": "e0d03eae536e8cce784c6cfad2027aebc0aaf73f302831c32200e80a74c85f02",
    "ReverseTelephoto": "1515317bb9a5c10c7d7b59ab38179e319036b0185e644540eee8af65e2f57c3a",
    "Phase3FeatureDemo": "b023d56fbbf5b85bea5e2cce802c20a017040eacbe2745dd30adf2663330138b",
}


@pytest.mark.parametrize(
    ("spec", "surface_count"),
    [(M3_SINGLET_REF, 2), (REVERSE_TELEPHOTO, 13), (PHASE3_FEATURE_DEMO, 4)],
)
def test_canonical_prescriptions_are_frozen(spec, surface_count) -> None:
    assert len(spec.surfaces) == surface_count
    assert spec.fingerprint() == FROZEN_FINGERPRINTS[spec.name]
    # The digest is a property of the prescription, not of how it was reached.
    assert OpticalSystemSpec.from_dict(spec.canonical_dict()).fingerprint() == spec.fingerprint()


def test_canonical_reverse_telephoto_matches_the_bundled_sample_structurally() -> None:
    """Every surface, material, spacing, field and wavelength, not a summary.

    ``Optic.to_dict()`` is a complete structural export -- geometry class and
    parameters, resolved material file, coordinate system, stop flag,
    interaction model, aperture, fields, wavelengths -- so equality here means
    the transcription is complete rather than merely plausible. Only the system
    name differs: the bundled sample never set one.
    """
    objectives = pytest.importorskip("optiland.samples.objectives")

    built = build_optiland_system(REVERSE_TELEPHOTO).to_dict()
    bundled = objectives.ReverseTelephoto().to_dict()
    assert built.pop("name") == "ReverseTelephoto"
    assert bundled.pop("name") is None
    assert built == bundled


@pytest.mark.parametrize("field_hy", [0.0, 0.5, 1.0])
def test_canonical_reverse_telephoto_traces_identically(field_hy) -> None:
    objectives = pytest.importorskip("optiland.samples.objectives")

    built = build_optiland_system(REVERSE_TELEPHOTO)
    bundled = objectives.ReverseTelephoto()
    canonical = built.trace(Hx=0.0, Hy=field_hy, wavelength=0.5876, num_rays=16)
    sample = bundled.trace(Hx=0.0, Hy=field_hy, wavelength=0.5876, num_rays=16)

    for name in ("x", "y", "z", "L", "M", "N", "opd", "i", "w"):
        canonical_values = np.asarray(getattr(canonical, name), dtype=float)
        sample_values = np.asarray(getattr(sample, name), dtype=float)
        assert canonical_values.shape == sample_values.shape, name
        # Element-wise identical, not merely close: the same prescription
        # through the same solver has no room to differ.
        assert np.array_equal(canonical_values, sample_values), name

    assert float(built.paraxial.f2()) == float(bundled.paraxial.f2())


def test_canonical_singlet_reproduces_the_frozen_protocol_geometry() -> None:
    """M3-SINGLET-REF against benchmarks/slice_protocol.yaml, not against itself.

    The protocol is M3.2's independently frozen record of this system, so it is
    the right oracle for a construction change: if the canonical prescription
    moved a radius, a thickness or the stop, the paraxial geometry below would
    not land on the frozen numbers.
    """
    protocol = yaml.safe_load(SLICE_PROTOCOL.read_text())
    system = next(s for s in protocol["systems"] if s["id"] == "M3-SINGLET-REF")
    construction = system["construction"]
    derived = system["derived"]

    # The prescription carries the protocol's own construction numbers.
    surface = M3_SINGLET_REF.surfaces[0]
    assert isinstance(surface.geometry, SphericalGeometrySpec)
    assert surface.geometry.resolved_radius_mm == construction["radius_mm"]
    assert surface.thickness_mm == construction["center_thickness_mm"]
    assert isinstance(surface.material, IdealMaterialSpec)
    assert surface.material.refractive_index == construction["refractive_index"]
    assert surface.is_stop is True
    assert M3_SINGLET_REF.stop_index == 0

    optic = build_optiland_system(M3_SINGLET_REF)
    assert float(optic.paraxial.f2()) == pytest.approx(derived["efl_mm"], rel=1e-12)
    image_z = float(np.asarray(optic.surfaces.surfaces[-1].geometry.cs.z))
    assert image_z == pytest.approx(derived["image_plane_z_mm"], rel=1e-12)
    assert image_z == pytest.approx(
        construction["center_thickness_mm"] + derived["bfl_mm"], rel=1e-12
    )
    epd = float(optic.aperture.value)
    assert epd == pytest.approx(derived["efl_mm"] / derived["f_number"], rel=1e-12)
    assert epd / 2.0 == pytest.approx(derived["semi_aperture_mm"], abs=1e-4)


def test_adapter_regression_fingerprints_are_unmoved_by_the_migration(tmp_path) -> None:
    """The frozen M1 scientific-array hash still reproduces through the adapter.

    This is the check that the migration did not move any number a downstream
    consumer depends on: same request, same recorded SHA-256 and summary
    metrics, now built from a canonical prescription instead of the bundled
    sample class.
    """
    result = OptilandAdapter().run_standalone({"output_directory": tmp_path / "baseline"})
    assert result.status is RunStatus.SUCCEEDED

    expected = load_probe_expected("optiland", "standalone_baseline")
    assert result.scientific_array_sha256 == expected["stable_result"]["scientific_array_sha256"]
    assert result.summary_metrics == expected["stable_result"]["summary_metrics"]


# ---------------------------------------------------------------------------
# 5. Adapter integration: naming, inline prescriptions, and what stays refused
# ---------------------------------------------------------------------------


def test_registry_names_are_the_adapter_supported_set() -> None:
    from adapters import optiland_adapter

    assert prescription_names() == PRESCRIPTION_NAMES
    assert optiland_adapter._SUPPORTED_SAMPLES == PRESCRIPTION_NAMES
    assert set(PRESCRIPTION_NAMES) == {"ReverseTelephoto", "M3SingletRef"}
    for name in PRESCRIPTION_NAMES:
        assert resolve_prescription(name).name == name
    with pytest.raises(PrescriptionError, match="PRESCRIPTION_NAME_UNKNOWN"):
        resolve_prescription("WideAngle100FOV")


def _request(config: dict) -> ModelRunRequest:
    return ModelRunRequest(
        run_id="pb5-run",
        node_id="lens",
        inputs={},
        config=config,
        design_parameters={},
        require_gradients=False,
    )


def test_adapter_accepts_an_inline_canonical_prescription(tmp_path) -> None:
    """A system defined directly in the schema reaches the solver, and is traced.

    This is the capability CHE-56 opens: not "an arbitrary Optiland object is
    accepted" but "a validated canonical prescription is".
    """
    spec = _minimal_spec(name="InlineDemo")
    adapter = OptilandAdapter()

    report = adapter.validate_request(
        _request({"prescription": spec, "output_directory": str(tmp_path / "inline")})
    )
    assert report.valid

    result = adapter.run(
        _request(
            {
                "prescription": spec.canonical_dict(),
                "num_rays": 8,
                "output_directory": str(tmp_path / "inline"),
            }
        )
    )
    assert result.status is RunStatus.SUCCEEDED
    assert result.diagnostics["sample"] == "InlineDemo"
    assert result.diagnostics["prescription_spec_version"] == OPTICAL_SYSTEM_SPEC_VERSION
    assert result.diagnostics["prescription_fingerprint"] == spec.fingerprint()
    assert result.diagnostics["prescription_source"] == "config['prescription']"
    assert result.outputs


def test_adapter_records_which_prescription_built_a_named_system(tmp_path) -> None:
    result = OptilandAdapter().run(
        _request(
            {"sample": "M3SingletRef", "num_rays": 8, "output_directory": str(tmp_path / "named")}
        )
    )
    assert result.status is RunStatus.SUCCEEDED
    assert result.diagnostics["sample"] == "M3SingletRef"
    assert result.diagnostics["prescription_source"] == "config['sample']"
    assert result.diagnostics["prescription_fingerprint"] == M3_SINGLET_REF.fingerprint()


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            {"prescription": _minimal_spec(), "sample": "M3SingletRef"},
            "PRESCRIPTION_CONFLICTING_SOURCES",
        ),
        (
            {"prescription": {"spec_version": "optical-system-spec/0", "name": "x"}},
            "PRESCRIPTION_SCHEMA_VERSION_UNSUPPORTED",
        ),
        (
            {"prescription": "ReverseTelephoto"},
            "PRESCRIPTION_NOT_A_MAPPING",
        ),
        (
            {"sample": "WideAngle100FOV"},
            "not a canonical prescription",
        ),
    ],
)
def test_adapter_rejects_a_bad_prescription_before_importing_the_solver(config, match) -> None:
    adapter = OptilandAdapter()
    with pytest.raises(UnsupportedCapabilityError, match=match):
        adapter.run(_request(config))
    report = adapter.validate_request(_request(config))
    assert not report.valid
    assert any(
        issue.code in {"OPTILAND_INVALID_PRESCRIPTION", "OPTILAND_UNSUPPORTED_SAMPLE"}
        for issue in report.errors
    )


def test_adapter_rejects_an_incomplete_inline_prescription() -> None:
    """A malformed prescription fails validation, never half-builds a lens."""
    payload = _minimal_spec().canonical_dict()
    del payload["aperture"]
    adapter = OptilandAdapter()
    with pytest.raises(
        UnsupportedCapabilityError, match=r"OPTILAND_INVALID_PRESCRIPTION|aperture"
    ):
        adapter.run(_request({"prescription": payload}))
