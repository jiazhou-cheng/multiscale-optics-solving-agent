"""Generic Optiland construction from a canonical prescription (CHE-56 / PB5).

One function, :func:`build_optiland_system`, turns an
:class:`~multiscale_optics_agent.core.optical_system.OpticalSystemSpec` into an
ordinary ``optiland.optic.Optic``. It is the *only* system-construction path for
adapter-owned prescriptions: ``M3SingletRef`` no longer has a hand-written
builder, and ``ReverseTelephoto`` no longer needs
``optiland.samples.objectives`` at runtime (the bundled sample is retained as a
regression oracle in the tests, which is a different job).

Why the builder validates before it constructs
----------------------------------------------
Optiland's ``GeometryFactory.create`` filters ``**kwargs`` down to the fields of
the geometry config it selected, so a prescription key that does not belong to
the chosen surface type is **discarded with no error at all** -- measured in
``knowledge/solvers/optiland/probes/system_construction_probe.py``, case
``surface_kwargs_are_silently_filtered``. Passing a prescription straight
through would therefore convert a caller's mistake into a silently different
optical system. Every mapping below is explicit: the surface type is chosen
here, the exact keyword set for that type is assembled here, and anything the
Phase 3 schema cannot express raises before ``Optic`` is touched.

Determinism
-----------
Surfaces are created in prescription order from a tuple; materials are resolved
by an explicit per-kind branch; the stop comes from the single surface the
schema guarantees is marked; aperture, fields and wavelengths are configured in
one fixed order with every solver-side default stated explicitly rather than
inherited. No set iteration, no dict ordering, no RNG, no clock.

Units are the schema's (``core.optical_system.UNITS``) and match the solver's
without conversion: millimetres for geometry, micrometres for wavelength *and*
for grating period, radians for groove orientation, degrees for angular fields.
"""

from __future__ import annotations

from typing import Any

from multiscale_optics_agent.core.errors import AdapterDependencyError
from multiscale_optics_agent.core.optical_system import (
    FIELD_VIGNETTING_FACTORS,
    FIELD_WEIGHT,
    WAVELENGTH_WEIGHT,
    AirMaterialSpec,
    CatalogMaterialSpec,
    EvenAsphereGeometrySpec,
    GeometryKind,
    IdealMaterialSpec,
    InteractionKind,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
    SurfaceSpec,
)

#: Optiland surface_type strings this builder emits, and nothing else.
_SURFACE_TYPE_STANDARD = "standard"
_SURFACE_TYPE_EVEN_ASPHERE = "even_asphere"
_SURFACE_TYPE_GRATING = "grating"

#: The wavelength unit Optiland itself records on every wavelength it stores.
#: Stated explicitly so the schema's micrometre contract is asserted at the
#: boundary rather than assumed from the solver's default.
_OPTILAND_WAVELENGTH_UNIT = "um"


def _import_optiland_construction() -> tuple[Any, Any, Any, Any]:
    """Import exactly what construction needs, and nothing else.

    Kept out of module scope so importing this module never imports the solver
    (the adapter registry imports every adapter eagerly).
    """
    try:
        import optiland.backend as be  # type: ignore[import-untyped]
        from optiland.materials import (  # type: ignore[import-untyped]
            IdealMaterial,
            Material,
        )
        from optiland.optic import Optic  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - environment failure
        raise AdapterDependencyError(
            f"optiland could not be imported: {type(exc).__name__}: {exc}. "
            "Install it via `pip install optiland==0.6.0` or this project's "
            "'torch' extra (`pip install .[torch]`, which also pins torch)."
        ) from exc
    return be, Optic, IdealMaterial, Material


def resolve_catalog_material(spec: CatalogMaterialSpec, material_cls: Any) -> Any:
    """Resolve a catalog glass, refusing an ambiguous or unexpected match.

    Optiland's own lookup is a substring filter over three columns followed by a
    Levenshtein ranking, and it returns the best row rather than insisting on an
    exact one. So ``SK15`` legitimately resolves to HIKARI while ``N-SK10``
    resolves to SCHOTT, and a typo can resolve to a real but different glass.
    Two guards close that:

    * ``require_exact_match`` (default true) demands the winning row's catalog
      name equal the requested name, case-insensitively.
    * ``expected_catalog_file``, when the prescription records it, must equal
      the catalog file actually chosen -- so a database change in a future
      Optiland release is an error here instead of a quietly different trace.
    """
    try:
        material = material_cls(name=spec.name, reference=spec.catalog)
    except PrescriptionError:  # pragma: no cover - defensive
        raise
    except Exception as exc:
        # Optiland raises a bare ValueError for "no matches" and for "multiple
        # matches with robust_search off". Neither is a solver failure: it is a
        # prescription this builder cannot resolve, and it must arrive at the
        # caller as one.
        raise PrescriptionError(
            "PRESCRIPTION_CATALOG_MATERIAL_UNRESOLVED",
            f"catalog name {spec.name!r} (catalog={spec.catalog!r}) could not be "
            f"resolved against the pinned material database: "
            f"{type(exc).__name__}: {exc}",
            path="material.name",
            expected="a name present in the pinned Optiland material catalog",
        ) from exc
    row = material.material_data
    resolved_name = str(row["name"])
    resolved_file = str(row["filename"])
    # Optiland's own exactness criterion, taken from the row it selected rather
    # than re-derived: `similarity_score` is the minimum Levenshtein distance
    # from the requested name to the row's category name, catalog name, or
    # filename stem, and 0 means one of them matched exactly. Comparing only
    # against the catalog name would reject legitimate entries -- 'N-BK7'
    # resolves to a row whose name is 'N-BK7 (SCHOTT)' and whose filename stem
    # is 'N-BK7'.
    similarity = row.get("similarity_score")
    if spec.require_exact_match and (similarity is None or float(similarity) != 0.0):
        raise PrescriptionError(
            "PRESCRIPTION_CATALOG_MATERIAL_INEXACT",
            f"catalog name {spec.name!r} resolved to {resolved_name!r} "
            f"({resolved_file}) by similarity (score {similarity!r}) rather than exactly",
            path="material.name",
            expected="a catalog name that matches a catalog entry exactly",
        )
    if spec.expected_catalog_file is not None and resolved_file != spec.expected_catalog_file:
        raise PrescriptionError(
            "PRESCRIPTION_CATALOG_FILE_MISMATCH",
            f"catalog name {spec.name!r} (catalog={spec.catalog!r}) resolved to "
            f"{resolved_file!r}, not the recorded {spec.expected_catalog_file!r}",
            path="material.expected_catalog_file",
            expected="the catalog file recorded when this prescription was validated",
        )
    return material


def _material_argument(surface: SurfaceSpec, ideal_cls: Any, material_cls: Any) -> Any:
    """The ``material=`` value for ``surfaces.add``, i.e. the medium *after* it."""
    material = surface.material
    if isinstance(material, AirMaterialSpec):
        # The literal string is what Optiland's MaterialFactory turns into
        # IdealMaterial(n=1.0, k=0.0); passing it keeps this builder's output
        # identical to a bundled sample that omitted `material=` entirely.
        return "air"
    if isinstance(material, IdealMaterialSpec):
        return ideal_cls(n=material.refractive_index, k=material.absorption)
    if isinstance(material, CatalogMaterialSpec):
        return resolve_catalog_material(material, material_cls)
    raise PrescriptionError(  # pragma: no cover - the union is closed
        "PRESCRIPTION_MATERIAL_UNSUPPORTED",
        f"material kind {getattr(material, 'kind', material)!r} is not implemented",
        path="surface.material",
        supported=("air", "ideal", "catalog"),
    )


def _geometry_arguments(surface: SurfaceSpec, be: Any) -> tuple[str, dict[str, Any]]:
    """Choose the Optiland ``surface_type`` and its complete keyword set.

    The pair is decided jointly by geometry and interaction because a grating is
    a geometry class in the pinned solver, not an attribute of one:
    ``surface_type='grating'`` yields ``PlaneGrating`` for an infinite base
    radius and ``StandardGratingGeometry`` otherwise, and the surface factory
    then selects the diffractive interaction model from the same string.
    """
    geometry = surface.geometry
    interaction = surface.interaction

    if isinstance(geometry, PlaneGeometrySpec):
        radius: Any = be.inf
        conic = 0.0
    elif isinstance(geometry, (SphericalGeometrySpec, EvenAsphereGeometrySpec)):
        radius = geometry.resolved_radius_mm
        conic = geometry.conic
    else:  # pragma: no cover - the union is closed
        raise PrescriptionError(
            "PRESCRIPTION_GEOMETRY_UNSUPPORTED",
            f"geometry kind {getattr(geometry, 'kind', geometry)!r} is not implemented",
            path="surface.geometry",
            supported=tuple(kind.value for kind in GeometryKind),
        )

    if interaction.kind is InteractionKind.GRATING:
        # SurfaceSpec already refused an aspheric grating base; assert the
        # remaining invariant rather than trust it silently.
        if isinstance(geometry, EvenAsphereGeometrySpec):  # pragma: no cover
            raise PrescriptionError(
                "PRESCRIPTION_GRATING_GEOMETRY_UNSUPPORTED",
                "a grating cannot be ruled on an even-aspheric base surface",
                path="surface.interaction",
                supported=("plane", "spherical"),
            )
        return _SURFACE_TYPE_GRATING, {
            "radius": radius,
            "conic": conic,
            "grating_order": interaction.order,
            "grating_period": interaction.period_um,
            "groove_orientation_angle": interaction.groove_orientation_rad,
        }

    if isinstance(geometry, EvenAsphereGeometrySpec):
        return _SURFACE_TYPE_EVEN_ASPHERE, {
            "radius": radius,
            "conic": conic,
            "coefficients": list(geometry.coefficients),
        }

    # Plane and spherical both go through 'standard': Optiland's own factory
    # returns a Plane when the radius is infinite, which is exactly how every
    # existing prescription in this repository spells a flat surface.
    return _SURFACE_TYPE_STANDARD, {"radius": radius, "conic": conic}


def build_optiland_system(spec: OpticalSystemSpec) -> Any:
    """Construct an Optiland ``Optic`` from a canonical prescription.

    The returned object is an ordinary ``Optic``: tracing, paraxial analysis and
    every downstream adapter path treat it exactly as they treat a bundled
    sample.

    Raises:
        PrescriptionError: the prescription is outside the Phase 3 supported
            set, or a catalog material did not resolve as recorded. Raised
            before any surface is added, so a caller never receives a partially
            constructed lens.
        AdapterDependencyError: optiland is not importable.
    """
    if not isinstance(spec, OpticalSystemSpec):
        raise PrescriptionError(
            "PRESCRIPTION_NOT_A_SPEC",
            f"expected an OpticalSystemSpec, got {type(spec).__name__}",
            path="",
            expected=(
                "an OpticalSystemSpec instance; a serialized prescription must be "
                "parsed with OpticalSystemSpec.from_dict first so its schema "
                "version is checked"
            ),
        )

    be, optic_cls, ideal_cls, material_cls = _import_optiland_construction()

    # 1. Resolve everything that can fail *before* touching Optiland, so a
    #    rejected prescription never leaves a half-built Optic behind.
    plans: list[tuple[SurfaceSpec, str, dict[str, Any], Any]] = []
    for index, surface in enumerate(spec.surfaces):
        try:
            surface_type, geometry_kwargs = _geometry_arguments(surface, be)
            material = _material_argument(surface, ideal_cls, material_cls)
        except PrescriptionError as exc:
            raise PrescriptionError(
                exc.code,
                str(exc).split("] ", 1)[-1],
                path=f"surfaces[{index}]" + (f".{exc.path}" if exc.path else ""),
                expected=exc.expected,
                supported=exc.supported,
            ) from exc
        plans.append((surface, surface_type, geometry_kwargs, material))

    # 2. Construct, in prescription order.
    optic = optic_cls(name=spec.name)

    # Object surface: a plane in air. `thickness` is the distance to the first
    # optical surface, infinite for an object at infinity.
    optic.surfaces.add(
        index=0,
        radius=be.inf,
        thickness=be.inf if spec.object_distance_mm is None else spec.object_distance_mm,
    )

    for offset, (surface, surface_type, geometry_kwargs, material) in enumerate(plans):
        optic.surfaces.add(
            index=offset + 1,
            surface_type=surface_type,
            thickness=surface.thickness_mm,
            material=material,
            is_stop=surface.is_stop,
            comment=surface.comment,
            **geometry_kwargs,
        )

    # Image surface: a plane in air at zero further spacing, placed by the last
    # optical surface's thickness.
    optic.surfaces.add(index=len(plans) + 1, radius=be.inf, thickness=0.0)

    # 3. Aperture, fields, wavelengths -- one fixed order, no inherited default.
    optic.set_aperture(aperture_type=spec.aperture.kind.value, value=spec.aperture.value_mm)

    optic.fields.set_type(field_type=spec.field_kind.value)
    vignette_x, vignette_y = FIELD_VIGNETTING_FACTORS
    for field in spec.fields:
        optic.fields.add(
            y=field.y_deg,
            x=field.x_deg,
            vx=vignette_x,
            vy=vignette_y,
            weight=FIELD_WEIGHT,
        )

    for wavelength in spec.wavelengths:
        optic.wavelengths.add(
            value=wavelength.value_um,
            is_primary=wavelength.is_primary,
            unit=_OPTILAND_WAVELENGTH_UNIT,
            weight=WAVELENGTH_WEIGHT,
        )

    return optic


__all__ = ["build_optiland_system", "resolve_catalog_material"]
