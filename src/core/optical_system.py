"""Canonical, project-owned optical-system prescription schema (CHE-56 / PB5).

This module is the authoritative representation of an optical system in this
repository. It is deliberately **solver-agnostic**: it imports no ray tracer,
declares its own units, and knows nothing about how Optiland spells a surface.
The Optiland-specific translation lives in
:mod:`solvers.optiland.builder`, which is the only
place allowed to import Optiland (repository rule: keep external solver
imports inside adapter modules).

Why a project-owned schema rather than a solver file format
-----------------------------------------------------------
The adapter previously carried two independent construction implementations:
bundled samples came from ``optiland.samples.objectives`` and the
adapter-owned ``M3SingletRef`` was a hand-written sequence of
``Optic.surfaces.add`` calls. Prescription *data* and construction *procedure*
were entangled, so a new system meant new code. Here the data is a typed,
versioned, serializable value and the procedure is one generic builder.

Units (fixed by the schema contract; CHE-12 established the Optiland side)
-------------------------------------------------------------------------
- All lengths that describe geometry -- radius, thickness, aperture diameter,
  object distance -- are **millimetres**.
- Curvature, where used instead of radius, is **1/millimetre**.
- Even-asphere coefficient ``coefficients[i]`` multiplies ``r**(2*(i+1))`` with
  ``r`` in millimetres, so its unit is ``mm**(1 - 2*(i+1))``. The series starts
  at ``r**2``, verified to 5.6e-17 mm against an independent evaluation in
  ``knowledge/solvers/optiland/probes/system_construction_probe.py``.
- Wavelengths are **micrometres**.
- Grating period is **micrometres** -- deliberately *not* the geometry unit.
  ``knowledge/solvers/optiland/probes/system_construction_probe.py`` establishes
  this by reproducing ``sin(theta_m) = m * lambda / d`` exactly for three
  periods; the millimetre reading is wrong by 1000x.
- Angular field coordinates are **degrees**; grating groove orientation is
  **radians**, matching the two respective solver-side conventions rather than
  silently unifying them.

These conventions are asserted by tests; they are not free to drift.

Deterministic construction
--------------------------
Every collection in this schema is an ordered tuple, every model forbids
undeclared fields, and :meth:`OpticalSystemSpec.canonical_dict` emits a
key-sorted, JSON-safe normalization suitable for equality and fingerprint
testing. No validation or normalization step consults a set, a dict ordering,
a random source, or the clock.

Scope
-----
Phase 3 (CHE-56) covers plane / spherical / even-aspheric geometry, refractive
and grating interactions, air / ideal-index / catalog materials, one aperture
stop, an entrance-pupil-diameter aperture, angular fields, and a wavelength
list with one primary. Anything outside that raises
:class:`PrescriptionError` eagerly rather than being approximated. Decentres,
tilts, coordinate breaks, GRIN, freeform sag, coatings and BSDFs are out of
scope by ticket and are not silently mapped onto a supported feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.errors import MultiScaleOpticsError

#: Identifier of this schema. Bump the integer when a change would make an
#: existing serialized prescription mean something different; a reader that
#: sees an unknown version must refuse it rather than reinterpret it.
OPTICAL_SYSTEM_SPEC_VERSION: Final = "optical-system-spec/1"

#: Declared unit conventions, exported so a caller (or a test) can assert them
#: instead of trusting prose.
UNITS: dict[str, str] = {
    "radius": "mm",
    "curvature": "1/mm",
    "thickness": "mm",
    "object_distance": "mm",
    "aperture_epd": "mm",
    "wavelength": "um",
    "grating_period": "um",
    "groove_orientation": "rad",
    "angular_field": "deg",
    "asphere_coefficient": "mm**(1 - 2*(i+1)) for coefficients[i]",
}

# Solver-side constants that the builder must state explicitly rather than
# inherit, because a change in them would change the constructed system. They
# live here so the schema documents the whole optical system, including the
# parts Phase 3 fixes rather than parameterizes.
#: Vignetting factors applied to every field. Phase 3 does not model vignetting
#: factors; zero means "no vignetting compression", which is what both migrated
#: prescriptions use.
FIELD_VIGNETTING_FACTORS: tuple[float, float] = (0.0, 0.0)
#: Per-field and per-wavelength weights. Phase 3 traces one field at a time and
#: does not form weighted merit functions, so every weight is unity.
FIELD_WEIGHT = 1.0
WAVELENGTH_WEIGHT = 1.0


class PrescriptionError(MultiScaleOpticsError):
    """A prescription is invalid, incomplete, or outside the supported set.

    Structured on purpose: ``code`` is stable and matchable, ``path`` says
    *where* in the prescription the problem is, ``expected`` says what would
    have been acceptable, and ``supported`` enumerates the alternatives. The
    adapter boundary re-raises this as ``UnsupportedCapabilityError`` where its
    published contract requires that type, without losing these fields.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "",
        expected: str = "",
        supported: tuple[str, ...] = (),
    ) -> None:
        detail = message
        if path:
            detail = f"{path}: {detail}"
        if expected:
            detail = f"{detail} Expected {expected}."
        if supported:
            detail = f"{detail} Supported: {', '.join(supported)}."
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.path = path
        self.expected = expected
        self.supported = supported


class _Frozen(BaseModel):
    """Immutable, closed base model.

    ``extra='forbid'`` is load-bearing rather than stylistic: Optiland's
    ``GeometryFactory`` silently filters unrecognized keyword arguments down to
    the fields of the selected geometry config, so a misspelled prescription
    key reaching the solver produces a *different optical system with no error*
    (``system_construction_probe.py``, case
    ``surface_kwargs_are_silently_filtered``). Refusing unknown fields here is
    the only place that mistake can be caught.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class GeometryKind(StrEnum):
    PLANE = "plane"
    SPHERICAL = "spherical"
    EVEN_ASPHERE = "even_asphere"


class InteractionKind(StrEnum):
    REFRACTIVE = "refractive"
    GRATING = "grating"


class MaterialKind(StrEnum):
    AIR = "air"
    IDEAL = "ideal"
    CATALOG = "catalog"


class ApertureKind(StrEnum):
    #: Entrance pupil diameter, in millimetres.
    EPD = "EPD"


class FieldKind(StrEnum):
    #: Field coordinates are angles in degrees.
    ANGLE = "angle"


def _resolve_radius(radius_mm: float | None, curvature_per_mm: float | None, path: str) -> float:
    """Return the radius in mm from whichever of the two forms was supplied.

    Both representations are offered because prescriptions in the literature
    use both; exactly one must be given so the stored value is never a
    derived-and-possibly-inconsistent duplicate.
    """
    if (radius_mm is None) == (curvature_per_mm is None):
        raise PrescriptionError(
            "PRESCRIPTION_RADIUS_UNDERSPECIFIED",
            "curved geometry needs exactly one of radius_mm or curvature_per_mm",
            path=path,
            expected="exactly one of radius_mm, curvature_per_mm",
        )
    if radius_mm is not None:
        value = float(radius_mm)
        if not math.isfinite(value) or value == 0.0:
            raise PrescriptionError(
                "PRESCRIPTION_RADIUS_NOT_CURVED",
                f"radius_mm={value!r} does not describe a curved surface",
                path=path,
                expected="a finite non-zero radius; use geometry kind 'plane' for a flat surface",
            )
        return value
    curvature = float(curvature_per_mm)  # type: ignore[arg-type]
    if not math.isfinite(curvature) or curvature == 0.0:
        raise PrescriptionError(
            "PRESCRIPTION_CURVATURE_NOT_CURVED",
            f"curvature_per_mm={curvature!r} does not describe a curved surface",
            path=path,
            expected="a finite non-zero curvature; use geometry kind 'plane' for a flat surface",
        )
    return 1.0 / curvature


class PlaneGeometrySpec(_Frozen):
    """A flat surface. Carries no radius: a plane is not a sphere of infinite
    radius in this schema, it is its own kind, so ``radius=inf`` can never
    arrive as a float and defeat JSON serialization."""

    kind: Literal[GeometryKind.PLANE] = GeometryKind.PLANE

    @property
    def radius_mm(self) -> float:
        return math.inf

    @property
    def conic(self) -> float:
        return 0.0


class SphericalGeometrySpec(_Frozen):
    """A rotationally symmetric conic surface: sphere when ``conic == 0``."""

    kind: Literal[GeometryKind.SPHERICAL] = GeometryKind.SPHERICAL
    radius_mm: float | None = None
    curvature_per_mm: float | None = None
    conic: float = 0.0

    @model_validator(mode="after")
    def _check_curvature(self) -> SphericalGeometrySpec:
        _resolve_radius(self.radius_mm, self.curvature_per_mm, "geometry")
        return self

    @property
    def resolved_radius_mm(self) -> float:
        return _resolve_radius(self.radius_mm, self.curvature_per_mm, "geometry")


class EvenAsphereGeometrySpec(_Frozen):
    """Base conic plus an even polynomial in the radial coordinate.

    ``coefficients[i]`` multiplies ``r**(2*(i+1))``: the series starts at
    ``r**2``, not ``r**4``. That is a measured property of the pinned install,
    not a convention chosen here.
    """

    kind: Literal[GeometryKind.EVEN_ASPHERE] = GeometryKind.EVEN_ASPHERE
    radius_mm: float | None = None
    curvature_per_mm: float | None = None
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _check_curvature(self) -> EvenAsphereGeometrySpec:
        _resolve_radius(self.radius_mm, self.curvature_per_mm, "geometry")
        for index, value in enumerate(self.coefficients):
            if not math.isfinite(value):
                raise PrescriptionError(
                    "PRESCRIPTION_ASPHERE_COEFFICIENT_NOT_FINITE",
                    f"coefficients[{index}]={value!r} is not finite",
                    path="geometry.coefficients",
                    expected="a finite float",
                )
        return self

    @property
    def resolved_radius_mm(self) -> float:
        return _resolve_radius(self.radius_mm, self.curvature_per_mm, "geometry")


GeometrySpec = Annotated[
    PlaneGeometrySpec | SphericalGeometrySpec | EvenAsphereGeometrySpec,
    Field(discriminator="kind"),
]


class RefractiveInteractionSpec(_Frozen):
    """Ordinary refraction at the surface (Snell's law)."""

    kind: Literal[InteractionKind.REFRACTIVE] = InteractionKind.REFRACTIVE


class GratingInteractionSpec(_Frozen):
    """A diffraction grating ruled on the surface.

    ``period_um`` is micrometres, sharing units with the wavelength rather than
    with the surrounding millimetre geometry; ``groove_orientation_rad`` is
    radians, and at 0 the grating vector points along +y so the diffracted
    order is deviated in y. Both established by
    ``knowledge/solvers/optiland/probes/system_construction_probe.py``.
    """

    kind: Literal[InteractionKind.GRATING] = InteractionKind.GRATING
    order: int
    period_um: float = Field(gt=0.0)
    groove_orientation_rad: float = 0.0

    @model_validator(mode="after")
    def _check_finite(self) -> GratingInteractionSpec:
        if not math.isfinite(self.period_um):
            raise PrescriptionError(
                "PRESCRIPTION_GRATING_PERIOD_NOT_FINITE",
                f"period_um={self.period_um!r} is not finite",
                path="interaction.period_um",
                expected="a finite positive period in micrometres",
            )
        if not math.isfinite(self.groove_orientation_rad):
            raise PrescriptionError(
                "PRESCRIPTION_GROOVE_ORIENTATION_NOT_FINITE",
                f"groove_orientation_rad={self.groove_orientation_rad!r} is not finite",
                path="interaction.groove_orientation_rad",
                expected="a finite angle in radians",
            )
        return self


InteractionSpec = Annotated[
    RefractiveInteractionSpec | GratingInteractionSpec,
    Field(discriminator="kind"),
]


class AirMaterialSpec(_Frozen):
    """The medium following the surface is air, modelled as index exactly 1.

    This is the *modelled* air of the solver -- a unit-index, lossless medium --
    not an environmental air-index model. Optiland's ``environment`` module is
    not exercised by this repository.
    """

    kind: Literal[MaterialKind.AIR] = MaterialKind.AIR


class IdealMaterialSpec(_Frozen):
    """A constant, dispersionless index. Independent of any glass catalog."""

    kind: Literal[MaterialKind.IDEAL] = MaterialKind.IDEAL
    refractive_index: float = Field(gt=0.0)
    absorption: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_finite(self) -> IdealMaterialSpec:
        if not math.isfinite(self.refractive_index):
            raise PrescriptionError(
                "PRESCRIPTION_INDEX_NOT_FINITE",
                f"refractive_index={self.refractive_index!r} is not finite",
                path="material.refractive_index",
                expected="a finite positive index",
            )
        return self


class CatalogMaterialSpec(_Frozen):
    """A named glass from the solver's bundled catalog.

    Optiland resolves a bare name by substring match over three columns and
    then by Levenshtein similarity, returning the best row -- so ``SK15``
    resolves to HIKARI while ``N-SK10`` resolves to SCHOTT, and up to seven rows
    survive the filter before scoring
    (``knowledge/solvers/optiland/probes/system_construction_probe.py``). A
    prescription that records only the name has therefore not pinned its glass.

    ``catalog`` narrows the search the way the solver's own ``reference``
    argument does. ``expected_catalog_file`` is the guard: when set, the builder
    resolves the material and refuses to continue if the winning catalog file
    differs, so a database change surfaces as an error instead of as a quietly
    different trace.
    """

    kind: Literal[MaterialKind.CATALOG] = MaterialKind.CATALOG
    name: str = Field(min_length=1)
    catalog: str | None = None
    expected_catalog_file: str | None = None
    #: Require the resolved row to be an exact (similarity 0) name match. On by
    #: default because every glass in the migrated prescriptions is exact, and a
    #: fuzzy match is a silent substitution of a different material.
    require_exact_match: bool = True


MaterialSpec = Annotated[
    AirMaterialSpec | IdealMaterialSpec | CatalogMaterialSpec,
    Field(discriminator="kind"),
]


class SurfaceSpec(_Frozen):
    """One optical surface, and the medium and spacing that follow it.

    ``material`` is the medium *after* the surface, matching the sequential
    convention: a lens is a surface carrying glass followed by a surface
    carrying air. ``thickness_mm`` is the axial distance from this surface to
    the next one in the sequence; on the last surface it is the distance to the
    image plane.
    """

    geometry: GeometrySpec
    thickness_mm: float
    material: MaterialSpec = AirMaterialSpec()
    is_stop: bool = False
    interaction: InteractionSpec = RefractiveInteractionSpec()
    comment: str = ""

    @model_validator(mode="after")
    def _check_surface(self) -> SurfaceSpec:
        if not math.isfinite(self.thickness_mm):
            raise PrescriptionError(
                "PRESCRIPTION_THICKNESS_NOT_FINITE",
                f"thickness_mm={self.thickness_mm!r} is not finite",
                path="surface.thickness_mm",
                expected="a finite axial spacing in millimetres",
            )
        # A grating is a *geometry* class in the pinned solver (PlaneGrating for
        # an infinite base radius, StandardGratingGeometry otherwise), so it can
        # only be ruled on a plane or a conic base. An aspheric grating is not
        # representable and must not be silently downgraded to its base conic.
        if (
            self.interaction.kind is InteractionKind.GRATING
            and self.geometry.kind is GeometryKind.EVEN_ASPHERE
        ):
            raise PrescriptionError(
                "PRESCRIPTION_GRATING_GEOMETRY_UNSUPPORTED",
                "a grating cannot be ruled on an even-aspheric base surface in the "
                "pinned solver, which implements gratings as plane or conic geometry classes",
                path="surface.interaction",
                expected="geometry kind 'plane' or 'spherical' under a grating interaction",
                supported=("plane", "spherical"),
            )
        return self


class ApertureSpec(_Frozen):
    """System aperture. Phase 3 supports entrance pupil diameter only."""

    kind: Literal[ApertureKind.EPD] = ApertureKind.EPD
    value_mm: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_finite(self) -> ApertureSpec:
        if not math.isfinite(self.value_mm):
            raise PrescriptionError(
                "PRESCRIPTION_APERTURE_NOT_FINITE",
                f"value_mm={self.value_mm!r} is not finite",
                path="aperture.value_mm",
                expected="a finite positive entrance pupil diameter in millimetres",
            )
        return self


class FieldSpec(_Frozen):
    """One field point, in degrees, for an angular field definition."""

    x_deg: float = 0.0
    y_deg: float = 0.0

    @model_validator(mode="after")
    def _check_finite(self) -> FieldSpec:
        for name, value in (("x_deg", self.x_deg), ("y_deg", self.y_deg)):
            if not math.isfinite(value):
                raise PrescriptionError(
                    "PRESCRIPTION_FIELD_NOT_FINITE",
                    f"{name}={value!r} is not finite",
                    path=f"field.{name}",
                    expected="a finite angle in degrees",
                )
        return self


class WavelengthSpec(_Frozen):
    """One wavelength, in micrometres."""

    value_um: float = Field(gt=0.0)
    is_primary: bool = False

    @model_validator(mode="after")
    def _check_finite(self) -> WavelengthSpec:
        if not math.isfinite(self.value_um):
            raise PrescriptionError(
                "PRESCRIPTION_WAVELENGTH_NOT_FINITE",
                f"value_um={self.value_um!r} is not finite",
                path="wavelength.value_um",
                expected="a finite positive wavelength in micrometres",
            )
        return self


class OpticalSystemSpec(_Frozen):
    """A complete optical prescription.

    ``surfaces`` lists the optical surfaces in order, starting at the first
    surface light meets and ending at the last one before the image plane. Two
    surfaces are *not* listed because Phase 3 fixes them rather than
    parameterizing them, and both are stated here so nothing about the
    constructed system is implicit:

    - the object surface, a plane in air, placed ``object_distance_mm`` before
      the first surface (``None`` means an object at infinity);
    - the image surface, a plane in air at zero further spacing, placed by the
      last listed surface's ``thickness_mm``.

    Making either of them a general surface is out of Phase 3 scope; a
    prescription needing a curved object or a tilted image plane must extend
    this schema under a new version rather than smuggle it through.
    """

    # Spelled as a literal string rather than as ``Literal[OPTICAL_SYSTEM_SPEC_VERSION]``
    # because a Literal cannot be parameterized by a variable. The two are kept in
    # step by ``test_schema_version_is_declared_and_enforced``.
    spec_version: Literal["optical-system-spec/1"] = "optical-system-spec/1"
    name: str = Field(min_length=1)
    surfaces: tuple[SurfaceSpec, ...]
    aperture: ApertureSpec
    fields: tuple[FieldSpec, ...]
    wavelengths: tuple[WavelengthSpec, ...]
    field_kind: Literal[FieldKind.ANGLE] = FieldKind.ANGLE
    object_distance_mm: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check_system(self) -> OpticalSystemSpec:
        if not self.surfaces:
            raise PrescriptionError(
                "PRESCRIPTION_NO_SURFACES",
                "a system needs at least one optical surface",
                path="surfaces",
                expected="one or more surfaces, in the order light meets them",
            )
        stops = [index for index, surface in enumerate(self.surfaces) if surface.is_stop]
        if len(stops) != 1:
            raise PrescriptionError(
                "PRESCRIPTION_STOP_AMBIGUOUS",
                f"{len(stops)} surfaces are marked is_stop (indices {stops})",
                path="surfaces",
                expected=(
                    "exactly one aperture stop; the generic builder does not support "
                    "a system with no stop or with several"
                ),
            )
        if not self.fields:
            raise PrescriptionError(
                "PRESCRIPTION_NO_FIELDS",
                "a system needs at least one field point",
                path="fields",
                expected="one or more field points in degrees",
            )
        if not self.wavelengths:
            raise PrescriptionError(
                "PRESCRIPTION_NO_WAVELENGTHS",
                "a system needs at least one wavelength",
                path="wavelengths",
                expected="one or more wavelengths in micrometres",
            )
        primaries = [
            index
            for index, wavelength in enumerate(self.wavelengths)
            if wavelength.is_primary
        ]
        if len(primaries) != 1:
            raise PrescriptionError(
                "PRESCRIPTION_PRIMARY_WAVELENGTH_AMBIGUOUS",
                f"{len(primaries)} wavelengths are marked is_primary (indices {primaries})",
                path="wavelengths",
                expected="exactly one primary wavelength",
            )
        if self.object_distance_mm is not None and (
            not math.isfinite(self.object_distance_mm) or self.object_distance_mm < 0.0
        ):
            raise PrescriptionError(
                "PRESCRIPTION_OBJECT_DISTANCE_INVALID",
                f"object_distance_mm={self.object_distance_mm!r} is not a "
                "finite non-negative distance",
                path="object_distance_mm",
                expected="a finite distance >= 0 in millimetres, or null for infinity",
            )
        return self

    @property
    def stop_index(self) -> int:
        """Index into :attr:`surfaces` of the single aperture stop."""
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index
        # Unreachable: the validator requires exactly one stop.
        raise PrescriptionError(
            "PRESCRIPTION_STOP_AMBIGUOUS",
            "no aperture stop is declared",
            path="surfaces",
            expected="exactly one aperture stop",
        )

    @property
    def primary_wavelength_um(self) -> float:
        for wavelength in self.wavelengths:
            if wavelength.is_primary:
                return wavelength.value_um
        raise PrescriptionError(
            "PRESCRIPTION_PRIMARY_WAVELENGTH_AMBIGUOUS",
            "no primary wavelength is declared",
            path="wavelengths",
            expected="exactly one primary wavelength",
        )

    # -- canonical normalization -------------------------------------------
    #
    # `canonical_dict` is the normalized form: every field present (so an
    # omitted default and an explicitly written default compare equal), every
    # container a list in prescription order, every mapping key-sorted, and no
    # non-finite float anywhere -- which is what makes `canonical_json` real
    # JSON and `fingerprint` stable.

    def canonical_dict(self) -> dict[str, Any]:
        normalized = _sort_keys(self.model_dump(mode="json"))
        assert isinstance(normalized, dict)  # model_dump on a BaseModel is a mapping
        return normalized

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def fingerprint(self) -> str:
        """SHA-256 of :meth:`canonical_json`; equal prescriptions, equal digest."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> OpticalSystemSpec:
        """Parse a serialized prescription, checking the schema version first.

        A wrong or missing version is refused outright: silently reinterpreting
        an older prescription under newer field semantics is exactly the failure
        the version identifier exists to prevent.
        """
        if not isinstance(data, Mapping):
            raise PrescriptionError(
                "PRESCRIPTION_NOT_A_MAPPING",
                f"expected a mapping, got {type(data).__name__}",
                path="",
                expected="a mapping of prescription fields",
            )
        version = data.get("spec_version")
        if version != OPTICAL_SYSTEM_SPEC_VERSION:
            raise PrescriptionError(
                "PRESCRIPTION_SCHEMA_VERSION_UNSUPPORTED",
                f"spec_version={version!r} cannot be read by this build",
                path="spec_version",
                expected=f"{OPTICAL_SYSTEM_SPEC_VERSION!r}",
            )
        return cls.model_validate(data)


def _sort_keys(value: Any) -> Any:
    """Recursively key-sort mappings, leaving sequence order untouched.

    Sequence order is prescription order and carries physical meaning, so it is
    never sorted; mapping order carries none, so it is normalized away.
    """
    if isinstance(value, dict):
        return {key: _sort_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(item) for item in value]
    return value


__all__ = [
    "FIELD_VIGNETTING_FACTORS",
    "FIELD_WEIGHT",
    "OPTICAL_SYSTEM_SPEC_VERSION",
    "UNITS",
    "WAVELENGTH_WEIGHT",
    "AirMaterialSpec",
    "ApertureKind",
    "ApertureSpec",
    "CatalogMaterialSpec",
    "EvenAsphereGeometrySpec",
    "FieldKind",
    "FieldSpec",
    "GeometryKind",
    "GeometrySpec",
    "GratingInteractionSpec",
    "IdealMaterialSpec",
    "InteractionKind",
    "InteractionSpec",
    "MaterialKind",
    "MaterialSpec",
    "OpticalSystemSpec",
    "PlaneGeometrySpec",
    "PrescriptionError",
    "RefractiveInteractionSpec",
    "SphericalGeometrySpec",
    "SurfaceSpec",
    "WavelengthSpec",
]
