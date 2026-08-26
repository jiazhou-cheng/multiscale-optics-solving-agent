"""Composing whole optical systems out of individually specified components (CHE-139).

``core.optical_system`` already owns the authoritative prescription schema, and
``solvers.optiland.builder`` is the one path from that schema to a built
``Optic``. Neither is changed here. What was missing between them is the step a
catalog user actually performs:

    three separately specified physical components
      -> ordering, orientation and air gaps
        -> ONE ``OpticalSystemSpec``
          -> the existing builder

``OpticalSystemSpec.surfaces`` is a flat sequence, so before CHE-139 the only
way to express "an achromat 50 mm in front of a bi-convex singlet" was to write
the concatenated surface list by hand, per system. That hand-concatenation is
exactly where a reversed lens, a dropped air gap or a misplaced glass boundary
hides, and it produces a *different optical system with no error*. This module
makes the composition itself a typed, testable value.

What a component is, and where its thickness stops
--------------------------------------------------
A :class:`ComponentSpec` is one physical part: ``n`` surfaces in the order light
meets them, ``n - 1`` internal axial thicknesses, and ``n - 1`` internal media.
A cemented doublet is three surfaces, two thicknesses (5.0 mm crown, 2.17 mm
flint) and two glasses. Crucially the component owns **no trailing thickness**:
the distance from its last vertex to whatever comes next is an *assembly*
parameter, not a property of the part, and keeping it out of the component is
what makes the same component reusable at any spacing.

``core.optical_system.SurfaceSpec`` instead carries the medium and the spacing
*after* each surface, because that is the sequential-trace convention the solver
speaks. :meth:`ComponentSpec.surfaces` is the one place the two conventions
meet.

Orientation
-----------
:meth:`ComponentSpec.reversed` flips a component end-for-end. Under the
reflection ``z -> -z``:

- surface order reverses, and so do the internal thicknesses and media;
- a spherical surface's radius changes sign -- the sag equation
  ``z = c r**2 / (1 + sqrt(1 - (1 + k) c**2 r**2))`` maps to ``-z`` under
  ``c -> -c`` with ``k`` untouched, so the conic constant does *not* change;
- a plane stays a plane.

Even-aspheric geometry would additionally need every polynomial coefficient
negated, and that transformation has no verification in this repository, so it
is **refused** rather than applied. Same for a grating: this layer builds
refractive components only, and a diffractive interaction cannot be flipped by
negating a radius. Both refusals are structured
:class:`~core.optical_system.PrescriptionError`\\ s, raised before any surface is
emitted.

Units are the schema's (``core.optical_system.UNITS``) throughout: millimetres
for every length, micrometres for wavelength, degrees for angular field.

Determinism
-----------
Every collection is an ordered tuple, every model forbids undeclared fields, and
nothing here consults a set, a dict ordering, a random source or the clock. The
emitted :class:`~core.optical_system.OpticalSystemSpec` therefore has the same
stable :meth:`~core.optical_system.OpticalSystemSpec.fingerprint` on every run.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.optical_system import (
    AirMaterialSpec,
    ApertureSpec,
    EvenAsphereGeometrySpec,
    FieldSpec,
    GeometryKind,
    GeometrySpec,
    MaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)

#: Identifier of the component/assembly schema. Bump the integer when a change
#: would make an existing serialized component mean something different.
OPTICAL_COMPONENT_SPEC_VERSION = "optical-component-spec/1"


class _Frozen(BaseModel):
    """Immutable, closed base model.

    Mirrors ``core.optical_system._Frozen`` deliberately, for the same reason
    stated there: ``extra='forbid'`` is what turns a misspelled key into an
    error instead of into a silently different optical system.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Orientation(StrEnum):
    """Which way round a component is installed.

    ``AS_SPECIFIED`` is the surface order the component was written in --
    normally the manufacturer's own, so a catalog page that says "steepest
    convex surface should face the infinite conjugate" is honoured by writing
    the component that way once. ``REVERSED`` is that component end-for-end.
    """

    AS_SPECIFIED = "as_specified"
    REVERSED = "reversed"


def _reverse_geometry(geometry: GeometrySpec, path: str) -> GeometrySpec:
    """The same surface seen from the other side.

    Radius sign flips, conic does not (see the module docstring). Anything whose
    reversal is not established by that argument alone is refused.
    """
    if isinstance(geometry, PlaneGeometrySpec):
        return PlaneGeometrySpec()
    if isinstance(geometry, EvenAsphereGeometrySpec):
        raise PrescriptionError(
            "COMPONENT_REVERSAL_UNSUPPORTED_GEOMETRY",
            "reversing an even-aspheric surface also negates every polynomial "
            "coefficient, and that transformation has no verification in this "
            "repository; it is refused rather than applied",
            path=path,
            expected="geometry kind 'plane' or 'spherical' on a reversed component",
            supported=(GeometryKind.PLANE.value, GeometryKind.SPHERICAL.value),
        )
    if isinstance(geometry, SphericalGeometrySpec):
        # Emitted in radius form regardless of which form arrived: one
        # normalized spelling means a reversed component has one fingerprint.
        return SphericalGeometrySpec(
            radius_mm=-geometry.resolved_radius_mm,
            conic=geometry.conic,
        )
    raise PrescriptionError(  # pragma: no cover - the union is closed
        "COMPONENT_REVERSAL_UNSUPPORTED_GEOMETRY",
        f"geometry kind {getattr(geometry, 'kind', geometry)!r} cannot be reversed here",
        path=path,
        supported=(GeometryKind.PLANE.value, GeometryKind.SPHERICAL.value),
    )


class ComponentSpec(_Frozen):
    """One physical optical component: ``n`` surfaces, ``n - 1`` internal media.

    The component deliberately does not know what follows it. See the module
    docstring for why the trailing air gap is an assembly parameter.

    ``clear_aperture_mm`` is recorded because a catalog publishes it and a
    reviewer needs it, **not** because it is enforced: ``optical-system-spec/1``
    has no per-surface aperture field, so no physical rim exists in the built
    system and no ray is vignetted by one. A caller that needs the constraint
    respected must keep the entrance pupil inside it and say so; a caller that
    needs it *enforced* needs a schema that can express it, which this module
    does not silently invent.
    """

    spec_version: str = Field(default=OPTICAL_COMPONENT_SPEC_VERSION, frozen=True)
    name: str = Field(min_length=1)
    geometries: tuple[GeometrySpec, ...]
    internal_thicknesses_mm: tuple[float, ...]
    internal_materials: tuple[MaterialSpec, ...]
    clear_aperture_mm: float | None = None
    surface_comments: tuple[str, ...] = ()
    description: str = ""

    @model_validator(mode="after")
    def _check_component(self) -> ComponentSpec:
        if self.spec_version != OPTICAL_COMPONENT_SPEC_VERSION:
            raise PrescriptionError(
                "COMPONENT_SCHEMA_VERSION_UNSUPPORTED",
                f"spec_version={self.spec_version!r} cannot be read by this build",
                path="spec_version",
                expected=f"{OPTICAL_COMPONENT_SPEC_VERSION!r}",
            )
        surface_count = len(self.geometries)
        if surface_count < 2:
            raise PrescriptionError(
                "COMPONENT_TOO_FEW_SURFACES",
                f"a refractive component needs at least two surfaces, got {surface_count}",
                path="geometries",
                expected="two or more surfaces, in the order light meets them",
            )
        for label, values in (
            ("internal_thicknesses_mm", self.internal_thicknesses_mm),
            ("internal_materials", self.internal_materials),
        ):
            if len(values) != surface_count - 1:
                raise PrescriptionError(
                    "COMPONENT_INTERNAL_LENGTH_MISMATCH",
                    f"{label} has {len(values)} entries for {surface_count} surfaces",
                    path=label,
                    expected=(
                        "exactly one entry per gap between consecutive surfaces, i.e. "
                        f"{surface_count - 1}; the distance AFTER the last surface is an "
                        "assembly parameter, not a component property"
                    ),
                )
        for index, thickness in enumerate(self.internal_thicknesses_mm):
            if not math.isfinite(thickness) or thickness <= 0.0:
                raise PrescriptionError(
                    "COMPONENT_INTERNAL_THICKNESS_INVALID",
                    f"internal_thicknesses_mm[{index}]={thickness!r} is not a finite "
                    "positive axial distance",
                    path=f"internal_thicknesses_mm[{index}]",
                    expected="a finite thickness > 0 in millimetres",
                )
        if self.clear_aperture_mm is not None and (
            not math.isfinite(self.clear_aperture_mm) or self.clear_aperture_mm <= 0.0
        ):
            raise PrescriptionError(
                "COMPONENT_CLEAR_APERTURE_INVALID",
                f"clear_aperture_mm={self.clear_aperture_mm!r} is not a finite positive diameter",
                path="clear_aperture_mm",
                expected="a finite diameter > 0 in millimetres, or null when unpublished",
            )
        if self.surface_comments and len(self.surface_comments) != surface_count:
            raise PrescriptionError(
                "COMPONENT_COMMENT_LENGTH_MISMATCH",
                f"surface_comments has {len(self.surface_comments)} entries for "
                f"{surface_count} surfaces",
                path="surface_comments",
                expected="either no comments at all, or exactly one per surface",
            )
        return self

    @property
    def surface_count(self) -> int:
        return len(self.geometries)

    @property
    def axial_thickness_mm(self) -> float:
        """Vertex-to-vertex thickness of the part itself, excluding any air gap."""
        return float(sum(self.internal_thicknesses_mm))

    def reversed(self) -> ComponentSpec:
        """This component installed end-for-end. See the module docstring."""
        return ComponentSpec(
            name=f"{self.name} (reversed)",
            geometries=tuple(
                _reverse_geometry(geometry, f"geometries[{index}]")
                # Enumerate the ORIGINAL order so a refusal names the surface the
                # caller wrote, not the position it would have ended up in.
                for index, geometry in reversed(list(enumerate(self.geometries)))
            ),
            internal_thicknesses_mm=tuple(reversed(self.internal_thicknesses_mm)),
            internal_materials=tuple(reversed(self.internal_materials)),
            clear_aperture_mm=self.clear_aperture_mm,
            surface_comments=tuple(reversed(self.surface_comments)),
            description=self.description,
        )

    def oriented(self, orientation: Orientation) -> ComponentSpec:
        """This component in the requested orientation."""
        if orientation is Orientation.AS_SPECIFIED:
            return self
        if orientation is Orientation.REVERSED:
            return self.reversed()
        raise PrescriptionError(  # pragma: no cover - the enum is closed
            "COMPONENT_ORIENTATION_UNKNOWN",
            f"orientation {orientation!r} is not implemented",
            path="orientation",
            supported=tuple(member.value for member in Orientation),
        )

    def surfaces(
        self,
        *,
        trailing_thickness_mm: float,
        orientation: Orientation = Orientation.AS_SPECIFIED,
        stop_surface_index: int | None = None,
    ) -> tuple[SurfaceSpec, ...]:
        """The component as ``SurfaceSpec``s, ready to concatenate.

        ``trailing_thickness_mm`` is the assembly's distance from this
        component's last vertex to whatever comes next -- the next component's
        first surface, or the image plane. It becomes the last surface's
        ``thickness_mm``, and that surface's medium is air.

        ``stop_surface_index`` indexes surfaces of the component *in the
        requested orientation*, so it names the surface a reviewer sees in the
        assembled sequence rather than one that moves when the part is flipped.
        """
        if not math.isfinite(trailing_thickness_mm) or trailing_thickness_mm < 0.0:
            raise PrescriptionError(
                "COMPONENT_TRAILING_THICKNESS_INVALID",
                f"trailing_thickness_mm={trailing_thickness_mm!r} is not a finite "
                "non-negative axial distance",
                path="trailing_thickness_mm",
                expected="a finite distance >= 0 in millimetres",
            )
        oriented = self.oriented(orientation)
        count = oriented.surface_count
        if stop_surface_index is not None and not 0 <= stop_surface_index < count:
            raise PrescriptionError(
                "COMPONENT_STOP_INDEX_OUT_OF_RANGE",
                f"stop_surface_index={stop_surface_index} is not a surface of "
                f"{oriented.name!r}, which has {count}",
                path="stop_surface_index",
                expected=f"an integer in [0, {count - 1}]",
            )
        comments = oriented.surface_comments or ("",) * count
        emitted: list[SurfaceSpec] = []
        for index, geometry in enumerate(oriented.geometries):
            is_last = index == count - 1
            emitted.append(
                SurfaceSpec(
                    geometry=geometry,
                    thickness_mm=(
                        trailing_thickness_mm
                        if is_last
                        else oriented.internal_thicknesses_mm[index]
                    ),
                    # The medium AFTER this surface: an internal medium inside the
                    # part, air once the part has been left.
                    material=(
                        AirMaterialSpec() if is_last else oriented.internal_materials[index]
                    ),
                    is_stop=index == stop_surface_index,
                    comment=comments[index],
                )
            )
        return tuple(emitted)


class ComponentPlacement(_Frozen):
    """One component installed in a system: which part, which way, how far on.

    ``air_gap_after_mm`` is measured from this component's last vertex to the
    next component's first vertex; on the last placement it is the distance to
    the image plane. It is air, and it is the only spacing the assembly owns --
    every other distance belongs to a component.
    """

    component: ComponentSpec
    orientation: Orientation = Orientation.AS_SPECIFIED
    air_gap_after_mm: float
    note: str = ""

    @model_validator(mode="after")
    def _check_placement(self) -> ComponentPlacement:
        if not math.isfinite(self.air_gap_after_mm) or self.air_gap_after_mm < 0.0:
            raise PrescriptionError(
                "PLACEMENT_AIR_GAP_INVALID",
                f"air_gap_after_mm={self.air_gap_after_mm!r} is not a finite "
                "non-negative axial distance",
                path="air_gap_after_mm",
                expected="a finite distance >= 0 in millimetres",
            )
        return self


def assemble_optical_system(
    *,
    name: str,
    placements: tuple[ComponentPlacement, ...],
    aperture: ApertureSpec,
    fields: tuple[FieldSpec, ...],
    wavelengths: tuple[WavelengthSpec, ...],
    object_distance_mm: float | None = None,
    stop_component_index: int = 0,
    stop_surface_index: int = 0,
    description: str = "",
) -> OpticalSystemSpec:
    """Concatenate placed components into one canonical prescription.

    The result is an ordinary :class:`~core.optical_system.OpticalSystemSpec`,
    which means it goes to the solver through the same single builder every
    other prescription in this repository uses. Nothing here touches Optiland.

    Raises:
        PrescriptionError: no placements, a stop index that names no surface, an
            invalid gap, or a component whose orientation cannot be represented.
    """
    if not placements:
        raise PrescriptionError(
            "ASSEMBLY_NO_COMPONENTS",
            "an assembled system needs at least one placed component",
            path="placements",
            expected="one or more placements, in the order light meets them",
        )
    if not 0 <= stop_component_index < len(placements):
        raise PrescriptionError(
            "ASSEMBLY_STOP_COMPONENT_OUT_OF_RANGE",
            f"stop_component_index={stop_component_index} names no placement; "
            f"there are {len(placements)}",
            path="stop_component_index",
            expected=f"an integer in [0, {len(placements) - 1}]",
        )

    surfaces: list[SurfaceSpec] = []
    for index, placement in enumerate(placements):
        surfaces.extend(
            placement.component.surfaces(
                trailing_thickness_mm=placement.air_gap_after_mm,
                orientation=placement.orientation,
                stop_surface_index=(
                    stop_surface_index if index == stop_component_index else None
                ),
            )
        )
    return OpticalSystemSpec(
        name=name,
        description=description,
        object_distance_mm=object_distance_mm,
        surfaces=tuple(surfaces),
        aperture=aperture,
        fields=fields,
        wavelengths=wavelengths,
    )


def component_surface_span(
    placements: tuple[ComponentPlacement, ...],
) -> tuple[dict[str, Any], ...]:
    """Which surfaces of the assembled sequence each placement contributed.

    Derived from the placements rather than recorded alongside them, so it
    cannot disagree with what :func:`assemble_optical_system` actually emitted.
    ``first_vertex_z_mm`` is measured from the first surface vertex of the
    assembly, which is the frame the builder places the system in.
    """
    spans: list[dict[str, Any]] = []
    surface_index = 0
    vertex_z = 0.0
    for order, placement in enumerate(placements):
        oriented = placement.component.oriented(placement.orientation)
        spans.append(
            {
                "order": order,
                "component_name": oriented.name,
                "orientation": placement.orientation.value,
                "first_surface_index": surface_index,
                "last_surface_index": surface_index + oriented.surface_count - 1,
                "surface_count": oriented.surface_count,
                "first_vertex_z_mm": vertex_z,
                "last_vertex_z_mm": vertex_z + oriented.axial_thickness_mm,
                "axial_thickness_mm": oriented.axial_thickness_mm,
                "air_gap_after_mm": placement.air_gap_after_mm,
                "clear_aperture_mm": oriented.clear_aperture_mm,
                "note": placement.note,
            }
        )
        surface_index += oriented.surface_count
        vertex_z += oriented.axial_thickness_mm + placement.air_gap_after_mm
    return tuple(spans)


def surface_table(spec: OpticalSystemSpec) -> tuple[dict[str, Any], ...]:
    """The assembled surface sequence, flattened for inspection.

    Read off the *emitted* prescription, not off the placements that produced
    it, so a reviewer comparing this against
    :func:`component_surface_span` is comparing two independent readings of the
    same assembly. ``vertex_z_mm`` accumulates ``thickness_mm`` from the first
    surface, matching the builder's own placement.
    """
    rows: list[dict[str, Any]] = []
    vertex_z = 0.0
    for index, surface in enumerate(spec.surfaces):
        geometry = surface.geometry
        radius: float | None
        if isinstance(geometry, PlaneGeometrySpec):
            radius = None
        else:
            radius = float(geometry.resolved_radius_mm)
        material = surface.material
        rows.append(
            {
                "index": index,
                "geometry_kind": geometry.kind.value,
                "radius_mm": radius,
                "conic": float(geometry.conic),
                "thickness_mm": float(surface.thickness_mm),
                "vertex_z_mm": vertex_z,
                "material_kind": material.kind.value,
                "material_name": getattr(material, "name", None),
                "refractive_index": getattr(material, "refractive_index", None),
                "is_stop": bool(surface.is_stop),
                "comment": surface.comment,
            }
        )
        vertex_z += float(surface.thickness_mm)
    return tuple(rows)


__all__ = [
    "OPTICAL_COMPONENT_SPEC_VERSION",
    "ComponentPlacement",
    "ComponentSpec",
    "Orientation",
    "assemble_optical_system",
    "component_surface_span",
    "surface_table",
]
