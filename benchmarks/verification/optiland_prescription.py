"""Read a constructed Optiland `Optic` back into this project's `OpticalSetup`.

CHE-239 (workstream A) §A.1: *"Use canonical `optiland.samples.*` prescriptions
directly when a tutorial uses them -- read the prescription out of the sample
class into the setup payload. Do not hand-transcribe canonical sample lenses."*

This is the inverse of `backends.optiland.system.build_lens`, and it exists so
that the question workstream A asks -- *can this project's schema express the
system the tutorial builds, and does the supported observable agree?* -- is
answered against the tutorial's own lens rather than against a transcription of
it. A transcription would make the transcriber the thing under test.

What "expressible" means here
-----------------------------
`problems.OpticalSetup` is a deliberately small schema (see its docstring: no
gratings, no tilts, no coordinate breaks, no freeform sag, no reflective flag).
So the extractor's real output is not a setup -- it is a **verdict**, and the
setup only when the verdict is "yes". Every refusal is raised as `Unexpressible`
carrying a machine-readable `category`, because CHE-239 §A.2 requires a
documented refusal to be distinguishable from a failure.

The refusal categories, all of them measured against the 29 canonical samples
rather than imagined:

| category | what it is | seen in |
| --- | --- | --- |
| `geometry` | a surface class `SurfaceSpec` has no field for | none of the 29 |
| `reflective` | a mirror; the schema has only refracting interfaces | `HubbleTelescope`,
  `UVReflectingMicroscope` |
| `tilt_decenter` | a non-identity surface coordinate system | none of the 29 |
| `aperture_definition` | an aperture that is not an entrance pupil diameter |
  `NavarroWideAngleEye` (`float_by_stop_size`) |
| `field_definition` | a field type that is not an angle | `UVProjectionLens`, tutorial 4e |
| `finite_object` | the object plane is at a finite distance | -- |
| `no_stop` | no surface is the stop | -- |
| `construction` | the sample or notebook cell did not build a lens at all | -- |

`UVProjectionLens` declares **both** `objectNA` and an `ObjectHeightField`, and it
is tabled under `field_definition` because that is the check that runs first --
the category a caller sees is the first refusal, not the worst one.

`imageFNO` is **not** in that table, and the reason is the one interesting
judgement call in this module. An image-space F-number and an entrance pupil
diameter are different declarations of the same aperture, and 18 of the 29
canonical samples use the former while `build_lens` can only set the latter. The
extractor therefore reads `optic.paraxial.EPD()` off the *sample's own*
constructed lens and declares that -- Optiland's own answer to "what entrance
pupil does this F-number imply", not an arithmetic conversion invented here. The
result is recorded as `aperture_conversion` on the provenance mapping.

**What does not check the conversion**, stated because the obvious check is
vacuous: comparing `paraxial.EPD()` or `paraxial.FNO()` between the original and
the rebuilt lens proves nothing. Read from the pinned 0.6.0,
`ImageFNOAperture.compute_epd` is `f2() / FNO` and `EPDAperture.compute_epd` is
the stored value, while `Paraxial.FNO()` returns the declared F-number directly
for an `imageFNO` lens and `f2() / EPD()` otherwise -- so the EPD delta is
identically zero by construction and the FNO delta reduces exactly to the EFL
residual already reported beside it. What *does* check the conversion is the ray
regression in `ray_tier2`: if the rebuilt aperture admitted a different pupil,
the traced intersection coordinates would differ, and they agree to 0.0.

Units
-----
Optiland's native lengths are millimetres and its wavelengths micrometres, which
is exactly `problems.ray_trace.UNITS`. So nothing is converted here and no scale
factor appears below; `backends/optiland/system.py::_require_native_units`
asserts that the pinned package still agrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from problems import UNAPERTURED, Material, OpticalSetup, SurfaceSpec

__all__ = [
    "REFUSAL_CATEGORIES",
    "Unexpressible",
    "extract_setup",
    "sample_fields_deg",
    "sample_wavelengths_um",
]

#: Every reason this module refuses to produce a setup. A closed set, so a
#: driver can aggregate by category instead of by message text.
REFUSAL_CATEGORIES: tuple[str, ...] = (
    "geometry",
    "reflective",
    "tilt_decenter",
    "aperture_definition",
    "field_definition",
    "finite_object",
    "no_stop",
    "construction",
)


class Unexpressible(Exception):
    """`problems.OpticalSetup` cannot express this lens, and here is precisely why.

    A class rather than a `ValueError` with a formatted message: CHE-239 §A.2
    makes "documented refusal" a *result status* distinct from failure, and a
    caller that has to regex-match a message to tell the two apart will
    eventually get it wrong. `category` is from `REFUSAL_CATEGORIES`; `detail`
    is the sentence a human reads.
    """

    def __init__(self, category: str, detail: str) -> None:
        if category not in REFUSAL_CATEGORIES:
            raise ValueError(f"category={category!r} is not one of {list(REFUSAL_CATEGORIES)}")
        super().__init__(f"[{category}] {detail}")
        self.category = category
        self.detail = detail


@dataclass(frozen=True)
class _Geometry:
    """The three sag fields `SurfaceSpec` takes, pulled off one native geometry."""

    radius_mm: float | None
    conic: float
    aspheric_coefficients: tuple[float, ...]


def _float(value: Any) -> float:
    """A native scalar as a host float.

    Optiland's backend hands back a 0-d array on numpy and a tensor on torch, and
    `float()` accepts both. Written once so no call site has to remember it.
    """
    return float(value)


def _geometry_of(surface: Any, *, where: str) -> _Geometry:
    """Which of `SurfaceSpec`'s three sag fields this native geometry maps onto.

    Dispatch is on the *class name* rather than `isinstance`, because importing
    the six geometry classes to compare against would pull the whole geometry
    package in for what is a three-way branch, and because an unrecognized name
    is the answer this function most needs to give well.

    Only three classes appear across the 29 canonical samples -- `Plane`,
    `StandardGeometry`, `EvenAsphere` -- which is measured, not assumed; the
    survey is in the workstream's record. Everything else refuses.
    """
    name = type(surface.geometry).__name__
    geometry = surface.geometry

    if name == "Plane":
        return _Geometry(radius_mm=None, conic=0.0, aspheric_coefficients=())

    if name in ("StandardGeometry", "EvenAsphere"):
        radius = _float(geometry.radius)
        # A `StandardGeometry` of infinite radius *is* a plane, and `SurfaceSpec`
        # refuses a non-finite `radius_mm` outright -- its docstring's rule is
        # that a plane omits both radius forms so `inf` never has to be
        # represented. Mapping it here is a translation, not a change of system.
        radius_mm = None if not math.isfinite(radius) else radius
        conic = _float(getattr(geometry, "k", 0.0) or 0.0)
        coefficients: tuple[float, ...] = ()
        if name == "EvenAsphere":
            # `order` is Optiland's own statement of where the series starts:
            # `order=2` means `coefficients[0]` multiplies `r**2`, which is
            # exactly `SurfaceSpec.aspheric_coefficients`' documented convention.
            # Checked rather than trusted, because the other reading -- a series
            # starting at `r**4` -- is the common one in the literature and
            # `SurfaceSpec`'s own docstring records it as wrong by four orders of
            # magnitude on the same surface.
            order = int(getattr(geometry, "order", 2))
            if order != 2:
                raise Unexpressible(
                    "geometry",
                    f"{where}: EvenAsphere declares order={order}; SurfaceSpec's polynomial "
                    "starts at r**2 (order 2) and a different start is a different surface",
                )
            coefficients = tuple(_float(value) for value in geometry.coefficients)
        return _Geometry(radius_mm=radius_mm, conic=conic, aspheric_coefficients=coefficients)

    raise Unexpressible(
        "geometry",
        f"{where}: surface geometry {name!r} has no SurfaceSpec field. The schema carries a "
        "conic radius, a conic constant and an even aspheric polynomial, and nothing else",
    )


def _material_after(surface: Any, *, where: str) -> Material:
    """The medium following this surface, as a `problems.Material`.

    Three native shapes map onto the three `MATERIAL_KINDS`:

    * `IdealMaterial` at unit index -> `air`. Modelled air is what
      `build_lens` puts back, so this round-trips.
    * `IdealMaterial` at any other index -> `ideal`, carrying that index.
    * `Material` (a catalog row) -> `catalog`, carrying the glass name and the
      database file it resolved to. The filename goes on `expected_catalog_file`
      for the reason `problems.Material` gives: a solver that resolves a bare
      glass name by fuzzy matching can silently return a *different glass*, and
      recording which row the prescription came from is what turns that into an
      error instead of a quietly different trace.
    """
    material = surface.material_post
    name = type(material).__name__

    if name == "IdealMaterial":
        index = _float(material.n(0.55))
        if index == 1.0:
            return Material(kind="air")
        return Material(kind="ideal", refractive_index=index)

    if name == "Material":
        glass = str(getattr(material, "name", "") or "")
        if not glass.strip():
            raise Unexpressible(
                "geometry", f"{where}: a catalog material with no glass name is not transcribable"
            )
        # The *database-relative* path and not `material.filename`, which is
        # absolute and therefore site-specific: `build_lens` compares what it
        # resolved against this string, and an absolute path would make every
        # extracted setup refuse to rebuild on a machine with a different
        # site-packages prefix. Measured: `material.filename` is
        # `/usr/.../optiland/database/data-nk/glass/hikari/SK16.yml` where the
        # backend reports `glass/hikari/SK16.yml`.
        data = getattr(material, "material_data", None) or {}
        # `reference` is the vendor the prescription narrowed to, and dropping it
        # is not a simplification -- it is a different glass. Measured on
        # `CookeTriplet`, which declares `material=("F2", "schott")`: re-resolving
        # the bare name `F2` with no vendor returns `glass/hikari/F2.yml` at
        # similarity score **0**, the same exactness score the schott row gets. So
        # Optiland's own exactness criterion does not distinguish them and only the
        # recorded vendor does. `problems.Material.catalog` is that vendor and
        # `expected_catalog_file` is the row it has to land on.
        return Material(
            kind="catalog",
            name=glass,
            catalog=str(getattr(material, "reference", "") or "") or None,
            expected_catalog_file=str(data.get("filename", "") or "") or None,
        )

    # Anything else -- an `AbbeMaterial`, a `MaterialFile` subclass this branch
    # does not know, a gradient index -- is refused rather than approximated by
    # its index at one wavelength, which would silently drop the dispersion.
    raise Unexpressible(
        "geometry",
        f"{where}: material class {name!r} is not one of the three problems.MATERIAL_KINDS. "
        "Approximating it by its index at one wavelength would drop its dispersion silently",
    )


def _clear_semi_diameter(surface: Any) -> float | str:
    """The surface's physical rim in mm, or `UNAPERTURED`.

    Only a `RadialAperture` with a finite outer radius maps onto
    `SurfaceSpec.clear_semi_diameter_mm`, which is a single clear **semi**-
    diameter. A radial aperture with an inner radius is an *obscuration* -- a
    central stop -- and the schema has no field for one, so it is left
    unapertured and the caller sees the rim it does declare go missing. That
    would be a silent change of system, so it refuses instead.
    """
    aperture = getattr(surface, "aperture", None)
    if aperture is None:
        return UNAPERTURED

    r_max = getattr(aperture, "r_max", None)
    r_min = getattr(aperture, "r_min", None)
    if r_min is not None and _float(r_min) > 0.0:
        raise Unexpressible(
            "geometry",
            f"surface carries a central obscuration (r_min={_float(r_min)} mm); SurfaceSpec has "
            "one clear semi-diameter and no inner radius, so the obscuration would vanish",
        )
    if r_max is None:
        raise Unexpressible(
            "geometry",
            f"surface aperture {type(aperture).__name__!r} declares no outer radius, so it is "
            "not a clear semi-diameter",
        )
    return _float(r_max)


def _require_untilted(surface: Any, *, where: str) -> None:
    """Refuse a surface whose coordinate system is not the identity.

    `OpticalSetup` lists surfaces along one axis with a scalar spacing between
    them; there is no field for a decentre, a tilt or a coordinate break, and
    `z` is the spacing itself and so is not checked here.
    """
    cs = surface.geometry.cs
    offsets = {
        axis: _float(getattr(cs, axis, 0.0) or 0.0) for axis in ("x", "y", "rx", "ry", "rz")
    }
    nonzero = {axis: value for axis, value in offsets.items() if value != 0.0}
    if nonzero:
        raise Unexpressible(
            "tilt_decenter",
            f"{where}: the surface is decentred or tilted ({nonzero}); OpticalSetup places "
            "surfaces on one axis and has no field for either",
        )


def _require_refracting(surface: Any, *, where: str) -> None:
    """Refuse a mirror.

    `SurfaceSpec` has no reflective flag and `build_lens` sets none, so a
    reflective surface would be rebuilt as a refracting one -- a different
    system that traces perfectly well and reports a plausible spot.
    """
    model = getattr(surface, "interaction_model", None)
    if bool(getattr(model, "is_reflective", False)):
        raise Unexpressible(
            "reflective",
            f"{where}: the surface is reflective. SurfaceSpec carries only refracting "
            "interfaces, so rebuilding it would silently turn a mirror into a lens",
        )


def _entrance_pupil_diameter(optic: Any) -> tuple[float, str | None]:
    """`(EPD in mm, how it was obtained)`.

    `build_lens` sets `aperture_type="EPD"`, so a lens declaring anything else
    has to have an entrance pupil diameter derived for it. Only one derivation is
    used and it is not arithmetic done here: `optic.paraxial.EPD()` is Optiland's
    own answer for the lens as the sample built it. See the module docstring for
    why that is a conversion rather than an invention, and the driver for the
    residual it is checked against.

    `objectNA` and `float_by_stop_size` refuse instead. Both are aperture
    definitions whose meaning depends on the object or the stop rather than on
    the entrance pupil, so an EPD read off one lens does not reconstruct the
    same aperture when the rebuilt lens re-solves it.
    """
    aperture = optic.aperture
    kind = str(aperture.ap_type)
    if kind == "EPD":
        return _float(aperture.value), None
    if kind == "imageFNO":
        return _float(optic.paraxial.EPD()), f"imageFNO={_float(aperture.value)} -> paraxial.EPD()"
    raise Unexpressible(
        "aperture_definition",
        f"aperture_type={kind!r} is not an entrance pupil diameter and does not reduce to one: "
        "its meaning depends on the object or the stop, so an EPD read off this lens would not "
        "reconstruct the same aperture once the rebuilt lens re-solves it. build_lens declares "
        "'EPD' and only 'EPD'",
    )


def extract_setup(optic: Any, *, name: str) -> tuple[OpticalSetup, dict[str, Any]]:
    """`(the setup this lens is, what had to be interpreted to say so)`.

    Raises:
        Unexpressible: the schema cannot express this lens. `category` says which
            of `REFUSAL_CATEGORIES` applies.
    """
    surfaces = list(optic.surfaces.surfaces)
    if len(surfaces) < 3:
        raise Unexpressible(
            "construction",
            f"{name}: the lens has {len(surfaces)} surface(s) counting the object and image "
            "planes, so there is no optical surface between them",
        )

    # Optiland's list is [object, ...optical..., image]; `OpticalSetup.surfaces`
    # is the middle, because the object plane is placed by the source and the
    # image plane by the last listed thickness.
    optical = surfaces[1:-1]
    positions = [_float(value) for value in optic.surfaces.positions.ravel()]

    # The object plane, which `OpticalSetup` does not list because the *source*
    # places it. A lens whose object surface has a finite thickness is a
    # finite-conjugate system, and every driver here constructs its `SourceSpec`
    # with the default `object_distance_mm=None`. Extracting one without refusing
    # would therefore substitute an infinite-conjugate system silently -- the same
    # class of substitution the reflective and tilt gates exist for. Not reached
    # by any of the 29 canonical samples (only `UVProjectionLens` has a finite
    # object, at 110.85883544 mm, and it refuses on its field type first), which
    # is exactly why it is a gate rather than a comment.
    object_thickness = _float(positions[1] - positions[0])
    if math.isfinite(object_thickness):
        raise Unexpressible(
            "finite_object",
            f"{name}: the object plane is {object_thickness} mm before the first surface, so "
            "this is a finite-conjugate system. OpticalSetup does not carry the object "
            "distance -- SourceSpec.object_distance_mm does -- and every caller here declares "
            "an object at infinity, so extracting it would trace a different system. Note that "
            "the finite-conjugate path DOES exist -- SourceSpec.object_distance_mm and "
            "backends.optiland.solver:trace both support it -- and what is missing is a verified "
            "object-height-to-field-angle conversion, which CHE-239 forbids inventing here",
        )

    field_definition = type(optic.fields.field_definition).__name__
    if field_definition != "AngleField":
        raise Unexpressible(
            "field_definition",
            f"{name}: the lens declares field type {field_definition!r}; build_lens declares "
            "'angle', and SourceSpec's field_angle_deg is an angle at infinity or a position "
            "at a finite distance -- neither is an object or image height",
        )

    entrance_pupil_mm, aperture_conversion = _entrance_pupil_diameter(optic)

    specs: list[SurfaceSpec] = []
    stop_index: int | None = None
    for offset, surface in enumerate(optical):
        where = f"{name}: surfaces[{offset}]"
        _require_refracting(surface, where=where)
        _require_untilted(surface, where=where)
        geometry = _geometry_of(surface, where=where)
        # positions[] is indexed over the *native* list, so this surface is at
        # `offset + 1` and its thickness runs to the next entry -- which for the
        # last optical surface is the image plane, exactly as SurfaceSpec defines.
        thickness_mm = positions[offset + 2] - positions[offset + 1]
        specs.append(
            SurfaceSpec(
                thickness_mm=thickness_mm,
                radius_mm=geometry.radius_mm,
                conic=geometry.conic,
                aspheric_coefficients=geometry.aspheric_coefficients,
                clear_semi_diameter_mm=_clear_semi_diameter(surface),
                material=_material_after(surface, where=where),
                comment=str(getattr(surface, "comment", "") or ""),
            )
        )
        if bool(getattr(surface, "is_stop", False)):
            if stop_index is not None:
                raise Unexpressible(
                    "no_stop",
                    f"{name}: surfaces {stop_index} and {offset} are both the stop; "
                    "OpticalSetup.stop_index is one index and cannot express two",
                )
            stop_index = offset

    if stop_index is None:
        raise Unexpressible(
            "no_stop",
            f"{name}: no optical surface is the aperture stop, and OpticalSetup.stop_index has "
            "to name one of the listed surfaces",
        )

    primary = [
        _float(wavelength.value)
        for wavelength in optic.wavelengths.wavelengths
        if bool(wavelength.is_primary)
    ]
    if len(primary) != 1:
        raise Unexpressible(
            "construction",
            f"{name}: the lens declares {len(primary)} primary wavelength(s); the setup's "
            "reference wavelength is exactly one and there is no default",
        )

    setup = OpticalSetup(
        name=name,
        surfaces=tuple(specs),
        entrance_pupil_diameter_mm=entrance_pupil_mm,
        stop_index=stop_index,
        reference_wavelength_um=primary[0],
        description=f"extracted from a constructed optiland Optic by {__name__}",
    )
    provenance: dict[str, Any] = {
        "native_surface_count": len(surfaces),
        "optical_surface_count": len(optical),
        "native_aperture": {
            "type": str(optic.aperture.ap_type),
            "value": _float(optic.aperture.value),
        },
        "aperture_conversion": aperture_conversion,
        "native_field_definition": field_definition,
        "native_fields_deg": sample_fields_deg(optic),
        "native_wavelengths_um": sample_wavelengths_um(optic),
        "geometry_classes": [type(surface.geometry).__name__ for surface in optical],
        "surface_apertures_declared": sum(
            1 for surface in optical if getattr(surface, "aperture", None) is not None
        ),
    }
    return setup, provenance


def sample_fields_deg(optic: Any) -> list[tuple[float, float]]:
    """Every field the lens declares, as `(x_deg, y_deg)`.

    `SourceSpec` takes one field per solve, so a lens declaring three is three
    sources. This is what a driver enumerates over rather than what it passes.
    """
    return [(_float(field.x), _float(field.y)) for field in optic.fields.fields]


def sample_wavelengths_um(optic: Any) -> list[float]:
    """Every wavelength the lens declares, in micrometres. One solve is one of them."""
    return [_float(wavelength.value) for wavelength in optic.wavelengths.wavelengths]
