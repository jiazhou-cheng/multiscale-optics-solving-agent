"""A sequential ray-trace, stated as physical intent: a setup, and a source.

CHE-156 (R04), split by CHE-218 (R05.7). A solver solves a *problem*, and this
module says what a ray-trace problem is in units and conventions the project
owns. It names no ray tracer, imports no backend, and contains no Optiland API
concept -- not `Optic`, not `surface_group`, not `add_surface`, not a normalized
field coordinate. The translation into any one solver's construction calls
belongs to that solver's adapter (R05), which is the only place allowed to know
how Optiland spells a surface.

Two independent inputs, not one record
--------------------------------------
`OpticalSetup` is the optical configuration being traced through. `SourceSpec` is
a declared illumination. They are separate because a ray trace takes two
independent inputs, and because holding them in one record made the coupling
*executable*: the pinned solver normalizes an angular field against the largest
field the system declares, so tracing an existing system at a new field angle
meant editing the system. R05.6 then made the second input path real -- an
already-materialized `RayBundle` from `couplers.scalar_to_ray`, which has no field
angle and no object distance to give -- and a record that required both forced a
caller to invent them so that a lens could be built.

A source at that argument position is one of two things: this declarative
specification, or an existing `RayBundle`. They are alternatives at the same
position, not two functions over two system types. `RayTraceProblem` is gone
rather than aliased; the tree is pre-cutover and a compatibility wrapper is what
the clean-slate rule in `AGENTS.md` bans.

Three classes, not twenty
-------------------------
The reference implementation spent 20 classes in `core/optical_system.py` on this
same schema: three geometry classes, two interaction classes, three material
classes, an aperture class, a field class, a wavelength class, four kind enums, a
frozen base and an error type. Every one of them existed to hold two or three
fields and a validator.

Here the geometry *kinds* are the value of a field rather than a choice of class
-- a surface with no radius is a plane, a surface with a radius is a conic -- and
material is a `Material` mapping. What is lost is the ability to write
`isinstance(geometry, PlaneGeometrySpec)`; nothing did that except the builder's
own dispatch, which is one `if` either way.

Units, fixed by this schema and asserted by `tests/problems/test_ray_trace.py`
------------------------------------------------------------------------------
Millimetres for every geometric length -- radius, thickness, entrance pupil
diameter, object distance. Reciprocal millimetres for curvature. **Micrometres**
for wavelength, which is the one deliberate exception: it is the unit optical
prescriptions are written in, and the alternative is a schema whose numbers no
longer match the literature they were transcribed from. Degrees for field
angles.

These are prescription units, not the SI the representations use. A solver
adapter converts at its boundary, and `UNITS` below is exported so that
conversion can be written against a declared fact rather than an assumption.

What is deliberately not here
-----------------------------
* **No named prescription, and no way to ask for one.** There is no function
  that turns a prescription name into a lens, no list of supported names, and no
  concrete lens anywhere under `src/`. A caller builds the setup it wants, or
  reads a fixture from `tests/fixtures/systems.py`. A production catalog of three
  benchmark lenses is a catalog of three things the project happened to measure,
  and it made a solver call whose argument is a name this repository invented the
  shortest path for every caller.
* **No gratings, no odd aspheres, no decentres, tilts, coordinate breaks, GRIN,
  coatings or freeform sag.** Nothing in the new tree traces one yet. They are
  additions to this schema when a ticket needs them, not placeholders now.
  CHE-207 (R05.5) is what an addition looks like: even aspheric coefficients and
  the finite-object point source landed together with the solver support and the
  physical verification for both, rather than as fields waiting for one.
* **No serialization, fingerprint or schema version.** The reference schema had
  all three because prescriptions were data files loaded by name. These are
  values constructed in Python by the caller that needs them.
* **No field list and no wavelength list.** Both were dropped by CHE-218, and
  each was a capability decision rather than a cleanup. The field *set* existed
  only to give the pinned solver a `max_field` to normalize against, which is a
  construction detail and now lives at the adapter boundary; dropping it is what
  makes a setup traceable at a field angle no record enumerated in advance. The
  wavelength set existed only to select a primary, which was never the wavelength
  a trace evaluated at -- that was always a free value -- and the primary itself
  is a property of the setup's characterization, kept as
  `OpticalSetup.reference_wavelength_um`. Neither list ever reached a trace as a
  list: one solve is one field and one wavelength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

__all__ = [
    "MATERIAL_KINDS",
    "UNITS",
    "Material",
    "OpticalSetup",
    "SourceSpec",
    "SurfaceSpec",
]

#: The declared unit of every quantity in this schema, exported so a test or an
#: adapter can assert it instead of trusting prose.
UNITS: dict[str, str] = {
    "radius": "mm",
    "curvature": "1/mm",
    "thickness": "mm",
    "object_distance": "mm",
    "entrance_pupil_diameter": "mm",
    "wavelength": "um",
    "field_angle": "deg",
    # Index-dependent, which is why `aspheric_coefficients` carries no unit
    # suffix in its name: `aspheric_coefficients[i]` multiplies `r**(2*(i+1))`
    # and a single suffix would be wrong for every entry but one.
    "aspheric_coefficient": "mm**(1 - 2*(i+1)) for aspheric_coefficients[i]",
}

#: The three media a surface can be followed by.
MATERIAL_KINDS: tuple[str, ...] = ("air", "ideal", "catalog")


class Material(TypedDict):
    """The medium *following* a surface.

    A `TypedDict` rather than three classes discriminated on `kind`, which is
    what the reference implementation used: the three variants share no
    behaviour, nothing subclasses them, and the only code that ever branched on
    the type was the builder's dispatch.

    * `kind="air"` -- modelled air: unit index, lossless. Not an environmental
      air model; this project does not exercise one.
    * `kind="ideal"` -- a constant, dispersionless `refractive_index`,
      independent of any glass catalog.
    * `kind="catalog"` -- a named glass. `catalog` narrows which vendor
      collection it is drawn from.

    `expected_catalog_file` is the one field here that looks like solver detail
    and is not. A solver that resolves a bare glass name by fuzzy matching can
    return a *different glass* without erroring -- the measured case is two
    similarly-spelled names selecting rows from two different vendors, with up to
    seven rows surviving the filter before scoring
    (`pre-rewrite-2026-08-30:benchmarks/probes/optiland/system_construction_probe.py`).
    Recording which catalog row the prescription was transcribed against is
    material provenance: it is what lets an adapter turn a database change into
    an error instead of into a quietly different trace. An adapter that resolves
    glass differently is free to ignore it.
    """

    kind: Literal["air", "ideal", "catalog"]
    refractive_index: NotRequired[float]
    name: NotRequired[str]
    catalog: NotRequired[str | None]
    expected_catalog_file: NotRequired[str | None]


#: Which keys each material kind may carry, beyond `kind` itself.
#:
#: Checked rather than trusted, because a `TypedDict` is a static-analysis
#: annotation and disappears at runtime: `{"kind": "ideal", "refactive_index":
#: 1.5}` is a perfectly good dict. The reference implementation learned this the
#: expensive way at the layer below -- Optiland's `GeometryFactory` silently
#: filters keyword arguments it does not recognize down to the fields of the
#: selected geometry, so a misspelled prescription key reaching the solver
#: produces a different optical system with no error at all. Refusing unknown
#: keys here is the only place that can be caught.
_MATERIAL_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # kind: (required, optional)
    "air": (frozenset(), frozenset()),
    "ideal": (frozenset({"refractive_index"}), frozenset()),
    "catalog": (frozenset({"name"}), frozenset({"catalog", "expected_catalog_file"})),
}


def _check_material(material: Material, *, where: str) -> None:
    """Refuse a material that is misspelled, incomplete, or over-specified."""
    kind = material.get("kind")
    if kind not in _MATERIAL_KEYS:
        raise ValueError(
            f"{where}: material kind {kind!r} is not one of {list(MATERIAL_KINDS)}"
        )
    required, optional = _MATERIAL_KEYS[kind]
    keys = set(material) - {"kind"}
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        detail = []
        if unknown:
            detail.append(
                f"does not take {sorted(unknown)} -- a key a solver does not recognize is "
                "filtered out silently by the pinned ray tracer, which yields a different "
                "optical system and no error"
            )
        if missing:
            detail.append(f"needs {sorted(missing)}")
        raise ValueError(f"{where}: material kind {kind!r} " + "; and ".join(detail))
    if kind == "ideal":
        index = float(material["refractive_index"])
        if not math.isfinite(index) or index <= 0.0:
            raise ValueError(
                f"{where}: refractive_index={index!r} is not a finite positive index"
            )
    if kind == "catalog" and not str(material["name"]).strip():
        raise ValueError(f"{where}: a catalog material needs a non-empty glass name")


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    """One optical surface, and the medium and spacing that follow it.

    Minimality rule 1 -- a shared invariant across several fields. Curvature,
    conic and the following medium and spacing describe one interface; a radius
    given twice in two forms, or a thickness that is not a number, makes the
    surface silently a different one rather than an invalid one.

    Sequential convention, stated because getting it wrong is a whole-system
    error that traces successfully: `material` is the medium **after** this
    surface, so a lens in air is a surface carrying glass followed by a surface
    carrying air, and `thickness_mm` is the axial distance from this surface to
    the next -- on the last surface, to the image plane.

    Geometry is three fields rather than three classes:

    * neither `radius_mm` nor `curvature_per_mm` -- a **plane**. A plane is not a
      sphere of infinite radius here; `radius_mm` is simply absent, so `inf`
      never has to be represented.
    * exactly one of them -- a **conic** of that curvature. Both forms are
      accepted because prescriptions in the literature use both, and exactly one
      may be given so the stored value is never a derived-and-possibly-
      inconsistent duplicate. This is the reference implementation's
      radius/curvature resolution rule, reused unchanged.
    * `conic` -- 0 for a sphere, the conic constant otherwise.
    * `aspheric_coefficients` -- an **even** aspheric polynomial added to the
      conic sag. Empty (the default) is a plain conic; see below.

    The aspheric convention, which is measured rather than chosen
    ------------------------------------------------------------
    `sag(r) = conic_sag(r) + sum_i aspheric_coefficients[i] * r**(2*(i+1))`, so
    **the series starts at `r**2`** and `aspheric_coefficients[0]` is the `r**2`
    term, not the `r**4` term. That is the pinned solver's own convention, and it
    is worth stating loudly because the other reading is the common one: a
    prescription written as `A4, A6, A8` starting at `r**4` must be passed here as
    `(0.0, A4, A6, A8)`.

    Measured, not inferred
    (`pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/system_construction_probe.json`,
    case `even_asphere_sag_matches_analytic`, and re-measured for CHE-207): the
    observed sag matches the analytic series to 5.6e-17 mm, while assuming the
    series started at `r**4` is wrong by 1.1e-2 mm on the same surface -- four
    orders of magnitude apart, so the falsifier is not a tolerance away.

    A planar base with aspheric terms is a legitimate aspheric plate and is
    accepted: the sag is then the polynomial alone, verified exactly.
    """

    thickness_mm: float
    radius_mm: float | None = None
    curvature_per_mm: float | None = None
    conic: float = 0.0
    #: Even aspheric polynomial coefficients, outermost-first in the series
    #: `sum_i c[i] * r**(2*(i+1))`. A tuple rather than a list because
    #: `SurfaceSpec` is frozen and a list would make the frozen surface mutable
    #: through one of its fields -- the same reason `material` gets a fresh dict
    #: per surface rather than a shared default.
    aspheric_coefficients: tuple[float, ...] = ()
    # A fresh dict per surface, not a shared module-level constant: the dataclass
    # is frozen but a dict is not, so one shared default would let a caller who
    # mutated one surface's material change the default for every other.
    material: Material = field(default_factory=lambda: Material(kind="air"))
    comment: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.thickness_mm):
            raise ValueError(
                f"thickness_mm={self.thickness_mm!r} is not a finite axial spacing in mm"
            )
        if self.radius_mm is not None and self.curvature_per_mm is not None:
            raise ValueError(
                "a curved surface takes exactly one of radius_mm or curvature_per_mm; "
                "given both, the stored value is a duplicate that can disagree with itself"
            )
        if self.radius_mm is not None and (
            not math.isfinite(self.radius_mm) or self.radius_mm == 0.0
        ):
            raise ValueError(
                f"radius_mm={self.radius_mm!r} does not describe a curved surface; omit "
                "both radius_mm and curvature_per_mm for a plane"
            )
        if self.curvature_per_mm is not None and (
            not math.isfinite(self.curvature_per_mm) or self.curvature_per_mm == 0.0
        ):
            raise ValueError(
                f"curvature_per_mm={self.curvature_per_mm!r} does not describe a curved "
                "surface; omit both radius_mm and curvature_per_mm for a plane"
            )
        if not math.isfinite(self.conic):
            raise ValueError(f"conic={self.conic!r} is not finite")

        # Normalized to a tuple of floats before it is validated, so a caller who
        # passes a list -- which is what a prescription transcribed from a paper
        # looks like -- does not end up with a mutable field on a frozen surface.
        # `object.__setattr__` is how a frozen dataclass normalizes; `slots=True`
        # does not change that.
        try:
            coefficients = tuple(float(value) for value in self.aspheric_coefficients)
        except TypeError as exc:
            raise ValueError(
                f"aspheric_coefficients={self.aspheric_coefficients!r} is not a sequence of "
                "numbers; it is the even polynomial added to the conic sag, and () is a "
                "plain conic"
            ) from exc
        for index, value in enumerate(coefficients):
            if not math.isfinite(value):
                raise ValueError(
                    f"aspheric_coefficients[{index}]={value!r} is not finite; every "
                    "coefficient of the sag polynomial has to be a number"
                )
        object.__setattr__(self, "aspheric_coefficients", coefficients)

        _check_material(self.material, where="surface")

    @property
    def has_planar_base(self) -> bool:
        """Whether the *conic* term is flat -- neither radius form was given.

        Separate from `is_plane` because an aspheric plate has a planar base and is
        not a plane. This is the property a solver adapter needs when it decides
        what base radius to pass; `is_plane` is the one a caller needs when it asks
        whether the surface is flat.
        """
        return self.radius_mm is None and self.curvature_per_mm is None

    @property
    def has_aspheric_terms(self) -> bool:
        """Whether any aspheric coefficient is non-zero.

        An empty tuple and a tuple of zeros are the same surface -- a plain conic
        -- so both answer `False`. A prescription transcribed as
        `(0.0, 0.0, A8)` is genuinely aspheric and answers `True`.

        Stated as "any non-zero" rather than "non-empty" because the solver
        selects a *different geometry class* on the answer, and an all-zero
        polynomial must not change which class is built. Measured: a
        zero-coefficient even asphere and the standard surface of the same radius
        and conic agree **bitwise** in sag, traced position and accumulated path
        -- so the choice is safe either way today, and pinning it here means it
        stays safe if that ever stops being true.
        """
        return any(value != 0.0 for value in self.aspheric_coefficients)

    @property
    def is_plane(self) -> bool:
        """Whether this surface is flat: a planar base **and** no aspheric terms."""
        return self.has_planar_base and not self.has_aspheric_terms

    @property
    def resolved_radius_mm(self) -> float:
        """The base radius in mm, from whichever form was given. `inf` for a plane.

        The *base* radius: an aspheric surface's polynomial is not folded in here,
        because the two are separate arguments at the solver boundary.
        """
        if self.radius_mm is not None:
            return self.radius_mm
        if self.curvature_per_mm is not None:
            return 1.0 / self.curvature_per_mm
        return math.inf


@dataclass(frozen=True, slots=True)
class OpticalSetup:
    """The optical configuration a trace passes through. No illumination.

    CHE-218 (R05.7). This was the system half of `RayTraceProblem`, which held the
    optical system and the light in one record and named the second group outright
    in its own docstring. The coupling was executable rather than cosmetic: because
    the pinned solver normalizes an angular field against the largest field the
    system declares, "trace this system at 3 degrees" required *editing the optical
    system*, and an already-materialized `RayBundle` -- which has no field angle and
    no object distance at all -- forced a caller to invent both so that a lens could
    be built. `SourceSpec` is the other half, and the two are independent inputs to
    a trace.

    Minimality rule 1 -- the fields are jointly constrained, and rule 2 -- this is
    the public model a solver consumes. `stop_index` has to index `surfaces`, and
    neither means anything alone.

    **The aperture stop is an index, not a flag on a surface.** The reference
    schema put `is_stop: bool` on the surface and validated that exactly one was
    set, which made "no stop" and "three stops" representable states that a
    validator then rejected. One index cannot express either.

    Two surfaces are not listed, and both are fixed rather than parameterized: the
    object plane, in air, placed by the *source* (`SourceSpec.object_distance_mm`),
    and the image plane, in air, placed by the last listed surface's
    `thickness_mm`. A curved object or a tilted image plane is an extension to this
    schema, not something to smuggle through a field that already means something
    else.

    `reference_wavelength_um`, and why the one wavelength-shaped field stayed
    -----------------------------------------------------------------------
    **This is not illumination.** It is the wavelength at which this setup's own
    *paraxial characterization* is defined -- specifically the exit pupil, whose
    location and diameter a solver reads from the system rather than from the
    light. Every prescription in the literature states one, for the same reason:
    "the exit pupil is 3.05 mm before the image" is not a fact until a wavelength
    is named.

    It stayed because it was **measured** to matter, not because it was convenient
    to keep. On M3-REVERSE-TELEPHOTO the pinned solver evaluates `paraxial.XPL()`
    and `XPD()` at whichever declared wavelength is primary: at 0.5876 um they are
    -3.0545788978518327 mm and 0.46053493637581633 mm, and at 0.55 um they are
    -3.0550180932891653 mm and 0.4607610620693788 mm. So a setup that carried no
    reference wavelength would have to locate the exit pupil at whatever wavelength
    the trace happened to use, which is a different reference surface and a
    different frozen record.

    Note what that means and what it does not. The traced wavelength is the
    *source's* and is free -- it need not equal this one, and the frozen ray
    records deliberately trace M3-REVERSE-TELEPHOTO at 550 nm against a 587.6 nm
    reference. That the exit pupil is then located at the reference wavelength
    rather than the traced one is a real convention, and CHE-218 made it visible
    instead of changing it: it was already the behaviour, implied by
    `primary_wavelength_index`, and altering it would move a frozen number.
    """

    name: str
    surfaces: tuple[SurfaceSpec, ...]
    entrance_pupil_diameter_mm: float
    stop_index: int
    #: The wavelength this setup's paraxial characterization is defined at. No
    #: default: see the class docstring for the measurement that forced it to
    #: exist, which is also the reason a default would be an invented convention.
    reference_wavelength_um: float
    description: str = ""

    def __post_init__(self) -> None:
        problems: list[str] = []

        if not self.name.strip():
            problems.append("`name` is empty")
        if not self.surfaces:
            problems.append("`surfaces` is empty; a system needs at least one surface")
        elif not 0 <= self.stop_index < len(self.surfaces):
            problems.append(
                f"`stop_index`={self.stop_index} does not index the {len(self.surfaces)} "
                "surface(s); the aperture stop is one of the listed surfaces"
            )
        if not math.isfinite(self.entrance_pupil_diameter_mm) or (
            self.entrance_pupil_diameter_mm <= 0.0
        ):
            problems.append(
                f"`entrance_pupil_diameter_mm`={self.entrance_pupil_diameter_mm!r} is not a "
                "finite positive diameter in mm"
            )
        if not math.isfinite(self.reference_wavelength_um) or (
            self.reference_wavelength_um <= 0.0
        ):
            problems.append(
                f"`reference_wavelength_um`={self.reference_wavelength_um!r} is not a finite "
                "positive wavelength in micrometres. It is not the wavelength to trace at -- "
                "that belongs to the source -- it is the one this setup's exit pupil is "
                "located at, and there is no default because a wavelength nobody chose is "
                "an invented convention"
            )

        if problems:
            raise ValueError(
                f"optical setup {self.name!r} is not usable:\n  " + "\n  ".join(problems)
            )


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A declared illumination: which light, from where, in which direction.

    CHE-218 (R05.7). The other half of the split, and a **declaration** rather
    than a constructor: it says what the illumination is, and something else turns
    it into physical state. `docs/architecture_principles.md` §2 sanctions exactly
    this -- "the declaration of a source may live in `problems/`; the constructor
    that turns it into state is the source" -- which is why this is here and not in
    `sources/`.

    Minimality rule 1, and the shared invariant is the interesting part: **the
    meaning of `field_angle_deg` depends on `object_distance_mm`.** At infinity a
    field angle is a *direction*; at a finite distance it is a *position*. The two
    fields are not independently interpretable, which is what makes them one
    record rather than two arguments.

    One solve, one wavelength
    -------------------------
    `wavelength_um` is a scalar, not a list, and a sequence is refused rather than
    looped over. One solver invocation is a single-wavelength solve; a spectrum is
    several sources, evaluated separately. The record this replaced carried
    `wavelengths_um` plus a `primary_wavelength_index`, and the other N-1 entries
    never reached a trace -- they existed only to set the solver's primary
    wavelength, which is a property of the *setup*'s characterization and now lives
    there as `OpticalSetup.reference_wavelength_um`.

    The two source geometries, and why the difference is physical
    ------------------------------------------------------------
    `object_distance_mm is None` -- an object at **infinity**. Every ray of one
    field arrives with the same direction, and the incoming wavefront is a plane.

    `object_distance_mm = d` (finite, positive) -- a **point source**, i.e. a
    spherical wave. The source sits at
    `(-tan(x_deg) * d, -tan(y_deg) * d, -d)` in the traced frame, so the field
    angle of a finite-conjugate problem is a *position* rather than a direction,
    and it is the direction of the chief ray from that point through the stop.
    Measured to twelve digits for CHE-207 at three fields including one off axis in
    both x and y; the sign is negative, so a positive field angle places the source
    below the axis and its image above it.

    Every ray of one field then leaves that **single point**, which is what makes
    the launch state a common spherical wavefront: the optical path from that
    wavefront to each launch point is identically zero, because the launch point
    *is* the wavefront. `solvers/optiland/rays.py` states the consequence for the
    declared optical path, and the difference from the infinite case is not
    cosmetic -- there the origins spread over a plane and the directions are
    common, here the origin is common and the directions spread.

    Note that a finite `object_distance_mm` is generally coupled to the *setup*'s
    image spacing -- a conjugate pair is a conjugate pair -- but that coupling is
    the caller's arithmetic, not a field on either record. `tests/fixtures/systems.py`
    is where a matched pair is written down.
    """

    #: Vacuum wavelength in micrometres. One value; see the class docstring.
    wavelength_um: float
    #: `(x_deg, y_deg)`. A direction at infinity, a position at a finite distance.
    field_angle_deg: tuple[float, float] = (0.0, 0.0)
    #: How far the point source sits before the first surface, or `None` for an
    #: object at infinity.
    object_distance_mm: float | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []

        wavelength = self.wavelength_um
        if isinstance(wavelength, list | tuple | set | frozenset):
            problems.append(
                f"`wavelength_um`={wavelength!r} is a sequence of {len(wavelength)}. One "
                "solver invocation is a SINGLE-WAVELENGTH solve, so this field is one "
                "value and a spectrum is several sources evaluated separately -- it is "
                "not a list to be looped over. Passing several here would silently trace "
                "one of them"
            )
        else:
            try:
                value = float(wavelength)
            except (TypeError, ValueError):
                problems.append(
                    f"`wavelength_um`={wavelength!r} is not a wavelength in micrometres"
                )
            else:
                if not math.isfinite(value) or value <= 0.0:
                    problems.append(
                        f"`wavelength_um`={wavelength!r} is not a finite positive wavelength "
                        "in micrometres"
                    )
                else:
                    object.__setattr__(self, "wavelength_um", value)

        angles = self.field_angle_deg
        # Deliberately duck-typed rather than `isinstance(v, int | float)`: a
        # numpy float32 is not a Python float and a transcribed field angle is
        # legitimately an int. What is being checked is "two things that are
        # finite numbers", which is what `math.isfinite` answers -- and a pair of
        # *pairs*, or a two-character string, raises `TypeError` there and is
        # refused for it. That is how a list of fields is caught.
        try:
            valid = len(angles) == 2 and all(math.isfinite(value) for value in angles)
        except TypeError:
            valid = False
        if not valid:
            problems.append(
                f"`field_angle_deg`={angles!r} is not a finite (x_deg, y_deg) pair. One "
                "field per solve, in degrees: a list of fields is several sources"
            )
        else:
            object.__setattr__(
                self, "field_angle_deg", (float(angles[0]), float(angles[1]))
            )

        if self.object_distance_mm is not None and (
            not math.isfinite(self.object_distance_mm) or self.object_distance_mm <= 0.0
        ):
            problems.append(
                f"`object_distance_mm`={self.object_distance_mm!r} is not a finite "
                "positive distance in mm; it is how far the point source sits BEFORE the "
                "first surface, so zero would put the source on that surface and leave no "
                "object space at all. Use None for an object at infinity."
            )

        if problems:
            raise ValueError("source is not usable:\n  " + "\n  ".join(problems))

    @property
    def object_at_infinity(self) -> bool:
        return self.object_distance_mm is None
