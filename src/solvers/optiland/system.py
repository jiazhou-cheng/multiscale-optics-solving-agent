"""Neutral setup in, native Optiland lens out, through one generic path.

CHE-179 (R05.1), CHE-218 (R05.7). The construction half of the anti-corruption
layer: `problems.OpticalSetup` -> `optiland.optic.Optic`. One function,
`build_lens`, and adding a system means handing it a different setup -- never
writing a builder.

The source is a construction argument, not a system property
------------------------------------------------------------
`build_lens(setup, source)`. R05.7 split the illumination out of the system
record, and this is where the consequence lands: the pinned backend needs an
object surface and at least one declared field before an `Optic` exists at all,
so those come from *what the caller asked to trace* rather than from a list the
caller had to declare in advance. Two facts make that safe rather than merely
tidier, and both were measured on M3-REVERSE-TELEPHOTO:

* **the declared field set does not affect the system.** Declaring one field at
  6 deg instead of three at 0/21/30 deg leaves `EPD`, `XPL` and `XPD` bitwise
  identical and changes only `max_field`, which is the normalization the trace
  divides by and multiplies back. So the field list was never a system property,
  and dropping it is what lets a setup be traced at a field angle nothing
  enumerated in advance;
* **the declared wavelength set does**, through the primary: `XPL` moves from
  -3.0545788978518327 mm to -3.0550180932891653 mm between primaries 0.5876 and
  0.55 um. That is why `OpticalSetup.reference_wavelength_um` exists and why it
  is what this module declares -- see that class's docstring. The wavelength the
  trace is *evaluated* at is the source's and is passed to the trace call, not
  declared here; it never had to be a declared wavelength, and the frozen ray
  records deliberately trace outside the declared set.

`source=None` is not a defaulted illumination. It means no illumination was
declared, which is the R05.6 path: the caller supplies its own `RayBundle`, the
object surface is skipped, and no field is aimed at. The two surfaces the backend
requires anyway are still built, because an `Optic` cannot exist without them.

What is deliberately absent
---------------------------
There is no per-system builder, no builder class, and no name-to-system
resolution. The reference implementation's `_resolve_lens(spec)` read as a lookup
and was a one-line call to the generic builder; the indirection is gone and the
generic path is called directly. There is no list of supported prescription
names, because R04 removed the prescription catalog from production: a caller
constructs the `OpticalSetup` it wants.

Why this validates before it constructs
---------------------------------------
Optiland's `GeometryFactory.create` filters `**kwargs` down to the fields of the
geometry config it selected, and **discards the rest with no error at all**
(`pre-rewrite-2026-08-30:benchmarks/probes/optiland/system_construction_probe.py`,
case `surface_kwargs_are_silently_filtered`). Passing a prescription straight
through would therefore turn a caller's mistake into a silently *different*
optical system. So the surface type is chosen here, the exact keyword set for
that type is assembled here, and everything that can fail is resolved before the
first `Optic` is touched -- a rejected problem never leaves a half-built lens.

`problems.ray_trace` closes the other half of the same hazard: `_check_material`
refuses a material key the schema does not define, which is the only layer that
can catch a misspelling before it reaches the filter.

Units
-----
`problems.UNITS` and Optiland's native units agree, so there is no conversion in
this module at all: millimetres for every geometric length, micrometres for
wavelength, degrees for angular fields. That agreement is *asserted* rather than
assumed -- `_require_native_units` checks the problem schema's declared units
against what is passed here, so a schema change that silently rebased radius on
metres fails at construction instead of scaling `k * OPL` by a thousand
downstream. The metre boundary is `rays.py`'s, on the way out.

Determinism
-----------
Surfaces are created in problem order from a tuple; materials resolve through one
explicit per-kind branch; the stop is the single index the schema guarantees;
aperture, fields and wavelengths are configured in one fixed order with every
solver-side default stated rather than inherited. No set iteration, no RNG, no
clock.
"""

from __future__ import annotations

from typing import Any

from problems import UNITS, Material, OpticalSetup, SourceSpec, SurfaceSpec

__all__ = [
    "FIELD_VIGNETTING_FACTORS",
    "FIELD_WEIGHT",
    "NATIVE_UNITS",
    "WAVELENGTH_WEIGHT",
    "build_lens",
]

#: The two Optiland `surface_type` strings this builder emits, and nothing else.
#:
#: `standard` is the conic path: the pinned factory returns a `Plane` for an
#: infinite radius and a `StandardGeometry` otherwise, which is how every conic
#: problem the schema can state spells a surface.
#:
#: `even_asphere` is selected only when `SurfaceSpec.has_aspheric_terms` -- a
#: *non-zero* coefficient, not merely a non-empty tuple. An all-zero polynomial
#: must not change which geometry class is built, and measured for CHE-207 it does
#: not have to: a zero-coefficient `even_asphere` and the `standard` surface of the
#: same radius and conic agree **bitwise** in sag, in traced position and in
#: accumulated optical path. So the selection is safe either way today, and
#: choosing `standard` keeps it safe if that stops being true -- which is what
#: keeps the frozen collimated benchmarks bit-identical by construction rather
#: than by an equality this module would be relying on.
#:
#: Gratings are still absent: `problems.SurfaceSpec` cannot express one, so a
#: third string here would be a branch nothing can reach.
_SURFACE_TYPE_STANDARD = "standard"
_SURFACE_TYPE_EVEN_ASPHERE = "even_asphere"

#: The wavelength unit Optiland records on every wavelength it stores. Passed
#: explicitly so the schema's micrometre contract is stated at the boundary
#: rather than inherited from the solver's default.
_WAVELENGTH_UNIT = "um"

#: Optiland's own name for an entrance-pupil-diameter aperture, and for a field
#: given as an angle. The two strings the problem schema's `UNITS` imply.
_APERTURE_TYPE = "EPD"
_FIELD_TYPE = "angle"

#: Vignetting factors applied to every field. This project does not model
#: vignetting compression; zero means none, and it is set rather than inherited
#: because a non-zero default in a future release would change the traced system.
FIELD_VIGNETTING_FACTORS: tuple[float, float] = (0.0, 0.0)

#: Per-field and per-wavelength weights. One field and one wavelength are traced
#: per call and no weighted merit function is formed, so every weight is unity.
FIELD_WEIGHT = 1.0
WAVELENGTH_WEIGHT = 1.0

#: The unit each `problems.UNITS` entry must declare for this module to pass the
#: problem through unconverted. Checked, not assumed: mm-for-m is the error that
#: scales `k * OPL` by 1000 and shows up downstream as dense phase wraps, and it
#: is a one-line schema edit away at all times.
NATIVE_UNITS: dict[str, str] = {
    "radius": "mm",
    "curvature": "1/mm",
    "thickness": "mm",
    "object_distance": "mm",
    "entrance_pupil_diameter": "mm",
    "wavelength": "um",
    "field_angle": "deg",
    # The one entry where a unit error is *amplified* rather than merely scaled:
    # the r**4 coefficient carries mm**-3, so rebasing the schema on metres would
    # move that term by 1e9 rather than by 1e3.
    "aspheric_coefficient": "mm**(1 - 2*(i+1)) for aspheric_coefficients[i]",
}


def _require_native_units() -> None:
    """Refuse to build if the problem schema no longer declares Optiland's units."""
    mismatched = {
        name: (UNITS.get(name), expected)
        for name, expected in NATIVE_UNITS.items()
        if UNITS.get(name) != expected
    }
    if mismatched:
        raise ValueError(
            "the problem schema's declared units no longer match Optiland's native "
            f"units, so passing values through unconverted would be wrong: {mismatched!r}. "
            "Optiland's geometry unit is the prescription's own (millimetres here), its "
            "wavelengths are micrometres and its angular fields are degrees. Convert at "
            "this boundary, or do not change the schema's units."
        )


def _import_optiland_construction() -> tuple[Any, Any, Any, Any]:
    """Import exactly what construction needs, and nothing else.

    Kept inside the function so importing this module imports no solver:
    `tests/solvers/test_optiland_boundary.py` asserts that in a fresh
    interpreter, and it is what lets a caller read the module without paying for
    torch.
    """
    try:
        import optiland.backend as be
        from optiland.materials import IdealMaterial
        from optiland.materials import Material as CatalogMaterial
        from optiland.optic import Optic
    except ImportError as exc:  # pragma: no cover - environment failure
        raise ImportError(
            f"optiland could not be imported: {type(exc).__name__}: {exc}. Install it "
            "with this project's 'torch' extra (`pip install .[torch]`), which pins "
            "optiland>=0.6.0 and torch together."
        ) from exc
    return be, Optic, IdealMaterial, CatalogMaterial


def _resolve_catalog_material(material: Material, material_cls: Any, *, where: str) -> Any:
    """Resolve a named glass, refusing an inexact or unrecorded match.

    Optiland's own lookup is a substring filter over three columns followed by a
    Levenshtein ranking, and it returns the *best* row rather than insisting on an
    exact one. Measured on this project's own systems
    (`pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/system_construction_probe.json`,
    case `catalog_names_resolve_to_one_exact_match`): one glass name survives the
    substring filter as **seven** rows, and two similarly-spelled names in the same
    prescription select rows from two *different* vendors. So the winning
    manufacturer is not implied by the name, and a typo can resolve to a real but
    different glass with a plausible index. Two guards close that, and both are
    reused unchanged:

    * the winning row's `similarity_score` must be 0. That is Optiland's own
      exactness criterion -- the minimum Levenshtein distance from the requested
      name to the row's category name, catalog name or filename stem -- and using
      it rather than comparing names directly is what admits a legitimate entry
      whose row name carries a parenthesised vendor suffix while its filename stem
      is the bare name.
    * `expected_catalog_file`, when the problem records one, must equal the file
      actually chosen. That turns a material-database change in a future Optiland
      release into an error here instead of a quietly different trace.
    """
    name = material["name"]
    reference = material.get("catalog")
    try:
        resolved = material_cls(name=name, reference=reference)
        row = resolved.material_data
    except Exception as exc:
        # Optiland raises a bare ValueError for "no matches" and for "several
        # matches with robust search off". Neither is a solver failure: it is a
        # material this builder cannot resolve, and it has to reach the caller
        # as one.
        raise ValueError(
            f"{where}: catalog glass {name!r} (catalog={reference!r}) could not be "
            f"resolved against the pinned Optiland material database: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    resolved_file = str(row["filename"])
    similarity = row.get("similarity_score")
    if similarity is None or float(similarity) != 0.0:
        raise ValueError(
            f"{where}: catalog glass {name!r} resolved to {str(row['name'])!r} "
            f"({resolved_file}) by similarity (score {similarity!r}) rather than exactly. "
            "The pinned lookup ranks near-misses instead of refusing them, so an inexact "
            "match is a different glass with a plausible index, not an error."
        )
    expected = material.get("expected_catalog_file")
    if expected is not None and resolved_file != expected:
        raise ValueError(
            f"{where}: catalog glass {name!r} (catalog={reference!r}) resolved to "
            f"{resolved_file!r}, not the recorded {expected!r}. The problem records which "
            "catalog row it was transcribed against; a mismatch means the material "
            "database changed under it."
        )
    return resolved


def _material_argument(
    surface: SurfaceSpec, ideal_cls: Any, material_cls: Any, *, where: str
) -> Any:
    """The `material=` value for `surfaces.add` -- the medium *after* the surface."""
    material = surface.material
    kind = material["kind"]
    if kind == "air":
        # The literal string is what Optiland's MaterialFactory turns into
        # IdealMaterial(n=1.0, k=0.0). Passing it keeps this builder's output
        # identical to a bundled sample that omitted `material=` entirely.
        return "air"
    if kind == "ideal":
        # `k` is stated rather than omitted: absorption is not modelled by this
        # project and a lossless medium is a declaration, not a default.
        return ideal_cls(n=material["refractive_index"], k=0.0)
    if kind == "catalog":
        return _resolve_catalog_material(material, material_cls, where=where)
    raise ValueError(  # pragma: no cover - problems._check_material closes the set
        f"{where}: material kind {kind!r} has no Optiland construction path"
    )


def _geometry_arguments(surface: SurfaceSpec, be: Any) -> tuple[str, dict[str, Any]]:
    """The Optiland surface type for one surface, and its complete keyword set.

    The two are decided together and returned together, because they are one
    decision: `coefficients` belongs to `even_asphere` and to nothing else, and
    passing it alongside `standard` is **silently discarded**. Measured for
    CHE-207 against the pinned install: `surface_type='standard'` with
    `coefficients=[...]` raises nothing, builds a `StandardGeometry`, and the
    resulting object has no `coefficients` attribute at all -- so the asphere
    would simply not be there, and the trace would succeed on a different optical
    system. That is the same silent-filter hazard
    `pre-rewrite-2026-08-30:...system_construction_probe.json` recorded as
    `surface_kwargs_are_silently_filtered`, reached from the other direction.

    `radius` and `conic` are always both passed. A planar **base** is
    `radius = inf`, which is how the pinned factory is asked for a `Plane`;
    `problems.SurfaceSpec` represents it by the *absence* of a radius, so `inf` is
    produced here and never stored. The test is `has_planar_base` rather than
    `is_plane`, because an aspheric plate has a planar base and is not a plane.

    Nothing is forwarded from the problem that is not named below. That is the
    whole guard against the filter -- an unrecognized key cannot be discarded by
    Optiland if it was never assembled.
    """
    arguments: dict[str, Any] = {
        "radius": be.inf if surface.has_planar_base else surface.resolved_radius_mm,
        "conic": surface.conic,
    }
    if not surface.has_aspheric_terms:
        return _SURFACE_TYPE_STANDARD, arguments
    # A list, not the schema's tuple: the pinned geometry stores what it is handed
    # and indexes into it, and a tuple is not what its own samples pass. Built
    # from the surface's own coefficients, so the series convention -- index i
    # multiplies r**(2*(i+1)) -- crosses this boundary unchanged.
    arguments["coefficients"] = list(surface.aspheric_coefficients)
    return _SURFACE_TYPE_EVEN_ASPHERE, arguments


def build_lens(setup: OpticalSetup, source: SourceSpec | None = None) -> Any:
    """Construct the Optiland system an `OpticalSetup` describes.

    The returned object is an ordinary `optiland.optic.Optic`. It is native solver
    state and is not a project type: it stays inside `solvers/optiland/`, which is
    why the annotation is `Any` rather than a name this module would have to
    export.

    The signature carries no Optiland type, unit or API concept -- one neutral
    setup in, and one optional neutral source.

    Parameters
    ----------
    setup
        The optical configuration. Everything geometric comes from here, and so
        does the wavelength the paraxial characterization is evaluated at.
    source
        The declared illumination, or `None`. It contributes exactly two things,
        both of which the pinned backend requires before an `Optic` exists at
        all: where the object surface goes, and which single field is declared so
        there is something to normalize against. `None` means **no illumination
        was declared** -- the R05.6 path, where the caller supplies its own
        `RayBundle` -- and the object surface is then placed at infinity and the
        on-axis field declared, neither of which that path reads: it traces with
        the object surface skipped and never aims at a field. See the module
        docstring.

    Raises:
        TypeError: `setup` or `source` is not the type this builder takes.
        ValueError: the setup is outside the set this builder can construct, or a
            catalog glass did not resolve as the setup recorded. Raised before any
            surface is added.
        ImportError: optiland is not installed.
    """
    if not isinstance(setup, OpticalSetup):
        raise TypeError(
            f"build_lens takes an OpticalSetup, got {type(setup).__name__}. A setup is "
            "constructed and validated by the caller; this function does not parse one, "
            "and it resolves no prescription name into a lens."
        )
    if source is not None and not isinstance(source, SourceSpec):
        raise TypeError(
            f"build_lens's second argument is a SourceSpec or None, got "
            f"{type(source).__name__}. An already-materialized RayBundle is a source at "
            "the TRACE's argument position, not at this one: it is physical state and "
            "there is nothing here to construct from it."
        )
    _require_native_units()

    be, optic_cls, ideal_cls, material_cls = _import_optiland_construction()

    # 1. Resolve everything that can fail before touching Optiland, so a rejected
    #    setup never leaves a partially constructed lens behind.
    plans: list[tuple[SurfaceSpec, str, dict[str, Any], Any]] = []
    for index, surface in enumerate(setup.surfaces):
        where = f"{setup.name}: surfaces[{index}]"
        surface_type, geometry_kwargs = _geometry_arguments(surface, be)
        plans.append(
            (
                surface,
                surface_type,
                geometry_kwargs,
                _material_argument(surface, ideal_cls, material_cls, where=where),
            )
        )

    # 2. Construct, in setup order.
    optic = optic_cls(name=setup.name)

    # The object surface, which the setup does not list because it is not part of
    # the optical configuration: a plane in air, `object_distance_mm` before the
    # first listed surface, at infinity when the source says the object is or when
    # no source was declared.
    #
    # `thickness` is the *whole* of the source geometry, and both cases are one
    # line because the solver derives the rest. At `inf` the field is a direction
    # and every ray of one field is launched with a common direction on a plane
    # perpendicular to z. At a finite distance the field becomes a *position* --
    # the source sits at `(-tan(x_deg) * d, -tan(y_deg) * d, -d)`, measured to
    # twelve digits for CHE-207 -- and every ray of one field leaves that single
    # point. `rays.py` owns what that does to the declared optical path.
    object_distance = None if source is None else source.object_distance_mm
    optic.surfaces.add(
        index=0,
        radius=be.inf,
        thickness=be.inf if object_distance is None else object_distance,
    )
    for offset, (surface, surface_type, geometry_kwargs, material) in enumerate(plans):
        optic.surfaces.add(
            index=offset + 1,
            surface_type=surface_type,
            thickness=surface.thickness_mm,
            material=material,
            # The stop is one index on the setup, so "no stop" and "three stops"
            # are not representable states a validator has to reject.
            is_stop=offset == setup.stop_index,
            comment=surface.comment,
            **geometry_kwargs,
        )
    # The image surface: a plane in air at zero further spacing, placed by the
    # last listed surface's thickness.
    optic.surfaces.add(index=len(plans) + 1, radius=be.inf, thickness=0.0)

    # 3. Aperture, field, wavelength -- one fixed order, no inherited default.
    optic.set_aperture(aperture_type=_APERTURE_TYPE, value=setup.entrance_pupil_diameter_mm)

    # EXACTLY ONE field: the one being traced, or the axis when none is. That is
    # what makes `max_field` the field the caller asked for rather than the
    # largest of a list it had to declare in advance -- and it is measured not to
    # move any other paraxial quantity. See the module docstring.
    optic.fields.set_type(field_type=_FIELD_TYPE)
    x_deg, y_deg = (0.0, 0.0) if source is None else source.field_angle_deg
    vignette_x, vignette_y = FIELD_VIGNETTING_FACTORS
    optic.fields.add(y=y_deg, x=x_deg, vx=vignette_x, vy=vignette_y, weight=FIELD_WEIGHT)

    # EXACTLY ONE wavelength, and it is the setup's reference rather than the
    # source's. It is what the backend takes as primary, and the primary is what
    # `paraxial.XPL()`/`XPD()` are evaluated at -- the exit pupil is a property of
    # the system's characterization. The wavelength a trace is evaluated at is
    # passed to the trace call and does not have to be declared here.
    optic.wavelengths.add(
        value=setup.reference_wavelength_um,
        is_primary=True,
        unit=_WAVELENGTH_UNIT,
        weight=WAVELENGTH_WEIGHT,
    )

    return optic
