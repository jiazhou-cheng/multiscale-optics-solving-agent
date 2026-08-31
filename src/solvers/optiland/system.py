"""Neutral problem in, native Optiland lens out, through one generic path.

CHE-179 (R05.1). The construction half of the anti-corruption layer:
`RayTraceProblem` -> `optiland.optic.Optic`. One function, `build_lens`, and
adding a system means handing it a different problem -- never writing a builder.

What is deliberately absent
---------------------------
There is no per-system builder, no builder class, and no name-to-system
resolution. The reference implementation's `_resolve_lens(spec)` read as a lookup
and was a one-line call to the generic builder; the indirection is gone and the
generic path is called directly. There is no list of supported prescription
names, because R04 removed the prescription catalog from production: a caller
constructs the `RayTraceProblem` it wants.

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

from problems import UNITS, Material, RayTraceProblem, SurfaceSpec

__all__ = [
    "FIELD_VIGNETTING_FACTORS",
    "FIELD_WEIGHT",
    "NATIVE_UNITS",
    "WAVELENGTH_WEIGHT",
    "build_lens",
]

#: The Optiland `surface_type` this builder emits. One value: the pinned factory
#: returns a `Plane` for an infinite radius and a `StandardGeometry` otherwise,
#: which is how every problem the new tree can state spells a flat surface. Even
#: aspheres and gratings are additions to `problems.SurfaceSpec` first -- the
#: schema cannot express either today, so a second surface type here would be a
#: branch nothing can reach.
_SURFACE_TYPE = "standard"

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


def _geometry_arguments(surface: SurfaceSpec, be: Any) -> dict[str, Any]:
    """The complete geometry keyword set for one surface, assembled explicitly.

    `radius` and `conic` are always both passed. A plane is `radius = inf`, which
    is how the pinned factory is asked for a `Plane`; `problems.SurfaceSpec`
    represents a plane by the *absence* of a radius, so `inf` is produced here and
    never stored.

    Nothing is forwarded from the problem that is not named on this line. That is
    the whole guard against the silent keyword filter -- an unrecognized key
    cannot be discarded by Optiland if it was never assembled.
    """
    return {
        "radius": be.inf if surface.is_plane else surface.resolved_radius_mm,
        "conic": surface.conic,
    }


def build_lens(problem: RayTraceProblem) -> Any:
    """Construct the Optiland system a `RayTraceProblem` describes.

    The returned object is an ordinary `optiland.optic.Optic`. It is native
    solver state and is not a project type: it stays inside
    `solvers/optiland/`, which is why the annotation is `Any` rather than a name
    this module would have to export.

    The signature carries no Optiland type, unit or API concept -- one neutral
    problem in.

    Raises:
        ValueError: the problem is outside the set this builder can construct, or
            a catalog glass did not resolve as the problem recorded. Raised
            before any surface is added.
        ImportError: optiland is not installed.
    """
    if not isinstance(problem, RayTraceProblem):
        raise TypeError(
            f"build_lens takes a RayTraceProblem, got {type(problem).__name__}. A "
            "problem is constructed and validated by the caller; this function does not "
            "parse one, and it resolves no prescription name into a lens."
        )
    _require_native_units()

    be, optic_cls, ideal_cls, material_cls = _import_optiland_construction()

    # 1. Resolve everything that can fail before touching Optiland, so a rejected
    #    problem never leaves a partially constructed lens behind.
    plans: list[tuple[SurfaceSpec, dict[str, Any], Any]] = []
    for index, surface in enumerate(problem.surfaces):
        where = f"{problem.name}: surfaces[{index}]"
        plans.append(
            (
                surface,
                _geometry_arguments(surface, be),
                _material_argument(surface, ideal_cls, material_cls, where=where),
            )
        )

    # 2. Construct, in problem order.
    optic = optic_cls(name=problem.name)

    # The object surface, which the problem does not list because it is fixed: a
    # plane in air, `object_distance_mm` before the first listed surface, at
    # infinity when the problem says the object is.
    optic.surfaces.add(
        index=0,
        radius=be.inf,
        thickness=be.inf if problem.object_at_infinity else problem.object_distance_mm,
    )
    for offset, (surface, geometry_kwargs, material) in enumerate(plans):
        optic.surfaces.add(
            index=offset + 1,
            surface_type=_SURFACE_TYPE,
            thickness=surface.thickness_mm,
            material=material,
            # The stop is one index on the problem, so "no stop" and "three
            # stops" are not representable states a validator has to reject.
            is_stop=offset == problem.stop_index,
            comment=surface.comment,
            **geometry_kwargs,
        )
    # The image surface: a plane in air at zero further spacing, placed by the
    # last listed surface's thickness.
    optic.surfaces.add(index=len(plans) + 1, radius=be.inf, thickness=0.0)

    # 3. Aperture, fields, wavelengths -- one fixed order, no inherited default.
    optic.set_aperture(aperture_type=_APERTURE_TYPE, value=problem.entrance_pupil_diameter_mm)

    optic.fields.set_type(field_type=_FIELD_TYPE)
    vignette_x, vignette_y = FIELD_VIGNETTING_FACTORS
    for x_deg, y_deg in problem.field_angles_deg:
        optic.fields.add(y=y_deg, x=x_deg, vx=vignette_x, vy=vignette_y, weight=FIELD_WEIGHT)

    for index, wavelength_um in enumerate(problem.wavelengths_um):
        optic.wavelengths.add(
            value=wavelength_um,
            is_primary=index == problem.primary_wavelength_index,
            unit=_WAVELENGTH_UNIT,
            weight=WAVELENGTH_WEIGHT,
        )

    return optic
