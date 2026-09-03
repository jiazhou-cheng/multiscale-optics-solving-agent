"""Tier 2 of workstream A: system regression over the 25 in-scope Optiland tutorials.

CHE-239 §A.5. None of these tutorials prints a quantity a descriptor returns --
their outputs are paraxial reports, optimization traces, OPD, MTF, tolerancing
Monte Carlo and ML training curves, and §A.0 establishes that the catalog has no
observable for any of them. What the tutorials *do* supply is an optical system
and, for every system, goldens that Optiland can produce directly. So the
question this tier answers is not "does the notebook reproduce" but:

> Can `problems.OpticalSetup` express the system this tutorial builds, and when
> it can, does the system rebuilt from it trace the same rays as the tutorial's
> own lens?

The three checks, in the order they can fail
--------------------------------------------
1. **Expressibility.** `optiland_prescription.extract_setup` either produces a
   setup or refuses with a category. A refusal is `BLOCKED`, not `FAIL`, and
   §A.2 forbids inventing a representation to make one go away.
2. **Surface-table round-trip.** The setup is rebuilt with the same
   `backends.optiland.system.build_lens` a trace would use, and the rebuilt lens
   is compared against the tutorial's own, surface by surface: radius, conic,
   the index of the following medium at the reference wavelength, the axial
   position, the stop index, the clear semi-diameter. Then the paraxial
   characterization -- EFL, EPD, F-number -- which is where an aperture that was
   converted from `imageFNO` shows up if the conversion was wrong.
3. **Ray regression.** `SO_RAY_LAUNCH_TRACE` on the rebuilt system against
   `Optic.trace` on the tutorial's own, at a fixed pupil sampling: final
   intersection coordinates, direction cosines, accumulated optical path, and
   the clip/survival mask. This is the check that would catch a surface-table
   round-trip that agreed on every printed number and still traced differently.

Then `SOM_SPOT_DIAGRAM` and `SOM_PSF` where the system supports them.

Why the spot check runs at the setup's reference wavelength
-----------------------------------------------------------
Tier 1 confirmed that `spot_diagram` analyses at `setup.reference_wavelength_um`
regardless of what the source declares (see `ray_tier1._ATTRIBUTIONS` for the
measurement). Running Tier 2's spot check at a source wavelength that differs
from the reference would therefore produce 20-odd rows all failing for that one
known reason, which would bury whatever else this tier found. The source is set
to the reference wavelength, the row records that it was, and the discrepancy
stays a Tier-1 result with one cause rather than becoming a Tier-2 result with
twenty.

Where the systems come from
---------------------------
The notebooks are not vendored. Each is fetched at the pinned upstream commit
(CHE-238 §3.1) into a directory this driver is pointed at, and its code cells are
executed **in order until an `Optic` appears** -- at most `MAX_CELLS` of them,
under a wall-clock budget, with the first constructed lens winning. That bounds
the cost to the construction prologue and never reaches the optimization, ML or
tolerancing body that follows it, which is both the cheap-probe rule and the only
way 25 tutorials fit in an overnight slot at all.

Executing a fetched notebook's prologue is a deliberate choice over parsing it.
A parser would have to know that `Tutorial_4a` builds 34 surfaces across several
cells with tilts applied afterwards, and a parser that got that subtly wrong
would report a *different system's* regression as a pass. Executing the
tutorial's own code cannot.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import matplotlib

# Before `pyplot` is imported anywhere: a tutorial prologue calls `Optic.draw`,
# `SpotDiagram.view` and `plt.show()`, and an interactive backend in a headless
# container either blocks or fails. Set here rather than through MPLBACKEND so the
# driver does not depend on how it was invoked.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from backends.optiland.analysis import psf as native_psf
from backends.optiland.analysis import spot_diagram as native_spot_diagram
from backends.optiland.solver import trace as launch_trace
from backends.optiland.system import build_lens
from benchmarks.verification.optiland_prescription import (
    Unexpressible,
    extract_setup,
    sample_fields_deg,
)
from benchmarks.verification.record import (
    METRE_PER_MM,
    Row,
    device_execution,
    finish,
    provenance,
)
from problems import SourceSpec

#: The 25 in-scope tutorials, in CHE-239 §A.5's order, keyed by the short label
#: the ticket tables them under. The 14 out-of-scope ones are absent rather than
#: skipped: a driver that listed them would invite someone to turn the skip off.
IN_SCOPE: dict[str, str] = {
    "1a": "Tutorial_1a_Optiland_for_Beginners",
    "1b": "Tutorial_1b_Lens_Properties_and_Prescription",
    "1c": "Tutorial_1c_Material_Database_and_Catalogs",
    "1e": "Tutorial_1e_Design_a_Doublet_End_to_End",
    "2a": "Tutorial_2a_Tracing_and_Analyzing_Rays",
    "2b": "Tutorial_2b_Monte_Carlo_Raytracing",
    "2c": "Tutorial_2c_Aberration_Analyses",
    "2d": "Tutorial_2d_OPD_PSF_and_MTF_Calculations",
    "3a": "Tutorial_3a_Simple_Optimization",
    "3b": "Tutorial_3b_Advanced_Optimization",
    "3c": "Tutorial_3c_User_Defined_Optimization",
    "3d": "Tutorial_3d_Optimization_Case_Study_Cooke_Triplet",
    "3e": "Tutorial_3e_Glass_Expert_Categorical_Optimization",
    "4a": "Tutorial_4a_Tilts_Decenters_and_Asymmetric_Systems",
    "4b": "Tutorial_4b_Raytracing_Aspheres_and_Freeforms",
    "4e": "Tutorial_4e_Lithographic_Projection_System",
    "4f": "Tutorial_4f_Three_Mirror_Anastigmat",
    "6a": "Tutorial_6a_Tolerancing_Sensitivity_Analysis",
    "6b": "Tutorial_6b_Monte_Carlo_Tolerancing_Analysis",
    "8a": "Tutorial_8a_Custom_Surface_Types",
    "8c": "Tutorial_8c_Custom_Optimization_Algorithm",
    "9a": "Tutorial_9a_Predicting_Lens_Performance_with_Random_Forest",
    "9b": "Tutorial_9b_Classifying_Ray_Path_Failures_with_Machine_Learning",
    "9d": "Tutorial_9d_Optimizing_Aspheric_Singlets_via_Reinforcement_Learning",
    "9f": "Tutorial_9f_Predicting_Physical_Lens_Misalignments_from_Spot_Diagrams",
}

#: What the tutorial's output needs and the catalog does not have. Straight from
#: CHE-239 §A.5's "Output needs no current observable" column, so a `NOT-COVERED`
#: row can name *what* is not covered instead of saying so generically. A
#: tutorial absent from this mapping has no listed gap.
COVERAGE_GAPS: dict[str, tuple[str, ...]] = {
    "1b": ("paraxial-report",),
    "1c": ("paraxial-report",),
    "1e": ("geom-analysis", "optimization", "paraxial-report", "tolerancing"),
    "2b": ("opd/zernike",),
    "2c": ("geom-analysis",),
    "2d": ("mtf", "opd/zernike"),
    "3a": ("geom-analysis", "opd/zernike", "optimization"),
    "3b": ("geom-analysis", "optimization"),
    "3c": ("optimization",),
    "3d": ("optimization",),
    "3e": ("optimization",),
    "4a": ("optimization",),
    "4b": ("optimization",),
    "4e": ("geom-analysis", "mtf", "opd/zernike", "optimization"),
    "4f": ("mtf", "optimization"),
    "6a": ("optimization", "tolerancing"),
    "6b": ("optimization", "tolerancing"),
    "8c": ("optimization",),
    "9a": ("optimization",),
    "9d": ("geom-analysis", "mtf", "paraxial-report"),
    "9f": ("opd/zernike",),
}

#: Visualization the harvest replaces with a no-op for the duration of a prologue.
#:
#: A hard stop condition, not a convenience. `Optic.draw3D` opens a VTK render
#: window, and in this headless container VTK fails to reach an X server, EGL and
#: OSMesa in turn and then **aborts the process** -- not an exception the harvest
#: loop can catch, an abort that takes the whole run with it and leaves no record.
#: Measured: the Tier-2 sweep died during tutorial 1a with three
#: `vtkOpenGLRenderWindow` warnings and no output at all.
#:
#: Neutralizing the *method* rather than skipping the cell, which is what this
#: harness did first and got wrong: tutorial 4f builds its entire 4-surface mirror
#: system and calls `# lens.draw3D()` -- commented out -- in one cell, so a
#: substring match on the source skipped the construction and reported the
#: tutorial as having built no lens. A no-op method cannot do that. `draw3D`
#: renders a lens that already exists and returns nothing a later cell reads, so
#: replacing it changes no system.
NEUTRALIZED_METHODS: tuple[str, ...] = ("draw3D",)

#: How far into a notebook the harvester will go looking for lenses, and how long
#: it may spend. Both are stop conditions rather than tuning.
#:
#: The budget is per tutorial and it is deliberately short. Once the harvest stopped
#: at the first complete lens and started collecting all of them, it began running
#: into the tutorials' optimization and Monte-Carlo bodies, and 25 tutorials at two
#: minutes each is fifty minutes to answer a question the first twenty seconds
#: answers. A tutorial that exhausts the budget still yields whatever lenses it built
#: before it ran out, and the row records `cells_executed` against
#: `cells_available` so a partial harvest reads as one.
MAX_CELLS = 40
HARVEST_BUDGET_S = 20.0

#: The fixed pupil sampling every ray regression uses. A ring count, not a ray
#: count: 4 rings is 61 rays (`rays.hexapolar_ray_count`), which is enough to
#: cover the pupil to its edge -- where clipping and aspheric departure live --
#: without making 25 systems expensive.
REGRESSION_NUM_RINGS = 4


def _host(value: Any) -> np.ndarray:
    """A native array as host float64, whichever backend produced it.

    `.real` before the cast rather than after: `RealRays.i` is complex on the
    numpy backend, and `np.asarray(complex_array, dtype=float64)` raises a
    `ComplexWarning` and silently discards the imaginary part. Taking the real
    part explicitly says that discarding it is intended -- an intensity's
    imaginary part is zero and a nonzero one would be a different bug entirely.
    """
    array = getattr(value, "detach", lambda: value)()
    array = np.asarray(getattr(array, "cpu", lambda: array)())
    return np.asarray(array.real if np.iscomplexobj(array) else array, dtype=np.float64)


def _scalar(value: Any) -> float:
    """A native scalar that may arrive wrapped in a 1-element array."""
    array = np.asarray(getattr(value, "detach", lambda: value)())
    return float(array.reshape(-1)[0]) if array.ndim else float(array)


def _harvest_optic(notebook: Path) -> tuple[Any, dict[str, Any]]:
    """Execute the notebook's prologue and return its richest complete `Optic`.

    Returns the lens and a mapping describing how it was obtained -- how many
    cells ran, how many complete lenses were found, how many optical surfaces the
    chosen one has, and every cell that raised on the way. Cells that raise do
    **not** stop the harvest: a tutorial's prologue routinely contains a cell that
    needs a dependency, a download or a display this container does not have, and
    the surrounding construction is still valid. Every such failure is recorded,
    because a lens harvested past three broken cells is a lens whose provenance a
    reader should see.

    Raises:
        Unexpressible: no complete `Optic` appeared within the cell or time budget.
    """
    from optiland.optic import Optic

    cells = [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    namespace: dict[str, Any] = {"__name__": "__tutorial__"}
    errors: list[dict[str, str]] = []
    #: Every complete lens seen at any point in the prologue, in the order first
    #: seen. A list and not a set: `Optic` is unhashable, and identity is what
    #: "the same lens" has to mean here.
    optics: list[Any] = []
    started = time.perf_counter()

    executed = 0
    saved = {name: getattr(Optic, name, None) for name in NEUTRALIZED_METHODS}
    for name in NEUTRALIZED_METHODS:
        setattr(Optic, name, lambda *_arguments, **_keywords: None)
    # Run the prologue somewhere it can write. Measured: tutorial 6a's tolerancing
    # cell drops `sensitivity_analysis.csv` into the working directory, which for a
    # driver invoked from the repository root is the repository root. A harness that
    # leaves untracked files in the tree it is verifying is a harness whose next run
    # starts from a different state.
    scratch = notebook.parent / "_prologue_cwd" / notebook.stem
    scratch.mkdir(parents=True, exist_ok=True)
    origin = Path.cwd()
    os.chdir(scratch)
    try:
        for index, source in enumerate(cells[:MAX_CELLS]):
            if time.perf_counter() - started > HARVEST_BUDGET_S:
                break
            executed = index + 1
            try:
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    exec(compile(source, f"{notebook.name}[{index}]", "exec"), namespace)
                    plt.close("all")
            except BaseException as error:
                errors.append({"cell": str(index), "error": f"{type(error).__name__}: {error}"})

        # Snapshot after every cell rather than only at the end, and keep the
        # reference. A tutorial that rebinds the name, or mutates the lens into an
        # incomplete state, would otherwise lose it: measured on 4f, whose mirror
        # system is not in the namespace by the time the prologue finishes.
            for value in namespace.values():
                if (
                    isinstance(value, Optic)
                    and len(getattr(value.surfaces, "surfaces", ())) >= 3
                    and getattr(value, "aperture", None) is not None
                    and list(getattr(value.wavelengths, "wavelengths", ()))
                    and all(value is not seen for seen in optics)
                ):
                    optics.append(value)
    finally:
        os.chdir(origin)
        for name, original in saved.items():
            if original is not None:
                setattr(Optic, name, original)

    candidates = optics
    if candidates:
        chosen = max(candidates, key=lambda value: len(value.surfaces.surfaces))
        return chosen, {
            "cells_executed": executed,
            "cells_available": len(cells),
            "draw3d_neutralized": True,
            "complete_lenses_found": len(candidates),
            "optical_surfaces_chosen": len(chosen.surfaces.surfaces) - 2,
            "optical_surfaces_available": sorted(
                len(value.surfaces.surfaces) - 2 for value in candidates
            ),
            "cell_errors": errors,
            "notebook": notebook.name,
        }

    # Which of the two failures this was, because they mean different things: an
    # Optic that exists but declares no aperture or wavelength is a tutorial whose
    # lens is genuinely incomplete for a trace, while no Optic at all usually
    # means the construction is inside a function or a class the prologue never
    # instantiates. The distinction decides whether the follow-up is "extend the
    # harvester" or "this tutorial has no standalone system".
    incomplete = [
        {
            "surfaces": len(value.surfaces.surfaces),
            "has_aperture": getattr(value, "aperture", None) is not None,
            "wavelengths": len(list(getattr(value.wavelengths, "wavelengths", ()))),
        }
        for value in namespace.values()
        if isinstance(value, Optic)
    ]
    subcategory = _construction_subcategory(errors)
    if incomplete:
        reason = (
            f"{len(incomplete)} Optic(s) were built but none declares both an aperture and a "
            f"wavelength: {incomplete}"
        )
    elif subcategory == "structure":
        reason = (
            "no Optic was constructed at module level, and no cell raised on an import, so the "
            "tutorial builds its lens inside a function or class the prologue does not "
            "instantiate"
        )
    else:
        # Do not assert a structural cause the harness did not establish. When the
        # first cell fails on an import, every name it defined is missing and the
        # rest of the prologue cascades into `NameError`; the tutorial's lens may
        # well be at module level and simply never reached. Measured on 9b, whose
        # first cell needs `sklearn` and whose second is a bare `CookeTriplet()`.
        reason = (
            "no Optic was constructed; an import failed and the rest of the prologue cascaded, "
            "so whether the tutorial builds its lens at module level was not determined"
        )
    reason = f"[{subcategory}] {reason}"
    raise Unexpressible(
        "construction",
        f"{notebook.name}: {reason}. {min(len(cells), MAX_CELLS)} cell(s) executed, "
        f"{len(errors)} raised: {errors[:4]}",
    )


#: Why a tutorial's prologue did not produce a lens. A `construction` refusal is
#: not one thing, and the three cases have different owners:
#:
#: * `environment-dependency` -- a cell imported a package the container does not
#:   ship (`sklearn`, `tqdm`, `gymnasium`), which cascades into `NameError` for
#:   everything defined in that cell. Nothing about this project or the schema.
#: * `upstream-api-drift` -- a cell imported an `optiland` name the *installed*
#:   0.6.0 does not have while the pinned `master` notebook does
#:   (`optiland.diagnostics`, `optiland.materials.MaterialCatalog`). CHE-238 §3.1
#:   classifies this as an environment finding, not a physics failure.
#: * `structure` -- the prologue ran but builds its lens inside a function or
#:   class it does not instantiate, or builds one with no aperture or wavelength.
#:   This is the only case where extending the harvester would help.
CONSTRUCTION_SUBCATEGORIES = ("environment-dependency", "upstream-api-drift", "structure")


def _construction_subcategory(errors: list[dict[str, str]]) -> str:
    """Which of `CONSTRUCTION_SUBCATEGORIES` this prologue's failures were."""
    for entry in errors:
        message = entry["error"]
        if message.startswith(("ModuleNotFoundError", "ImportError")):
            return "upstream-api-drift" if "optiland" in message else "environment-dependency"
    return "structure"


def _surface_table(lens: Any, wavelength_um: float) -> list[dict[str, float | str | None]]:
    """The comparable content of a lens's surface list.

    Everything that changes the traced geometry and nothing that does not: the
    axial position rather than the thickness, because a thickness that is off by
    a compensating pair on two surfaces still puts every surface in the same
    place; the following medium's *index at the reference wavelength* rather than
    its name, because two catalog rows with the same index at every wavelength
    are the same medium for a trace and the name is checked separately.
    """
    positions = [float(value) for value in lens.surfaces.positions.ravel()]
    table: list[dict[str, float | str | None]] = []
    for offset, surface in enumerate(list(lens.surfaces.surfaces)[1:-1]):
        geometry = surface.geometry
        radius = float(geometry.radius)
        material = surface.material_post
        table.append(
            {
                "position_mm": positions[offset + 1],
                "radius_mm": radius if math.isfinite(radius) else None,
                "conic": float(getattr(geometry, "k", 0.0) or 0.0),
                "index_after": _scalar(material.n(wavelength_um)),
                "is_stop": bool(getattr(surface, "is_stop", False)),
                # Whether the surface has any *nonzero* aspheric term, which is
                # the physical question; the geometry class name is not. Optiland
                # builds an `EvenAsphere` for an all-zero coefficient list where
                # `build_lens` builds a `standard`, and `SurfaceSpec`'s docstring
                # records the measurement that the two agree bitwise in sag,
                # traced position and accumulated path. So the class names may
                # differ where this column does not.
                "aspheric_nonzero": any(
                    float(value) != 0.0 for value in getattr(geometry, "coefficients", ())
                ),
                "semi_diameter_mm": (
                    float(surface.aperture.r_max)
                    if getattr(surface, "aperture", None) is not None
                    and getattr(surface.aperture, "r_max", None) is not None
                    else None
                ),
                "geometry": type(geometry).__name__,
            }
        )
    return table


def _table_delta(ours: list[dict[str, Any]], theirs: list[dict[str, Any]]) -> dict[str, Any]:
    """Worst absolute difference per numeric column, and every categorical mismatch."""
    if len(ours) != len(theirs):
        return {"surface_count": {"ours": len(ours), "theirs": len(theirs)}}
    numeric = ("position_mm", "radius_mm", "conic", "index_after", "semi_diameter_mm")
    worst: dict[str, Any] = {}
    for column in numeric:
        differences = [
            abs(a[column] - b[column])
            for a, b in zip(ours, theirs, strict=True)
            if a[column] is not None and b[column] is not None
        ]
        mismatched_presence = [
            offset
            for offset, (a, b) in enumerate(zip(ours, theirs, strict=True))
            if (a[column] is None) != (b[column] is None)
        ]
        worst[column] = max(differences) if differences else 0.0
        if mismatched_presence:
            worst[f"{column}_present_on_one_side_only"] = mismatched_presence
    for column in ("is_stop", "aspheric_nonzero", "geometry"):
        mismatches = [
            {"surface": offset, "ours": a[column], "theirs": b[column]}
            for offset, (a, b) in enumerate(zip(ours, theirs, strict=True))
            if a[column] != b[column]
        ]
        if mismatches:
            worst[column] = mismatches
    return worst


def _paraxial(lens: Any) -> dict[str, float]:
    """The three first-order quantities a rebuilt lens has to reproduce.

    None of these has a descriptor (§A.0: paraxial getters are not in the
    catalog), which is exactly why they are useful here: they are Optiland's own
    numbers on both sides, so they check the reconstruction without this project
    having to claim a first-order capability it does not have.
    """
    paraxial = lens.paraxial
    return {
        "EFL_mm": float(paraxial.f2()),
        "EPD_mm": float(paraxial.EPD()),
        "FNO": float(paraxial.FNO()),
    }


def _trace_regression(
    *, lens: Any, setup: Any, source: SourceSpec
) -> tuple[dict[str, float], dict[str, Any]]:
    """`SO_RAY_LAUNCH_TRACE` on the rebuilt system against `Optic.trace` on the original.

    The native side is the tutorial's own lens traced through Optiland's public
    entry point at the same field, wavelength and ring count. The project side is
    the catalogued operation, whose output is a neutral `RayBundle` in metres.
    Compared: the final intersection coordinates, the final direction cosines, the
    accumulated optical path, and the survival mask.

    The optical path is compared as a **difference from its own mean** rather than
    absolutely. `backends/optiland/rays.py` declares the bundle's
    `optical_path_reference`, and the two sides do not have to agree on where zero
    is -- what a trace regression is about is whether the *wavefront* is the same
    shape. An absolute offset between two declared references is a convention
    difference; a varying difference is a different trace.

    **Off axis the varying difference is expected, and it is not a defect.**
    `declare_optical_path_m` adds an object-space term
    `n_object * (d0 . r_launch)` that `RealRays.opd` does not carry: the native
    accumulator is seeded on a plane perpendicular to z, and off axis the incoming
    wavefront is tilted with respect to that plane, so the two quantities differ by
    a *tilt* rather than a piston. That module's own docstring says the omission
    "IS the convergence tilt". So the caller decides whether this delta gates --
    `on_axis` below -- and off axis it is reported as a measured, expected
    difference with the tilt's magnitude, rather than as a failing comparison.
    Measured here on tutorial 1b at its 20 deg field: 2.1e-2 of the optical-path
    extent, against 0.0 on axis for the same system.
    """
    max_field = float(lens.fields.max_field)
    field_deg = source.field_angle_deg
    normalized = (
        0.0 if max_field == 0.0 else field_deg[0] / max_field,
        0.0 if max_field == 0.0 else field_deg[1] / max_field,
    )
    native = lens.trace(
        Hx=normalized[0],
        Hy=normalized[1],
        wavelength=source.wavelength_um,
        num_rays=REGRESSION_NUM_RINGS,
    )
    bundle = launch_trace(
        setup,
        source,
        sampling={"num_rings": REGRESSION_NUM_RINGS, "reference_surface": "image_surface"},
        execution=device_execution(),
    )

    native_x, native_y, native_z = (_host(getattr(native, axis)) for axis in ("x", "y", "z"))
    native_L, native_M, native_N = (_host(getattr(native, axis)) for axis in ("L", "M", "N"))
    native_opl = _host(native.opd)
    native_alive = _host(native.i) > 0.0

    ours = _host(bundle.positions_m)
    our_directions = _host(bundle.directions)

    # The clip/survival comparison is a comparison of **counts**, not of masks.
    # `rays.to_ray_bundle` filters clipped rays out of the returned bundle
    # (`optical_path_m[alive]` and the arrays beside it), so a `RayBundle` has no
    # dead row to compare against and an element-wise mask check would be
    # trivially all-True. What is actually checkable is that the bundle holds
    # exactly the rays the native trace says survived -- which is the same claim,
    # and is not vacuous when a system clips.
    survivor_count_delta = float(abs(ours.shape[0] - int(np.count_nonzero(native_alive))))

    if ours.shape[0] != int(np.count_nonzero(native_alive)):
        return (
            {"surviving_ray_count_mismatch": survivor_count_delta},
            {
                "ours": int(ours.shape[0]),
                "native_launched": int(native_x.shape[0]),
                "native_surviving": int(np.count_nonzero(native_alive)),
            },
        )
    # Compare row-for-row against the surviving native rays only.
    native_x, native_y, native_z = (
        array[native_alive] for array in (native_x, native_y, native_z)
    )
    native_L, native_M, native_N = (
        array[native_alive] for array in (native_L, native_M, native_N)
    )
    native_opl = native_opl[native_alive]

    native_positions = np.stack([native_x, native_y, native_z], axis=-1) * METRE_PER_MM
    native_directions = np.stack([native_L, native_M, native_N], axis=-1)

    scale = float(np.max(np.abs(native_positions))) or 1.0
    position_delta = float(np.max(np.abs(ours - native_positions))) / scale
    direction_delta = float(np.max(np.abs(our_directions - native_directions)))

    our_opl = _host(bundle.optical_path_m) / METRE_PER_MM
    opl_offset = float(np.mean(our_opl - native_opl))
    opl_shape_delta = float(np.max(np.abs((our_opl - native_opl) - opl_offset)))
    opl_scale = float(np.max(np.abs(native_opl))) or 1.0

    on_axis = source.field_angle_deg == (0.0, 0.0)
    opl_key = (
        "optical_path_shape_max_abs_over_extent"
        if on_axis
        else "optical_path_object_space_tilt_over_extent_EXPECTED"
    )
    deltas = {
        "position_max_abs_over_extent": position_delta,
        "direction_cosine_max_abs": direction_delta,
        opl_key: opl_shape_delta / opl_scale,
        "surviving_ray_count_mismatch": survivor_count_delta,
    }
    detail = {
        "rays_in_bundle": int(ours.shape[0]),
        "native_launched": int(native_x.shape[0]),
        "native_surviving": int(np.count_nonzero(native_alive)),
        "optical_path_constant_offset_mm": opl_offset,
        "optical_path_reference": bundle.optical_path_reference,
        "num_rings": REGRESSION_NUM_RINGS,
        "on_axis": on_axis,
    }
    return deltas, detail


def run_tutorial(label: str, notebook: Path) -> list[Row]:
    """Every row one tutorial contributes."""
    gaps = COVERAGE_GAPS.get(label, ())
    rows: list[Row] = []
    if gaps:
        rows.append(
            Row(
                case=label,
                configuration={"needs": list(gaps)},
                descriptor="",
                status="NOT-COVERED",
                measured={},
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=0.0,
                note=(
                    "the tutorial's printed output needs a capability the catalog does not "
                    "expose (CHE-239 §A.0). A coverage gap, not a failure of the optical system"
                ),
            )
        )

    started = time.perf_counter()
    try:
        lens, harvest = _harvest_optic(notebook)
    except Unexpressible as error:
        rows.append(
            Row(
                case=label,
                configuration={"notebook": notebook.name},
                descriptor="",
                status="BLOCKED",
                measured={"category": error.category, "detail": error.detail},
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=time.perf_counter() - started,
                note="no optical system could be harvested from the tutorial's prologue",
            )
        )
        return rows

    try:
        setup, provenance_detail = extract_setup(lens, name=label)
    except Unexpressible as error:
        rows.append(
            Row(
                case=label,
                configuration={"notebook": notebook.name, **harvest},
                descriptor="",
                status="BLOCKED",
                measured={"category": error.category, "detail": error.detail},
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=time.perf_counter() - started,
                note=(
                    "problems.OpticalSetup cannot express this system. CHE-239 §A.2 forbids "
                    "inventing a representation to make it expressible during this run"
                ),
            )
        )
        return rows

    # One field and one wavelength, and the wavelength is the setup's reference:
    # see the module docstring on why the Tier-1 spot discrepancy is not
    # re-litigated 25 times here.
    fields = sample_fields_deg(lens)
    field_deg = max(fields, key=lambda pair: abs(pair[0]) + abs(pair[1])) if fields else (0.0, 0.0)
    wavelength_um = float(setup.reference_wavelength_um)
    source = SourceSpec(wavelength_um=wavelength_um, field_angle_deg=field_deg)

    # 2. Surface-table round-trip.
    rebuilt = build_lens(setup, source)
    table_delta = _table_delta(
        _surface_table(rebuilt, wavelength_um), _surface_table(lens, wavelength_um)
    )
    ours_paraxial, theirs_paraxial = _paraxial(rebuilt), _paraxial(lens)
    paraxial_delta = {
        key: (
            abs(ours_paraxial[key] - theirs_paraxial[key]) / abs(theirs_paraxial[key])
            if theirs_paraxial[key]
            else abs(ours_paraxial[key] - theirs_paraxial[key])
        )
        for key in ours_paraxial
    }
    numeric_table = [value for value in table_delta.values() if isinstance(value, int | float)]
    worst_table = max([*numeric_table, *paraxial_delta.values()], default=0.0)
    categorical = {
        key: value for key, value in table_delta.items() if not isinstance(value, int | float)
    }
    # A `geometry` class-name mismatch does not gate when `aspheric_nonzero`
    # agrees on every surface. That is the documented all-zero-asphere case --
    # `build_lens` selects `standard` where the tutorial built an `EvenAsphere`
    # with an all-zero coefficient list -- and `SurfaceSpec.has_aspheric_terms`
    # records the measurement that the two are bitwise identical in sag, traced
    # position and accumulated path. Moved out of the gating set and kept in the
    # row, because "identical numbers, different class" is worth seeing and is
    # not worth failing. Any other categorical mismatch still gates.
    non_gating = {}
    if set(categorical) == {"geometry"} and "aspheric_nonzero" not in table_delta:
        non_gating = {"geometry_class_only_all_zero_asphere": categorical.pop("geometry")}
    rows.append(
        Row(
            case=label,
            configuration={
                "notebook": notebook.name,
                "reference_wavelength_um": wavelength_um,
                **harvest,
            },
            descriptor="",
            status="PASS" if worst_table <= 1e-9 and not categorical else "FAIL",
            measured={"paraxial": ours_paraxial, **provenance_detail},
            expected={"paraxial": theirs_paraxial},
            deltas={
                **{
                    f"table.{key}": value
                    for key, value in table_delta.items()
                    if isinstance(value, int | float)
                },
                **{f"paraxial.{key}": value for key, value in paraxial_delta.items()},
            },
            worst_relative_delta=worst_table,
            runtime_s=time.perf_counter() - started,
            note=(
                "surface-table round-trip: the lens rebuilt from the extracted setup against the "
                "tutorial's own. Positions and indices are absolute mm and index differences; "
                "paraxial entries are relative. Categorical mismatches, if any, are under "
                "'categorical'"
            ),
            extra={
                **({"categorical": categorical} if categorical else {}),
                **non_gating,
            },
        )
    )

    # 3. Ray regression.
    # Both the axis and the widest declared field. On axis the optical path is a
    # gating comparison; off axis it is the object-space tilt and is recorded
    # rather than gated -- see `_trace_regression`. Running only one of the two
    # would either lose the OPL check or lose every off-axis surface the widest
    # field is the only thing that reaches.
    for trace_field in dict.fromkeys([(0.0, 0.0), field_deg]):
        started = time.perf_counter()
        trace_source = SourceSpec(wavelength_um=wavelength_um, field_angle_deg=trace_field)
        try:
            ray_deltas, ray_detail = _trace_regression(
                lens=lens, setup=setup, source=trace_source
            )
            gating = [
                value
                for key, value in ray_deltas.items()
                if not key.endswith("_EXPECTED") and not math.isnan(value)
            ]
            status = "PASS" if max(gating, default=math.inf) <= 1e-9 else "FAIL"
            measured: dict[str, Any] = ray_detail
        except Exception as error:
            ray_deltas, status = {}, "FAIL"
            measured = {"exception": type(error).__name__, "message": str(error)}
            gating = []
        rows.append(
            Row(
                case=label,
                configuration={
                    "field_deg": list(trace_field),
                    "wavelength_um": wavelength_um,
                    "num_rings": REGRESSION_NUM_RINGS,
                },
                descriptor="SO_RAY_LAUNCH_TRACE",
                status=status,
                measured=measured,
                expected={"oracle": "Optic.trace on the tutorial's own lens"},
                deltas=ray_deltas,
                worst_relative_delta=max(gating, default=math.inf),
                runtime_s=time.perf_counter() - started,
                note=(
                    "final intersection coordinates, direction cosines, optical-path shape and "
                    "survival mask, rebuilt system vs original, at a fixed hexapolar sampling. "
                    "A delta whose name ends in _EXPECTED is a known convention difference and "
                    "does not decide the status"
                ),
            )
        )

    # 4. The two observables, where the system supports them.
    for descriptor, run in (
        ("SOM_SPOT_DIAGRAM", lambda: native_spot_diagram(
            setup, source, num_rings=REGRESSION_NUM_RINGS, execution=device_execution()
        )),
        ("SOM_PSF", lambda: native_psf(
            setup, source, method="fft", num_rays=128, execution=device_execution()
        )),
    ):
        started = time.perf_counter()
        try:
            result = run()
        except NotImplementedError as error:
            rows.append(
                Row(
                    case=label,
                    configuration={"field_deg": list(field_deg), "wavelength_um": wavelength_um},
                    descriptor=descriptor,
                    status="PASS-refused",
                    measured={"exception": type(error).__name__, "message": str(error)},
                    expected={"refusal": "documented"},
                    deltas={},
                    worst_relative_delta=0.0,
                    runtime_s=time.perf_counter() - started,
                    note=(
                        "the operation refused this system by design; the exception is the "
                        "evidence"
                    ),
                )
            )
            continue
        except Exception as error:
            rows.append(
                Row(
                    case=label,
                    configuration={"field_deg": list(field_deg), "wavelength_um": wavelength_um},
                    descriptor=descriptor,
                    status="FAIL",
                    measured={"exception": type(error).__name__, "message": str(error)},
                    expected={},
                    deltas={},
                    worst_relative_delta=math.inf,
                    runtime_s=time.perf_counter() - started,
                    note="the operation raised on a system the schema accepted",
                )
            )
            continue
        summary = (
            {
                "centroid_m": list(result.centroid_m),
                "rms_radius_m": result.rms_radius_m,
                "geometric_radius_m": result.geometric_radius_m,
            }
            if descriptor == "SOM_SPOT_DIAGRAM"
            else {
                "peak_intensity": result.peak_intensity,
                "peak_index": list(result.peak_index),
                "strehl_ratio": result.strehl_ratio,
                "image_shape": list(result.image_shape),
                "pixel_pitch_m": result.pixel_pitch_m,
                "normalization": result.normalization,
            }
        )
        rows.append(
            Row(
                case=label,
                configuration={"field_deg": list(field_deg), "wavelength_um": wavelength_um},
                descriptor=descriptor,
                status="BASELINE",
                measured=summary,
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=time.perf_counter() - started,
                note=(
                    "the observable exists for this system and is recorded as a value for a "
                    "future comparison. There is no upstream golden -- the tutorial prints "
                    "nothing this descriptor returns -- so this row compared nothing and "
                    "decides nothing; see record.STATUSES on why it is not PASS"
                ),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks",
        type=Path,
        required=True,
        help="directory holding the 25 tutorial .ipynb files at the pinned upstream commit",
    )
    parser.add_argument("--only", action="append", default=None, help="run only these labels")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/che-238-overnight/workstream-a/tier2.json"),
    )
    arguments = parser.parse_args(argv)

    selected = arguments.only or list(IN_SCOPE)
    rows: list[Row] = []
    for label in selected:
        notebook = arguments.notebooks / f"{IN_SCOPE[label]}.ipynb"
        started = time.perf_counter()
        if not notebook.is_file():
            produced = [
                Row(
                    case=label,
                    configuration={"expected_path": str(notebook)},
                    descriptor="",
                    status="BLOCKED",
                    measured={},
                    expected={},
                    deltas={},
                    worst_relative_delta=0.0,
                    runtime_s=0.0,
                    note="the notebook was not fetched, so nothing about this tutorial was run",
                )
            ]
        else:
            try:
                produced = run_tutorial(label, notebook)
            except Exception as error:
                produced = [
                    Row(
                        case=label,
                        configuration={"notebook": notebook.name},
                        descriptor="",
                        status="FAIL",
                        measured={"exception": type(error).__name__, "message": str(error)},
                        expected={},
                        deltas={},
                        worst_relative_delta=math.inf,
                        runtime_s=time.perf_counter() - started,
                        note="the tutorial raised outside any check the driver expected to fail",
                    )
                ]
        rows.extend(produced)
        statuses = ", ".join(sorted({row.status for row in produced}))
        print(
            f"{label:4s} {IN_SCOPE[label][:52]:54s} {len(produced):2d} row(s) [{statuses}]"
            f"  {time.perf_counter() - started:6.1f}s",
            flush=True,
        )

    record = {
        "workstream": "A-tier2",
        "ticket": "CHE-239",
        "produced_by": "benchmarks/verification/ray_tier2.py",
        "upstream_pin": "optiland/optiland@00c0837fbee5d66019a24a1735ff91cd4f9b2646",
        "notebooks": str(arguments.notebooks),
        **provenance(),
        "rows": [row.as_dict() for row in rows],
    }
    return finish(record, path=arguments.out, rows=rows)


if __name__ == "__main__":
    raise SystemExit(main())
