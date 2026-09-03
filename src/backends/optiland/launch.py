"""Where the rays start: the system-bound launch, captured rather than reconstructed.

CHE-219 (R05.8). One operation:

```python
launch(lens, source, *, num_rings=..., aiming=...) -> (RayBundle, declaration)
```

A source can be described without an optical system. A ray launch cannot
-------------------------------------------------------------------------
`problems.SourceSpec` states things that are true of the light alone: infinite or
finite conjugate, field angle, wavelength. Where the rays actually *start* is not
among them. The launch positions and directions depend on the stop, on the
entrance pupil's location and diameter, on every surface preceding the stop, on
the object distance, on the backend's pupil map, and -- off axis or at a finite
conjugate -- on the ray *aimer* and its convergence behaviour. Measured on the
fixtures: an object at infinity launches on the plane
`positions[1] - (EPD - min(positions[1:-1]))`, which is `z = -EPD` on both
fixtures only because both place the first surface at `z = 0`
(-0.49870735054738125 native units on M3-SINGLET-REF, -0.3 on
M3-REVERSE-TELEPHOTO). Either way it is a property of the constructed system and
of nothing else.

So the launch state is a property of **source + system + backend**, and this
module is where it is produced. `sources/` may not produce one, and after this
ticket it does not: `AGENTS.md` puts the stop, the pupil and the aim on the
solver/problem layer, and a `RayBundle` built from caller-supplied points with no
system in scope cannot say whether those points are the entrance pupil, the stop,
the first traced surface, a valid finite-conjugate aim, or nothing in the system
at all.

What this replaces, and why it is not a refactor
-------------------------------------------------
Before this ticket the generated-ray path created a physically meaningful launch
state inside `Optic.trace` and then **discarded it**, so two things downstream
reconstructed it afterwards:

* `rays._declare_measure` regenerated the hexapolar distribution, re-read the
  entrance-pupil diameter and recovered each ray's ring membership *from the
  traced output* in order to assign a pupil-area measure;
* `rays._object_space_reference` regenerated the whole launch state through
  `ray_tracer.ray_generator.generate_rays`, because `Optic.trace` retains none of
  it, and relied on that second invocation reproducing the first bit-for-bit.

Both are consequences of one omission, and both become untenable the moment a
surface aperture vignettes the fan (R05.9): traced output is then no longer a
reliable description of the sampling that existed before the trace. The
regeneration was *measured* faithful and the numbers were right; what was wrong
was that correctness rested on a coincidence nothing checked.

Here the launch state is captured once, before the trace, and everything that
describes it -- the pupil quadrature, the object-space optical-path reference, the
aiming mode -- is declared from that capture and carried forward as data.

The aiming mode is declared, not inherited
------------------------------------------
`RayGenerator` reads `optic.ray_tracer.ray_aiming_config` on every
`generate_rays`, so before this ticket the generated-ray path silently inherited
whatever `RealRayTracer.__init__` happened to set. That is a backend choice
nothing in the project stated. `launch` states it, through the supported
`RealRayTracer.set_aiming(...)` mechanism, and reports it back in the
declaration.

`DEFAULT_AIMING` is `"paraxial"` because that is the constructor default, and
setting it explicitly is **bit-identical** to not setting it at all: measured over
both fixture systems, on and off axis, and over the finite conjugate, the launch
columns agree to 0.0 exactly. So making the choice explicit moves no number. The
other two modes do move numbers, which is the point of exposing them -- off axis
on M3-REVERSE-TELEPHOTO `"iterative"` shifts the launch coordinates by 1.97e-2
native units, while on-axis M3-SINGLET-REF is unchanged at exactly 0.0 because
there is no aiming to do.

The trace path this ticket did *not* unify, and why
---------------------------------------------------
The preferred architecture in the ticket is `source -> launch -> RayBundle ->
trace_rays`, and it is **deliberately not taken here**. The reason is measured,
not stylistic: an object at infinity launches at `z = -EPD`, which is *not* the
first surface after the object surface (`z = 0` on both fixtures), and a point
source launches at the object plane. `rays.require_launch_surface` -- correctly --
refuses a supplied bundle that is not declared on the surface the trace starts
from, so the R05.6 path cannot carry a launch bundle without loosening the one
check that keeps a supplied bundle from silently tracing a different optical
system. On top of that, `to_traced_ray_bundle` composes an optical path with a
different declared reference (`COMPOSED_OPL_REFERENCE_VERSION`) and takes the
measure from the caller, so unifying would change both the declared reference
string and the frozen ray numbers -- which the ticket requires be presented as a
numerical change rather than as code movement.

`Optic.trace` therefore still runs the generated-ray trace, and what changed is
that the launch state feeding it is captured rather than reconstructed.
`tests/physics/test_optiland_finite_conjugate.py` holds the row-for-row
correspondence: the captured launch state traced through the system reproduces
`Optic.trace` bit-identically on both fixtures, on and off axis. Unification is
follow-up work and needs `require_launch_surface` to admit a declared launch
plane.

What is native here and what is not
-----------------------------------
The returned `RayBundle` is neutral and in SI. The second return value is a
**declaration** -- the retained launch facts `rays.to_ray_bundle` needs in order
to describe the traced rays -- and it still carries the object-space reference
term in native units, because that is the unit the accumulator it corrects is in.
It is package-internal state travelling between two functions of this package, in
the same way `to_ray_bundle`'s `lens` argument is.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backends.optiland.rays import (
    LAUNCH_PLANE_WAVEFRONT,
    LAUNCH_POINT_SOURCE,
    NATIVE_LENGTH_M,
    NATIVE_WAVELENGTH_M,
    _host,
    _scalar,
    hexapolar_area_weight_m2,
    hexapolar_ray_count,
    hexapolar_ring_index,
)
from problems import SourceSpec
from representations import ContractError, Frame, RayBundle, ReferenceSurface
from representations.rays import MeasureKind

__all__ = [
    "AIMING_MODES",
    "DEFAULT_AIMING",
    "LAUNCH_GEOMETRY_UNVERIFIED",
    "LAUNCH_OPL_REFERENCE",
    "capture_launch_rays",
    "launch",
    "normalized_field",
]

#: The aiming strategies the pinned solver's `create_ray_aimer` implements, and
#: therefore the ones this adapter will pass to `RealRayTracer.set_aiming`. Read
#: off the backend rather than invented: an unrecognized mode raises from inside
#: the aimer factory on the *next* `generate_rays`, which is far from the call
#: that chose it.
AIMING_MODES: tuple[str, ...] = ("paraxial", "iterative", "robust")

#: The mode used when a caller does not choose one.
#:
#: `"paraxial"` is `RealRayTracer.__init__`'s own default, so this default is the
#: behaviour every frozen ray record was taken under -- see the module docstring
#: for the measurement that says declaring it explicitly moves nothing. A
#: different default here would be this project silently overriding a backend
#: choice while claiming to be making it visible.
DEFAULT_AIMING: str = "paraxial"

#: What the launch bundle's `optical_path_m` is measured from.
#:
#: Not the same declaration as a traced bundle's, and deliberately a different
#: string: `rays.OPL_REFERENCE_VERSION` says "zero at the traced chief ray at the
#: declared plane", which a launch bundle is not. Here the path is the optical
#: path from the incoming wavefront to each launch point -- `n (d0 . r)` for an
#: object at infinity, exactly zero for a point source -- with no piston removed,
#: because there is no traced chief ray yet to remove one against.
LAUNCH_OPL_REFERENCE = "optiland-launch-opl/v1"

#: How closely the launch directions must agree before the incoming bundle counts
#: as collimated.
#:
#: Reused verbatim from the reference implementation's float64 direction bound
#: (`pre-rewrite-2026-08-30:src/backends/optiland/constants.py:_DIRECTION_NORM_TOLERANCE`),
#: and it earns its keep: the measured spread for a collimated off-axis bundle is
#: 2.78e-17, i.e. float64 round-off in the direction generator rather than a real
#: variation. An exact-zero test refuses a perfectly collimated bundle, which is a
#: false refusal of the one case the term exists for -- the off-axis field where
#: omitting it drops the whole convergence tilt. The *plane* test below stays at
#: exact zero, because the launch z spread is measured as exactly 0.0.
_LAUNCH_DIRECTION_TOLERANCE = 1.0e-12

#: What the launch geometry is called when it could not be *verified*.
#:
#: Distinct from `rays.LAUNCH_PLANE_WAVEFRONT` on purpose. The launch points do lie
#: on one plane -- `launch` refuses otherwise -- so a surface exists to declare, but
#: whether that plane is a *wavefront* of the incoming bundle is exactly what
#: `_object_space_reference` declined to establish. Defaulting to the collimated
#: geometry here would name a plane wavefront for a bundle nothing showed to be
#: collimated, which is the approximation this module refuses everywhere else.
#: `rays.declare_optical_path_m` reads the geometry off the term rather than off
#: this, so the two cannot disagree.
LAUNCH_GEOMETRY_UNVERIFIED = (
    "undetermined: the launch points lie on one plane, but no wavefront of the incoming "
    "bundle was established for it"
)

#: The name the launch `ReferenceSurface` carries, per launch geometry. A surface
#: has to be named for a consumer to check it is the one it expected, and these
#: are not interchangeable: one is a plane the collimated bundle crosses, one is
#: the plane through the object point every ray diverges from, and one is a plane
#: whose relation to the wavefront is unknown.
_LAUNCH_SURFACE_NAMES = {
    LAUNCH_PLANE_WAVEFRONT: "launch_plane",
    LAUNCH_POINT_SOURCE: "object_plane",
    LAUNCH_GEOMETRY_UNVERIFIED: "launch_surface",
}


def normalized_field(lens: Any, field_deg: tuple[float, float]) -> tuple[float, float]:
    """The solver's normalized field coordinates for a field angle in degrees.

    The solver aims a ray with `field = max_field * H`, so the coordinate it wants
    is the angle divided by the largest field the constructed system declares. The
    conversion therefore needs the lens rather than the setup alone, and it is done
    here, once, instead of at every call site. It lives beside `launch` because it
    is the first step of aiming a declarative source into a particular system --
    CHE-219 moved it out of `solver.py` for that reason and it has no second
    implementation.

    **The refusal this used to carry is gone, and that is CHE-218's point.** Before
    R05.7 the system record declared a *list* of fields and `max_field` was the
    largest of them, so a field the record had not enumerated normalized to a
    coordinate above 1 and had to be refused -- "trace this system at 3 degrees"
    meant editing the optical system. `build_lens` now declares exactly the field
    being traced, so `max_field` is that field and any field angle is expressible.

    What survives is a check rather than a limit. `max_field` is **read back off
    the constructed lens** and never assumed to equal the requested angle: the
    backend decides it (measured: for a single field it is `hypot(x, y)`), and a
    normalized coordinate above 1 would mean it decided something this function
    does not understand. That is refused rather than passed on, because a
    coordinate above 1 aims at a field nobody asked for.
    """
    max_field = float(lens.fields.max_field)
    x_deg, y_deg = (float(value) for value in field_deg)
    if not (math.isfinite(x_deg) and math.isfinite(y_deg)):
        raise ValueError(f"source field_angle_deg={field_deg!r} is not a finite (x, y) pair")
    if max_field == 0.0:
        # The on-axis case, and the only one where the division is not defined.
        # `build_lens` declared the requested field, so a zero maximum means the
        # request itself was on axis.
        if x_deg != 0.0 or y_deg != 0.0:  # pragma: no cover - build_lens declares the field
            raise ValueError(
                f"source field_angle_deg={field_deg!r} was requested, but the constructed "
                "system reports a maximum field angle of 0 deg, so the solver cannot "
                "express it. Normalizing it to the axis would trace a different field than "
                "the one asked for."
            )
        return (0.0, 0.0)
    normalized = (x_deg / max_field, y_deg / max_field)
    if max(abs(value) for value in normalized) > 1.0:  # pragma: no cover - see docstring
        raise ValueError(
            f"source field_angle_deg={field_deg!r} normalizes to {normalized!r} against the "
            f"maximum field the constructed system reports ({max_field} deg), which exceeds "
            "1. The system was built with exactly this field declared, so the backend's "
            "normalization is not the one this adapter understands."
        )
    return normalized


def capture_launch_rays(
    lens: Any,
    *,
    field: tuple[float, float],
    wavelength_um: float,
    num_rings: int,
    aiming: str = DEFAULT_AIMING,
) -> tuple[Any, Any, Any]:
    """Ask the solver for its launch state, and hand back the native rays unchanged.

    Returns `(rays, pupil_x, pupil_y)`: the native `RealRays` the configured aimer
    produced, and the normalized hexapolar pupil coordinates they were produced
    from. Both halves are needed and neither can be recovered from the other -- the
    pupil coordinate is the ring identity the measure is assigned from, and it does
    not survive into the traced output.

    **This is the same call `Optic.trace` makes**, with the same arguments in the
    same order: `generate_rays(repeat(Hx, N), repeat(Hy, N), distribution.x,
    distribution.y, wavelength)` over a `create_distribution("hexapolar")` fan of
    `num_rings`. Reproducing the argument list rather than the *result* is what
    makes the capture the trace's own launch state instead of a second opinion
    about it.

    Public within this package, and reached directly by the test that holds the
    row-for-row correspondence to `Optic.trace`: that evidence is a native reading
    by nature, and a launch state no test can see is a launch state nobody can
    check.
    """
    import optiland.backend as be
    from optiland.distribution import create_distribution

    if aiming not in AIMING_MODES:
        raise ValueError(
            f"aiming={aiming!r} is not one of {list(AIMING_MODES)}. An unrecognized mode "
            "raises from inside the solver's aimer factory on the next ray generation, "
            "which is a long way from the call that chose it."
        )
    # Declared through the solver's supported mechanism. `RayGenerator.generate_rays`
    # re-reads `ray_tracer.ray_aiming_config` and rebuilds its aimer when the config
    # changes, so setting it here is what the *trace* will use as well -- there is
    # one aiming configuration on the lens, not one per call.
    lens.ray_tracer.set_aiming(aiming)

    distribution = create_distribution("hexapolar")
    distribution.generate_points(num_rings)
    points = int(np.asarray(_host(distribution.x)).size)
    rays = lens.ray_tracer.ray_generator.generate_rays(
        be.repeat(be.atleast_1d(be.array(float(field[0]))), points),
        be.repeat(be.atleast_1d(be.array(float(field[1]))), points),
        distribution.x,
        distribution.y,
        wavelength_um,
    )
    return rays, distribution.x, distribution.y


def launch(
    lens: Any,
    source: SourceSpec,
    *,
    num_rings: int,
    aiming: str = DEFAULT_AIMING,
) -> tuple[RayBundle, dict[str, Any]]:
    """Launch `source` into the constructed system `lens`, without tracing it.

    The operation this ticket exists for: a declarative source plus a constructed
    optical system plus a sampling request, in; the physical launch representation
    plus the declarations the trace path needs, out. It consumes no representation,
    which is why it is not a coupler or an operator: the kind it would carry is
    `source` (`solver` before CHE-224 / R15.1), with `backend="optiland"`. It
    carries none today, because it is deliberately not in the catalog -- see this
    package's `__init__` on why a public launch operation needs a neutral
    signature first.

    Parameters
    ----------
    lens
        The constructed `optiland.optic.Optic`, from `system.build_lens(setup,
        source)`. Native solver state, and required: everything that decides where
        a ray starts is a property of it. This is the argument that makes a launch
        impossible to obtain from `sources/`.
    source
        The declarative illumination. Its `field_angle_deg` is normalized against
        the system's own `max_field` and its `wavelength_um` is what the aimer is
        asked for; `object_distance_mm` is already baked into `lens` by
        `build_lens` and is read back here from the object surface rather than
        re-derived.
    num_rings
        The hexapolar ring count. `1 + 3n(n + 1)` rays
        (`rays.hexapolar_ray_count`).
    aiming
        One of `AIMING_MODES`. See `DEFAULT_AIMING`: the default is the backend's
        own and is bit-identical to not choosing.

    Returns
    -------
    `(bundle, declaration)`.

    `bundle` is a neutral `RayBundle` on the launch surface, carrying the aimed
    positions and directions in SI, the launch amplitude, the hexapolar pupil-area
    `measure_weight` with its `measure_kind`, and -- when the launch geometry
    admits a wavefront -- `optical_path_m` measured from that wavefront with
    `LAUNCH_OPL_REFERENCE` naming it.

    `declaration` is the retained launch state `rays.to_ray_bundle` consumes: the
    normalized field, the wavelength, the ray count, the aiming report, the
    object-space reference term, the measure and its note, and the launch surface.
    Nothing in it is recomputed after the trace.

    Raises
    ------
    TypeError
        `source` is not a `SourceSpec`.
    ValueError
        `num_rings` is below 1, or `aiming` is not a recognized mode.
    ContractError
        The launch state cannot be *declared*: the solver returned columns that
        are not equal-length 1-D arrays, the state is not finite, the launch
        points do not lie on one plane, or the object-space refractive index
        cannot be read from the prescription. These are refusals rather than the
        degradations `rays.declare_optical_path_m` still applies to the
        object-space *term*, because launch is now the producer of the rays: a
        launch state that cannot be described is not a bundle with a missing
        field, it is no bundle at all.
    Exception
        Whatever the backend's aimer raises. An `iterative` or `robust` aim that
        does not converge is the solver's own failure and is deliberately not
        wrapped: it is not a state this adapter could not *describe*, it is a
        launch the backend could not *find*, and relabelling it here would hide
        which of the two happened. Only a launch that comes back readable but
        unusable becomes a `ContractError`.
    ImportError
        optiland is not installed.
    """
    if not isinstance(source, SourceSpec):
        raise TypeError(
            f"launch takes a SourceSpec as its second argument, got "
            f"{type(source).__name__}. An already-materialized RayBundle is not launched: "
            "it is already a launch, and it goes to `trace_rays`."
        )
    if num_rings < 1:
        raise ValueError(
            f"num_rings={num_rings!r} must be at least 1; it is a hexapolar ring count, "
            f"and n rings is {hexapolar_ray_count(1)} rays at n = 1"
        )

    wavelength_um = source.wavelength_um
    field = normalized_field(lens, source.field_angle_deg)
    native, pupil_x, pupil_y = capture_launch_rays(
        lens,
        field=field,
        wavelength_um=wavelength_um,
        num_rings=num_rings,
        aiming=aiming,
    )

    columns = _launch_columns(native)
    ray_count = int(columns["x"].size)
    index = _object_space_index(lens, wavelength_um=wavelength_um)
    object_space = _object_space_reference(lens, columns, index=index)

    plane_spread = float(np.ptp(columns["z"]))
    if plane_spread != 0.0:  # pragma: no cover - measured exactly 0.0 on every fixture
        raise ContractError(
            "MISSING_DECLARATION",
            f"the launch points do not lie on one plane (z spread {plane_spread:.3e} in "
            "native units), so there is no surface to declare the launch bundle on. Both "
            "geometries this adapter launches are planar in z: an object at infinity "
            "spreads over a plane normal to z, and a point source launches from one point.",
            declaration="reference_surface",
        )
    launch_geometry = object_space.get("launch_geometry") or LAUNCH_GEOMETRY_UNVERIFIED
    launch_surface = ReferenceSurface(
        name=_LAUNCH_SURFACE_NAMES[launch_geometry],
        z_m=float(columns["z"][0]) * NATIVE_LENGTH_M,
        medium_index=index,
    )

    measure_weight, measure_kind, measure_note = _declare_measure(
        lens,
        pupil_x=pupil_x,
        pupil_y=pupil_y,
        num_rings=num_rings,
        wavelength_um=wavelength_um,
    )

    # The one wavelength the solver actually stamped on the rays, read back rather
    # than echoed from the request -- the same discipline `configure_execution`
    # applies to the backend it just set.
    wavelengths = np.unique(columns["wavelength"])
    if wavelengths.size != 1:  # pragma: no cover - generate_rays broadcasts one value
        raise ContractError(
            "MISSING_DECLARATION",
            f"the launch state carries {wavelengths.size} distinct wavelengths; a RayBundle "
            "is monochromatic per evaluation and a spectrum is several bundles",
            declaration="wavelength_m",
        )

    optical_path_m, optical_path_reference = _declare_launch_optical_path(
        object_space, launch_geometry=launch_geometry
    )
    bundle = RayBundle(
        positions_m=np.stack([columns[name] for name in ("x", "y", "z")], axis=1)
        * NATIVE_LENGTH_M,
        directions=np.stack([columns[name] for name in ("L", "M", "N")], axis=1),
        wavelength_m=float(wavelengths[0]) * NATIVE_WAVELENGTH_M,
        reference_surface=launch_surface,
        frame=Frame(),
        # sqrt of a real, non-negative power-like weight: a phase-free amplitude,
        # the same mapping `rays.AMPLITUDE_MAPPING` states for a traced bundle.
        # 1.0 for every ray on both fixtures, which is a property of the pinned
        # solver rather than a coincidence: `RayGenerator.generate_rays` seeds
        # `intensity = ones_like(Px)` when there is no apodization, and neither
        # setup declares one. Read rather than assumed, because an apodization is
        # exactly the thing that would make it vary, and pinned by
        # `tests/backends/test_optiland_launch.py`.
        amplitude=np.sqrt(columns["intensity"]),
        optical_path_m=optical_path_m,
        optical_path_reference=optical_path_reference,
        measure_weight=measure_weight,
        measure_kind=measure_kind,
        # CHE-226 (R16). Nothing has divided: this is the generated fan itself, one
        # row per pupil sample. `rays.to_ray_bundle` declares the same of the traced
        # output for the same reason, and the two agree because the trace the
        # declaration describes does not split rays either.
        ray_splitting="unsplit",
    )

    declaration: dict[str, Any] = {
        "field": field,
        "wavelength_um": wavelength_um,
        "num_rings": num_rings,
        "ray_count": ray_count,
        "aiming": {
            "requested": aiming,
            # Read back off the lens, not echoed: this is the configuration the
            # trace's own `generate_rays` will consult.
            "observed": dict(lens.ray_tracer.ray_aiming_config),
            "modes": list(AIMING_MODES),
        },
        "object_space": object_space,
        "measure_weight": measure_weight,
        "measure_kind": measure_kind,
        "measure_note": measure_note,
        "launch_geometry": launch_geometry,
        "launch_surface": launch_surface,
        "optical_path_reference": optical_path_reference,
    }
    return bundle, declaration


def _launch_columns(native: Any) -> dict[str, np.ndarray[Any, Any]]:
    """The captured launch state as host float64 columns, checked for shape and finiteness.

    Deliberately float64 and **not** the trace's precision, for the reason
    `declare_optical_path_m` gives: the object-space term computed from these is a
    piston-and-tilt correction of order 1e4 waves, and computing it in float32
    would inject an error larger than the wavefront it corrects. The *bundle's*
    positions and directions inherit this, which is a difference from the traced
    bundle -- there `_host` preserves the solver's precision so a float32 trace
    stays visibly float32. A launch bundle is a description of where the rays
    began, not a trace result, and its consumer is the reference arithmetic.
    """
    columns = {
        name: np.asarray(_host(value), dtype=np.float64)
        for name, value in (
            ("x", native.x),
            ("y", native.y),
            ("z", native.z),
            ("L", native.L),
            ("M", native.M),
            ("N", native.N),
            ("intensity", native.i),
            ("wavelength", native.w),
            ("accumulator", native.opd),
        )
    }
    shapes = {name: array.shape for name, array in columns.items()}
    if columns["x"].ndim != 1 or len(set(shapes.values())) != 1:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the solver's launch state is not equal-length 1-D columns: {shapes!r}",
            declaration="positions_m",
        )
    if columns["x"].size == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            "the solver generated no launch rays; there is nothing to declare",
            declaration="positions_m",
        )
    unfinite = sorted(
        name for name, array in columns.items() if not bool(np.all(np.isfinite(array)))
    )
    if unfinite:
        raise ContractError(
            "NON_FINITE",
            f"the solver's launch state is not finite in {unfinite}. A launch state that "
            "cannot be read is not a bundle with a missing field; the aimer did not "
            "converge on a launch for this source and system.",
            declaration="positions_m",
        )
    return columns


def _object_space_index(lens: Any, *, wavelength_um: float) -> float:
    """The refractive index of the medium the rays are launched into.

    Read from the prescription's object-space material, because the optical path
    from a wavefront to a launch point is index-weighted and "it is air" is a
    property of two particular fixture systems rather than of lenses.
    """
    index = _scalar(lambda: lens.surfaces.surfaces[0].material_post.n(wavelength_um))
    if index is None or index <= 0.0:  # pragma: no cover - both fixtures read 1.0
        raise ContractError(
            "MISSING_DECLARATION",
            "the object-space refractive index could not be read from the prescription, "
            "so the launch surface has no medium to declare and the optical path from a "
            "wavefront to the launch state -- which is index-weighted -- cannot be formed.",
            declaration="ReferenceSurface.medium_index",
        )
    return index


def _object_space_reference(
    lens: Any, columns: dict[str, np.ndarray[Any, Any]], *, index: float
) -> dict[str, Any]:
    """CHE-41: the optical path from the incoming *wavefront* to each launch point.

    Every precondition the term depends on is checked rather than assumed, and a
    failed check returns `available=False` with the reason. It never returns a
    term it could not verify: an unavailable term becomes a refusal upstream in
    `rays.declare_optical_path_m` when the field is off axis, while a wrong one is
    a wavefront aimed at the wrong image point.

    CHE-219 changed **where the launch state comes from** and nothing about the
    physics. It is now the state `capture_launch_rays` took before the trace,
    passed in as `columns`; it used to be regenerated here after the trace, on the
    measured-but-unchecked premise that a second invocation of the backend
    reproduces the first bit-for-bit.
    """

    def unavailable(reason: str) -> dict[str, Any]:
        return {"available": False, "reason": reason, "offset_native": None, "span_native": None}

    try:
        object_at_infinity = bool(lens.object_surface.is_infinite)
    except Exception as exc:  # pragma: no cover - defensive
        return unavailable(
            f"the object surface could not be read ({type(exc).__name__}), so the launch "
            "geometry is unknown"
        )

    x0, y0, z0 = columns["x"], columns["y"], columns["z"]
    l0, m0, n0 = columns["L"], columns["M"], columns["N"]

    if not object_at_infinity:
        # A POINT SOURCE. The launch state is one point, so it *is* a wavefront --
        # a degenerate sphere of zero radius centred on the object -- and the
        # optical path from that wavefront to each launch point is identically
        # zero. See `_point_source_reference` for the checks and the measurement.
        return _point_source_reference(x0, y0, z0, index=index)

    # AN OBJECT AT INFINITY. The launch points spread over a plane and the
    # directions are common, which is the mirror image of the point source above,
    # and it is why the two cannot share one arithmetic.
    direction_spread = max(float(np.ptp(l0)), float(np.ptp(m0)), float(np.ptp(n0)))
    if direction_spread > _LAUNCH_DIRECTION_TOLERANCE:
        return unavailable(
            f"the launch directions are not common to every ray (spread "
            f"{direction_spread:.3e}), so the incoming bundle is not collimated and a "
            "single plane wavefront does not describe it"
        )
    plane_spread = float(np.ptp(z0))
    if plane_spread > 0.0:  # pragma: no cover - `launch` refuses this before reaching here
        return unavailable(
            f"the launch points do not lie on one plane (z spread {plane_spread:.3e} in "
            "native units), so the seeded reference surface is not the plane this term "
            "assumes"
        )

    # d0 . r_launch, index-weighted. The n0 * z0 part is common to every ray
    # because the launch plane is flat; it is retained rather than dropped so the
    # quantity is the optical path from ONE stated wavefront -- the one through
    # the global origin, perpendicular to d0 -- rather than from an unstated one.
    offset_native = index * (l0 * x0 + m0 * y0 + n0 * z0)
    return {
        "available": True,
        "reason": None,
        "offset_native": offset_native,
        "span_native": float(np.ptp(offset_native)),
        "launch_geometry": LAUNCH_PLANE_WAVEFRONT,
        "launch_direction": (float(l0[0]), float(m0[0]), float(n0[0])),
        "launch_plane_z_native": float(z0[0]),
        "object_space_refractive_index": index,
    }


def _point_source_reference(
    x0: np.ndarray[Any, Any],
    y0: np.ndarray[Any, Any],
    z0: np.ndarray[Any, Any],
    *,
    index: float,
) -> dict[str, Any]:
    """CHE-207: the object-space reference for a finite conjugate, which is zero.

    Not an assumed zero. The reasoning is structural and the premise is checked:

    * every ray of one field leaves the **same point**, so the launch state is a
      single point rather than a surface;
    * a single point is trivially a common wavefront -- the degenerate sphere of
      zero radius centred on it -- so the optical path from that wavefront to each
      launch point is `0` for every ray, exactly, with no arithmetic to round;
    * being constant, it is also a piston, so `declare_optical_path_m`'s chief-ray
      subtraction would remove it even if it were not zero. The declared path is
      therefore referenced to the spherical wavefront diverging from the object
      point, which is the physically meaningful surface for a finite conjugate.

    **Measured directly, not inferred from the infinite-object case**, which is
    what CHE-46 required before this refusal could be lifted. On the
    finite-conjugate singlet at 2f, at three fields -- on axis, off axis in y, and
    off axis in **both** x and y: the launch origin spread is **exactly 0.0** in x,
    y and z; the launch `opd` is **exactly 0.0** with zero spread; the launch
    *directions* spread by ~5e-2, confirming a diverging bundle rather than a
    collimated one; and the captured launch state traced through the system
    reproduces `Optic.trace` bit-identically (max |dx| = max |d opd| = 0.0).
    `tests/physics/test_optiland_finite_conjugate.py` is where each of those is
    asserted, so none of them is a number in a docstring.

    The premise is re-checked here on every call rather than trusted, because it
    is the whole justification: an origin spread that is not zero means the source
    is not a point, the wavefront is not a sphere about one centre, and this zero
    would be wrong. That case is refused rather than approximated -- an extended
    finite source is a different physical problem, not a looser tolerance.
    """
    origin_spread = max(float(np.ptp(x0)), float(np.ptp(y0)), float(np.ptp(z0)))
    if origin_spread > 0.0:
        return {
            "available": False,
            "reason": (
                f"the object is at a finite distance but the launch points do not coincide "
                f"(origin spread {origin_spread:.3e} in native units), so the source is not "
                "a POINT and its launch state is not a single spherical wavefront. An "
                "extended finite source is a different physical problem and this term is "
                "not defined for it."
            ),
            "offset_native": None,
            "span_native": None,
        }
    return {
        "available": True,
        "reason": None,
        # Exactly zero, per ray, by construction rather than by subtraction.
        "offset_native": np.zeros_like(x0),
        "span_native": 0.0,
        "launch_geometry": LAUNCH_POINT_SOURCE,
        "launch_point_native": (float(x0[0]), float(y0[0]), float(z0[0])),
        "object_space_refractive_index": index,
    }


def _declare_measure(
    lens: Any,
    *,
    pupil_x: Any,
    pupil_y: Any,
    num_rings: int,
    wavelength_um: float,
) -> tuple[np.ndarray[Any, Any] | None, MeasureKind, str]:
    """The per-ray pupil area element and its kind, or an honest absence.

    **The layer that chose the sampling is the layer that declares the measure.**
    This used to run after the trace, in `rays._declare_measure`, where it
    regenerated the hexapolar distribution and recovered each ray's ring
    membership from the traced output; the ring identity is now the pupil
    coordinate the launch was *generated from*, so nothing downstream reconstructs
    a quadrature and nothing checks a traced row count to guess one. That matters
    beyond tidiness: once a surface aperture vignettes the fan (R05.9) the traced
    output no longer describes the sampling that produced it, while these
    coefficients still describe the cells that were launched.

    A missing measure is not itself a physics error the way a missing off-axis
    tilt term is: the bundle is still coherent, it is simply unweighted. So this
    returns `(None, "undeclared", reason)` rather than raising, and R07's coupler
    is the thing that refuses to reconstruct from an undeclared measure.
    """
    # float64 by declaration, independent of the trace: the ring assignment is a
    # tolerance test on the ratio rho * num_rings, so it is computed at reference
    # precision whatever the trace runs in.
    x = np.asarray(_host(pupil_x), dtype=np.float64)
    y = np.asarray(_host(pupil_y), dtype=np.float64)

    # The *population* guard, and it is not the same check as the row-count one
    # `to_ray_bundle` runs. That one holds the declaration and the trace to the same
    # number of rows; this one holds the fan to the shape the quadrature is derived
    # for. `hexapolar_area_weight_m2` assigns `pi a^2 / (3 n^2)` per interior-ray
    # with 3/4 at the centre and 1/2 at the rim, which is the correct area element
    # *only* for one centre ray plus `6j` on ring `j`. A backend whose hexapolar
    # layout changed -- different points per ring, or more than one centre point --
    # would otherwise get a silently wrong absolute area rather than an honest
    # absence, and R07's kernel multiplies `measure_weight` into every
    # reconstruction. The pre-CHE-219 code got this check for free from the traced
    # row count; declaring the measure at launch means asking the fan directly.
    expected = hexapolar_ray_count(num_rings)
    if x.size != expected or y.size != expected:
        return (
            None,
            "undeclared",
            f"undeclared: the solver's hexapolar distribution generated {x.size} pupil "
            f"points for {num_rings} rings, but the area element this module assigns is "
            f"derived for 1 + 3n(n + 1) = {expected} -- one centre ray plus 6j on ring j. "
            "The layout is not the one the quadrature is defined for, and scaling the "
            "cells to it anyway would invent an area element.",
        )

    entrance_pupil_diameter_mm = _scalar(lens.paraxial.EPD)
    if entrance_pupil_diameter_mm is None or entrance_pupil_diameter_mm <= 0.0:
        return (
            None,
            "undeclared",
            "undeclared: the entrance pupil diameter could not be read, so the physical "
            "aperture area the relative cell areas scale to is unknown",
        )
    aperture_radius_m = (entrance_pupil_diameter_mm / 2.0) * NATIVE_LENGTH_M
    try:
        ring_index = hexapolar_ring_index(x, y, num_rings)
    except ContractError as exc:  # pragma: no cover - the fan is generated here
        return (None, "undeclared", f"undeclared: {exc.code}: {exc}")
    weight = hexapolar_area_weight_m2(ring_index, num_rings, aperture_radius_m)
    return (
        weight,
        "quadrature_area_m2",
        (
            "quadrature_area_m2: the absolute per-ray entrance-pupil area element a "
            f"{num_rings}-ring hexapolar fan represents, radial-trapezoid corrected at the "
            "centre (3/4) and rim (1/2). Assigned at LAUNCH, from the pupil coordinates the "
            "fan was generated from, while the complete sample and its ring identity are "
            f"still known. Aperture radius {aperture_radius_m!r} m, sum "
            f"{float(weight.sum()):.9e} m^2 against pi a^2 = "
            f"{math.pi * aperture_radius_m**2:.9e} m^2 (ratio "
            f"{float(weight.sum()) / (math.pi * aperture_radius_m**2):.12f}, exactly "
            f"1 + 1/(4 n^2)). Wavelength {wavelength_um!r} um does not enter: the measure is "
            "a property of how the pupil was sampled, not of the light."
        ),
    )


def _declare_launch_optical_path(
    object_space: dict[str, Any], *, launch_geometry: str
) -> tuple[np.ndarray[Any, Any] | None, str | None]:
    """The launch bundle's own optical path, measured from the incoming wavefront.

    This is `object_space["offset_native"]` in metres and nothing else -- the same
    term `rays.declare_optical_path_m` applies to the traced accumulator, which is
    the point: there is one arithmetic for "the optical path from the incoming
    wavefront to the launch point" and both the launch bundle and the traced
    bundle read it from the same place.

    When the term is unavailable the path is left **off** rather than defaulted to
    zero. A `RayBundle` with no optical path is honestly incoherent and
    `require_coherent()` says so; one carrying a zero it did not measure is a
    bundle claiming its launch surface is a wavefront when nothing established
    that.
    """
    if not object_space["available"]:
        return None, None
    return (
        np.asarray(object_space["offset_native"], dtype=np.float64) * NATIVE_LENGTH_M,
        (
            f"{LAUNCH_OPL_REFERENCE}: the optical path from the {launch_geometry} to each "
            "launch point, index-weighted by the object-space medium "
            f"({object_space['object_space_refractive_index']!r}). For an object at "
            "infinity that wavefront is the plane through the GLOBAL ORIGIN normal to the "
            "common launch direction, so the value is n (d0 . r) and is a non-zero "
            "constant even on axis -- it is n * z_launch there, not 0. For a point source "
            "it is the degenerate sphere centred on the object point and the value is "
            "exactly 0. No piston is removed -- there is no traced chief ray yet to remove "
            "one against -- so this is not `rays.OPL_REFERENCE_VERSION` and must not be "
            "read as one."
        ),
    )
