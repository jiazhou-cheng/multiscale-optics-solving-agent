"""Native Optiland rays to a neutral `RayBundle`: the OPL, the pupil, the measure.

CHE-180 (R05.2). The translation half of the anti-corruption layer, and the one
place in the tree where Optiland's native ray state is read. Everything the
package knows about `RealRays` -- that `.i` is a weight and not an amplitude,
that `.opd` is an accumulator and not a declared optical path, that lengths are
millimetres -- stops here.

Nothing below is new physics. Four measured results are being *applied*, and each
is cited where it is used:

`CHE-30` -- the `RealRays.opd` convention
    Absolute accumulated **optical** (index-weighted) path in the lens geometry
    unit, seeded to zero at the ray launch state. For an object at infinity
    `optiland/fields/field_types/angle.py` aims that launch plane at
    `positions[1] - (EPD - min(positions[1:-1]))`, so **the zero moves when the
    aperture changes**. Every part -- sign, unit, physical meaning, reference --
    was established against manufactured geometries with closed-form answers,
    each with the competing hypothesis it rules out, every case exact to float64
    round-off. `tests/physics/test_optiland_opl_convention.py` re-establishes it.

`CHE-32` -- the exit-pupil handoff
    The exported positions at the exit pupil are each ray's image-space
    *asymptote* at that plane, while the accumulator still ends at the final
    traced image surface. Position and phase would otherwise describe different
    planes. Image space is homogeneous, so the correction is exact:
    `n_image * (z_image - z_plane) / N` per ray, with `n_image` read from the
    prescription rather than assumed to be 1.

`CHE-40` -- the required conditioning
    Phase is formed from `OPL_i - OPL_ref`, never from the absolute value. On
    M3-SINGLET-REF the absolute path is 1.0e4 waves and the piston removed here
    is 1219 waves, against 11.7 waves of actual wavefront. A second, less obvious
    benefit: because the reference is a ray from the same trace, the
    aperture-dependent launch-plane zero is common to every ray and cancels
    *exactly* -- which makes the declared OPL invariant under the one thing CHE-30
    warned makes the absolute value meaningless.

`CHE-207` -- the finite conjugate
    A **point source** launches every ray of one field from a single point, which
    is a degenerate spherical wavefront centred on it, so the CHE-41 term is
    exactly zero rather than merely small. Measured directly on a finite-conjugate
    singlet rather than inferred from the infinite-object result, which is what
    the canceled CHE-46 required before the old refusal could be lifted: the
    launch origin spread is exactly 0.0 in all three coordinates, the launch `opd`
    is exactly 0.0, the launch *directions* spread (a diverging bundle, the mirror
    image of the collimated case), and the regenerated launch state reproduces
    `Optic.trace` bit-identically. An origin spread that is not zero is refused
    rather than approximated: an extended finite source is a different physical
    problem.

`CHE-41` -- which *surface* the path is measured from
    That cancellation is exact for the launch plane's **location** and says
    nothing about its **orientation**. `opd` is seeded on a plane perpendicular to
    z, which is a wavefront only for a bundle travelling along z. For an off-axis
    collimated bundle the plane and the wavefront differ by
    `n_object * (d0 . r_launch)` -- linear in the launch coordinate, so a piston
    on axis and a tilt off it. Omitting it leaves a pupil OPL that is a clean
    converging sphere aimed at the *axis* at every field angle: on
    M3-REVERSE-TELEPHOTO at `Hy = 0.2` it retained 0.13% of the required tilt and
    put the focus 209 um from the traced chief-ray intersection, with a
    0.072-wave-P-V residual that looked perfectly healthy. That is why it survived
    three tickets of on-axis verification, and why the term is measured here and
    *refused* rather than defaulted when it cannot be.

Three quantities, kept apart
----------------------------
`amplitude` is `sqrt(intensity)`. `RealRays.i` is a real, non-negative,
power-like per-ray weight and carries **no phase**; every radian of a
reconstructed field comes from the optical path. Reading `i` as a complex
amplitude would be a category error, and this module states the mapping rather
than leaving it to be inferred.

`measure_weight` is the absolute per-ray pupil area element, in square metres,
and is **never** folded into the amplitude. The reference implementation did fold
it in (CHE-47, `amplitude = sqrt(w) * dA`); `representations.RayBundle` separates
the two, so the area element travels as the declared measure and the coupler that
discretizes the aperture integral applies it. That is the same physics with the
declaration in the place that can refuse it.

`optical_path_m` is the *declared* path from `declare_optical_path_m`, never the
native accumulator. `to_ray_bundle` refuses an optical path whose reference does
not carry `OPL_REFERENCE_VERSION`, which is what makes "pass `opd_native` as an
OPL" a structured refusal rather than a piston-and-sign gamble.

Why the quadrature lives here
-----------------------------
The hexapolar area weight has exactly one producer -- this solver -- so it stays
inside the package rather than becoming a generic quadrature framework nothing
else uses yet. The *contract* is elsewhere and unchanged: `measure_kind` on
`RayBundle` must be declared or the consumer refuses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numerics import dtype_of
from representations import (
    UNVERIFIED,
    ContractError,
    Frame,
    MeasureKind,
    RayBundle,
    ReferenceSurface,
    direction_norm_tolerance,
)

__all__ = [
    "AMPLITUDE_MAPPING",
    "LAUNCH_PLANE_WAVEFRONT",
    "LAUNCH_POINT_SOURCE",
    "NATIVE_LENGTH_M",
    "NATIVE_WAVELENGTH_M",
    "OPL_REFERENCE_VERSION",
    "REFERENCE_SURFACES",
    "declare_optical_path_m",
    "exit_pupil",
    "hexapolar_area_weight_m2",
    "hexapolar_ray_count",
    "hexapolar_ring_index",
    "require_declared_optical_path",
    "to_ray_bundle",
]

#: The lens geometry unit, in metres. CHE-30 part 4: `RealRays.opd` and every
#: coordinate are in the prescription's geometry unit, millimetres for every
#: problem this project states, and `opd` scales exactly with the prescription
#: (measured: a 10x geometry gives a 10x `opd`, ratio error 0.0).
NATIVE_LENGTH_M = 1.0e-3

#: `RealRays.w` and every Optiland wavelength are micrometres.
NATIVE_WAVELENGTH_M = 1.0e-6

#: The surfaces a bundle may be declared on, and what each one means.
#:
#: `image_surface`
#:     The final traced surface. Positions are the traced intersections,
#:     untouched, which is what keeps the frozen ray fingerprints reproducible.
#: `exit_pupil`
#:     The image of the stop in image space, located by `Paraxial.XPL()`, which is
#:     **signed and measured from the image surface** rather than from the global
#:     origin. Positions there are each ray's image-space *asymptote*, not a
#:     physical intersection: the pupil is frequently virtual (on both fixture
#:     systems it is), so the extended line passes back through glass the ray
#:     never travelled in that state. That is the construction the exit pupil is
#:     defined by and what a wavefront-over-the-pupil calculation wants, and it is
#:     not "where the ray is".
REFERENCE_SURFACES: tuple[str, ...] = ("image_surface", "exit_pupil")

#: The version of the OPL reference this module declares, and the prefix
#: `to_ray_bundle` requires before it will carry an optical path.
#:
#: The prefix is load-bearing rather than decorative, and it is the reference
#: implementation's own device: a declaration that only a specific producer can
#: write is what stops the raw accumulator being handed over under a plausible
#: label. Versioned rather than edited in place because a handoff convention is
#: part of the boundary contract -- v1 (CHE-33) named the reference a plane and
#: used it as a wavefront, which was exactly right on axis and dropped the whole
#: convergence tilt off it.
OPL_REFERENCE_VERSION = "optiland-declared-opl/v2"

#: How `RealRays.i` becomes an amplitude. Stated because it is a modelling
#: decision, and this is the producer that makes it.
AMPLITUDE_MAPPING = (
    "amplitude = sqrt(intensity); intensity is Optiland RealRays.i, a real "
    "non-negative power-like per-ray weight. It carries no phase, so every radian "
    "of a reconstructed field comes from optical_path_m. No per-ray area factor is "
    "multiplied in and no 1/N is applied: the area element travels separately as "
    "measure_weight, and a traced ray set is a physical ensemble rather than a "
    "Monte-Carlo sample of one."
)

#: Absolute tolerance, in units of ring spacing, for a normalized pupil radius to
#: count as landing on a hexapolar ring. `optiland.distribution` places ring `j`
#: at exactly `rho = j / num_rings` in float64
#: (`r = linspace(0, 1, num_rings + 1)`), so an un-vignetted fan agrees to
#: float64 round-off. A bundle that does not is refused rather than mis-binned.
_RING_TOLERANCE = 1.0e-6

#: The two launch geometries the pinned solver produces, named so the declared
#: optical path can say which one its reference is, rather than leaving a reader
#: to infer it from the problem.
#:
#: They are mirror images, and that is the whole reason they need separate
#: arithmetic: at infinity the launch **directions** are common and the origins
#: spread over a plane, so the reference is a plane wavefront and the offset is
#: `n * (d0 . r_launch)`. For a point source the **origin** is common and the
#: directions spread, so the reference is a sphere about that point and the offset
#: is exactly zero.
LAUNCH_PLANE_WAVEFRONT = "plane wavefront of a collimated bundle, launched on a plane normal to z"
LAUNCH_POINT_SOURCE = "spherical wavefront diverging from a single object point"

#: How closely the regenerated launch directions must agree before the incoming
#: bundle counts as collimated.
#:
#: Reused verbatim from the reference implementation's float64 direction bound
#: (`pre-rewrite-2026-08-30:src/solvers/optiland/constants.py:_DIRECTION_NORM_TOLERANCE`),
#: and it earns its keep: the measured spread for a collimated off-axis bundle is
#: 2.78e-17, i.e. float64 round-off in the direction generator rather than a real
#: variation. An exact-zero test refuses a perfectly collimated bundle, which is a
#: false refusal of the one case the term exists for -- the off-axis field where
#: omitting it drops the whole convergence tilt. The *plane* test below stays at
#: exact zero, because the launch z spread is measured as exactly 0.0.
_LAUNCH_DIRECTION_TOLERANCE = 1.0e-12

#: What Optiland reports for a ray it clipped: the intensity is zeroed and the
#: row is kept with its state frozen at the clip. So a positive intensity is the
#: aliveness test, and non-finite geometry is treated the same way -- a NaN
#: position is not a position.
_ALIVE_RULE = (
    "intensity > 0 and every geometric component finite. Optiland clips by zeroing "
    "RealRays.i rather than by removing the row, so a zeroed ray still carries a "
    "position frozen at the clip; it is dropped here rather than travelling inside an "
    "artifact whose contract forbids reading it."
)


def hexapolar_ray_count(num_rings: int) -> int:
    """How many rays an un-vignetted `num_rings` hexapolar fan produces.

    `1 + 3n(n + 1)`: the central ray plus `6j` rays on each of `n` rings. Exposed
    because it is the row count a caller has to match to get a declared measure,
    and computing it at the call site is how a fan gets requested by density and
    checked against a guess.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    return 1 + 3 * num_rings * (num_rings + 1)


def hexapolar_ring_index(
    pupil_x: np.ndarray[Any, Any], pupil_y: np.ndarray[Any, Any], num_rings: int
) -> np.ndarray[Any, Any]:
    """Recover each ray's hexapolar ring index from its normalized pupil coordinate.

    `pupil_x` / `pupil_y` are the normalized entrance-pupil coordinates the fan was
    sampled on -- the unit disk, ring `j` at radius `j / num_rings` for
    `j = 0 .. num_rings`, with `j = 0` the single central ray.

    Raises:
        ContractError: `MEASURE_UNDECLARED` if any ray's normalized radius misses
            every ring by more than `_RING_TOLERANCE`. That means the fan was
            vignetted -- a dropped ray shifts which rows correspond to which ring
            -- or was never hexapolar, and in either case no area element can be
            assigned rather than one being guessed.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    x = np.asarray(pupil_x, dtype=np.float64)
    y = np.asarray(pupil_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"pupil_x {x.shape} must match pupil_y {y.shape}",
            declaration="pupil_y",
        )

    ring_float = np.hypot(x, y) * num_rings
    ring_index = np.round(ring_float).astype(np.int64)
    residual = np.abs(ring_float - ring_index)
    if bool(np.any(residual > _RING_TOLERANCE)) or bool(
        np.any((ring_index < 0) | (ring_index > num_rings))
    ):
        worst = int(np.argmax(residual))
        raise ContractError(
            "MEASURE_UNDECLARED",
            (
                f"{int(np.sum(residual > _RING_TOLERANCE))} of {x.size} rays do not land "
                f"on a ring of a {num_rings}-ring hexapolar fan; the worst has normalized "
                f"radius {float(np.hypot(x, y)[worst]):.9f} against the nearest ring at "
                f"{float(ring_index[worst]) / num_rings:.9f}. No pupil area element can be "
                "assigned to these rays, and assigning a uniform one would invent a "
                "quadrature."
            ),
            declaration="measure_weight",
            remedy=(
                "Declare a quadrature measure only for an un-vignetted hexapolar fan, "
                f"where the traced count is 1 + 3n(n + 1) = {hexapolar_ray_count(num_rings)}. "
                "Otherwise leave measure_kind 'undeclared' so the consumer refuses."
            ),
        )
    return ring_index


def hexapolar_area_weight_m2(
    ring_index: np.ndarray[Any, Any], num_rings: int, aperture_radius_m: float
) -> np.ndarray[Any, Any]:
    """The absolute per-ray pupil area element, in square metres.

    Ring `j`'s nominal cell is `pi a^2 / (3 n^2)` -- the aperture area over the
    `3n^2` rays a hexapolar fan asymptotically approaches. Every interior ring's
    `6j` points share that cell exactly, which is the right quadrature there. Two
    measured boundary corrections:

    * the central ray (`j = 0`) gets **3/4** of the nominal cell: it represents a
      disk of radius `a / (2n)`, i.e. `pi a^2 / (4 n^2)`.
    * the outermost ring (`j = n`) gets **1/2**: it sits exactly on `rho = a` and
      represents only the inner half of its annulus, there being no ray beyond the
      rim to average with.

    With these, `sum_i dA_i = pi a^2 (1 + 1/(4 n^2))` exactly, so the total
    converges on the aperture area as the ring count grows. That convergence is
    the point: without an *absolute* area element the reconstructed discrete power
    grew as `(ray count)^2.0024` instead of converging, and treating every ray as
    an equal-weight sample left a sensor-plane residual of 3.84e-3 that the
    corrected measure collapses to 4.07e-4.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    if not math.isfinite(aperture_radius_m) or aperture_radius_m <= 0.0:
        raise ContractError(
            "UNIT_NOT_SI",
            f"aperture_radius_m must be positive and finite, got {aperture_radius_m!r}; it "
            "is the physical radius the relative cell areas are scaled to",
            declaration="measure_weight",
        )
    index = np.asarray(ring_index)
    nominal_m2 = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)
    weight = np.full(index.shape, nominal_m2, dtype=np.float64)
    weight[index == 0] = 0.75 * nominal_m2
    weight[index == num_rings] = 0.5 * nominal_m2
    return weight


def _host(value: Any) -> np.ndarray[Any, Any]:
    """One solver array on the host, with its precision preserved.

    Deliberately not `np.asarray(..., dtype=np.float64)`. Forcing float64 was the
    single line that made a float32 or GPU trace indistinguishable from a float64
    host one downstream; for the default host/float64 path this is a no-op, so the
    frozen fingerprints are unchanged.
    """
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


def _scalar(thunk: Any) -> float | None:
    """A finite scalar read from the system, or `None` when it cannot be read.

    Everything including the attribute lookup happens inside the guard: an
    absence has to degrade to "not available" so a caller can refuse on it,
    rather than turning into an unrelated crash.
    """
    try:
        value = float(np.asarray(_host(thunk())).ravel()[0])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def exit_pupil(lens: Any, *, image_plane_z_mm: float) -> dict[str, Any]:
    """Read the exit pupil from the system, and say what the reading means.

    `Paraxial.XPL()` is signed and measured **from the image surface**, so the
    plane is `image_z + XPL`. Getting that wrong yields a plausible plane rather
    than an error, which is why it is stated here once.

    Raises:
        ContractError: `MISSING_DECLARATION` when the paraxial solver cannot
            supply `XPL()`/`XPD()`, or supplies a non-finite one. A telecentric or
            degenerate configuration has no finite exit pupil and this module will
            not substitute one -- an unresolved plane must not be mistaken for a
            plane at z = 0.
    """
    location_from_image_mm = _scalar(lens.paraxial.XPL)
    diameter_mm = _scalar(lens.paraxial.XPD)
    if location_from_image_mm is None or diameter_mm is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "an exit-pupil reference surface was requested, but the system's paraxial "
            f"solver did not supply a finite XPL()/XPD() (XPL={location_from_image_mm!r}, "
            f"XPD={diameter_mm!r}). The plane is read from the system, never guessed, so "
            "this refuses rather than exporting rays against an invented reference.",
            declaration="reference_surface",
            remedy="Declare 'image_surface', or use a system with a finite exit pupil.",
        )

    pupil_z_mm = image_plane_z_mm + location_from_image_mm
    beyond = [
        z
        for z in (
            _scalar(lambda surface=surface: surface.geometry.cs.z)
            for surface in lens.surfaces.surfaces[:-1]
        )
        if z is not None and z > pupil_z_mm
    ]
    return {
        "z_mm": pupil_z_mm,
        "location_from_image_mm": location_from_image_mm,
        "diameter_mm": diameter_mm,
        # A virtual pupil is not a wrong plane, but it changes what a position at
        # that plane is, so the fact is reported rather than left to be assumed.
        "is_virtual": bool(beyond),
        "refracting_surfaces_beyond_pupil_z_mm": beyond,
    }


def _object_space_reference(
    lens: Any,
    *,
    field: tuple[float, float],
    wavelength_um: float,
    num_rings: int,
    traced_count: int,
) -> dict[str, Any]:
    """CHE-41: the optical path from the incoming *wavefront* to each launch point.

    Every precondition the term depends on is checked rather than assumed, and a
    failed check returns `available=False` with the reason. It never returns a
    term it could not verify: an unavailable term becomes a refusal upstream,
    while a wrong one is a wavefront aimed at the wrong image point.

    The launch state is regenerated through the public
    `ray_tracer.ray_generator.generate_rays` over the same hexapolar distribution
    `Optic.trace` builds, because `Optic.trace` returns the traced rays and keeps
    no record of where they started. That regeneration reproduces the trace
    bit-identically (measured: max |dx| = max |dy| = max |d opd| = 0.0 over 3169
    rays), which is the only reason a per-ray term measured from it describes the
    exported rays row for row.
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

    try:
        import optiland.backend as be
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rings)
        points = int(np.asarray(_host(distribution.x)).size)
        field_x = be.atleast_1d(be.array(float(field[0])))
        field_y = be.atleast_1d(be.array(float(field[1])))
        launch = lens.ray_tracer.ray_generator.generate_rays(
            be.repeat(field_x, points),
            be.repeat(field_y, points),
            distribution.x,
            distribution.y,
            wavelength_um,
        )
    except Exception as exc:
        return unavailable(
            "the launch state could not be regenerated from "
            f"ray_tracer.ray_generator.generate_rays ({type(exc).__name__}: {exc}); "
            "Optic.trace does not retain it, so there is nothing to measure the "
            "object-space reference from"
        )

    def reference(value: Any) -> np.ndarray[Any, Any]:
        # Deliberately float64 and NOT the trace's precision. This is a
        # piston-and-tilt correction of order 1e4 waves; computing it in float32
        # would inject an error larger than the wavefront it corrects. Declared
        # here rather than inherited.
        return np.asarray(_host(value), dtype=np.float64)

    x0, y0, z0 = reference(launch.x), reference(launch.y), reference(launch.z)
    l0, m0, n0 = reference(launch.L), reference(launch.M), reference(launch.N)

    if x0.size != traced_count:
        return unavailable(
            f"the regenerated launch state has {x0.size} rays but the trace exported "
            f"{traced_count}; the two cannot be matched row for row"
        )
    if not all(np.all(np.isfinite(column)) for column in (x0, y0, z0, l0, m0, n0)):
        return unavailable("the regenerated launch state is not finite")

    index = _scalar(lambda: lens.surfaces.surfaces[0].material_post.n(wavelength_um))
    if index is None or index <= 0.0:
        return unavailable(
            "the object-space refractive index could not be read from the prescription, "
            "and the optical path from a wavefront to the launch state is index-weighted"
        )

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
    if plane_spread > 0.0:
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
    collimated one; and the regenerated launch state traced through the system
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


def declare_optical_path_m(
    opd_native: np.ndarray[Any, Any],
    *,
    direction_z: np.ndarray[Any, Any],
    pupil_radius_m: np.ndarray[Any, Any],
    image_z_mm: float,
    plane_z_mm: float,
    image_space_index: float,
    object_space: dict[str, Any],
    on_axis: bool,
) -> tuple[np.ndarray[Any, Any], str]:
    """Turn the native accumulator into a declared optical path, or refuse.

    **This is the only function that produces an admissible optical path**, and
    the reference string it returns is the only one `to_ray_bundle` accepts. Four
    steps, in this order, each the application of a measured result:

    1. native unit -> metres (CHE-30 part 4);
    2. transfer from the traced image surface to the declared plane (CHE-32).
       Exact rather than approximate, because image space is homogeneous;
    3. move the reference from Optiland's launch *state* onto the incoming
       *wavefront*, when and only when the term varies across the bundle. A
       constant term is a piston that step 4 removes exactly, so adding it would
       only spend float precision on something that cannot survive -- and not
       adding it is what keeps every on-axis number bit-identical rather than one
       rounding away. Which wavefront depends on the source, and the declaration
       names it:

       * an object at **infinity** (CHE-41): the launch plane is not a wavefront of
         a tilted bundle, and the difference `n * (d0 . r_launch)` is the whole
         convergence tilt off axis;
       * a **point source** (CHE-207): the launch state is one point, so it already
         *is* a wavefront and the term is exactly zero;
    4. remove the chief-ray piston (CHE-40), sign 'ray minus chief', so a larger
       value means a longer optical path.

    Computed in float64 regardless of the trace's precision, for the reason step 3
    gives: the removed piston is of order 1e4 waves and the signal is of order 10.

    Raises:
        ContractError: `MISSING_DECLARATION` when the object-space term is
            unavailable and the field is *not* on axis. That case cannot be
            repaired downstream -- the missing quantity is a function of the launch
            coordinate and no object-space coordinate survives into the exported
            pupil arrays -- and declaring the path without it produces a clean
            converging sphere aimed at the axis, 209 um from the traced chief-ray
            intersection with a 0.072-wave residual that looks healthy.
    """
    optical_path_m = np.asarray(opd_native, dtype=np.float64) * NATIVE_LENGTH_M

    # 2. Image surface -> declared plane. A no-op when they are the same plane, so
    #    there is one code path rather than a branch that can disagree with itself.
    if bool(np.any(np.asarray(direction_z) == 0.0)):
        raise ContractError(
            "NON_UNIT_DIRECTION",
            "a ray has N = 0 and never reaches the traced image surface, so its optical "
            "path cannot be referred to the declared plane",
            declaration="directions",
        )
    step_m = ((image_z_mm - plane_z_mm) * NATIVE_LENGTH_M) / np.asarray(
        direction_z, dtype=np.float64
    )
    optical_path_m = optical_path_m - image_space_index * step_m

    # 3. Launch plane -> incoming wavefront.
    if not object_space["available"]:
        if not on_axis:
            raise ContractError(
                "MISSING_DECLARATION",
                (
                    "this trace is off axis and the object-space reference term "
                    "n_object * (d0 . r_launch) is unavailable, so the accumulated path is "
                    "measured from a plane perpendicular to z rather than from a wavefront "
                    f"of the incoming bundle. The solver declined it because: "
                    f"{object_space['reason']}"
                ),
                declaration="optical_path_reference",
                remedy=(
                    "Trace on axis, or make the term available. It cannot be repaired "
                    "downstream: it is linear in the LAUNCH coordinate and the exported "
                    "pupil arrays carry no object-space coordinate to reconstruct it from."
                ),
            )
        object_space_note = (
            "unavailable, and the traced field is on axis, so the incoming bundle travels "
            "along z, Optiland's launch plane IS a wavefront of it, and the term would be "
            f"a constant that step 4 removes exactly. Accepted for this field only. Reason: "
            f"{object_space['reason']}"
        )
    elif object_space["launch_geometry"] == LAUNCH_POINT_SOURCE:
        # A point source. The term is exactly zero for every ray, because the
        # launch point IS the wavefront -- see `_point_source_reference`. Nothing
        # to add, and the reference surface named in the declaration below is the
        # sphere rather than the plane.
        object_space_note = (
            "exactly zero for every ray: the source is a single point, so the launch state "
            "is a degenerate spherical wavefront centred on it and there is no optical path "
            "from that wavefront to the launch point. Measured, not assumed -- the launch "
            f"origin spread is 0.0 in all three coordinates at "
            f"{object_space['launch_point_native']!r} in native units"
        )
    elif object_space["span_native"] == 0.0:
        object_space_note = (
            "present and constant across the bundle: a pure piston, not applied, because "
            "step 4 removes it exactly and adding it would move an on-axis number that "
            "nothing about this reference has any business moving"
        )
    else:
        optical_path_m = optical_path_m + object_space["offset_native"] * NATIVE_LENGTH_M
        object_space_note = (
            "applied: the term varies across the bundle by "
            f"{object_space['span_native'] * NATIVE_LENGTH_M:.6e} m, so the omission is a "
            "tilt rather than a piston, and off axis this term IS the convergence tilt"
        )

    # 4. Remove the chief-ray piston.
    radius = np.asarray(pupil_radius_m, dtype=np.float64)
    chief = int(np.argmin(radius))
    reference_m = float(optical_path_m[chief])
    relative_m = optical_path_m - reference_m

    # Which surface the path is measured FROM, named rather than assumed. Getting
    # this wrong in the declaration would be worse than getting it wrong in the
    # arithmetic: a consumer would be told the reference is a plane wavefront when
    # it is a diverging sphere, and no downstream check reads the object distance.
    reference_wavefront = object_space.get("launch_geometry") or (
        f"{LAUNCH_PLANE_WAVEFRONT} (assumed: the term was unavailable and the field is "
        "on axis, where the two references agree)"
    )
    declaration = (
        f"{OPL_REFERENCE_VERSION}: zero at the traced chief ray (smallest pupil radius, "
        f"row {chief}, rho = {float(radius[chief]):.6e} m) evaluated at the declared plane "
        f"z = {plane_z_mm * NATIVE_LENGTH_M!r} m; sign 'ray minus chief', so a larger value "
        f"is a longer optical path. The SURFACE the path is measured from is the "
        f"{reference_wavefront}, not the solver's launch state. Image-space index "
        f"{image_space_index!r}, read from the prescription. Object-space reference term: "
        f"{object_space_note}. Removed piston {reference_m:.9e} m."
    )
    return relative_m, declaration


def to_ray_bundle(
    lens: Any,
    traced: Any,
    *,
    field: tuple[float, float],
    wavelength_um: float,
    num_rings: int,
    reference_surface: str,
) -> tuple[RayBundle, dict[str, Any]]:
    """Translate one native trace into a neutral `RayBundle` plus diagnostics.

    `field` is the normalized pupil-field coordinate pair the trace was given --
    the solver's own spelling, which is why this function is internal to the
    package and `solvers.optiland.trace` is the public entry point.

    Returns the bundle and a diagnostics mapping. Nothing in either is native: no
    `RealRays`, no intensity, no accumulator, no millimetre.

    Raises:
        ContractError: the exit pupil could not be resolved, the image-space index
            could not be read, the optical path could not be declared, or every
            traced ray was clipped.
    """
    if reference_surface not in REFERENCE_SURFACES:
        raise ContractError(
            "MISSING_DECLARATION",
            f"reference_surface must be one of {list(REFERENCE_SURFACES)}, got "
            f"{reference_surface!r}. There is no default: the difference between the two "
            "is the whole pupil-to-focus distance.",
            declaration="reference_surface",
        )

    native = {
        name: _host(value)
        for name, value in (
            ("x", traced.x),
            ("y", traced.y),
            ("z", traced.z),
            ("L", traced.L),
            ("M", traced.M),
            ("N", traced.N),
            ("intensity", traced.i),
            ("wavelength", traced.w),
            ("accumulator", traced.opd),
        )
    }
    shapes = {name: array.shape for name, array in native.items()}
    if native["x"].ndim != 1 or len(set(shapes.values())) != 1:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the solver returned ray columns that are not equal-length 1-D arrays: {shapes!r}",
            declaration="positions_m",
        )
    traced_count = int(native["x"].size)
    if traced_count == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            "the solver returned no rays; there is nothing to declare",
            declaration="positions_m",
        )

    alive = (
        (native["intensity"] > 0)
        & np.isfinite(native["x"])
        & np.isfinite(native["y"])
        & np.isfinite(native["z"])
        & np.isfinite(native["L"])
        & np.isfinite(native["M"])
        & np.isfinite(native["N"])
        & np.isfinite(native["accumulator"])
    )
    alive_count = int(np.count_nonzero(alive))
    if alive_count == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            f"all {traced_count} traced rays were clipped or returned non-finite state, so "
            "the bundle would be empty. " + _ALIVE_RULE,
            declaration="positions_m",
        )
    if alive_count != traced_count:
        # A clipped ray's state is frozen at the clip, and it is dropped at the end
        # of this function -- but every array below is still full length so the
        # regenerated pupil and launch states can be matched row for row. Harmless
        # placeholders keep the intermediate arithmetic finite (a frozen ray can
        # legitimately have N = 0, which would otherwise divide by zero in a
        # quantity that is then discarded). Substituting is safe precisely because
        # nothing computed from a dead row survives into the bundle.
        native = {name: array.copy() for name, array in native.items()}
        for name, placeholder in (
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("L", 0.0),
            ("M", 0.0),
            ("N", 1.0),
            ("accumulator", 0.0),
        ):
            native[name][~alive] = placeholder

    wavelengths = np.unique(native["wavelength"])
    if wavelengths.size != 1:
        raise ContractError(
            "MISSING_DECLARATION",
            f"the trace carries {wavelengths.size} distinct wavelengths; a RayBundle is "
            "monochromatic per evaluation and a spectrum is several bundles",
            declaration="wavelength_m",
        )
    wavelength_m = float(wavelengths[0]) * NATIVE_WAVELENGTH_M

    image_z_mm = _scalar(lambda: lens.surfaces.surfaces[-1].geometry.cs.z)
    if image_z_mm is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "the final surface's axial position could not be read from the system, so no "
            "reference surface can be located",
            declaration="reference_surface",
        )

    if reference_surface == "exit_pupil":
        pupil = exit_pupil(lens, image_plane_z_mm=image_z_mm)
        plane_z_mm = pupil["z_mm"]
        # Each ray's image-space asymptote at the pupil plane. A reparameterization
        # along the ray, so directions are the traced values and no optical path is
        # added or removed here -- the OPL transfer is step 2 of the declaration.
        step_mm = (plane_z_mm - native["z"]) / native["N"]
        position_mm = (
            native["x"] + native["L"] * step_mm,
            native["y"] + native["M"] * step_mm,
            np.full_like(native["x"], plane_z_mm),
        )
    else:
        pupil = None
        plane_z_mm = image_z_mm
        # The traced coordinates, untouched, which is what keeps the frozen ray
        # fingerprints reproducible.
        position_mm = (native["x"], native["y"], native["z"])

    image_space_index = _scalar(
        lambda: lens.surfaces.surfaces[-1].material_pre.n(wavelength_um)
    )
    if image_space_index is None or image_space_index <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the image-space refractive index could not be read from the prescription, and "
            "the optical path between the traced image surface and the declared plane is "
            "index-weighted. 'It is air' is a property of two particular systems, not of "
            "lenses.",
            declaration="reference_surface",
        )

    object_space = _object_space_reference(
        lens,
        field=field,
        wavelength_um=wavelength_um,
        num_rings=num_rings,
        traced_count=traced_count,
    )
    # The chief ray is the smallest pupil radius among the rays that survived: a
    # clipped ray's frozen position could otherwise win the argmin and make the
    # removed piston the path of a ray nobody is allowed to read.
    pupil_radius_m = np.where(
        alive, np.hypot(position_mm[0], position_mm[1]) * NATIVE_LENGTH_M, np.inf
    )
    optical_path_m, optical_path_reference = declare_optical_path_m(
        native["accumulator"],
        direction_z=native["N"],
        pupil_radius_m=pupil_radius_m,
        image_z_mm=image_z_mm,
        plane_z_mm=plane_z_mm,
        image_space_index=image_space_index,
        object_space=object_space,
        on_axis=field == (0.0, 0.0),
    )

    measure_weight, measure_kind, measure_note = _declare_measure(
        lens, num_rings=num_rings, traced_count=traced_count, wavelength_um=wavelength_um
    )

    positions = np.stack([column[alive] for column in position_mm], axis=1) * NATIVE_LENGTH_M
    directions = np.stack([native[name][alive] for name in ("L", "M", "N")], axis=1)
    bundle = RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=wavelength_m,
        reference_surface=ReferenceSurface(
            name=reference_surface,
            z_m=plane_z_mm * NATIVE_LENGTH_M,
            medium_index=image_space_index,
        ),
        frame=Frame(),
        # sqrt of a real, non-negative power-like weight: a phase-free amplitude.
        # `adopt_array(widen_real=True)` widens it to the complex dtype of the
        # SAME precision, so a float32 trace does not gain ten digits here.
        amplitude=np.sqrt(native["intensity"][alive]),
        optical_path_m=optical_path_m[alive],
        optical_path_reference=optical_path_reference,
        measure_weight=None if measure_weight is None else measure_weight[alive],
        measure_kind=measure_kind,
    )
    require_declared_optical_path(bundle)

    diagnostics: dict[str, Any] = {
        "traced_ray_count": traced_count,
        "alive_ray_count": alive_count,
        "clipped_ray_count": traced_count - alive_count,
        "alive_rule": _ALIVE_RULE,
        "reference_surface": reference_surface,
        "reference_surface_z_m": plane_z_mm * NATIVE_LENGTH_M,
        "traced_image_surface_z_m": image_z_mm * NATIVE_LENGTH_M,
        "image_space_refractive_index": image_space_index,
        "amplitude_mapping": AMPLITUDE_MAPPING,
        "measure": measure_note,
        "optical_path_reference": optical_path_reference,
        "object_space_reference_available": object_space["available"],
        "object_space_reference_reason": object_space["reason"],
        "object_space_reference_span_m": (
            None
            if object_space["span_native"] is None
            else object_space["span_native"] * NATIVE_LENGTH_M
        ),
        "observed_dtype": str(dtype_of(positions)),
        "direction_norm_tolerance": direction_norm_tolerance(dtype_of(directions)),
        "polarization": "missing; RealRays carries no polarization state, and none is fabricated",
        "coherence": (
            "the bundle declares an amplitude and a referenced optical path, so "
            "RayBundle.require_coherent() accepts it. Sequential rays are not themselves a "
            "coherent complex field; the declaration is what makes one derivable."
        ),
        "exit_pupil": (
            None
            if pupil is None
            else {
                "source": "optic.paraxial.XPL() and XPD(), read not constructed",
                "z_m": pupil["z_mm"] * NATIVE_LENGTH_M,
                "location_from_image_m": pupil["location_from_image_mm"] * NATIVE_LENGTH_M,
                "diameter_m": pupil["diameter_mm"] * NATIVE_LENGTH_M,
                "is_virtual": pupil["is_virtual"],
                "position_semantics": (
                    "each ray's image-space ASYMPTOTE at the pupil plane, not a physical "
                    "intersection; the pupil is virtual on both fixture systems"
                ),
            }
        ),
    }
    return bundle, diagnostics


def _declare_measure(
    lens: Any, *, num_rings: int, traced_count: int, wavelength_um: float
) -> tuple[np.ndarray[Any, Any] | None, MeasureKind, str]:
    """The per-ray pupil area element and its kind, or an honest absence.

    A missing measure is not itself a physics error the way a missing off-axis
    tilt term is: the bundle is still coherent, it is simply unweighted. So this
    returns `(None, "undeclared", reason)` rather than raising, and R07's coupler
    is the thing that refuses to reconstruct from an undeclared measure.
    """
    expected = hexapolar_ray_count(num_rings)
    if traced_count != expected:
        return (
            None,
            "undeclared",
            f"undeclared: the trace exported {traced_count} rays but an un-vignetted "
            f"{num_rings}-ring hexapolar fan is {expected}, so a ring index cannot be "
            "assigned row for row and any area element would be invented",
        )
    try:
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rings)
        # float64 by declaration, independent of the trace: the ring assignment is
        # a tolerance test on the ratio rho * num_rings, so it is computed at
        # reference precision whatever the trace ran in.
        pupil_x = np.asarray(_host(distribution.x), dtype=np.float64)
        pupil_y = np.asarray(_host(distribution.y), dtype=np.float64)
    except Exception as exc:
        return (
            None,
            "undeclared",
            f"undeclared: the hexapolar pupil sampling could not be regenerated "
            f"({type(exc).__name__}: {exc})",
        )
    if pupil_x.size != traced_count:  # pragma: no cover - equal to `expected` by construction
        return (
            None,
            "undeclared",
            f"undeclared: the regenerated fan has {pupil_x.size} points against "
            f"{traced_count} traced rays",
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
        ring_index = hexapolar_ring_index(pupil_x, pupil_y, num_rings)
    except ContractError as exc:
        return (None, "undeclared", f"undeclared: {exc.code}: {exc}")
    weight = hexapolar_area_weight_m2(ring_index, num_rings, aperture_radius_m)
    return (
        weight,
        "quadrature_area_m2",
        (
            "quadrature_area_m2: the absolute per-ray entrance-pupil area element a "
            f"{num_rings}-ring hexapolar fan represents, radial-trapezoid corrected at the "
            "centre (3/4) and rim (1/2). Aperture radius "
            f"{aperture_radius_m!r} m, sum {float(weight.sum()):.9e} m^2 against "
            f"pi a^2 = {math.pi * aperture_radius_m**2:.9e} m^2 "
            f"(ratio {float(weight.sum()) / (math.pi * aperture_radius_m**2):.12f}, exactly "
            f"1 + 1/(4 n^2)). Wavelength {wavelength_um!r} um does not enter: the measure is "
            "a property of how the pupil was sampled, not of the light."
        ),
    )


def require_declared_optical_path(bundle: RayBundle) -> None:
    """Refuse a bundle whose optical path is the native accumulator in disguise.

    Two jobs, and both are the same check:

    * a **post-condition** on `to_ray_bundle`, which calls it on every bundle it
      emits. Not redundant with the call it follows: it is what makes "the emitted
      path came from `declare_optical_path_m`" a checked property of the artifact
      rather than a property of the current control flow.
    * a **callable refusal** for any consumer holding a bundle that claims to have
      come from this solver. Scaling the native accumulator into metres yields a
      quantity with plausible magnitude, plausible units and a plausible-looking
      spread. What it does not have is a declared reference -- and a wrong
      reference is a harmless piston while a wrong *sign* conjugates the
      wavefront, so a converging beam reconstructs as a diverging one and no
      intensity check can tell the difference.

    So the only admissible optical path is one whose reference carries
    `OPL_REFERENCE_VERSION`, which only `declare_optical_path_m` writes. A bundle
    with no optical path at all passes: carrying no phase is honest, and
    `RayBundle.require_coherent()` is what refuses to read one that is not there.
    """
    if bundle.optical_path_m is None:
        return
    reference = bundle.optical_path_reference or ""
    if not reference.startswith(OPL_REFERENCE_VERSION):
        raise ContractError(
            "OPL_REFERENCE_UNVERIFIED",
            (
                f"the optical path is declared {reference!r}, which is not a reference this "
                f"solver produced. Optiland's `opd` is a native accumulator, not a declared "
                f"physical quantity: it is absolute, its zero moves with the aperture, and "
                f"its orientation is a plane rather than a wavefront. Passing it through as "
                f"an optical path length is refused."
            ),
            declaration="optical_path_reference",
            remedy=(
                f"Obtain the path from declare_optical_path_m, whose reference starts with "
                f"{OPL_REFERENCE_VERSION!r}, or carry it with the reference declared "
                f"{UNVERIFIED!r} so require_coherent() refuses to read it as a phase."
            ),
        )
