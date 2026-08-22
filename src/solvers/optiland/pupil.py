"""Pupil and image-space geometry: where a plane is, and what a reading means.

Five resolvers that answer questions about the *system* rather than about the
adapter -- where the exit pupil sits, what the image space looks like, where a
bundle lands when projected to a plane, what the object-space reference offset
is, and how the pupil was sampled.

They were module-level private helpers inside the adapter, which made them read
as plumbing. They are optical geometry, each carrying a convention worth stating:
``Paraxial.XPL()`` is signed and measured from the image surface rather than from
the global origin, and getting that wrong yields a plausible plane instead of an
error.

``HandoffPlaneError`` belongs here because an unresolvable plane is a geometry
outcome. Carrying it as an exception rather than a sentinel is what stops a
caller mistaking an unresolved plane for one at z = 0.
"""

from __future__ import annotations

from typing import Any

from solvers.optiland.constants import (
    _DIRECTION_NORM_TOLERANCE,
    _GEOMETRY_M_PER_MM,
)
from solvers.optiland.execution import _host_array


class HandoffPlaneError(RuntimeError):
    """The requested handoff plane could not be resolved from the system.

    Carried as an exception rather than a sentinel so the caller cannot mistake
    an unresolved plane for one at z = 0. `run()` converts it to a structured
    failure; it is never allowed to reach the export.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code




def _resolve_exit_pupil(lens: Any, be_utils: Any, image_plane_z_mm: float) -> dict[str, Any]:
    """Read the exit pupil from the system, and say what the reading means.

    `Paraxial.XPL()` is signed and measured **from the image surface**, not from
    the global origin, so the plane is `image_z + XPL`
    (tmp_probes/optiland_exit_pupil_probe.py).

    The pupil is frequently *virtual* -- on `ReverseTelephoto` it lands at
    z = 2.15 mm with five refracting surfaces beyond it. That does not make the
    plane wrong, but it does change what a position at that plane is, and the
    returned metadata says so rather than leaving a reader to assume.
    """
    import numpy as np

    try:
        location_from_image_mm = float(
            np.asarray(be_utils.to_numpy(lens.paraxial.XPL())).ravel()[0]
        )
        diameter_mm = float(np.asarray(be_utils.to_numpy(lens.paraxial.XPD())).ravel()[0])
    except Exception as exc:
        raise HandoffPlaneError(
            "OPTILAND_EXIT_PUPIL_UNRESOLVED",
            "config['handoff_plane']='exit_pupil' was requested but Optiland's "
            f"paraxial solver could not supply XPL()/XPD(): {type(exc).__name__}: {exc}. "
            "The plane is read from the system, never guessed, so the run fails here "
            "rather than exporting rays against an invented reference.",
        ) from exc

    if not (np.isfinite(location_from_image_mm) and np.isfinite(diameter_mm)):
        raise HandoffPlaneError(
            "OPTILAND_EXIT_PUPIL_UNRESOLVED",
            "Optiland returned a non-finite exit pupil for this system "
            f"(XPL={location_from_image_mm!r}, XPD={diameter_mm!r}); a telecentric or "
            "degenerate configuration has no finite exit pupil plane, and this "
            "adapter will not substitute one.",
        )

    pupil_z_mm = image_plane_z_mm + location_from_image_mm
    surface_z = [
        float(np.asarray(be_utils.to_numpy(surface.geometry.cs.z)).ravel()[0])
        for surface in lens.surfaces.surfaces[:-1]
    ]
    beyond = [z for z in surface_z if np.isfinite(z) and z > pupil_z_mm]

    return {
        "z_mm": pupil_z_mm,
        "location_from_image_mm": location_from_image_mm,
        "diameter_mm": diameter_mm,
        "is_virtual": bool(beyond),
        "refracting_surfaces_beyond_pupil_z_mm": beyond,
    }


def _resolve_image_space(lens: Any, be_utils: Any, wavelength_um: float) -> dict[str, Any]:
    """Read the three facts a downstream OPL declaration must not assume.

    CHE-33 needs the image-space index to move an optical path between the traced
    image surface and any other plane in image space, and needs the entrance pupil
    diameter because CHE-30 showed the OPL zero of an infinite-object system moves
    with the aperture.

    Every value is read from the prescription and is ``None`` when the installed
    package does not expose it. It is not defaulted: ``n = 1`` happens to hold for
    both M3 systems, and a silent 1.0 for a system with a cover glass or an
    immersion medium would be a wrong optical path with nothing to notice it by.
    """
    import numpy as np

    def _scalar(thunk: Any) -> float | None:
        # Everything, including the attribute lookup, happens inside the guard:
        # this runs against fake lenses in the adapter's own failure tests, and a
        # missing attribute must degrade to "not available" rather than turn an
        # unrelated structured failure into a crash.
        try:
            value = float(np.asarray(be_utils.to_numpy(thunk())).ravel()[0])
        except Exception:
            return None
        return value if np.isfinite(value) else None

    index = _scalar(lambda: lens.surfaces.surfaces[-1].material_pre.n(wavelength_um))
    entrance_pupil_diameter_mm = _scalar(lambda: lens.paraxial.EPD())

    try:
        object_at_infinity: bool | None = bool(lens.object_surface.is_infinite)
    except Exception:
        object_at_infinity = None

    return {
        "image_space_refractive_index": index,
        "entrance_pupil_diameter_m": (
            entrance_pupil_diameter_mm * _GEOMETRY_M_PER_MM
            if entrance_pupil_diameter_mm is not None
            else None
        ),
        "object_at_infinity": object_at_infinity,
    }


def _project_rays_to_plane(rays: Any, be_utils: Any, target_z_mm: float) -> dict[str, Any]:
    """Advance each ray along its own image-space direction to `target_z_mm`.

    Returns the ray's image-space **asymptote** at that plane, which is what the
    exit pupil is defined by, not a physical intersection: for a virtual pupil the
    line being extended passes back through glass the ray never travelled in that
    state. See the probe for why that is nonetheless the right construction.

    Directions are unchanged -- this is a reparameterization along each ray, not a
    propagation, so no OPL is added or removed here. OPL is M3.4's.
    """
    import numpy as np

    # Precision preserved: this projection feeds the EXPORTED exit-pupil
    # positions, so widening here would make the exit_pupil handoff plane report
    # float64 for a float32 trace while image_surface reported float32.
    x = _host_array(be_utils, rays.x)
    y = _host_array(be_utils, rays.y)
    z = _host_array(be_utils, rays.z)
    direction_z = _host_array(be_utils, rays.N)

    if np.any(direction_z == 0.0):
        raise HandoffPlaneError(
            "OPTILAND_HANDOFF_PLANE_UNREACHABLE",
            "at least one traced ray has N = 0 and therefore never reaches the "
            "requested handoff plane; the projection is undefined for it.",
        )

    step_mm = (target_z_mm - z) / direction_z
    return {
        "x_mm": x + _host_array(be_utils, rays.L) * step_mm,
        "y_mm": y + _host_array(be_utils, rays.M) * step_mm,
        "z_mm": np.full_like(x, target_z_mm),
        "max_abs_step_mm": float(np.max(np.abs(step_mm))),
    }


def _resolve_object_space_reference(
    lens: Any,
    be: Any,
    be_utils: Any,
    *,
    hx: float,
    hy: float,
    wavelength_um: float,
    num_rays: int,
    traced_count: int,
) -> dict[str, Any]:
    """The optical path from an incoming *wavefront* to each ray's launch point.

    CHE-41. ``RealRays.opd`` is seeded to zero at the launch state, and for an
    object at infinity ``angle.py`` launches every ray on one plane
    **perpendicular to z** at ``positions[1] - (EPD - min(positions[1:-1]))``.
    A plane perpendicular to z is a wavefront only for a bundle travelling along
    z. For a bundle tilted by ``theta`` the two surfaces differ by
    ``n_object * (d0 . r_launch)``, which is *linear in the launch coordinate* --
    a tilt, not a piston. CHE-30 characterized the same launch plane and recorded
    only the piston consequence, because on axis that is all there is.

    Nothing is corrected here. The term is measured from the launch state and
    exported so that a *consumer* can declare its reference; the accumulated
    ``opd_native`` is left exactly as Optiland produced it.

    The launch state is regenerated through the public entry point
    ``ray_tracer.ray_generator.generate_rays`` over the same hexapolar
    distribution ``Optic.trace`` builds, which is the only reason this is
    possible at all: ``Optic.trace`` returns the traced rays and keeps no record
    of where they started.

    Every precondition the term depends on is *checked rather than assumed*, and
    a failed check returns ``available=False`` with the reason. It never returns
    a term it could not verify: an unavailable term is a structured refusal
    downstream, and a wrong one is a wavefront aimed at the wrong image point.
    """
    import numpy as np

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "unavailable_reason": reason,
            "offset_native": None,
        }

    try:
        object_at_infinity = bool(lens.object_surface.is_infinite)
    except Exception as exc:  # pragma: no cover - defensive
        return unavailable(
            f"the object surface could not be read ({type(exc).__name__}), so the "
            "launch geometry is unknown"
        )
    if not object_at_infinity:
        return unavailable(
            "the object is at a finite distance, so the launch state is a POINT "
            "rather than a plane. A point source is already a common wavefront and "
            "the term would be zero -- but no system in this repository exercises "
            "that path, and an untested zero is still an untested claim."
        )

    try:
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rays)
        pupil_x = distribution.x
        pupil_y = distribution.y
        pupil_points = int(np.asarray(be_utils.to_numpy(pupil_x)).size)
        field_x = be.atleast_1d(be.array(float(hx)))
        field_y = be.atleast_1d(be.array(float(hy)))
        launch = lens.ray_tracer.ray_generator.generate_rays(
            be.repeat(field_x, pupil_points),
            be.repeat(field_y, pupil_points),
            pupil_x,
            pupil_y,
            wavelength_um,
        )
    except Exception as exc:
        return unavailable(
            "the launch state could not be regenerated from "
            f"ray_tracer.ray_generator.generate_rays ({type(exc).__name__}: {exc}); "
            "Optic.trace does not retain it, so there is nothing to measure the "
            "object-space reference from"
        )

    def to_array(value: Any) -> Any:
        # Deliberately float64, and NOT the trace's precision: this is a
        # regenerated launch state used to compute the object-space OPL
        # reference, which is a piston-and-tilt correction of order 1e4 waves.
        # Computing that reference in float32 would inject an error larger than
        # the wavefront it corrects. Declared here rather than inherited.
        return np.asarray(be_utils.to_numpy(value), dtype=np.float64)

    x0, y0, z0 = to_array(launch.x), to_array(launch.y), to_array(launch.z)
    l0, m0, n0 = to_array(launch.L), to_array(launch.M), to_array(launch.N)

    if x0.size != traced_count:
        return unavailable(
            f"the regenerated launch state has {x0.size} rays but the trace exported "
            f"{traced_count}; the two cannot be matched row for row"
        )
    if not (
        np.all(np.isfinite(x0))
        and np.all(np.isfinite(y0))
        and np.all(np.isfinite(z0))
        and np.all(np.isfinite(l0))
        and np.all(np.isfinite(m0))
        and np.all(np.isfinite(n0))
    ):
        return unavailable("the regenerated launch state is not finite")

    direction_spread = max(float(np.ptp(l0)), float(np.ptp(m0)), float(np.ptp(n0)))
    plane_spread = float(np.ptp(z0))
    if direction_spread > _DIRECTION_NORM_TOLERANCE:
        return unavailable(
            "the launch directions are not common to every ray (spread "
            f"{direction_spread:.3e}), so the incoming bundle is not collimated and "
            "a single plane wavefront does not describe it"
        )
    if plane_spread > 0.0:
        return unavailable(
            f"the launch points do not lie on one plane (z spread {plane_spread:.3e} "
            "in native units), so the seeded reference surface is not the plane this "
            "term assumes"
        )

    index = None
    try:
        index = float(
            np.asarray(
                be_utils.to_numpy(lens.surfaces.surfaces[0].material_post.n(wavelength_um))
            ).ravel()[0]
        )
    except Exception:
        index = None
    if index is None or not np.isfinite(index) or index <= 0.0:
        return unavailable(
            "the object-space refractive index could not be read from the "
            "prescription, and the optical path from a wavefront to the launch "
            "plane is index-weighted"
        )

    # d0 . r_launch, index-weighted. The N0 * z0 part is common to every ray
    # because the launch plane is flat; it is retained rather than dropped so the
    # exported quantity is the optical path from ONE stated wavefront (the one
    # through the global origin, perpendicular to d0) rather than from an
    # unstated one.
    offset_native = index * (l0 * x0 + m0 * y0 + n0 * z0)

    return {
        "available": True,
        "unavailable_reason": None,
        "offset_native": offset_native,
        "launch_x_native": x0,
        "launch_y_native": y0,
        "launch_z_native": z0,
        "launch_direction": [float(l0[0]), float(m0[0]), float(n0[0])],
        "launch_plane_z_native": float(z0[0]),
        "object_space_refractive_index": index,
        "span_native": float(np.ptp(offset_native)),
    }


def _resolve_ray_pupil_sampling(
    lens: Any,
    be_utils: Any,
    *,
    num_rays: int,
    traced_count: int,
) -> dict[str, Any]:
    """The raw hexapolar pupil coordinates CHE-47's quadrature weight needs.

    CHE-38 found that the wavelet sum's dominant sensor-plane residual is a
    per-ray quadrature-weight error, not a kernel defect (section 14/15), and
    CHE-47 is the ticket that supplies the weight. Computing it needs to know
    which pupil ring each traced ray came from, and ``Optic.trace`` keeps no
    record of that -- so the same hexapolar distribution is regenerated from
    ``optiland.distribution.create_distribution`` and matched row for row
    against the trace, exactly as :func:`_resolve_object_space_reference`
    regenerates the launch state above.

    This function returns only the RAW normalized pupil coordinates, the ring
    count, and the aperture radius -- not a ring index or a weight. The actual
    quadrature math (`couplers.quadrature`) is coupler
    physics, not adapter physics, and this module must import no coupler: the
    M1 independence check (`benchmarks/level1/L1-RAY-01`) asserts that tracing
    a ray bundle loads no `couplers.*` module, and an
    import here would violate that for every caller of this adapter, not only
    CHE-47's. `optiland_handoff.py` (already coupler-side) computes the ring
    index and area weight from what this function returns.

    Every precondition is checked rather than assumed, exactly as CHE-41's
    object-space term is: a row-count mismatch (a vignetted ray, so the
    regenerated pupil no longer lines up one-to-one with the traced set) or an
    unreadable aperture diameter returns ``available=False`` with the reason,
    never a fabricated value.
    """
    import numpy as np

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "unavailable_reason": reason,
            "pupil_x": None,
            "pupil_y": None,
        }

    try:
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rays)
        # Also a float64 reference by declaration: these are regenerated
        # normalized pupil coordinates used to assign hexapolar RING INDICES by
        # comparing r against j / num_rings (CHE-47). That comparison is a
        # tolerance test on a ratio, so it is computed at reference precision
        # independently of what the trace ran in.
        pupil_x = np.asarray(be_utils.to_numpy(distribution.x), dtype=np.float64)
        pupil_y = np.asarray(be_utils.to_numpy(distribution.y), dtype=np.float64)
    except Exception as exc:
        return unavailable(
            "the hexapolar pupil sampling could not be regenerated from "
            f"optiland.distribution.create_distribution ({type(exc).__name__}: {exc})"
        )

    if pupil_x.size != traced_count:
        return unavailable(
            f"the regenerated pupil sampling has {pupil_x.size} points but the trace "
            f"exported {traced_count}; at least one ray was vignetted (or num_rays did "
            "not request a hexapolar fan), so a ring index cannot be assigned row for row"
        )

    try:
        epd_mm = float(np.asarray(be_utils.to_numpy(lens.paraxial.EPD())).ravel()[0])
    except Exception as exc:
        return unavailable(
            f"the entrance pupil diameter could not be read ({type(exc).__name__}: {exc}), "
            "so the physical aperture area a quadrature weight scales to is unknown"
        )
    if not np.isfinite(epd_mm) or epd_mm <= 0.0:
        return unavailable(f"entrance pupil diameter is not a positive finite value ({epd_mm!r})")
    aperture_radius_m = (epd_mm / 2.0) * _GEOMETRY_M_PER_MM

    return {
        "available": True,
        "unavailable_reason": None,
        "pupil_x": pupil_x,
        "pupil_y": pupil_y,
        "num_rings": num_rays,
        "aperture_radius_m": aperture_radius_m,
    }
