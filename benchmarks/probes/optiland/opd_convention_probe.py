"""Characterize `optiland.rays.real_rays.RealRays.opd` (CHE-30 / M3.1).

M1 left `opd_reference` and `opd_sign` recorded as `unverified`, so the
coupler contract layer refuses a real Optiland trace as an optical path
length. This probe establishes the convention against manufactured
geometries whose answers are known in closed form, so that the refusal can
be replaced by a declaration instead of a guess.

Every case states a prediction *before* reading the value, and every case
also states what a competing (wrong) hypothesis would have predicted, so a
passing case is falsifiable rather than merely consistent.

The accumulation site in the pinned install is
`optiland/surfaces/standard_surface.py`:

    rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))

which says `opd` is an *absolute accumulated optical path length* in lens
geometry units, index-weighted by the medium *preceding* each surface, and
non-decreasing because of the `be.abs`. The cases below test that reading.

Run inside the agent_solver container:
    ./run.sh python benchmarks/probes/optiland/opd_convention_probe.py
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays.real_rays import RealRays

WAVELENGTH_UM = 0.55


def _rays(x, y, z, L, M, N):
    """Build a RealRays bundle from broadcastable position/direction arrays."""
    x, y, z, L, M, N = (np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (x, y, z, L, M, N))
    count = x.size
    return RealRays(
        x.copy(),
        y.copy(),
        z.copy(),
        L.copy(),
        M.copy(),
        N.copy(),
        np.ones(count, dtype=np.float64),
        np.full(count, WAVELENGTH_UM, dtype=np.float64),
    )


def case_initial_opd_is_zero() -> dict:
    """The accumulator starts at zero, so `opd` is measured from the launch state."""
    rays = _rays(0.5, -0.25, 0.0, 0.0, 0.0, 1.0)
    return {
        "claim": "RealRays.opd == 0 at construction",
        "observed_opd": float(np.max(np.abs(np.asarray(rays.opd)))),
        "predicted_opd": 0.0,
        "meaning": (
            "opd is an accumulator seeded at the launch state, not a quantity "
            "referenced to a chief ray or to a system plane"
        ),
    }


def case_free_space_oblique(distance_mm: float = 10.0) -> dict:
    """Oblique free-space path: opd must be the slant distance, not the axial one."""
    L = np.array([-0.03, 0.0, 0.02], dtype=np.float64)
    M = np.array([0.01, -0.025, 0.015], dtype=np.float64)
    N = np.sqrt(1.0 - L**2 - M**2)
    x0 = np.array([-2.0, 0.25, 1.5], dtype=np.float64)
    y0 = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    rays = _rays(x0, y0, np.zeros(3), L, M, N)

    optic = Optic("opd-probe-free-space")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(index=1, radius=be.inf, z=distance_mm)
    optic.surfaces[1].trace(rays)

    observed = np.asarray(rays.opd, dtype=np.float64)
    predicted_slant = distance_mm / N
    return {
        "claim": "opd == slant distance travelled (absolute, n=1)",
        "distance_mm": distance_mm,
        "max_abs_error_slant_mm": float(np.max(np.abs(observed - predicted_slant))),
        "max_abs_error_if_axial_hypothesis_mm": float(np.max(np.abs(observed - distance_mm))),
        "observed_opd_mm": observed.tolist(),
        "predicted_slant_mm": predicted_slant.tolist(),
        "falsifier": (
            "an implementation accumulating the axial separation instead of the "
            "slant path would match 'axial_hypothesis' and miss 'slant'"
        ),
    }


def case_index_weighting(t_air_mm: float = 4.0, t_glass_mm: float = 6.0, n: float = 1.7) -> dict:
    """A glass slab must contribute n*t, which separates optical from geometric path."""
    rays = _rays(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    optic = Optic("opd-probe-index")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        radius=be.inf,
        z=t_air_mm,
        material=IdealMaterial(n=n),
    )
    optic.surfaces.add(index=2, radius=be.inf, z=t_air_mm + t_glass_mm)
    optic.surfaces[1].trace(rays)
    opd_after_first = float(np.asarray(rays.opd)[0])
    optic.surfaces[2].trace(rays)
    opd_after_second = float(np.asarray(rays.opd)[0])

    predicted_optical = t_air_mm + n * t_glass_mm
    predicted_geometric = t_air_mm + t_glass_mm
    return {
        "claim": "opd is index-weighted (optical) path, using the medium BEFORE each surface",
        "t_air_mm": t_air_mm,
        "t_glass_mm": t_glass_mm,
        "n": n,
        "opd_after_first_surface_mm": opd_after_first,
        "opd_after_second_surface_mm": opd_after_second,
        "predicted_optical_mm": predicted_optical,
        "predicted_geometric_mm": predicted_geometric,
        "abs_error_optical_mm": abs(opd_after_second - predicted_optical),
        "abs_error_if_geometric_hypothesis_mm": abs(opd_after_second - predicted_geometric),
        "falsifier": (
            "a geometric-path accumulator would match 'geometric' and be short by "
            f"{(n - 1.0) * t_glass_mm:.6g} mm"
        ),
    }


def case_trace_launch_plane(separation_mm: float = 10.0) -> dict:
    """Explain M1's `opd = 12` for a 10 mm separation, and pin the reference plane.

    For an object at infinity the pinned install computes the launch plane in
    `optiland/fields/field_types/angle.py`:

        offset = optic.paraxial.EPD() - min(positions[1:-1])
        z0 = positions[1] - offset

    so the OPL zero sits `EPD` in front of the first surface when that surface
    is at z=0. The reference plane therefore *moves when the aperture
    changes*, which is the fact that makes an undeclared OPL dangerous.
    """
    results = []
    for epd_mm in (2.0, 4.0, 7.5):
        optic = Optic("opd-probe-launch-plane")
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(index=1, radius=be.inf, thickness=separation_mm, is_stop=True)
        optic.surfaces.add(index=2)
        optic.set_aperture(aperture_type="EPD", value=epd_mm)
        optic.fields.set_type(field_type="angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)

        rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
        opd = np.asarray(rays.opd, dtype=np.float64)
        positions = np.asarray(be.to_numpy(optic.surfaces.positions), dtype=np.float64).ravel()
        predicted_offset = epd_mm - float(np.min(positions[1:-1]))
        predicted_opd = predicted_offset + separation_mm
        results.append(
            {
                "EPD_mm": epd_mm,
                "predicted_launch_z_mm": float(positions[1]) - predicted_offset,
                "predicted_opd_mm": predicted_opd,
                "ray_count": int(opd.size),
                "observed_opd_min_mm": float(np.min(opd)),
                "observed_opd_max_mm": float(np.max(opd)),
                "max_abs_error_mm": float(np.max(np.abs(opd - predicted_opd))),
                "abs_error_if_reference_were_first_surface_mm": float(
                    np.max(np.abs(opd - separation_mm))
                ),
            }
        )
    return {
        "claim": (
            "with an object at infinity the OPL zero is the aimed launch plane at "
            "z = positions[1] - (EPD - min(positions[1:-1])), NOT the first surface"
        ),
        "separation_mm": separation_mm,
        "sweeps": results,
        "explains_m1_anomaly": (
            "M1 observed opd=12 for a 10 mm separation at EPD=2.0: the missing 2 mm "
            "is the aperture-dependent launch offset, so opd was correct and the "
            "reference plane was simply unknown"
        ),
        "falsifier": (
            "if the reference were the first surface, every observed opd would equal "
            "the separation and 'abs_error_if_reference_were_first_surface_mm' would "
            "be zero; instead it equals EPD and grows with EPD"
        ),
    }


def case_finite_object(object_distance_mm: float = 50.0, separation_mm: float = 10.0) -> dict:
    """With a finite object the launch plane is the object surface itself.

    Predicted per ray as the straight-line distance from the on-axis object
    point to that ray's final position: the surfaces here are planar with air
    on both sides, so no ray bends and the whole path is one segment. That
    makes the marginal rays part of the oracle instead of unexplained spread.
    """
    optic = Optic("opd-probe-finite-object")
    optic.surfaces.add(index=0, radius=be.inf, thickness=object_distance_mm)
    optic.surfaces.add(index=1, radius=be.inf, thickness=separation_mm, is_stop=True)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)

    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
    opd = np.asarray(rays.opd, dtype=np.float64)
    x = np.asarray(rays.x, dtype=np.float64)
    y = np.asarray(rays.y, dtype=np.float64)
    axial_span = object_distance_mm + separation_mm
    predicted = np.sqrt(axial_span**2 + x**2 + y**2)
    return {
        "claim": "with a finite object the OPL zero is the object plane; no EPD offset applies",
        "object_distance_mm": object_distance_mm,
        "separation_mm": separation_mm,
        "ray_count": int(opd.size),
        "predicted_axial_opd_mm": axial_span,
        "observed_axial_opd_mm": float(np.min(opd)),
        "observed_opd_max_mm": float(np.max(opd)),
        "max_abs_error_vs_per_ray_slant_mm": float(np.max(np.abs(opd - predicted))),
        "abs_error_if_epd_offset_were_applied_mm": float(
            np.abs(float(np.min(opd)) - (axial_span + 2.0))
        ),
    }


def case_geometry_scale(distance_mm: float = 10.0, factor: float = 10.0) -> dict:
    """opd carries the geometry length unit: scaling the system scales opd exactly."""
    small = case_free_space_oblique(distance_mm)
    large = case_free_space_oblique(distance_mm * factor)
    small_opd = np.asarray(small["observed_opd_mm"], dtype=np.float64)
    large_opd = np.asarray(large["observed_opd_mm"], dtype=np.float64)
    ratio = large_opd / small_opd
    return {
        "claim": "opd is expressed in the lens geometry unit (mm for this project's prescriptions)",
        "factor": factor,
        "observed_ratio": ratio.tolist(),
        "max_abs_ratio_error": float(np.max(np.abs(ratio - factor))),
        "wavelength_unit_evidence": (
            "optiland/wavefront/strategy.py converts with `(wavelength * 1e-3)` when "
            "dividing an opd difference to obtain waves, i.e. um -> mm, so opd shares "
            "the mm geometry scale"
        ),
    }


def case_wavefront_sign_convention() -> dict:
    """Record Optiland's OWN wavefront sign, which is the reverse of L1-RAY-01's."""
    from optiland.wavefront import strategy

    source = inspect.getsource(strategy)
    return {
        "claim": (
            "Optiland's internal wavefront convention is chief-minus-ray; "
            "L1-RAY-01 declared ray-minus-chief"
        ),
        "optiland_expression_present": "opd_wv = (opd_ref - opd) / (wavelength * 1e-3)" in source,
        "optiland_reference_subtraction_present": "opd_ref = self._chief_ray.opd - opd_img"
        in source,
        "consequence": (
            "the two differ by an overall sign. A consumer that mixes Optiland's "
            "wavefront output with L1-RAY-01's opd_ray_minus_chief convention "
            "conjugates the wavefront"
        ),
    }


def case_paraxial_surface_breaks_opl(focal_mm: float = 50.0) -> dict:
    """Ideal-paraxial surfaces are NOT a valid OPL source, even though they image well.

    Physics: a plane wave through a perfect lens must reach the focus with the
    *same* optical path for every pupil height. The paraxial interaction model
    subtracts `(x^2 + y^2) / (2f)`, which is exactly the paraxial excess of
    `sqrt(f^2 + h^2)`, so the total would be flat -- but only if the following
    propagation adds the true Euclidean distance. The paraxial model sets
    `rays.N = copysign(1, N)` and leaves the direction un-normalized, so the
    propagation parameter is the axial distance instead.

    This case measures the residual OPL spread across the pupil, which decides
    whether M3.3's diffraction-limited system may use `surface_type="paraxial"`.
    """
    heights = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    rays = _rays(
        heights,
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.ones_like(heights),
    )

    optic = Optic("opd-probe-paraxial-thin-lens")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=focal_mm,
        thickness=focal_mm,
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=0.0)
    optic.surfaces.trace(rays, skip=1)

    opd = np.asarray(rays.opd, dtype=np.float64)
    relative = opd - opd[0]
    # A perfect lens gives a flat OPL to focus. The two candidate defects:
    euclidean_excess = np.sqrt(focal_mm**2 + heights**2) - focal_mm
    return {
        "claim": "OPL through an ideal paraxial surface is NOT flat at the focus",
        "focal_mm": focal_mm,
        "pupil_heights_mm": heights.tolist(),
        "observed_opd_mm": opd.tolist(),
        "observed_opd_minus_axial_mm": relative.tolist(),
        "perfect_lens_prediction_mm": np.zeros_like(heights).tolist(),
        "max_abs_departure_from_flat_mm": float(np.max(np.abs(relative))),
        "paraxial_lens_term_mm": (-(heights**2) / (2.0 * focal_mm)).tolist(),
        "euclidean_excess_mm": euclidean_excess.tolist(),
        "consequence_if_not_flat": (
            "the reconstructed wavefront would carry a spurious quadratic term, "
            "i.e. a defocus, so a diffraction-limited reference system must be "
            "built from real refractive surfaces rather than surface_type='paraxial'"
        ),
    }


def case_real_lens_opl_is_flat_at_focus() -> dict:
    """Control for the previous case: a real refractive singlet at its own focus.

    Uses a single spherical surface with a flat back, focused by Optiland's own
    paraxial solve, and reports the OPL spread across the pupil. A real surface
    normalizes directions, so the spread here should be spherical aberration
    rather than the paraxial model's artifact.
    """
    n = 1.5168
    radius_mm = 25.0
    thickness_mm = 2.0
    efl_mm = radius_mm / (n - 1.0)
    bfl_mm = efl_mm - thickness_mm / n

    heights = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    rays = _rays(
        heights,
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.zeros_like(heights),
        np.ones_like(heights),
    )

    optic = Optic("opd-probe-real-singlet")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        radius=radius_mm,
        thickness=thickness_mm,
        material=IdealMaterial(n=n),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=bfl_mm)
    optic.surfaces.add(index=3, radius=be.inf, thickness=0.0)
    optic.surfaces.trace(rays, skip=1)

    opd = np.asarray(rays.opd, dtype=np.float64)
    relative = opd - opd[0]
    return {
        "claim": "a real refractive singlet accumulates OPL with directions normalized",
        "n": n,
        "radius_mm": radius_mm,
        "thickness_mm": thickness_mm,
        "analytic_efl_mm": efl_mm,
        "analytic_bfl_mm": bfl_mm,
        "pupil_heights_mm": heights.tolist(),
        "observed_opd_mm": opd.tolist(),
        "observed_opd_minus_axial_mm": relative.tolist(),
        "note": (
            "residual spread is physical spherical aberration of a plano-convex "
            "singlet used at its paraxial focus, and grows as h^4; it is not the "
            "paraxial-surface artifact measured in the previous case"
        ),
    }


def main() -> None:
    report = {
        "probe": "opd_convention_probe",
        "issue": "CHE-30 (M3.1)",
        "accumulation_site": (
            "optiland/surfaces/standard_surface.py: "
            "rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))"
        ),
        "wavelength_um": WAVELENGTH_UM,
        "backend": "numpy (default; not switched)",
        "cases": {
            "initial_opd_is_zero": case_initial_opd_is_zero(),
            "free_space_oblique": case_free_space_oblique(),
            "index_weighting": case_index_weighting(),
            "trace_launch_plane": case_trace_launch_plane(),
            "finite_object": case_finite_object(),
            "geometry_scale": case_geometry_scale(),
            "wavefront_sign_convention": case_wavefront_sign_convention(),
            "paraxial_surface_breaks_opl": case_paraxial_surface_breaks_opl(),
            "real_lens_opl_at_focus": case_real_lens_opl_is_flat_at_focus(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
