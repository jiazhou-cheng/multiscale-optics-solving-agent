"""M3.2 feasibility envelope for the ray -> wave -> Chromatix slice (CHE-31).

The slice can be arithmetically infeasible before a line of it is written, so
this probe derives the binding numbers first and reports them as evidence for
the frozen protocol. It answers four questions:

1. **How large an aperture may the diffraction-limited reference system use?**
   Wavefront error is computed as optical path to a reference sphere centred on
   the nominal focus -- not to the image plane, which would confuse the
   plane/sphere difference with aberration. The aperture is chosen by the
   Rayleigh quarter-wave criterion, not by taste.
2. **What sample pitch does the coupler's per-axis Nyquist limit admit?** The
   limit is `pitch <= lambda / (2 * max|d_axis|)` evaluated over the *marginal*
   rays of a real trace, per axis, because a diagonal bin is exactly
   representable while its norm exceeds the scalar limit.
3. **How many grid points does that force**, given the pupil must still fit?
4. **What does that cost?** `C_RAY_TO_WAVE` contracts rays against pixels
   (`einsum("n,ny,nx->yx")`), so cost grows as rays x pixels. This measures the
   real throughput instead of extrapolating from M2's round-trip figure, which
   included the wave->ray Monte Carlo.

Optiland exposes exit pupil geometry directly as `optic.paraxial.XPL()` /
`XPD()`, so the handoff plane is read from the system rather than constructed.

Run inside the agent_solver container:
    ./run.sh python benchmarks/probes/m3_slice_feasibility.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays.real_rays import RealRays
from optiland.samples.objectives import ReverseTelephoto

WAVELENGTH_UM = 0.55
WAVELENGTH_MM = WAVELENGTH_UM * 1e-3
MM_TO_M = 1e-3

# Plano-convex singlet, convex toward the collimated side (the low-aberration
# orientation). N-BK7-like fixed index keeps the reference system independent of
# a glass catalog. The geometry is scalable because Chromatix's float32 cast
# makes the system's absolute SIZE a constraint, not only its f-number: at fixed
# f-number, scaling down shrinks the pupil-to-focus distance (which bounds the
# propagation error) and the wavefront aberration in waves, while leaving the
# numerical aperture, and therefore the Nyquist pitch, unchanged.
SINGLET_N = 1.5168
SINGLET_RADIUS_MM = 25.0
SINGLET_THICKNESS_MM = 2.0


def _rays_from_heights(heights_mm: np.ndarray) -> RealRays:
    """Collimated on-axis bundle at the given pupil heights (x axis)."""
    heights_mm = np.asarray(heights_mm, dtype=np.float64)
    zeros = np.zeros_like(heights_mm)
    return RealRays(
        heights_mm.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        np.ones_like(heights_mm),
        np.ones_like(heights_mm),
        np.full_like(heights_mm, WAVELENGTH_UM),
    )


def _build_singlet(*, semi_aperture_mm: float, scale: float = 1.0) -> tuple[Any, float, float]:
    """Plano-convex singlet, fully specified so `paraxial.XPL()`/`XPD()` resolve.

    The aperture/field/wavelength declarations are required by the paraxial
    solvers. They do not affect the manufactured-ray traces below, which call
    `surfaces.trace(rays, skip=1)` and therefore seed the OPL accumulator at
    the front vertex rather than at an aimed launch plane (CHE-30).
    """
    radius_mm = SINGLET_RADIUS_MM * scale
    thickness_mm = SINGLET_THICKNESS_MM * scale
    efl_mm = radius_mm / (SINGLET_N - 1.0)
    bfl_mm = efl_mm - thickness_mm / SINGLET_N

    optic = Optic("m3-reference-singlet")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=radius_mm,
        thickness=thickness_mm,
        material=IdealMaterial(n=SINGLET_N),
        is_stop=True,
    )
    # Rear vertex: glass -> air.
    optic.surfaces.add(index=2, radius=be.inf, thickness=bfl_mm)
    optic.surfaces.add(index=3, radius=be.inf, thickness=0.0)
    optic.set_aperture(aperture_type="EPD", value=2.0 * semi_aperture_mm)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return optic, efl_mm, bfl_mm


def wavefront_error_waves(
    heights_mm: np.ndarray, *, semi_aperture_mm: float, scale: float = 1.0
) -> dict[str, Any]:
    """Optical path to a sphere centred on the nominal focus, in waves.

    W_i = opd_i(at the rear vertex) + |X_i - F|, then referenced to its mean.
    Using a sphere rather than the image plane is what keeps this a wavefront
    error instead of a mixture of aberration and plane-vs-sphere geometry.
    """
    optic, efl_mm, bfl_mm = _build_singlet(semi_aperture_mm=semi_aperture_mm, scale=scale)
    rays = _rays_from_heights(heights_mm)
    # Stop at the rear vertex (surface 2) rather than the image plane, so the
    # remaining path to focus is the reference sphere's radius.
    optic.surfaces[1].trace(rays)
    optic.surfaces[2].trace(rays)

    x = np.asarray(rays.x, dtype=np.float64)
    y = np.asarray(rays.y, dtype=np.float64)
    z = np.asarray(rays.z, dtype=np.float64)
    opd = np.asarray(rays.opd, dtype=np.float64)

    # Nominal focus: on axis, one back focal length past the rear vertex.
    focus = np.array([0.0, 0.0, SINGLET_THICKNESS_MM * scale + bfl_mm], dtype=np.float64)
    to_focus_mm = np.sqrt((x - focus[0]) ** 2 + (y - focus[1]) ** 2 + (z - focus[2]) ** 2)
    total_opl_mm = opd + to_focus_mm
    w_mm = total_opl_mm - np.mean(total_opl_mm)
    w_waves = w_mm / WAVELENGTH_MM
    return {
        "efl_mm": efl_mm,
        "bfl_mm": bfl_mm,
        "peak_to_valley_waves": float(np.max(w_waves) - np.min(w_waves)),
        "rms_waves": float(np.sqrt(np.mean((w_waves - np.mean(w_waves)) ** 2))),
    }


def case_diffraction_limited_aperture(*, scale: float = 1.0) -> dict[str, Any]:
    """Largest semi-aperture of the singlet that stays inside Rayleigh's quarter wave."""
    sweep = []
    admissible = None
    for f_number in (48.0, 24.0, 16.0, 12.0, 9.7, 8.0, 6.0, 4.0):
        efl_nominal_mm = SINGLET_RADIUS_MM * scale / (SINGLET_N - 1.0)
        semi_aperture_mm = efl_nominal_mm / (2.0 * f_number)
        heights = np.linspace(0.0, semi_aperture_mm, 25)
        metrics = wavefront_error_waves(heights, semi_aperture_mm=semi_aperture_mm, scale=scale)
        efl_mm = metrics["efl_mm"]
        numerical_aperture = semi_aperture_mm / efl_mm
        entry = {
            "semi_aperture_mm": semi_aperture_mm,
            "f_number": efl_mm / (2.0 * semi_aperture_mm),
            "numerical_aperture": numerical_aperture,
            "peak_to_valley_waves": metrics["peak_to_valley_waves"],
            "rms_waves": metrics["rms_waves"],
            "rayleigh_quarter_wave_pass": metrics["peak_to_valley_waves"] <= 0.25,
            "marechal_strehl_estimate": float(np.exp(-((2.0 * np.pi * metrics["rms_waves"]) ** 2))),
            "airy_radius_um": 1.22 * WAVELENGTH_UM / numerical_aperture,
        }
        sweep.append(entry)
        if entry["rayleigh_quarter_wave_pass"]:
            admissible = entry
    return {
        "claim": (
            "the reference system's aperture is set by the quarter-wave criterion, "
            "and spherical aberration of a singlet scales as h^4 so the limit is sharp"
        ),
        "criterion": "peak-to-valley wavefront error <= lambda/4 (Rayleigh)",
        "scale": scale,
        "sweep": sweep,
        "largest_admissible": admissible,
        "note": (
            "wavefront error is optical path to a sphere centred on the nominal "
            "focus; surface_type='paraxial' is excluded as an OPL source by CHE-30"
        ),
    }


def _exit_pupil_report(optic: Any, rays: Any, label: str) -> dict[str, Any]:
    """Marginal direction cosines and pupil extent at the exit pupil plane."""
    x = np.asarray(be.to_numpy(rays.x), dtype=np.float64)
    y = np.asarray(be.to_numpy(rays.y), dtype=np.float64)
    z = np.asarray(be.to_numpy(rays.z), dtype=np.float64)
    L = np.asarray(be.to_numpy(rays.L), dtype=np.float64)
    M = np.asarray(be.to_numpy(rays.M), dtype=np.float64)
    N = np.asarray(be.to_numpy(rays.N), dtype=np.float64)

    xpl_mm = float(np.asarray(be.to_numpy(optic.paraxial.XPL())).ravel()[0])
    xpd_mm = float(np.asarray(be.to_numpy(optic.paraxial.XPD())).ravel()[0])
    image_z_mm = float(np.max(z))

    # Optiland reports XPL relative to the image surface. Rays travel in air
    # after the last surface, so projecting them along their own direction to
    # the pupil plane is exact rather than an approximation.
    pupil_z_mm = image_z_mm + xpl_mm
    step_mm = (pupil_z_mm - z) / N
    x_pupil_mm = x + L * step_mm
    y_pupil_mm = y + M * step_mm

    max_dx = float(np.max(np.abs(L)))
    max_dy = float(np.max(np.abs(M)))
    pitch_x_max_m = WAVELENGTH_UM * 1e-6 / (2.0 * max_dx) if max_dx > 0 else float("inf")
    pitch_y_max_m = WAVELENGTH_UM * 1e-6 / (2.0 * max_dy) if max_dy > 0 else float("inf")

    extent_x_m = 2.0 * float(np.max(np.abs(x_pupil_mm))) * MM_TO_M
    extent_y_m = 2.0 * float(np.max(np.abs(y_pupil_mm))) * MM_TO_M
    pitch_binding_m = min(pitch_x_max_m, pitch_y_max_m)
    extent_binding_m = max(extent_x_m, extent_y_m)

    n_critical = int(np.ceil(extent_binding_m / pitch_binding_m)) if pitch_binding_m > 0 else -1
    return {
        "system": label,
        "ray_count": int(x.size),
        "exit_pupil_location_mm_from_image": xpl_mm,
        "exit_pupil_diameter_mm": xpd_mm,
        "exit_pupil_z_mm": pupil_z_mm,
        "image_plane_z_mm": image_z_mm,
        "max_abs_direction_cosine_x": max_dx,
        "max_abs_direction_cosine_y": max_dy,
        "max_direction_norm_xy": float(np.max(np.sqrt(L**2 + M**2))),
        "nyquist_pitch_x_max_m": pitch_x_max_m,
        "nyquist_pitch_y_max_m": pitch_y_max_m,
        "pupil_extent_x_m": extent_x_m,
        "pupil_extent_y_m": extent_y_m,
        "grid_n_at_critical_pitch": n_critical,
        "grid_n_at_2x_oversampling": 2 * n_critical,
        "pixels_at_critical": n_critical**2,
    }


def case_singlet_envelope(semi_aperture_mm: float, *, scale: float = 1.0) -> dict[str, Any]:
    """Sampling envelope for the reference singlet at a given aperture and scale.

    Traced through `Optic.trace`, i.e. the hexapolar 2-D pupil the adapter
    actually samples, so both axes are exercised. A 1-D pupil would leave the
    y-axis Nyquist limit untested, and the limit is per axis.
    """
    optic, efl_mm, bfl_mm = _build_singlet(semi_aperture_mm=semi_aperture_mm, scale=scale)
    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=24)
    report = _exit_pupil_report(
        optic,
        rays,
        f"reference singlet, scale {scale}, semi-aperture {semi_aperture_mm} mm",
    )
    report["scale"] = scale
    report["propagation_distance_pupil_to_focus_mm"] = bfl_mm
    report["efl_mm"] = efl_mm
    report["f_number"] = efl_mm / (2.0 * semi_aperture_mm)
    # The float32 propagation error model confirmed by
    # case_chromatix_float32_vs_distance: relative error ~ eps32 * 2*pi*z/lambda.
    phase_rad = 2.0 * np.pi * bfl_mm * MM_TO_M / (WAVELENGTH_UM * 1e-6)
    report["propagation_transfer_phase_rad"] = phase_rad
    report["projected_float32_field_error"] = float(np.finfo(np.float32).eps * phase_rad)
    return report


def case_reverse_telephoto_envelope() -> dict[str, Any]:
    """Sampling envelope for the already-validated ReverseTelephoto sample."""
    be.set_backend("numpy")
    lens = ReverseTelephoto()
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=32)
    return _exit_pupil_report(lens, rays, "ReverseTelephoto (M1-validated sample)")


def case_coupler_throughput() -> dict[str, Any]:
    """Measure rays x pixels throughput of C_RAY_TO_WAVE, then price the envelope."""
    from multiscale_optics_agent.couplers.contracts import RayBundle, ReferencePlane
    from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave

    wavelength_m = WAVELENGTH_UM * 1e-6
    measurements = []
    for ray_count, grid in ((256, 64), (256, 128), (1024, 128), (1024, 256)):
        rng = np.random.default_rng(20260812)
        positions = np.zeros((ray_count, 3), dtype=np.float64)
        positions[:, 0] = rng.uniform(-1e-3, 1e-3, ray_count)
        positions[:, 1] = rng.uniform(-1e-3, 1e-3, ray_count)
        theta = rng.uniform(0.0, 0.02, ray_count)
        phi = rng.uniform(0.0, 2 * np.pi, ray_count)
        directions = np.stack(
            (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1
        )
        bundle = RayBundle(
            positions_m=positions,
            directions=directions,
            wavelength_m=wavelength_m,
            amplitude=np.ones(ray_count, dtype=np.complex128),
            optical_path_length_m=rng.uniform(0.0, 1e-6, ray_count),
            optical_path_length_reference="probe: synthetic, declared",
            reference_plane=ReferencePlane(name="m3-feasibility-timing-plane", z_m=0.0),
        )
        pitch = wavelength_m / (2.0 * 0.02) / 2.0
        started = time.perf_counter()
        ray_to_wave(
            bundle,
            grid_shape=(grid, grid),
            sample_pitch_m=(pitch, pitch),
            enforce_grid_nyquist=False,
        )
        elapsed = time.perf_counter() - started
        work = ray_count * grid * grid
        measurements.append(
            {
                "ray_count": ray_count,
                "grid": grid,
                "ray_pixel_products": work,
                "seconds": elapsed,
                "ray_pixels_per_second": work / elapsed,
            }
        )
    throughput = float(np.median([m["ray_pixels_per_second"] for m in measurements]))
    return {
        "claim": "C_RAY_TO_WAVE cost is rays x pixels; this prices the candidate grids",
        "measurements": measurements,
        "median_ray_pixels_per_second": throughput,
        "note": (
            "shared unpinned machine, same caveat as M1/M2 timings; used to "
            "reject infeasible configurations, not as a regression envelope"
        ),
    }


def _asm_reference_float64(
    u: np.ndarray, *, wavelength_m: float, pitch_m: float, z_m: float
) -> np.ndarray:
    """Independent float64 angular-spectrum propagation, no padding.

    Deliberately not Chromatix: this is the reference the float32 cast is
    measured against, so it must not share Chromatix's dtype behaviour.
    """
    n = u.shape[0]
    fx = np.fft.fftfreq(n, d=pitch_m)
    fy = np.fft.fftfreq(n, d=pitch_m)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    argument = (1.0 / wavelength_m) ** 2 - FX**2 - FY**2
    propagating = argument > 0.0
    kz = 2.0 * np.pi * np.sqrt(np.where(propagating, argument, 0.0))
    transfer = np.where(propagating, np.exp(1j * kz * z_m), 0.0)
    return np.fft.ifft2(np.fft.fft2(u) * transfer)


def case_chromatix_float32_vs_distance() -> dict[str, Any]:
    """How far can the float32 ASM propagate before its own dtype dominates?

    `chromatix.core.field.ScalarField.__init__` casts unconditionally to
    `complex64`. The transfer-function phase is `2*pi*z*sqrt(1/lambda^2 - f^2)`,
    so its magnitude grows as `z/lambda`: at 550 nm, 47 mm is ~5.4e5 radians,
    while M1's verified ASM evidence sits at 40 um, ~460 radians. Rounding a
    large phase argument in float32 perturbs the *differences* between spectral
    components, which is what forms the PSF, so this is measured rather than
    assumed.
    """
    import jax.numpy as jnp
    from chromatix import functional as cf

    grid = 128
    pitch_m = 4.0e-6
    wavelength_m = WAVELENGTH_UM * 1e-6

    # A converging spherical wave: the field this slice actually propagates.
    coords = (np.arange(grid) - grid // 2) * pitch_m
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    aperture = (0.4 * grid * pitch_m / 2.0) ** 2 >= (X**2 + Y**2)

    sweep = []
    for z_mm in (0.04, 0.4, 4.0, 47.06):
        z_m = z_mm * MM_TO_M
        phase = -2.0 * np.pi / wavelength_m * np.sqrt(X**2 + Y**2 + z_m**2)
        u = (aperture * np.exp(1j * phase)).astype(np.complex128)

        reference = _asm_reference_float64(u, wavelength_m=wavelength_m, pitch_m=pitch_m, z_m=z_m)
        field_in = cf.Field.build(
            jnp.asarray(u, dtype=jnp.complex64),
            jnp.asarray([[pitch_m, pitch_m]]),
            wavelength_m,
        )
        field_out = cf.asm_propagate(field_in, z=z_m, n=1.0, pad_width=0)
        observed = np.asarray(field_out.u, dtype=np.complex128).reshape(grid, grid)

        denominator = float(np.linalg.norm(reference))
        relative_l2 = float(np.linalg.norm(observed - reference) / denominator)
        transfer_phase_rad = 2.0 * np.pi * z_m / wavelength_m
        sweep.append(
            {
                "z_mm": z_mm,
                "transfer_phase_rad": transfer_phase_rad,
                "eps32_times_phase": float(np.finfo(np.float32).eps * transfer_phase_rad),
                "relative_l2_error_vs_float64": relative_l2,
                "peak_intensity_relative_error": float(
                    abs(np.max(np.abs(observed) ** 2) - np.max(np.abs(reference) ** 2))
                    / np.max(np.abs(reference) ** 2)
                ),
            }
        )
    return {
        "claim": (
            "the float32 cast inside Chromatix bounds the usable propagation "
            "distance, independently of grid or aperture"
        ),
        "reference": "independent float64 angular-spectrum propagation in this probe",
        "grid": grid,
        "sample_pitch_m": pitch_m,
        "sweep": sweep,
        "why_it_matters": (
            "a pupil-to-focus distance is set by the lens focal length, so if the "
            "error grows with z the reference system's SCALE is a protocol "
            "decision, not just its NA"
        ),
    }


def case_chosen_configurations(throughput: float) -> dict[str, Any]:
    """The two systems the protocol selects, fully specified and priced.

    The reference singlet is taken at scale 0.1 and f/9.7 rather than at its
    Rayleigh-limit aperture: at 1/10 scale the same f-number leaves ~lambda/60 of
    spherical aberration, so the Airy comparison in M3.8 measures the slice
    rather than the singlet's residual aberration.
    """
    chosen = {}

    scale, f_number = 0.1, 9.7
    efl_mm = SINGLET_RADIUS_MM * scale / (SINGLET_N - 1.0)
    semi_aperture_mm = efl_mm / (2.0 * f_number)
    heights = np.linspace(0.0, semi_aperture_mm, 33)
    wavefront = wavefront_error_waves(heights, semi_aperture_mm=semi_aperture_mm, scale=scale)
    envelope = case_singlet_envelope(semi_aperture_mm, scale=scale)
    envelope["peak_to_valley_waves"] = wavefront["peak_to_valley_waves"]
    envelope["rms_waves"] = wavefront["rms_waves"]
    envelope["marechal_strehl_estimate"] = float(
        np.exp(-((2.0 * np.pi * wavefront["rms_waves"]) ** 2))
    )
    numerical_aperture = envelope["max_abs_direction_cosine_x"]
    envelope["airy_radius_um"] = 1.22 * WAVELENGTH_UM / numerical_aperture
    envelope["airy_radius_in_pixels_at_2x"] = (
        envelope["airy_radius_um"] * 1e-6 / (envelope["nyquist_pitch_x_max_m"] / 2.0)
    )
    envelope["float32_intensity_error"] = _measure_float32_intensity_error(
        z_mm=envelope["propagation_distance_pupil_to_focus_mm"],
        numerical_aperture=numerical_aperture,
    )
    envelope["pricing"] = price_configuration(
        label="reference singlet, 2x oversampled, 4096 rays",
        rays=4096,
        grid=int(envelope["grid_n_at_2x_oversampling"]),
        throughput=throughput,
    )
    chosen["reference_singlet"] = envelope

    telephoto = case_reverse_telephoto_envelope()
    distance_mm = telephoto["image_plane_z_mm"] - telephoto["exit_pupil_z_mm"]
    telephoto["propagation_distance_pupil_to_image_mm"] = distance_mm
    telephoto["float32_intensity_error"] = _measure_float32_intensity_error(
        z_mm=distance_mm,
        numerical_aperture=telephoto["max_abs_direction_cosine_x"],
    )
    telephoto["pricing"] = price_configuration(
        label="ReverseTelephoto, 2x oversampled, 4096 rays",
        rays=4096,
        grid=int(telephoto["grid_n_at_2x_oversampling"]),
        throughput=throughput,
    )
    chosen["reverse_telephoto"] = telephoto
    return chosen


def _measure_float32_intensity_error(*, z_mm: float, numerical_aperture: float) -> dict[str, Any]:
    """Float32 vs float64 ASM at one distance, reported for field AND intensity.

    Both are reported because they differ by orders of magnitude: much of the
    float32 phase error is common to every spectral component, so it acts as a
    piston that cancels when the field is squared. The PSF is an intensity, so
    the intensity figure is the one that belongs in the tolerance budget -- but
    quoting only that would hide the field-level cost from anyone who later
    wants the complex field itself.
    """
    import jax.numpy as jnp
    from chromatix import functional as cf

    grid = 128
    wavelength_m = WAVELENGTH_UM * 1e-6
    pitch_m = wavelength_m / (2.0 * numerical_aperture) / 2.0
    z_m = z_mm * MM_TO_M

    coords = (np.arange(grid) - grid // 2) * pitch_m
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    aperture = (0.4 * grid * pitch_m / 2.0) ** 2 >= (X**2 + Y**2)
    phase = -2.0 * np.pi / wavelength_m * np.sqrt(X**2 + Y**2 + z_m**2)
    u = (aperture * np.exp(1j * phase)).astype(np.complex128)

    reference = _asm_reference_float64(u, wavelength_m=wavelength_m, pitch_m=pitch_m, z_m=z_m)
    field_in = cf.Field.build(
        jnp.asarray(u, dtype=jnp.complex64),
        jnp.asarray([[pitch_m, pitch_m]]),
        wavelength_m,
    )
    observed = np.asarray(
        cf.asm_propagate(field_in, z=z_m, n=1.0, pad_width=0).u, dtype=np.complex128
    ).reshape(grid, grid)

    intensity_observed = np.abs(observed) ** 2
    intensity_reference = np.abs(reference) ** 2
    return {
        "z_mm": z_mm,
        "sample_pitch_m": pitch_m,
        "relative_l2_field": float(
            np.linalg.norm(observed - reference) / np.linalg.norm(reference)
        ),
        "relative_l2_intensity": float(
            np.linalg.norm(intensity_observed - intensity_reference)
            / np.linalg.norm(intensity_reference)
        ),
        "peak_intensity_relative_error": float(
            abs(np.max(intensity_observed) - np.max(intensity_reference))
            / np.max(intensity_reference)
        ),
    }


def price_configuration(*, label: str, rays: int, grid: int, throughput: float) -> dict[str, Any]:
    work = rays * grid * grid
    return {
        "configuration": label,
        "ray_count": rays,
        "grid": grid,
        "ray_pixel_products": work,
        "projected_seconds": work / throughput,
        "field_bytes_complex128": grid * grid * 16,
    }


def main() -> None:
    float32_case = case_chromatix_float32_vs_distance()
    throughput_report = case_coupler_throughput()
    throughput = throughput_report["median_ray_pixels_per_second"]

    # The float32 result above makes the system's absolute SIZE a protocol
    # decision: the pupil-to-focus distance is the lens's back focal length, and
    # the propagation error grows with it. Scaling the singlet down at fixed
    # f-number shrinks that distance and the aberration in waves, and leaves the
    # numerical aperture -- hence the Nyquist pitch -- untouched.
    apertures = {}
    envelopes = {}
    pricing = []
    for scale in (1.0, 0.1):
        aperture = case_diffraction_limited_aperture(scale=scale)
        apertures[f"scale_{scale}"] = aperture
        admissible = aperture["largest_admissible"]
        if admissible is None:
            continue
        semi_aperture_mm = float(admissible["semi_aperture_mm"])
        envelope = case_singlet_envelope(semi_aperture_mm, scale=scale)
        envelopes[f"scale_{scale}"] = envelope
        pricing.append(
            price_configuration(
                label=(
                    f"singlet scale {scale}, f/{envelope['f_number']:.1f}, "
                    "2x oversampled, 4096 rays"
                ),
                rays=4096,
                grid=int(envelope["grid_n_at_2x_oversampling"]),
                throughput=throughput,
            )
        )

    telephoto = case_reverse_telephoto_envelope()
    telephoto_distance_mm = telephoto["image_plane_z_mm"] - telephoto["exit_pupil_z_mm"]
    telephoto_phase = 2.0 * np.pi * telephoto_distance_mm * MM_TO_M / (WAVELENGTH_UM * 1e-6)
    telephoto["propagation_distance_pupil_to_image_mm"] = telephoto_distance_mm
    telephoto["propagation_transfer_phase_rad"] = telephoto_phase
    telephoto["projected_float32_field_error"] = float(np.finfo(np.float32).eps * telephoto_phase)
    pricing.append(
        price_configuration(
            label="ReverseTelephoto, 2x oversampled, 4096 rays",
            rays=4096,
            grid=int(telephoto["grid_n_at_2x_oversampling"]),
            throughput=throughput,
        )
    )

    report = {
        "probe": "m3_slice_feasibility",
        "issue": "CHE-31 (M3.2)",
        "wavelength_um": WAVELENGTH_UM,
        "handoff_plane": (
            "exit pupil plane, read from optic.paraxial.XPL()/XPD() rather than "
            "constructed; rays travel in air after the last surface so projecting "
            "them to that plane along their own direction is exact"
        ),
        "chromatix_float32_vs_distance": float32_case,
        "diffraction_limited_aperture_by_scale": apertures,
        "singlet_envelopes_by_scale": envelopes,
        "reverse_telephoto_envelope": telephoto,
        "coupler_throughput": throughput_report,
        "pricing": pricing,
        "chosen_configurations": case_chosen_configurations(throughput),
        "verdict": (
            "the coupler and the grid are cheap; the binding constraint is "
            "Chromatix's float32 cast against the pupil-to-focus distance, which "
            "rules out the 48 mm-focal-length singlet and selects a scaled-down "
            "reference system in the same few-millimetre regime as ReverseTelephoto"
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
