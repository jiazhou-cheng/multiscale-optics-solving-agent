#!/usr/bin/env python3
"""demo3 — hologram plus a refractive lens, both routes (CHE-96, paper Fig 5c).

The configuration where Optiland genuinely enters. SI Table S2, System 3:
a 200x200 SLM at 6.3 um at z = 0, a 3.0 mm air gap, a Thorlabs LA1131-A
plano-convex singlet (front R = 25.8 mm, N-BK7 as index 1.5131, centre
thickness 5.3 mm, flat back, semi-diameter 12.7 mm), and a sensor at
z = 53 mm. Measured EFL of the built system: 50.283 mm.

**This is a characterization, never a validation.** The paper states that no
conventional reference exists for this system, and that is the point of Fig 5c.
So there is no oracle here and nothing below is scored against one. What is
checked instead:

* **RW-F against RW-P** -- the only cross-check available, and the SI S2
  relation applied to a system with a refractive element downstream;
* **seed-to-seed reproducibility** of the speckle across at least three
  independent realizations, which is the SI S6 / Figure S3 methodology and what
  separates physical interference from the undersampling artifacts of Figure S4;
* **energy through the Optiland trace** -- rays clipped by the solver, and the
  power that left with them.

Run in a dedicated GPU session:
    MOA_GPUS=device=0 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo3_hologram_lens.py --preset smoke --backend jax
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_support import (
    RECORDS,
    build_doe,
    device_memory_stats,
    enable_x64_if_needed,
    environment,
    mse_unit_sum,
    ncc,
    write_record,
)

from core.arrays import xp_for
from core.boundary import Frame, RayBundle, ReferencePlane
from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES
from core.coherent_batch import CoherentRayBatch
from core.optical_system import (
    ApertureSpec,
    FieldSpec,
    IdealMaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    SphericalGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)
from core.precision import ArrayNamespace, DeviceKind, DevicePlacement, DType, Precision
from couplers.cascade import planar_doe_step
from couplers.patch import patch_secondary_rays, plan_patches
from couplers.ray_to_wave import DEFAULT_KSPACE_OVERSAMPLE, Projection, Reconstruction
from couplers.streaming import StreamingReconstruction
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import (
    configure_optiland_execution,
    plan_trace_bridges,
    surface_positions_m,
    trace_ray_batch,
)

WAVELENGTH_M = 0.7e-6
PITCH_M = 6.3e-6
GRID_N = 200
GAP_DOE_LENS_MM = 3.0
SENSOR_Z_M = 53.0e-3

#: SI Table S2, System 3, verbatim.
TABLE_S2 = {
    "rw_f": {
        "patch": "full",
        "incident_rays": 3.3e4,
        "secondary_rays_per_incident": 1.6e5,
        "batches": 128,
        "peak_memory_gb": 16.919,
        "runtime_s": 28.387,
        "hardware": "1x NVIDIA RTX A6000 48 GB, CUDA 12.4",
    },
    "rw_p": {
        "patch_px": 100,
        "pad_factor": 2,
        "incident_rays": 2.6e5,
        "secondary_rays_per_incident": 1e4,
        "batches": 8,
        "peak_memory_gb": 17.003,
        "runtime_s": 41.137,
        "hardware": "1x NVIDIA RTX A6000 48 GB, CUDA 12.4",
    },
}


def demo3_system() -> OpticalSystemSpec:
    """The prescription, in the canonical schema, following `M3_SINGLET_REF`.

    Surface 1 is the DOE plane itself -- a plane in air carrying no power. It
    exists so that `surface_positions_m(lens)[1]` equals the launch plane
    exactly, which `trace_ray_batch` requires to within a nanometre: tracing a
    bundle whose declared plane is not the first traced surface would propagate
    it through a different system than the one the patches were taken from.

    N-BK7 as `IdealMaterialSpec(1.5131)` rather than the catalog glass, exactly
    as the issue specifies. That is deliberate: Optiland resolves a bare glass
    name by substring filter and Levenshtein ranking, and this reproduction is
    of a stated index, not of a dispersion curve.
    """
    return OpticalSystemSpec(
        name="Demo3HologramLA1131A",
        description=(
            "CHE-96 demo3 (paper Fig 5c): 200x200 SLM at z = 0, 3.0 mm air gap, "
            "Thorlabs LA1131-A plano-convex singlet (R = 25.8 mm, N-BK7 as index "
            "1.5131, 5.3 mm centre thickness, flat back), sensor at 53 mm."
        ),
        object_distance_mm=0.0,
        surfaces=(
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=GAP_DOE_LENS_MM,
                comment="DOE plane -- no power; the launch plane for the patch bundle",
            ),
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=25.8),
                thickness_mm=5.3,
                material=IdealMaterialSpec(refractive_index=1.5131),
                is_stop=True,
                comment="LA1131-A front (convex toward the DOE)",
            ),
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=44.7,
                comment="LA1131-A back (flat); 44.7 mm to the sensor at z = 53 mm",
            ),
        ),
        aperture=ApertureSpec(value_mm=25.4),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=0.7, is_primary=True),),
    )


def _incident_plane_wave(plane: ReferencePlane) -> RayBundle:
    """One on-axis ray, which `ray_to_wave` accumulates into a unit plane wave.

    `planar_doe_step` takes an incident *bundle*, not a field: it accumulates
    the bundle onto the DOE plane, multiplies by the transmission, transforms,
    and resamples. Demo3's illumination is a normally-incident plane wave, and
    a single ray with `reconstruction_normalization="none"` is exactly that --
    one wavelet with zero transverse ramp is a constant across the grid.
    """
    return RayBundle(
        positions_m=np.zeros((1, 3)),
        directions=np.array([[0.0, 0.0, 1.0]]),
        wavelength_m=WAVELENGTH_M,
        reference_plane=plane,
        frame=Frame(),
        amplitude=np.ones(1, dtype=np.complex128),
        optical_path_length_m=np.zeros(1),
        optical_path_length_reference=(
            "zero at the DOE plane; the illumination is a normally-incident "
            "plane wave and carries no path before it"
        ),
        reconstruction_normalization="none",
    )


def run_route(
    doe,
    lens,
    *,
    sensor_shape: tuple[int, int],
    sensor_pitch_m: float,
    patch_px: int,
    pad_factor: int,
    pad_width: int,
    patch_count: int | None,
    secondary_count: int | None,
    batches: int,
    seed: int,
    backend: str,
    precision: str,
    secondary_chunks: int = 1,
    route: str = "patch",
    reconstruction_route: str = "ramp_sum",
    kspace_oversample: float = DEFAULT_KSPACE_OVERSAMPLE,
) -> dict[str, Any]:
    """Rays from the DOE, an Optiland trace, and one coherent sensor field.

    Two emitters, and the distinction is the whole comparison:

    ``route="patch"`` is `C_PATCH_WFT` -- each incident ray gets its own local
    patch, transformed on its own padded grid. This is RW-P.

    ``route="full_field"`` is `C_PLANAR_DOE_STEP` -- one global angular spectrum
    of the whole transmitted field, resampled at many launch positions. This is
    RW-F, and for demo3 it needs many launch positions rather than demo2's one,
    because the refractive lens downstream introduces field-dependent
    aberrations that a single primary ray cannot represent. It is deliberately
    the CHE-95 operator and not "the patch route with one patch": those coincide
    only when there is a single launch position, which is demo2's case and not
    this one.
    """
    doe_plane = ReferencePlane(name="doe", z_m=0.0)
    sensor = ReferencePlane(name="sensor", z_m=SENSOR_Z_M)
    rng = np.random.default_rng(seed)
    device = DevicePlacement(
        kind=DeviceKind.CUDA if backend == "jax" else DeviceKind.CPU, index=0
    )
    namespace = ArrayNamespace.JAX if backend == "jax" else ArrayNamespace.NUMPY
    xp = xp_for(namespace)
    real_dtype = "float64" if precision == "fp64" else "float32"

    plan = None
    if route == "patch":
        plan = plan_patches(
            grid_shape=doe.grid_shape,
            sample_pitch_m=doe.pitch_m,
            patch_px=patch_px,
            pad_factor=pad_factor,
            patch_count=patch_count,
            rng=rng,
        )
        emitters = np.asarray(plan.centers_xy_m)
    else:
        # RW-F's "incident rays" are LAUNCH POSITIONS on one global spectrum,
        # not patch centres. Drawn uniformly on the DOE's own sample grid over
        # the aperture: the paper does not justify a particular primary-position
        # density, so the simplest declared one is used and recorded rather than
        # inherited silently from the reference implementation.
        half = doe.grid_shape[0] // 2
        rows = rng.integers(-half, half + 1, size=max(1, patch_count or 1))
        cols = rng.integers(-half, half + 1, size=max(1, patch_count or 1))
        emitters = np.column_stack(
            [cols * doe.pitch_m[1], rows * doe.pitch_m[0]]
        ).astype(np.float64)
    n_patches = emitters.shape[0]
    groups = np.array_split(np.arange(n_patches), max(1, min(batches, n_patches)))

    # Splitting the SECONDARY draw as well, not only the patch list. Chunking
    # over patches alone cannot bound the full-aperture route, which has exactly
    # one patch and would put its entire budget in a single call however many
    # batches were asked for -- measured as a 13.4 GB allocation on a 1680^2
    # sensor, refused by the device.
    #
    # This is legitimate rather than a workaround: each sub-chunk draws its own
    # independent sample from the same spectrum, the parts sum to the requested
    # budget, and the 1/N is applied once at finalize, so a split cannot change
    # the estimator. Enumeration is the exception -- there is nothing stochastic
    # to split -- and is left whole.
    if secondary_count is None:
        secondary_parts: list[int | None] = [None]
    else:
        wanted = max(1, int(np.ceil(int(secondary_count) / max(1, secondary_chunks))))
        remaining = int(secondary_count)
        secondary_parts = []
        while remaining > 0:
            take = min(wanted, remaining)
            secondary_parts.append(take)
            remaining -= take

    incident = _incident_plane_wave(doe_plane)

    def emit(indices: np.ndarray, part: int | None):
        """One chunk of secondary rays, from whichever emitter this route uses."""
        if route == "patch":
            bundle, patch_diagnostics = patch_secondary_rays(
                doe.transmission,
                plan=dataclasses.replace(plan, centers_xy_m=emitters[indices]),
                sample_pitch_m=doe.pitch_m,
                wavelength_m=WAVELENGTH_M,
                plane=doe_plane,
                secondary_count=part,
                rng=rng,
            )
            return bundle, patch_diagnostics.as_dict()
        bundle, _, cascade = planar_doe_step(
            incident,
            doe.transmission,
            grid_shape=doe.grid_shape,
            sample_pitch_m=doe.pitch_m,
            plane=doe_plane,
            launch_positions_xy_m=emitters[indices],
            secondary_count=part,
            pad_width=pad_width,
            rng=rng,
        )
        return bundle, cascade.as_dict()

    if secondary_count is None:
        probe, _ = emit(np.array([0]), None)
        per_patch = int(probe.count)
        del probe
    else:
        per_patch = int(secondary_count)
    total_rays = n_patches * per_patch

    reconstruction = StreamingReconstruction(
        grid_shape=sensor_shape,
        sample_pitch_m=(sensor_pitch_m, sensor_pitch_m),
        plane=sensor,
        wavelength_m=WAVELENGTH_M,
        namespace=namespace,
        complex_dtype=DType.COMPLEX128 if precision == "fp64" else DType.COMPLEX64,
        total_rays=total_rays,
        projection=Projection.ASM_CONSISTENT,
        # No matched k-grid is available here, and saying so is the point: the
        # rays reaching this plane have been refracted by the singlet, so their
        # directions are no longer bins of the DOE's spectrum and no k-grid
        # period puts them on nodes. demo2's reconstruction is exact; this one is
        # an interpolation, and `on_node_fraction` in the record reports which.
        reconstruction=Reconstruction(reconstruction_route),
        kspace_oversample=kspace_oversample,
    )

    trace_plans = None
    next_id = 0
    launched_power = 0.0
    survived_power = 0.0
    clipped = 0
    # A ray emitted from a patch that lies wholly outside the DOE carries zero
    # amplitude, and `trace_ray_batch` marks it invalid because Optiland's
    # `intensity > 0` test cannot distinguish "clipped by an aperture" from
    # "launched with no power". Counting them separately is the difference
    # between an energy loss and an empty draw, and only the first is a
    # physical statement about the system.
    launched_empty = 0
    captured_power = 0.0
    half_extent_m = 0.5 * sensor_shape[0] * sensor_pitch_m
    first_trace: dict[str, Any] | None = None
    started = time.perf_counter()

    chunk_count = 0
    # Per-stage wall clock, because CHE-96 attributed all of demo3's cost to the
    # O(N_rays x N_pixels) reconstruction and CHE-101 measured the reconstruction
    # kernel at 0.18 s per 1e6-ray chunk against a 2.4 s chunk. A cost model that
    # names the wrong stage sends the next ticket to optimize the wrong thing, so
    # the breakdown is recorded rather than inferred from a total.
    stage_s: dict[str, float] = {
        "emit_patch_spectra": 0.0,
        "host_to_device": 0.0,
        "optiland_trace": 0.0,
        "power_bookkeeping": 0.0,
        "reconstruct": 0.0,
    }
    for group in groups:
        if group.size == 0:
            continue
        for part in secondary_parts:
            chunk_count += 1
            stage_started = time.perf_counter()
            bundle, emitter_diagnostics = emit(group, part)
            stage_s["emit_patch_spectra"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            chunk_amplitude = np.abs(np.asarray(bundle.amplitude)) ** 2
            launched_power += float(chunk_amplitude.sum())
            launched_empty += int(np.count_nonzero(chunk_amplitude == 0.0))
            moved = dataclasses.replace(
                bundle,
                positions_m=xp.asarray(np.asarray(bundle.positions_m), dtype=real_dtype),
                directions=xp.asarray(np.asarray(bundle.directions), dtype=real_dtype),
                amplitude=xp.asarray(
                    np.asarray(bundle.amplitude),
                    dtype="complex128" if precision == "fp64" else "complex64",
                ),
                optical_path_length_m=xp.asarray(
                    np.asarray(bundle.optical_path_length_m), dtype=real_dtype
                ),
            )
            if str(moved.directions.dtype) != real_dtype:
                raise RuntimeError(
                    f"asked for {real_dtype} and got {moved.directions.dtype}; on JAX "
                    "this means jax_enable_x64 was not set before the first array"
                )
            batch = CoherentRayBatch(
                bundle=moved,
                ray_id=np.arange(next_id, next_id + moved.count, dtype=np.int64),
                valid=xp.ones(moved.count, dtype=bool),
            )
            next_id += moved.count
            stage_s["host_to_device"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if trace_plans is None:
                trace_plans = plan_trace_bridges(
                    batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=device
                )
            traced, trace_diagnostics = trace_ray_batch(
                batch, lens, image_plane=sensor, plans=trace_plans, skip=1
            )
            if first_trace is None:
                first_trace = {
                    key: value for key, value in trace_diagnostics.items() if key != "residency"
                }
                first_trace["residency"] = trace_diagnostics["residency"]
                first_trace["emitter"] = emitter_diagnostics
            stage_s["optiland_trace"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            clipped += int(trace_diagnostics["invalid_rays"])
            traced_power = np.abs(np.asarray(traced.bundle.amplitude)) ** 2
            survived_power += float(traced_power.sum())
            # How much of the traced power lands inside the sensor at all. The
            # sensor is a declared choice here, not a transcription, so the
            # fraction it captures is part of the result rather than an
            # assumption behind it: an intensity map that silently holds 40% of
            # the light invites a reader to treat it as the whole image.
            traced_xy = np.asarray(traced.bundle.positions_m)[:, :2]
            inside = (np.abs(traced_xy[:, 0]) <= half_extent_m) & (
                np.abs(traced_xy[:, 1]) <= half_extent_m
            )
            captured_power += float(traced_power[inside].sum())
            stage_s["power_bookkeeping"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            reconstruction.add_chunk(traced)
            stage_s["reconstruct"] += time.perf_counter() - stage_started
            del bundle, moved, batch, traced

    result = reconstruction.finalize(provenance={"probe": "demo3"})
    wall_clock_s = time.perf_counter() - started

    return {
        "field": np.asarray(result.field.u),
        "route_kind": route,
        "plan": (
            {
                "patch_px": plan.patch_px,
                "pad_px": plan.pad_px,
                "pad_requested": patch_px * pad_factor,
                "coverage": plan.coverage,
                "curvature_bound_rad": plan.curvature_bound_rad,
                "patch_count": n_patches,
            }
            if plan is not None
            else {
                "emitter": "C_PLANAR_DOE_STEP -- one global spectrum",
                "launch_positions": n_patches,
                "pad_width": pad_width,
            }
        ),
        "stage_wall_clock_s": {
            **{k: round(v, 3) for k, v in stage_s.items()},
            "note": (
                "Sums to slightly less than wall_clock_s: finalize and the plan "
                "setup are outside the loop. `power_bookkeeping` is this probe's "
                "own accounting -- three host round-trips per chunk -- not part of "
                "the physics, and is separated so it cannot be mistaken for one."
            ),
        },
        "reconstruction": {
            "route": reconstruction_route,
            "kspace_oversample": kspace_oversample if reconstruction_route != "ramp_sum" else None,
            "measured": (reconstruction._first_diagnostics or {}).get("kspace"),
        },
        "total_rays": total_rays,
        "secondary_per_patch": per_patch,
        "batches": chunk_count,
        "patch_groups": len(groups),
        "secondary_chunks_per_group": len(secondary_parts),
        "wall_clock_s": wall_clock_s,
        "rays_per_second": total_rays / wall_clock_s,
        "energy": {
            "note": (
                "sum |a|^2 over the emitted rays, before and after the trace. "
                "Optiland clips by zeroing intensity rather than by removing "
                "rows, and trace_ray_batch zeroes the matching amplitude, so the "
                "difference is exactly the power that left through an aperture. "
                "A clipped ray keeps its place in N_total: it was drawn, and the "
                "operator being estimated includes the vignetting."
            ),
            "launched_sum_abs_amplitude_squared": launched_power,
            "survived_sum_abs_amplitude_squared": survived_power,
            "transmitted_fraction": (
                survived_power / launched_power if launched_power > 0 else None
            ),
            "invalidated_rays": clipped,
            "launched_with_zero_amplitude": launched_empty,
            "clipped_with_power": max(0, clipped - launched_empty),
            "captured_by_sensor_fraction": (
                captured_power / survived_power if survived_power > 0 else None
            ),
            "sensor_half_extent_m": half_extent_m,
            "clipped_with_power_fraction": (
                max(0, clipped - launched_empty) / total_rays if total_rays else None
            ),
            "empty_draw_note": (
                "patch centres are drawn over the aperture DILATED by half a patch, "
                "so a centre near the rim yields a patch that is partly or wholly "
                "outside the DOE. Those rays carry zero amplitude by construction "
                "and are not a loss -- the coverage factor A_draw / A_patch is "
                "exactly the correction for them. They are counted here because "
                "Optiland's intensity > 0 test cannot tell them from a clip, and "
                "reporting the sum as 'clipped' would invent an energy loss."
            ),
        },
        "first_chunk_trace": first_trace,
        "streaming": result.as_dict(),
        "device_memory": device_memory_stats(),
        "actual": {
            "field_dtype": str(result.field.u.dtype),
            "field_device": str(getattr(result.field.u, "device", "host")),
        },
    }


#: Sensor pitch is 4.2 um everywhere, and it is a *constraint*, not a taste.
#: `ray_to_wave` refuses a grid that cannot represent the steepest ray:
#: |d| < lambda / (2 * pitch). Measured on the traced bundle, |d|max is 0.052
#: for the full-aperture route and **0.070** for the sub-aperture one, because
#: patch centres are drawn over the aperture dilated by half a patch and a ray
#: launched 0.95 mm off axis is deflected by h/f on top of its own diffraction
#: angle. 4.2 um gives a limit of 0.0833, which clears 0.070 with margin; 6.3 um
#: gives 0.0556 and is refused outright.
SENSOR_PITCH_M = 4.2e-6

#: 1680 x 4.2 um = 7.06 mm, which encloses **99.0%** of the traced power
#: (measured: 91.7% at +/- 3.0 mm, 99.1% at +/- 3.5 mm). The paper states no
#: sensor grid for this system, so this is a declared choice; it is set by the
#: image extent, which is the mask's own Nyquist frequency focused by a 50.3 mm
#: lens, +/- lambda f / (2 * pitch) = +/- 2.79 mm, plus the DOE's own width.
SENSOR_PX = 1680

PRESETS: dict[str, dict[str, Any]] = {
    # Plumbing only: a 128-px sensor sees 3% of the image. Enough to exercise
    # the patch -> Optiland -> reconstruction path on the CPU, and the record
    # says what it is.
    "smoke": {
        "sensor_px": 128,
        "sensor_pitch_m": SENSOR_PITCH_M,
        "rw_f": {
            "route": "full_field",
            "patch_px": None,
            "pad_factor": 1,
            "pad_width": 100,
            "patch_count": 64,
            "secondary_count": 500,
            "batches": 4,
            "precision": "fp64",
        },
        "rw_p": {
            "route": "patch",
            "patch_px": 101,
            "pad_factor": 2,
            "pad_width": 0,
            "patch_count": 64,
            "secondary_count": 1_000,
            "batches": 4,
            "precision": "fp64",
        },
    },
    # The Table S2 *configuration* at the budget our cost model can reach. The
    # ray counts are NOT the paper's and the record says so in
    # `budget_shortfall`: our reconstruction is O(rays x pixels) and this
    # sensor has 2.8e6 pixels, so the paper's 2.6e9 secondary rays would be
    # 7.4e15 ray-pixel products. That is the k-space fast path's case, made
    # numerically, and the issue asks for it to be reported as evidence rather
    # than as a failure.
    # The configuration that actually converges. Same optical system, same
    # Table S2 patch geometry, a smaller sensor and a much larger ray budget.
    #
    # The reason is arithmetic and worth stating plainly: reconstruction noise
    # is set by rays *per pixel*, and the full 1680^2 sensor at 8e6 rays is 2.8
    # rays per pixel. Measured there, both routes reproduce across seeds at NCC
    # 6e-4 and 5e-5 -- which is not a near miss, it is two independent noise
    # fields, and it is SI Figure S4's undersampling artifact reproduced exactly.
    # 420^2 at 3e7 rays is 341 rays per pixel and shows structure.
    #
    # The cost is coverage: this sensor spans +/- 0.88 mm of a +/- 2.8 mm image,
    # and each run records the fraction of traced power it actually captured.
    "characterization": {
        "sensor_px": 420,
        "sensor_pitch_m": SENSOR_PITCH_M,
        "rw_f": {
            "route": "full_field",
            "patch_px": None,
            "pad_factor": 1,
            "pad_width": 200,
            "patch_count": 3_000,
            "secondary_count": 20_000,
            "batches": 60,
            "precision": "fp32",
        },
        "rw_p": {
            "route": "patch",
            "patch_px": 101,
            "pad_factor": 2,
            "pad_width": 0,
            "patch_count": 3_000,
            "secondary_count": 20_000,
            "batches": 60,
            "precision": "fp32",
        },
    },
    "paper_configuration": {
        "sensor_px": SENSOR_PX,
        "sensor_pitch_m": SENSOR_PITCH_M,
        # Table S2 ratio kept (many launch positions, large SSR), budget cut.
        "rw_f": {
            "route": "full_field",
            "patch_px": None,
            "pad_factor": 1,
            "pad_width": 200,
            "patch_count": 2_000,
            "secondary_count": 10_000,
            "batches": 40,
            "precision": "fp32",
        },
        "rw_p": {
            "route": "patch",
            "patch_px": 101,
            "pad_factor": 2,
            "pad_width": 0,
            "patch_count": 2_000,
            "secondary_count": 10_000,
            "batches": 40,
            "precision": "fp32",
        },
    },
}


def build_doe_and_lens(*, backend: str, precision: str):
    """The demo3 optical system, built in the one order that works.

    Returns ``(doe, lens, spec, execution)``: the prescription and the resolved
    execution state come back with the objects because both are recorded in
    every demo3 artifact, and a probe that rebuilt them separately could record
    a configuration it did not run.

    Extracted so a second probe cannot rebuild it and get the order wrong.
    `configure_optiland_execution` switches Optiland's **global** backend, and a
    surface built under the previous one keeps its geometry parameters in that
    namespace: the trace then fails as `numpy.ndarray * Tensor` one frame inside
    an optiland geometry class, which is not where anyone would look for an
    ordering bug. Configure first, build second, and never the reverse.
    """
    doe = build_doe("demo3_smile_phase_profile.npy", pitch_m=PITCH_M)
    spec = demo3_system()
    execution = configure_optiland_execution(
        device=DevicePlacement(
            kind=DeviceKind.CUDA if backend == "jax" else DeviceKind.CPU, index=0
        ),
        precision=Precision.FP64 if precision == "fp64" else Precision.FP32,
        enable_grad=False,
    )
    lens = build_optiland_system(spec)
    if execution.grad_enabled:
        raise RuntimeError(
            "Optiland grad mode is enabled; this is a forward characterization and "
            "must not record computational graphs"
        )
    return doe, lens, spec, execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--routes", default="rw_f,rw_p")
    parser.add_argument("--seeds", default="20260822")
    parser.add_argument("--sensor-px", type=int, default=None)
    parser.add_argument("--sensor-pitch-um", type=float, default=None)
    parser.add_argument("--patch-count", type=int, default=None)
    parser.add_argument("--secondary-count", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None, help="patch groups")
    parser.add_argument(
        "--rays-per-chunk",
        type=float,
        default=2e5,
        help=(
            "device budget. The full-aperture route has ONE patch, so patch "
            "grouping alone cannot bound it; the secondary draw is split too."
        ),
    )
    parser.add_argument(
        "--reconstruction",
        choices=("ramp_sum", "kspace_splat"),
        default="ramp_sum",
        help="ramp_sum is the exact O(rays x pixels) route; kspace_splat is CHE-101's fast path",
    )
    parser.add_argument("--kspace-oversample", type=float, default=DEFAULT_KSPACE_OVERSAMPLE)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--agreement-from",
        default=None,
        help=(
            "two saved field files, comma separated, e.g. 'demo3_rw_f,demo3_rw_p'. "
            "Computes the route agreement across them and exits. Exists because "
            "one run of each route at a converged budget is ~150 s and six of "
            "them do not fit in one command; the alternative was to shrink the "
            "budget until the comparison stopped meaning anything."
        ),
    )
    args = parser.parse_args()

    if args.agreement_from:
        return _write_agreement(
            [name.strip() for name in args.agreement_from.split(",") if name.strip()],
            args.output_name or "demo3_route_agreement",
        )

    preset = PRESETS[args.preset]
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sensor_px = args.sensor_px or preset["sensor_px"]
    sensor_pitch_m = (
        args.sensor_pitch_um * 1e-6 if args.sensor_pitch_um else preset["sensor_pitch_m"]
    )
    x64 = enable_x64_if_needed(
        backend=args.backend,
        precisions=[preset[r]["precision"] for r in routes if r in preset],
    )

    doe, lens, spec, execution = build_doe_and_lens(
        backend=args.backend,
        precision=(
            "fp64"
            if any(preset[r]["precision"] == "fp64" for r in routes if r in preset)
            else "fp32"
        ),
    )

    record: dict[str, Any] = {
        "probe": "demo3_hologram_lens",
        "issue": "CHE-96",
        "system": "paper Fig 5c / SI Table S2 System 3 -- hologram + LA1131-A singlet",
        "preset": args.preset,
        "environment": {**environment(), **x64},
        "status_of_this_evidence": (
            "CHARACTERIZATION, not validation. The paper states no conventional "
            "reference exists for this system; that is the point of Fig 5c. "
            "Nothing here is scored against an oracle, and no number below is a "
            "pass threshold."
        ),
        "optical_system": {
            "prescription_fingerprint": spec.fingerprint(),
            "surface_positions_m": surface_positions_m(lens),
            "effective_focal_length_mm": float(lens.paraxial.f2()),
            "material_note": (
                "N-BK7 entered as IdealMaterialSpec(1.5131), the index the issue "
                "specifies, rather than the catalog glass. Optiland resolves a bare "
                "glass name by substring filter and Levenshtein ranking, and this "
                "reproduces a stated index, not a dispersion curve."
            ),
            "doe_plane_surface_note": (
                "surface 1 is the DOE plane itself, a plane in air carrying no "
                "power. It exists so the first traced surface coincides with the "
                "launch plane to within a nanometre, which trace_ray_batch requires."
            ),
        },
        "optiland_execution": execution.as_dict(),
        "configuration": {
            "wavelength_m": WAVELENGTH_M,
            "doe_grid": [GRID_N, GRID_N],
            "doe_pitch_m": PITCH_M,
            "gap_doe_lens_mm": GAP_DOE_LENS_MM,
            "sensor_z_m": SENSOR_Z_M,
            "sensor_grid": [sensor_px, sensor_px],
            "sensor_pitch_m": sensor_pitch_m,
            "sensor_extent_mm": sensor_px * sensor_pitch_m * 1e3,
            "rays_per_chunk": args.rays_per_chunk,
            "sensor_note": (
                "The paper does not state a sensor grid for this system, so this is "
                "a declared choice rather than a transcription. It is bounded from "
                "below by the image extent -- the mask's highest spatial frequency "
                "diffracts to about +/- lambda f / (2 pitch) = +/- 2.8 mm -- and "
                "from above by our O(rays x pixels) reconstruction, where every "
                "extra pixel costs every ray."
            ),
            "seeds": seeds,
            "backend_requested": args.backend,
            "reconstruction": args.reconstruction,
            "kspace_oversample": (
                args.kspace_oversample if args.reconstruction != "ramp_sum" else None
            ),
        },
        "conventions": {
            "origin_rule": "coordinate zero at index n // 2 (upstream uses (n-1)/2)",
            "phase_orientation": (
                "unflipped; the notebook's flip compensates DeepLens's Ray.flip_xy"
            ),
            "accumulation": "coherent field, per SI eq S5",
            "reconstruction_normalization": "one_over_n, applied once at finalize",
            "opl_convention": (
                "the patch step resets OPL to zero at the DOE plane; Optiland's "
                "traced `opd` is then carried forward by trace_ray_batch. Hazard "
                "H1 applies: Optiland's opd sign and reference plane are recorded "
                "as unverified (M1, CHE-13/CHE-17), so the ABSOLUTE phase reference "
                "at the sensor is not certified. Every comparison below is between "
                "two runs through the same path, where a shared offset cancels."
            ),
            "clipped_ray_policy": (
                "a clipped ray keeps its place in N_total and contributes zero "
                "amplitude; removing it would rescale the field by the survival "
                "fraction."
            ),
        },
        "paper_table_s2": TABLE_S2,
        "routes": {},
    }

    fields: dict[str, list[np.ndarray]] = {}
    for name in routes:
        settings = dict(preset[name])
        for key, value in (
            ("patch_count", args.patch_count),
            ("secondary_count", args.secondary_count),
            ("batches", args.batches),
        ):
            if value is not None and not (key == "patch_count" and settings[key] is None):
                settings[key] = value
        per_seed = []
        fields[name] = []
        for seed in seeds:
            run = run_route(
                doe,
                lens,
                sensor_shape=(sensor_px, sensor_px),
                sensor_pitch_m=sensor_pitch_m,
                patch_px=settings["patch_px"] or (GRID_N - 1),
                pad_factor=settings["pad_factor"],
                pad_width=settings["pad_width"],
                patch_count=settings["patch_count"],
                route=settings["route"],
                secondary_count=settings["secondary_count"],
                batches=settings["batches"],
                seed=seed,
                backend=args.backend,
                precision=settings["precision"],
                secondary_chunks=_secondary_chunks(settings, args.rays_per_chunk),
                reconstruction_route=args.reconstruction,
                kspace_oversample=args.kspace_oversample,
            )
            fields[name].append(run.pop("field"))
            run["seed"] = seed
            per_seed.append(run)
            print(
                f"  {name} seed {seed}: {run['total_rays']:,} rays  "
                f"{run['wall_clock_s']:.2f} s  "
                f"empty {run['energy']['launched_with_zero_amplitude']:,}  "
                f"clipped {run['energy']['clipped_with_power']:,}"
            )
        record["routes"][name] = {
            "requested": settings,
            "runs": per_seed,
            "seed_reproducibility": _seed_reproducibility(fields[name], seeds),
        }

    if len(routes) >= 2 and all(fields[r] for r in routes):
        a, b = fields[routes[0]][0], fields[routes[1]][0]
        record["route_agreement"] = {
            "note": (
                "SI S2 relation (2) with a refractive element downstream. The only "
                "cross-check this system has -- there is no external reference -- "
                "so it is reported as agreement between two of our own routes, not "
                "as a validation of either."
            ),
            "routes": routes[:2],
            "ncc_intensity": ncc(np.abs(a) ** 2, np.abs(b) ** 2),
            "mse_intensity_unit_sum": mse_unit_sum(np.abs(a) ** 2, np.abs(b) ** 2),
        }
        print(f"  {routes[0]} vs {routes[1]}: NCC {record['route_agreement']['ncc_intensity']:.6f}")

    name = args.output_name or f"demo3_{args.preset}_{args.backend}"
    path = write_record(name, record)
    fields_path = RECORDS / f"{name}_fields.npz"
    RECORDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fields_path,
        **{
            f"{route}_seed{seed}": fields[route][index]
            for route in routes
            for index, seed in enumerate(seeds)
        },
    )
    print(f"wrote {path}")
    print(f"wrote {fields_path}")
    return 0


def _write_agreement(names: list[str], output_name: str) -> int:
    """Cross-route agreement from two separately-run routes.

    Split out because a converged run of each route is ~150 s and six of them do
    not fit in one command. Splitting the *runs* is free -- they share no state
    -- and splitting the *comparison* out of them is what keeps the ray budget
    at a level where the comparison says anything.
    """
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        with np.load(RECORDS / f"{name}_fields.npz") as data:
            loaded[name] = {key: data[key] for key in data.files}
    keys = [sorted(v) for v in loaded.values()]
    first = loaded[names[0]][keys[0][0]]
    second = loaded[names[1]][keys[1][0]]
    record = {
        "probe": "demo3_route_agreement",
        "issue": "CHE-96",
        "environment": environment(),
        "sources": names,
        "compared": [keys[0][0], keys[1][0]],
        "note": (
            "SI S2 relation (2) with a refractive element downstream. The only "
            "cross-check this system has -- the paper states no conventional "
            "reference exists -- so this is agreement between two of our own "
            "routes, not a validation of either."
        ),
        "ncc_intensity": ncc(np.abs(first) ** 2, np.abs(second) ** 2),
        "mse_intensity_unit_sum": mse_unit_sum(np.abs(first) ** 2, np.abs(second) ** 2),
        "noise_limited_agreement": _noise_limited_agreement(loaded, names, keys),
        "cross_seed_matrix": [
            {
                "left": a,
                "right": b,
                "ncc_intensity": ncc(
                    np.abs(loaded[names[0]][a]) ** 2, np.abs(loaded[names[1]][b]) ** 2
                ),
            }
            for a in keys[0]
            for b in keys[1]
        ],
    }
    path = write_record(output_name, record)
    print(f"wrote {path}  route NCC {record['ncc_intensity']:.6f}")
    return 0


def _secondary_chunks(settings: dict[str, Any], rays_per_chunk: float) -> int:
    """How many pieces one patch's secondary draw is split into.

    Derived from the device budget rather than asked for, because the number
    that matters is rays *per chunk* and the caller thinks in total rays.
    """
    secondary = settings.get("secondary_count")
    if not secondary:
        return 1
    patches = settings.get("patch_count") or 1
    groups = max(1, min(int(settings.get("batches") or 1), patches))
    rays_per_group = (patches / groups) * secondary
    return max(1, int(np.ceil(rays_per_group / rays_per_chunk)))


def _noise_limited_agreement(loaded, names, keys) -> dict[str, Any]:
    """Do the two routes agree to within their own Monte Carlo noise?

    Neither route is individually converged at a budget that fits here, so
    "cross-route NCC = 0.014" on its own says nothing -- it could be two
    estimators of the same field, or two estimators of different fields, and a
    reader cannot tell which.

    The statistic that does distinguish them: if A and B estimate the SAME
    signal, each with independent additive noise, then

        NCC(A, B) ~= sqrt( NCC(A, A') * NCC(B, B') )

    where A' and A are independent realizations of the same route. Each factor
    is that route's own signal fraction, so the geometric mean is the agreement
    two unbiased estimators of one field are *entitled* to at this budget. A
    measured cross-route NCC at that value is evidence of agreement; one well
    below it is evidence of a systematic difference, and neither conclusion is
    available from the raw number.
    """
    self_ncc: dict[str, float] = {}
    for name, own_keys in zip(names, keys, strict=True):
        arrays = [np.abs(loaded[name][k]) ** 2 for k in own_keys]
        pairs = [
            ncc(arrays[i], arrays[j])
            for i in range(len(arrays))
            for j in range(i + 1, len(arrays))
        ]
        self_ncc[name] = float(np.mean(pairs)) if pairs else float("nan")

    cross = [
        ncc(np.abs(loaded[names[0]][a]) ** 2, np.abs(loaded[names[1]][b]) ** 2)
        for a in keys[0]
        for b in keys[1]
    ]
    measured = float(np.mean(cross))
    predicted = float(np.sqrt(self_ncc[names[0]] * self_ncc[names[1]]))
    return {
        "mean_self_ncc": self_ncc,
        "mean_cross_route_ncc": measured,
        "predicted_if_same_field": predicted,
        "ratio_measured_over_predicted": measured / predicted if predicted else None,
        "reading": (
            "a ratio near 1 means the two routes agree to within the Monte Carlo "
            "noise each carries at this budget -- which is the strongest statement "
            "available without converging either. A ratio well below 1 would be a "
            "systematic difference between the routes and would invalidate the "
            "SI S2 relation for this system."
        ),
    }


def _seed_reproducibility(fields: list[np.ndarray], seeds: list[int]) -> dict[str, Any]:
    """Is the speckle physical interference or an undersampling artifact?

    SI S6 / Figure S3's methodology: a speckle pattern that reproduces across
    independent realizations is structure in the field; one that does not is
    the Figure S4 artifact. Pairwise NCC between realizations is the statistic,
    and the per-pixel relative spread of the ensemble is reported next to it
    because a high NCC on a field dominated by one bright lobe can hide a
    completely unreproducible background.
    """
    if len(fields) < 2:
        return {
            "status": "not_measured",
            "reason": f"{len(fields)} realization(s); at least 3 are required by AC 4",
        }
    intensities = [np.abs(f) ** 2 for f in fields]
    pairs = [
        {
            "seeds": [seeds[i], seeds[j]],
            "ncc_intensity": ncc(intensities[i], intensities[j]),
            "mse_intensity_unit_sum": mse_unit_sum(intensities[i], intensities[j]),
        }
        for i in range(len(fields))
        for j in range(i + 1, len(fields))
    ]
    stack = np.stack(intensities)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    bright = mean > 0.05 * mean.max()
    return {
        "realizations": len(fields),
        "meets_ac4_minimum_of_three": len(fields) >= 3,
        "pairwise": pairs,
        "worst_pairwise_ncc": min(p["ncc_intensity"] for p in pairs),
        "relative_spread_over_bright_pixels": {
            "note": (
                "std / mean per pixel, averaged over pixels above 5% of the peak. "
                "Reported next to the NCC because a field dominated by one bright "
                "lobe can score a high NCC while its background is noise."
            ),
            "bright_pixel_count": int(bright.sum()),
            "mean_relative_spread": float(np.mean(std[bright] / mean[bright]))
            if bright.any()
            else None,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
