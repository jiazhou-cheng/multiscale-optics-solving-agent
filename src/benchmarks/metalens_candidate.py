"""One ``(P, S, density, seed, chunk)`` candidate, in its own process (CHE-70).

Run as a module, never imported by the controller's own process:

    python -m benchmarks.metalens_candidate \
        --config METALENS-AIR-100 --launch-count 16 --samples-per-launch 1024 \
        --seed 1 --chunk-size 65536 --device cuda --precision fp32 \
        --result <path>.json

Why a fresh process per candidate (Phase 15)
--------------------------------------------
CUDA memory is returned to the driver when the process exits and not reliably
before; allocator fragmentation accumulates across differently-shaped
allocations; and a Python reference held by an exception traceback can keep a
chunk alive past the loop that created it. One process per candidate makes all
three impossible rather than merely unlikely, and it means a candidate that dies
cannot poison the next one's numbers.

The result file is written to ``<path>.tmp`` and renamed, so a reader either sees
a complete result or no file at all.

Status vocabulary
-----------------
Exactly the Phase 30 set, and the process exits 0 whenever it managed to *write*
one -- a recorded ``FAIL_GPU_MEMORY`` is a successful measurement of a limit, not
a crash. A nonzero exit means no result was written and the parent must record
the candidate as lost.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# JAX preallocates 75% of the device by default, which would make every GPU
# measurement in this benchmark meaningless: `mem_get_info` would report ~38 GB
# taken before a single ray existed, the per-chunk memory-scaling study (Phase 32)
# would be flat by construction, and on a shared eight-GPU server it would hold
# memory no candidate is using. Set before any JAX import -- the flag is read once
# when the backend initialises -- and only when the caller has not chosen already.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from core.arrays import (
    device_of,
    dtype_of,
    namespace_of,
    numpy_dtype,
    to_namespace,
    xp_for,
)
from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES
from core.precision import (
    ArrayNamespace,
    DeviceKind,
    DevicePlacement,
    Precision,
)
from core.resources import (
    MemoryWatchdog,
    cuda_memory_snapshot,
    cuda_reset_peak_stats,
    host_memory_snapshot,
)

__all__ = ["STATUSES", "CandidateRequest", "main", "run_candidate"]

STATUSES = (
    "PASS_RUN",
    "SKIPPED_MEMORY_GUARD",
    "FAIL_GPU_MEMORY",
    "FAIL_HOST_MEMORY_PRESSURE",
    "FAIL_DEVICE_FALLBACK",
    "FAIL_NUMERICAL",
    "FAIL_ENVIRONMENT_NO_CUDA",
)

#: Grazing-mode floor for every CHE-70 run. Derived in
#: ``couplers.streaming.grazing_floor_for_phase_budget`` from the float32
#: representation of a 50 um propagation at a 0.01 rad budget (7.49e-3) and
#: rounded up, so a single frozen value serves both precisions and the two are
#: comparable over the same mode set.
DIRECTION_COSINE_FLOOR = 1.0e-2


@dataclass(frozen=True)
class CandidateRequest:
    """One point of the sweep. Everything that changes the numbers is here."""

    config: str
    launch_count: int
    samples_per_launch: int
    seed: int
    density: str
    chunk_size: int
    device: str
    precision: str
    direction_cosine_floor: float = DIRECTION_COSINE_FLOOR
    projection: str = "asm_consistent"
    host_reserve_bytes: int | None = None
    process_rss_budget_bytes: int | None = None
    label: str = ""

    @property
    def total_rays(self) -> int:
        return self.launch_count * self.samples_per_launch

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "launch_count_P": self.launch_count,
            "samples_per_launch_S": self.samples_per_launch,
            "total_rays": self.total_rays,
            "seed": self.seed,
            "density": self.density,
            "chunk_size": self.chunk_size,
            "device": self.device,
            "precision": self.precision,
            "direction_cosine_floor": self.direction_cosine_floor,
            "projection": self.projection,
            "label": self.label,
        }


def _device(request: CandidateRequest) -> DevicePlacement:
    if request.device.startswith("cuda"):
        _, _, index = request.device.partition(":")
        # `index is None` means "this kind of device, ordinal unspecified", which
        # is what a request means -- and it is all Optiland can express anyway.
        return DevicePlacement(DeviceKind.CUDA, int(index) if index else None)
    return DevicePlacement(DeviceKind.CPU, None)


def _namespace(device: DevicePlacement) -> ArrayNamespace:
    """NumPy on the host, JAX on a device.

    Not a preference. NumPy cannot hold device memory, so the CUDA path is
    JAX-only for both couplers -- which is exactly what
    ``C_RAY_TO_WAVE_CAPABILITIES.device_namespaces`` declares.
    """
    return ArrayNamespace.JAX if device.kind is DeviceKind.CUDA else ArrayNamespace.NUMPY


def _environment(device: DevicePlacement) -> dict[str, Any]:
    """Phase 0's manifest fields, every one of them read rather than configured."""
    import optiland

    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "optiland": optiland.__version__,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["torch_cuda_runtime"] = torch.version.cuda
        record["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            record["cuda_device_name"] = torch.cuda.get_device_name(device.index or 0)
            record["cuda_capability"] = list(torch.cuda.get_device_capability(device.index or 0))
            record["cuda_device_count"] = int(torch.cuda.device_count())
            record["cuda_driver_version"] = _driver_version()
    except ImportError:  # pragma: no cover
        record["torch"] = None
    if device.kind is DeviceKind.CUDA:
        import jax

        record["jax"] = jax.__version__
        record["jax_backend"] = jax.default_backend()
        record["jax_devices"] = [str(dev) for dev in jax.devices()]
        record["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    return record


def _driver_version() -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:  # pragma: no cover - diagnostics only
        return None


def run_candidate(request: CandidateRequest) -> dict[str, Any]:
    """Execute one candidate and return its complete record.

    Never raises for an expected failure -- a guard breach, an OOM or a device
    fallback all come back as a record with the corresponding status, because the
    sweep needs to *read* them to decide what to do next.
    """
    from adapters.optiland_builder import build_optiland_system
    from adapters.optiland_ray_trace import (
        configure_optiland_execution,
        plan_trace_bridges,
        trace_ray_batch,
    )
    from couplers.coherent_batch import CoherentRayBatch
    from couplers.contracts import PSF, ReferencePlane
    from couplers.ray_to_wave import Projection
    from couplers.streaming import (
        PositionalAngularSampler,
        StreamingReconstruction,
        band_limit_spectrum,
        build_chunk_bundle,
        chunk_plan,
        nested_aperture_launch_positions,
    )
    from couplers.wave_to_ray import SamplingDensity, decompose
    from evaluation.metalens import (
        CONFIGURATIONS,
        compare_psfs,
        metalens_field,
        optical_system_spec,
        reference_field,
    )

    started = time.time()
    config = CONFIGURATIONS[request.config]
    device = _device(request)
    namespace = _namespace(device)
    precision = Precision.FP32 if request.precision == "fp32" else Precision.FP64
    density_kind = (
        SamplingDensity.MAGNITUDE if request.density == "p_mag" else SamplingDensity.UNIFORM
    )
    projection = (
        Projection.ASM_CONSISTENT
        if request.projection == "asm_consistent"
        else Projection.SENSOR_OBLIQUITY
    )

    record: dict[str, Any] = {
        "request": request.as_dict(),
        "configuration": config.as_dict(),
        "started_unix": started,
        "status": "FAIL_NUMERICAL",
    }
    watchdog = MemoryWatchdog(
        host_reserve_bytes=request.host_reserve_bytes,
        process_rss_budget_bytes=request.process_rss_budget_bytes,
    )
    record["host_memory_before"] = watchdog.baseline.as_dict()

    if device.kind is DeviceKind.CUDA:
        try:
            import torch

            if not torch.cuda.is_available():
                record["status"] = "FAIL_ENVIRONMENT_NO_CUDA"
                record["failure"] = (
                    "torch.cuda.is_available() is False; there is deliberately no "
                    "CPU fallback"
                )
                return record
        except ImportError:  # pragma: no cover
            record["status"] = "FAIL_ENVIRONMENT_NO_CUDA"
            record["failure"] = "torch is not importable"
            return record

    record["environment"] = _environment(device)
    if device.kind is DeviceKind.CUDA and record["environment"].get("jax_backend") != "gpu":
        record["status"] = "FAIL_DEVICE_FALLBACK"
        record["failure"] = (
            f"jax.default_backend() is {record['environment'].get('jax_backend')!r} on a "
            "CUDA request. A process-global JAX platform pin does this, and it is "
            "irreversible once the backend is built."
        )
        return record

    watchdog.start()
    try:
        execution = configure_optiland_execution(
            device=device, precision=precision, enable_grad=False
        )
        record["optiland_execution"] = execution.as_dict()
        # Phase 13: the production sweep is a forward validation. Autograd off,
        # and read back from the solver rather than assumed from the request.
        if execution.grad_enabled:
            raise RuntimeError(
                "Optiland grad mode is enabled; the convergence sweep must not "
                "record computational graphs"
            )
        cuda_reset_peak_stats()
        record["gpu_memory_before"] = cuda_memory_snapshot().as_dict()
        # --- the one source field, cast once, explicitly -------------------
        source = metalens_field(config)
        complex_dtype = precision.complex_dtype
        assert complex_dtype is not None  # FP32/FP64 both have one
        u = to_namespace(
            np.asarray(source.u, dtype=numpy_dtype(complex_dtype)),
            namespace=namespace,
            device=device if namespace.can_leave_host else None,
            dtype=complex_dtype,
        )
        field_in = type(source)(
            u=u,
            sample_pitch_m=source.sample_pitch_m,
            wavelength_m=source.wavelength_m,
            reference_plane=source.reference_plane,
            frame=source.frame,
            normalization=source.normalization,
            provenance={
                **source.provenance,
                "precision_cast": {
                    "from": "complex128 host numpy",
                    "requested": str(complex_dtype),
                    "actual": str(dtype_of(u)),
                    "device": str(device_of(u)),
                    "namespace": str(namespace_of(u)),
                },
            },
        )
        record["input_field_residency"] = field_in.provenance["precision_cast"]

        spectrum = decompose(field_in)
        limited, band = band_limit_spectrum(
            spectrum,
            direction_cosine_floor=request.direction_cosine_floor,
            max_optical_path_m=config.sensor_distance_m,
            precision=str(precision),
            phase_budget_rad=1.0e-2,
        )
        record["band_limit"] = band.as_dict()
        record["spectrum"] = limited.as_dict()

        sampler, density = PositionalAngularSampler.build(
            limited,
            density_kind=density_kind,
            seed=request.seed,
            samples_per_launch=request.samples_per_launch,
        )
        record["sampler"] = sampler.as_dict()
        launch = nested_aperture_launch_positions(
            request.launch_count, aperture_radius_m=config.aperture_radius_m
        )
        record["launch_geometry"] = launch.as_dict()

        lens = build_optiland_system(optical_system_spec(config))
        sensor = ReferencePlane(name="sensor", z_m=config.sensor_distance_m)
        plan = chunk_plan(
            launch_count=request.launch_count,
            samples_per_launch=request.samples_per_launch,
            chunk_size=request.chunk_size,
        )
        record["chunking"] = {
            "requested_chunk_size": request.chunk_size,
            "chunk_count": len(plan),
            "work_items": sum(len(items) for items in plan),
        }

        reconstruction = StreamingReconstruction(
            grid_shape=config.grid_shape,
            sample_pitch_m=config.pitch_pair,
            plane=sensor,
            wavelength_m=config.wavelength_m,
            namespace=namespace,
            complex_dtype=complex_dtype,
            total_rays=request.total_rays,
            projection=projection,
        )

        trace_plans = None
        residency_log: dict[str, Any] = {}
        chunk_gpu_peaks: list[int] = []
        for index, items in enumerate(plan):
            if watchdog.verdict.breached:
                record["status"] = "FAIL_HOST_MEMORY_PRESSURE"
                record["failure"] = watchdog.verdict.detail
                record["aborted_after_chunks"] = index
                break
            bundle, ray_ids = build_chunk_bundle(limited, density, sampler, items, launch)
            batch = CoherentRayBatch(
                bundle=bundle,
                ray_id=ray_ids,
                valid=xp_for(namespace).ones(bundle.count, dtype=bool),
            )
            if trace_plans is None:
                trace_plans = plan_trace_bridges(
                    batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=device
                )
                record["bridge_plans"] = trace_plans.as_dict()
                residency_log["wave_to_ray_output"] = batch.residency()
            traced, trace_diagnostics = trace_ray_batch(
                batch, lens, image_plane=sensor, plans=trace_plans
            )
            if index == 0:
                residency_log["optiland_output"] = trace_diagnostics["residency"]
                record["first_chunk_trace"] = {
                    key: value
                    for key, value in trace_diagnostics.items()
                    if key != "residency"
                }
            reconstruction.add_chunk(traced)
            if device.kind is DeviceKind.CUDA:
                chunk_gpu_peaks.append(
                    cuda_memory_snapshot(synchronize=False).peak_allocated_bytes or 0
                )
            # Release the chunk's references before the next allocation, so peak
            # memory is one chunk rather than two.
            del bundle, batch, traced, ray_ids
        else:
            result = reconstruction.finalize(
                provenance={
                    "band_limit": band.as_dict(),
                    "launch_geometry": launch.as_dict(),
                    "sampler": sampler.as_dict(),
                }
            )
            residency_log["sensor_field"] = result.residency
            record["streaming"] = result.as_dict()

            measured = PSF.from_complex_field(
                result.field,
                normalization="raw: |u|^2 in the field's own amplitude units",
                coherence_model="monochromatic, fully coherent",
            )
            intensity = np.asarray(_host(measured.intensity), dtype=np.float64)
            reference = reference_field(
                config, direction_cosine_floor=request.direction_cosine_floor
            )
            reference_intensity = np.abs(np.asarray(reference.u)) ** 2
            comparison = compare_psfs(
                intensity, reference_intensity, pitch=config.sample_pitch_m
            )
            record["metrics"] = comparison.as_dict()
            record["residency"] = residency_log
            record["psf_peak_index"] = [
                int(value) for value in np.unravel_index(int(intensity.argmax()), intensity.shape)
            ]
            fallback = _device_fallback(residency_log, device)
            record["status"] = "FAIL_DEVICE_FALLBACK" if fallback else "PASS_RUN"
            if fallback:
                record["failure"] = fallback
            record["psf_arrays"] = {
                "test": intensity,
                "reference": reference_intensity,
                "test_field": np.asarray(_host(result.field.u)),
                "reference_field": np.asarray(reference.u),
            }
        if device.kind is DeviceKind.CUDA and chunk_gpu_peaks:
            record["gpu_peak_after_first_chunk_bytes"] = chunk_gpu_peaks[0]
            record["gpu_peak_growth_bytes"] = chunk_gpu_peaks[-1] - chunk_gpu_peaks[0]
    except Exception as exc:
        record["status"] = _classify(exc)
        record["failure"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        watchdog.stop()
        record["memory"] = watchdog.report()
        record["gpu_memory_after"] = cuda_memory_snapshot().as_dict()
        record["host_memory_after"] = host_memory_snapshot().as_dict()
        record["wall_time_s"] = time.time() - started

    # A guard that tripped after the loop finished still means the run was not
    # clean, and Phase 17 makes any swap growth a hard failure.
    if record["status"] == "PASS_RUN" and watchdog.verdict.breached:
        record["status"] = "FAIL_HOST_MEMORY_PRESSURE"
        record["failure"] = watchdog.verdict.detail
    return record


def _host(array: Any) -> np.ndarray:
    """Bring a small final result to the host. The only permitted transfer here."""
    from core.arrays import to_host_numpy

    return to_host_numpy(array, reason="serializing the final 100x100 field / PSF")


def _classify(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc)
    if "OutOfMemoryError" in name or "CUDA out of memory" in text:
        return "FAIL_GPU_MEMORY"
    if "FAIL_ENVIRONMENT_NO_CUDA" in text:
        return "FAIL_ENVIRONMENT_NO_CUDA"
    if "MemoryError" in name:
        return "FAIL_HOST_MEMORY_PRESSURE"
    return "FAIL_NUMERICAL"


def _device_fallback(residency: dict[str, Any], device: DevicePlacement) -> str | None:
    """Phase 21: any scientific array off the requested device is a failure.

    Checked against every logged boundary, not just the last one -- a mid-pipeline
    host round trip that lands back on the device would otherwise pass.
    """
    if device.kind is not DeviceKind.CUDA:
        return None
    for boundary, groups in residency.items():
        # Only the scientific group. `bookkeeping` is host by design -- see
        # CoherentRayBatch.residency -- and checking it would report the
        # deliberate seeded-sampler transfer as a fallback.
        arrays = groups.get("scientific", groups) if isinstance(groups, dict) else {}
        for name, state in arrays.items():
            if not isinstance(state, dict):
                continue
            observed = state.get("device", "")
            if not observed.startswith("cuda"):
                return (
                    f"{boundary}.{name} landed on {observed!r} while cuda was "
                    "requested; a scientific array left the device"
                )
    return None


def _write_atomic(path: Path, record: dict[str, Any]) -> None:
    arrays = record.pop("psf_arrays", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        array_path = path.with_suffix(".npz")
        # `.tmp.npz`, not `.npz.tmp`: numpy appends `.npz` to any filename that
        # does not already end in it, so the obvious spelling writes a third name.
        temporary = array_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, array_path)
        record["arrays_path"] = str(array_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, default=_json_default))
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="METALENS-AIR-100")
    parser.add_argument("--launch-count", type=int, required=True)
    parser.add_argument("--samples-per-launch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--density", choices=["p_mag", "p_uni"], default="p_mag")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--projection", default="asm_consistent")
    parser.add_argument("--grazing-floor", type=float, default=DIRECTION_COSINE_FLOOR)
    parser.add_argument("--host-reserve-bytes", type=int, default=None)
    parser.add_argument("--process-rss-budget-bytes", type=int, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)

    request = CandidateRequest(
        config=args.config,
        launch_count=args.launch_count,
        samples_per_launch=args.samples_per_launch,
        seed=args.seed,
        density=args.density,
        chunk_size=args.chunk_size,
        device=args.device,
        precision=args.precision,
        projection=args.projection,
        direction_cosine_floor=args.grazing_floor,
        host_reserve_bytes=args.host_reserve_bytes,
        process_rss_budget_bytes=args.process_rss_budget_bytes,
        label=args.label,
    )
    record = run_candidate(request)
    _write_atomic(args.result, record)
    print(f"{record['status']}  {args.result}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
