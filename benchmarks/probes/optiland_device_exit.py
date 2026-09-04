"""CHE-245 (T1) probes: where an Optiland trace exit lands, and what that costs.

Three questions, and the second one is allowed to stop the ticket.

**P1 placement.** After a `device="cuda"` `SO_RAY_LAUNCH_TRACE`, what does
`numerics.arrays.array_state(bundle.positions_m)` actually report? A code reading
says host NumPy; this turns that into a measurement, and it keeps reporting after
the fix because the *pre-fix* behaviour stays expressible as an explicit
`exit_state=host_exit_state(...)` request. Both arms are rows, so the record
carries its own before and after rather than being a snapshot that goes stale
the moment the code it described changed.

**P2 benefit.** Optiland `cuda` against `cpu` across ray counts, and then the
first repo-owned node downstream across the same counts. Two curves because they
answer different questions, and only the second one justifies the change.
Measured: the trace itself **loses** on CUDA below about 1e5 rays -- 0.66x at 1e3
and 0.83x at 1e4 -- and above it wins only marginally, 1.10x and 1.06x with
sample ranges that straddle 1.0; no run taken has beaten 1.5x. A host exit
meanwhile costs the *downstream* graph 5x to 63x over the same range, because
every consumer reads `rays.xp`. So CHE-245 is worth doing for the graph after the
exit and not for the trace, which is the opposite of the framing it was written
with.

**P3 bridge.** The on-device torch/JAX DLPack round trip against a same-size
device-to-host copy. This is the input to the JAX-vs-torch exit decision CHE-248
(T4) owns, and it is what says whether preserving a device exit is cheap.

No `src/` change, no gate, no oracle. Every row is `BASELINE` -- a recorded value
with nothing to have agreed with -- except P1's, which compares a requested exit
state against the observed one and is therefore a real `PASS`/`FAIL`.

Run, one device, on a host whose GPUs were checked first:

    MOA_GPUS=device=6 ./run.sh --gpu python -m benchmarks.probes.optiland_device_exit

Without a CUDA device attached every CUDA row is `BLOCKED-no-backend` and the
host rows still land, so the driver is runnable on the CPU image and says what it
could not reach.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from backends.optiland import trace
from backends.optiland.rays import hexapolar_ray_count, host_exit_state, trace_exit_state
from benchmarks.probes.record import RECORDS, probe_provenance
from benchmarks.verification.optiland_prescription import extract_setup
from benchmarks.verification.record import Row, finish
from numerics import ArrayNamespace, DevicePlacement, Precision, array_state
from numerics.arrays import matmul_precision_kwargs
from problems import SourceSpec

#: The system every row is measured on, and why it is a sample rather than a
#: fixture. `tests/fixtures/systems.py` holds this project's frozen
#: prescriptions, and `benchmarks/` does not import `tests/`; transcribing one
#: here would be a second copy of a prescription that has to stay in step with
#: the first. `optiland.samples` plus
#: `benchmarks/verification/optiland_prescription.extract_setup` reads a
#: prescription out of the pinned install instead, which is the route
#: `benchmarks/verification/ray_tier1.py` already established. Nothing measured
#: here is a property of the lens: a placement is a property of the buffer and a
#: timing is a property of the ray count.
SAMPLE = "CookeTriplet"

#: On axis at the reference wavelength, object at infinity. A field angle would
#: change the traced geometry and change no placement and no timing.
LIGHT = SourceSpec(wavelength_um=0.55, field_angle_deg=(0.0, 0.0), object_distance_mm=None)

#: Hexapolar ring counts whose ray counts bracket 1e3 / 1e4 / 1e5 / 1e6, the
#: decade points CHE-245 asks P2 for. `hexapolar_ray_count(n) = 1 + 3n(n+1)`, so
#: the count is a property of the fan and cannot be requested directly; the ring
#: count that lands nearest each decade is what is recorded.
RING_COUNTS: tuple[int, ...] = (18, 57, 182, 577)

#: The reconstruction grid the downstream curve is measured on. 256 x 256 is the
#: scale `tests/parity/test_ray_to_scalar_parity.py` and the wave benchmarks
#: work at, and the direct ramp sum is `O(N_rays x ny x nx)`, so the grid is half
#: of what the curve is a curve in.
GRID = (256, 256)
PITCH = (2.0e-6, 2.0e-6)

#: How many timed repetitions each cell contributes. Five after one warm-up: the
#: warm-up is discarded because a first call pays JIT compilation and a CUDA
#: context, neither of which is the steady-state cost being asked about.
REPETITIONS = 5


def _timing(call: Any) -> dict[str, Any]:
    """Median seconds **and the spread**, because on this host the spread matters.

    A bare median would be the wrong thing to record here and then quote
    elsewhere. This is a shared GPU server: re-running P2 an hour apart moved the
    1e5-ray trace ratio from 1.07x to 1.49x with no code change, which is
    contention and not a property of either device. So every sample is kept and
    the record carries `min`/`max` alongside the median, and anything citing
    these numbers should cite the range rather than three significant figures of
    the median.
    """
    call()
    samples = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return {
        "median_s": float(np.median(samples)),
        "min_s": float(np.min(samples)),
        "max_s": float(np.max(samples)),
        "samples_s": [float(sample) for sample in samples],
    }


def _cuda_unavailable_reason() -> str | None:
    """Why no CUDA row can be measured here, or `None` if they can.

    Both frameworks are consulted because the container can be half-enabled:
    torch drives the trace and JAX receives the exit, so a row needs both.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        return f"torch is not importable ({exc})"
    if torch.version.cuda is None:
        return f"torch is a CPU-only build ({torch.__version__})"
    if not torch.cuda.is_available():
        return "no CUDA device is attached to this container (run ./run.sh --gpu)"
    try:
        import jax
    except ImportError as exc:  # pragma: no cover - jax is a hard dependency
        return f"jax is not importable ({exc})"
    if not any(device.platform == "gpu" for device in jax.devices()):
        return f"jax reports no gpu device (backend={jax.default_backend()!r})"
    return None


# ---------------------------------------------------------------------------
# P1 -- placement
# ---------------------------------------------------------------------------


def probe_placement(setup: Any) -> list[Row]:
    """Where the launch-trace exit puts the bundle, requested against observed.

    Four cells: `{cpu-fp64, cuda-fp32} x {the default exit, an explicit host
    exit}`. The explicit-host arm is the pre-CHE-245 behaviour, kept measurable
    on purpose -- a "before" row that only exists in a record written before the
    fix goes stale the instant the fix lands, and this project's records are
    re-run rather than preserved.
    """
    rows: list[Row] = []
    blocked = _cuda_unavailable_reason()
    for device_name, precision_name in (("cpu", "fp64"), ("cuda", "fp32")):
        device = DevicePlacement.parse(device_name)
        precision = Precision.parse(precision_name)
        default = trace_exit_state(device=device, precision=precision)
        for arm, requested in (
            ("default", default),
            ("explicit_host", host_exit_state(precision.real_dtype)),
        ):
            case = f"p1-placement-{device_name}-{precision_name}-{arm}"
            configuration = {
                "system": SAMPLE,
                "device": device_name,
                "precision": precision_name,
                "num_rings": 8,
                "reference_surface": "exit_pupil",
                "exit_state_arm": arm,
            }
            if device_name == "cuda" and blocked is not None:
                rows.append(
                    Row(
                        case=case,
                        configuration=configuration,
                        descriptor="SO_RAY_LAUNCH_TRACE",
                        status="BLOCKED-no-backend",
                        measured={},
                        expected={},
                        deltas={},
                        worst_relative_delta=0.0,
                        runtime_s=0.0,
                        note=f"not measured: {blocked}",
                    )
                )
                continue
            # The default arm is what `solver.trace` passes; the explicit arm has
            # to reach `to_ray_bundle` directly, because `trace` deliberately
            # takes no exit-state argument -- the device and the precision are
            # what a caller names, and the exit follows from them.
            started = time.perf_counter()
            bundle = _trace_with_exit(setup, device=device, precision=precision, arm=arm)
            runtime = time.perf_counter() - started
            observed = array_state(bundle.positions_m)
            rows.append(
                Row(
                    case=case,
                    configuration=configuration,
                    descriptor="SO_RAY_LAUNCH_TRACE",
                    status="PASS" if _states_agree(requested, observed) else "FAIL",
                    measured={
                        "observed_exit_state": str(observed),
                        "observed_namespace": observed.namespace.value,
                        "observed_device": str(observed.device),
                        "observed_dtype": observed.dtype.value,
                        "ray_count": bundle.count,
                    },
                    expected={"requested_exit_state": str(requested)},
                    deltas={},
                    worst_relative_delta=0.0,
                    runtime_s=runtime,
                    note=(
                        "Read off the emitted buffer with numerics.arrays.array_state, never "
                        "from the argument that requested it. Device *kind* is compared and "
                        "the ordinal is reported: a request for 'cuda' carries none and its "
                        "observation is 'cuda:0'. The explicit_host arm is what every CUDA "
                        "trace returned before CHE-245 -- host NumPy, which pinned every "
                        "downstream repo-owned node to the host because they all read "
                        "rays.xp. It is a row rather than a deleted 'before' record."
                    ),
                )
            )
    return rows


def _trace_with_exit(
    setup: Any, *, device: DevicePlacement, precision: Precision, arm: str
) -> Any:
    """One trace, on the default exit or forced back onto the host.

    The explicit-host arm rebuilds what `solver.trace` does around
    `to_ray_bundle` rather than adding an argument to `trace`, because the
    argument does not belong there: a caller of the public entry point names a
    device and a precision, and where the rays are delivered follows from those.
    """
    from backends.optiland.launch import launch
    from backends.optiland.rays import to_ray_bundle
    from backends.optiland.solver import build_lens, configure_execution
    from numerics.precision import ArrayState

    sampling = {"num_rings": 8, "reference_surface": "exit_pupil"}
    if arm == "default":
        return trace(setup, LIGHT, sampling=sampling, execution={
            "device": str(device), "precision": str(precision)
        })
    namespace = ArrayNamespace.TORCH if precision is Precision.FP32 else ArrayNamespace.NUMPY
    configure_execution(device=device, precision=precision, namespace=namespace)
    lens = build_lens(setup, LIGHT)
    _, declaration = launch(lens, LIGHT, num_rings=8, aiming="paraxial")
    traced = lens.trace(
        Hx=declaration["field"][0],
        Hy=declaration["field"][1],
        wavelength=LIGHT.wavelength_um,
        num_rays=8,
    )
    exit_state: ArrayState = host_exit_state(precision.real_dtype)
    bundle, _ = to_ray_bundle(
        lens,
        traced,
        launch=declaration,
        reference_surface="exit_pupil",
        exit_state=exit_state,
    )
    return bundle


def _states_agree(requested: Any, observed: Any) -> bool:
    """Namespace and dtype exactly, device by *kind*. See P1's note."""
    return (
        observed.namespace is requested.namespace
        and observed.dtype is requested.dtype
        and observed.device.kind is requested.device.kind
    )


# ---------------------------------------------------------------------------
# P2 -- benefit
# ---------------------------------------------------------------------------


def probe_benefit(setup: Any) -> list[Row]:
    """Two curves: the Optiland trace, and the first repo-owned node after it.

    CHE-245 makes P2 able to stop the ticket -- "if CUDA does not win at the ray
    counts actually in use, say so plainly and stop". The trace curve is what
    that sentence is about and the downstream curve is what the change is
    actually for, so both are recorded and the report reads them together.
    """
    rows: list[Row] = []
    blocked = _cuda_unavailable_reason()
    for rings in RING_COUNTS:
        ray_count = hexapolar_ray_count(rings)
        configuration = {
            "system": SAMPLE,
            "num_rings": rings,
            "ray_count": ray_count,
            "precision": "fp32",
            "reference_surface": "image_surface",
            "repetitions": REPETITIONS,
            "statistic": "median of 5 after one discarded warm-up; min/max also recorded",
        }
        if blocked is not None:
            rows.append(
                Row(
                    case=f"p2-trace-{ray_count}",
                    configuration=configuration,
                    descriptor="SO_RAY_LAUNCH_TRACE",
                    status="BLOCKED-no-backend",
                    measured={},
                    expected={},
                    deltas={},
                    worst_relative_delta=0.0,
                    runtime_s=0.0,
                    note=f"not measured: {blocked}",
                )
            )
            continue
        sampling = {"num_rings": rings, "reference_surface": "image_surface"}
        timings = {}
        for device_name in ("cpu", "cuda"):
            execution = {"device": device_name, "precision": "fp32"}
            timings[device_name] = _timing(
                lambda d=execution, s=sampling: trace(setup, LIGHT, sampling=s, execution=d)
            )
        rows.append(
            Row(
                case=f"p2-trace-{ray_count}",
                configuration=configuration,
                descriptor="SO_RAY_LAUNCH_TRACE",
                status="BASELINE",
                measured={
                    "cpu_fp32": timings["cpu"],
                    "cuda_fp32": timings["cuda"],
                    "cuda_speedup_median": (
                        timings["cpu"]["median_s"] / timings["cuda"]["median_s"]
                    ),
                    "cuda_speedup_range": [
                        timings["cpu"]["min_s"] / timings["cuda"]["max_s"],
                        timings["cpu"]["max_s"] / timings["cuda"]["min_s"],
                    ],
                },
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=timings["cpu"]["median_s"] + timings["cuda"]["median_s"],
                note=(
                    "End-to-end trace, both legs at fp32 so a device change is not hiding "
                    "behind a precision change -- fp32 is torch on either device "
                    "(solver._resolve_namespace), so this compares torch-cpu against "
                    "torch-cuda. End-to-end, including build_lens, launch and the exit, "
                    "which is what a caller pays. Fixed and scaling cost are NOT "
                    "decomposed here, so this row says CUDA loses at the smallest counts "
                    "and does not say which component is responsible. `cuda_speedup_range` "
                    "is the ratio of the extreme samples and is wide because this is a "
                    "shared host: quote the range, not the median. No oracle -- a runtime "
                    "is not a correctness claim about either leg."
                ),
            )
        )
        rows.append(_downstream_row(ray_count))
    return rows


def _downstream_row(ray_count: int) -> Row:
    """The ramp sum, host against device, at the same ray count.

    The kernel is `couplers/ray_to_scalar.py`'s separable contraction --
    `einsum("n,ny,nx->yx", ..., optimize=True, **matmul_precision_kwargs(...))`,
    the same one `tests/parity/test_ray_to_scalar_parity.py` gates for accuracy.
    Measured on synthetic factors rather than through `ray_to_scalar` itself
    because a full coupler call includes a grazing check, a measure resolution
    and a diagnostics build that are `O(N_rays)` and not what scales; the
    contraction is the `O(N_rays x ny x nx)` term the curve is about, and it is
    read out of the production module rather than reimplemented.
    """
    import jax.numpy as jnp

    ny, nx = GRID
    generator = np.random.default_rng(20260903)
    coefficient = generator.standard_normal(ray_count).astype(np.complex64)
    ramp_y = np.exp(1j * generator.standard_normal((ray_count, ny))).astype(np.complex64)
    ramp_x = np.exp(1j * generator.standard_normal((ray_count, nx))).astype(np.complex64)
    host_dot = matmul_precision_kwargs(ArrayNamespace.NUMPY)
    device_dot = matmul_precision_kwargs(ArrayNamespace.JAX)
    host = _timing(
        lambda: np.einsum(
            "n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True, **host_dot
        )
    )
    device_coefficient = jnp.asarray(coefficient)
    device_ramp_y = jnp.asarray(ramp_y)
    device_ramp_x = jnp.asarray(ramp_x)
    device = _timing(
        lambda: jnp.einsum(
            "n,ny,nx->yx",
            device_coefficient,
            device_ramp_y,
            device_ramp_x,
            optimize=True,
            **device_dot,
        ).block_until_ready()
    )
    return Row(
        case=f"p2-downstream-ramp-sum-{ray_count}",
        configuration={
            "kernel": 'einsum("n,ny,nx->yx", optimize=True) with matmul_precision_kwargs',
            "source": "couplers/ray_to_scalar.py, the DIRECT reconstruction",
            "ray_count": ray_count,
            "grid_shape": list(GRID),
            "dtype": "complex64",
            "seed": 20260903,
            "repetitions": REPETITIONS,
            "statistic": "median of 5 after one discarded warm-up; min/max also recorded",
        },
        descriptor="C_RAY_TO_SCALAR",
        status="BASELINE",
        measured={
            "numpy_cpu": host,
            "jax_cuda": device,
            "cuda_speedup_median": host["median_s"] / device["median_s"],
            "cuda_speedup_range": [
                host["min_s"] / device["max_s"],
                host["max_s"] / device["min_s"],
            ],
        },
        expected={},
        deltas={},
        worst_relative_delta=0.0,
        runtime_s=host["median_s"] + device["median_s"],
        note=(
            "What a host exit costs the graph, not what the trace costs. This is the "
            "measurement CHE-245's design rests on: the bundle's namespace decides where "
            "every repo-owned node downstream runs, because they read rays.xp. block_until_"
            "ready is called on the device leg so an asynchronous dispatch is not recorded "
            "as a completed contraction. No oracle; accuracy of this same kernel is gated "
            "separately in tests/parity/test_ray_to_scalar_parity.py."
        ),
    )


# ---------------------------------------------------------------------------
# P3 -- bridge
# ---------------------------------------------------------------------------


def probe_bridge() -> list[Row]:
    """A torch-CUDA buffer into JAX: over DLPack, against through the host.

    CHE-248 (T4) owns the JAX-vs-torch exit decision and this is its input. The
    sizes are the ray-column footprints of the P2 fan sizes -- `3 x ray_count`
    float32, one `(N, 3)` geometry column set -- so the numbers read against the
    ray counts rather than against abstract byte sizes.
    """
    blocked = _cuda_unavailable_reason()
    if blocked is not None:
        return [
            Row(
                case="p3-bridge",
                configuration={"sizes_float32": [3 * n for n in RING_COUNTS]},
                descriptor="numerics.arrays.to_namespace",
                status="BLOCKED-no-backend",
                measured={},
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=0.0,
                note=f"not measured: {blocked}",
            )
        ]
    import jax.numpy as jnp
    import torch

    rows: list[Row] = []
    for rings in RING_COUNTS:
        ray_count = hexapolar_ray_count(rings)
        elements = 3 * ray_count
        tensor = torch.randn(elements, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()

        def bridge(t: Any = tensor) -> Any:
            out = jnp.from_dlpack(t)
            out.block_until_ready()
            return out

        def to_host(t: Any = tensor) -> Any:
            return t.detach().cpu().numpy()

        dlpack_timing = _timing(bridge)
        host_timing = _timing(to_host)
        observed = array_state(bridge())
        rows.append(
            Row(
                case=f"p3-bridge-{elements}",
                configuration={
                    "elements_float32": elements,
                    "mebibytes": elements * 4 / 2**20,
                    "ray_count": ray_count,
                    "note": "3 x ray_count float32, i.e. one (N, 3) geometry column set",
                    "repetitions": REPETITIONS,
                    "statistic": (
                        "median of 5 after one discarded warm-up; min/max also recorded"
                    ),
                },
                descriptor="numerics.arrays.to_namespace",
                status="BASELINE",
                measured={
                    "timed_callable_dlpack": "jnp.from_dlpack(t) then block_until_ready()",
                    "timed_callable_host": "t.detach().cpu().numpy()",
                    "dlpack_on_device": dlpack_timing,
                    "device_to_host_copy": host_timing,
                    "host_copy_over_dlpack_median": (
                        host_timing["median_s"] / dlpack_timing["median_s"]
                    ),
                    "dlpack_result_state": str(observed),
                },
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=dlpack_timing["median_s"] + host_timing["median_s"],
                note=(
                    "The DLPack path is a device-resident view, so its cost is a flat "
                    "dispatch overhead while the copy it replaces scales with the buffer. "
                    "Below the crossover the bridge is the MORE expensive read -- 7x at "
                    "3e3 float32 -- which is not a defect but what a flat overhead against "
                    "a scaling copy looks like. "
                    "Recorded as the input to CHE-248's exit decision, and as the reason a "
                    "device exit is cheap: the crossover is where preserving the device "
                    "starts paying for itself. The result state is read back off the bridged "
                    "buffer rather than assumed."
                ),
            )
        )
    return rows


PROBES = {
    "p1": ("t1_p1_trace_exit_placement.json", "P1 -- where a trace exit lands"),
    "p2": ("t1_p2_gpu_benefit_curve.json", "P2 -- what CUDA is worth, and where"),
    "p3": ("t1_p3_bridge_cost.json", "P3 -- the on-device torch/JAX bridge"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", choices=(*PROBES, "all"), default="all", help="which probe to run"
    )
    arguments = parser.parse_args()
    selected = tuple(PROBES) if arguments.probe == "all" else (arguments.probe,)

    setup, _ = extract_setup(_sample_optic(), name=SAMPLE)
    status = 0
    for name in selected:
        filename, title = PROBES[name]
        print(f"\n=== {title} ===", flush=True)
        rows = {"p1": lambda: probe_placement(setup),
                "p2": lambda: probe_benefit(setup),
                "p3": probe_bridge}[name]()
        for row in rows:
            print(f"  {row.status:20s} {row.case:36s} {row.measured}", flush=True)
        record = probe_provenance()
        record.update({"probe": name, "ticket": "CHE-245", "title": title,
                       "produced_by": f"benchmarks/probes/optiland_device_exit.py --probe {name}",
                       "rows": [row.as_dict() for row in rows]})
        status |= finish(record, path=RECORDS / filename, rows=rows)
    return status


def _sample_optic() -> Any:
    from optiland.samples.objectives import CookeTriplet

    return CookeTriplet()


if __name__ == "__main__":
    raise SystemExit(main())
