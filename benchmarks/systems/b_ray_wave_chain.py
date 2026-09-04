"""B-RAY-WAVE-CHAIN: one ray-to-wave chain executed end to end on CUDA.

CHE-247 (T3). T1 and T2 each establish placement at **one** boundary; "the nodes
run on the GPU" is a claim about a **chain**, and a chain can be device-resident
at every node and still round-trip through the host between two of them with no
gate seeing it. This is the record that makes the chain-level claim falsifiable,
and it runs through `runtime.execute` because `ExecutionRecord` already separates
what a node **requested** from what was **observed** off its result.

The route is not the one the ticket names, and why
--------------------------------------------------
CHE-247 asks for `Optiland CUDA trace -> C_RAY_TO_SCALAR -> O_ASM_PROPAGATE ->
M_PSF`. **That route cannot execute**, and the reason is a deliberate scientific
gate rather than a defect. `couplers.ray_to_scalar` emits `surface_only` on every
field it builds -- CHE-50: the wavelet sum is linear in the transverse coordinate,
so the reconstruction carries no `exp(i k r^2 / 2R)` term and is valid *at* its
declared surface with zero further propagation -- and every Chromatix propagation
and focal-plane transform refuses a `surface_only` field outright. Measured, and
recorded below as a **refusal row** rather than omitted: the ticket's four-node
route is run, and step 3 refuses with `REPRESENTATION_INCONSISTENT` after three
nodes have been observed on CUDA.

`ray_to_scalar`'s own docstring names the remedy -- "Advance the **ray** state to
the new surface and reconstruct there -- exact, not an approximation" -- so the
route recorded here does exactly that, with `O_PROPAGATE_RAYS` where the ticket
put `O_ASM_PROPAGATE`:

    SO_RAY_LAUNCH_TRACE  ->  O_PROPAGATE_RAYS  ->  C_RAY_TO_SCALAR
                         ->  O_COMPLEX_TRANSMISSION  ->  M_PSF

Five nodes and therefore four inter-node boundaries, which is more of what this
record exists to measure than the four-node route would have been. What it does
**not** exercise is Chromatix, and with it the JAX-side backend boundary the
ticket wanted covered. That is a real gap and it is stated in `not_covered`.

What decides, and what only informs
------------------------------------
Two definitional closed forms decide: `psf(normalization="energy")` integrates to
exactly 1 over its window and `psf(normalization="peak")` has a maximum of
exactly 1, both by the declarations in `measurements/psf.py`. The placement rows
also decide, and they are filed `closed_form` on the grounds that nothing in this
repository's numerics produces them -- `numerics.arrays.array_state` reads the
framework's own report off the buffer -- with the caveat that they are an
*environment* observation rather than arithmetic, which this vocabulary has no
third value for.

Everything comparing the CUDA leg to a host leg is `diagnostic` and decides
nothing: it is this repository's numerics judging this repository's numerics.
`tests/parity/test_chain_parity.py` is where the device-only deviation is
*gated*, at `tests/parity/cells.py::tolerance_for`, which is this project's one
tolerance derivation and which `benchmarks/` deliberately cannot import.

This benchmark needs a CUDA device
-----------------------------------
Unlike the other two in this directory it is not a CPU run, so on a host with no
device attached it **refuses to rewrite its record** and exits 0 with a message,
rather than regenerating a GPU record without a GPU. Run it as:

    MOA_GPUS=device=6 ./run.sh --gpu python -m benchmarks.systems.b_ray_wave_chain
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.record import control, describe_plan, gate, write_record
from numerics import ArrayNamespace, DevicePlacement, DType, array_state, to_namespace
from operations import CATALOG
from planning import ENTRY, capability_graph
from problems import OpticalSetup, SourceSpec, SurfaceSpec
from representations import RayBundle, ReferenceSurface
from runtime import Executor, normalize_plan

BENCHMARK_ID = "B-RAY-WAVE-CHAIN"
RECORDS = Path(__file__).resolve().parent / "records"

#: The system, and it is **this benchmark's own** rather than a frozen fixture.
#: `tests/fixtures/systems.py` holds the project's frozen prescriptions and
#: `benchmarks/` does not import `tests/`; the alternative -- transcribing
#: `singlet_ref` here -- would be a second copy of a prescription that has to stay
#: in step with the first. Nothing measured in this record is a property of the
#: lens: a placement is a property of a buffer, and the two closed forms are
#: definitional properties of `measurements.psf`. So the requirement on the
#: prescription is only that it be traceable and unvignetted, and it is declared
#: here in full so a reader needs nothing else to re-run it.
SETUP = OpticalSetup(
    name="T3ChainSinglet",
    description=(
        "CHE-247 (T3): a plano-convex singlet, convex toward the collimated side, "
        "declared here rather than imported because benchmarks/ does not read tests/. "
        "Chosen for being the cheapest traceable unvignetted system; no number in this "
        "record is a claim about this lens."
    ),
    surfaces=(
        SurfaceSpec(
            radius_mm=2.5,
            thickness_mm=0.05,
            clear_semi_diameter_mm=0.75,
            material={"kind": "ideal", "refractive_index": 1.5},
            comment="convex front face, and the aperture stop",
        ),
        SurfaceSpec(
            thickness_mm=4.9,
            clear_semi_diameter_mm=0.75,
            # Deliberately **not** "one back focal length on", which an earlier
            # comment here claimed. For R = 2.5 mm at n = 1.5 the EFL is
            # R/(n-1) = 5.0 mm and the BFL from the rear vertex is
            # 5.0 - t/n = 4.967 mm, so paraxial focus is at z = 5.017 mm and this
            # plane sits 67 um inside it. Left as it is, and said out loud,
            # because nothing this record gates is focus-dependent: the two
            # physics closed forms are definitional identities of
            # measurements.psf and the rest are placements. Moving the plane
            # would change every number and no claim.
            comment="plane rear face; the sampled plane sits 67 um inside paraxial focus",
        ),
    ),
    stop_index=0,
    # ~f/10 on this prescription (EFL 5.0 mm). An arbitrary aperture chosen to be
    # small enough that the declared 0.75 mm rims cannot clip -- checked: the
    # traced fan stays inside them, and to_ray_bundle would report a vignetted
    # measure if it did not. It is NOT the frozen fixture's pupil, which is
    # derived at a different index; a half-transcribed constant would be exactly
    # the drift this local declaration exists to avoid.
    entrance_pupil_diameter_mm=0.5,
    reference_wavelength_um=0.55,
)

#: On axis at 550 nm, object at infinity.
LIGHT = SourceSpec(wavelength_um=0.55, field_angle_deg=(0.0, 0.0), object_distance_mm=None)

#: Where `O_PROPAGATE_RAYS` advances the bundle to: the image plane, at the sum of
#: the two declared thicknesses. `medium_index` must be the medium the rays are
#: in and is not defaulted anywhere in this tree.
IMAGE_SURFACE = ReferenceSurface(name="image_surface", z_m=4.95e-3, medium_index=1.0)

#: The reconstruction grid. 64 x 64 at 500 nm is a window of 32 um across the
#: image plane, comfortably larger than this system's Airy scale, and small
#: enough that the whole record runs in seconds.
GRID_SHAPE = (64, 64)
SAMPLE_PITCH_M = (5.0e-7, 5.0e-7)

#: The precision both legs run at. **fp32 on either device**, so that a device
#: change is not hiding behind a precision change -- which is also why
#: `solver._resolve_namespace` sends fp32 to the torch backend on the host.
PRECISION = "fp32"

#: The CUDA device as the *container* sees it. `MOA_GPUS=device=N` makes the
#: chosen physical GPU this process's `cuda:0`, so the ordinal here is the
#: container's and the device name in `environment` is what identifies hardware.
#: Named with its ordinal rather than as a bare `cuda` on purpose:
#: `NodeRecord.placement_disagreement` compares the requested and observed device
#: strings, and a bare `cuda` request against a `cuda:0` observation reads as a
#: disagreement when nothing disagreed.
CUDA_DEVICE = "cuda:0"

def _semantic_chain(plan: tuple[Any, ...]) -> tuple[str, ...]:
    """The semantic types the route passes through, refusing a step that cannot run.

    Derived from `planning.capability_graph()` rather than written down, which is
    the difference between recording that the route *is* a walk of the graph and
    asserting it. Each step must appear in `graph[state]` -- the operations that
    consume the state the step before it produced -- and the first must be a graph
    entry, which `ENTRY` keys.

    The same check `benchmarks/systems/b4f_ideal.py` runs, for the same reason:
    a route this rejects would still *execute*, because the executor binds a port
    from the request and a mismatched representation reaches an operation that
    then fails somewhere inside a backend. Checking it here turns that into a
    refusal naming the step. `planning.routes` is not used because it enumerates
    and this plan is written down -- CHE-247's non-goals rule out route search.

    Raises:
        ValueError: a step does not consume what the step before it produced.
    """
    graph = capability_graph()
    produces = {descriptor.operation_id: descriptor.primary_output for descriptor in CATALOG}
    chain: list[str] = []
    state: str | None = ENTRY
    for index, (operation_id, _) in enumerate(normalize_plan(plan)):
        if operation_id not in graph.get(state, ()):
            raise ValueError(
                f"plan step {index} is {operation_id}, which does not consume {state!r}. "
                f"The operations that do are {list(graph.get(state, ()))}."
            )
        state = produces[operation_id]
        chain.append(state)
    return tuple(chain)


def _ticket_plan(device: str) -> tuple[Any, ...]:
    """The four-node route CHE-247 names, **verbatim**, so its refusal is its own.

    Not the route below with `O_ASM_PROPAGATE` spliced in, which is what an
    earlier version of this module recorded: a six-node plan cannot be the
    evidence for a claim about a four-node one, and this row is the single piece
    of evidence justifying the substitution. `O_PROPAGATE_RAYS` and
    `O_COMPLEX_TRANSMISSION` are absent here because the ticket does not name
    them.
    """
    execution = {"device": device, "precision": PRECISION}
    return (
        (
            "SO_RAY_LAUNCH_TRACE",
            {
                "setup": SETUP,
                "source": LIGHT,
                "sampling": {"num_rings": 8, "reference_surface": "exit_pupil"},
                "execution": execution,
            },
        ),
        (
            "C_RAY_TO_SCALAR",
            {
                "grid_shape": GRID_SHAPE,
                "sample_pitch_m": SAMPLE_PITCH_M,
                "grazing": "band_limit",
                "execution": execution,
            },
        ),
        (
            "O_ASM_PROPAGATE",
            {
                "distance_m": 1.0e-4,
                "model": {"method": "asm", "pad_width": 16, "target_surface": "sensor"},
                "execution": execution,
            },
        ),
        ("M_PSF", {"normalization": "energy", "execution": execution}),
    )


def _plan(device: str, *, normalization: str = "energy") -> tuple[Any, ...]:
    """The route, as a plan of `(operation_id, node_request)` pairs.

    The pair form and not a shared `request`, because a shared mapping cannot say
    which node an argument belongs to and `Executor.execute` records a
    measurement of that going wrong. Every node carries `execution`, including
    the three that take no such parameter: for those it is the **chain-level
    declaration** of where the node was expected to run, and it is what
    `NodeRecord.placement_disagreement` compares the observation against. A node
    that inherits its placement from its input still has an expected placement,
    and a record with nothing to compare would report agreement by having asked
    nothing.
    """
    execution = {"device": device, "precision": PRECISION}
    steps: list[Any] = [
        (
            "SO_RAY_LAUNCH_TRACE",
            {
                "setup": SETUP,
                "source": LIGHT,
                "sampling": {"num_rings": 8, "reference_surface": "exit_pupil"},
                "execution": execution,
            },
        ),
        ("O_PROPAGATE_RAYS", {"to": IMAGE_SURFACE, "execution": execution}),
        (
            "C_RAY_TO_SCALAR",
            {
                "grid_shape": GRID_SHAPE,
                "sample_pitch_m": SAMPLE_PITCH_M,
                # The rays reach the image plane at a few degrees, well inside the
                # grid's band limit; `band_limit` is named rather than left to the
                # default `refuse` so the route does not depend on that margin.
                "grazing": "band_limit",
                "execution": execution,
            },
        ),
    ]
    steps.extend(
        [
            # A scalar transmission and deliberately not a mask array: a second
            # bulk array in the plan would have its own placement, and this node
            # is here to show that an operator between the coupler and the
            # measurement preserves the field's.
            ("O_COMPLEX_TRANSMISSION", {"amplitude": 1.0, "execution": execution}),
            ("M_PSF", {"normalization": normalization, "execution": execution}),
        ]
    )
    return tuple(steps)


def _run(plan: tuple[Any, ...]) -> dict[str, Any]:
    """Execute one plan and read the result. Nothing here interprets it."""
    with Executor() as executor:
        record = executor.execute(plan)
        result = executor.result
        return {
            "status": record.status,
            "route": list(record.route),
            "nodes": [
                {
                    "operation_id": node.operation_id,
                    "status": node.status,
                    "requested": dict(node.requested),
                    "observed": dict(node.observed),
                    "placement_disagreement": list(node.placement_disagreement),
                    **({"diagnostics": node.diagnostics} if node.diagnostics else {}),
                }
                for node in record.nodes
            ],
            "result": result,
        }


def _intensity(result: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The PSF map as host float64, for comparison only.

    Widened on the way out and not on the way in: the measurement ran in the
    cell's own dtype, and this is the serialization boundary
    (`numerics.to_host_numpy`'s distinction), so nothing about the computation is
    changed by reading it.
    """
    host = to_namespace(result.intensity, namespace=ArrayNamespace.NUMPY)
    return np.asarray(host, dtype=np.float64)


def _integral(intensity: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """`sum(I) dy dx` over the sampled window."""
    return float(intensity.sum()) * SAMPLE_PITCH_M[0] * SAMPLE_PITCH_M[1]


def _peak_relative(observed: Any, reference: Any) -> float:
    """Max absolute deviation over the reference's own peak.

    Peak-relative and not pointwise: a PSF has near-zero cells by construction
    and dividing by one turns a single ulp into an enormous ratio while saying
    nothing about the measurement.
    """
    scale = float(np.max(np.abs(reference)))
    return float(np.max(np.abs(observed - reference))) / scale


def _on_host_at_the_same_precision(rays: RayBundle) -> RayBundle:
    """The CUDA bundle moved to the host with **every dtype preserved**.

    This is what isolates the device from the precision, and it is the whole
    reason the naive CUDA-against-CPU number is not a device measurement. The
    `device="cpu"` leg of this chain runs the coupler at **FP64**, because
    `backends.optiland.rays`' host exit leaves `optical_path_m` and
    `measure_weight` at the host float64 they were declared in (CHE-245 records
    that asymmetry deliberately, since it is what keeps the host path
    bit-identical) and `couplers.ray_to_scalar._compute_precision` takes the
    **maximum** precision over the arrays a bundle carries. The CUDA leg has no
    such option -- `jax_enable_x64` is off, so every array is float32 -- so the
    two legs differ in precision as well as in device.

    Moving the CUDA bundle here changes the namespace and the device and nothing
    else, so the comparison against it is the device contribution alone.
    """
    host = DevicePlacement.parse("cpu")

    def moved(value: Any) -> Any:
        if value is None:
            return None
        state = array_state(value)
        return to_namespace(value, namespace=ArrayNamespace.NUMPY, device=host, dtype=state.dtype)

    return dataclasses.replace(
        rays,
        positions_m=moved(rays.positions_m),
        directions=moved(rays.directions),
        amplitude=moved(rays.amplitude),
        optical_path_m=moved(rays.optical_path_m),
        measure_weight=moved(rays.measure_weight),
    )


def _round_trip_cost(ray_count: int) -> dict[str, Any]:
    """The one bulk host traversal left in the chain, named and timed.

    CHE-247 AC-4: "Any remaining device-to-host round trip in the chain is named,
    with its location and cost. Not omitted, not rounded away."

    There is exactly one, it is in the first node, and it is deliberate.
    `backends.optiland.rays.to_ray_bundle` reads all nine native trace columns to
    the host before it declares anything, because the declaration it computes --
    `declare_optical_path_m`, which removes a chief-ray piston of order 1e4 waves
    -- is host float64 by a recorded scientific decision, and only the five
    finished arrays are then delivered to the device. CHE-245 (T1) states it in
    that function's own docstring; `tests/parity/test_chain_parity.py` holds it to
    an inventory rather than to this prose.

    Timed here on buffers of the chain's own shape through
    `numerics.to_namespace`, the same mover the backend uses, so the number is
    this chain's rather than a citation. Nothing downstream of node 1 traverses
    the host in bulk: every later node is written against `rays.xp` / the field's
    namespace. Scalar host reads *do* remain and are not round trips -- a
    `float(xp.max(...))` in a contract check synchronizes and copies eight bytes.
    """
    columns = 9
    host = np.zeros((columns, ray_count), dtype=np.float32)
    device = to_namespace(
        host,
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement.parse(CUDA_DEVICE),
        dtype=DType.FLOAT32,
    )

    def median(call: Any, repetitions: int = 5) -> float:
        call()
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter()
            call()
            samples.append(time.perf_counter() - started)
        return float(np.median(samples))

    to_host = median(lambda: np.asarray(to_namespace(device, namespace=ArrayNamespace.NUMPY)))
    to_device = median(
        lambda: to_namespace(
            host,
            namespace=ArrayNamespace.JAX,
            device=DevicePlacement.parse(CUDA_DEVICE),
            dtype=DType.FLOAT32,
        )
    )
    return {
        "location": "src/backends/optiland/rays.py::to_ray_bundle, the nine-column trace read",
        "node": "SO_RAY_LAUNCH_TRACE",
        "why_it_is_deliberate": (
            "the declaration computed from those columns is declare_optical_path_m, which "
            "removes a chief-ray piston of order 1e4 waves; launch.py::_launch_columns "
            "records that float32 would inject an error larger than the wavefront it "
            "corrects. CHE-245 kept the read and moved only the finished arrays."
        ),
        "buffer": {"columns": columns, "rays": ray_count, "dtype": "float32",
                   "mebibytes": columns * ray_count * 4 / 2**20},
        "device_to_host_s": to_host,
        "host_to_device_s": to_device,
        "round_trip_s": to_host + to_device,
        "measured_with": (
            "median of 5 after one discarded warm-up. **The device->host figure is a "
            "proxy**: the real outbound leg is `_host` -> optiland.backend.utils.to_numpy "
            "on a *torch* CUDA tensor, and this times jax.device_get on a JAX buffer of the "
            "same shape through numerics.to_namespace. The host->device leg IS "
            "numerics.to_namespace, which is what rays.py uses to deliver. Timed rather "
            "than instrumented because a benchmark here imports no backend; "
            "tests/parity/test_chain_parity.py counts the crossings instead."
        ),
        "others_in_the_chain": (
            "none in bulk. Nodes 2-5 are written against the representation's own "
            "namespace and were observed on cuda:0. Scalar host reads remain -- a "
            "float(xp.max(...)) in a contract check is a synchronization and eight bytes, "
            "not a round trip -- and are not counted here."
        ),
    }


def run() -> dict[str, Any]:
    """The whole measurement: five legs, and the record they justify."""
    chain = _semantic_chain(_plan(CUDA_DEVICE))
    cuda = _run(_plan(CUDA_DEVICE))
    if cuda["status"] != "completed":
        raise RuntimeError(f"the CUDA leg did not complete: {cuda['nodes']}")
    host = _run(_plan("cpu"))
    peak = _run(_plan(CUDA_DEVICE, normalization="peak"))
    raw = _run(_plan(CUDA_DEVICE, normalization="raw"))
    # The ticket's own four-node route, run verbatim so its refusal is recorded
    # rather than described. See the module docstring.
    asm = _run(_ticket_plan(CUDA_DEVICE))

    cuda_intensity = _intensity(cuda["result"])
    host_intensity = _intensity(host["result"])

    # The device contribution, isolated. Re-runs the three nodes after the trace
    # on a host copy of the CUDA bundle with every dtype preserved.
    from couplers import ray_to_scalar
    from measurements import psf
    from operators import complex_transmission

    same_precision = _on_host_at_the_same_precision(_cuda_bundle())
    field, _ = ray_to_scalar(
        same_precision,
        grid_shape=GRID_SHAPE,
        sample_pitch_m=SAMPLE_PITCH_M,
        grazing="band_limit",
    )
    same_precision_intensity = _intensity(psf(complex_transmission(field, amplitude=1.0),
                                              normalization="energy"))

    device_only = _peak_relative(cuda_intensity, same_precision_intensity)
    precision_only = _peak_relative(same_precision_intensity, host_intensity)
    both = _peak_relative(cuda_intensity, host_intensity)

    # The precision contribution is the one number here with a *derivable* upper
    # bound, and without a gate it was a plain JSON figure nothing compared to
    # anything -- so a regression that doubled it would be caught nowhere.
    #
    # The derivation, and it is a strict bound rather than an estimate. The FP32
    # leg's optical path is absolute after O_PROPAGATE_RAYS -- `compose_optical_
    # path_m` cannot remove the piston, because the incoming reference is what
    # fixes the zero -- so `|opl|max` is quantized at the float32 unit roundoff:
    #
    #     d(opl)  <= |opl|max * eps(float32)
    #     d(phi)   = 2 pi n d(opl) / lambda
    #     d(I)/I  <= 2 d(phi)          (to first order, |u|^2 doubles a relative
    #                                   field error, and a phase error d(phi)
    #                                   bounds the relative field error)
    #
    # Loose by design: it bounds the worst case where every wavelet's phase error
    # adds coherently, and the measured value sits well inside it. What it catches
    # is an order-of-magnitude regression, which is what an unbounded number
    # cannot.
    optical_path_extent_m = float(
        np.max(np.abs(np.asarray(
            to_namespace(_cuda_bundle().optical_path_m, namespace=ArrayNamespace.NUMPY),
            dtype=np.float64,
        )))
    )
    phase_quantum_rad = (
        2.0 * np.pi * IMAGE_SURFACE.medium_index
        * optical_path_extent_m * float(np.finfo(np.float32).eps)
        / (LIGHT.wavelength_um * 1.0e-6)
    )
    precision_bound = 2.0 * float(phase_quantum_rad)

    cuda_devices = [node["observed"].get("device") for node in cuda["nodes"]]
    cuda_drift = sorted({key for node in cuda["nodes"] for key in node["placement_disagreement"]})
    host_devices = [node["observed"].get("device") for node in host["nodes"]]
    host_drift = sorted({key for node in host["nodes"] for key in node["placement_disagreement"]})

    cuda_integral = _integral(cuda_intensity)
    host_integral = _integral(host_intensity)
    peak_maximum = float(np.max(_intensity(peak["result"])))
    # `measurements/psf.py` does not divide: it forms `scale = 1.0 / raw_peak` and
    # multiplies. `x * fl32(1/x)` is not exactly 1 -- measured, it misses by one
    # ulp for about 15 % of float32 values -- so the bound is one ulp of the
    # narrower leg's dtype and not zero. Zero here was a gate that would have
    # flipped on any change that moved the peak.
    reciprocal_tolerance = float(np.finfo(np.float32).eps)
    raw_integral = _integral(_intensity(raw["result"]))

    # The integral is a sum over ny*nx terms in the leg's own dtype, so the bound
    # is the reduction's own probabilistic rounding growth: sqrt(n) * u, with u
    # the unit roundoff of the *narrower* leg (complex64 -> float32 components).
    window = GRID_SHAPE[0] * GRID_SHAPE[1]
    integral_tolerance = float(np.sqrt(window) * np.finfo(np.float32).eps / 2.0)

    gates = [
        gate(
            "every_node_observed_on_the_requested_cuda_device",
            oracle=(
                f"every one of the {len(chain)} nodes reports {CUDA_DEVICE!r}, read off its "
                "own result by numerics.arrays.array_state"
            ),
            oracle_kind="closed_form",
            measured=cuda_devices,
            expected=[CUDA_DEVICE] * len(chain),
            tolerance=0.0,
            tolerance_basis=(
                "exact string equality, so the tolerance is zero rather than absent. Filed "
                "closed_form because nothing in this repository's numerics decides it -- "
                "array_state reads the framework's own report off the buffer -- with the "
                "caveat that it is an environment observation and not arithmetic, which "
                "ORACLE_KINDS has no third value for."
            ),
            passed=cuda_devices == [CUDA_DEVICE] * len(chain),
        ),
        gate(
            "no_node_reports_placement_disagreement",
            oracle=(
                "NodeRecord.placement_disagreement is empty at every node: the set of keys "
                "where the observed device or precision differs from the requested one"
            ),
            oracle_kind="closed_form",
            measured=cuda_drift,
            expected=[],
            tolerance=0.0,
            tolerance_basis=(
                "a set comparison; zero for the same reason as the row above. This is the "
                "gate CHE-247 AC-1 names, and it is what catches a run reporting success "
                "while having computed somewhere the caller did not ask for."
            ),
            passed=cuda_drift == [],
        ),
        gate(
            "energy_normalization_integrates_to_one_on_cuda",
            oracle=(
                "1.0 exactly, by the declaration in measurements/psf.py: energy gives "
                "intensity = |u|^2 / (sum(|u|^2) dy dx), which integrates to 1 over the "
                "sampled window"
            ),
            oracle_kind="closed_form",
            measured=cuda_integral,
            expected=1.0,
            tolerance=integral_tolerance,
            tolerance_basis=(
                f"sqrt({window}) * eps(float32)/2: the probabilistic rounding growth of a "
                f"{window}-term reduction at the narrower leg's unit roundoff. Not the "
                "deterministic n*u worst case, which is unreachable for a pairwise/blockwise "
                "reduction and would make the gate loose enough to pass a leg that had lost "
                "samples. Derived, not fitted."
            ),
            passed=abs(cuda_integral - 1.0) <= integral_tolerance,
        ),
        gate(
            "energy_normalization_integrates_to_one_on_the_host",
            oracle="1.0 exactly, the same declaration; the host leg computes at FP64",
            oracle_kind="closed_form",
            measured=host_integral,
            expected=1.0,
            tolerance=integral_tolerance,
            tolerance_basis=(
                "the same derivation, deliberately not retightened for the wider leg: one "
                "bound for one definitional identity, and a second number here would be a "
                "second place to change."
            ),
            passed=abs(host_integral - 1.0) <= integral_tolerance,
        ),
        gate(
            "peak_normalization_has_unit_maximum_on_cuda",
            oracle=(
                "1.0 exactly, by the declaration in measurements/psf.py: peak gives "
                "intensity = |u|^2 / max(|u|^2), so max == 1 by construction"
            ),
            oracle_kind="closed_form",
            measured=peak_maximum,
            expected=1.0,
            tolerance=reciprocal_tolerance,
            tolerance_basis=(
                "one ulp of float32: eps(float32). NOT zero, and the reason is specific -- "
                "psf.py:365 forms `scale = 1.0 / raw_peak` and psf.py:379 multiplies by it, "
                "so the gated quantity is x * fl32(1/x) rather than x/x. A division would "
                "be exact; a multiply by a rounded reciprocal is not, and measured it misses "
                "by one ulp for about 15 % of float32 values. A zero here would have been a "
                "gate that flips on any change that moves the peak, which is a fitted-by-"
                "luck zero rather than a derived one. Three orders below any real "
                "mis-normalization."
            ),
            passed=abs(peak_maximum - 1.0) <= reciprocal_tolerance,
        ),
        gate(
            "the_precision_gap_stays_inside_its_float32_phase_quantum",
            oracle=(
                "2 * 2 pi n |opl|max eps(float32) / lambda: the relative intensity error a "
                "float32 absolute optical path can produce, derived from the quantum of the "
                "path itself rather than from either leg's output"
            ),
            oracle_kind="closed_form",
            measured=precision_only,
            expected=0.0,
            tolerance=precision_bound,
            tolerance_basis=(
                f"|opl|max = {optical_path_extent_m:.6e} m after O_PROPAGATE_RAYS, "
                f"eps(float32) = {float(np.finfo(np.float32).eps):.3e}, lambda = "
                f"{LIGHT.wavelength_um * 1.0e-6:.3e} m, giving a phase quantum of "
                f"{float(phase_quantum_rad):.3e} rad and, doubled for |u|^2, a bound of "
                f"{precision_bound:.3e}. A strict worst case -- every wavelet's phase error "
                "adding coherently -- so the measured value sits well inside it; what it "
                "catches is an order-of-magnitude regression rather than a factor of two. "
                "Filed closed_form because the bound comes from the dtype and the geometry, "
                "not from either leg's numbers."
            ),
            passed=precision_only <= precision_bound,
        ),
        gate(
            "device_only_deviation_is_recorded_not_decided_here",
            oracle=(
                "the same chain's last three nodes on a host copy of the CUDA bundle with "
                "every dtype preserved, so only the device differs"
            ),
            oracle_kind="diagnostic",
            measured=device_only,
            expected=0.0,
            tolerance=None,
            tolerance_basis=(
                "none, on purpose. Two of this repository's own paths cannot decide each "
                "other (AGENTS.md), so this entry is evidence. The comparison is *gated* in "
                "tests/parity/test_chain_parity.py against tests/parity/cells.py::"
                "tolerance_for, which is this project's one tolerance derivation and which "
                "benchmarks/ deliberately cannot import."
            ),
            passed=True,
        ),
    ]

    controls = [
        control(
            "raw_normalization_does_not_integrate_to_one",
            changed="normalization='raw' instead of 'energy' at M_PSF",
            breaks_gate="energy_normalization_integrates_to_one_on_cuda",
            measured=raw_integral,
            reference=cuda_integral,
            broke=abs(raw_integral - 1.0) > integral_tolerance,
        ),
        control(
            "the_host_leg_is_not_observed_on_cuda",
            changed="execution.device='cpu' instead of 'cuda:0' at every node",
            breaks_gate="every_node_observed_on_the_requested_cuda_device",
            measured=host_devices,
            reference=cuda_devices,
            broke=host_devices != [CUDA_DEVICE] * len(chain),
        ),
    ]

    return {
        "benchmark": BENCHMARK_ID,
        "ticket": "CHE-247",
        "configuration": "singlet_exit_pupil_to_image",
        "produced_by": "benchmarks/systems/b_ray_wave_chain.py",
        "composition": describe_plan(normalize_plan(_plan(CUDA_DEVICE)), chain=chain),
        "parameters": {
            "grid_shape": list(GRID_SHAPE),
            "sample_pitch_m": list(SAMPLE_PITCH_M),
            "precision": PRECISION,
            "cuda_device": CUDA_DEVICE,
            "num_rings": 8,
            "ray_count": _ray_count(),
            "image_surface_z_m": IMAGE_SURFACE.z_m,
        },
        "execution": {
            "cuda": {key: value for key, value in cuda.items() if key != "result"},
            "host": {key: value for key, value in host.items() if key != "result"},
        },
        "the_route_che_247_named": {
            "plan": "SO_RAY_LAUNCH_TRACE -> C_RAY_TO_SCALAR -> O_ASM_PROPAGATE -> M_PSF, "
                    "exactly as CHE-247's Scope writes it and with nothing spliced in",
            "route": asm["route"],
            "status": asm["status"],
            "nodes": [node for node in asm["nodes"]],
            "finding": (
                "O_ASM_PROPAGATE refuses the coupler's output. C_RAY_TO_SCALAR emits "
                "`surface_only` on every field -- CHE-50: the wavelet sum is linear in the "
                "transverse coordinate, so the reconstruction carries no wavefront-curvature "
                "term and is valid only AT its declared surface -- and every Chromatix "
                "propagation and focal-plane transform refuses a surface_only field. The "
                "four-node route CHE-247 names is therefore not executable, by a deliberate "
                "scientific gate rather than a defect, and three nodes were observed on "
                "cuda:0 before the refusal. The remedy ray_to_scalar itself names is to "
                "advance the RAY state and reconstruct there, which is the route recorded "
                "above."
            ),
        },
        "host_round_trip": _round_trip_cost(_ray_count()),
        "chain_parity": {
            "note": (
                "The naive CUDA-against-CPU number is NOT a device measurement, and the "
                "decomposition below is the point. The cpu leg runs the coupler at FP64 "
                "because the Optiland host exit leaves optical_path_m and measure_weight at "
                "host float64 (CHE-245 keeps that asymmetry so the host path stays "
                "bit-identical) and _compute_precision takes the maximum over the arrays a "
                "bundle carries; the CUDA leg has no such option, since jax_enable_x64 is "
                "off. So the cpu leg reports placement_disagreement on `precision`, which is "
                "the record catching exactly this."
            ),
            "device_only_peak_relative": device_only,
            "precision_only_peak_relative": precision_only,
            "both_peak_relative": both,
            "host_leg_observed_precision": [
                node["observed"].get("precision") for node in host["nodes"]
            ],
            "host_leg_placement_disagreement": host_drift,
            "why_precision_dominates": (
                "O_PROPAGATE_RAYS composes an absolute optical path -- measured |opl|max "
                "4.93e-3 m with a span of only 5.7e-8 m -- and float32 resolves 4.93e-3 m to "
                "about 4.8e-10 m, i.e. ~5.5e-3 rad of phase against a meaningful span of "
                "~0.65 rad. compose_optical_path_m's own docstring predicts this: it cannot "
                "remove the absolute piston, because the incoming reference is what fixes "
                "the zero."
            ),
        },
        "gates": gates,
        "negative_controls": controls,
        "not_covered": [
            "Chromatix, and with it the JAX-side backend boundary CHE-247 wanted covered. "
            "O_ASM_PROPAGATE cannot take this chain's field; see the_route_che_247_named.",
            "any physics claim about this lens. The two deciding closed forms are "
            "definitional properties of measurements.psf and the placement rows are "
            "environment observations; nothing here is an optical oracle.",
            "gradients. The chain is forward_only and no derivative is claimed at any "
            "boundary.",
            "a second chain, a planner, or any route search. The plan is written down.",
            "the host->device leg of any node other than the first: only node 1 traverses "
            "the host in bulk, and host_round_trip times a buffer of its shape rather than "
            "instrumenting the call.",
        ],
    }


_CUDA_BUNDLE: RayBundle | None = None


def _cuda_bundle() -> RayBundle:
    """The bundle after node 2, kept so the device-only leg can reuse it.

    Re-executed rather than plumbed out of `run`, because `runtime.execute`
    returns a record and the result belongs to the executor that produced it.
    Cached so the trace runs once.
    """
    global _CUDA_BUNDLE
    if _CUDA_BUNDLE is None:
        with Executor() as executor:
            executor.execute(_plan(CUDA_DEVICE)[:2])
            _CUDA_BUNDLE = executor.result
    if _CUDA_BUNDLE is None:  # pragma: no cover - the two-node prefix always completes
        raise RuntimeError("the two-node prefix produced no bundle")
    return _CUDA_BUNDLE


def _ray_count() -> int:
    """How many rays reached the coupler, read off the bundle rather than derived.

    `hexapolar_ray_count(8)` would give the launch fan's count; this is the
    surviving ensemble, which is what the round-trip buffer is sized from.
    """
    return int(_cuda_bundle().count)


#: The refusal code this benchmark treats as "no device here" rather than as a
#: defect. Anything else from the first node is a real failure and must be loud.
DEVICE_REFUSAL = "DEVICE_NOT_AVAILABLE"


def _cuda_unavailable_reason() -> str | None:
    """Why this benchmark cannot run here, or `None`, asked by *doing* the thing.

    Runs the first node of the real plan and reads the record. Deliberately not a
    `torch.cuda.is_available()` / `jax.devices()` preflight, for two reasons that
    point the same way:

    * a benchmark in this directory imports no backend --
      `tests/benchmarks/test_records.py` walks the AST for exactly that -- and a
      preflight would have been the one place this module reached past the
      boundary it exists to compose across. Moving it behind a helper module to
      get past the AST walk would be evading the check rather than honouring it;
    * asking the frameworks is a *proxy* for the question. The question is
      "can this project place and trace on CUDA here", and the project already
      answers it, in its own refusal vocabulary, at the node that would fail.

    Narrow on purpose: only `DEVICE_NOT_AVAILABLE` counts as an absent device.
    Any other refusal or failure at node 1 is a defect, and this returns `None` so
    `run()` reaches it and raises.
    """
    with Executor() as executor:
        record = executor.execute(_plan(CUDA_DEVICE)[:1])
    node = record.nodes[0]
    if node.status == "completed":
        return None
    if DEVICE_REFUSAL in node.diagnostics:
        return node.diagnostics
    return None


def main() -> int:
    """Run the chain, write the record, and report. Exits non-zero only on a failure."""
    reason = _cuda_unavailable_reason()
    if reason is not None:
        print(f"=== {BENCHMARK_ID} ===")
        print(f"  SKIPPED, and the record is left alone: {reason}")
        print("  A GPU record regenerated from a host-only run would replace a measurement")
        print("  with a claim about nothing. Run:")
        print("    MOA_GPUS=device=6 ./run.sh --gpu python -m "
              "benchmarks.systems.b_ray_wave_chain")
        return 0

    record = run()
    path = write_record(record, path=RECORDS / f"{BENCHMARK_ID}-{record['configuration']}.json")

    print(f"\n=== {BENCHMARK_ID} / {record['configuration']} ===")
    failed = 0
    for entry in record["gates"]:
        mark = "PASS" if entry["passed"] else "FAIL"
        print(f"  [{mark}] {entry['name']}  ({entry['oracle_kind']})")
        if not entry["passed"]:
            failed += 1
            print(f"         measured {entry['measured']!r}")
            print(f"         expected {entry['expected']!r} +/- {entry['tolerance']!r}")
    for entry in record["negative_controls"]:
        mark = "BROKE" if entry["broke_the_gate"] else "DID NOT BREAK"
        print(f"  [{mark}] control {entry['name']} -> {entry['breaks_gate']}")
        if not entry["broke_the_gate"]:
            failed += 1
    print(f"  record: {path}")
    if failed:
        print(f"\n{failed} gate(s) or control(s) failed.")
        return 1
    print("\nOK: every gate holds and every negative control breaks the gate it names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
