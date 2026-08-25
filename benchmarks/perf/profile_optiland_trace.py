"""Where demo3's Optiland trace stage actually spends its 98.96 s (CHE-118 / M5.1).

    MOA_GPUS=device=6 ./run.sh --gpu python benchmarks/perf/profile_optiland_trace.py decompose
    MOA_GPUS=device=6 ./run.sh --gpu python benchmarks/perf/profile_optiland_trace.py chunks
    MOA_GPUS=device=6 ./run.sh --gpu python benchmarks/perf/profile_optiland_trace.py precision

Why this exists, and the lesson it is built around
--------------------------------------------------
CHE-96 attributed demo3's whole runtime to the reconstruction. CHE-101 then made
the reconstruction 9.6x faster on the kernel and the end-to-end run moved
207 s -> 197 s, because the reconstruction was 7%. So M5.1 profiles before it
optimizes, and this file is the profile. It is committed rather than scratch for
the same reason M0.4 exists: the next reader has to be able to re-run it and
find out whether their number is comparable.

What the profile found
----------------------
The trace stage decomposes, on a real 200 k-ray demo3 chunk through the four
traced surfaces, into:

* **~96%** host-side construction of Optiland's refractive-index *cache key*.
  ``BaseMaterial.n``/``.k`` memoize on the contents of the wavelength array:
  ``_create_cache_key`` evaluates ``tuple(np.ravel(be.to_numpy(wavelength)))``,
  so handing over one wavelength per ray copied 200 000 floats to the host, built
  a 200 000-element Python tuple and hashed it -- four times per surface, every
  chunk, at ~19 ms a call.
* **~2.6%** Optiland's actual ray-geometry kernels: intersection, refraction,
  path accumulation, and the per-surface recording of eight full-length arrays.
* **~0.9%** everything this repository does around the solver call: ``|a|^2``,
  the two ``bridge_arrays`` conversions, the clipped-ray mask, and one host
  synchronization to count the clipped rays.

The response is therefore not to tune the loop and not to reduce the ray count:
it is to stop broadcasting a scalar. ``RayBundle.wavelength_m`` is a single
``float`` by contract, so the per-ray array was always N copies of one number,
and a size-1 array traces **bitwise identically** -- asserted in
``tests/test_coherent_bridge.py::TestMonochromaticWavelengthHandoff``, on the
glass configuration as well as the air one, so the index weighting is exercised.

Both arms stay measurable from committed code
---------------------------------------------
``trace_ray_batch`` now ships the size-1 handoff, so a probe that only called it
could not reproduce the 98.96 s it is being compared against. Every section here
measures **both** arms by calling ``lens.surfaces.trace`` directly with a
wavelength array of size N and of size 1, and reconstructs the as-committed cost
as ``solver_trace(N) + repository handoff``. The reconstruction is checked
against the committed per-chunk figure rather than asserted: see
``closure_against_committed_record`` in the ``decompose`` record.

Not measured here, on purpose
-----------------------------
The emitter (M5.2) and the estimator's ray budget (M5.3). This file profiles the
trace and nothing else, and one of its findings is that after the fix the trace
is no longer demo3's binding constraint -- the emitter is.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# `demo3_hologram_lens` is the source of truth for the demo3 prescription and for
# the one order in which Optiland may be configured and built. A probe that
# rebuilt either would be profiling a system demo3 does not run.
sys.path.insert(0, str(ROOT / "benchmarks" / "probes" / "ray_wave"))

import numpy as np  # noqa: E402

from core.performance import (  # noqa: E402
    PerformanceRecord,
    StageTimer,
    Workload,
    environment_fingerprint,
    fit_scaling,
    measure,
)

RECORDS = Path(__file__).resolve().parent / "records"
SCHEMA_PATH = ROOT / "benchmarks" / "schemas" / "performance.schema.json"

#: The committed baseline this profile decomposes: CHE-129's demo3
#: `characterization` run on the `ramp_sum` route, 60 M rays in 300 chunks.
COMMITTED = {
    "record": "demo3_characterization_rw_p_ramp_sum_cuda.json",
    "issue": "CHE-129",
    "optiland_trace_s": 98.956,
    "total_s": 211.9521,
    "rays": 60_000_000,
    "chunks": 300,
    "rays_per_chunk": 200_000,
}

#: One demo3 `characterization` chunk, exactly: 50 patch centres x 4000 secondary
#: rays. `rays_per_chunk` 200 000 and 3000 patches over 60 groups gives 300 such
#: chunks, which is the `batches: 300` in the committed record.
CHUNK_PATCHES = 50
CHUNK_SECONDARY = 4_000

#: Repeat bounds and the wall-clock budget that picks between them.
#:
#: A fixed repeat count is wrong here because the arms span 5 ms to 7 s. Five
#: repeats of a 7 s arm is 35 s of a shared GPU; five repeats of an 11 ms arm on
#: an 80-core box is not a median of anything stable, and the first realization of
#: the affine cost model showed exactly that -- the fitted intercept, which is
#: dominated by the sub-20 k points, moved 10% between runs and dragged the model's
#: in-domain error from 1.2% to 17%. This is the lesson `_SCALING_REPEATS` records
#: in `run_baselines.py` (7 repeats, not 3, because a 3-repeat median of a 0.04 s
#: call gave exponents spread over 0.46), applied per arm instead of per file.
#:
#: So: repeats are chosen from a time budget, and the count and the observed
#: spread are recorded alongside every number they produced.
WARMUP = 1
_ARM_BUDGET_S = 1.0
_MIN_REPEATS = 5
_MAX_REPEATS = 41


def _write(name: str, payload: dict[str, Any]) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def _validate(record: PerformanceRecord) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    payload = record.as_dict()
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(payload)
    return payload


def _sync() -> None:
    """Drain the CUDA stream, so a stage boundary is a real boundary.

    Every timing in this file brackets its region with this. Torch dispatches
    asynchronously, so without it the first stage that touches the host would be
    charged for all the device work queued by the stages before it -- which is
    precisely the misattribution this profile exists to avoid.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timing(NamedTuple):
    """A median, and enough about how it was taken to judge it."""

    median_s: float
    repeats: int
    #: ``(max - min) / median`` over the timed repeats. The noise floor any ratio
    #: built from this number has to clear.
    relative_spread: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": round(self.median_s, 6),
            "repeats": self.repeats,
            "relative_spread": round(self.relative_spread, 4),
        }


def _timing(fn, *, warmup: int = WARMUP) -> Timing:
    """Median wall clock of ``fn``, synchronized on both sides of each call.

    Repeats come from ``_ARM_BUDGET_S`` and one probe call: a cheap arm gets many
    and an expensive one gets the floor, so neither a 5 ms arm nor a 7 s arm is
    sampled at the wrong count. The probe call doubles as the last warmup.
    """
    for _ in range(max(warmup, 1)):
        fn()
    _sync()
    started = time.perf_counter()
    fn()
    _sync()
    probe_s = time.perf_counter() - started

    repeats = int(min(_MAX_REPEATS, max(_MIN_REPEATS, _ARM_BUDGET_S / max(probe_s, 1e-6))))
    samples = []
    for _ in range(repeats):
        _sync()
        started = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - started)
    median = float(np.median(samples))
    spread = (max(samples) - min(samples)) / median if median else 0.0
    return Timing(median_s=median, repeats=repeats, relative_spread=spread)


def _median_s(fn, warmup: int = WARMUP) -> float:
    """``_timing`` where only the median is wanted."""
    return _timing(fn, warmup=warmup).median_s


def _measured(produce, *, label: str, workload: Workload, notes: list[str]):
    """Run ``produce`` under `measure`, and return its validated record and result.

    Every section here reports per-arm medians rather than `measure`'s own
    median, because the arms are different computations and one median over them
    is not a quantity. But the record is still taken and still committed: it is
    what carries the environment fingerprint, the memory report and the swap
    evidence, and a committed performance artifact without those is exactly what
    M0.4 exists to stop. `measure`'s `measurement` block is the cost of the whole
    arm group and the notes say so.
    """
    record, result = _measure_wrap(produce, label=label, workload=workload, notes=notes)
    return _validate(record), result


def _measure_wrap(produce, *, label: str, workload: Workload, notes: list[str]):
    return measure(
        lambda _timer: produce(),
        label=label,
        workload=workload,
        repeats=1,
        warmup=0,
        notes=[
            *notes,
            "Each arm inside this record is itself a median of device-synchronized "
            f"repeats chosen from a {_ARM_BUDGET_S:g} s budget "
            f"({_MIN_REPEATS}-{_MAX_REPEATS}), after warmup. `measurement` times the "
            "whole arm group and is not the figure to quote -- read the per-arm "
            "seconds in the payload.",
        ],
    )


# ---------------------------------------------------------------------------
# The demo3 chunk, built through demo3's own code
# ---------------------------------------------------------------------------


class Demo3Chunk:
    """The demo3 system and one of its chunks, with both wavelength arms ready.

    Holds the *bridged* inbound arrays rather than rebuilding them per timing
    call, so a per-surface measurement times the surface and not the bridge. The
    two wavelength arrays are the only difference between the arms.
    """

    def __init__(self, *, precision: str, patches: int, secondary: int) -> None:
        from _demo_support import enable_x64_if_needed

        enable_x64_if_needed(backend="jax", precisions=[precision])

        from demo3_hologram_lens import (
            WAVELENGTH_M,
            build_doe_and_lens,
        )

        from core.arrays import xp_for
        from core.boundary import ReferencePlane
        from core.bridge import bridge_arrays
        from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES, OPTILAND_CAPABILITIES
        from core.coherent_batch import (
            CoherentRayBatch,
            metres_to_micrometres,
            metres_to_millimetres,
        )
        from core.precision import ArrayNamespace, DeviceKind, DevicePlacement
        from couplers.patch import patch_secondary_rays, plan_patches
        from solvers.optiland.coherent_trace import plan_trace_bridges

        self.precision = precision
        self.wavelength_m = WAVELENGTH_M
        self.device = DevicePlacement(kind=DeviceKind.CUDA, index=0)
        self.doe, self.lens, self.spec, self.execution = build_doe_and_lens(
            backend="jax", precision=precision
        )
        self.doe_plane = ReferencePlane(name="doe", z_m=0.0)
        self.sensor = ReferencePlane(name="sensor", z_m=0.053)

        # demo3's own emitter, at demo3's own seed and plan, so the traced
        # geometry is the geometry the committed record traced.
        rng = np.random.default_rng(20260822)
        plan = plan_patches(
            grid_shape=self.doe.grid_shape,
            sample_pitch_m=self.doe.pitch_m,
            patch_px=101,
            pad_factor=2,
            patch_count=3_000,
            rng=rng,
        )
        centres = np.asarray(plan.centers_xy_m)[:patches]
        bundle, _ = patch_secondary_rays(
            self.doe.transmission,
            plan=dataclasses.replace(plan, centers_xy_m=centres),
            sample_pitch_m=self.doe.pitch_m,
            wavelength_m=WAVELENGTH_M,
            plane=self.doe_plane,
            secondary_count=secondary,
            rng=np.random.default_rng(1234),
        )

        xp = xp_for(ArrayNamespace.JAX)
        real_dtype = "float64" if precision == "fp64" else "float32"
        complex_dtype = "complex128" if precision == "fp64" else "complex64"
        moved = dataclasses.replace(
            bundle,
            positions_m=xp.asarray(np.asarray(bundle.positions_m), dtype=real_dtype),
            directions=xp.asarray(np.asarray(bundle.directions), dtype=real_dtype),
            amplitude=xp.asarray(np.asarray(bundle.amplitude), dtype=complex_dtype),
            optical_path_length_m=xp.asarray(
                np.asarray(bundle.optical_path_length_m), dtype=real_dtype
            ),
        )
        if str(moved.directions.dtype) != real_dtype:
            raise RuntimeError(
                f"asked for {real_dtype} and got {moved.directions.dtype}; on JAX this "
                "means jax_enable_x64 was not set before the first array"
            )
        self.batch = CoherentRayBatch(
            bundle=moved,
            ray_id=np.arange(moved.count, dtype=np.int64),
            valid=xp.ones(moved.count, dtype=bool),
        )
        self.count = int(moved.count)
        self.plans = plan_trace_bridges(
            self.batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=self.device
        )

        # The inbound arrays, bridged once. `trace_ray_batch` does this per call;
        # the per-arm timings below want the solver call isolated from it, and the
        # bridge is measured separately as part of the repository-side handoff.
        intensity = xp.abs(moved.amplitude) ** 2
        inbound, _ = bridge_arrays(
            {
                "positions_mm": metres_to_millimetres(moved.positions_m),
                "directions": moved.directions,
                "intensity": intensity,
            },
            OPTILAND_CAPABILITIES,
            reference="positions_mm",
            policy=self.plans.inbound.policy,
            target_device=self.plans.inbound.target_device,
        )
        self._positions_mm = inbound["positions_mm"]
        self._directions = inbound["directions"]
        self._intensity = inbound["intensity"]

        from solvers.optiland.coherent_trace import _solver_module

        sp = _solver_module(self.plans.inbound.target_namespace)
        wavelength_um = metres_to_micrometres(WAVELENGTH_M)
        #: The pre-CHE-118 handoff: one wavelength per ray.
        self.wavelengths_per_ray = sp.full_like(self._intensity, wavelength_um)
        #: The shipped handoff: one wavelength, full stop.
        self.wavelengths_scalar = sp.full_like(self._intensity[:1], wavelength_um)
        self._surface_count = len(self.lens.surfaces.surfaces)

    @property
    def traced_surfaces(self) -> int:
        """Surfaces actually traced at ``skip=1`` -- the object surface is skipped."""
        return self._surface_count - 1

    def rays(self, wavelengths: Any) -> Any:
        from optiland.rays import RealRays

        return RealRays(
            self._positions_mm[:, 0],
            self._positions_mm[:, 1],
            self._positions_mm[:, 2],
            self._directions[:, 0],
            self._directions[:, 1],
            self._directions[:, 2],
            self._intensity,
            wavelengths,
        )

    def solver_trace_s(self, wavelengths: Any, *, skip: int = 1) -> float:
        """Median seconds inside ``lens.surfaces.trace``, ray construction removed.

        Optiland mutates the rays it traces, so each repeat needs a fresh
        ``RealRays``; its cost is measured on its own and subtracted rather than
        left inside the number and called the trace.
        """
        build = _median_s(lambda: self.rays(wavelengths))
        both = _median_s(lambda: self.lens.surfaces.trace(self.rays(wavelengths), skip=skip))
        return both - build

    def whole_call_timing(self) -> Timing:
        """The shipped ``trace_ray_batch``, end to end, with its repeat count."""
        from solvers.optiland.coherent_trace import trace_ray_batch

        return _timing(
            lambda: trace_ray_batch(
                self.batch,
                self.lens,
                image_plane=self.sensor,
                plans=self.plans,
                skip=1,
            )
        )

    def whole_call_s(self) -> float:
        return self.whole_call_timing().median_s

    def handoff_breakdown_s(self) -> dict[str, float]:
        """The repository-side handoff, piece by piece.

        Individually measured, and **not** a partition: a piece timed on its own
        can cost differently in place, because JAX dispatches asynchronously and
        because an array the previous piece has already realized is cheaper to
        read. Reported because after CHE-118's fix this bucket is the majority of
        the trace stage, so the next reader needs to know which piece it is --
        and the sum is checked against the measured bucket rather than presented
        as equal to it.
        """
        from core.arrays import array_state, xp_for
        from core.bridge import bridge_arrays
        from core.capabilities import OPTILAND_CAPABILITIES
        from core.coherent_batch import metres_to_millimetres, millimetres_to_metres
        from solvers.optiland.coherent_trace import (
            _require_launch_plane,
            _solver_module,
            trace_ray_batch,
        )

        bundle = self.batch.bundle
        amplitude, _ = bundle.require_coherent()
        xp = xp_for(array_state(bundle.positions_m).namespace)
        sp = _solver_module(self.plans.inbound.target_namespace)

        def inbound_bridge():
            return bridge_arrays(
                {
                    "positions_mm": metres_to_millimetres(bundle.positions_m),
                    "directions": bundle.directions,
                    "intensity": xp.abs(amplitude) ** 2,
                },
                OPTILAND_CAPABILITIES,
                reference="positions_mm",
                policy=self.plans.inbound.policy,
                target_device=self.plans.inbound.target_device,
            )

        traced_rays = self.lens.surfaces.trace(self.rays(self.wavelengths_scalar), skip=1)

        def mask():
            positions = sp.stack(
                [sp.asarray(traced_rays.x), sp.asarray(traced_rays.y), sp.asarray(traced_rays.z)],
                axis=1,
            )
            directions = sp.stack(
                [sp.asarray(traced_rays.L), sp.asarray(traced_rays.M), sp.asarray(traced_rays.N)],
                axis=1,
            )
            opd = sp.asarray(traced_rays.opd)
            valid = (
                (sp.asarray(traced_rays.i) > 0)
                & sp.all(sp.isfinite(positions), axis=1)
                & sp.all(sp.isfinite(directions), axis=1)
                & sp.isfinite(opd)
            )
            column = valid[:, None]
            zero1 = sp.zeros_like(opd)
            return (
                sp.where(column, positions, sp.zeros_like(positions)),
                sp.where(column, directions, sp.stack([zero1, zero1, sp.ones_like(opd)], axis=1)),
                sp.where(valid, opd, zero1),
                sp.where(valid, sp.ones_like(opd), zero1),
            )

        masked = mask()

        def outbound_bridge():
            return bridge_arrays(
                {
                    "positions_mm": masked[0],
                    "directions": masked[1],
                    "optical_path_length_mm": masked[2],
                    "valid": masked[3],
                },
                self.plans.home,
                reference="positions_mm",
                policy=self.plans.outbound.policy,
                target_device=self.plans.outbound.target_device,
            )

        outbound, _ = outbound_bridge()
        valid = outbound["valid"] > 0.5

        def artifact():
            return self.batch.with_traced_state(
                positions_m=millimetres_to_metres(outbound["positions_mm"]),
                directions=outbound["directions"],
                optical_path_length_m=millimetres_to_metres(
                    outbound["optical_path_length_mm"]
                ),
                amplitude=xp.where(valid, amplitude, xp.zeros_like(amplitude)),
                valid=valid,
                plane=self.sensor,
                ray_id=self.batch.ray_id,
                provenance={"trace": {}},
            )

        traced_batch, _ = trace_ray_batch(
            self.batch, self.lens, image_plane=self.sensor, plans=self.plans, skip=1
        )
        return {
            "launch_plane_check_s": _median_s(
                lambda: _require_launch_plane(self.lens, bundle.reference_plane, 1)
            ),
            "intensity_and_inbound_bridge_s": _median_s(inbound_bridge),
            "real_rays_construction_s": _median_s(lambda: self.rays(self.wavelengths_scalar)),
            "clipped_ray_mask_s": _median_s(mask),
            "outbound_bridge_s": _median_s(outbound_bridge),
            "traced_batch_construction_s": _median_s(artifact),
            "clipped_ray_count_host_read_s": _median_s(lambda: int(traced_batch.valid.sum())),
        }

    def material_call_s(self) -> dict[str, float]:
        """Cost of one refractive-index lookup, per arm, on the glass surface.

        An independent route to the same number as ``solver_trace(N) -
        solver_trace(1)``: ``_trace_real`` calls ``material_pre.n`` once and the
        refraction calls ``n`` on both materials plus ``k`` on one, so four
        lookups per surface. Two derivations agreeing is what makes the
        attribution a measurement rather than a story.
        """
        glass = self.lens.surfaces.surfaces[2]
        return {
            "n_per_ray_s": _median_s(lambda: glass.material_pre.n(self.wavelengths_per_ray)),
            "k_per_ray_s": _median_s(lambda: glass.material_pre.k(self.wavelengths_per_ray)),
            "n_scalar_s": _median_s(lambda: glass.material_pre.n(self.wavelengths_scalar)),
            "lookups_per_surface": 4,
        }


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def profile_decompose() -> None:
    """The 98.96 s, attributed to named contributions that sum to it."""
    chunk = Demo3Chunk(
        precision="fp32", patches=CHUNK_PATCHES, secondary=CHUNK_SECONDARY
    )
    surfaces = chunk.traced_surfaces

    def arms(t: StageTimer) -> dict[str, float]:
        with t.stage("solver_trace_per_ray_wavelength"):
            per_ray = chunk.solver_trace_s(chunk.wavelengths_per_ray)
        with t.stage("solver_trace_scalar_wavelength"):
            scalar = chunk.solver_trace_s(chunk.wavelengths_scalar)
        with t.stage("whole_trace_ray_batch_as_shipped"):
            whole = chunk.whole_call_s()
        return {"per_ray": per_ray, "scalar": scalar, "whole": whole}

    # `measure` here is for the record's environment, memory and swap guards: the
    # timings that go in the payload are the per-arm medians, because the three
    # arms are three different computations and one median over them is not a
    # quantity. The workload is the arm sweep, and `route` says so.
    record, arm_s = measure(
        arms,
        label=f"optiland_trace_decomposition_{chunk.count}rays_fp32",
        workload=Workload(
            size=chunk.count,
            unit="ray",
            route="optiland_trace_arms",
            detail={
                "traced_surfaces": surfaces,
                "precision": "fp32",
                "patches": CHUNK_PATCHES,
                "secondary_per_patch": CHUNK_SECONDARY,
                "arms": "per-ray wavelength array, size-1 wavelength array, shipped call",
            },
        ),
        repeats=1,
        warmup=0,
        notes=[
            "Each arm is itself a median of synchronized repeats chosen from a "
            f"{_ARM_BUDGET_S:g} s budget; `measurement` here times the three arms "
            "together and is not the figure to quote.",
        ],
    )

    per_ray, scalar, whole = arm_s["per_ray"], arm_s["scalar"], arm_s["whole"]
    material_s = per_ray - scalar
    # The shipped call now uses the scalar arm, so the repository-side handoff is
    # what the whole call costs beyond the solver trace it contains.
    handoff_s = whole - scalar
    as_committed_s = per_ray + handoff_s

    handoff_pieces = chunk.handoff_breakdown_s()
    sync_s = handoff_pieces["clipped_ray_count_host_read_s"]
    materials = chunk.material_call_s()
    # Three `n` lookups and one `k` per surface, counted from the call graph
    # rather than assumed: `_trace_real` calls `material_pre.n` once, and the
    # refraction reads `n` on both materials and `k` on one. Confirmed by
    # cProfile on a single surface trace (9 `n` and 3 `k` calls over 3 traces).
    lookups = materials["lookups_per_surface"] * surfaces
    material_from_lookups_s = surfaces * (
        3.0 * materials["n_per_ray_s"] + 1.0 * materials["k_per_ray_s"]
    )

    per_surface = {}
    for arm, wavelengths in (
        ("per_ray_wavelength", chunk.wavelengths_per_ray),
        ("scalar_wavelength", chunk.wavelengths_scalar),
    ):
        cumulative = {}
        previous = 0.0
        # `skip` counts surfaces NOT traced, so it walks from the last surface
        # backwards and each step adds the surface in front of the previous one.
        for skip in range(chunk.traced_surfaces + 1, 0, -1):
            total = chunk.solver_trace_s(wavelengths, skip=skip)
            cumulative[f"skip_{skip}"] = {
                "surfaces_traced": chunk.traced_surfaces + 1 - skip,
                "cumulative_s": round(total, 6),
                "this_surface_s": round(total - previous, 6),
            }
            previous = total
        per_surface[arm] = cumulative

    committed_per_chunk_s = COMMITTED["optiland_trace_s"] / COMMITTED["chunks"]
    contributions = {
        "optiland_material_index_cache_key": {
            "seconds": round(material_s, 6),
            "fraction_of_as_committed": round(material_s / as_committed_s, 4),
            "what": (
                "host-side construction and hashing of optiland's refractive-index "
                "cache key. `BaseMaterial._create_cache_key` evaluates "
                "`tuple(np.ravel(be.to_numpy(wavelength)))`, so a per-ray "
                "wavelength array is copied to the host and turned into an "
                f"{chunk.count}-element Python tuple, {lookups} times per chunk"
            ),
            "attribution": (
                "the difference between the two wavelength arms, which differ in "
                "nothing else. Corroborated independently by timing the lookups: "
                f"{lookups} lookups at the measured per-call cost predict "
                f"{material_from_lookups_s:.4f} s against {material_s:.4f} s measured"
            ),
            "independent_estimate_s": round(material_from_lookups_s, 6),
            "independent_estimate_ratio": (
                round(material_from_lookups_s / material_s, 4) if material_s else None
            ),
            "ours_to_fix": True,
        },
        "optiland_ray_geometry_kernels": {
            "seconds": round(scalar, 6),
            "fraction_of_as_committed": round(scalar / as_committed_s, 4),
            "what": (
                "the trace itself: surface intersection, aperture clip, refraction, "
                "optical-path accumulation, and `_record_real`'s copy of eight "
                "full-length arrays per surface"
            ),
            "attribution": "the size-1-wavelength arm, which does no cache-key work",
            "ours_to_fix": False,
        },
        "repository_side_handoff": {
            "seconds": round(handoff_s, 6),
            "fraction_of_as_committed": round(handoff_s / as_committed_s, 4),
            "what": (
                "everything `trace_ray_batch` does around the solver call: |a|^2, "
                "the launch-plane check, the two `bridge_arrays` conversions, the "
                "clipped-ray mask, and one host synchronization to count clipped rays"
            ),
            "attribution": "the shipped call minus the solver trace it contains",
            "ours_to_fix": True,
        },
    }
    accounted = sum(c["seconds"] for c in contributions.values())

    def share(name: str) -> float:
        """A contribution's percentage of the as-committed chunk, for the prose."""
        return contributions[name]["fraction_of_as_committed"] * 100.0

    _write(
        "optiland_trace_decomposition",
        {
            "probe": "perf_optiland_trace_decomposition",
            "issue": "CHE-118 (M5.1)",
            "environment": environment_fingerprint().as_dict(),
            "optiland_execution": chunk.execution.as_dict(),
            "chunk": {
                "rays": chunk.count,
                "patches": CHUNK_PATCHES,
                "secondary_per_patch": CHUNK_SECONDARY,
                "traced_surfaces": surfaces,
                "precision": "fp32",
                "prescription": chunk.spec.name,
                "note": (
                    "one demo3 `characterization` chunk, emitted by demo3's own "
                    "patch emitter at demo3's seed, so the traced geometry is the "
                    "geometry the committed record traced"
                ),
            },
            "arms_s": {
                "solver_trace_per_ray_wavelength": round(per_ray, 6),
                "solver_trace_scalar_wavelength": round(scalar, 6),
                "whole_trace_ray_batch_as_shipped": round(whole, 6),
                "reconstructed_as_committed": round(as_committed_s, 6),
            },
            "contributions": contributions,
            "accounted_s": round(accounted, 6),
            "unattributed_s": round(as_committed_s - accounted, 6),
            "unattributed_fraction": round((as_committed_s - accounted) / as_committed_s, 4),
            "why_the_remainder_is_zero": (
                "the three contributions are differences between measured arms and "
                "therefore partition the reconstructed total exactly, so a zero "
                "remainder here is arithmetic and not evidence. The two things that "
                "ARE evidence: `closure_against_committed_record`, which checks the "
                "reconstructed total against the 98.96 s this profile claims to "
                "decompose, and `independent_estimate_ratio`, which reproduces the "
                "dominant contribution from a separate measurement of the material "
                "lookups it is made of."
            ),
            "closure_against_committed_record": {
                "committed_record": COMMITTED["record"],
                "committed_issue": COMMITTED["issue"],
                "committed_optiland_trace_s": COMMITTED["optiland_trace_s"],
                "committed_chunks": COMMITTED["chunks"],
                "committed_s_per_chunk": round(committed_per_chunk_s, 6),
                "reconstructed_s_per_chunk": round(as_committed_s, 6),
                "ratio": round(as_committed_s / committed_per_chunk_s, 4),
                "note": (
                    "the committed stage timer brackets the whole "
                    "`trace_ray_batch` call per chunk, which is what "
                    "`reconstructed_as_committed` reconstructs. Agreement here is "
                    "what licenses reading the fractions above as fractions of the "
                    "98.96 s rather than of a probe's own workload."
                ),
            },
            "handoff_breakdown_s": {
                **{k: round(v, 6) for k, v in handoff_pieces.items()},
                "sum_of_pieces_s": round(sum(handoff_pieces.values()), 6),
                "measured_bucket_s": round(handoff_s, 6),
                "closure_ratio": (
                    round(sum(handoff_pieces.values()) / handoff_s, 4) if handoff_s else None
                ),
                "note": (
                    "individually measured pieces, NOT a partition of the bucket. A "
                    "piece timed on its own can cost differently in place: JAX "
                    "dispatches asynchronously, and an array a previous piece "
                    "already realized is cheaper for the next one to read. "
                    "`closure_ratio` is how far that goes here. Reported because "
                    "after CHE-118's fix this bucket is the majority of the trace "
                    "stage, so which piece it is has become the useful question -- "
                    "though the stage itself is now ~5 s of demo3, so the answer is "
                    "information for M3.1's cost model rather than a target."
                ),
            },
            "per_surface_s": per_surface,
            "material_lookup_s": {k: round(v, 6) for k, v in materials.items()},
            "per_ray_cost_model": {
                "seconds_per_ray_per_surface_as_committed": (
                    as_committed_s / chunk.count / surfaces
                ),
                "seconds_per_ray_per_surface_as_shipped": whole / chunk.count / surfaces,
                "note": (
                    "the as-shipped figure is the one to plan against; the "
                    "as-committed figure is what every demo3 record before this "
                    "issue was paying. Both are fp32 on the environment "
                    "fingerprint above, and neither is portable off it."
                ),
            },
            "host_synchronizations_in_the_trace_loop": {
                "found": ["int(valid.sum()) in trace_ray_batch's diagnostics"],
                "marginal_reduction_s": round(sync_s, 6),
                "marginal_reduction_note": (
                    "the reduction and host read, timed on the traced `valid` array "
                    "once the trace's own work has drained. It is a LOWER bound on "
                    "what the synchronization costs in place, where it may also "
                    "wait on outstanding device work."
                ),
                "bounded_above_by_s": round(handoff_s, 6),
                "bounded_above_by": (
                    "`repository_side_handoff`, which contains it -- so the "
                    "synchronization cannot be more than "
                    f"{handoff_s / as_committed_s * 100:.1f}% of the as-committed "
                    "chunk however much of that stage it turns out to be"
                ),
                "disposition": (
                    "kept, and justified rather than removed. It reads one scalar -- "
                    "the clipped-ray count -- which demo3's energy ledger consumes "
                    "per chunk as `invalid_rays` to separate an aperture loss from "
                    "an empty draw. Nothing else in the loop reads a device value "
                    "to the host, and no shape in it is data-dependent, so there is "
                    "no synchronization of the kind CHE-101 found costing 49 s in "
                    "the reconstruction."
                ),
            },
            "finding": (
                f"Of the reconstructed {as_committed_s * 1e3:.1f} ms per 200 k-ray "
                f"chunk, {share('optiland_material_index_cache_key'):.1f}% is "
                "optiland building and hashing a Python tuple of the wavelength "
                f"array, {share('optiland_ray_geometry_kernels'):.1f}% is the ray "
                f"geometry it exists to compute, and {share('repository_side_handoff'):.1f}% "
                "is this repository's handoff. The cost is uniform per surface -- the "
                "spherical surface's intersection solve is not distinguishable from "
                "the planes -- because the dominant term is per-surface material "
                "lookups, not geometry."
            ),
            "consequence": (
                "The lever is the wavelength handoff, not the chunk size, not the "
                "precision, and not the ray count. `RayBundle.wavelength_m` is a "
                "scalar by contract, so the per-ray array was N copies of one "
                "number; a size-1 array traces bitwise identically and is what "
                "`trace_ray_batch` now ships. After it the trace stage is ~3 s of "
                "demo3 rather than 99 s, and the emitter (M5.2) becomes the binding "
                "constraint."
            ),
            "record": _validate(record),
        },
    )
    print(f"per-ray arm {per_ray * 1e3:.2f} ms | scalar arm {scalar * 1e3:.2f} ms | "
          f"shipped {whole * 1e3:.2f} ms | speedup {as_committed_s / whole:.1f}x")


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------


#: Chunk sizes the affine cost model is fitted on, in rays. The 4 M point is
#: excluded and the exclusion is reported rather than silent: cost per ray turns
#: back up there, so a fit through it would describe neither regime. Its residual
#: against the fit is recorded as the evidence for leaving it out.
COST_MODEL_DOMAIN_RAYS = (1.0e3, 1.0e6)


def _affine_cost_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit ``t = fixed + per_ray_per_surface * rays * surfaces`` to the shipped arm.

    Affine and not a power law, because after CHE-118's fix the trace is neither
    flat nor linear: it is a fixed per-call cost plus per-ray work, and the two
    terms cross over inside the range demo3 uses. `fit_scaling` would return an
    exponent near 0.34 with an r^2 of 0.7, which is not a bad fit to a power law
    so much as evidence that this is not a power law -- and an executor planning
    against that exponent would misprice every chunk size.

    Least squares in the ray count, not a two-point slope, so the residuals are
    reported and a reader can see whether the affine form holds at all.
    """
    inside = [
        r
        for r in rows
        if COST_MODEL_DOMAIN_RAYS[0] <= r["rays"] <= COST_MODEL_DOMAIN_RAYS[1]
    ]
    outside = [r for r in rows if r not in inside]
    surfaces = inside[0]["traced_surfaces"]
    x = np.array([r["rays"] * r["traced_surfaces"] for r in inside], dtype=float)
    y = np.array([r["whole_trace_ray_batch_s"] for r in inside], dtype=float)
    design = np.stack([np.ones_like(x), x], axis=1)
    (fixed_s, per_ray_surface_s), *_ = np.linalg.lstsq(design, y, rcond=None)

    def predict(rays: float, surfaces_: int) -> float:
        return float(fixed_s + per_ray_surface_s * rays * surfaces_)

    residuals = [
        {
            "rays": r["rays"],
            "measured_s": r["whole_trace_ray_batch_s"],
            "predicted_s": round(predict(r["rays"], r["traced_surfaces"]), 6),
            "relative_error": round(
                predict(r["rays"], r["traced_surfaces"]) / r["whole_trace_ray_batch_s"] - 1.0,
                4,
            ),
            "in_domain": r in inside,
        }
        for r in rows
    ]
    return {
        "form": "seconds = fixed_per_call_s + seconds_per_ray_per_surface * rays * surfaces",
        "fixed_per_call_s": float(fixed_s),
        "seconds_per_ray_per_surface": float(per_ray_surface_s),
        "fitted_on_surfaces": surfaces,
        "domain_rays": list(COST_MODEL_DOMAIN_RAYS),
        "max_relative_error_in_domain": max(
            abs(r["relative_error"]) for r in residuals if r["in_domain"]
        ),
        "residuals": residuals,
        "excluded_points": [r["rays"] for r in outside],
        "why_excluded": (
            "cost per ray turns back up above ~1 M rays a chunk, so the affine form "
            "stops holding and the residual there says by how much. Extrapolating "
            "this model past its domain underpredicts; the domain is part of the "
            "model and is carried with it."
        ),
        "why_affine_and_not_a_power_law": (
            "the shipped arm is a fixed per-call cost plus per-ray work, and the two "
            "terms cross over inside the range demo3 uses. The power-law fit in "
            "`fits` returns a poor r^2 for exactly that reason, and it is kept "
            "beside this one as the evidence that a single exponent will not do."
        ),
        "note": (
            "fitted at one surface count, so `surfaces` enters the per-ray term by "
            "assumption rather than by measurement. `per_surface_s` in the "
            "decomposition record is what supports it: the four surfaces cost within "
            "a few percent of each other on both arms."
        ),
    }


def profile_chunks() -> None:
    """Is chunk size a lever? Fitted on both arms rather than argued from one point."""
    # (patches, secondary) pairs, 1 k - 4 M rays. The low end is not padding: the
    # standalone pupil trace `OptilandAdapter.estimate()` is asked about runs at a
    # few thousand rays (32 hexapolar rings is 3169), so a model fitted only on
    # demo3's 200 k chunks would refuse the configuration M0.4 scores it on. The
    # fixed per-call term is what dominates down here, and it has to be sampled
    # where it dominates rather than extrapolated to.
    sweep = (
        (1, 1_000),
        (1, 3_169),
        (2, 5_000),
        (5, 4_000),
        (25, 4_000),
        (50, 4_000),
        (50, 20_000),
        (200, 20_000),
    )
    rows: list[dict[str, Any]] = []
    for patches, secondary in sweep:
        chunk = Demo3Chunk(precision="fp32", patches=patches, secondary=secondary)

        def arms(chunk: Demo3Chunk = chunk) -> tuple[float, float, Timing]:
            return (
                chunk.solver_trace_s(chunk.wavelengths_per_ray),
                chunk.solver_trace_s(chunk.wavelengths_scalar),
                chunk.whole_call_timing(),
            )

        row_record, (per_ray, scalar, whole_timing) = _measured(
            arms,
            label=f"optiland_trace_chunk_{chunk.count}rays_fp32",
            workload=Workload(
                size=chunk.count,
                unit="ray",
                route="optiland_trace_arms",
                detail={
                    "traced_surfaces": chunk.traced_surfaces,
                    "precision": "fp32",
                    "patches": patches,
                    "secondary_per_patch": secondary,
                },
            ),
            notes=["One point of the chunk-size sweep, all three arms."],
        )
        whole = whole_timing.median_s
        rows.append(
            {
                "record": row_record,
                "whole_trace_ray_batch_timing": whole_timing.as_dict(),
                "patches": patches,
                "secondary_per_patch": secondary,
                "rays": chunk.count,
                "traced_surfaces": chunk.traced_surfaces,
                "solver_trace_per_ray_wavelength_s": round(per_ray, 6),
                "solver_trace_scalar_wavelength_s": round(scalar, 6),
                "whole_trace_ray_batch_s": round(whole, 6),
                "ns_per_ray_per_surface_as_committed": round(
                    (per_ray + whole - scalar) / chunk.count / chunk.traced_surfaces * 1e9, 3
                ),
                "ns_per_ray_per_surface_as_shipped": round(
                    whole / chunk.count / chunk.traced_surfaces * 1e9, 3
                ),
            }
        )
        print(f"  rays={chunk.count:>9,}  per-ray arm {per_ray * 1e3:8.2f} ms  "
              f"scalar arm {scalar * 1e3:7.2f} ms  shipped {whole * 1e3:8.2f} ms")
        del chunk

    fits = {
        "solver_trace_per_ray_wavelength": fit_scaling(
            [(r["rays"], r["solver_trace_per_ray_wavelength_s"]) for r in rows], axis="rays"
        ).as_dict(),
        "solver_trace_scalar_wavelength": fit_scaling(
            [(r["rays"], r["solver_trace_scalar_wavelength_s"]) for r in rows], axis="rays"
        ).as_dict(),
        "whole_trace_ray_batch": fit_scaling(
            [(r["rays"], r["whole_trace_ray_batch_s"]) for r in rows], axis="rays"
        ).as_dict(),
        # The per-ray arm's exponent, fitted only where that arm's own cost
        # dominates the fixed per-call floor. Below ~10 k rays the two are
        # comparable -- the arm is 6 ms at 1 k against a ~5 ms floor -- so a fit
        # across the whole sweep reads 0.87 and understates a term that really is
        # linear. Both are reported: the restricted fit is the claim, the full one
        # is why it needed restricting.
        "solver_trace_per_ray_wavelength_above_10k_rays": fit_scaling(
            [
                (r["rays"], r["solver_trace_per_ray_wavelength_s"])
                for r in rows
                if r["rays"] >= 1.0e4
            ],
            axis="rays",
        ).as_dict(),
    }
    linear_fit = fits["solver_trace_per_ray_wavelength_above_10k_rays"]
    cost_model = _affine_cost_model(rows)
    shipped = [r["ns_per_ray_per_surface_as_shipped"] for r in rows]
    committed = [r["ns_per_ray_per_surface_as_committed"] for r in rows]
    best_shipped = min(rows, key=lambda r: r["ns_per_ray_per_surface_as_shipped"])
    best_committed = min(rows, key=lambda r: r["ns_per_ray_per_surface_as_committed"])
    demo3_row = next(r for r in rows if r["rays"] == COMMITTED["rays_per_chunk"])
    shipped_headroom = (
        demo3_row["ns_per_ray_per_surface_as_shipped"]
        / best_shipped["ns_per_ray_per_surface_as_shipped"]
    )
    # The as-committed spread, stated over the range where the O(rays) term
    # dominates. Across the WHOLE sweep it reads 7.7x, and calling that "flat"
    # would be wrong for the opposite reason to the one being argued: at 1 k rays
    # even the per-ray arm is mostly the fixed per-call cost, so the rise there is
    # the floor showing through rather than the wavelength array scaling.
    committed_dominant = [
        r["ns_per_ray_per_surface_as_committed"] for r in rows if r["rays"] >= 1.0e4
    ]

    _write(
        "optiland_trace_chunk_sweep",
        {
            "probe": "perf_optiland_trace_chunk_sweep",
            "issue": "CHE-118 (M5.1)",
            "environment": environment_fingerprint().as_dict(),
            "held_fixed": {
                "precision": "fp32",
                "prescription": "Demo3HologramLA1131A",
                "traced_surfaces": rows[0]["traced_surfaces"],
                "reconstruction": "not run -- this sweep is the trace alone",
            },
            "rows": rows,
            "fits": fits,
            "affine_cost_model": cost_model,
            "chunk_size_optimum": {
                "as_committed": {
                    "optimum_rays": best_committed["rays"],
                    "ns_per_ray_per_surface": best_committed["ns_per_ray_per_surface_as_committed"],
                    "spread_across_sweep": round(max(committed) / min(committed), 3),
                    "demo3_ships_rays": COMMITTED["rays_per_chunk"],
                    "demo3_ns_per_ray_per_surface": demo3_row[
                        "ns_per_ray_per_surface_as_committed"
                    ],
                    "verdict": (
                        "chunk size is NOT a lever on the arm this issue was asked "
                        "to explain. The per-ray wavelength array costs O(rays) of "
                        "host work per surface -- fitted exponent "
                        f"{linear_fit['exponent']:.3f} at r^2 "
                        f"{linear_fit['r_squared']:.4f} above 10 k rays, where that "
                        "term dominates the fixed per-call floor -- so it is not a "
                        "per-chunk overhead that a larger chunk amortizes, and cost "
                        "per ray is flat to "
                        f"{max(committed_dominant) / min(committed_dominant):.2f}x "
                        "from 10 k rays up, where that term dominates. Tuning "
                        "`--rays-per-chunk` was the cheapest candidate response in "
                        "CHE-118's list and the measurement rejects it."
                    ),
                },
                "as_shipped": {
                    "optimum_rays": best_shipped["rays"],
                    "ns_per_ray_per_surface": best_shipped["ns_per_ray_per_surface_as_shipped"],
                    "spread_across_sweep": round(max(shipped) / min(shipped), 3),
                    "demo3_ships_rays": COMMITTED["rays_per_chunk"],
                    "demo3_ns_per_ray_per_surface": demo3_row[
                        "ns_per_ray_per_surface_as_shipped"
                    ],
                    "available_speedup_on_the_stage": round(
                        demo3_row["ns_per_ray_per_surface_as_shipped"]
                        / best_shipped["ns_per_ray_per_surface_as_shipped"],
                        2,
                    ),
                    "verdict": (
                        "chunk size BECOMES a lever once the cache-key cost is gone, "
                        "and this is the opposite of the pre-fix answer rather than a "
                        "refinement of it. With the O(rays) host work removed the "
                        "trace is dominated by a fixed per-call cost of about "
                        f"{cost_model['fixed_per_call_s'] * 1e3:.0f} ms: near-constant "
                        f"from {rows[0]['rays']:,} to "
                        f"{COST_MODEL_DOMAIN_RAYS[1]:,.0f} rays, so cost per ray falls "
                        f"{max(shipped) / min(shipped):.0f}x across the sweep and "
                        f"demo3's {COMMITTED['rays_per_chunk']:,} chunk sits "
                        f"{shipped_headroom:.1f}x above the optimum's cost per ray."
                    ),
                    "why_it_is_not_taken_here": (
                        "the prize is small and the knob is not the trace's to turn. "
                        "The stage is ~5 s of demo3 after the fix, so the whole "
                        "available speedup is a couple of percent of the run; and "
                        "`--rays-per-chunk` also sizes the patch emitter's spectra "
                        "and the reconstruction's accumulation, whose memory it "
                        "bounds -- the 4 M-ray point here is a trace-only "
                        "measurement and does not show that demo3 can run there. "
                        "Choosing it belongs to M5.2, which owns the stage that "
                        "actually pays for the chunk size."
                    ),
                },
            },
            "finding": (
                "The two arms scale differently, and that is the result. The "
                "per-ray-wavelength arm is linear in the ray count wherever its own "
                f"cost dominates -- exponent {linear_fit['exponent']:.3f} at r^2 "
                f"{linear_fit['r_squared']:.4f} above 10 k rays -- because it is "
                "O(rays) of host-side tuple construction done once per surface, which "
                "no chunk size amortizes. The size-1 arm is near-flat to 1 M rays: "
                "exponent "
                f"{fits['solver_trace_scalar_wavelength']['exponent']:.3f} at r^2 "
                f"{fits['solver_trace_scalar_wavelength']['r_squared']:.4f}, a poor "
                "power-law fit precisely because a flat cost is not a power law. What "
                "is left after the fix is per-call overhead, which is why the cost "
                "model below is affine rather than a scaling exponent."
            ),
            "consequence": (
                "The chunk-size question has two answers depending on which arm is "
                "running, and reporting one of them would have been wrong either "
                "way. Pre-fix: not a lever, which is what rejects the cheapest "
                "candidate response in CHE-118's list. Post-fix: a ~30x lever on "
                "cost per ray, on a stage now too small for that to matter much, and "
                "on a knob shared with the emitter and the reconstruction. Handed to "
                "M5.2 with the measurement rather than acted on here."
            ),
        },
    )


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------


def _host(traced: Any) -> dict[str, np.ndarray]:
    """Traced ray state on the host as float64, whichever namespace it came from.

    Promoted to float64 for the comparison only. The fp32 arm is *computed* in
    fp32 -- promoting a result does not recover the precision it was computed at,
    and the point of the comparison is to measure what was lost.
    """
    out = {}
    for key in ("x", "y", "z", "L", "M", "N", "opd"):
        value = getattr(traced, key)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        out[key] = np.asarray(value, dtype=np.float64)
    return out


def profile_precision() -> None:
    """fp32 versus fp64: what the faster trace costs in the answer, if anything.

    Both arms in one process, because the accuracy comparison needs the two
    traced results side by side and `configure_optiland_execution` is
    process-global. JAX's x64 mode is enabled for the whole process -- it has to
    be set before the first array -- and every dtype below is stated explicitly,
    so enabling it does not silently promote the fp32 arm.

    fp64 is measured **twice**, before and after the fp32 arm, and the difference
    between those two identical computations is the noise floor every ratio here
    is read against. Without it the first realization of this section reported
    fp64 running at 0.86x the fp32 cost on the whole call -- a ratio below one for
    the more expensive precision, which is not a physical result, it is the
    per-call overhead moving between two runs. A cost ratio that does not clear
    its own noise floor is reported as "no measurable difference" rather than as a
    number.
    """

    def timings(chunk: Demo3Chunk) -> dict[str, float]:
        return {
            "solver_trace_per_ray_wavelength_s": chunk.solver_trace_s(
                chunk.wavelengths_per_ray
            ),
            "solver_trace_scalar_wavelength_s": chunk.solver_trace_s(
                chunk.wavelengths_scalar
            ),
            "whole_trace_ray_batch_s": chunk.whole_call_s(),
        }

    def build(precision: str) -> Demo3Chunk:
        return Demo3Chunk(
            precision=precision, patches=CHUNK_PATCHES, secondary=CHUNK_SECONDARY
        )

    def measured(chunk: Demo3Chunk, label: str) -> tuple[dict[str, Any], dict[str, float]]:
        return _measured(
            lambda: timings(chunk),
            label=label,
            workload=Workload(
                size=chunk.count,
                unit="ray",
                route="optiland_trace_arms",
                detail={
                    "traced_surfaces": chunk.traced_surfaces,
                    "precision": chunk.precision,
                    "patches": CHUNK_PATCHES,
                    "secondary_per_patch": CHUNK_SECONDARY,
                },
            ),
            notes=[f"All three trace arms at {chunk.precision}."],
        )

    fp64 = build("fp64")
    fp64_record, fp64_timings = measured(fp64, "optiland_trace_precision_fp64")
    fp64_out = _host(fp64.lens.surfaces.trace(fp64.rays(fp64.wavelengths_scalar), skip=1))
    fp64_execution = fp64.execution.as_dict()
    del fp64

    # Rebuilt after reconfiguring: a surface built under one precision keeps its
    # geometry parameters in that dtype, so configure-then-build is the only
    # order that works. Same constraint demo3's `build_doe_and_lens` documents.
    fp32 = build("fp32")
    fp32_record, fp32_timings = measured(fp32, "optiland_trace_precision_fp32")
    fp32_out = _host(fp32.lens.surfaces.trace(fp32.rays(fp32.wavelengths_scalar), skip=1))
    wavelength_m, rays, surfaces = fp32.wavelength_m, fp32.count, fp32.traced_surfaces
    prescription, fp32_execution = fp32.spec.name, fp32.execution.as_dict()
    del fp32

    fp64_again = build("fp64")
    fp64_repeat_record, fp64_repeat_timings = measured(
        fp64_again, "optiland_trace_precision_fp64_repeat"
    )
    del fp64_again

    noise = {
        key: abs(fp64_repeat_timings[key] - fp64_timings[key])
        / max(fp64_repeat_timings[key], fp64_timings[key])
        for key in fp64_timings
    }
    ratios = {}
    for key in fp32_timings:
        # The two fp64 realizations bracket the arm; the ratio is taken against
        # their mean so it does not inherit whichever one happened to be slower.
        fp64_mean = 0.5 * (fp64_timings[key] + fp64_repeat_timings[key])
        ratio = fp64_mean / fp32_timings[key]
        # One-sided on purpose. fp64 cannot be cheaper than fp32 for the same
        # computation, so a ratio below one is this measurement's noise showing
        # its size, not a speedup -- and publishing it as `0.97x` would put a
        # number in a record that a reader would quote. Only an excess above the
        # floor counts as resolved.
        resolved = (ratio - 1.0) > noise[key]
        ratios[key] = {
            "fp64_over_fp32": round(ratio, 4),
            "noise_floor": round(noise[key], 4),
            "resolved_above_noise": bool(resolved),
            "reading": (
                f"fp64 costs {ratio:.2f}x the fp32 arm"
                if resolved
                else "no measurable fp64 surcharge on this arm. The ratio is at or "
                "below one, which for the more expensive precision bounds the "
                "effect at the run-to-run spread rather than measuring a speedup"
            ),
        }

    lambda_mm = wavelength_m * 1e3
    radial_um = 1e3 * np.hypot(fp32_out["x"] - fp64_out["x"], fp32_out["y"] - fp64_out["y"])
    opd_waves = np.abs(fp32_out["opd"] - fp64_out["opd"]) / lambda_mm
    opd_rms_waves = float(np.sqrt(np.mean(opd_waves**2)))
    accuracy = {
        "oracle": (
            "the fp64 trace of the same rays through the same prescription. Not an "
            "analytic oracle: it bounds what fp32 loses relative to fp64, which is "
            "the question precision-as-a-speed-knob raises, and says nothing about "
            "whether either matches the physics."
        ),
        "sensor_position_error_um": {
            "rms": float(np.sqrt(np.mean(radial_um**2))),
            "max": float(radial_um.max()),
            "sensor_pitch_um": 4.2,
        },
        "optical_path_error_waves": {
            "rms": opd_rms_waves,
            "max": float(opd_waves.max()),
            "rms_degrees_of_phase": 360.0 * opd_rms_waves,
            "wavelength_m": wavelength_m,
        },
        "direction_cosine_error": {
            "max": float(
                max(np.abs(fp32_out[k] - fp64_out[k]).max() for k in ("L", "M", "N"))
            )
        },
        "why_the_opd_is_the_one_that_matters": (
            "demo3 accumulates a COHERENT field, so the traced optical path enters "
            "the result as a phase. A fraction-of-a-wave error is a phase error of "
            "2 pi times that fraction and it does not average away over rays the way "
            "a position error well inside one sensor pixel does."
        ),
    }

    _write(
        "optiland_trace_precision",
        {
            "probe": "perf_optiland_trace_precision",
            "issue": "CHE-118 (M5.1)",
            "environment": environment_fingerprint().as_dict(),
            "chunk": {
                "rays": rays,
                "traced_surfaces": surfaces,
                "prescription": prescription,
            },
            "optiland_execution": {"fp32": fp32_execution, "fp64": fp64_execution},
            "timings_s": {
                "fp32": fp32_timings,
                "fp64": fp64_timings,
                "fp64_repeat": fp64_repeat_timings,
            },
            "records": {
                "fp32": fp32_record,
                "fp64": fp64_record,
                "fp64_repeat": fp64_repeat_record,
            },
            "fp64_cost_ratio": ratios,
            "accuracy": accuracy,
            "finding": (
                "Precision barely moves this trace. On the arm that survives the "
                "CHE-118 fix the fp32 and fp64 costs are not separable at all -- the "
                "ratio sits inside the spread of two identical fp64 runs -- and even "
                "the per-ray-wavelength arm shows only "
                f"{ratios['solver_trace_per_ray_wavelength_s']['fp64_over_fp32']:.2f}x, "
                "nowhere near the ~32x this device's fp64:fp32 arithmetic throughput "
                "would suggest, because the trace is bound by kernel launches, "
                "memory traffic and host-side work rather than by arithmetic. What "
                "fp32 does cost is phase: "
                f"{opd_rms_waves:.2e} waves RMS of optical path against the fp64 "
                f"trace, i.e. about {360.0 * opd_rms_waves:.1f} degrees, with a "
                f"maximum of {accuracy['optical_path_error_waves']['max']:.2e} waves."
            ),
            "consequence": (
                "Precision is not this issue's lever in either direction, and the "
                "acceptance criterion's warning does not bite: demo3 already runs "
                "fp32, so there is no faster-but-different trace on offer here. The "
                "useful conclusion runs the other way. Before CHE-118 the trace was "
                "99 s and fp64 was a 12% surcharge on the largest stage in the run; "
                "after it the stage is a few seconds and fp64 is free within noise, "
                "so a future benchmark that needs a certified phase can simply pay "
                "for it. That matters because fp32's ~4 degrees RMS of traced phase "
                "is not obviously negligible in a coherent sum -- it is a "
                "pre-existing property of every demo3 record, not something this "
                "change introduces, and it is handed to the convergence work rather "
                "than resolved here."
            ),
        },
    )


# ---------------------------------------------------------------------------
# verify: the optimization changed the cost and nothing else
# ---------------------------------------------------------------------------


DEMO_RECORDS = ROOT / "benchmarks" / "probes" / "records" / "ray_wave"


def verify_demo3(before: str, after: str) -> None:
    """Assert two demo3 runs got faster and produced the same physics, in one artifact.

    An optimization that changes the answer is a different route, not a faster
    one -- the rule CHE-101 set for the k-space reconstruction, where the new
    route was accepted only after it was shown to agree with the old one. Here the
    claim is stronger and so is the check: the traced geometry is bitwise
    identical, so the reconstructed field should be too, and `array_equal` is what
    says so. `allclose` would pass on a real change and is the wrong test.

    The two halves of the acceptance criterion belong in one file. `compare`
    answers "did it get faster, on a comparable environment"; `array_equal` on the
    reconstructed field answers "and is it still the same result". Either alone is
    quotable and misleading: a speedup whose answer moved is a different route, and
    an unchanged answer that was not measured for cost is not an optimization. The
    timing records are embedded rather than referenced so this artifact carries its
    own environment fingerprint, memory report and swap evidence.

    Takes the DEMO record names (the `perf_`-prefixed scientific records the demo
    wrote); the matching harness timing records are the same names without the
    prefix, which is the convention `run_baselines._demo` establishes.

    Runs on the host: it reads committed files and compares them.
    """
    from core.performance import Incomparable, compare

    fields = {}
    ledgers = {}
    provenance: dict[str, dict[str, Any]] = {}
    for role, name in (("before", before), ("after", after)):
        payload = json.loads((DEMO_RECORDS / f"{name}.json").read_text())
        arrays = np.load(DEMO_RECORDS / f"{name}_fields.npz")
        fields[role] = {key: arrays[key] for key in arrays}
        run = payload["routes"]["rw_p"]["runs"][0]
        stamp = payload.get("record_provenance", {})
        provenance[role] = {
            "source_commit": stamp.get("source_commit"),
            "working_tree_dirty": stamp.get("working_tree_dirty"),
            "code_fingerprint": stamp.get("code_fingerprint", {}).get("combined_sha256"),
        }
        ledgers[role] = {
            "energy": run["energy"],
            "seeds": payload["configuration"]["seeds"],
            "prescription_fingerprint": payload["optical_system"][
                "prescription_fingerprint"
            ],
            "rays_per_chunk": payload["configuration"]["rays_per_chunk"],
            "batches": run["batches"],
            "peak_bytes_in_use": run["device_memory"]["peak_bytes_in_use"],
        }

    keys_before, keys_after = sorted(fields["before"]), sorted(fields["after"])
    if keys_before != keys_after:
        raise SystemExit(f"different fields recorded: {keys_before} vs {keys_after}")
    field_report = {}
    for key in keys_before:
        left, right = fields["before"][key], fields["after"][key]
        identical = bool(np.array_equal(left, right))
        field_report[key] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "bitwise_identical": identical,
            "max_abs_difference": (
                0.0 if identical else float(np.abs(left - right).max())
            ),
        }

    energy_report = {}
    for key in sorted(ledgers["before"]["energy"]):
        left = ledgers["before"]["energy"][key]
        right = ledgers["after"]["energy"][key]
        if isinstance(left, bool) or not isinstance(left, int | float):
            continue
        energy_report[key] = {"before": left, "after": right, "identical": left == right}

    unchanged = all(v["bitwise_identical"] for v in field_report.values()) and all(
        v["identical"] for v in energy_report.values()
    )
    timing = {}
    for role, name in (("before", before), ("after", after)):
        path = RECORDS / f"{name.removeprefix('perf_')}.json"
        if not path.exists():
            raise SystemExit(
                f"no harness timing record at {path}. This artifact is the cost "
                "claim and the physics claim together, so it will not be written "
                "with only half of it."
            )
        timing[role] = json.loads(path.read_text())
    try:
        speed = compare(timing["before"], timing["after"])
    except Incomparable as refusal:
        # Not swallowed into a field. If the two runs are not comparable then
        # there is no before-and-after here, and writing the physics half alone
        # under a name that promises both would be the misleading artifact.
        raise SystemExit(f"REFUSED TO COMPARE: {refusal}") from None

    payload = {
        "probe": "perf_optiland_trace_demo3_equivalence",
        "issue": "CHE-118 (M5.1)",
        "demo_records": {"before": before, "after": after},
        "speedup": speed,
        "timing_records": timing,
        "seeds_match": ledgers["before"]["seeds"] == ledgers["after"]["seeds"],
        "prescription_match": (
            ledgers["before"]["prescription_fingerprint"]
            == ledgers["after"]["prescription_fingerprint"]
        ),
        "chunking_match": (
            ledgers["before"]["rays_per_chunk"] == ledgers["after"]["rays_per_chunk"]
            and ledgers["before"]["batches"] == ledgers["after"]["batches"]
        ),
        "record_provenance_of_each_arm": {
            **provenance,
            "note": (
                "the commit and code fingerprint each arm was produced under. The "
                "BEFORE arm's demo record cannot stay in the tree: "
                "`tests/test_provenance_fingerprint.py` requires every stamped "
                "record to describe the current code, and the whole point of this "
                "issue is that the code changed. So the pre-fix record was "
                "regenerated under CHE-118 like the other stale records, and the "
                "before arm is reproducible by checking out the commit named here "
                "and re-running the demo. What is preserved in the tree instead is "
                "stronger and cheaper to check: `timing_records` below embeds the "
                "pre-fix measurement verbatim, and "
                "`tests/test_coherent_bridge.py::TestMonochromaticWavelengthHandoff` "
                "asserts the bitwise equality this artifact confirmed at 60 M rays "
                "on every test run, per chunk, against a reference trace built the "
                "old way."
            ),
        },
        "fields": field_report,
        "energy_ledger": energy_report,
        "peak_device_bytes": {
            "before": ledgers["before"]["peak_bytes_in_use"],
            "after": ledgers["after"]["peak_bytes_in_use"],
            "note": (
                "the only figure here that is allowed to move, and it moves by "
                "allocator bookkeeping rather than by the workload: the size-1 "
                "wavelength array is smaller than the per-ray one, so a larger peak "
                "is caching granularity in JAX's allocator, not more memory held."
            ),
        },
        "verdict": (
            f"{speed['speedup']:.3f}x faster end to end, and the result is "
            "unchanged: same field, bit for bit, and an identical energy ledger. "
            "The optimization changed the cost and nothing else."
            if unchanged
            else "CHANGED. This is not an optimization; treat it as a different "
            "route and validate it as one."
        ),
        "why_bitwise_and_not_a_tolerance": (
            "the two runs differ only in the SHAPE of the wavelength array handed "
            "to optiland, and optiland's materials return the same index either "
            "way, so every subsequent floating-point operation is the same "
            "operation on the same bits. A tolerance would hide a real change; "
            "exact equality is the claim and it is the one asserted."
        ),
    }
    _write("optiland_trace_demo3_equivalence", payload)
    print(payload["verdict"])
    print(f"  {before} -> {after}")
    if not unchanged:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("decompose", help="attribute the trace stage to named contributions")
    sub.add_parser("chunks", help="is chunk size a lever? fitted on both arms")
    sub.add_parser("precision", help="fp32 vs fp64, cost and accuracy")
    verify = sub.add_parser(
        "verify-demo3", help="assert two demo3 runs produced the same physics"
    )
    verify.add_argument("before", help="demo3 record name under benchmarks/probes/records/ray_wave")
    verify.add_argument("after")
    args = parser.parse_args()
    if args.command == "verify-demo3":
        verify_demo3(args.before, args.after)
        return 0
    {"decompose": profile_decompose, "chunks": profile_chunks, "precision": profile_precision}[
        args.command
    ]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
