"""Where demo3's 86.9 s patch-spectrum emitter goes, and what threading it buys.

    ./run.sh python benchmarks/perf/profile_patch_emitter.py decompose
    ./run.sh python benchmarks/perf/profile_patch_emitter.py threads
    ./run.sh python benchmarks/perf/profile_patch_emitter.py overlap

CHE-119 (M5.2). Host-side, no GPU: the emitter is NumPy on the CPU, which is the
whole premise of the issue.

What the profile found
----------------------
On a real 200 k-ray demo3 chunk -- 50 patches x 4000 secondary modes -- the stage
decomposes into

* **58%** the per-patch padded FFT, ``301 x 301`` complex128,
* **23%** the per-patch draw, ``Generator.choice(p=...)`` over 90601 modes,
* **4.2%** re-deriving two arrays that do not depend on the patch,
* **3.1%** a boolean-mask gather over a mask that is entirely true,
* and ~11% extraction, density, assembly and bundle construction.

None of it is a device-port problem. The FFT and the draw are both *independent
per patch*, numpy releases the GIL in both, and the two loop-invariant items are
simply waste. So the response is threading and hoisting, not CUDA -- and every
part of it is bitwise neutral, which is what lets a 2.7x speedup ship without
re-measuring a single committed number.

Both arms stay measurable from committed code
---------------------------------------------
``patch_secondary_rays`` now ships the threaded form, so a probe that only called
it could not decompose the 86.9 s it is compared against. :func:`_as_committed`
replicates the pre-CHE-119 loop -- per patch, boolean gathers, ``rng.choice``,
single-threaded -- and is **asserted bitwise against the shipped function on
every run**. That assertion does double duty: it stops this profiler drifting
from the code it profiles, and it independently re-checks the claim that CHE-119
changed no ray.

The one honest limitation: an in-place decomposition has to instrument a copy of
the loop, because timing the pieces standalone gives nonsense. Measured: piece
timings taken on pre-materialized inputs summed to **137%** of the call, because
a spectrum that the FFT has just written is hot in cache and one read from a list
of fifty is not. The replica is the price of a decomposition whose parts add up.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# demo3's own DOE loader and prescription, so the profiled patches are the
# patches the committed record emitted.
sys.path.insert(0, str(ROOT / "benchmarks" / "probes" / "ray_wave"))

import numpy as np  # noqa: E402

from core.performance import (  # noqa: E402
    PerformanceRecord,
    Workload,
    environment_fingerprint,
    measure,
)

RECORDS = Path(__file__).resolve().parent / "records"
SCHEMA_PATH = ROOT / "benchmarks" / "schemas" / "performance.schema.json"

#: The committed baseline this profile decomposes. Post-CHE-118 demo3, whose
#: emitter stage is unchanged from the pre-CHE-118 record -- M5.1 removed a
#: competitor for the run's wall clock, not a dependency of this stage.
COMMITTED = {
    # The post-CHE-118 timing, which is this issue's before-arm. NOT the
    # un-suffixed `demo3_characterization_rw_p_ramp_sum_cuda.json`: that one is
    # still the PRE-CHE-118 measurement, because M5.1 refreshed the demo's
    # scientific record by running the probe directly and never re-ran the
    # harness baseline. Reading it here gave a device total of 113.2 s -- M5.1's
    # 98.96 s trace, which no longer exists -- and the overlap section reported
    # it before this note existed.
    "record": "demo3_characterization_rw_p_ramp_sum_che118_after_cuda.json",
    "issue": "CHE-118 (M5.1), the run this issue is measured against",
    "emit_patch_spectra_s": 86.879,
    "total_s": 120.36345417899429,
    "chunks": 300,
    "rays_per_chunk": 200_000,
}

#: One demo3 `characterization` chunk exactly: 3000 patches over 60 groups is 50
#: a group, and 20000 secondary modes split into 5 parts is 4000 a patch.
CHUNK_PATCHES = 50
CHUNK_SECONDARY = 4_000

WAVELENGTH_M = 0.7e-6
PITCH_M = 6.3e-6

#: Wall-clock budget per timed arm, and the repeat bounds it picks between. Same
#: reasoning as `profile_optiland_trace.py`: the arms here span 8 ms to 300 ms,
#: and a fixed repeat count would either over-sample the slow ones or leave the
#: fast ones as a median of noise.
_ARM_BUDGET_S = 1.5
_MIN_REPEATS = 5
_MAX_REPEATS = 31


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


def _measured(
    produce: Any, *, label: str, workload: Workload, notes: list[str]
) -> tuple[dict[str, Any], Any]:
    """Run ``produce`` under `measure`, and return its validated record and result.

    Every section reports per-arm medians rather than `measure`'s own median,
    because the arms are different computations and one median over them is not a
    quantity. The record is still taken and still committed: it is what carries
    the environment fingerprint, the memory report and the swap evidence, and a
    committed performance artifact without those is what M0.4 exists to stop.
    """
    record, result = measure(
        lambda _timer: produce(),
        label=label,
        workload=workload,
        repeats=1,
        warmup=0,
        notes=[
            *notes,
            "Each arm inside this record is a median of repeats chosen from a "
            f"{_ARM_BUDGET_S:g} s budget ({_MIN_REPEATS}-{_MAX_REPEATS}). "
            "`measurement` times the whole arm group and is not the figure to "
            "quote -- read the per-arm seconds in the payload.",
        ],
    )
    return _validate(record), result


def _median_s(fn: Any) -> tuple[float, int, float]:
    """Median seconds, the repeat count, and the observed relative spread."""
    fn()
    started = time.perf_counter()
    fn()
    probe = time.perf_counter() - started
    repeats = int(min(_MAX_REPEATS, max(_MIN_REPEATS, _ARM_BUDGET_S / max(probe, 1e-6))))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    median = float(np.median(samples))
    spread = (max(samples) - min(samples)) / median if median else 0.0
    return median, repeats, spread


# ---------------------------------------------------------------------------
# The demo3 chunk, built through demo3's own code
# ---------------------------------------------------------------------------


class Demo3Patches:
    """demo3's DOE, plan and patch centres, and an emission of any size."""

    def __init__(self) -> None:
        from _demo_support import build_doe

        from core.boundary import ReferencePlane
        from couplers.patch import plan_patches

        self.doe = build_doe("demo3_smile_phase_profile.npy", pitch_m=PITCH_M)
        self.plane = ReferencePlane(name="doe", z_m=0.0)
        # demo3's own seed and plan, so these are the patches the record emitted.
        self.plan = plan_patches(
            grid_shape=self.doe.grid_shape,
            sample_pitch_m=self.doe.pitch_m,
            patch_px=101,
            pad_factor=2,
            patch_count=3_000,
            rng=np.random.default_rng(20260822),
        )
        self.centers = np.asarray(self.plan.centers_xy_m)

    def plan_for(self, patches: int) -> Any:
        return dataclasses.replace(self.plan, centers_xy_m=self.centers[:patches])

    def emit(self, patches: int, secondary: int | None, seed: int = 1234) -> Any:
        from couplers.patch import patch_secondary_rays

        return patch_secondary_rays(
            self.doe.transmission,
            plan=self.plan_for(patches),
            sample_pitch_m=self.doe.pitch_m,
            wavelength_m=WAVELENGTH_M,
            plane=self.plane,
            secondary_count=secondary,
            rng=None if secondary is None else np.random.default_rng(seed),
        )


def _as_committed(
    patches: Demo3Patches, n_patches: int, secondary: int, seed: int = 1234
) -> tuple[dict[str, float], Any]:
    """The pre-CHE-119 loop, instrumented in place, with its own timings.

    A deliberate copy of code that no longer exists, and the only way to
    decompose a stage whose committed number was produced by it. Kept honest by
    the caller, which asserts its output is bitwise the shipped function's.
    """
    from core.boundary import Frame, RayBundle
    from couplers.patch import extract_patch

    plan = patches.plan_for(n_patches)
    pad, patch_px = plan.pad_px, plan.patch_px
    pitch_y, pitch_x = (float(v) for v in patches.doe.pitch_m)
    plane = patches.plane
    field = patches.doe.transmission
    rng = np.random.default_rng(seed)

    timings = dict.fromkeys(
        (
            "spectral_grid_setup",
            "extract_patch_and_pad",
            "patch_fft",
            "mask_gather_spectrum",
            "mask_gather_directions",
            "density",
            "draw",
            "ray_assembly",
            "bundle_construction",
        ),
        0.0,
    )

    mark = time.perf_counter()
    fy = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_y))
    fx = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_x))
    grid_fx, grid_fy = np.meshgrid(fx, fy)
    dir_x = grid_fx * WAVELENGTH_M
    dir_y = grid_fy * WAVELENGTH_M
    propagating = (dir_x**2 + dir_y**2) < 1.0
    n_propagating = int(propagating.sum())
    timings["spectral_grid_setup"] += time.perf_counter() - mark

    off = (pad - patch_px) // 2
    positions: list[Any] = []
    directions: list[Any] = []
    amplitudes: list[Any] = []

    for center in np.asarray(plan.centers_xy_m, dtype=np.float64):
        mark = time.perf_counter()
        patch = extract_patch(
            field,
            center_xy_m=(float(center[0]), float(center[1])),
            patch_px=patch_px,
            sample_pitch_m=patches.doe.pitch_m,
        )
        padded = np.zeros((pad, pad), dtype=np.complex128)
        padded[off : off + patch_px, off : off + patch_px] = patch
        timings["extract_patch_and_pad"] += time.perf_counter() - mark

        mark = time.perf_counter()
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) / (pad * pad)
        timings["patch_fft"] += time.perf_counter() - mark

        mark = time.perf_counter()
        modal = spectrum[propagating]
        timings["mask_gather_spectrum"] += time.perf_counter() - mark

        # The two loop-invariant gathers, recomputed per patch exactly as the
        # pre-CHE-119 loop did. Timed separately because "waste" is a claim and
        # this is the number behind it.
        mark = time.perf_counter()
        du = dir_x[propagating]
        dv = dir_y[propagating]
        timings["mask_gather_directions"] += time.perf_counter() - mark

        mark = time.perf_counter()
        magnitude = np.abs(modal)
        total = float(magnitude.sum())
        density = (
            magnitude / total if total > 0.0 else np.full(n_propagating, 1.0 / n_propagating)
        )
        timings["density"] += time.perf_counter() - mark

        mark = time.perf_counter()
        picks = rng.choice(n_propagating, size=int(secondary), p=density)
        timings["draw"] += time.perf_counter() - mark

        mark = time.perf_counter()
        amplitudes.append(plan.coverage * modal[picks] / density[picks])
        normal = np.sqrt(np.clip(1.0 - (du[picks] ** 2 + dv[picks] ** 2), 0.0, None))
        directions.append(np.column_stack([du[picks], dv[picks], normal]))
        positions.append(
            np.column_stack(
                [
                    np.full(picks.size, center[0]),
                    np.full(picks.size, center[1]),
                    np.full(picks.size, plane.z_m),
                ]
            )
        )
        timings["ray_assembly"] += time.perf_counter() - mark

    mark = time.perf_counter()
    bundle = RayBundle(
        positions_m=np.concatenate(positions),
        directions=np.concatenate(directions),
        wavelength_m=WAVELENGTH_M,
        reference_plane=plane,
        frame=Frame(),
        amplitude=np.concatenate(amplitudes),
        optical_path_length_m=np.zeros(sum(p.shape[0] for p in positions)),
        optical_path_length_reference=(
            f"zero at the patch plane {plane.name!r}; each patch's own centre "
            "phase is carried in the amplitude, so the path restarts here"
        ),
        reconstruction_normalization="one_over_n",
    )
    timings["bundle_construction"] += time.perf_counter() - mark
    return timings, bundle


def _identical(left: Any, right: Any) -> bool:
    return all(
        np.array_equal(np.asarray(getattr(left, f)), np.asarray(getattr(right, f)))
        for f in ("positions_m", "directions", "amplitude", "optical_path_length_m")
    )


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def profile_decompose() -> None:
    """The 86.9 s, attributed to named contributions that sum to it."""
    import couplers.patch as patch_module

    patches = Demo3Patches()
    shipped_bundle, diagnostics = patches.emit(CHUNK_PATCHES, CHUNK_SECONDARY)

    # The replica is validated before it is trusted. If this fails, either the
    # replica has drifted from the code it profiles or CHE-119 changed a ray --
    # and both are reasons to stop rather than to publish a decomposition.
    replica_timings, replica_bundle = _as_committed(
        patches, CHUNK_PATCHES, CHUNK_SECONDARY
    )
    if not _identical(shipped_bundle, replica_bundle):
        raise SystemExit(
            "the as-committed replica does not reproduce the shipped emitter "
            "bitwise. Either this profiler has drifted from couplers/patch.py, or "
            "the shipped emitter has stopped being numerics-neutral. Both are "
            "reasons not to write a record."
        )

    # Best of a few realizations: these are host timings on a shared 80-core box
    # and the minimum total is the least interference-contaminated sample.
    for _ in range(4):
        candidate, _ = _as_committed(patches, CHUNK_PATCHES, CHUNK_SECONDARY)
        if sum(candidate.values()) < sum(replica_timings.values()):
            replica_timings = candidate

    def timed(threads: int) -> tuple[float, int, float]:
        import os

        previous = os.environ.get(patch_module.EMITTER_THREADS_ENV)
        os.environ[patch_module.EMITTER_THREADS_ENV] = str(threads)
        try:
            return _median_s(lambda: patches.emit(CHUNK_PATCHES, CHUNK_SECONDARY))
        finally:
            if previous is None:
                os.environ.pop(patch_module.EMITTER_THREADS_ENV, None)
            else:
                os.environ[patch_module.EMITTER_THREADS_ENV] = previous

    record, _ = measure(
        lambda _timer: patches.emit(CHUNK_PATCHES, CHUNK_SECONDARY),
        label=f"patch_emitter_{CHUNK_PATCHES}x{CHUNK_SECONDARY}_shipped",
        workload=Workload(
            size=CHUNK_PATCHES * CHUNK_SECONDARY,
            unit="secondary_ray",
            route="patch_wft_emitter",
            detail={
                "patches": CHUNK_PATCHES,
                "secondary_per_patch": CHUNK_SECONDARY,
                "pad_px": patches.plan.pad_px,
                "patch_px": patches.plan.patch_px,
                "threads": patch_module.emitter_threads(),
                "precision": "complex128",
                "device": "cpu",
            },
        ),
        repeats=1,
        warmup=0,
        notes=[
            "One emission at the shipped thread count. The arm medians in the "
            "payload are the figures to quote; this block is what carries the "
            "environment fingerprint and the memory guards.",
        ],
    )

    sequential_s, seq_repeats, seq_spread = timed(1)
    shipped_s, ship_repeats, ship_spread = timed(patch_module.PATCH_EMITTER_THREADS)
    accounted = sum(replica_timings.values())

    committed_per_chunk = COMMITTED["emit_patch_spectra_s"] / COMMITTED["chunks"]
    contributions = {
        name: {
            "seconds": round(value, 6),
            "fraction_of_accounted": round(value / accounted, 4),
            "x_committed_chunks_s": round(value * COMMITTED["chunks"], 2),
        }
        for name, value in replica_timings.items()
    }
    for name, what in {
        "patch_fft": (
            f"the padded per-patch transform, {patches.plan.pad_px}^2 complex128. "
            "Independent per patch and numpy releases the GIL in it, which is why "
            "threading is available at all"
        ),
        "draw": (
            "`Generator.choice(p=density)` over the patch's 90601 modes. Internally "
            "a cumulative sum over all of them plus a search for `secondary_count` "
            "uniforms, so it is O(modes) per patch and not O(picks)"
        ),
        "mask_gather_directions": (
            "PURE WASTE: `dir_x[propagating]` and `dir_y[propagating]` are "
            "functions of the padded grid and the wavelength alone, and were "
            "recomputed for every patch"
        ),
        "mask_gather_spectrum": (
            "a boolean-mask gather over a mask that is entirely true on this grid, "
            "so it returns exactly `np.ravel` for 780x the cost"
        ),
        "spectral_grid_setup": "once per call, not per patch",
        "extract_patch_and_pad": "the DOE window and its zero-padded frame",
        "density": "abs, sum and divide over the mode vector",
        "ray_assembly": "amplitude, direction cosines and launch positions per patch",
        "bundle_construction": "`RayBundle` and its finite/contract validation",
    }.items():
        contributions[name]["what"] = what

    waste_share = (
        contributions["mask_gather_directions"]["fraction_of_accounted"]
        + contributions["mask_gather_spectrum"]["fraction_of_accounted"]
    )
    _write(
        "patch_emitter_decomposition",
        {
            "probe": "perf_patch_emitter_decomposition",
            "issue": "CHE-119 (M5.2)",
            "environment": environment_fingerprint().as_dict(),
            "chunk": {
                "patches": CHUNK_PATCHES,
                "secondary_per_patch": CHUNK_SECONDARY,
                "emitted_rays": int(shipped_bundle.count),
                "pad_px": patches.plan.pad_px,
                "patch_px": patches.plan.patch_px,
                "coverage": patches.plan.coverage,
                "propagating_modes": diagnostics.propagating_modes,
                "evanescent_modes": diagnostics.evanescent_modes,
                "note": (
                    "one demo3 `characterization` chunk, from demo3's own patch "
                    "plan at demo3's own seed. Every mode propagates on this grid, "
                    "which is why the boolean mask is a full-array gather."
                ),
            },
            "arms_s": {
                "as_committed_replica": round(accounted, 6),
                "shipped_single_threaded": round(sequential_s, 6),
                "shipped_threaded": round(shipped_s, 6),
                "threads": patch_module.PATCH_EMITTER_THREADS,
                "repeats": {"sequential": seq_repeats, "threaded": ship_repeats},
                "relative_spread": {
                    "sequential": round(seq_spread, 4),
                    "threaded": round(ship_spread, 4),
                },
            },
            "speedup": {
                "threaded_over_single_threaded": round(sequential_s / shipped_s, 3),
                "threaded_over_as_committed": round(accounted / shipped_s, 3),
                "note": (
                    "two denominators because they answer different questions. "
                    "Against the single-threaded SHIPPED code the ratio isolates "
                    "what the threads bought; against the as-committed replica it "
                    "is what demo3 actually gains, and it includes the hoist and "
                    "the ravel as well."
                ),
            },
            "contributions": contributions,
            "accounted_s": round(accounted, 6),
            "replica_validated_bitwise_against_shipped": True,
            "closure_against_committed_record": {
                "committed_record": COMMITTED["record"],
                "committed_issue": COMMITTED["issue"],
                "committed_stage_s": COMMITTED["emit_patch_spectra_s"],
                "committed_chunks": COMMITTED["chunks"],
                "committed_s_per_chunk": round(committed_per_chunk, 6),
                "replica_s_per_chunk": round(accounted, 6),
                "ratio": round(accounted / committed_per_chunk, 4),
                "note": (
                    "the replica's instrumented total against the committed stage "
                    "divided by its chunk count. Agreement is what licenses reading "
                    "the fractions above as fractions of the 86.9 s rather than of "
                    "a probe's own workload. The replica is instrumented, so it is "
                    "expected to read slightly under the shipped call -- the "
                    "difference is the emitter's own function-call and argument "
                    "handling, which no region covers."
                ),
            },
            "finding": (
                f"Of {accounted * 1e3:.0f} ms a chunk, "
                f"{contributions['patch_fft']['fraction_of_accounted'] * 100:.0f}% is "
                "the padded FFT and "
                f"{contributions['draw']['fraction_of_accounted'] * 100:.0f}% is the "
                "spectral draw. Both are independent per patch and both release the "
                "GIL, so both thread. A further "
                f"{waste_share * 100:.1f}% was not work at all: two loop-invariant "
                "gathers recomputed per patch, and a boolean mask that is entirely "
                "true."
            ),
            "consequence": (
                "The stage is not FFT-flop-bound in the way that would justify a "
                "device port, and it is not dominated by per-patch Python overhead "
                "either. It is embarrassingly parallel work being done on one core, "
                "plus measurable waste. Threading and hoisting take it "
                f"{accounted / shipped_s:.2f}x faster with every emitted ray bitwise "
                "unchanged, so no capability declaration widens, no estimator "
                "changes, and no committed number needs re-measuring."
            ),
            "record": _validate(record),
        },
    )
    print(
        f"as-committed {accounted * 1e3:.1f} ms | single-threaded {sequential_s * 1e3:.1f} ms "
        f"| threaded {shipped_s * 1e3:.1f} ms | {accounted / shipped_s:.2f}x"
    )


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------


def profile_threads() -> None:
    """The thread count and the pool gate, both by measurement."""
    import os

    import couplers.patch as patch_module

    patches = Demo3Patches()
    rows: list[dict[str, Any]] = []
    for n_patches in (1, 2, 4, 8, 16, 32, CHUNK_PATCHES):
        row: dict[str, Any] = {"patches": n_patches, "seconds_by_threads": {}}

        def sweep(n_patches: int = n_patches) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for threads in (1, 2, 4, 8, 16, 24):
                os.environ[patch_module.EMITTER_THREADS_ENV] = str(threads)
                try:
                    median, repeats, spread = _median_s(
                        lambda n=n_patches: patches.emit(n, CHUNK_SECONDARY)
                    )
                finally:
                    os.environ.pop(patch_module.EMITTER_THREADS_ENV, None)
                out[str(threads)] = {
                    "seconds": round(median, 6),
                    "repeats": repeats,
                    "relative_spread": round(spread, 4),
                }
            return out

        def forced(n_patches: int = n_patches) -> dict[str, Any]:
            """The pool ON regardless of ``POOL_MIN_PATCHES``.

            Without this the sweep cannot see the crossover it exists to
            establish: below the gate the shipped code does not create a pool, so
            its "threaded" and "serial" arms are the same computation and differ
            only by noise -- the first realization of this section reported a
            crossover of 1 patch for exactly that reason. Forcing the pool on is
            what measures the penalty the gate avoids.
            """
            previous = patch_module.POOL_MIN_PATCHES
            patch_module.POOL_MIN_PATCHES = 1
            os.environ[patch_module.EMITTER_THREADS_ENV] = str(
                patch_module.PATCH_EMITTER_THREADS
            )
            try:
                median, repeats, spread = _median_s(
                    lambda: patches.emit(n_patches, CHUNK_SECONDARY)
                )
            finally:
                patch_module.POOL_MIN_PATCHES = previous
                os.environ.pop(patch_module.EMITTER_THREADS_ENV, None)
            return {
                "seconds": round(median, 6),
                "repeats": repeats,
                "relative_spread": round(spread, 4),
            }

        row["record"], row["seconds_by_threads"] = _measured(
            sweep,
            label=f"patch_emitter_threads_{n_patches}patches",
            workload=Workload(
                size=n_patches * CHUNK_SECONDARY,
                unit="secondary_ray",
                route="patch_wft_emitter",
                detail={
                    "patches": n_patches,
                    "secondary_per_patch": CHUNK_SECONDARY,
                    "pad_px": patches.plan.pad_px,
                    "arms": "thread counts 1, 2, 4, 8, 16, 24",
                },
            ),
            notes=["One patch count, every thread count."],
        )
        row["forced_pool_at_default_threads"] = forced()
        serial = row["seconds_by_threads"]["1"]["seconds"]
        best = min(
            row["seconds_by_threads"].items(), key=lambda kv: kv[1]["seconds"]
        )
        row["best_threads"] = int(best[0])
        row["best_speedup_over_serial"] = round(serial / best[1]["seconds"], 3)
        row["ms_per_patch_at_best"] = round(best[1]["seconds"] / n_patches * 1e3, 3)
        # Which thread counts are indistinguishable from the best one, given the
        # best arm's own run-to-run spread. Reported instead of leaning on the
        # argmax: across realizations of this sweep the winner at 50 patches
        # flips between 8 and 16, and publishing whichever won today as "the
        # optimum" would be publishing noise.
        tolerance = best[1]["seconds"] * (1.0 + max(best[1]["relative_spread"], 0.05))
        row["threads_within_noise_of_best"] = sorted(
            int(k) for k, v in row["seconds_by_threads"].items()
            if v["seconds"] <= tolerance
        )
        rows.append(row)
        print(
            f"  patches={n_patches:3d}  serial {serial * 1e3:7.1f} ms  "
            f"best t={row['best_threads']:<2d} {best[1]['seconds'] * 1e3:7.1f} ms  "
            f"{row['best_speedup_over_serial']:.2f}x"
        )

    chunk_row = next(r for r in rows if r["patches"] == CHUNK_PATCHES)
    # The smallest patch count at which a FORCED pool beats the serial path. Read
    # off the forced arm, not the shipped one: below the gate the shipped code IS
    # the serial path, so comparing them there measures nothing.
    crossover = None
    for row in rows:
        serial = row["seconds_by_threads"]["1"]["seconds"]
        forced_s = row["forced_pool_at_default_threads"]["seconds"]
        row["forced_over_serial"] = round(forced_s / serial, 3)
        row["gate_is_active"] = row["patches"] < patch_module.POOL_MIN_PATCHES
        row["gate_saves_s"] = (
            round(forced_s - serial, 6) if row["gate_is_active"] else None
        )
        if crossover is None and forced_s < serial:
            crossover = row["patches"]

    _write(
        "patch_emitter_thread_sweep",
        {
            "probe": "perf_patch_emitter_thread_sweep",
            "issue": "CHE-119 (M5.2)",
            "environment": environment_fingerprint().as_dict(),
            "held_fixed": {
                "secondary_per_patch": CHUNK_SECONDARY,
                "pad_px": patches.plan.pad_px,
                "patch_px": patches.plan.patch_px,
                "precision": "complex128",
                "device": "cpu",
            },
            "rows": rows,
            "shipped_constants": {
                "PATCH_EMITTER_THREADS": patch_module.PATCH_EMITTER_THREADS,
                "POOL_MIN_PATCHES": patch_module.POOL_MIN_PATCHES,
                "logical_cpus": os.cpu_count(),
                "note": (
                    "the thread count is a code constant rather than a runtime "
                    "choice, so it lands in the code fingerprint; "
                    f"{patch_module.EMITTER_THREADS_ENV} overrides it and is "
                    "registered in core.performance's thread variables so an "
                    "override lands in the performance fingerprint too."
                ),
            },
            "measured_crossover_patches": crossover,
            "finding": (
                f"At demo3's {CHUNK_PATCHES}-patch chunk the speedup plateaus at "
                f"{chunk_row['best_speedup_over_serial']:.2f}x, and the thread counts "
                "indistinguishable from the best one are "
                f"{chunk_row['threads_within_noise_of_best']} -- so the argmax is not "
                "a measurement, the plateau is. Past the plateau more threads make it "
                "worse: this workload becomes memory-bandwidth-bound before it runs "
                "out of cores. Below "
                f"{crossover} patches a FORCED pool is a LOSS -- the penalty is "
                "roughly fixed per call and does not shrink with the work, which "
                "points at numpy's per-thread FFT plan cache (pad 301 has a prime "
                "factor of 43, so a fresh worker thread builds its Bluestein tables "
                "from scratch). That is what POOL_MIN_PATCHES avoids, and "
                "`gate_saves_s` per row is what it saves."
            ),
            "consequence": (
                "Two constants. Eight threads is chosen from the plateau rather than "
                "read off an argmax -- 8 and 16 are not separable here and 8 is the "
                "smaller footprint, which is the tie-break AGENTS.md asks for on a "
                "shared box. And a pool only above "
                f"{patch_module.POOL_MIN_PATCHES} patches. The gate matters for "
                "correctness of the *cost* claim rather than of the result -- both "
                "paths emit the same rays bitwise -- but without it the exactness "
                "tests and any single-patch caller would run 2.6x slower than "
                "before, which is a regression paid for by a stage they do not use. "
                "Eight is also a deliberately small fraction of this shared box's 80 "
                "cores, per AGENTS.md."
            ),
        },
    )


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------


def profile_overlap() -> None:
    """How much of demo3 is host and device work waiting for each other.

    Read from the committed stage timers rather than re-run: the question is
    which stages execute on which processor, and that is a property of the code,
    not something a new measurement would refine.
    """
    committed = json.loads((RECORDS / COMMITTED["record"]).read_text())
    stages = committed["stages"]["seconds"]
    total = committed["measurement"]["median_s"]

    #: Where each demo3 stage executes. `emit_patch_spectra` is NumPy on the host
    #: (and, after CHE-119, on host threads); the rest run on the GPU or are
    #: dominated by a transfer to it.
    placement = {
        "emit_patch_spectra": "host",
        "host_to_device": "transfer",
        "optiland_trace": "device",
        "power_bookkeeping": "host",
        "reconstruct": "device",
    }
    host_s = sum(v for k, v in stages.items() if placement.get(k) == "host")
    device_s = sum(v for k, v in stages.items() if placement.get(k) == "device")
    transfer_s = sum(v for k, v in stages.items() if placement.get(k) == "transfer")

    # After CHE-119 the host stage shrinks and the device stages do not, so the
    # overlap prize is recomputed at the post-change emitter cost.
    import couplers.patch as patch_module

    patches = Demo3Patches()
    emitter_record, (shipped_s, _, _) = _measured(
        lambda: _median_s(lambda: patches.emit(CHUNK_PATCHES, CHUNK_SECONDARY)),
        label="patch_emitter_post_che119_chunk",
        workload=Workload(
            size=CHUNK_PATCHES * CHUNK_SECONDARY,
            unit="secondary_ray",
            route="patch_wft_emitter",
            detail={
                "patches": CHUNK_PATCHES,
                "secondary_per_patch": CHUNK_SECONDARY,
                "pad_px": patches.plan.pad_px,
                "threads": patch_module.emitter_threads(),
            },
        ),
        notes=["The post-CHE-119 emitter cost the overlap arithmetic extrapolates."],
    )
    emitter_after_s = shipped_s * COMMITTED["chunks"]
    host_after_s = host_s - stages["emit_patch_spectra"] + emitter_after_s

    def prize(host: float, device: float) -> float:
        """Best case for perfect double-buffering: the smaller side disappears."""
        return min(host, device)

    projected_total_s = total - stages["emit_patch_spectra"] + emitter_after_s
    prize_after_share = prize(host_after_s, device_s) / projected_total_s

    _write(
        "patch_emitter_overlap",
        {
            "probe": "perf_patch_emitter_overlap",
            "issue": "CHE-119 (M5.2)",
            "environment": environment_fingerprint().as_dict(),
            "source": {
                "record": COMMITTED["record"],
                "note": (
                    "stage seconds from the committed post-CHE-118 baseline. The "
                    "emitter's post-CHE-119 cost is measured here and extrapolated "
                    "over the same chunk count."
                ),
            },
            "stage_placement": placement,
            "emitter_measurement": emitter_record,
            "before_che119_s": {
                "host": round(host_s, 3),
                "device": round(device_s, 3),
                "transfer": round(transfer_s, 3),
                "total_command": round(total, 3),
                "serialization_prize": round(prize(host_s, device_s), 3),
                "serialization_prize_fraction_of_run": round(
                    prize(host_s, device_s) / total, 4
                ),
            },
            "after_che119_s": {
                "emitter": round(emitter_after_s, 3),
                "host": round(host_after_s, 3),
                "device": round(device_s, 3),
                "transfer": round(transfer_s, 3),
                "serialization_prize": round(prize(host_after_s, device_s), 3),
                "projected_total_command": round(projected_total_s, 3),
                "serialization_prize_fraction_of_run": round(prize_after_share, 4),
                "note": (
                    "the prize is an UPPER bound and a generous one: it assumes "
                    "perfect double-buffering with zero scheduling cost, and it "
                    "assumes the host threads and the device queue do not contend "
                    "for the transfer. Treat it as the ceiling on what pipelining "
                    "could return, not as a forecast."
                ),
            },
            "peak_memory_cost": {
                "chunks_in_flight": 2,
                "what_doubles": (
                    "one chunk's rays and its padded patch buffers, held while the "
                    "previous chunk is still on the device. At demo3's chunk that is "
                    f"{CHUNK_PATCHES} padded patches "
                    f"({2 * CHUNK_PATCHES * patches.plan.pad_px ** 2 * 16 / 1e6:.0f} MB "
                    "for buffers) plus 200 000 rays of positions, directions, "
                    "amplitude and path."
                ),
                "measured_peak_child_rss_bytes": committed.get("subprocess", {}).get(
                    "peak_child_rss_bytes"
                ),
                "device_peak_note": (
                    "the device side is bounded by the reconstruction's accumulation "
                    "buffer, which is per-run rather than per-chunk, so pipelining "
                    "would add a second chunk's rays rather than a second sensor."
                ),
            },
            "finding": (
                f"Before CHE-119, {host_s:.1f} s of the {total:.1f} s run is host "
                f"work and {device_s:.1f} s is device work, strictly serialized in "
                "the chunk loop, so perfect pipelining could hide at most "
                f"{prize(host_s, device_s):.1f} s -- "
                f"{prize(host_s, device_s) / total * 100:.0f}% of the run. After "
                f"CHE-119 the emitter falls to {emitter_after_s:.1f} s and the two "
                f"sides become {host_after_s:.1f} s host against {device_s:.1f} s "
                f"device, so the ceiling is {prize(host_after_s, device_s):.1f} s."
            ),
            "consequence": (
                "Deferred, and the honest framing is that the prize GREW rather than "
                "shrank. It is 17% of the run before this issue and about "
                f"{prize_after_share * 100:.0f}% after it, because shrinking the "
                "host side moved the two sides closer together and it is the smaller "
                "side that a pipeline hides. "
                "So this is not 'too small to bother with'; it is out of scope for "
                "three reasons that are about ownership rather than size. It is a "
                "structural change to demo3's chunk loop, i.e. to a benchmark probe, "
                "when the thing that should own an execution schedule is M3.1's "
                "executor. It doubles the in-flight memory, so it needs its own "
                "envelope measurement rather than an argument. And it would put host "
                "threads and a device queue live at once, which AGENTS.md's rule on "
                "concurrent GPU work asks to be justified with a measurement. "
                "Recorded here as a quantified, deferred option so the next issue "
                "starts from the number rather than from the idea."
            ),
        },
    )
    print(
        f"host {host_s:.1f} s / device {device_s:.1f} s before; "
        f"{host_after_s:.1f} s / {device_s:.1f} s after; "
        f"overlap ceiling {prize(host_after_s, device_s):.1f} s"
    )
    del patch_module


# ---------------------------------------------------------------------------
# cost: the calibrated model estimate() plans against
# ---------------------------------------------------------------------------

#: The (patches, secondary) grid the cost model is fitted on. Two axes because
#: the stage has two terms with different scaling: the transform and the density
#: are O(patches x pad^2) and independent of the draw size, while the search is
#: O(patches x secondary log pad^2). Fitting one axis would attribute all of it
#: to whichever was varied.
COST_GRID = (
    (4, 4_000),
    (8, 1_000),
    (8, 4_000),
    (16, 2_000),
    (16, 4_000),
    (32, 2_000),
    (32, 4_000),
    (50, 1_000),
    (50, 4_000),
    (50, 10_000),
    (50, 20_000),
)

#: Patch counts the fit is taken over. Below this the thread pool is either off
#: (under `POOL_MIN_PATCHES`) or freshly on and still paying its per-call FFT
#: plan-cache penalty, so one affine form spans two regimes and fits neither: the
#: excluded points' residuals are reported as the evidence for the boundary,
#: exactly as `profile_optiland_trace.py` does for its 4 M-ray point.
COST_MIN_PATCHES = 16


def profile_cost() -> None:
    """Fit ``fixed + patches * (per_patch + per_secondary * S)`` at the shipped threads.

    Least squares on both axes, with residuals reported, because a model whose
    error is not stated is a number a planner will quote. The form is the one the
    decomposition implies rather than a curve chosen to fit: a per-patch term for
    the transform, the density and the cumulative sum, which do not depend on how
    many modes are drawn, and a per-secondary-ray term for the search and the
    assembly, which do.
    """
    import couplers.patch as patch_module

    patches = Demo3Patches()
    rows: list[dict[str, Any]] = []
    for n_patches, secondary in COST_GRID:
        point_record, (median, repeats, spread) = _measured(
            lambda n=n_patches, s=secondary: _median_s(lambda: patches.emit(n, s)),
            label=f"patch_emitter_cost_{n_patches}x{secondary}",
            workload=Workload(
                size=n_patches * secondary,
                unit="secondary_ray",
                route="patch_wft_emitter",
                detail={
                    "patches": n_patches,
                    "secondary_per_patch": secondary,
                    "pad_px": patches.plan.pad_px,
                    "threads": patch_module.emitter_threads(),
                },
            ),
            notes=["One point of the two-axis cost grid."],
        )
        rows.append(
            {
                "record": point_record,
                "patches": n_patches,
                "secondary_per_patch": secondary,
                "emitted_rays": n_patches * secondary,
                "seconds": round(median, 6),
                "repeats": repeats,
                "relative_spread": round(spread, 4),
                "threaded": n_patches >= patch_module.POOL_MIN_PATCHES,
            }
        )
        print(
            f"  patches={n_patches:3d} S={secondary:6d}  {median * 1e3:8.2f} ms  "
            f"spread {spread:.3f}"
        )

    inside = [r for r in rows if r["patches"] >= COST_MIN_PATCHES]
    design = np.array(
        [[1.0, float(r["patches"]), float(r["emitted_rays"])] for r in inside]
    )
    observed = np.array([r["seconds"] for r in inside])
    (fixed_s, per_patch_s, per_ray_s), *_ = np.linalg.lstsq(design, observed, rcond=None)

    def predict(n_patches: int, secondary: int) -> float:
        return float(fixed_s + per_patch_s * n_patches + per_ray_s * n_patches * secondary)

    for row in rows:
        predicted = predict(row["patches"], row["secondary_per_patch"])
        row["predicted_s"] = round(predicted, 6)
        row["relative_error"] = round(predicted / row["seconds"] - 1.0, 4)
        row["in_domain"] = row["patches"] >= COST_MIN_PATCHES
    worst = max(abs(r["relative_error"]) for r in rows if r["in_domain"])
    worst_excluded = max(
        (abs(r["relative_error"]) for r in rows if not r["in_domain"]), default=0.0
    )

    _write(
        "patch_emitter_cost_model",
        {
            "probe": "perf_patch_emitter_cost_model",
            "issue": "CHE-119 (M5.2)",
            "environment": environment_fingerprint().as_dict(),
            "held_fixed": {
                "pad_px": patches.plan.pad_px,
                "patch_px": patches.plan.patch_px,
                "threads": patch_module.emitter_threads(),
                "precision": "complex128",
                "device": "cpu",
            },
            "form": (
                "seconds = fixed_s + per_patch_s * patches "
                "+ per_secondary_ray_s * patches * secondary_count"
            ),
            "fixed_s": float(fixed_s),
            "per_patch_s": float(per_patch_s),
            "per_secondary_ray_s": float(per_ray_s),
            "max_relative_error_in_domain": round(worst, 4),
            "max_relative_error_excluded": round(worst_excluded, 4),
            "rows": rows,
            "domain": {
                "patches": [COST_MIN_PATCHES, COST_GRID[-1][0]],
                "secondary_per_patch": [1_000, 20_000],
                "why_small_patch_counts_are_excluded": (
                    f"below {COST_MIN_PATCHES} patches the model is off by up to "
                    f"{worst_excluded * 100:.0f}%, because the thread pool switches "
                    "on at POOL_MIN_PATCHES and its per-call cost has not amortized "
                    "yet. That is two regimes, and one affine form describes neither. "
                    "Callers below the boundary get no prediction rather than a "
                    "wrong one."
                ),
                "pad_px": patches.plan.pad_px,
                "note": (
                    "pad is HELD, not fitted. The per-patch term is O(pad^2 log pad) "
                    "and every point here shares one pad, so the constant absorbs it "
                    "-- the model must not be used across a different patch size. "
                    "That is a real limit and it is stated rather than papered over: "
                    "a pad sweep is the follow-up that would remove it."
                ),
            },
            "finding": (
                f"The stage is {per_patch_s * 1e3:.2f} ms a patch plus "
                f"{per_ray_s * 1e9:.1f} ns a secondary ray, with a "
                f"{fixed_s * 1e3:.2f} ms fixed cost, holding to "
                f"{worst * 100:.1f}% for {COST_MIN_PATCHES}+ patches. The per-patch "
                "term dominates at "
                "demo3's 4000 modes a patch: 50 patches cost "
                f"{per_patch_s * 50 * 1e3:.0f} ms of transform against "
                f"{per_ray_s * 50 * 4000 * 1e3:.0f} ms of draw and assembly."
            ),
            "consequence": (
                "Wired into `PatchWftCoupler.estimate()`, which previously reported "
                "no time at all. Bound to this environment fingerprint and this pad, "
                "and refusing rather than extrapolating off either -- the same rule "
                "`core.performance.compare` applies to a ratio."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("decompose", help="attribute the emitter stage")
    sub.add_parser("threads", help="thread count and pool gate, by measurement")
    sub.add_parser("overlap", help="quantify the emitter/trace serialization")
    sub.add_parser("cost", help="fit the cost model estimate() plans against")
    args = parser.parse_args()
    {
        "decompose": profile_decompose,
        "threads": profile_threads,
        "overlap": profile_overlap,
        "cost": profile_cost,
    }[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
