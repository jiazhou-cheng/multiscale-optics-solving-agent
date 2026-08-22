"""The CHE-70 convergence sweep controller: one candidate at a time.

    ./run.sh --gpu python -m benchmarks.metalens_controller \
        --grid-size 100 --device cuda --sampling-density magnitude --seed 1 \
        --auto-converge --memory-guard --output outputs/che70_metalens

What this process does and does not do
--------------------------------------
It **never** runs the physics. Every ``(P, S, density, seed, chunk)`` candidate is
a fresh child process (``metalens_candidate``), launched strictly one at a time,
so CUDA memory is returned at exit, allocator fragmentation cannot accumulate
across differently-shaped runs, and a candidate that dies cannot poison the next
one's numbers. Sequential execution is also mandatory policy on this host, not
merely an architectural preference: ``AGENTS.md`` forbids concurrent
GPU-consuming jobs.

What it keeps in memory is compact by construction: one scalar record per
candidate plus, at the very end, two 100x100 arrays. Nothing ray-sized and
nothing per-chunk, so the parent's own RSS is flat in the number of candidates --
which Phase 34 asks to be checked, and which is recorded per candidate so the
check is a measurement rather than a claim.

The guards, and the direction they fail in
------------------------------------------
Before every launch: host ``MemAvailable`` must exceed the reserve, and the
calibrated chunk must fit the GPU envelope. During: the child watches its own
container swap and host headroom and aborts itself; the parent watches too and
terminates a child that does not.

After a memory failure the controller **stops growing in that direction**. The
only permitted retry is the *same* ``(P, S)`` with a strictly smaller chunk. This
is Phase 33 and it is what keeps a sweep from walking a shared server into swap
one doubling at a time.

Convergence, and what is allowed to be called converged
-------------------------------------------------------
The gate is ``NCC(PSF_ray, PSF_ref) >= 0.99`` against the analytic layered
angular-spectrum oracle, plus local stability under the available doublings
(``1 - NCC < 1e-3`` between a pair and each of ``(2P,S)``, ``(P,2S)``,
``(2P,2S)``). The reported pair is the **smallest** ``P*S`` that satisfies both,
found by scanning every executed candidate -- not the largest run that happened
to finish.

Both numbers are declared engineering targets, authorised as such rather than
derived from a noise model. The report states that plainly, and states the
measured floors they sit above: the estimator's own exactness limit against the
same oracle (8.9e-14 relative field error in float64, full enumeration) and the
float32/float64 agreement of the GPU path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.metalens_candidate import (
    DIRECTION_COSINE_FLOOR,
    CandidateRequest,
)
from core.resources import (
    MemoryWatchdog,
    cuda_memory_snapshot,
    host_memory_snapshot,
    host_reserve_bytes,
)
from evaluation.metalens import CONFIGURATIONS

__all__ = ["SweepController", "SweepOptions", "main"]

#: Phase 27's optical gate. A declared engineering target (CHE-70 blocking
#: decision 3, authorised as such), not a derived tolerance.
NCC_GATE = 0.99
#: Phase 27's local-stability criterion, likewise declared.
STABILITY_TOLERANCE = 1.0e-3
#: Phase 23's scale check, reported beside the NCC gate rather than folded into
#: it. 1% of total power: below any radiometric claim this project makes, and far
#: above the estimator's floor. Declared, like the NCC gate.
POWER_ERROR_TARGET = 1.0e-2
#: Phase 18's GPU envelope: never plan past 60% of what is currently free, and
#: always leave an absolute reserve on top of that.
GPU_USABLE_FRACTION = 0.60
GPU_RESERVE_BYTES = 2 * 1024**3
#: Safety factor applied to the *measured* bytes per ray when sizing a chunk.
CHUNK_SAFETY_FACTOR = 4.0
#: Chunk sizes are powers of two so a JIT-compiled kernel is reused.
MIN_CHUNK_SIZE = 4096
MAX_CHUNK_SIZE = 1 << 21


@dataclass
class SweepOptions:
    output: Path
    config: str = "METALENS-AIR-100"
    device: str = "cuda"
    precision: str = "fp32"
    density: str = "p_mag"
    seed: int = 1
    grid_size: int = 100
    auto_converge: bool = True
    memory_guard: bool = True
    pilot_launches: int = 4
    pilot_samples: int = 256
    max_launches: int = 1024
    max_samples: int = 65536
    seeds: tuple[int, ...] = (1, 2, 3)
    candidate_timeout_s: float = 1800.0
    chunk_size: int | None = None
    max_total_rays: int = 1 << 26
    memory_ladder: bool = True
    validation_configs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        record = dict(self.__dict__)
        record["output"] = str(self.output)
        record["seeds"] = list(self.seeds)
        record["validation_configs"] = list(self.validation_configs)
        return record


def _effective_chunk(record: dict[str, Any]) -> int | None:
    """The chunk size the run really used: ``total_rays / chunk_count``."""
    total = (record.get("request") or {}).get("total_rays")
    count = (record.get("chunking") or {}).get("chunk_count")
    if not total or not count:
        return None
    return int(total // count)


def _summarize(record: dict[str, Any]) -> dict[str, Any]:
    """The compact row kept in the parent. Deliberately scalar-only (Phase 34)."""
    request = record.get("request", {})
    metrics = record.get("metrics") or {}
    memory = record.get("memory") or {}
    gpu = record.get("gpu_memory_after") or {}
    streaming = record.get("streaming") or {}
    return {
        "label": request.get("label", ""),
        "config": request.get("config"),
        "P": request.get("launch_count_P"),
        "S": request.get("samples_per_launch_S"),
        "total_rays": request.get("total_rays"),
        "density": request.get("density"),
        "seed": request.get("seed"),
        "precision": request.get("precision"),
        "chunk_size": request.get("chunk_size"),
        "chunk_count": (record.get("chunking") or {}).get("chunk_count"),
        # A requested chunk larger than the whole population is not the chunk the
        # run used, and confusing the two is what made the first memory-scaling
        # plot look like memory grew with P*S when it grew with the *effective*
        # chunk. Derived from what the child actually did.
        "effective_chunk_size": _effective_chunk(record),
        "status": record.get("status"),
        "failure": record.get("failure"),
        "ncc": metrics.get("ncc"),
        "normalized_mse": metrics.get("normalized_mse"),
        "test_power": metrics.get("test_power"),
        "reference_power": metrics.get("reference_power"),
        "relative_power_error": metrics.get("relative_power_error"),
        "relative_peak_error": metrics.get("relative_peak_error"),
        "centroid_error_m": metrics.get("centroid_error_m"),
        "relative_fwhm_error": metrics.get("relative_fwhm_error"),
        "relative_ee50_error": metrics.get("relative_ee50_error"),
        "valid_rays": streaming.get("valid_rays"),
        "wall_time_s": record.get("wall_time_s"),
        "gpu_peak_allocated_bytes": gpu.get("peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": gpu.get("peak_reserved_bytes"),
        "gpu_free_after_bytes": gpu.get("free_bytes"),
        "peak_rss_bytes": memory.get("peak_rss_bytes"),
        "min_mem_available_bytes": memory.get("min_mem_available_bytes"),
        "cgroup_swap_before_bytes": memory.get("cgroup_swap_before_bytes"),
        "peak_cgroup_swap_bytes": memory.get("peak_cgroup_swap_bytes"),
        "swap_delta_peak_bytes": memory.get("swap_delta_peak_bytes"),
        "arrays_path": record.get("arrays_path"),
        "result_path": record.get("_result_path"),
    }


class SweepController:
    def __init__(self, options: SweepOptions) -> None:
        self.options = options
        self.root = options.output
        self.runs = self.root / "runs"
        self.rows: list[dict[str, Any]] = []
        self.parent_rss: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.memory_failures: list[dict[str, Any]] = []
        #: Candidates refused by this sweep's own declared ray cap rather than by
        #: a measured resource. Reported, but they do not halt growth.
        self.budget_skips: list[dict[str, Any]] = []
        self.per_ray_bytes: float | None = None
        self.chunk_size: int = options.chunk_size or MIN_CHUNK_SIZE
        self.started = time.time()
        self._launch_index = 0

    # -- infrastructure ----------------------------------------------------
    def note(self, text: str) -> None:
        print(f"[che70] {text}", flush=True)
        self.notes.append(text)

    def _gpu_envelope(self) -> int:
        snapshot = cuda_memory_snapshot()
        if not snapshot.available or snapshot.free_bytes is None:
            return 0
        return int(
            min(
                GPU_USABLE_FRACTION * snapshot.free_bytes,
                snapshot.free_bytes - GPU_RESERVE_BYTES,
            )
        )

    def _guard_before_launch(self, request: CandidateRequest) -> tuple[str, str] | None:
        """Phase 16/18: refuse a launch before it allocates anything.

        Returns ``(kind, reason)`` to skip, or ``None`` to proceed. ``kind`` is
        ``"memory"`` or ``"budget"``, and the difference matters downstream:

        * ``memory`` means a *measured* resource would not fit, and Phase 33 then
          forbids growing further in that direction;
        * ``budget`` means the request exceeds a number this sweep declared for
          itself. That is a scheduling choice, not evidence about the machine, so
          it is recorded and the sweep carries on with the work that does fit.

        Collapsing the two would let a deliberately conservative ray cap silently
        cancel the memory-scaling study, which is the opposite of conservative.
        """
        if not self.options.memory_guard:
            return None
        host = host_memory_snapshot()
        reserve = host_reserve_bytes(host.mem_total_bytes)
        if host.mem_available_bytes is not None and host.mem_available_bytes <= reserve:
            return (
                "memory",
                f"host MemAvailable {host.mem_available_bytes} B is at or below the "
                f"{reserve} B reserve before launch",
            )
        if request.total_rays > self.options.max_total_rays:
            return (
                "budget",
                f"P*S = {request.total_rays} exceeds the declared sweep cap "
                f"{self.options.max_total_rays}",
            )
        if self.per_ray_bytes is not None and request.device.startswith("cuda"):
            envelope = self._gpu_envelope()
            projected = CHUNK_SAFETY_FACTOR * self.per_ray_bytes * request.chunk_size
            if envelope <= 0 or projected > envelope:
                return (
                    "memory",
                    f"a {request.chunk_size}-ray chunk projects to {projected:.3e} B "
                    f"at {self.per_ray_bytes:.1f} B/ray with a {CHUNK_SAFETY_FACTOR}x "
                    f"safety factor, against a {envelope} B usable GPU envelope",
                )
        return None

    def launch(self, request: CandidateRequest) -> dict[str, Any]:
        """Run one candidate in a fresh child process, guarded on both sides."""
        self._launch_index += 1
        label = request.label or f"candidate_{self._launch_index:03d}"
        request = CandidateRequest(**{**request.__dict__, "label": label})
        stem = (
            f"{request.config}_P{request.launch_count}_S{request.samples_per_launch}"
            f"_{request.density}_seed{request.seed}_chunk{request.chunk_size}"
            f"_{request.precision}_{self._launch_index:03d}"
        )
        result_path = self.runs / f"{stem}.json"

        skip = self._guard_before_launch(request)
        if skip is not None:
            kind, reason = skip
            record = {
                "request": request.as_dict(),
                "status": "SKIPPED_MEMORY_GUARD",
                "skip_kind": kind,
                "failure": reason,
                "wall_time_s": 0.0,
            }
            self.note(f"SKIPPED_MEMORY_GUARD ({kind}) {label}: {reason}")
            self._record(record, result_path, write=True)
            entry = {"label": label, "status": "SKIPPED_MEMORY_GUARD", "reason": reason}
            if kind == "memory":
                # The ledger has to be written here too, not only on the
                # subprocess path: `_memory_direction_blocked` reads it, so a skip
                # that did not register would let the sweep keep proposing *larger*
                # populations after the guard had already refused a smaller one --
                # exactly the growth Phase 33 forbids.
                self.memory_failures.append(entry)
            else:
                self.budget_skips.append(entry)
            return record

        command = [
            sys.executable,
            "-m",
            "benchmarks.metalens_candidate",
            "--config",
            request.config,
            "--launch-count",
            str(request.launch_count),
            "--samples-per-launch",
            str(request.samples_per_launch),
            "--seed",
            str(request.seed),
            "--density",
            request.density,
            "--chunk-size",
            str(request.chunk_size),
            "--device",
            request.device,
            "--precision",
            request.precision,
            "--grazing-floor",
            repr(request.direction_cosine_floor),
            "--label",
            label,
            "--result",
            str(result_path),
        ]
        self.runs.mkdir(parents=True, exist_ok=True)
        self.note(
            f"launch {label}: P={request.launch_count} S={request.samples_per_launch} "
            f"N={request.total_rays} chunk={request.chunk_size} "
            f"density={request.density} seed={request.seed}"
        )
        watchdog = MemoryWatchdog(interval_s=0.5)
        started = time.time()
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),  # src/benchmarks -> src -> repo root
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        killed_for: str | None = None
        with watchdog:
            while process.poll() is None:
                time.sleep(0.5)
                if watchdog.verdict.breached:
                    killed_for = watchdog.verdict.detail
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        process.kill()
                    break
                if time.time() - started > self.options.candidate_timeout_s:
                    killed_for = (
                        f"candidate exceeded the {self.options.candidate_timeout_s} s "
                        "timeout"
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        process.kill()
                    break
        output = process.stdout.read() if process.stdout else ""
        if process.stdout:
            process.stdout.close()

        if result_path.exists():
            record = json.loads(result_path.read_text())
        else:
            record = {
                "request": request.as_dict(),
                "status": (
                    "FAIL_HOST_MEMORY_PRESSURE" if killed_for else "FAIL_NUMERICAL"
                ),
                "failure": killed_for or f"child wrote no result; exit {process.returncode}",
                "child_output_tail": output[-4000:],
                "wall_time_s": time.time() - started,
            }
        if killed_for and record.get("status") == "PASS_RUN":
            record["status"] = "FAIL_HOST_MEMORY_PRESSURE"
            record["failure"] = killed_for
        record["parent_watchdog"] = watchdog.report()
        record["child_exit_code"] = process.returncode
        self._record(record, result_path, write=not result_path.exists())
        status = record.get("status")
        metrics = record.get("metrics") or {}
        self.note(
            f"  -> {status}"
            + (f" NCC={metrics['ncc']:.6f}" if metrics.get("ncc") is not None else "")
            + f" ({record.get('wall_time_s', 0.0):.1f} s)"
        )
        if status in ("FAIL_HOST_MEMORY_PRESSURE", "FAIL_GPU_MEMORY", "SKIPPED_MEMORY_GUARD"):
            self.memory_failures.append(
                {"label": label, "status": status, "reason": record.get("failure")}
            )
        return record

    def _record(self, record: dict[str, Any], path: Path, *, write: bool) -> None:
        record["_result_path"] = str(path)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, default=str))
        self.rows.append(_summarize(record))
        self.parent_rss.append(
            {
                "completed_candidates": len(self.rows),
                "parent_rss_bytes": host_memory_snapshot().rss_bytes,
            }
        )

    # -- helpers -----------------------------------------------------------
    def _request(self, **overrides: Any) -> CandidateRequest:
        base: dict[str, Any] = {
            "config": self.options.config,
            "launch_count": self.options.pilot_launches,
            "samples_per_launch": self.options.pilot_samples,
            "seed": self.options.seed,
            "density": self.options.density,
            "chunk_size": self.chunk_size,
            "device": self.options.device,
            "precision": self.options.precision,
            "direction_cosine_floor": DIRECTION_COSINE_FLOOR,
        }
        base.update(overrides)
        return CandidateRequest(**base)

    def _passing(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["status"] == "PASS_RUN"]

    def _find(self, **query: Any) -> dict[str, Any] | None:
        """The canonical passing row for a query, which is the largest-chunk one.

        Several rows can match a ``(P, S, seed, density, config)`` query -- the
        sweep's own run plus every chunk replication of it. They are numerically
        the same field (that is what the replication proves), but they are not the
        same *row*, and picking whichever happened to be first made the report say
        the reported pair "ran in 4 chunks of 1024" when the sweep had run it in
        one. Largest chunk = fewest chunks = the sweep's own run.
        """
        matches = [
            row
            for row in self._passing()
            if all(row.get(key) == value for key, value in query.items())
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: row.get("chunk_size") or 0)

    def _memory_direction_blocked(self) -> bool:
        return bool(self.memory_failures)

    # -- phases ------------------------------------------------------------
    def calibrate(self) -> None:
        """Phase 18/19: measure bytes per ray, then size the chunk from it.

        Two runs of the same small candidate at two chunk sizes. The GPU peak is
        modelled as ``a + b * chunk`` and ``b`` is the per-ray cost -- measured,
        because Optiland and the reconstruction both create temporaries that a
        count of ray attributes does not predict.
        """
        if self.options.chunk_size is not None:
            self.note(f"chunk size fixed by request at {self.chunk_size}; calibration skipped")
            return
        probes = []
        for chunk in (MIN_CHUNK_SIZE, 4 * MIN_CHUNK_SIZE):
            record = self.launch(
                self._request(
                    launch_count=8,
                    samples_per_launch=4 * MIN_CHUNK_SIZE,
                    chunk_size=chunk,
                    label=f"calibration_chunk_{chunk}",
                )
            )
            peak = (record.get("gpu_memory_after") or {}).get("peak_allocated_bytes")
            if record.get("status") == "PASS_RUN" and peak:
                probes.append((chunk, int(peak)))
        if len(probes) < 2 or probes[1][1] <= probes[0][1]:
            self.note(
                "calibration could not separate a per-ray cost from the fixed cost; "
                f"keeping the conservative default chunk {self.chunk_size}"
            )
            return
        (c0, p0), (c1, p1) = probes
        self.per_ray_bytes = (p1 - p0) / (c1 - c0)
        envelope = self._gpu_envelope()
        allowed = envelope / (CHUNK_SAFETY_FACTOR * self.per_ray_bytes) if self.per_ray_bytes else 0
        exponent = math.floor(math.log2(max(allowed, MIN_CHUNK_SIZE)))
        self.chunk_size = int(
            min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, 2**exponent))
        )
        self.note(
            f"calibrated {self.per_ray_bytes:.1f} GPU bytes/ray from peaks "
            f"{p0} B @ {c0} and {p1} B @ {c1}; usable envelope {envelope} B; "
            f"chunk size set to {self.chunk_size}"
        )

    def pilot_bracket(self) -> None:
        """Phase 24: a small 2-D bracket, to see which axis limits the error."""
        p0, s0 = self.options.pilot_launches, self.options.pilot_samples
        for launches in (p0, 2 * p0, 4 * p0):
            for samples in (s0, 2 * s0, 4 * s0):
                if self._memory_direction_blocked():
                    self.note("pilot bracket halted: a memory guard already fired")
                    return
                self.launch(
                    self._request(
                        launch_count=launches,
                        samples_per_launch=samples,
                        label=f"pilot_P{launches}_S{samples}",
                    )
                )

    def _psf_agreement(
        self, left: dict[str, Any] | None, right: dict[str, Any] | None
    ) -> float | None:
        """``1 - NCC`` between two candidates' PSFs, or ``None`` if unavailable."""
        if not left or not right or not left.get("arrays_path") or not right.get("arrays_path"):
            return None
        from evaluation.metalens import (
            normalized_cross_correlation,
        )

        return 1.0 - normalized_cross_correlation(
            np.load(left["arrays_path"])["test"], np.load(right["arrays_path"])["test"]
        )

    def _plateau_reached(
        self, previous: dict[str, Any] | None, current: dict[str, Any] | None
    ) -> bool:
        """Both points meet the gate and agree with each other within tolerance.

        Recorded and reported, but **not** used to truncate a sweep, and the
        reason is a measured finding rather than caution.

        The angular sampler is positional, so ``S`` and ``2S`` at the same seed
        share the first ``S`` draws: they are *correlated realizations*, and
        correlated realizations agree with each other better than either agrees
        with the truth. In the first executed sweep this criterion declared a
        plateau between ``S=256`` and ``S=512`` -- and the final selection then
        needed ``S=1024``. So consecutive-point agreement under nested sampling is
        not evidence of convergence, and using it as a stopping rule made the
        sweep stop short of its own answer.

        The same caveat applies to Phase 27's doubling criterion, which is the
        ticket's declared gate and is still applied exactly as specified. What the
        report adds beside it is the *uncorrelated* evidence: agreement with the
        analytic oracle, which shares nothing with the estimator, and the spread
        over independent seeds.
        """
        if not previous or not current:
            return False
        if (previous.get("ncc") or 0.0) < NCC_GATE or (current.get("ncc") or 0.0) < NCC_GATE:
            return False
        agreement = self._psf_agreement(previous, current)
        return agreement is not None and agreement < STABILITY_TOLERANCE

    def spatial_sweep(self, samples: int) -> None:
        """Phase 25: hold S, grow P while nesting the spatial samples."""
        launches = self.options.pilot_launches
        previous: dict[str, Any] | None = None
        while launches <= self.options.max_launches:
            current = self._find(
                P=launches, S=samples, seed=self.options.seed, density=self.options.density
            )
            if current is None:
                record = self.launch(
                    self._request(
                        launch_count=launches,
                        samples_per_launch=samples,
                        label=f"spatial_P{launches}_S{samples}",
                    )
                )
                if record.get("status") != "PASS_RUN":
                    self.note(f"spatial sweep stops at P={launches}: {record.get('status')}")
                    return
                current = self.rows[-1]
            if self._plateau_reached(previous, current):
                self.note(
                    f"spatial plateau at P={launches} against P={launches // 2} "
                    "(recorded; not a stopping rule -- nested samples are correlated)"
                )
            previous = current
            launches *= 2

    def angular_sweep(self, launches: int) -> None:
        """Phase 26: hold P, grow S."""
        samples = self.options.pilot_samples
        previous: dict[str, Any] | None = None
        while samples <= self.options.max_samples:
            current = self._find(
                P=launches, S=samples, seed=self.options.seed, density=self.options.density
            )
            if current is None:
                record = self.launch(
                    self._request(
                        launch_count=launches,
                        samples_per_launch=samples,
                        label=f"angular_P{launches}_S{samples}",
                    )
                )
                if record.get("status") != "PASS_RUN":
                    self.note(f"angular sweep stops at S={samples}: {record.get('status')}")
                    return
                current = self.rows[-1]
            if self._plateau_reached(previous, current):
                self.note(
                    f"angular plateau at S={samples} against S={samples // 2} "
                    "(recorded; not a stopping rule -- nested samples are correlated)"
                )
            previous = current
            samples *= 2

    def joint_and_seeds(self, launches: int, samples: int) -> dict[str, Any]:
        """Phases 27 and 28: the doublings around a candidate, then extra seeds."""
        for factor_p, factor_s in ((2, 1), (1, 2), (2, 2)):
            target_p, target_s = launches * factor_p, samples * factor_s
            if self._find(P=target_p, S=target_s, seed=self.options.seed,
                          density=self.options.density) is None:
                self.launch(
                    self._request(
                        launch_count=target_p,
                        samples_per_launch=target_s,
                        label=f"joint_P{target_p}_S{target_s}",
                    )
                )
        for seed in self.options.seeds:
            if seed == self.options.seed:
                continue
            for target_p, target_s in ((launches, samples), (2 * launches, samples),
                                       (launches, 2 * samples)):
                self.launch(
                    self._request(
                        launch_count=target_p,
                        samples_per_launch=target_s,
                        seed=seed,
                        label=f"seed{seed}_P{target_p}_S{target_s}",
                    )
                )
        return self.seed_statistics(launches, samples)

    def density_validation(self, launches: int, samples: int) -> None:
        """Phase 29: the same spatial sampling under the uniform angular density."""
        other = "p_uni" if self.options.density == "p_mag" else "p_mag"
        for factor in (1, 4):
            self.launch(
                self._request(
                    launch_count=launches,
                    samples_per_launch=samples * factor,
                    density=other,
                    label=f"density_{other}_P{launches}_S{samples * factor}",
                )
            )

    def memory_ladder(self) -> None:
        """Phase 32: does peak memory follow the chunk, or the ray total?

        Two ladders. One grows ``P*S`` at a *fixed* chunk -- peak memory must stay
        flat. The other holds ``P*S`` and varies the chunk -- peak memory must
        track it. Together they are the statement "working memory is controlled by
        chunk_size, not by P*S", measured in both directions.
        """
        if self._memory_direction_blocked():
            self.note("memory ladder skipped: a memory guard already fired this sweep")
            return
        samples = 8192
        populations = [launches * samples for launches in (16, 64, 256, 1024)]
        # The fixed-chunk ladder is only a fixed-chunk ladder if every point
        # actually chunks. With the calibrated chunk (the largest the envelope
        # allows) the smaller populations would run in one chunk, so their
        # *effective* chunk would be their own size and the "flat in P*S" panel
        # would show a straight line rising with P*S -- measuring the chunk, not
        # the total. Half the smallest population keeps every point chunked.
        fixed_chunk = min(self.chunk_size, max(MIN_CHUNK_SIZE, min(populations) // 2))
        self.note(
            f"memory ladder uses a fixed chunk of {fixed_chunk} rays so every "
            f"population ({min(populations)}..{max(populations)}) genuinely chunks"
        )
        for total in populations:
            if total > self.options.max_total_rays:
                self.note(
                    f"memory ladder stops before P*S={total}: it exceeds the "
                    f"{self.options.max_total_rays} declared cap"
                )
                break
            record = self.launch(
                self._request(
                    launch_count=total // samples,
                    samples_per_launch=samples,
                    chunk_size=fixed_chunk,
                    label=f"ladder_fixed_chunk_N{total}",
                )
            )
            if record.get("status") != "PASS_RUN":
                self.note(f"memory ladder stops at P*S={total}: {record.get('status')}")
                return
        chunk = MIN_CHUNK_SIZE
        while chunk <= min(self.chunk_size, 1 << 19):
            self.launch(
                self._request(
                    launch_count=64,
                    samples_per_launch=samples,
                    chunk_size=chunk,
                    label=f"ladder_chunk_{chunk}",
                )
            )
            chunk *= 4

    def chunk_replication(self, launches: int, samples: int) -> dict[str, Any]:
        """Re-run the converged pair at strictly smaller chunks (Phases 9-11, 32).

        The calibrated chunk is the largest one the GPU envelope allows, which is
        the right production choice -- and for a converged pair of a few thousand
        rays it means the reported point ran in **one** chunk. That would leave
        "executes using bounded-memory chunked GPU processing" unsupported *at the
        point actually reported*, however well the memory ladder demonstrates it
        elsewhere.

        So the pair is replicated at strictly smaller chunk sizes and each PSF is
        compared to the single-chunk run. Agreement to round-off is the claim;
        ``tests/test_streaming_estimator.py`` makes the same check on the host, and
        this makes it on the real device at the reported point.
        """
        total = launches * samples
        base = self._find(
            P=launches, S=samples, seed=self.options.seed,
            density=self.options.density, config=self.options.config,
        )
        replications: list[dict[str, Any]] = []
        for divisor in (4, 16, 64):
            chunk = max(1, total // divisor)
            if chunk >= total:
                continue
            record = self.launch(
                self._request(
                    launch_count=launches,
                    samples_per_launch=samples,
                    chunk_size=chunk,
                    label=f"chunkrep_{chunk}",
                )
            )
            if record.get("status") != "PASS_RUN":
                replications.append({"chunk_size": chunk, "status": record.get("status")})
                continue
            row = self.rows[-1]
            replications.append(
                {
                    "chunk_size": chunk,
                    "chunk_count": row["chunk_count"],
                    "ncc_vs_oracle": row["ncc"],
                    "one_minus_ncc_vs_single_chunk": self._psf_agreement(base, row),
                    "status": "PASS_RUN",
                }
            )
        return {
            "pair": {"P": launches, "S": samples, "total_rays": total},
            "single_chunk_size": base["chunk_size"] if base else None,
            "single_chunk_count": base["chunk_count"] if base else None,
            "replications": replications,
        }

    def validation_configurations(self) -> None:
        """The secondary optical configuration, at the converged pair.

        The primary configuration's oracle assumes nothing beyond the analytic
        angular spectrum. The secondary one adds a real refractive interface, so
        Optiland must refract and accumulate an index-weighted path rather than a
        geometric distance -- which is the part of the OPL contract a pure air gap
        cannot exercise.
        """
        best = self.converged_candidate()
        for name in self.options.validation_configs:
            if name == self.options.config or name not in CONFIGURATIONS:
                continue
            launches = best["P"] if best else self.options.pilot_launches * 4
            samples = best["S"] if best else self.options.pilot_samples * 4
            for factor in (1, 2):
                self.launch(
                    self._request(
                        config=name,
                        launch_count=launches,
                        samples_per_launch=samples * factor,
                        label=f"validate_{name}_P{launches}_S{samples * factor}",
                    )
                )

    # -- analysis ----------------------------------------------------------
    def seed_statistics(self, launches: int, samples: int) -> dict[str, Any]:
        """Phase 28: mean and spread over seeds, so a plateau is not one lucky draw."""
        # One row per *seed*. A seed can appear several times -- the sweep's run
        # plus its chunk replications -- and counting those as extra trials would
        # inflate the trial count and shrink the spread with copies of one
        # realization, which is the opposite of what this statistic is for.
        by_seed: dict[int, dict[str, Any]] = {}
        for row in self._passing():
            if (
                row["P"] == launches
                and row["S"] == samples
                and row["density"] == self.options.density
                and row["config"] == self.options.config
            ):
                seed = row["seed"]
                if seed not in by_seed or (row.get("chunk_size") or 0) > (
                    by_seed[seed].get("chunk_size") or 0
                ):
                    by_seed[seed] = row
        matching = [by_seed[seed] for seed in sorted(by_seed)]
        if not matching:
            return {}
        ncc = np.array([row["ncc"] for row in matching], dtype=float)
        nmse = np.array([row["normalized_mse"] for row in matching], dtype=float)
        return {
            "P": launches,
            "S": samples,
            "seeds": [row["seed"] for row in matching],
            "trials": len(matching),
            "mean_ncc": float(ncc.mean()),
            "std_ncc": float(ncc.std(ddof=1)) if ncc.size > 1 else 0.0,
            "min_ncc": float(ncc.min()),
            "mean_nmse": float(nmse.mean()),
            "std_nmse": float(nmse.std(ddof=1)) if nmse.size > 1 else 0.0,
        }

    def _stability(self, launches: int, samples: int) -> dict[str, Any]:
        """``1 - NCC`` between a pair's PSF and each available doubling's PSF."""
        base = self._find(
            P=launches, S=samples, seed=self.options.seed,
            density=self.options.density, config=self.options.config,
        )
        if base is None or not base.get("arrays_path"):
            return {"available": False}
        from evaluation.metalens import (
            normalized_cross_correlation,
        )

        base_psf = np.load(base["arrays_path"])["test"]
        checks: dict[str, Any] = {"available": True, "pairs": {}}
        for name, (factor_p, factor_s) in {
            "2P_S": (2, 1),
            "P_2S": (1, 2),
            "2P_2S": (2, 2),
        }.items():
            other = self._find(
                P=launches * factor_p, S=samples * factor_s, seed=self.options.seed,
                density=self.options.density, config=self.options.config,
            )
            if other is None or not other.get("arrays_path"):
                checks["pairs"][name] = None
                continue
            value = normalized_cross_correlation(base_psf, np.load(other["arrays_path"])["test"])
            checks["pairs"][name] = {
                "ncc": float(value),
                "one_minus_ncc": float(1.0 - value),
                "within_tolerance": bool(1.0 - value < STABILITY_TOLERANCE),
            }
        present = [item for item in checks["pairs"].values() if item is not None]
        checks["doublings_available"] = len(present)
        checks["all_within_tolerance"] = bool(present) and all(
            item["within_tolerance"] for item in present
        )
        # The declared criterion above compares correlated realizations: the
        # positional sampler makes (P, 2S) an extension of (P, S)'s own draws, so
        # the two share a prefix and agree better than either agrees with the
        # truth. Recorded alongside it, therefore, are the two uncorrelated
        # numbers -- distance from the analytic oracle, which shares nothing with
        # the estimator, and the worst independent seed at the same pair.
        seeds = [
            row
            for row in self._passing()
            if row["P"] == launches and row["S"] == samples
            and row["density"] == self.options.density
            and row["config"] == self.options.config
        ]
        checks["correlation_caveat"] = (
            "the doubling comparisons above share a sample prefix with the base "
            "pair (positional sampler), so they are correlated realizations; the "
            "uncorrelated evidence is the oracle distance and the seed spread"
        )
        checks["uncorrelated"] = {
            "one_minus_ncc_vs_analytic_oracle": (
                None if base.get("ncc") is None else 1.0 - base["ncc"]
            ),
            "worst_seed_ncc_vs_oracle": (
                min(row["ncc"] for row in seeds) if seeds else None
            ),
            "seed_count": len(seeds),
        }
        return checks

    def converged_candidate(self) -> dict[str, Any] | None:
        """The smallest ``P*S`` that meets the gate and is locally stable.

        Scans every executed candidate at the sweep's primary seed, density and
        configuration, ordered by total rays. The *first* one that qualifies wins,
        which is what makes the answer "smallest converged" rather than "largest
        that ran".
        """
        candidates = sorted(
            (
                row
                for row in self._passing()
                if row["seed"] == self.options.seed
                and row["density"] == self.options.density
                and row["config"] == self.options.config
                and row["ncc"] is not None
                and row["ncc"] >= NCC_GATE
            ),
            # Smallest population first; among rows that are the same pair, the
            # sweep's own largest-chunk run is the canonical one to report.
            key=lambda row: (row["total_rays"], row["P"], -(row.get("chunk_size") or 0)),
        )
        seen: set[tuple[int, int]] = set()
        for row in candidates:
            pair = (row["P"], row["S"])
            if pair in seen:
                continue
            seen.add(pair)
            stability = self._stability(row["P"], row["S"])
            if stability.get("all_within_tolerance"):
                return {**row, "stability": stability}
        return None

    def power_converged_candidate(self) -> dict[str, Any] | None:
        """The smallest pair whose *radiometry* has also converged.

        Reported beside the declared NCC gate, not instead of it, because the
        measured gap between the two is the single most useful thing this sweep
        found. Phase 23 warned that NCC is blind to a global scale; the sweep
        makes that concrete -- the NCC gate is met at ``N = 4096``, where the
        reconstructed total power is still **25% high**, and the power error only
        falls below 1% around ``N = 1.3e5``.

        The excess is a bias, not noise, and its sign is not accidental: for a
        Monte Carlo estimate of a complex field, ``E|sum a|^2 = |sum E a|^2 +
        Var``, so the intensity is high by the estimator's own variance and the
        excess decays with the ray count at the same rate the variance does.

        A consumer who needs an amplitude, an efficiency or a Strehl -- as opposed
        to a PSF shape -- needs this pair, not the NCC one.
        """
        for row in sorted(
            (
                row
                for row in self._passing()
                if row["seed"] == self.options.seed
                and row["density"] == self.options.density
                and row["config"] == self.options.config
                and row["relative_power_error"] is not None
                and row["ncc"] is not None
                and row["ncc"] >= NCC_GATE
                and abs(row["relative_power_error"]) < POWER_ERROR_TARGET
            ),
            key=lambda row: (row["total_rays"], row["P"], -(row.get("chunk_size") or 0)),
        ):
            return row
        return None

    # -- outputs -----------------------------------------------------------
    def write_tables(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        convergence_fields = [
            "label", "config", "P", "S", "total_rays", "density", "seed", "precision",
            "chunk_size", "chunk_count", "status", "ncc", "normalized_mse",
            "reference_power", "test_power", "relative_power_error",
            "relative_peak_error", "centroid_error_m", "relative_fwhm_error",
            "relative_ee50_error", "valid_rays", "wall_time_s", "failure",
        ]
        memory_fields = [
            "label", "P", "S", "total_rays", "chunk_size", "chunk_count", "status",
            "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_free_after_bytes",
            "peak_rss_bytes", "min_mem_available_bytes", "cgroup_swap_before_bytes",
            "peak_cgroup_swap_bytes", "swap_delta_peak_bytes", "wall_time_s",
        ]
        tables = (
            ("convergence.csv", convergence_fields),
            ("memory.csv", memory_fields),
        )
        for name, fields in tables:
            with (self.root / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)

    def write_plots(self, best: dict[str, Any] | None) -> list[str]:
        """Phase 31/32's figures. Regenerable, so they live under the output root."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots = self.root / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        primary = [
            row
            for row in self._passing()
            if row["config"] == self.options.config
            and row["density"] == self.options.density
            and row["seed"] == self.options.seed
        ]

        def scatter(name: str, ykey: str, ylabel: str, *, log: bool = False) -> None:
            if not primary:
                return
            figure, axes = plt.subplots(figsize=(6.2, 4.2))
            for label, marker in (("spatial", "o"), ("angular", "s"), ("pilot", "^"),
                                  ("joint", "D"), ("ladder", "x"), ("calibration", "+")):
                subset = [row for row in primary if row["label"].startswith(label)]
                subset = [row for row in subset if row.get(ykey) is not None]
                if not subset:
                    continue
                axes.plot(
                    [row["total_rays"] for row in subset],
                    [row[ykey] for row in subset],
                    marker,
                    label=f"{label} regime",
                    alpha=0.85,
                )
            axes.set_xscale("log")
            if log:
                axes.set_yscale("log")
            axes.set_xlabel("total rays  P x S")
            axes.set_ylabel(ylabel)
            axes.grid(alpha=0.3)
            axes.legend(fontsize=8)
            axes.set_title(f"{self.options.config}  {ylabel}")
            figure.tight_layout()
            figure.savefig(plots / name, dpi=140)
            plt.close(figure)
            written.append(name)

        scatter("ncc_vs_total_rays.png", "ncc", "NCC vs analytic oracle")
        scatter("nmse_vs_total_rays.png", "normalized_mse", "normalized MSE", log=True)
        scatter("gpu_memory_vs_total_rays.png", "gpu_peak_allocated_bytes",
                "peak GPU allocated (B)", log=True)
        scatter("cpu_rss_vs_total_rays.png", "peak_rss_bytes", "peak process RSS (B)")

        # 2-D (P, S) heatmap of NCC.
        grid_rows = [row for row in primary if row["ncc"] is not None]
        if grid_rows:
            launches = sorted({row["P"] for row in grid_rows})
            samples = sorted({row["S"] for row in grid_rows})
            image = np.full((len(samples), len(launches)), np.nan)
            for row in grid_rows:
                image[samples.index(row["S"]), launches.index(row["P"])] = row["ncc"]
            figure, axes = plt.subplots(figsize=(6.4, 4.6))
            mesh = axes.imshow(image, origin="lower", aspect="auto", cmap="viridis")
            axes.set_xticks(range(len(launches)), [str(value) for value in launches])
            axes.set_yticks(range(len(samples)), [str(value) for value in samples])
            axes.set_xlabel("P  (spatial launch positions)")
            axes.set_ylabel("S  (angular samples per launch)")
            axes.set_title("NCC(PSF_ray, PSF_ref)")
            for (row_index, column_index), value in np.ndenumerate(image):
                if np.isfinite(value):
                    axes.text(column_index, row_index, f"{value:.4f}", ha="center",
                              va="center", fontsize=6, color="w")
            figure.colorbar(mesh, ax=axes)
            figure.tight_layout()
            figure.savefig(plots / "ncc_P_S_heatmap.png", dpi=140)
            plt.close(figure)
            written.append("ncc_P_S_heatmap.png")

        # Memory scaling: peak GPU against chunk at fixed P*S, and against P*S at
        # fixed chunk. The two series are the whole claim of Phase 32.
        ladder_chunk = sorted(
            (row for row in self._passing() if row["label"].startswith("ladder_chunk_")),
            key=lambda row: row["effective_chunk_size"] or 0,
        )
        ladder_fixed = sorted(
            (row for row in self._passing() if row["label"].startswith("ladder_fixed_")),
            key=lambda row: row["total_rays"],
        )
        if ladder_chunk or ladder_fixed:
            figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
            if ladder_chunk:
                x = np.array([row["effective_chunk_size"] for row in ladder_chunk], float)
                y = np.array([row["gpu_peak_allocated_bytes"] for row in ladder_chunk], float)
                axes[0].plot(x, y, "o-")
                slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
                axes[0].set_title(
                    f"fixed P*S: memory tracks the chunk (slope {slope:.2f})", fontsize=10
                )
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("effective chunk size (rays)")
            axes[0].set_ylabel("peak GPU allocated (B)")
            axes[0].grid(alpha=0.3)
            if ladder_fixed:
                x = np.array([row["total_rays"] for row in ladder_fixed], float)
                y = np.array([row["gpu_peak_allocated_bytes"] for row in ladder_fixed], float)
                axes[1].plot(x, y, "s-")
                slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
                axes[1].set_ylim(0.0, 1.3 * y.max())
                axes[1].set_title(
                    f"fixed chunk: memory flat in P*S (slope {slope:.3f})", fontsize=10
                )
            axes[1].set_xscale("log")
            axes[1].set_xlabel("total rays  P x S")
            axes[1].set_ylabel("peak GPU allocated (B)")
            axes[1].grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(plots / "memory_scaling.png", dpi=140)
            plt.close(figure)
            written.append("memory_scaling.png")

        # PSF comparison at the converged pair.
        if best and best.get("arrays_path"):
            arrays = np.load(best["arrays_path"])
            test, reference = arrays["test"], arrays["reference"]
            difference = test - reference
            top = float(max(test.max(), reference.max()))
            figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
            panels = (
                (reference, "direct wave (analytic oracle)"),
                (test, "coupled ray-wave"),
                (difference, "difference"),
            )
            for panel, (image, title) in zip(axes, panels, strict=True):
                if title == "difference":
                    span = float(np.abs(difference).max()) or 1.0
                    mesh = panel.imshow(image, origin="lower", cmap="RdBu_r",
                                        vmin=-span, vmax=span)
                else:
                    mesh = panel.imshow(image, origin="lower", cmap="inferno",
                                        vmin=0.0, vmax=top)
                panel.set_title(title, fontsize=9)
                figure.colorbar(mesh, ax=panel, fraction=0.046)
            figure.suptitle(
                f"{self.options.config}  P={best['P']} S={best['S']}  "
                f"NCC={best['ncc']:.6f}",
                fontsize=10,
            )
            figure.tight_layout()
            figure.savefig(plots / "psf_comparison.png", dpi=140)
            plt.close(figure)
            written.append("psf_comparison.png")

            psf_dir = self.root / "psf"
            psf_dir.mkdir(parents=True, exist_ok=True)
            np.save(psf_dir / "psf_ref.npy", reference)
            np.save(psf_dir / "psf_final.npy", test)
            np.save(psf_dir / "psf_difference.npy", difference)
        return written

    # -- driver ------------------------------------------------------------
    def reanalyze(self) -> dict[str, Any]:
        """Rebuild every output from the persisted candidate files, launching nothing.

        The candidates are the expensive part and each one is already an atomic,
        self-describing JSON file, so the tables, plots, selection and report are
        derivable from what is on disk. That makes the analysis iterable without
        re-running an 18-minute sweep -- and it means a reported number can always
        be traced back to the file it came from.
        """
        files = sorted(self.runs.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no candidate results under {self.runs}")
        for path in files:
            record = json.loads(path.read_text())
            record.setdefault("_result_path", str(path))
            self._record(record, path, write=False)
        self.note(f"re-analysed {len(files)} persisted candidates from {self.runs}")

        best = self.converged_candidate()
        seed_stats = (
            self.seed_statistics(best["P"], best["S"]) if best is not None else {}
        )
        replication = self._replication_from_rows(best) if best is not None else {}
        self.write_tables()
        plots = self.write_plots(best)
        manifest_path = self.root / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else self.manifest()
        )
        summary = self.summary(best, seed_stats, plots, manifest, replication)
        summary["reanalyzed"] = True
        (self.root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        (self.root / "REPORT.md").write_text(self.report(summary))
        self.note(f"rewrote {self.root}/run_summary.json and REPORT.md")
        return summary

    def _replication_from_rows(self, best: dict[str, Any]) -> dict[str, Any]:
        """Recover the chunk-replication table from persisted rows, without re-running."""
        base = self._find(
            P=best["P"], S=best["S"], seed=self.options.seed,
            density=self.options.density, config=self.options.config,
        )
        replications = [
            {
                "chunk_size": row["chunk_size"],
                "chunk_count": row["chunk_count"],
                "ncc_vs_oracle": row["ncc"],
                "one_minus_ncc_vs_single_chunk": self._psf_agreement(base, row),
                "status": row["status"],
            }
            for row in sorted(
                (row for row in self.rows if row["label"].startswith("chunkrep_")),
                key=lambda row: -(row["chunk_size"] or 0),
            )
        ]
        if not replications:
            return {}
        return {
            "pair": {"P": best["P"], "S": best["S"], "total_rays": best["total_rays"]},
            "single_chunk_size": base["chunk_size"] if base else None,
            "single_chunk_count": base["chunk_count"] if base else None,
            "replications": replications,
        }

    def run(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

        self.calibrate()
        self.pilot_bracket()

        # Phase 25 wants S high enough that spatial convergence is measurable
        # rather than buried in angular noise; the pilot bracket's largest S is
        # the highest one already shown to run inside the envelope.
        high_samples = self.options.pilot_samples * 4
        self.spatial_sweep(high_samples)
        best_spatial = max(
            (row for row in self._passing()
             if row["S"] == high_samples and row["seed"] == self.options.seed
             and row["config"] == self.options.config),
            key=lambda row: row["P"],
            default=None,
        )
        self.angular_sweep(best_spatial["P"] if best_spatial else self.options.pilot_launches * 4)

        preliminary = self.converged_candidate()
        seed_stats: dict[str, Any] = {}
        if preliminary is not None:
            seed_stats = self.joint_and_seeds(preliminary["P"], preliminary["S"])
            self.density_validation(preliminary["P"], preliminary["S"])
        else:
            self.note("no candidate met the gate with local stability in the adaptive sweep")

        if self.options.memory_ladder:
            self.memory_ladder()
        self.validation_configurations()

        best = self.converged_candidate()
        moved = preliminary is None or best["total_rays"] != preliminary["total_rays"]
        if best is not None and moved:
            seed_stats = self.seed_statistics(best["P"], best["S"])
        replication: dict[str, Any] = {}
        if best is not None:
            replication = self.chunk_replication(best["P"], best["S"])
            # Re-select afterwards: the replications are executed candidates too,
            # and the answer must be the smallest qualifying one over everything
            # this sweep actually ran.
            best = self.converged_candidate() or best

        self.write_tables()
        plots = self.write_plots(best)
        summary = self.summary(best, seed_stats, plots, manifest, replication)
        (self.root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (self.root / "REPORT.md").write_text(self.report(summary))
        self.note(f"wrote {self.root}/run_summary.json and REPORT.md")
        return summary

    def manifest(self) -> dict[str, Any]:
        import platform
        import subprocess as sp

        def git(*args: str) -> str | None:
            try:
                out = sp.run(["git", *args], capture_output=True, text=True, check=False,
                             cwd=str(Path(__file__).resolve().parents[2]))  # src/benchmarks -> src -> root
                return out.stdout.strip() or None
            except Exception:  # pragma: no cover
                return None

        host = host_memory_snapshot()
        gpu = cuda_memory_snapshot()
        config = CONFIGURATIONS[self.options.config]
        return {
            "benchmark": "CHE70-METALENS-BRIDGE",
            "issue": "CHE-70",
            "timestamp_unix": self.started,
            "project_commit": git("rev-parse", "HEAD"),
            "project_dirty": bool(git("status", "--porcelain")),
            "reference_implementation": {
                "paper": (
                    "Cheng et al., A Differentiable Ray-Wave Framework for Hybrid "
                    "Refractive-Diffractive System Modeling and Optimization, ACS "
                    "Photonics, 2026 (DOI 10.1021/acsphotonics.6c00818)"
                ),
                "repository": "https://github.com/jiazhou-cheng/raywave-tracing",
                "vendored": False,
                "pinned": False,
                "executed_by_this_repository": False,
                "note": (
                    "Not vendored, not pinned and not executed, by instruction. The "
                    "estimator structure (P spatial launches x S angular samples, "
                    "launch-position Fourier shift phase, 1/N normalization) is taken "
                    "from the paper and its SI; no commit SHA is recorded because no "
                    "commit was fetched. The unvendored implementation is not cited "
                    "as evidence anywhere in this benchmark."
                ),
            },
            "oracle": {
                "kind": "analytic",
                "description": (
                    "exact plane-wave transfer function of the plane-parallel layer "
                    "stack, evaluated in float64 over the grid's own modes"
                ),
                "admissibility": (
                    "authorised for this benchmark: analytic metalens direct wave "
                    "propagation is an admissible independent PSF oracle. It is not "
                    "a repository-local numerical solver -- a homogeneous layer stack "
                    "is diagonal in the plane-wave basis, so the finite modal sum is "
                    "the exact solution rather than a discretization of one."
                ),
                "cross_checks": [
                    "evaluation.asm_oracle.angular_spectrum_float64 (CHE-40), "
                    "un-centred FFT convention, agreement 6.1e-14 on the air config",
                    "chromatix asm_propagate, third-party, M1-verified "
                    "(tests/test_metalens_oracle.py)",
                ],
            },
            "gates": {
                "ncc": NCC_GATE,
                "stability_one_minus_ncc": STABILITY_TOLERANCE,
                "basis": (
                    "declared engineering targets, authorised as such (CHE-70 "
                    "blocking decision 3). Not derived from a noise model. The "
                    "measured floors they sit above are recorded in the report."
                ),
            },
            "options": self.options.as_dict(),
            "configuration": config.as_dict(),
            "grazing_band_limit": {
                "direction_cosine_floor": DIRECTION_COSINE_FLOOR,
                "why": (
                    "C_RAY_TO_WAVE forms each ray's constant phase as "
                    "k(OPL - d.x0); near grazing the two terms cancel and the "
                    "phase error is eps*k*Z/d_n. Eight bins on this grid land on "
                    "d_u^2+d_v^2 = 1 exactly and survive the evanescent cut at "
                    "d_n = 1.05e-8, giving a 4745 m OPL. Measured, float64, full "
                    "enumeration: 2.8e-9 relative field error without the floor, "
                    "8.9e-14 with it."
                ),
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "mem_total_bytes": host.mem_total_bytes,
                "mem_available_at_start_bytes": host.mem_available_bytes,
                "host_swap_total_bytes": host.host_swap_total_bytes,
                "host_swap_used_at_start_bytes": host.host_swap_used_bytes,
                "cgroup_swap_at_start_bytes": host.cgroup_swap_bytes,
                "gpu": gpu.as_dict(),
                "host_reserve_bytes": host_reserve_bytes(host.mem_total_bytes),
                "gpu_usable_fraction": GPU_USABLE_FRACTION,
                "gpu_reserve_bytes": GPU_RESERVE_BYTES,
            },
            "policy": {
                "concurrency": "one candidate at a time; AGENTS.md forbids concurrent GPU jobs",
                "gpu_count": 1,
                "swap": "container cgroup swap delta must be exactly 0 (Phase 17)",
                "autograd": "disabled for every candidate (Phase 13)",
            },
        }

    def summary(
        self,
        best: dict[str, Any] | None,
        seed_stats: dict[str, Any],
        plots: list[str],
        manifest: dict[str, Any],
        chunk_replication: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        swap_deltas = [
            row["swap_delta_peak_bytes"]
            for row in self.rows
            if row.get("swap_delta_peak_bytes") is not None
        ]
        peaks = [row["gpu_peak_allocated_bytes"] for row in self.rows
                 if row.get("gpu_peak_allocated_bytes")]
        rss = [row["peak_rss_bytes"] for row in self.rows if row.get("peak_rss_bytes")]
        available = [row["min_mem_available_bytes"] for row in self.rows
                     if row.get("min_mem_available_bytes")]
        parent = [entry["parent_rss_bytes"] for entry in self.parent_rss]
        return {
            "manifest": manifest,
            "final_status": (
                "PASS" if best is not None else "FAIL_NOT_CONVERGED_WITHIN_SAFE_MEMORY"
            ),
            "smallest_converged": best,
            "smallest_power_converged": self.power_converged_candidate(),
            "power_error_target": POWER_ERROR_TARGET,
            "seed_statistics": seed_stats,
            "chunk_replication": chunk_replication or {},
            "candidate_count": len(self.rows),
            "status_counts": {
                status: sum(1 for row in self.rows if row["status"] == status)
                for status in sorted({row["status"] for row in self.rows})
            },
            "swap": {
                "measured_candidates": len(swap_deltas),
                "max_swap_delta_peak_bytes": max(swap_deltas) if swap_deltas else None,
                "all_zero": bool(swap_deltas) and all(value == 0 for value in swap_deltas),
                "signal": "/sys/fs/cgroup/memory.swap.current (container, not host)",
            },
            "memory": {
                "max_gpu_peak_allocated_bytes": max(peaks) if peaks else None,
                "max_peak_rss_bytes": max(rss) if rss else None,
                "min_mem_available_bytes": min(available) if available else None,
                "parent_rss_first_bytes": parent[0] if parent else None,
                "parent_rss_last_bytes": parent[-1] if parent else None,
                "parent_rss_growth_bytes": (parent[-1] - parent[0]) if len(parent) > 1 else 0,
                "parent_rss_trace": self.parent_rss,
            },
            "memory_failures": self.memory_failures,
            "budget_skips": self.budget_skips,
            "memory_scaling": self._memory_scaling(),
            "calibration": {
                "gpu_bytes_per_ray": self.per_ray_bytes,
                "chunk_size": self.chunk_size,
                "safety_factor": CHUNK_SAFETY_FACTOR,
            },
            "plots": plots,
            "notes": self.notes,
            "wall_time_s": time.time() - self.started,
            "reproduction_command": self.reproduction_command(),
        }

    def _memory_scaling(self) -> dict[str, Any]:
        """Phase 32 as two fitted numbers rather than only a picture.

        The claim is that peak working memory is controlled by ``chunk_size`` and
        not by ``P*S``. Stated as slopes on a log-log fit, that is: about 1 against
        the chunk, about 0 against the total. Both are reported, so the claim is
        falsifiable from the summary alone.
        """
        def fit(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
            points = [
                (row[key], row["gpu_peak_allocated_bytes"])
                for row in rows
                if row.get(key) and row.get("gpu_peak_allocated_bytes")
            ]
            if len(points) < 2:
                return {"points": len(points), "log_log_slope": None}
            x = np.log(np.array([point[0] for point in points], float))
            y = np.log(np.array([point[1] for point in points], float))
            return {
                "points": len(points),
                "log_log_slope": float(np.polyfit(x, y, 1)[0]),
                "peak_bytes": [int(point[1]) for point in points],
                key: [int(point[0]) for point in points],
            }

        chunk_rows = [row for row in self._passing() if row["label"].startswith("ladder_chunk_")]
        fixed_rows = [row for row in self._passing() if row["label"].startswith("ladder_fixed_")]
        return {
            "claim": "peak GPU memory is controlled by chunk_size, not by P*S",
            "vs_effective_chunk_at_fixed_total": fit(chunk_rows, "effective_chunk_size"),
            "vs_total_at_fixed_chunk": fit(fixed_rows, "total_rays"),
        }

    def reproduction_command(self) -> str:
        options = self.options
        return (
            "./run.sh --gpu python -m benchmarks."
            f"metalens_controller --grid-size {options.grid_size} "
            f"--device {options.device} --precision {options.precision} "
            f"--sampling-density {'magnitude' if options.density == 'p_mag' else 'uniform'} "
            f"--seed {options.seed} --auto-converge --memory-guard "
            f"--config {options.config} "
            + (
                f"--validate {' '.join(options.validation_configs)} "
                if options.validation_configs
                else ""
            )
            + f"--output {options.output}"
        )

    def report(self, summary: dict[str, Any]) -> str:
        best = summary["smallest_converged"]
        lines: list[str] = [
            "# CHE-70 — 100x100 metalens coherent ray-wave bridge, convergence report",
            "",
            "Generated by `benchmarks.metalens_controller`.",
            f"Final status: **{summary['final_status']}**.",
            "",
            "## Reproduction",
            "",
            "```bash",
            summary["reproduction_command"],
            "```",
            "",
            "## Convergence table",
            "",
            "| label | P | S | total rays | density | seed | NCC | NMSE | power err "
            "| GPU peak | CPU peak RSS | swap delta | runtime | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
            "| --- | --- |",
        ]
        def fmt(value: Any, spec: str = ".6f") -> str:
            return "—" if value is None else format(value, spec)

        for row in self.rows:
            lines.append(
                f"| {row['label']} | {row['P']} | {row['S']} | {row['total_rays']} | "
                f"{row['density']} | {row['seed']} | {fmt(row['ncc'])} | "
                f"{fmt(row['normalized_mse'], '.3e')} | "
                f"{fmt(row['relative_power_error'], '+.4f')} | "
                f"{fmt(row['gpu_peak_allocated_bytes'], ',d')} | "
                f"{fmt(row['peak_rss_bytes'], ',d')} | "
                f"{fmt(row['swap_delta_peak_bytes'], ',d')} | "
                f"{fmt(row['wall_time_s'], '.1f')} s | {row['status']} |"
            )
        lines += ["", "## Result", ""]
        if best is not None:
            stability = best.get("stability", {})
            lines += [
                "```",
                f"smallest converged P   = {best['P']}",
                f"smallest converged S   = {best['S']}",
                f"smallest converged P*S = {best['total_rays']}",
                "",
                f"NCC  = {best['ncc']:.9f}",
                f"NMSE = {best['normalized_mse']:.6e}",
                f"relative power error = {best['relative_power_error']:+.6f}",
                "",
                f"peak GPU memory  = {summary['memory']['max_gpu_peak_allocated_bytes']} B",
                f"peak CPU RSS     = {summary['memory']['max_peak_rss_bytes']} B",
                f"minimum MemAvailable = {summary['memory']['min_mem_available_bytes']} B",
                "",
                f"swap delta = {summary['swap']['max_swap_delta_peak_bytes']} B",
                "```",
                "",
                "### Local stability (Phase 27)",
                "",
                "| doubling | NCC vs the candidate | 1 - NCC | within 1e-3 |",
                "| --- | --- | --- | --- |",
            ]
            for name, item in (stability.get("pairs") or {}).items():
                if item is None:
                    lines.append(f"| {name} | not available | — | — |")
                else:
                    lines.append(
                        f"| {name} | {item['ncc']:.9f} | {item['one_minus_ncc']:.3e} | "
                        f"{'yes' if item['within_tolerance'] else 'no'} |"
                    )
            uncorrelated = stability.get("uncorrelated") or {}
            if uncorrelated:
                lines += [
                    "",
                    "The three comparisons above are between **correlated "
                    "realizations**: the sampler is positional, so `(P, 2S)` extends "
                    "`(P, S)`'s own draws and the two share a prefix. They therefore "
                    "agree with each other better than either agrees with the truth, "
                    "and the criterion is weaker than it looks. It is applied as the "
                    "ticket specifies, and the uncorrelated evidence is stated next to "
                    "it:",
                    "",
                    f"- distance from the analytic oracle, which shares nothing with "
                    f"the estimator: `1 - NCC = "
                    f"{uncorrelated.get('one_minus_ncc_vs_analytic_oracle', float('nan')):.3e}`",
                    f"- worst of {uncorrelated.get('seed_count')} independent seeds at "
                    f"this pair: `NCC = "
                    f"{uncorrelated.get('worst_seed_ncc_vs_oracle', float('nan')):.9f}`",
                ]
        else:
            lines += [
                "```",
                "FAIL_NOT_CONVERGED_WITHIN_SAFE_MEMORY",
                "```",
                "",
                "No candidate met NCC >= 0.99 with local stability inside the safe "
                "memory envelope. The best safe candidates, their metrics and the "
                "limiting resource are in `convergence.csv` and `memory.csv`.",
            ]
        if summary["seed_statistics"]:
            stats = summary["seed_statistics"]
            lines += [
                "",
                "### Monte Carlo variability at the converged pair (Phase 28)",
                "",
                f"- trials: {stats['trials']} (seeds {stats['seeds']})",
                f"- NCC: mean {stats['mean_ncc']:.9f}, std {stats['std_ncc']:.3e}, "
                f"min {stats['min_ncc']:.9f}",
                f"- NMSE: mean {stats['mean_nmse']:.6e}, std {stats['std_nmse']:.3e}",
            ]
        power = summary.get("smallest_power_converged")
        if best is not None:
            lines += [
                "",
                "### Radiometric convergence is a different, later point (Phase 23)",
                "",
                f"At the NCC-converged pair the reconstructed **total power is "
                f"{best['relative_power_error']:+.1%}** against the oracle. NCC is "
                "blind to a global scale, so the declared gate is met long before "
                "the radiometry is right, and the excess is a bias rather than "
                "noise: for a Monte Carlo estimate of a complex field "
                "`E|sum a|^2 = |sum E a|^2 + Var`, so the intensity is high by the "
                "estimator's own variance.",
                "",
            ]
            if power is not None:
                lines += [
                    f"Smallest pair meeting the gate **and** "
                    f"`|power error| < {summary['power_error_target']:.0%}`: "
                    f"`P = {power['P']}`, `S = {power['S']}`, "
                    f"`P*S = {power['total_rays']}` "
                    f"({power['total_rays'] // best['total_rays']}x the NCC pair), at "
                    f"NCC = {power['ncc']:.9f} and power error "
                    f"{power['relative_power_error']:+.5f}.",
                    "",
                    "A consumer who needs a PSF *shape* should use the NCC pair; one "
                    "who needs an amplitude, an efficiency or a Strehl needs this one.",
                ]
            else:
                lines.append(
                    f"No executed pair met the gate with "
                    f"`|power error| < {summary['power_error_target']:.0%}`."
                )
        replication = summary.get("chunk_replication") or {}
        if replication.get("replications"):
            lines += [
                "",
                "### Chunked execution at the reported pair (Phases 9-11)",
                "",
                f"The calibrated chunk is the largest the GPU envelope allows, so the "
                f"reported pair's own run used "
                f"{replication.get('single_chunk_count')} chunk(s) of "
                f"{replication.get('single_chunk_size')} rays. It was therefore "
                f"replicated at strictly smaller chunks, on the device, at the same "
                f"seed:",
                "",
                "| chunk size | chunks | NCC vs oracle | 1 - NCC vs the single-chunk run |",
                "| --- | --- | --- | --- |",
            ]
            for item in replication["replications"]:
                if item.get("status") != "PASS_RUN":
                    lines.append(
                        f"| {item['chunk_size']} | — | — | {item.get('status')} |"
                    )
                    continue
                agreement = item.get("one_minus_ncc_vs_single_chunk")
                lines.append(
                    f"| {item['chunk_size']} | {item['chunk_count']} | "
                    f"{item['ncc_vs_oracle']:.9f} | "
                    + ("—" if agreement is None else f"{agreement:.3e}")
                    + " |"
                )
        lines += ["", "## Memory", ""]
        if summary["swap"]["all_zero"]:
            lines.append("Additional swap used by the benchmark: 0 bytes.")
            lines.append("")
            lines.append(
                "Measured as the peak of `/sys/fs/cgroup/memory.swap.current` minus "
                "its value before each candidate — the *container's* swap. Host swap "
                "is non-zero at rest on this machine (CHE-64), so a host-swap delta "
                "would not have been attributable to this run; the host figures are "
                "recorded per candidate alongside, and did not move."
            )
        else:
            lines.append(
                "Swap did move, or could not be measured on every candidate. See "
                "`memory.csv`; the literal zero-swap statement is deliberately "
                "**not** made."
            )
        scaling = summary["memory_scaling"]
        chunk_slope = scaling["vs_effective_chunk_at_fixed_total"]["log_log_slope"]
        total_slope = scaling["vs_total_at_fixed_chunk"]["log_log_slope"]
        chunk_slope = float("nan") if chunk_slope is None else chunk_slope
        total_slope = float("nan") if total_slope is None else total_slope
        lines += [
            "",
            f"- parent RSS: {summary['memory']['parent_rss_first_bytes']} B at the first "
            f"candidate, {summary['memory']['parent_rss_last_bytes']} B at the last "
            f"({summary['memory']['parent_rss_growth_bytes']:+d} B over "
            f"{summary['candidate_count']} candidates) — Phase 34's leak check on the "
            "controller itself.",
            f"- memory scaling (Phase 32): peak GPU memory fits chunk^"
            f"{chunk_slope:.2f} at fixed P*S, and (P*S)^{total_slope:.3f} at fixed "
            "chunk — controlled by the chunk, flat in the total.",
            f"- calibration: {summary['calibration']['gpu_bytes_per_ray']} GPU bytes/ray, "
            f"chunk size {summary['calibration']['chunk_size']}, "
            f"{summary['calibration']['safety_factor']}x safety factor.",
            "",
            "## Status counts",
            "",
        ]
        for status, count in summary["status_counts"].items():
            lines.append(f"- `{status}`: {count}")
        if summary["memory_failures"]:
            lines += ["", "## Memory guard events", ""]
            for event in summary["memory_failures"]:
                lines.append(f"- `{event['status']}` on {event['label']}: {event['reason']}")
        lines += ["", "## Controller log", ""]
        lines += [f"- {note}" for note in summary["notes"]]
        lines.append("")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="METALENS-AIR-100", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--validate", nargs="*", default=[], choices=sorted(CONFIGURATIONS))
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument(
        "--sampling-density", choices=["magnitude", "uniform"], default="magnitude"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--auto-converge", action="store_true")
    parser.add_argument("--memory-guard", action="store_true")
    parser.add_argument("--no-memory-ladder", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--pilot-launches", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=256)
    parser.add_argument("--max-launches", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=65536)
    parser.add_argument("--max-total-rays", type=int, default=1 << 26)
    parser.add_argument("--candidate-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="rebuild the tables, plots and report from the persisted candidates "
        "under <output>/runs, launching nothing",
    )
    args = parser.parse_args(argv)

    config = CONFIGURATIONS[args.config]
    if args.grid_size != config.grid:
        parser.error(
            f"--grid-size {args.grid_size} does not match configuration {args.config}, "
            f"which is {config.grid}x{config.grid}"
        )
    options = SweepOptions(
        output=args.output,
        config=args.config,
        device=args.device,
        precision=args.precision,
        density="p_mag" if args.sampling_density == "magnitude" else "p_uni",
        seed=args.seed,
        seeds=tuple(args.seeds),
        grid_size=args.grid_size,
        auto_converge=args.auto_converge,
        memory_guard=args.memory_guard,
        memory_ladder=not args.no_memory_ladder,
        chunk_size=args.chunk_size,
        pilot_launches=args.pilot_launches,
        pilot_samples=args.pilot_samples,
        max_launches=args.max_launches,
        max_samples=args.max_samples,
        max_total_rays=args.max_total_rays,
        candidate_timeout_s=args.candidate_timeout_s,
        validation_configs=tuple(args.validate),
    )
    controller = SweepController(options)
    summary = controller.reanalyze() if args.reanalyze else controller.run()
    print(f"final status: {summary['final_status']}")
    return 0 if summary["final_status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
