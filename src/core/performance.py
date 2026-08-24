"""A declared, reproducible performance measurement, and the rule for comparing two.

CHE-105 (M0.4). Every performance number this repository has was produced by a
one-off scratch probe, and CHE-101's report says so in as many words: *"Timed
with a scratch probe that is deliberately not committed -- a wall-clock number
is not reproducible across hosts."* The instinct is right and the conclusion was
wrong. What follows from "a wall-clock number is not reproducible across hosts"
is not "do not commit it"; it is "commit it with enough of the host attached
that a later reader can tell whether their number is comparable".

The cost of not doing that is on the record. CHE-96 attributed the whole of
demo3's runtime to the reconstruction stage. CHE-101 then made that stage 9.6x
faster on the kernel and the end-to-end run went 207 s -> 197 s, because the
stage was 7% of the cost. A committed, stage-resolved baseline would have said
so before the work rather than after it.

The two fingerprints, and why they are different
------------------------------------------------
``core.provenance`` already has a fingerprint, and reusing it here would be a
category error in both directions.

* The **scientific** fingerprint deliberately projects timings OUT
  (``VOLATILE_KEYS``), because a result that changes when you run it on a busier
  machine is not a different result. Correct, and useless for performance.
* The **environment** fingerprint here covers exactly the things that change
  speed *without* changing the answer: GPU model and driver, container image,
  thread counts, CPU affinity, and the pinned numerics packages.

Two records are comparable when their environment fingerprints match. When they
do not, :func:`compare` **refuses** rather than returning a ratio, because the
ratio would be a number and numbers get quoted.

The one domain rule baked in
----------------------------
:class:`Workload` requires a unit, and cost-per-unit is only reported alongside
it. This is not ceremony. The demo3 record already carries the warning:
``seconds_per_ray`` means something different on ``ramp_sum``, which is
O(rays x pixels), than on ``kspace_splat``, which is O(rays) plus one FFT. Two
numbers with the same name and different meanings are worse than no number, so
the route is part of the fingerprint and the unit is mandatory.

Isolation is never claimed unless it was applied
------------------------------------------------
``benchmarks/protocols/coupler_protocol.yaml`` sets the discipline this module
follows: declared warmup, declared repeats, median as the primary statistic with
min and p95 beside it, explicit device synchronization around the timed region.
Thread counts and CPU affinity are *recorded as observed* rather than asserted;
:attr:`Isolation.applied` is false unless something actually pinned them.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from core.resources import (
    GpuMemorySnapshot,
    MemoryWatchdog,
    cuda_memory_snapshot,
    cuda_reset_peak_stats,
)

__all__ = [
    "EnvironmentFingerprint",
    "Incomparable",
    "Isolation",
    "Measurement",
    "PerformanceRecord",
    "ScalingFit",
    "StageTimer",
    "SwapGrowthAbort",
    "Workload",
    "compare",
    "environment_fingerprint",
    "fit_scaling",
    "measure",
]

#: Environment variables that set thread counts for the numerics stack. Recorded
#: because they change speed by large factors and change no answer, which is the
#: definition of what belongs in this fingerprint.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
)

_PERF_PACKAGES = ("numpy", "scipy", "jax", "jaxlib", "torch", "optiland", "chromatix")


class SwapGrowthAbort(RuntimeError):
    """The workload started using swap as working memory, so it was stopped.

    ``AGENTS.md``: *"Do not use swap as working memory. Growth in the workload's
    cgroup swap is a stop condition: terminate the workload and report the
    resource failure instead of continuing."* This is that failure, as an
    exception rather than a log line, so a harness cannot continue past it by
    accident and a caller cannot mistake a swapping run's timing for a result.
    """

    def __init__(self, growth_bytes: int, report: dict[str, Any]) -> None:
        super().__init__(
            f"cgroup swap grew by {growth_bytes} bytes during the measurement. "
            "The run is stopped and reported as a resource failure rather than "
            "timed: a workload that is swapping is measuring the disk. Reduce the "
            "batch, chunk the workload, or run it somewhere with more memory."
        )
        self.growth_bytes = growth_bytes
        self.report = report


class Incomparable(RuntimeError):
    """Two records were measured in environments that are not the same.

    Raised rather than returning a ratio. A ratio between records taken on
    different GPUs, or with different thread counts, is a number -- and a number
    in a report gets quoted without its caveat.
    """


@dataclass(frozen=True)
class Isolation:
    """What was actually done to isolate the measurement, as observed.

    ``applied`` is the honest field. Recording ``cpu_affinity`` proves nothing
    on its own -- every process has one. ``applied`` is true only when the
    caller states that it pinned something.
    """

    applied: bool
    cpu_affinity: tuple[int, ...] | None
    thread_env: dict[str, str]
    logical_cpus: int | None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cpu_affinity": list(self.cpu_affinity or ())}


def _observed_isolation(*, applied: bool = False) -> Isolation:
    try:
        affinity = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        affinity = None
    return Isolation(
        applied=applied,
        cpu_affinity=affinity,
        thread_env={k: os.environ[k] for k in _THREAD_VARS if k in os.environ},
        logical_cpus=os.cpu_count(),
    )


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except Exception:  # pragma: no cover - no GPU, or nvidia-smi absent
        return None
    return out.splitlines()[0].strip() if out else None


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Everything that changes speed without changing the answer."""

    python_version: str
    platform: str
    cpu_model: str | None
    logical_cpus: int | None
    thread_env: dict[str, str]
    container_image: str
    gpu_name: str | None
    gpu_driver: str | None
    #: How many CUDA devices this process can see. Part of the fingerprint
    #: because AGENTS.md requires GPU measurements to be single-GPU, and a count
    #: read from the runtime is the only form of that claim a record can carry --
    #: `--gpus device=6` is a host-side flag the container cannot observe, but
    #: its consequence, exactly one visible device, is observable.
    gpu_count: int | None
    visible_devices: str | None
    packages: dict[str, str]

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sha256": self.sha256}

    def differences(self, other: EnvironmentFingerprint) -> list[str]:
        """Which fields differ, so a refusal can say why rather than just refuse."""
        mine, theirs = asdict(self), asdict(other)
        return [key for key in sorted(mine) if mine[key] != theirs[key]]


def _gpu_count() -> int | None:
    """Visible CUDA devices, read from the runtime. ``None`` when there is no CUDA."""
    listing = _nvidia_smi("name")
    if listing is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except Exception:  # pragma: no cover - nvidia-smi vanished between calls
        return None
    return len([line for line in out.splitlines() if line.strip()])


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - non-Linux
        return None
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def environment_fingerprint() -> EnvironmentFingerprint:
    """Read, never declared. Absent hardware is ``None``, never a default."""
    import importlib.metadata

    versions: dict[str, str] = {}
    for name in _PERF_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count(),
        thread_env={k: os.environ[k] for k in _THREAD_VARS if k in os.environ},
        container_image=os.environ.get("MOA_IMAGE", "agent_solver"),
        gpu_name=_nvidia_smi("name"),
        gpu_driver=_nvidia_smi("driver_version"),
        gpu_count=_gpu_count(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("MOA_GPUS"),
        packages=versions,
    )


@dataclass(frozen=True)
class Workload:
    """How much work was done, in the unit that makes cost-per-unit mean something.

    ``route`` is not decoration. Cost per ray on ``ramp_sum`` (O(rays x pixels))
    and cost per ray on ``kspace_splat`` (O(rays) + one FFT) are different
    quantities wearing one name, and the demo3 record already warns about it.
    Anything that changes the asymptotic cost model belongs here.
    """

    size: float
    unit: str
    route: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageTimer:
    """Per-stage wall clock, accumulated across chunks.

    Generalizes the ``stage_s`` dict ``demo3_hologram_lens.py`` grew by hand, so
    a stage breakdown stops being something each probe reimplements. Stages
    accumulate rather than overwrite, because these workloads run in chunks and
    a stage is entered once per chunk.
    """

    stages: dict[str, float] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self.stages:
            self.stages[name] = 0.0
            self._order.append(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] += time.perf_counter() - started

    def as_dict(self, total_s: float | None = None) -> dict[str, Any]:
        """Stage seconds, and each stage's share of the total when one is given.

        The share is the number CHE-101 needed and did not have: a stage that is
        7% of the run cannot be made to matter by a 9.6x kernel speedup, and the
        percentage is what says so at a glance.
        """
        payload: dict[str, Any] = {
            "seconds": {name: round(self.stages[name], 4) for name in self._order},
            "accounted_s": round(sum(self.stages.values()), 4),
        }
        if total_s:
            payload["fraction_of_total"] = {
                name: round(self.stages[name] / total_s, 4) for name in self._order
            }
            payload["unaccounted_s"] = round(total_s - sum(self.stages.values()), 4)
        return payload


@dataclass(frozen=True)
class ScalingFit:
    """A fitted exponent, not two endpoints joined by a claim."""

    axis: str
    exponent: float
    intercept_log10: float
    r_squared: float
    points: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "points": [list(p) for p in self.points]}


def fit_scaling(points: Iterable[tuple[float, float]], *, axis: str) -> ScalingFit:
    """Least squares on log10(cost) against log10(size).

    Requires three points. Two points always fit a line exactly, which produces
    an exponent with no evidence behind it and an r^2 of 1.0 that means nothing
    -- the protocol asks for a fit precisely so the number can be doubted.
    """
    pairs = [(float(x), float(y)) for x, y in points if x > 0 and y > 0]
    if len(pairs) < 3:
        raise ValueError(
            f"a fitted exponent needs at least 3 positive points, got {len(pairs)}. "
            "Two points fit a line exactly and report r^2 = 1 regardless of whether "
            "the relationship is a power law."
        )
    xs = [math.log10(x) for x, _ in pairs]
    ys = [math.log10(y) for _, y in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    return ScalingFit(
        axis=axis,
        exponent=slope,
        intercept_log10=intercept,
        r_squared=1.0 - ss_res / ss_tot if ss_tot else 1.0,
        points=tuple(pairs),
    )


@dataclass(frozen=True)
class Measurement:
    """The timing statistics of one measured thing."""

    repeats: int
    warmup: int
    median_s: float
    min_s: float
    p95_s: float
    all_s: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "all_s": list(self.all_s)}


@dataclass(frozen=True)
class PerformanceRecord:
    """One measured workload, with everything needed to know if it is comparable."""

    label: str
    workload: Workload
    measurement: Measurement
    environment: EnvironmentFingerprint
    isolation: Isolation
    stages: dict[str, Any] | None = None
    peak_host_rss_bytes: int | None = None
    cuda: GpuMemorySnapshot | None = None
    swap_growth_bytes: int | None = None
    memory_report: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()

    @property
    def cost_per_unit(self) -> float | None:
        if not self.workload.size:
            return None
        return self.measurement.median_s / self.workload.size

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "workload": self.workload.as_dict(),
            "measurement": self.measurement.as_dict(),
            "cost_per_unit": self.cost_per_unit,
            "cost_per_unit_name": (
                f"seconds_per_{self.workload.unit}"
                + (f" [{self.workload.route}]" if self.workload.route else "")
            ),
            "environment": self.environment.as_dict(),
            "isolation": self.isolation.as_dict(),
            "stages": self.stages,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "cuda": self.cuda.as_dict() if self.cuda else None,
            "swap_growth_bytes": self.swap_growth_bytes,
            "memory_report": self.memory_report,
            "notes": list(self.notes),
        }


def _synchronize() -> None:
    """Settle both device backends before reading a clock.

    Both, not one. JAX and torch each have their own async queue and this
    repository runs graphs that touch both, so synchronizing only the backend a
    caller happens to be thinking about would time a dispatch rather than a
    computation.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - torch absent or no CUDA
        pass
    try:
        import jax

        for device in jax.devices():
            if device.platform != "cpu":
                jax.block_until_ready(jax.numpy.zeros(1, device=device))
    except Exception:  # pragma: no cover - jax absent
        pass


def measure(
    fn: Callable[[StageTimer], Any],
    *,
    label: str,
    workload: Workload,
    repeats: int = 3,
    warmup: int = 1,
    isolation_applied: bool = False,
    notes: Iterable[str] = (),
    watchdog_interval_s: float = 0.25,
) -> tuple[PerformanceRecord, Any]:
    """Time ``fn`` under the protocol, and return the record and its last result.

    ``fn`` receives a fresh :class:`StageTimer` per call; the timer from the
    final *timed* repeat is the one recorded, because a stage breakdown averaged
    over repeats hides which repeat was slow.

    The memory watchdog runs for the whole call, warmup included. Swap growth
    raises :class:`SwapGrowthAbort` rather than being reported at the end: a
    swapping measurement is not a slow measurement, it is a measurement of the
    disk, and continuing would produce a number.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    environment = environment_fingerprint()
    isolation = _observed_isolation(applied=isolation_applied)
    cuda_reset_peak_stats()

    timings: list[float] = []
    timer = StageTimer()
    result: Any = None

    watchdog = MemoryWatchdog(interval_s=watchdog_interval_s).start()
    try:
        for index in range(warmup + repeats):
            timer = StageTimer()
            _synchronize()
            started = time.perf_counter()
            result = fn(timer)
            _synchronize()
            elapsed = time.perf_counter() - started

            growth = watchdog.swap_growth_bytes
            if growth:
                raise SwapGrowthAbort(growth, watchdog.report())
            if index >= warmup:
                timings.append(elapsed)
    finally:
        watchdog.stop()

    ordered = sorted(timings)
    measurement = Measurement(
        repeats=repeats,
        warmup=warmup,
        median_s=statistics.median(timings),
        min_s=ordered[0],
        # For repeats <= 20 this is the slowest observed run rather than a real
        # 95th percentile, and it is reported under that name because the
        # protocol asks for a tail statistic and the honest tail of 3 samples is
        # the maximum. Not interpolated -- interpolating 3 points would invent a
        # smoothness the sample does not have.
        p95_s=ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        all_s=tuple(timings),
    )

    record = PerformanceRecord(
        label=label,
        workload=workload,
        measurement=measurement,
        environment=environment,
        isolation=isolation,
        stages=timer.as_dict(measurement.median_s) if timer.stages else None,
        peak_host_rss_bytes=watchdog.peak_rss_bytes,
        cuda=cuda_memory_snapshot(),
        swap_growth_bytes=watchdog.swap_growth_bytes or 0,
        memory_report=watchdog.report(),
        notes=tuple(notes),
    )
    return record, result


def compare(
    baseline: PerformanceRecord | dict[str, Any],
    candidate: PerformanceRecord | dict[str, Any],
) -> dict[str, Any]:
    """Speedup of ``candidate`` over ``baseline``, or a refusal.

    Refuses on two grounds, and the second is the one specific to this
    repository: a different environment fingerprint, and a different workload
    unit or route. Comparing ``seconds_per_ray`` across ``ramp_sum`` and
    ``kspace_splat`` is comparing two different cost models, and the answer
    would look like a speedup.
    """
    a = baseline.as_dict() if isinstance(baseline, PerformanceRecord) else baseline
    b = candidate.as_dict() if isinstance(candidate, PerformanceRecord) else candidate

    a_env, b_env = a["environment"], b["environment"]
    if a_env["sha256"] != b_env["sha256"]:
        differing = [
            key
            for key in sorted(set(a_env) | set(b_env))
            if key != "sha256" and a_env.get(key) != b_env.get(key)
        ]
        raise Incomparable(
            f"{a['label']!r} and {b['label']!r} were measured in different "
            f"environments and will not be compared. Differing: {differing}. "
            "Re-measure the baseline in this environment; a ratio across hosts is "
            "a number that will be quoted without this caveat."
        )

    a_unit, b_unit = a["workload"]["unit"], b["workload"]["unit"]
    a_route, b_route = a["workload"].get("route"), b["workload"].get("route")
    if a_unit != b_unit or a_route != b_route:
        raise Incomparable(
            f"workloads are not the same kind of work: {a_unit}/{a_route} vs "
            f"{b_unit}/{b_route}. Cost per unit means something different on each, "
            "so the ratio would not be a speedup."
        )

    a_med = a["measurement"]["median_s"]
    b_med = b["measurement"]["median_s"]
    return {
        "baseline": a["label"],
        "candidate": b["label"],
        "environment_sha256": a_env["sha256"],
        "median_s": {"baseline": a_med, "candidate": b_med},
        "speedup": a_med / b_med if b_med else None,
        "cost_per_unit": {
            "baseline": a.get("cost_per_unit"),
            "candidate": b.get("cost_per_unit"),
            "unit": a_unit,
            "route": a_route,
        },
    }
