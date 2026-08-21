"""Host, container and GPU memory instrumentation for long solver runs (CHE-70).

This is the measurement layer the CHE-70 memory guards are built on. It reads
``/proc`` and the cgroup v2 files directly rather than taking a dependency on
``psutil``, which is not installed in either project image -- and which would
not help anyway, because the one signal that matters here is a cgroup file
``psutil`` does not expose.

Why the container's swap and not the host's
-------------------------------------------
CHE-64 measured that **host swap is non-zero at rest on this machine** (about
700 MiB across ``/swapfile`` and ``/dev/sda1``, with no project process
running), so ``/proc/meminfo``'s ``SwapFree`` moves for reasons that have
nothing to do with this benchmark. A guard keyed on it would fire on unrelated
host activity and, worse, would read as a failure of the run it aborted.

``/sys/fs/cgroup/memory.swap.current`` is per-container and starts at 0 in the
``agent_solver`` images, so a *delta* on it is attributable. That is the signal
CHE-64's pytest guardrail uses and it is the signal used here;
``tests/test_resources.py`` pins the two to the same path so they cannot drift
apart into two different definitions of "swap".

Requested is not actual
-----------------------
Every reading here is a measurement. Nothing in this module accepts a
"configured" value and echoes it back: the GPU numbers come from
``torch.cuda.mem_get_info`` and ``torch.cuda.memory_stats``, the host numbers
from ``/proc``, and a reading that cannot be taken is reported as ``None``
rather than as a plausible default.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CGROUP_MEMORY_CURRENT",
    "CGROUP_SWAP_CURRENT",
    "GpuMemorySnapshot",
    "HostMemorySnapshot",
    "MemoryWatchdog",
    "MemoryWatchdogVerdict",
    "cuda_memory_snapshot",
    "cuda_reset_peak_stats",
    "host_memory_snapshot",
    "host_reserve_bytes",
    "process_rss_bytes",
]

#: The cgroup v2 files. Same paths as ``scripts/pytest_resource_profile.py``
#: (CHE-64); ``tests/test_resources.py`` asserts the equality so a change in one
#: place cannot leave the two guards measuring different things.
CGROUP_SWAP_CURRENT = Path("/sys/fs/cgroup/memory.swap.current")
CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_SELF_STATM = Path("/proc/self/statm")

#: Phase 16's floor: never plan to leave the operating system and unrelated
#: processes less than this. A floor, not a target.
HOST_RESERVE_FLOOR_BYTES = 4 * 1024**3
HOST_RESERVE_FRACTION = 0.15


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _meminfo_bytes(key: str) -> int | None:
    try:
        for line in _PROC_MEMINFO.read_text().splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_bytes() -> int:
    """Resident set size of this process, in bytes.

    Read from ``/proc/self/statm`` (page counts) rather than ``VmRSS`` in
    ``/proc/self/status``, because statm is a single short line and this is
    sampled on a polling thread.
    """
    try:
        fields = _PROC_SELF_STATM.read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):  # pragma: no cover - /proc absent
        return 0


def host_reserve_bytes(total_bytes: int | None = None) -> int:
    """Phase 16's envelope: ``max(4 GiB, 15% of total system RAM)``."""
    if total_bytes is None:
        total_bytes = _meminfo_bytes("MemTotal") or 0
    return max(HOST_RESERVE_FLOOR_BYTES, int(HOST_RESERVE_FRACTION * total_bytes))


@dataclass(frozen=True)
class HostMemorySnapshot:
    """One instant of host and container memory state.

    ``cgroup_swap_bytes`` is the guarded quantity. ``host_swap_used_bytes`` is
    recorded alongside it precisely so the report can *show* that the host
    number is non-zero and moving for unrelated reasons, rather than leaving a
    reader to wonder why it was ignored.
    """

    monotonic_s: float
    rss_bytes: int
    mem_total_bytes: int | None
    mem_available_bytes: int | None
    cgroup_memory_bytes: int | None
    cgroup_swap_bytes: int | None
    host_swap_total_bytes: int | None
    host_swap_used_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def host_memory_snapshot() -> HostMemorySnapshot:
    swap_total = _meminfo_bytes("SwapTotal")
    swap_free = _meminfo_bytes("SwapFree")
    return HostMemorySnapshot(
        monotonic_s=time.monotonic(),
        rss_bytes=process_rss_bytes(),
        mem_total_bytes=_meminfo_bytes("MemTotal"),
        mem_available_bytes=_meminfo_bytes("MemAvailable"),
        cgroup_memory_bytes=_read_int(CGROUP_MEMORY_CURRENT),
        cgroup_swap_bytes=_read_int(CGROUP_SWAP_CURRENT),
        host_swap_total_bytes=swap_total,
        host_swap_used_bytes=(
            None if swap_total is None or swap_free is None else swap_total - swap_free
        ),
    )


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """CUDA allocator and device state, read from torch rather than declared.

    ``available`` is false when torch has no CUDA device; every field is then
    ``None`` so a caller cannot mistake an absent measurement for zero usage.
    """

    available: bool
    device: str | None = None
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_memory_snapshot(*, synchronize: bool = True) -> GpuMemorySnapshot:
    """Read CUDA memory state. ``synchronize`` first, so the numbers are settled.

    Phase 14 asks for allocated/reserved/free/peak. ``mem_get_info`` gives the
    *device* view (what other processes leave available) while
    ``memory_allocated``/``memory_reserved`` give the *allocator* view; both are
    recorded because a run can be bounded in one and not the other.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is pinned in both images
        return GpuMemorySnapshot(available=False)
    if not torch.cuda.is_available():
        return GpuMemorySnapshot(available=False)
    if synchronize:
        torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return GpuMemorySnapshot(
        available=True,
        device=str(torch.cuda.current_device()),
        allocated_bytes=int(torch.cuda.memory_allocated()),
        reserved_bytes=int(torch.cuda.memory_reserved()),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
    )


def cuda_reset_peak_stats() -> None:
    """Zero the CUDA peak counters, so a per-candidate peak means that candidate."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class MemoryWatchdogVerdict:
    """Why the watchdog asked for a stop, or that it never did."""

    breached: bool = False
    #: One of ``swap_growth`` / ``mem_available`` / ``process_rss``.
    reason: str | None = None
    detail: str | None = None
    observed_bytes: int | None = None
    limit_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryWatchdog:
    """Polls host and container memory while a candidate runs (Phase 14/16/17).

    Three independent trip conditions, each with its own recorded reason:

    * container swap grew at all above the baseline (Phase 17: ``== 0`` required);
    * host ``MemAvailable`` fell to or below the reserve (Phase 16);
    * this process's RSS exceeded its declared budget.

    The watchdog *reports*; it does not kill. Whoever owns the process decides
    what to do -- the child cooperatively aborts at its next chunk boundary and
    the parent terminates it if it does not. A polling thread that called
    ``os._exit`` would make the failure unattributable, which is the opposite of
    the point.
    """

    interval_s: float = 0.25
    host_reserve_bytes: int | None = None
    process_rss_budget_bytes: int | None = None
    #: Container swap growth tolerated before tripping. Zero: Phase 17 is a
    #: hard requirement, and this file starts at 0 inside the image.
    swap_growth_tolerance_bytes: int = 0

    baseline: HostMemorySnapshot = field(init=False)
    peak_rss_bytes: int = field(init=False, default=0)
    peak_cgroup_swap_bytes: int | None = field(init=False, default=None)
    min_mem_available_bytes: int | None = field(init=False, default=None)
    samples: int = field(init=False, default=0)
    verdict: MemoryWatchdogVerdict = field(init=False, default_factory=MemoryWatchdogVerdict)

    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _stop: threading.Event = field(init=False, default_factory=threading.Event, repr=False)

    def __post_init__(self) -> None:
        self.baseline = host_memory_snapshot()
        if self.host_reserve_bytes is None:
            self.host_reserve_bytes = host_reserve_bytes(self.baseline.mem_total_bytes)
        self._absorb(self.baseline)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> MemoryWatchdog:
        if self._thread is not None:  # pragma: no cover - defensive
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(target=self._run, daemon=True, name="che70-memory")
        self._thread.start()
        return self

    def stop(self) -> MemoryWatchdog:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.sample()
        return self

    def __enter__(self) -> MemoryWatchdog:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _run(self) -> None:  # pragma: no cover - timing-dependent thread body
        while not self._stop.wait(self.interval_s):
            self.sample()

    # -- measurement -------------------------------------------------------
    def sample(self) -> HostMemorySnapshot:
        """Take one reading, fold it into the peaks, and evaluate the guards."""
        snapshot = host_memory_snapshot()
        self._absorb(snapshot)
        return snapshot

    def _absorb(self, snapshot: HostMemorySnapshot) -> None:
        self.samples += 1
        self.peak_rss_bytes = max(self.peak_rss_bytes, snapshot.rss_bytes)
        if snapshot.cgroup_swap_bytes is not None:
            self.peak_cgroup_swap_bytes = (
                snapshot.cgroup_swap_bytes
                if self.peak_cgroup_swap_bytes is None
                else max(self.peak_cgroup_swap_bytes, snapshot.cgroup_swap_bytes)
            )
        if snapshot.mem_available_bytes is not None:
            self.min_mem_available_bytes = (
                snapshot.mem_available_bytes
                if self.min_mem_available_bytes is None
                else min(self.min_mem_available_bytes, snapshot.mem_available_bytes)
            )
        if self.verdict.breached:
            return

        growth = self.swap_growth_bytes
        if growth is not None and growth > self.swap_growth_tolerance_bytes:
            self.verdict = MemoryWatchdogVerdict(
                breached=True,
                reason="swap_growth",
                detail=(
                    f"container swap grew by {growth} B above the "
                    f"{self.baseline.cgroup_swap_bytes} B baseline "
                    f"({CGROUP_SWAP_CURRENT})"
                ),
                observed_bytes=growth,
                limit_bytes=self.swap_growth_tolerance_bytes,
            )
            return
        reserve = self.host_reserve_bytes or 0
        if snapshot.mem_available_bytes is not None and snapshot.mem_available_bytes <= reserve:
            self.verdict = MemoryWatchdogVerdict(
                breached=True,
                reason="mem_available",
                detail=(
                    f"host MemAvailable fell to {snapshot.mem_available_bytes} B, "
                    f"at or below the {reserve} B reserve"
                ),
                observed_bytes=snapshot.mem_available_bytes,
                limit_bytes=reserve,
            )
            return
        budget = self.process_rss_budget_bytes
        if budget is not None and snapshot.rss_bytes > budget:
            self.verdict = MemoryWatchdogVerdict(
                breached=True,
                reason="process_rss",
                detail=f"process RSS {snapshot.rss_bytes} B exceeded its {budget} B budget",
                observed_bytes=snapshot.rss_bytes,
                limit_bytes=budget,
            )

    # -- derived -----------------------------------------------------------
    @property
    def swap_growth_bytes(self) -> int | None:
        """``peak_swap - swap_before`` on the container's cgroup, or ``None``.

        Phase 17's ``swap_delta_peak``. ``None`` means the cgroup file could not
        be read, which is reported rather than treated as zero.
        """
        if self.peak_cgroup_swap_bytes is None or self.baseline.cgroup_swap_bytes is None:
            return None
        return max(0, self.peak_cgroup_swap_bytes - self.baseline.cgroup_swap_bytes)

    def report(self) -> dict[str, Any]:
        final = host_memory_snapshot()
        return {
            "signal": str(CGROUP_SWAP_CURRENT),
            "signal_rationale": (
                "container cgroup swap, not host swap: host swap is non-zero at "
                "rest on this machine (CHE-64), so a host-swap delta is not "
                "attributable to this run"
            ),
            "samples": self.samples,
            "interval_s": self.interval_s,
            "before": self.baseline.as_dict(),
            "after": final.as_dict(),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cgroup_swap_bytes": self.peak_cgroup_swap_bytes,
            "cgroup_swap_before_bytes": self.baseline.cgroup_swap_bytes,
            "cgroup_swap_after_bytes": final.cgroup_swap_bytes,
            "swap_delta_peak_bytes": self.swap_growth_bytes,
            "min_mem_available_bytes": self.min_mem_available_bytes,
            "host_reserve_bytes": self.host_reserve_bytes,
            "process_rss_budget_bytes": self.process_rss_budget_bytes,
            "verdict": self.verdict.as_dict(),
        }
