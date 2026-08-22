"""The memory instrumentation the CHE-70 guards are built on (Phase 14/16/17).

The single most important test here is
``test_the_swap_signal_is_the_same_file_che64_guards``. CHE-64 established, by
measurement, that host swap is non-zero at rest on this machine, so a host-swap
delta is not attributable to any one run and the container's cgroup file is the
only usable signal. If this module and CHE-64's pytest guardrail ever came to
read different files, the project would have two different definitions of "did
this cause swapping" and one of them would be wrong. The test pins them together.

The watchdog's three trip conditions are each driven past their threshold with
injected readings. Exhausting 377 GiB of host RAM to test a memory guard on a
shared server is exactly the outcome the guard exists to prevent, so what is
faked is *when* the threshold is crossed -- never the detection path, the recorded
reason, or the arithmetic.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.resources import (  # noqa: E402
    CGROUP_MEMORY_CURRENT,
    CGROUP_SWAP_CURRENT,
    HOST_RESERVE_FLOOR_BYTES,
    MemoryWatchdog,
    cuda_memory_snapshot,
    host_memory_snapshot,
    host_reserve_bytes,
    process_rss_bytes,
)


class TestSignalParity:
    def test_the_swap_signal_is_the_same_file_che64_guards(self):
        """One definition of "swap", shared with `scripts/pytest_resource_profile.py`."""
        from scripts.pytest_resource_profile import (
            CGROUP_MEMORY_CURRENT as profiler_memory,
        )
        from scripts.pytest_resource_profile import (
            CGROUP_SWAP_CURRENT as profiler_swap,
        )

        assert profiler_swap == CGROUP_SWAP_CURRENT
        assert profiler_memory == CGROUP_MEMORY_CURRENT

    def test_the_cgroup_files_are_readable_inside_the_container(self):
        snapshot = host_memory_snapshot()
        assert snapshot.cgroup_swap_bytes is not None, (
            f"{CGROUP_SWAP_CURRENT} is unreadable; the swap guard would have no signal"
        )
        assert snapshot.cgroup_memory_bytes is not None
        assert snapshot.cgroup_swap_bytes >= 0

    def test_host_swap_is_recorded_alongside_but_is_not_the_signal(self):
        """Recorded so a report can *show* why the host number was not used."""
        snapshot = host_memory_snapshot()
        assert snapshot.host_swap_total_bytes is not None
        assert snapshot.host_swap_used_bytes is not None


class TestSnapshots:
    def test_the_host_snapshot_reads_real_numbers(self):
        snapshot = host_memory_snapshot()
        assert snapshot.rss_bytes > 0
        assert snapshot.mem_total_bytes and snapshot.mem_total_bytes > 0
        assert snapshot.mem_available_bytes and snapshot.mem_available_bytes > 0
        assert snapshot.mem_available_bytes <= snapshot.mem_total_bytes
        assert snapshot.monotonic_s > 0.0

    def test_rss_tracks_a_real_allocation(self):
        import numpy as np

        before = process_rss_bytes()
        block = np.ones(64 * 1024**2 // 8, dtype=np.float64)  # 64 MiB, touched
        block[:] = 1.0
        after = process_rss_bytes()
        del block
        assert after - before > 32 * 1024**2, (
            "RSS did not move for a touched 64 MiB allocation; the reading is not live"
        )

    def test_the_reserve_is_the_larger_of_the_floor_and_the_fraction(self):
        assert host_reserve_bytes(0) == HOST_RESERVE_FLOOR_BYTES
        assert host_reserve_bytes(1024**4) == int(0.15 * 1024**4)
        assert host_reserve_bytes(8 * 1024**3) == HOST_RESERVE_FLOOR_BYTES

    def test_an_absent_gpu_is_reported_as_absent_not_as_zero_usage(self):
        snapshot = cuda_memory_snapshot()
        if not snapshot.available:
            assert snapshot.allocated_bytes is None
            assert snapshot.free_bytes is None
        else:
            assert snapshot.total_bytes and snapshot.total_bytes > 0
            assert snapshot.allocated_bytes is not None


class TestWatchdog:
    def test_a_quiet_run_does_not_trip_anything(self):
        watchdog = MemoryWatchdog(interval_s=0.05)
        with watchdog:
            time.sleep(0.2)
        assert watchdog.verdict.breached is False
        assert watchdog.samples >= 2
        assert watchdog.peak_rss_bytes > 0
        assert watchdog.swap_growth_bytes == 0

    def test_the_report_states_which_signal_it_used_and_why(self):
        watchdog = MemoryWatchdog(interval_s=0.05)
        with watchdog:
            time.sleep(0.1)
        report = watchdog.report()
        assert report["signal"] == str(CGROUP_SWAP_CURRENT)
        assert "host swap is non-zero at rest" in report["signal_rationale"]
        assert report["swap_delta_peak_bytes"] == 0
        assert report["before"]["cgroup_swap_bytes"] is not None

    def test_container_swap_growth_trips_the_guard(self):
        """Phase 17: any growth at all, not a threshold."""
        watchdog = MemoryWatchdog(interval_s=10.0)
        baseline = watchdog.baseline
        watchdog._absorb(
            replace(baseline, cgroup_swap_bytes=(baseline.cgroup_swap_bytes or 0) + 4096)
        )
        assert watchdog.verdict.breached is True
        assert watchdog.verdict.reason == "swap_growth"
        assert watchdog.verdict.observed_bytes == 4096
        assert watchdog.verdict.limit_bytes == 0
        assert str(CGROUP_SWAP_CURRENT) in watchdog.verdict.detail
        assert watchdog.swap_growth_bytes == 4096

    def test_one_byte_of_container_swap_is_enough(self):
        watchdog = MemoryWatchdog(interval_s=10.0)
        watchdog._absorb(
            replace(
                watchdog.baseline,
                cgroup_swap_bytes=(watchdog.baseline.cgroup_swap_bytes or 0) + 1,
            )
        )
        assert watchdog.verdict.breached is True

    def test_host_headroom_below_the_reserve_trips_the_guard(self):
        watchdog = MemoryWatchdog(interval_s=10.0, host_reserve_bytes=8 * 1024**3)
        watchdog._absorb(
            replace(watchdog.baseline, mem_available_bytes=4 * 1024**3)
        )
        assert watchdog.verdict.reason == "mem_available"
        assert watchdog.verdict.limit_bytes == 8 * 1024**3

    def test_a_process_rss_budget_trips_the_guard(self):
        """A budget already exceeded at construction fires immediately.

        That is the wanted behaviour, not an artefact: the guard's first reading is
        the baseline, so a process that is *already* over budget is caught before
        it allocates a chunk rather than after.
        """
        watchdog = MemoryWatchdog(interval_s=10.0, process_rss_budget_bytes=1024)
        assert watchdog.verdict.reason == "process_rss"
        assert watchdog.verdict.observed_bytes == watchdog.baseline.rss_bytes
        assert watchdog.verdict.limit_bytes == 1024

    def test_a_budget_crossed_mid_run_trips_the_guard(self):
        budget = host_memory_snapshot().rss_bytes * 4
        watchdog = MemoryWatchdog(interval_s=10.0, process_rss_budget_bytes=budget)
        assert watchdog.verdict.breached is False
        watchdog._absorb(replace(watchdog.baseline, rss_bytes=budget + 1))
        assert watchdog.verdict.reason == "process_rss"
        assert watchdog.verdict.observed_bytes == budget + 1

    def test_the_first_breach_is_the_one_reported(self):
        """A later condition must not overwrite the reason the run actually stopped."""
        watchdog = MemoryWatchdog(interval_s=10.0, process_rss_budget_bytes=1024)
        assert watchdog.verdict.reason == "process_rss"
        watchdog._absorb(
            replace(
                watchdog.baseline,
                cgroup_swap_bytes=(watchdog.baseline.cgroup_swap_bytes or 0) + 4096,
            )
        )
        assert watchdog.verdict.reason == "process_rss"

    def test_the_peaks_and_minima_are_tracked_across_samples(self):
        watchdog = MemoryWatchdog(interval_s=10.0)
        base = watchdog.baseline
        watchdog._absorb(replace(base, rss_bytes=10, mem_available_bytes=10**12))
        watchdog._absorb(replace(base, rss_bytes=99, mem_available_bytes=10**11))
        watchdog._absorb(replace(base, rss_bytes=50, mem_available_bytes=10**13))
        assert watchdog.peak_rss_bytes >= 99
        assert watchdog.min_mem_available_bytes == 10**11

    def test_an_unreadable_cgroup_reports_none_rather_than_zero(self):
        watchdog = MemoryWatchdog(interval_s=10.0)
        object.__setattr__(watchdog, "baseline", replace(watchdog.baseline, cgroup_swap_bytes=None))
        watchdog.peak_cgroup_swap_bytes = None
        assert watchdog.swap_growth_bytes is None, (
            "an absent measurement must not be reported as zero swap growth"
        )

    def test_stopping_twice_is_harmless(self):
        watchdog = MemoryWatchdog(interval_s=0.05).start()
        watchdog.stop()
        watchdog.stop()
        assert watchdog.verdict.breached is False

    def test_starting_twice_is_refused(self):
        watchdog = MemoryWatchdog(interval_s=0.05).start()
        try:
            with pytest.raises(RuntimeError):
                watchdog.start()
        finally:
            watchdog.stop()
