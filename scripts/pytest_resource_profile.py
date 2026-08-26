"""Per-test runtime/memory profiler with a swap guardrail (CHE-64).

Opt-in pytest plugin. It is *not* loaded by default -- enable it explicitly:

    ./run.sh pytest -q -p scripts.pytest_resource_profile \\
        --resource-profile=outputs/CHE-64/tierA.json --swap-guard

Two jobs, deliberately in one plugin because the second needs the first's
bookkeeping to say *which test* was running when it fired.

1. Record, per test: wall time by phase, RSS at entry/exit, true peak RSS from a
   sampling thread, the process high-water mark, markers, and outcome.
2. Abort the session the moment the container starts swapping.

Why the swap signal is ``/sys/fs/cgroup/memory.swap.current``
------------------------------------------------------------
The obvious check -- "is any swap in use?" -- is useless here and would have
produced a guardrail that fires on every run forever. Measured on this host
inside the container:

    SwapTotal   1955610616 kB      (1.82 TiB)
    SwapFree    1954862184 kB      => ~748 MiB ALREADY in use at rest
    pswpout       44610212         cumulative pages out, since boot

``/proc/meminfo`` inside the container is the *host's*, and this is a shared GPU
server, so both of those reflect whatever anyone else has ever run. Neither an
absolute threshold nor "swap used > 0" can distinguish "this test suite started
thrashing" from "someone's job swapped last Tuesday".

``/sys/fs/cgroup/memory.swap.current`` is the container's own swap charge under
cgroup v2. It reads ``0`` at rest, is scoped to this container, and rises only
when *our* pages go out. That is the signal the guardrail gates on.

Host ``SwapFree`` and ``pswpout`` are still recorded, as deltas from a baseline
taken at session start, for two reasons: they attribute pressure we caused that
landed in another cgroup, and they let a reader tell "we swapped" apart from "the
box was already swapping when we started".

Both ``memory.max`` and ``memory.swap.max`` read ``max`` on this host, so there
is no cgroup ceiling to hit before the host's -- the guardrail is the only thing
standing between a memory-hungry test and host-wide memory pressure.
"""

from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest

#: Container-scoped swap charge (cgroup v2). Reads 0 at rest. THE gate signal.
CGROUP_SWAP_CURRENT = Path("/sys/fs/cgroup/memory.swap.current")
CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
PROC_MEMINFO = Path("/proc/meminfo")
PROC_VMSTAT = Path("/proc/vmstat")
PROC_SELF_STATM = Path("/proc/self/statm")

#: Sampling period for the peak-RSS / swap watcher thread. 50 ms keeps the
#: overhead unmeasurable against a 43 s Tier A while still catching the peak of
#: a sub-second test.
SAMPLE_INTERVAL_S = 0.05

#: Default swap-growth tolerance, in KiB. Not zero: cgroup swap accounting can
#: tick by a page or two from unrelated kernel bookkeeping, and a guardrail that
#: cries wolf gets disabled by the next person. 4 MiB is far below anything that
#: would degrade CI and far above accounting noise.
DEFAULT_SWAP_GUARD_KIB = 4 * 1024

#: Write the profile to disk every N completed tests, so an outer timeout or a
#: kill -9 leaves usable partial results instead of nothing.
FLUSH_EVERY = 5

#: Also flush immediately after any test at least this slow, so expensive tests
#: are never the ones lost.
FLUSH_IF_SLOWER_THAN_S = 2.0

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _meminfo_kib(key: str) -> int | None:
    try:
        match = re.search(rf"^{key}:\s+(\d+)", PROC_MEMINFO.read_text(), re.M)
    except OSError:
        return None
    return int(match.group(1)) if match else None


def _vmstat(key: str) -> int | None:
    try:
        match = re.search(rf"^{key} (\d+)", PROC_VMSTAT.read_text(), re.M)
    except OSError:
        return None
    return int(match.group(1)) if match else None


def _rss_kib() -> int:
    """Current RSS in KiB, from /proc/self/statm (cheap: one small read)."""
    try:
        return int(PROC_SELF_STATM.read_text().split()[1]) * _PAGE_SIZE // 1024
    except (OSError, IndexError, ValueError):
        return 0


def _swap_snapshot() -> dict[str, int | None]:
    return {
        "cgroup_swap_current_kib": (
            None if (v := _read_int(CGROUP_SWAP_CURRENT)) is None else v // 1024
        ),
        "cgroup_memory_current_kib": (
            None if (v := _read_int(CGROUP_MEMORY_CURRENT)) is None else v // 1024
        ),
        "host_swap_free_kib": _meminfo_kib("SwapFree"),
        "host_mem_available_kib": _meminfo_kib("MemAvailable"),
        "host_pswpout": _vmstat("pswpout"),
    }


class _Watcher(threading.Thread):
    """Samples RSS and container swap while one test runs.

    Runs as a daemon thread so an aborted session can never be held open by it.
    It does not touch pytest state; the main thread reads the results.
    """

    def __init__(self, plugin: ResourceProfiler) -> None:
        super().__init__(daemon=True, name="che64-resource-watcher")
        self._plugin = plugin
        # NOT `_stop`: threading.Thread._stop is a real internal method that
        # join() invokes, and shadowing it with an Event breaks teardown.
        self._stop_event = threading.Event()
        self.peak_rss_kib = _rss_kib()
        self.peak_cgroup_swap_kib = _read_int(CGROUP_SWAP_CURRENT)
        self.peak_cgroup_swap_kib = (
            None if self.peak_cgroup_swap_kib is None else self.peak_cgroup_swap_kib // 1024
        )

    def run(self) -> None:
        self._sample()
        while not self._stop_event.wait(SAMPLE_INTERVAL_S):
            self._sample()

    def _sample(self) -> None:
        self.peak_rss_kib = max(self.peak_rss_kib, _rss_kib())
        raw = _read_int(CGROUP_SWAP_CURRENT)
        if raw is None:
            return
        swap_kib = raw // 1024
        if self.peak_cgroup_swap_kib is None or swap_kib > self.peak_cgroup_swap_kib:
            self.peak_cgroup_swap_kib = swap_kib
        self._plugin.check_swap(swap_kib)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)


class ResourceProfiler:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.out_path: Path | None = None
        if raw := config.getoption("resource_profile"):
            self.out_path = Path(raw)
        self.guard_enabled: bool = bool(config.getoption("swap_guard"))
        self.guard_kib: int = int(config.getoption("swap_guard_kib"))

        self.records: list[dict[str, Any]] = []
        self.session_start: float = time.time()
        self.baseline: dict[str, int | None] = _swap_snapshot()
        self.baseline_swap_kib: int = self.baseline["cgroup_swap_current_kib"] or 0

        #: Set by the watcher thread the first time the guard trips. Read by the
        #: main thread; a plain dict assignment is atomic enough for one writer.
        self.swap_breach: dict[str, Any] | None = None
        self._session_finished = False
        self._current: dict[str, Any] | None = None
        self._watcher: _Watcher | None = None

    # -- guardrail ---------------------------------------------------------

    def check_swap_now(self) -> None:
        """Read the gate signal and check it on the CALLING thread.

        The sampling thread alone is not sufficient: a test that finishes inside
        ``SAMPLE_INTERVAL_S`` is never sampled, so it would never be checked. This
        is called at every test boundary from the main thread so that detection
        does not depend on the sampler happening to land inside a slow test. One
        small ``read()`` per test is not measurable against any real test.
        """
        raw = _read_int(CGROUP_SWAP_CURRENT)
        if raw is not None:
            self.check_swap(raw // 1024)

    def check_swap(self, swap_kib: int) -> None:
        """Records and escalates once. Safe to call from any thread."""
        if not self.guard_enabled or self.swap_breach is not None:
            return
        growth = swap_kib - self.baseline_swap_kib
        if growth <= self.guard_kib:
            return

        active = self._current
        self.swap_breach = {
            "detected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_into_session": round(time.time() - self.session_start, 3),
            "container_swap_kib": swap_kib,
            "baseline_container_swap_kib": self.baseline_swap_kib,
            "growth_kib": growth,
            "threshold_kib": self.guard_kib,
            "active_test": None if active is None else active["nodeid"],
            # _start is popped by _finalize, so a post-test boundary check sees a
            # record without it. Fall back to the recorded duration.
            "active_test_elapsed_s": (
                None
                if active is None
                else round(time.time() - active["_start"], 3)
                if "_start" in active
                else active.get("duration_s")
            ),
            "active_test_rss_kib_at_detection": _rss_kib(),
            "completed_tests_before_breach": len(self.records),
            "host_at_detection": _swap_snapshot(),
            "interpretation": (
                "The CONTAINER's own cgroup swap charge grew past the threshold, so "
                "these pages are ours -- this is not pre-existing host swap. The run "
                "was terminated rather than allowed to continue under memory pressure."
            ),
        }
        # Write the evidence before signalling: a SIGINT-driven teardown must not
        # be able to lose the one record that explains why it happened.
        self._flush()
        self.escalate()

    def escalate(self) -> None:
        """Terminate the session. Separate method so it can be stubbed in tests.

        Signals are delivered to the main thread in CPython, where pytest turns
        SIGINT into a KeyboardInterrupt and tears the session down with a nonzero
        exit status. That is what "terminate the run" has to mean here: setting
        ``session.shouldstop`` would let the *current* -- possibly runaway --
        test run to completion first, which is exactly the case the guardrail
        exists to interrupt.
        """
        os.kill(os.getpid(), signal.SIGINT)

    # -- per-test bookkeeping ----------------------------------------------

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None):
        self._current = {
            "nodeid": item.nodeid,
            "file": str(item.path.relative_to(self.config.rootpath)) if item.path else None,
            "name": item.name,
            "markers": sorted({m.name for m in item.iter_markers()}),
            "rss_kib_before": _rss_kib(),
            "_start": time.perf_counter(),
            "_start_wall": time.time(),
            "phase_seconds": {},
            "outcome": "unknown",
            "skip_reason": None,
        }
        # Wall-clock start is what the watcher reports elapsed against.
        self._current["_start"] = time.time()
        watcher = _Watcher(self)
        self._watcher = watcher
        watcher.start()
        self.check_swap_now()
        try:
            yield
        finally:
            watcher.stop()
            self._watcher = None
            record = self._current
            self._current = None
            if record is not None:
                self._finalize(record, watcher)
                # Flush periodically, not only at sessionfinish. A profiling run
                # long enough to be worth doing is long enough to get killed by
                # an outer timeout, and CHE-38 already lost two 25-minute runs to
                # "every measurement made, then nothing written".
                # Flush on a count boundary OR whenever the test that just ran
                # was expensive. The count rule alone is useless exactly where it
                # matters most: a chunk of 13 multi-minute tutorials can be killed
                # by an outer timeout having completed fewer than FLUSH_EVERY
                # tests, and then writes nothing at all.
                if (
                    len(self.records) % FLUSH_EVERY == 0
                    or record["duration_s"] >= FLUSH_IF_SLOWER_THAN_S
                ):
                    self._flush()
            # Checked with `record` still attributable: if the guard trips here,
            # the breach names the test that just ran rather than nothing.
            self._current = record
            try:
                self.check_swap_now()
            finally:
                self._current = None

    def _finalize(self, record: dict[str, Any], watcher: _Watcher) -> None:
        rss_after = _rss_kib()
        record["duration_s"] = round(time.time() - record.pop("_start"), 4)
        record.pop("_start_wall", None)
        record["rss_kib_after"] = rss_after
        record["peak_rss_kib"] = max(watcher.peak_rss_kib, rss_after)
        # Retained growth: what this test did NOT give back. This is the number
        # that matters for suite-level memory creep, and it is distinct from the
        # peak, which is what matters for a single-test OOM.
        record["rss_growth_kib"] = rss_after - record["rss_kib_before"]
        record["peak_cgroup_swap_kib"] = watcher.peak_cgroup_swap_kib
        record["phase_seconds"] = {k: round(v, 4) for k, v in record["phase_seconds"].items()}
        self.records.append(record)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo):
        outcome = yield
        report: pytest.TestReport = outcome.get_result()
        current = self._current
        if current is None:
            return
        current["phase_seconds"][report.when or "?"] = call.duration
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            current["outcome"] = report.outcome
            if report.outcome == "skipped" and isinstance(report.longrepr, tuple):
                current["skip_reason"] = str(report.longrepr[2])[:400]

    # -- output ------------------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        return {
            "issue": "CHE-64",
            "generated_by": "scripts/pytest_resource_profile.py",
            "session_started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.session_start)
            ),
            "session_seconds": round(time.time() - self.session_start, 3),
            "selection": {
                "markexpr": self.config.getoption("markexpr") or "",
                "keyword": self.config.getoption("keyword") or "",
                "args": list(self.config.invocation_params.args),
            },
            "swap_guard": {
                "enabled": self.guard_enabled,
                "threshold_kib": self.guard_kib,
                "signal": str(CGROUP_SWAP_CURRENT),
                "baseline": self.baseline,
                "why_not_absolute_swap": (
                    "/proc/meminfo inside this container is the HOST's, and the host "
                    "already had ~748 MiB of swap in use at rest with 44.6M cumulative "
                    "pages out. An absolute or >0 check would fire on every run "
                    "regardless of this suite's behaviour."
                ),
                "breach": self.swap_breach,
                "final": _swap_snapshot(),
            },
            "complete": self._session_finished,
            "test_count": len(self.records),
            "tests": self.records,
        }

    def _flush(self) -> None:
        if self.out_path is None:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self._payload(), indent=2) + "\n")

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._session_finished = True
        self._flush()

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self.swap_breach is not None:
            b = self.swap_breach
            terminalreporter.section(
                "CHE-64 SWAP GUARD TRIPPED - RUN TERMINATED", red=True, bold=True
            )
            terminalreporter.write_line(
                f"Container cgroup swap grew {b['growth_kib']} KiB above baseline "
                f"(threshold {b['threshold_kib']} KiB) after "
                f"{b['seconds_into_session']} s."
            )
            terminalreporter.write_line(f"  active test : {b['active_test']}")
            terminalreporter.write_line(
                f"  elapsed in that test : {b['active_test_elapsed_s']} s, "
                f"RSS {b['active_test_rss_kib_at_detection']} KiB"
            )
            terminalreporter.write_line(
                f"  tests completed before breach : {b['completed_tests_before_breach']}"
            )
            terminalreporter.write_line(
                "  THIS RUN IS FAILED. Profiling results are incomplete and must not "
                "be treated as a clean inventory."
            )
            if self.out_path:
                terminalreporter.write_line(f"  evidence : {self.out_path}")
            return
        if self.out_path is not None and self.records:
            slowest = sorted(self.records, key=lambda r: -r["duration_s"])[:5]
            terminalreporter.section("CHE-64 resource profile")
            terminalreporter.write_line(f"wrote {self.out_path}  ({len(self.records)} tests)")
            for r in slowest:
                terminalreporter.write_line(
                    f"  {r['duration_s']:7.3f}s  peak {r['peak_rss_kib'] // 1024:5d} MiB  "
                    f"{r['nodeid']}"
                )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("che64", "per-test resource profiling (CHE-64)")
    group.addoption(
        "--resource-profile",
        action="store",
        default=None,
        metavar="PATH",
        help="write a per-test runtime/memory profile JSON to PATH",
    )
    group.addoption(
        "--swap-guard",
        action="store_true",
        default=False,
        help="terminate the run if the CONTAINER's cgroup swap charge grows past "
        "--swap-guard-kib above its session baseline",
    )
    group.addoption(
        "--swap-guard-kib",
        action="store",
        type=int,
        default=DEFAULT_SWAP_GUARD_KIB,
        help=f"swap-growth tolerance in KiB (default {DEFAULT_SWAP_GUARD_KIB})",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not (config.getoption("resource_profile") or config.getoption("swap_guard")):
        return

    # CHE-140: refuse a sharded run rather than degrade in it.
    #
    # Both halves of this plugin assume one process owns the session. The profile
    # is per-process, so under `-n` it measures a worker and attributes a fraction
    # of the run; worse, escalation is `os.kill(os.getpid(), SIGINT)`, which in a
    # worker kills the *worker*. Measured: the guard tripped, gw0 went down, and
    # the run reported "worker crashed while running ..." with no SWAP GUARD
    # TRIPPED banner and no operator-facing diagnosis -- exactly the silent
    # degradation AGENTS.md forbids for a resource stop-condition.
    #
    # This is reachable by accident now that addopts carries `-n 12`, so it is a
    # usage error rather than a warning. `-n 0` is the fix and is what the
    # substantial runs this plugin exists for should be using anyway.
    if getattr(config, "workerinput", None) is None and (
        config.getoption("numprocesses", None) or getattr(config.option, "dist", "no") != "no"
    ):
        raise pytest.UsageError(
            "--resource-profile/--swap-guard cannot run under pytest-xdist: the "
            "profile would describe one worker, and the swap guard's SIGINT "
            "escalation would kill that worker instead of failing the session. "
            "Re-run with `-n 0`."
        )

    config.pluginmanager.register(ResourceProfiler(config), "che64-resource-profiler")
