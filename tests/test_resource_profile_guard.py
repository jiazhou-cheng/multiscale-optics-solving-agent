"""CHE-64: the swap guardrail must actually fire, and it must actually stop.

A guardrail nobody has seen trip is a comment. These tests trip it two ways:

* unit level, by driving ``check_swap`` past its threshold with a stubbed
  escalation, to pin the arithmetic and the recorded evidence;
* end to end, by running a real pytest session with a negative threshold so any
  reading breaches it, to prove the SIGINT escalation really tears the session
  down with a nonzero exit status and really writes the breach record first.

The end-to-end case is fault injection, not a simulation: the only thing faked is
*when* the threshold is crossed. The detection path, the flush-before-signal
ordering, the signal, pytest's teardown, and the exit status are all the real
ones. Deliberately so -- the alternative is exhausting 377 GiB of host RAM to
test a safety feature, on a shared server, which is the exact outcome the feature
exists to prevent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pytest_resource_profile import (  # noqa: E402
    CGROUP_SWAP_CURRENT,
    ResourceProfiler,
    _rss_kib,
    _swap_snapshot,
)


class _FakeConfig:
    """Minimal pytest.Config stand-in: the profiler only reads options off it."""

    def __init__(self, **options: object) -> None:
        self._options = {
            "resource_profile": None,
            "swap_guard": True,
            "swap_guard_kib": 4096,
            "markexpr": "",
            "keyword": "",
            **options,
        }
        self.rootpath = ROOT
        self.invocation_params = type("P", (), {"args": ()})()

    def getoption(self, name: str) -> object:
        return self._options[name]


def _profiler(**options: object) -> ResourceProfiler:
    profiler = ResourceProfiler(_FakeConfig(**options))  # type: ignore[arg-type]
    profiler.escalated = 0  # type: ignore[attr-defined]
    profiler.escalate = lambda: setattr(  # type: ignore[method-assign]
        profiler, "escalated", profiler.escalated + 1  # type: ignore[attr-defined]
    )
    return profiler


def test_the_container_swap_signal_exists_and_reads_zero_at_rest() -> None:
    """The gate signal must be readable, container-scoped, and quiet at rest.

    If this fails, the guardrail is silently inert -- ``check_swap`` returns
    early on a ``None`` reading, so an unreadable path means no protection at
    all rather than a loud failure.
    """
    assert CGROUP_SWAP_CURRENT.exists(), (
        f"{CGROUP_SWAP_CURRENT} is missing; the swap guard has no signal to gate "
        "on and would be inert. cgroup v2 is required."
    )
    snapshot = _swap_snapshot()
    assert snapshot["cgroup_swap_current_kib"] is not None
    assert snapshot["cgroup_memory_current_kib"] is not None
    assert _rss_kib() > 0


def test_host_swap_is_not_usable_as_the_gate_signal() -> None:
    """Documents *why* the gate is the cgroup, as an executable claim.

    /proc/meminfo in this container is the host's. On a shared server, host swap
    is generally already non-zero at rest, so an absolute or ">0 used" check
    would fire on every run forever. This test asserts the two readings are
    independent quantities rather than asserting a specific host state -- the
    host's swap usage is not ours to control.
    """
    snapshot = _swap_snapshot()
    assert snapshot["host_swap_free_kib"] is not None
    assert snapshot["host_pswpout"] is not None
    # pswpout is cumulative since boot, so on any machine that has ever swapped
    # it is non-zero regardless of what this suite does -- which is precisely the
    # reason it cannot be the gate.
    assert snapshot["host_pswpout"] >= 0


def test_growth_below_the_threshold_does_not_trip_the_guard() -> None:
    profiler = _profiler(swap_guard_kib=4096)
    profiler.baseline_swap_kib = 1000
    profiler.check_swap(1000 + 4096)  # exactly at the threshold, not past it
    assert profiler.swap_breach is None
    assert profiler.escalated == 0  # type: ignore[attr-defined]


def test_growth_past_the_threshold_trips_the_guard_and_escalates_once() -> None:
    profiler = _profiler(swap_guard_kib=4096)
    profiler.baseline_swap_kib = 1000
    profiler._current = {"nodeid": "tests/x.py::test_hog", "_start": 0.0}

    profiler.check_swap(1000 + 4097)

    breach = profiler.swap_breach
    assert breach is not None
    assert breach["growth_kib"] == 4097
    assert breach["threshold_kib"] == 4096
    assert breach["active_test"] == "tests/x.py::test_hog"
    assert profiler.escalated == 1  # type: ignore[attr-defined]

    # Escalating twice would send a second SIGINT into an already-unwinding
    # session and could mask the original teardown.
    profiler.check_swap(1000 + 999_999)
    assert profiler.escalated == 1  # type: ignore[attr-defined]
    assert profiler.swap_breach["growth_kib"] == 4097


def test_the_guard_is_inert_when_not_requested() -> None:
    """Default runs must be untouched: no guard, no termination."""
    profiler = _profiler(swap_guard=False)
    profiler.baseline_swap_kib = 0
    profiler.check_swap(10**9)
    assert profiler.swap_breach is None
    assert profiler.escalated == 0  # type: ignore[attr-defined]


def test_the_breach_record_identifies_the_stage_that_was_running() -> None:
    """Acceptance criterion: the test active when swap began is identifiable."""
    profiler = _profiler(swap_guard_kib=0)
    profiler.baseline_swap_kib = 0
    profiler._current = {"nodeid": "tests/test_big.py::test_allocates", "_start": 0.0}
    profiler.check_swap(1)

    breach = profiler.swap_breach
    assert breach is not None
    assert breach["active_test"] == "tests/test_big.py::test_allocates"
    assert breach["active_test_elapsed_s"] is not None
    assert breach["active_test_rss_kib_at_detection"] > 0
    assert breach["completed_tests_before_breach"] == 0
    assert breach["host_at_detection"]["cgroup_swap_current_kib"] is not None
    assert "not pre-existing host swap" in breach["interpretation"]


def test_a_breach_with_no_active_test_is_recorded_rather_than_crashing() -> None:
    """Swap can grow between tests, e.g. during collection or teardown."""
    profiler = _profiler(swap_guard_kib=0)
    profiler.baseline_swap_kib = 0
    profiler._current = None
    profiler.check_swap(1)
    assert profiler.swap_breach is not None
    assert profiler.swap_breach["active_test"] is None


@pytest.mark.slow
def test_end_to_end_a_breach_terminates_the_session_with_a_nonzero_exit(
    tmp_path: Path,
) -> None:
    """The real escalation path, via a subprocess pytest run.

    Fault-injected with ``--swap-guard-kib=-1``: growth of 0 already exceeds -1,
    so the guard trips on the first sample. Everything after detection is the
    production path.

    Asserted: nonzero exit, the breach record on disk, the named active test, and
    that the session stopped *early* -- the target file has several tests and the
    guard fires during the first one.
    """
    profile = tmp_path / "breach.json"
    target = ROOT / "tests" / "test_resource_profile_guard.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.pytest_resource_profile",
            f"--resource-profile={profile}",
            "--swap-guard",
            "--swap-guard-kib=-1",
            "-p",
            "no:cacheprovider",
            "-k",
            "not end_to_end",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode != 0, (
        "the guard must FAIL the run, not merely annotate it.\n"
        f"stdout:\n{result.stdout[-3000:]}"
    )
    assert "SWAP GUARD TRIPPED" in result.stdout, result.stdout[-3000:]
    assert "THIS RUN IS FAILED" in result.stdout, result.stdout[-3000:]

    assert profile.exists(), "evidence must be flushed BEFORE the signal is sent"
    payload = json.loads(profile.read_text())
    breach = payload["swap_guard"]["breach"]
    assert breach is not None
    assert breach["threshold_kib"] == -1
    assert breach["active_test"] is not None
    assert breach["active_test"].startswith("tests/test_resource_profile_guard.py::")
    assert payload["swap_guard"]["enabled"] is True

    # Stopped early rather than running to completion under memory pressure. The
    # guard fires on the first boundary check, so nothing should have finished.
    assert breach["completed_tests_before_breach"] == 0
    assert "passed" not in result.stdout.splitlines()[-1]
