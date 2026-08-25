"""The performance harness, and the three things it must refuse to do.

CHE-105 (M0.4). A timing harness earns trust by what it declines to report, not
by what it reports. Three refusals, each tested below:

1. **Comparing across environments.** A ratio between two hosts is a number, and
   numbers get quoted without their caveats.
2. **Comparing across cost models.** `seconds_per_ray` on `ramp_sum`
   (O(rays x pixels)) and on `kspace_splat` (O(rays) + one FFT) share a name and
   are different quantities. The demo3 record already carries this warning in
   prose; here it is a refusal.
3. **Timing a run that is swapping.** `AGENTS.md` makes swap growth a stop
   condition. A swapping run is not a slow run, it is a measurement of the disk.

Plus the thing that made this milestone necessary: a fitted exponent, rather
than two endpoints and an assertion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from core.performance import (
    Incomparable,
    PerformanceRecord,
    StageTimer,
    SwapGrowthAbort,
    Workload,
    compare,
    environment_fingerprint,
    fit_scaling,
    measure,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks" / "schemas" / "performance.schema.json"
RUN_BASELINES = ROOT / "benchmarks" / "perf" / "run_baselines.py"


def load_run_baselines():
    """Import the baseline runner from its path.

    `benchmarks/perf/` is not a package and is not on `sys.path`, so a plain
    import would not find it. Loaded here because the M0.4 review found that no
    test imported this script at all -- which is why its nonzero-exit refusal and
    its private-global restore, the two behaviours its own comments call out as
    load-bearing, were unverified.
    """
    spec = importlib.util.spec_from_file_location("perf_run_baselines", RUN_BASELINES)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trivial(timer: StageTimer) -> int:
    with timer.stage("a"):
        time.sleep(0.005)
    with timer.stage("b"):
        time.sleep(0.005)
    return 7


def _record(label: str = "unit", **workload: object) -> PerformanceRecord:
    kwargs: dict = {"size": 1000.0, "unit": "ray", "route": "ramp_sum"}
    kwargs.update(workload)
    record, _ = measure(
        _trivial, label=label, workload=Workload(**kwargs), repeats=2, warmup=1
    )
    return record


# ---------------------------------------------------------------------------
# It measures the things the ticket lists
# ---------------------------------------------------------------------------


def test_a_measurement_carries_every_required_quantity() -> None:
    record = _record()

    assert record.measurement.repeats == 2
    assert record.measurement.warmup == 1
    assert record.measurement.min_s <= record.measurement.median_s <= record.measurement.p95_s
    assert record.peak_host_rss_bytes and record.peak_host_rss_bytes > 0
    assert record.cuda is not None
    assert record.swap_growth_bytes == 0
    assert record.cost_per_unit == pytest.approx(record.measurement.median_s / 1000.0)


def test_the_returned_value_is_the_workloads_own_result() -> None:
    """The harness times a callable; it must not eat what the callable produced."""
    _, result = measure(
        _trivial, label="x", workload=Workload(size=1.0, unit="call"), repeats=1, warmup=0
    )
    assert result == 7


def test_stage_timings_accumulate_and_report_their_share() -> None:
    """The generalization of demo3's hand-rolled `stage_s`.

    The share is the number CHE-101 lacked: a 9.6x speedup on a stage worth 7%
    of the run moves the run by 7%, and a breakdown in seconds alone does not
    make that obvious at a glance.
    """
    record = _record()
    stages = record.stages

    assert stages is not None
    assert set(stages["seconds"]) == {"a", "b"}
    assert stages["accounted_s"] > 0
    assert sum(stages["fraction_of_total"].values()) <= 1.05
    assert stages["unaccounted_s"] >= -1e-6


def test_a_stage_entered_repeatedly_accumulates_rather_than_overwrites() -> None:
    """These workloads run in chunks, so a stage is entered once per chunk."""
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("chunked"):
            time.sleep(0.002)
    assert timer.stages["chunked"] >= 0.005


def test_the_record_validates_against_its_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_record().as_dict())


# ---------------------------------------------------------------------------
# Refusal 1 and 2: what may not be compared
# ---------------------------------------------------------------------------


def test_two_records_from_the_same_environment_compare() -> None:
    a, b = _record("baseline"), _record("candidate")
    verdict = compare(a, b)

    assert verdict["speedup"] > 0
    assert verdict["cost_per_unit"]["unit"] == "ray"
    assert verdict["cost_per_unit"]["route"] == "ramp_sum"


def test_comparing_across_environments_is_refused_and_says_which_field_moved() -> None:
    baseline = _record("baseline").as_dict()
    candidate = _record("candidate").as_dict()
    # Stand in for the same workload measured on a different card.
    candidate["environment"] = {**candidate["environment"], "gpu_name": "NVIDIA H100"}
    candidate["environment"]["sha256"] = "f" * 64

    with pytest.raises(Incomparable) as excinfo:
        compare(baseline, candidate)

    assert "gpu_name" in str(excinfo.value)
    assert "different environments" in str(excinfo.value)


def test_comparing_across_reconstruction_routes_is_refused() -> None:
    """The repository-specific one, and the reason `route` exists at all.

    `ramp_sum` is O(rays x pixels); `kspace_splat` is O(rays) plus one FFT.
    Dividing one `seconds_per_ray` by the other produces something that looks
    like a speedup and is a change of cost model.
    """
    ramp = _record("ramp", route="ramp_sum")
    kspace = _record("kspace", route="kspace_splat")

    with pytest.raises(Incomparable, match="not the same kind of work"):
        compare(ramp, kspace)


def test_comparing_across_units_is_refused() -> None:
    with pytest.raises(Incomparable, match="not the same kind of work"):
        compare(_record("rays", unit="ray"), _record("pixels", unit="pixel"))


# ---------------------------------------------------------------------------
# Refusal 3: a swapping run is not timed
# ---------------------------------------------------------------------------


def test_swap_growth_aborts_the_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demonstrated, not asserted -- the acceptance criterion's own wording.

    Real swap cannot be induced in a test without putting the shared server
    under exactly the memory pressure `AGENTS.md` forbids, so the watchdog's
    reading is forced instead. What is under test is the harness's RESPONSE to
    swap growth: raise, do not record a timing. That is the whole behaviour;
    detecting the growth is `core.resources`' job and has its own tests.
    """
    import core.performance as perf

    class SwappingWatchdog:
        peak_rss_bytes = 1 << 20
        swap_growth_bytes = 4096

        def start(self):
            return self

        def stop(self):
            return self

        def report(self):
            return {"forced": "a watchdog that reports swap growth"}

    monkeypatch.setattr(perf, "MemoryWatchdog", lambda **_: SwappingWatchdog())

    with pytest.raises(SwapGrowthAbort) as excinfo:
        measure(
            _trivial,
            label="swapping",
            workload=Workload(size=1.0, unit="call"),
            repeats=1,
            warmup=0,
        )

    assert excinfo.value.growth_bytes == 4096
    assert "stop" in str(excinfo.value).lower() or "stopped" in str(excinfo.value).lower()
    assert "measuring the disk" in str(excinfo.value)
    assert excinfo.value.report == {"forced": "a watchdog that reports swap growth"}


def test_the_abort_happens_during_warmup_too() -> None:
    """A run that swaps in warmup must not proceed to the timed repeats."""
    import core.performance as perf

    calls = {"n": 0}

    class SwappingWatchdog:
        peak_rss_bytes = 1
        swap_growth_bytes = 1

        def start(self):
            return self

        def stop(self):
            return self

        def report(self):
            return {}

    def counted(timer: StageTimer) -> None:
        calls["n"] += 1

    original = perf.MemoryWatchdog
    perf.MemoryWatchdog = lambda **_: SwappingWatchdog()  # type: ignore[assignment]
    try:
        with pytest.raises(SwapGrowthAbort):
            measure(
                counted,
                label="swap-in-warmup",
                workload=Workload(size=1.0, unit="call"),
                repeats=5,
                warmup=2,
            )
    finally:
        perf.MemoryWatchdog = original  # type: ignore[assignment]

    assert calls["n"] == 1, "the abort must fire on the first sample, not after the sweep"


# ---------------------------------------------------------------------------
# A fitted exponent, not two endpoints
# ---------------------------------------------------------------------------


def test_a_known_power_law_is_recovered() -> None:
    fit = fit_scaling([(1.0, 2.0), (2.0, 8.0), (4.0, 32.0), (8.0, 128.0)], axis="n")

    assert fit.exponent == pytest.approx(2.0, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.axis == "n"


def test_a_noisy_power_law_reports_its_r_squared() -> None:
    fit = fit_scaling([(1.0, 1.0), (2.0, 3.6), (4.0, 17.0), (8.0, 61.0)], axis="rays")

    assert 1.8 < fit.exponent < 2.2
    assert 0.9 < fit.r_squared < 1.0


def test_two_points_are_refused_because_they_always_fit() -> None:
    """The whole reason the protocol asks for a fit rather than a ratio."""
    with pytest.raises(ValueError, match="at least 3"):
        fit_scaling([(1.0, 1.0), (2.0, 4.0)], axis="n")


# ---------------------------------------------------------------------------
# The environment fingerprint is distinct from the scientific one
# ---------------------------------------------------------------------------


def test_the_performance_fingerprint_covers_what_the_scientific_one_omits() -> None:
    """They answer different questions and must not be interchangeable.

    `core.provenance.environment_fingerprint` exists to say whether a RESULT is
    reproducible, and deliberately ignores everything that changes only speed.
    This one exists to say whether a TIMING is comparable, so it must carry the
    host.
    """
    from core.provenance import environment_fingerprint as scientific

    performance = environment_fingerprint().as_dict()
    correctness = scientific()

    for host_field in ("gpu_name", "gpu_driver", "cpu_model", "logical_cpus", "container_image"):
        assert host_field in performance, f"{host_field} decides comparability of a timing"
        assert host_field not in correctness, (
            f"{host_field} must NOT be in the scientific fingerprint: a result that "
            "changes with the GPU model is a bug, not a different result"
        )

    assert performance["sha256"] != correctness["combined_sha256"]


def test_a_thread_count_change_changes_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thread counts change speed by large factors and change no answer."""
    before = environment_fingerprint().sha256
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert environment_fingerprint().sha256 != before


def test_isolation_is_not_claimed_unless_it_was_applied() -> None:
    """Recording an affinity mask does not make a run isolated."""
    record = _record()

    assert record.isolation.applied is False
    assert record.isolation.cpu_affinity  # observed, and reported as observation only


# ---------------------------------------------------------------------------
# The committed baselines
# ---------------------------------------------------------------------------

PERF_RECORDS = ROOT / "benchmarks" / "perf" / "records"


def _committed_records() -> list[Path]:
    return sorted(PERF_RECORDS.glob("*.json"))


def test_there_are_committed_baselines_at_all() -> None:
    assert _committed_records(), (
        "benchmarks/perf/records/ is empty. The harness without baselines is a "
        "measuring instrument nobody has measured anything with."
    )


@pytest.mark.parametrize("path", _committed_records(), ids=lambda p: p.stem)
def test_a_committed_baseline_validates_against_the_schema(path: Path) -> None:
    """Every record, whether it is one measurement or a document containing several.

    The composite records (`framework_overhead`, `scaling_ray_axis`,
    `estimate_accuracy`) carry `PerformanceRecord`s nested under their own keys
    rather than being one, so the validator is pointed at every nested record it
    can find instead of at the top level only.
    """
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    payload = json.loads(path.read_text())

    found = 0
    for record in _nested_records(payload):
        validator.validate(record)
        found += 1
    assert found >= 1, f"{path.name} contains no performance record to validate"


def _nested_records(node: object) -> list[dict]:
    """Every object that looks like a PerformanceRecord, at any depth."""
    out: list[dict] = []
    if isinstance(node, dict):
        if {"label", "workload", "measurement", "environment"} <= set(node):
            out.append(node)
        else:
            for value in node.values():
                out.extend(_nested_records(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_nested_records(value))
    return out


@pytest.mark.parametrize("path", _committed_records(), ids=lambda p: p.stem)
def test_no_committed_baseline_was_measured_while_swapping(path: Path) -> None:
    """A swapping run is a resource failure, so it must never reach a record.

    `measure` raises on swap growth, so this can only fail if a record was
    hand-written or the guard was bypassed. Cheap to check and it pins the
    invariant at the artifact rather than only at the code.
    """
    for record in _nested_records(json.loads(path.read_text())):
        assert record["swap_growth_bytes"] in (0, None), (
            f"{path.name} / {record['label']!r} records swap growth of "
            f"{record['swap_growth_bytes']} bytes. That run should have aborted."
        )


@pytest.mark.parametrize("path", _committed_records(), ids=lambda p: p.stem)
def test_a_gpu_baseline_was_measured_on_exactly_one_device(path: Path) -> None:
    """AGENTS.md: GPU measurements are single-workload, single-GPU, sequential.

    Only the single-GPU half is checkable from a record, and this is it. A
    record that saw two devices was measured under a configuration the policy
    forbids, and its timing is not the timing of a single-GPU run.
    """
    for record in _nested_records(json.loads(path.read_text())):
        count = record["environment"]["gpu_count"]
        if count:
            assert count == 1, (
                f"{path.name} / {record['label']!r} saw {count} CUDA devices. "
                "GPU measurements must be single-GPU."
            )


def test_the_scaling_baseline_separates_the_two_cost_models() -> None:
    """The M0.4 finding, as the committed record states it.

    This reads the frozen artifact, so it guards the RECORD and not the code: it
    fails if someone edits `scaling_ray_axis.json` into disagreeing with the
    finding it is quoted for, and it does *not* notice a change to
    `_ray_density_diagnostic` or to the scan limit. The live-code half of the
    claim is `test_the_density_diagnostic_still_dominates_the_frozen_call`
    below, which re-measures.
    """
    path = PERF_RECORDS / "scaling_ray_axis.json"
    if not path.exists():
        pytest.skip("scaling baseline not recorded")
    payload = json.loads(path.read_text())

    reconstruction = payload["fits"]["reconstruction_only"]
    diagnostic = payload["fits"]["ray_density_diagnostic"]

    assert 0.9 < reconstruction["exponent"] < 1.3, (
        "the reconstruction should be linear in rays at fixed grid"
    )
    assert reconstruction["r_squared"] > 0.95
    assert 1.8 < diagnostic["exponent"] < 2.2, "a pairwise scan is quadratic"
    assert diagnostic["r_squared"] > 0.95

    frozen = next(r for r in payload["rows"] if r["rays"] == 3169)
    assert frozen["diagnostic_share_of_call"] > 0.8, (
        "at the frozen configuration the diagnostic dominated the call when this "
        "was measured; if it no longer does, re-derive the M5.2 target that was "
        "handed off on the strength of it"
    )

    # And the reason two fits exist rather than one.
    across = payload["fits"]["as_shipped_do_not_use"]
    assert across["r_squared"] < 0.5, (
        "a single exponent fitted across the scan-limit threshold should be a bad "
        "fit -- that is the evidence for splitting it"
    )


# ---------------------------------------------------------------------------
# CHE-129: the paths the M0.4 review found untested, and the fixes it asked for
# ---------------------------------------------------------------------------


def test_the_density_diagnostic_still_dominates_the_frozen_call() -> None:
    """The live-code half of the claim the committed record is quoted for.

    `test_the_scaling_baseline_separates_the_two_cost_models` reads a frozen
    artifact and therefore cannot notice a change to `_ray_density_diagnostic` or
    to the scan limit. This one re-measures, small and in-process, so a change
    that makes the M5.2 target wrong fails a test rather than being discovered by
    whoever acts on it.
    """
    import numpy as np

    import couplers.ray_to_wave as rtw

    rng = np.random.default_rng(20260824)
    count = 2000  # comfortably under the 4096 scan limit
    assert count <= rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT

    from core.boundary import Frame, RayBundle, ReferencePlane
    from core.coherent_batch import declared_launch_opl_reference

    plane = ReferencePlane(name="launch", z_m=0.0)
    transverse = rng.uniform(-0.05, 0.05, (count, 2))
    bundle = RayBundle(
        positions_m=np.column_stack(
            [
                rng.uniform(-1e-4, 1e-4, count),
                rng.uniform(-1e-4, 1e-4, count),
                np.zeros(count),
            ]
        ),
        directions=np.column_stack(
            [transverse, np.sqrt(1.0 - (transverse**2).sum(axis=1))]
        ),
        wavelength_m=5.5e-7,
        reference_plane=plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.ones(count, dtype=np.complex128),
        optical_path_length_m=np.zeros(count),
        optical_path_length_reference=declared_launch_opl_reference(plane),
        reconstruction_normalization="one_over_n",
    )
    grid, pitch = (96, 96), 2.0e-6

    def call() -> None:
        rtw.ray_to_wave(bundle, grid_shape=grid, sample_pitch_m=(pitch, pitch))

    def time_once() -> float:
        started = time.perf_counter()
        call()
        return time.perf_counter() - started

    call()  # warm the import/JIT path so neither arm pays for it
    shipped = min(time_once() for _ in range(3))

    previous = rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
    rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = 0
    try:
        reconstruction = min(time_once() for _ in range(3))
    finally:
        rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = previous

    share = (shipped - reconstruction) / shipped
    assert share > 0.5, (
        f"below the scan limit the O(N^2) ray-density diagnostic was {share:.1%} of "
        f"the call ({shipped:.4f} s shipped, {reconstruction:.4f} s with it forced "
        "off). M0.4 measured ~91% at 3169 rays and handed M5.2 a target on the "
        "strength of it; if the diagnostic is no longer the cost, that target is "
        "stale and someone has to say so."
    )


def test_the_scan_limit_is_restored_even_when_the_measured_arm_raises() -> None:
    """The private-global mutation must not leak into the rest of the process.

    `run_baselines.scan_limit_forced_off` reaches into
    `couplers.ray_to_wave._NEAREST_NEIGHBOUR_SCAN_LIMIT`. A leak would silently
    disable the diagnostic for every later call in the same process -- including
    other baselines in the same invocation -- so the restore is asserted rather
    than assumed from reading the `finally`.
    """
    import couplers.ray_to_wave as rtw

    scan_limit_forced_off = load_run_baselines().scan_limit_forced_off

    original = rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
    assert original > 0

    def live_limit() -> int:
        return int(rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT)

    with scan_limit_forced_off():
        assert live_limit() == 0
    assert live_limit() == original

    # Restored on the exception this harness actually raises...
    with pytest.raises(SwapGrowthAbort), scan_limit_forced_off():
        raise SwapGrowthAbort(1, {})
    assert live_limit() == original

    # ...and on the one a person at a terminal raises.
    with pytest.raises(KeyboardInterrupt), scan_limit_forced_off():
        raise KeyboardInterrupt
    assert live_limit() == original


def test_a_nonzero_exit_is_refused_and_no_record_is_written(tmp_path: Path) -> None:
    """Timing a crash and committing it as a baseline is the failure this guards.

    It already happened once: L2-PSF-01 exited 1 in 0.5 s and was written out as a
    0.5-second bundle baseline. The number was real and the label was a lie.
    """
    run_baselines = load_run_baselines()

    before = set(run_baselines.RECORDS.glob("*.json"))
    with pytest.raises(RuntimeError, match="exited 3"):
        run_baselines._timed_command(
            [sys.executable, "-c", "import sys; print('partial output'); sys.exit(3)"],
            label="a command that fails",
            workload=Workload(size=1.0, unit="run", route="failing"),
            notes=[],
        )
    assert set(run_baselines.RECORDS.glob("*.json")) == before, (
        "a refused command must not leave a record behind"
    )


def test_a_zero_exit_command_is_recorded_with_its_argv() -> None:
    """The other half: the guard must not refuse a command that worked."""
    run_baselines = load_run_baselines()

    payload = run_baselines._timed_command(
        [sys.executable, "-c", "print('done')"],
        label="a command that works",
        workload=Workload(size=1.0, unit="run", route="working"),
        notes=[],
    )
    assert payload["subprocess"]["returncode"] == 0
    assert payload["subprocess"]["tail"] == ["done"]
    # The child's memory, not the parent's. `peak_host_rss_bytes` on a
    # whole-command record is this process's RSS and describes nothing.
    assert "peak_child_rss_bytes" in payload["subprocess"]
    # Only the tails, not the whole stream: a chunked demo prints thousands of
    # lines and a provenance record is not a log file.
    assert len(payload["subprocess"]["tail"]) <= 8
    assert len(payload["subprocess"]["stderr_tail"]) <= 12
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(payload)


def test_an_unreadable_swap_file_is_reported_as_null_and_not_as_zero() -> None:
    """`None` means "could not read", and a zero would make the guard unfalsifiable.

    On a host where the cgroup swap file is absent, coercing `None` to `0` makes
    every record assert "this run did not swap" and makes
    `test_no_committed_baseline_was_measured_while_swapping` pass vacuously.
    """

    class _Blind:
        """A watchdog that cannot read the cgroup, which is what `None` means."""

        peak_rss_bytes = 1
        verdict = type("V", (), {"breached": False, "reason": None, "detail": ""})()
        swap_growth_bytes = None

        def start(self) -> _Blind:
            return self

        def stop(self) -> None:
            pass

        def report(self) -> dict[str, object]:
            return {"samples": 0}

    import core.performance as perf

    original = perf.MemoryWatchdog
    perf.MemoryWatchdog = lambda **_: _Blind()  # type: ignore[assignment,misc]
    try:
        record, _ = measure(
            lambda timer: None,
            label="blind to the cgroup",
            workload=Workload(size=1.0, unit="call", route=None),
            repeats=1,
            warmup=0,
        )
    finally:
        perf.MemoryWatchdog = original  # type: ignore[assignment]

    assert record.swap_growth_bytes is None, (
        "an unreadable cgroup file must stay null; a zero here is a fabricated "
        "claim that the run did not swap"
    )
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(record.as_dict())


def test_a_stage_share_above_one_is_refused_rather_than_published() -> None:
    """A `fraction_of_total` over 1.0 means two different calls were divided.

    The committed `framework_overhead.json` had one at 101.57% -- one repeat's
    stage seconds over a median across repeats -- and a slow final repeat could
    push it to several hundred percent under a name a reader takes for a share.
    """
    from core.performance import StageAccountingError

    timer = StageTimer()
    with timer.stage("work"):
        time.sleep(0.01)

    # Paired with its own call: fine, and the shares are bounded.
    payload = timer.as_dict(timer.stages["work"] + 0.005)
    assert 0 < payload["fraction_of_total"]["work"] <= 1.0
    assert payload["total_s"] == pytest.approx(timer.stages["work"] + 0.005, abs=1e-4)

    # Divided by somebody else's faster call: refused.
    with pytest.raises(StageAccountingError, match="different calls"):
        timer.as_dict(timer.stages["work"] * 0.5)


def test_measured_stage_shares_never_exceed_the_call_that_contained_them() -> None:
    """End to end: a slow final repeat must not produce a 498% share."""
    calls = {"n": 0}

    def fn(timer: StageTimer) -> None:
        calls["n"] += 1
        # The last timed repeat is much slower than the others, which is exactly
        # the case that broke when the denominator was the median.
        duration = 0.05 if calls["n"] >= 3 else 0.005
        with timer.stage("work"):
            time.sleep(duration)

    record, _ = measure(
        fn,
        label="uneven repeats",
        workload=Workload(size=1.0, unit="call", route=None),
        repeats=3,
        warmup=0,
    )
    assert record.stages is not None
    assert record.stages["fraction_of_total"]["work"] <= 1.0
    assert record.stages["unaccounted_s"] >= -1e-4
    assert record.stages["total_s"] >= record.stages["accounted_s"] - 1e-4
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(record.as_dict())


def test_compare_refuses_two_records_that_did_different_work() -> None:
    """The 11x "speedup" `route` alone did not catch.

    The scaling baseline records the shipping call and the same call with the
    O(N^2) diagnostic forced off. With the marker only in `workload.detail`, unit
    and route matched and `compare` divided them.
    """
    shipped, _ = measure(
        lambda timer: None,
        label="as shipped",
        workload=Workload(
            size=100.0, unit="ray", route="ramp_sum", detail={"diagnostic": "as shipped"}
        ),
        repeats=1,
        warmup=0,
    )
    forced_off, _ = measure(
        lambda timer: None,
        label="diagnostic forced off",
        workload=Workload(
            size=100.0, unit="ray", route="ramp_sum", detail={"diagnostic": "forced off"}
        ),
        repeats=1,
        warmup=0,
    )
    with pytest.raises(Incomparable, match="detail differs"):
        compare(shipped, forced_off)

    # And it still compares two records that really are the same work.
    same, _ = measure(
        lambda timer: None,
        label="as shipped again",
        workload=Workload(
            size=100.0, unit="ray", route="ramp_sum", detail={"diagnostic": "as shipped"}
        ),
        repeats=1,
        warmup=0,
    )
    assert compare(shipped, same)["speedup"] is not None


def test_no_committed_record_divides_two_different_computations() -> None:
    """The committed artifacts must not be divisible into a wrong answer.

    `scaling_ray_axis.json` carries a shipping arm and a diagnostic-disabled arm
    per ray count. Wherever the diagnostic actually ran, the two are different
    computations and `compare` must refuse -- unmarked, it returned 11x.

    Only that record: `estimate_accuracy.json` serializes the diagnostic-disabled
    arm alone, so there is no pair in it to divide. Its `route`/`detail` marking is
    for whoever adds the shipping arm later.
    """
    scaling = PERF_RECORDS / "scaling_ray_axis.json"
    if not scaling.exists():
        pytest.skip("scaling baseline not recorded")
    payload = json.loads(scaling.read_text())
    checked = 0
    for row in payload["rows"]:
        if not row["diagnostic_runs"]:
            continue
        checked += 1
        with pytest.raises(Incomparable):
            compare(row["full_record"], row["reconstruction_record"])
    assert checked, "no row in the scaling baseline ran the diagnostic"


def test_the_tail_rule_travels_with_the_number() -> None:
    """`p95_s` at these sample sizes is a maximum, and must say so in the record."""
    from core.performance import Measurement

    for n in (1, 3, 5, 19):
        value, rule = Measurement.tail([float(i) for i in range(n)])
        assert value == float(n - 1), f"n={n} should take the slowest run"
        assert rule == f"max_of_{n}"
    # 20 is where `ceil(0.95n) - 1` stops being the maximum, which the schema
    # previously described as the boundary rather than one past it.
    value, rule = Measurement.tail([float(i) for i in range(20)])
    assert value == 18.0
    assert rule == "order_statistic_19_of_20"

    record, _ = measure(
        lambda timer: None,
        label="tail rule",
        workload=Workload(size=1.0, unit="call", route=None),
        repeats=3,
        warmup=0,
    )
    assert record.measurement.tail_rule == "max_of_3"
    assert record.measurement.p95_s == max(record.measurement.all_s)


def test_a_flat_cost_curve_reports_no_r_squared_rather_than_a_perfect_one() -> None:
    """The one degenerate case where r^2 = 1 would mean nothing at all."""
    fit = fit_scaling([(1.0, 2.0), (10.0, 2.0), (100.0, 2.0)], axis="rays")
    assert fit.exponent == pytest.approx(0.0)
    assert fit.r_squared is None


def test_measure_rejects_a_negative_warmup() -> None:
    """Otherwise the loop runs fewer samples than `repeats` claims."""
    with pytest.raises(ValueError, match="warmup"):
        measure(
            lambda timer: None,
            label="bad warmup",
            workload=Workload(size=1.0, unit="call", route=None),
            repeats=3,
            warmup=-1,
        )


def test_a_subprocess_measurement_does_not_initialize_a_device_here() -> None:
    """The parent must not preallocate the card the child needs.

    Measured, not theorized: demo2's RW-P route at the Table S2 budget (1.6e8
    rays) completes in 94 s when run standalone and died with
    `RESOURCE_EXHAUSTED` after 16 s under a parent that had synchronized JAX
    first, because JAX preallocates ~78% of the device on initialization. A
    process boundary is a stronger barrier than a device sync -- the child cannot
    exit with work outstanding -- so the sync buys nothing and costs the run.
    """
    import core.performance as perf

    synced: list[str] = []

    def spy(result: object = None) -> None:
        synced.append("touched")

    original = perf._synchronize
    perf._synchronize = spy  # type: ignore[assignment]
    try:
        child, _ = measure(
            lambda timer: None,
            label="a child process",
            workload=Workload(size=1.0, unit="run", route=None),
            repeats=1,
            warmup=0,
            touch_devices=False,
        )
        assert synced == [], "a subprocess measurement must not touch a device here"
        assert child.cuda is None, (
            "the parent's allocator snapshot is not the child's peak, so it must be "
            "null rather than reported as the workload's GPU memory"
        )

        # And the in-process default is unchanged: it still syncs, both sides.
        in_process, _ = measure(
            lambda timer: None,
            label="in this process",
            workload=Workload(size=1.0, unit="call", route=None),
            repeats=1,
            warmup=0,
        )
        assert len(synced) == 2
        assert in_process.cuda is not None
    finally:
        perf._synchronize = original  # type: ignore[assignment]

    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(child.as_dict())


def test_the_whole_command_baselines_all_declare_the_process_boundary() -> None:
    """Every subprocess record must carry null `cuda` and say why.

    A committed record with a `cuda` block would mean somebody reinstated the
    parent-side sync, which is the demo2 OOM waiting to happen again.
    """
    for path in sorted(PERF_RECORDS.glob("*.json")):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or "subprocess" not in record:
            continue
        assert record["cuda"] is None, (
            f"{path.name} measured a subprocess but reports a `cuda` block, which "
            "is this process's allocator and not the child's"
        )
        assert any("process boundary" in note for note in record["notes"]), (
            f"{path.name} does not say why it did not synchronize a device"
        )


def test_overlapping_stages_are_refused_rather_than_double_counted() -> None:
    """A nesting bug must not surface as "two different calls".

    `StageTimer`'s stages are a partition of one call's wall clock, not a call
    tree. Nesting them would add the inner interval to the outer one, push the
    shares above 1, and then be reported by `as_dict` as a numerator/denominator
    mismatch -- the wrong diagnosis for the actual mistake.
    """
    from core.performance import StageAccountingError

    # `__enter__` directly rather than a nested `with`: the guard fires on entry,
    # and this keeps the failing call on one line instead of inside a block that
    # never executes.
    timer = StageTimer()
    outer = timer.stage("a")
    outer.__enter__()
    with pytest.raises(StageAccountingError, match="already open"):
        timer.stage("a").__enter__()
    outer.__exit__(None, None, None)

    fresh = StageTimer()
    open_stage = fresh.stage("outer")
    open_stage.__enter__()
    with pytest.raises(StageAccountingError, match="while"):
        fresh.stage("inner").__enter__()
    open_stage.__exit__(None, None, None)

    # A stage re-entered SEQUENTIALLY still accumulates -- that is the chunk-loop
    # case the timer exists for, and it must keep working.
    chunked = StageTimer()
    for _ in range(3):
        with chunked.stage("per_chunk"):
            time.sleep(0.002)
    assert chunked.stages["per_chunk"] >= 0.005


def test_host_memory_pressure_is_reported_and_does_not_fail_the_measurement() -> None:
    """A shared box's other tenants must not fail this repository's gate.

    `mem_available` is host-wide, is evaluated from the watchdog's baseline
    snapshot so it can already be breached before the measured workload allocates
    anything, and never resets. Raising on it would let another tenant's memory
    use fail the ~20 `measure()`-based unit tests in the default gate while they
    time trivial closures. The guards that ARE attributable to this process still
    raise -- `test_swap_growth_aborts_the_measurement` above covers that.
    """
    import core.performance as perf
    from core.resources import MemoryWatchdogVerdict

    class HostUnderPressure:
        peak_rss_bytes = 1 << 20
        swap_growth_bytes = 0
        verdict = MemoryWatchdogVerdict(
            breached=True,
            reason="mem_available",
            detail="host MemAvailable fell to 1 B, at or below the 2 B reserve",
        )

        def start(self) -> HostUnderPressure:
            return self

        def stop(self) -> None:
            pass

        def report(self) -> dict[str, object]:
            return {"forced": "a host under pressure"}

    original = perf.MemoryWatchdog
    perf.MemoryWatchdog = lambda **_: HostUnderPressure()  # type: ignore[assignment,misc]
    try:
        record, _ = measure(
            _trivial,
            label="measured on a busy host",
            workload=Workload(size=1.0, unit="call", route=None),
            repeats=1,
            warmup=0,
        )
    finally:
        perf.MemoryWatchdog = original  # type: ignore[assignment]

    assert any("HOST MEMORY PRESSURE" in note for note in record.notes), (
        "a host-wide breach must be recorded on the measurement it happened "
        "during, so nobody reads the timing as a clean baseline"
    )
    assert any("upper bound" in note for note in record.notes)
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(record.as_dict())
