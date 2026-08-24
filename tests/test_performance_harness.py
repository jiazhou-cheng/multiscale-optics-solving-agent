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

import json
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
    """The M0.4 finding, pinned so it cannot quietly stop being true.

    At the frozen M3-SINGLET-REF ray count the O(N^2) ray-density diagnostic --
    not the reconstruction -- is the overwhelming majority of `C_RAY_TO_WAVE`'s
    wall time. If a later change makes that false, this test fails and someone
    has to say what changed, which is the point.
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
