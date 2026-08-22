"""The sweep controller's guards, bookkeeping and failure handling (CHE-70).

Every test here is about the controller *refusing* or *recording* something. The
physics is tested elsewhere; what matters here is that:

* an unsafe request is skipped **before** it allocates anything (Phase 16/18);
* a child that dies, hangs or writes nothing is recorded as a candidate with a
  status rather than taking the sweep down (Phase 15);
* growth stops in a direction that has already failed for memory (Phase 33);
* "smallest converged" means smallest, not "largest that ran" (Phase 27);
* the controller keeps only scalars, so its own RSS is flat in the number of
  candidates (Phase 34).

The child process is stubbed for most of these. That is deliberate: the point is
the controller's decision logic, and running real CUDA candidates to test a
guard's arithmetic would be both slow and less precise about what is being
checked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.metalens_candidate import (
    STATUSES,
    CandidateRequest,
)
from benchmarks.metalens_controller import (
    CHUNK_SAFETY_FACTOR,
    NCC_GATE,
    POWER_ERROR_TARGET,
    STABILITY_TOLERANCE,
    SweepController,
    SweepOptions,
    main,
)

pytestmark = [pytest.mark.coupler]


def _controller(tmp_path: Path, **overrides) -> SweepController:
    options = SweepOptions(output=tmp_path / "sweep", **overrides)
    return SweepController(options)


def _stub_record(
    controller: SweepController,
    *,
    launches: int,
    samples: int,
    ncc: float,
    status: str = "PASS_RUN",
    seed: int | None = None,
    label: str = "stub",
    arrays: Path | None = None,
    swap_delta: int | None = 0,
    gpu_peak: int | None = 1024,
    power_error: float = 0.001,
    chunk_size: int | None = None,
    chunk_count: int = 1,
) -> None:
    """Insert a completed candidate without running one."""
    request = CandidateRequest(
        config=controller.options.config,
        launch_count=launches,
        samples_per_launch=samples,
        seed=seed if seed is not None else controller.options.seed,
        density=controller.options.density,
        chunk_size=chunk_size if chunk_size is not None else controller.chunk_size,
        device=controller.options.device,
        precision=controller.options.precision,
        label=label,
    )
    record = {
        "request": request.as_dict(),
        "status": status,
        "metrics": (
            {
                "ncc": ncc,
                "normalized_mse": 1.0 - ncc,
                "relative_power_error": power_error,
            }
            if status == "PASS_RUN"
            else {}
        ),
        "memory": {"swap_delta_peak_bytes": swap_delta, "peak_rss_bytes": 1024},
        "gpu_memory_after": {"peak_allocated_bytes": gpu_peak},
        "chunking": {"chunk_count": chunk_count},
        "wall_time_s": 0.1,
        "arrays_path": str(arrays) if arrays else None,
    }
    controller._record(
        record, controller.runs / f"{label}.json", write=True
    )


class TestGates:
    def test_the_declared_gates_are_the_ticket_s_numbers(self):
        assert NCC_GATE == 0.99
        assert STABILITY_TOLERANCE == 1.0e-3

    def test_the_manifest_states_the_gates_are_declared_targets(self, tmp_path):
        manifest = _controller(tmp_path).manifest()
        assert manifest["gates"]["ncc"] == NCC_GATE
        assert "declared engineering targets" in manifest["gates"]["basis"]

    def test_the_manifest_records_the_reference_implementation_as_unused(self, tmp_path):
        """The knowledge-pack claim is not being changed: not vendored, not executed."""
        reference = _controller(tmp_path).manifest()["reference_implementation"]
        assert reference["vendored"] is False
        assert reference["pinned"] is False
        assert reference["executed_by_this_repository"] is False
        assert "no commit SHA is recorded" in reference["note"]
        assert "acsphotonics" in reference["paper"]

    def test_the_manifest_declares_the_oracle_and_its_cross_checks(self, tmp_path):
        oracle = _controller(tmp_path).manifest()["oracle"]
        assert oracle["kind"] == "analytic"
        assert len(oracle["cross_checks"]) == 2
        assert any("chromatix" in check for check in oracle["cross_checks"])

    def test_the_manifest_records_the_band_limit_and_why(self, tmp_path):
        band = _controller(tmp_path).manifest()["grazing_band_limit"]
        assert band["direction_cosine_floor"] == 1.0e-2
        assert "4745 m" in band["why"]

    def test_the_manifest_records_the_sequential_gpu_policy(self, tmp_path):
        policy = _controller(tmp_path).manifest()["policy"]
        assert policy["gpu_count"] == 1
        assert "one candidate at a time" in policy["concurrency"]
        assert policy["autograd"].startswith("disabled")


class TestHostGuard:
    def test_a_request_beyond_the_declared_ray_cap_is_skipped_not_run(self, tmp_path):
        controller = _controller(tmp_path, memory_guard=True, max_total_rays=1000)
        record = controller.launch(
            controller._request(launch_count=100, samples_per_launch=100)
        )
        assert record["status"] == "SKIPPED_MEMORY_GUARD"
        assert "exceeds the declared sweep cap" in record["failure"]
        assert record["wall_time_s"] == 0.0

    def test_an_unsafe_chunk_is_skipped_before_it_allocates(self, tmp_path):
        """Phase 15's requirement: refused *before* allocation, not caught after.

        The projection uses the calibrated bytes/ray, so setting an absurd
        per-ray cost is the same arithmetic a real GPU-envelope breach would take.
        """
        controller = _controller(tmp_path, memory_guard=True)
        controller.per_ray_bytes = 1.0e12
        record = controller.launch(
            controller._request(
                launch_count=4, samples_per_launch=4, chunk_size=16
            )
        )
        assert record["status"] == "SKIPPED_MEMORY_GUARD"
        assert record["skip_kind"] == "memory"
        assert "usable GPU envelope" in record["failure"]
        assert f"{CHUNK_SAFETY_FACTOR}x" in record["failure"]
        assert controller.memory_failures, "a measured breach must halt growth"

    def test_the_guard_can_be_turned_off_and_then_does_not_skip(self, tmp_path):
        controller = _controller(tmp_path, memory_guard=False, max_total_rays=1)
        assert (
            controller._guard_before_launch(
                controller._request(launch_count=100, samples_per_launch=100)
            )
            is None
        )

    def test_the_host_reserve_blocks_a_launch_when_headroom_is_gone(
        self, tmp_path, monkeypatch
    ):
        import benchmarks.metalens_controller as module
        from core.resources import host_memory_snapshot

        real = host_memory_snapshot()
        monkeypatch.setattr(
            module,
            "host_memory_snapshot",
            lambda: real.__class__(**{**real.__dict__, "mem_available_bytes": 1024}),
        )
        controller = _controller(tmp_path, memory_guard=True)
        record = controller.launch(controller._request())
        assert record["status"] == "SKIPPED_MEMORY_GUARD"
        assert record["skip_kind"] == "memory"
        assert "at or below the" in record["failure"]


class TestChildFailures:
    """Requirement 17: a child that fails is a data point, not a crash."""

    def test_a_child_that_writes_nothing_is_recorded_cleanly(self, tmp_path, monkeypatch):
        controller = _controller(tmp_path, memory_guard=False)
        monkeypatch.setattr(
            sys, "executable", "/nonexistent/python-that-does-not-exist"
        )
        with pytest.raises(FileNotFoundError):
            controller.launch(controller._request())

    def test_a_child_that_exits_nonzero_without_a_result_is_a_failure_row(
        self, tmp_path, monkeypatch
    ):
        import benchmarks.metalens_controller as module

        class _Process:
            returncode = 3
            stdout = None

            def poll(self):
                return 3

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 3

        monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: _Process())
        controller = _controller(tmp_path, memory_guard=False)
        record = controller.launch(controller._request())
        assert record["status"] == "FAIL_NUMERICAL"
        assert "child wrote no result" in record["failure"]
        assert record["child_exit_code"] == 3
        assert controller.rows[-1]["status"] == "FAIL_NUMERICAL"

    def test_a_child_killed_for_memory_pressure_overrides_a_pass(self, tmp_path, monkeypatch):
        """Requirement 16: the watchdog's verdict wins over the child's own status."""
        import benchmarks.metalens_controller as module
        from core.resources import MemoryWatchdogVerdict

        controller = _controller(tmp_path, memory_guard=False)
        request = controller._request()
        stem = (
            f"{request.config}_P{request.launch_count}_S{request.samples_per_launch}"
            f"_{request.density}_seed{request.seed}_chunk{request.chunk_size}"
            f"_{request.precision}_001"
        )
        result_path = controller.runs / f"{stem}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "PASS_RUN",
                    "metrics": {"ncc": 0.999},
                    "request": request.as_dict(),
                }
            )
        )

        class _Process:
            returncode = 0
            stdout = None
            _calls = 0

            def poll(self):
                _Process._calls += 1
                return None if _Process._calls < 3 else 0

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        breach = MemoryWatchdogVerdict(
            breached=True, reason="swap_growth", detail="container swap grew by 4096 B"
        )
        original = module.MemoryWatchdog

        class _Watchdog(original):
            def __post_init__(self):
                super().__post_init__()
                self.verdict = breach

            def start(self):
                return self

            def stop(self):
                return self

        monkeypatch.setattr(module, "MemoryWatchdog", _Watchdog)
        monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: _Process())
        record = controller.launch(request)
        assert record["status"] == "FAIL_HOST_MEMORY_PRESSURE"
        assert "container swap grew" in record["failure"]
        assert controller.memory_failures[-1]["status"] == "FAIL_HOST_MEMORY_PRESSURE"

    def test_every_recorded_status_is_in_the_declared_vocabulary(self, tmp_path):
        controller = _controller(tmp_path, memory_guard=True, max_total_rays=1)
        controller.launch(
            controller._request(launch_count=8, samples_per_launch=8)
        )
        assert all(row["status"] in STATUSES for row in controller.rows)


class TestFailureMonotonicity:
    """Phase 33: after a memory failure, do not try a bigger population."""

    def test_a_memory_failure_halts_the_pilot_bracket(self, tmp_path):
        """A *measured* resource failure stops the bracket."""
        controller = _controller(tmp_path, memory_guard=True)
        controller.per_ray_bytes = 1.0e12
        controller.pilot_bracket()
        assert controller.memory_failures, "the GPU envelope should have fired"
        assert len(controller.rows) < 9, (
            "the bracket must stop after a memory guard, not run all nine points"
        )

    def test_the_declared_ray_cap_does_not_masquerade_as_a_memory_failure(self, tmp_path):
        """A scheduling budget is not evidence about the machine.

        Collapsing the two would let a conservative ray cap silently cancel the
        memory-scaling study, which is the opposite of conservative.
        """
        controller = _controller(tmp_path, memory_guard=True, max_total_rays=1)
        record = controller.launch(
            controller._request(launch_count=8, samples_per_launch=8)
        )
        assert record["status"] == "SKIPPED_MEMORY_GUARD"
        assert record["skip_kind"] == "budget"
        assert controller.memory_failures == []
        assert controller.budget_skips
        assert controller._memory_direction_blocked() is False

    def test_a_memory_failure_skips_the_memory_ladder_entirely(self, tmp_path):
        controller = _controller(tmp_path, memory_guard=True)
        controller.memory_failures.append(
            {"label": "x", "status": "FAIL_GPU_MEMORY", "reason": "y"}
        )
        before = len(controller.rows)
        controller.memory_ladder()
        assert len(controller.rows) == before
        assert any("memory ladder skipped" in note for note in controller.notes)


class TestConvergenceSelection:
    def test_the_smallest_qualifying_pair_wins_not_the_largest(self, tmp_path):
        """Phase 27's "smallest converged", with three qualifying candidates."""
        controller = _controller(tmp_path)
        psf = np.zeros((8, 8))
        psf[4, 4] = 1.0
        paths = {}
        for launches, samples in [(4, 4), (8, 4), (4, 8), (8, 8), (16, 16), (32, 32)]:
            path = tmp_path / f"a_{launches}_{samples}.npz"
            np.savez(path, test=psf, reference=psf)
            paths[(launches, samples)] = path
        for launches, samples in [(4, 4), (8, 4), (4, 8), (8, 8), (16, 16), (32, 32)]:
            _stub_record(
                controller,
                launches=launches,
                samples=samples,
                ncc=0.999,
                label=f"s_{launches}_{samples}",
                arrays=paths[(launches, samples)],
            )
        best = controller.converged_candidate()
        assert best is not None
        assert (best["P"], best["S"]) == (4, 4)
        assert best["stability"]["all_within_tolerance"] is True
        assert best["stability"]["doublings_available"] == 3

    def test_a_pair_below_the_gate_never_qualifies(self, tmp_path):
        controller = _controller(tmp_path)
        psf = np.zeros((8, 8))
        psf[4, 4] = 1.0
        for launches, samples in [(4, 4), (8, 4), (4, 8), (8, 8)]:
            path = tmp_path / f"b_{launches}_{samples}.npz"
            np.savez(path, test=psf, reference=psf)
            _stub_record(
                controller,
                launches=launches,
                samples=samples,
                ncc=0.5,
                label=f"t_{launches}_{samples}",
                arrays=path,
            )
        assert controller.converged_candidate() is None

    def test_a_pair_whose_doublings_disagree_does_not_qualify(self, tmp_path):
        """Meeting the gate is not enough; the plateau has to be local."""
        controller = _controller(tmp_path)
        # Every candidate gets a *distinct* PSF, so no doubling ever agrees with
        # its base and nothing can qualify -- including the pairs that are
        # themselves someone else's doubling.
        reference = np.zeros((16, 16))
        reference[8, 8] = 1.0
        for index, (launches, samples) in enumerate(
            [(4, 4), (8, 4), (4, 8), (8, 8), (16, 8), (8, 16), (16, 16)]
        ):
            distinct = np.zeros((16, 16))
            distinct[index, index] = 1.0
            path = tmp_path / f"c_{launches}_{samples}.npz"
            np.savez(path, test=distinct, reference=reference)
            _stub_record(
                controller,
                launches=launches,
                samples=samples,
                ncc=0.995,
                label=f"u_{launches}_{samples}",
                arrays=path,
            )
        assert controller.converged_candidate() is None

    def test_a_pair_with_no_doublings_at_all_does_not_qualify(self, tmp_path):
        controller = _controller(tmp_path)
        psf = np.zeros((8, 8))
        psf[4, 4] = 1.0
        path = tmp_path / "d.npz"
        np.savez(path, test=psf, reference=psf)
        _stub_record(controller, launches=4, samples=4, ncc=0.999, label="v", arrays=path)
        assert controller.converged_candidate() is None, (
            "with nothing to compare against, local stability is not established"
        )

    def test_only_the_primary_seed_density_and_configuration_are_eligible(self, tmp_path):
        controller = _controller(tmp_path)
        psf = np.zeros((8, 8))
        psf[4, 4] = 1.0
        for launches, samples in [(4, 4), (8, 4), (4, 8), (8, 8)]:
            path = tmp_path / f"e_{launches}_{samples}.npz"
            np.savez(path, test=psf, reference=psf)
            _stub_record(
                controller,
                launches=launches,
                samples=samples,
                ncc=0.999,
                seed=999,
                label=f"w_{launches}_{samples}",
                arrays=path,
            )
        assert controller.converged_candidate() is None


class TestSeedStatistics:
    def test_the_spread_over_seeds_is_reported_not_the_best_draw(self, tmp_path):
        controller = _controller(tmp_path, seeds=(1, 2, 3))
        for seed, ncc in ((1, 0.9990), (2, 0.9995), (3, 0.9985)):
            _stub_record(
                controller, launches=8, samples=8, ncc=ncc, seed=seed, label=f"x{seed}"
            )
        stats = controller.seed_statistics(8, 8)
        assert stats["trials"] == 3
        assert stats["mean_ncc"] == pytest.approx(0.999, abs=1e-6)
        assert stats["std_ncc"] > 0.0
        assert stats["min_ncc"] == pytest.approx(0.9985)

    def test_a_single_trial_reports_zero_spread_rather_than_nan(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(controller, launches=8, samples=8, ncc=0.999, label="y")
        assert controller.seed_statistics(8, 8)["std_ncc"] == 0.0

    def test_an_absent_pair_reports_nothing_rather_than_inventing_it(self, tmp_path):
        assert _controller(tmp_path).seed_statistics(8, 8) == {}


class TestBookkeeping:
    def test_the_controller_keeps_only_scalars_per_candidate(self, tmp_path):
        """Phase 34: nothing ray-sized or PSF-sized may accumulate in the parent."""
        controller = _controller(tmp_path)
        for index in range(20):
            _stub_record(controller, launches=4, samples=4, ncc=0.9, label=f"z{index}")
        for row in controller.rows:
            for value in row.values():
                assert not isinstance(value, np.ndarray)
                assert not isinstance(value, (list, dict)), (
                    "a container in a summary row is how a leak starts"
                )
        assert len(controller.parent_rss) == 20

    def test_the_parent_rss_trace_is_recorded_for_the_leak_check(self, tmp_path):
        controller = _controller(tmp_path)
        for index in range(5):
            _stub_record(controller, launches=4, samples=4, ncc=0.9, label=f"r{index}")
        summary = controller.summary(None, {}, [], controller.manifest())
        assert summary["memory"]["parent_rss_first_bytes"] > 0
        assert len(summary["memory"]["parent_rss_trace"]) == 5

    def test_the_tables_have_a_row_per_candidate(self, tmp_path):
        import csv

        controller = _controller(tmp_path)
        for index in range(4):
            _stub_record(
                controller, launches=4, samples=4 * (index + 1), ncc=0.9, label=f"c{index}"
            )
        controller.write_tables()
        for name in ("convergence.csv", "memory.csv"):
            with (controller.root / name).open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 4, name

    def test_a_nonzero_swap_delta_suppresses_the_literal_zero_swap_statement(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(controller, launches=4, samples=4, ncc=0.999, label="s1", swap_delta=0)
        _stub_record(controller, launches=8, samples=4, ncc=0.999, label="s2", swap_delta=4096)
        report = controller.report(controller.summary(None, {}, [], controller.manifest()))
        assert "Additional swap used by the benchmark: 0 bytes." not in report
        assert "deliberately" in report

    def test_all_zero_swap_yields_the_literal_statement(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(controller, launches=4, samples=4, ncc=0.999, label="s1", swap_delta=0)
        report = controller.report(controller.summary(None, {}, [], controller.manifest()))
        assert "Additional swap used by the benchmark: 0 bytes." in report
        assert "container" in report

    def test_an_unmeasurable_swap_reading_is_not_reported_as_zero(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(controller, launches=4, samples=4, ncc=0.9, label="s", swap_delta=None)
        summary = controller.summary(None, {}, [], controller.manifest())
        assert summary["swap"]["all_zero"] is False
        assert summary["swap"]["measured_candidates"] == 0

    def test_a_failed_sweep_reports_the_declared_failure_code(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(controller, launches=4, samples=4, ncc=0.5, label="f")
        summary = controller.summary(None, {}, [], controller.manifest())
        assert summary["final_status"] == "FAIL_NOT_CONVERGED_WITHIN_SAFE_MEMORY"
        assert "FAIL_NOT_CONVERGED_WITHIN_SAFE_MEMORY" in controller.report(summary)


class TestReproductionCommand:
    def test_the_command_runs_through_run_sh_and_has_no_placeholders(self, tmp_path):
        controller = _controller(
            tmp_path, validation_configs=("METALENS-SLAB-100",), seed=7
        )
        command = controller.reproduction_command()
        assert command.startswith("./run.sh --gpu python -m")
        assert "--seed 7" in command
        assert "METALENS-SLAB-100" in command
        assert "<" not in command and ">" not in command
        assert "actual" not in command


class TestCommandLine:
    def test_a_grid_size_that_contradicts_the_configuration_is_refused(self):
        with pytest.raises(SystemExit):
            main(["--output", "/tmp/x", "--grid-size", "256"])

    def test_an_unknown_configuration_is_refused(self):
        with pytest.raises(SystemExit):
            main(["--output", "/tmp/x", "--config", "NOPE"])


class TestPowerConvergence:
    """Phase 23: NCC is blind to a global scale, so the scale is reported too."""

    def test_the_power_criterion_selects_a_later_pair_than_ncc_alone(self, tmp_path):
        controller = _controller(tmp_path)
        # NCC is already met at the small pair, but its power is 25% high; the
        # larger pair is the first one whose radiometry has also converged.
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.9989, power_error=0.2469,
            label="small",
        )
        _stub_record(
            controller, launches=256, samples=1024, ncc=0.99998, power_error=0.0021,
            label="large",
        )
        power = controller.power_converged_candidate()
        assert power is not None
        assert (power["P"], power["S"]) == (256, 1024)

    def test_a_pair_below_the_ncc_gate_never_counts_as_power_converged(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(
            controller, launches=4, samples=16, ncc=0.5, power_error=0.0, label="bad"
        )
        assert controller.power_converged_candidate() is None

    def test_a_negative_power_error_counts_by_magnitude(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(
            controller, launches=8, samples=8, ncc=0.999, power_error=-0.5, label="under"
        )
        assert controller.power_converged_candidate() is None
        _stub_record(
            controller, launches=16, samples=8, ncc=0.999, power_error=-0.001, label="ok"
        )
        assert controller.power_converged_candidate()["P"] == 16

    def test_the_target_is_declared(self):
        assert POWER_ERROR_TARGET == 1.0e-2


class TestEffectiveChunk:
    """A requested chunk larger than the population is not the chunk used."""

    def test_the_effective_chunk_is_derived_from_what_the_child_did(self, tmp_path):
        controller = _controller(tmp_path)
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.999, label="one",
            chunk_size=2097152, chunk_count=1,
        )
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.999, label="four",
            chunk_size=1024, chunk_count=4,
        )
        assert controller.rows[0]["effective_chunk_size"] == 4096
        assert controller.rows[1]["effective_chunk_size"] == 1024

    def test_the_canonical_row_for_a_pair_is_the_largest_chunk_one(self, tmp_path):
        """Otherwise the report says the reported pair chunked when it did not."""
        controller = _controller(tmp_path)
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.999, label="rep",
            chunk_size=64, chunk_count=64,
        )
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.999, label="sweep",
            chunk_size=2097152, chunk_count=1,
        )
        found = controller._find(P=4, S=1024)
        assert found["label"] == "sweep"
        assert found["chunk_count"] == 1

    def test_chunk_replications_do_not_inflate_the_seed_trial_count(self, tmp_path):
        """They are copies of one realization, not extra draws from the ensemble."""
        controller = _controller(tmp_path, seeds=(1, 2))
        for label, chunk, count in (("sweep", 4096, 1), ("rep_a", 1024, 4), ("rep_b", 256, 16)):
            _stub_record(
                controller, launches=4, samples=1024, ncc=0.999, seed=1,
                label=label, chunk_size=chunk, chunk_count=count,
            )
        _stub_record(
            controller, launches=4, samples=1024, ncc=0.998, seed=2, label="seed2"
        )
        stats = controller.seed_statistics(4, 1024)
        assert stats["trials"] == 2
        assert stats["seeds"] == [1, 2]

    def test_the_memory_scaling_slopes_are_reported_as_numbers(self, tmp_path):
        controller = _controller(tmp_path)
        for chunk in (4096, 16384, 65536):
            _stub_record(
                controller, launches=64, samples=8192, ncc=0.999,
                label=f"ladder_chunk_{chunk}", chunk_size=chunk,
                chunk_count=524288 // chunk, gpu_peak=1000 * chunk,
            )
        for launches in (16, 64, 256):
            _stub_record(
                controller, launches=launches, samples=8192, ncc=0.999,
                label=f"ladder_fixed_chunk_N{launches * 8192}", chunk_size=4096,
                chunk_count=launches * 8192 // 4096, gpu_peak=4_000_000,
            )
        scaling = controller.summary(None, {}, [], controller.manifest())["memory_scaling"]
        assert scaling["vs_effective_chunk_at_fixed_total"]["log_log_slope"] == pytest.approx(
            1.0, abs=1e-9
        )
        assert scaling["vs_total_at_fixed_chunk"]["log_log_slope"] == pytest.approx(
            0.0, abs=1e-9
        )
