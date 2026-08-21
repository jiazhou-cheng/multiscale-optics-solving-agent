"""The V1 agent-benchmark harness, without running a solver or an agent (CHE-71).

This is the half of the benchmark that belongs in the default suite: the grader's
staging, the taxonomy, the trial arithmetic, the context policy and the CLI's
argument handling are all deterministic and cost milliseconds. The half that runs
solvers and agents is `benchmarks_agent/`, opt-in by location.

The split matters for one specific reason. The grader decides whether an agent
passed, so a regression in *it* silently changes every score the benchmark has
ever produced — which makes it exactly the kind of thing a required gate should
cover. Nothing here imports Optiland or Chromatix.
"""

from __future__ import annotations

import json
import math

import pytest

from multiscale_optics_agent.benchmarks.agent_suite import (
    OUTCOME_REMEDY,
    SUITE_V1,
    AgentTask,
    CheckSpec,
    ContextPolicy,
    Outcome,
    TaskResult,
    TrialResult,
    broken_participant,
    command_participant,
    grade,
    main,
    run_suite,
    task_by_id,
)


def _task(**overrides) -> AgentTask:
    defaults = {
        "task_id": "A1-TEST-01",
        "title": "a task",
        "library": "optiland",
        "context_policy": ContextPolicy.COLD,
        "exercises": "nothing; this is a fixture",
        "checks": (
            CheckSpec(
                key="value", description="a value", expected=10.0, unit="mm",
                rtol=1e-2, tolerance_basis="a fixture, so the basis is that it is a fixture",
            ),
        ),
        "reference": lambda: {"library": "optiland", "value": 10.0},
    }
    defaults.update(overrides)
    return AgentTask(**defaults)


class TestRegistry:
    def test_the_suite_is_six_tasks_with_unique_ids_in_the_a1_space(self):
        ids = [task.task_id for task in SUITE_V1]
        assert len(ids) == len(set(ids)) == 6
        assert all(task_id.startswith("A1-") for task_id in ids)

    def test_the_id_space_does_not_collide_with_the_solver_benchmark_registry(self):
        """Design decision 4: beside `benchmarks/manifest.yaml`, not inside it.

        Those IDs grade a solver's physics and their value is reproducible
        fingerprints; these grade an agent's behaviour and are nondeterministic.
        Sharing a namespace would invite someone to merge the two.
        """
        import yaml

        from multiscale_optics_agent.benchmarks.agent_suite import REPO_ROOT

        manifest = yaml.safe_load((REPO_ROOT / "benchmarks" / "manifest.yaml").read_text())
        existing = set()
        for value in _walk(manifest):
            if isinstance(value, str) and value.count("-") >= 1:
                existing.add(value)
        assert not {task.task_id for task in SUITE_V1} & existing

    def test_a_task_can_be_looked_up_and_an_unknown_one_names_the_suite(self):
        assert task_by_id("A1-OPT-01").library == "optiland"
        with pytest.raises(KeyError) as raised:
            task_by_id("A1-NOPE-99")
        assert "A1-OPT-01" in str(raised.value)

    def test_every_task_has_a_prompt_and_a_recorded_expectation_on_disk(self):
        for task in SUITE_V1:
            assert task.prompt_path.exists(), task.task_id
            assert task.expected_path.exists(), task.task_id
            recorded = json.loads(task.expected_path.read_text())
            assert recorded["task_id"] == task.task_id

    def test_every_task_states_what_it_exercises(self):
        for task in SUITE_V1:
            assert len(task.exercises) > 40, task.task_id

    def test_every_outcome_code_has_a_remedy(self):
        assert set(OUTCOME_REMEDY) == set(Outcome)
        for code, remedy in OUTCOME_REMEDY.items():
            assert remedy, code


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)
    else:
        yield node


class TestCheckSpec:
    def test_a_value_inside_the_tolerance_passes(self):
        spec = _task().checks[0]
        assert spec.evaluate(10.05).passed
        assert spec.evaluate(10.05).relative_error == pytest.approx(5e-3, rel=1e-6)

    def test_a_value_outside_the_tolerance_fails(self):
        spec = _task().checks[0]
        result = spec.evaluate(11.0)
        assert not result.passed
        assert "relative error" in result.detail

    @pytest.mark.parametrize("value", ["10.0", None, [10.0], {"v": 10.0}])
    def test_a_non_number_is_not_silently_coerced(self, value):
        result = _task().checks[0].evaluate(value)
        assert not result.passed
        assert result.observed is None

    def test_a_bool_is_not_a_number(self):
        """`True == 1` in Python, so this needs saying explicitly."""
        assert not _task().checks[0].evaluate(True).passed

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_value_fails_rather_than_comparing(self, value):
        result = _task().checks[0].evaluate(value)
        assert not result.passed
        assert "not finite" in result.detail

    def test_the_sign_is_graded(self):
        """Several tasks turn on a sign, so the metric must not be magnitude-only."""
        assert not _task().checks[0].evaluate(-10.0).passed


class TestGradingStages:
    """The staging is the substance: the FIRST failed stage is what gets reported."""

    def test_a_correct_submission_passes(self):
        task = _task()
        result = grade(task, {"library": "optiland", "value": 10.0})
        assert result.outcome is Outcome.PASS
        assert result.passed

    def test_no_submission_is_distinct_from_a_crash(self):
        assert grade(_task(), None).outcome is Outcome.FAIL_NO_SUBMISSION

    def test_a_self_reported_execution_error_is_not_a_physics_failure(self):
        result = grade(_task(), {"library": "optiland", "error": "RuntimeError: boom"})
        assert result.outcome is Outcome.FAIL_TOOL_EXECUTION

    def test_a_self_reported_construction_error_gets_its_own_code(self):
        result = grade(
            _task(), {"library": "optiland", "error": "could not construct the system"}
        )
        assert result.outcome is Outcome.FAIL_SIMULATION_CONSTRUCTION

    def test_the_wrong_library_fails_even_with_the_right_number(self):
        """A right answer from the wrong tool is still a failure of the measurement."""
        result = grade(_task(), {"library": "numpy", "value": 10.0})
        assert result.outcome is Outcome.FAIL_TOOL_SELECTION
        assert "wrong tool" in result.detail

    def test_the_library_match_is_case_insensitive_and_trimmed(self):
        assert grade(_task(), {"library": " Optiland ", "value": 10.0}).passed

    def test_a_missing_quantity_is_a_problem_understanding_failure(self):
        result = grade(_task(), {"library": "optiland"})
        assert result.outcome is Outcome.FAIL_PROBLEM_UNDERSTANDING
        assert "value" in result.detail

    def test_a_non_numeric_answer_is_a_problem_understanding_failure(self):
        result = grade(_task(), {"library": "optiland", "value": "ten"})
        assert result.outcome is Outcome.FAIL_PROBLEM_UNDERSTANDING

    def test_a_wrong_number_is_a_physical_result_failure_and_nothing_else(self):
        """The central distinction: it ran, it answered, the answer is wrong."""
        result = grade(_task(), {"library": "optiland", "value": 42.0})
        assert result.outcome is Outcome.FAIL_PHYSICAL_RESULT
        assert result.checks and not result.checks[0].passed

    def test_tool_selection_is_checked_before_the_numbers(self):
        """Otherwise a wrong-tool run with a wrong number reports the wrong cause."""
        result = grade(_task(), {"library": "numpy", "value": 999.0})
        assert result.outcome is Outcome.FAIL_TOOL_SELECTION

    def test_a_missing_quantity_is_reported_before_a_wrong_one(self):
        task = _task(
            checks=(
                CheckSpec("a", "a", 1.0, "mm", 1e-2, "fixture basis, long enough to pass"),
                CheckSpec("b", "b", 2.0, "mm", 1e-2, "fixture basis, long enough to pass"),
            )
        )
        result = grade(task, {"library": "optiland", "a": 99.0})
        assert result.outcome is Outcome.FAIL_PROBLEM_UNDERSTANDING

    def test_partial_correctness_still_fails_and_says_which_check(self):
        task = _task(
            checks=(
                CheckSpec("a", "a", 1.0, "mm", 1e-2, "fixture basis, long enough to pass"),
                CheckSpec("b", "b", 2.0, "mm", 1e-2, "fixture basis, long enough to pass"),
            )
        )
        result = grade(task, {"library": "optiland", "a": 1.0, "b": 99.0})
        assert result.outcome is Outcome.FAIL_PHYSICAL_RESULT
        assert "b:" in result.detail and "a:" not in result.detail
        assert [check.passed for check in result.checks] == [True, False]


class TestTrialArithmetic:
    """Design decision 3: a pass *rate*, and a harness fault is void not failed."""

    def test_the_pass_rate_is_reported_with_its_denominator(self):
        task = TaskResult(task=_task())
        task.trials = [
            TrialResult("A1-TEST-01", Outcome.PASS, "", ()),
            TrialResult("A1-TEST-01", Outcome.PASS, "", ()),
            TrialResult("A1-TEST-01", Outcome.FAIL_PHYSICAL_RESULT, "", ()),
        ]
        assert task.pass_rate == pytest.approx(2 / 3)
        assert len(task.valid_trials) == 3

    def test_a_harness_fault_is_excluded_from_the_denominator(self):
        task = TaskResult(task=_task())
        task.trials = [
            TrialResult("A1-TEST-01", Outcome.PASS, "", ()),
            TrialResult("A1-TEST-01", Outcome.FAIL_HARNESS, "", ()),
        ]
        assert task.pass_rate == 1.0, "a broken harness must not count against the agent"
        assert len(task.valid_trials) == 1

    def test_a_task_with_only_harness_faults_has_no_rate_rather_than_zero(self):
        task = TaskResult(task=_task())
        task.trials = [TrialResult("A1-TEST-01", Outcome.FAIL_HARNESS, "", ())]
        assert task.pass_rate is None

    def test_a_single_trial_still_reports_a_rate_so_the_denominator_is_visible(self):
        task = TaskResult(task=_task())
        task.trials = [TrialResult("A1-TEST-01", Outcome.PASS, "", ())]
        assert task.pass_rate == 1.0
        assert task.as_dict()["trials"] == 1

    def test_zero_trials_is_refused(self):
        with pytest.raises(ValueError):
            run_suite(broken_participant("silent"), name="x", trials=0, tasks=(_task(),))

    def test_a_participant_that_raises_becomes_a_void_trial_not_a_crash(self, tmp_path):
        def explode(task, workspace):
            raise RuntimeError("the participant exploded")

        suite = run_suite(
            explode, name="explosive", trials=2, output=tmp_path, tasks=(_task(),)
        )
        assert all(
            trial.outcome is Outcome.FAIL_HARNESS
            for trial in suite.tasks[0].trials
        )
        assert suite.as_dict()["total_valid_trials"] == 0

    def test_the_declared_trial_count_reaches_the_result(self, tmp_path):
        suite = run_suite(
            broken_participant("silent"), name="x", trials=3, output=tmp_path,
            tasks=(_task(),),
        )
        record = suite.as_dict()
        assert record["declared_trials_per_task"] == 3
        assert record["results"][0]["trials"] == 3


class TestContextPolicy:
    """Design decision 1: declared per task, recorded in every result."""

    def test_a_cold_task_is_offered_nothing_but_its_prompt(self):
        assert _task(context_policy=ContextPolicy.COLD).context_files() == ()

    def test_a_warm_task_is_offered_the_card_and_the_examples_but_not_conventions(self):
        names = [path.name for path in _task(context_policy=ContextPolicy.WARM).context_files()]
        assert "solver_card.yaml" in names
        assert "conventions.md" not in names

    def test_a_guided_task_is_additionally_offered_the_conventions(self):
        warm = _task(context_policy=ContextPolicy.WARM).context_files()
        guided = _task(context_policy=ContextPolicy.GUIDED).context_files()
        assert len(guided) > len(warm)
        assert any(path.name == "conventions.md" for path in guided)

    def test_the_policy_is_recorded_for_every_task_in_the_result(self, tmp_path):
        suite = run_suite(
            broken_participant("silent"), name="x", trials=1, output=tmp_path,
        )
        recorded = suite.as_dict()["context_policies"]
        assert recorded == {task.task_id: str(task.context_policy) for task in SUITE_V1}


class TestSuiteRecord:
    def test_the_result_declares_its_disjoint_id_space(self, tmp_path):
        record = run_suite(
            broken_participant("silent"), name="x", trials=1, output=tmp_path,
            tasks=(_task(),),
        ).as_dict()
        assert "disjoint" in record["id_space"]

    def test_the_record_is_json_serializable_and_written_to_the_output(self, tmp_path):
        run_suite(
            broken_participant("silent"), name="x", trials=1, output=tmp_path,
            tasks=(_task(),),
        )
        record = json.loads((tmp_path / "results.json").read_text())
        assert record["participant"] == "x"
        assert record["results"][0]["results"][0]["remedy"]

    def test_each_trial_carries_its_remedy_so_a_score_is_actionable(self):
        result = grade(_task(), {"library": "numpy", "value": 1.0})
        assert result.as_dict()["remedy"] == OUTCOME_REMEDY[Outcome.FAIL_TOOL_SELECTION]

    def test_the_stderr_tail_is_bounded(self):
        result = grade(_task(), None, stderr="x" * 10_000)
        assert len(result.as_dict()["stderr_tail"]) == 2000


class TestCommandParticipant:
    def test_a_missing_command_is_reported_rather_than_raised(self, tmp_path):
        participant = command_participant(["/nonexistent/agent-binary"])
        submission, stderr = participant(SUITE_V1[0], tmp_path)
        assert submission is None
        assert "not installed" in stderr

    def test_the_prompt_and_the_permitted_context_are_staged_in_the_workspace(self, tmp_path):
        warm = next(
            task for task in SUITE_V1 if task.context_policy is not ContextPolicy.COLD
        )
        command_participant(["/nonexistent/agent-binary"])(warm, tmp_path)
        assert (tmp_path / "prompt.md").read_text() == warm.prompt()
        staged = {path.name for path in (tmp_path / "context").iterdir()}
        assert staged == {path.name for path in warm.context_files()}

    def test_a_cold_task_stages_an_empty_context_directory(self, tmp_path):
        cold = next(
            task for task in SUITE_V1 if task.context_policy is ContextPolicy.COLD
        )
        command_participant(["/nonexistent/agent-binary"])(cold, tmp_path)
        assert list((tmp_path / "context").iterdir()) == []

    def test_malformed_submission_json_is_an_execution_failure_not_a_crash(self, tmp_path):
        (tmp_path / "submission.json").write_text("{not json")
        participant = command_participant(["true"])
        submission, _ = participant(SUITE_V1[0], tmp_path)
        assert submission is not None and "not valid JSON" in submission["error"]
        assert grade(SUITE_V1[0], submission).outcome is Outcome.FAIL_TOOL_EXECUTION


class TestCommandLine:
    def test_an_unknown_participant_is_refused(self):
        with pytest.raises(SystemExit):
            main(["--participant", "nonsense"])

    def test_a_context_policy_that_contradicts_a_task_is_refused(self):
        """The policy is a property of the task, not of the invocation."""
        with pytest.raises(SystemExit):
            main(["--participant", "broken:silent", "--context-policy", "guided"])

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(SystemExit) as raised:
            main(["--task", "A1-NOPE-99"])
        assert raised.value.code != 0

    def test_a_failing_participant_exits_nonzero(self, tmp_path):
        code = main(
            [
                "--participant", "broken:silent", "--trials", "1",
                "--task", "A1-OPT-01", "--output", str(tmp_path),
            ]
        )
        assert code == 1

    def test_asserting_the_policy_that_a_task_actually_declares_is_accepted(self, tmp_path):
        code = main(
            [
                "--participant", "broken:silent", "--trials", "1",
                "--task", "A1-OPT-01", "--context-policy", "cold",
                "--output", str(tmp_path),
            ]
        )
        assert code == 1  # the participant fails; the policy assertion did not
