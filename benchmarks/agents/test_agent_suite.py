"""The graded V1 agent-benchmark run. Opt-in by location (CHE-71).

    ./run.sh pytest -q benchmarks/agents          # or: make test-agent-benchmark

Never collected by `./run.sh pytest -q`, because `testpaths = ["tests"]` and
`norecursedirs` both exclude this directory. That is deliberate and follows the
CHE-67 precedent: a suite that runs an agent is nondeterministic, slow and
consumes model tokens, and the default suite is a required gate after every
change. Opting *in* by naming the directory is the safe default.

What runs here, and what does not
---------------------------------
Everything here runs the **reference** and **negative** participants, so it is
fully deterministic and takes seconds. It is the suite's own gate:

* every reference solution must pass every check — a harness whose known-good
  solutions fail is measuring something other than what it claims, and no agent
  score from it would mean anything;
* every outcome code must be *reachable* — a code nothing has ever emitted cannot
  be trusted to fire when it matters;
* the two measured traps must produce `FAIL_PHYSICAL_RESULT` and not
  `FAIL_TOOL_EXECUTION` — the distinction the whole taxonomy exists for.

Running an actual agent is `--participant command:...` on the CLI runner, and is
not done here: it needs credentials and a token budget that belong to whoever owns
them. See `README.md` §"Running an agent".
"""

from __future__ import annotations

import json

import pytest

from agent.benchmark_suite import (
    SUITE_V1,
    ContextPolicy,
    Outcome,
    broken_participant,
    reference_participant,
    run_suite,
)

pytestmark = [pytest.mark.integration]

_IDS = [task.task_id for task in SUITE_V1]


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    """One reference pass over the whole suite. Deterministic, so one trial suffices."""
    return run_suite(
        reference_participant,
        name="reference",
        trials=1,
        output=tmp_path_factory.mktemp("reference"),
    )


class TestReferenceSolutionsGradeTheGrader:
    def test_every_reference_solution_passes_every_check(self, reference_run):
        failures = [
            (task.task.task_id, trial.outcome, trial.detail)
            for task in reference_run.tasks
            for trial in task.trials
            if not trial.passed
        ]
        assert not failures, f"the harness cannot grade its own solutions: {failures}"
        assert reference_run.as_dict()["suite_pass_rate"] == 1.0

    def test_the_suite_is_six_tasks_three_per_library(self, reference_run):
        libraries = [task.task.library for task in reference_run.tasks]
        assert len(libraries) == 6
        assert libraries.count("optiland") == 3
        assert libraries.count("chromatix") == 3

    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_every_check_is_analytic_and_declares_its_tolerance_basis(self, task):
        """A tolerance with no stated basis is a number someone picked."""
        assert task.checks
        for check in task.checks:
            assert check.kind == "analytic", (
                f"{task.task_id}.{check.key} is {check.kind}; V1 requires an "
                "analytic oracle, because a recorded solver output cannot tell a "
                "wrong answer from a wrong reference"
            )
            assert len(check.tolerance_basis) > 40, (
                f"{task.task_id}.{check.key} has no substantive tolerance basis"
            )
            assert 0.0 < check.rtol <= 0.05

    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_the_recorded_expectation_matches_the_analytic_oracle(self, task):
        """`expected/` is a regression signal; the closed form is the oracle.

        Both are checked against each other so a re-recording cannot quietly become
        the thing being graded.
        """
        recorded = json.loads(task.expected_path.read_text())
        assert recorded["task_id"] == task.task_id
        assert "not this recording" in recorded["graded_against"]
        for check in task.checks:
            observed = recorded["recorded_reference_output"][check.key]
            assert check.evaluate(observed).passed, (
                f"{task.task_id}.{check.key}: the recorded reference output no "
                "longer satisfies its own closed form"
            )


class TestOutcomeCodesAreReachable:
    """A code nothing has ever emitted cannot be trusted to fire when it matters."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("silent", Outcome.FAIL_NO_SUBMISSION),
            ("wrong_tool", Outcome.FAIL_TOOL_SELECTION),
            ("missing_quantity", Outcome.FAIL_PROBLEM_UNDERSTANDING),
            ("non_numeric", Outcome.FAIL_PROBLEM_UNDERSTANDING),
            ("execution_error", Outcome.FAIL_TOOL_EXECUTION),
            ("construction_error", Outcome.FAIL_SIMULATION_CONSTRUCTION),
            ("trap", Outcome.FAIL_PHYSICAL_RESULT),
        ],
    )
    def test_each_negative_participant_produces_exactly_its_code(
        self, mode, expected, tmp_path
    ):
        suite = run_suite(
            broken_participant(mode), name=f"broken:{mode}", trials=1,
            output=tmp_path / mode,
        )
        observed = {
            trial.outcome for task in suite.tasks for trial in task.trials
        }
        assert observed == {expected}, f"broken:{mode} produced {observed}"
        assert suite.as_dict()["total_passes"] == 0

    def test_every_declared_code_except_the_harness_fault_is_covered(self):
        """The one code with no participant is the one an agent cannot cause."""
        covered = {
            Outcome.PASS,
            Outcome.FAIL_NO_SUBMISSION,
            Outcome.FAIL_TOOL_SELECTION,
            Outcome.FAIL_PROBLEM_UNDERSTANDING,
            Outcome.FAIL_TOOL_EXECUTION,
            Outcome.FAIL_SIMULATION_CONSTRUCTION,
            Outcome.FAIL_PHYSICAL_RESULT,
        }
        assert set(Outcome) - covered == {Outcome.FAIL_HARNESS}


class TestTheTrapsAreTheInterestingFailure:
    """`it ran` must not be reported as `it worked`."""

    @pytest.mark.parametrize("task_id", ["A1-OPT-03", "A1-CHX-03"])
    def test_a_trap_task_declares_its_measured_trap(self, task_id):
        task = next(item for item in SUITE_V1 if item.task_id == task_id)
        assert "MEASURED TRAP" in task.trap
        assert task.trap.count(".") > 3, "a trap note must say what wrong number it gives"

    def test_the_ar_unit_slip_is_graded_as_wrong_physics_not_a_crash(self, tmp_path):
        """The submission is well-formed, the code ran, the reflectance is wrong."""
        task = next(item for item in SUITE_V1 if item.task_id == "A1-OPT-03")
        suite = run_suite(
            broken_participant("trap"), name="trap", trials=1,
            output=tmp_path / "ar", tasks=(task,),
        )
        trial = suite.tasks[0].trials[0]
        assert trial.outcome is Outcome.FAIL_PHYSICAL_RESULT
        # The thickness is right and the reflectance is not: the report can tell a
        # unit slip inside the model from a wrong design intent.
        by_key = {check.spec.key: check for check in trial.checks}
        assert by_key["coating_thickness_nm"].passed
        assert by_key["uncoated_reflectance"].passed
        assert not by_key["coated_reflectance"].passed

    def test_the_kykx_confusion_is_graded_as_wrong_physics(self, tmp_path):
        task = next(item for item in SUITE_V1 if item.task_id == "A1-CHX-03")
        suite = run_suite(
            broken_participant("trap"), name="trap", trials=1,
            output=tmp_path / "kykx", tasks=(task,),
        )
        trial = suite.tasks[0].trials[0]
        assert trial.outcome is Outcome.FAIL_PHYSICAL_RESULT
        assert trial.checks[0].observed < 0.0, "the trap reproduces the sign flip too"


class TestContextPolicyIsDeclaredAndRecorded:
    def test_every_task_declares_a_policy_and_it_reaches_the_result(self, reference_run):
        recorded = reference_run.as_dict()["context_policies"]
        assert set(recorded) == set(_IDS)
        for task in SUITE_V1:
            assert recorded[task.task_id] == str(task.context_policy)

    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_a_cold_task_offers_no_context_and_a_warm_one_does(self, task):
        files = task.context_files()
        if task.context_policy is ContextPolicy.COLD:
            assert files == (), "a cold task must hand over nothing but the prompt"
        else:
            assert files, (
                f"{task.task_id} is {task.context_policy} but no knowledge-pack file "
                "for its library exists"
            )
            assert all(path.exists() for path in files)

    def test_the_suite_uses_more_than_one_policy(self):
        """Otherwise the knob is documentation rather than a measurement."""
        assert len({task.context_policy for task in SUITE_V1}) >= 2


class TestPromptsDoNotLeakTheAnswer:
    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_the_prompt_names_no_function_module_or_tutorial(self, task):
        prompt = task.prompt()
        assert len(prompt) > 400
        lowered = prompt.lower()
        for leak in (
            "optiland", "chromatix", "asm_propagate", "ff_lens", "thinfilmstack",
            "add_layer", "thinfilmcoating", "plane_wave", "paraxial", "tutorial",
        ):
            assert leak not in lowered, (
                f"{task.task_id}'s prompt names {leak!r}; choosing the tool is the "
                "thing being measured"
            )
        # No Python either. The submission contract's JSON block is the only code
        # a prompt may contain; a `python` fence would be handing over the answer.
        assert "```python" not in lowered, f"{task.task_id}'s prompt contains code"

    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_the_prompt_asks_for_exactly_the_graded_keys(self, task):
        prompt = task.prompt()
        for key in task.required_keys():
            assert key in prompt, f"{task.task_id} grades {key} without asking for it"
        assert "submission.json" in prompt
        assert '"library"' in prompt

    @pytest.mark.parametrize("task", SUITE_V1, ids=_IDS)
    def test_the_prompt_states_the_units_it_will_be_graded_in(self, task):
        prompt = task.prompt().lower()
        units = {check.unit for check in task.checks}
        for unit in units:
            token = {"um": "micrometre", "mm": "millimetre", "nm": "nanometre",
                     "fraction": "fraction"}[unit]
            assert token in prompt, (
                f"{task.task_id} grades in {unit} without telling the agent"
            )
