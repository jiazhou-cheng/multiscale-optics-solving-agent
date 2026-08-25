"""The agent-evaluation harness: what a task is, how a trial is graded, what a
run reports. (CHE-71, retired task set removed by CHE-133.)

This grades an **agent's behaviour**, which is nondeterministic by nature --
distinct from a physics family, whose whole value is a reproducible fingerprint.
Putting a nondeterministic score into the family registry would spoil it, so
this sits beside it with its own runner.

**There is no shipped suite.** The six ``A1-*`` tasks this harness was written
for were retired by CHE-133: each named one library implicitly and had one right
answer, so it graded tool use rather than modelling, and none was ever run
against a real agent. Their closed forms and their two measured traps were
promoted out first -- ``verification/analytic.py`` and ``verification/hazards.py``
-- and M9 authors the replacement over the B0-B4 families.

What survived, and why each piece is worth keeping
--------------------------------------------------
The harness is the part that took the arguing:

* **Context policy is a property of the task, not the invocation.** ``cold`` /
  ``warm`` / ``guided`` change *what is being measured* -- discovery, correct use
  of a documented tool, or whether a handed warning gets read -- so ``--context-
  policy`` asserts rather than sets.
* **Trials are declared and reported as a rate.** A single agent run is one
  realization of a stochastic process, and this project's rule for stochastic
  claims is ensemble statistics. The denominator is always printed.
* **Void is not failed.** A trial that broke because the *harness* broke is
  excluded from the denominator rather than charged to the agent, and a task
  whose every trial was void reports no rate at all instead of zero.
* **Eight structured outcome codes, not a boolean.** ``FAIL_PHYSICAL_RESULT`` --
  it ran, it produced numbers, and the numbers are wrong -- is a different
  finding from ``FAIL_TOOL_EXECUTION``, and a suite that collapsed them could
  express neither of the measured traps.
* **Grading reports the FIRST stage that failed.** A run that never produced a
  number cannot also be judged on its physics, and reporting the later failure
  would misattribute the cause.
* **A pluggable ``command:`` participant**, so any agent CLI can be measured
  without changing this file.

A task defines its own oracle and its own trap submission. The harness knows
nothing about optics; that is what let the task set be deleted without touching
it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.paths import repository_root

__all__ = [
    "SUITES",
    "AgentTask",
    "CheckResult",
    "CheckSpec",
    "ContextPolicy",
    "Outcome",
    "Participant",
    "SuiteResult",
    "TaskResult",
    "TrialResult",
    "broken_participant",
    "command_participant",
    "grade",
    "main",
    "reference_participant",
    "registered_tasks",
    "run_suite",
    "task_by_id",
]

# Located by marker, not by counting parents. The count was 3 before CHE-89
# flattened src/, and would have needed changing again for CHE-90's move -- and a
# wrong count does not fail at import, it points REPO_ROOT somewhere plausible.
REPO_ROOT = repository_root()
SUITE_DIR = REPO_ROOT / "benchmarks/agents"


class ContextPolicy(StrEnum):
    """What the agent under test is allowed to see. Design decision 1.

    This is the most consequential knob in the whole harness, because it changes
    *what is being measured* rather than how well. With the knowledge pack in
    context the benchmark asks "can it follow our cards"; without it, "can it
    discover and use the tool". Both are legitimate and they are different
    benchmarks, so the policy is declared per task and recorded in every result —
    otherwise scores are not comparable across runs.
    """

    #: Problem statement only. Measures discovery: the agent must find the library
    #: and its API itself.
    COLD = "cold"
    #: Problem statement plus the solver card and minimal API examples. Measures
    #: whether the agent can use a documented tool correctly.
    WARM = "warm"
    #: WARM plus the conventions document, which names the unit and sign hazards.
    #: Measures whether the agent reads a warning it has been handed.
    GUIDED = "guided"


class Outcome(StrEnum):
    """Structured outcome codes. Design decision 5.

    Follows the repository's existing pattern (`ContractCode`, the precision
    codes): a code, a reason and a remedy, never a free-text string, so results
    aggregate.

    The ordering matters. Grading walks these in sequence and reports the *first*
    stage that failed, because a run that never produced a field cannot also be
    judged on its physics, and reporting the later failure would misattribute the
    cause.
    """

    #: Every check passed.
    PASS = "AGENT_PASS"
    #: Nothing was submitted inside the budget. Distinct from a crash: silence and
    #: a traceback need different follow-ups.
    FAIL_NO_SUBMISSION = "FAIL_NO_SUBMISSION"
    #: A submission exists but does not answer the question asked -- a required
    #: quantity is missing, or is not a number, or its declared unit is wrong.
    FAIL_PROBLEM_UNDERSTANDING = "FAIL_PROBLEM_UNDERSTANDING"
    #: The task's library was never used. The agent solved (or guessed) something
    #: else -- possibly with the right answer, which is still a benchmark failure
    #: because the task is about tool use.
    FAIL_TOOL_SELECTION = "FAIL_TOOL_SELECTION"
    #: The library was used but the setup was invalid: the solver refused it, or
    #: the declared configuration contradicts the problem statement.
    FAIL_SIMULATION_CONSTRUCTION = "FAIL_SIMULATION_CONSTRUCTION"
    #: The setup was accepted and the run raised, hung, or produced nothing.
    FAIL_TOOL_EXECUTION = "FAIL_TOOL_EXECUTION"
    #: **It ran, it produced numbers, and the numbers are wrong.** The one code
    #: the two trap tasks exist to produce.
    FAIL_PHYSICAL_RESULT = "FAIL_PHYSICAL_RESULT"
    #: The harness broke. Never charged to the agent, and never counted as a
    #: failed trial in the pass rate.
    FAIL_HARNESS = "FAIL_HARNESS"


#: What each outcome means for whoever reads the score, and what to do about it.
OUTCOME_REMEDY: dict[Outcome, str] = {
    Outcome.PASS: "nothing",
    Outcome.FAIL_NO_SUBMISSION: (
        "raise the budget, or check the participant wrote submission.json where the "
        "task said to"
    ),
    Outcome.FAIL_PROBLEM_UNDERSTANDING: (
        "the prompt may be ambiguous about the required outputs or their units; "
        "read the submission before blaming the agent"
    ),
    Outcome.FAIL_TOOL_SELECTION: (
        "under a cold policy this is the measurement, not a defect; under warm or "
        "guided it means the pack did not make the tool findable"
    ),
    Outcome.FAIL_SIMULATION_CONSTRUCTION: (
        "usually a schema or convention the pack does not state; candidate for a "
        "knowledge-pack addition"
    ),
    Outcome.FAIL_TOOL_EXECUTION: (
        "an API the pack documents wrongly, or an environment gap; check the "
        "captured stderr before anything else"
    ),
    Outcome.FAIL_PHYSICAL_RESULT: (
        "the interesting failure. The agent's code ran; its physics did not. Check "
        "the task's trap note first"
    ),
    Outcome.FAIL_HARNESS: "fix the harness; this trial is void, not failed",
}


@dataclass(frozen=True)
class CheckSpec:
    """One graded quantity, its oracle, and the strength of that oracle.

    ``kind`` follows CHE-57's classification so the two inventories can be read
    together. Every check in V1 is ``analytic``: the expected value is a closed
    form, verified against the pinned solver before the task shipped. A recorded
    solver output would be weaker in a way that matters -- it cannot distinguish a
    wrong answer from a wrong reference.
    """

    key: str
    description: str
    expected: float
    unit: str
    #: Relative tolerance. Stated per check with a reason in ``tolerance_basis``,
    #: never a house default.
    rtol: float
    tolerance_basis: str
    kind: str = "analytic"

    def evaluate(self, value: Any) -> CheckResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return CheckResult(
                spec=self, observed=None, passed=False,
                detail=f"{self.key} is {type(value).__name__}, not a number",
            )
        observed = float(value)
        if not math.isfinite(observed):
            return CheckResult(
                spec=self, observed=observed, passed=False,
                detail=f"{self.key} is not finite",
            )
        error = (
            abs(observed - self.expected) / abs(self.expected)
            if self.expected
            else abs(observed)
        )
        return CheckResult(
            spec=self,
            observed=observed,
            passed=error <= self.rtol,
            detail=f"relative error {error:.3e} against rtol {self.rtol:.3e}",
            relative_error=error,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "expected": self.expected,
            "unit": self.unit,
            "rtol": self.rtol,
            "tolerance_basis": self.tolerance_basis,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class CheckResult:
    spec: CheckSpec
    observed: float | None
    passed: bool
    detail: str
    relative_error: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "expected": self.spec.expected,
            "observed": self.observed,
            "unit": self.spec.unit,
            "passed": self.passed,
            "relative_error": self.relative_error,
            "detail": self.detail,
        }


def _repo_relative(path: Path) -> str:
    """A repository-relative path where that is meaningful, absolute otherwise.

    A prompt outside the checkout is legitimate -- an external suite, a test's
    tmpdir -- and ``relative_to`` raising on one would make the record
    unwritable for a reason that has nothing to do with the run.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class AgentTask:
    """One benchmark task: the problem, what it exercises, and how it is graded."""

    task_id: str
    title: str
    #: The import name the submission must show evidence of having used. This is
    #: what makes tool *selection* gradable separately from tool *use*.
    library: str
    context_policy: ContextPolicy
    #: What capability the task is meant to exercise, in one sentence. Read by a
    #: human deciding whether a failure is interesting.
    exercises: str
    checks: tuple[CheckSpec, ...]
    reference: Callable[[], dict[str, Any]]
    #: Non-empty only for the trap tasks: the plausible mistake, and the wrong
    #: number it produces. Printed next to a FAIL_PHYSICAL_RESULT.
    trap: str = ""
    #: The submission the *measured* plausible mistake actually produces, used by
    #: the negative participants. A task owns this rather than the harness
    #: keyed by task id -- the harness must not know any optics, which is what
    #: let CHE-133 delete the A1 task set without touching this file.
    trap_submission: Callable[[], dict[str, Any]] | None = None
    notes: str = ""
    #: Where the prompt lives, when it is not the conventional location. A suite
    #: that keeps its prompts elsewhere -- or a test that writes one to a tmpdir
    #: -- should not have to plant a file in `benchmarks/agents/prompts/`.
    prompt_file: Path | None = None

    @property
    def prompt_path(self) -> Path:
        return self.prompt_file or SUITE_DIR / "prompts" / f"{self.task_id}.md"

    @property
    def expected_path(self) -> Path:
        return SUITE_DIR / "expected" / f"{self.task_id}.json"

    def prompt(self) -> str:
        return self.prompt_path.read_text()

    def context_files(self) -> tuple[Path, ...]:
        """The files a participant may read, per this task's declared policy."""
        if self.context_policy is ContextPolicy.COLD:
            return ()
        solver = REPO_ROOT / "knowledge" / "solvers" / self.library
        files = [solver / "card.yaml", solver / "api_minimal_examples.md"]
        if self.context_policy is ContextPolicy.GUIDED:
            files.append(solver / "conventions.md")
        return tuple(path for path in files if path.exists())

    def required_keys(self) -> tuple[str, ...]:
        return tuple(check.key for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "library": self.library,
            "context_policy": str(self.context_policy),
            "exercises": self.exercises,
            "required_submission_keys": list(self.required_keys()),
            "checks": [check.as_dict() for check in self.checks],
            "trap": self.trap,
            "notes": self.notes,
            "prompt": _repo_relative(self.prompt_path),
        }


# --------------------------------------------------------------------------- #
# The suite registry
# --------------------------------------------------------------------------- #
#
# Empty. The six A1-* tasks were retired by CHE-133 and M9 owns the replacement,
# which will pose problems over the B0-B4 families rather than over one library
# each. An empty registry is the honest state: a placeholder task would be a
# benchmark nobody designed, and leaving the old set in place would have kept a
# task layer alive purely so this dict had a value.

#: name -> tasks. ``run_suite`` takes the tasks it is given, so a caller with its
#: own tasks (a test, an experiment) does not need to register them here.
SUITES: dict[str, tuple[AgentTask, ...]] = {}


def registered_tasks() -> tuple[AgentTask, ...]:
    return tuple(task for suite in SUITES.values() for task in suite)


def task_by_id(task_id: str) -> AgentTask:
    known = registered_tasks()
    for task in known:
        if task.task_id == task_id:
            return task
    if not known:
        raise KeyError(
            f"unknown task {task_id!r}: no suite is registered. The A1-* set was "
            "retired by CHE-133 and M9 owns its replacement."
        )
    raise KeyError(f"unknown task {task_id!r}; registered: {[t.task_id for t in known]}")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def grade(task: AgentTask, submission: dict[str, Any] | None, *, stderr: str = "") -> TrialResult:
    """Judge one submission, reporting the FIRST stage that failed.

    The staging is the substance of the taxonomy. A run that never produced a
    number cannot also be judged on its physics, and reporting the later failure
    would misattribute the cause -- so the checks below are ordered and the walk
    stops at the first one that fails.
    """
    started = time.time()
    if submission is None:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_NO_SUBMISSION,
            detail="no submission.json was produced", checks=(), stderr=stderr,
            wall_time_s=0.0,
        )
    if submission.get("error"):
        # The participant itself reports it could not run. Which stage that was is
        # its own claim, and is trusted only as far as the vocabulary allows.
        claimed = str(submission["error"])
        outcome = (
            Outcome.FAIL_SIMULATION_CONSTRUCTION
            if "construct" in claimed.lower()
            else Outcome.FAIL_TOOL_EXECUTION
        )
        return TrialResult(
            task_id=task.task_id, outcome=outcome, detail=claimed, checks=(),
            stderr=stderr, wall_time_s=0.0,
        )
    library = str(submission.get("library", "")).strip().lower()
    if library != task.library:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_TOOL_SELECTION,
            detail=(
                f"the submission declares library={library!r}; this task is about "
                f"{task.library!r}. A right answer from the wrong tool is still a "
                "failure of the thing being measured"
            ),
            checks=(), stderr=stderr, wall_time_s=0.0,
        )
    missing = [key for key in task.required_keys() if key not in submission]
    if missing:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_PROBLEM_UNDERSTANDING,
            detail=f"required quantities absent from the submission: {missing}",
            checks=(), stderr=stderr, wall_time_s=0.0,
        )
    results = tuple(check.evaluate(submission[check.key]) for check in task.checks)
    non_numeric = [r for r in results if r.observed is None]
    if non_numeric:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_PROBLEM_UNDERSTANDING,
            detail="; ".join(r.detail for r in non_numeric),
            checks=results, stderr=stderr, wall_time_s=0.0,
        )
    failed = [r for r in results if not r.passed]
    outcome = Outcome.PASS if not failed else Outcome.FAIL_PHYSICAL_RESULT
    return TrialResult(
        task_id=task.task_id,
        outcome=outcome,
        detail=(
            "every check passed"
            if not failed
            else "; ".join(f"{r.spec.key}: {r.detail}" for r in failed)
        ),
        checks=results,
        stderr=stderr,
        wall_time_s=time.time() - started,
    )


@dataclass
class TrialResult:
    task_id: str
    outcome: Outcome
    detail: str
    checks: tuple[CheckResult, ...]
    stderr: str = ""
    wall_time_s: float = 0.0
    trial: int = 0

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "outcome": str(self.outcome),
            "detail": self.detail,
            "remedy": OUTCOME_REMEDY[self.outcome],
            "checks": [check.as_dict() for check in self.checks],
            "wall_time_s": self.wall_time_s,
            "stderr_tail": self.stderr[-2000:] if self.stderr else "",
        }


@dataclass
class TaskResult:
    """Every trial of one task, and the pass *rate* rather than a pass/fail.

    Design decision 3. A single agent run is one realization of a stochastic
    process, so one lucky run must not read as a capability. The rate is reported
    even when the trial count is 1, so a reader always sees the denominator.
    """

    task: AgentTask
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def valid_trials(self) -> list[TrialResult]:
        """Trials the agent is accountable for. A harness failure is void."""
        return [t for t in self.trials if t.outcome is not Outcome.FAIL_HARNESS]

    @property
    def pass_rate(self) -> float | None:
        valid = self.valid_trials
        return (sum(t.passed for t in valid) / len(valid)) if valid else None

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial in self.trials:
            counts[str(trial.outcome)] = counts.get(str(trial.outcome), 0) + 1
        return {
            "task": self.task.as_dict(),
            "trials": len(self.trials),
            "valid_trials": len(self.valid_trials),
            "passes": sum(t.passed for t in self.valid_trials),
            "pass_rate": self.pass_rate,
            "outcome_counts": counts,
            "results": [trial.as_dict() for trial in self.trials],
        }


@dataclass
class SuiteResult:
    participant: str
    context_policies: dict[str, str]
    trials: int
    tasks: list[TaskResult] = field(default_factory=list)
    started_unix: float = 0.0
    wall_time_s: float = 0.0
    #: Which registered suite this was, or a caller-supplied label. Recorded
    #: rather than hard-coded: the id space is no longer a fixed one.
    suite: str = "unregistered"

    def as_dict(self) -> dict[str, Any]:
        valid = [trial for task in self.tasks for trial in task.valid_trials]
        rates = [task.pass_rate for task in self.tasks if task.pass_rate is not None]
        return {
            "suite": self.suite,
            "participant": self.participant,
            "declared_trials_per_task": self.trials,
            "context_policies": self.context_policies,
            "started_unix": self.started_unix,
            "wall_time_s": self.wall_time_s,
            "task_count": len(self.tasks),
            "total_valid_trials": len(valid),
            "total_passes": sum(trial.passed for trial in valid),
            "suite_pass_rate": (sum(rates) / len(rates)) if rates else None,
            "tasks_fully_passed": sum(1 for task in self.tasks if task.pass_rate == 1.0),
            "outcome_counts": {
                code: sum(1 for trial in valid if str(trial.outcome) == code)
                for code in sorted({str(trial.outcome) for trial in valid})
            },
            "results": [task.as_dict() for task in self.tasks],
        }


# --------------------------------------------------------------------------- #
# Participants. A participant turns a task and a workspace into a submission.
# --------------------------------------------------------------------------- #

#: A participant is called with (task, workspace) and returns (submission, stderr).
Participant = Callable[[AgentTask, Path], tuple[dict[str, Any] | None, str]]


def reference_participant(task: AgentTask, workspace: Path) -> tuple[dict[str, Any], str]:
    """Run the shipped reference solution.

    This exists to grade the **grader**. A harness whose reference solutions do not
    pass is measuring something other than what it claims, and no agent score from
    it means anything -- so the reference run is the first thing the opt-in suite
    does, and it is a hard gate.
    """
    submission = task.reference()
    (workspace / "submission.json").write_text(json.dumps(submission, indent=2))
    return submission, ""


def broken_participant(mode: str) -> Participant:
    """A participant that fails in exactly one declared way.

    Every outcome code needs a participant that produces it, or the taxonomy is
    decoration: a code nothing has ever emitted cannot be trusted to fire when it
    matters. Each mode below is a *plausible* mistake, and the two ``trap`` modes
    reproduce the measured unit and convention errors the trap tasks are built on.
    """

    def participate(task: AgentTask, workspace: Path) -> tuple[dict[str, Any] | None, str]:
        if mode == "silent":
            return None, ""
        if mode == "wrong_tool":
            submission: dict[str, Any] = {
                "library": "numpy",
                **{key: 0.0 for key in task.required_keys()},
            }
        elif mode == "missing_quantity":
            submission = {"library": task.library}
        elif mode == "non_numeric":
            submission = {
                "library": task.library,
                **{key: "about right" for key in task.required_keys()},
            }
        elif mode == "execution_error":
            submission = {
                "library": task.library,
                "error": "RuntimeError: the solver raised while tracing",
            }
        elif mode == "construction_error":
            submission = {
                "library": task.library,
                "error": "PrescriptionError: could not construct the system",
            }
        elif mode == "trap":
            submission = _trap_submission(task)
        else:  # pragma: no cover - guarded by the CLI's choices
            raise ValueError(f"unknown broken mode {mode!r}")
        (workspace / "submission.json").write_text(json.dumps(submission, indent=2))
        return submission, ""

    return participate


def _trap_submission(task: AgentTask) -> dict[str, Any]:
    """The answer the *measured* plausible mistake actually produces.

    Owned by the task, not by a lookup table here: the value must be the output
    of running the mistaken code against the pinned solver, so that the taxonomy
    is exercised by the failure mode the task was designed around rather than by
    an invented placeholder. A task with no trap submission cannot be handed to
    the ``broken:trap`` participant, and saying so is better than substituting
    something plausible.
    """
    if task.trap_submission is None:
        raise KeyError(
            f"{task.task_id} declares no trap_submission, so there is no measured "
            "wrong answer to hand a negative participant"
        )
    return task.trap_submission()


def command_participant(argv: Sequence[str], *, timeout_s: float = 1800.0) -> Participant:
    """Run an external agent CLI in the workspace and read its ``submission.json``.

    The contract is deliberately minimal so any agent can satisfy it: the prompt is
    written to ``prompt.md``, the permitted context files (per the task's declared
    policy) are copied into ``context/``, and the command is run with the workspace
    as its working directory. ``{prompt}`` and ``{workspace}`` in ``argv`` are
    substituted.

    **Not executed in the CHE-71 delivery run**, and the reason is recorded rather
    than glossed: the container has no agent CLI installed and no API credentials,
    so running an agent would need a decision about spending model tokens that
    belongs to whoever owns the budget. What is delivered is the harness plus the
    reference and negative participants that validate it.
    """

    def participate(task: AgentTask, workspace: Path) -> tuple[dict[str, Any] | None, str]:
        prompt_path = workspace / "prompt.md"
        prompt_path.write_text(task.prompt())
        context = workspace / "context"
        context.mkdir(exist_ok=True)
        for source in task.context_files():
            (context / source.name).write_text(source.read_text())
        command = [
            argument.replace("{prompt}", str(prompt_path)).replace(
                "{workspace}", str(workspace)
            )
            for argument in argv
        ]
        try:
            completed = subprocess.run(
                command, cwd=str(workspace), capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired:
            return None, f"the participant exceeded its {timeout_s} s budget"
        except FileNotFoundError as exc:
            return None, f"the participant command is not installed: {exc}"
        target = workspace / "submission.json"
        if not target.exists():
            return None, stderr
        try:
            return json.loads(target.read_text()), stderr
        except json.JSONDecodeError as exc:
            return {"error": f"submission.json is not valid JSON: {exc}"}, stderr

    return participate


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_suite(
    participant: Participant,
    *,
    name: str,
    trials: int = 3,
    output: Path | None = None,
    tasks: Sequence[AgentTask] = (),
    suite: str = "unregistered",
) -> SuiteResult:
    """Run every task ``trials`` times, sequentially, and record everything."""
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    started = time.time()
    result_bundle = SuiteResult(
        participant=name,
        context_policies={task.task_id: str(task.context_policy) for task in tasks},
        trials=trials,
        started_unix=started,
        suite=suite,
    )
    for task in tasks:
        task_result = TaskResult(task=task)
        for trial in range(trials):
            workspace = (
                (output / "workspaces" / f"{task.task_id}_trial{trial}")
                if output
                else Path(os.environ.get("TMPDIR", "/tmp")) / f"agentsuite_{task.task_id}_{trial}"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            try:
                submission, stderr = participant(task, workspace)
                result = grade(task, submission, stderr=stderr)
            except Exception as exc:
                result = TrialResult(
                    task_id=task.task_id, outcome=Outcome.FAIL_HARNESS,
                    detail=f"{type(exc).__name__}: {exc}", checks=(),
                )
            result.trial = trial
            task_result.trials.append(result)
            print(
                f"[agent-suite] {task.task_id} trial {trial}: {result.outcome}"
                + (f" — {result.detail}" if not result.passed else ""),
                flush=True,
            )
        result_bundle.tasks.append(task_result)
    result_bundle.wall_time_s = time.time() - started
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(
            json.dumps(result_bundle.as_dict(), indent=2, default=str)
        )
    return result_bundle


def write_expected() -> list[Path]:
    """Record each reference solution's output under ``benchmarks/agents/expected/``.

    Mirrors the CHE-57 tutorial pattern: a recorded output is a regression signal,
    **not** the oracle. The oracle is the closed form on every ``CheckSpec``, which
    is why re-recording cannot make a wrong answer pass.
    """
    written = []
    for task in registered_tasks():
        record = task.reference()
        payload = {
            "task_id": task.task_id,
            "library": task.library,
            "recorded_reference_output": record,
            "checks": [check.as_dict() for check in task.checks],
            "graded_against": (
                "the closed forms in `checks`, not this recording. Re-recording "
                "cannot make a wrong answer pass."
            ),
        }
        task.expected_path.parent.mkdir(parents=True, exist_ok=True)
        task.expected_path.write_text(json.dumps(payload, indent=2, default=str))
        written.append(task.expected_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default=None,
        help="a registered suite name. None are registered; see SUITES.",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--participant", default="reference",
        help=(
            "'reference', 'broken:<mode>', or 'command:<argv...>' with {prompt} and "
            "{workspace} placeholders"
        ),
    )
    parser.add_argument(
        "--context-policy", default="per-task",
        help=(
            "informational only: the policy is declared per task and recorded in "
            "the results. Pass 'per-task' (the default) or a policy name to assert "
            "that every task uses it"
        ),
    )
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args(argv)

    if args.write_expected:
        for path in write_expected():
            print(f"wrote {path}")
        return 0

    if args.suite is not None and args.suite not in SUITES:
        parser.error(
            f"unknown suite {args.suite!r}; registered: {sorted(SUITES)}. The A1-* set "
            "was retired by CHE-133 and M9 owns its replacement."
        )
    try:
        if args.task:
            tasks = tuple(task_by_id(task_id) for task_id in args.task)
        elif args.suite is not None:
            tasks = SUITES[args.suite]
        else:
            tasks = registered_tasks()
    except KeyError as exc:
        # A mistyped --task is a usage error, not a traceback. The message already
        # lists the suite, so surface it rather than re-deriving one.
        parser.error(str(exc.args[0]))
        return 2
    if not tasks:
        parser.error(
            "no tasks to run: no suite is registered. The A1-* set was retired by "
            "CHE-133; M9 authors the replacement over the B0-B4 families."
        )
    if args.context_policy != "per-task":
        mismatched = [
            task.task_id for task in tasks if str(task.context_policy) != args.context_policy
        ]
        if mismatched:
            parser.error(
                f"--context-policy {args.context_policy!r} does not match the declared "
                f"policy of {mismatched}; the policy is a property of the task"
            )

    spec = args.participant
    if spec == "reference":
        participant, name = reference_participant, "reference"
    elif spec.startswith("broken:"):
        mode = spec.split(":", 1)[1]
        participant, name = broken_participant(mode), spec
    elif spec.startswith("command:"):
        participant = command_participant(spec.split(":", 1)[1].split())
        name = spec
    else:
        parser.error(f"unknown participant {spec!r}")
        return 2

    result_bundle = run_suite(
        participant,
        name=name,
        trials=args.trials,
        output=args.output,
        tasks=tasks,
        suite=args.suite or "unregistered",
    )
    record = result_bundle.as_dict()
    print(
        f"\n[agent-suite] participant={name} tasks={record['task_count']} "
        f"trials/task={args.trials} "
        f"passes={record['total_passes']}/{record['total_valid_trials']} "
        f"suite_pass_rate={record['suite_pass_rate']}"
    )
    for code, count in record["outcome_counts"].items():
        print(f"      {code}: {count}")
    return 0 if record["total_passes"] == record["total_valid_trials"] else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
