# The agent-evaluation harness — can an agent turn a physics problem into a right answer?

CHE-71 built this around six `A1-*` tasks. **CHE-133 retired those tasks and kept
the harness**, which is the part that took the arguing. Each task named one
library implicitly and had one right answer, so it graded tool use rather than
modelling, and none was ever run against a real agent. M9 authors the
replacement, posing problems over the B0–B4 families in
`src/verification/families/` rather than over one library each.

What was promoted out of the task set before it went, because it was the
expensive part:

| what | where it lives now |
|---|---|
| the five closed forms, each verified against the pinned solver before it shipped, with its measured agreement and the wrong answer its tolerance rejects | `src/verification/analytic.py`, destined for `B1-RAY-EFL`, `B1-RAY-PLATE`, `B1-WAVE-GAUSS`, `B1-WAVE-AIRY`, `B1-WAVE-TILT` |
| the two measured traps — Optiland's µm/nm slip and Chromatix's `kykx` 2π-and-sign — with the numbers the mistaken code actually returns | `src/verification/hazards.py`, destined for `B0-UNITS-01` and `B0-UNITS-02` |
| the exclusion table below, with its measured reasons | here, unchanged |

The question this asks is deliberately different from the one the family
registry asks. A family grades a **solver's physics** and its whole value is a
reproducible fingerprint. This grades an **agent's behaviour**: given a physics
problem in prose, can it work out which tool applies, build the simulation, run
it, and produce a number that is *right*?

```bash
# The harness's own unit tests. Deterministic, in the default gate, seconds.
./run.sh pytest -q tests/test_agent_benchmark.py

# The runner. No suite is registered, so this reports that and exits non-zero.
./run.sh python -m agent.benchmark_suite
```

---

## The five design decisions, as recorded

These are what survived the task set, and each is a decision with a reason
rather than a default.

### 1. What context does the agent under test receive?

**Declared per task, and recorded in every result.** This is the most
consequential knob here, because it changes *what is being measured* rather than
how well. This repository has mature knowledge packs —
`knowledge/solvers/{optiland,chromatix}/` each carry `card.yaml`,
`api_minimal_examples.md`, `conventions.md`, `usage_notes.md`,
`failure_guide.md` — and `AGENTS.md` instructs an agent to load a solver card when
the issue uses that solver. With those in context the benchmark asks *"can it
follow our cards"*; without them, *"can it discover and use the tool"*. Both are
legitimate, and they are **different benchmarks**.

| policy | what the participant is handed | what it measures |
|---|---|---|
| `cold` | the problem statement only | discovery: find the library and its API unaided |
| `warm` | `+ card.yaml`, `api_minimal_examples.md` | correct use of a documented tool |
| `guided` | `+ conventions.md` (which names the unit and sign hazards) | whether it reads a warning it has been handed |

The retired V1 set used `cold` for its four straightforward tasks and `warm` for
its two traps, on purpose: the point of a trap is not whether an agent can find
the library, it is whether it gets a unit right, and `cold` would confound the
two. The experiment that was never run and is still the interesting one:
`conventions.md` names both measured hazards, so the difference between `warm`
and `guided` on a trap task measures something specific — **whether handing an
agent the warning is enough.**

The policy is a property of the task, not of the invocation. `--context-policy`
therefore *asserts* rather than sets, and errors if it disagrees.

### 2. Where the suite lives

**Its own top-level directory, opt-in by location.** `testpaths = ["tests"]` and
`norecursedirs` both exclude `benchmarks/agents/`, following the CHE-67 precedent
exactly: a suite that runs an agent is nondeterministic, slow and consumes model
tokens, and `./run.sh pytest -q` is a required gate after every change. Opting
*in* by naming the directory is the safe default, and it cannot be forgotten the
way a marker exclusion can.

The split within that, and why:

| what | where | in the default suite? |
|---|---|---|
| harness and task registry (empty) | `src/agent/benchmark_suite.py` | — |
| grader / taxonomy / trial arithmetic tests | `tests/test_agent_benchmark.py` | **yes**, deterministic, fast, and it defines its own throwaway tasks so it tests the harness rather than a task set |
| prompts, recorded expectations, the graded run | `benchmarks/agents/` | no, opt-in |

The grader decides whether an agent passed, so a regression in *it* silently
changes every score the benchmark has ever produced. That is exactly what a
required gate should cover, which is why it is tested in `tests/` and imports no
solver.

### 3. How many trials per task

**Three, declared, and reported as a pass *rate* rather than a pass/fail.** A
single agent run is one realization of a stochastic process, and this project's
frozen rule for stochastic claims is ensemble statistics over one realization. The
rate is reported even at `--trials 1`, so a reader always sees the denominator.

A trial that failed because *the harness* broke is **void, not failed** — it is
excluded from the denominator rather than counted against the agent. A task whose
every trial was void reports no rate at all instead of zero.

Grading is cheap (the whole suite is seconds of solver time), so the trial count
costs nothing on this side. The budget it spends is the agent's.

### 4. Relationship to the existing benchmark registry

**Beside it, in a disjoint id space. Not inside it.** A physics family grades a
solver with declared tolerances and a reproducible fingerprint; merging a
nondeterministic agent score into a registry whose entire value is
reproducibility would spoil it. The family registry enforces its own
`family_id` uniqueness, which is what the retired `A1-*` collision test used to
guard — an agent task set is registered here, not there.

### 5. Failure taxonomy as structured codes

Following the repository's existing pattern (`ContractCode`, the 14 precision
codes): a code, a reason, and a remedy — never a free-text string — so results
aggregate.

| code | meaning |
|---|---|
| `AGENT_PASS` | every check passed |
| `FAIL_NO_SUBMISSION` | nothing produced inside the budget. Distinct from a crash: silence and a traceback need different follow-ups |
| `FAIL_PROBLEM_UNDERSTANDING` | a submission exists but does not answer the question: a required quantity is missing, or is not a number |
| `FAIL_TOOL_SELECTION` | the task's library was never used. A right answer from the wrong tool is still a failure of the thing being measured |
| `FAIL_SIMULATION_CONSTRUCTION` | the library was used, the setup was invalid |
| `FAIL_TOOL_EXECUTION` | the setup was accepted and the run raised, hung, or produced nothing |
| `FAIL_PHYSICAL_RESULT` | **it ran, it produced numbers, and the numbers are wrong** |
| `FAIL_HARNESS` | the harness broke. Never charged to the agent |

**Grading reports the FIRST stage that failed.** A run that never produced a
number cannot also be judged on its physics, and reporting the later failure would
misattribute the cause. So the order above is the order of the walk, and
`test_tool_selection_is_checked_before_the_numbers` pins it.

`FAIL_PHYSICAL_RESULT` is the code a trap task exists to produce, and every code
has a negative participant that produces it — `TestOutcomeCodesAreReachable`
proves each one fires. A code nothing has ever emitted cannot be trusted to fire
when it matters.

A task owns its own trap submission (`AgentTask.trap_submission`), so the
harness knows no optics. That is what let CHE-133 delete the task set without
touching this file.

---

### Candidates excluded, with the reason measured rather than guessed

Recording why a candidate is **not** a benchmark is expensive knowledge, and it
outlives the task set the candidates were rejected from. From the CHE-71 context
and the CHE-57 inventories:

| excluded | why |
|---|---|
| `t21_surface_roughness_scattering` | flaky: a hard threshold on an unseedable random quantity (CHE-65, open). Optiland's numba BSDFs are unreachable from NumPy's RNG and two identical calls differ by ~1% |
| `Optic.draw3D()` / `optic_viewer_3d` | hangs indefinitely headlessly (VTK finds no X server, no EGL, no OSMesa) |
| anything reading a vendor image | `scikit-image` is not installed, and installing it would change the pinned environment |
| `t08`/`t24` vendor `.zmx`/`.seq` | the artifacts are unreachable; `thorlabs.com` answers the documented URL with a 1313-byte HTML page |
| anything gated on `jax_enable_x64=True` | pinned `False` everywhere, and a process that flipped it would change every recorded number in the repository |
| `c04`/`c10` Adam-state examples | genuine traps, but each is ~5 min of optimization. Too expensive for a 3-trial suite; revisit for V2 with a smaller surrogate |
| `t35` `res.success` trap | the same objection — an optimizer run, and the trap needs the full 3-mirror system to reproduce |
| `t27` `workers=-1` | forks one process per CPU; `AGENTS.md` forbids parallel solver processes on this shared machine |
| `t39` wrong surface normal | a real, silent physics error, but the task would be "write correct calculus", not "simulate optics" |

The two Adam-state traps and `t35` are the most valuable rejections to record:
they are exactly the "it ran and it is wrong" shape this suite wants, and the
**only** reason they are out is runtime. A cheaper surrogate for them is the
single highest-value extension, and they are the shortlist M9 should start from.

---

## Adding a task

There is no task set to add to yet; M9 defines one. The requirements a task must
meet do not change:

1. **An analytic oracle**, verified against the pinned solver *before* the task
   ships. A recorded solver output is weaker in a way that matters — it cannot
   tell a wrong answer from a wrong reference.
2. **A `tolerance_basis` per `CheckSpec`** that states the measured agreement and
   names a wrong answer the tolerance rejects.
   `test_every_check_is_analytic_and_declares_its_tolerance_basis` fails without
   one.
3. **A prompt that names no library, function, module or tutorial.** Choosing the
   tool is the thing being measured, and `TestPromptsDoNotLeakTheAnswer` enforces
   it.
4. **A `trap_submission`** if it is a trap: the *measured* wrong number the
   plausible mistake produces, not an invented one.
5. Register it in `SUITES` in `src/agent/benchmark_suite.py`.

Start from the exclusion table above rather than from a blank page.

---

## Running an agent

The `command` participant is the pluggable slot. The contract is deliberately
minimal so any agent can satisfy it:

- the prompt is written to `<workspace>/prompt.md`;
- the files the task's context policy permits are copied into `<workspace>/context/`
  (empty for a `cold` task);
- the command runs with `<workspace>` as its working directory, with `{prompt}` and
  `{workspace}` substituted in its argv;
- it must leave a `submission.json` in the workspace.

```bash
./run.sh python -m agent.benchmark_suite \
    --suite <name> --trials 3 \
    --participant "command:my-agent --prompt {prompt} --cwd {workspace}" \
    --context-policy per-task --output outputs/agent_run
```

**This was implemented and never executed, and the reason is recorded rather than
glossed:** the `agent_solver` container has no agent CLI installed and no API
credentials, so running one would need a decision about spending model tokens
that belongs to whoever owns the budget. The container *does* have outbound
network access, so the only missing pieces are a CLI and a key.

What *was* executed is the whole harness against the reference and negative
participants, which is what establishes that the grading is sound. See
`benchmarks/reports/2026-08/agent_benchmark_v1.md` — a historical report, whose
per-task numbers describe the retired set and whose harness findings stand.

## Result format

`results.json` in the output directory. Per trial: the outcome code, its remedy,
every check with its expected/observed/relative-error, and a bounded stderr tail.
Per task: the trial count, the *valid* trial count, the pass rate and the outcome
histogram. Per suite: the participant, the declared trials, the per-task context
policies, and the suite pass rate.
