# V1 agent benchmark — can an agent turn a physics problem into a right answer?

CHE-71. Six tasks, three per supported library, every one graded against an
analytic oracle.

The question this asks is deliberately different from the one
`benchmarks/manifest.yaml` asks. That registry grades a **solver's physics** and
its whole value is reproducible fingerprints. This grades an **agent's
behaviour**: given a physics problem in prose, can it work out which tool applies,
build the simulation, run it, and produce a number that is *right*?

```bash
# The suite's gate on itself: reference + negative participants. Seconds.
make test-agent-benchmark            # ./run.sh pytest -q benchmarks_agent

# One end-to-end run, 3 trials per task, against the reference participant.
make agent-benchmark

# Against a deliberately-wrong participant, to see the taxonomy fire.
make agent-benchmark PARTICIPANT=broken:trap

# Against a real agent (see "Running an agent" below).
./run.sh python -m multiscale_optics_agent.benchmarks.agent_suite \
    --suite v1 --trials 3 --participant "command:my-agent --prompt {prompt}" \
    --context-policy per-task --output outputs/che71_agent_v1
```

---

## The five design decisions, as recorded

### 1. What context does the agent under test receive?

**Declared per task, and recorded in every result.** This is the most
consequential knob here, because it changes *what is being measured* rather than
how well. This repository has mature knowledge packs —
`knowledge/solvers/{optiland,chromatix}/` each carry `solver_card.yaml`,
`api_minimal_examples.md`, `conventions.md`, `capability_notes.md`,
`failure_guide.md` — and `AGENTS.md` instructs an agent to load a solver card when
the issue uses that solver. With those in context the benchmark asks *"can it
follow our cards"*; without them, *"can it discover and use the tool"*. Both are
legitimate, and they are **different benchmarks**.

| policy | what the participant is handed | what it measures |
|---|---|---|
| `cold` | the problem statement only | discovery: find the library and its API unaided |
| `warm` | `+ solver_card.yaml`, `api_minimal_examples.md` | correct use of a documented tool |
| `guided` | `+ conventions.md` (which names the unit and sign hazards) | whether it reads a warning it has been handed |

V1 uses `cold` for the four straightforward tasks and `warm` for the two traps.
The traps are `warm` on purpose: the point is not whether an agent can find the
library, it is whether it gets a unit right, and `cold` would confound the two. A
`guided` run of the same two tasks is the natural next experiment — `conventions.md`
names both hazards, so the difference between `warm` and `guided` on `A1-OPT-03`
and `A1-CHX-03` measures something specific: **whether handing an agent the
warning is enough.**

The policy is a property of the task, not of the invocation. `--context-policy`
therefore *asserts* rather than sets, and errors if it disagrees.

### 2. Where the suite lives

**Its own top-level directory, opt-in by location.** `testpaths = ["tests"]` and
`norecursedirs` both exclude `benchmarks_agent/`, following the CHE-67 precedent
exactly: a suite that runs an agent is nondeterministic, slow and consumes model
tokens, and `./run.sh pytest -q` is a required gate after every change. Opting
*in* by naming the directory is the safe default, and it cannot be forgotten the
way a marker exclusion can.

The split within that, and why:

| what | where | in the default suite? |
|---|---|---|
| harness, task registry, reference implementations | `src/multiscale_optics_agent/benchmarks/agent_suite.py` | — |
| grader / taxonomy / trial arithmetic tests | `tests/test_agent_benchmark.py` | **yes**, deterministic, 53 tests, 0.1 s |
| prompts, recorded expectations, the graded run | `benchmarks_agent/` | no, opt-in |

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

**Beside it, with a disjoint ID space (`A1-*`). Not inside it.** Those benchmarks
(`L1-RAY-01`, `L2-PSF-01`, `L3-HYBRID-01`) grade a solver's physics with declared
tolerances and scientific fingerprints; merging a nondeterministic agent score
into a registry whose entire value is reproducibility would spoil it.
`manifest.yaml`, its entries, tolerances and gates are untouched, and
`test_the_id_space_does_not_collide_with_the_solver_benchmark_registry` holds the
namespaces apart so nobody merges them later by accident.

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

`FAIL_PHYSICAL_RESULT` is the code the two trap tasks exist to produce, and every
other code has a negative participant that produces it —
`TestOutcomeCodesAreReachable` proves each one fires. A code nothing has ever
emitted cannot be trusted to fire when it matters.

---

## The tasks

| id | title | library | policy | oracle | trap |
|---|---|---|---|---|---|
| `A1-OPT-01` | Focal length of a thick plano-convex singlet | optiland | cold | `R/(n-1)` and `EFL - t/n`, exact | — |
| `A1-OPT-02` | Focal shift caused by a plane-parallel plate | optiland | cold | `t(1 - 1/n)`, exact; sign graded | — |
| `A1-OPT-03` | Single-layer AR coating at 550 nm | optiland | warm | Fresnel + single-layer quarter-wave, exact | **µm vs nm** |
| `A1-CHX-01` | Diffractive spreading of a Gaussian beam | chromatix | cold | `w0 sqrt(1 + (z/zR)^2)`, exact | — |
| `A1-CHX-02` | First dark ring of a focused circular aperture | chromatix | cold | `0.61 lambda / NA`, exact | — |
| `A1-CHX-03` | Lateral walk-off of a tilted beam | chromatix | warm | `z tan(theta)`, exact; sign graded | **`kykx` 2π** |

Every check is `analytic`: the expected value is a closed form, **verified against
the pinned solver before the task shipped**, and every one carries a
`tolerance_basis` stating the measured agreement and what the tolerance is wide
enough to admit and narrow enough to reject. A recorded solver output would be
weaker in a way that matters — it cannot tell a wrong answer from a wrong
reference. `expected/` therefore holds a *regression signal*, not the oracle, and
says so in every file.

### The two traps, and why they are the interesting failures

Both are **measured** on the pinned versions, not hypothetical, and both are cases
where the code runs perfectly and the physics is wrong:

**`A1-OPT-03` — micrometres where the literature says nanometres.**
`ThinFilmStack.add_layer` takes µm; a quarter-wave MgF₂ layer for 550 nm is
naturally quoted as 99.64 nm. Passing `99.64` builds a layer 1000× too thick and
the reflectance comes back **0.04216384** against bare glass's **0.04216456** — the
coating does nothing, nothing raises, and the number looks like a reflectance.
CHE-57 recorded the same hazard on upstream tutorial t07. The task grades
`coating_thickness_nm` *separately* from the reflectances precisely so the report
can tell a unit slip inside the model from a wrong design intent.

**`A1-CHX-03` — the same parameter name, two units, and a sign.** `kykx` means
cycles per length on `asm_propagate` and radians per length on `plane_wave` — a
factor of 2π — and the resulting displacement runs *opposite* in sign to the
parameter (CHE-57 finding on example c06). Either mistake is 6.28× or a sign away
from the truth, far outside the tolerance, and neither raises.

A taxonomy that collapsed "it ran" into "it worked" could express neither.

---

## Adding a benchmark

1. **Find a candidate with an analytic oracle.** Start from the CHE-57 inventories
   (`knowledge/solvers/{optiland,chromatix}/tutorials/README.md`) — they classify
   every check as reference / analytic / invariant / qualitative, so the
   `analytic` column is the shortlist. Check it against the exclusions below.
2. **Verify the oracle against the pinned solver first**, before writing anything
   else. If the closed form and the solver disagree, you have found something more
   interesting than a benchmark task, and it belongs in a ticket.
3. **Write the reference implementation** as a `_reference_<id>()` in
   `agent_suite.py`, returning a dict with `library` and one key per check.
4. **Write the `AgentTask`**: id in the `A1-*` space, library, context policy, what
   it `exercises`, and a `CheckSpec` per graded quantity. Every `CheckSpec` needs a
   `tolerance_basis` that states the measured agreement and names a wrong answer
   the tolerance rejects. `test_every_check_is_analytic_and_declares_its_tolerance_basis`
   will fail without one.
5. **Write the prompt** in `prompts/<id>.md`. State the physics, the required
   submission keys and their units. Do **not** name a library, a function, a module
   or a tutorial — `TestPromptsDoNotLeakTheAnswer` enforces this, and choosing the
   tool is the thing being measured.
6. **If it is a trap**, fill in `trap` with the *measured* wrong number the
   plausible mistake produces, and add the mistake to `_trap_submission` so a
   negative participant exercises it.
7. `./run.sh python -m multiscale_optics_agent.benchmarks.agent_suite --write-expected`
8. `make test-agent-benchmark` and `./run.sh pytest -q tests/test_agent_benchmark.py`.

### Candidates excluded, with the reason measured rather than guessed

From the CHE-71 context and the CHE-57 inventories:

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
they are exactly the "it ran and it is wrong" shape this suite wants, and the only
reason they are out is runtime. A V2 with a cheaper surrogate for them is the
single highest-value extension.

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
./run.sh python -m multiscale_optics_agent.benchmarks.agent_suite \
    --suite v1 --trials 3 \
    --participant "command:my-agent --prompt {prompt} --cwd {workspace}" \
    --context-policy per-task --output outputs/che71_agent_v1
```

**This was implemented but not executed in the CHE-71 delivery run, and the reason
is recorded rather than glossed:** the `agent_solver` container has no agent CLI
installed and no API credentials, so running one would need a decision about
spending model tokens that belongs to whoever owns the budget. The container *does*
have outbound network access, so the only missing pieces are a CLI and a key.

What *was* executed is the whole harness against the reference and negative
participants, which is what establishes that the grading is sound. See
`benchmarks/CHE71_AGENT_BENCHMARK_REPORT.md`.

## Result format

`results.json` in the output directory. Per trial: the outcome code, its remedy,
every check with its expected/observed/relative-error, and a bounded stderr tail.
Per task: the trial count, the *valid* trial count, the pass rate and the outcome
histogram. Per suite: the participant, the declared trials, the per-task context
policies, and the suite pass rate.
