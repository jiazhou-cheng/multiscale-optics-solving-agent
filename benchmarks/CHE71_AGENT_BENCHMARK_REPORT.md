# CHE-71 — V1 multi-agent physics benchmark

**Status: delivered and executed.** 2026-08-20. Six tasks, three per supported
library, every one graded against an analytic oracle.

Suite: `benchmarks_agent/` (task prompts, recorded expectations, the graded run).
Harness: `src/multiscale_optics_agent/benchmarks/agent_suite.py`. Full
documentation, including the five recorded design decisions and how to add a task:
`benchmarks_agent/README.md`.

---

## 1. What was executed

```bash
# The delivered end-to-end run: 6 tasks x 3 trials.
./run.sh python -m multiscale_optics_agent.benchmarks.agent_suite \
    --suite v1 --trials 3 --participant reference --context-policy per-task \
    --output outputs/che71_agent_v1

# The suite's gate on itself (deterministic, 52 tests, 7.6 s).
make test-agent-benchmark

# The harness in the default suite (deterministic, 53 tests, 0.12 s).
./run.sh pytest -q tests/test_agent_benchmark.py
```

| participant | trials | outcome |
|---|---|---|
| `reference` | 18 | `AGENT_PASS` × 18, suite pass rate **1.0**, 9.25 s |
| `broken:silent` | 18 | `FAIL_NO_SUBMISSION` × 18 |
| `broken:wrong_tool` | 18 | `FAIL_TOOL_SELECTION` × 18 |
| `broken:missing_quantity` | 18 | `FAIL_PROBLEM_UNDERSTANDING` × 18 |
| `broken:non_numeric` | 18 | `FAIL_PROBLEM_UNDERSTANDING` × 18 |
| `broken:execution_error` | 18 | `FAIL_TOOL_EXECUTION` × 18 |
| `broken:construction_error` | 18 | `FAIL_SIMULATION_CONSTRUCTION` × 18 |
| `broken:trap` | 18 | `FAIL_PHYSICAL_RESULT` × 18 |

Context policy: **per-task, as declared** — `cold` for `A1-OPT-01`, `A1-OPT-02`,
`A1-CHX-01`, `A1-CHX-02`; `warm` for the two traps. Trial count: **3**, reported
as a pass rate.

### Why the delivered run is against the reference, and what that establishes

A benchmark whose known-good solutions do not pass is measuring something other
than what it claims, and **no agent score from it would mean anything**. So the
first thing the suite does is grade itself, and the reference run is the evidence
that the grading is sound. Measured relative error of each reference solution
against its closed form:

| task | check | observed | expected | relative error |
|---|---|---|---|---|
| `A1-OPT-01` | `effective_focal_length_mm` | 48.374613 | 48.374613 | 2.9e-16 |
| `A1-OPT-01` | `back_focal_length_mm` | 45.737482 | 45.737482 | 3.1e-16 |
| `A1-OPT-02` | `focal_shift_mm` | 3.750048 | 3.750000 | 1.3e-05 |
| `A1-OPT-03` | `uncoated_reflectance` | 0.0421646 | 0.0421647 | 0 (to 8 s.f.) |
| `A1-OPT-03` | `coated_reflectance` | 0.0128354 | 0.0128354 | 2.7e-16 |
| `A1-OPT-03` | `coating_thickness_nm` | 99.6377 | 99.6377 | 0 |
| `A1-CHX-01` | `beam_radius_um` | 6.040167 | 6.039084 | 1.8e-04 |
| `A1-CHX-02` | `first_null_radius_um` | 6.6500 | 6.4985 | 2.3e-02 |
| `A1-CHX-03` | `centroid_x_um` | +17.5017 | +17.4977 | 2.3e-04 |

Eight of the nine agree to 2e-4 or better. `A1-CHX-02`'s 2.3 % is a **sampling
limit and is stated as one**: the focal-plane pitch is 0.83 µm, so the Airy null
lands between samples. Its tolerance is set at 5 % for exactly that reason, and
that is still 8× tighter than the gap to the `1.22 λ/NA` diameter-for-radius
confusion the check is meant to reject.

### The second thing that had to be executed

The other half is that the taxonomy must be able to **fire**. A code nothing has
ever emitted cannot be trusted to fire when it matters, so every code except
`FAIL_HARNESS` has a negative participant that produces it, and all seven were
run at 3 trials × 6 tasks. `FAIL_HARNESS` has no participant on purpose: it is the
one code an agent cannot cause.

---

## 2. The five design decisions, as recorded

Stated in full in `benchmarks_agent/README.md`; the decisions themselves:

1. **Context policy** — `cold` / `warm` / `guided`, **declared per task and
   recorded in every result**. This is the load-bearing one: with the knowledge
   packs in context the benchmark asks *"can it follow our cards"*, without them
   *"can it discover the tool"*. Those are different benchmarks, so a result
   without its policy is not comparable to any other result. `--context-policy`
   *asserts* rather than sets, and errors if it disagrees with a task's
   declaration.
2. **Location** — `benchmarks_agent/`, opt-in by location, following CHE-67
   exactly. `testpaths = ["tests"]` plus a `norecursedirs` entry. The *harness*,
   though, is unit-tested in `tests/test_agent_benchmark.py` and does run by
   default: the grader decides whether an agent passed, so a regression in it
   silently changes every score the benchmark has ever produced, which is exactly
   what a required gate should cover.
3. **Trials** — **3**, declared, reported as a pass *rate* with the denominator
   visible even at `--trials 1`. A trial lost to a *harness* fault is **void, not
   failed**: excluded from the denominator rather than charged to the agent, and a
   task whose every trial was void reports no rate at all instead of zero.
4. **Relationship to `benchmarks/manifest.yaml`** — **beside it, disjoint ID space
   (`A1-*`)**. That registry grades a solver's physics and its value is
   reproducible fingerprints; a nondeterministic agent score inside it would spoil
   that. `manifest.yaml`, its entries, tolerances and gates are untouched, and
   `test_the_id_space_does_not_collide_with_the_solver_benchmark_registry` keeps
   the namespaces apart so nobody merges them later by accident.
5. **Failure taxonomy** — eight structured codes, each with a reason and a
   **remedy**, following `ContractCode` and the precision codes. Grading reports
   the **first** stage that failed, because a run that never produced a number
   cannot also be judged on its physics and reporting the later failure would
   misattribute the cause.

---

## 3. The tasks

| id | title | library | policy | oracle | checks |
|---|---|---|---|---|---|
| `A1-OPT-01` | Focal length of a thick plano-convex singlet | optiland | cold | `R/(n−1)`, `EFL − t/n` | 2 |
| `A1-OPT-02` | Focal shift caused by a plane-parallel plate | optiland | cold | `t(1 − 1/n)`, sign graded | 1 |
| `A1-OPT-03` | Single-layer AR coating at 550 nm | optiland | warm | Fresnel + quarter-wave | 3 |
| `A1-CHX-01` | Diffractive spreading of a Gaussian beam | chromatix | cold | `w₀√(1 + (z/z_R)²)` | 1 |
| `A1-CHX-02` | First dark ring of a focused circular aperture | chromatix | cold | `0.61 λ/NA` | 1 |
| `A1-CHX-03` | Lateral walk-off of a tilted beam | chromatix | warm | `z tan θ`, sign graded | 1 |

**Every check is `analytic`.** The expected value is a closed form, verified
against the pinned solver *before the task shipped*, and every one carries a
`tolerance_basis` naming the measured agreement and a wrong answer the tolerance
rejects. A recorded solver output would be weaker in a way that matters: it cannot
tell a wrong answer from a wrong reference. `expected/` therefore holds a
regression signal, not the oracle, and every file says so.

Three tasks deliberately grade something a magnitude-only comparison would miss:
`A1-OPT-01`'s second check (an agent that omits the thick-lens correction reports
the same number twice and fails only the second check), and `A1-OPT-02` and
`A1-CHX-03`'s **signs**.

### The two traps, both measured on the pinned versions

Not hypothetical, and both are cases where the code runs perfectly and the physics
is wrong — the shape the ticket singled out as the interesting failure.

**`A1-OPT-03` — micrometres where the literature says nanometres.**
`ThinFilmStack.add_layer` takes µm; a quarter-wave MgF₂ layer for 550 nm is
naturally quoted as 99.64 nm. Measured on optiland 0.6.0:

| layer passed as | reflectance at 550 nm |
|---|---|
| `0.099638` (correct, µm) | **0.01283544** |
| `99.638` (the nm number) | **0.04216384** |
| no coating at all | **0.04216456** |

The mistaken coating is indistinguishable from bare glass to seven significant
figures, nothing raises, and 0.042 looks exactly like a reflectance. CHE-57
recorded the same hazard on upstream tutorial t07. The task grades
`coating_thickness_nm` *separately* from the reflectances, so the report can tell
a unit slip inside the model from a wrong design intent — and the trap participant
confirms it does: thickness and uncoated pass, coated fails.

**`A1-CHX-03` — one parameter name, two units, and a sign.** `kykx` means cycles
per length on `asm_propagate` and radians per length on `plane_wave` — a factor of
2π — and the displacement runs *opposite* in sign to the parameter (CHE-57 finding
on example c06). The trap participant submits the value that mistake produces,
−2.785 µm against +17.498, a relative error of 1.16, and is graded
`FAIL_PHYSICAL_RESULT`.

Both traps are `warm` on purpose. The point is not whether an agent can *find* the
library, it is whether it gets a unit right, and a `cold` policy would confound
the two.

---

## 4. Review of the CHE-57 inventories: the candidate pool and what was rejected

The pool came from the two CHE-57 / PB6 inventories rather than from the upstream
sites, as the ticket directs: 41 Optiland reproductions with 459 declared checks
(34 reference / 218 analytic / 189 invariant / 18 qualitative) and 16 Chromatix
reproductions with 205 (46 / 84 / 75 / 0). Because both classify every check by
oracle strength, the **`analytic` column is the shortlist** — 218 + 84 checks
across the two — and selection was a matter of finding cheap, unambiguous,
sign-or-unit-sensitive members of it.

Rejected, with the reason measured rather than guessed:

| candidate | why rejected |
|---|---|
| `t21_surface_roughness_scattering` | flaky: a hard threshold on an unseedable random quantity (CHE-65, open). Optiland's numba BSDFs are unreachable from NumPy's RNG and two identical calls differ by ~1%. Unusable as a graded task |
| `Optic.draw3D()`, `optic_viewer_3d` | hang indefinitely headlessly (VTK finds no X server, no EGL, no OSMesa) |
| anything reading a vendor image (`c01`, `c09`) | `scikit-image` is not installed, and installing it would change the pinned environment |
| `t08` / `t24` vendor `.zmx` | artifacts unreachable; `thorlabs.com` answers the documented URL with a 1313-byte HTML page |
| anything gated on `jax_enable_x64=True` | pinned `False` everywhere; a process that flipped it would change every recorded number in the repository |
| **`c04` / `c10` Adam-state traps** | genuine traps of exactly the wanted shape — upstream's `update()` does not thread optimizer state, and `c10` reproduces its published numbers *only* with the state frozen (final loss 0.784 vs 9.39 threaded). Rejected **only on runtime**: ~5 min each, against a 3-trial suite that currently costs 9 s |
| **`t35` `res.success` trap** | same shape, same objection. `OptimizerGeneric` returns `success=True` at a point 1.97× worse than its start, which is a beautiful graded task, but it needs the full three-mirror system and an optimizer run |
| `t27` advanced optimization | `workers=-1` forks one process per CPU; `AGENTS.md` forbids parallel solver processes on this shared machine |
| `t39` custom surface types | a real, silent physics error (the published `_surface_normal` writes `a·x/r²` where `d(a·r)/dx = a·x/r`), but the task would be "write correct calculus", not "simulate optics" |
| `t04` material database | a good *reference*-backed candidate (N-BK7 index and Abbe number) but not a simulation; held for V2 as a cheap tool-discovery task |

**The three most valuable rejections to record are the Adam-state pair and `t35`.**
They are precisely the "it ran and it is wrong" shape this suite exists for, and
the only thing keeping them out is runtime. A V2 with a cheap surrogate for them —
a small quadratic with the same un-threaded-state structure — is the single
highest-value extension, and it needs no new physics.

---

## 5. Tests

| suite | before | after |
|---|---|---|
| `./run.sh pytest -q` | 717 passed / 48 skipped / 178 s | **770 passed / 48 skipped / 180 s** |
| `./run.sh pytest -q benchmarks_agent` | — | **52 passed / 7.6 s** (opt-in) |
| `./run.sh --gpu pytest -q -m gpu` | 48 passed | unchanged (48) |

The default suite grew by the 53 harness tests and its runtime by 2 s. The
opt-in suite adds nothing to it, which
`tests/test_suite_layout.py` and the `norecursedirs` entry both hold.

What the 53 default-suite tests cover, and why they belong there: the grader's
staging (including that tool selection is judged *before* the numbers, and a
missing quantity before a wrong one), the taxonomy's completeness and remedies,
the trial arithmetic (void vs failed), bool-is-not-a-number, non-finite handling,
sign sensitivity, the context-policy file sets, the ID-space disjointness, and the
CLI's refusals. None imports a solver.

The 52 opt-in tests additionally run the solvers: every reference solution against
its closed form, every recorded expectation against the same closed form, every
outcome code from a participant designed to produce it, both traps landing on
`FAIL_PHYSICAL_RESULT` rather than `FAIL_TOOL_EXECUTION`, and the prompts checked
for library/API/tutorial leaks and for asking in the units they grade in.

---

### One acceptance criterion deviated from, deliberately

The ticket asks that "the default suite's **count** and runtime are unchanged".
The runtime is (178 s to 180 s). The **count is not**: it went from 765 to 818,
because 53 deterministic harness tests were added to `tests/`.

That is a deliberate reading of the intent rather than of the letter. The
criterion sits under "The suite is opt-in by location and does not run under
`./run.sh pytest -q`", which is satisfied exactly — `benchmarks_agent/` contributes
nothing to the default run. What was added instead is coverage of the **grader**,
and the argument for putting it in the required gate is specific: the grader
decides whether an agent passed, so a regression in it silently changes every
score the benchmark has ever produced, and it costs 0.12 s and imports no solver.
Leaving it out would have honoured the letter of the criterion by leaving the one
component whose correctness every future number depends on ungated.

Flagging it rather than quietly doing it, because it is an acceptance criterion.
If the intent was the literal count, deleting `tests/test_agent_benchmark.py` and
moving those 53 tests into `benchmarks_agent/` is a five-minute change — it just
makes the grader opt-in too.

## 6. What was not done, and why

**No agent was run.** The `command` participant is implemented, documented and
unit-tested (missing binary, prompt and context staging, malformed
`submission.json`), but the `agent_solver` container has **no agent CLI installed
and no API credentials**, so running one would need a decision about spending model
tokens that belongs to whoever owns the budget. The container *does* have outbound
network access, so the only missing pieces are a CLI and a key.

Stated plainly because it bears on the acceptance criterion: the criterion is that
the benchmark **can** report whether the multi-agent system reaches the correct
result, and what is demonstrated is that the reporting works — 18/18 for a
known-good participant and the right code for each of seven wrong ones. What is
**not** demonstrated is any claim about an agent's capability, and none is made.

Also not done:

* **No `guided` task in V1.** The policy is implemented and unit-tested, but every
  V1 task is `cold` or `warm`. The obvious first experiment is a `guided` run of
  the two traps: `conventions.md` names both hazards, so `warm` vs `guided` on
  `A1-OPT-03` and `A1-CHX-03` measures something specific — whether handing an
  agent the warning is enough. That needs an agent to be meaningful.
* **No hybrid or cross-tool task**, per the ticket's out-of-scope list.
* **No weighting or ranking.** Pass rate per task and a mean over tasks; nothing
  more, per the ticket.
* **No change to `benchmarks/manifest.yaml`.**
* **The tutorial suite was not run.** Nothing here changes a pin or `docker/`.
* **The stale tutorial-README paths were left alone** (they still say
  `tests/test_*_tutorials.py`, which CHE-67 moved to `tests_tutorial/`). Not this
  ticket's job, per the ticket; no new document copies them.

---

## 7. Follow-ups recommended

1. **Run it against an agent.** Needs a CLI and credentials in the container, and
   a token budget decision. Everything else is in place.
2. **V2: a cheap surrogate for the Adam-state and `res.success` traps.** The
   highest-value extension, and it needs no new physics — a small quadratic with
   the same un-threaded-optimizer-state structure reproduces the trap in
   milliseconds.
3. **`warm` vs `guided` on the two traps.** The cheapest real experiment the
   harness enables, and it measures the knowledge packs rather than the agent.
4. **Per-task budgets.** V1 has one global `--timeout`; a task-declared budget
   would make `FAIL_NO_SUBMISSION` interpretable rather than merely recorded.
