# Scientific Verification

Load this when a task actually carries scientific risk — see the escalation
triggers in `AGENTS.md`. Routine implementation and debugging tasks are decided
by the issue acceptance criteria, the existing contract, and the existing tests;
they do not need to construct an oracle.

`AGENTS.md` holds the invariants that always apply. This file holds the method
for the cases that need more.

## When an oracle is required

Build or identify a scientific oracle when the change:

- introduces or changes a physical or scientific claim;
- changes a solver or coupler representation boundary;
- changes physical conventions or numerical algorithms;
- changes benchmark oracles, tolerances, or acceptance criteria;
- introduces a new executable capability; or
- cannot be resolved correctly from the existing contract and tests.

Otherwise, the existing tests are the verification. Adding an oracle to a task
that did not need one is scope expansion, not rigour.

## Choosing an oracle

In descending order of trust:

1. **Analytic closed form.** A textbook result for the same configuration.
   Primary whenever one exists.
2. **Conservation law or invariant.** Energy, flux, étendue, reciprocity,
   symmetry. Weaker than a closed form — it constrains rather than determines —
   but it is independent.
3. **Convergence study.** The right exponent approached at the right rate.
   Establishes that a discretization is doing what it claims.
4. **Independent implementation.** A second code path that shares no
   implementation with the one under test.

Never let our own numerical code decide a correctness gate for itself. An oracle
that shares code with the thing it checks is `SHARES_CODE`, which is
characterization, not validation — it is diagnostic-only and cannot gate. The
same holds for cross-route agreement between two of our own routes.

## Conventions to make explicit and testable

At every model boundary, state and test: units, axes, coordinate frame,
handedness, wavelength/frequency, phasor sign, polarization basis, coherence
model, normalization, sampling, and reference plane.

- Use SI internally unless a task explicitly defines and tests another
  convention.
- Complex fields are amplitudes, not intensities.
- The core boundary artifacts are `RayBundle`, `WavefrontSamples`,
  `ComplexField`, and `PSF`. Add new universal types only when a real
  cross-model contract requires them.

## Claims that need evidence before they are made

- A solver API call succeeding does not prove the physical approximation is
  appropriate. A runnable script or a visually plausible result is not, by
  itself, a scientifically trustworthy result.
- Never claim a gradient across an untested boundary. Cross-framework handoffs
  are `forward_only` by default until a derivative contract and finite-difference
  validation pass.
- Failed or unsupported solvers return structured diagnostics. Never invent
  fields, metrics, convergence, or provenance.
- Do not widen a tolerance merely to make something pass. Report the open gate
  instead.
- Record uncertainty explicitly. A reported number without a basis for its error
  is incomplete.

## Probe design

A probe exists to answer one question. Make it the cheapest thing that answers
it:

- tiny synthetic inputs, not realistic workloads;
- CPU unless the question is specifically about GPU behavior;
- minimal ray, sample, and grid counts;
- one parameter point, not a sweep, unless the question *is* the trend;
- an existing targeted test in preference to a new exploratory script.

A diagnostic probe should normally finish in seconds. If one is taking
substantially longer than expected without producing decisive evidence, stop it
and reassess rather than letting the investigation grow.

A sweep, convergence study, or GPU run is a *deliverable* — it belongs to an
acceptance criterion or an identified regression risk, and it is planned as
such. It is not an instrument for understanding a routine implementation issue.

Probe records under `benchmarks/probes/records/` are **provenance, never
oracles**.

## Full independent-review trigger list

Independent review is **required** when the change affects any of:

- external solver adapter behavior or solver API use;
- couplers or representation-changing boundaries;
- physical assumptions or conventions;
- units, coordinate systems, wavelength/frequency handling,
  polarization/coherence, normalization, sampling, or reference planes, when
  those contracts change;
- numerical algorithms, interpolation, quadrature, precision, tolerances, or
  convergence behavior;
- gradients, autodiff, differentiability, or cross-framework derivative claims;
- shared core scientific boundary artifacts;
- executable model/coupler capability claims;
- benchmark oracles, scientific tolerances, or acceptance criteria;
- GPU/RAM allocation, batching, workload scale, or otherwise substantial
  resource behavior.

It is normally **optional** for documentation-only changes that do not alter a
scientific claim, formatting and comments, isolated developer tooling changes,
test cleanup that alters no scientific oracle or tolerance, and narrow
implementation changes with no solver/API/physics/boundary/numerical/resource
contract impact. Routine tasks do not inherit the full scientific-review
workflow.

## When the full suite is *not* the gate

`./run.sh pytest -q` is normally unnecessary for documentation-only changes that
do not alter a scientific claim, narrowly isolated implementation fixes, local
validation or failure-path fixes, registry metadata changes that introduce no new
executable scientific capability, and test cleanup that changes no oracle,
tolerance, or contract. Targeted verification reduces cost, not standards: a
change that runs, or produces visually plausible output, is still unverified
until the changed contract itself has evidence.

## Independent review depth

The reviewer scopes its depth to the risk domains the diff actually touches and
stops once the affected acceptance criteria and changed-code risks have
sufficient evidence. It consumes the implementation agent's recorded evidence
rather than recreating it, and may run one narrow read-only check through
`./run.sh` when a specific review question cannot be settled otherwise. It does
not duplicate passing runs, or rerun expensive tutorial/GPU/full benchmark
suites unless the task specifically requires them.

Findings are classified as **must fix before merge**, **should fix soon**, or
**safe to merge / no blocker**.

The detailed reviewer prompt lives in `.claude/agents/code-reviewer.md`.

## Related

- [`docs/benchmark_design.md`](benchmark_design.md) — family/instance
  methodology, the B0–B4 categories, and what a family must declare.
