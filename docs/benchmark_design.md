# Benchmark Design

Load this when a task authors, changes, or retires a benchmark family, instance,
oracle, tolerance, metric, or negative control — or when it changes what a
benchmark *claims*. Routine implementation and debugging tasks do not need it.

`AGENTS.md` holds the repository-wide rules; this file holds the methodology.

## The substrate

The benchmark system is a **shared scientific verification substrate**, not a
list of tasks:

```
BenchmarkFamily / BenchmarkInstance
  -> GraphExecutor    emits an ExecutionRecord    (what happened)
  -> verify(...)      emits a VerificationResult  (what it means)
  -> fixed evaluation, future generated evaluation, agent scoring
```

A **family** is a physical question with a declared parameter space, an oracle
and its independence, executable validity predicates, metrics, tolerances with
bases, and negative controls. An **instance** is one point in that space with a
stable fingerprint.

## The five categories

Categories are defined by *what may decide them*:

| | what it asks | what may decide it |
| -- | -- | -- |
| **B0** | contract and recovery: does the component refuse what it cannot do in a way a caller can act on — including silent hazards where the contract is `ok` and the physics is wrong | declared capability, structured refusal codes |
| **B1** | primitive correctness inside one representation | analytic closed form or invariant |
| **B2** | a representation transition: ray to wave, wave to ray, patch to global | exactness limit, conservation, convergence |
| **B3** | a composed chain whose correctness is still decidable | analytic form, a genuinely independent route, or intermediate invariant evidence |
| **B4** | characterization: convergence, cost, variance, reproducibility, cross-route consistency | **nothing — B4 never gates, by construction** |

## The layer axis

Orthogonal to the categories. B0–B4 classify **what may decide correctness**;
the layer classifies **what scientific question is being asked**. Both are
required on every family, and neither implies the other — `B3-PSF-SINGLET` and
`B3-DUALROUTE` share a category, and one is a statement about an optical system
while the other compares two numerical realizations of the same one.

| layer | what is being claimed | examples |
| -- | -- | -- |
| **A — qualification** | is this operator the thing it claims to be? Conventions: OPL, phasor sign, projection factor, power accounting, off-axis reference, round trips, estimator exactness and unbiasedness | all of B0 and B1; `B2-R2W-EXACT`, `B2-W2R-STOCH`, `B2-EQUIV`, `B2-ROUNDTRIP` |
| **B — numerical realization and validity** | does a choice that should *not* move the answer stay inside its declared budget, and where does it stop? Convergence, oversampling, grid parity, patch granularity, cost | `B2-R2W-ROUTE`, `B3-DUALROUTE`, `B4-COST`, `B4-DUALROUTE-AGREEMENT` |
| **C — system** | is a physically meaningful end-to-end optical system modelled correctly? | `B3-PSF-SINGLET`, `B3-DEMO2`, `B4-DEMO3` |

`BenchmarkLayer` is required on `BenchmarkFamily` and has **no default**. That
is the point: a family cannot be authored without stating whether it qualifies a
primitive, characterizes a numerical realization, or claims a system.

Three consistency rules follow, and each is enforced in
`BenchmarkFamily.__post_init__` rather than written down here and hoped for:

- a **layer-C** family must declare a `topology` of at least three stages and at
  least two distinct metrics. Three stages because two is a representation
  transition, which is layer A or B; two metrics because **no system collapses
  to a single threshold** — one scalar is blind to whatever it is blind to, and
  NCC alone has already certified a route that lost 1.7% of the power. A
  topology declared at layer A or B is refused as a contradiction;
- a **layer-B** family must declare a refinement dimension — a parameter with
  `refines_toward` set. Characterizing a numerical realization with no declared
  direction of refinement is a comparison, not a convergence study;
- a **layer-A** family must declare at least one negative control. A convention
  check that has only ever been shown to agree has not been shown to be able to
  disagree.

### Numerical realization choices are never a family axis

`RAMP_SUM` and `KSPACE_SPLAT`, FFT padding, oversampling, patch size, sampling
density and streaming strategy are **numerical realization choices**. They are
characterized at layer B, against a declared error budget, and they do not
define a physics benchmark. "Exact route versus fast route" is not a benchmark
family axis: the exactness question is layer A and belongs to the operator, the
route question is layer B and belongs to the realization.

`B2-R2W-EXACT` and `B2-R2W-ROUTE` still carry the retired vocabulary in their
identifiers. Once they carry `QUALIFICATION` and `NUMERICAL` respectively the
ontological problem is gone and the names are cosmetic, so **they are not
renamed**: a family identifier change stales every committed instance record for
that family, and renaming without regenerating is not an option. Nine records —
`B2-R2W-EXACT-01`, the four `ONNODE` and four `OFFNODE` route records — would be
regenerated to express a taxonomy change the layer field already expresses.
Recorded here as a decision rather than left to drift.

### Where layer-C artifacts live

`benchmarks/systems/` is the home for layer-C specs, drivers and records
authored from M2.7 onward. Existing evidence is **re-homed by classification,
not moved on disk**: `B3-PSF-SINGLET`, `B3-DEMO2` and `B4-DEMO3` became layer C
by acquiring a field, and physically relocating committed records to express
that would invalidate every fingerprint for no scientific gain. The generated
layer view in `benchmarks/INVENTORY.md` and
`benchmarks/validation/coverage_matrix.md` is what makes the grouping legible
without reading source.

## The four separations

Each is enforced by code rather than convention.

1. **Execution is not correctness.** The executor records what happened; the
   verifier decides what it means. `ExecutionRecord` carries no metric,
   tolerance or verdict, and `VerificationResult` has no pass boolean and no
   score.
2. **Validation is not characterization.** A `SHARES_CODE` or `CROSS_ROUTE`
   oracle forces category B4, and a B4 family cannot carry a gating tolerance.
3. **Executed successfully is not physically correct.** Silent wrong answers are
   first-class benchmark targets.
4. **Evidence outlives its wrapper.** Retiring the old task layer preserved its
   oracles, tolerance derivations, measured traps and exclusion reasons rather
   than the wrappers that ran them; `benchmarks/inventory.yaml` records where
   each went.

## What a family must declare

Scientific evidence is expressed as a family, not as a script with a hard-coded
parameter set. A family declares, as applicable:

- the physical question and approximation being tested, and which components it
  speaks about;
- its parameters, split into `PhysicalParameter` (moves the correct answer),
  `NumericalParameter` (moves achieved accuracy and cost, not the answer),
  `RepresentationParameter` and `ExecutionParameter` (neither, beyond a declared
  budget);
- executable `ValidityPredicate`s with a normalized signed margin — positive
  inside, zero at the boundary, negative outside — aggregating to `INSIDE` /
  `NEAR_BOUNDARY` / `OUTSIDE` / `FAR_OUTSIDE`;
- the oracle (kind, independence, callable);
- metrics, each stating what it is **blind to**;
- tolerances, each with its basis and whether that basis may gate;
- invariants and negative controls, including any control known to fire
  backwards;
- the stochastic policy — a stochastic family owes exactness limit,
  unbiasedness, convergence exponent and variance, and requires more than one
  seed;
- the execution policy (allowed devices and dtypes, runtime and memory
  envelope), canonical instances, the sampler or a recorded reason for having
  none, and the provenance rule.

## Rules that follow from that

- A family whose `NumericalParameter` moves its oracle value has a defect. The
  parameter split is what makes that testable, and it is why measuring how much
  a parameter that should not change the answer does is itself a benchmark.
- Every reported number carries an uncertainty and a basis for it. A value with
  no error bar is a schema violation, not a pass.
- Every successful round trip needs a deliberately broken twin that fails, and a
  gate a known-wrong twin can pass is reported as untrustworthy rather than
  green.
- For composed families, test both the components and the handoff — a correct
  final image can hide an incorrect intermediate convention.
- Do not widen a tolerance merely to make a benchmark pass.
- Do not treat an old milestone benchmark as canonical simply because it exists.
  If its oracle, scope, or gate is no longer scientifically trusted, replace or
  retire it explicitly — and preserve the evidence separately from the wrapper
  that ran it.

## The agentic benchmark

The higher-level framework built on top: the agent is given problems drawn from
the B0–B4 families and must select the model(s), load the relevant knowledge,
configure and execute the graph, and interpret and verify the result. It
consumes `VerificationResult`; it does not grade physics itself, and
`src/verification/` imports nothing from `src/agent/`.

For agent benchmarks, grade the reasoning-relevant behavior: model selection,
knowledge use, graph construction, parameterization, error handling, and
interpretation — not only whether the final number happens to match.

## Where the files are

See [`benchmarks/README.md`](../benchmarks/README.md) for the layout of
protocols, probes, records, instances, reports, and the artifact inventory.
