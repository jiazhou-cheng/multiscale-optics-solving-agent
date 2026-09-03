# Repository Instructions — Canonical Shared Context

Repository-wide operating rules for the clean-slate rewrite. Task goals,
acceptance criteria, non-goals, and ownership live in the linked Linear issue.

**Mission.** Build a scientifically trustworthy agentic system for multi-scale
optical and nanophotonic simulation by composing existing physics solvers through
explicit, testable boundaries and compact versioned knowledge packs. Forward
simulation is the first priority; inverse design is a later extension.

The initial integration targets are the Optiland ray model, the Chromatix
scalar-wave model, and repository-owned ray/wave couplers. A target is not a
supported capability until its adapter, contract, and verification have landed.

## Sources of Truth

Use these in precedence order:

1. the Linear issue's explicit scope, acceptance criteria, and non-goals;
2. this file and [`docs/architecture_principles.md`](docs/architecture_principles.md);
3. executable contracts and tests that have actually landed in the new tree;
4. the relevant `knowledge/` pack;
5. the pinned installed package, its versioned documentation, and executable
   probes;
6. recorded scientific evidence.

Current code is evidence of what is implemented, not automatic authority for what
the architecture should become. Historical code is reference material only.
Surface conflicts instead of silently choosing whichever source is most
convenient.

Do not infer capability from roadmap text, old milestone reports, package names,
or an unverified implementation stub.

## Clean-Slate Source Rule

`src/` is being rebuilt incrementally from a clean slate.

- A package does not need to exist because a design document names it.
- Create a package only when the current issue has real production code to place
  in it.
- Do not add empty scaffolding, placeholder interfaces, speculative base classes,
  compatibility wrappers, or registries "for later."
- Do not copy the pre-rewrite package tree into the new `src/`.
- Historical implementations may be read through Git history/tags or explicit
  reference documentation for physics, conventions, failure modes, and test
  ideas; they are never an architecture template and the new tree must not import
  them.
- If the issue requires a boundary that the target architecture has not yet
  covered, update the architecture deliberately rather than recreating an old
  package to make the change fit.

A missing package, checker, registry, or runtime component is not itself a defect
during bootstrap. Claim only what has landed.

## Architecture Boundaries

Full definitions: [`docs/architecture_principles.md`](docs/architecture_principles.md).
Read it before adding a production package, module, class, or cross-package
import.

Two concepts: **representations** are physical state at a declared boundary;
**operations** consume and produce representations, problems, or measurements —
except a **source**, which produces one without consuming any.

There are four *primitive* operation kinds — `source`, `coupler`,
`physical_operator`, and `measurement` — represented as **descriptor metadata, not
four class hierarchies**. The `kind` field has a fifth value, `composed`, which is
not a primitive and names a fusion of them. A **backend is not one of them**: which third-party library
executes an operation is a separate axis from what the operation does to physical
state, and it is a separate descriptor field (`backend`). CHE-224 (R15.1)
separated the two, replacing a `solver` kind with `source`; see
`docs/architecture_principles.md` §2.

The four are **primitive**. One callable may fuse several of them; it carries
`kind=composed` and declares the ordered stages in `OperationDescriptor.composes`.
Where it *leaves* the state is `composes[-1]`, read off the `terminal_stage`
property. CHE-237 (R03.7) decided this and reversed CHE-225 (R15.2), under which
`kind` named the terminal stage and a composite therefore borrowed the kind of its
last stage. `composes` is non-`None` **if and only if** `kind is composed`, so the
two cannot disagree; every stage is still primitive, and `SO_`/`SOM_` remain id
prefixes rather than kinds. There are three composites today —
`SO_RAY_LAUNCH_TRACE`, which initializes rays and then refracts them through every
surface so neither `source` nor `physical_operator` alone is a true claim about it,
plus `SOM_SPOT_DIAGRAM` and `SOM_PSF`, which then reduce the result to an
observable. A composition is **not** a pipeline description and nothing can execute
a stage of one.

- **representation** — physical state with explicit conventions. The initial public target is one ray representation and one scalar-field representation. PSF is a measurement, not a representation. Coherence is a stronger contract by default, not a subtype.
- **backend** — an adapter package that **provides** operations of the other kinds and is not itself an operation kind. It owns external-library API, compatibility, and version-specific behavior, and `backends/<backend>/` is the only place permitted to import that library. A backend answers *who executes*; a kind answers *what happens to physical state*, and an operation has exactly one of each. Package location follows the provider; kind is declared in the catalog — so a backend-provided measurement lives in `backends/<backend>/` and needs no `measurements/ -> backends/` edge.
- **source** — owns the physically meaningful **initialization** of a representation. A representation defines the structure and conventions of physical state at a declared boundary; a source defines how that state is initialized from physical source parameters — a plane-wave source initializes a `ScalarField`'s complex amplitude and phase from its wavelength, propagation direction, amplitude, sampling grid and reference surface. **A source does not consume an existing physical representation; it creates the initial state of one.** But **a source may be described without an optical system and a ray launch may not**: the launch positions and directions of a source into a system depend on the stop, the entrance pupil, the surfaces before the stop, the object distance, the field, the backend's pupil map and the ray aimer, so ray launch is a `backends/<backend>/` operation taking the constructed system as a required argument, and `sources/` produces no system-launch `RayBundle`. CHE-219 (R05.8) decided this; see `docs/architecture_principles.md` §2. It registers as `source`-kind, and what separates `sources/` from `backends/<backend>/` is not the kind — both provide `source`-kind operations — but the provider: a source in `sources/` has no external backend, so its descriptor carries `backend=None`.
- **coupler** — changes *representation* while preserving the same physical state at the same boundary. Heavy numerics do not make it an operator.
- **physical operator** — changes physical state. Propagation and surface interactions are operators, not couplers.
- **measurement** — derives an observable from state.
- **composite operation** — one callable that fuses more than one primitive stage, declaring them in order. `kind` is `composed` and `terminal_stage` is where the state ends up; `composes` is `None` — not `()` — for everything that fuses nothing. The two rules that ask where an operation leaves the state, the observable-producer check and the `kind` query in `find`, read `terminal_stage`. A composite exists only where a single primitive kind would be a *false* claim, not merely a simplification — and it is not a route, a plan or a pipeline description. `O_DIFFRACTIVE_SURFACE` is internally coupler → operator → coupler and deliberately declares no composition, because its representation types do not change at its ports; whether it should is an open question, not a defect.
- **operation descriptor** — lightweight discovery/execution metadata; the target design resolves implementation paths lazily.

The target dependency allowlist, **for packages that exist**, is:

```
numerics/            -> (nothing in the project)
representations/     -> numerics
problems/            -> representations, numerics
operations/          -> numerics
backends/<backend>/  -> problems, representations, numerics (+ its backend)
sources/             -> problems, representations, numerics
couplers/            -> representations, numerics
operators/           -> representations, couplers, numerics
measurements/        -> representations, numerics
planning/            -> operations
runtime/             -> planning, operations, representations
```

Do not create these packages just to realize the diagram. When a package lands,
add or extend executable dependency checks for the part of the graph that now
exists. `numerics/` is intended to be the bottom, not a generic `core/`.
`operations/` is a sibling of implementations, not a layer that imports them.
Representations remain backend-neutral. Do not create `src/io/` under a flat
source layout.

`sources/` is **representation-independent as a package and
representation-explicit at every public operation**: it may initialize a
`ScalarField` or any other landed representation whose state is determined by
source parameters alone, but each public operation's return representation must be
unambiguous in the signature and in its descriptor. No subpackage per representation, and no constructor whose return
representation depends on its arguments. `sources/` is upstream of everything that
consumes state, so it may not import `backends/`, `couplers/`, `operators/` or
`measurements/`.

A class must justify itself by a shared invariant, versioned public data model,
mutable resource lifecycle, runtime polymorphism across at least two *current*
implementations, or a real runtime/plugin boundary. Otherwise prefer a function,
module, frozen dataclass, TypedDict, tuple, Literal, or Enum.

The clean rewrite has no inherited project-wide class-count ceiling. If a class
budget is introduced later, derive it from the new tree and make it a visibility
gate, not a substitute for design review.

## Execution Environment

Use the execution path that has actually landed in the repository and is named by
the current issue or repository tooling. Do not claim a command is canonical
before that tool exists.

Once `./run.sh` is present and designated as the repository execution wrapper,
run project Python, imports, probes, tests, linters, solver jobs, and benchmarks
through it rather than directly on the host. If the execution harness is not yet
part of the bootstrap state, use the issue's explicit bootstrap procedure and
report what you ran.

Never silently fall back from a required containerized/GPU environment to a
different environment. Report the mismatch.

## Shared GPU Server Policy

When work is run on the shared GPU server, **system stability has priority over
throughput.**

- Inspect GPU and host/container memory before substantial GPU or memory-intensive work.
- Never use swap as working memory; workload swap growth is a stop condition.
- Prefer a single GPU per workload unless the issue explicitly requires multi-GPU.
- Do not modify host swap, mounts, drivers, system services, or reboot the machine without explicit permission.
- Do not leave detached compute (`nohup`, background `&`, `screen`, `tmux`) without explicit authorization.
- Read-only analysis may run in parallel; compute-heavy jobs must respect the server's current resource policy and must not overlap merely for speed.

Environment-specific preferred GPU IDs, worker counts, runtime estimates, and
test counts belong in current operations/testing documentation, not in this
canonical architecture context, because they change independently of source
design.

## Scientific Non-Negotiables

- When a physical boundary lands or changes, make its conventions explicit and testable: units, axes, frame, handedness, wavelength, phasor sign, polarization, coherence, normalization, sampling, and reference plane. Use SI internally unless a scoped contract explicitly defines and tests another convention. Complex fields are amplitudes, not intensities.
- A solver call succeeding does not prove the approximation is appropriate. A runnable script or plausible-looking plot is not by itself a trustworthy result.
- Never claim a gradient across an untested boundary. Cross-framework handoffs are `forward_only` until an independent finite-difference or equivalent validation supports the claim.
- Never invent fields, metrics, convergence, or provenance. Failed or unsupported paths return explicit diagnostics once that vocabulary exists; before it exists, fail plainly rather than fabricating a structured contract.
- Do not widen a tolerance merely to make a benchmark pass; report the open gate and justify tolerance changes scientifically.
- Report what you actually ran. "Not tested" is valid; unverified claimed as verified is not.
- Repository numerical code must not be the sole correctness oracle for the same numerical code. Shared-code characterization is useful evidence, not an independent correctness gate.

## Default Workflow

1. Read the issue and identify acceptance criteria and non-goals.
2. Inspect the smallest relevant current surface. During bootstrap this may be
   only docs, configuration, and a nearly empty `src/`.
3. Establish the current state with the cheapest useful check: reproduce a
   failure when one exists, or demonstrate the missing contract/package when the
   task is additive.
4. Make the smallest change that satisfies the acceptance criteria. Do not
   opportunistically recreate old architecture or scaffold unrelated future work.
5. Add or update tests for logic, contracts, conventions, dependency boundaries,
   and failure paths introduced by the change.
6. Run the smallest relevant verification that has actually landed. Do not rerun
   overlapping subsets merely for reassurance.
7. Escalate only when a trigger below fires.
8. Report tests/probes run, checks not available or not run, remaining scientific
   or resource risks, intentional non-goals, and follow-up work.

**Investigation scope.** Do not expand beyond what is necessary to satisfy the
issue. Do not derive new theory, characterize adjacent behavior, reproduce
upstream research, or investigate unrelated scientific questions unless the task
requires it or cannot be resolved without it.

**Cheap probes.** Prefer the smallest and cheapest thing that answers the
immediate question: tiny synthetic inputs, CPU unless GPU behavior is the
question, minimal sample counts, one parameter point rather than a sweep, and an
existing targeted test over a new ad-hoc script.

**Stop condition.** If an exploratory command runs substantially longer than
expected without decisive evidence, stop it and reassess. A substantial GPU run
or benchmark needs a concrete link to an acceptance criterion or identified
regression risk.

## Escalation Triggers

Routine implementation/debugging is decided by the acceptance criteria, the
contracts that currently exist, and the smallest relevant tests. Do not require
infrastructure that has not landed merely because the target architecture names
it.

Escalate when the change carries the corresponding risk:

- **Deeper scientific verification** — introduces or alters a physical claim, representation boundary, convention, numerical algorithm, benchmark oracle or tolerance, gradient claim, or executable scientific capability; or existing evidence cannot settle the question.
- **Broader/full-suite verification** — changes shared contracts or boundary artifacts, affects multiple callers, changes dependencies/environment/test infrastructure, or targeted verification exposes plausible repo-wide regression. If a canonical full-suite command has not landed yet, run all available relevant gates and report the missing infrastructure instead of inventing one.
- **Independent review** — touches solver API use, couplers/representation boundaries, physical assumptions or conventions, numerical methods/precision/tolerances/convergence, gradient claims, shared boundary artifacts, executable capability claims, benchmark oracles/acceptance criteria, or substantial GPU/RAM workload scale.

Where scientific risk exists, the author of the change must not be the only person
judging it. Give the reviewer the acceptance criteria, diff, tests/probes already
run and their results, known uncertainties, and what remains unverified.

## PR Contract

State what changed and why, the Linear issue, acceptance criteria checked, tests
or probes run, checks not run or not yet available, scientific/resource risks,
intentional non-goals, agent/reviewer involvement where relevant, and follow-up
work.

Do not claim that a target package, enforcement script, benchmark gate, registry,
or capability exists until it is present and verified in the new tree.
