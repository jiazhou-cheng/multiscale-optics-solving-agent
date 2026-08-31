# Repository Instructions — Canonical Shared Context

Repository-wide operating rules. Task goals, acceptance criteria, non-goals, and
ownership live in the linked Linear issue.

**Mission.** A scientifically trustworthy agentic system for multi-scale optical
and nanophotonic simulation: compose existing physics solvers through explicit,
testable couplers, driven by compact versioned knowledge packs. Forward
simulation is the priority; inverse design is a future extension. The supported
core is the Optiland ray model, the Chromatix scalar-wave model, and the
repository ray/wave couplers.``

**Sources of truth**, in precedence order: the Linear issue's scope and
acceptance criteria; this file; executable claims (`src/registry/`,
`src/core/capabilities.py`, tested adapter behavior); the relevant `knowledge/`
pack; the pinned installed package, its versioned docs, and executable probes;
recorded scientific evidence. Never infer capability from roadmap text or old
milestone reports. Existing code is evidence, not automatically the intended
design — surface conflicts instead of silently choosing one source.

## Architecture Boundaries

Full definitions: [`docs/architecture_principles.md`](docs/architecture_principles.md).
Read it before adding a module, a class or a package.

Two concepts. **Representations** are physical state at a declared boundary.
**Operations** consume and produce representations, problems or measurements.
There are four operation *kinds* — `solver`, `coupler`, `physical_operator`,
`measurement` — carried as **descriptor metadata, not four class hierarchies**.

- **representation** — physical state, conventions explicit. Exactly one public ray representation and one scalar-field representation. **PSF is a measurement, never a representation.** Coherence is a stronger contract on the ray representation, not a subtype.
- **solver** — maps a problem into a representation. The only place a backend's API or version pin appears (`solvers/<backend>/`).
- **coupler** — changes *representation*, same physical state at the same boundary. Heavy numerics do not make it an operator.
- **physical operator** — changes the *physical state*. Propagation and surface interaction are operators, never couplers.
- **measurement** — derives an observable from state. Serializability does not make its output a representation.
- **operation descriptor** — one lightweight record used to discover and reason about execution; holds its implementation path as a string and resolves lazily.

Dependency direction, enforced by `scripts/check_dependencies.py`:

```
numerics/ -> nothing        representations/ -> numerics       problems/ -> representations, numerics
operations/ -> numerics     couplers/ -> representations, numerics
solvers/<backend>/ -> problems, representations, numerics (+ its pinned backend)
operators/ -> representations, couplers, numerics             measurements/ -> representations, numerics
planning/ -> operations     runtime/ -> planning, operations, representations
```

`numerics/` is the bottom and imports nothing; `src/core/` is banned because a
package naming no domain accumulates whatever has no other home. `operations/` is
a *sibling* of the implementation packages, never a layer above them. `src/io/`
is banned (it would shadow the stdlib); artifact serialization lives in
`runtime/records.py`.

A class is justified only if it enforces a shared invariant across several
fields, is a public serialized/versioned data model, owns a mutable resource
lifecycle, needs runtime polymorphism across ≥2 *current* implementations, or is
a real plugin boundary. Otherwise: function, module, frozen dataclass, TypedDict,
tuple, Literal, Enum. Counted by `scripts/class_budget.py`; both gates run in the
default suite via `tests/unit/`. Every ticket reports production classes added
and deleted. No base class or placeholder interface exists "so the structure is
visible".

- `knowledge/` is agent-facing knowledge, not executable evidence; `benchmarks/` holds protocols, probes, records, instances, inventory; `archive/` and `tests_tutorial/` are not part of the active surface.
- The tree still contains the pre-rewrite production source (`core/`, `discovery/`, `registry/`, `verification/`, `agent/`, `studies/`, `cli.py`, and the old `couplers/`, `solvers/`, `runtime/`). It is a reference for physics, **never** for architecture, and the new tree must not import it. `docs/rewrite/reference_inventory.md` says what it proves we need; `pre-rewrite-2026-08-30` is the citable tag.

## Execution Environment

`./run.sh` is the only supported entry point for project execution. Run Python,
imports, probes, tests, linters, solver jobs, and benchmarks inside the
`agent_solver` container. Do not run project commands such as `python`, `pip`,
`pytest`, `ruff`, or `mypy` directly on the host; host-side work is limited to
editing files and Git/Linear operations. Do not silently fall back to host
execution — report the environment failure.

- `./run.sh --no-build pytest -q` — reuse the existing image; the default. ~60 s:
  sharded across 8 workers and excluding `slow`, per `addopts` (CHE-140,
  re-measured by CHE-171 — the suite saturates at 8 workers, so 12 was paying
  four extra processes for nothing).
- `./run.sh --rebuild pytest -q` — after Dockerfile/dependency changes.
- `./run.sh pytest -q -m slow` (`make test-slow`) — ~105 s of expensive numerical
  characterization the default gate deselects. Required before merging a change to
  coupler numerics, sampling densities, estimator weights, or a benchmark family,
  oracle or tolerance. Deselected because it is not per-commit, not because it is
  optional.
- `./run.sh pytest -q -m "" -n 0` (`make test-serial`) — everything, unsharded. The
  arbiter when a failure might be a cross-test interaction or a worker artifact.
- `MOA_GPUS=device=6 make test-gpu` — opt-in GPU image and one device, ~75 s for
  66 tests. It overrides `addopts` because the default shards across 8 workers
  and there is one GPU; see `docs/testing/gpu_environment.md`. A bare
  `./run.sh --gpu pytest -q -m gpu` is no longer correct.

## Shared GPU Server Policy

Shared server. **System stability has priority over throughput.** Before any
substantial GPU or memory-intensive run, inspect `nvidia-smi`, `free -h`, and
container/cgroup memory and swap, and confirm the workload fits without swap.

- Prefer GPUs **6 and 7**; avoid **5** unless 6/7 are unavailable. At most **2 GPUs** for this project at once, **1 per workload** unless the task requires multi-GPU. Set visibility through `run.sh`; do not mutate host GPU state.
- **Never use swap as working memory.** Growth in the workload's cgroup swap is a stop condition: terminate and report the resource failure. Prefer chunking, smaller batches, or sequential runs over server-wide memory pressure.
- Do not modify swap, `/etc/fstab`, mounts, drivers, or systemd; never reboot without explicit permission.
- Read-only analysis agents may run in parallel; compute-intensive jobs must obey the limits above and must not overlap unless the task plans for it. No detached compute (`nohup`, `&`, `screen`, `tmux`) without explicit authorization.

## Scientific Non-Negotiables

- When you change a boundary, make its conventions explicit and testable: units, axes, frame, handedness, wavelength, phasor sign, polarization, coherence, normalization, sampling, reference plane. SI internally unless a task defines and tests otherwise. Complex fields are amplitudes, not intensities.
- A solver call succeeding does not prove the approximation is appropriate; a runnable script or plausible-looking plot is not a trustworthy result.
- Never claim a gradient across an untested boundary; cross-framework handoffs are `forward_only` until finite-difference validation passes.
- Never invent fields, metrics, convergence, or provenance. Failed or unsupported paths return structured diagnostics.
- Do not widen a tolerance merely to make a benchmark pass; report the open gate.
- Report what you actually ran. "Not tested" is a valid answer; unverified claimed as verified is not.

## Default Workflow

1. Read the issue; identify acceptance criteria and non-goals.
2. Inspect only the directly relevant code and tests.
3. Reproduce or establish the current failure with the cheapest useful check.
4. Make the smallest change that satisfies the acceptance criteria. Do not opportunistically refactor; keep solver-specific behavior in its adapter; add or update tests for changed logic, contracts, and failure paths.
5. Run targeted verification for the changed contract — the smallest relevant probe or test. No overlapping pytest subsets, no rerunning what already passed.
6. Escalate only when a trigger below fires.
7. Report tests run, tests not run, remaining risks, resource issues, intentional non-goals, and follow-up work.

**Investigation scope.** Do not expand investigation beyond what is necessary to satisfy the acceptance criteria. Do not derive new theory, characterize adjacent behavior, reproduce upstream research, or investigate unrelated scientific questions unless the task requires it or the issue cannot be resolved without it. This bounds investigation, not only edits.

**Cheap probes.** The smallest *and cheapest* thing that answers the immediate question: tiny synthetic inputs, CPU unless GPU behavior is the question, minimal ray/sample counts, one parameter point rather than a sweep, an existing targeted test over a new script. A diagnostic probe should finish in seconds, not minutes.

**Stop condition.** If an exploratory command runs substantially longer than expected without decisive evidence, stop it and reassess rather than letting the investigation grow. A substantial GPU or benchmark run needs a concrete link to an acceptance criterion or an identified regression risk — it is never the tool for understanding a routine implementation issue.

## Escalation Triggers

A routine implementation or debugging task is decided by the acceptance
criteria, the existing contract, and the existing tests. It does not need an
oracle, the full suite, or a reviewer. Escalate when the change actually carries
the risk:

- **Deeper scientific verification** — it introduces or alters a physical claim, a representation boundary, a convention, a numerical algorithm, a benchmark oracle or tolerance, or a new executable capability; or existing contracts and tests cannot settle it. Method: [`docs/scientific_verification.md`](docs/scientific_verification.md). Benchmark family/instance methodology and the B0–B4 categories: [`docs/benchmark_design.md`](docs/benchmark_design.md).
- **Full suite** (`./run.sh pytest -q`) — shared core contracts or boundary artifacts changed; solver/coupler behavior changed in a way that affects multiple callers; dependency, Docker, or environment changed; test collection, fixtures, or repo-wide infrastructure changed; the task requires it; or targeted verification exposed a plausible repo-wide regression. It is not the default gate. Tutorial tests, agent benchmarks, and GPU tests stay on-demand for the tasks that change them. Note that `./run.sh pytest -q` is *not* everything: it deselects `slow` and shards by file. A change to coupler numerics or a benchmark family owes `make test-slow` as well, and a suspected cross-test interaction owes `make test-serial`. See `docs/testing/suite_runtime.md`.
- **Independent review** — the change touches solver adapters or solver API use, couplers or representation boundaries, physical assumptions or conventions, numerical methods/precision/tolerances/convergence, gradient claims, shared core boundary artifacts, executable capability claims, benchmark oracles or acceptance criteria, or substantial GPU/RAM/workload scale. Full trigger list: `docs/scientific_verification.md`; reviewer prompt: `.claude/agents/code-reviewer.md`.

Where scientific risk exists, the agent that wrote the change must not be the only one judging it. Give the reviewer the acceptance criteria, the diff, the tests already run and their results, known uncertainties, and what was left unverified; do not rerun an expensive gate solely so both agents watch it pass.

## PR Contract

State what changed and why, the Linear issue, acceptance criteria checked, tests
and benchmarks run, tests not run, scientific/resource risks, intentional
non-goals, agent involvement, and follow-up work.
