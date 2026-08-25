# Repository Instructions — Canonical Shared Context

This is the repository-wide instruction source for coding and research agents. Task-specific goals, acceptance criteria, non-goals, and ownership live in the linked Linear issue or explicit task prompt.

## Mission

Build a **scientifically trustworthy agentic system for multi-scale optical and nanophotonic simulation**. The system should compose existing physics solvers through explicit, testable couplers and use compact, versioned knowledge packs to choose, configure, and verify those components.

Forward simulation is the current priority. Inverse design is a future extension and must not be treated as supported across a boundary until the relevant gradients are explicitly verified.

A runnable script or visually plausible result is not, by itself, a scientifically trustworthy result.

## System Model

Each supported physics model has four parts: an external solver/API, a narrow repository adapter, an agent-facing knowledge pack, and independent verification evidence and benchmarks. Each coupler has the same separation: implementation, explicit boundary contracts, agent-facing knowledge, and independent verification.

Knowledge packs live under `knowledge/` and contain the compact information an agent needs to use a component correctly — typically `card.yaml`, `conventions.md`, `usage_notes.md`, `api_minimal_examples.md`, `failure_guide.md`, and coupler theory where needed. Do not duplicate those details in this file. Load the relevant pack only when a task uses that solver or coupler.

The currently supported core is the Optiland geometric-ray model, the Chromatix scalar-wave model, and the repository ray/wave couplers. Treat `src/registry/` and executable capability declarations as the source of truth for what is actually supported; do not infer capability from roadmap text or old milestone reports.

## Current Phase

The project is now in a **benchmark redesign + codebase hardening + agent integration** phase.

The benchmark system is a **shared scientific verification substrate**, not a list of tasks: `BenchmarkFamily`/`BenchmarkInstance` -> `GraphExecutor` emits an `ExecutionRecord` -> `verify(family, instance, record)` emits a `VerificationResult` -> fixed evaluation, future generated evaluation and agent scoring all consume that result.

A **family** is a physical question with a declared parameter space, an oracle and its independence, executable validity predicates, metrics, tolerances with bases, and negative controls; an **instance** is one point in that space with a stable fingerprint. Five categories, by what may decide them:

| | what it asks | what may decide it |
| -- | -- | -- |
| **B0** | contract and recovery: does the component refuse what it cannot do in a way a caller can act on — including silent hazards where the contract is `ok` and the physics is wrong | declared capability, structured refusal codes |
| **B1** | primitive correctness inside one representation | analytic closed form or invariant |
| **B2** | a representation transition: ray to wave, wave to ray, patch to global | exactness limit, conservation, convergence |
| **B3** | a composed chain whose correctness is still decidable | analytic form, a genuinely independent route, or intermediate invariant evidence |
| **B4** | characterization: convergence, cost, variance, reproducibility, cross-route consistency | **nothing — B4 never gates, by construction** |

Four separations hold throughout, each enforced by code rather than convention:

1. **Execution is not correctness.** The executor records what happened; the verifier decides what it means. `ExecutionRecord` carries no metric, tolerance or verdict, and `VerificationResult` has no pass boolean and no score.
2. **Validation is not characterization.** A `SHARES_CODE` or `CROSS_ROUTE` oracle forces category B4, and a B4 family cannot carry a gating tolerance.
3. **Executed successfully is not physically correct.** Silent wrong answers are first-class benchmark targets, and retiring the old task layer preserved its oracles, tolerance derivations, measured traps and exclusion reasons rather than the wrappers that ran them; `benchmarks/inventory.yaml` records where each went.

The **agentic benchmark** is the higher-level framework built on top: the agent is given problems drawn from the B0–B4 families and must select the model(s), load the relevant knowledge, configure and execute the graph, and interpret and verify the result. It consumes `VerificationResult`; it does not grade physics itself, and `src/verification/` imports nothing from `src/agent/`.

## Sources of Truth

Use the following precedence:

- Repository-wide operating rules: this file.
- Task scope and acceptance criteria: the linked Linear issue or explicit task prompt.
- Executable model/coupler claims: `src/registry/`, `src/core/capabilities.py`, and tested adapter behavior.
- Solver/coupler usage knowledge: the relevant pack under `knowledge/`.
- Package API truth: the pinned installed package, official versioned documentation, and executable probes.
- Scientific verification: analytic cases, conservation laws, convergence studies, or independent implementations and their recorded benchmark evidence.

Existing code is evidence, not automatically the intended design. Surface conflicts instead of silently choosing one source.

## Context Loading

Keep startup context small.

- Start with this file, the task/Linear issue, and the directly relevant code/tests.
- Read only the registry entries and knowledge packs for components used by the task.
- Load minimal API examples and failure guides after the component path is known.
- Do not recursively read all of `docs/`, `knowledge/`, `benchmarks/`, or `archive/`.
- Historical reports and papers are reference material unless the task explicitly depends on them.

## Architecture Boundaries

- `src/core/`: shared typed artifacts, graph contracts, execution status, precision/device policy.
- `src/solvers/`: narrow adapters around external solver APIs. External solver imports stay here.
- `src/couplers/`: representation-changing physics between models. Couplers must not depend on solver-specific implementation details unless the contract explicitly requires it.
- `src/registry/`: declarations of supported models and couplers; it does not execute them.
- `src/verification/`: independent scientific oracles, measurements, the benchmark family substrate (`families/`), and the physics verifier. Imports nothing from `src/agent/`.
- `src/agent/`: agent execution/benchmark infrastructure, not solver physics.
- `knowledge/`: compact agent-facing knowledge, not executable evidence or tutorial test code.
- `benchmarks/`: benchmark protocols, probes, records, instances, and the artifact inventory.
- `tests_tutorial/`: on-demand upstream tutorial reproductions.
- `archive/`: historical and non-runnable material.

The core boundary artifacts remain `RayBundle`, `WavefrontSamples`, `ComplexField`, and `PSF`. Add new universal types only when a real cross-model contract requires them.

## Scientific Non-Negotiables

At every model boundary, make the relevant conventions explicit and testable: units, axes, coordinate frame, handedness, wavelength/frequency, phasor sign, polarization basis, coherence model, normalization, sampling, and reference plane.

- Use SI internally unless a task explicitly defines and tests another convention.
- Complex fields are amplitudes, not intensities.
- A solver API call succeeding does not prove the physical approximation is appropriate.
- Never claim a gradient across an untested boundary.
- Cross-framework handoffs are `forward_only` by default until a derivative contract and finite-difference validation pass.
- Prefer an analytic oracle, conservation law, convergence study, or independent implementation over self-comparison.
- Do not widen a tolerance merely to make a benchmark pass.
- Failed or unsupported solvers return structured diagnostics; never invent fields, metrics, convergence, or provenance.

## Benchmark Requirements

Scientific evidence is expressed as a **family**, not as a script with a hard-coded parameter set. A family declares, as applicable:

- the physical question and approximation being tested, and which components it speaks about; its parameters, split into `PhysicalParameter` (moves the correct answer), `NumericalParameter` (moves achieved accuracy and cost, not the answer), `RepresentationParameter` and `ExecutionParameter` (neither, beyond a declared budget);
- executable `ValidityPredicate`s with a normalized signed margin — positive inside, zero at the boundary, negative outside — aggregating to `INSIDE` / `NEAR_BOUNDARY` / `OUTSIDE` / `FAR_OUTSIDE`; the oracle (kind, independence, callable); metrics, each stating what it is **blind to**; tolerances, each with its basis and whether that basis may gate; invariants and negative controls, including any control known to fire backwards;
- the stochastic policy — a stochastic family owes exactness limit, unbiasedness, convergence exponent and variance, and requires more than one seed;
- the execution policy (allowed devices and dtypes, runtime and memory envelope), canonical instances, the sampler or a recorded reason for having none, and the provenance rule.

A family whose `NumericalParameter` moves its oracle value has a defect. The parameter split is what makes that testable, and it is why measuring how much a parameter that should not change the answer does is itself a benchmark. Every reported number carries an uncertainty and a basis for it; a value with no error bar is a schema violation, not a pass. Every successful round trip needs a deliberately broken twin that fails, and a gate a known-wrong twin can pass is reported as untrustworthy rather than green. For composed families, test both the components and the handoff — a correct final image can hide an incorrect intermediate convention. For agent benchmarks, grade the reasoning-relevant behavior: model selection, knowledge use, graph construction, parameterization, error handling, and interpretation — not only whether the final number happens to match.

Do not treat an old milestone benchmark as canonical simply because it exists. If its oracle, scope, or gate is no longer scientifically trusted, replace or retire it explicitly — and preserve the evidence separately from the wrapper that ran it.

## Execution Environment

`./run.sh` is the only supported entry point for project execution. Run Python, imports, probes, tests, linters, solver jobs, benchmarks, and generated scripts inside the `agent_solver` container. Do not run project commands such as `python`, `pip`, `pytest`, `ruff`, or `mypy` directly on the host; host-side work is limited to editing files, Git/Linear operations, and invoking Docker through `run.sh`.

- `./run.sh --no-build pytest -q` — reuse the existing image; this is the default.
- `./run.sh --rebuild pytest -q` — rebuild the image after `docker/Dockerfile` or dependency changes, or when it does not exist.
- `./run.sh --gpu pytest -q -m gpu` — the opt-in `agent_solver_gpu` image, in its own session. `MOA_GPUS` picks devices. See `docs/testing/gpu_environment.md`.

Do not silently fall back to host-side project execution. If a required check cannot run through `./run.sh`, report the environment failure.

Testing policy — targeted verification is the default, and verification cost scales with the regression risk the diff actually carries. Run the smallest relevant probe or test that establishes the changed contract; do not run several overlapping pytest subsets; do not rerun an expensive check that already passed unless a later edit could invalidate it. The full repository suite is **not** the default gate for every meaningful implementation change.

Run `./run.sh pytest -q` when broader regression risk justifies it, including when: shared core contracts or widely used boundary artifacts changed; solver or coupler behavior changed in a way that can affect multiple callers; dependency, Docker, or environment configuration changed; test collection, common fixtures, or repository-wide infrastructure changed; the task explicitly requires the full gate; or targeted verification exposed a plausible repository-wide regression.

The full suite is normally unnecessary for documentation-only changes that do not alter a scientific claim, narrowly isolated implementation fixes, local validation or failure-path fixes, registry metadata changes that introduce no new executable scientific capability, and test cleanup that changes no oracle, tolerance, or contract.

Do not rerun an expensive gate solely so both the implementation agent and the independent reviewer can observe the same passing result. A reviewer may rely on recorded test evidence unless the diff, subsequent edits, or the evidence itself gives a concrete reason to distrust it.

Scoped surfaces stay on-demand: tutorial tests are not run routinely; agent benchmarks run when agent prompts, tasks, graders, or orchestration change; GPU tests/benchmarks run only when the task needs GPU evidence; archived tests are not part of the active test surface. Targeted verification reduces cost, not standards: a change that runs, or produces visually plausible output, is still unverified until the changed contract itself has evidence.

Do not hard-code historical test counts or runtimes into this file; they become stale quickly.

## Shared GPU Server Policy

This is a shared server. **System stability has priority over throughput.**

Before any substantial GPU or memory-intensive run, inspect GPU occupancy (`nvidia-smi`), physical RAM (`free -h`), and the container/cgroup memory and swap state, and confirm the selected workload fits without relying on swap.

GPU selection:

- Prefer GPUs **6 and 7**.
- Avoid GPU **5** unless 6/7 are unavailable or the task specifically requires it.
- Use no more than **2 GPUs total** for this project at once.
- Unless a task explicitly requires multi-GPU execution, each individual workload should use **1 GPU**.
- Configure visibility through `run.sh` / container GPU settings; do not mutate host GPU state.

Memory safety:

- Monitor system RAM and cgroup memory/swap during substantial runs, not only before them.
- **Do not use swap as working memory.** Growth in the workload's cgroup swap is a stop condition: terminate the workload and report the resource failure instead of continuing.
- Prefer chunking, streaming, smaller batches, or sequential runs over risking server-wide memory pressure.
- Do not modify swap, `/etc/fstab`, mounts, GPU drivers, or systemd settings.
- Never reboot or shut down the server without explicit permission.

Parallelism: read-only analysis agents may run in parallel; compute-intensive agent jobs must obey the GPU/RAM limits above and should not overlap unless the task explicitly plans for it. Do not leave detached compute with `nohup`, `&`, `screen`, or `tmux` unless the task explicitly authorizes unattended execution.

## Required Workflow

Before editing:

- Read the task/Linear issue and identify its acceptance criteria and non-goals.
- Inspect `git status`, relevant code, tests, registry entries, and knowledge packs.
- Identify the scientific input/output contract and the verification oracle.
- Prefer the smallest executable probe that can resolve an API or physics uncertainty.

While editing:

- Implement only the requested scope.
- Do not opportunistically refactor unrelated code.
- Keep solver-specific behavior inside the appropriate adapter.
- Add or update tests for changed logic, boundary contracts, conventions, and failure paths.
- Record uncertainty explicitly; never fabricate unsupported solver behavior.

Before completion:

- Run the task-required checks and the targeted verification for the changed contract; escalate to the full gate only when the testing policy above calls for it.
- Review the diff for unrelated changes.
- Check the independent-review triggers below, and obtain a review when one applies.
- Report tests run, tests not run, scientific risks, resource issues, intentional non-goals, and follow-up work.

## Independent Code Review

Implementation and review are separate roles. Where a change carries scientific risk, the agent that wrote it must not be the only agent judging whether it is correct. Review is triggered by that risk, not by diff size.

Independent review is **required** when the change affects any of: external solver adapter behavior or solver API use; couplers or representation-changing boundaries; physical assumptions or conventions; units, coordinate systems, wavelength/frequency handling, polarization/coherence, normalization, sampling, or reference planes, when those contracts change; numerical algorithms, interpolation, quadrature, precision, tolerances, or convergence behavior; gradients, autodiff, differentiability, or cross-framework derivative claims; shared core scientific boundary artifacts; executable model/coupler capability claims; benchmark oracles, scientific tolerances, or acceptance criteria; or GPU/RAM allocation, batching, workload scale, or otherwise substantial resource behavior.

Independent review is normally **optional** for documentation-only changes that do not alter a scientific claim, formatting and comments, isolated developer tooling changes, test cleanup that alters no scientific oracle or tolerance, and narrow implementation changes with no solver/API/physics/boundary/numerical/resource contract impact.

When invoking the reviewer, supply: the task acceptance criteria, the relevant diff, the tests/probes already run and their results, known uncertainties, and areas intentionally left unverified. The reviewer is expected to consume that evidence rather than recreate it.

The repository may provide a tool-specific reviewer definition (for Claude Code, `.claude/agents/code-reviewer.md`). Keep the detailed reviewer prompt there rather than expanding this file. The reviewer is read-only, scopes its depth to the risk domains the diff actually touches, and stops once the affected acceptance criteria and changed-code risks have sufficient evidence.

Review findings are classified as **must fix before merge**, **should fix soon**, or **safe to merge / no blocker**. A reviewer may run one narrow read-only check through `./run.sh` when a specific review question cannot be settled from the available evidence, but should not duplicate the implementation agent's passing runs or expensive tutorial/GPU/full benchmark runs unless the task specifically requires them.

## PR Contract

Every PR should state what changed and why, the task/Linear issue, acceptance criteria checked, tests and benchmarks run, tests not run, scientific/resource risks, intentional non-goals, agent involvement, and follow-up work.
