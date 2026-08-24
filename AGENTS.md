# Repository Instructions — Canonical Shared Context

This is the repository-wide instruction source for coding and research agents. Task-specific goals, acceptance criteria, non-goals, and ownership live in the linked Linear issue or explicit task prompt.

## Mission

Build a **scientifically trustworthy agentic system for multi-scale optical and nanophotonic simulation**. The system should compose existing physics solvers through explicit, testable couplers and use compact, versioned knowledge packs to choose, configure, and verify those components.

Forward simulation is the current priority. Inverse design is a future extension and must not be treated as supported across a boundary until the relevant gradients are explicitly verified.

A runnable script or visually plausible result is not, by itself, a scientifically trustworthy result.

## System Model

Each supported physics model has four parts:

1. an external solver/API,
2. a narrow repository adapter,
3. an agent-facing knowledge pack,
4. independent verification evidence and benchmarks.

Each coupler has the same separation: implementation, explicit boundary contracts, agent-facing knowledge, and independent verification.

Knowledge packs live under `knowledge/` and contain the compact information an agent needs to use a component correctly, such as:

- `card.yaml`
- `conventions.md`
- `usage_notes.md`
- `api_minimal_examples.md`
- `failure_guide.md`
- coupler theory where needed

Do not duplicate those details in this file. Load the relevant pack only when a task uses that solver or coupler.

The currently supported core is the Optiland geometric-ray model, the Chromatix scalar-wave model, and the repository ray/wave couplers. Treat `src/registry/` and executable capability declarations as the source of truth for what is actually supported; do not infer capability from roadmap text or old milestone reports.

## Current Phase

The project is now in a **benchmark redesign + codebase hardening + agent integration** phase.

The benchmark system has two physics benchmark classes:

1. **single-component benchmarks** — one solver or one coupler in isolation,
2. **composed physics benchmarks** — solver -> coupler -> solver chains.

These physics benchmarks are the scientific tasks that the agentic benchmark operates on; they are not parallel to it. The **agentic benchmark is the higher-level evaluation framework**: the agent is given benchmark problems drawn from the single-component and composed-physics sets and must select the appropriate model(s), load the relevant knowledge, configure and execute the graph, and interpret and verify the result correctly.

The purpose of the benchmark system is therefore twofold: the underlying physics benchmarks establish numerical and scientific correctness, while the agentic benchmark tests whether an agent can use those verified components correctly. Together they should cover model choice, boundary conventions, numerical convergence, physical correctness, failure behavior, reproducibility, and scientific interpretation.

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
- `src/verification/`: independent scientific oracles and measurements.
- `src/agent/`: agent execution/benchmark infrastructure, not solver physics.
- `knowledge/`: compact agent-facing knowledge, not executable evidence or tutorial test code.
- `benchmarks/`: benchmark protocols, probes, records, and agent tasks.
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

Every scientific benchmark should declare, as applicable:

- the physical question and approximation being tested,
- input geometry/source and boundary conditions,
- solver/coupler graph,
- units, conventions, device, dtype, and sampling,
- independent or analytic oracle,
- quantitative metrics and pass/fail tolerance,
- convergence variables and convergence criterion,
- expected failure or out-of-domain cases,
- reproducibility/provenance information,
- runtime and memory budget when the workload is substantial.

For coupled benchmarks, test both the individual components and the handoff. A correct final image can hide an incorrect intermediate convention.

For agent benchmarks, grade the reasoning-relevant behavior: model selection, knowledge use, graph construction, parameterization, error handling, and interpretation — not only whether the final number happens to match.

Do not treat an old milestone benchmark as canonical simply because it exists. If its oracle, scope, or gate is no longer scientifically trusted, replace or retire it explicitly.

## Execution Environment

`./run.sh` is the only supported entry point for project execution. Run Python, imports, probes, tests, linters, solver jobs, benchmarks, and generated scripts inside the project container.

Do not silently fall back to host-side project execution. If a required check cannot run through `./run.sh`, report the environment failure.

Normal testing policy:

- During implementation, run the smallest relevant probe or test needed to answer the current question.
- Do not shotgun many overlapping pytest subsets.
- For a meaningful code change, normally run the default suite once before completion: `./run.sh pytest -q`, unless the task defines a different gate.
- Tutorial tests are on-demand; do not run them routinely.
- Agent benchmarks are on-demand when agent prompts, tasks, graders, or orchestration change.
- GPU tests/benchmarks run only when the task needs GPU evidence.
- Archived tests are not part of the active test surface.

Do not hard-code historical test counts or runtimes into this file; they become stale quickly.

## Shared GPU Server Policy

This is a shared server. **System stability has priority over throughput.**

Before any substantial GPU or memory-intensive run:

- inspect GPU occupancy with `nvidia-smi`,
- inspect physical RAM with `free -h`,
- inspect the container/cgroup memory and swap state,
- confirm the selected workload fits without relying on swap.

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

Parallelism:

- Read-only analysis agents may run in parallel.
- Compute-intensive agent jobs must obey the GPU/RAM limits above and should not overlap unless the task explicitly plans for it.
- Do not leave detached compute with `nohup`, `&`, `screen`, or `tmux` unless the task explicitly authorizes unattended execution.

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

- Run the task-required checks and the appropriate default repository gate.
- Review the diff for unrelated changes.
- Obtain an independent code review for non-trivial implementation changes.
- Report tests run, tests not run, scientific risks, resource issues, intentional non-goals, and follow-up work.

## Independent Code Review

Implementation and review are separate roles. The agent that writes a non-trivial change should not be the only agent judging whether that change is correct.

The repository may provide a tool-specific reviewer definition (for Claude Code, `.claude/agents/code-reviewer.md`). Keep the detailed reviewer prompt there rather than expanding this file.

The reviewer is read-only by default and should evaluate:

- compliance with the task scope and acceptance criteria,
- external solver API correctness,
- scientific validity of the selected approximation,
- units, conventions, reference planes, and artifact boundaries,
- coupler assumptions and information preservation,
- numerical stability, sampling, and convergence,
- gradient/differentiability claims,
- structured failures and unsupported cases,
- adequacy and independence of tests/oracles,
- GPU/RAM behavior for substantial workloads,
- accidental unrelated changes.

Review findings should be classified as:

- **must fix before merge**,
- **should fix soon**,
- **safe to merge / no blocker**.

A reviewer may run narrow read-only checks through `./run.sh`, but should not duplicate expensive tutorial/GPU/full benchmark runs unless the task specifically requires them.

## PR Contract

Every PR should state what changed and why, the task/Linear issue, acceptance criteria checked, tests and benchmarks run, tests not run, scientific/resource risks, intentional non-goals, agent involvement, and follow-up work.
