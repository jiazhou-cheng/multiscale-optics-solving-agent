# Repository Instructions — Canonical Shared Context

This is the single repository-wide instruction source for Codex and Claude Code; task-specific scope lives in Linear and in files explicitly linked by the Linear issue.

## Mission

- Build a research-grade system that composes optical models through explicit, testable couplers.
- Prefer one verified vertical slice over many speculative adapters.
- Current milestone: `Optiland -> ray-to-wave coupler -> Chromatix -> PSF`.
- A runnable script is not a verified scientific result.

## Current Scope

- Ray model: Optiland, exposed through a narrow adapter.
- Coupler: the repository's existing ray-wave implementation, audited before refactoring.
- Wave model: Chromatix, exposed through a narrow propagation/PSF adapter.
- Primary output: a reproducible forward ray-to-wave demo with physics checks.
- Gradient work starts only after the forward path and conventions are verified.
- Broad planner work and unrelated benchmarks are out of scope unless a Linear issue explicitly brings them in. FMMAX, FDTDX, JAX-FEM and SAX are past that: each was **removed outright**, dependency and all, and none can be restored by a revert. SAX went first (CHE-72) because its `klujax` dependency pinned `jax_platform_name='cpu'` at import time, silently disabling the GPU for the whole process; the other three went with CHE-87, atomically across adapter, registry entry, example graph, knowledge pack, pytest marker and dependency pin, because a partial retirement leaves a registry claim that reads to a planner as a supported capability. Reintroducing any of them means a fresh, isolated integration with explicit backend behavior. Intent and the findings worth keeping (FMMAX's unresolved Fresnel sign convention; JAX-FEM's GPLv3 against this project's MIT): `benchmarks/roadmap.md`.

## Sources of Truth

- Repository-wide rules: this file.
- Task goal, acceptance criteria, non-goals, and ownership: the Linear issue.
- Scientific details for a task: only files linked by that issue.
- Package API truth: pinned installed behavior, official versioned docs, and executable probes.
- Existing code is evidence, not automatically the intended design; surface
  any conflict between these sources rather than silently choosing one.

## Execution Environment — Container Only

- `./run.sh` is the only supported entry point for executing project code.
- Run all Python, package imports, API probes, tests, linters, formatters, type checks, CLIs, solver jobs, benchmarks, and generated scripts inside the `agent_solver` container.
- Do not run project commands such as `python`, `pip`, `pytest`, `ruff`, `mypy`, or solver executables directly on the host.
- Use these command forms from the repository root:

```bash
./run.sh                              # existing image, then interactive shell
./run.sh pytest -q                    # existing image, then test in container
./run.sh python path/to/probe.py      # existing image, then run probe in container
./run.sh --no-build pytest -q         # do not rebuild the image, default
./run.sh --rebuild pytest -q          # rebuild the image (cached) when the package information changes, then test
```

- The default path reuses the existing `agent_solver` image. Run with `--rebuild` after changing `docker/Dockerfile`, dependency files, or when the image does not exist.
- The repository is mounted at `/workspace`, which is also the container working directory, so source edits are visible immediately to the editable install.
- Host-side work is limited to editing files, Git/Linear operations, and invoking Docker through `run.sh`; scientific or project-code execution remains containerized.
- Do not assume GPU access, and record the actual device used. The default image is CPU-only by construction; GPU work uses the opt-in `agent_solver_gpu` image via `./run.sh --gpu ...` (`MOA_GPUS` picks devices, max 2). Setup, evidence, and two traps that silently disable the GPU: `docs/testing/gpu_environment.md`.
- Precision, dtype, device and array namespace are four separate concepts, not one string (CHE-61). Never write a requested device or precision into an artifact: read it off the array. What each package can actually execute is declared once in `core/capabilities.py`, and cross-model conversion goes through `core/precision.py`'s bridge planner under an explicit policy. Registry `devices`/`dtypes` reflect that model and are updated only after executable tests pass. Policy, measured tolerances, and the two silent precision losses this uncovered: `docs/precision/precision_device_policy.md`.
- If a required check cannot run through `./run.sh`, report a structured environment/setup failure instead of silently falling back to the host.

## Test Command Surface

Five groups, split by *location* rather than by marker discipline (CHE-67; what moved where and why: `docs/testing/test_archive.md`). Location is load-bearing: `testpaths = ["tests"]` is what makes the expensive and the historical groups opt-in, so no command has to remember to exclude them.

- **Default active suite** (required after every change, 889 tests): `./run.sh pytest -q` — everything under `tests/`, including the 19 `slow` scientific tests (Monte Carlo/wave-to-ray convergence, coupler gradient and round-trip, and the wheel-install check). Last measured by CHE-89 on a rebuilt image: **841 passed, 48 skipped in 188 s**. `-m "not slow"` is the fast subset CHE-53 called Tier A — **822 passed, 48 skipped, 39 s**. Subsystem subsets stay independently invocable and overlap by design: `-m optiland`, `-m chromatix`, `-m coupler`, `-m slow`. 48 of the 889 are `gpu`-marked and skip unless the session is GPU-dedicated. The count grew across the CHE-84 cleanup from 817: CHE-88 restored 46 archived PSF-oracle tests and added 12, and CHE-89 added 14 flat-layout guards.
- **On-demand tutorial suite** (60 tests, **~33 min**): `make test-tutorial`, i.e. `./run.sh pytest -q tests_tutorial`. Reproductions of upstream Optiland/Chromatix tutorials — a gate on the *pinned dependency*, not on this repository's physics, and 76% of the old suite's runtime. Run it when a pin or `docker/` changes, before/after a substantial solver-integration change, or as a weekly sweep; not per PR. Active and maintained, just never collected unless the directory is named: `tests_tutorial/README.md`.
- **Archived** (276 tests, `archive/tests/gen1/`): the superseded L1/L2/M1 benchmark suite, the FMMAX/FDTDX adapter tests (whose adapters CHE-87 has since deleted, so those files are archived tests of code that no longer exists), and the outdated M1/M2/M3/M3R milestone suites. (CHE-67 archived 288 in 22 files; CHE-72 then deleted the 12 SAX adapter tests outright along with the package.) Preserved, not deleted, and not runnable — naming an archived file aborts the session with a usage error. Restoring one is an explicit `git mv` back under `tests/`, justified in a Linear issue (`archive/tests/gen1/README.md`). The cost, stated plainly: those milestone behaviors are now **unguarded**, so a change that would have broken them now fails nothing; if an issue depends on one, unarchive it instead of assuming it still holds.
- **On-demand agent benchmark** (52 tests, **~8 s**): `make test-agent-benchmark`, i.e. `./run.sh pytest -q benchmarks/agents`. The CHE-71 V1 agent benchmark, graded against its own reference and negative participants. Opt-in by location for the same reason as the tutorial suite — the point of it is to run an *agent*, which is nondeterministic and consumes model tokens — but its own gate is deterministic and cheap, so run it when a task, prompt or grader changes. Its harness *is* in the default suite (`tests/test_agent_benchmark.py`, 53 tests): the grader decides whether an agent passed, so a regression in it would silently change every score. Design decisions, task format and how to add one: `benchmarks/agents/README.md`.
- **GPU** (48 tests): `./run.sh --gpu pytest -q -m gpu`, in its own session, on an attached CUDA device. Revalidated by CHE-89 on a **rebuilt** `agent_solver_gpu` — **48 passed in 69 s** on one RTX A6000, matching CHE-72/CHE-73's 2026-08-20 figure of 70 s (CHE-70 measured 48 passed in 77 s on the pre-SAX-removal image, which cleared CHE-60's outstanding dedicated pass). The suite now asserts real kernel execution *and* one coherent CUDA 12.6 dependency family, not device enumeration; `jax_platform_name` is neither set nor repaired anywhere. Setup, the resolved CUDA stack, and the traps: `docs/testing/gpu_environment.md`.

`benchmark` remains a registered marker that now selects nothing, so an unarchived file is valid pytest the moment it moves back. The `sax` marker was deleted by CHE-72 and `fmmax`/`fdtdx` by CHE-87, with their integrations; the tier commands recorded under `docs/archive/2026-08-testing/` still run and simply match less, because a recorded `-m` expression naming a deleted marker still parses -- an unknown name evaluates false; there is no Tier C until the replacement benchmark baseline lands. Per-test runtime/memory, the tier review, and the opt-in profiler whose swap guardrail fails a run when the *container's* cgroup swap grows (host swap is non-zero at rest here, so it cannot be the signal): `docs/archive/2026-08-testing/test_runtime_audit.md` (CHE-64), whose paths predate CHE-67.

This is orthogonal to the `jax`/`torch`/`integration` markers, which mean "requires that optional install," not "is slow" or "is out of scope" — and to `gpu` (CHE-60), which means "needs an attached CUDA device." Enabling the GPU mutates process-global JAX state, so `gpu` tests run only in a dedicated `./run.sh --gpu pytest -q -m gpu` session and skip whenever anything else is selected with them; that skip is what keeps every other command green unchanged.

## Context Loading

- Do not read every file under `docs/` or `knowledge/` by default.
- Start with this file, the Linear issue, its linked spec, and the files/tests directly involved.
- Load a solver or coupler card only when the issue uses that solver or coupler,
  and minimal API examples/failure guides only after the adapter path is selected.
- Long paper, catalog, benchmark, and historical architecture documents are reference material, not startup context.

## Required Workflow

Before editing:

- Read the Linear issue and restate its acceptance criteria and non-goals in the work log.
- Inspect `git status`, relevant code, tests, and current implementation patterns.
- Identify the scientific input/output contract and verification oracle.
- Run or create the smallest executable probe before designing a broad wrapper.

While editing:

- Implement only the linked issue.
- Do not opportunistically refactor unrelated code.
- Preserve existing behavior unless the issue explicitly changes it.
- Keep external solver imports inside adapter modules.
- Add or update tests for changed logic, contracts, conventions, and failure paths.
- Record uncertainty explicitly; never fabricate solver output or unsupported API behavior.

Before completion:

- Run the checks required by the issue plus the narrowest relevant repository checks.
- Review the diff for unrelated changes.
- Report tests run, tests not run, risks, intentional non-goals, and follow-up issues.
- Update the Linear issue with the result and link the PR.

## Scientific Non-Negotiables

- Declare units, axes, coordinate frame, handedness, wavelength/frequency, phasor sign, polarization basis, coherence model, normalization, sampling, and reference plane at every model boundary.
- Use SI internally unless an issue defines and tests another convention.
- Complex fields represent amplitudes, not intensities.
- Never claim a gradient across an untested boundary.
- A PyTorch-to-JAX handoff is `forward_only` by default.
- Promote a cross-framework path only after an explicit custom derivative and directional finite-difference test pass.
- Verification must use an analytic case, conservation law, convergence study, or independent implementation when feasible.
- Failed solvers return structured diagnostics; they do not return invented fields, metrics, or convergence claims.

## Granularity Rules

Distinguish three levels:

- Physics-graph node: one independently executable physical approximation with a stable artifact contract.
- Adapter capability: one narrow operation exposed from an external package.
- Python module/class: an implementation unit organized around one cohesive responsibility.

For the current slice:

- The entire sequential ray trace to a declared pupil/reference plane is one model node.
- Individual refractive surfaces are internal ray-model data by default, not graph nodes.
- A surface becomes a graph boundary only when it is independently simulated, replaced by another physical model, or emits/consumes a reusable typed artifact.
- The ray-to-wave transformation is a coupler because it changes representation
  and carries physical assumptions; Chromatix propagation is a separate model node.

Split a code module when it mixes scientific contracts, imports multiple solver families, cannot be tested independently, or requires unrelated changes to evolve. Do not split solely to make every file small.

## Initial Artifact Boundary

The first slice should stabilize only these artifacts:

- `RayBundle`: positions, directions, wavelength, optical path length, amplitude/weight, polarization/coherence metadata, and reference plane.
- `WavefrontSamples`: pupil coordinates plus phase/OPD and amplitude samples before rasterization.
- `ComplexField`: sampled complex field plus spacing, axes, wavelength, normalization, and frame metadata.
- `PSF`: intensity field plus coordinate and normalization metadata.

Do not add a large universal type system to complete a narrow issue.

## Existing Ray-Wave Code

- Treat `ray_wave`, `ray_ewave`, and similarly named code as untrusted until characterized.
- Preserve names and numerical behavior during the audit while determining actual
  inputs, outputs, units, phase sign, amplitude weighting, interpolation, and differentiability.
- Add characterization tests before renaming, merging, or optimizing.
- Map the implementation to `C_RAY_TO_WAVE` only after its semantics are known.

## Adapter Definition of Done

A narrow adapter is complete only when it has:

- A typed request and result contract.
- A pinned package version or commit, one import probe, and one minimal forward probe.
- Explicit conventions and supported devices/dtypes.
- One analytic or independently reviewed validation case.
- Structured failure behavior.
- A gradient test only when differentiability is claimed.

## PR and Review Contract

Every PR states:

- What changed and why.
- The Linear issue.
- Acceptance criteria checked.
- How it was tested and what was not tested.
- Risks and intentionally excluded work.
- Agent involvement and follow-up issues.

Classify feedback against the linked Linear issue as: must fix before merge,
should fix soon, or safe to merge.

## GPU server resource policy

This is a shared GPU server with limited system RAM relative to total GPU
memory. Stability is more important than parallelism.

When working on this machine:

- Never run multiple GPU-consuming jobs concurrently; always run them sequentially.
- Do not use more than 2 GPUs in this project; configure visibility via the
  container (`--gpus`/`NVIDIA_VISIBLE_DEVICES` through `run.sh`), not host state.
- Never launch training, inference, CUDA, PyTorch, JAX, or similar workloads
  as background jobs.
- Before starting a GPU-intensive command, run `nvidia-smi` and `free -h`, and
  check whether another GPU job is already running.
- If substantial GPU or system memory is already occupied, ask me before
  starting another workload.
- Do not use agent teams or parallel agents for tasks that execute code;
  subagents may only be used for read-only analysis, never to launch
  compute-intensive processes.
- Do not use nohup, &, screen, tmux, or other mechanisms to leave compute
  jobs running without explicit permission.
- Do not modify swap, /etc/fstab, mounts, GPU drivers, or systemd settings.
- Never reboot or shut down this server without explicit permission.
