# Multiscale Optics Agent

Agent-facing infrastructure for **verified** multi-scale optical simulation:
typed physics models joined by explicit, testable couplers, where every model
boundary declares its units, axes, frame, phasor sign, normalization and
sampling — and where a runnable script is not accepted as a scientific result.

The design premise is that the hard part of composing optical solvers is not
calling them. It is the handoffs: a ray bundle's optical path length is not a
phase until someone says what it is referenced to, and a plausible field is the
most expensive kind of wrong answer.

## What executes today

```
Optiland  ──▶  C_RAY_TO_WAVE  ──▶  Chromatix  ──▶  PSF
 ray trace      wavelet sum        scalar ASM      measurement,
                                                   not a coupler
```

Two models and two couplers, and that is the whole registry — components with no
implementation were removed rather than kept as intentions. `benchmarks/roadmap.md`
records what was removed and why.

| Component | What it is | Maturity |
| -- | -- | -- |
| `M_RAY_OPTILAND` | sequential geometric ray tracing, Optiland 0.6.0 | characterized |
| `M_WAVE_CHROMATIX` | scalar angular-spectrum propagation, Chromatix @ `d24bdf0` | characterized |
| `C_RAY_TO_WAVE` | rays → complex field, by coherent wavelet sum | characterized |
| `C_WAVE_TO_RAY` | complex field → rays, Monte Carlo over the angular spectrum. **Library API, no graph node** | characterized |

`characterized` means measured against an analytic case, a conservation law, a
convergence study or an independent implementation. It does not mean
`validated`: **no gradient across any coupler is certified**, and a
PyTorch-to-JAX handoff is forward-only by default.

The one live benchmark, `benchmarks/physics/L2-PSF-01/`, has an **unmet** gate —
`1.0e-3 fft_oracle_intensity_relative_l2`, measured at 2.2e-3 on the real traced
system. It is carried forward as an explicit open limitation rather than
widened.

## Running anything

**`./run.sh` is the only supported entry point.** It runs the command inside the
`agent_solver` container with the repository mounted at `/workspace`, so source
edits are live against the editable install. Do not run `python`, `pip`,
`pytest`, `ruff` or `mypy` on the host; if a check cannot run through `run.sh`,
report a structured environment failure rather than falling back.

```bash
./run.sh                          # interactive shell
./run.sh pytest -q                # the default suite
./run.sh --rebuild pytest -q      # after a docker/ or dependency change
./run.sh --gpu pytest -q -m gpu   # the CUDA image, in its own session
```

## The five test commands

| Command | Size | When |
| -- | -- | -- |
| `./run.sh pytest -q` | 889 tests, ~190 s | after every change |
| `./run.sh pytest -q -m "not slow"` | ~870, ~39 s | while iterating |
| `./run.sh pytest -q tests_tutorial` | 60, **~33 min** | when a pin or `docker/` changes |
| `./run.sh pytest -q benchmarks/agents` | 52, ~8 s | when an agent task, prompt or grader changes |
| `./run.sh --gpu pytest -q -m gpu` | 48, ~70 s | **its own session**, on an attached CUDA device |

The split is by *location*, not by marker discipline: `testpaths = ["tests"]` is
what makes the expensive and the historical groups opt-in, so no command has to
remember to exclude them. GPU tests skip whenever anything else is selected with
them, because enabling the GPU mutates process-global JAX state.

## Where things are

```text
src/
  core/         the shared vocabulary: the four boundary artifacts
                (RayBundle, WavefrontSamples, ComplexField, PSF), the
                precision/device/dtype policy, graph specs, execution status.
                Imports nothing from its siblings — enforced by a test.
  solvers/      one subpackage per external engine: optiland/, chromatix/.
                Never imports couplers/.
  couplers/     the ray-wave physics, and only that. Never imports solvers/.
  verification/ independent oracles — Airy, Fraunhofer, float64 ASM — and the
                terminal PSF measurement.
  studies/      specific investigations (metalens), not reusable capabilities.
  agent/        the agent benchmark harness.
  registry/     what exists, declared in YAML. Declares; does not execute.
  cli.py        `multiscale-optics list-models | list-couplers | validate`

tests/          the default suite
tests_tutorial/ upstream tutorial reproductions: a gate on the pinned
                dependency, not on this repository's physics
benchmarks/     protocols, the one live suite, probes, records, milestone
                reports, and roadmap.md for what is planned but not executable
benchmarks/agents/ the agent benchmark's tasks and prompts
knowledge/      concise agent-facing context: one card per component
docs/           architecture, precision policy, testing, prescriptions
archive/        frozen. Preserved, not runnable; each generation's README says
                what is now unguarded
examples/graphs/ example graph YAML, validated by `scripts/validate_package.py`
```

Two structural rules are enforced by tests rather than by review:
`tests/test_package_dependencies.py` pins the import direction between packages
and refuses cycles, and `tests/test_flat_layout.py` pins that every top-level
name resolves inside this repository — the accepted cost of a flat `src/` is
that this distribution owns names like `core`, so a shadowing install must fail
loudly instead of silently serving someone else's.

## Where to go next

| Question | File |
| -- | -- |
| What are the rules for working here? | `AGENTS.md` — the single canonical instruction source |
| How do the layers fit together? | `docs/architecture/solver_layering.md` |
| Why is dtype not the same as precision? | `docs/precision/precision_device_policy.md` |
| How do I use Optiland / Chromatix correctly? | `knowledge/solvers/<name>/` |
| What does a coupler assume? | `knowledge/couplers/<name>/` |
| Why can't I reach a GPU? | `docs/testing/gpu_environment.md` |
| What is planned but not built? | `benchmarks/roadmap.md` |

Work is tracked in Linear. `AGENTS.md` is the canonical repository-wide
instruction source; `CLAUDE.md` imports it and adds nothing.
