# M0.1 — Code, Adapter, Test, and Entry-Point Audit

**Issue:** CHE-5 (M0 — Repository Audit and Archive)
**Date:** 2026-08-11
**Commit audited:** `43a0360` (working tree clean of implementation changes at audit start)
**Rule:** read-only. No implementation file was modified by this audit.

Every finding below is labeled **[EXEC]** (verified by running a command in the
`agent_solver` container) or **[READ]** (inference from reading source, not executed).

## How this audit was produced

```bash
./run.sh pytest -q                                          # baseline
./run.sh python docs/audit/probes/audit_import_graph.py     # inventory + import graph
./run.sh python scripts/validate_package.py
./run.sh python -m multiscale_optics_agent.cli list-models
./run.sh python -m multiscale_optics_agent.cli list-couplers
./run.sh python -m multiscale_optics_agent.cli validate examples/graphs/<each>.yaml
./run.sh python scripts/export_schemas.py                   # drift check
```

Raw output: `docs/audit/logs/`
(`m01_pytest_baseline.txt`, `m01_import_graph.txt`, `m01_entrypoints.txt`,
`m01_export_schemas.txt`).

### Environment blocker found and worked around [EXEC]

`run.sh` calls `docker run --rm -it`. A caller without a TTY (agent shell, CI job)
gets `the input device is not a TTY` and exit 1. Every command in this audit was
therefore run as:

```bash
script -qec "./run.sh <command>" /dev/null
```

This still executes inside `agent_solver`; no project code was run on the host.
Recommended follow-up (not applied — outside CHE-5's read-only scope): make the
`-t` flag conditional on `[ -t 0 ]` in `run.sh`.

`tmp_probes/` is **root-owned** (`drwxr-xr-x root root`) because earlier container
runs created it as root, so the host user cannot write there. The audit probes were
placed in `docs/audit/probes/` instead. Any future workflow that writes files from
inside the container will hit the same ownership problem.

## Headline counts [EXEC]

| Metric | Value |
|---|---|
| Python files audited | 36 |
| active (imported by non-test code) | 8 |
| test-only (imported only by tests) | 6 |
| test modules | 10 |
| standalone scripts | 3 |
| unreferenced by any Python import | 9 |
| non-Python assets audited | 14 |
| assets not referenced from Python | 6 |
| Test baseline | **52 passed, 2 xfailed, 1 xpassed in 54.74s** |

The five external solver packages (`optiland`, `chromatix`, `fmmax`, `fdtdx`, `sax`,
plus `jax`, `torch`, `numpy`, `networkx`, `pydantic`, `typer`, `rich`) are all
importable inside the image [EXEC].

## A. Module inventory

`status` is computed from the import graph, not from reading:
`active` = imported by at least one non-test module; `test-only` = imported only by
tests; `unreferenced-by-python` = no Python file imports it (it may still be an entry
point or a package placeholder — see the entry-point table).

| module | path | loc | status | imported_by | purpose |
| --- | --- | --- | --- | --- | --- |
| `multiscale_optics_agent` | `src/multiscale_optics_agent/__init__.py` | 16 | unreferenced-by-python | — | Typed physics graphs for multi-scale optical simulation. |
| `multiscale_optics_agent.adapters` | `src/multiscale_optics_agent/adapters/__init__.py` | 22 | active | adapters.registry, tests.test_chromatix_adapter | External physics-solver adapters. |
| `multiscale_optics_agent.adapters.base` | `src/multiscale_optics_agent/adapters/base.py` | 63 | active | adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.registry, adapters.sax_adapter, couplers.base, tests.test_chromatix_adapter, tests.test_fdtdx_adapter, tests.test_fmmax_adapter, tests.test_optiland_adapter, tests.test_sax_adapter | Stable interface implemented by external physics-solver adapters. |
| `multiscale_optics_agent.adapters.chromatix_adapter` | `src/multiscale_optics_agent/adapters/chromatix_adapter.py` | 602 | test-only | tests.test_chromatix_adapter | Adapter for ``M_WAVE_CHROMATIX``: Chromatix scalar angular-spectrum propagation. |
| `multiscale_optics_agent.adapters.fdtdx_adapter` | `src/multiscale_optics_agent/adapters/fdtdx_adapter.py` | 599 | test-only | tests.test_fdtdx_adapter | Forward-only adapter for ``M_EM_FDTDX`` (FDTDX, pinned version 0.6.2). |
| `multiscale_optics_agent.adapters.fmmax_adapter` | `src/multiscale_optics_agent/adapters/fmmax_adapter.py` | 649 | test-only | tests.test_fmmax_adapter | Adapter for ``M_RCWA_FMMAX`` — FMMAX (Fourier Modal Method / RCWA), pinned ``fmmax==1.7.1``. |
| `multiscale_optics_agent.adapters.optiland_adapter` | `src/multiscale_optics_agent/adapters/optiland_adapter.py` | 669 | test-only | tests.test_optiland_adapter | Adapter for ``M_RAY_OPTILAND`` (Optiland sequential ray tracer, pinned 0.6.0). |
| `multiscale_optics_agent.adapters.registry` | `src/multiscale_optics_agent/adapters/registry.py` | 48 | test-only | tests.test_adapter_registry | Runtime discovery of concrete ModelAdapter implementations. |
| `multiscale_optics_agent.adapters.sax_adapter` | `src/multiscale_optics_agent/adapters/sax_adapter.py` | 652 | test-only | tests.test_sax_adapter | Adapter for ``M_CIRCUIT_SAX`` -- differentiable photonic-circuit S-parameter |
| `multiscale_optics_agent.agents` | `src/multiscale_optics_agent/agents/__init__.py` | 1 | unreferenced-by-python | — | Agents package placeholder for the first implementation milestone. |
| `multiscale_optics_agent.cli` | `src/multiscale_optics_agent/cli.py` | 78 | unreferenced-by-python | — | Command-line interface for registry inspection and graph validation. |
| `multiscale_optics_agent.core` | `src/multiscale_optics_agent/core/__init__.py` | 1 | unreferenced-by-python | — | Framework-neutral graph schemas and validation. |
| `multiscale_optics_agent.core.artifacts` | `src/multiscale_optics_agent/core/artifacts.py` | 30 | active | adapters.base, adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.sax_adapter, couplers.base, scripts.export_schemas, tests.test_artifacts, tests.test_chromatix_adapter, tests.test_fmmax_adapter, tests.test_optiland_adapter | Framework-neutral records for arrays and scientific artifacts produced by solvers. |
| `multiscale_optics_agent.core.errors` | `src/multiscale_optics_agent/core/errors.py` | 45 | active | adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.registry, adapters.sax_adapter, registry.loader, tests.test_adapter_registry, tests.test_chromatix_adapter, tests.test_fdtdx_adapter, tests.test_fmmax_adapter, tests.test_optiland_adapter, tests.test_sax_adapter | Project-specific exceptions. |
| `multiscale_optics_agent.core.graph` | `src/multiscale_optics_agent/core/graph.py` | 434 | active | multiscale_optics_agent, adapters.base, adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.sax_adapter, cli, couplers.base, scripts.validate_package, tests.test_graph_validation | Deterministic validation for typed model-coupler graphs. |
| `multiscale_optics_agent.core.provenance` | `src/multiscale_optics_agent/core/provenance.py` | 27 | active | scripts.export_schemas | Minimal provenance schema for reproducible benchmark runs. |
| `multiscale_optics_agent.core.specs` | `src/multiscale_optics_agent/core/specs.py` | 258 | active | multiscale_optics_agent, adapters.base, adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.sax_adapter, core.artifacts, core.graph, couplers.base, registry.loader, scripts.export_schemas, tests.test_artifacts, tests.test_chromatix_adapter, tests.test_fmmax_adapter, tests.test_graph_validation, tests.test_optiland_adapter, tests.test_sax_adapter | Pydantic schemas for the model-coupler intermediate representation. |
| `multiscale_optics_agent.couplers` | `src/multiscale_optics_agent/couplers/__init__.py` | 1 | unreferenced-by-python | — | Couplers package placeholder for the first implementation milestone. |
| `multiscale_optics_agent.couplers.base` | `src/multiscale_optics_agent/couplers/base.py` | 44 | unreferenced-by-python | — | Stable interface for first-class physical couplers. |
| `multiscale_optics_agent.evaluation` | `src/multiscale_optics_agent/evaluation/__init__.py` | 1 | unreferenced-by-python | — | Evaluation package placeholder for the first implementation milestone. |
| `multiscale_optics_agent.evaluation.checks` | `src/multiscale_optics_agent/evaluation/checks.py` | 30 | unreferenced-by-python | — | Serializable outputs of numerical, physical, and derivative verification checks. |
| `multiscale_optics_agent.registry` | `src/multiscale_optics_agent/registry/__init__.py` | 5 | unreferenced-by-python | — | Model and coupler registries. |
| `multiscale_optics_agent.registry.loader` | `src/multiscale_optics_agent/registry/loader.py` | 74 | active | multiscale_optics_agent, adapters.chromatix_adapter, adapters.fdtdx_adapter, adapters.fmmax_adapter, adapters.optiland_adapter, adapters.sax_adapter, cli, core.graph, registry, scripts.validate_package, tests.conftest, tests.test_graph_validation, tests.test_registry | Load model and coupler specifications from versioned YAML registries. |
| `scripts.check_context_sync` | `scripts/check_context_sync.py` | 57 | standalone-script | — | Validate the static Codex/Claude Code context entrypoints. |
| `scripts.export_schemas` | `scripts/export_schemas.py` | 36 | standalone-script | — | Export JSON Schemas used by planners, editors, and external tooling. |
| `scripts.validate_package` | `scripts/validate_package.py` | 52 | standalone-script | — | Fast repository consistency checks that do not import optional physics solvers. |
| `tests.conftest` | `tests/conftest.py` | 29 | test | — | Shared pytest fixtures and helpers for the test suite. |
| `tests.test_adapter_registry` | `tests/test_adapter_registry.py` | 18 | test | — | — |
| `tests.test_artifacts` | `tests/test_artifacts.py` | 18 | test | — | — |
| `tests.test_chromatix_adapter` | `tests/test_chromatix_adapter.py` | 323 | test | — | Tests for the M_WAVE_CHROMATIX adapter (chromatix_adapter.py). |
| `tests.test_fdtdx_adapter` | `tests/test_fdtdx_adapter.py` | 242 | test | — | Tests for the forward-only M_EM_FDTDX adapter. |
| `tests.test_fmmax_adapter` | `tests/test_fmmax_adapter.py` | 344 | test | — | Tests for the ``M_RCWA_FMMAX`` (FMMAX RCWA) adapter. |
| `tests.test_graph_validation` | `tests/test_graph_validation.py` | 98 | test | — | — |
| `tests.test_optiland_adapter` | `tests/test_optiland_adapter.py` | 252 | test | — | Tests for the M_RAY_OPTILAND adapter (Optiland 0.6.0). |
| `tests.test_registry` | `tests/test_registry.py` | 14 | test | — | — |
| `tests.test_sax_adapter` | `tests/test_sax_adapter.py` | 331 | test | — | Tests for the M_CIRCUIT_SAX adapter (src/multiscale_optics_agent/adapters/sax_adapter.py). |

## B. Non-Python assets

| asset | bytes | referenced_by_python |
| --- | --- | --- |
| `benchmarks/manifest.yaml` | 1706 | — |
| `examples/graphs/chromatix_smoke.yaml` | 605 | — |
| `examples/graphs/fdtdx_smoke.yaml` | 706 | — |
| `examples/graphs/fmmax_smoke.yaml` | 722 | — |
| `examples/graphs/optiland_smoke.yaml` | 798 | — |
| `examples/graphs/ray_to_wave.yaml` | 1203 | tests.test_graph_validation |
| `examples/graphs/sax_smoke.yaml` | 713 | — |
| `schemas/artifact.schema.json` | 2748 | scripts.export_schemas |
| `schemas/coupler.schema.json` | 8011 | scripts.export_schemas |
| `schemas/graph.schema.json` | 6146 | scripts.export_schemas |
| `schemas/model.schema.json` | 8303 | scripts.export_schemas |
| `schemas/provenance.schema.json` | 2300 | scripts.export_schemas |
| `src/multiscale_optics_agent/registry/couplers.yaml` | 10887 | multiscale_optics_agent.registry.loader |
| `src/multiscale_optics_agent/registry/models.yaml` | 16870 | multiscale_optics_agent.registry.loader |

## C. Executable entry points [EXEC]

Every entry point below was executed through `./run.sh`; full output in
`docs/audit/logs/m01_entrypoints.txt`.

| Entry point | Command | Result |
|---|---|---|
| `run.sh` | `./run.sh python -c ...` | exit 0 (needs a TTY; see blocker above) |
| `Makefile: test` | `./run.sh pytest -q` | exit 0 — 52 passed, 2 xfailed, 1 xpassed |
| `Makefile: validate` | `./run.sh python scripts/validate_package.py` | exit 0 — "Validated 8 models, 10 couplers, all YAML files, and all example graphs." |
| `Makefile: list-models` | `./run.sh python -m multiscale_optics_agent.cli list-models` | exit 0 — 8 models |
| `Makefile: list-couplers` | `./run.sh python -m multiscale_optics_agent.cli list-couplers` | exit 0 — 10 couplers |
| `Makefile: clean` | not run | destructive; skipped deliberately |
| `scripts/validate_package.py` | see above | exit 0 |
| `scripts/export_schemas.py` | `./run.sh python scripts/export_schemas.py` | exit 0 — rewrote all 5 schemas, **`git status` clean afterwards ⇒ no schema drift** |
| `scripts/check_context_sync.py` | `./run.sh python scripts/check_context_sync.py` | exit 0 (extended under CHE-7) |
| `multiscale-optics` console script | declared in `pyproject.toml` `[project.scripts]` | not run directly; the equivalent `python -m ...cli` path was run [READ] |
| `examples/graphs/chromatix_smoke.yaml` | `cli validate` | exit 0 — GRAPH_VALID |
| `examples/graphs/fdtdx_smoke.yaml` | `cli validate` | exit 0 — GRAPH_VALID |
| `examples/graphs/fmmax_smoke.yaml` | `cli validate` | exit 0 — GRAPH_VALID |
| `examples/graphs/optiland_smoke.yaml` | `cli validate` | exit 0 — GRAPH_VALID |
| `examples/graphs/ray_to_wave.yaml` | `cli validate` | exit 0 — **warning** `GRADIENT_PATH_NOT_FULLY_VERIFIED` (expected: C_RAY_TO_WAVE + both models declare unverified derivatives) |
| `examples/graphs/sax_smoke.yaml` | `cli validate` | exit 0 — GRAPH_VALID |

`Makefile` targets invoke bare `pytest` / `python`, i.e. they assume host execution
and contradict the container-only rule in `AGENTS.md` [READ]. They work only when
invoked as `./run.sh make <target>`, which is not documented anywhere.

## D. Test-suite attribution [EXEC]

Baseline `./run.sh pytest -q`: **52 passed, 2 xfailed, 1 xpassed**. Nothing is
skipped and nothing fails. The three non-plain outcomes are deliberate upstream
locks, not defects in this repository:

| Test | Outcome | Attribution |
|---|---|---|
| `test_fdtdx_adapter.py::test_wavelength_gradient_matches_finite_difference_lock` | xfail | known `fdtdx==0.6.2` bug: `jax.grad` w.r.t. source wavelength returns exact 0.0 against a large finite-difference estimate |
| `test_fdtdx_adapter.py::test_permittivity_gradient_raises_concretization_error_lock` | xfail | known `fdtdx==0.6.2` bug: `place_objects()` is not traceable (`ConcretizationTypeError`) |
| `test_sax_adapter.py::test_gradient_through_assembled_circuit_not_yet_verified` | xpass (`strict=False`) | single-point gradient agreement that is real but insufficient to declare `derivative.verified=true` |

**No test covers the ray-to-wave coupler numerically.** `C_RAY_TO_WAVE` is exercised
only as a registry contract by `tests/test_graph_validation.py` via
`examples/graphs/ray_to_wave.yaml` [EXEC].

## E. Lint state [EXEC]

`./run.sh ruff check --output-format concise src scripts tests` reports **7
pre-existing errors** unrelated to M0 work:

- `src/multiscale_optics_agent/cli.py:72` E501
- `src/multiscale_optics_agent/core/graph.py:152,308` E501
- `src/multiscale_optics_agent/core/graph.py:360` B009
- `src/multiscale_optics_agent/core/graph.py:369` B905, RUF007
- `tests/test_graph_validation.py:1` I001

So `ruff check` is **not** currently clean on the tree; anyone adding a lint gate must
either fix these seven or scope the gate. Not fixed here (out of scope for CHE-5).

## F. Duplication and overlap findings

1. **No duplicated adapters.** [EXEC] The five adapters
   (`optiland`, `chromatix`, `fmmax`, `fdtdx`, `sax`) are distinct modules of
   599–669 lines each, each imported by exactly one test module and by nothing else.
   `adapters/registry.py` discovers them by `pkgutil` walk, which is why the import
   graph shows them as `test-only` rather than `unreferenced` [READ].
2. **`couplers/base.py` is dead weight today.** [EXEC] 44 lines defining the
   `Coupler` interface, imported by nothing — no test, no adapter, no CLI. There is
   no coupler implementation in the repository at all.
3. **Two artifact-description layers coexist.** [READ] `core/artifacts.py`
   (`ArtifactRecord`, 30 lines) describes arrays generically, while `core/specs.py`
   (258 lines) carries the `ArtifactKind` enum used by the registry. CHE-8/M2's
   `RayBundle`/`WavefrontSamples`/`ComplexField`/`PSF` contracts must be reconciled
   with both; neither is currently a typed physical artifact.
4. **`agents/`, `evaluation/`, `couplers/` are placeholders.** [EXEC]
   `agents/__init__.py` and `evaluation/__init__.py` are 1-line placeholders;
   `evaluation/checks.py` (30 lines) is imported by nothing.
5. **`benchmarks/` is documentation only.** [EXEC] Four READMEs plus
   `manifest.yaml`; no Python, no executable benchmark, and `manifest.yaml` is not
   read by any code.
6. **78 dangling `CLAUDE.md section N` citations across 33 files.** [EXEC]
   `CLAUDE.md` is now a single `@AGENTS.md` line with no numbered sections, so every
   one of these citations resolves to nothing. 20 of them are in `src/` and `tests/`
   (`chromatix_adapter.py`, `fmmax_adapter.py`, `optiland_adapter.py`,
   `sax_adapter.py`, `registry/models.yaml`, four test modules); the rest are in
   `knowledge/`. Also dangling: `AGENT_KNOWLEDGE_BASE.md` (13 files),
   `docs/SOLVER_AND_COUPLER_CATALOG.md` (5), `docs/ARCHITECTURE.md` (2),
   `docs/BENCHMARK_SPECIFICATION.md` (1) — none of these files exist.
   This is a shared finding with CHE-6 and is the largest single cleanup item in M0.

## G. Where is `ray_wave` / `ray_ewave`? [EXEC]

**Neither exists in this repository, in any form.** A repository-wide grep for
`ray_wave|ray_ewave|raywave` (excluding this audit's own files) returns five hits,
all of them prose or metadata:

| Hit | Nature |
|---|---|
| `src/multiscale_optics_agent/registry/couplers.yaml:32` | `tags: [ray_wave, pupil, cross_framework]` on `C_RAY_TO_WAVE` |
| `src/multiscale_optics_agent/registry/models.yaml:61-62` | a note deferring a "ray_wave/ray_ewave bridge audit" |
| `AGENTS.md:127` | the rule to treat such code as untrusted |
| `linear/BACKLOG_RAY_WAVE.md:31` | a proposed characterization issue |

Consequence for planning: **CHE-6/M2's "characterize the existing ray_wave
implementations" has no local subject.** The implementation lives in the external
repository `https://github.com/jiazhou-cheng/raywave-tracing`, which is not vendored,
not pinned, not a dependency in `pyproject.toml`, and not installed in the image
[EXEC — no `raywave` distribution is importable]. That issue must first decide how
the external code enters this repository (submodule, vendored copy, or pinned
dependency).

## H. Registry state relevant to the slice [EXEC]

`cli list-couplers` returns 10 couplers. The relevant ones:

- `C_RAY_TO_WAVE`: `wavefront_samples → complex_field`, `derivative.mode:
  finite_difference`, `verified: false`, `lossy: true`, `framework: internal`,
  invariants `phase_reference_consistency`, `pupil_power_consistency`.
- `C_FIELD_TO_PSF`: `complex_field → psf`, `native_autodiff`.
- **There is no `C_WAVE_TO_RAY`.** The only registered coupler pointing in the
  wave→ray direction is `C_GENERALIZED_SNELL` (`phase_profile → ray_bundle`).
  See the CHE-6 report for the documentation side of this gap.

`cli list-models` returns 8 models, including `M_RAY_OPTILAND` (pytorch) and
`M_WAVE_CHROMATIX` (jax) — the PyTorch/JAX boundary the project treats as
forward-only.

## I. Keep in active context (ray-to-wave slice)

Files a coding session for M1/M2 should actually load:

| Path | Why |
|---|---|
| `AGENTS.md` | canonical rules |
| `docs/context/CURRENT_SCOPE.md` | the canonical graph for this slice |
| `docs/context/RAY_WAVE_VERTICAL_SLICE.md` | execution plan |
| `src/multiscale_optics_agent/core/specs.py` | artifact kinds + registry schema |
| `src/multiscale_optics_agent/core/artifacts.py` | array record type |
| `src/multiscale_optics_agent/core/errors.py` | structured failure contract |
| `src/multiscale_optics_agent/adapters/base.py` | adapter interface |
| `src/multiscale_optics_agent/adapters/optiland_adapter.py` | ray model adapter |
| `src/multiscale_optics_agent/adapters/chromatix_adapter.py` | wave model adapter |
| `src/multiscale_optics_agent/couplers/base.py` | coupler interface (currently unused) |
| `src/multiscale_optics_agent/registry/{models,couplers}.yaml` | the contracts being implemented |
| `tests/test_optiland_adapter.py`, `tests/test_chromatix_adapter.py` | the behavior to preserve |
| `examples/graphs/ray_to_wave.yaml` | the only C_RAY_TO_WAVE graph |
| `knowledge/solvers/optiland/**`, `knowledge/solvers/chromatix/**` | pinned conventions and probes |

Everything under `benchmarks/`, `examples/graphs/{fmmax,fdtdx,sax}_smoke.yaml`, and
`knowledge/solvers/{fmmax,fdtdx,jax_fem,sax}/**` is not needed by this slice.

## J. Archive candidates (recommendation only — decided in CHE-8)

- `MIGRATION_PLAN.md` and `VALIDATION_REPORT.md` — superseded, see CHE-6.
- `linear/PROJECT_SETUP.md`, `linear/BACKLOG_RAY_WAVE.md`, `linear/ISSUE_TEMPLATE.md`
  — superseded by the live Linear project.
- `knowledge/solver_cards/` — routing-card layer duplicated by the nested cards.
- `knowledge/solvers/{fmmax,fdtdx,jax_fem,sax}/**` and their smoke graphs —
  off-scope, but note their adapters and 3 of the 19 probes are *live and passing*,
  so archiving the knowledge packs without moving the adapters would break the
  `knowledge/` ↔ `adapters/` pairing. **Do not archive these unless the adapters and
  their tests move with them.**

Nothing under `src/` is recommended for archiving in v0.1: all five adapters are
covered by passing tests.

## K. What this audit did NOT verify

- The console script `multiscale-optics` was not invoked by name [READ only].
- `Makefile clean` was not run.
- `mypy` was not run (not required by CHE-5; note `VALIDATION_REPORT.md` claims it
  was unavailable, which is now untrue for `ruff` at least).
- No numerical or physical property of any adapter was checked; this audit only
  establishes what runs and what is wired to what.
- Semantics of `C_RAY_TO_WAVE` were not characterized — that is M2 work, and per
  section G it has no local implementation to characterize yet.
