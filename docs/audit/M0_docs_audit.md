# M0.2 — Knowledge and Documentation Audit

**Issue:** CHE-6 (M0 — Repository Audit and Archive)
**Date:** 2026-08-11
**Rule:** read-only. No file under `knowledge/`, `docs/context/`, or `linear/` was
moved, renamed, edited, or deleted by this audit.

**Scope reminder used throughout:** the v0.1 active scope is Optiland, Chromatix, and
the **ray-wave coupling layer in both directions** (`C_RAY_TO_WAVE` and
`C_WAVE_TO_RAY`). The initial executable forward pipeline
(`ray → C_RAY_TO_WAVE → Chromatix → PSF`) exercises only `C_RAY_TO_WAVE`;
wave-to-ray material is **not** classified as off-scope for being unexercised.

## How this audit was produced

```bash
./run.sh python docs/audit/probes/audit_docs_inventory.py   # inventory + term scan
./run.sh python docs/audit/probes/classify_docs.py          # rule-based classification
./run.sh python docs/audit/probes/run_solver_probes.py      # replay all 19 probes
./run.sh diff -u knowledge/solver_cards/optiland.yaml knowledge/solvers/optiland/solver_card.yaml
./run.sh diff -u knowledge/solver_cards/chromatix.yaml knowledge/solvers/chromatix/solver_card.yaml
./run.sh python scripts/check_context_sync.py
grep -rn -i "ray_wave|ray_ewave|raywave" ...                # host-side text search only
```

Raw output: `docs/audit/logs/m02_*.txt`. The classification rules themselves live in
`docs/audit/probes/classify_docs.py`, so the table below is reproducible rather than
hand-asserted.

## Headline counts

| Metric | Value |
|---|---|
| Documentation files classified | 125 |
| canonical | 54 |
| off-scope reference | 60 |
| duplicate-of-X | 8 |
| stale | 3 |
| unclassified | 0 |
| Byte-identical duplicate groups | **0** |
| Solver probes replayed | 19 |
| Probes matching their recorded `expected/*.json` | **19 / 19** |
| Files naming `ray→wave` | 12 |
| Files naming `wave→ray` | **0** |

## A. Classification table

| path | classification | rationale |
| --- | --- | --- |
| `AGENTS.md` | canonical | declared canonical_static_context in CONTEXT_MANIFEST.yaml |
| `CLAUDE.md` | canonical | one-line pointer to AGENTS.md; enforced by check_context_sync |
| `CONTEXT_MANIFEST.yaml` | canonical | machine-readable loading policy |
| `MIGRATION_PLAN.md` | stale | describes a context migration whose steps 1-3 are already done and lists eight documents to archive (PROJECT_PLAN.md, PAPER_INTRODUCTION.md, ...) that no longer exist in this repository; its 'Target Structure' remains the only written source for docs/archive/, so extract that before archiving |
| `README.md` | canonical | repository entry doc; rewritten under CHE-7 to carry the container-only rule |
| `VALIDATION_REPORT.md` | stale | dated 2026-07-29; claims 8 tests, solvers not installed, ruff/mypy unavailable. Current evidence: 52-test baseline, all 19 solver probes MATCH, ruff runs |
| `benchmarks/README.md` | off-scope reference | benchmark suite is M3, not M0-M2 |
| `benchmarks/level1/README.md` | off-scope reference | benchmark suite is M3 |
| `benchmarks/level2/README.md` | off-scope reference | benchmark suite is M3 |
| `benchmarks/level3/README.md` | off-scope reference | benchmark suite is M3 |
| `benchmarks/manifest.yaml` | off-scope reference | benchmark suite is M3 |
| `docs/audit/logs/m01_entrypoints.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m01_export_schemas.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m01_import_graph.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m01_pytest_baseline.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m02_classification.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m02_docs_inventory.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m02_legacy_terms.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m02_solver_probes.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m02_stale_crossrefs.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m05_context_sync.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/logs/m05_pytest_after.txt` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/probes/audit_docs_inventory.py` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/probes/audit_import_graph.py` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/probes/classify_docs.py` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/audit/probes/run_solver_probes.py` | canonical | M0 audit output produced by CHE-5/CHE-6 |
| `docs/context/CURRENT_SCOPE.md` | canonical | current-scope doc for the ray/wave slice; names the canonical graph |
| `docs/context/MODULE_GRANULARITY.md` | canonical | granularity rules referenced by AGENTS.md |
| `docs/context/RAY_WAVE_VERTICAL_SLICE.md` | canonical | execution plan for the active slice |
| `examples/graphs/chromatix_smoke.yaml` | off-scope reference | smoke graph for a solver outside the v0.1 slice |
| `examples/graphs/fdtdx_smoke.yaml` | off-scope reference | smoke graph for a solver outside the v0.1 slice |
| `examples/graphs/fmmax_smoke.yaml` | off-scope reference | smoke graph for a solver outside the v0.1 slice |
| `examples/graphs/optiland_smoke.yaml` | off-scope reference | smoke graph for a solver outside the v0.1 slice |
| `examples/graphs/ray_to_wave.yaml` | canonical | the only graph exercising C_RAY_TO_WAVE; loaded by tests/test_graph_validation.py |
| `examples/graphs/sax_smoke.yaml` | off-scope reference | smoke graph for a solver outside the v0.1 slice |
| `knowledge/README.md` | canonical | explains the knowledge/ layout; still accurate for solver_cards/ and papers/ |
| `knowledge/solver_cards/chromatix.yaml` | duplicate-of knowledge/solvers/chromatix/solver_card.yaml | routing-card subset of the nested validation card (in-scope); verified divergent by machine diff, different key sets |
| `knowledge/solver_cards/fdtdx.yaml` | duplicate-of knowledge/solvers/fdtdx/solver_card.yaml | routing-card subset of the nested validation card (off-scope); verified divergent by machine diff, different key sets |
| `knowledge/solver_cards/fmmax.yaml` | duplicate-of knowledge/solvers/fmmax/solver_card.yaml | routing-card subset of the nested validation card (off-scope); verified divergent by machine diff, different key sets |
| `knowledge/solver_cards/jax_fem.yaml` | duplicate-of knowledge/solvers/jax_fem/solver_card.yaml | routing-card subset of the nested validation card (off-scope); verified divergent by machine diff, different key sets |
| `knowledge/solver_cards/optiland.yaml` | duplicate-of knowledge/solvers/optiland/solver_card.yaml | routing-card subset of the nested validation card (in-scope); verified divergent by machine diff, different key sets |
| `knowledge/solver_cards/sax.yaml` | duplicate-of knowledge/solvers/sax/solver_card.yaml | routing-card subset of the nested validation card (off-scope); verified divergent by machine diff, different key sets |
| `knowledge/solvers/chromatix/api_minimal_examples.md` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/capability_notes.md` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/conventions.md` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/expected/gradient_probe.json` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/expected/import_probe.json` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/expected/propagation_probe.json` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/failure_guide.md` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/probes/gradient_probe.py` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/probes/import_probe.py` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/probes/propagation_probe.py` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/solver_card.yaml` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/chromatix/source_manifest.yaml` | canonical | in-scope solver knowledge pack (chromatix) |
| `knowledge/solvers/fdtdx/api_minimal_examples.md` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/capability_notes.md` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/conventions.md` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/expected/gradient_probe.json` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/expected/import_probe.json` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/expected/propagation_probe.json` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/failure_guide.md` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/probes/gradient_probe.py` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/probes/import_probe.py` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/probes/propagation_probe.py` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/solver_card.yaml` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fdtdx/source_manifest.yaml` | off-scope reference | fdtdx is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/api_minimal_examples.md` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/capability_notes.md` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/conventions.md` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/expected/fresnel_oracle_probe.json` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/expected/gradient_probe.json` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/expected/import_probe.json` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/failure_guide.md` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/probes/fresnel_oracle_probe.py` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/probes/gradient_probe.py` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/probes/import_probe.py` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/solver_card.yaml` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/fmmax/source_manifest.yaml` | off-scope reference | fmmax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/api_minimal_examples.md` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/capability_notes.md` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/conventions.md` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/expected/import_probe.json` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/expected/mesh_probe.json` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/expected/solver_failure_probe.json` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/failure_guide.md` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/probes/import_probe.py` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/probes/mesh_probe.py` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/probes/solver_failure_probe.py` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/solver_card.yaml` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/jax_fem/source_manifest.yaml` | off-scope reference | jax_fem is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/optiland/api_minimal_examples.md` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/capability_notes.md` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/conventions.md` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/expected/gradient_probe.json` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/expected/import_probe.json` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/expected/raytrace_probe.json` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/failure_guide.md` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/probes/gradient_probe.py` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/probes/import_probe.py` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/probes/raytrace_probe.py` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/solver_card.yaml` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/optiland/source_manifest.yaml` | canonical | in-scope solver knowledge pack (optiland) |
| `knowledge/solvers/sax/api_minimal_examples.md` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/capability_notes.md` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/conventions.md` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/expected/circuit_probe.json` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/expected/component_model_probe.json` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/expected/gradient_probe.json` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/expected/import_probe.json` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/failure_guide.md` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/probes/circuit_probe.py` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/probes/component_model_probe.py` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/probes/gradient_probe.py` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/probes/import_probe.py` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/solver_card.yaml` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/solvers/sax/source_manifest.yaml` | off-scope reference | sax is out of scope for v0.1; retrieval-only |
| `knowledge/source_manifest.yaml` | canonical | authoritative upstream source links |
| `linear/BACKLOG_RAY_WAVE.md` | duplicate-of Linear project description | a proposed backlog superseded by the live project's M0-M4 issue list |
| `linear/ISSUE_TEMPLATE.md` | duplicate-of Linear project description | the Linear project mandates a different eight-section issue format (Goal/Context/Acceptance Criteria/Out of Scope/Dependencies/Likely Files Affected/Verification Commands/Required Deliverables); this file's template differs |
| `linear/PROJECT_SETUP.md` | stale | describes a project named 'Ray-Wave Vertical Slice' with workflow states and labels that do not match the live Linear project or team states (Backlog/Todo/In Progress/In Review/Done/Canceled/Duplicate, no labels defined) |
| `schemas/artifact.schema.json` | canonical | generated by scripts/export_schemas.py; regeneration produced no diff |
| `schemas/coupler.schema.json` | canonical | generated by scripts/export_schemas.py; regeneration produced no diff |
| `schemas/graph.schema.json` | canonical | generated by scripts/export_schemas.py; regeneration produced no diff |
| `schemas/model.schema.json` | canonical | generated by scripts/export_schemas.py; regeneration produced no diff |
| `schemas/provenance.schema.json` | canonical | generated by scripts/export_schemas.py; regeneration produced no diff |

## B. `knowledge/solver_cards/*.yaml` vs `knowledge/solvers/*/solver_card.yaml`

**Resolution: they serve different purposes and are both live — do not merge blindly,
but the flat layer is the weaker one.**

Evidence (machine diff, `docs/audit/logs/m02_solver_card_diffs.txt`):

- No pair is identical. Sizes differ roughly 2× (optiland: 41 vs 79 lines; chromatix:
  30 vs 63; jax_fem: 41 vs 112).
- The key sets are disjoint in intent. Flat cards carry routing keys
  (`agent_should_use_for`, `agent_should_not_assume`, `install`, `required_probes`,
  `knowledge_pack`, `benchmarks`). Nested cards carry validation keys
  (`adapter_location`, `devices_tested`, `dtypes_observed`, `validated_probe_ids`,
  `limitations_and_failure_modes`, `not_yet_probed`, `cost_scaling`).
- The nested optiland card states the relationship explicitly in its header comment:
  *"This supplements, and does not replace, `knowledge/solver_cards/optiland.yaml`
  (the flat routing card format used by the current registry docs)."*

Two problems follow:

1. **Both layers disagree with reality on validation status.** All 12 cards (6 flat +
   6 nested) say `validation_status: unvalidated`, yet the nested cards' own
   `validated_probe_ids` list passing probes dated 2026-07-30, and this audit
   re-ran all 19 probes today with **19/19 MATCH**. Per `knowledge/README.md`,
   `unvalidated` means "planning only" — that label is now under-claiming the
   available evidence. Whoever owns M1 should decide the correct status value rather
   than leaving a permanently-wrong field.
2. **The flat cards' header comment points at `docs/AGENT_KNOWLEDGE_BASE.md`
   section 2.1**, which does not exist (13 files reference it). See section E.

Recommendation for CHE-8: keep the nested cards as canonical; treat
`knowledge/solver_cards/` as an archive candidate **only** after confirming nothing
in the registry docs reads the routing keys (nothing in Python does — verified by the
CHE-5 import graph: no asset in `knowledge/` is referenced from Python at all).

## C. Probe replay status

All 19 probes were executed in the container and their stdout compared key-by-key to
the recorded `expected/*.json`.

| solver | probe | status |
|---|---|---|
| chromatix | `import_probe` | MATCH |
| chromatix | `propagation_probe` | MATCH |
| chromatix | `gradient_probe` | MATCH |
| optiland | `import_probe` | MATCH |
| optiland | `raytrace_probe` | MATCH |
| optiland | `gradient_probe` | MATCH |
| fmmax | `import_probe` | MATCH |
| fmmax | `gradient_probe` | MATCH |
| fmmax | `fresnel_oracle_probe` | MATCH |
| fdtdx | `import_probe` | MATCH |
| fdtdx | `propagation_probe` | MATCH |
| fdtdx | `gradient_probe` | MATCH |
| sax | `import_probe` | MATCH |
| sax | `circuit_probe` | MATCH |
| sax | `component_model_probe` | MATCH |
| sax | `gradient_probe` | MATCH |
| jax_fem | `import_probe` | MATCH |
| jax_fem | `mesh_probe` | MATCH |
| jax_fem | `solver_failure_probe` | MATCH |

Note on method: the three `jax_fem` probes print an ASCII banner on import before
their JSON, so a strict whole-stream `json.loads` fails on them. The runner tolerates
a non-JSON preamble; without that tolerance they report `NON_JSON`, which would be a
harness artifact, not a probe failure.

**Consequence:** the `knowledge/solvers/**` packs are live, reproducible evidence, not
stale prose. None of them should be archived on suspicion of rot. The off-scope ones
are off-scope by *topic*, not by *decay*.

## D. `CONTEXT_MANIFEST.yaml` versus reality

Declared paths, checked for existence:

| Declared | Exists |
|---|---|
| `AGENTS.md` | yes |
| `CLAUDE.md` | yes |
| `docs/context/` | yes |
| `knowledge/` | yes |
| `docs/archive/` | **no** |

`docs/archive/` was the only missing path. Under CHE-7 it was removed from the
manifest and replaced with a comment instructing CHE-9 (M0.4) to re-add it when the
directory is actually created, so the new strict existence check can pass. (Prose
entries such as "selected solver/coupler cards" and "relevant source files" are not
paths and are correctly ignored by the checker.)

Top-level directories with **no** manifest entry at all — the manifest says nothing
about whether an agent should load them:

`benchmarks/`, `docker/`, `examples/`, `linear/`, `schemas/`, `scripts/`, `src/`,
`tests/`, `tmp_probes/`

`src/` and `tests/` are implicitly covered by the prose entries "relevant source
files" / "relevant tests". `benchmarks/`, `linear/`, and `examples/` are genuinely
unclassified by the loading policy and should be assigned a tier in CHE-8.

## E. Dangling documentation references (shared finding with CHE-5)

| Referenced target | Exists? | Files referencing it |
|---|---|---|
| `CLAUDE.md section <N>` | no — `CLAUDE.md` is one line, no sections | 33 files, 78 occurrences |
| `AGENT_KNOWLEDGE_BASE.md` | no | 13 |
| `docs/SOLVER_AND_COUPLER_CATALOG.md` | no | 5 |
| `docs/ARCHITECTURE.md` | no | 2 |
| `docs/BENCHMARK_SPECIFICATION.md` | no | 1 |

Of the 33 files citing `CLAUDE.md section N`, 20 hits are in `src/` and `tests/`
(all four solver adapters, `registry/models.yaml`, four test modules) and the rest
are in `knowledge/`. These citations were written against a long-form `CLAUDE.md`
that the context migration replaced with `@AGENTS.md`.

This is the largest single documentation-integrity problem in the repository. It is
**not** fixable by archiving: the citations sit inside live, passing code. Options
for CHE-8 to choose between: (a) restore the cited sections in `AGENTS.md` with
stable anchors and rewrite citations to point at them, (b) rewrite citations to name
the rule instead of a section number, or (c) recover the historical `CLAUDE.md` from
Git history into `docs/archive/` and repoint citations there. Not decided here.

## F. Ray-wave coupling documentation

This section deliberately separates the two coupler directions.

### F.1 `C_RAY_TO_WAVE` — documented as a contract, not as an implementation

| Source | What it contributes |
|---|---|
| `src/multiscale_optics_agent/registry/couplers.yaml` (lines 2–32) | the only formal contract: `wavefront_samples → complex_field`; `framework: internal`; `derivative.mode: finite_difference`, `verified: false`; `lossy: true`; requires `[wavelength, coordinates, optical_path_length, amplitude, polarization]`; provides `[wavelength, sample_pitch, coordinate_frame, phasor, polarization, normalization]`; invariants `phase_reference_consistency`, `pupil_power_consistency`; assumptions about pupil sampling and caustics |
| `docs/context/CURRENT_SCOPE.md` | places it in the canonical graph `M_RAY_OPTILAND → C_RAY_TO_WAVE → M_WAVE_CHROMATIX → C_FIELD_TO_PSF`; lists the open questions (which reference plane, which amplitude weight, how phase is referenced) |
| `docs/context/RAY_WAVE_VERTICAL_SLICE.md` | execution plan for the slice |
| `examples/graphs/ray_to_wave.yaml` | the only executable artifact touching it; validates with a `GRADIENT_PATH_NOT_FULLY_VERIFIED` warning |
| `AGENTS.md` | the untrusted-until-characterized rule and the artifact boundary list |

No document states the coupler's actual mathematics: no reconstruction kernel, no
phase-sign convention, no amplitude weighting rule, no interpolation scheme. The
registry entry is a promise about I/O, not a description of an algorithm.

### F.2 `C_WAVE_TO_RAY` — **no documentation exists at all**

A term scan across all 125 documentation files plus `src/` returns **zero**
occurrences of `C_WAVE_TO_RAY`, `wave_to_ray`, `wave2ray`, or `wave->ray`.

- There is no `C_WAVE_TO_RAY` entry in `registry/couplers.yaml` (the 10 registered
  couplers are `C_RAY_TO_WAVE`, `C_EIKONAL_TO_WAVE`, `C_NEAR_TO_FAR`,
  `C_CELL_TO_SURFACE`, `C_FIELD_TO_PSF`, `C_ABSORPTION_TO_HEAT`,
  `C_TEMPERATURE_TO_MATERIAL`, `C_FIELD_TO_MODE`, `C_SMATRIX_TO_CIRCUIT`,
  `C_GENERALIZED_SNELL`).
- The only registered coupler pointing wave→ray is `C_GENERALIZED_SNELL`
  (`phase_profile → ray_bundle`, `analytic`, `lossy: true`). Whether it is intended
  as the `C_WAVE_TO_RAY` slot or is an unrelated metasurface coupler is **not stated
  anywhere** — reported as unresolved, not assigned.

**This is a gap, not an archive candidate.** Because no wave-to-ray document exists,
there is nothing to keep in active context for that direction and nothing to archive;
the correct follow-up is to author the contract, which is outside M0.

### F.3 Legacy `ray_wave` / `ray_ewave` terminology — where it points

Five occurrences repository-wide (excluding this audit's own files):

| Location | Direction it describes | Verdict |
|---|---|---|
| `registry/couplers.yaml:32` — `tags: [ray_wave, pupil, cross_framework]` | attached to `C_RAY_TO_WAVE` | ray→wave |
| `AGENTS.md:127` — "treat `ray_wave`, `ray_ewave` … as untrusted" | unspecified | **unresolved** — names code that does not exist in this repository |
| `registry/models.yaml:61–62` — defers a "`ray_wave`/`ray_ewave` bridge audit" | unspecified; mentions synthesizing an explicit bridge | **unresolved** |
| `linear/BACKLOG_RAY_WAVE.md:31` | "semantics for `ray_wave`, `ray_ewave` and related code" | **unresolved** |

No occurrence distinguishes the two directions, and `ray_ewave` is never defined
anywhere. Per CHE-6's rule these are reported as unresolved rather than assigned to a
direction. Note also the CHE-5 finding: **no `ray_wave`/`ray_ewave` implementation
exists in this repository**; the code lives in the external, unpinned
`https://github.com/jiazhou-cheng/raywave-tracing`.

## G. Keep in active context

### G.1 Optiland (ray model)

`knowledge/solvers/optiland/solver_card.yaml`, `conventions.md`,
`capability_notes.md`, `api_minimal_examples.md`, `failure_guide.md`,
`source_manifest.yaml`, `probes/*`, `expected/*`

### G.2 Chromatix (wave model)

`knowledge/solvers/chromatix/solver_card.yaml`, `conventions.md`,
`capability_notes.md`, `api_minimal_examples.md`, `failure_guide.md`,
`source_manifest.yaml`, `probes/*`, `expected/*`

### G.3 `C_RAY_TO_WAVE`

`src/multiscale_optics_agent/registry/couplers.yaml` (the `C_RAY_TO_WAVE` block),
`examples/graphs/ray_to_wave.yaml`, `docs/context/RAY_WAVE_VERTICAL_SLICE.md`

### G.4 `C_WAVE_TO_RAY`

**Empty — no document exists.** Open question to resolve: whether
`C_GENERALIZED_SNELL` occupies this slot.

### G.5 Shared by both coupler directions

`AGENTS.md` (artifact boundary, granularity rules, scientific non-negotiables),
`docs/context/CURRENT_SCOPE.md`, `docs/context/MODULE_GRANULARITY.md`,
`CONTEXT_MANIFEST.yaml`, `knowledge/README.md`, `knowledge/source_manifest.yaml`

### G.6 Required by the executable forward path only

`ray → C_RAY_TO_WAVE → Chromatix → PSF` needs exactly: the Optiland pack (G.1), the
Chromatix pack (G.2), the `C_RAY_TO_WAVE` and `C_FIELD_TO_PSF` registry blocks, and
`examples/graphs/ray_to_wave.yaml`. Nothing else in `knowledge/` is required to run
the demo.

## H. Archive candidates — recommendation only

Decided in CHE-8; nothing is moved by this issue.

| Candidate | Why | Caveat |
|---|---|---|
| `MIGRATION_PLAN.md` | steps 1–3 already executed; lists 8 documents to archive that no longer exist | extract its "Target Structure" first — it is the only written source for the `docs/archive/2026-07-research/` layout |
| `VALIDATION_REPORT.md` | dated 2026-07-29; every quantitative claim (8 tests, solvers not installed, ruff unavailable) is contradicted by today's evidence | supersede with a dated pointer rather than deleting the record |
| `linear/PROJECT_SETUP.md` | workflow states and labels do not match the live team | keep the MCP setup snippets somewhere |
| `linear/BACKLOG_RAY_WAVE.md` | superseded by the live M0–M4 issue list | — |
| `linear/ISSUE_TEMPLATE.md` | the live project mandates a different 8-section format | reconcile, don't silently drop |
| `knowledge/solver_cards/` (6 files) | routing-subset duplicates of the nested cards; unreferenced from Python | check registry docs first |
| `knowledge/solvers/{fmmax,fdtdx,jax_fem,sax}/**` (48 files) | off-scope for v0.1 | **their adapters and tests are live and passing, and 10 of the 19 passing probes are theirs. Archiving the packs without the adapters breaks the pairing. Recommend retrieval-only, not archive, until the adapters move too.** |
| `examples/graphs/{fmmax,fdtdx,sax,chromatix,optiland}_smoke.yaml` | off-scope | all still validate; cheap to keep |
| `benchmarks/**` | M3 material | keep — it is the spec for M3, not dead weight |
| `knowledge/papers/` | **empty directory** | remove or populate |

**Explicitly not an archive candidate:** anything describing `C_WAVE_TO_RAY` — there
is nothing to archive, and the absence is a gap to fill, not material to retire.

## I. What this audit did NOT do

- No file was moved, renamed, edited, or deleted.
- No mathematical characterization of either coupler direction (M2 work).
- No claim that `C_RAY_TO_WAVE` and `C_WAVE_TO_RAY` are inverses; the question was
  not investigated.
- Prose quality, scientific correctness, and internal consistency of the
  `knowledge/solvers/**` documents were not reviewed — only their currency
  (probe replay) and their scope.
- The archive plan itself is CHE-8.
