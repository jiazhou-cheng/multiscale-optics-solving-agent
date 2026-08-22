# M0.3 — Reviewed Archive Plan

**Issue:** CHE-8

**Date:** 2026-08-11

**Execution issue:** CHE-9 (M0.4)
**Decision:** archive five stale or superseded documents. Do not archive source,
tests, solver knowledge packs, solver cards, example graphs, or benchmarks.

This plan is the decision boundary: CHE-8 moves, renames, and deletes nothing.
Archiving means a history-preserving `git mv` into `docs/archive/`, never deletion.

## Evidence and decision rule

The plan consumes, rather than redoes, these findings:

- `docs/audit/M0_code_audit.md`, section J, recommends the two stale root reports
  and three superseded local Linear documents as candidates, while explicitly
  retaining all `src/` adapters because they are covered by passing tests.
- `docs/audit/M0_docs_audit.md`, section H, gives the document-specific reasons
  below. Sections B/C show that the nested solver packs and their probes remain
  live evidence; section G names the material needed by the current slice.
- `docs/audit/M0_docs_audit.md`, section D, directs CHE-9 to add the archive path
  to `CONTEXT_MANIFEST.yaml` only once the directory exists.

A candidate is approved only when it is both stale/superseded and has no live
code, test, script, Makefile, or graph dependency. Merely being off-scope is not
enough.

## Approved moves

| Current path | Destination under `docs/archive/` | Reason and audit citation | Imports/references and handling |
|---|---|---|---|
| `MIGRATION_PLAN.md` | `docs/archive/2026-07-research/MIGRATION_PLAN.md` | Its migration steps 1–3 are complete and most named source documents no longer exist. Its target archive layout remains useful history. M0.2 §§A,H; M0.1 §J. | No live-path hit. Moving the document itself preserves its target-layout record. |
| `VALIDATION_REPORT.md` | `docs/archive/2026-07-research/VALIDATION_REPORT.md` | Its 2026-07-29 claims (8 tests, solvers absent, ruff absent) are superseded by the M0.1 baseline and M0.2 probe replay. M0.2 §§A,H; M0.1 §§C–E,J. | No live-path hit. `docs/archive/README.md` will identify it as historical, not current validation evidence. |
| `linear/PROJECT_SETUP.md` | `docs/archive/2026-08-linear/PROJECT_SETUP.md` | Its project name, states, and labels do not match the live Linear project. M0.2 §§A,H; M0.1 §J. | No live-path hit. MCP snippets remain recoverable in the archived copy. |
| `linear/BACKLOG_RAY_WAVE.md` | `docs/archive/2026-08-linear/BACKLOG_RAY_WAVE.md` | The live Linear M0–M4 issue list supersedes this proposed backlog. M0.2 §§A,H; M0.1 §§G,J. | No live-path hit. Linear remains the task source of truth. |
| `linear/ISSUE_TEMPLATE.md` | `docs/archive/2026-08-linear/ISSUE_TEMPLATE.md` | Its template differs from the live project's required eight-section format. M0.2 §§A,H; M0.1 §J. | No live-path hit. The live Linear issue format remains authoritative. |

## Executable reference evidence

Run from the repository root before any move:

```bash
rg -n --hidden --glob '!docs/audit/**' --glob '!.git/**' \
  'MIGRATION_PLAN\.md|VALIDATION_REPORT\.md|linear/(PROJECT_SETUP|BACKLOG_RAY_WAVE|ISSUE_TEMPLATE)\.md' .

./run.sh grep -RInE \
  'MIGRATION_PLAN\.md|VALIDATION_REPORT\.md|linear/(PROJECT_SETUP|BACKLOG_RAY_WAVE|ISSUE_TEMPLATE)\.md' \
  src tests scripts examples/graphs \
  --include='*.py' --include='*.yaml' --include='*.md'

./run.sh python docs/audit/probes/audit_import_graph.py
```

Observed on 2026-08-11: both path searches returned no matches. The import-graph
probe completed successfully and inventories only Python modules plus registered
YAML assets; none of the five approved documents is imported or loaded. Therefore
no import, test, script, Makefile target, or live graph resolves to an approved
archive path, and no reference migration is required.

The broader search did find active references to `knowledge/solver_cards/*.yaml`
from nested solver documentation and `docker/requirements.txt`. Those six cards
are therefore rejected from this plan despite M0.2's duplicate classification.

## Material that stays active

The following minimum 62-file set remains in place for the Optiland → ray-to-wave
coupler → Chromatix → PSF slice:

- Repository context/execution (8): `AGENTS.md`, `CLAUDE.md`,
  `CONTEXT_MANIFEST.yaml`, `README.md`, `pyproject.toml`, `run.sh`,
  `docker/Dockerfile`, `docker/requirements.txt`.
- Current scope (3): every file in `docs/context/`.
- Implementation contracts/path (17): package root; the `adapters/` package files
  for base, registry, Optiland, and Chromatix; the `core/` package files for
  artifacts, errors, graph, and specs; the `couplers/` package and base interface;
  and the `registry/` package, loader, `models.yaml`, and `couplers.yaml`.
- Direct tests (7): `tests/conftest.py`, adapter-registry, artifact, graph,
  Optiland-adapter, Chromatix-adapter, and registry tests.
- Executable graph (1): `examples/graphs/ray_to_wave.yaml`.
- Scientific knowledge (26): `knowledge/README.md`,
  `knowledge/source_manifest.yaml`, and all 24 files in
  `knowledge/solvers/{optiland,chromatix}/`.

Also explicitly retained: all off-scope solver packs and their paired adapters/tests
(live, passing evidence), all smoke graphs (valid and cheap to keep), all benchmark
material (future M3 specification), and all six flat solver cards (actively
referenced). Nothing under `src/`, `tests/`, `scripts/`, `examples/`, `knowledge/`,
or `benchmarks/` is approved for archiving by this plan.

## Ordered M0.4 command sequence

Every relocation must use `git mv` in exactly this order:

```bash
mkdir -p docs/archive/2026-07-research docs/archive/2026-08-linear
git mv MIGRATION_PLAN.md docs/archive/2026-07-research/MIGRATION_PLAN.md
git mv VALIDATION_REPORT.md docs/archive/2026-07-research/VALIDATION_REPORT.md
git mv linear/PROJECT_SETUP.md docs/archive/2026-08-linear/PROJECT_SETUP.md
git mv linear/BACKLOG_RAY_WAVE.md docs/archive/2026-08-linear/BACKLOG_RAY_WAVE.md
git mv linear/ISSUE_TEMPLATE.md docs/archive/2026-08-linear/ISSUE_TEMPLATE.md
```

M0.4 then adds `docs/archive/README.md`, adds `docs/archive/` under
`reference_not_startup` in `CONTEXT_MANIFEST.yaml`, and updates `README.md` so it
no longer presents the archived local Linear files as current. `AGENTS.md` needs no
change.

## M0.4 verification and rollback

Before and after the moves, run `./run.sh pytest -q` and require identical summary
counts. Then run:

```bash
./run.sh python scripts/check_context_sync.py
./run.sh python scripts/validate_package.py
rg -n 'docs/archive|MIGRATION_PLAN\.md|VALIDATION_REPORT\.md|linear/(PROJECT_SETUP|BACKLOG_RAY_WAVE|ISSUE_TEMPLATE)\.md' \
  src tests scripts Makefile examples/graphs
git diff --stat -M <M0.4-parent>..<M0.4-archive-commit>
git log --follow --oneline -- docs/archive/2026-07-research/MIGRATION_PLAN.md
git log --follow --oneline -- docs/archive/2026-08-linear/PROJECT_SETUP.md
```

The grep must have no live references other than checks that deliberately reject
archive references. Rollback is the non-destructive inverse commit:

```bash
git revert <M0.4-archive-commit>
```

CHE-9 must replace the placeholder with its actual archive commit SHA in its work
log. Before commit, the equivalent local rollback is the reverse `git mv` sequence;
no `rm` is permitted.

## Approval gate

Human approval is required before CHE-9 begins. It is recorded in the CHE-8 work
log: on 2026-08-11 the user explicitly authorized Codex to complete M0.3 and proceed
directly through M0.4 without an additional approval prompt. That authorization
approves only the five moves listed above.
