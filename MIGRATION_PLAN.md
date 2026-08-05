# Documentation and Context Migration Plan

## Target Structure

```text
AGENTS.md                         # canonical, always loaded
CLAUDE.md                         # only: @AGENTS.md
CONTEXT_MANIFEST.yaml             # machine-readable loading policy
docs/context/                     # concise, task-linkable current context
knowledge/                        # solver/coupler cards and validated probes; retrieval only
docs/archive/2026-07-research/    # long historical specs; never startup context
linear/                           # issue template, project workflow, proposed backlog
scripts/check_context_sync.py     # verifies the shared entrypoint
```

## Move, Do Not Delete First

Archive the current long documents before deciding what to remove:

- `PROJECT_PLAN.md`
- `PAPER_INTRODUCTION.md`
- `RELATED_WORK_MATRIX.md`
- `BENCHMARK_SPECIFICATION.md`
- `ARCHITECTURE.md`
- `SOLVER_AND_COUPLER_CATALOG.md`
- `AGENT_KNOWLEDGE_BASE.md`
- the old long `CLAUDE.md`

Keep paper and benchmark documents available for research, but do not auto-load them into coding sessions.

## Merge Repeated Operational Rules

- Merge `workflow.md` and `pr-standard.md` into `AGENTS.md` and the Linear issue template.
- Keep one version of each rule.
- Delete or archive the originals only after repository references are updated.

## Replace Tool-Specific Duplication

- Add root `AGENTS.md` as the canonical text.
- Replace root `CLAUDE.md` with exactly `@AGENTS.md`.
- Do not maintain a separate `CODEX.md`.
- Put tool-specific personal preferences in user-local configuration, not the repository contract.

## Progressive Disclosure

Always load:

- `AGENTS.md`.
- The current Linear issue.

Load only when linked:

- One current-scope document.
- Relevant code/tests.
- The selected solver/coupler card.
- Minimal pinned probes and failure guide.

Never load by default:

- The full solver catalog.
- The full benchmark suite.
- The paper introduction and related-work matrix.
- Cards for solvers not used by the issue.

## Migration Sequence

1. Commit the new shared context files without deleting old docs.
2. Run `scripts/check_context_sync.py`.
3. Create the Linear project and seed the first audit/probe issues.
4. Update README links to distinguish current context, retrieval knowledge, and archive.
5. Move long documents to the archive in a mechanical PR.
6. Run one Codex session and one Claude Code session on the same read-only audit issue.
7. Compare which files each agent loaded and correct the context manifest.
8. Only then delete redundant workflow/PR files or stale copies.

## Acceptance Test for the Context Migration

- Codex reports that it loaded root `AGENTS.md`.
- Claude Code `/memory` reports `CLAUDE.md` and imported `AGENTS.md`.
- Both agents independently restate the same current milestone, non-goals, workflow, and scientific rules.
- Both open the same Linear issue and use the same acceptance criteria.
- Neither reads archived broad documents unless the issue links them.
