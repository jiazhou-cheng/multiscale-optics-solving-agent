# Historical Archive

Material preserved because it records how the repository got here, not because
it is still true. Every document below described the tree accurately when it was
written and does not now.

Archived material is **reference-only**. Agents must not load it as startup
context; consult it on demand for provenance. Everything was moved with
`git mv`, so nothing was deleted or rewritten, and `git log --follow` still
works. Stale paths inside these files are left stale on purpose — repairing them
would edit a historical record, and `scripts/validate_package.py` skips the
archive trees for exactly that reason.

## `2026-07-research/` — archived 2026-08-11 (CHE-9, plan approved in CHE-8)

The original context-migration plan and an old validation snapshot. The snapshot
is historical evidence, not the current test or solver-validation status.

## `2026-08-linear/` — archived 2026-08-11 (CHE-9)

Local planning documents superseded by the live Linear project and its issue
format.

## `2026-07-m0/` — archived 2026-08-22 (CHE-86, Phase 1 of CHE-84)

The July M0 audit: `M0_code_audit.md`, `M0_docs_audit.md`, `M0_archive_plan.md`,
four probe scripts, and twelve raw logs. Formerly `docs/audit/`.

**Why superseded:** it describes an implementation state that no longer exists.
Its findings were acted on and then overtaken — the two modules it flagged as
unreferenced (`agents/`, `evaluation/checks.py`) were deleted by the same issue
that archived it.

The probe scripts under `probes/` still exist but are not maintained; the
current per-module coverage map lives in
`docs/architecture/cleanup_baseline.md`.

## `2026-08-testing/` — archived 2026-08-22 (CHE-86, Phase 1 of CHE-84)

Five test-suite documents, formerly under `docs/testing/`. Four of them already
opened by announcing their own obsolescence.

| File | Superseded by | Why |
| -- | -- | -- |
| `test_audit.md` | CHE-67 | the PB1 audit of a suite whose files have since moved |
| `test_inventory.md` / `.json` (1,043 lines) | CHE-67 | every `tests/...` path in it predates the archival; CHE-67 records explicitly that regenerating would cost 2,642 s of profiling and declined |
| `tier_restructure.md` | CHE-67, CHE-72 | Tier A/B/C was replaced by the five location-based groups, and Tier C no longer exists |
| `test_runtime_audit.md` | CHE-67 | the CHE-64 per-test runtime and memory measurements, indexed by pre-archival paths |
| `pb3_shrink_review.md` | — | a closed one-off review |

`docs/testing/` keeps only `test_archive.md` (the CHE-67 record, which is the
current explanation of the five-group split) and `gpu_environment.md` (live
setup, actively maintained).

The CHE-64 runtime and memory *numbers* are still the best available
measurements — nothing has re-profiled the suite since. They are cited from
AGENTS.md at the archived path deliberately: the data stands, the paths do not.
