# CHE-67 — Archive the outdated tests; make the tutorials on-demand

**Issue:** CHE-67
**Date:** 2026-08-19
**Consumes:** `docs/archive/2026-08-testing/test_inventory.md` / `.json` (CHE-64) for every count
and runtime quoted here. **This issue ran no tests**, so nothing below is a new
measurement of test *behavior*; the numbers are collection counts (measured) and
CHE-64 runtimes (reused).

## What changed

| Group | Where it lives | How to run it | Tests |
|---|---|---|---:|
| Default active suite | `tests/` | `./run.sh pytest -q` | 565 |
| On-demand tutorials | `tests_tutorial/` | `make test-tutorial` | 60 |
| Archived | `archive/tests/gen1/` | not runnable; unarchive first | 288 |
| GPU (unchanged) | `tests/`, `-m gpu` | `./run.sh --gpu pytest -q -m gpu` | 33 |

The GPU tests are counted inside the 565: they are collected by the default suite
and skip there. That is CHE-60's design and CHE-67 did not touch it.

905 tests before → 565 in the default suite. 288 archived, 60 moved to the
tutorial suite; the arithmetic closes (565 + 288 + 60 = 913 against 905 collected
before, the 8 difference being this issue's new `tests/test_suite_layout.py`).

## Why location, not markers

The suite was already marker-tiered (CHE-53), and `-m "not tutorial"` already
existed (CHE-64). Marker-tiering is the wrong tool for this particular job:

- **A marker is a filter you have to remember.** Forgetting `-m "not tutorial"`
  costs 33 minutes; forgetting `-m "not benchmark"` runs a superseded protocol.
  The default has to be safe with no flags, because the default is what an agent
  or a new contributor types.
- **`-m` in `addopts` is not a fix**: a command-line `-m` *replaces* it, so
  `pytest -m chromatix` would have silently re-selected the Chromatix tutorials.
- **A marker cannot express "archived"**, only "deselected." `@pytest.mark.skip`
  leaves a test collected, imported and reported — visibly rotting rather than
  visibly retired.

`testpaths = ["tests"]` gives the property actually wanted: opting *in* requires
naming a directory, and no filter can be forgotten because there is no filter.

## The three-layer archive guard, and why one layer was not enough

Measured during this issue, in order:

1. `testpaths = ["tests"]` — a bare `pytest` collects 565 tests and no archived
   file. ✔
2. But `pytest .` collected **625** tests: an explicit path argument overrides
   `testpaths`, so the tutorial suite came back. Fixed by adding both `archive`
   and `tests_tutorial` to `norecursedirs`; `pytest .` now collects 565. Crucially,
   `norecursedirs` is *not* applied to a path named on the command line, so
   `pytest tests_tutorial` still collects its 60 — the exact asymmetry wanted.
3. But naming an archived *file* still walked past both: `pytest
   archive/tests/gen1/tests/test_m1_protocol.py` collected and ran it. Fixed by
   `archive/tests/gen1/conftest.py`, which raises `pytest.UsageError` from
   `pytest_collection_modifyitems` if any collected item lives under the
   generation. The abort is session-wide, so a mixed
   `pytest tests archive/...` selection fails loudly instead of quietly running
   the half that still exists.

Layer 3 is the one that matters in practice, because the paths in
`docs/archive/2026-08-testing/test_inventory.md` all still read `tests/...` — that document was
generated before this archival, and copy-pasting a node id out of it is exactly
the accident layers 1 and 2 do not cover.

A fourth, unplanned layer turned out to exist: five archived files do
`from conftest import load_probe_expected`, which resolves to the *generation's*
`conftest.py` when collected in place, so they fail to import there. That is why
the documented restore is a `git mv` back under `tests/` rather than "run it from
the archive."

## Verification (all measured; no test bodies were executed except the new guards)

| Check | Command | Result |
|---|---|---|
| Default collection | `pytest -q --collect-only` | 565 collected, 4.4 s |
| No archived/tutorial/out-of-scope test in it | grep over the collected node ids | only benign name matches: two `test_m3_singlet_ref_*` and one `test_m1_opd_*` in *active adapter* files, one `test_importing_sax_...` in the GPU suite, three in the new layout guard |
| Fast subset | `-m "not slow"` | 547 collected (18 `slow` deselected). **Not timed** — CHE-53's ~31 s was measured when this expression selected 499 tests |
| Old Tier A expression still runs | `-m "not slow and not benchmark and not fmmax and not fdtdx and not sax"` | 547 — same set, as intended |
| Archived markers select nothing | `-m "benchmark or fmmax or fdtdx or sax"` | 0 collected, 565 deselected |
| Root-directory sweep | `pytest --collect-only .` | 565 (was 625 before the `norecursedirs` fix) |
| Tutorial suite discoverable | `pytest --collect-only tests_tutorial` | 60 |
| ...and marker-selectable | `pytest --collect-only tests_tutorial -m tutorial` | 60 — the same 60, so `tutorial` names the whole suite |
| Archived dir | `pytest --collect-only archive` | aborted by the guard; no tests ran |
| Archived file, real run | `pytest -q archive/tests/gen1/tests/test_m1_protocol.py` | `no tests ran`, usage error |
| Mixed selection | `pytest -q tests archive/.../test_m3_convergence.py` | aborted; nothing ran |
| GPU in a dedicated session | `pytest -q -m gpu` (CPU container) | 33 skipped, 532 deselected — skipped, not executed |
| GPU alongside a non-GPU test | `pytest -q tests/test_gpu_environment.py tests/test_artifacts.py` | 1 passed, 8 skipped ("needs a dedicated session") |
| New guards pass | `pytest -q tests/test_suite_layout.py` | 8 passed |
| New guards actually trip | fault injection: a tutorial-marked file, a copied tutorial file, and an unarchived `test_m1_protocol.py` under `tests/` | 2, 2 and 2 failures respectively; all reverted |
| Lint/types | `ruff check`, `ruff format --check`, `mypy` on the new and changed files | clean; the 9 pre-existing `tests/` findings and the 3 in the tutorial files are unchanged by this issue (verified against `HEAD`) |

**Not run, deliberately:** the tutorial suite (33 min), any archived test, the GPU
suite on a real device, and the default suite end-to-end. The last one is the
material gap in this issue's evidence: the reorganization is a file move plus
config, and no active test imports an archived or tutorial module (checked: the
only cross-file import under `tests/` is
`test_optiland_canonical_prescriptions.py`'s `from conftest import ...`), but
"collection succeeds" is weaker than "the suite passes."

## Consequences worth stating

- **288 tests' worth of behavior is now unguarded.** The M3 pupil-to-focus
  tolerances, the M1 protocol shape, the M2 coupler contract and the L1/L2
  benchmark fingerprints are no longer checked by anything. The reports in
  `benchmarks/` still *claim* those results; nothing now re-verifies them. Treat a
  dependence on one of those behaviors as a reason to unarchive, not as an
  assumption.
- **The out-of-scope adapters are no longer unguarded — they are gone.** This
  entry recorded FMMAX and FDTDX as at risk of rotting silently against
  dependency updates. CHE-87 (2026-08-22) resolved that the other way: both
  adapters were deleted together with their registry entries, example graphs,
  knowledge packs, pytest markers and dependency pins, as was JAX-FEM, which
  never had an adapter. Their archived tests remain in `archive/tests/gen1/` and
  are now archived tests of deleted code, so unarchiving one is not a route back
  — a restoration is a fresh scoped integration. Intent and the findings worth
  keeping: `benchmarks/roadmap.md`.
  SAX was already in that state: CHE-72 (2026-08-20) deleted the
  integration, its tests, its knowledge pack and its `sax`/`klujax` pins, because
  klujax pinned `jax_platform_name='cpu'` at import and silently disabled the GPU
  process-wide. A future SAX integration should be built fresh rather than
  unarchived. The measurement rows above are
  left as recorded on 2026-08-19.
- **The tutorial gate now depends on a human cadence.** Nothing fails if nobody
  runs `make test-tutorial` for a month; a pin change can therefore land
  unverified. If that turns out to matter, the fix is a scheduled run, not
  re-adding 33 minutes to every PR.
- **`test_inventory.{md,json}`, `test_audit.md`, `test_runtime_audit.md`
  and `tier_restructure.md` were not regenerated.** They are CHE-52/53/64 evidence
  and describe the tree as it was; every `tests/...` path in them that this issue
  moved is now stale. Regenerating would require re-running the profiler
  (2642 s), which this issue is explicitly not doing.

## Follow-ups this leaves open

1. Establish the replacement benchmark suite (the reason the old one is archived).
2. A dedicated GPU validation pass on a CUDA device.
3. Decide whether the 18 remaining `slow` tests (164 s) should also become
   on-demand or nightly.
4. Regenerate the per-test inventory once the suite stops moving.
