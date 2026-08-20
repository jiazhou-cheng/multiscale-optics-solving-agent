# PB2 — Tier A/B/C Restructure

**Issue:** CHE-53 (PB2), milestone M3.5
**Date:** 2026-08-18
**Consumes:** `docs/testing/test_audit.md` (CHE-52/PB1) — this document does not
re-derive the inventory, it applies it.

## What PB1 handed to this issue

PB1 found zero `REWRITE`/`MERGE`/`ARCHIVE`/`DELETE` candidates (see PB1 §4), so
this issue's "act on PB1's disposition table" step is: assign every
`KEEP_REQUIRED` test to Tier A, every `KEEP_TARGETED` test to a Tier B subsystem
group (excluded from Tier A), and every `KEEP_FULL_REGRESSION` test to Tier C
only (also excluded from Tier A). No test content, fixture, or assertion changed
in this issue — only `pyproject.toml`'s marker registry and per-file
`pytestmark`/`@pytest.mark.*` annotations.

## Marker taxonomy (confirmed against PB1's inventory, not assumed)

CHE-53's own example list (`core, optiland, chromatix, coupler, integration, gpu,
slow, benchmark, regression`) does not match PB1's actual findings in three ways,
documented here per CHE-53's own instruction to confirm rather than assume:

1. **No `core` marker was added.** PB1 found the "no external solver, cheap"
   files (`test_adapter_registry.py`, `test_artifacts.py`, `test_registry.py`,
   `test_run_sh.py`, `test_context_sync.py`, `test_graph_validation.py`, plus most
   of `test_m2_coupler_protocol.py`/`test_coupler_contracts.py`/etc. once they
   also get a `coupler` tag) to be the *majority* of the suite. Tagging ~30 files
   with a marker whose only purpose is to be the logical complement of the other
   markers was rejected as a needless abstraction (`AGENTS.md`: "don't add
   abstractions beyond what the task requires"). Tier A is instead defined as an
   **exclusion** expression (below) over the markers that do carry information —
   which files are expensive or out of current-milestone scope. This is a
   deliberate deviation from CHE-53's example list; flagging it per `AGENTS.md`'s
   "surface any conflict" rule.
2. **`jax`/`torch`/`integration` (pre-existing) were kept, unchanged, alongside
   new subsystem markers, not replaced.** PB1 §3 found them applied
   inconsistently with cost/scope (e.g. `test_chromatix_adapter.py` is
   `jax`+`integration`-marked and current-milestone-required; `test_fdtdx_adapter.py`
   is also `jax`+`integration`-marked and out-of-scope). Reusing them as the Tier A
   exclusion criterion would have wrongly excluded the current-milestone adapter
   contracts. New markers (`optiland`, `chromatix`, `fmmax`, `fdtdx`, `sax`)
   separate "which solver" from the pre-existing "requires an optional install."
3. **No `gpu` or `regression` marker was added.** Nothing in the suite currently
   requires a real GPU device (the container runs CPU-only per `AGENTS.md`'s "Do
   not assume GPU access"), so a `gpu` marker would mark nothing. A `regression`
   marker (for the probe-record-comparison half of files like
   `test_m3_convergence.py`) was considered and rejected: PB1 found those files'
   live and probe-reading tests to have near-identical (cheap) cost, so splitting
   them by marker would add bookkeeping without changing which tier anything
   lands in.

Markers actually added to `pyproject.toml` (full text in that file):

| Marker | Meaning | Files/tests carrying it |
|---|---|---|
| `optiland` | exercises the real Optiland engine | 164 tests across 11 files |
| `chromatix` | exercises the real Chromatix engine | 114 tests across 8 files (1 test in `test_ray_to_wave.py`, tagged individually, not module-wide — see note below) |
| `coupler` | exercises `C_RAY_TO_WAVE`/`C_WAVE_TO_RAY` coupler physics or protocol | 371 tests across 20 files |
| `fmmax` | FMMAX RCWA adapter (out of current milestone scope) | 7 tests, `test_fmmax_adapter.py` |
| `fdtdx` | FDTDX EM adapter (out of current milestone scope) | 5 tests, `test_fdtdx_adapter.py` |
| `sax` | SAX photonic-circuit adapter (out of current milestone scope) | 12 tests, `test_sax_adapter.py` |
| `benchmark` | full-scale Level-1/Level-2 milestone benchmark bundle reproduction | 85 tests across 9 files (all of `tests/benchmarks/` plus `test_optiland_ray_benchmark.py`) |
| `slow` | individually expensive characterization/convergence test, applied at test granularity where a file is otherwise fast | 19 tests across 6 files |

A test can and does carry more than one marker (e.g. `test_m3_pupil_to_focus.py`
is `coupler` + `optiland` + `chromatix`; `test_ray_to_wave.py` is `coupler`
module-wide, with `chromatix` on only its one test that does
`pytest.importorskip("chromatix")` inside the test body rather than at module
level — tagging the whole file would have mislabeled the 19 other tests in it
that need no external solver at all).

"Graph/integration behavior" (the fourth Tier B group CHE-53's example names) is
covered by `test_graph_validation.py` and `test_ray_to_wave_node.py`
(tagged `coupler`+`optiland`); both are already cheap and sit in Tier A, so no
separate marker was needed to make them independently runnable — `-m coupler` or
direct file selection already does that. No dedicated graph-only marker was added
because nothing in that category is expensive enough to need excluding from
Tier A or grouping away from it.

## Tier definitions

- **Tier A** (`./run.sh pytest -q -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"`):
  everything not individually flagged expensive (`slow`), not a full benchmark
  reproduction (`benchmark`), and not an out-of-current-scope solver adapter
  (`fmmax`/`fdtdx`/`sax`). This is a superset of PB1's `KEEP_REQUIRED` rows.
- **Tier B** (per-subsystem, run independently): `-m optiland`, `-m chromatix`,
  `-m coupler`, `-m fmmax`, `-m fdtdx`, `-m sax`, `-m benchmark`, `-m slow`. These
  overlap by design (e.g. `-m optiland` includes both fast Tier-A-resident
  Optiland tests and the heavy `tests/benchmarks/` Optiland reproduction) — a
  developer touching the Optiland adapter runs `-m optiland` to see everything
  that exercises it, regardless of tier.
- **Tier C** (`./run.sh pytest -q`, no marker filter): unchanged, the full 627-test
  suite. Nothing was dropped from it — every marker addition is additive
  metadata, no test was skipped, deleted, or moved to a different file.

## Runtime, measured via `./run.sh` (not estimated)

| Selector | Tests | Time |
|---|---|---|
| Tier A (`not slow and not benchmark and not fmmax and not fdtdx and not sax`) | 478 passed, 21 skipped (F3, pre-existing) | **31.08s** |
| `-m slow` | 19 passed | 158.03s |
| `-m benchmark` | 85 passed | 410.20s |
| `-m fmmax or fdtdx or sax` | 21 passed, 2 xfailed, 1 xpassed | 40.86s |
| `-m coupler` | 350 passed, 21 skipped | 159.25s |
| `-m optiland` | 164 passed | 282.17s (includes the 3 heaviest benchmark fixtures, which are also optiland-tagged) |
| `-m chromatix` | 114 passed | 407.45s (includes the 3 heaviest benchmark fixtures, which are also chromatix-tagged) |
| Tier C, full suite, no filter | 627 collected | ~635s (PB1 baseline; unchanged — no test content was altered) |

**Tier A result: 31.08s, comfortably under the ≤3-minute (180s) gate — by a
factor of ~6, with no problem-size shrinking.** This confirms PB1 §7's
projection. Consequently:

## PB3 (CHE-54) is not required to shrink anything to meet the gate

CHE-54's own acceptance criteria anticipate this: "if any test cannot be
cheapened without losing its claim, it is reclassified to Tier B and documented
as such rather than force-fit into Tier A" — that reclassification is exactly
what this issue already did (marker-based, not runtime-hack-based). PB3's
remaining scope, per its own acceptance criteria, is documentation: state in
`AGENTS.md` (or a doc it references) which of the four command tiers
(required / targeted-by-subsystem / full-regression / the pre-existing
`optiland`/`torch`/`jax` extras) applies to each situation. No fixture or
parametrization shrinking is needed or attempted here.

## Verification

```bash
./run.sh --no-build pytest --collect-only -q                                        # 627 tests, unchanged from PB1
./run.sh --no-build pytest -q --durations=0 -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"   # Tier A: 31.08s
./run.sh --no-build pytest -q -m optiland                                            # Tier B: Optiland group
./run.sh --no-build pytest -q -m chromatix                                           # Tier B: Chromatix group
./run.sh --no-build pytest -q -m coupler                                             # Tier B: coupler group
./run.sh --no-build pytest -q -m "fmmax or fdtdx or sax"                             # Tier B: out-of-scope solver groups
./run.sh --no-build pytest -q -m benchmark                                           # Tier B: full benchmark bundles
./run.sh --no-build pytest -q                                                        # Tier C: full regression, unchanged
```

## Confirmation that PB1's disposition was applied

PB1 proposed 0 `REWRITE`/`MERGE`/`ARCHIVE`/`DELETE` rows, so there is nothing to
apply beyond tier/marker assignment. Every `KEEP_TARGETED` and
`KEEP_FULL_REGRESSION` row in PB1 §5 now carries a marker that excludes it from
Tier A (`slow`, `benchmark`, `fmmax`, `fdtdx`, or `sax`); every `KEEP_REQUIRED`
row remains selected by the Tier A exclusion expression by construction (it
carries none of those markers). PB1's Finding F3 (missing
`m3r_sensor_handoff.json` probe record) is unchanged by this issue — the 21
affected tests still skip under every tier, as shown in the Tier A run above;
regenerating that record remains the recommended follow-up issue from PB1 §6,
out of scope for both PB2 and PB3.

CHE-62 audited those 21 skips rather than leaving the count unexplained: only the
record-backed assertions in `tests/test_m3r_sensor_handoff.py` depend on the
record, the skip message now names the cause and the regeneration command, and
regeneration is tracked in CHE-63. See
`benchmarks/M3_M3_5_CLEANUP_DISPOSITION.md` item 1. Note that Tier A's skip
total has since grown from 21 to 54: CHE-60/CHE-61 added 33 `gpu`-quarantined
skips, which are designed behavior and not a debt.
