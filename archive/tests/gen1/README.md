# Archived test generation `gen1` — 2026-08-19, CHE-67

276 tests in 21 files, removed from pytest collection but **not deleted**. They
are preserved here as historical evidence of what earlier milestones actually
verified. Directory layout mirrors the original tree, so every file's path here
is its old path with `archive/tests/gen1/` prefixed.

Nothing in this directory runs. See "Unarchiving" below for the only way back.

## Why each group was archived

| Group | Why | Restored by |
|---|---|---|
| Superseded benchmark suite (`benchmark`-marked) | A replacement benchmark baseline is being defined; the old L1/L2/M1 bundle reproductions encode the *previous* protocol, so keeping them green would gate current work on frozen requirements. Not modified to match the new design — that would destroy the evidence. | the follow-up benchmark ticket, if any old case is still wanted |
| Out-of-scope solver adapters (`fmmax`/`fdtdx`) | FMMAX and FDTDX are outside the current milestone scope per `AGENTS.md`. | a Linear issue that brings that solver back into scope |
| Outdated milestone suites (M1/M2/M3/M3R) | Frozen assumptions and recorded evidence from completed milestones. Tier classification does not exempt them: several were Tier A and cheap, but "cheap" is not "still the right requirement". | a ticket that intentionally needs that behavior again |

### Superseded benchmark suite — 85 tests, 410 s

| file | tests | s |
|---|---:|---:|
| `tests/benchmarks/test_l1_ray_scaling.py` | 4 | 24.6 |
| `tests/benchmarks/test_l1_wave_accuracy.py` | 20 | 63.9 |
| `tests/benchmarks/test_l1_wave_scaling.py` | 4 | 79.5 |
| `tests/benchmarks/test_l2_coupler_bundle.py` | 17 | 0.4 |
| `tests/benchmarks/test_l2_psf_bundle.py` | 10 | 0.3 |
| `tests/benchmarks/test_m1_bundle_projection.py` | 5 | 0.0 |
| `tests/benchmarks/test_m1_report.py` | 16 | 0.0 |
| `tests/benchmarks/test_m1_reproducibility.py` | 4 | 234.7 |
| `tests/test_optiland_ray_benchmark.py` | 5 | 6.6 |

### Out-of-scope solver adapters — 12 tests, 31 s

| file | tests | s |
|---|---:|---:|
| `tests/test_fdtdx_adapter.py` | 5 | 16.6 |
| `tests/test_fmmax_adapter.py` | 7 | 14.2 |

`tests/test_sax_adapter.py` (12 tests, 2.8 s) was archived here by CHE-67 and
then **deleted** by CHE-72, which removed the SAX integration entirely along with
its `klujax` dependency — klujax pinned `jax_platform_name='cpu'` at import time
and the test harness had to undo that on every GPU run. Deleting rather than
keeping it is the one exception to "archiving preserves" on this page, and it is
deliberate: the package is no longer installed, so the file could not be restored
into a working state. `git log` before CHE-72 still has it if the text is ever
wanted. Per CHE-72, a future SAX integration should be rebuilt fresh against the
requirements at that time rather than resurrected.

### Outdated milestone suites — 179 tests, 24 s

| file | tests | s |
|---|---:|---:|
| `tests/test_m1_protocol.py` | 5 | 5.3 |
| `tests/test_m2_coupler_protocol.py` | 9 | 0.1 |
| `tests/test_m3_slice_protocol.py` | 21 | 1.5 |
| `tests/test_m3_pupil_to_focus.py` | 14 | 2.8 |
| `tests/test_m3_psf_measurement.py` | 27 | 1.8 |
| `tests/test_m3_psf_verification.py` | 19 | 0.6 |
| `tests/test_m3_convergence.py` | 27 | 0.3 |
| `tests/test_m3_off_axis_handoff.py` | 21 | 0.2 |
| `tests/test_m3_quadrature_weight.py` | 9 | 0.0 |
| `tests/test_m3r_sensor_handoff.py` | 27 | 10.9 |

Counts and runtimes are the CHE-64 measurements in `docs/testing/test_inventory.json`
(one profiled session; not re-measured by CHE-67, which ran no tests).

## What archival does *not* claim

Archiving is a **scope** decision, not a verdict on correctness. Every file here
was passing when it was archived. Nothing was rewritten, re-toleranced, or
skipped to make the current suite green — the files are byte-identical to their
last active revision, so `git log --follow` and `git diff` against the commit
before CHE-67 both still work.

The scientific claims these tests backed are *not* retracted by moving them. The
narrative evidence they were written to defend still lives in `benchmarks/`
(`M1_BASELINE_REPORT.md`, `M2_COUPLER_REPORT.md`, `M3_SLICE_REPORT.md`,
`M3_5_EXIT_REPORT.md`, `M3_9R_SENSOR_HANDOFF_REPORT.md`) and in
`knowledge/`. What archival *does* mean: from now on those claims are
**unguarded** — a regression in M3 pupil-to-focus behavior will no longer be
caught by CI. If a current issue depends on one of these behaviors, unarchive the
relevant file rather than assuming it still holds.

## Unarchiving

Unarchiving is an explicit, reviewable change — never a flag:

```bash
git mv archive/tests/gen1/tests/test_m3_pupil_to_focus.py tests/
./run.sh pytest -q tests/test_m3_pupil_to_focus.py
```

The `git mv` is not ceremony. Five of these files do
`from conftest import load_probe_expected`, which in place resolves to *this
directory's* `conftest.py` and fails to import (measured in CHE-67); they only
find `tests/conftest.py` again once they are back under `tests/`.

Then expect to do real work: the file was frozen against the repository as it was
on 2026-08-19, so a stale API or a stale recorded expectation is the normal
outcome, and fixing that is part of whichever issue needs the test back. Say in
that issue why the behavior is required again.

Removing this generation's `conftest.py`, editing `norecursedirs`, or widening
`testpaths` is **not** unarchiving. Those only delete the guard;
`tests/test_suite_layout.py` fails if any of them happens.
