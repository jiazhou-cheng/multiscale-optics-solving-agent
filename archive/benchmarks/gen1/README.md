# Archived benchmark generation `gen1` — 2026-08-22, CHE-88

The superseded Level-1 and Level-2 benchmark suites, together with the four
production modules that only they consumed. Preserved, **not runnable**, and
not deleted.

`archive/tests/gen1/` archived the *tests* of this generation on 2026-08-19
(CHE-67). This directory archives the benchmark code those tests were about,
which had continued to live in `benchmarks/` and — for the four modules below —
inside the shipped package.

## What moved here

### Benchmark suites

| From | Why |
| -- | -- |
| `benchmarks/level1/L1-RAY-01/` | gen1. Its tests were archived by CHE-67 as "superseded benchmark suite"; it encodes the `M1-BASELINE-CPU-V2` protocol. |
| `benchmarks/level1/L1-WAVE-01/` | gen1, `M1-BASELINE-CPU-V1`. Case 3 (high-NA vectorial) was already `status: blocked` on a defective upstream `high_na_ff_lens`: refining only the pupil sampling moves the focal scale by 10×, while the independent oracle converges to 2e-14. |
| `benchmarks/level2/L2-COUPLER-01/` | the M2 coupler protocol. Its physics is covered by the active `tests/test_coupler_round_trip.py`, `test_coupler_gradient.py` and `test_curvature_bound.py`. |
| `benchmarks/verify_m1_independence.py` | consumes all three retired harnesses; nothing else. |

`benchmarks/level2/L2-PSF-01/` did **not** move. It is the one live benchmark:
its `1.0e-3 fft_oracle_intensity_relative_l2` gate is unmet and explicitly
carried into M4, with CHE-48 reopened to decompose the residual.

### Modules that left the shipped package

Each ends in `_adapter.py` and none declares a `MODEL_ID`, so the old filename
scan in `adapters/registry.py` imported all three on every adapter lookup while
they registered nothing. CHE-87 replaced that scan with an explicit map, which
removed the import; CHE-88 removes the files.

| From | Lines | Consumers |
| -- | -- | -- |
| `adapters/optiland_benchmark_adapter.py` | 459 | L1-RAY-01's runners, `verify_m1_independence.py` |
| `adapters/chromatix_benchmark_adapter.py` | 150 | L1-WAVE-01's `evaluate.py`, `verify_m1_independence.py` |
| `adapters/chromatix_scaling_adapter.py` | 197 | L1-WAVE-01's `run_scaling.py`, `verify_m1_independence.py` |
| `evaluation/m1_bundle.py` | 480 | L1-RAY-01 and L1-WAVE-01's `run_all.py` — **and see below** |

`m1_bundle.py` was **not** uniformly dead, which is why it moved last.
`VOLATILE_KEYS` and `_strip_volatile` — the reproducibility-fingerprint
projection — were imported from it by both live Level-2 benchmarks, as private
names across a package boundary. They were promoted into
`core/provenance.py` as public API *before* this module moved, and the L2-PSF-01
scientific fingerprint was measured identical either side of the promotion
(`b073a4616c0fda245dace0ef77ac46f4ca7efe065bef7db839b5652fc9cc0dab`). Everything
else in the file is gen1 branch-bundle machinery.

## What is now unguarded

Stated plainly, in the same spirit as `archive/tests/gen1/README.md`:

* **The M1 baseline protocol is no longer executable from the active tree.**
  `M1-BASELINE-CPU-V1` and `V2` — fresh-process isolation, no cross-engine
  import, the required artifact set, structured blockers instead of fabricated
  values — are contracts that nothing now runs. `M1_BASELINE_REPORT.md` still
  records what they produced.
* **The M1 independence check is no longer run.** It was the executable evidence
  that the ray and wave baselines do not import each other or a coupler. That
  property is now asserted by nothing.
* **The L1-WAVE-01 analytic oracles** — the exact Helmholtz plane-wave
  eigenmode case, the paraxial Fresnel/Fourier case, and the independent float64
  angular-spectrum comparison — are not exercised. `evaluation/asm_oracle.py`
  remains active and tested, so the ASM comparison itself is not lost; the
  benchmark's own three-case structure is.
* **The L1-RAY-01 analytic ray cases** — manufactured free-space propagation,
  ideal paraxial focusing, an Edmund Optics catalog lens — are not exercised
  here. `tests/test_optiland_canonical_prescriptions.py` covers adjacent ground
  but is not the same benchmark.
* **The M2 coupler protocol's bundle-level gates** are not run. The underlying
  physics is guarded by three active coupler tests, which is why this suite was
  archived rather than kept; the *protocol's* per-bundle tolerances and
  fingerprint rules are not.

If an issue depends on one of these, unarchive the relevant file rather than
assuming it still holds.

## What archival does not claim

Archiving is a scope decision, not a verdict on correctness. Everything here was
in its last working state when it moved, and nothing was rewritten,
re-toleranced, or weakened to make the active tree green. The scientific claims
these suites backed are not retracted: `M1_BASELINE_REPORT.md`,
`M2_COUPLER_REPORT.md` and the milestone reports in `benchmarks/` still record
them, and `benchmarks/manifest.yaml` still points at the M1 exit report.

One exception to "unmodified": the two live Level-2 benchmarks were repointed at
`core/provenance.py` *before* L2-COUPLER-01 was archived, so the archived copy
carries the new import rather than a broken one. That is a deliberate departure
from freezing it exactly — a frozen file with an import that cannot resolve is
worse history than a frozen file that still reads correctly.

## Unarchiving

An explicit, reviewable `git mv` back, justified in a Linear issue — never a
flag:

```bash
git mv archive/benchmarks/gen1/benchmarks/L1-RAY-01 benchmarks/level1/
git mv archive/benchmarks/gen1/src/optiland_benchmark_adapter.py src/multiscale_optics_agent/adapters/
```

Then expect real work. These files were frozen against the repository as it was
on 2026-08-22, and three things have moved under them since:

1. `adapters/registry.py` is an explicit map, so restoring a `*_adapter.py`
   filename no longer registers anything — and for these three that is correct,
   since none declares a `MODEL_ID`. Do not add one to make discovery work; they
   are benchmark harnesses, not adapters, and the naming is the original mistake.
2. `evaluation/m1_bundle.py`'s fingerprint helpers now live in
   `core/provenance.py` under public names. Restoring the module would create
   two copies of the projection, which is exactly the drift CHE-88 removed.
3. The FMMAX, FDTDX and JAX-FEM packages are no longer installed (CHE-87), so
   any path through `verify_m1_independence.py` that touched them will not run.

Say in the issue why the behavior is required again.
