# Benchmarks

The scientific evidence layer. Its unit is a **family**, not a task.

A `BenchmarkFamily` is a physical question with a declared parameter space, an
oracle and its independence, executable validity predicates, metrics, tolerances
with their bases, and negative controls. A `BenchmarkInstance` is one point in
that space with a stable fingerprint. Both live in
[`src/verification/families/`](../src/verification/families/); the pipeline they
sit in is

```
BenchmarkFamily / BenchmarkInstance
  -> GraphExecutor           emits an ExecutionRecord   (what happened)
  -> verify(...)             emits a VerificationResult (what it means)
  -> fixed evaluation, generated evaluation, agent scoring
```

[`docs/benchmark_design.md`](../docs/benchmark_design.md) defines the B0–B4
categories, the four separations they enforce, and what a family must declare.
This page says where the files are.

## What is here

| | |
| -- | -- |
| **Inventory** | [`inventory.yaml`](inventory.yaml) / [`INVENTORY.md`](INVENTORY.md) — every artifact in this tree classified as reusable evidence, candidate canonical case, or retired task layer, with the destination for each. `tests/test_benchmark_inventory.py` fails if a file in the tree is missing from it. |
| **Manifest** | [`manifest.yaml`](manifest.yaml) — the `characterizations:` block, which records which scientific tasks **cannot** become validation targets and why. The `levels:` block is gone; see the note in the file for where each piece of its content went. |
| **Protocols** | `protocols/` — frozen contracts holding tolerance derivations and the per-axis Nyquist and reference-plane decisions. They go only once the families express their content executably: CHE-106 (M1.1) deleted the M1 baseline protocol (`M1-BASELINE-CPU-V1`/`V2`) on exactly that ground, and `inventory.yaml`'s `deleted:` block records where each of its parts went. The M2 and M3 contracts remain. |
| **Probes** | `probes/` — one-off executable evidence behind card claims, and `probes/records/` the outputs. Records are **provenance, never oracles**; `probes/records/REGISTER.yaml` tracks which are provenance-stamped. |
| **Performance** | `perf/` — the M0.4 cost harness and its committed baselines, destined for the `B4-COST` family. |
| **Reports** | `reports/` — the milestone record. Historical, and their numbers stand; where a report names a retired task identifier it is describing what was run at the time. |
| **Agent harness** | `agents/README.md` — the agent-evaluation harness's design decisions and its measured exclusion table. It ships no task set; M9 authors the replacement. |
| **Roadmap** | `roadmap.md` — retired components and the findings worth knowing before restoring one. Explicitly non-executable. |
| **Live runner** | `physics/L2-PSF-01/` — the singlet workload. Its `1.0e-3 fft_oracle_intensity_relative_l2` gate is **unmet at 2.2072e-3**, queryable through `verification.claim_ledger.open_gates()`. The runner is retired-by-design. CHE-115 (M3.3) showed the executor reproduces this gate's number bit-identically from `examples/graphs/psf_singlet_sensor.yaml` via `instances/b3_psf_singlet.py`, which is the precondition its amendment set; the deletion itself is CHE-116's, and until then the runner keeps the convergence ladder, the O2 characterization oracle and two controls that have no replacement yet. |

## Archived (CHE-88)

`archive/benchmarks/gen1/` holds the first-generation ray, wave and coupler
suites plus `verify_m1_independence.py`, with the three benchmark-only adapters
and `m1_bundle.py` that only they consumed. They are preserved and **not
runnable**; `archive/benchmarks/gen1/README.md` records what each one guarded and
what is therefore unguarded now.

## Reproducibility

A run's *scientific fingerprint* is the hash of what it computed with everything
about the particular execution projected out — timestamps, run identifiers,
paths, process IDs, timings, peak RSS. The projection is
`core.provenance.VOLATILE_KEYS` and `strip_volatile`. The git dirty flag, package
versions, device and dtype are deliberately **not** projected out: they change
what was computed. Performance is comparable only when the separate environment
fingerprint also matches.

An instance's fingerprint is a separate thing and answers a different question:
SHA-256 over `(family_id, family_version, parameters, seed, execution_policy)`,
stable across processes, invalidated by a `family_version` bump.
