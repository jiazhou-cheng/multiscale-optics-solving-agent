# Benchmark Implementation Area

`manifest.yaml` is the machine-readable task index. Each implemented benchmark
directory carries its own task definition, oracle, metrics, variants, and
tolerances using the layout below.

Each implemented task should use:

```text
benchmarks/levelN/<task_id>/
  task.md                 # public scientific request
  public_config.yaml      # public inputs and budget
  oracle_graph.yaml       # expert graph; hidden during agent run
  generate_reference.py  # reproducible reference generation
  evaluate.py             # evaluator entry point
  tolerances.yaml         # private in final evaluation
  cases/public/           # public instances
  cases/hidden/           # not disclosed to the evaluated agent
  expected/               # hashes/statistics, not hand-edited outputs
  README.md
```

Every run must emit `graph.yaml`, `result.json`, scientific artifacts, and `provenance.json`. Freeze task specifications and evaluator versions before comparing agents.

## What is here

| | |
| -- | -- |
| **Live suite** | `level2/L2-PSF-01/` -- the one maintained benchmark. Its `1.0e-3 fft_oracle_intensity_relative_l2` gate is **unmet** and carried into M4 as an explicit open limitation; see `manifest.yaml`'s `gate_disposition`. |
| **Protocols** | `protocol.yaml`, `coupler_protocol.yaml`, `slice_protocol.yaml`, and the three `*_PROTOCOL.md` contracts. Frozen. |
| **Reports** | the milestone record. Historical, and their numbers stand. |
| **Probes** | `probes/` -- one-off executable evidence behind card claims. |
| **Roadmap** | `roadmap.md` -- planned tasks and retired components, explicitly non-executable. |

## Archived (CHE-88)

`L1-RAY-01`, `L1-WAVE-01`, `L2-COUPLER-01` and `verify_m1_independence.py` moved
to `archive/benchmarks/gen1/`, with the three benchmark-only adapters and
`m1_bundle.py` that only they consumed. They are preserved and **not runnable**;
`archive/benchmarks/gen1/README.md` records what each one guarded and what is
therefore unguarded now.

The M1 exit report ([`reports/2026-08/ray_and_wave_baselines.md`](reports/2026-08/ray_and_wave_baselines.md)) and the
`M1-BASELINE-CPU-V1`/`V2` contracts still describe those runs accurately. They
are the milestone record; the code that produced them is one directory away.

## Reproducibility

A run's *scientific fingerprint* is the hash of what it computed with everything
about the particular execution projected out -- timestamps, run identifiers,
paths, process IDs, timings, peak RSS. The projection is
`core.provenance.VOLATILE_KEYS` and `strip_volatile`, which CHE-88 moved out of
the archived `m1_bundle.py` because reproducibility is not an M1 concern. The
git dirty flag, package versions, device and dtype are deliberately **not**
projected out: they change what was computed. Performance is comparable only
when the separate environment fingerprint also matches.
