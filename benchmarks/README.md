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

The M1 ray and wave baselines additionally follow the frozen
[`M1-BASELINE-CPU-V1`](M1_BASELINE_PROTOCOL.md) contract. Each baseline runs
in a fresh process without importing the other engine or a coupler, and must
emit `result.json`, `provenance.json`, `arrays.npz`, `plot.png`,
`tolerances.yaml`, and `README.md`. Accuracy and performance results remain
separate. Missing solver results are reported as structured blockers, never
replaced with fabricated values.

Reproduce and independently review the complete M1 branches with:

```bash
./run.sh python benchmarks/level1/L1-RAY-01/run_all.py
./run.sh python benchmarks/level1/L1-WAVE-01/run_all.py
./run.sh python benchmarks/verify_m1_independence.py
```

The reviewed outcome of those runs is the M1 exit report,
[`M1_BASELINE_REPORT.md`](M1_BASELINE_REPORT.md): exact commands, commit,
environment, per-branch accuracy and performance evidence, independence and
claim-audit results, and the recorded limitations that M2 must carry forward.
Note that the ray branch runs the amended `M1-BASELINE-CPU-V2` contract while
the wave branch runs `M1-BASELINE-CPU-V1`; the report records that gap as
limitation L2.

Each `run_all.py` command launches its standalone, analytic, and scaling work
in separate child processes. The root bundle records a scientific fingerprint
that excludes timestamps, run identifiers, paths, process IDs, timings, peak
RSS, and the Git dirty flag; performance is comparable only when the separate
environment fingerprint matches.
