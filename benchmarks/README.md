# Benchmark Implementation Area

The normative task definitions, oracles, metrics, hidden variants, and suggested thresholds are in [`../docs/BENCHMARK_SPECIFICATION.md`](../docs/BENCHMARK_SPECIFICATION.md). `manifest.yaml` is the machine-readable task index.

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
