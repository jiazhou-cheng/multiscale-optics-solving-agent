# CHE-140 — The default suite: 375 s → 54 s

**Issue:** CHE-140
**Date:** 2026-08-26
**Every number below was measured in this issue**, on the `agent_solver` CPU
image, on the 80-core shared host. Commands are quoted verbatim so they can be
re-run; `-p no:cacheprovider` appears in the measurement commands only to keep
runs independent and makes no difference to the timings.

## Where the suite stands

| Suite | Command | Result | Wall |
|---|---|---|---|
| **Default gate** | `./run.sh pytest -q` (= `make test`) | 2510 passed, 58 skipped | **54.5 s** |
| Slow selection | `make test-slow` | 41 passed | 103.7 s |
| Everything, serial | `make test-serial` | 2551 passed, 58 skipped | 379.1 s |
| Opt-in GPU | `MOA_GPUS=device=6 make test-gpu` | 66 passed, 2555 deselected | 75.0 s |
| On-demand tutorials | `make test-tutorial` | unchanged by this issue | ~33 min |

Baseline before this issue, same tree: **374.86 s**, 2546 passed, 58 skipped.

The counts reconcile exactly, which is the point of quoting all three:

- 2608 collected now vs 2604 before = the 5 tests this issue added (4 in
  `tests/test_suite_layout.py`, 1 in `tests/test_resource_profile_guard.py`)
  minus… nothing. 2551 serial passes = 2546 + 5.
- 2567 selected + 41 deselected = 2608. The 41 are the 19 tests that already
  carried `slow` plus the 22 B2 stochastic-transition items this issue marked.
- 2510 passed + 58 skipped = 2567 selected.

**No test was deleted, weakened, or skipped by this issue.** One test was made
*less* environment-dependent and one new refusal path was added.

## What the 375 s actually was

`./run.sh --no-build pytest -q --durations=0`. 364.6 s of the 374.86 s was
attributable to individual tests; the rest is collection and container start
(2604 tests collected in 7.8 s). The distribution was the finding — **about 30
tests cost ~300 s and the other ~2400 cost ~52 s**:

| Bucket | Cost |
|---|---:|
| 19 tests already carrying `slow` | 151 s |
| `B2-W2R-STOCH` sweep, one module fixture | 60 s |
| `test_executor_integration.py`, real Optiland → Chromatix chain | 31 s |
| `test_provenance_fingerprint.py` | 30 s |
| b0 / b1_ray / b1_wave / b2_equiv drivers + `test_substrate_proof.py` | 23 s |
| `test_claim_ledger.py` collectibility subprocess | 8.8 s |
| `test_patch_positions.py` parametrized Monte Carlo | 8.3 s |
| the remaining ~2400 tests | ~52 s |

So the suite was never broadly slow. It was a fast suite with a short, very
expensive tail, and every entry in that tail is a *numerical characterization* —
a convergence fit, an unbiasedness ensemble, a negative-control battery. None of
it is the kind of coverage a per-commit gate is for, and all of it is coverage
this repository must keep.

## The changes

Four in this issue; a fifth added afterwards by CHE-107, which is the first
caller to hit `addopts` from outside the CPU gate.

### 1. `-m "not slow"` — the marker was already there

`slow` was declared in `pyproject.toml`, documented as "individually expensive
numerical characterization or convergence test", applied to 19 tests — and then
selected anyway, because `addopts` was `-ra`. It was decoration. It now filters.

### 2. `B2-W2R-STOCH` — a lazy fixture, then 20 markers

`tests/test_b2_transition_instances.py` built its `runs` fixture with
`run_all()`. Profiling the driver one instance at a time:

```
60.80s  B2-W2R-STOCH-01
 0.58s  B2-ROUNDTRIP-RAYWAVERAY-MONTE_CARLO-01
 ...
 0.00s  B2-W2R-STOCH-02 .. -08      (read back from the driver's own cache)
```

The whole family is **one** body of evidence — a six-point convergence ladder to
N=80000, an eight-seed unbiasedness ensemble, a five-control battery, a
two-spectrum variance study — and the first lookup computes all of it. Eagerly
building the mapping charged that 61 s to all 43 items in the file, including the
22 that never touch a STOCH run.

`_LazyRuns` executes an instance on first lookup instead. Which items genuinely
need the sweep was then determined **by measurement, not by reading section
headings**: the mapping was temporarily made to raise on any `B2-W2R-STOCH-*`
lookup and the suite run. 22 items failed, and one of them —
`test_no_round_trip_is_accepted_without_a_failing_twin` — sits under the
`B2-ROUNDTRIP` heading. Reading the file would have got it wrong.

Those 20 test functions carry `slow`. What stays in the default gate is the 21
items that do not: the four-conventions exactness gate, the entire R2W route
budget, and the round-trip directions with their failing twins. File cost in the
default suite: **61 s → 2.1 s**.

Nothing about the sweep was made cheaper. Its ladder, seeds, tolerances and
controls are untouched, and no benchmark record was regenerated.

### 3. `-n 12 --dist loadfile` — the worker count re-measured to 8 by CHE-171

`pytest-xdist==3.8.0`, pinned in `docker/requirements.txt`.

> **CHE-171 (R01.1) re-measured the constant and moved it to `-n 8`.** R01.1 is
> required not to inherit a number measured against a different suite, so the
> curve was taken again on the 80-core host, default selection, `--dist loadfile`,
> 2727 passed / 67 skipped:
>
> | `-n` | wall | | `-n` | wall |
> | --- | --- | --- | --- | --- |
> | 0 | 196.1 s | | 8 | 59.7 s |
> | 4 | 76.9 s | | 12 | 60.1 s |
> | 5 | 69.7 s | | 16 | 60.1 s |
> | 6 | 66.7 s | | | |
>
> The suite saturates at ~60 s from eight workers on — 8, 12 and 16 sit within
> 0.5 s of each other — so 8 is the smallest count that reaches the floor, and on
> a shared host that makes it the right one. Everything section 3 says about
> `loadfile` is unchanged and is still a correctness requirement.
>
> The critical path also moved. Per-file totals now put
> `test_provenance_fingerprint.py` first at **44.0 s**, ahead of
> `test_executor_integration.py` at 31.0 s: it recomputes an AST-normalized source
> fingerprint for every stamped record, so the record mechanism is its own
> critical path.
>
> **Two caveats on that 44.0 s, and both are re-measure triggers.** First, it was
> measured on a *failing* file: six of `test_provenance_fingerprint.py`'s tests
> fail at this commit (four stamped `m3_*` records drifted from
> `src/couplers/ray_to_wave.py` — see `docs/rewrite/reference_inventory.md` §5.1a),
> and a failing assertion does not do the same work as a passing one. When R07 or
> R13 regenerates those records the number will move. Second, the host was shared
> and under load (~15 of 80 cores busy) throughout, so the 0.4 s spread across
> n=8/12/16 is inside single-run noise — the plateau is real, but the sub-second
> ordering within it is not. **Re-measure before moving `-n` again.**
>
> **Why the budget is a documented number and not an executable timing gate.**
> CHE-171's acceptance criterion asks for a *measured* runtime budget, and the
> measurement is above. It is deliberately not a test that fails past N seconds:
> on a host this project shares, wall-clock depends on other users' jobs, so such
> a test would fail for reasons the committer cannot see or fix — and the honest
> response to a flaky resource gate is the one CHE-140 found in the swap guard,
> which is to key it on something attributable. Nothing here is attributable. So
> the budget is enforced socially, by this document and the `addopts` comment
> naming the number and the trigger, rather than by a check that would be
> disabled the third time it cried wolf. If a hard gate is wanted, it needs a
> quiet-host CI runner, which this project does not have.

`loadfile` is a **correctness** requirement, not tuning. About forty files here
use a module-scoped fixture that runs a benchmark driver or builds a solver
system once per file. xdist scopes fixtures per worker, so under its default
per-test distribution two tests from one file can land on two workers and each
re-runs that fixture — for the B2 file, that is the 61 s sweep, twice. The finer
mode would be slower *and* would silently multiply real solver compute.

12 rather than `auto`: this is a shared host and AGENTS.md puts stability ahead
of throughput. Each worker is a process importing jax, chromatix and optiland, so
workers cost RSS, and past ~12 there is nothing to win — the floor is the longest
single file (`test_executor_integration.py`, 31 s), not the core count. A `-n` on
the command line overrides the default; `-n 0` is the serial escape hatch.

### 4. What sharding exposed

Three failures appeared under `-n 12` that had never failed serially. All three
were pre-existing latent problems, and none was worked around:

- **`test_cli.py::test_list_models_names_every_registered_model`.** Asserted
  `"M_WAVE_CHROMATIX" in result.output` and got `M_WAVE_CHROMATI…`. `cli` builds
  its Rich `Console` at module scope, so the terminal width is fixed when the
  test file *imports* `cli` — before `CliRunner(env={"COLUMNS": "200"})` can
  apply. It passed for a year because a developer shell and `./run.sh` with a tty
  both hand Rich a wide terminal. A worker has no tty, Rich falls back to 80
  columns, and a 16-character id no longer fits. The fixture now replaces
  `cli.console` with `Console(width=200)`, so the test means the same thing on a
  tty, in a worker and in CI. Fixed in the test, not in `src/cli.py`: any edit
  under `src/` changes that file's code fingerprint and would invalidate every
  committed benchmark record.

- **`test_claim_ledger.py::test_every_cited_test_node_is_collectible` (×11).**
  This fixture shells out to `pytest --collect-only` to ask *"can pytest find
  this node id"*. The subprocess inherited `addopts`, so with `-m "not slow"` the
  question silently became *"is this node in the default run"*, and eleven
  perfectly valid citations to slow tests were reported as uncollectible. The
  subprocess now passes `-m "" -n 0`.

- **`test_resource_profile_guard.py::test_end_to_end_...`.** The real finding.
  The swap guard escalates with `os.kill(os.getpid(), SIGINT)`, which in a worker
  kills the *worker*: the guard tripped, gw0 went down, and the run reported
  `worker 'gw0' crashed while running ...` with no `SWAP GUARD TRIPPED` banner and
  no operator-facing diagnosis. For a cgroup-swap stop condition — an AGENTS.md
  resource-safety mechanism — that is exactly the silent degradation the policy
  forbids. Since `addopts` now carries `-n 12`, it is reachable by forgetting a
  flag. `scripts/pytest_resource_profile.py` now raises `pytest.UsageError`
  naming `-n 0` as the remedy rather than running degraded, and
  `test_the_guard_refuses_a_sharded_session_rather_than_degrading_in_one` covers
  it.

### 5. The GPU suite needed its own command (added by CHE-107)

Not part of the original issue, and found by CHE-107 trying to run the GPU
criterion it owns: `addopts` reaches invocations the CPU gate never sees, and
`./run.sh --gpu pytest -q -m gpu` — the command AGENTS.md documented — stopped
working the moment `-n 12` landed.

Two separate problems, and the second is the interesting one:

- `agent_solver_gpu` was built before this issue pinned `pytest-xdist`, so `-n`
  is an **unrecognized argument** there. The run exits 4 without collecting.
  Rebuilding the image would fix that and is not the fix, because:
- 49 of the 66 GPU tests would then be **sharded across 12 workers onto one
  device**. Each worker imports jax and opens its own CUDA context, and JAX
  preallocates a large fraction of device memory per process, so the second
  worker OOMs on a GPU the first is holding. This is exactly the swap-guard
  finding above with a different resource: a mechanism whose whole purpose is
  resource safety, degrading silently because a default reached it.

`make test-gpu` overrides `addopts` wholesale rather than appending `-n 0`, so it
is correct on the current image and on the next rebuild alike. Pinned by
`tests/test_suite_layout.py::test_the_gpu_suite_has_a_command_and_is_not_sharded`.

The stale image remains a real inconsistency — `docker/requirements.txt` declares
a package the GPU image does not have — and is left as follow-up rather than
rebuilt here: it is a ~10 GB rebuild on a shared host at 88% disk, and nothing in
the GPU path needs the plugin.

## What was considered and rejected

- **Trimming the STOCH ladder or the control battery.** That is the declared
  gate; shrinking it would change what the benchmark measures and stale every
  instance record. AGENTS.md: do not widen a tolerance to make a benchmark pass.
- **Caching `core.provenance._normalized_source_digest`.** ~20 s of the suite is
  58 parametrized record checks re-AST-hashing the same few hundred source files.
  This was implemented, measured — and reverted: editing `src/core/provenance.py`
  changed its own code fingerprint and invalidated all 65 instance records
  (`changed_files=('src/core/provenance.py',)`). Regenerating records is an
  explicit non-goal here, and the target was reachable without it. **Left as
  follow-up**; the waste is real and the fix belongs in a ticket that owns record
  regeneration.
- **Deleting or merging tests.** Nothing in the tail was redundant. The cost was
  repeat work and mis-tiering, not duplicated coverage.

## The boundary is now guarded

`tests/test_suite_layout.py` guarded three boundaries (default / tutorial /
archive), all directory-shaped. This issue's boundary is `addopts`, and it drifts
the same way for a worse reason — invisibly, in both directions. Dropping
`-m "not slow"` puts the gate back to six minutes; dropping `slow` off one test
puts 60 s of sweep back into every run. Four new tests pin it:

| Test | What it prevents |
|---|---|
| `test_the_default_selection_excludes_the_slow_marker` | `slow` becoming decoration again |
| `test_the_default_selection_shards_by_file_and_not_finer` | `loadfile` → `load`, i.e. re-running module fixtures per worker |
| `test_the_slow_suite_has_a_command` | a deselected test with no way back in |
| `test_the_slow_marker_is_still_declared` | `--strict-markers` is off, so an undeclared `slow` would match nothing and select everything |

## When to run what

- `make test` — every change. 54 s. This is the gate.
- `make test-slow` — **before merging any change to coupler numerics, sampling
  densities, estimator weights, a benchmark family, oracle or tolerance.** 104 s.
  These are exit gates, not optional extras; they are deselected because they are
  not per-commit, not because they are less important.
- `make test-serial` — when a failure is suspected to be a cross-test interaction
  or a worker artifact. If it reproduces here, it is real. 6.3 min.
- `MOA_GPUS=device=6 make test-gpu` — a change to device or precision handling,
  or to a placement claim. 75 s, one GPU, never sharded.
- `make test-tutorial` — pin or `docker/` change. ~33 min. Unchanged.
