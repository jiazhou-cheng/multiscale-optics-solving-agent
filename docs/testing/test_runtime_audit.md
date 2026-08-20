# Test-suite runtime audit and tier review (CHE-64)

Per-test runtime, peak memory, purpose and tier for all 904 profiled tests, the
manual review that followed, and the swap guardrail built to make the profiling
safe on a shared server.

- Machine-readable inventory: `docs/testing/test_inventory.json`
- Review table (sortable by cost, one row per test): `docs/testing/test_inventory.md`
- Profiler: `scripts/pytest_resource_profile.py` — opt-in pytest plugin
- Report builder: `scripts/build_test_inventory.py`
- Guardrail tests: `tests/test_resource_profile_guard.py`

Measured in the `agent_solver` container via `./run.sh`, CPU, at `ae4d897` +
this ticket's changes. Python 3.12.13, 80 logical CPUs, 377 GiB RAM.

---

## The answer in one line

**The suite is slow because of the upstream tutorial reproductions: 57 tests
costing 2003 s, which is 76 % of all measured wall time.** Tier A — the gate that
actually runs after every change — is 698 tests in 34 s of test time and was
never the problem.

| Tier | Tests | Σ per-test seconds | Share |
|---|---:|---:|---:|
| B (`slow`) | 64 | **2164.1** | 82 % |
| C (`benchmark`) | 85 | 410.1 | 16 % |
| A | 698 | 33.9 | 1.3 % |
| B (out-of-scope solvers) | 24 | 33.5 | 1.3 % |
| GPU (own session) | 33 | 0.0 | skipped here |
| **total** | **904** | **2641.8** | |

Splitting `slow` by what the tests are *for*:

| Group | Tests | Σ seconds |
|---|---:|---:|
| Upstream tutorial reproductions | 45 slow (+15 already in Tier A) | **2003** |
| Genuine slow physics (Monte-Carlo convergence, gradient bias) | 19 | 164 |

Those two groups answer different questions. The reproductions ask *"has the
pinned third-party solver changed?"*; the rest ask *"does our estimator
converge?"*. They were both wearing the `slow` label, so there was no way to run
one without the other.

**Note on units.** These are sums of per-test durations, not a single-session
wall time — the profiling had to be split into chunks to fit a 10-minute
command cap, so a whole-suite wall clock was never measured in one run. Real wall
time is higher (collection, imports, fixture setup between tests).

## Nine tests are 71 % of everything

| s | peak MiB | test |
|---:|---:|---|
| 284.5 | 1593 | `c04_zernike_fitting` |
| 276.0 | 1626 | `c10_seidel_fitting` |
| 233.9 | 725 | `test_m1_reproducibility.py::test_each_branch_reproduces_its_scientific_fingerprint` |
| 196.1 | 585 | `t36_glass_expert` |
| 193.4 | 1486 | `c03_computer_generated_holography` |
| 189.2 | 625 | `t28_optimization_case_study` |
| 177.1 | 763 | `t32_needle_synthesis` |
| 168.6 | 602 | `t27_advanced_optimization` |
| 144.9 | 617 | `t33_lithographic_projection_system` |

Eight of the nine are tutorial reproductions, and every one of those eight is
expensive for the same reason: it runs a real **optimization or fitting loop**
(glass expert, needle synthesis, Zernike/Seidel fitting, CGH). Their cost is
intrinsic to reproducing the tutorial — it cannot be trimmed without ceasing to
reproduce it.

Memory is not a constraint on this host. The heaviest test is
`c02_holoscope` at **6.2 GiB peak RSS** (then `c08_rescaled_propagation` at
4.9 GiB, `c06_off_axis_propagation` at 3.6 GiB). Against 377 GiB of RAM these are
comfortable, and **the container's cgroup swap charge stayed at 0 for every
profiling run** — no run was terminated for memory pressure.

---

## Manual review

### Delete: nothing

**No test is recommended for deletion.** Two candidate classes were checked
explicitly and both came back clean:

- **Duplication.** `tests/test_optiland_ray_benchmark.py` (5 tests) and
  `tests/benchmarks/test_l1_ray_scaling.py` (4 tests) look like overlap but
  assert different things — bundle artifacts and convention regressions versus
  scaling counts, timing separation and evaluator rejection. No shared assertion.
- **Vacuous tests.** An AST scan flagged 10 tests with no `assert` statement.
  All 10 were inspected and all 10 assert properly, via
  `np.testing.assert_allclose` or `jsonschema.validate` — calls, not `assert`
  statements. **The heuristic was wrong; there are no vacuous tests.** Recorded
  here so the next audit does not re-raise it as a finding.

This matches CHE-52 (PB1), which also proposed zero deletions. The suite's
problem is not junk tests; it is that one expensive category could not be
deselected.

### Reclassify: a `tutorial` marker (applied)

The 57 tutorial reproductions now carry `pytest.mark.tutorial` in addition to
their existing markers. **Additive on purpose — `slow` is unchanged, so every
existing tier command selects exactly what it selected before.** Verified: Tier A
still collects 731 and still reports 677 passed / 54 skipped in 42 s.

What it buys:

```bash
# milestone regression WITHOUT the pinned-dependency gate: 848 tests, ~639 s
./run.sh pytest -q -m "not tutorial"

# the pinned-dependency gate on its own: 57 tests, ~2003 s
./run.sh pytest -q -m tutorial
```

The reproductions only tell you something new when a pin moves. Running them on
every full regression pays 2003 s to re-learn that `optiland==0.6.0` and
`chromatix==0.6.0` still do what they did last week. The right trigger is a
change to `docker/Dockerfile` or the dependency pins — the same trigger that
already requires `./run.sh --rebuild`.

**No test changed tier in a way that weakens a gate.** Nothing moved out of
Tier A, nothing lost a marker, no tolerance was touched.

### Keep, unchanged

- **The 19 non-tutorial `slow` tests (164 s).** Monte-Carlo convergence rates,
  gradient-estimator bias, round-trip convergence. This is our physics, the cost
  is the statistics, and 164 s is proportionate.
- **The 85 `benchmark` tests (410 s).** Milestone bundle reproductions. The
  234 s `test_m1_reproducibility` outlier re-runs whole benchmark bundles to
  compare scientific fingerprints; that is the point of it.
- **The 21 skipped `test_m3r_sensor_handoff.py` tests.** Per CHE-62 they skip
  because CHE-38's consolidated record has never landed; regeneration is CHE-63.
  They are the single-artifact reproducibility check that ticket asked for.
- **The 33 `gpu` tests.** Quarantined to their own session by design (CHE-60).

---

## Findings

### F1 — `t21_surface_roughness_scattering` is flaky: a hard threshold on a random quantity

Found by the profiler, then measured directly: **~2 failures in ~22 runs (≈9 %)**.
It passes in isolation, so this is not order dependence — it is genuine
nondeterminism.

Root cause, in
`knowledge/solvers/optiland/tutorials/t21_surface_roughness_scattering.py`:

```python
out["centroid_is_on_axis"] = bool(stats["centroid_offset_over_rms"] < 0.1)   # line 122
...
    stats["centroid_offset_over_rms"] < 0.1,                                  # line 135
```

`centroid_offset_over_rms` is the intensity-weighted centroid of a *scattered*
ray distribution divided by its RMS radius — the module's own comment calls it "a
near-zero random quantity". The comparison is a hard threshold, and the resulting
boolean is then compared **exactly** by the test harness
(`test_optiland_tutorials.py` line 112–113 compares booleans with `==`,
bypassing the `metric_rtol=0.35` the reproduction declares for its floats).

It cannot be fixed by seeding: `optiland.scatter` is numba-compiled (`njit`/
`prange`) and draws from an RNG `np.random.seed` cannot reach. The reproduction
documents this and even asserts it as a finding.

With N = 10,921 rays a naive `sqrt(2/N)` estimate puts the fluctuation near
0.014, which would make 0.1 a safe bound — the measured 9 % failure rate says the
intensity-weighted Lambertian tail makes the real spread much wider. **Not fixed
here:** the threshold guards a physical claim ("scattering broadens the spot
without moving its centroid"), so widening it is a tolerance decision, and
AGENTS.md forbids promoting a tolerance silently. Recommended fix is to make the
bound statistical rather than a magic constant — derive it from the measured
spread across trials, with margin — or raise the ray count until 0.1 genuinely is
a bound. Needs its own ticket.

### F2 — Tier C's documented cost is stale by roughly 4×

`AGENTS.md` and `docs/testing/tier_restructure.md` state Tier C is "627 tests,
~11 min". It is now 905 tests and at least 2642 s of per-test time. The entire
increase is CHE-57's tutorial reproductions, which landed after PB3 measured. The
`tutorial` marker above restores a ~639 s regression path; the documentation is
updated to say both numbers and which is which.

### F3 — Profiling a suite this long needs incremental flushing, and that was learned the hard way

The first profiler wrote its JSON only at `sessionfinish`. Three chunks were
killed by the outer 10-minute command cap and wrote **nothing at all**, discarding
completed measurements — the same failure that cost CHE-38 two 25-minute probe
runs. The profiler now flushes every 5 tests *and* after any test slower than
2 s, and marks partial payloads `"complete": false` so a truncated file cannot be
mistaken for a full inventory. Two chunks in the current data set are partial and
labelled as such; their tests were re-profiled in smaller chunks.

### F4 — Killing the client does not kill the container

A timed-out `./run.sh` leaves the container running the tests. Three orphaned
`agent_solver` containers were created and stopped by hand during this work. On a
shared GPU server that is a real hazard: the operator believes the job is dead
and starts another. Worth a `run.sh` fix (propagate the signal, or `--rm` with an
exec-based teardown) — out of scope here, recorded for a follow-up.

---

## The swap guardrail

`scripts/pytest_resource_profile.py --swap-guard` terminates the run when the
container starts swapping.

**The obvious implementation is broken, and this is the important part.** Checking
"is swap in use?" or any absolute threshold would fire on every run forever.
Measured inside the container:

```
SwapTotal   1955610616 kB    (1.82 TiB)
SwapFree    1954862184 kB    => ~748 MiB ALREADY in use at rest
pswpout       44610212       cumulative pages out, since boot
```

`/proc/meminfo` inside the container is the **host's**, and this is a shared
server, so both figures reflect whatever anyone else has ever run. Neither can
distinguish "this suite started thrashing" from "someone's job swapped last
Tuesday".

The gate signal is therefore **`/sys/fs/cgroup/memory.swap.current`** — the
container's own swap charge under cgroup v2. It reads `0` at rest, is scoped to
this container, and rises only when our pages go out. Host `SwapFree` and
`pswpout` are still recorded as deltas from a session baseline, so a reader can
attribute pressure that landed in another cgroup and tell "we swapped" from "the
box was already swapping".

Both `memory.max` and `memory.swap.max` read `max` here, so there is no cgroup
ceiling to hit first: the guardrail is the only thing between a memory-hungry
test and host-wide memory pressure.

Behaviour on breach:

1. Record the breach — timestamp, growth vs baseline, threshold, **the test that
   was executing**, its elapsed time and RSS, how many tests had completed, and a
   full host snapshot.
2. **Flush the JSON before signalling**, so a teardown cannot lose the record
   that explains it.
3. `SIGINT` the process. pytest turns it into a `KeyboardInterrupt` and tears the
   session down with a **nonzero exit status**. `session.shouldstop` was
   deliberately rejected: it would let the current — possibly runaway — test run
   to completion, which is the exact case the guardrail exists to interrupt.
4. Print a red `CHE-64 SWAP GUARD TRIPPED - RUN TERMINATED` block saying **THIS
   RUN IS FAILED** and that the results must not be treated as a clean inventory.

Detection runs both in a 50 ms sampling thread and at every test boundary on the
main thread. The boundary check exists because of a hole the end-to-end test
found: a test finishing inside the sampling interval is never sampled, so with
the thread alone it would never be checked at all.

Default tolerance is 4 MiB of growth, not zero — cgroup accounting can tick by a
page or two, and a guardrail that cries wolf gets switched off by the next
person.

### It is verified, not asserted

`tests/test_resource_profile_guard.py` — 8 tests, in Tier A except the
subprocess one.

The end-to-end test runs a **real pytest session** with `--swap-guard-kib=-1`, so
any reading breaches the threshold, and asserts a nonzero exit, the terminal
banner, and the flushed breach record naming the active test. Only *when* the
threshold is crossed is faked; detection, flush ordering, signal, teardown and
exit status are all the production path. That is deliberate — the alternative is
exhausting 377 GiB of host RAM to test a safety feature, on a shared server,
which is the outcome the feature exists to prevent.

Also covered: below-threshold growth does not trip; escalation happens exactly
once (a second `SIGINT` into an unwinding session could mask the first); the guard
is completely inert unless requested; a breach with no active test is recorded
rather than crashing; and the cgroup signal exists and is readable — because
`check_swap` returns early on an unreadable path, a missing signal would make the
guard **silently inert** rather than loudly broken.

---

## Reproducing this

```bash
# Tier A, with profile and guardrail
./run.sh python -m pytest -q -p scripts.pytest_resource_profile \
    --resource-profile=outputs/CHE-64/tierA.json --swap-guard \
    -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"

# the expensive groups, in chunks small enough to finish inside a command cap
./run.sh python -m pytest -q -p scripts.pytest_resource_profile \
    --resource-profile=outputs/CHE-64/benchmark.json --swap-guard -m benchmark

# rebuild the inventory from every chunk in outputs/CHE-64/
./run.sh python scripts/build_test_inventory.py
```

`-p scripts.pytest_resource_profile` needs `python -m pytest` rather than bare
`pytest`, so that the repository root is on `sys.path` when plugins are imported.
