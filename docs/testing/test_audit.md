# PB1 — Test Suite Inventory, Classification, and Disposition

**Issue:** CHE-52 (PB1), milestone M3.5
**Date:** 2026-08-18
**Rule:** read-only. No test file, fixture, or config was modified to produce this audit.
**Blocks:** CHE-53 (PB2) — this document is PB2's required input; CHE-53 acts on the
disposition column, it does not re-derive it.

## Acceptance criteria restated (from CHE-52)

- Inventory every test module (and individual test, where grouping is coarse) with
  file path, purpose, and runtime.
- Classify each into one of: core contract validation, current solver functionality,
  coupler functionality, integration behavior, benchmark validation, regression
  coverage, legacy/migration coverage, duplicated coverage, outdated-vs-architecture,
  expensive numerical validation, environment/backend-specific.
- For anything other than `KEEP_REQUIRED`: document original purpose → why no longer
  required (or: why it doesn't belong in the required gate) → replacement coverage,
  if any → proposed action.
- No `DELETE` without a named replacement, for a still-valid invariant.
- Measure and record baseline runtime via `./run.sh`.
- Produce one reviewable artifact (this document); do not act on it.

Non-goals for this issue (explicitly out of scope, deferred to PB2/PB3): deleting,
archiving, merging, or rewriting any test; introducing markers or tier structure;
shrinking numerical problem sizes.

## How this audit was produced

```bash
./run.sh --no-build pytest --collect-only -q               # 627 tests collected, 42 files
./run.sh --no-build pytest -q --durations=0                 # timed out at the 10-minute cap, unfinished
./run.sh --no-build pytest -q --durations=0 <chunk of 6-8 files>   # run six times, see below
```

The full suite was run once as a single invocation and did not complete inside a
10-minute window — this is itself a finding (see §2). To get complete timing data,
the 42 files were run in six sequential chunks (never in parallel, per the repo's
GPU/host resource policy); each chunk's `time` and pytest's own `--durations=0`
report are both recorded below. Every command ran inside the `agent_solver`
container via `run.sh`, never on the host, per `AGENTS.md`.

Test purposes were established by reading each module's docstring and, where a
docstring was absent or thin, its body and imports directly — not by running
individual tests and guessing intent from pass/fail.

## 1. Inventory scale

- 42 test files under `tests/` (34 top-level, 8 under `tests/benchmarks/`).
- 627 collected test items (including parametrized cases) from ~449 test functions.
- `tests/conftest.py` provides two shared fixtures (`registry`, and probe-evidence
  loader helpers `load_probe_expected` / `load_coupler_probe_expected`) used across
  most files — it is shared infrastructure, not a test module, and is not scored
  below.
- Existing `pyproject.toml` markers: `jax`, `torch`, `integration` (all currently
  used for "requires an optional solver install", not for "is slow" or "is out of
  current milestone scope" — see §3).

## 2. Baseline runtime

| Chunk | Files | Tests | pytest-reported time | wall time (`time`, incl. container start) |
|---|---|---|---|---|
| 1 | `test_context_sync`, `test_m3_off_axis_handoff`, `test_fmmax_adapter`, `test_coupler_gradient`, `test_sax_adapter`, `test_run_sh`, `test_m3_convergence` | 101 passed, 1 xpassed | 95.80s | 99.60s |
| 2 | `test_graph_validation`, `test_registry`, `test_ray_to_wave_node`, `test_adapter_registry`, `test_m2_coupler_protocol`, `test_m3r_sensor_handoff`, `test_wave_to_ray` | 76 passed, 21 skipped | 57.37s | 59.69s |
| 3 | `test_artifacts`, `test_carrier_removed_asm`, `test_coupler_contracts`, `test_optiland_ray_benchmark`, `test_m3_psf_verification`, `test_m3_pupil_to_focus`, `test_m3_quadrature_weight` | 94 passed | 17.97s | 20.89s |
| 4 | `test_curvature_bound`, `test_optiland_adapter`, `test_m3_slice_protocol`, `test_coupler_round_trip`, `test_optiland_opd_convention`, `test_quadrature`, `test_m1_protocol` | 127 passed | 24.52s | 27.02s |
| 5 | `test_ray_to_wave`, `test_optiland_coherent_handoff`, `test_coupler_knowledge_pack`, `test_m3_psf_measurement`, `test_fdtdx_adapter`, `test_chromatix_adapter` | 125 passed, 2 xfailed | 38.52s | 42.85s |
| 6 | `tests/benchmarks/` (8 files) | 80 passed | 401.01s | 403.03s |
| **Total** | **42 files** | **627** | **~635.2s (10m 35s)** | **~653.1s (10m 53s)** |

**Finding F1 — the full suite cannot be run as a single command inside a 10-minute
window.** `./run.sh --no-build pytest -q --durations=0` (no file filter) was killed
by the tool timeout after 600s without finishing. The chunked total above
(635.2s of pytest-internal time, before any container-startup overhead) confirms
this isn't a fluke of that one invocation — the suite is genuinely over the
10-minute mark. This is the direct motivation for PB2/PB3's tiering.

**Finding F2 — runtime is dominated by four files, not spread evenly.**
`tests/benchmarks/` alone is 401.0s of the 635.2s total (63%), and within it, four
tests each pay for one full, real solver run inside a `scope="module"` fixture:

| Test | Setup time |
|---|---|
| `tests/benchmarks/test_m1_reproducibility.py::test_each_branch_reproduces_its_scientific_fingerprint` | 230.89s |
| `tests/benchmarks/test_l1_wave_scaling.py::test_grid_padding_and_timing_contract` | 76.95s |
| `tests/benchmarks/test_l1_wave_accuracy.py::test_bundle_contains_required_protocol_and_scientific_artifacts` | 63.52s |
| `tests/benchmarks/test_l1_ray_scaling.py::test_scaling_bundle_records_actual_counts_and_separate_timings` | 24.76s |

These four module-scoped fixtures sum to 396.1s — essentially all of the
benchmarks/ cost, and 62% of the whole suite. All four already carry
`@pytest.mark.integration` (three also carry `@pytest.mark.jax`), so they are
already structurally separable from a fast gate; PB2 does not need to invent new
machinery for this, only to route the existing markers into a tier.

The next tier of expensive individual tests, all **outside** `tests/benchmarks/`
and **none currently marked** `slow`/`integration` (`test_optiland_ray_benchmark.py`
is the one exception, already `integration`-marked):

| Test | Time |
|---|---|
| `test_coupler_gradient.py::test_uniform_and_magnitude_sampling_are_both_unbiased_for_the_gradient` | 19.97s |
| `test_coupler_gradient.py::test_detaching_the_density_is_what_makes_the_estimator_unbiased` | 19.96s |
| `test_coupler_round_trip.py::test_wave_to_ray_to_wave_converges_at_the_monte_carlo_rate` | 13.23s |
| `test_wave_to_ray.py::test_error_falls_as_n_to_the_minus_one_half[p_mag]` | 13.10s |
| `test_wave_to_ray.py::test_error_falls_as_n_to_the_minus_one_half[p_uni]` | 13.10s |
| `test_coupler_gradient.py::test_no_bias_appears_as_the_ray_count_grows` | 11.53s |
| `test_m3r_sensor_handoff.py::test_the_circular_reference_is_far_from_the_straight_edge_it_replaced` | 10.42s |
| `test_coupler_gradient.py::test_the_fixed_direction_estimator_is_unbiased_on_a_fixed_spectral_grid[quadratic]` | 9.99s |
| `test_coupler_gradient.py::test_the_fixed_direction_estimator_is_unbiased_on_a_fixed_spectral_grid[linear]` | 9.97s |
| `test_fdtdx_adapter.py::test_wavelength_gradient_matches_finite_difference_lock` (xfail-locked) | 7.78s |
| `test_chromatix_adapter.py::test_gradient_probe_regression_thin_lens_transform_propagate` | 6.62s |
| `test_optiland_ray_benchmark.py::test_bundle_contains_required_protocol_and_scientific_artifacts` (setup) | 6.33s |
| `test_fdtdx_adapter.py::test_smoke_run_succeeds_with_documented_axis_order` | 6.20s |
| `test_fmmax_adapter.py::test_smoke_grating_run_succeeds` | 5.93s |
| `test_m1_protocol.py::test_engine_probe_uses_independent_compatible_processes` (subprocess) | 4.87s |
| `test_wave_to_ray.py::test_ensemble_mean_is_unbiased_within_three_standard_errors[p_uni]` | 4.91s |
| `test_wave_to_ray.py::test_ensemble_mean_is_unbiased_within_three_standard_errors[p_mag]` | 4.89s |
| `test_wave_to_ray.py::test_omitting_the_importance_weight_is_detected_as_a_bias` | 4.89s |
| `test_fmmax_adapter.py::test_gradient_through_run_matches_recorded_probe` | 4.83s |

`test_coupler_gradient.py` alone (CHE-28, gradient-bias characterization) costs
~74.6s across 5 of its 9 tests — the single most expensive non-benchmark file, and
entirely unmarked today.

**Finding F3 — a real, unflagged coverage gap: 21 of 27 tests in
`test_m3r_sensor_handoff.py` are silently skipped in this environment**, not
because of an optional-dependency guard but because
`benchmarks/probes/records/m3r_sensor_handoff.json` does not exist on disk. `git log`
confirms this file was never committed: commit `ec55839`
(`CHE-38/CHE-47/CHE-39: M3.9R sensor-handoff verdict...`) added
`benchmarks/probes/m3r_sensor_handoff.py` (the probe script, 2711 lines) and
`tests/test_m3r_sensor_handoff.py` (the test file, 410 lines) but not the probe's
output record. `pytest -q` reports this as `21 skipped` — easy to read as "these
tests don't apply here" rather than "the evidence CHE-38/CHE-47's verdict rests on
is not present." The 9 tests that run live (no probe file needed) do pass,
including the 10.42s `test_the_circular_reference_is_far_from_the_straight_edge...`.
This is not a PB1/PB2/PB3 action item (regenerating and committing the probe record
is solver work, not test-structure work) — it is flagged here so it is not lost,
and a follow-up issue is recommended in §6.

## 3. Marker-application finding (input to PB2, not a PB1 decision)

`jax`/`torch`/`integration` are applied **inconsistently** with respect to "is this
test expensive / does it touch a real solver":

- `test_chromatix_adapter.py`, `test_optiland_adapter.py`, `test_fmmax_adapter.py`,
  `test_fdtdx_adapter.py`, `test_sax_adapter.py`, and all of `tests/benchmarks/`
  self-mark with `jax`/`torch`/`integration` and use `pytest.importorskip` as a
  belt-and-suspenders guard.
- `test_ray_to_wave.py`, `test_optiland_coherent_handoff.py`,
  `test_optiland_opd_convention.py`, `test_m3_off_axis_handoff.py`,
  `test_m3_pupil_to_focus.py`, `test_m3_quadrature_weight.py`,
  `test_m3_psf_measurement.py`, `test_ray_to_wave_node.py`, `test_carrier_removed_asm.py`
  — all core, current-milestone files that trace real Optiland lenses or run real
  Chromatix propagation via `pytest.importorskip` — carry **no marker at all**.

A Tier A rule of "exclude anything marked `jax`/`torch`/`integration`" would
therefore incorrectly exclude the Chromatix/Optiland *adapter* contract tests
(core to the current milestone) while correctly excluding `fmmax`/`fdtdx`/`sax`
(explicitly out of current milestone scope per `AGENTS.md` §"Current Scope") and
the four heavy benchmark-reproduction tests. **PB2 needs a marker axis, not a
reuse of the existing three** — e.g. separate "requires optional solver install"
(what `jax`/`torch`/`integration` already mean) from "expensive" (new, e.g. `slow`)
and from "out of current milestone" (new, e.g. a marker per out-of-scope solver, or
one shared `solver_out_of_scope` marker for `fmmax`/`fdtdx`/`sax`). This is a
recommendation for PB2 to decide against its own acceptance criteria
("confirm exact names against PB1's inventory rather than assuming this list"),
not a PB1 decision.

## 4. Headline conclusion: no obsolete, migration-era, or duplicated coverage was found

CHE-52's own description hypothesizes "superseded contracts (e.g. retired
`C_FIELD_TO_PSF`)... temporary migration behavior from the M0 archive... duplicate
coverage across baseline/coupler/integration layers." Reading all 42 files against
that hypothesis:

- The `C_FIELD_TO_PSF` retirement (CHE-36/M3.7) is **still guarded**, correctly, by
  `test_graph_validation.py::test_field_to_psf_is_not_a_registered_coupler` and by
  `test_m3_psf_measurement.py`'s architectural section. The *retired coupler* is
  obsolete; the *test that keeps it retired* is a live, still-necessary
  architectural invariant. Nothing here is `DELETE`-eligible.
- No test references M0-archive-era migration behavior. `test_context_sync.py`
  (CHE-7/M0.5) is the closest match by name, but it exercises a permanent
  consistency-checking script, not one-time migration logic.
- No two files were found asserting the same physical or architectural claim.
  Pairs that look superficially similar are documented, in their own docstrings, as
  deliberately complementary: `test_m1_bundle_projection.py` explicitly exists as
  the near-instant counterpart to the ~7-minute
  `test_m1_reproducibility.py`, catching the same defect class cheaply;
  `test_l2_psf_bundle.py` "mirrors" `test_l2_coupler_bundle.py`'s bundle-honesty
  checks but for a different (M3, not M2) bundle.
- Per-milestone protocol-freeze tests (`test_m1_protocol.py`, `test_m2_coupler_protocol.py`,
  `test_m3_slice_protocol.py`) look repetitive by name but freeze three different
  milestones' contracts, each still load-bearing.

**Consequence for PB2/PB3:** the `REWRITE` / `MERGE` / `ARCHIVE` / `DELETE` columns
below are empty. PB2's job, per this audit, reduces to marker design and tier
assignment — there is no backlog of bad tests to clean up first. This is worth
stating plainly because it contradicts CHE-52's own framing, and `AGENTS.md`
requires surfacing exactly this kind of conflict rather than silently picking a
side.

## 5. Disposition table

Action vocabulary: `KEEP_REQUIRED` (belongs in the required/fast gate) |
`KEEP_TARGETED` (real, current coverage; too expensive or too out-of-scope for the
required gate — a targeted/Tier B suite) | `KEEP_FULL_REGRESSION` (real coverage,
only affordable as part of a full/Tier C run) | `REWRITE` | `MERGE` | `ARCHIVE` |
`DELETE`. The last four are unused — see §4.

Categories: **CONTRACT** (core contract validation) · **SOLVER** (current
in-scope solver/adapter functionality: Optiland, Chromatix) · **COUPLER**
(ray↔wave coupler physics) · **INTEGRATION** (graph/registry/end-to-end wiring) ·
**BENCH** (benchmark bundle validation) · **REGRESSION** (pins a prior,
already-established finding, often against a recorded probe) · **OUT-OF-SCOPE**
(fmmax/fdtdx/sax — valid, but outside the current milestone per `AGENTS.md`) ·
**EXPENSIVE** (numerical characterization/convergence study — legitimately costly)
· **ENV** (execution-environment / repo-hygiene, not physics).

| File | Tests | Category | Purpose | Runtime | Action | Notes |
|---|---|---|---|---|---|---|
| `test_adapter_registry.py` | 2(+4 param) | CONTRACT | Registry discovery contract: unknown model raises, every discovered adapter's spec id matches its registry key | <0.1s | `KEEP_REQUIRED` | |
| `test_artifacts.py` | 1 | CONTRACT | `ArtifactRecord` preserves typed metadata | <0.01s | `KEEP_REQUIRED` | |
| `test_registry.py` | 2 | CONTRACT | Packaged registry loads; model/coupler ids unique | <0.1s | `KEEP_REQUIRED` | |
| `test_run_sh.py` | 2 | ENV | `run.sh`'s Docker tty argument construction (CHE — container-only execution contract) | 0.04s | `KEEP_REQUIRED` | Protects the only supported entry point (`AGENTS.md` §"Execution Environment"). Cheap; no reason to move it. |
| `test_context_sync.py` | 15 | ENV / CONTRACT | `scripts/check_context_sync.py` rules, exercised on a passing and a deliberately-broken repo copy (CHE-7/M0.5) | 0.11s | `KEEP_REQUIRED` | |
| `test_graph_validation.py` | 7 | INTEGRATION | Graph validator: cycles, artifact-kind mismatch, unverified-gradient policy, and that `C_FIELD_TO_PSF` is *not* a registered coupler | <0.05s ea. | `KEEP_REQUIRED` | Guards the CHE-36 retirement (see §4) |
| `test_m1_protocol.py` | 5 | CONTRACT / BENCH | M1 protocol freeze; one subprocess call to `benchmarks/probes/verify_m1_engines.py` | 4.87s (one subprocess test), rest <0.03s | `KEEP_REQUIRED` | The 4.87s subprocess test is the one item in this file worth a second look in PB3 if the Tier A budget is tight — see §2 table 2 |
| `test_m2_coupler_protocol.py` | 9 | CONTRACT | M2 coupler-protocol freeze (stochastic vs. deterministic result handling, three-way blocked/failed/unconverged split) | <0.02s ea. | `KEEP_REQUIRED` | |
| `test_coupler_contracts.py` | 30 | CONTRACT | Typed bidirectional boundary artifacts (CHE-23) — a missing declaration must error, not default | <0.02s ea. | `KEEP_REQUIRED` | |
| `test_coupler_knowledge_pack.py` | 10 | REGRESSION | Coupler knowledge packs stay honest about evidence vs. documentation (CHE-22) | ~0.03s ea. | `KEEP_REQUIRED` | Doc-drift guard, not physics, but cheap and load-bearing |
| `test_coupler_gradient.py` | 9 | EXPENSIVE / COUPLER | Gradient-estimator bias characterization (CHE-28) — deliberately does **not** certify a gradient | **~74.6s total**, 5 of 9 tests >9s | `KEEP_TARGETED` | Genuine numerical characterization (multiple step sizes / grid points), not shrinkable without weakening the claim per its own docstring. Single most expensive non-benchmark file; move out of the required gate. |
| `test_coupler_round_trip.py` | 14 | COUPLER / EXPENSIVE | Wave→ray→wave round trip consistency (CHE-26); 13 fast structural tests + 1 convergence-rate fit | 13.23s in one test, rest <0.02s | Split: fast tests `KEEP_REQUIRED`; `test_wave_to_ray_to_wave_converges_at_the_monte_carlo_rate` `KEEP_TARGETED` | The one slow test fits a convergence exponent — needs a sweep of `N`, not shrinkable to one size without losing the claim |
| `test_curvature_bound.py` | 20 | CONTRACT | SI eq. S9 curvature bound must actually bound a built measurement (CHE-27) | negligible (not in top-40 durations) | `KEEP_REQUIRED` | |
| `test_quadrature.py` | 7(+16 param) | COUPLER | Hexapolar quadrature-weight math in isolation (CHE-47 ext.) — pure functions, no engine | <0.02s ea. | `KEEP_REQUIRED` | |
| `test_m3_quadrature_weight.py` | 9 | COUPLER / REGRESSION | Per-ray quadrature weight in production, live math + adapter wiring + probe-record comparison (CHE-47 ext.) | ~0.04s total | `KEEP_REQUIRED` | Both the live and the probe-reading halves are cheap here (probe record exists and is small) |
| `test_m3r_sensor_handoff.py` | 27 | COUPLER / REGRESSION | Sensor-side handoff verdict (CHE-38/M3.9R); 9 live tests + **21 tests reading a probe record that is missing from the repo** | 9 live tests run (one at 10.42s); 21 unconditionally skipped | Live tests `KEEP_REQUIRED`/`KEEP_TARGETED` (see note); probe-record tests unresolved pending the record | **Finding F3** — see §2. Not this issue's action, but flagged for a follow-up issue (§6). The single 10.42s live test is a PB3 sizing candidate. |
| `test_m3_convergence.py` | 27 | COUPLER / REGRESSION | Convergence study and the aperture defect it found (CHE-38/M3.9); live synthetic-oracle tests + probe-record comparison (record present) | all <0.15s | `KEEP_REQUIRED` | Deliberately pins two known gate *failures* and a non-monotone trend — must not silently start passing |
| `test_m3_off_axis_handoff.py` | 21 | COUPLER / REGRESSION | Off-axis OPL reference (CHE-41); real-trace + oracle-file + probe-record comparison (record present) | ~1.8s (one setup) | `KEEP_REQUIRED` | |
| `test_m3_psf_verification.py` | 19 | COUPLER / REGRESSION | Slice verified against independent oracles (CHE-37/M3.8); live oracle tests + probe-record comparison (record present) | all <0.4s | `KEEP_REQUIRED` | |
| `test_m3_psf_measurement.py` | 25 | CONTRACT / COUPLER | PSF measurement semantics + the CHE-36/M3.7 retirement of `C_FIELD_TO_PSF` | ~1.6s (one setup) | `KEEP_REQUIRED` | See §4 — guards a still-live invariant, not obsolete |
| `test_m3_pupil_to_focus.py` | 14 | COUPLER | Reconstructed pupil field propagated to focus; two named losses quantified against a probe record (CHE-35/M3.6) | ~3.5s total | `KEEP_REQUIRED` | |
| `test_m3_slice_protocol.py` | 20(+2 param) | CONTRACT | M3 slice protocol freeze (CHE-31/M3.2) | ~0.07s ea. | `KEEP_REQUIRED` | |
| `test_carrier_removed_asm.py` | 13 | COUPLER | Carrier-removed exact ASM and the conditioning it fixes (CHE-40/M3.2A) | ~3.4s total | `KEEP_REQUIRED` | |
| `test_optiland_adapter.py` | 19(+9 param) | SOLVER | `M_RAY_OPTILAND` adapter contract, numpy + opt-in torch backend | <1s total | `KEEP_REQUIRED` | Already `torch`/`integration`-marked on the relevant subset; current-milestone core |
| `test_optiland_coherent_handoff.py` | 14(+4 param) | SOLVER / COUPLER | Declared coherent handoff from a real Optiland trace: OPL convention + intensity→amplitude map (CHE-33/M3.4) | ~1.6s (one setup) | `KEEP_REQUIRED` | |
| `test_optiland_opd_convention.py` | 10(+3 param) | SOLVER | Established `RealRays.opd` sign/reference convention and its falsifiers (CHE-30/M3.1) | negligible | `KEEP_REQUIRED` | |
| `test_optiland_ray_benchmark.py` | 5 | BENCH / SOLVER | L1-RAY-01 accuracy-benchmark bundle contract, real catalog-lens trace | 6.33s (module setup) + 0.31s | `KEEP_TARGETED` | Already `integration`-marked; module-scoped real-trace fixture belongs with the other benchmark files, not the required gate |
| `test_chromatix_adapter.py` | 17 | SOLVER | `M_WAVE_CHROMATIX` adapter contract — the current-milestone wave model | ~11s total (dominated by one 6.62s gradient-probe-regression test) | `KEEP_REQUIRED` | Core to current scope; already `jax`/`integration`-marked per §3 finding. The 6.62s gradient-regression test is a PB3 sizing candidate but is regression-lock evidence, not premature gradient certification |
| `test_fdtdx_adapter.py` | 5 | OUT-OF-SCOPE | Forward-only FDTD adapter; two `xfail(strict=True)` locks on known fdtdx 0.6.2/jax bugs | ~15.9s total | `KEEP_TARGETED` | FDTDX is explicitly out of current milestone scope (`AGENTS.md` §"Current Scope"); real, still-valid coverage from an earlier milestone, not obsolete |
| `test_fmmax_adapter.py` | 7 | OUT-OF-SCOPE | `M_RCWA_FMMAX` adapter contract | ~13.5s total | `KEEP_TARGETED` | Same reasoning as `test_fdtdx_adapter.py` |
| `test_sax_adapter.py` | 12 | OUT-OF-SCOPE | `M_CIRCUIT_SAX` adapter contract | ~2.5s total | `KEEP_TARGETED` | Out of current scope by `AGENTS.md`, but individually cheap — grouping is a scope-consistency call for PB2, not a cost-forced one |
| `test_ray_to_wave.py` | 20(+8 param) | COUPLER | `C_RAY_TO_WAVE` verified against an exact analytic oracle, an independent Chromatix implementation, and per-term negative controls (CHE-24) | ~3s total | `KEEP_REQUIRED` | The most central file in the repo — this is the coupler the current milestone exists to build |
| `test_ray_to_wave_node.py` | 14(+8 param) | INTEGRATION / COUPLER | `C_RAY_TO_WAVE` as an executable graph edge, bit-identical to the direct call (CHE-34/M3.5) | ~3.4s (one setup) | `KEEP_REQUIRED` | |
| `test_wave_to_ray.py` | 20(+4 param) | COUPLER / EXPENSIVE | `C_WAVE_TO_RAY` as a characterized (not certified) Monte Carlo estimator (CHE-25) | **~42s** across convergence/unbiasedness subtests, rest <0.05s | Split: structural/reproducibility tests `KEEP_REQUIRED`; the four convergence-rate and unbiasedness-ensemble parametrized tests `KEEP_TARGETED` | Same shape as `test_coupler_gradient.py`: a genuine convergence study, not shrinkable in place |
| `tests/benchmarks/test_m1_report.py` | 8 | BENCH | Structural checks on the M1 exit report — deliberately solver-free (CHE-19) | 0.01s | `KEEP_TARGETED` | Cheap, but grouped with `benchmarks/` for command-surface consistency |
| `tests/benchmarks/test_m1_bundle_projection.py` | 5 | BENCH | Unit checks on the M1 fingerprint projection — the fast counterpart to `test_m1_reproducibility.py` | negligible | `KEEP_TARGETED` | Explicitly designed as the cheap early-warning check for what the 230.89s test also catches |
| `tests/benchmarks/test_m1_reproducibility.py` | 4 | BENCH / EXPENSIVE | Full four-branch scientific-fingerprint reproduction (CHE-19) | **230.89s** | `KEEP_FULL_REGRESSION` | Needs the real end-to-end run by design; do not shrink (see `test_m1_bundle_projection.py`'s own docstring) |
| `tests/benchmarks/test_l1_ray_scaling.py` | 4 | BENCH / EXPENSIVE | L1-RAY-01 scaling/timing bundle, real Optiland catalog trace | 24.76s | `KEEP_FULL_REGRESSION` | |
| `tests/benchmarks/test_l1_wave_accuracy.py` | 17 | BENCH / EXPENSIVE | L1-WAVE-01 Chromatix accuracy bundle against a Richards-Wolf oracle | 63.52s | `KEEP_FULL_REGRESSION` | Also documents an expected-blocked case (Chromatix 0.6.0 `high_na_ff_lens`) — a negative result, not a bug in the test |
| `tests/benchmarks/test_l1_wave_scaling.py` | 4 | BENCH / EXPENSIVE | L1-WAVE-01 grid/padding/timing scaling bundle | 76.95s | `KEEP_FULL_REGRESSION` | |
| `tests/benchmarks/test_l2_coupler_bundle.py` | 17 | BENCH | L2-COUPLER-01 bundle-honesty checks (CHE-29) — does not re-derive physics | 0.29s | `KEEP_TARGETED` | |
| `tests/benchmarks/test_l2_psf_bundle.py` | 10 | BENCH | L2-PSF-01 bundle-honesty checks (CHE-39/M3.10), mirrors `test_l2_coupler_bundle.py` for a different bundle | 0.29s | `KEEP_TARGETED` | |

Totals: **0 files** proposed for `REWRITE`/`MERGE`/`ARCHIVE`/`DELETE`; **31 files**
(or file-subsets) `KEEP_REQUIRED`; **9** `KEEP_TARGETED`; **4**
`KEEP_FULL_REGRESSION` (all four inside `tests/benchmarks/`, already the same four
tests flagged as F2's cost driver).

## 6. Sign-off needed before PB2 proceeds

Per this audit, **nothing requires sign-off to reclassify, rewrite, merge, archive,
or delete** — there is no such proposal in §5. Two items still need the user's
attention, but neither blocks PB2 from starting:

1. **File a follow-up issue**: regenerate and commit
   `benchmarks/probes/records/m3r_sensor_handoff.json` (Finding F3). This is solver
   probe work, not test-structure work, so it does not belong in PB2/PB3, but it
   should not be forgotten — right now, 21 tests documenting CHE-38/CHE-47's sensor-
   handoff verdict are silently skipped in every fresh checkout of this repository.
2. **Confirm the marker-axis recommendation in §3** (separate "requires optional
   solver install" from "expensive" from "out of current milestone") before PB2
   commits to specific marker names, since PB2's own acceptance criteria defer that
   naming decision to this audit's inventory.

## 7. What this hands to PB2

- A complete cost map (§2) showing PB2 needs to route ~400s of `tests/benchmarks/`
  content plus ~150s of scattered characterization tests (`test_coupler_gradient.py`
  in full; four parametrized cases in `test_wave_to_ray.py`; one test each in
  `test_coupler_round_trip.py`, `test_m3r_sensor_handoff.py`, `test_optiland_ray_benchmark.py`)
  out of the required gate to have any chance at the ≤3-minute Tier A target.
- Doing only that (no shrinking) leaves an estimated Tier A of roughly
  635.2s − 396.1s (four benchmark fixtures) − 74.6s (`test_coupler_gradient.py`) −
  ~42s (`test_wave_to_ray.py` convergence/unbiasedness subset) − 13.2s
  (`test_coupler_round_trip.py`'s one test) − 15.9s (`test_fdtdx_adapter.py`) −
  13.5s (`test_fmmax_adapter.py`) − 2.5s (`test_sax_adapter.py`) − 6.6s
  (`test_optiland_ray_benchmark.py`) ≈ **71s** for the remaining ~33 files — already
  comfortably under the 3-minute gate without any problem-size shrinking. If that
  number holds after PB2's actual restructuring, **PB3 may turn out to be a
  documentation-only issue** (command-surface docs) rather than requiring further
  shrinkage; PB3's own acceptance criteria already anticipate this
  ("if any test cannot be cheapened... it is reclassified... rather than force-fit").
- The disposition table above at file (and, where a file mixes fast/slow content,
  test) granularity, ready to drive marker assignment.
