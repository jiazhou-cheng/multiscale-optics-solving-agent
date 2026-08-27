# M4.1 exit — the three B3 families, the deleted entry point, and two findings

CHE-116 (M4.1). This report closes M4.1. It covers only what was not already
landed by the partial commit `2c1d42b` (the three families themselves) or by
CHE-115 (the executor migration that was M4.1's blocking precondition).

Two of the four remaining acceptance criteria were satisfied. Doing them
produced two findings that were not, and both are recorded in the fields a
caller reads rather than only here.

---

## 1. The last bespoke entry point is gone

`benchmarks/physics/L2-PSF-01/run_benchmark.py` — 600 lines, deleted.
`evaluate.py` — 63 lines, deleted with it: its only input was the bundle that
runner wrote, so it could not be run after the runner went.

CHE-115's own amendment made this deletion conditional on the executor
reproducing the case, and that condition is met: `GraphExecutor` over
`examples/graphs/psf_singlet_sensor.yaml` returns
`fft_oracle_intensity_relative_l2 = 0.0022072391812867093` as the same float64,
re-verified in this issue after the family edit below
(`benchmarks/instances/records/B3-PSF-SINGLET-01.json`, `== ` not `approx`).

**What the runner owned, and where each piece is now.** Nothing physics-bearing
was lost, because the runner *called* the probes rather than re-deriving them:

| the bundle's piece | where it is now |
|---|---|
| the three-node graph and its frozen gate | `examples/graphs/psf_singlet_sensor.yaml` + `benchmarks/instances/b3_psf_singlet.py` |
| `opl_sign_flip` negative control | a graph variant through `runtime.variants.with_config_overrides`; fires at 0.00220724 → 3.85714, margin 1747.5× |
| `near_sensor_fine` three-node demonstration | `b3_psf_singlet.run_near_sensor_fine` |
| the 12-rung ladder, the O2 ASM/RS oracle, absolute-power convergence | unchanged in `benchmarks/probes/quadrature_weight.py` and `benchmarks/probes/sensor_handoff_convergence.py`, each with its own record |
| the `exit_pupil_hard_support_reconstruction` (O4) validity-limit control | unchanged in `sensor_handoff_convergence.py::_exit_pupil_negative_control` |
| `result.json` / `provenance.json` / `arrays.npz` / `convergence.json` / `plot.png`, and `evaluate.py`'s hash check | superseded by the stamped instance record |

**One thing was deliberately not carried forward:** the
`quadrature_weight_regression` pass/fail restatement. CHE-117 established that
its verdict flips sign with ray count (10.7 at 8 rings, 1.02 at 181, 0.42 at
512, 0.69 at 1024) — it measures where two convergence curves cross, not whether
the weight is right. The family keeps it as `inverted-quadrature-weight` with
`KNOWN_FIRES_BACKWARDS` and that finding attached; nothing gates on it. The
underlying measurement is untouched in the probe record.

**Callers repointed**, so the deletion leaves nothing pointing at a corpse:

- `benchmarks/perf/run_baselines.py`: `l2-psf-01` → `b3-psf-singlet`, now timing
  `benchmarks/instances/b3_psf_singlet.py`. The committed `l2_psf_01_cpu.json`
  (170.1 s) is **not** overwritten and is **not** the before-arm of a speed-up:
  it measured a wider workload (ladder + O2 + two controls), so 170.1 s against
  48.6 s compares scopes, not implementations. New record:
  `benchmarks/perf/records/b3_psf_singlet_cpu.json`.
- `benchmarks/inventory.yaml`: the two `run_benchmark.py` rows and the
  `evaluate.py` row removed — a row for a path that is not there reads as
  coverage and is not. `INVENTORY.md` regenerated.
- `benchmarks/manifest.yaml`, `benchmarks/physics/L2-PSF-01/README.md`,
  `tests/test_retired_taxonomy.py`'s allowlist reason,
  `tests/test_benchmark_inventory.py`'s docstring.

`tolerances.yaml` **stays**. It is the file `b3_composed.py` migrates its
tolerance bases from verbatim, and a test compares them character for character.

Pinned by `tests/test_b3_b4_families.py::test_the_last_bespoke_entry_point_for_the_singlet_is_gone`.

---

## 2. Runtime and memory envelopes, recorded rather than declared

New probe: `benchmarks/probes/b3_energy_accounting.py`. Record:
`benchmarks/probes/records/b3_energy_accounting.json`.

| family | declared wall / memory | observed wall | observed peak RSS | inside? | device |
|---|---|---:|---:|---|---|
| `B3-PSF-SINGLET` | 600 s / 16 GiB | 26.6 s | 11.12 GiB | yes / yes | CPU |
| `B3-DUALROUTE` | 600 s / 16 GiB | 0.67 s | 1.02 GiB | yes / yes | CPU |
| `B3-DEMO2` | 600 s / 40 GiB | 94.9 s | 2.61 GiB | yes / yes | 1 × A6000 |

Every figure is read out of `b3_energy_accounting.json`, so the table and the
record cannot drift. The two CPU rows are measured in-process under
`core.resources.MemoryWatchdog`, so their peak is the run's own.

Container cgroup swap growth was **0** on both measured runs, which is the
shared-host condition that matters more than either number.

**Every case fits one GPU.** The two CPU cases use none. `B3-DEMO2` ran on one
A6000 with its 1.6e8 rays chunked 40 ways specifically to hold the working set
to ~6 GB; the paper's own Table S2 figure for the same route is 29.2 GB in 2
batches, which is why the 40 GiB envelope is declared and why one device is
enough.

**`B3-DEMO2` was not re-run**, and that is a scope decision rather than an
omission. Its envelope is read off the committed GPU runs
(`benchmarks/probes/records/ray_wave/demo2_paper_jax.json`,
`benchmarks/perf/records/demo2_paper_rw_p_ramp_sum_cuda.json`): a 95-second
1.6e8-ray job whose numbers are committed CHE-96/CHE-101 evidence, and
re-running it here would restamp records this issue has no reason to move.

**Which peak, and a defect this issue's first draft had.** The perf record
carries two: `memory_report.peak_rss_bytes` = 29.4 MB and
`subprocess.peak_child_rss_bytes` = 2.80 GB. The harness times a *subprocess*, so
the first is the harness's own footprint and never samples the run; the second is
the run. The first draft used the first, and so produced
`memory_inside_declared_envelope: True` against a 40 GiB envelope out of a number
95× too small — a passing verdict manufactured from a value the record itself
declared irrelevant. Independent review caught it. The record now uses
`subprocess.peak_child_rss_bytes` (2.61 GiB), keeps the harness figure under a
name that says what it is, and `_fits` returns `None` rather than a verdict when
the observed value is absent. The report table above no longer shows "—" beside
"yes".

**The device side of demo2's envelope is UNMEASURED and is labelled so.** 2.61
GiB is *host* RSS. Neither committed record reports CUDA peak bytes for this run
(the perf record's `cuda` field is `null`), and the 29.213 GB / 8.086 GB figures
in the science record are the *paper's* Table S2 numbers for its own
implementation, not ours. The 40 GiB envelope was sized against the 40-chunk
plan's working set, and no measurement in this repository confirms it on the
device.

The singlet's 11.12 GiB against a declared 16 GiB is the one number here with
little headroom, and it is a real in-process measurement of the frozen
787,969-ray configuration rather than of a cheaper proxy.

Pinned by `test_every_b3_family_has_a_recorded_runtime_and_memory_envelope`.

---

## 3. FINDING — the energy-accounting intermediate cannot be measured, on any of the three

M4.1's fourth criterion asks for the intermediate to be *checked*:

> Intermediate invariants are checked, not just the final result: energy
> accounting at the handoff plane, and the declared-versus-actual reference
> plane. A correct final image can hide an incorrect intermediate convention.

All three families declare one and all three gate on it:

| family | invariant / metric | threshold | `may_gate` |
|---|---|---:|---|
| `B3-PSF-SINGLET` | `HANDOFF_ENERGY_CLOSES` / `handoff_power_ratio` | 1e-3 | True |
| `B3-DEMO2` | `PATCH_ENERGY_CLOSES` / `patch_handoff_power_ratio` | 1e-3 | True |
| `B3-DUALROUTE` | `route_power_ratio` — the family's *whole* gate | 1e-2 | True |

**None of the three is formable from the shipping surface, and the reason is one
convention rather than three bugs.**
`couplers.ray_to_wave.ReconstructionReport` exposes exactly two power figures and
they are incommensurable:

- `reconstructed_discrete_power` is `ComplexField.discrete_power()` =
  `sum(|u|^2) * dy * dx` — an integral over the output plane;
- `incident_amplitude_power_sum` is `sum(|amplitude|^2)` over rays — a bare sum,
  and `couplers/handoff.py` declares
  `amplitude = sqrt(weight) * quadrature_weight_m2`, putting the per-ray area
  element *inside the field*.

Measured (`b3_energy_accounting.json`):

| family | reconstructed power | incident sum | quotient | against a gate of |
|---|---:|---:|---:|---:|
| `B3-PSF-SINGLET` | 1.353e-24 | 4.847e-20 | 2.79e-5 | 1e-3 |
| `B3-DUALROUTE` (20°, 64 rings) | 5.819e-21 | 4.978e-13 | 1.17e-8 | 1e-2 |
| `B3-DEMO2` (first of 40 chunks) | 4.103e6 | 1.935e11 | 2.12e-5 | 1e-3 |

(The absolute magnitudes are not comparable between rows and are not meant to be:
Optiland's per-ray intensity weights have no SI calibration, which is a separate
recorded fact and a second reason nothing here is a watt.)

### The ray-count arm: what it shows, and one claim withdrawn in review

The arm runs the same `B3-DUALROUTE` configuration at 32 and 64 hexapolar rings,
nothing else changed.

| rings | rays | quotient | quotient ÷ grid area | border energy (flat field would give) | `ray_density_status` |
|---:|---:|---:|---:|---|---|
| 32 | 3,169 | 1.054004e-08 | **1.0052** | 7.678e-3 (7.797e-3) | `wavelet_approximation_holds` |
| 64 | 12,481 | 1.168898e-08 | **1.1147** | 8.759e-3 (7.797e-3) | `not_computed_above_scan_limit` |

Rays × 3.938 → **Σ\|a\| × 0.99982, Σ\|a\|² × 0.2519**, quotient × 1.1090.

**What this shows.** `sum(|amplitude|)` is invariant to 0.02% under refinement
while `sum(|amplitude|²)` falls by very nearly the ray-count factor. That is the
squared area element sitting in the denominator, measured directly rather than
inferred from the dimensional argument.

**One claim was withdrawn.** The first draft of this report, of the probe record,
and of `B3-DUALROUTE`'s gate note read the 10.9% drift between the rungs as
ruling out a missing calibration constant — "a constant could be absorbed; this
moves, so it is not one". **Independent review showed that inference is not
supported by these measurements, and it is retracted.** The reason is in the table
above: at 32 rings the quotient sits within 0.5% of the plain grid area
`512² × (0.2 µm)² = 1.0486e-8 m²`, and the measured border energy matches what a
perfectly flat field would give — every ray splats a full-grid ramp, so on this
configuration the reconstruction is dominated by a near-uniform floor rather than
by the focal spot. The drift between the rungs is the spot's share rising above a
floor that falls as 1/N. Two points on a floor-dominated grid decide nothing about
a constant, and the arm was run on the least informative of the three cases
(the singlet's quotient is 1700× *its* grid area, i.e. focus-dominated).

So: **whether some per-configuration normalization would close the quantity is
UNTESTED**, and the record, the family note and the test all say so. The
dimensional finding does not depend on it.

### What was NOT done about it, and why

Nothing was widened. No `may_gate` was dropped to `False`. No corrected
normalization was invented, and no inference beyond the measurements was kept.

Forming the right quantity needs the incident power
`sum(weight_i * quadrature_weight_m2_i)` — intensity times area, once — plus an
argument about what the coherent wavelet sum's cross terms do to it. That is a
conservation claim across a representation boundary: it owes a derivation, an
oracle that is not the coupler, and independent review, and it is not in M4.1's
scope (three B3 families with decidable oracles, and a deletion). Inventing a
factor that makes the number read 1.0 is the fabrication the verification layer
exists to prevent.

`benchmarks/instances/b3_psf_singlet.py` had already reached this conclusion for
the singlet alone and said so in its measurement note — "nothing here measures
the traced bundle's power, so it is reported without a tolerance rather than
gated on the wrong quantity". What is new is that it is a property of the
boundary shared by all three families, and that it is now measured.

**Recorded where a caller reads it:** `B3-DUALROUTE`'s `gate_disposition` note
now gives this as the reason its gate is `NOT_MEASURED`, naming the other two
families' identical invariant, the measured numbers and the record. The reason
moved from "PB7 did not measure it" to "the quantity is not formable from the
shipping surface", which is a sharper open question and a worse thing to have
left implicit.

**Follow-up, not filed by this issue:** the ray-to-wave absolute-power
normalization, and whether `ReconstructionReport` should carry an incident-power
field at all. Whoever takes it inherits three gating tolerances that will start
deciding things the moment the quantity exists.

---

## 4. FINDING — B3-DEMO2's oracle named the paper's figure; every number is against our own ASM

M4.1's first criterion requires each family to declare an "oracle with declared
independence", and the ticket's stated reason for promoting `B3-DEMO2` is
externality:

> the one composed reproduction that works, and the only one graded against an
> **external published** reference rather than against ourselves. That makes it
> the regression anchor.

The family declared that faithfully — `description` read "the published figure
from the paper's own implementation ... a different group's code, run before this
project existed", `reference` was "ACS Photonics 2026 demo2, SI Table S2", and
both metrics said "against the published intensity".

**No such comparison exists in this repository.** Every number the family
carries, including the `0.999418` in its `gate_disposition`, is
`routes.rw_p.vs_oracle.ncc_intensity` from
`benchmarks/probes/records/ray_wave/demo2_paper_jax.json`, measured against
`verification/asm_oracle.angular_spectrum_float64` — this repository's own float64
angular-spectrum propagator. The paper supplies Table S2's summary numbers (NCC
0.997, MSE 4.414e-10, each against the *paper's* own oracle) and a printed
figure, not a field array, so there is nothing to compute a metric against. The
probe record says as much itself, in
`paper_numbers_are_context_not_thresholds`: *"Quoting NCC 0.997 as a gate would
be circular validation."*

A near-coincidence made this easy to miss: the RW-P ↔ RW-F **route agreement** at
the same budget reads 0.9994180762 while the **oracle** comparison reads
0.9994182326. The number carried is the oracle one; two different quantities
agree to six figures in the same record.

### What was corrected, and what was not

**Corrected — attribution only.** `oracle.description` and `oracle.reference`
now name `verification/asm_oracle.angular_spectrum_float64` as the decider and
state plainly that the published figure is not an oracle here and never was; both
metric descriptions say what they measure; the `gate_disposition` note names the
record field the number comes from; the family's `notes` quote the original
promotion claim, say it does not hold, and give the one that does. `FIXED_V1`'s
justification for `B3-DEMO2-01` was corrected the same way.

**Not changed.** `independence` stays `INDEPENDENT`, which is correct on the
enum's own terms — "shares no code and no traced data with the thing under test".
The thing under test is `C_PATCH_WFT` + `C_RAY_TO_WAVE`; the oracle is a
different algorithm on the same input mask and shares neither their kernel nor
their traced rays. **No threshold moved** (`demo2_ncc` 0.999, `demo2_relative_l2`
5e-2). This is a re-attribution, not a re-scoping, and the test asserts the
thresholds unchanged precisely so that stays visible. The observed value gained
digits and did not move: `0.999418` → `0.9994182326189224`, because at six
figures it is indistinguishable from the RW-P/RW-F route agreement
(`0.9994180762008337`) and the two are different quantities. A test now looks the
full-precision value up in the record field it claims to come from, so the
attribution is executable rather than argued.

### The objection this correction created, raised rather than buried

Independent review pointed out that the corrected declaration is in tension with
the same file: three hundred lines above, `B3-PSF-SINGLET` declares an ASM oracle
with `may_gate=False`, on the grounds that "using our own numerical code as the
answer key for our own numerical code is circular validation" — and `B3-DEMO2`
now truthfully names an in-repository ASM propagator as its decider with
`may_gate=True`. Before the correction the tension was hidden behind a false
claim of externality. After it, it has to be argued or escalated.

Both were done, in the oracle description itself:

- **The argument is about the input, not the algorithm.** The singlet's O2 is
  built from a ring-averaged, linearly interpolated fit to *the coupler's own
  traced pupil* (`sensor_handoff_convergence.py::_traced_pupil_wavefront`), so it
  shares the traced data and inherits that fit's unresolved resolution error —
  which is exactly how it once set a control floor that had to be retired.
  demo2's ASM starts from the physical input mask and shares no traced ray with
  either coupler; it is the same reference `B1-WAVE` and `B0-contract` already
  use. That satisfies the enum literally.
- **The `may_gate` decision is flagged, not resolved.** Whether an
  in-repository propagator may decide a gate *at all*, given the recorded
  no-circular-validation rule, is the owner's call. `may_gate` stays `True` on
  both demo2 tolerances and this issue did not change it. Dropping it silently
  would be a weakening with no mandate; keeping it silently without stating the
  tension would be worse. Nothing enforces it today — the gate is
  `MEASURED_OFF_GATE` and no collection re-runs the case.

**This is the one item in this issue that wants an owner's decision rather than
more evidence.**

**Not softened.** `sampler_absent_note` still says "external independence",
which the same issue found unsupported one field above. That was left alone on
purpose: the reason this case must not be generated is that its configuration
*and ray budget* are the paper's and a drawn point has neither, and that argument
survives intact. Rewriting the note to look consistent would have hidden the
correction.

**The B3 classification survives.** An oracle independent of the couplers under
test is exactly what B3 requires; "independent of the coupler" and "external to
this repository" are different guarantees, and only the first was ever needed for
the category. What does not survive is the claim that this family is the one
graded against something outside the repository. As of this issue, **no B3 family
is** — `B3-PSF-SINGLET`'s O1 analytic Airy is the closest thing, and it is a
closed form rather than an external measurement.

Pinned by `test_demo2_names_the_oracle_that_actually_decided_its_number` (which
replaces `test_demo2_is_graded_against_something_outside_this_repository` and
quotes what that test used to assert) and
`test_the_demo2_promotion_argument_was_corrected_not_dropped`.

---

## 5. What is still open on M4.1's criteria

| criterion | state |
|---|---|
| three B3 families with the full field set | met (`2c1d42b`) |
| tolerance bases migrated verbatim | met, character-for-character test |
| **each family runs through the M3 executor and the M0.5.3 verifier** | **partly met.** `B3-PSF-SINGLET` does both (CHE-115, re-verified here). `B3-DEMO2` and `B3-DUALROUTE` do neither — see below. |
| intermediate invariants checked | **checked, and they do not close.** §3 |
| negative controls declared and demonstrated to fire | partly met. The singlet's `opl-sign-flip` fires through the executor at 1747.5× margin. The other declared controls are honestly reported `NOT_RUN` in the record, with `inverted-quadrature-weight`'s backwards firing attributed. `B3-DEMO2`'s and `B3-DUALROUTE`'s are unrun, for the reasons below. |
| `run_benchmark.py` deleted | met. §1 |
| envelopes recorded, one GPU | met. §2 |

### Why `B3-DEMO2` and `B3-DUALROUTE` are not enrolled, specifically

**`B3-DEMO2` cannot be a graph document at all.** CHE-115 established and
recorded the reason in the executor's docstring: no registered model emits or
consumes what `C_PLANAR_DOE_STEP` needs — demo2 is a bare SLM and a sensor with
no refractive surface — and RW-P's 1.6e8 rays in 40 chunks hit the executor's
streaming refusal. That is an M3.1 capability gap, filed there. Inventing an
`M_SOURCE_ARRAY` to unblock one benchmark would register a capability no oracle
has checked, which CHE-115 explicitly declined and this issue does not overturn.
The verifier half is reachable via `runtime.instance_runner.record_from_probe` —
the mechanism eight other family drivers use — and is left to whoever closes the
executor gap, so the two halves land together rather than leaving a
`record_from_probe` driver to be rewritten immediately.

**`B3-DUALROUTE` has no measurable gate metric.** Its entire gate is
`route_power_ratio`, which §3 shows is not formable. A verifier run would
produce a `VerificationResult` whose only content is
`metric_missing_from_record`, which is what the family's `gate_disposition`
already says in words and with numbers. Enrolling it would add an artifact
without adding a measurement. It becomes enrollable the moment the normalization
question in §3 is settled, and not before.

Both are stated as open rather than closed by a widened tolerance or a
plausible-looking driver.

---

## 6. Verification run

| suite | command | result |
|---|---|---|
| Default gate | `./run.sh --no-build pytest -q` | **8 failed, 2724 passed, 67 skipped**, 57 s — all 8 pre-existing, attributed below |
| Slow | `./run.sh --no-build pytest -q -m slow` | **43 passed**, 101 s |
| Targeted | `pytest tests/test_b3_b4_families.py -n 0` | 43 passed (5 new tests) |
| Targeted | `pytest tests/test_benchmark_inventory.py tests/test_retired_taxonomy.py -n 0` | 48 passed |
| Provenance | `pytest tests/test_provenance_fingerprint.py -n 4` | all 71 instance records green |
| Lint | `ruff check` on every file touched | clean (one pre-existing import-sort error in `b3_composed.py` predates this change, verified by stash) |

**Not run:** GPU tests (`make test-gpu`) — nothing here changes GPU behaviour, and
`B3-DEMO2` was deliberately not re-executed. `make test-serial` — no cross-test
interaction was suspected; the two family edits were followed by a full record
regeneration whose fingerprints are reported below.

**Instance records regenerated.** Editing `b3_composed.py` and `fixed_suite.py`
invalidated all 71 stamped instance records, and all 71 were re-run through their
drivers. **70 of 71 scientific fingerprints came back bit-identical.** The one
that moved is `B0-META-01`, whose refusal message carries a `uuid4` and therefore
rehashes on every run — a known non-determinism, not a measurement change. The
singlet's frozen gate number is bit-identical after the edit, re-checked with
`==`.

**Independent review** was obtained (required by the ticket for couplers,
composed boundaries, oracle independence, tolerances and resource behaviour). It
returned three must-fix findings and all three were fixed, not argued down:

1. the ray-count arm's "rules out a calibration constant" inference was
   unsupported — withdrawn from the probe, the record, the family note, this
   report, two inventory rows and the test that pinned it; the reviewer's own
   measurement (quotient ≈ grid area, floor-dominated grid) is now recorded;
2. `B3-DEMO2`'s corrected oracle contradicted `B3-PSF-SINGLET`'s O2
   non-gating rule — the distinction is now argued on the input and the
   `may_gate` question escalated to the owner rather than resolved here;
3. demo2's memory verdict was computed from the harness process's RSS —
   now `subprocess.peak_child_rss_bytes`, with `None` rather than a verdict when
   the value is absent.

Two lower-severity findings (a wall-time figure that disagreed with the record it
cited; a truncated observed value that made the oracle attribution unresolvable)
were also fixed.

**Pre-existing failures not caused by this issue**, each verified by stashing
this issue's changes and re-running:

- `test_fixed_suite.py::test_success_metric_s1_is_met` and
  `test_every_registered_family_contributes_at_least_one_instance` — both failed
  identically before this issue; filed as `claim_ledger` gaps by CHE-146 and by
  M2.8–M2.12's six unenrolled families.
- Six `test_provenance_fingerprint.py` failures over **probe** records, whose
  verdicts name `changed_files=('src/couplers/ray_to_wave.py',)`. That change
  arrived in the `origin/main` merge `eb3d792` and is a comment dedupe plus the
  removal of a duplicate `edge_tolerance = 1e-6` assignment — numerically inert,
  but it moves the normalized source digest. Regenerating those records is the
  merge's debt, and four of them are the expensive singlet ladder probes. On the
  merge commit without this issue's changes the same file fails **77** tests; with
  them it fails **6**.
