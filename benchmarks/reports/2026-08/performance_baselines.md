# Performance baselines and the harness that produced them

CHE-105 (M0.4), completed by CHE-129. Written after the measurements, not before
them.

Every number below is transcribed from a committed record under
`benchmarks/perf/records/`. The first version of this report was not: it was
written against an earlier execution than the records that shipped, and
disagreed with them on every headline figure including the *sign* of the coupler
overhead. That is the failure this document exists to prevent, so the tables are
now read out of the JSON rather than remembered.

## What this exists to prevent

CHE-96 attributed the whole of demo3's runtime to the reconstruction stage.
CHE-101 then made that stage 9.6× faster on the kernel and the end-to-end run
went `207 s → 197 s`, because the stage was 7% of the cost. The work was correct
and the target was wrong, and a stage-resolved baseline would have said so
before the effort rather than after it.

M5 is a whole milestone of that shape, so this one measures first.

## The finding

**At the frozen `M3-SINGLET-REF` configuration — 3169 rays, which is what the
M3 correctness benchmarks run — 91% of `C_RAY_TO_WAVE`'s wall time is a sampling
diagnostic, not physics.**

Scoped deliberately, because the demo baselines further down this report measure
this repository's *largest* workloads at 6e7–1.6e8 rays, where the diagnostic is
skipped entirely and the target is `optiland_trace + emit_patch_spectra` (87.6%
of demo3 **as measured here** — CHE-118 has since removed 93.5% of the
`optiland_trace` term, so on current code the target is the emitter alone at 72.2%
of a 120.36 s run; see `optiland_trace_profile.md`). Both findings are real and they point at different code. An earlier
draft stated the 91% as a property of "the ray count this repository actually
uses", which conflated them.

`couplers.ray_to_wave._ray_density_diagnostic` runs an O(N²) pairwise
nearest-neighbour scan below `_NEAREST_NEIGHBOUR_SCAN_LIMIT` (4096 rays) and is
skipped above it. Measured share of the call, at fixed 188² grid
(`scaling_ray_axis.json`):

| rays | diagnostic runs | full call | reconstruction only | diagnostic share |
| --: | :-: | --: | --: | --: |
| 817 | yes | 0.0368 s | 0.0109 s | **70.3%** |
| 1801 | yes | 0.1533 s | 0.0229 s | **85.1%** |
| 3169 | yes | 0.4488 s | 0.0399 s | **91.1%** |
| 4921 | no | 0.0631 s | 0.0735 s | — |
| 7057 | no | 0.1105 s | 0.1119 s | — |
| 12481 | no | 0.2031 s | 0.2033 s | — |

Seven repeats per point, not the protocol's three, and that is a correction this
report has to own. At three repeats, four realizations of this sweep returned
ray-density-diagnostic exponents of **2.071, 2.020, 1.609 and 2.040** — a spread
of 0.46 on the number quoted below as evidence that a pairwise scan is quadratic.
The outlier came from the 817-ray row, whose call is ~0.04 s on a shared 80-core
box where a 3-repeat median is not a median of anything stable. At seven repeats,
three realizations give 2.013 / 2.057 / 2.038, a spread of 0.044. The whole sweep
is ten seconds, so this cost nothing and the earlier exponent should not have
been quoted to four digits from one run.

What was stable at three repeats and stays stable: the 3169-ray diagnostic share
(90.8–91.1% over six realizations) and the reconstruction exponent (1.05–1.13).
The headline 91% never depended on the noise.

3169 rays is the frozen `M3-SINGLET-REF` configuration. Note the row above and
below it: **4921 rays runs 7.1× faster than 3169 rays**, because it crosses the
threshold and stops paying for the scan.

The last three rows report **no** diagnostic share, and that is deliberate.
Above the scan limit the diagnostic does not run in either arm, so the difference
between them is run-to-run noise between two identical computations — it was
previously published as `-10.8%` and `-2.5%` under a field name a downstream tool
reads as a share. It is now `null`, with the difference kept under
`arm_difference_s` so the noise floor stays visible. At seven repeats those three
rows agree to within 1.1%, 1.3% and 0.1% — which is the floor the 70/85/91%
shares should be read against, and they survive it comfortably.

Two fits, because one exponent across that threshold describes neither side:

| fit | exponent | r² | spread over 3 realizations |
| -- | --: | --: | --: |
| reconstruction only | 1.0920 | 0.9960 | 1.092 – 1.134 |
| ray-density diagnostic | 2.0378 | 1.0000 | 2.013 – 2.057 |
| both together (do not use) | 0.3791 | **0.1770** | — |

The third row is the argument for the first two. It is also the argument for
fitting at all rather than quoting endpoints: two points would have produced an
exponent and an r² of 1.0 regardless.

Consequences:

* At the frozen 3169-ray configuration, optimizing the reconstruction kernel
  moves **8.9%** of the wall time. **The diagnostic is the target there**, and it
  is not physics — it is a sampling check that could be sampled rather than
  computed exhaustively, or cached across calls on the same bundle.
* That ~9% is **not** a constant across the sub-4096-ray range. The share rises
  as the ray count falls, reaching **29.7%** at 817 rays, because the diagnostic
  is quadratic and the reconstruction is linear. The earlier version of this
  report stated the 9% as a property of the whole range; it is a property of one
  configuration. Read the per-row share for a given ray count.
* The reconstruction is **linear in rays at fixed grid** (1.0920), consistent
  with the O(rays × pixels) product model that `RayToWaveCoupler.estimate()`'s
  docstring argues for, and not with the registry's `O(rays + pixels)`.
* None of this describes the demo workloads. At 6e7 rays the diagnostic is
  skipped and the reconstruction is 6.7% of demo3 — see the demo baselines below,
  which is the number M5 should be planning against.

## L2-PSF-01 could not run

Attempting to baseline the bundle found it exits `1` in 0.5 s:
`run_benchmark.py` loaded `benchmarks/probes/m3r_sensor_handoff.py` and
`m3_quadrature_weight.py`, the names those probes had before the CHE-93
reorganization renamed them. The module docstring named the correct paths; only
the two constants below it were left behind.

Nothing noticed because no test invokes the bundle and the manifest records its
verdict rather than re-deriving it. Fixed here. On the repaired path it runs in
**170.1 s**, peak child RSS 12.48 GB, and reproduces the state the claim ledger
records: `gate_met_on_production_configuration: false`,
`negative_controls_pass: false`.

Its `scientific_fingerprint` is now `411181d0…` against the `b073a461…` in
`docs/architecture/overnight_run_2026_08_22.md`. Attributed to CHE-102, which
moved host traces from Optiland's torch backend to NumPy; that changes the trace
at float64 round-off and therefore the fingerprint. No test asserts the old
value. Note that `overnight_run_2026_08_22.md` still presents `b073a461…` as a
fingerprint "held constant across every phase", for a window in which this report
shows the bundle could not execute at all; that document's claim is left standing
and is worth a correction by whoever owns it.

This is also why the runner **refuses to record a nonzero exit**. The first
attempt wrote a 0.5-second "bundle baseline" and was, as a number, perfectly
real. The refusal now has a test.

## Framework overhead (S5)

Same physics, through the abstraction and directly (`framework_overhead.json`):

| | framework | direct | ratio |
| -- | --: | --: | --: |
| `M_RAY_OPTILAND` — `adapter.run` vs `build_optiland_system` + `optic.trace` | 19.93 ms | 8.29 ms | **2.41×** (+11.65 ms) |
| `C_RAY_TO_WAVE` — `node.transform` vs `ray_to_wave` | 463.03 ms | 454.76 ms | **1.018×** (+8.27 ms) |

The solver's overhead is a fixed ~12 ms of validation, precision/device
negotiation, `.npy` write and artifact assembly. It matters for a graph of many
small trace nodes and is irrelevant for one large one.

The coupler node's `+8.27 ms` on a ~455 ms kernel is 1.8%, which is inside the
~10% run-to-run spread the scaling rows measure at 3 repeats. It is reported as
measured rather than rounded to 1.0, but it should be read as "not resolved by
this measurement" rather than as a cost. The earlier version of this report had
the direct arm *slower* than the framework arm and described the ratio as
"slightly under 1.0"; that was a transcription error against the record, and the
sign has always been the other way.

## `estimate()` versus measured

From `estimate_accuracy.json`:

| component | predicted | measured | verdict |
| -- | --: | --: | -- |
| `M_RAY_OPTILAND` | `None` | 20.6 ms | **Superseded by CHE-118 (M5.1).** This row read "no prediction, and it does not know the surface or traced-ray count". Both gaps are now closed by measurement: `estimate()` reads the surface count from the prescription, converts the ring count to a traced ray count, and calls a calibrated affine cost model. It still returns `None` on a host whose environment fingerprint is not the calibration's, which is a different and narrower refusal. Re-scored in `estimate_accuracy.json`. |
| `C_RAY_TO_WAVE` | 0.2036 s | 0.4643 s full / 0.0538 s reconstruction | **Wrong in both directions.** Under-predicts the shipping call by 2.28× because it models the reconstruction and the call is dominated by the diagnostic. Over-predicts the reconstruction by 3.78× because `_RAY_PIXEL_PRODUCTS_PER_SECOND` is not calibrated to this host. |

M3's executor and M6's planner both intend to order work by `CostEstimate`. On
this evidence neither can yet — for `C_RAY_TO_WAVE`. `M_RAY_OPTILAND` was fixed by
CHE-118; see `optiland_trace_profile.md`.

## The demo baselines (CHE-129)

Deferred by CHE-105 at the owner's direction and recorded here. All four are
single-GPU (`environment.gpu_count == 1`, enforced by
`test_a_gpu_baseline_was_measured_on_exactly_one_device`), run sequentially with
nothing else of this project's on the card, swap growth zero throughout. They
were launched with `MOA_GPUS=device=6`; see the note below on which parts of that
the record can and cannot attest to.

Both demo3 rows are **pre-CHE-118** measurements: that issue cut the trace stage
from 98.96 s to 6.32 s, so demo3's whole-command figure is now 120.36 s. The rows
are left as measured rather than edited — they are what the code cost at
`6846ec3` — and the after-figures are in `optiland_trace_profile.md`. The demo2
rows are unaffected: demo2 does not trace through Optiland.

| baseline | rays | whole command | demo's own clock | s/ray | peak child RSS | record |
| -- | --: | --: | --: | --: | --: | -- |
| demo2 `paper` RW-F, Table S2 budget | 1.1e6 | 6.37 s | 3.51 s | 5.795e-6 | 2.24 GB | `demo2_paper_rw_f_paper_budget_ramp_sum_cuda.json` |
| demo2 `paper` RW-P, Table S2 budget | 1.6e8 | 95.66 s | 92.77 s | 5.979e-7 | 2.75 GB | `demo2_paper_rw_p_ramp_sum_cuda.json` |
| demo3 `characterization` RW-P, `ramp_sum` | 6.0e7 | 211.95 s | 205.40 s | 3.533e-6 | 2.89 GB | `demo3_characterization_rw_p_ramp_sum_cuda.json` |
| demo3 `characterization` RW-P, `kspace_splat` | 6.0e7 | 202.90 s | 196.25 s | 3.382e-6 | 2.94 GB | `demo3_characterization_rw_p_kspace_splat_cuda.json` |

Two clocks, because the difference matters at the short end. **Whole command** is
what the harness times and what `s/ray` divides: interpreter start, JAX
compilation, the plan setup and the record write are inside it. **Demo's own
clock** is the probe's internal `wall_clock_s`, summed over routes and seeds. On
demo3 the gap is 3%; on demo2 RW-F it is **45% of the 6.37 s**, so that row's
`5.795e-6 s/ray` is mostly fixed cost and should not be read as a per-ray rate at
all. Quote the demo3 rows for scaling and the demo2 RW-F row only for "what does
running this cost".

`seconds_per_ray` is also not one quantity across these rows: `ramp_sum` is
O(rays × pixels) and `kspace_splat` is O(rays) plus one FFT, and demo2 and demo3
are different systems. `core.performance.compare` refuses the divisions that
would look like speedups.

The ray counts are **read from each demo's own record**, not from the `--rays`
argument the operator typed. Both agree here, which is the point of checking.

One provenance defect in these four records, found in review and stated rather
than papered over: they all carry `environment.container_image: "agent_solver"`
even though they ran on `agent_solver_gpu`. `run.sh` switches the image but never
exports `MOA_IMAGE`, which is what both fingerprint functions read. The evidence
that these are GPU runs is therefore `packages.torch == 2.13.0+cu126` and
`gpu_name: "NVIDIA RTX A6000"`, not the image field — and the image component of
`EnvironmentFingerprint.sha256` is currently dead weight, so `compare()`'s
cross-image refusal rests entirely on the torch build. `MOA_GPUS=device=6` is
likewise not recorded: `visible_devices` is `null` and only `gpu_count: 1`
survives into the record, which is the part `AGENTS.md` actually requires to be
checkable. Fixing `run.sh` to export `MOA_IMAGE` is follow-up work; it is not
worth re-running 350 s of GPU for a redundant field.

### The demo3 stage split reproduces

CHE-129 asked for the published `99.7 / 86.9 / 14.4 / 3.5 / 1.0` split to be
reproduced or the difference explained. It reproduces, on the `ramp_sum` route:

| stage | published | measured | share |
| -- | --: | --: | --: |
| `optiland_trace` | 99.7 s | **98.96 s** | 46.7% |
| `emit_patch_spectra` | 86.9 s | **86.78 s** | 40.9% |
| `reconstruct` | 14.4 s | **14.20 s** | 6.7% |
| `host_to_device` | 3.5 s | **3.48 s** | 1.6% |
| `power_bookkeeping` | 1.0 s | **0.83 s** | 0.4% |
| unaccounted | — | 7.71 s | 3.6% |

Every stage is within ~1% of the published figure except `power_bookkeeping`,
which is 0.83 s against 1.0 s — a 0.17 s difference on a 212 s run, at the
resolution where this probe's own host round-trips are the thing being timed.
**The trace and the emitter are 87.6% of the run**, which is the M5.1/M5.2
premise, now re-measured rather than inherited.

The 7.71 s unaccounted (3.6%) is real and named: the stage timers are inside the
chunk loop, and the denominator here is the **whole command** — interpreter
start, JAX compilation, the plan setup and the record write are outside it. It is
reported rather than distributed across the stages.

The same run on the k-space route:

| stage | `ramp_sum` | `kspace_splat` |
| -- | --: | --: |
| `optiland_trace` | 98.96 s | 99.36 s |
| `emit_patch_spectra` | 86.78 s | 84.29 s |
| `reconstruct` | 14.20 s | **7.62 s** |
| `host_to_device` | 3.48 s | 3.48 s |
| `power_bookkeeping` | 0.83 s | 0.88 s |
| **total** | **211.95 s** | **202.90 s** |

CHE-101's result, independently reproduced: the fast path halves the
reconstruction stage and moves the end-to-end run by 4.3%, because the stage is
6.7% of the cost. Everything outside the reconstruction is unchanged within
noise, which is what a reconstruction-only change should look like.

## Baselines recorded

| baseline | value | record |
| -- | --: | -- |
| default test suite | 211.4 s, 1179 passed / 48 skipped, peak child RSS 2.69 GB | `suite_default_cpu.json` |
| L2-PSF-01 bundle | 170.1 s, peak child RSS 12.48 GB | `l2_psf_01_cpu.json` |
| framework overhead | above | `framework_overhead.json` |
| ray-axis scaling | above | `scaling_ray_axis.json` |
| `estimate()` accuracy | above | `estimate_accuracy.json` |
| demo2 × 2, demo3 × 2 | above | see the demo table |

The CPU baselines are on `agent_solver`, 80 logical cores, no thread pinning
(`isolation.applied: false` — recording an affinity mask does not make a run
isolated). The GPU baselines were run on `agent_solver_gpu`, one RTX A6000 —
though as noted above the records say `agent_solver` for the image, and the
`+cu126` torch build is the field that actually attests to it. Swap growth zero
throughout every record.

The suite baseline counts 1179 rather than the 1182 the gate reports, because it
times a run in which its own record is absent, so three record-parameterized
tests do not collect. A self-timing gate cannot avoid this. Immaterial to a 211 s
timing, and stated rather than reconciled away.

## Two harness defects the demo baselines found

Neither could have been found without running them, which is the argument against
leaving a subcommand implemented-but-never-executed.

**The parent process was starving the child of the GPU.** `measure` synchronized
devices around the timed region. For a whole-command baseline that means the
*parent* initializes JAX before forking — and JAX preallocates ~78% of the card
on initialization, leaving the child ~12.5 GB of a 48 GB device. demo2's RW-P
route at the Table S2 budget completes in 94 s standalone and died with
`RESOURCE_EXHAUSTED` after 16 s under the harness. Fixed with
`measure(touch_devices=False)` for subprocess measurements: a process boundary is
a stronger barrier than any device sync, since the child cannot exit with work
outstanding. Pinned by
`test_a_subprocess_measurement_does_not_initialize_a_device_here`.

**`peak_host_rss_bytes` on a whole-command record described nothing.** It is the
parent's RSS. It read 0.32 GB while the parent imported torch and 0.03 GB once it
stopped — neither of which is the workload's memory. Whole-command records now
carry `subprocess.peak_child_rss_bytes` from `ru_maxrss` over waited-for
children, which is where the 12.48 GB for L2-PSF-01 above comes from. `cuda` is
`null` on these records for the same reason: a parent's allocator snapshot is not
the child's peak, and reporting its zeros would read as "the workload used no GPU
memory".

## What CHE-129 changed in the harness

An independent review of `6846ec3` (which CHE-105 shipped without) found four
must-fix items. All four are fixed and tested:

* **The report disagreed with its own records** on every headline number,
  including the sign of the coupler overhead. This document is rewritten from the
  JSON.
* **`swap_growth_bytes` fabricated a zero** when the cgroup file was unreadable,
  making the swap guard unfalsifiable on any host lacking that path. It is `None`
  now, as the schema always said it should be.
* **`compare()` divided the shipping call by the diagnostic-disabled call** — an
  11× "speedup" between two different computations, obtainable from committed
  artifacts with no refusal, because the two arms differed only in
  `workload.detail`. The diagnostic state is now part of `workload.route`, and
  `compare` refuses on differing `detail` as well.
* **`stages.fraction_of_total` mixed a per-repeat numerator with a cross-repeat
  denominator**, publishing a 101.57% share and admitting several hundred percent
  with one slow final repeat. The denominator is now the elapsed time of the same
  call, carried in the record as `total_s`, and a share above 1 raises rather
  than being written.

Also from that review: the memory watchdog now enforces all three of its guards
rather than only swap, and polls a running child rather than checking after it
has finished; `p95_s` carries a machine-readable `tail_rule` (`max_of_3`, not a
percentile) and the schema's off-by-one "repeats <= 20" is corrected to 19;
`fit_scaling` reports `r_squared: null` rather than 1.0 when there is no variance
to explain; the private-global mutation is one tested context manager instead of
two hand-rolled `try`/`finally` blocks; and the two failure paths the commit
message called load-bearing — the nonzero-exit refusal and the scan-limit restore
— now have tests, as does the live-code half of the 91% claim, which previously
rested entirely on a frozen artifact.

## Record provenance

`REGISTER.yaml` deferred the whole `ray_wave/*.json` corpus and named CHE-105 as
the ticket that would enroll it. CHE-105 did not, so CHE-129 did what it could
and states the rest.

`_demo_support.write_record` now stamps every record it writes, so a demo cannot
produce an unstamped record. The four configurations above are enrolled and
code-verified. The deferral is narrowed from `ray_wave/*.json` to
`ray_wave/demo2_*.json` and `ray_wave/demo3_*.json`, which is the 60 CHE-96 and
CHE-101 records nothing has re-run — so a *new* unstamped ray_wave record now
fails the gate instead of being covered silently.

One measured fact bounds what enrolling the rest can ever mean.
`provenance.environment_fingerprint` includes the torch build, which is
`2.13.0+cu126` in `agent_solver_gpu` and `2.13.0+cpu` in `agent_solver`. A
GPU-produced record's **environment** half therefore cannot reproduce under the
default CPU gate, however much compute is spent on it; its **code** half is
image-independent and does reproduce, verified on all four. So enrollment for
this corpus is code-only, which is also the half CHE-100's defect lived in — a
source module moving under a committed record, not a package bump. The
environment half needs the same environment-aware verification mode the
`optiland/*.json` and `chromatix/*.json` entries already ask for, from the other
direction.

`test_every_stamped_record_still_describes_this_tree_code` closes a related gap:
until now `ENROLLED_PROBES` was a hand-maintained list, so a record could carry a
provenance block, count as enrolled, and be verified by nothing.

## Not done

* **The 60 unstamped `ray_wave` demo records** are still deferred, now against
  M5 (CHE-118 / CHE-119 / CHE-120), which re-measures demo2 and demo3 and can
  stamp them at the cost of the compute alone.
* **An environment-only (or code-only) verification mode** in
  `core.provenance`, without which no GPU-produced record can ever be fully
  enrolled. Named in `REGISTER.yaml` in three places now.
* **demo3 `rw_f` and the enumerated shards** were not timed. CHE-129 asked for
  demo3 on both *reconstruction* routes at the 60 M-ray configuration, which is
  the `ramp_sum`/`kspace_splat` pair above; the `rw_f` emitter route at that
  budget is a different and much larger workload.
* **`benchmarks/probes/records/ray_wave/demo3_enumerated_positions.npz`** is
  untracked, unreferenced by anything in the tree, and produced by nothing that
  still exists — so it is not regenerable and was left in place rather than
  deleted. It predates this work. Somebody who knows what wrote it should decide.
* **`run.sh` does not export `MOA_IMAGE`**, so every record in this repository
  claims `container_image: "agent_solver"` regardless of which image ran it. The
  field is redundant with `packages.torch` for the GPU/CPU distinction, which is
  why it did not corrupt any conclusion here, but it is a false provenance field
  on committed evidence and it makes one component of the environment fingerprint
  dead. Worth a one-line fix in `run.sh` on a ticket that is already regenerating
  records, since fixing it invalidates every existing fingerprint by design.
* **`docs/architecture/overnight_run_2026_08_22.md`'s `b073a461…` claim**, noted
  above, is not this report's to correct.
* **The `mem_available` guard is reported, not enforced.** `measure` raises on
  cgroup swap growth and on the process RSS budget, but a host-wide
  `MemAvailable` breach is recorded in the notes instead of stopping the run —
  otherwise another tenant of this shared box could fail the ~20 `measure()`-based
  unit tests in the default gate while they time trivial closures, and the breach
  can predate the measured workload's first allocation. That is a deliberate
  trade and it means "swap watched throughout" is enforced while "host memory
  watched throughout" is only observed.
