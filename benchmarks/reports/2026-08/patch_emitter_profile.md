# CHE-119 (M5.2) — the emitter was 72% of demo3, and it was one core doing parallel work

After M5.1 removed the Optiland trace as a competitor, the patch-spectrum emitter
was **86.88 s of a 120.36 s run — 72%**. This issue was framed as "move it off the
host", with a device port as option 3 and its full reproducibility bill attached:
a JAX emitter would resample the spectral modes, and *"a device emitter that
changes the sampled modes changes every committed number."*

It did not come to that. The stage is **58% padded FFT and 22% spectral draw**,
both *independent per patch*, and numpy releases the GIL in both. Eight host
threads plus removing two pieces of measurable waste took the stage **2.5× faster
with every emitted ray bitwise unchanged** — so no capability widened, no
estimator changed, and not one committed number needed re-measuring.

demo3: **120.36 s → 68.34 s, 1.761×**. Reconstructed field bitwise identical at
60 M rays; energy ledger identical.

Records: `benchmarks/perf/records/patch_emitter_decomposition.json`,
`patch_emitter_thread_sweep.json`, `patch_emitter_cost_model.json`,
`patch_emitter_overlap.json`, `patch_emitter_demo3_equivalence.json`. Profiler:
`benchmarks/perf/profile_patch_emitter.py`.

## The decomposition

One real demo3 `characterization` chunk — 50 patch centres × 4000 secondary modes,
from demo3's own patch plan at demo3's own seed, pad 301.

| contribution | ms/chunk | share | ×300 chunks | ours? |
| -- | --: | --: | --: | -- |
| padded patch FFT | 159.6 | 56.6% | 47.9 s | parallelizable |
| spectral draw (`Generator.choice(p=…)`) | 62.4 | 22.1% | 18.7 s | parallelizable |
| `RayBundle` construction + validation | 13.4 | 4.7% | 4.0 s | |
| density (abs, sum, divide) | 11.9 | 4.2% | 3.6 s | |
| **`dir_x[propagating]`, `dir_y[propagating]`** | 11.3 | 4.0% | 3.4 s | **pure waste** |
| ray assembly | 8.8 | 3.1% | 2.6 s | |
| **`spectrum[propagating]`** | 8.3 | 2.9% | 2.5 s | **near-pure waste** |
| extract + zero-pad | 5.2 | 1.8% | 1.6 s | |
| spectral grid setup (once per call) | 1.2 | 0.4% | 0.3 s | |
| **accounted** | **282.1** | | **84.6 s** | |

The committed stage is 86.88 s over 300 chunks = **0.28960 s/chunk**, against
0.28206 s reconstructed — a **ratio of 0.974**, inside the ±5% the issue asked
for. The instrumented replica reads slightly under the shipped call because no
region covers the function's own call and argument handling.

**The replica is validated before it is trusted.** An in-place decomposition has
to instrument a copy of the loop, and a copy drifts. So the profiler asserts, on
every run, that its replica reproduces the shipped emitter's bundle *bitwise*; if
that fails it writes no record. That single assertion does two jobs: it stops the
profiler describing code it no longer matches, and it independently re-checks the
claim that this issue changed no ray.

That check exists because the obvious alternative does not work. Timing each
piece standalone, on pre-materialized inputs, summed to **137% of the call** — a
spectrum the FFT has just written is hot in cache and one read out of a list of
fifty is not. The same trap M5.1 hit in its handoff sub-split, here severe enough
to make the numbers useless.

## Two pieces of waste, and why they were invisible

`propagating` is the mask of modes inside the unit circle. On demo3's grid the
Nyquist direction cosine is λ/(2·pitch) = 0.056, so **every one of the 90601 modes
propagates and the mask is entirely true** — which makes `a[propagating]` a
full-array boolean gather returning exactly `np.ravel(a)`, for 157 µs against
0.2 µs. And `dir_x[propagating]`/`dir_y[propagating]` are functions of the padded
grid and the wavelength alone; they were recomputed for all fifty patches.

Both are removed. The boolean path is kept for grids where the mask means
something, and `test_a_grid_with_evanescent_modes_still_takes_the_boolean_path`
exists so the fast path cannot quietly become the only path.

At the whole-call level the two hoists are roughly cancelled by the block buffer
the threading needs: shipped single-threaded is 274.5 ms against the replica's
282.1 ms. Worth saying plainly rather than adding 5.9 s to the ledger — **the win
here is the threading.**

## Why threading is an optimization and not a second estimator

The parallel path calls `np.fft.fft2` on the same input the serial path did. There
is no reformulation to validate, only a scheduling change — so the result is
bitwise identical *by construction*, and the tests hold it there:

| case | what it covers |
| -- | -- |
| `demo3_chunk_50x4000`, `50x20000` | the shipping configuration |
| `single_patch` | one patch, below the pool gate |
| `multi_block_200x500` | more patches than `PATCH_BATCH_PATCHES`, so the block loop runs twice |
| `rim_patches_20x1000` | patches partly outside the DOE, so padded rows are entirely zero |
| `evanescent_grid_6x300` | a pitch below λ/2, so the boolean-mask path runs |
| `evanescent_enumerated`, `demo3_enumerated_2` | `secondary_count=None`, no RNG |

All eight bitwise identical to the pre-change code, at 1, 2, 8 and 16 threads.

The one place arithmetic was rewritten rather than rescheduled is the draw.
`Generator.choice(p=…)` normalizes a cumulative sum and searches it with `size`
uniforms; the emitter now writes that out. Two reasons, and the second is the real
one: it skips numpy's validation passes over a 90601-element vector
(1203 µs → 696 µs), and it **separates the RNG-ordered part from the
parallelizable part**. The uniforms must come off the generator in patch order;
the cumulative sum and the search are pure functions of one patch. So the block's
uniforms are drawn in a single `rng.random((block, S))` — which consumes exactly
the stream that `block` sequential per-patch draws would — and the search runs in
the pool. 61 ms → 19 ms, picks bitwise identical.

`test_the_inlined_draw_is_bitwise_numpys_choice` asserts the picks *and the
resulting stream position* against `rng.choice` at three sizes. Getting the picks
right and the stream position wrong would corrupt every patch after the first.
It is also the tripwire for a numpy upgrade: if numpy changes how `choice`
consumes the stream, that fails loudly instead of silently changing every
committed demo2 and demo3 field.

Because the draw is blocked, `PATCH_BATCH_PATCHES` could in principle change the
answer. `test_the_block_size_does_not_change_a_single_emitted_ray` covers block
sizes that divide the patch count and ones that do not.

## The two constants, both measured

**Eight threads.** The speedup plateaus at **2.88×** on the 50-patch chunk, and
the thread counts indistinguishable from the best are **[8, 16]** — so the argmax
is not a measurement, the plateau is; across realizations the winner flips between
the two. Eight is the smaller footprint, which is the tie-break AGENTS.md asks for
on a shared box, and it is a deliberately small fraction of its 80 cores. Past the
plateau it gets worse: this workload is memory-bandwidth-bound before it runs out
of cores.

**A pool only above eight patches.** Threading a small call is a *loss*, and by a
lot. Forcing the pool on:

| patches | serial | forced pool | ratio | the gate saves |
| --: | --: | --: | --: | --: |
| 1 | 8.6 ms | 22.7 ms | 2.64× worse | 14.1 ms |
| 2 | 13.0 ms | 23.8 ms | 1.83× worse | 10.8 ms |
| 4 | 22.5 ms | 29.4 ms | 1.31× worse | 6.9 ms |
| 8 | 43.3 ms | 30.9 ms | 0.71× | — |
| 50 | 277.9 ms | 100.9 ms | 0.36× | — |

The measured crossover is **8**, which is `POOL_MIN_PATCHES`. The penalty is
roughly fixed per call and does not shrink with the work, which points at numpy's
per-thread FFT plan cache: pad 301 has a prime factor of 43, so a fresh worker
thread builds its Bluestein tables from scratch. Amortized over fifty patches it
is invisible; on one patch it is the whole call.

Without the gate, every exactness test and every single-patch caller would run
~2.6× slower than before — a regression paid for by a stage they do not use. Note
the sweep had to *force the pool on* to measure this at all: below the gate the
shipped code is the serial path, so comparing them there measures nothing. The
first realization of that section reported a crossover of 1 patch for exactly that
reason.

## The rejected alternatives

**Option 3, a JAX/device emitter — rejected, and the profile is why.** The stage
was never FFT-flop-bound in a way that needed a device. It was parallel work on
one core, plus waste. Threading captured 2.5× at zero reproducibility cost;
a device port would have resampled the modes, made every committed demo2 and demo3
number a re-measurement, and required its own capability entry, its own exactness
gate at whatever precision it computed in, and a measured agreement against the
FP64 host emitter. That bill is not worth paying for a stage that is now 35 s.

**Option 2, pipelining the emitter against the trace — deferred, and the honest
framing is that the prize *grew*.** Before this issue, 87.7 s of the 120.4 s run
is host work and 20.9 s is device work, strictly serialized in the chunk loop, so
perfect double-buffering could hide at most **20.9 s — 17% of the run**. After
this issue the emitter falls to ~37 s, the two sides move closer together, and
since a pipeline hides the *smaller* side the ceiling is still 20.9 s — now
**~30%** of a ~70 s run. So this is not "too small to bother with". It is out of
scope for three reasons that are about ownership rather than size: it is a
structural change to a benchmark probe's chunk loop when the thing that should own
an execution schedule is M3.1's executor; it doubles the in-flight memory and so
needs its own envelope measurement rather than an argument; and it would put host
threads and a device queue live at once, which AGENTS.md's rule on concurrent GPU
work asks to be justified with a measurement. Quantified and handed on, not waved
away.

**Option 4, accept and document — overtaken.** It was the right outcome only if
the stage were flop-bound in float64. It is not.

## Before and after

Same environment fingerprint (`fb66d7be…`), same seeds, same prescription
fingerprint, same 300 chunks, single GPU (device 6), sequential, cgroup swap
growth **0**.

| stage | before (s) | after (s) |
| -- | --: | --: |
| `emit_patch_spectra` | **86.88** | **34.70** |
| `reconstruct` | 14.43 | 14.51 |
| `optiland_trace` | 6.45 | 5.94 |
| `host_to_device` | 3.66 | 3.90 |
| `power_bookkeeping` | 0.79 | 1.28 |
| **total** | **120.36** | **68.34** |

**2.50× on the stage, 1.761× on the run.** Peak child RSS 2.82 → 2.94 GB: the
block buffer is 50 padded patches at pad 301, 72 MB, and that is where the
120 MB went.

The result is unchanged, asserted rather than assumed: the reconstructed
420×420 complex64 field is **bitwise identical** and every entry of the energy
ledger matches exactly. `array_equal`, not `allclose`.

Two independent confirmations, both from records CHE-103's staleness gate required
re-running anyway:

* **demo3 `kspace_splat`** — 58.74 s total, emitter 34.14 s. The same emitter fix
  on the other reconstruction route.
* **demo2 `paper` RW-P at 1.6e8 rays** — **95.66 s → 65.5 s, 1.46×**. A different
  DOE, a different patch size, a different pad, and 2.7× the ray budget: the
  emitter fix is not specific to demo3's configuration.
* **demo2 `paper` RW-F** — 6.37 → 6.54 s, i.e. unchanged within noise. That route
  is the single full-aperture patch, which sits below the pool gate: the gate
  visibly doing its job on a shipping configuration.

## The cost model

`couplers.patch_cost` gives `PatchWftCoupler.estimate()` a wall time where it
previously reported none:

```
seconds = 0.02393 + 0.000881 × patches + 1.890e-7 × patches × secondary
```

Two terms because the stage has two, read off the decomposition rather than
chosen to fit: the transform, the density and the cumulative sum are O(pad²) per
patch and do not care how many modes are drawn from the result, while the search
and the ray assembly are O(S) per patch. Fitted on both axes together, because
fitting either alone attributes the whole stage to whichever moved. Holds to
**7.3%** for 16–50 patches at pad 301.

It refuses rather than extrapolating in three cases, each with evidence:

* **off its environment fingerprint** — the same rule `core.performance.compare`
  applies to a ratio;
* **at a different pad** — `pad_px` is *held* at 301, not fitted, so the constant
  has absorbed the O(pad² log pad); a pad sweep would remove this and has not been
  run;
* **below 16 patches** — the pool gate makes that region two regimes rather than
  one, measured at **31.6%** error against 7.3% inside.

`tests/test_patch_wft.py` pins every constant to the committed record, so
re-running the sweep and forgetting to update the source fails a test instead of
mispricing a plan.

**A pre-existing bug in the same estimator, found and fixed.** It computed
`pad = patch_px * pad_factor`, but that product is a *floor*: `resolve_pad_px`
raises it until clearance, centring and oddness all hold. demo3 asks for 101×2 =
202 and runs at **301**, so the estimate described a transform 2.2× smaller than
the one that executes — and under enumeration a mode count 2.2× too small with it.
It now resolves the real pad, and reports both numbers so a reader can see the
floor move.

## What the capability declaration does and does not say

`C_PATCH_WFT` is **unchanged**: still `devices: [cpu]`, `namespaces: [numpy]`,
`native_compute_dtypes: [complex128]`, `minimum_compute_precision: FP64`. No CUDA
or JAX path executed, so none is declared. The FP64 reason still stands and is
still substantive: the exactness relation that makes this operator trustworthy is
measured at 1.4e-12, below float32 epsilon.

What *was* corrected is the **cost argument** in the notes, which had been
overtaken twice and had come to say the opposite of the truth. It read: the
expensive half of a patch run is the O(rays × pixels) reconstruction, and the
patch transform "is not where the time goes". CHE-101 made the reconstruction
9.6× faster (7% of demo3) and this issue measured the transform and its draw at
42%. The note now records that history, and records why the response was threading
rather than a device — which is precisely why the declaration still does not
widen.

## The records this change made stale

CHE-103's gate flagged eight. All were regenerated rather than exempted:

* **The four `m3_*` records** — stale on `capabilities.py`, `patch.py` and
  `patch_node.py`. Regenerating confirms rather than assumes that nothing moved:
  no scientific value changed in any of them; three are byte-identical outside the
  provenance stamp and the fourth differs only in per-stage seconds and peak RSS.
* **demo2 RW-F, demo2 RW-P, demo3 `kspace_splat`, demo3 `ramp_sum`** — re-run on
  the GPU, and their post-change numbers are the confirmations above.
* **`perf_demo3_…_ramp_sum_che118_after_cuda`** — this issue's *before* arm, which
  cannot stay stamped in a tree whose code changed. Removed; its measurement is
  embedded verbatim in `patch_emitter_demo3_equivalence.json` with the commit it
  came from, and the equivalence claim is re-derived from committed code by the
  unit suite on every run rather than depending on that file.

The field evidence is chained rather than assumed across the final re-run: the
demo3 record had to be regenerated once more after the `capabilities.py` and
`patch_node.py` edits, and its field is bitwise identical to the first
post-CHE-119 run — which was in turn bitwise identical to the pre-CHE-119 one.

## A harness gap this found

`run.sh` did not forward `MOA_PATCH_THREADS` into the container. Four runs were
launched with it set on the host and all four silently used the default — the
records honestly reported the default, and nothing errored. Now forwarded, along
with `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`, and only
when actually set. `MOA_PATCH_THREADS` is registered in
`core.performance._THREAD_VARS`, so an override lands in the performance
fingerprint: it changes the emitter's wall clock by 2.5× and the emitted rays not
at all, which is that fingerprint's definition.

A consequence worth knowing: **setting `MOA_PATCH_THREADS=8` makes a run
incomparable to one that leaves it unset**, even though 8 is the default and the
two runs are identical. The first after-measurement here was taken that way and
`compare` refused it — correctly. Re-run without the override.

## What this changes for M5

demo3 is now:

| stage | share of 68.34 s |
| -- | --: |
| `emit_patch_spectra` | **50.8%** |
| `reconstruct` | 21.2% |
| `optiland_trace` | 8.7% |
| `host_to_device` | 5.7% |
| `power_bookkeeping` | 1.9% |

The emitter is still the largest stage, and M5.1's plus M5.2's fixes together have
taken demo3 from **211.95 s to 68.34 s, 3.10×**, with the reconstructed field
bitwise unchanged throughout.

**M5.3 (CHE-120) inherits a sharper question.** The emitter is now 0.881 ms per
patch plus 189 ns per secondary ray, so the ray budget's cost is measured rather
than asserted, and the estimator-variance question — is 2.6e9 rays required? — is
now the only large lever left. Halving the ray count now halves half the run.

## Scope and limits

* **One pad.** Every calibration point shares pad 301. The *mechanism* is
  pad-independent — any patch count above the gate parallelizes — but the cost
  constants are not, and the model refuses off that pad rather than extrapolating.
* **No device path, and therefore no new capability entry.** That is the decision,
  not an omission.
* **Pipelining not implemented**, ceiling quantified at 20.9 s (~30% of the
  post-change run). Deferred to whatever owns the execution schedule.
* **Memory is not calibrated.** `estimate()` still reports `peak_memory_bytes`
  from the analytic `16 × (pad² + 4 × emitted)` expression, not from a
  measurement, and the cost model contributes nothing to it.
* **The draw is now this repository's copy of numpy's sampler.** That is a real
  transfer of ownership. It is guarded by a test that compares against
  `rng.choice` on every run, which makes a numpy change a loud failure rather than
  a silent renumbering — arguably a better posture than before, but it is a
  change of posture and not a free win.
* **No convergence claim.** Nothing here touches whether the ray budget or the
  patch count is sufficient; that is M5.3 and M2.3.
