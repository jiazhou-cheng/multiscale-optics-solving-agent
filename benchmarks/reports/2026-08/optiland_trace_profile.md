# CHE-118 (M5.1) — the Optiland trace was 48% of demo3, and 95% of that was hashing one number

demo3's largest stage was the ray trace: **98.96 s of a 211.95 s run**, 60 M rays
through four surfaces, on one RTX A6000. This issue was asked to profile it before
optimizing it, and to be prepared for the answer "this is what the traced geometry
costs, and the lever is elsewhere."

That is not the answer. **95.9% of the trace stage was Optiland building and
hashing a Python tuple of the wavelength array**, and the tuple had 200 000
entries because we handed it 200 000 copies of one number.

Fixing that made demo3 **1.761× faster end to end** — `211.95 s → 120.36 s` — with
the reconstructed field **bitwise identical** at 60 M rays and an identical energy
ledger.

Records: `benchmarks/perf/records/optiland_trace_decomposition.json`,
`optiland_trace_chunk_sweep.json`, `optiland_trace_precision.json`,
`optiland_trace_demo3_equivalence.json`, and the after-baseline
`demo3_characterization_rw_p_ramp_sum_che118_after_cuda.json`. Profiler:
`benchmarks/perf/profile_optiland_trace.py`.

## The decomposition

One real demo3 `characterization` chunk — 50 patch centres × 4000 secondary rays
= 200 000 rays, emitted by demo3's own patch emitter at demo3's seed, traced
through the four surfaces after the object surface.

| contribution | s per chunk | share | ours to fix |
| -- | -- | -- | -- |
| Optiland's refractive-index **cache key** | 0.3114 | 95.9% | yes — we built the input |
| Optiland's ray-geometry kernels | 0.0050 | 1.6% | no |
| repository-side handoff | 0.0082 | 2.5% | yes |
| **reconstructed total** | **0.3246** | | |

The reconstruction is checked, not asserted: the committed baseline's trace stage
is 98.956 s over 300 chunks = **0.32985 s per chunk**, against 0.32462 s
reconstructed here — a **ratio of 0.984**, inside the ±5% the issue asked for.
That agreement is what licenses reading the shares above as shares of the 98.96 s
rather than of a probe's own workload. The three contributions are differences
between measured arms, so they partition the total exactly and a zero remainder is
arithmetic, not evidence; the evidence is the closure ratio and the corroboration
below.

### The mechanism

`optiland/materials/base.py`:

```python
def _create_cache_key(self, wavelength, **kwargs) -> tuple:
    if be.is_array_like(wavelength):
        wavelength_key = tuple(np.ravel(be.to_numpy(wavelength)))
    ...
```

`BaseMaterial.n` and `.k` memoize on the *contents* of the wavelength array. Each
call copied 200 000 floats from the device to the host, built a 200 000-element
Python tuple, and hashed it. Measured at **18.8 ms for `n`** and **18.8 ms for
`k`**, per call.

`_trace_real` calls `material_pre.n` once and the refraction reads `n` on both
materials and `k` on one: **3 `n` and 1 `k` per surface, 16 lookups per chunk**
(counted from the call graph, and confirmed by cProfile on a single surface
trace). 4 surfaces × (3 × 18.84 + 18.76) ms = **0.3011 s** against **0.3114 s**
measured as the difference between the two wavelength arms — an independent
estimate at **0.967** of it. Two derivations from separate measurements agreeing
to 3% is what makes this an attribution rather than a story.

It also explains why the cost is **uniform per surface**. Per-surface, per-ray
wavelength: 76.4 / 77.8 / 78.1 / 76.7 ms. The spherical front surface's
intersection solve is not distinguishable from the three planes, because the
dominant term is not geometry.

### What the fix is

`RayBundle.wavelength_m` is a scalar `float` by contract, so the per-ray array was
always N copies of one number — and `trace_ray_batch` built it itself, one line
above where it was consumed:

```python
-    wavelengths = _solver_module(...).full_like(intensity_t, wavelength_um)
+    wavelengths = _solver_module(...).full_like(intensity_t[:1], wavelength_um)
```

A size-1 array takes Optiland's own documented scalar path — `IdealMaterial`
returns `index[0]` unchanged for `size <= 1`, and a dispersive material evaluates
its index once at that wavelength — and the scalar broadcasts against the N-ray
geometry. The lookup drops from 18.8 ms to **0.035 ms**, and the four-surface
solver trace from **316 ms to 5.0 ms**.

This is not a monkeypatch and not a fork: it changes what we hand a third-party
API, inside the adapter whose job is that handoff. The rule is named
(`MONOCHROMATIC_WAVELENGTH_RULE`), reported in the trace diagnostics and
provenance, and pinned by
`tests/test_coherent_bridge.py::TestMonochromaticWavelengthHandoff` — which
asserts *bitwise* equality against a reference trace built the old way, on the
glass configuration as well as the air one so the index weighting is exercised,
and separately shows on a dispersive material that the equivalence is
monochromatic rather than an artifact of a constant index.

## The rejected alternatives

The candidates CHE-118 listed, in its order of preference, and why each is not the
answer:

**Chunk size — rejected, by measurement.** On the arm this issue was asked to
explain, the cache-key cost is O(rays) host work done once per surface, not a
fixed per-chunk overhead: fitted exponent **0.981 at r² 0.9983** above 10 k rays,
where that term dominates the fixed floor, and cost per ray varying only **1.70×**
across a 400× range of chunk sizes from 10 k to 4 M. No chunk size amortizes it.
This was the cheapest candidate response and the measurement kills it.

*The post-fix answer is the opposite, and that is worth stating rather than
burying.* With the O(rays) work gone the trace is dominated by a **fixed ~13.2 ms
per call**, so cost per ray falls **~550×** across the sweep and demo3's 200 k
chunk sits **3.2× above** the optimum's cost per ray (1 M rays). It is not taken
here: the stage is now ~6 s of a 120 s run, so the whole prize is a few percent,
and `--rays-per-chunk` also sizes the patch emitter's spectra and the
reconstruction's accumulation. Choosing it belongs to M5.2, which owns the stage
that pays for it.

**Precision — not a lever in either direction.** demo3 already runs fp32, so
there is no fp32 speedup left to take. And fp64 is nearly free. On the solver arm
that survives the fix, fp32 and fp64 are **not separable at all**: ratio 0.99
against a noise floor of 0.004, measured as the spread between two identical fp64
runs rather than assumed. The whole `trace_ray_batch` call resolves at **1.02×**,
and even the pre-fix arm shows only **1.12×** — nowhere near this card's ~32×
fp64:fp32 arithmetic ratio, because the trace is bound by launches, memory
traffic and host work. The noise floor is measured and published for a reason: the
first realization of this section, at a fixed 5 repeats, reported fp64 running at
**0.86× the fp32 cost** — a sub-1.0 ratio for the more expensive precision, which
is not a physical result. Repeats are now chosen from a time budget and the
resolution test is one-sided.

What fp32 *does* cost is phase, against the fp64 trace of the same rays:
**1.1e-2 waves RMS** of optical path (~4.0°), max 4.5e-2 waves; sensor position
3.9e-4 µm RMS against a 4.2 µm pitch. The OPD is the one that matters, because
demo3 accumulates a coherent field and a fraction-of-a-wave error is a phase
error that does not average away. This is a **pre-existing property of every
demo3 record**, not something this change introduces, and it is out of scope
here — but note the useful inversion: before this fix, fp64 was a 12% surcharge on
the largest stage in the run; after it, a benchmark that needs a certified phase
can simply pay for it. Handed to the convergence work.

**Reducing the ray count — still M5.3's lever, and it is now the *only* one left
in the trace.** After the fix the trace is 5.4% of demo3. There is nothing more to
win here.

**Upstream engagement or a specialized path — not needed, and the evidence now
exists if anyone wants it.** CHE-118 called this the expensive option requiring
evidence first. The evidence is in `optiland_trace_decomposition.json`:
content-keyed memoization of a per-element function is pathological for any
large-array caller, and Optiland's own `IdealMaterial` has the scalar fast path
that makes it unnecessary. Worth reporting upstream as a follow-up; it blocks
nothing, because the fix is entirely on our side of the boundary.

**Monkeypatching `BaseMaterial._create_cache_key` — rejected.** It would mutate a
pinned third-party class process-globally, and it would change results for a
genuinely polychromatic trace. The size-1 handoff is correct *because* the batch
is monochromatic, and that condition is a contract we can state.

## Host synchronization inside the trace loop

CHE-101 found a data-dependent shape forcing a mid-pipeline host sync that cost
49 s. There is one host read in this loop and it is not that:
`int(valid.sum())` in `trace_ray_batch`'s diagnostics. Measured at **0.23 ms**
marginal per 200 k-ray chunk and bounded above by the whole repository-side
handoff (2.5% of the as-committed chunk). **Kept and justified**: it is the
clipped-ray count, which demo3's energy ledger consumes per chunk as
`invalid_rays` to separate an aperture loss from an empty draw. No shape in the
loop is data-dependent, so there is no synchronization of CHE-101's kind.

## The handoff bucket, now that it is the majority of the stage

Post-fix the repository-side handoff is 8.2 ms of a 13.2 ms trace call. Its
pieces, individually measured (not a partition — a piece timed alone can cost
differently in place; closure 0.86):

| piece | ms |
| -- | -- |
| `with_traced_state` artifact construction | 4.03 |
| `_require_launch_plane` | 1.29 |
| `|a|²` + inbound bridge | 0.56 |
| outbound bridge | 0.54 |
| clipped-ray mask | 0.30 |
| `int(valid.sum())` host read | 0.23 |
| `RealRays` construction | 0.06 |

Information for M3.1's cost model, not a target: the stage is ~6 s of demo3.

## Before and after

Same environment fingerprint (`fb66d7be…`), same seeds, same prescription
fingerprint, same 300 chunks of 200 000 rays, single GPU (device 6), sequential,
cgroup swap growth **0**, peak child RSS 2.82 GB.

| stage | before (s) | after (s) |
| -- | -- | -- |
| `emit_patch_spectra` | 86.78 | 86.88 |
| `host_to_device` | 3.48 | 3.66 |
| `optiland_trace` | **98.96** | **6.45** |
| `power_bookkeeping` | 0.83 | 0.79 |
| `reconstruct` | 14.20 | 14.43 |
| **total** | **211.95** | **120.36** |

**15.3× on the trace stage, 1.761× on the run.** Every other stage is unchanged
within ~1%, which is the check that the fix is local.

The result is unchanged, asserted rather than assumed: the reconstructed
420×420 complex64 field is **bitwise identical**, and every entry of the energy
ledger matches exactly (transmitted fraction, captured fraction, invalidated ray
count, both amplitude sums). `array_equal`, not `allclose` — a tolerance would
pass on a real change. Peak device bytes moved by 0.4 MB, which is JAX allocator
granularity; the size-1 array is *smaller* than the one it replaced.
`profile_optiland_trace.py verify-demo3` regenerates that artifact, and it carries
the two timing records so the speedup claim and the equivalence claim live in one
file.

## The per-ray cost model, and what it refuses to do

`solvers.optiland.cost_model` replaces the registry's
`O(number_of_surfaces × number_of_rays)` as the *usable* cost information. M0.4
scored the old estimator and recorded the consequence: *"NO PREDICTION … a planner
cannot use this estimator to order work by cost."*

```
seconds = 0.01320 + 2.715e-9 × rays × surfaces
```

Affine because that is what was measured: after the fix the trace is a fixed
per-call cost plus per-ray device work, and the two cross over inside the range
demo3 uses. A power law over the same points fits at exponent 0.19, r² 0.59 —
which is the evidence that a single exponent will not do, not a usable model.
The affine form holds to **1.9%** over **1e3–1e6 rays per call**.

`estimate_trace_seconds` returns **no prediction** rather than a number when the
environment fingerprint differs from the calibration's, and when the ray count is
outside the fitted domain (the measured residual at 4 M rays is **−44%**, so the
model would underpredict). `core.performance.compare` already refuses to divide
records across environment fingerprints; a cost model that happily extrapolated
across the same boundary would be the identical error with a friendlier interface.

`OptilandAdapter.estimate()` now uses it, and closes the two gaps its old note
disclaimed: it reads the surface count from the prescription, and it converts
`config["num_rays"]` — a *ring* count — into a traced ray count with
`1 + 3N(N+1)`, verified against all six ring counts in the committed
`scaling_ray_axis.json` (16→817 … 64→12481, exactly). It still says plainly that
the calibration covers ray propagation and not Optiland's own ray generation or
artifact writing, so the prediction is the trace's share of that call.

`peak_memory_bytes` stays `None` throughout. Memory was not measured across chunk
sizes and a planner would size a batch with a guess.

## The records this change made stale, and what regenerating them showed

`tests/test_provenance_fingerprint.py` (CHE-103) flagged six stamped records as no
longer describing the code that produced them. All six were regenerated rather
than exempted:

* **`m3_convergence`, `m3_first_null_grid_convergence`, `m3_off_axis_handoff`,
  `m3_psf_verification`** — stale on `src/solvers/optiland/adapter.py`, which this
  issue touched only in `estimate()`. These probes never call it, and the
  regeneration confirms that rather than assuming it: **no scientific value moved
  in any of the four.** Three are byte-identical outside the provenance stamp; the
  fourth (`m3_convergence`) differs only in per-stage seconds and peak RSS, which
  are resource fields the probe reports and not results.
* **`perf_demo3_characterization_rw_p_ramp_sum_cuda`** and
  **`..._kspace_splat_cuda`** — stale on `coherent_trace.py`, i.e. on the fix
  itself. Regenerated on the GPU. The ramp_sum record reproduces the after-run's
  field bitwise, and the kspace record gives that route's post-fix trace stage.

A consequence worth being explicit about: a *pre-fix* record cannot remain stamped
in the tree, because the gate requires stamped records to describe the current
code and the code is what changed. The pre-fix measurement is therefore preserved
by being embedded verbatim in `optiland_trace_demo3_equivalence.json`, together
with the commit (`6846ec3`) and code fingerprint it was produced under, so the
before arm is reproducible by checkout. The bitwise claim itself does not depend
on that file: `TestMonochromaticWavelengthHandoff` re-derives it from committed
code on every test run.

## Two harness defects this ticket found by being the first real before-and-after

Both in M0.4's harness, both only reachable by actually needing a before and an
after of one configuration:

1. **`_demo` overwrote its own baseline.** The record name is derived from the
   arguments, so re-running demo3 at the same preset destroyed the record it was
   being compared against — the exact failure `_device_suffix`'s docstring
   describes for demo2, which the device suffix cannot catch when the device is
   the same. Fixed with `--label-suffix`.
2. **`compare` refused every before-and-after.** Its `detail` check exists
   because `route` alone once let two different computations divide into an 11×
   "speedup", and it is the right rule — but it read `label_suffix` and
   `rays_read_from` as evidence of different work. Both describe the measurement,
   not the work. Named in `MEASUREMENT_PROVENANCE_KEYS`, excluded from the check,
   and **reported in the result** as `detail_differences_ignored` so a reader can
   see what the comparison looked past.

## What this changes for M5

The premise M5 was written on has moved. demo3 is now:

| stage | share of 120.36 s |
| -- | -- |
| `emit_patch_spectra` | **72.2%** |
| `reconstruct` | 12.0% |
| `optiland_trace` | 5.4% |
| `host_to_device` | 3.0% |
| `power_bookkeeping` | 0.7% |

**M5.2 (CHE-119) is now the whole ticket.** The emitter was 41% of the run and is
72% of it, and it is unchanged in absolute terms — the fix removed a competitor,
not a dependency. **M5.3 (CHE-120)** keeps its lever: the ray budget still
multiplies the emitter. And the trace is no longer worth optimizing at any of the
levers examined here.

## Scope and limits

* One prescription (demo3's plano-convex singlet plus the DOE plane), one card,
  one container image. The mechanism is prescription-independent — any monochromatic
  trace of more than a few thousand rays through Optiland was paying it — but the
  constants are not.
* The `M_RAY_OPTILAND` standalone pupil path was **not** re-measured. It traces
  through the same materials and almost certainly paid a smaller version of the
  same cost (it generates its own wavelength array, and its ray counts are
  thousands rather than hundreds of thousands), but that is inference and is
  recorded as such rather than claimed. A follow-up, and cheap.
* demo3's `kspace_splat` route **was** re-run, because CHE-103's staleness gate
  required it: its trace stage went **99.36 s → 5.34 s** (18.6×), independently
  reproducing the fix on the other reconstruction route. Only the stage timer, not
  a full harness baseline, so there is no `compare`-able before/after for that
  route.
* No convergence claim. Nothing about the ray budget, the estimator's variance, or
  whether 60 M rays is the right number is touched — that is M5.3.
