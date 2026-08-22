# CHE-70 — GPU coherent `wave_to_ray → Optiland → ray_to_wave` bridge on a 100×100 metalens

**Status: PASS.** Executed 2026-08-20 on the target host, one candidate at a time,
on one RTX A6000.

This is the tracked report. `outputs/` is gitignored by design, so the numbers
that must survive are here; the regenerable artifacts (48 candidate JSONs, two
CSV tables, seven figures, three `.npy` fields) live under
`outputs/che70_metalens/` and are reproduced by the command in §1.

---

## 1. Reproduction

```bash
MOA_GPUS=device=0 ./run.sh --gpu python -m benchmarks.metalens_controller \
    --grid-size 100 --device cuda --precision fp32 --sampling-density magnitude \
    --seed 1 --auto-converge --memory-guard --config METALENS-AIR-100 \
    --validate METALENS-SLAB-100 --output outputs/che70_metalens
```

48 candidates, 1083 s wall, every one `PASS_RUN`. The analysis alone is
re-derivable from the persisted candidates without re-running anything:

```bash
./run.sh python -m benchmarks.metalens_controller \
    --output outputs/che70_metalens --config METALENS-AIR-100 \
    --validate METALENS-SLAB-100 --device cuda --reanalyze
```

Two practical notes. The controller must be run in the foreground of a shell that
outlives it; a `./run.sh` killed by an outer timeout leaves its container running
the job (CHE-66), so check `docker ps` before retrying rather than launching a
second one. And `MOA_GPUS` selects the device — Optiland 0.6.0's `set_device`
takes `"cpu"` or `"cuda"` and has **no device ordinal at all**, so which physical
GPU is used is decided by the container's visible set and by nothing inside the
process.

---

## 2. Final acceptance statement

> The 100×100 metalens direct-wave PSF and the CUDA `wave_to_ray → Optiland →
> ray_to_wave` PSF agree at **NCC = 0.998904840 ≥ 0.99** for the numerically
> demonstrated smallest converged pair **(P = 4, S = 1024), P·S = 4096**. Spatial
> and angular convergence were each independently varied and each converge.
> Complex amplitude and optical phase are preserved through the Optiland handoff:
> the amplitude returns bit-identical, and under full bin enumeration the whole
> route reproduces the analytic field to **8.9 × 10⁻¹⁴** relative error. The
> benchmark executes using bounded-memory chunked GPU processing — peak GPU memory
> fits `chunk^0.997` at fixed `P·S` and `(P·S)^(-3e-16)` at fixed chunk — never
> materializes an unsafe `P×S` population, does not silently fall back to the CPU,
> and causes zero additional swap usage.

> **Additional swap used by the benchmark: 0 bytes.**

Measured as the peak of `/sys/fs/cgroup/memory.swap.current` minus its
pre-candidate value, on all 48 candidates. That is the *container's* swap. Host
swap is non-zero at rest on this machine (CHE-64), so a host-swap delta would not
have been attributable to this run; the host figures were recorded alongside every
candidate and did not move (485 MB at start, 444 MB at the end — it fell).

Two qualifications belong in the same breath as the pass, and §6 and §7 are about
them: the declared NCC gate is met at a point where the **total power is still
24.7 % high**, and in this configuration `P` and `S` are provably *not*
independent axes.

---

## 3. Headline numbers

| quantity | value |
|---|---|
| smallest converged `P` | 4 |
| smallest converged `S` | 1024 |
| smallest converged `P·S` | 4096 |
| NCC vs the analytic oracle | 0.998904840 |
| normalized MSE | 4.696 × 10⁻³ |
| relative power error | **+0.24690** |
| relative peak error | −0.05891 |
| centroid error | 6.25 × 10⁻⁸ m (0.25 pixel) |
| FWHM error | +2.52 % |
| EE50 radius error | +11.8 % |
| peak GPU allocated (whole sweep) | 463 482 880 B (442 MiB) |
| peak process RSS (whole sweep) | 2 514 051 072 B (2.34 GiB) |
| minimum host `MemAvailable` | 393 875 927 040 B (367 GiB) |
| **swap delta, all 48 candidates** | **0 B** |
| controller RSS drift over 48 candidates | +1 310 720 B (+1.25 MiB) |

Local stability (Phase 27), all three doublings within the declared 10⁻³:

| doubling | NCC vs the candidate | 1 − NCC | within 10⁻³ |
|---|---|---|---|
| (2P, S) | 0.999275598 | 7.244 × 10⁻⁴ | yes |
| (P, 2S) | 0.999275590 | 7.244 × 10⁻⁴ | yes |
| (2P, 2S) | 0.999013846 | 9.862 × 10⁻⁴ | yes |

Monte Carlo variability at the pair, three independent seeds (Phase 28): NCC mean
0.998689686, std 2.91 × 10⁻⁴, worst 0.998358113 — every seed clears the gate.

---

## 4. Configuration

| | `METALENS-AIR-100` (primary) | `METALENS-SLAB-100` (validation) |
|---|---|---|
| grid | 100 × 100 | 100 × 100 |
| pitch | 250 nm (λ/2) | 250 nm |
| wavelength | 500 nm | 500 nm |
| aperture | 20 µm circular (80 % of the window) | same |
| design focal length | 50 µm | 50 µm |
| stack | 50 µm air | 10 µm n = 1.5 plate, then 43.3 µm air |
| sensor plane | z = 50 µm | z = 53.33 µm (plate-shifted focus) |
| NA | 0.1961 | 0.1961 |
| Airy radius | 1.555 µm = 6.22 pixels | same |
| propagating modes | 10 000 total, **7 825 retained** | same |
| evanescent power discarded | 1.889 × 10⁻³ | same |

The metalens is an ideal phase-only element: unit amplitude inside the aperture,
zero outside, carrying `-k(√(r² + f²) − f)`. Both routes start from the **same**
`ComplexField` — `evaluation.metalens.metalens_field` is the only producer, so the
reference and the ray route cannot be tuned independently.

`METALENS-SLAB-100` exists because a pure air gap does not exercise the OPL
contract: with a plate, Optiland must refract at two interfaces and accumulate an
*index-weighted* path rather than a geometric distance. It reaches NCC 0.998832 at
`P·S = 4096` and 0.999464 at 8192 — the same convergence as the air case, with the
refraction doing real work.

### Why the sensor sits where it does

The plate shifts the focus back by `t(1 − 1/n)`, and the sensor is put there
rather than the metalens phase being retuned. So the same phase profile serves
both configurations, the plate's spherical aberration is present, both routes see
it, and neither sees a system built to flatter it.

---

## 5. The oracle, and why it can carry a gate

`evaluation.metalens.reference_field` evaluates the **exact** plane-wave transfer
function of the layer stack in float64:

```
per layer:  exp( i 2π t √( (n/λ)² − f² ) )
```

A plane-parallel homogeneous stack is diagonal in the plane-wave basis and the
transverse wavevector is continuous across every interface, so for a field sampled
on a periodic grid the finite sum over the grid's own modes **is** the exact
solution — not a discretization of one. That is what makes it analytic in the
strict sense this project requires, rather than a repository-local numerical
solver that would be circular as a gate.

It is cross-checked against two things written for other reasons:

| route | agreement (air configuration) |
|---|---|
| `evaluation.asm_oracle.angular_spectrum_float64` (CHE-40), *un-centred* FFT convention | 6.1 × 10⁻¹⁴ piston-aligned relative field error |
| Chromatix `asm_propagate`, third-party, M1-verified, complex64 | < 5 × 10⁻⁴ (its own dtype floor); PSF 1 − NCC < 10⁻⁶ |

Both live in `tests/test_metalens_oracle.py`. The Chromatix leg is corroboration,
not a gate: Chromatix is complex64-only, so it cannot be tighter than its own
representation.

For the slab configuration no prior repository reference exists, so the one
assumption the layered form adds — ideal transmission at both interfaces, which is
what the pinned solver does with no coatings — is **tested against the traced
intensity** rather than trusted
(`test_the_pinned_solver_applies_no_fresnel_amplitude_loss`), together with
`n sin θ` conservation at the interface.

### The gates are declared, and here is what they sit above

`NCC ≥ 0.99` and `1 − NCC < 10⁻³` are **declared engineering targets**, authorised
as such, not derived from a noise model. Recording the floors they sit above is
what stops them being arbitrary:

* the estimator's own exactness limit against the same oracle, float64, full
  enumeration: **8.9 × 10⁻¹⁴** relative field error;
* float32-on-GPU against float64-on-host at the same pair: NCC agrees to
  **< 10⁻⁶**, so the GPU answer is set by its sampling and not by its arithmetic;
* chunked against single-chunk at the reported pair, on the device: 1 − NCC
  ≈ 3 × 10⁻¹⁴.

---

## 6. Finding 1 — near-grazing modes destroy the phase, and the band limit is a correctness requirement

This is the finding that changed the design, and it was measured rather than
anticipated.

`C_RAY_TO_WAVE` forms each ray's constant phase as `k(OPL − d·x₀)`. For a mode
with axial direction cosine `dₙ`, propagating `Z` makes both terms scale as
`Z/dₙ` while their difference is `Z·dₙ`. Near grazing they nearly cancel, so the
*relative* precision of the inputs sets the *absolute* error of the phase:

```
Δφ  ~  ε · k · Z / dₙ
```

On this exact grid, **eight bins land on `d_u² + d_v² = 1`** — the (30, 40) and
(40, 30) Pythagorean triples and their sign variants — and survive the strict
`radial < 1` evanescent cut at `dₙ = 1.05 × 10⁻⁸`. Over a 50 µm propagation their
OPL is **4745 m**. They carry 2.25 × 10⁻⁷ of the field's power and are drawn at
full importance weight.

Measured, float64, full enumeration against the analytic oracle:

| band limit | exactness-limit relative field error |
|---|---|
| none | 2.8 × 10⁻⁹ |
| `dₙ ≥ 10⁻²` | **8.9 × 10⁻¹⁴** |

In float32 the same 4745 m OPL is a phase of 6 × 10¹⁰ rad, whose representation
error alone is ~7 × 10³ rad. Those eight bins are pure noise, and the GPU path is
float32. So the floor is not tidying: without it the benchmark cannot make the
phase claim it exists to make.

`couplers.streaming.grazing_floor_for_phase_budget` derives the number rather than
picking one — requiring `Δφ ≤ 0.01 rad` in float32 over 50 µm gives
`dₙ ≥ 7.49 × 10⁻³`, rounded up to the frozen `1 × 10⁻²`. One value serves both
precisions so the mode set (7 825 bins) is identical and the two are comparable.

The floor is applied to the ray ensemble **and to the oracle alike** — both from
the same two conditions, and `test_the_retained_mask_is_the_one_the_coupler_uses`
holds the two implementations together. It is a declared property of the benchmark
configuration, not a change to either coupler.

Note the asymmetry: in float32 the eight bins fail the `radial < 1` test anyway
(float32 rounds `0.36 + 0.64` to exactly 1), so `excluded_bin_count` reads 0 there
and 8 in float64. The retained set is 7 825 either way. The floor is what makes
that agreement a guarantee instead of a coincidence of rounding.

**Follow-up recommended.** This is a property of `C_RAY_TO_WAVE`'s kernel, not of
this benchmark, and any future caller propagating a wide-angle spectrum will meet
it. It is recorded as hazard **H4** in
`knowledge/couplers/ray_to_wave/conventions.md`, and deciding whether the coupler
should refuse or band-limit such modes itself needs its own ticket and its own
oracle — the same disposition CHE-50 took.

---

## 7. Finding 2 — launch position cancels exactly, so `P` and `S` are not independent axes here

`spectrum_to_rays` applies `exp(i k d·r_p)` at emission. The reconstruction
subtracts `d·x₀` at the sensor, and for a straight ray `x₀ = r_p + Z d/dₙ`, so:

```
k[ Z/dₙ + d·(x − r_p − Z d/dₙ) ]  =  k[ Z dₙ + d·x ]
```

The launch position cancels **identically**. This is a theorem about any
shift-invariant stack, and it has a corollary that cannot be engineered away: for
an exact plane-wave oracle to exist at all, the system must be shift-invariant —
so *any* configuration admitting this benchmark's oracle also makes launch
position cancel. The two cannot both be had.

Measured, not argued. `P = 4` and `P = 1` over the same modes agree to
**< 10⁻¹¹** (`test_launch_position_cancels_exactly_for_a_shift_invariant_system`),
and the sweep shows it across the (P, S) grid — NCC is constant along `P·S = const`
to nine digits:

| (P, S) | P·S | NCC |
|---|---|---|
| (4, 1024) | 4096 | 0.9989048395 |
| (8, 512) | 4096 | 0.9989048415 |
| (16, 256) | 4096 | 0.9989048427 |
| (256, 1024) | 262144 | 0.9999815134 |
| (1024, 256) | 262144 | 0.9999815137 |

The positional sampler is the second half of the reason: ray `p·S + s` draws from
a PCG64 stream advanced to that index, so the *union* of samples over all launches
depends only on `P·S`. Combined with launch cancellation, `(P, S)` with fixed
`P·S` is literally the same field.

**What this does and does not cost.** The DoD's spatial and angular convergence
items are still met — each was independently varied over a full axis and each
converges — but they are the same curve, and the report will not claim otherwise.
The 2-D heatmap (`plots/ncc_P_S_heatmap.png`) shows the anti-diagonal structure
directly.

**What it buys.** Because the cancellation is exact rather than approximate, the
launch phase is verifiable to machine precision instead of through a convergence
study, which is a far stronger test. `P` remains a genuine variance-reduction axis
(independent draws per launch), and it is load-bearing for any *non*-shift-invariant
system — which is the case the reference implementation is built for and the case
M4 will bring.

**Consequence for the sweep's design.** An early-stop rule that watched for
consecutive-point agreement was implemented, run, and then **removed**: nested
samples make `(P, S)` and `(P, 2S)` correlated realizations, they agree with each
other better than either agrees with the truth, and the rule declared a plateau at
`S = 512` when the final selection needed `S = 1024`. The same caveat applies to
Phase 27's doubling criterion, which is the ticket's declared gate and is still
applied exactly as specified — with the uncorrelated evidence (oracle distance,
seed spread) stated next to it.

---

## 8. Finding 3 — the NCC gate is met 64× before the radiometry converges

Phase 23 warned that NCC is blind to a global scale. The sweep makes it concrete.

At the NCC-converged pair, `P·S = 4096`, the reconstructed **total power is 24.7 %
high**. The excess is a bias, not noise, and its sign is not accidental: for a
Monte Carlo estimate of a complex field, `E|Σa|² = |ΣEa|² + Var`, so the intensity
is high by the estimator's own variance and decays with the ray count at the same
rate.

| P·S | 1 − NCC | relative power error |
|---|---|---|
| 1 024 | 7.08 × 10⁻³ | **+0.978** |
| 2 048 | 2.42 × 10⁻³ | +0.433 |
| **4 096** | **1.10 × 10⁻³** | **+0.247** |
| 8 192 | 4.74 × 10⁻⁴ | +0.114 |
| 32 768 | 9.15 × 10⁻⁵ | +0.037 |
| 131 072 | 2.93 × 10⁻⁵ | +0.010 |
| **262 144** | **1.85 × 10⁻⁵** | **+0.0021** |
| 1 048 576 | 4.26 × 10⁻⁶ | −0.0001 |
| 8 388 608 | 6.53 × 10⁻⁷ | +0.0003 |
| 67 108 864 | 6.35 × 10⁻⁸ | −0.0001 |

So the report gives **two** pairs, and the distinction is the practical one:

* **smallest NCC-converged**: `P = 4`, `S = 1024`, `P·S = 4096` — the ticket's
  declared gate. Use it for a PSF *shape*.
* **smallest radiometrically converged** (gate **and** `|power error| < 1 %`):
  `P = 256`, `S = 1024`, `P·S = 262 144` — **64×** more rays, NCC 0.999981513,
  power error +0.21 %. Use it for an amplitude, an efficiency or a Strehl.

`1 − NCC` falls as `N^-1.05` over five decades — consistent with the estimator's
`N^-1/2` field-error convergence that CHE-25 fitted, since `1 − NCC` is quadratic
in the field error. That is an independent confirmation of the M2 characterization
from a completely different observable.

---

## 9. Memory, and the guards that were not needed

Peak GPU memory as a function of what it should depend on, fitted on log-log:

| series | fitted slope | claim |
|---|---|---|
| vs effective chunk, at fixed `P·S` = 524 288 | **0.997** | memory tracks the chunk |
| vs `P·S`, at fixed chunk = 65 536 | **−3.0 × 10⁻¹⁶** | memory is flat in the total |

The second is exactly flat: 14 549 504 B at `P·S` = 131 072 and the identical
value at 8 388 608, a 64× larger population. That is Phase 32's claim as a number
rather than a picture (`plots/memory_scaling.png`).

Getting that plot right took one correction worth recording. The first ladder used
the *calibrated* chunk — the largest the envelope allows — so the smaller
populations ran in a single chunk and their *effective* chunk was their own size.
The "flat in P·S" panel then showed a straight rising line, measuring the chunk
while claiming to measure the total. The ladder now picks a fixed chunk below the
smallest population so every point genuinely chunks, and
`effective_chunk_size = total_rays / chunk_count` is a recorded column so the two
can never be confused again.

Chunked execution **at the reported pair**, on the device (the calibrated chunk is
larger than 4096 rays, so the sweep's own run of the pair used one chunk):

| chunk size | chunks | NCC vs oracle | 1 − NCC vs the single-chunk run |
|---|---|---|---|
| 1 024 | 4 | 0.998904842 | 3.64 × 10⁻¹⁴ |
| 256 | 16 | 0.998904841 | 2.90 × 10⁻¹⁴ |
| 64 | 64 | 0.998904843 | 4.25 × 10⁻¹⁴ |

Calibration measured **221 GPU bytes per ray** from two probe runs (917 504 B at a
4 096-ray chunk, 3 633 152 B at 16 384), giving a chunk of 2 097 152 under a 4×
safety factor against a 30.5 GB usable envelope (60 % of free, less a 2 GB
reserve).

**No guard fired.** `memory_failures` and `budget_skips` are both empty, and every
one of the 48 candidates is `PASS_RUN`. The guards are nevertheless tested rather
than merely present — the swap watchdog, the host reserve, the GPU envelope, the
per-process budget and the child-kill path are each driven past their threshold in
`tests/test_resources.py` and `tests/test_metalens_controller.py`, because a
guardrail nobody has seen trip is a comment.

One implementation note with a real effect: JAX preallocates 75 % of the device by
default. `metalens_candidate` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any
JAX import, without which `mem_get_info` would report ~38 GB taken before a single
ray existed, the memory-scaling study would be flat by construction, and the
process would hold memory no candidate was using on a shared eight-GPU server.

---

## 10. Device residency and precision

Read off the arrays at every boundary, never from a request
(`tests/test_metalens_bridge_gpu.py`):

| boundary | positions | directions | amplitude | OPL |
|---|---|---|---|---|
| `wave_to_ray` output | float32 cuda:0 jax | float32 cuda:0 jax | complex64 cuda:0 jax | float32 cuda:0 jax |
| Optiland output | float32 cuda:0 jax | float32 cuda:0 jax | complex64 cuda:0 jax | float32 cuda:0 jax |
| sensor accumulator | — | — | complex64 cuda:0 jax | — |

The `wave_to_ray → Optiland → ray_to_wave` bridge is a **namespace change and
nothing else** in both directions: `namespace_conversion` true,
`host_transfer` false, `device_transfer` false, `dtype_conversion` false,
`lossy` false. On CUDA that is DLPack, so the buffer never leaves the device.

`ray_id` is the one deliberately host array, and it is reported in a separate
`bookkeeping` group precisely so a residency check does not cry wolf on it: the
ids come from the seeded host sampler that pins reproducibility, they carry no
precision and no physics, and putting them on the device would buy a transfer and
nothing else.

Optiland, read back from the solver rather than echoed: backend `torch`, device
`cuda`, precision `float32`, **grad mode disabled**. JAX: backend `gpu`, device
`cuda:0`, `jax_enable_x64` false. Environment: torch 2.13.0+cu126, CUDA runtime
12.6, driver 550.163.01, NVIDIA RTX A6000 (sm_86), optiland 0.6.0, numpy 2.2.6,
jax 0.6.2, Python 3.12.14.

No `FAIL_DEVICE_FALLBACK` occurred. The check is not vacuous: the candidate
verifies `jax.default_backend() == "gpu"` before doing any work and refuses
otherwise, because the klujax/SAX pin (PB4a) produces exactly a *successful* run on
the wrong device, and it is irreversible once JAX's backend is built.

### The amplitude never enters the solver

`RealRays.i` is a real bookkeeping quantity and the pinned solver has no complex
ray field. So the complex amplitude is not among the arrays bridged into torch: it
stays on the wave side and is re-attached by ray id.
`|a|²` is computed on the wave side and bridged in as an intensity so Optiland's
clipping bookkeeping is meaningful, and the value that comes back is read for
exactly one purpose — deciding which rays were clipped (none were, in any
candidate). The amplitude returns **bit-identical**
(`test_the_complex_amplitude_survives_the_trace_unchanged`).

Ray order is *checked*, not assumed. Optiland 0.6.0 clips by zeroing intensity
rather than by removing rows, so order happens to be preserved — but that is a
property of the pinned version, so `CoherentRayBatch.with_traced_state` refuses a
reordered or shortened id array rather than pairing the wrong amplitude with the
wrong ray.

### The OPL contract on this path

CHE-30/CHE-41's refusal to promote `opd_native` to an OPL is **not** being
revisited. It does not apply here for a structural reason: on this path the caller
*constructs* the rays, so `opd` starts at exactly zero on a plane the caller
declared — the same plane the wave side decomposed on — and `Surface._trace_real`
accumulates `opd += |t| · n_pre(λ)`, an index-weighted geometric path in
millimetres. The reference is known by construction rather than inferred, and
`declared_launch_opl_reference` records that in the artifact.

Verified against the closed form for both configurations, including the glass leg:
`Σ_layers n·t/dₙ'` with `dₙ'` from Snell, agreeing to `rtol = 10⁻¹¹`
(`test_the_optical_path_matches_the_analytic_geometry`).

---

## 11. Sampling-density validation (Phase 29)

Measured, not asserted as folklore. Same spatial sampling, same seed:

| P·S | `p_mag` | `p_uni` |
|---|---|---|
| 4 096 | 0.998905 | 0.980998 |
| 16 384 | 0.999770 | 0.998371 |

`p_mag` reaches at 4 096 rays what `p_uni` needs ~16 384 for — about **4× more
efficient**, which is the expected behaviour for a spectrum concentrated in one
lobe (NA 0.196 out of a full hemisphere). Both converge toward the same reference,
so the importance weight is not biased; `test_dropping_the_importance_weight_biases_the_result`
is the negative control, and it is run under `p_mag` deliberately — under `p_uni`
the `1/p` factor is a constant and omitting it would only be a global scale, which
would make the control vacuous (CHE-44).

---

## 12. CHE-50: did the missing curvature term become observable?

**No, and the reason is structural rather than lucky.** This benchmark
reconstructs directly **at** the sensor plane with zero further propagation, which
is the configuration in which the absent `exp(i k r²/2R)` term is not merely
invisible in `|U|²` but genuinely absent from the answer. The comparison is
against an oracle evaluated on the same plane, and the exactness limit reaches
8.9 × 10⁻¹⁴ in the *complex field* — not only in intensity — which it could not do
if a curvature term were missing from the operator being tested.

So CHE-70 is **not** CHE-50's trigger condition. The composition CHE-50 named —
"a propagation-sensitive hybrid composition" — is one that propagates the
reconstructed field *further*, and this one does not. A future benchmark that
reconstructs at a pupil and then propagates to a sensor with Chromatix would be
that trigger. Recorded here so the next ticket does not have to re-derive whether
CHE-70 settled it: it did not, and it did not need to.

---

## 13. Tests

| suite | before | after |
|---|---|---|
| `./run.sh pytest -q` | 536 passed, 33 skipped, 175 s | **717 passed, 48 skipped, 178 s** (CHE-71 later took this to 770) |
| `./run.sh --gpu pytest -q -m gpu` | 33 passed | **48 passed**, 77 s |

The 18 regression tests the ticket requires, and where each lives:

| # | requirement | test |
|---|---|---|
| 1 | multiple launch positions are actually used | `test_multiple_launch_positions_produce_p_times_s_rays` |
| 2 | launch-position phase changes correctly | `test_the_launch_phase_is_the_fourier_shift_phase` |
| 3 | complex amplitude survives the Optiland handoff | `test_the_complex_amplitude_survives_the_trace_unchanged` |
| 4 | Optiland intensity is not substituted for amplitude | `test_an_optiland_intensity_cannot_stand_in_for_the_amplitude` |
| 5 | path/phase increment has the right sign and units | `test_the_optical_path_matches_the_analytic_geometry`, `TestUnitBoundary` |
| 6 | free-space plane-wave reconstruction | `TestExactnessLimit` (both configurations) |
| 7 | two-ray interference | `TestTwoRayInterference` (5 tests) |
| 8 | P-dependent power normalization | `test_growing_the_spatial_count_does_not_multiply_the_power` |
| 9 | magnitude-importance weighting | `test_magnitude_importance_sampling_is_the_more_efficient_one_here` |
| 10 | uniform-sampling weighting | `test_both_densities_approach_the_same_analytic_reference` |
| 11 | GPU device residency | `TestDeviceResidency` (8 tests, `gpu`-marked) |
| 12 | full vs chunked numerical equivalence | `test_the_complex_field_is_unchanged_by_the_chunk_size` |
| 13 | chunk size does not change normalization | `test_the_normalization_is_the_estimators_not_the_chunks` |
| 14 | memory-safe angular sampling | `test_the_memory_safe_sampler_reproduces_the_reference_distribution`, `test_drawing_a_huge_sample_allocates_only_the_sample` |
| 15 | the memory guard rejects an unsafe request | `test_an_unsafe_chunk_is_skipped_before_it_allocates` |
| 16 | swap watchdog failure path | `test_container_swap_growth_trips_the_guard`, `test_one_byte_of_container_swap_is_enough` |
| 17 | child-process failure is recorded cleanly | `TestChildFailures` (4 tests) |
| 18 | no complete large ray bundle reaches the CPU | `test_the_only_declared_host_transfers_are_the_final_grid_sized_arrays`, `test_the_process_rss_does_not_grow_with_the_ray_population` |

Requirement 14's memory half is worth spelling out: drawing 10⁶ samples grows RSS
by less than 200 MiB, against the 62 GB an `(n, W)` conditional table would need
at that `n`. The bound is not a close call, which is the point — Phase 8's
`O(H·W + chunk)` requirement holds by construction, not by tuning.

**Two tests are deliberately falsifiable and were falsified in the wrong
configuration**, which is what makes them worth having:
`test_the_grazing_band_limit_is_what_makes_the_exactness_limit_exact` asserts that
*removing* the floor makes the exactness limit orders of magnitude worse, and
`test_the_pinned_solver_applies_no_fresnel_amplitude_loss` will fail if a future
Optiland starts applying Fresnel coefficients — at which point the slab oracle
needs them too.

### Test statistics that were corrected during development

Two tests were written wrong first and are worth naming, because the fix is the
interesting part:

* the sampler goodness-of-fit initially bounded the *worst* bin's relative
  deviation. Over 7 825 bins with expected counts ~50 the worst bin is several
  Poisson sigma out by construction, so the bound was both wrong and flaky. It is
  now a chi-square with a known null (`χ²/dof = 1 ± √(2/dof)`), plus a two-sample
  χ² against `draw_indices`'s own frequencies.
* the two-sample statistic's null was initially taken as `dof/2`. For two
  equal-size random draws `Var(O₁ − O₂) = 2E` while `(O₁ + O₂)` estimates `2E`, so
  the expectation is `dof`. Measured 0.9986.

---

## 14. What was not tested, and what is out of scope

* **The tutorial suite** (60 tests, ~33 min) was not run. Nothing here changes a
  dependency pin or `docker/`.
* **Gradients.** Autograd is disabled in every candidate and *checked* to be off,
  per Phase 13. No gradient claim is made or implied; `derivative.verified` remains
  false everywhere and the cross-framework boundary remains `forward_only`. The
  differentiability smoke test Phase 13 permits was **not** run — it would be a
  smoke test only, it promotes nothing, and it belongs with the custom derivative
  and directional finite-difference test that a gradient claim would need.
* **Multi-GPU.** Every candidate ran on one device, sequentially. `AGENTS.md`
  forbids concurrent GPU jobs and Phase 15 forbids parallel candidates; the
  authorisation for two GPUs was not needed and not used.
* **The reference implementation was not vendored, pinned, fetched or executed**,
  by instruction. `knowledge/couplers/*/coupler_card.yaml`'s `vendored: false`,
  `pinned: false`, `executed_by_this_repository: false` claims and
  `test_reference_implementation_is_recorded_as_unused` are therefore unchanged.
  The estimator structure (`P` spatial launches × `S` angular samples, the
  launch-position Fourier shift phase, the `1/N` normalization) is taken from the
  paper and its SI. **No commit SHA is recorded, because no commit was fetched**,
  and the unvendored implementation is not cited as evidence anywhere.
* **Off-axis illumination, polarization, broadband.** The configuration is
  monochromatic, scalar, normally incident, fully coherent, and says so.
* **Larger grids.** 100 × 100 is the ticket's specification. The band limit's
  derivation is grid- and distance-dependent, so a different grid needs its floor
  re-derived — `grazing_floor_for_phase_budget` does that, but no other grid was
  run.

---

## 15. Follow-up issues recommended

1. **`C_RAY_TO_WAVE`'s near-grazing cancellation (§6).** The kernel loses the
   phase of modes near grazing, and the loss is silent. Options are a declared
   refusal, a coupler-level band limit, or reformulating the constant phase so the
   cancellation does not occur. Needs its own oracle. Recorded as H4 on the
   coupler conventions; a new ticket should decide.
2. **`P`/`S` separability in a non-shift-invariant system (§7).** The one thing
   this benchmark structurally cannot test, because an exact plane-wave oracle
   requires shift invariance. M4's composition is the natural place; the oracle
   there has to be something other than a plane-wave transfer function.
3. **Radiometric convergence as a first-class gate (§8).** The 64× gap between the
   NCC pair and the power pair is large enough that any downstream efficiency or
   Strehl claim needs the latter. Worth deciding whether the project's PSF gates
   should carry a power criterion generally, not only here.
4. **Phase 27's doubling criterion under nested sampling (§7).** It compares
   correlated realizations and is weaker than it reads. Worth restating in the
   protocol as "oracle distance plus seed spread", with the doubling test as a
   cheap screen.
