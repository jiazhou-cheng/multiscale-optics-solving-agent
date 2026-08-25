# demo3 estimator variance — M5.3 (CHE-120)

The issue asked a scientific question rather than an engineering one: **can the
variance of the demo3 estimator be reduced at fixed ray count, without biasing
it?** The answer has three parts, and two of them are negative results with
numbers attached.

1. **Yes: 1.47x, unbiased, from the position density.** The variance-optimal
   primary-position density is `q_c ~ ||U~_c||_1`, derived rather than guessed.
   Its ceiling is computed exactly (**1.4413x**), captured to 0.006% by a
   density that costs one integral image, and measured end to end at
   **1.4685x** with an anti-bias result at 0.3x its own fluctuation.
2. **No reweighting of the direction axis can help.** `p ~ |U~|` is already the
   Cauchy-Schwarz optimum over densities. That is a proof, not a measurement —
   and it is a statement about *densities*, which is why the position density
   still reduces the direction term (measured: `B` falls 1.4391x, `A` falls
   1.5849x).
3. **The `(P, S)` split was already optimal**, to 0.1% in the objective. The
   closed-form optimum is `S* ≈ 2.15e4` against a shipped `S = 20,000`.

Along the way the committed cost ceiling needed correcting, and not because the
code got faster: **the extrapolation law was one power out**, and the reason it
was not caught is that the old ladder's rungs sat too close to the noise floor.

## What was measured

Everything below is `rw_p` (the patch route) on demo3's characterization preset:
200x200 DOE at 6.3 um pitch behind a circular stop, 420x420 sensor at 4.2 um,
fp32, JAX on one A6000. Records in `benchmarks/probes/records/ray_wave/`,
probe `benchmarks/probes/ray_wave/demo3_variance.py`.

### The metric, and why it is not the NCC

Seed-to-seed intensity NCC is the coordinate the committed records use, and it
is reported throughout for continuity. The variance work is scored on the
**absolute field variance** `V = sum_px Var_r[F_r(px)]` over `R >= 3` seeds,
with a leave-one-seed-out error bar. Three reasons, all of them things that were
tried first and failed:

* **`V` needs no signal estimate at all.** Every arm is an unbiased estimator
  of the same field on the same grid with the same `one_over_n` normalization,
  so `V` is already in comparable units and its ratios are exact. Normalizing
  would mean dividing by `sum|F_bar|^2 - V/R`, which *is* measurable here — at
  `P = 1000`, `S = 20000`, `R = 6` it reads `301.7 - 245.7 = 56.0`, a **3.7%**
  per-realization signal fraction known to a few percent — but it would add that
  error to every ratio for no gain. (An earlier draft of this report claimed the
  denominator carried a ~100% error and the signal fraction was ~1%. The records
  do not support either number; the metric choice stands on comparability.)
* **The NCC saturates**, so it is nonlinear in the very thing being compared.
* **At `kspace_oversample = 1.5` the NCC is contaminated.** Across the four
  allocation cells `V` varies by 1.73x while the NCC moves from 0.0873 to
  0.0788 — that is, not at all. At the committed records' `oversample = 8.0` the
  same configuration reads NCC 0.0054 and the contamination is gone. **A
  seed-to-seed NCC measured at low k-space oversampling is not a measure of
  estimator convergence**, and nothing in this report uses one.

`V` also gives the convergence answer directly and without a saturating
coordinate in the way: **`V ~ N^-0.9953` over an 8x range in rays**. That is
what "noise-limited, more rays fix it" means, measured, and it is a cleaner
statement than any NCC slope.

### 1. Where the variance is

Two draws, so exactly two terms: `V(P, S) = A/P + B/(PS)`. Multiplying through,
`P V = A + B/S` is linear in `1/S`, so an `S` sweep at fixed `P` separates them.
`P = 1000`, `R = 4` seeds per rung:

| `S` | rays | `V` | jackknife |
| --: | --: | --: | --: |
| 2,500 | 2.5e6 | 9178.7 | 1.1% |
| 5,000 | 5.0e6 | 4787.3 | 0.9% |
| 10,000 | 1.0e7 | 2580.0 | 1.1% |
| 20,000 | 2.0e7 | 1468.7 | 0.9% |

`A = 3.751e5`, `B = 2.202e10`, **relative RMS residual 0.14%**. So at the
shipped `S = 20,000`, `A/P = 375.1` and `B/(PS) = 1101.0`: **74.6% of the
variance sits in the term that falls with `P x S` and 25.4% in the term that
falls only with `P`.**

**Read that split for what it is.** It says how `V` responds to the two knobs,
which is what sets `S*` below. It is **not** a division into reducible and
irreducible shares, and reading it as one was this report's first mistake: it
would have capped the achievable gain at `1/(0.746 + 0.254/1.4413) = 1.084x`,
which the measured 1.4685x falsifies outright. The reason is that a position
density multiplies *whole patches*, so it scales both terms — the per-ray weight
modulus `||U~_c||_1` is itself a per-patch quantity. Running the same `S` sweep
on the adopted arm measures that directly:

| term | uniform | adopted | ratio |
| -- | --: | --: | --: |
| `A` (falls with `P`) | 3.751e5 | 2.367e5 | **1.5849** |
| `B` (falls with `P x S`) | 2.202e10 | 1.530e10 | **1.4391** |

`B` is the term the closed form predicts, and it predicts **1.4413** — agreement
to **0.15%**. `A` falls slightly further, because for this mask the coherent
per-patch contribution happens to have much the same `sqrt(n_c)` shape. Both
residuals are 0.13-0.14%, so the two-term model holds for both arms.

The residual is the model check, and it passes a harder one. The fit was made on
the `S` axis at fixed `P`; it then predicts the *other* axis — the `P` sweep at
fixed `P x S = 2e7` — with no free parameters:

| `P` | predicted `V` | measured `V` |
| --: | --: | --: |
| 400 | 2039 | 2032 |
| 1,000 | 1476 | 1469 |
| 2,500 | 1251 | 1243 |
| 5,000 | 1176 | 1171 |

Within 0.6% everywhere. The two-term decomposition is not a parameterization
that happens to fit; it predicts an axis it was not fitted on.

**The reconstruction route contributes no variance term.** It changes `V`'s
absolute value (1469 at oversample 1.5 against 2237 at oversample 8, same
configuration) and leaves both the `1/N` scaling and every ratio below intact —
the variance ratio of the adopted estimator is 1.4685 at oversample 1.5 and
1.4709 at oversample 8.

### 2. The direction axis: nothing is available, and that is provable

The estimator draws mode `m` with `p_m = |U~_m| / ||U~||_1` and weights the ray
by `U~_m / p_m`. That weight has **constant modulus `||U~||_1` within a patch**.
Its second moment is therefore `||U~||_1^2`, and by Cauchy-Schwarz

    min_p sum_m |U~_m|^2 / p_m  =  (sum_m |U~_m|)^2,   attained at p ~ |U~|.

So the shipped direction density is exactly the variance-optimal one, and no
choice of density can improve the term that carries three quarters of the
variance. This is why no direction-axis candidate was run: one would have been
measuring noise. (The bound constrains *densities*. It says nothing about
correlated sampling — see "what would move it".)

### 3. The position axis: 1.44x, and that is the ceiling

The same argument on the outer sum gives the optimal position density
`q_c ~ f_c` with `f_c` the modulus of the weight a ray from position `c`
carries — which, by the paragraph above, is `f_c = ||U~_c||_1`, the L1 norm of
that patch's own spectrum. So the "unjustified asymmetry" the registry flagged
is not a difference of principle at all: it is the *same* principle applied to
one axis and not the other.

`||U~_c||_1` was computed **exactly** for all 90,601 candidate positions (335 s
on CPU, `demo3_position_spectral_l1.json`), which turns the available reduction
from a model into a number:

| position density | variance ratio vs uniform |
| -- | --: |
| uniform (shipped) | 1.0000 |
| `q ~ window energy` | 1.0965 |
| `q ~ sqrt(window energy)` | 1.4412 |
| `q ~ ||U~_c||_1` (exact optimum) | **1.4413** |

Two things follow. **1.4413x is the ceiling on this axis** — nothing beats the
optimum by construction. And the cheap proxy captures 99.994% of it: for a
phase-only mask the patch is a quasi-random phase over the `n_c` aperture
samples in its window, so `||U~_c||_1 ~ sqrt(n_c)`, and the measured correlation
between the exact map and `sqrt(window energy)` is **0.99995**. The 335 s
computation never needs repeating; one integral image suffices.

The `sqrt` matters and is not a free parameter: `q ~ energy` gets 1.0965, three
quarters of the way back to uniform.

**Why 1.44 and not 4.96.** M2 measured a 4.96x advantage for magnitude sampling
on a concentrated spectrum. The position axis cannot do that here because
demo3's DOE amplitude is **flat inside its aperture** — it is a phase mask
behind a circular stop. The only spatial structure left is the rim taper of the
dilated draw region: mean window fill is 0.3467, and **10.36% of candidate
positions hold no aperture sample at all**, so under the uniform density about
one patch in ten spends its entire secondary budget on zero-amplitude rays.
Independent confirmation that this model is the shipped geometry and not a
parallel one: it predicts 311 wholly-empty patches out of 3000, and the
committed perf record measured 296 (0.9 sigma on Poisson).

None of this transfers to a DOE with structured transmission. The ratio is a
property of the mask and is reported per configuration.

### 4. Candidates measured, with anti-bias

`P = 1000`, `S = 20,000` (2e7 rays), `R = 6` seeds, `oversample = 1.5`. The
control arm is routed through the shipped code path, not through a
re-implementation of it, and `tests/test_patch_positions.py` asserts the
uniform draw is bitwise what `plan_patches` produces.

| arm | variance ratio at fixed rays | at fixed cost | anti-bias ratio | vs fluctuation |
| -- | --: | --: | --: | --: |
| `uniform_iid` (control) | 1.0000 | 1.0000 | — | — |
| `uniform_jittered_grid` | 1.0077 ± 0.8% | 1.0210 | 1.0082 | 1.7x |
| `sqrt_energy_iid` | **1.4523** ± 0.8% | 1.4688 | 1.0021 | 0.4x |
| `sqrt_energy_jittered_grid` | 1.0895 ± 0.8% | 1.2508 | 1.0074 | 1.6x |
| `sqrt_energy_stratified_cdf` | **1.4685** ± 0.8% | 1.5779 | 1.0015 | **0.3x** |

Measured 1.4523 against a prediction of 1.4412 computed from the DOE alone,
before any of these runs: **0.8% agreement between a closed form and a
measurement** (and 0.15% on the `B` term the closed form actually predicts).

Two caveats on the fixed-cost column, which the issue asks for and which is the
weaker of the two. The arms differ by up to 13% in median wall clock while the
within-arm spread is 3-7%, and **that difference is not attributed** — same ray
count, same reconstruction, so it is probably real but this report does not say
why. And `sqrt_energy_jittered_grid` realized 926 of 1000 patches, because 74
equal-area cells held no density; its fixed-rays figure is therefore scaled by
the measured `V ∝ 1/N` rather than measured at matched rays directly.

**The anti-bias test, and why it is stronger than CHE-101's.** CHE-101 checked
that a fast path had not moved the seed-to-seed noise. Necessary, not
sufficient: an estimator can keep its noise and shift its mean. So each arm is
also scored on

    bias_ratio = ||F_bar_A - F_bar_B||^2 / (V_A/R_A + V_B/R_B),

which is 1 when two estimators have the same mean and grows without bound when
they do not. It pools all 1.76e5 pixels, so its own fluctuation is ~0.0034 — a
sharp test rather than a formality — but the fluctuation is **not** just the
pixel count. For complex per-pixel noise the numerator's own scatter is
`1/sqrt(N_px) = 0.0024`, and the denominator is estimated from the same six
seeds, contributing 0.0046 through the arms' jackknife errors. Combined, 0.005,
which the record reports term by term. **Every arm is then inside 1.7 sigma and
the two adopted arms inside 0.4** — an earlier draft quoted `sqrt(2/N_px)`
alone, which is the real-Gaussian figure, and read the `jittered_grid` arms at
2.0-2.5x as a near-detection. They are not. Speckle pixels are also not
independent, which pushes the true scale up further, so this is a lower bound on
the fluctuation.

That is the demo3-scale test. The **exact** version runs in
`tests/test_patch_positions.py`: on a 15x15 apertured DOE where enumerating all
361 candidate positions and every mode gives the estimator's own mean exactly,
every scheme is scored against that oracle on a scalar functional and gated at
3 sigma of its measured standard error. Seven schemes, and the tolerance is the
error bar rather than a chosen number.

**Two negative results, and they are informative.**

* **Stratification alone buys nothing** (1.0077, inside its error bar). `P` =
  1000 draws over 90,601 candidate positions is far too sparse for
  stratification to bite on an integrand that decorrelates at the sample pitch.
* **Equal-area strata cancel importance sampling** (1.0895, down from 1.4523).
  One draw per equal-*area* cell makes the between-cell allocation uniform by
  construction whatever the density, and the density varies on the scale of the
  aperture rim rather than within a cell. The fix is equal-**mass** strata: one
  draw per equal-probability interval of the CDF, for which `pi_c = P q_c`
  exactly and the weights are unchanged. That arm recovers the reduction and
  adds 1.1% (1.4685 against 1.4523), which is stratification's entire
  contribution here.

### 5. The `(P, S)` allocation was already optimal

Fixed `P x S` is **not** fixed cost: the padded transform is per patch and the
draw, trace and reconstruction are per ray, so `c(P, S) = P (a + b S)`. Fitted
on the same runs that measured the variance: `a = 2.039e-3` s/patch,
`b = 2.586e-7` s/ray, residual 6.4%. Minimizing `V` at fixed `c` gives

    S* = sqrt(a B / (b A)) = 2.15e4  (2.26e4 for the adopted arm),

against a shipped `S = 20,000` — **0.1% from the optimum in the objective**
(0.3% for the adopted arm). Quoted to two significant figures on purpose: `S*`
goes as `sqrt(a/b)` and the cost model's own residual is 6.4%, so further digits
would be timing noise. Measured directly, `V x cost` over the four cells at
fixed `P x S = 2e7`: 13,897 / **9,610** / 12,144 / 18,365 for
`P` = 400 / 1,000 / 2,500 / 5,000. The shipped cell is the measured minimum, and
the closed form agrees with it.

So the smallest `P x S` meeting a variance target is reached at the shipped
split, and the allocation lever is worth nothing. Reporting it as a *result*
rather than as a tuning opportunity is the point: the optimum is quadratically
flat, which is why it was never worth searching and why it should not be
searched again.

## The cost ceiling, corrected

### The old extrapolation law was one power out

The committed record anchored its extrapolation at **slope 1**, on the argument
that an unbiased estimator's signal fraction grows linearly in the ray count
while it is small. That argument is off by one power. What two realizations
*share* is the deterministic `|mu|^2`; what differs is the noise, whose spatial
variance goes as `(E|n|^2)^2` once the noise dominates, so

    NCC ~ Var_px(|mu|^2) / V^2 ~ N^2,

with the saturating form `NCC = 1 / (1 + c N^-p)`, which is linear in `log N`
after a logit and respects `NCC <= 1` — which the power law does not, at a
target of 0.9.

Why the old ladder did not see it: its rungs sat **1.5-2.9x above the noise
floor** `3/sqrt(N_px) = 0.00714`, and a rung near the floor is biased toward the
floor, which flattens the fit. The issue warned about exactly this failure mode
in the other direction; this is the same defect at a milder amplitude. A rerun
of that probe on current code reproduces its NCC values to seven significant
figures (0.0108605, 0.0152590, 0.0207371) and its slope of 0.927, so the
discrepancy is the ladder's placement and not drift.

A four-rung ladder starting **3.2x above the floor** and spanning 8x in rays,
`R = 3` seeds, `oversample = 8.0`:

| rays | `V` | NCC |
| --: | --: | --: |
| 4.0e7 | 1111.25 | 0.02262 |
| 8.0e7 | 551.96 | 0.07247 |
| 1.6e8 | 278.95 | 0.19430 |
| 3.2e8 | 139.93 | 0.41050 |

* `V ~ N^-0.9953` — noise-limited, confirmed on the metric that does not saturate.
* fitted power law: slope **1.397** → 5.199e8 rays for NCC 0.9
* saturating law: `p` = **1.636**, logit RMS residual 0.039 → **1.494e9 rays**

**The top rung is an out-of-sample test between the two models**, computed in
the record rather than by hand (`trend.out_of_sample_top_rung`). Fitted on the
first three rungs alone, the saturating law predicts **0.4414** at 3.2e8 rays and
the power law **0.5867**. Measured: **0.4105**. The saturating law is **7.5%**
out and the power law **42.9%** out, so the saturating law is the one quoted and
the power law is the disfavoured lower bound rather than a co-equal estimate.

### Throughput, re-measured rather than inherited

The same probe, same configuration, same seeds, before and after M5.1 + M5.2:

| rays | M0.4-era wall clock (2 seeds) | current | speedup |
| --: | --: | --: | --: |
| 2.0e7 | 96.21 s | 19.34 s | 4.98x |
| 3.0e7 | 134.76 s | 20.19 s | 6.67x |
| 4.0e7 | 179.93 s | 26.63 s | **6.76x** |

`2.249e-6` s/ray → `3.328e-7` s/ray. Larger than the 1.76x M5.2 reported on
`ramp_sum`, and consistent with it: on `ramp_sum` the `O(N_rays x N_pixels)`
reconstruction dominates and dilutes the trace and emitter fixes, while on
`kspace_splat` those two stages are nearly the whole run. **The NCC values are
unchanged to seven significant figures**, so this is throughput and not a
different estimator.

### Where that leaves the budget

| estimator | rays for NCC 0.9 | s/ray | hours/run |
| -- | --: | --: | --: |
| shipped, M0.4-era record | 1.736e9 (slope-1 anchor) | 2.249e-6 | 1.085 |
| shipped, corrected law | 1.494e9 | 3.350e-7 | **0.139** |
| adopted (`sqrt_energy_stratified_cdf`) | 1.016e9 | 3.332e-7 | **0.094** |
| adopted, its own 3-rung fit | 8.333e8 | 3.332e-7 | 0.077 |

The adopted row is the control's four-rung fit divided by the variance ratio
measured at the same oversampling (`1.494e9 / 1.4709`). It is quoted ahead of the
estimator's own three-rung fit because the four-rung ladder is better
constrained and is the only one with an out-of-sample test; the three-rung fit's
0.077 h is the optimistic end, and the 1.22x between them is fit spread rather
than a further gain. The disfavoured power law would put the shipped estimator at
5.20e8 rays and 0.048 h; that 2.9x spread is the dominant uncertainty and is
stated rather than averaged.

**A converged demo3 run goes from 1.085 h to 0.094 h, 11.5x**, and it
decomposes exactly:

    6.71x (throughput) x 1.162x (extrapolation law) x 1.471x (variance) = 11.5x

The throughput term is the per-ray cost `2.249e-6 / 3.350e-7`; the law term is
`1.736e9 / 1.494e9`; the variance term is the measured ratio. Note that
`6.76 x 1.47 = 9.9`, not 11.5 — the law correction is the missing factor, and an
earlier draft quoted 13.6x without naming either it or the fit spread.

### Against the paper's own budget

SI Table S2 gives 2.6e9 rays for this system under RW-P. The shipped estimator
now extrapolates to **1.494e9, i.e. 1.74x below it** — the same order, on a
two-decade extrapolation with a 2.9x model spread, which is agreement rather
than tension. The issue's framing stands: **the budget is a property of the
system rather than of our implementation.** The adopted estimator then lands
2.6x below the paper's figure, which is what a strictly better estimator
should do — the reference implementation draws positions uniformly and this one
does not.

## Is demo3 a validation target?

**No, and it cannot become one.** The paper states no conventional reference
exists for this system; that is the point of its Figure 5c. Every number in
every demo3 record is a self-comparison — one realization against another, or
one arm of a probe against another arm. Nothing here is scored against an
independent oracle, and no threshold in any demo3 record is a pass gate.

This is now recorded in `benchmarks/manifest.yaml` so that nobody plans against
a validation that cannot exist. What demo3 *can* support:

* **characterization** of cost, convergence rate and variance — this report;
* **anti-bias** work, because the estimator's unbiasedness is testable at a
  reduced size where an enumerated oracle does exist (`test_patch_positions.py`),
  and that result transfers to demo3 by construction rather than by measurement;
* **relative** claims between two arms of the same probe.

What it cannot support is any claim that the demo3 field is *correct*. The
components are validated elsewhere — the full-aperture patch against an
independent float64 ASM at 7.1e-13, demo2 against the same oracle — and demo3
is the composition of validated components at a scale where the composition
itself has no reference.

## M5's exit

**No numeric M5 target was recorded at milestone open.** The milestone says
"targets set at milestone open against M0's baseline harness"; neither the
milestone description nor M0.4's report records a number. So this reports
achieved factors against M0.4's baseline and characterizes the binding
constraint, which the milestone lists as the acceptable alternative exit.

Achieved, all against M0.4's own baseline records:

* **throughput 6.71x** on demo3's k-space route (per-ray cost), field unchanged
  to seven significant figures;
* **variance 1.4685x**, against an exactly-computed ceiling of 1.4413x on the
  position axis;
* **a corrected extrapolation law**, worth a further 1.162x on the ray
  requirement;
* **converged-run wall clock 1.085 h → 0.094 h, 11.5x.**

**The binding constraint is that every lever of the form "choose a better
density" is now at its optimum.** The direction density is the Cauchy-Schwarz
optimum, so no reweighting of modes can help. The position density is at its
exactly-computed ceiling, to within the 1.1% that equal-mass stratification adds
on top. The `(P, S)` split is at its optimum to 0.1% and stays there under the
new density (`S* = 2.26e4` for the adopted arm). The dominant term `B` — 74.6% of
the variance at the shipped `S` — has been reduced 1.4391x by the *position*
density, which is as far as any density gets it.

**What would move it**, none of which was attempted here:

* **Correlated sampling on the direction axis.** Cauchy-Schwarz bounds the
  choice of *density*; it says nothing about drawing the `S` modes as a
  stratified or quasi-Monte-Carlo set rather than i.i.d. The position-axis
  result is a caution rather than an encouragement — stratification bought
  1.008x there — but the direction axis is a different situation: `S = 20,000`
  draws over 90,601 modes is 100x denser than 1,000 draws over 90,601 positions,
  and stratification improves with density. This is the one lever with both a
  real mechanism and no measurement, and it is where the next attempt should go.
* **Control variates** against a paraxial or single-plane surrogate whose exact
  answer is known, as the issue suggests. Not attempted: it needs a surrogate
  accurate enough to correlate with the estimator, and constructing one is a
  separate piece of work rather than a sampling change.
* **A larger patch.** `A` and `B` both depend on `patch_px`, which was held at
  the preset's 101 throughout. The cost model's per-patch term scales as
  `pad^2 log pad`, so this trades variance against cost on an axis this report
  did not sweep.

## What is not claimed

* **No gradient.** Nothing here touches differentiability; the registry's
  `derivative.mode=surrogate, verified=false` is unchanged.
* **The default estimator is unchanged.** `PatchPlan.center_weights` defaults to
  `None` and the multiply is *absent* from the default path rather than a no-op
  in it, asserted bitwise. Every committed demo2/demo3 number remains the
  estimator that produced it, which is why this change does not trigger the
  issue's "re-measure every committed number" clause — the regenerated records
  reproduce their predecessors' physics to seven significant figures and differ
  only in wall clock and in their provenance stamp. Every stamped record the
  diff invalidated was nevertheless regenerated: the four enrolled `m3_*`
  records (differing only in timings and RSS, checked leaf by leaf), the four
  `perf_demo*` M0.4 baselines (NCC and relative-L2 unchanged), and
  `demo3_convergence_kspace_rw_p`.
* **The adopted density is not reachable through the graph node.** It is a
  library-level option on `couplers.patch_positions.plan_positions`;
  `patch_node.py` still exposes the uniform draw only. Deliberate: exposing it
  through the node is an API change with its own validation surface.
* **The 1.44x is a property of demo3's mask**, not of the method. A DOE with
  structured transmission would have a different ceiling, possibly much larger.
  `predicted_variance_ratio` computes it from the mask alone in negligible time,
  so this is a thing to check per configuration rather than to inherit.
* **The `oversample = 1.5` seed-to-seed NCC values in these records are not
  convergence measurements** and are reported only because the probe emits them.
  See "the metric".
* **`bias_ratio` assumes independent pixel noise** in its fluctuation estimate,
  which speckle only approximately satisfies. It is read against a few multiples
  of that scale, not one, and the exact unbiasedness gate is the enumerated-
  oracle test rather than this.

## Resources

All runs sequential on one GPU (device 6), peak 0.75 GB device memory, swap
growth zero throughout. About 80 minutes of GPU time including a full
regeneration pass after review, and 6 minutes of CPU for the exact
`||U~_c||_1` map. One command hit the 10-minute limit and was
terminated; its container exited cleanly and its work was re-run split across
two commands, which is what `--stage ladderfit` exists for.
