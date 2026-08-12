# C_WAVE_TO_RAY — conventions

Shares the frozen frame, unit, and phasor conventions of
[`../ray_to_wave/conventions.md`](../ray_to_wave/conventions.md), including the
three inherited hazards H1–H3. This file records what is specific to the
wave→ray direction, which is where the stochastic behaviour lives.

## Frozen conventions specific to this direction

| Item | Convention |
|---|---|
| Spectral axes | `(k_v, k_u)` matching the `(y, x)` spatial order. The FFT is taken over the spatial axes in that order and never transposed silently |
| Spectral origin | Zero frequency at index `n//2` after an explicit shift, matching the spatial `n//2` origin convention |
| Evanescent cut | `k_u² + k_v² > k²` discarded. The discarded power fraction is a required output, not a diagnostic detail |
| Normal component | `k_n = +√(k² − k_u² − k_v²)`, positive root, corresponding to `+z` propagation |
| Direction | `d̂ = (k_u, k_v, k_n)/k`, unit by construction; the unit norm is asserted, not assumed |
| Importance weight | `a = Ũ/p` always applied. There is no "uniform sampling so the weight cancels" shortcut — with uniform `p` the weight is a constant, and dropping a constant changes the normalization |
| Launch phase | `φ = k_u x_p + k_v y_p` for a ray launched at `(x_p, y_p)`; `OPL` initialized to 0 |
| Normalization | `1/N` applied, because this ensemble *is* a Monte Carlo sample of an integral (SI eq S5) |
| RNG | `numpy.random.Generator(PCG64)`, explicit seed, drawn outside the coupler core |
| Sampled indices | An **input** to the core, not a side effect of it |

## Sampling is an input, not a side effect

The core takes pre-drawn spectral indices as an argument. Three consequences,
each of which would otherwise have to be engineered later:

- **Bitwise determinism is trivial**, because the core is a pure function of
  its arguments.
- **One implementation serves both the reference and the gradient study**, since
  the core contains no RNG to differentiate through.
- **SI Algorithm S2's "directions held fixed during backpropagation" becomes
  structural.** The directions are inputs, so nothing can accidentally
  differentiate through the draw. The `.detach()` on `p` in the paper's
  pseudocode is not a trick to remember; it is the shape of the interface.

## Determinism is not accuracy

M1 could conflate these harmlessly: its baselines were analytic and used no
RNG, so a reproducible number and a correct number were the same number. Here
they are different claims:

| Claim | Evidence |
|---|---|
| Reproducible | Same `(seed, config)` → bitwise-identical wavevectors, amplitudes, field |
| Exact in the enumeration limit | All propagating bins + importance weight → deterministic reference, at dtype round-off |
| Unbiased | Ensemble mean over ≥ 32 seeds within 3 standard errors of the reference, where the tolerance *is* the measured standard error |
| Converging | Fitted exponent `−0.5 ± 0.1` over ≥ 5 sweep points |

A bitwise-reproducible wrong answer is the failure mode this separation exists
to catch. `coupler_protocol.yaml` states the rule directly: bitwise determinism
"is never evidence that the estimator is accurate."

## Reporting rules

Three practices are forbidden by `M2-COUPLER-CPU-V1` and are worth restating
because each is an easy, natural mistake:

1. **Never report a single realization** as the accuracy result. Report the
   ensemble mean with its standard error and the realization spread.
2. **Never select the best of several seeds.** The seed sequence is fixed in
   advance.
3. **Never tune `N` or the realization count after seeing the metric.** The
   convergence sweep is specified before it is run.

SI Figure S3 is the reason the first rule matters: realization-to-realization
variation concentrates in dim background regions, where relative standard
deviation is large precisely because the mean approaches zero. A single
realization of a converged system still looks speckled (SI Figure S4), and an
undersampled system produces "strong, seed-dependent speckle-like fluctuations
across the entire sensor plane" that are *not* spatially reproducible. The
distinction between physical speckle and Monte Carlo noise is exactly
reproducibility across seeds — so it cannot be assessed from one seed.

## Sign and orientation checklist

Each has a negative-test twin in `probes/`.

1. **`k_n` sign.** Flipping to `−√(...)` reverses propagation. Detected by
   round-tripping through a nonzero axial distance, never by a `z = 0` test.
2. **Evanescent cut.** Omitting it admits `k_n` imaginary; with a naive `√` this
   silently produces `nan` or, worse, a real value from a complex cast. The test
   uses a field with deliberate super-`k` content.
3. **Importance weight.** Omitting `1/p` under non-uniform sampling biases the
   result. The test uses `p_mag` on a strongly concentrated spectrum, where the
   bias is large; under `p_uni` the omission is a constant and the test would
   pass for the wrong reason.
4. **Launch phase.** Omitting `φ` is invisible for a single centred launch
   position and destroys the field for `P > 1`. The test uses off-centre
   launches.
5. **Spectral axis order.** An axis-asymmetric spectrum is mandatory; a
   circularly symmetric test case cannot detect a transpose.

## Open questions the paper leaves implicit

| Question | Status |
|---|---|
| Exact form of `f(·)` mapping `Ũ` to a density in Algorithm S1 line 4 | Paper names two: uniform, and `∝ \|Ũ\|`. Implemented as a named, declared choice rather than a hidden default |
| Whether `p` is normalized over bins or over the continuous domain | Resolved by requiring the exactness-limit check to pass: any normalization error shows up immediately as a scale error against the deterministic reference |
| Whether the `1/N` of eq S5 and the per-bin measure `dk_u dk_v` of eq S2 are combined | Same resolution: pinned by the enumeration limit rather than argued |
| Treatment of the `k_n = 0` grazing bin at the propagating/evanescent boundary | Open. Excluded by default as a boundary case and recorded, since `1/k_n`-type factors are singular there |
