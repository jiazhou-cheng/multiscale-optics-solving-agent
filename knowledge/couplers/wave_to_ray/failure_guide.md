# C_WAVE_TO_RAY — failure guide

The wave→ray direction fails in two distinct ways: as a spectral transform, and
as a Monte Carlo estimator. Diagnose which before changing anything, because
the fixes are unrelated.

## Is it a transform bug or a sampling bug?

Run the **exactness limit** first: enumerate every propagating bin with the
importance weight applied. That estimator has no sampling error at all.

| Exactness limit | Ensemble mean | Verdict |
|---|---|---|
| fails | — | Transform bug. Stop; do not tune `N` |
| passes | fails | Bias — most likely a missing or wrong `1/p` |
| passes | passes, high variance | Genuine Monte Carlo noise. Raise `N` or change `p` |

This ordering exists so that sampling cannot be used as an excuse for a
deterministic error, and it is why the protocol makes the exactness limit
mandatory rather than optional.

## Transform failures

| Symptom | Cause | Note |
|---|---|---|
| `nan` in directions, or a real `k_n` where it should be imaginary | Evanescent cut omitted; `√` of a negative number | Use a field with deliberate super-`k` content to test; a smooth field may have negligible evanescent content and pass by accident |
| Field propagates backwards | `k_n` sign flipped to the negative root | Invisible at `z = 0`. Test through a nonzero axial distance |
| Result transposed | Spectral axes not `(k_v, k_u)` matching spatial `(y, x)` | A circularly symmetric spectrum cannot detect this. Use an axis-asymmetric one |
| Constant scale error | Normalization: the `1/N`, the per-bin measure, or `p`'s normalization | Pinned by the exactness limit rather than argued about |
| Phase relation lost between launch positions | Launch phase `φ = k_u x_p + k_v y_p` omitted | Invisible for a single centred launch; destroys the field for `P > 1` |
| Large discarded-power fraction | Not a bug | The field has significant evanescent content and should probably not be turned into rays. Report it; do not suppress it |

## Estimator failures

| Symptom | Cause | Note |
|---|---|---|
| Ensemble mean offset from the reference by more than 3 standard errors | Missing `1/p` under non-uniform sampling | Test with `p_mag` on a strongly concentrated spectrum. Under `p_uni` the omission is a constant, so the test would pass for the wrong reason |
| Fitted convergence exponent far from `−0.5` | Correlated draws, a degenerate `p` (zeros where `Ũ ≠ 0`), or a deterministic error dominating the sampling error | A `p` with a zero where the spectrum is nonzero makes the estimator *inconsistent*, not merely slow: those modes are never drawn and `1/p` cannot rescue them |
| Variance far worse under `p_mag` than `p_uni` | Expected for multilobed spectra; the paper reports comparable rates there | Only a bug if `p_mag` is worse on a strongly concentrated spectrum |
| Results change between runs at fixed seed | RNG drawn inside the core, or a nondeterministic reduction order | Sampled indices are an input to the core precisely to prevent this |

## Speckle: physical or numerical?

The single most misdiagnosed symptom. Both look like speckle.

- **Physical interference** is *spatially reproducible across independent
  seeds*. The paper confirms exactly this for the hologram–lens system, where
  the resolved interference structure persists across independent Monte Carlo
  realizations and is attributed to real ray–lens interaction in the
  nonparaxial regime.
- **Monte Carlo undersampling** produces strong, seed-dependent fluctuations
  across the whole plane that are *not* reproducible (SI Figure S4).

So the diagnostic is: rerun with a different seed and compare spatially. A
single realization cannot answer this, which is why the protocol forbids
reporting one.

Note also that relative standard deviation is naturally large in dim
background regions, where the mean approaches zero (SI Figure S3b). A large
*relative* deviation there is not evidence of non-convergence; check the bright
features and the integrated power instead.

### Deciding it quantitatively

"Rerun with a different seed and compare spatially" is the right instinct and it
has three failure modes of its own. Each has bitten this repository.

| Symptom | Cause | Check |
|---|---|---|
| Seed-to-seed NCC near zero, or negative | Below the noise floor `3/√N_px` — that is zero with an error bar, not a small correlation | Compute the floor and refuse to fit through rungs under it. A ladder that did returned a convergence slope of **5.7** |
| A convergence ladder's slope is suspiciously shallow | Rungs too close to the floor: the floor biases each rung upward and flattens the fit | Start at ≥ 3× the floor. Rungs at 1.5–2.9× turned a slope of ~1.4 into 0.93 |
| Seed-to-seed NCC does not move when the ray count or the variance does | The reconstruction, not the estimator, is setting it — a coarse `kspace_oversample` leaves structure common to both realizations | demo3 at `oversample=1.5` reads NCC ~0.08 while the variance changes 1.73×; at 8.0 it reads 0.0054 and tracks |
| A ladder extrapolates to an implausible budget | Slope-1 extrapolation. NCC goes as `N²` while small, and saturates | Fit `NCC = 1/(1 + c·N^-p)`; it respects `NCC ≤ 1`, which matters at a target of 0.9 |

The measurement that avoids all four: **the variance itself.** `V = Σ_px
Var_r[F_r]` over ≥ 3 seeds has no floor and no saturation, and `V ∝ 1/N` is the
direct statement of "noise-limited, more rays fix it" (measured at `N^-0.995`
over 8× on demo3). Compare arms on **absolute** `V`: every arm estimates the
same field with the same `one_over_n` normalization, so `V` is already
comparable and its ratios are exact, while dividing by the signal estimate
`Σ|F̄|² − V/R` (3.7% of the total at demo3's 2e7-ray configuration) would add
that estimate's own error to every ratio for nothing.

Full worked case, including the position-density optimum and the `(P, S)`
allocation: `benchmarks/reports/2026-08/demo3_estimator_variance.md`.

## Structured failures the coupler must emit

| Code | Condition |
|---|---|
| `MISSING_DECLARATION` | Input omits units, frame, phasor, origin, pitch, or reference plane |
| `SEED_NOT_DECLARED` | Sampling requested without an explicit seed |
| `DEGENERATE_DENSITY` | `p` is zero on a bin where `Ũ` is nonzero — the estimator is inconsistent, not just noisy |
| `ALL_MODES_EVANESCENT` | No propagating modes survive the cut |
| `CURVATURE_BOUND_EXCEEDED` | `arcsin(D/2R)` exceeds the caller's stated error threshold |
| `GRAZING_BIN_INCLUDED` | A `k_n = 0` bin was requested; excluded by default as singular |
| `PAD_STATE_UNKNOWN` | Field arrived without a declared pad width, so its extent cannot be trusted |

## What is *not* a bug

- **A converged result that still looks speckled.** See above.
- **Discarded evanescent power.** Correct behaviour, provided it is reported.
- **A biased gradient.** The estimator is biased by construction (SI S7.2); it
  holds sampled directions fixed. That is a documented property to be measured,
  not a defect to be fixed — and fixing it via Gumbel–Softmax was evaluated by
  the authors and rejected on memory cost.
