# C_PATCH_WFT — the mathematics, and which claim rests on which equation

Source: DOI `10.1021/acsphotonics.6c00818`, SI S10 and SI Algorithm S1. The
equations are the paper's. Every *number* here is produced by this repository.

The two composed halves are not re-derived: `knowledge/couplers/ray_to_wave/theory.md`
has the accumulation kernel and `.../wave_to_ray/theory.md` has the spectral
resampling estimator. This document covers only the windowing.

## 1. The operator

Let `U(r)` be the field incident on a planar DOE with complex transmission
`t(r)`, and let `{w_j}` be a set of patch windows with centres `c_j`. A patch is
an indicator window — not a taper (§4) — so

```
w_j(r) = 1  for |r - c_j| within the patch, 0 otherwise
sum_j w_j(r) = 1   (partition of unity, over a covering set)
```

For each patch, the coupler forms the local windowed Fourier transform of the
transmitted field, pads it to the derived pad width, and resamples the resulting
angular spectrum into secondary rays:

```
V_j(k) = FFT[ w_j(r) t(r) U(r) ]          on the patch's own pad grid
rays_j = resample( V_j, launch position c_j )
```

and sums the patches coherently. Because the windows partition unity, the sum of
the windowed transmitted fields *is* the transmitted field:

```
sum_j w_j t U = t U
```

That identity is the whole equivalence. Everything below is a statement about
when the discretization preserves it.

## 2. Why the global step is a special case, not a peer

With one window covering the entire aperture, `w_1 ≡ 1` and the local WFT is the
global transform. SI S10 states the relation directly: the patch-based local WFT
is the *direct implementation* and the global single-plane aggregation is the
*shortcut*. So `C_PLANAR_DOE_STEP` is this coupler at `patch_count = 1,
patch_px = grid_n`, and the registry records it as a special case rather than an
alternative.

This is what makes their agreement an **equivalence relation** rather than a
self-comparison: one route is the limit of the other. Two independent
implementations of the same operator agreeing tells you they agree; a route
agreeing with its own limit tells you the discretization is consistent — and if
the limit is *also* checked against an independent oracle, the pair is evidence.

Measured, on a 33×33 grid with all 1089 modes enumerated:

| | measured | oracle |
| --- | --- | --- |
| full-aperture patch vs global | `1.44474e-12` | `angular_spectrum_float64`, pad 0 |
| enumerated sub-aperture sum (361 positions), at z = 0 | `1.739e-15` | same |

## 3. The sub-aperture direction, and its rate

For a *drawn* subset of `P` centres from the dilated aperture, the sum is a Monte
Carlo estimate of the partition:

```
Ê = (A_draw / A_patch) * sum_{j in draw} rays_j
```

The prefactor is the **coverage correction** (§5). Enumerating every draw
position removes the sampling error entirely, which is the family's exactness
limit; drawing `P` of them leaves a residual that falls with `P`. Measured:

| P | residual (z = 0, pad-0 oracle) |
| --- | --- |
| 4 | 1.57 |
| 16 | 0.806 |
| 64 | 0.427 |
| 225 | 0.193 |
| 361 (enumerated) | 1.739e-15 |

The fitted slope is consistent with `P^-1/2`. **No expected exponent is
declared**, and that is deliberate: the rate in the number of drawn centres is a
Monte Carlo rate over a *finite* population, so asserting `-1/2` would be
asserting a model this family has not established. That the residual falls is the
convergence statement; the fitted slope is reported with its standard error
beside it.

## 4. Why there is no window taper

A taper `w_j(r) < 1` breaks §1's identity: it removes field that no other patch
replaces, so

```
sum_j w_j t U  <  t U
```

and the sum converges to something other than the full-DOE response. The
convergence *relation* is therefore not a smoothness property that a taper would
improve — the partition of unity is exactly what a taper destroys. This is why
the coupler carries no apodization and why applying one is a declared negative
control rather than a configuration option.

## 5. The coverage correction and the launch phase

Both exist only because of patching, and both are **exactly inert at one
full-aperture patch**:

* **Coverage.** `A_draw / A_patch` — `plan.coverage`, which is
  `draw_positions / patch_px**2` — is 1 when one patch covers everything. An
  inverted correction is then a no-op, which is how a real `A_patch/A_draw` for
  `A_draw/A_patch` inversion survived every full-aperture test in this
  repository. It is applied in `patch_secondary_rays`, once, on the emitted
  amplitude.
* **Launch phase.** Patch `j`'s window offset `c_j` carries a linear phase
  `exp(i k · c_j)`, applied once. For a single patch centred on the origin
  `c_1 = 0`, so double-counting it is a no-op.

  **Where it is applied matters, and it is not here.** `patch.py` applies *no*
  launch phase; the inter-patch phase relationship is carried by
  `C_RAY_TO_WAVE`'s per-ray `Δr` ramp, because each emitted ray already knows the
  position it launched from. Adding an explicit `exp(i k · c_j)` in the emitter is
  therefore not a missing term — it is the double-count, and `patch.py` names it
  as the thing about that module most likely to be got wrong.

Consequently the exactness gate belongs to the anchor and these two gates belong
to the sub-aperture instances. The anchor still *measures* both mutations and
records their values as a blindness, because a number is a stronger record than a
note saying the control could not run.

## 6. Where the equivalence holds, and where it stops

The identity in §1 is continuous. The implementation is discrete, and the
discrete statement is narrower: **the equivalence is exact where the patch mode
grid and the comparison grid coincide, and only there.**

At full aperture with `pad_factor = 1`, the patch's mode set and the unpadded
oracle's mode set are the same set — which is why padding the anchor (a general
clearance rule that is right everywhere else) moves its modes off the oracle's
grid and the residual jumps from `1.4e-12` to O(1). The exemption is a property
of the limit.

On a sub-aperture decomposition the patch pad grid (21 px, here) is not
commensurate with the reconstruction grid (15 px), so at a large propagation
distance the ray sum is the **non-periodic** propagated field while a discrete
ASM is the **periodic** one. Measured: `1.7e-15` at the DOE plane, `0.84` at
z = 1.26 mm, and neither route is wrong. Only the comparison is ill-posed.

The same lesson in one more coordinate: a discrete ASM's padding *is* part of the
oracle. The same route scores `8.8e-3` against a pad-200 oracle and `0.33`
against a pad-101 one.

## 7. What may decide this coupler, and what may not

| Route | May it gate? | Why |
| --- | --- | --- |
| independent float64 ASM at a named pad | **yes** | shares no code with the coupler |
| the full-aperture limit of the coupler itself | **yes**, jointly with the above | it is the special case, and it is *also* checked against the ASM |
| RW-F against RW-P | **no** | `CROSS_ROUTE` oracle → category B4 → may not gate |
| the noise-limited relation | **no** | there is no oracle in it at all (§8) |

## 8. The noise-limited relation

For two noisy estimates of the same field, the achievable agreement is bounded by
each route's own self-consistency:

```
NCC(A, B) ≈ sqrt( NCC(A, A') · NCC(B, B') )
```

where `A'` and `B'` are independent realizations of the same route. Measured on
demo3, where the paper states no conventional reference exists: predicted
`0.012943`, measured `0.014713`, ratio `1.1367`.

A ratio near 1 says the two routes' disagreement *is* the Monte Carlo noise each
carries at that budget — the strongest statement available without converging
either. A ratio well below 1 would be a systematic difference between the routes.

This is how a characterization is made rigorous **without** an oracle, and it is
a first-class instrument (`benchmarks/instances/b2_equiv.py::noise_limited_relation`)
rather than a footnote. It is still characterization: "the disagreement is noise"
is a strictly weaker claim than "the answer is right."

## 9. Gradients

`derivative.mode = surrogate`, `verified = false`. Secondary directions are held
fixed and only amplitudes carry a derivative, so the gradient is a deliberately
biased surrogate inherited from `C_WAVE_TO_RAY`. No gradient across this coupler
is certified, and no downstream code may claim one.
