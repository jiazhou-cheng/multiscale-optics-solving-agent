# The singlet residual, attributed

**CHE-117 (M4.2).** `B3-PSF-SINGLET` carries an unmet gate and a negative control
that fires backwards. This report says what both of them are.

## What was open

The composed chain `M_RAY_OPTILAND -> C_RAY_TO_WAVE -> M_WAVE_CHROMATIX` is the
only one in the repository with a genuinely independent decider: O1, the analytic
Airy pattern -- paraxial, aberration-free, sharing no code and no traced data with
the coupler it judges. On the frozen `M3-SINGLET-REF` system at 787,969 rays it
measures **2.21e-3** against a **1.0e-3** gate frozen since M3.2.

Two things about that were unexplained.

1. The production *weighted* configuration measures 2.21e-3 and the *uniform*
   configuration measures 9.21e-4 -- so the pre-CHE-47 code, without the
   quadrature weight, agrees with the analytic oracle **better**. The weight is
   independently required (it is what fixes CHE-33's `N^2.0024` absolute-power
   divergence), so this reads as the correct code being penalised.
2. That ordering is exactly the `inverted-quadrature-weight` negative control:
   adding the weight must improve agreement by at least 1.2x, and it measures
   0.42. A control reading 0.42 against a 1.2 floor is not a passing gate with a
   caveat; it is the suite reporting that it does not understand its own subject.

CHE-48 was opened to decompose this and was closed with no comment, no commit and
no artifact.

## What was done

Two probes, both on O1 only. **O2 -- our own float64 ASM/RS propagator -- is not
consulted anywhere in this work.** It is the oracle L2-PSF-01 already had to
retire from this gate as circular validation, and nothing here readmits it.

### 1. Is it the sensor sampling? No.

`benchmarks/probes/singlet_residual_grid.py` holds the 128 um physical window
fixed and refines the sensor pitch through 1.0, 0.5, 0.25 and 0.125 um -- 6.5 to
51.9 pixels across the Airy radius, an 8x refinement bracketing the frozen
configuration on both sides.

| px / Airy radius | weighted vs O1 | uniform vs O1 |
| --- | --- | --- |
| 6.49 | 2.207239178834053e-3 | 9.211697e-4 |
| 12.97 | 2.2072391812867093e-3 | 9.211638e-4 |
| 25.94 | 2.2072391810006816e-3 | 9.211656e-4 |
| 51.88 | 2.2072391812489756e-3 | 9.211662e-4 |

Identical to ten significant figures. This also retires half of CHE-103's
contribution to the question: the 2.44-pixels-per-Airy-radius finding is about
the separate pupil-to-focus grid, and the sensor grid this residual is measured
on is converged.

### 2. Which part of the weight is responsible? The rim, and only the rim.

The production weight differs from uniform in exactly two places: the central ray
gets 3/4 of a nominal cell and the outermost ring gets 1/2. Substituting variants
of `couplers.quadrature.hexapolar_area_weight_m2` that remove one correction at a
time:

| rings | production | rim removed | centre removed | uniform |
| --- | --- | --- | --- | --- |
| 128 | 2.18905e-3 | 3.93282e-3 | 2.19472e-3 | 3.92895e-3 |
| 256 | 2.20357e-3 | 1.14103e-3 | 2.20498e-3 | 1.14070e-3 |
| 512 | 2.20724e-3 | 9.20796e-4 | 2.20759e-3 | 9.21164e-4 |

Removing the rim half-weight reproduces the uniform arm to within 0.099%.
Removing the centre correction reproduces the production arm to within 0.26%. The
two arms differ in one place.

### 3. Is the rim half-weight the defect? No -- and this is the finding.

Extending the ray ladder past the committed 512 rings:

| rings | rays | weighted vs O1 | uniform vs O1 | improvement factor |
| --- | --- | --- | --- | --- |
| 128 | 49,537 | 2.18905e-3 | 3.92895e-3 | 1.7948 |
| 256 | 197,377 | 2.20357e-3 | 1.14070e-3 | 0.5177 |
| 362 | 394,219 | 2.20601e-3 | 7.03878e-4 | 0.3191 |
| 512 | 787,969 | 2.20724e-3 | 9.21164e-4 | 0.4173 |
| 724 | 1,574,701 | 2.20785e-3 | 1.24973e-3 | 0.5660 |
| 1024 | 3,148,801 | 2.20816e-3 | 1.51637e-3 | 0.6867 |

**The two arms are converging to the same number.** The weighted arm is flat to
0.87% over a 64x range in ray count here, and flat since 24 rings in the
committed ladder (`m3_quadrature_weight.json`, where the uniform arm starts at
8.3e-2). The uniform arm descends, crosses the weighted arm near 181 rings,
reaches a minimum of 7.04e-4 at 362 rings, and climbs monotonically back toward
it.

## Conclusions

**The 2.208e-3 residual is converged in both refinable directions and is not
caused by the quadrature weight.** It is what the correctly-quadratured
reconstruction of a real traced singlet measures against an aberration-free
paraxial Airy pattern.

**The uniform arm's 9.21e-4 was never a competing converged value.** It is a
point on a transient dip on the way back up. The pre-CHE-47 code was not more
accurate; it was less converged, in a direction that happened to cancel.

**The negative control is mis-specified, not backwards.** It compares two arms at
one ray count with neither required to be converged, so its verdict is a function
of that ray count: 10.7 at 8 rings, 1.79 at 128, 1.02 at 181, 0.42 at 512, 0.69
at 1024. A control whose sign flips with a numerical parameter is measuring where
two convergence curves cross. The fix is to respecify it to require convergence
of both arms before comparing them -- a change to the control, not to the 1.2
floor, which is unchanged.

## What part 1 did not establish

That 2.208e-3 is *correct*. O1 is aberration-free and paraxial; `M3-SINGLET-REF`
is a real traced singlet, so some residual is expected from the system's own
aberration and part 1 did not separate that term. Part 2, below, separates it.

---

# Part 2 -- can O1 decide this at all? No, and by 4.4x

`benchmarks/probes/o1_applicability.py` ->
`benchmarks/probes/records/o1_applicability.json`. 34 s, CPU, O1 on both sides of
every comparison. **O2 is not consulted and no second Optiland PSF route is
consulted** (PB7/CHE-58 F2: `FFTPSF` and `HuygensPSF` share one `Wavefront`/OPD
front end and are one oracle, not two).

## 4. What the gate metric actually resolves

`[2 J1(v)/v]^2` has exactly one free parameter, the Airy scale, and it enters the
gate metric linearly. Comparing O1 at `NA` against O1 at `NA (1 + eps)` over the
same 5-Airy-radius disc -- no coupler, no traced data anywhere in this experiment:

| fractional scale offset | rel-L2 | rel-L2 / offset |
| --- | --- | --- |
| 1e-5 | 1.53189e-5 | 1.5319 |
| 1e-4 | 1.53176e-4 | 1.5318 |
| 1e-3 | 1.53038e-3 | 1.5304 |
| 1e-2 | 1.51667e-2 | 1.5167 |

Linear to 1% over four decades. Bisecting for the offset that reads exactly
`1.0e-3`: **`6.532e-4`**. So the frozen gate is, precisely, the statement that
this system's Airy scale is known to 0.065%.

## 5. How well this system knows its own Airy scale. It does not

`M3-SINGLET-REF` admits two defensible declarations of the same image-space `NA`,
and both are properties of the system rather than inputs to the coupler:

| declaration | value |
| --- | --- |
| paraxial geometric, `a / sqrt(a^2 + R^2)` | 0.051566680929690 |
| largest traced transverse direction cosine (the frozen declaration) | 0.051716318272919 |

They differ by **`2.902e-3`** -- 0.29% -- because the marginal ray crosses the
axis **14.0 um** before the declared image plane. That is this singlet's residual
spherical aberration, and it reproduces CHE-38 section 3's independently measured
14.1 um from the pupil side. In metric units the span is **`4.445e-3`**:

> **4.4x the gate, and twice the entire observed residual.**

O1's own free parameter is less determined on this system than the threshold it
is being asked to decide. That is the answer to the question part 1 left open.

## 6. Where the observed residual sits inside that

Minimising the gate metric over O1's one free parameter, on the frozen 512-ring
reconstruction taken off `GraphExecutor` (the residual at the declared `NA` comes
back as `0.0022072391812867093`, bit-identical to the frozen number, so this is
the gate's own field):

| quantity | value |
| --- | --- |
| rel-L2 at the declared `NA` | 2.2072391812867093e-3 |
| best-fit `NA` | 0.051645733465397 |
| best-fit offset from declared | -1.3648e-3 |
| **rel-L2 at the best-fit `NA`** | **7.0208e-4** -- inside the gate |
| scale term, in quadrature | 2.0926e-3 -- **94.8%** |
| shape term | 7.0208e-4 -- 5.2% |

And the best-fit `NA` lands **inside** the 0.29% interval the system's own
geometry leaves open. So: 94.8% of the residual is a scale offset that O1's
paraxial aberration-free assumption cannot pin on this system, and the 5.2% O1
*can* speak about is inside the gate.

**This is not a claim that the gate is met.** Fitting the oracle's scale to the
field under test removes the independence that makes O1 the only admissible
decider here -- a scale-fitted Airy pattern cannot reject a wrong answer of the
same shape. The best-fit number is the oracle's resolving power, measured. The
gate stays at the declared `NA` and stays unmet.

## 7. M0.2's amplitude drift is not in this number

CHE-103 attributed a ~20-order-of-magnitude absolute-power drift in three
committed records to CHE-47's amplitude change -- `sqrt(intensity) *
quadrature_weight_m2` -- which is the *same code change* as the quadrature weight
part 1 investigated. So the two could have been one problem. They are not, and
the reason is checkable rather than arguable.

The drift is the **global** per-ray cell-area factor, and the gate metric
peak-normalizes both of its inputs, so a global amplitude factor divides out
exactly. Rescaling the measured intensity by `2^64` (exact in binary), by `1e20`
(the order of the drift) and by the recorded uniform/weighted power ratio itself
moves the residual by at most `1.0e-14` relative -- the float64 round-off floor.

That the 24-order power gap between the two arms *is* that global factor is also
arithmetic on the committed record: `sqrt(P_weighted / P_uniform) = 2.4813e-13
m^2` against the nominal hexapolar cell `pi a^2 / (3 n^2) = 2.4923e-13 m^2` at
512 rings -- agreement to **0.44%**, and the 0.44% left over is the two boundary
corrections. Which is the non-global part, the only part that can reach this
metric, and precisely the part section 3 showed vanishes on convergence.

**Same root change, different root cause.** The drift cannot contribute to the
2.207e-3.

## 8. The negative control: retired and replaced, not widened

Part 1 called the control mis-specified and proposed requiring both arms to be
converged. That does not rescue it, and the reason is part 1's own finding: if
both arms converge to the same residual, the converged improvement factor is
**1.0**, so the premise -- *adding the weight improves rel-L2 agreement with O1
by at least 1.2x* -- is **false at convergence**. No ray count makes that control
fire honestly, and a floor cannot fix a false premise.

What CHE-47 actually established is **absolute-power convergence**: with the
per-ray area weight, reconstructed power is invariant under ray refinement;
without it, power scales as `(traced rays)^1.995` (CHE-33's `N^2.0024`). Measured
through `GraphExecutor` as `HandoffPerturbation(apply_quadrature_weight=False)`,
so the broken arm is the shipping path with one declared term removed:

| arm | P(64 rings) | P(128 rings) | ratio | `abs(ratio - 1)` |
| --- | --- | --- | --- | --- |
| production (weighted) | 1.351771e-24 | 1.352817e-24 | 1.000774 | **7.743e-4** |
| `apply_quadrature_weight=False` | 5.442243e-3 | 8.636231e-2 | 15.8689 | **14.869** |

Detection margin **1.92e4**. Both coarse values reproduce
`m3_quadrature_weight.json`'s `absolute_power` block bit-for-bit, which is an
incidental confirmation that the committed ladder still describes this tree.

So the family now declares `uniform-weight-power-divergence` on a new metric,
`reconstructed_power_ray_doubling_excess`, with a threshold of `0.05` -- about 7x
above the converged arm's own drift (the committed ladder's relative spread from
1801 rays up is 7.16e-3) and 297x below the divergence it must reject. It has no
oracle, and that is the point: a convergence property of one quantity under
refinement of its own discretization can fire on a system whose O1 gate cannot be
decided at all.

The retired 1.2 floor is **unchanged, not reworded and not deleted**. It stays in
`tolerances.yaml`, and `verification.claim_ledger` keeps its row `NOT_MET` with
the retirement recorded, because a claim whose premise the evidence falsified is
a different thing from a threshold somebody relaxed.

`negative_controls_pass` on `B3-PSF-SINGLET-01` is still `false`, now for an
honestly separate reason: `axis-transpose` and `launch-phase-error` are identity
operations at this on-axis instance, are declared `NOT_IMPLEMENTED` rather than
dropped, and `result.negative_controls_pass` correctly requires every declared
control to have fired. Exercising those two needs an off-axis instance.

## Gate disposition

**ATTRIBUTED AND UNMET.** The third of CHE-117's three admissible outcomes. The
1.0e-3 threshold is not widened, 2.2072391812867093e-3 does not meet it, and
every term of it now has a name and a number:

| term | value | share |
| --- | --- | --- |
| sensor sampling | 0 (ten significant figures over 8x refinement) | 0% |
| the quadrature weight | 0 at convergence | 0% |
| the central ray's 3/4 correction | 0.26% of the arm difference | negligible |
| O1's Airy-scale freedom, which this system leaves open to 0.29% | 2.0926e-3 | **94.8%** |
| what remains at O1's best fit | 7.0208e-4 -- inside the gate | 5.2% |

What a met gate on this chain would require is a configuration where O1's
assumptions hold -- CHE-38's synthetic aberration-free bundle reaches `4.07e-4`
-- not a wider tolerance, not O2, and not a second Optiland route.

## Artifacts

* `benchmarks/probes/singlet_residual_grid.py` -> `benchmarks/probes/records/singlet_residual_grid.json`
* `benchmarks/probes/singlet_residual_attribution.py` -> `benchmarks/probes/records/singlet_residual_attribution.json`
* `benchmarks/probes/o1_applicability.py` -> `benchmarks/probes/records/o1_applicability.json`

All three carry `record_provenance`, and all three are fresh against this tree.
Part 1's two records were regenerated during part 2, because the M2/M3 merge moved
`src/couplers/{handoff,node,ray_to_wave}.py` under them: **every physics value came
back identical**, with only the provenance hashes and wall times differing, so the
attribution above is unchanged by that merge and the frozen gate number still
comes off `GraphExecutor` bit-identically. Runtime: 5 min 58 s, 7 min 20 s and 34 s,
CPU, float64/numpy -- deliberately not the shipping complex64 jax path, so that
no part of the residual can be charged to a float32 cast that is not in the
number.
