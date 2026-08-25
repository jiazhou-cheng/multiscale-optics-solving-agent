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

## What this does not establish

That 2.208e-3 is *correct*. O1 is aberration-free and paraxial; `M3-SINGLET-REF`
is a real traced singlet, so some residual is expected from the system's own
aberration and this work does not separate that term. Doing so needs an oracle
this family does not have, and O2 does not qualify.

## Gate disposition

**Unchanged and unmet.** The 1.0e-3 threshold is not widened and 2.208e-3 does
not meet it. What changed is the shape of the open question: not *why does the
correct weight make agreement worse* -- answered, it does not -- but *can an
aberration-free paraxial oracle decide a real traced singlet at the 1e-3 level at
all*.

## Artifacts

* `benchmarks/probes/singlet_residual_grid.py` -> `benchmarks/probes/records/singlet_residual_grid.json`
* `benchmarks/probes/singlet_residual_attribution.py` -> `benchmarks/probes/records/singlet_residual_attribution.json`

Both records carry `record_provenance`. Runtime: 6 min 1 s and 7 min 13 s, CPU,
float64/numpy -- deliberately not the shipping complex64 jax path, so that no
part of the residual can be charged to a float32 cast that is not in the number.
