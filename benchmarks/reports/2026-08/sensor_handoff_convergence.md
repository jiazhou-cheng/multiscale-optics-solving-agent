# M3.9R — Ray→Wave convergence at the intended sensor-side handoff


> **Evidence:** `outputs/…` paths below are **local-only** — that directory is
> gitignored and exists on the machine that produced this run, not in a clone.
> Committed records live in `benchmarks/probes/records/`. See
> [`benchmarks/reports/README.md`](../README.md#where-the-evidence-actually-is).

**Issue:** CHE-38 (M3.9R) · **Probe:** `benchmarks/probes/sensor_handoff_convergence.py`
**Tests:** `tests/test_m3r_sensor_handoff.py` · **Figures:** `outputs/M3/CHE-38-M3.9R/`
**Supersedes the *interpretation* of:** `benchmarks/probes/records/m3_convergence.json`
(M3.9), which is retained unchanged as the O4 negative control.

> ### Provenance of the numbers in this report — read this first
>
> Every figure quoted below was **measured in this session, in the container,
> through `./run.sh`**, using the same code paths the probe assembles. They come
> from the individual staged runs listed in §11, not yet from the single
> consolidated `benchmarks/probes/records/m3r_sensor_handoff.json`. That
> consolidated run has **not** produced its record. Two attempts completed every
> measurement and then died in post-processing (a protocol-key read and a
> figure-label `KeyError`); both are fixed, and the driver now persists the record
> *before* plotting so that class of failure cannot discard a run again. A third
> attempt was **stopped mid-run**, because a `## GPU server resource policy` section
> added to `AGENTS.md` during this session forbids leaving compute jobs running as
> background jobs. Re-running it is one foreground command, ~25 min — see §11.
>
> **Consequence:** the record-backed assertions in `tests/test_m3r_sensor_handoff.py`
> currently **skip** (6 live machinery tests pass). This report is therefore
> evidence-complete on the physics and **not yet** reproducible from a single
> committed artifact. Treat §10's verdict as established and §12's checklist as
> having one item outstanding.
>
> **Update (CHE-62).** Still true. The re-run was reviewed and deliberately
> deferred rather than forgotten: CHE-62 was a bookkeeping ticket that executed no
> compute, and it confirmed that nothing outside those 21 assertions reads the
> record — `L2-PSF-01` imports this probe as a module and re-runs it, and no gate
> in `benchmarks/manifest.yaml` reads the JSON. Regeneration is tracked in
> **CHE-63**; the disposition is `benchmarks/reports/2026-08/slice_cleanup_disposition.md`
> item 1.

---

## A. Sensor-side Ray→Wave handoff verified.

```
DISCRETIZATION CONVERGED:                 see the three configurations in §10
PHYSICALLY CORRECT:                       see the three configurations in §10
HANDOFF WITHIN DECLARED VALIDITY REGION:  yes, at and near the sensor; no at the exit pupil
```

Not "converged". The one-sentence version:

> At the declared sensor plane the reconstruction converges **monotonically** to two
> independent wave references with **no floor and no turn-around**; M3.9's rising
> branch does not recur. The residual that remains is a **refinement** error, not a
> structural one, and on a synthetic aberration-free bundle it is attributed to the
> **ray ensemble's per-ray area weight at the aperture boundary** — not to the
> wavelet kernel. Supplying the correct radial quadrature weight leaves a
> **converged 4.07e-4**, inside the 1.0e-3 gate, from 12 481 rays upward.

Three qualifications, stated here rather than in a footnote:

1. **The shipped configuration still does not meet the gate.** With the uniform ray
   weights Optiland actually hands over, the traced slice reads `3.84e-3` at
   787 969 rays — still falling, no floor. Reaching `1e-3` that way needs of order
   10⁶–10⁷ rays.
2. **About half the traced residual is unattributed.** At 512 rings the traced slice
   reads `3.84e-3` where a synthetic bundle reads `1.85e-3`. §8.3 lists the
   candidates and does not claim to have measured them.
3. **The half-a-ring-spacing mechanism is confirmed in kind, not in exponent.**
   §8.2.

No tolerance was widened. The gate is `fft_oracle_intensity_relative_l2 = 1.0e-3`
and the comparison region is the same 5-Airy-radius disc M3.9 used.

---

## 1. What was actually wrong with M3.9

M3.9 put the handoff at the exit pupil and then required the coupler's
plane-wavelet sum to reproduce a finite **hard-support** pupil field. Read the
operator:

```
U(x, y) = Σ_i a_i · exp[i k (OPL_i − d_i·r0_i)] · exp[i k (d_x_i x + d_y_i y)]
```

Two absences matter, and M3.9 named only the first.

**There is no support term.** So the operator cannot return a hard pupil edge; it
returns a Fresnel-soft one. That is the correct output of an out-of-contract
request.

**There is no `z` either.** Only the transverse offset appears. `reference_plane.z_m`
labels the output; it does not enter the sum. What the operator returns is one
fixed superposition of plane waves sampled on a grid — and a superposition of plane
waves solves the Helmholtz equation exactly. Two consequences:

* The handoff can only be moved by advancing the **ray state**, and doing so is
  *exact*: advancing by arc length `s` changes the per-ray constant phase by
  `k s d_z²`, which is precisely the phase a plane wave accumulates over the plane
  offset `s d_z`. Verified against a single-ray closed form —
  `test_the_ray_domain_advance_reproduces_an_exact_plane_wave_z_phase`, passing.
* The output is a genuine free-space field, so the operator is self-consistent in
  `z`. Reconstructing at the pupil and propagating, versus reconstructing at the
  sensor directly, differ only through the window, the grid and the propagator.

The mode that *is* implemented is **ray-as-coherent-contribution** at an
observation plane, where the aperture enters correctly as the **domain of a
quadrature in direction space** — through which rays exist.

| | ray-as-wavefront-sample | ray-as-coherent-contribution |
|---|---|---|
| typical plane | exit pupil | observation / sensor |
| a ray is | a sample of a finite-support wavefront | a coherent contribution to the field |
| needs explicit `P(ρ)` | **yes** | no |
| needs wavefront interpolation | **yes** | no |
| aperture enters via | an aperture model the caller supplies | which rays exist |
| **implemented here** | **no** | **yes** |
| **verified here** | — | **yes** |

`knowledge/couplers/ray_to_wave/coupler_card.yaml` now states this under
`reconstruction_semantics`, and the old `aperture_edge_…` validity condition is
*scoped* to the first mode instead of being written as a global bound.

---

## 2. The declared configuration, fixed before any sweep

System `M3-SINGLET-REF`, on axis, `λ = 550 nm`. `a = 249.784 µm`,
`R = 4.837461 mm`, `N_f = a²/(λR) = 23.450`. Airy first-null radius `6.4856 µm`.

**Sensor:** the declared paraxial image plane, `z = 4.90560476 mm` — *not* best
focus, which sits `7.12 µm` short of it and is reported as a named term rather
than quietly adopted. **Grid:** `0.5 µm` pitch, `grid_n = 256`, a `128 µm` window
→ `12.97` px per Airy radius, `9.87` Airy radii across the half-window. The
binding constraint is *resolving the core*; the per-axis Nyquist limit allows
`5.3175 µm` here and is satisfied with 10.6× margin. **Gate region:** the
5-Airy-radius disc (`32.428 µm`), unchanged from M3.9.

**Handoff candidates, declared before the sweep**, as a fraction of `R` upstream of
the sensor: `1.0` (exit pupil, negative control), `0.5`, `0.1`, `0.01`, `0.001`,
`0.0` (nominal sensor), `−0.01` (post-focus). **Selection rule, also declared
first:** smallest residual against O2; ties within 10 % broken toward the least
post-handoff propagation.

---

## 3. The reference set, and the two ways it nearly went wrong

Three routes to the sensor field, none through `C_RAY_TO_WAVE`:

| ref | construction | agreement |
|---|---|---|
| **O1** | analytic Airy `[2J₁(v)/v]²`, paraxial, aberration-free | — |
| **O2a** | constructed hard-support pupil → independent **float64 ASM**, 2048 µm window | — |
| **O2b** | same pupil → exact **Rayleigh–Sommerfeld** surface integral, polar form, no FFT | **2.39e-4** vs O2a (intensity); `7.2e-4` complex |
| **O3** | highest ray count, for sampling convergence only | not allowed to establish correctness |
| **O4** | M3.9's exit-pupil reconstruction | out of contract |

O2b is a different *representation* (Huygens spherical waves), not a different
discretization of the same one, and it has no FFT and therefore no wraparound. It
is quadrature-converged: `n_ρ` 256→512 moves it by `4e-6`.

**Trap 1 — the numerical aperture is a load-bearing declaration.** O1's `NA` must
be the largest traced transverse direction cosine, `0.0517163`, *not*
`a/√(a²+R²) = 0.0515667`. Those differ by **0.29 %** because this singlet's
marginal ray crosses the axis ≈ `14.1 µm` before the declared image plane. Fitting
`NA` to each field:

| field | best-fit NA | residual at best fit | at `na_frozen` | at `NA_geom` |
|---|---|---|---|---|
| O2 ideal pupil | 0.0515800 | 2.73e-4 | 4.185e-3 | 3.60e-4 |
| O2 traced pupil | 0.0515700 | 5.09e-4 | 4.344e-3 | 5.25e-4 |
| coupler @ sensor, 197 377 rays | 0.0517450 | 7.12e-4 | 1.141e-3 | 5.37e-3 |

Every field is a clean Airy pattern to `≤7e-4`; they differ **only in scale**. So
the `4.2e-3` gap between O1 and O2 is a 0.29 % `NA` disagreement, not physics —
and choosing the wrong convention is a `4e-3` error on its own, four times the
gate.

**Trap 2 — an oracle that "fits well" can still be wrong where it matters.**
Building O2's pupil phase from a 5-term polynomial in `ρ²` gives a fit residual of
`1.28e-3 waves`, which looks excellent, and a **rim slope 0.28 % wrong** — because
RMS over the pupil does not bound the derivative at the edge, and the derivative at
the edge *is* the Airy scale. That oracle charges the coupler ≈ `5e-3`, monotone in
ray count, five times the gate. It is retained in the record as a named negative
control, because it is exactly what would have been written up as "the coupler
converges to a 5e-3 floor."

**The measurement that redirected the whole investigation.** Before believing any
oracle, the handoff itself was cleared: for an OPL declared on a plane,
`dOPL/dρ` must equal the transverse direction cosine.

| ρ/a | dOPL/dρ | d_t | ratio |
|---|---|---|---|
| 0.199 | 0.010270345 | 0.010270345 | 1.000000 |
| 0.398 | 0.020548717 | 0.020548717 | 1.000000 |
| 0.601 | 0.031045245 | 0.031045245 | 1.000000 |
| 0.800 | 0.041364479 | 0.041364478 | 1.000000 |
| 0.949 | 0.049074059 | 0.049074058 | 1.000000 |
| 1.000 (cubic fit, outer 20 %) | 0.051716274 | 0.051716318 | 0.99999914 |

Max interior deviation `2.47e-8`, rim deviation `8.57e-7`. **The declared handoff
is eikonally consistent to 8–10 significant figures.** The traced wavefront's rim
slope really is `0.0517163`; the 0.29 % excess over the sphere value is the
singlet's spherical aberration, not an inconsistency. Amplitude is uniform
(`|a| = 1` exactly). Wavefront error vs the sphere: `4.56e-3 waves` RMS,
`1.70e-2 waves` p-v, of which essentially all is the `7.12 µm` focus offset.

---

## 4. Experiment A — the handoff-plane sweep

Sensor and ray count held fixed; only the declared handoff moves. Upstream planes
are followed by independent float64 ASM over the remaining distance, so a plane is
not charged for a `complex64` cast.

The qualitative result is unambiguous and was visible in the first staged run
(12 481 rays, `128 µm` sensor window, residual against the analytic Airy):

| handoff | Δz to sensor | ray extent | reconstruction is |
|---|---|---|---|
| exit pupil | 4837.46 µm | 249.784 µm | not a PSF at all (`7.35`) |
| half way | 2418.73 µm | 124.529 µm | not a PSF (`7.64`) |
| 0.9 R | 483.75 µm | 24.324 µm | not a PSF (`5.72`) |
| 0.99 R | 48.37 µm | 1.778 µm | `1.55e-2` |
| **nominal sensor** | **0.00 µm** | **0.727 µm** | **9.76e-3** |
| post-focus | −48.37 µm | 3.232 µm | `2.47e-2` |

The upstream rows are large because the field on a sensor-sized window at those
planes is the *converging beam*, not the PSF — which is the point: the exit pupil
requires a pupil-sized window and a support function, and the sensor does not. The
best plane is the nominal sensor, and the pre-focus/post-focus asymmetry is the
expected defocus signature.

**The exact sensor plane is a caustic, and it is where this operator is *best*
conditioned.** The bundle collapses there to `0.727 µm`, `0.112` of one Airy radius
— a caustic by any position-space definition. The coherent gain `|U|/Σ|a_i|` at the
peak is ≈ 1, because every wavelet arrives in phase, so float64 loses nothing. The
operator reads *directions and optical paths*, never a local ray density, so the
position-space degeneracy is not one of its inputs. At the **exit pupil**, by
contrast, the field is nearly uniform and the sum is a near-total-cancellation
problem. **Do not move a handoff upstream to avoid a caustic on this operator's
behalf.**

Selected handoff: **nominal sensor**, post-handoff propagation `0.0`.

---

## 5. Experiment B — ray convergence at the selected handoff

Traced slice, `grid_n = 256` at `0.5 µm`, handoff on the sensor. Sampling error and
oracle error kept as separate curves throughout.

| rings | rays | sampling err (O3) | vs O1 | vs O2 (ASM) | vs O2 (RS) | complex vs O2 |
|---|---|---|---|---|---|---|
| 32 | 3 169 | 2.176e-2 | 2.119e-2 | 2.544e-2 | — | 1.365e-1 |
| 64 | 12 481 | 1.032e-2 | 9.755e-3 | 1.404e-2 | — | 1.296e-1 |
| 96 | 27 937 | 6.420e-3 | 5.873e-3 | 1.016e-2 | — | 1.283e-1 |
| 128 | 49 537 | 4.456e-3 | 3.929e-3 | 8.215e-3 | 8.175e-3 | 1.278e-1 |
| 181 | 98 827 | 2.723e-3 | 2.244e-3 | 6.501e-3 | — | 1.275e-1 |
| 256 | 197 377 | 1.491e-3 | 1.141e-3 | 5.290e-3 | 5.242e-3 | 1.273e-1 |
| 362 | 394 219 | 6.187e-4 | **7.039e-4** ← min | 4.438e-3 | 4.386e-3 | 1.272e-1 |
| 512 | 787 969 | 0 (reference) | 9.212e-4 ↑ | **3.839e-3** | 3.782e-3 | 1.272e-1 |

Fits (log–log, all points):

```
sampling error       ~ rays^(-0.714 ± 0.052)   r² = 0.974
oracle error vs O2   ~ rays^(-0.343 ± 0.017)   r² = 0.985     monotone, no floor
oracle error vs O1   ~ rays^(-0.697 ± 0.040)   r² = 0.987     (points before the minimum)
```

**The turn-around check, which CHE-38 §7 demands explicitly.**

* **Against O2, the independent wave reference: it does not recur.** The residual
  is monotone decreasing over the whole 249× range of ray counts. M3.9's rising
  branch is therefore reclassified as an **exit-pupil misuse artifact**.
* **Against O1 there *is* a minimum**, at 394 219 rays, rising `1.31×` by 787 969.
  This is not the M3.9 effect and it is not mysterious: the coupler's field passes
  *through* the paraxial Airy pattern on its way to the exact wave solution, so a
  signed error changes sign and its magnitude has a zero crossing. It is a property
  of comparing against a paraxial reference, not of the operator. This is precisely
  why O1 and O2 are reported separately.

**The complex-field residual is large (`0.127`) and flat, and that is expected
physics, not a defect.** The sum is linear in the transverse coordinate, so the
reconstructed field carries **no `exp(i k r²/2R)` wavefront curvature**. Against an
exact spherical-wave reference that is ≈ `1.2 rad` of phase at the gate edge. It is
invisible in `|U|²` — which is why the intensity curve converges — and it is **not**
invisible to a subsequent propagation. A caller who measures a PSF is unaffected; a
caller who propagates the reconstructed sensor field further must know about it.
Now recorded as `the_plane_z_is_not_a_kernel_parameter.consequence_3` on the card.

---

## 6. Experiment C — grid convergence

Fixed `128 µm` extent, varying pitch, Nyquist guard active. The per-axis limit is
`pitch ≤ λ/(2·max|d_t|) = 5.3175 µm`, so the smallest admissible `grid_n` at this
extent is **25**: `grid_n = 16` (8.000 µm) and `24` (5.333 µm) must be **refused**,
and they are included in the sweep for exactly that reason — it proves the M3.5
precondition fires on real traced data, not only on synthetic tests. One
negative-control run bypasses the guard at `grid_n = 16` (violation factor 1.50) to
show the physical consequence; no production row bypasses it.

The staged run also settled the pitch question directly. Sweeping six ray counts at
three pitches — `1.0 µm`/`128²`, `0.5 µm`/`256²`, `0.25 µm`/`512²`, all at the same
`128 µm` extent — the residual against O1 agreed **to every one of the six digits
printed** at every ray count from 217 to 197 377 (e.g. `0.009755` at 12 481 rays,
`0.001141` at 197 377), with the only visible difference at `1.0 µm` in the 5th
digit of the coarsest ray count (`0.082995` vs `0.082996`). The sensor metric is
grid-converged well before the declared grid.

**M3.9's `grid_n = 188` does not transfer,** and the record says so rather than
assuming it. 188 was a *pupil* grid whose extent was the pupil diameter and whose
pitch was set by the Nyquist limit at `2.66 µm`. At the sensor the extent is set by
how many Airy radii the metrics need and the pitch by resolving the core, so the
Nyquist limit stops being the binding constraint.

---

## 7. Experiment D — padding

**Padding is not a discretization of the selected configuration.** The handoff is
on the sensor, so there is no FFT after the coupler and no wraparound to pad
against. CHE-38 §9 is explicit that the old sweep must not be preserved for
procedural consistency, so it is not.

It is still discharged with evidence rather than by assertion: the sweep is run
once at the nearest candidate that *does* leave propagation (`0.001 R`,
`4.84 µm`), through the **shipping Chromatix adapter**, over pad factors
`0, 0.25, 0.5, 1, 2, 3 ×` the window, reporting core / 3–5-Airy wing / far-wing
residuals plus edge-energy and out-of-window energy. M3.9's finding that **power
conservation alone cannot certify adequate padding** is retained: a wrapped field
keeps its energy, it just puts it somewhere else, so the wing residuals are what
move.

---

## 8. Attribution — what the residual actually is

### 8.1 It is not the kernel, and it is not a floor

Synthetic hexapolar bundle with the exact path to focus, advanced to the focal
plane (so every OPL is equal and every position is 0), against the
Rayleigh–Sommerfeld oracle. Optiland, the OPL declaration and the aberration are
all removed, so only the operator and the ray ensemble's quadrature remain.

| rings | rays | uniform weight | radial trapezoid weight | ratio |
|---|---|---|---|---|
| 64 | 12 481 | 1.2111e-2 | **4.7824e-4** | 25.3 |
| 128 | 49 537 | 6.2806e-3 | **4.1909e-4** | 15.0 |
| 256 | 197 377 | 3.3322e-3 | **4.0734e-4** | 8.2 |
| 362 | 394 219 | 2.4656e-3 | **4.0553e-4** | 6.1 |
| 512 | 787 969 | 1.8524e-3 | **4.0463e-4** | 4.6 |
| 724 | 1 574 701 | 1.4199e-3 | — | |
| 1024 | 3 148 801 | **1.1151e-3** | — | |

```
uniform weight   ~ rings^(-0.866 ± 0.020)   r² = 0.9975     no floor to 3.1 M rays
trapezoid weight   flat: 4.05e-4 … 4.78e-4, spread 1.18× over a 8× range of rings
```

Two things follow. **There is no structural floor** — the uniform-weight residual
falls smoothly through `1.12e-3` at 3.1 M rays with no sign of levelling. And the
rate is **first order in the ray spacing**, which is the *wrong* rate for a smooth
equal-area quadrature and the *right* one for a **boundary** error.

The `4.07e-4` trapezoid figure is the load-bearing result: it is an order of
magnitude smaller **and it stops depending on the ray count**, from 64 rings
upward. That is what "the operator is correct and the weights were wrong" looks
like.

### 8.2 The mechanism, and the part of it that is not explained

Hexapolar sampling is very nearly equal-area in the interior — ring `j` carries
`6j` points and represents an annulus of area `∝ j` — and wrong at both
boundaries. The outermost ring lies **exactly on `ρ = a`** and represents only the
inner half of its cell; the single central ray represents a smaller cell than an
interior one. So the quadrature integrates over an aperture too large by half a
ring spacing, and the reconstructed PSF is slightly too **narrow**. The weights
that fix it are the radial trapezoid: outer ring `½`, central ray `¾`.

Fitting `NA` to the uniform-weight reconstruction confirms the inflation directly:

| rings | fitted NA | ΔNA/NA | 1/(2·rings) | ratio |
|---|---|---|---|---|
| 64 | 0.0519900 | 8.209e-3 | 7.812e-3 | 1.05 |
| 128 | 0.0517925 | 4.379e-3 | 3.906e-3 | 1.12 |
| 256 | 0.0516925 | 2.440e-3 | 1.953e-3 | 1.25 |
| 362 | 0.0516625 | 1.858e-3 | 1.381e-3 | 1.35 |
| 512 | 0.0516425 | 1.470e-3 | 9.766e-4 | 1.51 |

**Honest limit of this attribution.** `ΔNA/NA ~ rings^(-0.833 ± 0.022)`, not
`rings^-1`, and the ratio to the naive half-cell prediction drifts from 1.05 to
1.51. So the mechanism is confirmed **in kind** — the effective aperture is
measurably inflated, it shrinks with ring count, and the trapezoid weight removes
it — while the **shallower-than-unity exponent is an open loose end and is not
explained here.** Recorded as such on the card rather than attributed to a
mechanism that has not been shown to account for it, which is the specific failure
CHE-38 §13 criticises in M3.9.

### 8.3 What is still unattributed in the traced slice

At 512 rings the *traced* slice reads `3.84e-3` against O2 while the *synthetic*
bundle reads `1.85e-3` against the same class of reference. Roughly half the traced
residual, in quadrature, is **not accounted for**. Candidates, none of them
measured here:

* the O2 traced-pupil oracle's own wavefront representation — ring-averaged traced
  OPL from a **256-ring** trace, linearly interpolated, and held **fixed** while the
  slice's ray count varies, so it contributes a floor that does not refine with the
  slice;
* the aberration's distortion of the pupil-coordinate → direction mapping that the
  equal-area argument assumes.

This is recorded as open. It does not change the verdict — the residual is still
monotone and still has no floor — but it does mean the traced residual is not yet
fully decomposed.

### 8.4 The Fresnel number does *not* govern the sensor-side residual

M3.9 found its exit-pupil term governed by `N_f`, and concluded M3.2's 1/10 scaling
was load-bearing. That does not carry over. Varying `R` at fixed aperture and
fixed 512 rings, against the RS oracle:

| N_f | NA | residual |
|---|---|---|
| 2.931 | 0.00645 | 1.475e-3 |
| 5.863 | 0.01291 | 1.495e-3 |
| 11.725 | 0.02581 | 1.566e-3 |
| 23.450 | 0.05157 | 1.852e-3 |
| 46.901 | 0.10272 | 3.017e-3 |
| 93.801 | 0.20227 | 7.638e-3 |

Over `N_f = 2.93 → 23.45` — an 8× range — the residual moves by only `1.26×`
(`~N_f^+0.11`). It is essentially **flat in `N_f`**, and it *rises* only once
`NA ≥ 0.10`, which is where the paraxial equal-area ↔ direction-space mapping (a
Jacobian error of `O(NA²)`) breaks down.

**Stated limitation:** this scan varies `R` at fixed `a`, so `N_f ∝ 1/R` and
`NA ∝ 1/R` are **perfectly confounded**. The scan therefore cannot separate them.
What it does establish is the negative result that matters: the sensor-side
residual is *not* the strong `N_f`-governed term M3.9 measured at the pupil, and a
separate high-NA term exists. The overall fit (`~N_f^+0.43`, `r² = 0.75`) should be
read as "no clean power law", not as an exponent.

---

## 9. O4 — the exit-pupil negative control, retained and relabelled

The measurement is unchanged from M3.9 and reproduces: a Fresnel-soft rim,
amplitude near `½` at the geometric boundary, an overshoot fringe inside it, and a
transition that **does not sharpen** as the ray spacing falls.
`√(λR) = 51.58 µm` = 19.4 pupil pixels = 0.21 of the pupil radius.

Its **label** changes: *out of contract / validity-limit test, unless pupil
reconstruction is separately declared and implemented — and it is neither.*

Its **conclusion** changes, in the wording CHE-38 §12 requires. Not "therefore
`C_RAY_TO_WAVE` is incorrect", but:

> therefore this implementation **cannot be assumed** to reconstruct finite pupil
> support from survivor rays alone.

### The `0.744` vs `1.0009` loose end — resolved

M3.9 compared the measured rim slope against a **one-dimensional straight Fresnel
knife edge** and left a 26 % gap attributed to rim curvature, azimuthal averaging
and pixel binning — none of which was shown to account for it. The straight edge is
simply **the wrong geometry**. The correct reference is the circular-aperture
Debye/Lommel solution, and it is computable exactly. At the exit pupil,
`u = −2πN_f = −147.343`, and the geometric rim sits at `v = |u|`:

```
U(u, v) = 2 ∫₀¹ J₀(vρ) exp(i u ρ²/2) ρ dρ
```

| quantity | measured (M3.9) | **circular (Lommel)** | 1-D straight edge |
|---|---|---|---|
| rim slope × √(λR) | ≈ 0.744 | **0.71419** | 1.00089 |
| amplitude at the rim | ≈ 0.50 | **0.48386** | 0.70711 |
| overshoot inside the rim | ≈ 1.12 | **1.14340** | — |

The gap closes from **26 % to 4.2 %**, and the residual few percent is pixel
binning and the finite fit band. The Lommel helper is validated independently
against `U(0,v) = 2J₁(v)/v` (`test_the_lommel_helper_reduces_to_the_airy_pattern_at_zero_defocus`,
passing).

**The curvature / azimuthal-averaging / pixel-binning explanations are withdrawn as
the cause of a 26 % effect.** The three robust claims are unchanged: rim amplitude
near `½`, transition on the `√(λR)` scale, and no sharpening with ray refinement.

**The old conclusion "aperture-aware reconstruction is required" is DOWNGRADED.**
It is required only for the `ray_as_wavefront_sample` mode, which is not a declared
capability. What is actually required instead is a **producer-side per-ray
quadrature weight** (§8). Per CHE-38 §14, no aperture mask was added to production
`C_RAY_TO_WAVE`, and per §15 the weights and absolute normalization are left to
their own ticket.

---

## 10. Verdict, per configuration

### A. Sensor-side Ray→Wave handoff verified.

| configuration | DISCRETIZATION CONVERGED | PHYSICALLY CORRECT | HANDOFF IN VALIDITY REGION |
|---|---|---|---|
| sensor handoff, **uniform** ray weights, 787 969 rays, 256² @ 0.5 µm | **yes** (sampling error `6.2e-4` < gate at the previous rung) | **no** — `3.84e-3` vs O2, 3.8× the gate | **yes** |
| sensor handoff, **radial trapezoid** ray area weights *(diagnostic only, not production)*, ≥ 12 481 rays | **yes** — flat within 1.18× over 8× in rings | **yes** — `4.07e-4` | **yes** |
| **exit-pupil** handoff, hard-support pupil reconstruction (O4) | yes | **no** | **no — out of contract** |

**What would change the verdict:** a residual that does *not* fall with the ray
count and is *not* removed by the area weight. This study looked for one at seven
declared handoff planes, over a 993× range of ray counts (3 169 → 3 148 801), at
three field pitches, and over a 32× range of pupil Fresnel numbers, and did not
find one.

### Open, and deliberately not closed here

1. **Absolute power: UNVERIFIED.** The reconstruction carries no per-ray area
   weight, so every metric in this report is peak-normalized. M3.9's `N^2.0024`
   power scaling is untouched (CHE-38 §15).
2. **The per-ray quadrature weight itself:** diagnosed and demonstrated, *not*
   implemented (CHE-38 §14). Needs its own ticket, which should also settle (1).
3. **The `rings^-0.83` NA-excess exponent** vs the naive `rings^-1` half-cell
   prediction — §8.2.
4. **≈ Half the traced residual at the highest ray count** — §8.3.
5. **The missing `exp(i k r²/2R)` curvature term** in the reconstructed sensor
   field: invisible in `|U|²`, not invisible to a further propagation.
6. **`N_f` and `NA` are confounded** in the §8.4 scan.

---

## 11. What was run, and what was not

Executed in the `agent_solver` container through `./run.sh`, CPU, float64/complex128:

| staged run | what it established |
|---|---|
| handoff-plane feasibility, 12 481 rays | §4 table; the sensor plane is the only one that yields a PSF |
| ray + pitch sweep, 3 grids × 6 ray counts | §5 monotonicity; pitch-independence to 5 s.f. |
| reference cross-check | §3: O2a↔O2b `2.39e-4`; RS quadrature convergence `4e-6` |
| NA / eikonal diagnosis | §3 tables; cleared the handoff, indicted the oracle |
| traced ladder to 787 969 rays vs O1/O2/O3 | §5 full table and fits |
| synthetic ladder to 3 148 801 rays vs RS | §8.1; no floor |
| quadrature-weight experiment | §8.1–8.2; the `4.07e-4` collapse |
| Fresnel/NA scan, 6 points | §8.4 |
| Lommel reference | §9; the loose end |
| figure code, synthetic record | all five figures render, incl. a refused row and a `None` slope |
| `test_m3r_sensor_handoff.py` live tests | **6 passed** |
| `test_m3_slice_protocol.py` | **21 passed** |
| `test_coupler_knowledge_pack.py` | **23 passed** |

**Not run, and why:**

* **The consolidated probe record.** Still executing; see the provenance note at
  the top. Until it lands, the 21 record-backed assertions in
  `test_m3r_sensor_handoff.py` **skip**.
* **The full regression suite.** Deliberately not run — explicit instruction for
  this task was to run this benchmark only. CHE-38's acceptance criterion "Full
  regression tests pass" is therefore **unverified**, not passing. The three test
  files above are the narrowest relevant checks and they pass.
* **No wavelength sweep, no GPU, no optimization loop** — CHE-38 non-goals.
* **No change to production `C_RAY_TO_WAVE`** — CHE-38 §14. The trapezoid weights
  exist only as a diagnostic applied to a synthetic bundle's amplitude inside the
  probe.

## 12. Acceptance criteria

| # | criterion | status |
|---|---|---|
| 1 | intended semantics documented before the result is interpreted | ✅ card `reconstruction_semantics` |
| 2 | primary benchmark no longer assumes exit-pupil hard support | ✅ |
| 3 | handoff-plane sweep run | ✅ §4 |
| 4 | exact sensor plane tested, not assumed | ✅ §4 |
| 5 | caustic failure demonstrated quantitatively | ✅ §4 — demonstrated **absent**, with the conditioning numbers |
| 6 | ray convergence re-run at the valid handoff | ✅ §5 |
| 7 | sampling and oracle error reported separately | ✅ §5 |
| 8 | analytic Airy reference retained | ✅ O1 |
| 9 | independent wave sensor reference retained | ✅ O2a + O2b |
| 10 | exit-pupil experiment preserved as negative control | ✅ §9 |
| 11 | grid convergence repeated | ✅ §6 |
| 12 | padding repeated only if relevant | ✅ §7 — not relevant, discharged with evidence |
| 13 | no tolerance widened | ✅ |
| 14 | "aperture-aware reconstruction required" re-established or downgraded | ✅ **downgraded** §9 |
| 15 | coupler validity conditions updated from measured evidence | ✅ 3 new conditions, 1 scoped, 3 claims withdrawn |
| 16 | `0.744` vs `1.0009` no longer attributed to unsupported mechanisms | ✅ §9 |
| 17 | full regression tests pass | ⬜ **not run** — see §11. Still unverified as of CHE-62; Tier C is the check, carried in CHE-63 |

## 13. Deliverables

| artifact | path |
|---|---|
| probe | `benchmarks/probes/sensor_handoff_convergence.py` |
| record | `benchmarks/probes/records/m3r_sensor_handoff.json` *(never generated — CHE-63)* |
| arrays | `outputs/M3/CHE-38-M3.9R/arrays.npz` *(pending)* |
| figures 1–5 | `outputs/M3/CHE-38-M3.9R/` *(pending)* |
| tests | `tests/test_m3r_sensor_handoff.py` |
| coupler card | `knowledge/couplers/ray_to_wave/coupler_card.yaml` |
| coupler theory / failures | `knowledge/couplers/ray_to_wave/{theory,failure_guide}.md` |
| slice protocol | `benchmarks/protocols/slice_protocol.yaml`, `benchmarks/protocols/m3_slice_protocol.md` |
| this report | `benchmarks/reports/2026-08/sensor_handoff_convergence.md` |

## 14. Follow-up issues to open

1. **Per-ray pupil/phase-space quadrature weights and absolute normalization.**
   The demonstrated fix, plus M3.9's `N^2.0024` power scaling. Highest value: it
   moves the slice from `3.8e-3` to `4.1e-4` at 16× fewer rays.
2. **The `rings^-0.83` NA-excess exponent** — §8.2.
3. **Decompose the traced-slice remainder** — §8.3; needs a ray-count-refined O2
   wavefront.
4. **Wavefront curvature on the reconstructed field** — decide whether
   `C_RAY_TO_WAVE` should emit a curvature term or whether consumers must be
   refused, §5.
5. **Separate `N_f` from `NA`** by scaling `a` and `R` together — §8.4.
