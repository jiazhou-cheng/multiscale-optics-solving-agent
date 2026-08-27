# C_RAY_TO_WAVE — theory

Source: Cheng et al., ACS Photonics 2026, DOI `10.1021/acsphotonics.6c00818`,
main text eq 2 and SI Figure S1. Equation labels below are the paper's.

## The physical claim

A geometric ray is not a curve carrying an intensity. It is a **plane wavelet**:
a local plane-wave segment with a propagation direction `d̂` and a phase set by
its accumulated optical path length (SI Figure S1b),

```
wavelet phase = exp(+i k · OPL),      k = 2π/λ
```

A field is therefore recoverable from a ray ensemble by coherent summation,
provided the ensemble samples the wavefront densely enough that the
plane-wavelet approximation holds locally. SI Figure S1a states the converse
that makes this consistent: an arbitrary wavefront decomposes into plane-wave
modes, which is exactly what the angular spectrum method does.

## The governing equation

For an ensemble of rays `i` arriving at a reconstruction plane with unit normal
`n̂`, the field at a point `(x, y)` on that plane is (main text eq 2)

```
U_plane(x, y) = Σ_i  a⁽ⁱ⁾ · exp[ i k ( OPL⁽ⁱ⁾ + Δr⁽ⁱ⁾(x, y) ) ] · ⟨ n̂ , d̂⁽ⁱ⁾ ⟩
```

with three factors that each mean something distinct, and each of which is a
separate way to get the physics wrong:

| Symbol | Meaning | Units | Failure if omitted |
|---|---|---|---|
| `a⁽ⁱ⁾` | complex amplitude carried by the ray | field amplitude (not intensity) | Using a ray *intensity* here silently squares the field |
| `OPL⁽ⁱ⁾` | optical path length accumulated to the ray's intersection with the plane | m | Wrong absolute phase; a constant piston is harmless, a *reference* error is not |
| `Δr⁽ⁱ⁾(x, y)` | additional path from the ray's intersection point on the plane to the field point `(x, y)`, along the wavelet direction | m | The wavelet stops being a tilted plane wave; oblique rays get a piston instead of a phase ramp |
| `⟨n̂, d̂⁽ⁱ⁾⟩` | projection (obliquity) factor between wavelet direction and plane normal | dimensionless | Oblique contributions are over-weighted; power is not conserved at large angles |
| `k = 2π/λ` | free-space wavenumber | rad/m | — |

### The `Δr` term, explicitly

A ray crossing the plane at `r₀⁽ⁱ⁾ = (x₀⁽ⁱ⁾, y₀⁽ⁱ⁾)` with direction
`d̂⁽ⁱ⁾ = (d_x, d_y, d_z)` is, near that point, the plane wave
`exp(+i k d̂·r)`. Evaluating it at a nearby in-plane point `(x, y)` gives the
extra path

```
Δr⁽ⁱ⁾(x, y) = d_x⁽ⁱ⁾ (x − x₀⁽ⁱ⁾) + d_y⁽ⁱ⁾ (y − y₀⁽ⁱ⁾)
```

so the ray contributes a **linear phase ramp** across the plane, not a point.
This is the same statement as the paper's rule of thumb for planar DOEs (main
text Figure 2b): the response of an off-centre incident ray is obtained by
applying a linear phase ramp to the centred response, by the Fourier shift
theorem.

### Why the ramp is what makes a collimated bundle a plane wave

SI Figure S1c: a collimated bundle launched from different lateral positions
represents a single angular-spectrum mode only if each ray's phase compensates
the OPL difference associated with its lateral position. Rays share a direction
but not a launch point, so without the ramp they do not sum to a plane wave.
This is the sharpest available test of the ray→wave direction, because the
oracle `exp(+i k·r)` is exact — see `probes/` and `conventions.md`.

## Normalization

SI eqs S3 and S5 carry an explicit `1/N` in front of the sum when the ensemble
is a Monte Carlo sample of a spectrum. Main text eq 2 has no `1/N` because it
sums a *given* ray ensemble. These are not in conflict; they answer different
questions:

- Reconstructing from a **given** ray set (a traced bundle): no `1/N`. The rays
  are the physical ensemble.
- Reconstructing from a **sampled** spectrum: `1/N`, because the sum is a Monte
  Carlo estimate of an integral.

This repository requires the normalization to be declared per call rather than
inferred, and reports discrete power on both sides of every transformation.

## Validity conditions

1. **Sampling density.** The ray ensemble must sample the wavefront finely
   enough that the phase difference between adjacent rays is `< π` at the
   reconstruction plane, or the reconstruction aliases. This is a Nyquist
   condition on the wavefront, not on the grid.
2. **Single-valued wavefront.** At a caustic, or where multiple ray branches
   overlap, the "local plane wavelet" picture holds per branch but the phase is
   multi-valued. Coherent summation still works — that is the point of summing
   complex amplitudes rather than unwrapping a phase — but any downstream step
   that *unwraps* phase is invalid there.
3. **Amplitude must be an amplitude.** Ray weights that represent power, energy,
   or hit counts are not `a⁽ⁱ⁾` and must be converted with a declared,
   tested map. This repository refuses the conversion by default; see
   `conventions.md`.
4. **Plane, not surface.** Eq 2 reconstructs onto a plane with a single normal
   `n̂`. A curved reconstruction surface needs the tangent-plane treatment and
   inherits the curvature bound of `../wave_to_ray/theory.md`.

5. **The aperture is the quadrature domain, not a mask.** The sum is a quadrature
   over the ray ensemble, and the only thing that tells it where the aperture is
   is *which rays exist*. It follows that (a) the operator cannot impose a hard
   pupil support, and (b) the per-ray **area weight** matters, because a
   quadrature with the wrong weights integrates over the wrong aperture. See
   "the effective aperture" below.

## What this direction does not do

It does not propagate. It reconstructs the field *at the plane the rays already
reached*. Propagation away from that plane is a wave-model operation
(`M_WAVE_CHROMATIX`), and the M1 evidence for `asm_propagate` is what makes that
handoff usable.

Note carefully what "the plane the rays already reached" means, because the
equation is more literal than it looks: **`z` does not appear in the kernel at
all**, only the transverse offset does. `reference_plane.z_m` labels the output;
it does not enter the sum. So the plane the field is reconstructed on is wherever
the *rays* are, and moving it means advancing the ray state — positions along each
direction, optical path by `n ×` arc length. CHE-38 (M3.9R) showed that this is
exact rather than approximate: advancing by arc length `s` changes the per-ray
constant phase by `k s d_z²`, which is exactly the phase a plane wave accumulates
over the plane offset `s d_z`. The corollary is that the output is a genuine
free-space field — a superposition of plane waves solves the Helmholtz equation —
so the operator is self-consistent in `z`. The corollary of *that* is point 5
above: the sum is linear in the transverse coordinate, so it carries no
`exp(i k r²/2R)` wavefront curvature. Invisible in `|U|²`; not invisible to a
subsequent propagation.

## Two semantics for "a ray at a plane", and only one is implemented

At an **exit pupil** a ray is naturally read as a *sample of a finite-support
wavefront*: to get a field you interpolate the samples and multiply by `P(ρ)`.
This operator does neither, and cannot — support is not one of its inputs.

At an **observation plane** a ray is a *coherent contribution to the measured
field*, and no support term is needed because the aperture is already encoded in
the quadrature domain. This is the mode eq 2 implements and the mode CHE-38
(M3.9R) verified. Asking the first question of an operator that answers the second
is what produced M3.9's Fresnel-soft pupil rim, and the correct reference for that
rim is the circular-aperture Debye/Lommel solution (`0.7142` in units of
`1/√(λR)`), not a one-dimensional straight knife edge (`1.0009`).

## The effective aperture, and why a per-ray area weight is not cosmetic

Hexapolar sampling is very nearly equal-area in the interior — ring `j` carries
`6j` points and represents an annulus of area `∝ j` — and wrong at both
boundaries. The outermost ring lies exactly on `ρ = a` and represents only the
inner half of its cell; the single central ray represents a smaller cell than an
interior one. With uniform weights the quadrature therefore integrates over an
aperture that is too large by half a ring spacing, so the reconstructed PSF is
slightly too *narrow*.

CHE-38 (M3.9R) measured this on a synthetic aberration-free bundle against a
Rayleigh–Sommerfeld reference. The residual falls as `ring_count^-0.87` — first
order in the ray spacing, which is the wrong rate for a smooth equal-area
quadrature and the right one for a boundary error — and the fitted effective-NA
excess tracks `1/(2 × rings)` over a 16× range of ring counts. Applying the radial
trapezoid weight (outer ring `½`, central ray `¾`) collapses the residual to a
*converged* `4.07e-4`, flat from 64 rings upward.

This is a **producer-side** obligation: the coupler cannot compute the weight
because it does not know where the aperture is. CHE-38 deliberately did not
implement it (§14) and assigned it to a follow-up ticket that must also settle
absolute normalization (§15).

**CHE-47 (M3.9R extension) implements it.** `couplers.
quadrature.hexapolar_area_weight_m2` computes the same radial-trapezoid weight
CHE-38 measured, scaled to an absolute area in m² (`π a² / (3 n²)` per interior
ray, `¾`/`½` of that at the center/outer-ring boundaries). The Optiland adapter
regenerates the hexapolar pupil sampling `Optic.trace` used and matches it row
for row against the traced set — the same technique CHE-41 already uses for the
off-axis object-space term — and exports it as `quadrature_weight_m2`.
`optiland_handoff.declare_coherent_bundle` folds it into the amplitude
declaration by default (`a_i = sqrt(intensity_i) · quadrature_weight_m2,i`)
whenever the adapter could confirm an un-vignetted hexapolar fan; `C_RAY_TO_WAVE`
itself is untouched.

Two results, measured rather than assumed:

* **Absolute power now converges.** The old mapping made discrete power grow as
  `(ray count)^2` (CHE-33/CHE-38's `N^2.0024`, reproduced exactly on the legacy
  path: fitted exponent `1.9948`, `r² = 0.9999`). With the area weight it is flat
  in ray count (`-0.0098`, `r² = 0.23` — noise, not a trend), because the sum now
  approximates a fixed integral over a fixed aperture instead of accumulating
  more equal-sized contributions as the ray count grows.
* **The sensor-plane residual improves but does not close on the real system.**
  CHE-38's `4.07e-4` was measured on a *synthetic, aberration-free* bundle. On the
  real (residually aberrated) `M3-SINGLET-REF` trace, the same weight improves the
  sensor residual against the independent wave oracle `1.58×` (`3.91e-3 → 2.48e-3`
  at 787 969 rays) but stays above the `1e-3` gate. The analytic Airy oracle
  (aberration-free, sharing no code with the trace) is *closer* to the weighted
  result (`2.21e-3`) than the aberration-matched wave oracle is (`2.48e-3`) —
  which would not happen if the leftover gap were a coupler-side
  aberration/quadrature defect, since the aberration-matched oracle should then
  track the coupler more closely, not less. The likelier explanation is the wave
  oracle's own ring-averaged, linearly-interpolated pupil-fit quality at this
  resolution; not decomposed further (`benchmarks/probes/quadrature_weight.py`).

  **CHE-117 (M4.2) decomposed it, and the answer did not need O2 at all.** The
  `2.207e-3` is converged — flat to 0.87% from 49,537 to 3,148,801 rays and
  identical to ten significant figures across an 8× sensor-pitch refinement — and
  it is *not* caused by the quadrature weight: the uniform arm converges to the
  same number from below after a transient dip through `7.04e-4`, so the
  `9.21e-4` that used to look like better agreement was a point on that dip. What
  the residual *is*: **94.8% of it, in quadrature, is an Airy-scale offset that
  O1's paraxial aberration-free assumption cannot pin on this system.** The gate
  metric is linear in fractional scale error (slope 1.52), so `1.0e-3` resolves
  the Airy scale to `6.53e-4`, while `M3-SINGLET-REF` admits two defensible
  image-space `NA` declarations — paraxial geometric `0.0515667` and largest
  traced direction cosine `0.0517163` — that differ by `2.902e-3` because the
  marginal ray focuses 14.0 µm short of the declared image plane. That span is
  `4.445e-3` of gate metric, 4.4× the gate. At O1's own best-fit `NA` (`0.0516457`,
  *inside* that interval) the residual is `7.021e-4`, inside the gate.

  **What that means for a caller:** on a real traced system with residual
  spherical aberration, an analytic Airy oracle is not a `1e-3`-level decider,
  because the quantity it is most sensitive to is the one the system leaves
  undetermined. It is *not* a licence to read the gate as met at a fitted `NA` —
  fitting the oracle's scale to the field under test removes the independence that
  made it admissible. Evidence: `benchmarks/probes/records/o1_applicability.json`,
  `benchmarks/reports/2026-08/singlet_residual_attribution.md`.
