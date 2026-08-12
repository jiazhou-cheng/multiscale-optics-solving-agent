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

## What this direction does not do

It does not propagate. It reconstructs the field *at the plane the rays already
reached*. Propagation away from that plane is a wave-model operation
(`M_WAVE_CHROMATIX`), and the M1 evidence for `asm_propagate` is what makes that
handoff usable.
