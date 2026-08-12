# L1-WAVE-01 design — what is an oracle, and what each case can prove

CHE-18. This is the boundary the benchmark is reviewed against.

## Why a progression rather than one case

A single "solver versus theory" number cannot distinguish a wrong solver from
an approximate oracle, a coarse grid, a mismatched estimator, or a convention
error. The suite is therefore ordered so that each rung removes one class of
excuse before the next rung introduces it:

- **Case 1** has *no* approximation. The oracle is exact, the grid is exact
  (the mode is an eigenmode), no padding is involved, and the estimator is a
  direct phase comparison. If Case 1 fails, nothing above it is interpretable.
  Anything Case 1 measures is either the phasor sign, the dispersion relation,
  the SI scaling, the axis order, or floating-point round-off — and the
  tolerance is derived from the last of those, so the other four are the only
  things that can actually fail.
- **Case 2** adds exactly two approximations: the paraxial (Fresnel) oracle
  and a hard aperture edge on a finite grid. Both are *measured*, not assumed,
  by carrying an independent float64 angular spectrum alongside.
- **Case 3** adds vector physics and the Debye approximation.

## Evidence classification

**Independent analytic oracles** (`oracles.py`, imports no Chromatix, no JAX;
a test enforces that):

- Exact discrete plane-wave eigenmode transfer, `u · exp(i k_z z)`.
- Fresnel/Fourier focal field of a rectangular pupil behind an ideal thin lens.
- Richards–Wolf (Debye–Wolf) quadrature for an x-polarized aplanatic objective.

**Cross-check, not an oracle:** a float64 NumPy angular-spectrum propagation.
Its only job is to split a paraxial oracle's own model error from the solver's
implementation error.

**Solver under test:** `ChromatixAdapter.run_standalone` for Cases 1–2;
`chromatix_benchmark_adapter.high_na_vector_focus` for Case 3.

**Explicitly not an oracle:** any Chromatix-generated value, including the
recorded `propagation_probe.json` snapshot. That file establishes that
behaviour has not changed, not that it is right.

## Case 1: why unpadded, and why the tolerance is derived

An FFT-bin plane wave is periodic on the grid, so the discrete angular
spectrum reproduces it exactly. Zero-padding would introduce an aperture edge
that is not in the physics, converting an exact test into an approximate one;
`pad_width` is therefore 0 and the evaluator gates on the solver having
applied no padding.

Since nothing is approximated, a chosen tolerance would be arbitrary. The one
real error source is that `exp(i k_z z)` is evaluated in single precision on
an accumulated phase of order 10³ rad, giving `eps_float32 · |k_z z|`.
Observed (max wrapped phase error, radians):

| `|k_z z|` | observed | derived bound |
|---|---|---|
| 118 | 6.5e-6 | 7.0e-5 |
| 2952 | 3.3e-4 | 1.8e-3 |

The bound tracks the measurement across the range, which is the point: a
constant tolerance would have been either vacuous or unmeetable.

The `case1_paraxial_dispersion` perturbation exists to prove the case really
resolves `√(1 − (λf)²)` rather than merely agreeing with some plausible axial
wavenumber. Substituting the Taylor expansion produces 3.1 rad of error
against a 0.016 rad threshold.

## Case 2: measurement choices that are not obvious

**No `D4σ`.** The focal pattern of a hard rectangular aperture is a `sinc²`,
whose tails fall as `1/t²`, so its second moment *diverges*. A second-moment
width would measure the window, not the beam. Widths are FWHM by sub-pixel
linear interpolation of the half-maximum crossings. The *first* moment does
converge by symmetry, so the centroid remains a valid position estimator.

**Odd aperture sample counts.** 201 × 121 samples, so the aperture is exactly
symmetric about the origin on the `n//2`-centred grid and the sampled pupil
width matches the continuous width the oracle assumes. An even count leaves a
half-pixel inconsistency that shows up as a spurious width error.

**Different widths on the two axes, and an asymmetric tilt.** A square
aperture on axis is invariant under transposition, so an axis-order error
would be undetectable. 201 × 121 with tilt `(2e-3, 1e-3)` makes it visible in
both the lobe widths and the focal position.

**The residual quadratic phase is kept.** The focal field carries
`exp(i k r'²/2f)`; it is not an aberration. Dropping it from the oracle would
leave the intensity metrics untouched while silently breaking the complex
overlap — so the overlap is what tests it.

Attribution at the reference configuration:

| component | x axis | y axis |
|---|---|---|
| `discretization_and_window` | 9.6e-5 | 1.2e-4 |
| `paraxial_model` | 7.7e-5 | 1.3e-2 |
| `solver_implementation` | 2.4e-5 | 8.1e-5 |
| `convention` | 1.3e-6 | — |

Chromatix agrees with the independent float64 propagation to `1e-6` overlap.
The y-axis residual is the narrower aperture's discrete-versus-continuous
Fourier gap, which the independent implementation reproduces — so it is model
error, and the 2 % FWHM tolerance is set to accommodate it knowingly rather
than by accident.

## Case 3: what it can and cannot prove

The oracle side is sound and is demonstrated so: Gauss–Legendre quadrature
over `θ ∈ [0, arcsin(NA/n)]` converges to a relative `2e-14` by 25 nodes, and
reproduces the known aplanatic `Iz/Ix = 0.150087` at NA 0.9.

The solver side fails the prerequisite. Refining only the pupil sampling — a
change with no physical content — moves the `|E_z|` ring radius from 246 nm to
2536 nm and `Iz/Ix` from 0.126 to 0.366. There is therefore no converged
solver quantity to compare, and the case is reported **blocked** rather than
failed: "failed" would imply a measured disagreement between two well-defined
numbers, which is not what happened.

Root cause, from the pinned source: `s_z` is derived from `field.f_grid · λ/n`
instead of the pupil position grid, so `|s_grid| ≈ 0.015` and `s_z ≈ 1`; the
`1/cosθ` obliquity Jacobian, the `exp(i k f cosθ)` defocus, and the
`zoom_factor` that sets the output scale all degenerate. Recorded for whoever
fixes it: `cartesian_to_spherical` *is* the correct Richards–Wolf aplanatic
polarization transform, the component order is `(E_z, E_y, E_x)`, and no
`√cosθ` apodization is applied by the library.

Case 3 does not gate `accuracy.pass`, but it is surfaced in the summary, in
`result.json` under `blocked_cases`, and on stderr, so it cannot be mistaken
for a clean run.

## Deliberately excluded

A catalog plano-convex lens, for the same class of reason Case 3 is blocked:
Chromatix 0.6.0's `thick_plano_convex_lens` is implemented through an ABCD
matrix and `ray_transfer`, not a surface-resolved wave-optical prescription,
so it would test a ray model wearing a wave model's interface.

Also out of scope: aperture/Airy diffraction as a separate physical case,
chromatic propagation, GPU, gradients, and both coupling directions.
