# Chromatix capability notes

Grounded in the real `chromatix.functional` API surface of the pinned
commit (`d24bdf0`, tag `0.6.0`) and the probes in `probes/`.

## Use Chromatix for

- Scalar and vector free-space wave propagation via two distinct methods
  with different tradeoffs: `transform_propagate` (single-FFT Fresnel,
  changes sampling, cheap) and `asm_propagate` (angular spectrum, preserves
  sampling, needs explicit padding and is more expensive).
- Composable optical elements applied functionally to a `Field`: thin/thick
  lenses (`thin_lens`, `df_lens`, `ff_lens`, `high_na_ff_lens`,
  `thick_plano_convex_lens`), phase/amplitude masks (`phase_change`,
  `amplitude_change`, `sawtooth_grating`, `sinusoid_grating`), pupils
  (`circular_pupil`, `gaussian_pupil`, `tukey_pupil`, ...), polarization
  elements (`jones_vector`, `linear_polarizer`, `wave_plate`, ...), and
  sources (`plane_wave`, `point_source`, `objective_point_source`,
  `gaussian_plane_wave`).
- Differentiable end-to-end imaging pipelines: `jax.grad` flows through the
  `Field` pytree (a `Field` is an `equinox.Module`). Confirmed for one
  parameter path (lens focal length) in `probes/gradient_probe.py`.
- Batched/chromatic simulation: `Field` supports leading batch dims (e.g.
  multiple `z` planes from a vectorized `z` argument) and a trailing
  wavelength axis for `ChromaticScalarField`/`ChromaticVectorField`.
- Basic sensor/image formation (`basic_sensor`) and simple thick/multislice
  sample models (`multislice_thick_sample`, `polarized_multislice_thick_sample`,
  `fluorescent_multislice_thick_sample`) for computational microscopy.

## Do not assume (per repository scientific-contract requirements)

- That the output array shape equals the input shape after propagation —
  `asm_propagate` returns the padded grid (see `conventions.md`).
- That `dx` is preserved across a propagation call — `transform_propagate`
  changes it.
- That any particular length unit is implied — Chromatix is scale-agnostic;
  the adapter is responsible for the meter conversion.
- That vector/polarization conventions match this project's Jones-basis and
  propagation-frame requirements (repository scientific conventions) — unverified.
- That `derivative.verified: true` is warranted from the one probe run here
  — it covers a single narrow path, not the full repository gradient-verification
  contract.

## Validated in this repository (scalar ASM, CPU only)

- `ChromatixAdapter.run_standalone` (CHE-14): a deterministic, typed scalar
  angular-spectrum baseline. Two CPU runs produce identical summary metrics
  and byte-identical field arrays; nine distinct rejection paths each return
  a structured diagnostic and no field.
- `benchmarks/level1/L1-WAVE-01` Case 1 (CHE-18): **exact** homogeneous
  propagation. FFT-bin plane-wave eigenmodes propagated unpadded reproduce
  `u * exp(i k_z z)` with the exact `sqrt` dispersion relation to
  floating-point round-off -- max wrapped phase error `6.5e-6` rad at
  `|k_z z| = 118` and `3.3e-4` rad at `|k_z z| = 2952`, both tracking the
  predicted `eps_float32 * |k_z z|` bound. Amplitude, discrete power, and a
  `+z` then `-z` round trip are all exact to `<= 5e-7`. This validates the
  phasor sign, the non-paraxial dispersion relation, the SI scaling, the axis
  order, and the grid centring with no approximation in the way.
- `benchmarks/level1/L1-WAVE-01` Case 2 (CHE-18): ideal paraxial focusing of a
  finite rectangular pupil with three signed tilts. The focus lands at
  `+f * theta` to within `0.055` input pixels, FWHM matches `0.8859 lambda f / L`
  to `1e-4` (x) and `1.3e-2` (y), and the `sinc^2` first-sidelobe ratio matches
  `0.047180` to `2.6%`. Chromatix agrees with an independent float64
  angular-spectrum implementation to an overlap of `0.999999`, so the `0.4%`
  gap to the paraxial Fresnel oracle is the *oracle's* model error, not the
  solver's.

That validation covers scalar, monochromatic, complex64, CPU `asm_propagate`
and nothing else.

## Known defective: `high_na_ff_lens` (do not use for quantitative work)

`benchmarks/level1/L1-WAVE-01` Case 3 attempted to validate the vectorial
high-NA path against an independent float64 Richards–Wolf quadrature oracle
and found the solver **not sampling-independent**. Refining only the pupil
sampling, with wavelength, NA, focal length, output grid, and output pitch all
fixed, moves the `|E_z|` ring radius from 246 nm to 2536 nm (oracle: 197 nm)
and `Iz/Ix` from 0.126 to 0.366 (oracle: 0.150087, converged to `2e-14`). Best
achievable vector overlap was `0.070`.

Root cause, read from the pinned source: `high_na_ff_lens` derives `s_z` from
`field.f_grid * lambda / n` — the *frequency* grid — rather than from the pupil
position grid. On any physically sampled pupil `|s_grid| ~ 0.015`, so
`s_z ~ 1`, and the intended `1/cos(theta)` obliquity Jacobian, the
`exp(i k f cos(theta))` defocus, and the `zoom_factor` that sets the output
scale all degenerate to constants.

What *is* correct there, recorded for whoever fixes it upstream:

- `cartesian_to_spherical` implements the standard Richards–Wolf aplanatic
  polarization transform exactly (verified algebraically against
  `1 + (cos t - 1) cos^2 phi`).
- `VectorField.u` component order on the last axis is `(E_z, E_y, E_x)` — the
  reverse of this project's `(E_x, E_y, E_z)`.
- No `sqrt(cos theta)` apodization is applied anywhere; an aplanatic objective
  requires the caller to supply it in the pupil field.

The same class of caveat applies to `thick_plano_convex_lens`, which is
implemented through an ABCD matrix and `ray_transfer` rather than a
surface-resolved wave-optical prescription.

## Not yet exercised in this repository

- GPU/TPU execution (this environment has CPU only; `jax.devices()` reports
  a single `TFRT_CPU_0`).
- Vector fields / polarization elements.
- Chromatic (multi-wavelength) batched propagation.
- Airy-pattern / hard-aperture diffraction comparison — listed as a required
  probe in `knowledge/solver_cards/chromatix.yaml`, still not implemented.
  The Gaussian benchmark deliberately uses a smooth, band-limited field and
  therefore says nothing about how this path handles a sharp aperture edge.
- Any gradient through `asm_propagate`. The one gradient probe covers
  `thin_lens -> transform_propagate`, which neither adapter entry point
  implements.
