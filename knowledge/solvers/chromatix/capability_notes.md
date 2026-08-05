# Chromatix capability notes

Grounded in the real `chromatix.functional` API surface of the pinned
commit (`d24bdf0`, tag `0.6.0`) and the probes in `probes/`. See
`docs/SOLVER_AND_COUPLER_CATALOG.md` section on wave models for how this
fits alongside TorchOptics/D-Flat in the broader catalog.

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

## Do not assume (per CLAUDE.md section 3, rule 1)

- That the output array shape equals the input shape after propagation —
  `asm_propagate` returns the padded grid (see `conventions.md`).
- That `dx` is preserved across a propagation call — `transform_propagate`
  changes it.
- That any particular length unit is implied — Chromatix is scale-agnostic;
  the adapter is responsible for the meter conversion.
- That vector/polarization conventions match this project's Jones-basis and
  propagation-frame requirements (CLAUDE.md section 7) — unverified.
- That `derivative.verified: true` is warranted from the one probe run here
  — it covers a single narrow path, not the full CLAUDE.md section 6.2
  contract.

## Not yet exercised in this repository

- GPU/TPU execution (this environment has CPU only; `jax.devices()` reports
  a single `TFRT_CPU_0`).
- Vector fields / polarization elements.
- Chromatic (multi-wavelength) batched propagation.
- Any comparison against an analytic oracle (Airy pattern, Gaussian beam) —
  listed as a required probe in `knowledge/solver_cards/chromatix.yaml` but
  not yet implemented.
- An actual `ModelAdapter` implementation under
  `src/multiscale_optics_agent/adapters/` — this pass only installs the
  package, confirms it imports and runs, and documents its real behavior.
