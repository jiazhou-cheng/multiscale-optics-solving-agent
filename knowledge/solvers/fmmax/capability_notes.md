# FMMAX capability notes

Grounded in the real `fmmax` API surface (`fmmax==1.7.1`) and the probes in
`probes/`. See `docs/SOLVER_AND_COUPLER_CATALOG.md` for how this fits
alongside Meent/TORCWA/TorchRDIT in the broader RCWA catalog.

## Use FMMAX for

- Periodic Fourier-modal / RCWA scattering simulation of stratified,
  laterally-periodic structures: `eigensolve_isotropic_media`,
  `eigensolve_anisotropic_media`, `eigensolve_general_anisotropic_media`
  per layer, then `stack_s_matrix` (or `stack_s_matrix_scan` for many
  layers) to cascade a full stack via the Redheffer star product
  (`redheffer_star_product`).
- PML-terminated (non-periodic-in-propagation-direction) stacks via
  `apply_uniaxial_pml` and `PMLParams`.
- Source construction: plane waves (`plane_wave_in_plane_wavevector`),
  Gaussian beams (`beams`/`gaussian_source`), dirac-delta/dipole-like
  sources (`dirac_delta_source`), and Bloch-mode sweeps
  (`brillouin_zone_in_plane_wavevector`).
- Far-field post-processing: `farfield`, `farfield_profile`,
  `farfield_integrated_flux`.
- Field reconstruction on real-space grids: `fields_on_grid`,
  `fields_on_coordinates`, `stack_fields_3d`, `unit_cell_coordinates`.
- Differentiable geometry/material parameterizations via `jax.grad` --
  confirmed for one narrow path (substrate index -> bare-interface
  reflectance, relative error 1.24e-4 vs. finite difference).

## Do not assume (per CLAUDE.md section 3, rule 1)

- That `s11` is a reflection coefficient -- it is **transmission** in
  FMMAX's labeling; `s21` is reflection. See `conventions.md`.
- That raw `|amplitude|^2` from a scattering matrix is physical power --
  it is not (see the failed naive energy-conservation check in
  `conventions.md`); use `amplitude_poynting_flux`/
  `directional_poynting_flux` instead, and test that path before trusting
  it.
- That all FMMAX return values share one batch-axis convention -- most are
  leading-batch, but amplitudes/fields are trailing-batch per the docs.
- That FMMAX's stated `exp(-i*omega*t)` time convention has been
  independently re-derived from source in this repository -- it is
  vendor-stated only so far.
- That any particular length unit is implied -- FMMAX is scale-agnostic
  (`c = eps0 = mu0 = 1`); the adapter is responsible for the meter
  conversion.
- That `derivative.verified: true` is warranted from the one probe run
  here -- it covers a single scalar parameter on a non-periodic
  (homogeneous-limit) structure, not a real grating or Fourier-order
  truncation.

## Update (2026-07-30, adapter pass): energy conservation via Poynting flux now closed

`src/multiscale_optics_agent/adapters/fmmax_adapter.py` implements the
`fmmax.directional_poynting_flux`-based `R + T ~= 1` check flagged below and
in `conventions.md`/`failure_guide.md` as not yet done -- it now closes to
~2e-7 residual for a bare interface and ~1e-7 for a small lamellar grating.
See `conventions.md`'s "Energy conservation via Poynting flux: now closed"
section for the exact formula. The "do not assume raw `\|amplitude\|^2` is
physical power" guidance below is unchanged and still correct; only the
"not yet done" status of the *follow-up* check has changed.

## Not yet exercised in this repository

- A real periodic grating (only the `approximate_num_terms=1` /
  zero-order homogeneous limit was tested, deliberately, as the Fresnel
  oracle check).
- Fourier-order convergence sweep (listed as a required probe in
  `knowledge/solver_cards/fmmax.yaml`, not yet done).
- Anisotropic / fully-anisotropic media eigensolves.
- GPU/TPU execution (this environment has CPU only).
- A working energy-conservation (`R+T=1`) check via the correct
  Poynting-flux accessors.
- An actual `ModelAdapter` implementation under
  `src/multiscale_optics_agent/adapters/` -- this pass only installs the
  package, confirms it imports and runs, and documents its real behavior.
