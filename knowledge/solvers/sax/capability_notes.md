# SAX capability notes

Grounded in the real `sax`/`sax.models` API surface of the installed
`0.18.2` package and the probes in `probes/`.

## Use SAX for

- Composing S-parameter circuits from a netlist (`instances`, `connections`,
  `ports` dict) via `sax.circuit(netlist, models)` -- verified end-to-end
  with a real MZI, matched against an analytic oracle to ~1.4e-15 relative
  error (see `conventions.md`).
- A substantial library of built-in ideal and dispersive component models
  (`sax.models`): `coupler_ideal`/`coupler`, `straight` (dispersive
  waveguide with `neff`/`ng`/`loss_dB_cm`), `mmi1x2`/`mmi2x2` (and `_ideal`
  variants), `grating_coupler`, `bend`, `phase_shifter`, `crossing_ideal`,
  `mirror`, `reflector`, `terminator`, `attenuator`, `circulator`,
  `isolator`.
- RF/microwave circuit elements too, not just photonics: `resistor`,
  `capacitor`, `inductor`, `admittance`, `impedance`,
  `transmission_line_s_params`, `microstrip`/`coplanar_waveguide` (with
  `cpw_epsilon_eff`/`cpw_z0`/`microstrip_z0` helper functions) -- useful if
  a benchmark ever needs an electrical/RF S-parameter path alongside
  photonics.
- Multiple S-matrix representations (`SDict`, `SDense`, `SCoo`) and
  multiple circuit-evaluation backends (`default`, `klu`, `additive`,
  `filipsson_gunnar`, `forward`) selectable via `sax.circuit(...,
  backend=...)` -- only `default` has been exercised here.
- Differentiable circuit parameters: `jax.grad` flows through plain
  JAX-array-valued S-parameter returns. Verified for one closed-form
  component (`coupler_ideal`'s `coupling`); NOT yet verified through an
  assembled circuit's matrix-solve backend.
- Measurement/instrument data interop: `sax.parse_touchstone`,
  `sax.write_touchstone`, `sax.parse_lumerical_dat`, `sax.fit` (fitting a
  parametric model to measured/simulated data) -- potentially useful for
  validating a coupler against externally measured S-parameters.

## Do not assume (per CLAUDE.md section 3, rule 1)

- That port names are `in0`/`out0` or any other fixed scheme -- they depend
  on `sax.set_port_naming_strategy`, which is global mutable state (see
  `conventions.md`).
- That the canonical repository is `github.com/flaport/sax` -- it moved to
  `github.com/gdsfactory/sax` (the old URL still redirects, but should not
  be cited as current).
- That git tags reflect the real release history -- PyPI is authoritative
  for this package (see `solver_card.yaml` install_hazard).
- That gradient-verified for one component model implies the assembled
  circuit (matrix-solve) path is differentiable -- not yet tested.
- That every SAX phase/sign convention matches this project's canonical
  `exp(-i omega t)` the way `straight` does -- only `straight` has been
  checked; other dispersive/lossy models are unverified.

## Not yet exercised in this repository

- Port permutation failure test (does SAX raise or silently misconnect if a
  netlist references a nonexistent port? Listed as a required probe in
  `knowledge/solver_cards/sax.yaml`, not yet done).
- Gradient through a full assembled `sax.circuit` (only a bare component
  model was gradient-tested).
- The non-ideal `coupler` model (only `coupler_ideal` was tested) and any
  lossy (`loss_dB_cm != 0`) waveguide behavior.
- GPU numerical agreement (this environment is CPU-only).
- `sax.fit`, touchstone import/export, and the RF/microwave element family.
- An actual `ModelAdapter` implementation under
  `src/multiscale_optics_agent/adapters/` -- this pass only installs the
  package, confirms it imports and runs, and documents its real behavior.
