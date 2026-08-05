# FDTDX capability notes

Grounded in the real `dir(fdtdx)` API surface of the pinned `0.6.2` install
and the probes in `probes/`. See `docs/SOLVER_AND_COUPLER_CATALOG.md` for
how this fits alongside Jaxwell/Tidy3D/fdtdz in the broader catalog.

## Use FDTDX for

- General 3D full-wave FDTD electromagnetic simulation: arbitrary
  `SimulationVolume` geometry built from `Cylinder`, `Sphere`,
  `ExtrudedPolygon` (including direct GDS import via
  `extruded_polygon_from_gds`/`extruded_polygon_from_gds_path`), periodic
  or PML boundaries (`BoundaryConfig`, `PeriodicBoundary`,
  `PerfectlyMatchedLayer`, also `PerfectElectricConductor`/
  `PerfectMagneticConductor`, `BlochBoundary` for oblique-incidence
  periodic problems).
- Dispersive materials via pole models (`DispersionModel`, `LorentzPole`,
  `DrudePole`, `compute_pole_coefficients`).
- Detectors/monitors: `EnergyDetector` (confirmed real-valued
  `(component, x, y, z)` field array, see `conventions.md`),
  `FieldDetector`, `PhasorDetector`, `ModeOverlapDetector`,
  `PoyntingFluxDetector`.
- S-parameter extraction as a first-class workflow:
  `setup_sparams_simulation`, `calculate_sparam`, `calculate_sparams`
  (not probed in this pass, but present and clearly intended for exactly
  this project's `C_FIELD_TO_MODE`/S-parameter coupler needs).
- Inverse-design parameterization: `Device`, `ParameterContainer`,
  `TanhProjection`, `SubpixelSmoothedProjection`,
  `BinaryMedianFilterModule`, symmetry/size constraints
  (`DiagonalSymmetry2D/3D`, `SizeConstraint`, `SizeExtensionConstraint`) --
  this machinery is clearly where the real differentiable-parameter path
  lives, not raw `jax.grad` over construction-time arguments (see
  `conventions.md`).
- Time-reversed backward propagation (`full_backward`) as the package's
  intended efficient-gradient mechanism, per its own cited method
  (Schubert et al., ACS Omega 2024) -- not yet exercised here.

## Do not assume (per CLAUDE.md section 3, rule 1 and rule 2)

- That the GitHub `main`-branch README example matches the pinned release's
  API. It does not (`UniformGrid`/`grid=` vs. `resolution=`) -- always
  verify against the actually-installed version's `inspect.signature`
  before trusting an example. See `failure_guide.md`.
- That leaving `SimulationConfig(backend=...)` unset means "GPU or error".
  It silently falls back to CPU with a log warning.
- That `jax.grad` over the whole setup-and-run function works for an
  arbitrary parameter. Two attempts both failed outright (see
  `conventions.md` and `solver_card.yaml` `derivative.verified_note`) --
  **do not claim any FDTDX parameter is verified-differentiable** until the
  `Device`/`ParameterContainer`/`apply_params` pattern is worked out and
  tested.
- That array axis order matches Chromatix's `(..., height, width)`. FDTDX's
  field arrays are `(component, x, y, z)` -- component-major, not
  component-trailing.

## Not yet exercised in this repository

- PML boundaries and domain-size/grid convergence (probe used periodic
  boundaries only, per `required_probes` in
  `knowledge/solver_cards/fdtdx.yaml`).
- Fresnel-interface or waveguide-mode analytic comparison.
- The correct differentiable-parameter pattern via `Device`/
  `ParameterContainer`/`apply_params`, and `full_backward`.
- S-parameter extraction (`calculate_sparam(s)`).
- GPU execution and CPU/GPU numerical agreement (no GPU in this
  environment).
- An actual `ModelAdapter` implementation under
  `src/multiscale_optics_agent/adapters/`.
