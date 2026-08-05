# Ray-to-Wave Vertical Slice — Execution Plan

## Stage 0 — Repository Audit

- Inventory existing ray, wave, ray-wave, and ray-ewave code.
- Mark each path as: executable, partially executable, dead, duplicate, or unknown.
- Record imports, entry points, tests, package versions, and observed failures.
- Do not refactor during this stage.

Deliverable: an evidence-based map and a list of candidate code to preserve, quarantine, or delete later.

## Stage 1 — Minimal Solver Probes

### Optiland

- Pin the installed version.
- Import the package.
- Run one deterministic sequential trace.
- Export the quantities available at a named plane.
- Check whether intended parameter paths retain PyTorch gradients.

### Chromatix

- Pin the installed version.
- Import the package.
- Propagate a known field on a tiny grid.
- Compare with a simple analytic diffraction or Gaussian-beam case.
- Check local JAX gradients on a scalar objective.

Deliverable: tiny probes that execute in seconds and become adapter tests.

## Stage 2 — Stabilize Project-Owned Artifacts

Define only the fields required by the first slice:

- `RayBundle`.
- `WavefrontSamples`.
- `ComplexField`.
- `PSF`.

For every artifact, define units, axis order, frame, wavelength, phase convention, normalization, dtype, device, and reference plane.

Deliverable: serialization and validation tests independent of Optiland and Chromatix.

## Stage 3 — Characterize the Existing Coupler

- Identify its real mathematical map.
- Freeze current numerical output on tiny deterministic inputs.
- Test plane wave, spherical wave, OPD offset invariance, phase sign, ray-density refinement, and grid refinement.
- Identify amplitude weighting, interpolation kernel, filtering, phase unwrapping, and caustic behavior.
- Label non-smooth and unsupported cases.

Deliverable: characterization report and tests. No renaming or optimization yet.

## Stage 4 — Narrow Adapters

- Optiland adapter returns project-owned ray or wavefront artifacts.
- Coupler consumes only project-owned artifacts.
- Chromatix adapter consumes only project-owned complex fields.
- Solver-specific objects do not leak across boundaries.

Deliverable: one stable path per solver, not a broad wrapper.

## Stage 5 — End-to-End Forward Demo

Use a simple lens and a declared wavelength/field point:

- Trace rays to the selected plane.
- Reconstruct the complex field.
- Propagate in Chromatix.
- Produce a normalized PSF and diagnostics.
- Compare against a direct ideal-pupil or analytic reference.
- Repeat at increasing ray and grid resolution.

Deliverable: one command that produces results, provenance, plots, and verification status.

## Stage 6 — Gradient Characterization

Only after Stage 5 passes:

- Verify one Optiland-local design parameter against finite differences.
- Verify one coupler-local parameter, if mathematically smooth.
- Verify one Chromatix-local parameter against finite differences.
- Keep the framework boundary forward-only unless a separately reviewed custom derivative is implemented.

Deliverable: derivative-path report; do not claim end-to-end autodiff by default.

## Exit Condition for the Current Milestone

- A clean checkout can reproduce the forward demo.
- Solver and coupler assumptions are explicit.
- At least one analytic or independent reference agrees within a threshold frozen in the corresponding Linear issue.
- Sampling refinement and failure diagnostics are recorded.
- Legacy code has an evidence-backed disposition.
- Follow-up architecture work is based on observed coupling and test boundaries.
