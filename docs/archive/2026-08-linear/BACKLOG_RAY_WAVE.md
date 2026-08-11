# Proposed Linear Backlog — Ray/Wave Vertical Slice

Each issue should use `ISSUE_TEMPLATE.md` and produce one independently verifiable result.

## RW-001 — Inventory Existing Ray/Wave Code

- Type: audit.
- Output: map of entry points, imports, tests, versions, failures, duplicate/dead paths, and ownership.
- Non-goal: no refactor or renaming.

## RW-002 — Establish Optiland Minimal Probe

- Output: pinned import and one deterministic sequential trace.
- Verify: ray count, intersections, reference plane, available OPD/amplitude/polarization data, device/dtype.
- Non-goal: broad Optiland adapter.

## RW-003 — Establish Chromatix Minimal Probe

- Output: pinned import and one tiny known-field propagation.
- Verify: sampling coordinates, power normalization, analytic diffraction or Gaussian reference, local gradient.
- Non-goal: broad Chromatix adapter.

## RW-004 — Freeze Minimal Artifact Contracts

- Output: validated `RayBundle`, `WavefrontSamples`, `ComplexField`, and `PSF` schemas.
- Verify: serialization, units, axes, reference plane, normalization, invalid metadata rejection.
- Non-goal: universal artifact taxonomy.

## RW-005 — Characterize Legacy Ray-Wave Implementations

- Output: actual mathematical/implementation semantics for `ray_wave`, `ray_ewave`, and related code.
- Verify: plane wave, spherical wave, OPD offset, phase sign, density/grid refinement, amplitude weighting.
- Non-goal: rename, merge, optimize, or claim differentiability.

## RW-006 — Implement Narrow Optiland Export Adapter

- Output: one adapter operation that returns project-owned ray or wavefront artifacts at a named plane.
- Verify: deterministic output, metadata, structured failure, package version.
- Depends on: RW-002, RW-004.

## RW-007 — Stabilize `C_RAY_TO_WAVE` Reference Path

- Output: coupler consuming project-owned artifacts and returning `ComplexField`.
- Verify: characterization tests, power/refinement behavior, sampling diagnostics.
- Depends on: RW-004, RW-005.

## RW-008 — Implement Narrow Chromatix Propagation Adapter

- Output: one adapter operation from project `ComplexField` to propagated field/PSF.
- Verify: analytic/reference case, metadata, structured failure, local gradient.
- Depends on: RW-003, RW-004.

## RW-009 — Build End-to-End Forward Demo

- Output: one command for Optiland -> coupler -> Chromatix -> PSF with provenance and plots.
- Verify: independent/analytic PSF comparison and sampling refinement.
- Depends on: RW-006, RW-007, RW-008.

## RW-010 — Characterize Gradient Boundaries

- Output: derivative-path report for ray parameter, coupler parameter, and wave parameter.
- Verify: framework-local directional finite differences.
- Default conclusion: PyTorch/JAX boundary remains forward-only.
- Depends on: RW-009.

## RW-011 — Review Module Boundaries from Evidence

- Output: architecture decision recording which graph nodes, adapter capabilities, and Python modules should remain, split, merge, or be deleted.
- Evidence: change coupling, test isolation, artifact reuse, failure boundaries, and profiling.
- Depends on: RW-009.
