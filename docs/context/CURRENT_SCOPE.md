# Current Scope — Ray/Wave Vertical Slice

## Objective

- Produce one verified forward pipeline from a sequential lens model to a diffraction PSF.
- Use Optiland for ray tracing, the project ray-wave method as the coupler, and Chromatix for wave propagation.
- Learn the correct model and module boundaries from executable experiments instead of fixing the entire architecture first.

## Canonical Graph

```text
M_RAY_OPTILAND
  -> C_RAY_TO_WAVE
  -> M_WAVE_CHROMATIX
  -> C_FIELD_TO_PSF
```

`C_FIELD_TO_PSF` may remain a small local operation inside the Chromatix adapter during the first forward slice. Promote it only when another model needs the same contract.

## Required Inputs

- A small deterministic sequential lens system.
- Field point and wavelength.
- Named pupil or reference plane.
- Ray sampling configuration.
- Wave grid spacing, extent, propagation distance/model, and normalization.

## Required Outputs

- Ray trace diagnostics.
- Exported pupil/reference-plane samples.
- Reconstructed complex field.
- Propagated complex field and PSF.
- Sampling and power/energy diagnostics.
- Package versions, device, dtype, seed, and configuration.

## Immediate Non-Goals

- General multi-solver planning.
- Full benchmark suite.
- Broad wrappers for every Optiland or Chromatix API.
- End-to-end cross-framework autodiff.
- FMMAX, FDTDX, JAX-FEM, SAX, thermal, circuit, or metasurface integration.
- Renaming or deleting legacy ray-wave code before characterization.

## Forward Success Criteria

- The complete pipeline runs from a clean environment using pinned dependencies.
- Each stage emits a typed artifact with explicit scientific metadata.
- Plane-wave and spherical/reference-wave cases expose phase and amplitude errors.
- Reconstructed power is stable under increasing ray density and field-grid refinement.
- A simple lens produces a PSF consistent with an analytic or independently generated diffraction reference after conventions are matched.
- Known sampling failures produce actionable diagnostics rather than plausible output.

## Gradient Policy

- The Optiland/PyTorch to Chromatix/JAX boundary is forward-only initially.
- Verify local gradients inside each framework separately before designing a bridge.
- A custom cross-framework derivative is a later issue with its own finite-difference acceptance test.

## Open Questions to Resolve Experimentally

- Which Optiland reference plane and exported quantities are reliable enough for the coupler?
- Does the existing paper implementation consume rays, OPD samples, eikonal data, or a different representation?
- Which amplitude weight is physically correct for the current ray sampling measure?
- How should phase be unwrapped and referenced?
- Which Chromatix propagation method matches the intended optical regime?
- Which parts of the current implementation deserve stable adapters versus quarantine or deletion?
