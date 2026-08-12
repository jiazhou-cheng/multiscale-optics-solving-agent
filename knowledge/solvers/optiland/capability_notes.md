# Optiland capability notes

Grounded in the real `optiland` 0.6.0 API surface and the probes in
`probes/`.

## Use Optiland for

- Sequential geometric ray tracing through lens/mirror systems built either
  from scratch (`optiland.optic.Optic`) or from bundled samples
  (`optiland.samples.objectives.ReverseTelephoto` and others — not yet
  enumerated in this pass).
- The CHE-13 standalone `ReverseTelephoto` CPU baseline through
  `OptilandAdapter.run_standalone()`: typed request/result, SI position and
  wavelength output, native OPD preservation, finite/unit-direction checks,
  survivor-aware shapes, scientific-array hashing, and deterministic replay.
- Paraxial analysis (`Optic.paraxial`, e.g. `.f2()` for effective focal
  length) without needing the torch backend at all.
- Differentiable lens-parameter optimization **once the torch backend is
  explicitly selected**: confirmed end-to-end gradient flow from a surface's
  radius of curvature, through `Optic.trace()`, to a scalar RMS-spot-style
  objective (`probes/gradient_probe.py`, relative error 1.11e-03 vs.
  centered finite difference).
- Wavefront/aberration/PSF/MTF/Zernike analysis, tolerancing, and
  optimization modules exist (`optiland.wavefront`, `optiland.psf`,
  `optiland.mtf`, `optiland.zernike`, `optiland.tolerancing`,
  `optiland.optimization`) — present in the API but **not exercised** in
  this pass; treat as unverified capability, not confirmed capability.

## Do not assume (per repository scientific-contract requirements)

- That `pip install optiland` gives you differentiability or GPU support —
  it does not. torch is optional, not a declared dependency, and gradients
  require an explicit `optiland.backend.set_backend('torch')` call. See
  `conventions.md` and `failure_guide.md`.
- That `derivative.mode: native_autodiff` / `framework: pytorch` in this
  project's registry (`M_RAY_OPTILAND`) holds unconditionally — it holds
  only for code paths that explicitly opted into the torch backend, and
  only one such path (lens-radius -> trace -> RMS-spot proxy) has actually
  been verified here, with a tolerance looser than every JAX-based solver
  probed in this repository (1.11e-03 vs. typically 1e-4–1e-13).
- That the number of rays returned by `Optic.trace(num_rays=N)` equals `N`
  — apertures/vignetting/pupil sampling change the survivor count (observed
  16 requested -> 817 returned for the bundled sample system; the exact
  relationship between requested and returned ray count was not
  characterized further).
- That `optiland.backend`'s `supports_gpu`/`supports_gradients` are
  functions — they are plain `bool` attributes.
- That solver-native lengths are already SI. CHE-12 verified geometry in mm
  and trace wavelength in um; the adapter performs the explicit conversion.
- That `RealRays.opd` is absolute OPL or piston-removed OPD. Its reference and
  sign remain unverified and the standalone baseline labels it regression-only.

## Not yet exercised in this repository

- GPU execution (`supports_gpu` is `False` in this CPU-only container even
  under the torch backend; no CUDA device available to test).
- Broad hand-built prescription coverage. CHE-17 exercises one narrow
  construction path: planar/free-space surfaces, an ideal paraxial surface,
  and the two-surface Edmund Optics #45-362 N-BK7 plano-convex lens.
- Wavefront/OPD export, pupil power, and coordinate-orientation conventions
  — all three are explicitly listed as `required_probes` in
  `knowledge/solver_cards/optiland.yaml` and remain undone.
- Hard-stop/vignetting non-smoothness under autograd (the existing solver
  card already flags this as a should-not-assume item; not tested here).
- Root-causing why the torch-backend gradient tolerance (1.11e-03) is
  looser than the JAX-based solvers' tolerances in this repository.

## CHE-17 analytic accuracy evidence

`benchmarks/level1/L1-RAY-01/` independently checks free-space intersection
and path accumulation, ideal paraxial focusing at three launch slopes, and a
full trace of Edmund Optics TECHSPEC #45-362. The catalog case uses the
manufacturer's published prescription and SCHOTT N-BK7 dispersion rather
than an Optiland sample as its oracle. For these explicitly constructed
surfaces, `RealRays.opd` is checked as accumulated optical path and benchmark
OPD is defined as ray-minus-chief within each field. This does not establish
the opaque ReverseTelephoto sample's internal OPD reference.
