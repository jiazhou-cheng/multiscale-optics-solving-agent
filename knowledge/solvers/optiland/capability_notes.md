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
- Wavefront/aberration/PSF/MTF/Zernike analysis, tolerancing, thin-film
  design, polarization and optimization are all **exercised and validated** by
  CHE-57's 41 tutorial reproductions (`tutorials/README.md`). Specifically
  confirmed working, each with recorded evidence:
  - `wavefront.OPD` / `OPDFan` / `ZernikeOPD` (standard, fringe, noll), and the
    per-surface third-order coefficients on `Optic.aberrations`.
  - `psf.FFTPSF` / `psf.HuygensPSF` with `strehl_ratio()`, and `mtf.FFTMTF` /
    `mtf.GeometricMTF`.
  - The whole `optiland.analysis` gallery (SpotDiagram, RayFan, YYbar,
    Distortion, GridDistortion, FieldCurvature, RmsSpotSizeVsField,
    RmsWavefrontErrorVsField, PupilAberration).
  - `optiland.thin_film`: `ThinFilmStack` (energy-conserving to 1.1e-16),
    `SpectralAnalyzer`, `analyze_color` (CIE 1931 / sRGB), `ThinFilmOptimizer`
    with custom registered operands, `NeedleSynthesis`, and the thin-film
    tolerancing suite.
  - `optiland.tolerancing`: `Tolerancing`, `SensitivityAnalysis`, `MonteCarlo`
    with `RangeSampler` / `DistributionSampler`.
  - `optiland.optimization`: `OptimizerGeneric`, `LeastSquares`,
    `DifferentialEvolution`, `GlassExpert`, the `operand_registry` extension
    point, `pickups`, and every variable type this repository might use
    (`radius`, `thickness`, `conic`, `material`, `decenter`, `tilt`,
    `asphere_coeff`, `polynomial_coeff`, `zernike_coeff`).
  - `optiland.multiconfig.MultiConfiguration` and `Optic.scale_system`.
  - Polarization: `PolarizationState`, `create_polarization`,
    `surfaces.set_fresnel_coatings()`, `PolarizerCoating`,
    `RealRays.update_intensity`.
  - Extension points: subclassing `NewtonRaphsonGeometry`, `BaseCoating` and
    `OptimizerGeneric`.
  See the "Confirmed NOT trustworthy" section below for the parts that are
  present but wrong or misleading.

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
  under the torch backend; no CUDA device available to test). `be.set_device`
  exists and accepts `'cpu'`; no other device has been exercised.
- ~~Broad hand-built prescription coverage.~~ **Substantially addressed by
  CHE-57**: the 41 tutorial reproductions build systems from 4 to 44 surfaces
  covering spherical, plane, even-aspheric, polynomial, Zernike-freeform, plane
  and tilted mirror surfaces; ideal, Abbe and catalog materials; `EPD`,
  `imageFNO` and `objectNA` apertures; angle and object-height fields;
  absolute (`z=`, `y=`) and relative (`thickness=`, `dy=`) placement; per-surface
  tilts and decenters; and multi-configuration systems. What remains
  un-exercised is listed under "Still not exercised" below.
- Wavefront/OPD export, pupil power, and coordinate-orientation conventions
  — all three are explicitly listed as `required_probes` in
  `knowledge/solver_cards/optiland.yaml` and remain undone.
- ~~Hard-stop/vignetting non-smoothness under autograd~~ — **evidence found by
  CHE-57, though on the numpy path rather than under autograd.** On the freeform
  tilted-mirror system of `tutorials/t35_three_mirror_anastigmat.py`,
  `OptimizerGeneric` (L-BFGS-B on a finite-difference gradient) terminates with
  `success=True` and message `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`
  after 4 iterations at a point **1.97x worse** than its start, at every
  `maxiter` from 60 to 500, inside the declared bounds. That is exactly the
  non-smooth-merit-function failure the solver card warned about. Treat
  `res.success` as no evidence at all: compare the merit before and after.
- ~~Root-causing why the torch-backend gradient tolerance (1.11e-03) is
  looser than the JAX-based solvers' tolerances in this repository.~~
  **RESOLVED by CHE-57**: the torch backend defaults to `precision=32`. The
  1.11e-03 was float32 finite-difference cancellation noise, not autodiff
  error. Under `be.set_precision('float64')` the same probe point agrees with a
  centered difference to 6.3e-9 and converges as O(eps^2). See
  `conventions.md` "Torch backend precision defaults to float32".

## CHE-17 analytic accuracy evidence

`benchmarks/level1/L1-RAY-01/` independently checks free-space intersection
and path accumulation, ideal paraxial focusing at three launch slopes, and a
full trace of Edmund Optics TECHSPEC #45-362. The catalog case uses the
manufacturer's published prescription and SCHOTT N-BK7 dispersion rather
than an Optiland sample as its oracle. For these explicitly constructed
surfaces, `RealRays.opd` is checked as accumulated optical path and benchmark
OPD is defined as ray-minus-chief within each field. This does not establish
the opaque ReverseTelephoto sample's internal OPD reference.

## Confirmed NOT trustworthy in the pinned version (CHE-57)

Each of these is present in the API, runs without error, and is wrong or
misleading. Every one has a reproduction and recorded evidence under
`tutorials/`.

- **`AbbeMaterialE`** does not reproduce its own Abbe number: recovered `V_e` is
  0.57-0.83x the requested value across `V_e = 20..80`, and it errs by 1.4e-2 in
  index against real N-BK7 over 0.42-0.70 um — 98x worse than
  `AbbeMaterial(model='buchdahl')`, which is self-consistent to 0.2% and matches
  N-BK7 to 1.4e-4. Use `model='buchdahl'` for anything quantitative.
- **`AbbeMaterial`'s 0.6.0 default model is `polynomial`**, a catalog-wide fit
  that misses its own defining numbers (`n_d = 1.49993` for a requested 1.5,
  `V_d = 63.96` for a requested 65). It emits a `FutureWarning` that the default
  becomes `buchdahl` in 0.7.0. Always pass `model=` explicitly.
- **`OptimizerGeneric` can return `success=True` while degrading the design**
  (see above).
- **`add_operand(target=...)` / `add_operand(min_val=...)` are preferences, not
  constraints.** In `t27` a weight-1 `seidel` target of zero against three
  weight-10 spot operands moves only 1.9x; in `t31` a declared `min_val=0.99`
  converges to 0.9719.
- **`GlassExpert.run()` consumes the material variables**: the problem holds 25
  variables before the call and 19 after. A caller reusing the problem gets a
  continuous-only problem silently.
- **A `ThinFilmCoating` shared across interfaces with reversed media yields
  `rays.i` > 1** (measured ~3.6 on the tutorial's own cemented doublet), i.e. an
  energy-non-conserving transmittance, with no error or warning.
- **`Material(name, reference=...)` prints its `k = 0` assumption below the
  Python stream layer.** `warnings.catch_warnings` sees nothing and
  `contextlib.redirect_stdout`/`redirect_stderr` do not capture it. A caller
  cannot detect programmatically that a material is being treated as lossless.
- **Custom coatings are not validated at all.** A `BaseCoating` subclass that
  assigns `rays.i = 1.5` or `rays.i = -0.25` is accepted without clamp or
  warning.
- **The official "Custom Surface Types" tutorial ships a wrong surface normal**
  (`a*x/r2` where `d(a*r)/dx = a*x/r`), off by 0.52 in direction cosine. It
  normalises to a unit vector and traces without raising, so a unit-norm
  assertion is not a validation of a hand-written derivative — compare against a
  finite difference of `sag()`.

## Still not exercised (CHE-57 scope boundary)

- GRIN media (`optiland.propagation.grin`), coordinate breaks, NURBS, Forbes
  Q-polynomial, Chebyshev, toroidal, biconic, grid-sag and odd-asphere
  geometries, and `optiland.sources`.
- Reading a genuine vendor-authored `.zmx` or `.seq`: both catalog tutorials'
  artifacts are unavailable (`t08`, `t24`), so only Optiland's own
  `save_zemax_file` output has been parsed back. That round trip is **lossy** —
  Zemax files store curvature, so R1 = 25.84 comes back as 25.84000000165376.
  The `.json` (`save_optiland_file`) round trip is exact.
- `Optic.draw3D()` and `optiland.visualization.system.optic_viewer_3d`: they hang
  indefinitely headlessly.
- `optiland.ml`, `optiland.solves`, `optiland.environment` (air-index models),
  `optiland.apodization`, `optiland.analysis.image_simulation`, and the
  `codev` fileio path.
- Anything on a GPU.
