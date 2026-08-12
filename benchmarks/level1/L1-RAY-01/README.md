# L1-RAY-01 — Optiland analytic ray accuracy

Generate and evaluate the complete CPU/float64 evidence bundle with:

```bash
./run.sh python benchmarks/level1/L1-RAY-01/run_benchmark.py \
  --output-dir outputs/L1-RAY-01
```

Reproduce the standalone, analytic, and scaling evidence as one independent
M1 branch bundle with:

```bash
./run.sh python benchmarks/level1/L1-RAY-01/run_all.py \
  --output-dir outputs/M1/ray
```

This wrapper rejects scientific artifact corruption, emits a canonical
scientific fingerprint, and does not import Chromatix or any coupler.

Run the CHE-16 sampling, determinism, and CPU scaling section separately:

```bash
./run.sh python benchmarks/level1/L1-RAY-01/run_scaling.py --backend numpy
./run.sh python benchmarks/level1/L1-RAY-01/evaluate.py --section scaling
```

The scaling section holds the accepted CHE-13 `ReverseTelephoto` prescription,
field `(Hx, Hy)=(0, 0)`, and wavelength `0.55 µm` fixed while varying only the
requested hexapolar sampling density. It reports generated/traced/valid-surviving
ray counts separately and invokes this unchanged analytic bundle as a mandatory
accuracy gate.

The command runs Optiland only: no wave model and no coupler. Numerical
metrics in `result.json` are authoritative for pass/fail; figures are
human-readable diagnostic evidence.

## Protocol amendment

The benchmark ID remains `L1-RAY-01`. CHE-20 amends the frozen V1 design to
`M1-BASELINE-CPU-V2` because surface-shape and clear-aperture behavior are new
authoritative checks. The original three physical cases and their V1 metrics
remain reproducible; V2 adds evidence rather than changing the lens
prescription.

## What the three cases validate

### 1. Homogeneous free-space propagation

Three manufactured rays travel from `z=0` to `z=100 mm` through ideal air.
The independent oracle evaluates
`t=(z_plane-z0)/N`, `x=x0+tL`, `y=y0+tM`, geometric path `t`, and OPL `n*t`
with `n=1`. Position, direction, path/OPL, direction norm, finiteness,
invalid/vignetted counts, and a deliberately wrong unit conversion are
reported. `free_space_diagnostics.png` shows the ray layout and per-ray
residuals against their tolerances.

“Free-space” refers to propagation through a homogeneous medium and must not
be confused with “freeform optics.”

### 2. Ideal paraxial thin lens

An ideal `f=50 mm` paraxial model surface receives five pupil heights for each
of three signed launch slopes. The independent ABCD oracle is
`u_after=u_before-y/f` and `y_image=f*u_before`, where
`u_before=tan(launch_angle)`. `paraxial_diagnostics.png` shows the ideal model
plane, incoming/outgoing segments, image plane, Optiland intercept markers,
ABCD reference lines, and residuals. Solver residual is reported separately
from physical paraxial-model approximation; higher-order physics is outside
this ideal case.

### 3. Edmund #45-362 catalog lens

The highest-complexity case traces the published spherical plano-convex N-BK7
prescription: `R1=+25.84 mm`, planar rear, center thickness `3.23 mm`, clear
aperture `19 mm`, catalog EFL `50.00 mm`, catalog BFL `47.87 mm`, and
`587.6 nm` wavelength. A single parsed prescription object drives the
Optiland optic, metadata, spherical-sag/plane oracles, saved rendering
geometry, and plots.

Independent analytic evidence includes SCHOTT dispersion, thick-lens EFL/BFL,
the axial chief-ray geometric path/OPL, the spherical-front sag equation, the
planar-rear equation, and dedicated aperture classification at `±9.4 mm`
(inside) and `±9.6 mm` (outside). Edmund values are manufacturer-reference
evidence. Marginal ray coordinates, spot RMS, and ray-minus-chief OPD are
deterministic regression evidence, not independent accuracy oracles.

`catalog_diagnostics.png` contains the full layout, a lens close-up with
surface intersections/refraction, image intercept versus pupil height, and
ray-minus-chief OPD. Optical surfaces are solid, N-BK7 is transparently
filled, and the BFL image/reference plane is dashed.

Edmund #45-362 is modeled as a spherical plano-convex N-BK7 lens. L1-RAY-01
does not contain a freeform optical surface.

## Pass/fail evidence

`result.json` records per-case pass flags and the underlying errors. In
addition to the V1 propagation, focusing, material, focal-length, path,
centroid, symmetry, direction, validity, and OPD evidence, V2 requires:

- `surface_shape_pass`: front intersections satisfy spherical sag and rear
  intersections satisfy the rear plane within `tolerances.yaml`.
- `aperture_classification_pass`: all dedicated inside rays transmit and all
  outside rays vignette. These rays do not enter centroid or spot-RMS metrics.
- the convention-negative test detects an intentionally incorrect mm-to-m
  scale.

`error_attribution.json` separates solver/numerical error, ideal-model
approximation, prescription/reference uncertainty, sampling, and
aperture/vignetting effects.

## Artifact inventory

- `result.json`: machine-readable metrics, tolerances, conventions, and pass.
- `provenance.json`: V2 protocol amendment, command, environment, engine
  versions, prescription, and SHA-256 hashes.
- `arrays.npz`: SI ray outputs, exact catalog surface-rendering arrays, image
  plane, clear aperture, and dedicated aperture-subtest outputs.
- `ray_inputs.npz`: exact SI inputs for all primary and aperture rays.
- `expected.npz`: independently calculated oracle arrays.
- `plot.png`: exactly one overview panel for each of the three cases.
- `free_space_diagnostics.png`: free-space layout and residuals.
- `paraxial_diagnostics.png`: paraxial layout, ABCD comparison, and residuals.
- `catalog_diagnostics.png`: full/close-up layouts, intercepts, and OPD.
- `input_config.yaml`, `prescription.yaml`, and `tolerances.yaml`: copied
  execution inputs.
- `error_attribution.json`: evidence classification and error sources.
- `README.md`: this interpretation and reproduction guide.

See `benchmark_design.md` for the pre-implementation V1 design and its V2
amendment.
