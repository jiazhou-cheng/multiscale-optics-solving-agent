# M1 independent baseline protocol

CHE-12 freezes the execution and reporting contract for `L1-RAY-01` and
`L1-WAVE-01`. It does not implement either scientific baseline. The
machine-readable contract is [`protocol.yaml`](protocol.yaml); benchmark
implementations must validate their JSON files against the schemas in
[`schemas/`](schemas/).

## Protocol amendments

| Protocol ID | Applies to | Issue | What changed |
|---|---|---|---|
| `M1-BASELINE-CPU-V1` | `L1-RAY-01`, `L1-WAVE-01` | CHE-12 | Base contract frozen below. |
| `M1-BASELINE-CPU-V2` | `L1-RAY-01` | CHE-20 | Adds authoritative surface-shape and clear-aperture evidence. |

An amendment may only **add** required evidence. It may not relax a V1
requirement, and it may not change a physical case, a prescription, or a
tolerance — those require a new `protocol_id` and a new frozen design. Because
V2 only adds, every V1 metric of `L1-RAY-01` remains reproducible under V2.

`M1-BASELINE-CPU-V2` adds two required `result.json` accuracy metrics —
`surface_shape_pass` (front intersections satisfy the spherical sag equation,
rear intersections satisfy the rear plane) and `aperture_classification_pass`
(dedicated rays at `±9.4 mm` transmit, at `±9.6 mm` vignette; these rays are
excluded from centroid and spot-RMS metrics) — and six required artifacts:
`ray_inputs.npz`, `expected.npz`, `error_attribution.json`, and the three
per-case `*_diagnostics.png` figures. The machine-readable form of this table
is the `amendments:` block in [`protocol.yaml`](protocol.yaml).

`L1-WAVE-01` remains on `M1-BASELINE-CPU-V1`. The two branches therefore
legitimately declare different protocol IDs; this is recorded rather than
normalized, because retrofitting V2 evidence onto the wave branch would mean
re-freezing a design that already passed review.

The coupler benchmarks introduced in M2 are governed by a separate contract,
[`protocols/m2_coupler_protocol.md`](protocols/m2_coupler_protocol.md), which extends rather than
replaces the measurement and artifact rules below.

## Verified engine decision

The container probe `benchmarks/probes/engine_independence.py` passed on
2026-08-11. The ray process observed Optiland 0.6.0, Torch 2.13.0+cpu,
NumPy 2.2.6, and Python 3.12.13. The wave process observed Chromatix 0.6.0
installed from commit `d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee`, JAX
0.6.2 on CPU, NumPy 2.2.6, and Python 3.12.13. Both processes recorded the
CPU model, logical CPUs, and process affinity. Neither loaded the other
engine or a repository coupler.

Run the checks independently:

```bash
./run.sh python benchmarks/probes/engine_independence.py --engine ray
./run.sh python benchmarks/probes/engine_independence.py --engine wave
```

## Ray baseline boundary

| Contract item | Frozen `L1-RAY-01` convention | Verification state |
|---|---|---|
| SI conversion | Convert Optiland geometry mm to m with `1e-3`; wavelength µm to m with `1e-6` at the adapter/artifact boundary | Verified by pinned-source marker plus executable 10 mm trace |
| Axes/order | Ray arrays are flat, same-length arrays for `x,y,z,L,M,N,intensity,opd`; preserve field names, no implicit matrix axis | Observed |
| Frame | Right-handed Cartesian coordinates are required in emitted benchmark metadata | Must be asserted by baseline implementation |
| Handedness | Right-handed project frame; no silent reflection/permutation | Protocol decision; oracle check pending |
| Wavelength | Optiland trace input/output is µm; artifact/provenance value is m | Verified |
| Reference plane | Name the final traced surface and its axial coordinate | Required; benchmark prescription pending |
| OPL/OPD | Preserve Optiland `opd` separately; declare reference and piston removal before comparison | Semantics unverified; cannot be used as an oracle yet |
| Phasor | Not applicable to geometric ray coordinates; any later phase conversion is outside this baseline | Frozen non-goal |
| Amplitude/weight | `RealRays.intensity` is a ray weight/intensity, not a coherent field amplitude | Observed contract limitation |
| Normalization | Report raw and explicitly normalized ray metrics separately | Protocol decision |
| Sampling | Record requested pupil sampler and returned ray count; never assume `num_rays` equals returned count | Observed |
| Validity | Sequential geometric optics; hard stops/vignetting may be non-smooth | Registry/card contract |
| Dtype/device | NumPy float64, CPU for M1 benchmark | Verified environment |
| Derivatives | No derivative or gradient claim in M1 baseline | Frozen non-goal |

The known-distance probe traces a 10-unit planar separation and observes
final `z=10`; the pinned Optiland source explicitly labels the same geometry
coordinates in mm. Its `opd=12` is deliberately not interpreted because the
reference convention has not been established.

## Wave baseline boundary

| Contract item | Frozen `L1-WAVE-01` convention | Verification state |
|---|---|---|
| SI conversion | Supply `dx`, wavelength, and `z` in meters; Chromatix accepts any one consistent length scale | Library behavior observed; SI is project decision |
| Axes/order | Spatial arrays are `(y,x)`; chromatic fields append wavelength as the last axis | Verified |
| Frame | Array row is `y`, column is `x`; record coordinate origin and increasing-axis direction | Axis order verified; benchmark grid pending |
| Handedness | Right-handed project frame with propagation along `+z` | Protocol decision; manufactured check pending |
| Wavelength | Monochromatic wavelength in meters for this baseline | Protocol decision |
| Reference plane | Input plane is `z=0`; output plane is named and records signed propagation distance | Protocol decision |
| OPL/OPD | Not an output of the propagation baseline | Frozen non-goal |
| Phasor | Spatial factor `exp(+i k·r)`, consistent with project `exp(-iωt)`; record this string verbatim | Source verified; cross-solver check pending |
| Amplitude/weight | `ScalarField.u` is complex amplitude; intensity is `abs(u)^2` | Library contract |
| Normalization | Record input/output discrete power and analytic-oracle normalization; do not infer FFT normalization | Protocol decision |
| Sampling | Record shape, `(dy,dx)`, padding/cropping, and output pitch; propagators may change shape or pitch | Verified |
| Validity | State paraxial/band-limit and padding conditions for each case | Required; case construction pending |
| Dtype/device | Complex64, JAX CPU, x64 disabled for M1 benchmark | Verified environment |
| Derivatives | No derivative or gradient claim in M1 baseline | Frozen non-goal |

## Measurement and artifacts

Each baseline runs in its own fresh process through `./run.sh`. Use two
untimed warmups and seven timed repeats. Synchronize lazy backends before
and after timing, and report median as the primary statistic plus minimum
and p95. Record CPU affinity, logical CPU count, CPU model, and relevant
thread counts. Affinity is an observable fact: do not label a run isolated
unless pinning was actually applied.

Measure peak resident memory with `resource.getrusage` when supported. If
the platform cannot supply a meaningful value, emit `status: unsupported`
with a diagnostic instead of a number. Use seed `20260811`. Hash exact
artifact file bytes with SHA-256 after generation.

Every completed run emits `result.json`, `provenance.json`, `arrays.npz`,
`plot.png`, `tolerances.yaml`, and `README.md`. Accuracy and performance
occupy separate `result.json` sections. A failed solver emits a structured
`blocked` result; it must not contain invented arrays, timings, metrics, or
convergence claims.
