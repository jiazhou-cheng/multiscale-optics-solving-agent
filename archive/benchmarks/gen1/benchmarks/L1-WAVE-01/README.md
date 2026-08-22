# L1-WAVE-01 — Chromatix wave-propagation accuracy

Generate the analytic reference and evaluate every CPU case in one command:

```bash
./run.sh python benchmarks/level1/L1-WAVE-01/evaluate.py --case gaussian
```

Reproduce the standalone, analytic, and scaling evidence as one independent
M1 branch bundle with:

```bash
./run.sh python benchmarks/level1/L1-WAVE-01/run_all.py \
  --output-dir outputs/M1/wave
```

This wrapper rejects scientific artifact corruption, emits a canonical
scientific fingerprint, and does not import Optiland or any coupler.

The analytic oracles can also be built and reviewed on their own, without
Chromatix or JAX being imported at all:

```bash
./run.sh python benchmarks/level1/L1-WAVE-01/generate_reference.py
```

The benchmark runs no ray model and no coupler, and fails if `optiland` or
`multiscale_optics_agent.couplers` is in `sys.modules` at exit.

Run the CHE-15 fixed-Gaussian grid, automatic-padding, determinism, and CPU
scaling section separately:

```bash
./run.sh python benchmarks/level1/L1-WAVE-01/run_scaling.py --device cpu
./run.sh python benchmarks/level1/L1-WAVE-01/evaluate.py --section scaling
```

CHE-15 still names a Gaussian scaling case although the current CHE-18 review
redesign below replaced its original Gaussian suite. The scaling command keeps
the original fixed Gaussian radius/power gates and also embeds this unchanged
CHE-18 suite as a mandatory independent accuracy gate; neither accepted record
is silently rewritten.

## The progression

The suite climbs deliberately from a case with **no approximation anywhere**
to one where the approximations are the interesting part. Each rung is only
meaningful because the one below it passed.

| | Case | Physics | Oracle | Solver path |
|---|---|---|---|---|
| 1 | Exact homogeneous primitive | FFT-bin plane-wave eigenmodes, unpadded | `u · exp(i k_z z)` — **exact** | `run_standalone` (scalar ASM) |
| 2 | Signed paraxial focusing | Rectangular pupil + ideal thin lens + 3 signed tilts | Fresnel/Fourier separable `sinc` | `run_standalone` (scalar ASM) |
| 3 | High-NA vectorial focusing | x-polarized aplanatic objective, NA 0.9 | float64 Richards–Wolf quadrature | `chromatix_benchmark_adapter` (vector) |

Sampling, padding, reference-quadrature convergence, and negative
perturbations **qualify** the benchmark; they are not additional physical
cases.

## Case 1 — the exact primitive

A plane wave placed exactly on an FFT bin is an eigenmode of the discrete
angular-spectrum operator, so the propagated field is the input times
`exp(i k_z z)` with `k_z = (2πn/λ)√(1 − (λ/n)²|f|²)` — no paraxial expansion,
no window truncation, no interpolation. This must run **unpadded**: the mode
is periodic on the grid, and zero-padding it would manufacture an aperture
edge the physics does not contain.

Because nothing is approximated, the tolerance is **derived rather than
chosen**. The propagator evaluates `exp(i k_z z)` in single precision, so the
accumulated phase — which reaches ~2950 rad here — carries one float32
epsilon of relative error, giving a bound `∝ eps·|k_z z|`. Observed phase
error tracks that bound across a 25× range of propagation distance. A fixed
absolute tolerance would be either vacuous at long `z` or impossible at short
`z`.

Modes include an asymmetric `(m_y, m_x) = (−7, 13)` and a large-angle
`(25, 40)` at `sinθ = 0.39`, so axis order and the non-paraxial `√` are both
observable. Each case also propagates back to `z = 0` and must recover the
launched field.

## Case 2 — ideal signed paraxial focusing

Input `rect(x/L_x)·rect(y/L_y)·exp(−ik r²/2f)·exp(ik(θ_x x + θ_y y))`,
propagated one focal length. The lens quadratic phase cancels exactly under
Fresnel propagation, leaving a scaled Fourier transform of the pupil.

Gated metrics, all against the analytic oracle:

| metric | oracle | tolerance |
|---|---|---|
| focal centroid | **`+f·θ`** (signed) | 0.1 input px |
| FWHM x, y | `0.8859·λf/L` | 2 % |
| first sidelobe / peak | `0.047180` (pure shape number) | 5 % |
| complex overlap vs Fresnel | 1 | ≥ 0.99 |
| complex overlap vs independent float64 ASM | 1 | ≥ 0.9999 |
| power ratio | 1 | 1e-3 |

Two measurement decisions worth knowing: the aperture is specified in **odd
sample counts** (201 × 121) so it is exactly symmetric about the origin and
the sampled pupil matches the width the oracle assumes; and widths are FWHM,
**not** `D4σ`, because the second moment of a `sinc²` diverges and would
measure the window rather than the beam.

## Case 3 — high-NA vectorial focusing: BLOCKED

Case 3 is implemented in full — oracle, metrics, and qualification sweep — and
its result is that **Chromatix 0.6.0 cannot currently be validated on this
path**. It is reported but does not gate the benchmark.

The qualification is simple: refine the pupil sampling with every physical
parameter fixed. A physical focal field cannot move.

```
independent Richards–Wolf oracle        chromatix high_na_ff_lens
  nq=  25  Iz/Ix = 0.150087403            N=128  Iz/Ix=0.126  |Ez| ring r =  246 nm
  nq= 100  Iz/Ix = 0.150087403            N=256  Iz/Ix=0.161  |Ez| ring r =  246 nm
  nq= 400  Iz/Ix = 0.150087403            N=384  Iz/Ix=0.279  |Ez| ring r =  788 nm
  nq= 800  Iz/Ix = 0.150087403            N=512  Iz/Ix=0.366  |Ez| ring r = 1725 nm
                                          N=768  Iz/Ix=0.122  |Ez| ring r = 2536 nm
                                          oracle |Ez| ring r =  197 nm
```

The oracle converges to a relative `2e-14`; the solver's focal scale moves by
a factor of ten. Best achievable vector overlap is 0.070.

**Root cause**, read from the pinned source: `high_na_ff_lens` derives `s_z`
from `field.f_grid · λ/n` — the *frequency* grid — rather than from the pupil
position grid. On any physically sampled pupil `|s_grid| ≈ 0.015`, so
`s_z ≈ 1` and both the intended `1/cosθ` obliquity Jacobian and the
`exp(i k f cosθ)` defocus degenerate to constants; the `zoom_factor` that sets
the output scale comes from the same quantity.

What *is* correct there, and is recorded for whoever fixes this:
`cartesian_to_spherical` implements the standard Richards–Wolf aplanatic
polarization transform exactly, and the vector component order is
`(E_z, E_y, E_x)`. No `√cosθ` apodization is applied — the caller must supply
it, and this benchmark does.

A catalog plano-convex lens is deliberately not used anywhere in this suite,
for the same class of reason: Chromatix 0.6.0's `thick_plano_convex_lens` is
implemented through an ABCD matrix and `ray_transfer` rather than a
surface-resolved wave-optical prescription.

## Oracle boundary

Primary oracles live in `oracles.py`, which imports neither Chromatix nor JAX;
a test asserts that. No Chromatix-generated value is ever used as an
expectation — the recorded `propagation_probe.json` snapshot is regression
evidence only.

Because the Case 2 oracle is paraxial, every Case 2 case additionally reports
a float64 NumPy angular-spectrum propagation written from the definition.
That second independent implementation separates *the oracle's approximation*
from *the solver's error*: see `error_attribution.json`, which splits each
case into `discretization_and_window`, `paraxial_model`,
`solver_implementation`, `normalization`, and `convention`, per axis. The
result is that Chromatix matches the independent float64 propagation to
`1e-6` while the paraxial oracle differs from both by up to `1.3e-2`.

## Negative perturbations

Each has an unperturbed control that must pass, and counts as detected only at
10× its metric's tolerance.

| perturbation | what it corrupts | observed | threshold |
|---|---|---|---|
| `case1_paraxial_dispersion` | exact `√` dispersion → Taylor expansion | 3.1 rad | 0.016 rad |
| `case2_lens_sign_flip` | converging → diverging lens | 0.94 | 0.10 |
| `case2_axis_transpose` | `(y, x)` order | 0.71 | 0.10 |
| `case2_si_scale` | wavelength in µm as if metres | 0.98 | 0.10 |

## Emitted bundle

`result.json`, `provenance.json`, `arrays.npz`, `reference.json`,
`reference_fields.npz`, `input_config.yaml`, `tolerances.yaml`,
`error_attribution.json`, `solver_summaries.json`, a six-panel `plot.png`, and
this file. Generated scientific values are never hand-edited.
