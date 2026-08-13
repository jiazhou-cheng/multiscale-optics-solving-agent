# M3 slice protocol — `M3-SLICE-CPU-V1`

CHE-31 (M3.2). This document explains the frozen protocol in
`benchmarks/slice_protocol.yaml`, which is the machine-readable source of truth.
Every number here comes from `benchmarks/probes/m3_slice_feasibility.py` or from
a cited M1/M2 result.

**Purpose:** decide whether the ray → wave → Chromatix slice is executable, and
on what configuration, *before* any of it is wired. Four tickets of plumbing is
an expensive way to discover an arithmetic impossibility.

---

## What the feasibility analysis found

The expected binding constraint was the coupler's per-axis Nyquist limit forcing
an unaffordable grid. It was not.

| Candidate constraint | Result |
|---|---|
| Coupler cost (rays × pixels) | **not binding** — measured ~5.5e8 ray-pixel products/s; both grids reconstruct in < 0.5 s |
| Memory | **not binding** — largest field is 1.0 MB at `complex128` |
| Per-axis Nyquist grid size | **not binding** — 188 and 254 points per side |
| **Chromatix's `complex64` cast vs. propagation distance** | **binding** — rejected the first candidate system outright |

### The binding constraint

`chromatix.core.field.ScalarField.__init__` casts unconditionally to
`complex64`. The ASM transfer-function phase is `2πz·sqrt(1/λ² − f²)`, whose
magnitude is set by `2πz/λ`. Rounding a large phase argument in float32
perturbs the *differences* between spectral components, and those differences
are what form the PSF. So the relative field error should grow as
`ε₃₂ · 2πz/λ`. Measured against an independent float64 angular-spectrum
implementation:

| `z` | transfer phase (rad) | predicted `ε₃₂·φ` | measured field error | ratio |
|---|---|---|---|---|
| 0.04 mm | 4.57e2 | 5.4e-5 | 2.5e-5 | 0.46 |
| 0.4 mm | 4.57e3 | 5.4e-4 | 1.7e-4 | 0.31 |
| 4 mm | 4.57e4 | 5.4e-3 | 5.3e-3 | 0.96 |
| 47 mm | 5.38e5 | 6.4e-2 | 6.3e-2 | **0.98** |

The model is confirmed rather than fitted — nothing was tuned to make those
columns agree. M1's verified ASM evidence sits at 40 µm, the top row, which is
four orders of magnitude short of a 47 mm pupil-to-focus distance.

**Consequence:** a pupil-to-focus distance is the lens's back focal length, so
this makes the reference system's *absolute size* a protocol decision, not just
its f-number. That was not anticipated when M3 was planned.

### What was rejected, and why the fix is not a fudge

The first candidate was a plano-convex singlet with a 25 mm radius: EFL
48.4 mm, pupil-to-focus 47.1 mm, measured float32 field error **6.3e-2** —
200× M1's own float32 figure for this engine. Rejected.

Scaling the same prescription to 1/10 at fixed f-number is the fix, and it
improves three things at once for one reason: at fixed f-number, geometry scales
linearly while the wavelength does not.

- Propagation distance falls 10×, so the float32 error falls 10×.
- Spherical aberration *in waves* falls 10× (it scales as `h⁴/R³`, i.e. linearly
  in scale at fixed `h/R`), so the system becomes *more* diffraction-limited.
- Numerical aperture is unchanged, so the Nyquist pitch is unchanged and the
  grid shrinks only because the pupil is smaller.

Nothing was traded away. Both selected systems then land in the same
few-millimetre propagation regime — including `ReverseTelephoto`, whose EFL is
2.0 mm, so M1's chosen sample was already scale-appropriate.

---

## The two systems

| | `M3-SINGLET-REF` | `M3-REVERSE-TELEPHOTO` |
|---|---|---|
| Role | primary verification vehicle | real aberrated case |
| Construction | plano-convex singlet, real refractive surfaces | bundled Optiland sample, M1-validated |
| EFL | 4.837 mm | 2.005 mm |
| f-number | 9.7 | — |
| NA (measured, marginal ray) | 0.0517 | 0.0753 |
| Exit pupil ⌀ | 0.4987 mm | 0.4605 mm |
| Pupil → focus | 4.706 mm | 3.055 mm |
| Wavefront error | **0.017 waves P-V**, Strehl 0.9991 | aberrated by construction |
| Airy radius | 12.97 µm | not applicable |
| Analytic oracle | **Airy** | none — FFT oracle only |

The singlet is deliberately taken at f/9.7 rather than at its Rayleigh-limit
aperture (f/6 at this scale). At the Rayleigh limit the residual aberration
would be comparable to the effect being measured; at f/9.7 it is λ/59, so
M3.8's Airy comparison measures the slice rather than the singlet.

Why the aberrated case gets no Airy oracle: a real multi-surface objective is
not aberration-free, so an Airy pattern is not its expected PSF. Only an
independent implementation over the *same* wavefront can bound it.

### Wavefront error is measured to a sphere, not to the image plane

`W_i = OPL_i(at the rear vertex) + |X_i − F|`, referenced to its mean, with `F`
the nominal focus. Measuring optical path to the image *plane* instead would mix
plane-versus-sphere geometry into the number and inflate it: at NA 0.05 and a
5 µm spot that error is ~0.4 waves, larger than the aberration being measured.

---

## Handoff plane

**The exit pupil, read from `optic.paraxial.XPL()` / `XPD()` — not constructed.**
Optiland exposes exit pupil location and diameter directly, which the M3 plan
had listed as an open question. Rays travel in air after the last surface, so
projecting each ray from the image plane to the pupil plane along its own
direction is exact rather than approximate.

It is a **plane**, not a reference sphere: `C_RAY_TO_WAVE` accumulates plane
wavelets onto a plane. `examples/graphs/ray_to_wave.yaml` carried
`reference_sphere: exit_pupil`, which nothing implements; that file is now
annotated with what executes in M3 and what is M4 scope.

---

## Sampling

The Nyquist limit is **per axis**:

```
pitch_axis <= lambda / (2 * max_over_rays |direction_cosine_axis|)
```

evaluated over the marginal rays of an actual trace. M2 fixed this after CHE-24
tested the direction *norm*, which both over- and under-constrains: a diagonal
FFT bin has `|d| = √2·λ/(2·pitch)` yet is exactly representable.

Both systems are traced through `Optic.trace`, i.e. the hexapolar 2-D pupil the
adapter actually samples, so both axes are exercised. A 1-D pupil sweep would
have left the y-axis limit untested — and untested is how a per-axis bug
survives.

`2×` oversampling is used. At the critically admissible pitch the Airy radius
spans only ~2.44 pixels, enough to locate a peak and a first null but not to
compare a profile; at 2× it spans 4.88.

| | pitch max (Nyquist) | pitch used | pupil extent | grid |
|---|---|---|---|---|
| `M3-SINGLET-REF` | 5.317 µm | 2.659 µm | 499.6 µm | **188** |
| `M3-REVERSE-TELEPHOTO` | 3.652 µm | 1.826 µm | 462.0 µm | **254** |

Ray count starts at 4096 from a one-ray-per-Nyquist-cell criterion and is
**tested, not assumed**, by M3.9. Optiland's `num_rays` is a hexapolar density
request rather than an output count, so the criterion applies to the traced
survivor count.

---

## Tolerance budget

Built from named error sources. Deliberately *not* read off a passing run —
M3.8 tests against these, so deriving them from observed output would make the
comparison circular.

| Term | Value | Level | Source |
|---|---|---|---|
| Coupler float64 round-off | 1e-12 | field, rel. | M2 measured 7.82e-14 at 16×16; restated with headroom |
| **Chromatix `complex64`** | **3.5e-4** | intensity, rel. | measured here, at each system's own `z` and NA |
| — same term, field level | 3.4e-3 | field, rel. | recorded because a field consumer pays it |
| Reference-system residual aberration | 9.1e-4 | Strehl deficit | measured; *not an error* |
| Ray sampling | to be measured | intensity, rel. | M3.9 |
| Grid truncation / padding | to be measured | intensity, rel. | M3.6 |

**Field versus intensity differ by 10×, and it matters which one is quoted.**
Most of the float32 phase error is common to every spectral component, so it
acts as a piston that cancels when the field is squared. The PSF is an
intensity, so 3.5e-4 is the budget term. Quoting only that figure would hide
the field-level cost from anyone who later wants the complex field itself, so
both are recorded.

That 3.5e-4 agrees with M1's declared 3.00e-04 for this engine, which was
derived independently from `5·ε₃₂` per radian. Two routes to the same number.

**The residual aberration term is a physical property, not a defect**, and it
belongs in exactly one place: the Airy comparison, because Airy is the
perfect-lens limit. It must *not* appear in the FFT-oracle tolerance, because
that oracle is built from the same wavefront and therefore contains the same
aberration.

### Gates

| Gate | Value | Composition |
|---|---|---|
| Airy peak intensity, relative | 2.0e-3 | 9.1e-4 aberration + 3.5e-4 float32 + margin |
| FFT oracle intensity, relative L2 | 1.0e-3 | 3.5e-4 float32 + margin; no aberration term |
| Unexplained energy residual | 1.0e-3 | bounds the *unattributed* remainder, not total loss |

No gate may be satisfied by widening it. A failing gate is a finding to
diagnose and report — M2 exited with no gate satisfied by loosening a tolerance,
and M3 inherits that rule.

---

## Inadmissible sources

**`surface_type="paraxial"` must not supply a wavefront or OPL to a coupler**
(CHE-30). The paraxial interaction model subtracts `(x²+y²)/(2f)` but leaves the
direction un-normalized, so the following propagation adds the axial distance
rather than the Euclidean one; the intended cancellation never happens and the
subtraction survives in full. Measured OPL at the focus is `f − h²/(2f)`:
0.36 mm, about **655 waves**, at `f = 50 mm`, `h = 6 mm`. That is a defocus.
This is why the reference system is built from real refractive surfaces.

**An undeclared Optiland OPL stays refused.** `OPL_REFERENCE_UNVERIFIED` remains
enforced. Because the infinite-object reference plane is aperture-dependent
(CHE-30), a declaration must record the EPD it was taken at.

---

## Reproducing this analysis

```bash
./run.sh python benchmarks/probes/m3_slice_feasibility.py
./run.sh pytest tests/test_m3_slice_protocol.py -q
```

Timings are same-machine relative figures on a shared, unpinned host, as in M1
and M2. They exist to reject infeasible configurations, not as a regression
envelope.
