# M3 slice protocol — `M3-SLICE-CPU-V1`

CHE-31 (M3.2), amended by CHE-40 (M3.2A). This document explains the frozen
protocol in `benchmarks/slice_protocol.yaml`, which is the machine-readable
source of truth. Every number here comes from
`benchmarks/probes/m3_slice_feasibility.py`,
`benchmarks/probes/m3_carrier_phase.py`, or a cited M1/M2 result.

**Purpose:** decide whether the ray → wave → Chromatix slice is executable, and
on what configuration, *before* any of it is wired. Four tickets of plumbing is
an expensive way to discover an arithmetic impossibility.

> **Amendment A1 (CHE-40).** M3.2 concluded that the optical system's absolute
> *size* was a binding numerical constraint and scaled the reference
> prescription to 1/10 on that basis. The measurement was right; the attribution
> was wrong. The error came from propagating a ~5.4e5 rad absolute carrier phase
> that contributes nothing to a single-path PSF, not from the `complex64` engine.
> Removing the carrier drops the 47 mm intensity error from 8.2e-3 to 3.9e-6 with
> nothing else changed. **Absolute optical-system scale is not a binding
> constraint on a carrier-conditioned path**, the 1/10 scaling is now a cost
> choice rather than a requirement, and carrier-conditioned propagation is
> *mandatory* for phase-insensitive M3 PSF paths. See
> [What CHE-40 changed](#what-che-40-changed).

---

## What the feasibility analysis found

The expected binding constraint was the coupler's per-axis Nyquist limit forcing
an unaffordable grid. It was not.

| Candidate constraint | Result |
|---|---|
| Coupler cost (rays × pixels) | **not binding** — measured ~5.5e8 ray-pixel products/s; both grids reconstruct in < 0.5 s |
| Memory | **not binding** — largest field is 1.0 MB at `complex128` |
| Per-axis Nyquist grid size | **not binding** — 188 and 254 points per side |
| ~~Chromatix's `complex64` cast vs. propagation distance~~ | **superseded by CHE-40** — binding only on an unconditioned path |

### The constraint M3.2 found binding

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

## What CHE-40 changed

The reasoning above is internally consistent and its measurements reproduce. It
is also built on an assumption that was never stated as one: that the error
belonged to the wave engine. It belonged to the number.

The exact transfer function factors, with no approximation:

```
exp(i z k_z)  =  exp(i k z) · exp(i z (k_z − k))
```

The first factor is constant over the whole spectrum. It is a global piston: it
cannot change intensity, and along a single propagation path it cannot change
relative phase either. Only the second factor diffracts. Its magnitude on the
M3 grids is `max |z(k_z − k)|` — 578 rad for `M3-SINGLET-REF`, against `kz` of
5.4e4. **Ninety-three times smaller, and it is the number float32 has to round.**

Evaluating it without cancellation needs the exact identity

```
k_z − k  =  −(k_x² + k_y²) / (k_z + k)
```

which is algebra, not a paraxial expansion — `tests/test_carrier_removed_asm.py`
pins the equality in float64 and separately pins that a paraxial substitution
would be caught.

### Measured, at fixed prescription, NA, grid, oracle, and input field

| `z` | current, raw field | current, piston-aligned | current, intensity | carrier-removed, piston-aligned | carrier-removed, intensity |
|---|---|---|---|---|---|
| 0.04 mm | 2.5e-5 | 1.3e-5 | 6.7e-6 | **2.0e-7** | **2.1e-7** |
| 0.4 mm | 1.7e-4 | 1.6e-4 | 6.9e-5 | **2.9e-7** | **2.4e-7** |
| 4 mm | 5.4e-3 | 1.4e-3 | 7.2e-4 | **2.4e-6** | **1.2e-6** |
| 47 mm | 7.0e-2 | 2.1e-2 | 2.4e-2 | **2.4e-5** | **2.8e-5** |
| 470 mm | 6.8e-1 | 1.8e-1 | 1.9e-1 | **2.5e-4** | **2.7e-4** |

The current path's raw error tracks `kz` across three decades, as M3.2 modelled.
The carrier-removed path's error tracks `max |z(k_z − k)|` instead, which is what
it now represents — and at 470 mm, ten times past the distance that rejected the
first system, it is still inside M3.2's own 3.5e-4 intensity term.

**The rejected system, re-examined on its own geometry.** Its 4.987 mm exit pupil
propagated the full 47.06 mm to focus, 2048² at its own 2.659 µm pitch:
`4.49e-4` relative intensity error on the absolute-phase path — over M3.2's
3.5e-4 term, so **M3.2's rejection was correct for the propagation it had** —
against `2.21e-7` carrier-conditioned, a factor of 2030. The float64 oracle, the
current path, and the carrier-removed path all agree on the focal core's
departure from the far-field Airy limit to within 1e-4 relative, which is what
makes that 3.5e-2 departure attributable to finite-distance Fresnel physics
rather than to any implementation.

### The same mechanism shows up in float64, nine orders down

Absolute-phase and carrier-removed ASM must agree exactly. In float64 they do
not quite, and the residual is not noise:

| `z` | piston-aligned difference | `eps64 · kz` |
|---|---|---|
| 0.04 mm | 2.7e-14 | 1.0e-13 |
| 0.4 mm | 3.3e-13 | 1.0e-12 |
| 4 mm | 3.1e-12 | 1.0e-11 |
| 47 mm | 4.4e-11 | 1.2e-10 |
| 470 mm | 3.6e-10 | 1.2e-9 |

A constant fraction of `eps64 · kz` at every distance. Representing the absolute
carrier costs `eps · kz` in whatever precision you have; float32 is not special.
This is also why the ticket's flat `1e-12` equivalence target is unreachable
beyond a few millimetres and the acceptance criterion is stated against the floor
instead — the failure of the flat target is evidence *for* the hypothesis.

### Consequences, all of them binding

1. **Absolute optical-system scale is not a protocol constraint** on a
   carrier-conditioned path.
2. **The 1/10 scaling is a safe fallback, not a requirement.** It stays as the
   frozen primary for a different reason: 188² versus 2048² is ~120× the pixels
   for the same physics, and M3.9 sweeps grid and ray count on top of that. The
   choice is now a budget decision.
3. **The unscaled ~48 mm singlet is reinstated as admissible**, with measured
   numbers, for any later ticket that wants a macroscopic case. Not on an
   unconditioned path.
4. **The tolerance budget distinguishes three levels** — absolute field phase,
   piston-aligned field, intensity — because on a conditioned path they no longer
   collapse into one number. See below.
5. **Carrier-conditioned propagation is required**, not offered, for
   phase-insensitive M3 PSF paths.

### Global-phase policy

The removed `exp(i k z)` is **retained as float64 metadata and never reapplied**.
Folding it back into a `complex64` field would reintroduce precisely the rounding
it was removed to avoid, so there is no convenience path that quietly undoes the
fix. A consumer needing absolute optical phase calls
`reconstruct_absolute_phase`; a consumer needing intensity or single-path
relative phase needs nothing. **No consumer may read absolute optical phase off
the propagated field.**

One trap worth naming: `chromatix` stores `Field.spectrum` in float32, so a
carrier phase derived from the field itself is good only to ~3e-8 relative —
about 0.02 rad at 47 mm, larger than everything carrier removal buys back.
`carrier_removed_asm_propagate` therefore takes an explicit `wavelength_m` and
records which source was used in `wavelength_source`.

### The same rule on the ray side

`C_RAY_TO_WAVE` has the identical exposure, and CHE-30 already measured the
quantity: `RealRays.opd` is absolute accumulated optical path from the ray launch
state, so at a few tens of millimetres it is ~1e5 waves. Coherent phase must
therefore be formed as

```
phi_i = (2π/λ) · (OPL_i − OPL_ref)
```

never from absolute accumulated OPL. This protocol fixes only that the reference
subtraction must happen before phase is formed; the choice of `OPL_ref`, the sign
convention, and global-phase provenance belong to the M3.4 (CHE-33) contract.
Recorded, not implemented.

Reproduce with:

```bash
./run.sh python benchmarks/probes/m3_carrier_phase.py
./run.sh pytest tests/test_carrier_removed_asm.py -q
```

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
| **Chromatix `complex64`, carrier-conditioned** | **1.0e-4** | intensity, rel. | `ε₃₂·max\|z(k_z−k)\|`: 6.89e-5 and 9.55e-5 on the two systems' own grids |
| — same term, piston-aligned field | 1.0e-4 | field, rel. | same derivation; see below for why it is not 10× the intensity term |
| — same term, absolute field phase | not preserved | — | the conditioned path discards the carrier by design |
| Reference-system residual aberration | 9.1e-4 | Strehl deficit | measured; *not an error* |
| Ray sampling | to be measured | intensity, rel. | M3.9 |
| Grid truncation / padding | to be measured | intensity, rel. | M3.6 |

**Three levels, because on a conditioned path they stop collapsing into one.**
M3.2 quoted 3.5e-4 intensity against 3.4e-3 field and explained the 10× gap
correctly: most of the float32 phase error was a common piston, which cancels
under squaring. Removing the carrier removes that piston up front. What remains
is per-bin error, which does *not* cancel, so the gap does not carry over and the
intensity term is bounded **at** the piston-aligned field term rather than at a
tenth of it.

Absolute field phase is a third, separate thing: the conditioned path does not
preserve it at all. Taken from the absolute-phase path instead it would cost
`ε₃₂·kz` = 6.4e-3 at `M3-SINGLET-REF`'s 4.706 mm. Stating it as "not preserved"
rather than as a number is the point — a consumer must reconstruct it
deliberately, in float64, or not claim it.

Both figures are derived from the phase the transfer function has to represent,
not read off a passing run: the measured errors sit 20–50× below the bound, and
M3.8 tests against the bound. M3.2's 3.5e-4 / 3.4e-3 pair is retained in the YAML
as the fallback budget for any path that propagates the absolute carrier, and it
still agrees with M1's independently derived 3.00e-04 for that case.

**The residual aberration term is a physical property, not a defect**, and it
belongs in exactly one place: the Airy comparison, because Airy is the
perfect-lens limit. It must *not* appear in the FFT-oracle tolerance, because
that oracle is built from the same wavefront and therefore contains the same
aberration.

### Gates

| Gate | Value | Composition |
|---|---|---|
| Airy peak intensity, relative | 2.0e-3 | 9.1e-4 aberration + 1.0e-4 conditioned `complex64` + margin |
| FFT oracle intensity, relative L2 | 1.0e-3 | 1.0e-4 conditioned `complex64` + margin; no aberration term |
| Unexplained energy residual | 1.0e-3 | bounds the *unattributed* remainder, not total loss |

CHE-40 cut the `complex64` term from 3.5e-4 to 1.0e-4 but **the gates were not
tightened to match**, because `ray_sampling_error` and
`grid_truncation_and_padding` are still unmeasured and have to fit inside the
same gates. Tightening now would be setting a tolerance against terms nobody has
measured.

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
