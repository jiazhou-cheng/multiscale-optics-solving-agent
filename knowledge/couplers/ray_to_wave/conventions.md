# C_RAY_TO_WAVE — conventions

The algebra in `theory.md` is short. Essentially every way this coupler goes
wrong is in this file instead.

## Frozen conventions

Inherited from the M1 baselines and re-asserted at the coupler boundary. M1's
exit report states these directly: *"The conventions that M1 pinned are the
coupler's contract."*

| Item | Convention |
|---|---|
| Units | SI throughout. The coupler core accepts and emits metres, radians, and seconds only |
| Field array order | `(y, x)` |
| Coordinate origin | Array index `n//2` is coordinate zero on each spatial axis |
| Frame | Right-handed Cartesian, propagation along `+z` |
| Reconstruction plane | Declared explicitly: axial coordinate and unit normal `n̂` |
| Wavelength | Metres, monochromatic. Each evaluation is single-wavelength, as the paper states |
| Time convention | `exp(−i ω t)` |
| Spatial factor | `exp(+i k z)`; ray wavelet phase `exp(+i k·OPL)` |
| Complex arrays | Amplitude, never intensity. Intensity is `\|u\|²` |
| Normalization | Declared per call. No `1/N` when summing a *given* ray ensemble; `1/N` when the ensemble is a Monte Carlo sample of a spectrum |
| Launch amplitude scale | `a = √w · dA`, with `dA` the ray's physical quadrature area element in m². Ray-density-independent, but **not** SI-absolute: the kernel omits the diffraction prefactor, so `U` is `iλz` times the SI field — see below |
| Polarization | Scalar only in M2. A polarization basis must be declared before any vector claim |
| Coherence | Fully coherent. Partial coherence is not modelled |

The phasor pair is the one to get right first, because a sign error there
produces a plausible-looking field with a mirrored wavefront. Chromatix's own
conventions note is blunt about it: *any coupler joining Chromatix to a solver
with an explicit time convention must verify sign agreement with a manufactured
traveling-wave test, not assume it.*

## Four hazards

H1–H3 are *inherited*: each was recorded by M0 or M1 and was unresolved at M2
open. **Status today (CHE-69):** H1 is *resolved* — CHE-30 and CHE-41
characterized it, and the refusal it motivated survives for a different reason,
stated below. H2 and H3 are not "unresolved" so much as permanent properties of
the engines; they are handled explicitly at the boundary rather than fixed.

H4 is *this kernel's own*, found by CHE-70 and open.

### H1 — Optiland `opd_native` sign and reference — **characterized (CHE-30/CHE-41)**

*As recorded at M2 open:* M1 recorded `opd_native` with
`opd_reference: "unverified"` and `opd_sign: "unverified"`, and deliberately
declined to interpret a probe that returned `opd = 12` for a 10 mm separation.
M1's exit report carried this forward as item 4 of what M2 inherits.

**Resolution for M2:** `opd_native` is *not* an admissible `OPL⁽ⁱ⁾` source.
The coupler core requires an OPL in metres whose reference plane the caller
declares.

**What changed since, and what did not (CHE-69).** The characterization is
*done*. CHE-30 (M3.1) established `opd_native`'s sign, units, physical meaning,
reference and behaviour under known axial propagation against manufactured
geometries with closed-form answers — every case exact to float64 round-off,
with a falsifiable negative test — and the Optiland solver card was promoted from
`unverified` on that evidence. CHE-41 then closed the off-axis half, where the
declared pupil OPL omitted the field tilt; the OPL reference is versioned rather
than merely asserted. The forward slice runs on that work.

**The refusal stands, for a different reason than it started with.** It is no
longer "nobody has characterized this yet" — it is that a *native accumulator is
not a declared physical quantity*. The adapter carries `opd_native` in
provenance and never promotes it to an OPL; it emits a declared OPL with a
versioned reference instead, and `OPL_REFERENCE_UNVERIFIED` is the structured
refusal a caller gets for passing the raw value. Read `deliberately_refused`
on the coupler card as a design decision, not as an admission of ignorance.

Why refusal rather than a default: a wrong OPL *reference* is a constant piston
and mostly harmless; a wrong OPL *sign* conjugates the wavefront, turning a
converging beam into a diverging one. Those two failure modes are
indistinguishable downstream, so the ambiguity cannot be allowed through. That
argument is unaffected by the characterization — it is why the *caller* declares
the reference, rather than why the value was refused pending a probe.

### H2 — Optiland `intensity` is a weight, not a complex amplitude

The Optiland adapter emits `intensity` with an explicit
`intensity_is_not_amplitude` marker, and declares `polarization: "missing"` and
`coherence: "missing"`.

Meanwhile the registry's `C_RAY_TO_WAVE` source port declares
`requires_metadata: [wavelength, coordinates, optical_path_length, amplitude,
polarization]` — two of which the Optiland adapter documents that it refuses to
fabricate. This mismatch is recorded in `registry/models.yaml` itself.

**Resolution for M2:** the contract is corrected rather than the adapter. The
coupler requires a complex amplitude; if a caller has only a real ray weight
`w`, the conversion to an amplitude is a *modelling decision* (is `w` power, so
`a = √w`? a photon count? already an amplitude?) that must be declared and
tested by the caller. The coupler does not choose. `polarization` is dropped
from the required metadata for the scalar M2 path and reinstated when a vector
claim is made.

### H3 — Chromatix `asm_propagate` returns padded arrays

M1 measured a 256² input growing to a 1756² output, and recorded that padding,
not the input grid, drives the memory cost.

**Resolution for M2:** pitch and extent are explicit on every `ComplexField`
crossing the boundary, and the coupler never infers an extent from an array
shape. A field handed to or received from the wave model records its pad width
and whether it has been cropped.

### H4 — near-grazing modes lose their phase to cancellation — **open (CHE-70)**

The kernel forms each ray's constant phase as `k(OPL − d·x₀)`. For a mode whose
axial direction cosine is `dₙ`, propagating a distance `Z` makes **both** terms
scale as `Z/dₙ` while their difference is only `Z·dₙ`. So the *relative* precision
of the inputs sets the *absolute* error of the phase:

```
Δφ  ~  ε · k · Z / dₙ
```

CHE-70 measured this on a 100×100, 250 nm-pitch, 500 nm grid. Eight bins land on
`d_u² + d_v² = 1` exactly — the (30, 40) and (40, 30) Pythagorean triples and their
sign variants — and survive the strict `radial < 1` evanescent cut at
`dₙ = 1.05e-8`. Over a 50 µm propagation their OPL is **4745 m**, they carry
`2.25e-7` of the field's power, and under `p_uni` they are drawn at full
importance weight.

| band limit | exactness limit vs an analytic angular-spectrum oracle, float64 |
|---|---|
| none | `2.8e-09` |
| `dₙ ≥ 1e-2` | `8.9e-14` |

In float32 the same 4745 m OPL is a phase of `6e10` rad, whose representation error
alone is `~7e3` rad. Those bins are then pure noise, so this is a **correctness
requirement for any float32 path**, not a tidying step.

**Why it survived M2 and M3.** It is invisible unless three things coincide: a
spectrum wide enough to reach grazing (M3's pupil bundles are within a few degrees
of the axis), a propagation long enough for `Z/dₙ` to dominate the float budget,
and a *complex-field* comparison — the affected bins carry too little power to move
`|U|²`.

**Current handling (CHE-70): declared band limit at the caller, not a kernel
change.** `couplers.streaming.grazing_floor_for_phase_budget` derives the floor
from the compute precision and the axial extent
(`dₙ ≥ ε·k·Z / Δφ_budget`; float32 over 50 µm at 0.01 rad gives `7.49e-3`, frozen
at `1e-2`), `band_limit_spectrum` applies it to the ray ensemble, and the
comparison oracle applies the identical mask so both routes carry the same modes.
The excluded bin count and power fraction travel with the artifact.

**Not resolved.** Whether the coupler should refuse such modes itself, band-limit
them itself, or reformulate the constant phase so the cancellation does not occur
is a kernel question with its own oracle, and it needs its own ticket — the same
disposition CHE-50 took. Any caller propagating a wide-angle spectrum through this
kernel will meet it, and nothing in the kernel currently warns them.

Evidence: `benchmarks/reports/2026-08/metalens_bridge.md` §6,
`tests/test_coherent_bridge.py::TestExactnessLimit`.


## Launch amplitude carries the area element, and that fixes the absolute scale

**The convention: `amplitude = sqrt(ray_intensity) * quadrature_weight_m2`, with
`quadrature_weight_m2` the physical area in m² that the ray's quadrature cell
represents.** Producer-side, in `couplers.handoff.declare_coherent_bundle`; the
`C_RAY_TO_WAVE` kernel is unchanged by it and still sums whatever amplitude the
bundle declares.

### Why the area element, and not a bare relative weight

The wavelet sum

    U(r) = Σᵢ aᵢ · exp[i k (OPLᵢ + drᵢ(r))]

is a discretization of a surface integral over the aperture, so `aᵢ` must carry
the area element that integral is taken with. The consequence that was actually
measured is numerical: `Σᵢ dAᵢ → π a²` as the ring count grows, so the
reconstructed discrete power **converges under ray refinement**. Without it the
sum is pinned to the ray count instead of to the aperture, which is CHE-33's
measured `N^2.0024` raw-power scaling. CHE-47 closed that by introducing this
weight.

### What the scale is, and what it is not

It is **ray-density-independent**. It is **not** an SI power, and reporting
`propagated_power_out` in watts would be wrong by about 18 orders of magnitude.
Two factors are missing:

* The kernel above has **no `1/(iλz)` Kirchhoff prefactor** — it sums wavelets
  and stops. So `U` is `iλz` times the SI field.
* The incident amplitude density `A₀` is never declared; `√w` is Optiland's
  bookkeeping weight, which on the frozen M3 systems is identically 1.

The first factor has a clean analytic check, and the repository already records
it. Stationary phase on a converging bundle gives `∫dA·exp(ik|r′−r|²/2R) = iλR`,
so reconstructing a unit-amplitude-density pupil should return `|U| = λR`.
`aperture_edge_hypothesis.edge_vs_ray_count[*].plateau_amplitude` in
`benchmarks/probes/records/m3_convergence.json` converges to `2.675e-9` against
`λR = 0.55e-6 × 4.8375e-3 = 2.6606e-9` — a ratio of 1.005, at the ~0.5% level
the rest of that record sits at. Treat that as the oracle for this convention:
if the reconstructed pupil plateau stops matching `λR`, the launch amplitude
scale has moved.

So `propagated_power_out` is a **relative** quantity. It is comparable between
two runs of the same configuration, and it is not a physical power.

A hexapolar fan's cell is `π a² / (3 n²)` in the interior, `3/4` of that for the
central ray, and `1/2` for the rim ring, which sits on the aperture boundary and
represents only the inner half of its annulus
(`couplers.quadrature.hexapolar_area_weight_m2`).

### The scale this replaced, and how to recognize it

Before CHE-47 (`ec55839`) the launch amplitude was `√w` alone, and on the
frozen M3 configurations `w ≡ 1`, so **every ray launched with amplitude exactly
1**. The absolute scale of everything downstream therefore differs between the
two conventions by `dA²` — about `3.9e-21` on `M3-SINGLET-REF`, whose per-ray
cell is `6.2e-11 m²`.

That factor is the signature to look for. A propagated power of `7.0e-04` on
this system is the **pre-CHE-47** convention; `2.7e-24` is the current one. Any
committed evidence still carrying the former was produced before the weight
existed. CHE-103 found three such records and regenerated them.

Peak-normalized metrics cannot see this, which is exactly why it survived: the
frozen M3 oracle normalization divides by the peak, so a global scale cancels
and only the *raw* fields distinguish the two conventions.

### What the weight is not

It is not an apodization, and it is not a substitute for one. The weight is
fixed by how the pupil was sampled, not by the physics of the aperture, so a
non-uniform launch amplitude arising from it does **not** exercise the
reconstruction's response to a physically apodized, vignetted or Fresnel-weighted
pupil. Those remain untested (CHE-103; `amplitude_degree_of_freedom` in
`benchmarks/probes/records/m3_psf_verification.json`).

It also should not be blamed for sampling artifacts. On the frozen
configuration the weight appears to shift the measured first-null radius by
3.6%; refining the grid shows that is the grid's 2.44 pixels per Airy radius,
not the weight (`benchmarks/probes/records/m3_first_null_grid_convergence.json`).

What it *does* do, and what is not yet explained, is off-axis. On the
`ReverseTelephoto` field the reconstruction's residual against the analytic
Airy profile went from `1.48e-3` to `1.11e-2` when the weight was introduced,
and removing the weight again improves that residual by 7.5x
(`off_axis_negative_controls` in
`benchmarks/probes/records/m3_psf_verification.json`). On axis the same metric
barely moved, `5.87e-3 → 5.51e-3`. The rim taper is a plausible cause — a
uniform-pupil Airy oracle sees a tapered pupil as a mismatch, and an off-axis
pupil is sampled asymmetrically — but that is a hypothesis, not a measurement.
Owned by M2.1 (CHE-109), which sets the ray→wave error budget.

## Sign and orientation checklist

Run these before believing any reconstruction. Each has a negative-test twin in
`probes/`.

1. **Phasor sign.** A wavelet travelling along `+z` must gain phase, not lose
   it. A collimated bundle at normal incidence reconstructed at two planes
   separated by `Δz` must differ by `exp(+i k Δz)`.
2. **Tilt direction.** A bundle with `d̂ = (sin θ, 0, cos θ)` must produce a
   field whose phase increases with `+x` for `θ > 0`. Getting this backwards is
   the classic `(y, x)`-vs-`(x, y)` transposition, which a rotationally
   symmetric test case will never catch — so the test case must be
   axis-asymmetric.
3. **Projection factor.** Reconstructing the same physical bundle at increasing
   incidence angle must show the `⟨n̂, d̂⟩` reduction. Omitting it is
   undetectable at normal incidence, which is exactly why the check must be run
   off-axis.
4. **Units.** A metre-for-millimetre error scales `k·OPL` by 1000 and produces
   dense phase wraps. Cheap to detect, and cheap to miss if no case has a large
   `k·OPL`.

## Open questions the paper leaves implicit

Recorded rather than guessed, per AGENTS.md.

| Question | Status |
|---|---|
| Whether `a⁽ⁱ⁾` in eq 2 already includes the `1/p` importance weight when the rays came from `C_WAVE_TO_RAY` | **Resolved** by reading eq 1 with eq S4: yes, `a = Ũ/p` *is* the ray's amplitude. Eq 2 then sums those amplitudes |
| Whether `⟨n̂, d̂⟩` belongs in a field reconstruction at all | **Resolved by measurement — see below.** It does not |
| The reference for `OPL⁽ⁱ⁾` when rays arrive from a real trace | Not fixed by the paper. This repository requires the caller to declare it; see H1 |

## Finding: eq 2 and eq S5 are different operators

Main-text eq 2 carries the factor `⟨n̂, d̂⟩`. SI eq S5, which derives the same
wavelet sum as an estimator of the angular-spectrum integral (eq S2), does
not. The paper does not flag the difference, and it is not cosmetic.

CHE-25 measured which one preserves a field. Summing **every** propagating mode
of a random field on a 16×16 grid:

| Form | Agreement with the source field |
|---|---|
| without `⟨n̂, d̂⟩` (eq S5) | `7.1e-15` — round-off |
| with `⟨n̂, d̂⟩` (eq 2) | `2.2 %` of peak amplitude |

The `2.2 %` tracks the smallest `cos θ` on that grid. So:

- **`Projection.ASM_CONSISTENT`** (no factor) is what the coupler uses. A
  representation change must preserve the field, and only this form does. It is
  the default, and it is what makes a round trip exact.
- **`Projection.SENSOR_OBLIQUITY`** (with the factor) is main-text eq 2,
  retained as an explicitly named **sensor** model: a detector whose response
  depends on incidence angle. It is not a field reconstruction.

Picking one silently would have produced a coupler that quietly loses a few
percent off-axis, round-trips inexactly, and gives no test a name to fail
under. Both forms are implemented, both are tested against the thing each
actually claims to reproduce, and the choice is recorded in every
reconstructed field's provenance.

## Consumer-facing limitation: no wavefront-curvature term (CHE-50)

The reconstructed field is valid **at** the declared reference plane, with zero
further propagation. It carries no `exp(i k r²/2R)` term, because the wavelet sum
is linear in the transverse coordinate.

| Consumer | Affected? |
|---|---|
| measures an intensity or PSF at the handoff plane | no |
| composes the field into a further propagation | **yes**, and `\|U\|²` will not warn it |

Measured on `M3-SINGLET-REF`, on axis, 550 nm (CHE-38,
`benchmarks/reports/2026-08/sensor_handoff_convergence.md` §5): ~`1.2 rad` of phase against an
exact spherical-wave reference at the 5-Airy-radius gate edge, while the
intensity residual sits at `1e-3` and the complex-field residual (`~0.127`) is
flat rather than convergent. PB7 (CHE-58) put three PSF routes on the Cooke
Triplet and did not see the term — its post-handoff propagation distance is
zero, which is a property of that configuration and not of this operator. PB7
finding F3 states this.

**Disposition (CHE-50): tracked known limitation, no kernel change.** Revisit
when a propagation-sensitive hybrid composition independently requires it.
Either resolution — emitting the term, or refusing an undeclared
further-propagation request — is new, separately-verified physics with its own
ticket and its own oracle. Neither is claimed here.

**Correct remedy today:** to obtain a field on a different plane, advance the
*ray state* to that plane and reconstruct there. That is exact, not an
approximation (`consequence_1_moving_the_handoff` on the coupler card). Do not
propagate the reconstructed field as a substitute.

The declaration travels with the artifact: every emitted `ComplexField` carries
it in `provenance["validity"]`, so a downstream consumer can read it off the
field rather than having to have read this file.
