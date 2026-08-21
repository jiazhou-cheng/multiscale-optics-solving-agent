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

Evidence: `benchmarks/CHE70_METALENS_BRIDGE_REPORT.md` §6,
`tests/test_coherent_bridge.py::TestExactnessLimit`.

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
`benchmarks/M3_9R_SENSOR_HANDOFF_REPORT.md` §5): ~`1.2 rad` of phase against an
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
