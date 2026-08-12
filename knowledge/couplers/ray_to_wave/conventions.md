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

## Three inherited hazards

These are not hypothetical. Each was recorded by M0 or M1 and is unresolved at
M2 open.

### H1 — Optiland `opd_native` sign and reference are unverified

M1 recorded `opd_native` with `opd_reference: "unverified"` and
`opd_sign: "unverified"`, and deliberately declined to interpret a probe that
returned `opd = 12` for a 10 mm separation. M1's exit report carries this
forward as item 4 of what M2 inherits.

**Resolution for M2:** `opd_native` is *not* an admissible `OPL⁽ⁱ⁾` source.
The coupler core requires an OPL in metres whose reference plane the caller
declares. A caller who wants to use Optiland's OPD must first characterize it
against a known geometry — that is separate work, and until it is done the
coupler refuses the input rather than guessing a sign.

Why refusal rather than a default: a wrong OPL *reference* is a constant piston
and mostly harmless; a wrong OPL *sign* conjugates the wavefront, turning a
converging beam into a diverging one. Those two failure modes are
indistinguishable downstream, so the ambiguity cannot be allowed through.

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
| Whether `a⁽ⁱ⁾` in eq 2 is intended to already include the `1/p` importance weight when the rays came from `C_WAVE_TO_RAY` | Resolved by reading eq 1 with eq S4: yes, `a = Ũ/p` *is* the ray's amplitude. Eq 2 then sums those amplitudes. Verified by the round trip rather than assumed |
| Whether `⟨n̂, d̂⟩` is an amplitude or an intensity projection | Read as amplitude, since eq 2 is a sum of complex amplitudes. To be confirmed by the power-conservation check, which distinguishes the two |
| The reference for `OPL⁽ⁱ⁾` when rays arrive from a real trace | Not fixed by the paper. This repository requires the caller to declare it; see H1 |
