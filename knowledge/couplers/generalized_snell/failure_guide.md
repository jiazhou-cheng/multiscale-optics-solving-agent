# Failure modes for C_GENERALIZED_SNELL, keyed by symptom

## "It refuses with MODEL_NOT_APPLICABLE naming an evanescent order"

**Symptom:** `ContractError(MODEL_NOT_APPLICABLE)`, declaration `"order"`,
message names `PROPAGATING_ORDER_EXISTS` and a negative margin.

**Cause:** `|k_t^out| >= n_t k0` for at least one ray: the requested order has
no outgoing propagating direction at that position, for that incidence angle,
those indices, and that order. This is not a bug -- it is the hard limit the
governing equation has, and returning a direction anyway would be a normalized
vector pointing nowhere physical.

**What to check:** the requested `order`; whether `n_transmitted` is declared
correctly (a smaller `n_t` shrinks the propagating cone); whether the incident
angle plus the grating deflection together exceed what the transmitted medium
can support. This is a property of the *configuration*, not of the
implementation -- a real grating at these parameters would produce no
transmitted order there either.

## "It refuses with MISSING_DECLARATION naming the gradient estimate"

**Symptom:** `ContractError(MISSING_DECLARATION)`, declaration `"patch_px"`,
message names `LOCAL_GRADIENT_SMOOTHNESS` and a negative margin.

**Cause:** either (a) the local phase genuinely has a sharp discontinuity near
the ray's position (a masking edge, a stitched sub-aperture boundary, a
deliberately blazed sawtooth reset), or (b) the surface is undersampled at that
location relative to the declared transverse scale, so the finite-difference
estimator cannot distinguish "smooth but fast-varying" from "aliased".

**What to check:** is a discontinuity expected there (an edge, a phase-wrap
reset by design)? If so, this model is the wrong tool for that ray -- use
`LOCAL_PATCH` or `FULL_FIELD`, which form an actual field and do not need a
local gradient. If not expected, sample the surface's declared phase on a
finer grid, or reduce `patch_px` so the declared transverse scale shrinks.

**Known blind spot, not a bug to chase:** the same check reads the *raw*
wrapped phase step at the estimator's own sampling baseline. A signal that is
*uniformly* aliased across the whole surface (every sample off by the same
wrong multiple of `2 pi`) can produce a raw step that wraps back to something
comfortably below `pi` and passes this check while still returning a
completely wrong gradient. This is a fundamental limit of any local,
single-point finite-difference phase estimator, not something this ticket
attempted to solve in general -- see `AGENTS.md`'s scope discipline and
CHE-143's stated non-goal: "do not attempt to solve arbitrary phase unwrapping
as a new project." If a result looks physically implausible on a
high-line-density grating, check the surface is sampled at least a few times
per period before trusting this model's output there.

## "It refuses with MISSING_DECLARATION naming the substrate"

**Symptom:** `ContractError(MISSING_DECLARATION)`, declaration `"substrate"`.

**Cause:** `substrate=Substrate.CONFORMAL`. This model always refuses a
conformal substrate in this delivery -- there is no per-ray local tangent
frame declaration mechanism, and approximating one with the flat-plane frame
would silently produce a direction computed in the wrong local basis.

**What to do:** use `LOCAL_PATCH` (SI S10's own recommendation for a conformal
surface), which is at least the right model even though its own conformal path
is separately unimplemented today (a different refusal, `MISSING_DECLARATION`
from `plan_patches`, for a different reason -- see the `patch_wft` pack).

## "It refuses with MISSING_DECLARATION naming n_incident or n_transmitted"

**Symptom:** raised from `DiffractiveSurface` used with `FULL_FIELD` or
`LOCAL_PATCH`, not from this model.

**Cause:** this is the *other two* models refusing a declared index, not this
one. `GENERALIZED_SNELL` is the model that accepts a declared index -- if you
hit this refusal, you likely meant to request `model='generalized_snell'`.

## "The outgoing direction is mirrored / conjugated from what I expected"

**Cause:** almost always a phasor-sign or order-sign mistake -- the two are
easy to confuse because they produce the identical symptom (the deflection
flips sign) for different reasons. See
`tests/test_diffractive_interaction.py::test_control_a_phasor_sign_flip_conjugates_the_deflection`
and its order-sign sibling. Check the surface was built with
`DiffractiveSurface.from_phase` (which applies the repository's `exp(+i phi)`
convention in the one correct place) rather than a hand-rolled
`exp(-i phi)`, and check `GeneralizedSnellParameters.order` matches the
physically intended order, not its negative.

## "single_order_dominance is lower than I expected for a surface I believe is a pure single tone"

**Cause, usually not a bug:** rectangular-window spectral leakage. A truncated
sinusoid's windowed transform puts roughly 81-82% of its energy in the
mainlobe (`0.903**2` for a separable 2-D rectangular window, a standard DFT
result), not 100% -- and `single_order_dominance` integrates over the window's
own native angular resolution, not a single interpolated FFT bin, specifically
so heavy zero-padding (which `resolve_pad_px` applies, for a different reason:
avoiding reconstruction replica aliasing) does not artificially depress the
number further by spreading the same mainlobe over many interpolated bins. If
the measured value is well below ~0.7-0.8 for what you believe is one tone,
check the window (`patch_px`) actually spans several periods of the surface's
own spatial frequency -- a window narrower than one period cannot resolve a
tone from its own truncation sidelobes at all.
