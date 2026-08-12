# C_RAY_TO_WAVE — failure guide

Read this before debugging a surprising reconstruction. Ranked by how often the
symptom is misdiagnosed, not by severity.

## The reconstruction looks like noise

| Cause | Distinguishing evidence | Fix |
|---|---|---|
| **Undersampled wavefront** | Error falls when ray count rises; the phase difference between adjacent rays exceeds π at the plane | Raise ray density. This is a Nyquist condition on the wavefront, not on the output grid — refining the output grid alone will not help and is the usual wasted step |
| Unit error (mm for m) | `k·OPL` is ~1000× too large; phase wraps densely everywhere | Check the adapter boundary conversion, not the coupler |
| OPL reference drifting per ray | Noise, not a clean tilt or defocus | Declare one reference plane for the whole ensemble |

The first two look identical at a glance. The discriminator is whether error
responds to ray count.

## The field is mirrored, or a converging beam diverges

Almost always the phasor sign. `exp(−i k·OPL)` instead of `exp(+i k·OPL)`
conjugates the wavefront. Symptoms:

- a focus appears on the wrong side of the plane;
- a tilt runs the wrong way in `x`;
- propagating the result with Chromatix moves the beam backwards.

Test with a manufactured travelling wave at two planes separated by `Δz`; the
phase must advance by `+k Δz`. Do not infer the sign from a symmetric case — it
cannot distinguish.

The related failure is an `(x, y)`/`(y, x)` transposition. A rotationally
symmetric test will never catch it. Use an axis-asymmetric case.

## Power is not conserved

| Cause | Signature |
|---|---|
| Projection factor `⟨n̂, d̂⟩` omitted | Discrepancy grows with incidence angle; exact at normal incidence |
| `⟨n̂, d̂⟩` applied as an intensity rather than amplitude factor | Discrepancy scales as `cos²θ` where `cosθ` is expected |
| Rays outside the reconstruction aperture | Loss is real and should be reported as truncation, not chased |
| Double-applied normalization | Constant factor; independent of geometry |

A constant, geometry-independent factor is a normalization bug. A
geometry-dependent one is physics being dropped. Check which before editing.

## Off-axis rays contribute a piston instead of a ramp

The `Δr⁽ⁱ⁾(x, y)` term is missing. Invisible for a single on-axis ray, which is
why an on-axis smoke test passes and everything else is wrong. The signature is
that a tilted collimated bundle reconstructs as a *uniform-phase* patch rather
than a linear phase ramp.

## Structured failures the coupler must emit

Never a silently degraded field. Each of these returns a diagnostic naming the
violated condition:

| Code | Condition |
|---|---|
| `MISSING_DECLARATION` | Input omits units, frame, phasor, reference plane, or OPL reference |
| `OPL_REFERENCE_UNVERIFIED` | Caller passed `opd_native`, whose sign and reference M1 recorded as unverified |
| `AMPLITUDE_IS_A_WEIGHT` | Caller passed a real ray weight where a complex amplitude is required |
| `UNDERSAMPLED_WAVEFRONT` | Adjacent-ray phase difference exceeds π at the plane |
| `RAYS_NOT_AT_PLANE` | Ray intersections are not coplanar with the declared reconstruction plane |
| `NON_UNIT_DIRECTION` | Direction vectors are not unit-norm within tolerance |
| `EMPTY_ENSEMBLE` | No surviving rays reach the plane |

Per AGENTS.md, a failed coupler returns structured diagnostics and does not
return an invented field.

## What is *not* a bug

- **Speckle in a reconstruction from sampled rays.** If the rays came from
  `C_WAVE_TO_RAY`, structure that is reproducible across seeds is physical
  interference; structure that is not reproducible is Monte Carlo noise. The
  paper makes this exact distinction for the hologram–lens system. Check
  reproducibility across seeds before concluding either way.
- **Phase discontinuities at a caustic.** Coherent complex summation is valid
  there; phase *unwrapping* is not. If a downstream step unwraps, that step is
  the defect.
