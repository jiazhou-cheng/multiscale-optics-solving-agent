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

## The PSF is slightly too narrow, and refining the ray count barely helps

The residual against an independent wave reference falls only as `1/rings` and the
fitted numerical aperture comes out a few tenths of a percent too large. That is
not the kernel; it is the **per-ray area weight**. The sum is a quadrature over the
ray ensemble, and hexapolar sampling puts its outermost ring exactly on `ρ = a`,
where it represents half a cell but is counted as a whole one. The effective
aperture is then too large by half a ring spacing, so `ΔNA/NA ≈ 1/(2 × rings)`.

Diagnosis: fit an Airy pattern's `NA` to the reconstructed PSF and compare the
excess against `1/(2 × rings)`. If it tracks, this is the cause. Confirm by
reweighting the outer ring to `½` and the central ray to `¾`; the residual should
drop by an order of magnitude *and stop depending on the ray count*.

Do not chase this by adding rays — it is first order in the spacing, so 1e-3 on a
`N_f ≈ 23` system needs of order `10⁶` rays. Measured in CHE-38 (M3.9R). The evidence key is
`attribution_quadrature_weights` in `benchmarks/probes/records/
m3r_sensor_handoff.json` — a record that **has never been generated**; read
`benchmarks/M3_9R_SENSOR_HANDOFF_REPORT.md` §8.1–8.2 instead until CHE-63
lands it (disposition: `benchmarks/M3_M3_5_CLEANUP_DISPOSITION.md` item 1).

**CHE-47 implemented the reweighting this section describes**, as a producer-side
default: `optiland_handoff.declare_coherent_bundle` folds
`multiscale_optics_agent.couplers.quadrature.hexapolar_area_weight_m2` into the
amplitude whenever the Optiland adapter can confirm an un-vignetted hexapolar
fan. If you are seeing this failure mode on a *current* Optiland-traced bundle,
check `handoff.declarations["quadrature_weight"]["status"]` first: `"applied"`
means the fix is already active and the residual you are seeing is something
else (see below); `"unavailable"` means the adapter could not regenerate a
matching hexapolar fan (vignetting, or a non-hexapolar distribution) and you are
on the legacy path — `handoff.declarations["quadrature_weight"]["reason"]`
names why. The fix does **not** fully close the gate on a real (aberrated)
system the way it does on CHE-38's synthetic aberration-free bundle: on
`M3-SINGLET-REF` it improves the sensor residual `1.58×` (`3.91e-3 → 2.48e-3` at
787 969 rays) but does not reach `1e-3` (`benchmarks/probes/records/
m3_quadrature_weight.json`). If your residual is still above `1e-3` after
confirming `status == "applied"`, it is most likely the independent wave
oracle's own pupil-fit quality, not a further coupler defect — see that record's
`verdict` before concluding otherwise.

## The reconstructed pupil edge is soft and will not sharpen

Expected, and out of contract. The operator has no aperture support term, so it
cannot return a hard pupil boundary; the rim comes back Fresnel-soft over
`√(λR)`, with amplitude near `½` at the geometric boundary and an overshoot fringe
inside it, and the transition scale does not shrink with the ray spacing. Compare
against the circular-aperture Debye/Lommel solution, **not** a one-dimensional
straight knife edge — the straight edge is the wrong geometry and disagrees by
about 26%. If you need a hard pupil, apply the aperture yourself: that is the
`ray_as_wavefront_sample` mode, and it is not implemented here.

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

One code above the table does not surface as a `declare_coherent_bundle`
refusal: `NON_HEXAPOLAR_SAMPLING` (CHE-47, raised by
`multiscale_optics_agent.couplers.quadrature.hexapolar_ring_index`/
`hexapolar_area_weight_m2`) is caught internally by the Optiland adapter's
`_resolve_ray_quadrature_weight` and recorded as
`quadrature_weight.unavailable_reason` rather than propagated. A missing
quadrature weight is not treated as a correctness failure the way a missing
off-axis OPL term is — an unweighted bundle is still coherent, just not
scale-invariant under ray refinement — so the handoff falls back to the legacy
`sqrt(intensity_i)` amplitude rather than refusing.

## What is *not* a bug

- **Speckle in a reconstruction from sampled rays.** If the rays came from
  `C_WAVE_TO_RAY`, structure that is reproducible across seeds is physical
  interference; structure that is not reproducible is Monte Carlo noise. The
  paper makes this exact distinction for the hologram–lens system. Check
  reproducibility across seeds before concluding either way.
- **Phase discontinuities at a caustic.** Coherent complex summation is valid
  there; phase *unwrapping* is not. If a downstream step unwraps, that step is
  the defect.
