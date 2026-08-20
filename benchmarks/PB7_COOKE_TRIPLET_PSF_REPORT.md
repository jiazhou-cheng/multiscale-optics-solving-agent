# PB7 / CHE-58 — Cooke Triplet: FFT vs Huygens vs ray→wave PSF

Written conclusion for the run recorded in `outputs/PB7/pb7_cooke_triplet_psf_comparison.json`
(git `4fa9bbb`, optiland 0.6.0 / chromatix 0.6.0 / jax 0.6.2, CPU, 110.6 s wall clock).

Status: **diagnostic benchmark**. No tolerance, no pass/fail, no gate. Nothing here
changes coupler, adapter, or measurement physics, and no test imports it.

## What was run

One optical system (`optiland.samples.objectives.CookeTriplet`), λ = 0.55 µm, two
fields — on-axis `(0, 0)` and full field `(0, 1.0)` = 20°. Three routes:

| | route | sampling |
|---|---|---|
| A | Optiland `psf.FFTPSF` | 128 rays, 2048 grid, `strategy='chief_ray'`, `remove_tilt=False` |
| B | Optiland `psf.HuygensPSF` | 128 rays, 256 image size, same strategy |
| C | Optiland trace → `C_RAY_TO_WAVE` → Chromatix ASM → \|U\|² | 64 hexapolar rings (12481 rays), 0.2 µm pitch, 512² sensor |

A and B run on the bundled sample object; C runs on the repository's canonical
`OpticalSystemSpec` transcription. The transcription is **checked, not trusted**:
surface positions, paraxial f/EPD/EPL/XPD/XPL/FNO, and every traced ray array
(x, y, z, L, M, N, i, opd) at both fields are bit-identical (`all_identical: true`).
The run aborts otherwise.

Method C's wave leg propagates 0 m — CHE-38 selected the sensor plane itself as the
handoff — so the leg is run only to exercise the shipping path; its measured identity
against the coupler output is 2.9e-7 relative L2.

All three are peak-normalized on a common grid (0.2 µm pitch, half-width the largest
whole µm strictly inside every method's own native window, so no method is
extrapolated), linear interpolation, same call for all three.

## On-axis: all three agree

| pair | relative L2 | peak separation |
|---|---|---|
| A vs B | 0.0080 | ~0 |
| A vs C | 0.0053 | ~0 |
| B vs C | 0.0085 | ~0 |

FWHM 2.3149 / 2.3198 / 2.3139 µm (A/B/C) against an Airy first null of 3.339 µm;
EE50 4.243 / 4.243 / 4.238 µm; EE80 6.277 / 6.277 / 6.251 µm. Peaks coincide with
the chief ray to below floating-point resolution.

**C is no further from A or B than A and B are from each other.** The resampling
artefact floor is a few × 1e-3, so these three residuals are essentially at the
floor this comparison can resolve. Method C's ray-count sensitivity (64 vs 32 rings)
is 0.0014 — well below the A-vs-B residual, so ray sampling is not what separates C.

## Full field (20°): B and C agree; A is the outlier — and the cause is identified

| pair | relative L2 | peak separation |
|---|---|---|
| A vs B | 0.313 | 0.677 µm (3.39 px) |
| A vs C | 0.315 | 0.671 µm (3.36 px) |
| **B vs C** | **0.0138** | **0.006 µm (0.03 px)** |

Morphology is the same coma-like lobe-plus-tail structure in all three, displaced
+7.0 / +7.7 / +7.7 µm in y from the chief ray (A/B/C). The B−C difference map peaks
at 0.008 of peak; the A−B and A−C maps peak at 0.43 — a factor of ~55 larger, and
spatially a dipole straddling the core, i.e. the signature of a *scale/registration*
error, not a different physical PSF.

**Attribution.** Off axis, the image-space pupil has anisotropic direction-space
extent: F/#ₓ = 5.284, F/#_y = 6.030 (anisotropy 1.141), against the single scalar
working F/# = 5.480 that `FFTPSF` uses to set its pixel scale `dx = λ·F/#/Q`. That
predicts A is mis-scaled by s_y = 1.100, s_x = 0.964. Measured by a deterministic
scale fit: s_y = 1.0955, s_x = 0.967 onto B, and s_y = 1.0948, s_x = 0.9638 onto C
— matching the prediction to within 0.5% on both axes. Correcting only that scale
drops A-vs-B from 0.313 to 0.023 and A-vs-C from 0.315 to 0.028, i.e. **93% / 91% of
the off-axis residual is Optiland `FFTPSF`'s isotropic-F/# pixel scale**, not a
disagreement about the field. No reported PSF is scaled by this; the fit is
diagnostic only.

Strehl as reported by Optiland (centre pixel / 100, so off axis it is not the peak):
A 0.0351, B 0.0432; peak-over-centre 2.93 and 2.05. Method C has no absolute Strehl —
its scale is uncalibrated.

## Answers to the ticket's questions

- **Same morphology?** Yes, at both fields, all three.
- **Centre consistent?** On axis yes, to numerical zero. Off axis B and C agree to
  0.03 px; A sits 3.4 px away, explained by the pixel-scale error above.
- **Width/structure consistent?** Yes. On-axis FWHM agrees to 0.3%; off-axis B and C
  agree to 0.4% in FWHM_y and 0.7% in FWHM_x.
- **Off-axis aberration structure reproduced by all three?** Yes.
- **Obvious disagreements?** One: A's off-axis pixel scale.

## Relation to known open issues

- **CHE-50** (missing wavefront-curvature term) — **not active** in what is measured
  here. The reconstructed field carries no exp(i k r²/2R) term, which is invisible in
  \|U\|² at the handoff plane, and the handoff *is* the sensor plane, so there is zero
  post-handoff propagation for it to act over. It becomes active the moment a caller
  propagates the method-C field further.
- **CHE-47 / CHE-48** (per-ray quadrature weight, unattributed sensor residual) —
  **potentially active**; the hexapolar area weight was applied at both fields
  (`quadrature_weight_status: applied`). It would show as a width error, and C's
  widths track B's closely, so nothing here points at it.
- **CHE-51** (Fresnel number vs NA) — **not separated**. One system, one wavelength:
  N_f and NA are not independently varied. Named so the absence is explicit.

## Dominant caveat

A and B are two implementations *inside one package* sharing the same Wavefront/OPD
front end — same reference sphere, same launch-tilt removal, same pupil sampling.
They are not independent in the way an analytic oracle would be, so the A-vs-B
residual **understates** the true uncertainty of the Optiland pair. C is the only
route with a different front end. This is a three-way consistency check, not a
validation against truth.

## What to investigate next

1. Confirm the FFT pixel-scale finding against Optiland upstream (is the isotropic
   working F/# intentional for `FFTPSF`?) before treating A as a reference anywhere.
2. Get a genuinely independent oracle for the off-axis case — the analytic route,
   not another Optiland path — before any quantitative tolerance is set.
3. Re-run C with non-zero post-handoff propagation to make CHE-50 observable; that
   is the configuration this benchmark deliberately cannot see.
4. Only then consider whether this becomes a quantitative validation test
   (explicitly out of scope here).
