# L2-COUPLER-01 — bidirectional ray–wave coupler

Protocol [`M2-COUPLER-CPU-V1`](../../M2_COUPLER_PROTOCOL.md). Exercises both
coupler directions together, on CPU at `float64`/`complex128`, through
`./run.sh` only.

```bash
./run.sh python benchmarks/level2/L2-COUPLER-01/run_benchmark.py \
    --output-dir outputs/M2/coupler
```

The coupler core imports neither Optiland nor Chromatix; `provenance.json`
records `forbidden_modules_loaded` observed at exit, and the M1 branches remain
untouched by this benchmark.

## What each section establishes

**Accuracy** — deterministic gates, all evaluated before any timing is accepted.

| Gate | What it rules out |
|---|---|
| Exactness limit (two spectra) | With every propagating bin enumerated there is no sampling error, so a failure here is a transform defect and tuning `N` would be beside the point |
| Analytic plane-wave oracle | Pins the OPL handling, the `Δr` ramp, the phasor sign and the projection convention at once — removing any one breaks it (SI Figure S1c) |
| Round trip, plus a mismatched pairing | A shared convention error cancels between the two directions. The mismatched case must *fail*, or the exactness result proves nothing |
| Cascade | Ray count after a planar DOE is the caller's `P·S` budget, not the incoming count (SI Algorithm S1); a pure-phase DOE conserves discrete power |
| Curvature bound | `arcsin(D/2R)` must bound a *measurement* across the Figure 3c regime, not merely be plotted beside one |
| Negative controls | Five deliberate defects, each run through the shipping implementation with one term removed |

**Stochastic** — `C_WAVE_TO_RAY` is a Monte Carlo estimator, so "reproducible"
and "accurate" are separate claims. Reported: the exactness limit, unbiasedness
against the *measured* standard error over ≥ 32 seeds, a convergence exponent
fitted over a six-point sweep, and variance for both sampling densities on both
a concentrated and a multilobed spectrum.

**Differentiability** — a characterization, never a promotion. Records what the
SI S7.2 estimator was measured to compute, in what regime, together with the
density-live control that *is* biased. `derivative.verified` stays false.

**Performance** — recorded only after the gates pass, with two untimed warmups
and seven timed repeats, median primary.

## Reading the numbers

`result.json` is authoritative; figures are diagnostic. Three cautions:

1. **A single realization is never a result.** Report the ensemble mean with its
   standard error. Structure that is reproducible across seeds is physical
   interference; structure that is not is Monte Carlo noise, and one seed cannot
   tell them apart (SI Figures S3, S4).
2. **The curvature bound is conservative below `sqrt(2 λ R)`.** Under that width
   the patch's own diffraction limit exceeds the curvature spread, so a large
   bound-to-measured ratio there is a property of the aperture, not slack in the
   bound.
3. **Discarded evanescent power is reported, not netted away.** A large fraction
   means the field should probably not be turned into rays at all.

## Artifacts

`result.json`, `provenance.json`, `arrays.npz`, `tolerances.yaml`,
`convergence.json`, `ensemble_statistics.json`, `README.md`. Every file is
SHA-256 hashed into `provenance.json`.

The scientific fingerprint covers physics only, reusing the M1 volatile-key
stripping: ensemble means over a fixed seed sequence are physics and are
included; per-realization wall-clock is not. That exclusion is the M1.8 lesson,
where per-case runtimes had leaked into the wave fingerprint and made it track
machine load.
