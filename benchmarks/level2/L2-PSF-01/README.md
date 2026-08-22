# L2-PSF-01 — Optiland → C_RAY_TO_WAVE → Chromatix → PSF

Protocol `M3-SLICE-CPU-V1`. Graph `[M_RAY_OPTILAND, C_RAY_TO_WAVE,
M_WAVE_CHROMATIX]`, terminating at the propagated `ComplexField`. PSF
extraction (`verification.psf_measurement.measure_psf`) is a benchmark-layer
measurement on that terminal state, not a graph node: `C_FIELD_TO_PSF` was
retired by CHE-36 (M3.7).

```bash
./run.sh python benchmarks/level2/L2-PSF-01/run_benchmark.py \
    --output-dir outputs/M3/L2-PSF-01
```

## What this bundle is, and is not

This bundle **packages** the physics CHE-38 (M3.9R) and CHE-47 (its
extension) already measured — it calls those two probes' own code directly
(`benchmarks/probes/m3r_sensor_handoff.py`, `benchmarks/probes/
m3_quadrature_weight.py`) rather than re-deriving the sensor-plane
construction, the oracles, or the quadrature weight a second time. What it
adds: formal `result.json`/`provenance.json` packaging in `L2-COUPLER-01`'s
pattern (CHE-29), a genuine three-node graph demonstration that actually
routes work through the Chromatix adapter, and two negative controls a
probe study did not need but a benchmark bundle does.

**The primary physical gate is NOT met on the real traced system, and it is
decided by O1 alone.** O1 (analytic Airy, paraxial, aberration-free, shares
no code or traced data with the coupler) is the sole oracle that decides
`gate_met` / `PHYSICALLY_CORRECT` / `pass` anywhere in this bundle. O2 (an
independent float64 ASM + Rayleigh–Sommerfeld propagator) is *our own*
implementation, written specifically to check this same coupler — using it
to decide correctness would be validating custom code against custom code,
so every O2 figure in `result.json` is suffixed `_diagnostic_only` (or lives
under `accuracy.production.*_vs_o2_asm`) and never gates anything. CHE-47
measured the production (weighted) configuration at `2.21e-3` against O1 at
787,969 rays (CHE-38's synthetic aberration-free diagnostic reaches `4.07e-4`,
inside the gate — the gap is real geometry, not a coupler defect). No
tolerance is widened to hide this. `result.json` reports `status: "complete"`
with `accuracy.verdict.physical_correctness: "characterized_gate_not_met"` —
the pipeline runs correctly end to end and produces a well-attributed,
reproducible number; the number itself does not clear the gate.

## Sections

**`accuracy.production`** — CHE-47's own characterization: the full
217→787,969-ray ladder, weighted (production default) vs uniform (pre-CHE-47,
retained as a regression baseline) ray area weight, against O1 (analytic
Airy) and O2 (independent float64 ASM + Rayleigh-Sommerfeld). Includes
absolute-power convergence (CHE-33's `N^2.0024` finding, resolved by CHE-47's
quadrature weight).

**`accuracy.full_graph_demonstration`** — both probes' primary configuration
places the handoff exactly on the sensor, where CHE-38 found the required
post-handoff Chromatix propagation is zero. That does not exercise
`M_WAVE_CHROMATIX` with real work, so this bundle additionally runs CHE-38's
own `near_sensor_fine` candidate (0.001·R upstream) through the **actual**
Chromatix adapter (`asm_carrier_removed`) back to the sensor, and reports its
agreement with the zero-propagation configuration.

**`accuracy.negative_controls`** —

| control | what it must detect | current result |
|---|---|---|
| `opl_sign_flip` | `HandoffPerturbation(opl_sign=-1)` must wreck the sensor PSF against O1, the gate-deciding oracle (`≥ 0.5` relative L2); O2 is reported alongside for characterization only | **detected = true** |
| `quadrature_weight_regression` | production quadrature weight must measurably beat uniform ray weight (`≥ 1.2×`) against O1, the gate-deciding oracle — restating CHE-47's own finding as a pass/fail gate | **detected = false** — against O1 the production weight is `2.40×` *worse* than uniform on this real aberrated system (uniform `9.21e-4` vs weighted `2.21e-3`); the opposite ordering holds against O2 (`1.58×` improvement), which is diagnostic-only evidence, not a reason to override O1. Reported honestly, not hidden — see `accuracy.negative_controls.quadrature_weight_regression` and `tolerances.yaml`. |
| `exit_pupil_hard_support_reconstruction` (O4) | the exit-pupil reconstruction must continue to show a Fresnel-soft rim that does not sharpen with ray refinement — retained from M3.9/CHE-38 as an out-of-contract validity-limit test, not evidence the coupler is wrong | **detected = true** |

Because `quadrature_weight_regression` does not fire, `negative_controls_pass`
and `accuracy.pass` are both currently `false`. That is the honest result of
switching the decisive oracle from O2 to O1; it is not a regression in the
coupler itself (`DISCRETIZATION_CONVERGED` and the absolute-power fix in
`accuracy.production.absolute_power` are unaffected).

**`differentiability`** — not measured here. `derivative.verified` stays
`false` for `C_RAY_TO_WAVE`; the Optiland→Chromatix handoff is `forward_only`
per AGENTS.md until a custom derivative and a directional finite-difference
test exist (M4 scope).

## Reading the numbers

`result.json` is authoritative; `plot.png` is diagnostic. Three cautions,
carried forward from CHE-38/CHE-47:

1. **The exact sensor plane is a caustic in the position-space sense, and it
   is where this operator is best conditioned** — it reads ray directions
   and optical paths, never a local ray density, so the position-space
   degeneracy at focus is not one of its inputs.
2. **The reconstructed sensor field carries no `exp(i k r² / 2R)` curvature
   term.** Invisible in `|U|²` (which is why the intensity residual
   converges); not invisible to a caller who propagates the field further.
3. **The traced-system residual against O1 is not decomposed further here**
   (CHE-47 open item): O1 (sharing no code or traced data) sits closer to the
   weighted result than O2 does, which is the wrong direction for a genuine
   aberration-sensitive coupler defect — the likelier candidate is O2's own
   ring-averaged, linearly interpolated pupil-fit resolution. That is exactly
   why O2 is diagnostic-only and never the gate: a custom oracle with its own
   unresolved fit error must not decide whether a different custom
   implementation is "correct."

## Artifacts

`result.json`, `provenance.json`, `arrays.npz`, `tolerances.yaml`,
`convergence.json` (the full sensor ray ladder), `plot.png`, `README.md`.
Every file is SHA-256 hashed into `provenance.json`.

The scientific fingerprint covers physics only (`core.provenance.VOLATILE_KEYS`
stripped): wall-clock and run-identity keys are excluded, so
the fingerprint is bit-identical across two runs on an unchanged tree — the
same guarantee L2-COUPLER-01 (CHE-29) established for M2.
