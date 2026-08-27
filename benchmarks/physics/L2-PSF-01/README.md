# L2-PSF-01 — RETIRED. The workload lives on as `B3-PSF-SINGLET-01`

**There is no runnable entry point in this directory.** CHE-116 (M4.1) deleted
`run_benchmark.py` (600 lines) and `evaluate.py` (the bundle-hash evaluator,
whose only input was the bundle that runner wrote). What remains here is
`tolerances.yaml` — the evidence file the family's tolerance bases are migrated
from verbatim, and cited by `src/verification/families/b3_composed.py` — and
this README, which records what the bundle was and measured.

Run the workload here instead:

```bash
./run.sh python benchmarks/instances/b3_psf_singlet.py --write
```

That path is `GraphExecutor` over the committed graph document
`examples/graphs/psf_singlet_sensor.yaml`, then `verify()` against the
`B3-PSF-SINGLET` family. CHE-115 (M3.3) established the precondition CHE-116's
deletion waited on: it reproduces this bundle's frozen gate number
`fft_oracle_intensity_relative_l2 = 0.0022072391812867093` **bit-identically**
(`tests/test_substrate_proof.py::test_the_frozen_number_is_reproduced_bit_identically`
asserts `==`, not `approx`).

## What the deleted runner owned, and where each piece went

| the bundle's piece | where it is now |
|---|---|
| the graph `[M_RAY_OPTILAND, C_RAY_TO_WAVE, M_WAVE_CHROMATIX]` and its frozen gate | `examples/graphs/psf_singlet_sensor.yaml` + `benchmarks/instances/b3_psf_singlet.py` |
| the `opl_sign_flip` negative control | a graph variant through `runtime.variants.with_config_overrides`, run by the driver and recorded in `benchmarks/instances/records/B3-PSF-SINGLET-01.json` |
| the `near_sensor_fine` three-node demonstration | `b3_psf_singlet.run_near_sensor_fine` |
| the 12-rung convergence ladder, the O2 ASM/RS characterization oracle, absolute-power convergence | unchanged, in the probes this bundle already called rather than re-derived: `benchmarks/probes/quadrature_weight.py`, `benchmarks/probes/sensor_handoff_convergence.py`, with their own committed records |
| the `exit_pupil_hard_support_reconstruction` (O4) validity-limit control | unchanged, in `benchmarks/probes/sensor_handoff_convergence.py::_exit_pupil_negative_control` |
| the `quadrature_weight_regression` pass/fail restatement | **NOT carried forward as a gate, and now RETIRED AND REPLACED.** CHE-117 first found the control mis-specified — a verdict that flips sign with ray count (10.7 at 8 rings, 0.42 at 512, 0.69 at 1024) is measuring where two convergence curves cross, not whether the weight is right — and then found that respecifying it on converged arms does not rescue it either, because both arms converge to the *same* residual, so the converged improvement factor is 1.0 and its premise is false. The family now declares `uniform-weight-power-divergence` on `reconstructed_power_ray_doubling_excess`, which tests the property CHE-47 did establish: `|P(2N)/P(N) − 1|` reads `7.7e-4` with the weight and `14.9` without it. The `1.2` floor is unchanged, recorded as retired in `tolerances.yaml` and `verification.claim_ledger`. |
| `result.json` / `provenance.json` / `arrays.npz` / `convergence.json` / `plot.png` packaging, and `evaluate.py`'s hash check of it | superseded by `benchmarks/instances/records/B3-PSF-SINGLET-01.json`, stamped by `verification.evidence.write_instance_record` and swept by `tests/test_provenance_fingerprint.py` |

## The disposition, after CHE-117 (M4.2)

The two open scientific problems this bundle carried are closed, and neither was
closed by widening anything. Stated here because the sections below are the
bundle's own past-tense record and predate the answer.

**The primary gate is ATTRIBUTED AND UNMET.** `1.0e-3` unchanged;
`2.2072391812867093e-3` observed; every term of that number named:

| term | value | how it was established |
|---|---|---|
| sensor sampling | **0** | identical to ten significant figures across an 8× sensor-pitch refinement at fixed window, 6.5 → 51.9 px per Airy radius (`singlet_residual_grid.json`) |
| the quadrature weight | **0 at convergence** | the weighted arm is flat to 0.87% from 49,537 to 3,148,801 rays; the uniform arm crosses it near 181 rings, bottoms out at `7.04e-4` at 362, and climbs back toward it (`singlet_residual_attribution.json`) |
| O1's own Airy-scale freedom | **`2.093e-3`, 94.8% in quadrature** | the metric is linear in fractional scale error (slope 1.52), so the gate resolves the scale to `6.53e-4`; this system's two defensible NA declarations — paraxial geometric `0.0515667` and largest traced direction cosine `0.0517163` — differ by `2.902e-3`, spanning `4.445e-3` of metric, **4.4× the gate** (`o1_applicability.json`) |
| what is left | **`7.021e-4`, inside the gate** | the residual at O1's own best-fit NA `0.0516457`, which lands *inside* the interval the system's geometry leaves open |

**So O1 cannot decide this gate at `1e-3` on this system** — settled by
measurement, not asserted. The correct response is a configuration where O1's
assumptions hold (CHE-38's synthetic aberration-free bundle reaches `4.07e-4`),
not a wider tolerance and not a second route through our own code. The gate is
**not** met at the fitted NA and must not be read that way: fitting the oracle's
scale to the field under test removes the independence that makes O1 the only
admissible decider here.

**M0.2 cross-check.** The ~20-order-of-magnitude amplitude drift CHE-103
attributed to CHE-47 is the *same code change* as the quadrature weight but not
the same quantity, and it cannot reach this metric: the drift is the global
per-ray cell-area factor, and the gate metric peak-normalizes both inputs, so
rescaling the measured intensity by `2^64`, by `1e20` or by the recorded
uniform/weighted power ratio moves the residual by `1.0e-14` relative — float64
round-off. Same root change, different root cause.

**What was not consulted:** O2 (our own ASM/RS propagator) and any second
Optiland PSF route (PB7/CHE-58 F2). Neither appears anywhere in CHE-117's
evidence.

Everything below is the bundle's own description, kept because it records what
was run and what it found. Read it in the past tense.

## What this bundle was, and was not

This bundle **packages** the physics CHE-38 (M3.9R) and CHE-47 (its
extension) already measured — it calls those two probes' own code directly
(`benchmarks/probes/sensor_handoff_convergence.py`, `benchmarks/probes/
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
| `quadrature_weight_regression` | production quadrature weight must measurably beat uniform ray weight (`≥ 1.2×`) against O1, the gate-deciding oracle — restating CHE-47's own finding as a pass/fail gate | **detected = false** — against O1 the production weight is `2.40×` *worse* than uniform on this real aberrated system (uniform `9.21e-4` vs weighted `2.21e-3`); the opposite ordering holds against O2 (`1.58×` improvement), which is diagnostic-only evidence, not a reason to override O1. Reported honestly, not hidden — see `accuracy.negative_controls.quadrature_weight_regression` and `tolerances.yaml`. **Retired by CHE-117:** its premise is false at convergence, and `uniform-weight-power-divergence` replaced it. |
| `exit_pupil_hard_support_reconstruction` (O4) | the exit-pupil reconstruction must continue to show a Fresnel-soft rim that does not sharpen with ray refinement — retained from M3.9/CHE-38 as an out-of-contract validity-limit test, not evidence the coupler is wrong | **detected = true** |

Because `quadrature_weight_regression` does not fire, `negative_controls_pass`
and `accuracy.pass` are both currently `false`. That is the honest result of
switching the decisive oracle from O2 to O1; it is not a regression in the
coupler itself (`DISCRETIZATION_CONVERGED` and the absolute-power fix in
`accuracy.production.absolute_power` are unaffected).

*After CHE-117:* the control that did not fire was retired and its replacement
does fire, at a detection margin of `1.9e4`
(`benchmarks/instances/records/B3-PSF-SINGLET-01.json`,
`negative_control_results`). `negative_controls_pass` is still `false` on
`B3-PSF-SINGLET-01`, and now for a different and honestly separate reason: two of
the four declared controls — `axis-transpose` and `launch-phase-error` — are
identity operations at this on-axis instance and are declared `NOT_IMPLEMENTED`
rather than quietly dropped. `result.negative_controls_pass` requires *every*
declared control to have fired, which is the correct rule; exercising those two
needs an off-axis instance and is not CHE-117's scope.

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
   implementation is "correct." *CHE-117 did that decomposition — see the
   disposition section above. The answer did not need O2 at all: 94.8% of the
   residual is an Airy-scale offset, and the scale is what this system leaves
   undetermined to 0.29% through its own residual spherical aberration.*

## Artifacts (as the bundle wrote them — no longer produced)

`result.json`, `provenance.json`, `arrays.npz`, `tolerances.yaml`,
`convergence.json` (the full sensor ray ladder), `plot.png`, `README.md`.
Every file was SHA-256 hashed into `provenance.json`.

The scientific fingerprint covered physics only (`core.provenance.VOLATILE_KEYS`
stripped): wall-clock and run-identity keys were excluded, so
the fingerprint was bit-identical across two runs on an unchanged tree — the
same guarantee L2-COUPLER-01 (CHE-29) established for M2. That guarantee is now
carried by the `scientific_fingerprint` field of
`benchmarks/instances/records/B3-PSF-SINGLET-01.json`.
