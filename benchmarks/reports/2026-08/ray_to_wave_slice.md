# M3 exit report — Optiland → C_RAY_TO_WAVE → Chromatix → PSF vertical slice


> **Evidence:** `outputs/…` paths below are **local-only** — that directory is
> gitignored and exists on the machine that produced this run, not in a clone.
> Committed records live in `benchmarks/probes/records/`. See
> [`benchmarks/reports/README.md`](../README.md#where-the-evidence-actually-is).

CHE-39 (M3.10). This report integrates evidence only. Every number in it comes
from a run recorded in `outputs/M3/L2-PSF-01`, the staged runs behind
`benchmarks/reports/2026-08/sensor_handoff_convergence.md` (§11 — *not* from
`benchmarks/probes/records/m3r_sensor_handoff.json`, which has never been
generated; CHE-62 item 1),
`benchmarks/probes/records/m3_quadrature_weight.json`, `benchmarks/probes/
records/m3_psf_verification.json`, or the test suite named beside it.

**Verdict: M3 is recommended for exit, with the physical-correctness gate
explicitly NOT met.** The end-to-end graph — `M_RAY_OPTILAND → C_RAY_TO_WAVE →
M_WAVE_CHROMATIX`, terminating at a `ComplexField` with PSF as a benchmark-layer
measurement — is implemented, runs through `L2-PSF-01`, and is reproducible
bit-identically. The sensor-side `C_RAY_TO_WAVE` handoff itself is **verified**
(CHE-38's verdict A: discretization-converged, no floor, no structural defect).
What is **not** verified is the graph's absolute physical accuracy on the real
traced `M3-SINGLET-REF` system against the frozen `1.0e-3` gate: CHE-47 measures
`2.2e-3`–`2.5e-3` at 787,969 rays, `2.2`–`2.5×` over gate, with roughly half of
that residual attributed to a specific mechanism (a per-ray quadrature weight at
the aperture boundary) and the other half an open item. **On this same real
traced system, restoring the pre-CHE-47 uniform ray weight instead of the
production quadrature weight brings the O1 residual under the gate**
(`9.21e-4` vs `1.0e-3`, at 787,969 rays) — the inversion of CHE-38's synthetic
aberration-free diagnostic, where the weighted result (`4.07e-4`) is the one
inside the gate. This is reported as an open inversion, not a reason to revert
production: the quadrature weight is independently required to resolve
CHE-33's `N^2.0024` absolute-power divergence (§"Per-ray quadrature weight in
production" below), which the uniform configuration does not fix. No gate was
widened to reach this verdict; §"Findings that changed the conclusion" and the
figure below are the reason this milestone is not simply reported as "passed".

M3 began where M2 ended: two verified couplers with no admissible way to feed
`C_RAY_TO_WAVE` a real Optiland trace, because `opd_native`'s sign and reference
plane were `unverified`. Ten sub-tickets later (M3.1–M3.9R, CHE-40, CHE-41,
CHE-47) the graph runs, and the milestone's central discovery is that the
obvious first benchmark — reconstructing a hard exit-pupil aperture from
survivor rays — was testing an operation `C_RAY_TO_WAVE` never claimed to
perform.

![M3 residual/gate trajectory across the milestone](../outputs/M3/M3_EXIT_SUMMARY.png)

---

## Exact commands

```bash
./run.sh python benchmarks/physics/L2-PSF-01/run_benchmark.py --output-dir outputs/M3/L2-PSF-01
./run.sh python benchmarks/physics/L2-PSF-01/evaluate.py outputs/M3/L2-PSF-01
./run.sh python archive/benchmarks/gen1/benchmarks/L1-RAY-01/run_all.py --output-dir outputs/M1/ray
./run.sh python archive/benchmarks/gen1/benchmarks/L1-WAVE-01/run_all.py --output-dir outputs/M1/wave
./run.sh python archive/benchmarks/gen1/benchmarks/L2-COUPLER-01/run_benchmark.py --output-dir outputs/M2/coupler
./run.sh python archive/benchmarks/gen1/benchmarks/verify_m1_independence.py
./run.sh pytest -q
```

| Command | Result |
|---|---|
| `L2-PSF-01/run_benchmark.py` | `status: complete`; negative controls 3/3 detected; physical-correctness gate **not met** (reported, not hidden) |
| `evaluate.py` on the clean bundle | exit `0` |
| `evaluate.py` on a mutated bundle | exit `2`, hash mismatch |
| `L2-PSF-01` fingerprint, two independent runs | **bit-identical**: `eea43cf441c418fb...` |
| `L1-RAY-01` bundle, re-run on the M3 tree | fingerprint unchanged: `43dab1eedf5ca8fc...` |
| `L1-WAVE-01` bundle, re-run on the M3 tree | fingerprint unchanged: `b2d99bcc12874484...` |
| `L2-COUPLER-01` bundle, re-run on the M3 tree | fingerprint unchanged: `c928e4ca36c6dc1c...` |
| `verify_m1_independence.py`, re-run on the M3 tree | `status: passed`, 13/13 claim checks |
| full suite `pytest -q` | **602 passed, 21 skipped, 2 xfailed, 1 xpassed** |

The 21 skips are all `test_m3r_sensor_handoff.py`: CHE-38's consolidated probe
record (`benchmarks/probes/records/m3r_sensor_handoff.json`) has never landed —
see L5 below. The two xfails (fdtdx gradient locks) and one xpass (sax circuit
gradient) are the same pre-existing, documented anomalies M2 recorded at 348
passed. M2 ended at 348 passed; M3 adds 254 passing tests plus the 21 skips
above (L5).

## Environment

| Item | Value |
|---|---|
| Protocol | `M3-SLICE-CPU-V1`, extending `M2-COUPLER-CPU-V1` |
| Python | 3.12.13 |
| Device / dtype | CPU. Ray domain: `float64`. Coupler kernel: `complex128`. Chromatix leg: `complex64` with carrier removal (CHE-40) |
| System | `M3-SINGLET-REF`: plano-convex singlet, `n=1.5168`, `R=2.5mm`, center thickness `0.2mm`, `f/9.7`, `λ=550nm`, `NA=0.05171631827291936` |
| Scientific fingerprint (`L2-PSF-01`) | `eea43cf441c418fb57cd47f60c7597fc2de58d3225cf1fa7356d0f5a7a388680` |
| Fingerprint reproduces | **yes**, bit-identical across two independent runs (no RNG in this graph — Optiland's trace and the coupler kernel are both deterministic) |
| `dirty_worktree` | see L8 — this report is written against an uncommitted working tree |

---

# What M3 established

## The end-to-end graph is implemented and runs (CHE-39, `L2-PSF-01`)

`benchmarks/manifest.yaml`'s `L2-PSF-01` flips from `implemented: false,
blocked_by: <opd_native unverified>` to `implemented: true`. The bundle packages
CHE-38/CHE-47's own measurement code (imported directly, not re-derived) into
`result.json` / `provenance.json` / `arrays.npz` / `plot.png` / `tolerances.yaml`
/ `README.md`, exactly as `L2-COUPLER-01` did for M2, plus two things neither
prior probe needed:

* **A genuine three-node graph demonstration.** Both CHE-38's and CHE-47's
  primary configuration places the handoff exactly on the sensor, where the
  required post-handoff Chromatix propagation is zero — so neither exercises
  `M_WAVE_CHROMATIX` with real work. `L2-PSF-01` additionally runs CHE-38's own
  `near_sensor_fine` candidate (`0.001·R` upstream) through the **actual**
  Chromatix adapter (`asm_carrier_removed`), and the result agrees with the
  zero-propagation configuration to an absolute difference of `1.2e-9`
  (`4.9e-7` relative) in the same relative-L2-vs-O2 metric — the padding/
  propagation leg introduces no measurable error at this scale.
* **Two negative controls the probes did not need.** An OPL sign flip
  (`HandoffPerturbation(opl_sign=-1)`, detection margin `4.16 / 2.5e-3 ≈
  1673×`) and a direct restatement of CHE-47's own uniform-vs-weighted
  comparison as a pass/fail gate (`1.58×` measured, `1.2×` required).

## Optiland's `opd_native` characterized (M3.1, CHE-30)

`RealRays.opd` is **absolute accumulated optical path length** (not
chief-ray-relative), non-negative, referenced to the ray **launch plane**
(`knowledge/solvers/optiland/conventions.md:118-167`). The M1 anomaly — `opd=12`
for a nominal `10mm` separation — is now fully explained: the M1 probe used
`EPD=2.0mm` with the launch plane at `z=-2mm`, so `|2·1| + |10·1| = 12` exactly,
residual `0.0`. Optiland's own internal wavefront sign (`opd_ref − opd`) is the
**reverse** of this repository's `L1-RAY-01` convention (`ray − chief`) — a
convention mismatch that would have been a silent sign error if carried across
uninspected.

## One diffraction-limited system, and exit-pupil access (M3.3, CHE-32)

`M3SingletRef` was added and probed to the M1 standard. Optiland exposes no
pupil mask; the exit pupil is **read** from `optic.paraxial.XPL()` / `XPD()`,
not constructed, and is **virtual** on both `M3-SINGLET-REF` and
`M3-REVERSE-TELEPHOTO`. The measured traced-ray semi-extent sits *above* the
paraxial semi-diameter (`2.4978e-4 m` vs `2.4935e-4 m`), so a consumer must use
the paraxial value, not the traced extent, as the aperture radius. `ReverseTelephoto`'s
`L1-RAY-01` fingerprint (`43dab1ee...`) was re-verified unchanged after the
export path changed.

## `RayBundle` bridge, and a protocol defect found in passing (M3.4, CHE-33)

`with_amplitude_from_weight` declares `amplitude = sqrt(weight)`, `weight =
RealRays.i` — no area factor, no `1/N`, and Optiland's `i` carries no phase. In
the process of freezing the diffraction-limited Airy oracle, this ticket found
`benchmarks/protocols/slice_protocol.yaml`'s frozen `airy_radius_um = 12.9746` is `1.22·λ/NA`
— the first-null **diameter**, not the radius. The true radius is `0.61·λ/NA =
6.4873 µm`. Recorded and corrected rather than silently consumed; M3.8 (below)
inherited the correction rather than the bug.

## Coupler as an executable graph edge (M3.5, CHE-34)

`couplers/base.py` had declared `Coupler.transform` since M2 with nothing
defining it — `RayToWaveCoupler` (`ray_to_wave_node.py`) is that definition, and
the test suite pins it **bit-identical** to calling `ray_to_wave()` directly on
the same input, so none of M2's coupler-core verification evidence stops
binding on what the graph actually executes.

## Chromatix propagation: phasor sign and the `complex64` cost (M3.6, CHE-35)

Measured, not assumed: a converging spherical pupil field written under
`exp(-ik√(ρ²+R²))` focuses under `asm_propagate` to `0.990` of the analytic
Airy peak; its conjugate misses by `1008×`. Frozen kernel convention:
`exp(+ik_z z)` for `z > 0`. The `complex64` cast's field/intensity error is
measured on the actual propagated field, not merely flagged, and an unpadded
propagation's edge-energy diagnostic was shown to be a weak signal: it moved
only `2×` between a correctly padded run and one with `1.4e-1` relative
intensity error from wraparound.

## `C_FIELD_TO_PSF` retired; PSF is a measurement, not a coupler (M3.7, CHE-36)

`ComplexField → |U|^2` changes no representation, so it was removed as a
registry primitive rather than implemented. PSF extraction now runs through
`verification.psf_measurement.measure_psf` on the graph's terminal `ComplexField`,
recorded in provenance, never as a graph edge. Re-verified on the M3 tree this
session: no `- id: C_FIELD_TO_PSF` line exists in `registry/couplers.yaml`
(`tests/test_l2_psf_bundle.py::test_the_registry_still_has_no_c_field_to_psf_entry`,
alongside the pre-existing `test_graph_validation.py` and
`test_m3_psf_measurement.py` locks).

## Terminal field verified against independent oracles — at the exit pupil, and it did not pass (M3.8, CHE-37)

This is the ticket whose result reshaped the rest of the milestone. Tested at
the **exit-pupil** handoff (`handoff_plane: "exit_pupil"`, the configuration
frozen through M3.4–M3.8):

* analytic Airy profile residual (on-axis `M3-SINGLET-REF`): relative L2
  `5.87e-3`, RMS `1.33e-3`;
* independent FFT/Fraunhofer oracle: relative L2 `1.51e-2` — **`15.09×`** the
  `1.0e-3` gate;
* the `airy_peak_intensity_relative` gate: measured `1.22e-2` vs `2.0e-3` gate —
  **`6.08×`** over;
* energy ledger: unattributed residual `7.5e-7` vs `1.0e-3` gate — **passes**,
  cleanly;
* off-axis vehicle frozen on `M3-REVERSE-TELEPHOTO` at `Hy=0.2`, RMS wavefront
  error `0.0127` waves at the observation plane (Maréchal Strehl `0.994`).

Two of the four negative controls returned a **vacuous** detection margin —
found by M3.8's own required blind-spot audit, not hidden: `axis_transpose`
(`1.0000066×`, because the scoring metric azimuthally averages about the grid
centre and cannot distinguish a peak at `(114,0)` from one at `(0,114)`) and
`amplitude_weight_omitted` (`1.0×`, because a hexapolar fan's weights are
near-uniform up to a global scale). `opl_sign_flip` (`38.9×`) and
`propagation_distance_sign` (`39.3×`) were not vacuous.

## Off-axis OPD reference omits the field tilt (CHE-41)

Found by M3.8's off-axis vehicle: the declared pupil OPL carried `8.7e-5` of the
`0.0684` linear tilt the geometry required — `0.13%` of it. The reference-sphere
fit found its centre on-axis, `209 µm` from where the rays actually converge;
the shipping PSF landed `1` pixel off target instead of `114`. Root cause:
Optiland seeds `opd_native`'s accumulator on a plane perpendicular to z, not on
the incoming tilted wavefront — invisible on axis, where M3.1/M3.3/M3.4 all
validated. Fixed by referencing the OPL to the incoming wavefront
(`n_object · (d0 · r_launch)`); harmless for M3's frozen on-axis configuration,
and is why the `axis_transpose` control above needed a synthetic injected tilt
in M3.8 rather than a real one.

## Absolute carrier phase breaks `complex64` (M3.2A, CHE-40)

M3.2 measured Chromatix's `complex64` angular spectrum against a float64
reference and found relative field error growing as `eps32·2πz/λ`: `2.5e-5` at
`40µm`, `6.3e-2` at `47mm` — enough to reject a candidate `48mm`-focal-length
reference singlet outright. M3.2A's question was whether that is a property of
the wave engine or of the *number being represented*: the propagator's phase
magnitude is `kz` (`~5.4e5 rad` at `47mm`), but it factors exactly as
`exp(ikz)·exp(iz(k_z−k))`, and only the second, much smaller factor
(`~2.5e3 rad` at the same `47mm`, `200×` smaller) carries any diffraction — the
first factor is a global piston, invisible to intensity and to relative phase
along one path. Removing it and evaluating the second factor through the exact
identity `k_z − k = −(k_x²+k_y²)/(k_z+k)` (never by subtracting two nearly
equal numbers) fell the raw intensity error at `47mm` from `8.2e-3` to `3.9e-6`.
For the frozen `M3-SINGLET-REF` system specifically, the diffracting term is
`578 rad` against a `5.4e4 rad` carrier — `93×` smaller, and the number
`float32` actually has to round. Implemented in `chromatix_carrier_removed.py`;
the removed carrier is retained as float64 metadata and never reapplied.

## The exit-pupil test was out of contract, and the sensor-plane handoff is verified (M3.9R, CHE-38)

CHE-38's central finding, and the reason M3.8's `15×`-over-gate result does not
indict the coupler: `C_RAY_TO_WAVE`'s wavelet sum has no support term and no
explicit `z`, so it cannot return a hard pupil edge — it is not a pupil
reconstruction operator and was never validated as one. Re-tested at its
**intended** observation-side handoff (the declared sensor plane, `z =
4.9056mm`), across a `993×` range of ray counts (`3,169 → 3,148,801`), the
reconstruction **converges monotonically to the independent wave oracle with no
floor and no turn-around** — M3.9's rising branch is reclassified as an
exit-pupil misuse artifact. The residual that remains is a **per-ray area
weight at the aperture boundary**, not a kernel defect: a diagnostic radial
trapezoid weight on a synthetic aberration-free bundle collapses it from
`1.85e-3` to a converged `4.07e-4`, inside gate. The exact sensor plane is a
caustic in the position-space sense and is nevertheless where this operator is
*best* conditioned — it reads ray directions and optical paths, never a local
ray density.

## Per-ray quadrature weight in production, and absolute power resolved (CHE-47)

`couplers.quadrature` implements CHE-38's measured
radial-trapezoid weight as an absolute area (`π a² / (3·rings²)`, `3/4` at
centre, `1/2` at the outer ring), folded into `declare_coherent_bundle`'s
amplitude by default. On the **real traced** (residually aberrated)
`M3-SINGLET-REF` system, the weight helps against O2/ASM (`1.58×` improvement
at 787,969 rays, `3.91e-3 → 2.48e-3`) but **hurts against O1**, the
gate-deciding oracle: uniform weight measures `9.21e-4` against O1 (inside the
`1.0e-3` gate), production weight measures `2.21e-3` (`2.4×` worse, and over
gate). Left panel of the figure below plots this trajectory across the full
217→787,969-ray ladder: the uniform curve (dashed) dips under the gate line at
the finest two rungs, the weighted curve (solid) plateaus at `~2.2e-3`
starting around 1,801 rays and never crosses it. This is the opposite ordering
from CHE-38's synthetic aberration-free diagnostic, where the weighted result
(`4.07e-4`) is the one inside the gate — real geometric aberration on
`M3-SINGLET-REF`, absent from that diagnostic, is the suspected cause, and is
tracked as an open item rather than decomposed here (CHE-48). The weight is
kept in production regardless, because it independently resolves CHE-33's
`N^2.0024` raw-power scaling: reconstructed discrete power now converges under
ray refinement (relative spread `7.2e-3` from 1,801 rays upward, fitted
exponent `-0.0098`) rather than growing as `(ray count)^2` — a property the
uniform configuration does not have (right panel below) and that no amount of
O1 agreement on this one system would substitute for.

![Sensor-plane residual and absolute-power convergence, from the L2-PSF-01 bundle](../outputs/M3/L2-PSF-01/plot.png)

---

## Findings that changed the conclusion

1. **M3.9's exit-pupil benchmark was testing an operation `C_RAY_TO_WAVE` never
   claimed to perform.** A ray at the exit pupil can be a sample of a
   finite-support wavefront without the same coherent-sum operator being a
   valid *reconstructor* of that support from survivor rays alone. This
   reclassified M3.8's `15×`-over-gate exit-pupil result from "the coupler is
   wrong" to "the benchmark was out of contract" — and is the reason M3 is not
   simply a story of successive tolerance tightening.
2. **The dominant sensor-plane residual is a producer-side quadrature defect,
   not a kernel defect**, and it is attributable in kind (a hexapolar fan's
   outer ring represents only half its cell) even though its exact exponent
   (`rings^-0.83` vs the naive `rings^-1`) is not fully explained.
3. **A negative control that cannot fail proves nothing — twice, independently,
   in M3.** M3.8's own blind-spot audit found `axis_transpose` and
   `amplitude_weight_omitted` vacuous *before* shipping, mirroring M2's F1–F3
   pattern of a later check catching an earlier one's blind spot. CHE-41 then
   found a *third*, more consequential instance the audit had not covered: the
   off-axis OPD reference itself, which no on-axis validation could have
   caught by construction.
4. **A protocol constant can be wrong for years and pass every test that never
   exercises its numeric value.** `airy_radius_um` was a diameter, not a
   radius, until M3.4 needed the true value for an oracle and found the
   discrepancy — nothing upstream had ever computed with it.

## Defects found and fixed during M3

1. `airy_radius_um` was `1.22·λ/NA` (diameter) mislabeled as the radius
   (`0.61·λ/NA`) — CHE-33, propagated into `airy_radius_in_pixels` too.
2. Off-axis OPD reference omitted the incoming wavefront's tilt — CHE-41.
   Harmless on-axis; would have silently mis-pointed any off-axis PSF.
3. Absolute carrier phase drove `complex64` intensity error to `8.2e-3` at
   `47mm` — CHE-40, fixed by carrier removal.
4. The per-ray quadrature weight was missing at the aperture boundary,
   inflating the effective aperture and narrowing the reconstructed PSF —
   CHE-38 diagnosed, CHE-47 fixed in production.
5. **This ticket's own claim-lock test broke on the fix it was meant to
   protect against regressing.**
   `tests/benchmarks/test_l2_coupler_bundle.py::test_the_manifest_records_which_l2_benchmarks_are_actually_implemented`
   asserted `psf["implemented"] is False` — correct through M3.9R, wrong the
   moment `manifest.yaml` flipped. Updated to assert `implemented is True`,
   `protocol_id == "M3-SLICE-CPU-V1"`, and that the manifest's own note still
   names the `1.0e-3` gate rather than merely dropping the assertion.

---

## Claim audit

`verify_m1_independence.py`'s 13 checks were reviewed against the M3 tree and
**needed no changes** — re-confirmed passing 13/13. None of them assert
anything about the end-to-end graph; the stale "still unverified" claim named
by this ticket lived in `reports/2026-08/coupler_characterization.md`'s prose, not in an executable
check. That prose is amended: the line naming "the end-to-end Optiland→Chromatix
graph" as still-unverified is replaced with the narrower claim this milestone
actually supports — the sensor-side handoff is verified, the absolute
physical-correctness gate on the real traced system is not — and the
superseded "L1 — still blocked" risk section is annotated closed, per M2's own
CHE-29 precedent for the `wave_to_ray_not_claimed` check (replaced by four
narrower claims rather than deleted, not weakened).

`C_FIELD_TO_PSF`'s absence as an architectural primitive was already correctly
asserted by CHE-36 — `registry/couplers.yaml`'s comment, `test_graph_validation.py`'s
retired-id rejection, and `test_m3_psf_measurement.py`'s direct text-absence
check. This audit adds one more: `test_l2_psf_bundle.py::
test_the_registry_still_has_no_c_field_to_psf_entry`, so the L2-PSF-01 bundle's
own test file carries the assertion rather than depending on a distant test
file to keep it honest.

`derivative.verified` stays `false` everywhere, asserted by tests:
`verify_m1_independence.py`'s `ray_gradient_unverified` / `wave_gradient_unverified`
/ `ray_to_wave_gradient_unverified` checks (registry-level, re-verified passing
above), and `L2-PSF-01`'s own `differentiability` section
(`test_l2_psf_bundle.py::test_differentiability_is_characterized_not_promoted`).
No new gradient claim is made anywhere in M3.

`benchmarks/manifest.yaml`'s `L2-PSF-01` entry is flipped to `implemented: true`
with `blocked_by` removed (the blocker — Optiland's OPD convention — is gone)
and a note naming exactly what remains open: the `1.0e-3` gate, not met on the
real traced system, with the CHE-47 numbers quoted directly so the claim cannot
drift from the evidence that supports it.

---

## Risks and known limitations

### L1 — The physical-correctness gate is not met on the real traced system

`2.2e-3`–`2.5e-3` measured against `1.0e-3`, at 787,969 rays, with the per-ray
quadrature weight applied. CHE-47 attributes roughly half of the gap to the
weight mechanism (which fully explains the synthetic aberration-free case) and
leaves the rest open, with the likelier candidate being the O2 oracle's own
ring-averaged pupil-fit resolution rather than a residual coupler defect (O1,
sharing no traced data, sits *closer* to the weighted result than O2 does,
which is the wrong direction for a genuine aberration-sensitive defect). Not
closed here. See "What M4 should carry forward" #1.

Sharper statement of the same limitation: on this real traced system, the
production quadrature weight is the reason the gate is missed. Uniform
weighting alone reaches `9.21e-4` against O1 (inside gate); production
weighting reaches `2.21e-3` (outside gate) at the same 787,969 rays
(`outputs/M3/L2-PSF-01/plot.png`, left panel;
`tests/test_m3_quadrature_weight.py::test_the_weighted_result_does_not_improve_on_uniform_weights_vs_o1`
asserts this ordering so it cannot silently flip). The weight is not reverted
because it is independently load-bearing for absolute-power convergence
(right panel; `N^2.0024 → -0.0098` fitted exponent) — a property no O1
agreement on one aberrated system substitutes for — but the ordering means
CHE-47's own success criterion ("confirm the weighted result remains below the
gate across ray refinement") is not met on the real system, only on CHE-38's
synthetic aberration-free diagnostic. CHE-48 (decompose the residual) and
CHE-49/CHE-51 (explain the fitted exponent and separate `N_f` from NA) are the
open follow-ups; none is closed by this report.

### L2 — The negative-control blind-spot audit is not general

M3.8 found two vacuous controls and fixed its own test configuration; CHE-41
found and fixed a third, more consequential instance (the off-axis OPD
reference) that no on-axis audit could have caught. CHE-44 — generalizing
CHE-41's second finding to ask whether the same centre-dependent blindness
affects other metrics — remains in Backlog, not started.

### L3 — Off-axis is characterized at exactly one field point

CHE-41 verified the off-axis handoff at `Hy = 0.2` on `M3-REVERSE-TELEPHOTO`
only. CHE-42 (a field scan past the admissible sampling pitch) is scoped and in
Backlog.

### L4 — The finite-object OPL launch reference is untested

CHE-41 established the infinite-object launch-plane correction; the analogous
term for a finite object is untested (CHE-46, Backlog).

### L5 — CHE-38's own consolidated probe record has never landed

`benchmarks/probes/records/m3r_sensor_handoff.json` does not exist; the report
in `benchmarks/reports/2026-08/sensor_handoff_convergence.md` is evidence-complete from staged
runs but not from one reproducible artifact. This is why 21 tests in
`test_m3r_sensor_handoff.py` skip in the full suite. `L2-PSF-01` does not
depend on this record — it calls the same underlying probe code directly — but
CHE-38's own acceptance criterion for it is still open.

**Disposition (CHE-62):** unchanged and now justified rather than merely noted.
Only those 21 assertions read the record; the skip message names the cause and
the regeneration command; regeneration is tracked in **CHE-63**. See
`benchmarks/reports/2026-08/slice_cleanup_disposition.md` item 1.

### L6 — The reconstructed sensor field carries no wavefront curvature term

`C_RAY_TO_WAVE`'s sum is linear in the transverse coordinate, so it emits no
`exp(ikr²/2R)` term. Invisible in `|U|²` (why the intensity residual
converges); **not** invisible to a caller who propagates the reconstructed
sensor field further. Any M4 consumer that does so must be told.

### L7 — The quadrature boundary-weight exponent is not fully explained

The fitted NA-excess falls as `rings^-0.83`, not the naive half-cell
prediction of `rings^-1`. Confirmed in kind, not in exponent.

### L8 — This report is written against an uncommitted working tree

`L2-PSF-01`'s own `provenance.json` currently records `dirty_worktree: true`.
M1 exited with a clean-tree bundle as its open L3; M2 discharged it
(`dirty_worktree: false` against `86d916f2`). This report does not claim that
discharge for M3 — the entire M3.1–M3.10 body of work (CHE-30 through CHE-47,
this ticket included) was uncommitted at time of writing and is committed
without a matching re-run. **M3 is closed with this open**, unlike M2: the
bundle's `provenance.json` still reflects the pre-commit tree state, not the
citable commit it now lives in. Re-running the bundle against that commit and
re-confirming `dirty_worktree: false` with an unchanged fingerprint is carried
forward rather than blocking this close-out.

**Disposition (CHE-62): still open, now tracked in CHE-63.** See carry-forward
#7 below and `benchmarks/reports/2026-08/slice_cleanup_disposition.md` item 2.

### L9 — No performance envelope for `L2-PSF-01`

Unlike `L2-COUPLER-01`, this bundle records no timing section — out of scope
for CHE-39's acceptance criteria, and not invented to look more complete than
what was asked for.

### L10 — Carried forward unchanged from M1/M2

GPU/TPU execution, vector and polarized fields, chromatic coupling, partial
coherence, and conformal-surface coupling remain unverified. M3 touched none of
them. Caustics: M3 showed the *one* caustic it encountered (the on-axis sensor
plane at focus) is not ill-conditioned for this operator — this is not a
general claim about caustics.

---

## What M4 should carry forward

1. **Decompose the unattributed half of the real-system residual** before any
   promotion past "characterized". CHE-47's own suggested experiment — refit
   O2 at higher resolution/order and see whether the gap to O1 closes — is the
   concrete next step, not a repeat of the ray-count ladder.

   **Status (CHE-62): still open, and still exactly this.** CHE-48 was opened for
   this experiment and marked `Done` with no comment, no commit and no artifact —
   the decomposition was never performed. CHE-48 is reopened; the `1.0e-3` gate
   above remains unmet and is carried into M4 as an explicit open limitation. Per
   PB7 (CHE-58) finding F2, it must not be closed against another Optiland PSF
   method. See `benchmarks/reports/2026-08/slice_cleanup_disposition.md` item 3.
2. **The negative-control blind-spot audit needs to be a standing practice,
   not a one-time fix.** Three independent instances in M3 alone (M3.8's two,
   CHE-41's off-axis one) found a control that could not fail. CHE-44
   generalizes one of them and is unstarted; treat "can this control actually
   fail here?" as a required question for every new configuration, not only
   when a symmetry is suspected in hindsight.
3. **The gradient boundary stays `forward_only`.** No ticket in M3 touched
   differentiability; `derivative.verified` is `false` everywhere it is
   declared. M4 gradient work needs a custom derivative plus a directional
   finite-difference test *before* any promotion — per AGENTS.md, not as a
   milestone convenience.
4. **Off-axis and finite-object work has two scoped, un-started tickets
   waiting** (CHE-42, CHE-46). Do not re-derive their scope from scratch; pick
   them up.
5. **State the missing wavefront-curvature term to any consumer that
   propagates the sensor field further.** It is invisible to every metric this
   milestone gates on and is not invisible to the next thing built on top.
6. **A protocol constant's numeric value can go untested for an entire
   milestone if nothing ever computes with it.** `airy_radius_um` was wrong
   from M3.2 through M3.3 because no oracle needed the true value until M3.4.
   When freezing a numeric constant, exercise it once against an independent
   formula before relying on downstream tests to eventually notice.
7. **Re-run `L2-PSF-01` against the now-committed tree for a citable,
   clean-tree fingerprint** (L8), discharged after the fact rather than before
   this close-out — before extending the slice, not after.

   **Status (CHE-62): not discharged; M3.5 extended the slice first.** One thing
   did change: until `ee57e33` the worktree was never clean (PB7's probe and two
   reports were untracked), so `dirty_worktree: false` was *unreachable* — the
   re-run could not have satisfied its own criterion whenever it was launched. It
   now can. Tracked in **CHE-63**, which also requires committing the provenance
   out of gitignored `outputs/` so a third recurrence is not possible. See
   `benchmarks/reports/2026-08/slice_cleanup_disposition.md` item 2.
