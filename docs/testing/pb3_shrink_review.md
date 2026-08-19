# PB3 — Tier A Numerical-Cost Review and Command-Surface Documentation

**Issue:** CHE-54 (PB3), milestone M3.5
**Date:** 2026-08-18
**Consumes:** `docs/testing/tier_restructure.md` (CHE-53/PB2) — this issue only acts
on tests PB2 kept in Tier A.

## Review outcome: no shrink needed

CHE-53 (PB2) measured Tier A at 31.08s for 499 tests — already ~6x under the
≤3-minute (180s) gate before this issue started any work. This issue's job was
to check whether that headroom is hiding individually oversized fixtures that
happen to average out cheap, not just to confirm the aggregate number.

**Runtime check.** Re-running Tier A with `--durations=20` (top 20 slowest,
regardless of the 0.005s default hide-threshold) shows a maximum single-test
time of **0.31s**
(`test_m3_pupil_to_focus.py::test_the_carrier_removed_path_marks_its_absolute_phase_unphysical`).
Every other test in the top 20 is ≤0.26s. There is no outlier.

```
478 passed, 21 skipped, 128 deselected in 30.85s
```

(21 skipped = PB1 Finding F3, the still-unaddressed missing
`benchmarks/probes/records/m3r_sensor_handoff.json`; unrelated to this issue.)

**Static check, independent of this machine's speed.** Grepped every Tier A
test file for the large-ray-count / large-grid / dense-sweep patterns CHE-54's
acceptance criteria names (`N_GRID = <n>`, `GRID = (<n>, <n>)`,
`num_rings >= 100`, `ray_count`/`n_rays >= 1000`, `range(>=1000)`), excluding
files already outside Tier A (`test_coupler_gradient.py`, `test_wave_to_ray.py`,
`test_coupler_round_trip.py`, `test_m3r_sensor_handoff.py`, `test_m1_protocol.py`,
`test_chromatix_adapter.py` — all already `slow`- or otherwise-excluded per PB2).
Four hits, all modest and already cheap:

| File | Grid | Why it's fine at this size |
|---|---|---|
| `test_optiland_coherent_handoff.py` | `GRID = (64, 64)` | Chromatix FFT at 64×64 is sub-millisecond; the test is about the OPL/amplitude declaration, not grid convergence |
| `test_carrier_removed_asm.py` | `GRID = 64` | Same — carrier-removal correctness doesn't depend on grid size, per that file's own docstring (CHE-40) |
| `test_m3_psf_measurement.py` | `GRID = (9, 11)` | Deliberately small and non-square by design (see the file's docstring: symmetry would hide an axis-transpose bug) |
| `test_ray_to_wave.py` | `GRID = (32, 32)` | Analytic-oracle reconstruction test; 32×32 is already the minimum that resolves the test's own tolerance derivation |

`test_quadrature.py`'s `test_area_weight_sums_to_aperture_area_in_the_limit[512]`
(512 hexapolar rings) is pure NumPy arithmetic with no solver in the loop — 0.20s
— and is exactly the kind of "manufactured analytic oracle, cheap despite a large
N" case CHE-54's acceptance criteria treats as already acceptable.

**Conclusion:** PB2's marker-based exclusion (routing every genuinely expensive
characterization, convergence study, and out-of-scope-solver test to `slow`,
`benchmark`, `fmmax`, `fdtdx`, or `sax`) already did the job this issue exists to
do. Nothing in Tier A needed shrinking, and nothing needed reclassifying out of
Tier A during this pass — the reclassification CHE-54 anticipates ("if any test
cannot be cheapened... it is reclassified to Tier B... rather than force-fit")
happened one issue early, in PB2, because PB1's inventory already identified the
expensive tests before PB2 assigned markers.

## Required deliverables

- **Final measured Tier A runtime:** 30.85s (478 passed, 21 skipped, 128
  deselected), `--durations=20` output above. Consistent with PB2's 31.08s
  measurement (sub-second variance between runs, both far under budget).
- **Tests reclassified out of Tier A during this pass:** none. (All
  reclassification happened in PB2; see `tier_restructure.md`.)
- **Updated documentation distinguishing the tiers:** `AGENTS.md`'s new "Test
  Command Surface" section, added by this issue, states the required/
  subsystem-specific/full-regression commands in the same language the ticket
  format already uses, and links to `docs/testing/test_audit.md` and
  `docs/testing/tier_restructure.md` for the full rationale rather than
  duplicating it.

## Verification

```bash
./run.sh --no-build pytest -q --durations=20 -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"
```
