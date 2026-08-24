# Performance baselines and the harness that produced them

CHE-105 (M0.4). Written after the measurements, not before them.

## What this exists to prevent

CHE-96 attributed the whole of demo3's runtime to the reconstruction stage.
CHE-101 then made that stage 9.6× faster on the kernel and the end-to-end run
went `207 s → 197 s`, because the stage was 7% of the cost. The work was correct
and the target was wrong, and a stage-resolved baseline would have said so
before the effort rather than after it.

M5 is a whole milestone of that shape, so this one measures first.

## The finding

**At the ray count this repository actually uses, 91% of `C_RAY_TO_WAVE`'s wall
time is a sampling diagnostic, not physics.**

`couplers.ray_to_wave._ray_density_diagnostic` runs an O(N²) pairwise
nearest-neighbour scan below `_NEAREST_NEIGHBOUR_SCAN_LIMIT` (4096 rays) and is
skipped above it. Measured share of the call, at fixed 188² grid:

| rays | diagnostic runs | full call | reconstruction only | diagnostic share |
| --: | :-: | --: | --: | --: |
| 817 | yes | 0.038 s | 0.011 s | **70%** |
| 1801 | yes | 0.151 s | 0.023 s | **85%** |
| 3169 | yes | 0.451 s | 0.042 s | **91%** |
| 4921 | no | 0.066 s | 0.073 s | ~0 |
| 7057 | no | 0.108 s | 0.109 s | ~0 |
| 12481 | no | 0.201 s | 0.200 s | ~0 |

3169 rays is the frozen `M3-SINGLET-REF` configuration. Note the row above and
below it: **4921 rays runs 6.8× faster than 3169 rays**, because it crosses the
threshold and stops paying for the scan.

Two fits, because one exponent across that threshold describes neither side:

| fit | exponent | r² |
| -- | --: | --: |
| reconstruction only | 1.077 | 0.998 |
| ray-density diagnostic | 2.019 | 0.9999 |
| both together (do not use) | 0.373 | **0.176** |

The third row is the argument for the first two. It is also the argument for
fitting at all rather than quoting endpoints: two points would have produced an
exponent and an r² of 1.0 regardless.

Consequences:

* Optimizing the reconstruction kernel below 4096 rays moves at most ~9% of the
  wall time. **The diagnostic is the target there**, and it is not physics — it
  is a sampling check that could be sampled rather than computed exhaustively,
  or cached across calls on the same bundle. Handed to M5.2 (CHE-119) as a
  measured target.
* The reconstruction is **linear in rays at fixed grid** (1.077), consistent
  with the O(rays × pixels) product model that `RayToWaveCoupler.estimate()`'s
  docstring argues for, and not with the registry's `O(rays + pixels)`.

## L2-PSF-01 could not run

Attempting to baseline the bundle found it exits `1` in 0.5 s:
`run_benchmark.py` loaded `benchmarks/probes/m3r_sensor_handoff.py` and
`m3_quadrature_weight.py`, the names those probes had before the CHE-93
reorganization renamed them. The module docstring named the correct paths; only
the two constants below it were left behind.

Nothing noticed because no test invokes the bundle and the manifest records its
verdict rather than re-deriving it. Fixed here. On the repaired path it runs in
**172 s** and reproduces the state the claim ledger records:
`gate_met_on_production_configuration: false`, `negative_controls_pass: false`.

Its `scientific_fingerprint` is now `411181d0…` against the `b073a461…` in
`docs/architecture/overnight_run_2026_08_22.md`. Attributed to CHE-102, which
moved host traces from Optiland's torch backend to NumPy; that changes the trace
at float64 round-off and therefore the fingerprint. No test asserts the old
value.

This is also why the runner now **refuses to record a nonzero exit**. The first
attempt wrote a 0.5-second "bundle baseline" and was, as a number, perfectly
real.

## Framework overhead (S5)

Same physics, through the abstraction and directly:

| | framework | direct | overhead |
| -- | --: | --: | --: |
| `M_RAY_OPTILAND` — `adapter.run` vs `build_optiland_system` + `optic.trace` | 20.1 ms | 8.3 ms | **2.4×** (+11.8 ms) |
| `C_RAY_TO_WAVE` — `node.transform` vs `ray_to_wave` | 462 ms | 468 ms | **1.0×** (within noise) |

The solver's overhead is a fixed ~12 ms of validation, precision/device
negotiation, `.npy` write and artifact assembly. It matters for a graph of many
small trace nodes and is irrelevant for one large one. The coupler node's
overhead is below the measurement noise floor of a 460 ms kernel — a ratio
slightly under 1.0 is not a speedup, it is noise, and it is reported rather than
rounded up to 1.0.

## `estimate()` versus measured

| component | predicted | measured | verdict |
| -- | --: | --: | -- |
| `M_RAY_OPTILAND` | `None` | 20 ms | No prediction, and it says why: it does not import optiland and cannot know the surface or traced-ray count. An honest refusal, and it means a planner cannot order work by this estimator. |
| `C_RAY_TO_WAVE` | 0.204 s | 0.501 s full / 0.065 s reconstruction | **Wrong in both directions.** Under-predicts the shipping call by 2.5× because it models the reconstruction and the call is dominated by the diagnostic. Over-predicts the reconstruction by 3.1× because `_RAY_PIXEL_PRODUCTS_PER_SECOND` is not calibrated to this host. |

M3's executor and M6's planner both intend to order work by `CostEstimate`. On
this evidence neither can yet.

## Baselines recorded

| baseline | value | record |
| -- | --: | -- |
| default test suite | 205.5 s, 1149 passed / 48 skipped, peak RSS 0.32 GB | `suite_default_cpu.json` |
| L2-PSF-01 bundle | 172.5 s, peak RSS 0.32 GB | `l2_psf_01_cpu.json` |
| framework overhead | above | `framework_overhead.json` |
| ray-axis scaling | above | `scaling_ray_axis.json` |
| `estimate()` accuracy | above | `estimate_accuracy.json` |

All on CPU, `agent_solver`, 80 logical cores, no thread pinning
(`isolation.applied: false` — recording an affinity mask does not make a run
isolated). Swap growth zero throughout.

## Not done

**demo2 and demo3 were not run**, at the owner's direction during the ticket.
Their subcommands are implemented and exercised by the same `_timed_command`
path the suite and L2-PSF-01 baselines use, so nothing about the harness is
untested as a result — but two of the five named baselines are absent, and M5
cannot set targets for the demo3 stage split without them. Filed as a follow-up.

An interim CPU measurement taken before the deferral, for scale only and not
committed as a baseline: demo2 `paper` RW-F (1.1e6 rays) 6.6 s, RW-P (1.6e8
rays) 298 s, against the 2.8 s / 94.9 s the ticket quotes from GPU runs. The
gap is the device, which is exactly what the environment fingerprint exists to
make un-ignorable — `compare()` refuses to divide those numbers.
