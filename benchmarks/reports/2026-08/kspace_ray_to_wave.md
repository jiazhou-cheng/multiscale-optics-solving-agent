# CHE-101 — the k-space ray→wave fast path, and what it did not fix

`C_RAY_TO_WAVE` gained a second evaluation strategy for the same operator:
`Reconstruction.KSPACE_SPLAT`, a bilinear splat into a k-grid followed by one
inverse FFT, against the existing `Reconstruction.RAMP_SUM`, which evaluates each
wavelet's phase ramp directly in real space. The exact route remains the default
and the oracle.

The headline is not the speedup. It is that **the stage this ticket optimized was
7% of demo3's cost**, and CHE-96 attributed the whole of it there.

## What was measured

### The reconstruction kernel alone, on one RTX A6000

1e6 rays onto a 420×420 sensor, complex64, repeated calls after warm-up. Timed
with a scratch probe that is deliberately not committed — a wall-clock number is
not reproducible across hosts, so what the suite asserts instead is the
*structural* claim (no rays×pixels factor is ever formed, and per-ray work is
identical across grid sizes) in `tests/test_ray_to_wave_kspace.py`:

| route | k-grid | seconds |
| -- | -- | -- |
| `ramp_sum` | — | 0.183 |
| `kspace_splat`, 1.5× | 630² | 0.012 |
| `kspace_splat`, 8× | 3360² | 0.019 |

**9.6× on the kernel, and flat in k-grid size** from 2× to 8×, so oversampling
accuracy is effectively free. Per-ray cost no longer scales with pixel count
(AC 4): the same 1e6 rays splat identically onto a 32² and a 128² grid, only the
FFT differs, and `tests/test_ray_to_wave_kspace.py` asserts that structurally by
making `xp.outer` and `xp.einsum` raise.

### demo2 (paper Fig 5b) — the fast route is *exact* here

demo2's secondary rays are drawn from `fftshift(fftfreq(pad, d=pitch))`, so every
transverse wavevector is an exact bin of the padded grid. Choose a k-grid whose
period is a whole number of spectral periods and the bilinear weights collapse to
(1, 0): the splat becomes a relabelling, not an approximation.
`_demo_support.matched_kspace_grid` derives that grid and the record reports
`on_node_fraction` so the claim is measured, not assumed.

| route | budget | exact route | fast route |
| -- | -- | -- | -- |
| RW-F, enumerated | 39,601 rays | 7.1147e-13 | **7.1153e-13** |
| RW-F, Table S2 | 1.1e6 rays | 0.998693 / 8.8716e-2 | 0.998693 / 8.8716e-2 |
| RW-P, Table S2 | 1.6e8 rays | 0.999418 / 2.8562e-2 | 0.999418 / 2.8562e-2 |

Every committed number reproduced (AC 2). demo2 is **not** faster (95.0 s against
92.5 s for RW-P): at 1e4 pixels the reconstruction was never the bottleneck, and
the k-grid is 4× the output grid. Worth stating plainly — the fast path's benefit
is in pixels, and demo2 has few.

### demo3 (paper Fig 5c) — where the time actually goes

Rays here are refracted by the singlet before the sensor, so their directions are
continuous, no k-grid puts them on nodes, and the splat genuinely interpolates.
Measured against the exact route on **identical rays** (same seed, same trace):

| k-grid oversampling | NCC | rel-L2 field |
| -- | -- | -- |
| 1.5 (upstream's default) | 0.914622 | 2.63e-1 |
| 2 | 0.969218 | 1.58e-1 |
| 4 | 0.997929 | 4.21e-2 |
| 8 | 0.999870 | 1.07e-2 |

Upstream's default leaves **9% intensity error** on this system, so it is not used
at that value; 8× is, and costs nothing (above). At the full 60 M-ray budget the
same comparison reads NCC 0.999868, rel-L2 1.063e-2, power ratio 0.9832 — the
interpolation loses 1.7% of the power, and the residual grows off-axis, which is
the splat kernel's signature rather than a ray-count effect.

Then the stage breakdown of one 60 M-ray run, 420×420, exact route:

| stage | seconds | share |
| -- | -- | -- |
| Optiland trace | 99.7 | 48% |
| emit patch spectra | 86.9 | 42% |
| **reconstruction** | **14.4** | **7%** |
| host→device | 3.5 | 2% |
| probe bookkeeping | 1.0 | 0.5% |

So the whole run goes 207 s → 197 s. A 9.6× improvement to a 7% stage cannot do
more, and **demo3 still does not converge**: the ladder rerun through the fast
path gives essentially the same estimator as before.

| rung | exact NCC | fast NCC |
| -- | -- | -- |
| 2.0e7 rays | 0.01037 | 0.01086 |
| 3.0e7 rays | 0.01467 | 0.01526 |
| 4.0e7 rays | 0.02022 | 0.02074 |
| log-log slope | 0.956 | **0.927** |
| rays for NCC 0.9 | 1.780e9 | **1.736e9** |
| hours per run | 1.19 | **1.09** |

That the seed-to-seed noise is unchanged (exact route 0.0385–0.0469, fast route
0.0386–0.0470 across three seeds at 60 M rays) is the anti-bias result the ticket
asked for: a fast path that had biased the estimator would have moved it.

## The two defects this found, both silent

1. **The outermost spectral bins were dropped.** A mode at k-index `-K//2` has a
   fractional grid index of exactly 0 in real arithmetic and `-1e-14` in floating
   point, so a bare `>= 0` bound discards it. On demo2's enumerated patch that
   dropped 397 of 39,601 rays — precisely one row and one column — and reported
   8.5e-2 where the answer is 7.1e-13, with `on_node_fraction` still reading 1.0
   for the survivors. Nothing else in the record would have said why. Fixed by
   testing the bound with a tolerance and clipping; regression-tested at three
   k-grid parities.
2. **Compacting the ray list made the fast path slower than the exact one.**
   `fy[representable]` is a data-dependent shape, which under JAX forces a host
   synchronization and a gather mid-pipeline: every chunk then had to finish
   tracing before its splat could be enqueued. Measured 194 s against 145 s —
   the *fast* route losing — while the kernel in isolation was already quicker.
   Fixed by zero-weighting unrepresentable rays instead of removing them, and
   fusing the four corner scatters into one, keeping every shape static.

Both produced a plausible field, which is why they are recorded rather than
quietly fixed.

## Deliberate divergences from upstream

| Upstream | Here | Why |
| -- | -- | -- |
| crop at `(K - n) // 2` | `K//2 - n//2` | Upstream's is a sample off this repository's `n // 2` origin whenever the grids differ in parity — a half-pixel tilt, not a visible failure. Checked over every parity combination. |
| drop rays in the top k-bin (4-neighbour footprint) | clamp the upper neighbour | Its interpolation weight is zero there, so clamping keeps a ray the exact route uses. |
| drop rays landing outside the sensor | not ported | A wavelet contributes a ramp across the whole plane regardless of where it crosses it. That crop is a sensor model, and the exact route has none; porting it would make the two routes disagree for a reason unrelated to the algorithm. |
| obliquity factor hard-coded | caller's `Projection` wins | `ASM_CONSISTENT` is this repository's coupler default for the reason CHE-25 measured (7.1e-15 against 2.2%-of-peak). |
| unrepresentable rays dropped silently | counted and reported | A field missing a third of its rays and one missing none must not produce the same record. |

## A dead end worth recording

The bilinear splat is a convolution with a triangle kernel in k, so its real-space
envelope is analytically `sinc²(π m / K)` and can be divided out — the standard
NUFFT deapodization. Measured gain: **2.0–2.6×** on the field error, at every
oversampling. Not worth a production code path, because the residual is *aliasing*
from the 2-tap kernel's tails, not the envelope, and oversampling buys an order of
magnitude for free. A wider gridding kernel (Kaiser-Bessel) is the real lever if
this ever needs to be cheaper than 8× oversampling.

## What is now the binding constraint

Not the reconstruction. To reach the paper's Fig 5c budget, the next ticket has to
address the emitter (87 s, host-side NumPy patch-spectrum sampling, transferred to
the device per chunk) and the Optiland trace (100 s for 60 M rays through four
surfaces). Those are 90% of the run, and neither is a coupler.

## Provenance

`deeplens/raywave.py::RayWave.huygens_psf` was read in full at revision
`777e753be7778b09bf13fb55c633a56ac4ad04e5` on 2026-08-22 and its *algorithm* was
implemented here. Still never vendored and never executed. The equations remain
the paper's — a plane wavelet is a k-space delta by SI S2's own construction,
which is why the ported route reproduces the exact route's full-enumeration limit
to 7.1e-13 rather than merely resembling it. Upstream ships no tests, so nothing
was inherited as verified: every claim above is measured against this
repository's exact route or its independent float64 ASM oracle.
Recorded in `knowledge/couplers/ray_to_wave/source_manifest.yaml` and asserted by
`tests/test_coupler_knowledge_pack.py::test_the_ported_reconstruction_algorithm_is_recorded_with_its_revision`.

## Artifacts

| What | Where |
| -- | -- |
| demo2 through the fast path | `benchmarks/probes/records/ray_wave/demo2_paper_kspace_jax.json` |
| oversampling sweep, 420² | `.../demo3_equivalence_characterization_jax.json` |
| demo3 fast path, 3 seeds, 60 M rays | `.../demo3_kspace_rw_p.json` (+ `_fields.npz`) |
| demo3 exact route with stage timings | `.../demo3_stage_ramp.json` |
| convergence ladder, fast path | `.../demo3_convergence_kspace_rw_p.json` |
| figures | `outputs/CHE-96/demo3_kspace_vs_exact.png`, `demo2_fig5b_sensor_fields.png`, `demo3_fig5c_sensor_fields.png` |

## Tests

`./run.sh pytest -q` — **986 passed, 48 skipped, 198 s**, from a 958-passed
baseline; +28, and no existing test changed outcome. New:
`tests/test_ray_to_wave_kspace.py` (27) and one knowledge-pack provenance test.
`./run.sh --gpu pytest -q -m gpu` in its own session: 48 passed. Not run: the
tutorial suite, since no dependency pin changed.
