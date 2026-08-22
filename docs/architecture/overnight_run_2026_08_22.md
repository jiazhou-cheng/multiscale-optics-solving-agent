# Overnight run, 2026-08-22 — CHE-84…CHE-94 cleanup, then CHE-95/CHE-96

Progress log for a single autonomous session. Two priorities in order: finish
the cleanup epic and leave the tree stable, then build the new feature on top of
it and demo it on a GPU. A hard constraint ran across both: stop any
memory-intensive process the moment host swap starts growing.

## Priority 1 — CHE-84 … CHE-94 (cleanup epic)

Complete. The full accounting, phase by phase, is
[`cleanup_baseline.md`](cleanup_baseline.md): baseline 769 passed / 48 skipped /
183 s at commit `b5fbe42`, exit 899/48/188 s, +130 tests, coverage 68.8% → 79.6%,
and not one existing test changing outcome. Four pre-existing defects were found
by *running* things the epic required to be run and were filed rather than fixed
in a cleanup phase: **CHE-97**, **CHE-98**, **CHE-99**, **CHE-100 (Urgent)**.

Two scientific fingerprints were held constant across every phase, which is what
makes "behaviour preserved" a measurement rather than a claim:

| Fingerprint | Value |
| -- | -- |
| Canonical trace array hash | `a84fe53f6184c097072bce9ef4c245470f865cf4f3099d492bc3a7afe6f3434a` |
| L2-PSF-01 | `b073a4616c0fda245dace0ef77ac46f4ca7efe065bef7db839b5652fc9cc0dab` |

## Priority 2 — CHE-95, CHE-96 (new feature)

### CHE-95 — the batched planar DOE step as a graph node

Complete before this log's window; `C_PLANAR_DOE_STEP` gained a capability
declaration, a registry entry and `couplers/doe_node.py`. It is used below as
demo3's RW-F route.

### CHE-96 — patch-based local WFT, and the paper's Fig 5b/5c reproductions

Landed in four commits.

| Commit | What |
| -- | -- |
| `c2d21ae` | `couplers/patch.py` + `tests/test_patch_wft.py` (the fast guard) |
| `da2e757` | capability, registry entry, `couplers/patch_node.py`, `reference_data` manifests |
| `5f48039` | demo2 both routes on GPU + the cost sweep |
| `8fbccfa` | demo3 both routes on GPU + the convergence ladder |

#### Results

**demo2** (Fig 5b), one RTX A6000 — the same GPU class as the paper's:

| Route | Budget | NCC | rel-L2 | Wall clock |
| -- | -- | -- | -- | -- |
| RW-F, enumerated | 39,601 rays | 1.000000 | **7.11e-13** | 2.9 s |
| RW-F, Table S2 budget | 1.1e6 rays | 0.998693 | 8.87e-2 | 2.8 s |
| RW-P, Table S2 budget | 1.6e8 rays | 0.999418 | 2.86e-2 | 94.9 s |

Row 1 is AC 1 on the paper's own system. Row 3 agrees with row 1 at NCC 0.999418,
which is AC 5. The 5×3 cost sweep is monotone in both incident and secondary
count. The paper's 0.997 is quoted as context; it is a different implementation
and is not a threshold.

**demo3** (Fig 5c) — a characterization, never a validation, because the paper
states no conventional reference exists. Neither route converges at any budget
that fits, and the deliverable is the proof that this is noise rather than
disagreement:

* **Noise-limited agreement.** For two unbiased estimators of one field,
  `NCC(A,B) ≈ √(NCC(A,A′)·NCC(B,B′))`. Predicted 0.0129, measured 0.0147,
  **ratio 1.14**.
* **Convergence slope.** Seed-to-seed NCC against ray count fits a log-log slope
  of **0.956** — linear, i.e. noise more rays remove.
* **Extrapolation.** NCC 0.9 needs **1.78e9 rays, 1.2 h per run**. The paper's
  own RW-P budget for this system is 2.6e9. Landing within 1.5× of their choice
  is independent confirmation that the budget is a property of the system.
* **Energy.** Transmitted fraction 1.0000000 through the Optiland trace, zero
  rays clipped with power.

#### Things that had to be got right, and were got wrong first

Each was found by measurement, and each produced a *plausible* wrong field —
which is why they are recorded rather than quietly fixed.

1. **`resolve_pad_px` looped forever on an even `patch_px`.** "Odd pad" and
   "even (pad − patch)" cannot both hold for one. The underlying reason is
   physical: an even patch has no centre sample. Now refused, not rounded —
   the paper's own sizes (40, 50, 100) are all even.
2. **Clearance had one legitimate exemption and no more.** The full-aperture
   single patch *is* the window. Padding it moves the mode grid off the
   unpadded oracle's and the exactness anchor reads 0.57 instead of 1.4e-12.
3. **The oracle is not well defined until its padding is.** A route at pad 199
   scored against an oracle at pad 200 reads 8.8e-3; at pad 101, 0.33. Neither
   is an error in either implementation — both are wraparound between two
   periods. Every score now names its oracle's pad.
4. **Coverage was inverted** (`A_patch/A_draw` for `A_draw/A_patch`) and **a
   launch-position phase was double-counted.** Both are exactly 1 on the
   full-aperture anchor, which is how they survived until the sub-aperture
   relation was measured.
5. **Continuous patch centres** injected a sub-sample linear phase; the sweep
   plateaued at ~0.28 instead of converging. Centres are snapped to the grid.
6. **`configure_optiland_execution` must precede `build_optiland_system`.** It
   switches Optiland's global backend, and a lens built before it keeps its
   geometry in the old namespace. The trace then fails as
   `numpy.ndarray * Tensor` one frame inside an optiland geometry class.
7. **The first demo3 convergence ladder started below the noise floor.** An NCC
   from 1.76e5 pixels has a standard error near 0.0024; the bottom rungs
   measured 9e-5 and **−3.5e-4**, and fitting through them returned a slope of
   5.7 and an extrapolation an order of magnitude too optimistic. The ladder now
   starts above 3/√N_px.
8. **JAX silently truncates a requested float64 to float32.** The probes enable
   x64 before the first array and then read the dtype back off the array,
   refusing if it disagrees.

#### The one thing deliberately not delivered

demo3 at SI Table S2's ray budget (5.3e9 for RW-F, 2.6e9 for RW-P). Our
reconstruction is O(N_rays × N_pixels) — `ray_to_wave` contracts a separable
einsum — against upstream's O(N_rays) + one FFT. The shortfall is measured, not
estimated, and is reported as the evidence for the k-space fast path on CHE-95,
which is how the issue itself asks for it to be treated.

## Resource events

**One swap event, and the constraint fired as intended.** While
`tests/test_patch_wft.py` was being written, its enumeration case at the anchor's
33-px grid was 3.7 M rays and ~4 GB in a single call, and a companion sweep was
sized at 51.8 M rays. Host swap grew from 266 MiB to 717 MiB. Every container was
stopped, the cause was diagnosed, and the test was cut to a 15-px grid — 159 k
rays, ~76 MB — with the reasoning in its docstring, because a fast guard has no
business allocating gigabytes and the cost curve belongs in a probe. Swap does
not page back automatically and has sat at **693–697 MiB, unmoving, through every
GPU run since**, including the 1.6e8-ray demo2 and the six 60 M-ray demo3 runs.

**Two 10-minute command timeouts**, both leaving an orphan container, both
stopped explicitly before retrying. They are why the demo3 route agreement is
computed from saved fields in a second step rather than in one process: a
converged run of each route is ~150 s and six of them do not fit in one command,
and the alternative was to shrink the budget until the comparison stopped
meaning anything.

**GPU 5 remains faulted at NVML level** and blocks `nvidia-container-cli` for
every GPU container on this host. `run.sh`'s explicit device-passthrough fallback
carried every GPU run in this session. Recorded as Trap 3 in
`docs/testing/gpu_environment.md`.

No GPU job was ever run concurrently with another, nothing was backgrounded, and
swap, mounts, drivers and systemd were not touched.

## Test state at the end of the run

| Command | Result |
| -- | -- |
| `./run.sh pytest -q` | **958 passed, 48 skipped, 195 s** |
| `tests/test_patch_wft.py` | 33 tests, 0.8 s — the fast guard AC 9 asks for |
| `./run.sh --gpu pytest -q -m gpu` | unchanged from CHE-72/CHE-73: 48 passed |
