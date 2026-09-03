# Overnight verification — ray tutorials, wave kernel sweep, Demo2/Demo3

**CHE-238** (parent) with workstreams **CHE-239** (A, ray), **CHE-240** (B, wave),
**CHE-241** (C, Demo2), **CHE-242** (D, Demo3).

This document is the parent's deliverable. It is written incrementally: each
workstream appends its own section when it runs, and every section states what
actually executed. A section that says a check was not run means it was not run.

---

## 1. Executive summary

All four workstreams ran. **No production code changed** — `src/` is untouched, and
every discrepancy found is recorded rather than fixed, because CHE-238's
code-change policy makes that the default and each of them changes a physical
claim or a frozen number.

**Three findings, in descending order of consequence.**

1. **`SOM_SPOT_DIAGRAM` analyses at the wrong wavelength** (§5.3). It builds its
   lens with `build_lens`, which declares exactly one wavelength —
   `setup.reference_wavelength_um` — then asks the solver for `wavelengths="all"`.
   The source wavelength never reaches the analysis, while the returned record
   labels the result with it. Confirmed, minimally reproduced, and invisible to the
   current suite because every existing test sets source = reference. This is a
   defect in a landed operation and a fabricated provenance field, not a
   characterization gap.
2. **A recorded "exactly 0.0" is overstated** (§6.2). Substituting one factor in
   `_carrier_removed_propagator` is claimed to reproduce the Fresnel kernel with
   maximum difference exactly zero; measured on the propagator arrays it is
   6.1e-5. The claim's substance holds — the substitution is algebraically exact
   and lands below the float32 floor — but the wording does not.
3. **A committed probe script is 40 626 NUL bytes** (§8.6).
   `demo3_enumerated_reference.py` on `che-140` has never contained code in git
   history; it was all-zero at its only content commit. That blocks one of
   workstream D's six evidence items outright.

**What holds.** All six native PSF analyses agree with Optiland bit-exactly on the
full intensity grid. All 13 expressible tutorials round-trip their surface table
and paraxial characterization exactly and trace to 0.0 against the tutorial's own
lens. Five of seven wave kernel checks reproduce, including all three paraxial-bound
cosines against a closed form. Demo2's RW-F exactness anchor holds to float64
round-off and RW-P converges to it monotonically across 4.2 decades of ray budget.
Demo3's estimator-variance model reproduces to under 1% in both its fitted terms;
its budget optimum moves with the timing constants (`S*` 1.92e4 against 2.15e4) and
still brackets the shipped `S = 20 000`.

**What the numbers cannot say.** Demo3's routes disagree at NCC 0.014 at a budget
where each route's own seed-to-seed NCC is 0.004–0.046, so the reconstruction is
noise-dominated and a bare cross-route NCC has no resolution on the optical model
(§8.1–8.2). The probe's own noise-limited-agreement statistic, computed through its
`--agreement-from` path over the full 3 × 3 cross-seed matrix, predicts 0.0158
against a measured 0.0163 — **ratio 1.03**, consistent with two noise-limited
estimates of the *same* signal, which is the strongest statement available without
converging either. The committed baseline shows the same picture, so this is the
existing state rather than a regression.

| Workstream | Status | Section |
| --- | --- | --- |
| A — ray tutorial / system regression | **run** — 38 Tier-1 rows, 96 Tier-2 rows; one confirmed defect | §5 |
| B — wave kernel sweep | **run** — 24 rows; a second overstated record found | §6 |
| C — Demo2 reproduction | **run, `PASS-native`** — Option B not needed | §7 |
| D — Demo3 characterization | **run, `PASS-native`** — 5 of 6 items; item 5 `BLOCKED` on a corrupt file | §8 |

---

## 2. Execution gate

The parent carries an explicit gate: *"DO NOT start overnight compute until it is
reviewed and approved"*, and an **OPEN DECISION** that blocks approval — no single
branch can run all four workstreams.

**Gate treated as opened by** the owner's overnight execution instruction of
2026-09-03, which names CHE-238 → 239 → 240 → 241 → 242 as the night's work and
adds two binding constraints of its own:

* commit to the current branch;
* **do not create or switch branches.**

Those constraints decide the open question, below. Nothing else in the ticket was
edited, split, or reinterpreted.

### 2.1 The OPEN DECISION, resolved

The parent's three options were: (1) split by branch, (2) descope A+B to the
pre-rewrite equivalents on `che-140`, (3) descope C+D.

Re-verified on 2026-09-03 at `241c783`, against the trees rather than the ticket
text:

| Claim in the ticket | Verified? | Evidence |
| --- | --- | --- |
| A/B surface exists only on `che-152-greenfield-rewrite` | yes | `src/operations/catalog.py`, `src/backends/optiland/analysis.py`, `src/backends/chromatix/solver.py` present here |
| C/D probe inputs exist only on `che-140` | yes | `git ls-tree -r --name-only origin/chengjiazhou4802/che-140-default-test-suite-375s-under-60s \| grep -c ray_wave` → **84**, all under `benchmarks/probes/`; the same grep against `241c783` → **0**. (Re-running that grep on a later commit returns 1, because this report's own filename contains `ray_wave`.) |
| `benchmarks/` here holds only `record.py` + `systems/` | **nearly** | it also holds `README.md` and `__init__.py`, and now this `reports/` directory. The load-bearing part of the claim — no `probes/`, no probe records, no `perf/` — is true. |
| the catalog has 15 descriptors | **stale** | it has 17 at `241c783`; CHE-228 added `O_FRESNEL_PROPAGATE` and CHE-236 added `SOM_PSF` after the count was taken. Verified separately: `backend="optiland"` appears on exactly four descriptors at `241c783`, and they are exactly the four named in CHE-239 §A.0 — `SO_RAY_LAUNCH_TRACE`, `O_RAY_TRACE`, `SOM_SPOT_DIAGRAM`, `SOM_PSF`. (One of them, `SOM_PSF`, postdates the stale count, so "the four are unchanged" is true relative to CHE-239's table, not relative to the 15.) |

**Chosen: option 1 (split by branch), implemented read-only.** A and B run here on
`che-152-greenfield-rewrite` and are recorded against `241c783`. C and D need the
`che-140` tree, which is reachable without switching branches — `git worktree` at a
detached `origin/che-140…` in a scratch path, never a checkout of this working
tree, never a new branch. All *committed* artifacts land on the current branch.
Each workstream records its own SHA.

Consequence, stated rather than implied: the report covers two trees, so
cross-workstream comparison between A/B and C/D is limited to the level of
"both were run", not "both measured the same code".

Option 2 was rejected because it discards the descriptor vocabulary that is the
entire point of workstream A. Option 3 was rejected because the owner's
instruction names CHE-241 and CHE-242 explicitly.

### 2.2 Missing input, carried forward

`optiland_notebook_link_index.csv` — the SHA-pinned index of all 190 notebooks
referenced by CHE-239 — is **not attached to the issue and not in either tree**.
Workstream A's Tier-2 scope is therefore taken from the 25-row in-scope table
written into CHE-239 itself, which is self-contained and does not need the index.
The index is only required to *extend* coverage beyond tonight's 25 + 11, which is
already a non-goal. Recorded as an open input, not a blocker.

---

## 3. Environment

Recorded before any compute. All project execution goes through `./run.sh` into
the `agent_solver` container; nothing in this report was run with host `python`,
`pytest`, or `pip`.

| | |
| --- | --- |
| repository | `/home/chengjz/multiscale_optics_agent` |
| branch | `chengjiazhou4802/che-152-greenfield-rewrite` |
| commit (A, B) | `241c783996e399c420afe27374a0aa33cf42e690` |
| working tree at preflight | clean (`git status --porcelain` empty) |
| container image | `agent_solver:latest`, id `82487dd7da27`, built 2026-08-25 |
| OS / kernel | Linux 6.8.0-84-generic, x86_64, glibc 2.41 |
| Python | 3.12.14 |

Installed packages, from `importlib.metadata` inside the container:

| package | version |
| --- | --- |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| jax / jaxlib | 0.6.2 / 0.6.2 |
| chromatix | 0.6.0 |
| **optiland** | **0.6.0** |
| torch | **2.13.0+cpu** |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 |

`jax.devices()` inside the **CPU** image → `[CpuDevice(id=0)]`.
`torch.cuda.is_available()` → `False`; `torch.version.cuda` → `None`.

**Corrected during workstream C.** Those two lines describe the *CPU* image, which
is what `./run.sh` uses by default, and §3.2 below originally generalized them into
"the night's declared workloads are CPU-only". The separately-built
`agent_solver_gpu` image exists (`b626c10c5dca`, 9.61 GB) and carries
**`torch 2.13.0+cu126` with `torch.cuda.is_available() → True`** and a CUDA jaxlib
(`jax.devices() → [CudaDevice(id=0)]`). Workstreams A and B genuinely needed no
device; **workstream C ran on GPU 6**. The original sentence was true of the image
it was measured in and false as a statement about the night.

### 3.1 Upstream pin vs installed version

The parent pins upstream Optiland at `00c0837fbee5d66019a24a1735ff91cd4f9b2646`
(`master`, 2026-09-01). The **installed** package is `optiland 0.6.0`, a release
artifact, not that commit. Both are recorded, as the ticket requires. Everything
measured in workstream A is measured against **the installed 0.6.0**, and any
notebook/API divergence from `master` is classified as an *environment finding*,
not a physics failure.

### 3.2 Resource preflight

Taken immediately before compute, per the shared-GPU policy.

```
Mem:   377Gi total, 27Gi used, 253Gi free, 347Gi available
Swap:  1.8Ti total, 0B used            <- baseline; any growth is a stop condition
GPU 0-7: NVIDIA RTX A6000, 49140 MiB each, 2 MiB used, 0% util   (all idle)
```

**Workstreams A and B are CPU-only** and needed no device: none of the seven wave
kernel checks or the three ray observables uses one. **Workstream C ran on GPU 6**
through `MOA_GPUS=device=6 ./run.sh --gpu`, which is the `agent_solver_gpu` image
described in the correction above — the Demo2 probes are GPU workloads by
construction and their committed baselines were produced on one. Workstream D is
not yet run and this section makes no claim about it.

One GPU, never two, and never concurrently with another job. Swap was re-checked
after every GPU run and stayed at **0 B**; GPU 6 returned to 2 MiB after each.

---

## 4. Record and report contract

* **Where a record goes, and why it is not committed JSON.**
  `tests/unit/test_suite_shape.py::test_the_only_committed_records_are_the_live_benchmark_outputs`
  is a landed gate in the default suite: every committed `*.json` outside
  `.claude/`, `outputs/`, `runs/` and `tmp_probes/` must live under
  `benchmarks/systems/records/` or `knowledge/capabilities/`. A verification
  record from this run belongs to neither tree, and this ticket may not relax a
  gate to make room for its own artifacts. So:

  * the **raw** per-case records go to `outputs/che-238-overnight/<workstream>/`,
    which is `.gitignore`d — reproducible, full provenance, not committed;
  * the **committed** evidence is this report, whose tables carry the same
    numbers, and the harness scripts under `benchmarks/verification/`, which
    regenerate the raw records.

  This is a deliberate trade: the gate says a committed record must have a
  declared generator in one of two trees, and honouring it costs a level of
  indirection rather than a widened rule.
* Every record carries provenance: git SHA, branch, UTC timestamp, command,
  device, package versions (including `optiland`), dtype, the numerical
  parameters, seed where one exists, runtime, and status — **for the records this
  run writes itself**. Workstreams C and D reuse the `che-140` probes' own record
  writer, whose schema predates this contract and omits the branch, the command
  and the status, and whose `commit` field returned `"unknown"` under a detached
  worktree. Where that happens the report attests those fields instead and says so
  (§7.1).
* Records are **new files**; no historical record is overwritten. Where a
  workstream reads a `che-140` record for comparison it reads it out of that
  tree and writes its own result here.
* Status vocabulary is the ticket's: `PASS`, `PASS-refused`, `FAIL`, `BLOCKED`,
  `NOT-COVERED`, extended for C/D by CHE-241/242 §4.4 with `PASS-native`,
  `PASS-graph-only`, `PASS-transcribed`, `BLOCKED-no-backend`,
  `BLOCKED-untranscribable`.
* No tolerance is widened anywhere in this run. A number that does not reproduce
  is recorded as a discrepancy and classified.
* Nothing is promoted from characterization to validation.

---

## 5. Workstream A — ray tutorial / system regression (CHE-239)

Ran at `241c783` on CPU/float64 through `./run.sh`. Two drivers, both committed:

```
./run.sh python -m benchmarks.verification.ray_tier1
./run.sh python -m benchmarks.verification.ray_tier2 --notebooks outputs/che-238-overnight/notebooks
```

Tier 1 takes 14 s; Tier 2 takes 7 m 45 s and peaks at 1.0 GiB RSS. Raw records:
`outputs/che-238-overnight/workstream-a/{tier1,tier2}.json`. The 36 notebooks were
fetched from `raw.githubusercontent.com` at the pinned commit
`00c0837fbee5d66019a24a1735ff91cd4f9b2646`; all 36 returned HTTP 200.

### 5.0 Aggregate

| | PASS | BASELINE | PASS-refused | FAIL | BLOCKED | NOT-COVERED | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier 1 (11 notebooks) | 23 | 0 | 1 | **12** | 2 | 0 | 38 |
| Tier 2 (25 tutorials) | 35 | 26 | 0 | **0** | 12 | 21 | 94 |

`BASELINE` is a status this harness added, and the reason is in
`benchmarks/verification/record.py`: an `SOM_SPOT_DIAGRAM` or `SOM_PSF` row on a
tutorial has **no upstream golden** — the tutorial prints nothing those
descriptors return — so it carries no expected value and no delta. Counting those
26 rows as `PASS` would have inflated the agreement count by 75 % with rows that
compared nothing. `Row.__post_init__` now refuses a `PASS` or `FAIL` that has
neither an expected value nor a delta.

**All 12 Tier-1 failures have one cause**, in §5.3. There are no unexplained
failures in either tier.

### 5.1 What a Tier-1 comparison is evidence of

Both sides of every Tier-1 diff run the same Optiland code: `SOM_SPOT_DIAGRAM`
and `SOM_PSF` construct a lens from `problems.OpticalSetup` and delegate. A
Tier-1 diff is therefore a **plumbing regression** — the setup extracted from the
notebook's lens rebuilds that lens, the field and wavelength reach the right
arguments, the unit conversion out is right — and it is not validation of the
physics. `AGENTS.md`'s rule that repository code may not be its own correctness
oracle is why. A zero diff here is silent about whether Optiland's spot is the
right spot.

### 5.2 Tier 1, case by case

| # | Notebook | Descriptor | Result |
| --- | --- | --- | --- |
| 1 | `rms_spot_size_vs_field` (TessarLens) | SOM_SPOT_DIAGRAM | 3 PASS / 6 FAIL — §5.3 |
| 2 | `spot` (CookeTriplet) | SOM_SPOT_DIAGRAM | 1 PASS-refused, 3 PASS / 6 FAIL — §5.3 |
| 3 | `through_focus_spot_diagram` (CookeTriplet) | SOM_SPOT_DIAGRAM | **11 PASS**, all exact |
| 4 | `cylindrical_lens` | SOM_SPOT_DIAGRAM | **BLOCKED** — `ToroidalGeometry` |
| 5 | `nurbs_parabolic_mirror` | SOM_SPOT_DIAGRAM | **BLOCKED** — reflective (and NURBS) |
| 6 | `fft_psf_2d` (CookeTriplet) | SOM_PSF `fft` | **PASS** — grid relative L2 **0.0** |
| 7 | `fft_psf_3d` (TessarLens, 20.5°) | SOM_PSF `fft` | **PASS** — grid relative L2 **0.0** |
| 8 | `huygens_psf_2d` (DoubleGauss) | SOM_PSF `huygens` | **PASS** — grid relative L2 **0.0** |
| 9 | `huygens_psf_3d` (DoubleGauss) | SOM_PSF `huygens` | **PASS** — grid relative L2 **0.0** |
| 10 | `mmdft_psf_2d` (CookeTriplet) | SOM_PSF `mmdft` | **PASS** — grid relative L2 **0.0** |
| 11 | `mmdft_psf_3d` (DoubleGauss) | SOM_PSF `mmdft` | **PASS** — grid relative L2 **0.0** |

All six PSF cases agree with the notebook's own PSF class **bit-exactly on the
full intensity grid**, not only at the peak: max absolute difference 0.0, relative
L2 0.0, peak index identical. Peak values, in Optiland's Strehl-percent
normalization: fft 30.576757 / 97.236385, huygens 7.332453, mmdft 30.797721 /
7.348339.

Three things measured on the way, none of them defects:

* **`fft` and `mmdft` both reduce the requested pupil sampling.** `fft` with
  `num_rays=128` used **64**; `mmdft` with `num_rays=512` used **128**. Both sides
  report the same reduced count, so nothing disagrees — but a caller reading
  `num_rays` off the record is reading what ran, not what it asked for, which is
  what `NativePsfAnalysis.num_rays` documents.
* **`huygens_psf_2d` and `huygens_psf_3d` are the same computation.** Both
  notebooks are DoubleGauss at `field=(0, 0.0)`, λ = 0.55 µm, and differ only in
  the `projection=` argument to `.view()`. They produce bit-identical maps
  (peak 7.332453 both). One of the eleven Tier-1 cases adds no numerical coverage;
  it is kept because the ticket enumerates eleven.
* **`nurbs_parabolic_mirror` cannot be built as the notebook writes it.** The
  pinned-`master` cell calls
  `Surface(geometry=…, material_pre=…, material_post=…, is_stop=…)`; installed
  optiland 0.6.0 takes `Surface(previous_surface, material_post, geometry, …)` and
  has **no `material_pre`**, so the cell raises `TypeError`. Built with the
  installed signature instead. Pin-vs-release drift, classified per §3.1 as an
  environment finding, and carried on the row.

`through_focus_spot_diagram` deserves its own note because it is the one Tier-1
case that is *not* a tautology. The native side moves
`image_surface.geometry.cs.z`, which is what `ThroughFocusSpotDiagram` itself
does; ours lengthens the last surface's `thickness_mm`, which is where
`OpticalSetup` puts the image plane. Two different mechanisms, five focus steps at
`nominal + (i − 2) × 0.1 mm`, RMS radius varying monotonically 1.6e-5 → 4.3e-6 →
1.6e-5 m across them, and exact agreement at every step.

### 5.3 Confirmed defect — `SOM_SPOT_DIAGRAM` analyses at the wrong wavelength

**This is the run's principal finding.** It is confirmed, minimally reproduced,
and deliberately **not fixed** in this ticket.

`backends/optiland/analysis.py::spot_diagram` builds its lens with `build_lens`,
which declares **exactly one** wavelength — `setup.reference_wavelength_um` —
and then asks `optiland.analysis.SpotDiagram` for `wavelengths="all"`. "All" is
therefore the *setup's reference* wavelength. But the returned
`NativeSpotAnalysis` sets `wavelength_m = source.wavelength_um * 1e-6`.

So **the source wavelength does not reach the analysis at all**, and the record
labels the reference wavelength's answer with the source's.

Minimal reproduction — CookeTriplet, on axis, `num_rings=6`, RMS spot radius:

| | value |
| --- | --- |
| native `SpotDiagram(…, wavelengths=[0.48])` | 3.7913354614484123e-3 mm |
| native `SpotDiagram(…, wavelengths=[0.55])` | 4.2936895642576465e-3 mm |
| `spot_diagram(setup(ref=0.55), SourceSpec(0.48))` | **4.293689564257647e-3 mm**, record says `wavelength_m = 4.8e-07` |
| `spot_diagram(setup(ref=0.48), SourceSpec(0.48))` | 3.7913354614484123e-3 mm |
| `spot_diagram(setup(ref=0.48), SourceSpec(0.55))` | **3.7913354614484123e-3 mm**, record says `wavelength_m = 5.5e-07` |

Every Tier-1 spot row carries a `our_numbers_match` attribution computed from a
second native run at the reference wavelength. All 12 failures report
`setup_reference_wavelength`; all 3 passes report `indistinguishable` because
source and reference coincide there. Worst relative deviation observed: **6.31 on
a centroid** and 0.38 on a geometric radius (CookeTriplet, 20°, 0.65 µm).

Three things this is **not**:

* Not a `psf` defect. `psf` passes `source.wavelength_um` to the PSF class
  explicitly, and the four Tier-1 PSF rows whose source (0.55 µm) differs from
  the sample's primary (0.5876 µm) agree with a native call at 0.55 µm to 0.0.
* Not contaminating any committed record. No benchmark under
  `benchmarks/systems/` calls `spot_diagram`.
* Not visible to the current suite. `tests/backends/test_optiland_analysis.py`
  exercises setups whose reference wavelength equals the source's, which is
  exactly the `indistinguishable` case.

It is also a **record-contract** violation and not only a wrong number:
`NativeSpotAnalysis` is a provenance-bearing artifact whose `wavelength_m` field
is fabricated relative to the numbers beside it, and `spot_diagram`'s own
docstring — *"what is analysed is `source.wavelength_um`"* — is now known false.

**Not fixed here**, per CHE-238's code-change policy: the default expectation is
no production change, and this is a behaviour change to a solver adapter that
moves frozen numbers, which `AGENTS.md` puts behind independent review and its own
ticket. Recommended follow-up in §10.

### 5.4 Tier 2 — expressibility and system regression

Per tutorial: harvest the lens by executing the notebook's prologue, extract a
setup, rebuild it with the same `build_lens` a trace uses, compare the surface
table and the paraxial characterization, then run `SO_RAY_LAUNCH_TRACE` against
`Optic.trace` on the tutorial's *own* lens at a fixed hexapolar sampling
(`num_rings=4`, 61 rays), on axis and at the widest declared field.

**13 of 25 expressible, and every one of the 13 round-trips and traces exactly.**
Surface table (position, radius, conic, index after, stop, semi-diameter) and
paraxial (EFL, EPD, F-number) deltas are 0.0; traced intersection coordinates and
direction cosines agree to 0.0; the surviving ray count matches; the on-axis
optical-path shape agrees to ~1e-16.

| # | Tut | Setup | Trace | Spot | PSF | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1a | PASS | PASS | BASELINE | BASELINE | on-axis field only, so one trace row |
| 2 | 1b | PASS | PASS ×2 | BASELINE | BASELINE | CookeTriplet |
| 3 | 1c | **BLOCKED** | — | — | — | upstream-api-drift |
| 4 | 1e | **BLOCKED** | — | — | — | upstream-api-drift |
| 5 | 2a | PASS | PASS ×2 | BASELINE | BASELINE | ReverseTelephoto, 13 surfaces |
| 6 | 2b | PASS | PASS | BASELINE | BASELINE | |
| 7 | 2c | PASS | PASS ×2 | BASELINE | BASELINE | 3 lenses harvested, richest chosen |
| 8 | 2d | PASS | PASS ×2 | BASELINE | BASELINE | EyepieceErfle, 9 surfaces |
| 9 | 3a | PASS | PASS ×2 | BASELINE | BASELINE | |
| 10 | 3b | PASS | PASS ×2 | BASELINE | BASELINE | |
| 11 | 3c | **BLOCKED** | — | — | — | `PolynomialGeometry` |
| 12 | 3d | PASS | PASS ×2 | BASELINE | BASELINE | |
| 13 | 3e | PASS | PASS ×2 | BASELINE | BASELINE | 11 surfaces; the one `imageFNO` case |
| 14 | 4a | **BLOCKED** | — | — | — | reflective |
| 15 | 4b | PASS | PASS | BASELINE | BASELINE | |
| 16 | 4e | **BLOCKED** | — | — | — | finite object (110.85883544 mm) |
| 17 | 4f | **BLOCKED** | — | — | — | reflective |
| 18 | 6a | PASS | PASS ×2 | BASELINE | BASELINE | |
| 19 | 6b | **BLOCKED** | — | — | — | tilt/decenter — see below |
| 20 | 8a | **BLOCKED** | — | — | — | custom `NewGeometry` |
| 21 | 8c | PASS | PASS | BASELINE | BASELINE | all-zero asphere — see below |
| 22 | 9a | **BLOCKED** | — | — | — | environment-dependency (`tqdm`) |
| 23 | 9b | **BLOCKED** | — | — | — | environment-dependency (`sklearn`) |
| 24 | 9d | **BLOCKED** | — | — | — | environment-dependency (`gymnasium`) |
| 25 | 9f | **BLOCKED** | — | — | — | environment-dependency (`sklearn`) |

21 `NOT-COVERED` rows record, per tutorial, *what* the printed output needs that
the catalog does not expose — `optimization`, `opd/zernike`, `mtf`,
`paraxial-report`, `tolerancing`, `geom-analysis` — taken from CHE-239 §A.5's own
column. A coverage gap is not a failure of the optical system, and none is
counted as one.

### 5.5 The twelve blocks, classified

Four kinds, and only one of them is about this project's schema.

**Schema refusals (6)** — the real answer to "can `OpticalSetup` express this":

| Tutorial | Category | What the schema lacks |
| --- | --- | --- |
| 3c | `geometry` | `PolynomialGeometry` — no freeform sag field |
| 8a | `geometry` | the tutorial's own `NewGeometry` custom surface |
| 4a, 4f | `reflective` | no reflective flag; rebuilding would turn a mirror into a lens |
| 4e | `finite_object` | the object plane is 110.86 mm away |
| 6b | `tilt_decenter` | surfaces decentred by ~0.15 mm and tilted by ~5.6 mrad |

Two of these have a footnote worth having:

* **4e** was expected by CHE-239 §A.5 to be `PASS-refused` — trace-only, with the
  two analyses raising `NotImplementedError` on the finite-conjugate source. It
  refuses *earlier* than that, at extraction, and for two independent reasons: a
  finite object distance and an `ObjectHeightField`. The finite-conjugate path
  does exist (`SourceSpec.object_distance_mm` and `solver:trace` both support it);
  what is missing is a verified object-height → field-angle conversion, and
  CHE-239 forbids inventing one during this run. Follow-up in §10.
* **6b** refuses on tilt because the lens harvested is the one the tutorial's
  Monte-Carlo tolerancing has already perturbed. That is arguably the right lens —
  a perturbed system is what this tutorial is about — but it is not the nominal
  Cooke triplet, and the row's `cells_executed` says so.

**Upstream pin-vs-release drift (2)** — CHE-238 §3.1 environment findings, not
physics:

| Tutorial | Missing from installed 0.6.0 |
| --- | --- |
| 1c | `optiland.materials.MaterialCatalog` |
| 1e | the whole `optiland.diagnostics` module |

In both, the failing import is in an early cell, so every name it defined is
missing and the rest of the prologue cascades into `NameError`. The harness
therefore reports that **whether these tutorials build a lens at module level was
not determined** — it does not claim a structural cause it did not establish.

**Container dependency gaps (4)** — `tqdm` (9a), `sklearn` (9b, 9f), `gymnasium`
(9d). Nothing about this project or the schema; the `agent_solver` image does not
ship them. Note that 9b's system is `CookeTriplet`, which Tier 1 regresses in
full, so the *system* is covered even though the tutorial is not.

**Nothing else.** No Tier-2 block is a harness bug.

### 5.6 Judgements the harness makes, and why each is not a hole

Four places where this harness decided that a difference does not gate. Each is
backed by a measurement rather than by a preference.

1. **Off-axis optical path is not gated.** `declare_optical_path_m` adds an
   object-space term `n_object · (d₀ · r_launch)` that `RealRays.opd` does not
   carry; off axis that is a *tilt*, and that module's own docstring says the
   omission "IS the convergence tilt". So the trace regression runs on axis, where
   the term is a constant that is removed exactly and the residual is ~1e-16, and
   off axis records the difference under a key ending `_EXPECTED` that does not
   decide the status. Measured on 1b at 20°: 2.1e-2 of the optical-path extent,
   against 0.0 on axis for the same system.
2. **A geometry class-name mismatch does not gate when the aspheric terms
   agree.** Tutorial 8c builds an `EvenAsphere` with an all-zero coefficient list;
   `build_lens` selects `standard`. `SurfaceSpec.has_aspheric_terms` records the
   measurement that the two agree *bitwise* in sag, traced position and
   accumulated path. Independently corroborated here: 8c's own
   `SO_RAY_LAUNCH_TRACE` row passes at 2.6e-16. The row still carries the
   mismatch under `geometry_class_only_all_zero_asphere`.
3. **The clip/survival check is a count, not a mask.** `rays.to_ray_bundle`
   filters clipped rays out of the returned bundle, so an element-wise mask
   comparison would be trivially all-True. What is compared is that the bundle
   holds exactly the rays the native trace says survived. **Not exercised**: every
   harvested tutorial reports `surface_apertures_declared: 0`, so no system in
   Tier 2 clips at all. Recorded as a gap rather than as a passing check.
4. **The `imageFNO → paraxial.EPD()` conversion is not checked by a paraxial
   residual.** Read from the pinned 0.6.0: `ImageFNOAperture.compute_epd` is
   `f2()/FNO`, `EPDAperture.compute_epd` is the stored value, and
   `Paraxial.FNO()` returns the declared F-number directly for an `imageFNO` lens
   and `f2()/EPD()` otherwise — so the EPD delta is identically zero *by
   construction* and the FNO delta reduces to the EFL residual reported beside
   it. What actually checks the conversion is the ray regression: a different
   pupil would move the traced intersection coordinates, and they agree to 0.0.
   Only tutorial 3e exercises the conversion in Tier 2; 18 of the 29 canonical
   samples use `imageFNO`.

### 5.7 Coverage this tier did not reach

* **The 29 canonical `optiland.samples` prescriptions were surveyed** and 25 of 29
  extract cleanly. The four that do not: `HubbleTelescope` and
  `UVReflectingMicroscope` (reflective), `NavarroWideAngleEye`
  (`float_by_stop_size` aperture), `UVProjectionLens` (`ObjectHeightField`; it
  also declares `objectNA` and a finite object, and the field check refuses
  first). Only three geometry classes appear across all 29 — `Plane`,
  `StandardGeometry`, `EvenAsphere` — and none is decentred or tilted.
* **No surface aperture is exercised anywhere in workstream A**, so the clipping
  path is untested by it. See §5.6 item 3.
* **The 139 remaining gallery notebooks are not scheduled**, per the ticket.
* **`optiland_notebook_link_index.csv` was never available** (§2.2); Tier-2 scope
  came from CHE-239's own 25-row table.
* Tier 1's PSF rows do not compare `pixel_pitch_m`, so the µm/mm translation trap
  in `_native_sample_pitch_um` is not covered here. It is pinned by
  `tests/backends/test_optiland_psf.py`.
* Three tutorials ran a long single cell inside the harvest (3b 152 s, 4e 157 s,
  4f 31 s). The 20 s budget is checked *between* cells, so one long cell overruns
  it; 3b and 3d additionally stopped part-way through their prologue
  (`cells_executed` 6/11 and 14/23).

## 6. Workstream B — wave kernel sweep (CHE-240)

Ran at `241c783` on CPU/complex64 through `./run.sh`. One driver:

```
./run.sh python -m benchmarks.verification.wave_kernel_sweep
```

11 s, 968 MiB peak RSS. Record: `outputs/che-238-overnight/workstream-b/kernel_sweep.json`.

### 6.0 Aggregate

24 rows: **17 PASS, 5 FAIL, 2 NOT-COVERED.** All seven checks have numerical
records. No tolerance was widened and no propagator was touched.

| Sweep | Recorded expectation | Measured | Status |
| --- | --- | --- | --- |
| 1 kernel identity | max difference **exactly 0.0** | **6.108e-05** on the arrays | **FAIL** — stale record |
| 2 hard edge | ≈ 2.3e-1 of peak, pad-independent | 3.20e-1, flat to 8.8e-4 over 4 paddings | PASS |
| 3 soft edge | ≈ 4.9e-6 of peak | 6.24e-7 (circular), 1.75e-4 (square) | **FAIL** — setup |
| 4 sampling bound | refusal at `z = N·pitch²/λ` | **no runtime refusal on either side** | **NOT-COVERED** |
| 5 paraxial bound | π/4, 25.5 rad, 175 rad | π/4 exactly, 25.4640, 174.533 | PASS ×3 |
| 6 tilted beam | `z·sinθ` not `z·tanθ`; delay linear in `f` | 6.2e-7…6.8e-5 samples; ratio flat to 1.1e-6 | PASS ×4 |
| 7 focal plane | focus `f·sinθ`; grid `λf/(nNdx)` | pitch exact; focus to 1.5e-7 relative | PASS ×2 |

"101's grid" is fully specified by `VALIDITY_NOTES['paraxial']` and was not
re-chosen: **512², dx = 0.3 µm, λ₀ = 0.532 µm, n = 1.33, z = 50 µm**.

### 6.1 Which oracle decided what

Three standings, not equal:

* **Closed form** — the paraxial residual `n k₀ z (1 − cos − sin²/2)`, the walk-off
  `z sinθ`, the focus at `f sinθ`, the Fourier pitch `λf/(nNdx)`. Arithmetic,
  independent of this repository. Sweeps 5, 6, 7 are decided by these.
* **Kernel identity** — sweep 1 is an algebraic claim about two of this project's
  own kernels. It says nothing about either being right.
* **Diagnostic** — sweeps 2, 3, 4 compare Fresnel against this repository's own
  angular spectrum, which `AGENTS.md` forbids as a correctness gate. They
  reproduce a *recorded measurement*; they do not certify a kernel.

Every oracle grid is built from the declared pitch and shape in numpy. Sweep 1 is
the one place that reaches for `native.f_grid` and
`compute_transfer_propagator`, because its claim is about the propagator arrays
and cannot be made without them.

### 6.2 Second confirmed finding — "exactly 0.0" is overstated

`backends/chromatix/solver.py`'s module docstring, repeated in
`operations/catalog.py`'s `O_FRESNEL_PROPAGATE` record, says that replacing
`delay + 1.0` with `2.0` in `_carrier_removed_propagator` *"reproduces the Fresnel
phase with a maximum difference of **exactly 0.0** in float32 over a 512² grid"*.

Measured on the propagator arrays themselves — the substituted kernel against
chromatix's own `compute_transfer_propagator`, same grid, same float32:

| quantity | value |
| --- | --- |
| max absolute difference, arrays | **6.108e-05** |
| max phase difference, arrays | 6.108e-05 rad |
| per-bin phase difference through the public ops | 3.050e-05 rad |
| the same, unpatched | 2.127 rad |
| collapse factor from the substitution | **69 727×** |
| predicted float32 phase floor (4·ε₃₂·198 rad) | 9.450e-05 rad |

The two kernels are **not the same expression**: chromatix computes
`-π·(λ/n)·z·l2_sq_norm(f_grid)` where the substituted kernel computes
`-2π|z|·(λ/n)·f²/2.0` — a different multiply order in float32.

So the *substance* of the claim holds: the substitution is algebraically exact,
the difference collapses by nearly five orders of magnitude, and 6.1e-5 sits below
the predicted float32 floor. What does not hold is "exactly 0.0", which should
read "to float32 round-off, ~6e-5 rad".

**Not fixed here.** CHE-240's non-goals forbid converting a stale recorded
expectation into a new baseline without a separate ticket. Recommended follow-up
in §10.

This was caught by the independent review, not by the first version of this sweep.
The first version compared propagated *fields* rather than the propagator arrays,
found 8.4e-5, attributed it to FFT round-off, and filed the row as PASS with a
`setup` classification whose stated reason — that the two kernels are the same
expression and therefore bit-identical — was false. Recorded here because it is
the failure mode this whole run exists to avoid: measuring an adjacent quantity
and reporting a clean result.

### 6.3 Sweeps 2 and 3 — the two recorded intensity figures

The hard edge reproduces: **3.20e-1** against a recorded 2.3e-1, flat across
paddings 0/128/256/512 to 8.8e-4 relative, which is the second half of the
recorded claim (pad-independence is what says the difference is the kernel and not
wraparound).

The soft edge does not: **6.24e-7** against a recorded 4.9e-6, a factor of 8.

Both figures are held to a factor-of-two band, and the basis for that band is
**measured setup underspecification**, not significant figures. Two sensitivity
sweeps establish it:

| what the record does not state | measured span on 101's grid | contains 4.9e-6 / 2.3e-1? |
| --- | --- | --- |
| aperture size (hard edge, 4 widths) | 1.39e-1 … 2.64e-1 | **yes** |
| edge profile, square family (5 powers) | 1.75e-4 … 1.16e-3 | **no** |
| edge profile, circular family (5 powers) | 4.77e-7 … 5.61e-4 | **yes** |

Two conclusions, both attributable:

* The hard edge's +39% is inside what the unstated aperture size alone spans, so
  the recorded 2.3e-1 was never reproducible to better than a factor.
* **The record's soft field is radially symmetric, not a softened square.** Only
  the circular family's span contains 4.9e-6; softening the hard case's own square
  aperture lands 36× away and never reaches it. The aperture *geometry* is a
  larger term than the edge softness, which is not what the record's wording
  ("a soft-edged field on the identical grid") suggests.

Classified **setup**. Not a stale record and not an implementation change — the
figure is inside the space the record's own unstated parameters span.

The soft-edge pad-independence row passes on a scale-aware criterion: at 6e-7 of
peak the difference is a handful of complex64 epsilons, so its 18% pad-to-pad
variation is arithmetic rather than a boundary effect, and asserting independence
there would be asserting something about round-off. The hard edge, at 3.2e-1, is
where the question is meaningful and it is flat.

### 6.4 Sweep 4 — a declared bound with no enforcement

The ticket expects a refusal at `z = N·pitch²/λ`. **The bound is declared and
nothing enforces it.**

`operations/catalog.py` carries `"z <= N pitch^2 / lambda, the transfer function's
own sampling bound"` as a `validity` entry on **both** `O_ASM_PROPAGATE` and
`O_FRESNEL_PROPAGATE`. No runtime refusal exists: searched `MODELS`,
`_require_model`/`_require_pad_target_crop`, `ScalarField.__post_init__`, the 22
`CONTRACT_CODES` and `numerics.REFUSAL_CODES` — none mentions a
transfer-function sampling bound.

Both records preserved, as the ticket requires:

| | `z` | `z / bound` | refused? | unpadded-vs-padded drift |
| --- | --- | --- | --- | --- |
| inside | 50 µm | 0.577 | **no** | 8.32e-7 |
| outside | 200 µm | 2.309 | **no** | 9.54e-7 |

The bound is 86.6 µm on this grid. "Declared domain, unenforced at runtime" is a
materially different gap from "absent", and that is what the rows say — the first
version of this sweep claimed the bound did not exist in the tree at all, which
the independent review corrected.

Note the declared criterion omits the medium index: in-medium it would be
`n·N·dx²/λ₀`, 115 µm rather than 86.6 µm. Reproduced as the tree words it rather
than corrected here; both probed distances fall on the same side either way.

The drift figure quantifies the gap and is **diagnostic** — it compares two runs of
the same implementation. It is also small on both sides, which is its own
information: at these distances the unenforced bound is not where the damage is,
and a sweep that only looked at the drift would have concluded there is no gap.

### 6.5 Sweep 5 — the paraxial bound, and an asymmetry in the record

All three pass. Wrapped residuals against the closed form at the cosine actually
probed: **3.8e-6, 2.6e-5, 1.3e-4 rad** — at phase arguments up to 169 rad.

| record's cosine | record | full residual `n k₀ z(1−cos−sin²/2)` | leading term `n k₀ z sin⁴/8` |
| --- | --- | --- | --- |
| 0.29907 (the bound) | π/4 | 0.82262 (+4.7%) | **0.7853981634 = π/4, exact** |
| 0.66667 (per-axis Nyquist) | 25.5 | **25.4640 (0.14%)** | 19.393 |
| 0.94281 (grid corner) | 175 | **174.533 (0.27%)** | 77.570 |

**The record quotes the leading term at the bound and the full residual at the
other two.** That is not an error: `VALIDITY_NOTES` says literally "at which the
leading error reaches pi/4", and at the bound the leading term is π/4 *exactly* by
construction — `n k₀ z·sin⁴/8` at `sin = (λ₀/(nz))^¼` is `n k₀ z·(λ₀/(nz))/8 =
2π/8`. Measured agreement 3.3e-16 relative. Reporting only the full residual would
have made a correctly-stated record look 5% wrong.

Two measurement points worth recording:

* **The 0.943 case had to be a 2-D probe.** 0.943 is √2 times the per-axis
  Nyquist, so it exists only as the *corner* of the 2-D frequency grid
  (`fx = fy = Nyquist`). The first version of this sweep built a single-axis ramp at
  that frequency — above Nyquist, therefore aliased — and reported a 2.41 rad
  residual against the closed form, which was the alias and not a disagreement.
  Every case is now parameterized by DFT **bin pair** and the cosine computed from
  the bins.
* Bin `N//2 − 1` rather than `N//2`, because the Nyquist bin is the degenerate ±
  bin where a ramp advances by exactly π per sample. So the cosines probed sit just
  below the record's, and every row carries both.

### 6.6 Sweeps 6 and 7 — the closed forms

**Tilted beam.** Landing point within 6.2e-7 / 4.1e-6 / 6.8e-5 samples of
`z sinθ` at 2°, 5°, 10°. `z tanθ` is rejected only where the two separate: they
are 0.01 samples apart at 2° and 1.07 at 10°, so each row carries
`distinguishes_the_two_oracles` and the discrimination is claimed only at 10°.
Group-delay linearity: `landing / sinθ` constant to 1.1e-6 across the three
angles.

**Focal-plane transform.** Output pitch exactly `λf/(nNdx)` per axis on a grid
asymmetric in *both* count (48×64) and pitch (0.30/0.25 µm), so a transposed
`(y, x)` could not pass — relative error 0.0. Focus at `f sinθ` to 1.5e-7
relative at 20°, both signs, with `f tanθ` rejected by more than 2 output samples
and the focus staying on the y axis (which is what a transposition would break and
is invisible in a rotationally symmetric case).

### 6.7 Not covered by this workstream

* The sampling-bound refusal (§6.4) — declared, unenforced, untested.
* No Chromatix gallery notebook was run; the ticket excludes it.
* Nothing here touches GPU, torch, or float64 propagation.
* The `pad_width`-dependent single-FFT Fresnel method (`transform_propagate`) is
  not in the tree and was not probed.

## 7. Workstream C — Demo2 reproduction (CHE-241)

**Status: `PASS-native`.** Option B (graph + PyTorch transcription) was not needed
and, per the ticket, was therefore not run as a substitute.

Ran against the `che-140` tree at `eb3d792` through a **detached `git worktree`**
(CHE-238 §2.1) — the working branch never changed and no branch was created. All
four probe records are copied to
`outputs/che-238-overnight/workstream-c/che238_demo2_*.json`, byte-identical to
the worktree originals.

**These four records use the `che-140` probe's own schema, not §4's contract.**
They carry `environment.commit: "unknown"`, `record_provenance.source_commit:
null` and `working_tree_dirty: true`, and no branch, command or status field —
because a detached worktree has no branch and the probe's provenance code reads
`git` in a way that returned nothing here. So the SHA `eb3d792`, the branch, the
commands and the `PASS-native` status are **this report's** attestation rather than
the records' own. What the records do carry is a code fingerprint over 34 source
files, the environment fingerprint, the full parameter set and every metric; the
substance is intact and the provenance fields are not. §4's blanket "every record
carries provenance" is false for these four and is corrected there.

### 7.1 Capability preflight (§2), and the decision

The 5-minute timebox was not needed; the preflight took under two minutes.

| # | Question | Answer |
| --- | --- | --- |
| 1 | `torch.__version__`, build, CUDA, devices | CPU image: `2.13.0+cpu`, `cuda_is_available False`. **GPU image: `2.13.0+cu126`, `True`**, `jax.devices() → [CudaDevice(id=0)]` |
| 2 | Do the descriptors declare a torch-capable backend, and can backend selection be pointed at torch without a code change? | **The question does not arise.** `demo2_hologram.py`'s own `--backend` choices are `numpy` and `jax`. Torch is not an option on this path and never was; the probe reaches jax through `_demo_support.enable_x64_if_needed` and numpy otherwise. |
| 3 | Cheapest native smoke | `--preset smoke --backend numpy`: **succeeded in 3.6 s.** RW-F NCC 1.000000, complex rel-L2 7.34e-13 (7.11e-13 phase-aligned); RW-P NCC 0.844177 |

**Decision: Option A (native).** The smoke succeeded, so §2 forbids running Option
B at all. And the amendment's premise — "it is not yet established that the
probe/solver path in this branch has a working PyTorch backend" — resolves in a
way the amendment did not anticipate: the probe path does not *want* a torch
backend. It offers numpy and jax, jax has a working CUDA build in the GPU image,
and that is the path the committed baselines were produced on.

So there is no graph serialization, no transcription, no parameter parity table
and no `BLOCKED-no-backend` in this workstream. Nothing under
`benchmarks/probes/records/ray_wave/transcription/` was created.

### 7.2 What was run

Four runs, all with `--output-name che238_*` so no historical record was touched.

| run | device | routes | wall clock |
| --- | --- | --- | --- |
| `--preset smoke --backend numpy` | CPU | rw_f, rw_p | 3.6 s |
| `--preset paper --backend jax` | **GPU 6** | rw_f, rw_p | 69 s |
| `--preset paper --routes rw_f_paper_budget` | **GPU 6** | rw_f_paper_budget | 7 s |
| `demo2_cost_sweep.py --backend jax` | **GPU 6** | 15 cells | 2 m 41 s |

### 7.3 RW-F — the exactness anchor holds

| configuration | rays | NCC | intensity MSE | complex rel-L2 | phase-aligned rel-L2 | peak GB | t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| enumerated, every propagating mode | 39 601 | **1.000000** | **1.03e-32** | **7.34e-13** | **7.11e-13** | 0.228 | 2.64 s |
| Table S2's stochastic budget | 1 100 000 | 0.998693 | 1.62e-10 | 8.87e-02 | 8.87e-02 | 6.327 | 3.49 s |

One full-aperture patch enumerating every propagating mode reproduces the ASM
reference **to float64 round-off** — rel-L2 7.3e-13 on the complex
field, intensity MSE 1e-32. That is the anchor CHE-241 asks for and it is intact.

Note what the second row shows: at the paper's own *stochastic* 1.1e6-secondary
budget the same route gives rel-L2 8.9e-2 — a factor of 1.2e11 worse than
enumerating. The anchor is a property of enumeration, not of the full-aperture
patch.

**Standing.** The oracle is `src/verification/asm_oracle.angular_spectrum_float64`
— *this repository's own* float64 angular spectrum. `AGENTS.md` forbids repository
numerical code as the sole correctness oracle for the same numerical code, so this
is **characterization**, not validation. It establishes that the patch route and
the ASM route agree where the theory says they must; it does not establish that
either is right. The probe's own record says the oracle is "independent of the
coupler under test", which is true of the *coupler* and not of the repository.

### 7.4 RW-P — convergence toward the anchor

**The precision differs between the two, and it is declared rather than implied.**
`rw_f` ran fp64 / `complex128`; `rw_p` and every cost-sweep cell ran fp32 /
`complex64` — which is the presets' own choice, not this run's. Nothing below is
near a complex64 floor (the best rel-L2 is 2.9e-2, seven orders above it), so the
convergence claim is unaffected; it is stated because the anchor and the sweep are
different dtypes and dtype is one of this report's declared non-negotiables.

The full 15-cell `(incident, secondary)` grid, no cells skipped:

| incident \ secondary | 100 | 1 000 | 10 000 |
| --- | --- | --- | --- |
| **100** | 0.291035 | 0.808733 | 0.908363 |
| **400** | 0.692094 | 0.954695 | 0.980163 |
| **1 600** | 0.917171 | 0.987766 | 0.994256 |
| **6 400** | 0.978978 | 0.995614 | 0.997272 |
| **16 000** | 0.991700 | 0.998696 | **0.999418** |

NCC against the matched-periodicity oracle. **Monotone in both arguments, every
row and every column**, from 0.291 at 1e4 rays to 0.999418 at 1.6e8 — 4.2 decades
of ray budget, with no non-monotonicity anywhere. `1 − NCC` falls 0.709 → 5.8e-4,
about 3.1 decades, and complex rel-L2 falls 2.763 → 0.0286 over the same span. That is the convergence relationship CHE-241
asks for, and it is not merely "monotone-ish".

Cost, which is the other half of the sweep. **This is one slice** — the
`secondary = 10 000` column, plus the 1e4 corner — and the slice matters, because
total rays does not determine cost:

| incident × secondary | total rays | wall clock | rays/s | batches |
| --- | --- | --- | --- | --- |
| 100 × 100 | 1e4 | 2.69 s | 3.7e3 | 1 |
| 100 × 10 000 | 1e6 | 2.41 s | 4.1e5 | 1 |
| 1 600 × 10 000 | 1.6e7 | 5.96 s | 2.7e6 | 4 |
| 6 400 × 10 000 | 6.4e7 | 24.0 s | 2.7e6 | 16 |
| **16 000 × 10 000** | **1.6e8** | **58.2 s** | **2.7e6** | **40** |

Throughput saturates at ~2.7e6 rays/s **on this slice**, and the small cells are
dominated by fixed cost — the shape the probe's own `cost_model` note predicts for
an O(N_rays × N_pixels) reconstruction. Off the slice it does not hold: the
*other* 1.6e7-ray cell (16 000 × 1 000) took 17.66 s at 9.1e5 rays/s, and the
1.6e6-ray `secondary = 100` cell ran at 1.2e5 rays/s. Secondary count per
incident, not total rays, is what sets throughput here.

Peak device memory is omitted from the rows above rather than repeated: JAX
reports a process high-water mark, so every cell after the first large one shows
the same 11.52 GB and the column would say nothing per row. §7.9 has the figure
and what it means.

### 7.5 Comparison to the existing records — reproduces exactly

Against the committed `che-140` records at commit `da2e757b`:

| quantity | committed | this run | agreement |
| --- | --- | --- | --- |
| rw_f NCC (paper) | 1.000000 | 1.000000 | exact |
| rw_f rel-L2 (paper) | 7.3383e-13 | 7.3383e-13 | exact |
| rw_p NCC (paper) | 0.999418 | 0.999418 | exact |
| rw_p rel-L2 (paper) | 2.8562e-02 | 2.8562e-02 | exact |
| rw_f_paper_budget NCC | 0.998693 | 0.998693 | exact |
| smoke-numpy rw_p NCC | 0.844177 | 0.844177 | exact |
| cost sweep best NCC | 0.9994182326189224 | 0.9994182326189224 | exact |
| cost sweep cells measured | 15 | 15 | — |

Every accuracy number reproduces to all printed digits. Only wall clock differs
(rw_p paper: 94.9 s committed against 62.5 s here — a different A6000 and a
different driver, not a numerical change), which is the expected axis of variation
and is why the ticket separates cost metrics from accuracy metrics.

One pre-existing record does *not* line up with the current preset, and it is a
record-vintage question rather than a discrepancy: `demo2_smoke_jax.json` reports
`rw_f` with **20 000** rays at NCC 0.921456, where the current `smoke` preset's
`rw_f` enumerates **39 601** modes at NCC 1.000000. The record predates a preset
change. Recorded rather than chased — reproducing it was not asked for.

### 7.6 Comparison to the paper — reported separately, never a threshold

SI Table S2, System 2, on 1× RTX A6000 48 GB / CUDA 12.4:

| | paper | this run | note |
| --- | --- | --- | --- |
| RW-F secondary rays | 1.1e6 | 1.1e6 (matched budget) | |
| RW-F MSE | 4.414e-10 | 1.62e-10 | different reconstruction |
| RW-F NCC | 0.997 | 0.998693 | |
| RW-F peak memory | 8.086 GB | 6.327 GB | |
| RW-F runtime | 0.097 s | 3.49 s | **36× slower** |
| RW-P incident × secondary | 1.6e4 × 1e4 | 1.6e4 × 1e4 | matched |
| RW-P batches | 2 | **40** | forced; see below |
| RW-P peak memory | 29.213 GB | 11.52 GB | |
| RW-P runtime | 2.275 s | 62.47 s | **27× slower** |

The runtime gap is the probe's own documented cost-model difference, not a
regression: this reconstruction is O(N_rays × N_pixels) — a separable
`einsum("n,ny,nx->yx")` — where the paper's is O(N_rays) + one FFT. The 40 batches
against the paper's 2 follow from the same fact: 1.6e8 rays at 100² needs ~256 GB
of separable factors in one call — the probe's own estimate is "4e6 rays × 100
pixels × 2 separable factors × 8 B is ~6 GB", which scales to 256 GB and is
consistent with 40 × ~6.4 GB — and 40 chunks holds it to ~11.5 GB. Batching
cannot change the estimator — the total is known before the first chunk, the
`1/N` is applied once at `finalize`, and chunking is over whole patches
(`_demo_support.patch_route`) — so it costs accuracy nothing. That argument is
**structural and unmeasured here**: `demo2_cost_sweep.py` derives the batch count
as `ceil(total / 4e6)`, so no cell in the grid holds the ray budget fixed while
varying batches, and §7.4 therefore does not confirm it. A controlled
batch-invariance run is a follow-up (§10).

**None of these is a pass threshold.** Different implementation, different ray
budget, different reconstruction algorithm — quoting NCC 0.997 as a gate would be
circular validation, and the probe's own record says so in a top-level field.

### 7.7 Optiland is not exercised

Stated because it is the one thing a reader could wrongly infer. Demo2 is a bare
SLM behind a circular amplitude mask and a sensor, with **no refractive surface**,
so nothing here validates the ray engine either way. The probe records
`optiland_used: false` with that reason attached, and this workstream adds no ray
evidence to workstream A's.

### 7.8 Three deliberate deviations from the notebook, preserved

Carried through unchanged from the probe, and each is scored rather than argued:

* **Coherent field accumulation** (SI eq S5) rather than the notebook's
  `|field|²`-then-square-again. The notebook variant is computed alongside on
  every route: NCC 0.982 against 1.000 on rw_f, 0.979 against 0.999 on rw_p.
* **Unflipped phase.** The notebook's `flip(phase, dims=[0,1])` compensates
  DeepLens's `Ray.flip_xy`, which this pipeline does not have. Both orientations
  are scored once: the flipped mask scores 0.623 against 1.000 on rw_f, so the
  flip is not physically required here.
* **Origin at index `n // 2`**, this repository's rule, where upstream uses
  `(n−1)/2`.

### 7.9 The figure

`outputs/che-238-overnight/workstream-c/figures/demo2_fig5b_sensor_fields.png`,
rendered by the probes' own `demo_figures.py::figure_demo2` from this run's
fields. Eight panels: the DOE transmission phase, the float64 ASM oracle at both
paddings, the three routes, the spatially resolved RW-P residual, and a
log-scale central row cut with all five curves overlaid.

All five reconstructions are indistinguishable by eye, which is what the panel
captions' numbers say quantitatively (NCC 1.000000 / 0.998693 / 0.999418). **The
informative panel is the residual**: `|RW-P − matched oracle| / max|oracle|` is
concentrated on the bright ring and arc edges rather than spread over the frame,
which is the signature of estimator shot noise scaling with signal amplitude —
not a structural error in the reconstruction.

**How it was produced, because it required one extra run.** `figure_demo2` reads
`demo2_paper_figure_jax_fields.npz`, and `*_fields.npz` is `.gitignore`d, so the
figure cannot be rendered from a fresh checkout at all. The three-route run was
therefore repeated with `--save-fields` under the name `che238_demo2_figure`
(rw_f, rw_f_paper_budget, rw_p in one process, 73 s on GPU 6, numbers identical to
§7.3–§7.4), and rendered through a scratch records directory that
`demo_figures.RECORDS` was repointed at. **The figure code itself is unmodified
and no tracked record was overwritten** — writing under the canonical name would
have clobbered the committed `demo2_paper_figure_jax.json`. `git status` in the
worktree stayed free of modifications throughout.

### 7.10 Resources

Swap **0 B** before, during and after. One GPU, one job at a time, nothing
detached, and GPU 6 back to 2 MiB after each run. No stop condition fired.

Two GPU-memory figures, and for a shared server the second is the load-bearing
one:

* **11.52 GB** is JAX's in-use high-water mark on the largest cell, which is what
  the records carry as `device_memory.peak_bytes_in_use`. It is the accumulator's
  own footprint.
* **~38 GB** is what the process actually held. The records report
  `bytes_limit = 38 275 448 832` — 75 % of the card — and neither `run.sh` nor the
  probes set `XLA_PYTHON_CLIENT_PREALLOCATE`, so the JAX client reserved that much
  on GPU 6 for the duration of each run. Anyone sizing a concurrent job on GPU 6
  needs the 38 GB, not the 11.5 GB.

## 8. Workstream D — Demo3 characterization (CHE-242)

**Status: `PASS-native`** for five of the six evidence items. **Item 5
(enumerated reference) is `BLOCKED`**, and the reason is not a resource limit —
see §8.6.

Same detached `che-140` worktree at `eb3d792` as workstream C, same record-schema
caveat (§7.1). Six records at
`outputs/che-238-overnight/workstream-d/che238_demo3_*.json`.

**Demo3 is characterization, not validation, and no golden was invented.** The
probes say so themselves in a top-level `status_of_this_evidence` field: *"demo3
has no conventional reference; every number here is a property of the estimator,
measured against itself across seeds or against another arm of the same probe."*
Nothing below is scored against an oracle.

### 8.0 Preflight and the option decision

Item 2 of §2 resolves exactly as it did for workstream C: `demo3_hologram_lens.py`
offers `--backend {numpy, jax}` and no torch option. The smoke succeeded on CPU in
7.2 s, so **Option A**, and Option B was not run.

**But Optiland itself runs on torch here, and on CUDA.** The records carry
`optiland_execution.observed = {backend: "torch", device: "cuda", precision:
"float32", grad_enabled: false}` with `torch 2.13.0+cu126` on an RTX A6000. So the
one thing the amendment worried about — a torch path — is present, is exercised,
and belongs to the *ray engine* rather than to the estimator. Item 6 is therefore
`PASS-native` on its own terms and never needed transcribing, which is what §7 of
the ticket requires of it.

| run | device | wall clock |
| --- | --- | --- |
| `demo3_hologram_lens --preset smoke --backend numpy` | CPU | 7.2 s |
| `--preset characterization --seeds 20260822` | GPU 6 | 1 m 44 s |
| `--preset characterization --seeds 20260822,7,101` | GPU 6 | 4 m 51 s |
| `demo3_reconstruction_equivalence --preset characterization` | GPU 6 | 2 m 20 s |
| `demo3_variance --stage decomposition` | GPU 6 | 2 m 09 s |
| `demo3_variance --stage allocation` | GPU 6 | 4 m 10 s |

Configuration for every characterization run: 420² sensor at 4.2 µm, patch 101²,
3 000 patches × 20 000 secondary = **6e7 rays per route per seed**, fp32 /
`complex64`, 60 patch groups × 5 secondary chunks = 300 chunks.

### 8.1 Item 1 — RW-F against RW-P

**NCC 0.014333**, intensity MSE 6.454e-11, at 6e7 rays per route.

That is not agreement, and the probe's own note says why it should not be read as
disagreement about the *optics*: *"the only cross-check this system has — there is
no external reference — so it is reported as agreement between two of our own
routes, not as a validation of either."*

CHE-242 asks that a disagreement be attributed among estimator variance,
reconstruction error, ray clipping, patch coverage and optical-model disagreement
before concluding. The other five items do exactly that, and the attribution is
unambiguous:

| candidate cause | measured | verdict |
| --- | --- | --- |
| **estimator variance** | seed-to-seed NCC **0.0037–0.046** *within* each route (§8.2) | **dominant** |
| reconstruction error | **not in this comparison at all** — both routes ran `ramp_sum`, the exact O(rays × pixels) path, with `kspace_oversample: null` | excluded by construction |
| ray clipping | **0 rays clipped with power**; energy conserved to ≤8.7e-9 (§8.5) | not it |
| patch coverage | 8.88× draw coverage, `A_draw/A_patch` correction applied; ~10% sensor capture, by design | not it |
| optical-model disagreement | see the agreement statistic below | **not separable, but not excluded** |
| transcription error | not applicable — Option A | n/a |

**The right statistic is one the probe already ships, and it changes the reading.**
`demo3_hologram_lens.py::_noise_limited_agreement`, reachable through the
`--agreement-from` entry point, exists for exactly this question, and its
docstring says a bare cross-route NCC "on its own says nothing": if two routes
estimate the *same* signal under independent noise, then

    NCC(A, B) ~= sqrt( NCC(A, A') * NCC(B, B') )

**Computed through the probe's own path**, not by hand — `--agreement-from` over
two 3-seed field files, which averages the full 3 × 3 cross-seed matrix rather
than one seed pair:

| | value |
| --- | --- |
| `mean_self_ncc`, rw_f | 0.0056150 |
| `mean_self_ncc`, rw_p | 0.0442990 |
| `predicted_if_same_field` = `sqrt(·×·)` | **0.015771** |
| `mean_cross_route_ncc` (9 pairs) | **0.016312** |
| **`ratio_measured_over_predicted`** | **1.034** |

By the probe's own reading that is *evidence of agreement* — the two routes are as
correlated as two noise-limited estimates of the same signal should be — and it is
the strongest statement available without converging either. **The uncertainty on
that ratio is not quantified**, so it is a consistency check and not a measured
agreement.

An earlier draft of this section reported **0.909** for that ratio. That figure was
computed by hand against the *same-seed* cross-route NCC (0.014333) where the
statistic is defined on the **mean over all nine cross-seed pairs** (0.016312).
Same-seed is the wrong numerator: two routes sharing a seed do not share an RNG
stream, so it is one draw from the same distribution the other eight come from,
and using it alone both biases the estimate and discards eight ninths of the data.
The corrected value is closer to unity, so the conclusion strengthens rather than
changes.

One claim an earlier draft of this section made is **false and is withdrawn**: that
"each route disagrees with itself across seeds by more than the two routes disagree
with each other". That holds for rw_f (worst pairwise 0.0037 against 0.0143) and
**not** for rw_p, whose pairwise NCCs are 0.0416–0.0460, all *above* the
route-to-route figure. The correct statement is narrower: at 6e7 rays the
reconstruction is noise-dominated, a bare cross-route NCC has no resolution on the
optical model, and the geometric-mean statistic is what carries what little signal
there is.

### 8.2 Item 2 — seed-to-seed convergence, three realizations

Seeds 20260822, 7, 101. `meets_ac4_minimum_of_three: true` on both routes.

| route | pairwise NCC | worst | mean per-pixel relative spread | bright pixels |
| --- | --- | --- | --- | --- |
| **rw_f** | 0.005413, 0.007775, 0.003657 | **0.003657** | **0.6507** | 164 741 |
| **rw_p** | 0.045954, 0.045364, 0.041579 | **0.041579** | **0.6463** | 149 601 |

The noise floor declared by the sibling `demo3_variance` records — `3/sqrt(N_px)`,
identical here because both use a 420² sensor — is **0.007143**, so rw_f's
pairwise NCCs straddle it: *two of the three are at or below the floor and are
zero with an error bar.*

**So the answer to "is speckle reproducible enough to support the existing
characterization" is no.** Per-pixel intensity varies by 65% of its own mean
between seeds, and two of rw_f's three pairwise NCCs are at or below the floor.

What establishes that this is the *existing* state rather than a regression is the
committed record, not the source comment. `demo3_characterization_rw_f.json`
reports worst pairwise NCC **0.001849** and `mean_relative_spread` **0.6511**,
against this run's 0.003657 and 0.6507 — the same picture. (An earlier draft cited
the preset's own source comment about "two independent noise fields ... SI Figure
S4's undersampling artifact reproduced exactly". That comment is about the
**rejected** 1680² sensor at 8e6 rays, and its very next line says "420^2 at 3e7
rays is 341 rays per pixel and shows structure" — i.e. it asserts the opposite
about the preset actually run. The citation was wrong; the committed record is the
right support and says what the draft claimed.)

No per-pixel claim can rest on a single seed here.

Against the committed baselines (seeds 20260822, 7, 131 — two of three shared):

| quantity | committed | this run | agreement |
| --- | --- | --- | --- |
| rw_f pairwise NCC, seeds (20260822, 7) | 0.005412906224785008 | 0.005412944928383165 | 7e-6 relative |
| rw_f captured fraction, seed 20260822 | 0.10101172987776913 | 0.10101173096943086 | 1e-8 relative |
| rw_f captured fraction, seed 7 | 0.10009195604296937 | 0.100092 | agrees |
| rw_p captured fraction, seed 20260822 | 0.0989791657901167 | 0.09903421954874955 | **5.6e-4 relative** |

The rw_f numbers reproduce to float32 GPU nondeterminism. The rw_p captured
fraction differs in the fourth digit — 0.056% — for the *same* seed, and it is
**an open minor discrepancy with no mechanism offered**.

An earlier draft blamed the RNG stream moving under a different chunking. The
records refute that: the empty-draw counts are **bit-identical** to the committed
baseline on both shared seeds (5 920 000 at 20260822, 5 660 000 at 7), so the
patch-centre stream demonstrably did not move; and rw_f underwent the same
chunking change and reproduced to 1e-8. The one configuration difference that is
known: the baseline ran `--rays-per-chunk 1e6` where this run took the probe
default of 2e5, giving 300 chunks against 60. Why that would move an rw_p power
sum by 0.056% and leave rw_f at 1e-8 is not established here. It is 0.056% on a
coverage bookkeeping figure and moves no conclusion, so it is recorded rather than
chased.

Wall clock is where the two runs genuinely differ: 37.9–55.3 s per route-seed here
against 124.9–150.1 s committed. A ~3× speedup on a different A6000, no numerical
consequence.

### 8.3 Item 3 — estimator variance, `V(P,S) = A/P + B/(P·S)`

`P·V = A + B/S` is linear in `1/S`, so a sweep in `S` at fixed `P = 1000`
separates the terms by a straight-line fit rather than by an argument:

| S | V (field variance sum) | jackknife | P·V |
| --- | --- | --- | --- |
| 2 500 | 9 228.58 | ±0.84% | 9.2286e6 |
| 5 000 | 4 813.66 | ±0.71% | 4.8137e6 |
| 10 000 | 2 590.45 | ±0.88% | 2.5904e6 |
| 20 000 | 1 474.16 | ±0.77% | 1.4742e6 |

**The two-term form holds**: relative RMS residual of the linear fit **0.147%**.

Fitted terms, against `benchmarks/reports/2026-08/demo3_estimator_variance.md`:

| | existing report | this run | difference |
| --- | --- | --- | --- |
| `A` (falls with `P`) | 3.751e5 | **3.7398e5** | 0.3% |
| `B` (falls with `P·S`) | 2.202e10 | **2.2148e10** | 0.6% |
| fit residual | 0.14% | 0.147% | — |
| direction share at shipped `S` | 74.6% | **74.75%** | 0.2% |

**Budget allocation stays consistent.** Cost model `c(P,S) = P(a + bS)` refitted:
`a = 1.6514e-3` s/patch (report: 2.039e-3), `b = 2.6635e-7` s/ray (report:
2.586e-7), residual 6.24% (report: 6.4%). Then

    S* = sqrt(a·B / (b·A)) = 1.92e4      (existing report: 2.15e4)

against a shipped `S = 20 000` — the two independent fits **bracket the shipped
value from below and above**, so the existing report's conclusion that the shipped
split was already optimal is reproduced rather than merely repeated.

Measured `V × cost` over four cells at fixed `P·S = 2e7`:

| P × S | V | modelled cost | V × cost, this run | V × cost, report |
| --- | --- | --- | --- | --- |
| 400 × 50 000 | 2 037.4 | 5.99 s | 1.22e4 | 13 897 |
| **1 000 × 20 000** (shipped) | 1 474.2 | 6.98 s | **1.029e4** | **9 610** |
| 2 500 × 8 000 | 1 244.7 | 9.46 s | 1.18e4 | 12 144 |
| 5 000 × 4 000 | 1 171.7 | 13.58 s | 1.59e4 | 18 365 |

**The shipped cell is the measured minimum in both runs.** Note that `V` alone
falls monotonically with `P` at fixed ray count — 2 037 → 1 172 — so a sweep that
ignored cost would have concluded the opposite; fixed `P·S` is not fixed cost, and
that is the whole reason the allocation question is not answered by the ray count.

Only the timing constants moved, `a` by 19%, which is the axis a different GPU is
expected to move. `A` and `B` are properties of the estimator and reproduce under
1%.

### 8.4 Item 4 — reconstruction equivalence, fast against exact

`ramp_sum` (exact, O(rays × pixels)) against `kspace_splat` (fast) at five
oversampling factors, 6e7 rays, 420² sensor:

Route `rw_p` only.

| oversample | NCC | relative L2 | **power ratio to exact** | wall clock | speedup | dropped |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.741464 | 4.954e-1 | **0.4491** | 20.68 s | 1.67× | 0 |
| **1.5** (shipped default) | 0.915061 | 2.626e-1 | **0.6488** | 19.11 s | 1.81× | 0 |
| 2.0 | 0.969531 | 1.567e-1 | **0.7734** | 19.06 s | 1.82× | 0 |
| 3.0 | 0.993508 | 7.322e-2 | **0.8884** | 19.22 s | 1.80× | 0 |
| **4.0** | **0.997901** | **4.197e-2** | **0.9349** | 19.94 s | 1.74× | 0 |

Exact route: 34.63 s. Monotone in oversample and **no rays dropped at any factor**.

**The power ratio is the column that matters most and NCC hides it.** NCC is
normalization-blind, so "the same field to NCC 0.998" is true at oversample 4 and
the fast path still carries **6.5% less total power** than the exact route there —
and **35% less** at the shipped default of 1.5, where NCC reads 0.915. A consumer
comparing intensities rather than correlations gets a different answer from the two
paths at every factor measured. Reported because an equivalence check that quotes
only NCC would have called this equivalent.

**The probe's tolerance was not changed.** `--oversamples` was left at its default
`1.0,1.5,2.0,3.0,4.0` and the shipped default oversample is 1.5, which scores
0.915 — so this run says what the fast path costs at its shipped setting rather
than only at its best. Nothing here is reported as a met tolerance.

### 8.5 Item 6 — Optiland clipping and energy accounting (`PASS-native`)

The only item in this workstream where Optiland genuinely enters, and it ran on
its **torch/CUDA** backend at float32 with gradients off.

| | rw_f | rw_p |
| --- | --- | --- |
| rays launched | 6e7 | 6e7 |
| `launched_sum_abs_amplitude_squared` | 1.884686800968e12 | 1.161869021786e13 |
| `survived_sum_abs_amplitude_squared` | 1.884686796800e12 | 1.161869026099e13 |
| relative energy change (seed 20260822) | **−2.2e-9** | **+3.7e-9** |
| worst over the three seeds | −8.7e-9 | −4.7e-9 |
| **`clipped_with_power`** | **0** | **0** |
| `clipped_with_power_fraction` | 0.0 | 0.0 |
| `invalidated_rays` | 0 | 5 920 000 |
| `launched_with_zero_amplitude` | 0 | 5 920 000 |
| `captured_by_sensor_fraction` | 0.1010 | 0.0990 |

**No anomalous clipping: not one ray carrying power was clipped by the refractive
singlet**, and energy is conserved through the trace to float32 round-off. The
rw_p `survived` sum is very slightly *larger* than `launched`, by 3.7e-9 relative
— that is accumulation round-off in fp32, not created energy, and it is the right
size for a 6e7-term sum.

The 5.92e6 invalidated rays in rw_p are **not** a clip and the record insists on
the distinction: patch centres are drawn over the aperture dilated by half a
patch, so a centre near the rim yields a patch partly outside the DOE, and those
rays carry zero amplitude *by construction*. The `A_draw/A_patch` coverage factor
(8.88×) is exactly their correction. They are counted separately because
Optiland's `intensity > 0` test cannot tell them from a clip, and folding them in
would invent an energy loss. Across the three seeds the count is 5.66e6–5.94e6,
consistent with a random draw.

`captured_by_sensor_fraction ≈ 0.10` is the deliberate coverage cost of this
preset, not a loss: the 420² sensor spans ±0.882 mm of a ±2.8 mm image. Stated
because 10% capture would otherwise look like an anomaly.

### 8.6 Item 5 — `BLOCKED`: the probe script is 40 626 NUL bytes

`benchmarks/probes/ray_wave/demo3_enumerated_reference.py` **exists as a path and
contains no code.** It is 40 626 bytes, every one of them `0x00`.

The corruption is in `che-140`'s history, not in this checkout:

* `git cat-file -p HEAD:...demo3_enumerated_reference.py` returns 40 626 NULs;
* `git cat-file -s` agrees on the size;
* `git status` reports the worktree file as unmodified, so checkout and blob match;
* the file has exactly one content commit, **`7625556` (CHE-101, 2026-08-24,
  "enumerate the modes, and shard the estimator that no longer fits")**, and the
  blob is all-NUL *there*. It was never good in git.
* Its eight sibling probes in the same directory contain zero NUL bytes.

So item 5 cannot run, and the reason is data corruption in the source branch
rather than the resource limit CHE-242 §7 anticipated ("enumerated-reference
evidence where resources permit"). Resources were not the constraint: GPU 6 was
idle throughout.

What does still exist is that probe's **output**:
`demo3_enumerated_reference_rwf_kspace.json`, `..._rwf_ramp.json`,
`demo3_enumerated_positions.npz` and several `demo3_enum_*` records, evidently
written before the commit that corrupted the script. Those records were **not**
used as a substitute: a record whose generator cannot be read or re-run is not
evidence this run can stand behind, and quoting its numbers as a reproduction
would be the opposite of what this ticket is for.

Not repaired. Reconstructing the script would be writing a *new* probe, which is
out of scope and would not be the probe the ticket names. Follow-up in §10.

### 8.7 Nothing promoted, nothing invented

* No golden was invented for Demo3. Every comparison is route-against-route,
  seed-against-seed, or arm-against-arm of the same probe.
* No probe tolerance was changed. `--oversamples`, `--replicates`, `--patches`,
  `--secondary-count`, `--ladder-patches` and `--target-ncc` all ran at their
  defaults.
* No historical record was written to. `git status` in the worktree shows only
  untracked `che238_*` additions and **no modifications**, verified after every
  run. `benchmarks/perf/records/` was never touched.
* Two variance stages were not run — `candidates` (whether the variance can be
  *reduced*, by importance density) and `ladder`/`ladderfit`/`l1map`. CHE-242 item
  3 asks for the model check and budget-allocation consistency, which are
  `decomposition` and `allocation`; the reduction question belongs to CHE-120 and
  is not this ticket's.

### 8.8 The figure

`outputs/che-238-overnight/workstream-d/figures/demo3_fig5c_sensor_fields.png`,
from `demo_figures.py::figure_demo3`. Six panels: the DOE phase profile, both
routes at one seed, both routes as a coherent mean of the three seeds, and a text
panel the figure generates itself.

**The figure is honest about what it shows and the text panel is the reason to
keep it.** It prints its own statistics — mean cross-route NCC 0.0163, predicted
0.0158, ratio 1.03, seed-to-seed 0.0056 and 0.0443 — and then states the reading:
*"each route disagrees with ITSELF at the same order as with the other one. Both
routes are unbiased estimators of the same field. Their disagreement is fully
explained by their own seed-to-seed scatter, so it is Monte-Carlo variance, not a
physics discrepancy between the routes."* Note the phrasing the probe's author
chose — **"at the same order as"**, not "by more than", which is exactly the
hedging §8.1's first draft got wrong.

The single-seed panels look like noise, and the figure's title says so:
`NOISE-LIMITED, not converged`. The three-seed coherent means begin to show the
ring structure, RW-P more visibly than RW-F. The figure also carries the
convergence extrapolation — log-log slope 0.956 in ray count, NCC 0.9 at 1.78e9
rays (~1.2 h per run) against the paper's own 2.6e9 — and states that no oracle
panel exists because the paper states no conventional reference does.

Producing it needed three extra runs: rw_f and rw_p each at the characterization
preset with all three seeds (2 m 05 s and 2 m 59 s on GPU 6), then
`--agreement-from` over the two field files. The single-route numbers are
identical to §8.2's. Rendered through the same scratch-records indirection as
§7.9; no tracked record was overwritten.

**One demo3 figure cannot be produced at all.** `figure_demo3_kspace_rwf` — the
*converged* full-aperture panel, and per `demo_figures.py`'s own docstring "the
only demo3 panel here that is allowed to look like a formed image" — reads
`demo3_enumerated_reference_rwf_{ramp,kspace}_fields.npz`, which are written by
`demo3_enumerated_reference.py`. That is the 40 626-NUL-byte file of §8.6. So the
corrupt script does not only block evidence item 5; it blocks the one demo3 figure
that would show a converged image.

### 8.9 Resources

Swap **0 B** throughout, re-checked after every run. GPU 6 only, one job at a
time, nothing detached; peak JAX in-use 2.38 GB (characterization) and 0.29 GB
(variance) against the same ~38 GB client reservation §7.9 describes. GPU 6 idle
at 2 MiB after each run. No stop condition fired.

## 9. Code changes, verification gates, and what was not run

*Accumulated as the night proceeds.*

### 9.1 Code changes

**No production code changed. `src/` is untouched.** What landed is the
verification harness under `benchmarks/verification/` (six modules) and this
report, plus one test constant: `tests/unit/test_documentation_references.py`'s
`benchmarks/` file count moved 10 → 17, which is the growth that gate's own
docstring sanctions.

All three findings (§5.3, §6.2, §8.6) are recorded and deliberately unfixed. Each
would change a physical claim, a frozen number or another branch's history, and
CHE-238's code-change policy puts all three behind their own ticket and independent
review.

### 9.2 Gates run

| gate | when | result |
| --- | --- | --- |
| `make test` | after CHE-238 | 1745 passed, 7 skipped, 12 deselected, 106 s |
| `make check-arch` | after CHE-238 | dependency graph OK, class budget OK |
| `./run.sh ruff check .` | after CHE-239 | all checks passed |
| `make test` | after CHE-239 | 1745 passed, 7 skipped, 12 deselected |
| `./run.sh ruff check .` | after CHE-240 | all checks passed |
| `make test` | after CHE-240 | 1745 passed, 7 skipped, 12 deselected |
| worktree cleanliness | after CHE-241 | `git status` in the `che-140` worktree shows only untracked `che238_*` records; every historical record untouched |
| `make test` | after CHE-242 | 1745 passed, 7 skipped, 12 deselected |
| `./run.sh ruff check .` | after CHE-242 | all checks passed |
| worktree cleanliness | after CHE-242 | re-verified after every run: only untracked additions, **no modifications** |

| `make test-slow` | after CHE-242 | **12 passed**, 1752 deselected, 102 s |
| `MOA_GPUS=device=6 make test-gpu` | after CHE-242 | **7 passed**, 1757 deselected, 7 s |

**A correction, because the first draft of this section got it backwards.** It said
`make test-slow` and `make test-gpu` "select nothing in this tree" and that "the GPU
tests went with the old tree, and the conftest gating hook with them". Both claims
are false. `-m slow` collects **12** tests — including
`test_the_estimator_converges_at_the_monte_carlo_rate` and three
`test_psf_verification` cases — and `-m gpu` collects **7**;
`tests/conftest.py`'s opening line is *"CHE-173 (R02.1) restores the `gpu`-marker
gating that the greenfield deletion removed"*, and it is what produces the 7 skips
`make test`'s own line reports. The arithmetic was visible on the same line all
along: 1745 + 7 + 12 = 1764 = the whole suite.

The claim came from the stale comments at `Makefile:28-31` and `Makefile:46-49`,
copied rather than checked. Both gates have now been run and both pass. The Makefile
comments are a follow-up (§10).

**Genuinely not run:** the `che-140` branch's own test suite, which this run never
invoked because it changed no code there.

### 9.3 Resource incidents

One, and it was contained. The Tier-2 sweep aborted mid-run when a tutorial cell
called `Optic.draw3D`: VTK failed to reach an X server, EGL and OSMesa in turn and
then **aborted the process**, which no `except` can catch, losing the whole run's
record. Fixed by neutralizing the method for the duration of a prologue rather
than by skipping the cell — the first attempt did skip the cell and silently lost
tutorial 4f's entire lens, which was built in the same cell as a *commented-out*
`# lens.draw3D()`.

One Bash-level timeout left an orphaned `agent_solver` container running the
Tier-2 sweep; it was `docker stop`ped before the next run. Swap stayed at 0 B
throughout. Peak host RSS across the night so far: 1.0 GiB (workstream A's Tier 2).

Workstream C used **GPU 6** and nothing else. Peak JAX in-use 11.52 GB with a
~38 GB client reservation (§7.9); GPU 6 idle at 2 MiB before and after each run.
No stop condition fired at any point.

### 9.4 The temporary worktree, removed

Workstreams C and D reached the `che-140` tree through
`git worktree add --detach <scratch> origin/chengjiazhou4802/che-140-…`. That
registration was repo state this run created, so it was removed with
`git worktree remove --force` and `git worktree prune` once the records were
copied out. `git worktree list` is back to the two pre-existing entries, no branch
was created, and the working branch never moved off
`chengjiazhou4802/che-152-greenfield-rewrite`.

### 9.5 Figures — omitted on the first pass, then produced

The parent's execution order names "figures + final report" as step 9. The first
pass through this report produced the report and **no figures at all**, and did
not list them in the not-run section either — so the omission was invisible in a
document whose contract is that an unrun check says so. Both failures are mine and
the second is the worse one.

Two figures now exist, rendered by the probes' own `demo_figures.py` with the code
unmodified:

| figure | path |
| --- | --- |
| Demo2, paper Fig 5b | `outputs/che-238-overnight/workstream-c/figures/demo2_fig5b_sensor_fields.png` |
| Demo3, paper Fig 5c | `outputs/che-238-overnight/workstream-d/figures/demo3_fig5c_sensor_fields.png` |

Neither could be rendered from the records the first pass saved: `figure_demo2`
and `figure_demo3` read `*_fields.npz` arrays, `benchmarks/probes/records/**/*_fields.npz`
is `.gitignore`d, and workstream C never passed `--save-fields`. Four extra runs
(73 s + 2 m 05 s + 2 m 59 s + an `--agreement-from` pass) supplied them. Details in
§7.9 and §8.8.

Producing them also **corrected a number in §8.1**: the noise-limited-agreement
ratio is 1.034 through the probe's own path, not the 0.909 this report first
carried from a hand computation against the wrong cross-route quantity.

**A third demo3 figure remains unproducible** and its cause is §8.6's corrupt
script — see §8.8.

### 9.6 What did not run, across the whole night

* The sampling-bound refusal (§6.4) — declared in the catalog, enforced nowhere,
  so there is nothing to test.
* Item 5 of workstream D (§8.6) — blocked on a corrupt script, not on resources.
* No surface aperture is exercised anywhere in workstream A, so `to_ray_bundle`'s
  clipping path is untested by it (§5.6). Workstream D exercises clipping
  bookkeeping instead and finds zero clipped rays with power (§8.5).
* The 139 remaining gallery notebooks, the Chromatix gallery, and the four
  tutorials blocked on container dependencies.
* Demo3's `candidates` / `ladder` / `l1map` variance stages, which belong to
  CHE-120 rather than to this ticket (§8.7).
* No gradient was claimed anywhere. Every descriptor touched is `forward_only`.
* `figure_demo3_kspace_rwf` and `figure_demo3_kspace` — the first blocked by the
  corrupt script (§8.8), the second needing `demo3_stage_ramp` and
  `demo3_kspace_rw_p` field dumps this run did not regenerate.

## 10. Follow-up tickets recommended

*Accumulated as the night proceeds.*

**Priority 1 — from §5.3.** `SOM_SPOT_DIAGRAM` analyses at
`setup.reference_wavelength_um` while reporting `source.wavelength_um`. Needs its
own ticket: it changes a solver adapter's behaviour, moves frozen numbers, and
`AGENTS.md` puts it behind independent scientific review. The fix is small
(`build_lens` declares one wavelength; the analysis has to be asked for the
source's, or the record has to stop claiming one that did not run) but the
*decision* between those two is not this ticket's to make. Regression coverage
must include a case where the source wavelength differs from the reference — the
current tests do not have one, which is why this survived.

**Priority 2 — from §6.2.** The recorded "maximum difference of **exactly 0.0** in
float32 over a 512² grid" does not reproduce in the quantity it names: the
propagator arrays differ by 6.1e-5. The substitution is algebraically exact and the
claim's substance holds, so the fix is to the *wording* in
`src/backends/chromatix/solver.py` and `src/operations/catalog.py` — "to float32
round-off, ~6e-5 rad". A separate ticket because CHE-240's non-goals forbid
converting a stale recorded expectation into a new baseline here.

**Priority 3 — from §6.4.** `z <= N pitch^2 / lambda` is declared as a `validity`
entry on both propagators and enforced nowhere. Either add the refusal (with a
`CONTRACT_CODES` entry) or state in the descriptor that the domain is declared and
unenforced. The declared form also omits the medium index.

Also worth tickets:

* The recorded 2.3e-1 and 4.9e-6 in `VALIDITY_NOTES['paraxial']` do not state the
  aperture size or the aperture geometry, and both matter — the geometry by 36×
  (§6.3). Adding "square, half-width w, circular super-Gaussian edge" or whatever
  the original setup was would make the figures reproducible.
* Extend `extract_setup` to the finite-conjugate path: read the object distance
  into `SourceSpec.object_distance_mm` and verify an object-height → field-angle
  conversion. That is what unblocks tutorial 4e and the `UVProjectionLens` sample
  (§5.5).
* A controlled batch-count invariance run on Demo2: hold the ray budget fixed and
  vary `--batches`. The estimator argument is structural and currently unmeasured
  (§7.6), and the cost sweep cannot measure it because it derives batches from the
  budget.
* Exercise the clip/survival path. No system in workstream A declares a surface
  aperture, so `to_ray_bundle`'s filtering is untested by this run (§5.6 item 3).
* Add `tqdm`, `scikit-learn` and `gymnasium` to the `agent_solver` image if
  tutorial coverage past 9a/9b/9d/9f is wanted (§5.5).
* **Restore or delete `demo3_enumerated_reference.py`** on `che-140` (§8.6). Its
  only content commit stored 40 626 NUL bytes, so no readable version exists in git
  history and its committed *records* have no readable generator. A scan of all 860
  tracked files at that HEAD found exactly one all-NUL file, so this is isolated —
  the only other NUL-bearing files are legitimate binaries. Whoever owns CHE-101 is
  the one who can say whether a working copy survives outside git.
* **`kspace_splat` loses power that NCC does not show** (§8.4): 6.5% at oversample
  4 and 35% at the shipped default of 1.5. Either the fast path needs a
  normalization fix or its record needs to state the power deficit beside the NCC.
* **`Makefile`'s comments on `test-slow` and `test-gpu` are stale** and say both
  targets select nothing. They select 12 and 7 tests respectively (§9.2). The
  comments sent this run's first draft to a false claim.
* Attach `optiland_notebook_link_index.csv` to CHE-239, or drop the reference —
  see §2.2.
* CHE-238's own text says the catalog has 15 descriptors; it has 17. Cosmetic,
  recorded in §2.1, not corrected in the ticket (the ticket is not to be edited).
