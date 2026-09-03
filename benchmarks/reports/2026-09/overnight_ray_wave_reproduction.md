# Overnight verification — ray tutorials, wave kernel sweep, Demo2/Demo3

**CHE-238** (parent) with workstreams **CHE-239** (A, ray), **CHE-240** (B, wave),
**CHE-241** (C, Demo2), **CHE-242** (D, Demo3).

This document is the parent's deliverable. It is written incrementally: each
workstream appends its own section when it runs, and every section states what
actually executed. A section that says a check was not run means it was not run.

---

## 1. Executive summary

*(Filled in after the last workstream. Until then the per-workstream sections
below are the report.)*

| Workstream | Status | Section |
| --- | --- | --- |
| A — ray tutorial / system regression | **run** — 38 Tier-1 rows, 96 Tier-2 rows; one confirmed defect | §5 |
| B — wave kernel sweep | **run** — 24 rows; a second overstated record found | §6 |
| C — Demo2 reproduction | not yet run | §7 |
| D — Demo3 characterization | not yet run | §8 |

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

`jax.devices()` inside the CPU image → `[CpuDevice(id=0)]`.
`torch.cuda.is_available()` → `False`; `torch.version.cuda` → `None`.

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

The night's declared workloads are **CPU-only**: the default image ships
`torch+cpu` and a CPU-only jaxlib, and none of the seven wave kernel checks or the
three ray observables needs a device. No GPU was requested and `--gpu` was not
used, so the GPU preference (6 or 7) never became relevant. This is stated because
"we used GPU 6" and "we needed no GPU" are different results.

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
* Records are **new files**; no historical record is overwritten. Where a
  workstream reads a `che-140` record for comparison it reads it out of that
  tree and writes its own result here.
* Every record carries provenance: git SHA, branch, UTC timestamp, command,
  device, package versions (including `optiland`), dtype, the numerical
  parameters, seed where one exists, runtime, and status.
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

*Not yet run.*

## 8. Workstream D — Demo3 characterization (CHE-242)

*Not yet run.*

## 9. Code changes, verification gates, and what was not run

*Accumulated as the night proceeds.*

### 9.1 Code changes so far

**No production code changed.** `src/` is untouched. What landed is the
verification harness under `benchmarks/verification/` (five modules) and this
report. The one defect found (§5.3) is recorded and left unfixed on purpose.

### 9.2 Gates run so far

| gate | when | result |
| --- | --- | --- |
| `make test` | after CHE-238 | 1745 passed, 7 skipped, 12 deselected, 106 s |
| `make check-arch` | after CHE-238 | dependency graph OK, class budget OK |
| `./run.sh ruff check .` | after CHE-239 | all checks passed |
| `make test` | after CHE-239 | 1745 passed, 7 skipped, 12 deselected |
| `./run.sh ruff check .` | after CHE-240 | all checks passed |
| `make test` | after CHE-240 | see below |

### 9.3 Resource incidents so far

One, and it was contained. The Tier-2 sweep aborted mid-run when a tutorial cell
called `Optic.draw3D`: VTK failed to reach an X server, EGL and OSMesa in turn and
then **aborted the process**, which no `except` can catch, losing the whole run's
record. Fixed by neutralizing the method for the duration of a prologue rather
than by skipping the cell — the first attempt did skip the cell and silently lost
tutorial 4f's entire lens, which was built in the same cell as a *commented-out*
`# lens.draw3D()`.

One Bash-level timeout left an orphaned `agent_solver` container running the
Tier-2 sweep; it was `docker stop`ped before the next run. Swap stayed at 0 B
throughout. No GPU was used. Peak RSS across the night so far: 1.0 GiB.

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
* Exercise the clip/survival path. No system in workstream A declares a surface
  aperture, so `to_ray_bundle`'s filtering is untested by this run (§5.6 item 3).
* Add `tqdm`, `scikit-learn` and `gymnasium` to the `agent_solver` image if
  tutorial coverage past 9a/9b/9d/9f is wanted (§5.5).
* Attach `optiland_notebook_link_index.csv` to CHE-239, or drop the reference —
  see §2.2.
* CHE-238's own text says the catalog has 15 descriptors; it has 17. Cosmetic,
  recorded in §2.1, not corrected in the ticket (the ticket is not to be edited).
