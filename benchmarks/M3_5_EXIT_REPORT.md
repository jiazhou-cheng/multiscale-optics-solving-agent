# M3.5 exit report — Pre-Benchmark Stabilization and Capability Expansion

CHE-59 (PB8). Milestone: M3.5, issues PB1–PB7 = CHE-52, CHE-53, CHE-54,
CHE-55/CHE-60/CHE-61, CHE-56, CHE-57, CHE-58.

M3.5 produced no new optical physics by design. It made the suite affordable,
gave device/precision an explicit contract, replaced hand-built sample lenses
with a canonical spec, converted the two upstream tutorial sets into knowledge
assets, and put three PSF routes side by side on one lens. This report states
what is frozen, what is not, and what M4 inherits.

**Read the verdict first: M3.5's technical work is done and the evidence is
real, but the milestone is not cleanly exitable as it stands.** Six items below
(R1–R6, "What still has to happen") are open, and two of them repeat failures
M3 already recorded against itself.

> **CHE-62 update.** R1–R4 are now resolved or explicitly dispositioned and R2/R3
> are discharged; R5 and R6 remain open and out of CHE-62's scope. Every item
> below carries its disposition inline. The single reconciled statement of what
> each M3/M3.5 item *is* now lives in
> **`benchmarks/M3_M3_5_CLEANUP_DISPOSITION.md`** — read that if this report and
> another file appear to disagree. CHE-62 changed no tolerance, no gate, and no
> physics claim.
>
> **CHE-68 update.** **R5 and R6 are now discharged**, which closes the last two
> items on this list — see their entries below, and L4. CHE-68 was documentation
> and metadata only: no kernel change, no new physics claim, no new tolerance.
> One stale statement in the Linear project description was found and
> deliberately *not* changed, because it was outside CHE-68's scope: the
> "Current Highest-Priority Blocker" section still names Optiland `opd_native`
> characterization as the blocker, which CHE-30 (M3.1) resolved. It needs its own
> ticket.
>
> **CHE-69 update.** That follow-up is done. The section now names the real
> blocker — **there is no independent PSF oracle**, so M4 cannot set a tolerance
> without circularity (PB7 F2) — and lists the two items gated on it (the
> unticketed benchmark #3 tolerance, R1(b) below; and CHE-48's unmet `1.0e-3`
> gate) plus the secondary open items by ticket. CHE-69 also found the same stale
> claim in a live knowledge asset — `knowledge/couplers/ray_to_wave/`
> `conventions.md` hazard H1 and the matching `failure_guide.md` row read as
> though `opd_native` were still uncharacterized — and corrected both. The
> refusal itself is unchanged and still correct; what changed is its stated
> reason, from "nobody has probed this" to "a native accumulator is not a
> declared physical quantity." No gate, tolerance or physics claim moved.

---

## Environment

All numbers below were measured inside the `agent_solver` container via
`./run.sh`. GPU numbers used the opt-in `agent_solver_gpu` image.

- Python 3.12.13, Linux 6.8.0-84, glibc 2.41
- optiland 0.6.0, chromatix 0.6.0 @ d24bdf0, jax 0.6.2, numpy 2.2.6,
  scipy 1.15.3, matplotlib 3.11.1
- Tree at time of writing: branch
  `chengjiazhou4802/che-61-pb4b-...` @ `4fa9bbb`; `main` @ `ec55839`

---

# What M3.5 established

## PB1–PB3 — the suite is tiered, and the dev gate is met with room to spare

CHE-52 inventoried and classified every test and produced a disposition table
(`docs/testing/test_audit.md`). CHE-53 restructured into Tier A/B/C behind
pytest markers (`docs/testing/tier_restructure.md`). CHE-54 reviewed Tier A's
numerical cost and found **no shrink was needed**
(`docs/testing/pb3_shrink_review.md`).

| Tier | Command | Measured |
| --- | --- | --- |
| A (required after every change) | `-m "not slow and not benchmark and not fmmax and not fdtdx and not sax"` | **30.85 s**, 478 passed / 21 skipped / 128 deselected *(as PB3 measured it)* |
| B (subsystem) | `-m optiland`, `-m chromatix`, `-m coupler`, `-m "fmmax or fdtdx or sax"`, `-m slow` | independently invocable, overlaps A by design |
| C (full regression) | `./run.sh pytest -q` | 627 tests, ~11 min |

The ≤~3-minute gate is met by ~6×. Slowest single Tier A test is 0.31 s; there
is no outlier to remove. The 21 Tier A skips are **not** incidental — see L1.

**Re-measured at `ee57e33` (CHE-62): 42.78 s, 670 passed / 54 skipped / 173
deselected.** The PB3 row above is what PB3 measured and is left as-is for that
attribution, but it is not the current tree: CHE-55/CHE-60/CHE-61 added tests
after PB3 ran. The gate is still met by ~4×. The skip total grew from 21 to 54
for one reason, and it is not a new debt — 21 are still L1's missing record, and
the other 33 are CHE-60/CHE-61's `gpu` quarantine (8 `test_gpu_environment.py`,
19 `test_precision_gpu_pipeline.py`, 6 `test_precision_execution_matrix.py`),
which is designed behavior and did not exist when PB3 measured.

The marker vocabulary is orthogonal on purpose: `jax`/`torch`/`integration`
mean "needs that optional install", not "slow" or "out of scope"; `gpu` means
"needs an attached CUDA device" and is quarantined to its own session.

## PB4/PB4a/PB4b — precision, dtype, device and namespace are four things

CHE-55 established capability gates; CHE-60 built and validated the GPU
container; CHE-61 unified the contract and made cross-model conversion a
negotiated bridge. Policy, capability table, measured tolerances, failure codes:
`docs/precision/precision_device_policy.md`. GPU setup and its two silent-kill
traps: `docs/testing/gpu_environment.md`.

Frozen by this work:

- **Capability is declared once** in `core/capabilities.py`; cross-model
  conversion goes through `core/precision.py`'s bridge planner under an explicit
  policy. Registry `devices`/`dtypes` are updated only after executable tests
  pass.
- **Never write a requested device or precision into an artifact — read it off
  the array.** Requested / resolved / actual are three distinct things.
- **14 structured failure codes** (`UNSUPPORTED_PRECISION`,
  `OPTILAND_CUDA_UNAVAILABLE`, `SILENT_DTYPE_DOWNCAST`,
  `REPRESENTATION_INCONSISTENT`, …) fire *before* any solver call. No late
  framework traceback where the answer was knowable in advance.
- **Two silent precision losses were found by measurement, not by reading docs**:
  JAX drops 64-bit requests without erroring, and XLA:GPU computes complex64
  matmuls in TF32.
- **Two deliberate float64-by-declaration exceptions**, called out at their call
  sites rather than inherited silently: the object-space OPL reference (~1e4
  waves — float32 would inject more error than the wavefront it corrects) and
  hexapolar ring-index assignment (a tolerance test on a ratio).
- **Refusals are real refusals**: float16 has no Optiland path and is refused;
  Chromatix has no complex128 path and it is not faked.
- **No silent CPU fallback.** Enabling the GPU mutates process-global JAX state,
  so `gpu`-marked tests run only in `./run.sh --gpu pytest -q -m gpu` and
  `conftest.py` skips them whenever anything else is selected alongside — which
  is exactly what keeps every other tier command green unchanged.

## PB5 — one canonical optical-system spec

CHE-56 introduced `OpticalSystemSpec` plus a generic deterministic Optiland
builder (`registry/prescriptions.py`, `docs/prescriptions/canonical_optical_systems.md`).
Existing sample systems are expressed in it without behavior change. The adapter
deliberately does **not** construct systems from Optiland's bundled sample
classes.

PB7 then exercised this as a real constraint rather than a stylistic one: it
transcribed the bundled `CookeTriplet` into the canonical schema and proved the
transcription bit-identical — surface positions, paraxial f/EPD/EPL/XPD/XPL/FNO,
and every traced ray array (x, y, z, L, M, N, i, opd) at both fields — aborting
if not. That is the first independent evidence the builder is faithful on a lens
nobody wrote it for.

## PB6 — both upstream tutorial sets reproduced and converted

CHE-57 executed and validated **41 Optiland tutorials and 16 Chromatix
tutorials**, then folded the results into knowledge assets rather than leaving
them as scripts: `knowledge/solvers/{optiland,chromatix}/` each now carry
`api_minimal_examples.md`, `capability_notes.md`, `conventions.md`,
`failure_guide.md`, `solver_card.yaml`, `source_manifest.yaml`, plus `probes/`,
`expected/`, and `tutorials/` (45 and 20 files respectively).

## PB7 — three PSF routes on one lens, and the FFT route is the outlier

CHE-58, full report: `benchmarks/PB7_COOKE_TRIPLET_PSF_REPORT.md`; record:
`outputs/PB7/pb7_cooke_triplet_psf_comparison.json`; entry point
`./run.sh python benchmarks/probes/pb7_cooke_triplet_psf_comparison.py`.

Cooke Triplet, λ = 0.55 µm, on-axis and 20°. A = Optiland `FFTPSF`,
B = Optiland `HuygensPSF`, C = trace → `C_RAY_TO_WAVE` → Chromatix ASM → |U|².

| | on-axis rel L2 | 20° rel L2 |
| --- | --- | --- |
| A vs B | 0.0080 | 0.313 |
| A vs C | 0.0053 | 0.315 |
| B vs C | 0.0085 | **0.0138** |

On axis all three sit at the resampling floor, and **C is no further from A or B
than A and B are from each other**. Off axis B and C agree to 1.4 % with peaks
0.03 px apart, while A sits 3.4 px away — and the cause is identified, not
merely observed: the off-axis image-space pupil is anisotropic in direction
space (F/#ₓ = 5.284, F/#_y = 6.030) but `FFTPSF` sets its pixel scale from one
scalar F/# = 5.480. That predicts a per-axis mis-scale of s_y = 1.100,
s_x = 0.964; the measured fit gives 1.0955 / 0.967 against B and
1.0948 / 0.9638 against C, within 0.5 %. Removing only that scale collapses
A–B to 0.023 and A–C to 0.028 — **91–93 % of the off-axis residual is Optiland's
FFT pixel scale, not our ray→wave path.**

Two things PB7 deliberately did *not* do, and one it exposed:

- It set **no tolerance and gates nothing** — explicitly out of scope in CHE-58.
- It could not separate N_f from NA (one system, one wavelength) — CHE-51's
  question, now canceled.
- It exposed that **A and B are not independent**: two implementations inside one
  package sharing the same Wavefront/OPD front end, same reference sphere, same
  launch-tilt removal, same pupil sampling. The A-vs-B residual therefore
  *understates* the Optiland pair's uncertainty, and C is the only route with a
  different front end. This is a three-way consistency check, not validation
  against truth. **It is the single most important finding for M4**, because M4
  benchmark #3 cannot use Optiland as its oracle.

---

# Findings that changed the conclusion

**F1 — The FFT PSF is not a reference off axis.** Before PB7 the natural
assumption was that a 31 % disagreement between our coupler route and Optiland's
FFT PSF would be our defect. It is Optiland's pixel scale, demonstrated by a
prediction from per-axis F/# that the measurement then matched to 0.5 %. Any M4
work that treats `FFTPSF` as ground truth off axis will chase a phantom.

**F2 — Two Optiland PSF methods are not two oracles.** See above. Circular
validation was avoided here only because the failure was diagnosed rather than
absorbed.

**F3 — CHE-50 is invisible only because the handoff plane is the sensor plane.**
PB7 confirms the missing `exp(ikr²/2R)` term does not appear in |U|² when
post-handoff propagation is zero. That is a property of the configuration, not
of the coupler. It activates the moment a caller propagates C's field further —
which M4's hybrid compositions will do.

**F4 — A capability claim survives only as long as someone re-measures it.**
PB4b rewrote the M1 claim audit "from tested truth" and found registry entries
that were aspirational. The same discipline is why float16 and complex128 are
refused instead of emulated.

---

# Claim audit

- `derivative.verified` stays **`false` everywhere**. No M3.5 ticket touched
  differentiability. The PyTorch→JAX and cross-framework boundary remains
  `forward_only`. PB7 records `gradient_claim: none. Forward only.`
- `C_FIELD_TO_PSF` remains absent as an architectural primitive; PSF is a
  measurement on the terminal `ComplexField`, per CHE-36. PB7 uses those frozen
  semantics unchanged and introduced no new coupler.
- Registry `devices`/`dtypes` now reflect **executed** capability, not intent
  (CHE-61). This is a strengthening of an M1-era claim, not a new one.
- `benchmarks/manifest.yaml`'s `L2-PSF-01` note — the `1.0e-3` gate unmet on the
  real traced system at 2.2e-3–2.5e-3 — **is unchanged by M3.5 and still
  stands.** See R4.
- No new physics claim is made anywhere in M3.5.

---

# Risks and known limitations

### L1 — CHE-38's consolidated probe record still has never landed
`benchmarks/probes/records/m3r_sensor_handoff.json` does not exist. This is why
21 tests skip in Tier A — the same 21 skips PB3 measured and reported as a
clean baseline. Carried unchanged from M3's L5. `L2-PSF-01` does not depend on
it, but CHE-38's own acceptance criterion is still open, and the skip count is
now embedded in M3.5's headline test numbers.

**Disposition (CHE-62): open, deliberately not regenerated, and no longer
unexplained.** CHE-62 executed no compute; the ~25-min run is tracked in
**CHE-63**. What it did establish is what actually depends on the record —
*only* those 21 assertions. `L2-PSF-01` imports the probe as a module and re-runs
it, and no gate in `manifest.yaml` reads the JSON. The skip message now names
CHE-38/CHE-62/CHE-63 and prints the regeneration command, so the skip count is
self-documenting from test output alone. The record is **not** declared
unnecessary. Item 1 of the disposition doc.

### L2 — `L2-PSF-01`'s provenance is still from a dirty tree
`outputs/M3/L2-PSF-01/provenance.json` still records `dirty_worktree: true`
against `a69fe6d9`. M3's carry-forward #7 said to discharge this "before
extending the slice, not after". M3.5 extended the slice. Unchanged.

**Disposition (CHE-62): open, tracked in CHE-63 — but the precondition is
satisfiable for the first time.** Until `ee57e33` this repository never had a
clean worktree: PB7's probe and two reports were untracked, and `dirty_worktree`
is `bool(git status --porcelain)`, so `false` was *unreachable* regardless of
when the re-run was launched (see R3). It is reachable now. CHE-63 also has to
fix the reason this survived two exits: `outputs/` is gitignored, so even a clean
re-run's provenance is not durable evidence unless it is committed to a tracked
path. Item 2 of the disposition doc.

### L3 — ~~The physical-correctness gate's disposition is now ambiguous~~ RESOLVED (CHE-62)
CHE-48 — decompose the unattributed sensor-plane residual, which *is* M3's L1 —
is marked Done with no comment, no commit, and no artifact anywhere in the tree.
Meanwhile `manifest.yaml` and `M3_SLICE_REPORT.md` still record the gate as
unmet. One of the two is wrong. See R4.

**Resolved: the ticket was wrong.** `git log --all --grep=CHE-48` returns only
`ec55839`, which *mentions* CHE-48 as a follow-up rather than resolving it, and no
file in the tree holds a CHE-48 result. The decomposition was never performed, so
CHE-47's proposed experiment (refit O2 above the frozen 256-ring linear
interpolation) is still the concrete next step. `manifest.yaml` and
`M3_SLICE_REPORT.md` were **correct and are unchanged** — nothing is promoted.
CHE-48 is reopened, and the `1.0e-3` gate is carried into M4 as an explicit open
limitation on benchmark #3. Item 3 of the disposition doc.

### L4 — ~~CHE-50's decision is not reflected in the repository~~ DISCHARGED (CHE-68)
CHE-50 closed with a decision ("no kernel change for now; revisit when a
propagation-sensitive benchmark requires it") — a legitimate outcome. But M3's
carry-forward #5 required *telling consumers*, and no coupler card, docstring,
or artifact field warned a caller that the emitted sensor field carries no
curvature term. The decision lived only in a Linear comment.

**Now stated in four places, one of which travels with the artifact:** the
`C_RAY_TO_WAVE` module docstring, the graph-edge wrapper's docstring,
`knowledge/couplers/ray_to_wave/` (`coupler_card.yaml`
`known_limitations.no_wavefront_curvature_term` and `conventions.md`), and
`provenance["validity"]` on every emitted `ComplexField` — so a downstream
consumer reads it off the field rather than having to have read a document. Two
tests pin it: one that the declaration is present and says what it must, one that
the reconstruction is still bit-identical, because "no kernel change" is a claim
that should be asserted rather than promised. The card also records why PB7's
silence is not clearance (F3), so the next reader cannot cite three agreeing PSF
routes as evidence the term is absent.

### L5 — PSF cross-method semantics are compared, not frozen with tolerances
PB8's own acceptance criterion asks for "frozen, cross-comparable semantics with
justified tolerances". CHE-58 delivered frozen, cross-comparable *semantics* —
declared frame, origin, normalization, explicit resampling, no hidden
manipulation — and deliberately no tolerances. Given F2, that was the right call:
a tolerance fitted against a non-independent oracle would have been circular.
But the criterion as written is not met. See R1.

### L6 — Off-axis remains thin
One off-axis field in y, on one system, at one wavelength. CHE-42 (field scan
past the admissible pitch), CHE-43 (off-axis in x, to close the symmetric-error
gap) both unstarted. PB7 adds a second system at 20° but does not change the
shape of this gap.

### L7 — The negative-control blind-spot audit is still not general
CHE-44 unstarted, as at M3 exit. M3 found three independent vacuous controls;
nothing in M3.5 re-asked the question for the new device/precision
configurations, which are exactly the kind of new configuration where a control
can quietly stop being able to fail.

### L8 — Carried forward unchanged
CHE-45 (`./run.sh` aborts in a git worktree), CHE-46 (finite-object OPL launch
reference untested), CHE-49 (`rings^-0.83` NA-excess exponent unexplained). All
Backlog, none touched.

### L9 — Vector/polarized fields, broadband, partial coherence, conformal coupling
Unverified, unchanged. M3.5 touched none of them. **GPU/TPU is no longer on this
list** — see R6.

### L10 — ~~This report is written against an unmerged stack~~ DISCHARGED (CHE-62)
Every M3.5 commit sits on one branch, unmerged. See R2. This is M3's L8
recurring one milestone later.

**Discharged:** CHE-62 landed the stack on `main`. See R2.

---

# What still has to happen before M4 opens

Ordered by what blocks the gate.

### R1 — ~~Reconcile PB7's scope with PB8's acceptance criterion, and close CHE-58~~ PARTLY DISCHARGED (CHE-62)
CHE-58 is still in **Backlog** while PB8 is blocked by it. Its work is complete
and its evidence is on disk. Two decisions needed: (a) mark CHE-58 Done; (b)
amend PB8's PB7 criterion to match what CHE-58 actually scoped — "frozen,
cross-comparable semantics, tolerances deferred" — and open an M4 ticket for the
tolerance, which per F2 must wait for a genuinely independent oracle.

**(a) done — and the sentence above is stale:** CHE-58 and CHE-59 are both
`Done`. **(b) still open:** the PSF tolerance has no ticket yet, and per F2 it
must not be set against an Optiland PSF route. It belongs to M4 benchmark #3,
which is where the independent-oracle requirement already lives (M4 carry-forward
#1); establishing a new PSF tolerance was explicitly out of CHE-62's scope.

### R2 — ~~Land M3.5 on `main`~~ DISCHARGED (CHE-62)
`main` is still at `ec55839` (M3 exit). All 23 M3.5 commits (CHE-52 → CHE-61)
are stacked on `chengjiazhou4802/che-61-pb4b-...`. The per-issue branches for
CHE-55, CHE-56, CHE-57 and CHE-60 exist locally but were never pushed, and
`che-54`'s branch tip is actually a CHE-56 commit. No PRs. An exit gate
asserting "PB1–PB7 done" against an unmerged stack is not a gate.

**Discharged:** CHE-62 merged the stack into `main`. `main` was a strict ancestor
of the branch tip (nothing to reconcile), so the whole M3.5 history landed in
order. The observations about the stray per-issue branches stand and are
cosmetic — `che-54`'s tip really is a CHE-56 commit — but the commits themselves
are all reachable from `main` now, so no work is stranded.

### R3 — ~~Commit PB7's code~~ DISCHARGED (CHE-62)
`benchmarks/probes/pb7_cooke_triplet_psf_comparison.py` (1983 lines) is
untracked, on the CHE-61 branch. `outputs/` is gitignored, so PB7's figures and
JSON are regenerable-only by design — but the written conclusion has been moved
to `benchmarks/PB7_COOKE_TRIPLET_PSF_REPORT.md` so it is not lost with them.

**Discharged** at `ee57e33`, together with this report and the PB7 report, which
were themselves untracked. The probe is committed unreformatted: `ruff` reports
findings on it and on the already-committed `m3r_sensor_handoff.py` alike, so
benchmark probes are not under the lint gate and reformatting would have been an
unrelated change. This also unblocked L2 — a clean worktree is what
`dirty_worktree: false` requires, and until this commit there wasn't one.

### R4 — ~~Decide what CHE-48 "Done" means~~ RESOLVED (CHE-62)
Either the residual was decomposed (then the artifact and the `manifest.yaml` /
`M3_SLICE_REPORT.md` gate language need updating), or it was closed as
superseded (then say so, and the `1.0e-3` gate carries into M4 as an open
limitation on benchmark #3). Right now the ticket and the repository disagree.
Same for CHE-51, canceled with no reason recorded — PB7's "N_f and NA not
separated here" is a sufficient reason, it just needs writing down.

**Resolved — it is the second branch.** CHE-48 was closed without the
decomposition being performed (no comment, no commit, no artifact; see L3), so the
`1.0e-3` gate carries into M4 as an open limitation on benchmark #3 and the
`manifest.yaml` / `M3_SLICE_REPORT.md` language is **unchanged because it was
already right**. CHE-48 is reopened so Linear matches the repository.

**CHE-51's rationale is now recorded**, and it is stronger than "not separated
here". CHE-51's own description makes the three-way comparison a *precondition*
("the benchmark reference itself has not yet been cross-validated … should be
completed first"). PB7 ran that comparison, and its outcome — F2, that `FFTPSF`
and `HuygensPSF` share one Wavefront/OPD front end — removes CHE-51's premise
rather than satisfying it: a scan built on those two would be circular
validation. Separating `N_f` from NA needs several systems/wavelengths against a
genuinely independent oracle, which is the same prerequisite M4 benchmark #3
already carries. Items 3 and 4 of the disposition doc.

### R5 — ~~Make CHE-50's decision visible to consumers~~ DISCHARGED (CHE-68)
One line in the `C_RAY_TO_WAVE` coupler card and/or the emitted artifact's
diagnostics: the reconstructed field carries no wavefront-curvature term, valid
at the handoff plane, not valid after further propagation. M4's hybrid
compositions are precisely the consumer this was meant to warn.

**Done, and in more than one line** — see L4 above for where. It is a
declaration, not a refusal: CHE-50 deliberately did not add one, so
`provenance["validity"]["further_propagation_verified"]` records that further
propagation is outside what CHE-24/CHE-38 verified rather than prohibiting it.

### R6 — ~~Apply the M4 scope note to the project description~~ DISCHARGED (CHE-68)
Reviewed against what M3.5 established, four things in the Linear project
description are now stale:

1. **"Supported Scope Entering M3 … CPU; float64/complex128 reference numerics"**
   → now also CUDA GPU, and float32/complex64 with a measured per-subsystem
   verdict and explicit refusals.
2. **"Still unverified: … GPU/TPU"** → GPU is verified (CHE-60/CHE-61); TPU is
   not, and untested.
3. **"Deferred Beyond v0.1: … GPU/TPU"** → same; GPU should move out of this
   list.
4. **M4 benchmark #3 ("Real Lens → Chromatix → PSF")** should inherit PB7
   directly: reuse the canonical Cooke Triplet spec and the declared common-grid
   comparison, and add the requirement F2 forces — **an oracle that is not
   another Optiland PSF method**. Without that, benchmark #3's tolerance would be
   fitted against a route that shares a front end with the thing under test.

**All four applied (CHE-68).** Items 1–3: the scope section is retitled
"Supported Scope Entering M4" and now carries CUDA GPU as established with its
evidence, `float32`/`complex64` with the measured per-subsystem verdict and the
explicit refusals, and the CHE-60 caveat that the `gpu`-marked suite has not been
re-run; TPU is the only accelerator left under "Still unverified" and under
"Deferred Beyond v0.1". Item 4: benchmark #3 gained two subsections — what it
inherits from PB7, and the independent-oracle requirement stated as a hard
prerequisite before any tolerance is *defined or finalized*, plus the `FFTPSF`
per-axis mis-scale and the CHE-50 curvature term as things not to re-learn the
hard way. **No tolerance was set**, per F2.

The knowledge-asset summary cards were reviewed in the same pass and no longer
claim "no GPU/accelerator in this container": both solver cards, both coupler
cards, `knowledge/solver_cards/{optiland,chromatix}.yaml` and
`knowledge/solvers/optiland/capability_notes.md`. Card pointers to tests CHE-67
archived are now labelled as archived and unguarded rather than reading as live
evidence.

Optional, cosmetic: the milestone "M3.5 — Pre-Benchmark Stabilization" collides
by name with issue CHE-34, "M3.5 — Make C_RAY_TO_WAVE executable as a graph
edge". Untouched by CHE-68.

**~~Still stale in the project description, and deliberately left for its own
ticket~~ DISCHARGED (CHE-69):** the "Current Highest-Priority Blocker" section
named Optiland `opd_native` characterization as the blocker on the forward
pipeline. CHE-30 (M3.1) established its sign, units, reference and behaviour
under known propagation against manufactured geometries, CHE-41 closed the
off-axis half, and M3 exited on that evidence. This was outside CHE-68's stated
scope, so CHE-68 reported it and CHE-69 rewrote it.

The replacement says the blocker changed *shape* rather than disappeared:
nothing blocks execution, and what blocks M4's benchmark claims is the **absence
of an independent PSF oracle** (F2 above). CHE-69 also corrected hazard H1 in
`knowledge/couplers/ray_to_wave/conventions.md`, which still read as though
`opd_native` were uncharacterized. Documentation only.

---

# What M4 should carry forward

1. **Do not use an Optiland PSF as the oracle for the ray→wave path.** F2. Get
   an analytic case, a conservation law, or a genuinely independent
   implementation before any tolerance is set.
2. **`FFTPSF` is mis-scaled off axis by a known, predictable per-axis factor.**
   Confirm upstream before treating it as a reference anywhere, and never
   silently correct it.
3. **The gradient boundary stays `forward_only`.** Unchanged from M3's #3. A
   custom derivative plus a directional finite-difference test comes *before*
   any promotion.
4. **The missing curvature term becomes observable the first time M4 propagates
   a reconstructed sensor field.** Expect it; instrument for it.
5. **Ask "can this control actually fail here?" for every new device/precision
   configuration**, not only for new physics. M3.5 added a large new
   configuration space and did not re-ask.
6. **Pick up CHE-42, CHE-43, CHE-44, CHE-46 rather than re-deriving their
   scope.** All are scoped and unstarted.
7. **Discharge L1 and L2** (the missing probe record, the dirty-tree provenance).
   Both have now survived two milestone exits. **Now tracked as CHE-63** with
   the exact commands, rather than as a report footnote — which is how they
   survived the first two. CHE-63 also requires committing `L2-PSF-01`'s
   provenance out of gitignored `outputs/`, because non-durable evidence is the
   mechanism that let L2 recur.
8. **The `1.0e-3` physical-correctness gate is inherited unmet, not silently
   resolved.** CHE-48 was closed without decomposing the residual (L3/R4). M4
   benchmark #3 owns it, and per F2 must not close it against another Optiland
   PSF method. Read `benchmarks/M3_M3_5_CLEANUP_DISPOSITION.md` before
   re-litigating any of the five M3/M3.5 items it covers.
