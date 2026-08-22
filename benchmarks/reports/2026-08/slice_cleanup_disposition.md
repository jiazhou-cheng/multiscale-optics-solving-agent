# M3 / M3.5 cleanup — dispositions


> **Evidence:** `outputs/…` paths below are **local-only** — that directory is
> gitignored and exists on the machine that produced this run, not in a clone.
> Committed records live in `benchmarks/probes/records/`. See
> [`benchmarks/reports/README.md`](../README.md#where-the-evidence-actually-is).

CHE-62. Bookkeeping and evidence reconciliation only. **No new physics claim, no
new tolerance, no gate change, no benchmark scope change is made anywhere in this
ticket.** `derivative.verified` stays `false` everywhere it is declared.

This file is the single place that records what each outstanding M3/M3.5 exit
item was decided to *be*. Where a milestone report, `benchmarks/manifest.yaml`,
`benchmarks/protocols/slice_protocol.yaml`, a coupler knowledge asset, or a Linear ticket
touches one of these five items, it points here rather than restating it — so
there is one statement to change, not seven that can drift apart.

## Audit tree

| Item | Value |
|---|---|
| Tree audited | `ee57e33`, branch `chengjiazhou4802/che-61-pb4b-unified-precisiondtypedevice-contract-and-cross-model` |
| `main` before this ticket | `ec55839` (M3 exit) |
| Worktree state | clean at `ee57e33` — see item 2 |
| Tier A measured at `ee57e33` | **670 passed, 54 skipped, 173 deselected, 42.78 s** |
| Environment | `agent_solver` container via `./run.sh`; Python 3.12.13, Linux 6.8.0-84, glibc 2.41; optiland 0.6.0, chromatix 0.6.0 @ d24bdf0, jax 0.6.2, numpy 2.2.6, scipy 1.15.3 |
| Compute re-runs | **none** — deliberate, see items 1 and 2 |

Tier A's 54 skips decompose exactly as:

| count | file | reason |
|---|---|---|
| 21 | `tests/test_m3r_sensor_handoff.py` | CHE-38's consolidated record absent — item 1 |
| 8 | `tests/test_gpu_environment.py` | `gpu` marker quarantine (CHE-60) |
| 19 | `tests/test_precision_gpu_pipeline.py` | `gpu` marker quarantine (CHE-60/61) |
| 6 | `tests/test_precision_execution_matrix.py` | `gpu` marker quarantine (CHE-61) |

The 33 `gpu` skips are the designed behavior of the quarantine: enabling the GPU
mutates process-global JAX state, so those tests run only in
`./run.sh --gpu pytest -q -m gpu`. They are not an M3/M3.5 debt.

---

## Item 1 — CHE-38's consolidated probe record

**Finding.** `benchmarks/probes/records/m3r_sensor_handoff.json` does not exist
and never has. Three prior generation attempts failed: two completed every
measurement and then died in post-processing (a protocol-key read and a
figure-label `KeyError` — both fixed, and the driver now persists the record
*before* plotting so that class of failure cannot discard a run again), and a
third was stopped mid-run by the GPU-server resource policy.

**Disposition — deliberately not regenerated in CHE-62, and NOT declared
unnecessary.** The run is ~25 min of foreground container compute; this ticket is
bookkeeping-only and does not execute benchmarks. The record is still required by
CHE-38's own acceptance criteria. It is tracked in **CHE-63**, with the exact
command.

**Why the 21 skips are justified rather than discharged.**

*What actually depends on the record:* only the 21 record-backed assertions in
`tests/test_m3r_sensor_handoff.py`. Nothing else in the repository reads it. The
6 live-machinery tests in that same file pass.

*What does not depend on it, contrary to how the pointers read:*

- `L2-PSF-01` loads `benchmarks/probes/sensor_handoff_convergence.py` **as a module** and
  re-runs it, precisely so the bundle and the probe cannot silently disagree
  about what "the sensor plane" or "the gate" means. It never reads the JSON.
- `benchmarks/reports/2026-08/sensor_handoff_convergence.md` is evidence-complete from the
  staged runs tabulated in its §11, measured in-container through `./run.sh`.
  It is *not* reproducible from one committed artifact, and says so.
- No correctness gate in `benchmarks/manifest.yaml` depends on it.

*Why skipping is the correct behavior and not a hidden failure:* the assertions
have no record to read. They are not `xfail` — nothing is expected to fail — and
they are not deleted, because they are exactly the single-artifact
reproducibility check CHE-38 asked for. A skip that names its cause is the honest
encoding of "this evidence has not been generated yet".

*What changed here:* the skip message is now self-documenting. It names CHE-38,
CHE-62 and CHE-63 and prints the regeneration command, so the skip count no
longer depends on an unexplained missing artifact — reading the test output is
enough to know what is absent, why, and how to produce it.

**Carried into M4.** CHE-63 item 1. Separately, CHE-38 acceptance criterion 17
("full regression tests pass") was never run for that ticket and remains
unverified — Tier C is the check.

---

## Item 2 — `L2-PSF-01`'s dirty-tree provenance

**Finding.** `outputs/M3/L2-PSF-01/provenance.json` records
`dirty_worktree: true`, `git_commit a69fe6d9c9af72d4181c0eaecf0fec865a9b03f0`,
`timestamp_utc 2026-08-19T04:03:05Z`. The field is `bool(git status --porcelain)`,
so any untracked file sets it. This is M3's carry-forward #7 and M3.5's L2; it
has survived two milestone exits.

**Disposition — deliberately not regenerated in CHE-62.** Same reason as item 1:
this ticket does not execute benchmarks. Tracked in **CHE-63** item 2.

**What did change: the precondition is satisfiable for the first time.** Until
`ee57e33`, PB7's probe and two reports were untracked, so `dirty_worktree: false`
was *unreachable* — a re-run could not have produced clean provenance no matter
when it was launched. As of `ee57e33` the worktree is clean, so the re-run in
CHE-63 can now succeed on the criterion it is meant to satisfy.

**A durability problem that must be fixed with the re-run, or this recurs a third
time.** `outputs/` is gitignored, so even a clean re-run's provenance is not
durable evidence — which is plausibly how the dirty record survived two exits in
the first place. CHE-63 requires committing the provenance into a tracked path.

**Carried into M4.** CHE-63 item 2. The re-run must confirm `dirty_worktree:
false` *and* an unchanged scientific fingerprint (`eea43cf441c418fb...`); a
changed fingerprint is a finding, not a formality.

---

## Item 3 — CHE-48 and the `1.0e-3` physical-correctness gate

**Finding — the ticket was wrong, not the repository.** CHE-48 was marked `Done`
at `2026-08-19T22:14:37Z` with no comment, no commit, and no artifact.
`git log --all --grep=CHE-48` returns only `ec55839`, which *mentions* CHE-48 as a
follow-up rather than resolving it. No file in the tree contains a CHE-48 result;
the four files that name it — `reports/2026-08/ray_to_wave_slice.md`, `reports/2026-08/ray_to_wave_slice_exit.md`,
`reports/2026-08/cooke_triplet_psf_routes.md`, `pb7_cooke_triplet_psf_comparison.py` — all
describe it as open.

**Disposition: CHE-48 was closed without the decomposition being performed.**
The unattributed half of the sensor-plane residual is still unattributed. CHE-47's
own proposed experiment — refit the O2 oracle at higher ring resolution and/or a
higher-order interpolation than the frozen 256-ring linear fit, and see whether
the gap to O1 closes — has not been run.

Consequently:

- The frozen `fft_oracle_intensity_relative_l2 = 1.0e-3` gate remains **unmet**
  on the real traced `M3-SINGLET-REF` system, at `2.2e-3`–`2.5e-3` at 787,969
  rays with CHE-47's production per-ray quadrature weight applied. A synthetic
  aberration-free bundle reaches a converged `4.07e-4`, inside the gate.
- `benchmarks/manifest.yaml` and `benchmarks/reports/2026-08/ray_to_wave_slice.md` keep their "gate
  not met" language unchanged. It was never stale — it was correct, and only the
  ticket state contradicted it. **Nothing is promoted here.**
- CHE-48 is reopened in Linear so the tracker matches the repository.

**Carried into M4 as an explicit open limitation.** The sensor-side
`C_RAY_TO_WAVE` handoff is *verified* (CHE-38 verdict A: discretization-converged,
no floor, no structural defect, within its declared validity region). What is not
verified is the graph's absolute physical accuracy on the real traced system
against the frozen gate. M4 benchmark #3 inherits this unmet gate, and per PB7's
finding F2 must not close it against another Optiland PSF route.

---

## Item 4 — CHE-51's cancellation

**Finding.** CHE-51 was canceled at `2026-08-19T22:19:34Z` with no rationale
recorded anywhere.

**Disposition: the cancellation is correct, on two grounds, now written down.**

1. **Its own stated precondition was executed.** CHE-51's description says the
   attribution "is premature because the benchmark reference itself has not yet
   been cross-validated. A direct three-way comparison between Optiland FFT PSF,
   Optiland Huygens PSF, and the sensor-plane ray-to-wave PSF should be completed
   first." PB7 (CHE-58) ran exactly that comparison, on the canonical Cooke
   Triplet at λ = 0.55 µm, on-axis and 20°.

2. **PB7's outcome removes CHE-51's premise rather than satisfying it.** PB7
   finding F2: Optiland's `FFTPSF` and `HuygensPSF` are two implementations
   *inside one package*, sharing the same Wavefront/OPD front end, reference
   sphere, launch-tilt removal and pupil sampling. They are not independent, so
   the A-vs-B residual understates the Optiland pair's uncertainty and the pair
   cannot cross-validate the reference. An `N_f`-vs-NA scan built on them would
   be circular validation.

**What PB7 explicitly did not do, and this ticket does not claim.** PB7 did not
separate `N_f` from NA: one system, one wavelength means the two are not
independently varied. That absence is recorded in
`benchmarks/reports/2026-08/cooke_triplet_psf_routes.md` ("Relation to known open issues")
and in `benchmarks/reports/2026-08/ray_to_wave_slice_exit.md`. Adopting the PB7 result as sufficient
grounds for cancellation is *not* a claim that the confounding was resolved — it
is a statement that the experiment CHE-51 proposed cannot answer the question
with the oracles available.

**Carried into M4.** Separating `N_f` from NA needs several systems and/or
wavelengths measured against a genuinely independent oracle — not another
Optiland PSF route. That is the same prerequisite M4 benchmark #3 already carries
from F2, so it is not tracked as a second ticket.

---

## Item 5 — milestone gate audit

Every M3/M3.5 exit claim touched by items 1–4, re-checked at `ee57e33`.

| Claim | Where | Verdict |
|---|---|---|
| `L2-PSF-01` does not meet the `1.0e-3` gate (`2.2e-3`–`2.5e-3`) | `manifest.yaml`, `reports/2026-08/ray_to_wave_slice.md`, `reports/2026-08/ray_to_wave_slice_exit.md` | **Stands, unchanged.** Item 3 confirms it; only the ticket state was wrong |
| Sensor-side `C_RAY_TO_WAVE` handoff verified (CHE-38 verdict A) | `reports/2026-08/sensor_handoff_convergence.md` | Stands. Untouched by this ticket |
| CHE-38's consolidated record has never landed | `reports/2026-08/ray_to_wave_slice.md` L5, `reports/2026-08/ray_to_wave_slice_exit.md` L1 | **Stands.** Now justified rather than merely noted — item 1 |
| `L2-PSF-01` provenance is from a dirty tree | `reports/2026-08/ray_to_wave_slice.md` L8, `reports/2026-08/ray_to_wave_slice_exit.md` L2 | **Stands.** Precondition now satisfiable — item 2 |
| "CHE-48 is Done but the repository says the gate is unmet; one of the two is wrong" | `reports/2026-08/ray_to_wave_slice_exit.md` L3 | **Resolved.** The ticket was wrong — item 3 |
| Tier A is "30.85 s, 478 passed / 21 skipped / 128 deselected" | `reports/2026-08/ray_to_wave_slice_exit.md` PB1–PB3 table | **Stale as a present-tense claim.** True of PB3's tree; re-measured 42.78 s, 670 passed / 54 skipped / 173 deselected at `ee57e33`. The 21 record skips are unchanged; the other 33 are CHE-60/61's `gpu` quarantine, which did not exist when PB3 measured |
| "CHE-58 is still in Backlog" | `reports/2026-08/ray_to_wave_slice_exit.md` R1 | **Stale.** CHE-58 and CHE-59 are both `Done` |
| "PB7's code is untracked" | `reports/2026-08/ray_to_wave_slice_exit.md` R3 | **Discharged** at `ee57e33` |
| "M3.5 sits on an unmerged stack; `main` is at `ec55839`" | `reports/2026-08/ray_to_wave_slice_exit.md` L10, R2 | **Discharged** by this ticket's merge |
| CHE-51 canceled with no reason recorded | `reports/2026-08/ray_to_wave_slice_exit.md` R4 | **Resolved** — item 4 |
| No new physics claim in M3.5 | `reports/2026-08/ray_to_wave_slice_exit.md` claim audit | Stands. CHE-62 adds none either |

**Not in CHE-62's scope, and still open** — recorded here so they are not lost by
being absent from this ticket:

- **R5** — making CHE-50's decision (the reconstructed sensor field carries no
  `exp(ikr²/2R)` term) visible to consumers in the coupler card or the emitted
  artifact. Still lives only in a Linear comment. CHE-50 is explicitly out of
  CHE-62's scope.
- **R6** — the four stale items in the Linear project description (GPU moving out
  of "unverified" and "deferred", the entering-scope precision line, and M4
  benchmark #3 inheriting PB7 plus F2's independent-oracle requirement).
- **L5–L9** of `reports/2026-08/ray_to_wave_slice_exit.md` — PSF tolerances deferred (correctly, per
  F2), off-axis thinness (CHE-42, CHE-43), the negative-control blind-spot audit
  (CHE-44), and CHE-45/CHE-46/CHE-49. All scoped, none touched here.
