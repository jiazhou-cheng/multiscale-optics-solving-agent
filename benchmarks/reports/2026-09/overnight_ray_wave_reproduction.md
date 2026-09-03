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
| A — ray tutorial / system regression | not yet run | §5 |
| B — wave kernel sweep | not yet run | §6 |
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

*Not yet run.*

## 6. Workstream B — wave kernel sweep (CHE-240)

*Not yet run.*

## 7. Workstream C — Demo2 reproduction (CHE-241)

*Not yet run.*

## 8. Workstream D — Demo3 characterization (CHE-242)

*Not yet run.*

## 9. Code changes, verification gates, and what was not run

*Accumulated as the night proceeds.*

### 9.1 Code changes so far

None beyond this report and the per-workstream harnesses recorded in §5–§8.

### 9.2 Gates run so far

None. §3.2 is measurement, not a gate.

### 9.3 Resource incidents so far

None. Swap at 0 B at preflight.

## 10. Follow-up tickets recommended

*Accumulated as the night proceeds.*

* Attach `optiland_notebook_link_index.csv` to CHE-239, or drop the reference —
  see §2.2.
* CHE-238's own text says the catalog has 15 descriptors; it has 17. Cosmetic,
  recorded in §2.1, not corrected in the ticket (the ticket is not to be edited).
