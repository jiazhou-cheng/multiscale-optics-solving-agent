# Cleanup baseline — the numbers every later phase is judged against

Frozen at commit `b5fbe42` on 2026-08-22, on a clean working tree, before any
deletion, move, or rename in the repository-wide cleanup.

Its purpose is narrow: make a regression attributable. Before this record,
AGENTS.md's figures came from three different image states, so "did this phase
break something" had no single answer to compare against. Everything below was
measured in one sitting, on one image, at one commit.

## Tree state

| | |
| -- | -- |
| Commit | `b5fbe42` |
| Branch | `main` |
| `git status` | empty |
| Host | shared 8×RTX A6000 box, 377 GiB RAM, driver 550.163.01 |
| CPU image | `agent_solver`, rebuilt (cached) at the start of this session, `sha256:c89704eb8079` |
| GPU image | `agent_solver_gpu`, reused, `sha256:6dd3dfb6d1cd` |

Two uncommitted things were resolved first, in `b5fbe42`, so the baseline had a
tree to describe:

* `docs/architecture/repository_cleanup_plan.md` — the archived prior epic's
  draft, superseded by the current audit. Nothing references it. Committed as
  deleted.
* The two paper phase masks (`demo2_smile_phase_profile.npy`,
  `demo3_smile_phase_profile.npy`) — untracked at the repository root. Moved to
  `benchmarks/probes/data/` and committed as reference data.

## Measured gates

| Command | Image | Result | Wall clock |
| -- | -- | -- | -- |
| `./run.sh --rebuild pytest -q` | rebuilt | **769 passed, 48 skipped** | **183.00 s** |
| `./run.sh pytest -q -m "not slow"` | reused | **751 passed, 48 skipped, 18 deselected** | **36.56 s** |
| `./run.sh python scripts/check_context_sync.py` | reused | pass (303 files checked) | < 5 s |
| `./run.sh python scripts/validate_package.py` | reused | pass — 7 models, 10 couplers, all YAML, all example graphs | < 5 s |
| `./run.sh python scripts/export_schemas.py` | reused | **regenerates a diff — see Finding 1** | < 5 s |
| `./run.sh --gpu pytest -q -m gpu` (own session, `MOA_GPUS=device=0`) | reused | **48 passed, 769 deselected** | **69.68 s** |
| `./run.sh pytest -q benchmarks_agent` | reused | **52 passed** | 7.51 s |
| `make test-agent-benchmark` | — | **fails on the host — see Finding 2** | — |
| `make test-tutorial` | — | deliberately skipped — see below | — |

All four suite results reproduce AGENTS.md's recorded figures exactly
(769/48/182 s, 751/48/37 s, 48/70 s, 52/8 s), so the documented numbers are
confirmed rather than merely inherited.

`make test-tutorial` (60 tests, ~33 min) was skipped deliberately: nothing in
this phase touches a pin or `docker/`. It also exceeds the 10-minute command
timeout, so when a later phase requires it — Phase 4 and Phase 7 both do, since
they move the paths it loads — it needs a detached run, not a chunked one.

## Resource envelope

Host swap sits at **266 MiB at rest** on this box and never moved during any of
the runs above; peak was 266 MiB throughout. Minimum available RAM across the
full suite was 355 GiB of 377 GiB. Every command was run under a watchdog that
polls `/proc/meminfo` and kills the process group if swap-used grows more than
256 MiB above its value at launch. It never fired.

Because host swap is non-zero at rest, *growth* is the signal, not level — the
same distinction `docs/archive/2026-08-testing/test_runtime_audit.md` records for the container
cgroup guardrail.

## Per-module coverage map

Measured with `./run.sh pytest -q --cov=multiscale_optics_agent`, i.e. the
default active suite only. Phase 3 has to decide "test or archive" for the
unguarded modules; this is the map that decision is checked against.

**Zero coverage from the default suite (6 modules, 534 statements):**

| Module | Statements | Note |
| -- | -- | -- |
| `evaluation/psf_oracles.py` | 227 | the independent Airy / Richards-Wolf oracles |
| `evaluation/m1_bundle.py` | 135 | gen1 machinery **plus** the two fingerprint helpers L2-PSF-01 and L2-COUPLER-01 still import |
| `evaluation/psf_measurement.py` | 92 | named as the terminal measurement by `registry/couplers.yaml` and `examples/graphs/ray_to_wave.yaml` |
| `cli.py` | 39 | the packaged console script |
| `evaluation/checks.py` | 21 | no importer anywhere; Phase 1 deletes it |
| `core/provenance.py` | 20 | consumed only by `scripts/export_schemas.py`, which the suite does not run |

`core/provenance.py` at 0% is worth flagging beyond the audit's list: Phase 3
promotes the fingerprint helpers *into* it, so it goes from unguarded to
load-bearing and must gain a test in the same phase.

**Under 50% (7 modules)** — the four out-of-scope or gen1 ones are retired by
Phases 2-3, which is most of the deficit:

| Module | Statements | Coverage | Disposition |
| -- | -- | -- | -- |
| `adapters/fmmax_adapter.py` | 172 | 16% | Phase 2 deletes |
| `benchmarks/metalens_candidate.py` | 254 | 16% | keep; `studies/metalens/` in Phase 5 |
| `adapters/fdtdx_adapter.py` | 168 | 22% | Phase 2 deletes |
| `adapters/optiland_benchmark_adapter.py` | 176 | 36% | Phase 3 archives |
| `adapters/chromatix_benchmark_adapter.py` | 39 | 41% | Phase 3 archives |
| `adapters/chromatix_scaling_adapter.py` | 100 | 42% | Phase 3 archives |
| `benchmarks/metalens_controller.py` | 663 | 48% | keep; `studies/metalens/` in Phase 5 |

**Totals:** 7,337 statements, **68.8%** covered. The full per-module table is
reproducible with the command above; the JSON lands in `tmp_probes/`, which is
gitignored scratch by design.

The two solver adapters that Phase 6 splits sit at 87% (`optiland_adapter.py`,
509 statements) and 72% (`chromatix_adapter.py`, 567 statements). Those are the
figures the characterization suite has to hold or improve — a split that drops
either is a split that moved behavior.

## Findings

Phase 0 changes nothing, so these are recorded rather than fixed. Each names
the phase that owns it.

### Finding 1 — `schemas/` is already stale at HEAD

`scripts/export_schemas.py` regenerates a real diff against the committed
files: `coupler.schema.json` gains a `Device` enum plus `devices`, `dtypes` and
`pinned_commit`; `model.schema.json` gains `pinned_commit`. The pydantic models
grew those fields and the generated artifacts were never regenerated.

This is pre-existing drift, not something this phase caused, and it is exactly
what Phase 8's drift test exists to catch. It is fixed at the top of **Phase 1**
rather than at Phase 8, because otherwise every intervening phase's
`export_schemas` gate is noisy and a *new* drift would hide inside a diff that
is already non-empty.

The regenerated files were reverted here so the baseline tree stays clean.

### Finding 2 — `make test-agent-benchmark` does not run at all

The target is bare `pytest -q benchmarks_agent`, which executes on the **host**,
where the dependencies do not exist; it aborts with a collection error in 0.11 s.
The suite itself is healthy — `./run.sh pytest -q benchmarks_agent` is 52 passed
in 7.51 s.

This is Phase 8 Part E, and it is worse than the "contradicts AGENTS.md"
framing: the target is not merely policy-violating, it is non-functional. Every
`make` target that invokes bare `pytest` or `python` has the same problem.

Consequence for this epic: **do not use `make` targets as gates.** Use the
`./run.sh` forms until Phase 8 fixes the Makefile.

### Finding 3 — GPU containers could not start; the cause is host hardware

`./run.sh --gpu` failed at the container runtime, before any test:

```
nvidia-container-cli: detection error: nvml error: unknown error
```

GPU 5 (`0000:B2:00.0`) is in a fault state at the NVML level: `nvidia-smi -L`
lists the other seven and fails on that one. `nvidia-container-cli`'s prestart
hook enumerates *every* GPU before deciding which to expose, so one faulted
device blocks GPU containers entirely, however healthy the requested device is.
Recovering it needs a GPU reset or a reboot — root on a shared host, and out of
bounds for this work.

`run.sh` gained a fallback rather than the epic losing its GPU gate. When
`nvidia-container-cli info` fails, `--gpu` binds the requested `/dev/nvidia<N>`
nodes and the host's userspace driver libraries directly, skipping enumeration.
It is strictly narrower than `--gpus` — only the devices named by `MOA_GPUS`
become visible — changes no host state, and works unprivileged, which matters
because the container runs as the invoking user and so cannot create symlinks or
run `ldconfig`; each library is bind-mounted straight onto its SONAME instead.

Verified end to end on device 0: `nvidia-smi -L`, `jax.devices() ->
[CudaDevice(id=0)]`, real matmul kernels, `torch.cuda.is_available() -> True`,
and then the full `-m gpu` suite at 48 passed in 69.68 s, matching the
CHE-72/CHE-73 record of 48 passed in 70 s.

`MOA_GPU_PASSTHROUGH=0` forces the stock `--gpus` path back on, so once the host
is repaired nothing has to be un-done.

### Finding 4 — the working tree was not clean, and one deletion was undecided

Recorded for completeness; resolved in `b5fbe42` as described above. The
`tmp_probes/` scratch (3.6 MB) and the root `.DS_Store` are gitignored, so they
did not block a clean `git status`; they are Phase 1's to remove.

## What the later phases must reproduce

* **769 passed, 48 skipped** on `./run.sh pytest -q`, adjusted only by counts a
  phase explicitly predicts. Phase 2 predicts a drop of about one per removed
  adapter, from the parametrized adapter-discovery test.
* **48 passed** on `./run.sh --gpu pytest -q -m gpu`, in its own session.
* **52 passed** on `./run.sh pytest -q benchmarks_agent`.
* Zero swap growth. If a phase's run pushes the host into swap, stop it and
  reduce the work — do not record the number and move on.
