# CHE-115 (M3.3) — the substrate proof: the frozen number, off the executor

**Issue:** CHE-115
**Date:** 2026-08-27
Every number below was measured in this issue, on the `agent_solver` CPU image.
Commands are quoted verbatim.

## The claim, and how it is checked

`B3-PSF-SINGLET`'s gate metric `fft_oracle_intensity_relative_l2` is frozen at
`0.0022072391812867093` against a `1.0e-3` threshold. That value was produced
before any executor existed, by `benchmarks/probes/quadrature_weight.py::characterize()`
calling `benchmarks/probes/sensor_handoff_convergence.py`'s helpers directly —
trace, declare, advance, reconstruct, mask, divide.

It now comes off a `GraphExecutor` record produced from a committed YAML
document, as **the same float64**:

```
./run.sh python benchmarks/instances/b3_psf_singlet.py --write

  fft_oracle_intensity_relative_l2: 0.0022072391812867093  tol 0.001  met=False
  frozen 0.0022072391812867093  bit_identical=True
```

`tests/test_substrate_proof.py::test_the_frozen_number_is_reproduced_bit_identically`
asserts `==`, not `approx`. Reproducing a frozen number to a tolerance would be
a migration that moved a number and called it a migration.

Graph: `examples/graphs/psf_singlet_sensor.yaml`
Driver: `benchmarks/instances/b3_psf_singlet.py`
Record: `benchmarks/instances/records/B3-PSF-SINGLET-01.json`
Scientific fingerprint: `7f90cc04533ca976bb124f132aebfc2c34949019cbbb667be51740ef36b2eb2e`

## Two things had to be fixed, and one of them would have been easy to bless

**1. A graph could not name the plane the field is reconstructed at.**
`handoff_plane` declares where the optical path length is *referenced from*. The
plane the field lands on was whatever the adapter exported, and the adapter
resolves only `image_surface` and `exit_pupil` — so the sensor-plane handoff
CHE-38 (M3.9R) established as the intended contract of `C_RAY_TO_WAVE` could not
be written as a graph document at all. That is why the frozen configuration lived
in a probe.

`config['advance_to_z_m']` on the edge is that plane, implemented by
`couplers/handoff.py::advance_bundle_to_plane`: each ray advanced along its own
direction by `s = (z - z0) / d_z`, optical path grown by `n s` with `n = 1`,
directions unchanged. The resulting per-ray constant phase differs from the
original by `k s d_z²`, the phase an exact plane wave accumulates over the plane
offset `s d_z` — so this does not approximate the field at the new plane, it
evaluates the same 3-D superposition there.

`sensor_handoff_convergence.py::_advance_bundle_to_z` is **not edited**: editing
it would restamp records it did not produce. So there are two copies, pinned
equal on the real traced M3-SINGLET-REF bundle by
`tests/test_ray_to_wave_node.py::test_the_promoted_advance_equals_the_probes_own_copy`
— positions, directions, OPL, plane and reference string, all `array_equal`,
rather than assumed equal because one was written from the other.

**2. The measurement normalized twice, and it cost exactly one ULP.**
Reading the PSF with `PsfNormalization.PEAK` and handing it to
`disc_relative_l2_intensity`, which peak-normalizes its own inputs, divides by
the peak twice. Measured at 256 rings:

| path | value |
|---|---|
| frozen (probe) | `0.002203572853045589` |
| graph, `PsfNormalization.PEAK` | `0.0022035728530455903` |
| graph, `PsfNormalization.RAW` | `0.002203572853045589` |

A 5.9e-16 relative difference. It would have been entirely comfortable to call
that round-off and re-record the number, and the diagnosis is the reason not to:
with `RAW`, the intensity arrays off the two paths are `np.array_equal` — not
close, **identical** — so there was never any round-off to accept. The metric is
defined on `|u|²` and does its own normalization; the double division was a defect
in the measurement call.

## The wave node, and what it costs

In the frozen configuration the handoff is exactly **on** the sensor, so the
required post-handoff propagation is zero and `M_WAVE_CHROMATIX` is a complex64
round trip. The frozen gate is therefore defined on the float64 sensor-plane
field, which is the coupler's output — and that is what the driver measures.

The round trip is not free, and it is the gate metric's reported error bar rather
than a footnote:

| artifact | `fft_oracle_intensity_relative_l2` |
|---|---|
| `sensor_reconstruction:input_field` (float64, the frozen definition) | `0.0022072391812867093` |
| `wave:output_field` (complex64, zero-distance round trip) | `0.0022070734366051404` |

7.5e-5 relative, entirely Chromatix's unconditional complex64 cast. Reporting
only the terminal number would have moved the frozen gate by more than round-off
with no statement of why.

Because a zero-distance node demonstrates two nodes rather than three, the same
document is run with the reconstruction moved 0.001 R upstream — CHE-38's own
`near_sensor_fine` candidate, its Experiment D plane — through
`runtime.variants.with_config_overrides`. One key changes; the wave node reads
`target_plane_z_m` and derives its distance from the input field's own plane, so
the propagation follows:

```
./run.sh python benchmarks/instances/b3_psf_singlet.py --near-sensor-fine

advance_to_z_m                       0.004900767298924901
propagation_m                        4.8374613003093064e-06
reconstruction_relative_l2_vs_o1     0.0020376816972012836
terminal_relative_l2_vs_o1           0.0022072792393057067
wave_node_actual_device              cpu
```

The reconstruction 4.84 µm upstream reads `2.0377e-3`; after a real
angular-spectrum propagation to the sensor it reads `2.20728e-3`, which is
**1.8e-5 relative** from the frozen `2.20724e-3`. The wave leg lands on the
frozen number from a different plane. This is reported, not gated: CHE-38 section
7's padding sweep is the evidence for how much a residual post-handoff
propagation moves this number, and re-deriving that sweep is not this issue's
job.

## The negative control, as a graph rather than as a script

The family declares `opl-sign-flip` and until now the only way to run it was a
driver building the bundle itself, so `verify()` reported it `NOT_RUN` —
correctly. It is now `config['perturbation'] = {'opl_sign': -1}` on the edge, run
as its own graph with its own `graph_fingerprint`:

```
control opl-sign-flip: fired
  baseline 0.00220724 -> mutated 3.85714 against a gate of 0.001;
  detection margin 1747.5x
```

Decided against O1 (the analytic Airy pattern) only. O2, our own ASM/Rayleigh-
Sommerfeld propagator, is built from the same traced pupil and never decides
pass/fail.

An unknown perturbation key is **refused**, not dropped
(`couplers/node.py::_perturbation`), and an override naming a node or edge the
graph does not have raises `VariantError`. A control silently ignored for a typo
produces a run identical to the unperturbed one that reports itself as a control,
which is the worst outcome available here.

Three of the family's four controls remain `NOT_RUN`, and the result therefore
still reports `gate_is_trustworthy: False`. Two of them (`axis-transpose`,
`launch-phase-error`) are identities on an on-axis rotationally symmetric pattern
and the family says so. The third, `inverted-quadrature-weight`, is the one
CHE-117 found fires backwards for reasons in the control rather than in the
weight; running it is another 512-ring pass and was not required here.

## Framework overhead (S5)

`./run.sh python benchmarks/perf/run_baselines.py overhead` →
`benchmarks/perf/records/framework_overhead.json`. The `executor` arm is new in
this issue; CHE-105 could not measure it because there was no executor.

| arm | framework | direct | ratio | ≤10%? |
|---|---:|---:|---:|:--:|
| solver (`adapter.run` vs `build + trace`) | 0.0200 s | 0.00822 s | **2.436** | no |
| coupler (`node.transform` vs `ray_to_wave`) | 0.4608 s | 0.4535 s | **1.016** | yes |
| **executor** (`GraphExecutor.run` vs the three calls by hand) | 1.3701 s | 1.3156 s | **1.041** | **yes** |

The executor arm is 128 rings / 49,537 rays / 256² grid, and that choice is the
honest part: at the frozen 512 rings the reconstruction alone is ~17 s and any
fixed per-run cost rounds to a ratio of 1.00, which would prove nothing about the
layer. **1.041 is an upper bound on what the frozen configuration pays, not an
estimate of it.** Both arms run the same three stages in the same order and write
the same artifacts; the 54 ms difference is `GraphValidator`, `topological_order`,
the per-node `CostEstimate`, the `MemoryWatchdog`, artifact keying, the
`SolverStateProtocol` capture and the `ExecutionRecord`.

The solver arm's 2.436 is unchanged from CHE-105 and is not a regression: it is
8 ms of absolute overhead on a 32-ring trace, and the ≤10% target is a statement
about workloads rather than about microbenchmarks.

## Device placement is read off the arrays

`_observe_precision` takes `actual_device` and `actual_dtype` off the produced
artifact, never off the request, and the node is recorded `EXECUTED_LOSSY` when
they disagree with what was asked. On this run:

| node | requested device | actual device | actual dtype |
|---|---|---|---|
| `sensor_reconstruction` | cpu | cpu | complex128 |
| `wave` | cpu | cpu | **complex64** |

The wave row is the point. Nothing asked for complex64; Chromatix casts
unconditionally, and a record that reported the requested dtype would have
recorded a fiction. Asserted by
`tests/test_substrate_proof.py::test_the_device_placement_is_read_off_the_arrays_not_off_the_request`.

The mismatch case the criterion names -- asked for CUDA, landed on the host -- is
asserted by `tests/test_executor.py::test_a_run_that_asked_for_cuda_and_got_the_host_is_detected`,
against a faked adapter rather than a real device. That is deliberate: what is
under test is that the executor believes the array and not the request, and a real
CUDA run only exercises the case where the two agree. The node is recorded
`EXECUTED_LOSSY`, `honoured` is `False`, and `cost.device` is `cpu` so a timing
read out of the record cannot be attributed to hardware it never touched.

**Not measured here: CUDA.** This workload is CPU by declaration
(`CANONICAL_PARAMETERS["device"] = "cpu"`), the family's `ExecutionPolicy` permits
both, and no GPU run was made in this issue.

## demo2 / B3-DEMO2-01 was NOT migrated, and why

Two structural blockers, both in M3.1 rather than in the benchmark, now recorded
in `runtime/executor.py`'s module docstring beside the streaming refusal:

1. **No node to hang the edge on.** Every `GraphSpec` edge names a source *node*
   and a target *node*, and a node names a registered model. The registry has
   exactly two, `M_RAY_OPTILAND` and `M_WAVE_CHROMATIX`. demo2 is a bare SLM
   behind a circular amplitude mask with a sensor 1.26 mm downstream and **no
   refractive surface** — its own record says `optiland_used: false` — and the
   operation it exercises is `C_PLANAR_DOE_STEP`, which consumes an incident ray
   bundle *and* a DOE transmission and emits a ray bundle. No registered model
   produces either input or consumes the output. demo2 cannot be written as a
   graph document no matter what the couplers support.
2. **Streaming.** RW-P is 1.6e8 rays in 40 chunks. `GraphExecutor` refuses a node
   declaring `streaming: true` with `UNSUPPORTED_CAPABILITY`, by design and
   documented since CHE-113.

The missing piece is a **source/sink node contract**.
`runtime/instance_runner.py::field_source` is the shape of half of it and is
deliberately not a registered model: it reaches the executor through `inputs`,
which works for a wave node whose port is `input_field` and does nothing for a
coupler that needs a source node to exist. Inventing `M_SOURCE_ARRAY` and
`M_SENSOR_PLANE` to unblock one benchmark would register two capability claims no
oracle has ever checked, which is the opposite of what the registry is for.

So `B3-DEMO2-01`'s committed numbers — enumerated `7.11e-13`, RW-F Table S2 NCC
`0.998693` / rel-L2 `8.87e-2`, RW-P NCC `0.999418` / rel-L2 `2.86e-2` — were
**not re-measured** and no claim about them is made here. They stand where they
were.

## Bespoke entry points: the denominator for S4

Counted as files under `benchmarks/` containing `if __name__ == "__main__"`:

| directory | entry points |
|---|---:|
| `benchmarks/probes/` | 39 |
| `benchmarks/instances/` | 7 |
| `benchmarks/perf/` | 3 |
| `benchmarks/systems/` | 3 |
| `benchmarks/physics/` | 2 |
| `benchmarks/applied/` | 1 |
| **total** | **55** |

Of those 55, **two** execute a real `GraphSpec` through the executor:
`benchmarks/instances/b1_wave.py` (a single wave node driven through `inputs`) and
`benchmarks/instances/b3_psf_singlet.py` (the three-node ray → wave → field chain
this issue landed). The other eight family drivers under `instances/` and
`systems/` build their `ExecutionRecord` with
`runtime.instance_runner.record_from_probe` and say so in their own docstrings —
they are enrolled in the record/verify half of the substrate and not in the
execute half, mostly because their physics has no registered node either.

**S4 is therefore 2/55 as of this issue**, up from 1/55. That is the honest
number and it is small; the reason it is small is the source/sink gap above, not
fifty-three drivers nobody has looked at.

## Line-count delta

| | lines |
|---|---:|
| `benchmarks/physics/L2-PSF-01/run_benchmark.py` (bespoke, retained) | 600 |
| `benchmarks/instances/b3_psf_singlet.py` (this issue, ~250 of it prose) | 507 |
| `examples/graphs/psf_singlet_sensor.yaml` (mostly comment) | 142 |
| `src/runtime/variants.py` (new, general) | 98 |
| `src/couplers/handoff.py::advance_bundle_to_plane` (new) | 57 |

**Nothing was deleted.** CHE-115's own amendment assigns the deletion of
`run_benchmark.py` to CHE-116 and makes reproducing this case the precondition,
which is now satisfied. What the runner still owns and this path does not
reproduce: the 12-rung convergence ladder, the O2 ASM/Rayleigh-Sommerfeld
characterization oracle, the exit-pupil hard-support reconstruction control, the
quadrature-weight regression control, and the `result.json`/`plot.png` bundle
packaging. Deleting it today would lose all five.

## Suites run

| suite | command | result |
|---|---|---|
| Default gate | `./run.sh --no-build pytest -q` | 2714 passed, 67 skipped, **3 failed**, 58.7 s |
| Slow | `./run.sh --no-build pytest -q -m slow` | 43 passed, 98.8 s |
| Targeted | `pytest tests/test_substrate_proof.py -m "not slow" -n 0` | 16 passed, 26.9 s |
| Targeted | `pytest tests/test_substrate_proof.py -m slow -n 0` | 2 passed, 68.8 s |
| Targeted | `pytest tests/test_ray_to_wave_node.py -m "" -n 0` | 29 passed, 4.2 s |
| Provenance | `pytest tests/test_provenance_fingerprint.py -m "" -n 4` | 166 passed, 1 failed |

**All three default-gate failures are pre-existing**, verified by stashing this
issue's changes and re-running:

- `test_fixed_suite.py::test_success_metric_s1_is_met` — `C_GENERALIZED_SNELL` has
  no required-tier instance with an independent oracle. Already filed as a
  `claim_ledger` gap by CHE-146.
- `test_fixed_suite.py::test_every_registered_family_contributes_at_least_one_instance`
  — six families registered by M2.8-M2.12 are not in `FIXED_V1`.
- `test_provenance_fingerprint.py::test_every_stamped_record_still_describes_this_tree_code`
  — see below.

The default gate was **54.5 s** at CHE-140 and is **58.7 s** here. The frozen arm
of `test_substrate_proof.py` went from ~10 s to ~27 s, which is still under
`test_executor_integration.py`'s ~31 s, so it did not become the critical shard;
CHE-140's report is left as the historical record of what it measured.

## Records regenerated, and why

`src/couplers/handoff.py` and `src/couplers/node.py` are real code changes, so
every stamped record whose run imported them is stale by construction. Regenerated
through their own producers, never hand-edited:

- **70 instance records**, all of `benchmarks/instances/records/` except the new
  one: `b0_contract.py`, `b1_ray.py`, `b1_wave.py`, `b1_gsl_validity.py`,
  `b2_equiv.py`, `b2_transitions.py`, each `--write`. ~2.5 minutes total.
- **4 enrolled probe records**: `m3_convergence` (3.0 min),
  `m3_first_null_grid_convergence`, `m3_psf_verification`, `m3_off_axis_handoff`.

**Not regenerated, and pre-existing:** `test_every_stamped_record_still_describes_this_tree_code`
already failed on the committed tree before this issue, for 16 GPU-produced
records stale since CHE-142/CHE-148 edited `src/couplers/ontology.py`,
`interaction.py`, `capabilities.py` and `verification/claim_ledger.py`, and
additionally reporting an environment change. Those need GPU time this issue did
not spend and did not make worse. Verified by stashing this issue's changes and
re-running the sweep: 1 failed, 164 passed before; the same 1 failed after.
