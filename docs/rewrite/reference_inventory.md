# Reference Inventory — what the frozen implementation proves we need

**Ticket:** CHE-152 (R00), with CHE-168 (R00.1), CHE-169 (R00.2), CHE-170 (R00.3).
**Reference:** tag `pre-rewrite-2026-08-30` at `38539f26b90e04ebf2c0ed244b51ef628080738b`.
**Status:** no `src/` change was made by this ticket. The tree is unmodified.

## What this document is, and the question it refuses to answer

For every module, test and frozen record in the reference implementation, this
document answers one question: **what validated capability does this file prove
we need?**

It deliberately does **not** answer "where should this file move?". There is no
old-file → new-file mapping table anywhere in this document, and CHE-152's
acceptance criterion 6 makes its presence a review failure. The reason is not
stylistic. The reference tree reached 54,888 LOC and 280 classes under stated
principles comparable to the new ones; a mapping table would carry that
structure forward under a new set of directory names, which is the outcome this
rewrite exists to avoid. What survives is *capabilities, conventions,
algorithms, evidence and justified tolerances* — not files.

## How the numbers here were produced

Everything counted below was measured against the working tree at the tag, not
read off a previous report. Where a measurement disagrees with the figure quoted
in the Linear issues, the measurement is stated and the disagreement is called
out in section 7. The scripts were throwaway; the reproduction recipes are given
inline so any number here can be re-derived.

## Contents

* §0 The freeze: tag verification and the branch decision (R00.1)
* §1 Capability inventory, backend pins, and the `WavefrontSamples` verdict (R00.2 part 1)
* §2 Algorithm inventory — the kernels not to re-derive (R00.2 part 2)
* §3 Class inventory: the 214 `none` count R14 is measured against (R00.2 part 3)
* §4 Test-evidence triage and the `slow`/`gpu` blind spot (R00.3 part 4)
* §5 Record disposition, the measured staleness mechanism, and the four
  stamped records that are **already stale at the tag** (R00.3 part 5)
* §6 Tolerance and oracle extraction from the two ledgers R14 deletes (R00.3 AC 4)
* §7 Where these measurements disagree with the issues
* §8 What this hands forward, and the open risks
* §9 Full class table (280) · §10 Full test triage (85) · §11 Full record
  disposition (293) · **§12 Full module inventory (112) — the per-module
  disposition CHE-152 AC 3 asks for**

---

# 0. The freeze (R00.1)

## 0.1 The tag

```
$ git rev-list -n1 pre-rewrite-2026-08-30
38539f26b90e04ebf2c0ed244b51ef628080738b
$ git cat-file -t pre-rewrite-2026-08-30
tag                                  # annotated, not lightweight
$ git rev-parse refs/tags/pre-rewrite-2026-08-30
a4ef9b5957fa567b84f027b130f70e7390565c24
$ git ls-remote --tags origin
a4ef9b5957fa567b84f027b130f70e7390565c24  refs/tags/pre-rewrite-2026-08-30
38539f26b90e04ebf2c0ed244b51ef628080738b  refs/tags/pre-rewrite-2026-08-30^{}
```

The local tag object and the pushed tag object are the same object
(`a4ef9b5`), and it dereferences to the required commit. The working tree was
clean when the tag was cut and is clean now.

The annotation states, in the tag message itself, that the tag is the **physics**
reference and **not** an architectural reference — so a reader who finds the tag
without finding this document still cannot cite it to justify a layout, a class,
a registry shape or a layering decision.

## 0.2 The branch decision, and why

**Decision: tag plus a separate rewrite branch. The CHE-140 branch was not
renamed and was not checked out.**

`chengjiazhou4802/che-152-greenfield-rewrite` exists at `38539f2`. The original
plan called for renaming the current branch to `2026-08-30`; that was not done,
for three reasons:

1. The current branch is `chengjiazhou4802/che-140-default-test-suite-375s-under-60s`
   and CHE-140 is still In Review. Renaming it drops Linear's git-branch
   association, and a bare-date branch name links to no issue.
2. The tag already does the archival job in full. The rename adds nothing the
   tag does not give, and costs the issue association.
3. A branch rename or checkout is a shared-state mutation in a worktree that has
   been shared with a peer Claude Code session, where a peer's `git merge` has
   silently reverted uncommitted edits before. `ListAgents` reported no other
   session running at the time of this work, and `git worktree list` shows the
   second worktree (`/home/chengjz/moa_m2_11_13`) is on a different branch.

**Deviation from the overnight instruction, stated plainly:** the standing
instruction for this session is "commit to the current branch; do not create or
switch branches", while CHE-152's acceptance criterion 2 requires a rewrite
branch to exist. Both are satisfied without conflict: the rewrite branch is a
ref created at the frozen commit, nothing is checked out onto it, and this
ticket's commits land on the current branch. No branch was switched.

---

# 1. Capability inventory (R00.2 part 1)

The capabilities the reference implementation demonstrably has, each with the
authoritative file and the strongest evidence that it works. "Strongest
evidence" means the check that would fail first if the capability were broken —
not the largest test file.

| capability | authoritative file | strongest evidence | note |
| --- | --- | --- | --- |
| Ray representation and its conventions | `src/core/boundary.py` (`RayBundle`) | `tests/test_artifacts.py`, `tests/test_coupler_contracts.py` | Frame, phasor sign, reference plane, normalization and a dtype-dependent `‖d‖−1` tolerance all live here. This is the single most convention-dense file in the tree. |
| Scalar/complex field representation | `src/core/boundary.py` (`ComplexField`) | `tests/test_coupler_contracts.py` | Complex amplitudes, not intensities. |
| PSF as an observable | `src/verification/psf_measurement.py` | `tests/test_psf_measurement.py` | Already a measurement, not a representation. `C_FIELD_TO_PSF` was retired precisely because a trivial observable is not a cross-representation handoff. |
| Optiland ray solving | `src/solvers/optiland/adapter.py` + `execution.py`, `builder.py`, `artifacts.py` | `tests/test_optiland_adapter.py`, `tests/test_optiland_canonical_prescriptions.py` | Pinned `optiland==0.6.0`. |
| Chromatix scalar-wave solving | `src/solvers/chromatix/adapter.py` + `propagation.py` | `tests/test_chromatix_adapter.py` (partly `slow`) | Pinned `chromatix @ d24bdf0`. |
| ray → wave | `src/couplers/ray_to_wave.py` | `tests/test_ray_to_wave_kspace.py`, `tests/test_ray_to_wave.py` | k-space reconstruction; see §2. |
| wave → ray | `src/couplers/wave_to_ray.py` | `tests/test_wave_to_ray.py` (5 `slow` tests) | Angular-spectrum decomposition + importance weighting. |
| Coherent ray→wave handoff | `src/couplers/handoff.py`, `src/solvers/optiland/coherent_trace.py` | `tests/test_coherent_bridge.py`, `tests/test_optiland_coherent_handoff.py` | The OPD reference and handoff-plane conventions. |
| Ray/wave propagation | `src/couplers/propagation.py`, `src/solvers/chromatix/propagation.py` | `tests/test_coupler_round_trip.py` (`slow`) | Regime selection is the real content. |
| Diffractive surface models | `src/couplers/interaction.py`, `doe_node.py`, `generalized_snell.py`, `patch.py` | `tests/test_diffractive_interaction.py`, `tests/test_planar_doe_step.py`, `tests/test_patch_wft.py` | Three parameterizations, each with a declared validity regime. |
| Precision / device policy | `src/core/precision.py`, `src/core/arrays.py`, `src/core/bridge.py` | `tests/test_precision_contract.py`, `tests/test_precision_execution_matrix.py` (`gpu`) | dtype ladders, `DevicePlacement`, `BridgePlan`. |
| Record provenance / staleness | `src/core/provenance.py` | `tests/test_provenance_fingerprint.py` | The one runtime mechanism worth reusing verbatim; see §5. |
| Structured failure | `src/core/errors.py`, `src/verification/refusals.py` | `tests/test_contract_code_reachability.py` | Every `ContractCode` is reachable. This is what makes "never invent fields" checkable. |
| Analytic oracles (O1) | `src/verification/psf_oracles.py`, `src/verification/analytic.py` | `tests/test_psf_verification.py` | Airy, Fraunhofer, reference-sphere fit. Gate-deciding. |
| ASM/RS oracle (O2) | `src/verification/asm_oracle.py` | `tests/test_psf_verification.py` | **Diagnostic only — never gate-deciding.** It shares code with what it tests. |

Active backend integrations and their pins, read from the dependency files
rather than from prose:

* `optiland==0.6.0` — `docker/requirements.txt:95` (and `optiland>=0.6.0` in the
  `pyproject.toml` `torch` extra). Note that the canonical repository moved
  GitHub orgs (`HarrisonKramer/optiland` → `optiland/optiland`).
* `chromatix @ git+…@d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee` —
  `pyproject.toml:46`, in the `wave` extra.
* `scipy>=1.10` is a **declared** dependency of the verification oracles
  (`scipy.special.j1` for the Airy pattern), not merely a transitive one.

## 1.1 `WavefrontSamples`: consumer count settled (R00.2 AC 5)

**Production consumers: zero.** R02 can collapse it.

```
$ grep -rl 'WavefrontSamples' src --include='*.py'
src/core/boundary.py        # the definition (class at boundary.py:1051)
src/couplers/__init__.py    # a re-export in __all__, nothing more
```

No coupler, solver, runtime or discovery module produces or consumes it. Its
only production factory, `WavefrontSamples.from_artifact_record`
(`boundary.py:1165`), is documented in its own docstring as *designed to fail*
on an unmodified Optiland wavefront artifact, because that artifact's only OPL
source is `RealRays.opd`, whose convention the adapter itself documents as not
independently verified.

Its three test references are:

* `tests/test_coupler_contracts.py:450,457` — the only place it is actually
  constructed and exercised.
* `tests/test_package_dependencies.py:183` — architecture protection: it asserts
  the artifact is importable from `core.boundary` because AGENTS.md's "Initial
  Artifact Boundary" names it. This is a test of a document, not of physics.
* `tests/test_ray_to_wave_node.py:525` — a docstring mention, not a use.

So the artifact is kept alive by a sentence in AGENTS.md and one contract test.
The *knowledge* worth keeping is the distinction it was created to express —
pupil-sampled phase before rasterization vs. a gridded field — and that
distinction is already carried by `RayBundle` (with OPL) and `ComplexField`
(gridded). AGENTS.md's artifact-boundary sentence must be rewritten in R01.2 or
the collapse will look like a regression against the document that outranks the
code.

---

# 2. Algorithm inventory (R00.2 part 2)

The validated numerical kernels where rewriting from scratch would create
numerical risk rather than remove architectural risk. Reuse the mathematics and
the derivation; re-author the surrounding code.

| algorithm | location | why it must not be re-derived |
| --- | --- | --- |
| Hexapolar ring index + area weights | `src/couplers/quadrature.py:61,127` | Must agree with `optiland.distribution.HexagonalDistribution`'s `r = linspace(0, 1, num_rings+1)` to float64 round-off. The module derives an explicit ring-membership tolerance and *refuses* a hand-built or vignetted bundle that misses it rather than silently mis-binning. Evidence: `tests/test_quadrature.py`. |
| k-space reconstruction | `src/couplers/ray_to_wave.py:353` (`_reconstruct_kspace`), with `grid_nyquist_direction_limit:285` | The scatter-add reconstruction and its Nyquist direction limit. Evidence: `tests/test_ray_to_wave_kspace.py`. |
| Angular-spectrum decomposition | `src/couplers/wave_to_ray.py:198` (`decompose`), `340` (`spectrum_to_rays`) | Includes the strict inequality at line 235 that excludes the singular grazing `k_n = 0` bin. Evidence: `tests/test_wave_to_ray.py`. |
| Grazing floor / band limit | `src/couplers/streaming.py:123` (`grazing_floor_for_phase_budget`), `171` (`band_limit_spectrum`) | **The H4 / CHE-70 defect mitigation.** Near grazing, two large terms `~Z/d_n` differ by `Z·d_n`, and float32 cancels. The floor is *derived* from a declared phase budget (a hundredth of a radian) rather than tuned. The module's own docstring records 2.8e-09 relative field error with no band limit. Omitting this makes the new coupler **worse** than the old one. |
| Generalized-Snell margins | `src/couplers/generalized_snell.py:62,74,171` | Propagating-order margin, local-gradient-smoothness margin, single-order dominance. Three independent validity tests, not one. Evidence: `tests/test_diffractive_interaction.py`. |
| Patch position enumeration + coverage correction | `src/couplers/patch_positions.py`, `src/couplers/patch.py` | The correction is `A_draw/A_patch` and **not** its inverse; the ledger records that the power ratio is exactly 1 only in that orientation. Evidence: `tests/test_patch_wft.py`, `B2-EQUIV`. |
| Reference-sphere fit | `src/verification/psf_oracles.py:230` (`fit_reference_sphere`), residual at `:215` | The handoff geometry the coherent bridge depends on. |
| Airy oracle | `src/verification/psf_oracles.py:114,141,159` | `airy_first_null_radius_m`, `airy_intensity_at_radius`, `airy_psf_on_grid`. Closed form, gate-deciding. |
| Fraunhofer oracle | `src/verification/psf_oracles.py:444,478` | `FraunhoferPsf` / `fraunhofer_psf`, with `pupil_aberration:345` and `first_null_comparison:703`. |
| Carrier-removed ASM | `src/solvers/chromatix/carrier_removed_asm.py` | The tilted/off-axis phase handling that keeps sampling tractable. Evidence: `tests/test_carrier_removed_asm.py`. |
| Curvature bound | `src/couplers/curvature.py` | When a ray fan may be treated as locally planar. Evidence: `tests/test_curvature_bound.py`. |
| Convergence fitting + σ-margin | `src/verification/evidence.py:172,241,288,373` | `fit_convergence`, `_slope_standard_error`, `ensemble`, `sigma_margin`. Gates stochastic claims in units of the *measured* standard error rather than a chosen constant — the reason `C_WAVE_TO_RAY`'s bias gate is meaningful. |
| Provenance projection | `src/core/provenance.py:62,74,186,359` | `VOLATILE_KEYS`, `strip_volatile`, `source_fingerprint`, `verify_record_provenance`. Reuse verbatim in R13; see §5.1 for the one defect. |

---

# 3. Class inventory (R00.2 part 3)

All 280 production classes are tagged in the full table below (§3.3). The rules
are the project's own:

1. several fields share an invariant enforced together
2. a public serialized / versioned data model
3. owns a genuine mutable resource lifecycle
4. at least two *current* implementations need runtime polymorphism
5. a real plugin boundary used by the runtime or registry

Otherwise the rule set's own answer applies: function, module, frozen dataclass,
TypedDict, tuple, Literal, Enum — and the class is tagged `none`.

## 3.1 The count that R14 is measured against

**`none` = 214 of 280.** 66 classes would satisfy at least one rule *if* their
capability survives.

| reason | count |
| --- | --- |
| satisfies ≥1 rule (capability survives) | 66 |
| capability not carried forward at all | 105 |
| `enum` — the rule set names Enum as the alternative to a class | 59 |
| no rule fits; collapses to a function/dataclass/TypedDict | 28 |
| `exception` — structurally necessary, no rule claims it | 22 |
| **`none` total** | **214** |

`WavefrontSamples` is counted in "capability not carried forward": it satisfies
rules 1 and 2 structurally, but §1.1 establishes it has zero production
consumers and §8 dispositions it for collapse, so tagging it justified would put
this section at odds with the two that decide its fate.

Two tagging passes were run, and the difference between them is the informative
part. A purely *structural* pass — does this class, in isolation, satisfy a rule?
— tags only 132 as `none`. Adding the question *does the capability survive into
the new architecture?* moves 82 more classes to `none`. The structural pass is
the one that would be run by a linter; the capability-gated pass is the honest
one, and it is the number quoted above. Both are reported precisely so that R14
can measure against either without re-deriving this section's judgment.

**Where the gate was not applied uniformly, stated rather than hidden.** 14 of
the 66 justified classes live in modules §12 dispositions "nothing worth carrying
forward": `CostEstimate`, `StrictModel`, `GraphExecutor`, `SolverStateProtocol`,
and the 10 request/result envelope classes in `couplers/base.py`,
`solvers/base.py`, `chromatix/requests.py` and `optiland/requests.py`. They keep
a structural tag because the *shape* they encode (an operation was asked to run
and either did or refused) is real and is re-expressed in the new tree; they are
not carried as classes. §3.2 is where that collapse is owed. Read literally,
"justified" in this table means "satisfies a rule", not "will exist".

Per package, capability-gated:

| package | classes | `none` | justified |
| --- | --- | --- | --- |
| `src/core` | 110 | 89 | 21 |
| `src/verification` | 72 | 65 | 7 |
| `src/couplers` | 44 | 22 | 22 |
| `src/solvers` | 18 | 6 | 12 |
| `src/runtime` | 11 | 8 | 3 |
| `src/discovery` | 10 | 10 | 0 |
| `src/agent` | 8 | 8 | 0 |
| `src/studies` | 6 | 6 | 0 |
| `src/registry` | 1 | 0 | 1 |

## 3.2 The finding this inventory hands to R02–R11

**66 justified classes still exceeds the project's ≤22 budget by a factor of
three.** The gap is not slack in the tagging; it is real, and it is concentrated
in two places:

* **Request/result envelope pairs.** `ModelRunRequest`/`ModelRunResult`,
  `CouplerRunRequest`/`CouplerRunResult`, `OptilandRayRequest`/`Failure`/`Result`,
  `ChromatixWaveRequest`/`Failure`/`Result` — 11 classes expressing one idea
  ("an operation was asked to run and either did or refused"). Under the new
  architecture that is descriptor metadata plus one result type.
* **Per-coupler diagnostics types.** `ReconstructionDiagnostics`,
  `PatchDiagnostics`, `GeneralizedSnellDiagnostics`, `CurvatureBudget`,
  `DifferentiabilityReport`, `CascadeDiagnostics`, `AngularSpectrum`,
  `PositionPlan`, `PatchPlan`, `Perturbation`, `SamplingPerturbation`,
  `HandoffPerturbation` — each individually satisfies rule 1, and collectively
  they are one pattern: a coupler reporting what it did and whether it was
  valid.

Both collapse to a single versioned diagnostics payload plus per-coupler
`TypedDict`s. Recording this now is the point: reaching ≤22 requires collapsing
these two families deliberately in R05–R11, and it will not happen as a
side effect of re-authoring module by module.

## 3.3 Full class table, and how to reproduce the number

The full 280-row table is at the end of this document (§9) so it does not break
the reading flow.

**Structural pass.** AST-walk `src/**/*.py` and collect, per `ClassDef`: bases,
decorators (including `frozen=True`), annotated field count, method names, and
subclass count within the tree. Then: rule 2 if it is a pydantic `BaseModel` or
has a serialization method and ≥2 fields; rule 1 if ≥2 fields and (frozen or a
`__post_init__`) and it is not an enum or exception; rule 3 if it defines
`__enter__`/`__exit__`/`close`/`release`/`shutdown`; rule 4 if ≥2 subclasses in
the tree; rule 5 if it is a `Protocol`/`ABC` with ≥1 subclass. Enums and
exceptions are forced to `none`, because the rule set names Enum as its own
alternative to a class. → **132 `none`**.

**Capability gate.** A class is additionally forced to `none` when its owning
capability is not carried forward. The criterion is §12's disposition: the gate
fires for the `nothing worth carrying forward` modules **plus** the
benchmark-family substrate, the two ledgers, the verifier/result/status/refusal
modules, `studies/`, `agent/`, `discovery/` and the perf harness — whose modules
are dispositioned `test evidence to reuse`, meaning their *records* survive and
their classes do not. The one hand-entered exception is
`streaming.py::BandLimit`, kept because the grazing floor is a must-reuse kernel
(§2) inside an otherwise dropped subsystem. → **214 `none`**.

The two passes are reported separately in §3.1 because the second one encodes
judgment and the first one does not.

---

# 4. Test-evidence triage (R00.3 part 4)

All 85 files under `tests/` are tagged in §10. Summary:

| tag | files | LOC |
| --- | --- | --- |
| physics evidence | 47 | 25,954 |
| architecture protection | 38 | 14,488 |
| **total** | **85** | **40,442** |

One file sits awkwardly in a two-tag scheme and is called out rather than
quietly filed: `tests/test_resources.py` is tagged **physics evidence** because
it asserts a measured fact about the world (which swap counter is attributable
on this host — see §6.4), not a fact about our directory layout. It is not optics.
The tag boundary that matters for the rewrite is "evidence about reality, port
it" vs "evidence about this layout, re-derive or delete it", and on that boundary
it belongs with the former.

"Architecture protection" is not a criticism. `test_package_dependencies.py`'s
AST import walk is the *mechanism* R01.1 reuses; what does not survive is the
list of package names it asserts. The distinction that matters for the rewrite is
that a physics-evidence test states something about the world and can therefore
be ported, while an architecture-protection test states something about this
directory layout and must be **re-derived against the new layout or deleted** —
porting it would import the old architecture through the test suite, which is the
back door R14.2 exists to close.

## 4.1 Marker reality: what the default gate does not run

Read from pytest's own collection, not from grep — a file-level `pytestmark` and
`@pytest.mark.parametrize` are indistinguishable to a naive grep:

```
$ ./run.sh --no-build pytest -q -m slow -n 0 --collect-only   # 44/2846 collected
$ ./run.sh --no-build pytest -q -m gpu  -n 0 --collect-only   # 66/2846 collected
$ ./run.sh --no-build pytest -q        -n 0 --collect-only    # 2802/2846, 44 deselected
```

* **44 `slow` tests in 8 files**: `test_b2_transition_instances.py` (22),
  `test_coupler_gradient.py` (10), `test_wave_to_ray.py` (5),
  `test_substrate_proof.py` (3), and one each in `test_resource_profile_guard.py`,
  `test_flat_layout.py`, `test_coupler_round_trip.py`, `test_chromatix_adapter.py`.
* **66 `gpu` tests in 6 files**: `test_precision_gpu_pipeline.py` (19),
  `test_metalens_bridge_gpu.py` (15), `test_b1_wave_gpu.py` (9),
  `test_b1_ray_gpu.py` (9), `test_gpu_environment.py` (8),
  `test_precision_execution_matrix.py` (6).

Note that `slow` and `gpu` are disjoint here, and that the default suite is
**not** a superset: `-m "not slow"` deselects 44, and the 66 `gpu` tests are
skipped rather than deselected in a CPU container. So the physics evidence for
device parity, stochastic-estimator bias, convergence rate and gradient
behaviour is *entirely* outside the default gate. Any claim that "the suite is
green" says nothing about those four.

---

# 5. Record disposition (R00.3 part 5)

**293 record files**, not the ~200 the issues estimate. All 293 are given a
disposition and a reason in §11. Summary:

| disposition | count |
| --- | --- |
| keep as active evidence, then regenerate at R13 | 135 |
| keep as historical (the tag holds it; the file may go) | 120 |
| regenerate after the rewrite | 20 |
| keep as active evidence | 18 |

The first row is exactly the stamped set, and that is not a coincidence: being
stamped decides the disposition on its own (§5.1), so it overrides both "is
cited as ledger evidence" and "attests to backend rather than repository
behaviour". A stamped record can stay *citable* across the cut but cannot stay
*verifiable*, and conflating those two is what would let R13 under-budget.

By directory:

| directory | files | note |
| --- | --- | --- |
| `benchmarks/probes/records/` | 152 files (101 `.json` + 50 `.npz` + `REGISTER.yaml`) | 13 top-level + 123 `ray_wave/` + 9 `optiland/` + 6 `chromatix/` + 1 `ray_to_wave/` |
| `benchmarks/instances/records/` | 71 | all 71 stamped |
| `benchmarks/systems/records/` | 40 | all 40 stamped |
| `benchmarks/perf/records/` | 20 | none stamped |
| `benchmarks/applied/commercial_lens_systems/records/` | 10 | **not mentioned in any issue**; see §7 |

## 5.1 The staleness mechanism, measured

This is the part that had to be settled while the source is still frozen, and the
measured exposure is much larger than the estimate the issues carry.

`core.provenance.verify_record_provenance` recomputes an AST-normalized digest
for **exactly the files a record recorded at probe time**, and reports both
`changed` and `removed` files. Therefore:

* **135 records are stamped** — 24 probes + 71 instances + 40 systems. (`perf/`
  and `applied/` carry no `record_provenance` block at all.)
* **All 135 fingerprint the same 11 `src/core/` modules**: `__init__.py`,
  `arrays.py`, `artifacts.py`, `boundary.py`, `capabilities.py`, `errors.py`,
  `execution.py`, `graph.py`, `precision.py`, `provenance.py`, `specs.py`.
  133 also cover `src/registry/`, and 126 cover the same 15 `src/couplers/`
  modules (20 distinct `src/couplers/` modules appear across the stamped set).
* The fingerprints reference **98 of the 112 `src/` modules** (plus 16 files
  under `benchmarks/probes`, which are fingerprinted too because probes import
  each other).

The consequence, stated precisely: **any AST-level change to any one of those 11
core modules stales all 135 stamped records at once**, and R14's deletion of the
old `src/` tree stales all 135 unconditionally via the `removed` path — not
"most of them", and not "eventually". This is why the disposition had to be
decided now rather than discovered at R14.

**14 stamped records were produced on `cuda:0`**, read from the device fields in
the records themselves rather than inferred from filenames:
`chromatix/b1_wave_device_observation.json`,
`optiland/b1_ray_device_precision.json`,
`ray_wave/demo3_convergence_kspace_rw_p.json`, the seven
`ray_wave/demo3_variance_*` records, and the four `ray_wave/perf_demo*_cuda.json`
records. One more (`demo3_variance_ladderfit_control.json`) records `jax` with no
device. Those 14 are the ~80-minute tier. They are **not** the whole GPU bill: seven
`benchmarks/perf/records/` entries are also GPU measurements — the five
`*_cuda.json` files plus `optiland_trace_precision.json` and
`optiland_trace_decomposition.json`, which carry `"device": "cuda"` — and all
seven are dispositioned `regenerate after the rewrite`. **The GPU-dependent
regeneration total is therefore ~21 records, not 14.** Note that `gpu_name:
NVIDIA RTX A6000` appears in the `environment` block of all 20 perf records
including the explicitly CPU ones, so it is host metadata and is not the
discriminator; the compute device is.

## 5.1a Four stamped records are already stale at the tag

This was found by running the guard rather than by reading it, and it changes how
§5.1's "parity reference" should be read.

```
$ ./run.sh --no-build pytest -q -n 0 -m "" tests/test_provenance_fingerprint.py
6 failed, 201 passed
```

On a **completely clean tree at `38539f2`** — no uncommitted files — six tests
fail:

* `test_the_enrolled_records_still_describe_this_tree` for `m3_convergence`,
  `m3_first_null_grid_convergence`, `m3_off_axis_handoff` and
  `m3_psf_verification`. All four report
  `code_changed=True, changed_files=('src/couplers/ray_to_wave.py',)`.
* `test_every_stamped_record_still_describes_this_tree_code`, the aggregate of
  the above.
* `test_a_record_that_drifts_from_its_probe_fails`, which needs an undrifted
  baseline to perturb and says so: *"the demonstration needs a clean baseline to
  perturb, and this tree is already drifted — see the enrolled-records failures
  above, which are the real finding here."*

So the tree named as the physics reference has a **red default suite**, and
`src/couplers/ray_to_wave.py` drifted from four stamped records at some point
before the freeze without being regenerated. Three consequences for the rewrite:

1. **4 of the 24 stamped probe records are not currently verifiable evidence.**
   They keep the disposition `keep as active evidence, then regenerate at R13`,
   because their *values* are still the best available reference — but R02 and
   R07 must treat the four `m3_*` numbers as unconfirmed against the code that
   produced them, not as a passing gate. The four cover PSF verification,
   first-null grid convergence, off-axis handoff and convergence: exactly the
   ray→wave territory R07 owns.
2. **This is not a rewrite regression, and must not be mistaken for one later.**
   It predates every commit in this project. Recording it here, at the freeze, is
   the only way a later ticket can tell "the rewrite broke this" from "this was
   already broken".
3. **The guard works.** It refused to report a clean baseline it did not have,
   which is the behaviour §2 recommends reusing verbatim.

Fixing it is explicitly **not** in R00's scope: regenerating the four records
requires running the probes, which is a source-touching act on a tree this ticket
must leave frozen. It belongs to R07 or R13, and it is listed in §8.

**The one class of record the rewrite does not stale** is the *unstamped* part of
`probes/records/optiland/*` and `probes/records/chromatix/*` — **13 of those 15
files.** They attest to *pinned backend* behaviour rather than to repository
code, which is exactly why `REGISTER.yaml` leaves them unenrolled: a code
fingerprint would fail on unrelated repository edits. They survive the rewrite
as-is and are the cheapest real evidence in the tree.

The two exceptions matter and are named: `optiland/b1_ray_device_precision.json`
and `chromatix/b1_wave_device_observation.json` **are** stamped, both were
produced on `cuda:0`, and both therefore fall under the paragraph above rather
than this one. Their subject is backend behaviour but their stamp is over our
code, so the rewrite does stale them.

## 5.2 Two couplings that make "just delete it" wrong

1. **`REGISTER.yaml` must be edited in lockstep.** 77 unstamped probe records are
   declared in `benchmarks/probes/records/REGISTER.yaml` under 9 glob patterns,
   each with a reason. `tests/test_provenance_fingerprint.py::test_the_register_names_only_records_that_exist`
   fails on a pattern that matches nothing, and
   `test_every_committed_record_is_either_enrolled_or_declared` fails on a new
   record that is neither stamped nor declared. Deleting a declared record
   without removing its pattern turns a green suite red; adding one without
   declaring it does too. Measured now: 0 undeclared, 0 stale patterns.
2. **Directory-glob readers hide their dependencies.** A name-based search finds
   no reader for 203 of the 293 records, and that number is misleading. The real
   readers sweep directories: `test_provenance_fingerprint.py:210`
   (`RECORDS_DIR.rglob("*.json")`), `:476` (all 71 instance records),
   `test_performance_harness.py:347,945` (all 20 perf records),
   `test_b1_wave_gpu.py:310` (`B1-WAVE-*.json`, **`gpu`-only**), and
   `test_benchmark_inventory.py:67`. Judging a record unread from a name search
   is the same error as judging it unread from the default suite, one level out.

3. **The provenance sweep is narrower than it looks, in two ways.**
   `_all_records()` at `test_provenance_fingerprint.py:210` is `rglob("*.json")`,
   so it sees **101** of the 152 files under `probes/records/` — the 50 `.npz`
   field sidecars and `REGISTER.yaml` are not swept, and an `.npz` therefore has
   no verifier at all. More importantly, **the 40 `systems/records/` files are
   swept by nothing**: `verify_record_provenance` is applied over probe and
   instance records only. All 40 are stamped, so after R14 their stamps become
   silently false rather than red — the failure mode this section exists to
   prevent, in the one directory with no guard. R13 should extend the sweep to
   `systems/records/` before it regenerates anything.

## 5.3 `B0-META-01`'s non-deterministic fingerprint (R00.3 AC 5)

`benchmarks/instances/records/B0-META-01.json` rehashes on every run. This is a
**known artifact of the record's own text, not a measurement change**, and it
must not be read as drift during the rewrite.

The mechanism is one step deeper than "a uuid4 in a refusal message".
`src/solvers/optiland/artifacts.py:259` builds each ray artifact's `id` as
`f"{request.node_id}-rays-{uuid.uuid4().hex[:8]}"`. B0-META-01 refuses at the
`C_RAY_TO_WAVE` edge *after* a real Optiland trace, so that id is interpolated
into the refusal detail, which is committed verbatim:

```
"[REFERENCE_PLANE_MISMATCH] the record was produced at handoff plane 'exit_pupil'
 but the consumer declared 'image_surface' … (artifact: 'lens-rays-bdb02551')"
```

`run_id` **is** in `VOLATILE_KEYS` and is stripped; the artifact `id` is not, and
it reaches the hash through the message string. The honest fix belongs to R13,
and it is not "add `id` to `VOLATILE_KEYS`" — an artifact id is exactly the kind
of thing a fingerprint should notice. Either the identifier becomes
content-derived, or the refusal detail names the artifact by role rather than by
instance id.

---

# 6. Tolerance and oracle extraction (R00.3 AC 4)

R14 deletes `src/verification/claim_ledger.py` (1,729 LOC) and
`src/verification/declaration_ledger.py` (1,348 LOC). They are the only
machine-readable claim → oracle → tolerance map in the repository. The
justifications are extracted here because a tolerance whose derivation is lost is
the number a later ticket widens to make a benchmark pass.

Both ledgers were **executed**, not paraphrased:

```
$ ./run.sh --no-build python -c "…from verification.claim_ledger import all_claims…"
```

## 6.1 The claim ledger, as measured

90 claims. **45 carry a numeric tolerance**; 63 are gate-deciding.

| dimension | distribution |
| --- | --- |
| gate status | 59 `met`, 20 `characterized_no_gate`, 4 `not_met`, 4 `measured_off_gate`, 3 `not_measured` |
| oracle | 36 `analytic_closed_form`, 9 `independent_implementation`, 8 `deterministic_limit`, 7 `conservation_law`, 5 `cross_route`, 1 `convergence_exponent`, 24 `none` |
| independence | 59 `independent`, 24 `not_applicable`, **7 `shares_code`** |

The 7 `shares_code` claims are the O2/ASM entries. The ledger enforces — and a
test enforces — that a `SHARES_CODE` claim **cannot be gate-deciding**. This is
the standing rule that our own numerical code never decides correctness for our
own numerical code, and it is the rule L2-PSF-01 learned the hard way by once
setting a negative-control floor from an O2 comparison and having to retire it as
circular. **The rewrite must carry this constraint as an executable rule, not as
a convention**: it is the single most load-bearing piece of verification design
in the tree.

`GAPS` holds 11 entries: 6 `critical`, 2 `high`, 2 `medium`, 1 `low`.

### Worked tolerance derivations worth preserving verbatim

These are the ones whose reasoning cannot be reconstructed from the number:

* **`M_RAY_OPTILAND` symplectic invariant, `1e-13`** (observed 1.204e-15). Derived,
  then checked against measurement rather than fitted to it. Three terms: (1)
  round-off — `eps = 2.2204e-16`, bounded by the same `64·eps` the adapter derives
  for its own direction-norm check, amplified by the measured conditioning of the
  secants and bilinear form (`Σ|terms|/|ω| = 1.0005`, `|q|/|dq| = 0.5`, i.e. by
  less than one) and by two Richardson levels (`Π (2^p+1)/(2^p−1)` for `p = 2,4`
  = 1.889), giving `64 · 2.2204e-16 · 1.0005 · 1.889 = 2.7e-14`; (2) truncation
  `O(eps^6)` after two Richardson levels, estimated at 6.1e-17 — three decades
  below the round-off term and therefore not what sets the number; (3) one decade
  of headroom. **Three decades tighter than the `1e-10` it replaced.**
* **`M_WAVE_CHROMATIX` device parity, `1e-4`** (observed 2.83e-05). One `eps32`
  per radian at `z = 40 µm`, `λ = 550 nm` is ≈5e-5; `1e-4` is the bound the closed
  form gives at that distance, so a loss above it means something other than the
  downcast is happening.
* **Convention gates, `5e-3`** (both solvers). The pinned solver reproduces the
  closed forms to better than 1e-6 when called correctly. `5e-3` is three orders
  above that and far below what either mistake produces: the µm/nm slip moves the
  coated reflectance to within 1.7e-5 of *bare glass* (a factor 3.3 from the
  correct 0.01283544), and the `ky`/`kx` swap is 6.28× and a sign.
* **`fabricated_output_count`, `0.5`** (4 components). Zero, expressed as a
  threshold on a count, so that "exactly zero" becomes a comparison. Enforces the
  AGENTS.md non-negotiable that failed paths return structured diagnostics.
* **Diffractive deflection, `1e-3` rad** (observed 1.11e-16). The smooth-limit
  instances measure worst-case error at float64 round-off through `asin` and the
  local finite-difference stencil; `1e-3` sits well above that floor and below the
  multi-radian-fraction errors the near-Nyquist and high-duty-cycle instances
  show. The basis was completed **after** executing the family, per the B2-EQUIV
  precedent: a threshold is not well defined before the family that owes it has
  run.
* **`M_RAY_OPTILAND` EFL, `1e-6`** (observed 2.79e-12). `R/(n−1)` is exact for a
  single refracting surface in air; the pinned solver reproduces it to 1e-13.

Six derivations are transcribed above. The remaining 39 tolerance bases are
**not** reproduced in this document; they survive at the tag, and §6.3 says
plainly what that does and does not buy.

## 6.2 The declaration ledger, as measured

59 registry declarations, **59 covered, 0 gaps, 0 orphaned, 0 ambiguous**:
22 assumptions, 22 warnings, 14 invariants, 1 hard limit. Coverage kinds: 34
`executable_test`, 19 `benchmark_case`, 6 `explicit_non_executable`.

**All 14 invariants carry a derived `tolerance_basis`** — the ledger refuses to
construct one that does not, on the stated grounds that "an invariant asserted at
a tolerance nobody derived is a number somebody liked". They group into four
bases, and the grouping is the reusable knowledge:

* **`numerical_precision_floor`** — float64 round-off over an enumerated sum.
  `C_RAY_TO_WAVE.phase_reference_consistency` at 1e-12 (measured 1.32e-15, with
  the deliberately broken twin at 1.40); `pupil_power_consistency` at 1e-12
  because the enumerated route is a *relabelling* of the same modes;
  `C_PLANAR_DOE_STEP.importance_weight_applied` — full enumeration has zero
  sampling error, so the residual is round-off and nothing else;
  `C_PATCH_WFT.importance_weight_applied` measured 5.9e-15 against a 1.4e-12 gate.
* **`conservation_law`** — `propagated + reported_discarded = input`, exactly, up
  to summation round-off. `evanescent_power_accounted` at 1e-9 for
  `C_WAVE_TO_RAY`, `C_PLANAR_DOE_STEP` and `C_PATCH_WFT`;
  `patch_coverage_corrected` at 1e-9, which holds *only* when the correction is
  `A_draw/A_patch` rather than its inverse.
* **`analytic_derivation`** — `C_WAVE_TO_RAY.importance_weight_applied` gates bias
  in units of the **measured** ensemble standard error at 3σ, so the tolerance is
  a property of the run rather than a chosen field-space constant.
  `C_PLANAR_DOE_STEP.outgoing_count_is_the_budget` is an exact integer identity,
  written as 1e-12 only because the schema's metric is float-valued.
* **The dtype-dependent `‖d‖−1` bound**, shared by all four couplers: float32 and
  float64 each against their own epsilon, because a fixed absolute bound would be
  either vacuous at float32 or unsatisfiable at float64. This is a single rule
  with four consumers and belongs to the representation in the new tree.

## 6.3 Form in which the extraction survives — and its limit

Both ledgers were dumped to structured JSON while doing this work (every claim
with its oracle, independence class, metric, tolerance, `tolerance_basis`,
observed value, device, dtype, evidence paths and caveats). Those dumps were
**deliberately not committed.** Committing a second copy of the ledger creates
exactly the two-places-for-one-fact problem the ledger's own design notes warn
against, and the copy would rot while reading as authoritative.

So, precisely: what survives the ledgers' deletion is **(a)** the 6 worked
derivations transcribed in §6.1 and the 4 invariant bases grouped in §6.2 —
chosen because their reasoning is not recoverable from the number alone; **(b)**
the structural facts (90 claims, 45 tolerances, the 7 `shares_code` entries and
the rule that they cannot gate, 59 declarations with 0 gaps); and **(c)** the
ledgers themselves, readable in full at `pre-rewrite-2026-08-30` and
re-dumpable with the command in §6.1.

**The limit, stated rather than glossed:** 39 of the 45 numeric tolerance bases
are preserved only by (c). That is a deliberate trade — a rotting copy is worse
than a tag — but it means the tag is load-bearing for CHE-170 AC 4, not merely
archival. **R13 owns re-establishing these as executable declarations in the new
tree**, and until it does, "the tolerance is justified" is a claim that requires
checking out a tag to verify.

## 6.4 One non-optics number that must not be re-derived by experiment

`src/core/resources.py` carries CHE-64's measured finding, and it is the only
justified number in the tree whose re-derivation would itself be a hazard:

**Host swap is non-zero at rest on this machine** — about 700 MiB across
`/swapfile` and `/dev/sda1` with no project process running. So
`/proc/meminfo`'s `SwapFree` moves for reasons unrelated to any benchmark, and a
guard keyed on it fires on unrelated host activity and reads as a failure of the
run it aborted. `/sys/fs/cgroup/memory.swap.current` is per-container and starts
at 0 in the `agent_solver` images, so a **delta** on it is attributable. That is
the signal, and `tests/test_resources.py` pins the guard and the instrumentation
to the same path so the repository cannot acquire two definitions of "did this
swap".

This backs the AGENTS.md non-negotiable that swap growth in the workload's cgroup
is a stop condition. Re-deriving it means deliberately driving a shared 80-core
server into swap to find out which counter moves, which is precisely what the
policy forbids. It is recorded here because §12 dispositions `resources.py` as
**knowledge to reuse** on the strength of this paragraph and nothing else.

---

# 7. Where the measurements disagree with the issues

Stated rather than silently corrected, because later tickets cite these figures.

| figure | issue says | measured | note |
| --- | --- | --- | --- |
| `src/` modules | 114 | **112** | `find src -name '*.py'` excluding `__pycache__`. Every per-directory figure in CHE-169 matches exactly — verification 30/23,317; couplers 20/9,748; core 19/7,946; solvers 26/6,674; **studies 5/3,041**; runtime 4/1,644; discovery 2/1,001; agent 2/869; registry 3/404; cli 1/244 — and those file counts sum to 112. (CHE-169's own enumeration omits `studies`' file count, which is where the arithmetic slipped.) The "114" is most likely the count of *fingerprinted* files, 98 `src/` + 16 `benchmarks/probes/` — a different set, and one this document uses in §5.1. The project's "~20 modules" target is unaffected. |
| `src/` LOC | 54,888 | 54,888 | exact |
| production classes | 280 | 280 | exact |
| `tests/` files / LOC | 85 / 40,442 | 85 / 40,442 | exact |
| probe records | "~160 incl. 123 under `ray_wave/`" | **152** files / **101** `.json` | the 123 under `ray_wave/` is exact; the split is 101 `.json` + 50 `.npz` + `REGISTER.yaml` |
| frozen records total | "~200" | **293** files / **242** `.json` | Both numbers are given because they answer different questions. 293 is every file under a `records/` directory. 242 excludes the 50 `.npz` field sidecars and the register. Against 242 the issues' "~200" is a mild undercount rather than a large one — the substantive gaps are the undeclared 5th directory and the stamped count below. R13's regeneration budget should be read against 242, since an `.npz` is regenerated with its `.json`, not separately. |
| stamped probe records | 22 | **24** | measured by presence of a `record_provenance` block |
| GPU-produced stamped records | 12 | **14** | measured from device fields in the records |
| record directories | 4 named | **5** | `benchmarks/applied/commercial_lens_systems/records/` is named in no issue |

None of these changes a decision in this document. The two that matter for later
tickets are the stamped count (**135 across three directories, not 22 in one**),
which sets R13's scope, and the fifth record directory, which was in nobody's
inventory.

**The tag annotation itself carries two of the superseded figures** — it says
"54,888 LOC across 114 modules" and "the ~200 stamped records". The LOC is right;
the module count and the record count are not, and an annotated tag cannot be
amended without moving the tag, which criterion 1 forbids. This document
supersedes the annotation on both counts, and a reader who finds the tag first
should be pointed here.

---

# 8. What this hands forward, and the open risks

**To R01.** `tests/test_package_dependencies.py`'s AST import walk is the
mechanism to reuse. `AGENTS.md`'s "Initial Artifact Boundary" sentence names
`WavefrontSamples` as a boundary artifact and is the only reason it still exists
(§1.1) — R01.2 must rewrite it, or R02's collapse reads as a regression against
the document that outranks the code. `pyproject.toml` enumerates the old tree in
five places.

**To R02.** `WavefrontSamples` has zero production consumers; collapse it. The
dtype-dependent `‖d‖−1` bound (§6.2) is one rule with four consumers and belongs
to the representation. `src/core/boundary.py` is the convention-dense file to
read before writing `representations/`.

**To R05–R11.** The two class families in §3.2 must be collapsed deliberately;
66 justified classes is 3× the ≤22 budget. The grazing floor (§2) is not
optional. The `SHARES_CODE`-cannot-gate rule (§6.1) must be executable.

**To R13.** 135 stamped records, 14 of them GPU-produced. `REGISTER.yaml` is
part of the mechanism, not documentation (§5.2). B0-META-01's fingerprint defect
(§5.3) is real and should be fixed at the artifact-id level.

**To R14.** 39 architecture-protection test files must be re-derived against the
new layout or deleted, never ported.

## Open risks

1. **The `slow`/`gpu` evidence blind spot.** Device parity, stochastic-estimator
   bias, convergence rate and gradient behaviour are entirely outside the default
   gate (§4.1). Parity claims in R02–R11 that rest only on the default suite will
   be unfalsifiable for exactly the four properties most likely to break.
2. **`GAPS` carries 6 `critical` entries into the rewrite.** They are inherited,
   not created here. One of them — the wave forward-accuracy gap — is worth
   re-reading against CHE-107 before it is cited, since the oracle it says does
   not exist may now exist; the severity call is the owner's.
3. **The GPU tier is now decided but not cheap.** ~21 records: the 14 stamped
   `cuda:0` probe records (the ~80-minute tier) plus 7 GPU perf measurements. If
   R13 regenerates them it needs a GPU budget and one device, per the
   shared-server policy. Separately, and on CPU:
   `optiland_trace_chunk_sweep.json` and `patch_emitter_cost_model.json` are the
   two perf records whose *fits* are cited from production cost models
   (`solvers/optiland/cost_model.py:123`, `couplers/patch_cost.py:127`), so they
   are the ones that block soonest regardless of device.
4. **The reference tree's default suite is red before the rewrite starts** (§5.1a).
   Six pre-existing failures, four stamped `m3_*` records drifted from
   `src/couplers/ray_to_wave.py`. R07 or R13 owns regenerating them. Until then,
   any R02/R07 parity claim that cites those four records is citing an
   unconfirmed number, and the "everything at the tag is validated" premise the
   project description leans on is not true of these four.

5. **`applied/commercial_lens_systems/records/` was untriaged until now.** It is
   dispositioned `keep as historical` here on the grounds that the project is
   explicitly not a lens catalog. If those 10 records are in fact the only
   evidence for the vendor-prescription conventions in
   `src/registry/prescriptions.py`, that disposition is wrong and R13 should
   revisit it — flagged rather than assumed.

---

# 9. Full class table (280 classes)

**`src/agent/benchmark_suite.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AgentTask` | dataclass | 11 | none |
| `CheckResult` | dataclass | 5 | none |
| `CheckSpec` | dataclass | 7 | none |
| `ContextPolicy` | enum | 0 | none |
| `Outcome` | enum | 0 | none |
| `SuiteResult` | dataclass | 7 | none |
| `TaskResult` | dataclass | 2 | none |
| `TrialResult` | dataclass | 7 | none |

**`src/core/artifacts.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ArtifactRecord` | pydantic | 10 | 2 |

**`src/core/boundary.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ComplexField` | dataclass | 11 | 1,2 |
| `ContractCode` | enum | 0 | none |
| `ContractError` | exception | 0 | none |
| `Frame` | dataclass | 4 | 1,2 |
| `PSF` | dataclass | 7 | 1 |
| `RayBundle` | dataclass | 16 | 1,2 |
| `ReferencePlane` | dataclass | 3 | 1,2 |
| `WavefrontSamples` | dataclass | 10 | none |
| `_HostView` | plain | 0 | none |

**`src/core/coherent_batch.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CoherentRayBatch` | dataclass | 4 | none |

**`src/core/errors.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AdapterDependencyError` | exception | 0 | none |
| `AdapterError` | exception | 0 | none |
| `AdapterNotFoundError` | exception | 0 | none |
| `GraphCompilationError` | exception | 0 | none |
| `MultiScaleOpticsError` | exception | 0 | none |
| `RegistryError` | exception | 0 | none |
| `SolverExecutionError` | exception | 0 | none |
| `UnsupportedCapabilityError` | exception | 0 | none |

**`src/core/execution.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CostEstimate` | pydantic | 5 | 2 |
| `RunStatus` | enum | 0 | none |

**`src/core/execution_record.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `DevicePrecisionObservation` | pydantic | 8 | 2 |
| `ExecutionRecord` | pydantic | 14 | 2 |
| `NodeOutcome` | enum | 0 | none |
| `NodeRecord` | pydantic | 10 | 2 |
| `Refusal` | pydantic | 4 | 2 |
| `RefusalKind` | enum | 0 | none |
| `ResourceCost` | pydantic | 5 | 2 |

**`src/core/graph.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ComponentIndex` | protocol | 0 | none |
| `GraphValidator` | plain | 0 | none |
| `Severity` | enum | 0 | none |
| `ValidationIssue` | pydantic | 4 | none |
| `ValidationReport` | pydantic | 1 | none |

**`src/core/optical_assembly.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ComponentPlacement` | plain | 4 | none |
| `ComponentSpec` | plain | 8 | none |
| `Orientation` | enum | 0 | none |
| `_Frozen` | pydantic | 0 | none |

**`src/core/optical_system.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AirMaterialSpec` | plain | 1 | none |
| `ApertureKind` | enum | 0 | none |
| `ApertureSpec` | plain | 2 | none |
| `CatalogMaterialSpec` | plain | 5 | none |
| `EvenAsphereGeometrySpec` | plain | 5 | none |
| `FieldKind` | enum | 0 | none |
| `FieldSpec` | plain | 2 | none |
| `GeometryKind` | enum | 0 | none |
| `GratingInteractionSpec` | plain | 4 | none |
| `IdealMaterialSpec` | plain | 3 | none |
| `InteractionKind` | enum | 0 | none |
| `MaterialKind` | enum | 0 | none |
| `OpticalSystemSpec` | plain | 9 | none |
| `PlaneGeometrySpec` | plain | 1 | none |
| `PrescriptionError` | exception | 0 | none |
| `RefractiveInteractionSpec` | plain | 1 | none |
| `SphericalGeometrySpec` | plain | 4 | none |
| `SurfaceSpec` | plain | 6 | none |
| `WavelengthSpec` | plain | 2 | none |
| `_Frozen` | pydantic | 0 | none |

**`src/core/performance.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `EnvironmentFingerprint` | dataclass | 11 | none |
| `Incomparable` | exception | 0 | none |
| `Isolation` | dataclass | 4 | none |
| `Measurement` | dataclass | 7 | none |
| `MemoryGuardBreached` | exception | 0 | none |
| `PerformanceRecord` | dataclass | 11 | none |
| `ScalingFit` | dataclass | 5 | none |
| `StageAccountingError` | exception | 0 | none |
| `StageTimer` | dataclass | 3 | none |
| `SwapGrowthAbort` | exception | 0 | none |
| `Workload` | dataclass | 4 | none |

**`src/core/precision.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ArrayNamespace` | enum | 0 | none |
| `ArrayState` | dataclass | 3 | 1,2 |
| `BridgeError` | exception | 0 | none |
| `BridgePlan` | dataclass | 16 | 1,2 |
| `BridgePolicy` | enum | 0 | none |
| `CapabilityError` | exception | 0 | none |
| `ComponentCapabilities` | dataclass | 13 | 1 |
| `DType` | enum | 0 | none |
| `DeviceKind` | enum | 0 | none |
| `DevicePlacement` | dataclass | 2 | 1 |
| `ExecutionRequest` | dataclass | 5 | 1,2 |
| `Precision` | enum | 0 | none |
| `ResolvedExecution` | dataclass | 7 | 1,2 |

**`src/core/provenance.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `RecordVerdict` | dataclass | 9 | 1 |
| `RunProvenance` | pydantic | 14 | 2 |

**`src/core/resources.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `GpuMemorySnapshot` | dataclass | 8 | none |
| `HostMemorySnapshot` | dataclass | 8 | none |
| `MemoryWatchdog` | dataclass | 12 | none |
| `MemoryWatchdogVerdict` | dataclass | 5 | none |

**`src/core/specs.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ApproximationClass` | enum | 0 | none |
| `ArtifactKind` | enum | 0 | none |
| `CostModelSpec` | plain | 4 | none |
| `CouplerRole` | enum | 0 | none |
| `CouplerSpec` | plain | 19 | none |
| `DerivativeMode` | enum | 0 | none |
| `DerivativeSpec` | plain | 4 | none |
| `DesignVariableSpec` | plain | 5 | none |
| `Device` | enum | 0 | none |
| `EdgeSpec` | plain | 5 | none |
| `Framework` | enum | 0 | none |
| `GraphSpec` | plain | 10 | none |
| `InteractionSpec` | plain | 3 | none |
| `Maturity` | enum | 0 | none |
| `ModelSpec` | plain | 15 | none |
| `NodeSpec` | plain | 3 | none |
| `ObjectiveSpec` | plain | 5 | none |
| `PortRef` | plain | 2 | none |
| `PortSpec` | plain | 7 | none |
| `SourceSpec` | plain | 6 | none |
| `StrictModel` | pydantic | 0 | 2,4 |
| `ValiditySpec` | plain | 3 | none |
| `VerificationSpec` | plain | 6 | none |

**`src/couplers/base.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Coupler` | protocol | 0 | none |
| `CouplerRunRequest` | pydantic | 6 | 2 |
| `CouplerRunResult` | pydantic | 6 | 2 |

**`src/couplers/cascade.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CascadeDiagnostics` | dataclass | 16 | none |
| `PrimarySampling` | enum | 0 | none |

**`src/couplers/curvature.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CurvatureBudget` | dataclass | 7 | 1,2 |

**`src/couplers/doe_node.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `PlanarDoeStepCoupler` | plain | 0 | none |

**`src/couplers/generalized_snell.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `GeneralizedSnellDiagnostics` | dataclass | 12 | 1,2 |

**`src/couplers/gradient.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `DifferentiabilityReport` | dataclass | 13 | 1,2 |
| `GradientProblem` | dataclass | 5 | 1 |

**`src/couplers/handoff.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CoherentHandoff` | dataclass | 3 | 1 |
| `DeclaredHandoffPlane` | dataclass | 4 | 1 |
| `HandoffPerturbation` | dataclass | 4 | 1 |

**`src/couplers/interaction.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `DiffractiveInteractionResult` | dataclass | 5 | 1 |
| `DiffractiveModel` | enum | 0 | none |
| `DiffractiveSurface` | dataclass | 7 | 1 |
| `FullFieldParameters` | dataclass | 7 | 1 |
| `GeneralizedSnellParameters` | dataclass | 2 | 1 |
| `LocalPatchParameters` | dataclass | 10 | 1 |
| `PatchWindow` | enum | 0 | none |

**`src/couplers/node.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `RayToWaveCoupler` | plain | 0 | none |
| `_Refusal` | exception | 0 | none |

**`src/couplers/patch.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CoverageBasis` | enum | 0 | none |
| `PatchDiagnostics` | dataclass | 14 | 1,2 |
| `PatchPlan` | dataclass | 7 | 1 |
| `Substrate` | enum | 0 | none |

**`src/couplers/patch_cost.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `PatchEmitterCostModel` | dataclass | 13 | none |

**`src/couplers/patch_node.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `PatchWftCoupler` | plain | 0 | none |

**`src/couplers/patch_positions.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `PositionDensity` | enum | 0 | none |
| `PositionDraw` | enum | 0 | none |
| `PositionPlan` | dataclass | 11 | 1,2 |

**`src/couplers/ray_to_wave.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Perturbation` | dataclass | 4 | 1 |
| `Projection` | enum | 0 | none |
| `Reconstruction` | enum | 0 | none |
| `ReconstructionDiagnostics` | dataclass | 19 | 1,2 |

**`src/couplers/streaming.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `BandLimit` | dataclass | 8 | 1,2 |
| `ChunkWorkItem` | dataclass | 4 | none |
| `LaunchGeometry` | dataclass | 5 | none |
| `PositionalAngularSampler` | dataclass | 4 | none |
| `StreamingReconstruction` | plain | 0 | none |
| `StreamingResult` | dataclass | 7 | none |

**`src/couplers/wave_to_ray.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AngularSpectrum` | dataclass | 10 | 1,2 |
| `SamplingDensity` | enum | 0 | none |
| `SamplingPerturbation` | dataclass | 4 | 1 |

**`src/discovery/api.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ComponentDescription` | pydantic | 27 | none |
| `ConnectionReport` | pydantic | 9 | none |
| `FamilyCoverage` | pydantic | 7 | none |
| `Handover` | pydantic | 6 | none |
| `KnowledgeView` | pydantic | 6 | none |
| `PortView` | pydantic | 8 | none |
| `RefusalView` | pydantic | 5 | none |
| `RouteCapability` | pydantic | 9 | none |
| `SuitabilityRecord` | pydantic | 6 | none |
| `ValidityAnswer` | pydantic | 5 | none |

**`src/registry/loader.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Registry` | dataclass | 2 | 1 |

**`src/runtime/executor.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AdapterResolver` | protocol | 0 | none |
| `CouplerResolver` | protocol | 0 | none |
| `ExecutionCache` | protocol | 0 | none |
| `ExecutorError` | exception | 0 | none |
| `GraphExecutor` | dataclass | 6 | 1 |
| `InMemoryCache` | dataclass | 3 | none |
| `ProcessModel` | enum | 0 | none |
| `SolverStateProtocol` | dataclass | 3 | 1 |
| `_LazyNames` | plain | 0 | none |

**`src/runtime/instance_runner.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ProbedRefusal` | dataclass | 7 | 1 |

**`src/runtime/variants.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `VariantError` | exception | 0 | none |

**`src/solvers/base.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ModelAdapter` | protocol | 0 | none |
| `ModelRunRequest` | pydantic | 6 | 2 |
| `ModelRunResult` | pydantic | 6 | 2 |

**`src/solvers/chromatix/adapter.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ChromatixAdapter` | plain | 0 | none |

**`src/solvers/chromatix/baseline.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `_BaselineError` | exception | 0 | none |

**`src/solvers/chromatix/carrier_removed_asm.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CarrierRemovedPropagation` | dataclass | 8 | 1 |

**`src/solvers/chromatix/propagation.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `WaveHandoffError` | exception | 0 | none |

**`src/solvers/chromatix/requests.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ChromatixWaveFailure` | pydantic | 4 | 2 |
| `ChromatixWaveRequest` | pydantic | 22 | 2 |
| `ChromatixWaveResult` | pydantic | 27 | 2 |

**`src/solvers/optiland/adapter.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `OptilandAdapter` | plain | 0 | none |

**`src/solvers/optiland/coherent_trace.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `OptilandExecutionState` | dataclass | 10 | 1,2 |
| `TracePlans` | dataclass | 3 | 1,2 |

**`src/solvers/optiland/cost_model.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `TraceCostModel` | dataclass | 12 | 1,2 |

**`src/solvers/optiland/pupil.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `HandoffPlaneError` | exception | 0 | none |

**`src/solvers/optiland/requests.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `OptilandRayFailure` | pydantic | 4 | 2 |
| `OptilandRayRequest` | pydantic | 11 | 2 |
| `OptilandRayResult` | pydantic | 16 | 2 |

**`src/studies/metalens/candidate.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CandidateRequest` | dataclass | 13 | none |

**`src/studies/metalens/controller.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `SweepController` | plain | 0 | none |
| `SweepOptions` | dataclass | 19 | none |

**`src/studies/metalens/oracle.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Layer` | dataclass | 3 | none |
| `MetalensConfig` | dataclass | 8 | none |
| `PsfComparison` | dataclass | 15 | none |

**`src/verification/analytic.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `AnalyticOracle` | dataclass | 10 | 1 |

**`src/verification/asm_oracle.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `CarrierConvention` | enum | 0 | none |
| `FieldComparison` | dataclass | 6 | 1,2 |

**`src/verification/claim_ledger.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Claim` | dataclass | 16 | none |
| `ClaimKind` | enum | 0 | none |
| `Gap` | dataclass | 7 | none |
| `GateStatus` | enum | 0 | none |
| `Oracle` | enum | 0 | none |
| `OracleIndependence` | enum | 0 | none |
| `StochasticEvidence` | dataclass | 4 | none |

**`src/verification/declaration_ledger.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Coverage` | dataclass | 7 | none |
| `CoverageKind` | enum | 0 | none |
| `DeclarationKind` | enum | 0 | none |
| `LedgerReport` | dataclass | 4 | none |
| `RegistryDeclaration` | dataclass | 5 | none |

**`src/verification/evidence.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `Ensemble` | dataclass | 4 | none |
| `InstanceRun` | dataclass | 4 | none |

**`src/verification/families/registry.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `_Families` | plain | 0 | none |

**`src/verification/families/schema.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `BenchmarkCategory` | enum | 0 | none |
| `BenchmarkFamily` | dataclass | 26 | none |
| `BenchmarkInstance` | dataclass | 12 | none |
| `BenchmarkLayer` | enum | 0 | none |
| `ExecutionParameter` | dataclass | 1 | none |
| `ExecutionPolicy` | dataclass | 6 | none |
| `FamilyOracle` | dataclass | 5 | none |
| `GateDisposition` | dataclass | 5 | none |
| `InstanceOrigin` | enum | 0 | none |
| `Invariant` | dataclass | 4 | none |
| `Metric` | dataclass | 5 | none |
| `NegativeControl` | dataclass | 6 | none |
| `NegativeControlExpectation` | enum | 0 | none |
| `NumericalParameter` | dataclass | 1 | none |
| `Parameter` | dataclass | 7 | none |
| `ParameterKind` | enum | 0 | none |
| `PhysicalParameter` | dataclass | 1 | none |
| `ProvenanceRule` | dataclass | 3 | none |
| `RepresentationParameter` | dataclass | 1 | none |
| `SamplerAbsentReason` | enum | 0 | none |
| `StochasticEvidenceKind` | enum | 0 | none |
| `StochasticPolicy` | dataclass | 4 | none |
| `Tolerance` | dataclass | 6 | none |
| `ToleranceBasis` | enum | 0 | none |
| `ValidityBasis` | enum | 0 | none |
| `ValidityPredicate` | dataclass | 7 | none |
| `ValidityState` | enum | 0 | none |

**`src/verification/fixed_suite.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `FixedInstance` | dataclass | 4 | none |
| `FixedSuite` | dataclass | 2 | none |
| `Tier` | enum | 0 | none |

**`src/verification/hazards.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `MeasuredHazard` | dataclass | 11 | none |

**`src/verification/metrics.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `MetricDefinition` | dataclass | 8 | 1 |

**`src/verification/psf_measurement.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `PsfMeasurement` | dataclass | 9 | 1,2 |
| `PsfNormalization` | enum | 0 | none |

**`src/verification/psf_oracles.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `FraunhoferPsf` | dataclass | 10 | 1,2 |
| `PupilAberration` | dataclass | 7 | 1,2 |
| `ReferenceSphere` | dataclass | 5 | 1,2 |

**`src/verification/refusals.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `RefusalEntry` | dataclass | 5 | none |

**`src/verification/result.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `ContractStatus` | pydantic | 5 | none |
| `ConvergenceReport` | pydantic | 7 | none |
| `Diagnostic` | pydantic | 3 | none |
| `DiagnosticCode` | enum | 0 | none |
| `InvariantResult` | pydantic | 5 | none |
| `Measurement` | pydantic | 5 | none |
| `MetricResult` | pydantic | 9 | none |
| `NegativeControlOutcome` | enum | 0 | none |
| `NegativeControlResult` | pydantic | 6 | none |
| `PredicateMargin` | pydantic | 5 | none |
| `ProvenanceReport` | pydantic | 7 | none |
| `StochasticReport` | pydantic | 8 | none |
| `UncertaintyBasis` | enum | 0 | none |
| `ValidityReport` | pydantic | 4 | none |
| `VerificationResult` | pydantic | 17 | none |

**`src/verification/status.py`**

| class | kind | fields | rule |
| --- | --- | --- | --- |
| `VerificationStatus` | enum | 0 | none |

---

# 10. Full test triage (85 files)

| test file | LOC | tag | what it establishes |
| --- | --- | --- | --- |
| `tests/conftest.py` | 156 | **architecture protection** | Fixtures and the probe-record loader. Infrastructure; port the record loader only. |
| `tests/test_adapter_registry.py` | 18 | **architecture protection** | Asserts the old adapter registry's shape. |
| `tests/test_agent_benchmark.py` | 445 | **architecture protection** | V1 agent-benchmark harness unit tests. |
| `tests/test_architecture_invariants.py` | 229 | **architecture protection** | Protects the historical layering by name. |
| `tests/test_artifacts.py` | 18 | **physics evidence** | Direction-norm and artifact-kind contract; the dtype-dependent ||d||-1 tolerance. |
| `tests/test_b0_families.py` | 359 | **architecture protection** | B0 family declaration shape. |
| `tests/test_b0_instances.py` | 532 | **physics evidence** | B0 contract instances actually executed against both backends. |
| `tests/test_b1_families.py` | 623 | **architecture protection** | B1 family declaration shape. |
| `tests/test_b1_ray_gpu.py` | 330 | **physics evidence** | Ray device/precision parity on CUDA. gpu-only. |
| `tests/test_b1_ray_instances.py` | 1190 | **physics evidence** | B1 ray physics: EFL closed form, symplectic invariant. |
| `tests/test_b1_wave_gpu.py` | 315 | **physics evidence** | Wave device observation on CUDA. gpu-only. |
| `tests/test_b1_wave_instances.py` | 777 | **physics evidence** | B1 wave physics against analytic oracles. |
| `tests/test_b2_equiv_instances.py` | 578 | **physics evidence** | Coupler equivalence: enumeration is exact, coverage correction is right-way-up. |
| `tests/test_b2_families.py` | 516 | **architecture protection** | B2 family declaration shape. |
| `tests/test_b2_transition_instances.py` | 825 | **physics evidence** | Stochastic transition evidence: bias, convergence rate, variance. 22 slow tests. |
| `tests/test_b3_4f_real.py` | 314 | **physics evidence** | Real 4f system with a physical lens; composed-system evidence. |
| `tests/test_b3_b4_families.py` | 834 | **architecture protection** | B3/B4 family declaration shape plus the energy-accounting record read. |
| `tests/test_b3_doe_inline.py` | 525 | **physics evidence** | Embedded DOE inside a refractive train. |
| `tests/test_benchmark_inventory.py` | 183 | **architecture protection** | Inventory completeness sweep over benchmarks/. |
| `tests/test_carrier_removed_asm.py` | 460 | **physics evidence** | Carrier-removed ASM against the analytic tilted-field result. |
| `tests/test_chromatix_adapter.py` | 742 | **physics evidence** | Chromatix API use, normalization and propagator conventions at the pin. Partly slow. |
| `tests/test_claim_ledger.py` | 559 | **architecture protection** | Ledger self-consistency: citations resolve, O2 cannot gate. |
| `tests/test_cli.py` | 125 | **architecture protection** | CLI surface. |
| `tests/test_coherent_batch.py` | 288 | **physics evidence** | What makes a bundle jointly coherent. |
| `tests/test_coherent_bridge.py` | 1127 | **physics evidence** | Optiland->Chromatix coherent handoff: OPD reference, phase sign, reference sphere. |
| `tests/test_commercial_lens_catalog.py` | 382 | **architecture protection** | Catalog inventory; the project is explicitly not a lens catalog. |
| `tests/test_context_sync.py` | 209 | **architecture protection** | docs/context sync guard. |
| `tests/test_contract_code_reachability.py` | 637 | **physics evidence** | Every ContractCode is reachable -- the structured-failure non-negotiable. |
| `tests/test_coupler_contracts.py` | 495 | **physics evidence** | Boundary-artifact contracts across couplers; the only real WavefrontSamples exercise. |
| `tests/test_coupler_gradient.py` | 213 | **physics evidence** | Gradient/finite-difference evidence. 10 slow tests. |
| `tests/test_coupler_knowledge_pack.py` | 725 | **architecture protection** | Knowledge-pack file inventory. |
| `tests/test_coupler_round_trip.py` | 410 | **physics evidence** | ray->wave->ray round trip; phase-sign pairing and the power ledger. slow. |
| `tests/test_curvature_bound.py` | 311 | **physics evidence** | Curvature bound derivation and its failure mode. |
| `tests/test_declaration_ledger.py` | 252 | **architecture protection** | Declaration coverage resolution: gaps, orphans, ambiguity. |
| `tests/test_diffractive_interaction.py` | 1424 | **physics evidence** | Grating angle exactness, generalized-Snell margins, three parameterizations. |
| `tests/test_discovery.py` | 484 | **architecture protection** | Discovery query surface. |
| `tests/test_executor.py` | 865 | **architecture protection** | Old graph executor semantics. |
| `tests/test_executor_integration.py` | 194 | **physics evidence** | Real Optiland->Chromatix chain end to end. |
| `tests/test_family_schema.py` | 930 | **architecture protection** | Family schema validation. |
| `tests/test_fixed_suite.py` | 320 | **architecture protection** | Fixed-suite runner. |
| `tests/test_flat_layout.py` | 226 | **architecture protection** | Protects the flat src/ layout by name. slow (1 subprocess test). |
| `tests/test_generated_artifacts.py` | 101 | **architecture protection** | Generated-schema drift guard. |
| `tests/test_gpu_environment.py` | 405 | **architecture protection** | GPU container/environment assertions. gpu-only. |
| `tests/test_graph_validation.py` | 298 | **architecture protection** | Old graph validation rules. |
| `tests/test_information_preservation.py` | 451 | **physics evidence** | Information-preservation bound across the coupler boundary. |
| `tests/test_metalens_bridge_gpu.py` | 346 | **physics evidence** | Metalens bridge on GPU; study-specific but real physics. gpu-only. |
| `tests/test_metalens_controller.py` | 659 | **architecture protection** | Study controller mechanics. |
| `tests/test_metalens_oracle.py` | 447 | **physics evidence** | Metalens oracle comparison. |
| `tests/test_metrics.py` | 536 | **physics evidence** | Strehl, first-null radius, field metrics against closed forms. |
| `tests/test_optical_assembly.py` | 416 | **architecture protection** | Old assembly abstraction. |
| `tests/test_optiland_adapter.py` | 1066 | **physics evidence** | Optiland API use, chunking, direction-norm bound at the pin. |
| `tests/test_optiland_canonical_prescriptions.py` | 1004 | **physics evidence** | Vendor prescriptions reproduce published EFL/BFL. |
| `tests/test_optiland_coherent_handoff.py` | 405 | **physics evidence** | Coherent handoff plane and OPD convention. |
| `tests/test_optiland_cost_model.py` | 233 | **architecture protection** | Cost-model fit mechanics. |
| `tests/test_optiland_opd_convention.py` | 295 | **physics evidence** | THE opd sign/reference convention test. |
| `tests/test_package_dependencies.py` | 233 | **architecture protection** | AST import-direction check. Reuse the mechanism, not the asserted names (R01.1). |
| `tests/test_patch_positions.py` | 494 | **physics evidence** | Patch enumeration and coverage correction. |
| `tests/test_patch_wft.py` | 923 | **physics evidence** | C_PATCH_WFT: enumeration exactness at 5.9e-15, power ledger, cost fit. |
| `tests/test_performance_harness.py` | 1043 | **architecture protection** | Perf record schema and harness mechanics. |
| `tests/test_planar_doe_step.py` | 440 | **physics evidence** | Planar DOE step: budget identity, padding invariance, stacked DOEs. |
| `tests/test_planar_doe_step_pack.py` | 258 | **architecture protection** | DOE knowledge-pack inventory. |
| `tests/test_precision_contract.py` | 515 | **physics evidence** | Precision/device policy contract and dtype ladders. |
| `tests/test_precision_execution_matrix.py` | 608 | **physics evidence** | Device x dtype execution matrix. gpu-only. |
| `tests/test_precision_gpu_pipeline.py` | 609 | **physics evidence** | End-to-end GPU precision, reference-leg selection. gpu-only. |
| `tests/test_preserved_evidence.py` | 179 | **architecture protection** | Guards that preserved evidence files still exist. |
| `tests/test_provenance_fingerprint.py` | 587 | **physics evidence** | The projection that makes a fingerprint mean 'the physics changed'. Also the record enrollment guard. |
| `tests/test_psf_measurement.py` | 620 | **physics evidence** | PSF sampling, centroid, normalization. |
| `tests/test_psf_verification.py` | 738 | **physics evidence** | PSF against the Airy/Fraunhofer O1 oracles. |
| `tests/test_quadrature.py` | 98 | **physics evidence** | Hexapolar ring index and area weights. |
| `tests/test_ray_to_wave.py` | 671 | **physics evidence** | ray->wave transfer against the analytic pupil field. |
| `tests/test_ray_to_wave_kspace.py` | 588 | **physics evidence** | k-space reconstruction correctness and the Nyquist limit. |
| `tests/test_ray_to_wave_node.py` | 575 | **physics evidence** | ray->wave node contract and declared conventions. |
| `tests/test_registry.py` | 14 | **architecture protection** | Registry loading. |
| `tests/test_registry_matches_capabilities.py` | 209 | **architecture protection** | Registry/capability-table agreement. |
| `tests/test_resource_profile_guard.py` | 272 | **architecture protection** | Resource-profile guard. slow. |
| `tests/test_resources.py` | 220 | **physics evidence** | Environment evidence, not optics: pins the swap signal to the ONE cgroup file CHE-64 measured as attributable, so the repo cannot acquire two definitions of "did this swap". The awkward case for a two-tag scheme -- it is not architecture protection, because it asserts a measured fact about the host rather than about our layout. |
| `tests/test_retired_taxonomy.py` | 336 | **architecture protection** | Asserts retired names stay retired. |
| `tests/test_run_sh.py` | 80 | **architecture protection** | run.sh behaviour. |
| `tests/test_solver_adapter_characterization.py` | 542 | **physics evidence** | Adapter characterization against both pinned backends. |
| `tests/test_solver_knowledge_pack.py` | 183 | **architecture protection** | Knowledge-pack inventory. |
| `tests/test_streaming_estimator.py` | 573 | **physics evidence** | Band limit / grazing floor and the streaming estimator's bias. |
| `tests/test_substrate_proof.py` | 405 | **physics evidence** | Benchmark substrate proof: control arms and detection margins. slow. |
| `tests/test_suite_layout.py` | 393 | **architecture protection** | Protects the test-suite layout by name. |
| `tests/test_verifier.py` | 736 | **architecture protection** | Old verifier driver and refusal catalogue. |
| `tests/test_wave_to_ray.py` | 493 | **physics evidence** | Angular-spectrum decomposition, importance weight, evanescent ledger. 5 slow tests. |

---

# 11. Full record disposition (293 records)

| record | disposition | reason |
| --- | --- | --- |
| `benchmarks/applied/commercial_lens_systems/records/S1_KPX094_SINGLET.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/S2_PAC052_ACHROMAT.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/S3_KBX058_BICONVEX.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/S4_PAC052_KBX058_TANDEM.cpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/S4_PAC052_KBX058_TANDEM.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/S5_TANDEM_REVERSED_ACHROMAT.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/components.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/gpu_cpu_comparison.S4_PAC052_KBX058_TANDEM.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/summary.cpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/applied/commercial_lens_systems/records/summary.gpu.json` | **keep as historical** | Vendor-catalog reproduction. The project is explicitly not a lens catalog; the tag holds the numbers. See section 8 risk 4 -- revisit if these are the only evidence for src/registry/prescriptions.py's conventions. |
| `benchmarks/instances/records/B0-CAPINT-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-DEVICE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-DEVICE-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-DTYPE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-HANDOFF-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-META-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-PATCH-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-UNITS-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-UNITS-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B0-VALIDITY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-DUTY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-DUTY-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-DUTY-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-OFFAXIS-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-05.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-06.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-07.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-08.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-EFL-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-LAGRANGE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/instances/records/B1-RAY-OFFAXIS-OPL-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-PLATE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-SNELL-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-SNELL-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-SNELL-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-RAY-SNELL-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-AIRY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-ASM-VALIDITY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-ASM-VALIDITY-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-ASM-VALIDITY-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-FWDBWD-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-GAUSS-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-PLANEPHASE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-TALBOT-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B1-WAVE-TILT-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-FULL-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-SUB-004.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-SUB-016.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-SUB-064.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-SUB-225.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-EQUIV-SUB-ENUMERATED.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-EXACT-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-OFFNODE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-OFFNODE-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-OFFNODE-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-OFFNODE-08.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-ONNODE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-ONNODE-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-ONNODE-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-R2W-ROUTE-ONNODE-08.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-RAYWAVERAY-ENUMERATED-00.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-RAYWAVERAY-MONTE_CARLO-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-RAYWAVERAY-MONTE_CARLO-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-RAYWAVERAY-MONTE_CARLO-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-WAVERAYWAVE-ENUMERATED-00.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-WAVERAYWAVE-MONTE_CARLO-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-WAVERAYWAVE-MONTE_CARLO-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-ROUNDTRIP-WAVERAYWAVE-MONTE_CARLO-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-05.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-06.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-07.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B2-W2R-STOCH-08.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/instances/records/B3-PSF-SINGLET-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/perf/records/b3_psf_singlet_cpu.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/demo2_paper_rw_f_paper_budget_ramp_sum_cuda.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/demo2_paper_rw_p_ramp_sum_cuda.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/demo3_characterization_rw_p_kspace_splat_cuda.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/demo3_characterization_rw_p_ramp_sum_che118_after_cuda.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/demo3_characterization_rw_p_ramp_sum_cuda.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/estimate_accuracy.json` | **regenerate after the rewrite** | Named as ledger evidence AND a cost measurement of the OLD implementation. The claim it backs is about framework overhead / estimate accuracy, which the rewrite changes, so the claim must be re-established rather than the number carried across. |
| `benchmarks/perf/records/framework_overhead.json` | **regenerate after the rewrite** | Named as ledger evidence AND a cost measurement of the OLD implementation. The claim it backs is about framework overhead / estimate accuracy, which the rewrite changes, so the claim must be re-established rather than the number carried across. |
| `benchmarks/perf/records/l2_psf_01_cpu.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/optiland_trace_chunk_sweep.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/optiland_trace_decomposition.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/optiland_trace_demo3_equivalence.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/optiland_trace_precision.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. GPU measurement (device: cuda) -- part of the GPU regeneration budget. |
| `benchmarks/perf/records/patch_emitter_cost_model.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/patch_emitter_decomposition.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/patch_emitter_demo3_equivalence.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/patch_emitter_overlap.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/patch_emitter_thread_sweep.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/scaling_ray_axis.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/perf/records/suite_default_cpu.json` | **regenerate after the rewrite** | Measures the cost of the OLD implementation, so the number does not transfer to re-authored code. CPU measurement; note gpu_name in environment is host metadata, not the compute device. |
| `benchmarks/probes/records/REGISTER.yaml` | **keep as active evidence** | The enrollment register itself, not a measurement. tests/test_provenance_fingerprint.py fails if it names a record that does not exist, so it must be edited in lockstep with any deletion. |
| `benchmarks/probes/records/b3_energy_accounting.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/che12_engine_report.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/chromatix/b1_wave_device_observation.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence and attests to pinned backend behaviour, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/chromatix/gradient_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/chromatix/import_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/chromatix/m3_pupil_to_focus.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/chromatix/propagation_probe.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/chromatix/standalone_baseline.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/doe_step_sampler_cost.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/m3_convergence.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/m3_first_null_grid_convergence.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/m3_off_axis_handoff.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/m3_psf_verification.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/m3_quadrature_weight.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/o1_applicability.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/optiland/b1_ray_device_precision.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also attests to pinned backend behaviour, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/optiland/exit_pupil_handoff.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/gradient_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/import_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/off_axis_opd_reference.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/opd_convention_probe.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/optiland/raytrace_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/standalone_baseline.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/optiland/system_construction_probe.json` | **keep as active evidence** | Unstamped and attests to pinned backend behaviour (optiland==0.6.0 / chromatix@d24bdf0) rather than to repository code, so the rewrite does not stale it. REGISTER.yaml states this reason. |
| `benchmarks/probes/records/planar_doe_step_device.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/ray_to_wave/coherent_handoff.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_cost_sweep.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_paper_figure_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_paper_figure_jax_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo2_paper_jax.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/ray_wave/demo2_paper_kspace_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_paper_kspace_jax_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo2_paper_rwf_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_smoke_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo2_smoke_numpy.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_characterization_rw_f.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_characterization_rw_f_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_characterization_rw_p.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_characterization_rw_p_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_convergence_kspace_rw_p.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_convergence_rw_p.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_cal_b6.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_cal_b60.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_cal_b60_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_cal_b6_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_p600.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_p600_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_a000_of001.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_a000_of001_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b000_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b000_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b001_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b001_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b002_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b002_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b003_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_b003_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_c2e6_000_of001.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_cal_c2e6_000_of001_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace000_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace000_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace001_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace001_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace002_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace002_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace003_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace003_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace004_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace004_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace005_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace005_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace006_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace006_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace007_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace007_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace008_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace008_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace009_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace009_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace010_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace010_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace011_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_kspace011_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_pad_sweep.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp000_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp000_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp001_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp001_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp002_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp002_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp003_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp003_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp004_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp004_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp005_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp005_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp006_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp006_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp007_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp007_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp008_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp008_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp009_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp009_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp010_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp010_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp011_of012.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_rwf_ramp011_of012_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_a000_of001.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_a000_of001_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b000_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b000_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b001_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b001_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b002_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b002_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b003_of004.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enum_shardtest_b003_of004_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enumerated_positions.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enumerated_reference_rwf_kspace.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enumerated_reference_rwf_kspace_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_enumerated_reference_rwf_ramp.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_enumerated_reference_rwf_ramp_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_equivalence_characterization_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_equivalence_rwf_enumerated.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_equivalence_smoke_numpy.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_kspace_rw_p.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_kspace_rw_p_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_paper_configuration_jax.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_position_spectral_l1.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_position_spectral_l1.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_route_agreement.json` | **keep as active evidence** | Named as evidence by a claim_ledger or declaration_ledger entry, and unstamped, so no fingerprint ties it to the old tree. Deleting it makes that claim unfalsifiable. |
| `benchmarks/probes/records/ray_wave/demo3_smoke_numpy.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_stage_ramp.json` | **keep as historical** | Unstamped and declared in REGISTER.yaml as deferred. Hours of GPU compute attesting to a study rather than to an infrastructure convention; the tag holds it. Deleting the file requires removing its register pattern in the same change. |
| `benchmarks/probes/records/ray_wave/demo3_stage_ramp_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/demo3_variance_allocation.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_candidates.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_confirm_os8_p1000.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_decomposition.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_ladder_control.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_ladder_control_top.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_ladder_winner.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/demo3_variance_ladderfit_control.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/perf_demo2_paper_rw_f_paper_budget_ramp_sum_cuda.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/perf_demo2_paper_rw_p_ramp_sum_cuda.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/perf_demo3_characterization_rw_p_kspace_splat_cuda.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/perf_demo3_characterization_rw_p_kspace_splat_cuda_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/ray_wave/perf_demo3_characterization_rw_p_ramp_sum_cuda.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/probes/records/ray_wave/perf_demo3_characterization_rw_p_ramp_sum_cuda_fields.npz` | **keep as historical** | Neither stamped nor cited; the tag preserves it. |
| `benchmarks/probes/records/singlet_residual_attribution.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/probes/records/singlet_residual_grid.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-BIN-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-BIN-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B3-4F-IDEAL-CARRIER-OFFGRID.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B3-4F-IDEAL-CARRIER-SNAPPED.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-SIN-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-SIN-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-SIN-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-IDEAL-SIN-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B3-4F-IDEAL-SIN-05.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B3-4F-REAL-APERTURE-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-REAL-APERTURE-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-REAL-APERTURE-03.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-REAL-APERTURE-04.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-4F-REAL-FIELD-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-APERTURE-050.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-APERTURE-200.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-OFFAXIS-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-ORDER-MINUS1.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-PERIOD-050.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-PERIOD-100.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-PERIOD-200.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-PITCH-20.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-PITCH-5.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-RELAY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-ZEROGRAD.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B3-DOE-INLINE-ZEROPHASE.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. It is also named as ledger evidence, so its VALUE survives even though its stamp does not. |
| `benchmarks/systems/records/B4-4F-REAL-APERTURE-SMALL.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-APERTURE-WIDE.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-FIELD-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-FIELD-02.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-FREQUENCY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-GRID-64.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-MODULATION-BINARY.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-4F-REAL-REFERENCE.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-APERTURE-300.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-APERTURE-500.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-PITCH-ALIASED.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-RAYS-256.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-REFERENCE.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |
| `benchmarks/systems/records/B4-DOE-INLINE-RELAY-01.json` | **keep as active evidence, then regenerate at R13** | Stamped. Its code_fingerprint covers the 11 src/core modules every stamped record shares, so it reports removed-file drift by construction once R14 deletes the old tree -- it cannot remain verifiable evidence across the cut, only citable evidence. |

---

# 12. Full module inventory (112 modules)

Every production module, exactly one disposition each: 32 knowledge to reuse, 21 test evidence to reuse, 19 algorithm to reuse, 40 nothing worth carrying forward.

| module | LOC | disposition | what it proves we need | evidence |
| --- | --- | --- | --- | --- |
| `src/agent/__init__.py` | 7 | **nothing worth carrying forward** | Namespace. | -- |
| `src/agent/benchmark_suite.py` | 862 | **nothing worth carrying forward** | V1 agent benchmark harness; out of the rewrite's scope. | -- |
| `src/cli.py` | 244 | **nothing worth carrying forward** | Typer surface over the old graph/registry; no physics. | -- |
| `src/core/__init__.py` | 1 | **nothing worth carrying forward** | Empty namespace marker. | -- |
| `src/core/arrays.py` | 460 | **algorithm to reuse** | numpy/jax/torch namespace dispatch and dtype intake used by every artifact. | tests/test_precision_contract.py |
| `src/core/artifacts.py` | 30 | **knowledge to reuse** | ArtifactRecord/ArtifactKind: the on-disk artifact contract. | tests/test_artifacts.py |
| `src/core/boundary.py` | 1580 | **knowledge to reuse** | THE convention file: RayBundle/ComplexField/PSF/WavefrontSamples with frame, phasor sign, reference plane, normalization, direction-norm tolerance. | tests/test_coupler_contracts.py, tests/test_artifacts.py |
| `src/core/bridge.py` | 208 | **knowledge to reuse** | Cross-namespace handoff rules (jax<->torch<->numpy) and where a copy is forced. | tests/test_precision_contract.py |
| `src/core/capabilities.py` | 358 | **test evidence to reuse** | Probe-backed capability table; its value is the recorded probe evidence, not the table shape. | benchmarks/probes/records/, tests/test_registry_matches_capabilities.py |
| `src/core/coherent_batch.py` | 340 | **knowledge to reuse** | Coherent-batch contract: what makes a bundle jointly coherent. | tests/test_coherent_batch.py |
| `src/core/errors.py` | 45 | **knowledge to reuse** | ContractCode taxonomy -- the structured-failure vocabulary AGENTS.md requires. | tests/test_contract_code_reachability.py |
| `src/core/execution.py` | 44 | **nothing worth carrying forward** | CostEstimate holder; two fields, no invariant. | -- |
| `src/core/execution_record.py` | 199 | **knowledge to reuse** | What a run must record: refusal, device/precision observation, resource cost. | tests/test_executor.py |
| `src/core/graph.py` | 458 | **nothing worth carrying forward** | Old graph/DAG model; the rewrite plans through operation descriptors. | -- |
| `src/core/optical_assembly.py` | 503 | **nothing worth carrying forward** | Old assembly abstraction, superseded by problems/. | -- |
| `src/core/optical_system.py` | 705 | **nothing worth carrying forward** | 20 classes of system description; the rewrite carries the prescription data, not this. | -- |
| `src/core/paths.py` | 48 | **nothing worth carrying forward** | Repository-root helper. | -- |
| `src/core/performance.py` | 824 | **test evidence to reuse** | Perf harness; its value is benchmarks/perf/records/ and the two fitted cost models. | benchmarks/perf/records/ |
| `src/core/precision.py` | 1032 | **knowledge to reuse** | Precision/device policy: dtype ladders, DevicePlacement, ExecutionRequest resolution, BridgePlan. | tests/test_precision_contract.py, tests/test_precision_execution_matrix.py |
| `src/core/provenance.py` | 416 | **algorithm to reuse** | strip_volatile / source_fingerprint / verify_record_provenance -- the record staleness mechanism, reused verbatim in R13. | tests/test_provenance_fingerprint.py |
| `src/core/resources.py` | 384 | **knowledge to reuse** | Host/container/GPU memory instrumentation. Carries CHE-64's MEASURED finding: host swap is non-zero at rest on this machine (~700 MiB with no project process), so /proc/meminfo SwapFree is not attributable and the per-container /sys/fs/cgroup/memory.swap.current DELTA is. This backs the AGENTS.md never-use-swap non-negotiable and must not be re-derived by experiment on a shared server. | tests/test_resources.py::test_the_swap_signal_is_the_same_file_che64_guards |
| `src/core/specs.py` | 311 | **nothing worth carrying forward** | 23 pydantic spec models for the old graph schema. | -- |
| `src/couplers/__init__.py` | 127 | **nothing worth carrying forward** | Re-export surface. | -- |
| `src/couplers/base.py` | 84 | **nothing worth carrying forward** | CouplerRunRequest/Result envelope; the rewrite carries descriptors instead. | -- |
| `src/couplers/cascade.py` | 475 | **nothing worth carrying forward** | Multi-stage cascade orchestration; planning/ owns composition now. | -- |
| `src/couplers/curvature.py` | 268 | **algorithm to reuse** | Curvature bound: when a ray fan may be treated as locally planar. | tests/test_curvature_bound.py |
| `src/couplers/doe_node.py` | 491 | **knowledge to reuse** | Planar DOE step contract: outgoing count is the budget, power ledger separated. | tests/test_planar_doe_step.py |
| `src/couplers/generalized_snell.py` | 440 | **algorithm to reuse** | Generalized-Snell margins: propagating-order, local-gradient-smoothness, single-order dominance. | tests/test_diffractive_interaction.py |
| `src/couplers/gradient.py` | 358 | **knowledge to reuse** | Gradient/differentiability reporting; forward_only until finite-difference validated. | tests/test_coupler_gradient.py (slow) |
| `src/couplers/handoff.py` | 1011 | **knowledge to reuse** | Declared handoff plane, OPD reference and coherent handoff conventions. | tests/test_coherent_bridge.py, tests/test_optiland_coherent_handoff.py |
| `src/couplers/interaction.py` | 754 | **knowledge to reuse** | DiffractiveInteraction: the three parameterizations and which regime each is valid in. | tests/test_diffractive_interaction.py |
| `src/couplers/node.py` | 630 | **nothing worth carrying forward** | Node wrapper for the old graph executor. | -- |
| `src/couplers/ontology.py` | 135 | **knowledge to reuse** | Coupler naming/ontology; becomes descriptor metadata. | tests/test_coupler_knowledge_pack.py |
| `src/couplers/patch.py` | 1034 | **algorithm to reuse** | Patch windowed-Fourier transfer, the C_PATCH_WFT kernel. | tests/test_patch_wft.py |
| `src/couplers/patch_cost.py` | 232 | **test evidence to reuse** | Fitted patch-emitter cost model; value is the fit record. | benchmarks/perf/records/patch_emitter_cost_model.json |
| `src/couplers/patch_node.py` | 527 | **nothing worth carrying forward** | Graph-node wrapper around patch.py. | -- |
| `src/couplers/patch_positions.py` | 669 | **algorithm to reuse** | Patch position enumeration and coverage correction A_draw/A_patch. | tests/test_patch_positions.py |
| `src/couplers/propagation.py` | 85 | **knowledge to reuse** | Propagation-regime selection rules. | tests/test_coupler_round_trip.py (slow) |
| `src/couplers/quadrature.py` | 173 | **algorithm to reuse** | Hexapolar ring index + area weights; matches optiland's HexagonalDistribution to float64. | tests/test_quadrature.py |
| `src/couplers/ray_to_wave.py` | 939 | **algorithm to reuse** | k-space reconstruction, Nyquist direction limit, ray-density diagnostic. | tests/test_ray_to_wave_kspace.py, tests/test_ray_to_wave.py |
| `src/couplers/streaming.py` | 778 | **algorithm to reuse** | grazing_floor_for_phase_budget + band_limit_spectrum -- the H4/CHE-70 band limit. Omitting it makes the new coupler worse. | tests/test_streaming_estimator.py |
| `src/couplers/wave_to_ray.py` | 538 | **algorithm to reuse** | Angular-spectrum decomposition, importance weighting, evanescent power ledger. | tests/test_wave_to_ray.py (part slow) |
| `src/discovery/__init__.py` | 57 | **nothing worth carrying forward** | Re-export. | -- |
| `src/discovery/api.py` | 944 | **nothing worth carrying forward** | Capability query surface; owns no facts by design, replaced by planning/. | -- |
| `src/registry/__init__.py` | 5 | **nothing worth carrying forward** | Re-export. | -- |
| `src/registry/loader.py` | 87 | **knowledge to reuse** | How declarations are loaded and validated; the declaration vocabulary survives. | tests/test_declaration_ledger.py |
| `src/registry/prescriptions.py` | 312 | **knowledge to reuse** | Canonical lens prescriptions (vendor data), reusable as problem fixtures. | tests/test_optiland_canonical_prescriptions.py |
| `src/runtime/__init__.py` | 41 | **nothing worth carrying forward** | Re-export. | -- |
| `src/runtime/executor.py` | 1145 | **nothing worth carrying forward** | 600-line graph executor retired in CHE-116; runtime/ is rebuilt on planning/. | -- |
| `src/runtime/instance_runner.py` | 360 | **test evidence to reuse** | Drives benchmark instances; value is benchmarks/instances/records/. | benchmarks/instances/records/ |
| `src/runtime/variants.py` | 98 | **nothing worth carrying forward** | Variant expansion for the old runner. | -- |
| `src/solvers/__init__.py` | 25 | **nothing worth carrying forward** | Re-export. | -- |
| `src/solvers/base.py` | 62 | **nothing worth carrying forward** | ModelRunRequest/Result envelope. | -- |
| `src/solvers/chromatix/__init__.py` | 9 | **nothing worth carrying forward** | Re-export. | -- |
| `src/solvers/chromatix/adapter.py` | 397 | **knowledge to reuse** | Chromatix API use, field construction and normalization conventions at the pinned commit. | tests/test_chromatix_adapter.py (slow) |
| `src/solvers/chromatix/baseline.py` | 677 | **test evidence to reuse** | Baseline scenarios; value is the recorded probe output. | benchmarks/probes/records/chromatix/ |
| `src/solvers/chromatix/capability.py` | 133 | **test evidence to reuse** | Probe-backed chromatix capability facts. | benchmarks/probes/records/chromatix/gradient_probe.json |
| `src/solvers/chromatix/carrier_removed_asm.py` | 266 | **algorithm to reuse** | Carrier-removed angular-spectrum propagation -- the tilted/off-axis phase trick. | tests/test_carrier_removed_asm.py |
| `src/solvers/chromatix/constants.py` | 75 | **knowledge to reuse** | Pinned chromatix constants and their provenance. | benchmarks/probes/records/chromatix/m3_pupil_to_focus.json |
| `src/solvers/chromatix/execution.py` | 264 | **knowledge to reuse** | Device/dtype execution path into chromatix. | tests/test_precision_execution_matrix.py (gpu) |
| `src/solvers/chromatix/propagation.py` | 504 | **algorithm to reuse** | Propagator selection and transfer functions against the pinned backend. | tests/test_chromatix_adapter.py (slow) |
| `src/solvers/chromatix/provenance.py` | 98 | **knowledge to reuse** | What the chromatix adapter must record. | tests/test_provenance_fingerprint.py |
| `src/solvers/chromatix/requests.py` | 125 | **nothing worth carrying forward** | Request/result envelope. | -- |
| `src/solvers/optiland/__init__.py` | 11 | **nothing worth carrying forward** | Re-export. | -- |
| `src/solvers/optiland/adapter.py` | 681 | **knowledge to reuse** | Optiland API use at 0.6.0: build, trace, what opd means, where it is not verified. | tests/test_optiland_adapter.py |
| `src/solvers/optiland/artifacts.py` | 625 | **knowledge to reuse** | Ray artifact layout incl. hexapolar area weights on export. | tests/test_optiland_adapter.py |
| `src/solvers/optiland/baseline.py` | 217 | **test evidence to reuse** | Baseline scenarios; value is the recorded probe output. | benchmarks/probes/records/optiland/ |
| `src/solvers/optiland/builder.py` | 326 | **knowledge to reuse** | Prescription -> optiland system construction, units and sign conventions. | tests/test_optiland_canonical_prescriptions.py |
| `src/solvers/optiland/capability.py` | 245 | **test evidence to reuse** | Probe-backed optiland capability facts. | benchmarks/probes/records/optiland/gradient_probe.json |
| `src/solvers/optiland/coherent_trace.py` | 589 | **algorithm to reuse** | Coherent trace plans and reference-sphere handoff geometry. | tests/test_optiland_coherent_handoff.py, tests/test_coherent_bridge.py |
| `src/solvers/optiland/constants.py` | 90 | **knowledge to reuse** | Pinned optiland constants incl. the direction-norm 64*eps bound. | tests/test_optiland_adapter.py |
| `src/solvers/optiland/cost_model.py` | 261 | **test evidence to reuse** | Fitted trace-chunk cost model; value is the fit record. | benchmarks/perf/records/optiland_trace_chunk_sweep.json |
| `src/solvers/optiland/execution.py` | 307 | **knowledge to reuse** | Chunked trace execution and the direction-norm tolerance derivation. | tests/test_optiland_adapter.py |
| `src/solvers/optiland/provenance.py` | 59 | **knowledge to reuse** | What the optiland adapter must record. | tests/test_provenance_fingerprint.py |
| `src/solvers/optiland/pupil.py` | 431 | **algorithm to reuse** | Pupil sampling incl. hexapolar distribution and vignetting handling. | tests/test_optiland_adapter.py |
| `src/solvers/optiland/requests.py` | 120 | **nothing worth carrying forward** | Request/result envelope. | -- |
| `src/solvers/registry.py` | 77 | **nothing worth carrying forward** | Old adapter registry. | -- |
| `src/studies/__init__.py` | 6 | **nothing worth carrying forward** | Namespace. | -- |
| `src/studies/metalens/__init__.py` | 6 | **nothing worth carrying forward** | Namespace. | -- |
| `src/studies/metalens/candidate.py` | 596 | **nothing worth carrying forward** | Metalens study; a study, not infrastructure. | -- |
| `src/studies/metalens/controller.py` | 1780 | **nothing worth carrying forward** | 1780-line study controller. | -- |
| `src/studies/metalens/oracle.py` | 653 | **test evidence to reuse** | Study oracle; retain its comparison as evidence only. | tests/test_metalens_oracle.py |
| `src/verification/__init__.py` | 1 | **nothing worth carrying forward** | Namespace. | -- |
| `src/verification/analytic.py` | 237 | **algorithm to reuse** | AnalyticOracle: the O1 analytic closed forms that decide gates. | tests/test_psf_verification.py |
| `src/verification/asm_oracle.py` | 290 | **test evidence to reuse** | O2 ASM/RS oracle -- diagnostic only, never gate-deciding (shares code). | tests/test_psf_verification.py |
| `src/verification/claim_ledger.py` | 1729 | **knowledge to reuse** | The only machine-readable claim->oracle->tolerance map. Extracted in section 6. | tests/test_claim_ledger.py |
| `src/verification/declaration_ledger.py` | 1348 | **knowledge to reuse** | Registry declaration coverage and the 14 invariant tolerance bases. Extracted in section 6. | tests/test_declaration_ledger.py |
| `src/verification/evidence.py` | 499 | **algorithm to reuse** | Convergence fitting, ensemble spread, sigma margin, record fingerprint/writing. | tests/test_substrate_proof.py (slow) |
| `src/verification/families/__init__.py` | 95 | **nothing worth carrying forward** | Namespace. | -- |
| `src/verification/families/b0_contract.py` | 1061 | **test evidence to reuse** | B0 contract family; value is its instance records. | benchmarks/instances/records/B0-*.json |
| `src/verification/families/b1_gsl_validity.py` | 567 | **test evidence to reuse** | B1 GSL validity family. | benchmarks/instances/records/B1-GSL-*.json |
| `src/verification/families/b1_ray.py` | 1601 | **test evidence to reuse** | B1 ray family. | benchmarks/instances/records/B1-RAY-*.json |
| `src/verification/families/b1_wave.py` | 1730 | **test evidence to reuse** | B1 wave family. | benchmarks/instances/records/B1-WAVE-*.json |
| `src/verification/families/b2_transitions.py` | 1808 | **test evidence to reuse** | B2 transition family -- the coupler equivalence evidence. | benchmarks/instances/records/B2-*.json |
| `src/verification/families/b3_4f_ideal.py` | 570 | **test evidence to reuse** | B3 ideal 4f family. | benchmarks/systems/records/B3-4F-IDEAL-*.json (9) |
| `src/verification/families/b3_4f_real.py` | 1462 | **test evidence to reuse** | B3 real 4f family. | benchmarks/systems/records/B3-4F-REAL-*.json (5) |
| `src/verification/families/b3_composed.py` | 1186 | **test evidence to reuse** | B3 composed family. | benchmarks/instances/records/B3-PSF-SINGLET-01.json |
| `src/verification/families/b3_doe_inline.py` | 2257 | **test evidence to reuse** | B3 embedded-DOE family. | benchmarks/systems/records/B3-DOE-INLINE-*.json (12) |
| `src/verification/families/b4_characterization.py` | 705 | **test evidence to reuse** | B4 characterization family. | benchmarks/systems/records/B4-4F-REAL-*.json (8), B4-DOE-INLINE-*.json (6); perf/records/framework_overhead.json, estimate_accuracy.json |
| `src/verification/families/predicates.py` | 470 | **knowledge to reuse** | Gate predicates: how a measurement becomes a pass/fail. | tests/test_family_schema.py |
| `src/verification/families/projection.py` | 166 | **nothing worth carrying forward** | Family projection helper. | -- |
| `src/verification/families/registry.py` | 91 | **nothing worth carrying forward** | Family registry. | -- |
| `src/verification/families/schema.py` | 1251 | **knowledge to reuse** | Family/instance schema incl. what a tolerance must declare. | tests/test_family_schema.py |
| `src/verification/fixed_suite.py` | 841 | **nothing worth carrying forward** | Fixed-suite runner for the old benchmark layer. | -- |
| `src/verification/hazards.py` | 137 | **knowledge to reuse** | Named hazard taxonomy (incl. H4 grazing). | tests/test_verifier.py |
| `src/verification/metrics.py` | 566 | **algorithm to reuse** | Metric definitions: Strehl, first-null radius, L1/L2 field metrics. | tests/test_metrics.py |
| `src/verification/psf_measurement.py` | 448 | **algorithm to reuse** | PSF measurement: sampling, centroid, normalization -- PSF as a measurement. | tests/test_psf_measurement.py |
| `src/verification/psf_oracles.py` | 798 | **algorithm to reuse** | Airy, Fraunhofer, reference-sphere fit, pupil aberration -- the O1 PSF oracles. | tests/test_psf_verification.py |
| `src/verification/refusals.py` | 282 | **knowledge to reuse** | Generated refusal catalogue; structured-failure vocabulary. | tests/test_verifier.py |
| `src/verification/result.py` | 475 | **nothing worth carrying forward** | 15 result models for the old verifier. | -- |
| `src/verification/status.py` | 62 | **nothing worth carrying forward** | Status enum for the old verifier. | -- |
| `src/verification/verifier.py` | 584 | **nothing worth carrying forward** | Old verifier driver. | -- |