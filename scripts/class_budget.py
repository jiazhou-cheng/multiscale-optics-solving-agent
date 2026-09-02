"""Production classes per package, against a declared budget.

CHE-171 (R01.1). The reference implementation reached **280 production classes**
under stated principles comparable to the new ones. That is the entire argument
for this script existing in R01 rather than in R15: a principle nothing counts is
a principle that gets restated in every ticket and enforced in none.

The rule this counts against is the project's, quoted so it is checkable rather
than remembered. A class is justified only if:

1. several fields share an invariant enforced together;
2. it is a public serialized / versioned data model;
3. it owns a genuine mutable resource lifecycle;
4. at least two *current* implementations need runtime polymorphism;
5. it is a real plugin boundary used by the runtime or registry.

Otherwise the answer is a function, a module, a frozen dataclass, a TypedDict, a
tuple, a Literal or an Enum.

**What this script can and cannot do, stated plainly.** It counts. It cannot tell
whether a class satisfies one of the five rules -- that is a judgement, and
pretending otherwise would make the gate authoritative about something it does
not know. So the budget is the reviewed artifact: raising a number requires a
ticket to say which rule the new class satisfies, and this script makes an
unjustified raise visible instead of silent. `docs/architecture_principles.md`
labels this a judgement call for exactly that reason.

`PROJECT_CEILING` is the second half. Without it, a package could raise its own
budget indefinitely and every individual raise would look local; the ceiling makes
the sum of the budgets fail against a project-wide target. That target started at
22 and is not restated here -- it has ratcheted since, and a second copy of the
number in this docstring went stale the first time it moved. `PROJECT_CEILING`
below is the value, and the note attached to it is the history of every raise.

**Why this is a script *and* a test.** Same two-layer split as
`scripts/check_dependencies.py`, for the same reason:

* **This file is the CLI and report layer.** No pytest dependency. `make
  check-arch` prints every package, its count, its budget and the fully-qualified
  name of each class counted -- which is the output you want when deciding
  whether a raise is justified, and which an assertion failure does not give you.
* **`tests/unit/test_class_budget.py` is the gate and the meta-test.** It runs
  `verify()` in the default suite, and it drives `_classes_in` against synthetic
  modules to prove the counter behaves as claimed: that it finds nested classes
  (the cheapest way to hide growth under one budgeted name), that it does not
  charge for the five rules' sanctioned alternatives, and that it *does* charge
  for `TypedDict` and `Enum` -- a known limit pinned as a test rather than left as
  a surprise.

Run directly for a report, or through `tests/unit/test_class_budget.py`, which is
what puts it in the default suite.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# The repo root, so `scripts` resolves as a package whether this file is run
# directly (`python scripts/class_budget.py`) or imported by the test that gates
# it. `LANDED` is imported rather than restated: the migration state is one fact
# and a second copy would drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_dependencies import LANDED

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Production classes allowed per package of the new tree.
#:
#: One entry per package in `LANDED`, and the numbers are the reviewed artifact:
#: a raise belongs to the ticket that adds the class and must name the rule that
#: class satisfies. `tests/unit/test_class_budget.py` requires each number to
#: equal what is actually on disk, so a budget cannot be raised in advance of the
#: code -- headroom is not a pre-authorization.
#:
#: `representations` is 2, raised from 0 by CHE-174 (R02.2). Both are rule 1 --
#: a shared invariant across several fields -- and the invariant is named, because
#: "these fields go together" is the justification every unjustified class also
#: claims:
#:
#:   `Frame`             rule 1 -- axis order, handedness, origin rule and
#:                       propagation axis are one mapping from array indices to
#:                       physical directions. Each fixes part of the same mapping,
#:                       and getting any one wrong silently mirrors, transposes or
#:                       shifts a wavefront instead of raising. All four are
#:                       validated in `__post_init__`.
#:   `ReferenceSurface`  rule 1 -- axial coordinate, unit normal and medium index
#:                       are only meaningful together: an optical path is `n * s`
#:                       projected onto the normal, so a valid `z_m` beside an
#:                       unnormalized normal or an index nobody set yields an OPL
#:                       wrong by a factor no later check can attribute back.
#:
#: Raised again to 5 by CHE-175 (R02.3, +2) and CHE-176 (R02.4, +1):
#:
#:   `ContractError`     exception -- the one catchable failure type in the
#:                       package. None of the five rules claims an exception and
#:                       R00.2 counted 22 of them in the old tree the same way;
#:                       it is a class because `except ContractError` is what
#:                       lets a coupler return a diagnostic instead of an
#:                       invented field, and `except ValueError` would also
#:                       swallow every unrelated arithmetic error.
#:   `RayBundle`         rules 1 + 2 -- geometry, coherent state and sampling
#:                       measure are three groups with joint invariants (per-ray
#:                       length, one device and one namespace, an optical path
#:                       without its reference or a measure without its kind is
#:                       unusable), and it is the public model a solver produces
#:                       and a coupler consumes.
#:   `ScalarField`       rules 1 + 2 -- array, pitch, wavelength, surface and pad
#:                       state are one physical object; a pitch that does not
#:                       belong to this array gives a plausible extent that is
#:                       wrong by a factor.
#:
#: Seven names did *not* land against that +3: `CoherentRayBatch`,
#: `WavefrontSamples`, `GeometricRayBundle`, `CoherentRayBundle`, `RayBundleBase`,
#: `TrackedRayBundle` and `RayBatch`, plus `PSF` as a representation.
#: `tests/representations/test_rays.py` and `test_scalar.py` assert their absence,
#: because a budget records what exists and cannot record what was avoided.
#:
#: `numerics` is 7, raised from 0 by CHE-173 (R02.1). Three of the seven are
#: production classes and the ticket names the rule each satisfies:
#:
#:   `DevicePlacement`        rule 1 -- kind and index are one invariant; an index
#:                            without a kind is meaningless and a host index is a
#:                            contradiction, refused in `__post_init__`.
#:   `ArrayState`             rule 1 -- namespace, device and dtype are one
#:                            observation of one buffer. Split apart they invite a
#:                            namespace read from data beside a device read from a
#:                            config value, describing no array that exists.
#:   `ComponentCapabilities`  rule 2 -- the public, probe-backed capability model
#:                            every descriptor and solver reasons against, and the
#:                            thing a probe re-run confirms or falsifies.
#:
#: The other four -- `Precision`, `DType`, `DeviceKind`, `ArrayNamespace` -- are
#: `StrEnum`s, which the five rules list as a *sanctioned alternative* to a class.
#: They are counted anyway because `StrEnum` is written with `class` syntax and
#: this gate is an AST count, not a judgement; `tests/unit/test_class_budget.py`
#: pins that behaviour deliberately. Four enums is the honest arithmetic, not four
#: classes the architecture wanted avoided.
#: `operations` is 2, raised from 0 by CHE-177 (R03.1). CHE-178 (R03.2) adds none:
#: the registry is module-level state plus four functions, and a `Registry` class
#: would need two consumers wanting independent registries to justify it.
#:
#:   `OperationDescriptor`  rules 2 + 5 -- the public model planning and the
#:                          runtime read, and the plugin boundary between an
#:                          operation and the layer that selects one. The four
#:                          operation kinds are a *field* on it; four subclasses
#:                          would share every field and override nothing, and the
#:                          distinction they would encode (a coupler changes
#:                          representation, a physical operator changes physical
#:                          state) is not one an `isinstance` check can enforce.
#:   `OperationKind`        a `StrEnum`, counted for the same reason the four in
#:                          `numerics` are: this gate is an AST count and
#:                          `StrEnum` is written with `class` syntax. CHE-177 calls
#:                          the class delta +1 on the grounds that an enum is a
#:                          sanctioned alternative to a class; 2 is what is on
#:                          disk, and the budget records disk.
#:
#: Not landed against that +2, and asserted absent by
#: `tests/operations/test_descriptors.py`: `SolverDescriptor`, `CouplerDescriptor`,
#: `PhysicalOperatorDescriptor`, `MeasurementDescriptor`, `Registry`, and the
#: 13 spec classes of the old `core/specs.py` that became fields on one record.
#: `problems` is 3, raised from 0 by CHE-156 (R04). The ticket budgets "2 public
#: classes, with material / aperture / source / wavelength as TypedDict, Literal
#: or frozen tuples"; three is what an AST count sees, and the third is the
#: TypedDict:
#:
#:   `OpticalSetup`     rules 1 + 2 -- `stop_index` has to index `surfaces`, so
#:                      the fields are jointly constrained and none of them means
#:                      anything alone; and it is the public model a solver adapter
#:                      consumes. Was `RayTraceProblem` until CHE-218 (R05.7),
#:                      which split the illumination out of it; see the +1 below.
#:   `SurfaceSpec`      rule 1 -- curvature, conic, following medium and spacing
#:                      describe one interface. A radius given twice in two forms
#:                      makes the surface silently a *different* one rather than
#:                      an invalid one, which is the failure mode the joint
#:                      validation exists for.
#:   `Material`         a `TypedDict`, which the five rules list as a sanctioned
#:                      alternative to a class, counted for the same reason the
#:                      enums in `numerics` and `operations` are: this gate is an
#:                      AST count. Its runtime validation is a *function*
#:                      (`_check_material`), because a TypedDict is an annotation
#:                      that disappears at run time.
#:
#: **Raised to 4 by CHE-218 (R05.7)**, which splits `RayTraceProblem` into
#: `OpticalSetup` and `SourceSpec`. Not a new capability given a class: one record
#: became two because it held two unrelated things, and the coupling was
#: *executable* -- the pinned solver normalizes an angular field against the largest
#: field the system declares, so tracing an existing system at a new field angle
#: meant editing the optical system, and an already-materialized `RayBundle`
#: (R05.6) forced a caller to invent a field angle and an object distance so that a
#: lens could be built.
#:
#:   `SourceSpec`       rules 1 + 2. Rule 1, and the shared invariant is the point:
#:                      **the meaning of `field_angle_deg` depends on
#:                      `object_distance_mm`** -- at infinity a field angle is a
#:                      direction, at a finite distance it is a position
#:                      (`-tan(theta) * d`), measured to twelve digits by CHE-207.
#:                      The two fields are not independently interpretable. Rule 2,
#:                      because it is the second half of what a solver adapter
#:                      consumes and its field names are the interface.
#:
#: What the +1 bought, so the raise is reviewable against something: two *fields*
#: were removed with their validators -- `field_angles_deg` and `wavelengths_um`,
#: each with its index bound -- and `primary_wavelength_index` with them. A plural
#: was replaced by a singular in both cases, because one solve is one field and one
#: wavelength and neither list ever reached a trace as a list. The alternative to
#: the class was a `TypedDict`, which this gate counts identically, or three loose
#: arguments, which would have put the field-angle-means-two-things invariant
#: nowhere.
#:
#: Twenty class names did not land against that +3 -- the whole of
#: `core/optical_system.py`'s geometry, interaction, material, aperture, field and
#: wavelength hierarchies, its four kind enums and its error type, plus
#: `core/optical_assembly.py`'s three and every builder.
#: `tests/problems/test_ray_trace.py` asserts their absence, and now also asserts
#: that `RayTraceProblem` is gone rather than aliased.
#: `solvers` is 2, raised from 0 by CHE-181 (R05.3). **Both are `TypedDict`s and
#: neither is a public class**: R05 budgets "0 public classes, at most 1 private",
#: and the private one (`_OptilandTraceBatch`, for grouped native ray state) did
#: not land -- the native columns are a local dict inside one function, and no
#: second function exchanges them. Two is what an AST count sees, for the same
#: reason `problems.Material` is counted: this gate counts `class` syntax and does
#: not judge, and `TypedDict` is written with it.
#:
#:   `Sampling`   a `TypedDict`, the sanctioned alternative -- the fields share no
#:                invariant enforced across them, nothing subclasses it, and its
#:                runtime validation is a *function* (`_require_keys`) because a
#:                TypedDict is an annotation that disappears at run time. It is
#:                the public argument schema of `trace`, and the alternative was
#:                `Mapping[str, Any]`, which is the untyped config dict this
#:                rewrite is removing.
#:   `Execution`  the same, for device and precision. Both keys are required
#:                rather than defaulted, because this is the process-global solver
#:                state whose inheritance was a measured source of
#:                nondeterminism.
#:
#: Six class names did **not** land against that +2, and the tests name them:
#: `OptilandAdapter` (a one-instance facade behind `get_adapter()`),
#: `OptilandExecutionState`, `TracePlans`, `OptilandRayRequest` /
#: `OptilandRayFailure` / `OptilandRayResult`, `PatchEmitterCostModel`, and
#: `HandoffPlaneError` as a separate exception type -- an unresolvable plane is a
#: `ContractError` with a code, because a coupler branches on the code and not on
#: the class. `solvers/optiland/baseline.py` did not land either, in its entirety.
#: `operators` is **1**, raised from 0 by CHE-194 (R10.2), which adds
#: `DiffractiveSurface` -- rules 1 and 2. Rule 1: the transmission, the pitch it is
#: sampled at and the surface it lives on are one physical object, and the failure
#: mode is specific rather than tidiness -- before the reference implementation
#: gathered them, "the diffractive surface" was four loose arguments repeated at
#: every call site, so a caller could describe one surface to one function and a
#: different one to the next. Rule 2: it is the public argument schema of the
#: operation.
#:
#: R10.2 writes `DiffractiveModel` as a `StrEnum` and budgets the change at "+1";
#: this gate counts a `StrEnum` as a class, so it landed as a `Literal`, the same
#: stricter reading R08.1 and R08.2 took. **Eight** class names did not land against
#: that +1, and `tests/physics/test_diffractive_surface_full_field.py` asserts each:
#: `DiffractiveSurfaceBase`, `FullFieldDiffractiveSurfaceSubclass`,
#: `PlanarDoeStepCoupler` (491 LOC of node wrapper), `CascadeDiagnostics`,
#: `DiffractiveInteractionResult`, `FullFieldParameters`, `PrimarySampling` and
#: `DiffractiveModel` itself.
#:
#: `sources` is 0, declared by CHE-210 (R06.5), and that zero is genuine rather
#: than budgeted-at-zero-for-now: an illumination is three keyword arguments and a
#: return value. `operators` was zero for the same kind of reason until R10.2 -- a
#: mask is an array plus the grid it was built on, and the grid already lives on
#: the `ScalarField`. Nine class names did **not** land against those two
#: packages, and the tests name them: `ThinElement`, `PhaseMask`, `AmplitudeMask`, `Pupil` and
#: `Grating` as separate operators or result types, and `Illumination`,
#: `PlaneWaveSource`, `IlluminationAngle` and `SourceResult` on the source side.
#: The `Illumination` frozen dataclass is the one that has a real argument -- lambda,
#: `k_t` and the medium index are coupled by `|k_t| <= n k0`, which is rule 1 -- and
#: it did not land because the coupling is checked *at the point the field is
#: built*, where the grid's Nyquist limit is also known and is the second refusal;
#: a declaration object could not check that half, so it would validate less than
#: the function does while adding a type.
#: `couplers` is 2, raised from 0 by CHE-185 (R07.1). Neither is a coupler *object*:
#: R07 budgets one diagnostics record plus enums, and `RayToWaveCoupler` -- 533 LOC
#: of node wrapper in the reference implementation -- did not land, nor did
#: `CoherentHandoff`, `DeclaredHandoffPlane`, `Perturbation`,
#: `HandoffPerturbation`, `StreamingReconstruction`, `StreamingResult`,
#: `PositionalAngularSampler`, `LaunchGeometry`, `BandLimit`, `ChunkWorkItem`,
#: `CurvatureBudget`, the `Coupler` protocol with `CouplerRunRequest` /
#: `CouplerRunResult`, or `GradientProblem` / `DifferentiabilityReport`.
#: `tests/physics/test_ray_to_scalar.py` asserts their absence.
#:
#:   `ReconstructionDiagnostics`  rule 2 -- the public record a caller reads back,
#:                                and the thing three of R07's acceptance criteria
#:                                are statements about (which route produced the
#:                                field, which measure was applied, what power was
#:                                excluded). The alternative was a free-form dict,
#:                                which is the provenance mapping R02.4 removed
#:                                from `ScalarField` for the same reason.
#:   `Projection`                 a `StrEnum`, the sanctioned alternative, counted
#:                                because this gate is an AST count. It is not a
#:                                boolean because the two members name two
#:                                *operators* -- SI eq S5 and main-text eq 2 -- and
#:                                a flag would name the factor instead of the
#:                                physics.
#:
#: Raised to 3 by CHE-186 (R07.2), which adds `Reconstruction` -- a `StrEnum`
#: again, and the ticket's own budget is 0 production classes plus it. What it did
#: *not* add is what the budget is protecting: no second k-space coupler, no
#: separate module, and no module-level route registry. The route is an argument,
#: because the semantic port pair is identical and one operation with two numerical
#: realizations is what that means.
#:
#: **That raise put the declared total at exactly `PROJECT_CEILING`**, which is what
#: forced the ceiling question below to be answered rather than deferred again.
#:
#: Raised to 4 by CHE-189 (R08.1), which adds `SamplingDiagnostics` -- rule 2, the
#: public record a caller reads back and that three of R08's acceptance criteria are
#: statements about (how the modes were selected, from which density, under which
#: seed). The alternative was a free-form mapping, which is the provenance dict
#: R02.4 removed from `ScalarField` for exactly this reason.
#:
#: R08.1 also writes `SamplingDensity` as a `StrEnum` while budgeting the whole
#: change at "+1 class". This gate counts a `StrEnum` as a class, so the two
#: readings differ by one; the stricter one is taken and `SamplingDensity` landed as
#: a `Literal`, as `couplers.GrazingPolicy` and `representations.MeasureKind`
#: already had. Five class names did **not** land against that +1, and
#: `tests/physics/test_scalar_to_ray.py` asserts their absence: `AngularSpectrum` as
#: a public type (an intermediate, not a boundary artifact -- a third representation
#: in a tree whose whole point is that there are two), `SamplingPerturbation`,
#: `PositionPlan`, `PatchPlan` and `Ensemble`.
#:
BUDGETS: dict[str, int] = {
    "couplers": 4,
    "measurements": 1,
    "numerics": 7,
    "operations": 2,
    "operators": 1,
    #: `planning` is 0, and that is the reviewed number rather than a placeholder.
    #: CHE-164 (R12) budgeted one class, `CapabilityGraph`, and asked that it be
    #: justified against a named rule first. It cannot be: a capability graph is
    #: derived from the catalog on every call, holds no invariant the catalog does
    #: not already enforce, is not serialized, owns no resource and has one
    #: implementation -- rules 1 through 5, all failed. So `capability_graph()`
    #: returns a plain mapping and `routes()` is a function.
    #:
    #: The budget note directly below is why that mattered beyond tidiness: the
    #: project's last authorized unit is reserved for `runtime.Executor`, and a
    #: record failing every rule must not be what spends it.
    "planning": 0,
    "problems": 4,
    "representations": 5,
    #: `runtime` is 3. `ExecutionRecord` and `NodeRecord` landed with CHE-199
    #: (R13.1), both rule 2 -- public serialized provenance models. `Executor`
    #: landed with CHE-200 (R13.2) on **rule 3**, and it is the only class in the
    #: new architecture on that rule: it owns a mutable resource lifecycle, which
    #: `runtime/executor.py` names as **a memory sampling thread** -- started on
    #: `__enter__`, joined on `__exit__`, and the thing the shared-server
    #: swap-growth stop condition needs in order to exist. This is the unit reserved
    #: at the 25 -> 26 raise in the note below, spent on what it was reserved for.
    #:
    #: **Correction, recorded rather than edited away.** That reservation, and the
    #: first version of `executor.py`, named the resource as "process-global solver
    #: backend, device and precision state plus a memory sampling thread". The first
    #: half was false and could not have been true: `check_dependencies` gives
    #: `runtime` only `{planning, operations, representations}`, so this package
    #: cannot reach `configure_execution` at all -- and it does not need to, because
    #: that function sets all three on every call and never inherits what a previous
    #: call left. Rule 3 holds on the thread alone, which is a real resource with a
    #: real lifetime; the raise is unchanged and the justification is now the one
    #: the code supports.
    "runtime": 3,
    "solvers": 2,
    "sources": 0,
}

#: The project's declared target for the whole production tree. The reference
#: implementation had 280; R00's inventory found 66 of them would satisfy a rule
#: at all, which is still three times this number, so the collapse R02-R11 owes is
#: real work and not rounding.
#:
#: **The inconsistency this constant used to record has now bound, and 22 -> 23 is
#: the minimal response to it.** Raised by CHE-189 (R08.1). The situation the
#: earlier note described:
#:
#: * `AGENTS.md` and `docs/architecture_principles.md` were both rewritten for the
#:   clean slate and state that the rewrite does **not** inherit a project-wide
#:   class ceiling, on the grounds that 22 was derived from a tree that no longer
#:   exists. Those two documents are source-of-truth rank 2; this script is rank 3.
#: * This script nonetheless enforced 22, and CHE-186 (R07.2) took the declared
#:   total to exactly that. R08.1 needs one class -- `SamplingDiagnostics`, named by
#:   its own ticket -- and there is no way to land it under 22.
#:
#: What was done, and what deliberately was **not**: the ceiling is raised *by
#: one*, not deleted, not rounded up to leave headroom, and not made advisory. The
#: earlier note warned against quietly deleting it "to make a raise fit"; the
#: operative word is quietly, and this is its opposite -- the raise is exactly the
#: size of the one class that forced it, it lands in the commit that adds that
#: class, and it is flagged on CHE-189 for the owner rather than buried here.
#:
#: **The owner still has to settle whether this constant should exist at all.**
#: `AGENTS.md` says a class budget introduced later should be "derived from the new
#: tree and made a visibility gate, not a substitute for design review", and a
#: number inherited from the deleted tree is neither derived nor a gate anyone
#: reviewed. The per-package budgets above are the part doing real work: they are
#: *exact-equality* gates, so every one of them is attached to the code and to the
#: rule that justifies it, and headroom cannot be reserved in advance. If this
#: constant stays it should be re-derived from the new tree; if it goes, that
#: per-package equality is what remains, and it is already the stronger check.
#: Until the owner decides, it ratchets by one per class -- which at least makes
#: every raise a visible, reviewable line in a diff. **23 -> 24 by CHE-194
#: (R10.2)**, one class, same handling: the raise is the size of the class that
#: forced it and lands in the commit that adds it.
#: **24 -> 25 by CHE-197 (R11.1)**, which lands `measurements/` with one class:
#: `PsfResult`, on rule 2 -- the public serialized record a consumer reads back,
#: and the thing three of that ticket's acceptance criteria are statements about
#: (which normalization, at what scale, with how much energy on the border). Its
#: `PsfNormalization` is a `Literal` rather than the `StrEnum` the ticket names,
#: for the reason R08.1 established: this gate counts a `StrEnum` as a class, the
#: two readings of "+1" differ by one, and the stricter one is taken.
#:
#: **25 -> 26 for CHE-200 (R13.2), and this is the first raise that does *not*
#: land in the commit that adds the class.** The class is `Executor`, in
#: `src/runtime/executor.py`, on rule 3 -- the one genuine mutable resource
#: lifecycle in the new architecture: process-global solver backend, device and
#: precision state, plus the memory guard the shared-server policy requires. That
#: ticket names it as rule 3 explicitly and calls it the only class in the tree
#: justified that way, so the justification is not being invented here.
#:
#: Why the raise is split, stated plainly because the split is the irregular part.
#: The other half of it is `BUDGETS["runtime"] = 1`, and that half cannot land
#: yet: `runtime` is not in `check_dependencies.LANDED` and `src/runtime/` is not
#: on disk, so declaring a budget for it trips the structural check above ("a
#: budget for a package that does not exist reads as headroom nobody is using"),
#: and declaring it at 1 against 0 counted files fails the per-package equality
#: test as well. So between this commit and R13.2 the declared total is 25 against
#: a ceiling of 26, which is one unit of exactly the standing headroom the note
#: above argues against.
#:
#: That is a real, if small, weakening of the ratchet, and it is recorded rather
#: than smoothed over: the ceiling was raised on the owner's instruction, ahead of
#: the code, with the forcing class named in advance. What is preserved is that
#: the raise is still the size of one class and still names which class and which
#: rule. R13.2 owes the matching `BUDGETS["runtime"] = 1` and the `LANDED` entry in
#: the commit that adds `executor.py`; until then this line is a pre-authorization
#: and nothing here can consume it, because per-package equality still forbids any
#: package from growing.
#:
#: **26 -> 27 by CHE-218 (R05.7)**, for `problems.SourceSpec` -- the split of
#: `RayTraceProblem` into a setup and an illumination, justified against rules 1
#: and 2 in the `problems` note above. Same handling as every raise before it: one
#: class, named, in the commit that adds it.
#:
#: The raise is +1 rather than +0 **on purpose, and this is the part that needed
#: deciding.** The declared total before this commit was 25 against a ceiling of
#: 26, so there was one unit of slack and `problems: 3 -> 4` would have fitted
#: inside it without touching this constant. That unit is not slack: the note
#: directly above reserves it for R13.2's `runtime.Executor`, pre-authorized by the
#: owner ahead of the code. Consuming it here would silently spend another
#: ticket's authorization and leave R13.2 unable to land without a raise nobody
#: had agreed to -- which is exactly the "every individual raise looks local"
#: failure the ceiling exists to prevent. So the ceiling moves by one and the
#: declared total goes to 26 against 27, leaving R13.2's unit where it was.
#:
#: **27 -> 29 by CHE-199 (R13.1)**, and this is a raise of *two*, which no previous
#: one has been. **Flagged for the owner** rather than treated as routine, because
#: the ratchet's whole value is that a raise is one visible line per class.
#:
#: The two are `runtime.ExecutionRecord` and `runtime.NodeRecord`, both on **rule
#: 2** -- public serialized provenance models. CHE-199 names both and budgets +2, so
#: the justification is not invented here, and `records.py`'s docstring records
#: which of the reference tree's classes each one replaces: the seven of
#: `core/execution_record.py` collapse into these two plus a `Literal` and two
#: string mappings, `RunProvenance` and `RecordVerdict` do not come back, and none
#: of the eleven of `core/performance.py` does either.
#:
#: Why two and not one. They are the two halves the deletion test is *about*: a
#: `NodeRecord` is one operation's outcome and an `ExecutionRecord` is the run,
#: and collapsing them would put a node's status, diagnostics and observed
#: placement into parallel lists indexed by position -- which is the arrangement
#: `nodes[i].operation_id == route[i]` currently checks and would then be unable
#: to. Rule 2 is satisfied separately by each: both are serialized, both are read
#: back by `from_json`, and a consumer reads a node without reading the run.
#:
#: **R13.2's unit is still untouched.** The declared total after this commit is 28
#: against 29: 26 as before, plus these two. `BUDGETS["runtime"] = 2` here, and
#: R13.2 raises it to 3 for `Executor` in the commit that adds `executor.py` --
#: spending the unit reserved at 25 -> 26 above, which is what it was reserved
#: for. Note that this raise lands *with* its classes, unlike that one.
#:
#: Flagged for the owner on CHE-218, because the standing question above is now
#: two units old: this constant has ratcheted five times without ever being
#: re-derived from the new tree, and `AGENTS.md` says a class budget introduced
#: later should be derived from that tree and made a visibility gate. The
#: per-package equality gates are the half doing real work.
PROJECT_CEILING = 29


@dataclass(frozen=True)
class PackageCount:
    package: str
    counted: int
    budget: int
    classes: tuple[str, ...]

    @property
    def over(self) -> int:
        return max(0, self.counted - self.budget)

    @property
    def headroom(self) -> int:
        return max(0, self.budget - self.counted)


def _classes_in(path: Path) -> list[str]:
    """Top-level and nested class names defined in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _modules_of(package: str) -> list[Path]:
    base = SRC / package
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in str(p))


def count() -> list[PackageCount]:
    counts: list[PackageCount] = []
    for package in sorted(LANDED):
        found: list[str] = []
        for module in _modules_of(package):
            where = module.relative_to(SRC).as_posix()
            found.extend(f"{where}::{name}" for name in _classes_in(module))
        counts.append(
            PackageCount(
                package=package,
                counted=len(found),
                budget=BUDGETS.get(package, 0),
                classes=tuple(sorted(found)),
            )
        )
    return counts


def verify() -> tuple[list[str], list[str], list[PackageCount]]:
    """Return (budget failures, structural problems, per-package counts)."""
    counts = count()
    failures: list[str] = []
    structural: list[str] = []

    if not LANDED:
        structural.append(
            "no package of the new tree has been landed, so this gate would pass "
            "without counting anything"
        )

    for package in sorted(LANDED):
        if package not in BUDGETS:
            structural.append(
                f"src/{package}/ is a landed package with no entry in BUDGETS, so its "
                "class count is unbudgeted. Declare a number, even if it is 0."
            )
    for package in sorted(BUDGETS.keys() - LANDED):
        structural.append(
            f"BUDGETS declares {package!r}, which is not landed. A budget for a package "
            "that does not exist reads as headroom nobody is using."
        )

    declared = sum(BUDGETS.values())
    if declared > PROJECT_CEILING:
        failures.append(
            f"the budgets sum to {declared}, over the project ceiling of {PROJECT_CEILING}. "
            "Raising one package's budget cannot be a local decision: the target is for the "
            "whole production tree."
        )

    for entry in counts:
        if entry.over:
            failures.append(
                f"src/{entry.package}/ has {entry.counted} production class(es) against a "
                f"budget of {entry.budget} -- {entry.over} over.\n"
                + "\n".join(f"      {name}" for name in entry.classes)
                + "\n    Either collapse them (function, module, frozen dataclass, TypedDict, "
                "tuple, Literal, Enum) or raise the budget in a ticket that names which of the "
                "five minimality rules each new class satisfies."
            )

    return failures, structural, counts


def _report() -> int:
    failures, structural, counts = verify()
    total = sum(entry.counted for entry in counts)
    declared = sum(BUDGETS.values())
    print(f"production classes: {total} / {declared} budgeted (ceiling {PROJECT_CEILING})")
    for entry in counts:
        print(f"  src/{entry.package}/: {entry.counted} / {entry.budget}")
        for name in entry.classes:
            print(f"      {name}")
    if structural:
        print("\nSTRUCTURAL PROBLEM -- this gate cannot be trusted as it stands:")
        for problem in structural:
            print(f"  {problem}")
    if failures:
        print("\nBUDGET EXCEEDED:")
        for failure in failures:
            print(f"  {failure}")
    if not structural and not failures:
        print("\nOK: within budget.")
    return 1 if (structural or failures) else 0


if __name__ == "__main__":
    sys.exit(_report())
