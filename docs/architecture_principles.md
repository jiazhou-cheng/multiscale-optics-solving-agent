# Architecture Principles

**CHE-172 — clean-slate rewrite.** This document defines the *target*
architecture for building the new `src/` tree from scratch. It is a design
contract, not an inventory of packages, scripts, tests, or capabilities that
already exist.

A package named below is not required to exist until a scoped issue needs it.
Do not create empty packages, placeholder classes, base classes, protocols, or
registries merely to make the target tree visible.

Every principle is labelled with one of three statuses:

* **[ENFORCED]** — executable code in the current tree checks the rule and the
  check is part of the repository's normal verification path.
* **[LANDING GATE]** — the rule becomes mandatory when the affected package or
  capability is introduced. The same change that lands that surface must add the
  executable check where the rule is mechanically checkable.
* **[JUDGEMENT]** — the rule requires engineering or scientific review and cannot
  be decided reliably by a script alone.

Do not label a rule **[ENFORCED]** merely because the previous implementation had
a checker for it. Historical checks are evidence for what to rebuild, not
evidence that the new tree is currently protected.

---

## 1. Two concepts

The target system has exactly two conceptual categories.

**Representations** are physical state at a declared boundary. **Operations**
consume and produce representations, problems, or measurements — with one
asymmetric case: a **source** produces a representation without consuming one,
because it *initializes* physical state rather than transforming it (§2).

There are four operation kinds — `source`, `coupler`, `physical_operator`, and
`measurement` — and they are **descriptor metadata, not four class hierarchies**.
The kind is a field on one operation descriptor.

A **backend** is not among them. Which third-party library executes an operation
is a separate axis from what the operation does to physical state, and it is a
separate field: `OperationDescriptor.backend`. CHE-224 (R15.1) separated the two;
see the `backend` term in §2.

The four are **primitive**. One callable may fuse several of them, and it says so
in `OperationDescriptor.composes` with `kind` naming the terminal stage — see
*composite operation* in §2. A composition is not a fifth kind and not a pipeline
language: it exists because one landed operation initializes state *and* evolves
it, so any single word for it is a false claim.

The previous implementation expressed these kinds as families of base classes,
request/result envelopes, and per-family diagnostics. The clean rewrite must not
recreate that shape by default.

*[JUDGEMENT]* Whether a specific concept is a representation or an operation.

*[LANDING GATE]* When operation discovery/registration lands, the executable
contract must represent the four kinds as descriptor data. Do not introduce a
base-class family per kind unless a later issue demonstrates a real runtime need
that satisfies the class-minimality rules below.

---

## 2. The eight terms

Each term is defined by the boundary that separates it from its nearest
neighbour. These definitions are target semantics even before their corresponding
packages exist.

### representation

Describes physical state at a declared boundary, with conventions explicit and
testable: units, axes, frame, handedness, wavelength, phasor sign, polarization,
coherence, normalization, sampling, and reference plane.

**Boundary against *measurement*:** a representation is state; a measurement is
an observable *derived* from state. **PSF is a measurement, not a
representation.** A trivial observable is not a cross-representation handoff.

**Boundary against a serialized record:** being serializable does not make an
object a representation. A run record is not physical state.

The initial public target is one ray representation and one scalar-field
representation. Coherence is a *stronger contract* on the ray representation
(e.g. `require_coherent()`), not a subtype, unless a concrete implementation
issue shows that distinct runtime identity is required.

### backend

An **adapter package that provides operations of the other kinds, and is not
itself an operation kind.** A backend adapter is the boundary at which an external
library's API, conventions, compatibility logic, or version requirements may
appear, and the only place in the tree permitted to import that library. Backend
imports belong in `backends/<backend>/`.

**Boundary against *operation kind*:** a backend answers **who executes**; a kind
answers **what happens to physical state**. Every operation has exactly one of
each, and they are two fields — `backend` and `kind` — not one. A backend does not
appear in `OperationKind` at all: `backends/optiland/` provides one
`physical_operator` (`O_RAY_TRACE`) and one **composite** whose terminal stage is a
physical operator (`SO_RAY_LAUNCH_TRACE`, §2 *composite operation*), and a
backend-provided `measurement` would live there too. The package a callable lives
in follows its **provider**; its kind is declared in the catalog.

**Boundary against *coupler*:** a backend adapter is where an external solver
enters the system; a coupler is repository-owned physics between representations.

*[JUDGEMENT]* **This term replaced a `solver` term on CHE-224 (R15.1), and the
change is recorded rather than substituted.** The old term read "Maps a
**problem** into a physical representation", with `solver` as one of the four
operation kinds. What went wrong is that this definition and the boundary beneath
it were about two different things: mapping a problem into a representation is a
statement about state, and owning an external library's API is a statement about
who executes. `coupler`, `physical_operator` and `measurement` are all the first
kind of statement, so `solver` sat on a different axis from the other three.

Three consequences were live in the tree, not hypothetical. `source` — a term this
section has always defined — had no member in `OperationKind`, so all three source
records declared `kind=solver`. The `S_` id prefix therefore meant "solver" on
`S_RAY_OPTILAND` and "source" on `S_SOURCE_PLANE_WAVE`, and `kind` read `solver`
for both. And `backends.chromatix.solver:propagate` needed **two** catalog records,
because one field was answering two questions; `operations/catalog.py` said so in
its own docstring.

What replaces it: `backend` is a provider, defined above; `source` is the kind, and
its definition below did not change a word. The fact that an operation drives a
library is carried by `backend`.

*[JUDGEMENT]* **CHE-224 also concluded that a backend adapter's *problem-driven
solve* is therefore a plain `source`, and CHE-225 (R15.2) retracts that.** The
argument was structural: an `OpticalSetup` is a constructor argument and not a
port, so the schema "cannot distinguish `S_RAY_OPTILAND` from
`S_SOURCE_PLANE_WAVE`". It proves too much. It reasons from *ports* to *kind*, and
`kind` exists precisely to state what the ports cannot — on the state axis the two
are not alike at all, because `trace` initializes rays **and then refracts them
through every surface**. The record's own `approximation` said so, so the catalog
carried a `kind` its own prose contradicted. That operation is now the composite
`SO_RAY_LAUNCH_TRACE`; see *composite operation* below.

### source

Owns the **physically meaningful initialization** of a representation: how the
state of a representation is created from physical source parameters.

A representation defines the *structure and conventions* of physical state at a
declared boundary. A source defines how that state is *initialized*. A plane-wave
source initializes the complex amplitude and phase of a `ScalarField` from its
wavelength, propagation direction, amplitude, sampling grid, and reference
surface.

**A source may be described without an optical system. A ray launch may not.**
This sentence used to carry "a collimated ray source initializes a `RayBundle`
from its spatial sampling and common propagation direction" as its second
example, and **CHE-219 (R05.8) removed it deliberately.** A source can state
what is true of the light alone — infinite versus finite conjugate, field angle
or object position, wavelength, source type, source-side physical parameters.
Where the rays of that source actually *start* is not among them: the launch
positions and directions depend on the stop, the entrance pupil's location and
diameter, every surface preceding the stop, the object distance, the field, the
backend's pupil mapping, and the ray aimer and its convergence behaviour. For an
off-axis or finite-conjugate field the backend delegates the launch to its
aimer, so the launch state is a property of **source + system + backend**.

A `RayBundle` built from caller-supplied points and a shared direction, with no
system in scope, cannot say whether those points are the entrance pupil, the
stop, the first traced surface, a valid finite-conjugate aim, or anything in the
constructed system at all. That is why ray launch belongs to the **backend** —
backend ownership beats taxonomy — and why it takes the constructed system as a
required argument. It is `backends/optiland/launch.py` today, and it is not in the
catalog: it takes native solver state, and a public launch operation needs a
neutral signature first. (This sentence read "is a **solver** operation" until
CHE-224 (R15.1); `solver` was the operation kind then, and the claim it was making
was about which package owns the code.)

Note precisely what this does and does not narrow. `sources/` may still
initialize any representation whose state is genuinely determined by source
parameters, and the rule below is unchanged. What it may not do is manufacture
rays first and let the system decide afterwards what they meant; there is
deliberately no middle state.

**Boundary against *representation*:** the representation owns the declaration —
units, axes, frame, handedness, phasor sign, sampling, reference surface — and
validates it. The source owns the physics that fills it. Putting initialization
physics in `representations/` would make the data model own physics it exists only
to declare.

**Boundary against every other operation:** a source **does not consume an
existing physical representation. It creates the initial state of one.** Every
other kind takes a representation in — a coupler re-describes one, a physical
operator changes one, a measurement derives an observable from one. A source is
the only operation in the graph with no input representation, and that asymmetry
is why it has its own package and its own row in §3.

**Boundary against *problem*:** a problem is physical *intent*. The declaration of
a source may live in `problems/`; the constructor that turns it into state is the
source. A source may read a problem; it is not one.

**Kind:** `source`, which is its own member of `OperationKind` since CHE-224
(R15.1) and was `solver` before it. What separates `sources/` from
`backends/<backend>/` is not the kind — both provide `source`-kind operations —
but the **provider**: a source in `sources/` has **no external backend**, so its
descriptor carries `backend=None`, and it is the project's own arithmetic on the
project's own grid, which per-backend organization has nothing to organize.

`sources/` is **representation-independent at the package level and
representation-explicit at each public operation.** A source operation may
initialize a `ScalarField`, a `RayBundle`, or any other landed representation, and
which one it returns must be unambiguous in the public API — in the signature's
return type and in the descriptor's `output`. The package is therefore not
partitioned by representation and must not grow a subpackage per representation;
the individual operation is never ambiguous about what it produces.

*[LANDING GATE]* A source operation's return representation is declared in its
public signature and in its descriptor. A constructor that could return either of
two representations depending on its arguments is a design error, not a
convenience.

*[LANDING GATE]* No operation in `sources/` produces a system-launch `RayBundle`,
and the package resolves no pupil, stop, entrance-pupil, aiming or launch-surface
quantity. This is a semantic rule and not a dependency-direction one: the
dependency graph already forbids `sources/ -> backends/`, and the hazard is a
function that returns a launch `RayBundle` while importing nothing at all.
`tests/sources/test_sources_package.py` checks it.

**A descriptor can say *produces without consuming*, and CHE-222 (R03.5) is what
made it able to.** `OperationDescriptor.inputs` is a tuple of representation ports,
and `()` means this operation consumes no upstream representation. The three
sources and the problem-driven ray solve declare it, and `ENTRY_KINDS` restricts
`()` to `source`-kind — a coupler with no input would change the representation of
nothing, an operator the state of nothing, a measurement would observe nothing.
(It restricted `()` to `solver`-kind until CHE-224 (R15.1), which **contradicted
the sentence above it**: that set had to name `solver` because there was no
`SOURCE` member to name.)

This paragraph used to be a `[LANDING GATE]` recording the opposite: that
`OperationDescriptor.input` had no vocabulary for "no input representation", so a
source named the representation it *produces* on both sides — CHE-210 registered
`S_SOURCE_PLANE_WAVE` with `input='scalar_field'`, following the precedent R05.3
set for the ray solver. Two things about that deferral were wrong and are recorded
here rather than quietly dropped. It called the fake input "an imprecision in the
descriptor, not in this definition", when a record contradicting both the code and
this document is a false claim rather than an imprecision. And it named **R12's
capability graph** as the natural home, which put the fix behind the first
consumer: the closure belonged before a planner existed, so a planner would never
have to be written against ports it could not trust. `input` and `output` are gone
rather than aliased, and `find(entry=True)` selects the entry operations —
`input=None` still means "do not filter", because a filter with two readings would
have been worse than the fake input it replaced.

### coupler

Changes **representation** while preserving the same physical state at the same
physical boundary.

**Boundary against *physical operator*:** a coupler changes how the state is
*described*; an operator changes the physical state. Computational cost is not
the test. Ray-to-scalar and scalar-to-ray handoffs are couplers when they
preserve the same state and boundary.

### physical operator

Changes the **physical state**. Propagation and surface interactions such as
refraction or a diffractive step are operators, not couplers, even when their
implementation performs representation bookkeeping.

### measurement

Derives an **observable** from physical state.

**Boundary against *representation*:** its output is a number, profile, or image
*about* the state, not the state itself. PSF, Strehl ratio, and first-null radius
are measurements.

### composite operation

One callable that **fuses more than one primitive stage**, declaring the ordered
stages it fuses. `OperationDescriptor.composes` carries them; `kind` names the
**terminal** stage — where the operation leaves the state, and therefore which
boundary its output sits at. `()` is the default and means "this record is exactly
its `kind`", which is twelve of the thirteen landed records.

**A composite is not a fifth kind.** Every stage is a primitive from the four, and
the id prefix spells the composition rather than naming a new category: `SO_` is
source-then-operator. There is exactly one today, `SO_RAY_LAUNCH_TRACE`
(`backends.optiland.solver:trace`), which materializes and declares its rays and
then refracts them through every surface.

**Boundary against *the primitive kinds*:** a primitive record answers "what
happens to physical state" with one word. A composite exists only when one word is
a *false* claim — not when it would merely be a simplification. `trace` initializes
state and evolves it; calling it a source denies the refraction and calling it an
operator denies that it consumes nothing.

**Boundary against a route or a plan:** a composite is **not** a pipeline
description. It records that a fusion happened and which primitives it fused —
not the arguments each stage took, not their intermediate representations, and not
a way to execute them separately. Nothing can run a stage. A route is
`planning.routes`' business and lives outside the descriptor entirely.

*[JUDGEMENT]* **Why this exists rather than a `solve` kind, and what it is
labelling.** The honest decomposition is `launch` + `O_RAY_TRACE`, and it is
blocked by measured numerical facts rather than by taxonomy: an object at infinity
launches at `z = -EPD`, which is not the surface the trace starts from, and
`to_traced_ray_bundle` composes the optical path under a different declared
reference. Unifying moves frozen ray numbers, so it needs its own evidence and its
own ticket. A fifth primitive kind was rejected because the operation is two known
primitives fused, not a new category.

**The deeper reason a single `kind` could not express it** is that this schema has
no notion of *which reference surface* a returned representation sits at. A source
produces state where it initialized; `trace` returns state N interfaces
downstream. `inputs=()` plus `returns=("ray_bundle",)` is true of both. So
`composes` **labels that gap honestly; it does not close it.** Closing it is a
port-vocabulary change and is what the decomposition ticket needs first.

*[JUDGEMENT]* `O_DIFFRACTIVE_SURFACE` is internally coupler → operator → coupler
and deliberately declares **no** composition. Its input and output representation
types do not change and it presents a single operator-like transformation at its
boundary, so its net primitive kind is the whole truth about it *at the ports*.
Whether it should nonetheless expose that structure is an open design question to
revisit once the composition model settles — recorded, not decided.

*[LANDING GATE]* A composition is well-formed: at least two stages, every stage a
primitive kind, and `composes[-1] == kind`. A record whose `kind` disagrees with
its own last stage is refused at construction, which is what stops `kind` being a
free choice between stages — the shape that produced CHE-224's false claim.

### operation descriptor

The lightweight record used to **discover and reason about** execution: what an
operation consumes, what it produces, its kind, declared validity, and where its
implementation lives.

**Boundary against the implementation:** a descriptor is not the operation. The
target design stores an implementation path as data and resolves it lazily so
reading discovery metadata does not import solver backends.

*[JUDGEMENT]* Every classification in this section.

*[LANDING GATE]* When the relevant package lands, backend imports must stay
inside backend adapters and discovery metadata must not require importing
implementation packages. Add executable dependency/import tests with that
surface.

---

## 3. Dependency direction

The following is the **target** allowlist for packages that exist:

```
numerics/            -> (nothing in the project)
representations/     -> numerics
problems/            -> representations, numerics
operations/          -> numerics
backends/<backend>/  -> problems, representations, numerics (+ its backend)
sources/             -> problems, representations, numerics
couplers/            -> representations, numerics
operators/           -> representations, couplers, numerics
measurements/        -> representations, numerics
planning/            -> operations
runtime/             -> planning, operations, representations
```

Absence of any row from the current tree is valid. Do not create a package only
because it appears in this graph. A package is introduced when a scoped issue has
real code to put in it.

Five properties are load-bearing:

1. **It is an allowlist.** Each landed package declares what it *may* import;
   everything else is forbidden. A denylist fails open when new packages appear.
2. **`numerics/` is the intended bottom.** It exists only when precision, array,
   or numerical policy needs a shared home. It may not become a generic `core/`.
3. **`operations/` is a sibling, not a layer above implementations.** It stores
   metadata and import paths; it does not import solver/coupler implementations.
4. **`representations/` is backend-neutral.** A representation must not know
   which backend produced it.
5. **`sources/` owns initialization, and it is representation-independent as a
   package while each of its operations is representation-explicit.** The package
   may initialize any landed representation; a public operation declares exactly
   one. No subpackage per representation, and no constructor whose return
   representation depends on its arguments. See §2.

**`sources/` was added to this graph by CHE-210 (R06.5)**, as a deliberate
architecture change and not as a convenience. §2 defines the term; the row exists
because a source is the one operation with **no input representation**, so none of
the packages that already existed could hold it without changing what they are.
`representations/` would own initialization physics it exists only to *declare*;
`operators/` is wrong by definition, because an operator consumes a representation
and a source does not; `problems/` may hold a source *declaration* but the
constructor is not the problem; and `backends/<backend>/` is organized per backend,
which a source has none of. Widening an existing package's remit to make a source
fit is the move the allowlist exists to prevent, so the row was added instead.

The row is `sources/ -> problems, representations, numerics`. It reaches
`representations/` because it constructs one, `numerics/` because an initialized
state must respect the same dtype and device policy as everything downstream of
it, and `problems/` because a source may read a physical source declaration. It
may **not** import `backends/`, `couplers/`, `operators/` or `measurements/`: a
source is upstream of all of them by construction, and an edge in the other
direction would describe an initial state that cannot be created without the
thing that consumes it.

*[LANDING GATE]* Dependency enforcement is package-by-package. The change that
introduces a package must add it to the dependency classifier/checker if one has
already landed; the first change that creates cross-package dependencies must
land the checker itself. A new package must never appear outside the allowlist
without an explicit architecture change.

`src/io/` is **reserved and must not be introduced** under a flat `src/`
namespace, because it can collide with the Python standard-library `io` module.

*[LANDING GATE]* Once top-level package import checks exist, include this
collision case in them.

*[JUDGEMENT]* If artifact serialization is needed, prefer `runtime/records.py` so
serialization stays out of physical representations. Do not create the module
before there is a real record to serialize.

---

## 4. Class minimality

A class is justified only if at least one of these is true:

1. several fields share an invariant that must be enforced together;
2. it is a public serialized/versioned data model;
3. it owns a genuine mutable resource lifecycle;
4. at least two *current* implementations require runtime polymorphism; or
5. it is a real plugin boundary used by the runtime or registry.

Otherwise prefer a function, module, frozen dataclass, `TypedDict`, tuple,
`Literal`, or `Enum`.

"*Current*" matters. Two implementations, one of which is hypothetical, is one
implementation.

*[JUDGEMENT]* Whether a class satisfies one of the five rules.

The clean rewrite does not inherit a project-wide numeric class ceiling from the
previous implementation. A fixed ceiling is useful only after the new tree has
enough real code to establish meaningful package budgets.

*[LANDING GATE]* If class-count automation is introduced, its budgets must be
reviewed from the new architecture rather than copied from the old tree. The
script may make growth visible; it cannot replace the five-rule review above.

No class, protocol, abstract base class, or placeholder interface exists merely
"so the structure is visible." Package documentation can describe a future
boundary; code arrives with the issue that has a real implementation to put
there.

---

## 5. What this project is not

*[JUDGEMENT]* The repository is not a standalone optical application, standalone
ray tracer, standalone wave solver, benchmark product, lens catalog, demo
collection, or archive of previous implementations.

The new production `src/` tree is clean-slate. Do not copy historical source into
`src/` under names such as `legacy/`, `archive/`, `old/`, `compatibility/`, or
`deprecated/`.

Historical implementations may be consulted through Git history/tags and explicit
reference documentation. They are evidence for scientific behaviour and failure
modes, not architecture to preserve.

---

## 6. Scientific rules that outrank structural convenience

These rules apply from the first line of scientific code. Where a scientific rule
conflicts with structural convenience, the scientific rule wins.

* *[JUDGEMENT]* Complex fields are amplitudes, not intensities. Use SI internally unless a scoped contract explicitly defines and tests another convention. A representation declares units and normalization; whether the numbers satisfy the declaration is a physics review.
* *[JUDGEMENT]* A solver call succeeding does not prove its approximation is appropriate. Validity claims require scientific justification.
* *[JUDGEMENT]* Never claim a gradient across an untested boundary. A cross-framework handoff is `forward_only` until finite-difference validation (or an equivalent independent check) supports the gradient claim.
* *[LANDING GATE]* Failed or unsupported paths must not invent fields, metrics, convergence, or provenance. When a failure vocabulary/diagnostic model lands, make its reachable failure modes executable and test them.
* *[JUDGEMENT]* Do not widen a tolerance merely to make a benchmark pass. Report the open gate and require a derivation for tolerance changes.
* *[LANDING GATE]* Repository numerical code must not be the sole correctness oracle for the same numerical code. When benchmark/oracle infrastructure lands, encode provenance or independence strongly enough that a shared-code oracle cannot decide a correctness gate.

The previous implementation may contain executable examples of these protections.
Reusing the idea is encouraged; claiming the new tree is protected requires the
new executable check to exist.
