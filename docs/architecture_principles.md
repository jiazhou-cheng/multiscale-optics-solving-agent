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

There are four operation kinds — `solver`, `coupler`, `physical_operator`, and
`measurement` — and they are **descriptor metadata, not four class hierarchies**.
The kind is a field on one operation descriptor.

The previous implementation expressed these kinds as families of base classes,
request/result envelopes, and per-family diagnostics. The clean rewrite must not
recreate that shape by default.

*[JUDGEMENT]* Whether a specific concept is a representation or an operation.

*[LANDING GATE]* When operation discovery/registration lands, the executable
contract must represent the four kinds as descriptor data. Do not introduce a
base-class family per kind unless a later issue demonstrates a real runtime need
that satisfies the class-minimality rules below.

---

## 2. The seven terms

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

### solver

Maps a **problem** into a physical representation. A solver adapter is the
boundary at which an external backend's API, conventions, compatibility logic, or
version requirements may appear.

**Boundary against *coupler*:** a solver is where an external solver enters the
system; a coupler is repository-owned physics between representations. Backend
imports belong in `solvers/<backend>/` once that adapter exists.

### source

Owns the **physically meaningful initialization** of a representation: how the
state of a representation is created from physical source parameters.

A representation defines the *structure and conventions* of physical state at a
declared boundary. A source defines how that state is *initialized*. A plane-wave
source initializes the complex amplitude and phase of a `ScalarField` from its
wavelength, propagation direction, amplitude, sampling grid, and reference
surface; a collimated ray source initializes a `RayBundle` from its spatial
sampling and common propagation direction.

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

**Kind:** `solver`. A source maps a problem statement into a representation, which
is this document's definition of a solver, and there is no fifth operation kind.
What separates `sources/` from `solvers/<backend>/` is that a source has **no
external backend**: it is the project's own arithmetic on the project's own grid,
so per-backend organization has nothing to organize.

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

*[LANDING GATE]* `OperationDescriptor.input` has no vocabulary for "no input
representation", so a landed source currently names the representation it produces
on both sides — CHE-210 registered `S_SOURCE_PLANE_WAVE` with
`input='scalar_field'`, following the precedent R05.3 set for the ray solver,
which also names the representation it works in rather than the problem it
consumes. That is an imprecision in the descriptor, not in this definition. The
ticket that gives a descriptor a way to say *produces without consuming* — R12's
capability graph is the natural home — closes it.

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

### operation descriptor

The lightweight record used to **discover and reason about** execution: what an
operation consumes, what it produces, its kind, declared validity, and where its
implementation lives.

**Boundary against the implementation:** a descriptor is not the operation. The
target design stores an implementation path as data and resolves it lazily so
reading discovery metadata does not import solver backends.

*[JUDGEMENT]* Every classification in this section.

*[LANDING GATE]* When the relevant package lands, backend imports must stay
inside solver adapters and discovery metadata must not require importing
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
solvers/<backend>/   -> problems, representations, numerics (+ its backend)
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
constructor is not the problem; and `solvers/<backend>/` is organized per backend,
which a source has none of. Widening an existing package's remit to make a source
fit is the move the allowlist exists to prevent, so the row was added instead.

The row is `sources/ -> problems, representations, numerics`. It reaches
`representations/` because it constructs one, `numerics/` because an initialized
state must respect the same dtype and device policy as everything downstream of
it, and `problems/` because a source may read a physical source declaration. It
may **not** import `solvers/`, `couplers/`, `operators/` or `measurements/`: a
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
