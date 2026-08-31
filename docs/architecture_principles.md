# Architecture Principles

**CHE-172 (R01.2).** The vocabulary the whole rewrite is judged against, the
dependency direction, and the class-minimality rule.

Every principle below is labelled one of two ways, and there is no third
category:

* **[ENFORCED]** — a named script fails when it is violated. If you disagree with
  the principle, you have to change the script, and that is a reviewable act.
* **[JUDGEMENT]** — nothing can check it. It is a call a reviewer makes, and
  saying so is the point: an unlabelled principle reads as enforced and is not.

A principle with neither label is a bug in this document.

---

## 1. Two concepts

The system has exactly two kinds of thing.

**Representations** are physical state. **Operations** consume and produce
representations, problems or measurements.

There are four *kinds* of operation — `solver`, `coupler`, `physical_operator`,
`measurement` — and they are **descriptor metadata, not four class hierarchies**.
The kind is a field on an operation descriptor. This is the single most important
sentence in this document: the reference implementation expressed the same four
kinds as four families of base classes, request/result envelope pairs and
per-family diagnostics types, and that is a large part of how it reached 280
production classes.

*[JUDGEMENT]* Whether a given thing is a representation or an operation.
*[ENFORCED]* That the kinds do not become class hierarchies — indirectly, by
`scripts/class_budget.py`, which makes the class count of the attempt visible.

---

## 2. The six terms

Each is defined by the boundary that separates it from its nearest neighbour,
because every one of these distinctions has been got wrong in this repository
before.

### representation

Describes physical state at a declared boundary, with its conventions explicit
and testable: units, axes, frame, handedness, wavelength, phasor sign,
polarization, coherence, normalization, sampling, reference plane.

**Boundary against *measurement*:** a representation is state; a measurement is
an observable *derived* from state. **PSF is not a representation.** It is what
you get when you point a detector at a field, and the reference implementation
already learned this the expensive way — a `C_FIELD_TO_PSF` "coupler" was retired
because a trivial observable is not a cross-representation handoff.

**Boundary against a serialized record:** being serializable does not make
something a representation. A record of a run is not physical state.

Exactly **one** public ray representation and **one** public scalar-field
representation. Coherence is a *stronger contract* on the ray representation
(`require_coherent()`), not a subtype: a coherent bundle is the same physical
state with an additional guarantee, and expressing it as a subclass forces every
consumer to know which one it has.

### solver

Maps a **problem** into a physical representation. It is the only place an
external backend's API, conventions or version pin may appear.

**Boundary against *coupler*:** a solver is where the outside world is; a coupler
is where our physics is. If it imports `optiland` or `chromatix`, it is a solver
adapter, and it lives in `solvers/<backend>/`.

### coupler

Changes **representation** while preserving the same physical state at the same
physical boundary.

**Boundary against *physical operator*, and this is the one that gets confused:**
a coupler changes how the state is *described*; an operator changes the state.
A coupler may do a great deal of heavy numerical work — a k-space reconstruction
or an angular-spectrum decomposition is not cheap — and **that does not make it an
operator**. The test is not effort. The test is whether the light is somewhere
else, or different, afterwards. Ray→scalar and scalar→ray are couplers.

### physical operator

Changes the **physical state**.

Propagation is an operator. A surface interaction — refraction, a diffractive
step — is an operator. **Never couplers**, however much representation
bookkeeping they happen to require.

### measurement

Derives an **observable** from physical state.

**Boundary against *representation*:** its output is a number, a profile or an
image *about* the state, not the state. Serializability does not promote it. PSF,
Strehl ratio and first-null radius are measurements.

### operation descriptor

The one lightweight record used to **discover and reason about** execution: what
an operation consumes, what it produces, its kind, its declared validity, and
**where its implementation lives, as a string**.

**Boundary against the implementation:** a descriptor is not the operation. It
holds an import path and resolves it lazily, which is what makes "reading the
registry pulls in no backend" a structural fact rather than a discipline someone
maintains. Not four class hierarchies; one record with a `kind` field.

*[JUDGEMENT]* Every classification in this section.
*[ENFORCED]* Only the consequences: that a backend appears solely under
`solvers/` and that `operations/` cannot import an implementation package
(`scripts/check_dependencies.py`).

---

## 3. Dependency direction

**[ENFORCED] — `scripts/check_dependencies.py`, run in the default suite by
`tests/unit/test_dependency_direction.py`.**

*With one honest qualification.* The checker walks the packages that have been
authored, which at R01 is `numerics/` and `representations/`. The other eight rows
are **declared now and enforced per package as it lands** — their rules are
unit-tested at the classifier level, so the direction is real code rather than
prose, but they govern no modules until the packages exist. The checker's `LANDED`
set is what makes that explicit, and it fails if a new-architecture package
appears on disk without being added to it.

```
numerics/            -> (nothing in the project)
representations/     -> numerics
problems/            -> representations, numerics
operations/          -> numerics
solvers/<backend>/   -> problems, representations, numerics (+ its pinned backend)
couplers/            -> representations, numerics
operators/           -> representations, couplers, numerics
measurements/        -> representations, numerics
planning/            -> operations
runtime/             -> planning, operations, representations
```

Four things about this graph are load-bearing:

1. **It is an allowlist.** Each package declares what it *may* import; everything
   else fails. A denylist fails open — a package added later is unconstrained
   until someone remembers to add it to every other package's forbidden set.
2. **`numerics/` imports nothing.** It is the bottom, and anything it imported
   would join the bottom. It exists because precision and array policy need a
   home and `core/` is banned: a package that names no domain accumulates
   whatever has no other home, which is how the reference implementation put 110
   classes in `core/`. `numerics/` names one job.
3. **`operations/` is a sibling, not a layer above.** It may not import
   `solvers/` or `couplers/`, and they may not import it. It holds import paths
   as strings.
4. **`representations/` may not import a backend.** A representation that knows
   which package produced it has stopped being neutral ground. The reference
   implementation's two solver↔coupler import cycles both began with an artifact
   living wherever its first consumer was.

`src/io/` is banned: it would shadow the stdlib `io` under the flat `src/`
namespace root. *[ENFORCED]* — but by a different check than this section's
header:
`tests/test_flat_layout.py::test_every_top_level_name_resolves_inside_this_repository`
catches it, because `import io` returns the already-cached stdlib module and the
assertion fails.

*[JUDGEMENT]* Artifact serialization lives in `runtime/records.py`. Nothing checks
the location; it is written down once here rather than re-argued.

---

## 4. Class minimality

**[ENFORCED, partially] — `scripts/class_budget.py`, run in the default suite by
`tests/unit/test_class_budget.py`.**

A class is justified only if:

1. several fields share an invariant enforced together;
2. it is a public serialized / versioned data model;
3. it owns a genuine mutable resource lifecycle;
4. at least two *current* implementations need runtime polymorphism;
5. it is a real plugin boundary used by the runtime or registry.

Otherwise the answer is a function, a module, a frozen dataclass, a `TypedDict`,
a tuple, a `Literal` or an `Enum`.

Note "*current*" in rule 4. Two implementations one of which is hypothetical is
one implementation.

**What is enforced and what is not, precisely.** The script *counts* production
classes per package against a declared budget, and fails on unjustified growth
and on the budgets summing past the project ceiling of 22. It **cannot** decide
whether a class satisfies one of the five rules — *[JUDGEMENT]*. So the budget
number is the reviewed artifact: raising it requires a ticket to name which rule
each new class satisfies, and the script's job is to make an unjustified raise
visible instead of silent.

Every implementation ticket reports **production classes added / deleted /
functions added**. Unnecessary net growth in class count is a design failure.

*[JUDGEMENT]* One more thing no script can catch: a class created "so the
structure is visible". No base class, protocol or placeholder interface exists to
show where something will go later. The package docstring says it; the code
arrives with the ticket that has something true to put in it.

---

## 5. What this project is not

*[JUDGEMENT]*, all of it, and worth stating because each of these has grown here
before: not a standalone optical application, not a standalone ray tracer, not a
standalone wave solver, not a benchmark framework, not a catalog of lens systems,
not a collection of demos, and not an archive of previous implementations.

`legacy/`, `archive/`, `old/`, `compatibility/` and `deprecated/` do not exist
inside the production source.

---

## 6. Scientific rules that outrank all of the above

These are in `AGENTS.md` and are repeated here only so this document cannot be
read as the complete picture. Where they conflict with a structural principle,
they win.

Labelled individually, because they are not uniform — some are checkable and some
are irreducibly a reviewer's call, and an unlabelled list would hide which:

* *[JUDGEMENT]* Complex fields are amplitudes, not intensities. SI internally. A
  representation declares its units and normalization; whether the declaration is
  *true* of the numbers is a physics review.
* *[JUDGEMENT]* A solver call succeeding does not prove the approximation is
  appropriate. Nothing can check this, and it is the reason operations declare a
  validity regime at all.
* *[JUDGEMENT]* Never claim a gradient across an untested boundary; a
  cross-framework handoff is `forward_only` until finite-difference validation
  passes. The *claim* is data and becomes checkable against its evidence when R12
  lands the capability graph; today it is a reviewer's call.
* *[JUDGEMENT at R01; ENFORCED once the failure vocabulary lands]* Never invent
  fields, metrics, convergence or provenance — failed paths return structured
  diagnostics. The reference implementation enforced this
  (`tests/test_contract_code_reachability.py` proved every failure code was
  reachable) and the new tree owes the equivalent.
* *[JUDGEMENT]* Do not widen a tolerance to make a benchmark pass; report the open
  gate. Widening is always locally defensible, which is precisely why no script
  can adjudicate it. The countermeasure is that a tolerance carries a derivation.
* *[ENFORCED in the reference implementation, and owed again]* **Our own numerical
  code never decides correctness for our own numerical code.** An oracle sharing
  code with what it tests is characterization, not a gate. This *was* executable —
  the claim ledger refused to let a `shares_code` claim be gate-deciding, and a
  test enforced it — and R14 deletes that ledger. Until it is re-established the
  rule is prose here, which is a downgrade this document names rather than hides.
