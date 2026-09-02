"""One record per executable operation, with the four kinds as metadata.

CHE-177 (R03.1). The reference implementation described the same information
with 23 pydantic classes in `core/specs.py` plus 45 KB of YAML mirroring a
358-line Python table. This is one frozen dataclass and one `StrEnum`, and the
difference is not terseness for its own sake: every one of those 23 classes was
a field group some consumer wanted, and each new group made the next one look
cheap.

Why the kind is a field and not a hierarchy
-------------------------------------------
`solver`, `coupler`, `physical_operator` and `measurement` differ in what they
*mean physically*, not in what a caller does with the record. Discovery,
capability queries and lazy resolution are identical for all four, so four
subclasses would share every field and override nothing. The distinction that
matters -- a coupler changes representation while a physical operator changes
physical state -- is enforced by review and by the boundary tests of the packages
that implement them, and no `isinstance` check would have caught a coupler that
quietly propagated.

Field creep is the failure this record is designed against
----------------------------------------------------------
A field exists here only if a *current* consumer reads it. R12's capability graph
and R13's runtime are the only planned consumers, and both postdate this ticket,
so the fields below are the ones the ticket names and nothing more. In
particular there is no `maturity`, no `tags`, no `version`, no `source`, no port
*objects* and no separate spec type per kind.

`capabilities` is a **citation, not a copy**. Device and dtype support is
measured, and the measurement lives in `knowledge/capabilities/<component>.json`
with the probe that produced it. Restating either here would recreate exactly the
two-source arrangement R03 exists to remove -- the old tree had a passing test
asserting its two sources agreed, which made the duplication feel safe rather
than removable. A descriptor names a component; the record stays where the
evidence is.

**And the citation is validated by format, not by membership** -- CHE-223 (R03.6).
`__post_init__` used to check `capabilities` against
`numerics.COMPONENT_CAPABILITIES`, which meant constructing any descriptor
required the concrete measured table to be importable. That was an asymmetry
inside one dataclass: `implementation` is a lazily resolved string and
`capabilities` was a string *plus* an eager global. It was also the last thing
pinning the rows into the foundational layer. Now both fields are references,
checked for shape here and for resolution by
`tests/operations/test_capability_references.py` -- the same division `resolve`
already had.

What a planner has to be able to ask, and CHE-222 (R03.5)
---------------------------------------------------------
R03.1 described every operation as `one representation in -> one representation
out`. Four shapes in the landed tree are not that, and for two of them the record
was **false**: `S_SOURCE_PLANE_WAVE` declared `input="scalar_field"` for a function
that consumes no field, and `S_RAY_OPTILAND` declared `input="ray_bundle"` for one
that consumes no bundle. Both contradicted the code, `sources/__init__.py` and
`docs/architecture_principles.md` §2, which all say a source is the one operation
with no input.

The fields below are the minimum that answers eight questions and nothing else.
The specification *is* those eight questions -- a field that answers none of them
does not belong here, which is the discipline this record's own docstring already
demanded and had no list to check against:

1. is an upstream representation edge required?  -> `inputs`, `is_graph_entry`
2. what representation inputs are required?      -> `inputs`, in call order
3. what other required values must be supplied?  -> `requires`
4. which parameters are optional?                -> `optional`
5. what is the primary returned value?           -> `primary_output`
6. is auxiliary data also returned?              -> `returns_auxiliary`
7. do two records over one callable stay distinct? -> `operation_id`, below
8. does any record claim an input its callable does not take? -> nothing here;
   `tests/operations/test_catalog_signatures.py` derives all four tuples from the
   resolved signature and compares, so the answer is checked against the code
   rather than against a table.

`input` and `output` are **gone**, not aliased. Shipping `input` beside `inputs`
as a convenience would put two spellings of one fact in one dataclass, which is
the two-source arrangement R03 exists to remove. The readers were
`operations.catalog`, `registry.find`, `matches` and the implementation tests, and
all four moved in the same change.

This is deliberately **not a schema language.** It carries no types, no default
*values* (`diffractive_surface` has one required parameter and sixteen optional
ones; copying sixteen defaults would guarantee drift), and validates nothing about
arguments. `requires` and
`optional` are parameter *names*, and the units in those names --
`distance_m`, `focal_length_m`, `sample_pitch_m`, `phase_budget_rad` -- are part
of the public contract rather than decoration, which is why the names are carried
verbatim from the signature.

Semantic identity is not implementation identity
------------------------------------------------
Two descriptors may name one callable and remain two distinct planning choices.
`S_WAVE_CHROMATIX` (`solver`) and `O_ASM_PROPAGATE` (`physical_operator`) both
resolve to `solvers.chromatix.solver:propagate` with different `approximation` and
`validity`: one answers "what backend does this project drive", the other "what
happens to the physical state". **Nothing may deduplicate the catalog by
implementation string.** The id is the planning identity.

Nothing here imports a backend, and since CHE-223 (R03.6) this module imports
nothing from this project at all: `implementation` is a string, `capabilities` is a
string, and `operations.resolve` is the only function in the package that turns
either kind of reference into anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DERIVATIVE_MODES",
    "ENTRY_KINDS",
    "OBSERVABLE_TYPES",
    "SEMANTIC_TYPES",
    "OperationDescriptor",
    "OperationKind",
]


class OperationKind(StrEnum):
    """The four kinds, as metadata on one record.

    Exactly four members, and the set is closed: these are the four operation
    kinds `docs/architecture_principles.md` defines. A fifth kind is an
    architecture change, not a registry entry.
    """

    SOLVER = "solver"
    COUPLER = "coupler"
    PHYSICAL_OPERATOR = "physical_operator"
    MEASUREMENT = "measurement"


#: The semantic types an operation may consume or produce, as plain strings.
#:
#: Strings and not `representations` classes, because `operations/` must not
#: import `representations/`: the registry has to be readable without loading the
#: physical data model, and importing it here would put the whole representation
#: layer behind every capability query.
#:
#: **Three entries: the two boundaries R02 landed -- `RayBundle` and
#: `ScalarField` -- and the one measurement result type R11.1 landed, `psf`.**
#: `psf` is here as the *output port of a measurement*, and it is deliberately not
#: a representation: `measurements/psf.py` says why. `OBSERVABLE_TYPES` below is
#: what keeps that distinction from being prose.
#:
#: This is deliberately not the reference
#: implementation's `ArtifactKind`, which enumerated 26 members of which the tree
#: could produce a handful; the rest read as capability the project did not have.
#: A problem type, a measurement result type or a second field type joins this
#: tuple in the ticket that lands the boundary it names, in the same change --
#: the same discipline `scripts/check_dependencies.py` applies with `LANDED`.
#:
#: A closed vocabulary is also what lets `operations.find` refuse an unknown
#: semantic type instead of returning an empty result, which is how a typo in a
#: query becomes "there is no operation for that" rather than an error.
SEMANTIC_TYPES: tuple[str, ...] = (
    "ray_bundle",
    "scalar_field",
    "psf",
)

#: The subset of `SEMANTIC_TYPES` that are **observables**, not representations.
#:
#: An observable is derived from physical state; it is not physical state at a
#: boundary. Two rules follow, and `__post_init__` enforces both:
#:
#: * **only a `measurement` may produce one.** A coupler that produced a `psf`
#:   would be `C_FIELD_TO_PSF`, which CHE-36 removed from the reference registry
#:   for changing no representation and consulting no convention it did not
#:   already hold; a trivial observable in the coupler list, complete with a
#:   framework and a derivative mode it had no numerics for, made the category
#:   unfalsifiable. R11's acceptance criterion 3 says it must not come back, and
#:   this is where "must not" becomes a construction error.
#: * **nothing may consume one.** An observable is terminal. An operation reading
#:   a PSF as its input is either a second measurement of a measurement, or a
#:   physical operation that has mistaken an intensity for a state -- and the
#:   representation it should have consumed is still sitting upstream.
OBSERVABLE_TYPES: frozenset[str] = frozenset({"psf"})

#: The operation kinds that may be a **graph entry** -- `inputs=()`, no upstream
#: representation edge.
#:
#: Only `solver`. Two cases are real and both are `solver`-kind per
#: `docs/architecture_principles.md` §2: a **source**, which initializes a
#: representation from source parameters alone, and a **problem-driven solve**
#: (`S_RAY_OPTILAND`), which turns an `OpticalSetup` plus a `SourceSpec` into a
#: bundle. A coupler with no input would change the representation of nothing; a
#: physical operator with no input would change the state of nothing; a measurement
#: with no input would observe nothing. `__post_init__` refuses all three, which is
#: the check that makes `inputs=()` an honest declaration rather than a hole.
ENTRY_KINDS: frozenset[str] = frozenset({"solver"})

#: The shape of a component id, which is what `capabilities` cites.
#:
#: Screaming snake case, starting with a letter. Deliberately permissive about the
#: *prefix*: the two measured records today are component-level (`M_RAY_OPTILAND`,
#: `M_WAVE_CHROMATIX`) because that is what the probes measured -- the packages'
#: device and dtype behaviour, not any one semantic operation -- and CHE-223 keeps
#: the door open for an operation-level record when one is independently measured.
#: A pattern that demanded `M_` would forbid that; one that accepted anything would
#: admit a typo like `"m_ray_optiland"` as a citation nothing resolves.
#:
#: Shape only. Whether a record exists is a question about the knowledge pack, and
#: answering it here would put the concrete table back behind every construction.
_COMPONENT_ID = re.compile(r"[A-Z][A-Z0-9_]*")

#: What may be claimed about differentiating through an operation.
#:
#: Two values, because the project's rule is binary: a gradient is either
#: validated across this boundary or it is not. The old tree's nine-member
#: `DerivativeMode` described *how* a gradient would be taken, which is a
#: property of an implementation nobody had yet; what a caller needs to know is
#: whether it may trust one.
DERIVATIVE_MODES: tuple[str, ...] = (
    "forward_only",
    "differentiable",
)


@dataclass(frozen=True, slots=True)
class OperationDescriptor:
    """Everything needed to discover an operation, and nothing needed to run it.

    Minimality rules 2 and 5: it is the public data model planning and the
    runtime read, and it is the plugin boundary between an operation and the
    layer that selects one. Constructing it imports no backend and no
    implementation module.

    Fields:

    `operation_id`
        Stable identity. What `resolve` takes and what a plan records, and the
        **planning** identity: two records may share an `implementation`.
    `kind`
        One of the four. Accepts the string form for convenience and normalizes.
    `inputs`
        The **representation** ports the callable consumes, as semantic types from
        `SEMANTIC_TYPES`, **in call order**. `()` means this operation consumes no
        upstream representation and is a graph entry -- see `ENTRY_KINDS`, which is
        what makes `()` a declaration rather than an omission.

        A tuple rather than one string because two landed operations genuinely take
        more than one thing: `trace_rays(setup, rays, *, execution)` has one
        representation port and **two** required non-representation inputs, and the
        schema has to be able to tell it from `propagate_rays(rays, *, to)`, which
        also has one port but requires something else entirely. Written as `("ray_bundle",)` for the
        common single-port case; nothing today has two ports, and the tuple is what
        lets one land without a schema change.
    `returns`
        The returned values **in order**, element 0 being the primary semantic
        result. `("scalar_field", "reconstruction_diagnostics")` for
        `ray_to_scalar`; `("ray_bundle",)` for `propagate_rays`; `("psf",)` for
        `psf`, whose callable returns a `PsfResult` record carrying the observable.

        This is what `output` could not say. `output="ray_bundle"` was identical for
        `propagate_rays`, which returns a bundle, and `diffractive_surface`, which
        returns a 2-tuple -- so a runtime reading only the descriptor would either
        unpack a `RayBundle` or fail to unpack a tuple, and the only way to know
        which was a switch keyed on `operation_id`. That switch is a second
        per-operation database, which is the thing this record exists to prevent.

        `returns[0]` must be a `SEMANTIC_TYPES` member; `returns[1:]` must **not**
        be, so an auxiliary value cannot be mistaken for a representation a planner
        could route. A second *representation* output is therefore not expressible,
        which is correct today -- nothing produces one -- and is a schema change for
        the ticket that lands one rather than a state to leave ambiguous.
    `requires`
        Parameter **names** of the required non-representation inputs, in signature
        order. Twelve of the fourteen landed records need one -- every record except
        `O_COMPLEX_TRANSMISSION` and `C_SCALAR_TO_RAY`, for both of which a field
        plus nothing else is a complete call -- and a planner that cannot see them
        cannot call anything. `psf` is the sharpest small case --
        `normalization` is keyword-only with no default, and which normalization
        was used is the subject of three of R11's acceptance criteria, so a runtime
        must not pick one for the caller.

        Names, not types and not values. The unit in the name is part of the
        contract, which is why they are carried verbatim.
    `optional`
        Parameter names that are part of the public contract and have defaults.
        Names only -- mirroring `diffractive_surface`'s sixteen defaults would
        guarantee drift against the signature.
    `implementation`
        `"module.path:attribute"`. A **string**, resolved only on request.
    `approximation`
        What the operation approximates and what error that introduces. Required
        and free text: a physical claim in a sentence a reviewer can check beats
        an enum member that compresses it to a word.
    `evidence`
        References to the probes, tests or records behind the claims above.
        Required with no default -- an empty tuple is allowed, but it has to be
        *written*, so "no evidence yet" is a statement rather than an omission.
    `validity`
        Conditions under which the operation is applicable at all.
    `capabilities`
        The id of a component with a measured capability record under
        `knowledge/capabilities/`, or `None`. Cited, never copied, and checked for
        **shape** rather than for membership -- see above. `None` is the honest
        citation for an operation with no measured device/dtype row of its own,
        which is every operation that imports no backend.
    `cost`
        Scaling information when it is known, `None` when it is not.
    `derivative` / `derivative_evidence`
        See `__post_init__`.
    """

    operation_id: str
    kind: OperationKind
    #: `()` for a graph entry. Required rather than defaulted, so "no upstream
    #: representation" has to be written down.
    inputs: tuple[str, ...]
    returns: tuple[str, ...]
    implementation: str
    approximation: str
    evidence: tuple[str, ...]
    requires: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    validity: tuple[str, ...] = ()
    capabilities: str | None = None
    cost: str | None = None
    derivative: str = "forward_only"
    derivative_evidence: str | None = None

    def __post_init__(self) -> None:
        """Normalize the two coercible fields, then refuse an unusable record.

        Every check below is a way a descriptor could claim something it has not
        got. They are collected and raised together, because a record with three
        problems should not take three edits to find out about.

        The derivative rule is the scientific one and it is not negotiable here:
        `forward_only` is the default, and `differentiable` requires
        `derivative_evidence` naming a finite-difference (or equivalent)
        validation. AGENTS.md forbids claiming a gradient across an untested
        boundary; this is where that becomes a construction error rather than a
        convention.
        """
        for name in ("inputs", "returns", "evidence", "requires", "optional", "validity"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        problems: list[str] = []

        try:
            object.__setattr__(self, "kind", OperationKind(self.kind))
        except ValueError:
            problems.append(
                f"`kind` is {self.kind!r}; the four kinds are "
                f"{[k.value for k in OperationKind]}"
            )

        if not self.operation_id or self.operation_id.strip() != self.operation_id:
            problems.append("`operation_id` must be a non-empty string with no surrounding space")

        for index, port in enumerate(self.inputs):
            if port not in SEMANTIC_TYPES:
                problems.append(
                    f"`inputs[{index}]` is {port!r}, which is not a declared semantic type "
                    f"{list(SEMANTIC_TYPES)}. Add it in the ticket that lands the boundary "
                    "it names; an operation cannot introduce one by using it."
                )
            elif port in OBSERVABLE_TYPES:
                problems.append(
                    f"`inputs[{index}]` is {port!r}, which is an observable and not a "
                    "representation. Nothing consumes an observable: it is derived from "
                    "physical state and the state it was derived from is still upstream."
                )

        # A graph entry, and the three kinds that cannot be one. `inputs=()` became
        # expressible in CHE-222, which is what makes this refusal necessary: a
        # coupler with no input would change the representation of nothing, an
        # operator the state of nothing, a measurement would observe nothing.
        if not self.inputs and str(self.kind) not in ENTRY_KINDS:
            problems.append(
                f"`inputs` is empty but `kind` is {getattr(self.kind, 'value', self.kind)!r}. "
                f"Only {sorted(ENTRY_KINDS)} may be a graph entry: a source initializes a "
                "representation from source parameters alone, and a problem-driven solve "
                "turns a problem statement into one. A coupler changes the representation "
                "of something, an operator changes the state of something, and a "
                "measurement observes something."
            )

        if not self.returns:
            problems.append(
                "`returns` is empty. Every operation returns something, and element 0 is "
                "the primary semantic result a planner routes onward."
            )
        else:
            primary = self.returns[0]
            if primary not in SEMANTIC_TYPES:
                problems.append(
                    f"`returns[0]` is {primary!r}, which is not a declared semantic type "
                    f"{list(SEMANTIC_TYPES)}. The primary result is what the next operation "
                    "consumes, so it has to be a declared boundary."
                )
            if primary in OBSERVABLE_TYPES and self.kind is not OperationKind.MEASUREMENT:
                problems.append(
                    f"`returns[0]` is the observable {primary!r} but `kind` is "
                    f"{getattr(self.kind, 'value', self.kind)!r}. Only a measurement produces "
                    "an observable. A coupler that produced one would be C_FIELD_TO_PSF, "
                    "which changes no representation and consults no convention it does not "
                    "already hold -- CHE-36 removed it, and R11 criterion 3 keeps it out."
                )
            for index, auxiliary in enumerate(self.returns[1:], start=1):
                if not str(auxiliary).strip():
                    problems.append(f"`returns[{index}]` is an empty name")
                elif auxiliary in SEMANTIC_TYPES:
                    problems.append(
                        f"`returns[{index}]` is {auxiliary!r}, a declared semantic type. Only "
                        "`returns[0]` may be one: an auxiliary value is diagnostics a caller "
                        "reads, not a representation a planner can route, and a second "
                        "routable output is a schema change for the ticket that lands one."
                    )

        for field_name in ("requires", "optional"):
            for index, parameter in enumerate(getattr(self, field_name)):
                if not str(parameter).strip():
                    problems.append(f"`{field_name}[{index}]` is an empty parameter name")
                elif parameter in SEMANTIC_TYPES:
                    problems.append(
                        f"`{field_name}[{index}]` is {parameter!r}, which is a semantic type "
                        "rather than a parameter name. Representation ports go in `inputs`; "
                        f"`{field_name}` names arguments the callable takes."
                    )
        overlap = set(self.requires) & set(self.optional)
        if overlap:
            problems.append(
                f"{sorted(overlap)} appear in both `requires` and `optional`. A parameter "
                "either has a default or it does not."
            )
        if ":" not in self.implementation or self.implementation.startswith(":"):
            problems.append(
                f"`implementation` is {self.implementation!r}; it must be "
                "'module.path:attribute' so it can be resolved without being guessed at"
            )
        if not self.approximation.strip():
            problems.append(
                "`approximation` is empty. What an operation approximates, and the error "
                "that introduces, is the part a caller cannot recover from the code."
            )
        if any(not str(item).strip() for item in self.evidence):
            problems.append("`evidence` contains an empty reference")
        if self.capabilities is not None and not _COMPONENT_ID.fullmatch(self.capabilities):
            problems.append(
                f"`capabilities` cites {self.capabilities!r}, which is not the shape of a "
                f"component id ({_COMPONENT_ID.pattern}). Device and dtype support is "
                "measured and lives with its probe under knowledge/capabilities/; a "
                "descriptor cites a component and nothing here loads one, so whether the "
                "record EXISTS is checked by "
                "tests/operations/test_capability_references.py rather than at "
                "construction. Citing a well-formed id with no record behind it claims a "
                "measurement nobody took, and that test is what catches it."
            )
        if self.derivative not in DERIVATIVE_MODES:
            problems.append(
                f"`derivative` is {self.derivative!r}; the declared modes are "
                f"{list(DERIVATIVE_MODES)}"
            )
        elif self.derivative != "forward_only" and not (self.derivative_evidence or "").strip():
            problems.append(
                f"`derivative` is {self.derivative!r} but `derivative_evidence` is empty. A "
                "gradient across this boundary is a claim, and the project makes it only "
                "with a finite-difference or equivalent validation to cite."
            )

        if problems:
            raise ValueError(
                f"operation descriptor {self.operation_id!r} is not usable:\n  "
                + "\n  ".join(problems)
            )

    @property
    def is_graph_entry(self) -> bool:
        """Whether this operation consumes no upstream representation.

        Question 1. `inputs == ()`, which `ENTRY_KINDS` restricts to `solver`-kind,
        so this is also "is this a source or a problem-driven solve".
        """
        return not self.inputs

    @property
    def primary_output(self) -> str:
        """The semantic type of the primary returned value. Question 5.

        One field access for every record, with no `operation_id` switch -- which is
        the whole reason `returns` is ordered rather than a set.
        """
        return self.returns[0]

    @property
    def returns_auxiliary(self) -> bool:
        """Whether anything is returned beside the primary result. Question 6.

        `True` for exactly `ray_to_scalar`, `scalar_to_ray` and
        `diffractive_surface`, which return 2-tuples. A runtime reads this to know
        whether to unpack.
        """
        return len(self.returns) > 1

    def matches(
        self,
        *,
        input: str | None,
        output: str | None,
        kind: OperationKind | None,
        entry: bool | None = None,
    ) -> bool:
        """Whether this descriptor satisfies a query. `None` means "do not filter".

        `input` matches if the named representation is on **any** of this
        operation's ports, since a multi-port operation is a candidate for an edge
        carrying either of them. `output` matches the **primary** result only:
        auxiliary diagnostics are not something a planner routes onward.

        `entry` is separate from `input` on purpose. `input=None` has always meant
        "do not filter", and once `inputs` could be empty that collided with
        "select the operations that need no input" -- a filter with two readings is
        worse than the fake input CHE-222 removed. So graph entry is its own
        tri-state: `None` does not filter, `True` selects entries, `False` selects
        everything that needs an upstream edge.
        """
        return (
            (input is None or input in self.inputs)
            and (output is None or self.primary_output == output)
            and (kind is None or self.kind == kind)
            and (entry is None or self.is_graph_entry is entry)
        )
