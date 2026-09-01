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
measured, and the measurement already lives in `numerics.COMPONENT_CAPABILITIES`
with the probe that produced it. Restating either here would recreate exactly the
two-source arrangement R03 exists to remove -- the old tree had a passing test
asserting its two sources agreed, which made the duplication feel safe rather
than removable. A descriptor names a row; the row stays where the evidence is.

Nothing here imports a backend. `implementation` is a string, and
`operations.resolve` is the only function in the package that turns one into
code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from numerics import COMPONENT_CAPABILITIES

__all__ = [
    "DERIVATIVE_MODES",
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
        Stable identity. What `resolve` takes and what a plan records.
    `kind`
        One of the four. Accepts the string form for convenience and normalizes.
    `input` / `output`
        Semantic types from `SEMANTIC_TYPES`. Singular, because R12's graph is
        bipartite `state -> operation -> state`; an operation that genuinely
        needs two inputs is a modelling question that ticket resolves with a
        real case in front of it.
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
        The name of a row in `numerics.COMPONENT_CAPABILITIES`, or `None`. Cited,
        never copied.
    `cost`
        Scaling information when it is known, `None` when it is not.
    `derivative` / `derivative_evidence`
        See `__post_init__`.
    """

    operation_id: str
    kind: OperationKind
    input: str
    output: str
    implementation: str
    approximation: str
    evidence: tuple[str, ...]
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
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "validity", tuple(self.validity))

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
        for name in ("input", "output"):
            value = getattr(self, name)
            if value not in SEMANTIC_TYPES:
                problems.append(
                    f"`{name}` is {value!r}, which is not a declared semantic type "
                    f"{list(SEMANTIC_TYPES)}. Add it in the ticket that lands the boundary "
                    "it names; an operation cannot introduce one by using it."
                )
        if self.input in OBSERVABLE_TYPES:
            problems.append(
                f"`input` is {self.input!r}, which is an observable and not a "
                "representation. Nothing consumes an observable: it is derived from "
                "physical state and the state it was derived from is still upstream."
            )
        if self.output in OBSERVABLE_TYPES and self.kind is not OperationKind.MEASUREMENT:
            problems.append(
                f"`output` is the observable {self.output!r} but `kind` is "
                f"{getattr(self.kind, 'value', self.kind)!r}. Only a measurement produces "
                "an observable. A coupler that produced one would be C_FIELD_TO_PSF, "
                "which changes no representation and consults no convention it does not "
                "already hold -- CHE-36 removed it, and R11 criterion 3 keeps it out."
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
        if self.capabilities is not None and self.capabilities not in COMPONENT_CAPABILITIES:
            problems.append(
                f"`capabilities` cites {self.capabilities!r}, which is not a row in "
                f"numerics.COMPONENT_CAPABILITIES {sorted(COMPONENT_CAPABILITIES)}. Device "
                "and dtype support is measured and lives with its probe; a descriptor "
                "cites a row, and citing one that does not exist claims a measurement "
                "nobody took."
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

    def matches(
        self, *, input: str | None, output: str | None, kind: OperationKind | None
    ) -> bool:
        """Whether this descriptor satisfies a query. `None` means "do not filter"."""
        return (
            (input is None or self.input == input)
            and (output is None or self.output == output)
            and (kind is None or self.kind == kind)
        )
