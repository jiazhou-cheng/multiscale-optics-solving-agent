"""One record per executable operation, with the four kinds as metadata.

Four *primitive* kinds. A record that fuses more than one declares the fusion in
`composes`; `SO_` is a composite id prefix and not a fifth kind.

CHE-177 (R03.1). The reference implementation described the same information
with 23 pydantic classes in `core/specs.py` plus 45 KB of YAML mirroring a
358-line Python table. This is one frozen dataclass and one `StrEnum`, and the
difference is not terseness for its own sake: every one of those 23 classes was
a field group some consumer wanted, and each new group made the next one look
cheap.

Why the kind is a field and not a hierarchy
-------------------------------------------
`source`, `coupler`, `physical_operator` and `measurement` differ in what they
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
that consumes no field, and `S_RAY_OPTILAND` (`SO_RAY_LAUNCH_TRACE` since CHE-225)
declared `input="ray_bundle"` for one that consumes no bundle. Both contradicted
the code, `sources/__init__.py` and `docs/architecture_principles.md` §2, which
all say a source is the one operation with no input.

The fields below are the minimum that answers ten questions and nothing else.
The specification *is* those ten questions -- a field that answers none of them
does not belong here, which is the discipline this record's own docstring already
demanded and had no list to check against:

1. is an upstream representation edge required?  -> `inputs`, `is_graph_entry`
2. what representation inputs are required?      -> `inputs`, in call order
3. what other required values must be supplied?  -> `requires`
4. which parameters are optional?                -> `optional`
5. what is the primary returned value?           -> `primary_output`
6. is auxiliary data also returned?              -> `returns_auxiliary`
7. what identifies an operation for planning and for a run record? ->
   `operation_id`. This question used to read "do two records over one callable
   stay distinct?", which no record needs any more; see below.
8. does any record claim an input its callable does not take? -> nothing here;
   `tests/operations/test_catalog_signatures.py` derives all four tuples from the
   resolved signature and compares, so the answer is checked against the code
   rather than against a table.
9. which backend executes this, without resolving the implementation? -> `backend`
10. does one callable fuse more than one primitive operation, and in what order?
    -> `composes`, `entry_stage`

Question 9 is CHE-224 (R15.1)'s, and adding it removed a field rather than
adding one on balance: the catalog lost a record. It is a *separate* question from
1-8, all of which are about physical state and arguments, and that separateness is
the point -- see `OperationKind` on why `solver` was never an answer to any of
them.

Question 10 is CHE-225 (R15.2)'s, and it exists because **one landed record's
`kind` was otherwise a false claim.** `backends.optiland.solver:trace` materializes
and declares its rays and then refracts them through every surface: it initializes
state *and* evolves it, so neither `source` nor `physical_operator` alone describes
it. CHE-224 declared it `source` and was wrong; `composes` lets the record say
`(SOURCE, PHYSICAL_OPERATOR)` and lets `kind` mean the terminal stage. See
`ENTRY_KINDS` for the retraction of the argument that produced the false claim, and
for the schema gap -- no notion of *which reference surface* a result sits at --
that made it possible.

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

One record per callable, once the two questions were separated
--------------------------------------------------------------
This section used to read "Semantic identity is not implementation identity" and
argued that two descriptors may name one callable and remain two distinct planning
choices: `S_WAVE_CHROMATIX` (`solver`) and `O_ASM_PROPAGATE` (`physical_operator`)
both resolved to `backends.chromatix.solver:propagate` with different
`approximation` and `validity`, because "one answers 'what backend does this
project drive', the other 'what happens to the physical state'".

**Both halves of that sentence were true, and together they were the diagnosis
rather than the justification.** Two questions were being asked of one field, so
the only way to answer both was two records over one function -- and a planner
reading the catalog saw two routes where the tree has one callable. CHE-224
(R15.1) gave the first question its own field, `backend`, and `S_WAVE_CHROMATIX`
was deleted: `O_ASM_PROPAGATE` carries `backend="chromatix"` and says everything
the pair said between them.

So the rule is now the plain one. **One record per `implementation`**, checked by
`tests/operations/test_catalog.py`. The id remains the planning identity and
nothing deduplicates the catalog by implementation string -- but nothing needs to,
because no two records share one. A future callable that genuinely needs two
records would be a modelling claim to argue on its own ticket, not a shape this
schema is holding open.

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

    **`SOLVER` became `SOURCE` on CHE-224 (R15.1)**, and the count did not change.
    `solver` was never answering the question this field asks. It described *who
    executes* -- an adapter over an external library -- while `coupler`,
    `physical_operator` and `measurement` all describe *what happens to physical
    state*, so one member of a four-member set was on a different axis from the
    other three. The consequence was mechanical rather than aesthetic: `source` is
    one of the seven terms §2 defines and had no member at all, so all three source
    records carried `SOLVER`, `S_` meant "solver" on one record and "source" on the
    next, and `propagate` needed two records because one field was being asked two
    questions. `backend` on `OperationDescriptor` now answers the execution
    question, and this enum answers only the state question.
    """

    SOURCE = "source"
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
#: Only `source`, which is what the module docstring above has always said §2 says:
#: a source is the one operation with no input. Until CHE-224 (R15.1) this set read
#: `frozenset({"solver"})` and **contradicted that sentence eight lines above it**.
#: The code was right about the records as they stood and wrong about the intent,
#: and it was wrong for one reason: there was no `SOURCE` member to put here, so the
#: three sources were declared `SOLVER` and the set had to name `solver` to admit
#: them.
#:
#: A coupler with no input would change the representation of nothing; a physical
#: operator with no input would change the state of nothing; a measurement with no
#: input would observe nothing. `__post_init__` refuses all three, which is the
#: check that makes `inputs=()` an honest declaration rather than a hole.
#:
#: **The membership test is the ENTRY STAGE, not `kind`** -- CHE-225 (R15.2). For
#: a record that fuses nothing, the two are the same thing. For a composite the
#: entry stage is `composes[0]`, so `SO_RAY_LAUNCH_TRACE` is admitted as an entry
#: because its *first* stage is a source, while its `kind` stays
#: `PHYSICAL_OPERATOR` because that is where it leaves the state.
#:
#: **CHE-224 declared `S_RAY_OPTILAND` a `SOURCE` here, and that was wrong.**
#: The argument it gave is retracted rather than reworded, because the reasoning
#: is the interesting part. It said: an `OpticalSetup` is a constructor *argument*
#: and not a port, so `S_RAY_OPTILAND` and `S_SOURCE_PLANE_WAVE` are
#: "indistinguishable in this schema" -- both `inputs=()`, both turning declared
#: arguments into physical state -- so collapsing them "loses no information the
#: catalog held".
#:
#: **That proves too much.** It reasons from *ports* to *kind*, and `kind` exists
#: precisely to state what the ports cannot. On the state axis the two records are
#: not remotely alike: `plane_wave` initializes a field at its declared reference
#: surface and stops, while `trace` initializes rays *and then refracts them
#: through every surface* to arrive at the image surface. The collapsed record's
#: own `approximation` said so -- "a surface interaction is refraction at a real
#: interface" -- so the catalog contained a `kind` contradicted by the record's own
#: prose.
#:
#: **What the schema actually cannot say, and this is the root cause.** There is no
#: notion here of *which reference surface* a returned representation sits at.
#: `inputs=()` plus `returns=("ray_bundle",)` is true of both a source and a fused
#: launch-and-trace, and nothing distinguishes "state at the surface I initialized
#: on" from "state at a surface N interfaces downstream". `composes` labels that
#: gap honestly; it does not close it. Closing it is a port-vocabulary change, and
#: the decomposition ticket needs it before `launch` + `O_RAY_TRACE` can replace
#: the fused record.
ENTRY_KINDS: frozenset[str] = frozenset({"source"})

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
        One of the four, as a **scalar**. Accepts the string form for convenience
        and normalizes.

        **The terminal stage** -- CHE-225 (R15.2) narrowed this from "what this
        operation is" to "where this operation leaves the state". For a record that
        fuses nothing the two readings coincide, which is twelve of thirteen. For a
        composite it is `composes[-1]`, and `__post_init__` enforces the agreement,
        so the scalar is well defined rather than a free choice between stages.
    `composes`
        The ordered primitive stages this one callable fuses, when it fuses more
        than one. `()` means "this record is exactly its `kind`" and is the default
        and the common case.

        Question 10, and it exists because one landed record needs it rather than
        because composition is an interesting shape. `SO_RAY_LAUNCH_TRACE` is
        `(SOURCE, PHYSICAL_OPERATOR)`: `backends.optiland.solver:trace`
        materializes and declares the rays and *then* refracts them through every
        surface, so it initializes state and evolves it in one call. Declaring only
        one of those is a false claim either way round, and CHE-224 made the
        `SOURCE` half of that mistake.

        **Ordered, and the order is physical.** Initialize-then-evolve is the
        operation; the reverse is meaningless. `composes[0]` is what decides
        whether the record may be a graph entry (see `ENTRY_KINDS`) and
        `composes[-1]` is what `kind` must equal.

        This is **not** the schema becoming a pipeline language. It records that a
        fusion happened and which primitives it fused -- not the arguments each
        stage took, not their intermediate representations, not a way to execute
        them separately. Nothing here can run a stage. When the launch/trace
        boundary is aligned numerically the fused record is replaced by two real
        records and this field goes back to being empty everywhere.

        `O_DIFFRACTIVE_SURFACE` is internally coupler -> operator -> coupler and
        deliberately keeps `composes=()`. Its input and output representation types
        do not change and it presents a single operator-like transformation at its
        boundary, so its net primitive kind is the whole truth about it at the
        ports. Whether it should nonetheless expose that structure is a recorded
        follow-up design question, not a defect.
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
    `backend`
        The third-party library the implementation drives -- `"optiland"`,
        `"chromatix"` -- or `None` for project-owned code that drives none.
        Question 9, and the field CHE-224 (R15.1) added to stop `kind` being asked
        two questions at once.

        **Declared, not derived.** It is checkable against `implementation`'s module
        path, and `tests/operations/test_catalog.py` checks it (gate G1), but it is
        not *read off* that path -- for the same reason
        `scripts/check_dependencies.py::LANDED` is declared rather than probed from
        the filesystem. That comment already makes the argument: a stray checkout or
        a package created ahead of the code that justifies it would read as
        "landed" to a probe, so declaring it means joining the graph is an edit
        someone reviews. Joining the set of driven backends is the same kind of
        edit. A path prefix is a parse; the library a module drives is a fact about
        the module.

        Orthogonal to `capabilities`, which is unaffected and keeps its meaning: it
        cites a *measured* device/dtype row by component id, which is a different
        question from which library runs. `SO_RAY_LAUNCH_TRACE` and `O_RAY_TRACE` both
        have `backend="optiland"` and both cite `M_RAY_OPTILAND`, and neither
        implies the other -- a backend-driving operation with no measured row of its
        own would carry a `backend` and `capabilities=None`.
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
    #: The driven third-party library, or `None` for project-owned arithmetic.
    #: Defaulted rather than required because `None` is the common case -- ten of
    #: the thirteen records -- and, unlike `inputs=()`, it is not a claim that can
    #: be made by omission: a record that drives a backend and forgets to say so
    #: fails G1 against its own `implementation` path.
    backend: str | None = None
    #: The ordered primitive stages one callable fuses, or `()` for "exactly its
    #: `kind`" -- twelve of the thirteen records. See the class docstring.
    composes: tuple[OperationKind, ...] = ()
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
        for name in (
            "inputs",
            "returns",
            "evidence",
            "requires",
            "optional",
            "validity",
            "composes",
        ):
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

        # G5 -- `composes` is well formed and `kind` is its terminal stage. Checked
        # before the entry rule below, which reads `composes[0]`. A one-stage
        # composition is refused because it is not a composition: `()` already says
        # "exactly its kind", so `(SOURCE,)` on a source is a second spelling of one
        # fact, which is the two-source arrangement this record exists to avoid.
        stages: list[OperationKind] = []
        for index, stage in enumerate(self.composes):
            try:
                stages.append(OperationKind(stage))
            except ValueError:
                problems.append(
                    f"`composes[{index}]` is {stage!r}; the stages of a composition are "
                    f"primitive kinds from {[k.value for k in OperationKind]}. A "
                    "composition fuses declared primitives; it cannot introduce one."
                )
        if len(stages) == len(self.composes):
            object.__setattr__(self, "composes", tuple(stages))
        if self.composes:
            if len(self.composes) < 2:
                problems.append(
                    f"`composes` is {[k.value for k in self.composes]}, a single stage. "
                    "`()` already means 'this record is exactly its kind'; a one-stage "
                    "composition states the same fact twice."
                )
            elif not problems and self.composes[-1] != self.kind:
                problems.append(
                    f"`composes` ends in {self.composes[-1].value!r} but `kind` is "
                    f"{getattr(self.kind, 'value', self.kind)!r}. `kind` is the TERMINAL "
                    "stage -- where the operation leaves the state -- so it must equal "
                    "the last stage it fuses."
                )

        # A graph entry, and the three kinds that cannot be one. `inputs=()` became
        # expressible in CHE-222, which is what makes this refusal necessary: a
        # coupler with no input would change the representation of nothing, an
        # operator the state of nothing, a measurement would observe nothing.
        #
        # Keyed on the ENTRY STAGE since CHE-225 (R15.2): `composes[0]` for a
        # composite, `kind` otherwise. A fused launch-and-trace consumes no upstream
        # representation because its *first* stage is a source, and refusing it here
        # for the `kind` of its *last* stage would be the mirror of CHE-224's error.
        entry_stage = self.composes[0] if self.composes else self.kind
        if not self.inputs and str(entry_stage) not in ENTRY_KINDS:
            described = (
                f"`kind` is {getattr(self.kind, 'value', self.kind)!r}"
                if not self.composes
                else f"the first stage of `composes` is {entry_stage.value!r}"
            )
            problems.append(
                f"`inputs` is empty but {described}. Only {sorted(ENTRY_KINDS)} may begin "
                "a graph entry: a source initializes a representation from source "
                "parameters alone, whether those parameters describe the light or a "
                "system to launch into. A coupler changes the representation of "
                "something, an operator changes the state of something, and a "
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
        if self.backend is not None and not self.backend.strip():
            problems.append(
                "`backend` is an empty string. `None` is how a record says it drives no "
                "third-party library; a blank name says it drives one and declines to "
                "say which."
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

        Question 1. `inputs == ()`, which `ENTRY_KINDS` restricts to a `source`
        entry *stage*, so this is exactly "does this operation begin with a source".

        Unchanged by CHE-225 (R15.2), and deliberately: a planner asks this to
        decide whether a node needs an upstream edge, and the answer for
        `SO_RAY_LAUNCH_TRACE` is no -- it consumes no representation, whatever it
        does internally afterwards. `find(entry=True)` therefore still returns it.
        """
        return not self.inputs

    @property
    def entry_stage(self) -> OperationKind:
        """The primitive kind this operation *begins* with. Question 10.

        `composes[0]` for a composite, `kind` otherwise -- and for the twelve
        records that fuse nothing those are the same value, which is why this is a
        property rather than a second field. `ENTRY_KINDS` is checked against this
        and not against `kind`, so a fused launch-and-trace is admitted as a graph
        entry on its source stage while `kind` keeps naming where it leaves the
        state.
        """
        return self.composes[0] if self.composes else self.kind

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
