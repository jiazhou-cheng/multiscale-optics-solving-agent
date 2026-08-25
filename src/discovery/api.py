"""The queries, and the sources every answer is derived from.

CHE-114 (M3.2). Nothing here is hand-maintained. Every field names where it came
from, and ``tests/test_discovery.py`` asserts the answer still agrees with that
source -- one truth, several views.

The three questions nothing could answer before
------------------------------------------------
**"Can these two be connected, and under what declaration?"** The compatibility
rules live inside ``GraphValidator._validate_edge_contract`` and were reachable
only by constructing a candidate graph and validating it. :func:`check_connection`
does exactly that -- builds the candidate and validates it -- so there is one
implementation of the rules rather than two, and returns the required edge
declarations so an agent can supply them *without first being refused*.

**"At what device and precision can this route execute?"** A route's capability
is the intersection of its parts, and that intersection can be **empty**:
Chromatix accepts only ``complex64``, ``C_PATCH_WFT`` computes only in
``complex128`` on CPU. :func:`route_capability` returns the empty set with the
pair that emptied it named, before execution, rather than a traceback at node
three. That is project risk R5.

**"Should I use this here?"** Capability is not suitability. Chromatix *can* run
any scalar propagation and is not valid for every high-NA task; the k-space
route *can* run on demo3 and costs 1.7% of the power there while being exact on
demo2. :func:`validity_of` answers with an executable predicate's signed margin
-- ``INSIDE`` / ``NEAR_BOUNDARY`` / ``OUTSIDE`` / ``FAR_OUTSIDE`` -- rather than
with a paragraph, because prose cannot be checked before running.

On the prose
------------
``ValiditySpec``'s three lists of strings stay. A ``conventions.md`` explaining
*why* the phasor sign matters is worth more to a reader than a struct, and
``AGENTS.md`` deliberately keeps that detail out of code. What this adds is an
index over it. Where the structured predicate and the prose disagree, the
predicate wins and the prose is stale.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.capabilities import COMPONENT_CAPABILITIES, capabilities_for
from core.graph import GraphValidator, Severity
from core.paths import repository_root
from core.precision import CapabilityError, ComponentCapabilities
from core.specs import CouplerSpec, GraphSpec, ModelSpec
from registry.loader import Registry
from verification.claim_ledger import (
    KNOWLEDGE_PACK_REQUIRED_FILES,
    Claim,
    GateStatus,
    claims_for,
)
from verification.families import FAMILIES, BenchmarkCategory, ValidityState
from verification.refusals import REFUSAL_CATALOGUE

__all__ = [
    "ComponentDescription",
    "ConnectionReport",
    "FamilyCoverage",
    "Handover",
    "KnowledgeView",
    "PortView",
    "RefusalView",
    "RouteCapability",
    "SuitabilityRecord",
    "ValidityAnswer",
    "check_connection",
    "describe_component",
    "families_for_component",
    "knowledge_for",
    "route_capability",
    "validity_of",
]

def _root() -> Path | None:
    """The source checkout, or ``None`` when running from an installed wheel.

    Resolved lazily and tolerantly. ``repository_root()`` locates a *source
    tree* and says so in its own error message: code that must also work from an
    installed distribution should not call it at import time. The only thing
    this module needs it for is knowledge-pack lookup, which is a source-tree
    concern -- an installed distribution ships no ``knowledge/`` -- so the
    honest answer there is "no pack root", not an import failure.
    """
    try:
        return repository_root()
    except RuntimeError:
        return None


def _registry() -> Registry:
    return Registry.from_package()


def _spec(component: str) -> ModelSpec | CouplerSpec:
    registry = _registry()
    if component in registry.models:
        return registry.models[component]
    if component in registry.couplers:
        return registry.couplers[component]
    raise KeyError(
        f"no component {component!r}; registered models {sorted(registry.models)}, "
        f"couplers {sorted(registry.couplers)}"
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PortView(BaseModel):
    """A port, with the metadata it requires and provides.

    Derived from ``registry/*.yaml``'s ``PortSpec``. The metadata lists are the
    connection contract: a producer that provides nothing a consumer requires is
    an incompatible edge however well their artifact kinds match.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    artifact: str
    direction: str
    units: str | None = None
    requires_metadata: list[str] = Field(default_factory=list)
    provides_metadata: list[str] = Field(default_factory=list)
    optional: bool = False
    description: str = ""


class SuitabilityRecord(BaseModel):
    """One validity declaration, made machine-readable.

    ``condition`` is the prose from the registry. ``predicate_id`` links it to an
    executable :class:`~verification.families.schema.ValidityPredicate` where a
    family has one; ``None`` means the declaration is still prose only, which is
    a gap rather than a property and is reported as such.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    condition: str
    #: The executable counterpart, where one exists.
    predicate_id: str | None = None
    #: Which family declares that predicate, so a caller can find its evidence.
    declared_by_family: str | None = None
    structurable: bool = True
    note: str = ""


class RefusalView(BaseModel):
    """What happens if you get it wrong, from the M1.3 catalogue."""

    model_config = ConfigDict(extra="forbid")

    code: str
    status: str
    trigger: str
    remedy: str
    could_have_proceeded: bool


class FamilyCoverage(BaseModel):
    """Which benchmark families speak about a component, and what each can decide.

    ``gate_deciding`` is the field that stops an agent planning against a B4
    characterization as though it were a validation.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str
    category: str
    question: str
    gate_deciding: bool
    gate_status: str
    #: Present only where something has been measured against a threshold.
    observed: float | None = None
    metric: str | None = None


class KnowledgeView(BaseModel):
    """Which pack files exist for a component and which the policy permits.

    The V1 agent harness's ``cold`` / ``warm`` / ``guided`` policies copy these
    by hand into a workspace. Formalizing the lookup here means M7 can vary the
    policy without re-implementing file copying.
    """

    model_config = ConfigDict(extra="forbid")

    component: str
    pack_root: str | None
    present: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    permitted_by_policy: list[str] = Field(default_factory=list)
    policy: str = "cold"


class ComponentDescription(BaseModel):
    """Everything one query can say about one component.

    "Can", "should", and "what happens if I do it wrong", answerable together.
    """

    model_config = ConfigDict(extra="forbid")

    component: str
    version: str
    kind: str
    description: str
    approximation: str
    framework: str
    maturity: str

    inputs: list[PortView] = Field(default_factory=list)
    outputs: list[PortView] = Field(default_factory=list)

    devices: list[str] = Field(default_factory=list)
    precisions: list[str] = Field(default_factory=list)
    native_compute_dtypes: list[str] = Field(default_factory=list)
    lossy_input_dtypes: list[str] = Field(default_factory=list)
    namespaces: list[str] = Field(default_factory=list)
    device_namespaces: dict[str, list[str]] = Field(default_factory=dict)
    capability_evidence: str = ""

    lossy: bool = False
    invariants: list[str] = Field(default_factory=list)
    cost_model: dict[str, Any] = Field(default_factory=dict)

    derivative_mode: str = "none"
    #: R4. Surfaced at the top level and never nested, because a consumer that
    #: has to dig for it will read a gradient claim into a component with none.
    derivative_verified: bool = False
    derivative_warning: str | None = None

    suitability: list[SuitabilityRecord] = Field(default_factory=list)
    refusals: list[RefusalView] = Field(default_factory=list)
    validation_claims: list[dict[str, Any]] = Field(default_factory=list)
    families: list[FamilyCoverage] = Field(default_factory=list)
    knowledge: KnowledgeView | None = None


class ConnectionReport(BaseModel):
    """Whether two ports can be connected, and what the edge must declare."""

    model_config = ConfigDict(extra="forbid")

    source_component: str
    source_port: str
    target_component: str
    target_port: str
    compatible: bool
    #: The coupler that mediates it, if one does.
    coupler: str | None = None
    #: What the edge must declare for the coupler to accept it. Currently
    #: discoverable only by being refused; returned here so an agent can supply
    #: them first.
    required_edge_declarations: list[str] = Field(default_factory=list)
    #: Structured reasons, from GraphValidator. Never free text alone.
    issues: list[dict[str, str]] = Field(default_factory=list)
    #: Refusals this edge is known to be able to produce.
    possible_refusals: list[RefusalView] = Field(default_factory=list)


class Handover(BaseModel):
    """One adjacent pair, and whether the artifact can cross between them."""

    model_config = ConfigDict(extra="forbid")

    producer: str
    consumer: str
    #: Dtypes the producer emits that the consumer accepts natively.
    exact_dtypes: list[str] = Field(default_factory=list)
    #: Dtypes it emits that the consumer accepts LOSSILY. Chromatix's
    #: complex128 entry is the live case: it ingests one and truncates it, and
    #: keeping that out of accepted_input_dtypes is what makes the bridge report
    #: the loss instead of letting it happen inside ScalarField.
    lossy_dtypes: list[str] = Field(default_factory=list)
    possible: bool = True
    lossy: bool = False


class RouteCapability(BaseModel):
    """What an ordered route can execute, at two different levels.

    The two are genuinely different questions and conflating them produces a
    false negative on the repository's own flagship route:

    ``feasible``
        can the artifact cross every adjacent pair? A ray bundle in float64
        handing over to a wave model in complex64 is a *representation change*,
        so the dtypes differ by construction and the route runs anyway.
    ``uniform_precision_available``
        is there one precision at which the WHOLE route computes? Often not,
        and that is the R5 statement: Chromatix computes only in complex64,
        C_PATCH_WFT only in complex128, so a route through both has no single
        precision -- reported here, before execution, with the pair named.

    An earlier draft intersected native compute dtypes and called the result
    feasibility, which declared ``M_RAY_OPTILAND -> C_RAY_TO_WAVE ->
    M_WAVE_CHROMATIX`` infeasible. That route executes; it is
    ``examples/graphs/ray_to_wave.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    route: list[str]
    feasible: bool
    devices: list[str] = Field(default_factory=list)
    handovers: list[Handover] = Field(default_factory=list)
    lossy_handovers: list[str] = Field(default_factory=list)
    #: The intersection of every node's native compute dtypes. EMPTY is a
    #: legitimate and important answer.
    uniform_compute_dtypes: list[str] = Field(default_factory=list)
    uniform_precision_available: bool = True
    #: ``[producer, consumer, what emptied]`` for the first pair that did.
    blocking_pair: list[str] | None = None
    reason: str = ""


class ValidityAnswer(BaseModel):
    """Where a parameter point sits relative to a component's declared bounds."""

    model_config = ConfigDict(extra="forbid")

    component: str
    state: str
    margins: dict[str, float] = Field(default_factory=dict)
    #: Per predicate: the statement, the basis, and what it is blind to.
    predicates: list[dict[str, Any]] = Field(default_factory=list)
    #: Declarations with no executable counterpart yet. A gap, reported.
    prose_only: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

#: Which structured predicate answers which prose declaration. The ONLY mapping
#: in this module, and it is a mapping between two existing sources rather than
#: a third copy of either: the prose stays in the registry and the predicate
#: stays on the family.
_PROSE_TO_PREDICATE = {
    "band-limit": "ASM_TF_SAMPLING",
    "band limit": "ASM_TF_SAMPLING",
    "sampling": "ASM_TF_SAMPLING",
    "nyquist": "PER_AXIS_NYQUIST",
    "curvature": "SI_S3_CURVATURE",
    "planar": "DECLARED_PLANARITY",
    "hexapolar": "HEXAPOLAR_RING",
    "paraxial": "PARAXIAL_FIELD_ANGLE",
    "evanescent": "PROPAGATING_BAND",
}


def _predicate_index() -> dict[str, str]:
    """predicate_id -> the family that declares it."""
    return {
        predicate.predicate_id: family.family_id
        for family in FAMILIES
        for predicate in family.validity
    }


def _suitability(spec: ModelSpec | CouplerSpec) -> list[SuitabilityRecord]:
    index = _predicate_index()
    records: list[SuitabilityRecord] = []
    declarations = (
        [("assumption", text) for text in spec.validity.assumptions]
        + [("warning", text) for text in spec.validity.warnings]
        + [("hard_limit", f"{key}: {value}") for key, value in spec.validity.hard_limits.items()]
    )
    for kind, condition in declarations:
        lowered = condition.lower()
        predicate = next(
            (pid for token, pid in _PROSE_TO_PREDICATE.items() if token in lowered), None
        )
        records.append(
            SuitabilityRecord(
                kind=kind,
                condition=condition,
                predicate_id=predicate if predicate in index else None,
                declared_by_family=index.get(predicate) if predicate else None,
                structurable=predicate is not None and predicate in index,
                note=(
                    ""
                    if predicate in index
                    else (
                        "prose only: no executable ValidityPredicate covers this yet. "
                        "A caller cannot check it before running."
                    )
                ),
            )
        )
    return records


def _capabilities(component: str) -> ComponentCapabilities | None:
    try:
        return capabilities_for(component)
    except CapabilityError:
        return None


def _families(component: str) -> list[FamilyCoverage]:
    coverage: list[FamilyCoverage] = []
    for family in FAMILIES:
        if component not in family.components:
            continue
        disposition = family.gate_disposition
        coverage.append(
            FamilyCoverage(
                family_id=family.family_id,
                category=str(family.category),
                question=family.question,
                # A B4 family cannot gate by construction; a family with no
                # gating tolerance cannot either. Both are derived, not declared.
                gate_deciding=family.is_gate_deciding,
                gate_status=(
                    disposition.status.value
                    if disposition is not None
                    else GateStatus.CHARACTERIZED_NO_GATE.value
                ),
                observed=disposition.observed if disposition is not None else None,
                metric=disposition.metric if disposition is not None else None,
            )
        )
    return coverage


def _claim_view(claim: Claim) -> dict[str, Any]:
    return {
        "kind": claim.kind.value,
        "claim": claim.claim,
        "oracle": claim.oracle.value,
        "oracle_independence": claim.oracle_independence.value,
        "gate_status": claim.gate_status.value,
        "gate_deciding": claim.gate_deciding,
        "metric": claim.metric,
        "tolerance": claim.tolerance,
        "observed": claim.observed,
        "caveats": list(claim.caveats),
    }


def _refusals_for(component: str) -> list[RefusalView]:
    """Which catalogued refusals this component can produce.

    Derived from what the component's own declarations make possible rather than
    from a per-component list: a coupler with a required handoff declaration can
    produce ``OPL_REFERENCE_UNVERIFIED``, and one without cannot.
    """
    spec = _spec(component)
    required: set[str] = set()
    ports = (
        [*spec.inputs, *spec.outputs]
        if isinstance(spec, ModelSpec)
        else [spec.source, spec.target]
    )
    for port in ports:
        required.update(port.requires_metadata)

    codes: set[str] = {"MISSING_DECLARATION", "ARTIFACT_KIND_MISMATCH", "NON_FINITE"}
    if "reference_plane" in required:
        codes |= {"OPL_REFERENCE_UNVERIFIED", "REFERENCE_PLANE_MISMATCH"}
    if "direction" in required:
        codes |= {"NON_UNIT_DIRECTION", "OBJECT_SPACE_REFERENCE_MISSING"}
    if "sample_pitch" in required:
        codes |= {"SAMPLE_PITCH_MISMATCH", "PAD_STATE_UNKNOWN"}
    if "phasor" in required:
        codes.add("PHASOR_MISMATCH")
    if "coordinate_frame" in required:
        codes |= {"FRAME_MISMATCH", "AXIS_ORDER_MISMATCH"}
    if isinstance(spec, CouplerSpec) and "hexapolar" in " ".join(
        spec.validity.assumptions + spec.validity.warnings
    ).lower():
        codes.add("NON_HEXAPOLAR_SAMPLING")

    return [
        RefusalView(
            code=entry.code,
            status=entry.status.value,
            trigger=entry.trigger,
            remedy=entry.remedy,
            could_have_proceeded=entry.could_have_proceeded,
        )
        for code, entry in sorted(REFUSAL_CATALOGUE.items())
        if code in codes
    ]


def _port_view(port: Any, direction: str) -> PortView:
    return PortView(
        name=port.name,
        artifact=port.artifact.value,
        direction=direction,
        units=port.units,
        requires_metadata=list(port.requires_metadata),
        provides_metadata=list(port.provides_metadata),
        optional=port.optional,
        description=port.description,
    )


def describe_component(component: str, *, policy: str = "cold") -> ComponentDescription:
    """Everything one query can say about one component, all of it derived."""
    spec = _spec(component)
    caps = _capabilities(component)
    is_model = isinstance(spec, ModelSpec)

    if isinstance(spec, ModelSpec):
        inputs = [_port_view(p, "input") for p in spec.inputs]
        outputs = [_port_view(p, "output") for p in spec.outputs]
    else:
        # A coupler has exactly one source and one target: the representation
        # change IS the component, so there is nothing to fan in or out.
        inputs = [_port_view(spec.source, "input")]
        outputs = [_port_view(spec.target, "output")]

    derivative_warning = None
    if not spec.derivative.verified:
        derivative_warning = (
            f"derivative.verified is FALSE for {component}. Whatever "
            f"derivative.mode says ({spec.derivative.mode.value}), no gradient "
            "through this component has been validated against a finite "
            "difference, and none may be claimed across it."
        )

    return ComponentDescription(
        component=spec.id,
        version=spec.version,
        kind="model" if is_model else "coupler",
        description=spec.description,
        approximation=spec.approximation.value,
        framework=spec.framework.value,
        maturity=spec.maturity.value,
        inputs=inputs,
        outputs=outputs,
        devices=sorted(str(d) for d in caps.devices) if caps else [],
        precisions=sorted(str(p) for p in caps.precisions) if caps else [],
        native_compute_dtypes=(
            sorted(str(d) for d in caps.native_compute_dtypes) if caps else []
        ),
        lossy_input_dtypes=(
            sorted(str(d) for d in caps.lossy_input_dtypes) if caps else []
        ),
        namespaces=sorted(str(n) for n in caps.namespaces) if caps else [],
        device_namespaces=(
            {
                str(device): sorted(str(n) for n in namespaces)
                for device, namespaces in sorted(
                    caps.device_namespaces.items(), key=lambda kv: str(kv[0])
                )
            }
            if caps
            else {}
        ),
        capability_evidence=caps.evidence if caps else "",
        lossy=getattr(spec, "lossy", False),
        invariants=list(getattr(spec, "invariants", [])),
        cost_model=spec.cost_model.model_dump(),
        derivative_mode=spec.derivative.mode.value,
        derivative_verified=spec.derivative.verified,
        derivative_warning=derivative_warning,
        suitability=_suitability(spec),
        refusals=_refusals_for(component),
        validation_claims=[_claim_view(c) for c in claims_for(component)],
        families=_families(component),
        knowledge=knowledge_for(component, policy=policy),
    )


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------


def _mediating_coupler(source_artifact: str, target_artifact: str) -> CouplerSpec | None:
    for coupler in _registry().couplers.values():
        if (
            coupler.source.artifact.value == source_artifact
            and coupler.target.artifact.value == target_artifact
        ):
            return coupler
    return None


def check_connection(
    source_component: str,
    source_port: str,
    target_component: str,
    target_port: str,
) -> ConnectionReport:
    """Can these be connected, and what must the edge declare?

    Implemented by building the candidate edge and handing it to
    ``GraphValidator``. That is deliberately the same code path a real graph
    takes: there is exactly one implementation of the compatibility rules, and
    an answer here that disagreed with validation would be worse than no answer.
    """
    registry = _registry()
    source_spec = _spec(source_component)
    target_spec = _spec(target_component)

    producer = (
        source_spec.output_port(source_port)
        if isinstance(source_spec, ModelSpec)
        else source_spec.target
    )
    consumer = (
        target_spec.input_port(target_port)
        if isinstance(target_spec, ModelSpec)
        else target_spec.source
    )
    if producer is None or consumer is None:
        return ConnectionReport(
            source_component=source_component,
            source_port=source_port,
            target_component=target_component,
            target_port=target_port,
            compatible=False,
            issues=[
                {
                    "code": "UNKNOWN_PORT",
                    "message": (
                        f"{source_component}.{source_port} -> "
                        f"{target_component}.{target_port}: one of these ports is not "
                        "declared"
                    ),
                }
            ],
        )

    coupler = _mediating_coupler(producer.artifact.value, consumer.artifact.value)
    if coupler is None:
        return ConnectionReport(
            source_component=source_component,
            source_port=source_port,
            target_component=target_component,
            target_port=target_port,
            compatible=False,
            issues=[
                {
                    "code": "NO_MEDIATING_COUPLER",
                    "message": (
                        f"no registered coupler transforms {producer.artifact.value} "
                        f"into {consumer.artifact.value}"
                    ),
                }
            ],
        )

    candidate = GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "producer", "model": source_component},
                {"id": "consumer", "model": target_component},
            ],
            "edges": [
                {
                    "id": "candidate",
                    "coupler": coupler.id,
                    "source": {"node": "producer", "port": source_port},
                    "target": {"node": "consumer", "port": target_port},
                }
            ],
        }
    )
    report = GraphValidator(registry).validate(candidate)
    errors = [i for i in report.issues if i.severity is Severity.ERROR]

    # What the edge must declare, taken from the coupler's own required
    # metadata plus the target's. Currently discoverable only by being refused.
    declarations = sorted(
        set(coupler.source.requires_metadata) | set(consumer.requires_metadata)
    )

    return ConnectionReport(
        source_component=source_component,
        source_port=source_port,
        target_component=target_component,
        target_port=target_port,
        compatible=not errors,
        coupler=coupler.id,
        required_edge_declarations=declarations,
        issues=[
            {"code": i.code, "message": i.message, "location": i.location or ""}
            for i in report.issues
        ],
        possible_refusals=_refusals_for(coupler.id),
    )


# ---------------------------------------------------------------------------
# route capability
# ---------------------------------------------------------------------------


def route_capability(route: Sequence[str]) -> RouteCapability:
    """Whether an ordered route can execute, and at what precision.

    Two answers, because they are two questions. See :class:`RouteCapability`.
    """
    ids = list(route)
    if not ids:
        return RouteCapability(
            route=[], feasible=False, reason="an empty route has no capability"
        )

    unknown = [c for c in ids if c not in COMPONENT_CAPABILITIES]
    if unknown:
        return RouteCapability(
            route=ids, feasible=False, reason=f"no capability declaration for {unknown}"
        )

    caps = {c: capabilities_for(c) for c in ids}

    devices = set(caps[ids[0]].devices)
    for component in ids[1:]:
        devices &= caps[component].devices
    if not devices:
        blocker = next(
            (
                b
                for a, b in pairwise(ids)
                if not (caps[a].devices & caps[b].devices)
            ),
            ids[-1],
        )
        return RouteCapability(
            route=ids,
            feasible=False,
            blocking_pair=[ids[ids.index(blocker) - 1], blocker, "device"],
            reason="no device runs every node of this route",
        )

    handovers: list[Handover] = []
    lossy: list[str] = []
    for producer, consumer in pairwise(ids):
        emitted = caps[producer].output_dtypes
        exact = emitted & caps[consumer].accepted_input_dtypes
        degraded = emitted & caps[consumer].lossy_input_dtypes
        handover = Handover(
            producer=producer,
            consumer=consumer,
            exact_dtypes=sorted(str(d) for d in exact),
            lossy_dtypes=sorted(str(d) for d in degraded),
            possible=bool(exact or degraded),
            lossy=not exact and bool(degraded),
        )
        handovers.append(handover)
        if handover.lossy:
            lossy.append(f"{producer} -> {consumer}")
        if not handover.possible:
            return RouteCapability(
                route=ids,
                feasible=False,
                devices=sorted(str(d) for d in devices),
                handovers=handovers,
                blocking_pair=[producer, consumer, "dtype"],
                reason=(
                    f"{producer} emits "
                    f"{sorted(str(d) for d in emitted)} and {consumer} accepts "
                    f"{sorted(str(d) for d in caps[consumer].accepted_input_dtypes)} "
                    "natively or lossily -- the artifact cannot cross this edge at all"
                ),
            )

    uniform = set(caps[ids[0]].native_compute_dtypes)
    uniform_blocker: list[str] | None = None
    for producer, consumer in pairwise(ids):
        before = set(uniform)
        uniform &= caps[consumer].native_compute_dtypes
        if before and not uniform and uniform_blocker is None:
            uniform_blocker = [producer, consumer, "native_compute_dtype"]

    reason = ""
    if not uniform:
        reason = (
            "no single precision computes the whole route. That is not a failure: a "
            "representation change legitimately computes in different dtypes on either "
            "side, and the handovers above say whether the artifact can cross. It IS "
            "the answer to 'at what precision does this route run', and the answer is "
            "'more than one'."
        )

    return RouteCapability(
        route=ids,
        feasible=True,
        devices=sorted(str(d) for d in devices),
        handovers=handovers,
        lossy_handovers=lossy,
        uniform_compute_dtypes=sorted(str(d) for d in uniform),
        uniform_precision_available=bool(uniform),
        blocking_pair=uniform_blocker,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


def validity_of(component: str, parameters: dict[str, Any]) -> ValidityAnswer:
    """Where a parameter point sits relative to the executable bounds.

    The amendment's requirement: an answer that is a state and a signed margin
    rather than a paragraph. Aggregated over every predicate any family declares
    about this component, because a bound is a property of the physics and not
    of whichever family happened to write it down.
    """
    states: list[ValidityState] = []
    margins: dict[str, float] = {}
    predicates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for family in FAMILIES:
        if component not in family.components:
            continue
        for predicate in family.validity:
            if predicate.predicate_id in seen:
                continue
            try:
                margin, state = predicate.evaluate(parameters)
            except (KeyError, ValueError, TypeError):
                # The parameter point does not carry what this predicate reads.
                # Not an error: a caller asking about a device configuration
                # should not have to supply a patch width.
                continue
            seen.add(predicate.predicate_id)
            margins[predicate.predicate_id] = margin
            states.append(state)
            predicates.append(
                {
                    "predicate_id": predicate.predicate_id,
                    "statement": predicate.statement,
                    "basis": str(predicate.basis),
                    "margin": margin,
                    "state": state.value,
                    "blind_to": list(predicate.blind_to),
                    "declared_by_family": family.family_id,
                }
            )

    from verification.families.schema import aggregate_validity

    prose_only = [
        record.condition
        for record in _suitability(_spec(component))
        if not record.structurable
    ]
    return ValidityAnswer(
        component=component,
        state=aggregate_validity(states).value,
        margins=margins,
        predicates=predicates,
        prose_only=prose_only,
    )


# ---------------------------------------------------------------------------
# knowledge and families
# ---------------------------------------------------------------------------

#: What each policy permits. Mirrors the agent harness's three declared
#: policies; formalized here so M7 can vary the policy without re-implementing
#: file copying.
_POLICY_FILES = {
    "cold": (),
    "warm": ("card.yaml", "api_minimal_examples.md"),
    "guided": ("card.yaml", "api_minimal_examples.md", "conventions.md"),
}


def _pack_root(component: str) -> tuple[str | None, str]:
    """``(relative path, kind)`` for a component's knowledge pack."""
    root = _root()
    kind_default = "solver" if component.startswith("M_") else "coupler"
    if root is None:
        return None, kind_default
    for kind, folder in (("solver", "solvers"), ("coupler", "couplers")):
        for name in _pack_names(component):
            candidate = root / "knowledge" / folder / name
            if candidate.is_dir():
                return str(candidate.relative_to(root)), kind
    return None, kind_default


def _pack_names(component: str) -> tuple[str, ...]:
    """The directory names a component's pack might use.

    ``M_RAY_OPTILAND`` -> ``optiland``; ``C_RAY_TO_WAVE`` -> ``ray_to_wave``.
    Derived from the id rather than tabulated, so a new component is covered
    without editing this module.
    """
    stem = component.split("_", 1)[1].lower()
    return (stem, stem.replace("_", ""), stem.split("_")[-1])


def knowledge_for(component: str, *, policy: str = "cold") -> KnowledgeView:
    if policy not in _POLICY_FILES:
        raise KeyError(f"unknown context policy {policy!r}; have {sorted(_POLICY_FILES)}")
    checkout = _root()
    root, kind = _pack_root(component)
    required = KNOWLEDGE_PACK_REQUIRED_FILES[kind]
    present: list[str] = []
    missing: list[str] = []
    for name in required:
        if root is not None and checkout is not None and (checkout / root / name).is_file():
            present.append(name)
        else:
            missing.append(name)
    return KnowledgeView(
        component=component,
        pack_root=root,
        present=present,
        missing=missing,
        permitted_by_policy=[n for n in _POLICY_FILES[policy] if n in present],
        policy=policy,
    )


def families_for_component(component: str) -> list[FamilyCoverage]:
    """Which families speak about this component, and which of them can decide.

    The query that stops an agent planning against a B4 characterization as
    though it were a validation.
    """
    return _families(component)


def _gate_deciding_categories() -> frozenset[BenchmarkCategory]:
    return frozenset(c for c in BenchmarkCategory if c.may_gate)
