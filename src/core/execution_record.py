"""What a run of a graph actually did, and no opinion about any of it.

CHE-132 (M0.5.3) needs a record shape before CHE-113 (M3.1) builds the thing that
produces one, and that is the right dependency direction: the verifier states
what evidence it requires, and the executor is written to supply it. It lives in
``core/`` rather than in either package because the executor produces it, the
verifier consumes it, and neither owns it.

A separate module from ``core/execution.py`` for a concrete reason as well as an
architectural one. The architectural one: ``execution.py`` is the *node-level*
vocabulary -- a run status and a cost estimate -- shared by models and couplers,
while this is the *graph-level* record of one whole run. The concrete one:
``core.provenance`` fingerprints every source file a probe loaded, so appending
to a module the ray-wave probes import marks eighteen committed records stale,
fourteen of them GPU records that cost hours to reproduce. A new contract goes in
a new file.

The separation this record exists to hold: **it records what happened. It does
not decide whether what happened was right.** There is deliberately no metric, no
tolerance and no verdict anywhere below. A field that said "passed" here would
move the scientific decision into the thing that ran the code, which is the
arrangement where a run grades itself.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.execution import CostEstimate, RunStatus

__all__ = [
    "DevicePrecisionObservation",
    "ExecutionRecord",
    "NodeOutcome",
    "NodeRecord",
    "Refusal",
    "RefusalKind",
    "ResourceCost",
]


class NodeOutcome(StrEnum):
    """What happened at one node. Distinct from a verdict about its physics."""

    EXECUTED = "executed"
    #: The component refused before executing -- capability, policy, or guard.
    REFUSED = "refused"
    #: It raised.
    RAISED = "raised"
    #: Never reached, because an upstream node did not produce its input.
    NOT_REACHED = "not_reached"
    #: Executed, and a declared precision or representation loss was taken.
    EXECUTED_LOSSY = "executed_lossy"


class RefusalKind(StrEnum):
    """Why something refused, structured so a caller can act on it.

    A generic failure is the thing this enum exists to prevent: "unsupported"
    and "out of validity" call for completely different responses, and a caller
    handed one string can distinguish neither.
    """

    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INVALID_CONFIGURATION = "invalid_configuration"
    OUT_OF_DECLARED_VALIDITY = "out_of_declared_validity"
    RESOURCE_GUARD = "resource_guard"
    UNVERIFIED_DERIVATIVE = "unverified_derivative"
    MISSING_EDGE_DECLARATION = "missing_edge_declaration"


class Refusal(BaseModel):
    """A structured refusal: a code, what was asked, and what would work."""

    model_config = ConfigDict(extra="forbid")

    kind: RefusalKind
    detail: str
    #: The declaration that made the refusal necessary -- a capability field, a
    #: validity key, a policy name. Never free text alone.
    declaration: str | None = None
    remedy: str | None = None


class DevicePrecisionObservation(BaseModel):
    """Requested versus actual, and what the difference cost.

    Requested is not evidence of actual. Chromatix casts to ``complex64``
    unconditionally, so a graph that asked for FP64 and reports FP64 because
    that is what it asked for has recorded a fiction. ``measured_loss`` is a
    number rather than a warning for the same reason.
    """

    model_config = ConfigDict(extra="forbid")

    requested_device: str
    actual_device: str
    requested_dtype: str
    actual_dtype: str
    requested_namespace: str | None = None
    actual_namespace: str | None = None
    #: Relative error introduced by the downcast, where it was measured. ``None``
    #: means *not measured*, which is different from zero and must stay so.
    measured_loss_relative: float | None = None
    measured_loss_basis: str | None = None

    @property
    def honoured(self) -> bool:
        return (
            self.requested_device == self.actual_device
            and self.requested_dtype == self.actual_dtype
        )


class ResourceCost(BaseModel):
    """What the run actually cost, beside what it was predicted to cost."""

    model_config = ConfigDict(extra="forbid")

    wall_seconds: float
    solver_seconds: float | None = None
    peak_memory_bytes: int | None = None
    device: str = "cpu"
    estimate: CostEstimate | None = None

    @property
    def estimate_ratio(self) -> float | None:
        """Actual over predicted wall time, or ``None`` with no prediction."""
        if self.estimate is None or not self.estimate.wall_time_s:
            return None
        return self.wall_seconds / self.estimate.wall_time_s


class NodeRecord(BaseModel):
    """One node's execution, including how it failed if it did."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    component: str
    outcome: NodeOutcome
    refusal: Refusal | None = None
    error_type: str | None = None
    error_message: str | None = None
    #: ``ContractCode`` values raised at this node's boundaries, as strings so
    #: the record does not import the boundary module.
    contract_codes: list[str] = Field(default_factory=list)
    device_precision: DevicePrecisionObservation | None = None
    cost: ResourceCost | None = None
    #: Keys into ``ExecutionRecord.artifacts`` this node produced.
    outputs: list[str] = Field(default_factory=list)


class ExecutionRecord(BaseModel):
    """Everything the verifier needs, and no opinion about any of it.

    ``artifacts`` holds the computed objects by key -- fields, bundles, PSFs --
    and is excluded from serialization because arrays do not belong in a JSON
    record; the record carries their keys and the run writes them beside it.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    status: RunStatus
    #: The instance this executed, so a record cannot be matched to the wrong one.
    instance_id: str | None = None
    instance_fingerprint: str | None = None
    graph_sha256: str | None = None
    nodes: list[NodeRecord] = Field(default_factory=list)
    #: The parameters as actually realized. May differ from what was requested
    #: -- a snapped grid, a clamped ray count -- and the difference is what lets
    #: the verifier re-evaluate validity against what ran rather than what was asked.
    observed_parameters: dict[str, Any] = Field(default_factory=dict)
    #: Seeds actually used, in order. A stochastic run with an empty list is a
    #: run whose ensemble cannot be enumerated or reproduced.
    seeds: list[int] = Field(default_factory=list)
    device_precision: DevicePrecisionObservation | None = None
    cost: ResourceCost | None = None
    refusal: Refusal | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    #: Structured, never free text: each entry is ``{"code": ..., "detail": ...}``.
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def contract_codes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for node in self.nodes:
            for code in node.contract_codes:
                if code not in seen:
                    seen.append(code)
        return tuple(seen)

    def node(self, node_id: str) -> NodeRecord | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)
