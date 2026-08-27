"""Run a validated ``GraphSpec`` and record what happened.

CHE-113 (M3.1). There was no graph executor. ``GraphSpec`` is a complete strict
description of a workflow and ``GraphValidator`` checks it thoroughly, and
nothing executed any of it: seventeen hand-written entry points each
re-implemented node sequencing, artifact plumbing, timing, memory guarding and
record writing, and they had already drifted apart.

The consequence that reorders the project is the second one. "The agent
constructs a workflow" is meaningless if constructing a workflow means writing a
Python script -- the agent would be generating code rather than composing
verified components, which is the failure mode this project exists to avoid.

**This module records facts. It does not decide correctness.** There is no
metric, no tolerance and no verdict anywhere below, and
``verification/verifier.py`` is what turns an :class:`ExecutionRecord` into a
scientific statement. Keeping that boundary is what lets one record be judged
against a family's gate today and against different criteria later without
re-running anything.

The process model, decided
--------------------------
Optiland's ``set_backend`` / ``set_precision`` / ``set_device`` mutate
process-wide state and are documented as not thread-safe; JAX's platform pin and
``jax_enable_x64`` are process-global; and the ordering hazard is real. A
single-process executor running a ray node and then a wave node inherits all of
it.

**Chosen: a single process with a strict state-transition protocol**, recorded
per node in the run record. The protocol is not new -- CHE-61 already made the
Optiland adapter set all three globals explicitly on *every* run, at the
defaults included, precisely so a previous run's choices cannot leak. What this
executor adds is that the applied state is captured per node and carried in the
record, so a fingerprint comparison can tell "the same graph under different
solver state" from "the same graph twice".

**Rejected for now: process-per-node**, which ``studies/metalens/controller.py``
uses successfully and for good reasons -- CUDA memory returned at exit, no
allocator fragmentation across differently-shaped runs, a dying candidate cannot
poison the next. It is rejected because it needs artifact *serialization* across
the boundary, and ``ArtifactRecord`` deliberately keeps arrays in solver-owned
storage: the record is a reference, and the live object travels in
:attr:`ExecutionRecord.artifacts`, which does not cross a process boundary. That
is a real piece of work and a named blocker, not an oversight.
:class:`ProcessModel` declares it, and asking for it raises rather than silently
running in-process -- an executor that quietly ignored the request would make
every record's process-model field a lie.

JAX x64 is refused rather than flipped. ``jax_enable_x64`` is pinned ``False``
everywhere in this repository and a process that flipped it would change every
recorded number; a graph asking for it gets a structured refusal.

Streaming is out of scope for the graph, explicitly
---------------------------------------------------
demo3 runs 60M rays in chunks and cannot be held in memory.
``couplers/streaming.py`` and ``core/coherent_batch.py`` implement chunked
coherent accumulation, and neither is reachable through a ``GraphSpec``: there is
no node contract for "consumes a stream". Rather than let a graph run such a
workload un-chunked and out of memory, a node declaring ``streaming: true`` is
refused with :attr:`RefusalKind.UNSUPPORTED_CAPABILITY`.

The consequence, stated: **an agent cannot run demo3 through the executor.**
demo3 stays a probe until a streaming node contract exists. That is a gap in
what the agent can compose, and it is better as a refusal than as a run that
dies at 40 GB.

A graph needs a node to hang an edge on, and two workloads have none
--------------------------------------------------------------------
Found by CHE-115 (M3.3) trying to migrate demo2 and recorded here rather than
worked around there, because it is this layer's gap and not that benchmark's.

Every edge in a ``GraphSpec`` names a source **node** and a target **node**, and
a node names a registered model. The registry has exactly two:
``M_RAY_OPTILAND`` and ``M_WAVE_CHROMATIX``. demo2 is a bare SLM behind a
circular amplitude mask with a sensor 1.26 mm downstream and **no refractive
surface at all** -- so its own record says ``optiland_used: false`` -- and the
operation it exercises is ``C_PLANAR_DOE_STEP``, which consumes an incident ray
bundle *and* a DOE transmission and emits a ray bundle. There is no registered
model that emits either input and none that consumes the output, so demo2 cannot
be written as a graph document no matter what the couplers support. Its RW-P
route is additionally 1.6e8 rays in 40 chunks, which the streaming refusal above
covers independently.

So the missing piece is a **source/sink node contract**: a way for a graph to
declare "this array, on this plane, with these conventions, is an input" and
"this field is the terminal state". ``runtime/instance_runner.py::field_source``
is the shape of half of it and is deliberately *not* a registered model --
``GraphExecutor.run`` takes it through ``inputs``, which works for a wave node
whose port is ``input_field`` and does nothing for a coupler that needs a source
node to exist. Naming the two workloads that are blocked, rather than inventing
two models to unblock one benchmark, is the choice here: a registered model is a
capability claim, and ``M_SOURCE_ARRAY`` would be claiming one that no oracle has
ever checked.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from core.artifacts import ArtifactRecord
from core.execution import CostEstimate, RunStatus
from core.execution_record import (
    DevicePrecisionObservation,
    ExecutionRecord,
    NodeOutcome,
    NodeRecord,
    Refusal,
    RefusalKind,
    ResourceCost,
)
from core.graph import GraphValidator, ValidationReport
from core.resources import MemoryWatchdog, host_memory_snapshot
from core.specs import EdgeSpec, GraphSpec, NodeSpec

__all__ = [
    "EXECUTOR_VERSION",
    "ExecutionCache",
    "ExecutorError",
    "GraphExecutor",
    "InMemoryCache",
    "ProcessModel",
    "SolverStateProtocol",
    "graph_fingerprint",
    "topological_order",
]

#: Bumped when the executor changes what a record means. Part of provenance, so
#: two records can be told apart by how they were produced.
EXECUTOR_VERSION = "1.0.0"


class ExecutorError(RuntimeError):
    """The executor refused before running anything.

    Distinct from a node failure: nothing was executed, so there is no partial
    result to preserve and no record to interpret.
    """


class ProcessModel(StrEnum):
    """How solver state is isolated between nodes. Recorded in every run."""

    #: One process, with every node re-establishing the global solver state it
    #: needs before it runs. Implemented.
    IN_PROCESS = "in_process"
    #: One subprocess per node. Declared and NOT implemented: it needs artifact
    #: serialization across the boundary, which ``ArtifactRecord`` does not do.
    PROCESS_PER_NODE = "process_per_node"


# ---------------------------------------------------------------------------
# Solver state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverStateProtocol:
    """What global solver state a node ran under, as observed rather than asked.

    A node's *request* names a backend, a device and a precision; what the
    adapter actually applied is a separate fact, and CHE-61's whole point is
    that the two can differ. The executor reads the applied state back out of
    the adapter's diagnostics where the adapter reports it, and records
    ``None`` where it does not -- which is itself information, because a node
    whose applied state is unknown cannot be cache-matched.
    """

    node_id: str
    requested: Mapping[str, Any]
    applied: Mapping[str, Any] | None = None

    @property
    def known(self) -> bool:
        return self.applied is not None

    def key(self) -> str:
        """A stable digest of the state, for cache matching."""
        payload = {"requested": _canonical(self.requested), "applied": _canonical(self.applied)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple | set | frozenset):
        return sorted((_canonical(v) for v in value), key=repr)
    return repr(value)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ExecutionCache(Protocol):
    """A content-addressed node cache.

    The validity condition is the whole design: a cache that returns a result
    computed under different precision is worse than no cache, because it turns
    a precision question into an invisible one. The key therefore includes the
    environment fingerprint and the solver-state digest, and a miss on either is
    a miss rather than a stale hit.
    """

    def get(self, key: str) -> tuple[dict[str, ArtifactRecord], dict[str, Any]] | None: ...

    def put(
        self, key: str, outputs: Mapping[str, ArtifactRecord], diagnostics: Mapping[str, Any]
    ) -> None: ...


@dataclass
class InMemoryCache:
    """The default. Process-scoped, so it cannot outlive the state it was keyed on."""

    entries: dict[str, tuple[dict[str, ArtifactRecord], dict[str, Any]]] = field(
        default_factory=dict
    )
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> tuple[dict[str, ArtifactRecord], dict[str, Any]] | None:
        found = self.entries.get(key)
        if found is None:
            self.misses += 1
            return None
        self.hits += 1
        return found

    def put(
        self, key: str, outputs: Mapping[str, ArtifactRecord], diagnostics: Mapping[str, Any]
    ) -> None:
        self.entries[key] = (dict(outputs), dict(diagnostics))


# ---------------------------------------------------------------------------
# Graph ordering
# ---------------------------------------------------------------------------


def topological_order(spec: GraphSpec) -> tuple[str, ...]:
    """Node ids in dependency order.

    Kahn's algorithm over the edge set. A cycle raises rather than picking an
    arbitrary order: ``GraphSpec.allow_cycles`` exists for a future iterative
    solver, and until something implements one, executing a cyclic graph would
    mean choosing a fixed point silently.
    """
    node_ids = [node.id for node in spec.nodes]
    incoming: dict[str, set[str]] = {nid: set() for nid in node_ids}
    outgoing: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for edge in spec.edges:
        outgoing[edge.source.node].add(edge.target.node)
        incoming[edge.target.node].add(edge.source.node)

    ready = [nid for nid in node_ids if not incoming[nid]]
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for downstream in sorted(outgoing[nid]):
            incoming[downstream].discard(nid)
            if not incoming[downstream]:
                ready.append(downstream)
    if len(order) != len(node_ids):
        remaining = sorted(set(node_ids) - set(order))
        raise ExecutorError(
            f"the graph has a cycle through {remaining}. allow_cycles declares intent "
            "for an iterative solver; nothing implements one, and executing a cycle "
            "would mean choosing a fixed point without saying so."
        )
    return tuple(order)


def graph_fingerprint(spec: GraphSpec) -> str:
    """SHA-256 over the canonical graph, for provenance and cache keying."""
    payload = spec.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


class AdapterResolver(Protocol):
    def __call__(self, model_id: str) -> Any: ...


class CouplerResolver(Protocol):
    def __call__(self, coupler_id: str) -> Any: ...


def _default_adapter_resolver(model_id: str) -> Any:
    from solvers.registry import get_adapter_for_model

    return get_adapter_for_model(model_id)


#: Coupler id -> the module exposing ``get_coupler()``. A table rather than a
#: scan, for the reason ``solvers/registry.py`` gives about its own: a
#: discovered registry agrees with itself by construction, including about which
#: entries exist.
_COUPLER_MODULES = {
    "C_RAY_TO_WAVE": "couplers.node",
    "C_PLANAR_DOE_STEP": "couplers.doe_node",
    "C_PATCH_WFT": "couplers.patch_node",
}


def _default_coupler_resolver(coupler_id: str) -> Any:
    import importlib

    module_name = _COUPLER_MODULES.get(coupler_id)
    if module_name is None:
        raise ExecutorError(
            f"no executable coupler registered for {coupler_id!r}; "
            f"registered: {sorted(_COUPLER_MODULES)}"
        )
    return importlib.import_module(module_name).get_coupler()


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


@dataclass
class GraphExecutor:
    """Deterministic execution of a validated graph. No LLM, no heuristics.

    Anything requiring judgment belongs to the agent, not here: this decides
    order, plumbing, refusals and bookkeeping, and nothing else.
    """

    registry: Any
    process_model: ProcessModel = ProcessModel.IN_PROCESS
    cache: ExecutionCache | None = None
    #: Poll interval for the memory watchdog. Zero disables it, which is only
    #: appropriate for a unit test that is not running a solver.
    watchdog_interval_s: float = 0.25
    adapter_resolver: AdapterResolver = _default_adapter_resolver
    coupler_resolver: CouplerResolver = _default_coupler_resolver

    def __post_init__(self) -> None:
        if self.process_model is not ProcessModel.IN_PROCESS:
            raise ExecutorError(
                f"process_model={self.process_model.value} is declared and not "
                "implemented: it needs artifact serialization across a process "
                "boundary, and ArtifactRecord keeps arrays in solver-owned storage. "
                "Running in-process while reporting process-per-node would make the "
                "record's process_model field a lie, so this raises instead."
            )

    # -- validation ------------------------------------------------------

    def validate(self, spec: GraphSpec) -> ValidationReport:
        """Structural validation. **No solver is imported by this call.**"""
        return GraphValidator(self.registry).validate(spec)

    def _refuse_unsupported_nodes(self, spec: GraphSpec) -> Refusal | None:
        for node in spec.nodes:
            if bool(node.config.get("streaming", False)):
                return Refusal(
                    kind=RefusalKind.UNSUPPORTED_CAPABILITY,
                    detail=(
                        f"node {node.id!r} declares streaming: true. Chunked coherent "
                        "accumulation exists (couplers/streaming.py, "
                        "core/coherent_batch.py) and has no GraphSpec node contract, so "
                        "the executor cannot chunk it. Running it un-chunked would put "
                        "a 60M-ray workload in memory at once."
                    ),
                    declaration="node.config.streaming",
                    remedy=(
                        "run the workload through its probe until a streaming node "
                        "contract exists; demo3 is the case this affects"
                    ),
                )
            if bool(node.config.get("jax_enable_x64", False)):
                return Refusal(
                    kind=RefusalKind.INVALID_CONFIGURATION,
                    detail=(
                        f"node {node.id!r} asks for jax_enable_x64. It is process-global "
                        "and pinned False everywhere in this repository; flipping it "
                        "would change every recorded number in the process."
                    ),
                    declaration="node.config.jax_enable_x64",
                    remedy="drop the request, or run the workload in its own process",
                )
        for edge in spec.edges:
            if edge.coupler not in _COUPLER_MODULES:
                return Refusal(
                    kind=RefusalKind.UNSUPPORTED_CAPABILITY,
                    detail=(
                        f"edge {edge.id!r} names coupler {edge.coupler}, which is "
                        "declared in the registry and has no executable graph node. "
                        "C_WAVE_TO_RAY is the live case: it is a library component the "
                        "patch and DOE couplers wrap, and it is not itself composable as "
                        "an edge. Refused here rather than at the call site, so a graph "
                        "that cannot run says so before anything is executed."
                    ),
                    declaration=f"edge.coupler = {edge.coupler}",
                    remedy=(
                        "use a coupler with a graph node "
                        f"({sorted(_COUPLER_MODULES)}), or drive this one directly"
                    ),
                )
        if spec.require_verified_gradients:
            unverified = [
                node.id
                for node in spec.nodes
                if not self.registry.models[node.model].derivative.verified
            ]
            if unverified:
                return Refusal(
                    kind=RefusalKind.UNVERIFIED_DERIVATIVE,
                    detail=(
                        "the graph requires verified gradients and these nodes' models "
                        f"declare derivative.verified = false: {unverified}"
                    ),
                    declaration="GraphSpec.require_verified_gradients",
                    remedy="drop the requirement, or verify the derivative first",
                )
        return None

    # -- execution -------------------------------------------------------

    def run(
        self,
        spec: GraphSpec,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        instance_id: str | None = None,
        instance_fingerprint: str | None = None,
        inputs: Mapping[str, ArtifactRecord] | None = None,
    ) -> ExecutionRecord:
        """Execute the graph and return one record of what happened.

        ``inputs`` supplies the graph's declared sources, keyed
        ``"<node>.<port>"``: what a node with no upstream producer consumes. A
        graph whose entry node needs one and is not given one is refused by the
        adapter with the missing port named, rather than run against an invented
        field.

        Refuses, without importing a solver, when: the graph is invalid; it
        declares a capability the executor does not have; or it requires
        verified gradients its models do not have. Every refusal is structured
        and names the declaration that caused it.
        """
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        graph_sha = graph_fingerprint(spec)

        report = self.validate(spec)
        if not report.valid:
            return self._refused(
                run_id,
                graph_sha,
                instance_id,
                instance_fingerprint,
                Refusal(
                    kind=RefusalKind.INVALID_CONFIGURATION,
                    detail="; ".join(i.message for i in report.errors),
                    declaration="GraphSpec",
                    remedy="fix the graph; no solver was invoked",
                ),
                diagnostics=[
                    {"code": issue.code, "detail": issue.message, "location": issue.location}
                    for issue in report.errors
                ],
                seed=seed,
                started=started,
            )

        refusal = self._refuse_unsupported_nodes(spec)
        if refusal is not None:
            return self._refused(
                run_id, graph_sha, instance_id, instance_fingerprint, refusal, [], seed, started
            )

        try:
            order = topological_order(spec)
        except ExecutorError as exc:
            return self._refused(
                run_id,
                graph_sha,
                instance_id,
                instance_fingerprint,
                Refusal(
                    kind=RefusalKind.INVALID_CONFIGURATION,
                    detail=str(exc),
                    declaration="GraphSpec.edges",
                    remedy="break the cycle",
                ),
                [],
                seed,
                started,
            )

        env = _environment_fingerprint()
        watchdog = (
            MemoryWatchdog(interval_s=self.watchdog_interval_s).start()
            if self.watchdog_interval_s > 0
            else None
        )

        # Graph-level declared sources. Keyed ``"<node>.<port>"``, they are what
        # feeds a node with no upstream producer -- the input field of a
        # standalone wave graph, for instance. Resolved exactly like an upstream
        # output, so a node cannot tell the difference and the executor does not
        # need two plumbing paths.
        artifacts: dict[str, Any] = {}
        declared: dict[tuple[str, str], ArtifactRecord] = {}
        for key, artifact in (inputs or {}).items():
            node_id, _, port = str(key).partition(".")
            if not port:
                raise ExecutorError(
                    f"declared source {key!r} must be keyed '<node>.<port>'; without the "
                    "port the executor would have to guess which input it feeds"
                )
            declared[(node_id, port)] = artifact
            artifacts[str(key)] = artifact
        available: dict[tuple[str, str], ArtifactRecord] = {}
        node_records: list[NodeRecord] = []
        diagnostics: list[dict[str, Any]] = []
        states: list[SolverStateProtocol] = []
        stopped = False

        edges_by_target = _edges_by_target(spec)
        node_by_id = {node.id: node for node in spec.nodes}

        try:
            for node_id in order:
                node = node_by_id[node_id]
                if stopped:
                    node_records.append(
                        NodeRecord(
                            node_id=node_id,
                            component=node.model,
                            outcome=NodeOutcome.NOT_REACHED,
                        )
                    )
                    continue

                # -- edges into this node ---------------------------------
                node_inputs, edge_records, edge_diags, edge_failed = self._run_edges(
                    run_id, edges_by_target.get(node_id, ()), available, artifacts
                )
                node_records.extend(edge_records)
                diagnostics.extend(edge_diags)
                if edge_failed:
                    stopped = True
                    node_records.append(
                        NodeRecord(
                            node_id=node_id, component=node.model, outcome=NodeOutcome.NOT_REACHED
                        )
                    )
                    continue

                for (declared_node, port), artifact in declared.items():
                    if declared_node == node_id:
                        node_inputs.setdefault(port, artifact)

                record, outputs, state, ok = self._run_node(
                    run_id, node, node_inputs, seed, env, artifacts
                )
                node_records.append(record)
                states.append(state)
                if not ok:
                    stopped = True
                    continue
                for port, artifact in outputs.items():
                    available[(node_id, port)] = artifact

                if watchdog is not None:
                    watchdog.sample()
                    if watchdog.verdict.breached:
                        stopped = True
                        diagnostics.append(
                            {
                                "code": "RESOURCE_GUARD_TRIPPED",
                                "detail": watchdog.verdict.detail or watchdog.verdict.reason,
                                "reason": watchdog.verdict.reason,
                            }
                        )
        finally:
            if watchdog is not None:
                watchdog.stop()

        wall = time.perf_counter() - started
        breached = watchdog is not None and watchdog.verdict.breached
        completed = [
            r
            for r in node_records
            if r.outcome in (NodeOutcome.EXECUTED, NodeOutcome.EXECUTED_LOSSY)
        ]
        incomplete = len(completed) != len(node_records)

        # PARTIAL means "some of this is usable". A run whose guard tripped is
        # not usable regardless of how far it got -- the numbers were produced
        # under a resource condition the project treats as a stop -- so the
        # breach outranks the arithmetic.
        if breached:
            status = RunStatus.FAILED
        elif not incomplete:
            status = RunStatus.SUCCEEDED
        elif completed:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.FAILED

        run_refusal = (
            Refusal(
                kind=RefusalKind.RESOURCE_GUARD,
                detail=watchdog.verdict.detail or "memory guard tripped",  # type: ignore[union-attr]
                declaration=f"core.resources.MemoryWatchdog.{watchdog.verdict.reason}",  # type: ignore[union-attr]
                remedy=(
                    "the only permitted retry is the same configuration with a smaller "
                    "chunk; growing in this direction again is not"
                ),
            )
            if breached
            else None
        )

        return ExecutionRecord(
            run_id=run_id,
            status=status,
            instance_id=instance_id,
            instance_fingerprint=instance_fingerprint,
            graph_sha256=graph_sha,
            nodes=node_records,
            observed_parameters=_observed_parameters(spec),
            seeds=[seed] if seed is not None else [],
            cost=ResourceCost(
                wall_seconds=wall,
                solver_seconds=sum(
                    r.cost.wall_seconds for r in node_records if r.cost is not None
                ),
                peak_memory_bytes=(watchdog.peak_rss_bytes if watchdog is not None else None),
                device="cpu",
            ),
            refusal=run_refusal,
            provenance=_provenance(
                graph_sha, env, self.process_model, states, seed, spec
            ),
            diagnostics=diagnostics,
            artifacts=artifacts,
        )

    # -- pieces ----------------------------------------------------------

    def _refused(
        self,
        run_id: str,
        graph_sha: str,
        instance_id: str | None,
        instance_fingerprint: str | None,
        refusal: Refusal,
        diagnostics: list[dict[str, Any]],
        seed: int | None,
        started: float,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            run_id=run_id,
            status=RunStatus.FAILED,
            instance_id=instance_id,
            instance_fingerprint=instance_fingerprint,
            graph_sha256=graph_sha,
            nodes=[],
            seeds=[seed] if seed is not None else [],
            cost=ResourceCost(wall_seconds=time.perf_counter() - started),
            refusal=refusal,
            provenance={
                "executor_version": EXECUTOR_VERSION,
                "process_model": str(self.process_model),
                "graph_sha256": graph_sha,
            },
            diagnostics=diagnostics,
        )

    def _run_edges(
        self,
        run_id: str,
        edges: Sequence[EdgeSpec],
        available: Mapping[tuple[str, str], ArtifactRecord],
        artifacts: dict[str, Any],
    ) -> tuple[dict[str, ArtifactRecord], list[NodeRecord], list[dict[str, Any]], bool]:
        """Run every edge feeding one node. Returns its resolved inputs."""
        from couplers.base import CouplerRunRequest

        inputs: dict[str, ArtifactRecord] = {}
        records: list[NodeRecord] = []
        diagnostics: list[dict[str, Any]] = []

        for edge in edges:
            source = available.get((edge.source.node, edge.source.port))
            if source is None:
                records.append(
                    NodeRecord(
                        node_id=edge.id,
                        component=edge.coupler,
                        outcome=NodeOutcome.NOT_REACHED,
                        error_message=(
                            f"{edge.source.node}.{edge.source.port} produced nothing"
                        ),
                    )
                )
                return inputs, records, diagnostics, True

            coupler = self.coupler_resolver(edge.coupler)
            request = CouplerRunRequest(
                run_id=run_id, edge_id=edge.id, source=source, config=dict(edge.config)
            )
            estimate = _safe_estimate(coupler.estimate, request)
            began = time.perf_counter()
            try:
                result = coupler.transform(request)
            except Exception as exc:  # recorded, not swallowed
                records.append(
                    _raised_record(edge.id, edge.coupler, exc, began, estimate)
                )
                diagnostics.append(
                    {
                        "code": "EDGE_RAISED",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "edge": edge.id,
                    }
                )
                return inputs, records, diagnostics, True

            elapsed = time.perf_counter() - began
            contract_codes = _contract_codes(result.diagnostics, result.error_type)
            if result.status is not RunStatus.SUCCEEDED or result.target is None:
                records.append(
                    NodeRecord(
                        node_id=edge.id,
                        component=edge.coupler,
                        outcome=NodeOutcome.REFUSED,
                        refusal=Refusal(
                            kind=_refusal_kind_for(contract_codes),
                            detail=result.error_message or "coupler declined the handoff",
                            declaration=str(result.diagnostics.get("declaration") or edge.id),
                            remedy=str(result.diagnostics.get("remedy") or "") or None,
                        ),
                        error_type=result.error_type,
                        error_message=result.error_message,
                        contract_codes=contract_codes,
                        cost=ResourceCost(wall_seconds=elapsed, estimate=estimate),
                    )
                )
                diagnostics.append(
                    {"code": "EDGE_REFUSED", "detail": result.error_message, "edge": edge.id}
                )
                return inputs, records, diagnostics, True

            key = f"{edge.id}:{edge.target.port}"
            artifacts[key] = result.target
            inputs[edge.target.port] = result.target
            records.append(
                NodeRecord(
                    node_id=edge.id,
                    component=edge.coupler,
                    outcome=NodeOutcome.EXECUTED,
                    contract_codes=contract_codes,
                    device_precision=_observe_precision(dict(edge.config), result.target),
                    cost=ResourceCost(wall_seconds=elapsed, estimate=estimate),
                    outputs=[key],
                )
            )
        return inputs, records, diagnostics, False

    def _run_node(
        self,
        run_id: str,
        node: NodeSpec,
        node_inputs: Mapping[str, ArtifactRecord],
        seed: int | None,
        env: str,
        artifacts: dict[str, Any],
    ) -> tuple[NodeRecord, dict[str, ArtifactRecord], SolverStateProtocol, bool]:
        from solvers.base import ModelRunRequest

        config = dict(node.config)
        if seed is not None:
            # One seed policy, threaded rather than left to each adapter's
            # default. Recorded per node so two runs at one seed are comparable.
            config.setdefault("seed", seed)

        state = SolverStateProtocol(
            node_id=node.id,
            requested={
                "backend": config.get("backend"),
                "device": config.get("device"),
                "precision": config.get("precision"),
            },
        )

        adapter = self.adapter_resolver(node.model)
        request = ModelRunRequest(
            run_id=run_id, node_id=node.id, inputs=dict(node_inputs), config=config
        )
        estimate = _safe_estimate(adapter.estimate, request)

        cache_key = _cache_key(node, node_inputs, env, state, seed)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                outputs, diags = cached
                for port, artifact in outputs.items():
                    artifacts[f"{node.id}:{port}"] = artifact
                return (
                    NodeRecord(
                        node_id=node.id,
                        component=node.model,
                        outcome=NodeOutcome.EXECUTED,
                        contract_codes=_contract_codes(diags),
                        cost=ResourceCost(wall_seconds=0.0, estimate=estimate),
                        outputs=[f"{node.id}:{port}" for port in outputs],
                    ),
                    dict(outputs),
                    state,
                    True,
                )

        began = time.perf_counter()
        try:
            result = adapter.run(request)
        except Exception as exc:  # recorded, not swallowed
            return (
                _raised_record(node.id, node.model, exc, began, estimate),
                {},
                state,
                False,
            )
        elapsed = time.perf_counter() - began

        contract_codes = _contract_codes(result.diagnostics, result.error_type)
        applied = result.diagnostics.get("execution") or result.diagnostics.get("applied")
        state = SolverStateProtocol(
            node_id=node.id,
            requested=state.requested,
            applied=applied if isinstance(applied, Mapping) else None,
        )

        if result.status is not RunStatus.SUCCEEDED:
            return (
                NodeRecord(
                    node_id=node.id,
                    component=node.model,
                    outcome=NodeOutcome.REFUSED,
                    refusal=Refusal(
                        kind=_refusal_kind_for(contract_codes),
                        detail=result.error_message or "the adapter declined the request",
                        declaration=str(result.diagnostics.get("stage") or node.id),
                    ),
                    error_type=result.error_type,
                    error_message=result.error_message,
                    contract_codes=contract_codes,
                    cost=ResourceCost(wall_seconds=elapsed, estimate=estimate),
                ),
                {},
                state,
                False,
            )

        outputs = dict(result.outputs)
        for port, artifact in outputs.items():
            artifacts[f"{node.id}:{port}"] = artifact
        if self.cache is not None:
            self.cache.put(cache_key, outputs, dict(result.diagnostics))

        primary = next(iter(outputs.values()), None)
        precision = _observe_precision(config, primary)
        lossy = precision is not None and not precision.honoured

        return (
            NodeRecord(
                node_id=node.id,
                component=node.model,
                outcome=NodeOutcome.EXECUTED_LOSSY if lossy else NodeOutcome.EXECUTED,
                contract_codes=contract_codes,
                device_precision=precision,
                cost=ResourceCost(
                    wall_seconds=elapsed,
                    device=str(primary.device) if primary is not None else "cpu",
                    estimate=estimate,
                ),
                outputs=[f"{node.id}:{port}" for port in outputs],
            ),
            outputs,
            state,
            True,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _edges_by_target(spec: GraphSpec) -> dict[str, list[EdgeSpec]]:
    grouped: dict[str, list[EdgeSpec]] = {}
    for edge in spec.edges:
        grouped.setdefault(edge.target.node, []).append(edge)
    return grouped


def _safe_estimate(estimate: Any, request: Any) -> CostEstimate | None:
    """An estimate is a prediction, and a refusal to predict is information.

    A component that raises here has said it cannot cost this request; that is
    recorded as an absent estimate rather than propagated, because failing a run
    over its cost model would make the model harder to improve than to remove.
    """
    try:
        predicted = estimate(request)
    except Exception:  # a cost model must never fail a run
        return None
    return predicted if isinstance(predicted, CostEstimate) else None


def _raised_record(
    node_id: str, component: str, exc: BaseException, began: float, estimate: CostEstimate | None
) -> NodeRecord:
    """A node that raised, with no partially-populated artifact.

    The outputs list is empty on purpose: a downstream node must not receive a
    half-built artifact, and the executor stops the graph rather than letting it
    try.
    """
    return NodeRecord(
        node_id=node_id,
        component=component,
        outcome=NodeOutcome.RAISED,
        error_type=type(exc).__name__,
        error_message=str(exc)[:2000],
        contract_codes=_contract_codes(
            getattr(exc, "diagnostics", None), getattr(exc, "code", None)
        ),
        cost=ResourceCost(wall_seconds=time.perf_counter() - began, estimate=estimate),
        outputs=[],
    )


def _contract_codes(diagnostics: Any, error_type: str | None = None) -> list[str]:
    """Pull ``ContractCode``-shaped values out of a diagnostics mapping.

    Structured, never free text: the codes are what aggregate across runs, and
    a message is what a human reads afterwards.

    ``error_type`` is consulted too, and that is not tidiness. ``ContractError``
    carries its code *there* -- a refused handoff arrives as
    ``error_type="REFERENCE_PLANE_MISMATCH"`` with nothing in ``diagnostics`` --
    so reading only the mapping would drop the one field a caller can act on and
    leave the code visible only inside a prose message.
    """
    codes: list[str] = []
    if error_type and error_type in _CONTRACT_CODE_NAMES:
        codes.append(error_type)
    if not isinstance(diagnostics, Mapping):
        return codes
    for key in ("code", "contract_code", "codes", "contract_codes", "validation_codes"):
        value = diagnostics.get(key)
        if isinstance(value, str):
            codes.append(value)
        elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
            codes.extend(str(v) for v in value)
    seen: list[str] = []
    for code in codes:
        if code not in seen:
            seen.append(code)
    return seen


def _observe_precision(
    config: Mapping[str, Any], artifact: ArtifactRecord | None
) -> DevicePrecisionObservation | None:
    """Requested versus actual, with actual read off the artifact.

    Requested is not evidence of actual. Chromatix casts to ``complex64``
    unconditionally, so a node that asked for FP64 and reported FP64 because
    that is what it asked for would have recorded a fiction.
    """
    if artifact is None:
        return None
    requested_device = str(config.get("device", "cpu"))
    requested_dtype = str(config.get("dtype") or config.get("precision") or artifact.dtype or "")
    return DevicePrecisionObservation(
        requested_device=requested_device,
        actual_device=str(artifact.device),
        requested_dtype=requested_dtype,
        actual_dtype=str(artifact.dtype or ""),
        actual_namespace=str(artifact.framework),
        measured_loss_relative=None,
        measured_loss_basis=(
            "not measured by the executor: quantifying a downcast needs the same "
            "computation at both precisions, which is a benchmark and not a run"
        ),
    )


def _contract_code_names() -> frozenset[str]:
    """Every declared ``ContractCode`` value, read from the enum.

    Imported lazily and cached: ``core.boundary`` is not needed to run a graph
    that never refuses, and a module-scope import here would make the executor
    depend on the boundary layer for a lookup table.
    """
    from core.boundary import ContractCode

    return frozenset(code.value for code in ContractCode)


class _LazyNames:
    """A set that materializes on first membership test."""

    def __init__(self) -> None:
        self._names: frozenset[str] | None = None

    def __contains__(self, value: object) -> bool:
        if self._names is None:
            self._names = _contract_code_names()
        return value in self._names


_CONTRACT_CODE_NAMES = _LazyNames()


_REFUSAL_BY_CODE = {
    "OPL_REFERENCE_UNVERIFIED": RefusalKind.MISSING_EDGE_DECLARATION,
    "MISSING_DECLARATION": RefusalKind.MISSING_EDGE_DECLARATION,
    "REFERENCE_PLANE_MISMATCH": RefusalKind.MISSING_EDGE_DECLARATION,
    "PAD_STATE_UNKNOWN": RefusalKind.MISSING_EDGE_DECLARATION,
    "UNSUPPORTED_CAPABILITY": RefusalKind.UNSUPPORTED_CAPABILITY,
    "NON_HEXAPOLAR_SAMPLING": RefusalKind.OUT_OF_DECLARED_VALIDITY,
}


def _refusal_kind_for(codes: Sequence[str]) -> RefusalKind:
    """Map a contract code to a refusal kind, defaulting to invalid configuration.

    The mapping is deliberately small and explicit. A code with no entry is an
    invalid configuration rather than being guessed into a more specific kind:
    a wrong specific answer is worse than a correct general one, because the
    caller acts on it.
    """
    for code in codes:
        kind = _REFUSAL_BY_CODE.get(code)
        if kind is not None:
            return kind
    return RefusalKind.INVALID_CONFIGURATION


def _cache_key(
    node: NodeSpec,
    inputs: Mapping[str, ArtifactRecord],
    env: str,
    state: SolverStateProtocol,
    seed: int | None,
) -> str:
    """The validity condition, as a key.

    Includes the environment fingerprint and the solver-state digest, so a hit
    cannot cross a differing pinned-solver version or a differing precision
    configuration. That is the difference between a cache and a source of silent
    wrong answers.
    """
    payload = {
        "node": node.model_dump(mode="json"),
        "inputs": {
            port: {"sha256": a.sha256, "id": a.id, "dtype": a.dtype, "device": str(a.device)}
            for port, a in sorted(inputs.items())
        },
        "environment": env,
        "solver_state": state.key(),
        "seed": seed,
        "executor_version": EXECUTOR_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _environment_fingerprint() -> str:
    from core.provenance import environment_fingerprint

    return str(environment_fingerprint()["combined_sha256"])


def _observed_parameters(spec: GraphSpec) -> dict[str, Any]:
    """Every node and edge config, flattened, for the verifier to re-check validity
    against what actually ran rather than what an instance declared."""
    observed: dict[str, Any] = {}
    for node in spec.nodes:
        for key, value in node.config.items():
            observed[f"{node.id}.{key}"] = value
    for edge in spec.edges:
        for key, value in edge.config.items():
            observed[f"{edge.id}.{key}"] = value
    return observed


def _provenance(
    graph_sha: str,
    env: str,
    process_model: ProcessModel,
    states: Sequence[SolverStateProtocol],
    seed: int | None,
    spec: GraphSpec,
) -> dict[str, Any]:
    from core.provenance import environment_fingerprint

    fingerprint = environment_fingerprint()
    return {
        "executor_version": EXECUTOR_VERSION,
        # Part of what makes a fingerprint reproducible: the same graph under a
        # different process model is not the same computation.
        "process_model": str(process_model),
        "graph_sha256": graph_sha,
        "task_id": spec.task_id,
        "environment_sha256": env,
        "python_version": fingerprint["python_version"],
        "packages": fingerprint["packages"],
        "seed": seed,
        "solver_state": [
            {"node": s.node_id, "requested": dict(s.requested), "applied": dict(s.applied or {})}
            for s in states
        ],
        "host_memory": host_memory_snapshot().as_dict(),
    }
