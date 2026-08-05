"""Deterministic validation for typed model-coupler graphs."""

from __future__ import annotations

from enum import StrEnum

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from multiscale_optics_agent.core.specs import (
    DerivativeMode,
    EdgeSpec,
    GraphSpec,
    ModelSpec,
    PortSpec,
)
from multiscale_optics_agent.registry.loader import Registry


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Severity
    code: str
    message: str
    location: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]


class GraphValidator:
    """Compile-time validator for graph structure and scientific contracts."""

    def __init__(self, registry: Registry):
        self.registry = registry

    def validate(self, spec: GraphSpec) -> ValidationReport:
        issues: list[ValidationIssue] = []
        node_by_id = {node.id: node for node in spec.nodes}

        for node in spec.nodes:
            if node.model not in self.registry.models:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_MODEL",
                        f"Node {node.id!r} references unknown model {node.model!r}.",
                        f"nodes.{node.id}",
                    )
                )

        pair_to_edge: dict[tuple[str, str], EdgeSpec] = {}
        for edge in spec.edges:
            source_node = node_by_id.get(edge.source.node)
            target_node = node_by_id.get(edge.target.node)
            if source_node is None:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_SOURCE_NODE",
                        f"Edge {edge.id!r} references unknown source node {edge.source.node!r}.",
                        f"edges.{edge.id}.source",
                    )
                )
            if target_node is None:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_TARGET_NODE",
                        f"Edge {edge.id!r} references unknown target node {edge.target.node!r}.",
                        f"edges.{edge.id}.target",
                    )
                )
            coupler = self.registry.couplers.get(edge.coupler)
            if coupler is None:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_COUPLER",
                        f"Edge {edge.id!r} references unknown coupler {edge.coupler!r}.",
                        f"edges.{edge.id}",
                    )
                )
            pair = (edge.source.node, edge.target.node)
            if pair in pair_to_edge:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "PARALLEL_EDGE_UNSUPPORTED",
                        "The initial compiler supports one coupler edge per ordered node pair; "
                        f"{pair!r} is used by both {pair_to_edge[pair].id!r} and {edge.id!r}.",
                        f"edges.{edge.id}",
                    )
                )
            else:
                pair_to_edge[pair] = edge

            if source_node is not None and target_node is not None and coupler is not None:
                source_model = self.registry.models.get(source_node.model)
                target_model = self.registry.models.get(target_node.model)
                if source_model is not None and target_model is not None:
                    issues.extend(
                        self._validate_edge_contract(edge, source_model, target_model)
                    )

        graph = nx.DiGraph()
        graph.add_nodes_from(node_by_id)
        graph.add_edges_from(
            (edge.source.node, edge.target.node)
            for edge in spec.edges
            if edge.source.node in node_by_id and edge.target.node in node_by_id
        )
        if not spec.allow_cycles and not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "CYCLE_NOT_ALLOWED",
                    f"Graph contains a cycle but allow_cycles is false: {cycles[:3]!r}.",
                    "edges",
                )
            )

        for variable in spec.design_variables:
            if variable.node not in node_by_id:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_VARIABLE_NODE",
                        f"Design variable {variable.name!r} references unknown node {variable.node!r}.",
                        f"design_variables.{variable.name}",
                    )
                )
        for objective in spec.objectives:
            node = node_by_id.get(objective.node)
            if node is None:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_OBJECTIVE_NODE",
                        f"Objective {objective.name!r} references unknown node {objective.node!r}.",
                        f"objectives.{objective.name}",
                    )
                )
                continue
            model = self.registry.models.get(node.model)
            if model is not None and model.output_port(objective.port) is None:
                issues.append(
                    self._issue(
                        Severity.ERROR,
                        "UNKNOWN_OBJECTIVE_PORT",
                        f"Objective {objective.name!r} references missing output port "
                        f"{objective.port!r} on model {model.id!r}.",
                        f"objectives.{objective.name}",
                    )
                )

        if graph.number_of_nodes() and nx.is_directed_acyclic_graph(graph):
            issues.extend(self._validate_gradient_paths(spec, graph, pair_to_edge, node_by_id))

        if not issues:
            issues.append(
                self._issue(
                    Severity.INFO,
                    "GRAPH_VALID",
                    "Graph passed structural, port, and derivative-path checks.",
                )
            )
        return ValidationReport(issues=issues)

    def _validate_edge_contract(
        self,
        edge: EdgeSpec,
        source_model: ModelSpec,
        target_model: ModelSpec,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        coupler = self.registry.couplers[edge.coupler]
        source_port = source_model.output_port(edge.source.port)
        target_port = target_model.input_port(edge.target.port)

        if source_port is None:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "UNKNOWN_SOURCE_PORT",
                    f"Model {source_model.id!r} has no output port {edge.source.port!r}.",
                    f"edges.{edge.id}.source.port",
                )
            )
        if target_port is None:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "UNKNOWN_TARGET_PORT",
                    f"Model {target_model.id!r} has no input port {edge.target.port!r}.",
                    f"edges.{edge.id}.target.port",
                )
            )
        if source_port is None or target_port is None:
            return issues

        if source_port.artifact != coupler.source.artifact:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "COUPLER_SOURCE_TYPE_MISMATCH",
                    f"{source_model.id}.{source_port.name} emits {source_port.artifact.value}, but "
                    f"{coupler.id} requires {coupler.source.artifact.value}.",
                    f"edges.{edge.id}",
                )
            )
        if coupler.target.artifact != target_port.artifact:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "COUPLER_TARGET_TYPE_MISMATCH",
                    f"{coupler.id} emits {coupler.target.artifact.value}, but "
                    f"{target_model.id}.{target_port.name} requires {target_port.artifact.value}.",
                    f"edges.{edge.id}",
                )
            )

        issues.extend(
            self._metadata_issues(
                producer=source_port,
                consumer=coupler.source,
                location=f"edges.{edge.id}.source",
                code="COUPLER_SOURCE_METADATA_MISSING",
            )
        )
        issues.extend(
            self._metadata_issues(
                producer=coupler.target,
                consumer=target_port,
                location=f"edges.{edge.id}.target",
                code="COUPLER_TARGET_METADATA_MISSING",
            )
        )

        if source_port.units and coupler.source.units and source_port.units != coupler.source.units:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "COUPLER_SOURCE_UNIT_MISMATCH",
                    f"Source units {source_port.units!r} do not match coupler units "
                    f"{coupler.source.units!r}.",
                    f"edges.{edge.id}.source",
                )
            )
        if coupler.target.units and target_port.units and coupler.target.units != target_port.units:
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "COUPLER_TARGET_UNIT_MISMATCH",
                    f"Coupler units {coupler.target.units!r} do not match target units "
                    f"{target_port.units!r}.",
                    f"edges.{edge.id}.target",
                )
            )
        return issues

    def _validate_gradient_paths(
        self,
        spec: GraphSpec,
        graph: nx.DiGraph,
        pair_to_edge: dict[tuple[str, str], EdgeSpec],
        node_by_id: dict[str, object],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for variable in spec.design_variables:
            if not variable.requires_gradient or variable.node not in graph:
                continue
            for objective in spec.objectives:
                if not objective.requires_gradient or objective.node not in graph:
                    continue
                if variable.node == objective.node:
                    paths = [[variable.node]]
                elif nx.has_path(graph, variable.node, objective.node):
                    paths = list(nx.all_simple_paths(graph, variable.node, objective.node))
                else:
                    issues.append(
                        self._issue(
                            Severity.ERROR,
                            "NO_OBJECTIVE_PATH",
                            f"No graph path connects design variable {variable.name!r} to objective "
                            f"{objective.name!r}.",
                            f"design_variables.{variable.name}",
                        )
                    )
                    continue

                path_is_usable = False
                path_messages: list[str] = []
                for path in paths:
                    usable, messages = self._check_one_gradient_path(
                        path,
                        pair_to_edge,
                        node_by_id,
                        strict=spec.require_verified_gradients,
                    )
                    path_messages.extend(messages)
                    path_is_usable = path_is_usable or usable
                if not path_is_usable:
                    issues.append(
                        self._issue(
                            Severity.ERROR,
                            "NO_DIFFERENTIABLE_PATH",
                            f"No admissible derivative path connects {variable.name!r} to "
                            f"{objective.name!r}: {'; '.join(path_messages)}",
                            f"design_variables.{variable.name}",
                        )
                    )
                elif path_messages:
                    issues.append(
                        self._issue(
                            Severity.WARNING,
                            "GRADIENT_PATH_NOT_FULLY_VERIFIED",
                            f"Derivative path {variable.name!r} → {objective.name!r} is executable "
                            f"but has qualifications: {'; '.join(sorted(set(path_messages)))}",
                            f"design_variables.{variable.name}",
                        )
                    )
        return issues

    def _check_one_gradient_path(
        self,
        path: list[str],
        pair_to_edge: dict[tuple[str, str], EdgeSpec],
        node_by_id: dict[str, object],
        *,
        strict: bool,
    ) -> tuple[bool, list[str]]:
        messages: list[str] = []
        usable = True
        for node_id in path:
            node = node_by_id[node_id]
            model = self.registry.models[getattr(node, "model")]
            usable, messages = self._update_gradient_status(
                usable,
                messages,
                label=model.id,
                mode=model.derivative.mode,
                verified=model.derivative.verified,
                strict=strict,
            )
        for source, target in zip(path, path[1:]):
            edge = pair_to_edge[(source, target)]
            coupler = self.registry.couplers[edge.coupler]
            usable, messages = self._update_gradient_status(
                usable,
                messages,
                label=coupler.id,
                mode=coupler.derivative.mode,
                verified=coupler.derivative.verified,
                strict=strict,
            )
        return usable, messages

    @staticmethod
    def _update_gradient_status(
        usable: bool,
        messages: list[str],
        *,
        label: str,
        mode: DerivativeMode,
        verified: bool,
        strict: bool,
    ) -> tuple[bool, list[str]]:
        if mode is DerivativeMode.NONE:
            return False, [*messages, f"{label} has no derivative"]
        if strict and not verified:
            return False, [*messages, f"{label} derivative is unverified"]
        if not verified:
            messages.append(f"{label} derivative is unverified")
        if mode in {DerivativeMode.SURROGATE, DerivativeMode.FINITE_DIFFERENCE}:
            messages.append(f"{label} uses {mode.value}")
        return usable, messages

    @staticmethod
    def _metadata_issues(
        *,
        producer: PortSpec,
        consumer: PortSpec,
        location: str,
        code: str,
    ) -> list[ValidationIssue]:
        missing = sorted(set(consumer.requires_metadata) - set(producer.provides_metadata))
        if not missing:
            return []
        return [
            GraphValidator._issue(
                Severity.ERROR,
                code,
                f"Required metadata are not provided: {missing!r}.",
                location,
            )
        ]

    @staticmethod
    def _issue(
        severity: Severity,
        code: str,
        message: str,
        location: str | None = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            location=location,
        )
