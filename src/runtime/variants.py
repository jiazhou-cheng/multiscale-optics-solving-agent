"""A graph, with one thing changed on purpose.

CHE-115 (M3.3). A benchmark family declares negative controls -- "negate the
declared OPL", "swap the reconstruction axes" -- and until now the only way to
run one was to call the coupler directly from a driver, which is exactly the
bespoke path the executor exists to remove. A control that cannot be expressed
as a graph document is a control the executor cannot record having run, and a
result that reports it as exercised is then trusting a script.

So a variant is a **new GraphSpec**, not a flag on a run. It re-validates through
the same model, and :func:`runtime.executor.graph_fingerprint` gives it a
different fingerprint by construction -- which is the property that matters: a
perturbed run and its unperturbed control cannot be mistaken for each other in
the record.

What this module deliberately does not do
-----------------------------------------
It does not know what a perturbation *means*. ``{"perturbation": {"opl_sign":
-1}}`` is meaningful to ``C_RAY_TO_WAVE`` and meaningless anywhere else, and the
coupler is what refuses an unknown key (``couplers/node.py::_perturbation``).
Teaching this module the vocabulary would put the same list in two places, and
the copy here would be the one that goes stale.

It also does not merge nested mappings recursively. A one-level merge is
``{**old, **new}`` per config key: naming ``perturbation`` replaces the whole
perturbation, which is the behaviour a control wants -- a control built by
partially overlaying a previous control's fields is not a control anybody
declared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.specs import GraphSpec

__all__ = ["VariantError", "with_config_overrides"]


class VariantError(ValueError):
    """The override named a node or edge the graph does not have.

    Raised rather than ignored. An override silently dropped because of a typo
    produces a run that is identical to the unperturbed one and reports itself
    as a control, which is the single worst outcome available here.
    """


def with_config_overrides(
    spec: GraphSpec,
    *,
    nodes: Mapping[str, Mapping[str, Any]] | None = None,
    edges: Mapping[str, Mapping[str, Any]] | None = None,
    task_id: str | None = None,
) -> GraphSpec:
    """Return a copy of ``spec`` with per-node and per-edge config overridden.

    Parameters
    ----------
    nodes, edges
        ``{id: {config_key: value}}``. Keys present replace; keys absent are
        left alone. An id that is not in the graph is a :class:`VariantError`.
    task_id
        Renames the variant. Worth setting: ``task_id`` is what a reader sees on
        the record, and two records differing only in a config value deep inside
        an edge are otherwise indistinguishable at a glance.

    The input spec is not mutated -- ``model_copy(deep=True)`` first, because a
    caller comparing a variant against its baseline holds both.
    """
    node_overrides = dict(nodes or {})
    edge_overrides = dict(edges or {})

    unknown_nodes = sorted(set(node_overrides) - {node.id for node in spec.nodes})
    unknown_edges = sorted(set(edge_overrides) - {edge.id for edge in spec.edges})
    if unknown_nodes or unknown_edges:
        raise VariantError(
            f"graph {spec.task_id!r} has no node(s) {unknown_nodes!r} and no "
            f"edge(s) {unknown_edges!r}; nothing was overridden"
        )

    variant = spec.model_copy(deep=True)
    for node in variant.nodes:
        override = node_overrides.get(node.id)
        if override:
            node.config = {**node.config, **dict(override)}
    for edge in variant.edges:
        override = edge_overrides.get(edge.id)
        if override:
            edge.config = {**edge.config, **dict(override)}
    if task_id is not None:
        variant.task_id = task_id

    # Re-validate rather than trust the mutation: the config dicts went through
    # a plain dict update, and GraphSpec is a strict model whose invariants are
    # cheaper to re-check than to reason about.
    return GraphSpec.model_validate(variant.model_dump(mode="python"))
