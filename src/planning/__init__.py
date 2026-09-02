"""What this project can compose, derived from the operation catalog.

`planning/` imports `operations/` and nothing else in this project --
`scripts/check_dependencies.py` enforces that, and it is what makes "asking what
composes imports no backend" a structural fact rather than a discipline. It does
not import `representations/`, because a route is a statement about metadata and
not about physical state.

One module, landed by CHE-164 (R12):

* `graph` -- `capability_graph()`, the adjacency from each kind of state to the
  operations that consume it, and `routes(frm=..., to=...)`, **every** sequence of
  operations carrying one kind to another, shortest first. No default length bound:
  the search terminates because no operation may repeat in a route, and a bound
  that truncated silently would omit compositions the project cares about.

**There is no planner here, and no class.** This package answers "what compositions
exist"; choosing among them, supplying each operation's required arguments and
executing anything are R13's and later. `graph.py`'s docstring records why
`CapabilityGraph` was not built -- it fails all five of `AGENTS.md`'s class rules
and would have spent the project's last authorized class unit, which
`scripts/class_budget.py` reserves for `runtime.Executor`.

The graph owns no facts. Every edge is read off an `OperationDescriptor`'s `inputs`
and `primary_output`, and nothing here restates a validity condition, an
approximation, a cost or a device. A caller that needs those reads the descriptors
`routes` names.
"""

from planning.graph import ENTRY, capability_graph, routes

__all__ = [
    "ENTRY",
    "capability_graph",
    "routes",
]
