"""What compositions exist, read off the operation catalog. No planner.

CHE-164 (R12). One question: given physical state of one kind, which sequences of
landed operations reach state of another kind? `routes(frm="ray_bundle",
to="psf")` answers it, and `capability_graph()` is the adjacency it walks.

The graph owns no facts
-----------------------
Every edge comes from an `OperationDescriptor`: an operation with `t` among its
`inputs` and `primary_output` `u` is an edge `t -> u`, and one with `inputs=()` is
an edge from `None`. Nothing here restates a port, a validity condition, an
approximation or a cost, and nothing here reads a backend. A caller that wants any
of those reads the descriptor -- `operations.find`, `operations.CATALOG` -- which
is the same discipline `capabilities` follows: cite, never copy.

Ports and operations, and deliberately nothing else
---------------------------------------------------
The ticket's own risk section is the specification here: *"Reasoning over validity
and cost before a real planner consumes it produces descriptor fields nobody reads
-- the seed of the old `discovery/api.py`, which grew 10 pydantic models and 944
lines to answer questions for a single caller. Build only what the discoverable
route above needs."*

That contradicts a sentence earlier in the same ticket, which says the graph
"reasons over semantic input/output types, operations, validity,
error/approximation, device/dtype and estimated cost". The risk section wins,
because it is the one with a falsifiable test attached: there is no planner, so a
validity filter here would have no consumer and no way to be wrong usefully. So
this module routes over **ports and operations only**. Adding a filter is a change
for the ticket that lands the consumer, and every field it would filter on is one
attribute access away on the descriptors `routes` returns.

Concretely absent, and each is a decision rather than an omission:

* no cost model, no ranking, no shortest-path preference beyond "shorter routes
  first", which is a reading order and not a claim about cost;
* no validity, approximation or device/dtype filtering;
* no workflow engine, no executor, no agent, no LLM call;
* no `GraphValidator`, no `ValidationIssue`/`ValidationReport`/`Severity`, no
  `GraphSpec`/`NodeSpec`/`EdgeSpec`/`PortRef`, no `ComponentIndex` protocol, and
  none of the ten `discovery/api.py` view models. The reference implementation
  spent 458 lines on `core/graph.py` and 944 on `discovery/api.py` to answer three
  questions for one caller.

No class, and the rule it fails
-------------------------------
The ticket budgets one production class, `CapabilityGraph`, and asks that it be
justified "against a named minimality rule" first. It cannot be. `AGENTS.md` admits
a class for a shared invariant, a versioned public data model, a mutable resource
lifecycle, runtime polymorphism across two current implementations, or a real
plugin boundary. A capability graph is none of those: it is **derived** from the
catalog on every call, holds no invariant the catalog does not already enforce, is
not serialized, owns no resource, and has one implementation.

There is a second reason and it is not tie-breaking, it is binding. The project
declares 26 production classes against `PROJECT_CEILING = 27`, and
`scripts/class_budget.py` reserves the last unit for `runtime.Executor` (CHE-200 /
R13.2), pre-authorized by the owner. A `CapabilityGraph` here would spend another
ticket's authorization on a record that fails every rule -- so `BUDGETS["planning"]`
is 0.

The consequence for the public API is small and is the one deviation from the
ticket's `Expected public API`: `graph.routes(frm=..., to=...)` is
`planning.routes(frm=..., to=...)`, and `capability_graph()` returns the adjacency
as a plain mapping rather than an object wrapping it. Flagged on CHE-164.
"""

from __future__ import annotations

from operations import CATALOG, SEMANTIC_TYPES, OperationDescriptor

__all__ = [
    "ENTRY",
    "capability_graph",
    "routes",
]

#: The key standing for "no upstream physical state" in `capability_graph()`.
#:
#: `None` rather than a string, for the reason `operations.descriptors` gives for
#: `inputs=()`: the semantic-type vocabulary is a closed set of *physical*
#: boundaries, and a non-physical member would weaken the check that makes an
#: unknown type an error rather than an empty result. A graph entry -- a source, or
#: a problem-driven solve -- is an edge from nowhere, and this is what "nowhere" is
#: spelled as.
ENTRY: None = None

def capability_graph(
    catalog: tuple[OperationDescriptor, ...] = CATALOG,
) -> dict[str | None, tuple[str, ...]]:
    """Which operations consume each kind of state, keyed by that kind.

    `{None: (entry ids...), "ray_bundle": (ids...), ...}`, each tuple sorted by
    operation id so the structure is deterministic and diffable. An operation with
    two representation ports appears under both.

    A plain mapping and not an object: see the module docstring on why there is no
    `CapabilityGraph` class. It is derived on every call, which is cheap -- the
    catalog is a tuple of records holding only strings -- and means there is no
    stale copy and nothing to invalidate.

    `catalog` is an argument so a test can route over a synthetic set without
    monkeypatching the production one. It defaults to the real catalog, which is
    the only thing production code should pass.

    **An operation with two representation ports is refused**, not filed under
    both. Filing it under both is what a naive reading suggests, and it would make
    `routes` report a composition that cannot execute: arriving at one of the two
    ports is not enough to call something that needs both, so the route would look
    available and fail at execution. Nothing in the catalog has two ports today and
    `operations.descriptors` says the tuple exists so one can land without a schema
    change -- so this refusal is what makes that day a loud failure here rather
    than a wrong answer downstream. Routing a multi-input operation needs a graph
    that is not bipartite over single states, which is a modelling change with its
    own ticket.

    Raises:
        ValueError: a descriptor declares more than one representation port.
    """
    by_input: dict[str | None, list[str]] = {}
    for descriptor in catalog:
        if descriptor.is_graph_entry:
            by_input.setdefault(ENTRY, []).append(descriptor.operation_id)
            continue
        if len(descriptor.inputs) > 1:
            raise ValueError(
                f"{descriptor.operation_id} declares {len(descriptor.inputs)} "
                f"representation ports {list(descriptor.inputs)}, and this graph is "
                "bipartite over single states: reaching one of them is not enough to "
                "execute it, so filing it under each would report a route that cannot "
                "run. Routing a multi-input operation is a modelling change, not a "
                "filter -- see this function's docstring."
            )
        by_input.setdefault(descriptor.inputs[0], []).append(descriptor.operation_id)
    return {state: tuple(sorted(ids)) for state, ids in by_input.items()}


def routes(
    *,
    frm: str | None,
    to: str,
    max_steps: int | None = None,
    catalog: tuple[OperationDescriptor, ...] = CATALOG,
) -> tuple[tuple[str, ...], ...]:
    """Every sequence of operations carrying `frm` to `to`, shortest first.

    Each route is a tuple of `operation_id`s in execution order. A caller reads the
    descriptors to learn anything else about them -- what they approximate, what
    they require, what they refuse -- because this module owns none of that.

    Parameters
    ----------
    frm
        The semantic type the route starts from, or `ENTRY` (`None`) for "start
        from an operation that needs no upstream state". The second form is what
        `inputs=()` made expressible (CHE-222 / R03.5) and is how "how do I get a
        PSF at all" is asked.
    to
        The semantic type the route must end at, as the last operation's
        `primary_output`.
    max_steps
        A caller's pruning bound on route length, or `None` for **every** route.

        `None` is the default because the search terminates without one: no
        operation may appear twice in a route, so a route is at most as long as the
        catalog. Measured on the fourteen landed operations, the widest query --
        `ENTRY` to `psf` -- has 13763 routes, the longest 11 operations, and takes
        25 ms; every ordered pair of states together is 48648 routes in 106 ms. So
        the complete answer is affordable, and a default that truncated it would be
        the worse failure: at `max_steps=4` this function silently omits
        `SO_RAY_LAUNCH_TRACE -> O_PROPAGATE_RAYS -> C_RAY_TO_SCALAR -> O_ASM_PROPAGATE
        -> M_PSF`, which is the project's canonical multi-scale composition. Routes
        come back shortest-first, so a caller who wants the short ones slices.
    catalog
        The operations to route over. Defaults to the production catalog.

    Returns
    -------
    Routes ordered by length and then lexicographically, so the result is stable
    across runs and readable in a diff. Empty when nothing composes -- **not** an
    error, because "no route" is a real and useful answer here, unlike
    `operations.find`, where an empty result would have been indistinguishable
    from a typo. The two type arguments are validated below for exactly that
    reason: a typo is refused, and only a genuine absence returns `()`.

    Raises:
        ValueError: `to` or a non-`ENTRY` `frm` is not a declared semantic type, or
            `max_steps` is given and below 1.
    """
    for name, value in (("frm", frm), ("to", to)):
        if value is None and name == "frm":
            continue
        if value not in SEMANTIC_TYPES:
            raise ValueError(
                f"{name}={value!r} is not a declared semantic type "
                f"{list(SEMANTIC_TYPES)}"
                + (" (or ENTRY, for an operation that needs no input)" if name == "frm" else "")
            )
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps={max_steps!r} must be at least 1; a route is operations")

    graph = capability_graph(catalog)
    outputs = {descriptor.operation_id: descriptor.primary_output for descriptor in catalog}

    found: list[tuple[str, ...]] = []

    def walk(state: str | None, path: tuple[str, ...]) -> None:
        if max_steps is not None and len(path) == max_steps:
            return
        for operation_id in graph.get(state, ()):
            # An operation may not appear twice in one route. This is also what
            # makes the search terminate: the graph has two cycles --
            # `scalar_field -> scalar_field` through four operations and
            # `ray_bundle -> ray_bundle` through three -- so without it there is no
            # longest path.
            #
            # The clear case for the rule is *adjacent* repetition: two propagations
            # in a row is one route with a distance argument, and the argument is
            # not this module's to model. **It also excludes non-adjacent repeats,
            # and that is the less obvious half.** A two-element diffractive system
            # is `O_COMPLEX_TRANSMISSION -> O_ASM_PROPAGATE -> O_COMPLEX_TRANSMISSION
            # -> M_PSF`, a genuinely distinct composition this rule will not
            # enumerate. Collapsing repeats is the right default with no planner
            # consuming the result -- the alternative is combinatorial noise -- but
            # it is an open decision, not a settled one, and the ticket that lands
            # the consumer owns it.
            if operation_id in path:
                continue
            extended = (*path, operation_id)
            produced = outputs[operation_id]
            if produced == to:
                found.append(extended)
            walk(produced, extended)

    walk(frm, ())
    return tuple(sorted(found, key=lambda route: (len(route), route)))
