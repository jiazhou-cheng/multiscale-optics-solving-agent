"""Orchestration: run a validated graph, and record what happened.

CHE-113 (M3.1). A package of its own rather than a module in ``core/``, and the
layering test is what said so: ``core/`` is the shared vocabulary every other
package speaks, so anything it imports becomes vocabulary too. An executor has
to reach for ``solvers`` and ``couplers`` by definition -- that is what
orchestration is -- and putting it in ``core/`` would have made both of them
part of the neutral ground the four boundary artifacts stand on.

``runtime/`` sits above ``solvers/`` and ``couplers/`` and below ``agent/``. It
knows how to run a graph; it does not know what any of the numbers mean, which
is ``verification/``'s job and a dependency this package deliberately does not
have.
"""

from runtime.executor import (
    EXECUTOR_VERSION,
    ExecutionCache,
    ExecutorError,
    GraphExecutor,
    InMemoryCache,
    ProcessModel,
    SolverStateProtocol,
    graph_fingerprint,
    topological_order,
)

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
