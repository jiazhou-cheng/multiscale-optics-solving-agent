"""One queryable surface over what each component can do, when to use it, and how to connect it.

CHE-114 (M3.2). The project brief's "particularly important question", stated
directly: *how does the agent know what each model can do, when it should be
used, what inputs it accepts, and how it can be connected to other models?*

Today the answer is spread across five places with different formats and
different audiences -- ``core/capabilities.py``, ``registry/*.yaml``,
``knowledge/**``, ``GraphValidator``, and the ledger. A human can read all five.
An agent needs one surface, and building agent-specific glue over five
inconsistent sources is the "agent-specific hack around weak APIs" the brief
says to avoid.

Why this is its own package
---------------------------
The ticket suggested ``registry/introspection.py``. It cannot live there:
``registry/`` declares what exists and must not import ``verification``, and
this API's whole value is joining the registry to the ledger, the families and
the refusal catalogue. ``runtime/`` cannot import ``verification`` either, for
the reason that keeps an executor from grading its own run.

So ``discovery/`` sits above all of them and below ``agent/``. It owns no facts;
every field it returns is derived, and ``tests/test_discovery.py`` fails if an
answer disagrees with its source.
"""

from discovery.api import (
    ComponentDescription,
    ConnectionReport,
    FamilyCoverage,
    KnowledgeView,
    RouteCapability,
    SuitabilityRecord,
    ValidityAnswer,
    check_connection,
    describe_component,
    families_for_component,
    knowledge_for,
    route_capability,
    validity_of,
)

__all__ = [
    "ComponentDescription",
    "ConnectionReport",
    "FamilyCoverage",
    "KnowledgeView",
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
