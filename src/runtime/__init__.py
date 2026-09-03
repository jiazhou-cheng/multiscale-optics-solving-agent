"""Execution bookkeeping: what happened, never whether it was right.

`runtime/` may import `planning/`, `operations/` and `representations/` --
`scripts/check_dependencies.py` enforces that -- and is the top of the graph:
nothing imports it. That direction is what makes the rule this package is built
around structurally true rather than a discipline someone maintains, because a
physical layer *cannot* read a provenance field.

Two modules:

* `records` -- CHE-199 (R13.1). `ExecutionRecord` and `NodeRecord`, the serialized
  provenance models; `strip_volatile`, the source and environment fingerprints, and
  the JSON round trip.
* `executor` -- CHE-200 (R13.2). `execute(plan, *, request)`, and `Executor`, the
  one class in the new architecture justified by rule 3 -- a mutable resource
  lifecycle. The resource is **a memory sampling thread**, started on `__enter__`
  and joined on `__exit__`, and the shared-server policy's swap-growth stop
  condition is what it exists for. It does *not* own backend configuration: this
  package cannot import `backends/`, and `configure_execution` sets device and
  precision on every call rather than inheriting -- `executor.py` records that
  correction, because an earlier version of this note claimed otherwise.

**Deleting any provenance field changes no physical result.** That is §13's rule
and `tests/unit/test_provenance_separation.py` executes it on every field: a
record's `route` and `request` are what a re-run reads, its `nodes` and
`provenance` are observations nothing reads back, and the test re-derives the
physics from the record after stripping each provenance key in turn. A field that
failed would not be provenance -- it would be an input, and it would belong in the
typed representation, problem or request instead.

There is no verdict here, and no field for one. Whether a result was *right* is
the verification layer's question.

No `src/io/`: the target shape named `src/io/artifacts.py`, and under a flat `src/`
namespace root a top-level `io` package shadows the standard library's.
Serialization is `records.to_json` / `records.from_json`.
"""

from runtime.executor import (
    SAMPLE_INTERVAL_S,
    Executor,
    PlanNode,
    execute,
    memory_snapshot,
    normalize_plan,
)
from runtime.records import (
    FINGERPRINTED_PACKAGES,
    NODE_STATUSES,
    PROVENANCE_SCHEMA_VERSION,
    VOLATILE_KEYS,
    ExecutionRecord,
    NodeRecord,
    environment_fingerprint,
    from_json,
    record_provenance,
    require_stable_payload,
    source_fingerprint,
    strip_volatile,
    to_json,
)

__all__ = [
    "FINGERPRINTED_PACKAGES",
    "NODE_STATUSES",
    "PROVENANCE_SCHEMA_VERSION",
    "SAMPLE_INTERVAL_S",
    "VOLATILE_KEYS",
    "ExecutionRecord",
    "Executor",
    "NodeRecord",
    "PlanNode",
    "environment_fingerprint",
    "execute",
    "from_json",
    "memory_snapshot",
    "normalize_plan",
    "record_provenance",
    "require_stable_payload",
    "source_fingerprint",
    "strip_volatile",
    "to_json",
]
