"""Operation metadata: what this project can execute, without executing it.

`operations/` is a sibling of the packages that implement operations, not a layer
above them. It imports `numerics/` and nothing else in this project -- in
particular not `solvers/`, `couplers/`, `operators/`, `measurements/` or
`representations/` -- which `scripts/check_dependencies.py` enforces. That is
what makes "reading the registry imports no backend" a structural fact instead of
a discipline someone maintains.

Two concepts and one property:

* `descriptors` -- CHE-177 (R03.1). `OperationDescriptor`, one record per
  executable operation, and `OperationKind`, the four kinds as *metadata* rather
  than four class hierarchies.
* `registry` -- CHE-178 (R03.2). Explicit registration, capability queries, and
  `resolve` -- the only call in the package that imports an implementation.

The registry is **empty at import** and stays empty until an operation lands with
a registration site that names it. There is no filename discovery and no
import-time scan.

Device and dtype support is not stored here. A descriptor cites a row of
`numerics.COMPONENT_CAPABILITIES` by name, so the measurement stays with the
probe that produced it and there is exactly one place to change it. There is no
YAML mirror of this package and no generated manifest;
`tests/operations/test_descriptors.py` asserts the absence, because the
reference implementation's second source was 45 KB of YAML with a passing test
that it agreed with the first.
"""

from operations.descriptors import (
    DERIVATIVE_MODES,
    SEMANTIC_TYPES,
    OperationDescriptor,
    OperationKind,
)
from operations.registry import find, register, registered_ids, resolve

__all__ = [
    "DERIVATIVE_MODES",
    "SEMANTIC_TYPES",
    "OperationDescriptor",
    "OperationKind",
    "find",
    "register",
    "registered_ids",
    "resolve",
]
