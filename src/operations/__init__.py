"""Operation metadata: what this project can execute, without executing it.

`operations/` is a sibling of the packages that implement operations, not a layer
above them. It imports `numerics/` and nothing else in this project -- in
particular not `solvers/`, `couplers/`, `operators/`, `measurements/` or
`representations/` -- which `scripts/check_dependencies.py` enforces. That is
what makes "reading the registry imports no backend" a structural fact instead of
a discipline someone maintains.

Three modules and one property:

* `descriptors` -- CHE-177 (R03.1). `OperationDescriptor`, one record per
  executable operation, and `OperationKind`, the four kinds as *metadata* rather
  than four class hierarchies.
* `catalog` -- CHE-221 (R03.4). `CATALOG`, the **single canonical declaration** of
  the fourteen operations this project has landed. Every `implementation` is a
  `"module.path:attribute"` string, which is why the catalog can live here without
  any dependency edge to the packages it names.
* `registry` -- CHE-178 (R03.2), rewired by CHE-221. The by-id index over the
  catalog, capability queries, and `resolve` -- the only call in the package that
  imports an implementation.

The catalog is **populated at import and loads no backend**, which is the property
worth pinning and the one `tests/operations/test_registry_imports_no_backend.py`
checks in a fresh interpreter. There is no filename discovery, no import-time scan
and no public `register()`: an operation is discoverable because `catalog.py`
declares it. Each implementation package declares `OPERATIONS`, a tuple of
strings, and `tests/operations/test_catalog.py` walks the two against each other
in both directions so a landed operation cannot exist without exactly one record.

**Production-complete and planner-ready, as of CHE-222 (R03.5).** A descriptor's
`inputs` is a tuple of representation ports with `()` for a graph entry, `returns`
is ordered with the primary result first, and `requires`/`optional` name the
arguments a caller must and may supply. `input` and `output` are gone: single
strings could not say "produces without consuming", and two records were false
because of it. The eight questions those fields answer are listed in
`descriptors.py`, each has a test, and
`tests/operations/test_catalog_signatures.py` derives the argument tuples from
`inspect.signature` and compares -- so the metadata is a checked restatement of
the code rather than a hand-maintained one.

`capabilities` was the last field with an eager coupling, and CHE-223 (R03.6)
removed it: the measured rows are data under `knowledge/capabilities/`, and a
descriptor's `capabilities` is a component id validated for **shape** here and
resolved by `tests/operations/test_capability_references.py`. So both reference
fields now behave the same way, and this package imports nothing in the project --
not even `numerics`.

Device and dtype support is not stored here. A descriptor cites a component by id
and the measurement lives in `knowledge/capabilities/<component>.json` with the
probe that produced it, so there is exactly one place to change it. There is no
YAML mirror of this package and no generated manifest;
`tests/operations/test_descriptors.py` asserts the absence, because the
reference implementation's second source was 45 KB of YAML with a passing test
that it agreed with the first.
"""

from operations.catalog import CATALOG
from operations.descriptors import (
    DERIVATIVE_MODES,
    SEMANTIC_TYPES,
    OperationDescriptor,
    OperationKind,
)
from operations.registry import find, registered_ids, resolve

__all__ = [
    "CATALOG",
    "DERIVATIVE_MODES",
    "SEMANTIC_TYPES",
    "OperationDescriptor",
    "OperationKind",
    "find",
    "registered_ids",
    "resolve",
]
