"""The by-id index over the catalog, capability queries, and the one call that imports code.

CHE-178 (R03.2), rewired by CHE-221 (R03.4). Module-level state and three
functions, no `Registry` class: a class would be justified if two consumers needed
independent registries, and naming those two consumers is the bar for introducing
one. There is one process and one set of operations, so the module *is* the
registry.

The whole layer exists for one property
---------------------------------------
Asking what this project can do must not load what it can do it with. Optiland
pulls torch, Chromatix pulls JAX, and both cost seconds and a chunk of GPU memory
at import. So a descriptor holds an import path as a string and `resolve` is the
only function here that turns one into an object.
`tests/operations/test_registry_imports_no_backend.py` asserts that against `sys.modules`
in a fresh interpreter, because the failure is transitive: a module that looks
backend-free can pull one three levels down.

Registration is *pulled*, never *pushed*
----------------------------------------
The tempting arrangement is for `solvers/optiland/solver.py` to call a
`register(...)` at import time. That inverts the dependency -- an implementation
would import `operations/` -- and it defeats the property above the moment
anything imports the implementation package for any other reason, because then
listing the registry means the implementations were already loaded. So there is
no import-time scan, no filename convention and no entry-point discovery.

**The registration site is `operations.catalog`, inside this package.** CHE-221
put the fourteen landed operations there, and this module builds its by-id index
from `catalog.CATALOG` at import. That needs no dependency edge in either
direction, because `implementation` is a string: the catalog *names*
`solvers.optiland.solver:trace` without importing it.

So there is no public `register()` any more. `_build_index` below is what replaced
it, and it kept the one behaviour that mattered -- a duplicate id is refused
rather than overwritten -- as an error at first import rather than at a call
nobody makes. The function was public only because a registration site had to live
outside this package; once the site moved in, its last caller was a test
subprocess that existed to make the no-backend check non-vacuous, and the real
catalog does that better (it names `optiland` and `chromatix` outright).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from operations.catalog import CATALOG
from operations.descriptors import SEMANTIC_TYPES, OperationDescriptor, OperationKind

__all__ = ["find", "registered_ids", "resolve"]


def _build_index(
    catalog: tuple[OperationDescriptor, ...],
) -> dict[str, OperationDescriptor]:
    """The by-id mapping over a catalog, refusing a duplicate id.

    `CATALOG` is a tuple and not a dict literal keyed by `operation_id`
    specifically so that this check exists. A dict literal would silently keep the
    last of two entries sharing an id; two descriptors under one id means two
    answers to "what does this operation do", and last-write-wins would make which
    one you get depend on nothing a reader can see.

    Raised at first import of `operations`, so a duplicate cannot reach a caller
    at all. This is what `register()` used to do at a call site, kept after the
    call site moved into this package.
    """
    index: dict[str, OperationDescriptor] = {}
    for descriptor in catalog:
        if not isinstance(descriptor, OperationDescriptor):
            raise TypeError(
                f"the catalog holds {type(descriptor).__name__}, not an "
                "OperationDescriptor"
            )
        existing = index.get(descriptor.operation_id)
        if existing is not None:
            raise ValueError(
                f"{descriptor.operation_id!r} appears twice in the catalog, as "
                f"{existing.implementation!r} and {descriptor.implementation!r}. Ids are "
                "unique; keeping one of the two would make the answer depend on "
                "declaration order. Two records MAY name one callable -- "
                "S_WAVE_CHROMATIX and O_ASM_PROPAGATE do -- but they need two ids."
            )
        index[descriptor.operation_id] = descriptor
    return index


#: The index, derived from the one canonical declaration at import.
#:
#: Not a second source of truth: `catalog.CATALOG` is the declaration and this is a
#: lookup over it. Private, and no longer mutable by anything -- `register()` is
#: gone, so there is no call that can add to it.
_BY_ID: dict[str, OperationDescriptor] = _build_index(CATALOG)


def find(
    *,
    input: str | None = None,
    output: str | None = None,
    kind: OperationKind | str | None = None,
    entry: bool | None = None,
) -> tuple[OperationDescriptor, ...]:
    """Every registered operation matching the filters, ordered by id.

    No argument enumerates the whole catalog, which is what makes "listing
    everything imports no backend" a statement about a real call -- and since
    CHE-221 it is a statement about fourteen real records naming `optiland` and
    `chromatix`, rather than about an empty dict.

    An unknown semantic type or kind is an error, not an empty result. A query
    that silently returns nothing is indistinguishable from a correct answer,
    and `find(input="rays")` -- with the representation named `ray_bundle` --
    is the shape of typo that would otherwise read as a missing capability.

    `input` matches a representation on **any** port; `output` matches the
    **primary** returned value only. CHE-222 (R03.5) added `entry`, which is
    deliberately not spelled `input=None`: that has always meant "do not filter",
    and once an operation could declare `inputs=()` the two readings collided. A
    filter with two meanings is worse than the fake input the same ticket removed,
    so `entry=True` selects the graph entries -- the three sources and the
    problem-driven ray solve -- `entry=False` selects everything that needs an
    upstream edge, and `entry=None` does not filter.
    """
    for name, value in (("input", input), ("output", output)):
        if value is not None and value not in SEMANTIC_TYPES:
            raise ValueError(
                f"{name}={value!r} is not a declared semantic type {list(SEMANTIC_TYPES)}"
            )
    if kind is not None:
        try:
            kind = OperationKind(kind)
        except ValueError as exc:
            raise ValueError(
                f"kind={kind!r} is not one of {[k.value for k in OperationKind]}"
            ) from exc
    if entry is not None and not isinstance(entry, bool):
        # The same rule as the two filters above, for the same reason: `matches`
        # compares with `is`, so `entry=1` or `entry="true"` would return () and be
        # indistinguishable from "there is no such operation".
        raise ValueError(
            f"entry={entry!r} must be True, False or None. True selects the graph "
            "entries, False selects everything that needs an upstream representation, "
            "and None does not filter."
        )
    return tuple(
        _BY_ID[key]
        for key in sorted(_BY_ID)
        if _BY_ID[key].matches(input=input, output=output, kind=kind, entry=entry)
    )


def registered_ids() -> tuple[str, ...]:
    """The ids in the catalog, sorted. For error messages and tests."""
    return tuple(sorted(_BY_ID))


def resolve(operation_id: str) -> Callable[..., Any]:
    """Import the implementation of one operation and return it.

    **The only function in `operations/` that imports anything outside it**, and
    the only place a backend can enter the process through this package. Every
    other call above reads strings.

    The import is not cached here: `importlib.import_module` already returns the
    module from `sys.modules` on a second call, and a cache of our own would be a
    second place the loaded state lives.
    """
    try:
        descriptor = _BY_ID[operation_id]
    except KeyError as exc:
        raise KeyError(
            f"{operation_id!r} is not in the catalog. Catalogued: "
            f"{list(registered_ids())}. An operation is discoverable because "
            "`operations.catalog` declares it, not because something registered it."
        ) from exc

    module_path, _, attribute = descriptor.implementation.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{operation_id!r} declares implementation {descriptor.implementation!r}, but "
            f"{module_path!r} could not be imported: {exc}"
        ) from exc
    try:
        implementation = getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(
            f"{operation_id!r} declares implementation {descriptor.implementation!r}, but "
            f"{module_path!r} has no attribute {attribute!r}"
        ) from exc
    if not callable(implementation):
        raise TypeError(
            f"{operation_id!r} resolves to {type(implementation).__name__}, which is not "
            "callable. An operation is something the runtime can execute."
        )
    resolved: Callable[..., Any] = implementation
    return resolved
