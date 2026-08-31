"""Explicit registration, capability queries, and the one call that imports code.

CHE-178 (R03.2). Module-level state and four functions, no `Registry` class: a
class would be justified if two consumers needed independent registries, and
naming those two consumers is the bar for introducing one. There is one process
and one set of operations, so the module *is* the registry.

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
The tempting arrangement is for `solvers/optiland/adapter.py` to call
`register(...)` at import time. That inverts the dependency -- an implementation
would import `operations/` -- and it defeats the property above the moment
anything imports the implementation package for any other reason, because then
listing the registry means the implementations were already loaded. So there is
no import-time scan, no filename convention and no entry-point discovery: a
descriptor is in the registry because a registration site constructed it and
called `register`.

**Today there is no registration site, and the registry is empty at import.** No
operation has landed: `solvers/`, `couplers/`, `operators/` and `measurements/`
do not exist in the new tree. When the first one lands it brings its descriptor
and the call that registers it. An empty registry is the honest state, and
`tests/operations/test_registry.py` pins it rather than leaving it to be
discovered.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from operations.descriptors import SEMANTIC_TYPES, OperationDescriptor, OperationKind

__all__ = ["find", "register", "registered_ids", "resolve"]

#: The registry. A dict keyed by operation id, mutated only by `register`.
#:
#: Module-level mutable state is a real cost and it is taken deliberately: the
#: alternative is threading a registry object through planning and the runtime
#: for a mapping there is exactly one of. It is private so that the only way to
#: add to it is the function that validates the addition.
_REGISTERED: dict[str, OperationDescriptor] = {}


def register(descriptor: OperationDescriptor) -> OperationDescriptor:
    """Add one operation to the registry, and return it.

    Returns the descriptor so a registration site can read as a declaration:
    `RAY_TO_WAVE = register(OperationDescriptor(...))`.

    A duplicate id is refused rather than overwritten. Two descriptors under one
    id means two answers to "what does this operation do", and last-write-wins
    would make which one you get depend on import order -- the property this
    module is otherwise built to avoid depending on.
    """
    if not isinstance(descriptor, OperationDescriptor):
        raise TypeError(
            f"register() takes an OperationDescriptor, got {type(descriptor).__name__}"
        )
    existing = _REGISTERED.get(descriptor.operation_id)
    if existing is not None:
        raise ValueError(
            f"{descriptor.operation_id!r} is already registered, as "
            f"{existing.implementation!r}. Registration is explicit and ids are unique; "
            "overwriting one would make the answer depend on import order."
        )
    _REGISTERED[descriptor.operation_id] = descriptor
    return descriptor


def find(
    *,
    input: str | None = None,
    output: str | None = None,
    kind: OperationKind | str | None = None,
) -> tuple[OperationDescriptor, ...]:
    """Every registered operation matching the filters, ordered by id.

    No argument enumerates the whole registry, which is what makes "listing
    everything imports no backend" a statement about a real call.

    An unknown semantic type or kind is an error, not an empty result. A query
    that silently returns nothing is indistinguishable from a correct answer,
    and `find(input="rays")` -- with the representation named `ray_bundle` --
    is the shape of typo that would otherwise read as a missing capability.
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
    return tuple(
        _REGISTERED[key]
        for key in sorted(_REGISTERED)
        if _REGISTERED[key].matches(input=input, output=output, kind=kind)
    )


def registered_ids() -> tuple[str, ...]:
    """The ids currently registered, sorted. For error messages and tests."""
    return tuple(sorted(_REGISTERED))


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
        descriptor = _REGISTERED[operation_id]
    except KeyError as exc:
        raise KeyError(
            f"{operation_id!r} is not registered. Registered: "
            f"{list(registered_ids()) or '(nothing -- no operation has landed yet)'}"
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
