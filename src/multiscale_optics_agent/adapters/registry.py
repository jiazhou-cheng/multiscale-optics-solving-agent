"""Runtime discovery of concrete ModelAdapter implementations.

This module enumerates adapters by scanning the ``adapters`` package
directory rather than importing a manually maintained list, so adding a new
solver adapter never requires editing this file. See ``adapters/__init__.py``
for the naming convention every adapter module must follow.
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from typing import cast

import multiscale_optics_agent.adapters as _adapters_pkg
from multiscale_optics_agent.adapters.base import ModelAdapter
from multiscale_optics_agent.core.errors import AdapterNotFoundError


@lru_cache(maxsize=1)
def _discover() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for info in pkgutil.iter_modules(_adapters_pkg.__path__):
        if not info.name.endswith("_adapter"):
            continue
        module = importlib.import_module(f"multiscale_optics_agent.adapters.{info.name}")
        model_id = getattr(module, "MODEL_ID", None)
        if model_id:
            mapping[model_id] = info.name
    return mapping


def available_model_ids() -> frozenset[str]:
    """Model IDs with a discoverable adapter module."""
    return frozenset(_discover())


def get_adapter_for_model(model_id: str) -> ModelAdapter:
    """Instantiate the adapter registered for ``model_id``.

    Raises ``AdapterNotFoundError`` if no adapter module declares this ID.
    """
    module_name = _discover().get(model_id)
    if module_name is None:
        raise AdapterNotFoundError(f"No adapter implementation registered for {model_id!r}")
    module = importlib.import_module(f"multiscale_optics_agent.adapters.{module_name}")
    return cast(ModelAdapter, module.get_adapter())
