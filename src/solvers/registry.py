"""The explicit map from model id to the adapter module that implements it.

This used to scan the package directory for ``*_adapter.py`` and import every
match. That is convenient exactly once -- when adding an adapter -- and wrong
the rest of the time, for two reasons this module exists to fix:

1. **A filename is not a declaration.** Three gen1 benchmark harnesses ended in
   ``_adapter.py``, so discovery imported all three on every lookup while they
   registered nothing. Import side effects of code nobody asked for are not
   free, and "is this a runnable entry point?" became a question about a
   filename rather than about intent.
2. **Scanning cannot fail loudly.** Two modules claiming one ``MODEL_ID`` was
   resolved silently by whichever came later in directory order.

Adding an adapter now costs one line here. That line is the registration.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import cast

from core.errors import AdapterNotFoundError
from solvers.base import ModelAdapter

#: ``(model_id, module)``, one entry per executable adapter. A sequence rather
#: than a dict literal so a duplicated id is a startup error instead of a silent
#: last-one-wins. The module is a full dotted path, not a name relative to this
#: package: CHE-90 moved the adapters into per-solver subpackages, and a
#: relative name would have to encode that layout here as well as in the tree.
_REGISTRATIONS: tuple[tuple[str, str], ...] = (
    ("M_RAY_OPTILAND", "solvers.optiland.adapter"),
    ("M_WAVE_CHROMATIX", "solvers.chromatix.adapter"),
)


@lru_cache(maxsize=1)
def _registry() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model_id, module_name in _REGISTRATIONS:
        if model_id in mapping:
            raise RuntimeError(
                f"Duplicate adapter registration for {model_id!r}: "
                f"{mapping[model_id]!r} and {module_name!r} both claim it. "
                "One model id maps to exactly one adapter module; remove or "
                "rename one of them in solvers/registry.py."
            )
        mapping[model_id] = module_name
    return mapping


def available_model_ids() -> frozenset[str]:
    """Model IDs with a registered adapter module."""
    return frozenset(_registry())


def get_adapter_for_model(model_id: str) -> ModelAdapter:
    """Instantiate the adapter registered for ``model_id``.

    Raises ``AdapterNotFoundError`` if no module is registered for this ID.
    """
    module_name = _registry().get(model_id)
    if module_name is None:
        raise AdapterNotFoundError(
            f"No adapter implementation registered for {model_id!r}. "
            f"Registered: {sorted(_registry())}."
        )
    module = importlib.import_module(module_name)
    declared = getattr(module, "MODEL_ID", None)
    if declared != model_id:
        raise RuntimeError(
            f"solvers/registry.py maps {model_id!r} to {module_name!r}, but that "
            f"module declares MODEL_ID={declared!r}. The map and the module must "
            "agree; fix whichever is wrong."
        )
    return cast(ModelAdapter, module.get_adapter())
