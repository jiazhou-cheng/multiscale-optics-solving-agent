"""External physics-solver adapters.

Every concrete adapter module in this package must follow this convention, and
must additionally be listed in ``adapters/registry.py``'s ``_REGISTRATIONS``:

  - a module-level ``MODEL_ID: str`` constant naming the registered
    ``ModelSpec.id`` (e.g. ``"M_WAVE_CHROMATIX"``) this adapter implements
  - a module-level ``get_adapter() -> ModelAdapter`` factory function
  - one line in ``adapters/registry.py``. CHE-87 replaced filename scanning
    with an explicit map: a name ending in ``_adapter`` used to be treated as a
    declaration, which imported three gen1 benchmark harnesses that register
    nothing, and could not fail on a duplicated ``MODEL_ID``. Registration is
    now something a module does, not something its filename implies.
  - the external solver package must be imported lazily, inside a private
    ``_import_<solver>()`` helper called from ``run()``/``estimate()``, never
    at module import time -- importing this package (or
    any of its siblings) must never require any heavy optional dependency to be
    installed.

Adapters report failures via the exception hierarchy in
``core.errors``: ``AdapterDependencyError`` for a
missing/unusable dependency, ``UnsupportedCapabilityError`` (raised before
any solver call) for a deliberately unimplemented request, and
``SolverExecutionError`` for a solver that ran but failed on its input.
"""
