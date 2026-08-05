"""External physics-solver adapters.

Every concrete adapter module in this package must follow this convention so
that `multiscale_optics_agent.adapters.registry` can discover it without any
manually maintained list:

  - file name: ``<solver>_adapter.py`` (must end in ``_adapter``)
  - a module-level ``MODEL_ID: str`` constant naming the registered
    ``ModelSpec.id`` (e.g. ``"M_WAVE_CHROMATIX"``) this adapter implements
  - a module-level ``get_adapter() -> ModelAdapter`` factory function
  - the external solver package must be imported lazily, inside a private
    ``_import_<solver>()`` helper called from ``run()``/``estimate()``, never
    at module import time -- importing this package (or
    ``multiscale_optics_agent`` itself) must never require any heavy optional
    dependency to be installed.

Adapters report failures via the exception hierarchy in
``multiscale_optics_agent.core.errors``: ``AdapterDependencyError`` for a
missing/unusable dependency, ``UnsupportedCapabilityError`` (raised before
any solver call) for a deliberately unimplemented request, and
``SolverExecutionError`` for a solver that ran but failed on its input.
"""
