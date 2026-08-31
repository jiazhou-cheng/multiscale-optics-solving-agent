"""Optiland sequential ray tracing, behind an anti-corruption boundary.

CHE-179 / CHE-180 / CHE-181 (R05.1 / R05.2 / R05.3). The public surface is one
function:

```python
solvers.optiland.trace(problem, sampling=..., execution=...) -> RayBundle
```

A `problems.RayTraceProblem` goes in and a `representations.RayBundle` comes out.
Nothing else about Optiland crosses this line: no `RealRays`, no `.i`, no `.opd`,
no `opd_native`, no millimetre.
`tests/solvers/test_optiland_boundary.py` asserts that with an AST walk over every
module outside this package and a `sys.modules` check in a fresh interpreter.

Three modules, and the order is the dependency order:

* `system` -- CHE-179. `build_lens(problem)`, the one generic construction path.
  Adding a system means handing it a different problem, never writing a builder.
* `rays` -- CHE-180. Native trace to neutral bundle: the declared optical path
  with its versioned reference, the exit-pupil and image-space geometry, and the
  hexapolar quadrature measure. `require_declared_optical_path` is the refusal a
  consumer can apply to a bundle claiming to have come from here: an optical path
  whose reference is not this module's own declaration is the native accumulator
  under a plausible label, and it is refused rather than read as a phase.
* `solver` -- CHE-181. The trace entry point, plus the process-global backend,
  device and precision made explicit and idempotent.

Importing this package imports **no solver**. `optiland` and `torch` are imported
inside the functions that need them, so reading the module -- or the capability
row it cites -- costs neither.

`system` and `rays` are exported for the tests that hold this package's physics
to the frozen records; `trace` is the API. A consumer outside `solvers/` uses
`trace`, and the rest of this package is native-facing by construction.
"""

from solvers.optiland.solver import (
    CAPABILITIES,
    DERIVATIVE,
    Execution,
    Sampling,
    configure_execution,
    trace,
)

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "Execution",
    "Sampling",
    "configure_execution",
    "trace",
]
