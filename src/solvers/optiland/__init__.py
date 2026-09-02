"""Optiland sequential ray tracing, behind an anti-corruption boundary.

CHE-179 / CHE-180 / CHE-181 (R05.1 / R05.2 / R05.3) and CHE-217 (R05.6). The
public surface is two functions, one per kind of input:

```python
solvers.optiland.trace(problem, sampling=..., execution=...) -> RayBundle
solvers.optiland.trace_rays(problem, rays, execution=...) -> RayBundle
```

`trace` generates its rays inside the solver from a field coordinate and a
hexapolar ring count. `trace_rays` **consumes** a `RayBundle` the project already
holds -- what `couplers.scalar_to_ray` produces -- and carries it through the
system with its complex amplitude and its declared quadrature intact. Neither is
expressible as the other: an importance-weighted ensemble drawn from a scalar
grid's angular spectrum is not a hexapolar fan at a field angle, and no field
angle, object distance or ring count reproduces it.

A `problems.RayTraceProblem` goes in and a `representations.RayBundle` comes out.
Nothing else about Optiland crosses this line: no `RealRays`, no `.i`, no `.opd`,
no `opd_native`, no millimetre.
`tests/solvers/test_optiland_boundary.py` asserts that with an AST walk over every
module outside this package and a `sys.modules` check in a fresh interpreter.

Three modules, and the order is the dependency order:

* `system` -- CHE-179. `build_lens(problem)`, the one generic construction path.
  Adding a system means handing it a different problem, never writing a builder.
* `rays` -- CHE-180, CHE-217. Both translations. Native trace to neutral bundle:
  the declared optical path with its versioned reference, the exit-pupil and
  image-space geometry, and the hexapolar quadrature measure. And, for a supplied
  bundle, the reverse direction plus a translation back that takes the amplitude
  and the measure from the *caller* rather than from the trace -- with the
  composed optical path, the launch-surface check and the survival convention each
  named as a rule rather than left in the arithmetic.
  `require_declared_optical_path` is the refusal a consumer can apply to a bundle
  claiming to have come from here: an optical path whose reference is neither of
  this module's two declarations is the native accumulator under a plausible
  label, and it is refused rather than read as a phase.
* `solver` -- CHE-181, CHE-217. The two trace entry points, plus the
  process-global backend, device and precision made explicit and idempotent.

Importing this package imports **no solver**. `optiland` and `torch` are imported
inside the functions that need them, so reading the module -- or the capability
row it cites -- costs neither.

`system` and `rays` are exported for the tests that hold this package's physics
to the frozen records; `trace` and `trace_rays` are the API. A consumer outside
`solvers/` uses one of those two, and the rest of this package is native-facing by
construction.
"""

from solvers.optiland.solver import (
    CAPABILITIES,
    DERIVATIVE,
    Execution,
    Sampling,
    configure_execution,
    trace,
    trace_rays,
)

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "Execution",
    "Sampling",
    "configure_execution",
    "trace",
    "trace_rays",
]
