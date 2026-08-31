"""What a solver is asked to solve, stated without naming a solver.

A problem is physical intent: this optical system, this light, these units. It
is not a construction procedure, and it is not a name that some registry knows
how to turn into a lens.

`problems/` may import `representations/` and `numerics/`; today it imports
neither, because a ray-trace problem is a prescription rather than physical state
at a boundary. It must never import a solver or a backend --
`scripts/check_dependencies.py` enforces that, and it is what keeps a problem
statable by a caller that has no ray tracer installed.

One module, landed by CHE-156 (R04):

* `ray_trace` -- `RayTraceProblem` and `SurfaceSpec`, the sequential
  ray-tracing problem. Material is a `TypedDict`; aperture, source, field and
  wavelength are plain fields and frozen tuples.

Concrete lens prescriptions are **not** here and are not anywhere under `src/`,
and nothing resolves a prescription name into a lens. The benchmark systems live
in `tests/fixtures/systems.py`, which is where a measured example belongs: it is
evidence used by tests, not a capability the production tree offers by name.
"""

from problems.ray_trace import (
    MATERIAL_KINDS,
    UNITS,
    Material,
    RayTraceProblem,
    SurfaceSpec,
)

__all__ = [
    "MATERIAL_KINDS",
    "UNITS",
    "Material",
    "RayTraceProblem",
    "SurfaceSpec",
]
