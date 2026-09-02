"""Optiland sequential ray tracing, behind an anti-corruption boundary.

CHE-179 / CHE-180 / CHE-181 (R05.1 / R05.2 / R05.3), CHE-217 (R05.6) and
CHE-219 (R05.8). The public surface is two functions, one per kind of input:

```python
backends.optiland.trace(setup, source, sampling=..., execution=..., aiming=...) -> RayBundle
backends.optiland.trace_rays(setup, rays, execution=...) -> RayBundle
```

`trace` launches its rays into the constructed system from a field angle and a
hexapolar ring count. `trace_rays` **consumes** a `RayBundle` the project already
holds -- what `couplers.scalar_to_ray` produces -- and carries it through the
system with its complex amplitude and its declared quadrature intact. Neither is
expressible as the other: an importance-weighted ensemble drawn from a scalar
grid's angular spectrum is not a hexapolar fan at a field angle, and no field
angle, object distance or ring count reproduces it.

A `problems.OpticalSetup` goes in -- with either a `problems.SourceSpec` or a
`representations.RayBundle` as the light -- and a `representations.RayBundle`
comes out. CHE-218 (R05.7) is what made the setup and the illumination
independent, so a setup can be traced at a field angle nothing enumerated in
advance and a supplied bundle needs no source parameters invented for it.
Nothing else about Optiland crosses this line: no `RealRays`, no `.i`, no `.opd`,
no `opd_native`, no millimetre.
`tests/backends/test_optiland_boundary.py` asserts that with an AST walk over every
module outside this package and a `sys.modules` check in a fresh interpreter.

Four modules, and the order is the dependency order:

* `system` -- CHE-179, CHE-218. `build_lens(setup, source)`, the one generic
  construction path. Adding a system means handing it a different setup, never
  writing a builder. The source is a *construction* argument -- the pinned backend
  needs an object surface and one declared field before an `Optic` exists -- and
  `source=None` is the supplied-bundle path, where neither is read.
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
* `launch` -- CHE-219. `launch(lens, source, num_rings=..., aiming=...)`, the
  system-bound materialization of a declarative source: the aimed launch state,
  captured before the trace, with the pupil quadrature and the object-space
  optical-path reference declared from it. A source can be described without an
  optical system; a ray launch cannot, which is why this is the solver's
  operation and not `sources/`'s, and why the constructed lens is a required
  argument rather than something the function could do without. `normalized_field`
  moved here from `solver` for the same reason.
* `solver` -- CHE-181, CHE-217, CHE-219. The two trace entry points, plus the
  process-global backend, device and precision made explicit and idempotent.

Importing this package imports **no solver**. `optiland` and `torch` are imported
inside the functions that need them, so reading the module -- or the capability
row it cites -- costs neither.

`system`, `launch` and `rays` are reachable as submodules for the tests that hold
this package's physics to the frozen records, and `launch` additionally for a
caller that wants a launch bundle without a trace -- it takes native solver state
(the constructed `Optic`) and so is package-facing by construction. `trace` and
`trace_rays` are the API. A consumer outside
`backends/` uses one of those two, and the rest of this package is native-facing by
construction.
"""

from backends.optiland.solver import (
    CAPABILITIES,
    DERIVATIVE,
    Execution,
    Sampling,
    configure_execution,
    trace,
    trace_rays,
)

#: The public callables in this package that are **semantic operations**, as
#: strings, one per `operations.catalog` record. CHE-221 (R03.4).
#:
#: Strings, and *this package does not import* `operations`: the dependency
#: allowlist gives no implementation package an edge to `operations/`, and an edge
#: would end the one property that package exists to provide -- listing what the
#: project can do would have loaded what it does it with.
#: `tests/operations/test_catalog.py` walks this tuple against the catalog in both
#: directions.
#:
#: Hand-maintained, and deliberately not derived from `__all__`:
#: `configure_execution` sets process-global backend state and returns no representation, and
#: `CAPABILITIES`, `DERIVATIVE`, `Execution` and `Sampling` are declarations and argument types.
#: **`launch` is absent on purpose**, and is not in `__all__` either -- see this module's own
#: note on it: it takes native solver state, a constructed `Optic`, and is package-facing by
#: construction. A public launch operation needs a neutral signature first.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = ("trace", "trace_rays")

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "OPERATIONS",
    "Execution",
    "Sampling",
    "configure_execution",
    "trace",
    "trace_rays",
]
