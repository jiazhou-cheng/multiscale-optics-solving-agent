"""Solver adapters: one package per external backend, and the backend stays inside.

A solver maps a *problem* into a *representation* and owns everything about the
external package it drives -- its API, its version-specific behaviour, its unit
conventions, its process-global state and its failure modes. That ownership is
the point: `backends/<backend>/` is the only place in the tree allowed to import
`<backend>`, which `scripts/check_dependencies.py` enforces as a rule about every
*other* package rather than as a permission granted to this one.

`backends/` may import `problems/`, `representations/` and `numerics/`. It may not
import `operations/`, `couplers/`, `operators/` or `measurements/`: a solver that
knew about the registry would make reading the registry import a backend, and a
solver that knew about a coupler would put the coupler's conventions inside the
solver's anti-corruption boundary instead of outside it.

Two backends have landed:

* `optiland` -- CHE-179/180/181 (R05.1/R05.2/R05.3), CHE-217/218 (R05.6/R05.7).
  Sequential ray tracing, with two entry points that differ in what the light is:
  `backends.optiland.trace(setup, source, sampling=..., execution=...)` takes an
  `OpticalSetup` plus a declarative `SourceSpec`, and
  `backends.optiland.trace_rays(setup, rays, execution=...)` takes the same setup
  plus a `RayBundle` the project already holds. Both return a `RayBundle`. Nothing
  else about Optiland is observable from outside the package: no `RealRays`, no
  `.i`, no `.opd`, no millimetre.
* `chromatix` -- CHE-183/184 (R06.1/R06.2). Scalar-wave angular-spectrum
  propagation. `backends.chromatix.propagate(field, distance_m=..., model=...)`
  takes a `ScalarField` and returns a `ScalarField`, whose typed `validity` says
  whether its phase is absolute or carrier-removed. No chromatix type and no JAX
  buffer crosses the line.

`chromatix` holds a *physical operator* -- propagation changes physical state
rather than representation -- and that is not a violation of what this package
is. Backend ownership beats taxonomy: the operation is inseparable from the
package's FFT convention, frequency grid, padding and evanescent policy, so
relocating it to `operators/` would put a forwarding wrapper outside the
anti-corruption boundary and the conventions inside it. The *descriptor* records
the operation as a `physical_operator`; the code lives with the backend it drives.
"""
