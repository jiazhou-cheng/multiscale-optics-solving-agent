"""Solver adapters: one package per external backend, and the backend stays inside.

A solver maps a *problem* into a *representation* and owns everything about the
external package it drives -- its API, its version-specific behaviour, its unit
conventions, its process-global state and its failure modes. That ownership is
the point: `solvers/<backend>/` is the only place in the tree allowed to import
`<backend>`, which `scripts/check_dependencies.py` enforces as a rule about every
*other* package rather than as a permission granted to this one.

`solvers/` may import `problems/`, `representations/` and `numerics/`. It may not
import `operations/`, `couplers/`, `operators/` or `measurements/`: a solver that
knew about the registry would make reading the registry import a backend, and a
solver that knew about a coupler would put the coupler's conventions inside the
solver's anti-corruption boundary instead of outside it.

One backend has landed:

* `optiland` -- CHE-179/180/181 (R05.1/R05.2/R05.3). Sequential ray tracing.
  `solvers.optiland.trace(problem, sampling=..., execution=...)` takes a
  `RayTraceProblem` and returns a `RayBundle`. Nothing else about Optiland is
  observable from outside the package: no `RealRays`, no `.i`, no `.opd`, no
  millimetre.
"""
