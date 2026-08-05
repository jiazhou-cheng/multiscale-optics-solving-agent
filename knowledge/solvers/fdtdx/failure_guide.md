# FDTDX failure guide

Real errors hit while building this knowledge pack (2026-07-30), with
repairs. Add to this file rather than silently working around a new one.

## `AttributeError: module 'fdtdx' has no attribute 'UniformGrid'`

**Symptom:** copying the GitHub `main`-branch example
(`examples/simulate_gaussian_source.py`) verbatim, specifically
`fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=100e-9), ...)`.

**Cause:** the pinned release (`0.6.2`, the latest on PyPI as of
2026-07-30) does not have a `grid=`/`UniformGrid` config field at all. The
real signature (confirmed via
`inspect.signature(fdtdx.SimulationConfig.__init__)` on the installed
package) is:
```
SimulationConfig(*, time: float, resolution: float,
                  backend: Literal['gpu','tpu','cpu','METAL'] = 'gpu',
                  dtype=..., use_complex_fields=None,
                  courant_factor=0.99, gradient_config=None)
```
The `main`-branch README examples are ahead of (or diverged from) the
released 0.6.2 wheel. Treat GitHub `main` as a discovery aid, not ground
truth for a pinned version -- always re-check `inspect.signature` on the
actually-installed package before trusting any example from the repo.

**Fix:** use `resolution=100e-9` in place of `grid=fdtdx.UniformGrid
(spacing=100e-9)`.

## Default `backend='gpu'` on a CPU-only host

**Symptom:** none, actually -- but it's a real gotcha to know about before
assuming it needs a fix. `SimulationConfig`'s signature default is
`backend='gpu'`.

**Cause / actual behavior:** `SimulationConfig.__post_init__` catches
`jax.devices('gpu')` raising `RuntimeError` and silently downgrades
`self.backend` to `'cpu'`, logging a warning via `loguru`. Confirmed:
constructing `SimulationConfig(time=..., resolution=..., dtype=...)` with
no `backend` argument on this CPU-only container produced a config object
whose `.backend` field reads `'cpu'`.

**Fix:** not strictly required, but pass `backend='cpu'` explicitly in any
script/adapter meant to run on CPU -- it documents intent and avoids a
warning-level log line on every run.

## Missing library for gmsh-adjacent tooling

Not applicable to fdtdx itself (that's a jax-fem issue, see
`knowledge/solvers/jax_fem/failure_guide.md`), but fdtdx does depend on
`gdstk` and `trimesh`/`moviepy` for geometry/video export, none of which
needed extra system libraries in this pass (all had prebuilt manylinux
wheels).

## `jax.errors.ConcretizationTypeError` from `place_objects`

**Symptom:**
```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered
where concrete value is expected: traced array with shape float32[] ...
```
raised from deep inside `fdtdx/materials.py::_is_property_isotropic`
(`math.isclose(prop[0], prop[4])`), when calling `fdtdx.place_objects(...)`
from within a function traced by `jax.grad`/`jax.jit`, where a `Material
(permittivity=<traced value>)` was constructed using that traced value.

**Cause:** `place_objects` performs real Python-level (non-JAX) branching
and introspection on material properties while building the static
object/array graph -- it is not meant to be traced.

**Fix (partial -- not fully solved in this pass):** call `place_objects`
once with concrete values to build the object graph and the `params`
pytree it returns; differentiate only through `apply_params(arrays,
objects, params, key)` + `run_fdtd`, with respect to `params`, not by
closing over traced values in constructor arguments. See
`conventions.md`.

## Phantom zero gradient w.r.t. source wavelength

**Symptom:** `jax.grad(objective)(wavelength)` returns exactly `0.0` with
no error, while a finite-difference check at the same point shows a large,
clearly nonzero true sensitivity (~`-1.02e6` for the probe's objective).

**Cause:** not fully diagnosed. Likely an internal `round()`/integer
step-count-per-period computation somewhere between `WaveCharacter
(wavelength=...)` and the discretized temporal source profile, which has
zero gradient almost everywhere (a classic AD trap: a hard `round()` or
`int()` cast silently kills gradient flow instead of raising).

**Fix:** none yet. Treat wavelength as effectively non-differentiable
through this code path until investigated further. Do not silently accept
a zero-valued AD gradient as evidence of zero true sensitivity -- always
cross-check with finite differences, exactly as CLAUDE.md section 6.2
requires, precisely because this kind of failure produces no exception at
all.

## Python version

Requires `>=3.11,<3.14` (confirmed via the wheel's `Requires-Python`
metadata). The `agent_solver` image's `python:3.12-slim` base satisfies
this.
