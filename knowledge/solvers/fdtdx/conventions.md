# FDTDX conventions (pinned version `0.6.2`)

Every fact below was either read via `inspect.signature`/`inspect.getsource`
on the installed package, or observed directly by running
`knowledge/solvers/fdtdx/probes/*.py` inside the `agent_solver` container.
None of it is copied from the README/main-branch examples without
re-verifying against the pinned 0.6.2 install (see failure_guide.md for why
that distinction matters here specifically).

## Units

FDTDX uses **SI units directly** in the API surface exercised so far:
`SimulationConfig(time=5e-15, resolution=100e-9, ...)`, `SimulationVolume
(partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6), ...)`, `WaveCharacter
(wavelength=1.0e-6)` -- all in meters/seconds, not microns or normalized
units. This is a real (welcome) difference from Chromatix/FMMAX, which are
unit-scale-agnostic. Not exhaustively confirmed for every parameter (e.g.
material dispersion pole units were not probed), but every quantity probed
here was SI.

## Array shape and axis order

Confirmed via `propagation_probe.py`: for a `(3.0e-6, 3.0e-6, 3.0e-6)`
volume at `resolution=100e-9` (30 grid points per axis),
`arrays.fields.E.shape == (3, 30, 30, 30)`.

**Axis order is `(component, x, y, z)`** -- the leading axis of size 3 is
the vector field component (Ex, Ey, Ez), *not* a trailing axis. Do not
assume `(x, y, z, component)` (the convention some other tools use, e.g.
FDTDX's own `GaussianPlaneSource.fixed_E_polarization_vector=(1, 0, 0)`
matches this leading-component convention). `arrays.fields.H` has the same
shape convention.

## Backend / device selection

`SimulationConfig(backend=...)` accepts `'gpu'` (the literal default in the
signature), `'tpu'`, `'cpu'`, `'METAL'`. Confirmed via
`inspect.getsource(SimulationConfig.__post_init__)`: if the requested
backend is unavailable, it **silently falls back to CPU with a
`logger.warning`** (via loguru) rather than raising. On a CPU-only host,
leaving `backend` unset still works correctly (verified: the constructed
config's `backend` field reads `cpu` even though the signature default
says `'gpu'`), but the fallback is logged noise -- pass `backend='cpu'`
explicitly in any script/adapter that expects to run on CPU, for
determinism and a clean log.

## Boundary conditions

`fdtdx.BoundaryConfig.from_uniform_bound(boundary_type=...)` accepts at
least `'periodic'` (used in the probe) and `'pml'` (per the upstream
example, not probed here). PML thickness and domain-size convergence are
listed as a required probe in `knowledge/solver_cards/fdtdx.yaml` and are
**not yet done** -- the propagation probe here used periodic boundaries
only, to keep the minimal example simple.

## Object graph construction is NOT traceable

`fdtdx.place_objects(object_list=..., config=..., constraints=..., key=...)`
performs **concrete Python-level introspection** of material properties
(e.g. `math.isclose` on permittivity tensor components inside
`fdtdx/materials.py::_is_property_isotropic`, called from
`ArrayContainer.all_objects_isotropic_permittivity`). Calling
`place_objects` inside a `jax.grad`-traced function with a traced value
baked into a `Material(...)` constructor raises
`jax.errors.ConcretizationTypeError`. **Consequence:** the object/array
graph must be built once with concrete (non-traced) values; differentiable
parameters are meant to flow through the separate `params`
(`ParameterContainer`) object that `place_objects` returns, consumed later
by `fdtdx.apply_params(arrays, objects, params, key)`. This project has not
yet worked out the correct `Device`/`ParameterContainer` pattern for
attaching a differentiable parameter -- see `capability_notes.md` and
`solver_card.yaml`.

## A real, reproducible zero-gradient case

Differentiating `sum(E**2)` at the end of a short run with respect to the
**source wavelength** (closed over as a traced scalar passed into
`WaveCharacter(wavelength=...)`, with `place_objects` called with a
concrete permittivity so it doesn't hit the issue above) gives `jax.grad
== 0.0` exactly, while a centered finite difference (`eps=1e-9`) gives a
large nonzero value (~`-1.02e6`). This was reproduced exactly by
`probes/gradient_probe.py` (see `expected/gradient_probe.json`). The most
likely cause is an internal `round()`-based conversion from wavelength to
an integer number of time steps per period somewhere in the temporal
source profile, which has zero gradient almost everywhere. **Treat
`wavelength` as a non-differentiable (or at best not-yet-differentiable)
parameter of a `GaussianPlaneSource`/`WaveCharacter` until this is
investigated further.** This is exactly the kind of "sharp/ill-conditioned
region" CLAUDE.md section 6.2 asks a gradient test to include -- here it
surfaced as a hard break rather than merely reduced accuracy.

## Differentiability mechanism (as designed, per the package's own claim)

The package's stated headline mechanism for efficient gradients is **time
reversal** of Maxwell's equations (`fdtdx.full_backward`), citing Schubert
et al., ACS Omega (2024), rather than "just wrap everything in `jax.grad`
and hope XLA reverse-mode is memory-efficient enough". This project has
not yet exercised `full_backward` or confirmed it produces the expected
adjoint-based gradient; the two probes above only tried the naive
`jax.grad`-over-the-forward-run pattern, which is a different (and, per
the results above, currently broken for the parameters tried) code path.
