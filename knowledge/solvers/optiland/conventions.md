# Optiland conventions (pinned version `0.6.0`)

Every fact below was either read from `optiland.backend.__doc__` /
`inspect` on the installed package, or observed directly by running
`knowledge/solvers/optiland/probes/*.py` inside the `agent_solver`
container. None of it is copied from memory or from a description of an
older/different version.

## Backend abstraction (the central convention)

Optiland has exactly two numerical backends, selected process-globally at
runtime, not per-object and not at install time:

```python
import optiland.backend as be
be.set_backend('torch')   # or 'numpy' (the default)
```

Verbatim from `optiland.backend.__doc__` (Kramer Harrison, 2025):

> Provides a unified module-level interface for numerical operations backed
> by either NumPy or PyTorch. The default backend is NumPy; switch with
> `set_backend('torch')`.
>
> **Note on `to_numpy`**: `be.to_numpy` is a **boundary utility** for
> converting backend arrays to NumPy at system boundaries (tests, IO,
> visualization). It is not a computation function and breaks the backend
> abstraction. Internal code should import it directly from
> `optiland.backend.utils`.
>
> **Note on thread safety**: `set_backend` modifies global module state and
> is **not thread-safe**. It is intended to be called once at program
> startup or at the beginning of a test. Concurrent calls from multiple
> threads are not supported.

**Implications for this project:**

- Every call to `be.to_numpy` (or `optiland.backend.utils.to_numpy`) is
  exactly the kind of derivative boundary CLAUDE.md section 3 rule 3 (“no
  silent detach or host copy”) requires recording explicitly in provenance
  — treat it the same as a PyTorch `.detach()`/`.cpu()`/NumPy round trip.
- `set_backend` is global, mutable, non-thread-safe process state. **Never
  call it concurrently from multiple tasks/threads** in an agent runtime
  that might run more than one Optiland simulation at once — do it once at
  process/worker startup, or isolate each backend choice in its own
  process.
- `supports_gpu` and `supports_gradients` are **plain module-level
  attributes (`bool`), not functions** — do not call them (`be.supports_gpu()`
  raises `TypeError: 'bool' object is not callable`; use `be.supports_gpu`
  directly). Observed values: `numpy` backend -> both `False`; `torch`
  backend on this CPU-only container -> `supports_gpu=False`,
  `supports_gradients=True`.

## Differentiability is opt-in, not automatic

A bare `pip install optiland` gives you the NumPy backend and **zero
gradients**. Confirmed: `be.supports_gradients` is `False` under the
default backend. Differentiability requires (a) `torch` installed
separately (not a declared optiland dependency — see `failure_guide.md`)
and (b) an explicit `set_backend('torch')` call before building/tracing the
lens. Do not assume `derivative.mode: native_autodiff` in the model
registry is satisfied by installing optiland alone.

## Top-level namespace is nearly empty

`dir(optiland)` returns only `['annotations']` before any submodule is
imported (importing a submodule, e.g. `optiland.backend`, attaches it as an
attribute of the parent package afterward — this is normal Python behavior,
not something optiland does specially). All real functionality lives in
submodules: `optiland.optic` (main `Optic` container), `optiland.backend`,
`optiland.samples` (bundled example systems), `optiland.surfaces`,
`optiland.rays`, `optiland.paraxial`, `optiland.wavefront`, `optiland.psf`,
`optiland.mtf`, `optiland.zernike`, `optiland.tolerancing`,
`optiland.optimization`, `optiland.materials`, `optiland.coatings`,
`optiland.jones`, `optiland.thin_film`, `optiland.ml`,
`optiland.visualization`. Do not expect useful `dir(optiland)` output —
import the specific submodule you need.

## Array / tensor shapes

`Optic.trace(Hx, Hy, wavelength, num_rays=N)` returns an
`optiland.rays.real_rays.RealRays` object. For `ReverseTelephoto`,
`num_rays=16` at `Hx=Hy=0`, `wavelength=0.55` produced `rays.x.shape ==
(817,)` — **not 16**. Ray counts after tracing reflect the actual number of
rays that survive apertures/vignetting/pupil sampling for the requested
grid density, not the raw `num_rays` argument. Do not assume the output ray
count equals the requested `num_rays`.

Under the NumPy backend, `rays.x`/`rays.y` are `numpy.ndarray` with dtype
`float64`. Under the torch backend with an explicit `dtype=torch.float64`
input parameter, the traced ray coordinates remain differentiable
`torch.Tensor`s of the same dtype.

## API surface has already changed once (deprecation, not yet removed)

`Optic.surface_group` is deprecated in favor of `Optic.surfaces` — a real
`DeprecationWarning` is emitted (`Optic.surface_group is deprecated; use
Optic.surfaces instead.`) but the old name still works in `0.6.0`. Prefer
`Optic.surfaces` in any new adapter code; treat `surface_group` as a
staleness signal if it disappears in a future pinned version.

## Units

Not independently verified in this pass (no analytic-oracle probe was run
against a known focal length or refractive index in physical units).
`ReverseTelephoto.paraxial.f2()` returned `2.005240270799113` for the
bundled sample system, in whatever length unit that sample's prescription
uses (not confirmed to be mm, though that is the near-universal convention
in the lens-design field). Do not assume meters; a future adapter must
locate and test the actual unit convention against a known reference
design before trusting cross-solver unit conversion.
