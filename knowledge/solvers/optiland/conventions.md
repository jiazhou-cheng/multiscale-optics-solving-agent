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
  exactly the kind of derivative boundary repository scientific-contract requirements (“no
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

CHE-12 verified that Optiland 0.6.0 geometry coordinates use millimetres and
trace wavelengths use micrometres. The executable probe
`benchmarks/probes/verify_m1_engines.py --engine ray` constructs planar
surfaces separated by `thickness=10.0` and observes every final ray at
`z=10.0`. The pinned `optiland.paraxial` source independently labels an
offset in the same prescription coordinates as `10 mm before first
surface`. A trace requested at `wavelength=0.55` returns `rays.w=0.55`, and
the installed `Optic.trace` contract labels wavelength in micrometres.

At the project boundary, convert geometry with `1e-3 m/mm` and wavelength
with `1e-6 m/um`. The probe observes `rays.opd=12.0`, not the declared
10 mm separation. CHE-30 explains that observation exactly; see the section
below.

## `RealRays.opd` — established convention (CHE-30)

**Superseded:** M1 recorded `opd_reference` and `opd_sign` as `unverified`,
and the coupler contract layer refused a real trace as an optical path
length on that basis. Both are now established. The machine-readable form is
`opd_convention:` in `solver_card.yaml`; the probe is
`probes/opd_convention_probe.py`, its recorded output is
`expected/opd_convention_probe.json`, and `tests/test_optiland_opd_convention.py`
asserts each claim **together with the competing hypothesis it rules out**.

The accumulation site in the pinned install is
`optiland/surfaces/standard_surface.py`:

```python
rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))
```

Four parts, each verified against a manufactured geometry with a closed-form
answer, every case exact to float64 round-off:

1. **Quantity — absolute accumulated *optical* path length.** Not an OPD
   relative to a chief ray, despite the field name. `RealRays` seeds the
   accumulator to zero at construction. An oblique free-space ray accumulates
   `d / N` (the slant path), not `d`; a `n = 1.7`, 6 mm slab contributes
   `10.2 mm`, not `6 mm`. The medium *preceding* a surface weights that
   segment, so the glass index does not appear until the far surface is
   crossed.
2. **Sign — non-negative, larger means longer.** The `be.abs` makes the
   accumulator non-decreasing under propagation and refraction. It is **not
   monotonic in general**: `thin_lens_interaction_model` subtracts
   `(x²+y²)/(2f)` and `phase_interaction_model` adds a signed `opd_shift`, so
   monotonicity is a property of the purely refractive path.
3. **Reference — the ray launch state, and for an infinite object that plane
   is aperture-dependent.** `fields/field_types/angle.py::_get_starting_z_offset`
   computes `offset = EPD - min(positions[1:-1])` and launches at
   `positions[1] - offset`. So **changing the aperture moves the OPL zero**,
   which is the reason an undeclared absolute Optiland OPL is dangerous: it is
   only meaningful at a declared EPD. Verified across `EPD` = 2.0, 4.0, 7.5 mm,
   where the observed OPL tracks `EPD + separation` exactly and the
   "reference is the first surface" hypothesis is wrong by exactly `EPD`.
   With a *finite* object the zero is the object plane and no offset applies.
4. **Unit — millimetres**, the lens geometry unit. Scaling the prescription by
   10 scales `opd` by exactly 10. Optiland's own wavefront code independently
   converts with `(wavelength * 1e-3)` (µm → mm) when dividing an `opd`
   difference to obtain waves.

**M1's `opd = 12` for a 10 mm separation is fully explained:** the probe set
`EPD = 2.0`, so the aimed launch plane sat at `z = -2 mm` and the ray
accumulated `|2·1| + |10·1| = 12`. The value was correct; only the reference
plane was unknown. Residual after the explanation: `0.0`.

### Two traps this uncovered

**Optiland's own wavefront sign is the reverse of L1-RAY-01's.**
`wavefront/strategy.py` reports `opd_wv = (opd_ref - opd) / (wavelength * 1e-3)`
with `opd_ref` from the chief ray — that is *chief minus ray*. L1-RAY-01
declared *ray minus chief* for its evaluator (see the CHE-17 note below). The
two differ by an overall sign, and a consumer that mixes them conjugates the
wavefront. Neither is wrong; they must not be combined without a deliberate
negation.

**`surface_type="paraxial"` is not an admissible OPL source.** A plane wave
through a perfect lens must reach the focus with equal optical path at every
pupil height. The paraxial interaction model subtracts `(x²+y²)/(2f)`, which
is exactly the paraxial excess of `sqrt(f²+h²)` — but it also sets
`rays.N = copysign(1, N)` and leaves the direction un-normalized, so the
following propagation adds the *axial* distance rather than the Euclidean one.
The intended cancellation never happens and the subtraction survives in full:
measured OPL at the focus is `f - h²/(2f)`, matching the bare lens term to
`1e-12`. At `f = 50 mm`, `h = 6 mm` that is `0.36 mm`, about **655 waves** at
550 nm — a defocus, not a rounding error. A real refractive singlet at the
same heights spreads by `< 0.01 mm` and scales as `h⁴`, i.e. physical
spherical aberration. Any system whose wavefront is handed to a coupler must
therefore be built from real refractive surfaces.

This does **not** invalidate L1-RAY-01's paraxial case, which gated centroids
and spot sizes rather than OPL.

## CHE-13 standalone ray-state boundary

`OptilandRayRequest` selects exactly the `ReverseTelephoto` prescription,
NumPy backend, CPU device, float64 dtype, wavelength in um, normalized field
coordinates, hexapolar pupil-sampling request, output directory, and recorded
seed. The hexapolar sampler is deterministic and does not consume an exposed
random seed; the seed is retained for the shared M1 provenance contract.

The saved `rays.npz` contains flat, equal-length arrays `x_m`, `y_m`, `z_m`,
`L`, `M`, `N`, `intensity`, `wavelength_m`, `opd_native`, and `survived`.
Coordinates and wavelength are SI. `(L,M,N)` are dimensionless direction
cosines in a right-handed Cartesian frame with nominal propagation along
`+z`. The reference plane is the final traced image surface (surface 14 for
the pinned sample), whose axial coordinate is recorded in metres. Every
exported row is a surviving ray; Optiland does not expose rejected input
candidates through `RealRays`, so pre-filter invalid/vignetted counts are not
invented. The direction unit-norm tolerance is `1e-12` for the float64 NumPy
baseline.

`intensity` is the raw real Optiland ray intensity/weight. It is neither a
complex field amplitude nor normalized power. Polarization and coherence are
missing. `opd_native` preserves the solver value, with reference and sign both
explicitly `unverified`.

CHE-17 narrows that final caveat for explicitly constructed benchmark
surfaces only. `L1-RAY-01` verifies Optiland's accumulated `RealRays.opd`
against closed-form free-space and axial catalog-lens optical paths. It then
defines comparison OPD by subtracting the pupil-height-zero chief ray within
each field, with sign `ray minus chief`. This verified evaluator convention
must not be projected onto bundled sample systems whose internal construction
and OPD reference have not been independently audited.
