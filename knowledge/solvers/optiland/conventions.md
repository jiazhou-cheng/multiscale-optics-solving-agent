# Optiland conventions (pinned version `0.6.0`)

Every fact below was either read from `optiland.backend.__doc__` /
`inspect` on the installed package, or observed directly by running
`benchmarks/probes/optiland/*.py` inside the `agent_solver`
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

## Torch backend precision defaults to float32 (CHE-57)

**This resolves the open item `usage_notes.md` recorded as "root-causing why
the torch-backend gradient tolerance (1.11e-03) is looser than the JAX-based
solvers' tolerances in this repository".**

`optiland.backend` exposes `get_precision()` / `set_precision('float32'|'float64')`
and `set_device(str)` alongside `set_backend`. Under the torch backend
`get_precision()` returns **`32`** immediately after `set_backend('torch')`:
single precision is the default, not float64.

`probes/gradient_probe.py` passes a `dtype=torch.float64` parameter tensor, but the
lens it is traced through is built with `be.array`, which follows the *global*
precision. The traced objective therefore comes back `torch.float32`, and the
finite-difference reference is float32 too. Measured at the probe's own operating
point (`ReverseTelephoto`, surface-1 radius, `mean(x^2+y^2)`, 64 rays):

| precision | eps=1e-3 | eps=1e-4 | eps=1e-5 |
|---|---|---|---|
| float32 | 1.32e-04 | **1.11e-03** | 3.26e-02 |
| float64 | 6.24e-05 | 6.24e-07 | 6.28e-09 |

The 1.11e-03 was **finite-difference cancellation noise, not autodiff error**: at
float32 the relative error *grows* as the step shrinks. Under
`be.set_precision('float64')` the directional derivative agrees with a centered
difference to 6.3e-9 and the error falls quadratically with step size, which is the
behaviour a correct reverse-mode gradient must show.

Consequences for this project:

- Any gradient claim through Optiland must declare the precision. `set_precision`
  is as load-bearing as `set_backend`, and it is **process-global** in the same way.
- A float64 parameter tensor alone is not enough; the whole lens must be built after
  `set_precision('float64')`.
- `be.grad_mode` (`enable`/`disable`/`temporary_enable`/`requires_grad`) is a
  separate global switch again.

Evidence: `tests_tutorial/cases/optiland/t10_differentiable_ray_tracing.py`
and its recorded output.

## Paraxial cardinal points use two different reference planes (CHE-57)

`Optic.paraxial` reports its cardinal points against **different origins**, and
mixing them up produces a plausible-looking wrong answer:

- `P1()`, `F1()`, `N1()` are relative to **surface 1** (the first optical surface).
- `P2()`, `F2()`, `N2()` are relative to the **image surface**. `F2()` is therefore
  the residual defocus of the *placed* image plane, not a position in prescription z.

With that, the Gaussian conjugate equation `1/s' + 1/s = 1/f` holds **exactly** --
measured max `|f*(1/s' + 1/s - 1/f)| = 5.0e-16` over 33 object positions -- when `s`
is measured from `P1` and `s'` from `P2`. Reading either from a surface vertex
instead gives a 16-61% error, and it is also what makes a naive
`BFL = F2() - z(last surface)` computation wrong.

Back focal length from the accessors is `BFL = z_image + F2() - z_last_lens_surface`;
verified against the Edmund #45-362 datasheet value of 47.87 mm.

Evidence: `tutorials/t24_thorlabs_catalogue.py`, `tutorials/t08_edmund_optics_catalogue.py`.

## Even-asphere coefficients start at r^2, not r^4 (CHE-57)

`optiland.geometries.even_asphere.EvenAsphere.sag` is

```python
r2 = x**2 + y**2
z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
for i, Ci in enumerate(self.coefficients):
    z = z + Ci * r2 ** (i + 1)
```

The loop index starts at `i = 0`, so **`coefficients[0]` multiplies r^2**. That is
*not* the Zemax/CODE V even-asphere convention, where the polynomial starts at r^4
because the r^2 term is degenerate with the base curvature. Transcribing a vendor
prescription term-for-term into Optiland shifts every coefficient by one order; on
the tutorial's own coefficient list the two readings differ by up to 13 mm of sag at
a 10 mm semi-aperture. Verified against both closed forms written out independently.

Evidence: `tutorials/t12_raytracing_aspheres.py`.

## Wavefront OPD vs. the raw `RealRays.opd` accumulator (CHE-57)

These are two different quantities with two different references and two different
units, and the repository already depends on knowing which is which:

| | `RealRays.opd` | `wavefront.OPD` |
|---|---|---|
| quantity | absolute accumulated optical path | wavefront error |
| reference | the ray launch state (see above) | the **chief ray** -- pupil-centre value is exactly 0 |
| unit | millimetres | **waves** at the requested wavelength |
| magnitude on `EyepieceErfle` on axis | 335129 waves | 0.2165 waves peak-to-valley |

Two further specifics:

- `OPD.rms()` is `sqrt(mean(opd**2))` over rays with `intensity > 0` and **leaves
  piston in**. It is therefore *not* the conventional piston-removed RMS wavefront
  error: on axis it reads 0.1337 waves where the piston-removed value is 0.0664. A
  Marechal or Strehl estimate built on `OPD.rms()` is wrong.
- `WavefrontData.pupil_x` / `pupil_y` are **physical millimetres** on the reference
  sphere, not normalised `Px`/`Py`.

Evidence: `tutorials/t16_opd_calculations.py`, `tutorials/t25_psf_and_mtf.py`.

## Zernike conventions: only `standard` and `noll` are orthonormal (CHE-57)

`wavefront.ZernikeOPD(zernike_type=...)` supports `standard`, `fringe` and `noll`.
All three agree on the piston term to 3e-12 (the mean of a wavefront is
basis-independent), and `standard` and `noll` give *identical* `sqrt(sum_{k>=1} c_k^2)`
-- they differ only in term ordering, and both are RMS-normalised, so that
quadrature sum **is** the piston-removed RMS wavefront error.

`fringe` is normalised to unit **peak**, not unit RMS. The same quadrature sum gives
0.0769 waves where standard/noll give 0.0444: a factor of 1.73. Computing an RMS
wavefront error from Fringe coefficients the way one legitimately can from Standard
or Noll coefficients is a silent 1.7x error.

Evidence: `tutorials/t26_zernike_decomposition.py`.

## MTF and PSF grid conventions (CHE-57)

- `GeometricMTF.freq` ends at exactly the incoherent cutoff `1/(lambda * F/#)` and
  equals its own `cutoff_freq` attribute. `FFTMTF`'s grid extends to **2.02x** that.
  The two curves are therefore not comparable index-by-index.
- `FFTMTF.freq` is `(num_fields, 128)` and `FFTMTF.mtf` is one
  `[sagittal, tangential]` pair per field -- not a flat curve.
- `FFTPSF` and `HuygensPSF` return **different grid sizes** for the same `num_points`
  request (256x256 vs 128x128 on `CookeTriplet`) over different physical extents.
  Compare them by Strehl ratio, or interpolate onto a common physical grid; a
  pixelwise or pixel-radius comparison is meaningless. Their Strehl ratios agree to
  21% at the edge field.
- `FFTPSF.strehl_ratio()` agrees with the Marechal estimate `exp(-(2 pi sigma)^2)`
  to 7% on axis when `sigma` is the *piston-removed* RMS from `wavefront.OPD`.

Evidence: `tutorials/t25_psf_and_mtf.py`.

## Third-order aberration coefficients are per surface (CHE-57)

`Optic.aberrations.seidels()` returns the five primary Seidel sums, but the twelve
named accessors (`TSC`, `SC`, `CC`, `TCC`, `TAC`, `AC`, `TPC`, `PC`, `DC`, `TAchC`,
`LchC`, `TchC`) return **per-surface arrays**; the system aberration is their sum.

Relationships verified against third-order theory on `TripletTelescopeObjective`:

- `TCC == 3 * CC` exactly (tangential coma is three times sagittal coma), per surface.
- Every longitudinal coefficient is its transverse partner over `-u'`, the final
  marginal ray slope from `paraxial.marginal_ray()`: `SC/TSC`, `AC/TAC`, `PC/TPC` and
  `LchC/TAchC` all equal `5.600000` to 8.9e-16, across four physically unrelated
  aberration types.
- `sum(PC) == -h_img^2 * P / 2`, where `P = sum (n' - n)/(n n' R)` is the Petzval sum
  read off the prescription without Optiland, and `h_img` is the paraxial chief-ray
  image height. This pins what `PC` *means* (edge-field longitudinal sag).

Evidence: `tutorials/t14_first_third_order_aberrations.py`.

## `rays.i` is an amplitude-squared transmission, not an intensity transmittance (CHE-57)

After a *coated* refraction, `RealRays.i` carries `|t|^2`, **not** the intensity
transmittance `T = 1 - R`. The two differ by the radiance factor
`n2 cos(th2) / (n1 cos(th1))`, which cancels only when the ray returns to its
original medium. Verified exactly on plane air/N-BK7 interfaces:

- one interface, image plane left inside the glass: `i = (2/(1+n))^2 = 0.630621`
- two interfaces, ray back in air: `i = (1 - R)^2 = 0.917021`

Three further specifics:

- An **uncoated** surface applies no Fresnel loss, but `rays.i` is still not exactly
  1: an uncoated cemented doublet gives 0.999784 with a ray-dependent spread of
  ~3e-5. A coating can only be validated against a **matched uncoated trace**.
- `coatings.SimpleCoating(transmittance=t)` is a scalar intensity factor applied
  multiplicatively per surface, with no angle or wavelength dependence: the ratio to
  the uncoated baseline equals the declared product to 1e-16.
- A `ThinFilmCoating` carries its **own** declared incident/substrate materials. The
  same object attached to interfaces whose media are reversed evaluates the wrong
  direction and yields `rays.i` ~ 3.6, i.e. a transmittance above unity, silently.

Evidence: `tutorials/t07_anti_reflective_coating.py`, `tutorials/t18_introduction_to_coatings.py`,
`tutorials/t40_custom_coating_types.py`.

## Grating parameters use the wavelength's unit, not the geometry's (CHE-56)

Every length in a prescription is millimetres (see "Units" below) -- except
`grating_period`, which is **micrometres**, because
`DiffractiveInteractionModel` and `RealRays.gratingdiffract` only ever use it as
the ratio `m * rays.w / d`, and `rays.w` is micrometres.

Measured at normal incidence on a plane transmission grating in air, order 1,
lambda = 0.55 um (`probes/system_construction_probe.py`):

| `grating_period` | measured `sin(theta)` | `m*lambda/d` with d in um | with d in mm |
| -- | -- | -- | -- |
| 2.0 | 0.275 | 0.275 | 2.75e-4 |
| 4.0 | 0.1375 | 0.1375 | 1.375e-4 |
| 8.0 | 0.06875 | 0.06875 | 6.875e-5 |

Exact agreement, to 0.0 in float64, across a 4x sweep. A millimetre reading
would be wrong by 1000x and no single measurement would reveal it.

`groove_orientation_angle` is **radians**: `PlaneGrating.grating_vector` returns
`(-sin(angle), cos(angle), 0)`, so at 0 the grating vector is +y and the
diffracted order is deviated in y, and at pi/2 the deviation moves to -x with the
same magnitude. Read as degrees, pi/2 would be 1.57 deg and the deviation would
still be essentially in y.

A grating is a *geometry class*, not an attribute of one:
`surface_type='grating'` yields `PlaneGrating` when the base radius is infinite
and `StandardGratingGeometry` otherwise, and `SurfaceFactory.create_surface` then
selects `DiffractiveInteractionModel` from the same string. There is therefore no
aspheric grating in this version, and asking for one must be refused rather than
downgraded to its base conic -- which would trace without complaint.

## Unrecognized surface keywords are silently discarded (CHE-56)

`GeometryFactory.create` filters `**kwargs` down to the dataclass fields of the
geometry config it selected:

```python
config_fields = {f.name for f in fields(config_cls)}
filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```

So `surfaces.add(index=1, radius=10.0, coefficients=[...])` on a `standard`
surface silently builds a plain sphere, and a misspelled keyword builds a
different optical system with no error, no warning, and no attribute left behind
to notice it by (`probes/system_construction_probe.py`,
`surface_kwargs_are_silently_filtered`). Validate a prescription before it
reaches `surfaces.add`; this repository does that in
`core/optical_system.py` (`extra="forbid"`) and
`solvers/optiland/builder.py` (explicit per-type keyword sets).

## A bare glass name is a fuzzy query (CHE-56)

`Material(name)` is not a lookup. `_find_material_matches` filters the catalog by
substring over `category_name`, `name` and `filename_no_ext`, scores the
survivors by Levenshtein distance, sorts, and `_retrieve_file` takes the best
row. `robust_search` defaults to `True`, so an inexact best match is returned
rather than refused, and only a printed line (not a warning) says so.

Measured (`probes/system_construction_probe.py`):

| requested | rows surviving the filter | exact (score 0) | resolved file |
| -- | -- | -- | -- |
| `N-SK10` | 1 | 1 | `glass/schott/N-SK10.yml` |
| `SK15` | 7 | 1 | `glass/hikari/SK15.yml` |
| `BASF2` | 4 | 1 | `glass/hikari/BASF2.yml` |
| `FK3` | 1 | 1 | `glass/schott/FK3.yml` |
| `SF15` + `hikari` | 3 | 1 | `glass/hikari/SF15.yml` |
| `N-LAK12` | 1 | 1 | `glass/schott/N-LAK12.yml` |

Two consequences. First, the manufacturer is not implied by the name: `SK15`
resolves to HIKARI while `N-SK10` resolves to SCHOTT. Second, a near-miss still
resolves -- `SK1` returns `SK16`. The selected row carries its own
`similarity_score`, and `== 0` is Optiland's own exactness criterion; compare
against `material_data['name']` alone and you will reject legitimate rows, since
`N-BK7` resolves to a row named `N-BK7 (SCHOTT)` whose `filename_no_ext` is what
matched. Record the resolved `filename` in the prescription if the glass matters.

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
`benchmarks/probes/engine_independence.py --engine ray` constructs planar
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
`opd_convention:` in `card.yaml`; the probe is
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

### The launch plane's orientation (CHE-41), and why point 3 was incomplete

Point 3 above located the launch plane and recorded the consequence of its
*position*: the OPL zero moves with the aperture. It said nothing about the
plane's *orientation*, and every case that established it was on axis, where a
plane perpendicular to z and a wavefront of the incoming bundle are the same
surface. They are not the same surface off axis.

`angle.py` computes one `z0` for the whole bundle (`z0 = be.full_like(Px, z)`) and
one direction, so the seeded surface is a **plane perpendicular to z** carrying a
tilted collimated bundle. Measured at `Hy = 0.2` on `ReverseTelephoto`: launch `z`
spread exactly `0.0` and launch direction spread `2.8e-17` across 3169 rays,
against the `tan(theta) · EPD = 0.031531 mm` spread a wavefront-seeded launch
would show. The launch direction is `(0, sin theta, cos theta)` to `1.4e-17`.

An accumulated path measured from that plane differs from one measured from a
wavefront by

    n_object * (d0 . r_launch)

which is **linear in the launch coordinate**: a constant on axis, and the entire
convergence tilt off axis. Omitting it leaves a pupil OPL that is a clean
converging sphere aimed at the *axis* whatever the field angle — on
`M3-REVERSE-TELEPHOTO` at `Hy = 0.2`, 209 µm from where the rays go, with a
0.072-wave-peak-to-valley residual against its own fitted sphere. That is the
worst possible failure mode: internally consistent, diffraction-limited-looking,
and wrong. It survived CHE-30, CHE-32 and CHE-33 because all three validated on
axis, and it was found by CHE-37 and fixed by CHE-41.

The term cannot be recovered from an exit-pupil export, because it is evaluated at
the launch coordinate and no object-space coordinate survives the export. The
adapter therefore regenerates the launch state through
`ray_tracer.ray_generator.generate_rays` over the same hexapolar distribution
`Optic.trace` builds, and exports the term as `object_space_reference_offset_m`. A
re-trace of the regenerated state reproduces `Optic.trace` exactly (max `|dx|`,
`|dy|`, `|d opd|` all `0.0`), which is what makes a per-ray term measured from it
admissible.

One consequence worth stating plainly, because it is what made the defect
invisible: `opd_is_relative_to_chief_ray: false` is **correct**, and on axis it is
also untestable in the way that matters. A genuinely chief-ray-referenced OPD and
a plane-referenced absolute OPL both predict a tilt-free pupil wavefront off axis;
they differ in whether `opd[chief]` is zero. Off axis it is `11051.3` waves, so the
flag stands — but the flag was never the question. The question was which surface
the accumulation starts from, and that is now declared
(`opd_reference_surface`, `opd_omits_incoming_wavefront_tilt`).

Evidence: `benchmarks/probes/optiland/off_axis_opd_reference.py` and its
recorded output, plus `benchmarks/probes/off_axis_handoff.py` for the
downstream consequence.

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
