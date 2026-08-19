# Optiland failure guide

Real errors and surprises hit while building this knowledge pack
(2026-07-30), with repairs. Add to this file rather than silently working
around a new one.

## `pip install optiland` does not give you gradients or GPU support

**Symptom:** code that assumes `derivative.mode: native_autodiff` (as this
project's registry currently states for `M_RAY_OPTILAND`) just... doesn't
differentiate, or `optiland.backend.supports_gradients` reads `False`.

**Cause:** torch is not a declared dependency of the `optiland` PyPI
package (confirmed via `importlib.metadata.distribution('optiland').requires`
— matplotlib, numba, numpy, pandas, pyyaml, requests, scipy, seaborn,
tabulate, typing-extensions, vtk; no torch). The default backend is NumPy,
which has no autodiff.

**Fix:** install `torch` separately (already done in
`docker/requirements.txt` / `docker/Dockerfile`, from the CPU-only wheel
index) and call `optiland.backend.set_backend('torch')` before building or
tracing the lens you want gradients through.

## `TypeError: 'bool' object is not callable`

**Symptom:** raised by `be.supports_gpu()` or `be.supports_gradients()`.

**Cause:** these are plain module-level `bool` attributes, not functions,
despite reading like predicates.

**Fix:** use `be.supports_gpu` / `be.supports_gradients` directly, no
parentheses.

## `DeprecationWarning: Optic.surface_group is deprecated; use Optic.surfaces instead.`

**Symptom:** a warning (not an error) on `lens.surface_group` access.

**Cause:** real, observed API migration in progress within `0.6.0` itself —
the old name still works but is on its way out.

**Fix:** use `Optic.surfaces` in any new code. Treat the eventual removal
of `surface_group` as a staleness signal for this knowledge pack.

## Repository URL redirects (not broken, just not canonical)

**Symptom:** none functionally — `https://github.com/HarrisonKramer/optiland`
still works.

**Cause:** the project moved to a dedicated `optiland` GitHub org
(`https://github.com/optiland/optiland`), confirmed both by an HTTP 301
redirect and by the PyPI package's own `Project-URL` metadata.

**Fix:** cite the new URL in new documentation; the old one is fine to
follow but is not the canonical source of truth going forward.

## Ray count after `trace()` does not equal the requested `num_rays`

**Symptom:** `lens.trace(..., num_rays=16)` returns a `RealRays` object
with `.x.shape == (817,)`, not `(16,)`.

**Cause:** `num_rays` controls a pupil-sampling density parameter, not a
literal output count; aperture/vignetting/pupil geometry determines how
many rays actually survive tracing.

**Fix:** never hardcode an expected output array length from `num_rays`;
read `.x.shape` (or equivalent) from the returned object instead.

## Unverified: behavior when torch is requested but not installed

**Status:** NOT independently tested in this pass. This container always
has torch installed, so `optiland.backend.set_backend('torch')` always
succeeds here. Do not assume a specific failure mode (immediate
`ImportError` at `set_backend()` time vs. a deferred failure only when a
torch-specific operation actually runs) without testing a torch-less
environment directly. An attempted test using a Python import-blocker
(`sys.meta_path` with a legacy `find_module`/`load_module` finder) was
tried and found to **not work** on Python 3.12 (the legacy finder protocol
is silently ignored; `import torch` still succeeded) — that approach is a
dead end, not a valid negative result. A real test would need a second
Docker image built without torch installed.

## Python version

Optiland's `Requires-Python` is `>=3.10` per its wheel metadata — looser
than chromatix's `>=3.12` or sax's `>=3.11`. Not a source of conflict in
this repository's shared `python:3.12-slim` image, but worth knowing if
optiland is ever split into its own lighter environment.

## CHE-13 standalone structured failures

`OptilandAdapter.run_standalone()` never returns fabricated ray arrays after a
failure. Malformed prescriptions, non-CPU/non-NumPy/gradient requests, and
invalid scalar inputs return `OPTILAND_INVALID_BASELINE_REQUEST`. A missing
package returns `OPTILAND_DEPENDENCY_UNAVAILABLE`. Empty, unequal-shape,
non-finite, or non-unit float64 NumPy ray output returns
`OPTILAND_INVALID_OR_EMPTY_OUTPUT`. Each result records the failure stage and
exception type, while `arrays_path` and scientific metrics remain absent.

## L1-RAY-01 benchmark blockers

`run_benchmark.py` writes a `status: blocked` result with code
`L1_RAY_01_EXECUTION_FAILED` if the pinned dependency, prescription
construction, trace, artifact generation, or provenance stage fails. It does
not substitute sample-lens output or analytic values for missing solver rays.
The evaluator also fails if any forbidden Chromatix or coupler module appears
in the ray-only process.

---

# CHE-57 (PB6) failure guide additions

Every entry below was hit while reproducing the official 41-tutorial Optiland
scope against the pinned `0.6.0`. Each has an executable reproduction under
`knowledge/solvers/optiland/tutorials/` and recorded evidence in `expected/`.

## `Optic.draw3D()` never returns

**Symptom:** the process prints three VTK warnings and then hangs forever:

```
vtkXOpenGLRenderWindow: bad X server connection. DISPLAY=
vtkOpenGLRenderWindow: Failed to load EGL! ...
vtkOSOpenGLRenderWindow: libOSMesa not found ...
```

**Cause:** `draw3D` builds a VTK render window. The `agent_solver` container has
no X server, no EGL and no OSMesa, and VTK blocks rather than failing.

**Fix:** never call `draw3D` (or `visualization.system.optic_viewer_3d`) from a
probe, test or script. `Optic.draw()` works fine under `MPLBACKEND=Agg`. Three
official tutorials (`your-first-optical-system`, `non-rotationally-symmetric`,
`extending-surfaces`) call it; the reproductions skip it deliberately.

**How to detect earlier:** any Optiland call that touches `optiland.visualization.system.optic_viewer_3d`.

## The whole `Optic.add_*` API is deprecated for removal in 0.7.0

**Symptom:** `DeprecationWarning: Optic.add_surface is deprecated and will be
removed in v0.7.0; use optic.surfaces.add() instead.` — and the same for
`add_field`, `add_wavelength`, `set_field_type`, `set_thickness`,
`update_paraxial`, `set_polarization` and `Optic.surface_group`.

**Cause:** a real in-progress migration to sub-object APIs. Every official
tutorial still uses the old names.

**Fix / mapping:**

| Deprecated | Replacement |
|---|---|
| `Optic.add_surface(...)` | `Optic.surfaces.add(...)` |
| `Optic.add_field(...)` | `Optic.fields.add(...)` |
| `Optic.set_field_type(...)` | `Optic.fields.set_type(...)` |
| `Optic.add_wavelength(...)` | `Optic.wavelengths.add(...)` |
| `Optic.set_thickness(...)` | `Optic.updater.set_thickness(...)` |
| `Optic.update_paraxial()` | `Optic.updater.update_paraxial()` |
| `Optic.set_polarization(...)` | `Optic.updater.set_polarization(...)` |
| `Optic.surface_group` | `Optic.surfaces` (the same object) |

`Optic.set_aperture` is **not** deprecated. Verified: the deprecated and modern
paths produce bit-identical traces (`t01`).

## The torch backend silently runs in float32

**Symptom:** an autodiff gradient that agrees with a finite difference only to
~1e-3, and *worse* as the step shrinks.

**Cause:** `be.get_precision()` is `32` by default. A `dtype=torch.float64`
parameter tensor is not enough — the lens is built with `be.array`, which follows
the global precision.

**Fix:** `be.set_precision('float64')` **before** constructing the lens. Then the
gradient agrees to 6e-9 and converges as O(eps^2). Full table in
`conventions.md`.

**How to detect earlier:** print `be.get_precision()` and the `.dtype` of the
traced objective.

## `OptimizerGeneric` reports `success=True` on a design it made worse

**Symptom:** `optimize()` returns an `OptimizeResult` with `success=True` and
`message='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'`, but
`problem.sum_squared()` is larger than before the call.

**Cause:** L-BFGS-B on a finite-difference gradient of a non-smooth merit
function. On a freeform tilted-mirror system rays vignette in and out as
surfaces move, so the gradient is unreliable and the line search accepts a bad
step. Measured 2.750 -> 5.417 after 4 iterations and 924 function evaluations;
`maxiter` 60, 200 and 500 all stop at the same point, and it is inside the
declared bounds.

**Fix:** always record `problem.sum_squared()` before and after, and revert if it
grew. `res.success` is not evidence.

**Reproduction:** `tutorials/t35_three_mirror_anastigmat.py`.

## `ThinFilmCoating` takes NANOMETRES; `ThinFilmStack.add_layer` takes MICROMETRES

**Symptom:** an AR coating that does nothing — mean reflectance 4.23% instead of
0.48% — with no error and no warning.

**Cause:** `ThinFilmStack.add_layer(material, thickness_um, name)` is in
micrometres, but `ThinFilmCoating(pre, post, layers)` forwards to
`add_layer_nm`, i.e. nanometres. Transcribing a design's um numbers into the
coating constructor builds a 1000x-too-thin stack.

Related: the official tutorial passes a `ThinFilmStack` as the third argument,
which the pinned version rejects with
`TypeError: 'ThinFilmStack' object is not iterable`.

**Fix:** `ThinFilmCoating(air, glass, [(mat, thickness_um * 1000.0, name), ...])`,
then cross-check `coating.stack.compute_rtRTA(...)` against a separately built
`ThinFilmStack`.

**Reproduction:** `tutorials/t07_anti_reflective_coating.py`.

## Sharing one `ThinFilmCoating` across interfaces gives transmittance > 1

**Symptom:** `rays.i` in [3.56, 3.67] — every ray "transmitting" 360%.

**Cause:** a stack declares its own incident and substrate materials. Attached to
a glass->air interface, the air->glass direction is evaluated anyway. The
official tutorial attaches one coating object to all four surfaces of a cemented
doublet, so two of them are reversed.

**Fix:** one coating per interface, each declaring that interface's actual media.
That restores `rays.i <= 1`.

**How to detect earlier:** assert `0 <= rays.i <= 1` after any coated trace.

## `Material(name, reference=...)` announces `k = 0` where no caller can see it

**Symptom:** `WARNING: No extinction coefficient data found for Li-o.yml.
Assuming it is 0.` appears in the terminal, but
`warnings.catch_warnings(record=True)` records nothing and
`contextlib.redirect_stdout` / `redirect_stderr` capture nothing.

**Cause:** the message is emitted below the Python stream objects.

**Fix:** none available programmatically. Treat any `Material(..., reference=...)`
lookup as potentially lossless-by-assumption and say so in provenance. Every
candidate material in the needle-synthesis tutorial triggers it.

## `AbbeMaterialE` does not reproduce its own Abbe number

**Symptom:** `AbbeMaterialE(n=1.5, abbe=65)` yields `V_e = 40.46` when measured at
the e/F'/C' lines it is defined against.

**Cause:** the model's LASSO-fitted Buchdahl coefficients are simply inaccurate.
The error is systematic: recovered/requested is 0.83 at `V_e = 20` falling to
0.57 at `V_e = 80`. Against real N-BK7 it errs by 1.4e-2 in index at 0.42 um.

**Fix:** use `AbbeMaterial(n, abbe, model='buchdahl')`, which is self-consistent
to 0.2% and matches N-BK7 to 1.4e-4. Also avoid the 0.6.0 *default*
`model='polynomial'`, which misses its own defining numbers by ~1.6% and emits a
`FutureWarning` saying the default changes in 0.7.0 — always pass `model=`.

**Reproduction:** `tutorials/t04_material_database.py`.

## `AttributeError: 'RandomDistribution' object has no attribute 'x'`

**Symptom:** raised by `Optic.trace(..., distribution=RandomDistribution(seed=7))`.

**Cause:** `Optic.trace` calls `distribution.generate_points(num_rays)` **only**
when `distribution` is a string. A `BaseDistribution` *instance* is used as-is,
and `num_rays` is ignored.

**Fix:** call `generate_points(...)` on the instance yourself before passing it.

## Three different RNGs, three different seeding rules

| Source | Seedable | How |
|---|---|---|
| `distribution.RandomDistribution` | yes | constructor `seed=`. `Optic.trace(distribution="random")` builds an **unseeded** one, so two identical calls return different rays |
| `optimization.DifferentialEvolution` | yes, indirectly | no `seed` parameter, but SciPy falls back to NumPy's global RNG — `np.random.seed(...)` before `optimize()` makes it bit-reproducible |
| `tolerancing.perturbation.DistributionSampler` | yes, **only** explicitly | builds its own `be.default_rng(seed)`; `seed=None` ignores `np.random.seed` entirely |
| `optiland.scatter.GaussianBSDF` / `LambertianBSDF` | **no** | numba-compiled (`njit`/`prange`) with an RNG unreachable from NumPy. Two identical traces differ by ~1% in RMS spot radius |

**Consequence:** a BSDF scattering result **cannot** be frozen as a bit-exact
repository fixture. `tutorials/t21_surface_roughness_scattering.py` declares a
statistical tolerance instead and records only stable quantities.

## `GlassExpert.run()` removes the material variables from the problem

**Symptom:** `len(problem.variables)` drops from 25 to 19 across the call.

**Cause:** the six `material` variables are consumed once glasses are chosen.

**Fix:** capture the variable count and the glass names *before* the call if you
need them; do not reuse the problem expecting a mixed discrete/continuous
formulation afterwards.

## A hand-written `_surface_normal` is not validated by anything Optiland does

**Symptom:** a custom `NewtonRaphsonGeometry` subclass traces without error and
gives wrong refraction. The **official** "Custom Surface Types" tutorial has
this bug: for the sag term `a*r` it writes `d/dx = a*x/r2` where the correct
expression is `a*x/r`. Measured disagreement against a central difference of its
own `sag()` is 0.52 in direction cosine.

**Why it hides:** the wrong gradient still normalises to a unit vector, so a
`|n| == 1` assertion passes. The trace does not raise. Only the physics changes.

**Fix:** always compare `_surface_normal(x, y)` against a central difference of
`sag`:

```python
dzdx = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
dzdy = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
mag = sqrt(dzdx**2 + dzdy**2 + 1)
expected = (dzdx / mag, dzdy / mag, -1 / mag)
```

Also note `NewtonRaphsonGeometry` is **abstract** (`sag` and `_surface_normal`),
so it cannot be instantiated directly; `StandardGeometry` is the concrete conic.

**Reproduction:** `tutorials/t39_custom_surface_types.py`.

## A custom sag term can make the vertex normal undefined

**Symptom:** `ZeroDivisionError: float division by zero` from
`_surface_normal(0.0, 0.0)` with Python floats, or a silent `[nan, nan, nan]`
with numpy arrays — and then one NaN ray out of the traced bundle.

**Cause:** `a*r` is non-differentiable at `r = 0` (a conical cusp for `a != 0`),
so both `a*x/r2` and the correct `a*x/r` are 0/0 there. The official tutorial's
`warnings.catch_warnings` suppresses only the numpy path.

**Fix:** special-case `r = 0`, or do not use an `r`-linear sag term. Whether the
failure is loud depends on the **input type**, which is worth knowing.

## A custom `BaseCoating` is completely unchecked

**Symptom:** none — that is the problem. A coating that assigns `rays.i = 1.5` or
`rays.i = -0.25` is accepted with no clamp and no warning.

Two more specifics worth knowing when writing one:

- The `rays` handed to `transmit()` carry the **coated surface's** coordinates,
  not the image plane's (verified to 1.1e-16), and `rays.w` is the wavelength in
  **micrometres**.
- `rays.i = ...` **overwrites**; a composable coating must use `rays.i *= ...`.
  Two assigning coatings in series leave only the last one's value.
- The final `rays.i` is the coating's value times a further ray-dependent factor
  from the *uncoated* downstream surfaces (~3e-3). Validate a coating against a
  **matched uncoated trace**, never against an absolute number.

**Reproduction:** `tutorials/t40_custom_coating_types.py`.

## `.zmx` round trips are lossy; `.json` round trips are exact

**Symptom:** `save_zemax_file` -> `load_zemax_file` returns R1 = 25.84000000165376
for a prescribed 25.84, and the trace shifts by ~1e-9.

**Cause:** Zemax files record **curvature**, so the radius is reconstructed
through a reciprocal at text precision.

**Fix:** use `save_optiland_file`/`load_optiland_file` (JSON) when exactness
matters — that round trip reproduces `Optic.to_dict()` and the whole trace
element-wise. Reserve `.zmx` for interchange.

Also: neither catalog tutorial's artifact is reachable. The Edmund tutorial
expects a manual website download, and the Thorlabs URL
(`thorlabs.com/_sd.cfm?fileName=20565-S03.zmx&...`) answers with a 1313-byte
HTML page, so `load_zemax_file(url)` raises `ValueError: Failed to read Zemax
file.` even with outbound network access. **Genuine vendor-authored `.zmx`
parsing therefore remains unverified in this repository.**

## Analysis result shapes are not uniform

There is no single convention for `.data`, and guessing wrong raises deep inside
numpy:

| Class | `.data` / accessor shape |
|---|---|
| `SpotDiagram.rms_spot_radius()` / `geometric_spot_radius()` | nested `[field][wavelength]` |
| `Distortion.data` | list of one 128-point curve per wavelength, in per-cent |
| `FieldCurvature.data` | list of `(2, 128)` arrays |
| `RmsWavefrontErrorVsField.data` | dict keyed by `((Hx, Hy), wavelength)` -> **`WavefrontData`**, not a scalar |
| `wavefront.OPD.data` | dict keyed by `((Hx, Hy), wavelength)`; use `get_data(field, wl)` |
| `FFTMTF.freq` / `.mtf` | `(num_fields, 128)` / per-field `[sagittal, tangential]` |
| `GeometricMTF.freq` / `.mtf` | flat `(256,)` / per-field `[sagittal, tangential]`, ending at the incoherent cutoff |
| `Tolerancing` / `MonteCarlo.get_results()` | pandas `DataFrame`; `perturbation_type` is a **display label** ("Radius of Curvature, Surface 1"), not the `add_perturbation` keyword, and operand columns are prefixed by declaration order ("0: f2") |

## A surface keyword that does not belong to the surface type is dropped

**Symptom:** a prescription traces successfully and gives the wrong answer.

`GeometryFactory.create` keeps only the kwargs that are dataclass fields of the
geometry config for the requested `surface_type`, so `coefficients=[...]` on a
`standard` surface, or any misspelled keyword, is discarded with no error and no
warning. The resulting geometry has no attribute to inspect afterwards
(`geometry_has_coefficients_attribute` is `False`).

**Fix:** validate the prescription before construction and assemble the keyword
set per surface type explicitly. In this repository that is
`core/optical_system.py` + `adapters/optiland_builder.py`; see
`probes/system_construction_probe.py` for the measurement.

## `Material('SK1')` silently returns SK16

**Symptom:** a system traces with a glass you did not ask for.

`Material.__init__` defaults to `robust_search=True`. The lookup is a substring
filter plus a Levenshtein ranking, and the best row wins even when nothing matched
exactly; the only notice is a `print` ("Warning: No exact matches found for
material SK1"), which is not a `warning` and cannot be caught with
`warnings.catch_warnings`.

**Fix:** require `material_data['similarity_score'] == 0`, and pin the resolved
`material_data['filename']` if the glass matters. Do not compare against
`material_data['name']` alone -- `N-BK7` legitimately resolves to a row named
`N-BK7 (SCHOTT)`. `robust_search=False` is a blunter alternative: it raises when
more than one row survives the *substring* filter, which is 7 rows for `SK15`
even though only one is exact.

## `ValueError: No matches found for material <name>`

**Symptom:** a bare `ValueError` from deep inside `Material._retrieve_file`
crossing an adapter boundary.

Raised when the substring filter returns nothing at all (e.g. `N-BK7X`). It is
not a solver failure -- it is an unresolvable prescription -- so it should be
translated into a structured capability/validation error at the boundary rather
than propagated as a bare `ValueError`.

## There is no aspheric grating in 0.6.0

**Symptom:** an aspheric grating prescription traces, and the asphere is ignored.

`surface_type='grating'` selects `PlaneGrating` (infinite base radius) or
`StandardGratingGeometry` (finite), and `SurfaceFactory` derives
`DiffractiveInteractionModel` from the same string. There is no even-asphere
grating geometry, so a request for one either loses the grating (if the surface
type stays `even_asphere`) or loses the aspheric terms (if it becomes `grating`).
Refuse it; do not pick one.

## A custom operand's `input_data` keys are its callable's parameter names

**Symptom:** `TypeError: spot_ellipse_ratio() got an unexpected keyword argument 'optic'`.

**Cause:** `input_data` is splatted as keyword arguments. Built-in operands take
`{"optic": ...}`; a user function whose parameter is named `lens` needs
`{"lens": ...}`.

**Fix:** match the key names to the callable's signature. Also note
`operand_registry.register(name, fn)` raises
`ValueError: Operand "<name>" is already registered.` without `overwrite=True`,
which is what stops a custom operand silently shadowing a built-in.
