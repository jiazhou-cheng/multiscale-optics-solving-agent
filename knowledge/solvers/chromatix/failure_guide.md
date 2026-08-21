# Chromatix failure guide

Real errors hit while building this knowledge pack (2026-07-30), with
repairs. Add to this file rather than silently working around a new one.

## `pip install chromatix` installs the wrong package

**Symptom:** installs successfully, but `import chromatix` gives a package
with no useful attributes (a 22-byte stub).

**Cause:** the PyPI name `chromatix` (version `0.0.1`, 2022-07-10) is an
unrelated namesquat. The real chromatix-team library is not published to
PyPI.

**Fix:** install from GitHub, pinned to a commit:
```
pip install "chromatix @ git+https://github.com/chromatix-team/chromatix.git@d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"
```
(already done in `docker/requirements-chromatix.txt`).

## `ModuleNotFoundError: No module named 'chromatix'` / venv creation fails

**Symptom:** `python3.10 -m venv .venv` fails with `ensurepip is not
available... apt install python3.10-venv`; separately, Chromatix's
`pyproject.toml` declares `requires-python = ">=3.12"`, which this host's
system Python (3.10/3.8) cannot satisfy at all.

**Fix:** don't fight the host Python. Use `./run.sh` (Docker, `python:3.12-slim`
base) instead of a host venv.

## `AttributeError: 'MonoSpectrum' object has no attribute 'squeeze'`

**Symptom:** raised when calling `compute_padding_transfer(..., field.spectrum.squeeze(), ...)`.

**Cause:** `compute_padding_transfer(height, wavelength, dx, z)` expects a
bare scalar wavelength, not the `Field.spectrum`/`Field.wavelength` object.

**Fix:** pass the scalar wavelength you originally used to build the field,
e.g. `compute_padding_transfer(shape[0], wavelength, dx, z)`.

## Large output arrays from `asm_propagate`

**Symptom:** an initially small field (e.g. 128x128) balloons to
thousands of pixels per side after `asm_propagate`, especially for a
point-source-like field with most of its energy concentrated near a
delta function and a large propagation distance relative to `dx`.

**Cause:** `compute_padding_transfer`/`compute_padding_transform` estimate
padding conservatively from wavelength/dx/z; a poorly-conditioned
combination (fine `dx`, long `z`) produces very large recommended padding.
This is real memory-and-compute-cost behavior, not a bug — see
`solver_card.yaml` `cost_scaling`.

**Fix:** check the returned pad width before committing to a shape in an
adapter's cost estimator; do not call the propagator blind on
user-controlled `z`/`dx` without a resource check (the repository resource-safety policy says
adapters must expose cost-estimation metadata).

## Structured diagnostics from the standalone wave baseline (CHE-14)

`ChromatixAdapter.run_standalone` never raises and never fabricates a field.
Every rejection comes back as `ChromatixWaveResult(status=FAILED,
failure=ChromatixWaveFailure(code, message, stage, exception_type))` with
`output_field_path is None` and empty `summary_metrics`. The codes, and the
stage at which each is decided:

| Code | Stage | Triggered by |
|---|---|---|
| `CHROMATIX_INVALID_BASELINE_REQUEST` | `request_validation` | Malformed payload, or not exactly one of `input_field_path` / `input_field_array` |
| `CHROMATIX_UNSUPPORTED_PROPAGATION` | `capability_gate` | Any kernel other than `angular_spectrum` |
| `CHROMATIX_UNSUPPORTED_FIELD_KIND` | `capability_gate` | `field_kind="vector"` |
| `CHROMATIX_GRADIENTS_NOT_SUPPORTED` | `capability_gate` | `require_gradients=True` |
| `CHROMATIX_UNSUPPORTED_DEVICE` / `_DTYPE` | `capability_gate` | Anything but `cpu` / `complex64` |
| `CHROMATIX_INVALID_METADATA` | `metadata_validation` | Blank `phasor`, `coordinate_frame`, `origin`, `reference_plane`, or `normalization` |
| `CHROMATIX_INVALID_SAMPLING` | `sampling_validation` | Non-finite or non-positive wavelength, pitch, or index; non-finite `z` |
| `CHROMATIX_INVALID_PADDING` | `padding_validation` | Policy/`pad_width` mismatch, negative width, bad `output_mode`, or `auto_transfer` with non-square pixels |
| `CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED` | `resource_estimate` | Padded grid above `max_output_pixels` (nothing is executed) |
| `CHROMATIX_DEPENDENCY_UNAVAILABLE` | `dependency_gate` | chromatix/jax not importable |
| `CHROMATIX_INPUT_FIELD_UNREADABLE` | `input_field_load` | Missing or unreadable `.npy` |
| `CHROMATIX_INPUT_FIELD_NOT_COMPLEX` / `_NOT_2D` / `_NOT_FINITE` | `input_field_validation` | Real-valued array (intensity/amplitude confusion), wrong rank, NaN/Inf |
| `CHROMATIX_NONFINITE_OUTPUT` | `output_validation` | NaN/Inf in the propagated field |
| `CHROMATIX_SOLVER_EXECUTION_FAILED` | `solver_call` | Chromatix itself raised |

Note the deliberate asymmetry with `ChromatixAdapter.run()` (the graph-facing
path), which still *raises* `AdapterDependencyError` /
`UnsupportedCapabilityError` per the repository exception policy. Both
behaviors are intentional: the graph path fails loudly at composition time,
the baseline path returns a machine-readable record for a benchmark bundle.

## A real-valued input array is rejected, not promoted

**Symptom:** `CHROMATIX_INPUT_FIELD_NOT_COMPLEX` from `run_standalone`, or a
`SolverExecutionError` from `run()`, when passing a `float32`/`float64` array.

**Cause:** deliberate. A real array at a field boundary is almost always an
intensity map, and silently promoting it to a complex amplitude would
reinterpret `I` as `u` — a factor-of-two error in every phase-sensitive
downstream result.

**Fix:** pass `u`, not `|u|^2`. If you genuinely have an amplitude stored as
a real array, cast it yourself (`array.astype(np.complex64)`) so the
intent is explicit and reviewable in the caller, not hidden in the adapter.

## Python version

Chromatix's `pyproject.toml` requires `>=3.12`. This project's own
`pyproject.toml` requires `>=3.11`. There is no conflict for the *project*, but any
environment that needs to import chromatix specifically must be 3.12+.
The `agent_solver` Docker image is what actually satisfies this — do not
assume the host's `.venv` (if one exists) is new enough.

## `AssertionError: Number of wavelengths does not match` from `Field.build`

**Symptom:** raised inside
`chromatix.utils.shapes._broadcast_dx_to_grid` when calling
`chromatix.Field.build(u, dx, spectrum)` with a monochromatic (scalar
wavelength) `spectrum` and a non-square pixel `dx` passed as a bare 1D
array of shape `(2,)`, e.g. `jnp.asarray([pitch_y, pitch_x])`.

**Cause:** `_broadcast_dx_to_grid` treats a 1D array of length `N` as "one
scalar `dx` per wavelength" when `N` does not equal 1; for a `MonoSpectrum`
(1 wavelength), a length-2 array is misread as "2 wavelengths' worth of
scalar dx", which does not match the field's actual 1 wavelength.

**Fix:** for a non-square pixel on a monochromatic field, pass `dx` as a 2D
array of shape `(1, 2)`, e.g. `jnp.asarray([[pitch_y, pitch_x]])`. A bare
Python/NumPy scalar (square pixel) works unchanged. See
`chromatix_adapter.py`'s `_run_asm_propagate` for the adapter's use of this.

## `jax_enable_x64` leaks across the test suite

**Symptom:** `chromatix.functional.asm_propagate` returns a `complex128`
output field (instead of the expected `complex64`, per
`expected/propagation_probe.json`) only when the chromatix adapter's tests
are run as part of the full repository test suite, not in isolation.

**Cause:** `jax_enable_x64` is a *process-global* flag, so whoever sets it last
wins for every later test in the same pytest process. Two routes reach it:
a module that sets it as an import side effect (Python runs a module body once,
so a later `import` will not re-set it), combined with
`multiscale_optics_agent.adapters.registry._discover()` importing every
`*_adapter.py` module just to read `MODEL_ID`; and this repository's own float64
characterization tests, which set it deliberately.

Historically the import-side-effect offender was `sax.saxtypes.core`
(`jax.config.update("jax_enable_x64", True)`), reached via `sax_adapter.py`
during adapter discovery. CHE-72 removed SAX, which closes that specific route
but not the class of problem.

**Fix:** `chromatix_adapter._do_import_chromatix()` pins `jax_enable_x64=False`
on every call rather than relying on ambient state, which is what makes the
adapter correct independently of import and collection order. Any other
JAX-based code that depends on default (non-x64) precision should assert its own
requirement the same way rather than assuming a clean process.

---

# CHE-57 (PB6) failure guide additions

Every entry below was hit while reproducing Chromatix 101 and all 15 documented
examples against the pinned commit `d24bdf0`. Each has an executable reproduction
under `knowledge/solvers/chromatix/tutorials/` and recorded evidence in `expected/`.

## A tilted beam lands in the wrong place after `transform_propagate`

**Symptom:** the beam is 6% short of `z * tan(theta)` at 20 degrees; the error grows
with angle and vanishes as the angle does.

**Cause:** the single-FFT Fresnel propagator's output coordinate is the
direction-cosine (Fourier) mapping `x' = lambda * z * f_x`, so it reports
`z * sin(theta)`, not the geometric `z * tan(theta)`.

**Fix:** use `asm_propagate` or `transform_propagate_sas` when the geometric
position matters; they agree with `z * tan(theta)` and with each other to 0.3%.
Never mix the two coordinate conventions in one chain.

**Reproduction:** `tutorials/c05_scalable_angular_spectrum.py`.

## A `kykx` tilt is off by a factor of `2*pi`

**Symptom:** the beam moves `2*pi` times too far or too little, or in the wrong
direction.

**Cause:** `plane_wave(kykx=)` is an **angular wavenumber** (radians per length,
`sin theta = kykx/k0`) while `asm_propagate(kykx=)` is a **spatial frequency**
(cycles per length, `sin theta = lambda*kykx`). The `asm_propagate` displacement is
also opposite in sign to the parameter.

**Fix:** convert explicitly at the boundary. See the measured sweep in
`conventions.md`, "`kykx` means two different things".

## `use_czt=True` gives an amplitude 14x different from `use_czt=False`

**Symptom:** two calls that differ only in `use_czt` produce fields whose norms
differ by an order of magnitude, while looking identical when each is normalised.

**Cause:** the modified-kernel and chirp-z implementations of scaled/shifted
propagation do not share a normalisation. Upstream's own example prints
`3.1434343` and `44.420246` and compares only after normalising.

**Fix:** normalise before comparing, and do not read either norm as a physical
power. Both agree with an independent brute-force oversampled BLAS propagation at
r = 0.9999, so neither is "the wrong one" -- they are on different scales.

**How to detect earlier:** assert on a normalised quantity, or compare `Field.power`
against an independent computation.

## `asm_propagate` output is aliased at long range

**Symptom:** structure appears that is not in the physics -- wrapped energy, ringing
that changes with padding.

**Cause:** without `bandlimit=True` the transfer function is sampled far past its
Matsushima-Shimobaba limit. At `z = 100*D` with 512 px of padding on a 1024 px
window, that limit is **4% of Nyquist**.

**Fix:** pass `bandlimit=True`, and expect it to *remove* power (0.89% here) --
that is the aliased content, not a loss of signal. It matters more off axis.

## `Field.power` does not include the `Spectrum` density weights

**Symptom:** a chromatic radiometric budget comes out double-counted.

**Cause:** `Field.power` is 1.0 **per wavelength** regardless of `density`; the
weights enter `Field.intensity`, which sums over the wavelength axis.

**Fix:** never multiply `power` by `density`. Use `intensity` for a weighted
quantity.

## `VectorField.u`'s components are in the opposite order to this project's

**Symptom:** an x-polarized field appears to be z-polarized, or a coupler's Jones
vector is reversed.

**Cause:** Chromatix orders the trailing axis `(E_z, E_y, E_x)`; this project uses
`(E_x, E_y, E_z)`.

**Fix:** transpose at the boundary. Established by measurement from three entry
points -- `cf.linear(0)`, `gaussian_plane_wave(amplitude=[0,0,1])`, and
`modified_born_series.solve()`'s output -- see `conventions.md`, "Polarization".

## `modified_born_series` raises a broadcasting `TypeError`, or its output looks transposed

**Symptom:** `TypeError: mul got incompatible shapes for broadcasting:
(320, 405, 1, 3), (3, 320, 405, 1)` from `split_trans_long_ft`.

**Cause:** despite `solve()`'s docstring ("the first (left-most) axis the
polarization vector"), both the input current density and the returned field are
**component-last** `(*spatial, 3)`.

**Fix:** build the source component-last and index the result with `[..., i]`.

Two neighbouring traps in the same module:

- `Source(current_density=..., k0=...)` takes a **current density**, not a field.
  `Source(field=...)` raises `TypeError`. It stores
  `field = -1j/k0 * 1e-6 * c * mu_0 * current_density`.
- `add_absorbing_bc` **pads** the sample: `[256, 341, 1]` becomes `(320, 405, 1)` at
  `thickness=2.0`. Use `Sample.ROI` (a tuple of slices) to recover the original
  region; indexing the padded array with an un-padded mask raises `IndexError`.

**Reproduction:** `tutorials/c15_modified_born_series.py`.

## An `equinox.Module` with a NumPy array as a static field breaks `jax.jit`

**Symptom:** `ValueError: Exception raised while checking equality of metadata
fields of pytree. Make sure that metadata fields are hashable and have simple
equality semantics. (Note: arrays cannot be passed as metadata fields!)` -- but only
once a *second* instance of the module reaches the same jitted function.

**Cause:** `eqx.field(static=True, default_factory=lambda: np.arange(1, 11))` makes
the module unhashable-by-equality. A single instance works; two do not. The
upstream Zernike-fitting example declares `ansi_indices` this way.

**Fix:** use a tuple (or any hashable) for the static field and convert inside the
forward pass.

## `optax` optimizer state silently frozen

**Symptom:** an Adam loop converges much better -- or much worse -- than expected,
and the difference does not respond to `maxiter`.

**Cause:** two of the four Chromatix optimization examples define

```python
@jax.jit
def update(model, opt_state, data):
    grads, metrics = jax.grad(loss_fn, has_aux=True)(model, data)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = optax.apply_updates(model, updates)
    return model, metrics          # <-- opt_state is NOT returned
```

and call it as `model, metrics = update(model, opt_state, data)`, so the **initial**
optimizer state is re-passed on every iteration. Every step is then a fresh
bias-corrected Adam step -- effectively sign descent at a fixed step of `lr` -- and
the moment estimates never accumulate.

**Consequence:** this is not a cosmetic bug. `c10_seidel_fitting` reproduces its
published numbers **only** with the state frozen (final loss 0.784 vs 9.39 when
threaded), while `c03` and `c09` thread it correctly. Before trusting any published
Chromatix optimization result, check whether its `update()` returns the new state.

**Reproductions:** `tutorials/c04_zernike_fitting.py`, `tutorials/c10_seidel_fitting.py`.

## `scikit-image` is not installed

**Symptom:** `ModuleNotFoundError: No module named 'skimage'` from the Fourier
ptychography and DMD examples, which use `skimage.data.camera()`, `moon()` and
`cat()` as targets.

**Fix:** substitute a deterministic target (`chromatix.utils.siemens_star` is
bundled and works well). Note that upstream's published loss/correlation values are
properties of those photographs and cannot be reproduced against a different
target -- only the behavioural claims can.

## `pollen_3d`'s occupancy and `radius` are both counter-intuitive

**Symptom:** `count_nonzero` reports 52% of a "small object" volume occupied, and
reducing `radius` makes the object *bigger*.

**Cause:** `pollen_3d` returns a real `float64` field, not a mask, and it contains
subnormal doubles down to `4.9e-324`. Its `radius` is not an object radius: at the
default `0.8` the phantom is compact and interior, and at `0.25` it fills the whole
volume and clips against the boundary.

**Fix:** measure occupancy above a threshold relative to the maximum, and keep
`radius` at or above the default if the phantom must be paddable. Note also that
upstream colourises it through `np.angle`, which is identically 0 for a
non-negative real array -- that plot's hue axis carries no information.

## An unseeded `filaments_3d` cannot be a fixture

**Symptom:** a phantom-based test is flaky.

**Cause:** the documented example calls `filaments_3d` without a seed.

**Fix:** the signature *does* accept `seed=` (the Holoscope example uses
`seed=972920147`) and is bit-reproducible with it. Always pass one.

## `defocused_ramps` requires exactly six `delta` entries

**Symptom:** `IndexError: list index out of range` from
`chromatix/utils/initializers.py`.

**Cause:** the function indexes `delta[ramp_idx]` for six fixed ramps, so the
six-view geometry is structural rather than configurable.

**Fix:** pass a 6-element `delta`.

## An FFT-built convolution target is circularly shifted

**Symptom:** a hologram appears not to form at the requested voxel.

**Cause:** `jnp.fft.fftn(kernel, s=sample.shape)` places the kernel's **origin** at
index 0 rather than centring it, so the circular convolution translates every
feature by half the kernel width on each axis. The CGH example's three seeded
voxels end up 12 voxels away from where the blobs actually are.

**Fix:** `fftshift` the kernel before transforming, or evaluate quality at the
shifted coordinates. Verified: at the shifted centres the optimized intensity is
75x, 195x and 155x the volume mean; at the seeded coordinates one of the three
shows no enhancement at all.
