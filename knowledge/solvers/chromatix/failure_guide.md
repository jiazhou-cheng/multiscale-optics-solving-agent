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

## `jax_enable_x64` leaks across the test suite via `sax`

**Symptom:** `chromatix.functional.asm_propagate` returns a `complex128`
output field (instead of the expected `complex64`, per
`expected/propagation_probe.json`) only when the chromatix adapter's tests
are run as part of the full repository test suite, not in isolation.

**Cause:** `sax.saxtypes.core` calls
`jax.config.update("jax_enable_x64", True)` as an import side effect, and
`multiscale_optics_agent.adapters.registry._discover()` (invoked by
`test_adapter_registry.py`, which collects/runs before
`test_chromatix_adapter.py`) imports every `*_adapter.py` module -- including
`sax_adapter.py` -- just to read `MODEL_ID`. `jax_enable_x64` is a
process-global flag, so this leaks into unrelated later tests in the same
pytest process.

**Fix:** `tests/test_chromatix_adapter.py` has an autouse fixture that pins
`jax_enable_x64=False` for the duration of each test in that file and
restores the previous value afterward. Any other JAX-based adapter test
that depends on default (non-x64) precision should do the same rather than
assuming a clean process.
