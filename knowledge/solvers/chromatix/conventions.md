# Chromatix conventions (pinned commit `d24bdf0`, tag `0.6.0`)

Every fact below was either read from `inspect.getsource`/`inspect.getdoc` on
the installed package, or observed directly by running
`knowledge/solvers/chromatix/probes/propagation_probe.py` inside the
`agent_solver` container. None of it is copied from memory or from a
description of an older/different version.

## Units

Chromatix is **unit-scale-agnostic**: nothing in the library enforces SI
units, meters, or microns. `dx`, wavelength (`spectrum`), and `z` just need
to be expressed in the *same* length unit as each other; the examples in the
upstream docs happen to use microns because that is the natural scale for
visible-light optics, not because the library requires it.

**Implication for this project:** the project's canonical convention (SI,
meters; CLAUDE.md section 7) is not automatically satisfied. A future
`M_WAVE_CHROMATIX` adapter must convert `sample_pitch`/`wavelength` from the
project's meter-valued artifacts into whatever consistent scale is used for
a given call, and convert back on the way out. This conversion boundary
must be tested explicitly (round trip), not assumed.

## Array shape and axis order

Confirmed via `Field.spatial_dims` (`(field.dims.y, field.dims.x)`, i.e. axes
are tracked dynamically per-`Field`, not hardcoded) and via
`propagation_probe.json`:

- `ScalarField.u` has shape `(..., height, width)` with zero or more leading
  batch dims (e.g. depth/z stacks).
- `ChromaticScalarField.u` (multi-wavelength) appends a trailing wavelength
  axis: `(..., height, width, wavelengths)`.
- Axis order is **(y, x)**, i.e. height before width — do not assume `(x,
  y)`.
- `field.dx` and `field.wavelength` are returned as small arrays (observed
  shapes `(2,)` and `(1,)` respectively for a monochromatic square-pixel
  field), not bare Python scalars, even when a scalar was passed in.

## Propagator behavior (observed, not assumed)

From `expected/propagation_probe.json` (shape `(128,128)`, `dx=0.3`,
`wavelength=0.532`, arbitrary consistent length unit):

| Propagator | Changes `dx`? | Output shape vs input | Power conservation observed |
|---|---|---|---|
| `transform_propagate` (single-FFT Fresnel) | **Yes** — `dx` went from 0.3 to 4.618 for `z=500` | same as input (128,128) | 0.997 (near 1, not exact) |
| `asm_propagate` (angular spectrum) | No — `dx` unchanged | **padded**, e.g. (128,128) input -> (1056,1056) output for `z=50` with the library's own `compute_padding_transfer` estimate | 0.999997 |

Two consequences for coupler design:

1. `asm_propagate` returns the *padded* array; a coupler or downstream
   consumer must explicitly crop back to the region of interest if a fixed
   output grid is required. Do not assume the returned shape equals the
   input shape.
2. `transform_propagate` changes the sample pitch as a side effect of using
   a single FFT (this is stated in its docstring: "this method changes the
   sampling of the resulting field"). Any coupler chaining propagators must
   track `field.dx` explicitly rather than assuming it is constant.

## Phase / Fourier sign convention

From `inspect.getsource(compute_asm_propagator)`: the forward angular
spectrum kernel is `exp(+1j * phase)` for `z >= 0` (and its conjugate for
`z < 0`), where `phase = 2*pi*|z|*n/wavelength * delay` and `delay =
sqrt(1 - (wavelength/n)^2 * |f_grid - kykx|^2)`. Plane waves are generated as
`exp(1j * kykx . grid)` (`plane_wave` docstring).

This is the standard `exp(+i k.r)` spatial convention. Chromatix has no
explicit time dependence in its `Field` representation (fields are
snapshots at a plane, not functions of time), so it does not itself declare
a `exp(-i omega t)` vs. `exp(+i omega t)` time convention. Combined with
`exp(+i k.r)`, it is **consistent with** this project's canonical
`exp(-i omega t)` phasor convention (CLAUDE.md section 7) under the usual
identification of a forward-traveling wave, but this has not been
cross-checked against a second solver (e.g. FDTDX) in this repository. Any
coupler joining Chromatix to a solver with an explicit time convention must
verify sign agreement with a manufactured traveling-wave test, not assume
it from this note.

`chromatix.functional.fft`/`ifft` wrap plain `jnp.fft.fft2`/`ifft2` (NumPy's
default normalization: unnormalized forward FFT, `1/N` on the inverse), with
an optional `fftshift`/`ifftshift` pair for centering. Field axes are
`(1, 2)` by default in these free functions, but the real per-`Field`
propagators drive the axes dynamically via `field.spatial_dims` rather than
this default.

## Polarization

`VectorField` and elements like `jones_vector`, `linear_polarizer`,
`quarterwave_plate`, `halfwave_plate` exist (see `api_minimal_examples.md`
function list). **Not probed in this pass** — no Jones-basis ordering or
propagation-frame convention has been verified yet. Treat any
vector/polarization claim in the model registry as unverified until a
dedicated probe is added (see `solver_card.yaml` `not_yet_probed`).

## Numerical dtype

Default complex dtype observed is `complex64` (single precision). No
`complex128` path has been exercised yet.

**Update (2026-07-30, from implementing `chromatix_adapter.py`):**
`chromatix.core.field.ScalarField.__init__` unconditionally does
`self.u = jnp.asarray(u, dtype=jnp.complex64)` -- it does not preserve a
`complex128` input array. This means the `M_WAVE_CHROMATIX` registry entry's
`dtypes: [complex64, complex128]` overstates what Chromatix itself actually
supports for `ScalarField`; a `complex128` input is silently downcast to
`complex64` *inside Chromatix*, not by the adapter. Separately, under
`jax.config.update("jax_enable_x64", True)` (a *process-global* JAX flag),
`chromatix.functional.asm_propagate` was observed to return a `complex128`
output field even when its input was explicitly `complex64` -- i.e. the
*output* dtype can still silently follow the ambient x64 setting despite the
input-side downcast above. `sax.saxtypes.core` sets `jax_enable_x64=True` as
an import side effect, and
`multiscale_optics_agent.adapters.registry._discover()` eagerly imports
every `*_adapter.py` module (including `sax_adapter.py`) to read its
`MODEL_ID`, so this flag can already be flipped process-wide before any
chromatix-specific test runs, purely as a consequence of test collection
order. `tests/test_chromatix_adapter.py` pins `jax_enable_x64=False` in an
autouse fixture to keep its assertions reproducible regardless of order.

## Field construction from an arbitrary array (`Field.build`)

`chromatix.Field.build(u, dx, spectrum)` picks `ScalarField`/`VectorField`
automatically from `u.ndim`/`spectrum` type and is the correct way to wrap
an externally-produced 2D array (as opposed to `cf.plane_wave`/
`cf.point_source`, which only generate synthetic fields). For a
monochromatic (`MonoSpectrum`) field, `dx` must be either a bare scalar
(square pixel) or a 2D array of shape `(1, 2)` -- i.e. `(wavelengths, 2)`.
Passing a plain 1D array of shape `(2,)` for a non-square pixel raises
`AssertionError: Number of wavelengths does not match` inside
`chromatix.utils.shapes._broadcast_dx_to_grid`, because a length-2 1D array
is interpreted as "one scalar dx per wavelength" (here 2 wavelengths),
not as an explicit `(pitch_y, pitch_x)` pair. Verified directly against the
installed package (commit `d24bdf0`, tag `0.6.0`).
