# Chromatix conventions (pinned commit `d24bdf0`, tag `0.6.0`)

CHE-12 re-verified the installed distribution as version 0.6.0 from exact
Git commit `d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee`, with JAX 0.6.2 on
CPU, NumPy 2.2.6, and `jax_enable_x64=False`. See
`benchmarks/probes/verify_m1_engines.py --engine wave`. This is an
environment and convention check, not an analytic propagation benchmark.

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
meters; repository scientific conventions) is not automatically satisfied. A future
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

## `kykx` means two different things (CHE-57)

`kykx` appears on both `plane_wave` and `asm_propagate` and the two are a factor of
`2*pi` apart. Established by sweeping three values and measuring unclipped lateral
displacements over a known distance:

| function | unit of `kykx` | relation to tilt |
|---|---|---|
| `plane_wave(..., kykx=)` | angular wavenumber, radians per length | `sin(theta) = kykx / (2*pi/lambda)` |
| `asm_propagate(..., kykx=)` | **spatial frequency, cycles per length** | `sin(theta) = lambda * kykx` |

For `asm_propagate` the displacement is also **opposite in sign** to the parameter.
Measured at `lambda = 0.532`, `z = 108953.6`:

| `kykx_x` | `lambda*|k|*z` | `|k|/(2*pi/lambda)*z` | measured |
|---|---|---|---|
| -0.004589 | 266.00 | 42.34 | **265.83** |
| -0.009178 | 532.00 | 84.67 | **532.30** |
| -0.018356 | 1064.00 | 169.34 | **1063.15** |

Reading either convention for the other is a `2*pi` position error. Evidence:
`tutorials/c06_off_axis_propagation.py`, cross-checked against
`tutorials/c05_scalable_angular_spectrum.py` where the upstream example writes
`kx = 2*pi/lambda * sin(20 deg)` for `plane_wave` and gets a 20-degree beam.

## `transform_propagate` reports the paraxial position (CHE-57)

The single-FFT Fresnel propagator's output coordinate is the direction-cosine
(Fourier) mapping `x' = lambda * z * f_x`, so a tilted beam lands at
`z * sin(theta)` rather than the geometric `z * tan(theta)`. Measured at
`theta = 20 deg`, `z = 1024`: peak at **350.000**, which is `z*sin(theta) = 350.229`
to 0.05% and **6.1% short** of `z*tan(theta) = 372.706`.

`transform_propagate_sas` and `asm_propagate` both give the geometric position and
agree with each other and with `z*tan(theta)` to 0.3% -- from two completely
different discretisations. Any coupler chaining propagators must not mix the two
coordinate conventions. Evidence: `tutorials/c05_scalable_angular_spectrum.py`.

SAS additionally rescales its output pitch by **exactly** the magnification the
caller asks for (`dx_out / dx_in = 8.000000` for `M_box = 8`), where ASM instead
preserves the pitch and pays with a padded grid.

## Rescaled/shifted propagation: the two paths disagree in amplitude (CHE-57)

`asm_propagate(output_dx=..., shift_yx=...)` has two implementations, selected by
`use_czt`. They agree on **structure** (r = 0.998-0.9999, normalised RMSE 7.5e-06)
and **disagree on amplitude by a large factor**: at 4x zoom the CZT output's norm is
`44.409` against the modified kernel's `3.1434`, i.e. 14.13x.

This is upstream-known: the "Scaled and Shifted Free-Space Propagation" example's
own printed output is `3.1434343` and `44.420246`, and it compares the two only
after normalising each by its own norm. It is documented nowhere else.
**Never treat `use_czt=True` as a drop-in alternative, and never read either norm
as a physical power.** Both were separately verified against an independent
4096x4096 brute-force BLAS propagation and agree with it at r = 0.9999.

Evidence: `tutorials/c08_rescaled_propagation.py`, `tutorials/c06_off_axis_propagation.py`.

## Band limiting is not optional at long range (CHE-57)

`asm_propagate(bandlimit=True)` applies the Matsushima-Shimobaba limit
`f <= 1/(lambda*sqrt((2z/L_pad)^2 + 1))`. At `z = 100*D` on a 1024 px window with
512 px padding that limit is **4.0% of Nyquist**, i.e. 96% of the sampled band is
undersampled and the un-band-limited kernel wraps energy back into the window.

Measured consequences: `bandlimit=True` changes the amplitude by 6.9% RMS on axis
and 9.0% with a displaced window, and **removes** 0.89% of the discrete power --
it discards the aliased content rather than synthesising anything. Band limiting
matters *more* off axis, because a displaced window samples a steeper part of the
transfer function. Evidence: `tutorials/c07_bandlimited_angular_spectrum.py`.

## `Spectrum` density weights do not scale `Field.power` (CHE-57)

For a `ChromaticScalarField` built with
`Spectrum(wavelength=[0.532, 0.512], density=[0.6, 0.4])`, `Field.power` is
**1.0 per wavelength**, not density-weighted. The weights enter through
`Field.intensity`, which sums over the trailing wavelength axis -- so a
two-wavelength field whose densities sum to 1 has the same *mean intensity* as the
equivalent monochromatic field while its *power* stays 1 per wavelength. A consumer
that multiplies `power` by `density` double-counts.

`Field.dx` for a chromatic field is `(num_wavelengths, 2)`, one row per wavelength.
Evidence: `tutorials/c00_chromatix_101.py`.

## The `elements`/`systems` layer equals the `functional` layer (CHE-57)

`OpticalSystem([PlaneWave(...), FFLens(...)])()` and
`ff_lens(plane_wave(...), ...)` produce **bit-identical** fields (max
`|delta u| = 0`). The two APIs are interchangeable, which is what allows the
adapter to use the functional one while the documentation teaches the element one.

Field names read off the pinned dataclasses rather than the docs:
`Optical4FSystemPSF(shape, spacing, f_tube, phase)` and
`Microscope(system_psf, sensor, f, n, NA, spectrum, padding_ratio, ...)` --
`padding_ratio` lives on `Microscope`, not on `Optical4FSystemPSF`, and `spectrum`
is required. Evidence: `tutorials/c00_chromatix_101.py`.

## Phase / Fourier sign convention

From `inspect.getsource(compute_asm_propagator)`: the forward angular
spectrum kernel is `exp(+1j * phase)` for `z >= 0` (and its conjugate for
`z < 0`), where `phase = 2*pi*|z|*n/wavelength * delay` and `delay =
sqrt(1 - (wavelength/n)^2 * |f_grid - kykx|^2)`. Plane waves are generated as
`exp(1j * kykx . grid)` (`plane_wave` docstring).

This is the standard `exp(+i k.r)` spatial convention. Chromatix has no
explicit time dependence in its `Field` representation (fields are
snapshots at a plane, not functions of time), so it does not itself declare
a `exp(-i omega t)` vs. `exp(+i omega t)` time convention.

### Established by measurement (CHE-35, M3.6)

Reading the kernel source tells you what Chromatix computes, not whether a
field written under *this project's* declaration will focus in it. That is now
settled by a manufactured test rather than by the "consistent with" argument
this section used to make.

Probe: `knowledge/solvers/chromatix/probes/m3_pupil_to_focus.py`; recorded
output: `knowledge/solvers/chromatix/expected/m3_pupil_to_focus.json`;
regression tests: `archive/tests/gen1/tests/test_m3_pupil_to_focus.py` — **archived
by CHE-67 and not runnable**, so nothing in the default suite fails if this
convention regresses. The probe and its recorded output above still run.

Under `exp(-i omega t)` with `exp(+i k z)`, a wave converging to a focus a
distance `R` downstream has pupil field `exp(-i k sqrt(rho^2 + R^2))`, because
the optical path still to travel is longest at the pupil edge. Both that field
and its complex conjugate were propagated by `+R` through
`asm_propagate`. Exactly one must concentrate, and it is not inferable which:

| field | peak on axis | peak intensity | concentration vs input |
|---|---|---|---|
| `exp(-i k sqrt(rho^2 + R^2))` (project convention) | **yes** | 4943 | 4943x |
| its conjugate | no | 4.90 | 4.9x |

Peak ratio 1008x. The converging case also reaches **0.990** of the analytic
Airy peak `(pi a^2 / (lambda R))^2` for a clear circular aperture, so the test
pins the geometry as well as the sign.

**Established:** `asm_propagate` implements `exp(+i k_z z)` for `z > 0`, which
is this project's declared spatial factor. A field written under
`exp(-i omega t)` / `exp(+i k z)` focuses; its conjugate does not.

Consequence in the adapter: a mismatched input `phasor` is now a **structured
refusal** (`CHROMATIX_PHASOR_MISMATCH`), not a warning. For a converging pupil
field the two conventions differ by focusing versus defocusing, and no
downstream metric distinguishes them.

Still not established: agreement with a second solver that declares an explicit
time convention (e.g. FDTDX). Any coupler joining those two must verify sign
agreement with its own manufactured traveling-wave test.

`chromatix.functional.fft`/`ifft` wrap plain `jnp.fft.fft2`/`ifft2` (NumPy's
default normalization: unnormalized forward FFT, `1/N` on the inverse), with
an optional `fftshift`/`ifftshift` pair for centering. Field axes are
`(1, 2)` by default in these free functions, but the real per-`Field`
propagators drive the axes dynamically via `field.spatial_dims` rather than
this default.

## Polarization

`VectorField` and elements like `jones_vector`, `linear_polarizer`,
`quarterwave_plate`, `halfwave_plate` exist (see `api_minimal_examples.md`
function list).

### Component order established (CHE-57)

**`VectorField.u`'s trailing axis is ordered `(E_z, E_y, E_x)`** — the *reverse* of
this project's `(E_x, E_y, E_z)`. Any coupler must transpose it.

Established from three independent entry points:

- `cf.linear(0)` (x-polarized) into `plane_wave(scalar=False)` puts 100.0% of the
  field energy at component index **2** and 0.0% at indices 0 and 1
  (`tutorials/c11_polarized_multislice.py`).
- `gaussian_plane_wave(amplitude=jnp.array([0.0, 0.0, 1.0]), scalar=False)` likewise
  gives an x-polarized pupil (`tutorials/c12_high_na_psf.py`).
- The upstream birefringence example's own code corroborates it: it normalises by
  `field.u[90, 90, 2]` to set the *x* amplitude and labels `amplitude[..., 2]` as
  "Amplitude Ex".
- `chromatix.experimental.modified_born_series.solve()` uses the same ordering for
  its own output (`tutorials/c15_modified_born_series.py`).

This matches, and now measures, the note `capability_notes.md` recorded from
reading `high_na_ff_lens`'s source during CHE-18.

### Tensorial propagation verified (CHE-57)

`cf.polarized_multislice_thick_sample` with a per-voxel 3x3 permittivity tensor
converts **2.17%** of an x-polarized input's energy into the `E_y`/`E_z` components
the input did not have, and the four differently-oriented uniaxial beads of the
upstream example give `|E_y|^2` responses spanning **1022x** — so the crystal
orientation genuinely enters the calculation rather than being averaged away. The
assembled tensor is symmetric per voxel to 1.3e-07 relative with 24.8% off-diagonal
content. Output phase is wrapped into `[-pi, pi]`. Evidence:
`tutorials/c11_polarized_multislice.py`.

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
input-side downcast above. `jax_enable_x64` is process-global, and
`multiscale_optics_agent.adapters.registry._discover()` eagerly imports every
`*_adapter.py` module to read its `MODEL_ID`, so any module that sets the flag as
an import side effect can flip it process-wide before a chromatix-specific test
runs, purely as a consequence of collection order. This repository's own float64
characterization tests set it deliberately, which is the same hazard from a
different direction. `chromatix_adapter._do_import_chromatix()` therefore pins
`jax_enable_x64=False` on every call, so its assertions are reproducible
regardless of order.

Until CHE-72 the concrete offender was `sax.saxtypes.core`, which set
`jax_enable_x64=True` on import; SAX has since been removed. The defence stays
because the flag is still process-global and still mutable by anything.

## Grid centering and coordinate origin (CHE-14, verified)

`Field.grid` was read directly for a `(128, 128)` field at `dx = 0.5 um`: the
coordinate range is `[-3.2e-5, +3.15e-5]`, i.e. exactly
`(arange(n) - n // 2) * dx`. **Index `n // 2` is coordinate zero on each
spatial axis**, and for even `n` the sampled window is asymmetric by one
pixel (one more negative sample than positive). The same rule was confirmed
on the padded `(640, 640)` output. Any oracle or coupler that builds its own
coordinate vector must use this rule; assuming a symmetric `linspace(-L/2,
L/2, n)` introduces a half-pixel offset that shows up directly as a centroid
error.

Note that `field.dx` for an input built at `dx = 5e-7` reads back as
`4.999999987e-07` (float32 storage), while the propagated output reports
`5e-07`. Compare sample pitch with a tolerance, never for exact equality.

## Discrete power semantics (CHE-14, verified)

`field.power` is `sum(|u|^2) * dy * dx` over the **sampled window only**. It
is not a radiometric power in watts, and it is not conserved by construction:
energy that leaves the window is simply absent from the output sum. For the
CHE-14 canonical Gaussian case (beam well inside the window, edge-energy
fraction ~3e-10) the observed ratio was `power_out / power_in = 0.9999996`.
Treat any deviation as a **window-truncation diagnostic**, not a physical
conservation law, and read it together with the edge-energy fractions that
`ChromatixAdapter.run_standalone` emits.

## Padding: `compute_padding_transfer` is a worst-case estimator (CHE-14)

`compute_padding_transfer(height, wavelength, dx, z)` derives padding from the
Fresnel number of the **whole window at full bandwidth**:

```
D = height * dx;  Nf = (D / 2)**2 / (wavelength * z)
Q = 2 * max(1, height / (4 * Nf));  pad_width = ceil(Q * height / 2) * 2 - height
```

For the CHE-14 case (128 px, `dx = 0.5 um`, `lambda = 532 nm`, `z = 1.77 mm`)
this returns `pad_width = 7400`, i.e. a `14928 x 14928` grid (~3.3 GB at
complex64), because it assumes the field occupies the full Nyquist bandwidth.
A Gaussian with a `10 um` waist occupies only `~0.05 um^-1` of a `1 um^-1`
Nyquist band, and an explicit `pad_width = 256` (a `640 x 640` grid)
reproduces the analytic result to `7e-5` relative beam radius — verified in
`benchmarks/level1/L1-WAVE-01`.

**Implication:** choose padding from the *occupied* bandwidth (the physical
ray displacement `z * lambda * f_max` must fit inside the padded window), not
from `compute_padding_transfer`, and always gate it with a resource estimate.
`run_standalone` refuses any grid above `max_output_pixels` with a
`CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED` diagnostic instead of attempting it.

`asm_propagate` also accepts `mode="same"`, which crops the padded result back
to the input shape inside Chromatix. The baseline defaults to `mode="full"`
and records `cropped` explicitly, so a crop is never silent.

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
