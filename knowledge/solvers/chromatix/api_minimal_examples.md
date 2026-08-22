# Minimal Chromatix examples (validated against pinned commit `d24bdf0`)

Every snippet below was actually executed inside the `agent_solver`
container (`./run.sh python -c "..."` or the probe scripts in `probes/`)
against the pinned install in `docker/requirements-chromatix.txt`. Output
values shown are real, captured on 2026-07-30, not illustrative.

## 1. Import / initialization

```python
import chromatix
import chromatix.functional as cf
import jax
jax.devices()  # -> [CpuDevice(id=0)] in this environment
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Minimal forward simulation (field creation + two propagators)

```python
import chromatix.functional as cf
from chromatix.functional.propagation import compute_padding_transfer

shape, dx, wavelength, n = (128, 128), 0.3, 0.532, 1.0

plane = cf.plane_wave(shape=shape, dx=dx, spectrum=wavelength, power=1.0)
# plane.u.shape == (128, 128), dtype complex64; plane.power ~= 1.0

# Fresnel (single-FFT) propagation -- changes dx, keeps shape:
fresnel = cf.transform_propagate(plane, z=500.0, n=n, pad_width=32)
# fresnel.dx ~= [4.618, 4.618] (was [0.3, 0.3]); power ~= 0.997

# Angular spectrum propagation -- keeps dx, pads shape:
pad = compute_padding_transfer(shape[0], wavelength, dx, 50.0)
asm = cf.asm_propagate(plane, z=50.0, n=n, pad_width=int(pad))
# asm.u.shape == (1056, 1056) for this pad; dx unchanged; power ~= 0.999997
```

Full probe: `probes/propagation_probe.py`; captured output:
`expected/propagation_probe.json`.

`compute_padding_transfer(height, wavelength, dx, z)` takes **scalars**
(not `Field`/`Spectrum` objects) — passing `field.spectrum` directly raises
`AttributeError: 'MonoSpectrum' object has no attribute 'squeeze'`. See
`failure_guide.md`.

## 3. Batched / vectorized example

**Verified by CHE-57.** A 1D `z` adds a leading batch axis, so one propagation call
produces a whole 3D stack and one `jax.grad` differentiates every plane at once:

```python
z = jnp.linspace(0.0, 100.0e4, num=51)
field = cx.transfer_propagate(field, z, n, pad_width=0, mode="same")
field.intensity.shape   # -> (51, 256, 256, 1, 1)
```

Confirmed on a 51-plane `transfer_propagate` stack
(`tutorials/c03_computer_generated_holography.py`, which reproduces upstream's
printed `(51, 256, 256)`) and a 40-plane `objective_point_source` stack at
1920x1920 (`tutorials/c02_holoscope.py`). `jax.vmap` over an illumination
parameter also works: `tutorials/c01_fourier_ptychography.py` vmaps 121 tilted
illuminations through a full 4f system.

## 4. Gradient example

```python
import jax, jax.numpy as jnp
import chromatix.functional as cf

def objective(f):
    field = cf.plane_wave(shape=(64, 64), dx=1.0, spectrum=0.532, power=1.0)
    field = cf.thin_lens(field, f=f, n=1.0)
    field = cf.transform_propagate(field, z=f, n=1.0, pad_width=32)
    return jnp.sum(field.intensity)

jax.grad(objective)(1000.0)  # -> -1.1463e-4
```

Centered finite difference at the same point (`eps=0.5`) gives
`-1.1463e-4` as well (relative error `2.30e-5`). This is one narrow
directional-derivative check, not the full repository gradient test. Full
probe: `probes/gradient_probe.py`; captured output:
`expected/gradient_probe.json`.

## 5. Serialization / export

Not yet exercised.

## 6. Common error signatures and repairs

See `failure_guide.md`.

## 7. Chromatic (multi-wavelength) fields

```python
from chromatix import Spectrum

field = cx.plane_wave(
    shape=(512, 512), dx=0.3,
    spectrum=Spectrum(wavelength=[0.532, 0.512], density=[0.6, 0.4]),
)
type(field).__name__   # -> 'ChromaticScalarField'
field.shape            # -> (512, 512, 2)   trailing wavelength axis
field.dx               # -> [[0.3, 0.3], [0.3, 0.3]]   one ROW per wavelength
field.power            # -> 1.0 PER WAVELENGTH, not density-weighted
```

The `density` weights enter `Field.intensity` (which sums over the wavelength axis),
**not** `Field.power`. See `conventions.md`. Evidence:
`tutorials/c00_chromatix_101.py`.

## 8. The `elements`/`systems` layer, and its equivalence to `functional`

```python
from chromatix.elements import PlaneWave, FFLens
from chromatix.systems import OpticalSystem

system = OpticalSystem([
    PlaneWave(shape=(128, 128), dx=0.3, spectrum=0.532),
    FFLens(f=100.0, n=1.0),
])
np.max(np.abs(system().u - cx.ff_lens(cx.plane_wave((128, 128), 0.3, 0.532), 100.0, 1.0).u))
# -> 0.0   bit-identical
```

Pinned dataclass fields, read off the installed package rather than the docs:
`Optical4FSystemPSF(shape, spacing, f_tube, phase)` and
`Microscope(system_psf, sensor, f, n, NA, spectrum, padding_ratio, taper_width, ...)`.
`padding_ratio` is on `Microscope`, not `Optical4FSystemPSF`, and `spectrum` is
required. Evidence: `tutorials/c00_chromatix_101.py`.

## 9. Scaled / shifted / band-limited propagation

```python
# band-limited, cropped back to the input shape
out = cx.asm_propagate(field, z, n, pad_width=(512, 512), mode="same", bandlimit=True)

# a zoomed, laterally shifted output window -- two implementations
yu  = cx.asm_propagate(field, z, n, pad_width=pad, mode="same", bandlimit=True,
                       output_dx=field.dx / 4, shift_yx=[0.0, w / 2], use_czt=False)
czt = cx.asm_propagate(field, z, n, pad_width=pad, mode="same", bandlimit=True,
                       output_dx=field.dx / 4, shift_yx=[0.0, w / 2], use_czt=True)
np.linalg.norm(yu.amplitude), np.linalg.norm(czt.amplitude)   # -> 3.143, 44.409
```

**The two paths differ in amplitude by 14.13x** -- normalise before comparing. And
`asm_propagate`'s `kykx` is a spatial frequency while `plane_wave`'s is an angular
wavenumber. Both caveats are in `conventions.md` and `failure_guide.md`. Evidence:
`tutorials/c06_off_axis_propagation.py`, `tutorials/c07_bandlimited_angular_spectrum.py`,
`tutorials/c08_rescaled_propagation.py`.

## 10. Vector fields

```python
field = cx.plane_wave((180, 180), 0.065, 0.405, amplitude=cx.linear(0), scalar=False)
type(field).__name__          # -> 'VectorField'
field.u.shape                 # -> (180, 180, 3)
# component order is (E_z, E_y, E_x) -- the REVERSE of this project's convention.
# cf.linear(0) is x-polarized, so all its energy is at index 2.
out = cx.polarized_multislice_thick_sample(field, potential, n_background, spacing, NA=NA)
```

Evidence: `tutorials/c11_polarized_multislice.py`, `tutorials/c12_high_na_psf.py`.

## 11. The full-wave solver (`chromatix.experimental`)

```python
from chromatix.experimental.modified_born_series.sample import (
    EmptySample, Source, add_absorbing_bc,
)
from chromatix.experimental.modified_born_series.solver import solve

sample = EmptySample([256, 341, 1], wavelength / 8)
sample = sample.replace(permittivity=refractive_index**2)
sample = add_absorbing_bc(sample, axis=(0, 1), thickness=2.0, max_extinction=0.25)
# NOTE: this PADS the sample; use sample.ROI to recover the original region.

current_density = jnp.zeros((*sample.permittivity.shape, 3), dtype=jnp.complex64)  # component LAST
current_density = current_density.at[entrance, sample.ROI[1], :, 2].set(1.0)
E = solve(sample, Source(current_density=current_density, k0=2 * jnp.pi / wavelength),
          maxiter=400, tol=1e-4)
E.shape   # -> (*spatial, 3)   component LAST, despite the docstring
```

`Source` takes a **current density**, not a field. Validated against a closed form:
on a homogeneous domain the axial phase gradient is `12.5656` against the analytic
`k0*n = 12.5664`. Evidence: `tutorials/c15_modified_born_series.py`.

## 12. Full-fidelity example reproductions

16 executable reproductions of Chromatix 101 and all 15 documented examples live
under `tests_tutorial/cases/chromatix/`, each printing machine-readable
evidence. They are the fastest way to find a working minimal example of a
specific API:

```bash
./run.sh python tests_tutorial/cases/chromatix/c06_off_axis_propagation.py
```

See `tutorials/README.md` for the API-to-example index, the coverage table, and the
list of upstream claims that do not reproduce.
