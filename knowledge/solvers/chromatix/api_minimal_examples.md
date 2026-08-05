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

Not yet executed in this repository. `z` accepts a 1D array per the
`asm_propagate`/`transform_propagate` docstrings ("a 1D array of distances,
in which case a batch dimension will be added"); this should be probed
before an adapter relies on it.

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
directional-derivative check, not the full CLAUDE.md section 6.2 gradient
test. Full probe: `probes/gradient_probe.py`; captured output:
`expected/gradient_probe.json`.

## 5. Serialization / export

Not yet exercised.

## 6. Common error signatures and repairs

See `failure_guide.md`.
