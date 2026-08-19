# Minimal Optiland examples (validated against pinned version `0.6.0`)

Every snippet below was actually executed inside the `agent_solver`
container against the pinned install in `docker/requirements.txt`
(`optiland==0.6.0` plus a separately-pinned CPU-only `torch==2.13.0`).
Output values shown are real, captured on 2026-07-30, not illustrative.

## 1. Import / initialization

```python
import optiland
dir(optiland)  # -> ['annotations'] -- nearly empty; import the submodule you need
import optiland.backend as be
be.get_backend()            # -> 'numpy' (default)
be.list_available_backends()  # -> ['numpy', 'torch']
be.supports_gpu             # -> False (a bool attribute, not a function)
be.supports_gradients       # -> False under numpy
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Minimal forward simulation (bundled sample lens, NumPy backend)

```python
from optiland.samples.objectives import ReverseTelephoto

lens = ReverseTelephoto()
rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=16)
# type(rays) == optiland.rays.real_rays.RealRays
# rays.x.shape == (817,)  -- NOT 16; aperture/pupil sampling changes the count
# rays.x.dtype == float64

f2 = lens.paraxial.f2()
# f2 == 2.005240270799113 mm
```

Full probe: `probes/raytrace_probe.py`; captured output:
`expected/raytrace_probe.json`.

## 3. Batched / vectorized example

Not yet executed in this repository (no batched-system API is used by the current
slice). Per-ray vectorization is inherent: `Optic.trace` returns whole bundles and
`Optic.surfaces.{x,y,z,L,M,N,opd,i}` are `(num_surfaces, num_rays)` arrays.

## 4. Gradient example (opt-in torch backend)

```python
import torch
import optiland.backend as be
from optiland.samples.objectives import ReverseTelephoto

be.set_backend('torch')     # required -- the numpy default has no gradients
be.set_device('cpu')
be.set_precision('float64')  # CHE-57: the torch backend DEFAULTS to float32
be.grad_mode.enable()

def rms_spot(radius_value):
    lens = ReverseTelephoto()
    surf = lens.surfaces.surfaces[1]  # Optic.surface_group is deprecated; use .surfaces
    surf.geometry.radius = radius_value
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=64)
    return (rays.x**2 + rays.y**2).mean()

r0 = torch.tensor(1.6911, dtype=torch.float64, requires_grad=True)
value = rms_spot(r0)          # -> 2.251810400366594e-07
value.backward()
r0.grad.item()                # -> 4.12612817668375e-05
```

Centered finite difference at the same point (`eps=1e-4`) gives
`4.130711772631912e-05` — relative error `1.11e-03`. **That figure is a float32
artifact, and `probes/gradient_probe.py` does not call `set_precision`.** CHE-57
established that the torch backend defaults to `precision=32`; adding
`be.set_precision('float64')` as shown above brings the same comparison to
`6.24e-07` at `eps=1e-4` and `6.28e-09` at `eps=1e-5`, converging as O(eps^2).
See `conventions.md` "Torch backend precision defaults to float32" for the full
table, and `tutorials/t10_differentiable_ray_tracing.py` for the executable form.
Full probe: `probes/gradient_probe.py`; captured output:
`expected/gradient_probe.json`.

Backend state is process-global and not thread-safe. Restore it in a `finally`
block if a test or probe touches it:

```python
original = be.get_backend()
try:
    be.set_backend('torch')
    be.set_precision('float64')
    ...
finally:
    be.set_precision('float32')
    be.set_backend(original)
```

**What happens if torch is not installed was NOT independently verified in
this pass.** This container has torch installed, so `set_backend('torch')`
always succeeds here; a reliable no-torch test would need a second image
built without it, which was out of scope for this pass. Do not assume a
specific error message/timing (e.g. immediate `ImportError` at
`set_backend` time vs. a later failure only when a torch-specific op runs)
without testing it directly.

## 5. Serialization / export

Use the typed CHE-13 boundary rather than serializing solver objects:

```bash
./run.sh python knowledge/solvers/optiland/probes/standalone_baseline.py \
  --output-dir /tmp/optiland-che13
```

The command performs two real CPU traces, writes each SI ray bundle and
machine-readable summary, then verifies stable summary metrics and the
scientific-array SHA-256. The tracked regression fixture
`expected/standalone_baseline.json` is generated only by passing
`--write-expected` to this executable.

## 6. Common error signatures and repairs

See `failure_guide.md`.

## 7. The modern (non-deprecated) construction API

Every `Optic.add_*` / `set_field_type` / `set_thickness` / `update_paraxial` /
`set_polarization` call is deprecated for removal in 0.7.0. The equivalent modern
form, verified bit-identical (`tutorials/t01_singlet_lens.py`):

```python
import numpy as np
from optiland import optic

lens = optic.Optic(name='singlet')
lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
lens.surfaces.add(index=1, radius=20.0, thickness=7.0, is_stop=True, material='N-SF11')
lens.surfaces.add(index=2, radius=np.inf, thickness=18.0)
lens.surfaces.add(index=3)
lens.set_aperture(aperture_type='EPD', value=25.0)   # NOT deprecated
lens.fields.set_type(field_type='angle')
lens.fields.add(y=0)
lens.wavelengths.add(value=0.5, is_primary=True)
lens.updater.update_paraxial()
```

`surfaces.add` accepts `surface_type=` (`'standard'`, `'even_asphere'`,
`'polynomial'`, `'zernike'`, ...), `conic=`, `coefficients=`, `aperture=`,
`coating=`, `bsdf=`, the placement kwargs `thickness=`/`z=`/`y=`/`x=`, and the
perturbation kwargs `dx=`/`dy=`/`rx=`/`ry=`/`rz=` (tilts in **radians**). Absolute
(`z=`, `y=`) and relative (`thickness=`, `dy=`) placement of the same system give
element-wise identical traces (`tutorials/t09_non_rotationally_symmetric.py`).

`material='mirror'` produces a reflective surface, but **no distinct surface or
interaction class**: every surface reports
`interaction_model=RefractiveReflectiveModel`, so reflectivity is not detectable
from the type. Detect it geometrically (a 45-degree fold deviates the chief ray
by 90 degrees) or from `material_post`.

## 8. Full-fidelity tutorial reproductions

41 executable reproductions of the official Optiland tutorial index live under
`knowledge/solvers/optiland/tutorials/`, each printing machine-readable evidence.
They are the fastest way to find a working minimal example of a specific API:

```bash
./run.sh python knowledge/solvers/optiland/tutorials/t26_zernike_decomposition.py
```

See `tutorials/README.md` for the API-to-tutorial index and the coverage table.
