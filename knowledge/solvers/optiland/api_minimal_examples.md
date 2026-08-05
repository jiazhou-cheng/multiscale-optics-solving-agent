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
# f2 == 2.005240270799113  (unit not independently confirmed)
```

Full probe: `probes/raytrace_probe.py`; captured output:
`expected/raytrace_probe.json`.

## 3. Batched / vectorized example

Not yet executed in this repository.

## 4. Gradient example (opt-in torch backend)

```python
import torch
import optiland.backend as be
from optiland.samples.objectives import ReverseTelephoto

be.set_backend('torch')  # required -- the numpy default has no gradients

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
`4.130711772631912e-05` — relative error `1.11e-03`. This is one narrow
directional-derivative check with a looser tolerance than the JAX-based
solvers in this repository (not yet root-caused), not the full CLAUDE.md
section 6.2 gradient test. Full probe: `probes/gradient_probe.py`;
captured output: `expected/gradient_probe.json`.

**What happens if torch is not installed was NOT independently verified in
this pass.** This container has torch installed, so `set_backend('torch')`
always succeeds here; a reliable no-torch test would need a second image
built without it, which was out of scope for this pass. Do not assume a
specific error message/timing (e.g. immediate `ImportError` at
`set_backend` time vs. a later failure only when a torch-specific op runs)
without testing it directly.

## 5. Serialization / export

Not yet exercised (`optiland.fileio` exists but was not probed).

## 6. Common error signatures and repairs

See `failure_guide.md`.
