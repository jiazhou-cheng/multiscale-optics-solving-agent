# Minimal FMMAX examples (validated against pinned version `1.7.1`)

Every snippet below was actually executed inside the `agent_solver`
container against the pinned install (`fmmax==1.7.1`, `docker/requirements.txt`).
Output values shown are real, captured on 2026-07-30, not illustrative.
The overall pattern is adapted from the maintainers' own
`examples/ar_coating.py` in https://github.com/invrs-io/fmmax (fetched
directly from GitHub, not reconstructed from memory).

## 1. Import / initialization

```python
import fmmax
import jax
jax.devices()  # -> [CpuDevice(id=0)] in this environment
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Minimal forward simulation: bare interface vs. analytic Fresnel oracle

```python
import jax.numpy as jnp
import fmmax

n_ambient, n_substrate, wavelength = 1.0, 1.5, 0.55

in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
    wavelength=jnp.asarray(wavelength),
    polar_angle=jnp.asarray(0.0),
    azimuthal_angle=jnp.asarray(0.0),
    permittivity=jnp.asarray(n_ambient**2),
)
primitive_lattice_vectors = fmmax.LatticeVectors(u=fmmax.X, v=fmmax.Y)
expansion = fmmax.generate_expansion(
    primitive_lattice_vectors=primitive_lattice_vectors,
    approximate_num_terms=1,  # homogeneous/zero-order limit -> Fresnel oracle
    truncation=fmmax.Truncation.CIRCULAR,
)
perms = [jnp.asarray(n_ambient**2)[..., None, None],
         jnp.asarray(n_substrate**2)[..., None, None]]
layer_solve_results = [
    fmmax.eigensolve_isotropic_media(
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=p, expansion=expansion,
        formulation=fmmax.Formulation.FFT,
    ) for p in perms
]
s_matrix = fmmax.stack_s_matrix(layer_solve_results, [jnp.asarray(0.0), jnp.asarray(0.0)])
r_te = s_matrix.s21[..., 0, 0]       # s21 = REFLECTION (see conventions.md)
t_te = s_matrix.s11[..., 0, 0]       # s11 = TRANSMISSION (see conventions.md)
# |r_te|^2 == 0.039999988; analytic Fresnel R == 0.040000000 (rel. error ~3.0e-7)
```

Full probe: `probes/fresnel_oracle_probe.py`; captured output:
`expected/fresnel_oracle_probe.json`.

Multi-layer stacks just extend the `perms`/`layer_solve_results`/
`thicknesses` lists (see `examples/ar_coating.py` upstream for a full
multi-film antireflection-coating optimization built on this exact
pattern with `scipy.optimize`).

## 3. Batched / vectorized example

Not yet executed in this repository. The upstream docs describe batching
via leading array dimensions on `wavelength`/`permittivity`/etc. (except
amplitudes/fields, which batch on a trailing axis -- see
`conventions.md`); this should be probed with a real multi-wavelength
array before an adapter relies on it.

## 4. Gradient example

```python
import jax, jax.numpy as jnp
import fmmax

def reflectance(n_substrate):
    ...  # same construction as above, with n_substrate as the traced argument
    return jnp.abs(s_matrix.s21[..., 0, 0]) ** 2

jax.grad(reflectance)(jnp.asarray(1.5))  # -> 0.12799996
```

Centered finite difference at the same point (`eps=1e-3`) gives
`0.12801588` (relative error `1.24e-4`). This is one narrow
directional-derivative check on a non-periodic (homogeneous-limit)
structure, not the full repository gradient-verification test. Full probe:
`probes/gradient_probe.py`; captured output: `expected/gradient_probe.json`.

## 5. Serialization / export

Not yet exercised.

## 6. Common error signatures and repairs

See `failure_guide.md`.
