# Minimal FDTDX examples (validated against pinned version `0.6.2`)

Every snippet below was actually executed inside the `agent_solver`
container against the pinned install in `docker/requirements.txt`. Output
values shown are real, captured on 2026-07-30, not illustrative. `run_fdtd`
prints a live tqdm-style progress bar (`FDTD (checkpointed): ...`) to
stdout; strip it with `| grep -v "FDTD (checkpointed)"` if you want clean
output.

## 1. Import / initialization

```python
import fdtdx
import jax
jax.devices()  # -> [CpuDevice(id=0)] in this environment
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Minimal forward simulation (vacuum volume, periodic boundary, Gaussian source)

```python
import jax, jax.numpy as jnp, fdtdx

key = jax.random.PRNGKey(0)
config = fdtdx.SimulationConfig(
    time=5e-15, resolution=100e-9, backend="cpu",
    dtype=jnp.float32, courant_factor=0.99,
)  # config.time_steps_total == 26

constraints, object_list = [], []
volume = fdtdx.SimulationVolume(
    partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6),
    material=fdtdx.Material(permittivity=1.0),
)
object_list.append(volume)

bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
constraints.extend(c_list)
object_list.extend(list(bound_dict.values()))

source = fdtdx.GaussianPlaneSource(
    partial_grid_shape=(None, None, 1),
    partial_real_shape=(2e-6, 2e-6, None),
    fixed_E_polarization_vector=(1, 0, 0),
    wave_character=fdtdx.WaveCharacter(wavelength=1.0e-6),
    radius=1e-6, std=1 / 3, direction="+",
)
constraints.append(source.place_relative_to(
    volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0),
))
object_list.append(source)

energy_detector = fdtdx.EnergyDetector(name="energy")
constraints.extend(energy_detector.same_position_and_size(volume))
object_list.append(energy_detector)

key, subkey = jax.random.split(key)
objects, arrays, params, config, _ = fdtdx.place_objects(
    object_list=object_list, config=config, constraints=constraints, key=subkey,
)
# arrays.fields.E.shape == (3, 30, 30, 30), dtype float32 -- see conventions.md
# for why the leading axis of size 3 is (Ex, Ey, Ez), not trailing.

arrays2, new_objects, info = fdtdx.apply_params(arrays, objects, params, subkey)
final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
_, arrays_out = final_state
# max(|E|) ~= 0.0389 after 26 steps, no NaNs.
```

Full probe: `probes/propagation_probe.py`; captured output:
`expected/propagation_probe.json`.

**Do not copy the GitHub README's `examples/simulate_gaussian_source.py`
verbatim** -- it uses `fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(...))`,
which raises `AttributeError: module 'fdtdx' has no attribute 'UniformGrid'`
on this pinned version. Use `resolution=` as shown above. See
`failure_guide.md`.

## 3. Batched / vectorized example

Not yet executed in this repository.

## 4. Gradient example

**Both attempts below failed** -- included here specifically because a
failing, well-documented example is more useful than no example, per
CLAUDE.md's ban on fabricated gradient claims.

```python
# (a) Gradient w.r.t. source wavelength -- returns exactly 0.0 (wrong):
jax.grad(run_with_wavelength)(1.0e-6)  # -> 0.0
# Centered finite difference at the same point (eps=1e-9) gives ~-1.02e6.
# See conventions.md "A real, reproducible zero-gradient case".

# (b) Gradient w.r.t. background permittivity passed into the Material()
# constructor -- raises jax.errors.ConcretizationTypeError, because
# place_objects() does concrete (non-traceable) introspection of
# material properties. See conventions.md "Object graph construction is
# NOT traceable".
jax.grad(run_with_permittivity)(1.0)  # -> ConcretizationTypeError
```

Full probe: `probes/gradient_probe.py`; captured output:
`expected/gradient_probe.json`. Do not use either pattern in an adapter.
The likely-correct pattern (differentiating through `apply_params(arrays,
objects, params, key)` with respect to `params`, or using
`fdtdx.full_backward`) has not yet been worked out -- see
`capability_notes.md`.

## 5. Serialization / export

Not yet exercised (`export_json`, `export_stl`, `export_vti` exist in the
API but were not probed).

## 6. Common error signatures and repairs

See `failure_guide.md`.
