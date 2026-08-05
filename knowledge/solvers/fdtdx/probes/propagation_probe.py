"""Minimal forward-simulation probe for M_EM_FDTDX: a periodic vacuum volume
with a single Gaussian plane source, run for a handful of time steps.

The GitHub README/main-branch example (examples/simulate_gaussian_source.py)
uses `fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=...))`, which does
NOT exist on the pinned 0.6.2 wheel (`AttributeError: module 'fdtdx' has no
attribute 'UniformGrid'`) -- see failure_guide.md. This probe uses the real
installed signature (`resolution=`, `backend=`) instead.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/fdtdx/probes/propagation_probe.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import fdtdx


def main() -> None:
    key = jax.random.PRNGKey(0)

    config = fdtdx.SimulationConfig(
        time=5e-15,
        resolution=100e-9,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
    )

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
        radius=1e-6,
        std=1 / 3,
        direction="+",
    )
    constraints.append(
        source.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0)
        )
    )
    object_list.append(source)

    energy_detector = fdtdx.EnergyDetector(name="energy")
    constraints.extend(energy_detector.same_position_and_size(volume))
    object_list.append(energy_detector)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=subkey
    )

    report = {
        "time_steps_total": config.time_steps_total,
        "E_shape_before_run": list(arrays.fields.E.shape),
        "E_dtype": str(arrays.fields.E.dtype),
    }

    arrays2, new_objects, info = fdtdx.apply_params(arrays, objects, params, subkey)
    final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
    _, arrays_out = final_state

    report["E_shape_after_run"] = list(arrays_out.fields.E.shape)
    report["max_abs_E"] = float(jnp.max(jnp.abs(arrays_out.fields.E)))
    report["any_nan"] = bool(jnp.any(jnp.isnan(arrays_out.fields.E)))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
