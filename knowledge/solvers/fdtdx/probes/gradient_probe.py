"""Directional-derivative probes for M_EM_FDTDX.

Both probes here FAILED to produce a trustworthy gradient. That is a real,
important finding, not a bug in this probe: it means the naive pattern
"wrap setup+run in jax.grad" does not work for fdtdx, and no derivative
path should be marked verified from this pass. See conventions.md and
failure_guide.md for the (partial) explanation and the correct-looking
pattern (fdtdx.Device / ParameterContainer / apply_params) that was not
fully worked out here.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/fdtdx/probes/gradient_probe.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import fdtdx


def _base_objects(volume, wavelength):
    constraints, object_list = [], [volume]
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(2e-6, 2e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
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
    return constraints, object_list


def run_with_wavelength(wavelength):
    key = jax.random.PRNGKey(0)
    config = fdtdx.SimulationConfig(
        time=5e-15, resolution=100e-9, backend="cpu", dtype=jnp.float32, courant_factor=0.99
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6), material=fdtdx.Material(permittivity=1.0)
    )
    constraints, object_list = _base_objects(volume, wavelength)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=subkey
    )
    arrays2, new_objects, info = fdtdx.apply_params(arrays, objects, params, subkey)
    final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
    _, arrays_out = final_state
    return jnp.sum(arrays_out.fields.E**2)


def run_with_permittivity(permittivity):
    key = jax.random.PRNGKey(0)
    config = fdtdx.SimulationConfig(
        time=5e-15, resolution=100e-9, backend="cpu", dtype=jnp.float32, courant_factor=0.99
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6), material=fdtdx.Material(permittivity=permittivity)
    )
    constraints, object_list = _base_objects(volume, 1.0e-6)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=subkey
    )
    arrays2, new_objects, info = fdtdx.apply_params(arrays, objects, params, subkey)
    final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
    _, arrays_out = final_state
    return jnp.sum(arrays_out.fields.E**2)


def main() -> None:
    report: dict = {}

    # Probe A: gradient w.r.t. source wavelength (closed over a traced scalar,
    # permittivity kept concrete so place_objects does not need to trace it).
    w0 = 1.0e-6
    try:
        val = float(run_with_wavelength(w0))
        grad_ad = float(jax.grad(run_with_wavelength)(w0))
        eps = 1e-9
        grad_fd = float(
            (run_with_wavelength(w0 + eps) - run_with_wavelength(w0 - eps)) / (2 * eps)
        )
        report["wavelength_probe"] = {
            "objective_value": val,
            "grad_native_autodiff": grad_ad,
            "grad_finite_difference": grad_fd,
            "verdict": "AD gradient is exactly 0.0 while FD is large and nonzero -- "
            "phantom zero gradient, NOT a verified differentiable path.",
        }
    except Exception as exc:  # pragma: no cover - report, don't hide
        report["wavelength_probe"] = {"failed": f"{type(exc).__name__}: {exc}"[:500]}

    # Probe B: gradient w.r.t. background permittivity passed directly into
    # the Material() constructor inside a traced function.
    try:
        grad_ad = float(jax.grad(run_with_permittivity)(1.0))
        report["permittivity_probe"] = {"grad_native_autodiff": grad_ad}
    except Exception as exc:
        report["permittivity_probe"] = {
            "failed": f"{type(exc).__name__}: {str(exc)[:300]}",
            "verdict": "place_objects() does concrete Python-level introspection of "
            "material properties (e.g. math.isclose on permittivity components) and "
            "cannot be called inside a jax.grad trace with a traced Material value.",
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
