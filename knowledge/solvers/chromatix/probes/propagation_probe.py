"""Minimal forward-simulation probe for M_WAVE_CHROMATIX.

Exercises: field construction (plane_wave, point_source), array
shape/dtype/axis conventions, and both supported free-space propagators
(transform_propagate = single-FFT Fresnel, asm_propagate = angular
spectrum). Captures real numbers rather than assumed ones so the solver
card and conventions notes are grounded in observed behavior of the pinned
commit (docker/requirements-chromatix.txt).

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/chromatix/probes/propagation_probe.py
"""

from __future__ import annotations

import json

import jax.numpy as jnp

import chromatix.functional as cf
from chromatix.functional.propagation import compute_padding_transfer


def _describe(field) -> dict:
    return {
        "u_shape": list(field.u.shape),
        "u_dtype": str(field.u.dtype),
        "dx": jnp.asarray(field.dx).tolist(),
        "wavelength": jnp.asarray(field.wavelength).tolist(),
        "power": float(jnp.asarray(field.power).sum()),
    }


def main() -> None:
    shape = (128, 128)
    dx = 0.3
    wavelength = 0.532
    n = 1.0

    report: dict = {"shape": list(shape), "dx": dx, "wavelength": wavelength, "n": n}

    plane = cf.plane_wave(shape=shape, dx=dx, spectrum=wavelength, power=1.0)
    report["plane_wave"] = _describe(plane)

    z_fresnel = 500.0
    fresnel = cf.transform_propagate(plane, z=z_fresnel, n=n, pad_width=32)
    report["transform_propagate"] = {
        "z": z_fresnel,
        **_describe(fresnel),
    }

    z_asm = 50.0
    pad = compute_padding_transfer(shape[0], wavelength, dx, z_asm)
    asm = cf.asm_propagate(plane, z=z_asm, n=n, pad_width=int(pad))
    report["asm_propagate"] = {
        "z": z_asm,
        "computed_pad_width": int(pad),
        **_describe(asm),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
