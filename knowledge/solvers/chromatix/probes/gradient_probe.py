"""Minimal directional-derivative probe for M_WAVE_CHROMATIX.

This is a probe, not the repository gradient test required by
CLAUDE.md section 6.2 (that needs multiple step sizes, a convergence
table, and a deliberately ill-conditioned case). It only establishes that
`jax.grad` produces a finite value that agrees with a single centered
finite-difference estimate for one parameter path: lens focal length ->
Fresnel (transform_propagate) -> intensity objective.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/chromatix/probes/gradient_probe.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import chromatix.functional as cf

SHAPE = (64, 64)
DX = 1.0
WAVELENGTH = 0.532
N = 1.0


def objective(f: float) -> jnp.ndarray:
    field = cf.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH, power=1.0)
    field = cf.thin_lens(field, f=f, n=N)
    field = cf.transform_propagate(field, z=f, n=N, pad_width=32)
    return jnp.sum(field.intensity)


def main() -> None:
    f0 = 1000.0
    eps = 0.5

    value = float(objective(f0))
    grad_ad = float(jax.grad(objective)(f0))
    grad_fd = float((objective(f0 + eps) - objective(f0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / abs(grad_fd)

    report = {
        "parameter": "thin_lens focal length f (through transform_propagate)",
        "f0": f0,
        "fd_step": eps,
        "objective_value": value,
        "grad_native_autodiff": grad_ad,
        "grad_finite_difference": grad_fd,
        "relative_error": relative_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
