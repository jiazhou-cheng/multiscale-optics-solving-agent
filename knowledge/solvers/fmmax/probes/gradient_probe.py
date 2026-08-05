"""Directional-derivative probe for M_RCWA_FMMAX.

One narrow path: reflectance |r|^2 of a bare ambient/substrate interface,
differentiated via jax.grad w.r.t. the substrate refractive index, checked
against a centered finite difference. This is NOT the full CLAUDE.md
section 6.2 gradient test (needs multiple step sizes, a convergence table,
and an ill-conditioned case) -- it only establishes that autodiff produces
a finite, FD-consistent value for this one parameter.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \\
        python knowledge/solvers/fmmax/probes/gradient_probe.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import fmmax

WAVELENGTH = 0.55
N_AMBIENT = 1.0


def reflectance(n_substrate: jnp.ndarray) -> jnp.ndarray:
    in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength=jnp.asarray(WAVELENGTH),
        polar_angle=jnp.asarray(0.0),
        azimuthal_angle=jnp.asarray(0.0),
        permittivity=jnp.asarray(N_AMBIENT**2),
    )
    primitive_lattice_vectors = fmmax.LatticeVectors(u=fmmax.X, v=fmmax.Y)
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=1,
        truncation=fmmax.Truncation.CIRCULAR,
    )
    perms = [
        jnp.asarray(N_AMBIENT**2)[..., None, None],
        (n_substrate**2)[..., None, None],
    ]
    layer_solve_results = [
        fmmax.eigensolve_isotropic_media(
            wavelength=jnp.asarray(WAVELENGTH),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=fmmax.Formulation.FFT,
        )
        for p in perms
    ]
    s_matrix = fmmax.stack_s_matrix(
        layer_solve_results, [jnp.asarray(0.0), jnp.asarray(0.0)]
    )
    r_te = s_matrix.s21[..., 0, 0]
    return jnp.abs(r_te) ** 2


def main() -> None:
    n0 = jnp.asarray(1.5)
    value = float(reflectance(n0))
    grad_ad = float(jax.grad(reflectance)(n0))
    eps = 1e-3
    grad_fd = float((reflectance(n0 + eps) - reflectance(n0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / abs(grad_fd)

    report = {
        "parameter": "substrate refractive index (bare interface reflectance)",
        "n_substrate_0": float(n0),
        "fd_step": eps,
        "objective_value": value,
        "grad_native_autodiff": grad_ad,
        "grad_finite_difference": grad_fd,
        "relative_error": relative_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
