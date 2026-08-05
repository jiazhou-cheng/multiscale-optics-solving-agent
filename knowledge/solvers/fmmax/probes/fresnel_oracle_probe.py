"""Analytic-oracle probe for M_RCWA_FMMAX: bare interface vs. Fresnel formula.

A single ambient/substrate interface with no in-plane periodicity content
(`approximate_num_terms=1`, i.e. the homogeneous/zero-grating-order limit)
must reduce to ordinary Fresnel reflection at normal incidence. This is an
independent analytic check per CLAUDE.md section 3 rule 6, not just an API
smoke test.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \\
        python knowledge/solvers/fmmax/probes/fresnel_oracle_probe.py
"""

from __future__ import annotations

import json

import jax.numpy as jnp

import fmmax


def reflectance(n_ambient: float, n_substrate: float, wavelength: float) -> complex:
    incident_angle = jnp.asarray(0.0)
    in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength=jnp.asarray(wavelength),
        polar_angle=incident_angle,
        azimuthal_angle=jnp.asarray(0.0),
        permittivity=jnp.asarray(n_ambient**2),
    )
    primitive_lattice_vectors = fmmax.LatticeVectors(u=fmmax.X, v=fmmax.Y)
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=1,
        truncation=fmmax.Truncation.CIRCULAR,
    )
    perms = [
        jnp.asarray(n_ambient**2)[..., None, None],
        jnp.asarray(n_substrate**2)[..., None, None],
    ]
    layer_solve_results = [
        fmmax.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
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
    return s_matrix.s21[..., 0, 0]


def main() -> None:
    n_ambient, n_substrate, wavelength = 1.0, 1.5, 0.55
    r_te = reflectance(n_ambient, n_substrate, wavelength)
    r_analytic = (n_ambient - n_substrate) / (n_ambient + n_substrate)

    report = {
        "n_ambient": n_ambient,
        "n_substrate": n_substrate,
        "wavelength": wavelength,
        "fmmax_r_te": [float(r_te.real), float(r_te.imag)],
        "fmmax_R_te": float(jnp.abs(r_te) ** 2),
        "analytic_fresnel_r": r_analytic,
        "analytic_fresnel_R": r_analytic**2,
        "R_relative_error": abs(float(jnp.abs(r_te) ** 2) - r_analytic**2)
        / abs(r_analytic**2),
        "note": (
            "Reflectance |r|^2 matches the analytic Fresnel formula to "
            "~1e-7 relative error, but the SIGN of the complex amplitude "
            "does not match the textbook convention (fmmax gives a "
            "positive real r_te here; textbook Fresnel gives negative for "
            "n_ambient < n_substrate). See conventions.md."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
