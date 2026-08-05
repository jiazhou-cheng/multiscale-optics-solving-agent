"""Minimal directional-derivative probe for M_CIRCUIT_SAX.

This is a probe, not the repository gradient test required by CLAUDE.md
section 6.2. It only establishes that `jax.grad` produces a finite value
that agrees with a centered finite-difference estimate for one parameter
path: `coupler_ideal` coupling ratio -> thru-port power. This model is a
closed-form trigonometric expression, so tight agreement here is expected
and does NOT by itself demonstrate differentiability through a full
assembled `sax.circuit` (matrix-solve) path, which has not been probed.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/sax/probes/gradient_probe.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import sax
import sax.models as sm


def thru_power(coupling: float) -> jnp.ndarray:
    s = sm.coupler_ideal(wl=1.55, coupling=coupling)
    return jnp.abs(s[("o1", "o4")]) ** 2


def main() -> None:
    sax.set_port_naming_strategy("optical")

    c0 = 0.3
    eps = 1e-4

    value = float(thru_power(c0))
    grad_ad = float(jax.grad(thru_power)(c0))
    grad_fd = float((thru_power(c0 + eps) - thru_power(c0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / abs(grad_fd)

    report = {
        "parameter": "coupler_ideal coupling (closed-form model, not a full circuit)",
        "c0": c0,
        "fd_step": eps,
        "objective_value": value,
        "grad_native_autodiff": grad_ad,
        "grad_finite_difference": grad_fd,
        "relative_error": relative_error,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
