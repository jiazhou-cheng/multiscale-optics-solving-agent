"""Minimal directional-derivative probe for M_RAY_OPTILAND, torch backend.

This is a probe, not the repository gradient test (which needs multiple step
sizes, a convergence table, and a
deliberately ill-conditioned case). It establishes that torch autograd
flows through `Optic.trace()` for one parameter path: a lens surface's
radius of curvature -> ray trace -> RMS spot size at the origin field
point. torch is NOT a declared optiland dependency (see failure_guide.md)
and must be installed separately -- this probe will fail with
ImportError if it is not.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \
        python benchmarks/probes/optiland/gradient_probe.py
"""

from __future__ import annotations

import json

import optiland.backend as be
import torch
from optiland.samples.objectives import ReverseTelephoto

R0 = 1.6911
WAVELENGTH = 0.55
NUM_RAYS = 64


def rms_spot(radius_value: torch.Tensor) -> torch.Tensor:
    lens = ReverseTelephoto()
    # Optic.surface_group is deprecated in this version; use Optic.surfaces.
    surf = lens.surfaces.surfaces[1]
    surf.geometry.radius = radius_value
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH, num_rays=NUM_RAYS)
    return (rays.x**2 + rays.y**2).mean()


def main() -> None:
    be.set_backend("torch")

    r0 = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
    value = rms_spot(r0)
    value.backward()
    grad_ad = r0.grad.item()

    eps = 1e-4
    with torch.no_grad():
        v_plus = rms_spot(torch.tensor(R0 + eps, dtype=torch.float64))
        v_minus = rms_spot(torch.tensor(R0 - eps, dtype=torch.float64))
    grad_fd = (v_plus.item() - v_minus.item()) / (2 * eps)

    report = {
        "parameter": "surfaces.surfaces[1].geometry.radius (ReverseTelephoto sample lens)",
        "objective": "mean(x^2 + y^2) over traced rays at Hx=Hy=0 (RMS spot proxy)",
        "r0": R0,
        "fd_step": eps,
        "objective_value": value.item(),
        "grad_native_autodiff": grad_ad,
        "grad_finite_difference": grad_fd,
        "relative_error": abs(grad_ad - grad_fd) / abs(grad_fd),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
