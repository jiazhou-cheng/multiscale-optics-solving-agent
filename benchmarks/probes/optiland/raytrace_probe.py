"""Minimal ray-trace probe for M_RAY_OPTILAND, numpy backend (default).

Exercises a bundled sample lens system (no torch, no custom prescription)
to confirm basic ray tracing and paraxial analysis work out of the box, and
to record real return types/shapes rather than assumed ones.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \
        python benchmarks/probes/optiland/raytrace_probe.py
"""

from __future__ import annotations

import json

from optiland.samples.objectives import ReverseTelephoto


def main() -> None:
    lens = ReverseTelephoto()
    rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=16)
    f2 = lens.paraxial.f2()

    report = {
        "lens_class": type(lens).__name__,
        "rays_class": f"{type(rays).__module__}.{type(rays).__name__}",
        "rays_x_shape": list(rays.x.shape),
        "rays_x_dtype": str(rays.x.dtype),
        "paraxial_f2": float(f2),
        "backend_used": "numpy (default; not switched)",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
