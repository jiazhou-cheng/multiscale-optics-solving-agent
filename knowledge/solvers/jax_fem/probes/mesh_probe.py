"""Minimal mesh-generation probe for M_THERMAL_JAX_FEM.

Confirms that `jax_fem.problem` (Problem/Mesh/FiniteElement) and
`jax_fem.generate_mesh` (mesh generation via gmsh) both import and run
without petsc4py. This is the only part of jax-fem's advertised workflow
that is currently usable in this container -- see solver_failure_probe.py
for the part that is NOT usable.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/jax_fem/probes/mesh_probe.py
"""

from __future__ import annotations

import json

import jax_fem.generate_mesh as gm


def main() -> None:
    mesh = gm.rectangle_mesh(Nx=4, Ny=4, domain_x=1.0, domain_y=1.0)
    report = {
        "mesh_type": type(mesh).__module__ + "." + type(mesh).__qualname__,
        "points_shape": list(mesh.points.shape),
        "cell_block_keys": list(mesh.cells_dict.keys()),
        "num_quads": list(mesh.cells_dict["quad"].shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
