"""Import/initialization probe for the M_THERMAL_JAX_FEM solver card.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/jax_fem/probes/import_probe.py

`jax_fem` prints an ASCII-art banner to stdout on import (via its
top-level `Figlet`/`f` exports) -- this is real, observed behavior, not a
bug in this probe. The top-level package exposes only logging/banner
utilities; the documented `Problem`/`Mesh`/`solver()` entry points live in
submodules (jax_fem.problem, jax_fem.generate_mesh, jax_fem.solver).
"""

from __future__ import annotations

import json

import jax_fem
import jax_fem.generate_mesh as gm
import jax_fem.problem as p


def main() -> None:
    report = {
        "jax_fem_file": jax_fem.__file__,
        "jax_fem_top_level_exports": sorted(
            n for n in dir(jax_fem) if not n.startswith("_")
        ),
        "jax_fem_problem_exports": sorted(
            n for n in dir(p) if not n.startswith("_")
        ),
        "jax_fem_generate_mesh_exports": sorted(
            n for n in dir(gm) if not n.startswith("_")
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
