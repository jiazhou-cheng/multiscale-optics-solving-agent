"""Regression probe: confirms `jax_fem.solver` cannot be imported without
petsc4py in this environment.

This is a *negative* probe: it is expected to fail, and the failure is the
useful signal. `jax_fem.solver` does an unconditional top-level
`from petsc4py import PETSc`, so the entire documented `solver()` entry
point (the actual FEM linear/nonlinear solve, per the official Quickstart
at https://deepmodeling.github.io/jax-fem/guide/Quickstart.html) is
unusable here. petsc4py has no prebuilt PyPI wheel (source-dist only,
requires compiling PETSc) and is deliberately not installed in this
lightweight pip-based image -- see failure_guide.md for follow-up options.

Run this probe on every refresh (package-refresh policy):
if it ever reports "UNEXPECTED: import succeeded", the blocker has been
resolved (e.g. a future jax-fem release made petsc4py optional, or the
image gained a working petsc4py) and solver_card.yaml / capability_notes.md
should be updated to reflect that the solve path is usable.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/jax_fem/probes/solver_failure_probe.py
"""

from __future__ import annotations

import json


def main() -> None:
    try:
        import jax_fem.solver  # noqa: F401

        report = {"status": "UNEXPECTED: import succeeded", "blocked": False}
    except ModuleNotFoundError as exc:
        report = {
            "status": "EXPECTED FAILURE",
            "blocked": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
