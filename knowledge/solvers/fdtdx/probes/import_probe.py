"""Import/initialization probe for M_EM_FDTDX.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/fdtdx/probes/import_probe.py
"""

from __future__ import annotations

import json

import fdtdx
import jax


def main() -> None:
    report = {
        "fdtdx_file": fdtdx.__file__,
        "fdtdx_version": getattr(fdtdx, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "top_level_api_sample": sorted(
            n for n in dir(fdtdx) if not n.startswith("_")
        )[:15],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
