"""Import/initialization probe for the M_RCWA_FMMAX solver card.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \\
        python knowledge/solvers/fmmax/probes/import_probe.py

Verifies that fmmax (a real PyPI package, no namesquat issue) imports
correctly and reports the environment actually used.
"""

from __future__ import annotations

import json

import fmmax
import jax


def main() -> None:
    report = {
        "fmmax_version": getattr(fmmax, "__version__", "unknown"),
        "fmmax_file": fmmax.__file__,
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "api_sample": sorted(
            n for n in dir(fmmax) if not n.startswith("_")
        )[:15],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
