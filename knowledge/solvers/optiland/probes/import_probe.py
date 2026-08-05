"""Import/initialization probe for the M_RAY_OPTILAND solver card.

Run inside the agent_solver container:
    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \
        python knowledge/solvers/optiland/probes/import_probe.py

Verifies that optiland (a real PyPI package, unlike chromatix) imports
correctly, that its top-level namespace is nearly empty (functionality
lives in submodules), and reports the real backend-abstraction state
without assuming torch is installed or selected.
"""

from __future__ import annotations

import json

import optiland
import optiland.backend as be


def main() -> None:
    report = {
        "optiland_file": optiland.__file__,
        "optiland_version": getattr(optiland, "__version__", "unknown"),
        "top_level_dir": [n for n in dir(optiland) if not n.startswith("_")],
        "default_backend": be.get_backend(),
        "available_backends": be.list_available_backends(),
        "supports_gpu_default": be.supports_gpu,
        "supports_gradients_default": be.supports_gradients,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
