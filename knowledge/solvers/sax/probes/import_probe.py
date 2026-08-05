"""Import/initialization probe for M_CIRCUIT_SAX.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/sax/probes/import_probe.py

Verifies that `pip install sax` (a real PyPI package, unlike chromatix)
installs the intended gdsfactory-team library and reports the environment
actually used.
"""

from __future__ import annotations

import json

import jax
import sax


def main() -> None:
    report = {
        "sax_version": sax.__version__,
        "sax_file": sax.__file__,
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "top_level_sample": sorted(
            n for n in dir(sax) if not n.startswith("_") and callable(getattr(sax, n))
        )[:10],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
