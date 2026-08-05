"""Import/initialization probe for the M_WAVE_CHROMATIX solver card.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/chromatix/probes/import_probe.py

Verifies that the real chromatix-team/chromatix package (installed from
GitHub, pinned in docker/requirements-chromatix.txt) imports correctly and
reports the environment actually used, rather than assuming any of this.
"""

from __future__ import annotations

import json

import chromatix
import chromatix.functional as cf
import jax


def main() -> None:
    report = {
        "chromatix_file": chromatix.__file__,
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "functional_api_sample": sorted(
            n
            for n in dir(cf)
            if not n.startswith("_") and callable(getattr(cf, n))
        )[:10],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
