"""Generate and verify the deterministic CHE-13 standalone ray baseline.

Run inside the supported container entry point:

    ./run.sh python knowledge/solvers/optiland/probes/standalone_baseline.py \
        --output-dir /tmp/optiland-che13

The expected regression fixture is generated only by this executable:

    ./run.sh python knowledge/solvers/optiland/probes/standalone_baseline.py \
        --output-dir /tmp/optiland-che13 \
        --write-expected knowledge/solvers/optiland/expected/standalone_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from solvers.base import RunStatus
from solvers.optiland.adapter import (
    OptilandAdapter,
    OptilandRayRequest,
)


def _stable_result(result) -> dict:
    return {
        "package_version": result.package_version,
        "backend": result.backend,
        "device": result.device,
        "dtype": result.dtype,
        "requested_sampling": result.requested_sampling,
        "surviving_ray_count": result.surviving_ray_count,
        "scientific_array_sha256": result.scientific_array_sha256,
        "summary_metrics": result.summary_metrics,
    }


def _forbidden_modules_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == "chromatix"
        or name.startswith("chromatix.")
        or name.startswith("couplers")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-expected", type=Path)
    args = parser.parse_args()

    adapter = OptilandAdapter()
    common = {
        "prescription": "ReverseTelephoto",
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        "wavelength_um": 0.55,
        "field_hx": 0.0,
        "field_hy": 0.0,
        "pupil_sampling": 16,
        "seed": 20260811,
        "require_gradients": False,
    }
    first = adapter.run_standalone(
        OptilandRayRequest(**common, output_directory=args.output_dir / "run_a")
    )
    second = adapter.run_standalone(
        OptilandRayRequest(**common, output_directory=args.output_dir / "run_b")
    )

    failures = [r.failure.model_dump(mode="json") for r in (first, second) if r.failure]
    if any(r.status is not RunStatus.SUCCEEDED for r in (first, second)):
        report = {
            "schema_version": 1,
            "status": "failed",
            "failures": failures,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    stable_first = _stable_result(first)
    stable_second = _stable_result(second)
    report = {
        "schema_version": 1,
        "status": "passed",
        "request": common,
        "deterministic": stable_first == stable_second,
        "stable_result": stable_first,
        "runtime_seconds": [first.runtime_seconds, second.runtime_seconds],
        "runtime_under_10_seconds": all(
            runtime is not None and runtime < 10.0
            for runtime in (first.runtime_seconds, second.runtime_seconds)
        ),
        "forbidden_modules_loaded": _forbidden_modules_loaded(),
    }
    if (
        not report["deterministic"]
        or not report["runtime_under_10_seconds"]
        or report["forbidden_modules_loaded"]
    ):
        report["status"] = "failed"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "determinism_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.write_expected:
        args.write_expected.parent.mkdir(parents=True, exist_ok=True)
        expected = {key: value for key, value in report.items() if key != "runtime_seconds"}
        args.write_expected.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
