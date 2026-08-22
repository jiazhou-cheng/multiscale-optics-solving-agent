"""CHE-32 (M3.3): record the exit-pupil handoff export at the M1 standard.

Two things need evidence before M3.3 can close, and they share a run, so they
share a probe:

1. ``M3SingletRef`` -- the adapter-owned diffraction-limited system -- must be
   pinned to the same standard as every bundled system already is: a
   deterministic trace, finite output, unit-norm direction cosines, and a stable
   scientific-array hash, all read through the ordinary export path
   (``OptilandAdapter.run`` -> ``_build_ray_bundle_artifact``) rather than a
   separate custom-prescription path, which stays refused.
2. ``handoff_plane="exit_pupil"`` must reproduce the exit-pupil geometry frozen
   in ``benchmarks/slice_protocol.yaml`` for both M3 systems, and the projection
   it performs must be checked as *geometry* rather than by re-running the
   formula that produced it.

For (2) the check recorded here is collinearity: the exported pupil-plane point
must lie on the traced ray's own image-space line. That is an independent
invariant of the construction -- recomputing ``x + L*(z_target - z)/N`` would
only restate the implementation. Collinearity plus "z is exactly the pupil
plane" plus "directions are byte-identical to the image-surface run" pins the
projection completely.

What the exported positions MEAN at the exit pupil is recorded by the adapter
itself (``conventions.exit_pupil.position_semantics``): they are each ray's
image-space asymptote, not a physical intersection, because the pupil is
virtual on both M3 systems. This probe records ``is_virtual`` for both so that
the claim is evidence and not a remark.

Run inside the agent_solver container:

    ./run.sh python knowledge/solvers/optiland/probes/exit_pupil_handoff.py

The expected regression fixture is generated only by this executable:

    ./run.sh python knowledge/solvers/optiland/probes/exit_pupil_handoff.py \
        --write-expected knowledge/solvers/optiland/expected/exit_pupil_handoff.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adapters.base import ModelRunRequest, RunStatus
from adapters.optiland_adapter import OptilandAdapter

ROOT = Path(__file__).resolve().parents[4]
SLICE_PROTOCOL = ROOT / "benchmarks" / "slice_protocol.yaml"

SYSTEMS = {
    "ReverseTelephoto": "M3-REVERSE-TELEPHOTO",
    "M3SingletRef": "M3-SINGLET-REF",
}
PLANES = ("image_surface", "exit_pupil")

REQUEST_COMMON = {
    "backend": "numpy",
    "device": "cpu",
    "dtype": "float64",
    "wavelength": 0.55,
    "Hx": 0.0,
    "Hy": 0.0,
    "num_rays": 16,
}


def _run(adapter: OptilandAdapter, sample: str, plane: str, output_directory: Path):
    request = ModelRunRequest(
        run_id="che32-exit-pupil-handoff",
        node_id="lens",
        inputs={},
        config={
            **REQUEST_COMMON,
            "sample": sample,
            "handoff_plane": plane,
            "output_directory": str(output_directory),
        },
        design_parameters={},
        require_gradients=False,
    )
    return adapter.run(request)


def _stable_export(result) -> dict[str, Any]:
    """Everything about an export that must not drift, and nothing that must."""
    artifact = result.outputs["rays"]
    conventions = artifact.metadata["conventions"]
    exit_pupil = conventions["exit_pupil"]
    boundary = artifact.metadata["pupil_boundary"]
    return {
        "surviving_ray_count": int(artifact.shape[0]),
        "dtype": artifact.dtype,
        "scientific_array_sha256": artifact.metadata["scientific_array_sha256"],
        "summary_metrics": artifact.metadata["summary_metrics"],
        "handoff_plane": conventions["handoff_plane"],
        "reference_plane": conventions["reference_plane"],
        "reference_plane_z_m": conventions["reference_plane_z_m"],
        "exit_pupil": (
            None
            if exit_pupil is None
            else {
                "source": exit_pupil["source"],
                "location_from_image_m": exit_pupil["location_from_image_m"],
                "z_m": exit_pupil["z_m"],
                "diameter_m": exit_pupil["diameter_m"],
                "is_virtual": exit_pupil["is_virtual"],
                "refracting_surfaces_beyond_pupil_z_m": exit_pupil[
                    "refracting_surfaces_beyond_pupil_z_m"
                ],
                "max_projection_step_m": exit_pupil["max_projection_step_m"],
            }
        ),
        "pupil_boundary": {
            key: boundary[key]
            for key in (
                "representation",
                "mask_available_from_optiland",
                "measured_semi_extent_x_m",
                "measured_semi_extent_y_m",
                "paraxial_semi_diameter_m",
            )
        },
    }


def _projection_geometry(image_surface_result, exit_pupil_result) -> dict[str, Any]:
    """Is the pupil-plane point on the traced ray's own line, and only moved along it?"""
    at_image = np.load(image_surface_result.outputs["rays"].uri)
    at_pupil = np.load(exit_pupil_result.outputs["rays"].uri)

    direction = np.stack([at_image["L"], at_image["M"], at_image["N"]], axis=1)
    start = np.stack([at_image["x_m"], at_image["y_m"], at_image["z_m"]], axis=1)
    end = np.stack([at_pupil["x_m"], at_pupil["y_m"], at_pupil["z_m"]], axis=1)

    displacement = end - start
    step = np.linalg.norm(displacement, axis=1)
    # || d x (P_pupil - P_image) || is zero exactly when the two points and the
    # ray direction are collinear. Normalized by the step so it reads as an
    # angle-like residual rather than growing with the projection distance.
    perpendicular = np.linalg.norm(np.cross(direction, displacement), axis=1)
    nonzero = step > 0.0

    direction_at_pupil = np.stack([at_pupil["L"], at_pupil["M"], at_pupil["N"]], axis=1)
    return {
        "collinearity_residual_max": float(
            np.max(perpendicular[nonzero] / step[nonzero]) if np.any(nonzero) else 0.0
        ),
        "direction_max_abs_change": float(np.max(np.abs(direction_at_pupil - direction))),
        "directions_bitwise_identical": bool(np.array_equal(direction_at_pupil, direction)),
        "projected_z_is_single_valued": bool(np.all(at_pupil["z_m"] == at_pupil["z_m"][0])),
        "projected_z_m": float(at_pupil["z_m"][0]),
        "max_step_m": float(np.max(step)),
        "opd_native_unchanged": bool(
            np.array_equal(at_pupil["opd_native"], at_image["opd_native"])
        ),
        "intensity_unchanged": bool(np.array_equal(at_pupil["intensity"], at_image["intensity"])),
    }


def _protocol_cross_check(
    protocol_system: dict[str, Any], export: dict[str, Any]
) -> dict[str, Any]:
    """Compare the exported plane against the numbers frozen by M3.2, in metres."""
    frozen_z_m = float(protocol_system["exit_pupil_z_mm"]) * 1e-3
    frozen_diameter_m = float(protocol_system["exit_pupil_diameter_mm"]) * 1e-3
    frozen_image_z_m = float(protocol_system["image_plane_z_mm"]) * 1e-3

    exported_z_m = export["exit_pupil"]["z_m"]
    exported_diameter_m = export["exit_pupil"]["diameter_m"]
    exported_image_z_m = exported_z_m - export["exit_pupil"]["location_from_image_m"]

    def _relative(observed: float, frozen: float) -> float:
        return abs(observed - frozen) / abs(frozen)

    return {
        "frozen_exit_pupil_z_m": frozen_z_m,
        "exported_exit_pupil_z_m": exported_z_m,
        "exit_pupil_z_relative_error": _relative(exported_z_m, frozen_z_m),
        "frozen_exit_pupil_diameter_m": frozen_diameter_m,
        "exported_exit_pupil_diameter_m": exported_diameter_m,
        "exit_pupil_diameter_relative_error": _relative(exported_diameter_m, frozen_diameter_m),
        "frozen_image_plane_z_m": frozen_image_z_m,
        "exported_image_plane_z_m": exported_image_z_m,
        "image_plane_z_relative_error": _relative(exported_image_z_m, frozen_image_z_m),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", type=Path)
    args = parser.parse_args()

    protocol = yaml.safe_load(SLICE_PROTOCOL.read_text())
    frozen_systems = {entry["id"]: entry["derived"] for entry in protocol["systems"]}

    adapter = OptilandAdapter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "exit_pupil_handoff",
        "issue": "CHE-32 (M3.3)",
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "request_common": dict(REQUEST_COMMON),
        "systems": {},
    }
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="che32-exit-pupil-") as tmp:
        root = Path(tmp)
        for sample, protocol_id in SYSTEMS.items():
            entry: dict[str, Any] = {"protocol_system_id": protocol_id, "planes": {}}
            results = {}
            for plane in PLANES:
                first = _run(adapter, sample, plane, root / sample / plane / "a")
                second = _run(adapter, sample, plane, root / sample / plane / "b")
                for label, result in (("a", first), ("b", second)):
                    if result.status is not RunStatus.SUCCEEDED:
                        failures.append(f"{sample}/{plane}/{label}: {result.error_message}")
                if first.status is not RunStatus.SUCCEEDED:
                    continue

                stable_first = _stable_export(first)
                stable_second = _stable_export(second)
                stable_first["deterministic"] = stable_first == stable_second
                if not stable_first["deterministic"]:
                    failures.append(f"{sample}/{plane}: two identical requests did not agree")
                entry["planes"][plane] = stable_first
                results[plane] = first

                if "package_version" not in report:
                    report["package_version"] = first.diagnostics["package_version"]

            if set(results) == set(PLANES):
                entry["projection_geometry"] = _projection_geometry(
                    results["image_surface"], results["exit_pupil"]
                )
                entry["protocol_cross_check"] = _protocol_cross_check(
                    frozen_systems[protocol_id], entry["planes"]["exit_pupil"]
                )
            report["systems"][sample] = entry

    for sample, entry in report["systems"].items():
        for plane, export in entry["planes"].items():
            metrics = export["summary_metrics"]
            if not metrics["all_finite"]:
                failures.append(f"{sample}/{plane}: non-finite scientific output")
            if metrics["max_direction_norm_error"] > metrics["direction_norm_tolerance"]:
                failures.append(f"{sample}/{plane}: direction cosines are not unit norm")
        cross_check = entry.get("protocol_cross_check", {})
        for key, value in cross_check.items():
            if key.endswith("_relative_error") and value > 1e-12:
                failures.append(f"{sample}: {key} = {value:.3e} exceeds 1e-12")

    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    if args.write_expected and report["status"] == "passed":
        args.write_expected.parent.mkdir(parents=True, exist_ok=True)
        args.write_expected.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
