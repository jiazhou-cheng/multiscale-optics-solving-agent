"""Verify the pinned M1 ray/wave engines and their CPU execution boundary.

Run only through the project container::

    ./run.sh python benchmarks/probes/verify_m1_engines.py --engine all

The ray and wave modes deliberately import only their own solver family.  A
failed import or pin check produces a structured ``blocked`` report and a
non-zero exit status; the probe never substitutes expected values for missing
solver output.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

EXPECTED = {
    "optiland": "0.6.0",
    "chromatix": "0.6.0",
    "chromatix_commit": "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee",
    "torch": "2.13.0",
    "jax": "0.6.2",
    "numpy": "2.2.6",
}


def _version_without_local_suffix(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return None


def _cpu_environment() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "affinity_cpu_count": len(affinity) if affinity is not None else None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "process_id": os.getpid(),
    }


def _loaded_forbidden_modules(prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    )


def _known_distance_report() -> dict[str, Any]:
    import numpy as np
    import optiland.backend as be
    import optiland.paraxial as optiland_paraxial
    from optiland.optic import Optic

    be.set_backend("numpy")
    distance_mm = 10.0
    wavelength_um = 0.55

    optic = Optic("m1-known-distance")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=be.inf,
        thickness=distance_mm,
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=wavelength_um, is_primary=True)

    rays = optic.trace(
        Hx=0.0,
        Hy=0.0,
        wavelength=wavelength_um,
        num_rays=3,
    )
    z = np.asarray(rays.z, dtype=np.float64)
    wavelength = np.asarray(rays.w, dtype=np.float64)
    direction_norm = np.sqrt(
        np.asarray(rays.L, dtype=np.float64) ** 2
        + np.asarray(rays.M, dtype=np.float64) ** 2
        + np.asarray(rays.N, dtype=np.float64) ** 2
    )

    # The pinned source explicitly labels a coordinate offset as millimetres.
    # The trace below then proves that a 10-unit surface separation produces
    # a final z coordinate of 10 in the same native geometry coordinates.
    source = inspect.getsource(optiland_paraxial)
    source_declares_mm = "10 mm before first surface" in source
    trace_doc = inspect.getdoc(Optic.trace) or ""
    trace_declares_microns = "micron" in trace_doc.lower()
    max_z_error = float(np.max(np.abs(z - distance_mm)))
    max_wavelength_error = float(np.max(np.abs(wavelength - wavelength_um)))
    max_direction_norm_error = float(np.max(np.abs(direction_norm - 1.0)))
    unit_verified = bool(
        source_declares_mm
        and trace_declares_microns
        and max_z_error <= 1e-12
        and max_wavelength_error <= 1e-12
        and max_direction_norm_error <= 1e-12
    )

    return {
        "status": "verified" if unit_verified else "unverified",
        "native_geometry_length_unit": "mm" if unit_verified else "unverified",
        "native_wavelength_unit": "um",
        "native_geometry_to_si_scale_m": 1e-3 if unit_verified else None,
        "native_wavelength_to_si_scale_m": 1e-6,
        "declared_surface_separation_mm": distance_mm,
        "expected_surface_separation_m": distance_mm * 1e-3,
        "observed_final_z_native_unique": sorted(float(value) for value in np.unique(z)),
        "max_z_error_native": max_z_error,
        "traced_wavelength_um": float(wavelength[0]),
        "max_wavelength_error_um": max_wavelength_error,
        "max_direction_norm_error": max_direction_norm_error,
        "ray_count": int(z.size),
        "source_unit_evidence": {
            "geometry": {
                "module": "optiland.paraxial",
                "marker": "10 mm before first surface",
                "found": source_declares_mm,
            },
            "wavelength": {
                "callable": "optiland.optic.Optic.trace",
                "marker": "micron",
                "found": trace_declares_microns,
            },
        },
    }


def _probe_ray() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    try:
        import numpy as np
        import optiland
        import optiland.backend as be
        import torch

        versions = {
            "optiland_distribution": metadata.version("optiland"),
            "optiland_module": getattr(optiland, "__version__", "unknown"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        pin_checks = {
            "optiland": versions["optiland_distribution"] == EXPECTED["optiland"],
            "torch": _version_without_local_suffix(versions["torch"]) == EXPECTED["torch"],
            "numpy": versions["numpy"] == EXPECTED["numpy"],
        }
        known_distance = _known_distance_report()
        forbidden = _loaded_forbidden_modules(("chromatix", "couplers"))
        if not all(pin_checks.values()):
            errors.append(
                {
                    "code": "RAY_PIN_MISMATCH",
                    "message": "Installed ray-stack versions do not match the M1 pins.",
                    "details": {"expected": EXPECTED, "observed": versions},
                }
            )
        if known_distance["status"] != "verified":
            errors.append(
                {
                    "code": "OPTILAND_LENGTH_UNIT_UNVERIFIED",
                    "message": "The executable Optiland millimetre-unit check failed.",
                    "details": known_distance,
                }
            )
        if forbidden:
            errors.append(
                {
                    "code": "RAY_BASELINE_IMPORT_LEAK",
                    "message": "The ray-only probe loaded a wave engine or coupler module.",
                    "details": {"modules": forbidden},
                }
            )

        return (
            {
                "engine": "M_RAY_OPTILAND",
                "versions": versions,
                "pin_checks": pin_checks,
                "backend": {
                    "selected": be.get_backend(),
                    "available": be.list_available_backends(),
                    "supports_gpu": bool(be.supports_gpu),
                    "supports_gradients": bool(be.supports_gradients),
                    "torch_cuda_available": bool(torch.cuda.is_available()),
                    "torch_num_threads": int(torch.get_num_threads()),
                },
                "known_distance": known_distance,
                "forbidden_modules_loaded": forbidden,
            },
            errors,
        )
    except Exception as exc:  # structured blocker; do not invent solver output
        return {}, [
            {
                "code": "RAY_ENGINE_PROBE_FAILED",
                "message": str(exc),
                "details": {"exception_type": type(exc).__name__},
            }
        ]


def _probe_wave() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    try:
        import chromatix
        import jax
        import numpy as np

        distribution = metadata.distribution("chromatix")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else {}
        vcs_info = direct_url.get("vcs_info", {})
        versions = {
            "chromatix_distribution": distribution.version,
            "chromatix_module": getattr(chromatix, "__version__", "unknown"),
            "chromatix_direct_url": direct_url,
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
        pin_checks = {
            "chromatix": versions["chromatix_distribution"] == EXPECTED["chromatix"],
            "chromatix_commit": vcs_info.get("commit_id") == EXPECTED["chromatix_commit"],
            "chromatix_requested_revision": (
                vcs_info.get("requested_revision") == EXPECTED["chromatix_commit"]
            ),
            "jax": versions["jax"] == EXPECTED["jax"],
            "numpy": versions["numpy"] == EXPECTED["numpy"],
        }
        forbidden = _loaded_forbidden_modules(("optiland", "couplers"))
        if not all(pin_checks.values()):
            errors.append(
                {
                    "code": "WAVE_PIN_MISMATCH",
                    "message": "Installed wave-stack versions do not match the M1 pins.",
                    "details": {"expected": EXPECTED, "observed": versions},
                }
            )
        if jax.default_backend() != "cpu":
            errors.append(
                {
                    "code": "WAVE_DEVICE_MISMATCH",
                    "message": "CHE-12 verifies the CPU baseline only.",
                    "details": {"jax_default_backend": jax.default_backend()},
                }
            )
        if forbidden:
            errors.append(
                {
                    "code": "WAVE_BASELINE_IMPORT_LEAK",
                    "message": "The wave-only probe loaded a ray engine or coupler module.",
                    "details": {"modules": forbidden},
                }
            )

        return (
            {
                "engine": "M_WAVE_CHROMATIX",
                "versions": versions,
                "pin_checks": pin_checks,
                "backend": {
                    "jax_default_backend": jax.default_backend(),
                    "jax_devices": [str(device) for device in jax.devices()],
                    "jax_enable_x64": bool(jax.config.jax_enable_x64),
                },
                "forbidden_modules_loaded": forbidden,
            },
            errors,
        )
    except Exception as exc:  # structured blocker; do not invent solver output
        return {}, [
            {
                "code": "WAVE_ENGINE_PROBE_FAILED",
                "message": str(exc),
                "details": {"exception_type": type(exc).__name__},
            }
        ]


def build_report(engine: str) -> dict[str, Any]:
    requested = [engine]
    engines: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    for name in requested:
        payload, engine_errors = _probe_ray() if name == "ray" else _probe_wave()
        if payload:
            engines[name] = payload
        errors.extend(engine_errors)

    return {
        "schema_version": 1,
        "probe": "CHE-12 M1 engine and convention verification",
        "status": "compatible" if not errors else "blocked",
        "requested_engines": requested,
        "expected_pins": EXPECTED,
        "environment": _cpu_environment(),
        "engines": engines,
        "errors": errors,
    }


def build_combined_report() -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for engine in ("ray", "wave"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--engine", engine],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "code": "ENGINE_SUBPROCESS_REPORT_INVALID",
                    "message": str(exc),
                    "details": {
                        "engine": engine,
                        "returncode": completed.returncode,
                        "stderr": completed.stderr,
                    },
                }
            )
            continue
        reports[engine] = report
        errors.extend(report["errors"])
        if completed.returncode != 0 and not report["errors"]:
            errors.append(
                {
                    "code": "ENGINE_SUBPROCESS_FAILED",
                    "message": "Engine subprocess returned non-zero without a diagnostic.",
                    "details": {"engine": engine, "returncode": completed.returncode},
                }
            )

    return {
        "schema_version": 1,
        "probe": "CHE-12 M1 engine and convention verification",
        "status": "compatible" if not errors else "blocked",
        "requested_engines": ["ray", "wave"],
        "execution": "independent_subprocesses",
        "expected_pins": EXPECTED,
        "reports": reports,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("ray", "wave", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_combined_report() if args.engine == "all" else build_report(args.engine)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if report["status"] == "compatible" else 1)


if __name__ == "__main__":
    main()
