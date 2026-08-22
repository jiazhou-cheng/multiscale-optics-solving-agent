"""Generate and verify the deterministic CHE-14 standalone Chromatix wave baseline.

Run inside the supported container entry point:

    ./run.sh python knowledge/solvers/chromatix/probes/standalone_baseline.py \
        --output-dir /tmp/chromatix-che14

The expected regression fixture is generated only by this executable:

    ./run.sh python knowledge/solvers/chromatix/probes/standalone_baseline.py \
        --output-dir /tmp/chromatix-che14 \
        --write-expected knowledge/solvers/chromatix/expected/standalone_baseline.json

The canonical case is a Gaussian amplitude profile evaluated analytically on
the sampled grid (no RNG, hence nothing to seed) and propagated one Rayleigh
range by ``chromatix.functional.asm_propagate``. This probe establishes
determinism, the field/grid/padding/power contract, and the runtime budget --
it is deliberately NOT an accuracy oracle. Analytic Gaussian accuracy is
benchmarked separately by ``archive/benchmarks/gen1/benchmarks/L1-WAVE-01``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from solvers.base import RunStatus
from solvers.chromatix.adapter import (
    ChromatixAdapter,
    ChromatixWaveRequest,
    ChromatixWaveResult,
)

# Canonical CHE-14 smoke case. Chosen so the beam is well inside the sampled
# window (half-window 3.2 w0) and the padded output window holds the
# propagated beam without wraparound, while staying far under the runtime and
# resource budgets.
WAVELENGTH_M = 532e-9
REFRACTIVE_INDEX = 1.0
WAIST_M = 10e-6
GRID = 128
PITCH_M = 0.5e-6
PAD_WIDTH = 256


def rayleigh_range_m(waist_m: float, wavelength_m: float, refractive_index: float) -> float:
    return float(np.pi * waist_m**2 * refractive_index / wavelength_m)


def gaussian_waist_field(grid: int, pitch_m: float, waist_m: float) -> np.ndarray:
    """Analytic Gaussian amplitude at its waist, on the chromatix (y, x) grid.

    Real-valued at the waist by construction, promoted to complex64 because a
    complex field stores amplitude; index ``grid // 2`` is coordinate zero on
    each axis, matching ``chromatix`` grid centering.
    """
    coordinates = (np.arange(grid) - grid // 2) * pitch_m
    x = coordinates[None, :]
    y = coordinates[:, None]
    return np.exp(-(x**2 + y**2) / waist_m**2).astype(np.complex64)


def _stable_result(result: ChromatixWaveResult) -> dict[str, Any]:
    """Project a result onto the values that must be identical across two runs.

    Excludes runtime, output paths, run identifiers, and the observed CPU /
    JAX backend strings: those are environment facts, not scientific output.
    """
    return {
        "package_version": result.package_version,
        "package_commit": result.package_commit,
        "propagation": result.propagation,
        "device": result.device,
        "dtype": result.dtype,
        "input_shape": list(result.input_shape or []),
        "output_shape": list(result.output_shape or []),
        "pad_width": result.pad_width,
        "padded": result.padded,
        "cropped": result.cropped,
        "input_field_sha256": result.input_field_sha256,
        "output_field_sha256": result.output_field_sha256,
        "scientific_array_sha256": result.scientific_array_sha256,
        "summary_metrics": result.summary_metrics,
        "field_metadata": {
            key: value
            for key, value in result.field_metadata.items()
            if key not in {"cpu_device", "jax_backend"}
        },
    }


def _forbidden_modules_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == "optiland"
        or name.startswith("optiland.")
        or name.startswith("couplers")
    )


def build_request(output_directory: Path, z_m: float) -> ChromatixWaveRequest:
    return ChromatixWaveRequest(
        input_field_array=gaussian_waist_field(GRID, PITCH_M, WAIST_M),
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=(PITCH_M, PITCH_M),
        z_m=z_m,
        refractive_index=REFRACTIVE_INDEX,
        padding_policy="explicit",
        pad_width=PAD_WIDTH,
        output_mode="full",
        reference_plane=(
            "input plane is the Gaussian waist at z=0; output plane is z=z_m, "
            "one Rayleigh range downstream"
        ),
        output_directory=output_directory,
    )


def _failure_examples(output_directory: Path, z_m: float) -> list[dict[str, Any]]:
    """Exercise every structured failure path required by CHE-14.

    Each entry records the code actually returned by the adapter. Nothing here
    asserts a physical result: the point is that a rejected request produces a
    diagnostic and no field.
    """
    adapter = ChromatixAdapter()
    base = build_request(output_directory / "unused", z_m).model_dump()
    base.pop("input_field_array")
    field = gaussian_waist_field(GRID, PITCH_M, WAIST_M)

    cases: list[tuple[str, dict[str, Any]]] = [
        ("wrong_propagation_kernel", {"propagation": "fresnel"}),
        ("vector_field_requested", {"field_kind": "vector"}),
        ("gradient_requested", {"require_gradients": True}),
        ("invalid_metadata_blank_phasor", {"phasor": "   "}),
        ("invalid_wavelength", {"wavelength_m": 0.0}),
        ("invalid_sample_pitch", {"sample_pitch_m": (PITCH_M, -1.0)}),
        ("excessive_resource_estimate", {"padding_policy": "auto_transfer", "pad_width": None}),
    ]

    examples: list[dict[str, Any]] = []
    for name, override in cases:
        request = {**base, **override, "input_field_array": field}
        request["output_directory"] = output_directory / "failures" / name
        result = adapter.run_standalone(request)
        assert result.status is RunStatus.FAILED, f"{name} unexpectedly succeeded"
        assert result.failure is not None
        examples.append(
            {
                "case": name,
                "code": result.failure.code,
                "stage": result.failure.stage,
                "produced_field": result.output_field_path is not None,
            }
        )

    # A non-finite input field must be rejected on its own path.
    corrupted = field.copy()
    corrupted[0, 0] = np.nan
    result = adapter.run_standalone(
        {
            **base,
            "input_field_array": corrupted,
            "output_directory": output_directory / "failures" / "non_finite_input_field",
        }
    )
    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    examples.append(
        {
            "case": "non_finite_input_field",
            "code": result.failure.code,
            "stage": result.failure.stage,
            "produced_field": result.output_field_path is not None,
        }
    )

    # A real-valued array is an intensity-vs-amplitude confusion, not a field.
    result = adapter.run_standalone(
        {
            **base,
            "input_field_array": np.abs(field).astype(np.float32),
            "output_directory": output_directory / "failures" / "real_valued_input_field",
        }
    )
    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    examples.append(
        {
            "case": "real_valued_input_field",
            "code": result.failure.code,
            "stage": result.failure.stage,
            "produced_field": result.output_field_path is not None,
        }
    )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-expected", type=Path)
    args = parser.parse_args()

    z_m = rayleigh_range_m(WAIST_M, WAVELENGTH_M, REFRACTIVE_INDEX)
    adapter = ChromatixAdapter()
    first = adapter.run_standalone(build_request(args.output_dir / "run_a", z_m))
    second = adapter.run_standalone(build_request(args.output_dir / "run_b", z_m))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if any(r.status is not RunStatus.SUCCEEDED for r in (first, second)):
        report = {
            "schema_version": 1,
            "status": "failed",
            "failures": [
                r.failure.model_dump(mode="json") for r in (first, second) if r.failure is not None
            ],
        }
        (args.output_dir / "determinism_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    stable_first = _stable_result(first)
    stable_second = _stable_result(second)

    # Byte-level check on the field arrays themselves, independent of the
    # summary dictionaries above.
    array_a = np.load(str(first.output_field_path))
    array_b = np.load(str(second.output_field_path))
    identical_arrays = bool(array_a.dtype == array_b.dtype and np.array_equal(array_a, array_b))

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "case": {
            "wavelength_m": WAVELENGTH_M,
            "refractive_index": REFRACTIVE_INDEX,
            "waist_m": WAIST_M,
            "grid": GRID,
            "sample_pitch_m": PITCH_M,
            "pad_width": PAD_WIDTH,
            "z_m": z_m,
            "z_over_rayleigh_range": 1.0,
        },
        "deterministic": stable_first == stable_second,
        "identical_field_arrays": identical_arrays,
        "stable_result": stable_first,
        "failure_examples": _failure_examples(args.output_dir, z_m),
        "runtime_seconds": [first.runtime_seconds, second.runtime_seconds],
        "runtime_under_10_seconds": all(
            runtime is not None and runtime < 10.0
            for runtime in (first.runtime_seconds, second.runtime_seconds)
        ),
        "warnings": first.warnings,
        "forbidden_modules_loaded": _forbidden_modules_loaded(),
    }
    if not (
        report["deterministic"]
        and report["identical_field_arrays"]
        and report["runtime_under_10_seconds"]
        and not report["forbidden_modules_loaded"]
    ):
        report["status"] = "failed"

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
