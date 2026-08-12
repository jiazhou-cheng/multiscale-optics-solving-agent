"""Build the independent analytic reference for L1-WAVE-01.

Evaluates all three oracles for every configured case and writes them out.
Imports neither Chromatix nor JAX, so the reference can be reviewed without
the solver under test being installed or invoked:

    ./run.sh python benchmarks/level1/L1-WAVE-01/generate_reference.py

``evaluate.py`` calls :func:`build_reference` in-process rather than reading
these files back, so the reference the evaluator compares against is
guaranteed to be the one this module defines. The written bundle is the
human-reviewable copy of exactly that object.

The Richards-Wolf quadrature convergence study also lives here, because it is
a property of the *oracle*, not of the solver: the reference must be shown
converged before it is allowed to judge anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import measurement  # noqa: E402
import oracles  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[2] / "outputs" / "L1-WAVE-01" / "reference"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((path or SCRIPT_DIR / "public_config.yaml").read_text())
    return config


# ---------------------------------------------------------------------------
# Case 1
# ---------------------------------------------------------------------------
def build_case1(config: dict[str, Any]) -> list[dict[str, Any]]:
    physics = config["physics"]
    wavelength_m = float(physics["wavelength_m"])
    refractive_index = float(physics["refractive_index"])
    case = config["case1_plane_wave"]
    n_grid = int(case["n_grid"])
    pitch_m = float(case["pitch_m"])

    cases: list[dict[str, Any]] = []
    for mode_y, mode_x in case["modes"]:
        mode_y, mode_x = int(mode_y), int(mode_x)
        frequency_y = oracles.fft_bin_frequency(mode_y, n_grid, pitch_m)
        frequency_x = oracles.fft_bin_frequency(mode_x, n_grid, pitch_m)
        axial = oracles.axial_wavenumber(frequency_y, frequency_x, wavelength_m, refractive_index)
        input_field = oracles.plane_wave_mode(n_grid, mode_y, mode_x)
        sin_theta = wavelength_m * float(np.hypot(frequency_x, frequency_y)) / refractive_index

        for z_m in case["distances_m"]:
            z_m = float(z_m)
            cases.append(
                {
                    "name": f"case1_m{mode_y}_{mode_x}_z{z_m:g}",
                    "case": "case1",
                    "role": "accuracy",
                    "grid": {
                        "n": n_grid,
                        "pitch_m": pitch_m,
                        "pad_width": int(case["pad_width"]),
                        "padded_n": n_grid + 2 * int(case["pad_width"]),
                    },
                    "mode": {
                        "mode_y": mode_y,
                        "mode_x": mode_x,
                        "frequency_y_per_m": frequency_y,
                        "frequency_x_per_m": frequency_x,
                        "sin_theta": sin_theta,
                    },
                    "planes": {"z_m": z_m},
                    "physics": {
                        "wavelength_m": wavelength_m,
                        "refractive_index": refractive_index,
                    },
                    "closed_form": {
                        "axial_wavenumber_rad_per_m": axial,
                        "accumulated_phase_rad": axial * z_m,
                        "amplitude_ratio": 1.0,
                        "power_ratio": 1.0,
                        "float32_phase_round_off_rad": oracles.float32_phase_round_off(axial, z_m),
                    },
                    "sampling": oracles.sampling_diagnostics(
                        n=n_grid,
                        pitch_m=pitch_m,
                        pad_width=int(case["pad_width"]),
                        wavelength_m=wavelength_m,
                        refractive_index=refractive_index,
                        occupied_frequency_per_m=float(np.hypot(frequency_x, frequency_y)),
                        z_m=z_m,
                    ),
                    "_input_field": input_field,
                    "_expected_field": oracles.plane_wave_transfer(input_field, axial, z_m),
                }
            )
    return cases


# ---------------------------------------------------------------------------
# Case 2
# ---------------------------------------------------------------------------
def build_case2(config: dict[str, Any]) -> list[dict[str, Any]]:
    physics = config["physics"]
    wavelength_m = float(physics["wavelength_m"])
    refractive_index = float(physics["refractive_index"])
    case = config["case2_paraxial_focus"]
    n_grid = int(case["n_grid"])
    pitch_m = float(case["pitch_m"])
    pad_width = int(case["pad_width"])
    focal_length_m = float(case["focal_length_m"])
    samples_x = int(case["aperture_samples_x"])
    samples_y = int(case["aperture_samples_y"])
    aperture_x_m = samples_x * pitch_m
    aperture_y_m = samples_y * pitch_m
    padded_n = n_grid + 2 * pad_width
    shape = oracles.sinc_squared_reference()

    cases: list[dict[str, Any]] = []
    for tilt_x_rad, tilt_y_rad in case["tilts_rad"]:
        tilt_x_rad, tilt_y_rad = float(tilt_x_rad), float(tilt_y_rad)
        input_field = oracles.thin_lens_pupil_field(
            n=n_grid,
            pitch_m=pitch_m,
            samples_x=samples_x,
            samples_y=samples_y,
            focal_length_m=focal_length_m,
            tilt_x_rad=tilt_x_rad,
            tilt_y_rad=tilt_y_rad,
            wavelength_m=wavelength_m,
            refractive_index=refractive_index,
        )
        expected_field = oracles.fresnel_focal_field(
            n=padded_n,
            pitch_m=pitch_m,
            aperture_x_m=aperture_x_m,
            aperture_y_m=aperture_y_m,
            focal_length_m=focal_length_m,
            tilt_x_rad=tilt_x_rad,
            tilt_y_rad=tilt_y_rad,
            wavelength_m=wavelength_m,
            refractive_index=refractive_index,
        )
        # Same estimators, applied to the analytic field on the analytic grid:
        # the discretization control.
        expected_intensity = np.abs(expected_field) ** 2
        row = int(
            np.argmin(
                np.abs(
                    measurement.grid_coordinates_m(padded_n, pitch_m) - focal_length_m * tilt_y_rad
                )
            )
        )
        column = int(
            np.argmin(
                np.abs(
                    measurement.grid_coordinates_m(padded_n, pitch_m) - focal_length_m * tilt_x_rad
                )
            )
        )
        on_grid = {
            "fwhm_x_m": measurement.fwhm_m(expected_intensity[row], pitch_m),
            "fwhm_y_m": measurement.fwhm_m(expected_intensity[:, column], pitch_m),
            "first_sidelobe_ratio": measurement.first_sidelobe_ratio(expected_intensity[row]),
            **measurement.intensity_centroid_m(expected_intensity, pitch_m),
        }

        cases.append(
            {
                "name": f"case2_tilt{tilt_x_rad:+.1e}_{tilt_y_rad:+.1e}",
                "case": "case2",
                "role": "accuracy",
                "grid": {
                    "n": n_grid,
                    "pitch_m": pitch_m,
                    "pad_width": pad_width,
                    "padded_n": padded_n,
                },
                "pupil": {
                    "aperture_samples_x": samples_x,
                    "aperture_samples_y": samples_y,
                    "aperture_x_m": aperture_x_m,
                    "aperture_y_m": aperture_y_m,
                    "focal_length_m": focal_length_m,
                    "tilt_x_rad": tilt_x_rad,
                    "tilt_y_rad": tilt_y_rad,
                },
                "planes": {"z_m": focal_length_m},
                "physics": {
                    "wavelength_m": wavelength_m,
                    "refractive_index": refractive_index,
                },
                "closed_form": {
                    # Signed focal position -- the sign test.
                    "centroid_x_m": focal_length_m * tilt_x_rad,
                    "centroid_y_m": focal_length_m * tilt_y_rad,
                    "fwhm_x_m": shape["fwhm_in_t"] * wavelength_m * focal_length_m / aperture_x_m,
                    "fwhm_y_m": shape["fwhm_in_t"] * wavelength_m * focal_length_m / aperture_y_m,
                    "first_null_x_m": wavelength_m * focal_length_m / aperture_x_m,
                    "first_null_y_m": wavelength_m * focal_length_m / aperture_y_m,
                    "first_sidelobe_ratio": shape["first_sidelobe_ratio"],
                    "power_ratio": 1.0,
                },
                "estimator_on_analytic_field": on_grid,
                "sampling": oracles.sampling_diagnostics(
                    n=n_grid,
                    pitch_m=pitch_m,
                    pad_width=pad_width,
                    wavelength_m=wavelength_m,
                    refractive_index=refractive_index,
                    occupied_frequency_per_m=(aperture_x_m / 2.0 / (wavelength_m * focal_length_m)),
                    z_m=focal_length_m,
                ),
                "_input_field": input_field,
                "_expected_field": expected_field,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Case 3
# ---------------------------------------------------------------------------
def case3_geometry(config: dict[str, Any]) -> dict[str, Any]:
    physics = config["physics"]
    case = config["case3_high_na_vector"]
    wavelength_m = float(physics["wavelength_m"])
    numerical_aperture = float(case["numerical_aperture"])
    output_pitch_m = wavelength_m / (
        float(case["output_pitch_over_lambda_na"]) * numerical_aperture
    )
    return {
        "wavelength_m": wavelength_m,
        "refractive_index": float(physics["refractive_index"]),
        "numerical_aperture": numerical_aperture,
        "focal_length_m": float(case["focal_length_m"]),
        "output_samples": int(case["output_samples"]),
        "output_pitch_m": output_pitch_m,
        "reference_pupil_samples": int(case["reference_pupil_samples"]),
        "pupil_sample_sweep": [int(v) for v in case["pupil_sample_sweep"]],
        "quadrature_sweep": [int(v) for v in case["quadrature_sweep"]],
        "reference_quadrature_points": int(case["reference_quadrature_points"]),
    }


def richards_wolf_quadrature_convergence(geometry: dict[str, Any]) -> dict[str, Any]:
    """Show the oracle converged before it is used to judge the solver."""
    ratios: list[float] = []
    for quadrature_points in geometry["quadrature_sweep"]:
        field = oracles.richards_wolf_on_grid(
            n=geometry["output_samples"],
            pitch_m=geometry["output_pitch_m"],
            defocus_m=0.0,
            numerical_aperture=geometry["numerical_aperture"],
            refractive_index=geometry["refractive_index"],
            wavelength_m=geometry["wavelength_m"],
            quadrature_points=quadrature_points,
        )
        ratios.append(measurement.vector_component_ratios(field)["iz_over_ix"])

    tail = ratios[-3:]
    spread = (max(tail) - min(tail)) / abs(np.mean(tail)) if tail else float("inf")
    return {
        "quadrature_points": geometry["quadrature_sweep"],
        "iz_over_ix": ratios,
        "tail_relative_spread": float(spread),
    }


def build_case3(config: dict[str, Any]) -> dict[str, Any]:
    geometry = case3_geometry(config)
    reference_field = oracles.richards_wolf_on_grid(
        n=geometry["output_samples"],
        pitch_m=geometry["output_pitch_m"],
        defocus_m=0.0,
        numerical_aperture=geometry["numerical_aperture"],
        refractive_index=geometry["refractive_index"],
        wavelength_m=geometry["wavelength_m"],
        quadrature_points=geometry["reference_quadrature_points"],
    )
    intensity_x = np.abs(reference_field[..., 0]) ** 2
    centre = geometry["output_samples"] // 2
    return {
        "name": "case3_high_na_vector",
        "case": "case3",
        "role": "accuracy",
        "geometry": geometry,
        "closed_form": {
            **measurement.vector_component_ratios(reference_field),
            **measurement.vector_symmetry_diagnostics(reference_field),
            "fwhm_x_m": measurement.fwhm_m(intensity_x[centre], geometry["output_pitch_m"]),
            "fwhm_y_m": measurement.fwhm_m(intensity_x[:, centre], geometry["output_pitch_m"]),
            "ez_peak_radius_m": _peak_radius_m(
                np.abs(reference_field[..., 2]) ** 2, geometry["output_pitch_m"]
            ),
        },
        "quadrature_convergence": richards_wolf_quadrature_convergence(geometry),
        "_expected_field": reference_field,
    }


def _peak_radius_m(intensity: np.ndarray, pitch_m: float) -> float:
    """Radius of the brightest sample, in metres. The natural scale ruler for E_z.

    ``E_z`` vanishes on axis and peaks in a ring, so the radius of that ring is
    a physical length that any correct implementation must reproduce
    independently of how the pupil happens to be sampled.
    """
    n = intensity.shape[0]
    row, column = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    return float(np.hypot(row - n // 2, column - n // 2) * pitch_m)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_reference(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "case1": build_case1(config),
        "case2": build_case2(config),
        "case3": build_case3(config),
    }


def serializable(reference: dict[str, Any]) -> dict[str, Any]:
    """Drop the field arrays, which go to the .npz rather than the .json."""

    def strip(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if not key.startswith("_")}

    return {
        "case1": [strip(record) for record in reference["case1"]],
        "case2": [strip(record) for record in reference["case2"]],
        "case3": strip(reference["case3"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    reference = build_reference(config)

    document = {
        "schema_version": 2,
        "benchmark_id": "L1-WAVE-01",
        "oracles": {
            "case1": (
                "exact discrete plane-wave eigenmode transfer u * exp(i k_z z) with "
                "k_z = (2 pi n / lambda) sqrt(1 - (lambda/n)^2 |f|^2); no approximation"
            ),
            "case2": (
                "analytic Fresnel/Fourier focal field of a rectangular pupil behind an "
                "ideal paraxial thin lens (separable sinc at +f*theta), plus an "
                "independent float64 angular spectrum for error attribution"
            ),
            "case3": (
                "independent float64 Richards-Wolf (Debye-Wolf) quadrature for an "
                "x-polarized aplanatic objective"
            ),
        },
        "conventions": config["conventions"],
        "cases": serializable(reference),
    }
    (output_dir / "reference.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )

    arrays: dict[str, np.ndarray] = {}
    for record in reference["case1"] + reference["case2"]:
        arrays[f"{record['name']}_input_field"] = record["_input_field"].astype(np.complex64)
        arrays[f"{record['name']}_expected_field"] = record["_expected_field"].astype(np.complex64)
    arrays["case3_expected_field"] = reference["case3"]["_expected_field"].astype(np.complex64)
    np.savez(output_dir / "reference_fields.npz", **arrays)

    convergence = reference["case3"]["quadrature_convergence"]
    print(
        json.dumps(
            {
                "status": "complete",
                "output_directory": str(output_dir),
                "case1_records": len(reference["case1"]),
                "case2_records": len(reference["case2"]),
                "case3_records": 1,
                "richards_wolf_quadrature_convergence": {
                    "quadrature_points": convergence["quadrature_points"],
                    "iz_over_ix": convergence["iz_over_ix"],
                    "tail_relative_spread": convergence["tail_relative_spread"],
                },
                "solver_imported": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
