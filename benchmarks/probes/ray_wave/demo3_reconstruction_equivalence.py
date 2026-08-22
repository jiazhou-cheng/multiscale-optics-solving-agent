#!/usr/bin/env python3
"""CHE-101 — is demo3's fast reconstruction the same field, or only a cheaper one?

demo2 answers this trivially: its rays are drawn from a padded FFT spectrum and
never refracted, so every transverse wavevector is an exact bin of that grid and
a matched k-grid makes the splat a relabelling (measured: 7.1153e-13 against the
exact route's 7.1147e-13). demo3 cannot use that argument. Its rays pass through
a refractive singlet before reaching the sensor, so their directions are
continuous and no k-grid period puts them on nodes. The splat is a genuine
interpolation there, and the size of the resulting error is the question this
probe exists to answer rather than assume.

Why it has to be answered before the convergence ladder is rerun: demo3's whole
deliverable is a convergence *rate*. A biased estimator can converge faster to
the wrong field, which would read as success. So the two routes are run on
**identical rays** -- same seed, same trace, same patches, only the
reconstruction differing -- and compared field to field.

Run:
    ./run.sh python benchmarks/probes/ray_wave/demo3_reconstruction_equivalence.py
    MOA_GPUS=device=0 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo3_reconstruction_equivalence.py --preset characterization
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_support import enable_x64_if_needed, environment, mse_unit_sum, ncc, write_record
from demo3_hologram_lens import (
    GRID_N,
    PRESETS,
    build_doe_and_lens,
    run_route,
)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _phase_free_l2(a: np.ndarray, b: np.ndarray) -> float:
    """The error a constant phase offset cannot explain.

    Kept separate because the two routes reference their phase to the same
    plane but reach it by different transforms, and a global piston is not a
    field difference the sensor could ever see.
    """
    x, y = a.ravel(), b.ravel()
    inner = np.vdot(y, x)
    if inner == 0:
        return _relative_l2(a, b)
    return float(np.linalg.norm(x * np.exp(-1j * np.angle(inner)) - y) / np.linalg.norm(y))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--route", default="rw_p")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--patch-count", type=int, default=None)
    parser.add_argument("--secondary-count", type=int, default=None)
    parser.add_argument("--sensor-px", type=int, default=None)
    parser.add_argument("--oversamples", default="1.0,1.5,2.0,3.0,4.0")
    parser.add_argument("--rays-per-chunk", type=float, default=1e6)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    settings = dict(preset[args.route])
    for key, value in (
        ("patch_count", args.patch_count),
        ("secondary_count", args.secondary_count),
    ):
        if value is not None:
            settings[key] = value
    sensor_px = args.sensor_px or preset["sensor_px"]
    sensor_pitch_m = preset["sensor_pitch_m"]
    x64 = enable_x64_if_needed(backend=args.backend, precisions=[settings["precision"]])

    doe, lens, spec, execution = build_doe_and_lens(
        backend=args.backend, precision=settings["precision"]
    )

    def once(reconstruction: str, oversample: float) -> dict[str, Any]:
        return run_route(
            doe,
            lens,
            sensor_shape=(sensor_px, sensor_px),
            sensor_pitch_m=sensor_pitch_m,
            patch_px=settings["patch_px"] or (GRID_N - 1),
            pad_factor=settings["pad_factor"],
            pad_width=settings["pad_width"],
            patch_count=settings["patch_count"],
            route=settings["route"],
            secondary_count=settings["secondary_count"],
            batches=settings["batches"],
            seed=args.seed,
            backend=args.backend,
            precision=settings["precision"],
            secondary_chunks=max(
                1, int(np.ceil(settings["secondary_count"] / args.rays_per_chunk))
            ),
            reconstruction_route=reconstruction,
            kspace_oversample=oversample,
        )

    exact = once("ramp_sum", 1.0)
    exact_field = exact.pop("field")
    print(
        f"ramp_sum: {exact['total_rays']:,} rays  {exact['wall_clock_s']:.2f} s "
        f"on {sensor_px}x{sensor_px}"
    )

    comparisons = []
    for oversample in [float(v) for v in args.oversamples.split(",")]:
        fast = once("kspace_splat", oversample)
        field = fast.pop("field")
        row = {
            "kspace_oversample": oversample,
            "kspace": fast["reconstruction"]["measured"],
            "wall_clock_s": fast["wall_clock_s"],
            "speedup_over_ramp_sum": exact["wall_clock_s"] / fast["wall_clock_s"],
            "ncc_intensity": ncc(np.abs(field) ** 2, np.abs(exact_field) ** 2),
            "mse_intensity_unit_sum": mse_unit_sum(np.abs(field) ** 2, np.abs(exact_field) ** 2),
            "relative_l2_field": _relative_l2(field, exact_field),
            "relative_l2_field_after_global_phase": _phase_free_l2(field, exact_field),
            # Power is the quantity a dropped or misplaced ray steals, so it is
            # reported next to the agreement rather than left to be inferred.
            "power_ratio_to_ramp_sum": float(
                np.sum(np.abs(field) ** 2) / np.sum(np.abs(exact_field) ** 2)
            ),
        }
        comparisons.append(row)
        print(
            f"  oversample {oversample:>4}: NCC {row['ncc_intensity']:.6f}  "
            f"rel-L2 {row['relative_l2_field_after_global_phase']:.4e}  "
            f"{row['wall_clock_s']:.2f} s ({row['speedup_over_ramp_sum']:.2f}x)  "
            f"on-node {row['kspace']['on_node_fraction']:.3f}  "
            f"dropped {row['kspace']['dropped_fraction']:.2e}"
        )

    record = {
        "probe": "demo3_reconstruction_equivalence",
        "issue": "CHE-101",
        "preset": args.preset,
        "route": args.route,
        "environment": {**environment(), **x64},
        "configuration": {
            "sensor_grid": [sensor_px, sensor_px],
            "sensor_pitch_m": sensor_pitch_m,
            "patch_count": settings["patch_count"],
            "secondary_count": settings["secondary_count"],
            "precision": settings["precision"],
            "backend": args.backend,
            "seed": args.seed,
            "total_rays": exact["total_rays"],
            "prescription_fingerprint": spec.fingerprint(),
            "optiland_execution": execution.as_dict(),
        },
        "note": (
            "Identical rays through both reconstructions -- same seed, same trace, "
            "same patches. demo3's rays are refracted before the sensor, so they "
            "are off-node by construction and the splat interpolates; this is the "
            "measurement of what that costs. Contrast demo2, where a matched "
            "k-grid makes the same operation exact."
        ),
        "reference": {
            "reconstruction": "ramp_sum",
            "wall_clock_s": exact["wall_clock_s"],
            "energy": exact.get("energy"),
        },
        "comparisons": comparisons,
    }
    path = write_record(
        args.output_name or f"demo3_equivalence_{args.preset}_{args.backend}", record
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
