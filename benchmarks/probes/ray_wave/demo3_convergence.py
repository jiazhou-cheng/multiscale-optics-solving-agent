#!/usr/bin/env python3
"""demo3 — is the seed-to-seed disagreement Monte Carlo noise, or a real one?

At any budget that fits in an overnight run, demo3's sensor field is
undersampled: two independent realizations of the same route correlate at
NCC 0.04, which is SI Figure S4's artifact rather than a converged speckle
pattern. Reporting that number alone would leave the important question open --
is it noise that more rays would remove, or a systematic disagreement that no
budget would?

This probe answers it by measuring the trend. Seed-to-seed NCC is the estimator's
own signal fraction; for an unbiased estimator with independent noise it grows
linearly in the ray count while it is small. A measured slope near 1 in
``log NCC`` versus ``log N`` says the disagreement is noise and gives the budget
that would remove it. That budget is the cost-ceiling evidence the issue asks
for, quantified rather than asserted.

Run in a dedicated GPU session:
    MOA_GPUS=device=0 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo3_convergence.py --route rw_p
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_support import (
    build_doe,
    device_memory_stats,
    enable_x64_if_needed,
    environment,
    ncc,
    write_record,
)
from demo3_hologram_lens import (
    GRID_N,
    PITCH_M,
    PRESETS,
    demo3_system,
    run_route,
)

from core.precision import DeviceKind, DevicePlacement, Precision
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import configure_optiland_execution

#: Multiples of `--base-patches`. Deliberately starting above the noise floor:
#: an NCC estimated from 1.76e5 pixels has a standard error near
#: 1/sqrt(N_px) = 0.0024, and the first attempt at this ladder began at 5e6 rays
#: where the measured values were 9e-5 and **-3.5e-4**. A negative correlation
#: is not a small one; it is zero with an error bar, and fitting a power law
#: through it returned a slope of 5.7 and an extrapolation an order of magnitude
#: too optimistic.
LADDER = (8, 12, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("rw_f", "rw_p"), default="rw_p")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--base-patches", type=int, default=125)
    parser.add_argument("--secondary-count", type=int, default=20_000)
    parser.add_argument("--seeds", default="20260822,7")
    parser.add_argument("--rays-per-chunk", type=float, default=1e6)
    parser.add_argument("--target-ncc", type=float, default=0.9)
    parser.add_argument(
        "--reconstruction", choices=("ramp_sum", "kspace_splat"), default="ramp_sum"
    )
    parser.add_argument("--kspace-oversample", type=float, default=1.5)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    preset = PRESETS["characterization"]
    settings = dict(preset[args.route])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    enable_x64_if_needed(backend=args.backend, precisions=[settings["precision"]])

    device = DevicePlacement(
        kind=DeviceKind.CUDA if args.backend == "jax" else DeviceKind.CPU, index=0
    )
    configure_optiland_execution(
        device=device,
        precision=Precision.FP32 if settings["precision"] == "fp32" else Precision.FP64,
        enable_grad=False,
    )
    lens = build_optiland_system(demo3_system())
    doe = build_doe("demo3_smile_phase_profile.npy", pitch_m=PITCH_M)
    sensor_px = preset["sensor_px"]
    sensor_pitch_m = preset["sensor_pitch_m"]

    rungs: list[dict[str, Any]] = []
    for multiple in LADDER:
        patches = args.base_patches * multiple
        total = patches * args.secondary_count
        fields = []
        wall = 0.0
        for seed in seeds:
            run = run_route(
                doe,
                lens,
                sensor_shape=(sensor_px, sensor_px),
                sensor_pitch_m=sensor_pitch_m,
                patch_px=settings["patch_px"] or (GRID_N - 1),
                pad_factor=settings["pad_factor"],
                pad_width=settings["pad_width"],
                patch_count=patches,
                secondary_count=args.secondary_count,
                batches=max(1, patches // 50),
                seed=seed,
                backend=args.backend,
                precision=settings["precision"],
                secondary_chunks=max(
                    1, int(np.ceil(50 * args.secondary_count / args.rays_per_chunk))
                ),
                route=settings["route"],
                reconstruction_route=args.reconstruction,
                kspace_oversample=args.kspace_oversample,
            )
            fields.append(run["field"])
            wall += run["wall_clock_s"]
        pairs = [
            ncc(np.abs(fields[i]) ** 2, np.abs(fields[j]) ** 2)
            for i in range(len(fields))
            for j in range(i + 1, len(fields))
        ]
        rungs.append(
            {
                "patches": patches,
                "secondary_per_patch": args.secondary_count,
                "total_rays": total,
                "rays_per_pixel": total / (sensor_px * sensor_px),
                "mean_seed_to_seed_ncc": float(np.mean(pairs)),
                "pairs": pairs,
                "wall_clock_s": wall,
            }
        )
        print(
            f"  {total:>12,} rays ({rungs[-1]['rays_per_pixel']:7.1f}/px): "
            f"NCC {rungs[-1]['mean_seed_to_seed_ncc']:.5f}  {wall:6.1f} s"
        )
        del fields

    # An NCC estimated from N_px pixels has a standard error near 1/sqrt(N_px);
    # below a few times that, the measurement is zero with an error bar and has
    # no business in a power-law fit.
    noise_floor = 3.0 / np.sqrt(sensor_px * sensor_px)
    usable = [r for r in rungs if r["mean_seed_to_seed_ncc"] > noise_floor]
    if len(usable) >= 2:
        logn = np.log([r["total_rays"] for r in usable])
        logncc = np.log([r["mean_seed_to_seed_ncc"] for r in usable])
        slope, intercept = np.polyfit(logn, logncc, 1)
        fitted_required = float(np.exp((np.log(args.target_ncc) - intercept) / slope))
    else:
        slope = intercept = float("nan")
        fitted_required = float("nan")

    # The theory-anchored estimate, reported alongside the fit rather than
    # instead of it. Signal fraction grows linearly in N while it is small, so
    # slope 1 is the prediction; extrapolating from the highest usable rung with
    # that slope is far less sensitive to scatter than a two-point fit, and the
    # gap between the two is the honest uncertainty on the answer.
    anchor = usable[-1] if usable else rungs[-1]
    anchored_required = anchor["total_rays"] * (
        args.target_ncc / max(anchor["mean_seed_to_seed_ncc"], 1e-12)
    )
    seconds_per_ray = rungs[-1]["wall_clock_s"] / (len(seeds) * rungs[-1]["total_rays"])
    required = anchored_required

    record = {
        "probe": "demo3_convergence",
        "issue": "CHE-96",
        "route": args.route,
        "environment": environment(),
        "configuration": {
            "sensor_grid": [sensor_px, sensor_px],
            "sensor_pitch_m": sensor_pitch_m,
            "secondary_per_emitter": args.secondary_count,
            "seeds": seeds,
            "precision": settings["precision"],
            "backend": args.backend,
            "reconstruction": args.reconstruction,
            "kspace_oversample": (
                args.kspace_oversample if args.reconstruction != "ramp_sum" else None
            ),
        },
        "rungs": rungs,
        "trend": {
            "note": (
                "seed-to-seed NCC is the estimator's own signal fraction. While it "
                "is small it grows linearly in the ray count for an unbiased "
                "estimator with independent noise, so a log-log slope near 1 says "
                "the disagreement is Monte Carlo noise and not a systematic one."
            ),
            "noise_floor_ncc": float(noise_floor),
            "rungs_above_noise_floor": len(usable),
            "log_log_slope": float(slope),
            "slope_reading": (
                "near 1: noise-limited, more rays fix it. Near 0: a systematic "
                "difference that no budget removes."
            ),
        },
        "cost_ceiling": {
            "note": (
                "Cost per ray, measured on THIS run's reconstruction route. On "
                "ramp_sum it is O(N_rays x N_pixels) and this is the shortfall "
                "CHE-96 reported as evidence; on kspace_splat it is O(N_rays) plus "
                "one FFT per chunk, and the same number becomes the budget that is "
                "now reachable. The route is recorded in `configuration` so the two "
                "are never compared without it."
            ),
            "target_seed_to_seed_ncc": args.target_ncc,
            "extrapolated_rays_required": required,
            "extrapolated_rays_required_from_fit": fitted_required,
            "extrapolation_basis": (
                "slope-1 theory anchored at the highest rung above the noise "
                "floor. The two-point log-log fit is reported next to it; the "
                "spread between them is the uncertainty on this number."
            ),
            "superseded_by": (
                "CHE-120. The slope-1 anchor is one power out. What two "
                "realizations SHARE is the deterministic |mu|^2; what differs is "
                "the noise, whose spatial variance goes as (E|n|^2)^2 once the "
                "noise dominates, so NCC ~ N^2 rather than N. This ladder does "
                "not see that because its rungs sit 1.5-2.9x above the noise "
                "floor, where the floor itself flattens the fit -- a four-rung "
                "ladder starting 3.2x above it and spanning 8x in rays measures "
                "1.40 (power law) and prefers a saturating law NCC = 1 / (1 + c "
                "N^-p) with p = 1.64, validated out of sample on its own top "
                "rung. Read `demo3_variance_ladderfit_control.json` for the "
                "current cost ceiling; the rungs and the wall clock here stand."
            ),
            "measured_seconds_per_ray": seconds_per_ray,
            "extrapolated_seconds_per_run": required * seconds_per_ray,
            "extrapolated_hours_per_run": required * seconds_per_ray / 3600.0,
            "paper_table_s2_rays": {
                "rw_f": 3.3e4 * 1.6e5,
                "rw_p": 2.6e5 * 1e4,
            },
        },
        "device_memory": device_memory_stats(),
    }
    path = write_record(args.output_name or f"demo3_convergence_{args.route}", record)
    print(
        f"wrote {path}\n"
        f"  slope {slope:.3f}; NCC {args.target_ncc} needs {required:.3e} rays "
        f"~= {required * seconds_per_ray / 3600:.1f} h/run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
