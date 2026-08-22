#!/usr/bin/env python3
"""demo2 cost ceiling — NCC and MSE per (incident, secondary) cell (CHE-96).

The structure of the paper's own Fig 3b, run on our reconstruction rather than
theirs, and the reason it is worth running is that the two reconstructions have
different complexity. Ours is O(N_rays x N_pixels): `ray_to_wave` contracts a
separable `einsum("n,ny,nx->yx")`. Upstream's is O(N_rays) + one FFT. So the
same ray budget is not the same cost, and the shape of this table is the
evidence for or against the k-space fast path tracked on CHE-95.

Two things are reported that a bare accuracy sweep would not be:

* **wall clock and rays/s per cell**, so the ceiling is a measured budget rather
  than a guess;
* **what was skipped**, explicitly. Cells beyond `--max-rays` are recorded with
  `status: "skipped"` and the reason. A sweep that silently drops its expensive
  corner reads as though it covered the space.

Run in a dedicated GPU session:
    MOA_GPUS=device=0 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo2_cost_sweep.py --backend jax
"""

from __future__ import annotations

import argparse
import sys
import time
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
    mse_unit_sum,
    ncc,
    patch_route,
    write_record,
)
from demo2_hologram import GRID_N, PITCH_M, SENSOR_Z_M, WAVELENGTH_M, oracle

INCIDENT = (100, 400, 1_600, 6_400, 16_000)
SECONDARY = (100, 1_000, 10_000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--patch-px", type=int, default=51)
    parser.add_argument("--pad-factor", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-rays",
        type=float,
        default=2e8,
        help="cells above this are skipped and recorded as skipped, not dropped",
    )
    parser.add_argument(
        "--rays-per-chunk",
        type=float,
        default=4e6,
        help=(
            "device budget. 4e6 rays x 100 pixels x 2 separable factors x 8 B is "
            "~6 GB, which leaves headroom on a 48 GB card."
        ),
    )
    parser.add_argument("--output-name", default="demo2_cost_sweep")
    args = parser.parse_args()

    enable_x64_if_needed(backend=args.backend, precisions=[args.precision])
    doe = build_doe("demo2_smile_phase_profile.npy", pitch_m=PITCH_M)

    cells: list[dict[str, Any]] = []
    started = time.perf_counter()
    for incident in INCIDENT:
        for secondary in SECONDARY:
            total = incident * secondary
            if total > args.max_rays:
                cells.append(
                    {
                        "incident_rays": incident,
                        "secondary_per_incident": secondary,
                        "total_rays": total,
                        "status": "skipped",
                        "reason": (
                            f"{total:.2e} rays exceeds the --max-rays ceiling of "
                            f"{args.max_rays:.2e}. Recorded rather than dropped: a "
                            "sweep that quietly omits its expensive corner reads as "
                            "though it covered the space."
                        ),
                    }
                )
                continue
            batches = max(1, int(np.ceil(total / args.rays_per_chunk)))
            run = patch_route(
                doe,
                wavelength_m=WAVELENGTH_M,
                sensor_z_m=SENSOR_Z_M,
                sensor_shape=(GRID_N, GRID_N),
                patch_px=args.patch_px,
                pad_factor=args.pad_factor,
                patch_count=incident,
                secondary_count=secondary,
                batches=batches,
                seed=args.seed,
                backend=args.backend,
                precision=args.precision,
            )
            field = run["field"]
            reference = oracle(doe.transmission, pad_to=run["plan"]["pad_px"])
            cells.append(
                {
                    "incident_rays": incident,
                    "secondary_per_incident": secondary,
                    "total_rays": total,
                    "status": "measured",
                    "batches": batches,
                    "pad_px": run["plan"]["pad_px"],
                    "ncc_intensity": ncc(np.abs(field) ** 2, np.abs(reference) ** 2),
                    "mse_intensity_unit_sum": mse_unit_sum(
                        np.abs(field) ** 2, np.abs(reference) ** 2
                    ),
                    "relative_l2_field": float(
                        np.linalg.norm(field - reference) / np.linalg.norm(reference)
                    ),
                    "wall_clock_s": run["wall_clock_s"],
                    "rays_per_second": total / run["wall_clock_s"],
                    "device_memory": device_memory_stats(),
                }
            )
            print(
                f"  {incident:>6} x {secondary:>6} = {total:>12,}  "
                f"NCC {cells[-1]['ncc_intensity']:.6f}  "
                f"{run['wall_clock_s']:7.2f} s  {batches:>3} batches"
            )

    measured = [c for c in cells if c["status"] == "measured"]
    record = {
        "probe": "demo2_cost_sweep",
        "issue": "CHE-96",
        "environment": environment(),
        "configuration": {
            "patch_px": args.patch_px,
            "pad_factor_requested": args.pad_factor,
            "precision": args.precision,
            "backend": args.backend,
            "seed": args.seed,
            "max_rays": args.max_rays,
            "rays_per_chunk": args.rays_per_chunk,
            "oracle": "matched-periodicity ASM at the route's own pad",
        },
        "cost_model": (
            "our reconstruction is O(N_rays x N_pixels) -- ray_to_wave contracts a "
            "separable einsum('n,ny,nx->yx'). Upstream's is O(N_rays) + one FFT. "
            "The same ray budget is therefore not the same cost, and the wall-clock "
            "column is the evidence for the k-space fast path tracked on CHE-95."
        ),
        "cells": cells,
        "total_wall_clock_s": time.perf_counter() - started,
        "summary": {
            "best_ncc": max((c["ncc_intensity"] for c in measured), default=None),
            "cells_measured": len(measured),
            "cells_skipped": len(cells) - len(measured),
        },
    }
    path = write_record(args.output_name, record)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
