#!/usr/bin/env python3
"""demo2 — planar hologram in free space, both routes (CHE-96, paper Fig 5b).

SI Table S2, System 2. lambda = 0.7 um; a 100x100 SLM at 6.3 um pitch at z = 0
behind a circular amplitude mask of radius 50 * 6.3 um; sensor 100x100 at the
same pitch at z = 4 * radius = 1.26 mm; NA 0.5.

Two routes, and the point is that they must agree:

* **RW-F** -- one full-aperture patch. Table S2: 1 incident ray, 1.1e6 secondary,
  1 batch, 8.086 GB, 0.097 s, MSE 4.414e-10, NCC 0.997. This is the exactness
  anchor: one patch as wide as the aperture must reproduce the independent ASM.
* **RW-P** -- sub-aperture patches. Table S2: patch 50^2, pad factor 4, 1.6e4
  incident, 1e4 secondary, 2 batches, 29.213 GB, 2.275 s. This is the direct
  demonstration of the SI S2 convergence claim.

Optiland is **not** exercised here. demo2 is a bare SLM and a sensor with no
refractive surface, and saying so is more useful than a record that lets a
reader infer a ray-engine validation that did not happen.

The paper's own numbers are context, never a pass threshold: different
implementation, different ray budget, different reconstruction algorithm.

Three corrections to the notebook are made deliberately and each is recorded:

1. the notebook accumulates **intensity** (`huygens_psf`'s 4th return value is
   `|field|^2`) and then squares the total again; SI eq S5 requires coherent
   *field* accumulation. Both are computed.
2. the notebook's `flip(phase, dims=[0,1])` compensates DeepLens's
   `Ray.flip_xy`, which this pipeline does not have. Unflipped is used; both
   orientations are scored once.
3. `ComplexField.coordinates` puts coordinate zero at index `n // 2`; upstream
   uses `(n-1)/2`. This repository's rule is kept.

Run:
    ./run.sh python benchmarks/probes/ray_wave/demo2_hologram.py --preset smoke
    MOA_GPUS=device=0 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo2_hologram.py --preset paper
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
    RECORDS,
    build_doe,
    device_memory_stats,
    enable_x64_if_needed,
    environment,
    mse_unit_sum,
    ncc,
    ncc_uncentred,
    patch_route,
    relative_l2,
    write_record,
)

from verification.asm_oracle import angular_spectrum_float64

WAVELENGTH_M = 0.7e-6
PITCH_M = 6.3e-6
GRID_N = 100
SENSOR_Z_M = 4 * (GRID_N // 2) * PITCH_M  # 1.26 mm

#: SI Table S2, System 2, verbatim. Recorded next to every measurement so the
#: comparison is to the paper rather than to a remembered version of it.
TABLE_S2 = {
    "rw_f": {
        "patch": "full",
        "incident_rays": 1,
        "secondary_rays": 1.1e6,
        "batches": 1,
        "peak_memory_gb": 8.086,
        "runtime_s": 0.097,
        "mse": 4.414e-10,
        "ncc": 0.997,
        "hardware": "1x NVIDIA RTX A6000 48 GB, CUDA 12.4",
    },
    "rw_p": {
        "patch_px": 50,
        "pad_factor": 4,
        "incident_rays": 1.6e4,
        "secondary_rays_per_incident": 1e4,
        "batches": 2,
        "peak_memory_gb": 29.213,
        "runtime_s": 2.275,
        "hardware": "1x NVIDIA RTX A6000 48 GB, CUDA 12.4",
    },
}


def oracle(transmission: np.ndarray, *, pad_to: int) -> np.ndarray:
    """The independent reference: zero-pad, propagate in float64, crop back.

    Upstream calls `asm(..., padding=True)`, which doubles the grid. `pad_to` is
    exposed because *the padding is part of the reference*, not a detail of it:
    a discrete ASM is periodic with its padded period, and comparing a patch
    route at period `pad_px` against an oracle at a different period measures
    the difference in wraparound rather than an error in either. The exactness
    comparison matches the two; the paper-configuration comparison uses 2x.
    """
    ny, nx = transmission.shape
    padded = np.zeros((pad_to, pad_to), dtype=np.complex128)
    y0, x0 = (pad_to - ny) // 2, (pad_to - nx) // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = transmission
    propagated = angular_spectrum_float64(
        padded,
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=PITCH_M,
        z_m=SENSOR_Z_M,
    )
    return np.asarray(propagated)[y0 : y0 + ny, x0 : x0 + nx]


def score(field: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    intensity = np.abs(field) ** 2
    reference_intensity = np.abs(reference) ** 2
    return {
        "ncc_intensity": ncc(intensity, reference_intensity),
        "ncc_intensity_uncentred": ncc_uncentred(intensity, reference_intensity),
        "mse_intensity_unit_sum": mse_unit_sum(intensity, reference_intensity),
        "relative_l2_field": relative_l2(field, reference),
        "relative_l2_field_after_global_phase": _phase_free_l2(field, reference),
    }


def _phase_free_l2(field: np.ndarray, reference: np.ndarray) -> float:
    """``||a e^{-i arg<a,b>} - b|| / ||b||`` -- the error a global phase cannot explain.

    Reported separately because a constant phase offset between the two routes
    is physically meaningless (the sensor measures intensity) but dominates a
    raw complex L2. Keeping both makes it visible whether a disagreement is a
    real field error or only a reference-plane phase.
    """
    a = np.asarray(field).ravel()
    b = np.asarray(reference).ravel()
    inner = np.vdot(b, a)
    if inner == 0:
        return relative_l2(field, reference)
    aligned = a * np.exp(-1j * np.angle(inner))
    return float(np.linalg.norm(aligned - b) / np.linalg.norm(b))


#: Every route is `(patch_px, pad_factor, patch_count, secondary_count, batches,
#: precision)`. `patch_count=None` means the single full-aperture patch;
#: `secondary_count=None` means enumerate every propagating mode.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    # Small enough for CPU and for the shared machine's memory budget. The
    # numbers are not the paper's and the record says so.
    "smoke": {
        "rw_f": {
            "patch_px": None,
            "pad_factor": 2,
            "patch_count": None,
            "secondary_count": None,
            "batches": 1,
            "precision": "fp64",
        },
        "rw_p": {
            "patch_px": 51,
            "pad_factor": 4,
            "patch_count": 200,
            "secondary_count": 500,
            "batches": 4,
            "precision": "fp64",
        },
    },
    # SI Table S2, System 2, with three forced substitutions, each recorded:
    #
    #   patch 50 -> 51        an even patch has no centre sample
    #   patch Full -> 99      same rule; the dropped row and column are zero
    #   2 batches -> 40       our reconstruction is O(rays x pixels) and the
    #                         paper's is O(rays) + one FFT. 1.6e8 rays at 100^2
    #                         needs 128 GB of separable factors in one call;
    #                         40 chunks holds it to ~6 GB. Batching cannot
    #                         change the estimator -- the 1/N is applied once at
    #                         finalize -- so this costs accuracy nothing and is
    #                         reported as the cost-model evidence it is.
    "paper": {
        # The exactness anchor: deterministic, every propagating mode.
        "rw_f": {
            "patch_px": None,
            "pad_factor": 2,
            "patch_count": None,
            "secondary_count": None,
            "batches": 1,
            "precision": "fp64",
        },
        # The same route at Table S2's stochastic budget, for comparability.
        "rw_f_paper_budget": {
            "patch_px": None,
            "pad_factor": 2,
            "patch_count": None,
            "secondary_count": 1_100_000,
            "batches": 1,
            "precision": "fp64",
        },
        "rw_p": {
            "patch_px": 51,
            "pad_factor": 4,
            "patch_count": 16_000,
            "secondary_count": 10_000,
            "batches": 40,
            "precision": "fp32",
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--routes", default="rw_f,rw_p")
    parser.add_argument("--batches", type=int, default=None, help="override the batch count")
    parser.add_argument(
        "--reconstruction",
        choices=("ramp_sum", "kspace_splat"),
        default="ramp_sum",
        help="ramp_sum is the exact O(rays x pixels) route; kspace_splat is CHE-101's fast path",
    )
    parser.add_argument(
        "--kspace-grid",
        default="matched",
        help=(
            "'matched' derives the k-grid that puts every drawn spectral bin on a "
            "node, which is the only setting under which this route measures "
            "exactness rather than interpolation; otherwise an oversampling factor"
        ),
    )
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--save-fields",
        action="store_true",
        help=(
            "also write <name>_fields.npz with each route's complex sensor field "
            "and both oracles, so a figure can be rendered without re-running the "
            "94.9 s RW-P route"
        ),
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    x64 = enable_x64_if_needed(
        backend=args.backend,
        precisions=[preset[r]["precision"] for r in routes if r in preset],
    )

    doe = build_doe("demo2_smile_phase_profile.npy", pitch_m=PITCH_M)
    flipped = build_doe("demo2_smile_phase_profile.npy", pitch_m=PITCH_M, flip=True)

    # The full-aperture patch is one sample narrower than the grid, because
    # `patch_px` must be odd and the grid is 100. Whether that loses anything is
    # a property of THIS mask, so it is checked rather than argued: the circular
    # amplitude mask has radius exactly 50 px on an origin at index 50, so the
    # outermost row and column sit at |y| = r and are excluded by the strict
    # inequality. If a future mask changes that, the record says so.
    full_patch_px = GRID_N - 1

    # Two oracles, because "the oracle" is not well defined until the padding
    # is. A discrete ASM is periodic with its padded period, so a comparison is
    # only an error measurement when the two periods match; against a different
    # period it measures wraparound. Both are reported:
    #
    #   * MATCHED -- the oracle padded to the route's own pad. This is the
    #     exactness comparison, and the only one AC 1 can be read from.
    #   * UPSTREAM -- padded 2x, exactly `asm(..., padding=True)`. This is the
    #     nearer-aperiodic, more physical field, and the one the paper's numbers
    #     describe.
    #
    # Measured on the full-aperture route: matched gives 7.3e-13 and upstream
    # gives 8.8e-3, and the second is the residual wraparound between period 199
    # and period 200 rather than an error in either.
    upstream_reference = oracle(doe.transmission, pad_to=2 * GRID_N)
    upstream_reference_flipped = oracle(flipped.transmission, pad_to=2 * GRID_N)

    record: dict[str, Any] = {
        "probe": "demo2_hologram",
        "issue": "CHE-96",
        "system": "paper Fig 5b / SI Table S2 System 2 -- planar hologram, free space",
        "preset": args.preset,
        "environment": {**environment(), **x64},
        "configuration": {
            "wavelength_m": WAVELENGTH_M,
            "sample_pitch_m": PITCH_M,
            "grid": [GRID_N, GRID_N],
            "aperture_radius_m": doe.radius_m,
            "sensor_z_m": SENSOR_Z_M,
            "numerical_aperture": 0.5,
            "seed": args.seed,
            "backend_requested": args.backend,
            "reconstruction": args.reconstruction,
            "kspace_grid_request": args.kspace_grid,
            "optiland_used": False,
            "optiland_note": (
                "demo2 is a bare SLM and sensor with no refractive surface, so no "
                "ray engine is exercised. Recorded explicitly so a reader cannot "
                "infer a ray-engine validation that did not happen."
            ),
        },
        "conventions": {
            "origin_rule": "coordinate zero at index n // 2 (upstream uses (n-1)/2)",
            "phase_orientation": (
                "unflipped. The notebook's flip(phase, dims=[0,1]) compensates "
                "DeepLens's Ray.flip_xy, which this pipeline does not have. Both "
                "orientations are scored below."
            ),
            "accumulation": (
                "coherent field, per SI eq S5. The notebook sums |field|^2 and "
                "squares the total; that variant is computed alongside."
            ),
            "reconstruction_normalization": "one_over_n, applied once at finalize",
            "oracle": (
                "verification/asm_oracle.angular_spectrum_float64, cropped back. "
                "Independent of the coupler under test. Scored at TWO paddings: "
                "matched to the route's own pad (the exactness comparison -- a "
                "discrete ASM is periodic with its padded period, so only a "
                "matched period measures an error rather than a wraparound "
                "difference), and 2x, which is upstream's asm(..., padding=True) "
                "and the nearer-aperiodic physical field."
            ),
            "opl_convention": (
                "the patch step resets OPL to zero at the DOE plane; the bundle is "
                "then advanced to the sensor along each ray's own direction, which "
                "is exact rather than paraxial."
            ),
            "full_aperture_patch_px": full_patch_px,
            "full_aperture_patch_note": (
                f"{full_patch_px}, not {GRID_N}: patch_px must be odd. The circular "
                "mask has radius exactly 50 px about index 50, so the outermost row "
                "and column are identically zero and the odd patch loses nothing. "
                f"Checked: dropped_border_is_empty={doe.dropped_border_is_empty}."
            ),
            "dropped_border_is_empty": doe.dropped_border_is_empty,
        },
        "paper_table_s2": TABLE_S2,
        "paper_numbers_are_context_not_thresholds": (
            "different implementation, different ray budget, different "
            "reconstruction algorithm. Quoting NCC 0.997 as a gate would be "
            "circular validation."
        ),
        "routes": {},
    }

    for name in routes:
        if name not in preset:
            raise SystemExit(f"{name!r} is not a route of preset {args.preset!r}")
        settings = dict(preset[name])
        if args.batches is not None:
            settings["batches"] = args.batches
        run = patch_route(
            doe,
            wavelength_m=WAVELENGTH_M,
            sensor_z_m=SENSOR_Z_M,
            sensor_shape=(GRID_N, GRID_N),
            patch_px=settings["patch_px"] or full_patch_px,
            pad_factor=settings["pad_factor"],
            patch_count=settings["patch_count"],
            secondary_count=settings["secondary_count"],
            batches=settings["batches"],
            seed=args.seed,
            backend=args.backend,
            precision=settings["precision"],
            reconstruction=args.reconstruction,
            kspace_grid_shape=("matched" if args.kspace_grid == "matched" else None),
            kspace_oversample=(
                None if args.kspace_grid == "matched" else float(args.kspace_grid)
            ),
        )
        run["requested"] = settings
        run["device_memory"] = device_memory_stats()
        record["routes"][name] = _route_record(
            run, doe, upstream_reference, upstream_reference_flipped, oracle
        )

    if "rw_f" in record["routes"] and "rw_p" in record["routes"]:
        f_field = record["routes"]["rw_f"]["_field"]
        p_field = record["routes"]["rw_p"]["_field"]
        record["route_agreement"] = {
            "note": (
                "SI S2 relation (2): sub-aperture patches uniformly covering the "
                "aperture must converge to the full-aperture response. This is the "
                "cross-check that does not involve the oracle at all."
            ),
            **score(p_field, f_field),
        }
    saved_fields = {name: route["_field"] for name, route in record["routes"].items()}
    for route in record["routes"].values():
        route.pop("_field", None)

    name = args.output_name or f"demo2_{args.preset}_{args.backend}"
    path = write_record(name, record)
    print(f"wrote {path}")
    if args.save_fields:
        # Both oracles go in the same file as the routes. A saved field is only
        # comparable against the oracle it was scored with, and the pad is part
        # of the oracle's identity, so storing them apart would let a later
        # figure pair a route with the wrong period.
        fields_path = RECORDS / f"{name}_fields.npz"
        RECORDS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fields_path,
            transmission=doe.transmission,
            oracle_upstream_pad=upstream_reference,
            **{
                f"oracle_matched_pad{record['routes'][n]['plan']['pad_px']}": oracle(
                    doe.transmission, pad_to=record["routes"][n]["plan"]["pad_px"]
                )
                for n in saved_fields
            },
            **{f"route_{n}": field for n, field in saved_fields.items()},
        )
        print(f"wrote {fields_path}")
    for name, route in record["routes"].items():
        print(
            f"  {name}: NCC {route['vs_oracle']['ncc_intensity']:.6f}  "
            f"rel-L2 {route['vs_oracle']['relative_l2_field_after_global_phase']:.4e}  "
            f"rays {route['total_rays']:,}  {route['wall_clock_s']:.2f} s"
        )
    if "route_agreement" in record:
        print(f"  RW-P vs RW-F: NCC {record['route_agreement']['ncc_intensity']:.6f}")
    return 0


def _route_record(run, doe, upstream_reference, upstream_reference_flipped, oracle_fn):
    field = run.pop("field")
    notebook = run.pop("notebook_variant_intensity")
    matched_reference = oracle_fn(doe.transmission, pad_to=run["plan"]["pad_px"])
    upstream_intensity = np.abs(upstream_reference) ** 2
    return {
        **run,
        "_field": field,
        "vs_oracle": {
            "note": (
                "MATCHED periodicity: the oracle zero-padded to this route's own "
                "pad. The exactness comparison, and the only one AC 1 can be read "
                "from -- a discrete ASM is periodic with its padded period, so a "
                "mismatched comparison measures wraparound rather than error."
            ),
            "oracle_pad_px": run["plan"]["pad_px"],
            **score(field, matched_reference),
        },
        "vs_oracle_upstream_padding": {
            "note": (
                "UPSTREAM configuration: the oracle at 2x padding, which is "
                "asm(..., padding=True). The nearer-aperiodic physical field, and "
                "the one the paper's own numbers describe. On the full-aperture "
                "route the gap to the matched comparison is the residual "
                "wraparound between period 199 and period 200."
            ),
            "oracle_pad_px": 2 * GRID_N,
            **score(field, upstream_reference),
        },
        "vs_oracle_flipped_mask": {
            "note": (
                "the same reconstruction scored against the oracle for the FLIPPED "
                "mask, at upstream padding. If the notebook's flip were physically "
                "required here, this would be the better score. Reported once "
                "rather than argued."
            ),
            **score(field, upstream_reference_flipped),
        },
        "notebook_variant": {
            "note": (
                "the literal notebook: |field|^2 accumulated per batch, then the "
                "total squared again. Not a coherent field, so it cannot match a "
                "field oracle; reported so the size of the difference is visible."
            ),
            "ncc_intensity": ncc(notebook, upstream_intensity),
            "mse_intensity_unit_sum": mse_unit_sum(notebook, upstream_intensity),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
