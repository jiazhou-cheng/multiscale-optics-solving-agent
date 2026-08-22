#!/usr/bin/env python3
"""Is the flat multinomial sampler the bottleneck at the paper's largest scale?

The reference implementation samples secondary directions in two stages --
marginal ``p(y)`` then conditional ``p(x|y)`` -- and gives two reasons: speed,
and CUDA's limit on the number of categories in one multinomial draw. This
repository draws from a **flat** distribution over propagating bins only, which
is already a smaller support than a full ``H x W``.

Smaller is not the same as small enough, so this measures it rather than
assuming. The configuration is the paper's SI Table S2 Hologram (RW-P) row --
501^2 grid, 4000 secondary rays -- which is the largest sampling load in the
benchmark set.

The decision rule is stated before the measurement, so it cannot be chosen
afterwards: adopt the marginal/conditional sampler only if the flat draw is a
**material fraction of the step**. Replacing a sampler that costs 1% of the run
with a more complicated one that costs 0.5% is not an optimization, it is a
second thing to get wrong.

Run:
    ./run.sh python benchmarks/probes/doe_step_sampler_cost.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.boundary import ComplexField, ReferencePlane
from couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    draw_indices,
    sampling_density,
    spectrum_to_rays,
)

#: SI Table S2, Hologram (RW-P): 501^2 samples, SSR 1e4. The grid is the
#: paper's; the secondary count here is the *per-draw* budget the sampler sees.
GRID_N = 501
SECONDARY_COUNT = 4000
WAVELENGTH_M = 0.7e-6
PITCH_M = 6.3e-6
REPEATS = 5


def _timed(fn, repeats: int = REPEATS) -> tuple[float, float]:
    """Best-of and median wall clock, in seconds.

    Best-of because the question is what the operation costs, not what a shared
    machine's scheduler did; the median is reported alongside so a large gap
    between them is visible rather than hidden by the choice.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples), float(np.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--secondary-count", type=int, default=SECONDARY_COUNT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(20260822)
    shape = (args.grid_n, args.grid_n)
    plane = ReferencePlane(name="doe", z_m=0.0)
    field = ComplexField(
        u=(rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128),
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        reference_plane=plane,
    )

    decompose_best, decompose_med = _timed(lambda: decompose(field))
    spectrum = decompose(field)

    density_best, density_med = _timed(
        lambda: sampling_density(spectrum, SamplingDensity.MAGNITUDE)
    )
    density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)

    draw_best, draw_med = _timed(
        lambda: draw_indices(density, args.secondary_count, np.random.default_rng(1))
    )
    indices = draw_indices(density, args.secondary_count, np.random.default_rng(1))

    emit_best, emit_med = _timed(
        lambda: spectrum_to_rays(
            spectrum, indices, density, launch_positions_xy_m=np.zeros((1, 2))
        )
    )

    total = decompose_best + density_best + draw_best + emit_best
    record = {
        "probe": "doe_step_sampler_cost",
        "question": (
            "is the flat multinomial draw a material fraction of the batched DOE "
            "step at the paper's largest sampling load?"
        ),
        "configuration": {
            "source": "SI Table S2, Hologram (RW-P)",
            "grid": [args.grid_n, args.grid_n],
            "grid_points": args.grid_n**2,
            "secondary_count": args.secondary_count,
            "propagating_bins": int(spectrum.propagating_count),
            "propagating_fraction_of_grid": float(
                spectrum.propagating_count / (args.grid_n**2)
            ),
            "wavelength_m": WAVELENGTH_M,
            "sample_pitch_m": PITCH_M,
            "dtype": "complex128",
            "device": "cpu",
            "repeats": REPEATS,
        },
        "seconds_best_of": {
            "decompose": decompose_best,
            "sampling_density": density_best,
            "draw_indices": draw_best,
            "spectrum_to_rays": emit_best,
            "total": total,
        },
        "seconds_median": {
            "decompose": decompose_med,
            "sampling_density": density_med,
            "draw_indices": draw_med,
            "spectrum_to_rays": emit_med,
        },
        "draw_fraction_of_total": draw_best / total,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
    }

    fraction = record["draw_fraction_of_total"]
    record["verdict"] = (
        "ADOPT the marginal/conditional sampler: the flat draw is "
        f"{fraction:.1%} of the step."
        if fraction > 0.20
        else (
            f"KEEP the flat sampler. The draw is {fraction:.1%} of the step at the "
            "paper's largest sampling load, so the two-stage sampler would optimize "
            "something that is not the bottleneck while adding a second place for a "
            "sampling bug to live. The upstream reasons do not transfer: this draw "
            "is over PROPAGATING BINS only "
            f"({record['configuration']['propagating_fraction_of_grid']:.1%} of the "
            "grid here), not over a flat H x W, and it runs on the host, where "
            "CUDA's category limit does not apply. Revisit if either changes -- a "
            "device-side draw is the case that would."
        )
    )

    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
