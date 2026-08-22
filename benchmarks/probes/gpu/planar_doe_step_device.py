#!/usr/bin/env python3
"""Does the batched planar DOE step execute on a CUDA device, and stay there?

Two separate questions, and only the second is interesting. Anything will
"run on the GPU" if the arrays are quietly copied to the host first; what the
step has to do is keep the accumulation, the transmission, the transform and
the resampling on the device, and hand back rays that are still there.

The exactness limit is re-measured on the device too. It is the composed step's
only oracle, and a device is precisely where a precision assumption goes wrong
silently -- XLA:GPU computes complex64 matmuls in TF32 by default, which this
repository has measured at 1500x less accurate than complex64 while the array
still reports `dtype=complex64`.

Run in a dedicated GPU session:
    MOA_GPUS=device=0 ./run.sh --gpu python benchmarks/probes/gpu/planar_doe_step_device.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from core.boundary import ComplexField, ReferencePlane
from couplers.cascade import planar_doe_step
from couplers.ray_to_wave import Projection, ray_to_wave
from couplers.wave_to_ray import (
    decompose,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)

GRID_N = 64
WAVELENGTH_M = 500e-9
PITCH_M = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    devices = [str(d) for d in jax.devices()]
    if not any("cuda" in d.lower() or "gpu" in d.lower() for d in devices):
        print(json.dumps({"status": "blocked", "reason": "no CUDA device", "devices": devices}))
        return 1

    n = args.grid_n
    plane = ReferencePlane(name="doe", z_m=0.0)
    pitch = (PITCH_M, PITCH_M)
    rng = np.random.default_rng(20260822)

    # Built on the device, in complex64, which is what Chromatix's floor makes
    # the realistic dtype for anything composed with the wave leg.
    field = ComplexField(
        u=jnp.asarray(
            (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
        ),
        sample_pitch_m=pitch,
        wavelength_m=WAVELENGTH_M,
        reference_plane=plane,
    )
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)
    doe = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(n, n))).astype(np.complex64)

    outgoing, transmitted, diagnostics = planar_doe_step(
        incident,
        doe,
        grid_shape=(n, n),
        sample_pitch_m=pitch,
        plane=plane,
        launch_positions_xy_m=np.zeros((1, 2)),
        secondary_count=None,
    )
    rebuilt, _ = ray_to_wave(
        outgoing,
        grid_shape=(n, n),
        sample_pitch_m=pitch,
        plane=plane,
        projection=Projection.ASM_CONSISTENT,
    )
    error = float(jnp.linalg.norm(rebuilt.u - transmitted.u) / jnp.linalg.norm(transmitted.u))

    record = {
        "probe": "planar_doe_step_device",
        "coupler": "C_PLANAR_DOE_STEP",
        "jax_devices": devices,
        "configuration": {
            "grid": [n, n],
            "wavelength_m": WAVELENGTH_M,
            "sample_pitch_m": PITCH_M,
            "secondary_count": None,
            "mode": "full enumeration -- the deterministic exactness limit",
        },
        # Read OFF THE ARRAYS, never off the request. A requested device is not
        # evidence of an actual one, and this repository has a measured case of
        # a successful run reporting the device it was asked for rather than the
        # one it used.
        "actual": {
            "outgoing_direction_device": str(outgoing.directions.device),
            "outgoing_direction_dtype": str(outgoing.directions.dtype),
            "transmitted_field_device": str(transmitted.u.device),
            "transmitted_field_dtype": str(transmitted.u.dtype),
        },
        "exactness_limit_relative_l2": error,
        "outgoing_ray_count": int(outgoing.count),
        "cascade": diagnostics.as_dict(),
    }
    # complex64 carries ~1e-7 relative precision per operation and this
    # accumulates one wavelet per propagating mode, so the float64 gate of 1e-12
    # is not the right bound here. What is asserted is that the limit is at
    # complex64 round-off rather than at a sampling error's scale.
    record["verdict"] = (
        "the exactness limit holds on the device at complex64 round-off"
        if error < 1e-4
        else f"the exactness limit does NOT hold on the device: {error:.3e}"
    )

    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if error < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
