"""Optiland's *default* precision, per backend (CHE-61, PB4b).

The finding this probe exists for: **Optiland's torch backend defaults to
float32 while its numpy backend defaults to float64.** Before CHE-61 the project
adapter never called ``set_precision`` and reported ``dtype: 'float64'``
regardless, so every torch-backend run traced in float32 under a float64 label.

Also records that ``get_precision()`` answers with an int *width* (32 / 64) while
``set_precision()`` takes a dtype *name* ('float32' / 'float64') -- an asymmetry
the adapter normalizes.

    ./run.sh python benchmarks/probes/precision/default_precision.py
"""

import json

import optiland.backend as be
import torch

from registry.prescriptions import resolve_prescription
from solvers.optiland.builder import build_optiland_system

WAVELENGTH_UM = 0.55
NUM_RAYS = 64


def observe(backend: str) -> dict[str, object]:
    be.set_backend(backend)
    array = be.array([1.0, 2.0])
    observed: dict[str, object] = {
        "get_precision": be.get_precision(),
        "array_dtype": str(array.dtype),
    }
    try:
        observed["get_device"] = str(be.get_device())
    except Exception as exc:  # numpy backend has no device concept
        observed["get_device"] = f"{type(exc).__name__}: {exc}"
    return observed


def traced(precision: str | None) -> dict[str, object]:
    """Trace a real system, optionally declaring a precision first."""
    be.set_backend("torch")
    if precision is not None:
        be.set_precision(precision)
    lens = build_optiland_system(resolve_prescription("ReverseTelephoto"))
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
    objective = (rays.x**2 + rays.y**2).mean()
    return {
        "backend_precision": be.get_precision(),
        "traced_dtype": str(rays.x.dtype),
        "traced_device": str(rays.x.device),
        "objective": float(objective.item()),
    }


def main() -> None:
    out: dict[str, object] = {
        "torch_version": torch.__version__,
        "numpy_backend_defaults": observe("numpy"),
        "torch_backend_defaults": observe("torch"),
        "trace_at_torch_default": traced(None),
        "trace_at_declared_float64": traced("float64"),
        "trace_at_declared_float32": traced("float32"),
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
