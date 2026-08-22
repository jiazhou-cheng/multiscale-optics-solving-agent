"""The Optiland gradient path at its default precision vs a declared one (CHE-61).

Quantifies what changed when the adapter started calling ``set_precision``
explicitly. The recorded probe evidence in
``knowledge/solvers/optiland/expected/gradient_probe.json`` was captured without
that call, i.e. at Optiland's torch default of float32, so:

* ``declared_float32`` reproduces the record bit-identically (relative error 0);
* ``declared_float64`` -- the adapter's default, and the precision it always
  claimed -- differs from the record by ~1.3e-05 on the objective and ~2.3e-06 on
  the gradient, and is the more accurate of the two.

Those two numbers are the tolerances the parameterized adapter test uses, so this
probe is where they come from.

    ./run.sh python benchmarks/probes/precision/grad_precision.py
"""

import json
from pathlib import Path

import optiland.backend as be
import torch

from adapters.optiland_builder import build_optiland_system
from registry.prescriptions import resolve_prescription

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "knowledge" / "solvers" / "optiland" / "expected" / "gradient_probe.json"

R0 = 1.6911
WAVELENGTH_UM = 0.55
NUM_RAYS = 64
PARAMETER_SURFACE = 1


def run(precision: str | None) -> dict[str, object]:
    """Trace and differentiate, optionally declaring a backend precision first.

    The radius leaf is float64 in every case, matching what the recorded probe
    did. That is what makes the default run a *mixed*-precision computation: a
    float64 leaf feeding a float32 backend.
    """
    be.set_backend("torch")
    if precision is not None:
        be.set_precision(precision)
    lens = build_optiland_system(resolve_prescription("ReverseTelephoto"))
    radius = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
    lens.surfaces.surfaces[PARAMETER_SURFACE].geometry.radius = radius
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS)
    objective = (rays.x**2 + rays.y**2).mean()
    objective.backward()
    return {
        "backend_precision": be.get_precision(),
        "traced_dtype": str(rays.x.dtype),
        "objective": float(objective.item()),
        "grad": float(radius.grad.item()),
    }


def main() -> None:
    recorded = json.loads(RECORD.read_text())
    out: dict[str, object] = {"recorded": recorded}

    for label, precision in (
        ("torch_backend_default", None),
        ("declared_float64", "float64"),
        ("declared_float32", "float32"),
    ):
        measured = run(precision)
        measured["objective_rel_vs_recorded"] = abs(
            measured["objective"] - recorded["objective_value"]
        ) / abs(recorded["objective_value"])
        measured["grad_rel_vs_recorded"] = abs(
            measured["grad"] - recorded["grad_native_autodiff"]
        ) / abs(recorded["grad_native_autodiff"])
        # The autodiff-vs-finite-difference gap the record documents as
        # not-yet-root-caused, recomputed at each precision so a precision change
        # cannot be mistaken for progress on it.
        measured["grad_rel_vs_recorded_finite_difference"] = abs(
            measured["grad"] - recorded["grad_finite_difference"]
        ) / abs(recorded["grad_finite_difference"])
        out[label] = measured

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
