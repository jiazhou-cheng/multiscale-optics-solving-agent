"""B3-PSF-SINGLET-01, end to end: executor -> record -> verifier.

CHE-115 (M3.3), partial. The substrate proof: one real workload -- a real
Optiland trace, the real C_RAY_TO_WAVE reconstruction, real Chromatix
angular-spectrum propagation -- run through ``GraphExecutor``, measured, and
handed to ``verify()`` against the ``B3-PSF-SINGLET`` family.

What this file is, and what it deliberately is not
--------------------------------------------------
It is the smallest thing that closes the loop:

    BenchmarkFamily / BenchmarkInstance
      -> GraphExecutor          emits an ExecutionRecord   (what happened)
      -> verify(...)            emits a VerificationResult (what it means)

It is **not** a bespoke driver of the kind M3.1 exists to remove. The four steps
below are: build a graph from the instance's parameters, run it, measure the
terminal field, and verify. Nothing here sequences nodes, plumbs artifacts,
times anything, guards memory or writes a record -- all of that is the
executor's now, and that is the whole difference from
``benchmarks/physics/L2-PSF-01/run_benchmark.py``, which is 600 lines and does
all of it by hand.

What remains per-family is the *measurement*: turning a terminal ComplexField
into a number an oracle can be compared against. That is physics and it differs
per family by construction; the executor cannot do it and should not try.

The number, and why it is NOT comparable to the frozen one
-----------------------------------------------------------
This run reads ``1.18e-2``. The committed configuration reads ``2.21e-3``. They
are not the same measurement and the difference is **not** attributed to the
migration, because two things differ by construction:

1. **Ray count.** This graph traces 256 hexapolar rings; the frozen
   configuration is 512 rings, 787,969 rays.
2. **The window, and this is the larger difference.** The frozen gate is a
   *radial-profile* residual over a 5-Airy-radius disc, implemented as
   ``_profile_residual`` inside ``benchmarks/probes/psf_oracle_verification.py``.
   This proof uses ``metrics.central_relative_l2_intensity`` over a centred half
   window, because reimplementing the profile residual here would fork it --
   which is the exact drift ``verification/metrics.py`` exists to stop.

So the honest statement is: **the loop closes and the fingerprint is not yet
reproduced.** Doing that needs the frozen 512-ring configuration *and* the
5-Airy-radius radial-profile residual promoted into ``verification/metrics.py``
as a named definition. Both are the rest of CHE-115, and re-recording this
number as the singlet's would be the thing M3.3 explicitly forbids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.execution import RunStatus
from core.paths import repository_root
from registry.loader import Registry
from runtime.executor import GraphExecutor
from verification.families import BenchmarkInstance
from verification.families.b3_composed import B3_PSF_SINGLET
from verification.metrics import central_relative_l2_intensity, power_ratio
from verification.psf_measurement import PsfNormalization, measure_psf_from_record
from verification.psf_oracles import airy_psf_on_grid
from verification.result import Measurement, UncertaintyBasis, VerificationResult
from verification.verifier import verify

__all__ = ["CANONICAL_PARAMETERS", "canonical_instance", "run_and_verify"]

ROOT = repository_root()
GRAPH_PATH = ROOT / "examples" / "graphs" / "ray_to_wave.yaml"

#: The instance this proof runs. Every value is one the graph actually uses, so
#: the instance and the graph cannot describe different computations.
CANONICAL_PARAMETERS: dict[str, Any] = {
    "prescription": "M3-SINGLET-REF",
    "field_angle_rad": 0.0,
    "wavelength_m": 5.5e-7,
    "numerical_aperture": 0.05171631827291936,
    # 256 rings, NOT the frozen 512. Stated on the instance rather than left to
    # be inferred from a record, so the difference from the committed 2.21e-3 is
    # a declared parameter rather than an unexplained drift.
    "pupil_rings": 256,
    "grid_n": 188,
    "pad_width": 566,
    "quadrature_weight": "weighted",
    "device": "cpu",
}


def canonical_instance() -> BenchmarkInstance:
    return B3_PSF_SINGLET.instantiate(
        "B3-PSF-SINGLET-01",
        CANONICAL_PARAMETERS,
        split_tag="extended",
        expected={
            "gate": "fft_oracle_intensity_relative_l2 <= 1.0e-3",
            "known_disposition": "NOT_MET at 2.21e-3 on the frozen 512-ring configuration",
        },
    )


def _measure(record: Any, instance: BenchmarkInstance) -> dict[str, Measurement]:
    """Terminal field -> the metrics the family declares.

    The one per-family step. The oracle is O1, the analytic Airy pattern, which
    shares no code and no traced data with the coupler it judges -- and it is
    evaluated on a **centred window**, which is what the frozen gate has always
    meant by "over the 5-Airy-radius gate disc". Reporting the whole-array number
    beside it is the CHE-44 audit: a centred metric cannot see an off-axis error,
    and on this on-axis instance the two should agree.
    """
    field_record = record.artifacts["wave:output_field"]
    measurement = measure_psf_from_record(field_record, normalization=PsfNormalization.PEAK)
    psf = measurement.intensity

    analytic = airy_psf_on_grid(
        shape=psf.shape,
        sample_pitch_m=measurement.sample_pitch_m,
        wavelength_m=float(instance.parameters["wavelength_m"]),
        numerical_aperture=float(instance.parameters["numerical_aperture"]),
    )

    # sqrt on both sides because the metric squares its inputs: the arrays here
    # are already intensities, and passing them straight in would compare the
    # squares of intensities.
    #
    # NOT the frozen gate's window. See the module docstring: the frozen number
    # is a radial-profile residual over a 5-Airy-radius disc, and this is a
    # centred half window. The two are different measurements of the same field.
    gated = central_relative_l2_intensity(np.sqrt(psf), np.sqrt(analytic), fraction=0.5)
    whole = central_relative_l2_intensity(np.sqrt(psf), np.sqrt(analytic), fraction=1.0)

    # The uncertainty is the grid's, not the estimator's: this is a deterministic
    # computation, and what bounds the number is that the frozen configuration
    # puts 2.44 pixels across the Airy radius (CHE-103) and is not converged for
    # radius-like quantities. Quoting a floating-point floor here would claim a
    # precision the sampling does not support.
    sampling_uncertainty = abs(whole - gated)

    return {
        "fft_oracle_intensity_relative_l2": Measurement(
            value=gated,
            uncertainty=sampling_uncertainty,
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                "centred half-window against the analytic Airy pattern. The spread "
                "against the whole-array value is the CHE-44 window sensitivity and is "
                "used as the error bar rather than a round-off floor, because the "
                "frozen grid is not converged for radius-like quantities."
            ),
        ),
        "handoff_power_ratio": Measurement(
            value=power_ratio(np.sqrt(psf), np.sqrt(analytic)),
            uncertainty=sampling_uncertainty,
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note="peak-normalized on both sides, so this compares shape rather than absolute power",
        ),
    }


def run_and_verify(*, seed: int | None = 1) -> tuple[Any, VerificationResult]:
    """Execute the graph, measure it, and verify it. Returns both records."""
    registry = Registry.from_package()
    instance = canonical_instance()

    graph = Registry.load_graph(GRAPH_PATH)
    record = GraphExecutor(registry).run(
        graph,
        seed=seed,
        instance_id=instance.instance_id,
        instance_fingerprint=instance.fingerprint,
    )

    if record.status is not RunStatus.SUCCEEDED:
        # No measurement is invented for a run that did not produce a field. The
        # verifier reads the record's refusal and reports the structured status,
        # which is the whole point of handing it the record rather than a number.
        return record, verify(B3_PSF_SINGLET, instance, record)

    return record, verify(
        B3_PSF_SINGLET, instance, record, measurements=_measure(record, instance)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    record, result = run_and_verify(seed=args.seed)

    print(f"run          {record.run_id}  status={record.status.value}")
    print(f"verification status={result.status.value}")
    for metric in result.physics_accuracy:
        gate = "" if metric.tolerance is None else f"  tol {metric.tolerance:.3g}"
        met = "" if metric.met is None else f"  met={metric.met}"
        print(
            f"  {metric.metric}: {metric.measured.value:.6e} "
            f"+/- {metric.measured.uncertainty:.1e}{gate}{met}"
        )
    print(f"gate trustworthy: {result.gate_is_trustworthy}")
    for diagnostic in result.diagnostics:
        print(f"  [{diagnostic.code.value}] {diagnostic.detail}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
