"""B3-PSF-SINGLET-01, end to end: executor -> record -> verifier.

CHE-115 (M3.3). The substrate proof: one real workload -- a real Optiland trace,
the real C_RAY_TO_WAVE reconstruction, real Chromatix angular-spectrum
propagation -- run through ``GraphExecutor`` from a committed graph document,
measured, and handed to ``verify()`` against the ``B3-PSF-SINGLET`` family.

What this file is, and what it deliberately is not
--------------------------------------------------
It is the smallest thing that closes the loop:

    BenchmarkFamily / BenchmarkInstance
      -> GraphExecutor          emits an ExecutionRecord   (what happened)
      -> verify(...)            emits a VerificationResult (what it means)

It is **not** a bespoke driver of the kind M3.1 exists to remove. The steps below
are: load a graph, run it, measure the terminal field, and verify. Nothing here
sequences nodes, plumbs artifacts, times anything, guards memory or writes a
record -- all of that is the executor's, and that is the whole difference from
``benchmarks/physics/L2-PSF-01/run_benchmark.py``, which is 600 lines and does
all of it by hand.

What remains per-family is the *measurement*: turning a terminal ComplexField
into a number an oracle can be compared against. That is physics, it differs per
family by construction, and CHE-36 established that it is a measurement rather
than a graph node.

The frozen number IS reproduced, bit-identically
------------------------------------------------
``fft_oracle_intensity_relative_l2 = 0.0022072391812867093`` -- the committed
value the ``B3-PSF-SINGLET`` gate disposition carries, produced before this
substrate existed by ``benchmarks/probes/quadrature_weight.py::characterize()``
calling ``sensor_handoff_convergence.py``'s helpers directly -- comes back off
the executor's record as the *same float64*, not as an approximation of it.
``tests/test_substrate_proof.py::test_the_frozen_number_is_reproduced_bit_identically``
asserts equality, not ``approx``.

Two things had to be true for that, and neither was true before this issue:

1. **The graph could not name the reconstruction plane.** ``handoff_plane``
   declares where the OPL is *referenced from*; the plane the field is
   reconstructed *at* was whatever the adapter exported. So the sensor-plane
   handoff CHE-38 established as the intended contract could not be written as a
   graph document. ``config['advance_to_z_m']`` on the edge is that plane, and
   ``couplers/handoff.py::advance_bundle_to_plane`` is the ray-domain advance,
   pinned equal to the probe's own copy on real traced rays by
   ``tests/test_ray_to_wave_node.py``.

2. **The measurement must not normalize twice.** Reading the PSF with
   ``PsfNormalization.PEAK`` and then handing it to a metric that peak-normalizes
   its own inputs costs exactly one ULP: measured at 256 rings, the double
   division reads ``0.0022035728530455903`` against the frozen
   ``0.002203572853045589``. That is a 5.9e-16 relative difference and it would
   have been perfectly easy to call round-off and re-record. It is a defect in
   the measurement call, and ``PsfNormalization.RAW`` is the fix: the metric is
   defined on ``|u|^2`` and does its own normalization.

What the wave node costs, measured rather than assumed
-----------------------------------------------------
In this configuration the handoff is exactly ON the sensor, so the required
post-handoff propagation is **zero** and ``M_WAVE_CHROMATIX`` is a complex64
round trip. That is the frozen configuration, not a shortcut, and the frozen
gate is therefore defined on the float64 sensor-plane field -- which is the
coupler's output, and which this file measures.

The round trip is not free, and what it costs is the gate metric's **error bar**
rather than a footnote: the same measurement on the terminal artifact reads
``0.0022070734366051404``, a 7.5e-5 relative shift from the float64 value,
entirely from Chromatix's unconditional complex64 cast. Both numbers are in the
metric's ``note``. Reporting only the terminal one would have moved the frozen
gate by more than round-off with no statement of why.

Because a zero-distance node proves nothing about the wave leg,
:func:`run_near_sensor_fine` runs the same graph with the reconstruction moved
0.001 R upstream -- CHE-38's own ``near_sensor_fine`` candidate -- so Chromatix
propagates a real distance. Both arms are variants of one committed document
through ``runtime.variants.with_config_overrides``, which is also how
:func:`run_negative_control` runs the family's ``opl-sign-flip`` control as a
graph rather than as a driver calling the coupler by hand.
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
from runtime.instance_runner import execute
from runtime.variants import with_config_overrides
from verification.evidence import InstanceRun, control_result, write_instance_record
from verification.families import BenchmarkInstance
from verification.families.b3_composed import B3_PSF_SINGLET
from verification.metrics import disc_relative_l2_intensity, power_ratio
from verification.psf_measurement import PsfNormalization, measure_psf_from_record
from verification.psf_oracles import airy_first_null_radius_m, airy_psf_on_grid
from verification.result import Measurement, UncertaintyBasis
from verification.verifier import verify

__all__ = [
    "CANONICAL_PARAMETERS",
    "FROZEN_OBSERVED",
    "GRAPH_PATH",
    "INSTANCE_ID",
    "NEAR_SENSOR_FINE_FRACTION_OF_R",
    "canonical_instance",
    "load_graph",
    "near_sensor_fine_graph",
    "opl_sign_flip_graph",
    "run_instance",
    "run_near_sensor_fine",
]

ROOT = repository_root()
GRAPH_PATH = ROOT / "examples" / "graphs" / "psf_singlet_sensor.yaml"

#: The one declared instance. The family's ``sampler`` is ``None`` with
#: ``SamplerAbsentReason.HISTORICAL_REGRESSION``: the frozen configuration is the
#: point every number in the residual investigation was measured at, and sampling
#: around it would make that investigation's own evidence incomparable.
INSTANCE_ID = "B3-PSF-SINGLET-01"

#: The committed value of the gate metric, carried by the family's
#: ``gate_disposition``. Restated here because this file is what has to reproduce
#: it, and a proof that reads its target out of the thing it is proving is not a
#: proof. Asserted equal to the family's own number by test.
FROZEN_OBSERVED = 0.0022072391812867093

#: The 5-Airy-radius gate disc, as radii. CHE-38's choice, unchanged: widening
#: the window is one of the ways a residual can be made to look smaller.
GATE_AIRY_RADII = 5.0

#: CHE-38's own declared handoff-plane candidate closest to the sensor that still
#: leaves a nonzero distance for Chromatix, as a fraction of R = the exit
#: pupil-to-image distance. Its Experiment D used this plane for exactly this
#: reason.
NEAR_SENSOR_FINE_FRACTION_OF_R = 0.001

#: Exit pupil z and image-plane z of M3-SINGLET-REF, so R can be formed without
#: re-reading the graph's numbers back out of it.
_PUPIL_Z_M = 6.814345991561233e-05
_SENSOR_Z_M = 4.90560476022521e-3

#: The instance this proof runs. Every value is one the committed graph actually
#: uses -- asserted key by key against the YAML by test -- so the instance and
#: the graph cannot describe different computations.
CANONICAL_PARAMETERS: dict[str, Any] = {
    "prescription": "M3-SINGLET-REF",
    "field_angle_rad": 0.0,
    "wavelength_m": 5.5e-7,
    "numerical_aperture": 0.05171631827291936,
    # 512 rings, 787,969 traced rays: the frozen count every number in the
    # residual investigation (CHE-47, CHE-103, CHE-117) was measured at.
    "pupil_rings": 512,
    "grid_n": 256,
    # Zero, and that is the configuration rather than an omission: the handoff is
    # on the sensor, so there is no distance left to propagate and no wraparound
    # to pad against.
    "pad_width": 0,
    "quadrature_weight": "weighted",
    "device": "cpu",
}


def canonical_instance() -> BenchmarkInstance:
    return B3_PSF_SINGLET.instantiate(
        INSTANCE_ID,
        CANONICAL_PARAMETERS,
        split_tag="extended",
        expected={
            "gate": "fft_oracle_intensity_relative_l2 <= 1.0e-3",
            "known_disposition": f"NOT_MET at {FROZEN_OBSERVED!r}, carried forward unwidened",
        },
    )


def load_graph() -> Any:
    """The committed graph document, unmodified."""
    return Registry.load_graph(GRAPH_PATH)


def near_sensor_fine_graph() -> Any:
    """The same graph with the reconstruction moved 0.001 R upstream.

    One key changes. The wave node reads ``target_plane_z_m`` and derives its
    distance from the input field's own declared plane, so moving the
    reconstruction is all it takes to give Chromatix real work -- which is the
    point: the frozen configuration's wave node propagates zero distance, and a
    three-node graph where one node is an identity is a two-node graph.
    """
    z_handoff = _SENSOR_Z_M - NEAR_SENSOR_FINE_FRACTION_OF_R * (_SENSOR_Z_M - _PUPIL_Z_M)
    return with_config_overrides(
        load_graph(),
        edges={"sensor_reconstruction": {"advance_to_z_m": z_handoff}},
        task_id="B3-PSF-SINGLET-01/near_sensor_fine",
    )


def opl_sign_flip_graph() -> Any:
    """The family's ``opl-sign-flip`` negative control, as a graph document.

    Negating the declared OPL conjugates the wavefront: a converging pupil field
    becomes diverging. The control is declared by the family and, until this
    issue, could only be run by a driver constructing the bundle itself -- so the
    executor could never record that it had run.
    """
    return with_config_overrides(
        load_graph(),
        edges={"sensor_reconstruction": {"perturbation": {"opl_sign": -1}}},
        task_id="B3-PSF-SINGLET-01/opl_sign_flip",
    )


def _oracle(grid_n: int, pitch_m: float, instance: BenchmarkInstance) -> np.ndarray:
    """O1, the analytic Airy pattern. Shares no code and no traced data."""
    return airy_psf_on_grid(
        shape=(grid_n, grid_n),
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=float(instance.parameters["wavelength_m"]),
        numerical_aperture=float(instance.parameters["numerical_aperture"]),
    )


def _gate_residual(record: Any, artifact_key: str, instance: BenchmarkInstance) -> float:
    """The frozen gate metric, evaluated on one artifact of the record.

    ``PsfNormalization.RAW`` on purpose. ``disc_relative_l2_intensity``
    peak-normalizes both of its inputs; measuring a pre-normalized PSF divides by
    the peak twice and costs one ULP, which is small enough to be mistaken for
    round-off and large enough to break a bit-identity claim.
    """
    measurement = measure_psf_from_record(
        record.artifacts[artifact_key], normalization=PsfNormalization.RAW
    )
    psf = np.asarray(measurement.intensity, dtype=np.float64)
    pitch_m = float(measurement.sample_pitch_m[0])
    radius_m = GATE_AIRY_RADII * airy_first_null_radius_m(
        float(instance.parameters["wavelength_m"]),
        float(instance.parameters["numerical_aperture"]),
    )
    return disc_relative_l2_intensity(
        psf,
        _oracle(psf.shape[0], pitch_m, instance),
        sample_pitch_m=pitch_m,
        max_radius_m=radius_m,
    )


#: The artifact the frozen gate is defined on: the coherent reconstruction at the
#: sensor plane, in float64, before Chromatix casts it. Keyed by
#: ``<edge id>:<target port>``, which is how the executor stores an edge output.
COUPLER_FIELD = "sensor_reconstruction:input_field"
#: The graph's terminal artifact. In this configuration it is the same field
#: after a zero-distance complex64 round trip.
TERMINAL_FIELD = "wave:output_field"


def _measure(record: Any, instance: BenchmarkInstance) -> dict[str, Measurement]:
    """Terminal field -> the metrics the family declares. The one per-family step.

    The gate is measured on the float64 sensor-plane field, because that is what
    the frozen number is: the handoff is on the sensor and the propagation is
    zero, so the sensor-plane field is the coupler's output. What the wave node
    costs is reported beside it rather than folded into it.
    """
    gate = _gate_residual(record, COUPLER_FIELD, instance)
    terminal = _gate_residual(record, TERMINAL_FIELD, instance)

    coupler = measure_psf_from_record(
        record.artifacts[COUPLER_FIELD], normalization=PsfNormalization.RAW
    )
    psf = np.asarray(coupler.intensity, dtype=np.float64)
    analytic = _oracle(psf.shape[0], float(coupler.sample_pitch_m[0]), instance)

    return {
        "fft_oracle_intensity_relative_l2": Measurement(
            value=gate,
            # The uncertainty is what routing the same field through the wave
            # node moves it by -- a measured property of this graph, not a
            # floating-point floor. Quoting round-off would claim a precision the
            # complex64 leg does not support.
            uncertainty=abs(terminal - gate),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                "peak-normalized, disc-masked relative L2 against the analytic Airy "
                f"pattern over the {GATE_AIRY_RADII:.0f}-Airy-radius gate disc, on the "
                "float64 sensor-plane reconstruction. The frozen definition: the "
                "handoff is ON the sensor, so the post-handoff propagation is zero and "
                "this field IS the sensor-plane field. The same measurement on the "
                f"graph's terminal artifact reads {terminal!r} -- Chromatix's "
                "unconditional complex64 cast, which is the error bar quoted here."
            ),
        ),
        "handoff_power_ratio": Measurement(
            # PEAK-normalized on both sides, and that is what this number means.
            # On raw amplitudes it reads 3.8e-14 -- Optiland's intensity weights
            # have no SI calibration, so an absolute power ratio against an
            # analytic Airy pattern is not a conservation measurement, it is a
            # unit mismatch. What survives normalization is where the power sits
            # relative to the peak, which is a shape statement and is reported as
            # one.
            value=power_ratio(
                np.sqrt(psf / psf.max()), np.sqrt(analytic / analytic.max())
            ),
            uncertainty=abs(terminal - gate),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                "peak-normalized on both sides, so this compares how the power is "
                "distributed relative to the peak rather than absolute power. It is "
                "NOT the family invariant HANDOFF_ENERGY_CLOSES: that invariant is "
                "about power crossing the handoff plane against power in the traced "
                "bundle, and nothing here measures the traced bundle's power, so it "
                "is reported without a tolerance rather than gated on the wrong "
                "quantity."
            ),
        ),
    }



def _control(
    instance: BenchmarkInstance, baseline: Measurement, *, seed: int | None
) -> dict[str, Any]:
    """``opl-sign-flip``, run as a graph variant and judged against the gate.

    Negating the declared OPL conjugates the wavefront: a converging pupil field
    becomes diverging. Until CHE-115 the only way to run this control was a
    driver building the bundle itself, so the executor could never record that
    it had run -- and ``verify()`` reported it ``NOT_RUN``, correctly.

    Decided against O1 only. O2, our own ASM/RS propagator, is built from the
    same traced pupil and must never be the thing that decides pass/fail.
    """
    record = execute(opl_sign_flip_graph(), instance, seed=seed)
    if record.status is not RunStatus.SUCCEEDED:
        return {}
    mutated = Measurement(
        value=_gate_residual(record, COUPLER_FIELD, instance),
        uncertainty=None,
        uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
        note=(
            "config['perturbation'] = {'opl_sign': -1} on the sensor_reconstruction "
            "edge of examples/graphs/psf_singlet_sensor.yaml, executed as its own "
            "graph. Its uncertainty is NOT_ESTIMATED rather than the baseline's "
            "complex64 bar: this arm is measured on the float64 reconstruction and "
            "nothing was run twice to bound it."
        ),
    )
    return {
        "opl-sign-flip": control_result(
            "opl-sign-flip",
            "fft_oracle_intensity_relative_l2",
            baseline=baseline,
            mutated=mutated,
            threshold=B3_PSF_SINGLET.tolerance_for(
                "fft_oracle_intensity_relative_l2"
            ).threshold,
            note="graph variant, not a driver-built bundle (CHE-115).",
        )
    }


def run_instance(
    instance_id: str = INSTANCE_ID, *, seed: int | None = 1, with_control: bool = True
) -> InstanceRun:
    """Execute the frozen graph, measure it, verify it.

    ``with_control=False`` skips the second 512-ring run the ``opl-sign-flip``
    control needs. The committed record is written *with* the control, because a
    record whose controls all read ``NOT_RUN`` cannot support a trustworthy gate;
    the default test gate takes the cheap arm and the slow arm is what re-derives
    the committed fingerprint.

    The baseline graph is executed once and the record is reused for both the
    measurement and the control's baseline. Running it twice would have been a
    tidier call into ``evidence.run_and_verify`` and 28 s of identical arithmetic.
    """
    if instance_id != INSTANCE_ID:
        raise ValueError(f"the only declared instance is {INSTANCE_ID!r}, got {instance_id!r}")
    instance = canonical_instance()
    record = execute(load_graph(), instance, seed=seed)

    measurements: dict[str, Measurement] | None = None
    controls: dict[str, Any] | None = None
    if record.status is RunStatus.SUCCEEDED:
        measurements = _measure(record, instance)
        if with_control:
            controls = _control(
                instance, measurements["fft_oracle_intensity_relative_l2"], seed=seed
            ) or None

    result = verify(
        B3_PSF_SINGLET,
        instance,
        record,
        measurements=measurements,
        negative_controls=controls,
    )
    return InstanceRun(
        family=B3_PSF_SINGLET, instance=instance, record=record, result=result
    )


def run_near_sensor_fine(*, seed: int | None = 1) -> dict[str, Any]:
    """The arm where ``M_WAVE_CHROMATIX`` does real work.

    Not part of the committed record: it is a different configuration from the
    frozen one, so folding its number into ``B3-PSF-SINGLET-01`` would report a
    measurement of a plane the instance does not declare. What it establishes is
    that the graph's middle node is not an identity.
    """
    instance = canonical_instance()
    graph = near_sensor_fine_graph()
    record = execute(graph, instance, seed=seed)
    advance_to_z_m = float(graph.edges[0].config["advance_to_z_m"])
    payload: dict[str, Any] = {
        "task_id": graph.task_id,
        "status": record.status.value,
        "advance_to_z_m": advance_to_z_m,
        "propagation_m": _SENSOR_Z_M - advance_to_z_m,
    }
    if record.status is RunStatus.SUCCEEDED:
        payload["terminal_relative_l2_vs_o1"] = _gate_residual(
            record, TERMINAL_FIELD, instance
        )
        payload["reconstruction_relative_l2_vs_o1"] = _gate_residual(
            record, COUPLER_FIELD, instance
        )
        wave = next(n for n in record.nodes if n.node_id == "wave")
        payload["wave_node_wall_seconds"] = wave.cost.wall_seconds if wave.cost else None
        payload["wave_node_actual_device"] = (
            wave.device_precision.actual_device if wave.device_precision else None
        )
    else:
        payload["error"] = next(
            (n.error_message for n in record.nodes if n.error_message), None
        )
    return {"record": record, **payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write benchmarks/instances/records/<id>.json"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--near-sensor-fine",
        action="store_true",
        help="also run the variant that gives M_WAVE_CHROMATIX a nonzero distance",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    run = run_instance(seed=args.seed)
    result = run.result
    print(f"run          {run.record.run_id}  status={run.record.status.value}")
    print(f"verification status={result.status.value}")
    for metric in result.physics_accuracy:
        gate = "" if metric.tolerance is None else f"  tol {metric.tolerance:.3g}"
        met = "" if metric.met is None else f"  met={metric.met}"
        print(f"  {metric.metric}: {metric.measured.value!r}{gate}{met}")
    residual = next(
        (
            m.measured.value
            for m in result.physics_accuracy
            if m.metric == "fft_oracle_intensity_relative_l2"
        ),
        None,
    )
    print(f"  frozen {FROZEN_OBSERVED!r}  bit_identical={residual == FROZEN_OBSERVED}")
    for control in result.negative_control_results:
        print(f"  control {control.control_id}: {control.outcome.value}  {control.note}")
    print(f"gate trustworthy: {result.gate_is_trustworthy}")
    for diagnostic in result.diagnostics:
        print(f"  [{diagnostic.code.value}] {diagnostic.detail}")
    print(f"scientific fingerprint: {run.fingerprint}")

    payload: dict[str, Any] = {"verification": result.model_dump(mode="json")}

    if args.near_sensor_fine:
        arm = run_near_sensor_fine(seed=args.seed)
        arm.pop("record")
        print("near_sensor_fine " + json.dumps(arm, sort_keys=True, default=float))
        payload["near_sensor_fine"] = arm

    if args.write:
        path = write_instance_record(run, driver="instances/b3_psf_singlet")
        print(f"wrote {path}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
