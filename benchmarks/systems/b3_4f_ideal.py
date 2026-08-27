"""B3-4F-IDEAL, end to end: the ideal 4f relay, executed and verified.

CHE-144 (M2.8). Every instance is realized directly with ``numpy.fft`` --
``ifft2c(mask * fft2c(object_field))`` -- rather than through
``GraphExecutor``: there is no ``M_WAVE_CHROMATIX`` capability for an ideal
lens's Fourier-transforming property (only ``asm_propagate``, a free-space
propagation by a distance, which this system does not use), and adding one is
not required by this family's acceptance criteria. ``record_from_probe``
builds the ``ExecutionRecord`` for a computation that legitimately runs
outside the graph model, the same mechanism B0's contract cases use.

Run it::

    ./run.sh python benchmarks/systems/b3_4f_ideal.py --write
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np

from core.execution_record import DevicePrecisionObservation
from core.paths import repository_root
from runtime.instance_runner import record_from_probe
from verification.evidence import InstanceRun, control_result, write_instance_record
from verification.families.b3_4f_ideal import (
    B3_4F_IDEAL,
    CHECKED_ORDERS,
    order_coefficients,
)
from verification.families.schema import ValidityState
from verification.result import Measurement, UncertaintyBasis
from verification.verifier import verify

__all__ = ["declared_instance_ids", "run_all", "run_instance"]

ROOT = repository_root()
RECORDS_DIR = ROOT / "benchmarks" / "systems" / "records"

#: Near-machine-precision floor: this is a deterministic FFT of a
#: deterministically built field, so there is no ensemble to estimate an
#: uncertainty from. float64 round-off over a grid this size.
_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# The relay itself
# ---------------------------------------------------------------------------


def _fft2c(a: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def _ifft2c(a: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def _object_field(grid_n: int, waist_px: float) -> np.ndarray:
    axis = np.arange(grid_n) - grid_n // 2
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    return np.exp(-(ii**2 + jj**2) / waist_px**2).astype(np.complex128)


def _mask(
    grid_n: int, modulation_type: str, samples_per_period: float, phase_depth_rad: float
) -> np.ndarray:
    fx = np.fft.fftshift(np.fft.fftfreq(grid_n))
    fxfx, _ = np.meshgrid(fx, fx, indexing="ij")
    dfx = fx[1] - fx[0]
    period = samples_per_period * dfx
    if modulation_type == "sinusoidal_phase":
        return np.exp(1j * phase_depth_rad * np.sin(2 * np.pi * fxfx / period))
    if modulation_type == "binary_phase":
        frac = np.mod(fxfx / period, 1.0)
        phase = np.where(frac < 0.5, phase_depth_rad, 0.0)
        return np.exp(1j * phase)
    if modulation_type == "pure_carrier":
        return np.exp(1j * 2 * np.pi * fxfx / period)
    raise ValueError(f"unknown modulation_type {modulation_type!r}")


def relay(object_field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The thing under test: FT, Fourier-plane modulation, inverse FT."""
    return _ifft2c(mask * _fft2c(object_field))


def build(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(params["grid_n"])
    object_field = _object_field(grid_n, float(params["object_waist_pixels"]))
    mask = _mask(
        grid_n,
        str(params["modulation_type"]),
        float(params["samples_per_period"]),
        float(params["phase_depth_rad"]),
    )
    return object_field, mask, relay(object_field, mask)


# ---------------------------------------------------------------------------
# Per-order measurement, before reduction to a scalar metric
# ---------------------------------------------------------------------------


def _predicted_shift_px(n: int, grid_n: int, samples_per_period: float) -> float:
    """``x0_n = -n / Lf``, in pixels. The sign was derived and verified against
    ``scipy.special.jv`` while authoring this family; see the family module's
    docstring."""
    return -n * grid_n / samples_per_period


def _peak_search(intensity_row: np.ndarray, predicted_idx: int, window: int = 4) -> int:
    n = len(intensity_row)
    lo = max(0, predicted_idx - window)
    hi = min(n, predicted_idx + window + 1)
    local = intensity_row[lo:hi]
    return int(np.argmax(local)) + lo


def _order_measurements(
    object_field: np.ndarray,
    image: np.ndarray,
    params: dict[str, Any],
    *,
    prediction_samples_per_period: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-order raw measured quantities. Controls mutate these before reducing."""
    grid_n = int(params["grid_n"])
    center = grid_n // 2
    samples_per_period = float(params["samples_per_period"])
    predicted_spp = (
        float(prediction_samples_per_period)
        if prediction_samples_per_period is not None
        else samples_per_period
    )
    obj_peak = object_field[center, center]
    obj_peak_intensity = float(abs(obj_peak) ** 2)
    row = center
    intensity_row = np.abs(image[:, row]) ** 2

    analytic = order_coefficients(params)
    per_order: dict[int, dict[str, Any]] = {}
    for n in CHECKED_ORDERS:
        true_idx = int(center + round(_predicted_shift_px(n, grid_n, samples_per_period))) % grid_n
        predicted_idx = int(center + round(_predicted_shift_px(n, grid_n, predicted_spp))) % grid_n
        c_analytic = analytic[n]
        c_measured = complex(image[true_idx, row] / obj_peak)
        per_order[n] = {
            "true_idx": true_idx,
            "predicted_idx": predicted_idx,
            "found_idx": _peak_search(intensity_row, true_idx),
            "c_analytic": c_analytic,
            "c_measured": c_measured,
            "p_analytic": abs(c_analytic) ** 2,
            "p_measured": float(intensity_row[true_idx] / obj_peak_intensity),
        }
    return per_order


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def _order_power_relative_l2(per_order: dict[int, dict[str, Any]]) -> float:
    err = sum((v["p_measured"] - v["p_analytic"]) ** 2 for v in per_order.values())
    ref = sum(v["p_analytic"] ** 2 for v in per_order.values())
    return math.sqrt(err / ref) if ref > 0.0 else math.sqrt(err)


def _order_phase_error_rad(
    per_order: dict[int, dict[str, Any]], *, phase_sign: float = 1.0
) -> float:
    sq = 0.0
    count = 0
    for v in per_order.values():
        if v["p_analytic"] <= 1e-6:
            continue
        measured_phase = phase_sign * float(np.angle(v["c_measured"]))
        analytic_phase = float(np.angle(v["c_analytic"]))
        sq += _angle_diff(measured_phase, analytic_phase) ** 2
        count += 1
    return math.sqrt(sq / count) if count else 0.0


def _order_location_error_frac(per_order: dict[int, dict[str, Any]], x0_1_px: float) -> float:
    """RMS location offset, over orders with non-negligible analytic power.

    A near-zero-power order (``pure_carrier``'s order 0, ``binary_phase``'s
    even orders) has no real peak to find, so its "location" is whatever the
    local floating-point floor happens to argmax to -- excluded for the same
    reason ``_order_phase_error_rad`` excludes it.
    """
    if x0_1_px <= 0.0:
        return 0.0
    checked = [v for v in per_order.values() if v["p_analytic"] > 1e-6]
    if not checked:
        return 0.0
    sq = sum(((v["found_idx"] - v["predicted_idx"]) / x0_1_px) ** 2 for v in checked)
    return math.sqrt(sq / len(checked))


def _total_power_relative_error(object_field: np.ndarray, image: np.ndarray) -> float:
    p_in = float(np.sum(np.abs(object_field) ** 2))
    p_out = float(np.sum(np.abs(image) ** 2))
    return abs(p_out / p_in - 1.0)


def _measure(
    object_field: np.ndarray, image: np.ndarray, params: dict[str, Any]
) -> dict[str, Measurement]:
    grid_n = int(params["grid_n"])
    samples_per_period = float(params["samples_per_period"])
    x0_1_px = grid_n / samples_per_period

    per_order = _order_measurements(object_field, image, params)

    return {
        "order_power_relative_l2": Measurement(
            value=_order_power_relative_l2(per_order),
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        "order_phase_error_rad": Measurement(
            value=_order_phase_error_rad(per_order),
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        "order_location_error_frac": Measurement(
            value=_order_location_error_frac(per_order, x0_1_px),
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        "total_power_relative_error": Measurement(
            value=_total_power_relative_error(object_field, image),
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
    }


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _controls(
    object_field: np.ndarray,
    mask: np.ndarray,
    image: np.ndarray,
    params: dict[str, Any],
    baseline: dict[str, Measurement],
) -> dict[str, Any]:
    grid_n = int(params["grid_n"])
    samples_per_period = float(params["samples_per_period"])
    x0_1_px = grid_n / samples_per_period

    results: dict[str, Any] = {}

    validity_state, _margins = B3_4F_IDEAL.evaluate_validity(params)
    if validity_state is not ValidityState.INSIDE:
        # A control's "detection margin" is a ratio against the baseline. On a
        # near-boundary or outside instance the baseline is already broken by
        # real physics (aliasing), which is not the same defect the control is
        # meant to catch and would report as a false "fires backwards" or a
        # false "did not fire". These controls are demonstrated on the clean
        # instances instead; see the family's negative_controls declarations.
        return results

    # 1. phasor-sign-flip: negate the measured phase before comparing. Only
    # meaningful where the analytic coefficients are genuinely complex --
    # sinusoidal_phase and pure_carrier are real-valued (phase in {0, pi}),
    # and negating 0 or pi leaves the angle unchanged, so this control cannot
    # detect anything there. binary_phase's odd orders are complex whenever
    # phase_depth_rad is not a multiple of pi, which is where this control is
    # exercised.
    per_order = _order_measurements(object_field, image, params)
    has_complex_orders = any(
        abs(v["c_analytic"].imag) > 1e-9 for v in per_order.values() if v["p_analytic"] > 1e-6
    )
    if has_complex_orders:
        mutated_phase = Measurement(
            value=_order_phase_error_rad(per_order, phase_sign=-1.0),
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        )
        results["phasor-sign-flip"] = control_result(
            "phasor-sign-flip",
            "order_phase_error_rad",
            baseline=baseline["order_phase_error_rad"],
            mutated=mutated_phase,
            threshold=B3_4F_IDEAL.tolerance_for("order_phase_error_rad").threshold,
        )

    # 2. axis-transpose: transpose the measured image before reading orders.
    per_order_t = _order_measurements(object_field, image.T, params)
    mutated_power_t = Measurement(
        value=_order_power_relative_l2(per_order_t),
        uncertainty=_FLOOR,
        uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
    )
    results["axis-transpose"] = control_result(
        "axis-transpose",
        "order_power_relative_l2",
        baseline=baseline["order_power_relative_l2"],
        mutated=mutated_power_t,
        threshold=B3_4F_IDEAL.tolerance_for("order_power_relative_l2").threshold,
    )

    # 3. modulation-in-image-plane: apply the mask directly, no FT sandwich.
    wrong_image = object_field * mask
    per_order_wrong_plane = _order_measurements(object_field, wrong_image, params)
    mutated_power_plane = Measurement(
        value=_order_power_relative_l2(per_order_wrong_plane),
        uncertainty=_FLOOR,
        uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
    )
    results["modulation-in-image-plane"] = control_result(
        "modulation-in-image-plane",
        "order_power_relative_l2",
        baseline=baseline["order_power_relative_l2"],
        mutated=mutated_power_plane,
        threshold=B3_4F_IDEAL.tolerance_for("order_power_relative_l2").threshold,
    )

    # 4. frequency-grid-two-pi: scale the mask period by 2*pi when PREDICTING
    # the order location only; the actual field is untouched.
    per_order_2pi = _order_measurements(
        object_field, image, params, prediction_samples_per_period=samples_per_period * 2 * math.pi
    )
    mutated_location = Measurement(
        value=_order_location_error_frac(per_order_2pi, x0_1_px),
        uncertainty=_FLOOR,
        uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
    )
    results["frequency-grid-two-pi"] = control_result(
        "frequency-grid-two-pi",
        "order_location_error_frac",
        baseline=baseline["order_location_error_frac"],
        mutated=mutated_location,
        threshold=B3_4F_IDEAL.tolerance_for("order_location_error_frac").threshold,
    )

    return results


def _grid_snap_control(snapped_metric: float, offgrid_metric: float) -> Any:
    return control_result(
        "grid-snapped-vs-continuous-carrier",
        "order_power_relative_l2",
        baseline=Measurement(
            value=snapped_metric,
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        mutated=Measurement(
            value=offgrid_metric,
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        threshold=B3_4F_IDEAL.tolerance_for("order_power_relative_l2").threshold,
    )


# ---------------------------------------------------------------------------
# Canonical instances
# ---------------------------------------------------------------------------

_COMMON = {"grid_n": 512, "object_waist_pixels": 3.0, "device": "cpu"}


def _params(
    modulation_type: str, samples_per_period: float, phase_depth_rad: float
) -> dict[str, Any]:
    return {
        "modulation_type": modulation_type,
        "samples_per_period": samples_per_period,
        "phase_depth_rad": phase_depth_rad,
        **_COMMON,
    }


CANONICAL_PARAMETERS: dict[str, dict[str, Any]] = {
    "B3-4F-IDEAL-SIN-01": _params("sinusoidal_phase", 16.0, 1.5),
    "B3-4F-IDEAL-SIN-02": _params("sinusoidal_phase", 8.0, 1.5),
    "B3-4F-IDEAL-SIN-03": _params("sinusoidal_phase", 4.0, 1.5),
    "B3-4F-IDEAL-SIN-04": _params("sinusoidal_phase", 2.0, 1.5),
    "B3-4F-IDEAL-SIN-05": _params("sinusoidal_phase", 1.0, 1.5),
    "B3-4F-IDEAL-BIN-01": _params("binary_phase", 16.0, 2.0),
    "B3-4F-IDEAL-BIN-02": _params("binary_phase", 2.0, 2.0),
    "B3-4F-IDEAL-CARRIER-SNAPPED": _params("pure_carrier", 16.0, 0.0),
    "B3-4F-IDEAL-CARRIER-OFFGRID": _params("pure_carrier", 16.7, 0.0),
}


#: Whether each instance is expected to meet order_power_relative_l2's 1e-6
#: gate, and why. Declared ahead of the run so a pass or fail is read against a
#: stated expectation rather than reverse-engineered from the number.
_EXPECTED: dict[str, dict[str, str]] = {
    "B3-4F-IDEAL-SIN-01": {"order_power_relative_l2": "met -- deep inside, float64 floor"},
    "B3-4F-IDEAL-SIN-02": {
        "order_power_relative_l2": "NOT met -- INSIDE by FFT_GRID_NYQUIST, but the mask's own "
        "Fourier tail already aliases the fundamental at this depth"
    },
    "B3-4F-IDEAL-SIN-03": {
        "order_power_relative_l2": "NOT met -- same aliasing mechanism, larger at coarser sampling"
    },
    "B3-4F-IDEAL-SIN-04": {"order_power_relative_l2": "NOT met -- declared NEAR_BOUNDARY"},
    "B3-4F-IDEAL-SIN-05": {"order_power_relative_l2": "NOT met -- declared FAR_OUTSIDE"},
    "B3-4F-IDEAL-BIN-01": {
        "order_power_relative_l2": "NOT met -- INSIDE by FFT_GRID_NYQUIST; a discontinuous "
        "mask's Fourier series decays as O(1/n), so its aliasing tail is far larger than the "
        "sinusoidal case's at the same samples_per_period"
    },
    "B3-4F-IDEAL-BIN-02": {"order_power_relative_l2": "NOT met -- declared NEAR_BOUNDARY"},
    "B3-4F-IDEAL-CARRIER-SNAPPED": {
        "order_power_relative_l2": "met -- grid-commensurate, single order"
    },
    "B3-4F-IDEAL-CARRIER-OFFGRID": {
        "order_power_relative_l2": "NOT met, by design -- the grid-snapped-vs-continuous-carrier "
        "negative control's mutated arm"
    },
}


def canonical_instance(instance_id: str) -> Any:
    return B3_4F_IDEAL.instantiate(
        instance_id, CANONICAL_PARAMETERS[instance_id], expected=_EXPECTED[instance_id]
    )


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(sorted(CANONICAL_PARAMETERS))


# ---------------------------------------------------------------------------
# Run + verify
# ---------------------------------------------------------------------------


def run_instance(instance_id: str) -> InstanceRun:
    instance = canonical_instance(instance_id)
    params = dict(instance.parameters)

    start = time.perf_counter()
    object_field, mask, image = build(params)
    measurements = _measure(object_field, image, params)
    controls = _controls(object_field, mask, image, params, measurements)

    if instance_id == "B3-4F-IDEAL-CARRIER-OFFGRID":
        snapped_params = dict(CANONICAL_PARAMETERS["B3-4F-IDEAL-CARRIER-SNAPPED"])
        snapped_object, _snapped_mask, snapped_image = build(snapped_params)
        snapped_per_order = _order_measurements(snapped_object, snapped_image, snapped_params)
        snapped_metric = _order_power_relative_l2(snapped_per_order)
        controls["grid-snapped-vs-continuous-carrier"] = _grid_snap_control(
            snapped_metric, measurements["order_power_relative_l2"].value
        )
    wall_seconds = time.perf_counter() - start

    record = record_from_probe(
        instance,
        component="M_WAVE_CHROMATIX",
        node_id="ideal_4f_relay",
        refusal=None,
        observed_parameters={},
        device_precision=DevicePrecisionObservation(
            requested_device="cpu",
            actual_device="cpu",
            requested_dtype="complex128",
            actual_dtype=str(image.dtype),
        ),
        wall_seconds=wall_seconds,
    )
    result = verify(
        B3_4F_IDEAL,
        instance,
        record,
        measurements=measurements,
        invariants={"TOTAL_POWER_CONSERVED": measurements["total_power_relative_error"]},
        negative_controls=controls,
    )
    return InstanceRun(family=B3_4F_IDEAL, instance=instance, record=record, result=result)


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=None, help="run only this instance id")
    parser.add_argument(
        "--write", action="store_true", help="write benchmarks/systems/records/<id>.json"
    )
    args = parser.parse_args()

    ids = (args.instance,) if args.instance else declared_instance_ids()
    for instance_id in ids:
        run = run_instance(instance_id)
        print(
            f"{instance_id}  status={run.record.status.value}  "
            f"verification={run.result.status.value}"
        )
        for metric in run.result.physics_accuracy:
            met = "" if metric.met is None else f"  met={metric.met}"
            tol = "n/a" if metric.tolerance is None else f"{metric.tolerance:.3g}"
            print(f"  {metric.metric}: {metric.measured.value:.6e}  tol={tol}{met}")
        for inv in run.result.invariant_results:
            print(f"  [invariant] {inv.invariant_id}: {inv.measured.value:.6e}  met={inv.met}")
        for control in run.result.negative_control_results:
            print(f"  [control] {control.control_id}: {control.outcome.value}")
        print(
            f"  validity: declared={run.result.validity.declared.value} "
            f"observed={run.result.validity.observed.value}"
        )
        if args.write:
            path = write_instance_record(run, driver="systems/b3_4f_ideal", directory=RECORDS_DIR)
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
