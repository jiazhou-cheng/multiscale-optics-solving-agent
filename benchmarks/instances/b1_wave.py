"""The seven B1 wave families, executed through the Chromatix graph node.

CHE-107 (M1.2). Three of these carried closed forms verified against the pinned
solver in a retired task set and four had never been executed at all. All nine
canonical instances now run through ``GraphExecutor`` -- the real node, the real
adapter, the real ``asm_propagate`` -- with the input field supplied as a
declared graph source.

Why the executor rather than ``chromatix.functional`` directly
--------------------------------------------------------------
Because the thing under test includes the adapter's conventions, not only
Chromatix's arithmetic. Going through the node exercises the precision bridge,
the artifact boundary's pad-state and normalization declarations, the
input-pitch/output-pitch distinction, and the device observation read off the
output array. A direct library call would skip all of it and measure a narrower
claim than the family states.

The one exception is documented where it occurs: ``B1-WAVE-TILT``'s
``kykx_argument`` encoding is not reachable through the node's config, and the
hazard there is the parameter's UNIT rather than the physics. It is measured by
``B0-UNITS-02`` instead, which is where a convention trap belongs.

What the ASM validity family is for, and why its third instance must fail
------------------------------------------------------------------------
``B1-WAVE-ASM-VALIDITY`` is the only family here whose subject is behaviour NEAR
a boundary, and it is what forces the validity margin to be signed and
normalized rather than boolean. Three distances straddle ``z = N pitch^2 /
lambda``. All three run. Crossing the limit raises nothing: it folds energy back
in from the other side and returns a field that looks like a Gaussian and is the
wrong size. A run that reported the gate met on the far side would mean the
metric cannot see aliasing, so the outside instance failing its gate is the
measurement, not a defect.

Run it::

    ./run.sh python benchmarks/instances/b1_wave.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from core.paths import repository_root
from core.specs import GraphSpec
from runtime.instance_runner import (
    execute,
    field_source,
    observed_placement,
    probe_refusal,
)
from verification.evidence import (
    InstanceRun,
    control_result,
    write_instance_record,
)
from verification.families.b1_wave import (
    B1_WAVE_AIRY,
    B1_WAVE_ASM_VALIDITY,
    B1_WAVE_FWDBWD,
    B1_WAVE_GAUSS,
    B1_WAVE_PLANEPHASE,
    B1_WAVE_TALBOT,
    B1_WAVE_TILT,
)
from verification.metrics import relative_l2_field, relative_l2_intensity
from verification.result import Measurement, UncertaintyBasis
from verification.verifier import verify

__all__ = [
    "declared_instance_ids",
    "device_observation",
    "run_all",
    "run_instance",
    "write_device_observation_record",
]

ROOT = repository_root()
UM_PER_M = 1e6


def _instance(family: Any, instance_id: str) -> Any:
    for candidate in family.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"{family.family_id} declares no instance {instance_id!r}")


# ---------------------------------------------------------------------------
# One propagation, through the node
# ---------------------------------------------------------------------------


def _wave_graph(*, z_m: float, pad_width: int, output_dir: Path, method: str) -> GraphSpec:
    """A single-node wave graph. The input arrives as a declared graph source.

    ``pad_width`` is passed explicitly and never left to the adapter: it refuses
    to guess one, and for these families the physics requires a specific choice.
    Every family below propagates a field that is either periodic on the grid or
    comfortably inside it, so zero padding is the honest setting -- padding would
    change the mode grid and, for the Talbot and plane-phase families, change
    which frequencies exist at all.
    """
    return GraphSpec.model_validate(
        {
            "schema_version": "0.1.0",
            "nodes": [
                {
                    "id": "wave",
                    "model": "M_WAVE_CHROMATIX",
                    "config": {
                        "propagation": "angular_spectrum",
                        "propagation_method": method,
                        "z_m": z_m,
                        "refractive_index": 1.0,
                        "pad_width": int(pad_width),
                        "output_dir": str(output_dir),
                    },
                }
            ],
        }
    )


def _propagate(
    field: np.ndarray,
    *,
    instance: Any,
    wavelength_m: float,
    pitch_m: float,
    z_m: float,
    pad_width: int = 0,
    method: str = "asm_propagate",
) -> dict[str, Any]:
    """Run the node once and return the output field with what was observed.

    Both the input and the output pitch are reported. Angular-spectrum
    propagation happens to preserve the pitch, which makes reading the OUTPUT
    pitch cheap rather than vacuous: it is the same assertion for a method that
    does not, and the failure it guards is silent -- every distance a downstream
    measurement reports would be rescaled by the ratio while the intensity map
    looked entirely ordinary.
    """
    directory = Path(tempfile.mkdtemp(prefix="b1-wave-"))
    source = field_source(
        np.asarray(field, dtype=np.complex64),
        wavelength_m=wavelength_m,
        sample_pitch_m=pitch_m,
        directory=directory,
        artifact_id=f"{instance.instance_id}-in",
    )
    graph = _wave_graph(
        z_m=z_m, pad_width=pad_width, output_dir=directory, method=method
    )
    record = execute(graph, instance, inputs={"wave.input_field": source})
    if not record.artifacts or "wave:output_field" not in record.artifacts:
        node = record.nodes[0] if record.nodes else None
        raise RuntimeError(
            f"{instance.instance_id}: the wave node produced no field "
            f"({getattr(node, 'error_type', None)}: {getattr(node, 'error_message', None)})"
        )
    output = record.artifacts["wave:output_field"]
    array = np.load(output.uri)
    return {
        "record": record,
        "artifact": output,
        "field": array,
        "input_pitch_m": pitch_m,
        "output_pitch_m": tuple(float(v) for v in output.metadata["sample_pitch"]),
        "placement": observed_placement(array),
        "pad_width": int(output.metadata.get("pad_width") or 0),
    }


# ---------------------------------------------------------------------------
# Field builders. Every one of them is analytic; none calls a solver.
# ---------------------------------------------------------------------------


def _grid(n: int, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(n, dtype=np.float64) - n // 2) * pitch_m
    return np.meshgrid(axis, axis, indexing="ij")


def _gaussian(n: int, pitch_m: float, waist_m: float) -> np.ndarray:
    yy, xx = _grid(n, pitch_m)
    return np.exp(-((yy**2 + xx**2) / waist_m**2))


def _second_moment_radius_m(intensity: np.ndarray, pitch_m: float) -> float:
    """The 1/e^2 radius of a Gaussian, from its intensity's second moment.

    For ``I = exp(-2 r^2 / w^2)`` the second moment of ``r^2`` is ``w^2 / 2``, so
    ``w = sqrt(2 <r^2>)``. Estimator rather than a fit, because a fit would need
    a model and the point is to measure what came back.
    """
    yy, xx = _grid(intensity.shape[0], pitch_m)
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError("an intensity that sums to zero has no second moment")
    mean_square = float(((yy**2 + xx**2) * intensity).sum() / total)
    return math.sqrt(2.0 * mean_square)


# ---------------------------------------------------------------------------
# B1-WAVE-GAUSS
# ---------------------------------------------------------------------------


def _run_gauss() -> InstanceRun:
    instance = _instance(B1_WAVE_GAUSS, "B1-WAVE-GAUSS-01")
    p = instance.parameters
    waist_m = float(p["waist_um"]) / UM_PER_M
    distance_m = float(p["distance_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])

    run = _propagate(
        _gaussian(n, pitch_m, waist_m),
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=distance_m,
    )
    measured_m = _second_moment_radius_m(np.abs(run["field"]) ** 2, pitch_m)

    rayleigh_m = math.pi * waist_m**2 / wavelength_m
    exact_m = waist_m * math.sqrt(1.0 + (distance_m / rayleigh_m) ** 2)
    error = abs(measured_m - exact_m) / exact_m
    unpropagated_error = abs(waist_m - exact_m) / exact_m

    # The estimator's own bias, measured by running it over the ANALYTIC field on
    # the same grid. Reported as the error bar rather than assumed to be zero:
    # a second-moment radius on a finite grid truncates the Gaussian's tails.
    analytic_intensity = np.exp(
        -2.0 * (sum(g**2 for g in _grid(n, pitch_m))) / exact_m**2
    )
    estimator_m = _second_moment_radius_m(analytic_intensity, pitch_m)
    estimator_bias = abs(estimator_m - exact_m) / exact_m

    measurements = {
        "gaussian_radius_relative_error": Measurement(
            value=error,
            uncertainty=estimator_bias,
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                f"second-moment 1/e^2 radius {measured_m * UM_PER_M:.6f} um against "
                f"w0 sqrt(1 + (z/zR)^2) = {exact_m * UM_PER_M:.6f} um. The error bar is "
                "the same estimator's bias on the ANALYTIC field over the same grid, "
                f"{estimator_bias:.3e}, so the truncation of the Gaussian's tails is "
                "visible rather than assumed away."
            ),
        )
    }
    controls = {
        "unpropagated-waist": control_result(
            "unpropagated-waist",
            "gaussian_radius_relative_error",
            baseline=measurements["gaussian_radius_relative_error"],
            mutated=Measurement(
                value=unpropagated_error,
                uncertainty=estimator_bias,
                uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
                note="w0 reported in place of w(z): what a zero-distance run returns",
            ),
            threshold=2e-2,
            note="the unpropagated waist is 17% low and is the wrong answer this rejects.",
        )
    }
    return _verified(
        B1_WAVE_GAUSS,
        instance,
        run,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "ESTIMATOR_BIAS_ON_THE_ANALYTIC_FIELD",
                "detail": (
                    f"the same second-moment estimator reads {estimator_m * UM_PER_M:.6f} "
                    f"um on the exact analytic field, a bias of {estimator_bias:.3e}"
                ),
                "location": "benchmarks/instances/b1_wave.py::_second_moment_radius_m",
            }
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-AIRY
# ---------------------------------------------------------------------------

#: The aperture radius is fixed and the focal length follows from the declared
#: NA, so NA is the one physical knob and ``0.61 lambda / NA`` depends on nothing
#: else.
AIRY_APERTURE_RADIUS_M = 30e-6


def _run_airy() -> InstanceRun:
    instance = _instance(B1_WAVE_AIRY, "B1-WAVE-AIRY-01")
    p = instance.parameters
    na = float(p["numerical_aperture"])
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    pitch_m = float(p["focal_plane_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])
    focal_m = AIRY_APERTURE_RADIUS_M / na

    yy, xx = _grid(n, pitch_m)
    radius = np.hypot(yy, xx)
    aperture = (radius <= AIRY_APERTURE_RADIUS_M).astype(np.float64)
    # A converging lens under exp(-i omega t) / exp(+i k z): exp(-i k r^2 / 2f).
    lens_phase = np.exp(-1j * (2.0 * math.pi / wavelength_m) * radius**2 / (2.0 * focal_m))
    pupil = aperture * lens_phase

    sampling_limit_m = n * pitch_m**2 / wavelength_m
    run = _propagate(
        pupil,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=focal_m,
    )
    intensity = np.abs(run["field"]) ** 2

    from verification.psf_oracles import first_null_comparison

    comparison = first_null_comparison(
        intensity,
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=wavelength_m,
        numerical_aperture=na,
    )
    predicted_m = 0.61 * wavelength_m / na
    measured_m = comparison["measured_first_null_m"]
    if measured_m is None:
        raise RuntimeError("no first null was found in the focal-plane intensity")
    error = abs(float(measured_m) - predicted_m) / predicted_m
    bias = abs(float(comparison["analytic_estimator_bias"] or 0.0))

    measurements = {
        "airy_first_null_relative_error": Measurement(
            value=error,
            uncertainty=bias,
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                f"first null {float(measured_m) * UM_PER_M:.6f} um against "
                f"0.61 lambda/NA = {predicted_m * UM_PER_M:.6f} um. The error bar is "
                "the estimator's OWN bias, measured by running it over the analytic "
                "Airy pattern on the same grid: a first-null estimator is badly biased "
                f"at coarse sampling and this grid has {predicted_m / pitch_m:.1f} "
                "samples per Airy radius. The bias-cancelled ratio is "
                f"{comparison['ratio_measured_over_analytic']:.6f}."
            ),
        )
    }
    controls = {
        "diameter-for-radius": control_result(
            "diameter-for-radius",
            "airy_first_null_relative_error",
            baseline=measurements["airy_first_null_relative_error"],
            mutated=Measurement(
                value=abs(2.0 * float(measured_m) - predicted_m) / predicted_m,
                uncertainty=bias,
                uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
                note="1.22 lambda/NA reported where 0.61 lambda/NA is meant",
            ),
            threshold=5e-2,
            note="the factor-of-two confusion between the null radius and its diameter.",
        )
    }
    return _verified(
        B1_WAVE_AIRY,
        instance,
        run,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "GEOMETRY_DERIVED_FROM_THE_DECLARED_NA",
                "detail": (
                    f"aperture radius {AIRY_APERTURE_RADIUS_M * UM_PER_M:.1f} um, focal "
                    f"length a/NA = {focal_m * UM_PER_M:.1f} um, "
                    f"{predicted_m / pitch_m:.2f} samples per Airy radius"
                ),
                "location": "benchmarks/instances/b1_wave.py::_run_airy",
            },
            {
                "code": "FOCUS_IS_INSIDE_THE_ASM_SAMPLING_LIMIT",
                "detail": (
                    f"z = {focal_m * UM_PER_M:.1f} um against N pitch^2 / lambda = "
                    f"{sampling_limit_m * UM_PER_M:.1f} um; margin "
                    f"{(sampling_limit_m - focal_m) / sampling_limit_m:.4f}. Checked "
                    "rather than assumed: past the limit the kernel aliases and the "
                    "focal spot is the wrong size with nothing raised."
                ),
                "location": "verification/families/predicates.py::asm_transfer_function_sampling",
            },
            {
                "code": "ESTIMATOR_BIAS_CANCELLED_RATIO",
                "detail": (
                    f"ratio_measured_over_analytic = "
                    f"{comparison['ratio_measured_over_analytic']}, analytic estimator "
                    f"bias {comparison['analytic_estimator_bias']}"
                ),
                "location": "verification/psf_oracles.py::first_null_comparison",
            },
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-TILT
# ---------------------------------------------------------------------------


def _run_tilt() -> InstanceRun:
    instance = _instance(B1_WAVE_TILT, "B1-WAVE-TILT-01")
    p = instance.parameters
    tilt_rad = float(p["tilt_rad"])
    distance_m = float(p["distance_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])

    waist_m = 12e-6
    exact_m = distance_m * math.tan(tilt_rad)

    def _centroid_for(cycles_per_m: float) -> float:
        yy, _xx = _grid(n, pitch_m)
        ramp = np.exp(1j * 2.0 * math.pi * cycles_per_m * yy)
        run = _propagate(
            _gaussian(n, pitch_m, waist_m) * ramp,
            instance=instance,
            wavelength_m=wavelength_m,
            pitch_m=pitch_m,
            z_m=distance_m,
        )
        intensity = np.abs(run["field"]) ** 2
        axis = (np.arange(intensity.shape[0], dtype=np.float64) - intensity.shape[0] // 2) * (
            pitch_m
        )
        profile = intensity.sum(axis=1)
        return float((profile * axis).sum() / profile.sum()), run

    cycles = math.sin(tilt_rad) / wavelength_m
    measured_m, run = _centroid_for(cycles)
    # SIGNED. The metric is the signed relative error, so a walk-off in the wrong
    # direction is a 2x error rather than a pass.
    signed_error = (measured_m - exact_m) / exact_m

    flipped_m, _ = _centroid_for(-cycles)
    two_pi_m, _ = _centroid_for(cycles * 2.0 * math.pi)

    measurements = {
        "tilt_centroid_signed_relative_error": Measurement(
            value=abs(signed_error),
            uncertainty=pitch_m / abs(exact_m),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"centroid {measured_m * UM_PER_M:+.6f} um against z tan(theta) = "
                f"{exact_m * UM_PER_M:+.6f} um, SIGN included. The error bar is one "
                "sample pitch of centroid resolution. The tolerance deliberately does "
                "not separate z sin(theta) from z tan(theta) -- 0.4% apart at 5 degrees "
                "-- and the family says so."
            ),
        )
    }
    controls = {
        "kykx-sign": control_result(
            "kykx-sign",
            "tilt_centroid_signed_relative_error",
            baseline=measurements["tilt_centroid_signed_relative_error"],
            mutated=Measurement(
                value=abs((flipped_m - exact_m) / exact_m),
                uncertainty=pitch_m / abs(exact_m),
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note=(
                    "the ramp's spatial frequency negated. Through the shipping "
                    "propagation, one term flipped and nothing else changed"
                ),
            ),
            threshold=2e-2,
            note="a sign error is a 2x relative error on a signed metric.",
        ),
        "kykx-two-pi": control_result(
            "kykx-two-pi",
            "tilt_centroid_signed_relative_error",
            baseline=measurements["tilt_centroid_signed_relative_error"],
            mutated=Measurement(
                value=abs((two_pi_m - exact_m) / exact_m),
                uncertainty=pitch_m / abs(exact_m),
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note=(
                    "the ramp's frequency multiplied by 2*pi -- the cycles/radians "
                    "confusion, applied where this family can see it"
                ),
            ),
            threshold=2e-2,
            note=(
                "2*pi too large leaves the paraxial regime entirely: the effective "
                "tilt is 33 degrees, so the ratio is not 2*pi either."
            ),
        ),
    }
    return _verified(
        B1_WAVE_TILT,
        instance,
        run,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "THE_KYKX_ENCODING_IS_MEASURED_ELSEWHERE",
                "detail": (
                    "tilt_encoding='kykx_argument' is not reachable through the graph "
                    "node's config, and the hazard there is the parameter's UNIT rather "
                    "than the physics: kykx means cycles per length on asm_propagate "
                    "and radians per length on plane_wave, and the displacement runs "
                    "opposite in sign to the parameter on the propagator. B0-UNITS-02 "
                    "measures both call sites. This family measures the walk-off."
                ),
                "location": "benchmarks/instances/b0_contract.py::_measure_kykx_hazard",
            }
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-PLANEPHASE
# ---------------------------------------------------------------------------


def _run_planephase() -> InstanceRun:
    instance = _instance(B1_WAVE_PLANEPHASE, "B1-WAVE-PLANEPHASE-01")
    p = instance.parameters
    frequency_per_m = float(p["transverse_frequency_per_um"]) * UM_PER_M
    distance_m = float(p["distance_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    index = float(p["medium_index"])
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])

    def _mean_phase(cycles_per_m: float) -> tuple[float, dict[str, Any]]:
        yy, _xx = _grid(n, pitch_m)
        field = np.exp(1j * 2.0 * math.pi * cycles_per_m * yy)
        run = _propagate(
            field,
            instance=instance,
            wavelength_m=wavelength_m,
            pitch_m=pitch_m,
            z_m=distance_m,
        )
        # The plane wave is periodic on the grid, so the propagated field is the
        # input times one global phase. Extracted as the argument of the inner
        # product, which is the exact minimiser rather than a fit.
        overlap = np.vdot(field.astype(np.complex128), run["field"].astype(np.complex128))
        return float(np.angle(overlap)), run

    on_axis_phase, _ = _mean_phase(0.0)
    tilted_phase, run = _mean_phase(frequency_per_m)

    # The RELATIVE advance, (k_z - k) z, which is unambiguous where the absolute
    # advance is thousands of radians and wraps.
    measured = float(np.angle(np.exp(1j * (tilted_phase - on_axis_phase))))
    k = 2.0 * math.pi * index / wavelength_m
    k_transverse = 2.0 * math.pi * frequency_per_m
    predicted = (math.sqrt(k * k - k_transverse * k_transverse) - k) * distance_m
    wrapped_predicted = float(np.angle(np.exp(1j * predicted)))
    residual = abs(float(np.angle(np.exp(1j * (measured - predicted)))))

    # The two controls, evaluated against the same measured advance.
    sign_flipped = abs(float(np.angle(np.exp(1j * (-measured - predicted)))))
    two_pi_frequency = 2.0 * math.pi * frequency_per_m
    two_pi_transverse = 2.0 * math.pi * two_pi_frequency
    if two_pi_transverse >= k:
        # 2*pi too large puts the mode past the light cone entirely: the "wrong
        # answer" is not a shifted phase, it is an evanescent mode. Reported as
        # the largest possible residual rather than as a number, because there
        # is no propagating phase to compare against.
        two_pi_residual = math.pi
        two_pi_note = (
            "a 2*pi frequency-grid error puts k_t past the light cone: the mode is "
            f"evanescent (k_t = {two_pi_transverse:.4g} against k = {k:.4g}), so there "
            "is no propagating phase advance at all. Reported as pi, the largest "
            "possible phase residual."
        )
    else:
        two_pi_predicted = (
            math.sqrt(k * k - two_pi_transverse**2) - k
        ) * distance_m
        two_pi_residual = abs(
            float(np.angle(np.exp(1j * (two_pi_predicted - predicted))))
        )
        two_pi_note = "a 2*pi frequency-grid error, as a phase residual"

    measurements = {
        "plane_wave_phase_residual_rad": Measurement(
            value=residual,
            uncertainty=float(np.finfo(np.float32).eps) * abs(k * distance_m),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"relative advance {measured:+.9f} rad against "
                f"(k_z - k) z = {predicted:+.9f} (wrapped {wrapped_predicted:+.9f}). "
                "Measured RELATIVE to an on-axis plane wave propagated the same "
                "distance, because the absolute advance k_z z is "
                f"{k * distance_m:.1f} rad and wraps. The error bar is one float32 "
                "epsilon per radian of that absolute advance, which is the "
                "representation floor Chromatix's complex64-only field imposes."
            ),
        )
    }
    controls = {
        "phasor-sign-flip": control_result(
            "phasor-sign-flip",
            "plane_wave_phase_residual_rad",
            baseline=measurements["plane_wave_phase_residual_rad"],
            mutated=Measurement(
                value=sign_flipped,
                uncertainty=float(np.finfo(np.float32).eps) * abs(k * distance_m),
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="exp(-i k_z z) in place of exp(+i k_z z): the advance negated",
            ),
            threshold=1e-2,
            note="on axis this would be invisible; off axis it is 2|(k_z - k) z|.",
        ),
        "frequency-grid-two-pi": control_result(
            "frequency-grid-two-pi",
            "plane_wave_phase_residual_rad",
            baseline=measurements["plane_wave_phase_residual_rad"],
            mutated=Measurement(
                value=two_pi_residual,
                uncertainty=float(np.finfo(np.float32).eps) * abs(k * distance_m),
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=two_pi_note,
            ),
            threshold=1e-2,
            note="the cycles/radians confusion in the frequency grid.",
        ),
    }
    return _verified(
        B1_WAVE_PLANEPHASE,
        instance,
        run,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "OFF_AXIS_IS_WHERE_THIS_IS_MEASURABLE",
                "detail": (
                    f"k_t = {k_transverse:.6g} /m against k = {k:.6g} /m, so "
                    f"k_z/k = {math.sqrt(1 - (k_transverse / k) ** 2):.9f}. On axis "
                    "k_z = k exactly and a frequency-grid scale error is invisible; "
                    "that is why the instance declares a nonzero transverse frequency."
                ),
                "location": "verification/families/b1_wave.py::_plane_wave_phase_advance",
            }
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-FWDBWD
# ---------------------------------------------------------------------------


def _run_fwdbwd() -> InstanceRun:
    instance = _instance(B1_WAVE_FWDBWD, "B1-WAVE-FWDBWD-01")
    p = instance.parameters
    distance_m = float(p["distance_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    fill = float(p["aperture_fill_fraction"])
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])

    yy, xx = _grid(n, pitch_m)
    radius = np.hypot(yy, xx)
    aperture_radius_m = fill * (n * pitch_m) / 2.0
    # A soft-edged disc: a hard edge has unbounded bandwidth and would alias on
    # any grid, which would make the round-trip residual a statement about the
    # aperture rather than about the propagator.
    field = np.exp(-((radius / aperture_radius_m) ** 8)).astype(np.complex64)

    forward = _propagate(
        field,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=distance_m,
    )
    backward = _propagate(
        forward["field"],
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=-distance_m,
    )
    residual = relative_l2_field(backward["field"], field)

    # The asymmetric control: a half-pixel shift applied between the two legs, so
    # the round trip is no longer the inverse of itself. Deliberately small --
    # a control that changes the field by a factor would prove nothing about a
    # 1e-5 gate.
    shifted = np.roll(forward["field"], 1, axis=0)
    asymmetric = _propagate(
        shifted,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=-distance_m,
    )
    asymmetric_residual = relative_l2_field(asymmetric["field"], field)

    # The wrapped-aperture control: the same round trip with an aperture that
    # fills the grid, so the field wraps and the two legs no longer see the same
    # spectrum.
    wide = np.exp(-((radius / (0.98 * (n * pitch_m) / 2.0)) ** 8)).astype(np.complex64)
    wide_forward = _propagate(
        wide,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=distance_m,
    )
    wide_back = _propagate(
        wide_forward["field"],
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=-distance_m,
    )
    wrapped_residual = relative_l2_field(wide_back["field"], wide)

    accumulated_rad = 2.0 * math.pi * distance_m / wavelength_m
    single_pass_floor = float(np.finfo(np.float32).eps) * accumulated_rad

    measurements = {
        "round_trip_relative_l2": Measurement(
            value=residual,
            uncertainty=single_pass_floor,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"forward {distance_m * UM_PER_M:.1f} um then backward, relative L2 on "
                "the FIELD. The error bar is the single-pass complex64 floor, one "
                f"float32 epsilon per radian of {accumulated_rad:.1f} rad accumulated "
                f"phase = {single_pass_floor:.3e}; the round trip lands below it "
                "because the two legs' phase errors are correlated."
            ),
        )
    }
    controls = {
        "asymmetric-fftshift": control_result(
            "asymmetric-fftshift",
            "round_trip_relative_l2",
            baseline=measurements["round_trip_relative_l2"],
            mutated=Measurement(
                value=asymmetric_residual,
                uncertainty=single_pass_floor,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="one sample of lateral shift inserted between the two legs",
            ),
            threshold=1e-5,
            note=(
                "a round trip that cannot be made to fail proves nothing. This is the "
                "smallest asymmetry the grid can express."
            ),
        ),
        "wrapped-aperture": control_result(
            "wrapped-aperture",
            "round_trip_relative_l2",
            baseline=measurements["round_trip_relative_l2"],
            mutated=Measurement(
                value=wrapped_residual,
                uncertainty=single_pass_floor,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="an aperture filling 98% of the grid, so the field wraps",
            ),
            threshold=1e-5,
            note="the round trip stops being unitary once energy leaves the window.",
        ),
    }
    return _verified(
        B1_WAVE_FWDBWD,
        instance,
        forward,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "WHAT_A_ROUND_TRIP_CANNOT_SEE",
                "detail": (
                    "a convention error SHARED by the two legs cancels exactly: a "
                    "phasor sign flipped in both directions returns the input. That is "
                    "why this family is not sufficient on its own and why "
                    "B1-WAVE-PLANEPHASE gates the one-way phase separately."
                ),
                "location": "verification/families/b1_wave.py::B1_WAVE_FWDBWD",
            }
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-TALBOT
# ---------------------------------------------------------------------------


def _talbot_non_paraxial_budget(
    *,
    period_um: float,
    wavelength_um: float,
    duty: float,
    periods: int,
    samples_per_period: int,
    order: int,
) -> dict[str, Any]:
    """Which term sets the Talbot residual, computed rather than asserted.

    The revival at ``z_T = 2 d^2 / lambda`` is a PARAXIAL result; Chromatix's
    angular spectrum is not paraxial. So the admissible residual is dominated by
    the phase the paraxial expansion drops, ``(1/8) k z (m lambda/d)^4``, which at
    ``z_T`` is ``(pi/2) m^4 (lambda/d)^2`` -- independent of ``z`` and set by the
    highest order the grid admits, ``m_max = samples_per_period / 2``.

    The two other terms the tolerance could plausibly rest on are separated here
    by construction rather than by argument. Both kernels below are float64 and
    share the same 1-D grating and the same grid:

    * the PARAXIAL kernel isolates finite-window truncation, because a paraxial
      propagator revives exactly. It returns ~1e-24, so truncation is not a
      budget item -- it is designed out by holding an exact integer number of
      periods per window and samples per period.
    * the EXACT kernel isolates the non-paraxial term at float64, so the gap
      between it and the measured complex64 run is the complex64 floor.

    DIAGNOSTIC ONLY, and deliberately so: this is our own numerical code, and the
    repository's standing rule is that custom propagators do not decide gates.
    The gate is the analytic revival; this decomposes the residual it leaves.
    One dimension is enough because the grating varies along one axis only.
    """
    n = periods * samples_per_period
    pitch_um = period_um / samples_per_period
    z_um = order * 2.0 * period_um**2 / wavelength_um
    column = ((np.arange(n) % samples_per_period) < round(duty * samples_per_period)).astype(
        np.float64
    )

    freq = np.fft.fftfreq(n, d=pitch_um)  # cycles per um
    under_root = 1.0 - (wavelength_um * freq) ** 2
    propagating = under_root > 0.0
    k = 2.0 * np.pi / wavelength_um
    spectrum = np.fft.fft(column)

    def _residual(kz: np.ndarray) -> float:
        out = np.fft.ifft(spectrum * np.where(propagating, np.exp(1j * kz * z_um), 0.0))
        return relative_l2_intensity(out, column)

    exact = _residual(k * np.sqrt(np.where(propagating, under_root, 0.0)))
    paraxial = _residual(k * (1.0 - 0.5 * (wavelength_um * freq) ** 2))

    m_max = samples_per_period // 2
    amplitudes = np.abs(spectrum) / n
    orders = []
    for m in range(m_max + 1):
        bin_index = m * periods
        if bin_index >= n:
            break
        orders.append(
            {
                "m": m,
                "amplitude": float(amplitudes[bin_index]),
                "dephasing_rad": (np.pi / 2.0) * m**4 * (wavelength_um / period_um) ** 2,
            }
        )
    carried = [o for o in orders if o["amplitude"] > 1e-12 and o["m"] > 0]
    dominant = max(carried, key=lambda o: o["m"]) if carried else None
    return {
        "m_max_admitted_by_grid": m_max,
        "dephasing_at_m_max_rad": (np.pi / 2.0) * m_max**4 * (wavelength_um / period_um) ** 2,
        "dominant_carried_order": dominant,
        "orders": orders,
        "exact_kernel_intensity_l2_float64": exact,
        "paraxial_kernel_intensity_l2_float64": paraxial,
    }


def _run_talbot() -> InstanceRun:
    instance = _instance(B1_WAVE_TALBOT, "B1-WAVE-TALBOT-01")
    p = instance.parameters
    period_m = float(p["period_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    duty = float(p["duty_cycle"])
    order = int(p["talbot_order"])
    periods = int(p["periods_across_grid"])
    samples_per_period = int(p["samples_per_period"])

    n = periods * samples_per_period
    pitch_m = period_m / samples_per_period
    talbot_m = order * 2.0 * period_m**2 / wavelength_m

    # A binary grating, exactly periodic on the grid: `period_um` is an exact
    # number of samples and the grid an exact number of periods, so nothing is
    # discontinuous at the wrap and the revival is a statement about the
    # propagator rather than about the edge.
    column = (np.arange(n) % samples_per_period) < round(duty * samples_per_period)
    grating = np.tile(column.astype(np.float64)[:, None], (1, n)).astype(np.complex64)

    revival = _propagate(
        grating,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=talbot_m,
    )
    residual = relative_l2_intensity(revival["field"], grating)

    half = _propagate(
        grating,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=talbot_m / 2.0,
    )
    half_residual = relative_l2_intensity(half["field"], grating)

    scaled = _propagate(
        grating,
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=talbot_m * (1.0 + 1.0 / 32.0),
    )
    scaled_residual = relative_l2_intensity(scaled["field"], grating)

    sampling_limit_m = n * pitch_m**2 / wavelength_m
    budget = _talbot_non_paraxial_budget(
        period_um=float(p["period_um"]),
        wavelength_um=float(p["wavelength_um"]),
        duty=duty,
        periods=periods,
        samples_per_period=samples_per_period,
        order=order,
    )
    measurements = {
        "talbot_revival_relative_l2": Measurement(
            value=residual,
            uncertainty=abs(scaled_residual - residual),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"intensity relative L2 against the input grating after "
                f"z_T = 2 d^2 / lambda = {talbot_m * UM_PER_M:.4f} um. The error bar is "
                "the sensitivity to a 3% distance error, which is what bounds how much "
                "of this residual could be a distance rather than a propagator."
            ),
        )
    }
    controls = {
        "half-talbot": control_result(
            "half-talbot",
            "talbot_revival_relative_l2",
            baseline=measurements["talbot_revival_relative_l2"],
            mutated=Measurement(
                value=half_residual,
                uncertainty=abs(scaled_residual - residual),
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note=(
                    "z_T/2, where the pattern revives SHIFTED by half a period. A field "
                    "that looks exactly as much like a grating as the revival does, "
                    "displaced -- which is what makes it the right control for a "
                    "revival claim rather than for a 'looks periodic' claim"
                ),
            ),
            threshold=5e-3,
            note="a shift-blind metric would pass this, and this metric is not.",
        ),
        "frequency-grid-scale": control_result(
            "frequency-grid-scale",
            "talbot_revival_relative_l2",
            baseline=measurements["talbot_revival_relative_l2"],
            mutated=Measurement(
                value=scaled_residual,
                uncertainty=abs(scaled_residual - residual),
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note=(
                    "the revival distance off by 1/32. A revival is a periodicity "
                    "check, so it is exactly the measurement a frequency-grid scale "
                    "error moves"
                ),
            ),
            threshold=5e-3,
            note="z_T goes as d^2, so a small scale error is a large distance error.",
        ),
    }
    return _verified(
        B1_WAVE_TALBOT,
        instance,
        revival,
        measurements=measurements,
        controls=controls,
        diagnostics=[
            {
                "code": "EXACTLY_PERIODIC_ON_THE_GRID",
                "detail": (
                    f"{periods} periods of {samples_per_period} samples each, "
                    f"pitch {pitch_m * UM_PER_M:.6f} um, grid {n}. Nothing is "
                    "discontinuous at the wrap, so the residual is the propagator's."
                ),
                "location": "benchmarks/instances/b1_wave.py::_run_talbot",
            },
            {
                "code": "REVIVAL_IS_INSIDE_THE_SAMPLING_LIMIT",
                "detail": (
                    f"z_T = {talbot_m * UM_PER_M:.4f} um against N pitch^2 / lambda = "
                    f"{sampling_limit_m * UM_PER_M:.4f} um"
                ),
                "location": "verification/families/predicates.py::asm_transfer_function_sampling",
            },
            {
                "code": "NON_PARAXIAL_DEPHASING_BUDGET",
                "detail": (
                    "which term sets this residual, decomposed rather than argued. The "
                    "grid admits orders up to m_max = samples_per_period / 2 = "
                    f"{budget['m_max_admitted_by_grid']}, whose dephasing from the "
                    "paraxial revival is (pi/2) m^4 (lambda/d)^2 = "
                    f"{budget['dephasing_at_m_max_rad']:.4f} rad; at duty {duty:g} the "
                    "highest order actually CARRYING amplitude is m = "
                    f"{budget['dominant_carried_order']['m']} at "
                    f"{budget['dominant_carried_order']['dephasing_rad']:.4e} rad. The "
                    "same grating on the same grid through a float64 PARAXIAL kernel "
                    f"returns {budget['paraxial_kernel_intensity_l2_float64']:.3e}, so "
                    "finite-window truncation is designed out by the exact-integer "
                    "periodicity rather than budgeted; through a float64 EXACT-ASM "
                    f"kernel it returns {budget['exact_kernel_intensity_l2_float64']:.3e} "
                    f"against the {residual:.3e} measured through the shipping "
                    "complex64 path, which leaves the complex64 floor as the difference "
                    "and a secondary term. Both kernels are DIAGNOSTIC ONLY: our own "
                    "numerical code does not decide this gate."
                ),
                "location": "benchmarks/instances/b1_wave.py::_talbot_non_paraxial_budget",
            },
            {
                "code": "WHY_A_REVIVAL_IS_A_STRONG_CHECK",
                "detail": (
                    "it depends on the propagator reproducing the RELATIVE phases of "
                    "many diffraction orders at once. A single-order comparison cannot "
                    "see an error that is common to all orders; a revival can."
                ),
                "location": "verification/families/b1_wave.py::B1_WAVE_TALBOT",
            },
        ],
    )


# ---------------------------------------------------------------------------
# B1-WAVE-ASM-VALIDITY
# ---------------------------------------------------------------------------


#: The sweep is run once and cached, because the two cross-instance controls --
#: "past the boundary must not pass" and "silent wrap" -- compare the far side
#: against the near side. A control whose baseline is a DIFFERENT instance of the
#: same family is the right shape here: the mutation is the propagation distance,
#: and the thing being demonstrated is that changing only that turns a correct
#: answer into a wrong one with nothing raised.
_ASM_SWEEP: dict[str, dict[str, Any]] = {}


def _asm_measure(instance: Any) -> dict[str, Any]:
    p = instance.parameters
    waist_m = float(p["waist_um"]) / UM_PER_M
    distance_m = float(p["distance_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    n = int(p["grid_n"])

    sampling_limit_m = n * pitch_m**2 / wavelength_m
    margin = (sampling_limit_m - distance_m) / sampling_limit_m

    run = _propagate(
        _gaussian(n, pitch_m, waist_m),
        instance=instance,
        wavelength_m=wavelength_m,
        pitch_m=pitch_m,
        z_m=distance_m,
    )
    intensity = np.abs(run["field"]) ** 2
    measured_m = _second_moment_radius_m(intensity, pitch_m)

    rayleigh_m = math.pi * waist_m**2 / wavelength_m
    exact_m = waist_m * math.sqrt(1.0 + (distance_m / rayleigh_m) ** 2)
    error = abs(measured_m - exact_m) / exact_m

    # The declared metric: power within ONE PITCH of the window edge. That band
    # is where an FFT-based propagator's wrap lands, and a beam that has power
    # there under a periodic boundary IS a beam that has wrapped. The analytic
    # field's own edge fraction is measured beside it, so "the beam is genuinely
    # that big" and "the beam folded back" are separable rather than conflated.
    yy, xx = _grid(n, pitch_m)
    half_window_m = (n * pitch_m) / 2.0
    edge = (np.abs(yy) > half_window_m - pitch_m) | (np.abs(xx) > half_window_m - pitch_m)
    edge_fraction = float(intensity[edge].sum() / intensity.sum())

    analytic_intensity = np.exp(-2.0 * (yy**2 + xx**2) / exact_m**2)
    analytic_edge = float(analytic_intensity[edge].sum() / analytic_intensity.sum())
    estimator_m = _second_moment_radius_m(analytic_intensity, pitch_m)
    estimator_bias = abs(estimator_m - exact_m) / exact_m

    return {
        "run": run,
        "margin": margin,
        "sampling_limit_m": sampling_limit_m,
        "distance_m": distance_m,
        "measured_radius_m": measured_m,
        "exact_radius_m": exact_m,
        "radius_error": error,
        "edge_fraction": edge_fraction,
        "analytic_edge_fraction": analytic_edge,
        "estimator_bias": estimator_bias,
    }


def _asm_sweep() -> dict[str, dict[str, Any]]:
    if not _ASM_SWEEP:
        for instance in B1_WAVE_ASM_VALIDITY.canonical_instances:
            _ASM_SWEEP[instance.instance_id] = _asm_measure(instance)
    return _ASM_SWEEP


def _run_asm_validity(instance_id: str) -> InstanceRun:
    instance = _instance(B1_WAVE_ASM_VALIDITY, instance_id)
    sweep = _asm_sweep()
    m = sweep[instance_id]
    side = str(instance.expected["side"])

    measurements = {
        "asm_radius_relative_error_vs_closed_form": Measurement(
            value=m["radius_error"],
            uncertainty=m["estimator_bias"],
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                f"second-moment radius {m['measured_radius_m'] * UM_PER_M:.6f} um "
                f"against {m['exact_radius_m'] * UM_PER_M:.6f} um analytic, at "
                f"z/limit = {m['distance_m'] / m['sampling_limit_m']:.4f}. Signed "
                f"normalized validity margin {m['margin']:+.6f}. The error bar is the "
                "same estimator's bias on the analytic field over the same grid, "
                f"{m['estimator_bias']:.3e}."
            ),
        ),
        "wrapped_power_fraction": Measurement(
            value=m["edge_fraction"],
            uncertainty=m["analytic_edge_fraction"],
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                "power within one pitch of the window edge, which is where an "
                "FFT-based propagator's wrap lands. The error bar is the ANALYTIC "
                f"field's own edge fraction, {m['analytic_edge_fraction']:.3e}, so "
                "'the beam is genuinely that big' and 'the beam folded back' stay "
                "separable."
            ),
        ),
    }

    controls: dict[str, Any] = {}
    if side == "outside":
        inside = sweep["B1-WAVE-ASM-VALIDITY-01"]
        controls["past-the-boundary-must-not-pass"] = control_result(
            "past-the-boundary-must-not-pass",
            "asm_radius_relative_error_vs_closed_form",
            baseline=Measurement(
                value=inside["radius_error"],
                uncertainty=inside["estimator_bias"],
                uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
                note=(
                    "the inside instance, at margin "
                    f"{inside['margin']:+.4f}: the same code, the same grid, the same "
                    "oracle, a distance inside the limit"
                ),
            ),
            mutated=measurements["asm_radius_relative_error_vs_closed_form"],
            threshold=2e-2,
            note=(
                "the mutation is the propagation DISTANCE and nothing else, which is "
                "what makes this a control rather than two unrelated runs."
            ),
        )
        controls["silent-wrap"] = control_result(
            "silent-wrap",
            "wrapped_power_fraction",
            baseline=Measurement(
                value=inside["edge_fraction"],
                uncertainty=inside["analytic_edge_fraction"],
                uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
                note="the inside instance's edge fraction: a run that did not wrap",
            ),
            mutated=measurements["wrapped_power_fraction"],
            threshold=1e-3,
            note=(
                "and it is SILENT: the far-side run succeeded, raised nothing, and "
                "returned a field that looks like a Gaussian."
            ),
        )

    return _verified(
        B1_WAVE_ASM_VALIDITY,
        instance,
        m["run"],
        measurements=measurements,
        controls=controls or None,
        diagnostics=[
            {
                "code": "THE_BOUNDARY_AND_WHICH_SIDE_THIS_IS",
                "detail": (
                    f"z = {m['distance_m'] * UM_PER_M:.3f} um, N pitch^2 / lambda = "
                    f"{m['sampling_limit_m'] * UM_PER_M:.3f} um, signed normalized "
                    f"margin {m['margin']:+.6f}, declared side {side!r}"
                ),
                "location": "verification/families/predicates.py::asm_transfer_function_sampling",
            },
            {
                "code": "IT_RAN",
                "detail": (
                    "no exception, no refusal, and a field came back: status "
                    f"{m['run']['record'].status.value}, output dtype "
                    f"{m['run']['placement']['dtype']}, edge power "
                    f"{m['edge_fraction']:.4e}. Crossing the sampling limit is SILENT, "
                    "which is the whole reason this family exists."
                ),
                "location": "benchmarks/instances/b1_wave.py::_asm_measure",
            },
            {
                "code": "WHY_THE_WAIST_IS_2_5_UM",
                "detail": (
                    "a tension resolved by measurement. At 8 um the beam's bandwidth is "
                    "1/(pi w0) = 0.04 cycles/um against a grid Nyquist of 2, so the "
                    "kernel's aliasing never touches it and even four times past the "
                    "limit the closed form is reproduced to 1.6e-5 -- the family could "
                    "not demonstrate its own failure mode. At 0.6 um it wraps but is "
                    "2.4 samples across the waist, so the estimator is 4% biased and "
                    "the inside instances fail too. 2.5 um is ten samples across the "
                    "waist and w(500 um) = 34.0 um against a 32 um half-window."
                ),
                "location": "src/verification/families/b1_wave.py::B1_WAVE_ASM_VALIDITY",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Shared verification
# ---------------------------------------------------------------------------


def _verified(
    family: Any,
    instance: Any,
    run: dict[str, Any],
    *,
    measurements: dict[str, Measurement],
    controls: dict[str, Any] | None = None,
    convergence: Any = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> InstanceRun:
    """Attach the placement observation and verify.

    The pitch check is here rather than per family because it is the same
    assertion every time: a PSF's or a radius's axes come from the propagated
    field's OUTPUT pitch, and reading the input pitch instead rescales every
    distance the measurement reports while leaving the intensity map entirely
    plausible.
    """
    record = run["record"]
    input_pitch = float(run["input_pitch_m"])
    output_pitch = run["output_pitch_m"]
    pitch_drift = max(abs(value - input_pitch) / input_pitch for value in output_pitch)

    extra = [
        {
            "code": "OBSERVED_PLACEMENT",
            "detail": (
                f"{run['placement']} -- read off the output array, never off the "
                "request. A process-global JAX platform pin produces a successful host "
                "run for a caller who asked for CUDA, with nothing raised."
            ),
            "location": "runtime/instance_runner.py::observed_placement",
        },
        {
            "code": "OUTPUT_PITCH_IS_THE_INPUT_PITCH",
            "detail": (
                f"input {input_pitch:.12e} m, output {output_pitch}, relative drift "
                f"{pitch_drift:.3e}. Angular-spectrum propagation preserves the pitch, "
                "which makes reading the OUTPUT pitch cheap rather than vacuous: it is "
                "the same assertion for a method that does not, and the failure it "
                "guards is silent."
            ),
            "location": "verification/psf_measurement.py",
        },
        {
            "code": "PAD_STATE",
            "detail": f"pad_width {run['pad_width']} on the output artifact",
            "location": "core/boundary.py::ComplexField",
        },
    ]
    merged = list(record.diagnostics) + extra + list(diagnostics or [])
    stamped = record.model_copy(update={"diagnostics": merged})
    # ``artifacts`` is excluded from the model dump, so a plain copy loses it.
    stamped.artifacts.update(record.artifacts)

    return InstanceRun(
        family=family,
        instance=instance,
        record=stamped,
        result=verify(
            family,
            instance,
            stamped,
            measurements=measurements,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


# ---------------------------------------------------------------------------
# WAVE-3: the device request that cannot be honoured
# ---------------------------------------------------------------------------


def device_observation() -> dict[str, Any]:
    """Where a Chromatix propagation actually ran, read off the array that ran it.

    One row per requested device. Each row that EXECUTED carries three
    independent readings of its placement, kept apart rather than collapsed:

    ``observed``
        The observation. ``propagation.py`` calls ``core.arrays.array_state`` on
        ``field_out.u`` *before* ``jax.device_get`` copies anything, and stamps
        it as ``metadata['execution']['actual']``. This is the only reading taken
        while the array is still where the computation happened.
    ``artifact``
        The same fact as the adapter stamped it on the boundary ``ArtifactRecord``
        -- a second reading of one observation, which catches the artifact and the
        metadata disagreeing.
    ``persisted``
        The ``.npy`` beside the record, which is host bytes by construction and
        therefore reports ``numpy``/``cpu`` *however* the run was placed. Carried
        so the two can be contrasted instead of confused. This used to be the
        only reading, and on the ray side the identical mistake reported a
        genuine ``cuda:0`` trace as a host downgrade -- the observation was the
        defect, not the execution, and it was invisible without a device.

    Why the request is never the answer: a process-global JAX platform pin
    produces a completely successful host run for a caller who asked for CUDA,
    with nothing raised. On a CPU-only image the CUDA row is REFUSED instead, and
    that half of the claim is what ``tests/test_b1_wave_instances.py`` asserts;
    the executed-on-CUDA half needs a device and lives in
    ``tests/test_b1_wave_gpu.py``.
    """
    instance = _instance(B1_WAVE_GAUSS, "B1-WAVE-GAUSS-01")
    p = instance.parameters
    n = int(p["grid_n"])
    pitch_m = float(p["sample_pitch_um"]) / UM_PER_M
    wavelength_m = float(p["wavelength_um"]) / UM_PER_M
    field = _gaussian(n, pitch_m, float(p["waist_um"]) / UM_PER_M)

    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray[Any, Any]] = {}
    for device in ("cpu", "cuda"):
        directory = Path(tempfile.mkdtemp(prefix="b1-wave-device-"))
        source = field_source(
            np.asarray(field, dtype=np.complex64),
            wavelength_m=wavelength_m,
            sample_pitch_m=pitch_m,
            directory=directory,
            artifact_id=f"device-{device}",
        )
        graph = GraphSpec.model_validate(
            {
                "schema_version": "0.1.0",
                "nodes": [
                    {
                        "id": "wave",
                        "model": "M_WAVE_CHROMATIX",
                        "config": {
                            "propagation": "angular_spectrum",
                            "z_m": float(p["distance_um"]) / UM_PER_M,
                            "refractive_index": 1.0,
                            "pad_width": 0,
                            "device": device,
                            "output_dir": str(directory),
                        },
                    }
                ],
            }
        )
        refusal, record = probe_refusal(
            lambda graph=graph, source=source: execute(
                graph, instance, inputs={"wave.input_field": source}
            )
        )
        row: dict[str, Any] = {"requested_device": device}
        if refusal is not None:
            row |= {
                "outcome": "refused",
                "code": refusal.code,
                "detail": refusal.detail,
            }
            rows.append(row)
            continue
        node = record.nodes[0]
        if node.refusal is not None or "wave:output_field" not in record.artifacts:
            row |= {
                "outcome": "refused",
                "code": None if node.refusal is None else str(node.refusal.kind),
                "detail": node.error_message or (node.refusal.detail if node.refusal else ""),
            }
            rows.append(row)
            continue

        artifact = record.artifacts["wave:output_field"]
        execution = dict(artifact.metadata.get("execution") or {})
        # THE reading, off the live JAX array before the host copy.
        observed = dict(execution.get("actual") or {})
        array = np.load(artifact.uri)
        row |= {
            "outcome": "executed",
            "observation_source": (
                "artifact.metadata['execution']['actual'], which propagation.py fills "
                "from core.arrays.array_state(field_out.u) before jax.device_get -- not "
                "the request, not JAX_PLATFORMS, not jax.default_backend()"
            ),
            "observed": observed,
            "artifact": {
                "device": str(artifact.device),
                "framework": str(artifact.framework),
                "dtype": str(artifact.dtype),
            },
            "persisted": observed_placement(array),
            "requested_in_metadata": execution.get("requested"),
            "resolved": execution.get("resolved"),
            "device_mismatch": execution.get("device_mismatch"),
            # A process-wide fact that says nothing about where THIS array landed,
            # recorded precisely so it can be seen not to have been used as the answer.
            "jax_default_backend": execution.get("jax_default_backend"),
            "honoured": observed.get("device", "").split(":")[0] == device,
        }
        rows.append(row)
        outputs[device] = array

    executed = tuple(r for r in rows if r["outcome"] == "executed")
    return {
        "rows": tuple(rows),
        "executed": executed,
        "refused": tuple(r for r in rows if r["outcome"] == "refused"),
        "cuda_executed": any(
            r["observed"].get("device", "").startswith("cuda") for r in executed
        ),
        "agreement": _device_agreement(outputs),
        "outputs": outputs,
    }


def _device_agreement(
    outputs: Mapping[str, np.ndarray[Any, Any]],
) -> dict[str, Any]:
    """Do the two devices compute the same field, and is that comparison available.

    A ``cuda`` label on an output nobody compared is a placement claim, not an
    execution claim: a stub that returned zeros on the device would satisfy every
    assertion about where it ran. So the two arms are differenced when both exist,
    against the complex64 floor -- one float32 epsilon per radian of accumulated
    phase, the same basis B1-WAVE-FWDBWD uses -- and reported as UNAVAILABLE with
    the reason when they do not, never as agreement by default.
    """
    if "cpu" not in outputs or "cuda" not in outputs:
        return {
            "status": "unavailable",
            "reason": f"executed arms: {sorted(outputs)}",
        }
    cpu, cuda = outputs["cpu"], outputs["cuda"]
    if cpu.shape != cuda.shape:
        return {"status": "unavailable", "reason": f"shapes {cpu.shape} vs {cuda.shape}"}
    instance = _instance(B1_WAVE_GAUSS, "B1-WAVE-GAUSS-01")
    accumulated_phase_rad = (
        2.0
        * math.pi
        * float(instance.parameters["distance_um"])
        / float(instance.parameters["wavelength_um"])
    )
    threshold = float(np.finfo(np.float32).eps) * accumulated_phase_rad
    measured = relative_l2_field(cuda, cpu)
    return {
        "status": "measured",
        "metric": "cpu_vs_cuda_relative_l2_field",
        "measured": measured,
        "threshold": threshold,
        "threshold_basis": (
            "one float32 epsilon per radian of accumulated phase, "
            f"eps32 * 2*pi*z/lambda = {threshold:.3e} over "
            f"{accumulated_phase_rad:.1f} rad. Chromatix is complex64 on both devices, "
            "so this bounds a reassociated FFT reduction and not a precision change"
        ),
        "met": measured <= threshold,
        "identical": bool(np.array_equal(cpu, cuda)),
    }


def write_device_observation_record(
    directory: Path | None = None, *, allow_downgrade: bool = False
) -> Path:
    """Persist the device observation so it is checkable without a device.

    The point of committing it is that the GPU-positive half of this criterion
    otherwise exists only in a session nobody can re-open. The GPU test re-measures
    and compares, so a record that went stale would fail rather than be believed.

    A CPU-only session refuses to overwrite a CUDA-positive record. That is not
    politeness about file ownership: the CPU run produces a structurally valid
    record whose CUDA row says "refused", and writing it would delete the only
    evidence for an acceptance criterion while leaving a file that still looks
    complete. ``allow_downgrade=True`` is the deliberate way to say the device
    really did go away.
    """
    from verification.evidence import record_provenance

    observation = device_observation()
    directory = directory or (ROOT / "benchmarks" / "probes" / "records" / "chromatix")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "b1_wave_device_observation.json"

    if path.is_file() and not observation["cuda_executed"] and not allow_downgrade:
        existing = json.loads(path.read_text())
        if existing.get("environment", {}).get("cuda_executed"):
            raise RuntimeError(
                f"{path.name} records a CUDA execution and this session has none. "
                "Refusing to overwrite it with a host-only observation: that would "
                "delete the evidence for the GPU acceptance criterion and leave a "
                "file that still looks complete. Re-run under `./run.sh --gpu`, or "
                "pass allow_downgrade=True if the device is genuinely gone."
            )

    payload: dict[str, Any] = {
        "probe": "instances/b1_wave::write_device_observation_record",
        "question": (
            "when the Chromatix graph node is asked for a device, is the device it "
            "reports read off the array it produced?"
        ),
        "measurement_method": (
            "one B1-WAVE-GAUSS-01 propagation per requested device through "
            "GraphExecutor; placement taken from "
            "artifact.metadata['execution']['actual'], which propagation.py fills "
            "from core.arrays.array_state(field_out.u) before jax.device_get. The "
            "persisted .npy placement is recorded alongside as a contrast, not as "
            "the observation. Cross-device agreement is differenced on the output "
            "fields against one float32 epsilon per radian of accumulated phase."
        ),
        "environment": {
            "cuda_executed": observation["cuda_executed"],
            "cuda_unavailable_reason": (
                None
                if observation["cuda_executed"]
                else next(
                    (
                        r.get("detail")
                        for r in observation["rows"]
                        if r["requested_device"] == "cuda" and r["outcome"] == "refused"
                    ),
                    "no CUDA row refused and none executed",
                )
            ),
            "jax": _jax_build(),
        },
        # The rows are JSON-safe as they stand; the output arrays live under
        # observation["outputs"] and deliberately do not reach the record.
        "rows": list(observation["rows"]),
        "agreement": observation["agreement"],
    }
    payload["record_provenance"] = record_provenance(
        probe="instances/b1_wave::write_device_observation_record", root=ROOT
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _jax_build() -> dict[str, Any]:
    """Which JAX is installed and whether it can reach a device.

    The default image ships CPU-only jaxlib and the GPU image adds
    ``jax-cuda12-plugin``; that is the difference between the CUDA row refusing
    and executing, so it belongs in the record rather than in the operator's
    memory of which image they used.
    """
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is pinned in both images
        return {"available": False}
    from solvers.chromatix.execution import _jax_gpu_unavailable_reason

    return {
        "available": True,
        "version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "gpu_unavailable_reason": _jax_gpu_unavailable_reason(),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, Any] = {
    "B1-WAVE-GAUSS-01": _run_gauss,
    "B1-WAVE-AIRY-01": _run_airy,
    "B1-WAVE-TILT-01": _run_tilt,
    "B1-WAVE-PLANEPHASE-01": _run_planephase,
    "B1-WAVE-FWDBWD-01": _run_fwdbwd,
    "B1-WAVE-TALBOT-01": _run_talbot,
    "B1-WAVE-ASM-VALIDITY-01": lambda: _run_asm_validity("B1-WAVE-ASM-VALIDITY-01"),
    "B1-WAVE-ASM-VALIDITY-02": lambda: _run_asm_validity("B1-WAVE-ASM-VALIDITY-02"),
    "B1-WAVE-ASM-VALIDITY-03": lambda: _run_asm_validity("B1-WAVE-ASM-VALIDITY-03"),
}

_FAMILIES = (
    B1_WAVE_GAUSS,
    B1_WAVE_AIRY,
    B1_WAVE_TILT,
    B1_WAVE_PLANEPHASE,
    B1_WAVE_FWDBWD,
    B1_WAVE_TALBOT,
    B1_WAVE_ASM_VALIDITY,
)


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(
        instance.instance_id for family in _FAMILIES for instance in family.canonical_instances
    )


def run_instance(instance_id: str) -> InstanceRun:
    try:
        runner = _RUNNERS[instance_id]
    except KeyError:
        raise KeyError(
            f"no runner for {instance_id!r}. Declared: {sorted(declared_instance_ids())}"
        ) from None
    return runner()


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


def _describe(metric: Any) -> str:
    verdict = "" if metric.met is None else (" MET" if metric.met else " UNMET")
    return f"{metric.metric}={metric.measured.value:.6g}{verdict}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--instance", default=None)
    parser.add_argument(
        "--device",
        action="store_true",
        help="run the device observation instead of the instances, and write its record",
    )
    args = parser.parse_args()

    if args.device:
        observation = device_observation()
        for row in observation["rows"]:
            if row["outcome"] == "executed":
                observed = row["observed"]
                print(
                    f"{row['requested_device']:<6} executed  "
                    f"{observed['namespace']}/{observed['device']}/{observed['dtype']}"
                    f"  persisted={row['persisted']['namespace']}/{row['persisted']['device']}"
                    f"  honoured={row['honoured']}"
                )
            else:
                print(f"{row['requested_device']:<6} {row['outcome']:<9} {row.get('code')}")
        agreement = observation["agreement"]
        if agreement["status"] == "measured":
            print(
                f"\ncpu vs cuda: {agreement['measured']:.4e} against "
                f"{agreement['threshold']:.4e}  met={agreement['met']}"
            )
        else:
            print(f"\ncpu vs cuda: UNAVAILABLE ({agreement['reason']})")
        try:
            path = write_device_observation_record()
        except RuntimeError as exc:
            print(f"\nnot written: {exc}")
            return 1
        print(f"\n-> {path.relative_to(ROOT)}")
        return 0

    runs = {args.instance: run_instance(args.instance)} if args.instance else run_all()
    for instance_id, run in runs.items():
        metrics = ", ".join(_describe(m) for m in run.result.physics_accuracy)
        print(f"{instance_id:<28} status={run.result.status.value:<18} {metrics}")
        controls = ", ".join(
            f"{c.control_id}:{c.outcome.value}" for c in run.result.negative_control_results
        )
        if controls:
            print(f"{'':<28} controls: {controls}")
        if args.write:
            path = write_instance_record(run, driver="instances/b1_wave")
            print(f"{'':<28} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
