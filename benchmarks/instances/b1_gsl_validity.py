"""B1-GSL-VALIDITY: sweep the smooth-to-diffractive transition (CHE-146, M2.10).

Runs ``couplers.generalized_snell`` (via ``couplers.interaction.diffractive_interaction``,
model=GENERALIZED_SNELL) against the analytic grating equation, over a
declared spatial-frequency sweep (period vs. a fixed 0.2 um sample pitch) and a
duty-cycle sweep (a binary grating's discontinuity, at a period where the
spatial-frequency axis alone is still smooth). ``FULL_FIELD`` runs alongside
each instance as CROSS_ROUTE diagnostic evidence, never as the oracle -- see
the family module's docstring for why.

Run it::

    ./run.sh python benchmarks/instances/b1_gsl_validity.py --write
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from core.boundary import RayBundle, ReferencePlane
from core.paths import repository_root
from couplers.generalized_snell import GeneralizedSnellDiagnostics
from couplers.generalized_snell import single_order_dominance as _measure_dominance
from couplers.interaction import (
    DiffractiveModel,
    DiffractiveSurface,
    FullFieldParameters,
    GeneralizedSnellParameters,
    diffractive_interaction,
)
from runtime.instance_runner import probe_refusal, record_from_probe
from verification.evidence import InstanceRun, write_instance_record
from verification.families.b1_gsl_validity import B1_GSL_VALIDITY, grating_direction_cosine
from verification.result import Measurement, NegativeControlOutcome, NegativeControlResult, UncertaintyBasis
from verification.verifier import verify

__all__ = ["declared_instance_ids", "run_all", "run_instance"]

ROOT = repository_root()

#: Fixed for every instance: a 256x256 grid at a 0.2 um pitch, comfortably
#: holding the 65-px dominance window and several periods even at the
#: coarsest (2e-5 m) sweep point.
_PITCH_M = (0.2e-6, 0.2e-6)
_N = 256
_PLANE = ReferencePlane(name="doe", z_m=0.0)
#: Rays launched across exactly one period, at the ray count that gave
#: GENERALIZED_SNELL's own acceptance tests (tests/test_diffractive_interaction.py)
#: a resolved answer without needing more than a handful.
_RAYS_PER_PERIOD = 9


def _instance(instance_id: str) -> Any:
    for candidate in B1_GSL_VALIDITY.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"B1-GSL-VALIDITY declares no instance {instance_id!r}")


def _surface(params: dict[str, Any]) -> DiffractiveSurface:
    """The declared grating, built from ``profile_kind``.

    ``blazed_ramp`` is a single-harmonic linear sawtooth: its local phase
    gradient is ``2 pi / period_m`` everywhere, so it carries all its power in
    one order at any period -- the period sweep's failure mode is therefore the
    finite-difference stencil aliasing against the sample pitch, not a
    multi-order split. ``binary_phase`` alternates between ``0`` and
    ``phase_depth_rad`` for a ``duty_cycle`` fraction of the period, which is a
    genuinely multi-harmonic profile at fixed period -- the axis
    ``single_order_dominance`` exists to measure.
    """
    n = _N
    x = (np.arange(n) - n // 2) * _PITCH_M[1]
    period_m = float(params["period_m"])
    if params["profile_kind"] == "blazed_ramp":
        phase_row = 2.0 * math.pi * x / period_m
    elif params["profile_kind"] == "binary_phase":
        frac = np.mod(x / period_m, 1.0)
        phase_row = np.where(frac < float(params["duty_cycle"]), float(params["phase_depth_rad"]), 0.0)
    else:
        raise ValueError(f"unknown profile_kind {params['profile_kind']!r}")
    phase2d = np.tile(phase_row, (n, 1))
    return DiffractiveSurface.from_phase(phase2d, sample_pitch_m=_PITCH_M, plane=_PLANE)


def _bundle_across_one_period(params: dict[str, Any]) -> RayBundle:
    """``_RAYS_PER_PERIOD`` rays spanning one period at ``y=0``.

    A periodic grating deflects every ray to the same angle regardless of
    where it lands within the period, so the spread of GENERALIZED_SNELL's
    actual outgoing angle across these positions IS the model's own breakdown
    signal -- see the family module's docstring.
    """
    period_m = float(params["period_m"])
    theta_in = float(params["incidence_angle_rad"])
    xs = np.linspace(-period_m / 2.0, period_m / 2.0, _RAYS_PER_PERIOD, endpoint=False)
    positions = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    direction = np.array([math.sin(theta_in), 0.0, math.cos(theta_in)])
    directions = np.tile(direction, (_RAYS_PER_PERIOD, 1))
    return RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=float(params["wavelength_m"]),
        reference_plane=_PLANE,
        amplitude=np.ones(_RAYS_PER_PERIOD, dtype=np.complex128),
        optical_path_length_m=np.zeros(_RAYS_PER_PERIOD),
        optical_path_length_reference="launch",
    )


def _run_generalized_snell(params: dict[str, Any]) -> tuple[Any, GeneralizedSnellDiagnostics, float]:
    surface = _surface(params)
    bundle = _bundle_across_one_period(params)
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=int(params["order"]), patch_px=int(params["patch_px"])),
    )
    diagnostics = result.model_diagnostics
    assert isinstance(diagnostics, GeneralizedSnellDiagnostics)

    dx_analytic = grating_direction_cosine(params)
    angle_analytic = math.asin(dx_analytic)
    dx_out = np.asarray(result.outgoing.directions[:, 0])
    angles_out = np.arcsin(np.clip(dx_out, -1.0, 1.0))
    worst_error = float(np.max(np.abs(angles_out - angle_analytic)))
    return surface, diagnostics, worst_error


def _run_full_field(surface: DiffractiveSurface, params: dict[str, Any]) -> float:
    """FULL_FIELD's own principal-order power fraction, from one launched ray.

    Diagnostic only -- see the family module's docstring for why FULL_FIELD is
    CROSS_ROUTE evidence and never the oracle.
    """
    bundle = RayBundle(
        positions_m=np.zeros((1, 3)),
        directions=np.array([[math.sin(float(params["incidence_angle_rad"])), 0.0, math.cos(float(params["incidence_angle_rad"]))]]),
        wavelength_m=float(params["wavelength_m"]),
        reference_plane=_PLANE,
        amplitude=np.ones(1, dtype=np.complex128),
        optical_path_length_m=np.zeros(1),
        optical_path_length_reference="launch",
    )
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None),
    )
    weights = np.abs(np.asarray(result.outgoing.amplitude))
    power = weights**2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.max(power)) / total


def _dominance_direct(surface: DiffractiveSurface, params: dict[str, Any]) -> tuple[float, float]:
    """``single_order_dominance`` measured directly, bypassing the per-ray hard gate.

    ``generalized_snell_step``'s smoothness check is a *local*, per-ray raw
    finite-difference step -- exactly right for deciding whether one ray's
    redirection can be trusted, and exactly why a ray whose stencil straddles a
    binary grating's edge must refuse rather than return a number. That is a
    different, stricter question from "how much of this window's spectral
    power sits in the requested order", which is a windowed measurement over
    many samples and stays well defined even when some individual ray-level
    stencils would not. Both are real, and reporting only the one that never
    refuses would hide exactly the boundary this family exists to find.
    """
    dx_analytic = grating_direction_cosine(params)
    dominance, margin = _measure_dominance(
        surface.transmission,
        sample_pitch_m=surface.sample_pitch_m,
        center_xy_m=(0.0, 0.0),
        patch_px=int(params["patch_px"]),
        wavelength_m=float(params["wavelength_m"]),
        target_dir_xy=(dx_analytic, 0.0),
    )
    return dominance, margin


def _run(instance_id: str) -> InstanceRun:
    instance = _instance(instance_id)
    params = dict(instance.parameters)
    surface = _surface(params)
    dominance, dominance_margin = _dominance_direct(surface, params)

    refusal, outcome = probe_refusal(lambda: _run_generalized_snell(params))

    if refusal is not None:
        # A genuine breakdown, not a bug: at least one of the rays spread
        # across this period sits close enough to a phase discontinuity that
        # the per-ray local-gradient estimate cannot be trusted there. This IS
        # "GSL breaks" -- the multi-order/discontinuous end of the ticket's own
        # regime table -- reported as a structured refusal per AGENTS.md rather
        # than as an invented angle.
        record = record_from_probe(
            instance,
            component="C_GENERALIZED_SNELL",
            node_id="gsl_validity_sweep",
            refusal=refusal,
            observed_parameters={"single_order_dominance": dominance},
            diagnostics=[
                {
                    "code": "WINDOWED_DOMINANCE_STILL_MEASURED",
                    "detail": (
                        f"single_order_dominance={dominance:.6f} (margin "
                        f"{dominance_margin:+.4f}), measured directly and "
                        "independent of the ray-level refusal above"
                    ),
                    "location": "src/couplers/generalized_snell.py::single_order_dominance",
                }
            ],
        )
        return InstanceRun(
            family=B1_GSL_VALIDITY,
            instance=instance,
            record=record,
            result=verify(B1_GSL_VALIDITY, instance, record, measurements=None),
        )

    _, diagnostics, worst_error = outcome
    full_field_dominance = _run_full_field(surface, params)
    dominance_agreement = abs(diagnostics.single_order_dominance - full_field_dominance)

    measurements = {
        "deflection_angle_worst_error_rad": Measurement(
            value=worst_error,
            uncertainty=None,
            uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
            note=(
                f"worst of {_RAYS_PER_PERIOD} rays spread across one period against "
                f"the analytic grating angle asin({grating_direction_cosine(params):.6f}). "
                f"single_order_dominance={diagnostics.single_order_dominance:.6f} "
                f"(margin {diagnostics.single_order_dominance_margin:+.4f}), "
                f"gradient-smoothness margin "
                f"{diagnostics.worst_local_gradient_smoothness_margin:+.4f}, "
                f"propagating-order margin "
                f"{diagnostics.worst_propagating_order_margin:+.4f}."
            ),
        ),
        "full_field_dominance_agreement": Measurement(
            value=dominance_agreement,
            uncertainty=None,
            uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
            note=(
                f"GENERALIZED_SNELL dominance {diagnostics.single_order_dominance:.6f} vs "
                f"FULL_FIELD principal-order power fraction {full_field_dominance:.6f}"
            ),
        ),
    }
    diagnostic_entries = [
        {
            "code": "M2_7_PREDICATE_MARGINS",
            "detail": (
                f"propagating_order_margin={diagnostics.worst_propagating_order_margin:+.6f}, "
                f"gradient_smoothness_margin="
                f"{diagnostics.worst_local_gradient_smoothness_margin:+.6f}, "
                f"single_order_dominance_margin="
                f"{diagnostics.single_order_dominance_margin:+.6f}"
            ),
            "location": "src/couplers/generalized_snell.py::GeneralizedSnellDiagnostics",
        },
    ]
    if float(params["incidence_angle_rad"]) != 0.0:
        # Found while authoring this instance, not a claim this family sets out
        # to make: generalized_snell_step's own single_order_dominance call
        # targets the FULL outgoing direction cosine (incident tilt + the
        # order's own kick), not the order's own spatial frequency in the
        # phase profile alone. The two coincide at normal incidence, which is
        # why every on-axis instance above reads a sensible dominance and this
        # one does not -- dominance collapses to ~0 here for a profile that is
        # still a pure single-harmonic ramp. This is a limitation of predicate
        # 3's off-axis target direction in couplers/generalized_snell.py
        # (CHE-143, M2.7), not a genuine incidence-angle breakdown; M2.10 did
        # not set out to audit or fix that code and does not do so here. The
        # deflection-angle metric above is unaffected -- it compares the
        # measured OUTGOING angle to the analytic one directly, not through
        # this window -- which is how the discrepancy was isolated.
        diagnostic_entries.append(
            {
                "code": "PREDICATE_3_OFF_AXIS_TARGET_DIRECTION_LIMITATION",
                "detail": (
                    f"single_order_dominance={diagnostics.single_order_dominance:.6f} at "
                    f"incidence_angle_rad={params['incidence_angle_rad']:.4f} on a pure "
                    "single-harmonic blazed ramp, where on-axis instances at the same "
                    "period read ~0.8. Not evidence of an incidence-angle breakdown -- "
                    "see the driver's note at this diagnostic's call site"
                ),
                "location": "src/couplers/generalized_snell.py::generalized_snell_step "
                "(centroid_target computation)",
            }
        )
    record = record_from_probe(
        instance,
        component="C_GENERALIZED_SNELL",
        node_id="gsl_validity_sweep",
        refusal=None,
        observed_parameters={
            "single_order_dominance": diagnostics.single_order_dominance,
            "measured_gradient_smoothness_margin": diagnostics.worst_local_gradient_smoothness_margin,
        },
        diagnostics=diagnostic_entries,
    )
    return InstanceRun(
        family=B1_GSL_VALIDITY,
        instance=instance,
        record=record,
        result=verify(
            B1_GSL_VALIDITY,
            instance,
            record,
            measurements=measurements,
        ),
    )


# ---------------------------------------------------------------------------
# Negative controls, run once and reported against the finest instance
# ---------------------------------------------------------------------------


def _negative_controls(baseline_error: float) -> dict[str, NegativeControlResult]:
    controls: dict[str, NegativeControlResult] = {}

    # order=25 at a well-resolved 50-samples-per-period ramp, rather than a
    # short period at order 1: an evanescent order=1 case at this wavelength
    # and pitch is unavoidably within a few samples of Nyquist, where the
    # finite-difference gradient itself can alias to a falsely "smooth" small
    # value (found while authoring this file -- period_m=4e-7, order=1 aliases
    # to worst_propagating_order_margin=+1.0 instead of refusing, a silent
    # wrongness rather than a clean control). Order=25 keeps the PHYSICAL
    # surface -- and therefore the gradient estimate -- exactly as smooth as
    # PERIOD-01, and isolates the evanescent-order boundary from the
    # sampling boundary.
    evanescent_params = {
        "period_m": 1e-5,
        "wavelength_m": 5e-7,
        "incidence_angle_rad": 0.0,
        "order": 25,
        "index_incident": 1.0,
        "index_transmitted": 1.0,
        "profile_kind": "blazed_ramp",
        "duty_cycle": 0.5,
        "phase_depth_rad": 0.0,
        "patch_px": 65,
    }
    # sin(theta_out) = 1.25 at this order: no real solution, both for the
    # analytic oracle and for the shipping coupler it is checked against.
    analytic_refused = False
    try:
        grating_direction_cosine(evanescent_params)
    except ValueError:
        analytic_refused = True
    coupler_refusal, _ = probe_refusal(lambda: _run_generalized_snell(evanescent_params))
    controls["period-past-evanescent-boundary"] = NegativeControlResult(
        control_id="period-past-evanescent-boundary",
        outcome=(
            NegativeControlOutcome.FIRED
            if (analytic_refused and coupler_refusal is not None)
            else NegativeControlOutcome.DID_NOT_FIRE
        ),
        target_metric="deflection_angle_worst_error_rad",
        note=(
            f"period_m={evanescent_params['period_m']:.2e} at wavelength "
            f"{evanescent_params['wavelength_m']:.2e}: order "
            f"{evanescent_params['order']} needs sin(theta_out) = 1.25, which "
            "has no real solution. analytic oracle refused: "
            f"{analytic_refused}; shipping coupler refused: "
            f"{coupler_refusal.code if coupler_refusal is not None else False}"
        ),
    )

    order2_params = {
        "period_m": 5e-6,
        "wavelength_m": 5e-7,
        "incidence_angle_rad": 0.0,
        "order": 2,
        "index_incident": 1.0,
        "index_transmitted": 1.0,
        "profile_kind": "blazed_ramp",
        "duty_cycle": 0.5,
        "phase_depth_rad": 0.0,
        "patch_px": 65,
    }
    _, diagnostics, worst_error = _run_generalized_snell(order2_params)
    controls["order-the-profile-does-not-carry"] = NegativeControlResult(
        control_id="order-the-profile-does-not-carry",
        outcome=(
            NegativeControlOutcome.FIRED
            if diagnostics.single_order_dominance_margin < 0.0
            else NegativeControlOutcome.DID_NOT_FIRE
        ),
        target_metric="deflection_angle_worst_error_rad",
        baseline=Measurement(
            value=baseline_error, uncertainty=None, uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
            note="order=1 against a profile built for order=1",
        ),
        mutated=Measurement(
            value=worst_error, uncertainty=None, uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
            note=(
                f"order=2 against a single-harmonic profile: dominance "
                f"{diagnostics.single_order_dominance:.6f} "
                f"(margin {diagnostics.single_order_dominance_margin:+.4f}), "
                f"angle error {worst_error:.4e} rad"
            ),
        ),
        note="dominance margin is the control's own gate here, not the angle error",
    )
    return controls


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(i.instance_id for i in B1_GSL_VALIDITY.canonical_instances)


def run_instance(instance_id: str) -> InstanceRun:
    return _run(instance_id)


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


def _describe(metric: Any) -> str:
    if not metric.tolerance_may_gate:
        return f"{metric.metric}={metric.measured.value:.6g} (reported, not gating)"
    verdict = "" if metric.met is None else (" MET" if metric.met else " UNMET")
    return f"{metric.metric}={metric.measured.value:.6g}{verdict}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--instance", default=None)
    args = parser.parse_args()

    runs = {args.instance: run_instance(args.instance)} if args.instance else run_all()
    for instance_id, run in runs.items():
        metrics = ", ".join(_describe(m) for m in run.result.physics_accuracy)
        print(f"{instance_id:<28} {run.result.status.value:<18} {metrics}")
        if args.write:
            path = write_instance_record(run, driver="instances/b1_gsl_validity")
            print(f"{'':<28} -> {path.relative_to(ROOT)}")

    if not args.instance:
        smooth = runs["B1-GSL-VALIDITY-PERIOD-05"]
        baseline_error = next(
            m.measured.value
            for m in smooth.result.physics_accuracy
            if m.metric == "deflection_angle_worst_error_rad"
        )
        controls = _negative_controls(baseline_error)
        for control_id, result in controls.items():
            print(f"control {control_id}: {result.outcome.value} -- {result.note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
