"""B1-GSL-VALIDITY: where GENERALIZED_SNELL stops being valid (CHE-146, M2.10).

The rule this family exists to obey: benchmark a *physical* smooth-to-diffractive
transition, not "GENERALIZED_SNELL versus FULL_FIELD as implementation A versus
B". One surface, one axis -- the spatial frequency of a phase profile against the
sample pitch that represents it -- extended into the discontinuous, multi-order
regime, with two secondary axes: duty cycle (a blazed ramp is one harmonic; a
binary grating is many) and incidence angle.

The primary reference is the analytic grating equation
``sin(theta_out) = (n_i sin(theta_in) + m lambda / Lambda) / n_t``, exact and
independent of both diffractive-interaction models. ``FULL_FIELD`` is not an
oracle: :class:`~verification.claim_ledger.Oracle` reserves ``CROSS_ROUTE`` for
exactly this case (per the docstring, "two routes through our own code that
should agree... Optiland's FFTPSF/HuygensPSF share one front end"), and
``knowledge/couplers/generalized_snell/card.yaml`` already states the same
scope split in its own maturity note. The full-field comparison here is
diagnostic, reported and never gating.

Two of GENERALIZED_SNELL's own M2.7 validity margins are carried through
unchanged rather than re-derived: ``single_order_dominance`` is
``predicates.generalized_snell_single_order_dominance`` verbatim, reading a
value this family's driver measures by actually running
``couplers.generalized_snell`` on the declared surface. The local-gradient-
smoothness margin is exposed the same way -- as a *measured* input, because it
depends on the coupler's own finite-difference stencil and re-deriving it here
from a closed form would be a second, competing definition of the same
quantity.

Which predicate tracks which axis is not arbitrary. A pure single-tone ramp
carries all its local spectral power in one order at any period -- dominance
stays high all the way to the sampling floor -- so the period sweep's failure
mode is the finite-difference stencil aliasing (predicate 2, smoothness), not a
multi-order split (predicate 3, dominance). The duty-cycle sweep is the
opposite: a binary grating's harmonic content genuinely splits power across
orders at fixed period, which is exactly what dominance measures. The family
therefore reads the two axes against the two predicates suited to them, and
reports whether each measured breakdown (the worst deflection-angle error
crossing the family's own tolerance) locates the same period/duty-cycle as the
corresponding margin's own zero crossing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import fractional_margin
from verification.families.predicates import (
    generalized_snell_single_order_dominance as _dominance_predicate,
)
from verification.families.registry import register
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkFamily,
    BenchmarkLayer,
    ClaimKind,
    ExecutionPolicy,
    FamilyOracle,
    GateDisposition,
    GateStatus,
    Metric,
    NegativeControl,
    NumericalParameter,
    Oracle,
    OracleIndependence,
    PhysicalParameter,
    RepresentationParameter,
    SamplerAbsentReason,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = [
    "B1_GSL_VALIDITY",
    "MEASURED_GRADIENT_SMOOTHNESS",
    "GRATING_ORDER_PROPAGATES",
    "grating_direction_cosine",
    "grating_order_propagation_margin",
]


# ---------------------------------------------------------------------------
# The analytic oracle: the grating equation, independent of both models
# ---------------------------------------------------------------------------


def grating_direction_cosine(params: Mapping[str, Any]) -> float:
    """``sin(theta_out) = (n_i sin(theta_in) + m lambda / Lambda) / n_t``, exact.

    Returns the outgoing direction cosine along the grating's dispersion axis.
    Past the propagating limit there is no real solution -- the same physical
    boundary ``GRATING_ORDER_PROPAGATES`` expresses as a validity margin below,
    mirroring how ``b1_ray.py::_snell_refraction_angle`` treats total internal
    reflection as a regime rather than an exception.
    """
    n_i = float(params["index_incident"])
    n_t = float(params["index_transmitted"])
    theta_in = float(params["incidence_angle_rad"])
    order = float(params["order"])
    wavelength_m = float(params["wavelength_m"])
    period_m = float(params["period_m"])
    dx = (n_i * math.sin(theta_in) + order * wavelength_m / period_m) / n_t
    if abs(dx) > 1.0:
        raise ValueError(
            f"order {order:.0f} has no propagating solution at period {period_m:.3e} m: "
            f"the grating equation gives sin(theta_out) = {dx:.4f}, |.| > 1"
        )
    return dx


def grating_order_propagation_margin(params: Mapping[str, Any]) -> float:
    """The same equation's signed margin, without raising past the boundary."""
    n_i = float(params["index_incident"])
    n_t = float(params["index_transmitted"])
    theta_in = float(params["incidence_angle_rad"])
    order = float(params["order"])
    wavelength_m = float(params["wavelength_m"])
    period_m = float(params["period_m"])
    dx = (n_i * math.sin(theta_in) + order * wavelength_m / period_m) / n_t
    return fractional_margin(abs(dx), 1.0)


GRATING_ORDER_PROPAGATES = ValidityPredicate(
    predicate_id="GRATING_ORDER_PROPAGATES",
    statement=(
        "the requested diffraction order has a real solution of the grating "
        "equation: |sin(theta_out)| <= 1"
    ),
    basis=ValidityBasis.GENERALIZED_SNELL_PROPAGATING_ORDER,
    margin=grating_order_propagation_margin,
    blind_to=("everything about the order except whether it exists at all",),
)


def _measured_margin(key: str):
    def margin(params: Mapping[str, Any]) -> float:
        return float(params[key])

    return margin


MEASURED_GRADIENT_SMOOTHNESS = ValidityPredicate(
    predicate_id="MEASURED_LOCAL_GRADIENT_SMOOTHNESS",
    statement=(
        "GENERALIZED_SNELL's own local-gradient-smoothness margin (predicate 2, "
        "M2.7), as measured by actually running "
        "couplers.generalized_snell.generalized_snell_step on the declared "
        "surface -- reported rather than re-derived, since the margin is a "
        "property of the coupler's own finite-difference stencil"
    ),
    basis=ValidityBasis.GENERALIZED_SNELL_GRADIENT_SMOOTHNESS,
    margin=_measured_margin("measured_gradient_smoothness_margin"),
    blind_to=(
        "a uniformly aliased gradient -- see "
        "couplers.generalized_snell.local_gradient_smoothness_margin's own note",
    ),
)


# ---------------------------------------------------------------------------
# Shared policy
# ---------------------------------------------------------------------------

GSL_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.FLOAT64}),
    namespaces=frozenset({ArrayNamespace.NUMPY}),
    max_wall_seconds=60.0,
    max_peak_memory_gib=1.0,
    notes=(
        "every instance is a handful of rays against a 128x128 (or smaller) "
        "transmission grid; there is no reason to run this on GPU or in torch"
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "every surface and ray position is declared per instance; nothing here "
        "draws a sample"
    ),
)


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------

_B1_GSL_VALIDITY = BenchmarkFamily(
    family_id="B1-GSL-VALIDITY",
    family_version="1.0.0",
    category=BenchmarkCategory.B1,
    layer=BenchmarkLayer.NUMERICAL,
    question=(
        "over what range of grating period (relative to wavelength and sample "
        "pitch), phase discontinuity and incidence angle does GENERALIZED_SNELL's "
        "per-ray local grating equation stay close to the analytic grating angle, "
        "and do its own declared validity margins predict the boundary?"
    ),
    components=("C_GENERALIZED_SNELL", "C_PLANAR_DOE_STEP"),
    claim_kind=ClaimKind.FORWARD_ACCURACY,
    parameters=(
        PhysicalParameter(
            "period_m", "the grating period Lambda", unit="m", domain=(3e-7, 5e-5),
            default=5e-6,
        ),
        PhysicalParameter(
            "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5e-7
        ),
        PhysicalParameter(
            "incidence_angle_rad", "angle of incidence from the surface normal",
            unit="rad", domain=(0.0, 0.5), default=0.0,
        ),
        PhysicalParameter(
            "order", "the requested diffraction order m", domain=(-5, 5), default=1
        ),
        PhysicalParameter(
            "index_incident", "index on the incident side", domain=(1.0, 2.5), default=1.0
        ),
        PhysicalParameter(
            "index_transmitted", "index on the transmitted side", domain=(1.0, 2.5),
            default=1.0,
        ),
        RepresentationParameter(
            "profile_kind",
            "which phase profile realizes the period: a single-harmonic blazed "
            "ramp (one order by construction) or a binary (0/phase_depth_rad) "
            "grating whose duty cycle controls how much power leaves the "
            "requested order",
            domain=("blazed_ramp", "binary_phase"),
            default="blazed_ramp",
        ),
        PhysicalParameter(
            "duty_cycle",
            "fraction of one period at the high phase level, for profile_kind="
            "binary_phase. 0.5 is the maximally discontinuous symmetric case; "
            "unused (and reported as 0.5) for blazed_ramp",
            domain=(0.01, 0.99), default=0.5,
        ),
        PhysicalParameter(
            "phase_depth_rad",
            "the binary grating's high phase level, for profile_kind=binary_phase. "
            "Unused (and reported as 0.0) for blazed_ramp, whose depth is the "
            "period itself",
            unit="rad", domain=(0.0, 2.0 * math.pi), default=math.pi,
        ),
        PhysicalParameter(
            "single_order_dominance",
            "MEASURED fraction of local spectral power in the requested order -- "
            "an input this family's driver computes by running "
            "couplers.generalized_snell.single_order_dominance on the declared "
            "surface, not a closed form",
            domain=(0.0, 1.0), default=1.0,
        ),
        PhysicalParameter(
            "measured_gradient_smoothness_margin",
            "MEASURED local-gradient-smoothness margin (M2.7 predicate 2) -- an "
            "input this family's driver computes by running "
            "couplers.generalized_snell.generalized_snell_step, not a closed form",
            domain=(-1.0, 1.0), default=1.0,
        ),
        NumericalParameter(
            "patch_px",
            "window side (samples) for the single-order-dominance measurement "
            "and the smoothness predicate's transverse scale -- the numerical "
            "realization axis this NUMERICAL-layer family declares",
            domain=(3, 129), default=65, refines_toward=1,
        ),
    ),
    validity=(GRATING_ORDER_PROPAGATES, MEASURED_GRADIENT_SMOOTHNESS, _dominance_predicate()),
    oracle=FamilyOracle(
        kind=Oracle.ANALYTIC,
        independence=OracleIndependence.INDEPENDENT,
        description=(
            "the grating equation sin(theta_out) = (n_i sin(theta_in) + m lambda / "
            "Lambda) / n_t, exact and shared by no code in couplers/"
        ),
        callable=grating_direction_cosine,
        reference="grating equation; no approximation and no fitted constant",
    ),
    metrics=(
        Metric(
            name="deflection_angle_worst_error_rad",
            description=(
                "the worst (max-|error|) absolute error, over several rays "
                "launched across one period, between GENERALIZED_SNELL's actual "
                "outgoing angle and the analytic grating angle. A periodic "
                "grating deflects every ray to the same angle regardless of "
                "where it lands in the period, so the spread across launch "
                "position IS the model's own breakdown signal"
            ),
            unit="rad",
            blind_to=(
                "the spatial pattern of the disagreement within the period -- a "
                "worst-case scalar cannot say whether the error is one outlier "
                "ray near a discontinuity or a uniform bias",
            ),
        ),
        Metric(
            name="full_field_dominance_agreement",
            description=(
                "|GENERALIZED_SNELL's single_order_dominance - FULL_FIELD's own "
                "principal-order power fraction (constructed from its enumerated "
                "outgoing amplitudes)|. Diagnostic: the two are two different "
                "constructions of 'order concentration', not the same quantity "
                "from two code paths, and FULL_FIELD is CROSS_ROUTE evidence, "
                "not an oracle -- see the module docstring"
            ),
            unit=None,
            blind_to=(
                "which model is closer to the analytic grating angle -- this "
                "metric only says whether the two routes' own order-concentration "
                "readings agree, never which one (if either) is right",
            ),
        ),
    ),
    tolerances=(
        Tolerance(
            metric="deflection_angle_worst_error_rad",
            threshold=1e-3,
            basis=(
                "the smooth-limit instances (period >> pitch, blazed ramp) measure "
                "worst-case error at or below float64 round-off through asin and "
                "the local finite-difference stencil; 1e-3 rad is set well above "
                "that floor and below the multi-radian-fraction errors the "
                "near-Nyquist and high-duty-cycle-discontinuity instances show. "
                "Basis completed after executing the family, per the B2-EQUIV "
                "precedent: a threshold is not well defined before the family "
                "that owes it has run"
            ),
            basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
            may_gate=True,
            rejects=(
                "a local-gradient estimate that has aliased against the sample "
                "pitch, and a multi-order surface whose local phase gradient no "
                "longer points at the requested order"
            ),
        ),
        Tolerance(
            metric="full_field_dominance_agreement",
            threshold=1.0,
            basis=(
                "diagnostic only: the threshold is the metric's own maximum "
                "possible value, so this tolerance can never gate and exists "
                "only so the metric can be reported through the same machinery "
                "as a gating one"
            ),
            basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
            may_gate=False,
        ),
    ),
    negative_controls=(
        NegativeControl(
            control_id="period-past-evanescent-boundary",
            description=(
                "request a period short enough that the grating equation has no "
                "real solution for the requested order, and require the "
                "GRATING_ORDER_PROPAGATES margin (and the coupler's own refusal) "
                "to report it rather than returning a normalized nonsense angle"
            ),
            mutation="period_m set below wavelength_m / (1 - n_i sin(theta_in))",
            target_metric="deflection_angle_worst_error_rad",
        ),
        NegativeControl(
            control_id="order-the-profile-does-not-carry",
            description=(
                "request order=2 of a single-harmonic blazed ramp, whose local "
                "phase gradient only ever encodes order 1. The dominance margin "
                "must go negative even though the order-1 grating equation has a "
                "perfectly good propagating solution"
            ),
            mutation="order=2 against a profile built for order=1",
            target_metric="deflection_angle_worst_error_rad",
        ),
    ),
    failure_semantics=(
        VerificationStatus.OUT_OF_VALIDITY,
        VerificationStatus.INVALID_CONFIGURATION,
    ),
    execution_policy=GSL_EXECUTION,
    stochastic_policy=DETERMINISTIC,
    gate_disposition=GateDisposition(
        status=GateStatus.MET,
        metric="deflection_angle_worst_error_rad",
        observed=1.11022e-16,
        evidence=(
            "benchmarks/instances/b1_gsl_validity.py",
            "benchmarks/instances/records/B1-GSL-VALIDITY-PERIOD-05.json",
        ),
        note=(
            "the primary claim -- GENERALIZED_SNELL agrees with the analytic "
            "grating angle in the smooth limit -- is MET at float64 round-off "
            "(worst 1.11e-16 rad over PERIOD-01..05, 9 to 45 samples/period). "
            "The deliberately UNMET instances (PERIOD-07, PERIOD-08: 1.36 and "
            "1.41 rad) are the demonstrated breakdown this family exists to "
            "find, not a defect against this gate.\n\n"
            "The open finding is in which of GENERALIZED_SNELL's own margins "
            "predicts that breakdown. MEASURED_LOCAL_GRADIENT_SMOOTHNESS "
            "(predicate 2) shrinks correctly from +0.93 to +0.11 across "
            "PERIOD-01..05 and refuses outright at PERIOD-06 (period=8e-7 m, "
            "exactly 4 samples/period -- the finite-difference stencil's own "
            "worst-case aliasing resonance, a phase step of exactly pi). Past "
            "that point it is NON-MONOTONIC: PERIOD-07/08 measure +0.14 and "
            "+0.33 -- comfortably 'inside' -- while the deflection angle is by "
            "then off by 1.3-1.4 rad. A gate that only checked 'did predicate "
            "2 refuse' would call PERIOD-07/08 fine.\n\n"
            "GENERALIZED_SNELL_SINGLE_ORDER_DOMINANCE (predicate 3) tracks the "
            "true breakdown far better on this axis: it stays near +0.63 "
            "through PERIOD-01..06 (0.813-0.817 dominance, including the "
            "refused instance measured directly) and collapses to about "
            "-0.9997 (dominance ~0.0001-0.0002) at PERIOD-07/08, matching the "
            "deflection-angle collapse. It was not built for this axis -- SI "
            "frames it around multi-order/discontinuous surfaces, and the "
            "period sweep is a single harmonic throughout -- so this is "
            "reported as a finding rather than a validated general rule: "
            "*a* predicate margin changes sign at the measured breakdown, but "
            "not the one whose own hard gate refuses there.\n\n"
            "On the duty-cycle axis dominance behaves as SI frames it: DUTY-"
            "01..03 (duty 0.5, 0.2, 0.05) read dominance 0.34, 0.13, 0.02 "
            "(margins -0.32, -0.75, -0.97), all correctly outside validity, "
            "and predicate 2's per-ray hard gate refuses on all three "
            "regardless of duty cycle -- a genuinely discontinuous edge is "
            "equally sharp at any duty cycle, so that gate cannot discriminate "
            "the axis at all; only the windowed dominance measurement can. See "
            "instances/b1_gsl_validity.py's per-instance records for the full "
            "numbers."
        ),
    ),
    sampler=None,
    sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
)

#: Fixed across every instance: a single incidence angle and diffraction order
#: is the primary object of this sweep, not a nuisance parameter, so it is
#: varied on its own dedicated instance rather than crossed with the others.
_WAVELENGTH_M = 5e-7

#: The two MEASURED parameters carry a placeholder here -- 1.0, the "fully
#: inside validity" value -- because their real value depends on running
#: couplers.generalized_snell on the declared surface, which
#: benchmarks/instances/b1_gsl_validity.py does per instance and reports
#: through ExecutionRecord.observed_parameters. verify()'s own _validity()
#: merges observed_parameters over the declared ones before evaluating margins
#: (verification/verifier.py::_validity), so the placeholder is what is
#: declared and the measurement is what decides.
_MEASURED_PLACEHOLDER = {
    "single_order_dominance": 1.0,
    "measured_gradient_smoothness_margin": 1.0,
}


def _expected(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analytic_direction_cosine": grating_direction_cosine(params),
        "why": (
            "the grating equation, exact and independent of both diffractive-"
            "interaction models -- what deflection_angle_worst_error_rad is "
            "measured against"
        ),
    }


def _period_instance(suffix: str, period_m: float) -> Any:
    params = {
        "period_m": period_m,
        "wavelength_m": _WAVELENGTH_M,
        "incidence_angle_rad": 0.0,
        "order": 1,
        "index_incident": 1.0,
        "index_transmitted": 1.0,
        "profile_kind": "blazed_ramp",
        "duty_cycle": 0.5,
        "phase_depth_rad": 0.0,
        "patch_px": 65,
    }
    return _B1_GSL_VALIDITY.instantiate(
        f"B1-GSL-VALIDITY-PERIOD-{suffix}",
        {**params, **_MEASURED_PLACEHOLDER},
        expected=_expected(params),
    )


def _duty_instance(suffix: str, duty_cycle: float) -> Any:
    params = {
        "period_m": 5e-6,
        "wavelength_m": _WAVELENGTH_M,
        "incidence_angle_rad": 0.0,
        "order": 1,
        "index_incident": 1.0,
        "index_transmitted": 1.0,
        "profile_kind": "binary_phase",
        "duty_cycle": duty_cycle,
        "phase_depth_rad": math.pi,
        "patch_px": 65,
    }
    return _B1_GSL_VALIDITY.instantiate(
        f"B1-GSL-VALIDITY-DUTY-{suffix}",
        {**params, **_MEASURED_PLACEHOLDER},
        expected=_expected(params),
    )


B1_GSL_VALIDITY = register(
    _B1_GSL_VALIDITY.with_instances(
        # The primary axis: period shrinking from far into the smooth regime
        # (period >> pitch) toward the sampling floor at fixed 0.2 um pitch.
        _period_instance("01", 1.2e-5),
        _period_instance("02", 3.0e-6),
        _period_instance("03", 1.2e-6),
        _period_instance("04", 1.0e-6),
        _period_instance("05", 9.0e-7),
        # The hard refusal point: worst_local_gradient_smoothness_margin's own
        # finite-difference stencil aliases maximally at 4 samples/period (a
        # phase step of exactly pi between adjacent samples), which is why this
        # sits at 8e-7 rather than somewhere smaller -- see the module docstring.
        _period_instance("06", 8.0e-7),
        # Past the refusal point the ray-level gate SUCCEEDS again -- its own
        # finite-difference stencil aliases back to a small step -- while the
        # actual answer (deflection angle, dominance) is by then garbage. This
        # is the open finding: predicate 2 is non-monotonic and a family that
        # only checked "did it refuse" would call this instance fine.
        _period_instance("07", 7.0e-7),
        _period_instance("08", 6.0e-7),
        # Secondary axis: duty cycle of a binary grating at a period where the
        # blazed ramp is still comfortably smooth, isolating discontinuity from
        # spatial frequency.
        _duty_instance("01", 0.5),
        _duty_instance("02", 0.2),
        _duty_instance("03", 0.05),
        # Secondary axis: incidence angle, blazed ramp, mid-sweep period.
        _B1_GSL_VALIDITY.instantiate(
            "B1-GSL-VALIDITY-OFFAXIS-01",
            {
                **{
                    "period_m": 5.0e-6,
                    "wavelength_m": _WAVELENGTH_M,
                    "incidence_angle_rad": 0.2,
                    "order": 1,
                    "index_incident": 1.0,
                    "index_transmitted": 1.0,
                    "profile_kind": "blazed_ramp",
                    "duty_cycle": 0.5,
                    "phase_depth_rad": 0.0,
                    "patch_px": 65,
                },
                **_MEASURED_PLACEHOLDER,
            },
            expected=_expected(
                {
                    "period_m": 5.0e-6,
                    "wavelength_m": _WAVELENGTH_M,
                    "incidence_angle_rad": 0.2,
                    "order": 1,
                    "index_incident": 1.0,
                    "index_transmitted": 1.0,
                }
            ),
        ),
    )
)
