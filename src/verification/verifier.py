"""``verify(family, instance, execution_record) -> VerificationResult``.

CHE-132 (M0.5.3). It measures. It does not decide task success and it does not
produce a reward -- ``tests/test_verifier.py`` pins that ``src/verification/``
imports nothing from ``src/agent/``, which is the mechanical guarantee behind
"the verifier is not inherently a reward function".

Three properties are worth stating before the code, because each is a rule about
what this function is *not* allowed to do.

**It reads three things: the family, the instance, and the record.** Never a
committed result file. Records are provenance, not oracles -- the failure being
designed against is the one CHE-103 found, where nineteen tests asserted on
recorded files and passed while measuring history.

**It re-evaluates validity itself.** The instance says where it *declared*
itself to be; the record says which parameters the run actually realized. The
verifier evaluates the family's predicates against the realized values, and a
disagreement is its own diagnostic. Trusting the declaration would let a run
that snapped a grid or clamped a ray count report as inside a domain it left.

**It reports negative controls whether or not they were run.** A gate whose
known-wrong twin was never executed is a gate with no evidence that it can tell
right from wrong, and that is a different state from a gate that has such
evidence and is green.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.execution import RunStatus
from core.execution_record import ExecutionRecord, NodeOutcome, RefusalKind
from verification.families.schema import (
    BenchmarkFamily,
    BenchmarkInstance,
    NegativeControlExpectation,
)
from verification.result import (
    ContractStatus,
    ConvergenceReport,
    Diagnostic,
    DiagnosticCode,
    InvariantResult,
    Measurement,
    MetricResult,
    NegativeControlOutcome,
    NegativeControlResult,
    PredicateMargin,
    ProvenanceReport,
    StochasticReport,
    UncertaintyBasis,
    ValidityReport,
    VerificationResult,
)
from verification.status import VerificationStatus

__all__ = ["VERIFIER_VERSION", "verify"]

#: Bumped when the verifier's own logic changes what a result means. Carried in
#: every result's provenance so two results can be told apart by *how* they were
#: judged, not only by what was run.
VERIFIER_VERSION = "1.0.0"


_REFUSAL_TO_STATUS = {
    RefusalKind.UNSUPPORTED_CAPABILITY: VerificationStatus.UNSUPPORTED,
    RefusalKind.INVALID_CONFIGURATION: VerificationStatus.INVALID_CONFIGURATION,
    RefusalKind.OUT_OF_DECLARED_VALIDITY: VerificationStatus.OUT_OF_VALIDITY,
    RefusalKind.RESOURCE_GUARD: VerificationStatus.BLOCKED,
    RefusalKind.UNVERIFIED_DERIVATIVE: VerificationStatus.BLOCKED,
    RefusalKind.MISSING_EDGE_DECLARATION: VerificationStatus.INVALID_CONFIGURATION,
}


def _first_refusal(record: ExecutionRecord):  # type: ignore[no-untyped-def]
    """The refusal that stopped the run, run-level first then node-level.

    First rather than worst: a downstream node refusing because its input never
    arrived describes the consequence, and the caller needs the cause.
    """
    if record.refusal is not None:
        return record.refusal
    for node in record.nodes:
        if node.refusal is not None:
            return node.refusal
    return None


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def _validity(
    family: BenchmarkFamily,
    instance: BenchmarkInstance,
    record: ExecutionRecord,
) -> tuple[ValidityReport, list[Diagnostic]]:
    """Re-evaluate the family's predicates against what actually ran."""
    realized: dict[str, Any] = dict(instance.parameters)
    drifted: dict[str, Any] = {}
    for name, value in record.observed_parameters.items():
        if name in realized and realized[name] != value:
            drifted[name] = {"declared": realized[name], "observed": value}
        realized[name] = value

    observed_state, margins = family.evaluate_validity(realized)
    margin_reports = [
        PredicateMargin(
            predicate_id=predicate.predicate_id,
            basis=str(predicate.basis),
            margin=margins[predicate.predicate_id],
            state=predicate.state(realized),
            blind_to=list(predicate.blind_to),
        )
        for predicate in family.validity
    ]

    diagnostics: list[Diagnostic] = []
    if drifted:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.PARAMETER_DRIFTED_DURING_EXECUTION,
                detail=(
                    "the run realized parameter values other than the instance's: "
                    + ", ".join(sorted(drifted))
                ),
                subject=",".join(sorted(drifted)),
            )
        )
    if observed_state is not instance.validity_status:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.DECLARED_VALIDITY_DISAGREES_WITH_OBSERVED,
                detail=(
                    f"instance declared {instance.validity_status.value}; the family's "
                    f"predicates evaluated against the realized parameters give "
                    f"{observed_state.value}"
                ),
            )
        )
    return (
        ValidityReport(
            declared=instance.validity_status,
            observed=observed_state,
            margins=margin_reports,
            drifted_parameters=drifted,
        ),
        diagnostics,
    )


# ---------------------------------------------------------------------------
# Physics accuracy
# ---------------------------------------------------------------------------


def _physics_accuracy(
    family: BenchmarkFamily,
    measurements: Mapping[str, Measurement],
) -> tuple[list[MetricResult], list[Diagnostic]]:
    results: list[MetricResult] = []
    diagnostics: list[Diagnostic] = []

    for metric in family.metrics:
        measured = measurements.get(metric.name)
        if measured is None:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.METRIC_MISSING_FROM_RECORD,
                    detail=f"the family declares {metric.name} and the run measured none",
                    subject=metric.name,
                )
            )
            continue
        if not measured.is_estimated:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.NO_UNCERTAINTY_ESTIMATED,
                    detail=(
                        f"{metric.name} is reported without an uncertainty estimate, so "
                        "nothing here bounds how far the value could be from the truth"
                    ),
                    subject=metric.name,
                )
            )

        tolerance = family.tolerance_for(metric.name)
        met: bool | None = None
        if tolerance is not None:
            met = measured.value <= tolerance.threshold

        results.append(
            MetricResult(
                metric=metric.name,
                measured=measured,
                tolerance=tolerance.threshold if tolerance else None,
                tolerance_basis=tolerance.basis if tolerance else None,
                tolerance_may_gate=bool(tolerance and tolerance.may_gate),
                met=met,
                oracle=family.oracle.kind,
                oracle_independence=family.oracle.independence,
                blind_to=list(metric.blind_to),
            )
        )

    if not family.oracle.may_decide_correctness:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.ORACLE_CANNOT_DECIDE_CORRECTNESS,
                detail=(
                    f"oracle {family.oracle.kind.value} with independence "
                    f"{family.oracle.independence.value}: every number here is "
                    "characterization, not a correctness verdict"
                ),
            )
        )
    return results, diagnostics


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _negative_controls(
    family: BenchmarkFamily,
    outcomes: Mapping[str, NegativeControlResult],
) -> tuple[list[NegativeControlResult], list[Diagnostic]]:
    """Every declared control appears, run or not.

    A control that was never executed is reported as ``NOT_RUN`` rather than
    omitted, because an omitted control reads exactly like a control that
    passed.
    """
    results: list[NegativeControlResult] = []
    diagnostics: list[Diagnostic] = []

    for control in family.negative_controls:
        result = outcomes.get(control.control_id)
        if result is None:
            expected_backwards = (
                control.expectation is NegativeControlExpectation.KNOWN_FIRES_BACKWARDS
            )
            result = NegativeControlResult(
                control_id=control.control_id,
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric=control.target_metric,
                note=(
                    control.caveat
                    if control.caveat
                    else "declared by the family and not exercised in this run"
                ),
            )
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.NEGATIVE_CONTROL_NOT_RUN,
                    detail=(
                        f"{control.control_id} was not exercised"
                        + (
                            " -- and it is the control already known to fire backwards"
                            if expected_backwards
                            else ""
                        )
                    ),
                    subject=control.control_id,
                )
            )
        results.append(result)
        if result.undermines_the_gate:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.NEGATIVE_CONTROL_UNDERMINES_GATE,
                    detail=(
                        f"{control.control_id} came out {result.outcome.value}: the gate "
                        f"on {control.target_metric} cannot separate this defect from a "
                        "correct run, so a met tolerance is not evidence here"
                    ),
                    subject=control.control_id,
                )
            )
    return results, diagnostics


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _status(
    family: BenchmarkFamily,
    record: ExecutionRecord,
    validity: ValidityReport,
    convergence: ConvergenceReport,
) -> VerificationStatus:
    """The one field that says what kind of outcome this was.

    Ordered by what a caller must deal with first. A refusal outranks a validity
    finding because nothing was measured; validity outranks convergence because
    a converged number outside its approximation's domain is a well-resolved
    wrong answer.
    """
    refusal = _first_refusal(record)
    if refusal is not None:
        return _REFUSAL_TO_STATUS[refusal.kind]

    if record.status is RunStatus.FAILED and not record.nodes:
        return VerificationStatus.INVALID_CONFIGURATION

    if not validity.observed.is_inside:
        return VerificationStatus.OUT_OF_VALIDITY

    if any(node.outcome is NodeOutcome.EXECUTED_LOSSY for node in record.nodes):
        return VerificationStatus.LOSSY_BUT_ALLOWED

    if convergence.converged is False:
        return VerificationStatus.UNCONVERGED

    return VerificationStatus.OK


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify(
    family: BenchmarkFamily,
    instance: BenchmarkInstance,
    record: ExecutionRecord,
    *,
    measurements: Mapping[str, Measurement] | None = None,
    invariants: Mapping[str, Measurement] | None = None,
    negative_controls: Mapping[str, NegativeControlResult] | None = None,
    convergence: ConvergenceReport | None = None,
    stochastic: StochasticReport | None = None,
    silent_hazard_ids: tuple[str, ...] = (),
) -> VerificationResult:
    """Measure what the record says about the instance, under the family's rules.

    ``measurements`` and friends are the metric values the run produced. They
    are passed in rather than recomputed here because the verifier's job is to
    *interpret* evidence against the family's declarations, and recomputing
    physics inside it would make it a second solver.

    What this function decides: which status the outcome has, whether each
    metric met a tolerance it is allowed to be judged by, whether the declared
    validity survived contact with the run, and whether the negative controls
    leave the gate worth anything. What it does not decide: whether the task
    succeeded, or what any of it is worth.
    """
    if instance.family_id != family.family_id:
        raise ValueError(
            f"instance {instance.instance_id} belongs to {instance.family_id}, not "
            f"{family.family_id}"
        )
    if instance.family_version != family.family_version:
        raise ValueError(
            f"instance {instance.instance_id} was built against family_version "
            f"{instance.family_version}; the family is at {family.family_version}. A "
            "version bump invalidates the instance rather than being ignored."
        )

    measurements = dict(measurements or {})
    diagnostics: list[Diagnostic] = []

    validity, validity_diagnostics = _validity(family, instance, record)
    diagnostics += validity_diagnostics

    metric_results, metric_diagnostics = _physics_accuracy(family, measurements)
    diagnostics += metric_diagnostics

    invariant_results = []
    for invariant in family.invariants:
        measured = (invariants or {}).get(invariant.invariant_id)
        if measured is None:
            continue
        invariant_results.append(
            InvariantResult(
                invariant_id=invariant.invariant_id,
                statement=invariant.statement,
                measured=measured,
                tolerance=invariant.tolerance.threshold,
                met=measured.value <= invariant.tolerance.threshold,
            )
        )

    control_results, control_diagnostics = _negative_controls(family, negative_controls or {})
    diagnostics += control_diagnostics

    convergence = convergence or ConvergenceReport()
    if convergence.converged is None and family.refinement_dimensions:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.CONVERGENCE_NOT_ESTABLISHED,
                detail=(
                    "the family declares a refinement dimension and this run carries no "
                    "ladder, so the value is a single point rather than a converged one"
                ),
            )
        )

    stochastic = stochastic or StochasticReport(seeds=list(record.seeds))
    if family.stochastic_policy.is_stochastic:
        if len(stochastic.seeds) < family.stochastic_policy.minimum_seeds:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.SINGLE_SEED_STOCHASTIC_RUN
                    if len(stochastic.seeds) < 2
                    else DiagnosticCode.STOCHASTIC_EVIDENCE_INCOMPLETE,
                    detail=(
                        f"{len(stochastic.seeds)} seed(s) against a declared minimum of "
                        f"{family.stochastic_policy.minimum_seeds}; an accuracy claim "
                        "from this is a claim about one realization"
                    ),
                )
            )
        missing = [
            kind.value
            for kind in family.stochastic_policy.required_evidence
            if not getattr(stochastic, _STOCHASTIC_FIELD[kind.value], None)
        ]
        if missing:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.STOCHASTIC_EVIDENCE_INCOMPLETE,
                    detail=f"required stochastic evidence not supplied: {', '.join(missing)}",
                )
            )

    device_precision = record.device_precision
    if device_precision is not None and not device_precision.honoured:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.REQUESTED_PRECISION_NOT_HONOURED,
                detail=(
                    f"requested {device_precision.requested_dtype} on "
                    f"{device_precision.requested_device}; the run computed in "
                    f"{device_precision.actual_dtype} on "
                    f"{device_precision.actual_device}"
                ),
            )
        )
        if device_precision.measured_loss_relative is None:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.PRECISION_LOSS_NOT_MEASURED,
                    detail=(
                        "a downcast happened and nothing measured what it cost. A "
                        "warning is not a number."
                    ),
                )
            )

    cost = record.cost
    if cost is not None:
        envelope = family.execution_policy
        if envelope.max_wall_seconds is not None and cost.wall_seconds > envelope.max_wall_seconds:
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.RESOURCE_ENVELOPE_EXCEEDED,
                    detail=(
                        f"{cost.wall_seconds:.1f} s against a declared envelope of "
                        f"{envelope.max_wall_seconds:.1f} s"
                    ),
                )
            )
        ratio = cost.estimate_ratio
        if ratio is not None and (ratio > 2.0 or ratio < 0.5):
            diagnostics.append(
                Diagnostic(
                    code=DiagnosticCode.COST_ESTIMATE_MISSED,
                    detail=f"actual/predicted wall time = {ratio:.2f}",
                )
            )

    fingerprint_matched = (
        record.instance_fingerprint is None or record.instance_fingerprint == instance.fingerprint
    )
    if not fingerprint_matched:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.INSTANCE_FINGERPRINT_MISMATCH,
                detail=(
                    f"the record was produced for {record.instance_fingerprint} and is "
                    f"being verified against {instance.fingerprint}; these are different "
                    "computations"
                ),
            )
        )

    refusal = _first_refusal(record)
    contract = ContractStatus(
        codes=list(record.contract_codes),
        capability_intersection_empty=(
            refusal is not None and refusal.kind is RefusalKind.UNSUPPORTED_CAPABILITY
        ),
        refusal_kind=refusal.kind.value if refusal is not None else None,
        refusal_detail=refusal.detail if refusal is not None else None,
        silent_hazard_ids=list(silent_hazard_ids),
    )

    return VerificationResult(
        instance_id=instance.instance_id,
        family_id=family.family_id,
        family_version=family.family_version,
        run_id=record.run_id,
        category=str(family.category),
        status=_status(family, record, validity, convergence),
        physics_accuracy=metric_results,
        validity=validity,
        invariant_results=invariant_results,
        contract_status=contract,
        convergence=convergence,
        stochastic_evidence=stochastic,
        device_precision_observation=device_precision,
        resource_cost=cost,
        negative_control_results=control_results,
        diagnostics=diagnostics,
        provenance=ProvenanceReport(
            run_id=record.run_id,
            instance_fingerprint=instance.fingerprint,
            family_version=family.family_version,
            verifier_version=VERIFIER_VERSION,
            source_commit=record.provenance.get("source_commit"),
            package_versions=dict(record.provenance.get("packages", {})),
            fingerprint_matched=fingerprint_matched,
        ),
    )


_STOCHASTIC_FIELD = {
    "exactness_limit": "exactness_limit",
    "unbiasedness": "unbiasedness",
    "convergence_exponent": "fitted_convergence_rate",
    "variance_characterization": "variance_by_sampling_density",
}


def exact(value: float, *, unit: str | None = None, note: str = "") -> Measurement:
    """A number that is exact by construction -- a count, an index, a flag."""
    return Measurement(
        value=value,
        uncertainty=0.0,
        uncertainty_basis=UncertaintyBasis.EXACT,
        unit=unit,
        note=note,
    )


__all__ += ["exact"]
