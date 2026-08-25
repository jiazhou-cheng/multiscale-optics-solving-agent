"""What the verifier must say, and what it must never collapse.

CHE-132 (M0.5.3). The concrete failure this suite is written against:
``L2-PSF-01`` reported a ``negative_controls_pass`` boolean that read ``false``
because one control fires backwards, and the bundle was still easy to plan
against as though it were green -- because the interesting information was a
paragraph of prose in a ``gate_disposition`` field. A boolean is where the
distinctions collapse, so most of what is asserted below is that a distinction
survived.

Execution records here are hand-constructed. That is the right dependency
direction and not a stopgap: the verifier states what evidence it needs, and
CHE-113's executor is written to supply it. It also means these tests cannot
accidentally become tests of the executor.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.execution import RunStatus
from core.execution_record import (
    DevicePrecisionObservation,
    ExecutionRecord,
    NodeOutcome,
    NodeRecord,
    Refusal,
    RefusalKind,
    ResourceCost,
)
from core.paths import repository_root
from core.precision import DeviceKind, DType
from verification.claim_ledger import ClaimKind, Oracle, OracleIndependence
from verification.families import (
    BenchmarkCategory,
    BenchmarkFamily,
    ExecutionPolicy,
    FamilyOracle,
    Invariant,
    Metric,
    NegativeControl,
    NegativeControlExpectation,
    NumericalParameter,
    PhysicalParameter,
    SamplerAbsentReason,
    StochasticEvidenceKind,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityState,
)
from verification.families.predicates import fractional_margin
from verification.families.schema import (
    GateDisposition,
    GateStatus,
    ValidityBasis,
    ValidityPredicate,
)
from verification.result import (
    ConvergenceReport,
    DiagnosticCode,
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    StochasticReport,
    UncertaintyBasis,
    VerificationResult,
)
from verification.status import VerificationStatus
from verification.verifier import verify

ROOT = repository_root()


# --------------------------------------------------------------------------- #
# Fixtures: one family, one instance, one record, each varied per test
# --------------------------------------------------------------------------- #

POLICY = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.FLOAT64}),
    max_wall_seconds=60.0,
)

ERROR_METRIC = Metric(
    name="relative_l2",
    description="relative L2 against the closed form",
    unit=None,
    blind_to=("a global phase offset",),
)

GATING = Tolerance(
    metric="relative_l2",
    threshold=1e-3,
    basis="the closed form is exact; 1e-3 rejects a sign error and a 2x scale slip",
    basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
    may_gate=True,
)

RAY_FLOOR = ValidityPredicate(
    predicate_id="RAY_COUNT_FLOOR",
    statement="at least 1000 rays for the quadrature to close",
    basis=ValidityBasis.PER_AXIS_NYQUIST,
    margin=lambda p: fractional_margin(1000.0, float(p["ray_count"])),
)


def make_family(**overrides) -> BenchmarkFamily:
    kwargs = dict(
        family_id="B1-RAY-VERIFY-PROBE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question="does the traced wavefront match the closed form?",
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter("radius_m", "surface radius", unit="m"),
            NumericalParameter("ray_count", "rays traced"),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description="the closed form",
        ),
        metrics=(ERROR_METRIC,),
        execution_policy=POLICY,
        stochastic_policy=StochasticPolicy(
            is_stochastic=False, determinism_reason="a deterministic trace over a fixed grid"
        ),
        validity=(RAY_FLOOR,),
        tolerances=(GATING,),
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="relative_l2",
            observed=1e-5,
            evidence=("tests/test_verifier.py",),
        ),
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
    )
    kwargs.update(overrides)
    return BenchmarkFamily(**kwargs)


def make_instance(family: BenchmarkFamily, **params):
    merged = {"radius_m": 0.1, "ray_count": 100_000}
    merged.update(params)
    return family.instantiate("probe-01", merged)


def make_record(**overrides) -> ExecutionRecord:
    kwargs = dict(
        run_id="run-1",
        status=RunStatus.SUCCEEDED,
        instance_id="probe-01",
        nodes=[
            NodeRecord(
                node_id="trace",
                component="M_RAY_OPTILAND",
                outcome=NodeOutcome.EXECUTED,
            )
        ],
        cost=ResourceCost(wall_seconds=1.0, device="cpu"),
    )
    kwargs.update(overrides)
    return ExecutionRecord(**kwargs)


def measured(value: float, uncertainty: float = 1e-9) -> Measurement:
    return Measurement(
        value=value,
        uncertainty=uncertainty,
        uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
    )


# --------------------------------------------------------------------------- #
# The structural guarantees
# --------------------------------------------------------------------------- #


def test_the_result_has_no_verdict_and_no_score() -> None:
    """The mechanical form of "do not collapse the distinctions".

    A top-level boolean or scalar would be the field every consumer reads and
    nobody looks past. Reduction to a verdict is the consumer's job, and a
    consumer that wants one has to name the fields it used.
    """
    fields = VerificationResult.model_fields
    banned = {"passed", "pass", "success", "ok", "score", "reward", "grade", "value"}
    offending = sorted(set(fields) & banned)
    assert not offending, f"VerificationResult grew a verdict field: {offending}"

    for name, info in fields.items():
        annotation = str(info.annotation)
        assert annotation not in ("bool", "float", "int"), (
            f"{name}: a bare {annotation} at the top level of a VerificationResult is a "
            "collapsed verdict waiting to be read as one"
        )


def test_verification_imports_nothing_from_the_agent_package() -> None:
    """The mechanical guarantee behind "the verifier is not a reward function".

    Parsed rather than imported, so the check does not depend on which modules a
    test happens to have loaded.
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "src/verification").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "agent" or name.startswith("agent."):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {name}")
    assert not offenders, (
        "src/verification/ must not import src/agent/. The verifier measures physics; "
        "turning a measurement into a reward is M9's job and lives on the other side "
        "of this line:\n  " + "\n  ".join(offenders)
    )


def test_the_verifier_reads_no_committed_record_as_truth() -> None:
    """No file reads anywhere in the verifier's own module.

    The failure being designed against is the one CHE-103 found: tests that
    asserted on recorded files and passed while measuring history. A record is
    provenance; an oracle is a closed form.
    """
    source = (ROOT / "src/verification/verifier.py").read_text(encoding="utf-8")
    for forbidden in ("open(", "read_text", "json.load", "np.load", "Path("):
        assert forbidden not in source, (
            f"verifier.py contains {forbidden!r}. It reads the family, the instance and "
            "the record it is handed, and nothing else."
        )


def test_every_reported_number_carries_an_uncertainty() -> None:
    with pytest.raises(ValueError, match="promises a number and none was given"):
        Measurement(
            value=2.2e-3,
            uncertainty=None,
            uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
        )


def test_not_estimated_cannot_be_dressed_up_as_zero() -> None:
    """Zero is a claim of exactness. "We did not measure it" is not that claim."""
    with pytest.raises(ValueError, match="claims an exactness nothing measured"):
        Measurement(value=1.0, uncertainty=0.0, uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED)


def test_a_metric_verdict_requires_a_threshold_to_have_met() -> None:
    from verification.result import MetricResult

    with pytest.raises(ValueError, match="A verdict without a threshold is an opinion"):
        MetricResult(metric="relative_l2", measured=measured(1e-4), met=True)


# --------------------------------------------------------------------------- #
# All seven statuses are individually reachable
# --------------------------------------------------------------------------- #


def test_status_ok() -> None:
    family = make_family()
    instance = make_instance(family)
    result = verify(
        family, instance, make_record(), measurements={"relative_l2": measured(1e-5)}
    )
    assert result.status is VerificationStatus.OK


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (RefusalKind.UNSUPPORTED_CAPABILITY, VerificationStatus.UNSUPPORTED),
        (RefusalKind.INVALID_CONFIGURATION, VerificationStatus.INVALID_CONFIGURATION),
        (RefusalKind.OUT_OF_DECLARED_VALIDITY, VerificationStatus.OUT_OF_VALIDITY),
        (RefusalKind.RESOURCE_GUARD, VerificationStatus.BLOCKED),
        (RefusalKind.UNVERIFIED_DERIVATIVE, VerificationStatus.BLOCKED),
        (RefusalKind.MISSING_EDGE_DECLARATION, VerificationStatus.INVALID_CONFIGURATION),
    ],
)
def test_each_refusal_kind_maps_to_its_own_status(
    kind: RefusalKind, expected: VerificationStatus
) -> None:
    """These must not collapse into one generic failure.

    "The solver cannot do complex128", "the graph is missing an edge
    declaration" and "the resource guard stopped it" call for three different
    responses, and a caller handed one string can act on none of them.
    """
    family = make_family()
    instance = make_instance(family)
    record = make_record(
        status=RunStatus.FAILED,
        refusal=Refusal(kind=kind, detail="probe", declaration="d"),
        nodes=[
            NodeRecord(node_id="trace", component="M_RAY_OPTILAND", outcome=NodeOutcome.REFUSED)
        ],
    )
    result = verify(family, instance, record)
    assert result.status is expected
    assert result.contract_status.refusal_kind == kind.value


def test_status_out_of_validity_from_the_verifier_own_predicates() -> None:
    """Not from what the run claimed. The instance never asserted it was outside;
    the family's predicate, evaluated against the realized ray count, says so."""
    family = make_family()
    instance = make_instance(family, ray_count=100)
    result = verify(
        family, instance, make_record(), measurements={"relative_l2": measured(1e-5)}
    )
    assert result.status is VerificationStatus.OUT_OF_VALIDITY
    assert result.validity.observed is ValidityState.FAR_OUTSIDE


def test_status_lossy_but_allowed() -> None:
    """Chromatix's unconditional complex64 cast is the motivating case: the run
    succeeded, the loss is real, and it is neither OK nor a failure."""
    family = make_family()
    instance = make_instance(family)
    record = make_record(
        nodes=[
            NodeRecord(
                node_id="propagate",
                component="M_WAVE_CHROMATIX",
                outcome=NodeOutcome.EXECUTED_LOSSY,
            )
        ],
        device_precision=DevicePrecisionObservation(
            requested_device="cpu",
            actual_device="cpu",
            requested_dtype="complex128",
            actual_dtype="complex64",
            measured_loss_relative=2.5e-5,
            measured_loss_basis="float64 ASM reference at z = 40 um",
        ),
    )
    result = verify(family, instance, record, measurements={"relative_l2": measured(1e-5)})
    assert result.status is VerificationStatus.LOSSY_BUT_ALLOWED
    assert result.device_precision_observation is not None
    assert result.device_precision_observation.measured_loss_relative == 2.5e-5


def test_status_unconverged() -> None:
    family = make_family()
    instance = make_instance(family)
    convergence = ConvergenceReport(
        dimension="ray_count",
        ladder=[1e3, 1e4, 1e5],
        values=[measured(3e-3), measured(2e-3), measured(1.5e-3)],
        converged=False,
        note="the residual is still moving by 25% per decade",
    )
    result = verify(
        family,
        instance,
        make_record(),
        measurements={"relative_l2": measured(1.5e-3)},
        convergence=convergence,
    )
    assert result.status is VerificationStatus.UNCONVERGED


def test_all_seven_statuses_are_covered_by_this_file() -> None:
    """A status nothing has ever produced cannot be trusted to appear when it
    matters -- the same argument the agent suite's taxonomy makes about its
    eight codes."""
    source = Path(__file__).read_text(encoding="utf-8")
    for status in VerificationStatus:
        assert f"VerificationStatus.{status.name}" in source, (
            f"{status.value} is declared and unreached by any test here"
        )


# --------------------------------------------------------------------------- #
# Declared versus observed validity
# --------------------------------------------------------------------------- #


def test_declared_and_observed_validity_are_separate_and_disagreeing_is_a_diagnostic() -> None:
    """A run that silently used an out-of-validity approximation must not report
    as fine because the instance said it was inside."""
    family = make_family()
    instance = make_instance(family, ray_count=100_000)
    assert instance.validity_status is ValidityState.INSIDE

    record = make_record(observed_parameters={"ray_count": 100})
    result = verify(family, instance, record, measurements={"relative_l2": measured(1e-5)})

    assert result.validity.declared is ValidityState.INSIDE
    assert result.validity.observed is ValidityState.FAR_OUTSIDE
    assert not result.validity.declaration_holds
    assert DiagnosticCode.DECLARED_VALIDITY_DISAGREES_WITH_OBSERVED in result.diagnostic_codes()
    assert DiagnosticCode.PARAMETER_DRIFTED_DURING_EXECUTION in result.diagnostic_codes()
    assert result.validity.drifted_parameters["ray_count"] == {
        "declared": 100_000,
        "observed": 100,
    }


def test_every_predicate_reports_its_signed_margin_and_its_blind_spot() -> None:
    family = make_family()
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})
    (margin,) = result.validity.margins
    assert margin.predicate_id == "RAY_COUNT_FLOOR"
    assert margin.margin == pytest.approx(1.0 - 1000.0 / 100_000.0)
    assert margin.state is ValidityState.INSIDE


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #


BACKWARDS = NegativeControl(
    control_id="inverted-quadrature-weight",
    description="invert the radial trapezoid weight and require the residual to grow",
    mutation="w -> 1/w on every ray before the coherent sum",
    target_metric="relative_l2",
    expectation=NegativeControlExpectation.KNOWN_FIRES_BACKWARDS,
    caveat=(
        "L2-PSF-01 measured this control improving the residual rather than degrading "
        "it, which is unexplained; CHE-117 owns the attribution"
    ),
)


def test_a_declared_control_that_was_not_run_is_reported_as_not_run() -> None:
    """An omitted control reads exactly like a control that passed."""
    family = make_family(negative_controls=(BACKWARDS,))
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})

    (control,) = result.negative_control_results
    assert control.outcome is NegativeControlOutcome.NOT_RUN
    assert DiagnosticCode.NEGATIVE_CONTROL_NOT_RUN in result.diagnostic_codes()


def test_a_control_that_fires_backwards_makes_the_gate_untrustworthy() -> None:
    """The L2-PSF-01 case, and the whole reason this field exists. A met gate
    whose broken twin also passes is not evidence."""
    family = make_family(negative_controls=(BACKWARDS,))
    instance = make_instance(family)
    outcome = NegativeControlResult(
        control_id="inverted-quadrature-weight",
        outcome=NegativeControlOutcome.FIRED_BACKWARDS,
        target_metric="relative_l2",
        baseline=measured(2.2e-3),
        mutated=measured(1.9e-3),
        note="the mutation improved the residual",
    )
    result = verify(
        family,
        instance,
        make_record(),
        measurements={"relative_l2": measured(5e-4)},
        negative_controls={"inverted-quadrature-weight": outcome},
    )
    assert result.physics_accuracy[0].met is True, "the metric itself is inside the tolerance"
    assert not result.gate_is_trustworthy, (
        "and the gate is still not evidence, because its broken twin came out better"
    )
    assert DiagnosticCode.NEGATIVE_CONTROL_UNDERMINES_GATE in result.diagnostic_codes()


def test_a_control_that_was_never_run_does_not_leave_the_gate_trustworthy() -> None:
    """Unexercised is not the same as passed, and it is not evidence either.

    A result with four declared controls and none exercised reporting a
    trustworthy gate would be exactly the green tick this structure refuses.
    """
    family = make_family(negative_controls=(BACKWARDS,))
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})

    assert result.negative_control_results[0].outcome is NegativeControlOutcome.NOT_RUN
    assert not result.gate_is_trustworthy
    assert len(result.untrustworthy_controls) == 1


def test_a_family_with_no_controls_at_all_is_not_trustworthy() -> None:
    """There is nothing to be trusted on."""
    family = make_family()
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})
    assert result.negative_control_results == []
    assert not result.gate_is_trustworthy


def test_a_control_that_fires_leaves_the_gate_trustworthy() -> None:
    family = make_family(negative_controls=(BACKWARDS,))
    instance = make_instance(family)
    outcome = NegativeControlResult(
        control_id="inverted-quadrature-weight",
        outcome=NegativeControlOutcome.FIRED,
        target_metric="relative_l2",
        baseline=measured(5e-4),
        mutated=measured(4e-2),
    )
    result = verify(
        family,
        instance,
        make_record(),
        measurements={"relative_l2": measured(5e-4)},
        negative_controls={"inverted-quadrature-weight": outcome},
    )
    assert result.gate_is_trustworthy


# --------------------------------------------------------------------------- #
# Oracle independence travels with the result
# --------------------------------------------------------------------------- #


def test_a_characterization_family_reports_that_nothing_here_gates() -> None:
    family = make_family(
        family_id="B4-DUALROUTE-PROBE",
        category=BenchmarkCategory.B4,
        oracle=FamilyOracle(
            kind=Oracle.CROSS_ROUTE,
            independence=OracleIndependence.SHARES_CODE,
            description="Optiland FFTPSF against Optiland HuygensPSF",
        ),
        tolerances=(
            Tolerance(
                metric="relative_l2",
                threshold=1e-3,
                basis="the two routes agreed to this on the Cooke triplet",
                basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
                may_gate=False,
            ),
        ),
        gate_disposition=None,
    )
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})

    assert result.gating_metrics == ()
    assert result.physics_accuracy[0].tolerance_may_gate is False
    assert result.physics_accuracy[0].met is True, (
        "the comparison still reports whether it landed inside the number -- what it "
        "does not do is let that decide anything"
    )
    assert DiagnosticCode.ORACLE_CANNOT_DECIDE_CORRECTNESS in result.diagnostic_codes()


# --------------------------------------------------------------------------- #
# Stochastic, convergence, cost, precision
# --------------------------------------------------------------------------- #


def test_one_seed_on_a_stochastic_family_is_a_diagnostic_not_a_result() -> None:
    family = make_family(
        stochastic_policy=StochasticPolicy(
            is_stochastic=True,
            required_evidence=(
                StochasticEvidenceKind.UNBIASEDNESS,
                StochasticEvidenceKind.VARIANCE_CHARACTERIZATION,
            ),
            minimum_seeds=8,
        ),
        tolerances=(),
        gate_disposition=None,
    )
    instance = family.instantiate("probe-01", {"radius_m": 0.1, "ray_count": 100_000}, seed=7)
    result = verify(
        family,
        instance,
        make_record(seeds=[7]),
        measurements={"relative_l2": measured(1e-3)},
        stochastic=StochasticReport(seeds=[7], trials=1),
    )
    codes = result.diagnostic_codes()
    assert DiagnosticCode.SINGLE_SEED_STOCHASTIC_RUN in codes
    assert DiagnosticCode.STOCHASTIC_EVIDENCE_INCOMPLETE in codes


def test_an_ensemble_standard_error_needs_an_ensemble() -> None:
    with pytest.raises(ValueError, match="One realization is never an accuracy result"):
        StochasticReport(seeds=[7], trials=1, ensemble_standard_error=1e-4)


def test_a_refinement_dimension_with_no_ladder_is_reported() -> None:
    family = make_family(
        parameters=(
            PhysicalParameter("radius_m", "surface radius", unit="m"),
            NumericalParameter("ray_count", "rays traced", refines_toward=1),
        )
    )
    instance = make_instance(family)
    result = verify(family, instance, make_record(), measurements={"relative_l2": measured(1e-5)})
    assert DiagnosticCode.CONVERGENCE_NOT_ESTABLISHED in result.diagnostic_codes()


def test_a_converged_verdict_needs_more_than_one_rung() -> None:
    with pytest.raises(ValueError, match="one rung is not a ladder"):
        ConvergenceReport(ladder=[1e5], values=[measured(1e-5)], converged=True)


def test_an_unhonoured_precision_request_without_a_measured_loss_is_two_diagnostics() -> None:
    """"A warning is not a number" -- the loss must be measured or its absence
    must be said out loud."""
    family = make_family()
    instance = make_instance(family)
    record = make_record(
        device_precision=DevicePrecisionObservation(
            requested_device="cpu",
            actual_device="cpu",
            requested_dtype="complex128",
            actual_dtype="complex64",
        )
    )
    result = verify(family, instance, record, measurements={"relative_l2": measured(1e-5)})
    codes = result.diagnostic_codes()
    assert DiagnosticCode.REQUESTED_PRECISION_NOT_HONOURED in codes
    assert DiagnosticCode.PRECISION_LOSS_NOT_MEASURED in codes


def test_the_resource_envelope_is_part_of_the_family_not_the_runner() -> None:
    family = make_family()
    instance = make_instance(family)
    record = make_record(cost=ResourceCost(wall_seconds=3600.0, device="cpu"))
    result = verify(family, instance, record, measurements={"relative_l2": measured(1e-5)})
    assert DiagnosticCode.RESOURCE_ENVELOPE_EXCEEDED in result.diagnostic_codes()


def test_a_metric_the_family_declares_and_the_run_did_not_measure_is_reported() -> None:
    family = make_family()
    instance = make_instance(family)
    result = verify(family, instance, make_record())
    assert DiagnosticCode.METRIC_MISSING_FROM_RECORD in result.diagnostic_codes()
    assert result.physics_accuracy == []


def test_a_number_with_no_uncertainty_estimate_is_reported_as_such() -> None:
    family = make_family()
    instance = make_instance(family)
    unestimated = Measurement(
        value=1e-5, uncertainty=None, uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED
    )
    result = verify(family, instance, make_record(), measurements={"relative_l2": unestimated})
    assert DiagnosticCode.NO_UNCERTAINTY_ESTIMATED in result.diagnostic_codes()


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_a_record_produced_for_another_instance_is_caught() -> None:
    family = make_family()
    instance = make_instance(family)
    record = make_record(instance_fingerprint="0" * 64)
    result = verify(family, instance, record, measurements={"relative_l2": measured(1e-5)})
    assert not result.provenance.fingerprint_matched
    assert DiagnosticCode.INSTANCE_FINGERPRINT_MISMATCH in result.diagnostic_codes()


def test_an_instance_from_another_family_version_is_refused_outright() -> None:
    """Not a diagnostic: a version bump invalidates the instance, and verifying
    it anyway would produce a result that looks comparable and is not."""
    family = make_family()
    instance = make_instance(family)
    bumped = make_family(family_version="2.0.0")
    with pytest.raises(ValueError, match="version bump invalidates the instance"):
        verify(bumped, instance, make_record())


def test_invariants_are_reported_beside_the_accuracy_metrics() -> None:
    family = make_family(
        invariants=(
            Invariant(
                invariant_id="ENERGY_CLOSES",
                statement="total power is conserved across the transition",
                metric="relative_l2",
                tolerance=Tolerance(
                    metric="relative_l2",
                    threshold=1e-12,
                    basis="energy conservation closes to float64 round-off",
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                ),
            ),
        )
    )
    instance = make_instance(family)
    result = verify(
        family,
        instance,
        make_record(),
        measurements={"relative_l2": measured(1e-5)},
        invariants={"ENERGY_CLOSES": measured(3e-14)},
    )
    (invariant,) = result.invariant_results
    assert invariant.met is True
    assert invariant.invariant_id == "ENERGY_CLOSES"
