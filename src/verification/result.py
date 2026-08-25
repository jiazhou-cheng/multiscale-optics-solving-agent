"""What a verification says, and the shapes that stop it saying less than it knows.

CHE-132 (M0.5.3). This is the single structure that makes fixed agent evaluation
and future variable-physics RL two *consumers* rather than two stacks: both call
``verify()`` and differ only in how they reduce its output.

The failure this is designed against has already happened here. ``L2-PSF-01``
reports a ``negative_controls_pass`` boolean, and it reads ``false`` because one
control fires backwards -- but the bundle was easy to plan against as though it
were green, because the interesting information was a paragraph of prose in a
``gate_disposition`` field. **A boolean is where the distinctions collapse.**

So, structurally:

* there is no top-level ``pass`` and no top-level score. Reduction to a verdict
  is the consumer's job, and a consumer that wants one has to say which fields
  it used;
* every reported number is a :class:`Measurement`, which cannot be constructed
  without an uncertainty and a basis for it. This is the anti-fabrication
  mechanism: a value with no error bar is a schema violation, not a pass;
* the seven statuses stay apart. ``unsupported``, ``invalid_configuration``,
  ``out_of_validity``, ``lossy_but_allowed`` and ``blocked`` are five different
  things to do next;
* declared and observed validity are separate fields, and disagreeing is its own
  diagnostic. A run that quietly used an out-of-validity approximation must not
  report as fine because the instance *said* it was inside;
* negative control results are part of every result. A gate a known-wrong twin
  can pass is reported as untrustworthy rather than green.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.execution_record import DevicePrecisionObservation, ResourceCost
from verification.claim_ledger import Oracle, OracleIndependence
from verification.families.schema import ValidityState
from verification.status import VerificationStatus

__all__ = [
    "ContractStatus",
    "ConvergenceReport",
    "Diagnostic",
    "DiagnosticCode",
    "InvariantResult",
    "Measurement",
    "MetricResult",
    "NegativeControlOutcome",
    "NegativeControlResult",
    "PredicateMargin",
    "ProvenanceReport",
    "StochasticReport",
    "UncertaintyBasis",
    "ValidityReport",
    "VerificationResult",
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class UncertaintyBasis(StrEnum):
    """Where an error bar came from. Required, so that "we did not estimate it"
    is a statement rather than a blank."""

    #: Standard error over an ensemble of seeds.
    ENSEMBLE_STANDARD_ERROR = "ensemble_standard_error"
    #: The floating-point floor of the execution precision.
    FLOATING_POINT_FLOOR = "floating_point_floor"
    #: Residual between the two finest rungs of a convergence ladder.
    GRID_CONVERGENCE = "grid_convergence"
    #: Propagated from the oracle's own stated error bound.
    ORACLE_ERROR_BOUND = "oracle_error_bound"
    #: Exact by construction -- a count, an integer, a boolean cast to 0/1.
    EXACT = "exact"
    #: Honestly not estimated. Legitimate, and loud: it is not zero, and a
    #: consumer can refuse to gate on it.
    NOT_ESTIMATED = "not_estimated"


class Measurement(BaseModel):
    """A number that cannot be reported without saying how well it is known.

    ``NOT_ESTIMATED`` requires ``uncertainty`` to be ``None`` rather than zero.
    Zero is a claim of exactness and is the specific fabrication this class
    exists to make impossible to write by accident.
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    uncertainty: float | None
    uncertainty_basis: UncertaintyBasis
    unit: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _uncertainty_matches_its_basis(self) -> Measurement:
        if self.uncertainty_basis is UncertaintyBasis.NOT_ESTIMATED:
            if self.uncertainty is not None:
                raise ValueError(
                    "NOT_ESTIMATED carries no uncertainty value. Reporting one -- "
                    "especially zero -- claims an exactness nothing measured."
                )
        elif self.uncertainty is None:
            raise ValueError(
                f"uncertainty_basis {self.uncertainty_basis.value} promises a number "
                "and none was given. If it was not estimated, say NOT_ESTIMATED."
            )
        elif self.uncertainty < 0.0:
            raise ValueError("an uncertainty cannot be negative")
        return self

    @property
    def is_estimated(self) -> bool:
        return self.uncertainty_basis is not UncertaintyBasis.NOT_ESTIMATED


# ---------------------------------------------------------------------------
# Physics accuracy
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    """One metric, its tolerance, and whether the oracle behind it may decide.

    ``met`` is per metric on purpose. It is not a verdict about the run: a
    result can have a met accuracy metric and an unmet invariant, and collapsing
    those loses the more interesting half.

    ``tolerance_may_gate`` travels with the result rather than being looked up
    later, so a consumer reducing this to a reward cannot accidentally gate on a
    characterization tolerance.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    measured: Measurement
    tolerance: float | None = None
    tolerance_basis: str | None = None
    tolerance_may_gate: bool = False
    met: bool | None = None
    oracle: Oracle = Oracle.NONE
    oracle_independence: OracleIndependence = OracleIndependence.NOT_APPLICABLE
    #: What this metric cannot see, carried from the family's declaration so a
    #: reader of the result does not have to go and find it.
    blind_to: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _a_verdict_needs_a_threshold(self) -> MetricResult:
        if self.met is not None and self.tolerance is None:
            raise ValueError(
                f"{self.metric}: reports met={self.met} with no tolerance to have met. "
                "A verdict without a threshold is an opinion."
            )
        if self.tolerance_may_gate and self.tolerance is None:
            raise ValueError(f"{self.metric}: may_gate with no tolerance")
        return self


class InvariantResult(BaseModel):
    """Something that had to hold regardless of the parameters, and whether it did."""

    model_config = ConfigDict(extra="forbid")

    invariant_id: str
    statement: str
    measured: Measurement
    tolerance: float
    met: bool


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


class PredicateMargin(BaseModel):
    """One validity predicate's normalized signed margin, as evaluated."""

    model_config = ConfigDict(extra="forbid")

    predicate_id: str
    basis: str
    margin: float
    state: ValidityState
    blind_to: list[str] = Field(default_factory=list)


class ValidityReport(BaseModel):
    """Declared and observed, separately, because they can disagree.

    The verifier evaluates the family's predicates *itself*, against the
    parameters the run actually realized. Trusting what the instance declared
    would mean a run that snapped a grid, clamped a ray count or fell back to a
    coarser step reports as inside a domain it left.
    """

    model_config = ConfigDict(extra="forbid")

    declared: ValidityState
    observed: ValidityState
    margins: list[PredicateMargin] = Field(default_factory=list)
    #: Parameters whose realized value differed from the instance's declaration.
    drifted_parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def declaration_holds(self) -> bool:
        return self.declared is self.observed


# ---------------------------------------------------------------------------
# Contract, convergence, stochastic
# ---------------------------------------------------------------------------


class ContractStatus(BaseModel):
    """Boundary conformance and capability facts, as recorded by the run."""

    model_config = ConfigDict(extra="forbid")

    codes: list[str] = Field(default_factory=list)
    capability_intersection_empty: bool = False
    refusal_kind: str | None = None
    refusal_detail: str | None = None
    #: The specific case B0 exists for: every boundary check passed and the
    #: physics is still wrong. ``A1-OPT-03``'s um/nm slip and ``A1-CHX-03``'s
    #: ``kykx`` 2*pi are both ``ok`` contracts with wrong numbers.
    silent_hazard_ids: list[str] = Field(default_factory=list)


class ConvergenceReport(BaseModel):
    """A refinement ladder and what it says, or an explicit absence of one."""

    model_config = ConfigDict(extra="forbid")

    #: The NUMERICAL parameter refined. ``None`` when no ladder was run.
    dimension: str | None = None
    ladder: list[float] = Field(default_factory=list)
    values: list[Measurement] = Field(default_factory=list)
    fitted_exponent: Measurement | None = None
    expected_exponent: float | None = None
    converged: bool | None = None
    note: str = ""

    @model_validator(mode="after")
    def _a_verdict_needs_a_ladder(self) -> ConvergenceReport:
        if self.converged is not None and len(self.ladder) < 2:
            raise ValueError(
                "converged is a statement about a ladder, and one rung is not a ladder"
            )
        if len(self.values) != len(self.ladder):
            raise ValueError("every rung of the ladder needs its measured value")
        return self


class StochasticReport(BaseModel):
    """Ensemble evidence, kept as four separate facts.

    They fail independently: an estimator can be exact in the enumeration limit
    and still biased; unbiased and still converge at the wrong rate; converge
    correctly and have variance that makes the required ray count unreachable.
    """

    model_config = ConfigDict(extra="forbid")

    seeds: list[int] = Field(default_factory=list)
    trials: int = 0
    ensemble_mean: Measurement | None = None
    #: Standard error of the mean over the seeds. The thing one realization
    #: cannot give you.
    ensemble_standard_error: float | None = None
    exactness_limit: Measurement | None = None
    unbiasedness: Measurement | None = None
    fitted_convergence_rate: Measurement | None = None
    variance_by_sampling_density: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_seed_is_not_an_ensemble(self) -> StochasticReport:
        if self.ensemble_standard_error is not None and len(self.seeds) < 2:
            raise ValueError(
                "an ensemble standard error over fewer than two seeds is not an "
                "ensemble statistic. One realization is never an accuracy result."
            )
        return self


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


class NegativeControlOutcome(StrEnum):
    #: The broken twin failed, as it must. The gate is worth something.
    FIRED = "fired"
    #: The broken twin passed. The gate cannot tell right from wrong here.
    DID_NOT_FIRE = "did_not_fire"
    #: It fired in the wrong direction -- the mutation *improved* the metric.
    #: L2-PSF-01's inverted quadrature weight.
    FIRED_BACKWARDS = "fired_backwards"
    #: Declared and not run in this verification.
    NOT_RUN = "not_run"


class NegativeControlResult(BaseModel):
    """Whether a deliberately broken twin behaved as a broken twin should."""

    model_config = ConfigDict(extra="forbid")

    control_id: str
    outcome: NegativeControlOutcome
    target_metric: str
    #: The metric under the mutation, beside the unmutated value, so a reader
    #: can see the separation the control is supposed to produce.
    mutated: Measurement | None = None
    baseline: Measurement | None = None
    note: str = ""

    @property
    def undermines_the_gate(self) -> bool:
        return self.outcome in (
            NegativeControlOutcome.DID_NOT_FIRE,
            NegativeControlOutcome.FIRED_BACKWARDS,
        )


# ---------------------------------------------------------------------------
# Diagnostics and provenance
# ---------------------------------------------------------------------------


class DiagnosticCode(StrEnum):
    """Structured, never free text, so results aggregate.

    Following the repository's existing pattern (``ContractCode``, the precision
    codes): a code, a detail, and where it came from.
    """

    DECLARED_VALIDITY_DISAGREES_WITH_OBSERVED = "declared_validity_disagrees_with_observed"
    PARAMETER_DRIFTED_DURING_EXECUTION = "parameter_drifted_during_execution"
    REQUESTED_PRECISION_NOT_HONOURED = "requested_precision_not_honoured"
    PRECISION_LOSS_NOT_MEASURED = "precision_loss_not_measured"
    NEGATIVE_CONTROL_UNDERMINES_GATE = "negative_control_undermines_gate"
    NEGATIVE_CONTROL_NOT_RUN = "negative_control_not_run"
    ORACLE_CANNOT_DECIDE_CORRECTNESS = "oracle_cannot_decide_correctness"
    NO_UNCERTAINTY_ESTIMATED = "no_uncertainty_estimated"
    STOCHASTIC_EVIDENCE_INCOMPLETE = "stochastic_evidence_incomplete"
    SINGLE_SEED_STOCHASTIC_RUN = "single_seed_stochastic_run"
    CONVERGENCE_NOT_ESTABLISHED = "convergence_not_established"
    INSTANCE_FINGERPRINT_MISMATCH = "instance_fingerprint_mismatch"
    METRIC_MISSING_FROM_RECORD = "metric_missing_from_record"
    RESOURCE_ENVELOPE_EXCEEDED = "resource_envelope_exceeded"
    COST_ESTIMATE_MISSED = "cost_estimate_missed"


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: DiagnosticCode
    detail: str
    #: What the reader should look at: a metric name, a parameter, a node id.
    subject: str | None = None


class ProvenanceReport(BaseModel):
    """Enough to say whether two results are comparable at all."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    instance_fingerprint: str
    family_version: str
    verifier_version: str
    source_commit: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    #: Whether the record's fingerprint matched the instance it was verified
    #: against. False means the two describe different computations.
    fingerprint_matched: bool = True


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """What the evidence says. Not what to do about it.

    There is no ``passed`` and no ``score`` here, and
    ``tests/test_verifier.py::test_the_result_has_no_verdict_and_no_score``
    fails if either appears. Reducing this to a verdict or a reward is the
    consumer's job (M9's ``agent/reward.py``), and forcing the consumer to name
    the fields it used is the point.
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    family_id: str
    family_version: str
    run_id: str
    category: str
    status: VerificationStatus

    physics_accuracy: list[MetricResult] = Field(default_factory=list)
    validity: ValidityReport
    invariant_results: list[InvariantResult] = Field(default_factory=list)
    contract_status: ContractStatus = Field(default_factory=ContractStatus)
    convergence: ConvergenceReport = Field(default_factory=ConvergenceReport)
    stochastic_evidence: StochasticReport = Field(default_factory=StochasticReport)
    device_precision_observation: DevicePrecisionObservation | None = None
    resource_cost: ResourceCost | None = None
    negative_control_results: list[NegativeControlResult] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    provenance: ProvenanceReport

    # -- read-only views, for consumers that want one number -----------------
    #
    # Properties rather than fields, deliberately. They are not serialized, so
    # nothing downstream can persist a collapsed verdict and then be read back
    # as though the verifier had produced it.

    @property
    def gating_metrics(self) -> tuple[MetricResult, ...]:
        """The metrics whose tolerance is allowed to decide anything."""
        return tuple(m for m in self.physics_accuracy if m.tolerance_may_gate)

    @property
    def unmet_gating_metrics(self) -> tuple[MetricResult, ...]:
        return tuple(m for m in self.gating_metrics if m.met is False)

    @property
    def gate_is_trustworthy(self) -> bool:
        """Whether the negative controls support believing a met gate.

        A control that did not fire, or fired backwards, means the gate cannot
        distinguish right from wrong on this instance -- so a met gate is not
        evidence, and this is the flag that says so instead of a green tick.
        """
        return not any(c.undermines_the_gate for c in self.negative_control_results)

    def diagnostic_codes(self) -> tuple[DiagnosticCode, ...]:
        return tuple(d.code for d in self.diagnostics)
