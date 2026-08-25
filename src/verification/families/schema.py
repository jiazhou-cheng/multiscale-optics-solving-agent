"""BenchmarkFamily, BenchmarkInstance, and the rules that cannot be opted out of.

CHE-131 (M0.5.2). Every benchmark in this repository before this file was
inseparable from one parameter set, which is what made the suite unable to serve
either downstream direction: a fixed benchmark cannot be resampled near a
validity boundary, and a variable-physics environment has nothing to sample from.
A family is the separation -- the physical question, its oracle, its validity
domain and its tolerances in one object, and the parameters as a declared space
rather than a hard-coded call.

This is **not** a from-scratch design. ``verification/claim_ledger.py`` already
carried :class:`~verification.claim_ledger.Oracle`,
:class:`~verification.claim_ledger.OracleIndependence`,
:class:`~verification.claim_ledger.GateStatus`,
:class:`~verification.claim_ledger.StochasticEvidence` and the enforced rule that
a ``SHARES_CODE`` oracle cannot gate. Those are imported here, not forked.

The four parameter kinds
------------------------
The load-bearing new idea, because it is what makes validity sampling,
convergence and compositional generalization all well defined at once:

===================== ============================== =========================
kind                  changes the correct answer?    examples
===================== ============================== =========================
``PHYSICAL``          **yes** -- the oracle recomputes  index, curvature, field
                                                     angle, wavelength, NA, z
``NUMERICAL``         no -- changes achieved accuracy  grid, pitch, ray count,
                      and cost                       oversampling, pad
``REPRESENTATION``    no, beyond a declared budget   route, coupler, patch
                                                     granularity
``EXECUTION``         no, beyond a declared budget   device, precision, seed
===================== ============================== =========================

A family whose ``NUMERICAL`` parameter moves its oracle value has a defect, and
this split is what makes that statement testable rather than rhetorical. It is
also why ``B2-R2W-ROUTE`` and ``B0-DTYPE`` are benchmarks at all: they measure
how much a parameter that should not change the answer does.

The rules enforced at construction
----------------------------------
These are structural because prose has already failed at them once:

* an oracle that only ``SHARES_CODE``, or whose kind is ``CROSS_ROUTE``, forces
  ``category = B4`` and cannot carry a gating tolerance. Our own numerical code
  never decides correctness for our own numerical code, and the live case is
  Optiland's FFTPSF/HuygensPSF pair -- one Wavefront/OPD front end, not two
  oracles;
* a ``B4`` family cannot carry a gating tolerance at all, so
  ``CHARACTERIZED_NO_GATE`` is structurally impossible to promote by accident;
* a tolerance whose basis is a recorded measurement rather than an analytic or
  independent derivation cannot gate. A number that came out of a run cannot
  decide whether the next run of the same code is right;
* ``sampler = None`` requires a recorded reason. Non-generative is a
  declaration, not an omission;
* a stochastic family must require more than one seed. One realization is never
  an accuracy result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, ClassVar

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.claim_ledger import (
    LEDGER_COMPONENTS,
    ClaimKind,
    GateStatus,
    Oracle,
    OracleIndependence,
    StochasticEvidence,
)
from verification.status import VerificationStatus

__all__ = [
    "BenchmarkCategory",
    "BenchmarkFamily",
    "BenchmarkInstance",
    "ExecutionParameter",
    "ExecutionPolicy",
    "FamilyOracle",
    "InstanceOrigin",
    "Invariant",
    "Metric",
    "NegativeControl",
    "NumericalParameter",
    "Parameter",
    "ParameterKind",
    "PhysicalParameter",
    "ProvenanceRule",
    "RepresentationParameter",
    "SamplerAbsentReason",
    "StochasticEvidenceKind",
    "StochasticPolicy",
    "Tolerance",
    "ToleranceBasis",
    "ValidityBasis",
    "ValidityPredicate",
    "ValidityState",
    "fingerprint_of",
]


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------


class BenchmarkCategory(StrEnum):
    """What kind of question a family asks, and therefore what may decide it."""

    #: Contract and recovery. Does the component refuse what it cannot do, and
    #: does it refuse it in a way a caller can act on? Includes the silent
    #: hazards, where the contract status is ``ok`` and the physics is wrong.
    B0 = "B0"
    #: Primitive correctness inside one representation, against an analytic
    #: closed form or an invariant.
    B1 = "B1"
    #: A representation transition -- ray to wave, wave to ray, patch to global.
    B2 = "B2"
    #: A composed chain whose correctness is still decidable by something
    #: independent.
    B3 = "B3"
    #: Characterization. Reports convergence, cost, variance, reproducibility
    #: and cross-route consistency. **Never** gates, by construction.
    B4 = "B4"

    @property
    def may_gate(self) -> bool:
        return self is not BenchmarkCategory.B4


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class ParameterKind(StrEnum):
    PHYSICAL = "physical"
    NUMERICAL = "numerical"
    REPRESENTATION = "representation"
    EXECUTION = "execution"

    @property
    def changes_the_answer(self) -> bool:
        """Whether moving this parameter is *expected* to move the oracle value."""
        return self is ParameterKind.PHYSICAL


@dataclass(frozen=True)
class Parameter:
    """One declared axis of a family's parameter space.

    Abstract: instantiate one of the four kind-specific subclasses. ``kind`` is a
    ``ClassVar`` rather than a field precisely so that a parameter's kind cannot
    be passed in at a call site -- it is a property of which class you chose, and
    the choice is the scientific statement.

    ``domain`` is either a ``(low, high)`` pair for a continuous axis or a tuple
    of admissible values for a discrete one. It is declared even though no
    sampler exists yet (M9), because a bound nobody wrote down is a bound nobody
    can sample near.
    """

    kind: ClassVar[ParameterKind]

    name: str
    description: str
    unit: str | None = None
    domain: tuple[float, float] | tuple[Any, ...] | None = None
    default: Any = None
    #: For NUMERICAL parameters: the direction that refines. ``+1`` means larger
    #: is more accurate (ray count, grid), ``-1`` means smaller is (pitch, step).
    refines_toward: int | None = None

    def __post_init__(self) -> None:
        if type(self) is Parameter:
            raise TypeError(
                "Parameter is abstract: choose PhysicalParameter, NumericalParameter, "
                "RepresentationParameter or ExecutionParameter. Which one a quantity is "
                "is the declaration this schema exists to force."
            )
        if not self.name:
            raise ValueError("parameter needs a name")
        if not self.description.strip():
            raise ValueError(f"{self.name}: a parameter without a description is a magic number")
        if self.refines_toward not in (None, 1, -1):
            raise ValueError(f"{self.name}: refines_toward must be +1, -1 or None")
        if self.refines_toward is not None and self.kind is not ParameterKind.NUMERICAL:
            raise ValueError(
                f"{self.name}: only a NUMERICAL parameter refines. A PHYSICAL parameter "
                "that 'refines' is a family that does not know what its oracle depends on."
            )


@dataclass(frozen=True)
class PhysicalParameter(Parameter):
    """Changes the correct answer. The oracle recomputes."""

    kind: ClassVar[ParameterKind] = ParameterKind.PHYSICAL


@dataclass(frozen=True)
class NumericalParameter(Parameter):
    """Changes achieved accuracy and cost, not the answer.

    A family whose oracle value moves with one of these has a defect, and that
    is the statement the four-way split makes testable.
    """

    kind: ClassVar[ParameterKind] = ParameterKind.NUMERICAL


@dataclass(frozen=True)
class RepresentationParameter(Parameter):
    """Changes how the same physics is represented -- route, coupler, patch
    granularity. Not the answer, beyond a declared budget, and measuring that
    budget is what ``B2-R2W-ROUTE`` is for."""

    kind: ClassVar[ParameterKind] = ParameterKind.REPRESENTATION


@dataclass(frozen=True)
class ExecutionParameter(Parameter):
    """Device, precision, seed. Not the answer, beyond a declared budget, and
    ``B0-DTYPE`` is the family that measures how much of a budget that is."""

    kind: ClassVar[ParameterKind] = ParameterKind.EXECUTION


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


class ValidityBasis(StrEnum):
    """Where a validity bound comes from. Not decoration: an executable bound
    whose basis is 'somebody thought so' is not a bound."""

    #: eps_curv <= arcsin(D / 2R), the SI S3 curvature bound.
    SI_S3_CURVATURE = "si_s3_curvature"
    #: Per-axis Nyquist from marginal ray angles
    #: (``benchmarks/probes/slice_feasibility.py``).
    PER_AXIS_NYQUIST = "per_axis_nyquist"
    #: The angular-spectrum sampling boundary.
    ASM_SAMPLING = "asm_sampling"
    #: Non-empty capability intersection (``core/capabilities.py``).
    CAPABILITY_INTERSECTION = "capability_intersection"
    #: Hexapolar ring membership (``NON_HEXAPOLAR_SAMPLING``).
    HEXAPOLAR_RING = "hexapolar_ring"
    #: Declared planarity, for ``C_PLANAR_DOE_STEP`` and ``C_PATCH_WFT``.
    DECLARED_PLANARITY = "declared_planarity"
    #: Fresnel number regime.
    FRESNEL_NUMBER = "fresnel_number"
    #: Paraxial regime, from the closed form the oracle is derived under.
    PARAXIAL_APPROXIMATION = "paraxial_approximation"


class ValidityState(StrEnum):
    """Aggregated position relative to a validity domain.

    Ordered worst-last so aggregation over several predicates is a ``max``.
    """

    INSIDE = "inside"
    NEAR_BOUNDARY = "near_boundary"
    OUTSIDE = "outside"
    FAR_OUTSIDE = "far_outside"

    @property
    def rank(self) -> int:
        return _VALIDITY_RANK[self]

    @property
    def is_inside(self) -> bool:
        return self in (ValidityState.INSIDE, ValidityState.NEAR_BOUNDARY)


_VALIDITY_RANK = {
    ValidityState.INSIDE: 0,
    ValidityState.NEAR_BOUNDARY: 1,
    ValidityState.OUTSIDE: 2,
    ValidityState.FAR_OUTSIDE: 3,
}


@dataclass(frozen=True)
class ValidityPredicate:
    """The executable counterpart of ``core/specs.py::ValiditySpec``.

    ``ValiditySpec`` is three lists of strings and stays as human documentation.
    This one answers *how far* inside or outside, which is what makes "just
    outside validity" a reachable sampling target rather than a hope.

    ``margin(params)`` returns a **normalized signed** float:

    * ``> 0`` inside, ``0`` exactly at the boundary, ``< 0`` outside;
    * normalized, so that magnitudes from different predicates are comparable
      and the near-boundary band is one number rather than one per predicate.

    The convention throughout is ``margin = (limit - value) / limit`` for an
    upper bound, so ``+1`` is "an order of magnitude of headroom in the
    fractional sense" and ``-1`` is "twice the limit".
    """

    predicate_id: str
    statement: str
    basis: ValidityBasis
    margin: Callable[[Mapping[str, Any]], float]
    #: |margin| within this band is NEAR_BOUNDARY: neither safely inside nor
    #: usefully outside. 5% by default.
    near_boundary_band: float = 0.05
    #: Below -this, the instance is FAR_OUTSIDE: not a boundary probe but a
    #: different regime. 50% by default.
    far_outside_band: float = 0.5
    #: What the predicate does *not* bound. A predicate presented as complete
    #: when it is one of three is how an out-of-validity run reports as fine.
    blind_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError(f"{self.predicate_id}: a validity bound needs a statement")
        if not (0.0 < self.near_boundary_band < self.far_outside_band):
            raise ValueError(f"{self.predicate_id}: need 0 < near_boundary_band < far_outside_band")

    def state(self, params: Mapping[str, Any]) -> ValidityState:
        m = float(self.margin(params))
        if abs(m) <= self.near_boundary_band:
            return ValidityState.NEAR_BOUNDARY
        if m > 0.0:
            return ValidityState.INSIDE
        if m >= -self.far_outside_band:
            return ValidityState.OUTSIDE
        return ValidityState.FAR_OUTSIDE

    def evaluate(self, params: Mapping[str, Any]) -> tuple[float, ValidityState]:
        return float(self.margin(params)), self.state(params)


def aggregate_validity(states: Sequence[ValidityState]) -> ValidityState:
    """The worst position any single predicate reports.

    A conjunction, not an average: being comfortably inside two bounds does not
    buy any headroom on the third.
    """
    if not states:
        return ValidityState.INSIDE
    return max(states, key=lambda s: s.rank)


# ---------------------------------------------------------------------------
# Oracle, metrics, tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyOracle:
    """What decides this family, and whether it is allowed to.

    ``callable`` maps an instance's parameters to the expected value(s). It is
    ``None`` only where the oracle is an invariant or a conservation law with no
    single expected number, and the family then leans on ``invariants``.
    """

    kind: Oracle
    independence: OracleIndependence
    description: str
    callable: Callable[[Mapping[str, Any]], Any] | None = None
    #: Where the closed form or independent implementation comes from.
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("an oracle must say what it computes")
        if self.kind is Oracle.NONE and self.independence is not OracleIndependence.NOT_APPLICABLE:
            raise ValueError("Oracle.NONE must declare independence NOT_APPLICABLE")

    @property
    def may_decide_correctness(self) -> bool:
        """Whether this oracle is allowed to gate.

        Two ways to fail: sharing code with the thing under test, and being a
        second route through our own stack. ``CROSS_ROUTE`` is called out
        separately from ``SHARES_CODE`` because the two agree for the wrong
        reason -- Optiland's FFTPSF and HuygensPSF share one Wavefront/OPD front
        end, so their agreement measures the back ends and nothing else.
        """
        return (
            self.independence is OracleIndependence.INDEPENDENT
            and self.kind is not Oracle.CROSS_ROUTE
            and self.kind is not Oracle.NONE
        )


@dataclass(frozen=True)
class Metric:
    """One measured quantity, defined once.

    ``blind_to`` is required. Every metric this project has argued about turned
    out to be blind to something -- an intensity L2 to a global phase, an NCC to
    a scale factor, a Strehl to everything outside the core -- and the argument
    is cheaper if the blind spot is written next to the definition.
    """

    name: str
    description: str
    unit: str | None
    blind_to: tuple[str, ...]
    #: Which entry of ``verification/metrics.py`` computes this, when one does.
    #: A family's metric NAME is domain-specific -- ``fft_oracle_intensity_
    #: relative_l2`` says what is being compared to what -- while the arithmetic
    #: is shared, and naming the shared definition is what stops two benchmarks
    #: computing "relative L2" differently. ``None`` means the quantity is a
    #: measurement of one array rather than a comparison of two, which is a
    #: different kind of thing and lives in ``psf_measurement.py``.
    definition: str | None = None

    def __post_init__(self) -> None:
        if not self.blind_to:
            raise ValueError(
                f"{self.name}: state what this metric cannot see. If it genuinely "
                "sees everything, say so explicitly as a one-element tuple."
            )


class ToleranceBasis(StrEnum):
    """Where a threshold came from, which decides whether it may gate."""

    #: Derived from the closed form itself -- the tolerance admits only a
    #: genuinely different answer. A measured agreement may *calibrate* the
    #: headroom without making the basis a measurement.
    ANALYTIC_DERIVATION = "analytic_derivation"
    #: Derived from an independent implementation's own error bound.
    INDEPENDENT_DERIVATION = "independent_derivation"
    #: A conservation law's closure, e.g. energy to round-off.
    CONSERVATION_LAW = "conservation_law"
    #: The floating-point floor of the declared execution precision.
    NUMERICAL_PRECISION_FLOOR = "numerical_precision_floor"
    #: "This is the number the run produced." Cannot gate: it cannot tell a
    #: wrong answer from a wrong reference.
    RECORDED_MEASUREMENT = "recorded_measurement"
    #: Agreement between two routes through our own code. Cannot gate.
    CROSS_ROUTE_AGREEMENT = "cross_route_agreement"

    @property
    def is_independently_derived(self) -> bool:
        return self in (
            ToleranceBasis.ANALYTIC_DERIVATION,
            ToleranceBasis.INDEPENDENT_DERIVATION,
            ToleranceBasis.CONSERVATION_LAW,
            ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        )


@dataclass(frozen=True)
class Tolerance:
    """A threshold, where it came from, and whether it is allowed to decide.

    ``may_gate`` is declared rather than derived so that a family can hold a
    tolerance to a *tighter* rule than the schema's minimum -- an analytic basis
    that the author still does not want gating yet is a legitimate state. What
    the schema forbids is the other direction.
    """

    metric: str
    threshold: float
    basis: str
    basis_kind: ToleranceBasis
    may_gate: bool
    #: What a wrong answer this tolerance rejects looks like. A threshold that
    #: rejects nothing nameable is a threshold nobody can defend.
    rejects: str = ""

    def __post_init__(self) -> None:
        if self.threshold <= 0.0:
            raise ValueError(f"{self.metric}: a tolerance must be positive")
        if not self.basis.strip():
            raise ValueError(
                f"{self.metric}: a tolerance without its basis cannot be re-justified "
                "later and cannot be checked for gating eligibility"
            )
        if self.may_gate and not self.basis_kind.is_independently_derived:
            raise ValueError(
                f"{self.metric}: basis_kind {self.basis_kind.value} is a recorded "
                "measurement or a cross-route agreement, so may_gate must be False. "
                "A number produced by the code under test cannot decide whether the "
                "next run of that code is right."
            )


# ---------------------------------------------------------------------------
# Invariants and negative controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Invariant:
    """Something that must hold regardless of the parameter values.

    Distinct from a metric-plus-tolerance because an invariant is checkable
    without an oracle: energy closes, a round trip returns, a direction cosine
    has unit norm.
    """

    invariant_id: str
    statement: str
    metric: str
    tolerance: Tolerance


class NegativeControlExpectation(StrEnum):
    """What a deliberately broken twin is supposed to do."""

    #: The mutation makes the gate fail. The normal, useful case.
    MUST_FAIL = "must_fail"
    #: The mutation is known **not** to fail, and that is a recorded defect in
    #: the control or the oracle rather than a passing result. L2-PSF-01's
    #: inverted quadrature weight is the live case.
    KNOWN_FIRES_BACKWARDS = "known_fires_backwards"
    #: Declared and not yet implemented. Explicit so it cannot read as coverage.
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class NegativeControl:
    """A deliberately wrong twin, and what it must do.

    A gate a known-wrong twin can pass is not a gate. Every family declares its
    controls here, and :class:`~verification.result.VerificationResult` reports
    whether each one fired -- so an untrustworthy gate reports as untrustworthy
    rather than green.
    """

    control_id: str
    description: str
    #: The specific defect injected, in enough detail to reimplement.
    mutation: str
    #: Which metric the mutation must move past its threshold.
    target_metric: str
    expectation: NegativeControlExpectation = NegativeControlExpectation.MUST_FAIL
    #: Required when the expectation is anything other than MUST_FAIL.
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.expectation is not NegativeControlExpectation.MUST_FAIL and not self.caveat.strip():
            raise ValueError(
                f"{self.control_id}: a control that does not simply fail must say why. "
                "An unexplained backwards control is indistinguishable from a passing one."
            )


@dataclass(frozen=True)
class GateDisposition:
    """Where this family's gate currently stands, and on what evidence.

    A family declares what *would* decide it; whether the decision has been
    taken, and how it came out, is a separate fact and it changes over time.
    Keeping it here rather than in a second table is what lets
    ``claim_ledger.CLAIMS`` be a projection: the migrated form of
    ``manifest.yaml``'s ``gate_disposition`` prose, with the status as a value
    instead of a paragraph.

    ``NOT_MET`` is a first-class state. The failure mode being designed against
    is a benchmark whose unmet gate is discoverable only by reading a note.
    """

    status: GateStatus
    #: The metric the disposition is about, when there is one.
    metric: str | None = None
    #: What was measured. ``None`` where the gate was never run.
    observed: float | None = None
    #: Pytest node ids, record paths, or report anchors that must resolve.
    evidence: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.status in (GateStatus.MET, GateStatus.NOT_MET):
            if self.metric is None:
                raise ValueError("a decided gate must name the metric it decided")
            if self.observed is None:
                raise ValueError(
                    f"{self.metric}: a decided gate must report what was observed. A "
                    "verdict with no number cannot be re-checked."
                )
            if not self.evidence:
                raise ValueError(f"{self.metric}: a decided gate must cite its evidence")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class StochasticEvidenceKind(StrEnum):
    """The four evidence kinds ``benchmarks/protocols/coupler_protocol.yaml``
    requires, as separate facts because they fail independently.

    An estimator can be exact in the enumeration limit and still biased;
    unbiased and still converge at the wrong rate; converge correctly and have
    variance that makes the required ray count unreachable.
    """

    EXACTNESS_LIMIT = "exactness_limit"
    UNBIASEDNESS = "unbiasedness"
    CONVERGENCE_EXPONENT = "convergence_exponent"
    VARIANCE_CHARACTERIZATION = "variance_characterization"


@dataclass(frozen=True)
class StochasticPolicy:
    """Whether this family samples, and what it owes if it does."""

    is_stochastic: bool
    required_evidence: tuple[StochasticEvidenceKind, ...] = ()
    #: One realization is never an accuracy result, so a stochastic family
    #: requires an ensemble.
    minimum_seeds: int = 1
    #: Required when ``is_stochastic`` is False, so "we did not think about it"
    #: and "there is nothing to sample" are distinguishable.
    determinism_reason: str = ""

    def __post_init__(self) -> None:
        if self.is_stochastic:
            if self.minimum_seeds < 2:
                raise ValueError(
                    "a stochastic family must require at least 2 seeds; one "
                    "realization is never an accuracy result"
                )
            if not self.required_evidence:
                raise ValueError("a stochastic family must name the evidence it owes")
        else:
            if not self.determinism_reason.strip():
                raise ValueError(
                    "a non-stochastic family must say why it is deterministic, so "
                    "that an unnoticed source of randomness is not silently declared away"
                )
            if self.required_evidence or self.minimum_seeds != 1:
                raise ValueError("a deterministic family has no stochastic evidence to owe")


@dataclass(frozen=True)
class ExecutionPolicy:
    """What this family is allowed to run on, and how big it is allowed to get.

    The envelope is part of the family rather than the runner because a
    resource budget is a scientific statement: a study that needs 2.6e9 rays is
    describing its estimator, not its scheduler.
    """

    devices: frozenset[DeviceKind]
    dtypes: frozenset[DType]
    namespaces: frozenset[ArrayNamespace] = frozenset()
    max_wall_seconds: float | None = None
    max_peak_memory_gib: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("execution policy must name at least one device")
        if not self.dtypes:
            raise ValueError("execution policy must name at least one dtype")


@dataclass(frozen=True)
class ProvenanceRule:
    """What must match for two runs of this family to be comparable.

    ``core.provenance.VOLATILE_KEYS`` projects out timestamps, run ids, paths,
    pids and timings. The git dirty flag, package versions, device and dtype are
    deliberately *not* projected out: they change what was computed.
    """

    #: Fields that must be equal for a fingerprint comparison to mean anything.
    must_match: tuple[str, ...] = ("git_commit", "package_versions", "device", "dtype")
    #: Fields deliberately projected out of the scientific fingerprint.
    projected_out: tuple[str, ...] = ("timestamp", "run_id", "paths", "pid", "timings")
    notes: str = ""


class SamplerAbsentReason(StrEnum):
    """Why a family is non-generative. A declaration, not an omission."""

    #: Building the oracle for a new parameter point is expensive -- an
    #: enumerated reference, a hand-derived closed form.
    ORACLE_CONSTRUCTION_EXPENSIVE = "oracle_construction_expensive"
    #: Generating instances would weaken the independence of the oracle.
    GENERATION_WEAKENS_INDEPENDENCE = "generation_weakens_independence"
    #: The case is an important historical regression and its value is that it
    #: does not move.
    HISTORICAL_REGRESSION = "historical_regression"
    #: The physical setup itself is what is being certified.
    SETUP_IS_THE_CERTIFICATE = "setup_is_the_certificate"
    #: A stable fingerprint is especially valuable here.
    STABLE_FINGERPRINT_VALUABLE = "stable_fingerprint_valuable"


class InstanceOrigin(StrEnum):
    CANONICAL = "canonical"
    GENERATED = "generated"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """A JSON-serializable, order-stable projection of a parameter value."""
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, frozenset | set):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, tuple | list):
        return [_canonical(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, float) and not math.isfinite(value):
        # Infinity is a legitimate PHYSICAL parameter here -- an infinite
        # substrate radius is the planar case -- and JSON has no literal for it.
        # Mapped to a sentinel rather than dropped or coerced, so the planar
        # instance keeps a stable fingerprint that is distinguishable from a very
        # large finite radius.
        return {float("inf"): "+inf", float("-inf"): "-inf"}.get(value, "nan")
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    raise TypeError(
        f"{type(value).__name__} is not fingerprintable. A parameter whose value "
        "cannot be canonically serialized cannot be part of a stable fingerprint."
    )


def fingerprint_of(
    *,
    family_id: str,
    family_version: str,
    parameters: Mapping[str, Any],
    seed: int | None,
    execution_policy: ExecutionPolicy,
) -> str:
    """A canonical hash over what makes an instance the instance it is.

    SHA-256 over sorted-key JSON, so it is stable across processes -- Python's
    ``hash()`` is salted per process and would make a committed fingerprint
    meaningless. A ``family_version`` bump invalidates every fingerprint under
    it, which is the point: a family whose oracle or tolerance changed is not
    the same family, and its old results are not comparable.
    """
    payload = {
        "family_id": family_id,
        "family_version": family_version,
        "parameters": _canonical(dict(parameters)),
        "seed": seed,
        "execution_policy": {
            "devices": sorted(str(d) for d in execution_policy.devices),
            "dtypes": sorted(str(d) for d in execution_policy.dtypes),
            "namespaces": sorted(str(n) for n in execution_policy.namespaces),
            "max_wall_seconds": execution_policy.max_wall_seconds,
            "max_peak_memory_gib": execution_policy.max_peak_memory_gib,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkInstance:
    """One point in a family's parameter space, with its validity resolved.

    Built through :meth:`BenchmarkFamily.instantiate` rather than directly, so
    that ``validity_status`` and ``validity_margins`` are *computed from the
    family's predicates* rather than asserted by whoever wrote the instance. The
    verifier re-evaluates them against what the run actually did; the difference
    between the two is its own diagnostic.
    """

    instance_id: str
    family_id: str
    family_version: str
    parameters: Mapping[str, Any]
    origin: InstanceOrigin
    validity_status: ValidityState
    validity_margins: Mapping[str, float]
    fingerprint: str
    seed: int | None = None
    sampler_config: Mapping[str, Any] | None = None
    #: Which evaluation collection this belongs to -- ``required``, ``extended``,
    #: or a generated split. Kept off the fingerprint on purpose: moving an
    #: instance between collections does not change what it computes.
    split_tag: str = "required"
    #: What the family expects here, when it is known ahead of the run: an
    #: oracle value, a structured refusal, a status. Never a recorded output.
    expected: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("an instance needs an id")


# ---------------------------------------------------------------------------
# Family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkFamily:
    """A physical question, its oracle, its validity domain and its parameters.

    The invariants enforced in ``__post_init__`` are the ones this repository
    has already been burned by. They are constructor errors rather than test
    assertions so that a family violating them cannot be imported at all.
    """

    family_id: str
    family_version: str
    category: BenchmarkCategory
    #: The physical question, in a sentence a reader can disagree with.
    question: str
    #: Which registry components this family makes a statement about. A family
    #: that names no component cannot appear in the coverage view, which is the
    #: only reason anyone reads the ledger.
    components: tuple[str, ...]
    #: Which axis of the coverage matrix this occupies, so the projection into
    #: ``claim_ledger.CLAIMS`` lands in the right cell.
    claim_kind: ClaimKind
    parameters: tuple[Parameter, ...]
    oracle: FamilyOracle
    metrics: tuple[Metric, ...]
    execution_policy: ExecutionPolicy
    stochastic_policy: StochasticPolicy
    validity: tuple[ValidityPredicate, ...] = ()
    invariants: tuple[Invariant, ...] = ()
    tolerances: tuple[Tolerance, ...] = ()
    negative_controls: tuple[NegativeControl, ...] = ()
    #: Which :class:`VerificationStatus` values this family's failure paths are
    #: expected to be able to produce. Declaring them is what lets B0 check that
    #: a declared refusal code is reachable rather than decorative.
    failure_semantics: tuple[VerificationStatus, ...] = ()
    canonical_instances: tuple[BenchmarkInstance, ...] = ()
    #: Where the gate stands now. Required when anything here can gate.
    gate_disposition: GateDisposition | None = None
    #: M9. Declared here and ``None`` everywhere until then.
    sampler: Callable[..., BenchmarkInstance] | None = None
    sampler_absent_reason: SamplerAbsentReason | None = None
    sampler_absent_note: str = ""
    provenance_rule: ProvenanceRule = field(default_factory=ProvenanceRule)
    #: Evidence already recorded for this family: probe records, reports, test
    #: node ids. Provenance, never oracles.
    evidence: tuple[str, ...] = ()
    notes: str = ""

    # -- structural rules --------------------------------------------------

    def __post_init__(self) -> None:
        fid = self.family_id
        if not fid.startswith(tuple(f"{c.value}-" for c in BenchmarkCategory)):
            raise ValueError(f"{fid}: a family id must start with its category, e.g. B1-RAY-EFL")
        if not fid.startswith(f"{self.category.value}-"):
            raise ValueError(
                f"{fid}: id prefix disagrees with category {self.category.value}. Two "
                "places for the same fact is one place for them to disagree."
            )
        if not self.question.strip():
            raise ValueError(f"{fid}: a family must state its physical question")
        if not self.parameters:
            raise ValueError(f"{fid}: a family with no declared parameters is a script")
        if not self.metrics:
            raise ValueError(f"{fid}: a family must declare what it measures")

        if not self.components:
            raise ValueError(f"{fid}: a family must name the components it speaks about")
        unknown_components = [c for c in self.components if c not in LEDGER_COMPONENTS]
        if unknown_components:
            raise ValueError(
                f"{fid}: names components the ledger does not know: {unknown_components}. "
                f"Known: {list(LEDGER_COMPONENTS)}"
            )

        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{fid}: duplicate parameter name")

        metric_names = {m.name for m in self.metrics}
        for tol in self.tolerances:
            if tol.metric not in metric_names:
                raise ValueError(
                    f"{fid}: tolerance on {tol.metric!r}, which is not a declared metric"
                )
        for control in self.negative_controls:
            if control.target_metric not in metric_names:
                raise ValueError(
                    f"{fid}: negative control {control.control_id} targets "
                    f"{control.target_metric!r}, which is not a declared metric"
                )

        # -- the oracle-independence rules, structurally ------------------
        gating = tuple(t for t in self.tolerances if t.may_gate)

        if self.category is BenchmarkCategory.B4 and gating:
            raise ValueError(
                f"{fid}: a B4 family cannot carry a gating tolerance. B4 is "
                "characterization; CHARACTERIZED_NO_GATE must be impossible to "
                f"promote by accident. Offending metrics: {[t.metric for t in gating]}"
            )

        if not self.oracle.may_decide_correctness:
            if self.category is not BenchmarkCategory.B4:
                raise ValueError(
                    f"{fid}: oracle kind {self.oracle.kind.value} with independence "
                    f"{self.oracle.independence.value} cannot decide correctness, so "
                    "this family must be category B4. Our own numerical code does not "
                    "decide correctness for our own numerical code, and two routes "
                    "through one front end are one oracle, not two."
                )
            if gating:
                raise ValueError(
                    f"{fid}: an oracle that cannot decide correctness cannot carry a "
                    f"gating tolerance. Offending metrics: {[t.metric for t in gating]}"
                )

        # -- gate disposition ---------------------------------------------
        if gating:
            if self.gate_disposition is None:
                raise ValueError(
                    f"{fid}: carries a gating tolerance and no gate_disposition. A gate "
                    "whose current standing is unrecorded reads as met."
                )
            if self.gate_disposition.status is GateStatus.CHARACTERIZED_NO_GATE:
                raise ValueError(
                    f"{fid}: CHARACTERIZED_NO_GATE on a family that gates. Those are "
                    "the two states this schema exists to keep apart."
                )
        elif self.gate_disposition is not None and self.gate_disposition.status in (
            GateStatus.MET,
            GateStatus.NOT_MET,
        ):
            raise ValueError(
                f"{fid}: reports a decided gate with no gating tolerance to decide. "
                "Either the tolerance may_gate, or the disposition is a characterization."
            )

        # -- sampler ------------------------------------------------------
        if self.sampler is None and self.sampler_absent_reason is None:
            raise ValueError(
                f"{fid}: sampler is None with no recorded reason. Non-generative is a "
                "declaration, not an omission -- name one of "
                f"{[r.value for r in SamplerAbsentReason]}."
            )
        if self.sampler is not None and self.sampler_absent_reason is not None:
            raise ValueError(f"{fid}: a family with a sampler has no absent reason")

        # -- instances ----------------------------------------------------
        for inst in self.canonical_instances:
            if inst.family_id != fid:
                raise ValueError(
                    f"{fid}: canonical instance {inst.instance_id} names another family"
                )
            if inst.family_version != self.family_version:
                raise ValueError(
                    f"{fid}: canonical instance {inst.instance_id} was built against "
                    f"family_version {inst.family_version}, not {self.family_version}. "
                    "A version bump invalidates its fingerprint; rebuild it."
                )
            if inst.origin is not InstanceOrigin.CANONICAL:
                raise ValueError(
                    f"{fid}: {inst.instance_id} is listed as canonical but its origin "
                    f"is {inst.origin.value}"
                )
        ids = [i.instance_id for i in self.canonical_instances]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{fid}: duplicate canonical instance id")

    # -- derived ------------------------------------------------------------

    @property
    def gating_tolerances(self) -> tuple[Tolerance, ...]:
        return tuple(t for t in self.tolerances if t.may_gate)

    @property
    def is_gate_deciding(self) -> bool:
        """Whether anything here can decide a pass/fail at all."""
        return bool(self.gating_tolerances)

    @property
    def refinement_dimensions(self) -> tuple[Parameter, ...]:
        """The NUMERICAL parameters that declare a refinement direction.

        A family with one of these owes a convergence ladder: it has said that
        somewhere in its parameter space is a direction along which the answer
        should stop moving, and a single point cannot show that it does.
        """
        return tuple(p for p in self.parameters if p.refines_toward is not None)

    def parameters_of_kind(self, kind: ParameterKind) -> tuple[Parameter, ...]:
        return tuple(p for p in self.parameters if p.kind is kind)

    def metric(self, name: str) -> Metric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(f"{self.family_id}: no metric {name!r}")

    def tolerance_for(self, metric: str) -> Tolerance | None:
        return next((t for t in self.tolerances if t.metric == metric), None)

    def with_instances(self, *instances: BenchmarkInstance) -> BenchmarkFamily:
        """A copy carrying these canonical instances.

        Instances are built by :meth:`instantiate`, which needs the family, so
        the family is constructed first and its canonical set attached second.
        ``dataclasses.replace`` re-runs ``__post_init__``, so the instances are
        checked against the family they claim to belong to.
        """
        from verification.families.registry import FAMILIES

        if self.family_id in FAMILIES and FAMILIES[self.family_id] is self:
            raise ValueError(
                f"{self.family_id} is already registered, so attaching instances now "
                "would produce a COPY that the registry does not hold -- the registered "
                "family would silently have no canonical instances. Build the family, "
                "attach its instances, then register the result."
            )
        return replace(self, canonical_instances=tuple(instances))

    def evaluate_validity(
        self, parameters: Mapping[str, Any]
    ) -> tuple[ValidityState, dict[str, float]]:
        """Every predicate's signed margin, and the worst state among them."""
        margins: dict[str, float] = {}
        states: list[ValidityState] = []
        for predicate in self.validity:
            try:
                margin, state = predicate.evaluate(parameters)
            except KeyError as missing:
                # A family that declares a bound its own parameters cannot feed
                # has declared a bound nothing can check. Found twice while
                # authoring FIXED-V1 -- B1-WAVE-AIRY and B1-WAVE-TALBOT both
                # carried the ASM sampling predicate without the distance and
                # pitch it reads -- so the message names all three things a
                # reader needs rather than raising a bare KeyError from inside a
                # lambda.
                raise ValueError(
                    f"{self.family_id}: validity predicate "
                    f"{predicate.predicate_id} reads {missing.args[0]!r}, which is not "
                    f"in this family's parameter space "
                    f"{sorted(p.name for p in self.parameters)}. Either the family "
                    "declares a bound it cannot evaluate, or the predicate is "
                    "configured against the wrong key names."
                ) from missing
            margins[predicate.predicate_id] = margin
            states.append(state)
        return aggregate_validity(states), margins

    def instantiate(
        self,
        instance_id: str,
        parameters: Mapping[str, Any],
        *,
        origin: InstanceOrigin = InstanceOrigin.CANONICAL,
        seed: int | None = None,
        sampler_config: Mapping[str, Any] | None = None,
        split_tag: str = "required",
        expected: Mapping[str, Any] | None = None,
        pinned_fingerprint: str | None = None,
    ) -> BenchmarkInstance:
        """Build an instance, resolving its validity from *this* family's predicates.

        ``pinned_fingerprint`` is the regression hook: pass the committed value
        and construction fails if the instance no longer hashes to it. That is a
        louder failure than discovering later that a "canonical" instance
        quietly moved.
        """
        declared = {p.name for p in self.parameters}
        unknown = set(parameters) - declared
        if unknown:
            raise ValueError(
                f"{self.family_id}/{instance_id}: undeclared parameters {sorted(unknown)}. "
                "A parameter that is not in the family's space cannot be sampled, "
                "bounded or fingerprinted meaningfully."
            )
        missing = {
            p.name for p in self.parameters if p.default is None and p.name not in parameters
        }
        if missing:
            raise ValueError(
                f"{self.family_id}/{instance_id}: missing parameters {sorted(missing)}"
            )
        resolved = {p.name: parameters.get(p.name, p.default) for p in self.parameters}

        if self.stochastic_policy.is_stochastic and seed is None:
            raise ValueError(
                f"{self.family_id}/{instance_id}: a stochastic family's instance must "
                "carry a seed, or the run is not reproducible and the ensemble has no "
                "members to enumerate."
            )

        status, margins = self.evaluate_validity(resolved)
        fp = fingerprint_of(
            family_id=self.family_id,
            family_version=self.family_version,
            parameters=resolved,
            seed=seed,
            execution_policy=self.execution_policy,
        )
        if pinned_fingerprint is not None and fp != pinned_fingerprint:
            raise ValueError(
                f"{self.family_id}/{instance_id}: fingerprint moved.\n"
                f"  pinned:   {pinned_fingerprint}\n"
                f"  computed: {fp}\n"
                "Either a parameter, the execution policy or family_version changed. "
                "Diagnose which before re-pinning -- a re-recorded fingerprint with no "
                "attribution is how a canonical instance silently becomes another one."
            )
        return BenchmarkInstance(
            instance_id=instance_id,
            family_id=self.family_id,
            family_version=self.family_version,
            parameters=resolved,
            origin=origin,
            validity_status=status,
            validity_margins=margins,
            fingerprint=fp,
            seed=seed,
            sampler_config=sampler_config,
            split_tag=split_tag,
            expected=expected,
        )


# Re-exported so a family author imports one module.
__all__ += ["ClaimKind", "GateDisposition", "GateStatus", "NegativeControlExpectation"]
__all__ += ["Oracle", "OracleIndependence", "StochasticEvidence"]
__all__ += ["aggregate_validity"]
