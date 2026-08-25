"""Turn an executed instance into evidence: measure, judge, fit, fingerprint, record.

CHE-106/107/108/109/110/111/112 (M1, M2). The families, the schema, the executor
and the verifier all existed and had never been joined up for anything except one
B3 proof, so twenty-one families reported ``NOT_MEASURED`` or
``MEASURED_OFF_GATE`` and the honest reading of that was: declared, never
executed. This module is the join.

Why it is here and not in ``runtime/``
--------------------------------------
``tests/test_package_dependencies.py`` refuses an import of ``verification/``
from ``runtime/``, on the argument that an executor which could grade its own run
makes the ``ExecutionRecord`` boundary meaningless. The dependency therefore runs
the other way: the side that *judges* drives the side that *runs*, and never the
reverse. ``runtime/instance_runner.py`` executes and knows nothing about
tolerances; everything below reads a record and says what it means.

What this module does **not** do is measure physics. Turning a terminal artifact
into a number an oracle can be compared against is per-family by construction,
it is the only interesting part, and a shared helper that did it would either be
a second solver or a lowest common denominator. Each driver under
``benchmarks/instances/`` supplies its own ``_measure``.

Why the exponent carries a standard error
-----------------------------------------
``core.performance.fit_scaling`` returns a slope and an r^2, which is what a cost
model needs. A *scientific* claim of the form "the exponent is -0.5 +/- 0.1"
needs the uncertainty on the slope, not the fraction of variance explained: an
r^2 of 0.999 over three tightly clustered rungs can sit beside a slope whose
standard error is 0.3. ``fit_convergence`` wraps the same least squares and adds
the residual standard error of the slope, so the reported exponent has an error
bar and the schema's "a value with no error bar is a schema violation" holds for
a fit as well as for a measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.execution import RunStatus
from core.execution_record import ExecutionRecord
from core.performance import fit_scaling
from core.provenance import record_provenance, strip_volatile
from core.specs import GraphSpec
from registry.loader import Registry
from runtime.instance_runner import RUNNER_VERSION, execute
from verification.families.schema import (
    BenchmarkFamily,
    BenchmarkInstance,
    NegativeControlExpectation,
)
from verification.result import (
    ConvergenceReport,
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    StochasticReport,
    UncertaintyBasis,
    VerificationResult,
)
from verification.verifier import verify

__all__ = [
    "INSTANCE_RECORDS_DIR",
    "Ensemble",
    "InstanceRun",
    "control_result",
    "ensemble",
    "fit_convergence",
    "result_fingerprint",
    "run_and_verify",
    "sigma_margin",
    "write_instance_record",
]


def _repository_root() -> Path:
    from core.paths import repository_root

    return repository_root()


#: Where drivers write. Separate from ``benchmarks/probes/records/`` because
#: these are family-instance results rather than probe outputs; the provenance
#: sweep covers both directories.
INSTANCE_RECORDS_DIR = "benchmarks/instances/records"


def _records_dir() -> Path:
    return _repository_root() / "benchmarks" / "instances" / "records"


# ---------------------------------------------------------------------------
# One executed instance, both halves kept apart
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceRun:
    """One instance, executed and interpreted. Both halves, kept apart."""

    family: BenchmarkFamily
    instance: BenchmarkInstance
    record: ExecutionRecord
    result: VerificationResult

    @property
    def succeeded(self) -> bool:
        return self.record.status is RunStatus.SUCCEEDED

    @property
    def fingerprint(self) -> str:
        return result_fingerprint(self.result)



def run_and_verify(
    family: BenchmarkFamily,
    instance: BenchmarkInstance,
    graph: GraphSpec,
    *,
    measure: Callable[[ExecutionRecord, BenchmarkInstance], Mapping[str, Measurement]]
    | None = None,
    invariants: Mapping[str, Measurement] | None = None,
    negative_controls: Mapping[str, NegativeControlResult] | None = None,
    convergence: ConvergenceReport | None = None,
    stochastic: StochasticReport | None = None,
    silent_hazard_ids: tuple[str, ...] = (),
    seed: int | None = None,
    inputs: Mapping[str, Any] | None = None,
    registry: Registry | None = None,
) -> InstanceRun:
    """Execute, measure, verify. Returns both records.

    A run that did not succeed is verified **without** measurements. That is the
    whole reason the record is handed to the verifier rather than a number: a
    refused run has a structured status and no physics, and inventing a metric
    for it would be the fabrication the contract layer exists to prevent.
    """
    record = execute(graph, instance, seed=seed, inputs=inputs, registry=registry)

    measurements: Mapping[str, Measurement] | None = None
    if record.status is RunStatus.SUCCEEDED and measure is not None:
        measurements = measure(record, instance)

    result = verify(
        family,
        instance,
        record,
        measurements=measurements,
        invariants=invariants if record.status is RunStatus.SUCCEEDED else None,
        negative_controls=negative_controls,
        convergence=convergence,
        stochastic=stochastic,
        silent_hazard_ids=silent_hazard_ids,
    )
    return InstanceRun(family=family, instance=instance, record=record, result=result)



# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def fit_convergence(
    dimension: str,
    points: Sequence[tuple[float, float]],
    *,
    expected_exponent: float | None = None,
    exponent_tolerance: float | None = None,
    note: str = "",
    uncertainty_basis: UncertaintyBasis = UncertaintyBasis.FIT_STANDARD_ERROR,
) -> ConvergenceReport:
    """A refinement ladder, its fitted exponent, and the exponent's error bar.

    ``points`` are ``(refinement value, error)`` pairs, at least four of them for
    a claim -- three is the minimum a fit is defined over, and the acceptance
    criteria in M1.1 and M2.2 ask for four and six respectively because a
    three-point fit cannot show curvature.

    ``converged`` is decided by the exponent against ``expected_exponent`` within
    ``exponent_tolerance`` when both are given, and is otherwise left ``None``:
    a ladder with no expected rate is a measurement, and calling it converged
    would be a verdict the caller did not supply the criterion for.
    """
    if len(points) < 3:
        raise ValueError(
            f"{dimension}: a fitted exponent needs at least 3 rungs, got {len(points)}. "
            "Two points fit a line exactly and say nothing about whether the "
            "relationship is a power law."
        )

    fit = fit_scaling(points, axis=dimension)
    slope_se = _slope_standard_error(fit.points, fit.exponent, fit.intercept_log10)

    converged: bool | None = None
    if expected_exponent is not None and exponent_tolerance is not None:
        converged = abs(fit.exponent - expected_exponent) <= exponent_tolerance

    ladder = [x for x, _ in fit.points]
    values = [
        Measurement(
            value=y,
            # A rung is one measured point on the ladder. Its own error bar is
            # NOT_ESTIMATED rather than zero: what the ladder establishes is the
            # trend, and quoting a per-rung uncertainty of zero would claim each
            # point is exact when what is being measured is that they are not.
            uncertainty=None,
            uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
            note=f"{dimension}={x:g}",
        )
        for x, y in fit.points
    ]

    return ConvergenceReport(
        dimension=dimension,
        ladder=ladder,
        values=values,
        fitted_exponent=Measurement(
            value=fit.exponent,
            uncertainty=slope_se,
            uncertainty_basis=uncertainty_basis,
            note=(
                f"log-log least squares over {len(ladder)} rungs; "
                f"r^2={fit.r_squared if fit.r_squared is not None else float('nan'):.6f}"
            ),
        ),
        expected_exponent=expected_exponent,
        converged=converged,
        note=note,
    )


def _slope_standard_error(
    points: Sequence[tuple[float, float]], slope: float, intercept: float
) -> float | None:
    """Residual standard error of the log-log slope.

    ``None`` for a three-point fit with zero residual, where the standard error
    is genuinely undefined rather than zero -- reporting 0.0 there would claim
    an exact exponent from a line through three collinear points.
    """
    n = len(points)
    if n < 3:
        return None
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    residuals = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    if residuals == 0.0:
        return None
    return math.sqrt(residuals / (n - 2) / sxx)



# ---------------------------------------------------------------------------
# Ensembles and negative controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ensemble:
    """Sample statistics over seeds. The thing one realization cannot give you."""

    values: tuple[float, ...]
    mean: float
    #: Sample standard deviation, ``ddof=1``. ``None`` for a single value, where
    #: it is undefined rather than zero.
    standard_deviation: float | None
    #: Standard error of the mean. The unit a bias is gated in.
    standard_error: float | None

    @property
    def seed_count(self) -> int:
        return len(self.values)


def ensemble(values: Sequence[float]) -> Ensemble:
    """Mean, sample sd and standard error, with ddof=1.

    ``ddof=1`` rather than 0 because these are samples from an estimator, not a
    population, and at three seeds the difference between the two is 22% of the
    reported spread -- which is a large fraction of a 3-sigma gate.
    """
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("an ensemble of nothing has no statistics")
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return Ensemble(values=tuple(xs), mean=mean, standard_deviation=None, standard_error=None)
    variance = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    sd = math.sqrt(variance)
    return Ensemble(
        values=tuple(xs),
        mean=mean,
        standard_deviation=sd,
        standard_error=sd / math.sqrt(len(xs)),
    )


def control_result(
    control_id: str,
    target_metric: str,
    *,
    baseline: Measurement,
    mutated: Measurement,
    threshold: float,
    expectation: NegativeControlExpectation = NegativeControlExpectation.MUST_FAIL,
    note: str = "",
) -> NegativeControlResult:
    """Judge a broken twin against the same threshold the gate uses.

    The outcome is decided by the mutated arm crossing the *gate's own*
    threshold, not by it differing from the baseline: a mutation that moves the
    metric by 1% and stays green has not demonstrated that the gate can tell
    right from wrong, which is the only thing a control is for.

    ``FIRED_BACKWARDS`` is reported when the mutation *improved* the metric.
    That is a real state -- L2-PSF-01's inverted quadrature weight -- and
    collapsing it into "did not fire" would lose the information that the
    control is measuring something with the wrong sign.
    """
    detection = _detection_margin(baseline.value, mutated.value)
    if mutated.value < baseline.value:
        outcome = NegativeControlOutcome.FIRED_BACKWARDS
    elif mutated.value > threshold:
        outcome = NegativeControlOutcome.FIRED
    else:
        outcome = NegativeControlOutcome.DID_NOT_FIRE

    if expectation is NegativeControlExpectation.KNOWN_FIRES_BACKWARDS and (
        outcome is NegativeControlOutcome.FIRED_BACKWARDS
    ):
        # Declared as firing backwards and observed doing so. Still reported as
        # FIRED_BACKWARDS -- the declaration explains it, it does not excuse it.
        pass

    detail = (
        f"baseline {baseline.value:.6g} -> mutated {mutated.value:.6g} against a "
        f"gate of {threshold:.6g}; detection margin {detection:.6g}x"
    )
    return NegativeControlResult(
        control_id=control_id,
        outcome=outcome,
        target_metric=target_metric,
        mutated=mutated,
        baseline=baseline,
        note=f"{note} {detail}".strip(),
    )


def _detection_margin(baseline: float, mutated: float) -> float:
    """How many times worse the broken arm is. ``inf`` when the baseline is zero.

    A ratio rather than a difference, so the margin is comparable across
    families whose metrics have different scales.
    """
    if baseline == 0.0:
        return math.inf if mutated != 0.0 else 1.0
    return abs(mutated) / abs(baseline)


def sigma_margin(baseline_mean: float, mutated_mean: float, standard_error: float) -> float:
    """A stochastic detection margin, in measured standard errors.

    The only admissible unit for a stochastic control: an absolute field-space
    difference says nothing without the noise it has to be seen against.
    """
    if standard_error <= 0.0:
        raise ValueError(
            "a sigma margin needs a positive measured standard error. A margin "
            "computed against zero noise is a margin computed against one seed."
        )
    return abs(mutated_mean - baseline_mean) / standard_error



# ---------------------------------------------------------------------------
# Fingerprints and records
# ---------------------------------------------------------------------------


def result_fingerprint(result: VerificationResult) -> str:
    """A hash over the *scientific content* of a result.

    ``strip_volatile`` removes run ids and timings at every depth, so two
    independent runs of the same configuration hash the same and a changed
    measurement changes the hash. That asymmetry is deliberate and is the
    property M1.1 and M1.2 ask to be verified across two runs.

    Uncertainties are kept. An error bar is part of what was measured, and a
    fingerprint that ignored it would call two runs identical when one of them
    quoted a precision the other did not support.
    """
    payload = strip_volatile(json.loads(result.model_dump_json()))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_instance_record(
    run: InstanceRun,
    *,
    driver: str,
    extra: Mapping[str, Any] | None = None,
    data_inputs: Sequence[Path] = (),
    directory: Path | None = None,
) -> Path:
    """Persist a result with a provenance stamp and its scientific fingerprint.

    Called at the end of a driver, after the physics has run, so the provenance
    block sees the modules the run actually imported -- which is what makes a
    later code change able to invalidate the record instead of leaving it
    silently stale.
    """
    directory = directory or _records_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.instance.instance_id}.json"

    payload: dict[str, Any] = {
        "instance_id": run.instance.instance_id,
        "family_id": run.family.family_id,
        "family_version": run.family.family_version,
        "instance_fingerprint": run.instance.fingerprint,
        "scientific_fingerprint": result_fingerprint(run.result),
        "runner_version": RUNNER_VERSION,
        "parameters": dict(run.instance.parameters),
        "expected": dict(run.instance.expected),
        "execution": {
            "status": str(run.record.status),
            "seeds": list(run.record.seeds),
            "observed_parameters": dict(run.record.observed_parameters),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "component": node.component,
                    "outcome": str(node.outcome),
                    "refusal": None
                    if node.refusal is None
                    else {
                        "kind": str(node.refusal.kind),
                        "detail": node.refusal.detail,
                        "declaration": node.refusal.declaration,
                        "remedy": node.refusal.remedy,
                    },
                }
                for node in run.record.nodes
            ],
        },
        "verification": json.loads(run.result.model_dump_json()),
    }
    if extra:
        payload |= dict(extra)

    payload["record_provenance"] = record_provenance(
        probe=driver,
        root=_repository_root(),
        data_inputs=data_inputs,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    return path
