"""The ten canonical B0 instances, run. Five statuses from five real refusals.

CHE-108 (M1.3), part B0.3. ``tests/test_b0_families.py`` asserts the
declarations are well formed; ``tests/test_contract_code_reachability.py``
asserts each code can be emitted. This file asserts the third thing, which is
the one the milestone is actually about: that the *substrate* -- shipping
component, execution record, verifier -- produces each of the five negative
outcomes from a real request, and reports the two silent traps as ``ok``.

Why this belongs in the default gate
------------------------------------
Ten instances, two real Optiland traces and two real Chromatix propagations,
about ten seconds total. The alternative is an on-demand run nobody makes, and
the property being guarded is exactly the kind that decays quietly: a status
that collapses into a generic failure looks like a passing test everywhere else.

CHE-108 found one such collapse while writing this. ``OPL_REFERENCE_UNVERIFIED``
is ``blocked`` in the refusal catalogue and was arriving as
``invalid_configuration`` through the record path, because the executor maps the
code to ``MISSING_EDGE_DECLARATION``. The five statuses were therefore *not*
individually reachable through the substrate before this file existed, and
nothing failed.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from core.execution import RunStatus
from core.paths import repository_root
from verification.result import NegativeControlOutcome
from verification.status import VerificationStatus

pytestmark = [pytest.mark.integration, pytest.mark.optiland, pytest.mark.chromatix]


def _driver():
    """Load the driver by path.

    ``benchmarks/`` is not an importable package and deliberately stays that
    way; the substrate proof loads its driver the same way for the same reason.
    """
    name = "b0_contract_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b0_contract.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runs():
    return _driver().run_all()


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_declared_instance_was_executed(runs) -> None:
    """A declared instance with no runner is the gap this file closes."""
    declared = set(_driver().declared_instance_ids())
    assert set(runs) == declared
    assert len(declared) == 10


def test_the_five_negative_outcomes_come_from_five_real_executions(runs) -> None:
    """The load-bearing assertion of M1.3.

    Each of the five must be produced by a *different* instance. One instance
    that happened to produce three of them would say nothing about whether an
    agent can tell them apart.
    """
    by_status: dict[VerificationStatus, list[str]] = {}
    for instance_id, run in runs.items():
        by_status.setdefault(run.result.status, []).append(instance_id)

    for status in (
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.INVALID_CONFIGURATION,
        VerificationStatus.OUT_OF_VALIDITY,
        VerificationStatus.LOSSY_BUT_ALLOWED,
        VerificationStatus.BLOCKED,
    ):
        assert status in by_status, (
            f"{status.value} was not produced by any executed instance; the five "
            "outcomes are not individually reachable through the substrate"
        )

    # Distinct instances, not distinct statuses from one instance. An instance
    # that happened to produce three of the five would say nothing about whether
    # an agent can tell them apart.
    producers = {
        status: instances
        for status, instances in by_status.items()
        if status is not VerificationStatus.OK
    }
    assert len({tuple(v)[0] for v in producers.values()}) == len(producers)


@pytest.mark.parametrize(
    "instance_id",
    [
        "B0-CAPINT-01",
        "B0-DEVICE-01",
        "B0-DEVICE-02",
        "B0-META-01",
        "B0-HANDOFF-01",
        "B0-PATCH-01",
        "B0-DTYPE-01",
        "B0-VALIDITY-01",
        "B0-UNITS-01",
        "B0-UNITS-02",
    ],
)
def test_the_observed_status_is_the_declared_one(runs, instance_id: str) -> None:
    """Expected versus observed, per instance, from the instance's own declaration."""
    run = runs[instance_id]
    expected = run.instance.expected.get("status") or run.instance.expected.get(
        "contract_status"
    )
    assert expected is not None, f"{instance_id} declares no expected status"
    assert run.result.status.value == expected, (
        f"{instance_id} declared {expected} and the substrate produced "
        f"{run.result.status.value}"
    )


# --------------------------------------------------------------------------- #
# No fabricated output on a refusal
# --------------------------------------------------------------------------- #


#: The two instances whose ``out_of_validity`` comes from the family's own
#: validity predicates rather than from a component refusing. That is the
#: verifier's second route to the status and it is the correct one for both:
#: the request EXECUTES and the answer would be wrong, which is what
#: out-of-validity means. Neither is a refusal, so neither is held to the
#: refusal-shape assertions -- they are held to their own, below.
_VALIDITY_ROUTE = ("B0-PATCH-01", "B0-VALIDITY-01")


def _refused(runs):
    return {
        instance_id: run
        for instance_id, run in runs.items()
        if instance_id not in _VALIDITY_ROUTE
        and run.result.status
        not in (VerificationStatus.OK, VerificationStatus.LOSSY_BUT_ALLOWED)
    }


def test_no_refusal_reports_a_metric_a_convergence_or_a_gate(runs) -> None:
    """An `AGENTS.md` non-negotiable, checked on real refusals.

    A refused run has a structured status and no physics. If it also carried a
    metric, a downstream consumer could compare that metric to a tolerance and
    report a verdict about a computation that did not happen.
    """
    for instance_id, run in _refused(runs).items():
        assert run.record.status is not RunStatus.SUCCEEDED, instance_id
        assert run.result.physics_accuracy == [], (
            f"{instance_id} refused and still reported metrics: "
            f"{[m.metric for m in run.result.physics_accuracy]}"
        )
        assert run.result.invariant_results == [], instance_id
        assert run.result.convergence.converged is None, instance_id


def test_the_node_that_refused_produced_nothing(runs) -> None:
    """The precise version of "no fabricated output", for a partial graph run.

    ``B0-META-01`` refuses at the C_RAY_TO_WAVE edge *after* a real Optiland
    trace has already succeeded, so the record legitimately carries the ray
    node's artifacts -- that node ran and those are its outputs. What must not
    exist is an output of the node that refused, because that is the one a
    downstream consumer would read as a field.
    """
    for instance_id, run in _refused(runs).items():
        for node in run.record.nodes:
            if node.refusal is None:
                continue
            assert node.outputs == [], (
                f"{instance_id}/{node.node_id} refused and declared outputs "
                f"{node.outputs}"
            )
            produced = [key for key in run.record.artifacts if key.startswith(f"{node.node_id}:")]
            assert not produced, (
                f"{instance_id}/{node.node_id} refused and left "
                f"{produced} for a downstream node to consume"
            )


def test_every_refusal_carries_a_code_a_reason_and_a_remedy(runs) -> None:
    """The shape of the refusal, not merely the fact of it."""
    from verification.refusals import REFUSAL_CATALOGUE

    for instance_id, run in _refused(runs).items():
        contract = run.result.contract_status
        assert contract.refusal_kind is not None, instance_id
        assert contract.refusal_detail and contract.refusal_detail.strip(), instance_id

        refusal = run.record.refusal or next(
            (n.refusal for n in run.record.nodes if n.refusal is not None), None
        )
        assert refusal is not None, instance_id
        catalogue_remedy = next(
            (
                REFUSAL_CATALOGUE[code].remedy
                for code in run.record.contract_codes
                if code in REFUSAL_CATALOGUE
            ),
            None,
        )
        assert (refusal.remedy or catalogue_remedy), (
            f"{instance_id} refused with no remedy at the raise site and none in "
            "the catalogue"
        )


def test_the_unsupported_cases_name_what_is_supported(runs) -> None:
    """So a caller can choose again without guessing.

    This is the difference between "no" and "no, and here is the set". An agent
    that has to enumerate dtypes by trial and error will spend its budget doing
    that.
    """
    for instance_id in ("B0-CAPINT-01", "B0-DEVICE-01"):
        detail = runs[instance_id].result.contract_status.refusal_detail or ""
        assert "Supported:" in detail, f"{instance_id}: {detail[:200]}"


def test_the_capability_intersection_is_reported_as_empty(runs) -> None:
    """Not just refused: reported as the *reason* being an empty intersection."""
    run = runs["B0-CAPINT-01"]
    assert run.result.contract_status.capability_intersection_empty
    codes = {d.get("code") for d in run.record.diagnostics}
    assert "NATIVE_COMPUTE_INTERSECTION" in codes
    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "NATIVE_COMPUTE_INTERSECTION"
    )
    assert "intersection []" in detail


# --------------------------------------------------------------------------- #
# blocked is not invalid_configuration
# --------------------------------------------------------------------------- #


def test_a_missing_declaration_the_component_could_have_defaulted_is_blocked(runs) -> None:
    """The collapse CHE-108 found, as a standing test.

    Nothing about the request is malformed. The coupler *could* proceed by
    assuming a reference, and refuses to, because a bare ``opd_native`` is an
    absolute accumulated path whose zero moves with the aperture. That is
    ``blocked``, and reporting it as ``invalid_configuration`` would send a
    caller to fix a request that has nothing wrong with it.
    """
    run = runs["B0-HANDOFF-01"]
    assert run.result.status is VerificationStatus.BLOCKED
    assert "OPL_REFERENCE_UNVERIFIED" in run.record.contract_codes


def test_a_malformed_request_is_invalid_configuration_not_blocked(runs) -> None:
    """The other side of the same distinction, so neither absorbs the other."""
    run = runs["B0-META-01"]
    assert run.result.status is VerificationStatus.INVALID_CONFIGURATION
    assert "REFERENCE_PLANE_MISMATCH" in run.record.contract_codes


@pytest.mark.parametrize("instance_id", _VALIDITY_ROUTE)
def test_out_of_validity_comes_from_a_run_that_succeeded(runs, instance_id: str) -> None:
    """The distinction that makes the status worth having.

    Both of these EXECUTE. Nothing refused, no exception was raised, and the
    answer would be wrong -- which is the whole content of ``out_of_validity``
    and the reason it must not collapse into ``invalid_configuration``. The
    status comes from the verifier re-evaluating the family's predicates against
    the parameters the run actually realized, which is the route that exists for
    exactly this case.
    """
    run = runs[instance_id]
    assert run.result.status is VerificationStatus.OUT_OF_VALIDITY
    assert run.record.status is RunStatus.SUCCEEDED
    assert run.record.refusal is None
    assert not run.result.validity.observed.is_inside
    # A signed, normalized margin, so a sampler could find the boundary. A
    # boolean would say only that it is outside.
    assert any(m.margin < 0.0 for m in run.result.validity.margins)


def test_a_dropped_quadrature_weight_names_the_code_it_dropped_it_for(runs) -> None:
    """The report exists, which is what separates this from a silent hazard.

    ``_ray_quadrature_weight`` catches ``NON_HEXAPOLAR_SAMPLING`` and falls back
    to the unweighted amplitude mapping, and the fallback is right: a legitimately
    vignetted hexapolar fan reaches the same condition and must still be usable.
    What it must not do is drop the weight without saying why -- "unavailable"
    alone means both "this record predates the weight" and "your sampling is not
    a hexapolar fan", which are a missing declaration and an out-of-validity
    request. CHE-108 found the reason being computed and discarded.
    """
    run = runs["B0-PATCH-01"]
    reported = {d["code"]: d["detail"] for d in run.record.diagnostics}
    assert "QUADRATURE_WEIGHT_NOT_APPLIED" in reported
    assert "NON_HEXAPOLAR_SAMPLING" in reported["QUADRATURE_WEIGHT_NOT_APPLIED"]
    assert "REPORTED_RATHER_THAN_REFUSED" in reported


def test_the_curvature_guard_is_live_at_the_instance_parameters(runs) -> None:
    """The bound is computed by shipping code, and the guard does refuse.

    Two things, because they are two questions. ``check_patch`` asks whether the
    geometry's error is inside the caller's tolerance and PASSES at the
    instance's generous 0.2 rad; asked for half the bound it refuses with a
    remedy naming the widest admissible patch. The family asks whether the
    declared ``eps_curv`` is inside what the geometry admits, and it is not.
    Conflating the two is how a validity claim gets made backwards.
    """
    run = runs["B0-VALIDITY-01"]
    reported = {d["code"]: d["detail"] for d in run.record.diagnostics}
    assert "SI_S9_BOUND" in reported
    assert "arcsin(D/2R)" in reported["SI_S9_BOUND"]
    assert "crossed = True" in reported["SI_S9_BOUND"]
    assert "GUARD_PASSES_AT_THE_DECLARED_THRESHOLD" in reported
    assert "GUARD_REFUSES_A_TIGHTER_THRESHOLD" in reported
    assert "Use a patch no wider than" in reported["GUARD_REFUSES_A_TIGHTER_THRESHOLD"]


def test_the_device_disagreement_is_read_off_the_array(runs) -> None:
    """Requested cuda, actual cpu -- detected, not reported as a CUDA run.

    The observation on the record carries both sides, and ``honoured`` is false.
    A run that copied the request into the actual field would report a successful
    accelerator run that happened on the host.
    """
    run = runs["B0-DEVICE-02"]
    assert run.result.status is VerificationStatus.INVALID_CONFIGURATION
    assert "REPRESENTATION_INCONSISTENT" in run.record.contract_codes
    observation = run.record.device_precision
    assert observation is not None
    assert observation.requested_device == "cuda"
    assert observation.actual_device == "cpu"
    assert not observation.honoured


# --------------------------------------------------------------------------- #
# The loss is a number
# --------------------------------------------------------------------------- #


def test_the_precision_loss_is_measured_against_an_independent_oracle(runs) -> None:
    """B0-DTYPE-01, and CHE-107's WAVE-2 in the same measurement.

    The oracle is the float64 angular spectrum in ``verification/asm_oracle.py``,
    which shares no code with Chromatix -- so what is measured is the cost of the
    representation rather than the cost of the implementation. Measuring it with
    a second Chromatix call would put the truncation on both sides.
    """
    run = runs["B0-DTYPE-01"]
    assert run.result.status is VerificationStatus.LOSSY_BUT_ALLOWED

    loss = next(m for m in run.result.physics_accuracy if m.metric == "measured_precision_loss")
    assert loss.measured.value > 0.0, "a truncation that cost nothing did not happen"
    assert loss.measured.uncertainty is not None
    assert loss.met is True, (
        f"{loss.measured.value:.6g} against a tolerance of {loss.tolerance:.6g}"
    )

    observation = run.record.device_precision
    assert observation is not None
    assert observation.measured_loss_relative == pytest.approx(loss.measured.value)
    assert observation.measured_loss_basis is not None
    assert "eps32" in observation.measured_loss_basis


def test_the_measured_loss_sits_under_one_eps32_per_radian(runs) -> None:
    """The tolerance basis, evaluated rather than quoted.

    ``eps32 * 2*pi*z/lambda`` is the representation floor for an absolute phase
    of that size. A measured loss above it would mean the propagation is losing
    more than the dtype explains, which is an implementation finding rather than
    a precision one.
    """
    import math

    import numpy as np

    run = runs["B0-DTYPE-01"]
    parameters = run.instance.parameters
    accumulated = (
        2.0
        * math.pi
        * float(parameters["propagation_distance_m"])
        / float(parameters["wavelength_m"])
    )
    bound = float(np.finfo(np.float32).eps) * accumulated

    loss = next(m for m in run.result.physics_accuracy if m.metric == "measured_precision_loss")
    assert loss.measured.value < bound, (
        f"{loss.measured.value:.6e} exceeds the eps32-per-radian floor {bound:.6e} "
        f"over {accumulated:.4f} rad; that is not a dtype cost"
    )


def test_safe_refuses_the_crossing_that_allow_downcast_records(runs) -> None:
    """Both halves of the pair, because neither alone is the claim."""
    run = runs["B0-DTYPE-01"]
    codes = {d.get("code") for d in run.record.diagnostics}
    assert "SAFE_REFUSES_THE_SAME_CROSSING" in codes
    assert "BRIDGE_PLAN" in codes


# --------------------------------------------------------------------------- #
# The two that run clean
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("instance_id", ["B0-UNITS-01", "B0-UNITS-02"])
def test_a_silent_hazard_reports_ok_and_fails_its_gate(runs, instance_id: str) -> None:
    """The point of B0 being a category rather than a test file.

    The contract status is ``ok`` -- there is nothing for the boundary layer to
    complain about -- and the physics metric is rejected by the family's own
    tolerance. A suite that stopped at the first half would call both of these
    passing runs.
    """
    run = runs[instance_id]
    assert run.result.status is VerificationStatus.OK
    assert run.record.status is RunStatus.SUCCEEDED
    assert run.result.contract_status.codes == []

    error = next(
        m for m in run.result.physics_accuracy if m.metric == "relative_error_vs_closed_form"
    )
    assert error.tolerance_may_gate
    assert error.met is False, (
        f"{instance_id} measured {error.measured.value:.6g} against a gate of "
        f"{error.tolerance:.6g}. A known-wrong configuration that passes the gate "
        "means the gate cannot see this class of defect."
    )


@pytest.mark.parametrize("instance_id", ["B0-UNITS-01", "B0-UNITS-02"])
def test_the_silent_hazard_is_named_on_the_result(runs, instance_id: str) -> None:
    """So a consumer can tell "ok because it is right" from "ok because nothing checked"."""
    run = runs[instance_id]
    assert run.result.contract_status.silent_hazard_ids, instance_id


def test_the_coating_is_indistinguishable_from_bare_glass(runs) -> None:
    """Why nobody notices B0-UNITS-01, as a measured number.

    A 1000x-too-thick quarter-wave layer does essentially nothing, so the
    reported reflectance sits next to the uncoated one -- and a reader checking
    that "the coating gives a small reflectance" sees what they expected. The
    error bar on the gated metric is exactly this separation.
    """
    run = runs["B0-UNITS-01"]
    error = next(
        m for m in run.result.physics_accuracy if m.metric == "relative_error_vs_closed_form"
    )
    # The distance from bare glass, carried as the uncertainty.
    assert error.measured.uncertainty is not None
    assert error.measured.uncertainty < 1e-2, (
        "if this ever grows, the trap has stopped being silent and the family's "
        "premise needs revisiting"
    )
    # And the distance from the CORRECT answer, which is what the gate rejects.
    assert error.measured.value > 1.0


def test_the_kykx_sign_inversion_is_located_on_the_propagator(runs) -> None:
    """A correction to the inherited hazard, measured on the pinned install.

    ``verification/hazards.py`` records one number and describes it as 2*pi too
    small *and* sign-flipped. The fresh measurement finds two different mistakes
    at two different call sites: ``plane_wave`` handed cycles-per-length is 2*pi
    too small with the sign preserved, and ``asm_propagate``'s ``kykx``
    displaces opposite to its parameter. The magnitude of the recorded number
    reproduces; its attribution does not, and this is where that is written down.
    """
    run = runs["B0-UNITS-02"]
    codes = {d.get("code") for d in run.record.diagnostics}
    assert "THE_SIGN_INVERSION_IS_ON_THE_PROPAGATOR" in codes
    assert "MEASUREMENT_DIFFERS_FROM_THE_INHERITED_RECORD" in codes
    assert "THE_FACTOR_IS_ONLY_2PI_IN_THE_PARAXIAL_LIMIT" in codes


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_every_record_is_keyed_to_the_instance_it_was_produced_for(runs) -> None:
    """A record verified against another instance would measure a different run."""
    for instance_id, run in runs.items():
        assert run.record.instance_id == instance_id
        assert run.record.instance_fingerprint == run.instance.fingerprint
        assert run.result.provenance.fingerprint_matched, instance_id


def test_the_scientific_fingerprint_is_stable_within_a_run(runs) -> None:
    """The projection is a function of the result, not of when it was hashed."""
    from verification.evidence import result_fingerprint

    for run in runs.values():
        assert result_fingerprint(run.result) == result_fingerprint(run.result)


def test_the_declared_controls_are_reported_as_not_run(runs) -> None:
    """Honesty about what this driver does not do.

    B0's negative controls are statements about a *suite* -- "check only that
    nothing raised, and observe that both hazards pass" -- so they are not
    executed per instance, and the verifier says ``not_run`` rather than
    implying coverage. A control reported as fired without having run would be
    the worst of the available outcomes.
    """
    for instance_id, run in runs.items():
        for control in run.result.negative_control_results:
            assert control.outcome is NegativeControlOutcome.NOT_RUN, (
                f"{instance_id}/{control.control_id}"
            )
