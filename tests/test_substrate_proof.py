"""The loop, closed on a real workload.

CHE-115 (M3.3), partial. Everything else in this repository tests one link:
``tests/test_executor*.py`` proves a graph runs, ``tests/test_verifier.py``
proves a hand-built record is interpreted correctly, and the family tests prove
the declarations are well formed. This file is the only place where a real
Optiland trace becomes a real ``ExecutionRecord`` and that record is handed to
``verify()`` against a real family.

What it asserts is deliberately not "the number is right". The number is
``1.18e-2`` and the frozen gate is ``1.0e-3``, and the two are not comparable --
different ray count, and a different window definition. What is asserted is that
each link carries what the next one needs, and that the verifier tells the truth
about a run with unexercised controls and no convergence ladder.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from core.execution import RunStatus
from core.execution_record import NodeOutcome
from core.paths import repository_root
from verification.families.b3_composed import B3_PSF_SINGLET
from verification.result import DiagnosticCode, NegativeControlOutcome
from verification.status import VerificationStatus

pytestmark = [pytest.mark.integration, pytest.mark.optiland, pytest.mark.chromatix]


def _driver():
    """Load the instance driver by path.

    ``benchmarks/`` is not an importable package and deliberately stays that
    way: pytest scans it, and turning it into one would change what collection
    does with ninety probe modules for the sake of one import here.
    """
    module_name = "b3_psf_singlet_driver"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repository_root() / "benchmarks" / "instances" / "b3_psf_singlet.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def proof():
    return _driver().run_and_verify(seed=1)


def test_the_instance_and_the_record_describe_the_same_computation(proof) -> None:
    """The fingerprint travels with the run. A record verified against an
    instance it was not produced for would compare two different computations
    and report a number about neither."""
    record, result = proof
    assert record.status is RunStatus.SUCCEEDED
    assert record.instance_fingerprint == _driver().canonical_instance().fingerprint
    assert result.provenance.fingerprint_matched
    assert DiagnosticCode.INSTANCE_FINGERPRINT_MISMATCH not in result.diagnostic_codes()


def test_all_three_stages_executed(proof) -> None:
    record, _ = proof
    assert [n.node_id for n in record.nodes] == ["lens", "pupil_reconstruction", "wave"]
    assert all(n.outcome is NodeOutcome.EXECUTED for n in record.nodes)


def test_the_verifier_reads_the_record_rather_than_a_committed_file(proof) -> None:
    """Records are provenance, not oracles. The result's run id is this run's."""
    record, result = proof
    assert result.run_id == record.run_id
    assert result.family_id == B3_PSF_SINGLET.family_id
    assert result.instance_id == "B3-PSF-SINGLET-01"


def test_the_measured_residual_is_reported_with_an_uncertainty(proof) -> None:
    """Every reported number carries an error bar. Here it is the window
    sensitivity rather than a floating-point floor, because the frozen grid puts
    2.44 pixels across the Airy radius and is not converged for radius-like
    quantities -- quoting round-off would claim a precision the sampling does
    not support."""
    from verification.result import UncertaintyBasis

    _record, result = proof
    residual = next(
        m for m in result.physics_accuracy if m.metric == "fft_oracle_intensity_relative_l2"
    )
    assert residual.measured.uncertainty is not None
    assert residual.measured.uncertainty_basis is UncertaintyBasis.GRID_CONVERGENCE
    assert residual.measured.value > 0.0


def test_the_gate_is_reported_unmet_rather_than_widened(proof) -> None:
    """1.18e-2 against a frozen 1.0e-3. The tolerance is not touched.

    This is NOT the frozen 2.21e-3 and this test does not claim it is: the run
    is 256 rings rather than 512, and the metric is a centred half window rather
    than the 5-Airy-radius radial-profile residual the frozen gate uses. Both
    differences are declared on the instance and in the module docstring, and
    reproducing the fingerprint is the rest of CHE-115.
    """
    _record, result = proof
    residual = next(
        m for m in result.physics_accuracy if m.metric == "fft_oracle_intensity_relative_l2"
    )
    assert residual.tolerance == 1.0e-3
    assert residual.tolerance_may_gate
    assert residual.met is False
    assert residual.measured.value > residual.tolerance


def test_the_oracle_independence_travels_into_the_result(proof) -> None:
    """O1 shares no code and no traced data with the coupler it judges, and the
    consumer must not have to look that up separately."""
    from verification.claim_ledger import Oracle, OracleIndependence

    _record, result = proof
    residual = next(
        m for m in result.physics_accuracy if m.metric == "fft_oracle_intensity_relative_l2"
    )
    assert residual.oracle is Oracle.ANALYTIC
    assert residual.oracle_independence is OracleIndependence.INDEPENDENT


def test_the_unexercised_controls_make_the_gate_untrustworthy(proof) -> None:
    """The result of this run is not "the gate failed"; it is "the gate failed
    and nothing established that it could have succeeded honestly".

    Four controls are declared and none was exercised. A result reporting a
    trustworthy gate here would be the green tick the whole structure refuses.
    """
    _record, result = proof
    assert not result.gate_is_trustworthy
    outcomes = {c.control_id: c.outcome for c in result.negative_control_results}
    assert set(outcomes) == {
        "opl-sign-flip",
        "inverted-quadrature-weight",
        "axis-transpose",
        "launch-phase-error",
    }
    assert all(o is NegativeControlOutcome.NOT_RUN for o in outcomes.values())
    assert DiagnosticCode.NEGATIVE_CONTROL_NOT_RUN in result.diagnostic_codes()


def test_a_single_point_is_not_reported_as_converged(proof) -> None:
    """The family declares a refinement dimension and this run carries no
    ladder, so the value is one point rather than a converged one."""
    _record, result = proof
    assert result.convergence.converged is None
    assert DiagnosticCode.CONVERGENCE_NOT_ESTABLISHED in result.diagnostic_codes()


def test_a_declared_metric_the_run_did_not_measure_is_reported_missing(proof) -> None:
    """O2 is declared as characterization evidence and this proof does not run
    it. Silence would read as though it had been measured and agreed."""
    _record, result = proof
    assert DiagnosticCode.METRIC_MISSING_FROM_RECORD in result.diagnostic_codes()
    measured = {m.metric for m in result.physics_accuracy}
    assert "o2_asm_intensity_relative_l2" not in measured


def test_the_run_is_reported_ok_which_is_not_a_pass(proof) -> None:
    """``status: ok`` means the run produced evidence the verifier could
    measure. It says nothing about whether the physics was right -- that is what
    the per-metric ``met`` flags are for, and one of them is False."""
    _record, result = proof
    assert result.status is VerificationStatus.OK
    assert result.unmet_gating_metrics


def test_the_validity_declared_matches_the_validity_observed(proof) -> None:
    """The verifier re-evaluates the family's predicates against the parameters
    the run realized, rather than trusting the instance."""
    _record, result = proof
    assert result.validity.declaration_holds
    assert DiagnosticCode.DECLARED_VALIDITY_DISAGREES_WITH_OBSERVED not in (
        result.diagnostic_codes()
    )
