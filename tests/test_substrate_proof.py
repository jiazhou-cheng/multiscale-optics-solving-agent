"""The loop, closed on a real workload, at the frozen configuration.

CHE-115 (M3.3). Everything else in this repository tests one link:
``tests/test_executor*.py`` proves a graph runs, ``tests/test_verifier.py``
proves a hand-built record is interpreted correctly, and the family tests prove
the declarations are well formed. This file is the only place where a real
Optiland trace becomes a real ``ExecutionRecord`` and that record is handed to
``verify()`` against a real family.

And unlike its previous form, it now asserts that **the number is the frozen
one, bit for bit.** ``fft_oracle_intensity_relative_l2`` off the executor's
record equals ``0.0022072391812867093`` -- the value produced before this
substrate existed, by ``benchmarks/probes/quadrature_weight.py`` calling the
coupler directly. ``==``, not ``approx``: a substrate proof that only reproduced
its target to a tolerance would have moved a number and called it a migration.

Which arms run where, and why
-----------------------------
The frozen arm costs ~28 s, which is inside the default gate (the critical file
is ``test_executor_integration.py`` at ~31 s, so this changes the gate's wall
clock by nothing). The two further arms -- ``near_sensor_fine``, which is the
only one that gives ``M_WAVE_CHROMATIX`` a nonzero distance, and the
``opl_sign=-1`` negative control, which needs a second full 512-ring run to
compare against -- are each another ~25-30 s and carry ``slow``. That is the
marker's declared meaning: expensive numerical characterization, required before
merging a change to coupler numerics, which any change reaching this file is.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
import yaml

from core.execution import RunStatus
from core.execution_record import NodeOutcome
from core.paths import repository_root
from verification.evidence import result_fingerprint
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
    """The cheap arm: one 512-ring run, no negative control.

    ``with_control=False`` is what keeps this file inside the default gate. The
    control needs a second full 512-ring run and lives in the slow arm below,
    together with the committed record's fingerprint.
    """
    run = _driver().run_instance(with_control=False, seed=1)
    return run.record, run.result


def _residual(result):
    return next(
        m for m in result.physics_accuracy if m.metric == "fft_oracle_intensity_relative_l2"
    )


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------


def test_the_frozen_number_is_reproduced_bit_identically(proof) -> None:
    """The substrate proof's whole point, as one assertion.

    ``0.0022072391812867093`` was measured by a probe that builds the bundle,
    advances it, calls ``ray_to_wave`` and masks the result itself. This value
    comes off a ``GraphExecutor`` record produced from a committed YAML document.
    Equality, because they are the same float64 computation reached two ways --
    if they were not, the migration would have changed the physics and the right
    response is to find out why, not to widen this to ``approx``.
    """
    _record, result = proof
    assert _residual(result).measured.value == _driver().FROZEN_OBSERVED


def test_the_family_and_the_driver_agree_on_what_the_frozen_number_is(proof) -> None:
    """The target is not restated independently of the thing that carries it.

    ``FROZEN_OBSERVED`` is written down in the driver so the assertion above does
    not read its expected value out of the object it is testing. That only works
    while the two agree, so they are pinned.
    """
    disposition = B3_PSF_SINGLET.gate_disposition
    assert disposition is not None
    assert disposition.observed == _driver().FROZEN_OBSERVED


def test_the_instance_declares_the_configuration_the_graph_executes() -> None:
    """A parameter on the instance that the graph does not use describes a run
    that did not happen. Every value that appears in both is asserted equal here,
    rather than being kept in step by whoever edits one of them."""
    driver = _driver()
    document = yaml.safe_load(driver.GRAPH_PATH.read_text(encoding="utf-8"))
    lens = next(node for node in document["nodes"] if node["id"] == "lens")
    wave = next(node for node in document["nodes"] if node["id"] == "wave")
    edge = next(e for e in document["edges"] if e["id"] == "sensor_reconstruction")
    parameters = driver.CANONICAL_PARAMETERS

    assert lens["config"]["num_rays"] == parameters["pupil_rings"]
    assert lens["config"]["wavelength"] * 1e-6 == parameters["wavelength_m"]
    assert lens["config"]["Hy"] == parameters["field_angle_rad"]
    assert wave["config"]["pad_width"] == parameters["pad_width"]
    assert edge["config"]["grid_n"] == parameters["grid_n"]
    # The frozen configuration reconstructs ON the sensor, which is what makes
    # the wave node's propagation zero and the gate a float64 measurement.
    assert edge["config"]["advance_to_z_m"] == wave["config"]["target_plane_z_m"]


# ---------------------------------------------------------------------------
# Each link carries what the next one needs
# ---------------------------------------------------------------------------


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
    assert [n.node_id for n in record.nodes] == ["lens", "sensor_reconstruction", "wave"]
    assert all(n.outcome is NodeOutcome.EXECUTED for n in record.nodes)


def test_the_verifier_reads_the_record_rather_than_a_committed_file(proof) -> None:
    """Records are provenance, not oracles. The result's run id is this run's."""
    record, result = proof
    assert result.run_id == record.run_id
    assert result.family_id == B3_PSF_SINGLET.family_id
    assert result.instance_id == "B3-PSF-SINGLET-01"


def test_the_device_placement_is_read_off_the_arrays_not_off_the_request(proof) -> None:
    """A run that asked for a device and got another one must say so.

    ``requested`` is not evidence of ``actual``: the executor reads the device
    and dtype off the produced artifact. Here both nodes are asked for the host
    and land on it, and the wave node's dtype is the interesting half -- it is
    asked for nothing in particular and Chromatix casts unconditionally, which is
    exactly the fiction a requested-only record would hide.
    """
    record, _ = proof
    observations = {n.node_id: n.device_precision for n in record.nodes}
    assert all(o is not None for o in observations.values())
    assert observations["sensor_reconstruction"].actual_device == "cpu"
    assert observations["sensor_reconstruction"].actual_dtype == "complex128"
    assert observations["wave"].actual_device == "cpu"
    assert observations["wave"].actual_dtype == "complex64"


# ---------------------------------------------------------------------------
# What the result says about itself
# ---------------------------------------------------------------------------


def test_the_measured_residual_is_reported_with_an_uncertainty(proof) -> None:
    """Every reported number carries an error bar. Here it is what routing the
    same field through the complex64 wave node moves the number by -- a measured
    property of this graph, not a quoted round-off floor."""
    from verification.result import UncertaintyBasis

    _record, result = proof
    residual = _residual(result)
    assert residual.measured.uncertainty is not None
    assert residual.measured.uncertainty_basis is UncertaintyBasis.FLOATING_POINT_FLOOR
    assert residual.measured.value > 0.0
    # The complex64 leg is worth ~7.5e-5 of the value, so the bar is neither zero
    # nor the size of the number.
    assert 0.0 < residual.measured.uncertainty < 0.1 * residual.measured.value


def test_the_gate_is_reported_unmet_rather_than_widened(proof) -> None:
    """2.2072e-3 against a frozen 1.0e-3. The tolerance is not touched."""
    _record, result = proof
    residual = _residual(result)
    assert residual.tolerance == 1.0e-3
    assert residual.tolerance_may_gate
    assert residual.met is False
    assert residual.measured.value > residual.tolerance


def test_the_oracle_independence_travels_into_the_result(proof) -> None:
    """O1 shares no code and no traced data with the coupler it judges, and the
    consumer must not have to look that up separately."""
    from verification.claim_ledger import Oracle, OracleIndependence

    _record, result = proof
    residual = _residual(result)
    assert residual.oracle is Oracle.ANALYTIC
    assert residual.oracle_independence is OracleIndependence.INDEPENDENT


def test_the_unexercised_controls_make_the_gate_untrustworthy(proof) -> None:
    """The result of this run is not "the gate failed"; it is "the gate failed
    and nothing in THIS run established that it could have succeeded honestly".

    Four controls are declared and this arm exercises none of them: the
    ``opl_sign`` control is a second 512-ring run and lives in the slow arm
    below, where the committed record is produced with it exercised. A result
    reporting a trustworthy gate here would be the green tick the whole structure
    refuses.
    """
    _record, result = proof
    assert not result.gate_is_trustworthy
    outcomes = {c.control_id: c.outcome for c in result.negative_control_results}
    assert set(outcomes) == {
        "opl-sign-flip",
        "uniform-weight-power-divergence",
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


# ---------------------------------------------------------------------------
# The arms that cost a second 512-ring run
# ---------------------------------------------------------------------------


def test_a_variant_is_a_different_graph_and_not_a_flag() -> None:
    """Cheap: no execution. A perturbed graph must be distinguishable from its
    baseline in the record, or a control and its control are one run.

    Also pins the refusal: an override naming an edge the graph does not have
    raises, because an override silently dropped for a typo produces a run
    identical to the unperturbed one that reports itself as a control.
    """
    from runtime.executor import graph_fingerprint
    from runtime.variants import VariantError, with_config_overrides

    driver = _driver()
    baseline = driver.load_graph()
    flipped = driver.opl_sign_flip_graph()
    assert graph_fingerprint(baseline) != graph_fingerprint(flipped)
    assert flipped.task_id == "B3-PSF-SINGLET-01/opl_sign_flip"
    # The baseline is not mutated by taking a variant of it.
    assert "perturbation" not in baseline.edges[0].config

    with pytest.raises(VariantError):
        with_config_overrides(baseline, edges={"no_such_edge": {"grid_n": 8}})


@pytest.mark.slow
def test_the_power_divergence_control_fires() -> None:
    """CHE-117 (M4.2). The control that replaced ``inverted-quadrature-weight``.

    The retired control asserted that the production quadrature weight improves
    rel-L2 agreement with O1 by at least 1.2x, and measured 0.42. Its premise is
    false at convergence -- both configurations converge to the same residual --
    so no ray count makes it fire honestly. This one tests the property CHE-47's
    weight did establish: with the weight, reconstructed power is invariant under
    ray refinement; without it, it scales as ``(traced rays)^1.995``.

    The control is called directly rather than through ``run_instance``, which
    would add a 512-ring baseline this assertion does not use. Four traces at 64
    and 128 rings, about 8 s. Marked slow anyway: it is a second Optiland trace
    in a file the default gate already pays 30 s for.
    """
    from verification.result import NegativeControlOutcome as Outcome

    driver = _driver()
    outcomes = driver._power_divergence_control(driver.canonical_instance(), seed=1)
    control = outcomes["uniform-weight-power-divergence"]
    assert control.target_metric == "reconstructed_power_ray_doubling_excess"
    threshold = B3_PSF_SINGLET.tolerance_for(
        "reconstructed_power_ray_doubling_excess"
    ).threshold
    assert control.baseline is not None and control.baseline.value < threshold
    assert control.mutated is not None and control.mutated.value > threshold
    # Not "differs from the baseline": the mutated arm has to cross the gate's
    # own threshold, and by a margin nobody could mistake for round-off. The
    # weight's absence is CHE-33's (traced rays)^1.995 divergence, so this is
    # four orders, not a few percent.
    assert control.mutated.value / control.baseline.value > 1e3
    assert control.outcome is Outcome.FIRED


@pytest.mark.slow
def test_the_wave_node_does_real_work_in_the_near_sensor_fine_variant() -> None:
    """The frozen configuration's wave node propagates zero distance. A three-node
    graph whose middle node is an identity has demonstrated two nodes, so the same
    document is run with the reconstruction 0.001 R upstream -- CHE-38's own
    ``near_sensor_fine`` candidate -- and Chromatix propagates the residual.

    The agreement asserted is deliberately loose. CHE-38 section 7's padding
    sweep is the evidence for how much a residual post-handoff propagation moves
    this number, and this arm exists to show the node ran with real work, not to
    re-derive that sweep.
    """
    arm = _driver().run_near_sensor_fine(seed=1)
    assert arm["status"] == "succeeded"
    assert arm["propagation_m"] > 0.0
    record = arm["record"]
    wave = next(n for n in record.nodes if n.node_id == "wave")
    assert wave.outcome is NodeOutcome.EXECUTED
    # The propagated field is a different field from the reconstruction it came
    # from -- if these agreed exactly the propagation would have been a no-op.
    assert arm["terminal_relative_l2_vs_o1"] != arm["reconstruction_relative_l2_vs_o1"]
    assert arm["terminal_relative_l2_vs_o1"] < 0.5


@pytest.mark.slow
def test_the_opl_sign_control_fires_and_the_committed_record_reproduces() -> None:
    """The family declares ``opl-sign-flip``; this runs it through the executor,
    and re-derives the committed record's scientific fingerprint from the result.

    Negating the declared OPL conjugates the wavefront -- a converging pupil
    field becomes diverging -- and the residual against O1 must exceed 0.5, or
    the metric cannot tell a scrambled wavefront from a converging one and no
    passing value it reports means anything. Decided against O1 only: O2 is our
    own propagator built from the same traced pupil.
    """
    from verification.result import NegativeControlOutcome as Outcome

    driver = _driver()
    run = driver.run_instance(with_control=True, seed=1)
    control = next(
        c for c in run.result.negative_control_results if c.control_id == "opl-sign-flip"
    )
    assert control.baseline is not None and control.baseline.value == driver.FROZEN_OBSERVED
    assert control.mutated is not None and control.mutated.value > 0.5
    assert control.outcome is Outcome.FIRED

    committed = json.loads(
        (repository_root() / "benchmarks" / "instances" / "records"
         / f"{driver.INSTANCE_ID}.json").read_text(encoding="utf-8")
    )
    assert run.instance.fingerprint == committed["instance_fingerprint"]
    assert result_fingerprint(run.result) == committed["scientific_fingerprint"], (
        "the substrate proof re-ran to a different scientific fingerprint. Identify "
        "why the measurement moved and regenerate the record through the driver "
        "(`--write`) rather than editing it."
    )
