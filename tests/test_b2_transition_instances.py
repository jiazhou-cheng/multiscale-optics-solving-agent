"""The B2 transition families, executed. M2.1, M2.2 and M2.4's exit gate.

CHE-109, CHE-110, CHE-112. The physics under these was already tested and none of
it was a *benchmark*: no declared oracle, no tolerance with a basis, no error
budget, and no place a route-selection decision could be read off. This asserts
the measurements, and asserts what each one is decided by.

Four constructions had to be corrected to get here, and every one of them had
produced a plausible number
-----------------------------------------------------------------------------
* **The reconstruction owes the 1/N of SI eq S5** for a bundle sampled from a
  spectrum. Without it the field is scaled by the mode count and the round trip
  read 0.995 for a correct round trip.

* **Enumerating a MAGNITUDE density is not the exactness limit.** With
  ``p ~ |U~|`` each mode is weighted by ``1/p[m]`` and the sum is not the field;
  with a uniform density ``1/p`` is the constant mode count and the enumeration
  is exact.

* **Unbiasedness must be measured on a signed LINEAR functional, and not against
  the field itself.** ``<U, U>`` is real and positive by construction, so its
  imaginary part is identically zero -- and comparing an identically-zero
  component to its own float64 round-off spread read 5 to 23 sigma for an
  estimator that is exactly unbiased. Measured across three sample counts and
  two ensemble sizes.

* **A centred real probe cannot see a phase flip or a transpose.** The
  round-trip probe was a centred real Gaussian, whose spectrum is real and
  Hermitian-symmetric, so both broken twins read identically to the correct arm
  at 5.4e-16. A round trip that cannot be made to fail proves nothing, and that
  is exactly how it happens.

Two tolerance bases were completed rather than moved, and both change how the
results read: the round trip's 1e-12 is the ENUMERATED arm's floor, and
``detection_margin`` is not gateable by a one-sided ``measured <= threshold``
schema at all.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Mapping

import pytest

from core.paths import repository_root
from verification.result import NegativeControlOutcome
from verification.status import VerificationStatus

pytestmark = [pytest.mark.integration, pytest.mark.coupler]

# CHE-140: twenty of the tests below additionally carry `slow`, so the default
# suite does not run them. They are exactly the ones that read a `B2-W2R-STOCH`
# run out of `runs` -- identified by measurement, not by section heading, which is
# why `test_no_round_trip_is_accepted_without_a_failing_twin` is among them and
# sits under B2-ROUNDTRIP. That family's evidence is a single 61 s body of
# compute (see `_LazyRuns`); nothing about it was made cheaper or weaker here,
# and `make test-slow` runs it. What stays per-PR is the twenty-one items that
# never touch it: the four-conventions gate, the whole R2W route budget, and the
# round-trip directions with their failing twins.


def _driver():
    name = "b2_transitions_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b2_transitions.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _LazyRuns(Mapping):
    """``run_all()``, except an instance is executed when it is first looked up.

    CHE-140. ``run_all()`` is a comprehension over ``run_instance``, so this is
    the same mapping -- but eagerly building it made every test in this file pay
    for every family. Measured per instance: ``B2-W2R-STOCH-01`` is 60.8 s and
    each of the other twenty-four is under 0.6 s, because the stochastic family
    computes one shared body of evidence (a six-point convergence ladder to
    N=80000, an eight-seed unbiasedness ensemble, a five-control battery and a
    two-spectrum variance study) and the remaining seven STOCH instances read it
    back out of the driver's own cache.

    So the 60 s was being charged to the twenty-two tests in this file that never
    touch a STOCH run -- the four-conventions gate, the whole route budget and
    the whole round-trip family, all of which are the cheap ones. Those keep
    running per-PR; the nineteen that do need the sweep carry ``slow`` and are
    selected by ``make test-slow``.

    ``keys()`` deliberately does not execute anything, which is what lets
    ``test_every_declared_instance_was_executed`` stay a coverage check on the
    *declaration* rather than a reason to run the sweep.
    """

    def __init__(self, driver):
        self._driver = driver
        self._ids = tuple(driver.declared_instance_ids())
        self._cache: dict[str, object] = {}

    def __getitem__(self, instance_id: str):
        if instance_id not in self._ids:
            raise KeyError(instance_id)
        if instance_id not in self._cache:
            self._cache[instance_id] = self._driver.run_instance(instance_id)
        return self._cache[instance_id]

    def __iter__(self):
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


@pytest.fixture(scope="module")
def runs():
    return _LazyRuns(_driver())


def _metric(run, name):
    return next(m for m in run.result.physics_accuracy if m.metric == name)


def _control(run, name):
    return next(c for c in run.result.negative_control_results if c.control_id == name)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_declared_instance_was_executed(runs) -> None:
    assert set(runs) == set(_driver().declared_instance_ids())
    assert len(runs) == 25, sorted(runs)


@pytest.mark.slow
def test_all_four_families_are_covered(runs) -> None:
    assert {run.family.family_id for run in runs.values()} == {
        "B2-R2W-EXACT",
        "B2-R2W-ROUTE",
        "B2-W2R-STOCH",
        "B2-ROUNDTRIP",
    }


# --------------------------------------------------------------------------- #
# B2-R2W-EXACT
# --------------------------------------------------------------------------- #


def test_the_exact_route_pins_four_conventions_at_once(runs) -> None:
    """SI Figure S1c, as a gate.

    Once each ray's OPL compensates its launch position, every ray of a
    collimated bundle contributes the SAME plane wave -- so the sum is
    ``N exp(i k d.r)`` with no residual position dependence. Remove the OPL
    compensation, the ``Delta-r`` ramp, the phasor sign or the projection factor
    and that identity breaks, which is why ONE comparison decides four
    conventions.
    """
    run = runs["B2-R2W-EXACT-01"]
    metric = _metric(run, "exactness_relative_l2_field")
    assert metric.met is True, f"{metric.measured.value:.3e}"
    assert metric.measured.value < 1e-13

    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "FOUR_CONVENTIONS_AT_ONCE"
    )
    for term in ("opl-and-ramp", "phasor-sign", "axis-transpose", "projection-factor"):
        assert term in detail, term

    control = _control(run, "dropped-term")
    assert control.outcome is NegativeControlOutcome.FIRED
    # Reported at the WEAKEST of the four, so the control is stated at its least
    # favourable rather than at the one that separates best. That is
    # projection-factor at 4.70e-05 -- the smallest of the four and still 3.2e10
    # above the gate, while the other three are O(1).
    assert control.mutated.value > 1e-5
    assert control.mutated.value / metric.tolerance > 1e6


def test_the_exactness_tolerance_is_derived_from_the_dtype(runs) -> None:
    """CHE-109 asks for derived, not chosen, and the derivation is recorded.

    ``sqrt(N) eps64`` for an N-term unit-modulus sum plus ``eps64`` per radian of
    the largest phase argument. The declared 1e-12 gate is the looser of the two,
    and the measurement is reported against both.
    """
    run = runs["B2-R2W-EXACT-01"]
    detail = next(
        d["detail"]
        for d in run.record.diagnostics
        if d["code"] == "TOLERANCE_DERIVED_FROM_THE_DTYPE"
    )
    assert "eps64" in detail
    assert "1e-12" in detail


def test_on_node_alignment_makes_the_splat_a_relabelling(runs) -> None:
    """And it is the CONDITION under which the two routes may be compared exactly.

    Every ray's transverse wavevector is an exact bin of the k-grid, so the
    bilinear splat weights collapse to (1, 0). Half a bin off, the same route
    interpolates -- which is the family's own declared control.
    """
    run = runs["B2-R2W-EXACT-01"]
    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "ON_NODE_IS_A_RELABELLING"
    )
    assert "on_node_fraction" in detail

    control = _control(run, "off-node-is-not-exact")
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.baseline.value < 1e-12 < control.mutated.value


def test_the_static_shape_guarantee_is_structural_and_says_so(runs) -> None:
    """Declared NOT_IMPLEMENTED as a run-time control, on purpose.

    The property is that no rays x pixels factor may be formed, and it is
    asserted by making ``xp.outer`` and ``xp.einsum`` raise -- which survives a
    host change where a wall-clock comparison would not. Reporting it as a
    measured control would be claiming a measurement that is not the evidence.
    """
    control = _control(runs["B2-R2W-EXACT-01"], "static-shape-violation")
    assert control.outcome is NegativeControlOutcome.NOT_RUN
    assert "structural" in control.note
    assert "xp.outer" in control.note


def test_the_pupil_power_invariant_holds(runs) -> None:
    invariant = next(
        i
        for i in runs["B2-R2W-EXACT-01"].result.invariant_results
        if i.invariant_id == "PUPIL_POWER_CONSISTENCY"
    )
    assert invariant.met


# --------------------------------------------------------------------------- #
# B2-R2W-ROUTE: the budget
# --------------------------------------------------------------------------- #


def _route(runs, tag: str, oversampling: int):
    return runs[f"B2-R2W-ROUTE-{tag}-{oversampling:02d}"]


def test_the_route_budget_is_measured_on_both_systems(runs) -> None:
    """Four oversampling values on each, and the ASYMMETRY is the finding.

    On an on-node system the splat is a relabelling and the route is exact at
    every oversampling; on an off-node one it interpolates. Averaging the two
    would report a number describing neither, which is why ``system`` is a
    PHYSICAL parameter of the family.
    """
    for oversampling in (1, 2, 4, 8):
        on_node = _metric(
            _route(runs, "ONNODE", oversampling), "route_field_relative_l2"
        ).measured.value
        assert on_node < 1e-14, f"{oversampling}x on-node reads {on_node:.3e}"

    off_node = [
        _metric(_route(runs, "OFFNODE", k), "route_field_relative_l2").measured.value
        for k in (1, 2, 4, 8)
    ]
    # Monotone improvement, and by more than an order over the ladder.
    assert off_node == sorted(off_node, reverse=True)
    assert off_node[0] / off_node[-1] > 10.0


def test_the_on_node_fraction_is_measured_rather_than_assumed(runs) -> None:
    """So a route CLAIMING on-node status while dropping rays is visible."""
    on_node = next(
        d["detail"]
        for d in _route(runs, "ONNODE", 8).record.diagnostics
        if d["code"] == "ON_NODE_FRACTION_IS_MEASURED"
    )
    assert "1.000000000" in on_node
    off_node = next(
        d["detail"]
        for d in _route(runs, "OFFNODE", 8).record.diagnostics
        if d["code"] == "ON_NODE_FRACTION_IS_MEASURED"
    )
    assert "0.000000000" in off_node


def test_ncc_cannot_see_the_power_the_route_loses(runs) -> None:
    """The blindness, measured. At 1x the route loses 5% and NCC reads 0.9989.

    NCC is normalized, so it is blind to absolute scale by construction -- not by
    accident and not fixably -- which is why both are reported and why a gate on
    NCC alone would call the coarse route fine.
    """
    coarse = _route(runs, "OFFNODE", 1)
    power = _metric(coarse, "route_power_ratio").measured.value
    ncc = _metric(coarse, "route_ncc").measured.value
    assert power > 0.04, f"the coarse route should lose real power, not {power:.3e}"
    assert 1.0 - ncc < power / 10.0, (
        f"NCC reads {ncc:.6f} where the power ratio is off by {power:.3e}: the "
        "blindness is the point of this assertion"
    )
    assert _control(coarse, "ncc-alone-would-have-passed-it").outcome is (
        NegativeControlOutcome.FIRED
    )


def test_the_residual_grows_off_axis(runs) -> None:
    """The splat kernel's signature, and a centred metric cannot see it.

    CHE-44's concern in this coordinate: the whole-grid residual against the
    centred-window one is a factor, and reporting only the centred number would
    understate the error where the kernel is worst.
    """
    detail = next(
        d["detail"]
        for d in _route(runs, "OFFNODE", 1).record.diagnostics
        if d["code"] == "OFF_AXIS_RESIDUAL_GROWTH"
    )
    assert "factor of" in detail
    growth = float(detail.split("a factor of ")[1].split(".")[0] + "." + detail.split("a factor of ")[1].split(".")[1][:4])
    assert growth > 2.0, f"off-axis growth {growth}"


def test_the_route_controls_are_not_run_where_they_would_be_noise(runs) -> None:
    """On an exact route there is nothing for either control to demonstrate.

    Reporting them as fired there would be reporting float64 summation order as
    evidence, and NOT_RUN with the reason is the honest state.
    """
    for oversampling in (1, 2, 4, 8):
        run = _route(runs, "ONNODE", oversampling)
        for name in ("ncc-alone-would-have-passed-it", "oversampling-does-not-help"):
            control = _control(run, name)
            assert control.outcome is NegativeControlOutcome.NOT_RUN
            assert "EXACT on this system" in control.note


def test_oversampling_buys_accuracy_only_where_there_is_error_to_buy(runs) -> None:
    control = _control(_route(runs, "OFFNODE", 8), "oversampling-does-not-help")
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.mutated.value > control.baseline.value


def test_the_route_budget_does_not_claim_the_paper_scale_numbers(runs) -> None:
    """Honesty about what this measures.

    The recorded 1.07e-2 field error and 1.7% power loss at 8x come from a 60M-ray
    demo3 run that stays a probe. What is reproduced here is the SHAPE of the
    budget on a tractable off-node system, and the gate disposition says so.
    """
    note = runs["B2-R2W-ROUTE-OFFNODE-08"].family.gate_disposition.note
    assert "NOT the paper-scale demo3 numbers" in note
    assert "B4-DEMO3" in note


# --------------------------------------------------------------------------- #
# B2-W2R-STOCH: four kinds, in order
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_four_evidence_kinds_are_present_in_order(runs) -> None:
    """And the ORDER is enforced by construction rather than documented.

    Each stage reads the previous stage's result out of the dict it is building,
    so a run that tried to report a fitted exponent without an exactness limit
    would raise rather than produce a partial record. An estimator that is wrong
    in the enumeration limit has a transform defect, and tuning N would be
    beside the point.
    """
    run = runs["B2-W2R-STOCH-01"]
    stochastic = run.result.stochastic_evidence
    assert stochastic.exactness_limit is not None
    assert stochastic.unbiasedness is not None
    assert stochastic.fitted_convergence_rate is not None
    assert stochastic.variance_by_sampling_density

    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "FOUR_KINDS_IN_ORDER"
    )
    assert detail.index("1 exactness limit") < detail.index("2 unbiasedness")
    assert detail.index("2 unbiasedness") < detail.index("3 fitted exponent")
    assert detail.index("3 fitted exponent") < detail.index("4 variance advantage")


@pytest.mark.slow
def test_the_exactness_limit_comes_first_and_is_exact(runs) -> None:
    metric = _metric(runs["B2-W2R-STOCH-01"], "enumeration_limit_relative_l2")
    assert metric.met is True
    assert metric.measured.value < 1e-14


@pytest.mark.slow
def test_unbiasedness_is_gated_against_the_measured_standard_error(runs) -> None:
    """Not against a chosen field-space constant.

    And measured on a SIGNED linear functional against a fixed independent probe
    vector, because ``<U, U>`` is real and positive by construction: its
    imaginary part is identically zero, and comparing that to its own round-off
    spread read 5 to 23 sigma for an estimator that is exactly unbiased.
    """
    run = runs["B2-W2R-STOCH-01"]
    metric = _metric(run, "ensemble_mean_bias")
    assert metric.met is True, f"{metric.measured.value:.4f} sigma"
    assert metric.tolerance == 3.0
    assert "SIGNED overlap" in metric.measured.note
    assert "probe vector" in metric.measured.note

    stochastic = run.result.stochastic_evidence
    assert len(stochastic.seeds) >= 8, "the declared minimum, and the schema enforces it"
    assert stochastic.ensemble_standard_error is not None


@pytest.mark.slow
def test_the_convergence_exponent_is_fitted_over_six_points(runs) -> None:
    convergence = runs["B2-W2R-STOCH-01"].result.convergence
    assert len(convergence.ladder) >= 6
    assert convergence.expected_exponent == -0.5
    exponent = convergence.fitted_exponent
    assert exponent is not None
    assert exponent.value == pytest.approx(-0.5, abs=0.1), f"{exponent.value:+.4f}"
    assert convergence.converged is True
    assert exponent.uncertainty is not None


@pytest.mark.slow
def test_the_variance_advantage_is_reported_as_a_size(runs) -> None:
    """Not as a pass. Magnitude sampling exploits concentration and is merely
    comparable to uniform without it, and the SIZE of that difference is the
    property the paper's Figure 4 claim is about."""
    run = runs["B2-W2R-STOCH-01"]
    metric = _metric(run, "variance_at_sampling_density")
    assert not metric.tolerance_may_gate
    assert metric.measured.value > 1.5, "no advantage on a concentrated spectrum"
    densities = run.result.stochastic_evidence.variance_by_sampling_density
    assert "concentrated_advantage" in densities
    assert "multilobed_advantage" in densities
    assert densities["concentrated_advantage"] > densities["multilobed_advantage"], (
        "the whole claim is that concentration is what magnitude sampling exploits"
    )


@pytest.mark.slow
def test_the_negative_control_battery_is_five_deep(runs) -> None:
    """M2.2 asks for at least five, each with a passing unperturbed arm.

    Every one is a real switch on ``SamplingPerturbation`` or ``Perturbation``
    rather than a hand-written variant, and each reports its detection margin in
    sigma rather than a bare boolean.
    """
    run = runs["B2-W2R-STOCH-01"]
    outcomes = {c.control_id: c for c in run.result.negative_control_results}
    assert len(outcomes) >= 5, sorted(outcomes)
    for control_id, control in outcomes.items():
        assert control.outcome is NegativeControlOutcome.FIRED, control_id
    # The unperturbed arm passes, which is what makes a firing broken arm mean
    # something.
    unperturbed = outcomes["omitted-importance-weight"].baseline
    assert unperturbed is not None
    assert unperturbed.value < 3.0, f"the control arm is itself biased at {unperturbed.value}"


@pytest.mark.slow
def test_two_controls_needed_their_own_configuration_to_be_observable(runs) -> None:
    """The blind-spot lesson applied to the battery, not an exception to it.

    ``kn-sign`` is exactly inert at z = 0 and ``evanescent-cut`` is inert on a
    grid with no evanescent content. A control run where the term it removes is
    inert reports green and proves nothing.
    """
    outcomes = {c.control_id: c for c in runs["B2-W2R-STOCH-01"].result.negative_control_results}
    kn = outcomes["kn-sign"]
    assert "advanced to z" in (kn.mutated.note if kn.mutated else "")
    assert "inert" in (kn.mutated.note if kn.mutated else "")
    evanescent = outcomes["evanescent-cut"]
    assert "lambda/3" in (evanescent.mutated.note if evanescent.mutated else "")


@pytest.mark.slow
def test_the_kn_sign_control_is_a_refusal_rather_than_a_number(runs) -> None:
    """The strongest outcome a control can have.

    Reversing the normal component makes every ray travel away from the
    observation plane, and the shipping ``advance_bundle_to_plane`` declines to
    drop them: "a bundle that quietly loses members produces a plausible field
    with missing power". A refusal is stronger evidence than a numerical
    separation.
    """
    control = _control(runs["B2-W2R-STOCH-01"], "kn-sign")
    assert control.outcome is NegativeControlOutcome.FIRED
    assert "REFUSED" in (control.mutated.note if control.mutated else "")


@pytest.mark.slow
def test_the_three_blind_spots_are_measured(runs) -> None:
    """Each one a number, because the claim is quantitative: THIS configuration
    cannot see THAT term."""
    codes = {d["code"] for d in runs["B2-W2R-STOCH-01"].record.diagnostics}
    assert "BLIND_SPOT_A_PROJECTION_AT_NORMAL_INCIDENCE" in codes
    assert "BLIND_SPOT_B_OBLIQUE_RAMP_FOR_ONE_CENTRED_RAY" in codes
    assert "BLIND_SPOT_C_UNIFORM_SAMPLING_HIDES_THE_WEIGHT" in codes


@pytest.mark.parametrize(
    ("code", "phrase"),
    [
        ("BLIND_SPOT_A_PROJECTION_AT_NORMAL_INCIDENCE", "cannot prove projection handling"),
        ("BLIND_SPOT_B_OBLIQUE_RAMP_FOR_ONE_CENTRED_RAY", "cannot prove the ramp"),
        (
            "BLIND_SPOT_C_UNIFORM_SAMPLING_HIDES_THE_WEIGHT",
            "would certify the biased estimator",
        ),
    ],
)
@pytest.mark.slow
def test_each_blind_spot_states_what_it_makes_unprovable(runs, code: str, phrase: str) -> None:
    detail = next(
        d["detail"] for d in runs["B2-W2R-STOCH-01"].record.diagnostics if d["code"] == code
    )
    assert phrase in detail
    # And it states a measured number rather than only the reasoning: each blind
    # spot is the claim that THIS configuration cannot see THAT term, and the
    # size of the insensitivity is what makes it a measurement.
    assert any(part.replace(".", "").replace("e-", "").replace("e+", "").isdigit()
               for part in detail.replace(",", " ").split()), detail


@pytest.mark.slow
def test_blind_spot_c_shows_the_weight_becoming_a_pure_scale(runs) -> None:
    """Under uniform sampling p is constant, so omitting 1/p multiplies the field
    by that constant and nothing else -- and after rescaling the two agree to
    round-off. An NCC or any scale-invariant metric would certify the biased
    estimator, which is why the importance-weight control runs under MAGNITUDE
    sampling."""
    detail = next(
        d["detail"]
        for d in runs["B2-W2R-STOCH-01"].record.diagnostics
        if d["code"] == "BLIND_SPOT_C_UNIFORM_SAMPLING_HIDES_THE_WEIGHT"
    )
    before = float(detail.split("before rescaling ")[1].split(",")[0])
    after = float(detail.split("it is ")[1].split(",")[0])
    assert after < before / 100.0, (
        f"rescaling should remove essentially all of it: {before:.3e} -> {after:.3e}"
    )


@pytest.mark.slow
def test_the_evanescent_power_ledger_is_reported(runs) -> None:
    invariant = next(
        i
        for i in runs["B2-W2R-STOCH-01"].result.invariant_results
        if i.invariant_id == "EVANESCENT_POWER_ACCOUNTED"
    )
    assert invariant.met
    metric = _metric(runs["B2-W2R-STOCH-01"], "evanescent_power_fraction")
    assert "REPORTED rather than absorbed" in metric.measured.note


@pytest.mark.slow
def test_reproducibility_is_labelled_reproducibility(runs) -> None:
    """And is deliberately NOT one of the four evidence kinds.

    Bitwise reproducibility at a fixed seed is not evidence of accuracy, and the
    protocol says so. Recording it under its own name is the whole distinction.
    """
    detail = next(
        d["detail"]
        for d in runs["B2-W2R-STOCH-01"].record.diagnostics
        if d["code"] == "REPRODUCIBILITY_NOT_ACCURACY"
    )
    assert "not evidence of accuracy" in detail


@pytest.mark.slow
def test_the_surrogate_bias_is_measured_and_nothing_is_certified(runs) -> None:
    """``derivative.verified`` stays false, which is the deliverable.

    The paper states directly that holding the sampled wavevectors fixed during
    backpropagation and detaching the sampling density neglects the gradient
    contribution from the directions' own motion. So the estimator is
    deliberately biased and what is owed is a bounded, recorded figure.
    """
    from registry.loader import Registry

    detail = next(
        d["detail"]
        for d in runs["B2-W2R-STOCH-01"].record.diagnostics
        if d["code"] == "SURROGATE_GRADIENT_BIAS"
    )
    assert "derivative_verified': False" in detail or "derivative_verified\": false" in detail
    assert "NOT certified" in detail

    # And the registry still says so.
    registry = Registry.from_package()
    assert registry.couplers["C_WAVE_TO_RAY"].derivative.verified is False
    assert registry.couplers["C_WAVE_TO_RAY"].derivative.mode.value == "surrogate"


# --------------------------------------------------------------------------- #
# B2-ROUNDTRIP
# --------------------------------------------------------------------------- #


def _roundtrips(runs, arm: str):
    return {k: v for k, v in runs.items() if k.startswith("B2-ROUNDTRIP-") and arm in k}


def test_both_round_trip_directions_return_the_input(runs) -> None:
    """wave -> rays -> wave AND ray -> wave -> ray, at the enumeration limit."""
    for direction in ("WAVERAYWAVE", "RAYWAVERAY"):
        run = runs[f"B2-ROUNDTRIP-{direction}-ENUMERATED-00"]
        metric = _metric(run, "round_trip_relative_rms")
        assert metric.met is True, f"{direction}: {metric.measured.value:.3e}"
        assert metric.measured.value < 1e-14


@pytest.mark.slow
def test_no_round_trip_is_accepted_without_a_failing_twin(runs) -> None:
    """The schema rule, and the reason for it.

    A shared convention error cancels between the two directions, so a round trip
    that cannot be made to fail proves nothing about the pair. Every instance
    reports its twin, and the twin fires.
    """
    for instance_id, run in _roundtrips(runs, "ROUNDTRIP").items():
        for name in ("mismatched-phase-sign", "axis-transpose"):
            control = _control(run, name)
            assert control.outcome is NegativeControlOutcome.FIRED, f"{instance_id}/{name}"
            assert control.baseline.value < control.mutated.value
        assert run.record.observed_parameters["broken_twin_ran"] is True


def test_the_probe_field_can_see_a_sign_flip_and_a_transpose(runs) -> None:
    """CHE-44's concern answered by construction rather than audited after.

    The first version used a centred REAL Gaussian, whose spectrum is real and
    Hermitian-symmetric -- so conjugating it is a no-op and transposing it is a
    no-op, and both broken twins read identically to the correct arm at 5.4e-16.
    The probe is now offset, elliptical and phase-ramped.
    """
    run = runs["B2-ROUNDTRIP-WAVERAYWAVE-ENUMERATED-00"]
    phase = _control(run, "mismatched-phase-sign")
    transpose = _control(run, "axis-transpose")
    assert phase.mutated.value > 1.0, "a conjugated field should be O(1) away"
    assert transpose.mutated.value > 1.0
    note = run.family.gate_disposition.note
    assert "centred REAL Gaussian" in note


def test_the_monte_carlo_arm_reports_unmet_rather_than_being_exempted(runs) -> None:
    """A 1e-12 floor is the ENUMERATED arm's, and the basis now says so.

    The sampled arm measures 3e-2, which is sampling error and not a defect. Its
    claim is the ensemble and the twin. Reporting it UNMET against a gate that
    belongs to the other arm is more honest than quietly exempting it.
    """
    run = runs["B2-ROUNDTRIP-WAVERAYWAVE-MONTE_CARLO-01"]
    metric = _metric(run, "round_trip_relative_rms")
    assert metric.met is False
    assert 1e-3 < metric.measured.value < 1e-1
    tolerance = run.family.tolerance_for("round_trip_relative_rms")
    assert "ENUMERATED arm's" in tolerance.basis
    assert "not expected to" in tolerance.basis


def test_the_monte_carlo_arm_reports_ensemble_statistics(runs) -> None:
    """Because a single realization is never an accuracy result."""
    run = runs["B2-ROUNDTRIP-WAVERAYWAVE-MONTE_CARLO-01"]
    stochastic = run.result.stochastic_evidence
    assert len(stochastic.seeds) >= 3
    assert stochastic.ensemble_standard_error is not None
    assert stochastic.exactness_limit is not None
    assert stochastic.exactness_limit.value < 1e-14, (
        "the enumerated arm of the same direction is the exactness limit, and it is "
        "carried on the sampled arm's report rather than left in another file"
    )


def test_the_detection_margin_is_reported_and_not_gated(runs) -> None:
    """A schema limitation, named rather than worked around.

    ``MetricResult.met`` is ``measured <= threshold`` everywhere, so a quantity
    where LARGER IS BETTER cannot be expressed as a gating tolerance: gating a
    detection margin at <= 1e3 asserts the opposite of the claim. Inverting the
    number to fit the schema would hide the limitation in a metric name, so the
    margin is reported and the under-powered-control finding is carried by the
    control outcomes -- where the direction is the right way round.
    """
    run = runs["B2-ROUNDTRIP-WAVERAYWAVE-ENUMERATED-00"]
    metric = _metric(run, "detection_margin")
    assert not metric.tolerance_may_gate
    assert metric.measured.value > 1e12
    tolerance = run.family.tolerance_for("detection_margin")
    assert tolerance.may_gate is False
    assert "LARGER IS BETTER" in tolerance.basis
    assert "undermines_the_gate" in tolerance.basis


def test_the_ray_wave_ray_direction_is_compared_in_the_spectral_domain(runs) -> None:
    """Because no per-ray correspondence survives an accumulation.

    The outgoing amplitude is a spectral amplitude ``U~[m]/p[m]``, not a
    transformed incident weight, so the round trip cannot be compared ray by ray.
    What survives is the spectrum.
    """
    detail = next(
        d["detail"]
        for d in runs["B2-ROUNDTRIP-RAYWAVERAY-ENUMERATED-00"].record.diagnostics
        if d["code"] == "WHAT_DOES_NOT_SURVIVE"
    )
    assert "no per-ray correspondence" in detail
    assert "SPECTRAL domain" in detail


def test_the_phase_reference_invariant_holds_on_the_enumerated_arm(runs) -> None:
    invariant = next(
        i
        for i in runs["B2-ROUNDTRIP-WAVERAYWAVE-ENUMERATED-00"].result.invariant_results
        if i.invariant_id == "PHASE_REFERENCE_CONSISTENCY"
    )
    assert invariant.met


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_every_reported_number_carries_an_uncertainty_and_a_basis(runs) -> None:
    from verification.result import UncertaintyBasis

    for instance_id, run in runs.items():
        for metric in run.result.physics_accuracy:
            basis = metric.measured.uncertainty_basis
            if basis is UncertaintyBasis.NOT_ESTIMATED:
                assert metric.measured.uncertainty is None, (
                    f"{instance_id}/{metric.metric}: NOT_ESTIMATED must not carry a number"
                )
            else:
                assert metric.measured.uncertainty is not None, (
                    f"{instance_id}/{metric.metric}"
                )


@pytest.mark.slow
def test_every_record_is_keyed_to_its_instance(runs) -> None:
    for instance_id, run in runs.items():
        assert run.record.instance_id == instance_id
        assert run.record.instance_fingerprint == run.instance.fingerprint
        assert run.result.provenance.fingerprint_matched


@pytest.mark.slow
def test_the_family_gates_agree_with_what_was_measured(runs) -> None:
    """A gate recorded as MET whose instance measures worse than its tolerance is
    the failure mode a hand-maintained disposition always eventually has.

    Checked on the instance the disposition is ABOUT -- the exact one for the
    exactness families and the enumerated arm for the round trip -- because the
    other instances are declared to exceed the gate and are the family's own
    evidence for the budget.
    """
    from verification.claim_ledger import GateStatus

    about = {
        "B2-R2W-EXACT": "B2-R2W-EXACT-01",
        "B2-R2W-ROUTE": "B2-R2W-ROUTE-OFFNODE-08",
        "B2-W2R-STOCH": "B2-W2R-STOCH-01",
        "B2-ROUNDTRIP": "B2-ROUNDTRIP-WAVERAYWAVE-ENUMERATED-00",
    }
    for family_id, instance_id in about.items():
        run = runs[instance_id]
        disposition = run.family.gate_disposition
        assert disposition.status is GateStatus.MET, family_id
        metric = _metric(run, disposition.metric)
        assert metric.met is True, (
            f"{family_id} claims MET and {metric.metric} measures "
            f"{metric.measured.value:.3e} against {metric.tolerance:.3e}"
        )
        assert [r for r in disposition.evidence if "::" in r], family_id


@pytest.mark.slow
def test_the_declared_stochastic_minimum_seeds_are_actually_used(runs) -> None:
    """One realization is never an accuracy result -- for an instance that samples.

    An ENUMERATED arm is not one realization of anything: it sums every
    propagating mode, so it has no sampling error to average down and one seed is
    the complete measurement. Requiring three there would be requiring three
    copies of the same number. The split is asserted in both directions so an
    instance cannot get the exemption by merely being deterministic.
    """
    checked = 0
    for instance_id, run in runs.items():
        policy = run.family.stochastic_policy
        if not policy.is_stochastic:
            continue
        seeds = run.result.stochastic_evidence.seeds
        if run.instance.parameters.get("arm") == "enumerated":
            assert len(seeds) == 1, (
                f"{instance_id} enumerates, so extra seeds would be extra copies "
                f"of one number: {seeds}"
            )
            continue
        assert len(seeds) >= policy.minimum_seeds, f"{instance_id}: {seeds}"
        checked += 1
    assert checked >= 3, f"only {checked} sampling instances were checked"


def test_the_route_budget_ladder_is_not_silently_truncated() -> None:
    """Four rungs per system, stated, so a shortened ladder is a visible change."""
    driver = _driver()
    assert set(driver._ROUTE_SYSTEMS) == {"demo2_paper", "demo3_characterization"}
    for system in driver._ROUTE_SYSTEMS:
        assert set(driver._route_budget(system)) == {1, 2, 4, 8}
    assert not math.isnan(0.0)
