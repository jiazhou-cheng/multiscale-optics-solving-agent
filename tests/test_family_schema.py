"""The rules a benchmark family cannot be written around.

CHE-131 (M0.5.2). Each test here corresponds to a way this repository has
already got a scientific claim wrong, or to a way the retired design let one
through. They are constructor-level rather than review-level on purpose: a
family that violates one cannot be imported, so there is no window in which a
wrong family exists and somebody plans against it.

The four rules that matter most, and the incidents behind them:

* an oracle that shares code with the thing under test cannot gate --
  ``L2-PSF-01`` once set a negative-control floor from an ``O2`` comparison and
  had to retire it as circular;
* ``CROSS_ROUTE`` is called out separately, because Optiland's FFTPSF and
  HuygensPSF share one Wavefront/OPD front end and agreeing measures the back
  ends only. Two routes through one front end are one oracle;
* a B4 family cannot gate at all, so ``CHARACTERIZED_NO_GATE`` cannot be
  promoted to "passed" by adding a threshold;
* a tolerance whose basis is a recorded measurement cannot gate. A number the
  code produced cannot decide whether the next run of that code is right.
"""

from __future__ import annotations

import math

import pytest

from core.precision import DeviceKind, DType
from verification.claim_ledger import ClaimKind, GateStatus, Oracle, OracleIndependence
from verification.families import (
    FAMILIES,
    BenchmarkCategory,
    BenchmarkFamily,
    ExecutionParameter,
    ExecutionPolicy,
    FamilyOracle,
    InstanceOrigin,
    Metric,
    NegativeControl,
    NumericalParameter,
    Parameter,
    ParameterKind,
    PhysicalParameter,
    RepresentationParameter,
    SamplerAbsentReason,
    StochasticEvidenceKind,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityState,
    fingerprint_of,
)
from verification.families.predicates import (
    asm_transfer_function_sampling,
    boolean_margin,
    capability_intersection_nonempty,
    fractional_margin,
    per_axis_nyquist_pitch,
    si_s3_curvature_bound,
)
from verification.families.schema import GateDisposition

# --------------------------------------------------------------------------- #
# A minimal well-formed family, so each test varies one thing
# --------------------------------------------------------------------------- #

POLICY = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.FLOAT64}),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason="closed-form oracle over a fixed grid; nothing samples",
)

METRIC = Metric(
    name="relative_error",
    description="relative error against the closed form",
    unit=None,
    blind_to=("a global phase offset",),
)


def make_family(**overrides) -> BenchmarkFamily:
    kwargs = dict(
        family_id="B1-RAY-EXAMPLE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question="does the traced focal length match R/(n-1)?",
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter("radius_m", "surface radius of curvature", unit="m"),
            NumericalParameter("ray_count", "rays in the bundle", refines_toward=1),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description="R/(n-1), exact for a single refracting surface in air",
        ),
        metrics=(METRIC,),
        execution_policy=POLICY,
        stochastic_policy=DETERMINISTIC,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
    )
    kwargs.update(overrides)
    return BenchmarkFamily(**kwargs)


def gating_tolerance(**overrides) -> Tolerance:
    kwargs = dict(
        metric="relative_error",
        threshold=1e-6,
        basis="R/(n-1) is exact; 1e-6 admits only a genuinely different answer",
        basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
        may_gate=True,
    )
    kwargs.update(overrides)
    return Tolerance(**kwargs)


MET = GateDisposition(
    status=GateStatus.MET,
    metric="relative_error",
    observed=1e-13,
    evidence=("tests/test_family_schema.py::test_a_well_formed_family_constructs",),
)


def test_a_well_formed_family_constructs() -> None:
    fam = make_family(tolerances=(gating_tolerance(),), gate_disposition=MET)
    assert fam.is_gate_deciding
    assert fam.gating_tolerances[0].threshold == 1e-6


# --------------------------------------------------------------------------- #
# The four parameter kinds
# --------------------------------------------------------------------------- #


def test_the_four_parameter_kinds_are_distinct_classes() -> None:
    """Kind is a property of which class the author chose, not an argument.

    Making it a ``ClassVar`` is what stops a call site from declaring a grid
    resolution to be physical, which is the mistake that makes convergence
    undefined.
    """
    kinds = {
        PhysicalParameter("a", "d").kind,
        NumericalParameter("b", "d").kind,
        RepresentationParameter("c", "d").kind,
        ExecutionParameter("e", "d").kind,
    }
    assert kinds == set(ParameterKind)
    assert PhysicalParameter("a", "d").kind.changes_the_answer
    assert not NumericalParameter("b", "d").kind.changes_the_answer


def test_the_base_parameter_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Parameter(name="a", description="d")


def test_only_a_numerical_parameter_refines() -> None:
    """A physical parameter that 'refines' is a family that does not know what
    its own oracle depends on."""
    with pytest.raises(ValueError, match="only a NUMERICAL parameter refines"):
        PhysicalParameter("wavelength_m", "wavelength", refines_toward=1)


def test_a_parameter_without_a_description_is_a_magic_number() -> None:
    with pytest.raises(ValueError, match="magic number"):
        NumericalParameter("grid_n", "  ")


# --------------------------------------------------------------------------- #
# Oracle independence, structurally
# --------------------------------------------------------------------------- #


def test_a_shares_code_oracle_forces_b4() -> None:
    with pytest.raises(ValueError, match="must be category B4"):
        make_family(
            oracle=FamilyOracle(
                kind=Oracle.INDEPENDENT_IMPLEMENTATION,
                independence=OracleIndependence.SHARES_CODE,
                description="the repository's own float64 ASM/RS propagator (O2)",
            )
        )


def test_a_cross_route_oracle_forces_b4_even_when_declared_independent() -> None:
    """The live case: FFTPSF against HuygensPSF.

    Both are marked independent by whoever wrote them, and both run through one
    Wavefront/OPD front end. Independence is not the author's opinion.
    """
    with pytest.raises(ValueError, match="must be category B4"):
        make_family(
            oracle=FamilyOracle(
                kind=Oracle.CROSS_ROUTE,
                independence=OracleIndependence.INDEPENDENT,
                description="Optiland FFTPSF against Optiland HuygensPSF",
            )
        )


def test_a_cross_route_family_in_b4_cannot_carry_a_gating_tolerance() -> None:
    with pytest.raises(ValueError, match="cannot carry a gating tolerance"):
        BenchmarkFamily(
            family_id="B4-DUALROUTE-EXAMPLE",
            family_version="1.0.0",
            category=BenchmarkCategory.B4,
            question="do the two Optiland PSF routes agree?",
            components=("M_RAY_OPTILAND",),
            claim_kind=ClaimKind.FORWARD_ACCURACY,
            parameters=(PhysicalParameter("wavelength_m", "wavelength", unit="m"),),
            oracle=FamilyOracle(
                kind=Oracle.CROSS_ROUTE,
                independence=OracleIndependence.SHARES_CODE,
                description="FFTPSF against HuygensPSF",
            ),
            metrics=(METRIC,),
            execution_policy=POLICY,
            stochastic_policy=DETERMINISTIC,
            tolerances=(gating_tolerance(),),
            gate_disposition=MET,
            sampler_absent_reason=SamplerAbsentReason.HISTORICAL_REGRESSION,
        )


def test_a_b4_family_cannot_gate_even_with_an_independent_analytic_oracle() -> None:
    """CHARACTERIZED_NO_GATE must be structurally impossible to promote.

    B4 is the category for measurements with no pass/fail, and the failure mode
    is somebody adding a threshold to a characterization and reading the result
    as a validation.
    """
    with pytest.raises(ValueError, match="B4 family cannot carry a gating tolerance"):
        make_family(
            family_id="B4-COST-EXAMPLE",
            category=BenchmarkCategory.B4,
            tolerances=(gating_tolerance(),),
            gate_disposition=MET,
        )


def test_a_b4_family_may_carry_a_non_gating_tolerance() -> None:
    """Reporting "we measured 2e-3 against a 1e-3 reference" is legitimate
    characterization. What is forbidden is that number deciding anything."""
    fam = make_family(
        family_id="B4-COST-EXAMPLE",
        category=BenchmarkCategory.B4,
        tolerances=(
            gating_tolerance(
                may_gate=False,
                basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
                basis="the value the shipped estimator produced at 1e6 rays",
            ),
        ),
    )
    assert not fam.is_gate_deciding


def test_a_measured_tolerance_basis_cannot_gate() -> None:
    with pytest.raises(ValueError, match="may_gate must be False"):
        Tolerance(
            metric="relative_error",
            threshold=2.2e-3,
            basis="the residual the shipping path produced at 787,969 rays",
            basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
            may_gate=True,
        )


def test_a_cross_route_agreement_basis_cannot_gate() -> None:
    with pytest.raises(ValueError, match="may_gate must be False"):
        Tolerance(
            metric="relative_error",
            threshold=1e-3,
            basis="the two Optiland PSF routes agree to this",
            basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
            may_gate=True,
        )


@pytest.mark.parametrize(
    "basis_kind",
    [
        ToleranceBasis.ANALYTIC_DERIVATION,
        ToleranceBasis.INDEPENDENT_DERIVATION,
        ToleranceBasis.CONSERVATION_LAW,
        ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
    ],
)
def test_an_independently_derived_basis_may_gate(basis_kind: ToleranceBasis) -> None:
    assert basis_kind.is_independently_derived
    assert gating_tolerance(basis_kind=basis_kind).may_gate


def test_a_tolerance_without_a_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="without its basis"):
        Tolerance(
            metric="relative_error",
            threshold=1e-6,
            basis="   ",
            basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
            may_gate=False,
        )


# --------------------------------------------------------------------------- #
# Gate disposition
# --------------------------------------------------------------------------- #


def test_a_gating_family_must_record_where_its_gate_stands() -> None:
    with pytest.raises(ValueError, match="no gate_disposition"):
        make_family(tolerances=(gating_tolerance(),))


def test_a_decided_gate_must_report_a_number_and_cite_evidence() -> None:
    with pytest.raises(ValueError, match="must report what was observed"):
        GateDisposition(status=GateStatus.NOT_MET, metric="relative_error", evidence=("x",))
    with pytest.raises(ValueError, match="must cite its evidence"):
        GateDisposition(status=GateStatus.NOT_MET, metric="relative_error", observed=2.2e-3)


def test_not_met_is_a_state_a_family_can_carry() -> None:
    """An honestly recorded unmet gate is the point, not a defect in the schema."""
    fam = make_family(
        tolerances=(gating_tolerance(threshold=1.0e-3),),
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MET,
            metric="relative_error",
            observed=2.2e-3,
            evidence=("benchmarks/physics/L2-PSF-01/README.md",),
            note="carried into M4 as an explicit open limitation",
        ),
    )
    assert fam.gate_disposition is not None
    assert fam.gate_disposition.status is GateStatus.NOT_MET


def test_characterized_no_gate_cannot_be_claimed_by_a_family_that_gates() -> None:
    with pytest.raises(ValueError, match="CHARACTERIZED_NO_GATE on a family that gates"):
        make_family(
            tolerances=(gating_tolerance(),),
            gate_disposition=GateDisposition(status=GateStatus.CHARACTERIZED_NO_GATE),
        )


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #


def test_a_family_with_no_sampler_must_say_why() -> None:
    with pytest.raises(ValueError, match="Non-generative is a declaration"):
        make_family(sampler_absent_reason=None)


def test_a_family_with_a_sampler_has_no_absent_reason() -> None:
    with pytest.raises(ValueError, match="no absent reason"):
        make_family(sampler=lambda **_: None)


# --------------------------------------------------------------------------- #
# Stochastic policy
# --------------------------------------------------------------------------- #


def test_one_seed_is_never_an_accuracy_result() -> None:
    with pytest.raises(ValueError, match="at least 2 seeds"):
        StochasticPolicy(
            is_stochastic=True,
            required_evidence=(StochasticEvidenceKind.UNBIASEDNESS,),
            minimum_seeds=1,
        )


def test_a_stochastic_family_names_the_evidence_it_owes() -> None:
    with pytest.raises(ValueError, match="must name the evidence it owes"):
        StochasticPolicy(is_stochastic=True, minimum_seeds=8)


def test_a_deterministic_family_must_say_why_it_is_deterministic() -> None:
    with pytest.raises(ValueError, match="must say why it is deterministic"):
        StochasticPolicy(is_stochastic=False)


def test_a_stochastic_instance_must_carry_a_seed() -> None:
    fam = make_family(
        stochastic_policy=StochasticPolicy(
            is_stochastic=True,
            required_evidence=(StochasticEvidenceKind.UNBIASEDNESS,),
            minimum_seeds=8,
        )
    )
    with pytest.raises(ValueError, match="must carry a seed"):
        fam.instantiate("x", {"radius_m": 0.1, "ray_count": 1000})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_a_metric_must_say_what_it_is_blind_to() -> None:
    with pytest.raises(ValueError, match="state what this metric cannot see"):
        Metric(name="ncc", description="normalized cross correlation", unit=None, blind_to=())


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #


def test_a_control_that_does_not_simply_fail_must_explain_itself() -> None:
    """The live case is L2-PSF-01's inverted quadrature weight, which fires
    backwards. An unexplained backwards control reads as a passing one."""
    from verification.families.schema import NegativeControlExpectation

    with pytest.raises(ValueError, match="must say why"):
        NegativeControl(
            control_id="inverted-quadrature-weight",
            description="invert the radial trapezoid weight",
            mutation="w -> 1/w on every ray",
            target_metric="relative_error",
            expectation=NegativeControlExpectation.KNOWN_FIRES_BACKWARDS,
        )


def test_a_control_must_target_a_declared_metric() -> None:
    with pytest.raises(ValueError, match="not a declared metric"):
        make_family(
            negative_controls=(
                NegativeControl(
                    control_id="c",
                    description="d",
                    mutation="m",
                    target_metric="strehl",
                ),
            )
        )


# --------------------------------------------------------------------------- #
# Category / id agreement
# --------------------------------------------------------------------------- #


def test_the_id_prefix_must_agree_with_the_category() -> None:
    with pytest.raises(ValueError, match="id prefix disagrees with category"):
        make_family(family_id="B2-RAY-EXAMPLE")


def test_a_family_must_name_components_the_ledger_knows() -> None:
    with pytest.raises(ValueError, match="components the ledger does not know"):
        make_family(components=("M_TMM_JAXLAYERLUMOS",))


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_the_fingerprint_is_stable_across_processes() -> None:
    """Not ``hash()``: Python salts string hashing per process, so a committed
    fingerprint computed that way would be meaningless."""
    import os
    import subprocess
    import sys

    from core.paths import repository_root

    code = (
        "from core.precision import DeviceKind, DType;"
        "from verification.families import ExecutionPolicy, fingerprint_of;"
        "p = ExecutionPolicy(devices=frozenset({DeviceKind.CPU}), "
        "dtypes=frozenset({DType.FLOAT64}));"
        "print(fingerprint_of(family_id='F', family_version='1.0.0', "
        "parameters={'a': 1.0, 'b': 'x'}, seed=7, execution_policy=p))"
    )
    here = fingerprint_of(
        family_id="F",
        family_version="1.0.0",
        parameters={"a": 1.0, "b": "x"},
        seed=7,
        execution_policy=POLICY,
    )
    root = repository_root()
    # A different PYTHONHASHSEED is the whole point: str.__hash__ is salted per
    # process, so a fingerprint that used it would differ here and nowhere say so.
    env = {**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": str(root / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
        env=env,
    )
    assert out.stdout.strip() == here


def test_parameter_order_does_not_change_the_fingerprint() -> None:
    a = fingerprint_of(
        family_id="F",
        family_version="1.0.0",
        parameters={"a": 1.0, "b": 2.0},
        seed=None,
        execution_policy=POLICY,
    )
    b = fingerprint_of(
        family_id="F",
        family_version="1.0.0",
        parameters={"b": 2.0, "a": 1.0},
        seed=None,
        execution_policy=POLICY,
    )
    assert a == b


def test_a_family_version_bump_invalidates_the_fingerprint() -> None:
    common = dict(
        family_id="F",
        parameters={"a": 1.0},
        seed=None,
        execution_policy=POLICY,
    )
    assert fingerprint_of(family_version="1.0.0", **common) != fingerprint_of(
        family_version="1.1.0", **common
    )


def test_a_pinned_fingerprint_that_moved_is_a_loud_failure() -> None:
    fam = make_family()
    inst = fam.instantiate("i1", {"radius_m": 0.1, "ray_count": 1000})
    with pytest.raises(ValueError, match="fingerprint moved"):
        fam.instantiate(
            "i1", {"radius_m": 0.2, "ray_count": 1000}, pinned_fingerprint=inst.fingerprint
        )


def test_an_unfingerprintable_parameter_is_refused() -> None:
    with pytest.raises(TypeError, match="not fingerprintable"):
        fingerprint_of(
            family_id="F",
            family_version="1.0.0",
            parameters={"callback": lambda: None},
            seed=None,
            execution_policy=POLICY,
        )


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #


def test_an_instance_cannot_carry_an_undeclared_parameter() -> None:
    fam = make_family()
    with pytest.raises(ValueError, match="undeclared parameters"):
        fam.instantiate("i", {"radius_m": 0.1, "ray_count": 10, "tilt_deg": 3.0})


def test_an_instance_must_supply_every_parameter_without_a_default() -> None:
    fam = make_family()
    with pytest.raises(ValueError, match="missing parameters"):
        fam.instantiate("i", {"radius_m": 0.1})


def test_a_canonical_instance_built_against_another_version_is_refused() -> None:
    fam = make_family()
    inst = fam.instantiate("i", {"radius_m": 0.1, "ray_count": 10})
    bumped = make_family(family_version="2.0.0")
    with pytest.raises(ValueError, match="version bump invalidates its fingerprint"):
        bumped.with_instances(inst)


def test_a_generated_instance_cannot_be_listed_as_canonical() -> None:
    fam = make_family()
    generated = fam.instantiate(
        "g", {"radius_m": 0.1, "ray_count": 10}, origin=InstanceOrigin.GENERATED
    )
    with pytest.raises(ValueError, match="listed as canonical"):
        fam.with_instances(generated)


# --------------------------------------------------------------------------- #
# Validity predicates and the signed margin
# --------------------------------------------------------------------------- #


def test_the_margin_sign_convention() -> None:
    assert fractional_margin(0.5, 1.0) == pytest.approx(0.5)  # inside
    assert fractional_margin(1.0, 1.0) == pytest.approx(0.0)  # exactly at it
    assert fractional_margin(2.0, 1.0) == pytest.approx(-1.0)  # twice the limit
    assert math.isinf(fractional_margin(1.0, math.inf))  # unbounded


def test_a_boolean_predicate_has_no_gradient_to_chase() -> None:
    """+-1 rather than a small number, so a boundary sampler fails to find a
    gradient instead of following a fabricated one."""
    assert boolean_margin(True) == 1.0
    assert boolean_margin(False) == -1.0


@pytest.mark.parametrize(
    ("margin_value", "expected"),
    [
        (1.0, ValidityState.INSIDE),
        (0.2, ValidityState.INSIDE),
        (0.01, ValidityState.NEAR_BOUNDARY),
        (0.0, ValidityState.NEAR_BOUNDARY),
        (-0.01, ValidityState.NEAR_BOUNDARY),
        (-0.2, ValidityState.OUTSIDE),
        (-0.9, ValidityState.FAR_OUTSIDE),
    ],
)
def test_the_four_validity_states_partition_the_margin(
    margin_value: float, expected: ValidityState
) -> None:
    from verification.families.schema import ValidityBasis, ValidityPredicate

    predicate = ValidityPredicate(
        predicate_id="P",
        statement="a bound",
        basis=ValidityBasis.PARAXIAL_APPROXIMATION,
        margin=lambda _params: margin_value,
    )
    assert predicate.state({}) is expected


def test_the_curvature_predicate_reproduces_eq_s9() -> None:
    """``eps_curv <= arcsin(D / 2R)``. At D = 1, R = 10 the bound is
    ``arcsin(0.05)``, which ``tests/test_curvature_bound.py`` already pins."""
    predicate = si_s3_curvature_bound()
    bound = math.asin(0.05)
    params = {"patch_width_m": 1.0, "substrate_radius_m": 10.0}
    assert predicate.margin({**params, "tangent_plane_error_rad": bound}) == pytest.approx(0.0)
    assert predicate.state({**params, "tangent_plane_error_rad": bound * 0.5}) is (
        ValidityState.INSIDE
    )
    assert predicate.state({**params, "tangent_plane_error_rad": bound * 2.0}) is (
        ValidityState.FAR_OUTSIDE
    )


def test_the_planar_case_admits_only_zero_tangent_plane_error() -> None:
    predicate = si_s3_curvature_bound()
    planar = {"patch_width_m": 1.0, "substrate_radius_m": math.inf}
    assert predicate.margin({**planar, "tangent_plane_error_rad": 0.0}) == 1.0
    assert predicate.margin({**planar, "tangent_plane_error_rad": 1e-9}) == -1.0


def test_the_nyquist_predicate_binds_on_the_marginal_direction_cosine() -> None:
    predicate = per_axis_nyquist_pitch()
    params = {"wavelength_m": 1e-6, "max_direction_cosine": 0.5}
    limit = 1e-6 / (2.0 * 0.5)
    assert predicate.margin({**params, "sample_pitch_m": limit}) == pytest.approx(0.0)
    assert predicate.state({**params, "sample_pitch_m": limit / 4}) is ValidityState.INSIDE
    assert predicate.state({**params, "sample_pitch_m": limit * 3}) is ValidityState.FAR_OUTSIDE


def test_the_asm_predicate_bounds_the_transfer_function_sampling() -> None:
    predicate = asm_transfer_function_sampling()
    params = {"grid_n": 512, "sample_pitch_m": 1e-6, "wavelength_m": 0.5e-6}
    limit = 512 * 1e-6 * 1e-6 / 0.5e-6
    assert predicate.margin({**params, "propagation_distance_m": limit}) == pytest.approx(0.0)
    assert predicate.state({**params, "propagation_distance_m": -limit / 10}) is (
        ValidityState.INSIDE
    ), "the bound is on |z|; a backward propagation of the same length is equally valid"


def test_the_capability_predicate_reads_the_probe_backed_table() -> None:
    """Chromatix has no complex128 path at any device, so the predicate must say
    outside rather than near-boundary."""
    predicate = capability_intersection_nonempty()
    inside = {"component": "M_WAVE_CHROMATIX", "device": "cpu", "dtype": "complex64"}
    outside = {"component": "M_WAVE_CHROMATIX", "device": "cpu", "dtype": "complex128"}
    assert predicate.state(inside) is ValidityState.INSIDE
    assert predicate.state(outside) is ValidityState.FAR_OUTSIDE


def test_validity_aggregates_to_the_worst_predicate() -> None:
    """A conjunction, not an average. Headroom on two bounds buys nothing on a
    third."""
    from verification.families.schema import ValidityBasis, ValidityPredicate, aggregate_validity

    def const(value: float, name: str) -> ValidityPredicate:
        return ValidityPredicate(
            predicate_id=name,
            statement="s",
            basis=ValidityBasis.PARAXIAL_APPROXIMATION,
            margin=lambda _p, v=value: v,
        )

    fam = make_family(validity=(const(5.0, "a"), const(-0.2, "b"), const(0.9, "c")))
    status, margins = fam.evaluate_validity({"radius_m": 0.1, "ray_count": 10})
    assert status is ValidityState.OUTSIDE
    assert margins == {"a": 5.0, "b": -0.2, "c": 0.9}
    assert aggregate_validity(()) is ValidityState.INSIDE


def test_an_instance_resolves_its_validity_from_the_family_not_the_author() -> None:
    from verification.families.schema import ValidityBasis, ValidityPredicate

    fam = make_family(
        validity=(
            ValidityPredicate(
                predicate_id="RAY_COUNT_FLOOR",
                statement="at least 1000 rays",
                basis=ValidityBasis.PARAXIAL_APPROXIMATION,
                margin=lambda p: fractional_margin(1000.0, float(p["ray_count"])),
            ),
        )
    )
    thin = fam.instantiate("thin", {"radius_m": 0.1, "ray_count": 500})
    thick = fam.instantiate("thick", {"radius_m": 0.1, "ray_count": 100000})
    assert thin.validity_status is ValidityState.FAR_OUTSIDE
    assert thick.validity_status is ValidityState.INSIDE
    assert thin.fingerprint != thick.fingerprint


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_family_ids_are_unique() -> None:
    """Replaces the retired ``A1-*`` id-space collision test."""
    ids = [f.family_id for f in FAMILIES]
    assert len(ids) == len(set(ids))


def test_registering_a_duplicate_id_is_refused() -> None:
    from verification.families import registry

    probe_id = "B1-RAY-DUPLICATE-PROBE"
    registry.register(make_family(family_id=probe_id))
    try:
        with pytest.raises(ValueError, match="duplicate family_id"):
            registry.register(make_family(family_id=probe_id))
    finally:
        # The registry is process-global, and a probe family left in it would
        # project into the ledger and be asked to resolve evidence it has none of.
        registry._REGISTRY.pop(probe_id, None)


def test_every_registered_family_projects_into_the_ledger() -> None:
    from verification.families.projection import claims_from_families

    projected = claims_from_families()
    expected = sum(len(f.components) for f in FAMILIES)
    assert len(projected) == expected
