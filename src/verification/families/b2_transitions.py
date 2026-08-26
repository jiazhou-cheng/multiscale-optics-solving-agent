"""B2: the representation transitions, and what each one provably discards.

CHE-109 through CHE-112 (M2.1-M2.4). Every coupler in this repository is
declared ``lossy: true``, so the question is not "is it correct" but "does
switching representations preserve the physically relevant behaviour to within a
declared budget, and what is provably gone afterwards".

Five families, and the split between the first two is the load-bearing one.

``B2-R2W-EXACT`` and ``B2-R2W-ROUTE`` are separate because the exact route and
the fast route are **not interchangeable**, and one family covering both would
let the fast route inherit the exact one's evidence. The exact route's
correctness instrument is enumeration -- enumerate everything the estimator
would otherwise sample and it must reduce to the reference exactly -- which is
independent of the *sampling* and not of the kernel. The fast route's question
is different: how much does a parameter that should not change the answer
actually change it, and that is a measured budget rather than a limit.

``B2-W2R-STOCH`` keeps the four stochastic evidence kinds apart because they
fail independently: an estimator can be exact in the enumeration limit and still
biased; unbiased and still converge at the wrong rate; converge correctly and
have variance that makes the required ray count unreachable. That last one is
the open question on ``C_WAVE_TO_RAY`` and it is why one seed is never a result.

``B2-ROUNDTRIP`` holds the rule that gives the whole set its meaning: **a round
trip that cannot be made to fail proves nothing.** A shared convention error
cancels between the two directions, so every round trip here is committed with
its deliberately broken twin, and the family cannot report a pass without the
twin having failed.

What provably does not survive
------------------------------
Stated as a deliverable rather than as a caveat, because an agent reasoning
about a chain of representations needs to know which quantities it may still ask
about downstream:

* **no per-ray correspondence survives an accumulation.** The outgoing amplitude
  is a spectral amplitude ``U~[m]/p[m]``, not a transformed incident weight, so
  "which incident ray became this outgoing ray" has no answer;
* **OPL is rebased at every DOE step.** An absolute path length from before the
  step is not comparable with one from after it;
* **evanescent power is discarded** at the wave-to-ray transition, and the
  amount is accounted rather than ignored;
* **a Monte Carlo sample discards everything it did not sample**, which is
  everything, up to a variance the family measures.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import (
    boolean_margin,
    fractional_margin,
    si_s3_curvature_bound,
)
from verification.families.registry import register
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkLayer,
    BenchmarkFamily,
    ClaimKind,
    ExecutionParameter,
    ExecutionPolicy,
    FamilyOracle,
    GateDisposition,
    GateStatus,
    Invariant,
    Metric,
    NegativeControl,
    NegativeControlExpectation,
    NumericalParameter,
    Oracle,
    OracleIndependence,
    PhysicalParameter,
    RepresentationParameter,
    SamplerAbsentReason,
    StochasticEvidenceKind,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = [
    "B2_EQUIV",
    "B2_R2W_EXACT",
    "B2_R2W_ROUTE",
    "B2_ROUNDTRIP",
    "B2_W2R_STOCH",
    "WHAT_DOES_NOT_SURVIVE",
]


#: Per transition, what is provably gone afterwards. Referenced by the families
#: below and by ``knowledge/couplers/README.md``.
WHAT_DOES_NOT_SURVIVE: Mapping[str, tuple[str, ...]] = {
    "C_RAY_TO_WAVE": (
        "per-ray identity: the field is an accumulation, so no output sample "
        "corresponds to any one input ray",
        "the rays themselves -- direction, position and OPL are consumed and not "
        "recoverable from the field",
    ),
    "C_WAVE_TO_RAY": (
        "evanescent power, discarded at the light cone and ACCOUNTED rather than "
        "ignored",
        "everything the Monte Carlo sample did not draw, up to the variance the "
        "family measures",
        "the phase of the unsampled spectrum",
    ),
    "C_PLANAR_DOE_STEP": (
        "the incident OPL reference: OPL is REBASED at every step, so an absolute "
        "path length from before the step is not comparable with one from after it",
        "per-ray correspondence: the outgoing amplitude is a spectral amplitude "
        "U~[m]/p[m], not a transformed incident weight",
    ),
    "C_PATCH_WFT": (
        "the global phase reference between patches, which the coverage correction "
        "restores statistically and not per-sample",
        "any structure finer than the patch, which the tangent-plane approximation "
        "cannot represent",
    ),
}


COUPLER_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    dtypes=frozenset({DType.COMPLEX64, DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    max_wall_seconds=900.0,
    max_peak_memory_gib=32.0,
    notes=(
        "Couplers are pure array math with no package dependency, so complex128 is "
        "genuinely available here even though Chromatix cannot ingest it. That is why "
        "the exactness families can be run in float64 and the composed ones cannot."
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason="an enumerated reconstruction draws nothing; every term is visited",
)


# ---------------------------------------------------------------------------
# B2-R2W-EXACT
# ---------------------------------------------------------------------------

_B2_R2W_EXACT = BenchmarkFamily(
        family_id="B2-R2W-EXACT",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "in the enumeration limit -- every term the estimator would sample, "
            "visited -- does C_RAY_TO_WAVE reduce to the reference field exactly?"
        ),
        components=("C_RAY_TO_WAVE",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.5e-7
            ),
            PhysicalParameter(
                "handoff_plane_z_m",
                "axial position of the declared handoff plane",
                unit="m",
                domain=(0.0, 1.0),
                default=6.814345991561233e-05,
            ),
            NumericalParameter(
                "grid_n", "reconstruction grid", domain=(16, 1024), default=64, refines_toward=1
            ),
            NumericalParameter(
                "target_sample_pitch_m",
                "target grid pitch",
                unit="m",
                domain=(1e-9, 1e-3),
                default=2.6587352810843895e-06,
                refines_toward=-1,
            ),
            RepresentationParameter(
                "sample_alignment",
                "whether the launch coordinates land ON grid nodes or between them. "
                "On-node is where an exactness claim is even meaningful; off-node is "
                "where interpolation error lives, and conflating the two is how a "
                "route gets credited with an exactness it only has on nodes",
                domain=("on_node", "off_node"),
                default="on_node",
            ),
            ExecutionParameter(
                "dtype", "complex64 or complex128", domain=("complex64", "complex128"),
                default="complex128",
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="ENUMERATION_COMPLETE",
                statement=(
                    "every term the estimator would sample is visited, so the comparison "
                    "is against the deterministic limit rather than against a large sample"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: boolean_margin(bool(p.get("enumeration_complete", True))),
                blind_to=(
                    "whether the enumeration is over the right set. It counts terms, not "
                    "the correctness of the term list",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the CLOSED-FORM plane wave N exp(i k d.r) on the reconstruction grid, "
                "which shares no code with the coupler. This is stronger than the "
                "enumerated-reference framing the family carried before it was "
                "executed, and the difference is the whole gate: an enumerated "
                "reference is the same arithmetic without the sampling, so a wrong "
                "kernel is wrong in both and cancels, and it can only establish the "
                "sampling. The analytic field pins the KERNEL -- once each ray's OPL "
                "compensates its launch position every ray of a collimated bundle "
                "contributes the same plane wave, so one comparison decides four "
                "conventions at once (OPL-and-ramp, phasor sign, axis order, "
                "projection factor) and each single-term removal breaks it"
            ),
            callable="benchmarks/instances/b2_transitions.py::_plane_wave_on_grid",
            reference="benchmarks/instances/b2_transitions.py",
        ),
        metrics=(
            Metric(
                name="exactness_relative_l2_field",
                description=(
                    "relative L2 of the reconstructed complex field against the "
                    "enumerated reference"
                ),
                unit=None,
                definition="relative_l2_field",
                blind_to=(
                    "a global phase common to the whole field, which the relative L2 on "
                    "the complex field DOES see but which a reader comparing this to an "
                    "intensity residual would not -- stated so the two are not confused",
                    "any convention error that is inert on THIS bundle. The bundle is "
                    "collimated but TILTED -- three k-bins, sin(theta) = 9.697e-3, "
                    "theta = 0.556 deg -- which is what keeps the oblique ramp and the "
                    "projection factor live: 1 - cos(theta) = 4.702e-05, and that is "
                    "exactly the residual the projection-factor removal produces. So "
                    "all four terms are pinned by the run rather than by assertion. "
                    "What stays blind is the SIZE of the tilt: at 0.556 deg the "
                    "projection factor is the weakest of the four removals by four "
                    "orders of magnitude, so a projection error whose effect scales "
                    "faster than 1 - cos(theta) would show here as a smaller residual "
                    "than it would at a realistic numerical aperture",
                ),
            ),
            Metric(
                name="exactness_power_ratio",
                description="power in the reconstruction over power in the enumerated reference",
                unit=None,
                definition="power_ratio",
                blind_to=("where the power is; a ratio says nothing about placement",),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="PUPIL_POWER_CONSISTENCY",
                statement=(
                    "the declared registry invariant: power crossing the pupil is "
                    "conserved into the reconstructed field"
                ),
                metric="exactness_power_ratio",
                tolerance=Tolerance(
                    metric="exactness_power_ratio",
                    threshold=1e-12,
                    basis=(
                        "float64 round-off over an accumulation of ~1e6 terms. The "
                        "full-aperture patch agrees with an independent float64 ASM at "
                        "7.1e-13, which is the same order"
                    ),
                    basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                    may_gate=True,
                    rejects="a dropped term or a double-counted one",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="exactness_relative_l2_field",
                threshold=1e-12,
                basis=(
                    "an exactness limit admits float64 round-off and nothing else: the "
                    "enumerated sum and the reconstruction are the same arithmetic in a "
                    "different order. The measured full-aperture patch agreement against "
                    "an independent float64 ASM is 7.1e-13, and the wave-rays-wave round "
                    "trip at the enumeration limit reads 1.32e-15"
                ),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects=(
                    "any term dropped from the sum, any interpolation introduced where "
                    "the claim is exactness, and the historical silent defects this "
                    "family exists to keep covered"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="off-node-is-not-exact",
                description=(
                    "run the same instance with launch coordinates BETWEEN grid nodes "
                    "and require the exactness tolerance to fail. An exactness claim "
                    "that survived off-node would be measuring something other than "
                    "exactness"
                ),
                mutation="sample_alignment = off_node",
                target_metric="exactness_relative_l2_field",
            ),
            NegativeControl(
                control_id="dropped-term",
                description="omit one term from the enumerated sum",
                mutation="skip a single launch coordinate in the accumulation",
                target_metric="exactness_relative_l2_field",
            ),
            NegativeControl(
                control_id="static-shape-violation",
                description=(
                    "a structural control rather than a numerical one: the enumerated "
                    "kernel is compiled against a static shape, and a ray count that "
                    "changes it must be refused rather than silently retraced"
                ),
                mutation="change the ray count between two calls in one process",
                target_metric="exactness_relative_l2_field",
                expectation=NegativeControlExpectation.NOT_IMPLEMENTED,
                caveat=(
                    "declared and not run: it needs the executor's cache-validity path "
                    "to be wired to the enumerated kernel, which is CHE-115"
                ),
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.INVALID_CONFIGURATION,
        ),
        execution_policy=COUPLER_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="exactness_relative_l2_field",
            observed=1.47949e-15,
            evidence=(
                "benchmarks/instances/b2_transitions.py",
                "tests/test_b2_transition_instances.py::test_the_exact_route_pins_four_conventions_at_once",
                "tests/test_b2_transition_instances.py::test_the_exactness_tolerance_is_derived_from_the_dtype",
            ),
            note=(
                "Executed. 1.48e-15 against a 1e-12 gate, on a collimated bundle of 64 "
                "rays that share a direction and differ only in launch position, "
                "compared to the analytic plane wave every one of them implies.\n\n"
                "The tolerance is DERIVED rather than chosen, which is what CHE-109 asked "
                "for: sqrt(N) eps64 for an N-term unit-modulus sum plus eps64 per radian "
                "of the largest phase argument. On this instance that is 3.6e-15, and the "
                "declared 1e-12 gate is the looser of the two -- so the measurement is "
                "reported against the derivation as well as against the gate.\n\n"
                "Four conventions, one comparison. Removing the Delta-r ramp reads "
                "1.4e+0, the phasor sign 2.0e+0, the axis transpose 1.4e+0, and the "
                "projection factor 1.3e-2 -- each through the shipping Perturbation with "
                "one term altered, and the weakest of the four is what the control "
                "reports. The static-shape guarantee is declared NOT_IMPLEMENTED as a "
                "run-time control on purpose: it is structural, asserted by making "
                "xp.outer and xp.einsum raise, and that survives a host change where a "
                "wall-clock comparison would not."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the oracle IS the enumeration, and enumerating a new parameter point is the "
            "expensive thing -- the demo3 enumerated reference is twelve GPU shards. "
            "That is the textbook case for a non-generative family."
        ),
        evidence=(
            "benchmarks/reports/2026-08/coupler_characterization.md",
            "benchmarks/probes/ray_wave/demo3_enumerated_reference.py",
            "benchmarks/probes/ray_to_wave/coherent_handoff.py",
        ),
        notes=(
            "The exactness limit is independent of the SAMPLING, not of the kernel. A "
            "family that read this number as 'the coupler is correct' would be claiming "
            "something the instrument cannot say."
        ),
)


# ---------------------------------------------------------------------------
# B2-R2W-ROUTE
# ---------------------------------------------------------------------------

#: The repository's primary correctness instrument, as one instance.
#:
#: A collimated bundle whose rays share a direction and differ only in launch
#: position, reconstructed against the analytic plane wave that every one of them
#: implies. SI Figure S1c is why one comparison is enough to pin four
#: conventions at once: once each ray's OPL compensates its launch position,
#: every ray contributes the SAME plane wave, so the sum is N exp(i k d.r) with
#: no residual position dependence. Remove the OPL compensation, the Delta-r
#: ramp, the phasor sign or the projection factor and that identity breaks.
#:
#: ``sample_alignment = on_node`` means the mode's transverse wavevector is an
#: exact bin of the reconstruction grid, so the k-space route is a RELABELLING
#: rather than an interpolation and both routes are exact. That is the condition
#: under which the two routes may be compared at round-off, and it is declared
#: rather than assumed.
B2_R2W_EXACT = register(
    _B2_R2W_EXACT.with_instances(
        _B2_R2W_EXACT.instantiate(
            "B2-R2W-EXACT-01",
            {
                "wavelength_m": 5.5e-7,
                "handoff_plane_z_m": 6.814345991561234e-05,
                "grid_n": 64,
                "target_sample_pitch_m": 2.6587352810843895e-06,
                "sample_alignment": "on_node",
                "dtype": "complex128",
            },
            expected={
                "why": (
                    "the enumeration IS the oracle here: zero sampling error means a "
                    "failure is a transform defect rather than a budget problem. The "
                    "family is declared non-generative for that reason -- a generator "
                    "that also chose the enumeration would be grading its own homework."
                ),
                "four_conventions": (
                    "OPL compensation, the Delta-r ramp, the phasor sign and the "
                    "projection factor. Each is removed individually and each must fail."
                ),
            },
        ),
    )
)


_B2_R2W_ROUTE = BenchmarkFamily(
        family_id="B2-R2W-ROUTE",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
        layer=BenchmarkLayer.NUMERICAL,
        question=(
            "how much does the reconstruction route -- a parameter that should not "
            "change the answer -- actually change it, and at what cost?"
        ),
        components=("C_RAY_TO_WAVE",),
        claim_kind=ClaimKind.CONVERGENCE,
        parameters=(
            PhysicalParameter(
                "system",
                "which system the routes are compared on. NOT interchangeable: the two "
                "routes agree to 7.1e-13 on demo2 and lose 1.7% of the power on demo3",
                domain=("demo2_paper", "demo3_characterization"),
                default="demo2_paper",
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.32e-7
            ),
            NumericalParameter(
                "oversampling",
                "k-grid oversampling. THE numerical parameter of this family: the "
                "k-space route's power loss is a function of it",
                domain=(1, 32),
                default=8,
                refines_toward=1,
            ),
            NumericalParameter(
                "ray_count", "traced rays", domain=(int(1e4), int(1e10)), default=int(1e6),
                refines_toward=1,
            ),
            RepresentationParameter(
                "route",
                "RAMP_SUM or the k-space splat. Should not change the answer; the "
                "measured budget is what this family reports",
                domain=("ramp_sum", "kspace_splat"),
                default="kspace_splat",
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="OVERSAMPLING_AT_OR_ABOVE_MEASURED",
                statement=(
                    "the k-grid oversampling is at least the 8x at which the route's "
                    "error budget was measured. Below it the budget is not a budget, "
                    "because nothing has measured what the route costs there"
                ),
                basis=ValidityBasis.ASM_SAMPLING,
                margin=lambda p: (float(p["oversampling"]) - 8.0) / 8.0,
                blind_to=(
                    "whether MORE oversampling helps. The predicate bounds where the "
                    "budget is known, not where the route is accurate",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.DETERMINISTIC_LIMIT,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the exact route's enumerated result, which B2-R2W-EXACT establishes "
                "separately. The fast route is measured AGAINST the exact one rather "
                "than beside it, so it cannot inherit the exact one's evidence"
            ),
            callable=None,
            reference="B2-R2W-EXACT",
        ),
        metrics=(
            Metric(
                name="route_field_relative_l2",
                description="relative L2 of the fast route's field against the exact route's",
                unit=None,
                definition="relative_l2_field",
                blind_to=(
                    "absolute power, which the ratio below is for",
                    "where the error is. A norm over the whole grid cannot distinguish "
                    "a large localized error from a small diffuse one, and the splat "
                    "kernel's residual is NOT uniform -- measured 3.8x larger over the "
                    "whole grid than over a centred window. That is CHE-44's concern in "
                    "this coordinate: a metric reported on axis understates this route's "
                    "error where the kernel is worst",
                ),
            ),
            Metric(
                name="route_power_ratio",
                description="fast-route power over exact-route power",
                unit=None,
                definition="power_ratio",
                blind_to=(
                    "placement. The measured 0.9832 on demo3 says 1.7% of the energy is "
                    "gone and nothing about where it went",
                ),
            ),
            Metric(
                name="route_ncc",
                description="normalized cross-correlation between the two routes' intensities",
                unit=None,
                definition="ncc",
                blind_to=(
                    "exactly the power loss the ratio measures. The k-space route reads "
                    "NCC 0.999868 while losing 1.7% of the energy, which is the clearest "
                    "case in this repository for why NCC alone certifies nothing",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="route_field_relative_l2",
                threshold=5e-2,
                basis=(
                    "the ERROR BUDGET the fast route is allowed, not an exactness claim. "
                    "Measured 1.063e-2 against the exact route on demo3 at 8x "
                    "oversampling; 5e-2 is roughly five times that, which admits the "
                    "oversampling range the family sweeps without admitting a route that "
                    "has stopped tracking"
                ),
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects="a route whose error grows with oversampling instead of shrinking",
            ),
            Tolerance(
                metric="route_power_ratio",
                threshold=5e-2,
                basis=(
                    "|1 - ratio|. Measured 0.9832 on demo3 at 8x oversampling -- a 1.7% "
                    "loss -- and 1 - 7.1e-13 on demo2. 5e-2 admits the measured demo3 "
                    "loss as a DECLARED budget rather than as a pass, and refuses a "
                    "route losing more than 5%"
                ),
                basis_kind=ToleranceBasis.CONSERVATION_LAW,
                may_gate=True,
                rejects="a route that quietly loses a tenth of the energy",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="ncc-alone-would-have-passed-it",
                description=(
                    "report only NCC on the demo3 k-space instance. It reads 0.999868 "
                    "while 1.7% of the power is gone, so a family gated on NCC alone "
                    "would certify a lossy route"
                ),
                mutation="evaluate route_ncc and omit route_power_ratio",
                target_metric="route_ncc",
            ),
            NegativeControl(
                control_id="oversampling-does-not-help",
                description=(
                    "if the field error does not shrink with oversampling, the route is "
                    "not converging to the exact one and the budget is not a budget"
                ),
                mutation="compare the error at 2x and 8x oversampling",
                target_metric="route_field_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=COUPLER_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="route_power_ratio",
            observed=1.21609e-3,
            evidence=(
                "benchmarks/instances/b2_transitions.py",
                "tests/test_b2_transition_instances.py::test_the_route_budget_is_measured_on_both_systems",
                "tests/test_b2_transition_instances.py::test_ncc_cannot_see_the_power_the_route_loses",
            ),
            note=(
                "The budget, measured over four oversampling values on each of two "
                "systems, and the ASYMMETRY is the finding rather than either number.\n\n"
                "ON-NODE: exact at every oversampling -- 9.0e-16 field error, power ratio "
                "1.000000000, on_node_fraction 1.0 -- because every ray's transverse "
                "wavevector is an exact bin and the bilinear splat weights collapse to "
                "(1, 0). The splat is a relabelling there, not an approximation.\n\n"
                "OFF-NODE: 2.20e-1 -> 4.26e-2 -> 1.14e-2 -> 1.81e-3 as oversampling goes "
                "1x -> 8x, with the power ratio 0.9492 -> 0.9804 -> 0.9946 -> 0.9988. "
                "on_node_fraction 0.0 throughout. At 1x -- upstream's default region -- "
                "the route loses 5.1% of the power and NCC still reads 0.9989, which is "
                "the whole reason both are reported: NCC is normalized and cannot see "
                "absolute scale by construction.\n\n"
                "The residual GROWS off axis by 3.8x, whole-grid against centred window. "
                "That is the splat kernel's signature rather than a ray-count effect, and "
                "a centred metric cannot see it -- CHE-44's concern, in this coordinate.\n\n"
                "These are NOT the paper-scale demo3 numbers and do not claim to be: the "
                "recorded 1.07e-2 field error and 1.7% power loss at 8x come from a 60M-ray "
                "run that stays a probe (B4-DEMO3). What is reproduced here is the SHAPE "
                "of the budget on a tractable off-node system."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the oracle is the exact route, and running it at a drawn point is the "
            "expensive half of every comparison."
        ),
        evidence=(
            "benchmarks/reports/2026-08/kspace_ray_to_wave.md",
            "benchmarks/probes/ray_wave/demo3_reconstruction_equivalence.py",
        ),
)


# ---------------------------------------------------------------------------
# B2-W2R-STOCH
# ---------------------------------------------------------------------------

#: Two systems and four oversampling values each, because one system cannot
#: state this budget.
#:
#: On an ON-NODE system the k-space route is exact at every oversampling, so a
#: budget measured there would read zero and say nothing. On an OFF-NODE system
#: it interpolates. Averaging the two would report a number that describes
#: neither, which is why the family declares ``system`` as a PHYSICAL parameter
#: -- a different system is a different correct answer, not a different way of
#: representing one.
B2_R2W_ROUTE = register(
    _B2_R2W_ROUTE.with_instances(
        *[
            _B2_R2W_ROUTE.instantiate(
                f"B2-R2W-ROUTE-{system_tag}-{oversampling:02d}",
                {
                    "system": system,
                    "wavelength_m": 5.32e-7,
                    "oversampling": oversampling,
                    "ray_count": 10000,
                    "route": "kspace_splat",
                },
                expected={
                    "budget": (
                        "exact on the on-node system at every oversampling, because the "
                        "splat is a relabelling there; interpolating on the off-node "
                        "one, where 8x oversampling still loses 1.7% of the power"
                    ),
                },
            )
            for system, system_tag in (
                ("demo2_paper", "ONNODE"),
                ("demo3_characterization", "OFFNODE"),
            )
            for oversampling in (1, 2, 4, 8)
        ]
    )
)


_B2_W2R_STOCH = BenchmarkFamily(
        family_id="B2-W2R-STOCH",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "is the wave-to-ray estimator exact in the enumeration limit, unbiased, "
            "converging at the expected rate, and at a variance the required ray count "
            "can reach? Four questions, and they fail independently."
        ),
        components=("C_WAVE_TO_RAY",),
        claim_kind=ClaimKind.CONVERGENCE,
        parameters=(
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.32e-7
            ),
            PhysicalParameter(
                "numerical_aperture",
                "how much of the spectrum is inside the light cone; the evanescent "
                "fraction is one minus this in the relevant sense",
                domain=(0.01, 0.99),
                default=0.5,
            ),
            NumericalParameter(
                "sample_count",
                "secondary directions drawn. The variance axis",
                domain=(int(1e3), int(1e10)),
                default=int(1e6),
                refines_toward=1,
            ),
            ExecutionParameter(
                "seed",
                "estimator seed. Mandatory: one realization is never an accuracy result",
                domain=(0, 2**31 - 1),
                default=0,
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="MULTINOMIAL_CATEGORY_LIMIT",
                statement=(
                    "the number of categories in one multinomial draw stays inside "
                    "CUDA's hard limit, which is why the sampler is two-stage "
                    "(marginal p(y) then conditional p(x|y)) rather than flat"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: fractional_margin(
                    float(p.get("categories", 1)), float(p.get("category_limit", 2**24))
                ),
                blind_to=(
                    "host execution, where the limit does not apply and the two-stage "
                    "sampler is a cost rather than a requirement",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.DETERMINISTIC_LIMIT,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the enumerated spectrum. Every direction the estimator could draw, "
                "visited, which bounds the sampling and not the kernel"
            ),
            callable=None,
            reference="tests/test_wave_to_ray.py; tests/test_patch_positions.py",
        ),
        metrics=(
            Metric(
                name="enumeration_limit_relative_l2",
                description="estimator against the enumerated spectrum, all terms visited",
                unit=None,
                definition="relative_l2_field",
                blind_to=(
                    "bias in the kernel, which the enumeration shares. Evidence kind 1 "
                    "of four, and it does not imply the other three",
                ),
            ),
            Metric(
                name="ensemble_mean_bias",
                description=(
                    "ensemble mean over seeds, minus the enumerated reference, in units "
                    "of the ensemble standard error"
                ),
                unit="sigma",
                blind_to=(
                    "a bias smaller than the ensemble can resolve, which shrinks only "
                    "as sqrt(seeds)",
                ),
            ),
            Metric(
                name="fitted_convergence_exponent",
                description="log-log slope of error against sample count, with its fit quality",
                unit=None,
                blind_to=(
                    "curvature outside the fitted range. An exponent measured over three "
                    "decades says nothing about the fourth",
                ),
            ),
            Metric(
                name="variance_at_sampling_density",
                description="estimator variance as a function of samples per unit area",
                unit=None,
                blind_to=("the spatial distribution of the variance",),
            ),
            Metric(
                name="evanescent_power_fraction",
                description=(
                    "the power discarded at the light cone, accounted rather than ignored"
                ),
                unit=None,
                definition="power_ratio",
                blind_to=("where in the spectrum it was discarded from",),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="EVANESCENT_POWER_ACCOUNTED",
                statement=(
                    "the registry-declared invariant: the power beyond the light cone is "
                    "reported, not silently dropped. It does not survive the transition "
                    "and the amount is part of the result"
                ),
                metric="evanescent_power_fraction",
                tolerance=Tolerance(
                    metric="evanescent_power_fraction",
                    threshold=1e-9,
                    basis=(
                        "the accounting must close to float64 round-off: propagating "
                        "power plus evanescent power equals total power. This is a "
                        "bookkeeping identity, not a physics tolerance"
                    ),
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects="an estimator that discards evanescent content without counting it",
                ),
            ),
            Invariant(
                invariant_id="UNIT_DIRECTION_NORM",
                statement="every emitted direction cosine has unit norm",
                metric="evanescent_power_fraction",
                tolerance=Tolerance(
                    metric="evanescent_power_fraction",
                    threshold=1e-12,
                    basis=(
                        "float64 round-off on a normalization; the artifact boundary "
                        "already checks it"
                    ),
                    basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                    may_gate=True,
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="enumeration_limit_relative_l2",
                threshold=1e-12,
                basis=(
                    "an exactness limit admits float64 round-off only; the measured "
                    "wave-rays-wave round trip at the enumeration limit is 1.32e-15"
                ),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects="a term dropped from the enumeration",
            ),
            Tolerance(
                metric="ensemble_mean_bias",
                threshold=3.0,
                basis=(
                    "three standard errors of the ensemble mean. A bias inside 3 sigma "
                    "is not distinguishable from zero by this many seeds, and a bias "
                    "outside it is not chance. Stated in sigma rather than in field units "
                    "because the resolvable bias depends on the ensemble size"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a systematically wrong importance weight, which shifts the mean "
                    "without widening the spread"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="omitted-importance-weight",
                description=(
                    "drop the importance weight. The estimator stays finite and its mean "
                    "moves, which is the shape a bias control needs"
                ),
                mutation="sample without reweighting by 1/p",
                target_metric="ensemble_mean_bias",
            ),
            # CHE-110 added four. M2.2 asks for at least five controls, each
            # through the SHIPPING implementation with one term removed and each
            # with a passing unperturbed arm, and the family declared two. Every
            # one of these is a real switch on `SamplingPerturbation` or
            # `Perturbation` rather than a hand-written variant, and two of them
            # need their own CONFIGURATION to be observable at all -- which is the
            # blind-spot lesson applied to the battery rather than an exception
            # to it.
            NegativeControl(
                control_id="launch-phase",
                description=(
                    "drop the launch-position phase exp(i k (d_u x_p + d_v y_p)). "
                    "Invisible for a single centred launch point, so the control is run "
                    "over sixteen scattered ones"
                ),
                mutation="SamplingPerturbation(apply_launch_phase=False)",
                target_metric="ensemble_mean_bias",
            ),
            NegativeControl(
                control_id="axis-transpose",
                description=(
                    "transpose the reconstruction grid. Invisible on any rotationally "
                    "symmetric field, which is why the probe is deliberately elliptical "
                    "and offset"
                ),
                mutation="Perturbation(transpose_axes=True) on the reconstruction leg",
                target_metric="ensemble_mean_bias",
            ),
            NegativeControl(
                control_id="kn-sign",
                description=(
                    "take the negative root for k_n. Reverses propagation, and is "
                    "EXACTLY INERT at z = 0 -- so the control is run on a bundle advanced "
                    "to an observation plane 20 um away, where the shipping code refuses "
                    "to advance it at all rather than dropping the rays"
                ),
                mutation="SamplingPerturbation(normal_sign=-1), measured off the source plane",
                target_metric="ensemble_mean_bias",
            ),
            NegativeControl(
                control_id="evanescent-cut",
                description=(
                    "keep the evanescent modes, whose k_n is imaginary so their "
                    "direction is not a direction. Inert on a grid that HAS no "
                    "evanescent content -- at the family's own pitch the fraction is "
                    "exactly zero -- so the control is run at a sub-wavelength pitch"
                ),
                mutation=(
                    "SamplingPerturbation(discard_evanescent=False) at pitch = lambda/3"
                ),
                target_metric="ensemble_mean_bias",
            ),
            NegativeControl(
                control_id="one-seed-accuracy-claim",
                description=(
                    "report accuracy from a single realization. The number exists and "
                    "means nothing, and a family that accepted it would be claiming "
                    "stochastic accuracy from one draw"
                ),
                mutation="run one seed and report its error as the estimator's",
                target_metric="ensemble_mean_bias",
            ),
        ),
        failure_semantics=(
            VerificationStatus.UNCONVERGED,
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNSUPPORTED,
        ),
        execution_policy=COUPLER_EXECUTION,
        stochastic_policy=StochasticPolicy(
            is_stochastic=True,
            required_evidence=(
                StochasticEvidenceKind.EXACTNESS_LIMIT,
                StochasticEvidenceKind.UNBIASEDNESS,
                StochasticEvidenceKind.CONVERGENCE_EXPONENT,
                StochasticEvidenceKind.VARIANCE_CHARACTERIZATION,
            ),
            minimum_seeds=8,
        ),
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="ensemble_mean_bias",
            observed=0.95292,
            evidence=(
                "benchmarks/instances/b2_transitions.py",
                "tests/test_b2_transition_instances.py::test_the_four_evidence_kinds_are_present_in_order",
                "tests/test_b2_transition_instances.py::test_the_negative_control_battery_is_five_deep",
                "tests/test_b2_transition_instances.py::test_the_three_blind_spots_are_measured",
            ),
            note=(
                "All four kinds run as one family over eight declared seeds.\n\n"
                "1. Exactness limit 5.31e-16 against a 1e-12 gate, over every propagating "
                "mode. FIRST, because an estimator wrong here has a transform defect and "
                "tuning N would be beside the point.\n"
                "2. Unbiasedness 0.95 sigma against a 3 sigma gate, on the SIGNED overlap "
                "functional <W, U_hat> against a fixed independent probe vector.\n"
                "3. Fitted exponent -0.5217 over six sample counts, against -0.5 +/- 0.1.\n"
                "4. Variance advantage 6.28x on a concentrated spectrum against 1.4x on a "
                "multilobed one -- the SIZE is the reported property.\n\n"
                "Two constructions had to be corrected to get here, and both had produced "
                "a plausible number. (a) The reconstruction owes the 1/N of SI eq S5 for a "
                "sampled bundle; without it the field is scaled by the mode count and the "
                "round trip read 0.995 for a correct round trip. (b) Unbiasedness must be "
                "measured on a signed LINEAR functional, and not against the field itself: "
                "<U, U> is real and positive by construction, so its imaginary part is "
                "identically zero and comparing it to its own round-off spread read 5 to "
                "23 sigma for an estimator that is exactly unbiased. Measured across three "
                "sample counts and two ensemble sizes.\n\n"
                "Six controls, all fired, each one term removed from the shipping "
                "implementation: importance weight 1288 sigma, axis transpose 91 sigma, "
                "launch phase 77 sigma, k_n sign REFUSED by the shipping advance rather "
                "than measured, evanescent cut at a sub-wavelength pitch, and the "
                "one-seed accuracy claim refused at StochasticReport construction. Two of "
                "them needed their own CONFIGURATION to be observable -- k_n sign is "
                "exactly inert at z = 0 and the evanescent cut is inert on a grid with no "
                "evanescent content -- which is the blind-spot lesson applied to the "
                "battery rather than an exception to it.\n\n"
                "NO GRADIENT IS CLAIMED. derivative.verified stays false and the "
                "surrogate's bias is measured and recorded, which is the deliverable."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the enumerated oracle is the expensive half, and the useful sampling is over "
            "sample_count and seed rather than over the physics -- which is a sampler "
            "with a budget policy, and that is M9's."
        ),
        evidence=(
            "tests/test_wave_to_ray.py",
            "benchmarks/protocols/coupler_protocol.yaml",
            "benchmarks/reports/2026-08/coupler_characterization.md",
        ),
        notes=(
            "ARCHITECTURE NOTE, answering M2.2's open question. C_WAVE_TO_RAY is "
            "declared in registry/couplers.yaml and has NO executable graph node: it is "
            "a library component that C_PATCH_WFT and C_PLANAR_DOE_STEP wrap internally. "
            "That is now explicit rather than implicit -- the executor refuses an edge "
            "naming it, before running anything, with the reason and the alternatives "
            "(tests/test_executor.py::"
            "test_a_coupler_with_no_graph_node_is_refused_before_anything_runs). It is "
            "not composable as an edge, and a graph that tries says so instead of dying "
            "in a resolver."
        ),
)


# ---------------------------------------------------------------------------
# B2-EQUIV
# ---------------------------------------------------------------------------

#: Eight seeds at the declared sample count, plus the convergence ladder.
#:
#: The seed count is the family's own declared minimum and it is not a
#: convention: a bias gated in measured standard errors needs enough
#: realizations for the standard error itself to be a measurement, and the
#: schema refuses fewer.
B2_W2R_STOCH = register(
    _B2_W2R_STOCH.with_instances(
        *[
            _B2_W2R_STOCH.instantiate(
                f"B2-W2R-STOCH-{seed:02d}",
                {
                    "wavelength_m": 5.32e-7,
                    "numerical_aperture": 0.5,
                    "sample_count": 20000,
                    "seed": seed,
                },
                seed=seed,
                expected={
                    "why": (
                        "four kinds of evidence in a mandated order: exactness limit "
                        "first, because an estimator that is wrong in the enumeration "
                        "limit has a transform defect and tuning N would be beside the "
                        "point; then unbiasedness against the MEASURED standard error; "
                        "then a fitted exponent over at least six sample counts; then "
                        "variance by sampling density, reported as a size rather than a "
                        "pass."
                    ),
                },
            )
            for seed in range(1, 9)
        ]
    )
)


_B2_EQUIV = BenchmarkFamily(
        family_id="B2-EQUIV",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does decomposing an aperture into patches and reassembling it give the "
            "same field as treating it globally, and how does the answer depend on "
            "patch granularity?"
        ),
        components=("C_PATCH_WFT", "C_PLANAR_DOE_STEP"),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "aperture_width_m", "full aperture width", unit="m", domain=(1e-6, 1e-1),
                default=1e-3,
            ),
            PhysicalParameter(
                "substrate_radius_m",
                "radius of curvature of the substrate. Infinite is the planar case the "
                "couplers declare; finite is where the SI S3 bound starts to bite",
                unit="m",
                domain=(1e-3, 1e6),
                default=float("inf"),
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.32e-7
            ),
            RepresentationParameter(
                "patch_count",
                "how many patches the aperture is decomposed into. THE representation "
                "parameter of this family: one patch is the global case, and the answer "
                "must not depend on the number",
                domain=(1, 4096),
                default=1,
            ),
            PhysicalParameter(
                "patch_width_m",
                "width of one patch. Declared rather than derived from "
                "aperture_width_m / patch_count, because a decomposition may overlap "
                "and the SI S3 bound takes the actual patch width",
                unit="m",
                domain=(1e-9, 1e-1),
                default=1e-3,
            ),
            PhysicalParameter(
                "tangent_plane_error_rad",
                "the direction error the tangent-plane approximation incurs on this "
                "patch. What eq S9 bounds",
                unit="rad",
                domain=(0.0, 1.5),
                default=0.0,
            ),
            RepresentationParameter(
                "grid_snapping",
                "whether patch centres are snapped to grid nodes. A representation "
                "choice with a measurable cost, not a detail",
                domain=("snapped", "exact"),
                default="exact",
            ),
            NumericalParameter(
                "pad_width", "FFT pad around each patch", domain=(0, 4096), default=566,
                refines_toward=1,
            ),
        ),
        validity=(
            si_s3_curvature_bound(
                error_key="tangent_plane_error_rad",
                width_key="patch_width_m",
                radius_key="substrate_radius_m",
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.INDEPENDENT_IMPLEMENTATION,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "an independent float64 ASM, written separately from the patch "
                "decomposition and sharing none of its code. The full-aperture patch "
                "agrees with it at 7.1e-13"
            ),
            callable=None,
            reference="src/verification/asm_oracle.py",
        ),
        metrics=(
            Metric(
                name="patch_vs_global_relative_l2",
                description="relative L2 of the reassembled field against the global one",
                unit=None,
                definition="relative_l2_field",
                blind_to=("per-patch structure that cancels in the sum",),
            ),
            Metric(
                name="coverage_corrected_power_ratio",
                description=(
                    "power after the coverage correction over power before "
                    "decomposition"
                ),
                unit=None,
                definition="power_ratio",
                blind_to=(
                    "whether the correction is right per patch. It restores the total "
                    "statistically, so a compensating pair of per-patch errors is "
                    "invisible",
                ),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="PATCH_COVERAGE_CORRECTED",
                statement=(
                    "the registry-declared invariant: patch coverage is corrected so the "
                    "reassembled power matches the aperture's"
                ),
                metric="coverage_corrected_power_ratio",
                tolerance=Tolerance(
                    metric="coverage_corrected_power_ratio",
                    threshold=1e-9,
                    basis="a bookkeeping identity; it closes to round-off or it is wrong",
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects="an uncorrected decomposition, which loses the overlap",
                ),
            ),
            Invariant(
                invariant_id="OUTGOING_COUNT_IS_THE_BUDGET",
                statement=(
                    "the composability invariant: N planar DOEs in series produce a "
                    "BOUNDED ray count, not a product. Two in series give 256 then 256, "
                    "not 256 x 64. Without it a multi-element diffractive system is "
                    "combinatorially unrunnable, which is why it belongs here rather "
                    "than in a per-coupler bundle"
                ),
                metric="coverage_corrected_power_ratio",
                tolerance=Tolerance(
                    metric="coverage_corrected_power_ratio",
                    threshold=1e-12,
                    basis="an exact count; the budget is met or the cascade is unbounded",
                    basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                    may_gate=True,
                    rejects="a cascade whose ray count multiplies",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="patch_vs_global_relative_l2",
                threshold=1e-9,
                basis=(
                    "the full-aperture case measures 1.4e-12 against an independent "
                    "float64 ASM and the enumerated sub-aperture case 1.7e-15, so 1e-9 "
                    "is three orders above the looser of the two exact relations.\n\n"
                    "CHE-111 completed this basis after executing the family, and the "
                    "completion decides how the results read. This is an EXACTNESS gate "
                    "and it belongs to the two instances that have an exact relation to "
                    "gate: the full-aperture limit, where one patch IS the window, and "
                    "the enumerated sub-aperture case, where every draw position is "
                    "evaluated exactly once and the estimator's EXPECTATION is computed "
                    "rather than sampled.\n\n"
                    "The DRAWN sub-aperture instances are Monte Carlo estimates and "
                    "cannot meet it: they measure 1.57, 0.81, 0.43 and 0.19 at 4, 16, 64 "
                    "and 225 patches, which is a clean P^-1/2 rate and not a defect. "
                    "Their claim is the convergence trend and their four negative "
                    "controls, and they report this metric UNMET against a gate that "
                    "belongs to the exact instances rather than being quietly exempted. "
                    "`patch_count` is a RepresentationParameter for exactly this reason."
                ),
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects=(
                    "a missing coverage correction, a launch-phase error between "
                    "patches, and grid snapping applied where the claim is exactness"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="omit-coverage-correction",
                description="reassemble without correcting for patch overlap",
                mutation="skip the coverage weight in the sum",
                target_metric="patch_vs_global_relative_l2",
            ),
            NegativeControl(
                control_id="launch-phase-per-patch",
                description=(
                    "give each patch its own phase origin instead of a common one. The "
                    "patches then interfere incorrectly and the reassembled field is "
                    "wrong while every patch is individually fine"
                ),
                mutation="rebase the phase reference per patch",
                target_metric="patch_vs_global_relative_l2",
            ),
            NegativeControl(
                control_id="grid-snapping-is-not-free",
                description=(
                    "snap patch centres to grid nodes and require the exactness "
                    "tolerance to fail. A representation choice with no measurable cost "
                    "would mean the metric cannot see it"
                ),
                mutation="grid_snapping = snapped",
                target_metric="patch_vs_global_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.INVALID_CONFIGURATION,
        ),
        execution_policy=COUPLER_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="patch_vs_global_relative_l2",
            observed=1.44474e-12,
            evidence=(
                "benchmarks/instances/b2_equiv.py",
                "tests/test_b2_equiv_instances.py::test_the_full_aperture_limit_is_exact",
                "tests/test_b2_equiv_instances.py::test_the_enumerated_sub_aperture_estimator_is_unbiased",
                "tests/test_b2_equiv_instances.py::test_the_sub_aperture_sweep_converges",
                "tests/test_patch_wft.py",
            ),
            note=(
                "Both directions executed. The full-aperture limit reads 1.44e-12 "
                "against the independent float64 ASM at pad 0, reproducing the recorded "
                "1.4e-12; the ENUMERATED sub-aperture case reads 1.74e-15; and the drawn "
                "sweep converges 1.57 -> 0.81 -> 0.43 -> 0.19 over 4, 16, 64 and 225 "
                "patches at a clean P^-1/2 rate. The granularity sweep is the whole "
                "point of making patch_count a RepresentationParameter and it now "
                "exists.\n\n"
                "The enumerated case is what makes the sub-aperture direction GATEABLE. "
                "Drawing centres is a Monte Carlo estimate of a finite sum over draw "
                "positions; evaluating that sum exactly separates 'is the estimator "
                "unbiased' from 'how fast does it converge', and only the first is a "
                "gate. A biased estimator converges to the wrong answer, which a rate "
                "measurement cannot distinguish from slow convergence.\n\n"
                "It is also where the two blind spots are gated, and the blindness is "
                "measured rather than asserted: at full aperture the coverage correction "
                "and the launch phase are both exactly 1, so inverting the correction and "
                "double-counting the phase change the anchor by nothing. On the "
                "enumerated sub-aperture case the same mutations read 9.95e-1 and "
                "1.00e+0 against 1.7e-15. That is how an inverted coverage correction "
                "(A_patch/A_draw for A_draw/A_patch) survived once. The 9.95e-1 is "
                "|1/coverage^2 - 1| exactly, which is the check that the mutation is "
                "the inversion: the shipping correction A_draw/A_patch multiplies the "
                "emitted amplitude once, so inverting it is a factor coverage^-2. An "
                "earlier version of this control used coverage^+2 -- it SQUARED the "
                "correction instead of inverting it, read 2.2e+2, and still reported "
                "FIRED. A control that fires is not thereby a control that ran the "
                "mutation it claims, and this one could be wrong by a power because it "
                "mutates amplitudes from outside the emitter rather than flipping a "
                "switch inside the shipping path.\n\n"
                "The two exactness instances establish DIFFERENT things and the "
                "difference is load-bearing. The full-aperture anchor is scored at "
                "z = 1.26 mm against the independent ASM, so it pins the transfer "
                "function as well as the decomposition. The enumerated sub-aperture "
                "instance is scored AT z = 0, where the ASM degenerates to the identity "
                "for this configuration -- so it establishes the DECOMPOSITION identity "
                "and the two corrections, and NOT a propagation equivalence. No "
                "sub-aperture propagation claim rests on this family.\n\n"
                "One measured finding worth carrying: the equivalence holds AT THE PLANE "
                "and stops being a well-posed comparison at a large propagation distance "
                "on a finite window. The enumerated sum reproduces the field at the DOE "
                "plane to 1.7e-15 and disagrees with the ASM at z = 1.26 mm at 0.84, "
                "because a sub-aperture patch's modes live on its own pad-21 grid, which "
                "is not commensurate with the 15-px reconstruction grid -- so the ray sum "
                "is the non-periodic propagated field and the ASM is the periodic one. At "
                "full aperture with pad_factor 1 the two mode sets coincide, which is why "
                "the anchor can be compared there and this cannot. The clearance "
                "exemption on the anchor is preserved for the same family of reasons and "
                "is measured: padding it to factor 3 moves the score off 1e-12."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the independent ASM reference has to be recomputed at every drawn aperture, "
            "and it is the float64 half of the comparison."
        ),
        evidence=(
            "tests/test_patch_wft.py",
            "tests/test_planar_doe_step.py",
            "src/couplers/cascade.py",
            "benchmarks/reports/2026-08/coupler_characterization.md",
        ),
        notes=(
            "The two couplers this family speaks about are also the two with NO "
            "knowledge pack (benchmarks/validation/knowledge_pack_audit.md). Authoring "
            "them is M2.3's other half and is not done here."
        ),
)


# ---------------------------------------------------------------------------
# B2-ROUNDTRIP
# ---------------------------------------------------------------------------

#: The full-aperture anchor, then a sub-aperture ladder.
#:
#: ``patch_count = 1`` with ``patch_width_m = aperture_width_m`` is the
#: full-aperture limit, where one patch IS the window and the two routes
#: coincide. It is the exactness anchor and it is also BLIND to two defects that
#: are exactly 1 there -- the coverage correction and the launch phase -- which
#: is how an inverted coverage correction once survived. The ladder instances are
#: what see them.
#:
#: ``pad_width`` is the declared refinement axis and the clearance exemption on
#: the full-aperture instance is legitimate: padding a full-aperture single patch
#: moves the mode grid off the unpadded oracle's, and the exactness anchor reads
#: 0.57 instead of 1.4e-12. That is not a defect to pad away.
B2_EQUIV = register(
    _B2_EQUIV.with_instances(
        _B2_EQUIV.instantiate(
            "B2-EQUIV-FULL-01",
            {
                "aperture_width_m": 33 * 6.3e-6,
                "substrate_radius_m": float("inf"),
                "wavelength_m": 0.7e-6,
                "patch_count": 1,
                "patch_width_m": 33 * 6.3e-6,
                "tangent_plane_error_rad": 0.0,
                "grid_snapping": "snapped",
                "pad_width": 0,
            },
            expected={
                "relative_l2": 1.4e-12,
                "why": (
                    "one patch covering the whole aperture IS the window, so the patch "
                    "route and the global route coincide and the residual is float64 "
                    "round-off against the independent float64 ASM. The clearance "
                    "exemption is legitimate and must be preserved."
                ),
                "blind_to": (
                    "the coverage correction and the launch phase, both exactly 1 here"
                ),
            },
        ),
        *[
            _B2_EQUIV.instantiate(
                f"B2-EQUIV-SUB-{patches:03d}",
                {
                    "aperture_width_m": 15 * 6.3e-6,
                    "substrate_radius_m": float("inf"),
                    "wavelength_m": 0.7e-6,
                    "patch_count": patches,
                    "patch_width_m": 5 * 6.3e-6,
                    "tangent_plane_error_rad": 0.0,
                    "grid_snapping": "snapped",
                    "pad_width": 0,
                },
                expected={
                    "why": (
                        "many patches coherently summed must converge to the full-DOE "
                        "response, and the partition-of-unity argument behind that "
                        "convergence is exactly what an apodization taper breaks. Every "
                        "score names its oracle's pad, because an oracle is not well "
                        "defined until its padding is."
                    ),
                },
            )
            for patches in (4, 16, 64, 225)
        ],
        _B2_EQUIV.instantiate(
            "B2-EQUIV-SUB-ENUMERATED",
            {
                "aperture_width_m": 15 * 6.3e-6,
                "substrate_radius_m": float("inf"),
                "wavelength_m": 0.7e-6,
                # 361 = (15 + 5 - 1)^2, every draw position of the dilated
                # aperture exactly once. At exactly that count the driver
                # ENUMERATES rather than draws, which computes the estimator's
                # expectation instead of sampling it.
                "patch_count": 361,
                "patch_width_m": 5 * 6.3e-6,
                "tangent_plane_error_rad": 0.0,
                "grid_snapping": "snapped",
                "pad_width": 0,
            },
            expected={
                "relative_l2": 1.7e-15,
                "why": (
                    "drawing centres is a Monte Carlo estimate of a finite sum over "
                    "draw positions, and evaluating that sum exactly separates 'is the "
                    "estimator unbiased' from 'how fast does it converge'. Only the "
                    "first is a gate: a biased estimator converges to the WRONG answer, "
                    "which a rate measurement alone cannot distinguish from slow "
                    "convergence. This is the sub-aperture instance the 1e-9 tolerance "
                    "decides, and it is where the coverage correction and the launch "
                    "phase are gated -- neither is 1 here."
                ),
            },
        ),
    )
)


_B2_ROUNDTRIP = BenchmarkFamily(
        family_id="B2-ROUNDTRIP",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "what survives a round trip through two representations, and -- the part "
            "that gives the answer meaning -- does the deliberately broken twin fail?"
        ),
        components=("C_RAY_TO_WAVE", "C_WAVE_TO_RAY"),
        claim_kind=ClaimKind.ROUND_TRIP,
        parameters=(
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.32e-7
            ),
            PhysicalParameter(
                "numerical_aperture", "spectral extent of the input field",
                domain=(0.01, 0.95), default=0.3,
            ),
            NumericalParameter(
                "grid_n", "grid", domain=(16, 1024), default=64, refines_toward=1
            ),
            NumericalParameter(
                "sample_count",
                "samples in the Monte Carlo arm. The enumeration arm ignores it",
                domain=(int(1e3), int(1e9)),
                default=int(1e6),
                refines_toward=1,
            ),
            RepresentationParameter(
                "direction",
                "ray -> wave -> ray, or wave -> ray -> wave. They discard different "
                "things and are not the same claim",
                domain=("ray_wave_ray", "wave_ray_wave"),
                default="wave_ray_wave",
            ),
            RepresentationParameter(
                "arm",
                "the enumeration limit, or the Monte Carlo estimator over an ensemble",
                domain=("enumerated", "monte_carlo"),
                default="enumerated",
            ),
            ExecutionParameter("seed", "seed for the Monte Carlo arm", domain=(0, 2**31 - 1),
                               default=0),
            ExecutionParameter(
                "broken_twin_ran",
                "whether the deliberately broken twin was executed for this instance. "
                "An EXECUTION parameter because it is a property of how the run was "
                "conducted rather than of the physics -- and it defaults to False, so "
                "an instance that does not say it ran one is treated as not having",
                domain=(False, True),
                default=False,
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="BROKEN_TWIN_RAN",
                statement=(
                    "the deliberately broken twin was executed for this instance. A "
                    "round trip whose twin was not run has established nothing, and this "
                    "predicate is what makes that a validity condition rather than a "
                    "convention"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: boolean_margin(bool(p.get("broken_twin_ran", False))),
                blind_to=(
                    "whether the twin is broken in an INTERESTING way. It checks that "
                    "one ran, not that it probes the right convention",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.CONSERVATION_LAW,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the input itself. A round trip's reference needs no external source, "
                "which is its strength and also its weakness: a shared convention error "
                "cancels between the two directions"
            ),
            callable=None,
            reference="tests/test_coupler_round_trip.py",
        ),
        metrics=(
            Metric(
                name="round_trip_relative_rms",
                description="relative RMS of the returned field against the input",
                unit=None,
                definition="relative_rms",
                blind_to=(
                    "ANY ERROR THAT IS ITS OWN INVERSE. The backward pass undoes whatever "
                    "the forward pass did, so a wrong kernel round-trips perfectly. This "
                    "is the reason the broken twin is a validity condition and not a "
                    "nice-to-have",
                    "evanescent content, removed on the way out and unable to return",
                ),
            ),
            Metric(
                name="broken_twin_relative_rms",
                description=(
                    "the same round trip with a mismatched phase sign. Measured 1.40 "
                    "against the correct round trip's 1.32e-15 -- fifteen orders of "
                    "separation, which is what a control should look like"
                ),
                unit=None,
                definition="relative_rms",
                blind_to=("the same things the correct round trip is blind to",),
            ),
            Metric(
                name="detection_margin",
                description=(
                    "broken-twin error over correct-round-trip error. Reported as a "
                    "number rather than a boolean, because a control that fires by a "
                    "factor of 1.1 and one that fires by 1e15 are different evidence"
                ),
                unit=None,
                blind_to=("what the twin would do on a DIFFERENT instance",),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="PHASE_REFERENCE_CONSISTENCY",
                statement=(
                    "the registry-declared C_RAY_TO_WAVE invariant: the phase reference "
                    "survives the transition, which is exactly what the mismatched-sign "
                    "twin breaks"
                ),
                metric="round_trip_relative_rms",
                tolerance=Tolerance(
                    metric="round_trip_relative_rms",
                    threshold=1e-12,
                    basis=(
                        "measured 1.32e-15 at the enumeration limit; 1e-12 is three "
                        "orders above it and fifteen below the 1.40 the mismatched-sign "
                        "twin produces"
                    ),
                    basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                    may_gate=True,
                    rejects="a phase-sign mismatch, at 1.40",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="round_trip_relative_rms",
                threshold=1e-12,
                basis=(
                    "float64 round-off over the enumerated round trip: measured at "
                    "1.32e-15 with the deliberately mismatched twin at 1.40, a "
                    "separation of fifteen orders.\n\n"
                    "CHE-112 completed this basis after executing the family, and the "
                    "completion matters for reading the results. The floor is the "
                    "ENUMERATED arm's -- zero sampling error, so the only admissible "
                    "residual is arithmetic. A MONTE CARLO arm cannot meet it and is "
                    "not expected to: at 20,000 samples over 1024 modes it measures "
                    "3.1e-2 to 3.7e-2, which is sampling error and not a defect. The "
                    "sampled arm's accuracy claim is its ensemble statistics and its "
                    "broken twin, and its instances therefore report this metric UNMET "
                    "against a gate that belongs to the other arm rather than being "
                    "quietly exempted from it. `arm` is a RepresentationParameter for "
                    "exactly this reason: the two arms answer the same question with "
                    "different evidence."
                ),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects="a mismatched phase sign, which reads 1.40",
            ),
            Tolerance(
                metric="detection_margin",
                threshold=1e3,
                basis=(
                    "the measured margin is 1e15, and a margin below 1e3 would mean the "
                    "control and the signal are within three orders of each other -- "
                    "close enough that a change in either could swap them. That is the "
                    "claim, and it is NOT GATEABLE by this schema, which is why "
                    "may_gate is false here.\n\n"
                    "CHE-112 found the reason while executing the family. "
                    "``MetricResult.met`` is ``measured <= threshold`` everywhere, so a "
                    "quantity where LARGER IS BETTER cannot be expressed as a gating "
                    "tolerance: gating a detection margin at <= 1e3 asserts the opposite "
                    "of the claim, and a run whose control barely separated would report "
                    "green. Reporting the margin and refusing to gate it is the honest "
                    "state; inverting the number to fit the schema would hide the "
                    "limitation in a metric name.\n\n"
                    "What carries the under-powered-control finding instead is the "
                    "negative-control machinery itself: a control that did not fire sets "
                    "``undermines_the_gate`` on the result, which is exactly 'an "
                    "under-powered control is a finding about the benchmark' with the "
                    "direction the right way round. A two-sided or minimum-bound "
                    "tolerance kind is follow-up work on the schema."
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=False,
                rejects=(
                    "nothing, by construction -- see the basis. The claim it would make "
                    "is carried by the control outcomes instead."
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="mismatched-phase-sign",
                description=(
                    "the twin. Flip the phasor sign convention on one leg of the round "
                    "trip: 1.40 against 1.32e-15"
                ),
                mutation="use exp(-ikz) on the return leg and exp(+ikz) on the outbound",
                target_metric="broken_twin_relative_rms",
            ),
            NegativeControl(
                control_id="axis-transpose",
                description="transpose the axis order between the two legs",
                mutation="swap x and y on the return leg",
                target_metric="broken_twin_relative_rms",
            ),
            NegativeControl(
                control_id="off-axis-blindness-audit",
                description=(
                    "CHE-44's concern, made a control: evaluate the round-trip metric on "
                    "a centred window and on the whole array, with the defect placed off "
                    "axis. The centred metric must MISS it. A control battery whose "
                    "metrics are all centred cannot see an off-axis error, and nothing "
                    "had audited that"
                ),
                mutation="place the injected defect outside the centred window",
                target_metric="detection_margin",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
            VerificationStatus.BLOCKED,
        ),
        execution_policy=COUPLER_EXECUTION,
        stochastic_policy=StochasticPolicy(
            is_stochastic=True,
            required_evidence=(
                StochasticEvidenceKind.EXACTNESS_LIMIT,
                StochasticEvidenceKind.UNBIASEDNESS,
            ),
            minimum_seeds=3,
        ),
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="round_trip_relative_rms",
            observed=5.31117e-16,
            evidence=(
                "benchmarks/instances/b2_transitions.py",
                "tests/test_b2_transition_instances.py::test_both_round_trip_directions_return_the_input",
                "tests/test_b2_transition_instances.py::test_no_round_trip_is_accepted_without_a_failing_twin",
                "tests/test_coupler_round_trip.py",
            ),
            note=(
                "Executed in BOTH directions and BOTH arms. wave -> rays -> wave at the "
                "enumeration limit reads 5.31e-16 and ray -> wave -> ray reads 5.26e-16, "
                "each with its mismatched-phase twin at 1.414 -- a detection margin of "
                "2.7e15. That PAIRING is the whole point: a round trip that cannot be "
                "made to fail proves nothing, because a shared convention error cancels "
                "between the two directions.\n\n"
                "MET refers to the ENUMERATED arm. The Monte Carlo arms measure 3.1e-2 to "
                "3.7e-2 and report the same metric UNMET, which is sampling error rather "
                "than a defect -- see the tolerance's basis. Their claim is the ensemble "
                "and the twin, and their detection margins are 38x to 46x rather than "
                "1e15 because the correct arm has real variance.\n\n"
                "One construction had to be corrected and it is the reason the family is "
                "worth running rather than declaring. The probe field was a centred REAL "
                "Gaussian, whose spectrum is real and Hermitian-symmetric -- so "
                "conjugating it is a no-op and transposing it is a no-op, and BOTH broken "
                "twins read identically to the correct arm at 5.4e-16. A round trip that "
                "cannot be made to fail proves nothing, and a probe that cannot see a "
                "sign flip is exactly how that happens. The probe is now offset, "
                "elliptical and phase-ramped, which is CHE-44's centre-dependent "
                "blindness concern answered by construction rather than audited after."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.GENERATION_WEAKENS_INDEPENDENCE,
        sampler_absent_note=(
            "a round trip's reference is its own input, so generating instances is cheap "
            "-- and every generated instance needs its broken twin generated too, or the "
            "family produces round trips with no controls. A sampler that emitted only "
            "the correct arm would quietly weaken every instance it drew, which is a "
            "sampler design problem rather than a budget one."
        ),
        evidence=(
            "tests/test_coupler_round_trip.py",
            "benchmarks/reports/2026-08/coupler_characterization.md",
            "src/verification/metrics.py",
        ),
        notes=(
            "What provably does not survive is declared in WHAT_DOES_NOT_SURVIVE above, "
            "per transition. An agent reasoning about a chain of representations needs "
            "to know which quantities it may still ask about downstream, and 'the round "
            "trip closed' does not answer that."
        ),
)



#: Both directions, both arms, and a broken twin for each.
#:
#: The schema rule this family exists to make mechanical: a successful round trip
#: is not accepted unless a deliberately broken twin demonstrably failed. A shared
#: convention error cancels between the two directions, so a round trip that
#: cannot be made to fail proves nothing -- and the ``BROKEN_TWIN_RAN`` predicate
#: is what stops one being reported alone.
B2_ROUNDTRIP = register(
    _B2_ROUNDTRIP.with_instances(
        *[
            _B2_ROUNDTRIP.instantiate(
                f"B2-ROUNDTRIP-{direction.replace('_', '').upper()}-{arm.upper()}-{seed:02d}",
                {
                    "wavelength_m": 5.32e-7,
                    "numerical_aperture": 0.3,
                    "grid_n": 32,
                    "sample_count": 20000,
                    "direction": direction,
                    "arm": arm,
                    "seed": seed,
                    "broken_twin_ran": True,
                },
                seed=seed,
                expected={
                    "why": (
                        "the enumerated arm is exact and gates; the Monte Carlo arm is "
                        "an ensemble and reports statistics. Neither is accepted without "
                        "its broken twin."
                    ),
                },
            )
            for direction in ("wave_ray_wave", "ray_wave_ray")
            for arm, seeds in (("enumerated", (0,)), ("monte_carlo", (1, 2, 3)))
            for seed in seeds
        ]
    )
)
