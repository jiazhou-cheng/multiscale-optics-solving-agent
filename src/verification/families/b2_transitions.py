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
    asm_transfer_function_sampling,
    boolean_margin,
    declared_planarity,
    fractional_margin,
    si_s3_curvature_bound,
)
from verification.families.registry import register
from verification.families.schema import (
    BenchmarkCategory,
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

B2_R2W_EXACT = register(
    BenchmarkFamily(
        family_id="B2-R2W-EXACT",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
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
            kind=Oracle.DETERMINISTIC_LIMIT,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the enumerated reference: the same sum with nothing sampled away. "
                "Independent of the SAMPLING and not of the kernel, which is the exact "
                "scope of what an exactness limit establishes"
            ),
            callable=None,
            reference="benchmarks/probes/ray_wave/demo3_enumerated_reference.py",
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
                    "any error the kernel and the enumeration share. The enumeration is "
                    "the same arithmetic without the sampling, so a wrong kernel is "
                    "wrong in both and cancels -- this establishes the sampling, not the "
                    "physics",
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="exactness_relative_l2_field",
            observed=7.1e-13,
            evidence=(
                "benchmarks/reports/2026-08/coupler_characterization.md",
                "benchmarks/probes/records/ray_wave/demo3_enumerated_reference_rwf_ramp.json",
            ),
            note=(
                "the full-aperture patch against an independent float64 ASM reads "
                "7.1e-13 and the wave-rays-wave round trip at the enumeration limit "
                "reads 1.32e-15. On record; nothing in the required gate re-runs the "
                "enumerated reference, which is the expensive half."
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
)


# ---------------------------------------------------------------------------
# B2-R2W-ROUTE
# ---------------------------------------------------------------------------

B2_R2W_ROUTE = register(
    BenchmarkFamily(
        family_id="B2-R2W-ROUTE",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
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
            asm_transfer_function_sampling(
                distance_key="propagation_distance_m",
                pitch_key="sample_pitch_m",
                grid_key="grid_n",
                wavelength_key="wavelength_m",
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
                blind_to=("absolute power, which the ratio below is for",),
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="route_power_ratio",
            observed=0.9832,
            evidence=("benchmarks/reports/2026-08/kspace_ray_to_wave.md",),
            note=(
                "the asymmetry is the finding: 7.1e-13 agreement on demo2 and a 1.7% "
                "power loss on demo3 at 8x oversampling, from the same route. The fast "
                "route is 9.6x faster on its kernel and it is NOT interchangeable with "
                "the exact one, which is why this family exists separately from "
                "B2-R2W-EXACT."
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
)


# ---------------------------------------------------------------------------
# B2-W2R-STOCH
# ---------------------------------------------------------------------------

B2_W2R_STOCH = register(
    BenchmarkFamily(
        family_id="B2-W2R-STOCH",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
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
            status=GateStatus.NOT_MEASURED,
            note=(
                "The four evidence kinds exist separately in the repository -- the "
                "enumeration limit and the round trip are established, the variance "
                "characterization is CHE-120's -- and nothing has run them as one family "
                "over a declared ensemble. NO GRADIENT IS CLAIMED: "
                "derivative.verified is false for this coupler and the surrogate's bias "
                "is characterized rather than validated."
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
)


# ---------------------------------------------------------------------------
# B2-EQUIV
# ---------------------------------------------------------------------------

B2_EQUIV = register(
    BenchmarkFamily(
        family_id="B2-EQUIV",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
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
            si_s3_curvature_bound(),
            declared_planarity(),
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
                    "the full-aperture (one-patch) case agrees with an independent "
                    "float64 ASM at 7.1e-13, so at patch_count = 1 the admissible error "
                    "is round-off. 1e-9 leaves three orders of headroom for the "
                    "sub-aperture cases the family sweeps, and still refuses a "
                    "decomposition that has lost the overlap"
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="patch_vs_global_relative_l2",
            observed=7.1e-13,
            evidence=(
                "tests/test_patch_wft.py",
                "benchmarks/reports/2026-08/coupler_characterization.md",
            ),
            note=(
                "the full-aperture case against the independent float64 ASM reads "
                "7.1e-13. What is NOT measured as one family is the granularity sweep -- "
                "the whole point of making patch_count a RepresentationParameter -- and "
                "the sub-aperture convergence that would show the decomposition is "
                "consistent at every granularity rather than only at one."
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
)


# ---------------------------------------------------------------------------
# B2-ROUNDTRIP
# ---------------------------------------------------------------------------

B2_ROUNDTRIP = register(
    BenchmarkFamily(
        family_id="B2-ROUNDTRIP",
        family_version="1.0.0",
        category=BenchmarkCategory.B2,
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
                    "the enumeration-limit round trip reads 1.32e-15. 1e-12 admits "
                    "float64 accumulation over a larger grid and nothing else"
                ),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects="a mismatched phase sign, which reads 1.40",
            ),
            Tolerance(
                metric="detection_margin",
                threshold=1e3,
                basis=(
                    "the measured margin is 1e15. A margin below 1e3 would mean the "
                    "control and the signal are within three orders of each other, which "
                    "is close enough that a change in either could swap them. This gates "
                    "the CONTROL, not the physics -- an under-powered control is a "
                    "finding about the benchmark"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects="a broken twin that barely fails, which is a twin that proves little",
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="round_trip_relative_rms",
            observed=1.32e-15,
            evidence=(
                "tests/test_coupler_round_trip.py",
                "benchmarks/reports/2026-08/coupler_characterization.md",
            ),
            note=(
                "wave -> rays -> wave at the enumeration limit reads 1.32e-15, and the "
                "same round trip with a mismatched phase sign reads 1.40 -- detected. "
                "That PAIRING is the whole point: a round trip that cannot be made to "
                "fail proves nothing, because a shared convention error cancels between "
                "the two directions."
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
)
