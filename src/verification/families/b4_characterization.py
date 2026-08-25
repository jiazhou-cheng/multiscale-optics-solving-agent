"""B4: measured, reported, and structurally unable to decide anything.

CHE-134 (M4.3). The old M4 asked for a "realistic research-style workflow"
benchmark as its third tier. There is one -- demo3, the paper's Fig 5c system --
and it has no conventional reference. The paper says so; that is the point of
the figure. Every demo3 number is a self-comparison: one realization against
another, or one arm of a probe against another arm.

CHE-120 already recorded that honestly in ``manifest.yaml``'s
``characterizations:`` block. This module makes it **structural** rather than a
comment somebody has to read: a B4 family cannot carry a gating tolerance, the
schema enforces it at construction, and the label cannot quietly be promoted.

What a B4 family is graded on: convergence, uncertainty, seed-to-seed
statistics, reproducibility, energy accounting, independent-route consistency
and intermediate-state evidence. All measured, all reported, **none of them a
threshold.**

The ``why_not`` prose below is migrated from the manifest unreworded. It is the
most valuable sentence in that file and rewriting it to sound better would be
the first step toward rewriting it to sound passable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import fractional_margin
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

__all__ = ["B4_COST", "B4_DEMO3", "B4_DUALROUTE_AGREEMENT", "DEMO3_WHY_NOT"]


#: Migrated VERBATIM from ``benchmarks/manifest.yaml``'s ``characterizations:``
#: block (CHE-120). Not reworded.
DEMO3_WHY_NOT = (
    "The paper states no conventional reference exists for this system; that is the "
    "point of its Fig 5c. Every demo3 number is a self-comparison -- one realization "
    "against another, or one arm of a probe against another arm. Nothing is scored "
    "against an independent oracle and no recorded threshold is a gate. This CANNOT be "
    "fixed by more compute: the missing thing is a reference, not a budget."
)

DEMO3_WHAT_IT_SUPPORTS = (
    "Characterization of cost, convergence rate and estimator variance; anti-bias work, "
    "because unbiasedness IS testable at a reduced size where an enumerated oracle "
    "exists (tests/test_patch_positions.py, tests/test_patch_wft.py) and transfers to "
    "demo3 by construction; and relative claims between two arms of one probe. The "
    "components are validated elsewhere -- the full-aperture patch against an "
    "independent float64 ASM at 7.1e-13, and demo2 against the same oracle. demo3 is "
    "those validated components composed at a scale where the composition has no "
    "reference of its own."
)


GPU_ENVELOPE = ExecutionPolicy(
    devices=frozenset({DeviceKind.CUDA}),
    dtypes=frozenset({DType.COMPLEX64}),
    namespaces=frozenset({ArrayNamespace.JAX}),
    #: 0.14 h per run on the shipped estimator at the NCC-0.9 budget, and the
    #: family sweeps several. Tier-X: these do not join the required gate.
    max_wall_seconds=3600.0,
    max_peak_memory_gib=40.0,
    notes=(
        "One GPU, per the shared-host policy. These are the expensive families and they "
        "are deliberately outside the required gate: a suite that ran them on every "
        "change would stop being run."
    ),
)


# ---------------------------------------------------------------------------
# B4-DEMO3
# ---------------------------------------------------------------------------

B4_DEMO3 = register(
    BenchmarkFamily(
        family_id="B4-DEMO3",
        family_version="1.0.0",
        category=BenchmarkCategory.B4,
        question=(
            "what does the patch-route hologram plus refractive lens "
            "(C_PATCH_WFT -> M_RAY_OPTILAND -> C_RAY_TO_WAVE) cost to converge, and how "
            "does its estimator variance behave? NOT: is it right -- there is nothing "
            "to be right against."
        ),
        components=("C_PATCH_WFT", "C_RAY_TO_WAVE"),
        claim_kind=ClaimKind.CONVERGENCE,
        parameters=(
            PhysicalParameter(
                "phase_profile",
                "the paper's Fig 5c smile profile; input data rather than a knob",
                domain=("demo3_smile",),
                default="demo3_smile",
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(4e-7, 8e-7), default=5.32e-7
            ),
            NumericalParameter(
                "ray_count",
                "traced rays. The measured ladder is 20/30/40M; the extrapolated "
                "NCC-0.9 budget is 1.49e9 on the shipped estimator",
                domain=(int(1e6), int(1e10)),
                default=int(4e7),
                refines_toward=1,
            ),
            NumericalParameter(
                "oversampling",
                "k-grid oversampling factor",
                domain=(1, 16),
                default=8,
                refines_toward=1,
            ),
            ExecutionParameter(
                "seed",
                "estimator seed. MANDATORY: one realization is not a result here, and "
                "the family's whole instrument is seed-to-seed statistics",
                domain=(0, 2**31 - 1),
                default=0,
            ),
            ExecutionParameter(
                "seed_count",
                "how many seeds the ensemble has. Declared rather than read with a "
                "default, so a family that forgets to say fails loudly instead of "
                "being treated as a single realization",
                domain=(1, 1024),
                default=1,
            ),
            RepresentationParameter(
                "reconstruction_route",
                "RAMP_SUM or the k-space splat. On demo2 they agree to 7.1e-13; on demo3 "
                "at 8x oversampling the k-space route loses 1.7% of the power. That "
                "asymmetry is the thing to characterize, and it is why the route is a "
                "declared parameter rather than an implementation detail",
                domain=("ramp_sum", "kspace_splat"),
                default="ramp_sum",
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="ENSEMBLE_SIZE",
                statement=(
                    "enough seeds have been run for a seed-to-seed statistic to mean "
                    "anything"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: fractional_margin(4.0, float(p["seed_count"])),
                blind_to=(
                    "whether the seeds are independent, which depends on the sampler and "
                    "not on how many were drawn",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.NONE,
            independence=OracleIndependence.NOT_APPLICABLE,
            description=(
                "There is none, and there cannot be one. " + DEMO3_WHY_NOT
            ),
            callable=None,
            reference="benchmarks/manifest.yaml characterizations: DEMO3-RW-P",
        ),
        metrics=(
            Metric(
                name="seed_to_seed_ncc",
                definition="ncc",
                description=(
                    "normalized cross-correlation between two independent realizations "
                    "at the same budget. An estimator-variance instrument, not an "
                    "accuracy one: two realizations of a biased estimator agree perfectly"
                ),
                unit=None,
                blind_to=(
                    "bias entirely. Two runs of the same wrong estimator correlate at "
                    "1.0, which is precisely why this number can never be a gate",
                    "an overall scale factor, which NCC normalizes away",
                ),
            ),
            Metric(
                name="ncc_vs_ray_count_slope",
                description=(
                    "log-log slope of seed-to-seed NCC against ray count, with its fit "
                    "quality. Measured 0.93-0.96 over 20/30/40M rays"
                ),
                unit=None,
                blind_to=(
                    "curvature in the trend: a slope fitted over three rungs cannot see "
                    "a knee outside them, and the extrapolation to 1.5e9 is two decades "
                    "past the furthest measured point",
                ),
            ),
            Metric(
                name="stage_power_ratio",
                definition="power_ratio",
                description=(
                    "power out over power in, at EVERY representation change, so a loss "
                    "is attributable to a stage rather than to the run"
                ),
                unit=None,
                blind_to=("where within a stage the power went",),
            ),
            Metric(
                name="route_agreement_ncc",
                definition="ncc",
                description=(
                    "RAMP_SUM against the k-space splat on the same instance. CROSS_ROUTE "
                    "and reported as characterization"
                ),
                unit=None,
                blind_to=(
                    "any error the two routes share, which is most of the pipeline: both "
                    "consume the same trace and the same patch decomposition",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="stage_power_ratio",
                threshold=2e-2,
                basis=(
                    "REPORTING THRESHOLD, NOT A GATE. The k-space route loses 1.7% of "
                    "the power at 8x oversampling on this system and agrees to 7.1e-13 "
                    "on demo2, so 2e-2 is the number that makes the demo3 loss visible "
                    "as an entry rather than a footnote. may_gate is False because this "
                    "family has no oracle: a power ratio inside a threshold says nothing "
                    "about whether the field is right"
                ),
                basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
                may_gate=False,
            ),
            Tolerance(
                metric="route_agreement_ncc",
                threshold=0.99,
                basis=(
                    "REPORTING THRESHOLD. Two routes through our own code, so by the "
                    "schema's rule this cannot gate however well they agree"
                ),
                basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
                may_gate=False,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="single-seed-is-not-a-result",
                description=(
                    "report a seed-to-seed NCC from one realization. It is undefined, and "
                    "a family that produced a number anyway would be fabricating the "
                    "ensemble"
                ),
                mutation="run one seed and ask for the seed-to-seed statistic",
                target_metric="seed_to_seed_ncc",
            ),
            NegativeControl(
                control_id="slope-from-two-rungs",
                description=(
                    "fit the log-log slope over two ray counts. Two points always fit a "
                    "line perfectly, so the fit quality is the only thing that "
                    "distinguishes a trend from an artifact"
                ),
                mutation="drop the middle rung from the ladder before fitting",
                target_metric="ncc_vs_ray_count_slope",
            ),
        ),
        failure_semantics=(
            VerificationStatus.UNCONVERGED,
            VerificationStatus.BLOCKED,
            VerificationStatus.OUT_OF_VALIDITY,
        ),
        execution_policy=GPU_ENVELOPE,
        stochastic_policy=StochasticPolicy(
            is_stochastic=True,
            required_evidence=(
                StochasticEvidenceKind.EXACTNESS_LIMIT,
                StochasticEvidenceKind.UNBIASEDNESS,
                StochasticEvidenceKind.CONVERGENCE_EXPONENT,
                StochasticEvidenceKind.VARIANCE_CHARACTERIZATION,
            ),
            minimum_seeds=4,
        ),
        gate_disposition=GateDisposition(
            status=GateStatus.CHARACTERIZED_NO_GATE,
            note=(
                "Permanently. " + DEMO3_WHY_NOT + " What it DOES support: "
                + DEMO3_WHAT_IT_SUPPORTS
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "generative in ray count, oversampling and seed, and it should be sampled "
            "there once M9 exists -- the parameter space is the point of the family. It "
            "is None now because each draw is a GPU-hour, so a sampler without a budget "
            "policy would be a way to spend the machine rather than to learn something."
        ),
        evidence=(
            "benchmarks/reports/2026-08/demo3_estimator_variance.md",
            "benchmarks/probes/ray_wave/demo3_convergence.py",
            "benchmarks/probes/ray_wave/demo3_hologram_lens.py",
        ),
        notes=(
            "The result worth stating: the extrapolated NCC-0.9 budget is 1.74e9-2.37e9 "
            "rays against the paper's own Table S2 choice of 2.6e9. Landing within 1.5x "
            "of an independent group's budget is evidence that the budget is a property "
            "of the SYSTEM rather than of our implementation -- a real scientific result, "
            "from a family that can never gate anything."
        ),
    )
)


# ---------------------------------------------------------------------------
# B4-COST
# ---------------------------------------------------------------------------


def _no_oracle(_params: Mapping[str, Any]) -> Any:
    raise NotImplementedError(
        "a cost family has no oracle: there is no correct number of seconds"
    )


B4_COST = register(
    BenchmarkFamily(
        family_id="B4-COST",
        family_version="1.0.0",
        category=BenchmarkCategory.B4,
        question=(
            "what does each node and each route cost, and how much of a run is framework "
            "overhead rather than physics?"
        ),
        components=("M_RAY_OPTILAND", "M_WAVE_CHROMATIX", "C_RAY_TO_WAVE", "C_PATCH_WFT"),
        # A new axis of the coverage matrix, added with this family. Cost is not
        # device parity: the same device can be fast or slow and both are right.
        claim_kind=ClaimKind.COST,
        parameters=(
            PhysicalParameter(
                "workload",
                "which configuration is being priced. A different workload is a "
                "different measurement, not a different sampling of one",
                domain=("l2_psf_01", "demo2_paper", "demo3_characterization"),
                default="l2_psf_01",
            ),
            NumericalParameter(
                "ray_count", "traced rays", domain=(int(1e3), int(1e10)), default=787969,
                refines_toward=1,
            ),
            RepresentationParameter(
                "route",
                "RAMP_SUM or k-space splat. 9.6x faster on its kernel, and 1.7% of the "
                "power lighter on demo3 at 8x oversampling. Route choice is a scientific "
                "decision with a measurable cost, and an agent cannot reason about it "
                "from a number nobody recorded",
                domain=("ramp_sum", "kspace_splat"),
                default="ramp_sum",
            ),
            ExecutionParameter("device", "cpu or cuda", domain=("cpu", "cuda"), default="cuda"),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="CALIBRATED_ENVIRONMENT",
                statement=(
                    "the environment fingerprint matches the one the cost model was "
                    "calibrated on; off it, the estimator returns None rather than a "
                    "number, which is the honest answer"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: 1.0 if p.get("environment_calibrated", True) else -1.0,
                blind_to=(
                    "machine load. This is a shared server and a calibrated environment "
                    "under contention is still slow",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.NONE,
            independence=OracleIndependence.NOT_APPLICABLE,
            description=(
                "There is no correct number of seconds. What is checkable is the cost "
                "MODEL against the measurement, and that is a calibration question rather "
                "than a physics one"
            ),
            callable=None,
            reference="benchmarks/perf/run_baselines.py; benchmarks/perf/records/",
        ),
        metrics=(
            Metric(
                name="wall_seconds",
                description="end-to-end wall time on the declared device",
                unit="s",
                blind_to=(
                    "machine contention, which on a shared server is the largest single "
                    "term and is not a property of the code",
                ),
            ),
            Metric(
                name="framework_overhead_fraction",
                description=(
                    "the share of wall time that is not solver time: validation, "
                    "ordering, plumbing, record construction"
                ),
                unit=None,
                blind_to=(
                    "where inside the framework the time went -- it is one number over "
                    "the whole run",
                ),
            ),
            Metric(
                name="estimate_over_actual",
                description=(
                    "the cost model's prediction divided by what it cost. M0.4 scores "
                    "the estimator on this"
                ),
                unit=None,
                blind_to=(
                    "an estimator that is right on average and wrong on every instance",
                ),
            ),
            Metric(
                name="peak_memory_gib",
                description="peak resident memory, host or device",
                unit="GiB",
                blind_to=("transient allocations shorter than the sampling interval",),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="framework_overhead_fraction",
                threshold=0.10,
                basis=(
                    "REPORTING THRESHOLD. M0.4's target is 10%, and the number that "
                    "matters is whichever one it actually is -- a cost family that failed "
                    "a run for being slow would be measuring the machine"
                ),
                basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
                may_gate=False,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="uncalibrated-environment",
                description=(
                    "ask for an estimate on an environment the model was not calibrated "
                    "on. It must return None rather than a number"
                ),
                mutation="run with a package version outside the calibration set",
                target_metric="estimate_over_actual",
            ),
            NegativeControl(
                control_id="warm-cache-is-not-a-measurement",
                description=(
                    "price a workload that was served from the executor's cache. The "
                    "number is real and it is not the cost of the physics"
                ),
                mutation="run the same instance twice with caching enabled",
                target_metric="wall_seconds",
                expectation=NegativeControlExpectation.NOT_IMPLEMENTED,
                caveat=(
                    "declared and not run: the perf harness does not use the executor's "
                    "cache yet, so there is nothing to exercise. It is declared because "
                    "the hazard arrives the moment CHE-115 routes the baselines through "
                    "the executor."
                ),
            ),
        ),
        failure_semantics=(
            VerificationStatus.BLOCKED,
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNSUPPORTED,
        ),
        execution_policy=GPU_ENVELOPE,
        stochastic_policy=StochasticPolicy(
            is_stochastic=False,
            determinism_reason=(
                "timing varies run to run and that is machine noise rather than a "
                "sampled quantity; the harness reports a spread over repeats without "
                "drawing anything"
            ),
        ),
        gate_disposition=GateDisposition(
            status=GateStatus.CHARACTERIZED_NO_GATE,
            note=(
                "Cost is measured and reported, never gated. A regression in it is a "
                "finding for a human, not a failed benchmark -- the machine is shared and "
                "a threshold on seconds would fire on contention."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "a cost baseline is only comparable against the same configuration on the "
            "same environment fingerprint. Sampling the parameter space would produce "
            "numbers with nothing to compare them to."
        ),
        evidence=(
            "benchmarks/perf/run_baselines.py",
            "benchmarks/perf/records/framework_overhead.json",
            "benchmarks/perf/records/estimate_accuracy.json",
            "benchmarks/reports/2026-08/performance_baselines.md",
        ),
    )
)


# ---------------------------------------------------------------------------
# B4-DUALROUTE-AGREEMENT
# ---------------------------------------------------------------------------

B4_DUALROUTE_AGREEMENT = register(
    BenchmarkFamily(
        family_id="B4-DUALROUTE-AGREEMENT",
        family_version="1.0.0",
        category=BenchmarkCategory.B4,
        question=(
            "how far apart are the three PSF routes on the Cooke triplet, and what "
            "accounts for the gap? NOT: which of them is right -- all three are our own "
            "code."
        ),
        components=("M_RAY_OPTILAND", "C_RAY_TO_WAVE"),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "field_angle_deg",
                "field angle. The two measured points behave completely differently and "
                "that IS the finding",
                unit="deg",
                domain=(0.0, 20.0),
                default=20.0,
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(4e-7, 8e-7), default=5.5e-7
            ),
            NumericalParameter(
                "pupil_rings", "hexapolar rings", domain=(8, 256), default=64, refines_toward=1
            ),
            RepresentationParameter(
                "route_pair",
                "which two routes are compared",
                domain=("fft_vs_huygens", "fft_vs_ray_to_wave", "huygens_vs_ray_to_wave"),
                default="huygens_vs_ray_to_wave",
            ),
        ),
        validity=(),
        oracle=FamilyOracle(
            kind=Oracle.CROSS_ROUTE,
            independence=OracleIndependence.SHARES_CODE,
            description=(
                "Two routes through our own code. FFTPSF and HuygensPSF share one "
                "Wavefront/OPD front end -- same reference sphere, same launch-tilt "
                "removal, same pupil sampling -- so the pair is not independent in the "
                "way an analytic oracle would be, and their residual UNDERSTATES the "
                "uncertainty. ray->wave is the only route with a different front end and "
                "still consumes the same trace"
            ),
            callable=None,
            reference="benchmarks/reports/2026-08/cooke_triplet_psf_routes.md",
        ),
        metrics=(
            Metric(
                name="route_pair_relative_l2",
                definition="relative_l2_intensity",
                description="relative L2 between the two routes' PSF intensities",
                unit=None,
                blind_to=(
                    "which of the two is wrong. That is the defining property of a "
                    "cross-route comparison and the reason it cannot gate",
                ),
            ),
            Metric(
                name="peak_separation_px",
                description="separation of the two routes' PSF peaks, in samples",
                unit="px",
                blind_to=("shape; two differently-shaped PSFs can peak at the same place",),
            ),
            Metric(
                name="fitted_scale_anisotropy",
                description=(
                    "the (s_x, s_y) scale fit of one route onto the other. The instrument "
                    "that ATTRIBUTED the off-axis disagreement rather than just measuring "
                    "it"
                ),
                unit=None,
                blind_to=("anything that is not a pure scale-and-registration error",),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="route_pair_relative_l2",
                threshold=1e-2,
                basis=(
                    "REPORTING THRESHOLD. The resampling-artefact floor of this "
                    "comparison is a few times 1e-3, and on axis all three pairs sit at "
                    "0.0053-0.0085 -- essentially at that floor. 1e-2 is the line between "
                    "'at the floor' and 'something to explain'. It cannot gate: a "
                    "CROSS_ROUTE oracle decides nothing"
                ),
                basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
                may_gate=False,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="agreement-is-not-correctness",
                description=(
                    "two routes sharing a front end agree by construction on anything "
                    "that front end gets wrong. The FFTPSF/HuygensPSF pair is the live "
                    "case and its residual understates the true uncertainty"
                ),
                mutation=(
                    "introduce a common-mode error in the shared Wavefront/OPD front end "
                    "and observe that fft_vs_huygens does not move"
                ),
                target_metric="route_pair_relative_l2",
                expectation=NegativeControlExpectation.NOT_IMPLEMENTED,
                caveat=(
                    "declared and not run: injecting into Optiland's front end means "
                    "patching an installed package, which this project does not do. It "
                    "is declared because it is the exact reason this family is B4, and "
                    "leaving it unstated would make the categorization look arbitrary."
                ),
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=ExecutionPolicy(
            devices=frozenset({DeviceKind.CPU}),
            dtypes=frozenset({DType.COMPLEX64}),
            namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
            max_wall_seconds=900.0,
            max_peak_memory_gib=8.0,
        ),
        stochastic_policy=StochasticPolicy(
            is_stochastic=False,
            determinism_reason="three deterministic PSF computations on one traced bundle",
        ),
        gate_disposition=GateDisposition(
            status=GateStatus.CHARACTERIZED_NO_GATE,
            note=(
                "Measured, attributed, and unable to decide anything. On axis all three "
                "routes sit at the resampling floor (0.0053-0.0085 relative L2). At 20 "
                "degrees FFTPSF is the outlier at 0.313-0.315 against the other two, "
                "which agree to 0.0138 -- and the cause is identified rather than left "
                "open: the image-space pupil is anisotropic (F/#x 5.284, F/#y 6.030, "
                "ratio 1.141) against the single scalar working F/# 5.480 that FFTPSF "
                "uses to set its pixel scale. Predicted mis-scaling s_y 1.100 / s_x "
                "0.964; measured by scale fit s_y 1.0955 / s_x 0.967. An attributed "
                "cross-route discrepancy is worth more than an unattributed agreement."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the finding is about a specific pupil anisotropy at a specific field angle. "
            "A drawn field angle would mostly land where the routes agree and the family "
            "has nothing to say."
        ),
        evidence=(
            "benchmarks/reports/2026-08/cooke_triplet_psf_routes.md",
            "benchmarks/probes/cooke_triplet_psf_routes.py",
        ),
        notes=(
            "Split out of what M4.1 called B3-DUALROUTE. See b3_composed.py's "
            "B3-DUALROUTE for the reasoning: the invariant can decide something, the "
            "route comparison cannot, and putting both in one family would have let the "
            "second borrow the first's authority."
        ),
    )
)
