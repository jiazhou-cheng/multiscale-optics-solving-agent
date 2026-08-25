"""B3: composed chains whose correctness can still be *decided*.

CHE-116 (M4.1). The old framing asked for composed benchmarks at three
"complexity tiers", which classifies by how many components are involved --
not the question that decides anything. The question is *what decides
correctness*, and that is what splits these from B4: a composed case with an
oracle and a composed case without one are not two points on one scale, and
calling both "benchmarks at different tiers" is exactly how a characterization
gets planned against as a validation.

Admissible deciders here: a closed form, a genuinely independent route, a
well-justified equivalence, or intermediate invariants.

What authoring these found
--------------------------
``B3-DUALROUTE`` could not be authored as the issue described it. The Cooke
triplet has no analytic PSF, and the three routes available -- Optiland's
``FFTPSF``, Optiland's ``HuygensPSF``, and ray->coupler->wave -- are all our own
code: the first two share one ``Wavefront``/OPD front end (PB7/CHE-58 finding
F2) and the third shares the trace. There is no independent leg to promote, and
the schema refuses a ``CROSS_ROUTE`` oracle outside B4.

So the family split, along the line the evidence already draws:

* ``B3-DUALROUTE`` gates on an **intermediate invariant** -- energy accounting
  through each route -- which the issue lists as an admissible decider and which
  needs no external reference at all.
* ``B4-DUALROUTE-AGREEMENT`` (in ``b4_characterization.py``) carries the route
  comparison itself, declared ``CROSS_ROUTE``, unable to gate, and reporting the
  attributed off-axis discrepancy rather than hiding it.

That is a better answer than either forcing an independent label onto a
cross-route comparison or dropping the case. It is recorded here because the
issue asked for one family and got two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import (
    fractional_margin,
    paraxial_field_angle,
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
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = ["B3_DEMO2", "B3_DUALROUTE", "B3_PSF_SINGLET"]


COMPOSED_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    dtypes=frozenset({DType.COMPLEX64}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    max_wall_seconds=600.0,
    max_peak_memory_gib=16.0,
    notes=(
        "The trace runs float64 on the numpy backend and the propagation runs "
        "complex64 in JAX, because Chromatix has no other option. The dtype declared "
        "here is the one the *answer* is computed in, which is the lower of the two."
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "a hexapolar trace at a declared ring count followed by an angular-spectrum "
        "propagation over a declared grid; nothing draws a sample"
    ),
)


# ---------------------------------------------------------------------------
# B3-PSF-SINGLET
# ---------------------------------------------------------------------------


def _airy_first_null_um(params: Mapping[str, Any]) -> float:
    """O1: the analytic Airy first null, ``0.61 lambda / NA``.

    Paraxial and aberration-free, and it shares no code and no traced data with
    the coupler. That is the whole reason this family is B3 and not B4: nothing
    else in the composed set can claim an independent decider.
    """
    return 0.61 * float(params["wavelength_m"]) / float(params["numerical_aperture"]) * 1e6


#: Migrated VERBATIM from ``benchmarks/physics/L2-PSF-01/tolerances.yaml``.
#: These strings are the hardest thing in the repository to reconstruct, and
#: none is reworded to sound better.
_FFT_ORACLE_BASIS = (
    "benchmarks/protocols/slice_protocol.yaml tolerance_budget.gates, unchanged since "
    "M3.2 (CHE-31). CHE-38's synthetic aberration-free diagnostic reaches 4.07e-4 "
    "(inside gate); CHE-47's real-traced-system measurement of the production "
    "(weighted) configuration reaches 2.21e-3 vs O1 (outside gate, by ~2.2x). Against "
    "O1 only, the uniform (pre-CHE-47) configuration reaches 9.21e-4 and clears the "
    "gate -- the opposite ordering from what the O2 oracle measured, which is itself "
    "evidence that O2's own pupil-fit resolution should not be trusted to decide "
    "correctness here. Open item, not closed by this bundle: why the production "
    "quadrature weight, which fixes the unrelated absolute-power N^2 divergence (see "
    "accuracy.production. absolute_power), does not also improve relative-L2 agreement "
    "with O1 on this aberrated system."
)

_OPL_SIGN_FLIP_BASIS = (
    "Anything below this would mean the negative control cannot distinguish a "
    "scrambled wavefront from a converging one."
)

_QUADRATURE_BASIS = (
    "benchmarks/probes/records/m3_quadrature_weight.json finest_configuration. "
    "improvement_factor_vs_o1 = 0.42 at 787,969 rays (uniform is CLOSER to O1 than "
    "weighted is: 9.21e-4 vs 2.21e-3) -- below the 1.2 floor, so this control "
    "currently reads detected=false. The previous basis for this floor "
    "(improvement_factor_vs_o2_asm = 1.575) used O2, our own custom ASM/RS "
    "propagator, as the decisive oracle; that was circular validation and has been "
    "retired as the gate input, though the O2 number is still reported in result.json "
    "for characterization."
)


B3_PSF_SINGLET = register(
    BenchmarkFamily(
        family_id="B3-PSF-SINGLET",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        question=(
            "on the diffraction-limited M3-SINGLET-REF system, does "
            "M_RAY_OPTILAND -> C_RAY_TO_WAVE -> M_WAVE_CHROMATIX reproduce the "
            "analytic Airy pattern at the sensor plane?"
        ),
        components=("C_RAY_TO_WAVE",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "prescription",
                "which canonical optical system is traced",
                domain=("M3-SINGLET-REF", "M3-REVERSE-TELEPHOTO"),
                default="M3-SINGLET-REF",
            ),
            PhysicalParameter(
                "field_angle_rad",
                "chief-ray field angle. Zero is the frozen configuration; off axis the "
                "weighted arm is 7.5x worse against the same oracle (CHE-103), which is "
                "an input to the residual investigation rather than a separate defect",
                unit="rad",
                domain=(0.0, 0.2),
                default=0.0,
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(4e-7, 8e-7), default=5.5e-7
            ),
            PhysicalParameter(
                "numerical_aperture",
                "image-space NA of the frozen configuration; the Airy oracle takes it "
                "directly",
                domain=(0.005, 0.2),
                default=0.05171631827291936,
            ),
            NumericalParameter(
                "pupil_rings",
                "hexapolar ring density. 512 rings is 787,969 traced rays, which is the "
                "count every measurement below was taken at",
                domain=(8, 1024),
                default=512,
                refines_toward=1,
            ),
            NumericalParameter(
                "grid_n",
                "reconstruction grid. The frozen 188 puts 2.44 pixels across the Airy "
                "radius and is NOT converged for radius-like metrics (CHE-103)",
                domain=(64, 2048),
                default=188,
                refines_toward=1,
            ),
            NumericalParameter(
                "pad_width", "ASM pad", domain=(0, 4096), default=566, refines_toward=1
            ),
            RepresentationParameter(
                "quadrature_weight",
                "uniform per-ray weights, or the production radial-trapezoid weight. It "
                "should not change the answer; it changes the residual by 2.4x, in the "
                "direction nobody has explained",
                domain=("uniform", "weighted"),
                default="weighted",
            ),
            ExecutionParameter("device", "cpu or cuda", domain=("cpu", "cuda"), default="cpu"),
        ),
        validity=(
            paraxial_field_angle(angle_key="field_angle_rad", max_angle_rad=math.radians(5.0)),
            ValidityPredicate(
                predicate_id="DIFFRACTION_LIMITED",
                statement=(
                    "the traced wavefront error stays under the Marechal quarter-wave "
                    "criterion, so the Airy oracle applies at all"
                ),
                basis=ValidityBasis.PARAXIAL_APPROXIMATION,
                margin=lambda p: fractional_margin(
                    float(p.get("wavefront_error_waves", 0.0)), 0.25
                ),
                blind_to=(
                    "which aberration: a quarter wave of spherical and a quarter wave "
                    "of coma are the same number here and different PSFs",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "O1, the analytic Airy pattern: paraxial, aberration-free, and sharing "
                "no code and no traced data with the coupler it judges"
            ),
            callable=_airy_first_null_um,
            reference="src/verification/psf_oracles.py",
        ),
        metrics=(
            Metric(
                name="fft_oracle_intensity_relative_l2",
                definition="central_relative_l2_intensity",
                description=(
                    "relative L2 of the reconstructed sensor-plane intensity against O1 "
                    "over the 5-Airy-radius gate disc"
                ),
                unit=None,
                blind_to=(
                    "phase entirely -- it is an intensity comparison, so a wavefront "
                    "with the right modulus and wrong curvature passes",
                    "everything outside the 5-Airy-radius disc, where a scattered halo "
                    "would live",
                ),
            ),
            Metric(
                name="o2_asm_intensity_relative_l2",
                definition="central_relative_l2_intensity",
                description=(
                    "the same comparison against O2, our own float64 ASM/RS propagator. "
                    "CHARACTERIZATION ONLY"
                ),
                unit=None,
                blind_to=(
                    "any error O2 shares with the coupler, which is the reason it cannot "
                    "gate: it is our own numerical code checking our own numerical code",
                ),
            ),
            Metric(
                name="handoff_power_ratio",
                definition="power_ratio",
                description=(
                    "power crossing the handoff plane, over power in the traced bundle"
                ),
                unit=None,
                blind_to=(
                    "where the power went: a ratio of 1.0 with the energy in the wrong "
                    "place is invisible here, which is why it is an invariant beside the "
                    "accuracy metric rather than instead of it",
                ),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="HANDOFF_ENERGY_CLOSES",
                statement=(
                    "the reconstruction conserves the traced bundle's power across the "
                    "handoff plane. A correct final image can hide an incorrect "
                    "intermediate convention, so the intermediate is checked too"
                ),
                metric="handoff_power_ratio",
                tolerance=Tolerance(
                    metric="handoff_power_ratio",
                    threshold=1e-3,
                    basis=(
                        "energy conservation across a lossless representation change, to "
                        "the coupler's quadrature accuracy. The composed energy ledger "
                        "already closes at this level (claim_ledger, C_RAY_TO_WAVE "
                        "conservation)"
                    ),
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects="a missing obliquity factor or a dropped clipped-ray population",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="fft_oracle_intensity_relative_l2",
                threshold=1.0e-3,
                basis=_FFT_ORACLE_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "an OPL sign flip, which scrambles the wavefront curvature entirely "
                    "(>>50% effect); and the 2.21e-3 the production configuration "
                    "currently measures, which is why this gate reads NOT_MET"
                ),
            ),
            Tolerance(
                metric="o2_asm_intensity_relative_l2",
                threshold=1.0e-3,
                basis=(
                    "reported at the same threshold as the O1 gate so the two numbers "
                    "are comparable, and may_gate is False because O2 is our own ASM/RS "
                    "propagator written to check this same coupler. Using our own "
                    "numerical code as the answer key for our own numerical code is "
                    "circular validation; L2-PSF-01 set a negative-control floor from an "
                    "O2 comparison once and had to retire it"
                ),
                basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
                may_gate=False,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="opl-sign-flip",
                description=(
                    "negate the declared OPL, which inverts the reconstructed wavefront "
                    "curvature"
                ),
                mutation="OPL -> -OPL before the coherent sum",
                target_metric="fft_oracle_intensity_relative_l2",
                caveat=_OPL_SIGN_FLIP_BASIS,
            ),
            NegativeControl(
                control_id="inverted-quadrature-weight",
                description=(
                    "the control that fires BACKWARDS. Adding the production weight is "
                    "required to improve agreement with O1 by at least 1.2x, and it "
                    "measures 0.42 -- the uniform configuration is CLOSER to the oracle "
                    "than the weighted one"
                ),
                mutation=(
                    "compare the weighted configuration's O1 residual against the "
                    "uniform configuration's; the ratio must exceed 1.2"
                ),
                target_metric="fft_oracle_intensity_relative_l2",
                expectation=NegativeControlExpectation.KNOWN_FIRES_BACKWARDS,
                caveat=_QUADRATURE_BASIS,
            ),
            NegativeControl(
                control_id="axis-transpose",
                description="transpose the reconstruction grid's axes",
                mutation="swap x and y on the target grid before the sum",
                target_metric="fft_oracle_intensity_relative_l2",
                expectation=NegativeControlExpectation.NOT_IMPLEMENTED,
                caveat=(
                    "declared and not run. On a rotationally symmetric on-axis Airy "
                    "pattern a transpose is the IDENTITY, so this control can only say "
                    "anything at the off-axis instance -- which is itself the finding, "
                    "and is why it is declared rather than quietly dropped"
                ),
            ),
            NegativeControl(
                control_id="launch-phase-error",
                description="omit the object-space launch-phase term",
                mutation="drop n_object * (d0 . r_launch) from the declared pupil OPL",
                target_metric="fft_oracle_intensity_relative_l2",
                expectation=NegativeControlExpectation.NOT_IMPLEMENTED,
                caveat=(
                    "declared and not run here. On axis the term is a constant and "
                    "cancels, so this control is only meaningful off axis; "
                    "B1-RAY-OFFAXIS-OPL owns it at the node level, where it can be "
                    "isolated from the propagation"
                ),
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=COMPOSED_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MET,
            metric="fft_oracle_intensity_relative_l2",
            observed=2.2072391812867093e-3,
            evidence=(
                "benchmarks/physics/L2-PSF-01/tolerances.yaml",
                "benchmarks/probes/records/m3_quadrature_weight.json",
                "benchmarks/reports/2026-08/ray_to_wave_slice_exit.md",
            ),
            note=(
                "Carried forward unwidened. The 1.0e-3 gate has been frozen since M3.2 "
                "and re-affirmed through M3.8 and M3.9R; the production configuration "
                "measures 2.21e-3 at 787,969 rays. CHE-117 attributes the residual. "
                "This gate must NOT be closed against another Optiland PSF route: "
                "FFTPSF and HuygensPSF share one Wavefront/OPD front end and are one "
                "oracle, not two (PB7/CHE-58 finding F2)."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.HISTORICAL_REGRESSION,
        sampler_absent_note=(
            "the frozen M3-SINGLET-REF configuration at 787,969 rays is the point every "
            "number in the residual investigation was measured at. Sampling around it is "
            "valuable once the residual is attributed; moving it now would make the "
            "investigation's own evidence incomparable."
        ),
        evidence=(
            "benchmarks/physics/L2-PSF-01/tolerances.yaml",
            "benchmarks/physics/L2-PSF-01/README.md",
            "benchmarks/probes/records/m3_quadrature_weight.json",
            "benchmarks/reports/2026-08/slice_cleanup_disposition.md",
        ),
        notes=(
            "The only composed chain in the repository with a genuinely independent "
            "analytic decider. That is what makes an unmet gate here worth carrying: it "
            "is a known-wrong answer with something real disagreeing with it, which is a "
            "better state than a passing comparison against ourselves."
        ),
    )
)


# ---------------------------------------------------------------------------
# B3-DEMO2
# ---------------------------------------------------------------------------

B3_DEMO2 = register(
    BenchmarkFamily(
        family_id="B3-DEMO2",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        question=(
            "does the composed ray-wave reproduction of the paper's demo2 hologram "
            "match the published figure at the paper's own Table S2 ray budget?"
        ),
        components=("C_RAY_TO_WAVE", "C_PATCH_WFT"),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "phase_profile",
                "the smile phase profile the paper specifies; the input data, not a knob",
                domain=("demo2_smile",),
                default="demo2_smile",
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(4e-7, 8e-7), default=5.32e-7
            ),
            NumericalParameter(
                "ray_count",
                "traced rays. 1.6e8 is the paper's Table S2 budget for the patch route",
                domain=(int(1e5), int(1e10)),
                default=int(1.6e8),
                refines_toward=1,
            ),
            RepresentationParameter(
                "reconstruction_route",
                "RAMP_SUM or the k-space splat. They agree to 7.1e-13 on this system, "
                "which is what makes the choice a representation rather than a physics "
                "decision HERE and not on demo3",
                domain=("ramp_sum", "kspace_splat"),
                default="ramp_sum",
            ),
            ExecutionParameter("device", "cpu or cuda", domain=("cpu", "cuda"), default="cuda"),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="AT_OR_ABOVE_PAPER_BUDGET",
                statement=(
                    "the ray budget is at least the paper's own Table S2 choice for "
                    "this route. Below it the comparison is against a published figure "
                    "the run has not converged to, which measures the budget rather "
                    "than the physics"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                margin=lambda p: (float(p["ray_count"]) - 1.6e8) / 1.6e8,
                blind_to=(
                    "whether the budget is enough for a DIFFERENT route. 1.6e8 is the "
                    "patch route's Table S2 number; the full-aperture route's is 1.1e6",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.INDEPENDENT_IMPLEMENTATION,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the published figure from the paper's own implementation. External to "
                "this repository in the strongest sense available: a different group's "
                "code, run before this project existed"
            ),
            callable=None,
            reference="ACS Photonics 2026 demo2, SI Table S2",
        ),
        metrics=(
            Metric(
                name="demo2_ncc",
                definition="ncc",
                description="normalized cross-correlation against the published intensity",
                unit=None,
                blind_to=(
                    "an overall scale factor, by construction -- NCC normalizes it away, "
                    "which is why the relative L2 is reported beside it",
                    "a global translation of a few pixels, which correlates well and is "
                    "not the same field",
                ),
            ),
            Metric(
                name="demo2_relative_l2",
                definition="relative_l2_intensity",
                description="relative L2 against the published intensity",
                unit=None,
                blind_to=("phase; this is an intensity comparison",),
            ),
            Metric(
                name="patch_handoff_power_ratio",
                definition="power_ratio",
                description=(
                    "power leaving the patch decomposition over power entering it, at "
                    "the C_PATCH_WFT -> trace boundary"
                ),
                unit=None,
                blind_to=(
                    "where the power went, and any loss that is exactly compensated by "
                    "a gain elsewhere in the same stage",
                ),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="PATCH_ENERGY_CLOSES",
                statement=(
                    "the patch decomposition conserves power into the trace. Checked "
                    "beside the final comparison because a correct hologram can hide a "
                    "compensating pair of intermediate errors"
                ),
                metric="patch_handoff_power_ratio",
                tolerance=Tolerance(
                    metric="patch_handoff_power_ratio",
                    threshold=1e-3,
                    basis=(
                        "the full-aperture patch agrees with an independent float64 ASM "
                        "at 7.1e-13, so its energy bookkeeping closes far below this. "
                        "1e-3 is loose against that and tight against the 1.7% the "
                        "k-space route loses on demo3 at 8x oversampling"
                    ),
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects=(
                        "a route that drops power at the patch boundary, as the k-space "
                        "route does on demo3"
                    ),
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="demo2_ncc",
                threshold=0.999,
                basis=(
                    "the measured agreement at the paper's own budget is NCC 0.999418 at "
                    "1.6e8 rays on the patch route (benchmarks/reports/2026-08/"
                    "kspace_ray_to_wave.md). 0.999 sits just below that measurement and "
                    "well above the 0.9987 the full-aperture route reaches at 1.1e6 "
                    "rays, so it separates the converged reproduction from an "
                    "under-sampled one"
                ),
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects=(
                    "an under-budgeted run, and any convention error large enough to "
                    "decorrelate the hologram"
                ),
            ),
            Tolerance(
                metric="demo2_relative_l2",
                threshold=5e-2,
                basis=(
                    "measured 2.8562e-2 at the Table S2 budget. 5e-2 admits the run-to-"
                    "run spread of a stochastic estimator at that budget without "
                    "admitting the 8.87e-2 the full-aperture route reaches at its own"
                ),
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects="a route or budget that has not converged to the published figure",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="phase-profile-conjugate",
                description="conjugate the input phase profile",
                mutation="phi -> -phi on the smile profile",
                target_metric="demo2_ncc",
            ),
            NegativeControl(
                control_id="tenth-budget",
                description=(
                    "run at a tenth of the Table S2 budget and require the gate to fail. "
                    "A tolerance that a deliberately under-sampled run still passes is "
                    "measuring something other than convergence"
                ),
                mutation="ray_count / 10",
                target_metric="demo2_ncc",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
            VerificationStatus.BLOCKED,
        ),
        execution_policy=ExecutionPolicy(
            devices=frozenset({DeviceKind.CUDA}),
            dtypes=frozenset({DType.COMPLEX64}),
            namespaces=frozenset({ArrayNamespace.JAX}),
            max_wall_seconds=600.0,
            max_peak_memory_gib=40.0,
            notes=(
                "95 s at the Table S2 budget on one GPU. CUDA-only in the execution "
                "policy because the CPU path exists and is not the configuration any "
                "number here was measured on."
            ),
        ),
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MEASURED_OFF_GATE,
            metric="demo2_ncc",
            observed=0.999418,
            evidence=(
                "benchmarks/reports/2026-08/kspace_ray_to_wave.md",
                "benchmarks/probes/records/ray_wave/demo2_paper_jax.json",
            ),
            note=(
                "NCC 0.999418 and relative L2 2.8562e-2 at the paper's own 1.6e8-ray "
                "budget, on record. MEASURED_OFF_GATE rather than MET because nothing in "
                "the required gate re-runs it: it is a 95-second GPU job and belongs in "
                "the extended collection, not the default suite."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the reference is a specific published figure at a specific budget. "
            "Generating parameters destroys the external independence that is the entire "
            "value of this case -- there is no published figure for a drawn point."
        ),
        evidence=(
            "benchmarks/reports/2026-08/kspace_ray_to_wave.md",
            "benchmarks/probes/records/ray_wave/demo2_paper_jax.json",
            "benchmarks/probes/ray_wave/demo2_hologram.py",
        ),
        notes=(
            "The one composed reproduction graded against an EXTERNAL published "
            "reference rather than against ourselves, which makes it the regression "
            "anchor for the whole ray-wave path."
        ),
    )
)


# ---------------------------------------------------------------------------
# B3-DUALROUTE -- gated on the invariant, not on the route agreement
# ---------------------------------------------------------------------------


def _unit_power(_params: Mapping[str, Any]) -> float:
    """Conservation: total power out over total power in, which must be 1.

    The reference needs no external source, which is the point. Two routes
    through our own code cannot decide which of them is right; both of them
    conserving energy is a statement neither can fake for the other.
    """
    return 1.0


B3_DUALROUTE = register(
    BenchmarkFamily(
        family_id="B3-DUALROUTE",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        question=(
            "on the Cooke triplet, do the ray-only and ray->coupler->wave PSF routes "
            "each conserve the traced bundle's power end to end?"
        ),
        components=("C_RAY_TO_WAVE", "M_RAY_OPTILAND"),
        claim_kind=ClaimKind.CONSERVATION,
        parameters=(
            PhysicalParameter(
                "field_angle_deg",
                "field angle. 0 and 20 degrees are the two the PB7 study measured, and "
                "they behave differently: on axis all three routes agree, off axis one "
                "is an attributed outlier",
                unit="deg",
                domain=(0.0, 20.0),
                default=0.0,
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(4e-7, 8e-7), default=5.5e-7
            ),
            NumericalParameter(
                "pupil_rings", "hexapolar rings", domain=(8, 256), default=64, refines_toward=1
            ),
            RepresentationParameter(
                "route",
                "which PSF route is exercised",
                domain=("optiland_fft", "optiland_huygens", "ray_to_wave"),
                default="ray_to_wave",
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="PUPIL_ANISOTROPY",
                statement=(
                    "the image-space pupil's direction-space extent is close enough to "
                    "isotropic that a single scalar working F/# describes it"
                ),
                basis=ValidityBasis.PARAXIAL_APPROXIMATION,
                margin=lambda p: fractional_margin(
                    abs(float(p.get("pupil_anisotropy", 1.0)) - 1.0), 0.05
                ),
                blind_to=(
                    "everything except the F/# ratio. It is a statement about FFTPSF's "
                    "pixel-scale assumption, not about the optical system",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.CONSERVATION_LAW,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "power in equals power out, along each route separately. An intermediate "
                "invariant, which M4.1 lists as an admissible decider and which needs no "
                "external reference -- so it survives the fact that every available PSF "
                "route here is our own code"
            ),
            callable=_unit_power,
            reference="energy conservation through a lossless representation change",
        ),
        metrics=(
            Metric(
                name="route_power_ratio",
                definition="power_ratio",
                description="power in the route's output PSF over power in the traced bundle",
                unit=None,
                blind_to=(
                    "the spatial distribution entirely. A route that conserved energy and "
                    "put it in the wrong place passes, which is exactly why the route "
                    "AGREEMENT is a separate family and not a metric here",
                ),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="CHIEF_RAY_REGISTRATION",
                statement=(
                    "the PSF peak coincides with the traced chief ray. Independent of "
                    "any other route, and it is what caught FFTPSF's off-axis "
                    "mis-scaling: a 3.39-pixel peak separation against 0.03 for the "
                    "other pair"
                ),
                metric="route_power_ratio",
                tolerance=Tolerance(
                    metric="route_power_ratio",
                    threshold=0.5,
                    basis=(
                        "half a pixel, expressed as a fraction of the sample pitch. The "
                        "measured on-axis separation is below floating-point resolution "
                        "and the measured off-axis FFTPSF separation is 3.39 pixels "
                        "(PB7/CHE-58), so this separates them by nearly seven times"
                    ),
                    basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                    may_gate=True,
                    rejects="a route whose pixel scale is set from the wrong F/#",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="route_power_ratio",
                threshold=1e-2,
                basis=(
                    "|1 - ratio|. Energy conservation through a lossless representation "
                    "change closes to the coupler's quadrature accuracy; 1e-2 is loose "
                    "against that and tight against a dropped vignetted-ray population, "
                    "which on this system at 20 degrees is tens of percent"
                ),
                basis_kind=ToleranceBasis.CONSERVATION_LAW,
                may_gate=True,
                rejects="a route that silently discards clipped rays rather than counting them",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="drop-clipped-rays",
                description="discard vignetted rays instead of accounting for their power",
                mutation="filter the bundle to survivors before summing power",
                target_metric="route_power_ratio",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNSUPPORTED,
        ),
        execution_policy=COMPOSED_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MEASURED,
            note=(
                "PB7/CHE-58 measured the route AGREEMENT and did not measure per-route "
                "energy closure, so this gate has no number yet. That is the gap this "
                "family exists to close: the agreement study could not decide anything, "
                "and the invariant can."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the Cooke triplet at 0 and 20 degrees is the configuration the route study "
            "was run on, and the off-axis point is the one where the routes diverge. A "
            "drawn field angle would mostly land where they agree."
        ),
        evidence=(
            "benchmarks/reports/2026-08/cooke_triplet_psf_routes.md",
            "benchmarks/probes/cooke_triplet_psf_routes.py",
        ),
        notes=(
            "AUTHORING NOTE. M4.1 asked for this as a B3 family comparing the routes "
            "against each other. It could not be: every route available is our own code, "
            "the two Optiland ones share a Wavefront/OPD front end, and the schema "
            "refuses a CROSS_ROUTE oracle outside B4. What can decide something here is "
            "an intermediate invariant, so this family gates on energy closure and "
            "chief-ray registration, and B4-DUALROUTE-AGREEMENT carries the route "
            "comparison where it cannot gate."
        ),
    )
)
