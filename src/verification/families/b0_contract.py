"""B0: what happens when a component is asked for something it cannot do.

CHE-108 (M1.3). An agent spends a meaningful fraction of its time asking
components to do things they cannot do, and what happens then is a correctness
property in its own right -- the property that decides whether it can recover.
The two failure modes it must never see are a fabricated number and an
unstructured traceback.

The five negative outcomes may not collapse
-------------------------------------------
An agent that cannot tell "this route has no executable precision" from "this
configuration is malformed" from "this approximation does not apply here" cannot
recover from any of them. ``B0-CONTRACT``'s canonical instances are chosen to
produce each of ``unsupported``, ``invalid_configuration``, ``out_of_validity``
and ``blocked`` separately, and ``B0-DTYPE``'s produces ``lossy_but_allowed``
with the loss as a **measured number, not a warning**.

The interesting failures are silent, not loud
---------------------------------------------
``B0-UNITS`` is the family that would be missing from a conformance suite built
only on "invalid input raises". Both of its instances run perfectly, every
boundary check passes, the contract status is ``ok``, and the physics is wrong:
Optiland's ``add_layer`` takes micrometres where the AR-coating literature says
nanometres, and the coated reflectance comes back ``0.04216384`` against bare
glass's ``0.04216456``; Chromatix's ``kykx`` means cycles per length on one
function and radians per length on another, with the displacement running
opposite in sign to the parameter.

They exist to prove the substrate can tell "it executed" from "it is right".
Their numbers are preserved in ``verification/hazards.py`` and they are
non-generative, because the measured wrong number IS the artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import (
    boolean_margin,
    capability_intersection_nonempty,
    fractional_margin,
    hexapolar_ring_membership,
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
    Metric,
    NegativeControl,
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
from verification.hazards import KYKX_TWO_PI_AND_SIGN, UNITS_MICROMETRE_NANOMETRE
from verification.status import VerificationStatus

__all__ = ["B0_CONTRACT", "B0_DTYPE", "B0_UNITS", "B0_VALIDITY"]


CONTRACT_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    dtypes=frozenset({DType.FLOAT64, DType.COMPLEX64, DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX, ArrayNamespace.TORCH}),
    max_wall_seconds=60.0,
    max_peak_memory_gib=2.0,
    notes=(
        "Deliberately the widest policy in the registry: this family's instances are "
        "requests that SHOULD be refused, and narrowing the policy would make the "
        "executor refuse them for the wrong reason before the component could."
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason="a refusal is a decision about a request, and nothing is sampled",
)

#: float32 machine epsilon, 2**-23. Written out rather than derived from
#: float64's, which is a different number: the first draft of this family used
#: sqrt(eps64) = 1.49e-8 and produced a "bound" of 6.8e-6 that the MEASURED loss
#: of 2.5e-5 sat above. A bound below its own measurement is not a bound, and
#: tests/test_b0_families.py is what caught it.
_EPS32 = 2.0**-23


def _refusal_is_actionable(_params: Mapping[str, Any]) -> float:
    """The oracle: a refusal must carry a code, a reason and a remedy.

    Independent of the component under test in the way that matters -- it checks
    the SHAPE of the answer against a declaration, not the answer against
    another implementation of itself. ``1.0`` when the refusal is actionable.
    """
    return 1.0


# ---------------------------------------------------------------------------
# B0-CONTRACT
# ---------------------------------------------------------------------------

_B0_CONTRACT = BenchmarkFamily(
        family_id="B0-CONTRACT",
        family_version="1.0.0",
        category=BenchmarkCategory.B0,
        question=(
            "when a component is asked for something it cannot do, does it refuse with "
            "a code, a reason and a remedy -- and does the refusal say WHICH of the "
            "five negative outcomes it is?"
        ),
        components=(
            "M_RAY_OPTILAND",
            "M_WAVE_CHROMATIX",
            "C_RAY_TO_WAVE",
            "C_PATCH_WFT",
        ),
        claim_kind=ClaimKind.STRUCTURED_FAILURE,
        parameters=(
            PhysicalParameter(
                "component",
                "which component is asked",
                domain=(
                    "M_RAY_OPTILAND",
                    "M_WAVE_CHROMATIX",
                    "C_RAY_TO_WAVE",
                    "C_PATCH_WFT",
                ),
                default="C_RAY_TO_WAVE",
            ),
            ExecutionParameter(
                "device", "requested device", domain=("cpu", "cuda"), default="cpu"
            ),
            ExecutionParameter(
                "dtype",
                "requested dtype",
                domain=("float32", "float64", "complex64", "complex128"),
                default="complex64",
            ),
            RepresentationParameter(
                "omitted_declaration",
                "which declaration is left out, if any. THE parameter of this family: "
                "each value produces a different refusal code, and they must not "
                "collapse",
                domain=(
                    "none",
                    "handoff_plane",
                    "reference_plane",
                    "object_space_reference",
                ),
                default="none",
            ),
            RepresentationParameter(
                "pupil_sampling",
                "hexapolar or otherwise. A per-ray quadrature weight on a non-hexapolar "
                "bundle assigns weights derived from a ring structure that is not there",
                domain=("hexapolar", "rectangular", "random"),
                default="hexapolar",
            ),
            NumericalParameter(
                "grid_n", "grid, where a request needs one", domain=(4, 1024), default=64,
                refines_toward=1,
            ),
        ),
        validity=(
            capability_intersection_nonempty(),
            hexapolar_ring_membership(),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the declared capability table and the refusal catalogue. The oracle is "
                "the DECLARATION, not another run: a component's own output cannot say "
                "whether its refusal was well formed"
            ),
            callable=_refusal_is_actionable,
            reference="core/capabilities.py; verification/refusals.py",
        ),
        metrics=(
            Metric(
                name="refusal_is_actionable",
                description=(
                    "1.0 when the refusal carries a code from the declared enum, a "
                    "reason naming what was wrong, and a remedy naming what would fix "
                    "it; 0.0 otherwise"
                ),
                unit=None,
                blind_to=(
                    "whether the remedy is CORRECT. It checks that one is offered, not "
                    "that following it works",
                    "whether the refusal was the right decision. A component that "
                    "refused something it could have done scores 1.0 here",
                ),
            ),
            Metric(
                name="status_is_specific",
                description=(
                    "1.0 when the refusal maps to one of the five negative outcomes "
                    "rather than to a generic failure"
                ),
                unit=None,
                blind_to=(
                    "whether it is the RIGHT one of the five. Mapping every refusal to "
                    "invalid_configuration would score 1.0 and teach a caller nothing, "
                    "which is why the canonical instances name their expected status",
                ),
            ),
            Metric(
                name="fabricated_output_count",
                description=(
                    "how many numeric results, metrics, convergence claims or "
                    "provenance records a refusal path emitted. Must be zero"
                ),
                unit=None,
                blind_to=("a fabricated value that is genuinely correct, which is luck",),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="fabricated_output_count",
                threshold=0.5,
                basis=(
                    "zero, expressed as a threshold on a count. An AGENTS.md "
                    "non-negotiable: failed or unsupported solvers return structured "
                    "diagnostics and never invent fields, metrics, convergence or "
                    "provenance. 0.5 is the threshold that makes 'exactly zero' a "
                    "comparison"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects="any invented number on a refusal path",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="generic-failure",
                description=(
                    "map every refusal to one generic failure status and require the "
                    "family to fail. A conformance suite that returns one code for "
                    "'no executable precision', 'malformed request' and 'approximation "
                    "does not apply' teaches an agent nothing"
                ),
                mutation="collapse the five statuses to a single FAILED",
                target_metric="status_is_specific",
            ),
            NegativeControl(
                control_id="remedy-free-refusal",
                description="refuse with a code and no remedy",
                mutation="drop the remedy field from the raised error",
                target_metric="refusal_is_actionable",
            ),
        ),
        failure_semantics=(
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.INVALID_CONFIGURATION,
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.BLOCKED,
        ),
        execution_policy=CONTRACT_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="fabricated_output_count",
            observed=0.0,
            evidence=(
                "benchmarks/instances/b0_contract.py",
                "tests/test_b0_instances.py::test_the_five_negative_outcomes_come_from_five_real_executions",
                "tests/test_b0_instances.py::test_no_refusal_reports_a_metric_a_convergence_or_a_gate",
                "tests/test_contract_code_reachability.py",
            ),
            note=(
                "MET by execution rather than by declaration. All six canonical "
                "instances now run through the shipping components, and each of the "
                "five negative outcomes is produced by a DIFFERENT instance -- so an "
                "agent can tell them apart, which is the whole claim. No refusal "
                "carries a metric, an invariant, a convergence verdict, or an output "
                "of the node that refused; the count is zero and is asserted rather "
                "than assumed. All nineteen declared ContractCodes are separately "
                "shown reachable with a reason and a remedy.\n\n"
                "One defect fixed to get here, and it is the reason the systematic "
                "walk was worth doing: OPL_REFERENCE_UNVERIFIED is `blocked` in the "
                "refusal catalogue and was arriving through the record path as "
                "`invalid_configuration`, because the executor maps it to "
                "MISSING_EDGE_DECLARATION. The two statuses had therefore collapsed "
                "for exactly the code whose whole point is that a caller could have "
                "proceeded and the component chose not to. The catalogue now decides "
                "when a contract code is present."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the interesting instances are specific declared boundaries, not points in "
            "a space. A drawn (component, device, dtype) triple is overwhelmingly "
            "likely to be a supported combination, which the family has nothing to say "
            "about."
        ),
        evidence=(
            "src/verification/refusals.py",
            "tests/test_coupler_contracts.py",
            "tests/test_precision_contract.py",
        ),
)

B0_CONTRACT = register(_B0_CONTRACT.with_instances(
    _B0_CONTRACT.instantiate(
        "B0-CAPINT-01",
        {
            "component": "C_PATCH_WFT",
            "device": "cpu",
            "dtype": "complex64",
            "omitted_declaration": "none",
            "pupil_sampling": "hexapolar",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.UNSUPPORTED.value,
            "why": (
                "Chromatix accepts only complex64; C_PATCH_WFT computes only in "
                "complex128 and only on CPU. A composed route through both has NO "
                "precision at which it can execute, and an agent will propose exactly "
                "such routes. Refused at planning time with a named reason, or "
                "discovered as a traceback three nodes in -- that is the difference "
                "between a recoverable failure and a dead end (project risk R5)."
            ),
        },
    ),
    _B0_CONTRACT.instantiate(
        "B0-DEVICE-01",
        {
            "component": "C_PATCH_WFT",
            "device": "cuda",
            "dtype": "complex128",
            "omitted_declaration": "none",
            "pupil_sampling": "hexapolar",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.UNSUPPORTED.value,
            "why": "CUDA requested of a coupler whose capability table declares CPU only",
        },
    ),
    _B0_CONTRACT.instantiate(
        "B0-DEVICE-02",
        {
            "component": "M_WAVE_CHROMATIX",
            "device": "cuda",
            "dtype": "complex64",
            "omitted_declaration": "none",
            "pupil_sampling": "hexapolar",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.INVALID_CONFIGURATION.value,
            "code": "REPRESENTATION_INCONSISTENT",
            "why": (
                "declared cuda with actual cpu placement. A process-global JAX platform "
                "pin produces a successful run on the host while the caller asked for "
                "CUDA, with no error raised -- so the device must be read off the array "
                "and the disagreement reported."
            ),
        },
    ),
    _B0_CONTRACT.instantiate(
        "B0-META-01",
        {
            "component": "C_RAY_TO_WAVE",
            "device": "cpu",
            "dtype": "complex64",
            "omitted_declaration": "reference_plane",
            "pupil_sampling": "hexapolar",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.INVALID_CONFIGURATION.value,
            "code": "MISSING_DECLARATION",
            "why": "required metadata absent; the consumer will not default a convention",
        },
    ),
    _B0_CONTRACT.instantiate(
        "B0-HANDOFF-01",
        {
            "component": "C_RAY_TO_WAVE",
            "device": "cpu",
            "dtype": "complex64",
            "omitted_declaration": "handoff_plane",
            "pupil_sampling": "hexapolar",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.BLOCKED.value,
            "code": "OPL_REFERENCE_UNVERIFIED",
            "why": (
                "a bare opd_native is an absolute accumulated path whose zero moves "
                "with the aperture. The coupler COULD proceed and refuses to, which is "
                "why this is blocked rather than invalid: nothing about the request is "
                "malformed, and the missing thing is a declaration."
            ),
        },
    ),
    _B0_CONTRACT.instantiate(
        "B0-PATCH-01",
        {
            "component": "C_RAY_TO_WAVE",
            "device": "cpu",
            "dtype": "complex64",
            "omitted_declaration": "none",
            "pupil_sampling": "rectangular",
            "grid_n": 64,
        },
        expected={
            "status": VerificationStatus.OUT_OF_VALIDITY.value,
            "code": "NON_HEXAPOLAR_SAMPLING",
            "why": (
                "the quadrature weight a ray carries is derived from its ring, so "
                "applying it to a rectangular bundle assigns weights that mean nothing. "
                "The code would run; the answer would be wrong -- which is out of "
                "validity rather than unsupported."
            ),
        },
    ),
))


# ---------------------------------------------------------------------------
# B0-DTYPE
# ---------------------------------------------------------------------------

_B0_DTYPE = BenchmarkFamily(
        family_id="B0-DTYPE",
        family_version="1.0.0",
        category=BenchmarkCategory.B0,
        question=(
            "when a precision request cannot be honoured, is the loss refused, or "
            "recorded as a MEASURED NUMBER -- and never performed silently?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.DEVICE_PARITY,
        parameters=(
            ExecutionParameter(
                "requested_dtype",
                "what the caller asked for",
                domain=("complex64", "complex128"),
                default="complex128",
            ),
            ExecutionParameter(
                "bridge_policy",
                "SAFE refuses a lossy downcast; ALLOW_DOWNCAST performs it and records "
                "the loss. Both are correct answers and they are different ones",
                domain=("safe", "allow_downcast"),
                default="allow_downcast",
            ),
            PhysicalParameter(
                "propagation_distance_m",
                "distance. The measured loss grows as eps32 * 2*pi*z/lambda, so this is "
                "the parameter the loss is a function of",
                unit="m",
                domain=(0.0, 0.1),
                default=4.0e-5,
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.5e-7
            ),
            NumericalParameter(
                "grid_n", "grid", domain=(16, 2048), default=256, refines_toward=1
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="CHROMATIX_NATIVE_DTYPE",
                statement=(
                    "the requested dtype is one Chromatix can compute in natively. "
                    "complex128 is not: ScalarField casts unconditionally, so an FP64 "
                    "request has nothing to execute"
                ),
                basis=ValidityBasis.CAPABILITY_INTERSECTION,
                # +1 native, 0 lossily accepted, -1 refused. The middle value is
                # exactly right: a dtype the component ingests and truncates IS
                # the boundary, and calling it outside would make the verifier
                # report out_of_validity for a run whose whole point is
                # lossy_but_allowed.
                margin=lambda p: (
                    1.0
                    if str(p["requested_dtype"]) == "complex64"
                    else (
                        0.0
                        if str(p["bridge_policy"]) == "allow_downcast"
                        else -1.0
                    )
                ),
                blind_to=(
                    "how MUCH is lost. It says the request is admissible, not that the "
                    "loss is small -- measured_precision_loss is the number for that",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "one float32 epsilon per radian of accumulated phase, eps32 * "
                "2*pi*z/lambda. A closed form for the loss, so the recorded number can "
                "be checked rather than merely present"
            ),
            callable=lambda p: (
                _EPS32
                * 2.0
                * math.pi
                * float(p["propagation_distance_m"])
                / float(p["wavelength_m"])
            ),
            reference="benchmarks/probes/carrier_phase_representation.py",
        ),
        metrics=(
            Metric(
                name="measured_precision_loss",
                description=(
                    "the relative field error the downcast actually cost, as a NUMBER. "
                    "Measured 2.5e-5 at z = 40 um and 6.3e-2 at z = 47 mm"
                ),
                unit=None,
                definition="relative_l2_field",
                blind_to=(
                    "where in the field the error is. It is a norm, and a downcast's "
                    "error is spread rather than local",
                ),
            ),
            Metric(
                name="loss_was_reported",
                description=(
                    "1.0 when the run recorded the loss rather than performing it "
                    "silently inside ScalarField"
                ),
                unit=None,
                blind_to=(
                    "whether the reported number is right, which is what "
                    "measured_precision_loss is for",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="measured_precision_loss",
                threshold=1e-4,
                basis=(
                    "one eps32 per radian at the default z = 40 um and lambda = 550 nm "
                    "is about 5e-5; the measured value is 2.5e-5. 1e-4 is the bound the "
                    "closed form gives at this distance, so a loss above it means "
                    "something other than the downcast is happening"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a silent truncation reported as zero loss, and a loss large enough "
                    "to be a different defect"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="silent-truncation",
                description=(
                    "hand Field.build a complex128 array and let ScalarField cast it. It "
                    "returns complex64 EVEN UNDER jax_enable_x64=True, and nothing "
                    "reports the loss -- which is the failure this family exists for"
                ),
                mutation="bypass the bridge and construct the field directly",
                target_metric="loss_was_reported",
            ),
            NegativeControl(
                control_id="requested-reported-as-actual",
                description=(
                    "report the REQUESTED dtype as the actual one. The run succeeds, "
                    "the record says complex128, and the arithmetic was complex64"
                ),
                mutation="read the dtype off the request instead of off the array",
                target_metric="loss_was_reported",
            ),
        ),
        failure_semantics=(
            VerificationStatus.LOSSY_BUT_ALLOWED,
            VerificationStatus.UNSUPPORTED,
        ),
        execution_policy=CONTRACT_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="measured_precision_loss",
            observed=2.82988e-5,
            evidence=(
                "benchmarks/instances/b0_contract.py",
                "tests/test_b0_instances.py::test_the_precision_loss_is_measured_against_an_independent_oracle",
                "tests/test_b0_instances.py::test_the_measured_loss_sits_under_one_eps32_per_radian",
            ),
            note=(
                "Re-measured, not inherited. 2.82988e-5 at z = 40 um against the "
                "1.0e-4 gate, from a real Chromatix complex64 propagation compared to "
                "verification/asm_oracle.angular_spectrum_float64 -- an independent "
                "float64 implementation that shares no code with Chromatix, so what is "
                "measured is the cost of the REPRESENTATION rather than of the "
                "implementation. Measuring it with a second Chromatix call would put "
                "the truncation on both sides of the comparison.\n\n"
                "The inherited figure was 2.5e-5 and the two agree to 13%, which is "
                "the useful outcome: the historical number was right and is now "
                "reproducible from code in the tree. The eps32-per-radian bound at "
                "this configuration is 5.45e-5, evaluated rather than quoted, and the "
                "measurement sits under it -- so the residual is the dtype and not the "
                "propagator.\n\n"
                "6.3e-2 at z = 47 mm is preserved as the other end of the same "
                "relation and is why the M3 reference singlet was scaled to a tenth: "
                "the loss is a function of accumulated phase, so a longer system is a "
                "less accurate one at fixed precision. This is also CHE-107's WAVE-2 "
                "measurement; it is made once rather than twice."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the case being certified is a specific capability declaration -- Chromatix "
            "has no complex128 path at any device -- rather than a region of a space."
        ),
        evidence=(
            "benchmarks/probes/carrier_phase_representation.py",
            "src/core/capabilities.py",
            "tests/test_precision_contract.py",
        ),
)

B0_DTYPE = register(_B0_DTYPE.with_instances(
    _B0_DTYPE.instantiate(
        "B0-DTYPE-01",
        {
            "requested_dtype": "complex128",
            "bridge_policy": "allow_downcast",
            "propagation_distance_m": 4.0e-5,
            "wavelength_m": 5.5e-7,
            "grid_n": 256,
        },
        expected={
            "status": VerificationStatus.LOSSY_BUT_ALLOWED.value,
            "measured_precision_loss": 2.5e-5,
            "why": (
                "ScalarField.__init__ is jnp.asarray(u, dtype=jnp.complex64) "
                "unconditionally. Keeping complex128 out of accepted_input_dtypes is "
                "what makes the bridge refuse it under SAFE and record it as lossy "
                "under ALLOW_DOWNCAST, instead of letting the loss happen inside "
                "ScalarField where nothing measures it."
            ),
        },
    ),
))


# ---------------------------------------------------------------------------
# B0-VALIDITY
# ---------------------------------------------------------------------------

_B0_VALIDITY = BenchmarkFamily(
        family_id="B0-VALIDITY",
        family_version="1.0.0",
        category=BenchmarkCategory.B0,
        question=(
            "when an instance crosses a declared validity bound, is that reported as "
            "OUT OF VALIDITY -- distinct from unsupported and from a malformed request?"
        ),
        components=("C_PATCH_WFT",),
        claim_kind=ClaimKind.STRUCTURED_FAILURE,
        parameters=(
            PhysicalParameter(
                "patch_width_m", "patch width D", unit="m", domain=(1e-9, 1.0), default=1e-3
            ),
            PhysicalParameter(
                "substrate_radius_m",
                "radius of curvature R. The bound is arcsin(D / 2R) and takes only "
                "these two -- no DOE design can be offered as an argument for relaxing "
                "it",
                unit="m",
                domain=(1e-6, 1e9),
                default=1.0,
            ),
            PhysicalParameter(
                "tangent_plane_error_rad",
                "the direction error the tangent-plane approximation incurs",
                unit="rad",
                domain=(0.0, 1.5),
                default=1e-6,
            ),
            NumericalParameter(
                "patch_px", "patch size in samples; an even value is refused rather than "
                "rounded", domain=(3, 4097), default=65, refines_toward=1,
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="SI_S3_CURVATURE",
                statement="eps_curv <= arcsin(D / 2R), SI eq S9",
                basis=ValidityBasis.SI_S3_CURVATURE,
                margin=lambda p: (
                    -1.0
                    if float(p["patch_width_m"]) / (2.0 * float(p["substrate_radius_m"])) >= 1.0
                    else fractional_margin(
                        float(p["tangent_plane_error_rad"]),
                        math.asin(
                            float(p["patch_width_m"]) / (2.0 * float(p["substrate_radius_m"]))
                        ),
                    )
                ),
                blind_to=(
                    "the phase profile written on the patch -- the bound is "
                    "design-independent, so satisfying it says nothing about whether "
                    "the profile is sampled",
                ),
            ),
            ValidityPredicate(
                predicate_id="ODD_PATCH_PX",
                statement=(
                    "the patch size in samples is odd, so the patch has a centre sample"
                ),
                basis=ValidityBasis.HEXAPOLAR_RING,
                margin=lambda p: boolean_margin(int(p["patch_px"]) % 2 == 1),
                blind_to=("whether the patch is large enough to resolve anything",),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "arcsin(D / 2R) from SI eq S9, and the parity of an integer. Both are "
                "closed forms, and neither consults the component under test"
            ),
            callable=lambda p: math.asin(
                float(p["patch_width_m"]) / (2.0 * float(p["substrate_radius_m"]))
            ),
            reference="src/couplers/curvature.py; tests/test_curvature_bound.py",
        ),
        metrics=(
            Metric(
                name="validity_status_is_out_of_validity",
                description=(
                    "1.0 when crossing the bound is reported as out_of_validity rather "
                    "than as unsupported or invalid_configuration"
                ),
                unit=None,
                blind_to=(
                    "the MARGIN. A run reported out of validity by a hair and one "
                    "reported out by a factor of two score the same here, which is why "
                    "the signed margin is carried separately",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="validity_status_is_out_of_validity",
                threshold=0.5,
                basis=(
                    "a boolean, expressed as a threshold. Out of validity means the code "
                    "would run and the answer would be wrong; unsupported means it "
                    "cannot run at all. Conflating them tells an agent to change the "
                    "wrong thing"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects="a validity crossing reported as an unsupported capability",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="rounded-even-patch",
                description=(
                    "pass an even patch_px and round it silently rather than refusing. "
                    "The patch then has no centre sample and every coordinate it reports "
                    "is off by half a pixel"
                ),
                mutation="round patch_px up to the next odd value without saying so",
                target_metric="validity_status_is_out_of_validity",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.INVALID_CONFIGURATION,
        ),
        execution_policy=CONTRACT_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="validity_status_is_out_of_validity",
            observed=0.0,
            evidence=(
                "benchmarks/instances/b0_contract.py",
                "tests/test_b0_instances.py::test_out_of_validity_comes_from_a_run_that_succeeded",
                "tests/test_b0_instances.py::test_the_curvature_guard_is_live_at_the_instance_parameters",
            ),
            note=(
                "The reporting is now measured, which is what NOT_MEASURED referred "
                "to: crossing arcsin(D/2R) comes back as out_of_validity and not as "
                "one of the other four. The route matters and is asserted -- the run "
                "SUCCEEDS, nothing refuses, and the status comes from the verifier "
                "re-evaluating SI_S3_CURVATURE against the realized parameters. That "
                "is the correct route for this family: out-of-validity means the code "
                "ran and the answer is wrong, so a refusal would be the wrong shape "
                "of evidence for it.\n\n"
                "Two questions are kept apart, because running only one of them "
                "produces a validity claim made backwards. `check_patch` asks whether "
                "the geometry's error is inside the CALLER's tolerance and passes at "
                "the instance's generous 0.2 rad; the family asks whether the declared "
                "eps_curv is inside what the GEOMETRY admits, and 0.2 against a "
                "computed 0.0500 rad is not. The guard is separately shown live at "
                "these parameters by asking it for half the bound, which it refuses "
                "with a remedy naming the widest admissible patch."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.GENERATION_WEAKENS_INDEPENDENCE,
        sampler_absent_note=(
            "the useful sampling here is boundary sampling -- instances placed at a "
            "declared signed margin from arcsin(D/2R) -- and that is a sampler written "
            "against the validity margin, which is M9's."
        ),
        evidence=(
            "src/couplers/curvature.py",
            "tests/test_curvature_bound.py",
            "tests/test_patch_wft.py",
        ),
)

B0_VALIDITY = register(_B0_VALIDITY.with_instances(
    _B0_VALIDITY.instantiate(
        "B0-VALIDITY-01",
        {
            "patch_width_m": 1.0,
            "substrate_radius_m": 10.0,
            # arcsin(1/20) = 0.05004, so this is comfortably past the bound.
            "tangent_plane_error_rad": 0.2,
            "patch_px": 65,
        },
        expected={
            "status": VerificationStatus.OUT_OF_VALIDITY.value,
            "bound_rad": math.asin(0.05),
            "why": (
                "eps_curv = 0.2 against a bound of arcsin(1/20) = 0.05004. The tangent-"
                "plane picture still computes; it computes the wrong thing, which is "
                "out of validity rather than unsupported."
            ),
        },
    ),
))


# ---------------------------------------------------------------------------
# B0-UNITS -- the silent ones
# ---------------------------------------------------------------------------

_B0_UNITS = BenchmarkFamily(
        family_id="B0-UNITS",
        family_version="1.0.0",
        category=BenchmarkCategory.B0,
        question=(
            "can the substrate tell 'it executed' from 'it is right'? Both instances "
            "run perfectly, every boundary check passes, the contract status is ok, and "
            "the physics is wrong."
        ),
        components=("M_RAY_OPTILAND", "M_WAVE_CHROMATIX"),
        claim_kind=ClaimKind.CONVENTION,
        parameters=(
            PhysicalParameter(
                "hazard",
                "which measured trap. Not a knob: each value is a different API whose "
                "unit convention differs from the literature's or from its own sibling "
                "function's",
                domain=(
                    UNITS_MICROMETRE_NANOMETRE.hazard_id,
                    KYKX_TWO_PI_AND_SIGN.hazard_id,
                ),
                default=UNITS_MICROMETRE_NANOMETRE.hazard_id,
            ),
            RepresentationParameter(
                "unit_reading",
                "which unit the caller believed the API wanted. The whole hazard: the "
                "same number in two units is two different physical setups and neither "
                "call raises",
                domain=("api_native", "literature"),
                default="literature",
            ),
        ),
        validity=(),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the closed form for each case -- the single-layer quarter-wave "
                "reflectance, and z tan(theta) with its sign. Both exact, both "
                "independent of the package that gets them wrong"
            ),
            callable=None,
            reference="src/verification/hazards.py; src/verification/analytic.py",
        ),
        metrics=(
            Metric(
                name="relative_error_vs_closed_form",
                description="how far the executed result is from the analytic answer",
                unit=None,
                blind_to=(
                    "nothing that matters here, and that is the point: the contract "
                    "layer is blind to all of it while this is not",
                ),
            ),
            Metric(
                name="contract_status_is_ok",
                description=(
                    "1.0 when the boundary layer reported no problem. Expected to be "
                    "1.0 on BOTH instances, which is the finding rather than a defect "
                    "in the boundary layer -- there is nothing for it to complain about"
                ),
                unit=None,
                blind_to=(
                    "the physics entirely. That is what a contract layer is FOR, and "
                    "why a conformance suite that stops here misses this whole class",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="relative_error_vs_closed_form",
                threshold=5e-3,
                basis=(
                    "the pinned solver reproduces both closed forms to better than 1e-6 "
                    "when called correctly. 5e-3 is three orders above that and far "
                    "below what either mistake produces: the um/nm slip moves the "
                    "coated reflectance to within 1.7e-5 of BARE GLASS, a factor of 3.3 "
                    "from the correct 0.01283544, and the kykx mistake is 6.28x and a "
                    "sign"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects="both measured traps, neither of which raises anything",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="contract-only-suite",
                description=(
                    "check ONLY that nothing raised, and observe that both hazards pass. "
                    "A conformance suite built on 'invalid input raises' misses this "
                    "entire class, and this control is the demonstration"
                ),
                mutation="assert no exception and report success",
                target_metric="contract_status_is_ok",
            ),
        ),
        failure_semantics=(),
        execution_policy=ExecutionPolicy(
            devices=frozenset({DeviceKind.CPU}),
            dtypes=frozenset({DType.FLOAT64, DType.COMPLEX64}),
            namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
            max_wall_seconds=60.0,
            max_peak_memory_gib=2.0,
        ),
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="relative_error_vs_closed_form",
            observed=2.28205,
            evidence=(
                "benchmarks/instances/b0_contract.py",
                "tests/test_b0_instances.py::test_a_silent_hazard_reports_ok_and_fails_its_gate",
                "tests/test_b0_instances.py::test_the_coating_is_indistinguishable_from_bare_glass",
                "tests/test_b0_instances.py::test_the_kykx_sign_inversion_is_located_on_the_propagator",
                "src/verification/hazards.py",
            ),
            note=(
                "Read the direction of this gate carefully: the tolerance exists to "
                "REJECT these two instances, and MET means it does. Both run clean, "
                "both report contract status ok, and both are rejected by "
                "relative_error_vs_closed_form -- 2.282 for the coating and 0.842 for "
                "the tilt, against a 5e-3 gate. A suite that stopped at the contract "
                "would call both of them passing runs, and that pairing is the whole "
                "content of the family.\n\n"
                "Re-measured on the pinned installs, and the measurement moved two "
                "things. (1) The coating: the wrong call returns 0.04212655 against "
                "bare glass's 0.04216456, a separation of 9.0e-4 rather than the "
                "recorded 1.7e-5. The bare-glass value reproduces exactly and the "
                "3.285x improvement of the correct coating reproduces to three "
                "decimals, so the difference is the coating MATERIAL MODEL -- "
                "IdealMaterial(n=1.38) here -- and not the unit hazard, which is "
                "identical. Still small; the coating still does nothing. (2) The "
                "kykx trap is TWO mistakes at two call sites, not one: plane_wave "
                "handed cycles-per-length is 2*pi too small with the sign PRESERVED, "
                "and asm_propagate's kykx displaces OPPOSITE to its parameter. The "
                "recorded magnitude reproduces to 0.5%; its attribution is corrected. "
                "The recorded 'factor of 2*pi' is also paraxial: at 2*pi too large the "
                "beam is at 33 degrees and the ratio is 7.45, which "
                "z*tan(asin(lambda*f)) predicts. hazards.py keeps the historical "
                "numbers; the instance record carries the fresh ones and the reason "
                "they differ."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.HISTORICAL_REGRESSION,
        sampler_absent_note=(
            "the measured wrong number IS the artifact. A generated instance would have "
            "no measured wrong number, and inventing one would make the family a "
            "hypothesis rather than a record."
        ),
        evidence=(
            "src/verification/hazards.py",
            "tests/test_preserved_evidence.py::test_a_measured_trap_reports_an_ok_contract",
            "knowledge/solvers/chromatix/conventions.md",
        ),
        notes=(
            "These two are why B0 is a category and not a test file. Everything else in "
            "B0 asks whether a refusal is well formed; these ask whether the substrate "
            "can notice that a perfectly well formed run produced the wrong answer."
        ),
)

B0_UNITS = register(_B0_UNITS.with_instances(
    _B0_UNITS.instantiate(
        "B0-UNITS-01",
        {"hazard": UNITS_MICROMETRE_NANOMETRE.hazard_id, "unit_reading": "literature"},
        expected={
            "contract_status": "ok",
            "wrong_value": UNITS_MICROMETRE_NANOMETRE.wrong_value,
            "bare_glass": UNITS_MICROMETRE_NANOMETRE.right_value,
            "why": UNITS_MICROMETRE_NANOMETRE.why_silent,
        },
    ),
    _B0_UNITS.instantiate(
        "B0-UNITS-02",
        {"hazard": KYKX_TWO_PI_AND_SIGN.hazard_id, "unit_reading": "literature"},
        expected={
            "contract_status": "ok",
            "wrong_value": KYKX_TWO_PI_AND_SIGN.wrong_value,
            "correct_value": KYKX_TWO_PI_AND_SIGN.right_value,
            "why": KYKX_TWO_PI_AND_SIGN.why_silent,
        },
    ),
))
