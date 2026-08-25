"""B1 ray-primitive families: Optiland against closed forms and invariants.

CHE-106 (M1.1). There has been no node-level scientific benchmark for the ray
model in the active tree since ``L1-RAY-01`` was archived: everything the suite
has on Optiland is unit- and characterization-level, which is coverage rather
than a correctness gate with a declared oracle and a tolerance.

This is also the milestone that has to come before M2, for the reason M2's own
exit report gives: M1 is what bounds where a coupler defect can be. Without it, a
coupler failure and a solver failure are indistinguishable.

What is here, and what is deliberately not
------------------------------------------
Five families, each a physical question with a parameter space rather than one
hard-coded case. Three inherit closed forms **already verified against the pinned
solver** in the retired A1 task set; that verification is not repeated, and the
oracles carry their measured agreement (``verification/analytic.py``).

``B1-RAY-OFFAXIS-OPL`` is the highest-value family here and the reason the others
exist around it. CHE-41 found that the declared pupil OPL omitted
``n_object * (d0 . r_launch)`` -- the object-space projection of the launch
coordinate onto the chief direction -- and that on ``M3-REVERSE-TELEPHOTO`` at
``Hy = 0.2`` only **0.13%** of the required convergence tilt survived without it.
The term is *linear in the launch coordinate*, so on axis it is a constant that
cancels in the chief-ray subtraction. That is why the defect survived CHE-30,
CHE-32 and CHE-33: every one of them looked on axis. The family therefore
declares an off-axis field as a `PhysicalParameter` with a domain that excludes
zero for its gating instance, and the omission is a declared negative control
rather than a side experiment.

**No Optiland output decides Optiland's correctness.** PB7/CHE-58 finding F2
established that ``FFTPSF`` and ``HuygensPSF`` share one ``Wavefront``/OPD front
end and are not two oracles. Every gate below is a closed form or a conservation
law, and the schema refuses a ``CROSS_ROUTE`` oracle outside B4 anyway.

Status
------
The three inherited families report ``MEASURED_OFF_GATE``: their result is on
record and nothing in the required gate re-checks it. The two new families
(``B1-RAY-SNELL``, ``B1-RAY-LAGRANGE``) and ``B1-RAY-OFFAXIS-OPL``'s tilt gate
report ``NOT_MEASURED`` -- declared here and not yet executed through the
substrate. That is the honest state, and it is what CHE-113's executor and
CHE-115's substrate proof turn into measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.analytic import (
    BFL,
    EFL_BFL,
    PLATE_FOCAL_SHIFT,
)
from verification.families.predicates import fractional_margin, paraxial_field_angle
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
    SamplerAbsentReason,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = [
    "B1_RAY_EFL",
    "B1_RAY_LAGRANGE",
    "B1_RAY_OFFAXIS_OPL",
    "B1_RAY_PLATE",
    "B1_RAY_SNELL",
]


# ---------------------------------------------------------------------------
# Shared policy
# ---------------------------------------------------------------------------

#: Optiland computes in float32 or float64 on CPU and CUDA, CUDA only through the
#: torch backend. Read off ``core/capabilities.py`` rather than restated: the
#: capability table is probe-backed and this is a consumer of it.
RAY_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    dtypes=frozenset({DType.FLOAT64}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.TORCH}),
    max_wall_seconds=120.0,
    max_peak_memory_gib=4.0,
    notes=(
        "float64 for the gating instances. float32 agreement is a B0-DTYPE question, "
        "not a correctness one, and mixing them here would let a precision finding be "
        "read as a physics finding."
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "a ray trace over a declared pupil sampling is deterministic; nothing here "
        "draws a sample, so there is no ensemble to take"
    ),
)

#: The closed forms below are paraxial. A marginal ray steep enough for spherical
#: aberration to move the measured focus is outside the oracle's domain, not
#: evidence against the solver.
PARAXIAL_MARGINAL = paraxial_field_angle(
    angle_key="marginal_ray_angle_rad", max_angle_rad=math.radians(5.0)
)


# ---------------------------------------------------------------------------
# B1-RAY-EFL
# ---------------------------------------------------------------------------

B1_RAY_EFL = register(
    BenchmarkFamily(
        family_id="B1-RAY-EFL",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "does a traced thick plano-convex singlet in air reproduce the paraxial "
            "closed forms R/(n-1) for the effective focal length and EFL - t/n for the "
            "back focal length from the rear vertex?"
        ),
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "radius_mm",
                "radius of curvature of the convex front surface",
                unit="mm",
                domain=(5.0, 500.0),
            ),
            PhysicalParameter(
                "index",
                "refractive index of the lens material at the working wavelength",
                domain=(1.3, 2.0),
            ),
            PhysicalParameter(
                "thickness_mm",
                "centre thickness; what makes this more than arithmetic, because it is "
                "the whole of the EFL/BFL difference",
                unit="mm",
                domain=(0.5, 20.0),
            ),
            PhysicalParameter(
                "wavelength_um",
                "working wavelength; the index is quoted at it",
                unit="um",
                domain=(0.4, 1.6),
                default=0.5876,
            ),
            PhysicalParameter(
                "marginal_ray_angle_rad",
                "half-angle of the traced cone, which is what decides whether the "
                "paraxial closed form applies at all",
                unit="rad",
                domain=(0.0, 0.35),
                default=0.01,
            ),
            NumericalParameter(
                "pupil_rings",
                "hexapolar rings across the pupil",
                domain=(4, 128),
                default=32,
                refines_toward=1,
            ),
        ),
        validity=(PARAXIAL_MARGINAL,),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=EFL_BFL.statement,
            callable=EFL_BFL.closed_form,
            reference="verification/analytic.py::EFL_BFL, BFL",
        ),
        metrics=(
            Metric(
                name="efl_relative_error",
                description="relative error of the traced EFL against R/(n-1)",
                unit=None,
                blind_to=(
                    "the sign convention of the focal length: it is a magnitude ratio, "
                    "so a system that focuses on the wrong side passes it",
                ),
            ),
            Metric(
                name="bfl_relative_error",
                description="relative error of the traced BFL against EFL - t/n",
                unit=None,
                blind_to=(
                    "which of the two terms is wrong -- an EFL error and a t/n error "
                    "of the same size are indistinguishable here, which is why the "
                    "EFL is graded separately",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="efl_relative_error",
                threshold=1e-6,
                basis=EFL_BFL.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=EFL_BFL.rejects,
            ),
            Tolerance(
                metric="bfl_relative_error",
                threshold=1e-6,
                basis=BFL.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=BFL.rejects,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="thin-lens-bfl",
                description=(
                    "report the EFL as the BFL, i.e. omit the thick-lens correction"
                ),
                mutation="return EFL in place of EFL - t/n",
                target_metric="bfl_relative_error",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.INVALID_CONFIGURATION,
        ),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MEASURED_OFF_GATE,
            metric="efl_relative_error",
            observed=1e-13,
            evidence=("src/verification/analytic.py",),
            note=(
                "verified against the pinned solver to 1e-13 relative before the "
                "retired A1 task set shipped. Nothing in the required gate re-runs it, "
                "which is what MEASURED_OFF_GATE says and why CHE-113/CHE-115 are the "
                "tickets that change it."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "generation is nearly free here -- the oracle is arithmetic in "
            "(R, n, t) -- but a sampler needs an Optiland prescription builder that "
            "produces a valid system for every drawn point, and that is M9's work. "
            "The domain is declared so the sampler has something to draw from."
        ),
        evidence=(
            "src/verification/analytic.py",
            "tests/test_preserved_evidence.py::test_the_thick_singlet_closed_forms_still_evaluate",
        ),
        notes=(
            "The EFL and BFL are graded separately on purpose: they differ by 2.64 mm "
            "on the reference prescription, so an implementation that reports the same "
            "number twice fails the second check and only the second check."
        ),
    )
)


# ---------------------------------------------------------------------------
# B1-RAY-PLATE
# ---------------------------------------------------------------------------

B1_RAY_PLATE = register(
    BenchmarkFamily(
        family_id="B1-RAY-PLATE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "does a plane-parallel plate in a converging beam move the focus by "
            "t(1 - 1/n), AWAY from the plate?"
        ),
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "thickness_mm", "plate thickness", unit="mm", domain=(0.5, 50.0)
            ),
            PhysicalParameter("index", "plate refractive index", domain=(1.3, 2.0)),
            PhysicalParameter(
                "focal_length_mm",
                "focal length of the converging beam the plate sits in",
                unit="mm",
                domain=(10.0, 1000.0),
                default=100.0,
            ),
            PhysicalParameter(
                "marginal_ray_angle_rad",
                "half-angle of the converging cone",
                unit="rad",
                domain=(0.0, 0.2),
                default=0.005,
            ),
            NumericalParameter(
                "axis_crossing_samples",
                "rays used to locate the axis crossing",
                domain=(3, 4096),
                default=64,
                refines_toward=1,
            ),
        ),
        validity=(PARAXIAL_MARGINAL,),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=PLATE_FOCAL_SHIFT.statement,
            callable=PLATE_FOCAL_SHIFT.closed_form,
            reference="verification/analytic.py::PLATE_FOCAL_SHIFT",
        ),
        metrics=(
            Metric(
                name="plate_focal_shift_signed_relative_error",
                description=(
                    "signed relative error of the traced focal shift against t(1 - 1/n)"
                ),
                unit=None,
                blind_to=(
                    "the mechanism -- a plate of the wrong thickness and one of the "
                    "wrong index produce the same shift when t(1-1/n) matches",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="plate_focal_shift_signed_relative_error",
                threshold=1e-3,
                basis=PLATE_FOCAL_SHIFT.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=PLATE_FOCAL_SHIFT.rejects,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="sign-flip",
                description=(
                    "report the shift toward the plate instead of away from it"
                ),
                mutation="negate the computed focal shift",
                target_metric="plate_focal_shift_signed_relative_error",
            ),
            NegativeControl(
                control_id="t-over-n",
                description="report t/n instead of t(1 - 1/n)",
                mutation="divide the thickness by the index instead of applying (1 - 1/n)",
                target_metric="plate_focal_shift_signed_relative_error",
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MEASURED_OFF_GATE,
            metric="plate_focal_shift_signed_relative_error",
            observed=1.3e-5,
            evidence=("src/verification/analytic.py",),
            note=(
                "a real trace at h = 0.5 mm into f = 100 mm gave 3.750048 mm against "
                "3.75 analytic before the A1 task shipped. On record, re-checked by "
                "nothing in the required gate."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note="same as B1-RAY-EFL: the domain is declared, the builder is M9's.",
        evidence=(
            "src/verification/analytic.py",
            "tests/test_preserved_evidence.py::test_the_plate_focal_shift_keeps_its_sign",
        ),
        notes=(
            "The metric is SIGNED. A magnitude-only comparison would pass a run that "
            "moved the focus the wrong way, which is the whole content of the claim."
        ),
    )
)


# ---------------------------------------------------------------------------
# B1-RAY-SNELL
# ---------------------------------------------------------------------------


def _snell_refraction_angle(params: Mapping[str, Any]) -> float:
    """``n1 sin(theta1) = n2 sin(theta2)``, exact.

    Returns the refracted angle in radians. Past the critical angle there is no
    real solution, which is what ``TIR_BOUNDARY`` below expresses as a validity
    margin rather than as an exception: total internal reflection is a physical
    regime, not a numerical failure.
    """
    n1 = float(params["index_incident"])
    n2 = float(params["index_transmitted"])
    theta1 = float(params["incidence_angle_rad"])
    ratio = n1 * math.sin(theta1) / n2
    if abs(ratio) > 1.0:
        raise ValueError(
            f"no refracted ray: n1 sin(theta1)/n2 = {ratio:.4f} > 1. This instance is "
            "past the critical angle and outside the family's validity domain."
        )
    return math.asin(ratio)


def _tir_margin(params: Mapping[str, Any]) -> float:
    """Signed distance from the critical angle, normalized by it.

    ``+1`` at normal incidence, ``0`` exactly at the critical angle, negative
    past it. With ``n2 >= n1`` there is no critical angle and the margin is
    ``+inf``: an unbounded direction is not the same as a comfortable one, and
    reporting a finite number here would invite a sampler to chase a boundary
    that is not there.
    """
    n1 = float(params["index_incident"])
    n2 = float(params["index_transmitted"])
    if n2 >= n1:
        return math.inf
    critical = math.asin(n2 / n1)
    return fractional_margin(float(params["incidence_angle_rad"]), critical)


TIR_BOUNDARY = ValidityPredicate(
    predicate_id="TIR_CRITICAL_ANGLE",
    statement=(
        "the incidence angle stays below the critical angle asin(n2/n1), so a "
        "refracted ray exists"
    ),
    basis=ValidityBasis.PARAXIAL_APPROXIMATION,
    margin=_tir_margin,
    blind_to=(
        "the Fresnel amplitudes -- an angle just inside the critical angle has a "
        "real refracted direction and almost no transmitted power, and this "
        "predicate says nothing about that",
    ),
)


B1_RAY_SNELL = register(
    BenchmarkFamily(
        family_id="B1-RAY-SNELL",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "does refraction at a single planar interface satisfy "
            "n1 sin(theta1) = n2 sin(theta2) across the incidence range, and is total "
            "internal reflection reported as a validity boundary rather than as a "
            "numerical failure?"
        ),
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.CONVENTION,
        parameters=(
            PhysicalParameter(
                "index_incident", "index on the incident side", domain=(1.0, 2.5)
            ),
            PhysicalParameter(
                "index_transmitted", "index on the transmitted side", domain=(1.0, 2.5)
            ),
            PhysicalParameter(
                "incidence_angle_rad",
                "angle of incidence from the surface normal; the family is swept over "
                "it, including near-grazing",
                unit="rad",
                domain=(0.0, math.pi / 2),
            ),
            ExecutionParameter(
                "device", "cpu or cuda", domain=("cpu", "cuda"), default="cpu"
            ),
        ),
        validity=(TIR_BOUNDARY,),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description="n1 sin(theta1) = n2 sin(theta2), exact at a planar interface",
            callable=_snell_refraction_angle,
            reference="Snell's law; no approximation and no fitted constant",
        ),
        metrics=(
            Metric(
                name="refraction_angle_absolute_error_rad",
                description="absolute error of the traced refracted angle, in radians",
                unit="rad",
                blind_to=(
                    "the azimuth: a planar interface refracts in the plane of "
                    "incidence, and an angle-only metric cannot see a ray that left it",
                ),
            ),
        ),
        invariants=(
            Invariant(
                invariant_id="DIRECTION_COSINES_UNIT_NORM",
                statement=(
                    "the refracted direction cosines have unit norm to the dtype's "
                    "floor; the artifact boundary already enforces this and the family "
                    "asserts it rather than assuming the boundary ran"
                ),
                metric="refraction_angle_absolute_error_rad",
                tolerance=Tolerance(
                    metric="refraction_angle_absolute_error_rad",
                    threshold=1e-12,
                    basis=(
                        "float64 round-off over a handful of trigonometric operations; "
                        "the boundary check uses a dtype-dependent tolerance of the "
                        "same order"
                    ),
                    basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                    may_gate=True,
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="refraction_angle_absolute_error_rad",
                threshold=1e-12,
                basis=(
                    "Snell's law is exact, so the only admissible error is float64 "
                    "round-off through asin and the direction-cosine construction. "
                    "1e-12 rad is about four orders above that floor and about ten "
                    "orders below any convention error"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a degrees-for-radians slip, a normal pointing the wrong way, and "
                    "the small-angle substitution sin(theta) -> theta, which is 1.7e-3 "
                    "rad off already at 10 degrees"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="small-angle-substitution",
                description="use theta in place of sin(theta) on both sides",
                mutation="n1 * theta1 = n2 * theta2",
                target_metric="refraction_angle_absolute_error_rad",
            ),
            NegativeControl(
                control_id="inverted-index-ratio",
                description="apply n2/n1 where n1/n2 is required",
                mutation="theta2 = asin(n2 sin(theta1) / n1)",
                target_metric="refraction_angle_absolute_error_rad",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.INVALID_CONFIGURATION,
        ),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MEASURED,
            note=(
                "declared here and never executed through the substrate. The repository "
                "has plenty of traces that would fail if Snell were wrong, and none of "
                "them is a gate on Snell."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the oracle is one line, but the interesting sampling is BOUNDARY sampling "
            "-- drawing incidence angles that straddle the critical angle -- and that "
            "needs the M9 sampler that reads validity margins."
        ),
        evidence=(
            "benchmarks/probes/records/optiland/raytrace_probe.json",
            "knowledge/solvers/optiland/conventions.md",
        ),
        notes=(
            "TIR is the point of the validity predicate. A family that treated it as "
            "an exception would report 'invalid configuration' for a real physical "
            "regime, and the substrate keeps those apart: out_of_validity means the "
            "approximation does not apply, invalid_configuration means the request is "
            "malformed."
        ),
    )
)


# ---------------------------------------------------------------------------
# B1-RAY-LAGRANGE
# ---------------------------------------------------------------------------


def _lagrange_invariant(params: Mapping[str, Any]) -> float:
    """``H = n (u y_bar - u_bar y)``, conserved through any paraxial system.

    A conservation law rather than a closed-form answer: the oracle value is
    whatever the invariant is at the object, and the claim is that it does not
    move.
    """
    n = float(params["index_object_space"])
    u = float(params["marginal_ray_angle_rad"])
    y_bar = float(params["chief_ray_height_mm"])
    u_bar = float(params["chief_ray_angle_rad"])
    y = float(params["marginal_ray_height_mm"])
    return n * (u * y_bar - u_bar * y)


B1_RAY_LAGRANGE = register(
    BenchmarkFamily(
        family_id="B1-RAY-LAGRANGE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "is the Lagrange invariant H = n(u y_bar - u_bar y) conserved surface by "
            "surface through a multi-element system?"
        ),
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.CONSERVATION,
        parameters=(
            PhysicalParameter(
                "index_object_space", "object-space index", domain=(1.0, 2.0), default=1.0
            ),
            PhysicalParameter(
                "marginal_ray_angle_rad",
                "marginal ray angle in object space",
                unit="rad",
                domain=(1e-4, 0.35),
            ),
            PhysicalParameter(
                "marginal_ray_height_mm",
                "marginal ray height at the object",
                unit="mm",
                domain=(0.0, 100.0),
                default=0.0,
            ),
            PhysicalParameter(
                "chief_ray_angle_rad",
                "chief ray angle in object space, i.e. the field",
                unit="rad",
                domain=(0.0, 0.35),
            ),
            PhysicalParameter(
                "chief_ray_height_mm",
                "chief ray height at the object",
                unit="mm",
                domain=(0.0, 100.0),
            ),
            PhysicalParameter(
                "surface_count",
                "how many refracting surfaces the invariant has to survive",
                domain=(2, 12),
                default=3,
            ),
        ),
        validity=(
            paraxial_field_angle(angle_key="chief_ray_angle_rad", max_angle_rad=math.radians(5.0)),
            PARAXIAL_MARGINAL,
        ),
        oracle=FamilyOracle(
            kind=Oracle.CONSERVATION_LAW,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the Lagrange invariant is conserved through any paraxial system; the "
                "reference value is its own object-space value, so nothing external is "
                "needed and nothing internal decides it"
            ),
            callable=_lagrange_invariant,
            reference="paraxial optics; Welford, Aberrations of Optical Systems, eq 2.16",
        ),
        metrics=(
            Metric(
                name="lagrange_invariant_relative_drift",
                description=(
                    "max |H_k - H_0| / |H_0| over all surfaces k in the system"
                ),
                unit=None,
                blind_to=(
                    "a compensating error: a system that scaled u and y_bar by "
                    "reciprocal factors would conserve H while getting both wrong",
                    "everything non-paraxial -- H is a paraxial invariant and says "
                    "nothing about aberration",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="lagrange_invariant_relative_drift",
                threshold=1e-10,
                basis=(
                    "conservation to float64 round-off accumulated over at most twelve "
                    "surfaces. Each surface contributes a few operations on quantities "
                    "of order 1, so 1e-10 is several orders above the floor and far "
                    "below any real transfer-matrix error"
                ),
                basis_kind=ToleranceBasis.CONSERVATION_LAW,
                may_gate=True,
                rejects=(
                    "a missing index factor at a refraction, which changes H by the "
                    "index ratio -- of order 0.5 relative, ten orders outside this"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="omit-index-at-refraction",
                description="drop the n factor when transferring across a surface",
                mutation="propagate u instead of n*u through one refraction",
                target_metric="lagrange_invariant_relative_drift",
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MEASURED,
            note=(
                "declared and never executed. It is the cheapest possible whole-system "
                "check on the ray model and the repository has never run it."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "generating a random multi-surface system that is neither degenerate nor "
            "aberration-dominated is the work, and it belongs with M9's sampler."
        ),
        evidence=(
            "src/registry/prescriptions.py",
            "docs/prescriptions/canonical_optical_systems.md",
        ),
        notes=(
            "A conservation law is the strongest oracle shape available for a "
            "multi-surface system, because it needs no external reference at all: the "
            "system is compared with itself at a different place, not with another "
            "implementation."
        ),
    )
)


# ---------------------------------------------------------------------------
# B1-RAY-OFFAXIS-OPL -- the load-bearing one
# ---------------------------------------------------------------------------


def _required_launch_tilt_waves(params: Mapping[str, Any]) -> float:
    """Peak-to-valley tilt, in waves, that an off-axis pupil OPL must carry.

    The omitted term is ``n_object * (d0 . r_launch)``: the object-space
    projection of the launch coordinate onto the chief direction. Over a pupil of
    diameter ``D`` with a chief direction tilted by ``theta``, its peak-to-valley
    span is ``n_object * D * sin(theta)``, which in waves is that over ``lambda``.

    Linear in the launch coordinate, and therefore a *constant* on axis, where it
    cancels in the chief-ray subtraction. That is the whole reason the defect
    survived three characterizations that all looked on axis.
    """
    n = float(params["index_object_space"])
    d = float(params["pupil_diameter_m"])
    theta = float(params["field_angle_rad"])
    lam = float(params["wavelength_m"])
    return n * d * math.sin(theta) / lam


B1_RAY_OFFAXIS_OPL = register(
    BenchmarkFamily(
        family_id="B1-RAY-OFFAXIS-OPL",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "does the pupil OPL declared from an Optiland trace carry the full "
            "convergence tilt an off-axis field requires, including the object-space "
            "term n_object * (d0 . r_launch)?"
        ),
        components=("M_RAY_OPTILAND",),
        claim_kind=ClaimKind.CONVENTION,
        parameters=(
            PhysicalParameter(
                "field_angle_rad",
                "chief-ray field angle. THE parameter of this family: at zero the "
                "omitted term is a constant and the defect is invisible",
                unit="rad",
                domain=(0.0, 0.35),
            ),
            PhysicalParameter(
                "pupil_diameter_m",
                "pupil diameter over which the tilt is measured",
                unit="m",
                domain=(1e-4, 0.2),
            ),
            PhysicalParameter(
                "wavelength_m", "wavelength", unit="m", domain=(3e-7, 2e-6), default=5.5e-7
            ),
            PhysicalParameter(
                "index_object_space",
                "object-space index; it multiplies the omitted term",
                domain=(1.0, 2.0),
                default=1.0,
            ),
            NumericalParameter(
                "pupil_rings",
                "hexapolar rings across the pupil",
                domain=(8, 128),
                default=32,
                refines_toward=1,
            ),
            PhysicalParameter(
                "prescription",
                "which canonical system is traced; a PHYSICAL parameter because a "
                "different prescription is a different optical system with a different "
                "correct answer, not a different way of representing one",
                domain=("M3-SINGLET-REF", "M3-REVERSE-TELEPHOTO"),
                default="M3-REVERSE-TELEPHOTO",
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="OFF_AXIS_FIELD_NONZERO",
                statement=(
                    "the field angle is far enough from zero that the omitted "
                    "object-space term is not a constant across the pupil"
                ),
                basis=ValidityBasis.PARAXIAL_APPROXIMATION,
                margin=lambda p: fractional_margin(
                    1e-3, max(float(p["field_angle_rad"]), 1e-12)
                ),
                blind_to=(
                    "whether the traced system actually has an off-axis field stop; a "
                    "nonzero requested angle that is vignetted away is still zero here",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the peak-to-valley launch tilt n_object * D * sin(theta) / lambda "
                "waves that the object-space term contributes, computed from geometry "
                "alone and not from any trace"
            ),
            callable=_required_launch_tilt_waves,
            reference="CHE-41; benchmarks/probes/optiland/off_axis_opd_reference.py",
        ),
        metrics=(
            Metric(
                name="launch_tilt_fraction_recovered",
                description=(
                    "the tilt the declared pupil OPL carries, as a fraction of the tilt "
                    "the geometry requires. 1.0 is correct; CHE-41 measured 0.0013 "
                    "before the term was added"
                ),
                unit=None,
                blind_to=(
                    "piston -- a constant OPL offset is invisible here, and correctly "
                    "so, because it is unobservable",
                    "higher-order aberration: this is the linear term only, so a "
                    "correctly-tilted wavefront with wrong curvature scores 1.0",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="launch_tilt_fraction_recovered",
                threshold=1e-3,
                basis=(
                    "the metric is |1 - recovered|, so the threshold is a fractional "
                    "shortfall in a term that is exactly computable from geometry. "
                    "There is no approximation on either side, so the only admissible "
                    "error is trace round-off"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "the CHE-41 defect itself: omitting n_object * (d0 . r_launch) "
                    "recovered 0.0013 of the required tilt on M3-REVERSE-TELEPHOTO at "
                    "Hy = 0.2, a shortfall of 0.9987 -- three orders outside this "
                    "threshold"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="omit-object-space-term",
                description=(
                    "the CHE-41 defect, reproduced deliberately: drop "
                    "n_object * (d0 . r_launch) from the declared pupil OPL"
                ),
                mutation=(
                    "in the shipping adapter, do not add the object-space projection of "
                    "the launch coordinate onto the chief direction before declaring "
                    "the OPL"
                ),
                target_metric="launch_tilt_fraction_recovered",
            ),
            NegativeControl(
                control_id="on-axis-cannot-detect-it",
                description=(
                    "run the SAME omission at field_angle_rad = 0 and require it NOT to "
                    "fire. This is the control on the control: it demonstrates that the "
                    "family's off-axis instance is doing the work, and that an on-axis "
                    "suite would have reported the defect as absent -- which is exactly "
                    "what happened three times"
                ),
                mutation="omit-object-space-term, evaluated on an on-axis instance",
                target_metric="launch_tilt_fraction_recovered",
                expectation=NegativeControlExpectation.MUST_FAIL,
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MEASURED,
            note=(
                "CHE-41 measured the defect and CHE-33 declared the fix, but no gate "
                "re-checks the fixed value: benchmarks/probes/records/"
                "m3_off_axis_handoff.json is a probe record, and a record is provenance "
                "rather than a gate. Turning this into MET is the single highest-value "
                "measurement M1 owes."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.HISTORICAL_REGRESSION,
        sampler_absent_note=(
            "M3-REVERSE-TELEPHOTO at Hy = 0.2 is the configuration the defect was found "
            "on and the one its fix was measured against. Sampling around it is "
            "valuable later; keeping that exact point stable is what makes the "
            "regression detectable now."
        ),
        evidence=(
            "benchmarks/probes/records/m3_off_axis_handoff.json",
            "benchmarks/probes/records/optiland/off_axis_opd_reference.json",
            "benchmarks/probes/off_axis_handoff.py",
        ),
        notes=(
            "The defect this family exists for converged CLEANLY -- 0.072 waves "
            "peak-to-valley against its own fitted reference sphere -- while putting "
            "the reconstructed wave 209 um from where the rays actually go. Nothing "
            "downstream noticed, and nothing downstream would have: it is the archetype "
            "of a run that succeeds and is wrong."
        ),
    )
)
