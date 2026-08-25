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

_B1_RAY_EFL = BenchmarkFamily(
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
            status=GateStatus.MET,
            metric="efl_relative_error",
            observed=2.7855e-12,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "tests/test_b1_ray_instances.py::test_the_thick_singlet_reproduces_both_closed_forms",
                "tests/test_b1_ray_instances.py::test_convergence_is_a_fitted_exponent_over_four_rungs",
            ),
            note=(
                "Executed, not inherited. 2.79e-12 relative on the EFL and 2.09e-12 on "
                "the BFL against a 1e-6 gate, from a real trace of a parameterized "
                "plano-convex singlet.\n\n"
                "The measurement is the h -> 0 limit of a four-rung ring ladder, and "
                "that is the substantive part rather than a refinement of it. A traced "
                "focal length is NOT a paraxial focal length: a real marginal ray at "
                "height h focuses short by an amount quadratic in h, so the innermost "
                "ring of a 64-ring fan reads 1.13e-6 and of an 8-ring fan 7.24e-5 -- a "
                "clean factor of four per doubling, which is the fitted exponent of "
                "-2.0000 the family now carries. Gating the finest RUNG instead of the "
                "limit would have made the 1e-6 tolerance a statement about the ray "
                "count: 1.13e-6 sits inside it by a hair and only because the ladder is "
                "deep.\n\n"
                "The inherited 1e-13 figure came from the retired A1 set and is not "
                "what this measures: that number was Optiland's paraxial solver against "
                "the closed form, and this is Optiland's TRACE against it, which is a "
                "different and harder claim. Both are correct; only one is a gate."
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


# ---------------------------------------------------------------------------
# B1-RAY-PLATE
# ---------------------------------------------------------------------------

#: The canonical instance, and the ladder it is measured on.
#:
#: A plano-convex singlet whose closed forms are exact rather than paraxial: the
#: rear surface is plano, so it has no power and ``R/(n-1)`` is the whole focal
#: length. What IS paraxial is the traced measurement -- a real marginal ray at
#: height ``h`` focuses short of the paraxial focus by an amount quadratic in
#: ``h`` -- so the traced value is extracted in the ``h -> 0`` limit from a
#: refinement ladder in the ring count, which is also what CHE-106's
#: fitted-exponent criterion asks for. The exponent is a prediction, not a
#: fitted convenience: spherical aberration is quadratic in aperture, so the
#: error must fall as ``rings^-2``.
B1_RAY_EFL = register(
    _B1_RAY_EFL.with_instances(
        _B1_RAY_EFL.instantiate(
            "B1-RAY-EFL-01",
            {
                "radius_mm": 50.0,
                "index": 1.5168,
                "thickness_mm": 4.0,
                "wavelength_um": 0.5876,
                # The largest marginal angle on the ladder's finest rung, which is
                # what the paraxial validity predicate is about. Well inside the
                # 5-degree bound.
                "marginal_ray_angle_rad": 0.0517,
                "pupil_rings": 64,
            },
            expected={
                "efl_mm": 50.0 / (1.5168 - 1.0),
                "bfl_mm": 50.0 / (1.5168 - 1.0) - 4.0 / 1.5168,
                "why": (
                    "R/(n-1) is exact for a plano-convex singlet in air and "
                    "EFL - t/n is exact for a plano rear surface. The two differ by "
                    "2.64 mm here, so an implementation that reports the same number "
                    "twice fails the BFL check and only the BFL check."
                ),
                "refinement": (
                    "measured in the h -> 0 limit from a rings ladder; a single ring "
                    "count is an aberrated focal length, not a paraxial one"
                ),
            },
        ),
    )
)


_B1_RAY_PLATE = BenchmarkFamily(
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
            status=GateStatus.MET,
            metric="plate_focal_shift_signed_relative_error",
            observed=2.42781e-12,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "tests/test_b1_ray_instances.py::test_the_plate_shift_carries_its_sign",
                "tests/test_b1_ray_instances.py::test_convergence_is_a_fitted_exponent_over_four_rungs",
            ),
            note=(
                "Executed. 2.43e-12 relative against a 1e-3 gate, from the h -> 0 limit "
                "of a four-rung ladder with a fitted exponent of -2.0001.\n\n"
                "Each rung is a DIFFERENCE of two traces of the same system with and "
                "without the plate, so the lens, the sampling and the axial-crossing "
                "extraction cancel and what is left is the plate. That is why the "
                "residual is nine orders below the inherited 1.3e-5 single-point "
                "figure: the inherited number was one ray at h = 0.5 mm, where the "
                "quadratic aperture term dominates everything else.\n\n"
                "The metric stays SIGNED and both controls fire: the shift reported "
                "toward the plate instead of away from it, and t/n reported in place of "
                "t(1 - 1/n) -- 6.25 mm where the answer is 3.75."
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


#: A 10 mm plate of index 1.6 in an f/10 converging beam. The shift is measured
#: as the DIFFERENCE between two traces of the same system with and without the
#: plate, so every common-mode property of the trace -- the lens, the sampling,
#: the axial-crossing extraction -- cancels, and what is left is the plate.
B1_RAY_PLATE = register(
    _B1_RAY_PLATE.with_instances(
        _B1_RAY_PLATE.instantiate(
            "B1-RAY-PLATE-01",
            {
                "thickness_mm": 10.0,
                "index": 1.6,
                "focal_length_mm": 100.0,
                "marginal_ray_angle_rad": 0.05,
                "axis_crossing_samples": 64,
            },
            expected={
                "shift_mm": 10.0 * (1.0 - 1.0 / 1.6),
                "sign": (
                    "positive means AWAY from the plate, i.e. the axial crossing moves "
                    "to larger z. A sign error returns -3.75 mm and is rejected"
                ),
                "why": (
                    "t(1 - 1/n) is the paraxial result and the trace approaches it "
                    "quadratically in ray height, so the shift is extracted in the "
                    "h -> 0 limit from the same rings ladder the EFL family uses"
                ),
            },
        ),
    )
)


_B1_RAY_SNELL = BenchmarkFamily(
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
            status=GateStatus.MET,
            metric="refraction_angle_absolute_error_rad",
            observed=1.66533e-16,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "tests/test_b1_ray_instances.py::test_snell_holds_to_the_floating_point_floor",
                "tests/test_b1_ray_instances.py::test_snell_is_measured_across_a_range_of_incidence_angles",
            ),
            note=(
                "Measured at four incidence angles from 0.075 to 0.467 rad, worst "
                "1.67e-16 rad against a 1e-12 gate -- one float64 epsilon, which is the "
                "only admissible residual because there is no approximation anywhere in "
                "the comparison. The repository had plenty of traces that would fail if "
                "Snell were wrong and none of them was a gate on Snell; this is.\n\n"
                "Two conventions had to be pinned first, and each had a wrong reading "
                "that produced a plausible number. (1) A collimated on-axis ray does not "
                "move transversely before the first surface, so sin(i) = rho/R exactly "
                "from the record's exported LAUNCH coordinate -- no intersection "
                "reconstruction, and no assumption about where the vertex sits in the "
                "traced frame. (2) The image plane REFRACTS: the medium after the last "
                "surface is what the ray is in when it arrives, and the image surface "
                "itself is air, so a system whose last surface carries glass applies a "
                "second refraction there. Reading the exported angle as an in-glass "
                "angle gave 22.3 degrees where the geometry requires 12.9. The oracle "
                "applies Snell twice because the geometry does.\n\n"
                "The angular RANGE is what does the work: small-angle substitution "
                "agrees with Snell to first order, so a paraxial-only suite would pass "
                "an implementation that had replaced sin with its argument. That control "
                "fires at the steep end."
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


#: Four incidence angles on one spherical interface, from paraxial to 27 degrees.
#:
#: The measurement needs no approximation anywhere. A collimated on-axis ray
#: travels parallel to z, so its transverse position where it meets the sphere is
#: exactly its launch position -- which the record exports -- and therefore
#: ``sin(i) = rho / R`` exactly. The traced direction at the image plane is then
#: compared against Snell applied twice: once at the sphere, once at the plane
#: exit face the last surface's air medium creates. Both applications are exact,
#: so the only admissible residual is float64 round-off, which is why the
#: tolerance is 1e-12 rather than something derived from a sampling argument.
#:
#: Four instances rather than one because "Snell's law at a single surface at
#: several incidence angles" is the claim, and a single angle cannot make it. The
#: steepest is chosen to sit just inside the exit face's total-internal-reflection
#: boundary, which is where the family's TIR predicate lives.
B1_RAY_SNELL = register(
    _B1_RAY_SNELL.with_instances(
        *[
            _B1_RAY_SNELL.instantiate(
                f"B1-RAY-SNELL-{index:02d}",
                {
                    "index_incident": 1.0,
                    "index_transmitted": 1.7,
                    "incidence_angle_rad": angle,
                    "device": "cpu",
                },
                expected={
                    "refraction_angle_rad": math.asin(math.sin(angle) / 1.7),
                    "why": (
                        "exact geometry on both sides: sin(i) = rho/R from the launch "
                        "coordinate, and Snell with no approximation and no fitted "
                        "constant"
                    ),
                },
            )
            for index, angle in enumerate(
                (0.0750699, 0.2269425, 0.3843967, 0.4667653), start=1
            )
        ]
    )
)


_B1_RAY_LAGRANGE = BenchmarkFamily(
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
            status=GateStatus.NOT_MET,
            metric="lagrange_invariant_relative_drift",
            observed=1.32863e-07,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "tests/test_b1_ray_instances.py::test_the_lagrange_gate_is_unmet_and_says_why",
                "tests/test_b1_ray_instances.py::test_the_lagrange_drift_vanishes_with_the_field",
            ),
            note=(
                "MEASURED AND NOT MET, and the tolerance is left exactly where it is. "
                "1.33e-7 relative drift at a 0.25-degree field and a 1/64 pupil "
                "fraction, against a 1e-10 gate.\n\n"
                "The reason is structural rather than numerical, and it is a finding "
                "about the tolerance rather than about the solver. The Lagrange "
                "invariant's two-ray bilinear form p_a.q_b - p_b.q_a is preserved by a "
                "LINEAR symplectic map. Ray refraction at a curved surface is symplectic "
                "and not linear, so only the DIFFERENTIAL form is exactly conserved, and "
                "any finite-real-ray evaluation carries an aberration residual. Measured "
                "directly while authoring this: the differential ratio between two rays "
                "of one fan converges to 1 + 7.1e-3 at a 5-degree field and does NOT "
                "approach 1 as their separation shrinks -- the signature of a "
                "finite-form residual rather than a numerical one.\n\n"
                "What the family CAN support is measured and reported: the drift "
                "vanishes with the field angle over five halvings, with a fitted "
                "exponent near 2.5. That is the conservation statement -- the invariant "
                "holds paraxially -- and it is why the drift is 1e-7 rather than 1e-2. "
                "It was the cheapest unmeasured whole-system check on the ray model and "
                "it is no longer unmeasured.\n\n"
                "The tolerance's declared basis is CONSERVATION_LAW, and the "
                "conservation law it names is paraxial while the measurement is of real "
                "rays. Re-deriving that basis against the aberration it cannot see is "
                "follow-up work; widening it to make this green would be exactly what "
                "AGENTS.md forbids."
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


#: A three-surface stack, and the family whose gate does NOT close.
#:
#: The measurement is honest and the tolerance is not achievable, which is a
#: finding rather than a failure to measure. See the gate disposition: the
#: Lagrange invariant's finite two-ray form is preserved by a *linear* symplectic
#: map, and ray refraction at a curved surface is symplectic but not linear, so a
#: finite-real-ray evaluation carries an aberration residual that no amount of
#: care removes. The residual is measured, its convergence in the field angle is
#: fitted, and the tolerance is left exactly where it is.
B1_RAY_LAGRANGE = register(
    _B1_RAY_LAGRANGE.with_instances(
        _B1_RAY_LAGRANGE.instantiate(
            "B1-RAY-LAGRANGE-01",
            {
                "index_object_space": 1.0,
                "marginal_ray_angle_rad": 0.0013,
                "marginal_ray_height_mm": 0.125,
                "chief_ray_angle_rad": 0.0043633,
                "chief_ray_height_mm": 0.0,
                "surface_count": 3,
            },
            expected={
                "drift": "0 in the paraxial limit",
                "known_disposition": (
                    "NOT_MET. Measured 1.3e-7 relative at 0.25 degrees and a 1/64 "
                    "pupil fraction, against a 1e-10 gate. The gate's basis is a "
                    "conservation law that holds paraxially; the measurement is of "
                    "real rays. Reported rather than accommodated."
                ),
            },
        ),
    )
)


_B1_RAY_OFFAXIS_OPL = BenchmarkFamily(
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
                    "an ON-AXIS field entirely. At theta = 0 the omitted term is a "
                    "constant across the pupil and the chief-ray subtraction removes "
                    "it exactly, so this metric reads 0 with the term and 0 without "
                    "it. That blindness is why the defect survived CHE-30, CHE-32 and "
                    "CHE-33, each of which looked on axis, and it is why the family's "
                    "OFF_AXIS_FIELD_NONZERO predicate puts an on-axis instance outside "
                    "its own validity rather than letting it report a pass",
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
            # CHE-106 removed a second entry here, `on-axis-cannot-detect-it`,
            # and the reason is worth keeping. It declared
            # `expectation=MUST_FAIL` while its own description said "require it
            # NOT to fire", and both cannot be true. More importantly it is not a
            # negative control at all: a NegativeControl is a deliberately WRONG
            # twin, and on axis the omitted term is exactly a piston that the
            # chief-ray subtraction removes -- so omitting it on axis is not
            # wrong, it is invisible. That is a blindness of the measurement, and
            # it belongs on the metric's `blind_to` (where it now is) plus an
            # executable demonstration
            # (tests/test_b1_ray_instances.py::
            #  test_the_omission_is_invisible_on_axis_which_is_why_it_survived).
            # Declaring it as a control that must fail would have made the family
            # report its own gate as untrustworthy every run.
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=RAY_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="launch_tilt_fraction_recovered",
            observed=6.10588e-4,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "tests/test_b1_ray_instances.py::test_the_off_axis_tilt_is_recovered_and_gated",
                "tests/test_b1_ray_instances.py::test_omitting_the_object_space_term_fails_through_the_shipping_adapter",
                "tests/test_b1_ray_instances.py::test_the_omission_is_invisible_on_axis_which_is_why_it_survived",
            ),
            note=(
                "The measurement M1 owed. 6.11e-4 against a 1e-3 gate on "
                "M3-REVERSE-TELEPHOTO at Hy = 0.2, the configuration CHE-41 found the "
                "defect on, and the control removes the term through the shipping "
                "adapter's own switch -- HandoffPerturbation("
                "reference_incoming_wavefront=False) -- so the broken arm runs the same "
                "code with n_object*(d0.r_launch) left out and nothing else changed. It "
                "recovers 9.5e-4 of the required tilt, a three-order separation from the "
                "unperturbed arm.\n\n"
                "The gated quantity is the OPL's linear SLOPE in the launch coordinate "
                "against n_object*sin(theta), with sin(theta) read off the record's "
                "chief direction rather than recomputed from the declared field. That "
                "choice matters: the peak-to-valley form needs a pupil extent on both "
                "sides, and getting the extent wrong is how this measurement first read "
                "0.6556 and looked like a 34% shortfall in a term that is exact. An "
                "off-axis collimated fan launches OFFSET transversely -- mean launch y "
                "is -0.173 mm here -- so 2*max|r| is 0.647 mm for a 0.300 mm pupil, and "
                "the peak-to-valley along the tilt axis is the entrance pupil diameter.\n\n"
                "The on-axis blindness is demonstrated rather than described: the same "
                "omission on the on-axis trace changes the tilt by exactly zero, and the "
                "term's own span is 0.000000000 waves. That is why the defect survived "
                "CHE-30, CHE-32 and CHE-33 -- every one of them looked on axis -- and it "
                "is now a declared blindness of the metric rather than a negative "
                "control that must fail."
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


#: M3-REVERSE-TELEPHOTO at Hy = 0.2: the configuration CHE-41 found the defect on,
#: and the one its fix was measured against. Kept exactly rather than sampled
#: around, because keeping that point stable is what makes the regression
#: detectable at all.
#:
#: ``pupil_diameter_m`` is the LAUNCH (entrance-pupil) extent, not the exit pupil.
#: The distinction is load-bearing and cost an hour to find: the omitted term is
#: ``n_object * (d0 . r_launch)``, a function of the launch coordinate, so its
#: peak-to-valley span is over the launch extent -- 0.300 mm on this system --
#: while the declared OPL is expressed over the exit pupil, 0.462 mm. Comparing
#: the tilt measured on one pupil to the span required over the other reads
#: 0.6556 and looks like a 34% shortfall in a term that is in fact exact.
B1_RAY_OFFAXIS_OPL = register(
    _B1_RAY_OFFAXIS_OPL.with_instances(
        _B1_RAY_OFFAXIS_OPL.instantiate(
            "B1-RAY-OFFAXIS-OPL-01",
            {
                "field_angle_rad": 0.10471975511965978,
                "pupil_diameter_m": 3.0000000000000004e-4,
                "wavelength_m": 5.5e-7,
                "index_object_space": 1.0,
                "pupil_rings": 16,
                "prescription": "M3-REVERSE-TELEPHOTO",
            },
            expected={
                "fraction_recovered_with_the_term": 1.0,
                "fraction_recovered_without_it": 0.0013,
                "why": (
                    "CHE-41: only 0.13% of the required convergence tilt survives "
                    "without n_object * (d0 . r_launch), and the reconstruction "
                    "converges CLEANLY 209 um from where the rays actually go. The "
                    "defect is invisible on axis and total off it."
                ),
            },
        ),
    )
)
