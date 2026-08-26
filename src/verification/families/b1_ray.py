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

One correction worth reading before the declarations
---------------------------------------------------
``B1-RAY-LAGRANGE`` gated on the wrong quantity and failed by 1329x for it, and
the fix was to the oracle rather than to the tolerance. Three things get called
"the Lagrange invariant": the paraxial form, the same bilinear form evaluated on
two finite real rays, and the differential symplectic form on tangent vectors.
Only the third is conserved by a real trace -- refraction at a curved surface is
symplectic and *not linear*, so the finite form carries an aberration residual
that is not round-off and never was. The family now gates on the differential
invariant at 1e-13, three decades TIGHTER than the 1e-10 it replaces, and keeps
the finite-ray number at its original value and its original threshold as a
non-gating characterization. See ``_symplectic_invariant`` for the derivation.

The general lesson is about oracle shape rather than about rays: a conservation
law is the strongest oracle available for a multi-surface system because it needs
no external reference, and that is not sufficient. The QUANTITY has to be one the
law is actually about, and a tolerance whose basis cannot bound its own metric is
invalid whichever side of the threshold the measurement lands on.

Status
------
All five families are executed against the shipping adapter by
``benchmarks/instances/b1_ray.py`` and asserted by
``tests/test_b1_ray_instances.py``. ``B1-RAY-OFFAXIS-OPL``'s tilt gate, the four
Snell instances, the two EFL closed forms, the plate shift and the corrected
Lagrange invariant all report ``MET``. Device and precision agreement is measured
on real CUDA hardware and persisted; see the RAY-4 section of the driver.
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
    """``H = n (u y_bar - u_bar y)``, the PARAXIAL Lagrange invariant.

    Retained because it is the reference value of the family's characterization
    metric, and because saying which of three distinct invariants a number is
    about is the whole content of CHE-106's Lagrange finding. Read
    :func:`_symplectic_invariant` for the one that gates.

    This form is conserved exactly by a *linear* symplectic map -- the paraxial
    transfer matrix. It is **not** conserved by a real ray map, and the residual
    is aberration rather than round-off. That is not a defect in the solver and
    it is not a numerical artifact; it is what the quantity is.
    """
    n = float(params["index_object_space"])
    u = float(params["marginal_ray_angle_rad"])
    y_bar = float(params["chief_ray_height_mm"])
    u_bar = float(params["chief_ray_angle_rad"])
    y = float(params["marginal_ray_height_mm"])
    return n * (u * y_bar - u_bar * y)


def _symplectic_invariant(params: Mapping[str, Any]) -> float:
    """The reference value of the DIFFERENTIAL optical invariant: zero drift.

    Three quantities get called "the Lagrange invariant" and only one of them is
    exactly conserved by a real ray trace. CHE-106 measured 1.33e-7 against a
    1e-10 gate by evaluating the second on data that only the third can support,
    and the correction is to state which is which.

    **1. The paraxial invariant.** ``H = n (u y_bar - u_bar y)`` on two chosen
    rays, with ``u`` a paraxial angle. Conserved exactly by the linear transfer
    matrix; a statement about the linearized system, not about any ray.

    **2. The finite-real-ray bilinear form.** The same expression evaluated on
    two real traced rays at finite separation. Conserved by a *linear* symplectic
    map only. Real refraction at a curved surface is symplectic and **not**
    linear, so this carries an aberration residual that does not vanish as the
    two rays approach each other -- because it is not a derivative of anything.
    ``lagrange_invariant_relative_drift`` is this quantity, and it is measured
    and reported and does not gate.

    **3. The differential symplectic invariant.** With the transverse position
    ``q = (x, y)`` on a plane of constant ``z`` and the canonical momentum
    ``p = n (L, M)`` -- the index-weighted direction cosines, *not* the ray
    slopes ``M/N`` -- the ray map between two such planes is the flow of a
    Hamiltonian system in which ``z`` is the evolution parameter. Its tangent map
    ``J`` therefore satisfies ``J^T Omega J = Omega`` at every point, however
    nonlinear the map itself is, and consequently

        omega(v_a, v_b) = sum_k (dp_k^a dq_k^b - dp_k^b dq_k^a)

    is conserved **exactly**, for real rays, at any aperture and any field, for
    any pair of tangent vectors ``v = (dq, dp)``. This is the invariant that a
    conservation-law gate can legitimately be written against, and it is a
    strictly stronger claim than (1): it holds where the paraxial statement does
    not. Luneburg, *Mathematical Theory of Optics*, ch. II; Born & Wolf sec. 3.3;
    Wolf, *Geometric Optics on Phase Space*, ch. 2.

    The oracle value is zero drift: ``omega`` at the image plane equals
    ``omega`` at the launch plane. Nothing external is needed and nothing
    internal decides it -- the same shape the family always had, applied to the
    quantity that actually has the property.

    What makes it executable on a solver that traces finite rays is that a
    tangent vector is reachable by extrapolation. Two symmetric secants about a
    common base ray -- one drawn across the pupil at fixed field, one across the
    field at fixed pupil -- approximate two linearly independent tangent vectors
    to ``O(eps^2)``, so the measured residual is a finite-difference truncation
    error with a known exponent and Richardson extrapolation removes it. Both
    directions are required: two secants drawn from the same pupil fan are
    parallel in the ``eps -> 0`` limit, ``omega`` between them is identically
    zero on the object side, and the ratio is a 0/0 that converges to nothing.
    That is a declared negative control here, and it is the specific error that
    produced CHE-106's "converges to 1 + 7.1e-3 and does not approach 1".
    """
    return 0.0


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
        family_version="2.0.0",
        category=BenchmarkCategory.B1,
        question=(
            "is the differential optical invariant omega(v_a, v_b) = sum_k "
            "(dp_k^a dq_k^b - dp_k^b dq_k^a) -- with q the transverse position on a "
            "plane of constant z and p = n (L, M) the index-weighted direction "
            "cosines -- conserved by the real ray map through a multi-element "
            "system? Equivalently: is the traced map symplectic?"
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
            NumericalParameter(
                "perturbation_scale",
                "normalized half-separation of the coarsest secant pair, as a "
                "fraction of both the pupil semi-diameter and the declared field "
                "angle. It moves the achieved accuracy of the tangent vectors and "
                "NOT the invariant they are evaluated on -- which is the property "
                "the eps^2 ladder below exists to demonstrate rather than assume",
                domain=(1e-3, 1.0),
                default=0.5,
                refines_toward=-1,
            ),
            NumericalParameter(
                "perturbation_rungs",
                "halvings of perturbation_scale. Six, because the residual has to "
                "be shown to be truncation error before it may be extrapolated "
                "away, and Richardson to the eps^4 column needs three rungs of "
                "the four the exponent fit already needs",
                domain=(4, 10),
                default=6,
                refines_toward=1,
            ),
            NumericalParameter(
                "richardson_levels",
                "orders of the finite-difference expansion removed by "
                "extrapolation. Two -- eps^2 then eps^4 -- because the third "
                "column already sits at the float64 round-off floor and a fourth "
                "would only amplify it",
                domain=(0, 4),
                default=2,
                refines_toward=1,
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
                "the differential optical invariant is conserved by any symplectic "
                "map, and the ray map with z as the evolution parameter is one. The "
                "reference value is the invariant's own value at the launch plane, "
                "so nothing external is needed and nothing internal decides it -- "
                "and unlike the paraxial form this holds for REAL rays, which is "
                "what makes a conservation-law tolerance defensible at all"
            ),
            callable=_symplectic_invariant,
            reference=(
                "Hamiltonian optics: Luneburg, Mathematical Theory of Optics ch. II; "
                "Born & Wolf sec. 3.3; Wolf, Geometric Optics on Phase Space ch. 2. "
                "The paraxial special case is Welford, Aberrations of Optical "
                "Systems, eq 2.16"
            ),
        ),
        metrics=(
            Metric(
                name="symplectic_invariant_relative_residual",
                description=(
                    "|omega_image - omega_object| / |omega_object| on two "
                    "independent tangent vectors at a common base ray, Richardson-"
                    "extrapolated to zero ray separation over the perturbation "
                    "ladder"
                ),
                unit=None,
                blind_to=(
                    "which prescription was traced. EVERY valid ray trace of EVERY "
                    "valid system is symplectic, so a system built from the wrong "
                    "radii or the wrong glass conserves this exactly. It says the "
                    "refraction is implemented as a canonical transformation, not "
                    "that the system is the one that was asked for -- B1-RAY-EFL, "
                    "-PLATE and -SNELL are what decide that. Measured, not argued: "
                    "replacing the second element's index 1.62 by 1.0 leaves the "
                    "extrapolated residual at 3.5e-15",
                    "a canonically conjugate compensating error: a map that scaled "
                    "q by alpha and p by 1/alpha has unit determinant and passes. "
                    "Scaling either one ALONE is caught, and both directions are "
                    "declared as controls, but the conjugate pair is not",
                    "the difference between the canonical momentum n*M and the ray "
                    "slope M/N at an axial base ray. The two agree to first order "
                    "there, so substituting the slope -- the previous formulation's "
                    "own error -- changes nothing in the eps -> 0 limit and shows "
                    "up only at finite separation. Measured: 1.52e-7 against the "
                    "canonical 5.48e-7 at the finest rung, 4.1e-16 extrapolated",
                    "everything aberration. Finite-ray aberration is precisely what "
                    "the extrapolation removes, and it is reported separately as "
                    "lagrange_invariant_relative_drift rather than folded in here",
                ),
            ),
            Metric(
                name="lagrange_invariant_relative_drift",
                description=(
                    "|H_image - H_object| / |H_object| for the PARAXIAL form "
                    "H = n(u y_bar - u_bar y) evaluated on two finite real rays: "
                    "the innermost meridional ray of the on-axis fan and the chief "
                    "ray of the off-axis fan. A characterization of the paraxial "
                    "domain, not a conservation statement"
                ),
                unit=None,
                blind_to=(
                    "a compensating error: a system that scaled u and y_bar by "
                    "reciprocal factors would conserve H while getting both wrong",
                    "everything non-paraxial -- H is a paraxial invariant and says "
                    "nothing about aberration",
                    "the fact that it is not a conservation statement at all for "
                    "finite real rays. The bilinear form p_a.q_b - p_b.q_a is "
                    "preserved by a LINEAR symplectic map; the residual this metric "
                    "reports is aberration, and it does NOT vanish as the two rays "
                    "approach one another, because it is not the derivative of "
                    "anything. Conserved quantities live on tangent vectors, which "
                    "is what symplectic_invariant_relative_residual measures",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="symplectic_invariant_relative_residual",
                threshold=1e-13,
                basis=(
                    "float64 round-off in the extrapolated invariant, derived and "
                    "then checked against the measurement rather than fitted to it. "
                    "Three terms, and no aberration term because the extrapolation "
                    "removes it: (1) ROUND-OFF. eps = 2.2204e-16. The traced state "
                    "carries the accumulated round-off of the trace itself, bounded "
                    "by the same 64*eps the adapter derives for its own direction-"
                    "norm check (solvers/optiland/execution.py::"
                    "_direction_norm_tolerance) -- a few operations per surface on "
                    "quantities of order one, over at most twelve surfaces. Forming "
                    "omega amplifies that by the measured conditioning of the "
                    "secants and of the bilinear form, sum|terms| / |omega| = "
                    "1.0005 and |q| / |dq| = 0.5, i.e. by less than one. Richardson "
                    "to two levels amplifies it by prod (2^p + 1)/(2^p - 1) for "
                    "p = 2, 4 = 1.889. So 64 * 2.2204e-16 * 1.0005 * 1.889 = "
                    "2.7e-14. (2) TRUNCATION of the extrapolation itself, O(eps^6) "
                    "after two Richardson levels, estimated from the next column at "
                    "6.1e-17 -- three decades below the round-off term and "
                    "therefore not what sets this. (3) HEADROOM: one decade, so a "
                    "deeper system or a slightly worse-conditioned base ray does "
                    "not need the number moved. 1e-13. This is three decades "
                    "TIGHTER than the 1e-10 it replaces, and the measured value "
                    "1.204e-15 sits 83x inside it and 5.4x above the single-"
                    "operation float64 floor"
                ),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects=(
                    "a refraction implemented as anything other than a canonical "
                    "transformation. Measured on the shipping trace: a 1e-6 "
                    "relative non-symplectic scaling of the image-plane momenta "
                    "reads 1.0000e-6, and the same scaling of the image-plane "
                    "positions reads 1.0000e-6 -- both seven decades outside this "
                    "gate, against a baseline of 1.204e-15, a detection margin of "
                    "8.3e8x. An omitted index factor across an index step would "
                    "change omega by the index ratio itself, of order 0.5 relative, "
                    "thirteen decades outside"
                ),
            ),
            Tolerance(
                metric="lagrange_invariant_relative_drift",
                threshold=1e-10,
                basis=(
                    "UNCHANGED AT 1e-10 AND NO LONGER GATING, and the reason is a "
                    "derivation rather than a convenience. Its previously declared "
                    "basis was CONSERVATION_LAW -- 'conservation to float64 round-"
                    "off accumulated over at most twelve surfaces' -- and that "
                    "basis does not apply to this metric. The conservation law it "
                    "named holds for the paraxial transfer matrix and for tangent "
                    "vectors; this metric evaluates the bilinear form on two REAL "
                    "rays at finite separation, where the leading residual is "
                    "aberration and is many orders above round-off. A tolerance "
                    "whose basis cannot bound its metric is invalid whichever side "
                    "of the threshold the measurement lands on, and no threshold "
                    "derived from float64 could ever have been the right one here. "
                    "The threshold is therefore left exactly where it was rather "
                    "than widened to fit the 1.33e-7 that is measured, the metric "
                    "is retained and reported because the paraxial domain it "
                    "characterizes is real, and the conservation claim it was "
                    "standing in for now has its own valid gate on "
                    "symplectic_invariant_relative_residual. Re-deriving a "
                    "defensible APPROXIMATION-error bound for the finite form -- "
                    "third-order aberration theory for this specific stack -- would "
                    "make this gateable again and is not done here"
                ),
                basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
                may_gate=False,
                rejects=(
                    "nothing it can be trusted to reject, which is the finding. At "
                    "1.33e-7 it is four decades below a missing index factor and "
                    "two decades above its own aberration floor, so it separates "
                    "gross prescription errors from correct ones and cannot "
                    "separate a correct implementation from a subtly wrong one"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="non-symplectic-momentum-scale",
                description=(
                    "scale the image-plane canonical momenta by 1 + 1e-6 with the "
                    "positions untouched, so the tangent map's determinant is no "
                    "longer one"
                ),
                mutation="p_image *= 1 + 1e-6",
                target_metric="symplectic_invariant_relative_residual",
            ),
            NegativeControl(
                control_id="non-symplectic-position-scale",
                description=(
                    "the conjugate half of the same test: scale the image-plane "
                    "positions by 1 + 1e-6 with the momenta untouched. Declared "
                    "separately because omega is bilinear and a metric sensitive "
                    "to one factor is not automatically sensitive to the other"
                ),
                mutation="q_image *= 1 + 1e-6",
                target_metric="symplectic_invariant_relative_residual",
            ),
            NegativeControl(
                control_id="degenerate-tangent-pair",
                description=(
                    "draw BOTH secants from the pupil fan instead of one from the "
                    "pupil and one from the field, so the two tangent vectors are "
                    "parallel in the limit. This is the construction error that "
                    "produced CHE-106's 'the differential ratio converges to "
                    "1 + 7.1e-3 and does NOT approach 1', and it is declared as a "
                    "control because the oracle's requirement of two LINEARLY "
                    "INDEPENDENT directions has to be executable rather than "
                    "advisory"
                ),
                mutation=(
                    "replace the field secant by a second pupil secant at twice "
                    "the separation"
                ),
                target_metric="symplectic_invariant_relative_residual",
            ),
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
            status=GateStatus.MET,
            metric="symplectic_invariant_relative_residual",
            observed=1.204e-15,
            evidence=(
                "benchmarks/instances/b1_ray.py",
                "benchmarks/instances/records/B1-RAY-LAGRANGE-01.json",
                "tests/test_b1_ray_instances.py::test_the_symplectic_invariant_closes_to_roundoff",
                "tests/test_b1_ray_instances.py::test_the_symplectic_residual_is_second_order_in_the_separation",
                "tests/test_b1_ray_instances.py::test_both_non_symplectic_controls_fire",
                "tests/test_b1_ray_instances.py::test_a_degenerate_tangent_pair_carries_no_invariant",
                "tests/test_b1_ray_instances.py::test_the_finite_ray_drift_is_reported_and_does_not_gate",
            ),
            note=(
                "MET at 1.204e-15 against a 1e-13 gate, after the ORACLE was "
                "corrected. The tolerance was not widened; it was re-derived three "
                "decades TIGHTER than the 1e-10 that preceded it.\n\n"
                "What was wrong. This family previously measured the two-ray "
                "bilinear form p_a.q_b - p_b.q_a on two finite real rays and gated "
                "it at 1e-10 on a basis of 'conservation to float64 round-off'. It "
                "read 1.328629e-07, 1329x over. The diagnosis recorded at the time "
                "was already half right -- that form is preserved by a LINEAR "
                "symplectic map and real refraction at a curved surface is "
                "symplectic but not linear -- but it drew the wrong conclusion: "
                "that no finite-ray evaluation can do better, on the strength of a "
                "measurement that 'the differential ratio between two rays of one "
                "fan converges to 1 + 7.1e-3 and does NOT approach 1 as their "
                "separation shrinks'.\n\n"
                "That measurement was of a DEGENERATE pair. Two rays of one pupil "
                "fan differ only in pupil coordinate, so as the separation shrinks "
                "both secants approach the same tangent direction; omega between "
                "parallel vectors is zero, and on the object side of a collimated "
                "bundle it is IDENTICALLY zero because every ray shares one "
                "direction and dp = 0 for both. The ratio was a 0/0. It is now a "
                "declared negative control, and it reads omega_object = 0.0 exactly "
                "at all six rungs while omega_image is nonzero -- an infinite "
                "relative residual, which is the control firing.\n\n"
                "What is right. With q the transverse position on a plane of "
                "constant z and p = n(L, M) the index-weighted direction cosines -- "
                "not the ray slopes M/N -- the ray map is the flow of a Hamiltonian "
                "system in z, so its TANGENT map is symplectic at every point "
                "however nonlinear the map is, and omega(v_a, v_b) is conserved "
                "exactly for real rays at any aperture and any field. Two symmetric "
                "secants about a common base ray, one across the pupil at fixed "
                "field and one across the field at fixed pupil, approximate two "
                "linearly independent tangent vectors to O(eps^2).\n\n"
                "The measurement, over six halvings on the three-surface stack at "
                "the canonical instance:\n"
                "  eps/eps_0   raw residual     eps^2 removed    eps^4 removed\n"
                "  1           5.61114e-04      -1.0802e-07      +9.6727e-12\n"
                "  1/2         1.40197e-04      -6.7425e-09      +1.4974e-13\n"
                "  1/4         3.50443e-05      -4.2126e-10      +2.6448e-15\n"
                "  1/8         8.76076e-06      -2.6326e-11      -1.2040e-15\n"
                "  1/16        2.19017e-06      -1.6465e-12\n"
                "  1/32        5.47541e-07\n"
                "The raw column fits an exponent of 2.00018 with r^2 = 0.99999999 "
                "over six points, which is the finite-difference truncation "
                "signature and not a physical residual: a physical one would not "
                "have an integer exponent in the SEPARATION. The eps^2 column falls "
                "by 16x per halving, which is the O(eps^4) remainder behaving as "
                "predicted. The eps^4 column reaches 1.204e-15, and the next two "
                "columns sit at 1.27e-15 and 3.2e-16 -- the float64 floor, 5.4 eps. "
                "The invariant closes to round-off.\n\n"
                "So the solver was not the defect and neither was the tolerance's "
                "value. The defect was that a differential invariant was being "
                "evaluated on finite-separated rays, and the tolerance's declared "
                "BASIS -- a conservation law -- could not bound the metric it was "
                "attached to whichever side of 1e-10 the number fell on.\n\n"
                "What is retained. lagrange_invariant_relative_drift still measures "
                "1.328629e-07 at a 0.25-degree field, its threshold is still 1e-10, "
                "and its five-rung field ladder still fits an exponent near 2.5. It "
                "no longer gates, and its basis now says why: the residual is "
                "aberration, so a float64 basis was never the right one for it. The "
                "number is not deleted and not widened.\n\n"
                "What this gate cannot see is declared on the metric and is real: "
                "every valid trace of every valid system is symplectic, so this "
                "says the refraction is a canonical transformation and says nothing "
                "about whether the prescription is the one that was asked for. "
                "Measured: replacing the second element's index 1.62 by 1.0 leaves "
                "the extrapolated residual at 3.5e-15. B1-RAY-EFL, -PLATE and "
                "-SNELL are the families that decide the prescription, and that "
                "division of labour is why the omit-index control stays pointed at "
                "the finite-ray metric, where it does fire."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "generating a random multi-surface system that is neither degenerate nor "
            "aberration-dominated is the work, and it belongs with M9's sampler. "
            "Worth noting for whoever writes it: the corrected oracle does not need "
            "the system to be near-paraxial at all, because the differential "
            "invariant is exact for real rays -- so its admissible domain is much "
            "wider than the two paraxial predicates this family still declares for "
            "the sake of its characterization metric."
        ),
        evidence=(
            "src/registry/prescriptions.py",
            "docs/prescriptions/canonical_optical_systems.md",
        ),
        notes=(
            "A conservation law is the strongest oracle shape available for a "
            "multi-surface system, because it needs no external reference at all: "
            "the system is compared with itself at a different place, not with "
            "another implementation. What CHE-106 learned is that the shape is not "
            "sufficient -- the QUANTITY has to be one the law is actually about. "
            "Three things are called the Lagrange invariant here (paraxial form, "
            "finite-real-ray bilinear form, differential symplectic form) and only "
            "the third is conserved by a real trace; see _symplectic_invariant for "
            "the distinction written out.\n\n"
            "The two paraxial validity predicates bound the CHARACTERIZATION metric "
            "and not the gate. The differential invariant holds outside them, which "
            "is exactly why it can gate at 1e-13 where the finite form could not "
            "defensibly gate at all."
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


#: A three-surface stack, and the family whose gate closes once the invariant is
#: the right one.
#:
#: The gate is the DIFFERENTIAL symplectic invariant, evaluated on two
#: independent tangent vectors reached by symmetric secants and extrapolated to
#: zero separation. It closes at 1.204e-15 against a 1e-13 round-off budget.
#: The finite-real-ray paraxial form is retained as a characterization metric at
#: its original 1.33e-7 with its original 1e-10 threshold, no longer gating,
#: because a conservation-law basis cannot bound an aberration residual. See the
#: gate disposition for the derivation and the six-rung ladder.
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
                "perturbation_scale": 0.5,
                "perturbation_rungs": 6,
                "richardson_levels": 2,
            },
            expected={
                "symplectic_residual": "0 exactly; float64 round-off in practice",
                "drift": "0 in the paraxial limit",
                "known_disposition": (
                    "MET. The differential invariant extrapolates to 1.204e-15 "
                    "against a 1e-13 round-off budget. The finite-real-ray form "
                    "still reads 1.3e-7 and no longer gates: its basis was a "
                    "conservation law and its residual is aberration, so the "
                    "threshold could not bound it either way. Corrected rather "
                    "than accommodated."
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
