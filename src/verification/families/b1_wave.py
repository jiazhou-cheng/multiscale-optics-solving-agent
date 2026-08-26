"""B1 wave-primitive families: Chromatix scalar ASM, and where it stops applying.

CHE-107 (M1.2). Same gap as the ray side: ``L1-WAVE-01`` was archived and nothing
in the active tree is a node-level scientific benchmark for ``M_WAVE_CHROMATIX``.

For a wave solver the validity domain is not a footnote. ``registry/models.yaml``
already warns that sampling and propagation must satisfy the relevant band-limit
conditions and that scalar propagation is not valid for every high-NA task, and
those warnings have had no executable form. ``B1-WAVE-ASM-VALIDITY`` is the first
family in this repository whose entire purpose is behaviour *near* a boundary,
and it is what forces the validity margin to be signed and normalized rather than
a boolean.

The archive records the model of what a validity finding looks like:
``L1-WAVE-01``'s high-NA vectorial case was already ``status: blocked`` on a
defective upstream ``high_na_ff_lens``, where refining only the pupil sampling
moved the focal scale by **10x** while the independent oracle converged to
``2e-14``. Not "the solver is correct", but "the solver is correct *here* and
demonstrably is not *there*".

Two families are new to the repository and worth calling out:

* ``B1-WAVE-FWDBWD`` -- propagate forward then backward and recover the input to
  dtype round-off. The cheapest possible round trip, and it catches a phasor-sign
  error or a 2*pi frequency-grid scale error immediately.
* ``B1-WAVE-TALBOT`` -- revival at ``z_T = 2 d^2 / lambda``. Nothing here tests
  periodic self-imaging, and a revival is a strong independent check on
  propagator phase that no existing probe covers.

On oracles
----------
``verification/asm_oracle.py`` is this repository's own float64 ASM/RS
propagator. Against a **coupler** it shares code and is diagnostic-only; against
**Chromatix** it shares none, so it would be admissible. It is still not used as
a gate below: the project's standing rule is that a custom numerical oracle does
not decide correctness, analytic closed forms are primary, and every gate here
has one available. Where O2 appears it is named as a diagnostic and its
tolerance carries ``may_gate=False``.

On precision
------------
Chromatix is ``complex64``-only, and not as a policy choice:
``ScalarField.__init__`` is ``jnp.asarray(u, dtype=jnp.complex64)``
unconditionally, and ``Field.build`` handed a ``complex128`` array returns
``complex64`` even under ``jax_enable_x64=True``. Every family here therefore
declares ``complex64`` as its execution dtype and its tolerances are sized
against that floor. Whether a ``complex128`` *request* is refused or recorded as
lossy is a contract question and belongs to B0, not here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.analytic import (
    AIRY_FIRST_NULL,
    GAUSSIAN_SPREADING,
    TILTED_BEAM_WALKOFF,
)
from verification.families.predicates import (
    asm_transfer_function_sampling,
    fractional_margin,
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
from verification.status import VerificationStatus

__all__ = [
    "B1_WAVE_AIRY",
    "B1_WAVE_ASM_VALIDITY",
    "B1_WAVE_FWDBWD",
    "B1_WAVE_GAUSS",
    "B1_WAVE_PLANEPHASE",
    "B1_WAVE_TALBOT",
    "B1_WAVE_TILT",
]


# ---------------------------------------------------------------------------
# Shared policy
# ---------------------------------------------------------------------------

WAVE_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    dtypes=frozenset({DType.COMPLEX64}),
    namespaces=frozenset({ArrayNamespace.JAX}),
    max_wall_seconds=120.0,
    max_peak_memory_gib=8.0,
    notes=(
        "complex64 only, and this is a capability fact rather than a choice: "
        "ScalarField.__init__ casts unconditionally, so an FP64 request has nothing "
        "to execute. Every tolerance below is sized against that floor."
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "an angular-spectrum propagation over a declared grid is deterministic; "
        "nothing here draws a sample"
    ),
)

#: The sampling bound every ASM family lives inside. Declared once and shared, so
#: that a family cannot quietly adopt a laxer version of it.
#:
#: Configured against this module's parameter names, which are in micrometres.
#: The bound ``z <= N pitch^2 / lambda`` is dimensionally consistent in any single
#: length unit, so it needs no conversion -- only that all three come from the
#: same one, which is what naming the keys explicitly guarantees.
ASM_SAMPLING = asm_transfer_function_sampling(
    distance_key="distance_um",
    pitch_key="sample_pitch_um",
    grid_key="grid_n",
    wavelength_key="wavelength_um",
)


def nyquist_from_direction_cosine(
    predicate_id: str,
    statement: str,
    direction_cosine: Callable[[Mapping[str, Any]], float],
    *,
    pitch_key: str = "sample_pitch_um",
    wavelength_key: str = "wavelength_um",
) -> ValidityPredicate:
    """``pitch <= lambda / (2 max|d|)`` where the field's own physics sets ``d``.

    ``benchmarks/probes/slice_feasibility.py`` derives this bound from *marginal
    ray angles*, which a wave family does not have. What it does have is the
    largest transverse direction cosine its own field carries -- the beam tilt,
    or a Gaussian's divergence -- so the bound is the same and only its input
    differs. Passing a callable rather than a key is what keeps that derivation
    beside the family it belongs to instead of in a shared parameter nobody owns.
    """

    def margin(params: Mapping[str, Any]) -> float:
        d_max = abs(float(direction_cosine(params)))
        if d_max <= 0.0:
            return math.inf
        limit = float(params[wavelength_key]) / (2.0 * d_max)
        return fractional_margin(float(params[pitch_key]), limit)

    return ValidityPredicate(
        predicate_id=predicate_id,
        statement=statement,
        basis=ValidityBasis.PER_AXIS_NYQUIST,
        margin=margin,
        blind_to=(
            "grid extent -- a pitch fine enough to resolve the angles can still be on "
            "a grid too small to hold the field",
        ),
    )


def _eps32_per_radian_basis(radians_of_phase: str) -> str:
    """The tolerance argument every complex64 propagation here uses.

    The residual against an FP64 reference grows as one float32 epsilon per
    radian of accumulated phase: ``eps32 * 2*pi*z/lambda``. CHE-40's carrier
    removal is what makes the claim scale-independent -- without it the absolute
    carrier dominates and the residual is a statement about ``z``, not about the
    propagator. Measured: 2.5e-5 at z = 40 um and 6.3e-2 at z = 47 mm, which is
    also why the M3 reference singlet was scaled to a tenth.
    """
    return (
        "complex64 accumulates about one float32 epsilon per radian of phase, so the "
        f"admissible residual is eps32 * ({radians_of_phase}). Measured on the pinned "
        "install: 2.5e-5 relative field error at z = 40 um and 6.3e-2 at z = 47 mm "
        "(benchmarks/probes/carrier_phase_representation.py). The threshold is that "
        "bound, not a number chosen to pass."
    )


# ---------------------------------------------------------------------------
# B1-WAVE-GAUSS
# ---------------------------------------------------------------------------

_B1_WAVE_GAUSS = BenchmarkFamily(
        family_id="B1-WAVE-GAUSS",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does a propagated Gaussian beam's 1/e^2 intensity radius follow "
            "w(z) = w0 sqrt(1 + (z/zR)^2)?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "waist_um", "1/e^2 field waist at z = 0", unit="um", domain=(0.5, 500.0)
            ),
            PhysicalParameter(
                "distance_um", "propagation distance", unit="um", domain=(0.0, 1e5)
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 4096), default=512, refines_toward=1
            ),
            NumericalParameter(
                "sample_pitch_um",
                "grid pitch",
                unit="um",
                domain=(0.01, 10.0),
                default=0.25,
                refines_toward=-1,
            ),
            ExecutionParameter("device", "cpu or cuda", domain=("cpu", "cuda"), default="cpu"),
        ),
        validity=(ASM_SAMPLING,),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=GAUSSIAN_SPREADING.statement,
            callable=GAUSSIAN_SPREADING.closed_form,
            reference="verification/analytic.py::GAUSSIAN_SPREADING",
        ),
        metrics=(
            Metric(
                name="gaussian_radius_relative_error",
                description="relative error of the measured 1/e^2 radius",
                unit=None,
                definition=None,  # a measurement of one image against a scalar
                blind_to=(
                    "beam quality: a beam that has grown wings while keeping its second "
                    "moment passes",
                    "the transverse position of the beam -- a radius says nothing about "
                    "where the beam is, which is what B1-WAVE-TILT is for",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="gaussian_radius_relative_error",
                threshold=2e-2,
                basis=GAUSSIAN_SPREADING.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=GAUSSIAN_SPREADING.rejects,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="unpropagated-waist",
                description="measure the input field instead of the propagated one",
                mutation="return w0 in place of w(z)",
                target_metric="gaussian_radius_relative_error",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="gaussian_radius_relative_error",
            observed=1.82072e-4,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_gaussian_reproduces_its_inherited_number",
                "tests/test_b1_wave_instances.py::test_a_met_gate_is_inside_its_own_validity_domain",
            ),
            note=(
                "Re-run through the graph node: 1.82e-4 against a 2e-2 gate, "
                "reproducing the inherited 1.8e-4 to the third significant figure. That "
                "agreement is the useful outcome -- the historical number was right and "
                "is now reproducible from code in the tree rather than from a report.\n\n"
                "The grid moved from 512 to 1024 and the physics did not, because the "
                "512-grid instance sat OUTSIDE this family's own validity predicate: "
                "z <= N pitch^2 / lambda is 60.15 um at 512 and the propagation is "
                "100 um. A metric inside its tolerance and a status of out_of_validity "
                "is not a pass; it is a measurement whose validity claim contradicts "
                "itself, and it is the kind of contradiction only running the family "
                "surfaces."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the oracle is arithmetic, but a drawn (waist, distance, pitch, grid) point "
            "has to land inside the ASM sampling bound to be a fair test, and choosing "
            "the grid from the physics is the sampler's job in M9."
        ),
        evidence=(
            "src/verification/analytic.py",
            "tests/test_preserved_evidence.py::test_the_gaussian_oracle_reproduces_its_measured_agreement",
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-AIRY
# ---------------------------------------------------------------------------

#: The A1-verified configuration: w0 = 5 um, z = 100 um, lambda = 0.532 um.
#: Kept exactly, because the recorded 6.040167 um second-moment radius against
#: 6.039084 analytic is what makes this an inherited-and-now-re-run measurement
#: rather than a new one.
B1_WAVE_GAUSS = register(
    _B1_WAVE_GAUSS.with_instances(
        _B1_WAVE_GAUSS.instantiate(
            "B1-WAVE-GAUSS-01",
            {
                "waist_um": 5.0,
                "distance_um": 100.0,
                "wavelength_um": 0.532,
                # 1024 rather than 512, and the reason is the family's own
                # predicate: z <= N pitch^2 / lambda is 60.15 um at 512 and
                # 120.30 um at 1024, so a 512-grid instance at z = 100 um would
                # be declared OUTSIDE its own validity domain while meeting its
                # gate to 1.8e-4. That combination is not a pass, it is a
                # measurement whose validity claim contradicts itself.
                "grid_n": 1024,
                "sample_pitch_um": 0.25,
                "device": "cpu",
            },
            expected={
                "radius_um": 6.039084,
                "why": (
                    "w0 sqrt(1 + (z/zR)^2) is exact for a paraxial Gaussian, and the "
                    "second moment of the propagated intensity is the estimator. The "
                    "2e-2 tolerance rejects the UNPROPAGATED waist, 5.0 um, which is "
                    "17% low and is what a run that propagated zero distance returns."
                ),
            },
        ),
    )
)


_B1_WAVE_AIRY = BenchmarkFamily(
        family_id="B1-WAVE-AIRY",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does the first dark ring of a focused circular aperture land at "
            "0.61 lambda / NA?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "numerical_aperture", "image-space NA", domain=(0.005, 0.6)
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 8192), default=1024, refines_toward=1
            ),
            NumericalParameter(
                "focal_plane_pitch_um",
                "focal-plane sample pitch. THE numerical parameter of this family: at "
                "0.83 um the null falls between samples and the measured radius is 2.3% "
                "high, which is a sampling limit and not a physics error",
                unit="um",
                domain=(0.01, 5.0),
                default=0.83,
                refines_toward=-1,
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="AIRY_CORE_SAMPLING",
                statement=(
                    "the focal-plane pitch puts at least two samples across the Airy "
                    "radius, so a first-null measurement has something to interpolate "
                    "between"
                ),
                basis=ValidityBasis.PER_AXIS_NYQUIST,
                margin=lambda p: fractional_margin(
                    float(p["focal_plane_pitch_um"]),
                    0.61 * float(p["wavelength_um"]) / float(p["numerical_aperture"]) / 2.0,
                ),
                blind_to=(
                    "convergence. Two samples per Airy radius is the floor at which the "
                    "measurement is DEFINED, not the point at which it has converged -- "
                    "CHE-103 measured the frozen configuration at 2.44 px and found it "
                    "not converged for radius-like quantities",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=AIRY_FIRST_NULL.statement,
            callable=AIRY_FIRST_NULL.closed_form,
            reference="verification/analytic.py::AIRY_FIRST_NULL",
        ),
        metrics=(
            Metric(
                name="airy_first_null_relative_error",
                description="relative error of the measured first-null radius",
                unit=None,
                blind_to=(
                    "everything outside the first ring -- the Strehl, the ring "
                    "structure, and any energy in the wings",
                    "an aperture that is not circular: an ellipse with the right mean "
                    "radius would pass a radially-averaged measurement",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="airy_first_null_relative_error",
                threshold=5e-2,
                basis=AIRY_FIRST_NULL.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=AIRY_FIRST_NULL.rejects,
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="diameter-for-radius",
                description="report 1.22 lambda / NA, the diameter formula used as a radius",
                mutation="multiply the oracle by 2",
                target_metric="airy_first_null_relative_error",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
        ),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="airy_first_null_relative_error",
            observed=1.75889e-2,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_airy_null_is_bias_corrected_rather_than_trusted",
            ),
            note=(
                "1.76e-2 against a 5e-2 gate, on a grid that puts 10.8 samples across "
                "the Airy radius rather than the frozen configuration's 2.44 -- so the "
                "sampling caveat CHE-103 raised is answered by the geometry rather than "
                "carried forward as a caveat.\n\n"
                "The estimator's own bias is measured rather than assumed away: the same "
                "first-null estimator is run over the ANALYTIC Airy pattern on the same "
                "grid, and the bias-cancelled ratio is reported beside the raw error. "
                "Comparing a measured null straight to 0.61 lambda/NA at coarse pitch "
                "measures the estimator and not the solver, which is what CHE-103's "
                "11.9%-at-2.44-px finding established.\n\n"
                "The geometry is derived from the declared NA rather than declared "
                "separately -- the aperture radius is fixed and the focal length follows "
                "as a/NA -- so NA is the one physical knob and the closed form depends "
                "on nothing else. The focus is checked to sit inside the ASM sampling "
                "limit (300 um against 346 um) rather than assumed to."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note="as B1-WAVE-GAUSS: the grid has to be chosen from the physics.",
        evidence=(
            "src/verification/analytic.py",
            "benchmarks/probes/records/m3_first_null_grid_convergence.json",
            "tests/test_preserved_evidence.py::test_the_airy_oracle_reproduces_its_measured_agreement",
        ),
        notes=(
            "The refinement dimension is the focal-plane pitch, not the pupil sampling. "
            "CHE-103 found that refining the pupil alone moves the measured ratio while "
            "the physics does not, which is exactly the case the NUMERICAL/PHYSICAL "
            "split exists to make expressible."
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-TILT
# ---------------------------------------------------------------------------

#: A circular pupil with a converging lens phase, propagated to its focus.
#:
#: The geometry is derived from the declared NA rather than declared separately:
#: the aperture radius is fixed at 30 um and the focal length follows as a/NA, so
#: NA is the one physical knob and the closed form 0.61 lambda / NA depends on
#: nothing else. Grid and pitch are chosen so the focal plane resolves the Airy
#: core -- 10.8 samples per Airy radius -- and so the focus stays inside the ASM
#: transfer function's own sampling limit, N pitch^2 / lambda = 346 um against a
#: 300 um focal length. Both are checked at run time rather than asserted here.
B1_WAVE_AIRY = register(
    _B1_WAVE_AIRY.with_instances(
        _B1_WAVE_AIRY.instantiate(
            "B1-WAVE-AIRY-01",
            {
                "numerical_aperture": 0.1,
                "wavelength_um": 0.532,
                "grid_n": 2048,
                "focal_plane_pitch_um": 0.3,
            },
            expected={
                "first_null_um": 0.61 * 0.532 / 0.1,
                "why": (
                    "0.61 lambda / NA is the exact first zero of the focal-plane "
                    "intensity. A first-null estimator is biased at finite sampling, so "
                    "the same estimator is also run over the analytic pattern on the "
                    "same grid and the bias-cancelled ratio is reported beside the raw "
                    "error -- the cancellation is visible rather than assumed."
                ),
            },
        ),
    )
)


_B1_WAVE_TILT = BenchmarkFamily(
        family_id="B1-WAVE-TILT",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does a tilted collimated beam walk off by z tan(theta), with the right "
            "SIGN, under a kykx-parameterized propagation?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.CONVENTION,
        parameters=(
            PhysicalParameter(
                "tilt_rad", "beam tilt from the optical axis", unit="rad", domain=(-0.3, 0.3)
            ),
            PhysicalParameter(
                "distance_um", "propagation distance", unit="um", domain=(0.0, 1e4)
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 4096), default=512, refines_toward=1
            ),
            NumericalParameter(
                "sample_pitch_um", "grid pitch", unit="um", domain=(0.01, 10.0), default=0.5,
                refines_toward=-1,
            ),
            RepresentationParameter(
                "tilt_encoding",
                "how the tilt is written onto the field: as an explicit exp(2 pi i f x) "
                "factor, or through the kykx argument. The SAME physics, and the "
                "repository has a measured hazard saying the two do not agree",
                domain=("explicit_phase_ramp", "kykx_argument"),
                default="explicit_phase_ramp",
            ),
        ),
        validity=(
            ASM_SAMPLING,
            nyquist_from_direction_cosine(
                "TILT_NYQUIST",
                "the grid pitch resolves the tilted beam's own transverse frequency",
                lambda p: math.sin(float(p["tilt_rad"])),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=TILTED_BEAM_WALKOFF.statement,
            callable=TILTED_BEAM_WALKOFF.closed_form,
            reference="verification/analytic.py::TILTED_BEAM_WALKOFF",
        ),
        metrics=(
            Metric(
                name="tilt_centroid_signed_relative_error",
                description=(
                    "signed relative error of the intensity centroid displacement "
                    "against z tan(theta)"
                ),
                unit=None,
                blind_to=(
                    "beam shape entirely: a centroid is a first moment, so a beam that "
                    "has sheared or split symmetrically about the right point passes",
                    "z sin(theta) versus z tan(theta) at small angles -- 0.4% apart at "
                    "5 degrees, inside this tolerance, and the oracle says so",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="tilt_centroid_signed_relative_error",
                threshold=2e-2,
                basis=TILTED_BEAM_WALKOFF.verified_against_pinned_solver,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "the two halves of the measured kykx hazard: a 2*pi factor (6.28x, "
                    "300 tolerances away) and a sign inversion (2.0 relative). "
                    "B0-UNITS-02 is the same hazard as a contract benchmark"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="kykx-two-pi",
                description=(
                    "the measured trap: read kykx as radians per length where the "
                    "function wants cycles per length"
                ),
                mutation="divide the spatial frequency by 2*pi before passing it",
                target_metric="tilt_centroid_signed_relative_error",
            ),
            NegativeControl(
                control_id="kykx-sign",
                description=(
                    "the other half of the same trap: the displacement runs opposite in "
                    "sign to the parameter"
                ),
                mutation="negate the spatial frequency",
                target_metric="tilt_centroid_signed_relative_error",
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="tilt_centroid_signed_relative_error",
            observed=9.92987e-5,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_tilt_walkoff_carries_its_sign",
                "tests/test_b1_wave_instances.py::test_the_kykx_unit_hazard_is_measured_where_it_belongs",
            ),
            note=(
                "9.93e-5 against a 2e-2 gate on the explicit phase-ramp encoding, with "
                "the SIGN gated: the measured centroid is +17.4960 um against "
                "+17.4977 analytic, and the sign-flip control is a 2x relative error "
                "rather than a marginal one.\n\n"
                "The pitch moved from 0.25 to 0.5 um so the instance satisfies its own "
                "sampling predicate -- N pitch^2 / lambda is 481 um at 0.5 and 120 um at "
                "0.25, against a 200 um propagation.\n\n"
                "The kykx_argument encoding is still not measured through THIS family, "
                "and that is now a decision rather than a gap: the hazard there is the "
                "parameter's UNIT rather than the physics, and B0-UNITS-02 measures both "
                "call sites -- plane_wave handed cycles-per-length is 2*pi too small with "
                "the sign preserved, and asm_propagate's kykx displaces opposite to its "
                "parameter. Measuring both here would conflate 'the physics is right' "
                "with 'the caller read the right unit'."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "the interesting variation here is the tilt_encoding REPRESENTATION "
            "parameter rather than the physics, and both of its values need to stay "
            "comparable at one physical point for the comparison to mean anything."
        ),
        evidence=(
            "src/verification/analytic.py",
            "src/verification/hazards.py",
            "tests/test_preserved_evidence.py::test_the_kykx_hazard_keeps_both_the_factor_and_the_sign",
        ),
        notes=(
            "tilt_encoding is a RepresentationParameter and the whole point of the "
            "family: it should not change the answer, the measured hazard says it does "
            "by 2*pi and a sign, and the schema's own rule is that a family whose "
            "representation parameter moves the oracle value has found a defect."
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-PLANEPHASE
# ---------------------------------------------------------------------------


def _plane_wave_phase_advance(params: Mapping[str, Any]) -> float:
    """``k_z z`` for a plane wave at the declared transverse frequency, in radians.

    Pins the spatial phasor sign and the ``k_z`` construction at once: an
    ``exp(-ikz)`` convention gives the negative of this, and a frequency grid off
    by ``2*pi`` gives a ``k_z`` that is wrong except exactly on axis.
    """
    lam = float(params["wavelength_um"])
    z = float(params["distance_um"])
    n = float(params["medium_index"])
    fx = float(params["transverse_frequency_per_um"])
    k = 2.0 * math.pi * n / lam
    kt = 2.0 * math.pi * fx
    if kt >= k:
        raise ValueError(
            f"evanescent: transverse frequency {fx} /um is past the light cone at "
            f"lambda = {lam} um, n = {n}. Outside this family's validity domain."
        )
    return math.sqrt(k * k - kt * kt) * z


#: A 5-degree tilt over 200 um: +17.4977 um of walk-off, sign included.
#:
#: The tilt is applied as an EXPLICIT phase ramp on the input field, which is the
#: encoding that goes through the graph node. The `kykx_argument` encoding is the
#: other declared value of `tilt_encoding` and is measured separately by
#: B0-UNITS-02, because the hazard there is the parameter's UNIT rather than the
#: physics: `kykx` means cycles per length on `asm_propagate` and radians per
#: length on `plane_wave`, and the resulting displacement runs opposite in sign to
#: the parameter on the propagator. Splitting them keeps this family about the
#: walk-off and that one about the convention.
B1_WAVE_TILT = register(
    _B1_WAVE_TILT.with_instances(
        _B1_WAVE_TILT.instantiate(
            "B1-WAVE-TILT-01",
            {
                "tilt_rad": 0.08726646259971647,
                "distance_um": 200.0,
                "wavelength_um": 0.532,
                # pitch 0.5 rather than 0.25, so N pitch^2 / lambda is 481 um
                # against a 200 um propagation. At pitch 0.25 the same grid
                # gives 120 um and the instance would sit outside its own
                # validity domain.
                "grid_n": 1024,
                "sample_pitch_um": 0.5,
                "tilt_encoding": "explicit_phase_ramp",
            },
            expected={
                "walkoff_um": 17.497732705184802,
                "sign": "positive: the beam walks toward +y for a +y tilt",
                "why": (
                    "z tan(theta) is exact geometry for a collimated beam and the SIGN "
                    "is part of the claim. The 2e-2 tolerance does not separate "
                    "z sin(theta) from z tan(theta) -- they differ by 0.4% at 5 degrees "
                    "-- and the family says so rather than claiming a separation it "
                    "does not have."
                ),
            },
        ),
    )
)


_B1_WAVE_PLANEPHASE = BenchmarkFamily(
        family_id="B1-WAVE-PLANEPHASE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does propagating a plane wave a distance z advance its phase by exactly "
            "k_z z, with the sign the repository's phasor convention declares?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.CONVENTION,
        parameters=(
            PhysicalParameter(
                "transverse_frequency_per_um",
                "transverse spatial frequency of the plane wave; zero is on-axis and "
                "cannot separate a frequency-grid scale error",
                unit="1/um",
                domain=(0.0, 1.5),
            ),
            PhysicalParameter(
                "distance_um", "propagation distance", unit="um", domain=(0.0, 1e4)
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            PhysicalParameter(
                "medium_index", "refractive index of the medium", domain=(1.0, 2.0), default=1.0
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 2048), default=256, refines_toward=1
            ),
            NumericalParameter(
                "sample_pitch_um", "grid pitch", unit="um", domain=(0.01, 10.0), default=0.25,
                refines_toward=-1,
            ),
        ),
        validity=(
            ASM_SAMPLING,
            ValidityPredicate(
                predicate_id="PROPAGATING_BAND",
                statement=(
                    "the transverse frequency stays inside the light cone, so the wave "
                    "propagates rather than decaying"
                ),
                basis=ValidityBasis.ASM_SAMPLING,
                margin=lambda p: fractional_margin(
                    float(p["transverse_frequency_per_um"]),
                    float(p["medium_index"]) / float(p["wavelength_um"]),
                ),
                blind_to=(
                    "how much power the grid puts near the cone -- a field mostly inside "
                    "it can still lose a measurable fraction across the edge",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "k_z z with k_z = sqrt((2 pi n / lambda)^2 - (2 pi f)^2), exact for a "
                "plane wave in a homogeneous medium"
            ),
            callable=_plane_wave_phase_advance,
            reference="the angular spectrum's own definition; no approximation",
        ),
        metrics=(
            Metric(
                name="plane_wave_phase_residual_rad",
                description=(
                    "absolute residual between the measured phase advance and k_z z, "
                    "unwrapped, in radians"
                ),
                unit="rad",
                blind_to=(
                    "amplitude entirely: a propagation that halved the field while "
                    "advancing the phase correctly passes",
                    "a residual that is an exact multiple of 2*pi, if the unwrapping is "
                    "done wrong -- which is why the family sweeps z rather than testing "
                    "one distance",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="plane_wave_phase_residual_rad",
                threshold=1e-2,
                basis=_eps32_per_radian_basis("2 pi n z / lambda"),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects=(
                    "an exp(-ikz) convention, which gives -k_z z and is 2 k_z z away; "
                    "and a frequency grid off by 2*pi, which is exact on axis and wrong "
                    "everywhere else -- hence the nonzero transverse frequency"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="phasor-sign-flip",
                description="use exp(-i k_z z) instead of exp(+i k_z z)",
                mutation="conjugate the transfer function",
                target_metric="plane_wave_phase_residual_rad",
            ),
            NegativeControl(
                control_id="frequency-grid-two-pi",
                description=(
                    "build the frequency grid in radians per length where cycles per "
                    "length is required"
                ),
                mutation="multiply fftfreq output by 2*pi before forming k_z",
                target_metric="plane_wave_phase_residual_rad",
            ),
        ),
        failure_semantics=(VerificationStatus.OUT_OF_VALIDITY,),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="plane_wave_phase_residual_rad",
            observed=4.89889e-6,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_plane_wave_phase_is_gated_off_axis",
            ),
            note=(
                "4.90e-6 rad against a 1e-2 gate. The phasor sign was documented and "
                "asserted nowhere as a gate on a propagated plane wave; it is now.\n\n"
                "The measured quantity is the phase advance RELATIVE to an on-axis plane "
                "wave propagated the same distance, which is (k_z - k) z. The absolute "
                "advance k_z z is 236 rad and wraps, so a residual against it would be "
                "a statement about the wrap. The instance declares a nonzero transverse "
                "frequency for the same reason the ray family declares a nonzero field "
                "angle: on axis k_z = k exactly and a frequency-grid scale error is "
                "invisible.\n\n"
                "Both controls fire, and the 2*pi one is worth reading: 2*pi too large "
                "puts k_t past the light cone entirely, so the wrong answer is not a "
                "shifted phase but an evanescent mode with no propagating advance at "
                "all."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "the oracle is one line and the family is genuinely generative; the sampler "
            "still needs to pick a grid that resolves the drawn frequency, which is M9."
        ),
        evidence=(
            "knowledge/solvers/chromatix/conventions.md",
            "benchmarks/probes/records/chromatix/propagation_probe.json",
        ),
        notes=(
            "The transverse frequency must be NONZERO for the gate to mean anything. On "
            "axis k_z = k and a 2*pi frequency-grid error is invisible -- the same shape "
            "of blind spot as B1-RAY-OFFAXIS-OPL's on-axis case."
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-FWDBWD
# ---------------------------------------------------------------------------

#: A plane wave at a nonzero transverse frequency, so k_z != k.
#:
#: On axis ``k_z = k`` and a frequency-grid scale error is exactly invisible;
#: off axis it is not. The instance therefore declares a nonzero
#: `transverse_frequency_per_um`, and the measured quantity is the phase advance
#: RELATIVE to an on-axis plane wave propagated the same distance -- which is
#: ``(k_z - k) z``, unambiguous modulo nothing, where the absolute advance
#: ``k_z z`` is thousands of radians and wraps.
B1_WAVE_PLANEPHASE = register(
    _B1_WAVE_PLANEPHASE.with_instances(
        _B1_WAVE_PLANEPHASE.instantiate(
            "B1-WAVE-PLANEPHASE-01",
            {
                "transverse_frequency_per_um": 0.09375,
                "distance_um": 20.0,
                "wavelength_um": 0.532,
                "medium_index": 1.0,
                "grid_n": 256,
                "sample_pitch_um": 0.25,
            },
            expected={
                "why": (
                    "exp(+i k_z z) exactly, with k_z = sqrt(k^2 - k_t^2). The frequency "
                    "is an exact grid frequency of a 256-sample 0.25 um grid "
                    "(3/(256*0.25) = 0.046875 cycles/um times 2), so the plane wave is "
                    "periodic on the grid and nothing wraps. A phasor-sign flip negates "
                    "the advance and a 2*pi frequency-grid error moves k_z off the light "
                    "cone entirely."
                ),
            },
        ),
    )
)


_B1_WAVE_FWDBWD = BenchmarkFamily(
        family_id="B1-WAVE-FWDBWD",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does propagating a field forward by z and then backward by z return the "
            "input to the dtype's round-off floor?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.ROUND_TRIP,
        parameters=(
            PhysicalParameter(
                "distance_um", "the distance travelled each way", unit="um", domain=(0.0, 1e4)
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            PhysicalParameter(
                "aperture_fill_fraction",
                "how much of the grid the input field occupies; the round trip is only "
                "exact for a field that does not wrap",
                domain=(0.05, 0.9),
                default=0.4,
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 2048), default=512, refines_toward=1
            ),
            NumericalParameter(
                "sample_pitch_um", "grid pitch", unit="um", domain=(0.01, 10.0), default=0.25,
                refines_toward=-1,
            ),
        ),
        validity=(ASM_SAMPLING,),
        oracle=FamilyOracle(
            kind=Oracle.CONSERVATION_LAW,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the input field itself. Free-space propagation is unitary on the "
                "propagating band, so forward-then-backward is the identity, and the "
                "reference is the input rather than another computation"
            ),
            callable=None,
            reference="unitarity of the angular-spectrum transfer function",
        ),
        metrics=(
            Metric(
                name="round_trip_relative_l2",
                definition="relative_rms",
                description=(
                    "||U_returned - U_input||_2 / ||U_input||_2 over the whole grid"
                ),
                unit=None,
                blind_to=(
                    "any error that is its own inverse. A transfer function with the "
                    "WRONG magnitude of k_z still round-trips exactly, because the "
                    "backward pass undoes whatever the forward pass did. This metric "
                    "catches sign and scale errors in the FREQUENCY GRID, not in k_z",
                    "evanescent content, which is removed on the way out and cannot "
                    "come back",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="round_trip_relative_l2",
                threshold=1e-5,
                basis=_eps32_per_radian_basis("2 * 2 pi z / lambda, two passes"),
                basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
                may_gate=True,
                rejects=(
                    "a frequency grid whose fftshift is applied on one pass and not the "
                    "other, which returns a field translated by half the grid -- of "
                    "order 1 relative"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="asymmetric-fftshift",
                description="shift the frequency grid on the forward pass only",
                mutation="apply fftshift to the transfer function in one direction",
                target_metric="round_trip_relative_l2",
            ),
            NegativeControl(
                control_id="wrapped-aperture",
                description=(
                    "fill the grid so the propagated field wraps, then round-trip it. "
                    "This must fail, and the family declares it OUT_OF_VALIDITY rather "
                    "than as a defect -- the round trip is only exact where the field "
                    "does not leave the window"
                ),
                mutation="set aperture_fill_fraction to 0.95 at a large z",
                target_metric="round_trip_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="round_trip_relative_l2",
            observed=2.75199e-7,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_round_trip_returns_the_input_and_can_be_made_to_fail",
                "tests/test_b1_wave_instances.py::test_the_round_trip_declares_what_it_cannot_see",
            ),
            note=(
                "2.75e-7 against a 1e-5 gate, and it lands BELOW the single-pass "
                "complex64 floor of 1.4e-4 -- one float32 epsilon per radian of the "
                "1181 rad the leg accumulates -- because the two legs' phase errors are "
                "correlated. That is the cheapest check in M1 and it now runs.\n\n"
                "Both controls are asymmetries the round trip cannot undo, and both are "
                "SMALL on purpose: one sample of lateral shift between the legs, and an "
                "aperture wide enough that energy leaves the window. A control that "
                "moved the field by an order of magnitude would say nothing about a "
                "1e-5 gate.\n\n"
                "What this family cannot see is stated rather than left implicit: a "
                "convention error SHARED by the two legs cancels exactly, so a phasor "
                "sign flipped in both directions returns the input. B1-WAVE-PLANEPHASE "
                "is the family that sees those, which is why both exist."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "there is no oracle to construct at all -- the reference is the input -- so "
            "this family is the most generative one here. The sampler is still M9's."
        ),
        evidence=(
            "src/solvers/chromatix/carrier_removed_asm.py",
            "benchmarks/probes/records/chromatix/propagation_probe.json",
        ),
        notes=(
            "What this is BLIND to matters as much as what it catches, and the metric "
            "says so: an error that the backward pass undoes is invisible here by "
            "construction. B1-WAVE-PLANEPHASE is the family that sees those."
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-TALBOT
# ---------------------------------------------------------------------------


def _talbot_distance_um(params: Mapping[str, Any]) -> float:
    """``z_T = 2 d^2 / lambda``, the revival distance of a periodic field."""
    d = float(params["period_um"])
    return 2.0 * d * d / float(params["wavelength_um"])


#: Forward then backward, and the input comes back.
#:
#: The cheapest round trip available, and it catches a phasor-sign error or a
#: 2*pi frequency-grid scale error immediately: the transfer function is
#: unit-modulus, so ``H(-z) = conj(H(z))`` and the product is exactly 1 for every
#: propagating mode. What it CANNOT catch is a shared convention error -- one that
#: cancels between the two legs -- which is why the family carries a
#: deliberately asymmetric control.
B1_WAVE_FWDBWD = register(
    _B1_WAVE_FWDBWD.with_instances(
        _B1_WAVE_FWDBWD.instantiate(
            "B1-WAVE-FWDBWD-01",
            {
                "distance_um": 100.0,
                "wavelength_um": 0.532,
                "aperture_fill_fraction": 0.4,
                # 120.30 um of sampling limit against a 100 um leg.
                "grid_n": 1024,
                "sample_pitch_um": 0.25,
            },
            expected={
                "residual": "dtype round-off",
                "why": (
                    "unitarity of the angular-spectrum transfer function. The 1e-5 "
                    "tolerance is the complex64 floor: the field is held in complex64 "
                    "throughout because Chromatix has no other precision, and one "
                    "float32 epsilon per radian of accumulated phase over 2*pi*100/0.532 "
                    "= 1181 rad is 1.4e-4 -- so a residual at 1e-5 is BELOW the "
                    "single-pass floor, which the round trip achieves because the two "
                    "legs' phase errors are correlated."
                ),
            },
        ),
    )
)


_B1_WAVE_TALBOT = BenchmarkFamily(
        family_id="B1-WAVE-TALBOT",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "does a periodic field revive itself at the Talbot distance "
            "z_T = 2 d^2 / lambda?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "period_um",
                "grating period; the revival distance goes as its square, so a small "
                "error here is a large error in z_T",
                unit="um",
                domain=(1.0, 200.0),
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            PhysicalParameter(
                "duty_cycle", "open fraction of the binary grating", domain=(0.1, 0.9),
                default=0.5,
            ),
            PhysicalParameter(
                "talbot_order",
                "which revival to test. 1 is the first full revival; a half-integer "
                "order is the shifted image and is a different claim",
                domain=(1, 4),
                default=1,
            ),
            NumericalParameter(
                "periods_across_grid",
                "how many periods the window holds. Fewer than a few and the edges "
                "dominate the revival",
                domain=(4, 128),
                default=32,
                refines_toward=1,
            ),
            NumericalParameter(
                "samples_per_period",
                "sampling of one period. Deliberately NOT declared as a refinement "
                "direction, and the sign of that is this family's main finding: the "
                "grid admits diffraction orders up to m_max = samples_per_period / 2, "
                "and order m dephases from the paraxial revival by "
                "(pi/2) m^4 (lambda/d)^2, so raising this DEGRADES the metric as its "
                "fourth power. 8 samples -> m_max 4 -> 2.2e-4; 32 samples -> m_max 16 "
                "-> 2.4e-1. The default is the measured-good value, because an "
                "instance that omits this parameter inherits it",
                domain=(4, 256),
                default=8,
            ),
        ),
        validity=(
            ValidityPredicate(
                predicate_id="TALBOT_TF_SAMPLING",
                statement=(
                    "the revival distance stays inside the angular-spectrum transfer "
                    "function's own sampling limit, z_T <= N pitch^2 / lambda, with "
                    "the grid and pitch DERIVED from the grating rather than declared "
                    "separately"
                ),
                basis=ValidityBasis.ASM_SAMPLING,
                margin=lambda p: fractional_margin(
                    2.0
                    * float(p["talbot_order"])
                    * float(p["period_um"]) ** 2
                    / float(p["wavelength_um"]),
                    (float(p["periods_across_grid"]) * float(p["samples_per_period"]))
                    * (float(p["period_um"]) / float(p["samples_per_period"])) ** 2
                    / float(p["wavelength_um"]),
                ),
                blind_to=(
                    "the window edges, which are excluded from the metric and are where "
                    "a too-small periods_across_grid would show first",
                ),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the input field itself, at z = m * 2 d^2 / lambda. The closed form is "
                "the DISTANCE; the reference field is the input, so nothing external "
                "has to be computed"
            ),
            callable=_talbot_distance_um,
            reference="Talbot self-imaging; Rayleigh 1881",
        ),
        metrics=(
            Metric(
                name="talbot_revival_relative_l2",
                definition="relative_l2_intensity",
                description=(
                    "||U(z_T) - U(0)||_2 / ||U(0)||_2 over the central periods, edges "
                    "excluded"
                ),
                unit=None,
                blind_to=(
                    "a global piston: the revival is up to a constant phase, and the "
                    "metric is taken after removing it",
                    "the edges of the window, which are excluded on purpose and are "
                    "where a too-small grid would show first",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="talbot_revival_relative_l2",
                threshold=5e-3,
                basis=(
                    "DOMINANT TERM, non-paraxial order dephasing. The revival at "
                    "z_T = 2 d^2 / lambda is a PARAXIAL result and Chromatix's angular "
                    "spectrum is not paraxial, so the leading admissible residual is "
                    "that difference and nothing else here is close to it. Order m of "
                    "the grating propagates with k_z = k sqrt(1 - (m lambda / d)^2); "
                    "the first term the paraxial expansion drops contributes "
                    "(1/8) k z (m lambda / d)^4, which at z = z_T is "
                    "delta_phi_m = (pi/2) m^4 (lambda/d)^2 -- independent of z, and set "
                    "entirely by the highest order the grid admits, "
                    "m_max = samples_per_period / 2. At d = 32 um and 8 samples per "
                    "period m_max = 4 (0.111 rad); at 50% duty only odd orders carry "
                    "amplitude, so m = 3 dominates at 0.0352 rad. Because the grating "
                    "is real and even and delta_phi_m is even in m, the first-order "
                    "intensity perturbation cancels and the INTENSITY L2 goes as the "
                    "SQUARE of the amplitude-weighted dephasing -- which is why 0.035 "
                    "rad of order-3 phase error reads 2.0e-4 here and 9.1e-3 on the "
                    "field metric. 5e-3 is a bound set 23x above that predicted "
                    "2.0e-4, not a fitted value. "
                    "SECONDARY, both far below it: the complex64 floor, which is the "
                    "1.9e-5 gap between the float64 exact-ASM prediction (1.99e-4) and "
                    "the measured complex64 run (2.18e-4); and finite-window "
                    "truncation, which is DESIGNED OUT rather than budgeted -- the "
                    "window holds an exact integer number of periods and each period an "
                    "exact integer number of samples, and the same grating through a "
                    "paraxial kernel on the same grid returns 3e-24. An earlier version "
                    "of this basis named truncation as the larger of the two "
                    "contributions; it is not a contribution at all"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a frequency-grid scale error, which moves the revival distance by "
                    "the square of the scale factor and produces a field with no "
                    "resemblance to the input at the nominal z_T; z_T/2, the "
                    "half-Talbot shifted image, which is anticorrelated with the input; "
                    "and the d = 8 um / 32 samples-per-period configuration this family "
                    "was first written with, which admits m = 16 at 455 rad and "
                    "measures 2.4e-1 -- a configuration in which no correct propagator "
                    "could pass, which is why the configuration moved and the threshold "
                    "did not"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="half-talbot",
                description=(
                    "evaluate at z_T/2, where the field is the input shifted by half a "
                    "period. It looks exactly as much like a grating and is not the "
                    "input"
                ),
                mutation="propagate to d^2 / lambda instead of 2 d^2 / lambda",
                target_metric="talbot_revival_relative_l2",
            ),
            NegativeControl(
                control_id="frequency-grid-scale",
                description="scale the frequency grid by 2*pi",
                mutation="build fftfreq in radians per length",
                target_metric="talbot_revival_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
        ),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="talbot_revival_relative_l2",
            observed=2.1774e-4,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_grating_revives_at_the_talbot_distance",
                "tests/test_b1_wave_instances.py::test_the_talbot_configuration_keeps_its_orders_paraxial",
            ),
            note=(
                "2.18e-4 against a 5e-3 gate. Nothing in this repository tested periodic "
                "self-imaging before; a revival is a strong independent check on "
                "propagator phase precisely because it depends on the propagator "
                "reproducing the RELATIVE phases of many diffraction orders at once, "
                "which a single-order comparison cannot see.\n\n"
                "The period is the load-bearing choice and the first one did not work. A "
                "binary grating carries every odd order, and the m-th order's phase after "
                "z_T departs from the paraxial 2*pi m^2 by (pi/2) m^4 (lambda/d)^2 -- so "
                "the residual is set by the HIGHEST PROPAGATING order the grid admits, "
                "m_max = d/(2 pitch). Measured across eleven configurations: at d = 8 um "
                "with 32 samples per period m_max is 16, the dephasing is 455 rad, and "
                "the residual is 2.4e-1 with no revival at all. At d = 32 um with 8 "
                "samples per period m_max is 4, the dephasing is 0.111 rad, and the "
                "residual is 2.2e-4. The tolerance did not move; the configuration in "
                "which the claim is true was found.\n\n"
                "The tolerance basis now leads with that dephasing, which is the term "
                "that dominates, and the decomposition is executable rather than "
                "asserted: the driver runs the same grating through a float64 EXACT-ASM "
                "kernel and a float64 PARAXIAL kernel on the same grid and records both "
                "(NON_PARAXIAL_DEPHASING_BUDGET). The paraxial arm returns 3e-24, so "
                "finite-window truncation is not a contribution to budget -- it is "
                "designed out by the exact-integer periodicity -- and the earlier basis, "
                "which called truncation the larger of the two terms, was wrong about "
                "which physics sets the floor. The exact arm returns 1.99e-4 against the "
                "2.18e-4 measured through the shipping complex64 path, which leaves the "
                "complex64 floor as a 1.9e-5 secondary term. The threshold did not move "
                "and is not affected: 5e-3 sits 23x above the predicted residual either "
                "way.\n\n"
                "One consequence is recorded on the parameter rather than only here: "
                "samples_per_period is NOT a refinement direction for this family. "
                "Raising it admits higher orders and degrades the metric as its fourth "
                "power, so declaring refines_toward=+1 on it -- as this family "
                "originally did -- would have been a false statement about which way "
                "the answer converges. periods_across_grid remains the genuine one.\n\n"
                "The half-Talbot control is the right control for a revival claim and "
                "not merely a convenient one: at z_T/2 the pattern revives SHIFTED by "
                "half a period, so it looks exactly as much like a grating as the "
                "revival does. A metric that only asked 'does this look periodic' would "
                "pass it."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.ORACLE_CONSTRUCTION_EXPENSIVE,
        sampler_absent_note=(
            "generative in (period, wavelength, order), but the grid must hold an "
            "integer number of periods exactly or the revival is spoiled by the window "
            "rather than by the propagator -- a constraint the M9 sampler has to respect."
        ),
        evidence=(
            "knowledge/solvers/chromatix/conventions.md",
            "src/solvers/chromatix/propagation.py",
        ),
)


# ---------------------------------------------------------------------------
# B1-WAVE-ASM-VALIDITY -- the load-bearing one
# ---------------------------------------------------------------------------

#: Periodic self-imaging at z_T = 2 d^2 / lambda.
#:
#: Nothing in this repository tested periodic self-imaging before, and a revival
#: is a strong independent check on propagator phase that no existing probe
#: covers: it depends on the propagator reproducing the RELATIVE phases of many
#: diffraction orders at once, which a single-order comparison cannot see.
#:
#: The period is chosen so that it is an exact number of samples and the grid an
#: exact number of periods -- otherwise the grating is not periodic on the grid
#: and the revival is contaminated by the discontinuity at the wrap rather than
#: by the propagator.
B1_WAVE_TALBOT = register(
    _B1_WAVE_TALBOT.with_instances(
        _B1_WAVE_TALBOT.instantiate(
            "B1-WAVE-TALBOT-01",
            {
                # 32 um at 8 samples per period, and the period is the
                # load-bearing choice rather than a convenience. A binary
                # grating carries every odd order, and the m-th order's phase
                # after z_T departs from the paraxial 2*pi m^2 by
                # (pi/2) m^4 (lambda/d)^2 -- so the residual is set by the
                # HIGHEST PROPAGATING order the grid admits, m_max = d/(2 pitch).
                # Measured across eleven configurations: at d = 8 um with 32
                # samples per period, m_max = 16 and the dephasing is 455 rad,
                # which puts the residual at 2.4e-1 and no revival at all. At
                # d = 32 um with 8 samples per period, m_max = 4, the dephasing
                # is 0.111 rad, and the residual is 2.0e-4. The tolerance is
                # unchanged; what changed is choosing a configuration in which
                # the claim is true.
                "period_um": 32.0,
                "wavelength_um": 0.532,
                "duty_cycle": 0.5,
                "talbot_order": 1,
                "periods_across_grid": 32,
                "samples_per_period": 8,
            },
            expected={
                "talbot_distance_um": 2.0 * 32.0 * 32.0 / 0.532,
                "why": (
                    "z_T = 2 d^2 / lambda = 3849.6 um. The half-Talbot control propagates "
                    "z_T/2 instead, where the pattern is laterally shifted by d/2 -- a "
                    "field that looks exactly as much like a grating as the revival does "
                    "and is displaced by half a period, which is what makes it the right "
                    "control for a revival claim."
                ),
            },
        ),
    )
)


_B1_WAVE_ASM_VALIDITY = BenchmarkFamily(
        family_id="B1-WAVE-ASM-VALIDITY",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
        layer=BenchmarkLayer.QUALIFICATION,
        question=(
            "where does the angular-spectrum method stop being trustworthy on this "
            "grid, and does crossing that boundary produce a plausible-looking wrong "
            "field rather than an exception?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.CONVERGENCE,
        parameters=(
            PhysicalParameter(
                "waist_um", "Gaussian waist of the probe field", unit="um", domain=(0.5, 200.0)
            ),
            PhysicalParameter(
                "distance_um",
                "propagation distance. THE swept parameter: the family exists to "
                "straddle z = N pitch^2 / lambda",
                unit="um",
                domain=(0.0, 1e6),
            ),
            PhysicalParameter(
                "wavelength_um", "wavelength", unit="um", domain=(0.2, 2.0), default=0.532
            ),
            NumericalParameter(
                "grid_n", "samples per axis", domain=(64, 4096), default=512, refines_toward=1
            ),
            NumericalParameter(
                "sample_pitch_um", "grid pitch", unit="um", domain=(0.01, 10.0), default=0.25,
                refines_toward=-1,
            ),
        ),
        validity=(
            ASM_SAMPLING,
            nyquist_from_direction_cosine(
                "GAUSSIAN_DIVERGENCE_NYQUIST",
                "the grid pitch resolves the probe Gaussian's own far-field divergence "
                "angle lambda / (pi w0)",
                lambda p: float(p["wavelength_um"]) / (math.pi * float(p["waist_um"])),
            ),
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the Gaussian closed form w(z) = w0 sqrt(1 + (z/zR)^2), which is exact "
                "at every distance and therefore stays a valid reference on BOTH sides "
                "of the sampling boundary -- that is what makes it usable to show the "
                "solver failing rather than merely disagreeing with another solver"
            ),
            callable=GAUSSIAN_SPREADING.closed_form,
            reference="verification/analytic.py::GAUSSIAN_SPREADING",
        ),
        metrics=(
            Metric(
                name="asm_radius_relative_error_vs_closed_form",
                definition=None,  # a radius measurement against a closed form
                description=(
                    "relative error of the measured 1/e^2 radius against the Gaussian "
                    "closed form, reported at every point of the sweep"
                ),
                unit=None,
                blind_to=(
                    "the SHAPE of the failure. Past the boundary the field wraps and "
                    "the second moment can come back through the right value by "
                    "coincidence, which is why the family also reports the wrapped "
                    "power fraction",
                ),
            ),
            Metric(
                name="wrapped_power_fraction",
                definition="power_ratio",
                description=(
                    "fraction of the total power within one pitch of the window edge -- "
                    "the direct evidence that energy has folded back in"
                ),
                unit=None,
                blind_to=(
                    "aliasing that lands away from the edge, which a periodic wrap can "
                    "produce anywhere in the window",
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="asm_radius_relative_error_vs_closed_form",
                threshold=2e-2,
                basis=(
                    "INSIDE the declared validity domain only. The same threshold "
                    "B1-WAVE-GAUSS gates on, from the same measured agreement; the "
                    "point of this family is that the number is met on one side of "
                    "z = N pitch^2 / lambda and not on the other"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a run that crossed the sampling boundary and reported a "
                    "plausible-looking field. That is the entire subject"
                ),
            ),
            Tolerance(
                metric="wrapped_power_fraction",
                threshold=1e-3,
                basis=(
                    "a Gaussian at 0.4 fill has less than 1e-9 of its power within a "
                    "pitch of the edge, so anything at the 1e-3 level is folded-back "
                    "energy rather than the tail"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects="the aliased regime, which is the whole point",
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="past-the-boundary-must-not-pass",
                description=(
                    "an instance deliberately placed at z = 3 * N pitch^2 / lambda. It "
                    "must be reported OUT_OF_VALIDITY, and its accuracy metric must NOT "
                    "be presented as a passing measurement"
                ),
                mutation="set distance_um to three times the sampling limit",
                target_metric="asm_radius_relative_error_vs_closed_form",
            ),
            NegativeControl(
                control_id="silent-wrap",
                description=(
                    "the same instance, checked for whether anything raised. Nothing "
                    "does: the failure mode is a plausible field, not an exception, and "
                    "a benchmark that only checked for exceptions would call it fine"
                ),
                mutation="none; observe that the out-of-validity run completes normally",
                target_metric="wrapped_power_fraction",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.UNCONVERGED,
        ),
        execution_policy=WAVE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="asm_radius_relative_error_vs_closed_form",
            observed=5.93565e-5,
            evidence=(
                "benchmarks/instances/b1_wave.py",
                "tests/test_b1_wave_instances.py::test_the_sweep_straddles_the_declared_boundary",
                "tests/test_b1_wave_instances.py::test_crossing_the_boundary_is_silent_and_wrong",
                "tests/test_b1_wave_instances.py::test_the_two_boundary_controls_are_cross_instance_and_fire",
            ),
            note=(
                "MET refers to the INSIDE instance, 5.94e-5 against a 2e-2 gate, and the "
                "family's content is the pair rather than either number. Three distances "
                "straddle z = N pitch^2 / lambda at margins +0.80, +0.069 and -15.6. All "
                "three RUN. The far one succeeds, raises nothing, returns a field that "
                "looks like a Gaussian, and its radius is 9.0% wrong with 0.8% of its "
                "power within one pitch of the window edge -- so its gate is UNMET and "
                "that is the measurement.\n\n"
                "The waist is 2.5 um and it is a tension resolved by measurement rather "
                "than a default. At 8 um -- the obvious choice -- the beam's own "
                "bandwidth is 1/(pi w0) = 0.04 cycles/um against a grid Nyquist of 2, a "
                "fiftyfold margin, so the kernel's aliasing never touches it and even "
                "FOUR TIMES past the limit the closed form is reproduced to 1.6e-5: the "
                "family could not demonstrate its own failure mode. At 0.6 um it wraps "
                "but is 2.4 samples across the waist, so the second-moment estimator is "
                "4% biased and the inside instances fail too. 2.5 um is ten samples "
                "across the waist, estimator bias 6e-5, and w(500 um) = 34.0 um against "
                "a 32 um half-window.\n\n"
                "The wrapped-power metric was also wrong first. Measured over the outer "
                "QUARTER of the window it reads 0.35 for the correct field at 500 um and "
                "0.35 for the aliased one, because the beam is genuinely that big -- so "
                "it conflated 'large' with 'folded'. It is the edge band, one pitch wide, "
                "with the ANALYTIC field's own edge fraction carried as the error bar.\n\n"
                "Both controls are cross-instance and the mutation is the propagation "
                "DISTANCE and nothing else: same code, same grid, same oracle, same "
                "estimator. That is what makes them controls rather than two unrelated "
                "runs."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.GENERATION_WEAKENS_INDEPENDENCE,
        sampler_absent_note=(
            "the sampling this family needs is BOUNDARY sampling -- points placed at a "
            "declared signed margin from z = N pitch^2 / lambda -- and a uniform draw "
            "over the declared domain would spend almost all of its budget far inside "
            "the domain, where the family has nothing to say. That is a sampler M9 has "
            "to write against the validity margin, not a generic one."
        ),
        evidence=(
            "src/verification/families/predicates.py",
            "benchmarks/probes/carrier_phase_representation.py",
            "tests/test_family_schema.py::test_the_asm_predicate_bounds_the_transfer_function_sampling",
        ),
        notes=(
            "This is the family that justifies the signed normalized margin. A boolean "
            "validity flag would make 'just outside' unreachable: you could ask whether "
            "an instance is valid, and not for an instance that is 5% outside. Every "
            "other family here uses the predicates defensively; this one uses them as "
            "the coordinate it sweeps."
        ),
)


#: Three distances straddling ``z = N pitch^2 / lambda``, and the point is the
#: pair rather than either one.
#:
#: This is the family that forces the validity margin to be signed and
#: normalized rather than boolean, because what it measures is behaviour NEAR a
#: boundary: the inside instance is well inside, the boundary instance sits just
#: under, and the outside instance is a factor past it. All three RUN -- crossing
#: the limit does not raise, it folds energy back in from the other side and
#: returns a field that looks like a Gaussian and is the wrong size, which is
#: exactly the class of silent wrongness B1 exists to catch.
#:
#: The oracle is the Gaussian closed form, which is exact at every distance and
#: therefore stays valid on BOTH sides. That is what makes it usable to show the
#: solver failing rather than merely disagreeing with another solver.
B1_WAVE_ASM_VALIDITY = register(
    _B1_WAVE_ASM_VALIDITY.with_instances(
        *[
            _B1_WAVE_ASM_VALIDITY.instantiate(
                f"B1-WAVE-ASM-VALIDITY-{index:02d}",
                {
                    # 2.5 um, and the choice is a tension resolved by
                    # measurement rather than a default. At a waist of 8 um the
                    # beam's own bandwidth is 1/(pi w0) = 0.04 cycles/um against
                    # a grid Nyquist of 2 -- a fiftyfold margin -- so the
                    # kernel's aliasing never touches the beam and even four
                    # times past the limit the closed form is reproduced to
                    # 1.6e-5: the family could not demonstrate its own failure
                    # mode. At 0.6 um the beam does wrap, and it is only 2.4
                    # samples across the waist, so the second-moment estimator
                    # is 4% biased and the INSIDE instances fail too. 2.5 um is
                    # ten samples across the waist -- estimator bias 6e-5 -- and
                    # w(500 um) = 34.0 um against a 32 um half-window, so the
                    # far side wraps for the right reason.
                    "waist_um": 2.5,
                    "distance_um": distance,
                    "wavelength_um": 0.532,
                    "grid_n": 256,
                    "sample_pitch_um": 0.25,
                },
                expected={
                    "sampling_limit_um": 256 * 0.25**2 / 0.532,
                    "side": side,
                    "why": why,
                },
            )
            for index, (distance, side, why) in enumerate(
                (
                    (
                        6.0,
                        "inside",
                        "well inside z <= N pitch^2 / lambda = 30.08 um; the closed "
                        "form is reproduced and the gate is met",
                    ),
                    (
                        28.0,
                        "near_boundary",
                        "just under the limit. The margin is small and positive, and "
                        "the point of having this instance is that the family's "
                        "predicate can say HOW close rather than only which side",
                    ),
                    (
                        500.0,
                        "outside",
                        "sixteen times the limit. It runs, it returns a plausible "
                        "Gaussian, and the radius is 9% wrong -- no exception anywhere. "
                        "The gate must NOT be met here, and a run that reported it met "
                        "would mean the metric cannot see aliasing",
                    ),
                ),
                start=1,
            )
        ]
    )
)

