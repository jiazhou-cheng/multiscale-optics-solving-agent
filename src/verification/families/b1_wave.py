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

B1_WAVE_GAUSS = register(
    BenchmarkFamily(
        family_id="B1-WAVE-GAUSS",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="gaussian_radius_relative_error",
            observed=1.8e-4,
            evidence=("src/verification/analytic.py",),
            note=(
                "6.040167 um measured against 6.039084 analytic before the A1 task "
                "shipped. On record; re-checked by nothing in the required gate."
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
)


# ---------------------------------------------------------------------------
# B1-WAVE-AIRY
# ---------------------------------------------------------------------------

B1_WAVE_AIRY = register(
    BenchmarkFamily(
        family_id="B1-WAVE-AIRY",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
        validity=(ASM_SAMPLING,),
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="airy_first_null_relative_error",
            observed=2.3e-2,
            evidence=(
                "src/verification/analytic.py",
                "benchmarks/probes/records/m3_first_null_grid_convergence.json",
            ),
            note=(
                "6.65 um measured against 6.4985 analytic. CHE-103's grid sweep showed "
                "the frozen M3 configuration puts only 2.44 pixels across the Airy "
                "radius and is NOT converged for radius-like metrics, so this family "
                "owes a ladder in focal_plane_pitch_um before its number means anything "
                "stronger than 'inside a sampling-limited tolerance'."
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
)


# ---------------------------------------------------------------------------
# B1-WAVE-TILT
# ---------------------------------------------------------------------------

B1_WAVE_TILT = register(
    BenchmarkFamily(
        family_id="B1-WAVE-TILT",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
            status=GateStatus.MEASURED_OFF_GATE,
            metric="tilt_centroid_signed_relative_error",
            observed=2.3e-4,
            evidence=(
                "src/verification/analytic.py",
                "knowledge/solvers/chromatix/conventions.md",
            ),
            note=(
                "+17.5017 um measured against +17.4977 analytic on the explicit "
                "phase-ramp encoding. The kykx_argument encoding is the one carrying "
                "the hazard and has NOT been measured through this family."
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


B1_WAVE_PLANEPHASE = register(
    BenchmarkFamily(
        family_id="B1-WAVE-PLANEPHASE",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
            status=GateStatus.NOT_MEASURED,
            note=(
                "new to the repository. The phasor sign is documented in "
                "knowledge/solvers/chromatix/conventions.md and asserted nowhere as a "
                "gate on a propagated plane wave."
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
)


# ---------------------------------------------------------------------------
# B1-WAVE-FWDBWD
# ---------------------------------------------------------------------------

B1_WAVE_FWDBWD = register(
    BenchmarkFamily(
        family_id="B1-WAVE-FWDBWD",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
            status=GateStatus.NOT_MEASURED,
            note=(
                "new to the repository, and the cheapest check in M1: two propagations "
                "and a norm. It is declared NOT_MEASURED rather than assumed because "
                "nothing has run it."
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
)


# ---------------------------------------------------------------------------
# B1-WAVE-TALBOT
# ---------------------------------------------------------------------------


def _talbot_distance_um(params: Mapping[str, Any]) -> float:
    """``z_T = 2 d^2 / lambda``, the revival distance of a periodic field."""
    d = float(params["period_um"])
    return 2.0 * d * d / float(params["wavelength_um"])


B1_WAVE_TALBOT = register(
    BenchmarkFamily(
        family_id="B1-WAVE-TALBOT",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
                "sampling of one period",
                domain=(4, 256),
                default=32,
                refines_toward=1,
            ),
        ),
        validity=(ASM_SAMPLING,),
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
                    "the revival is exact for an infinite periodic field; the admissible "
                    "residual is the finite-window truncation plus the complex64 floor. "
                    "At 32 periods the truncation is the larger of the two and scales as "
                    "1/N_periods, so 5e-3 is roughly an order above the expected 1.5e-3 "
                    "and is a bound to be measured against rather than a fitted value"
                ),
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a frequency-grid scale error, which moves the revival distance by "
                    "the square of the scale factor and produces a field with no "
                    "resemblance to the input at the nominal z_T; and z_T/2, the "
                    "half-Talbot shifted image, which is anticorrelated with the input"
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
            status=GateStatus.NOT_MEASURED,
            note=(
                "nothing in this repository tests periodic self-imaging. A revival is a "
                "strong independent check on propagator phase that no existing probe "
                "covers, and it is structurally unlike everything else here -- which is "
                "also what makes it the natural physics-family holdout later."
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
)


# ---------------------------------------------------------------------------
# B1-WAVE-ASM-VALIDITY -- the load-bearing one
# ---------------------------------------------------------------------------

B1_WAVE_ASM_VALIDITY = register(
    BenchmarkFamily(
        family_id="B1-WAVE-ASM-VALIDITY",
        family_version="1.0.0",
        category=BenchmarkCategory.B1,
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
            status=GateStatus.NOT_MEASURED,
            note=(
                "the sampling bound is derived and now executable "
                "(families/predicates.py::asm_transfer_function_sampling); the sweep "
                "across it has not been run. The archived L1-WAVE-01 high-NA case is "
                "the precedent for what the result should look like: refining pupil "
                "sampling alone moved the focal scale 10x while an independent oracle "
                "converged to 2e-14."
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
)
