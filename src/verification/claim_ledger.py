"""What has been scientifically validated, by which oracle, and whether it passed.

CHE-104 (M0.3). ``core/capabilities.py`` is the authoritative answer to "what can
this component *execute*" -- probe-backed, and guarded by a test that fails when
the registry drifts from it. There was no equivalent for "what has this component
been shown to be *correct* about", and the difference matters more: a capability
table says a coupler runs on CUDA in complex64, and says nothing about whether
the field it produces is right.

That information existed. It was spread across registry ``validity`` blocks, six
milestone reports, ``benchmarks/manifest.yaml``'s ``gate_disposition`` prose,
per-benchmark ``tolerances.yaml``, four knowledge-pack cards and ~80 probe
records. Answering "is ``C_PATCH_WFT`` validated, and by what?" meant reading
five files and a report, and the answer was not queryable by anything -- which is
how the one live composed benchmark came to have an unmet primary gate and a
false ``negative_controls_pass`` while being easy to plan around as though green.

Three rules this file follows, and the reasons they are rules
--------------------------------------------------------------

**Every claim names evidence that resolves.** A test node ID that pytest can
collect, or a record path that exists, or a report anchor. ``tests/
test_claim_ledger.py`` checks all three kinds. A ledger whose citations rot is
worse than no ledger, because it reads as coverage.

**Every claim declares whether its oracle shares code with the thing tested.**
This encodes the standing rule that our own numerical code never decides
correctness for our own numerical code. ``O2`` -- the repository's own float64
ASM/RS propagator -- is genuinely useful characterization evidence and must never
decide a gate; L2-PSF-01 already learned this the hard way, having once set a
negative-control floor from an O2 comparison and had to retire it as circular.
An entry marked :attr:`OracleIndependence.SHARES_CODE` cannot be
``gate_deciding``, and a test enforces that.

**An unmet gate is a state the ledger can express.** ``GateStatus.NOT_MET`` is a
first-class value, not an absence. The failure mode being designed against is a
component whose ``maturity`` reads ``characterized`` on all six entries -- which
is what the registry says today, and which carries almost no information. The
interesting distinctions are forward-path-validated-but-gradient-not,
deterministic-limit-gated-but-estimator-only-characterized, and
CPU-validated-CUDA-never-executed, and they live here.

What this file is not
---------------------
It is not a second copy of the measurements. Observed values are quoted so the
ledger is readable, and the evidence path is what makes them checkable; where the
two disagree the evidence wins and the ledger is stale. It is also not a
substitute for the benchmarks M1/M2/M4 will write -- :data:`GAPS` is the list of
what those milestones owe, which is the point of building this before them rather
than after.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

__all__ = [
    "CLAIMS",
    "GAPS",
    "KNOWLEDGE_PACK_REQUIRED_FILES",
    "LEDGER_COMPONENTS",
    "NO_VALIDATED_CLAIM",
    "RANKING_CRITERION",
    "Claim",
    "ClaimKind",
    "Gap",
    "GateStatus",
    "Oracle",
    "OracleIndependence",
    "StochasticEvidence",
    "all_claims",
    "claims_for",
    "coverage",
    "open_gates",
]


class Oracle(StrEnum):
    """What decided the claim.

    Ordered roughly by how much independent information each carries. The first
    four are answers arrived at without running the thing under test; the last
    two are comparisons against ourselves and are marked as such.
    """

    ANALYTIC = "analytic_closed_form"
    CONSERVATION_LAW = "conservation_law"
    CONVERGENCE_EXPONENT = "convergence_exponent"
    INDEPENDENT_IMPLEMENTATION = "independent_implementation"
    #: The deterministic limit of a stochastic estimator -- enumerate everything
    #: it would otherwise sample and it must reduce to the reference exactly.
    #: Independent of the *sampling*, not of the kernel.
    DETERMINISTIC_LIMIT = "deterministic_limit"
    #: Two routes through our own code that should agree. Diagnostic.
    CROSS_ROUTE = "cross_route"
    NONE = "none"


class OracleIndependence(StrEnum):
    """Whether the oracle shares code with what it is judging."""

    #: Shares no code and no traced data with the thing under test.
    INDEPENDENT = "independent"
    #: Shares the kernel or the front end. Characterization only; may not gate.
    SHARES_CODE = "shares_code"
    #: There is no oracle, so the question does not arise.
    NOT_APPLICABLE = "not_applicable"


class GateStatus(StrEnum):
    MET = "met"
    NOT_MET = "not_met"
    #: Measured and reported, with no pass/fail threshold declared. Legitimate
    #: for a characterization, and NOT a synonym for "passed".
    CHARACTERIZED_NO_GATE = "characterized_no_gate"
    #: Declared in the registry or capability table and never executed.
    NOT_MEASURED = "not_measured"
    #: Executed, with a result on record, but outside anything the required gate
    #: runs -- a GPU probe record, a benchmark nobody collects. Distinct from
    #: NOT_MEASURED because the scope it implies is "enrol this", not "measure
    #: this", and distinct from MET because nothing re-checks it.
    MEASURED_OFF_GATE = "measured_off_gate"


class ClaimKind(StrEnum):
    """The axes of the coverage matrix.

    Chosen so that a blank cell is a question someone can act on rather than a
    category error. Every component can in principle be asked all of these.
    """

    FORWARD_ACCURACY = "forward_accuracy"
    CONVENTION = "convention"
    CONSERVATION = "conservation"
    CONVERGENCE = "convergence"
    ROUND_TRIP = "round_trip"
    STRUCTURED_FAILURE = "structured_failure"
    GRADIENT = "gradient"
    DEVICE_PARITY = "device_parity"
    #: What it costs to run. Added by CHE-134: route choice is a scientific
    #: decision with a measurable cost, and an agent cannot reason about it from
    #: a number nobody recorded. Never gate-deciding -- there is no correct
    #: number of seconds -- which is why it only ever appears on a B4 family.
    COST = "cost"


@dataclass(frozen=True)
class StochasticEvidence:
    """The four evidence kinds ``benchmarks/protocols/coupler_protocol.yaml`` requires.

    Recorded as four separate facts rather than one "validated" flag because
    they fail independently and mean different things. An estimator can be exact
    in the enumeration limit and still biased; unbiased and still converge at the
    wrong rate; converge correctly and have variance that makes the required ray
    count unreachable, which is precisely the open question on ``C_WAVE_TO_RAY``.
    """

    exactness_limit: str | None = None
    unbiasedness: str | None = None
    convergence_exponent: str | None = None
    variance_characterization: str | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "exactness_limit",
                "unbiasedness",
                "convergence_exponent",
                "variance_characterization",
            )
            if getattr(self, name) is None
        )

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class Claim:
    """One scientific claim, its oracle, and whether it currently holds."""

    component: str
    kind: ClaimKind
    #: What is asserted, in a sentence a reader can disagree with.
    claim: str
    oracle: Oracle
    oracle_independence: OracleIndependence
    #: Evidence that must resolve: a pytest node ID (``path::name``), a
    #: repository-relative record path, or ``path#anchor`` into a report.
    evidence: tuple[str, ...]
    metric: str | None = None
    tolerance: float | None = None
    tolerance_basis: str | None = None
    observed: float | None = None
    gate_status: GateStatus = GateStatus.CHARACTERIZED_NO_GATE
    device: str = "cpu"
    dtype: str | None = None
    namespace: str | None = None
    stochastic: StochasticEvidence | None = None
    #: Declared-but-unmeasured territory this claim does NOT cover.
    caveats: tuple[str, ...] = ()

    @property
    def is_open_gate(self) -> bool:
        return self.gate_status is GateStatus.NOT_MET

    @property
    def gate_deciding(self) -> bool:
        """Whether this claim's verdict decides a benchmark's pass/fail.

        Derived rather than declared. It was a separate field for one revision
        and review found it was identical to ``gate_status in {MET, NOT_MET}``
        on all 26 claims -- a hand-maintained restatement, which is a second
        place for the same fact to live and therefore a place for the two to
        disagree. Having a declared threshold and being allowed to decide are
        the same thing here: a claim with no threshold is a characterization,
        and a characterization does not gate.
        """
        return self.gate_status in (GateStatus.MET, GateStatus.NOT_MET)


@dataclass(frozen=True)
class Gap:
    """Something the ledger says is not established, and who owes it."""

    component: str
    kind: ClaimKind
    gap: str
    #: Ranking input. See RANKING_CRITERION.
    blocks: tuple[str, ...]
    severity: str
    owner: str
    rationale: str = ""


#: The components the ledger must cover: everything in
#: ``core.capabilities.COMPONENT_CAPABILITIES`` and everything in the registry.
#: Duplicated here rather than imported, for the reason
#: ``tests/test_generated_artifacts.py`` gives about its own table -- importing
#: the source of truth would make the coverage test agree by construction,
#: including about which components exist at all.
LEDGER_COMPONENTS = (
    "M_RAY_OPTILAND",
    "M_WAVE_CHROMATIX",
    "C_RAY_TO_WAVE",
    "C_WAVE_TO_RAY",
    "C_PLANAR_DOE_STEP",
    "C_PATCH_WFT",
)

#: What a complete knowledge pack contains. Solver packs and coupler packs
#: differ in one file: a coupler needs ``theory.md`` (the paper equation it
#: implements) where a solver needs ``api_minimal_examples.md`` and
#: ``usage_notes.md`` (how to drive somebody else's library).
KNOWLEDGE_PACK_REQUIRED_FILES = {
    "solver": (
        "card.yaml",
        "conventions.md",
        "usage_notes.md",
        "api_minimal_examples.md",
        "failure_guide.md",
    ),
    "coupler": ("card.yaml", "conventions.md", "failure_guide.md", "theory.md"),
}

#: Components with no validated claim at all, and why. Explicit so that the
#: coverage test can distinguish "nobody has looked" from "the ledger forgot".
NO_VALIDATED_CLAIM: dict[str, str] = {}


RANKING_CRITERION = """Gaps are ranked by what they would let through, not by effort.

  critical -- a wrong answer could be produced and reported as correct, because
              nothing independent is checking. A missing oracle outranks a
              missing convergence study.
  high     -- either (a) an independent oracle IS checking and is failing, with
              the residual unattributed -- loud rather than silent, but a known
              wrong answer with no owner for the mechanism; or (b) a claim the
              repository makes in the registry that no executed evidence
              supports. The claim is the liability.
  medium   -- established on one path and extrapolated to another (a device, a
              dtype, a field position) without measurement; or a usability gap
              that blocks an agent from using a component correctly without
              itself making anything numerically wrong.
  low      -- a real gap whose failure mode is visible rather than silent.

Two buckets were added after a first pass ranked by importance instead of by
this criterion. A known-failing gate is NOT critical under the definition above,
because something independent is checking -- it is loudly wrong, which is the
better failure. And a missing knowledge pack is not a claim at all, so it never
fitted "high"; it blocks an agent rather than producing a wrong number.
"""


def _t(path: str, name: str) -> str:
    """A pytest node ID, spelled the way pytest collects it."""
    return f"tests/{path}::{name}"


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------
#
# Read this as an inventory of what is actually established, not as a summary of
# what has been worked on. The two differ most at C_WAVE_TO_RAY, which has more
# executed evidence than anything else here and still cannot support a forward
# accuracy claim on a real system, and at C_PLANAR_DOE_STEP / C_PATCH_WFT, which
# have graph nodes and no knowledge packs.

#: Claims not yet expressed as a :class:`~verification.families.schema.BenchmarkFamily`.
#:
#: CHE-131 made the family registry authoritative wherever it has content, and
#: :data:`CLAIMS` below is that registry's projection plus this remainder. The
#: two cannot drift, because ``tests/test_claim_ledger.py`` fails when a legacy
#: row and a projected claim occupy the same ``(component, kind)`` cell -- so
#: landing a family *forces* the row it replaces to be deleted rather than
#: leaving a silent duplicate.
_LEGACY_CLAIMS: tuple[Claim, ...] = (
    # -- M_RAY_OPTILAND ------------------------------------------------------
    Claim(
        component="M_RAY_OPTILAND",
        kind=ClaimKind.CONVENTION,
        claim=(
            "opd_native is an index-weighted slant optical path, seeded at zero, whose "
            "zero point is the object plane for a finite object and is aperture-dependent "
            "for an infinite one."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_optiland_opd_convention.py",
                "test_opd_is_the_slant_path_not_the_axial_separation",
            ),
            _t("test_optiland_opd_convention.py", "test_opd_is_index_weighted_optical_path"),
            _t(
                "test_optiland_opd_convention.py",
                "test_infinite_object_opl_zero_is_aperture_dependent",
            ),
            "benchmarks/probes/records/optiland/opd_convention_probe.json",
        ),
        metric="agreement with hand-computed geometric path on manufactured geometries",
        gate_status=GateStatus.MET,
        dtype="float64",
        namespace="numpy",
        caveats=(
            "A paraxial surface is NOT an admissible OPL source and is refused; the "
            "convention is established for standard surfaces only.",
        ),
    ),
    Claim(
        # COMPOSED, and filed here anyway. The exactness limit is the only place
        # an Optiland trace is held against a closed form, so leaving this row
        # blank would overstate the gap; attributing it to Optiland alone would
        # overstate the coverage. It is recorded as what it is, and the
        # solver-only gap it does NOT close is in the gap list.
        component="M_RAY_OPTILAND",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "In the zero-sampling-error limit, the composed bridge "
            "wave -> rays -> Optiland trace -> wave reproduces the analytic layered "
            "plane-wave oracle, in air and through a plane-parallel n=1.5 slab where "
            "Optiland must refract at two interfaces and accumulate an index-weighted "
            "path."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_coherent_bridge.py",
                "TestExactnessLimit::test_the_exactness_limit_reproduces_the_analytic_oracle",
            ),
            _t(
                "test_coherent_bridge.py",
                "TestExactnessLimit::test_the_grazing_band_limit_is_what_makes_the_exactness_limit_exact",
            ),
            "docs/precision/precision_device_policy.md",
        ),
        metric="max abs field error against the closed form",
        tolerance=1.0e-11,
        tolerance_basis=(
            "benchmarks/protocols/coupler_protocol.yaml zero-sampling-error limit, "
            "dtype-roundoff derived. Observed 9.1e-14 on the [air] parametrization "
            "since CHE-102 fixed the namespace/backend mismatch; it had been failing "
            "~32% of runs at 8.9e-9. Value quoted from "
            "docs/precision/precision_device_policy.md."
        ),
        observed=9.1e-14,
        gate_status=GateStatus.MET,
        dtype="float64",
        namespace="numpy",
        caveats=(
            "This is a COMPOSED claim over C_WAVE_TO_RAY + M_RAY_OPTILAND + "
            "C_RAY_TO_WAVE. It cannot isolate a solver-side error that the "
            "reconstruction happens to undo, and it does not establish anything about "
            "a CURVED refracting surface -- the slab has two flat interfaces.",
        ),
    ),
    Claim(
        component="M_RAY_OPTILAND",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "Invalid scalars, unsupported capability requests and gradient requests are "
            "refused with a structured diagnostic before any solver call, rather than "
            "producing a plausible number."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_solver_adapter_characterization.py",
                "test_optiland_invalid_scalars_fail_as_a_structured_result",
            ),
            _t(
                "test_solver_adapter_characterization.py",
                "test_optiland_capability_refusals_raise_before_any_solver_call",
            ),
            _t(
                "test_solver_adapter_characterization.py",
                "test_the_failure_code_inventory_is_unchanged",
            ),
        ),
        metric="failure-code inventory, pinned",
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="M_RAY_OPTILAND",
        kind=ClaimKind.DEVICE_PARITY,
        claim=(
            "The process-global solver backend matches the namespace the arrays were "
            "planned in, or the trace is refused rather than silently converted."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_coherent_bridge.py",
                "TestExactnessLimit::test_the_exactness_limit_reproduces_the_analytic_oracle",
            ),
            "docs/precision/precision_device_policy.md",
        ),
        metric="FAIL_BACKEND_NAMESPACE_MISMATCH raised on disagreement",
        gate_status=GateStatus.MET,
        caveats=(
            "CUDA is reachable only through the torch backend and the coherent path's "
            "CUDA execution is not covered by the default gate.",
        ),
    ),
    # -- M_WAVE_CHROMATIX ----------------------------------------------------
    Claim(
        component="M_WAVE_CHROMATIX",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "The carrier-removed ASM is the same propagation as the absolute-path ASM, "
            "exactly rather than paraxially, and matches Chromatix up to the removed "
            "carrier in float64."
        ),
        oracle=Oracle.INDEPENDENT_IMPLEMENTATION,
        oracle_independence=OracleIndependence.SHARES_CODE,
        evidence=(
            _t("test_carrier_removed_asm.py", "test_identity_is_not_a_paraxial_approximation"),
            _t(
                "test_carrier_removed_asm.py",
                "test_propagator_matches_chromatix_up_to_the_carrier_in_float64",
            ),
            _t(
                "test_carrier_removed_asm.py",
                "test_float64_carrier_conventions_are_the_same_propagation",
            ),
            "benchmarks/probes/records/chromatix/propagation_probe.json",
        ),
        metric="relative intensity error against the same-family propagator",
        gate_status=GateStatus.CHARACTERIZED_NO_GATE,
        dtype="complex64",
        namespace="jax",
        caveats=(
            "Marked shares_code deliberately: the reference is our own float64 "
            "propagator built from Chromatix's own kernel_propagate, so it checks the "
            "carrier bookkeeping and NOT whether Chromatix's ASM is right. It may not "
            "decide a gate.",
        ),
    ),
    Claim(
        component="M_WAVE_CHROMATIX",
        kind=ClaimKind.CONVENTION,
        claim=(
            "The removed carrier is reported and never silently reapplied, and the "
            "propagation pins jax_enable_x64 off regardless of ambient process state."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_carrier_removed_asm.py",
                "test_removed_carrier_is_reported_and_not_silently_reapplied",
            ),
            _t(
                "test_carrier_removed_asm.py",
                "test_propagation_pins_jax_x64_off_regardless_of_ambient_state",
            ),
            _t("test_carrier_removed_asm.py", "test_the_removed_factor_is_exactly_the_carrier"),
        ),
        gate_status=GateStatus.MET,
        caveats=(
            "CHE-103 finding: pinning x64 off cannot retroactively downcast a Field "
            "already built under x64, and nothing checks the flag at the boundary that "
            "depends on it. Committed records' precision depended on import order. See "
            "measurement_precision in benchmarks/probes/records/m3_psf_verification.json.",
        ),
    ),
    Claim(
        # Filed under the COMPOSED path, not under Chromatix. The measured
        # quantity is L2-PSF-01's end-to-end ledger from traced ray power to PSF
        # integral, and the one step in it that must be unity is
        # psf_integral / propagated_power_out -- which is |u|^2 on the same grid
        # with the same pitch, i.e. a measurement-layer identity rather than a
        # physical conservation law. The ASM step's own ratio (0.9999998) is
        # attributed to window escape by prose, and the record warns in the same
        # breath that on an UNPADDED run it reads 1.0 through wraparound, "which
        # is why 1.0 is not evidence of correctness". So this establishes that
        # the composed ledger closes and every step has a named mechanism. It
        # does NOT establish that Chromatix conserves power.
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.CONSERVATION,
        claim=(
            "The composed ray -> wave -> sensor energy ledger closes: every step from "
            "traced ray power to PSF integral has a named mechanism, and the "
            "unattributed remainder is bounded."
        ),
        oracle=Oracle.CONSERVATION_LAW,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            "benchmarks/probes/records/m3_psf_verification.json",
            "benchmarks/protocols/slice_protocol.yaml",
            _t("test_psf_verification.py", "test_the_energy_ledger_closes_where_it_must"),
        ),
        metric="energy_accounting_unexplained_residual",
        tolerance=1.0e-3,
        tolerance_basis=(
            "benchmarks/protocols/slice_protocol.yaml tolerance_budget.gates. Bounds "
            "the unattributed remainder across the whole chain, not any single step."
        ),
        observed=2.9084392072498133e-07,
        gate_status=GateStatus.MET,
        dtype="complex64",
        namespace="jax",
        caveats=(
            "Not a statement that the ASM conserves power. The only step required to "
            "be unity is a measurement-layer identity, and the propagation step's own "
            "ratio is attributed to window escape by prose rather than measured. "
            "M_WAVE_CHROMATIX therefore has NO conservation claim of its own, which "
            "is why that cell in the coverage matrix is blank.",
            "pupil_over_traced is explicitly NOT a conservation law: it converts ray "
            "weights into a field power and is reported as a measure conversion.",
        ),
    ),
    Claim(
        component="M_WAVE_CHROMATIX",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "A missing input, an invalid request and a gradient request are each refused "
            "with a structured diagnostic that accumulates every problem rather than "
            "reporting the first."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_solver_adapter_characterization.py",
                "test_chromatix_missing_input_is_a_structured_failure",
            ),
            _t(
                "test_solver_adapter_characterization.py",
                "test_chromatix_validate_request_accumulates_every_problem",
            ),
            _t(
                "test_solver_adapter_characterization.py",
                "test_chromatix_refuses_a_gradient_request_before_running",
            ),
        ),
        gate_status=GateStatus.MET,
    ),
    # -- C_RAY_TO_WAVE -------------------------------------------------------
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "A collimated ray bundle reconstructs the exact plane wave, superposed modes "
            "add coherently, and the reconstruction does not depend on where along the "
            "rays the bundle was launched."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t("test_ray_to_wave.py", "test_collimated_bundle_reconstructs_the_exact_plane_wave"),
            _t("test_ray_to_wave.py", "test_superposed_modes_add_coherently"),
            _t(
                "test_ray_to_wave.py",
                "test_reconstruction_is_independent_of_where_the_rays_are_launched",
            ),
            # Hand-built RayBundle straight into ray_to_wave -- no solver is
            # involved, so this is coupler evidence and is filed here.
            _t(
                "test_coherent_bridge.py",
                "TestTwoRayInterference::test_the_fringe_follows_the_path_difference_exactly",
            ),
            _t(
                "test_coherent_bridge.py",
                "TestTwoRayInterference::test_a_half_wave_path_difference_extinguishes",
            ),
        ),
        metric="max abs field error against the closed-form plane wave",
        gate_status=GateStatus.MET,
        dtype="complex128",
        namespace="numpy",
        caveats=(
            "Established on synthetic bundles with a known answer. It does NOT carry "
            "over to a real traced aberrated system, where the composed benchmark's "
            "gate is unmet -- see the L2-PSF-01 entry below.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "On the real traced M3-SINGLET-REF system the reconstructed sensor-plane "
            "intensity agrees with the analytic Airy oracle to within the frozen gate."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            "benchmarks/physics/L2-PSF-01/tolerances.yaml",
            "benchmarks/probes/records/m3_quadrature_weight.json",
            "benchmarks/probes/records/singlet_residual_grid.json",
            "benchmarks/probes/records/singlet_residual_attribution.json",
            "benchmarks/reports/2026-08/ray_to_wave_slice_exit.md",
        ),
        metric="fft_oracle_intensity_relative_l2 vs O1 over the 5-Airy-radius disc",
        tolerance=1.0e-3,
        tolerance_basis=(
            "Frozen by M3.2 (CHE-31), re-affirmed unwidened through M3.8 and M3.9R. "
            "benchmarks/physics/L2-PSF-01/tolerances.yaml states the basis in full."
        ),
        observed=2.2072391812867093e-3,
        gate_status=GateStatus.NOT_MET,
        # numpy/float64, NOT the shipping complex64 jax path. The measurement in
        # benchmarks/probes/records/m3_quadrature_weight.json is produced by
        # benchmarks/probes/quadrature_weight.py, which reuses
        # sensor_handoff_convergence._asm_float64 in preference to the Chromatix
        # adapter precisely "so that a plane is not blamed for a complex64 cast".
        # Stamping this complex64 would invite a reader to charge part of the
        # 2.21e-3 to float32 truncation that is not in the number.
        dtype="float64",
        namespace="numpy",
        caveats=(
            "Measured at 787,969 rays with the production (weighted) configuration. "
            "The UNIFORM configuration reaches 9.21e-4 and clears the gate -- the "
            "opposite ordering from the O2 diagnostic.",
            "CHE-117 attributed the residual and it is NOT a numerical artifact. It is "
            "converged in both refinable directions: flat to 0.87% from 49,537 to "
            "3,148,801 rays, and identical to ten significant figures across an 8x "
            "sensor-pitch refinement at fixed window (6.5 to 51.9 pixels per Airy "
            "radius). The uniform arm's 9.21e-4 is not a competing converged value -- "
            "it descends through the weighted arm near 181 rings, reaches a 7.0e-4 "
            "minimum at 362, and climbs back to 1.52e-3 by 1024 rings, heading for the "
            "same limit. The two arms differ in exactly one place, the outermost "
            "ring's half area weight; removing the central ray's 3/4 correction "
            "changes nothing.",
            "What remains open after CHE-117 is narrower than what it inherited: not "
            "why the weight hurts (it does not), but whether an aberration-free "
            "paraxial Airy oracle can decide a real traced singlet at the 1e-3 level "
            "at all. Separating the system's own aberration from a coupler error needs "
            "an oracle this family does not yet have, and O2 does not qualify.",
            "CHE-103 adds a candidate contributor the original write-up did not have: "
            "the frozen grid samples the Airy radius with 2.44 pixels and is not "
            "converged for radius-like metrics, and off-axis the weighted "
            "configuration is 7.5x worse against the same analytic oracle. CHE-117 "
            "retired the first of those two: the sensor grid this residual is measured "
            "on is converged, and the 2.44-pixel finding applies to the separate "
            "pupil-to-focus grid. The off-axis factor is untouched and still open.",
            "This gate must NOT be closed against another Optiland PSF route. FFTPSF "
            "and HuygensPSF share one Wavefront/OPD front end and are therefore one "
            "oracle, not two (PB7/CHE-58 finding F2). Migrated here by CHE-133 from "
            "benchmarks/manifest.yaml's gate_disposition prose, and enforced "
            "structurally by BenchmarkFamily, which refuses a CROSS_ROUTE oracle "
            "outside category B4 and refuses a gating tolerance inside it.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.CONVENTION,
        claim=(
            "L2-PSF-01's quadrature-weight negative control fires: adding the production "
            "weight improves agreement with the analytic oracle by at least 1.2x."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            "benchmarks/physics/L2-PSF-01/tolerances.yaml",
            "benchmarks/probes/records/m3_quadrature_weight.json",
            "benchmarks/probes/records/singlet_residual_attribution.json",
        ),
        metric="quadrature_weight_min_improvement_factor vs O1",
        tolerance=1.2,
        tolerance_basis=(
            "benchmarks/physics/L2-PSF-01/tolerances.yaml. The floor previously came "
            "from an O2 comparison at 1.575; that was circular validation -- our own "
            "ASM/RS propagator deciding correctness for our own coupler -- and was "
            "retired as the gate input while still being reported."
        ),
        observed=0.4173375512174577,
        gate_status=GateStatus.NOT_MET,
        caveats=(
            "This control fires BACKWARDS: 0.42 means the uniform configuration is "
            "CLOSER to the analytic oracle than the weighted one (9.21e-4 vs 2.21e-3). "
            "It is what makes L2-PSF-01's negative_controls_pass false, and it is "
            "reported rather than hidden or widened.",
            "The weight is independently required for absolute-power convergence "
            "(CHE-33's N^2.0024), so this is not an argument for removing it.",
            "CHE-117 attributed it, and the finding is that the CONTROL is "
            "mis-specified rather than the weight defective. The comparison is made at "
            "one ray count with neither arm required to be converged, so its verdict "
            "moves with that ray count: 10.7 at 8 rings, 1.79 at 128, 0.42 at 512, "
            "0.69 at 1024. The weighted arm is converged from 24 rings; the uniform "
            "arm descends past it, bottoms out at 7.0e-4 near 362 rings, and is "
            "climbing back toward the same 2.208e-3 at 1024 rings. Respecifying the "
            "control to require convergence of both arms is a change to the control, "
            "not to this floor, and the floor is not widened here.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.CONVENTION,
        claim=(
            "Every removed term of main-text eq 2 is detected by an independent oracle, "
            "and the two invisible-by-symmetry cases are recorded as invisible rather "
            "than as passes."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t("test_ray_to_wave.py", "test_each_removed_term_is_detected"),
            _t(
                "test_ray_to_wave.py",
                "test_projection_factor_omission_is_invisible_at_normal_incidence",
            ),
            _t(
                "test_ray_to_wave.py",
                "test_oblique_ramp_omission_is_invisible_for_a_single_centred_ray",
            ),
            "benchmarks/probes/records/m3_psf_verification.json",
        ),
        metric="detection margin against the unperturbed residual",
        gate_status=GateStatus.MET,
        caveats=(
            "Two controls do not fire on the frozen configuration and say so: "
            "axis_transpose (a circular on-axis PSF is transpose-symmetric) and "
            "amplitude_weight_omitted (margin 1.06 on axis). The latter is no longer "
            "an exact no-op -- CHE-103.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.CONVERGENCE,
        claim=(
            "The reconstruction converges under ray refinement, and the residual against "
            "the analytic oracle falls monotonically over the sweep."
        ),
        oracle=Oracle.CONVERGENCE_EXPONENT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            "benchmarks/probes/records/m3_convergence.json",
            "benchmarks/probes/records/m3_psf_verification.json",
            _t(
                "test_psf_verification.py",
                "test_the_gate_failures_are_attributed_by_a_measured_ray_count_trend",
            ),
        ),
        metric="fft_oracle_relative_l2 over a 6-point ray sweep",
        observed=6.157153193339412e-3,
        gate_status=GateStatus.CHARACTERIZED_NO_GATE,
        caveats=(
            "CHE-103: the airy_peak_intensity_relative metric's ray-sampling "
            "attribution is WITHDRAWN. best_over_ray_count_sweep is 5.40e-3 against a "
            "2.0e-3 gate, where before CHE-47 it was 2.93e-4, so refinement no longer "
            "reaches that gate. Owner M4.2 (CHE-117).",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "A bundle carrying only an Optiland weight is refused rather than promoted, "
            "and a pitch given in the wrong unit is caught by the grid condition."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t("test_ray_to_wave.py", "test_a_bundle_carrying_only_an_optiland_weight_is_refused"),
            _t(
                "test_ray_to_wave.py",
                "test_a_millimetre_for_metre_pitch_error_is_caught_by_the_grid_condition",
            ),
            _t(
                "test_ray_to_wave.py",
                "test_ray_density_diagnostic_reports_not_computed_rather_than_guessing",
            ),
        ),
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.ROUND_TRIP,
        claim=(
            "ray -> wave -> ray recovers the spectral content, and the pair's phase-sign "
            "and obliquity conventions must match for it to."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.SHARES_CODE,
        evidence=(
            _t(
                "test_coupler_round_trip.py",
                "test_ray_to_wave_to_ray_recovers_the_spectral_content",
            ),
            _t(
                "test_coupler_round_trip.py",
                "test_a_mismatched_phase_sign_pairing_breaks_the_round_trip",
            ),
            _t(
                "test_coupler_round_trip.py",
                "test_power_terms_are_reported_separately_rather_than_netted",
            ),
        ),
        gate_status=GateStatus.CHARACTERIZED_NO_GATE,
        caveats=(
            "shares_code by construction: a round trip runs the thing under test "
            "forwards and backwards, so a convention error the two directions SHARE "
            "cancels and is invisible. It may not gate. The single-direction analytic "
            "claims are the load-bearing ones.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.GRADIENT,
        claim="No gradient is claimed across this boundary.",
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=("src/registry/couplers.yaml",),
        gate_status=GateStatus.NOT_MEASURED,
        caveats=(
            "registry declares derivative.mode = finite_difference, verified = false. "
            "forward_only until a derivative contract and a finite-difference "
            "validation pass.",
        ),
    ),
    Claim(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.DEVICE_PARITY,
        claim=(
            "The xp-parameterized reconstruction executes on CUDA with the arrays "
            "staying resident, and the CPU and GPU routes agree within the declared "
            "float32 bound."
        ),
        oracle=Oracle.CROSS_ROUTE,
        oracle_independence=OracleIndependence.SHARES_CODE,
        evidence=(
            _t(
                "test_precision_gpu_pipeline.py",
                "TestLiveBoundaryAndEndToEnd::test_the_live_coupler_boundary_never_leaves_the_device",
            ),
            _t(
                "test_precision_gpu_pipeline.py",
                "TestLiveBoundaryAndEndToEnd::test_the_reduced_gpu_field_agrees_with_the_host_float64_reference",
            ),
            "benchmarks/probes/records/ray_wave/demo3_route_agreement.json",
        ),
        gate_status=GateStatus.MEASURED_OFF_GATE,
        device="cuda",
        dtype="complex64",
        namespace="jax",
        caveats=(
            "measured_off_gate, not met: the gpu-marked tests skip unless the session "
            "is GPU-dedicated, so nothing in the required gate re-checks this. The "
            "scope it implies is 'enrol it', not 'measure it'.",
            "Cross-route agreement is our own two routes agreeing, so it cannot decide "
            "correctness -- only that the device port did not change the answer.",
        ),
    ),
    # -- C_WAVE_TO_RAY -------------------------------------------------------
    Claim(
        component="C_WAVE_TO_RAY",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "Enumerating every propagating spectral bin with the importance weight "
            "applied reduces the estimator to the deterministic reference at round-off."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_wave_to_ray.py", "test_enumerating_every_propagating_bin_reproduces_the_field"
            ),
            _t("test_wave_to_ray.py", "test_centered_dft_pairing_is_what_makes_the_limit_exact"),
            _t(
                "test_coupler_round_trip.py",
                "test_wave_to_ray_to_wave_is_exact_in_the_enumeration_limit",
            ),
        ),
        metric="max abs field error in the full-enumeration limit",
        tolerance_basis="dtype_roundoff_derived, per coupler_protocol.yaml exactness_limit",
        gate_status=GateStatus.MET,
        stochastic=StochasticEvidence(
            exactness_limit=_t(
                "test_wave_to_ray.py", "test_enumerating_every_propagating_bin_reproduces_the_field"
            ),
            unbiasedness=_t(
                "test_wave_to_ray.py", "test_ensemble_mean_is_unbiased_within_three_standard_errors"
            ),
            convergence_exponent=_t(
                "test_wave_to_ray.py", "test_error_falls_as_n_to_the_minus_one_half"
            ),
            variance_characterization=_t(
                "test_wave_to_ray.py",
                "test_magnitude_sampling_helps_most_on_a_concentrated_spectrum",
            ),
        ),
        caveats=(
            "This is the estimator's exactness limit, not a claim about the sampled "
            "estimator at a usable ray count. The variance that decides whether the "
            "paper's ray budget is reachable is M5.3 (CHE-120).",
            "The reconstruction half of the comparison runs through couplers.ray_to_wave, "
            "so the limit is a wave->ray->wave round trip and inherits the round trip's "
            "blind spot: a convention error the PAIR shares cancels. What keeps this "
            "marked independent is test_centered_dft_pairing_is_what_makes_the_limit_exact, "
            "which holds the pairing against numpy's own centred FFT rather than against "
            "the coupler.",
        ),
    ),
    Claim(
        component="C_WAVE_TO_RAY",
        kind=ClaimKind.CONVENTION,
        claim=(
            "Omitting the importance weight, flipping the normal-component sign, and "
            "omitting the launch phase are each detected; the two that are invisible in "
            "degenerate configurations are recorded as invisible."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t("test_wave_to_ray.py", "test_omitting_the_importance_weight_is_detected_as_a_bias"),
            _t(
                "test_wave_to_ray.py",
                "test_flipping_the_normal_component_sign_is_detected_off_plane",
            ),
            _t(
                "test_wave_to_ray.py",
                "test_omitting_the_launch_phase_breaks_multi_position_emission",
            ),
            _t(
                "test_wave_to_ray.py",
                "test_launch_phase_omission_is_invisible_for_a_single_centred_position",
            ),
        ),
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="C_WAVE_TO_RAY",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "Stochastic sampling without an explicit seed is refused, a sampling density "
            "with a hole is refused as inconsistent, and selecting an evanescent bin is "
            "refused rather than returning a decaying nothing."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_wave_to_ray.py",
                "test_stochastic_sampling_without_an_explicit_seed_is_refused",
            ),
            _t("test_wave_to_ray.py", "test_a_density_with_a_hole_is_refused_as_inconsistent"),
            _t("test_wave_to_ray.py", "test_selecting_an_evanescent_bin_is_refused"),
            _t("test_wave_to_ray.py", "test_evanescent_power_is_reported_as_a_named_loss"),
        ),
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="C_WAVE_TO_RAY",
        kind=ClaimKind.ROUND_TRIP,
        claim=(
            "wave -> ray -> wave is exact in the enumeration limit and converges at the "
            "Monte Carlo rate otherwise; a mismatched phase-sign or obliquity pairing "
            "breaks it."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_coupler_round_trip.py",
                "test_wave_to_ray_to_wave_is_exact_in_the_enumeration_limit",
            ),
            _t(
                "test_coupler_round_trip.py",
                "test_wave_to_ray_to_wave_converges_at_the_monte_carlo_rate",
            ),
            _t(
                "test_coupler_round_trip.py",
                "test_a_mismatched_phase_sign_pairing_breaks_the_round_trip",
            ),
            _t(
                "test_coupler_round_trip.py",
                "test_the_obliquity_convention_must_also_match_across_the_pair",
            ),
        ),
        gate_status=GateStatus.MET,
        caveats=(
            "Round-trip exactness is a statement about the PAIR. It cannot detect a "
            "convention error that both directions share, which is why the "
            "single-direction analytic claims above are the load-bearing ones.",
        ),
    ),
    Claim(
        component="C_WAVE_TO_RAY",
        kind=ClaimKind.GRADIENT,
        claim=(
            "The fixed-direction estimator is unbiased for the gradient on a fixed "
            "spectral grid, and detaching the sampling density is what makes it so. "
            "Characterized, not certified."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_coupler_gradient.py",
                "test_the_fixed_direction_estimator_is_unbiased_on_a_fixed_spectral_grid",
            ),
            _t(
                "test_coupler_gradient.py",
                "test_detaching_the_density_is_what_makes_the_estimator_unbiased",
            ),
            _t(
                "test_coupler_gradient.py",
                "test_the_report_names_the_omitted_terms_and_carries_the_step_table",
            ),
        ),
        metric="estimator mean vs central finite difference, in standard errors",
        gate_status=GateStatus.CHARACTERIZED_NO_GATE,
        caveats=(
            "Unbiased on a FIXED spectral grid only. The omitted terms are named in the "
            "report rather than bounded, an intensity objective at the DOE plane has no "
            "derivative at all, and the registry keeps derivative.verified = false. "
            "No gradient is claimed across the solver boundary.",
        ),
    ),
    # -- C_PLANAR_DOE_STEP ---------------------------------------------------
    Claim(
        component="C_PLANAR_DOE_STEP",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "Full enumeration of the diffracted orders reproduces the transmitted field, "
            "and the outgoing ray budget is the caller's rather than a function of the "
            "incident count."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_planar_doe_step.py",
                "test_full_enumeration_still_reproduces_the_transmitted_field",
            ),
            _t(
                "test_planar_doe_step.py",
                "test_the_outgoing_count_does_not_depend_on_the_incident_count",
            ),
            _t(
                "test_planar_doe_step.py",
                "test_two_stacked_does_keep_the_outgoing_count_at_the_budget",
            ),
        ),
        gate_status=GateStatus.MET,
        stochastic=StochasticEvidence(
            exactness_limit=_t(
                "test_planar_doe_step.py",
                "test_full_enumeration_still_reproduces_the_transmitted_field",
            ),
        ),
        caveats=(
            "Only the exactness limit of the four required stochastic evidence kinds. "
            "No unbiasedness, convergence-exponent or variance evidence exists for the "
            "SAMPLED step.",
        ),
    ),
    Claim(
        component="C_PLANAR_DOE_STEP",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "Supplying both position sources is a conflict rather than a precedence, and "
            "asking for more incident positions than exist is refused."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t(
                "test_planar_doe_step.py",
                "test_supplying_both_position_sources_is_a_conflict_not_a_precedence",
            ),
            _t(
                "test_planar_doe_step.py",
                "test_asking_for_more_incident_positions_than_exist_is_refused",
            ),
            _t(
                "test_planar_doe_step.py",
                "test_preserve_energy_is_off_by_default_and_reported_when_on",
            ),
        ),
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="C_PLANAR_DOE_STEP",
        kind=ClaimKind.DEVICE_PARITY,
        claim="The step runs on the declared devices and preserves array residency.",
        oracle=Oracle.CROSS_ROUTE,
        oracle_independence=OracleIndependence.SHARES_CODE,
        evidence=("benchmarks/probes/records/planar_doe_step_device.json",),
        gate_status=GateStatus.CHARACTERIZED_NO_GATE,
        caveats=("No CUDA execution of this coupler is covered by the default gate.",),
    ),
    # -- C_PATCH_WFT ---------------------------------------------------------
    Claim(
        component="C_PATCH_WFT",
        kind=ClaimKind.FORWARD_ACCURACY,
        claim=(
            "A full-aperture patch reproduces the independent ASM at round-off, the "
            "advance is exact rather than paraxial, and enumerating every patch position "
            "is exact rather than merely convergent."
        ),
        oracle=Oracle.DETERMINISTIC_LIMIT,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t(
                "test_patch_wft.py",
                "test_a_full_aperture_patch_reproduces_the_independent_asm_at_roundoff",
            ),
            _t("test_patch_wft.py", "test_the_advance_is_exact_rather_than_paraxial"),
            _t(
                "test_patch_wft.py",
                "test_enumerating_every_patch_position_is_exact_not_merely_convergent",
            ),
        ),
        gate_status=GateStatus.MET,
        stochastic=StochasticEvidence(
            exactness_limit=_t(
                "test_patch_wft.py",
                "test_enumerating_every_patch_position_is_exact_not_merely_convergent",
            ),
        ),
        caveats=(
            "As with C_PLANAR_DOE_STEP, only the exactness limit exists. The sampled "
            "patch estimator has no unbiasedness, convergence or variance evidence.",
            "CPU / NumPy only, and correctly so: core.capabilities declares exactly "
            "that and says declaring CUDA 'would claim a device this operator has "
            "never run on'. There is therefore NO device-parity gap here -- the "
            "constraint is that a GPU composed graph cannot contain this node, which "
            "is a capability limit rather than a validation one.",
        ),
    ),
    Claim(
        component="C_PATCH_WFT",
        kind=ClaimKind.CONVENTION,
        claim=(
            "The SI S3 curvature bound is implemented as an executable precondition, "
            "matches the closed form of eq S9, and is tight where the effect is "
            "observable."
        ),
        oracle=Oracle.ANALYTIC,
        oracle_independence=OracleIndependence.INDEPENDENT,
        evidence=(
            _t("test_curvature_bound.py", "test_bound_matches_the_closed_form_of_eq_s9"),
            _t(
                "test_curvature_bound.py",
                "test_measured_direction_error_stays_under_the_analytic_bound",
            ),
            _t("test_curvature_bound.py", "test_the_bound_is_tight_where_the_effect_is_observable"),
        ),
        gate_status=GateStatus.MET,
    ),
    Claim(
        component="C_PATCH_WFT",
        kind=ClaimKind.STRUCTURED_FAILURE,
        claim=(
            "An even patch is refused rather than silently rounded, and a pad violating "
            "the clearance condition is shown to produce a plausible wrong field."
        ),
        oracle=Oracle.NONE,
        oracle_independence=OracleIndependence.NOT_APPLICABLE,
        evidence=(
            _t("test_patch_wft.py", "test_an_even_patch_is_refused_rather_than_silently_rounded"),
            _t(
                "test_patch_wft.py",
                "test_a_pad_that_violates_clearance_produces_a_plausible_wrong_field",
            ),
            _t("test_patch_wft.py", "test_the_derived_pad_satisfies_all_three_conditions"),
        ),
        gate_status=GateStatus.MET,
    ),
)


# ---------------------------------------------------------------------------
# What is not established, ranked
# ---------------------------------------------------------------------------
#
# This list is the point of the ledger. "Write benchmarks for everything" is not
# a scope; this is. Each entry names the milestone that owes it, so M1, M2 and
# M4 can be traced back here rather than re-derived.

GAPS: tuple[Gap, ...] = (
    Gap(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.FORWARD_ACCURACY,
        gap=(
            "The only forward-accuracy claim that survives on a REAL traced aberrated "
            "system fails its gate at 2.21e-3 against 1.0e-3, and the residual is not "
            "decomposed. The synthetic analytic claims pass and do not transfer."
        ),
        blocks=("L2-PSF-01", "every composed benchmark M4 builds on this edge"),
        severity="high",
        owner="M4.2 (CHE-117)",
        rationale=(
            "high, not critical: an independent oracle IS checking and IS failing, "
            "loudly, at 2.2x. What is missing is an attribution for the residual, not "
            "a check. Ranked above the convergence gaps because a composed benchmark "
            "cannot be built on an edge whose error nobody can decompose."
        ),
    ),
    Gap(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.CONVERGENCE,
        gap=(
            "The frozen M3 configuration samples the Airy radius with 2.44 pixels, so "
            "radius-like metrics measured on it are not converged, and the grid has "
            "never been swept as a convergence variable in the benchmark itself."
        ),
        blocks=("first-null accuracy", "airy_peak_intensity_relative attribution"),
        severity="critical",
        owner="M2.1 (CHE-109)",
        rationale=(
            "CHE-103 measured this and it silently invalidated a 1%-of-unity assertion "
            "that had been reading as a pass. Anything else measured on this grid at "
            "sub-pixel scale is suspect until the sweep exists."
        ),
    ),
    Gap(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.FORWARD_ACCURACY,
        gap=(
            "Off-axis, the reconstruction is 7.5x worse against the analytic Airy oracle "
            "with the production quadrature weight than without it, and on axis it is "
            "not. No explanation is established."
        ),
        blocks=("any off-axis composed benchmark", "the L2-PSF-01 open item"),
        severity="critical",
        owner="M2.1 (CHE-109)",
        rationale=(
            "Found by CHE-103 only because the records were regenerated; no test was "
            "asserting on it. The direction is the wrong way round for a correction "
            "that is supposed to improve the quadrature."
        ),
    ),
    Gap(
        component="C_PLANAR_DOE_STEP",
        kind=ClaimKind.CONVERGENCE,
        gap=(
            "Three of the four stochastic evidence kinds coupler_protocol.yaml requires "
            "are absent: no unbiasedness, no fitted convergence exponent, no variance "
            "characterization. Only the exactness limit exists."
        ),
        blocks=("any claim about the SAMPLED step at a usable ray count",),
        severity="critical",
        owner="M2.3 (CHE-111)",
        rationale=(
            "The textbook critical case: a biased estimator at a usable ray count "
            "would be reported as correct, because the only thing checking is the "
            "enumeration limit, which by construction removes the sampling that "
            "would carry the bias. Nothing independent checks the sampled step."
        ),
    ),
    Gap(
        component="C_PATCH_WFT",
        kind=ClaimKind.CONVERGENCE,
        gap="Same as C_PLANAR_DOE_STEP: exactness limit only, three kinds missing.",
        blocks=("any claim about the SAMPLED patch estimator",),
        severity="critical",
        owner="M2.3 (CHE-111)",
        rationale=(
            "Same as C_PLANAR_DOE_STEP: the exactness limit removes the sampling, so "
            "nothing checks the estimator anyone would actually run."
        ),
    ),
    # C_PLANAR_DOE_STEP's pack gap is CLOSED: knowledge/couplers/planar_doe_step
    # exists with all four required files, is registered in knowledge/README.md,
    # and is held to the standard by tests/test_planar_doe_step_pack.py. The
    # C_PATCH_WFT half of M2.3 is still open, below.
    Gap(
        component="C_PATCH_WFT",
        kind=ClaimKind.CONVENTION,
        gap=(
            "No knowledge pack at all: no card, no conventions, no failure guide, no "
            "theory. It is now the ONLY coupler with a graph node and no pack -- "
            "C_PLANAR_DOE_STEP's was written and this one was deliberately not."
        ),
        blocks=("agent use of this coupler", "M6's planner", "M3.2's discovery API"),
        severity="medium",
        owner="M2.3 (CHE-111)",
        rationale=(
            "A usability gap, not a correctness one: nothing here makes a number "
            "wrong. It does mean an agent asked to build a graph through this node "
            "has nothing to read about its conventions, and the conventions are "
            "exactly what a coupler gets wrong. Deferred rather than forgotten: this "
            "coupler's estimator contract -- the centre density, the unbiasedness "
            "weights, the measured variance reduction -- is being rewritten by "
            "CHE-120, so a pack written against the current state would be wrong on "
            "landing. Write it once that lands."
        ),
    ),
    Gap(
        component="M_WAVE_CHROMATIX",
        kind=ClaimKind.FORWARD_ACCURACY,
        gap=(
            "No INDEPENDENT forward-accuracy oracle. The ASM is checked against our own "
            "float64 propagator built from Chromatix's own kernel_propagate, which "
            "verifies the carrier bookkeeping and cannot verify the propagation."
        ),
        blocks=("any gate that would rest on Chromatix being right",),
        severity="critical",
        owner="M1.2 (CHE-107)",
        rationale=(
            "This is the circular-validation case the repository has a standing rule "
            "about. An analytic diffraction case -- a Gaussian beam's closed-form "
            "propagation, or a knife edge -- would settle it and does not exist."
        ),
    ),
    Gap(
        component="M_RAY_OPTILAND",
        kind=ClaimKind.FORWARD_ACCURACY,
        gap=(
            "No analytic oracle covers a CURVED refracting surface. Flat refraction is "
            "covered -- the exactness limit traces a plane-parallel n=1.5 slab against "
            "a layered plane-wave oracle at 1e-11, and the canonical-prescription tests "
            "check the grating equation and the even-asphere sag series against closed "
            "forms. What has no closed-form check is the traced OPD of a lens."
        ),
        blocks=("every claim that the traced OPD of a real lens is right",),
        severity="critical",
        owner="M1.1 (CHE-106)",
        rationale=(
            "Every composed benchmark starts from a traced OPD map of a CURVED system, "
            "and what validates it today is Optiland's own Wavefront agreeing with our "
            "reference-sphere fit -- two readings of one trace. A paraxial-limit or "
            "Seidel cross-check on the singlet would be independent and does not exist."
        ),
    ),
    Gap(
        component="M_WAVE_CHROMATIX",
        kind=ClaimKind.CONVENTION,
        gap=(
            "jax_enable_x64 is process-global and nothing checks it at the boundary that "
            "depends on it, so the precision of a committed measurement depends on "
            "import order."
        ),
        blocks=("reproducibility of any wave-side record", "M3's executor"),
        severity="high",
        owner="M3.1 (CHE-113)",
        rationale=(
            "CHE-103 established this caused a real, undetected change in committed "
            "evidence. It is CHE-102's rule -- check process-global solver state at the "
            "boundary, do not set it earlier and hope -- applied to the wave solver."
        ),
    ),
    Gap(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.DEVICE_PARITY,
        gap="No CUDA execution of the composed ray->wave path is covered by the default gate.",
        blocks=("any GPU performance or accuracy claim on this edge",),
        severity="medium",
        owner="M0.4 (CHE-105) for measurement, M1.3 (CHE-108) for the gate",
        rationale="Declared in the capability table and extrapolated from CPU.",
    ),
    Gap(
        component="C_RAY_TO_WAVE",
        kind=ClaimKind.GRADIENT,
        gap="No verified derivative across the ray->wave boundary; forward_only.",
        blocks=("inverse design through this edge",),
        severity="low",
        owner="deferred beyond this project phase",
        rationale=(
            "The failure mode is visible rather than silent: the registry says "
            "verified: false and the adapter refuses a gradient request before running."
        ),
    ),
)


# ---------------------------------------------------------------------------
# The ledger, as a projection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def all_claims() -> tuple[Claim, ...]:
    """Every claim this project makes: the registry's projection, then the rest.

    Deferred rather than computed at import, and for a concrete reason: the
    dependency runs family -> ledger-enums, so ``verification.families.schema``
    imports :class:`Oracle` and friends from this module. Projecting at module
    scope would invert that at exactly the wrong moment -- importing
    ``verification.families`` first would find this module's bottom half asking
    for a package that is still executing its own ``__init__``.

    Cached, so that ``CLAIMS`` is one tuple for the life of the process and two
    readers cannot see different ledgers.
    """
    from verification.families.projection import claims_from_families

    return claims_from_families() + _LEGACY_CLAIMS


if TYPE_CHECKING:  # pragma: no cover - the runtime value comes from __getattr__
    #: Declared for the type checker and the linter, both of which cannot see a
    #: PEP 562 module ``__getattr__``. The runtime value is :func:`all_claims`.
    CLAIMS: tuple[Claim, ...]


def __getattr__(name: str) -> object:
    """``CLAIMS`` as a lazily-projected module attribute (PEP 562)."""
    if name == "CLAIMS":
        return all_claims()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def claims_for(component: str) -> tuple[Claim, ...]:
    return tuple(claim for claim in all_claims() if claim.component == component)


def open_gates() -> tuple[Claim, ...]:
    """Every claim whose gate is declared and not met.

    The register M0.3 replaces ``gate_disposition`` prose with. An unmet gate
    that only exists as a paragraph in a manifest is one nobody queries.
    """
    return tuple(claim for claim in all_claims() if claim.is_open_gate)


def coverage() -> dict[str, dict[ClaimKind, tuple[Claim, ...]]]:
    """components x claim kinds. Empty cells are the questions nobody has asked."""
    return {
        component: {
            kind: tuple(c for c in claims_for(component) if c.kind is kind) for kind in ClaimKind
        }
        for component in LEDGER_COMPONENTS
    }
