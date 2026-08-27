"""FIXED-V1: the canonical instances, chosen rather than inherited.

CHE-135 (M4.4). Fixed evaluation stays a first-class capability -- scientific
regression, solver and coupler certification, reproducibility, comparison
between agent versions, paper results, and stable held-out final evaluation all
require it, and none of that is replaced by parameterized families.

What changed is *how the instances are chosen*. The old set was whatever
happened to have been written. Every instance below carries a positive
justification, and **"because it already existed" is not one** -- the same rule
that removed all six A1 tasks while promoting five of their closed forms.

Two tiers, and the split is not cosmetic
-----------------------------------------
``REQUIRED`` is deterministic, CPU, minutes, and enrolled in the default gate.
``EXTENDED`` is expensive, GPU or long-running, and opt-in. A required gate
nobody can afford to run is not a gate.

What this suite proves about itself
------------------------------------
* every one of the seven ``VerificationStatus`` values is the expected outcome
  of at least one instance, so the suite demonstrates the distinctions it claims
  to make rather than asserting them;
* every component in ``core.capabilities.COMPONENT_CAPABILITIES`` appears in at
  least one REQUIRED instance whose family has an independent oracle. That is
  success metric S1, made executable;
* every instance is ``TEST_HELDOUT``. M9 constructs the train/validation axes;
  this issue tags everything held out and stops.

Where the instances live
------------------------
Declared here, in one reviewable place, because the *selection* is the
deliverable -- "a canonical fixed suite chosen, not inherited" is a statement
about a list, and scattering it across thirty-two near-identical YAML files
would make the choice harder to review rather than easier. The generated
manifest at ``benchmarks/instances/FIXED-V1.yaml`` is the committed, versioned
artifact and is checked against this module by
``tests/test_fixed_suite.py``. That is a deliberate deviation from the ticket's
per-instance-file layout and it is recorded rather than quiet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.capabilities import COMPONENT_CAPABILITIES
from verification.families import BenchmarkFamily, BenchmarkInstance, family
from verification.status import VerificationStatus

__all__ = [
    "FIXED_V1",
    "FIXED_V1_VERSION",
    "FixedInstance",
    "FixedSuite",
    "Tier",
]

FIXED_V1_VERSION = "1.0.0"

#: Every instance in this suite. M9 constructs the train/validation axes; this
#: issue tags everything held out.
HELDOUT = "test_heldout"


class Tier(StrEnum):
    """Which collection an instance belongs to."""

    #: Deterministic, CPU, minutes. Enrolled in the default gate.
    REQUIRED = "required"
    #: Expensive, GPU, or long-running. Opt-in, following the precedent the GPU
    #: and tutorial suites already set.
    EXTENDED = "extended"


@dataclass(frozen=True)
class FixedInstance:
    """One canonical instance, with the reason it was selected.

    ``justification`` is checked by a test for the shapes that are not
    justifications. It has to say what this particular physical setup buys that
    another point in the same family would not.
    """

    instance: BenchmarkInstance
    tier: Tier
    justification: str
    #: What the verifier is expected to report. ``None`` where the outcome is
    #: the measurement rather than the status.
    expected_status: VerificationStatus | None = None

    def __post_init__(self) -> None:
        if self.instance.split_tag != HELDOUT:
            raise ValueError(
                f"{self.instance.instance_id}: FIXED-V1 instances are all "
                f"{HELDOUT!r}; this one is {self.instance.split_tag!r}"
            )
        if len(self.justification.split()) < 12:
            raise ValueError(
                f"{self.instance.instance_id}: a justification has to say what THIS "
                "setup buys that another point in the family would not"
            )


@dataclass(frozen=True)
class FixedSuite:
    version: str
    instances: tuple[FixedInstance, ...]

    def tier(self, tier: Tier) -> tuple[FixedInstance, ...]:
        return tuple(f for f in self.instances if f.tier is tier)

    def by_id(self, instance_id: str) -> FixedInstance:
        for fixed in self.instances:
            if fixed.instance.instance_id == instance_id:
                return fixed
        raise KeyError(f"no instance {instance_id!r} in FIXED-{self.version}")

    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for fixed in self.instances:
            if fixed.instance.family_id not in seen:
                seen.append(fixed.instance.family_id)
        return tuple(seen)

    def components_in(self, tier: Tier) -> frozenset[str]:
        return frozenset(
            component
            for fixed in self.tier(tier)
            for component in family(fixed.instance.family_id).components
        )

    def as_manifest(self) -> dict[str, Any]:
        """The committed, versioned artifact. Generated, never hand-edited."""
        return {
            "suite": "FIXED-V1",
            "version": self.version,
            "instances": [
                {
                    "instance_id": fixed.instance.instance_id,
                    "family_id": fixed.instance.family_id,
                    "family_version": fixed.instance.family_version,
                    "tier": fixed.tier.value,
                    "split_tag": fixed.instance.split_tag,
                    "fingerprint": fixed.instance.fingerprint,
                    "validity_status": fixed.instance.validity_status.value,
                    "expected_status": (
                        fixed.expected_status.value if fixed.expected_status else None
                    ),
                    "justification": " ".join(fixed.justification.split()),
                }
                for fixed in self.instances
            ],
        }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _held_out(
    family_id: str,
    instance_id: str,
    parameters: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> BenchmarkInstance:
    fam: BenchmarkFamily = family(family_id)
    return fam.instantiate(instance_id, parameters, split_tag=HELDOUT, seed=seed)


def _existing(family_id: str, instance_id: str) -> BenchmarkInstance:
    """A canonical instance the family already declares, re-tagged held out.

    B0's ten were authored with the family because each one exists to fire a
    specific declared code, and a code needs an instance rather than a region.
    Re-tagging rather than re-declaring keeps one definition.
    """
    fam = family(family_id)
    declared = next(i for i in fam.canonical_instances if i.instance_id == instance_id)
    return fam.instantiate(
        instance_id, declared.parameters, split_tag=HELDOUT, expected=declared.expected
    )


_B0 = (
    FixedInstance(
        instance=_existing("B0-DTYPE", "B0-DTYPE-01"),
        tier=Tier.REQUIRED,
        justification=(
            "the only case in the repository where a precision request cannot be "
            "honoured and the loss is a measurable number rather than a warning. "
            "2.5e-5 at z = 40 um, against a closed-form bound of one eps32 per radian."
        ),
        expected_status=VerificationStatus.LOSSY_BUT_ALLOWED,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-CAPINT-01"),
        tier=Tier.REQUIRED,
        justification=(
            "the one route with provably NO executable precision at all: "
            "C_PATCH_WFT computes only in complex128, Chromatix only in complex64. "
            "Project risk R5, and an agent will propose exactly this."
        ),
        expected_status=VerificationStatus.UNSUPPORTED,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-DEVICE-01"),
        tier=Tier.REQUIRED,
        justification=(
            "CUDA asked of a coupler whose probe-backed capability table declares CPU "
            "only. Distinct from B0-CAPINT-01: one device is missing, not every "
            "precision, and the remedy differs."
        ),
        expected_status=VerificationStatus.UNSUPPORTED,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-DEVICE-02"),
        tier=Tier.REQUIRED,
        justification=(
            "declared cuda with actual cpu placement. A process-global JAX platform "
            "pin produces a successful run on the host while the caller asked for "
            "CUDA, and nothing raises -- the case that motivated reading the device "
            "off the array."
        ),
        expected_status=VerificationStatus.INVALID_CONFIGURATION,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-META-01"),
        tier=Tier.REQUIRED,
        justification=(
            "a required declaration simply absent, which is the plainest form of an "
            "invalid configuration and the baseline the other four are read against."
        ),
        expected_status=VerificationStatus.INVALID_CONFIGURATION,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-HANDOFF-01"),
        tier=Tier.REQUIRED,
        justification=(
            "the only BLOCKED instance: the coupler could proceed and refuses to, "
            "because a bare opd_native is an absolute path whose zero moves with the "
            "aperture. Nothing about the request is malformed."
        ),
        expected_status=VerificationStatus.BLOCKED,
    ),
    FixedInstance(
        instance=_existing("B0-CONTRACT", "B0-PATCH-01"),
        tier=Tier.REQUIRED,
        justification=(
            "a quadrature weight applied to a sampling whose ring structure does not "
            "exist. The code runs and the weights mean nothing, which is out of "
            "validity rather than unsupported."
        ),
        expected_status=VerificationStatus.OUT_OF_VALIDITY,
    ),
    FixedInstance(
        instance=_existing("B0-VALIDITY", "B0-VALIDITY-01"),
        tier=Tier.REQUIRED,
        justification=(
            "eps_curv = 0.2 against a bound of arcsin(1/20) = 0.05004. A declared "
            "physical bound crossed, with a signed margin, rather than a capability "
            "absent."
        ),
        expected_status=VerificationStatus.OUT_OF_VALIDITY,
    ),
    FixedInstance(
        instance=_existing("B0-UNITS", "B0-UNITS-01"),
        tier=Tier.REQUIRED,
        justification=(
            "it runs clean and the physics is wrong. The coated reflectance comes back "
            "1.7e-5 from BARE GLASS and nothing raises -- the measured wrong number is "
            "the artifact, and it proves the verifier can tell 'it executed' from "
            "'it is right'."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_existing("B0-UNITS", "B0-UNITS-02"),
        tier=Tier.REQUIRED,
        justification=(
            "the same shape on the wave side and a different mechanism: one parameter "
            "name, two units a factor of 2*pi apart, and a displacement opposite in "
            "sign to the parameter. Two instances because two APIs fail differently."
        ),
        expected_status=VerificationStatus.OK,
    ),
)


_B1_RAY = (
    FixedInstance(
        instance=_held_out(
            "B1-RAY-EFL",
            "B1-RAY-EFL-01",
            {
                "radius_mm": 25.0,
                "index": 1.5168,
                "thickness_mm": 4.0,
                "wavelength_um": 0.5876,
                "marginal_ray_angle_rad": 0.01,
                "pupil_rings": 32,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "the thick-lens correction is what makes this more than arithmetic: EFL "
            "and BFL differ by 2.64 mm here, so an implementation that reports the "
            "same number twice fails the second check and only the second."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-RAY-PLATE",
            "B1-RAY-PLATE-01",
            {
                "thickness_mm": 10.0,
                "index": 1.6,
                "focal_length_mm": 100.0,
                "marginal_ray_angle_rad": 0.005,
                "axis_crossing_samples": 64,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "the sign is the claim. A plate in a converging beam moves the focus AWAY "
            "from it, and this configuration puts the wrong answer, -3.75 mm, two "
            "tolerances from the right one."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-RAY-SNELL",
            "B1-RAY-SNELL-01",
            {
                "index_incident": 1.5,
                "index_transmitted": 1.0,
                # 0.6 rad, against a critical angle of asin(1/1.5) = 0.7297. Chosen
                # NEAR the boundary rather than comfortably inside it.
                "incidence_angle_rad": 0.6,
                "device": "cpu",
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "placed at 82% of the critical angle rather than comfortably inside it, so "
            "the instance exercises the TIR predicate's margin near where it matters "
            "instead of somewhere the answer is obvious."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-RAY-LAGRANGE",
            "B1-RAY-LAGRANGE-01",
            {
                "index_object_space": 1.0,
                "marginal_ray_angle_rad": 0.02,
                "marginal_ray_height_mm": 0.0,
                "chief_ray_angle_rad": 0.01,
                "chief_ray_height_mm": 5.0,
                "surface_count": 3,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "three surfaces rather than one, because a conservation law through a "
            "single refraction is arithmetic and through a system is a statement "
            "about the transfer. The cheapest whole-system check available."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-RAY-OFFAXIS-OPL",
            "B1-RAY-OFFAXIS-OPL-01",
            {
                "field_angle_rad": 0.2,
                "pupil_diameter_m": 0.02,
                "wavelength_m": 5.5e-7,
                "index_object_space": 1.0,
                "pupil_rings": 32,
                "prescription": "M3-REVERSE-TELEPHOTO",
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "the highest-value instance in B1. The omitted n_object * (d0 . r_launch) "
            "is linear in the launch coordinate, so on axis it is a constant that "
            "cancels -- which is why the defect survived CHE-30, CHE-32 and CHE-33, "
            "all of which looked on axis. Hy = 0.2 on M3-REVERSE-TELEPHOTO is the "
            "exact configuration where 0.13% of the required tilt survived."
        ),
        expected_status=VerificationStatus.OK,
    ),
)


_B1_WAVE = (
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-GAUSS",
            "B1-WAVE-GAUSS-01",
            {
                "waist_um": 5.0,
                "distance_um": 100.0,
                "wavelength_um": 0.532,
                "grid_n": 2048,
                "sample_pitch_um": 0.25,
                "device": "cpu",
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "z = 100 um is 1.6 Rayleigh ranges for a 5 um waist, so the beam has "
            "genuinely spread: the discriminating wrong answer, the unpropagated "
            "5.0 um waist, sits 17% away and the tolerance separates them cleanly."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-AIRY",
            "B1-WAVE-AIRY-01",
            {
                "numerical_aperture": 0.05,
                "wavelength_um": 0.532,
                "grid_n": 1024,
                "focal_plane_pitch_um": 0.83,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "the frozen 0.83 um focal-plane pitch, deliberately: it is the sampling at "
            "which the null lands between samples and the measured radius reads 2.3% "
            "high. Freezing the UNDER-sampled point is what makes the convergence "
            "ladder mean something."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-TILT",
            "B1-WAVE-TILT-01",
            {
                "tilt_rad": 0.08726646259971647,
                "distance_um": 200.0,
                "wavelength_um": 0.532,
                "grid_n": 512,
                "sample_pitch_um": 0.5,
                "tilt_encoding": "explicit_phase_ramp",
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "5 degrees over 200 um, the configuration the kykx hazard was measured on. "
            "Frozen on the explicit-phase-ramp encoding so the kykx_argument arm has a "
            "fixed point to be compared against."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-PLANEPHASE",
            "B1-WAVE-PLANEPHASE-01",
            {
                # NONZERO on purpose: on axis k_z = k and a 2*pi frequency-grid
                # error is invisible.
                "transverse_frequency_per_um": 0.5,
                # 20 um against a sampling limit of 256 * 0.0625 / 0.532 = 30.1 um.
                "distance_um": 20.0,
                "wavelength_um": 0.532,
                "medium_index": 1.0,
                "grid_n": 256,
                "sample_pitch_um": 0.25,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "the transverse frequency is nonzero on purpose. On axis k_z = k and a "
            "2*pi frequency-grid error is invisible -- the same blind spot as the ray "
            "family's on-axis case, one representation over."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-FWDBWD",
            "B1-WAVE-FWDBWD-01",
            {
                # 40 um against a sampling limit of 512 * 0.0625 / 0.532 = 60.2 um.
                "distance_um": 40.0,
                "wavelength_um": 0.532,
                "aperture_fill_fraction": 0.4,
                "grid_n": 512,
                "sample_pitch_um": 0.25,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "0.4 fill so the propagated field does not wrap, which is the condition "
            "under which the round trip is exact at all. Two propagations and a norm: "
            "the cheapest instance in the suite and it catches a phasor-sign error."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-TALBOT",
            "B1-WAVE-TALBOT-01",
            {
                "period_um": 20.0,
                "wavelength_um": 0.532,
                "duty_cycle": 0.5,
                "talbot_order": 1,
                # z_T / limit = 2 * samples_per_period / periods_across_grid, so the
                # window has to hold at least twice as many periods as it samples one
                # period with. 128 against 32 puts the revival at half the limit.
                "periods_across_grid": 128,
                "samples_per_period": 32,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "new to this repository: nothing tests periodic self-imaging. A revival at "
            "2 d^2 / lambda is a strong independent check on propagator phase that no "
            "existing probe covers, and it is structurally unlike everything else here "
            "-- which is what makes it the natural holdout axis later."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B1-WAVE-ASM-VALIDITY",
            "B1-WAVE-ASM-VALIDITY-01",
            {
                "waist_um": 5.0,
                # 3x the sampling limit of 512 * 0.0625 / 0.532 = 60.2 um.
                "distance_um": 180.6,
                "wavelength_um": 0.532,
                "grid_n": 512,
                "sample_pitch_um": 0.25,
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "placed at three times the sampling limit, so the instance is the failing "
            "side of the boundary rather than the passing one. It must report "
            "out_of_validity while producing a plausible-looking field, which is the "
            "whole subject and cannot be shown from inside the domain."
        ),
        expected_status=VerificationStatus.OUT_OF_VALIDITY,
    ),
)


_B2 = (
    FixedInstance(
        instance=_held_out(
            "B2-R2W-EXACT",
            "B2-R2W-EXACT-01",
            {
                "wavelength_m": 5.5e-7,
                "handoff_plane_z_m": 6.814345991561233e-05,
                "grid_n": 64,
                "target_sample_pitch_m": 2.6587352810843895e-06,
                "sample_alignment": "on_node",
                "dtype": "complex128",
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "on-node and float64, which is the only configuration in which an "
            "exactness claim is meaningful at all. A 64-point grid keeps the "
            "enumeration inside the required gate's budget while still being an "
            "enumeration."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B2-R2W-ROUTE",
            "B2-R2W-ROUTE-01",
            {
                "system": "demo3_characterization",
                "wavelength_m": 5.32e-7,
                "oversampling": 8,
                "ray_count": 1_000_000,
                "route": "kspace_splat",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "demo3 at 8x oversampling on the k-space route: the configuration where "
            "the fast route loses 1.7% of the power while agreeing to 7.1e-13 on "
            "demo2. The asymmetry is the finding, and it only exists at this point."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B2-W2R-STOCH",
            "B2-W2R-STOCH-01",
            {
                "wavelength_m": 5.32e-7,
                "numerical_aperture": 0.5,
                "sample_count": 1_000_000,
                "seed": 20260825,
            },
            seed=20260825,
        ),
        tier=Tier.REQUIRED,
        justification=(
            "NA 0.5 puts a substantial fraction of the spectrum near the light cone, "
            "so the evanescent accounting is exercised rather than trivially zero. The "
            "declared seed is one member of the ensemble the family requires, not the "
            "result."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B2-EQUIV",
            "B2-EQUIV-01",
            {
                "aperture_width_m": 1e-3,
                "substrate_radius_m": float("inf"),
                "wavelength_m": 5.32e-7,
                "patch_count": 1,
                "grid_snapping": "exact",
                "pad_width": 566,
            },
        ),
        tier=Tier.REQUIRED,
        justification=(
            "patch_count = 1 is the full-aperture case, which is the one point where "
            "the decomposition and the global computation must agree to round-off "
            "(7.1e-13, measured against an independent float64 ASM). It is the anchor "
            "the sub-aperture granularities are read against."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B2-ROUNDTRIP",
            "B2-ROUNDTRIP-01",
            {
                "wavelength_m": 5.32e-7,
                "numerical_aperture": 0.3,
                "grid_n": 64,
                "sample_count": 1_000_000,
                "direction": "wave_ray_wave",
                "arm": "enumerated",
                "seed": 20260825,
                "broken_twin_ran": True,
            },
            seed=20260825,
        ),
        tier=Tier.EXTENDED,
        justification=(
            "the enumerated arm with its broken twin declared as having run -- the "
            "only configuration in which the family reports anything at all, because "
            "a round trip whose twin was not executed is FAR_OUTSIDE its own validity "
            "domain. 1.32e-15 against the twin's 1.40."
        ),
        expected_status=VerificationStatus.OK,
    ),
)


_B3 = (
    FixedInstance(
        instance=_held_out(
            "B3-PSF-SINGLET",
            "B3-PSF-SINGLET-01",
            {
                "prescription": "M3-SINGLET-REF",
                "field_angle_rad": 0.0,
                "wavelength_m": 5.5e-7,
                "numerical_aperture": 0.05171631827291936,
                "pupil_rings": 512,
                "grid_n": 188,
                "pad_width": 566,
                "quadrature_weight": "weighted",
                "device": "cpu",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "the frozen 512-ring configuration, which is the point every number in the "
            "residual investigation was measured at. Freezing it is what keeps CHE-117 "
            "anchored: a change in the 2.21e-3 becomes attributable to code rather "
            "than to a moved prescription."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B3-DEMO2",
            "B3-DEMO2-01",
            {
                "phase_profile": "demo2_smile",
                "wavelength_m": 5.32e-7,
                "ray_count": 160_000_000,
                "reconstruction_route": "ramp_sum",
                "device": "cuda",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "the paper's own Table S2 budget and the paper's own configuration, graded "
            "against verification/asm_oracle.angular_spectrum_float64 -- an oracle "
            "independent of both couplers under test but NOT external to this "
            "repository. CHE-116 corrected this justification: it read 'against the "
            "paper's own published figure ... the only composed case graded against "
            "something outside this repository', and no such comparison exists. The "
            "reason to freeze it survives the correction -- a different algorithm on "
            "the paper's setup at the paper's budget, so a change in the number is "
            "attributable to our code rather than to a moved setup, which is what "
            "makes it the regression anchor for the ray-wave path."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B3-DUALROUTE",
            "B3-DUALROUTE-01",
            {
                "field_angle_deg": 20.0,
                "wavelength_m": 5.5e-7,
                "pupil_rings": 64,
                "route": "ray_to_wave",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "20 degrees rather than on axis: it is the field angle at which the three "
            "routes diverge, and where the pupil anisotropy that explains the "
            "divergence is measurable. On axis all three sit at the resampling floor "
            "and the instance would say nothing."
        ),
        expected_status=VerificationStatus.OK,
    ),
)


_B4 = (
    FixedInstance(
        instance=_held_out(
            "B4-DEMO3",
            "B4-DEMO3-01",
            {
                "phase_profile": "demo3_smile",
                "wavelength_m": 5.32e-7,
                "ray_count": 40_000_000,
                "oversampling": 8,
                "seed": 20260825,
                "seed_count": 8,
                "reconstruction_route": "ramp_sum",
            },
            seed=20260825,
        ),
        tier=Tier.EXTENDED,
        justification=(
            "40M rays is the top rung of the measured 20/30/40M ladder, which is where "
            "the log-log slope is anchored and where the 1.49e9-ray extrapolation "
            "starts. Below it the seed-to-seed NCC is too small to fit a trend to."
        ),
        expected_status=VerificationStatus.UNCONVERGED,
    ),
    FixedInstance(
        instance=_held_out(
            "B4-DUALROUTE-AGREEMENT",
            "B4-DUALROUTE-AGREEMENT-01",
            {
                "field_angle_deg": 20.0,
                "wavelength_m": 5.5e-7,
                "pupil_rings": 64,
                "route_pair": "huygens_vs_ray_to_wave",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "the huygens-versus-ray-to-wave pair at 20 degrees: the ONE comparison of "
            "the three whose two legs do not share a Wavefront/OPD front end, and the "
            "one that agrees to 0.0138 while FFTPSF sits at 0.313. Freezing this pair "
            "rather than the FFT one is what keeps the attributed anisotropy finding "
            "readable."
        ),
        expected_status=VerificationStatus.OK,
    ),
    FixedInstance(
        instance=_held_out(
            "B4-COST",
            "B4-COST-01",
            {
                "workload": "l2_psf_01",
                "ray_count": 787_969,
                "route": "ramp_sum",
                "device": "cpu",
            },
        ),
        tier=Tier.EXTENDED,
        justification=(
            "the singlet workload on CPU at the frozen ray count, which is the one "
            "configuration whose cost baseline is already committed and whose "
            "environment fingerprint a later run can be compared against."
        ),
        expected_status=VerificationStatus.OK,
    ),
)


FIXED_V1 = FixedSuite(
    version=FIXED_V1_VERSION,
    instances=_B0 + _B1_RAY + _B1_WAVE + _B2 + _B3 + _B4,
)


def components_without_required_coverage() -> frozenset[str]:
    """Success metric S1, as a query.

    Every component should appear in at least one REQUIRED instance whose family
    has an independent oracle. Returned rather than asserted so the gap, if
    there is one, is a list a reader can act on.
    """
    covered = {
        component
        for fixed in FIXED_V1.tier(Tier.REQUIRED)
        for component in family(fixed.instance.family_id).components
        if family(fixed.instance.family_id).oracle.may_decide_correctness
    }
    return frozenset(COMPONENT_CAPABILITIES) - covered


__all__ += ["HELDOUT", "components_without_required_coverage"]
