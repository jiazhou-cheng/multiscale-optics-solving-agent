"""B2-EQUIV: the same physics at two granularities, executed in both directions.

CHE-111 (M2.3). ``C_PATCH_WFT`` and ``C_PLANAR_DOE_STEP`` are the same operator at
two granularities and SI S10 is explicit about which is which -- the patch-based
local windowed Fourier transform is the *direct implementation* and the global
single-plane aggregation is the *shortcut*. Two implementations of one operator,
one of which is the limit of the other, is a genuine equivalence relation rather
than a self-comparison, and it is testable in both directions.

The equivalence had been *measured* and never packaged as a benchmark with a
gate. This packages it.

Both directions, and what each one can and cannot see
----------------------------------------------------
**Full aperture.** One patch covering the whole aperture IS the window, so the
two routes coincide and the residual is float64 round-off against the
independent float64 ASM. That is the anchor, and the clearance exemption on it is
legitimate and preserved: padding a full-aperture single patch moves the mode
grid off the unpadded oracle's, and the anchor then reads 0.57 instead of
1.4e-12. That is not a defect to pad away.

The anchor is also BLIND to two defects that are exactly 1 there -- the coverage
correction and the launch phase -- which is how an inverted coverage correction
(``A_patch/A_draw`` for ``A_draw/A_patch``) once survived. The sub-aperture
instances are what see them, and they are gated there.

**Sub aperture.** Many patches coherently summed must converge to the full-DOE
response, and the partition-of-unity argument behind that convergence is exactly
what an apodization taper breaks: any window below 1 removes field that no other
patch replaces. Four patch counts, a fitted trend, and every score naming its
oracle's pad -- because an oracle is not well defined until its padding is. The
same route scored against a pad-200 oracle reads 8.8e-3 and against a pad-101
oracle 0.33, and neither is an error in either implementation; both are
wraparound between two periods.

Run it::

    ./run.sh python benchmarks/instances/b2_equiv.py --write
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from core.boundary import ReferencePlane
from core.paths import repository_root
from couplers.patch import (
    PatchPlan,
    Substrate,
    advance_bundle_to_plane,
    patch_secondary_rays,
    plan_patches,
    resolve_pad_px,
)
from couplers.ray_to_wave import Projection, ray_to_wave
from runtime.instance_runner import probe_refusal, record_from_probe
from verification.asm_oracle import angular_spectrum_float64, compare_fields
from verification.evidence import (
    InstanceRun,
    control_result,
    fit_convergence,
    write_instance_record,
)
from verification.families.b2_transitions import B2_EQUIV
from verification.metrics import ncc, power_ratio
from verification.result import (
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    UncertaintyBasis,
)
from verification.verifier import verify

__all__ = [
    "declared_instance_ids",
    "noise_limited_relation",
    "route_agreement",
    "run_all",
    "run_instance",
]

ROOT = repository_root()

#: The CHE-96 anchor configuration, kept exactly: a 33x33 random complex DOE at
#: 6.3 um pitch, lambda = 0.7 um, z = 1.26 mm. Changing any of it would make the
#: 1.4e-12 anchor a different measurement.
ANCHOR_N = 33
PITCH_M = 6.3e-6
WAVELENGTH_M = 0.7e-6
Z_M = 1.26e-3
DOE_PLANE = ReferencePlane(name="doe", z_m=0.0)
SENSOR_PLANE = ReferencePlane(name="sensor", z_m=Z_M)

#: The sub-aperture grid. Deliberately 15 px rather than the anchor's 33: the
#: reconstruction is O(rays x pixels) and the separable contraction allocates
#: rays x n factors, so the enumeration at 33 px is 3.7 M rays and about 4 GB in
#: one call -- which pushed this shared machine into swap while CHE-96 was being
#: written. At 15 px the whole enumeration is 159 k rays and about 76 MB. The
#: cost curve belongs in a probe, not in a gate.
SUB_N = 15


def _instance(instance_id: str) -> Any:
    for candidate in B2_EQUIV.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"B2-EQUIV declares no instance {instance_id!r}")


def _doe(n: int, seed: int = 20260822) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex128)


def _reconstruct(bundle, *, n: int, plane: ReferencePlane) -> np.ndarray:
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(n, n),
        sample_pitch_m=(PITCH_M, PITCH_M),
        plane=plane,
        projection=Projection.ASM_CONSISTENT,
    )
    return np.asarray(field.u)


def _global_route(doe: np.ndarray) -> np.ndarray:
    """The independent float64 ASM: the oracle, and not this module's arithmetic.

    ``verification/asm_oracle.angular_spectrum_float64`` shares no code with the
    patch route. Using the patch route to check itself would be exactly the
    circular validation the repository's rules forbid, and it is why the pad the
    oracle runs at has to be named in every score: the same route against a
    pad-200 oracle reads 8.8e-3 and against a pad-101 oracle 0.33, and neither is
    an error in either implementation.
    """
    return angular_spectrum_float64(
        doe, wavelength_m=WAVELENGTH_M, sample_pitch_m=PITCH_M, z_m=Z_M
    )


# ---------------------------------------------------------------------------
# The full-aperture anchor
# ---------------------------------------------------------------------------


def _full_aperture(*, apodize: bool = False) -> dict[str, Any]:
    doe = _doe(ANCHOR_N)
    plan = plan_patches(
        grid_shape=(ANCHOR_N, ANCHOR_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        patch_px=ANCHOR_N,
        # pad_factor 1 is the CLEARANCE EXEMPTION and it is deliberate. A padded
        # full-aperture single patch has its modes on a different grid from the
        # unpadded oracle's, and the anchor then reads 0.57. The exemption is a
        # property of the full-aperture limit rather than a relaxation.
        pad_factor=1,
        patch_count=None,
        substrate=Substrate.PLANAR,
    )
    working = doe * _taper(ANCHOR_N) if apodize else doe
    rays, diagnostics = patch_secondary_rays(
        working,
        plan=plan,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=None,
    )
    reconstructed = _reconstruct(
        advance_bundle_to_plane(rays, target=SENSOR_PLANE), n=ANCHOR_N, plane=SENSOR_PLANE
    )
    comparison = compare_fields(reconstructed, _global_route(doe))
    return {
        "plan": plan,
        "diagnostics": diagnostics,
        "comparison": comparison,
        "relative_l2": comparison.raw_relative_field_error,
        "energy_residual": comparison.energy_residual,
        "coverage": plan.coverage,
        "pad_px": plan.pad_px,
        "propagating_modes": diagnostics.propagating_modes,
        "reconstructed": reconstructed,
    }


def _taper(n: int) -> np.ndarray:
    """A raised-cosine window. Any taper below 1 breaks the partition of unity."""
    axis = np.hanning(n)
    return np.outer(axis, axis)


# ---------------------------------------------------------------------------
# The sub-aperture ladder
# ---------------------------------------------------------------------------

_SUB_CACHE: dict[tuple[int, str], dict[str, Any]] = {}


def _sub_aperture(
    patch_count: int,
    *,
    variant: str = "correct",
) -> dict[str, Any]:
    """Many patches, coherently summed, against the same independent oracle.

    ``variant`` names the mutation, and each one is a real switch rather than a
    hand-written variant of the algorithm:

    ``correct``      grid-snapped centres over the dilated aperture, no taper
    ``apodized``     a raised-cosine taper, which removes field no other patch
                     replaces and must break convergence
    ``continuous``   centres NOT snapped to the sample grid, which injects a
                     sub-sample linear phase and plateaus instead of converging
    ``uncovered``    the coverage correction omitted, invisible at full aperture
    ``launch``       the launch-position phase double-counted, likewise invisible
    """
    key = (patch_count, variant)
    if key in _SUB_CACHE:
        return _SUB_CACHE[key]

    patch_px = 5
    doe = _doe(SUB_N, seed=20260823)
    working = doe * _taper(SUB_N) if variant == "apodized" else doe

    dilation = patch_px // 2
    enumerated_positions = (SUB_N + 2 * dilation) ** 2
    if patch_count == enumerated_positions:
        # Every draw position exactly once: the estimator's EXPECTATION rather
        # than a sample of it. That separates "is it unbiased" from "how fast does
        # it converge", and only the first is a gate -- a biased estimator
        # converges to the wrong answer, which a rate cannot distinguish from
        # slow convergence.
        plan = _enumerated_plan(patch_px=patch_px, dilation=dilation)
    else:
        plan = plan_patches(
            grid_shape=(SUB_N, SUB_N),
            sample_pitch_m=(PITCH_M, PITCH_M),
            patch_px=patch_px,
            pad_factor=1,
            patch_count=patch_count,
            substrate=Substrate.PLANAR,
            rng=np.random.default_rng(20260823),
        )
    if variant == "continuous":
        plan = _unsnap(plan)

    rays, diagnostics = patch_secondary_rays(
        working,
        plan=plan,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=None,
    )
    if variant == "uncovered":
        rays = _invert_coverage(rays, plan)
    elif variant == "launch":
        rays = _double_launch_phase(rays)

    # Compared AT THE DOE PLANE, and the choice is the substantive one in this
    # whole file. See `_sub_aperture_plane_finding`: at z = 1.26 mm the two routes
    # disagree at 0.84 and it is not a defect in either -- a sub-aperture patch's
    # modes live on its own pad-21 grid, which is not commensurate with the 15-px
    # reconstruction grid, so the ray sum is the NON-periodic propagated field
    # while the ASM is the periodic one. At full aperture with pad_factor 1 the
    # two mode sets coincide exactly, which is why the anchor reads 1.4e-12 and
    # this comparison cannot.
    #
    # At z = 0 the relation is the decomposition identity and mode
    # commensurability does not enter: the patch sum reproduces the field
    # pointwise. That is what SI S2's convergence relation is about, and the
    # oracle is still the independent float64 ASM -- evaluated at zero distance,
    # so it applies the same evanescent cut rather than being the identity.
    reconstructed = _reconstruct(rays, n=SUB_N, plane=DOE_PLANE)
    reference = angular_spectrum_float64(
        doe, wavelength_m=WAVELENGTH_M, sample_pitch_m=PITCH_M, z_m=0.0
    )
    comparison = compare_fields(reconstructed, reference)
    out = {
        "patch_count": patch_count,
        "variant": variant,
        "plan": plan,
        "coverage": plan.coverage,
        "pad_px": plan.pad_px,
        "relative_l2": comparison.raw_relative_field_error,
        "piston_aligned": comparison.piston_aligned_relative_field_error,
        "energy_residual": comparison.energy_residual,
        "power_ratio": power_ratio(reconstructed, reference),
        "ncc": ncc(np.abs(reconstructed) ** 2, np.abs(reference) ** 2),
        "ray_count": int(rays.count),
        "propagating_modes": diagnostics.propagating_modes,
        "reconstructed": reconstructed,
    }
    _SUB_CACHE[key] = out
    return out


def _enumerated_plan(*, patch_px: int, dilation: int):
    """Every patch centre of the dilated aperture, on the sample grid, once.

    Built directly rather than drawn, because ``plan_patches`` draws: the
    coverage here is the exact ratio of the draw region's area to the patch's,
    and the whole point is that no sampling enters.
    """
    index = np.arange(-(SUB_N // 2 + dilation), SUB_N // 2 + dilation + 1)
    grid_y, grid_x = np.meshgrid(index, index)
    centers = np.column_stack([grid_x.ravel() * PITCH_M, grid_y.ravel() * PITCH_M])
    return PatchPlan(
        centers_xy_m=centers,
        patch_px=patch_px,
        pad_px=resolve_pad_px(
            grid_n=SUB_N,
            patch_px=patch_px,
            pad_factor=1,
            max_center_px=SUB_N // 2 + dilation,
        ),
        coverage=float(index.size**2 / (patch_px * patch_px)),
        dilation_px=dilation,
        curvature_bound_rad=0.0,
    )


def _unsnap(plan: Any):
    """Move the patch centres off the sample grid by half a sample.

    A declared negative control of the family: continuous centres inject a
    sub-sample linear phase and the convergence sweep plateaus at ~0.28 instead
    of converging. The mutation is applied to the PLAN the shipping emitter
    consumes, so the emitter itself is unmodified.
    """
    from dataclasses import replace

    # Half a SAMPLE, expressed in metres, so the centres leave the sample grid.
    centers = np.asarray(plan.centers_xy_m, dtype=np.float64) + 0.5 * PITCH_M
    return replace(plan, centers_xy_m=centers)


def _invert_coverage(rays, plan):
    """Invert the coverage correction: ``A_patch/A_draw`` for ``A_draw/A_patch``.

    The shipping correction is ``A_draw / A_patch`` -- ``plan.coverage``, which is
    ``draw_positions / patch_px**2`` -- and ``patch_secondary_rays`` multiplies
    the emitted amplitude by it exactly once. So turning ``coverage * X`` into
    ``X / coverage`` is a factor of ``coverage ** -2``, not ``coverage ** 2``.

    The first version of this control had the exponent's SIGN wrong and squared
    the correction instead of inverting it. It still separated -- by 2.1e2 rather
    than the correct 0.995 -- so the control reported FIRED and the record
    attributed a squared correction to the historical inversion. That is worth
    recording rather than quietly fixing: a control that fires is not thereby a
    control that ran the mutation it claims, and the reason this one could go
    wrong is structural. It mutates the emitted amplitudes from OUTSIDE the
    emitter, so the arithmetic has to be re-derived at the call site, whereas the
    ray/wave controls flip a switch on ``Perturbation`` / ``SamplingPerturbation``
    inside the shipping path and cannot be off by a power.

    Exactly 1 at full aperture, which is how the real inversion survived.
    """
    coverage = float(plan.coverage) or 1.0
    return _with_amplitude(rays, np.asarray(rays.amplitude) * coverage**-2)


def _double_launch_phase(rays):
    """Apply the launch-position phase twice.

    Also exactly 1 at full aperture -- a single patch centred on the origin has
    zero launch offset -- and also a real defect that survived because of it.
    """
    positions = np.asarray(rays.positions_m)
    directions = np.asarray(rays.directions)
    k = 2.0 * math.pi / rays.wavelength_m
    extra = np.exp(
        1j * k * (directions[:, 0] * positions[:, 0] + directions[:, 1] * positions[:, 1])
    )
    return _with_amplitude(rays, np.asarray(rays.amplitude) * extra)


def _with_amplitude(rays, amplitude):
    """A copy of the bundle carrying a mutated amplitude.

    ``RayBundle`` is frozen and exposes no amplitude setter, which is correct --
    the ways an amplitude may legitimately come into existence are named methods
    (``with_amplitude_from_weight``) rather than an open door. A negative control
    is not one of those ways, so it goes through ``dataclasses.replace`` here and
    the mutation is visible at the call site rather than hidden behind a helper
    on the artifact.
    """
    from dataclasses import replace

    return replace(rays, amplitude=amplitude)


# ---------------------------------------------------------------------------
# The noise-limited relation, as a reusable instrument
# ---------------------------------------------------------------------------


def noise_limited_relation(
    ncc_ab: float, ncc_aa: float, ncc_bb: float
) -> dict[str, float]:
    """``NCC(A,B) ~= sqrt(NCC(A,A') * NCC(B,B'))``, and how close it came.

    The instrument for systems where NEITHER route has converged, which is the
    situation a straight comparison cannot say anything useful about: two noisy
    estimates of the same field agree only as well as their own self-consistency
    allows, and that bound is computable from each route's own seed-to-seed
    agreement. Predicted 0.0129 against a measured 0.0147 on demo3, a ratio of
    1.14.

    This is CHARACTERIZATION and not a gate. There is no oracle in it: it says
    what agreement is *achievable* given two noise levels, so a route pair that
    matches the prediction has demonstrated that its disagreement is noise rather
    than a defect -- which is a different and weaker statement than being right.
    """
    predicted = math.sqrt(max(ncc_aa, 0.0) * max(ncc_bb, 0.0))
    return {
        "predicted": predicted,
        "measured": ncc_ab,
        "ratio": ncc_ab / predicted if predicted else float("nan"),
    }


# ---------------------------------------------------------------------------
# The instances
# ---------------------------------------------------------------------------


def _run_full_aperture() -> InstanceRun:
    instance = _instance("B2-EQUIV-FULL-01")
    anchor = _full_aperture()
    padded_refusal, padded = probe_refusal(
        lambda: _padded_full_aperture_anchor()
    )

    # The two defects the anchor is blind to, measured HERE so the blindness is a
    # number rather than a claim.
    uncovered = _blindness_at_full_aperture("uncovered")
    launch = _blindness_at_full_aperture("launch")

    eps = float(np.finfo(np.float64).eps)
    derived_floor = eps * math.sqrt(anchor["propagating_modes"]) * 64.0

    measurements = {
        "patch_vs_global_relative_l2": Measurement(
            value=anchor["relative_l2"],
            uncertainty=anchor["energy_residual"],
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"one patch over the whole {ANCHOR_N}x{ANCHOR_N} aperture, every one of "
                f"{anchor['propagating_modes']} modes enumerated, against the "
                "independent float64 ASM at pad 0. The error bar is the energy residual, "
                "which is what catches a transfer function that has stopped being "
                f"unit-modulus. Derived round-off floor {derived_floor:.3e}."
            ),
        ),
        "coverage_corrected_power_ratio": Measurement(
            value=abs(1.0 - anchor["coverage"]),
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                f"|1 - coverage| with coverage {anchor['coverage']:.15f}. Exactly 1 at "
                "full aperture BY CONSTRUCTION, which is precisely why this instance "
                "cannot gate the correction and the sub-aperture ones must."
            ),
        ),
    }
    invariants = {
        "PATCH_COVERAGE_CORRECTED": measurements["coverage_corrected_power_ratio"],
        "OUTGOING_COUNT_IS_THE_BUDGET": Measurement(
            value=0.0,
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                "the enumerated case emits every propagating mode per patch, so the "
                f"count is {anchor['propagating_modes']} by construction and the "
                "identity is exact. The budget-independence property is asserted on a "
                "cascade by tests/test_planar_doe_step.py."
            ),
        ),
    }
    controls = {
        "omit-coverage-correction": NegativeControlResult(
            control_id="omit-coverage-correction",
            outcome=NegativeControlOutcome.NOT_RUN,
            target_metric="patch_vs_global_relative_l2",
            baseline=Measurement(
                value=anchor["relative_l2"],
                uncertainty=anchor["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="the anchor",
            ),
            mutated=Measurement(
                value=uncovered,
                uncertainty=anchor["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=(
                    f"the correction inverted at full aperture: {uncovered:.6e}, which "
                    "is the anchor's own value. The correction is exactly 1 here"
                ),
            ),
            note=(
                "NOT_RUN on this instance ON PURPOSE, and the number above is why: the "
                "coverage correction is exactly 1 at full aperture, so inverting it "
                "changes nothing measurable. That is how the real inversion "
                "(A_patch/A_draw for A_draw/A_patch) survived. The control is run on the "
                "SUB-APERTURE instances, where the correction is not 1."
            ),
        ),
        "launch-phase-per-patch": NegativeControlResult(
            control_id="launch-phase-per-patch",
            outcome=NegativeControlOutcome.NOT_RUN,
            target_metric="patch_vs_global_relative_l2",
            baseline=Measurement(
                value=anchor["relative_l2"],
                uncertainty=anchor["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="the anchor",
            ),
            mutated=Measurement(
                value=launch,
                uncertainty=anchor["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=f"the launch phase double-counted at full aperture: {launch:.6e}",
            ),
            note=(
                "NOT_RUN here for the same reason: a single patch centred on the origin "
                "has zero launch offset, so double-counting its phase is a no-op. Run on "
                "the sub-aperture instances."
            ),
        ),
        "grid-snapping-is-not-free": NegativeControlResult(
            control_id="grid-snapping-is-not-free",
            outcome=NegativeControlOutcome.NOT_RUN,
            target_metric="patch_vs_global_relative_l2",
            note=(
                "a single full-aperture patch is centred by construction, so there is "
                "nothing to unsnap. Run on the sub-aperture instances, where continuous "
                "centres inject a sub-sample linear phase and the sweep plateaus."
            ),
        ),
    }
    record = record_from_probe(
        instance,
        component="C_PATCH_WFT + C_PLANAR_DOE_STEP",
        node_id="full_aperture_limit",
        refusal=None,
        observed_parameters={
            "patch_count": 1,
            "pad_width": int(anchor["pad_px"]),
            "grid_snapping": "snapped",
        },
        diagnostics=[
            {
                "code": "THE_ORACLE_AND_ITS_PAD",
                "detail": (
                    "verification/asm_oracle.angular_spectrum_float64 at pad 0, over the "
                    f"unpadded {ANCHOR_N}x{ANCHOR_N} grid. Named because an oracle is not "
                    "well defined until its padding is: the same route against a pad-200 "
                    "oracle reads 8.8e-3 and against a pad-101 oracle 0.33, and neither "
                    "is an error in either implementation -- both are wraparound between "
                    "two periods."
                ),
                "location": "src/verification/asm_oracle.py",
            },
            {
                "code": "THE_CLEARANCE_EXEMPTION_IS_PRESERVED",
                "detail": (
                    f"pad_px {anchor['pad_px']} at pad_factor 1. Padding a full-aperture "
                    "single patch moves its modes off the unpadded oracle's grid and the "
                    "anchor reads "
                    + (
                        f"{padded['relative_l2']:.4f}"
                        if padded is not None
                        else f"a refusal: {padded_refusal.code if padded_refusal else 'unknown'}"
                    )
                    + " instead of 1e-12. The exemption is a property of the "
                    "full-aperture limit, not a relaxation, and 'fixing' it by padding "
                    "would change the mode grid."
                ),
                "location": "src/couplers/patch.py::resolve_pad_px",
            },
            {
                "code": "WHAT_THE_ANCHOR_CANNOT_SEE",
                "detail": (
                    "the coverage correction and the launch phase are both exactly 1 "
                    f"here: inverting the correction gives {uncovered:.6e} and "
                    f"double-counting the launch phase gives {launch:.6e}, against an "
                    f"anchor of {anchor['relative_l2']:.6e}. Both are real defects that "
                    "survived because of exactly this, and the sub-aperture instances "
                    "are what gate them."
                ),
                "location": "benchmarks/instances/b2_equiv.py::_blindness_at_full_aperture",
            },
        ],
    )
    return InstanceRun(
        family=B2_EQUIV,
        instance=instance,
        record=record,
        result=verify(
            B2_EQUIV,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
        ),
    )


def _padded_full_aperture_anchor() -> dict[str, Any]:
    """The anchor with the clearance exemption removed, to measure what it costs."""
    doe = _doe(ANCHOR_N)
    plan = plan_patches(
        grid_shape=(ANCHOR_N, ANCHOR_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        patch_px=ANCHOR_N,
        pad_factor=3,
        patch_count=None,
        substrate=Substrate.PLANAR,
    )
    rays, _ = patch_secondary_rays(
        doe,
        plan=plan,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=None,
    )
    reconstructed = _reconstruct(
        advance_bundle_to_plane(rays, target=SENSOR_PLANE), n=ANCHOR_N, plane=SENSOR_PLANE
    )
    comparison = compare_fields(reconstructed, _global_route(doe))
    return {"relative_l2": comparison.raw_relative_field_error, "pad_px": plan.pad_px}


def _blindness_at_full_aperture(variant: str) -> float:
    """The same mutation at full aperture, where it is inert. A number, not a claim."""
    doe = _doe(ANCHOR_N)
    plan = plan_patches(
        grid_shape=(ANCHOR_N, ANCHOR_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        patch_px=ANCHOR_N,
        pad_factor=1,
        patch_count=None,
        substrate=Substrate.PLANAR,
    )
    rays, _ = patch_secondary_rays(
        doe,
        plan=plan,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=None,
    )
    rays = _invert_coverage(rays, plan) if variant == "uncovered" else _double_launch_phase(rays)
    reconstructed = _reconstruct(
        advance_bundle_to_plane(rays, target=SENSOR_PLANE), n=ANCHOR_N, plane=SENSOR_PLANE
    )
    return compare_fields(reconstructed, _global_route(doe)).raw_relative_field_error


def _run_sub_aperture(instance_id: str) -> InstanceRun:
    instance = _instance(instance_id)
    patch_count = int(instance.parameters["patch_count"])
    correct = _sub_aperture(patch_count)

    enumerated_positions = (SUB_N + 2 * (5 // 2)) ** 2
    counts = tuple(
        sorted(
            int(i.parameters["patch_count"])
            for i in B2_EQUIV.canonical_instances
            if i.instance_id.startswith("B2-EQUIV-SUB-")
            and int(i.parameters["patch_count"]) != enumerated_positions
        )
    )
    ladder = [(float(n), _sub_aperture(n)["relative_l2"]) for n in counts]

    convergence = fit_convergence(
        "patch_count",
        ladder,
        note=(
            "the sub-aperture residual against the independent float64 ASM at pad 0, "
            "over four patch counts. No expected exponent is declared: the coherent "
            "patch sum's convergence rate in the number of DRAWN centres is a Monte "
            "Carlo rate over a finite population, and asserting one would be asserting "
            "a model the family has not established. That the residual FALLS is the "
            "convergence statement."
        ),
    )

    # The controls run on the ENUMERATED instance, where the estimator's own
    # sampling error is zero -- so the separation they show is the mutation's and
    # nothing else. On a drawn instance a control is competing with the Monte
    # Carlo residual.
    is_finest = patch_count == enumerated_positions
    controls: dict[str, Any] = {}
    for control_id, variant, why in (
        (
            "omit-coverage-correction",
            "uncovered",
            "exactly 1 at full aperture, which is how the real inversion survived",
        ),
        (
            "launch-phase-per-patch",
            "launch",
            "exactly 1 for a single centred patch, likewise",
        ),
        (
            "grid-snapping-is-not-free",
            "continuous",
            "centres off the sample grid inject a sub-sample linear phase and the sweep "
            "plateaus instead of converging",
        ),
    ):
        if not is_finest:
            controls[control_id] = NegativeControlResult(
                control_id=control_id,
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric="patch_vs_global_relative_l2",
                note=(
                    "run once per family, on the finest patch count, because a control "
                    "on a coarse decomposition is dominated by the decomposition's own "
                    "residual rather than by the mutation."
                ),
            )
            continue
        broken = _sub_aperture(patch_count, variant=variant)
        controls[control_id] = control_result(
            control_id,
            "patch_vs_global_relative_l2",
            baseline=Measurement(
                value=correct["relative_l2"],
                uncertainty=correct["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=f"the correct arm at {patch_count} patches, oracle pad 0",
            ),
            mutated=Measurement(
                value=broken["relative_l2"],
                uncertainty=broken["energy_residual"],
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=f"{variant}: {why}",
            ),
            threshold=correct["relative_l2"],
            note=(
                "gated against the CORRECT arm at the same patch count rather than "
                "against an absolute number, because the sub-aperture residual is itself "
                "a function of the granularity."
            ),
        )

    # The apodization control: not one of the family's three, and reported as a
    # diagnostic, because what it breaks is the CONVERGENCE rather than a single
    # score. A taper below 1 removes field no other patch replaces, so the
    # coherent sum stops converging to the full-DOE response.
    apodized_ladder = [
        (float(n), _sub_aperture(n, variant="apodized")["relative_l2"]) for n in counts
    ]

    measurements = {
        "patch_vs_global_relative_l2": Measurement(
            value=correct["relative_l2"],
            uncertainty=correct["energy_residual"],
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"{patch_count} patches of 5 px over a {SUB_N}x{SUB_N} aperture, every "
                "propagating mode per patch, against the independent float64 ASM AT PAD "
                "0. The pad is named because the score is meaningless without it."
            ),
        ),
        "coverage_corrected_power_ratio": Measurement(
            value=abs(1.0 - correct["power_ratio"]),
            uncertainty=correct["energy_residual"],
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"|1 - power ratio| with the correction applied; coverage "
                f"{correct['coverage']:.9f}, which is NOT 1 here -- that is the whole "
                "reason the correction is gateable on this instance and not on the anchor."
            ),
        ),
    }
    invariants = {
        "PATCH_COVERAGE_CORRECTED": measurements["coverage_corrected_power_ratio"],
        "OUTGOING_COUNT_IS_THE_BUDGET": Measurement(
            value=abs(
                correct["ray_count"] - patch_count * correct["propagating_modes"]
            )
            / max(correct["ray_count"], 1),
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                f"{correct['ray_count']} rays from {patch_count} patches x "
                f"{correct['propagating_modes']} enumerated modes -- an exact integer "
                "identity, so the relative discrepancy is zero or the identity is broken."
            ),
        ),
    }
    record = record_from_probe(
        instance,
        component="C_PATCH_WFT + C_PLANAR_DOE_STEP",
        node_id=f"sub_aperture_{patch_count}",
        refusal=None,
        observed_parameters={
            "patch_count": patch_count,
            "pad_width": int(correct["pad_px"]),
            "grid_snapping": "snapped",
        },
        diagnostics=[
            {
                "code": "CONVERGENCE_LADDER_WITH_ITS_ORACLE_PAD",
                "detail": (
                    "oracle: angular_spectrum_float64 at pad 0. "
                    + "; ".join(f"{int(n)} patches -> {v:.6e}" for n, v in ladder)
                ),
                "location": "benchmarks/instances/b2_equiv.py::_sub_aperture",
            },
            {
                "code": "APODIZATION_BREAKS_THE_CONVERGENCE",
                "detail": (
                    "a raised-cosine taper, same ladder: "
                    + "; ".join(f"{int(n)} -> {v:.6e}" for n, v in apodized_ladder)
                    + ". Any window below 1 removes field that no other patch replaces, "
                    "so the partition-of-unity argument behind the convergence relation "
                    "is exactly what a taper breaks -- which is why the step carries no "
                    "apodization."
                ),
                "location": "src/registry/couplers.yaml#C_PATCH_WFT.validity.warnings",
            },
            {
                "code": "WHERE_THE_EQUIVALENCE_HOLDS_AND_WHERE_IT_STOPS",
                "detail": (
                    "measured, and it is the most useful finding in this family. The "
                    "enumerated sub-aperture sum reproduces the field AT THE DOE PLANE "
                    "to 1.7e-15 and disagrees with the independent ASM AT z = 1.26 mm at "
                    "0.84 -- and neither number is a defect in either implementation. A "
                    "sub-aperture patch's modes live on its own pad-21 grid, which is "
                    "not commensurate with the 15-px reconstruction grid, so the ray sum "
                    "is the NON-periodic propagated field while the ASM is the periodic "
                    "one, and they differ by the wrapped contributions. At full aperture "
                    "with pad_factor 1 the two mode sets coincide exactly, which is why "
                    "the anchor reads 1.4e-12 and this comparison cannot. It is the "
                    "CHE-96 oracle-padding lesson in a third coordinate: a score is not "
                    "defined until the oracle's grid is."
                ),
                "location": "knowledge/couplers/patch_wft/failure_guide.md",
            },
            {
                "code": "COVERAGE_IS_NOT_ONE_HERE",
                "detail": (
                    f"coverage {correct['coverage']:.9f} at {patch_count} patches. The "
                    "full-aperture anchor has coverage exactly 1, so it cannot see the "
                    "correction; this instance can."
                ),
                "location": "src/couplers/patch.py::plan_patches",
            },
        ],
    )
    return InstanceRun(
        family=B2_EQUIV,
        instance=instance,
        record=record,
        result=verify(
            B2_EQUIV,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


# ---------------------------------------------------------------------------
# The refusals the pack documents
# ---------------------------------------------------------------------------


def declared_refusals() -> dict[str, Any]:
    """The two configuration refusals M2.3 asks to be gated, executed.

    Both are cases where the code would run and the answer would be wrong, and
    both are refused rather than rounded -- which is the property, not the fact
    that an exception appears.
    """
    even_patch, _ = probe_refusal(
        lambda: plan_patches(
            grid_shape=(SUB_N, SUB_N),
            sample_pitch_m=(PITCH_M, PITCH_M),
            # The paper's own sizes are 40, 50 and 100 -- all even -- so an even
            # request is the LIKELY one, and rounding it would silently change
            # the operator the caller asked for.
            patch_px=4,
            pad_factor=1,
            patch_count=4,
            substrate=Substrate.PLANAR,
            rng=np.random.default_rng(1),
        )
    )
    conformal, _ = probe_refusal(
        lambda: plan_patches(
            grid_shape=(SUB_N, SUB_N),
            sample_pitch_m=(PITCH_M, PITCH_M),
            patch_px=5,
            pad_factor=1,
            patch_count=4,
            substrate=Substrate.CONFORMAL,
            rng=np.random.default_rng(1),
        )
    )
    # The pad is DERIVED, not taken: pad_factor is a preference and the step
    # raises it until clearance, centring and oddness all hold, then reports what
    # it used. Measured rather than asserted.
    derived = {
        f"pad_factor={factor}": int(
            resolve_pad_px(
                grid_n=SUB_N, patch_px=5, pad_factor=factor, max_center_px=SUB_N // 2 + 2
            )
        )
        for factor in (1, 2, 3)
    }
    return {
        "even_patch_px": {
            "refused": even_patch is not None,
            "code": None if even_patch is None else even_patch.code,
            "detail": None if even_patch is None else even_patch.detail,
        },
        "conformal_substrate": {
            "refused": conformal is not None,
            "code": None if conformal is None else conformal.code,
            "detail": None if conformal is None else conformal.detail,
        },
        "derived_pad": derived,
    }


# ---------------------------------------------------------------------------
# Route agreement on the paper's own systems (CHE-111 AC 3 and AC 4)
# ---------------------------------------------------------------------------

#: The two probe records the route-agreement characterization reads. Both are
#: 60M-ray CUDA runs that cannot execute in a gate, so what is executed here is
#: the *instrument* on their recorded outputs -- never a re-derivation of them.
_DEMO2_RECORD = ROOT / "benchmarks/probes/records/ray_wave/demo2_paper_kspace_jax.json"
_DEMO3_RECORD = ROOT / "benchmarks/probes/records/ray_wave/demo3_route_agreement.json"


def route_agreement() -> dict[str, Any]:
    """The two-route agreement on demo2 and demo3, and why neither may gate.

    CHE-111 asks for "route agreement gated on demo2 at a declared NCC". It
    cannot be gated, and the reason is the substrate's own rule rather than a
    convenience: RW-F against RW-P is a ``CROSS_ROUTE`` oracle, which forces
    category B4, and a B4 family may not carry a gating tolerance. Gating it
    would be exactly the promotion the architecture exists to prevent -- two of
    our own routes agreeing is not evidence either is right, and if they share a
    convention error they agree perfectly.

    What CAN carry the claim is on the same record: the sub-aperture route scored
    against the *independent* float64 ASM reads NCC 0.99941823, which is
    numerically indistinguishable from the 0.99941808 cross-route number. That
    coincidence is the useful finding. The two routes agree at the level at which
    each one independently matches an oracle, so the agreement is not hiding a
    shared error -- and that statement is available only because the independent
    comparison exists alongside the cross-route one.

    For demo3 the paper states no conventional reference exists, so there is no
    independent oracle at all and the noise-limited relation is the whole
    instrument. It is evaluated here by the shipping function on the record's own
    self- and cross-NCCs, not copied from the record's stored ratio.
    """
    demo2 = json.loads(_DEMO2_RECORD.read_text())
    demo3 = json.loads(_DEMO3_RECORD.read_text())

    self_ncc = demo3["noise_limited_agreement"]["mean_self_ncc"]
    recomputed = noise_limited_relation(
        ncc_ab=float(demo3["noise_limited_agreement"]["mean_cross_route_ncc"]),
        ncc_aa=float(self_ncc["demo3_characterization_rw_f"]),
        ncc_bb=float(self_ncc["demo3_characterization_rw_p"]),
    )
    return {
        "demo2": {
            "cross_route_ncc": float(demo2["route_agreement"]["ncc_intensity"]),
            "cross_route_relative_l2": float(demo2["route_agreement"]["relative_l2_field"]),
            "sub_aperture_vs_independent_oracle_ncc": float(
                demo2["routes"]["rw_p"]["vs_oracle"]["ncc_intensity"]
            ),
            "full_aperture_vs_independent_oracle_relative_l2": float(
                demo2["routes"]["rw_f"]["vs_oracle"]["relative_l2_field"]
            ),
            "oracle": "verification/asm_oracle.angular_spectrum_float64, pad matched to the route",
            "why_it_cannot_gate": (
                "RW-F vs RW-P is a CROSS_ROUTE oracle, which forces B4, and a B4 "
                "family may not carry a gating tolerance. The independent-oracle "
                "number beside it is what carries the claim."
            ),
        },
        "demo3": {
            "recomputed": recomputed,
            "as_recorded": {
                "predicted": float(
                    demo3["noise_limited_agreement"]["predicted_if_same_field"]
                ),
                "measured": float(
                    demo3["noise_limited_agreement"]["mean_cross_route_ncc"]
                ),
                "ratio": float(
                    demo3["noise_limited_agreement"]["ratio_measured_over_predicted"]
                ),
            },
            "no_independent_oracle": (
                "the paper states no conventional reference exists for this system, so "
                "the noise-limited relation is the entire instrument and there is "
                "nothing here that could be promoted to a gate."
            ),
        },
        "provenance": {
            "demo2_record": str(_DEMO2_RECORD.relative_to(ROOT)),
            "demo3_record": str(_DEMO3_RECORD.relative_to(ROOT)),
            "what_executed_here": (
                "noise_limited_relation, freshly, on the recorded self- and cross-NCCs. "
                "The 60M-ray CUDA runs that produced those NCCs are NOT re-executed and "
                "are not claimed as this benchmark's measurement."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(i.instance_id for i in B2_EQUIV.canonical_instances)


def run_instance(instance_id: str) -> InstanceRun:
    if instance_id == "B2-EQUIV-FULL-01":
        return _run_full_aperture()
    if instance_id.startswith("B2-EQUIV-SUB-"):
        return _run_sub_aperture(instance_id)
    raise KeyError(f"no runner for {instance_id!r}")


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


def _describe(metric: Any) -> str:
    if not metric.tolerance_may_gate:
        return f"{metric.metric}={metric.measured.value:.6g} (reported, not gating)"
    verdict = "" if metric.met is None else (" MET" if metric.met else " UNMET")
    return f"{metric.metric}={metric.measured.value:.6g}{verdict}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--instance", default=None)
    args = parser.parse_args()

    runs = {args.instance: run_instance(args.instance)} if args.instance else run_all()
    for instance_id, run in runs.items():
        metrics = ", ".join(_describe(m) for m in run.result.physics_accuracy)
        print(f"{instance_id:<24} {run.result.status.value:<18} {metrics}")
        controls = ", ".join(
            f"{c.control_id}:{c.outcome.value}" for c in run.result.negative_control_results
        )
        if controls:
            print(f"{'':<24} controls: {controls}")
        if args.write:
            path = write_instance_record(run, driver="instances/b2_equiv")
            print(f"{'':<24} -> {path.relative_to(ROOT)}")
    if not args.instance:
        print("refusals:", declared_refusals())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
