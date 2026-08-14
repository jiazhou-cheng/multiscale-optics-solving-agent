"""CHE-33 (M3.4): declare a real ray trace into a coherent :class:`RayBundle`.

The Optiland adapter emits an ``ArtifactRecord`` that is deliberately unusable
as coherent physics: it carries ``opd_native`` (an accumulated optical path
whose reference is a launch plane nobody downstream knows about) and
``intensity`` (a real weight explicitly marked as not an amplitude). The
contract layer refuses both, and this module is the one place that turns them
into declarations.

Nothing here is new physics. Three prior results are being *applied*:

``CHE-30`` -- the ``RealRays.opd`` convention
    Absolute accumulated **optical** path length in millimetres, index-weighted,
    referenced to the ray launch state. For an object at infinity that launch
    plane is placed at ``positions[1] - (EPD - min(positions[1:-1]))``, so the
    zero *moves when the aperture changes*. An absolute Optiland OPL is
    therefore only meaningful alongside a declared entrance pupil diameter, and
    both are recorded here.

``CHE-32`` -- the exit-pupil handoff
    The exported ``x_m``/``y_m`` are each ray's image-space **asymptote** at the
    exit-pupil plane, while ``opd_native`` is still accumulated to the final
    traced image surface. Position and phase would otherwise describe different
    planes. Image space is homogeneous, so the correction is exact: subtract
    ``n_image * (z_image - z_pupil) / N`` per ray. The adapter records
    ``image_space_refractive_index`` so ``n_image`` is read, not assumed --
    ``n = 1`` is true for both M3 systems and is checked rather than believed.

``CHE-40`` -- the required ray-side conditioning
    ``benchmarks/slice_protocol.yaml`` requires
    ``phi_i = (2*pi/lambda) (OPL_i - OPL_ref)``, never absolute accumulated OPL.
    That is not a stylistic preference: on ``M3-SINGLET-REF`` the exported
    absolute OPL is 1.0e4 waves at the image surface and the piston removed here
    is 1219 waves, against 11.7 waves of actual wavefront. Forming phase from the
    absolute value spends the float budget on something no single-path PSF can
    see. The removed piston is retained in float64 as ``removed_reference_opl_m``
    and is never folded back -- the same policy the wave side already applies to
    ``exp(i k z)``.

    A second, less obvious benefit: because the reference is a ray from the same
    trace, the aperture-dependent launch-plane zero is *common to every ray* and
    cancels exactly in the subtraction. The declared OPL is therefore invariant
    under the one thing CHE-30 warned makes the absolute value meaningless.

Verification, not assertion
---------------------------
``knowledge/couplers/ray_to_wave/probes/coherent_handoff.py`` checks the declared
OPL against an oracle that uses nothing from this repository: for a
diffraction-limited system every ray reaches the focus with equal total optical
path, so the pupil OPL must satisfy ``OPL(rho) - OPL(0) = R - sqrt(rho^2 + R^2)``.
On ``M3-SINGLET-REF`` the residual is **0.016999 waves peak-to-valley** against
M3.2's independently frozen 0.016996 and Optiland's own ``Wavefront`` value of
0.016999 -- three routes, one number. With the sign flipped it is **23.5 waves**,
a factor of 1380. The declaration is falsifiable, and was falsified in the wrong
configuration.

This module imports no solver engine. It reads a repository artifact record and
returns a repository contract type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.couplers.contracts import (
    ContractCode,
    ContractError,
    RayBundle,
    ReferencePlane,
)

__all__ = [
    "AMPLITUDE_MAPPING",
    "NATIVE_OPD_UNIT_M",
    "CoherentHandoff",
    "DeclaredHandoffPlane",
    "HandoffPerturbation",
    "declare_coherent_bundle",
    "reconstruct_hashed_arrays",
]

#: CHE-30 part 4: ``RealRays.opd`` is in the lens geometry unit, millimetres for
#: every prescription in this repository, and scales exactly with the
#: prescription.
NATIVE_OPD_UNIT_M = 1.0e-3

#: The intensity-to-amplitude declaration. The prefix is load-bearing:
#: :meth:`RayBundle.with_amplitude_from_weight` only performs the conversion
#: itself for a mapping that starts with ``amplitude = sqrt(weight)``, so the
#: string and the arithmetic cannot disagree.
AMPLITUDE_MAPPING = (
    "amplitude = sqrt(weight); weight is Optiland RealRays.i, a real per-ray "
    "intensity (power-like, non-negative, dimensionless). Optiland supplies NO "
    "phase in i, so every radian of the reconstructed field comes from the "
    "optical path length. No per-ray area factor and no 1/N is applied: the "
    "traced set is a physical ray ensemble, not a Monte Carlo sample of one."
)

#: Absolute tolerance on the handoff-plane axial coordinate, in metres. The
#: frozen protocol stores the plane as a float64 literal produced by the same
#: paraxial solver the adapter calls, so agreement is exact to round-off; 1 pm
#: is round-off headroom, not a physics allowance.
DEFAULT_PLANE_TOLERANCE_M = 1.0e-12


@dataclass(frozen=True)
class DeclaredHandoffPlane:
    """The plane the *consumer* expects, stated independently of the producer.

    Both halves are checked. ``handoff_plane`` catches an adapter that exported
    at the image surface when the exit pupil was asked for -- an error of the
    whole pupil-to-focus distance. ``z_m`` catches the subtler case where the
    right kind of plane sits at the wrong place, which is a defocus and is
    invisible in every rotationally symmetric sanity check.
    """

    handoff_plane: Literal["exit_pupil", "image_surface"]
    z_m: float
    tolerance_m: float = DEFAULT_PLANE_TOLERANCE_M
    name: str | None = None


@dataclass(frozen=True)
class HandoffPerturbation:
    """Deliberate defects, for negative tests only.

    Mirrors :class:`multiscale_optics_agent.couplers.ray_to_wave.Perturbation`
    and exists for the same reason: a negative test must exercise *this* code
    with one term altered, not a hand-written parallel copy that can drift away
    from what ships.
    """

    #: Flip the OPL sign. Conjugates the wavefront -- a converging pupil field
    #: becomes diverging -- and is exactly the error CHE-30 warned is
    #: indistinguishable from the correct one once it is downstream.
    opl_sign: Literal[1, -1] = 1
    #: Skip moving the OPL from the traced image surface to the declared plane.
    #: Leaves phase and position describing planes millimetres apart.
    transfer_opl_to_plane: bool = True

    @property
    def is_identity(self) -> bool:
        return self.opl_sign == 1 and self.transfer_opl_to_plane

    def describe(self) -> str:
        if self.is_identity:
            return "none"
        parts = []
        if self.opl_sign != 1:
            parts.append("opl_sign_flipped")
        if not self.transfer_opl_to_plane:
            parts.append("opl_plane_transfer_omitted")
        return "+".join(parts)


@dataclass(frozen=True)
class CoherentHandoff:
    """A coherent bundle plus everything that had to be declared to get one."""

    bundle: RayBundle
    declarations: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _conventions(record: ArtifactRecord) -> dict[str, Any]:
    conventions = record.metadata.get("conventions")
    if not isinstance(conventions, dict):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "ray record carries no conventions block",
            declaration="conventions",
            artifact_id=record.id,
        )
    return conventions


def _check_plane(
    record: ArtifactRecord, conventions: dict[str, Any], declared: DeclaredHandoffPlane
) -> None:
    produced_plane = conventions.get("handoff_plane")
    if produced_plane != declared.handoff_plane:
        raise ContractError(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            (
                f"the record was produced at handoff plane {produced_plane!r} but the "
                f"consumer declared {declared.handoff_plane!r}"
            ),
            declaration="conventions.handoff_plane",
            artifact_id=record.id,
            remedy=(
                "Re-run the ray model with config['handoff_plane'] set to the declared "
                "plane, or declare the plane the trace was actually exported at. These "
                "two planes are a pupil-to-focus distance apart, so accepting the "
                "mismatch would defocus the reconstruction rather than piston it."
            ),
        )
    produced_z = conventions.get("reference_plane_z_m")
    if produced_z is None or not math.isfinite(float(produced_z)):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "ray record declares no finite reference_plane_z_m",
            declaration="conventions.reference_plane_z_m",
            artifact_id=record.id,
        )
    offset = abs(float(produced_z) - declared.z_m)
    if offset > declared.tolerance_m:
        raise ContractError(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            (
                f"handoff plane is at z = {float(produced_z)!r} m but the consumer "
                f"declared z = {declared.z_m!r} m; offset {offset:.6e} m exceeds "
                f"tolerance {declared.tolerance_m:.1e} m"
            ),
            declaration="conventions.reference_plane_z_m",
            artifact_id=record.id,
            remedy=(
                "An axial offset between the plane the rays were exported at and the "
                "plane the field is declared on is a defocus, not a piston. Fix the "
                "declaration or the trace; do not widen the tolerance."
            ),
        )


def _require_positive_index(conventions: dict[str, Any], record: ArtifactRecord) -> float:
    value = conventions.get("image_space_refractive_index")
    if value is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            (
                "the record does not declare the image-space refractive index, so the "
                "optical path between the traced image surface and the declared plane "
                "cannot be evaluated"
            ),
            declaration="conventions.image_space_refractive_index",
            artifact_id=record.id,
            remedy=(
                "The ray adapter must export the index of the medium preceding the "
                "image surface. It is read from the prescription, never assumed to be 1."
            ),
        )
    index = float(value)
    if not math.isfinite(index) or index <= 0.0:
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"image-space refractive index must be positive and finite, got {index!r}",
            declaration="conventions.image_space_refractive_index",
            artifact_id=record.id,
        )
    return index


def declare_coherent_bundle(
    record: ArtifactRecord,
    *,
    declared_plane: DeclaredHandoffPlane,
    arrays: dict[str, np.ndarray] | None = None,
    perturbation: HandoffPerturbation = HandoffPerturbation(),
) -> CoherentHandoff:
    """Turn an Optiland ray record into a bundle ``require_coherent()`` accepts.

    The bundle is accepted **because two conventions are declared**, not because
    any check was relaxed: a record whose ``opd_native`` is left undeclared still
    fails with :attr:`ContractCode.OPL_REFERENCE_UNVERIFIED`, and this function
    is the only thing in the repository that supplies the declaration.

    Raises:
        ContractError: on a plane mismatch, a missing declaration, an absent
            ``opd_native``, or a weight that cannot be a power.
    """
    conventions = _conventions(record)
    _check_plane(record, conventions, declared_plane)

    bundle = RayBundle.from_artifact_record(record, arrays=arrays)

    opd_native = bundle.provenance.get("opd_native")
    if opd_native is None:
        raise ContractError(
            ContractCode.OPL_REFERENCE_UNVERIFIED,
            "the ray record carries no opd_native array, so no optical path length "
            "can be declared from it",
            declaration="opd_native",
            artifact_id=record.id,
            remedy=(
                "Export opd_native from the ray model. This function declares an "
                "existing optical path; it does not synthesize one."
            ),
        )

    # --- 1. Native unit -> SI (CHE-30 part 4) --------------------------------
    optical_path_at_image_m = np.asarray(opd_native, dtype=np.float64) * NATIVE_OPD_UNIT_M

    # --- 2. Move the OPL to the plane the positions are on (CHE-32) ----------
    image_space_index = _require_positive_index(conventions, record)
    z_plane_m = float(conventions["reference_plane_z_m"])
    if declared_plane.handoff_plane == "exit_pupil":
        exit_pupil = conventions.get("exit_pupil") or {}
        location_from_image_m = exit_pupil.get("location_from_image_m")
        if location_from_image_m is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "exit-pupil handoff declared, but the record carries no "
                "exit_pupil.location_from_image_m to locate the traced image surface",
                declaration="conventions.exit_pupil.location_from_image_m",
                artifact_id=record.id,
            )
        z_image_m = z_plane_m - float(location_from_image_m)
    else:
        z_image_m = z_plane_m

    direction_z = bundle.directions[:, 2]
    if np.any(direction_z == 0.0):
        raise ContractError(
            ContractCode.NON_UNIT_DIRECTION,
            "a ray has N = 0 and never reaches the traced image surface, so its "
            "optical path cannot be referred to the declared plane",
            declaration="directions",
            artifact_id=record.id,
        )
    step_m = (z_image_m - z_plane_m) / direction_z
    if perturbation.transfer_opl_to_plane:
        optical_path_at_plane_m = optical_path_at_image_m - image_space_index * step_m
    else:
        optical_path_at_plane_m = optical_path_at_image_m

    # --- 3. Remove the reference piston (CHE-40 required conditioning) -------
    radius_m = np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1])
    chief_index = int(np.argmin(radius_m))
    reference_opl_m = float(optical_path_at_plane_m[chief_index])
    relative_opl_m = float(perturbation.opl_sign) * (optical_path_at_plane_m - reference_opl_m)

    entrance_pupil_diameter_m = conventions.get("entrance_pupil_diameter_m")
    opl_reference_declaration = (
        f"traced chief ray (smallest pupil radius, row {chief_index}, "
        f"rho = {float(radius_m[chief_index]):.6e} m) evaluated at the "
        f"{declared_plane.handoff_plane} plane z = {z_plane_m!r} m; sign convention "
        "'ray minus chief', so a larger value means a longer optical path (CHE-30 "
        "part 2, and the same sign L1-RAY-01's evaluator declares). Source is "
        "RealRays.opd: absolute accumulated optical path in mm from the ray launch "
        "state (CHE-30 part 3), which for this infinite-object system sits at an "
        f"aperture-dependent plane at EPD = {entrance_pupil_diameter_m!r} m. That "
        "dependence is common to every ray in the trace and cancels exactly in this "
        "subtraction."
    )

    # --- 4. Declare, through the contract's own named entry points -----------
    bundle = bundle.with_declared_optical_path_length(
        relative_opl_m, reference=opl_reference_declaration
    )
    bundle = bundle.with_amplitude_from_weight(mapping=AMPLITUDE_MAPPING)

    declarations = {
        "issue": "CHE-33 (M3.4)",
        "optical_path_length_reference": opl_reference_declaration,
        "optical_path_length_unit_conversion": (
            f"opd_native * {NATIVE_OPD_UNIT_M!r} m per native unit (CHE-30 part 4: "
            "the lens geometry unit is millimetres)"
        ),
        "optical_path_length_plane_transfer": (
            "OPL is accumulated to the final traced image surface but the exported "
            "positions are the image-space asymptote at the declared plane, so "
            "n_image * (z_image - z_plane) / N is subtracted per ray. Image space is "
            "homogeneous, so this is exact geometry rather than an approximation; the "
            "index is read from the prescription."
            if perturbation.transfer_opl_to_plane
            else "OMITTED -- negative test only. Phase and position describe different planes."
        ),
        "amplitude_mapping": AMPLITUDE_MAPPING,
        "phase_source": (
            "the optical path length only. RealRays.i carries no phase, so a bundle "
            "with a correct amplitude and a wrong OPL is not a degraded field, it is "
            "a different one."
        ),
        "reconstruction_normalization": (
            f"{bundle.reconstruction_normalization!r}: a traced ray set is the physical "
            "ensemble, not a Monte Carlo sample of a spectrum, so SI eq S3/S5's 1/N "
            "must not be applied. CONSEQUENCE, recorded rather than hidden: with no "
            "per-ray area weight the reconstructed amplitude scales with ray density, "
            "so the field is convergent in shape but not in scale under ray "
            "refinement. Handed to CHE-38, which owns the ray-count convergence study."
        ),
        "global_phase_policy": (
            "retained_as_metadata_not_reapplied -- the removed reference OPL is kept "
            "in float64 in diagnostics and is never folded back into the declared "
            "path, mirroring the wave side's exp(i k z) policy (CHE-40). No consumer "
            "may read absolute optical phase off this bundle."
        ),
        "perturbation": perturbation.describe(),
    }

    diagnostics = {
        "ray_count": bundle.count,
        "handoff_plane": declared_plane.handoff_plane,
        "reference_plane_z_m": z_plane_m,
        "traced_image_surface_z_m": z_image_m,
        "image_space_refractive_index": image_space_index,
        "entrance_pupil_diameter_m": entrance_pupil_diameter_m,
        "chief_ray_index": chief_index,
        "chief_ray_radius_m": float(radius_m[chief_index]),
        "pupil_semi_extent_m": float(np.max(radius_m)),
        "plane_transfer_step_min_m": float(np.min(step_m)),
        "plane_transfer_step_max_m": float(np.max(step_m)),
        # The two numbers that justify the required conditioning: the piston that
        # was removed, and the signal that was left.
        "removed_reference_opl_m": reference_opl_m,
        "removed_reference_opl_waves": reference_opl_m / bundle.wavelength_m,
        "relative_opl_span_waves": float(np.ptp(relative_opl_m)) / bundle.wavelength_m,
        "relative_opl_min_m": float(np.min(relative_opl_m)),
        "relative_opl_max_m": float(np.max(relative_opl_m)),
        "weight_min": float(np.min(bundle.weight)) if bundle.weight is not None else None,
        "weight_max": float(np.max(bundle.weight)) if bundle.weight is not None else None,
        "weight_sum": float(np.sum(bundle.weight)) if bundle.weight is not None else None,
        "perturbation": perturbation.describe(),
    }

    bundle = bundle.with_provenance(
        handoff=declarations,
        handoff_diagnostics=diagnostics,
        removed_reference_opl_m=reference_opl_m,
    )
    return CoherentHandoff(bundle=bundle, declarations=declarations, diagnostics=diagnostics)


def reconstruct_hashed_arrays(bundle: RayBundle) -> dict[str, np.ndarray]:
    """Rebuild the exact array set the ray adapter hashed, from the bundle.

    Used to prove that the declaration step changed nothing it was not supposed
    to change: hashing this with the adapter's own ``_scientific_array_hash`` must
    reproduce ``metadata['scientific_array_sha256']`` byte for byte. A check that
    recomputed the hash from the file would only prove the file was not edited.

    The single traced wavelength is broadcast back to a per-ray array. If the
    trace had been polychromatic that broadcast would be wrong -- and the hash
    comparison is exactly what would catch it, which is why it is not guarded
    separately.
    """
    opd_native = bundle.provenance.get("opd_native")
    if opd_native is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "bundle carries no opd_native provenance, so the adapter's hashed array "
            "set cannot be reconstructed",
            declaration="provenance.opd_native",
        )
    count = bundle.count
    return {
        "x_m": bundle.positions_m[:, 0],
        "y_m": bundle.positions_m[:, 1],
        "z_m": bundle.positions_m[:, 2],
        "L": bundle.directions[:, 0],
        "M": bundle.directions[:, 1],
        "N": bundle.directions[:, 2],
        "intensity": bundle.weight,
        "wavelength_m": np.full(count, bundle.wavelength_m, dtype=np.float64),
        "opd_native": np.asarray(opd_native, dtype=np.float64),
        "survived": np.ones(count, dtype=np.bool_),
    }


def plane_from_reference(plane: ReferencePlane, handoff_plane: str) -> DeclaredHandoffPlane:
    """Convenience: state a declared plane from an existing :class:`ReferencePlane`."""
    return DeclaredHandoffPlane(
        handoff_plane=handoff_plane,  # type: ignore[arg-type]
        z_m=plane.z_m,
        name=plane.name,
    )
