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

``CHE-41`` -- which SURFACE the path is measured from
    The cancellation above is exact for the launch plane's *location*. It says
    nothing about the plane's *orientation*, and that is where v1 was wrong.
    ``RealRays.opd`` is seeded on a plane perpendicular to z, which is a wavefront
    only for a bundle travelling along z. For an off-axis collimated bundle the
    plane and the wavefront differ by ``n_object * (d0 . r_launch)`` -- linear in
    the launch coordinate, so a piston on axis and a tilt off it. Omitting it
    leaves a pupil OPL that is a clean converging sphere aimed at the *axis* at
    every field angle: on ``M3-REVERSE-TELEPHOTO`` at ``Hy = 0.2`` it retained
    0.13% of the required tilt and put the focus 209 um from the traced chief-ray
    intersection, with a 0.072-wave-P-V residual against its own fitted sphere.
    That is why it survived three tickets of on-axis verification.

    The term is measured by the ray adapter from the regenerated launch state and
    added here. When it is constant across the bundle it is *not* added, because a
    piston cannot survive step 3 -- which is what keeps every on-axis number
    bit-identical to v1's.

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
    "AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT",
    "NATIVE_OPD_UNIT_M",
    "OPL_REFERENCE_VERSION",
    "SUPERSEDED_OPL_REFERENCE",
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

#: CHE-47 (M3.9R extension): the declaration used when the ray record carries a
#: usable per-ray quadrature weight (see :mod:`multiscale_optics_agent.couplers.
#: quadrature`). ``amplitude = sqrt(weight) * quadrature_weight_m2`` is a LINEAR
#: correction on the amplitude, not on the intensity: the wavelet sum
#: approximates a surface integral over the aperture, and a quadrature weight is
#: the area element that integral discretizes with, which multiplies the field
#: value directly. Folding it in here -- rather than in C_RAY_TO_WAVE -- is what
#: keeps the kernel CHE-24/CHE-38 validated unchanged: the coupler still just
#: sums whatever amplitude the bundle declares.
AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT = (
    "amplitude = sqrt(weight) * quadrature_weight_m2; weight is Optiland "
    "RealRays.i as above. quadrature_weight_m2 is the absolute per-ray "
    "pupil/phase-space area element a hexapolar ring set represents "
    "(multiscale_optics_agent.couplers.quadrature.hexapolar_area_weight_m2), "
    "radial-trapezoid corrected at the center and outer-ring boundaries "
    "(CHE-38 sections 14-15, CHE-47). This replaces AMPLITUDE_MAPPING's 'no "
    "per-ray area factor' with the producer-supplied one the coupler card's "
    "per_ray_area_weight_at_the_aperture_boundary validity condition calls for."
)

#: The version of the OPL reference declaration this module produces. A handoff
#: convention is part of the coupler contract, so a change to it is versioned
#: rather than edited in place, and the superseded declaration is kept below
#: rather than deleted: every number M3.4-M3.8 reported was measured under v1,
#: and a reader of those records needs to be able to find out what they meant.
OPL_REFERENCE_VERSION = "C_RAY_TO_WAVE opl_reference v2 (CHE-41)"

#: What v1 said, and the one thing it got wrong.
SUPERSEDED_OPL_REFERENCE = {
    "version": "C_RAY_TO_WAVE opl_reference v1 (CHE-33)",
    "reference_surface": (
        "Optiland's ray launch state -- a plane perpendicular to z for an object at "
        "infinity -- with the traced chief ray's value subtracted as a piston."
    ),
    "what_it_got_right": (
        "everything measurable on axis. The declared OPL is numerically identical to "
        "v2's for any field whose object-space term is constant across the bundle, "
        "which is every configuration M3.4-M3.8 verified: 0.016999 waves P-V against "
        "the analytic sphere on M3-SINGLET-REF, agreeing with Optiland's own Wavefront "
        "and with M3.2's independently frozen 0.016996."
    ),
    "what_it_got_wrong": (
        "it named the reference a plane and used it as a wavefront. Off axis those "
        "differ by n_object * (d0 . r_launch), a term linear in the launch coordinate "
        "that carries the whole convergence tilt: on M3-REVERSE-TELEPHOTO at Hy = 0.2 "
        "the v1 pupil OPL retained 0.13% of the required tilt (slope 8.7e-5 against "
        "0.0684) and the reconstructed wave converged on axis, 209 um from the traced "
        "chief-ray intersection, with a 0.072-wave-P-V residual that looked healthy."
    ),
    "why_it_was_invisible": (
        "the failure is exactly zero on axis, and CHE-30, CHE-32 and CHE-33 all "
        "validated on axis only. It was found by CHE-37 and fixed by CHE-41."
    ),
}

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
    #: Skip moving the OPL's reference from Optiland's launch PLANE onto the
    #: incoming WAVEFRONT (CHE-41). Reproduces the defect CHE-37 found: on axis an
    #: exact no-op, off axis it removes the entire convergence tilt and the wave
    #: converges 209 um from where the rays go. This is the one perturbation in
    #: this class whose effect is zero for every configuration M3.4-M3.8 verified,
    #: which is precisely why it needed a ticket of its own.
    reference_incoming_wavefront: bool = True
    #: Skip folding the producer's quadrature weight into the amplitude (CHE-47),
    #: even when the record carries one. Reproduces CHE-38's uniform-weight
    #: configuration -- the ``3.84e-3`` sensor residual at 787 969 rays -- for a
    #: negative-test comparison against the corrected declaration.
    apply_quadrature_weight: bool = True

    @property
    def is_identity(self) -> bool:
        return (
            self.opl_sign == 1
            and self.transfer_opl_to_plane
            and self.reference_incoming_wavefront
            and self.apply_quadrature_weight
        )

    def describe(self) -> str:
        if self.is_identity:
            return "none"
        parts = []
        if self.opl_sign != 1:
            parts.append("opl_sign_flipped")
        if not self.transfer_opl_to_plane:
            parts.append("opl_plane_transfer_omitted")
        if not self.reference_incoming_wavefront:
            parts.append("incoming_wavefront_reference_omitted")
        if not self.apply_quadrature_weight:
            parts.append("quadrature_weight_omitted")
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


def _plane_tolerance_m(record: ArtifactRecord, declared: DeclaredHandoffPlane) -> float:
    """Plane-agreement bound scaled to the precision the plane was computed in.

    The declared default (1 pm) rests on a stated premise: "both come from the
    same float64 protocol literals, so any real disagreement is a modelling
    error, not round-off". That premise is exactly true for a float64 trace and
    exactly false for a float32 one, where the exit-pupil position comes out of
    ``Paraxial.XPL()`` evaluated in float32. Measured on the M3 singlet
    (CHE-61): the float32 pupil lands 9.2e-11 m from the float64 value, a
    relative difference of 1.4e-06, i.e. about eleven float32 epsilons.

    So the bound becomes ``64 * eps(dtype) * |z|`` where that exceeds the
    declared absolute -- round-off on the quantity itself, derived rather than
    chosen -- and stays at the declared absolute for float64, which is the case
    every existing caller is in.

    This is a widening of a numerical bound, so it needs a physical argument and
    not just an arithmetic one. The quantity being bounded is a defocus, whose
    wavefront error is ``~ (2 pi / lambda) * dz * NA^2 / 2``. At the M3 singlet's
    550 nm and its NA, 5.2e-10 m of axial offset is of order 1e-4 rad -- four
    orders below the 2 pi that would matter and five below the Rayleigh quarter
    wave. A float32 trace cannot place a plane more precisely than this, and
    refusing it would refuse float32 tracing altogether rather than catch a
    modelling error.

    An offset from a genuine plane mismatch is a pupil-to-focus distance --
    millimetres here -- so nothing that this bound now admits is anywhere near
    what it exists to catch.
    """
    dtype = record.dtype
    if not dtype:
        return declared.tolerance_m
    try:
        eps = float(np.finfo(np.dtype(str(dtype))).eps)
    except TypeError:  # pragma: no cover - a non-float dtype on a ray record
        return declared.tolerance_m
    return max(declared.tolerance_m, 64.0 * eps * abs(declared.z_m))


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
    tolerance_m = _plane_tolerance_m(record, declared)
    if offset > tolerance_m:
        raise ContractError(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            (
                f"handoff plane is at z = {float(produced_z)!r} m but the consumer "
                f"declared z = {declared.z_m!r} m; offset {offset:.6e} m exceeds "
                f"tolerance {tolerance_m:.1e} m (record dtype {record.dtype!r})"
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


def _object_space_reference(
    record: ArtifactRecord, bundle: RayBundle, *, applied: bool
) -> dict[str, Any]:
    """Decide whether Optiland's launch-plane reference has to be moved, and to what.

    CHE-41. Three outcomes, and which one occurs is decided by measurement rather
    than by configuration:

    ``applied``
        The record carries the object-space term and its span across the bundle is
        non-zero, so the omission is a *tilt*. The term is added.

    ``pure piston, not applied``
        The term is present and constant across every ray. A constant is removed
        exactly by step 3's chief-ray subtraction, so adding it would only spend
        float precision on a quantity that cannot survive -- the same policy CHE-40
        already applies to the removed reference OPL and the wave side applies to
        ``exp(i k z)``. This is the branch every on-axis configuration takes, and
        taking it is what makes the on-axis declared OPL *bit-identical* to
        CHE-33's rather than merely close to it.

    ``refused``
        The term is absent and the field is not on axis. There is no way to
        reconstruct it downstream: the missing quantity is object-space
        information, and the exit-pupil export does not contain it in any form. A
        pupil OPL declared without it is a converging sphere aimed at the axis
        whatever the field angle, which is a wrong answer that looks entirely
        healthy -- 0.072 waves P-V against its own fitted sphere.
    """
    offset = bundle.provenance.get("object_space_reference_offset_m")
    declaration = bundle.provenance.get("object_space_reference") or {}
    field = bundle.provenance.get("requested_field") or {}
    hx, hy = field.get("Hx"), field.get("Hy")
    on_axis = hx == 0.0 and hy == 0.0

    if offset is None:
        if on_axis:
            return {
                "applied": False,
                "offset_m": None,
                "span_m": None,
                "status": "absent, and the traced field is on axis",
                "reason": (
                    "the record carries no object_space_reference_offset_m. At "
                    "Hx = Hy = 0 the incoming bundle travels along z, so Optiland's "
                    "launch plane IS a wavefront of it and the term would be a "
                    "constant; step 3 removes constants exactly. Accepted for this "
                    "field only."
                ),
                "field": {"Hx": hx, "Hy": hy},
            }
        raise ContractError(
            ContractCode.OBJECT_SPACE_REFERENCE_MISSING,
            (
                "this record was traced at an off-axis field "
                f"(Hx = {hx!r}, Hy = {hy!r}) and carries no "
                "object_space_reference_offset_m, so the optical path it exports is "
                "measured from a plane perpendicular to z rather than from a "
                "wavefront of the incoming bundle"
                + (
                    f"; the ray model declined to supply it because: "
                    f"{declaration.get('unavailable_reason')}"
                    if declaration.get("unavailable_reason")
                    else ""
                )
            ),
            declaration="provenance.object_space_reference_offset_m",
            artifact_id=record.id,
            remedy=(
                "Re-run the ray model with a version that exports the object-space "
                "reference (CHE-41), or trace on axis. This cannot be repaired "
                "downstream: the missing term is n_object * (d0 . r_launch), it is "
                "linear in the LAUNCH coordinate, and the exit-pupil export carries "
                "no object-space coordinate to reconstruct it from. Declaring the OPL "
                "without it produces a clean converging sphere aimed at the axis "
                "instead of at the image point -- on M3-REVERSE-TELEPHOTO at Hy = 0.2, "
                "209 um away, with a residual of 0.072 waves P-V that looks like a "
                "healthy diffraction-limited wavefront (CHE-37)."
            ),
        )

    offset_m = np.asarray(offset, dtype=np.float64)
    if offset_m.shape != (bundle.count,):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                "object_space_reference_offset_m must carry one value per ray, got "
                f"{offset_m.shape} for {bundle.count} rays"
            ),
            declaration="provenance.object_space_reference_offset_m",
            artifact_id=record.id,
        )
    if not np.all(np.isfinite(offset_m)):
        raise ContractError(
            ContractCode.NON_FINITE,
            "object_space_reference_offset_m is not finite",
            declaration="provenance.object_space_reference_offset_m",
            artifact_id=record.id,
        )

    span_m = float(np.ptp(offset_m))
    if not applied:
        return {
            "applied": False,
            "offset_m": offset_m,
            "span_m": span_m,
            "status": "OMITTED -- negative test only",
            "reason": (
                "HandoffPerturbation(reference_incoming_wavefront=False). The declared "
                "OPL is referenced to Optiland's launch plane, which is the defect "
                "CHE-37 measured. Off axis this removes the entire convergence tilt."
            ),
            "field": {"Hx": hx, "Hy": hy},
        }
    if span_m == 0.0:
        return {
            "applied": False,
            "offset_m": offset_m,
            "span_m": 0.0,
            "status": "present, constant across the bundle: a pure piston, not applied",
            "reason": (
                "the term is identical for every ray, so it is a piston and step 3's "
                "chief-ray subtraction removes it exactly. Not added, so the declared "
                "OPL is bit-identical to the pre-CHE-41 value rather than one rounding "
                "away from it -- CHE-40's policy for quantities no single-path PSF can "
                "see."
            ),
            "field": {"Hx": hx, "Hy": hy},
        }
    return {
        "applied": True,
        "offset_m": offset_m,
        "span_m": span_m,
        "status": "applied: the term varies across the bundle, so the omission is a tilt",
        "reason": (
            "n_object * (d0 . r_launch) added to the accumulated path, moving its "
            "reference from Optiland's launch plane onto the plane wavefront of the "
            "incoming bundle through the global origin. Off axis this term IS the "
            "convergence tilt."
        ),
        "field": {"Hx": hx, "Hy": hy},
    }


def _ray_quadrature_weight(
    record: ArtifactRecord, bundle: RayBundle, *, applied: bool
) -> dict[str, Any]:
    """Decide whether a per-ray quadrature weight can be computed and folded in.

    CHE-47. Mirrors :func:`_object_space_reference`'s structure, with one
    difference in kind: a missing quadrature weight is not itself a physics
    error the way a missing off-axis tilt term is. A bundle without one is
    still coherent and still passes ``require_coherent()`` -- it is simply
    unweighted, which is CHE-38's pre-CHE-47 configuration and the ``3.84e-3``
    sensor residual it measured. So the ``unavailable`` branch here falls back
    to :data:`AMPLITUDE_MAPPING` rather than raising.

    The adapter exports only the RAW hexapolar pupil coordinates (it must
    import no coupler -- see ``optiland_adapter._resolve_ray_pupil_sampling``);
    this function does the actual coupler-side physics, calling
    :mod:`multiscale_optics_agent.couplers.quadrature` to turn them into a ring
    index and an absolute per-ray area weight.
    """
    from multiscale_optics_agent.couplers.quadrature import (
        hexapolar_area_weight_m2,
        hexapolar_ring_index,
    )

    pupil_x = bundle.provenance.get("pupil_normalized_x")
    pupil_y = bundle.provenance.get("pupil_normalized_y")
    declaration = bundle.provenance.get("quadrature_weight") or {}
    if pupil_x is None or pupil_y is None:
        return {
            "applied": False,
            "weight_m2": None,
            "status": "unavailable",
            "reason": (
                declaration.get("unavailable_reason")
                or "the record carries no pupil_normalized_x/y (predates CHE-47, or "
                "the adapter could not confirm an un-vignetted hexapolar fan)"
            ),
        }

    num_rings = declaration.get("num_rings")
    aperture_radius_m = declaration.get("aperture_radius_m")
    if num_rings is None or aperture_radius_m is None:
        return {
            "applied": False,
            "weight_m2": None,
            "status": "unavailable",
            "reason": (
                "pupil_normalized_x/y are present but conventions.quadrature_weight "
                "carries no num_rings/aperture_radius_m to scale a weight from"
            ),
        }

    pupil_x = np.asarray(pupil_x, dtype=np.float64)
    pupil_y = np.asarray(pupil_y, dtype=np.float64)
    if pupil_x.shape != (bundle.count,) or pupil_y.shape != (bundle.count,):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                "pupil_normalized_x/y must carry one value per ray, got "
                f"{pupil_x.shape}/{pupil_y.shape} for {bundle.count} rays"
            ),
            declaration="provenance.pupil_normalized_x",
            artifact_id=record.id,
        )

    try:
        ring_index = hexapolar_ring_index(pupil_x, pupil_y, int(num_rings))
        weight_m2 = hexapolar_area_weight_m2(ring_index, int(num_rings), float(aperture_radius_m))
    except ContractError as exc:
        return {
            "applied": False,
            "weight_m2": None,
            "status": "unavailable",
            "reason": f"{exc.code}: {exc}",
        }

    if not applied:
        return {
            "applied": False,
            "weight_m2": weight_m2,
            "status": "available, OMITTED -- negative test only",
            "reason": (
                "HandoffPerturbation(apply_quadrature_weight=False). Reproduces "
                "CHE-38's pre-CHE-47 uniform-weight configuration."
            ),
        }
    return {
        "applied": True,
        "weight_m2": weight_m2,
        "status": "applied",
        "reason": (
            "amplitude = sqrt(weight) * quadrature_weight_m2 (CHE-47), replacing "
            "the equal-weight sum CHE-38 measured as the dominant sensor-plane "
            "residual (3.84e-3 -> ~4e-4 at 787969 rays, on-axis M3-SINGLET-REF)."
        ),
    }


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

    # --- 2b. Move the reference onto the incoming wavefront (CHE-41) ---------
    object_space = _object_space_reference(
        record, bundle, applied=perturbation.reference_incoming_wavefront
    )
    if object_space["applied"]:
        optical_path_at_plane_m = optical_path_at_plane_m + object_space["offset_m"]

    # --- 3. Remove the reference piston (CHE-40 required conditioning) -------
    radius_m = np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1])
    chief_index = int(np.argmin(radius_m))
    reference_opl_m = float(optical_path_at_plane_m[chief_index])
    relative_opl_m = float(perturbation.opl_sign) * (optical_path_at_plane_m - reference_opl_m)

    entrance_pupil_diameter_m = conventions.get("entrance_pupil_diameter_m")
    opl_reference_declaration = (
        f"{OPL_REFERENCE_VERSION}. Zero at the traced chief ray (smallest pupil "
        f"radius, row {chief_index}, rho = {float(radius_m[chief_index]):.6e} m) "
        f"evaluated at the {declared_plane.handoff_plane} plane z = {z_plane_m!r} m; "
        "sign convention 'ray minus chief', so a larger value means a longer optical "
        "path (CHE-30 part 2, and the same sign L1-RAY-01's evaluator declares). The "
        "SURFACE the path is measured from is the plane WAVEFRONT of the incoming "
        "collimated bundle, not Optiland's launch plane: RealRays.opd accumulates from "
        "a plane perpendicular to z (CHE-30 part 3, at an aperture-dependent location "
        f"for EPD = {entrance_pupil_diameter_m!r} m), and CHE-41 adds "
        "n_object * (d0 . r_launch) to move that reference onto a wavefront. "
        f"Object-space term: {object_space['status']}. On axis the term is a constant "
        "and both the launch-plane and wavefront references give the same declared "
        "path; off axis the difference is the entire convergence tilt."
    )

    # --- 4. Declare, through the contract's own named entry points -----------
    bundle = bundle.with_declared_optical_path_length(
        relative_opl_m, reference=opl_reference_declaration
    )

    # --- 4b. Fold in the per-ray quadrature weight, if the record has one (CHE-47) -
    quadrature = _ray_quadrature_weight(
        record, bundle, applied=perturbation.apply_quadrature_weight
    )
    if quadrature["applied"]:
        amplitude = np.sqrt(bundle.weight).astype(np.complex128) * quadrature["weight_m2"].astype(
            np.complex128
        )
        bundle = bundle.with_amplitude_from_weight(
            mapping=AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT, amplitude=amplitude
        )
    else:
        bundle = bundle.with_amplitude_from_weight(mapping=AMPLITUDE_MAPPING)

    declarations = {
        "issue": "CHE-33 (M3.4), off-axis reference by CHE-41",
        "opl_reference_version": OPL_REFERENCE_VERSION,
        "superseded_opl_reference": SUPERSEDED_OPL_REFERENCE,
        "optical_path_length_reference": opl_reference_declaration,
        "optical_path_length_object_space_reference": {
            "declared_reference_surface": (
                "the plane wavefront of the incoming collimated bundle that passes "
                "through the global origin. CHE-41 declares the INCOMING TILTED "
                "WAVEFRONT rather than a chief-ray-referenced frame: it is a physical "
                "surface fixed by the bundle rather than a frame chosen per field, it "
                "leaves the reconstructed field in the same world coordinates the rays "
                "are traced in -- so 'the PSF lands at the traced chief-ray "
                "intersection' is a falsifiable statement rather than a definition -- "
                "and it does not require re-introducing a reference sphere aimed at a "
                "chosen image point, which is the construction whose approximate form "
                "manufactured a 1.0-wave false aberration finding in M3.8. The cost is "
                "that the pupil field carries the tilt and must be sampled at "
                "pitch <= lambda / (2 max|transverse direction cosine|)."
            ),
            "term": "n_object * (d0 . r_launch), from the ray record",
            "status": object_space["status"],
            "reason": object_space["reason"],
            "span_m": object_space["span_m"],
            "span_waves": (
                object_space["span_m"] / bundle.wavelength_m
                if object_space["span_m"] is not None
                else None
            ),
            "field": object_space["field"],
        },
        "quadrature_weight": {
            "declared_reference": (
                "the absolute per-ray pupil/phase-space area element a hexapolar "
                "ring set represents, radial-trapezoid corrected at the center and "
                "outer-ring boundaries (CHE-38 sections 14-15, CHE-47). See "
                "multiscale_optics_agent.couplers.quadrature."
            ),
            "status": quadrature["status"],
            "reason": quadrature["reason"],
            "weight_sum_m2": (
                float(np.sum(quadrature["weight_m2"]))
                if quadrature["weight_m2"] is not None
                else None
            ),
        },
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
        "amplitude_mapping": (
            AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT if quadrature["applied"] else AMPLITUDE_MAPPING
        ),
        "phase_source": (
            "the optical path length only. RealRays.i carries no phase, so a bundle "
            "with a correct amplitude and a wrong OPL is not a degraded field, it is "
            "a different one."
        ),
        "reconstruction_normalization": (
            f"{bundle.reconstruction_normalization!r}: a traced ray set is the physical "
            "ensemble, not a Monte Carlo sample of a spectrum, so SI eq S3/S5's 1/N "
            "must not be applied. "
            + (
                "CHE-47: the amplitude now carries a per-ray quadrature (area) weight, "
                "so the reconstructed discrete power converges under ray refinement "
                "instead of growing as (ray count)^2 -- CHE-33's N^2.0024 finding is "
                "resolved for this bundle, not merely relabeled."
                if quadrature["applied"]
                else "CONSEQUENCE, recorded rather than hidden: with no per-ray area "
                "weight the reconstructed amplitude scales with ray density, so the "
                "field is convergent in shape but not in scale under ray refinement "
                "(CHE-33, measured by CHE-38 as (ray count)^2.0024)."
            )
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
        "opl_reference_version": OPL_REFERENCE_VERSION,
        "object_space_reference_applied": object_space["applied"],
        "object_space_reference_status": object_space["status"],
        "object_space_reference_span_m": object_space["span_m"],
        "object_space_reference_span_waves": (
            object_space["span_m"] / bundle.wavelength_m
            if object_space["span_m"] is not None
            else None
        ),
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
        "quadrature_weight_applied": quadrature["applied"],
        "quadrature_weight_status": quadrature["status"],
        "quadrature_weight_sum_m2": (
            float(np.sum(quadrature["weight_m2"])) if quadrature["weight_m2"] is not None else None
        ),
        "reconstructed_amplitude_abs_min": float(np.min(np.abs(bundle.amplitude))),
        "reconstructed_amplitude_abs_max": float(np.max(np.abs(bundle.amplitude))),
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
