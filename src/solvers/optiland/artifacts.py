"""Building the two artifacts a trace exports, and persisting their arrays.

Record construction was 550 lines inside the adapter class, as two
``@staticmethod``s taking every input explicitly -- already module functions in
everything but location.

The responsibility is narrow and worth stating: take what the solver produced,
convert it to SI at the boundary, write it, hash it, and declare what the
declarations mean. It traces nothing and resolves no geometry.

One thing here is deliberate and easy to "tidy" away: the dtype the solver
produced is preserved rather than forced to float64. Forcing it was the single
line that made a float32 or GPU trace indistinguishable from a float64 host one
downstream.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from core.arrays import array_state, dtype_of
from core.artifacts import ArtifactRecord
from core.optical_system import (
    OpticalSystemSpec,
)
from core.precision import (
    ArrayNamespace,
)
from core.specs import ArtifactKind, Device, Framework
from solvers.base import (
    ModelRunRequest,
)
from solvers.optiland.builder import build_optiland_system

# Re-exported for callers that reach for them on this module. CHE-91 moved the
# definitions into cohesive siblings, but `solvers.optiland.adapter` stays the
# addressable surface: several tests patch `_import_optiland` and `_resolve_lens`
# *here*, and a patch target is part of a module's contract even when the name is
# private. Keeping the binding means the split needed no test edits, which is the
# whole standard a characterization refactor is held to.
from solvers.optiland.constants import (  # noqa: F401
    _BASELINE_SEED,
    _DEFAULT_HANDOFF_PLANE,
    _DEFAULT_HX,
    _DEFAULT_HY,
    _DEFAULT_NUM_RAYS,
    _DEFAULT_WAVELENGTH,
    _DIRECTION_NORM_TOLERANCE,
    _GEOMETRY_M_PER_MM,
    _MISSING_WAVEFRONT_METADATA,
    _OPD_WARNING,
    _SUPPORTED_BACKENDS,
    _SUPPORTED_HANDOFF_PLANES,
    _SUPPORTED_SAMPLES,
    _VALIDATED_DESIGN_PARAMETER_PATTERN,
    _WAVELENGTH_M_PER_UM,
    MODEL_ID,
)
from solvers.optiland.execution import (
    _direction_norm_tolerance,
    _host_array,
)
from solvers.optiland.provenance import (
    _scientific_array_hash,
)
from solvers.optiland.pupil import (
    _project_rays_to_plane,
)


def _resolve_lens(spec: OpticalSystemSpec) -> Any:
    """Build the system through the one generic construction path."""
    return build_optiland_system(spec)





def build_ray_bundle_artifact(
    request: ModelRunRequest,
    rays: Any,
    be_utils: Any,
    backend_name: str,
    sample_name: str,
    wavelength: float,
    hx: float,
    hy: float,
    num_rays: int,
    run_dir: Path,
    reference_plane_z_mm: float,
    handoff_plane: str = _DEFAULT_HANDOFF_PLANE,
    exit_pupil: dict[str, Any] | None = None,
    image_surface_index: int = 14,
    image_space: dict[str, Any] | None = None,
    object_space_reference: dict[str, Any] | None = None,
    ray_pupil_sampling: dict[str, Any] | None = None,
) -> ArtifactRecord:
    import numpy as np

    image_space = image_space or {}
    image_space_refractive_index = image_space.get("image_space_refractive_index")
    entrance_pupil_diameter_m = image_space.get("entrance_pupil_diameter_m")
    object_at_infinity = image_space.get("object_at_infinity")

    # The explicit execution -> serialization boundary for the trace. The
    # dtype the solver produced is preserved: forcing float64 here was the
    # single line that made a float32 or GPU trace indistinguishable from a
    # float64 host one downstream. For the default numpy/float64 path this is
    # a no-op, so L1-RAY-01's recorded fingerprint is unchanged.
    traced_state = array_state(rays.x)
    native = {
        name: _host_array(be_utils, value)
        for name, value in (
            ("x", rays.x),
            ("y", rays.y),
            ("z", rays.z),
            ("L", rays.L),
            ("M", rays.M),
            ("N", rays.N),
            ("intensity", rays.i),
            ("wavelength_um", rays.w),
            ("opd_native", rays.opd),
        )
    }
    shapes = {name: array.shape for name, array in native.items()}
    if not native["x"].size:
        raise ValueError("Optiland returned an empty surviving-ray set.")
    if len(set(shapes.values())) != 1 or native["x"].ndim != 1:
        raise ValueError(f"RealRays arrays must be equal-length 1-D arrays; got {shapes!r}.")
    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(array))) for name, array in native.items()
    }
    if any(nonfinite.values()):
        raise ValueError(f"Optiland returned non-finite scientific output: {nonfinite!r}.")

    direction_norm = np.sqrt(native["L"] ** 2 + native["M"] ** 2 + native["N"] ** 2)
    max_direction_norm_error = float(np.max(np.abs(direction_norm - 1.0)))
    # Bound scaled to the precision the trace ACTUALLY ran in. The float64
    # value is untouched; a float32 trace is held to float32 round-off
    # instead of being failed for arithmetic it never claimed to do.
    norm_tolerance = _direction_norm_tolerance(dtype_of(native["L"]))
    if backend_name == "numpy" and max_direction_norm_error > norm_tolerance:
        raise ValueError(
            "Optiland direction vectors are not unit norm: "
            f"max error {max_direction_norm_error:.17g} exceeds "
            f"{norm_tolerance:.1e} for {dtype_of(native['L'])}."
        )

    # At the image surface the exported coordinates are the traced ones,
    # untouched -- which is what keeps L1-RAY-01's fingerprint bit-identical.
    # At the exit pupil they are each ray's image-space asymptote evaluated at
    # the pupil plane; directions are unchanged either way.
    if handoff_plane == "exit_pupil":
        projected = _project_rays_to_plane(rays, be_utils, reference_plane_z_mm)
        position = {
            "x": projected["x_mm"],
            "y": projected["y_mm"],
            "z": projected["z_mm"],
        }
        max_projection_step_mm = projected["max_abs_step_mm"]
    else:
        position = {"x": native["x"], "y": native["y"], "z": native["z"]}
        max_projection_step_mm = 0.0

    arrays = {
        "x_m": position["x"] * _GEOMETRY_M_PER_MM,
        "y_m": position["y"] * _GEOMETRY_M_PER_MM,
        "z_m": position["z"] * _GEOMETRY_M_PER_MM,
        "L": native["L"],
        "M": native["M"],
        "N": native["N"],
        "intensity": native["intensity"],
        "wavelength_m": native["wavelength_um"] * _WAVELENGTH_M_PER_UM,
        "opd_native": native["opd_native"],
        # The trace API exposes survivors only. This derived boolean states
        # the membership of each exported row; it is not an Optiland pupil mask.
        "survived": np.ones(native["x"].shape, dtype=np.bool_),
    }
    # CHE-41. The object-space reference travels in the SAME file but under a
    # separate hash, and `arrays` is left untouched: `scientific_array_sha256`
    # is the frozen identity of the traced ray set (CHE-32 pinned
    # e494af41... on M3-SINGLET-REF, L1-RAY-01's 43dab1ee... downstream of the
    # same function), and adding a column to it would move a fingerprint that
    # nothing about this change has any business moving. The new arrays are
    # additional *object-space* data, not a revision of the traced output.
    object_space = object_space_reference or {"available": False}
    object_space_arrays: dict[str, Any] = {}
    if object_space.get("available"):
        object_space_arrays = {
            "object_space_reference_offset_m": (
                np.asarray(object_space["offset_native"], dtype=np.float64) * _GEOMETRY_M_PER_MM
            ),
            "launch_x_m": (
                np.asarray(object_space["launch_x_native"], dtype=np.float64)
                * _GEOMETRY_M_PER_MM
            ),
            "launch_y_m": (
                np.asarray(object_space["launch_y_native"], dtype=np.float64)
                * _GEOMETRY_M_PER_MM
            ),
            "launch_z_m": (
                np.asarray(object_space["launch_z_native"], dtype=np.float64)
                * _GEOMETRY_M_PER_MM
            ),
        }
        if object_space_arrays["object_space_reference_offset_m"].shape != arrays["x_m"].shape:
            raise ValueError(
                "the object-space reference term does not match the exported ray "
                f"count: {object_space_arrays['object_space_reference_offset_m'].shape} "
                f"vs {arrays['x_m'].shape}."
            )

    # CHE-47 (M3.9R extension). Same reasoning as the object-space block above:
    # additional per-ray columns, not a revision of the traced set, so they do
    # not enter `scientific_hash` below. Raw pupil coordinates only -- the ring
    # index and area weight are coupler physics, computed downstream in
    # optiland_handoff.py, never here (see _resolve_ray_pupil_sampling).
    quadrature = ray_pupil_sampling or {"available": False}
    quadrature_arrays: dict[str, Any] = {}
    if quadrature.get("available"):
        quadrature_arrays = {
            "pupil_normalized_x": np.asarray(quadrature["pupil_x"], dtype=np.float64),
            "pupil_normalized_y": np.asarray(quadrature["pupil_y"], dtype=np.float64),
        }
        if quadrature_arrays["pupil_normalized_x"].shape != arrays["x_m"].shape:
            raise ValueError(
                "the regenerated pupil sampling does not match the exported ray "
                f"count: {quadrature_arrays['pupil_normalized_x'].shape} vs "
                f"{arrays['x_m'].shape}."
            )

    scientific_hash = _scientific_array_hash(arrays)
    summary_metrics = {
        "max_direction_norm_error": max_direction_norm_error,
        "direction_norm_tolerance": _DIRECTION_NORM_TOLERANCE,
        "all_finite": True,
        "intensity_min": float(np.min(native["intensity"])),
        "intensity_max": float(np.max(native["intensity"])),
        "intensity_sum": float(np.sum(native["intensity"])),
        "opd_native_min": float(np.min(native["opd_native"])),
        "opd_native_max": float(np.max(native["opd_native"])),
        "x_m_min": float(np.min(arrays["x_m"])),
        "x_m_max": float(np.max(arrays["x_m"])),
        "y_m_min": float(np.min(arrays["y_m"])),
        "y_m_max": float(np.max(arrays["y_m"])),
        "z_m_min": float(np.min(arrays["z_m"])),
        "z_m_max": float(np.max(arrays["z_m"])),
    }

    path = run_dir / "rays.npz"
    np.savez(path, **arrays, **object_space_arrays, **quadrature_arrays)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return ArtifactRecord(
        id=f"{request.node_id}-rays-{uuid.uuid4().hex[:8]}",
        kind=ArtifactKind.RAY_BUNDLE,
        uri=str(path),
        sha256=digest,
        shape=tuple(native["x"].shape),
        dtype=str(native["x"].dtype),
        framework=Framework.PYTORCH if backend_name == "torch" else Framework.NUMPY,
        # OBSERVED, from the traced tensors, not from the request. A torch
        # trace pinned to cuda:0 says so here; before CHE-61 this was
        # hard-coded to CPU and a GPU trace was indistinguishable from a host
        # one in the record it produced.
        device=traced_state.device.to_spec_device(),
        units="SI for position and wavelength; dimensionless direction/intensity; native OPD",
        metadata={
            # Where the trace actually executed, and the fact that the .npz
            # alongside it is an explicit persistence copy rather than
            # evidence that the computation happened on the host.
            "execution": traced_state.as_dict(),
            "serialization": {
                "boundary": "explicit_persistence",
                "host_copy": traced_state.namespace is not ArrayNamespace.NUMPY,
                "kind": (
                    "serialization"
                    if traced_state.namespace is not ArrayNamespace.NUMPY
                    else "already_on_host"
                ),
                "reason": "npz persistence requires host bytes",
                "mechanism": (
                    "optiland.backend.utils.to_numpy, which calls "
                    "tensor.detach().cpu().numpy() -- a host transfer AND an "
                    "autodiff graph break, both confined to this boundary"
                ),
            },
            "length_unit": "m",
            "native_length_unit": "mm",
            "native_to_si_scale": _GEOMETRY_M_PER_MM,
            "wavelength_unit": "m",
            "native_wavelength_unit": "um",
            "native_wavelength_to_si_scale": _WAVELENGTH_M_PER_UM,
            "wavelength_m": float(arrays["wavelength_m"][0]),
            "coordinate_fields": ["x_m", "y_m", "z_m"],
            "direction_fields": ["L", "M", "N"],
            "intensity_field": "intensity",
            "intensity_is_not_amplitude": (
                "RealRays.i is a real-valued per-ray intensity, not a "
                "complex amplitude or Jones vector; no 'amplitude' or "
                "'polarization' array is produced by this adapter (see "
                "module docstring)."
            ),
            "requested_Hx": hx,
            "requested_Hy": hy,
            "requested_num_rays": num_rays,
            "traced_num_rays": int(native["x"].shape[0]),
            "survival_field": "survived",
            "survival_semantics": (
                "Every exported row survived the sequential trace. Optic.trace "
                "does not expose rejected input candidates, so invalid and "
                "vignetted counts before survivor filtering are unavailable."
            ),
            # CHE-32: M3.3 must state how the pupil boundary is represented,
            # and the answer is that Optiland does not represent it. RealRays
            # carries no mask (probed: no `mask`/`pupil_mask`/`vignetted`
            # attribute) and Optic.trace returns survivors only, so the
            # boundary is implicit in WHICH rows exist. The measured extent
            # below is derived from the survivors; it is emphatically not an
            # Optiland pupil mask, and `survived` -- an all-true derived
            # boolean -- must not be promoted into one.
            "pupil_boundary": {
                "representation": "implicit_in_surviving_rays",
                "mask_available_from_optiland": False,
                "derived_from": "measured extent of the traced survivors",
                "measured_semi_extent_x_m": float(np.max(np.abs(arrays["x_m"]))),
                "measured_semi_extent_y_m": float(np.max(np.abs(arrays["y_m"]))),
                "paraxial_semi_diameter_m": (
                    exit_pupil["diameter_mm"] / 2.0 * _GEOMETRY_M_PER_MM
                    if exit_pupil is not None
                    else None
                ),
                "warning": (
                    "the measured extent is a property of the traced set, not of "
                    "the aperture, and it does not bracket the paraxial diameter "
                    "in a predictable direction: sampling density pulls it inward "
                    "while pupil aberration pushes real marginal rays outward, and "
                    "on both M3 systems the measured value lands slightly ABOVE "
                    "the paraxial semi-diameter. A consumer needing the aperture "
                    "must use the paraxial diameter, not this."
                ),
            },
            "sample": sample_name,
            "backend": backend_name,
            "scientific_array_sha256": scientific_hash,
            "summary_metrics": summary_metrics,
            "conventions": {
                "axes": "x,y,z right-handed Cartesian; propagation is +z",
                "handedness": "right-handed",
                "direction": "(L,M,N) direction cosines in the same frame",
                "reference_plane": (
                    f"exit pupil, read from Paraxial.XPL()/XPD() "
                    f"({'virtual' if exit_pupil and exit_pupil['is_virtual'] else 'real'})"
                    if handoff_plane == "exit_pupil"
                    else (f"final traced image surface, surface index {image_surface_index}")
                ),
                "reference_plane_z_m": reference_plane_z_mm * _GEOMETRY_M_PER_MM,
                "handoff_plane": handoff_plane,
                "exit_pupil": (
                    {
                        "source": "optic.paraxial.XPL() and XPD(), read not constructed",
                        "location_from_image_m": (
                            exit_pupil["location_from_image_mm"] * _GEOMETRY_M_PER_MM
                        ),
                        "z_m": exit_pupil["z_mm"] * _GEOMETRY_M_PER_MM,
                        "diameter_m": exit_pupil["diameter_mm"] * _GEOMETRY_M_PER_MM,
                        "is_virtual": exit_pupil["is_virtual"],
                        "refracting_surfaces_beyond_pupil_z_m": [
                            z * _GEOMETRY_M_PER_MM
                            for z in exit_pupil["refracting_surfaces_beyond_pupil_z_mm"]
                        ],
                        "position_semantics": (
                            "x_m/y_m are each ray's IMAGE-SPACE ASYMPTOTE evaluated "
                            "at the pupil plane, not a physical intersection. The "
                            "exit pupil is the image of the stop in image space and "
                            "is frequently virtual -- when is_virtual is true the "
                            "extended line passes back through glass the ray never "
                            "travelled in that state. This is the construction the "
                            "exit pupil is defined by, and it is what a "
                            "wavefront-over-the-pupil calculation wants, but it is "
                            "not 'where the ray is'."
                        ),
                        "max_projection_step_m": (max_projection_step_mm * _GEOMETRY_M_PER_MM),
                        "directions_unchanged": (
                            "the projection is a reparameterization along each ray, "
                            "so (L,M,N) are the traced values and no optical path "
                            "was added or removed; OPL handling is M3.4's (CHE-33)"
                        ),
                    }
                    if exit_pupil is not None
                    else None
                ),
                "opd_field": "opd_native",
                "opd_unit": "Optiland native value (geometry-scale mm expected)",
                # CHE-30 established all four parts of this convention against
                # manufactured geometries with closed-form answers, each with the
                # competing hypothesis it rules out; M1's "unverified" is
                # superseded. Declaring it here does NOT admit opd_native as an
                # optical path length: the contract layer still refuses a bundle
                # whose OPL was never declared, because the value carried below is
                # absolute and its zero moves with the aperture.
                "opd_reference": (
                    "ray launch state, where RealRays seeds the accumulator to "
                    "zero. For this infinite-object system Optic.trace aims that "
                    "plane at positions[1] - (EPD - min(positions[1:-1])), so the "
                    "zero MOVES when the aperture changes -- the value is only "
                    "meaningful alongside entrance_pupil_diameter_m below."
                ),
                "opd_sign": (
                    "non-negative accumulation; larger means longer optical path. "
                    "standard_surface.py adds be.abs(t * n_pre) per surface, so it "
                    "never decreases along a purely refractive path."
                ),
                "opd_quantity": "absolute_accumulated_optical_path_length",
                "opd_is_relative_to_chief_ray": False,
                # CHE-41 tested this off axis, where it is testable: the chief
                # ray's own opd_native is 1.0e4 waves rather than zero, and the
                # accumulator's zero sits on a plane, so the field name's
                # promise of a chief-ray-relative OPD remains false. What CHE-30
                # could not see on axis is the SHAPE of that plane's failure,
                # stated in the next two entries.
                "opd_is_relative_to_chief_ray_verified_off_axis": "CHE-41, Hy = 0.2",
                "opd_reference_surface": (
                    "a plane PERPENDICULAR TO Z at the launch, which is a "
                    "wavefront only for a bundle travelling along z"
                ),
                "opd_omits_incoming_wavefront_tilt": True,
                "opd_omitted_term": (
                    "n_object * (d0 . r_launch), exported below as "
                    "object_space_reference_offset_m. Linear in the launch "
                    "coordinate, so on axis it is a piston that cancels in any "
                    "chief-ray subtraction and off axis it is the whole "
                    "convergence tilt: on M3-REVERSE-TELEPHOTO at Hy = 0.2 the "
                    "pupil OPL carries 0.13% of the tilt the geometry requires "
                    "without it (CHE-41, CHE-37)."
                ),
                "opd_convention_verified_by": "CHE-30, extended off axis by CHE-41",
                # CHE-41: the object-space information Optic.trace throws away.
                # Present as a declaration even when unavailable, because an
                # absent term is what a consumer must refuse an off-axis field
                # on, and it cannot refuse on a key that is not there.
                "object_space_reference": {
                    "available": bool(object_space.get("available")),
                    "unavailable_reason": object_space.get("unavailable_reason"),
                    "array": (
                        "object_space_reference_offset_m"
                        if object_space.get("available")
                        else None
                    ),
                    "unit": "m",
                    "quantity": (
                        "n_object * (d0 . r_launch): the optical path from the "
                        "plane wavefront of the incoming collimated bundle that "
                        "passes through the global origin, to each ray's launch "
                        "point. ADD it to the native accumulated path to obtain a "
                        "path measured from that wavefront. It is not applied "
                        "here; opd_native is exported exactly as Optiland "
                        "produced it."
                    ),
                    "launch_geometry": (
                        "collimated_bundle_launched_on_a_plane_perpendicular_to_z"
                        if object_space.get("available")
                        else None
                    ),
                    "launch_direction": object_space.get("launch_direction"),
                    "launch_plane_z_m": (
                        object_space["launch_plane_z_native"] * _GEOMETRY_M_PER_MM
                        if object_space.get("available")
                        else None
                    ),
                    "object_space_refractive_index": object_space.get(
                        "object_space_refractive_index"
                    ),
                    "span_m": (
                        object_space["span_native"] * _GEOMETRY_M_PER_MM
                        if object_space.get("available")
                        else None
                    ),
                    "span_is_zero_on_axis": (
                        "a zero span means the term is a pure piston, which the "
                        "consumer's chief-ray subtraction removes exactly. That is "
                        "the measured statement of 'this field is on axis', and it "
                        "is what keeps the on-axis declaration bit-identical to "
                        "CHE-33's."
                    ),
                    "source": (
                        "ray_tracer.ray_generator.generate_rays over the same "
                        "hexapolar distribution Optic.trace builds. Optic.trace "
                        "returns only the traced rays and retains no launch state, "
                        "so it is regenerated; the regeneration is checked to be "
                        "collimated, planar, finite and row-matched before the "
                        "term is offered, and CHE-41's probe re-traces it and "
                        "requires x, y and opd to match the shipping trace exactly."
                    ),
                    "verified_by": "CHE-41",
                    "sha256": (
                        _scientific_array_hash(object_space_arrays)
                        if object_space_arrays
                        else None
                    ),
                },
                # CHE-47 (M3.9R extension): the RAW hexapolar pupil coordinates a
                # per-ray quadrature weight is computed from downstream (CHE-38
                # identified the missing weight as the dominant sensor-plane
                # residual). Present as a declaration even when unavailable, for the
                # same reason as object_space_reference above. This adapter exports
                # coordinates only, never a ring index or a weight: that computation
                # is coupler physics (couplers.quadrature),
                # and this module must import no coupler -- see
                # _resolve_ray_pupil_sampling.
                "quadrature_weight": {
                    "available": bool(quadrature.get("available")),
                    "unavailable_reason": quadrature.get("unavailable_reason"),
                    "pupil_x_array": (
                        "pupil_normalized_x" if quadrature.get("available") else None
                    ),
                    "pupil_y_array": (
                        "pupil_normalized_y" if quadrature.get("available") else None
                    ),
                    "pupil_coordinate_unit": "dimensionless, normalized to the unit disk",
                    "quantity": (
                        "the normalized entrance-pupil coordinates (Px, Py) Optic.trace "
                        "sampled the hexapolar fan on, regenerated because Optic.trace "
                        "keeps no record of them. A CONSUMER computes each ray's ring "
                        "index and the absolute pupil/phase-space area element that ring "
                        "represents (couplers.quadrature."
                        "hexapolar_ring_index / hexapolar_area_weight_m2), radial-"
                        "trapezoid corrected at the two boundaries (center 3/4, outer "
                        "ring 1/2, interior 1x the nominal cell pi*a^2/(3*num_rings^2)), "
                        "then multiplies it onto sqrt(intensity) to get a per-ray "
                        "amplitude whose coherent sum is a converged quadrature of the "
                        "aperture. Not applied here; intensity is exported exactly as "
                        "Optiland produced it."
                    ),
                    "num_rings": quadrature.get("num_rings"),
                    "aperture_radius_m": quadrature.get("aperture_radius_m"),
                    "source": (
                        "optiland.distribution.create_distribution('hexapolar') "
                        "regenerated over the same num_rings Optic.trace used, matched "
                        "row for row against the traced set (CHE-38 sections 14-15, "
                        "CHE-47)."
                    ),
                    "verified_by": "CHE-47",
                    "sha256": (
                        _scientific_array_hash(quadrature_arrays) if quadrature_arrays else None
                    ),
                },
                "entrance_pupil_diameter_m": entrance_pupil_diameter_m,
                "object_at_infinity": object_at_infinity,
                # Read from the prescription, not assumed. A consumer moving an
                # optical path between the image surface and any other plane in
                # image space needs this index, and "it is air" is a property of
                # these two systems rather than of lenses.
                "image_space_refractive_index": image_space_refractive_index,
                "polarization": "missing; RealRays provides no polarization state",
                "coherence": "missing; sequential rays are not a coherent complex field",
                "normalization": "raw Optiland ray intensity/weight; not normalized",
                "sampling": (
                    "hexapolar pupil distribution; requested value is density, not output count"
                ),
            },
        },
    )

def build_wavefront_artifact(
    request: ModelRunRequest,
    rays: Any,
    be_utils: Any,
    backend_name: str,
    sample_name: str,
    wavelength: float,
    run_dir: Path,
) -> tuple[ArtifactRecord, list[str]]:
    import numpy as np

    x = be_utils.to_numpy(rays.x) * _GEOMETRY_M_PER_MM
    y = be_utils.to_numpy(rays.y) * _GEOMETRY_M_PER_MM
    opd = be_utils.to_numpy(rays.opd)
    traced_wavelength = be_utils.to_numpy(rays.w) * _WAVELENGTH_M_PER_UM

    path = run_dir / "wavefront.npz"
    np.savez(path, x_m=x, y_m=y, opd_native=opd, wavelength_m=traced_wavelength)
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    warnings = [
        "Output port 'wavefront' registry metadata declares "
        f"{_MISSING_WAVEFRONT_METADATA!r} but "
        "optiland.rays.real_rays.RealRays exposes neither a polarization "
        "state nor a pupil mask (only x, y, z, L, M, N, i, opd, w). These "
        "metadata keys are intentionally left unpopulated rather than "
        "fabricated (repository scientific-contract requirements)."
    ]

    artifact = ArtifactRecord(
        id=f"{request.node_id}-wavefront-{uuid.uuid4().hex[:8]}",
        kind=ArtifactKind.WAVEFRONT_SAMPLES,
        uri=str(path),
        sha256=digest,
        shape=tuple(x.shape),
        dtype=str(x.dtype),
        framework=Framework.PYTORCH if backend_name == "torch" else Framework.NUMPY,
        device=Device.CPU,
        units=None,
        metadata={
            "length_unit": "m",
            "wavelength_unit": "m",
            "wavelength": float(traced_wavelength[0]) if traced_wavelength.size else None,
            "coordinate_fields": ["x_m", "y_m"],
            "optical_path_length_source": (
                "RealRays.opd -- convention not independently verified "
                "(absolute optical path length vs. OPD relative to a "
                "chief/reference ray); see conventions.md."
            ),
            "missing_declared_metadata": _MISSING_WAVEFRONT_METADATA,
            "sample": sample_name,
            "backend": backend_name,
        },
    )
    return artifact, warnings
