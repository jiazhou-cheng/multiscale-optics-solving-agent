"""Executing a bridge plan on a boundary artifact (CHE-61).

:mod:`multiscale_optics_agent.core.precision` decides *whether* and *how* an
artifact may enter a component; this module is the part that carries it out on a
:class:`RayBundle` or a :class:`ComplexField` and hands back the plan it used, so
the conversion and the record of the conversion cannot come apart.

Two things here are less obvious than they look.

**One plan per artifact, not one per array.** A ray bundle holds real geometry
and a complex amplitude. Planning them independently would let the geometry land
in float32 while the amplitude stays complex128 -- an artifact at two precisions
at once, which the contract layer then has to reject. So the plan is negotiated
on the *real* representation (the geometry is what a device and a precision are
really about) and the complex arrays follow it into the complex dtype of the
same precision.

**Torch enters here or not at all.** ``RayBundle`` refuses a torch tensor
outright, because accepting one would mean some later operation silently
converting it. Optiland's torch backend is exactly the producer of such tensors,
so this module is the single doorway: it plans the namespace change, applies it
(DLPack on CUDA, so the buffer stays on the device), and reports the graph break
that ``.detach()`` causes rather than letting it happen unremarked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from multiscale_optics_agent.core.arrays import array_state, dtype_of, to_namespace
from multiscale_optics_agent.core.precision import (
    ArrayState,
    BridgePlan,
    BridgePolicy,
    ComponentCapabilities,
    DevicePlacement,
    DType,
    plan_bridge,
)
from multiscale_optics_agent.couplers.contracts import ComplexField, RayBundle

__all__ = [
    "bridge_arrays",
    "bridge_complex_field",
    "bridge_ray_bundle",
]


def _companion_dtype(source: DType, planned: DType) -> DType:
    """The dtype ``source`` takes when the plan selected ``planned`` for its peers.

    Same real/complex kind as ``source``, same precision as ``planned``. This is
    what keeps a bundle internally coherent: plan float64 -> float32 for the
    geometry and the amplitude goes complex128 -> complex64, not complex128.
    """
    companion = (
        planned.precision.complex_dtype if source.is_complex else planned.precision.real_dtype
    )
    if companion is None:
        # FP16 has no complex counterpart; the smallest one that exists is the
        # honest answer, and the plan records it as a promotion.
        return DType.COMPLEX64
    return companion


def bridge_arrays(
    arrays: Mapping[str, Any],
    target: ComponentCapabilities,
    *,
    reference: str,
    policy: BridgePolicy = BridgePolicy.SAFE,
    target_device: DevicePlacement | None = None,
    allow_device_transfer: bool = False,
) -> tuple[dict[str, Any], BridgePlan]:
    """Move a set of named arrays into ``target``'s representation under one plan.

    ``reference`` names the array whose state the plan is negotiated on -- the
    real geometry for a ray bundle, the field itself for a complex field.
    ``None`` values pass through untouched, so optional artifact fields need no
    special handling at the call site.
    """
    source_array = arrays[reference]
    plan = plan_bridge(
        array_state(source_array),
        target,
        policy=policy,
        target_device=target_device,
        allow_device_transfer=allow_device_transfer,
        compute_dtype=target.compute_dtype_for(dtype_of(source_array)),
    )

    converted: dict[str, Any] = {}
    for name, array in arrays.items():
        if array is None:
            converted[name] = None
            continue
        if plan.is_identity:
            converted[name] = array
            continue
        converted[name] = to_namespace(
            array,
            namespace=plan.target_namespace,
            # NumPy is a host namespace, so passing it a device would be a
            # contradiction rather than an instruction.
            device=plan.target_device if plan.target_namespace.can_leave_host else None,
            dtype=_companion_dtype(dtype_of(array), plan.target_dtype),
        )
    return converted, plan


def bridge_ray_bundle(
    bundle: RayBundle,
    target: ComponentCapabilities,
    *,
    policy: BridgePolicy = BridgePolicy.SAFE,
    target_device: DevicePlacement | None = None,
    allow_device_transfer: bool = False,
) -> tuple[RayBundle, BridgePlan]:
    """Return ``bundle`` in ``target``'s representation, plus the plan used.

    An already-admissible bundle is returned unchanged and identically -- not
    rebuilt -- so the no-conversion path costs nothing and cannot perturb the
    data it claims to preserve.
    """
    arrays = {
        "positions_m": bundle.positions_m,
        "directions": bundle.directions,
        "amplitude": bundle.amplitude,
        "weight": bundle.weight,
        "optical_path_length_m": bundle.optical_path_length_m,
    }
    converted, plan = bridge_arrays(
        arrays,
        target,
        reference="positions_m",
        policy=policy,
        target_device=target_device,
        allow_device_transfer=allow_device_transfer,
    )
    if plan.is_identity:
        return bundle, plan

    return (
        RayBundle(
            wavelength_m=bundle.wavelength_m,
            reference_plane=bundle.reference_plane,
            frame=bundle.frame,
            weight_semantics=bundle.weight_semantics,
            optical_path_length_reference=bundle.optical_path_length_reference,
            phasor=bundle.phasor,
            polarization=bundle.polarization,
            coherence=bundle.coherence,
            normalization=bundle.normalization,
            reconstruction_normalization=bundle.reconstruction_normalization,
            provenance={**bundle.provenance, "bridge_plan": plan.as_dict()},
            **converted,
        ),
        plan,
    )


def bridge_complex_field(
    field: ComplexField,
    target: ComponentCapabilities,
    *,
    policy: BridgePolicy = BridgePolicy.SAFE,
    target_device: DevicePlacement | None = None,
    allow_device_transfer: bool = False,
) -> tuple[ComplexField, BridgePlan]:
    """Return ``field`` in ``target``'s representation, plus the plan used.

    The Optiland -> coupler -> Chromatix path ends here: a complex128 host field
    entering Chromatix needs ``ALLOW_DOWNCAST`` and comes back complex64 with
    ``lossy=True`` on the plan, rather than being quietly truncated inside
    ``ScalarField.__init__`` where nothing measures the loss.
    """
    converted, plan = bridge_arrays(
        {"u": field.u},
        target,
        reference="u",
        policy=policy,
        target_device=target_device,
        allow_device_transfer=allow_device_transfer,
    )
    if plan.is_identity:
        return field, plan
    return (
        ComplexField(
            u=converted["u"],
            sample_pitch_m=field.sample_pitch_m,
            wavelength_m=field.wavelength_m,
            reference_plane=field.reference_plane,
            frame=field.frame,
            phasor=field.phasor,
            polarization=field.polarization,
            normalization=field.normalization,
            pad_width=field.pad_width,
            padded=field.padded,
            provenance={**field.provenance, "bridge_plan": plan.as_dict()},
        ),
        plan,
    )


def observed_state(artifact: RayBundle | ComplexField) -> ArrayState:
    """The artifact's own observed state -- a thin alias, for symmetry at call sites."""
    return artifact.state
