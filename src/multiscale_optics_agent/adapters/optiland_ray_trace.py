"""Trace a *caller-supplied* ray population through Optiland (CHE-70, Phase 1).

The existing :class:`~multiscale_optics_agent.adapters.optiland_adapter.OptilandAdapter`
traces rays Optiland itself generates from a field and a pupil sampling. That is
the right shape for a pupil benchmark and the wrong shape here: CHE-70's rays
come from an angular-spectrum decomposition of a wave field, so their positions
and directions are decided on the wave side and Optiland's job is only to
propagate them.

This is therefore a *second, narrow* capability rather than a change to the
first, and it is deliberately small: one entry point, no system construction
(the caller passes a built ``Optic``), no persistence, no field/pupil concepts.

Torch lives inside this function
--------------------------------
Boundary artifacts are NumPy/JAX by contract -- ``RayBundle`` refuses a torch
tensor outright, so that a conversion cannot happen implicitly. Optiland's CUDA
capability is torch-backend-only. Both facts hold, so the torch representation
exists *only between* the two ``bridge_arrays`` calls below: a JAX batch goes in,
a JAX batch comes out, and the two conversions are planned by
``core.precision.plan_bridge`` and reported, rather than performed by whichever
operation touched the array first. On CUDA both directions go through DLPack, so
the buffer never leaves the device.

Only a real intensity crosses into Optiland
-------------------------------------------
``RealRays.i`` is a real bookkeeping quantity, and the pinned solver has no
complex ray field at all. So the complex amplitude is **not** among the arrays
bridged into torch: it stays on the wave side and is re-attached by ray id after
the trace. ``|a|^2`` is computed on the wave side and bridged in as an intensity
so Optiland's clipping bookkeeping is meaningful, and the value that comes back
is read for exactly one purpose -- deciding which rays were clipped.

Skipping the object surface
---------------------------
A built ``Optic`` always carries an object surface, and for an object at infinity
it sits at ``z = -inf``. Rays supplied at a real plane must not be traced through
it, so ``skip=1`` is the default and the first non-skipped surface is *checked*
against the plane the batch declares. Getting that wrong is not a crash -- it is
a silently different optical system -- so it is a structured error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.core.arrays import array_state, xp_for
from multiscale_optics_agent.core.capabilities import OPTILAND_CAPABILITIES
from multiscale_optics_agent.core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from multiscale_optics_agent.core.precision import (
    ArrayNamespace,
    ArrayState,
    BridgePlan,
    BridgePolicy,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    Precision,
)
from multiscale_optics_agent.couplers.bridge import bridge_arrays
from multiscale_optics_agent.couplers.coherent_batch import (
    AMPLITUDE_SIDECAR_RULE,
    OPTILAND_INTENSITY_RULE,
    CoherentRayBatch,
    metres_to_micrometres,
    metres_to_millimetres,
    millimetres_to_metres,
)
from multiscale_optics_agent.couplers.contracts import (
    ContractCode,
    ContractError,
    ReferencePlane,
)

__all__ = [
    "OptilandExecutionState",
    "TracePlans",
    "configure_optiland_execution",
    "plan_trace_bridges",
    "surface_positions_m",
    "trace_ray_batch",
]

#: How far the first traced surface may sit from the batch's declared launch
#: plane before the pairing is refused, in metres. A nanometre: the two are the
#: *same* plane by construction, so any real difference is a setup error, and a
#: float32 representation of a 60 um position is good to about 4e-12 m.
_PLANE_TOLERANCE_M = 1.0e-9


@dataclass(frozen=True)
class OptilandExecutionState:
    """What Optiland was told, and what it reported back. Never one for the other."""

    requested_backend: str
    requested_device: str
    requested_precision: str
    observed_device: str
    observed_precision: str
    grad_enabled: bool
    torch_version: str | None
    cuda_version: str | None
    cuda_device_name: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": {
                "backend": self.requested_backend,
                "device": self.requested_device,
                "device_note": (
                    "optiland 0.6.0 set_device takes 'cpu' or 'cuda' only; the "
                    "physical GPU is selected by the container's visible devices"
                ),
                "precision": self.requested_precision,
            },
            "observed": {
                "device": self.observed_device,
                "precision": self.observed_precision,
                "grad_enabled": self.grad_enabled,
            },
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cuda_device_name": self.cuda_device_name,
        }


def _import_optiland() -> tuple[Any, Any]:
    try:
        import optiland.backend as be  # type: ignore[import-untyped]
        from optiland.rays import RealRays  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - environment failure
        raise AdapterDependencyError(
            f"optiland could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    return be, RealRays


def configure_optiland_execution(
    *,
    device: DevicePlacement,
    precision: Precision,
    enable_grad: bool = False,
) -> OptilandExecutionState:
    """Drive Optiland's global execution controls and report the observed state.

    ``device=cuda`` selects the torch backend, because it is the only one with a
    device concept -- ``set_device`` raises ``BackendCapabilityError`` on numpy,
    which ``OPTILAND_CAPABILITIES.device_namespaces`` already declares. That
    declaration is honoured here rather than re-derived.

    All three setters are process-global and none is thread-safe in optiland
    0.6.0, so all three are set on every call rather than inherited from
    whatever a previous call left behind.
    """
    be, _ = _import_optiland()
    namespaces = OPTILAND_CAPABILITIES.device_namespaces[device.kind]
    namespace = (
        ArrayNamespace.TORCH if ArrayNamespace.TORCH in namespaces else ArrayNamespace.NUMPY
    )
    backend_name = "torch" if namespace is ArrayNamespace.TORCH else "numpy"

    torch_module: Any = None
    cuda_name: str | None = None
    if backend_name == "torch":
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise AdapterDependencyError(f"the torch backend needs torch: {exc}") from exc
        torch_module = torch
        if device.kind is DeviceKind.CUDA:
            if not torch.cuda.is_available():
                raise UnsupportedCapabilityError(
                    "FAIL_ENVIRONMENT_NO_CUDA: torch cannot reach a CUDA device "
                    f"(torch {torch.__version__}, torch.version.cuda="
                    f"{torch.version.cuda!r}). There is deliberately no silent "
                    "fallback to the CPU; run under `./run.sh --gpu`."
                )
            cuda_name = torch.cuda.get_device_name(device.index or 0)

    be.set_backend(backend_name)
    be.set_precision(str(precision.real_dtype))
    # Optiland's torch backend takes exactly "cpu" or "cuda" and has no device
    # ordinal at all -- it uses whichever CUDA device torch considers current. So
    # the *kind* is what is set here, and which physical GPU that is comes from
    # the container's visible device set (`MOA_GPUS` through `run.sh --gpu`), not
    # from anything this process can choose. Passing "cuda:0" raises ValueError.
    device_string = str(device.kind)
    if backend_name == "torch":
        be.set_device(device_string)
        observed_device = str(be.get_device())
    else:
        observed_device = "cpu (numpy backend has no device concept)"
    if enable_grad:
        be.grad_mode.enable()
    else:
        be.grad_mode.disable()

    raw_precision = be.get_precision()
    return OptilandExecutionState(
        requested_backend=backend_name,
        requested_device=device_string,
        requested_precision=str(precision.real_dtype),
        observed_device=observed_device,
        observed_precision=(
            f"float{int(raw_precision)}" if isinstance(raw_precision, int) else str(raw_precision)
        ),
        grad_enabled=_grad_enabled(be),
        torch_version=None if torch_module is None else str(torch_module.__version__),
        cuda_version=None if torch_module is None else str(torch_module.version.cuda),
        cuda_device_name=cuda_name,
    )


def _solver_module(namespace: ArrayNamespace) -> Any:
    """``numpy`` or ``torch``, for array plumbing inside the solver boundary."""
    if namespace is ArrayNamespace.TORCH:
        import torch

        return torch
    if namespace is ArrayNamespace.NUMPY:
        return np
    raise UnsupportedCapabilityError(
        f"Optiland executes in numpy or torch, not {namespace}"
    )


def _grad_enabled(be: Any) -> bool:
    """Read Optiland's grad-mode flag under either spelling the pinned API uses."""
    mode = be.grad_mode
    for attribute in ("requires_grad", "enabled", "is_enabled"):
        value = getattr(mode, attribute, None)
        if isinstance(value, bool):
            return value
    return bool(getattr(mode, "_enabled", False))


def surface_positions_m(lens: Any) -> list[float]:
    """Every surface's axial position in metres, read from the built system."""
    import optiland.backend as be  # local: keeps the solver import inside the adapter

    positions = np.asarray(be.to_numpy(lens.surfaces.positions)).ravel()
    return [millimetres_to_metres(float(value)) for value in positions]


def _require_launch_plane(lens: Any, plane: ReferencePlane, skip: int) -> list[float]:
    positions = surface_positions_m(lens)
    if skip >= len(positions):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"skip={skip} leaves no surface to trace in a {len(positions)}-surface system",
            declaration="skip",
        )
    first = positions[skip]
    if not math.isfinite(first) or abs(first - plane.z_m) > _PLANE_TOLERANCE_M:
        raise ContractError(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            (
                f"the first traced surface sits at z = {first!r} m but the batch "
                f"declares its launch plane {plane.name!r} at z = {plane.z_m!r} m; "
                "tracing would propagate the rays through a different system than "
                "the one the wave side decomposed on"
            ),
            declaration="reference_plane",
            remedy=(
                "Build the system so the surface following the object surface is "
                "the launch plane, and pass skip=1."
            ),
        )
    return positions


@dataclass(frozen=True)
class TracePlans:
    """The two bridge plans one trace needs, negotiated once for a whole sweep.

    Chunking does not change any array's dtype, device or namespace, so the plans
    are constant across chunks. Negotiating them once and recording them in the
    manifest is what makes "the conversion that happened" a fact in the result
    rather than something re-decided a thousand times inside a loop.
    """

    inbound: BridgePlan
    outbound: BridgePlan
    #: The capability the outbound plan was negotiated against. Held here rather
    #: than passed to every trace, so the plan recorded in the manifest and the
    #: conversion actually executed cannot come from two different sources.
    home: ComponentCapabilities

    def as_dict(self) -> dict[str, Any]:
        return {
            "into_optiland": self.inbound.as_dict(),
            "out_of_optiland": self.outbound.as_dict(),
            "home_component": self.home.component,
        }


def plan_trace_bridges(
    batch: CoherentRayBatch,
    *,
    home: ComponentCapabilities,
    device: DevicePlacement,
    policy: BridgePolicy = BridgePolicy.SAFE,
) -> TracePlans:
    """Plan wave-side -> Optiland and Optiland -> wave-side for this batch's state.

    ``home`` is the capability the traced arrays must return to -- the
    reconstruction coupler's, since that is what consumes them next.
    """
    from multiscale_optics_agent.core.precision import plan_bridge

    source = array_state(batch.bundle.positions_m)
    inbound = plan_bridge(source, OPTILAND_CAPABILITIES, policy=policy, target_device=device)
    traced_state = ArrayState(
        inbound.target_dtype, inbound.target_device, ArrayNamespace.TORCH
    )
    outbound = plan_bridge(traced_state, home, policy=policy, target_device=device)
    return TracePlans(inbound=inbound, outbound=outbound, home=home)


def trace_ray_batch(
    batch: CoherentRayBatch,
    lens: Any,
    *,
    image_plane: ReferencePlane,
    plans: TracePlans,
    skip: int = 1,
) -> tuple[CoherentRayBatch, dict[str, Any]]:
    """Propagate ``batch`` through ``lens`` and return the traced batch.

    ``batch`` arrives in a compute namespace (NumPy or JAX) and leaves in the
    same one. The intermediate torch representation is created and discarded
    inside this call, under ``plans``.
    """
    be, real_rays_cls = _import_optiland()
    bundle = batch.bundle
    surfaces_m = _require_launch_plane(lens, bundle.reference_plane, skip)
    amplitude, _ = bundle.require_coherent()
    xp = xp_for(array_state(bundle.positions_m).namespace)

    # |a|^2 is formed on the wave side, so the only amplitude-derived quantity
    # that ever reaches the solver is unambiguously an intensity.
    intensity = xp.abs(amplitude) ** 2

    inbound, _ = bridge_arrays(
        {
            "positions_mm": metres_to_millimetres(bundle.positions_m),
            "directions": bundle.directions,
            "intensity": intensity,
        },
        OPTILAND_CAPABILITIES,
        reference="positions_mm",
        policy=plans.inbound.policy,
        target_device=plans.inbound.target_device,
    )
    positions_mm = inbound["positions_mm"]
    directions = inbound["directions"]
    intensity_t = inbound["intensity"]
    wavelength_um = metres_to_micrometres(bundle.wavelength_m)
    wavelengths = _solver_module(plans.inbound.target_namespace).full_like(
        intensity_t, wavelength_um
    )

    rays = real_rays_cls(
        positions_mm[:, 0],
        positions_mm[:, 1],
        positions_mm[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        intensity_t,
        wavelengths,
    )
    traced = lens.surfaces.trace(rays, skip=skip)

    # Plumbing only -- stack / isfinite / where / full_like, on which torch and
    # numpy agree exactly. Deliberately not `xp_for`, which refuses torch so that
    # no *physics* is written twice; nothing below is physics, and writing this
    # block twice for the two solver namespaces would be the real duplication.
    sp = _solver_module(plans.inbound.target_namespace)
    # `sp.asarray` first: optiland 0.6.0 wraps its numpy arrays in a subclass
    # whose `__array_wrap__` signature numpy 2.x deprecates, so comparing the raw
    # attributes emits a DeprecationWarning per chunk. Reading them as plain
    # arrays is both quieter and one less layer between the solver and the sum.
    positions_out_mm = sp.stack(
        [sp.asarray(traced.x), sp.asarray(traced.y), sp.asarray(traced.z)], axis=1
    )
    directions_out = sp.stack(
        [sp.asarray(traced.L), sp.asarray(traced.M), sp.asarray(traced.N)], axis=1
    )
    intensity_out = sp.asarray(traced.i)
    opd_out = sp.asarray(traced.opd)
    # Optiland clips by zeroing intensity, not by removing rows: a ray whose
    # intensity survived has meaningful geometry, one that was zeroed had its
    # state frozen at the clip and must not be summed. Non-finite geometry is
    # treated the same way -- a NaN position is not a position.
    valid_t = (
        (intensity_out > 0)
        & sp.all(sp.isfinite(positions_out_mm), axis=1)
        & sp.all(sp.isfinite(directions_out), axis=1)
        & sp.isfinite(opd_out)
    )
    # A clipped ray's geometry is never read as physics, but it still travels
    # inside an artifact whose contract forbids non-finite entries. Substituting
    # a harmless placeholder keeps the array shape -- and therefore the compiled
    # kernel -- constant across chunks; the amplitude is zeroed alongside it, so
    # the substitution cannot contribute to the reconstruction.
    valid_column = valid_t[:, None]
    zero3 = sp.zeros_like(positions_out_mm)
    safe_positions = sp.where(valid_column, positions_out_mm, zero3)
    zero1 = sp.zeros_like(opd_out)
    on_axis = sp.stack([zero1, zero1, sp.ones_like(opd_out)], axis=1)
    safe_directions = sp.where(valid_column, directions_out, on_axis)
    safe_opd = sp.where(valid_t, opd_out, zero1)
    valid_real = sp.where(valid_t, sp.ones_like(safe_opd), sp.zeros_like(safe_opd))

    outbound, _ = bridge_arrays(
        {
            "positions_mm": safe_positions,
            "directions": safe_directions,
            "optical_path_length_mm": safe_opd,
            "valid": valid_real,
        },
        plans.home,
        reference="positions_mm",
        policy=plans.outbound.policy,
        target_device=plans.outbound.target_device,
    )
    valid = outbound["valid"] > 0.5
    zeroed_amplitude = xp.where(valid, amplitude, xp.zeros_like(amplitude))

    out_batch = batch.with_traced_state(
        positions_m=millimetres_to_metres(outbound["positions_mm"]),
        directions=outbound["directions"],
        optical_path_length_m=millimetres_to_metres(outbound["optical_path_length_mm"]),
        amplitude=zeroed_amplitude,
        valid=valid,
        plane=image_plane,
        ray_id=batch.ray_id,
        provenance={
            "trace": {
                "skip": skip,
                "surface_positions_m": surfaces_m,
                "wavelength_um": wavelength_um,
                "bridge_plans": plans.as_dict(),
                "unit_conversions": {
                    "positions": "m -> mm inbound, mm -> m outbound",
                    "optical_path_length": "mm -> m outbound",
                    "wavelength": "m -> um inbound",
                    "directions": "dimensionless, unconverted",
                },
            }
        },
    )
    diagnostics = {
        "ray_count": batch.count,
        "invalid_rays": int(batch.count - int(valid.sum())),
        "optiland_backend": str(getattr(be, "get_backend", lambda: "unknown")()),
        "residency": out_batch.residency(),
        "amplitude_handling": AMPLITUDE_SIDECAR_RULE,
        "optiland_intensity_handling": OPTILAND_INTENSITY_RULE,
    }
    return out_batch, diagnostics
