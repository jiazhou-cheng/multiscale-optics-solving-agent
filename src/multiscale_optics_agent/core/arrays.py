"""Array-namespace introspection and explicit conversion (CHE-61).

This module answers three questions about a buffer without guessing:

* which ecosystem owns it (NumPy / JAX / Torch),
* which device it is physically on, including the ordinal,
* which dtype it actually has,

and performs the conversions a :class:`~multiscale_optics_agent.core.precision.BridgePlan`
authorizes -- and only those.

Why introspection rather than metadata
--------------------------------------
PB4a measured ``klujax.py:47`` running ``jax.config.update("jax_platform_name",
"cpu")`` at *import* time; klujax is a SAX dependency, so importing SAX
anywhere in a process silently moves every later JAX computation onto the host
with no warning and no ``JAX_PLATFORMS`` set. Under that failure a requested
device of ``cuda`` and an actual device of ``cpu`` coexist happily, and any code
that writes ``record.device = requested_device`` reports GPU execution that
never happened. Every device fact in this project therefore comes from
:func:`array_state`, which reads the buffer.

Why NumPy and JAX are the compute namespaces
---------------------------------------------
``jax.numpy`` is a near drop-in for ``numpy``, so a single physics
implementation parameterized by ``xp`` runs unchanged on the host and on the
GPU -- no CPU fork and no GPU fork. Torch is *not* a compute namespace here:
Optiland's torch backend produces torch tensors, and those are bridged into JAX
by :func:`to_namespace`, which on CUDA uses DLPack and therefore stays on the
device with no host round trip (measured: ``torch.float32`` on ``cuda:0`` ->
``jax`` on ``cuda:0``). That conversion is a real boundary operation and is
always recorded in the plan, never done implicitly.

External frameworks are imported lazily, per the adapter convention: importing
this module must not pull in JAX or Torch.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np

from multiscale_optics_agent.core.precision import (
    ArrayNamespace,
    ArrayState,
    BridgePlan,
    CapabilityError,
    DeviceKind,
    DevicePlacement,
    DType,
)

__all__ = [
    "COMPUTE_NAMESPACES",
    "apply_bridge",
    "array_state",
    "asarray",
    "device_of",
    "dtype_of",
    "matmul_precision_kwargs",
    "namespace_of",
    "numpy_dtype",
    "to_host_numpy",
    "to_namespace",
    "verify_dtype",
    "xp_for",
]

#: The namespaces one shared physics implementation can execute in. Torch is
#: deliberately absent -- see the module docstring.
COMPUTE_NAMESPACES = frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX})


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def namespace_of(value: Any) -> ArrayNamespace:
    """Which array ecosystem owns ``value``.

    Checks ``sys.modules`` before ``isinstance`` so that asking about a NumPy
    array in a process that never imported JAX does not import it.
    """
    if isinstance(value, np.ndarray) or np.isscalar(value):
        return ArrayNamespace.NUMPY
    module = type(value).__module__.split(".")[0]
    if module == "jaxlib" or module == "jax":
        return ArrayNamespace.JAX
    if module == "torch":
        return ArrayNamespace.TORCH
    if "jax" in sys.modules:
        import jax

        if isinstance(value, jax.Array):
            return ArrayNamespace.JAX
    if "torch" in sys.modules:
        import torch

        if isinstance(value, torch.Tensor):
            return ArrayNamespace.TORCH
    # Lists, tuples and Python scalars have no namespace of their own; NumPy is
    # what ``np.asarray`` would make of them, and saying so keeps the default
    # path (host float64) exactly what it has always been.
    return ArrayNamespace.NUMPY


def device_of(value: Any) -> DevicePlacement:
    """Where ``value`` physically is. Observed, with the ordinal preserved."""
    namespace = namespace_of(value)
    if namespace is ArrayNamespace.NUMPY:
        return DevicePlacement(DeviceKind.CPU)
    if namespace is ArrayNamespace.JAX:
        devices = list(value.devices())
        if not devices:  # pragma: no cover - jax always reports at least one
            return DevicePlacement(DeviceKind.CPU)
        device = devices[0]
        kind = DeviceKind.CUDA if device.platform == "gpu" else DeviceKind.CPU
        return DevicePlacement(kind, int(device.id) if kind is DeviceKind.CUDA else None)
    torch_device = value.device
    kind = DeviceKind.CUDA if torch_device.type == "cuda" else DeviceKind.CPU
    index = torch_device.index
    return DevicePlacement(kind, int(index) if kind is DeviceKind.CUDA and index is not None else
                           (0 if kind is DeviceKind.CUDA else None))


def dtype_of(value: Any) -> DType:
    """The dtype ``value`` actually has, in the project vocabulary."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return DType.parse(np.asarray(value).dtype)
    return DType.parse(dtype)


def array_state(value: Any) -> ArrayState:
    """The complete observed state of a buffer: dtype, device, namespace."""
    return ArrayState(dtype_of(value), device_of(value), namespace_of(value))


# ---------------------------------------------------------------------------
# Namespace modules and dtype translation
# ---------------------------------------------------------------------------


def xp_for(namespace: ArrayNamespace) -> ModuleType:
    """The numpy-compatible module for a compute namespace.

    Torch raises rather than returning ``torch``: its API differs enough
    (``torch.linalg``, no ``einsum(optimize=...)``, different ``fft`` shape
    rules) that pretending otherwise would produce a second physics
    implementation by accident, which is exactly what this contract forbids.
    """
    if namespace is ArrayNamespace.NUMPY:
        return np
    if namespace is ArrayNamespace.JAX:
        import jax.numpy as jnp

        return jnp
    raise CapabilityError(
        code="NAMESPACE_NOT_A_COMPUTE_NAMESPACE",
        component="couplers",
        message=(
            f"{namespace} is not a coupler compute namespace. Torch arrays are "
            "bridged into JAX (zero-copy on CUDA via DLPack) rather than given "
            "a parallel physics implementation."
        ),
        supported=COMPUTE_NAMESPACES,
    )


def numpy_dtype(dtype: DType) -> np.dtype:
    return np.dtype(str(dtype))


def matmul_precision_kwargs(namespace: ArrayNamespace) -> dict[str, Any]:
    """Keyword arguments that make a dot product compute at its declared dtype.

    On an Ampere GPU, XLA's default precision for an ``f32``/``c64`` dot is
    **TF32**, which carries a 10-bit mantissa rather than 24. Measured on this
    host (RTX A6000, ``benchmarks/probes/precision/gpu_matmul.py``), the coupler's
    ``einsum("n,ny,nx->yx", ...)`` over 256 complex64 wavelets returns:

        NumPy complex64 (host)      2.6e-07  relative to a complex128 reference
        JAX complex64, GPU default  3.5e-04
        JAX complex64, precision="highest"
                                    2.3e-07

    So "complex64 on the GPU" was 1500x less accurate than complex64, silently,
    with the array still reporting ``dtype=complex64``. That is the exact failure
    mode this contract exists to eliminate: a precision claim that only the
    dtype label supports. Requesting ``highest`` makes the declared dtype the
    computed dtype.

    NumPy has no such knob and needs none, so it gets an empty mapping. That
    keeps one physics implementation -- the operation is identical, only the
    "and mean it" flag differs by namespace.
    """
    if namespace is ArrayNamespace.JAX:
        return {"precision": "highest"}
    return {}


def verify_dtype(array: Any, requested: DType, *, context: str) -> Any:
    """Confirm a cast actually happened, and fail structurally when it did not.

    This exists because of one specific silent failure. JAX with
    ``jax_enable_x64`` disabled -- which is the state the Chromatix adapter
    *deliberately* enforces on every call, and the state klujax leaves behind --
    accepts ``astype(float64)`` and returns ``float32``. No warning, no error:
    the requested precision simply does not happen. Checking the dtype that came
    back rather than the one that was asked for turns a two-decimal-digit loss
    into a named capability failure.
    """
    observed = dtype_of(array)
    if observed is requested:
        return array
    detail = ""
    if namespace_of(array) is ArrayNamespace.JAX and requested.precision.bits == 64:
        try:
            import jax

            if not bool(jax.config.read("jax_enable_x64")):
                detail = (
                    " jax_enable_x64 is disabled in this process, so JAX silently "
                    "returns float32/complex64 for any 64-bit request. The "
                    "Chromatix adapter disables it on every call by design, and "
                    "importing SAX can leave it disabled too."
                )
        except ImportError:  # pragma: no cover - jax is pinned in both images
            pass
    raise CapabilityError(
        code="SILENT_DTYPE_DOWNCAST",
        component=context,
        message=(
            f"a cast to {requested} produced {observed} instead, so the requested "
            f"precision is not actually available here.{detail}"
        ),
        requested=requested,
        supported=[observed],
        remedy=(
            "Request the precision this namespace can represent, or enable the "
            "64-bit mode before any array is created -- never after, since the "
            "backend caches its configuration."
        ),
    )


def _torch_dtype(dtype: DType) -> Any:
    import torch

    return {
        DType.FLOAT16: torch.float16,
        DType.FLOAT32: torch.float32,
        DType.FLOAT64: torch.float64,
        DType.COMPLEX64: torch.complex64,
        DType.COMPLEX128: torch.complex128,
    }[dtype]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def asarray(
    value: Any, *, namespace: ArrayNamespace | None = None, dtype: DType | None = None
) -> Any:
    """Materialize ``value`` in a compute namespace without changing its device.

    Used where a contract type accepts a Python list or scalar: those have no
    device or dtype of their own, so the historical host/float64 default
    applies and nothing regresses. An array that *does* have a state is left
    where it is.
    """
    if namespace is None:
        namespace = namespace_of(value)
    if namespace is ArrayNamespace.TORCH:
        return value if dtype is None else value.to(_torch_dtype(dtype))
    xp = xp_for(namespace)
    if dtype is None:
        return xp.asarray(value)
    return verify_dtype(
        xp.asarray(value, dtype=numpy_dtype(dtype)), dtype, context=str(namespace)
    )


def to_namespace(
    value: Any,
    *,
    namespace: ArrayNamespace,
    device: DevicePlacement | None = None,
    dtype: DType | None = None,
) -> Any:
    """Convert a buffer, performing exactly the moves asked for.

    Every path here is a boundary operation that a :class:`BridgePlan` must
    already have authorized; :func:`apply_bridge` is the normal entry point.
    Called directly it will still do what it is told, which is why the plan --
    not this function -- is where the policy lives.
    """
    source = array_state(value)

    if namespace is ArrayNamespace.NUMPY:
        # A NumPy target is a host target by definition, so an explicit CUDA
        # request here is a contradiction rather than a copy instruction.
        if device is not None and device.kind is DeviceKind.CUDA:
            raise CapabilityError(
                code="NUMPY_CANNOT_LEAVE_HOST",
                component="arrays",
                message="a NumPy array cannot reside on a CUDA device.",
                requested=device,
            )
        out = _to_numpy(value, source)
        if dtype is not None and dtype_of(out) is not dtype:
            out = out.astype(numpy_dtype(dtype))
        return out

    target_device = device or source.device

    if namespace is ArrayNamespace.JAX:
        return _to_jax(value, source, target_device, dtype)

    return _to_torch(value, source, target_device, dtype)


def _to_numpy(value: Any, source: ArrayState) -> np.ndarray:
    if source.namespace is ArrayNamespace.NUMPY:
        return np.asarray(value)
    if source.namespace is ArrayNamespace.JAX:
        import jax

        return np.asarray(jax.device_get(value))
    # torch: detach() is the graph break the plan records; without it this call
    # raises rather than silently dropping gradients, which is worse UX for the
    # same outcome.
    return value.detach().cpu().numpy()


def _to_jax(value: Any, source: ArrayState, device: DevicePlacement, dtype: DType | None) -> Any:
    import jax
    import jax.numpy as jnp

    if source.namespace is ArrayNamespace.TORCH:
        # DLPack keeps a CUDA tensor on its device; jnp.asarray would route it
        # through the host. Measured on the GPU image: torch float32 cuda:0 ->
        # jax float32 cuda:0.
        tensor = value.detach() if getattr(value, "requires_grad", False) else value
        try:
            out = jnp.from_dlpack(tensor)
        except (RuntimeError, TypeError, ValueError):
            # float16/complex or an unsupported layout: fall back through the
            # host. Still explicit -- the plan already declared a namespace
            # conversion, and the device move is checked below.
            out = jnp.asarray(tensor.detach().cpu().numpy())
    else:
        out = jnp.asarray(value)

    if dtype is not None and dtype_of(out) is not dtype:
        out = verify_dtype(out.astype(numpy_dtype(dtype)), dtype, context="jax")
    if device_of(out) != device and not (
        device.index is None and device_of(out).kind is device.kind
    ):
        out = jax.device_put(out, _jax_device(jax, device))
    return out


def _jax_device(jax: Any, device: DevicePlacement) -> Any:
    platform = "gpu" if device.kind is DeviceKind.CUDA else "cpu"
    candidates = [d for d in jax.devices() if d.platform == platform]
    if not candidates:
        raise CapabilityError(
            code="DEVICE_NOT_AVAILABLE",
            component="jax",
            message=f"jax reports no {platform} device (backend={jax.default_backend()!r}).",
            requested=device,
            evidence=(
                "PB4a: importing SAX pins jax_platform_name='cpu' process-globally "
                "via klujax, which produces exactly this state on a GPU host"
            ),
        )
    if device.index is None:
        return candidates[0]
    for candidate in candidates:
        if int(candidate.id) == device.index:
            return candidate
    raise CapabilityError(
        code="DEVICE_ORDINAL_NOT_AVAILABLE",
        component="jax",
        message=f"jax has no {platform} device with ordinal {device.index}.",
        requested=device,
        supported=[str(d) for d in candidates],
    )


def _to_torch(value: Any, source: ArrayState, device: DevicePlacement, dtype: DType | None) -> Any:
    import torch

    if source.namespace is ArrayNamespace.TORCH:
        out = value
    elif source.namespace is ArrayNamespace.JAX:
        try:
            out = torch.from_dlpack(value)
        except (RuntimeError, TypeError, ValueError):
            import jax

            out = torch.as_tensor(np.asarray(jax.device_get(value)))
    else:
        out = torch.as_tensor(np.asarray(value))

    if dtype is not None:
        out = out.to(_torch_dtype(dtype))
    target = "cpu" if device.kind is DeviceKind.CPU else f"cuda:{device.index or 0}"
    if str(out.device) != target and out.device.type != target:
        out = out.to(target)
    return out


def apply_bridge(value: Any, plan: BridgePlan) -> Any:
    """Execute exactly the conversions a plan authorizes, and no others."""
    observed = array_state(value)
    if observed != plan.source:
        raise CapabilityError(
            code="BRIDGE_PLAN_SOURCE_MISMATCH",
            component="arrays",
            message=(
                f"the plan was built for a {plan.source} source but the array is "
                f"{observed}. A plan is only valid for the state it was planned "
                "against; re-plan rather than reuse."
            ),
            requested=str(observed),
        )
    if plan.is_identity:
        return value
    return to_namespace(
        value,
        namespace=plan.target_namespace,
        device=plan.target_device,
        dtype=plan.target_dtype,
    )


def to_host_numpy(value: Any, *, reason: str) -> np.ndarray:
    """The explicit serialization boundary: a host copy taken on purpose.

    Distinct from a computational fallback, and the distinction is the point.
    Persisting an ``ArtifactRecord`` genuinely needs host bytes; that must not
    be allowed to drag the live execution graph onto the CPU beforehand, so
    every such copy is taken here, at the moment of writing, with a stated
    reason recorded alongside it.
    """
    del reason  # recorded by the caller in provenance; named here for the API
    return _to_numpy(value, array_state(value))
