"""Array-namespace introspection and the conversions a negotiation authorized.

CHE-173 (R02.1). This module answers three questions about a buffer without
guessing -- which ecosystem owns it, which device it is physically on including
the ordinal, and which dtype it actually has -- and performs the conversions
`numerics.precision.negotiate` has already decided are legal, and only those.

Why introspection rather than metadata
--------------------------------------
A requested device is not evidence of an actual one. A process-global JAX
platform setting, a missing PTX compiler or a silent host fallback can leave a
requested device of `cuda` and an actual device of `cpu` coexisting happily with
no warning raised, and any code that writes `record.device = requested_device`
then reports GPU execution that never happened. The reference implementation
measured exactly that. Every device fact in this project therefore comes from
`array_state`, which reads the buffer.

Why NumPy and JAX are the compute namespaces
--------------------------------------------
`jax.numpy` is a near drop-in for `numpy`, so one physics implementation
parameterized by `xp` runs unchanged on the host and on the GPU -- no CPU fork
and no GPU fork. Torch is *not* a compute namespace here: Optiland's torch
backend produces torch tensors, and those are bridged into JAX by `to_namespace`,
which on CUDA uses DLPack and therefore stays on the device with no host round
trip. That conversion is a boundary operation and is never done implicitly.

External frameworks are imported lazily. Importing this module must not pull in
JAX, Torch, Optiland or Chromatix, and `tests/numerics/test_no_backend_import.py`
checks that against `sys.modules` rather than by inspection.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np

from numerics.precision import (
    ArrayNamespace,
    ArrayState,
    DeviceKind,
    DevicePlacement,
    DType,
    refusal,
)

__all__ = [
    "COMPUTE_NAMESPACES",
    "array_state",
    "device_of",
    "dtype_of",
    "matmul_precision_kwargs",
    "namespace_of",
    "numpy_dtype",
    "to_host_numpy",
    "to_namespace",
    "to_state",
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
    """Which array ecosystem owns `value`.

    Checks the module name and `sys.modules` before `isinstance`, so asking
    about a NumPy array in a process that never imported JAX does not import it.
    """
    if isinstance(value, np.ndarray) or np.isscalar(value):
        return ArrayNamespace.NUMPY
    module = type(value).__module__.split(".")[0]
    if module in ("jax", "jaxlib"):
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
    # Lists, tuples and Python scalars own no buffer; NumPy is what
    # `np.asarray` would make of them, and saying so keeps the default path
    # (host, float64) exactly what it has always been.
    return ArrayNamespace.NUMPY


def device_of(value: Any) -> DevicePlacement:
    """Where `value` physically is. Observed, with the ordinal preserved."""
    namespace = namespace_of(value)
    if namespace is ArrayNamespace.NUMPY:
        return DevicePlacement(DeviceKind.CPU)
    if namespace is ArrayNamespace.JAX:
        devices = list(value.devices())
        if not devices:  # pragma: no cover - jax always reports at least one
            return DevicePlacement(DeviceKind.CPU)
        device = devices[0]
        if device.platform != "gpu":
            return DevicePlacement(DeviceKind.CPU)
        return DevicePlacement(DeviceKind.CUDA, int(device.id))
    torch_device = value.device
    if torch_device.type != "cuda":
        return DevicePlacement(DeviceKind.CPU)
    # A tensor's index is None only for the ambiguous `torch.device('cuda')`
    # spelling; a materialized CUDA tensor is always on a specific ordinal, and
    # torch resolves that spelling to the current device, which is 0.
    return DevicePlacement(DeviceKind.CUDA, int(torch_device.index or 0))


def dtype_of(value: Any) -> DType:
    """The dtype `value` actually has, in the project vocabulary."""
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

    Torch is refused rather than returned: its API differs enough
    (`torch.linalg`, no `einsum(optimize=...)`, different `fft` shape rules)
    that pretending otherwise would produce a second physics implementation by
    accident, which is exactly what one shared `xp` implementation exists to
    prevent.
    """
    if namespace is ArrayNamespace.NUMPY:
        return np
    if namespace is ArrayNamespace.JAX:
        import jax.numpy as jnp

        return jnp
    raise refusal(
        code="NAMESPACE_NOT_A_COMPUTE_NAMESPACE",
        component="numerics.arrays",
        message=(
            f"{namespace} is not a compute namespace. Torch buffers are bridged into "
            "JAX (zero-copy on CUDA via DLPack) rather than given a parallel physics "
            "implementation."
        ),
        requested=namespace,
        supported=COMPUTE_NAMESPACES,
    )


def numpy_dtype(dtype: DType) -> np.dtype[Any]:
    """The NumPy spelling of a project dtype. Also what JAX's `astype` accepts."""
    return np.dtype(dtype.value)


def matmul_precision_kwargs(namespace: ArrayNamespace) -> dict[str, Any]:
    """Keyword arguments that make a dot product compute at its declared dtype.

    On an Ampere GPU, XLA's default precision for an f32/c64 dot is **TF32**,
    which carries a 10-bit mantissa rather than 24. Measured on this host
    (RTX A6000, `benchmarks/probes/precision/gpu_matmul.py` at
    `pre-rewrite-2026-08-30`), an `einsum("n,ny,nx->yx", ...)` over 256
    complex64 wavelets returned:

        NumPy complex64 (host)              2.6e-07  vs a complex128 reference
        JAX complex64, GPU default          3.5e-04
        JAX complex64, precision="highest"  2.3e-07

    So "complex64 on the GPU" was 1500x less accurate than complex64, silently,
    with the array still reporting `dtype=complex64`. Requesting `highest` makes
    the declared dtype the computed dtype. NumPy has no such knob and needs
    none, so it gets an empty mapping -- which keeps one physics implementation,
    with only the "and mean it" flag differing by namespace.
    """
    if namespace is ArrayNamespace.JAX:
        return {"precision": "highest"}
    return {}


def verify_dtype(array: Any, requested: DType, *, context: str) -> Any:
    """Confirm a cast actually happened, and fail structurally when it did not.

    This exists because of one specific silent failure: JAX with
    `jax_enable_x64` disabled accepts `astype(float64)` and returns `float32`.
    No warning, no error -- the requested precision simply does not happen, and
    the flag is process-global, so any import can leave it that way. Checking
    the dtype that came back rather than the one that was asked for turns a
    ten-decimal-digit loss into a named refusal.
    """
    observed = dtype_of(array)
    if observed is requested:
        return array
    detail = ""
    if namespace_of(array) is ArrayNamespace.JAX and requested.component_bits == 64:
        import jax

        # `jax.config.read` is unannotated upstream, hence the ignore.
        if not bool(jax.config.read("jax_enable_x64")):  # type: ignore[no-untyped-call]
            detail = (
                " jax_enable_x64 is disabled in this process, so JAX silently returns "
                "float32/complex64 for any 64-bit request."
            )
    raise refusal(
        code="SILENT_DTYPE_DOWNCAST",
        component=context,
        message=(
            f"a cast to {requested} produced {observed} instead, so the requested "
            f"precision is not actually available here.{detail}"
        ),
        requested=requested,
        supported=[observed],
        remedy=(
            "Request the precision this namespace can represent, or enable 64-bit mode "
            "before any array is created -- never after, since the backend caches its "
            "configuration."
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


def to_state(value: Any, target: ArrayState) -> Any:
    """Convert `value` into exactly the state a negotiation returned.

    The pairing with `numerics.precision.negotiate`: that function decides, this
    one executes, and neither does the other's job. A target equal to the
    observed state is an identity and returns the buffer untouched, so an
    admissible artifact is never copied for form's sake.
    """
    if array_state(value) == target:
        return value
    return to_namespace(
        value, namespace=target.namespace, device=target.device, dtype=target.dtype
    )


def to_namespace(
    value: Any,
    *,
    namespace: ArrayNamespace,
    device: DevicePlacement | None = None,
    dtype: DType | None = None,
) -> Any:
    """Perform exactly the moves asked for, and no others.

    Called directly it will do what it is told; the policy lives in `negotiate`,
    which is why `to_state` is the normal entry point.
    """
    source = array_state(value)

    if namespace is ArrayNamespace.NUMPY:
        # A NumPy target is a host target by definition, so an explicit CUDA
        # request here is a contradiction rather than a copy instruction.
        if device is not None and device.kind is DeviceKind.CUDA:
            raise refusal(
                code="NUMPY_CANNOT_LEAVE_HOST",
                component="numerics.arrays",
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


def _to_numpy(value: Any, source: ArrayState) -> np.ndarray[Any, Any]:
    if source.namespace is ArrayNamespace.NUMPY:
        return np.asarray(value)
    if source.namespace is ArrayNamespace.JAX:
        import jax

        return np.asarray(jax.device_get(value))
    # torch: `detach()` is the graph break. Without it this call raises rather
    # than silently dropping gradients, which is worse for the same outcome.
    return np.asarray(value.detach().cpu().numpy())


def _to_jax(value: Any, source: ArrayState, device: DevicePlacement, dtype: DType | None) -> Any:
    import jax
    import jax.numpy as jnp

    if source.namespace is ArrayNamespace.TORCH:
        # DLPack keeps a CUDA tensor on its device; `jnp.asarray` would route it
        # through the host.
        tensor = value.detach() if getattr(value, "requires_grad", False) else value
        try:
            out = jnp.from_dlpack(tensor)
        except (RuntimeError, TypeError, ValueError):
            # An unsupported layout or dtype: fall back through the host. Still
            # explicit -- the namespace change was already authorized, and the
            # device move is checked below.
            out = jnp.asarray(tensor.detach().cpu().numpy())
    else:
        out = jnp.asarray(value)

    if dtype is not None and dtype_of(out) is not dtype:
        out = verify_dtype(out.astype(numpy_dtype(dtype)), dtype, context="jax")
    observed = device_of(out)
    if observed != device and not (device.index is None and observed.kind is device.kind):
        out = jax.device_put(out, _jax_device(jax, device))
    return out


def _jax_device(jax: Any, device: DevicePlacement) -> Any:
    platform = "gpu" if device.kind is DeviceKind.CUDA else "cpu"
    candidates = [d for d in jax.devices() if d.platform == platform]
    if not candidates:
        raise refusal(
            code="DEVICE_NOT_AVAILABLE",
            component="jax",
            message=f"jax reports no {platform} device (backend={jax.default_backend()!r}).",
            requested=device,
            evidence=(
                "on a GPU host this means the JAX CUDA plugin is absent or a "
                "process-global platform pin (jax_platform_name / JAX_PLATFORMS) is in "
                "force"
            ),
        )
    if device.index is None:
        return candidates[0]
    for candidate in candidates:
        if int(candidate.id) == device.index:
            return candidate
    raise refusal(
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
            out = torch.from_dlpack(value)  # type: ignore[attr-defined]
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


def to_host_numpy(value: Any, *, reason: str) -> np.ndarray[Any, Any]:
    """The explicit serialization boundary: a host copy taken on purpose.

    Distinct from a computational fallback, and the distinction is the point.
    Persisting a record genuinely needs host bytes; that must not be allowed to
    drag the live execution graph onto the CPU beforehand, so every such copy is
    taken here, at the moment of writing, with a stated reason.
    """
    del reason  # recorded by the caller in provenance; named here for the API
    return _to_numpy(value, array_state(value))
