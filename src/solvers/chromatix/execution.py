"""The lazy Chromatix import, device selection, and the input precision bridge.

Grouped because all of it decides *where and in what precision* a propagation
will run, before any field exists. `_do_import_chromatix` is the only import
path and stays lazy: importing this module must not import chromatix or jax,
which `tests/test_solver_adapter_characterization.py` asserts in a subprocess.

The namesquat check lives here too. There is an unrelated package called
`chromatix` on PyPI, so "it imported" is not evidence that the right one
imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.capabilities import CHROMATIX_CAPABILITIES
from core.errors import (
    AdapterDependencyError,
)
from core.precision import (
    ArrayNamespace,
    BridgePlan,
    BridgePolicy,
    CapabilityError,
    DeviceKind,
    DevicePlacement,
    DType,
    ExecutionRequest,
    Precision,
    ResolvedExecution,
    plan_bridge,
)
from core.specs import Device
from solvers.chromatix.constants import (
    MODEL_ID,
)

if TYPE_CHECKING:
    pass



def _do_import_chromatix() -> tuple[Any, Any, Any, Any, Any]:
    """Unprotected import step; isolated so tests can force an ``ImportError``.

    Do not call this directly outside :func:`_import_chromatix` -- it does
    not translate the failure into ``AdapterDependencyError``.
    """
    import chromatix  # type: ignore[import-untyped]
    import chromatix.functional as cf  # type: ignore[import-untyped]
    import jax
    import jax.numpy as jnp
    from chromatix.functional.propagation import (  # type: ignore[import-untyped]
        compute_padding_transfer,
    )

    # jax_enable_x64 is process-global mutable state (like optiland's
    # set_backend -- never call this concurrently across threads). Anything else
    # in the same process may have already flipped it on, possibly as an import
    # side effect that -- because Python only executes module bodies once -- will
    # NOT be re-triggered by importing that module again later. This repository's
    # own float64 characterization tests do exactly that. Asserting our own
    # requirement explicitly,
    # every call, makes this adapter correct regardless of import/call order
    # instead of depending on ambient state: Chromatix's own
    # ScalarField.__init__ force-casts input to complex64 either way, but
    # downstream FFT-based propagation (asm_propagate) can still promote to
    # complex128 under x64, which would not match probe evidence captured
    # with x64 disabled.
    jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]

    return jax, jnp, chromatix, cf, compute_padding_transfer


def _import_chromatix() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import jax/chromatix, converting failure to ``AdapterDependencyError``.

    Never called at module import time (see ``adapters/__init__.py``
    convention); only called from inside ``run()``/``estimate()``.
    """
    try:
        return _do_import_chromatix()
    except ImportError as exc:
        raise AdapterDependencyError(
            "chromatix and/or jax could not be imported. This adapter requires "
            "the pinned commit in "
            "knowledge/solvers/chromatix/solver_card.yaml "
            "(git+https://github.com/chromatix-team/chromatix.git@"
            "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee, tag 0.6.0); the PyPI "
            "package literally named 'chromatix' is an unrelated namesquat. "
            f"Underlying error: {exc!r}"
        ) from exc


def _resolve_chromatix_execution(config: Mapping[str, Any]) -> ResolvedExecution:
    """Negotiate ``config['device'] / config['dtype']`` against Chromatix reality.

    The default precision is FP32, not FP64, because ``complex64`` is the only
    field storage Chromatix has. Defaulting to FP64 here would make every
    existing request look like a request for a precision the package cannot
    represent.
    """
    return CHROMATIX_CAPABILITIES.resolve(
        ExecutionRequest.from_config(
            MODEL_ID, config, default_precision=Precision.FP32
        )
    )


def _jax_gpu_unavailable_reason() -> str | None:
    """Why JAX cannot execute on a GPU here, or ``None`` if it can.

    Deliberately stricter than ``jax.devices()``. PB4a measured an image where
    ``jax.devices()`` returned ``[CudaDevice(id=0)]`` while the first jitted call
    died with "No PTX compilation provider is available" -- device enumeration
    needs only the driver, while running a kernel needs ptxas. So this compiles
    and runs one, which is the only question that matters.

    It also catches a process-global platform pin: with
    ``jax_platform_name='cpu'`` or ``JAX_PLATFORMS`` set anywhere in the process,
    JAX reports no GPU at all on a GPU host. CHE-72 removed the one dependency
    known to do that at import time, so this is now a defensive check rather than
    a characterized hazard -- but the state it detects is indistinguishable from
    a missing CUDA plugin, and both are worth naming.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - jax is pinned in both images
        return f"jax is not importable ({exc})"

    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpus:
        pinned = ""
        try:
            if jax.config.read("jax_platform_name") == "cpu":
                pinned = (
                    " -- jax_platform_name is pinned to 'cpu' process-globally, so "
                    "no GPU is reachable regardless of the hardware present"
                )
        except Exception:  # pragma: no cover - defensive on a config API change
            pass
        return f"jax reports no gpu device (backend={jax.default_backend()!r}){pinned}"

    try:
        probe = jnp.ones((4, 4), dtype=jnp.complex64)
        jax.block_until_ready(jnp.fft.fft2(jax.device_put(probe, gpus[0])))
    except Exception as exc:
        return (
            f"a CUDA device is visible ({gpus[0]}) but cannot execute a kernel: "
            f"{type(exc).__name__}: {exc}. Enumeration needs only the driver; "
            "compilation additionally needs ptxas (nvidia-cuda-nvcc-cu12) -- see "
            "docs/testing/gpu_environment.md"
        )
    return None


def _jax_device_for(jax: Any, device: DevicePlacement) -> Any:
    """The concrete JAX device for a resolved placement.

    Placement is explicit rather than left to ``jax.default_backend()``: with a
    GPU present, JAX puts everything on it by default, so a request for ``cpu``
    that did not say so would silently run on the GPU -- the same class of error
    as the reverse, and just as invisible.
    """
    platform = "gpu" if device.kind is DeviceKind.CUDA else "cpu"
    # jax.devices() lists only the DEFAULT platform's devices, so on a GPU host
    # it never mentions the CPU -- which made an explicit cpu request look
    # unavailable. jax.devices(platform) is the form that enumerates a specific
    # one; it raises when that platform has no backend, which is the real answer.
    try:
        candidates = list(jax.devices(platform))
    except RuntimeError:
        candidates = []
    if not candidates:
        raise CapabilityError(
            code="CHROMATIX_DEVICE_UNAVAILABLE",
            component=MODEL_ID,
            message=f"jax reports no {platform} device (backend={jax.default_backend()!r}).",
            requested=device,
            supported=[str(candidate) for candidate in jax.devices()],
        )
    if device.index is None:
        return candidates[0]
    for candidate in candidates:
        if int(candidate.id) == device.index:
            return candidate
    raise CapabilityError(
        code="CHROMATIX_DEVICE_ORDINAL_UNAVAILABLE",
        component=MODEL_ID,
        message=f"jax has no {platform} device with ordinal {device.index}.",
        requested=device,
        supported=[str(candidate) for candidate in candidates],
    )


def _plan_input_bridge(input_dtype: DType, config: Mapping[str, Any]) -> BridgePlan:
    """How the incoming field may enter Chromatix's complex64-only field path.

    The adapter's own input port defaults to ``ALLOW_DOWNCAST`` rather than the
    project-wide ``SAFE``, and that is a deliberate, narrow exception with
    history behind it: CHE-35 established that a complex128 input is truncated
    *inside* ``ScalarField.__init__`` whatever this adapter does, and chose to
    measure the loss rather than pretend it did not happen. Refusing the input
    instead would not prevent any truncation, it would only remove the
    measurement.

    What CHE-61 adds is that the truncation is now also a recorded
    :class:`BridgePlan` with ``lossy=True``, and that a caller who wants the
    refusal can have it with ``config['bridge_policy'] = 'safe'``.
    """
    policy_value = config.get("bridge_policy")
    policy = BridgePolicy.ALLOW_DOWNCAST if policy_value is None else BridgePolicy(
        str(policy_value)
    )
    return plan_bridge(
        # Namespace/device of the source are JAX-on-the-target by the time the
        # array is built; the dtype is the question this plan answers.
        array_state_for_dtype(input_dtype),
        CHROMATIX_CAPABILITIES,
        policy=policy,
        compute_dtype=DType.COMPLEX64,
    )


def array_state_for_dtype(dtype: DType) -> Any:
    """An ``ArrayState`` for a dtype whose buffer has not been created yet.

    Used only for planning the input cast, where the dtype is known from the
    record and the namespace/device are whatever the plan is about to select.
    """
    from core.precision import ArrayState

    return ArrayState(dtype, DevicePlacement(DeviceKind.CPU), ArrayNamespace.JAX)


def _map_device(backend: str) -> Device:
    if backend == "gpu":
        return Device.GPU
    if backend == "tpu":
        return Device.TPU
    return Device.CPU


def _pitch_to_pair(sample_pitch: Any) -> tuple[float, float]:
    """Normalize ``sample_pitch`` metadata to ``(pitch_y, pitch_x)`` in meters."""
    if isinstance(sample_pitch, int | float):
        return (float(sample_pitch), float(sample_pitch))
    pitch_y, pitch_x = sample_pitch
    return (float(pitch_y), float(pitch_x))


def _uri_to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return Path(uri)


# ---------------------------------------------------------------------------
# CHE-14: typed standalone wave-baseline contract
# ---------------------------------------------------------------------------
