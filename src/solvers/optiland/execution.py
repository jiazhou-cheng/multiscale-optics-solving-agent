"""Backend, device and precision resolution, and the lazy solver import.

Optiland's backend state is process-global: ``set_backend``, ``set_precision``
and ``set_device`` all mutate module state and none is thread-safe. So *when*
and *in what order* these run is part of the behaviour, not an implementation
detail, and grouping them here makes that order readable in one place instead of
interleaved with trace logic.

``_import_optiland`` is the only import path and stays lazy: importing this
module must not import optiland, which
``tests/test_solver_adapter_characterization.py`` asserts in a subprocess.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.arrays import numpy_dtype
from core.capabilities import OPTILAND_CAPABILITIES
from core.errors import (
    AdapterDependencyError,
)
from core.precision import (
    ArrayNamespace,
    CapabilityError,
    DeviceKind,
    DType,
    ExecutionRequest,
    ResolvedExecution,
)
from solvers.optiland.constants import (
    _DIRECTION_NORM_TOLERANCE,
    MODEL_ID,
)

_BACKEND_NAMESPACE = {
    "numpy": ArrayNamespace.NUMPY,
    "torch": ArrayNamespace.TORCH,
}


def _direction_norm_tolerance(dtype: DType) -> float:
    """Direction unit-norm bound appropriate to the precision actually traced in.

    The float64 constant (1e-12) stays exactly as it was. A float32 trace cannot
    satisfy it and it would be wrong to want it to: Optiland normalizes in
    float32, so ``|d| - 1`` sits at a few float32 epsilons before this adapter
    ever sees the rays. ``64 * eps`` is that round-off with headroom, derived
    rather than chosen, and it reduces to the historical value for float64.
    """
    import numpy as np

    eps = float(np.finfo(numpy_dtype(dtype)).eps)
    return max(_DIRECTION_NORM_TOLERANCE, 64.0 * eps)


def _resolve_optiland_execution(config: Mapping[str, Any]) -> ResolvedExecution:
    """Negotiate ``config['device'] / config['dtype'] / config['backend']``.

    One place, one capability table. Before CHE-61 this was two ``!=``
    comparisons against string constants, which is why the adapter reported
    "cpu"/"float64" whatever Optiland was actually told to do -- and it was never
    told anything, because ``set_device``/``set_precision`` were never called.

    ``config['backend']`` selects the array namespace, and that pairing is
    enforced rather than assumed: ``set_device`` raises
    ``BackendCapabilityError`` on the numpy backend, so ``device='cuda'`` with
    ``backend='numpy'`` is refused here instead of failing inside Optiland.
    """
    backend_name = str(config.get("backend", "numpy"))
    namespace = _BACKEND_NAMESPACE.get(backend_name)
    execution = ExecutionRequest.from_config(MODEL_ID, config)
    return OPTILAND_CAPABILITIES.resolve(
        ExecutionRequest(
            component=MODEL_ID,
            precision=execution.precision,
            device=execution.device,
            namespace=namespace,
            bridge_policy=execution.bridge_policy,
        )
    )


def _apply_optiland_execution(
    be: Any, torch_module: Any, resolved: ResolvedExecution, backend_name: str
) -> dict[str, Any]:
    """Drive Optiland's real execution controls and report what it actually did.

    ``set_backend``, ``set_precision`` and ``set_device`` are all process-global
    and none is thread-safe (Optiland's own docs). They are therefore set
    explicitly on every run -- never inherited from whatever a previous run in
    this process left behind -- exactly as ``set_backend`` already was.

    The return value is *observed*: ``get_device()``/``get_precision()`` read
    back from Optiland rather than echoing the request, because a request and a
    result are different facts and PB4a showed what happens when a project
    conflates them.
    """
    be.set_backend(backend_name)

    applied: dict[str, Any] = {
        "set_backend": backend_name,
        "set_precision": str(resolved.precision.real_dtype),
        "set_device": None,
    }
    # Optiland spells precision as a dtype name; the project spells it as a
    # policy. This is the one place the two vocabularies meet.
    be.set_precision(str(resolved.precision.real_dtype))

    if resolved.namespace is ArrayNamespace.TORCH:
        device_string = str(resolved.device)
        if resolved.device.kind is DeviceKind.CUDA:
            _require_cuda(torch_module, resolved)
        be.set_device(device_string)
        applied["set_device"] = device_string
        applied["get_device"] = str(be.get_device())
    else:
        # set_device raises BackendCapabilityError on the numpy backend, so it is
        # not called at all rather than called-and-caught.
        applied["get_device"] = "cpu (numpy backend has no device concept)"
    # get_precision returns an int width (32 / 64) on the pinned install, NOT the
    # dtype name set_precision takes -- an asymmetry worth normalizing here so the
    # diagnostics read in one vocabulary. Both spellings are accepted because the
    # setter's and getter's disagreement is exactly the kind of thing a minor
    # release changes, and a diagnostics field is not worth crashing a trace over.
    observed = be.get_precision()
    applied["get_precision_raw"] = observed
    applied["get_precision"] = (
        f"float{int(observed)}" if isinstance(observed, int) else str(observed)
    )
    return applied


def _cuda_unavailable_reason() -> str | None:
    """Why torch cannot reach a CUDA device here, or ``None`` if it can.

    Called only when a request actually asks for CUDA, so the default CPU path
    never pays for a torch import. Importing torch *is* how this question gets
    answered, which is why the eager gate can be import-free for every other
    request but not for this one -- and it is still eager, since it happens
    before any Optiland call.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is pinned in both images
        return f"torch is not importable ({exc})"
    if torch.version.cuda is None or "+cpu" in torch.__version__:
        return (
            f"torch is a CPU-only build ({torch.__version__}); the default "
            "agent_solver image installs it from the CPU wheel index"
        )
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False (no CUDA device attached to this container)"
    return None


def _require_cuda(torch_module: Any, resolved: ResolvedExecution) -> None:
    """Refuse a CUDA request the installed torch cannot actually serve.

    The default ``agent_solver`` image installs torch from the CPU-only wheel
    index, where ``torch.cuda.is_available()`` is ``False`` and
    ``be.set_device('cuda')`` would either raise from deep inside torch or -- on
    a half-provisioned image -- succeed and then fail at the first kernel. Both
    are worse than a named capability error here.
    """
    if torch_module is None:  # pragma: no cover - guarded by _import_optiland
        raise CapabilityError(
            code="OPTILAND_CUDA_REQUIRES_TORCH",
            component=MODEL_ID,
            message="a CUDA request needs the torch backend, which was not imported.",
            requested=resolved.device,
        )
    if not torch_module.cuda.is_available():
        raise CapabilityError(
            code="OPTILAND_CUDA_UNAVAILABLE",
            component=MODEL_ID,
            message=(
                f"config['device']={str(resolved.device)!r} was requested but this "
                f"torch install cannot reach a CUDA device (torch "
                f"{torch_module.__version__}, torch.version.cuda="
                f"{torch_module.version.cuda!r}, torch.cuda.is_available()=False)."
            ),
            requested=resolved.device,
            supported=["cpu"],
            evidence="docker/Dockerfile installs torch from the CPU-only wheel index",
            remedy=(
                "Run in the CUDA image: `./run.sh --gpu ...` (see "
                "docs/testing/gpu_environment.md). There is deliberately no "
                "silent fallback to the CPU."
            ),
        )


def _host_array(be_utils: Any, value: Any, *, dtype: Any = None) -> Any:
    """Copy solver data to the host for persistence, preserving its precision.

    This replaces ``np.asarray(be_utils.to_numpy(x), dtype=np.float64)``, which
    did three separable things in one expression: a device-to-host transfer, a
    precision force, and (for a torch tensor) an autodiff graph break. Only the
    first is actually required to write a ``.npz``.

    Dropping the ``dtype=np.float64`` is a no-op for the default numpy/float64
    path -- the arrays already are float64, so L1-RAY-01's recorded fingerprint
    is unchanged -- and is what lets a float32 trace be persisted as float32
    rather than silently gaining ten digits it never computed.

    ``dtype`` is still available for the few quantities that are deliberately
    computed at reference precision regardless of the trace; each such call site
    says why.
    """
    import numpy as np

    host = np.asarray(be_utils.to_numpy(value))
    return host if dtype is None else host.astype(dtype)


def _import_optiland(*, need_torch: bool) -> tuple[Any, Any, Any]:
    """Lazily import optiland (and torch, only if requested).

    Never called at module import time -- only from run()/estimate(). Returns
    (optiland.backend, optiland.backend.utils, torch-module-or-None).

    ``optiland.samples`` is deliberately absent since CHE-56: system
    construction goes through the canonical prescription registry and the
    generic builder, which imports the construction API it needs itself. This
    function covers the backend state and array-conversion surface that the
    trace/export path uses.
    """
    try:
        import optiland.backend as be  # type: ignore[import-untyped]
        import optiland.backend.utils as be_utils  # type: ignore[import-untyped]
    except Exception as exc:
        raise AdapterDependencyError(
            f"optiland could not be imported: {type(exc).__name__}: {exc}. "
            "Install it via `pip install optiland==0.6.0` or this project's "
            "'torch' extra (`pip install .[torch]`, which also pins torch)."
        ) from exc

    torch_module: Any = None
    if need_torch:
        try:
            import torch

            torch_module = torch
        except Exception as exc:
            raise AdapterDependencyError(
                "config['backend']='torch' (or require_gradients=True) needs "
                "the optional torch package. torch is NOT a declared optiland "
                "dependency (knowledge/solvers/optiland/failure_guide.md) and "
                f"must be installed separately: {type(exc).__name__}: {exc}"
            ) from exc

    return be, be_utils, torch_module
