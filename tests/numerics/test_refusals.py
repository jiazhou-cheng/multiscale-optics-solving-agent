"""Every declared refusal code is reachable.

CHE-173 (R02.1). `docs/architecture_principles.md` records that the reference
implementation enforced "failed paths return structured diagnostics" with a
reachability enumeration, and that the new tree owes the equivalent. This is
`numerics/`'s share of it: a code in `REFUSAL_CODES` that nothing can raise is a
claim about a failure path that does not exist, and it is exactly as misleading
as a capability row nobody measured.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from numerics.arrays import to_namespace, verify_dtype, xp_for
from numerics.knowledge import load_capabilities
from numerics.precision import (
    REFUSAL_CODES,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    negotiate,
)

CUDA0 = DevicePlacement(DeviceKind.CUDA, 0)
HOST = DevicePlacement(DeviceKind.CPU)


def _invalid_capability() -> None:
    ComponentCapabilities(
        component="T_INVALID",
        devices=frozenset({DeviceKind.CPU}),
        precisions=frozenset({Precision.FP32}),
        accepted_input_dtypes=frozenset({DType.FLOAT32}),
        native_compute_dtypes=frozenset({DType.FLOAT32}),
        output_dtypes=frozenset({DType.FLOAT32}),
        device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
        probe="not/a/probe/path.py",
        probe_tag="a-synthetic-tag",
        evidence="a claim with no measurement behind it",
    )


#: A real-dtype declaration reachable on both devices, so the dtype-kind and
#: device-transfer refusals below have something to fail against.
#:
#: Synthetic, since CHE-223 (R03.6): a refusal code's *reachability* is a property
#: of `negotiate`, not of Optiland, and this table used to be built from the two
#: measured rows. `UNKNOWN_COMPONENT` is the exception and now goes through the
#: loader, which is where that refusal lives.
REAL_DUAL_DEVICE = ComponentCapabilities(
    component="T_REAL_DUAL_DEVICE",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32, Precision.FP64}),
    accepted_input_dtypes=frozenset({DType.FLOAT32, DType.FLOAT64}),
    native_compute_dtypes=frozenset({DType.FLOAT32, DType.FLOAT64}),
    output_dtypes=frozenset({DType.FLOAT32, DType.FLOAT64}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.TORCH}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.TORCH}),
    },
    probe="benchmarks/probes/precision/tolerance.py",
    probe_tag="a-synthetic-tag",
    evidence=(
        "test fixture, not a component: real dtypes on both devices, so a complex input "
        "and an unpermitted transfer are both reachable (synthetic, 0.0.0)"
    ),
)

#: A complex, single-namespace declaration with a lossy input, so the downcast
#: refusal is reachable. Synthetic, for the same reason.
COMPLEX_LOSSY = ComponentCapabilities(
    component="T_COMPLEX_LOSSY",
    devices=frozenset({DeviceKind.CPU}),
    precisions=frozenset({Precision.FP32}),
    accepted_input_dtypes=frozenset({DType.COMPLEX64}),
    native_compute_dtypes=frozenset({DType.COMPLEX64}),
    output_dtypes=frozenset({DType.COMPLEX64}),
    device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.JAX})},
    lossy_input_dtypes=frozenset({DType.COMPLEX128}),
    probe="benchmarks/probes/precision/tolerance.py",
    probe_tag="a-synthetic-tag",
    evidence=(
        "test fixture, not a component: one complex dtype with a declared lossy input "
        "(synthetic, 0.0.0)"
    ),
)

#: A host-only declaration, so "the target does not execute there" is reachable
#: without widening a measured row. Both real records declare CUDA.
HOST_ONLY = ComponentCapabilities(
    component="T_HOST_ONLY",
    devices=frozenset({DeviceKind.CPU}),
    precisions=frozenset({Precision.FP32}),
    accepted_input_dtypes=frozenset({DType.FLOAT32}),
    native_compute_dtypes=frozenset({DType.FLOAT32}),
    output_dtypes=frozenset({DType.FLOAT32}),
    device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
    probe="benchmarks/probes/precision/tolerance.py",
    probe_tag="a-synthetic-tag",
    evidence=(
        "test fixture, not a component: a host-only row so the device refusal is "
        "reachable (synthetic, 0.0.0)"
    ),
)


def _jax_has_no_gpu_here() -> None:
    import jax

    from numerics.arrays import _jax_device

    _jax_device(jax, CUDA0)


#: One trigger per code. Written as a table rather than as one test each so that
#: adding a code without a way to reach it fails the last test in this file.
TRIGGERS: dict[str, Callable[[], object]] = {
    "UNKNOWN_PRECISION": lambda: Precision.parse("bfloat16"),
    "UNSUPPORTED_DTYPE": lambda: DType.parse("int32"),
    "UNSUPPORTED_DEVICE_SPELLING": lambda: DevicePlacement.parse("tpu:0"),
    "UNKNOWN_COMPONENT": lambda: load_capabilities("C_RAY_TO_WAVE"),
    "INVALID_CAPABILITY_DECLARATION": _invalid_capability,
    "NO_COMPATIBLE_DTYPE_KIND": lambda: negotiate(
        ArrayState(DType.COMPLEX64, HOST, ArrayNamespace.JAX), REAL_DUAL_DEVICE
    ),
    "LOSSY_DOWNCAST_REQUIRED": lambda: negotiate(
        ArrayState(DType.COMPLEX128, HOST, ArrayNamespace.JAX), COMPLEX_LOSSY
    ),
    "UNSUPPORTED_DEVICE": lambda: negotiate(
        ArrayState(DType.FLOAT32, HOST, ArrayNamespace.NUMPY), HOST_ONLY, target_device=CUDA0
    ),
    "DEVICE_TRANSFER_NOT_PERMITTED": lambda: negotiate(
        ArrayState(DType.FLOAT32, HOST, ArrayNamespace.NUMPY),
        REAL_DUAL_DEVICE,
        target_device=CUDA0,
    ),
    "NAMESPACE_NOT_A_COMPUTE_NAMESPACE": lambda: xp_for(ArrayNamespace.TORCH),
    "NUMPY_CANNOT_LEAVE_HOST": lambda: to_namespace(
        np.zeros(3), namespace=ArrayNamespace.NUMPY, device=CUDA0
    ),
    "SILENT_DTYPE_DOWNCAST": lambda: verify_dtype(
        np.zeros(3, dtype=np.float32), DType.FLOAT64, context="numpy"
    ),
    "DEVICE_NOT_AVAILABLE": _jax_has_no_gpu_here,
}

#: Reachable only where the host state makes it reachable, so it is covered by
#: `test_device_gpu.py::test_an_absent_ordinal_is_refused_by_ordinal` instead of
#: here. Named rather than omitted, so the completeness check below stays honest.
GPU_ONLY_CODES = frozenset({"DEVICE_ORDINAL_NOT_AVAILABLE"})


@pytest.mark.parametrize("code", sorted(TRIGGERS))
def test_the_refusal_carries_its_code(code: str) -> None:
    with pytest.raises(ValueError) as caught:
        TRIGGERS[code]()
    assert getattr(caught.value, "code", None) == code, (
        f"the trigger for {code} raised {getattr(caught.value, 'code', None)} instead: "
        f"{caught.value}"
    )
    assert str(caught.value).startswith(f"[{code}] ")


def test_every_declared_code_has_a_trigger() -> None:
    unreachable = set(REFUSAL_CODES) - set(TRIGGERS) - GPU_ONLY_CODES
    assert not unreachable, (
        f"{sorted(unreachable)} are declared refusal codes with no way to reach them. "
        "Either delete the code or delete the branch that claims to raise it."
    )


def test_no_trigger_names_a_code_that_is_not_declared() -> None:
    undeclared = (set(TRIGGERS) | GPU_ONLY_CODES) - set(REFUSAL_CODES)
    assert not undeclared, f"{sorted(undeclared)} are triggered but not declared"


def test_an_undeclared_code_cannot_be_raised() -> None:
    """The detection half: `refusal` refuses to invent a code."""
    from numerics.precision import refusal

    with pytest.raises(ValueError, match="not a declared refusal code"):
        refusal(code="MADE_UP_CODE", component="numerics", message="nope")
