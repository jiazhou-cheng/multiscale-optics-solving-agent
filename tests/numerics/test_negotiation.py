"""Device/dtype negotiation: preserve, promote by the minimum, or refuse.

CHE-173 (R02.1). `negotiate` is pure -- it takes no arrays and performs no
conversion -- so the policy can be tested against the capability table alone,
which is the point: policy resolution must not be entangled with optical
formulas.

Two flags, no policy object. A precision loss is a physics decision
(`allow_downcast`); a host/device copy is a cost decision
(`allow_device_transfer`). Both default to refusing.
"""

from __future__ import annotations

import pytest

from numerics.precision import (
    CHROMATIX_CAPABILITIES,
    OPTILAND_CAPABILITIES,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    negotiate,
)

HOST = DevicePlacement(DeviceKind.CPU)
CUDA0 = DevicePlacement(DeviceKind.CUDA, 0)


def host(dtype: DType, namespace: ArrayNamespace = ArrayNamespace.NUMPY) -> ArrayState:
    return ArrayState(dtype, HOST, namespace)


#: A host-only declaration, so "the target does not execute there" is reachable
#: without widening a measured row: both real rows declare CUDA.
HOST_ONLY = ComponentCapabilities(
    component="T_HOST_ONLY",
    devices=frozenset({DeviceKind.CPU}),
    precisions=frozenset({Precision.FP64}),
    accepted_input_dtypes=frozenset({DType.COMPLEX128}),
    native_compute_dtypes=frozenset({DType.COMPLEX128}),
    output_dtypes=frozenset({DType.COMPLEX128}),
    device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
    minimum_compute_precision=Precision.FP64,
    probe="benchmarks/probes/precision/tolerance.py",
    evidence=(
        "test fixture, not a component: a host-only row so the absence of an implicit "
        "fallback is testable (pre-rewrite-2026-08-30, 0.0.0)"
    ),
)


def test_an_admissible_artifact_is_returned_unchanged() -> None:
    """No float32 -> float64 -> float32 round trip for convenience."""
    source = host(DType.FLOAT32)
    assert negotiate(source, OPTILAND_CAPABILITIES) == source


def test_a_below_minimum_dtype_is_widened_by_the_smallest_admissible_step() -> None:
    """float16 -> float32, not float16 -> float64.

    Optiland is the real case: `set_precision` is `Literal['float32','float64']`,
    so float16 is not admissible, and the smallest lossless step into the
    accepted set is the one taken. Widening to float64 instead would double the
    trace's memory for nothing.
    """
    assert negotiate(host(DType.FLOAT16), OPTILAND_CAPABILITIES).dtype is DType.FLOAT32


def test_a_promotion_is_lossless_and_is_still_not_native_support() -> None:
    """The accepted/native split, seen from the negotiator's side."""
    assert DType.FLOAT16 not in OPTILAND_CAPABILITIES.native_compute_dtypes
    assert OPTILAND_CAPABILITIES.compute_dtype_for(DType.FLOAT16) is DType.FLOAT32


def test_a_dtype_the_target_cannot_hold_is_refused_rather_than_narrowed() -> None:
    """Chromatix's complex128: physically ingestible, silently truncated."""
    with pytest.raises(ValueError) as caught:
        negotiate(host(DType.COMPLEX128, ArrayNamespace.JAX), CHROMATIX_CAPABILITIES)
    assert caught.value.code == "LOSSY_DOWNCAST_REQUIRED"
    assert "allow_downcast" in str(caught.value)


def test_the_downcast_happens_only_when_it_is_asked_for() -> None:
    target = negotiate(
        host(DType.COMPLEX128, ArrayNamespace.JAX),
        CHROMATIX_CAPABILITIES,
        allow_downcast=True,
    )
    assert target.dtype is DType.COMPLEX64


def test_a_kind_the_target_accepts_nothing_of_is_refused() -> None:
    """Optiland is a ray tracer: it has no complex path at all."""
    with pytest.raises(ValueError) as caught:
        negotiate(host(DType.COMPLEX64, ArrayNamespace.JAX), OPTILAND_CAPABILITIES)
    assert caught.value.code == "NO_COMPATIBLE_DTYPE_KIND"


def test_residency_is_preserved_when_the_target_can_execute_there() -> None:
    source = ArrayState(DType.COMPLEX64, CUDA0, ArrayNamespace.JAX)
    assert negotiate(source, CHROMATIX_CAPABILITIES) == source


def test_there_is_no_implicit_host_fallback() -> None:
    """The failure where a 'GPU run' quietly executes on the CPU and succeeds."""
    source = ArrayState(DType.COMPLEX128, CUDA0, ArrayNamespace.JAX)
    with pytest.raises(ValueError) as caught:
        negotiate(source, HOST_ONLY)
    assert caught.value.code == "UNSUPPORTED_DEVICE"
    assert negotiate(source, HOST_ONLY, allow_device_transfer=True).device == HOST


def test_a_requested_device_the_target_cannot_execute_on_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        negotiate(host(DType.COMPLEX128), HOST_ONLY, target_device=CUDA0)
    assert caught.value.code == "UNSUPPORTED_DEVICE"


def test_a_transfer_the_caller_did_not_authorize_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        negotiate(host(DType.FLOAT32), OPTILAND_CAPABILITIES, target_device=CUDA0)
    assert caught.value.code == "DEVICE_TRANSFER_NOT_PERMITTED"


def test_reaching_cuda_through_optiland_selects_the_torch_namespace() -> None:
    """The measured fact, arrived at by negotiation rather than by an if-branch."""
    target = negotiate(
        host(DType.FLOAT32),
        OPTILAND_CAPABILITIES,
        target_device=CUDA0,
        allow_device_transfer=True,
    )
    assert target == ArrayState(DType.FLOAT32, CUDA0, ArrayNamespace.TORCH)


def test_a_device_buffer_is_not_pushed_through_the_host_to_change_ecosystem() -> None:
    source = ArrayState(DType.COMPLEX64, CUDA0, ArrayNamespace.TORCH)
    assert negotiate(source, CHROMATIX_CAPABILITIES).namespace is ArrayNamespace.JAX


def test_a_cuda_row_driven_only_by_numpy_is_refused_at_declaration() -> None:
    """Why `negotiate` has no "no admissible namespace" branch.

    NumPy cannot hold device memory, so a row declaring it as CUDA's driver
    claims a path that cannot exist. Catching it in the declaration means the
    negotiator never has to handle the case, and there is no refusal code for a
    state nothing can reach.
    """
    with pytest.raises(ValueError, match="host-only namespaces"):
        ComponentCapabilities(
            component="T_IMPOSSIBLE",
            devices=frozenset({DeviceKind.CUDA}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT32}),
            output_dtypes=frozenset({DType.FLOAT32}),
            device_namespaces={DeviceKind.CUDA: frozenset({ArrayNamespace.NUMPY})},
            probe="benchmarks/probes/precision/tolerance.py",
            evidence=(
                "test fixture, not a component: a declaration that cannot be true "
                "(pre-rewrite-2026-08-30, 0.0.0)"
            ),
        )


def test_negotiate_never_returns_a_numpy_buffer_on_a_device() -> None:
    """`ArrayState` refuses that combination, so a negotiation that produced it
    would raise from the constructor rather than return a nonsense target."""
    source = ArrayState(DType.FLOAT32, CUDA0, ArrayNamespace.TORCH)
    target = negotiate(source, OPTILAND_CAPABILITIES)
    assert target.namespace.can_leave_host
