"""Introspection of a real buffer, and only the conversions that were authorized.

CHE-173 (R02.1). Every device fact in this project comes from reading the array,
because a requested device is not evidence of an actual one. These run on the
host; `test_device_gpu.py` covers what only a device can answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics.arrays import (
    COMPUTE_NAMESPACES,
    array_state,
    device_of,
    dtype_of,
    matmul_precision_kwargs,
    namespace_of,
    numpy_dtype,
    to_host_numpy,
    to_namespace,
    to_state,
    verify_dtype,
    xp_for,
)
from numerics.precision import (
    ArrayNamespace,
    ArrayState,
    DeviceKind,
    DevicePlacement,
    DType,
    negotiate,
)

HOST = DevicePlacement(DeviceKind.CPU)


def test_a_numpy_buffer_reports_all_three_facts() -> None:
    state = array_state(np.zeros((2, 3), dtype=np.complex128))
    assert state == ArrayState(DType.COMPLEX128, HOST, ArrayNamespace.NUMPY)


def test_a_python_list_has_no_state_of_its_own_and_takes_the_host_default() -> None:
    """Lists and scalars own no buffer; NumPy is what `np.asarray` makes of them."""
    assert namespace_of([1.0, 2.0]) is ArrayNamespace.NUMPY
    assert device_of([1.0, 2.0]).is_host
    assert dtype_of([1.0, 2.0]) is DType.FLOAT64


def test_an_integer_array_is_refused_rather_than_silently_promoted() -> None:
    with pytest.raises(ValueError) as caught:
        dtype_of(np.zeros(3, dtype=np.int64))
    assert caught.value.code == "UNSUPPORTED_DTYPE"


def test_torch_is_not_a_compute_namespace() -> None:
    """Refused rather than returned: a second `xp` would be a second physics."""
    assert {ArrayNamespace.NUMPY, ArrayNamespace.JAX} == COMPUTE_NAMESPACES
    with pytest.raises(ValueError) as caught:
        xp_for(ArrayNamespace.TORCH)
    assert caught.value.code == "NAMESPACE_NOT_A_COMPUTE_NAMESPACE"
    assert xp_for(ArrayNamespace.NUMPY) is np


def test_the_matmul_precision_flag_is_a_namespace_difference_not_a_physics_one() -> None:
    """XLA's default for an f32/c64 dot on Ampere is TF32 -- 10 mantissa bits."""
    assert matmul_precision_kwargs(ArrayNamespace.JAX) == {"precision": "highest"}
    assert matmul_precision_kwargs(ArrayNamespace.NUMPY) == {}


def test_numpy_dtype_translates_the_vocabulary() -> None:
    assert numpy_dtype(DType.COMPLEX64) == np.dtype("complex64")


def test_verify_dtype_passes_a_cast_that_actually_happened() -> None:
    array = np.zeros(3, dtype=np.float32).astype(np.float64)
    assert verify_dtype(array, DType.FLOAT64, context="numpy") is array


def test_verify_dtype_refuses_a_cast_that_did_not() -> None:
    """The JAX `jax_enable_x64` failure, provoked without importing JAX.

    The mechanism under test is "compare what came back against what was asked
    for", and a float32 array standing in for a silently-truncated one exercises
    exactly that comparison.
    """
    with pytest.raises(ValueError) as caught:
        verify_dtype(np.zeros(3, dtype=np.float32), DType.FLOAT64, context="numpy")
    assert caught.value.code == "SILENT_DTYPE_DOWNCAST"


def test_an_admissible_artifact_is_not_copied_for_form() -> None:
    array = np.zeros(4, dtype=np.float64)
    assert to_state(array, array_state(array)) is array


def test_to_state_executes_exactly_what_negotiate_decided() -> None:
    from numerics.knowledge import load_capabilities

    array = np.zeros(4, dtype=np.float64)
    # The one array test that is about a *measured* target rather than about the
    # conversion machinery, so it loads the record rather than using a synthetic one.
    target = negotiate(array_state(array), load_capabilities("M_RAY_OPTILAND"))
    out = to_state(array, target)
    assert array_state(out) == target


def test_a_numpy_target_on_a_device_is_a_contradiction_not_a_copy() -> None:
    with pytest.raises(ValueError) as caught:
        to_namespace(
            np.zeros(3),
            namespace=ArrayNamespace.NUMPY,
            device=DevicePlacement(DeviceKind.CUDA, 0),
        )
    assert caught.value.code == "NUMPY_CANNOT_LEAVE_HOST"


def test_the_serialization_boundary_is_a_host_copy_taken_on_purpose() -> None:
    array = np.arange(4, dtype=np.float32)
    out = to_host_numpy(array, reason="persisting an execution record")
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, array)


def test_a_dtype_conversion_is_verified_not_assumed() -> None:
    out = to_namespace(
        np.zeros(3, dtype=np.float32), namespace=ArrayNamespace.NUMPY, dtype=DType.FLOAT64
    )
    assert dtype_of(out) is DType.FLOAT64
