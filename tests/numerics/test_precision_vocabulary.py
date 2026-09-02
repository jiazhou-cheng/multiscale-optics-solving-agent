"""The three concepts a single `config["dtype"]` string used to conflate.

CHE-173 (R02.1). Precision is a policy, dtype is an observation, device and
namespace are orthogonal. Each test below pins a distinction the reference
implementation got wrong at least once.
"""

from __future__ import annotations

import pytest

from numerics.precision import (
    ArrayNamespace,
    ArrayState,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
)


def test_complex64_is_fp32_because_accuracy_is_per_component() -> None:
    """The single most common way this gets stated wrongly.

    `complex64` occupies 64 bits and has float32 accuracy. Calling it FP64
    because of its width is a two-decimal-digit error in a tolerance argument.
    """
    assert DType.COMPLEX64.precision is Precision.FP32
    assert DType.COMPLEX64.component_bits == 32
    assert DType.COMPLEX128.precision is Precision.FP64


def test_fp16_has_no_complex_dtype_and_says_so() -> None:
    """None of the three namespaces has a first-class `complex32`.

    Returning `None` rather than inventing one is the difference between a table
    that is symmetrical and a table that is true.
    """
    assert Precision.FP16.complex_dtype is None
    assert Precision.FP32.complex_dtype is DType.COMPLEX64
    assert Precision.FP64.complex_dtype is DType.COMPLEX128


def test_a_precision_can_be_spelled_as_a_dtype_because_every_call_site_did() -> None:
    assert Precision.parse("float64") is Precision.FP64
    assert Precision.parse("complex64") is Precision.FP32
    assert Precision.parse("fp32") is Precision.FP32
    assert Precision.parse(Precision.FP16) is Precision.FP16


def test_an_unknown_precision_is_refused_naming_what_exists() -> None:
    with pytest.raises(ValueError) as caught:
        Precision.parse("bfloat16")
    assert caught.value.code == "UNKNOWN_PRECISION"
    assert "float64" in str(caught.value)


def test_integer_and_boolean_arrays_are_not_field_data() -> None:
    with pytest.raises(ValueError) as caught:
        DType.parse("int32")
    assert caught.value.code == "UNSUPPORTED_DTYPE"


def test_a_torch_dtype_spelling_parses() -> None:
    assert DType.parse("torch.float32") is DType.FLOAT32


def test_a_device_carries_the_ordinal_a_coarse_cpu_gpu_flag_drops() -> None:
    assert str(DevicePlacement(DeviceKind.CUDA, 6)) == "cuda:6"
    assert str(DevicePlacement(DeviceKind.CUDA)) == "cuda"
    assert str(DevicePlacement(DeviceKind.CPU)) == "cpu"
    assert DevicePlacement.parse("gpu:6") == DevicePlacement(DeviceKind.CUDA, 6)
    assert DevicePlacement.parse(None).is_host


def test_a_host_ordinal_is_a_contradiction_not_an_address() -> None:
    with pytest.raises(ValueError) as caught:
        DevicePlacement(DeviceKind.CPU, 0)
    assert caught.value.code == "UNSUPPORTED_DEVICE_SPELLING"
    with pytest.raises(ValueError):
        DevicePlacement.parse("cpu:0")


def test_a_tpu_is_outside_the_validated_set() -> None:
    with pytest.raises(ValueError) as caught:
        DevicePlacement.parse("tpu:0")
    assert caught.value.code == "UNSUPPORTED_DEVICE_SPELLING"


def test_namespace_and_device_are_orthogonal_except_where_physics_forbids_it() -> None:
    assert not ArrayNamespace.NUMPY.can_leave_host
    assert ArrayNamespace.JAX.can_leave_host
    assert ArrayNamespace.TORCH.can_leave_host
    assert not ArrayNamespace.NUMPY.is_differentiable


def test_a_numpy_buffer_cannot_be_observed_on_a_device() -> None:
    """The one combination of the three that describes no array that exists."""
    with pytest.raises(ValueError) as caught:
        ArrayState(DType.FLOAT32, DevicePlacement(DeviceKind.CUDA, 0), ArrayNamespace.NUMPY)
    assert caught.value.code == "NUMPY_CANNOT_LEAVE_HOST"


def test_an_observed_state_prints_and_serializes_all_three_facts() -> None:
    state = ArrayState(DType.COMPLEX64, DevicePlacement(DeviceKind.CUDA, 6), ArrayNamespace.JAX)
    assert str(state) == "jax:complex64@cuda:6"
    assert state.as_dict() == {
        "dtype": "complex64",
        "device": "cuda:6",
        "namespace": "jax",
    }
