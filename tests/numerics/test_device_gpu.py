"""What only an attached CUDA device can answer.

CHE-173 (R02.1), acceptance criterion 4. Run one device at a time, preferring 6
and 7 (AGENTS.md, "Shared GPU Server Policy"):

    MOA_GPUS=device=6 make test-gpu

The `gpu` marker and the gating in `tests/conftest.py` mean these skip -- rather
than fail -- on the CPU image and in any session that also selected non-GPU
tests.

Scope is deliberately narrow. These check that the *observation* machinery tells
the truth on a real device, which is the one claim in this package that a host
run cannot support: `array_state` reports where a buffer physically is, and the
torch->JAX bridge that Optiland's CUDA output has to cross stays on the device
instead of going through the host. Backend capability itself is not re-measured
here -- that is what the probes at `pre-rewrite-2026-08-30` are, and re-deriving
their conclusions from a live query is the failure this ticket names.
"""

from __future__ import annotations

import pytest

from numerics.arrays import _jax_device, array_state, device_of, to_namespace, to_state
from numerics.knowledge import load_capabilities
from numerics.precision import (
    ArrayNamespace,
    ArrayState,
    DeviceKind,
    DevicePlacement,
    DType,
    negotiate,
)

#: The measured Optiland record, loaded rather than imported as a constant --
#: CHE-223 (R03.6). This is one of the few tests that is genuinely *about* the
#: measured facts (does a CUDA trace really come back in the declared precision),
#: so it reads the record rather than a synthetic row.
OPTILAND = load_capabilities("M_RAY_OPTILAND")

pytestmark = pytest.mark.gpu


def test_a_jax_buffer_on_the_device_reports_cuda_with_its_ordinal() -> None:
    import jax
    import jax.numpy as jnp

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, "the gpu gate let a session through with no jax device"
    array = jax.device_put(jnp.zeros(4, dtype=jnp.complex64), gpus[0])

    state = array_state(array)
    assert state.namespace is ArrayNamespace.JAX
    assert state.device.kind is DeviceKind.CUDA
    assert state.device.index == int(gpus[0].id)
    assert state.dtype is DType.COMPLEX64


def test_a_torch_tensor_on_the_device_reports_cuda_with_its_ordinal() -> None:
    import torch

    tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
    state = array_state(tensor)
    assert state.namespace is ArrayNamespace.TORCH
    assert state.device == DevicePlacement(DeviceKind.CUDA, int(tensor.device.index or 0))
    assert state.dtype is DType.FLOAT32


def test_the_torch_to_jax_bridge_stays_on_the_device() -> None:
    """Optiland reaches CUDA only through torch, so this is the crossing that matters.

    A host round trip here would be invisible in the result -- same values, same
    dtype -- and would cost a synchronization and a copy on every handoff. The
    device of the *output* is the only thing that reports it.
    """
    import torch

    tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
    source = array_state(tensor)

    out = to_namespace(tensor, namespace=ArrayNamespace.JAX, device=source.device)
    observed = array_state(out)
    assert observed.namespace is ArrayNamespace.JAX
    assert observed.device.kind is DeviceKind.CUDA
    assert observed.device == source.device


def test_negotiating_optiland_onto_cuda_selects_torch_and_the_conversion_agrees() -> None:
    """The measured declaration and the executed conversion have to say the same thing."""
    import jax
    import jax.numpy as jnp

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    array = jax.device_put(jnp.zeros(4, dtype=jnp.float32), gpus[0])
    device = DevicePlacement(DeviceKind.CUDA, int(gpus[0].id))

    target = negotiate(array_state(array), OPTILAND, target_device=device)
    assert target == ArrayState(DType.FLOAT32, device, ArrayNamespace.TORCH)
    assert array_state(to_state(array, target)) == target


def test_an_absent_ordinal_is_refused_by_ordinal_rather_than_by_kind() -> None:
    """The distinction the CPU reachability table cannot reach.

    With one device attached, ordinal 7 does not exist while the *platform*
    plainly does -- so the refusal has to name the ordinal, not report that there
    is no GPU. Reporting the wrong one of these sends the reader to `nvidia-smi`
    instead of to `MOA_GPUS`.
    """
    import jax

    with pytest.raises(ValueError) as caught:
        _jax_device(jax, DevicePlacement(DeviceKind.CUDA, 7))
    assert caught.value.code == "DEVICE_ORDINAL_NOT_AVAILABLE"


def test_a_host_buffer_is_still_reported_as_host_on_the_gpu_image() -> None:
    """A requested device is never reported as an actual one.

    On the GPU image JAX puts every computation on the device by default, which
    is exactly the state in which "it ran on the GPU" stops being checkable by
    anything except reading the buffer.
    """
    import numpy as np

    assert device_of(np.zeros(4, dtype=np.float32)).is_host
