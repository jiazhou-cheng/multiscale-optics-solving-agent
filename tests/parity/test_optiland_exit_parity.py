"""Subject 3: an Optiland trace lands where `trace_exit_state` says it will.

CHE-245 (T1) acceptance criterion 2 -- "after a `device="cuda"`
`SO_RAY_LAUNCH_TRACE`, `array_state(bundle.positions_m)` reports CUDA and the
target namespace, asserted **through T0's fixture** rather than a bespoke test".

What this module borrows from CHE-244 is the *mechanism*, which is the part worth
having exactly once: `conftest.verify_placement` compares a request against an
observation read back off the buffer, and `conftest.unavailable_reason` keeps the
four outcomes distinguishable. What it cannot borrow is `cells.cells_for`, and
the reason is a real distinction rather than an inconvenience.
`knowledge/capabilities/M_RAY_OPTILAND.json` records `device_namespaces[cuda] =
["torch"]`, which is a measured fact about **which namespace drives the solver**
-- `set_device` raises `BackendCapabilityError` on the NumPy backend. Where the
traced rays are *delivered* is a different axis: torch is refused by
`representations.contracts.adopt_array`, so the exit can never be in the
namespace that executed. The pack declares the inbound axis and
`rays.trace_exit_state` declares the outbound one, so this module derives the
execution requests from the pack and the expected exit from the production
function, and still writes no matrix of its own.

Nothing here is an oracle. Agreement between a device exit and a host exit is not
evidence that either traced the lens correctly; `tests/physics/` holds the gates
that can settle that. This answers "did the buffer land where the code said", and
that question is the whole of what CHE-244 exists to make falsifiable.
"""

from __future__ import annotations

from typing import Any

import pytest

from backends.optiland import trace
from backends.optiland.rays import trace_exit_state
from numerics import ArrayNamespace, DeviceKind, DevicePlacement, Precision, load_capabilities
from parity.cells import Cell
from parity.conftest import unavailable_reason, verify_placement

#: The component whose measured capability table decides which executions exist.
#: `numerics.load_capabilities` is the reader; nothing here names a device or a
#: precision that the pack does not.
COMPONENT = "M_RAY_OPTILAND"

#: The one system every case traces. `M3-SINGLET-REF`, on the fixture tree's own
#: prescription -- a placement is a property of the buffer, so the lens is chosen
#: for being the cheapest declared system rather than for its physics.
SAMPLING = {"num_rings": 4, "reference_surface": "exit_pupil"}


def _execution_requests() -> list[Any]:
    """Every `(device, precision)` the capability pack admits, as pytest params.

    Derived, not listed: `devices` x `precisions` straight off the pack. The
    `gpu` mark follows from the *execution* device rather than from the exit cell,
    which is the one place the two axes visibly come apart -- a `cuda` FP64
    request executes on the device and exits on the host, so marking it from the
    exit would route it into `make test`, where no device exists to run it.
    """
    capabilities = load_capabilities(COMPONENT)
    params = []
    for device_kind in sorted(capabilities.devices, key=lambda d: d.value):
        for precision in sorted(capabilities.precisions, key=lambda p: p.value):
            device = DevicePlacement(device_kind)
            marks = (pytest.mark.gpu,) if device_kind is DeviceKind.CUDA else ()
            params.append(
                pytest.param(device, precision, id=f"{device}-{precision}", marks=marks)
            )
    return params


def _inbound_reason(device: DevicePlacement) -> str | None:
    """Why this *execution* cannot run here, in CHE-244's own vocabulary.

    The exit cell answers "can the buffer live there"; this answers "can the
    trace run there", and on CUDA they differ. The namespace asked about is the
    pack's -- `namespaces_for(cuda)` is `{torch}` -- so this is not a second
    opinion about what drives the solver.
    """
    if device.kind is not DeviceKind.CUDA:
        return None
    capabilities = load_capabilities(COMPONENT)
    namespaces = capabilities.namespaces_for(device.kind)
    assert len(namespaces) == 1, f"{COMPONENT} declares {namespaces} for cuda; expected exactly one"
    inbound = Cell(
        namespace=next(iter(namespaces)),
        device=device,
        dtype=Precision.FP32.real_dtype,
    )
    return unavailable_reason(inbound)


@pytest.mark.parametrize(("device", "precision"), _execution_requests())
def test_a_trace_lands_in_the_state_its_exit_declares(
    device: DevicePlacement, precision: Precision
) -> None:
    """All five bundle arrays, and the one dtype asymmetry the two exits have.

    All five and not just `positions_m`: `RayBundle.state` reads the geometry, so
    asserting on that alone would pass an artifact whose measure or amplitude was
    left behind on the host.

    **Which dtype each array is compared at is itself the assertion**, because the
    two exit arms differ and the difference is what AC-4 protects:

    `positions_m`, `directions`
        The exit dtype. These come off the trace at the precision it ran in.
    `amplitude`
        The *complex counterpart* of the exit dtype. `adopt_array(widen_real=
        True)`'s documented behaviour: `sqrt(intensity)` is a phase-free
        amplitude and is widened to the complex dtype of the **same** precision,
        so a float32 trace does not acquire ten digits at the boundary.
    `optical_path_m`, `measure_weight`
        The exit dtype on a **device** exit, and host float64 on a **host** exit.
        Not an inconsistency to be tidied away. Both are computed by
        `declare_optical_path_m` and the hexapolar measure, which are host float64
        by `launch.py::_launch_columns`' recorded decision -- the object-space term
        is a piston-and-tilt correction of order 1e4 waves that float32 cannot
        carry. A host exit therefore delivers them untouched, which is exactly why
        every frozen fingerprint is unchanged; a JAX exit has nowhere to put a
        float64 (`jax_enable_x64` is off) and casts them to the trace's declared
        precision. `require_same_representation` unifies namespace and device and
        deliberately not dtype, so a float32 geometry beside a float64 path is a
        legitimate FP32 artifact rather than a defect.
    """
    from fixtures.systems import singlet_ref, singlet_source

    from numerics import DType

    reason = _inbound_reason(device)
    if reason is not None:
        pytest.skip(reason)

    expected = trace_exit_state(device=device, precision=precision)
    cell = Cell(namespace=expected.namespace, device=expected.device, dtype=expected.dtype)
    reason = unavailable_reason(cell)
    if reason is not None:
        pytest.skip(reason)

    bundle = trace(
        singlet_ref(),
        singlet_source(),
        sampling=SAMPLING,
        execution={"device": str(device), "precision": str(precision)},
    )

    verify_placement(cell, bundle.positions_m)
    verify_placement(cell, bundle.directions)
    verify_placement(cell, bundle.amplitude, dtype=cell.dtype.precision.complex_dtype)

    declared = (
        DType.FLOAT64 if cell.namespace is ArrayNamespace.NUMPY else cell.dtype
    )
    verify_placement(cell, bundle.optical_path_m, dtype=declared)
    verify_placement(cell, bundle.measure_weight, dtype=declared)


def test_the_exit_state_derivation_says_where_each_request_goes() -> None:
    """`trace_exit_state`'s three branches, pinned without needing a device.

    A pure derivation, so it belongs with the placement test that consumes it
    rather than in a solver test: the parametrization above compares an
    observation against whatever this function returns, and if the function
    itself drifted the comparison would keep agreeing while covering something
    else. This is the half that says *what* it should return.

    The `cuda` FP64 row is the one to read carefully. It exits on the **host** on
    purpose: `jax_enable_x64` is off in every process this project runs, so JAX
    cannot represent float64 at all and `to_namespace` refuses the cast with
    `SILENT_DTYPE_DOWNCAST` rather than performing it. A host exit keeps a
    capability that works today; the trace still runs on the device.
    """
    from numerics import ArrayNamespace, DType

    cpu, cuda = DevicePlacement.parse("cpu"), DevicePlacement.parse("cuda")

    for precision, dtype in ((Precision.FP32, DType.FLOAT32), (Precision.FP64, DType.FLOAT64)):
        host = trace_exit_state(device=cpu, precision=precision)
        assert (host.namespace, host.device.kind, host.dtype) == (
            ArrayNamespace.NUMPY,
            DeviceKind.CPU,
            dtype,
        ), f"a cpu {precision} trace exits in host NumPy at {dtype}, got {host}"

    device = trace_exit_state(device=cuda, precision=Precision.FP32)
    assert (device.namespace, device.device.kind, device.dtype) == (
        ArrayNamespace.JAX,
        DeviceKind.CUDA,
        DType.FLOAT32,
    ), f"a cuda fp32 trace exits in JAX on the device, got {device}"

    fallback = trace_exit_state(device=cuda, precision=Precision.FP64)
    assert (fallback.namespace, fallback.device.kind, fallback.dtype) == (
        ArrayNamespace.NUMPY,
        DeviceKind.CPU,
        DType.FLOAT64,
    ), (
        f"a cuda fp64 trace has nowhere on the device to land -- jax_enable_x64 is off -- so "
        f"it exits on the host at float64, got {fallback}"
    )
