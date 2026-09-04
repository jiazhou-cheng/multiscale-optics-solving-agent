"""The one parity fixture: construct a buffer in a cell, and observe where it landed.

Every buffer a parity test computes on comes from the `place` fixture, and
`place` never returns without having read the buffer's state back off the buffer
itself. That is the whole mechanism. `knowledge/capabilities/M_WAVE_CHROMATIX.json`
records the failure it exists for:

    "A requested device must never be reported as an actual one -- a
    process-global JAX platform pin produces a successful complex64 run on the
    host while the caller asked for CUDA, with no error raised."

A test that trusts its own `device="cuda"` argument cannot see that. A test that
reads `array_state` back can, and `test_cells.py`'s
`test_a_buffer_that_landed_on_the_host_fails_a_cuda_cell` proves the read-back
fails when it should.

Four outcomes are kept distinguishable, because "skipped" without a reason is
how a suite comes to prove nothing:

dependency missing
    The namespace's library is not importable. Skipped here.
device not attached
    No CUDA device in this container. Skipped here, using
    `tests/conftest.py::cuda_unavailable_reason` -- imported rather than
    reimplemented, so the two cannot disagree about what "usable device" means.
session not dedicated
    Handled entirely by the landed `gpu`-marker hook in `tests/conftest.py`;
    CUDA cells carry `pytest.mark.gpu` via `Cell.param` and nothing in this
    package repeats that logic.
cell not admissible
    Never enters the parameter list. See `cells.cells_for`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from numerics.arrays import array_state, to_namespace
from numerics.precision import ArrayNamespace, DeviceKind, DType
from parity.cells import Cell

#: The import each non-NumPy namespace needs. NumPy is absent on purpose: it is
#: a hard dependency of `src/` itself, so "numpy is missing" is not a skip
#: condition, it is a broken environment.
_NAMESPACE_MODULES = {
    ArrayNamespace.JAX: "jax",
    ArrayNamespace.TORCH: "torch",
}


def unavailable_reason(cell: Cell) -> str | None:
    """Why this cell cannot run here, or `None` if it can.

    Ordered dependency-first: with no library there is no device question to
    ask, and reporting "no CUDA device" for a container that simply has no JAX
    would send a reader to the wrong problem.
    """
    module = _NAMESPACE_MODULES.get(cell.namespace)
    if module is not None:
        try:
            __import__(module)
        except ImportError as exc:
            return f"dependency missing: {cell.namespace.value} is not importable ({exc})"

    if cell.device.kind is DeviceKind.CUDA:
        # The landed helper, not a second opinion: it consults both frameworks
        # because a container can be half-enabled, and duplicating that here
        # would let this package call a device usable that `make test-gpu`
        # calls unusable.
        from conftest import cuda_unavailable_reason

        reason = cuda_unavailable_reason()
        if reason is not None:
            return f"device not attached: {reason}"
    return None


def verify_placement(cell: Cell, value: Any, *, dtype: DType | None = None) -> Any:
    """Fail the test unless `value` is really in `cell`'s state. Returns `value`.

    Compares field by field rather than by constructing an `ArrayState` from the
    cell, for two reasons. `ArrayState` is by declaration an observation and
    nothing in this project builds one from a request. And equality would be the
    wrong test even if it were allowed: a request for `cuda` carries no ordinal
    while its observation is `cuda:0`, so device **kind** is what is asserted
    and the ordinal is reported. A cell that asked for `cuda:1` and landed on
    `cuda:0` is out of scope here -- the shared-GPU policy is one device per
    workload, so there is no cell that names an ordinal.

    `dtype` overrides the compared dtype for the case where a kernel's output
    family differs from its input family by construction: `psf` returns
    `|u|^2`, which is real where the field was complex. Overriding it is
    narrower than skipping the check on outputs, which would leave the most
    interesting buffer -- the one a measurement returns -- unobserved.
    """
    expected_dtype = cell.dtype if dtype is None else dtype
    observed = array_state(value)
    problems = []
    if observed.namespace is not cell.namespace:
        problems.append(f"namespace {observed.namespace.value} != requested {cell.namespace.value}")
    if observed.device.kind is not cell.device.kind:
        problems.append(f"device {observed.device} != requested {cell.device}")
    if observed.dtype is not expected_dtype:
        problems.append(f"dtype {observed.dtype.value} != requested {expected_dtype.value}")
    if problems:
        pytest.fail(
            f"cell {cell} did not land where it was requested: {'; '.join(problems)}. "
            f"requested={cell.namespace.value}:{expected_dtype.value}@{cell.device}, "
            f"observed={observed}. A requested device reported as an actual one is the "
            "failure this fixture exists to catch, not a tolerance question."
        )
    return value


@pytest.fixture
def place() -> Any:
    """A callable putting host data into a cell, with the placement observed back.

    A fixture rather than a plain helper so that the skip decisions land inside
    a test's own setup, where pytest attributes them to the parametrized cell
    that could not run, and so that no parity test can reach a buffer by
    another route.
    """

    def _place(cell: Cell, values: Any, *, dtype: DType | None = None) -> Any:
        reason = unavailable_reason(cell)
        if reason is not None:
            pytest.skip(reason)
        target = cell.dtype if dtype is None else dtype
        # `to_namespace` is production code and is the only mover used here.
        # A bespoke `jnp.asarray` + `device_put` in the fixture would be a
        # second implementation of the move whose bugs this package could not
        # see, since it is also what T1/T2/T3 will be asserted through.
        moved = to_namespace(
            np.asarray(values), namespace=cell.namespace, device=cell.device, dtype=target
        )
        return verify_placement(cell, moved, dtype=target)

    return _place
