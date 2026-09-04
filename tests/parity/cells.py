"""What a cell is, where the cell set comes from, and the one tolerance derivation.

A **cell** is one `(namespace, device, dtype)` triple a kernel can be asked to
execute in. The cell set is *derived*, never listed: a hand-written matrix is a
second declaration of what a component supports, and it drifts from the first
one silently. Two derivation sources, because there are two kinds of component:

`load_capabilities(component)` succeeds
    The component has a measured capability pack, and the pack is the authority:
    devices from `devices`, namespaces from `namespaces_for(device)`, dtypes
    from `accepted_input_dtypes`. A cell the pack does not admit **never enters
    the parameter list at all** -- it is not a skip, because there is nothing to
    run and nothing to report.

`load_capabilities(component)` raises `UNKNOWN_COMPONENT`
    Ten of the seventeen catalog records carry `capabilities=None`, including
    both of this package's subjects (`M_PSF`, `C_RAY_TO_SCALAR`). They are
    repo-owned operations with no external backend to have measured, so nobody
    declared device or dtype rows for them. Their cells come from the two
    project-wide declarations that *do* exist: `COMPUTE_NAMESPACES` (which
    namespaces this project computes in at all) and
    `ArrayNamespace.can_leave_host` (which of those can hold device memory).
    **That asymmetry follows from who declared what and is not an omission.**

    The dtype axis in this branch is `PHASE_ACCUMULATION_FLOOR`, the project's
    declared working precision, and nothing wider. This is deliberate: a
    pack-less component has no *measured* dtype support, so enumerating
    `Precision` here would invent rows exactly as a hand-written list would.
    It also avoids a false failure -- `jax_enable_x64` is pinned off on every
    Chromatix backend import (`backends/chromatix/fields.py:179`), so a
    float64-family JAX cell would be a cell whose dtype cannot exist in this
    process, which `verify_dtype` refuses rather than silently downcasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from numerics.arrays import COMPUTE_NAMESPACES, numpy_dtype
from numerics.knowledge import load_capabilities
from numerics.precision import (
    PHASE_ACCUMULATION_FLOOR,
    ArrayNamespace,
    DeviceKind,
    DevicePlacement,
    DType,
)

#: Why a torch cell is expected to fail today, naming all three refusal sites so
#: the reason survives being read in isolation from a test report. CHE-248 (T4)
#: is the ticket that reverses them; the marks these cells carry are
#: `strict=True` precisely so that reversing the refusals without removing the
#: marks fails this suite rather than passing it quietly.
TORCH_IS_NOT_A_COMPUTE_NAMESPACE = (
    "torch is not a compute namespace yet: refused at numerics/arrays.py "
    "COMPUTE_NAMESPACES, at numerics/arrays.py::xp_for with "
    "NAMESPACE_NOT_A_COMPUTE_NAMESPACE, and at "
    "representations/contracts.py::adopt_array. CHE-248 (T4) reverses all three; "
    "this mark is strict so that landing T4 without updating this suite fails it."
)


@dataclass(frozen=True)
class Cell:
    """One execution state a kernel can be asked for. A request, not an observation.

    Rule 1 -- the three fields are one request: a namespace without a device is
    not an address, and a dtype means something different in each namespace
    (`float64` is ordinary in NumPy and unavailable in JAX with x64 off). They
    are validated and skipped as a unit.

    Deliberately **not** an `ArrayState`. `ArrayState`'s own docstring says
    "Every field observed, never requested. Nothing in this project constructs
    one from a request", and that rule is load-bearing here: the whole point of
    the read-back in `conftest.verify_placement` is that the requested state and
    the observed state are different objects that can disagree. A request also
    cannot be compared to an observation by equality -- a request for `cuda`
    carries no ordinal and the observation of it is `cuda:0` -- which is why
    `verify_placement` compares device *kind* and reports the ordinal rather
    than asserting on it.
    """

    namespace: ArrayNamespace
    device: DevicePlacement
    dtype: DType

    def __str__(self) -> str:
        return f"{self.namespace.value}-{self.device}-{self.dtype.value}"

    @property
    def param(self) -> Any:
        """This cell as a `pytest.param`, carrying the marks its state implies.

        The marks are derived from the cell rather than written beside it, so a
        newly derived cell cannot arrive unmarked:

        `gpu`
            Every CUDA cell. This is what routes cuda cells out of the default
            gate and into `make test-gpu`, and what makes the landed
            `tests/conftest.py` hook -- not a reimplementation of it here --
            responsible for the dedicated-session skip.
        `xfail(strict=True)`
            Every cell in a namespace this project does not compute in. Today
            that is exactly torch.
        """
        marks = []
        if self.device.kind is DeviceKind.CUDA:
            marks.append(pytest.mark.gpu)
        if self.namespace not in COMPUTE_NAMESPACES:
            marks.append(pytest.mark.xfail(strict=True, reason=TORCH_IS_NOT_A_COMPUTE_NAMESPACE))
        return pytest.param(self, id=str(self), marks=tuple(marks))


def cells_for(
    component: str, *, complex_data: bool, directory: Path | None = None
) -> tuple[Cell, ...]:
    """Every cell `component` admits, derived from a declaration rather than listed.

    Parameters
    ----------
    component
        A catalog operation id or a capability record id. Which of the two
        derivation branches runs is decided by whether a record exists, not by
        the caller.
    complex_data
        Whether the kernel's data is complex. A fact about the physics -- a
        scalar field amplitude is complex, a ray position is not -- and the only
        thing a caller states about dtypes. It selects a *family*, never a
        width; the width comes from the pack or from the project's floor.
    directory
        Forwarded to `load_capabilities`, so a synthetic record under `tmp_path`
        can drive this derivation without editing anything in `src/`. That is
        what makes the "no hand-written list" claim falsifiable rather than
        merely asserted.

    Returns a tuple ordered namespace-then-device-then-dtype, so test ids are
    stable across runs and a diff of a parameter list reads as a diff.
    """
    try:
        capabilities = load_capabilities(component, directory=directory)
    except ValueError as exc:
        if getattr(exc, "code", None) != "UNKNOWN_COMPONENT":
            raise
        return _cells_without_a_pack(complex_data=complex_data)

    dtypes = sorted(
        (d for d in capabilities.accepted_input_dtypes if d.is_complex is complex_data),
        key=lambda d: d.component_bits,
    )
    cells = [
        Cell(namespace=namespace, device=DevicePlacement(device), dtype=dtype)
        for namespace in sorted(capabilities.namespaces, key=lambda n: n.value)
        for device in sorted(capabilities.devices, key=lambda d: d.value)
        # `namespaces_for` rather than the flat `namespaces` set: Optiland reaches
        # CUDA only through torch, so `numpy-cuda` is a cell the pack refuses and
        # not one that skips. Also refused one layer down -- `ArrayState`
        # raises `NUMPY_CANNOT_LEAVE_HOST` -- which is why it must be excluded
        # here rather than allowed to fail as if it were a real result.
        if namespace in capabilities.namespaces_for(device)
        for dtype in dtypes
    ]
    return tuple(cells)


def _cells_without_a_pack(*, complex_data: bool) -> tuple[Cell, ...]:
    """The pack-less derivation. See this module's docstring for why it differs."""
    floor = PHASE_ACCUMULATION_FLOOR
    dtype = floor.complex_dtype if complex_data else floor.real_dtype
    if dtype is None:  # pragma: no cover - fp32 has a complex spelling
        raise AssertionError(f"{floor} declares no complex dtype to compute in")
    return tuple(
        Cell(namespace=namespace, device=DevicePlacement(kind), dtype=dtype)
        for namespace in sorted(COMPUTE_NAMESPACES, key=lambda n: n.value)
        for kind in DeviceKind
        if kind is not DeviceKind.CUDA or namespace.can_leave_host
    )


def tolerance_for(cell: Cell, *, accumulation_length: int, matmul: bool) -> float:
    """The only place in `tests/parity/` a floating-point tolerance is decided.

    `AGENTS.md`: "Do not widen a tolerance merely to make a benchmark pass."
    A single derivation point is what makes that rule enforceable -- a literal
    at a comparison site can be nudged by one character in a way no reviewer
    sees, and `test_cells.py::test_no_tolerance_is_spelled_at_a_comparison`
    forbids it mechanically.

    Derived from four things, each with a reason rather than a fudge factor:

    unit roundoff
        `np.finfo(dtype).eps / 2`, read off the dtype rather than tabulated, so
        a new dtype in the vocabulary needs no edit here. For complex64 NumPy
        reports its component's eps, which is the right number: the error is
        per real component.
    accumulation length
        `sqrt(n)`. The *probabilistic* growth of rounding error over an
        `n`-term sum, not the deterministic `n * u` worst case, because both
        NumPy and XLA reduce pairwise/blockwise and the worst case is
        unreachable for them -- using it would make the gate so loose that the
        TF32 failure this fixture exists to catch would pass. `n = 1` for an
        elementwise kernel, and it is the caller's job to say which.
    complex arithmetic
        A factor of 2. A complex multiply is four real multiplies and two adds,
        so each output component carries one more rounding than the real case.
    contraction freedom
        A further factor of 2 when `matmul` is set. `optimize=True` lets the
        library choose the contraction order, so two cells may associate the
        same sum differently; that is a legitimate difference between cells and
        not a defect in either.

    A safety factor of 2 covers the FMA contraction each backend is free to
    apply. It is stated here rather than folded into one of the factors above so
    that it can be argued about on its own.

    Measured against this derivation on 256 complex64 wavelets through the
    `couplers/ray_to_scalar.py` ramp sum (CHE-244, `agent_solver_gpu` on device
    6, JAX 0.6.2, RTX A6000), relative L2 against the numpy-cpu-complex64 leg:

        derived tolerance                    7.629e-06
        jax-cuda, precision="highest"        3.164e-07   24x inside the gate
        jax-cuda, precision flag dropped     2.664e-04   35x outside the gate

    So the negative control is **842x** the guarded error, and the gate sits
    between them with roughly a factor of 25 to spare in each direction. The
    same three runs against a complex128 host reference give 2.09e-07
    (numpy-cpu-complex64), 2.82e-07 (jax-cuda, highest) and 2.66e-04 (jax-cuda,
    no flag), reproducing `matmul_precision_kwargs`' own docstring measurement
    of 2.6e-07 / 2.3e-07 / 3.5e-04.

    None of these numbers was used to *set* the tolerance -- the derivation
    above is independent of them -- and they are recorded so that a future
    reader can see how much room the gate has rather than having to re-measure
    to find out.
    """
    if accumulation_length < 1:
        raise ValueError(f"accumulation_length must be at least 1, got {accumulation_length}")
    unit_roundoff = float(np.finfo(numpy_dtype(cell.dtype)).eps) / 2.0
    growth = math.sqrt(accumulation_length)
    complex_factor = 2.0 if cell.dtype.is_complex else 1.0
    contraction_factor = 2.0 if matmul else 1.0
    safety = 2.0
    return safety * unit_roundoff * growth * complex_factor * contraction_factor
