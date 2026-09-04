"""Subject 2: the ramp-sum contraction, and the TF32 failure it exists to catch.

This is the kernel `src/couplers/ray_to_scalar.py:1485` runs in its direct
reconstruction:

    u = xp.einsum("n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True, **dot)

and it is the exact contraction `numerics/arrays.py::matmul_precision_kwargs`
was measured on. Two halves, and the second is the whole reason for the first:

1. jax-cuda-complex64 agrees with jax-cpu-complex64 under `precision="highest"`,
   at the tolerance `tolerance_for` derives for a contraction of this length.
2. **A negative control that drops the precision flag breaks that gate.** Half 1
   on its own is not evidence. If the fixture cannot see XLA silently computing
   a complex64 dot in TF32 -- a 10-bit mantissa reported as complex64 -- then it
   cannot see the one failure mode it was built for, and a passing half 1 means
   only that two legs agreed about something.

**The operands are constructed here rather than obtained by calling the
coupler.** The coupler needs a `RayBundle`, a declared measure, a grazing
policy and a reconstruction choice, none of which bear on the contraction, and
`C_RAY_TO_SCALAR`'s own physics gates already live in
`tests/physics/test_ray_to_scalar.py`. What that trade costs is fidelity to the
production line, so `test_the_production_contraction_still_matches_this_mirror`
reads the source and fails if the spelling drifts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from numerics.arrays import matmul_precision_kwargs, to_namespace, xp_for
from numerics.precision import ArrayNamespace, DeviceKind, DType
from parity.cells import cells_for, tolerance_for
from parity.conftest import verify_placement

COMPONENT = "C_RAY_TO_SCALAR"
CELLS = cells_for(COMPONENT, complex_data=True)

#: 256 wavelets, matching the population `matmul_precision_kwargs`' docstring
#: measurement used, so the numbers in that docstring and the numbers this test
#: reports are about the same contraction length.
_N_RAYS = 256
_GRID = (32, 32)
_PITCH_M = (2.0e-6, 2.0e-6)
_WAVELENGTH_M = 550.0e-9

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "couplers" / "ray_to_scalar.py"


def _operands() -> tuple[Any, Any, Any]:
    """`(coefficient, ramp_y, ramp_x)` in complex128 on the host, deterministically.

    Shapes and roles mirror the production line exactly: `(n,)`, `(n, ny)`,
    `(n, nx)`. The directions span a cone rather than a single angle so the
    per-ray phases are spread across the grid -- a contraction whose terms all
    have the same phase would sum coherently and hide cancellation, which is
    where reduced mantissa precision actually shows up.
    """
    rng = np.random.default_rng(20260903)
    directions_xy = rng.uniform(-0.25, 0.25, size=(_N_RAYS, 2))
    amplitude = rng.uniform(0.5, 1.5, size=_N_RAYS)
    constant_phase = rng.uniform(-np.pi, np.pi, size=_N_RAYS)
    coefficient = (amplitude * np.exp(1j * constant_phase)).astype(np.complex128)

    ny, nx = _GRID
    dy, dx = _PITCH_M
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx
    wavenumber = 2.0 * np.pi / _WAVELENGTH_M
    ramp_y = np.exp(1j * wavenumber * np.outer(directions_xy[:, 1], y)).astype(np.complex128)
    ramp_x = np.exp(1j * wavenumber * np.outer(directions_xy[:, 0], x)).astype(np.complex128)
    return coefficient, ramp_y, ramp_x


_OPERANDS = _operands()


def _contract(xp: Any, operands: tuple[Any, Any, Any], *, dot: dict[str, Any]) -> Any:
    """The production contraction, with the precision flags the caller chose."""
    coefficient, ramp_y, ramp_x = operands
    return xp.einsum("n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True, **dot)


def _reference(dtype: DType) -> Any:
    """The host leg at the cell's own dtype. See `test_psf_parity._reference`."""
    operands = tuple(
        to_namespace(value, namespace=ArrayNamespace.NUMPY, dtype=dtype) for value in _OPERANDS
    )
    return _contract(np, operands, dot=matmul_precision_kwargs(ArrayNamespace.NUMPY))


def _relative_l2(observed: Any, reference: Any) -> float:
    """`||observed - reference|| / ||reference||`, both on the host in complex128.

    L2 rather than peak-relative here because a contraction distributes its
    error across the whole output, and the quantity the `matmul_precision_kwargs`
    docstring reports is this one.
    """
    a = np.asarray(to_namespace(observed, namespace=ArrayNamespace.NUMPY), dtype=np.complex128)
    b = np.asarray(to_namespace(reference, namespace=ArrayNamespace.NUMPY), dtype=np.complex128)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("cell", [cell.param for cell in CELLS])
def test_the_ramp_sum_agrees_across_cells(cell: Any, place: Any) -> None:
    """AC-6, first half: every cell contracts to the same field, at a derived tolerance."""
    operands = tuple(place(cell, value) for value in _OPERANDS)
    xp = xp_for(cell.namespace)
    result = _contract(xp, operands, dot=matmul_precision_kwargs(cell.namespace))
    verify_placement(cell, result)

    tolerance = tolerance_for(cell, accumulation_length=_N_RAYS, matmul=True)
    error = _relative_l2(result, _reference(cell.dtype))
    assert error <= tolerance, (
        f"{cell} contracted to a field {error:.3e} (relative L2) from the host leg, "
        f"above the derived tolerance {tolerance:.3e}"
    )


@pytest.mark.parametrize("cell", [cell.param for cell in CELLS])
def test_dropping_the_precision_flag_breaks_the_gate(cell: Any, place: Any) -> None:
    """AC-6, second half: the negative control, and it must actually fail.

    Skipped explicitly rather than passed silently on a non-CUDA cell. TF32 is a
    property of the device's tensor cores, so on the host there is no reduced
    precision to fall back to and `matmul_precision_kwargs` returns an empty
    mapping for NumPy by design -- a control that "passes" there would be
    reporting that nothing happened as though something had been ruled out.
    """
    if cell.device.kind is not DeviceKind.CUDA:
        pytest.skip(
            f"the TF32 negative control needs a device: {cell} computes on the host, where "
            "there is no reduced-precision dot to fall back to and "
            "matmul_precision_kwargs is empty by design"
        )

    operands = tuple(place(cell, value) for value in _OPERANDS)
    xp = xp_for(cell.namespace)
    reference = _reference(cell.dtype)
    tolerance = tolerance_for(cell, accumulation_length=_N_RAYS, matmul=True)

    guarded = _relative_l2(_contract(xp, operands, dot=matmul_precision_kwargs(cell.namespace)),
                           reference)
    unguarded = _relative_l2(_contract(xp, operands, dot={}), reference)

    assert unguarded > tolerance, (
        f"{cell} contracted to within {unguarded:.3e} of the host leg with the precision "
        f"flag DROPPED, inside the tolerance {tolerance:.3e}. The gate cannot see TF32 on "
        "this device, so the first half of AC-6 proves nothing: either the tolerance is too "
        "wide or this device does not use TF32 for a complex64 dot."
    )
    assert guarded < unguarded, (
        f"{cell} was no more accurate with precision='highest' ({guarded:.3e}) than without "
        f"it ({unguarded:.3e}); the flag is not reaching the contraction"
    )


def test_the_production_contraction_still_matches_this_mirror() -> None:
    """The operands are built here, so the source line is pinned rather than trusted.

    Three things have to hold for this module to be about the kernel it claims:
    the subscripts, `optimize=True`, and the precision mapping actually being
    passed. A drift in any of them makes this subject a test of a contraction
    the coupler no longer performs -- which would still pass, which is why it is
    checked mechanically.
    """
    source = _SOURCE.read_text(encoding="utf-8")
    contraction = re.search(
        r'xp\.einsum\(\s*"n,ny,nx->yx".*?\)', source, flags=re.DOTALL
    )
    assert contraction is not None, f"no 'n,ny,nx->yx' contraction left in {_SOURCE.name}"
    line = contraction.group(0)
    assert "optimize=True" in line, line
    assert "**dot" in line, line
    assert "dot = matmul_precision_kwargs(namespace)" in source, (
        "the coupler no longer derives its dot precision from "
        "numerics.arrays.matmul_precision_kwargs"
    )
