"""Subject 4: all three sources build the same field in every cell.

CHE-246 (T2) acceptance criteria 1, 3 and 5. Until T2 the three sources were
hardcoded NumPy with no device or namespace argument, so the wave path had no GPU
entry point at all and T0's `M_PSF` subject had to assemble its `ScalarField` by
hand. This module is where the three now answer to the cells.

Two claims, and the second is the one worth reading carefully.

**Placement** (AC-1). Each source produces a field in the cell's namespace on the
cell's device, observed with `numerics.arrays.array_state` through
`conftest.verify_placement` rather than trusted from the argument that requested
it. `knowledge/capabilities/M_WAVE_CHROMATIX.json` records why that distinction
is not pedantry: "a process-global JAX platform pin produces a successful
complex64 run on the host while the caller asked for CUDA, with no error raised."

**Bit-identity, which is how the float64 validity line becomes testable** (AC-3).
All three records declare "the phase ramp is accumulated in float64 before the
cast", and `jax_enable_x64` is off in every process this project runs, so JAX
*cannot* accumulate in float64. CHE-246 resolves that by accumulating on the host
and moving the cast array once (`sources/_grid.py` argues it at length), which
has a consequence a test can see from outside the package: every cell's array is
**bit-identical**, not merely close. So the assertion here is exact equality and
carries no tolerance at all -- `tolerance_for` is deliberately not imported.

That is a real gate rather than a tautology. Accumulating the ramp in the target
namespace instead -- the one edit that would break the declaration -- costs
~6e-5 rad on this grid against the ~6e-8 the cast itself costs, which is three
orders of magnitude above exact equality and fails immediately. A tolerance here
would have hidden exactly that.

Nothing here is an oracle. Two cells agreeing says they computed the same field,
not that the field is right; `tests/physics/test_coherent_sources.py` holds the
analytic gates that can settle that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from numerics.precision import ArrayNamespace, DType
from parity.cells import Cell, cells_for
from parity.conftest import unavailable_reason, verify_placement
from representations.geometry import ReferenceSurface
from sources import gaussian_beam, plane_wave, spherical_wave
from sources.plane_wave import transverse_wavevector_from_angle

#: The three source records. Each carries `capabilities=None`, so `cells_for`
#: takes the pack-less branch and the cell set is `COMPUTE_NAMESPACES` x
#: `can_leave_host` at the project's declared floor -- see `parity/cells.py` on
#: why that asymmetry follows from who declared what.
COMPONENTS = (
    "S_SOURCE_PLANE_WAVE",
    "S_SOURCE_GAUSSIAN_BEAM",
    "S_SOURCE_SPHERICAL_WAVE",
)

#: Every source's cell set, which is the same set for all three because all
#: three are pack-less. Derived rather than shared as a constant, so a pack
#: landing for one of them changes that source's cells and not the others'.
CELLS = {component: cells_for(component, complex_data=True) for component in COMPONENTS}

_SHAPE = (32, 32)
_PITCH_M = (2.0e-6, 2.0e-6)
_WAVELENGTH_M = 550.0e-9
_SURFACE = ReferenceSurface(name="image_surface", z_m=0.0, medium_index=1.0)

#: A real illumination angle rather than a raw `(k_y, k_x)`, through the
#: project's own converter. 0.05 rad at 550 nm is |k_t| ~ 5.7e5 rad/m against a
#: Nyquist limit of pi/d = 1.57e6, so the sampled ramp is about 2.7x inside the
#: aliasing bound and the refusals in `_grid.py` are not what is being tested.
_TILT_RAD_PER_M = transverse_wavevector_from_angle(
    0.05, 0.3, wavelength_m=_WAVELENGTH_M, medium_index=_SURFACE.medium_index
)

_COMMON: dict[str, Any] = {
    "sample_pitch_m": _PITCH_M,
    "wavelength_m": _WAVELENGTH_M,
    "reference_surface": _SURFACE,
}

#: What distinguishes each source, and nothing else. The geometry is chosen to be
#: comfortably inside every refusal each source declares: the waist sits 2.7x
#: inside the grid half-extent, and the point source is 1 mm upstream of the
#: plane, where the worst local `sin(theta)` is ~0.03 against the ~0.14 this
#: pitch carries.
_EXTRA: dict[str, dict[str, Any]] = {
    "S_SOURCE_PLANE_WAVE": {"transverse_wavevector_rad_per_m": _TILT_RAD_PER_M},
    "S_SOURCE_GAUSSIAN_BEAM": {
        "waist_radius_m": 1.2e-5,
        "transverse_wavevector_rad_per_m": _TILT_RAD_PER_M,
    },
    "S_SOURCE_SPHERICAL_WAVE": {"source_position_m": (0.0, 0.0, -1.0e-3)},
}

_SOURCES = {
    "S_SOURCE_PLANE_WAVE": plane_wave,
    "S_SOURCE_GAUSSIAN_BEAM": gaussian_beam,
    "S_SOURCE_SPHERICAL_WAVE": spherical_wave,
}


def _build(component: str, cell: Cell) -> Any:
    return _SOURCES[component](
        _SHAPE, **_COMMON, **_EXTRA[component], namespace=cell.namespace, device=cell.device
    )


def _host_cell(component: str) -> Cell:
    """The numpy-cpu cell of `component`'s own set, found rather than written out.

    Looked up instead of constructed so that this module still names no
    namespace/device/dtype triple of its own -- `parity/cells.py`'s "no
    hand-written list" rule applies here too.
    """
    for cell in CELLS[component]:
        if cell.namespace is ArrayNamespace.NUMPY:
            return cell
    raise AssertionError(f"{component} declares no host cell to compare against")


# AC-1's placement assertion lives inside the parametrized test below rather
# than in a test of its own. A separate one looping over `CELLS[component]`
# existed briefly and was wrong in a way worth recording: parametrized on
# `component` only, it carried no `gpu` mark, so its cuda leg was skipped by
# `make test` and *deselected* by `make test-gpu` -- covered by neither landed
# command, while still being able to allocate device memory in a mixed session,
# which `tests/conftest.py`'s dedicated-session rule exists to prevent. A cell
# that is not a pytest parameter cannot carry the mark its state implies.


@pytest.mark.parametrize("cell", [cell.param for cell in CELLS[COMPONENTS[0]]])
@pytest.mark.parametrize("component", COMPONENTS)
def test_a_source_is_bit_identical_across_cells(component: str, cell: Any) -> None:
    """AC-3: exact equality, which is what makes the float64 line falsifiable.

    `np.array_equal` on the complex64 bytes and no tolerance anywhere. See the
    module docstring: the host accumulation is what buys exactness, and a
    target-namespace accumulation would miss by ~1e3 x the cast's own error.
    """
    reason = unavailable_reason(cell)
    if reason is not None:
        pytest.skip(reason)

    observed = _build(component, cell)
    reference = _build(component, _host_cell(component))

    verify_placement(cell, observed.u)
    assert np.array_equal(np.asarray(observed.u), np.asarray(reference.u)), (
        f"{component} in {cell} is not bit-identical to its host cell. The three sources "
        "accumulate on the host and move the cast array once, precisely so that this is "
        "exact; a difference here means the arithmetic moved into the target namespace, "
        "where jax_enable_x64 is off and the float64 validity line these records declare "
        "cannot hold"
    )

    # The declarations that travel with the field, not just its bytes. A source
    # that placed the array and dropped a convention would pass the comparison
    # above and still be wrong about the field.
    assert observed.sample_pitch_m == reference.sample_pitch_m
    assert observed.wavelength_m == reference.wavelength_m
    assert observed.reference_surface == reference.reference_surface
    assert observed.validity == reference.validity


def test_every_source_declares_the_same_storage_dtype_in_every_cell() -> None:
    """One storage dtype across the whole matrix, and it is the declared one.

    `SOURCE_DTYPE_DECLARED` is read from the package rather than restated, so
    this cannot drift from it. The check exists because `complex64` is not a
    default that happened -- `numerics.negotiate` refuses `complex128` against
    the measured chromatix row with `LOSSY_DOWNCAST_REQUIRED`, so a wider field
    could not be propagated at all -- and a cell quietly delivering something
    else would be a field the wave path cannot consume.
    """
    from sources._grid import ACCUMULATION_DTYPE, SOURCE_DTYPE_DECLARED

    assert SOURCE_DTYPE_DECLARED is DType.COMPLEX64
    assert ACCUMULATION_DTYPE is DType.FLOAT64
    for component in COMPONENTS:
        assert {cell.dtype for cell in CELLS[component]} == {DType.COMPLEX64}, component
        assert len(CELLS[component]) == 3, [str(cell) for cell in CELLS[component]]
