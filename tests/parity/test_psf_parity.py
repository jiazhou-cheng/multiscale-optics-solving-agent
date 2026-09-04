"""Subject 1: `M_PSF` agrees across numpy-cpu, jax-cpu and jax-cuda.

**The field comes from a source, which is CHE-246 (T2)'s completion signal.**
Until T2 the three sources were hardcoded NumPy with no device or namespace
parameter, so no source in this project could construct a field on a device and
every cell here assembled its own `ScalarField` from host data through the
`place` fixture. T0 recorded that hand construction as a finding and named its
removal as the signal. It is removed: each cell now calls
`sources.gaussian_beam` with the cell's own `namespace` and `device`, and every
assertion below is unchanged.

**Why `gaussian_beam` and not `plane_wave`**, which is the function CHE-246's
AC-4 names. A plane wave has uniform `|u|`, so `|u|^2` has no unique maximum and
`peak_index` -- the one assertion in this module that is *exact* rather than
tolerant, and which this file's own docstring calls load-bearing -- becomes a
comparison between two arbitrary argmax tie-breaks over an array whose last bits
differ per namespace. `gaussian_beam` is the same carrier ramp times a real
envelope (`operations/catalog.py` says so of `S_SOURCE_GAUSSIAN_BEAM`: "exactly
the carrier ramp plane_wave writes"), so it exercises the same placement path and
the same complex arithmetic while leaving the peak unambiguous.
`test_sources_parity.py` is where `plane_wave` itself is held to the cells.

**What this subject proves, and what it does not.** `measurements.psf` is
`|u|^2` times a declared scalar -- purely elementwise, with one reduction for
the energy normalization and no dot product anywhere. So it never touches TF32,
and it cannot exercise the part of `tolerance_for` that exists for contraction
error. It proves the fixture runs end to end through a real operation on a real
representation. `test_ray_to_scalar_parity.py` is the subject that proves the
tolerance machinery can see anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from measurements.psf import PSF_NORMALIZATIONS, psf
from numerics.arrays import to_namespace
from numerics.precision import ArrayNamespace, DType
from parity.cells import Cell, cells_for, tolerance_for
from parity.conftest import unavailable_reason, verify_placement
from representations.geometry import ReferenceSurface
from representations.scalar import ScalarField
from sources import gaussian_beam
from sources.plane_wave import transverse_wavevector_from_angle

COMPONENT = "M_PSF"
CELLS = cells_for(COMPONENT, complex_data=True)

_SHAPE = (32, 32)
_PITCH_M = (2.0e-6, 2.0e-6)
_WAVELENGTH_M = 550.0e-9
_SURFACE = ReferenceSurface(name="image_surface", z_m=0.0, medium_index=1.0)

#: `w0`, chosen so the envelope is well inside a 32-sample grid at this pitch:
#: the grid half-extent is 32 um and this waist puts the `1/e` amplitude radius
#: at 12 um, i.e. 2.7 w0 of half-extent, which `S_SOURCE_GAUSSIAN_BEAM`'s own
#: truncation note puts past the ~1e-7 power level. A truncated Gaussian would
#: still be a valid parity subject -- both legs would truncate identically -- but
#: it would make `peak_index` sensitive to the rim rather than to the envelope.
_WAIST_M = 1.2e-5

#: A tilt, and it is load-bearing exactly as T0's hand-built one was: without it
#: every cell would agree on a real, separable array and the complex arithmetic
#: `tolerance_for` accounts for would never happen. Stated as an angle through
#: the project's own converter rather than as a raw `(k_y, k_x)`, so the cell is
#: a physical illumination rather than two numbers that happen to be inside
#: Nyquist. 0.05 rad at 550 nm is |k_t| ~ 5.7e5 rad/m against a Nyquist limit of
#: 1.57e6, so the ramp is sampled about 2.7x above the aliasing bound.
_TILT_RAD_PER_M = transverse_wavevector_from_angle(
    0.05, 0.3, wavelength_m=_WAVELENGTH_M, medium_index=_SURFACE.medium_index
)


def _field(cell: Cell) -> ScalarField:
    """The subject's field, constructed **in the cell** by a production source.

    This is what CHE-246 (T2) made possible and it is the whole of T0's
    completion signal: no `ScalarField(...)` assembled here, no `place` fixture
    on the input, and the namespace and device are the cell's.

    The `place` fixture is consequently no longer used by this subject: it exists
    to move host data into a cell, and there is no host data to move. What is
    still borrowed from `tests/parity/conftest.py` is the half that matters --
    `unavailable_reason` for the skip taxonomy and `verify_placement` for the
    read-back -- because a source that cannot reach a device *raises*, and "no
    CUDA device attached" is an environment fact rather than a failure.
    """
    field = gaussian_beam(
        _SHAPE,
        sample_pitch_m=_PITCH_M,
        wavelength_m=_WAVELENGTH_M,
        reference_surface=_SURFACE,
        waist_radius_m=_WAIST_M,
        transverse_wavevector_rad_per_m=_TILT_RAD_PER_M,
        namespace=cell.namespace,
        device=cell.device,
    )
    # The *input* observed too, not only `psf`'s output. Without this the subject
    # would assert that the measurement did not move the data while taking on
    # trust that the source put it there in the first place, which is the exact
    # substitution `conftest.verify_placement` exists to refuse.
    return _field_with_observed_placement(cell, field)


def _field_with_observed_placement(cell: Cell, field: ScalarField) -> ScalarField:
    verify_placement(cell, field.u)
    return field


def _reference(dtype: DType, normalization: str) -> Any:
    """The host leg, **at the cell's own dtype**. Characterization, not an oracle.

    Deliberately not a complex128 reference, and since CHE-246 that is a
    property of the source rather than a construction here. `gaussian_beam`
    accumulates its envelope and ramp in host float64 and casts once to
    complex64 whatever `namespace` asks for -- it must, because
    `jax_enable_x64` is off and JAX cannot represent float64 -- so **every cell
    receives bit-identical input bytes**, and what is left over is exactly the
    namespace and device difference this package exists to see. Comparing a
    complex64 cell against a complex128 leg would instead fold the input
    quantization into a number `tolerance_for` derives for *arithmetic*, so the
    gate would be measuring the cast.

    It follows that the numpy-cpu cell compares against itself. That comparison
    is not vacuous but it is weak -- it says the placement path is a no-op on
    the host, which is what a host code path is supposed to be -- and the
    asymmetry is the point of `tests/parity/__init__.py`'s note that the host leg
    is characterization.
    """
    return psf(_field(_host_cell(dtype)), normalization=normalization)


def _host_cell(dtype: DType) -> Cell:
    """The numpy-cpu cell of this subject's own set, found rather than written out.

    Looked up instead of constructed, for two reasons. `parity/cells.py`'s "no
    hand-written namespace/device/dtype list" rule applies to this module too.
    And taking the device from a fixed index -- `CELLS[0].device`, which is what
    this was -- silently assumes the ordering: if that entry were ever a CUDA
    cell, this would build a *numpy-cuda* request and `to_namespace` would raise
    `NUMPY_CANNOT_LEAVE_HOST` inside the reference leg, i.e. a failure in the
    comparison rather than a skip in the cell.
    """
    for cell in CELLS:
        if cell.namespace is ArrayNamespace.NUMPY:
            return Cell(namespace=cell.namespace, device=cell.device, dtype=dtype)
    raise AssertionError(f"{COMPONENT} declares no host cell to compare against")


def _host(value: Any) -> Any:
    return np.asarray(to_namespace(value, namespace=ArrayNamespace.NUMPY), dtype=np.float64)


def _relative(observed: Any, reference: Any) -> float:
    """Max absolute deviation, relative to the reference's own peak magnitude.

    Peak-relative rather than pointwise-relative: a PSF has near-zero cells by
    construction, and dividing by one of those turns a difference of a single
    ulp into an enormous ratio while saying nothing about the measurement.
    """
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:  # pragma: no cover - the fixture field carries energy
        raise AssertionError("the reference leg has no scale to compare against")
    return float(np.max(np.abs(np.asarray(observed) - np.asarray(reference)))) / scale


@pytest.mark.parametrize("normalization", PSF_NORMALIZATIONS)
@pytest.mark.parametrize("cell", [cell.param for cell in CELLS])
def test_psf_agrees_across_cells(cell: Any, normalization: str) -> None:
    """AC-5: the four reported quantities agree, in every cell, at a derived tolerance.

    `raw_window_energy` is compared at the reduction's own accumulation length
    (`ny * nx`) while the intensity map is compared elementwise, because they
    are different kernels reported by one call: one is a sum over the window and
    the other is not. Handing both the same tolerance would either overtighten
    the sum or loosen the map by the same factor.
    """
    reason = unavailable_reason(cell)
    if reason is not None:
        pytest.skip(reason)
    field = _field(cell)
    result = psf(field, normalization=normalization)
    reference = _reference(cell.dtype, normalization)

    # The measurement must not have moved the data off the cell. `psf` returns
    # `|u|^2`, so the dtype it lands in is the real member of the cell's
    # precision family, not the cell's own complex dtype.
    real_dtype = cell.dtype.precision.real_dtype
    verify_placement(cell, result.intensity, dtype=real_dtype)

    ny, nx = _SHAPE
    window = ny * nx
    elementwise = tolerance_for(cell, accumulation_length=1, matmul=False)
    reduced = tolerance_for(cell, accumulation_length=window, matmul=False)

    # The peak index is an integer answer to an unambiguous question. A cell
    # that disagrees here has found a different maximum, which no tolerance
    # makes acceptable.
    assert result.peak_index == reference.peak_index, (
        f"{cell} put the peak at {result.peak_index}, the host leg at {reference.peak_index}"
    )

    assert _relative(result.raw_peak_intensity, reference.raw_peak_intensity) <= elementwise, (
        f"{cell} raw_peak_intensity {result.raw_peak_intensity!r} vs host "
        f"{reference.raw_peak_intensity!r}"
    )
    assert _relative(result.raw_window_energy, reference.raw_window_energy) <= reduced, (
        f"{cell} raw_window_energy {result.raw_window_energy!r} vs host "
        f"{reference.raw_window_energy!r}"
    )

    # The energy normalization divides by the window sum, so its map inherits
    # that reduction's error; `raw` and `peak` do not.
    map_tolerance = reduced if normalization == "energy" else elementwise
    assert _relative(_host(result.intensity), _host(reference.intensity)) <= map_tolerance, (
        f"{cell} intensity map disagrees with the host leg under {normalization!r}"
    )


def test_the_subjects_dtype_axis_is_the_declared_floor() -> None:
    """Why this subject has one dtype per namespace rather than a matrix of them.

    `M_PSF` carries `capabilities=None`, so nothing has measured which dtypes it
    supports, and `cells_for` refuses to invent rows. Asserted rather than
    narrated because the alternative -- someone adding a complex128 cell here to
    'improve coverage' -- would fail in JAX for a reason unrelated to the PSF:
    `jax_enable_x64` is pinned off on Chromatix import, so the request would be
    refused by `verify_dtype`, and the fix would look like a tolerance problem.
    """
    assert {cell.dtype for cell in CELLS} == {DType.COMPLEX64}
    assert len(CELLS) == 3, [str(cell) for cell in CELLS]
