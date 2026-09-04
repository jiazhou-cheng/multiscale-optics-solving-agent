"""Subject 1: `M_PSF` agrees across numpy-cpu, jax-cpu and jax-cuda.

**The field is built by hand, and that is a finding rather than a convenience.**
`sources.plane_wave` and its two siblings are hardcoded NumPy with no device or
namespace parameter (`src/sources/_grid.py`), so no source in this project can
construct a field on a device. Every cell here therefore assembles its own
`ScalarField` from host data through `place`. **CHE-246 (T2) is the ticket that
fixes it, and replacing the hand construction below with `sources.plane_wave`
while these tests keep passing is T2's completion signal.**

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
from parity.cells import cells_for, tolerance_for
from parity.conftest import verify_placement
from representations.geometry import ReferenceSurface
from representations.scalar import ScalarField

COMPONENT = "M_PSF"
CELLS = cells_for(COMPONENT, complex_data=True)

_SHAPE = (32, 32)
_PITCH_M = (2.0e-6, 2.0e-6)
_WAVELENGTH_M = 550.0e-9
_SURFACE = ReferenceSurface(name="image_surface", z_m=0.0, medium_index=1.0)


def _amplitude() -> Any:
    """A deterministic complex field with a unique peak and a non-trivial phase.

    The tilt matters: without it every cell would agree on a real, separable
    array and the complex arithmetic `tolerance_for` accounts for would never
    happen. The Gaussian envelope gives `peak_index` a single unambiguous
    answer, so the index comparison below can be exact rather than tolerant.
    """
    ny, nx = _SHAPE
    y = (np.arange(ny, dtype=np.float64) - ny // 2)[:, None]
    x = (np.arange(nx, dtype=np.float64) - nx // 2)[None, :]
    envelope = np.exp(-(y**2 + x**2) / (2.0 * 6.0**2))
    phase = 0.37 * y + 0.11 * x
    return (envelope * np.exp(1j * phase)).astype(np.complex128)


_AMPLITUDE = _amplitude()


def _field(u: Any) -> ScalarField:
    return ScalarField(
        u=u,
        sample_pitch_m=_PITCH_M,
        wavelength_m=_WAVELENGTH_M,
        reference_surface=_SURFACE,
    )


def _reference(dtype: DType, normalization: str) -> Any:
    """The host leg, **at the cell's own dtype**. Characterization, not an oracle.

    Deliberately not a complex128 reference. Casting the same complex128 source
    to complex64 costs about one ulp of the amplitude, which becomes two ulp of
    `|u|^2`, and comparing a complex64 cell against a complex128 leg would fold
    that input quantization into a number `tolerance_for` derives for
    *arithmetic* -- so the gate would be measuring the cast and would need
    widening for a reason that has nothing to do with where the kernel ran.
    Every cell casting from the same source at the same dtype gets bit-identical
    inputs, and what is left over is exactly the namespace and device difference
    this package exists to see.

    It follows that the numpy-cpu cell compares against itself. That comparison
    is not vacuous but it is weak -- it says the placement path is a no-op on
    the host, which is what `_host`-style code paths are supposed to be -- and
    the asymmetry is the point of `tests/parity/__init__.py`'s note that the
    host leg is characterization.
    """
    u = to_namespace(_AMPLITUDE, namespace=ArrayNamespace.NUMPY, dtype=dtype)
    return psf(_field(u), normalization=normalization)


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
def test_psf_agrees_across_cells(cell: Any, normalization: str, place: Any) -> None:
    """AC-5: the four reported quantities agree, in every cell, at a derived tolerance.

    `raw_window_energy` is compared at the reduction's own accumulation length
    (`ny * nx`) while the intensity map is compared elementwise, because they
    are different kernels reported by one call: one is a sum over the window and
    the other is not. Handing both the same tolerance would either overtighten
    the sum or loosen the map by the same factor.
    """
    field = _field(place(cell, _AMPLITUDE))
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
