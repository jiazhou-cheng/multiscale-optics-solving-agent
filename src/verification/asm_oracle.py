"""Independent float64 angular-spectrum oracle, and the metrics used against it.

CHE-40 (M3.2A). M3.2 measured Chromatix's ``complex64`` cast against a float64
angular-spectrum reference written inline in
``benchmarks/probes/slice_feasibility.py``. M3.2A needs the same reference
from a probe *and* from tests, and needs it in two algebraically identical
forms, so it is factored here rather than copied.

**This is a reference, not an oracle in the analytic sense.** It is an
independent implementation, deliberately not Chromatix: pure NumPy, float64
throughout, so it does not share the dtype behaviour it is used to measure. It
cannot certify the physics -- only an analytic case can do that, and M1 already
did it for this engine at 40 um. What it can do is separate *implementation*
error from *representation* error, which is the whole question in M3.2A.

Two carrier conventions are provided, and they are exact algebraic rewrites of
each other, not two approximations:

``ABSOLUTE``
    ``H = exp(i z k_z)`` with ``k_z = sqrt(k^2 - k_x^2 - k_y^2)``. The form
    Chromatix evaluates. Its phase magnitude is set by ``k z = 2*pi*n*z/lambda``,
    which at 47 mm and 550 nm is ~5.4e5 rad.

``CARRIER_REMOVED``
    ``H_rel = exp(i z (k_z - k))``, evaluated through the exact identity
    ``k_z - k = -(k_x^2 + k_y^2) / (k_z + k)`` so that no cancellation occurs
    when ``k_z -> k``. The removed factor ``exp(i k z)`` is a global piston: it
    is constant over the whole spectrum, so it cannot change intensity, and for
    a single propagation path it cannot change relative phase either.

The two differ by exactly ``exp(i k z)``, which is why the piston-aligned field
error between them is a pure round-off measurement. In float64 that round-off
is not zero: representing a phase of ``k z`` costs ``eps64 * k z``, ~1.2e-10 at
47 mm. That is the same mechanism the float32 path suffers, nine orders of
magnitude down, and :func:`absolute_phase_representation_floor` states it.

Evanescent policy: components with ``k_z`` imaginary are **zeroed**, which is
what the M3.2 reference did. The policy is preserved rather than improved
because M3.2A compares against M3.2's own numbers, and changing the oracle
mid-experiment would invalidate that comparison. Chromatix instead lets them
decay. :func:`evanescent_bin_count` exists so a caller can report that the two
policies cannot differ on its grid, which is the case whenever the sample pitch
is coarser than ``lambda / 2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "ASM_ORACLE_ID",
    "CarrierConvention",
    "FieldComparison",
    "absolute_phase_representation_floor",
    "angular_spectrum_float64",
    "compare_fields",
    "evanescent_bin_count",
    "relative_phase_excursion_rad",
]

ASM_ORACLE_ID = "ASM-FLOAT64-V1"

_EPS64 = float(np.finfo(np.float64).eps)


class CarrierConvention(StrEnum):
    """Which form of the exact ASM transfer function to evaluate."""

    ABSOLUTE = "absolute"
    CARRIER_REMOVED = "carrier_removed"


def _frequency_grids(
    shape: tuple[int, int], sample_pitch_m: float | tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial-frequency grids in cycles/m, in NumPy's natural FFT order.

    Natural (unshifted) order, matching ``np.fft.fft2`` output directly. This is
    the ordering the M3.2 reference used; Chromatix builds a centred grid and
    then ``ifftshift``s the kernel, arriving at the same place by a different
    route.
    """
    n_y, n_x = shape
    pitch_y, pitch_x = _pitch_pair(sample_pitch_m)
    frequency_y = np.fft.fftfreq(n_y, d=pitch_y)
    frequency_x = np.fft.fftfreq(n_x, d=pitch_x)
    return np.meshgrid(frequency_x, frequency_y, indexing="xy")


def _pitch_pair(sample_pitch_m: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(sample_pitch_m, (int, float)):
        return float(sample_pitch_m), float(sample_pitch_m)
    pitch_y, pitch_x = sample_pitch_m
    return float(pitch_y), float(pitch_x)


def angular_spectrum_float64(
    u: np.ndarray,
    *,
    wavelength_m: float,
    sample_pitch_m: float | tuple[float, float],
    z_m: float,
    refractive_index: float = 1.0,
    carrier: CarrierConvention = CarrierConvention.ABSOLUTE,
) -> np.ndarray:
    """Exact (non-paraxial) angular-spectrum propagation in float64, no padding.

    Args:
        u: input complex field, ``(n_y, n_x)``.
        wavelength_m: vacuum wavelength.
        sample_pitch_m: scalar, or ``(pitch_y, pitch_x)``.
        z_m: propagation distance. Negative distances conjugate the transfer
            function, matching Chromatix.
        refractive_index: isotropic index of the propagation medium.
        carrier: ``ABSOLUTE`` reproduces the M3.2 reference exactly.
            ``CARRIER_REMOVED`` omits the global ``exp(i k z)`` factor, so its
            output carries physically meaningful *relative* phase and an
            arbitrary global piston.

    Returns:
        The propagated field, ``complex128``.
    """
    field = np.asarray(u, dtype=np.complex128)
    if field.ndim != 2:
        raise ValueError(f"expected a 2-D field, got shape {field.shape}")

    frequency_x, frequency_y = _frequency_grids(field.shape, sample_pitch_m)
    frequency_squared = frequency_x**2 + frequency_y**2

    index_over_wavelength = refractive_index / wavelength_m
    argument = index_over_wavelength**2 - frequency_squared
    propagating = argument > 0.0
    # sqrt of the clipped argument, so the zeroed evanescent bins never produce
    # a NaN that a later `where` would have to launder.
    delay = np.sqrt(np.where(propagating, argument, 0.0)) / index_over_wavelength

    if carrier is CarrierConvention.ABSOLUTE:
        phase = 2.0 * np.pi * index_over_wavelength * abs(z_m) * delay
    else:
        # k_z - k = -(k_x^2+k_y^2)/(k_z+k), exact; no cancellation as k_z -> k.
        phase = (
            -2.0
            * np.pi
            * abs(z_m)
            * wavelength_m
            * frequency_squared
            / (refractive_index * (delay + 1.0))
        )
    transfer = np.where(propagating, np.exp(1j * phase), 0.0)
    if z_m < 0.0:
        transfer = np.conj(transfer)
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def evanescent_bin_count(
    shape: tuple[int, int],
    *,
    wavelength_m: float,
    sample_pitch_m: float | tuple[float, float],
    refractive_index: float = 1.0,
) -> int:
    """How many spectral bins are evanescent on this grid.

    Zero means the oracle's zeroing policy and Chromatix's decay policy are
    indistinguishable here, so a comparison between them is not contaminated by
    the difference. Reported rather than assumed.
    """
    frequency_x, frequency_y = _frequency_grids(shape, sample_pitch_m)
    argument = (refractive_index / wavelength_m) ** 2 - frequency_x**2 - frequency_y**2
    return int(np.count_nonzero(argument <= 0.0))


def relative_phase_excursion_rad(
    shape: tuple[int, int],
    *,
    wavelength_m: float,
    sample_pitch_m: float | tuple[float, float],
    z_m: float,
    refractive_index: float = 1.0,
) -> float:
    """``max |z (k_z - k)|`` over the propagating bins of this grid.

    The phase magnitude the carrier-removed transfer function actually has to
    represent, as opposed to ``k z``, which is what the absolute form has to
    represent. The ratio between the two is the conditioning improvement
    available before any of it is measured.
    """
    frequency_x, frequency_y = _frequency_grids(shape, sample_pitch_m)
    frequency_squared = frequency_x**2 + frequency_y**2
    index_over_wavelength = refractive_index / wavelength_m
    argument = index_over_wavelength**2 - frequency_squared
    propagating = argument > 0.0
    delay = np.sqrt(np.where(propagating, argument, 0.0)) / index_over_wavelength
    excursion = (
        2.0
        * np.pi
        * abs(z_m)
        * wavelength_m
        * frequency_squared
        / (refractive_index * (delay + 1.0))
    )
    return float(np.max(np.where(propagating, excursion, 0.0)))


def absolute_phase_representation_floor(
    *, wavelength_m: float, z_m: float, refractive_index: float = 1.0
) -> float:
    """``eps64 * k z`` -- the float64 cost of carrying the absolute carrier.

    A float64 absolute-phase ASM cannot agree with a float64 carrier-removed ASM
    more closely than this, because ``k z`` itself is not exactly representable.
    At 47 mm and 550 nm it is ~1.2e-10, which is why M3.2A's AC1 is stated
    against this floor rather than against a flat 1e-12: the flat number is
    unreachable at that distance *for the reason the ticket is investigating*.
    """
    return _EPS64 * 2.0 * np.pi * refractive_index * abs(z_m) / wavelength_m


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One test field measured against one float64 reference field.

    Four numbers because they answer different questions and routinely differ by
    orders of magnitude:

    - ``raw_relative_field_error`` includes any global piston, so it is what a
      consumer of absolute optical phase would pay.
    - ``piston_aligned_relative_field_error`` removes the best global phase
      first, so it isolates *spatially varying* wavefront error. This is the
      field-level metric that says whether an error is physically meaningful.
    - ``relative_intensity_l2_error`` is what a PSF consumer pays.
    - ``energy_residual`` catches a transfer function that has stopped being
      unit-modulus, which neither of the above cleanly separates.
    """

    raw_relative_field_error: float
    piston_aligned_relative_field_error: float
    piston_rad: float
    relative_intensity_l2_error: float
    energy_residual: float
    piston_fraction_of_raw_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compare_fields(test: np.ndarray, reference: np.ndarray) -> FieldComparison:
    """Metrics of ``test`` against a float64 ``reference``.

    The optimal global phase is ``alpha = arg <reference, test>``: the value that
    maximises ``Re(e^{-i alpha} <reference, test>)`` and therefore minimises
    ``||test e^{-i alpha} - reference||``. It is the exact minimiser, not a
    search.
    """
    test_field = np.asarray(test, dtype=np.complex128)
    reference_field = np.asarray(reference, dtype=np.complex128)
    if test_field.shape != reference_field.shape:
        raise ValueError(
            f"shape mismatch: test {test_field.shape} vs reference {reference_field.shape}"
        )

    reference_norm = float(np.linalg.norm(reference_field))
    if reference_norm == 0.0:
        raise ValueError("reference field has zero norm; no relative error is defined")

    raw = float(np.linalg.norm(test_field - reference_field)) / reference_norm
    piston = float(np.angle(np.vdot(reference_field, test_field)))
    aligned = (
        float(np.linalg.norm(test_field * np.exp(-1j * piston) - reference_field)) / reference_norm
    )

    test_intensity = np.abs(test_field) ** 2
    reference_intensity = np.abs(reference_field) ** 2
    intensity = float(
        np.linalg.norm(test_intensity - reference_intensity) / np.linalg.norm(reference_intensity)
    )
    reference_energy = float(np.sum(reference_intensity))
    energy = abs(float(np.sum(test_intensity)) - reference_energy) / reference_energy

    return FieldComparison(
        raw_relative_field_error=raw,
        piston_aligned_relative_field_error=aligned,
        piston_rad=piston,
        relative_intensity_l2_error=intensity,
        energy_residual=energy,
        piston_fraction_of_raw_error=(1.0 - aligned / raw) if raw > 0.0 else 0.0,
    )
