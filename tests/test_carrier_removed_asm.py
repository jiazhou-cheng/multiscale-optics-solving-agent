"""CHE-40 (M3.2A) — carrier-removed exact ASM, and the sensitivity it removes.

M3.2 concluded that Chromatix's `complex64` cast bounds the usable propagation
distance, and rejected a 48 mm reference singlet on that basis. M3.2A found the
error was representational: the transfer function was carrying a ~5.4e5 rad
absolute carrier phase that contributes nothing to a single-path PSF.

Three things have to stay true, and each of these tests pins one of them:

1. **Carrier removal is a rewrite, not new physics.** If someone "simplifies"
   the identity `k_z - k = -(k_x^2+k_y^2)/(k_z+k)` into a paraxial expansion, the
   float64 equivalence test fails. That is the whole safety of the change.
2. **The conditioning gain is real and large.** A regression that reintroduced
   absolute-carrier sensitivity -- say by computing `k_z - k` as a subtraction,
   or by folding `exp(i k z)` back into the complex64 field for convenience --
   would pass a smoke test and quietly restore a 1e-2 error at 47 mm.
3. **Chromatix's conventions are untouched.** Only the transfer function differs;
   FFT ordering, padding, and the evanescent policy are Chromatix's own.

The distances here are the ticket's, not convenient ones. 47.06 mm is the
pupil-to-focus distance of the system M3.2 rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from multiscale_optics_agent.evaluation.asm_oracle import (
    CarrierConvention,
    absolute_phase_representation_floor,
    angular_spectrum_float64,
    compare_fields,
    evanescent_bin_count,
    relative_phase_excursion_rad,
)

WAVELENGTH_M = 5.5e-7
GRID = 64
SAMPLE_PITCH_M = 4.0e-6
REJECTED_SYSTEM_DISTANCE_M = 47.06e-3

pytest.importorskip("chromatix", reason="carrier-removed ASM is a Chromatix path")


@pytest.fixture(autouse=True)
def _pinned_wave_engine_precision():
    """Pin `jax_enable_x64` off before every test in this module.

    `sax` turns x64 on as an import side effect that Python will not re-trigger,
    so in a full-suite run these tests would otherwise inherit complex128 FFTs
    from whichever module imported first and measure an engine M3 does not
    declare. `chromatix_adapter` has carried the same defence since M1; this
    module needs it too because it builds `Field`s directly.
    """
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        pin_wave_engine_precision,
    )

    pin_wave_engine_precision()
    yield


def _converging_wave(focal_m: float, *, grid: int = GRID, pitch_m: float = SAMPLE_PITCH_M):
    coords = (np.arange(grid) - grid // 2) * pitch_m
    x, y = np.meshgrid(coords, coords, indexing="xy")
    aperture = (x**2 + y**2) <= (0.4 * grid * pitch_m / 2.0) ** 2
    phase = -2.0 * np.pi / WAVELENGTH_M * np.sqrt(x**2 + y**2 + focal_m**2)
    return (aperture * np.exp(1j * phase)).astype(np.complex128)


def _chromatix_field(u: np.ndarray, pitch_m: float = SAMPLE_PITCH_M):
    import jax.numpy as jnp
    from chromatix import functional as cf

    return cf.Field.build(
        jnp.asarray(u, dtype=jnp.complex64),
        jnp.asarray([[pitch_m, pitch_m]]),
        WAVELENGTH_M,
    )


def _absolute_path(u: np.ndarray, z_m: float, pitch_m: float = SAMPLE_PITCH_M) -> np.ndarray:
    from chromatix import functional as cf

    out = cf.asm_propagate(_chromatix_field(u, pitch_m), z=z_m, n=1.0, pad_width=0)
    return np.asarray(out.u, dtype=np.complex128).reshape(u.shape)


def _carrier_removed_path(u: np.ndarray, z_m: float, pitch_m: float = SAMPLE_PITCH_M):
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagate,
    )

    result = carrier_removed_asm_propagate(
        _chromatix_field(u, pitch_m),
        z_m=z_m,
        refractive_index=1.0,
        pad_width=0,
        wavelength_m=WAVELENGTH_M,
    )
    return np.asarray(result.field.u, dtype=np.complex128).reshape(u.shape), result


# ---------------------------------------------------------------------------
# AC1 -- the rewrite is exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_mm", [0.04, 0.4, 4.0, 47.06])
def test_float64_carrier_conventions_are_the_same_propagation(z_mm: float) -> None:
    """The two forms differ by exactly `exp(i k z)` -- nothing else.

    Measured against `eps64 * k z` rather than a flat 1e-12, because representing
    the absolute carrier at all costs that much: at 47 mm it is 1.2e-10, so a flat
    1e-12 is unreachable there *for the reason this ticket exists*. A paraxial
    substitution for the identity would miss by ~1e-6 and fail either way.
    """
    z_m = z_mm * 1e-3
    u = _converging_wave(2.0e-3)
    absolute = angular_spectrum_float64(
        u,
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=SAMPLE_PITCH_M,
        z_m=z_m,
        carrier=CarrierConvention.ABSOLUTE,
    )
    removed = angular_spectrum_float64(
        u,
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=SAMPLE_PITCH_M,
        z_m=z_m,
        carrier=CarrierConvention.CARRIER_REMOVED,
    )

    comparison = compare_fields(removed, absolute)
    floor = absolute_phase_representation_floor(wavelength_m=WAVELENGTH_M, z_m=z_m)
    assert comparison.piston_aligned_relative_field_error <= 10.0 * floor
    # Intensity is piston-blind, but not blind to the *per-bin* float64 rounding of
    # `k z * delay`, which is not a piston. So it is bounded by the same floor
    # rather than by a flat number.
    assert comparison.relative_intensity_l2_error <= 10.0 * floor


def test_the_removed_factor_is_exactly_the_carrier() -> None:
    """`H_absolute = exp(i k z) * H_relative`, bin by bin, not just in L2."""
    z_m = REJECTED_SYSTEM_DISTANCE_M
    impulse = np.zeros((GRID, GRID), dtype=np.complex128)
    impulse[0, 0] = 1.0  # its spectrum is flat, so this exposes every bin

    absolute = np.fft.fft2(
        angular_spectrum_float64(
            impulse,
            wavelength_m=WAVELENGTH_M,
            sample_pitch_m=SAMPLE_PITCH_M,
            z_m=z_m,
            carrier=CarrierConvention.ABSOLUTE,
        )
    )
    removed = np.fft.fft2(
        angular_spectrum_float64(
            impulse,
            wavelength_m=WAVELENGTH_M,
            sample_pitch_m=SAMPLE_PITCH_M,
            z_m=z_m,
            carrier=CarrierConvention.CARRIER_REMOVED,
        )
    )
    propagating = np.abs(removed) > 1e-12
    ratio = absolute[propagating] / removed[propagating]
    expected = np.exp(1j * 2.0 * np.pi * z_m / WAVELENGTH_M)

    assert np.allclose(np.abs(ratio), 1.0, atol=1e-9)
    assert np.max(np.abs(np.angle(ratio * np.conj(expected)))) < 1e-6


def test_identity_is_not_a_paraxial_approximation() -> None:
    """A paraxial kernel would agree at small angles and diverge at the band edge.

    The guard that would catch someone replacing `-(f^2)/(delay+1)` with the
    cheaper `-(f^2)/2`: at this grid's edge the two differ by ~0.3% of the
    relative phase, which is 8 rad at 47 mm.
    """
    z_m = REJECTED_SYSTEM_DISTANCE_M
    frequency = 1.0 / (2.0 * SAMPLE_PITCH_M)  # the band edge, one axis
    exact_delay = np.sqrt(1.0 - (WAVELENGTH_M * frequency) ** 2)
    exact = 2.0 * np.pi * z_m * WAVELENGTH_M * frequency**2 / (exact_delay + 1.0)
    paraxial = 2.0 * np.pi * z_m * WAVELENGTH_M * frequency**2 / 2.0

    assert abs(exact - paraxial) > 1.0, "the paraxial form must be distinguishable here"

    # And the oracle must land on the exact side of that difference. The helper
    # reports the corner bin, where both axes sit at the band edge, so its value
    # must exceed the single-axis figure computed above.
    measured = relative_phase_excursion_rad(
        (GRID, GRID),
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=SAMPLE_PITCH_M,
        z_m=z_m,
    )
    corner_delay = np.sqrt(1.0 - 2.0 * (WAVELENGTH_M * frequency) ** 2)
    corner_exact = 2.0 * np.pi * z_m * WAVELENGTH_M * 2.0 * frequency**2 / (corner_delay + 1.0)
    assert measured == pytest.approx(corner_exact, rel=1e-12)
    assert measured > exact


# ---------------------------------------------------------------------------
# AC3/AC4 -- the conditioning gain, and the sensitivity that must not return
# ---------------------------------------------------------------------------
def test_carrier_removal_beats_the_absolute_path_at_the_rejected_distance() -> None:
    """AC3: at least 10x on piston-aligned field error. Measured ~1000x.

    The assertion is at 50x rather than at the measured value: tight enough that
    a regression to the absolute-phase behaviour fails it by two orders of
    magnitude, loose enough not to encode one machine's rounding.
    """
    z_m = REJECTED_SYSTEM_DISTANCE_M
    u = _converging_wave(2.0e-3)
    reference = angular_spectrum_float64(
        u, wavelength_m=WAVELENGTH_M, sample_pitch_m=SAMPLE_PITCH_M, z_m=z_m
    )

    absolute = compare_fields(_absolute_path(u, z_m), reference)
    removed_field, _ = _carrier_removed_path(u, z_m)
    removed = compare_fields(removed_field, reference)

    improvement = (
        absolute.piston_aligned_relative_field_error / removed.piston_aligned_relative_field_error
    )
    assert improvement >= 50.0, f"carrier removal only improved by {improvement:.1f}x"


def test_intensity_error_stays_inside_the_m3_budget_at_47mm() -> None:
    """AC4: relative intensity L2 <= 1e-3, preferred <= 3.5e-4, at 47 mm."""
    z_m = REJECTED_SYSTEM_DISTANCE_M
    u = _converging_wave(2.0e-3)
    reference = angular_spectrum_float64(
        u, wavelength_m=WAVELENGTH_M, sample_pitch_m=SAMPLE_PITCH_M, z_m=z_m
    )
    removed_field, _ = _carrier_removed_path(u, z_m)
    intensity_error = compare_fields(removed_field, reference).relative_intensity_l2_error

    assert intensity_error <= 1.0e-3
    assert intensity_error <= 3.5e-4, "the M3.2 intensity budget term must still hold"


def test_error_no_longer_tracks_absolute_propagation_distance() -> None:
    """The regression guard the ticket asks for, stated as a ratio between distances.

    Between 0.4 mm and 47.06 mm the absolute carrier grows 118x. If the
    carrier-removed error grew with it, this fails. It is allowed to grow with the
    *relative* phase excursion -- that is the quantity it now represents, and it
    grows over the same span by the same 118x, so the assertion is stated against
    the absolute-carrier growth it must NOT follow, with margin for the relative
    growth it may.
    """
    u = _converging_wave(2.0e-3)
    errors = {}
    for z_m in (0.4e-3, REJECTED_SYSTEM_DISTANCE_M):
        reference = angular_spectrum_float64(
            u, wavelength_m=WAVELENGTH_M, sample_pitch_m=SAMPLE_PITCH_M, z_m=z_m
        )
        absolute = compare_fields(_absolute_path(u, z_m), reference)
        removed_field, _ = _carrier_removed_path(u, z_m)
        errors[z_m] = (
            absolute.piston_aligned_relative_field_error,
            compare_fields(removed_field, reference).piston_aligned_relative_field_error,
        )

    absolute_growth = errors[REJECTED_SYSTEM_DISTANCE_M][0] / errors[0.4e-3][0]
    removed_growth = errors[REJECTED_SYSTEM_DISTANCE_M][1] / errors[0.4e-3][1]
    carrier_growth = REJECTED_SYSTEM_DISTANCE_M / 0.4e-3

    assert absolute_growth > 0.1 * carrier_growth, (
        "the baseline path is supposed to track the carrier; if it no longer does, "
        "the premise of this ticket changed and the comparison needs revisiting"
    )
    assert removed_growth < absolute_growth
    # The absolute path stays two orders of magnitude worse at the far end.
    assert errors[REJECTED_SYSTEM_DISTANCE_M][1] < 0.02 * errors[REJECTED_SYSTEM_DISTANCE_M][0]


# ---------------------------------------------------------------------------
# Conventions must not drift
# ---------------------------------------------------------------------------
def test_propagator_matches_chromatix_up_to_the_carrier_in_float64() -> None:
    """Same grid, same ordering, same evanescent policy -- only the phase differs.

    Evaluated on the propagator arrays rather than on propagated fields, so an
    `ifftshift` that went missing shows up as a rearrangement instead of hiding
    inside an L2 norm. The comparison is against a float64 recomputation of
    Chromatix's own kernel, because Chromatix's is complex64 and would contribute
    the error being measured.
    """
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagator,
    )

    z_m = 1.0e-3
    field = _chromatix_field(np.ones((GRID, GRID), dtype=np.complex128))
    removed = np.asarray(carrier_removed_asm_propagator(field, z_m, 1.0), dtype=np.complex128)

    frequency = np.fft.fftfreq(GRID, d=SAMPLE_PITCH_M)
    fx, fy = np.meshgrid(frequency, frequency, indexing="xy")
    delay = np.sqrt(1.0 - WAVELENGTH_M**2 * (fx**2 + fy**2))
    expected = np.exp(-1j * 2.0 * np.pi * z_m * WAVELENGTH_M * (fx**2 + fy**2) / (delay + 1.0))

    # complex64 evaluation of a ~123 rad phase, so eps32 * phase is the yardstick.
    assert np.max(np.abs(np.angle(removed * np.conj(expected)))) < 1e-3


def test_evanescent_orders_decay_rather_than_being_zeroed() -> None:
    """Chromatix's default policy, preserved. A `maximum(kernel, 0)` would zero them.

    Needs a pitch below `lambda / 2` for evanescent bins to exist at all; the M3
    grids have none, which is why this is tested deliberately rather than assumed
    from the sweep.
    """
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagator,
    )

    fine_pitch_m = WAVELENGTH_M / 4.0  # Nyquist frequency at 2/lambda -> evanescent bins
    assert (
        evanescent_bin_count((GRID, GRID), wavelength_m=WAVELENGTH_M, sample_pitch_m=fine_pitch_m)
        > 0
    )

    field = _chromatix_field(np.ones((GRID, GRID), dtype=np.complex128), fine_pitch_m)
    kernel = np.asarray(
        carrier_removed_asm_propagator(field, WAVELENGTH_M, 1.0), dtype=np.complex128
    )

    frequency = np.fft.fftfreq(GRID, d=fine_pitch_m)
    fx, fy = np.meshgrid(frequency, frequency, indexing="xy")
    evanescent = (WAVELENGTH_M * np.hypot(fx, fy)) > 1.0

    magnitudes = np.abs(kernel[evanescent])
    assert np.all(magnitudes < 1.0), "evanescent orders must decay"
    assert np.any(magnitudes > 0.0), "decaying is not the same as zeroing"


def test_padding_is_chromatix_padding() -> None:
    """`pad_width` must reach Chromatix's own `pad`, and `mode='same'` must crop."""
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagate,
    )

    u = _converging_wave(2.0e-3)
    field = _chromatix_field(u)

    full = carrier_removed_asm_propagate(field, z_m=1.0e-3, pad_width=16)
    same = carrier_removed_asm_propagate(field, z_m=1.0e-3, pad_width=16, mode="same")

    assert full.field.u.shape[-2:] == (GRID + 32, GRID + 32)
    assert same.field.u.shape[-2:] == (GRID, GRID)


# ---------------------------------------------------------------------------
# AC6 -- the global-phase policy is explicit, and machine-readable
# ---------------------------------------------------------------------------
def test_removed_carrier_is_reported_and_not_silently_reapplied() -> None:
    """A consumer must be able to see that absolute phase is missing, and recover it."""
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        GLOBAL_PHASE_POLICY,
        reconstruct_absolute_phase,
    )

    z_m = REJECTED_SYSTEM_DISTANCE_M
    u = _converging_wave(2.0e-3)
    removed_field, result = _carrier_removed_path(u, z_m)

    assert result.global_phase_policy == GLOBAL_PHASE_POLICY
    assert result.absolute_phase_is_physical is False
    assert result.wavelength_source == "caller (float64)"
    assert result.removed_carrier_phase_rad == pytest.approx(
        2.0 * np.pi * z_m / WAVELENGTH_M, rel=1e-12
    )

    # The carrier is genuinely absent from the field, not folded back in.
    reference = angular_spectrum_float64(
        u, wavelength_m=WAVELENGTH_M, sample_pitch_m=SAMPLE_PITCH_M, z_m=z_m
    )
    raw = compare_fields(removed_field, reference).raw_relative_field_error
    aligned = compare_fields(removed_field, reference).piston_aligned_relative_field_error
    assert raw > 100.0 * aligned, "the field must still carry the discarded piston"

    # And reconstruction restores absolute phase, in float64.
    absolute_phase = reconstruct_absolute_phase(result)
    expected = np.angle(np.asarray(removed_field)) + result.removed_carrier_phase_rad
    assert np.allclose(absolute_phase, expected)


def test_field_derived_wavelength_is_float32_and_says_so() -> None:
    """The trap: `Field.spectrum` is float32, so a carrier read off the field is too.

    ~3e-8 relative is 0.018 rad of absolute phase at 47 mm -- larger than
    everything carrier removal buys back. The point of this test is not that the
    float32 route is wrong to offer, but that it must be labelled, so a consumer
    reconstructing absolute phase can see what it is standing on.
    """
    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagate,
    )

    z_m = REJECTED_SYSTEM_DISTANCE_M
    field = _chromatix_field(_converging_wave(2.0e-3))
    exact = 2.0 * np.pi * z_m / WAVELENGTH_M

    from_field = carrier_removed_asm_propagate(field, z_m=z_m)
    from_caller = carrier_removed_asm_propagate(field, z_m=z_m, wavelength_m=WAVELENGTH_M)

    assert from_field.wavelength_source.startswith("chromatix Field.spectrum")
    assert from_caller.wavelength_source == "caller (float64)"
    assert from_caller.removed_carrier_phase_rad == pytest.approx(exact, rel=1e-12)

    field_error_rad = abs(from_field.removed_carrier_phase_rad - exact)
    assert field_error_rad > 1e-3, (
        "if this ever drops, Chromatix started storing the spectrum at higher "
        "precision and the wavelength_m escape hatch can be reconsidered"
    )
    assert field_error_rad < 1.0  # still a float32 rounding, not a unit error


def test_propagation_pins_jax_x64_off_regardless_of_ambient_state() -> None:
    """The measurements must not depend on whether `sax` was imported first.

    Found by CHE-40 the expensive way: in isolation this module's tests passed,
    and in a full-suite run one failed, because `sax` had turned `jax_enable_x64`
    on and Chromatix's FFTs quietly promoted to complex128. An error figure that
    depends on unrelated import order is not evidence.
    """
    import jax

    from multiscale_optics_agent.adapters.chromatix_carrier_removed import (
        carrier_removed_asm_propagate,
    )

    jax.config.update("jax_enable_x64", True)
    try:
        # The Field must be built after the pin, which is what a caller has to do
        # too -- the propagate call cannot downcast a field that already exists.
        carrier_removed_asm_propagate(_chromatix_field(_converging_wave(2.0e-3)), z_m=1.0e-3)
        assert jax.config.jax_enable_x64 is False
    finally:
        jax.config.update("jax_enable_x64", False)


def test_negative_distance_conjugates_like_chromatix() -> None:
    """Back-propagation must invert forward propagation, not silently no-op."""
    z_m = 2.0e-3
    u = _converging_wave(2.0e-3)
    forward, _ = _carrier_removed_path(u, z_m)
    round_trip, _ = _carrier_removed_path(forward, -z_m)

    # complex64 round trip over a modest distance; the tolerance is the engine's.
    assert compare_fields(round_trip, u).piston_aligned_relative_field_error < 1e-5
