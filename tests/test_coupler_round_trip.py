"""CHE-26 — the two directions tested against each other.

Each direction verified alone can still be jointly inconsistent: a shared sign
or normalization error cancels in one direction and reappears in the other, or
worse, cancels in both and is never seen. So the round trip is only worth
running if it can actually fail, which is why the deliberately mismatched
pairing below is a required test rather than a nicety.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multiscale_optics_agent.couplers.cascade import planar_doe_step
from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    ContractError,
    ReferencePlane,
)
from multiscale_optics_agent.couplers.ray_to_wave import (
    Perturbation,
    Projection,
    ray_to_wave,
)
from multiscale_optics_agent.couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    draw_indices,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N_GRID = 16
GRID = (N_GRID, N_GRID)
PITCH = (PITCH_M, PITCH_M)
PLANE = ReferencePlane(name="cascade plane", z_m=0.0)


def _field(u: np.ndarray) -> ComplexField:
    return ComplexField(
        u=u.astype(np.complex128),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )


def _random_field(seed: int = 20260812) -> ComplexField:
    rng = np.random.default_rng(seed)
    return _field(rng.normal(size=GRID) + 1j * rng.normal(size=GRID))


def _reconstruct(bundle, projection=Projection.ASM_CONSISTENT, perturbation=None):
    field, _ = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        normalization="one_over_n",
        projection=projection,
        **({"perturbation": perturbation} if perturbation else {}),
    )
    return field.u


def _relative_rms(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.abs(estimate - truth) ** 2))
        / np.sqrt(np.mean(np.abs(truth) ** 2))
    )


# --- wave -> rays -> wave ------------------------------------------------------


def test_wave_to_ray_to_wave_is_exact_in_the_enumeration_limit() -> None:
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    assert _relative_rms(_reconstruct(bundle), field.u) < 1e-14


@pytest.mark.slow
def test_wave_to_ray_to_wave_converges_at_the_monte_carlo_rate() -> None:
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)

    counts = [256, 512, 1024, 2048, 4096]
    errors = []
    for count in counts:
        realizations = [
            _relative_rms(
                _reconstruct(
                    spectrum_to_rays(
                        spectrum, draw_indices(density, count, np.random.default_rng(600 + s)), density
                    )
                ),
                field.u,
            )
            for s in range(16)
        ]
        errors.append(float(np.mean(realizations)))

    exponent, _ = np.polyfit(np.log(counts), np.log(errors), 1)
    assert abs(exponent + 0.5) <= 0.1, f"round-trip exponent {exponent:.4f}"


# --- rays -> wave -> rays ------------------------------------------------------


def test_ray_to_wave_to_ray_recovers_the_spectral_content() -> None:
    """Start from rays, reconstruct, decompose again. The recovered modal
    amplitudes must match the ones the rays carried."""
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    rebuilt_field, _ = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        normalization="one_over_n",
        projection=Projection.ASM_CONSISTENT,
    )
    recovered = decompose(rebuilt_field)

    np.testing.assert_allclose(
        recovered.spectrum, spectrum.spectrum, rtol=1e-10, atol=1e-16
    )


# --- The round trip must be able to fail ----------------------------------------


def test_a_mismatched_phase_sign_pairing_breaks_the_round_trip() -> None:
    """The test that makes every other round-trip result meaningful.

    A shared convention error can cancel between the two directions. Here the
    reconstruction runs with a flipped phasor while the decomposition does not,
    so the pairing is inconsistent -- and the round trip must notice. If this
    passed, the exactness result above would prove nothing.
    """
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    consistent = _relative_rms(_reconstruct(bundle), field.u)
    mismatched = _relative_rms(
        _reconstruct(bundle, perturbation=Perturbation(phase_sign=-1)), field.u
    )

    assert consistent < 1e-14
    assert mismatched > 0.5, f"inconsistent pairing went undetected ({mismatched:.3e})"


def test_the_obliquity_convention_must_also_match_across_the_pair() -> None:
    """CHE-25's finding, seen from the round trip: using the sensor convention
    on the reconstruction side breaks exactness by the cos(theta) spread of the
    spectrum, even though nothing else changed."""
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    asm = _relative_rms(_reconstruct(bundle, Projection.ASM_CONSISTENT), field.u)
    sensor = _relative_rms(_reconstruct(bundle, Projection.SENSOR_OBLIQUITY), field.u)

    assert asm < 1e-14
    # Bounded below by the worst 1 - cos(theta) present in the spectrum.
    worst_obliquity_defect = 1.0 - float(np.min(bundle.directions[:, 2]))
    assert sensor > 0.1 * worst_obliquity_defect


# --- Power accounting -----------------------------------------------------------


def test_power_terms_are_reported_separately_rather_than_netted() -> None:
    """Evanescent loss, aperture truncation and Monte Carlo variance are three
    different things. A single 'power ratio' would let one hide another."""
    fine_pitch = 200e-9
    field = ComplexField(
        u=_random_field().u,
        sample_pitch_m=(fine_pitch, fine_pitch),
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )
    spectrum = decompose(field)

    assert spectrum.evanescent_power_fraction > 0.0
    # Total discrete power is the field's, independent of how modes are binned.
    assert spectrum.total_discrete_power == pytest.approx(field.discrete_power())
    # Parseval, with the 1/(ny*nx) folded into the spectrum definition.
    modal = float(np.sum(np.abs(spectrum.spectrum) ** 2))
    assert modal == pytest.approx(
        float(np.sum(np.abs(field.u) ** 2)) / field.u.size, rel=1e-12
    )


def test_the_bundle_declares_which_normalization_a_reconstruction_owes_it() -> None:
    """Found by the cascade: the 1/N of SI eq S5 was prose on the bundle that no
    component could act on, so the cascade reconstructed a field scaled by the
    mode count (256x on a 16x16 grid).

    A spectrally sampled bundle is a Monte Carlo estimate and needs 1/N; a
    traced bundle is the ensemble itself and must not be averaged. The bundle
    knows which it is, so it says so in structured form and the reconstruction
    follows it by default instead of every caller restating it.
    """
    from multiscale_optics_agent.couplers.ray_to_wave import collimated_bundle

    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    sampled = spectrum_to_rays(spectrum, enumerate_indices(density), density)
    assert sampled.reconstruction_normalization == "one_over_n"

    traced = collimated_bundle(
        positions_xy_m=np.zeros((4, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    assert traced.reconstruction_normalization == "none"

    # Following the declaration reproduces the field; ignoring it is off by
    # exactly the ray count.
    honoured, diagnostics = ray_to_wave(sampled, grid_shape=GRID, sample_pitch_m=PITCH)
    ignored, _ = ray_to_wave(
        sampled, grid_shape=GRID, sample_pitch_m=PITCH, normalization="none"
    )
    assert diagnostics.normalization == "one_over_n"
    assert _relative_rms(honoured.u, field.u) < 1e-14
    np.testing.assert_allclose(ignored.u, honoured.u * sampled.count, rtol=1e-12)


def test_an_unknown_reconstruction_normalization_is_refused() -> None:
    from multiscale_optics_agent.couplers.ray_to_wave import collimated_bundle

    bundle = collimated_bundle(
        positions_xy_m=np.zeros((2, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    with pytest.raises(ContractError, match="reconstruction_normalization"):
        bundle._replace(reconstruction_normalization="average_somehow")


# --- Cascade (SI Algorithm S1) ------------------------------------------------------


def _phase_doe(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-math.pi, math.pi, size=GRID)).astype(np.complex128)


def test_ray_count_after_a_planar_doe_is_the_budget_not_the_input_size() -> None:
    """The reason Algorithm S1 exists. Two surfaces in series must not multiply."""
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    launches = np.zeros((4, 2))
    rng = np.random.default_rng(20260812)

    first, _, diagnostics_a = planar_doe_step(
        incident,
        _phase_doe(1),
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=launches,
        secondary_count=64,
        rng=rng,
    )
    second, _, diagnostics_b = planar_doe_step(
        first,
        _phase_doe(2),
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=launches,
        secondary_count=64,
        rng=rng,
    )

    assert first.count == 4 * 64
    assert second.count == 4 * 64
    # The count after the second surface is identical to after the first, rather
    # than 256 * 64. That is the whole claim.
    assert second.count == first.count
    assert diagnostics_b.incident_ray_count == first.count
    assert diagnostics_a.count_growth < 2.0


def test_the_cascade_applies_the_doe_to_the_accumulated_field() -> None:
    """Order matters: accumulate all rays first, then transmit once. Applying
    the DOE per ray would destroy the interference the step preserves.

    Checked by comparing against the field route, which has no rays in it."""
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)
    doe = _phase_doe()

    _, transmitted, _ = planar_doe_step(
        incident,
        doe,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)),
        secondary_count=None,
    )

    np.testing.assert_allclose(transmitted.u, field.u * doe, rtol=1e-10, atol=1e-14)


def test_a_pure_phase_doe_conserves_discrete_power() -> None:
    """|exp(i phi)| = 1, so the transmitted power must equal the incident power.
    A conservation check the implementation cannot satisfy by accident."""
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    _, _, diagnostics = planar_doe_step(
        incident,
        _phase_doe(),
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)),
        secondary_count=None,
    )

    assert diagnostics.transmitted_discrete_power == pytest.approx(
        diagnostics.incident_discrete_power, rel=1e-10
    )


def test_an_absorbing_doe_loses_power_and_says_so() -> None:
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)
    half = np.full(GRID, 0.5 + 0.0j)

    _, _, diagnostics = planar_doe_step(
        incident,
        half,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)),
        secondary_count=None,
    )

    ratio = (
        diagnostics.transmitted_discrete_power / diagnostics.incident_discrete_power
    )
    assert ratio == pytest.approx(0.25, rel=1e-10)


def test_a_real_transmission_is_refused_as_an_undeclared_phase() -> None:
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    with pytest.raises(ContractError, match="undeclared phase"):
        planar_doe_step(
            incident,
            np.ones(GRID, dtype=np.float64),
            grid_shape=GRID,
            sample_pitch_m=PITCH,
            plane=PLANE,
            launch_positions_xy_m=np.zeros((1, 2)),
            secondary_count=None,
        )


def test_cascade_resampling_without_a_seed_is_refused() -> None:
    field = _random_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    incident = spectrum_to_rays(spectrum, enumerate_indices(density), density)

    with pytest.raises(ContractError, match="explicit seeded generator"):
        planar_doe_step(
            incident,
            _phase_doe(),
            grid_shape=GRID,
            sample_pitch_m=PITCH,
            plane=PLANE,
            launch_positions_xy_m=np.zeros((1, 2)),
            secondary_count=32,
        )
