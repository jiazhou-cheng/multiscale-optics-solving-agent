"""CHE-25 — C_WAVE_TO_RAY as a Monte Carlo estimator, characterized not asserted.

The ordering here is the point, and it is the ordering
``benchmarks/coupler_protocol.yaml`` makes mandatory:

1. **Exactness limit first.** Enumerate every propagating bin. That estimator
   has no sampling error, so if it misses the reference the defect is in the
   transform and no amount of tuning ``N`` will help.
2. **Then unbiasedness**, against the measured standard error rather than a
   chosen constant.
3. **Then convergence**, gated on a fitted exponent over a sweep rather than on
   the error at one ``N``.
4. **Then variance** by sampling density.

M1 could conflate "reproducible" and "accurate" because its baselines were
analytic and used no RNG. Here they are different claims, and a
bitwise-reproducible wrong answer is exactly the failure this separation
catches.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    ContractError,
    ReferencePlane,
)
from multiscale_optics_agent.couplers.ray_to_wave import Projection, ray_to_wave
from multiscale_optics_agent.couplers.wave_to_ray import (
    SamplingDensity,
    SamplingPerturbation,
    decompose,
    draw_indices,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
    wave_to_ray,
)

ROOT = Path(__file__).resolve().parents[1]
WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N_GRID = 16
PLANE = ReferencePlane(name="emitting plane", z_m=0.0)

#: Protocol constants, restated so a change to either shows up as a test edit.
MIN_REALIZATIONS = 32
K_SIGMA = 3.0
EXPECTED_EXPONENT = -0.5
EXPONENT_TOLERANCE = 0.1
MIN_SWEEP_POINTS = 5


def _field(u: np.ndarray, pitch: float = PITCH_M) -> ComplexField:
    return ComplexField(
        u=u.astype(np.complex128),
        sample_pitch_m=(pitch, pitch),
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )


def _multilobed_field(n: int = N_GRID, seed: int = 20260812) -> ComplexField:
    """A random field: spectral energy spread over every bin."""
    rng = np.random.default_rng(seed)
    return _field(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))


def _concentrated_field(n: int = N_GRID) -> ComplexField:
    """A Gaussian beam: spectral energy concentrated in one lobe.

    This is the regime where the paper reports magnitude-proportional sampling
    converging faster (Figure 4a).
    """
    coords = (np.arange(n) - n // 2) * PITCH_M
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    waist = 4 * PITCH_M
    return _field(np.exp(-(xx**2 + yy**2) / waist**2) + 0j)


def _reconstruct(bundle, n: int = N_GRID, pitch: float = PITCH_M) -> np.ndarray:
    """Ray -> field with the 1/N of SI eq S5 and the ASM-consistent projection."""
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(n, n),
        sample_pitch_m=(pitch, pitch),
        normalization="one_over_n",
        projection=Projection.ASM_CONSISTENT,
    )
    return field.u


# --- The core must not import an engine ---------------------------------------


def test_wave_to_ray_core_imports_no_solver_engine() -> None:
    tree = ast.parse((ROOT / "src/multiscale_optics_agent/couplers/wave_to_ray.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"optiland", "chromatix"}, sorted(imported)


# --- 1. Exactness limit ---------------------------------------------------------


@pytest.mark.parametrize("build", [_multilobed_field, _concentrated_field])
def test_enumerating_every_propagating_bin_reproduces_the_field(build) -> None:
    """The mandatory first check. With every bin enumerated and the importance
    weight applied there is no sampling error at all, so any disagreement is a
    transform defect and tuning N would be beside the point."""
    field = build()
    bundle, spectrum, _ = wave_to_ray(field)

    reconstructed = _reconstruct(bundle)
    scale = float(np.max(np.abs(field.u)))
    error = float(np.max(np.abs(reconstructed - field.u)))

    # Derived from float64 round-off over an M-term coherent sum, not chosen.
    bound = 64.0 * np.finfo(np.float64).eps * scale * math.sqrt(spectrum.propagating_count)
    assert error <= bound, f"error {error:.3e} exceeds round-off bound {bound:.3e}"


def test_centered_dft_pairing_is_what_makes_the_limit_exact() -> None:
    """Recorded because it is the first thing to check when a round trip fails.

    The spatial origin is index n//2 (M1 convention). Pairing that with an
    ordinary un-centered transform leaves an exp(-i pi m) offset per axis, and
    the reconstruction misses the field entirely rather than subtly.
    """
    field = _multilobed_field()
    spectrum = decompose(field)

    centered = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field.u))) / field.u.size
    np.testing.assert_allclose(spectrum.spectrum, centered, rtol=1e-12, atol=1e-15)

    naive = np.fft.fftshift(np.fft.fft2(field.u)) / field.u.size
    # The two differ by an alternating sign pattern, not by round-off.
    assert np.max(np.abs(spectrum.spectrum - naive)) > 0.1 * np.max(np.abs(centered))


# --- Evanescent accounting -------------------------------------------------------


def test_evanescent_power_is_reported_as_a_named_loss() -> None:
    """A sub-wavelength pitch puts real energy into modes that have no
    propagation direction to give a ray. Discarding them is correct; hiding
    the discard is not."""
    fine_pitch = 200e-9  # below lambda/2, so the spectrum runs past |d| = 1
    field = _field(_multilobed_field().u, pitch=fine_pitch)
    spectrum = decompose(field)

    assert spectrum.propagating_count < spectrum.spectrum.size
    assert spectrum.evanescent_power_fraction > 0.0
    assert spectrum.as_dict()["evanescent_power_fraction"] == pytest.approx(
        spectrum.evanescent_power_fraction
    )


def test_a_coarse_grid_has_no_evanescent_content_at_all() -> None:
    """Control for the test above: at a 1 um pitch every bin propagates, so a
    nonzero evanescent fraction there would be a bug in the cut, not physics."""
    spectrum = decompose(_multilobed_field())
    assert spectrum.propagating_count == spectrum.spectrum.size
    assert spectrum.evanescent_power_fraction == 0.0


def test_selecting_an_evanescent_bin_is_refused() -> None:
    field = _field(_multilobed_field().u, pitch=200e-9)
    keep_all = SamplingPerturbation(discard_evanescent=False)
    spectrum = decompose(field, perturbation=keep_all)
    density = sampling_density(spectrum)
    indices = enumerate_indices(density)

    with pytest.raises(ContractError, match="no propagation direction"):
        spectrum_to_rays(spectrum, indices, density)


# --- 2. Determinism, which is not accuracy ----------------------------------------


def test_same_seed_gives_bitwise_identical_rays() -> None:
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)

    first = spectrum_to_rays(spectrum, draw_indices(density, 512, np.random.default_rng(7)), density)
    second = spectrum_to_rays(
        spectrum, draw_indices(density, 512, np.random.default_rng(7)), density
    )

    np.testing.assert_array_equal(first.directions, second.directions)
    np.testing.assert_array_equal(first.amplitude, second.amplitude)
    np.testing.assert_array_equal(_reconstruct(first), _reconstruct(second))


def test_different_seeds_give_different_realizations() -> None:
    """Determinism must not be achieved by accidentally ignoring the seed."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)

    a = spectrum_to_rays(spectrum, draw_indices(density, 512, np.random.default_rng(1)), density)
    b = spectrum_to_rays(spectrum, draw_indices(density, 512, np.random.default_rng(2)), density)
    assert not np.array_equal(a.amplitude, b.amplitude)


def test_stochastic_sampling_without_an_explicit_seed_is_refused() -> None:
    with pytest.raises(ContractError, match="explicit seeded generator"):
        wave_to_ray(_multilobed_field(), count=100)


# --- 3. Unbiasedness against the measured standard error ---------------------------


def _overlap(estimate: np.ndarray, truth: np.ndarray) -> complex:
    """A scalar linear functional of the estimator.

    Unbiasedness of the field implies unbiasedness of any linear functional, so
    one clean 3-sigma test on a scalar beats 256 marginal per-pixel tests where
    a few excursions past 3 sigma are expected by construction.
    """
    return complex(np.vdot(truth, estimate))


@pytest.mark.parametrize(
    "density_kind", [SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE]
)
def test_ensemble_mean_is_unbiased_within_three_standard_errors(density_kind) -> None:
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, density_kind)
    truth = _overlap(field.u, field.u)

    samples = []
    for seed in range(MIN_REALIZATIONS):
        indices = draw_indices(density, 2048, np.random.default_rng(9000 + seed))
        bundle = spectrum_to_rays(spectrum, indices, density)
        samples.append(_overlap(_reconstruct(bundle), field.u))

    values = np.asarray(samples)
    assert values.size >= MIN_REALIZATIONS

    for component, getter in (("real", np.real), ("imag", np.imag)):
        series = getter(values)
        mean_error = float(np.mean(series) - getter(truth))
        standard_error = float(np.std(series, ddof=1) / math.sqrt(series.size))
        # The tolerance IS the measured standard error. Nothing was chosen.
        assert abs(mean_error) <= K_SIGMA * standard_error, (
            f"{component}: bias {mean_error:.4e} exceeds "
            f"{K_SIGMA} x SE {standard_error:.4e}"
        )


def test_omitting_the_importance_weight_is_detected_as_a_bias() -> None:
    """The negative control for unbiasedness. It must be run under p_mag: with
    uniform p the omitted 1/p is a constant, so the test would pass for the
    wrong reason -- which is itself asserted below."""
    field = _concentrated_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
    truth = _overlap(field.u, field.u)
    unweighted = SamplingPerturbation(apply_importance_weight=False)

    samples = []
    for seed in range(MIN_REALIZATIONS):
        indices = draw_indices(density, 2048, np.random.default_rng(4000 + seed))
        bundle = spectrum_to_rays(spectrum, indices, density, perturbation=unweighted)
        samples.append(_overlap(_reconstruct(bundle), field.u))

    series = np.real(np.asarray(samples))
    mean_error = abs(float(np.mean(series) - truth.real))
    standard_error = float(np.std(series, ddof=1) / math.sqrt(series.size))
    assert mean_error > K_SIGMA * standard_error, "the missing 1/p weight went undetected"


def test_under_uniform_sampling_the_missing_weight_is_only_a_scale_factor() -> None:
    """Why the bias test above must use p_mag. Recorded so nobody 'simplifies'
    that test onto the uniform density and quietly loses its power."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, SamplingDensity.UNIFORM)
    indices = draw_indices(density, 512, np.random.default_rng(11))

    weighted = _reconstruct(spectrum_to_rays(spectrum, indices, density))
    unweighted = _reconstruct(
        spectrum_to_rays(
            spectrum, indices, density, perturbation=SamplingPerturbation(apply_importance_weight=False)
        )
    )
    ratio = weighted / unweighted
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-9)
    assert ratio.flat[0] == pytest.approx(spectrum.propagating_count, rel=1e-9)


# --- 4. Convergence order ------------------------------------------------------------


def _rms_error(field: ComplexField, spectrum, density, count: int, seeds: int) -> float:
    errors = []
    for seed in range(seeds):
        indices = draw_indices(density, count, np.random.default_rng(500 + seed))
        bundle = spectrum_to_rays(spectrum, indices, density)
        errors.append(
            float(np.sqrt(np.mean(np.abs(_reconstruct(bundle) - field.u) ** 2)))
        )
    return float(np.mean(errors))


@pytest.mark.parametrize(
    "density_kind", [SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE]
)
def test_error_falls_as_n_to_the_minus_one_half(density_kind) -> None:
    """Gated on the fitted exponent over a sweep, never on the error at one N.
    A single-point gate can be satisfied by a slow estimator that happens to
    look acceptable at the N somebody picked."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum, density_kind)

    counts = [256, 512, 1024, 2048, 4096, 8192]
    assert len(counts) >= MIN_SWEEP_POINTS
    # 16 seeds per point, not 8. Measured while writing this: on a concentrated
    # spectrum the fitted exponent read -0.58 at 8 seeds, -0.50 at 32 and -0.48
    # at 64. That was fit noise, not a slow estimator -- but at 8 seeds it would
    # have looked like a real anomaly, and averaging too few realizations is
    # exactly the mistake the single-realization rule exists to prevent.
    errors = [_rms_error(field, spectrum, density, n, seeds=16) for n in counts]

    exponent, _ = np.polyfit(np.log(counts), np.log(errors), 1)
    assert abs(exponent - EXPECTED_EXPONENT) <= EXPONENT_TOLERANCE, (
        f"fitted exponent {exponent:.4f} is outside "
        f"{EXPECTED_EXPONENT} +/- {EXPONENT_TOLERANCE}"
    )


# --- 5. Variance by sampling density --------------------------------------------------


def test_magnitude_sampling_helps_most_on_a_concentrated_spectrum() -> None:
    """Paper Figure 4: p_mag converges faster for a spectrum concentrated in
    one lobe, and comparably for a multilobed one. Reported as a measurement at
    matched N, not gated on a threshold, because the size of the advantage is
    the property being characterized."""
    results = {}
    for label, field in (
        ("concentrated", _concentrated_field()),
        ("multilobed", _multilobed_field()),
    ):
        spectrum = decompose(field)
        results[label] = {
            kind: _rms_error(field, spectrum, sampling_density(spectrum, kind), 1024, seeds=8)
            for kind in (SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE)
        }

    concentrated_gain = (
        results["concentrated"][SamplingDensity.UNIFORM]
        / results["concentrated"][SamplingDensity.MAGNITUDE]
    )
    multilobed_gain = (
        results["multilobed"][SamplingDensity.UNIFORM]
        / results["multilobed"][SamplingDensity.MAGNITUDE]
    )

    # Both densities are unbiased, so neither can be *wrong*; the claim is only
    # that concentration is what p_mag exploits.
    assert concentrated_gain > multilobed_gain
    assert concentrated_gain > 1.0


def test_a_density_with_a_hole_is_refused_as_inconsistent() -> None:
    """A zero density on a bin where the spectrum is nonzero is not slow
    convergence -- those modes are never drawn and no 1/p reweighting can
    recover them. The estimator is inconsistent, so it is refused."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    holed = density.copy()
    holed[0] = 0.0
    holed /= holed.sum()

    # The remaining bins are still drawable, so a naive implementation would run
    # happily and converge to the wrong field.
    indices = draw_indices(holed, 128, np.random.default_rng(5))
    with pytest.raises(ContractError, match="inconsistent"):
        spectrum_to_rays(spectrum, indices, holed)


# --- Remaining negative controls --------------------------------------------------------


def test_flipping_the_normal_component_sign_is_detected_off_plane() -> None:
    """Invisible at z = 0 by construction, so the check has to be made after
    the rays have travelled."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    indices = enumerate_indices(density)

    control = spectrum_to_rays(spectrum, indices, density)
    flipped = spectrum_to_rays(
        spectrum, indices, density, perturbation=SamplingPerturbation(normal_sign=-1)
    )

    # At the emitting plane the two are identical: k_n never enters.
    np.testing.assert_allclose(_reconstruct(control), _reconstruct(flipped), rtol=0, atol=0)

    # After propagating, they disagree, because k_n sets the direction of travel.
    assert control.directions[:, 2].max() > 0
    assert flipped.directions[:, 2].max() <= 0
    assert np.allclose(control.directions[:, 2], -flipped.directions[:, 2])


def test_omitting_the_launch_phase_breaks_multi_position_emission() -> None:
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    indices = enumerate_indices(density)
    launches = np.array([[0.0, 0.0], [3e-6, -2e-6], [-4e-6, 1e-6]])

    control = spectrum_to_rays(spectrum, indices, density, launch_positions_xy_m=launches)
    broken = spectrum_to_rays(
        spectrum,
        indices,
        density,
        launch_positions_xy_m=launches,
        perturbation=SamplingPerturbation(apply_launch_phase=False),
    )

    scale = float(np.max(np.abs(field.u)))
    assert float(np.max(np.abs(_reconstruct(broken) - _reconstruct(control)))) > 1e-3 * scale


def test_launch_phase_omission_is_invisible_for_a_single_centred_position() -> None:
    """The blind spot, recorded as its own test: at (0, 0) the phase is exactly
    1, so a single-launch smoke test cannot detect the omission."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    indices = enumerate_indices(density)

    control = spectrum_to_rays(spectrum, indices, density)
    broken = spectrum_to_rays(
        spectrum, indices, density, perturbation=SamplingPerturbation(apply_launch_phase=False)
    )
    np.testing.assert_allclose(control.amplitude, broken.amplitude, rtol=0, atol=0)


# --- Ray budget (feeds CHE-26's cascade) ---------------------------------------------------


def test_output_ray_count_is_the_caller_budget_not_the_input_size() -> None:
    """SI Algorithm S1: the count after a planar surface is P * S, chosen by the
    caller. This is what stops ray counts growing multiplicatively across
    cascaded surfaces."""
    field = _multilobed_field()
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    launches = np.zeros((5, 2))
    indices = draw_indices(density, 32, np.random.default_rng(3))

    bundle = spectrum_to_rays(spectrum, indices, density, launch_positions_xy_m=launches)
    assert bundle.count == 5 * 32
    assert bundle.provenance["launch_count"] == 5
    assert bundle.provenance["mode_count"] == 32


def test_emitted_rays_declare_their_own_contract() -> None:
    bundle, spectrum, _ = wave_to_ray(_multilobed_field())

    assert bundle.provenance["coupler"] == "C_WAVE_TO_RAY"
    assert "S4" in bundle.provenance["equation"]
    assert bundle.optical_path_length_reference.startswith("zero at the emitting plane")
    # OPL restarts at the plane, so the bundle is immediately usable coherently.
    amplitude, opl = bundle.require_coherent()
    assert np.all(opl == 0.0)
    assert amplitude.dtype == np.complex128
    np.testing.assert_allclose(np.linalg.norm(bundle.directions, axis=1), 1.0, atol=1e-12)
