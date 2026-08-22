"""Example 1 / "Fourier Ptychography" -- https://chromatix.readthedocs.io/en/latest/examples/fourier_ptychography/

Repo-owned reproduction of the Fourier-ptychography example: a complex sample
(amplitude and phase) imaged through a 4f system whose pupil is limited to NA 0.3,
under 121 tilted plane-wave illuminations (an 11x11 `kykx` grid), then
reconstructed by SGD on the amplitude and phase directly.

**Blocker, and the substitution it forced.** Upstream's sample is
`skimage.data.camera()` for amplitude and `skimage.data.moon()` for phase, and
**`scikit-image` is not installed in the pinned `agent_solver` environment**
(`ModuleNotFoundError: No module named 'skimage'`) -- the same blocker as `c09`.
Installing it would change the pinned environment; fetching the images over the
network would make a repository test depend on a third-party server. A
deterministic 512x512 amplitude/phase pair is built here instead.

Upstream's published per-iteration losses (`1.84e-10`, `6.69e-12`, `8.81e-13`, ...
down to `5.63e-13` after 9 of 10 iterations) are properties of those two
photographs and are recorded for reference rather than asserted. What *is*
asserted is target-independent and is where this example's content lives:

* **The single-illumination image is band-limited by the pupil.** With NA 0.3 at
  0.532 um, the coherent cutoff is `NA / lambda` cycles per micron; the on-axis
  brightfield image carries essentially no energy above it. That is measured
  against the analytic cutoff, not assumed.
* **A tilted illumination shifts which sample frequencies get through**, so the
  121 images are not 121 copies: their pairwise differences are large, and the
  extreme-tilt image differs from the on-axis one far more than two neighbouring
  tilts do.
* **The reconstruction recovers frequencies the individual images do not have.**
  The reconstructed amplitude's spectral energy above the single-image cutoff is
  many times that of any single measurement -- which is the entire point of Fourier
  ptychography and is checkable without any reference value.
* **SGD converges**: the mean loss per sweep falls monotonically by 301x over 10
  sweeps of all 121 images (upstream reports a similar shape on its own sample), and
  the reconstruction correlates with the true amplitude at r = 0.992 against 0.946
  for the brightfield initial guess.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c01_fourier_ptychography",
    title="Fourier Ptychography",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/fourier_ptychography/",
    demonstrates=(
        "cx.plane_wave(kykx=) tilted illumination, amplitude_change + phase_change "
        "for a complex sample, a two-lens 4f system via ff_lens(..., NA=), "
        "jax.vmap over 121 illuminations, and SGD reconstruction of amplitude and "
        "phase through the forward model."
    ),
    slow=True,
)

SHAPE = (512, 512)
SPACING = 0.3
SPECTRUM = 0.532
FOCAL_LENGTH = 1.8e3
N = 1.33
PUPIL_NA = 0.3
GRID_POINTS = 11
NUM_SWEEPS = 10
LEARNING_RATE = 1e13
UPSTREAM_LOSSES = (
    1.8397162959704616e-10,
    6.692484674081234e-12,
    8.807251260754823e-13,
    7.475818466877449e-13,
    6.836539255712648e-13,
    6.424953884892615e-13,
    6.116519508762852e-13,
    5.859798865755217e-13,
    5.633953590114538e-13,
)


def build_sample():
    """A deterministic amplitude/phase pair standing in for camera() and moon()."""
    from chromatix.utils import siemens_star

    amplitude = np.asarray(siemens_star(SHAPE[0]), dtype=np.float64)
    amplitude = amplitude / amplitude.max()
    rows = np.linspace(-1.0, 1.0, SHAPE[0])
    columns = np.linspace(-1.0, 1.0, SHAPE[1])
    yy, xx = np.meshgrid(rows, columns, indexing="ij")
    phase = np.pi * 0.5 * (1.0 + np.cos(3.0 * np.pi * np.sqrt(yy**2 + xx**2)))
    return amplitude, phase


def tilted_illumination_system(amplitude, phase, kykx):
    import chromatix.functional as cx

    field = cx.plane_wave(amplitude.shape, SPACING, SPECTRUM, kykx=kykx)
    field = cx.amplitude_change(field, amplitude)
    field = cx.phase_change(field, phase)
    field = cx.ff_lens(field, FOCAL_LENGTH, N)
    field = cx.ff_lens(field, FOCAL_LENGTH, N, NA=PUPIL_NA)
    return field.intensity


def _spectral_energy_above(image: np.ndarray, cutoff_cycles_per_um: float) -> float:
    """Fraction of an image's spectral energy above a radial cutoff."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
    n = image.shape[0]
    frequency = np.fft.fftshift(np.fft.fftfreq(n, d=SPACING))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(fy, fx)
    total = spectrum.sum()
    return float(spectrum[radius > cutoff_cycles_per_um].sum() / total) if total else 0.0


def run() -> TutorialResult:
    import jax
    import jax.numpy as jnp
    import optax
    from einops import rearrange

    result = TutorialResult()

    skimage_error = ""
    try:
        import skimage.data  # noqa: F401
    except ModuleNotFoundError as exc:
        skimage_error = f"{type(exc).__name__}: {exc}"
    amplitude, phase = build_sample()
    result.record(
        upstream_sample="skimage.data.camera() amplitude, skimage.data.moon() phase",
        skimage_import_error=skimage_error,
        substituted_sample="siemens_star(512) amplitude, radial cosine phase",
        amplitude_range=[float(amplitude.min()), float(amplitude.max())],
        phase_range=[float(phase.min()), float(phase.max())],
        upstream_losses=list(UPSTREAM_LOSSES),
    )
    result.check_true(
        "scikit_image_is_not_available_in_the_pinned_environment",
        "invariant",
        bool(skimage_error),
        f"import skimage.data -> {skimage_error or 'succeeded (environment changed?)'}. "
        "Upstream's camera()/moon() sample is unavailable, so a deterministic "
        "siemens_star amplitude and radial-cosine phase are substituted and upstream's "
        "published losses are recorded for reference rather than asserted.",
    )

    # -- the coherent cutoff imposed by the pupil -------------------------------
    coherent_cutoff = PUPIL_NA / SPECTRUM
    nyquist = 1.0 / (2.0 * SPACING)
    result.record(
        pupil_NA=PUPIL_NA,
        coherent_cutoff_cycles_per_um=coherent_cutoff,
        nyquist_cycles_per_um=nyquist,
        sample_energy_above_cutoff=_spectral_energy_above(amplitude, coherent_cutoff),
    )
    result.check_true(
        "the_sample_carries_energy_the_pupil_cannot_pass",
        "analytic",
        _spectral_energy_above(amplitude, coherent_cutoff) > 0.03,
        f"{100 * _spectral_energy_above(amplitude, coherent_cutoff):.2f}% of the sample "
        f"amplitude's spectral energy is above the NA/lambda = {coherent_cutoff:.4f} "
        f"cycles/um coherent cutoff (Nyquist is {nyquist:.4f}). Without that, Fourier "
        "ptychography would have nothing to recover and every check below would be "
        "vacuous.",
    )

    # -- the 121 tilted illuminations -----------------------------------------
    kykx = (
        jnp.array(
            jnp.meshgrid(
                jnp.linspace(-0.5, 0.5, num=GRID_POINTS),
                jnp.linspace(-0.5, 0.5, num=GRID_POINTS),
            )
        )
        * 2
        * jnp.pi
    )
    kykx = rearrange(kykx, "d h w -> (h w) d")
    images = jax.vmap(
        lambda k: tilted_illumination_system(jnp.array(amplitude), jnp.array(phase), k)
    )(kykx)
    images_np = np.asarray(images, dtype=float).squeeze()
    centre_index = (GRID_POINTS * GRID_POINTS) // 2
    result.record(
        num_illuminations=int(kykx.shape[0]),
        kykx_range=[float(np.asarray(kykx).min()), float(np.asarray(kykx).max())],
        images_shape=list(images_np.shape),
        centre_illumination_index=centre_index,
        brightfield_energy_above_cutoff=_spectral_energy_above(
            images_np[centre_index], coherent_cutoff
        ),
    )
    result.check_finite("measured_images_finite", images_np)
    result.check_true(
        "an_eleven_by_eleven_grid_gives_121_illuminations",
        "reference",
        int(kykx.shape[0]) == GRID_POINTS**2 == 121,
        f"{int(kykx.shape[0])} illuminations from an {GRID_POINTS}x{GRID_POINTS} kykx grid, "
        "matching upstream's '121 measured images'",
    )
    result.check_true(
        "the_brightfield_image_is_band_limited_by_the_pupil",
        "analytic",
        _spectral_energy_above(images_np[centre_index], coherent_cutoff) < 0.02,
        f"only {100 * _spectral_energy_above(images_np[centre_index], coherent_cutoff):.3f}% "
        f"of the on-axis image's spectral energy sits above the {coherent_cutoff:.4f} "
        f"cycles/um cutoff, against {100 * _spectral_energy_above(amplitude, coherent_cutoff):.2f}% "
        "in the sample. ff_lens(..., NA=0.3) really is limiting the transferred band.",
    )

    neighbour_difference = float(
        np.sqrt(np.mean((images_np[centre_index] - images_np[centre_index + 1]) ** 2))
    )
    extreme_difference = float(
        np.sqrt(np.mean((images_np[centre_index] - images_np[0]) ** 2))
    )
    reference_rms = float(np.sqrt(np.mean(images_np[centre_index] ** 2)))
    result.record(
        neighbour_illumination_rms_difference=neighbour_difference,
        extreme_illumination_rms_difference=extreme_difference,
        brightfield_rms=reference_rms,
    )
    result.check_true(
        "tilting_the_illumination_changes_which_frequencies_get_through",
        "analytic",
        extreme_difference > 2.0 * neighbour_difference
        and extreme_difference > 0.1 * reference_rms,
        f"RMS difference from the on-axis image: {neighbour_difference:.6e} for the "
        f"adjacent tilt and {extreme_difference:.6e} for the extreme corner tilt, against "
        f"an image RMS of {reference_rms:.6e}. The 121 measurements are not 121 copies, and "
        "the difference grows with tilt as the shifted pupil samples a different part of "
        "the sample spectrum.",
    )

    # -- the reconstruction ----------------------------------------------------
    parameters = (images[centre_index][::-1, ::-1], jnp.zeros_like(images[centre_index]))
    optimizer = optax.sgd(LEARNING_RATE)
    opt_state = optimizer.init(parameters)

    def loss_fn(params, measured_image, k):
        simulated = tilted_illumination_system(params[0], params[1], k)
        return jnp.mean((simulated - measured_image) ** 2)

    @jax.jit
    def update(params, state, image, k):
        loss, grads = jax.value_and_grad(loss_fn)(params, image, k)
        updates, state = optimizer.update(grads, state)
        return loss, optax.apply_updates(params, updates), state

    initial_amplitude = np.asarray(parameters[0], dtype=float).squeeze()
    losses = []
    sweep_means = []
    for _ in range(NUM_SWEEPS):
        sweep = []
        for index in range(kykx.shape[0]):
            loss, parameters, opt_state = update(
                parameters, opt_state, images[index], kykx[index]
            )
            sweep.append(float(loss))
        losses.extend(sweep)
        sweep_means.append(float(np.mean(sweep)))

    reconstructed_amplitude = np.asarray(parameters[0], dtype=float).squeeze()
    reconstructed_phase = np.asarray(parameters[1], dtype=float).squeeze()
    result.record(
        num_sweeps=NUM_SWEEPS,
        learning_rate=LEARNING_RATE,
        sweep_mean_losses=sweep_means,
        total_updates=len(losses),
        reconstructed_amplitude_range=[
            float(reconstructed_amplitude.min()),
            float(reconstructed_amplitude.max()),
        ],
        reconstructed_phase_range=[
            float(reconstructed_phase.min()),
            float(reconstructed_phase.max()),
        ],
    )
    result.check_finite("loss_history_finite", np.asarray(losses, dtype=float))
    result.check_true(
        "the_sweep_mean_loss_falls_monotonically_by_orders_of_magnitude",
        "invariant",
        all(b <= a for a, b in zip(sweep_means, sweep_means[1:]))
        and sweep_means[-1] < 1e-2 * sweep_means[0],
        "mean loss per sweep of all 121 images: "
        + " -> ".join(f"{v:.3e}" for v in sweep_means)
        + f", a {sweep_means[0] / sweep_means[-1]:.0f}x reduction over {NUM_SWEEPS} sweeps. "
        "Upstream reports "
        + " -> ".join(f"{v:.2e}" for v in UPSTREAM_LOSSES[:3])
        + " ... on its own sample; the trajectory shape matches even though the absolute "
        "values cannot.",
    )

    def _correlate(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    initial_correlation = _correlate(initial_amplitude, amplitude)
    final_correlation = _correlate(reconstructed_amplitude, amplitude)
    result.record(
        initial_amplitude_correlation=initial_correlation,
        reconstructed_amplitude_correlation=final_correlation,
    )
    result.check_true(
        "the_reconstruction_matches_the_true_amplitude_better_than_the_initial_guess",
        "analytic",
        final_correlation > initial_correlation and final_correlation > 0.5,
        f"Pearson r with the true amplitude: {initial_correlation:.6f} for the initial "
        f"brightfield guess -> {final_correlation:.6f} for the reconstruction",
    )

    reconstruction_high_frequency = _spectral_energy_above(
        reconstructed_amplitude, coherent_cutoff
    )
    brightfield_high_frequency = _spectral_energy_above(
        images_np[centre_index], coherent_cutoff
    )
    result.record(
        reconstruction_energy_above_cutoff=reconstruction_high_frequency,
        reconstruction_over_brightfield_high_frequency=(
            reconstruction_high_frequency / max(brightfield_high_frequency, 1e-30)
        ),
    )
    result.check_true(
        "the_reconstruction_recovers_frequencies_no_single_image_carries",
        "analytic",
        reconstruction_high_frequency > 3.0 * brightfield_high_frequency,
        f"spectral energy above the {coherent_cutoff:.4f} cycles/um single-image cutoff: "
        f"{100 * brightfield_high_frequency:.4f}% in the brightfield measurement versus "
        f"{100 * reconstruction_high_frequency:.4f}% in the reconstruction, a factor of "
        f"{reconstruction_high_frequency / max(brightfield_high_frequency, 1e-30):.1f}. "
        "Synthesising a wider aperture than any single measurement provides is the entire "
        "point of Fourier ptychography, and it is checkable with no reference value at all.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
