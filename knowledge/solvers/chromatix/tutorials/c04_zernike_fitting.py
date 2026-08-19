"""Example 4 / "Aberration Phase Retrieval" -- https://chromatix.readthedocs.io/en/latest/examples/zernike_fitting/

Repo-owned reproduction of the Zernike phase-retrieval example: an
`equinox.Module` forward model built from `objective_point_source` ->
`zernike_aberrations` -> `phase_change` -> `ff_lens` -> `Field.intensity` ->
`ops.shot_noise`, then 1000 Adam steps of `jax.grad` through it to recover 10
ANSI Zernike coefficients from a single shot-noisy PSF.

**This example publishes a full loss history and a final error, and the
convergence does NOT reproduce in the pinned environment.** That is the headline
finding, and it is asserted rather than hidden:

| quantity | upstream | pinned `d24bdf0` |
|---|---|---|
| initial loss | 5.619491 | 5.598808 (0.4% apart -- a match) |
| loss after 1000 steps | 0.00174069 | 0.150986 |
| mean coefficient error | 6.355736e-06 | 1.32e-02 |
| max coefficient error | (not stated; ~0.003 waves implied) | 2.14 waves |

Running upstream's recipe **three times longer** does not close the gap: the loss
plateaus at 0.1122 by step 2000 and is 0.1122 at step 3000, with the max
coefficient error stuck at 2.15 waves. So this is a **local minimum, not an
iteration shortfall** -- recovering a phase from a single intensity image is
non-convex, and the published page was built from a different commit (the same
commit difference that makes 101's printed field power disagree in its last
digits).

What *does* reproduce, and is asserted:

* The **initial loss** matches upstream to 0.4%, so the forward model and the
  measurement it is fitting are the same problem.
* **The gradient path is real**: all ten coefficients receive a finite, non-zero
  gradient through `objective_point_source` -> `zernike_aberrations` ->
  `phase_change` -> `ff_lens` -> `Field.intensity`.
* **The loss does fall substantially** -- 37x over 1000 steps, 50x over 3000 --
  so the machinery works; it is the *solution* that is wrong.
* **The forward model is deterministic given a key**: two calls with the same
  `PRNGKey` produce a bit-identical shot-noisy PSF, and different keys do not.
  Without that, none of these numbers could be recorded as evidence.
* `zernike_aberrations(..., normalize=False)` consumes coefficients in the **same
  length unit as the wavelength** (upstream divides waves by `2*pi/lambda`), not
  radians and not waves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c04_zernike_fitting",
    title="Aberration Phase Retrieval (Zernike Fitting)",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/zernike_fitting/",
    demonstrates=(
        "chromatix.functional.objective_point_source / phase_change / ff_lens, "
        "chromatix.utils.zernike_aberrations(ansi_indices, coefficients, "
        "normalize=False), chromatix.ops.shot_noise, and jax.grad through an "
        "equinox.Module forward model driven by optax.adam."
    ),
    slow=True,
)

MAX_ITERATIONS = 1000
UPSTREAM_INITIAL_LOSS = 5.619491
UPSTREAM_FINAL_LOSS = 0.00174069
UPSTREAM_MEAN_COEFFICIENT_ERROR = 6.355736e-06
# Measured by running upstream's recipe for 3000 steps (414 s): the loss plateaus
# by step 2000 and does not improve, so the shortfall is a local minimum rather
# than an iteration limit. Recorded here so the reproduction can assert the
# diagnosis without paying for the 3x run every time.
OBSERVED_PLATEAU_LOSS_AT_3000_STEPS = 0.112171
OBSERVED_MAX_WAVE_ERROR_AT_3000_STEPS = 2.1496
TRUTH_WAVES = (2.0, 5.0, 3.0, 0, 1, 0, 1, 0, 1, 0)


def _model_class():
    import equinox as eqx
    import jax.numpy as jnp
    from jaxtyping import Array, PRNGKeyArray

    from chromatix.functional import ff_lens, objective_point_source, phase_change
    from chromatix.ops import shot_noise
    from chromatix.utils import zernike_aberrations

    class ZernikePSF(eqx.Module):
        coefficients: Array
        ansi_indices: Array = eqx.field(
            static=True, default_factory=lambda: np.arange(1, 11)
        )
        camera_shape: tuple[int, int] = eqx.field(static=True, default=(256, 256))
        camera_pixel_pitch: float = eqx.field(static=True, default=0.125)
        f: float = eqx.field(static=True, default=100.0)
        NA: float = eqx.field(static=True, default=0.8)
        n: float = eqx.field(static=True, default=1.33)
        wavelength: float = eqx.field(static=True, default=0.532)
        upsample: int = eqx.field(static=True, default=4)
        pad: int = eqx.field(static=True, default=128)

        def __call__(self, key: PRNGKeyArray | None = None) -> Array:
            shape = tuple(np.array(self.camera_shape) * self.upsample + self.pad)
            spacing = (
                self.upsample
                * self.f
                * self.wavelength
                / (self.n * shape[0] * self.camera_pixel_pitch)
            )
            field = objective_point_source(
                shape, spacing, self.wavelength, 0.0, self.f, self.n, self.NA, power=1e3
            )
            aberrations = zernike_aberrations(
                shape,
                spacing,
                self.wavelength,
                self.n,
                self.f,
                self.NA,
                self.ansi_indices,
                self.coefficients,
                normalize=False,
            )
            field = phase_change(field, aberrations)
            field = ff_lens(field, self.f, self.n)
            image = field.intensity
            if key is not None:
                image = shot_noise(key, image)
            return image

    return ZernikePSF


def run() -> TutorialResult:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import random

    result = TutorialResult()
    ZernikePSF = _model_class()
    key = random.PRNGKey(42)

    model = ZernikePSF(jnp.zeros((10,)))
    coefficients_truth = jnp.array(TRUTH_WAVES) / (2 * jnp.pi / model.wavelength)
    true_model = ZernikePSF(coefficients_truth)
    psf_truth = true_model(key=key)

    # -- the forward model must be deterministic given a key -------------------
    replay = ZernikePSF(coefficients_truth)(key=random.PRNGKey(42))
    other = ZernikePSF(coefficients_truth)(key=random.PRNGKey(1))
    same_key_identical = bool(np.array_equal(np.asarray(psf_truth), np.asarray(replay)))
    other_key_identical = bool(np.array_equal(np.asarray(psf_truth), np.asarray(other)))
    result.record(
        psf_shape=list(np.asarray(psf_truth).shape),
        psf_total_counts=float(np.asarray(psf_truth).sum()),
        coefficients_truth_length_units=np.asarray(coefficients_truth, dtype=float),
        truth_waves=list(TRUTH_WAVES),
        same_key_reproducible=same_key_identical,
        different_key_reproducible=other_key_identical,
    )
    result.check_true(
        "the_shot_noisy_forward_model_is_reproducible_given_a_key",
        "invariant",
        same_key_identical and not other_key_identical,
        f"PRNGKey(42) twice gives a bit-identical PSF ({same_key_identical}); PRNGKey(1) "
        f"does not ({other_key_identical}). ops.shot_noise draws only from the key it is "
        "handed, so the whole retrieval is replayable.",
    )
    result.check_finite("truth_psf_finite", np.asarray(psf_truth))
    result.check_true(
        "the_coefficients_are_in_length_units_not_waves_or_radians",
        "analytic",
        bool(
            np.allclose(
                np.asarray(coefficients_truth, dtype=float),
                np.asarray(TRUTH_WAVES) * model.wavelength / (2 * np.pi),
                rtol=1e-6,
            )
        ),
        "upstream divides its wave-valued coefficients by 2*pi/lambda, so "
        f"{TRUTH_WAVES[1]} waves becomes "
        f"{float(np.asarray(coefficients_truth)[1]):.6f} in the same length unit as the "
        f"{model.wavelength} um wavelength. zernike_aberrations(normalize=False) consumes "
        "that unit, not radians and not waves.",
    )

    # -- the optimization ------------------------------------------------------
    def loss_fn(candidate, data):
        estimate = candidate()
        loss = jnp.mean((estimate - data) ** 2) / jnp.mean(data**2)
        return loss, {"loss": loss}

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(model)

    @jax.jit
    def update(candidate, state, data):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(candidate, data)
        updates, state = optimizer.update(grads, state, candidate)
        return optax.apply_updates(candidate, updates), state, metrics, grads

    initial_loss = float(loss_fn(model, psf_truth)[0])
    history = []
    gradients_seen = None
    for _ in range(MAX_ITERATIONS):
        model, opt_state, metrics, grads = update(model, opt_state, psf_truth)
        history.append(float(metrics["loss"]))
        if gradients_seen is None:
            gradients_seen = np.asarray(grads.coefficients, dtype=float)

    history_array = np.asarray(history, dtype=float)
    coefficients_estimated = np.asarray(model.coefficients, dtype=float)
    truth_array = np.asarray(coefficients_truth, dtype=float)
    coefficient_error = (coefficients_estimated - truth_array) ** 2
    mean_error = float(coefficient_error.mean())

    result.record(
        max_iterations=MAX_ITERATIONS,
        initial_loss=initial_loss,
        loss_at_iterations=[
            float(history_array[i]) for i in (0, 99, 199, 299, 399, 499, 999)
        ],
        final_loss=float(history_array[-1]),
        best_loss=float(history_array.min()),
        coefficients_estimated=coefficients_estimated,
        coefficient_squared_error=coefficient_error,
        mean_coefficient_error=mean_error,
        first_gradient=gradients_seen,
    )
    result.check_finite("loss_history_finite", history_array)
    result.check_close(
        "initial_loss_matches_upstreams_5p619",
        "reference",
        initial_loss,
        UPSTREAM_INITIAL_LOSS,
        rel=0.02,
    )
    result.check_true(
        "upstreams_converged_loss_is_not_reproduced",
        "reference",
        float(history_array[-1]) > 10.0 * UPSTREAM_FINAL_LOSS,
        f"after {MAX_ITERATIONS} steps the loss is {float(history_array[-1]):.6f} against "
        f"upstream's published {UPSTREAM_FINAL_LOSS}, a factor of "
        f"{float(history_array[-1]) / UPSTREAM_FINAL_LOSS:.0f}. Running the same recipe "
        f"for 3000 steps reaches only {OBSERVED_PLATEAU_LOSS_AT_3000_STEPS} and is flat "
        "from step 2000, so this is a local minimum of a non-convex single-intensity "
        "phase retrieval, not an iteration shortfall.",
    )
    result.check_true(
        "upstreams_coefficient_error_is_not_reproduced",
        "reference",
        mean_error > 100.0 * UPSTREAM_MEAN_COEFFICIENT_ERROR,
        f"mean squared coefficient error {mean_error:.6e} against upstream's published "
        f"{UPSTREAM_MEAN_COEFFICIENT_ERROR}. The recovered coefficients are a different "
        "solution, not a noisier version of the same one.",
    )
    result.check_true(
        "the_loss_nevertheless_falls_by_more_than_an_order_of_magnitude",
        "invariant",
        float(history_array[-1]) < initial_loss * 0.1,
        f"{initial_loss:.6f} -> {float(history_array[-1]):.8f}, a "
        f"{initial_loss / float(history_array[-1]):.0f}x reduction over {MAX_ITERATIONS} "
        "Adam steps. The optimizer, the gradient path and the forward model all work; "
        "it is the solution that is wrong.",
    )
    result.check_true(
        "the_loss_history_is_monotonically_non_increasing_in_its_running_minimum",
        "invariant",
        bool(np.all(np.diff(np.minimum.accumulate(history_array)) <= 0.0)),
        "the running minimum of the loss never rises over 1000 Adam steps",
    )
    result.check_true(
        "every_coefficient_receives_a_finite_gradient",
        "analytic",
        gradients_seen is not None
        and gradients_seen.size == 10
        and bool(np.all(np.isfinite(gradients_seen)))
        and bool(np.all(np.abs(gradients_seen) > 0.0)),
        f"all 10 coefficient gradients finite and non-zero at the first step: "
        f"|grad| in [{np.abs(gradients_seen).min():.3e}, {np.abs(gradients_seen).max():.3e}]. "
        "jax.grad flows through objective_point_source -> zernike_aberrations -> "
        "phase_change -> ff_lens -> intensity.",
    )
    waves_estimated = coefficients_estimated * (2 * np.pi / model.wavelength)
    waves_truth = np.asarray(TRUTH_WAVES, dtype=float)
    result.record(
        coefficients_estimated_waves=waves_estimated,
        max_abs_wave_error=float(np.max(np.abs(waves_estimated - waves_truth))),
    )
    result.check_true(
        "the_recovered_coefficients_are_a_wrong_solution_of_order_two_waves",
        "reference",
        1.0 < float(np.max(np.abs(waves_estimated - waves_truth))) < 4.0,
        f"max |estimated - truth| = "
        f"{float(np.max(np.abs(waves_estimated - waves_truth))):.4f} waves over all 10 "
        f"ANSI terms; recovered {np.round(waves_estimated, 3).tolist()} against "
        f"{waves_truth.tolist()}. At 3000 steps this is "
        f"{OBSERVED_MAX_WAVE_ERROR_AT_3000_STEPS} waves, i.e. unchanged.",
    )
    zero_indices = [i for i, value in enumerate(TRUTH_WAVES) if value == 0]
    large_indices = [i for i, value in enumerate(TRUTH_WAVES) if value >= 2]
    result.record(
        zero_truth_indices=zero_indices,
        recovered_at_zero_truth_waves=waves_estimated[zero_indices],
        recovered_at_large_truth_waves=waves_estimated[large_indices],
    )
    result.check_true(
        "the_fit_does_not_separate_present_from_absent_aberrations",
        "reference",
        float(np.max(np.abs(waves_estimated[zero_indices]))) > 0.5,
        "coefficients whose truth is 0 come back at "
        f"{np.round(waves_estimated[zero_indices], 3).tolist()} waves, and the >= 2-wave "
        f"terms at {np.round(waves_estimated[large_indices], 3).tolist()}. The two groups "
        "are not separated, which is the sharpest statement of why this is the wrong "
        "solution rather than a slightly noisy right one -- and it is exactly the check "
        "that would have caught it upstream.",
    )
    result.note(
        "Do NOT use this example as evidence that Chromatix phase retrieval converges. "
        "What it establishes for this repository is narrower and still useful: jax.grad "
        "flows through objective_point_source -> zernike_aberrations -> phase_change -> "
        "ff_lens -> Field.intensity with finite non-zero gradients on all ten "
        "coefficients, and ops.shot_noise is key-deterministic. The published "
        "convergence is not reproducible on commit d24bdf0."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
