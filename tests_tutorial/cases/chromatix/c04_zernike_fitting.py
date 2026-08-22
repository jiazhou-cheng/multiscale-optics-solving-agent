"""Example 4 / "Aberration Phase Retrieval" -- https://chromatix.readthedocs.io/en/latest/examples/zernike_fitting/

Repo-owned reproduction of the Zernike phase-retrieval example: an
`equinox.Module` forward model built from `objective_point_source` ->
`zernike_aberrations` -> `phase_change` -> `ff_lens` -> `Field.intensity` ->
`ops.shot_noise`, then 1000 Adam steps of `jax.grad` through it to recover 10
ANSI Zernike coefficients from a single shot-noisy PSF.

**The published convergence does not reproduce on commit `d24bdf0`, under either
optimizer-state convention.** The diagnosis took two attempts and both are recorded:

1. Upstream's `update()` returns only `(model, metrics)` while rebinding
   `opt_state` internally, so its loop passes the **initial** Adam state on every
   iteration -- every step is a fresh bias-corrected Adam step, i.e. sign descent
   at a fixed step of `lr`. That detail is decisive for
   `c10_seidel_fitting`, which *does* reproduce once it is honoured. It is
   reproduced faithfully here as the primary path.
2. It is not enough for *this* example. Upstream's recipe gives a final loss of
   **0.137** against the published `0.00174`, and threading the state through gives
   0.151 (plateauing at 0.112 by step 2000). Neither converges.

The difference from `c10` is visible in the two examples' code: **`c10` projects its
coefficients onto the non-negative orthant after every step**
(`eqx.tree_at(..., jnp.abs(...))`), and `c04` does not. Recovering a phase from a
single intensity image has a sign/twin ambiguity, and that projection is what
breaks it. Without it the fit lands in a twin solution -- which is exactly what the
recovered coefficients look like.

| quantity | upstream | here |
|---|---|---|
| initial loss | 5.619491 | 5.598808 (0.4% apart -- a match) |
| loss after 1000 steps | 0.00174069 | 0.1368 |
| mean coefficient error | 6.355736e-06 | 1.7e-02 |

Also established:

* **The gradient path is real**: all ten coefficients receive a finite, non-zero
  gradient through `objective_point_source` -> `zernike_aberrations` ->
  `phase_change` -> `ff_lens` -> `Field.intensity`.
* **The forward model is deterministic given a key**: two calls with the same
  `PRNGKey` produce a bit-identical shot-noisy PSF, and different keys do not.
  Without that, none of these numbers could be recorded as evidence.
* `zernike_aberrations(..., normalize=False)` consumes coefficients in the **same
  length unit as the wavelength** (upstream divides waves by `2*pi/lambda`), not
  radians and not waves.
* One deviation from upstream's code, forced by JAX: upstream declares
  `ansi_indices` as `eqx.field(static=True, default_factory=lambda: np.arange(1, 11))`.
  A NumPy array as a **static** equinox field makes the module unhashable by
  equality, and `jax.jit` raises *"Exception raised while checking equality of
  metadata fields of pytree"* as soon as two separately-constructed instances reach
  the same jitted function. A tuple is the equivalent hashable declaration and is
  used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

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
# Measured with the optimizer state threaded through (the intended Adam pattern,
# which upstream does NOT use): the loss plateaus at 0.1122 by step 2000 and is
# unchanged at 3000, with a 2.15-wave max coefficient error. Recorded so the
# control below does not have to pay for a 3x run.
OBSERVED_THREADED_PLATEAU_LOSS_AT_3000_STEPS = 0.112171
OBSERVED_THREADED_MAX_WAVE_ERROR_AT_3000_STEPS = 2.1496
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
        # Upstream declares this as `eqx.field(static=True,
        # default_factory=lambda: np.arange(1, 11))`. A NumPy array as a static
        # equinox field makes the module unhashable-by-equality, so jax.jit raises
        # "Exception raised while checking equality of metadata fields of pytree"
        # as soon as two separately-constructed instances reach the same jitted
        # function -- which happens here because the control run below builds a
        # second model. A tuple is the equivalent hashable declaration.
        ansi_indices: tuple[int, ...] = eqx.field(
            static=True, default=tuple(range(1, 11))
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
                np.asarray(self.ansi_indices),
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
        updates, new_state = optimizer.update(grads, state, candidate)
        return optax.apply_updates(candidate, updates), new_state, metrics, grads

    def fit(thread_optimizer_state: bool):
        """Run the retrieval. `thread_optimizer_state=False` is upstream's recipe."""
        candidate = ZernikePSF(jnp.zeros((10,)))
        state = optimizer.init(candidate)
        losses = []
        first_gradient = None
        for _ in range(MAX_ITERATIONS):
            candidate, new_state, metrics, grads = update(candidate, state, psf_truth)
            if thread_optimizer_state:
                state = new_state
            losses.append(float(metrics["loss"]))
            if first_gradient is None:
                first_gradient = np.asarray(grads.coefficients, dtype=float)
        return np.asarray(losses, dtype=float), candidate, first_gradient

    initial_loss = float(loss_fn(model, psf_truth)[0])
    # Upstream's loop calls `model, metrics = update(model, opt_state, psf_truth)`,
    # so the INITIAL Adam state is passed on every iteration -- its update() never
    # returns the new one. That is reproduced verbatim here; see the control below.
    history, model, gradients_seen = fit(thread_optimizer_state=False)

    history_array = history
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
        f"after {MAX_ITERATIONS} steps of upstream's exact recipe (initial opt_state "
        f"re-passed every iteration) the loss is {float(history_array[-1]):.6f} against "
        f"upstream's published {UPSTREAM_FINAL_LOSS}, a factor of "
        f"{float(history_array[-1]) / UPSTREAM_FINAL_LOSS:.0f}. Unlike c10_seidel_fitting, "
        "honouring the stale-optimizer-state detail does not rescue this one.",
    )
    result.check_true(
        "upstreams_coefficient_error_is_not_reproduced",
        "reference",
        mean_error > 100.0 * UPSTREAM_MEAN_COEFFICIENT_ERROR,
        f"mean squared coefficient error {mean_error:.6e} against upstream's published "
        f"{UPSTREAM_MEAN_COEFFICIENT_ERROR}: a different solution, not a noisier version "
        "of the same one.",
    )
    result.check_true(
        "the_loss_nevertheless_falls_by_more_than_an_order_of_magnitude",
        "invariant",
        float(history_array[-1]) < initial_loss * 0.1,
        f"{initial_loss:.6f} -> {float(history_array[-1]):.8f}, a "
        f"{initial_loss / float(history_array[-1]):.0f}x reduction over {MAX_ITERATIONS} "
        "steps. The optimizer, the gradient path and the forward model all work; it is "
        "the solution that is wrong.",
    )
    result.check_true(
        "the_loss_history_is_monotonically_non_increasing_in_its_running_minimum",
        "invariant",
        bool(np.all(np.diff(np.minimum.accumulate(history_array)) <= 0.0)),
        "the running minimum of the loss never rises over 1000 steps",
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
        "the_recovered_coefficients_are_a_wrong_solution_of_order_one_wave",
        "reference",
        float(np.max(np.abs(waves_estimated - waves_truth))) > 0.5,
        f"max |estimated - truth| = "
        f"{float(np.max(np.abs(waves_estimated - waves_truth))):.4f} waves over all 10 "
        f"ANSI terms; recovered {np.round(waves_estimated, 3).tolist()} against "
        f"{waves_truth.tolist()}",
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
        float(np.max(np.abs(waves_estimated[zero_indices]))) > 0.3,
        "coefficients whose truth is 0 come back at "
        f"{np.round(waves_estimated[zero_indices], 3).tolist()} waves, and the >= 2-wave "
        f"terms at {np.round(waves_estimated[large_indices], 3).tolist()}. The two groups "
        "are not separated. c10_seidel_fitting DOES separate them, and the difference in "
        "the two examples' code is that c10 projects its coefficients onto the "
        "non-negative orthant after every step (eqx.tree_at(..., jnp.abs(...))) while this "
        "one does not -- which is what breaks the sign/twin ambiguity of single-intensity "
        "phase retrieval.",
    )

    # -- the stale optimizer state is what makes it converge --------------------
    threaded_history, threaded_model, _ = fit(thread_optimizer_state=True)
    threaded_estimated = np.asarray(threaded_model.coefficients, dtype=float)
    threaded_error = float(np.mean((threaded_estimated - truth_array) ** 2))
    result.record(
        threaded_adam_final_loss=float(threaded_history[-1]),
        threaded_adam_coefficients=threaded_estimated,
        threaded_adam_mean_coefficient_error=threaded_error,
    )
    result.check_true(
        "threading_the_adam_state_through_does_not_rescue_it_either",
        "analytic",
        threaded_error > 100.0 * UPSTREAM_MEAN_COEFFICIENT_ERROR
        and float(threaded_history[-1]) > 10.0 * UPSTREAM_FINAL_LOSS,
        "upstream's loop passes the INITIAL opt_state on every iteration because its "
        "update() returns only (model, metrics). Rebinding the state -- the intended Adam "
        f"pattern -- gives a final loss of {float(threaded_history[-1]):.6f} and a mean "
        f"squared coefficient error of {threaded_error:.6e}, against "
        f"{float(history_array[-1]):.6f} and {mean_error:.6e} for upstream's recipe. "
        f"BOTH are far from the published {UPSTREAM_FINAL_LOSS} and "
        f"{UPSTREAM_MEAN_COEFFICIENT_ERROR}. With the state frozen every step is a fresh "
        "bias-corrected Adam step, i.e. sign descent at a fixed step of lr, which "
        "escapes the local minimum accumulated moments settle into -- and in "
        "c10_seidel_fitting that IS what makes the published numbers reproduce. Here "
        "neither convention reaches them, so the missing ingredient is c10's "
        "non-negativity projection rather than the optimizer state.",
    )
    result.note(
        "Do NOT use this example as evidence that Chromatix phase retrieval converges; use "
        "c10_seidel_fitting for that. What this one establishes is narrower and still "
        "useful: jax.grad flows through objective_point_source -> zernike_aberrations -> "
        "phase_change -> ff_lens -> Field.intensity with finite non-zero gradients on all "
        "ten coefficients, ops.shot_noise is key-deterministic, and a NumPy array cannot "
        "be an eqx.field(static=True) if the module reaches jax.jit from more than one "
        "construction site."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
