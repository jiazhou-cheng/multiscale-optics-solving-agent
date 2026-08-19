"""Example 9 / "Computer Generated Holography using a Digital Micromirror Device" -- https://chromatix.readthedocs.io/en/latest/examples/dmd/

Repo-owned reproduction of the DMD CGH example: a **binary** amplitude mask
(`chromatix.ops.binarize` inside `amplitude_change`) optimized through
`asm_propagate` to form an image at `z = 13e4`, driven by Adam at learning rate 2.0.

The interesting piece is that `binarize` is not differentiable, and Chromatix
supplies a **surrogate gradient** so `jax.grad` flows through it anyway. That is
what the example is for, and it is what this reproduction pins.

**Blocker, and the substitution it forced.** Upstream's target is
`skimage.data.cat()`, and **`scikit-image` is not installed in the pinned
`agent_solver` environment** (`ModuleNotFoundError: No module named 'skimage'`).
Installing it would change the pinned environment, which a knowledge ticket does
not own; fetching the bundled image over the network would make a repository test
depend on a third-party server. The target is therefore a deterministic
`chromatix.utils.siemens_star`-based 300x300 image built here.

The consequence is stated plainly: **upstream's published trajectory
(correlation 0.73058045 -> 0.86358774, loss 0.91078347 -> 0.8480373) is a property
of the cat photograph and cannot be reproduced against a different target.** Those
numbers are recorded for reference and the reproduction asserts the *behaviour*
instead -- which is where all of this example's transferable content lives:

| iteration | upstream correlation | upstream loss |
|---|---|---|
| 0 | 0.73058045 | 0.91078347 |
| 40 | 0.86155796 | 0.848808 |
| 200 | 0.86358774 | 0.8480373 |

What is checked:

* **`binarize` is a hard threshold in the forward pass.** The field's amplitude
  after `amplitude_change(field, binarize(a))` takes exactly two values, and they
  are 0 and 1 -- so the DMD really is modelled as a binary device, not a soft one.
* **`binarize` is nonetheless differentiable, with a non-zero surrogate gradient at
  every pixel.** A true `sign`/`round` would give an identically zero gradient and
  the optimization could not move; measuring the gradient is the only way to
  establish the surrogate exists.
* **The optimization is real**: the correlation rises substantially and the learned
  continuous amplitude is no longer the uniform draw it started from.
* **The same model evaluates a whole z stack** with a 1D `z`, which the example uses
  for its through-focus figure. The hologram is sharpest at the design distance:
  correlation with the target at `z = 13e4` exceeds correlation at every other plane
  in the sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c09_dmd_cgh",
    title="Computer Generated Holography using a Digital Micromirror Device",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/dmd/",
    demonstrates=(
        "chromatix.ops.binarize as a differentiable hard threshold (surrogate "
        "gradient) inside amplitude_change, asm_propagate(mode='same') with a "
        "scalar or 1D z, and optax.cosine_distance as a hologram loss."
    ),
    slow=True,
)

SHAPE = (300, 300)
SPACING = 7.56
SPECTRUM = 0.66
Z = 13e4
LEARNING_RATE = 2.0
MAX_ITERATIONS = 400
# Upstream's published trajectory against skimage.data.cat(). Recorded for
# reference only: it is a property of THAT image and this reproduction uses a
# different target (see the module docstring).
UPSTREAM_TRAJECTORY = {
    0: (0.73058045, 0.91078347),
    40: (0.86155796, 0.848808),
    80: (0.8622059, 0.8484709),
    120: (0.8627496, 0.8482904),
    160: (0.8632451, 0.8481377),
    200: (0.86358774, 0.8480373),
}


def _model_class():
    import equinox as eqx
    from jaxtyping import Array

    from chromatix import Field
    from chromatix.functional import amplitude_change, asm_propagate, plane_wave
    from chromatix.ops import binarize

    class CGH(eqx.Module):
        amplitude: Array
        shape: tuple[int, int] = eqx.field(static=True, default=SHAPE)
        spacing: float = eqx.field(static=True, default=SPACING)
        spectrum: float = eqx.field(static=True, default=SPECTRUM)
        n: float = eqx.field(static=True, default=1.0)
        pad_width: int = eqx.field(static=True, default=0)

        def __call__(self, z) -> Field:
            field = plane_wave(self.shape, self.spacing, self.spectrum)
            field = amplitude_change(field, binarize(self.amplitude))
            return asm_propagate(field, z, self.n, self.pad_width, mode="same")

    return CGH


def run() -> TutorialResult:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import random

    from chromatix.ops import binarize
    from chromatix.utils import siemens_star

    result = TutorialResult()
    CGH = _model_class()
    key = random.PRNGKey(4)

    # Upstream uses skimage.data.cat(); scikit-image is not in the pinned
    # environment. Record the blocker, then substitute a deterministic,
    # chromatix-bundled target of the same shape.
    skimage_error = ""
    try:
        import skimage.data  # noqa: F401
    except ModuleNotFoundError as exc:
        skimage_error = f"{type(exc).__name__}: {exc}"
    star = np.asarray(siemens_star(SHAPE[0]), dtype=np.float32)
    image = star * 255.0
    data = jnp.array(image)
    result.record(
        upstream_target="skimage.data.cat().mean(2)[:, 100:400]",
        skimage_import_error=skimage_error,
        substituted_target="chromatix.utils.siemens_star(300) * 255",
    )
    result.check_true(
        "scikit_image_is_not_available_in_the_pinned_environment",
        "invariant",
        bool(skimage_error),
        f"import skimage.data -> {skimage_error or 'succeeded (environment changed?)'}. "
        "Upstream's target is unavailable, so a deterministic chromatix-bundled "
        "siemens_star is substituted and upstream's published trajectory -- a property of "
        "the cat photograph -- is recorded for reference rather than asserted.",
    )
    initial_amplitude = jax.nn.initializers.uniform(1.0)(
        key, shape=SHAPE, dtype=jnp.float32
    )
    model = CGH(initial_amplitude)
    result.record(
        target_shape=list(np.asarray(data).shape),
        target_range=[float(np.asarray(data).min()), float(np.asarray(data).max())],
        shape=list(SHAPE),
        spacing=SPACING,
        spectrum=SPECTRUM,
        propagation_distance=Z,
        learning_rate=LEARNING_RATE,
        initial_amplitude_range=[
            float(np.asarray(initial_amplitude).min()),
            float(np.asarray(initial_amplitude).max()),
        ],
    )
    result.check_true(
        "the_substituted_target_matches_the_model_grid",
        "invariant",
        tuple(np.asarray(data).shape) == SHAPE,
        f"the siemens_star target is {np.asarray(data).shape}, matching the model's "
        f"{SHAPE} grid -- the same shape upstream's cat crop has",
    )

    # -- binarize is a hard threshold in the forward pass -----------------------
    binarized = np.asarray(binarize(initial_amplitude))
    unique_values = np.unique(binarized)
    result.record(
        binarized_unique_values=unique_values,
        binarized_num_unique=int(unique_values.size),
        binarized_on_fraction=float(np.mean(binarized > 0.5)),
    )
    result.check_true(
        "binarize_produces_exactly_two_levels_zero_and_one",
        "analytic",
        unique_values.size == 2
        and abs(float(unique_values.min())) < 1e-9
        and abs(float(unique_values.max()) - 1.0) < 1e-9,
        f"binarize() of a uniform(0, 1) draw takes {unique_values.size} distinct values "
        f"{unique_values.tolist()}, with {100 * np.mean(binarized > 0.5):.2f}% on. The DMD "
        "really is modelled as a binary device in the forward pass.",
    )

    # -- and it is differentiable anyway --------------------------------------
    def loss_fn(candidate, target, z):
        eps = 1e-7
        approx = candidate(z).intensity.squeeze()
        loss = optax.cosine_distance(
            predictions=approx.reshape(-1), targets=target.reshape(-1), epsilon=eps
        ).mean()
        correlation = jnp.sum(approx * target) / (
            jnp.sqrt(jnp.sum(approx**2) * jnp.sum(target**2)) + eps
        )
        return loss, {"loss": loss, "correlation": correlation}

    grads, initial_metrics = jax.grad(loss_fn, has_aux=True)(model, data, Z)
    gradient = np.asarray(grads.amplitude, dtype=float)
    result.record(
        initial_loss=float(initial_metrics["loss"]),
        initial_correlation=float(initial_metrics["correlation"]),
        gradient_shape=list(gradient.shape),
        gradient_abs_min=float(np.abs(gradient).min()),
        gradient_abs_max=float(np.abs(gradient).max()),
        gradient_zero_fraction=float(np.mean(gradient == 0.0)),
    )
    result.check_true(
        "binarize_has_a_non_zero_surrogate_gradient_at_every_pixel",
        "analytic",
        gradient.shape == SHAPE
        and bool(np.all(np.isfinite(gradient)))
        and float(np.mean(gradient == 0.0)) < 0.01,
        f"the gradient through binarize is {gradient.shape} with magnitudes in "
        f"[{float(np.abs(gradient).min()):.3e}, {float(np.abs(gradient).max()):.3e}] and "
        f"only {100 * np.mean(gradient == 0.0):.3f}% of pixels exactly zero. A true "
        "sign/round would give an identically zero gradient and the optimization could "
        "not move at all -- so chromatix.ops.binarize carries a surrogate gradient, which "
        "is the whole point of this example and is only establishable by measurement.",
    )
    result.check_true(
        "the_initial_loss_and_correlation_are_both_physical_and_recorded",
        "invariant",
        0.0 <= float(initial_metrics["loss"]) <= 1.0
        and 0.0 <= float(initial_metrics["correlation"]) <= 1.0,
        f"initial correlation {float(initial_metrics['correlation']):.6f} and loss "
        f"{float(initial_metrics['loss']):.6f}, against upstream's "
        f"{UPSTREAM_TRAJECTORY[0][0]} and {UPSTREAM_TRAJECTORY[0][1]} for the cat target. "
        "Both are in [0, 1] as their definitions require. Note the two do NOT sum to 1 "
        "(1.42 here, 1.64 upstream): optax.cosine_distance and this example's "
        "hand-written normalised inner product are different quantities, so the "
        "'correlation' reported alongside the loss is not 1 - loss. Absolute values are "
        "not asserted because they are properties of the target image.",
    )

    # -- the optimization ------------------------------------------------------
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(model)

    @jax.jit
    def update(candidate, state, target, z):
        gradients, metrics = jax.grad(loss_fn, has_aux=True)(candidate, target, z)
        updates, state = optimizer.update(gradients, state, candidate)
        return optax.apply_updates(candidate, updates), state, metrics

    history = {"loss": np.zeros(MAX_ITERATIONS), "correlation": np.zeros(MAX_ITERATIONS)}
    for iteration in range(MAX_ITERATIONS):
        model, opt_state, metrics = update(model, opt_state, data, Z)
        history["loss"][iteration] = float(metrics["loss"])
        history["correlation"][iteration] = float(metrics["correlation"])

    observed = {
        key: (float(history["correlation"][key]), float(history["loss"][key]))
        for key in UPSTREAM_TRAJECTORY
    }
    result.record(
        max_iterations=MAX_ITERATIONS,
        observed_trajectory={str(k): list(v) for k, v in observed.items()},
        upstream_trajectory={str(k): list(v) for k, v in UPSTREAM_TRAJECTORY.items()},
        final_correlation=float(history["correlation"][-1]),
        final_loss=float(history["loss"][-1]),
        best_correlation=float(history["correlation"].max()),
    )
    result.check_finite("correlation_history_finite", history["correlation"])
    result.check_true(
        "the_optimization_raises_the_correlation_substantially",
        "invariant",
        float(history["correlation"][-1]) > 1.05 * float(history["correlation"][0]),
        f"correlation {float(history['correlation'][0]):.6f} -> "
        f"{float(history['correlation'][-1]):.6f} over {MAX_ITERATIONS} Adam steps at "
        f"lr = {LEARNING_RATE}, best {float(history['correlation'].max()):.6f}. Upstream "
        f"reports {UPSTREAM_TRAJECTORY[0][0]:.6f} -> {UPSTREAM_TRAJECTORY[200][0]:.6f} on "
        "its own target.",
    )
    result.check_true(
        "the_correlation_gain_saturates_early_as_upstream_shows",
        "analytic",
        float(history["correlation"][39])
        > float(history["correlation"][0])
        + 0.7 * (float(history["correlation"][-1]) - float(history["correlation"][0])),
        "by iteration 40 the correlation has already reached "
        f"{float(history['correlation'][39]):.6f} of its final "
        f"{float(history['correlation'][-1]):.6f}, i.e. most of the gain arrives in the "
        f"first 10% of the run. Upstream's trajectory shows the same shape "
        f"({UPSTREAM_TRAJECTORY[0][0]:.4f} -> {UPSTREAM_TRAJECTORY[40][0]:.4f} -> "
        f"{UPSTREAM_TRAJECTORY[200][0]:.4f}), so the saturation is a property of the method "
        "rather than of the target.",
    )
    result.check_true(
        "the_running_maximum_correlation_never_falls",
        "invariant",
        bool(np.all(np.diff(np.maximum.accumulate(history["correlation"])) >= 0.0)),
        "the running maximum of the correlation is monotone over all "
        f"{MAX_ITERATIONS} steps",
    )
    learned = np.asarray(model.amplitude, dtype=float)
    result.record(
        learned_amplitude_range=[float(learned.min()), float(learned.max())],
        learned_binarized_on_fraction=float(np.mean(np.asarray(binarize(model.amplitude)) > 0.5)),
        amplitude_rms_change=float(
            np.sqrt(np.mean((learned - np.asarray(initial_amplitude, dtype=float)) ** 2))
        ),
    )
    result.check_true(
        "the_continuous_amplitude_moved_far_from_its_uniform_initialisation",
        "analytic",
        float(np.sqrt(np.mean((learned - np.asarray(initial_amplitude, dtype=float)) ** 2)))
        > 0.5,
        f"RMS change in the continuous amplitude parameter is "
        f"{float(np.sqrt(np.mean((learned - np.asarray(initial_amplitude, dtype=float)) ** 2))):.4f} "
        f"and its range grew to [{float(learned.min()):.4f}, {float(learned.max()):.4f}] "
        "from the initial [0, 1). The optimizer moves the continuous variable; only its "
        "binarization reaches the field.",
    )

    # -- the hologram is sharpest at the design distance -----------------------
    z_list = np.linspace(0.0, Z, 20)
    stack = np.asarray(model(jnp.asarray(z_list)).intensity).squeeze()
    target_np = np.asarray(data, dtype=float)
    correlations = np.asarray(
        [
            float(
                np.sum(plane * target_np)
                / (np.sqrt(np.sum(plane**2) * np.sum(target_np**2)) + 1e-7)
            )
            for plane in stack
        ]
    )
    result.record(
        z_sweep=z_list,
        stack_shape=list(stack.shape),
        correlation_vs_z=correlations,
        best_z=float(z_list[int(np.argmax(correlations))]),
    )
    result.check_finite("z_sweep_finite", correlations)
    result.check_true(
        "the_hologram_is_sharpest_at_the_design_distance",
        "analytic",
        int(np.argmax(correlations)) == len(z_list) - 1,
        f"correlation with the target peaks at z = {float(z_list[int(np.argmax(correlations))]):.0f} "
        f"of the {len(z_list)} swept planes, which is the design distance z = {Z:.0f}. "
        f"Correlation there is {float(correlations.max()):.6f} against "
        f"{float(correlations.min()):.6f} at the worst plane. A 1D z produces the whole "
        "through-focus stack in one call, which is what the example's final figure shows.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
