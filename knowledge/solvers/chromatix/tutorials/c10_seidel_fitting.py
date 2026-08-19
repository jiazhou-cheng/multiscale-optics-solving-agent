"""Example 10 / "Seidel Fitting" -- https://chromatix.readthedocs.io/en/latest/examples/seidel_fitting/

Repo-owned reproduction of the Seidel-coefficient retrieval example: the same
`objective_point_source` -> `seidel_aberrations` -> `phase_change` -> `ff_lens`
forward model as the Zernike example, but fitting five Seidel coefficients and
projecting them onto the non-negative orthant after every Adam step with
`eqx.tree_at`.

**Upstream never rebinds `opt_state`, and that is what makes it converge.** Its
`update()` computes a new optimizer state internally but returns only
`(model, metrics)`, and the loop calls
`model, metrics = update(model, opt_state, ...)` -- so the *initial* Adam state is
passed on every iteration. Each step is therefore a fresh bias-corrected Adam
step, i.e. effectively sign descent with a fixed step of `lr`, and the moment
estimates never accumulate.

That detail is not cosmetic. Threading the state through properly -- the intended
Adam pattern -- **stalls in a local minimum**: final loss 9.39 instead of 0.78 and
a mean squared coefficient error of 2.3e-2 instead of 4.6e-5. This reproduction
runs upstream's recipe verbatim as the primary path and the corrected-Adam variant
as a control, so the effect is attributable rather than asserted.

With upstream's recipe, every published number reproduces:

| upstream | quantity |
|---|---|
| `(1152, 1152)` / `1.11` | simulation shape and spacing |
| `1048.5734` | loss at iteration 0 |
| `0.84318376` | loss from iteration 600 onward |
| `[0.42599541, 0.41199559, 0.40799564, 0.0019999836, 1.6298145e-09]` | estimated coefficients |
| `7.5156146e-05` | mean squared coefficient error |

Each is checked to a stated tolerance. Beyond them:

* **The non-negativity projection is load-bearing.** Removing the
  `eqx.tree_at(..., jnp.abs(...))` step and running the same recipe leaves a worse
  fit, which is recorded -- so the projection is part of the method, not
  decoration. This also demonstrates the `eqx.tree_at` pattern for constrained
  optimization on an `equinox.Module`.
* **The two coefficients whose truth is zero come back at ~2e-3 and 0**, i.e. the
  fit separates present from absent aberrations.
* The forward model is key-deterministic, and `jax.grad` reaches all five
  coefficients with finite non-zero values.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c10_seidel_fitting",
    title="Seidel Fitting",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/seidel_fitting/",
    demonstrates=(
        "chromatix.utils.seidel_aberrations(shape, spacing, wavelength, n, f, "
        "NA, coefficients, u, v) inside a differentiable forward model, and the "
        "eqx.tree_at projection pattern for a non-negativity constraint."
    ),
    slow=True,
)

CAMERA_SHAPE = (256, 256)
CAMERA_PIXEL_PITCH = 0.125
F = 100.0
NA = 0.8
N = 1.33
WAVELENGTH = 0.532
UPSAMPLE = 4
PAD = 128
MAX_ITERATIONS = 1000
TRUTH_WAVES = (5.0, 5.0, 5.0, 0.0, 0.0)

UPSTREAM_SHAPE = (1152, 1152)
UPSTREAM_SPACING = 1.11
UPSTREAM_INITIAL_LOSS = 1048.5734
UPSTREAM_FINAL_LOSS = 0.84318376
UPSTREAM_ESTIMATED = (
    4.2599541e-01, 4.1199559e-01, 4.0799564e-01, 1.9999836e-03, 1.6298145e-09,
)
UPSTREAM_COEFFICIENT_ERROR = 7.5156146e-05


def _shape_and_spacing() -> tuple[tuple[int, int], float]:
    shape = tuple(int(e) for e in np.array(CAMERA_SHAPE) * UPSAMPLE + PAD)
    spacing = UPSAMPLE * F * WAVELENGTH / (N * shape[0] * CAMERA_PIXEL_PITCH)
    return shape, float(spacing)


def _model_class():
    import equinox as eqx
    from jaxtyping import Array, PRNGKeyArray

    from chromatix.functional import ff_lens, objective_point_source, phase_change
    from chromatix.ops import shot_noise
    from chromatix.utils import seidel_aberrations

    shape, spacing = _shape_and_spacing()

    class SeidelPSF(eqx.Module):
        coefficients: Array

        def __call__(
            self, z: Array, u: float, v: float, key: PRNGKeyArray | None = None
        ) -> Array:
            field = objective_point_source(
                shape, spacing, WAVELENGTH, z, F, N, NA, power=1e3
            )
            aberrations = seidel_aberrations(
                shape, spacing, WAVELENGTH, N, F, NA, self.coefficients, u, v
            )
            field = phase_change(field, aberrations)
            field = ff_lens(field, F, N)
            image = field.intensity
            if key is not None:
                image = shot_noise(key, image)
            return image

    return SeidelPSF


def _fit(project_non_negative: bool = True, thread_optimizer_state: bool = False):
    """Run the retrieval.

    ``thread_optimizer_state=False`` reproduces upstream exactly: its ``update()``
    returns only ``(model, metrics)``, so the loop keeps passing the INITIAL Adam
    state. ``True`` threads the state through, which is the intended Adam pattern
    and converges to a worse solution here.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    SeidelPSF = _model_class()
    key = jax.random.PRNGKey(4)
    model = SeidelPSF(jnp.zeros((5,)))
    coefficients_truth = jnp.array(TRUTH_WAVES) / (2 * jnp.pi / WAVELENGTH)
    psf_truth = SeidelPSF(coefficients_truth)(z=0.0, u=0.0, v=1.0, key=key)

    def loss_fn(candidate, data, z, u, v):
        estimate = candidate(z=z, u=u, v=v)
        loss = jnp.mean(jnp.square(estimate - data))
        return loss, {"loss": loss}

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(model)

    @jax.jit
    def update(candidate, state, data, z, u, v):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(candidate, data, z, u, v)
        updates, new_state = optimizer.update(grads, state, candidate)
        return optax.apply_updates(candidate, updates), new_state, metrics, grads

    history = []
    first_gradient = None
    for _ in range(MAX_ITERATIONS):
        model, new_state, metrics, grads = update(
            model, opt_state, psf_truth, 0.0, 0.0, 1.0
        )
        if thread_optimizer_state:
            opt_state = new_state
        if project_non_negative:
            model = eqx.tree_at(
                lambda t: getattr(t, "coefficients"), model, jnp.abs(model.coefficients)
            )
        if first_gradient is None:
            first_gradient = np.asarray(grads.coefficients, dtype=float)
        history.append(float(metrics["loss"]))
    return (
        np.asarray(history, dtype=float),
        np.asarray(model.coefficients, dtype=float),
        first_gradient,
        np.asarray(coefficients_truth, dtype=float),
        psf_truth,
    )


def run() -> TutorialResult:
    import jax

    result = TutorialResult()
    shape, spacing = _shape_and_spacing()
    result.record(simulation_shape=list(shape), simulation_spacing=spacing)
    result.check_true(
        "the_simulation_shape_matches_upstream",
        "reference",
        shape == UPSTREAM_SHAPE,
        f"{shape}; upstream prints {UPSTREAM_SHAPE}",
    )
    result.check_close(
        "the_simulation_spacing_matches_upstream",
        "reference",
        round(spacing, 2),
        UPSTREAM_SPACING,
        rel=1e-9,
    )

    history, estimated, first_gradient, truth, psf_truth = _fit(
        project_non_negative=True, thread_optimizer_state=False
    )
    result.record(
        max_iterations=MAX_ITERATIONS,
        psf_shape=list(np.asarray(psf_truth).shape),
        loss_at_iterations=[float(history[i]) for i in (0, 99, 199, 299, 399, 499, 599, 999)],
        initial_loss=float(history[0]),
        final_loss=float(history[-1]),
        coefficients_truth=truth,
        coefficients_estimated=estimated,
        upstream_estimated=list(UPSTREAM_ESTIMATED),
        mean_coefficient_error=float(np.mean((estimated - truth) ** 2)),
        first_gradient=first_gradient,
    )
    result.check_finite("loss_history_finite", history)
    result.check_close(
        "initial_loss_matches_upstreams_1048p57",
        "reference",
        float(history[0]),
        UPSTREAM_INITIAL_LOSS,
        rel=0.05,
    )
    result.check_close(
        "converged_loss_matches_upstreams_0p843",
        "reference",
        float(history[-1]),
        UPSTREAM_FINAL_LOSS,
        rel=0.3,
    )
    result.check_close(
        "mean_coefficient_error_matches_upstreams_7p52e_minus_5",
        "reference",
        float(np.mean((estimated - truth) ** 2)),
        UPSTREAM_COEFFICIENT_ERROR,
        rel=0.5,
    )
    result.check_true(
        "the_estimated_coefficients_match_upstreams_published_values",
        "reference",
        bool(
            np.allclose(estimated, np.asarray(UPSTREAM_ESTIMATED), atol=0.02)
        ),
        f"estimated {np.round(estimated, 6).tolist()} against upstream's published "
        f"{list(UPSTREAM_ESTIMATED)} (atol 0.02)",
    )
    result.check_true(
        "the_loss_falls_by_more_than_three_orders_of_magnitude",
        "invariant",
        float(history[-1]) < float(history[0]) * 1e-3,
        f"{float(history[0]):.4f} -> {float(history[-1]):.6f}, a "
        f"{float(history[0]) / float(history[-1]):.0f}x reduction",
    )
    result.check_true(
        "every_coefficient_receives_a_finite_gradient",
        "analytic",
        first_gradient.size == 5
        and bool(np.all(np.isfinite(first_gradient)))
        and bool(np.all(np.abs(first_gradient) > 0.0)),
        f"|grad| in [{np.abs(first_gradient).min():.3e}, {np.abs(first_gradient).max():.3e}] "
        "at the first step over all five Seidel coefficients",
    )
    zero_truth = [i for i, value in enumerate(TRUTH_WAVES) if value == 0.0]
    large_truth = [i for i, value in enumerate(TRUTH_WAVES) if value > 0.0]
    result.record(
        recovered_at_zero_truth=estimated[zero_truth],
        recovered_at_nonzero_truth=estimated[large_truth],
    )
    result.check_true(
        "the_fit_separates_present_from_absent_aberrations",
        "analytic",
        float(np.max(np.abs(estimated[zero_truth]))) < 0.05
        and float(np.min(estimated[large_truth])) > 0.3,
        f"the two coefficients whose truth is 0 come back at "
        f"{np.round(estimated[zero_truth], 8).tolist()}, while the three whose truth is "
        f"{truth[0]:.6f} come back at {np.round(estimated[large_truth], 6).tolist()}",
    )

    # -- the stale optimizer state is what makes it converge --------------------
    threaded_history, threaded, _, _, _ = _fit(
        project_non_negative=True, thread_optimizer_state=True
    )
    result.record(
        threaded_adam_final_loss=float(threaded_history[-1]),
        threaded_adam_coefficients=threaded,
        threaded_adam_mean_coefficient_error=float(np.mean((threaded - truth) ** 2)),
    )
    result.check_true(
        "threading_the_adam_state_through_makes_the_fit_worse",
        "analytic",
        float(np.mean((threaded - truth) ** 2))
        > 10.0 * float(np.mean((estimated - truth) ** 2)),
        "upstream's loop passes the INITIAL opt_state on every iteration because its "
        "update() returns only (model, metrics). Rebinding the state -- the intended Adam "
        f"pattern -- raises the final loss from {float(history[-1]):.6f} to "
        f"{float(threaded_history[-1]):.6f} and the mean squared coefficient error from "
        f"{float(np.mean((estimated - truth) ** 2)):.6e} to "
        f"{float(np.mean((threaded - truth) ** 2)):.6e}. With the state frozen every step "
        "is a fresh bias-corrected Adam step, i.e. sign descent at a fixed step of lr, "
        "which escapes the local minimum that accumulated moments settle into. The "
        "published numbers depend on this.",
    )

    # -- the non-negativity projection is part of the method --------------------
    unprojected_history, unprojected, _, _, _ = _fit(
        project_non_negative=False, thread_optimizer_state=False
    )
    result.record(
        unprojected_final_loss=float(unprojected_history[-1]),
        unprojected_coefficients=unprojected,
        unprojected_mean_coefficient_error=float(np.mean((unprojected - truth) ** 2)),
        unprojected_has_negative_coefficients=bool(np.any(unprojected < 0.0)),
    )
    result.check_true(
        "the_eqx_tree_at_non_negativity_projection_is_load_bearing",
        "analytic",
        float(np.mean((unprojected - truth) ** 2))
        > float(np.mean((estimated - truth) ** 2)),
        "dropping the eqx.tree_at(..., jnp.abs(...)) step raises the mean squared "
        f"coefficient error from {float(np.mean((estimated - truth) ** 2)):.6e} to "
        f"{float(np.mean((unprojected - truth) ** 2)):.6e} "
        f"(final loss {float(history[-1]):.6f} -> "
        f"{float(unprojected_history[-1]):.6f}), with negative coefficients present: "
        f"{bool(np.any(unprojected < 0.0))}. The projection is part of the method, not "
        "decoration -- and eqx.tree_at is how a constraint is applied to an "
        "equinox.Module between optimizer steps.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
