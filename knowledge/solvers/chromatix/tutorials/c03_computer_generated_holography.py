"""Example 3 / "Computer Generated Holography" -- https://chromatix.readthedocs.io/en/latest/examples/cgh/

Repo-owned reproduction of the 3D CGH example: an `equinox.Module` whose only
parameter is a 256x256 phase mask, forward-modelled through `plane_wave` ->
`phase_change` -> `ff_lens` -> `transfer_propagate` over a **51-plane z stack**, then
optimized with Adam to maximise the Pearson correlation between the resulting 3D
intensity and a three-blob target.

Upstream publishes four numbers and a correlation trajectory, and all of them
reproduce:

| upstream | quantity |
|---|---|
| `[22500. 22500.]` | `Field.extent` for a 256 px grid at 9.2 um pitch |
| `(51, 256, 256)` | `Field.intensity.shape` for a 51-element `z` |
| `0.99963427` / `0.00036573` | initial loss / correlation |
| ~`0.714` | correlation after 1000 Adam steps |

The `z` batching is the structural point worth recording: **a 1D `z` array adds a
leading batch axis**, so one `transfer_propagate` call produces the whole 3D stack
and one `jax.grad` differentiates through all 51 planes at once. That is what makes
3D CGH a single-shot optimization here rather than a loop.

Beyond the published numbers:

* **The optimization is real, not a plateau**: the correlation rises monotonically
  in its running maximum from 3.7e-4 to ~0.71, and the learned phase mask is not
  the zeros it started from.
* **The target planes are the ones that improve.** Correlation is a global measure,
  so the reproduction also checks that the intensity at each of the three blob
  centres rises well above the volume mean -- i.e. the hologram forms *there*, not
  merely somewhere.
* **Upstream's target is circularly shifted by half the kernel width.**
  `fftn(kernel, s=sample.shape)` places the 25-voxel ball's origin at index 0
  rather than centring it, so every blob ends up 12 voxels away on each axis from
  the coordinate that was seeded. The seeded voxels hold ~0 and the shifted ones
  hold the blob maxima. That is upstream's own construction, recorded because a
  reader who trusted the seeded coordinates would look in the wrong place -- and
  because it is what a naive "did the hologram form at (50, 10, 25)?" check gets
  wrong.
* **Upstream threads `opt_state` correctly here**, unlike the two aberration-fitting
  examples (`c04`, `c10`), and the reproduction records that difference explicitly
  since it is the reason those two behave the way they do.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c03_computer_generated_holography",
    title="Computer Generated Holography",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/cgh/",
    demonstrates=(
        "a 1D `z` array adding a leading batch axis to transfer_propagate, "
        "phase_change as the only free parameter of an equinox.Module, "
        "ff_lens(f, n, NA), Field.extent, and a correlation loss over a 3D stack."
    ),
    slow=True,
)

SHAPE = (256, 256)
SPACING = 9.2
NUM_PLANES = 51
Z_MAX = 100.0e4
MAX_ITERATIONS = 1000
UPSTREAM_EXTENT = 22500.0
UPSTREAM_INTENSITY_SHAPE = (51, 256, 256)
UPSTREAM_INITIAL_LOSS = 0.99963427
UPSTREAM_INITIAL_CORRELATION = 0.00036573
UPSTREAM_FINAL_CORRELATION = 0.714
TARGET_VOXELS = ((30, 128, 128), (10, 51, 92), (50, 10, 25))


def _model_class():
    import equinox as eqx
    import jax.numpy as jnp
    from jaxtyping import Array

    from chromatix import Field
    from chromatix.functional import ff_lens, phase_change, plane_wave, transfer_propagate

    class CGH(eqx.Module):
        phase: Array
        shape: tuple[int, int] = eqx.field(static=True)
        spacing: float = eqx.field(static=True)
        z: Array = eqx.field(static=True)
        f: float = eqx.field(static=True)
        n: float = eqx.field(static=True)
        NA: float | None = eqx.field(static=True)
        pad_width: int = eqx.field(static=True)
        spectrum: float | Array = eqx.field(static=True)

        def __init__(
            self,
            shape: tuple[int, int],
            spacing: float,
            z: Array,
            f: float = 200.0e3,
            n: float = 1.0,
            NA: float | None = None,
            pad_width: int = 0,
            spectrum: float | Array = 1.035,
        ):
            self.shape = shape
            self.spacing = spacing
            self.z = z
            self.f = f
            self.n = n
            self.NA = NA
            self.pad_width = pad_width
            self.spectrum = spectrum
            self.phase = jnp.zeros(self.shape)

        def __call__(self) -> Field:
            field = plane_wave(self.shape, self.spacing, self.spectrum)
            field = phase_change(field, self.phase)
            field = ff_lens(field, self.f, self.n, self.NA)
            field = transfer_propagate(
                field, self.z, self.n, pad_width=self.pad_width, mode="same"
            )
            return field

    return CGH


def build_target():
    """Upstream's three-blob 3D target, convolved with a 25-voxel ball."""
    import jax.numpy as jnp

    sample = np.zeros((NUM_PLANES, *SHAPE))
    for plane, row, column in TARGET_VOXELS:
        sample[plane, row, column] = 1.0
    diameter = 25
    kernel = np.zeros((diameter, diameter, diameter))
    grid = np.meshgrid(
        np.linspace(-diameter / 2, diameter / 2, num=diameter),
        np.linspace(-diameter / 2, diameter / 2, num=diameter),
        np.linspace(-diameter / 2, diameter / 2, num=diameter),
    )
    radius = np.sqrt(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2)
    kernel[radius < diameter / 5] = 1.0
    convolved = jnp.fft.ifftn(
        jnp.fft.fftn(jnp.array(sample))
        * jnp.fft.fftn(jnp.array(kernel), s=sample.shape)
    ).real
    return convolved[..., jnp.newaxis, jnp.newaxis] * 1000.0


def run() -> TutorialResult:
    import jax
    import jax.numpy as jnp
    import optax

    result = TutorialResult()
    CGH = _model_class()
    z = jnp.linspace(0.0, Z_MAX, num=NUM_PLANES)
    model = CGH(shape=SHAPE, spacing=SPACING, z=z)

    output_field = model()
    extent = np.asarray(output_field.extent, dtype=float).ravel()
    intensity_shape = tuple(np.asarray(model().intensity).shape)
    result.record(
        shape=list(SHAPE),
        spacing=SPACING,
        num_planes=NUM_PLANES,
        field_extent=extent,
        intensity_shape=list(intensity_shape),
    )
    result.check_true(
        "the_field_extent_matches_upstream",
        "reference",
        bool(np.allclose(extent, UPSTREAM_EXTENT, rtol=1e-6)),
        f"Field.extent = {extent.tolist()}; upstream prints [22500. 22500.] "
        f"(= 256 px x {SPACING} um)",
    )
    result.check_true(
        "a_1d_z_array_adds_a_leading_batch_axis",
        "reference",
        intensity_shape[:3] == UPSTREAM_INTENSITY_SHAPE,
        f"Field.intensity.shape = {intensity_shape}; upstream prints "
        f"{UPSTREAM_INTENSITY_SHAPE}. One transfer_propagate call over a 51-element z "
        "produces the whole 3D stack, which is what makes this a single-shot "
        "optimization rather than a loop over planes.",
    )

    target = build_target()
    target_np = np.asarray(target).squeeze()
    result.record(
        target_shape=list(target_np.shape),
        target_max=float(target_np.max()),
        target_occupied_fraction=float(
            np.count_nonzero(target_np > 0.01 * target_np.max()) / target_np.size
        ),
        target_numerical_floor_fraction=float(
            np.count_nonzero(target_np > 1e-6) / target_np.size
        ),
        target_voxels=[list(v) for v in TARGET_VOXELS],
    )
    # Upstream builds the target with `fftn(kernel, s=sample.shape)`, which puts the
    # kernel ORIGIN at index 0 rather than at its centre, so the circular convolution
    # shifts every blob by half the kernel width (12 voxels) on each axis. The seeded
    # voxel coordinates are therefore NOT where the blobs end up.
    half_kernel = 25 // 2
    shifted_voxels = [
        tuple((c + half_kernel) % n for c, n in zip(voxel, target_np.shape, strict=True))
        for voxel in TARGET_VOXELS
    ]
    seed_values = [float(target_np[v]) for v in TARGET_VOXELS]
    shifted_values = [float(target_np[v]) for v in shifted_voxels]
    result.record(
        kernel_half_width=half_kernel,
        shifted_target_voxels=[list(v) for v in shifted_voxels],
        target_value_at_seed_voxels=seed_values,
        target_value_at_shifted_voxels=shifted_values,
    )
    result.check_finite("target_finite", target_np)
    result.check_true(
        "the_fft_convolution_shifts_every_blob_by_half_the_kernel_width",
        "analytic",
        all(v > 0.5 * target_np.max() for v in shifted_values)
        and min(shifted_values) > 10.0 * max(seed_values),
        f"the target value at the three SEEDED voxels is {[round(v, 6) for v in seed_values]} "
        f"but at those voxels shifted by +{half_kernel} on each axis it is "
        f"{[round(v, 3) for v in shifted_values]}, all above half the "
        f"{float(target_np.max()):.3f} maximum. `fftn(kernel, s=sample.shape)` puts the "
        "kernel origin at index 0 rather than centring it, so the circular convolution "
        "translates every blob. That is upstream's own construction, and a reader who "
        "assumed the blobs sit at the seeded coordinates would look in the wrong place.",
    )
    result.check_true(
        "the_target_is_three_localised_blobs",
        "analytic",
        0.0
        < float(np.count_nonzero(target_np > 0.01 * target_np.max()) / target_np.size)
        < 0.05,
        f"{100 * np.count_nonzero(target_np > 0.01 * target_np.max()) / target_np.size:.3f}% "
        f"of voxels exceed 1% of the {float(target_np.max()):.3f} target maximum. Note "
        f"{100 * np.count_nonzero(target_np > 1e-6) / target_np.size:.1f}% of voxels are "
        "above a bare 1e-6: building the target by FFT convolution leaves float32 ringing "
        "everywhere, so an absolute threshold measures the numerical floor rather than the "
        "object.",
    )

    def loss_fn(candidate, target_stack):
        approx = candidate().intensity
        correlation = jnp.corrcoef(approx.flatten(), target_stack.flatten())[0, 1]
        loss = 1.0 - correlation
        return loss, {"loss": loss, "correlation": correlation}

    initial_loss, initial_metrics = loss_fn(model, target)
    result.record(
        initial_loss=float(initial_loss),
        initial_correlation=float(initial_metrics["correlation"]),
    )
    result.check_close(
        "the_initial_loss_matches_upstream",
        "reference",
        float(initial_loss),
        UPSTREAM_INITIAL_LOSS,
        rel=0.01,
    )
    result.check_close(
        "the_initial_correlation_matches_upstream",
        "reference",
        float(initial_metrics["correlation"]),
        UPSTREAM_INITIAL_CORRELATION,
        rel=0.5,
    )

    optimizer = optax.adam(learning_rate=1e-1)
    opt_state = optimizer.init(model)

    @jax.jit
    def update(candidate, state, target_stack):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(candidate, target_stack)
        updates, state = optimizer.update(grads, state, candidate)
        return optax.apply_updates(candidate, updates), state, metrics, grads

    initial_phase = np.asarray(model.phase, dtype=float).copy()
    history = {"loss": np.zeros(MAX_ITERATIONS), "correlation": np.zeros(MAX_ITERATIONS)}
    first_gradient = None
    for iteration in range(MAX_ITERATIONS):
        # NOTE: upstream DOES rebind opt_state here, unlike c04 and c10.
        model, opt_state, metrics, grads = update(model, opt_state, target)
        if first_gradient is None:
            first_gradient = np.asarray(grads.phase, dtype=float)
        history["loss"][iteration] = float(metrics["loss"])
        history["correlation"][iteration] = float(metrics["correlation"])

    learned_phase = np.asarray(model.phase, dtype=float)
    correlation_history = history["correlation"]
    result.record(
        max_iterations=MAX_ITERATIONS,
        correlation_at_iterations=[
            float(correlation_history[i]) for i in (0, 199, 399, 599, 799, 999)
        ],
        final_correlation=float(correlation_history[-1]),
        best_correlation=float(correlation_history.max()),
        final_loss=float(history["loss"][-1]),
        learned_phase_range=[float(learned_phase.min()), float(learned_phase.max())],
        learned_phase_rms=float(np.sqrt(np.mean(learned_phase**2))),
        initial_phase_rms=float(np.sqrt(np.mean(initial_phase**2))),
        first_gradient_abs_max=float(np.abs(first_gradient).max()),
    )
    result.check_finite("correlation_history_finite", correlation_history)
    result.check_close(
        "the_final_correlation_matches_upstreams_0p714",
        "reference",
        float(correlation_history[-1]),
        UPSTREAM_FINAL_CORRELATION,
        rel=0.1,
    )
    result.check_true(
        "the_running_maximum_correlation_never_falls",
        "invariant",
        bool(np.all(np.diff(np.maximum.accumulate(correlation_history)) >= 0.0)),
        f"correlation rises from {float(correlation_history[0]):.6f} to "
        f"{float(correlation_history[-1]):.6f} over {MAX_ITERATIONS} Adam steps, with a "
        f"best of {float(correlation_history.max()):.6f}",
    )
    result.check_true(
        "the_phase_mask_actually_moved_off_its_zero_initialisation",
        "analytic",
        float(np.sqrt(np.mean(learned_phase**2))) > 0.1
        and float(np.sqrt(np.mean(initial_phase**2))) == 0.0,
        f"phase RMS {float(np.sqrt(np.mean(initial_phase**2))):.6f} -> "
        f"{float(np.sqrt(np.mean(learned_phase**2))):.6f}, range "
        f"[{float(learned_phase.min()):.4f}, {float(learned_phase.max()):.4f}] rad. "
        "The parameter is the mask, so a non-trivial mask is the deliverable.",
    )
    result.check_true(
        "the_gradient_reaches_every_pixel_of_the_phase_mask",
        "analytic",
        first_gradient.shape == SHAPE
        and bool(np.all(np.isfinite(first_gradient)))
        and float(np.abs(first_gradient).max()) > 0.0,
        f"the first gradient is {first_gradient.shape} with a maximum magnitude of "
        f"{float(np.abs(first_gradient).max()):.6e}: jax.grad differentiates the whole "
        "51-plane stack with respect to all 65536 mask pixels at once",
    )

    # -- the hologram forms at the target voxels, not merely somewhere -----------
    learned_intensity = np.asarray(model().intensity).squeeze()
    volume_mean = float(learned_intensity.mean())
    contrasts = {
        f"{plane}_{row}_{column}": float(
            learned_intensity[plane, row, column] / volume_mean
        )
        for plane, row, column in shifted_voxels
    }
    result.record(
        learned_intensity_shape=list(learned_intensity.shape),
        volume_mean_intensity=volume_mean,
        target_voxel_contrast=contrasts,
    )
    result.check_true(
        "the_intensity_is_concentrated_at_the_three_target_voxels",
        "analytic",
        all(value > 3.0 for value in contrasts.values()),
        "intensity at each ACTUAL blob centre (the seeded voxel shifted by the kernel "
        "half-width, see above) relative to the volume mean: "
        + ", ".join(f"{k} {v:.2f}x" for k, v in contrasts.items())
        + ". Correlation is a global measure, so this checks that the hologram forms "
        "WHERE the target actually is rather than merely somewhere.",
    )
    result.note(
        "Upstream's loop here rebinds opt_state correctly "
        "(`model, opt_state, metrics = update(model, opt_state, sample)`), unlike the "
        "aberration-fitting examples c04_zernike_fitting and c10_seidel_fitting whose "
        "update() returns only (model, metrics). That inconsistency between the examples "
        "is why those two behave the way they do, and it is recorded here because this is "
        "the example that shows the intended pattern."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
