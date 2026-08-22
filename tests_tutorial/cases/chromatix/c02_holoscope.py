"""Example 2 / "Holoscope" -- https://chromatix.readthedocs.io/en/latest/examples/holoscope/

Repo-owned reproduction of the Holoscope PSF-engineering example: a
`defocused_ramps` phase mask in the pupil of a 4f microscope, whose PSF encodes a
40-plane depth range into six laterally separated views on one 2D sensor, then the
crop/taper/downsample chain that maps the 1920x1920 simulation onto a
1280x1280 camera at 5x binning.

Upstream prints three numbers and two shapes, and all of them reproduce:

| upstream | quantity |
|---|---|
| `(1920, 1920)` | simulation shape = camera 1280 + pad 640 |
| `6.410` | input spacing `f_tube * lambda / (n * shape[0] * pitch)` |
| `6.500` | the camera pixel pitch it is designed to land on |
| `(40, 1920, 1920)` | `psf.intensity` over a 40-plane `z` |
| `(40, 256, 256)` | after `center_crop` + `sigmoid_taper` + 5x `einops.reduce` |

The engineering claim -- "PSF engineering to optimally encode a 3D volume into a 2D
image" -- is the thing worth validating, and it is checkable by comparison rather
than by eye:

* **The mask breaks the depth symmetry that an ordinary 4f system has, which is
  exactly what depth encoding requires.** Reflecting the z stack about focus and
  comparing plane by plane: the **unmasked** PSF is symmetric to 0.02% of its RMS --
  its 90%-energy radius is 155.8 px at both `z = +/-100 um` and 3.8 px at both
  `z = +/-2.6 um` -- so two emitters equidistant either side of focus are
  indistinguishable in one 2D image. The **engineered** PSF's reflected difference is
  a large fraction of its RMS, so the sign of z is recoverable. Running both systems
  is what turns "the picture looks like six streaks" into a measurement.
* **The engineered PSF fills the field at every depth** (90%-energy radius 482-510 px
  against the unmasked 4-156 px), which is the cost of the encoding and the reason
  the example needs the crop/taper/downsample chain.
* Note the *brightest* lobe hops between the six views as z changes (its x offset
  correlates with z at only r = 0.76), so a naive "track the argmax" reading of the
  depth code does not work -- recorded because it is the obvious thing to try.
* **The six views are structural, not a parameter.** `defocused_ramps` indexes
  `delta[ramp_idx]` for six fixed ramps: passing a 3-element `delta` raises
  `IndexError: list index out of range`. So the six-view geometry is built into the
  function, and the `[1296.0] * 6` in the example is not an arbitrary length.
* **The crop/downsample chain conserves energy**, because `einops.reduce` uses
  `reduction="sum"`: the downsampled PSF's total equals the cropped, tapered PSF's
  total to float32 round-off.
* **`filaments_3d` accepts a seed and is then reproducible** -- which is what makes
  the sample in this example a fixture, unlike the unseeded call in `c14`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c02_holoscope",
    title="Holoscope",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/holoscope/",
    demonstrates=(
        "chromatix.utils.defocused_ramps as an engineered pupil phase, "
        "objective_point_source over a 1D z, chromatix.utils.center_crop / "
        "sigmoid_taper, einops.reduce sum-downsampling, and "
        "filaments_3d(seed=...)."
    ),
    slow=True,
)

CAMERA_SHAPE = (1280, 1280)
CAMERA_PIXEL_PITCH = 6.5
F_OBJECTIVE = 8e3
F_TUBE = 200e3
N = 1.33
NA = 0.8
SPECTRUM = 0.532
DOWNSAMPLE_FACTOR = 5
PAD = 640
TAPER_WIDTH = 5
NUM_PLANES = 40
Z_RANGE = 100.0
NUM_RAMPS = 6
UPSTREAM_SHAPE = (1920, 1920)
UPSTREAM_SPACING = 6.410
UPSTREAM_PSF_SHAPE = (NUM_PLANES, 1920, 1920)
UPSTREAM_DOWNSAMPLED_SHAPE = (NUM_PLANES, 256, 256)


def _geometry() -> tuple[tuple[int, int], float]:
    shape = tuple(int(v) for v in np.array(CAMERA_SHAPE).astype(int) + PAD)
    spacing = F_TUBE * SPECTRUM / (N * shape[0] * CAMERA_PIXEL_PITCH)
    return shape, float(spacing)


def system_psf(phase, z):
    from chromatix.functional import ff_lens, objective_point_source, phase_change

    shape, spacing = _geometry()
    field = objective_point_source(
        shape, spacing, SPECTRUM, z, F_OBJECTIVE, N, NA
    )
    if phase is not None:
        field = phase_change(field, phase)
    return ff_lens(field, F_TUBE, N)


def crop_and_downsample(intensity):
    from einops import reduce

    from chromatix.utils import center_crop, sigmoid_taper

    cropped = center_crop(intensity, (None, PAD // 2, PAD // 2, None, None))
    tapered = cropped * sigmoid_taper(CAMERA_SHAPE, TAPER_WIDTH)
    return reduce(
        tapered,
        "d (h hf) (w wf) -> d h w",
        reduction="sum",
        hf=DOWNSAMPLE_FACTOR,
        wf=DOWNSAMPLE_FACTOR,
    ), tapered


def _radial_second_moment(plane: np.ndarray) -> float:
    height, width = plane.shape
    yy, xx = np.mgrid[:height, :width]
    weights = plane / plane.sum()
    centre_y = float((yy * weights).sum())
    centre_x = float((xx * weights).sum())
    return float(
        np.sqrt((((yy - centre_y) ** 2 + (xx - centre_x) ** 2) * weights).sum())
    )


def run() -> TutorialResult:
    import jax.numpy as jnp

    from chromatix.utils import defocused_ramps, filaments_3d

    result = TutorialResult()
    shape, spacing = _geometry()
    z = jnp.linspace(-Z_RANGE, Z_RANGE, num=NUM_PLANES)
    result.record(
        camera_shape=list(CAMERA_SHAPE),
        pad=PAD,
        simulation_shape=list(shape),
        input_spacing=spacing,
        camera_pixel_pitch=CAMERA_PIXEL_PITCH,
        num_planes=NUM_PLANES,
        z_range=[-Z_RANGE, Z_RANGE],
    )
    result.check_true(
        "the_simulation_shape_matches_upstream",
        "reference",
        shape == UPSTREAM_SHAPE,
        f"{shape}; upstream prints (1920, 1920) = camera {CAMERA_SHAPE[0]} + pad {PAD}",
    )
    result.check_close(
        "the_input_spacing_matches_upstream",
        "reference",
        round(spacing, 3),
        UPSTREAM_SPACING,
        rel=1e-9,
    )
    result.check_true(
        "the_input_spacing_is_close_to_but_not_equal_to_the_camera_pitch",
        "analytic",
        abs(spacing - CAMERA_PIXEL_PITCH) / CAMERA_PIXEL_PITCH < 0.02
        and spacing != CAMERA_PIXEL_PITCH,
        f"the Fourier-plane spacing f_tube*lambda/(n*shape[0]*pitch) = {spacing:.4f} um is "
        f"within {100 * abs(spacing - CAMERA_PIXEL_PITCH) / CAMERA_PIXEL_PITCH:.2f}% of the "
        f"{CAMERA_PIXEL_PITCH} um camera pitch but not equal to it. The simulation grid is "
        "chosen so the FFT lands near the sensor pitch; the residual mismatch is why the "
        "example downsamples rather than reads pixels off directly.",
    )

    # -- the engineered phase mask ---------------------------------------------
    phase = defocused_ramps(
        shape, spacing, SPECTRUM, N, F_OBJECTIVE, NA, delta=[1296.0] * NUM_RAMPS
    )
    phase_np = np.asarray(phase, dtype=float)
    result.record(
        phase_shape=list(phase_np.shape),
        phase_range=[float(phase_np.min()), float(phase_np.max())],
        phase_rms=float(np.sqrt(np.mean(phase_np**2))),
        num_ramps=NUM_RAMPS,
    )
    result.check_finite("phase_mask_finite", phase_np)
    result.check_true(
        "the_phase_mask_is_non_trivial_and_many_waves_deep",
        "analytic",
        float(phase_np.max() - phase_np.min()) > 2 * np.pi,
        f"the mask spans {float(phase_np.max() - phase_np.min()):.3f} rad = "
        f"{float(phase_np.max() - phase_np.min()) / (2 * np.pi):.2f} waves, with an RMS of "
        f"{float(np.sqrt(np.mean(phase_np**2))):.3f} rad. A depth-encoding mask has to be "
        "many waves deep; a sub-wave mask could not separate views.",
    )

    # -- the PSF with and without the mask -------------------------------------
    engineered = system_psf(phase, z)
    engineered_intensity = np.asarray(engineered.intensity).squeeze()
    plain_intensity = np.asarray(system_psf(None, z).intensity).squeeze()
    result.record(
        psf_shape=list(engineered_intensity.shape),
        psf_total_energy=float(engineered_intensity.sum()),
    )
    result.check_finite("psf_finite", engineered_intensity)
    result.check_true(
        "a_1d_z_gives_a_40_plane_psf_stack",
        "reference",
        tuple(engineered_intensity.shape) == UPSTREAM_PSF_SHAPE,
        f"psf.intensity is {tuple(engineered_intensity.shape)}; upstream's psf.shape cell "
        f"shows the same {UPSTREAM_PSF_SHAPE} for a 40-element z",
    )

    def _peak_offset(plane: np.ndarray) -> tuple[float, float, float]:
        """(dy, dx, radius) of the brightest pixel relative to the grid centre, in px."""
        index = int(np.argmax(plane))
        y, x = np.unravel_index(index, plane.shape)
        dy = float(y - plane.shape[0] // 2)
        dx = float(x - plane.shape[1] // 2)
        return dy, dx, float(np.hypot(dy, dx))

    def _energy_radius(plane: np.ndarray) -> float:
        """RMS radius of the brightest 90% of the energy, in px."""
        height, width = plane.shape
        yy, xx = np.mgrid[:height, :width]
        radius = np.hypot(yy - height // 2, xx - width // 2).ravel()
        flat = plane.ravel()
        order = np.argsort(flat)[::-1]
        keep = order[np.cumsum(flat[order]) <= 0.9 * flat.sum()]
        return float(np.sqrt((radius[keep] ** 2 * flat[keep]).sum() / flat[keep].sum()))

    z_np = np.asarray(z, dtype=float)
    engineered_peaks = [_peak_offset(plane) for plane in engineered_intensity]
    plain_peaks = [_peak_offset(plane) for plane in plain_intensity]
    engineered_radii = np.asarray([_energy_radius(plane) for plane in engineered_intensity])
    plain_radii = np.asarray([_energy_radius(plane) for plane in plain_intensity])
    engineered_dx = np.asarray([peak[1] for peak in engineered_peaks])
    plain_peak_radii = np.asarray([peak[2] for peak in plain_peaks])
    result.record(
        z_planes=z_np,
        engineered_peak_dx_px=engineered_dx,
        engineered_peak_radius_px=[peak[2] for peak in engineered_peaks],
        plain_peak_radius_px=plain_peak_radii,
        engineered_energy_radius_px=engineered_radii,
        plain_energy_radius_px=plain_radii,
    )
    result.check_finite(
        "psf_position_measures_finite",
        np.concatenate([engineered_dx, plain_peak_radii, engineered_radii, plain_radii]),
    )
    result.check_true(
        "the_unmasked_psf_peak_never_leaves_the_axis",
        "analytic",
        float(plain_peak_radii.max()) < 25.0,
        f"without the mask the brightest pixel stays within "
        f"{float(plain_peak_radii.max()):.1f} px of the grid centre at all "
        f"{NUM_PLANES} depths (0 px at most of them): an ordinary 4f system only "
        "defocuses, it does not translate the PSF.",
    )
    plain_symmetry = float(
        np.max(np.abs(plain_radii - plain_radii[::-1])) / plain_radii.mean()
    )
    result.record(plain_energy_radius_z_symmetry=plain_symmetry)
    result.check_true(
        "the_unmasked_psf_is_symmetric_about_focus_and_so_cannot_encode_the_sign_of_z",
        "analytic",
        plain_symmetry < 0.05,
        f"the unmasked 90%-energy radius mirrored about z = 0 agrees to "
        f"{100 * plain_symmetry:.3f}% of its mean -- e.g. {float(plain_radii[0]):.1f} px at "
        f"z = {z_np[0]:+.1f} against {float(plain_radii[-1]):.1f} px at z = {z_np[-1]:+.1f}. "
        "Two emitters equidistant either side of focus are indistinguishable in one 2D "
        "image, which is exactly the problem a holoscope solves.",
    )
    correlation = float(np.corrcoef(z_np, engineered_dx)[0, 1])
    result.record(engineered_peak_dx_vs_z_correlation=correlation)
    result.check_true(
        "tracking_the_brightest_lobe_is_NOT_a_usable_depth_code",
        "analytic",
        correlation < 0.9,
        f"the brightest pixel's x offset correlates with z at only r = {correlation:.6f}, "
        "because the global argmax hops between the six views as z changes (its offset runs "
        f"{np.round(engineered_dx[:4], 0).tolist()} ... {np.round(engineered_dx[-4:], 0).tolist()} px). "
        "Recorded because 'track the argmax' is the obvious way to read the depth code and "
        "it does not work; the symmetry-breaking measure below is the one that does.",
    )

    # -- depth-sign encoding, by reflection about focus -------------------------
    def _reflection_difference(stack: np.ndarray) -> float:
        """RMS difference between the stack and its z-reflection, over its own RMS."""
        reflected = stack[::-1]
        scale = float(np.sqrt(np.mean(stack.astype(np.float64) ** 2)))
        return float(
            np.sqrt(np.mean((stack.astype(np.float64) - reflected.astype(np.float64)) ** 2))
            / scale
        )

    engineered_reflection = _reflection_difference(engineered_intensity)
    plain_reflection = _reflection_difference(plain_intensity)
    result.record(
        engineered_z_reflection_difference=engineered_reflection,
        plain_z_reflection_difference=plain_reflection,
        reflection_difference_ratio=engineered_reflection / max(plain_reflection, 1e-30),
    )
    result.check_true(
        "the_unmasked_psf_stack_is_symmetric_under_reflection_about_focus",
        "analytic",
        plain_reflection < 0.01,
        f"reflecting the unmasked 40-plane stack about z = 0 and differencing gives an RMS "
        f"of {100 * plain_reflection:.4f}% of the stack's own RMS. An ordinary 4f PSF is an "
        "even function of defocus, so no 2D measurement can recover the sign of z.",
    )
    result.check_true(
        "the_engineered_psf_stack_breaks_that_symmetry_and_so_encodes_the_sign_of_z",
        "analytic",
        engineered_reflection > 20.0 * plain_reflection
        and engineered_reflection > 0.1,
        f"the same reflection test on the engineered stack gives "
        f"{100 * engineered_reflection:.2f}% against the unmasked "
        f"{100 * plain_reflection:.4f}%, a factor of "
        f"{engineered_reflection / max(plain_reflection, 1e-30):.0f}. The defocused_ramps "
        "mask makes the PSF an ODD-containing function of depth, which is precisely what "
        "'optimally encode a 3D volume into a 2D image' requires and what the unmasked "
        "system cannot do.",
    )
    result.check_true(
        "the_engineered_psf_is_spread_over_the_whole_field_at_every_depth",
        "analytic",
        float(engineered_radii.min()) > 2.5 * float(plain_radii.max()),
        f"the engineered 90%-energy radius stays in "
        f"[{float(engineered_radii.min()):.1f}, {float(engineered_radii.max()):.1f}] px "
        f"against [{float(plain_radii.min()):.1f}, {float(plain_radii.max()):.1f}] px "
        "unmasked. The six views occupy the field at all depths -- which is the cost of the "
        "encoding, and the reason the example needs the crop/taper/downsample chain below.",
    )

    # -- the six views are structural ------------------------------------------
    ramp_count_error = ""
    try:
        defocused_ramps(shape, spacing, SPECTRUM, N, F_OBJECTIVE, NA, delta=[1296.0] * 3)
    except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
        ramp_count_error = f"{type(exc).__name__}: {exc}"
    result.record(three_ramp_delta_error=ramp_count_error)
    result.check_true(
        "defocused_ramps_requires_exactly_six_delta_entries",
        "analytic",
        ramp_count_error.startswith("IndexError"),
        f"defocused_ramps(..., delta=[1296.0] * 3) -> "
        f"{ramp_count_error or 'accepted (the ramp count is configurable after all?)'}. The "
        "function indexes delta[ramp_idx] for six fixed ramps, so the six-view geometry is "
        "built in and the example's `[1296.0] * 6` is not an arbitrary length.",
    )

    # -- the crop / taper / downsample chain -----------------------------------
    downsampled, tapered = crop_and_downsample(np.asarray(engineered.intensity))
    downsampled_np = np.asarray(downsampled, dtype=float).squeeze()
    tapered_np = np.asarray(tapered, dtype=float).squeeze()
    result.record(
        tapered_shape=list(tapered_np.shape),
        downsampled_shape=list(downsampled_np.shape),
        tapered_total=float(tapered_np.sum()),
        downsampled_total=float(downsampled_np.sum()),
        energy_ratio=float(downsampled_np.sum() / tapered_np.sum()),
    )
    result.check_true(
        "the_crop_and_downsample_chain_gives_upstreams_shape",
        "reference",
        tuple(downsampled_np.shape) == UPSTREAM_DOWNSAMPLED_SHAPE,
        f"{tuple(downsampled_np.shape)}; upstream's cell prints "
        f"{UPSTREAM_DOWNSAMPLED_SHAPE} (1280 / {DOWNSAMPLE_FACTOR} = 256)",
    )
    result.check_close(
        "sum_reduction_downsampling_conserves_energy",
        "analytic",
        float(downsampled_np.sum() / tapered_np.sum()),
        1.0,
        rel=1e-4,
    )

    # -- filaments_3d with a seed is reproducible -------------------------------
    sample = np.asarray(
        filaments_3d((40, 256, 256), rand_offset=0.5, thickness=1.0, seed=972920147),
        dtype=np.float32,
    )
    replay = np.asarray(
        filaments_3d((40, 256, 256), rand_offset=0.5, thickness=1.0, seed=972920147),
        dtype=np.float32,
    )
    different = np.asarray(
        filaments_3d((40, 256, 256), rand_offset=0.5, thickness=1.0, seed=1),
        dtype=np.float32,
    )
    same = bool(np.array_equal(sample, replay))
    differs = not bool(np.array_equal(sample, different))
    result.record(
        sample_shape=list(sample.shape),
        sample_occupied_voxels=int(np.count_nonzero(sample)),
        seeded_reproducible=same,
        different_seed_differs=differs,
    )
    result.check_true(
        "filaments_3d_is_reproducible_when_given_a_seed",
        "invariant",
        same and differs,
        f"seed=972920147 twice gives a bit-identical array ({same}) and seed=1 gives a "
        f"different one ({differs}). c14_filaments_phantom reproduces the docs page's "
        "UNSEEDED call and records what that gives; this example shows the seeded form, "
        "which is what makes a filament phantom usable as a fixture.",
    )
    return result


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Minimal 4-connected component labelling, so scipy is not required."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x]:
                continue
            current += 1
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not labels[ny, nx]
                    ):
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
