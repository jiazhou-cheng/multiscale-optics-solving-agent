"""Example 12 / "High-NA vectorial PSF generation" -- https://chromatix.readthedocs.io/en/latest/examples/highNA_PSF/

Repo-owned reproduction of the high-NA PSF example: a Gaussian-apodised circular
pupil focused by `cf.high_na_ff_lens` at NA 1.3 in n = 1.5, computed once as a
scalar field and once as a `VectorField` polarized along x.

**This repository already knows `high_na_ff_lens` is defective.**
`archive/benchmarks/gen1/benchmarks/L1-WAVE-01` Case 3 (CHE-18) found it *not sampling-independent*
and root-caused it: the function derives `s_z` from `field.f_grid * lambda / n` --
the *frequency* grid -- rather than from the pupil position grid, so on any
physically sampled pupil `|s_grid| ~ 0.015`, `s_z ~ 1`, and the intended
`1/cos(theta)` obliquity Jacobian, the `exp(i k f cos(theta))` defocus and the
`zoom_factor` that sets the output scale all degenerate to constants. See
`usage_notes.md`, "Known defective: high_na_ff_lens".

The value of reproducing this example is therefore not to validate the function --
it is to turn that finding into an **executable guard**, so the repository notices if
the behaviour ever changes:

* The example runs and produces finite scalar and vectorial PSFs of the requested
  camera shape.
* **The result is not sampling-independent.** Refining only the pupil sampling
  (128 -> 192 -> 256 px) with wavelength, NA, focal length, camera shape and camera
  pitch all fixed changes the PSF's radial width and its `Iz / Ix` ratio by a large
  factor. A correct high-NA focusing calculation converges under pupil refinement;
  this one does not.
* **The vectorial calculation does generate longitudinal field.** `Iz / Ix` is
  non-zero, which is the qualitative signature the example is showing -- so the
  function is not simply the scalar case in disguise, it is a wrong vectorial
  calculation.
* `VectorField.u`'s `(E_z, E_y, E_x)` order is confirmed again here from a different
  entry point than `c11`: `amplitude=jnp.array([0.0, 0.0, 1.0])` produces an
  x-polarized pupil.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c12_high_na_psf",
    title="High-NA vectorial PSF generation",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/highNA_PSF/",
    demonstrates=(
        "cf.gaussian_plane_wave(waist=, pupil=, amplitude=, scalar=False) and "
        "cf.high_na_ff_lens(field, f, n, NA, camera_shape, camera_pixel_pitch) -- "
        "and that the latter is not sampling-independent in the pinned commit."
    ),
    slow=True,
)

CAMERA_SHAPE = (256, 256)
CAMERA_PIXEL_PITCH = 0.005
F = 100.0
NA = 1.3
N = 1.5
WAVELENGTH = 0.532


def _psf(pupil_pixels: int, scalar: bool):
    import jax.numpy as jnp

    import chromatix.functional as cf

    pupil_shape = (pupil_pixels, pupil_pixels)
    correction = F * NA / N
    dx = 2 * correction / pupil_shape[0]
    kwargs = {}
    if not scalar:
        kwargs = {"amplitude": jnp.array([0.0, 0.0, 1.0]), "scalar": False}
    pupil_field = cf.gaussian_plane_wave(
        pupil_shape,
        dx,
        WAVELENGTH,
        waist=correction,
        pupil=lambda field: cf.circular_pupil(field, 2 * correction),
        **kwargs,
    )
    field = cf.high_na_ff_lens(
        pupil_field, F, N, NA, CAMERA_SHAPE, CAMERA_PIXEL_PITCH
    )
    return pupil_field, field


def _radial_width(intensity: np.ndarray) -> float:
    """Second-moment radius of an intensity map, in pixels."""
    height, width = intensity.shape[-2], intensity.shape[-1]
    yy, xx = np.mgrid[:height, :width]
    centre_y, centre_x = height // 2, width // 2
    radius2 = (yy - centre_y) ** 2 + (xx - centre_x) ** 2
    weights = intensity / intensity.sum()
    return float(np.sqrt((radius2 * weights).sum()))


def run() -> TutorialResult:
    result = TutorialResult()

    # -- the example as published ----------------------------------------------
    scalar_pupil, scalar_field = _psf(128, scalar=True)
    psf_scalar = np.abs(np.asarray(scalar_field.u).squeeze()) ** 2
    vector_pupil, vector_field = _psf(128, scalar=False)
    psf_vectorial = np.abs(np.asarray(vector_field.u).squeeze()) ** 2

    result.record(
        camera_shape=list(CAMERA_SHAPE),
        camera_pixel_pitch=CAMERA_PIXEL_PITCH,
        focal_length=F,
        numerical_aperture=NA,
        refractive_index=N,
        wavelength=WAVELENGTH,
        pupil_correction=F * NA / N,
        scalar_pupil_class=type(scalar_pupil).__name__,
        vector_pupil_class=type(vector_pupil).__name__,
        scalar_psf_shape=list(psf_scalar.shape),
        vectorial_psf_shape=list(psf_vectorial.shape),
    )
    result.check_finite("scalar_psf_finite", psf_scalar)
    result.check_finite("vectorial_psf_finite", psf_vectorial)
    result.check_true(
        "the_example_produces_the_requested_camera_shape",
        "invariant",
        psf_scalar.shape == CAMERA_SHAPE and psf_vectorial.shape == (*CAMERA_SHAPE, 3),
        f"scalar PSF {psf_scalar.shape} and vectorial PSF {psf_vectorial.shape} against a "
        f"requested camera shape of {CAMERA_SHAPE}",
    )
    result.check_true(
        "the_vector_pupil_is_a_VectorField",
        "invariant",
        type(vector_pupil).__name__.startswith("Vector"),
        f"gaussian_plane_wave(scalar=False) gives {type(vector_pupil).__name__}",
    )

    # -- component order, from a second entry point -----------------------------
    pupil_component_energy = [
        float(np.sum(np.abs(np.asarray(vector_pupil.u)[..., i]) ** 2)) for i in range(3)
    ]
    pupil_total = sum(pupil_component_energy)
    result.record(
        pupil_component_energy_fraction=[e / pupil_total for e in pupil_component_energy]
    )
    result.check_true(
        "amplitude_0_0_1_gives_an_x_polarized_pupil",
        "analytic",
        pupil_component_energy[2] / pupil_total > 0.999,
        f"energy fractions per component index {[round(e / pupil_total, 9) for e in pupil_component_energy]} "
        "for amplitude=[0, 0, 1]: all at index 2, confirming the (E_z, E_y, E_x) order "
        "independently of c11_polarized_multislice, which reached it through cf.linear(0).",
    )

    intensity_x = psf_vectorial[..., 2]
    intensity_y = psf_vectorial[..., 1]
    intensity_z = psf_vectorial[..., 0]
    iz_over_ix = float(intensity_z.sum() / intensity_x.sum())
    result.record(
        vectorial_Ix_total=float(intensity_x.sum()),
        vectorial_Iy_total=float(intensity_y.sum()),
        vectorial_Iz_total=float(intensity_z.sum()),
        Iz_over_Ix=iz_over_ix,
    )
    result.check_true(
        "the_vectorial_calculation_generates_longitudinal_field",
        "invariant",
        iz_over_ix > 1e-3,
        f"Iz/Ix = {iz_over_ix:.6f} with an x-polarized pupil at NA {NA}. A high-NA focus "
        "must have a longitudinal component, so the function is not the scalar case in "
        "disguise -- it is a vectorial calculation that happens to be wrong (see below).",
    )

    # -- the executable guard on the known defect -------------------------------
    sweep = {}
    for pupil_pixels in (128, 192, 256):
        _, field = _psf(pupil_pixels, scalar=False)
        components = np.abs(np.asarray(field.u).squeeze()) ** 2
        sweep[str(pupil_pixels)] = {
            "Iz_over_Ix": float(components[..., 0].sum() / components[..., 2].sum()),
            "Ex_radial_width_px": _radial_width(components[..., 2]),
            "Ez_radial_width_px": _radial_width(components[..., 0]),
        }
    ratios = [entry["Iz_over_Ix"] for entry in sweep.values()]
    widths = [entry["Ez_radial_width_px"] for entry in sweep.values()]
    result.record(pupil_sampling_sweep=sweep)
    result.check_true(
        "high_na_ff_lens_is_not_sampling_independent",
        "analytic",
        max(ratios) / min(ratios) > 1.2 or max(widths) / min(widths) > 1.2,
        "refining ONLY the pupil sampling (128 -> 192 -> 256 px) with wavelength, NA, "
        "focal length, camera shape and camera pitch all fixed gives Iz/Ix = "
        + ", ".join(f"{r:.6f}" for r in ratios)
        + " and |E_z| second-moment radii of "
        + ", ".join(f"{w:.3f}" for w in widths)
        + " px. A correct high-NA focusing calculation converges under pupil refinement; "
        "this one does not. Root cause (CHE-18): high_na_ff_lens derives s_z from "
        "field.f_grid * lambda / n -- the FREQUENCY grid -- rather than the pupil position "
        "grid, so the obliquity Jacobian, the defocus phase and the output zoom factor all "
        "degenerate to constants. This check is a guard, not a validation: it should start "
        "failing if the bug is ever fixed upstream, which is the signal to re-validate "
        "against the Richards-Wolf oracle in archive/benchmarks/gen1/benchmarks/L1-WAVE-01.",
    )
    result.note(
        "Do NOT use high_na_ff_lens for quantitative work in this pinned commit. "
        "usage_notes.md records the full CHE-18 evidence: refining only the pupil "
        "sampling moves the |E_z| ring radius from 246 nm to 2536 nm against an oracle "
        "value of 197 nm, and the best achievable vector overlap with an independent "
        "float64 Richards-Wolf quadrature was 0.070. What IS correct there is "
        "cartesian_to_spherical (the aplanatic polarization transform), and note that no "
        "sqrt(cos theta) apodization is applied anywhere -- the caller must supply it."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
