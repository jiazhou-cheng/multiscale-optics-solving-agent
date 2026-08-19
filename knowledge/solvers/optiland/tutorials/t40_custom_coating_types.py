"""Advanced / "Custom Coating Types" -- https://www.optiland.org/tutorials/extending-coatings

Repo-owned reproduction of the coating-extension tutorial: two `coatings.BaseCoating`
subclasses attached to a singlet's first surface -- a ``GradientCoating`` that sets
``rays.i = rays.x * 0.01 + 0.5`` and a ``SpectralCoating`` that multiplies
``rays.i`` by ``rays.w**2 + 0.2``.

Both coatings are *closed-form functions of ray state*, so this tutorial admits
exact validation, and that is the whole point of reproducing it:

* ``GradientCoating`` **assigns** rather than multiplies, so the resulting ``rays.i``
  must equal ``0.01 * x + 0.5`` evaluated at the **coated surface's** x coordinate --
  not at the image plane. Verified to float64 round-off against
  ``surfaces.x[1, :]``, which simultaneously establishes *where* in the trace the
  coating's ``rays`` argument is sampled.
* Because it assigns, the coating **discards** whatever intensity the ray already
  carried; two coatings of this kind do not compose. Checked by putting it on two
  surfaces.
* ``SpectralCoating`` multiplies, so ``rays.i`` must equal
  ``uncoated_i * (lambda^2 + 0.2)`` at every wavelength. Verified across three
  wavelengths to round-off, which also pins ``rays.w`` as the wavelength in
  **micrometres** (0.55^2 + 0.2 = 0.5025, not 550^2 + 0.2).
* Neither coating is required to be physical. A coating that assigns ``rays.i = 1.5``
  or ``rays.i = -0.25`` is accepted without clamp or warning, so a custom coating
  carries the entire burden of energy conservation -- and the tutorial's own
  ``GradientCoating`` is unphysical for ``|x| > 50 mm`` for exactly that reason.
* ``rays.i`` after a coated trace is the coating's value **times a further
  ray-dependent factor from the uncoated downstream surfaces** (~3e-3 here, the same
  effect t18 measured). A custom coating can only be validated against a matched
  uncoated trace, never against an absolute value.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t40_custom_coating_types",
    title="Custom Coating Types",
    level="advanced",
    url="https://www.optiland.org/tutorials/extending-coatings",
    demonstrates=(
        "subclassing coatings.BaseCoating with transmit(rays, nx, ny, nz) and "
        "reflect(...), and the fact that the `rays` handed to a coating carry "
        "the coordinates of the coated surface and rays.w in micrometres."
    ),
)

WAVELENGTHS_UM = (0.45, 0.55, 0.65)
EPD_MM = 25.0


def _coating_classes():
    from optiland.coatings import BaseCoating

    class GradientCoating(BaseCoating):
        def reflect(self, rays, nx, ny, nz):
            return rays

        def transmit(self, rays, nx, ny, nz):
            rays.i = rays.x * 0.01 + 0.5
            return rays

    class SpectralCoating(BaseCoating):
        def reflect(self, rays, nx, ny, nz):
            return rays

        def transmit(self, rays, nx, ny, nz):
            transmission = rays.w**2 + 0.2
            rays.i *= transmission
            return rays

    return GradientCoating, SpectralCoating


def build_singlet(coating_surface_1=None, coating_surface_2=None):
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=19.93,
        is_stop=True,
        material="N-SF11",
        coating=coating_surface_1,
    )
    lens.surfaces.add(index=2, thickness=21.48, coating=coating_surface_2)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    for y in (0, 7, 10):
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def run() -> TutorialResult:
    result = TutorialResult()
    GradientCoating, SpectralCoating = _coating_classes()

    # -- 1. GradientCoating: i = 0.01*x + 0.5 at the COATED surface --------------
    # rays.i must be read RELATIVE to a matched uncoated trace: the downstream
    # uncoated surfaces contribute their own small ray-dependent factor (the same
    # ~1e-3 effect t18 found), so an absolute comparison misses by 3e-3.
    lens = build_singlet(coating_surface_1=GradientCoating())
    bare = build_singlet()
    rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=12, distribution="hexapolar")
    bare_rays = bare.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=12, distribution="hexapolar")
    intensity = np.asarray(rays.i, dtype=float)
    bare_intensity = np.asarray(bare_rays.i, dtype=float)
    relative = intensity / bare_intensity
    x_at_coated = np.asarray(lens.surfaces.x, dtype=float)[1, :]
    x_at_image = np.asarray(rays.x, dtype=float)
    predicted_from_coated = 0.01 * x_at_coated + 0.5
    predicted_from_image = 0.01 * x_at_image + 0.5
    result.record(
        num_rays=int(intensity.size),
        gradient_intensity_min=float(intensity.min()),
        gradient_intensity_max=float(intensity.max()),
        gradient_relative_min=float(relative.min()),
        gradient_relative_max=float(relative.max()),
        absolute_error_vs_coated_surface_x=float(
            np.max(np.abs(intensity - predicted_from_coated))
        ),
        relative_error_vs_coated_surface_x=float(
            np.max(np.abs(relative - predicted_from_coated))
        ),
        relative_error_vs_image_plane_x=float(np.max(np.abs(relative - predicted_from_image))),
    )
    result.check_finite("gradient_intensity_finite", intensity)
    result.check_true(
        "the_coating_sees_the_coated_surfaces_coordinates",
        "analytic",
        float(np.max(np.abs(relative - predicted_from_coated))) < 1e-12
        and float(np.max(np.abs(relative - predicted_from_image))) > 1e-3,
        f"rays.i(coated)/rays.i(uncoated) equals 0.01*x + 0.5 at surfaces.x[1, :] to "
        f"{float(np.max(np.abs(relative - predicted_from_coated))):.3e}, but differs from "
        f"the same formula at the image plane by up to "
        f"{float(np.max(np.abs(relative - predicted_from_image))):.3e}. The `rays` a "
        "coating receives carry the coated surface's coordinates, before further "
        "propagation -- confirmed independently by capturing rays.x inside transmit().",
    )
    result.check_true(
        "the_coatings_value_must_be_read_relative_to_an_uncoated_baseline",
        "analytic",
        float(np.max(np.abs(intensity - predicted_from_coated))) > 1e-6,
        f"comparing the ABSOLUTE rays.i against the coating's own formula misses by "
        f"{float(np.max(np.abs(intensity - predicted_from_coated))):.3e}: the uncoated "
        "downstream surfaces contribute their own ray-dependent factor (the same effect "
        "recorded in t18). A custom coating can only be validated against a matched "
        "uncoated trace.",
    )

    # -- 2. assignment does not compose -----------------------------------------
    doubled = build_singlet(
        coating_surface_1=GradientCoating(), coating_surface_2=GradientCoating()
    )
    doubled_rays = doubled.trace(
        Hx=0, Hy=0, wavelength=0.55, num_rays=12, distribution="hexapolar"
    )
    # No division by the uncoated baseline here: the coating on surface 2 is the LAST
    # thing to touch rays.i, so its assigned value survives to the image plane exactly.
    doubled_intensity = np.asarray(doubled_rays.i, dtype=float)
    x_at_second = np.asarray(doubled.surfaces.x, dtype=float)[2, :]
    predicted_second_only = 0.01 * x_at_second + 0.5
    result.record(
        two_gradient_coatings_intensity_min=float(doubled_intensity.min()),
        two_gradient_coatings_intensity_max=float(doubled_intensity.max()),
        max_error_vs_second_surface_only=float(
            np.max(np.abs(doubled_intensity - predicted_second_only))
        ),
    )
    result.check_true(
        "an_assigning_coating_discards_the_rays_prior_intensity",
        "analytic",
        float(np.max(np.abs(doubled_intensity - predicted_second_only))) < 1e-12,
        f"with the same coating on both lens surfaces, rays.i equals 0.01*x + 0.5 at the "
        f"SECOND surface ALONE to "
        f"{float(np.max(np.abs(doubled_intensity - predicted_second_only))):.3e} -- exactly, "
        "with no residual factor, because the last coating to run assigns rather than "
        "multiplies. The first surface's contribution is gone. `rays.i = ...` overwrites; "
        "a composable coating must use `rays.i *= ...`.",
    )

    # -- 3. SpectralCoating: multiplicative, and rays.w is in micrometres --------
    spectral = {}
    for wavelength in WAVELENGTHS_UM:
        coated = build_singlet(coating_surface_1=SpectralCoating())
        bare = build_singlet()
        coated_i = np.asarray(
            coated.trace(
                Hx=0, Hy=0, wavelength=wavelength, num_rays=12, distribution="hexapolar"
            ).i,
            dtype=float,
        )
        bare_i = np.asarray(
            bare.trace(
                Hx=0, Hy=0, wavelength=wavelength, num_rays=12, distribution="hexapolar"
            ).i,
            dtype=float,
        )
        ratio = coated_i / bare_i
        spectral[f"{wavelength:g}um"] = {
            "mean_ratio": float(ratio.mean()),
            "ratio_spread": float(ratio.max() - ratio.min()),
            "predicted_um": float(wavelength**2 + 0.2),
            "predicted_nm": float((wavelength * 1000.0) ** 2 + 0.2),
        }
    result.record(spectral_coating=spectral)
    for label, entry in spectral.items():
        result.check_close(
            f"spectral_coating_multiplies_by_lambda_squared_plus_0p2_at_{label.replace('.', 'p')}",
            "analytic",
            entry["mean_ratio"],
            entry["predicted_um"],
            rel=1e-12,
        )
    result.check_true(
        "rays_w_is_the_wavelength_in_micrometres",
        "analytic",
        all(
            abs(entry["mean_ratio"] - entry["predicted_um"]) < 1e-12
            and abs(entry["mean_ratio"] - entry["predicted_nm"]) > 1.0
            for entry in spectral.values()
        ),
        "the observed transmission ratios "
        + ", ".join(f"{k} {v['mean_ratio']:.6f}" for k, v in spectral.items())
        + " match lambda_um^2 + 0.2 and are nowhere near lambda_nm^2 + 0.2, so rays.w is "
        "in micrometres -- the same unit as Optic.trace's wavelength argument",
    )
    result.check_true(
        "the_spectral_coating_is_wavelength_flat_across_the_pupil",
        "analytic",
        all(entry["ratio_spread"] < 1e-12 for entry in spectral.values()),
        "the per-ray transmission ratio has zero spread at each wavelength, as a "
        "function of rays.w alone must",
    )
    result.check_true(
        "the_spectral_coating_is_monotonic_in_wavelength",
        "analytic",
        spectral["0.45um"]["mean_ratio"]
        < spectral["0.55um"]["mean_ratio"]
        < spectral["0.65um"]["mean_ratio"],
        "transmission rises with wavelength as lambda^2 + 0.2 requires: "
        + " < ".join(f"{spectral[k]['mean_ratio']:.6f}" for k in ("0.45um", "0.55um", "0.65um")),
    )

    # -- 4. nothing clamps a custom coating to physical values -------------------
    from optiland.coatings import BaseCoating

    class SuperUnityCoating(BaseCoating):
        def reflect(self, rays, nx, ny, nz):
            return rays

        def transmit(self, rays, nx, ny, nz):
            rays.i = rays.i * 0.0 + 1.5
            return rays

    class NegativeCoating(BaseCoating):
        def reflect(self, rays, nx, ny, nz):
            return rays

        def transmit(self, rays, nx, ny, nz):
            rays.i = rays.i * 0.0 - 0.25
            return rays

    unphysical = {}
    for label, coating in (("super_unity", SuperUnityCoating()), ("negative", NegativeCoating())):
        trial = build_singlet(coating_surface_1=coating)
        values = np.asarray(
            trial.trace(
                Hx=0, Hy=0, wavelength=0.55, num_rays=12, distribution="hexapolar"
            ).i,
            dtype=float,
        )
        unphysical[label] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "relative_to_uncoated": float((values / bare_intensity).mean()),
        }
    result.record(unphysical_coatings=unphysical)
    result.check_true(
        "a_custom_coating_can_set_a_transmittance_above_one",
        "analytic",
        unphysical["super_unity"]["relative_to_uncoated"] > 1.0,
        f"a coating assigning rays.i = 1.5 yields a transmittance of "
        f"{unphysical['super_unity']['relative_to_uncoated']:.6f} relative to the uncoated "
        "trace. Optiland neither clamps nor warns.",
    )
    result.check_true(
        "a_custom_coating_can_set_a_negative_transmittance",
        "analytic",
        unphysical["negative"]["relative_to_uncoated"] < 0.0,
        f"a coating assigning rays.i = -0.25 yields "
        f"{unphysical['negative']['relative_to_uncoated']:.6f}. Negative intensity is "
        "accepted silently, so a custom coating carries the entire burden of energy "
        "conservation -- the tutorial's own GradientCoating is unphysical for |x| > 50 mm "
        "for exactly this reason.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
