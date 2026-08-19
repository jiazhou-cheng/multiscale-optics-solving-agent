"""Intermediate / "Surface Roughness & Scattering" -- https://www.optiland.org/tutorials/surface-roughness-scattering

Repo-owned reproduction of the BSDF tutorial: the same N-SF11 singlet with its
rear surface given no BSDF, a ``scatter.GaussianBSDF(sigma=0.01)``, and a
``scatter.LambertianBSDF()``, comparing the intensity-weighted ray distribution on
the image plane.

**Two adaptations, both recorded as metrics.** Upstream traces 1,000,000 random
rays per case; this reproduction traces a 10921-ray hexapolar fill, which is
enough for the statistics below and fits the test budget. And upstream's
``distribution="random"`` is replaced by ``"hexapolar"`` so the *launch* is
deterministic -- but the scattering itself is **not** reproducible even so:
``optiland.scatter`` is numba-compiled (`njit`/`prange`) and draws from a RNG that
``np.random.seed`` does not reach. Two identical calls give RMS radii of 0.894 and
0.863 mm. This reproduction therefore declares ``metric_rtol`` and asserts only
statistically stable properties. That non-reproducibility is itself the most
important thing learned here and is recorded in `failure_guide.md`.

What is checked:

* **Scattering conserves total ray weight exactly.** ``sum(rays.i)`` is bit-identical
  across the unscattered, Gaussian and Lambertian cases. A BSDF redirects rays; it
  does not create or destroy energy in Optiland's model.
* The Gaussian BSDF broadens the spot, and by roughly the amount its sigma
  predicts: ``sigma * L`` over the 50 mm back distance, combined in quadrature with
  the unscattered RMS.
* The Lambertian BSDF is *radically* more diffuse -- two orders of magnitude wider
  than the Gaussian -- because a cosine-distributed exitance fills the hemisphere
  rather than a narrow lobe.
* Ray count and finiteness are unchanged: no ray is lost to scattering.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t21_surface_roughness_scattering",
    title="Surface Roughness & Scattering",
    level="intermediate",
    url="https://www.optiland.org/tutorials/surface-roughness-scattering",
    demonstrates=(
        "the per-surface bsdf= kwarg with scatter.GaussianBSDF(sigma) and "
        "scatter.LambertianBSDF(), and the fact that optiland.scatter's numba "
        "RNG is NOT controlled by np.random.seed."
    ),
    slow=True,
    # Stochastic: see the module docstring. The scattered RMS radii vary by a few
    # per cent run to run, so recorded metrics replay at a statistical budget.
    metric_rtol=0.35,
)

WAVELENGTH_UM = 0.58756180
NUM_RAYS = 60  # hexapolar rings -> 10921 rays
BACK_DISTANCE_MM = 50.0
GAUSSIAN_SIGMA = 0.01


def build_singlet(bsdf):
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(index=1, thickness=7, radius=50, is_stop=True, material="N-SF11")
    lens.surfaces.add(index=2, thickness=BACK_DISTANCE_MM, bsdf=bsdf)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25.4)
    lens.fields.set_type(field_type="angle")
    for y in (0, 10, 14):
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.48613270)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    lens.wavelengths.add(value=0.65627250)
    lens.image_solve()
    return lens


def _distribution(bsdf) -> dict[str, float]:
    lens = build_singlet(bsdf)
    rays = lens.trace(
        Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS, distribution="hexapolar"
    )
    x = np.asarray(rays.x, dtype=float)
    y = np.asarray(rays.y, dtype=float)
    i = np.asarray(rays.i, dtype=float)
    weight = i.sum()
    return {
        "num_rays": float(x.size),
        "total_weight": float(weight),
        "rms_radius_mm": float(np.sqrt(np.average(x**2 + y**2, weights=i))),
        "p95_radius_mm": float(np.percentile(np.hypot(x, y), 95)),
        # Centroids of a scattered distribution are near-zero random numbers, so
        # they are deliberately NOT recorded: a relative tolerance cannot bound a
        # quantity whose true value is zero. Their smallness is asserted below.
        "centroid_offset_over_rms": float(
            np.hypot(np.average(x, weights=i), np.average(y, weights=i))
            / np.sqrt(np.average(x**2 + y**2, weights=i))
        ),
        "all_finite": float(bool(np.all(np.isfinite(np.concatenate([x, y, i]))))),
    }


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import scatter

    result = TutorialResult()
    none = _distribution(None)
    gaussian = _distribution(scatter.GaussianBSDF(sigma=GAUSSIAN_SIGMA))
    lambertian = _distribution(scatter.LambertianBSDF())
    # `centroid_offset_over_rms` is a near-zero random quantity for the scattered
    # cases, so it is asserted below but recorded only as a verdict: no relative
    # tolerance can bound a number whose true value is zero.
    def _recordable(stats: dict[str, float]) -> dict[str, float | bool]:
        out = {k: v for k, v in stats.items() if k != "centroid_offset_over_rms"}
        out["centroid_is_on_axis"] = bool(stats["centroid_offset_over_rms"] < 0.1)
        return out

    result.record(
        no_scatter=_recordable(none),
        gaussian=_recordable(gaussian),
        lambertian=_recordable(lambertian),
    )

    for label, stats in (("no_scatter", none), ("gaussian", gaussian), ("lambertian", lambertian)):
        result.check_true(
            f"{label}_spot_stays_centred_on_axis",
            "analytic",
            stats["centroid_offset_over_rms"] < 0.1,
            f"|centroid| is {stats['centroid_offset_over_rms']:.4f} of the RMS radius: an "
            "isotropic BSDF on an on-axis rotationally symmetric system cannot shift the "
            "centroid, only broaden the distribution",
        )
        result.check_true(
            f"{label}_distribution_is_finite",
            "invariant",
            stats["all_finite"] == 1.0,
            "all image-plane coordinates and intensities are finite",
        )
        result.check_true(
            f"{label}_loses_no_rays",
            "invariant",
            stats["num_rays"] == none["num_rays"],
            f"{int(stats['num_rays'])} rays reach the image plane, same as the "
            f"{int(none['num_rays'])} unscattered",
        )

    result.check_true(
        "scattering_conserves_total_ray_weight_exactly",
        "analytic",
        gaussian["total_weight"] == none["total_weight"] == lambertian["total_weight"],
        f"sum(rays.i) = {none['total_weight']!r} for all three cases, bit-identical. A "
        "BSDF redirects rays in Optiland's model; it neither creates nor destroys "
        "energy, so any downstream throughput estimate must come from the spatial "
        "distribution, not from the total.",
    )

    # sigma * L is the added transverse spread over the back distance.
    predicted_added_mm = GAUSSIAN_SIGMA * BACK_DISTANCE_MM
    predicted_total_mm = float(np.hypot(none["rms_radius_mm"], predicted_added_mm))
    result.record(
        predicted_gaussian_added_spread_mm=predicted_added_mm,
        predicted_gaussian_rms_mm=predicted_total_mm,
        gaussian_over_predicted=gaussian["rms_radius_mm"] / predicted_total_mm,
    )
    result.check_true(
        "gaussian_bsdf_broadens_the_spot_by_about_sigma_times_the_back_distance",
        "analytic",
        0.5 < gaussian["rms_radius_mm"] / predicted_total_mm < 3.0,
        f"RMS radius {none['rms_radius_mm']:.4f} mm unscattered -> "
        f"{gaussian['rms_radius_mm']:.4f} mm with sigma={GAUSSIAN_SIGMA}, against the "
        f"{predicted_total_mm:.4f} mm predicted by combining the unscattered spot with "
        f"sigma*L = {predicted_added_mm:.4f} mm in quadrature (ratio "
        f"{gaussian['rms_radius_mm'] / predicted_total_mm:.3f}). Order-of-magnitude "
        "agreement only: sigma's exact parameterization inside the numba kernel is not "
        "documented upstream.",
    )
    result.check_true(
        "gaussian_scattering_strictly_broadens_the_spot",
        "analytic",
        gaussian["rms_radius_mm"] > 2.0 * none["rms_radius_mm"],
        f"{gaussian['rms_radius_mm']:.4f} mm > 2 x {none['rms_radius_mm']:.4f} mm",
    )
    result.check_true(
        "lambertian_scattering_is_two_orders_wider_than_gaussian",
        "analytic",
        lambertian["rms_radius_mm"] > 50.0 * gaussian["rms_radius_mm"],
        f"RMS radius {lambertian['rms_radius_mm']:.2f} mm (Lambertian) vs "
        f"{gaussian['rms_radius_mm']:.4f} mm (Gaussian sigma=0.01): a cosine-distributed "
        "exitance fills the hemisphere rather than a narrow lobe",
    )

    # -- non-reproducibility, measured ----------------------------------------
    trials = []
    for _ in range(2):
        np.random.seed(7)
        trials.append(_distribution(scatter.GaussianBSDF(sigma=GAUSSIAN_SIGMA))["rms_radius_mm"])
    spread = abs(trials[0] - trials[1]) / max(trials)
    # The trial values themselves are not recorded: they are the very thing that
    # is not reproducible. Only the verdict is.
    result.record(
        bsdf_scattering_is_reproducible_under_numpy_seed=bool(spread <= 1e-6),
        num_reproducibility_trials=len(trials),
    )
    result.check_true(
        "numpy_seeding_does_not_make_bsdf_scattering_reproducible",
        "invariant",
        spread > 1e-6,
        f"two runs with np.random.seed(7) set immediately before each trace give RMS "
        f"radii {trials[0]:.6f} and {trials[1]:.6f} mm ({spread * 100:.2f}% apart). "
        "optiland.scatter is numba-compiled and its RNG is not reachable from numpy, so "
        "a scattering result CANNOT be frozen as a bit-exact repository fixture.",
    )

    lens = build_singlet(scatter.GaussianBSDF(sigma=GAUSSIAN_SIGMA))
    rays = lens.trace(
        Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=12, distribution="hexapolar"
    )
    plt.hist2d(
        np.asarray(rays.x, dtype=float),
        np.asarray(rays.y, dtype=float),
        weights=np.asarray(rays.i, dtype=float),
        bins=32,
    )
    plt.close("all")
    result.check_true(
        "the_tutorials_weighted_hist2d_renders_headless",
        "qualitative",
        True,
        "plt.hist2d(x, y, weights=rays.i) completed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
