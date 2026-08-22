"""Intermediate / "Monte Carlo Raytracing Methods" -- https://www.optiland.org/tutorials/monte-carlo-ray-tracing

Repo-owned reproduction of the manufacturing-tolerance Monte Carlo tutorial:
sample a plano-convex singlet's radius of curvature (100 +/- 0.5 mm, 1 sigma) and
refractive index (1.5 +/- 1e-3), rebuild it once per draw, and collect RMS spot
radius (`analysis.SpotDiagram.rms_spot_radius`) and RMS wavefront error
(`wavefront.OPD(...).rms()`) into distributions, then correlate them.

**Adaptation.** Upstream draws ``num_systems = 1000``. This reproduction draws
200 from the same ``np.random.seed(42)`` stream so it stays inside the repository's
test budget; the reduction is recorded in the metrics and the seed makes the draw
exactly reproducible. Upstream's seaborn correlation heatmap is replaced by the
correlation matrix itself, which is the quantity the heatmap displays.

Upstream publishes no numbers. The validation here is a physical model of what
the Monte Carlo *must* find, which the tutorial never states.

The nominal system sits at paraxial focus -- ``EFL = R/(n-1) = 200 mm`` and
``BFL = EFL - t/n = 200 - 5/1.5 = 196.667 mm``, exactly the fixed back spacing --
so a draw's only first-order effect is defocus:

    dz(R, n) = R/(n-1) - t/n - 196.667

computed here without Optiland. Because the image plane is fixed while the focus
moves, the *image-plane* shift is ``-dz``. So the whole two-parameter tolerance
study must collapse onto a **one-dimensional through-focus curve**, and this
reproduction checks exactly that: the sampled RMS spot radii are compared
point-by-point against an independent scan that perturbs nothing but the back
spacing, evaluated at ``-dz``. Measured Pearson r = 0.998.

Two things fall out that the tutorial does not mention:

* The RMS spot radius is **not** monotonic in ``|dz|``, and is minimized at
  ``dz = +0.72 mm``, not at 0. The offset is the spherical-aberration best focus
  of an f/7.9 plano-convex singlet -- so a naive "correlate the metric with the
  perturbation" reading of this study gives a misleadingly weak Pearson r of 0.54.
* ``SpotDiagram.rms_spot_radius()`` returns a nested ``[field][wavelength]``
  structure (the tutorial's ``[0][0]``), and ``OPD.rms()`` is in **waves**.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t11_monte_carlo_raytracing",
    title="Monte Carlo Raytracing Methods",
    level="intermediate",
    url="https://www.optiland.org/tutorials/monte-carlo-ray-tracing",
    demonstrates=(
        "Rebuilding an Optic per Monte Carlo draw with materials.IdealMaterial, "
        "analysis.SpotDiagram.rms_spot_radius()[field][wavelength], and "
        "wavefront.OPD(optic, field=(Hx,Hy), wavelength=...).rms() in waves."
    ),
    slow=True,
)

# Upstream uses 1000; 200 keeps the reproduction inside the test budget.
NUM_SYSTEMS = 200
SEED = 42
NOMINAL_RADIUS_MM = 100.0
NOMINAL_INDEX = 1.5
CENTER_THICKNESS_MM = 5.0
BACK_SPACING_MM = 196.667
EPD_MM = 25.4
WAVELENGTH_UM = 0.55


def build_singlet(radius_of_curvature: float, refractive_index: float):
    from optiland import materials, optic

    lens = optic.Optic()
    ideal_material = materials.IdealMaterial(n=refractive_index, k=0)
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        thickness=CENTER_THICKNESS_MM,
        radius=radius_of_curvature,
        is_stop=True,
        material=ideal_material,
    )
    lens.surfaces.add(index=2, thickness=BACK_SPACING_MM)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def predicted_defocus_mm(radius_of_curvature: float, refractive_index: float) -> float:
    """Paraxial back-focal-distance error of a plano-convex singlet, no Optiland."""
    efl = radius_of_curvature / (refractive_index - 1.0)
    bfl = efl - CENTER_THICKNESS_MM / refractive_index
    return bfl - BACK_SPACING_MM


def run() -> TutorialResult:
    from optiland import analysis, wavefront

    result = TutorialResult()
    np.random.seed(SEED)
    refractive_index = np.random.normal(loc=NOMINAL_INDEX, scale=1e-3, size=NUM_SYSTEMS)
    radius_of_curvature = np.random.normal(loc=NOMINAL_RADIUS_MM, scale=0.5, size=NUM_SYSTEMS)

    rms_spot_radius = []
    rms_wavefront_error = []
    for k in range(NUM_SYSTEMS):
        lens = build_singlet(radius_of_curvature[k], refractive_index[k])
        spot = analysis.SpotDiagram(lens)
        rms_spot_radius.append(float(np.asarray(spot.rms_spot_radius()[0][0]).ravel()[0]))
        opd = wavefront.OPD(lens, field=(0, 0), wavelength=WAVELENGTH_UM)
        rms_wavefront_error.append(float(np.asarray(opd.rms()).ravel()[0]))

    rms_spot = np.asarray(rms_spot_radius, dtype=float)
    rms_wfe = np.asarray(rms_wavefront_error, dtype=float)
    defocus = np.asarray(
        [predicted_defocus_mm(r, n) for r, n in zip(radius_of_curvature, refractive_index, strict=True)],
        dtype=float,
    )

    result.record(
        num_systems=NUM_SYSTEMS,
        seed=SEED,
        upstream_num_systems=1000,
        radius_mean_mm=float(radius_of_curvature.mean()),
        radius_std_mm=float(radius_of_curvature.std(ddof=1)),
        index_mean=float(refractive_index.mean()),
        index_std=float(refractive_index.std(ddof=1)),
        rms_spot_mean_mm=float(rms_spot.mean()),
        rms_spot_std_mm=float(rms_spot.std(ddof=1)),
        rms_spot_min_mm=float(rms_spot.min()),
        rms_spot_max_mm=float(rms_spot.max()),
        rms_wfe_mean_waves=float(rms_wfe.mean()),
        rms_wfe_min_waves=float(rms_wfe.min()),
        rms_wfe_max_waves=float(rms_wfe.max()),
        predicted_defocus_min_mm=float(defocus.min()),
        predicted_defocus_max_mm=float(defocus.max()),
    )
    result.check_finite("rms_spot_distribution_finite", rms_spot)
    result.check_finite("rms_wavefront_distribution_finite", rms_wfe)
    result.check_true(
        "every_sampled_system_has_a_positive_spot_size",
        "invariant",
        bool(np.all(rms_spot > 0.0)),
        f"RMS spot radius in [{rms_spot.min():.6f}, {rms_spot.max():.6f}] mm",
    )
    result.check_true(
        "sampling_reproduces_the_declared_input_statistics",
        "invariant",
        abs(radius_of_curvature.mean() - NOMINAL_RADIUS_MM) < 5.0 * 0.5 / np.sqrt(NUM_SYSTEMS)
        and abs(refractive_index.mean() - NOMINAL_INDEX) < 5.0 * 1e-3 / np.sqrt(NUM_SYSTEMS),
        f"radius {radius_of_curvature.mean():.4f} +/- {radius_of_curvature.std(ddof=1):.4f} mm "
        f"(declared 100 +/- 0.5), index {refractive_index.mean():.6f} +/- "
        f"{refractive_index.std(ddof=1):.6f} (declared 1.5 +/- 0.001)",
    )

    # -- collapse onto a one-dimensional through-focus curve ------------------
    # The image plane is fixed while the focus moves, so a draw with predicted
    # focal shift dz is equivalent to shifting the IMAGE PLANE by -dz. Build that
    # reference by perturbing nothing but the back spacing of the nominal lens.
    def rms_at_image_shift(shift_mm: float) -> float:
        lens = build_singlet(NOMINAL_RADIUS_MM, NOMINAL_INDEX)
        lens.set_thickness(BACK_SPACING_MM + shift_mm, surface_number=2)
        return float(
            np.asarray(analysis.SpotDiagram(lens).rms_spot_radius()[0][0]).ravel()[0]
        )

    reference = np.asarray([rms_at_image_shift(-dz) for dz in defocus], dtype=float)
    corr_reference = float(np.corrcoef(rms_spot, reference)[0, 1])
    max_rel_dev = float(np.max(np.abs(rms_spot - reference) / reference))
    result.record(
        corr_mc_vs_pure_defocus_reference=corr_reference,
        max_relative_deviation_vs_reference=max_rel_dev,
        corr_abs_defocus_vs_rms_spot=float(np.corrcoef(np.abs(defocus), rms_spot)[0, 1]),
        corr_rms_spot_vs_rms_wfe=float(np.corrcoef(rms_spot, rms_wfe)[0, 1]),
    )
    result.check_true(
        "the_two_parameter_tolerance_study_collapses_onto_a_through_focus_curve",
        "analytic",
        corr_reference > 0.99,
        f"Pearson r = {corr_reference:.5f} between the sampled RMS spot radii and an "
        "independent back-spacing-only scan evaluated at -dz(R, n), with dz computed "
        "from the thick-lens formula outside Optiland",
    )
    result.check_true(
        "each_draw_agrees_with_its_equivalent_pure_defocus_within_a_few_percent",
        "analytic",
        max_rel_dev < 0.15,
        f"max relative deviation {max_rel_dev * 100:.1f}% over {NUM_SYSTEMS} draws; the "
        "residual is the second-order change in spherical aberration as R and n move, "
        "which the fixed-nominal reference does not model",
    )

    # The through-focus minimum is displaced from paraxial focus by the
    # spherical-aberration best focus, which is why a naive correlation is weak.
    order = np.argsort(defocus)
    best = int(order[int(np.argmin(rms_spot[order]))])
    result.record(
        defocus_at_minimum_spot_mm=float(defocus[best]),
        minimum_rms_spot_mm=float(rms_spot[best]),
        naive_corr_abs_defocus=float(np.corrcoef(np.abs(defocus), rms_spot)[0, 1]),
    )
    result.check_true(
        "spot_size_is_minimized_away_from_paraxial_focus",
        "analytic",
        float(defocus[best]) > 0.2,
        f"the smallest sampled spot ({float(rms_spot[best]):.6f} mm) occurs at a predicted "
        f"focal shift of {float(defocus[best]):+.4f} mm, not 0: the spherical-aberration "
        "best focus of this f/7.9 plano-convex singlet",
    )
    result.check_true(
        "naive_correlation_with_the_perturbation_understates_the_relationship",
        "analytic",
        float(np.corrcoef(np.abs(defocus), rms_spot)[0, 1]) < 0.8 < corr_reference,
        f"|dz| vs RMS spot gives only r = "
        f"{float(np.corrcoef(np.abs(defocus), rms_spot)[0, 1]):.3f} because the curve is "
        f"V-shaped about a nonzero minimum, while the through-focus model gives "
        f"r = {corr_reference:.3f}",
    )

    corr_spot_wfe = float(np.corrcoef(rms_spot, rms_wfe)[0, 1])
    result.check_true(
        "spot_and_wavefront_metrics_track_each_other",
        "analytic",
        corr_spot_wfe > 0.85,
        f"Pearson r = {corr_spot_wfe:.5f} between two independent Optiland metrics "
        "(geometric spot radius in mm and wavefront error in waves) over the same draws",
    )

    # -- reproducibility of the draw -----------------------------------------
    np.random.seed(SEED)
    index_again = np.random.normal(loc=NOMINAL_INDEX, scale=1e-3, size=NUM_SYSTEMS)
    result.check_true(
        "seeded_numpy_draw_is_reproducible",
        "invariant",
        bool(np.array_equal(index_again, refractive_index)),
        "np.random.seed(42) reproduces the identical index sample, so the whole "
        "Monte Carlo is replayable",
    )
    result.note(
        "SpotDiagram.rms_spot_radius() returns a nested [field][wavelength] "
        "structure, so the tutorial's `[0][0]` is (first field, first wavelength) -- "
        "not a flat array index. wavefront.OPD(...).rms() is in WAVES at the "
        "requested wavelength, not mm."
    )
    result.note(
        "Setting Surface.thickness directly does NOT move the surface: positions are "
        "resolved at construction, so the reference scan uses "
        "Optic.set_thickness(value, surface_number=...) instead."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
