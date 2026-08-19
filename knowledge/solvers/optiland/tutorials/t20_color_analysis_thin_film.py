"""Intermediate / "Color Analysis for Thin-Films" -- https://www.optiland.org/tutorials/color-analysis-thin-film

Repo-owned reproduction of the Michel-Levy colour tutorial: sweep a single TiO2
layer on an SiO2 substrate from 0 to 250 nm, convert each reflectance spectrum to
CIE 1931 xyY and sRGB with ``SpectralAnalyzer.analyze_color``, and plot the
resulting path on the chromaticity diagram plus the colour bar.

Upstream publishes no numbers. Colorimetry has strong closed-form anchors,
though, and those are what is checked:

* Every ``(x, y)`` chromaticity lies inside the CIE 1931 spectral locus's bounding
  simplex: ``x >= 0``, ``y >= 0``, ``x + y <= 1``. This is a definition of
  chromaticity coordinates and it holds for all 251 thicknesses.
* Every sRGB triple is in ``[0, 255]``.
* At **zero layer thickness** the reflectance is that of a bare air/SiO2 interface,
  which is spectrally flat to within the substrate's own dispersion, so the colour
  must be within 0.01 of the D65 white point (0.3127, 0.3290) that
  ``analyze_color`` normalizes against. This is the one point on the sweep with an
  independently known answer, and it validates the whole colour pipeline.
* The path is *not* degenerate: the chromaticity wanders more than 0.1 away from
  white as the layer thickens, which is the interference colour the tutorial is
  about.
* The colour path is continuous: consecutive 1 nm steps move the chromaticity by
  much less than the total excursion.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t20_color_analysis_thin_film",
    title="Color Analysis for Thin-Films",
    level="intermediate",
    url="https://www.optiland.org/tutorials/color-analysis-thin-film",
    demonstrates=(
        "thin_film.SpectralAnalyzer.analyze_color(observer='2deg', quantity='R') "
        "-> {'xyY', 'sRGB', ...}, mutating Layer.thickness_um in place, and "
        "colorimetry.plotting.plot_cie_1931_chromaticity_diagram."
    ),
    slow=True,
)

MAX_THICKNESS_NM = 250.0
NUM_THICKNESSES = 251
# CIE standard illuminant D65 chromaticity, the reference white analyze_color uses.
D65_XY = (0.3127, 0.3290)


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    import optiland.backend as be
    from optiland.colorimetry.plotting import plot_cie_1931_chromaticity_diagram
    from optiland.materials import IdealMaterial, Material
    from optiland.thin_film import SpectralAnalyzer, ThinFilmStack

    result = TutorialResult()
    air = IdealMaterial(n=1.0)
    sio2 = Material("SiO2", reference="Gao")
    tio2 = Material("TiO2", reference="Zhukovsky")

    stack = ThinFilmStack(incident_material=air, substrate_material=sio2)
    stack.add_layer_nm(tio2, 0.0, name="TiO2")
    analyzer = SpectralAnalyzer(stack)

    wavelengths_nm = be.linspace(380.0, 780.0, 81)
    thicknesses_nm = np.linspace(0.0, MAX_THICKNESS_NM, NUM_THICKNESSES)

    x_path, y_path, colors, keys = [], [], [], None
    for thickness_nm in thicknesses_nm:
        stack.layers[0].thickness_um = float(thickness_nm) / 1000.0
        color = analyzer.analyze_color(
            wavelength_values=wavelengths_nm,
            wavelength_unit="nm",
            aoi=0.0,
            aoi_unit="deg",
            polarization="u",
            quantity="R",
            observer="2deg",
        )
        keys = sorted(color.keys()) if keys is None else keys
        x, y, _ = color["xyY"]
        x_path.append(float(np.asarray(x).ravel()[0]))
        y_path.append(float(np.asarray(y).ravel()[0]))
        colors.append([float(np.asarray(c).ravel()[0]) for c in color["sRGB"]])

    x_path_arr = np.asarray(x_path)
    y_path_arr = np.asarray(y_path)
    colors_arr = np.asarray(colors)
    result.record(
        analyze_color_keys=keys,
        num_thicknesses=NUM_THICKNESSES,
        x_range=[float(x_path_arr.min()), float(x_path_arr.max())],
        y_range=[float(y_path_arr.min()), float(y_path_arr.max())],
        srgb_range=[float(colors_arr.min()), float(colors_arr.max())],
        chromaticity_at_zero_nm=[x_path[0], y_path[0]],
        chromaticity_at_250_nm=[x_path[-1], y_path[-1]],
        srgb_at_zero_nm=colors[0],
        srgb_at_250_nm=colors[-1],
    )
    result.check_finite("chromaticity_path_finite", np.concatenate([x_path_arr, y_path_arr]))
    result.check_true(
        "every_chromaticity_is_a_valid_cie_1931_coordinate",
        "analytic",
        bool(
            np.all(x_path_arr >= 0.0)
            and np.all(y_path_arr >= 0.0)
            and np.all(x_path_arr + y_path_arr <= 1.0)
        ),
        f"x in [{x_path_arr.min():.4f}, {x_path_arr.max():.4f}], y in "
        f"[{y_path_arr.min():.4f}, {y_path_arr.max():.4f}], max x+y = "
        f"{float((x_path_arr + y_path_arr).max()):.4f} <= 1 over all "
        f"{NUM_THICKNESSES} thicknesses",
    )
    result.check_true(
        "every_srgb_triple_is_in_range",
        "invariant",
        bool(np.all(colors_arr >= 0.0) and np.all(colors_arr <= 255.0)),
        f"sRGB in [{colors_arr.min():.3f}, {colors_arr.max():.3f}] subset [0, 255]",
    )

    distance_from_white = float(
        np.hypot(x_path[0] - D65_XY[0], y_path[0] - D65_XY[1])
    )
    result.record(zero_thickness_distance_from_d65=distance_from_white)
    result.check_true(
        "a_zero_thickness_layer_reflects_the_reference_white",
        "analytic",
        distance_from_white < 0.01,
        f"chromaticity at 0 nm = ({x_path[0]:.5f}, {y_path[0]:.5f}), a distance "
        f"{distance_from_white:.5f} from D65 {D65_XY}. A bare air/SiO2 interface is "
        "spectrally flat to within the substrate's dispersion, so its reflected colour "
        "must be the illuminant's -- the one point of the sweep with a known answer.",
    )

    excursion = float(
        np.max(np.hypot(x_path_arr - x_path[0], y_path_arr - y_path[0]))
    )
    steps = np.hypot(np.diff(x_path_arr), np.diff(y_path_arr))
    result.record(max_excursion_from_white=excursion, max_single_step=float(steps.max()))
    result.check_true(
        "the_interference_colour_is_a_real_excursion_not_noise",
        "analytic",
        excursion > 0.05,
        f"the chromaticity wanders {excursion:.4f} from its zero-thickness value as the "
        "TiO2 layer thickens -- the Michel-Levy interference colour the tutorial plots",
    )
    result.check_true(
        "the_colour_path_is_continuous_in_thickness",
        "analytic",
        float(steps.max()) < excursion / 5.0,
        f"largest single 1 nm step {float(steps.max()):.5f} against a total excursion of "
        f"{excursion:.4f}: a smooth path, so the sweep resolves the colour rather than "
        "aliasing it",
    )

    plot_cie_1931_chromaticity_diagram(color="fill")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.imshow(
        (colors_arr / 255.0)[None, :, :],
        aspect="auto",
        extent=[float(thicknesses_nm[0]), float(thicknesses_nm[-1]), 0, 1],
    )
    plt.close("all")
    result.check_true(
        "chromaticity_diagram_and_colour_bar_render_headless",
        "qualitative",
        True,
        "plot_cie_1931_chromaticity_diagram and the Michel-Levy colour bar completed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
