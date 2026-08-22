"""Intermediate / "Chromatic Aberrations" -- https://www.optiland.org/tutorials/chromatic-aberrations

Repo-owned reproduction of the achromat tutorial: an uncorrected N-SF6 singlet
and Laikin's cemented N-BAK1/SF2 achromatic doublet, both at EPD 3.4 over the
F/d/C lines, compared by first-order longitudinal colour
(``sum(aberrations.LchC())``) and ray fan.

**This is one of the few Optiland tutorials that publishes reference numbers**,
and they are the primary oracle here:

    singlet  First-order Longitudinal Color: -0.789
    doublet  First-order Longitudinal Color: -0.015

Both are reproduced to the three decimal places upstream prints. Beyond that:

* The doublet's residual is 50x smaller than the singlet's -- the entire point of
  cementing a flint to a crown.
* Independent confirmation from a real polychromatic trace, without touching
  ``aberrations``: the F-to-C spread of the paraxial back focus is measured
  directly for both systems and must collapse by the same order.
* The singlet's colour is checked against the thin-lens achromat condition
  ``LchC ~ -f/V`` evaluated from ``Material('N-SF6')``'s own Abbe number computed
  here, which ties the coefficient to catalogue dispersion rather than to another
  Optiland accessor.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t15_chromatic_aberrations",
    title="Chromatic Aberrations",
    level="intermediate",
    url="https://www.optiland.org/tutorials/chromatic-aberrations",
    demonstrates=(
        "sum(Optic.aberrations.LchC()) as first-order longitudinal colour, "
        "multi-wavelength systems via repeated wavelengths.add, the material "
        "tuple form ('SF2', 'schott'), and Optic.draw(wavelengths=[...])."
    ),
)

LINE_F = 0.48613270
LINE_D = 0.58756180
LINE_C = 0.65627250
EPD_MM = 3.4

# Published by the tutorial.
UPSTREAM_SINGLET_LCHC = -0.789
UPSTREAM_DOUBLET_LCHC = -0.015


def _finish(lens):
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=LINE_F)
    lens.wavelengths.add(value=LINE_D, is_primary=True)
    lens.wavelengths.add(value=LINE_C)
    return lens


def build_singlet():
    from optiland import optic

    lens = optic.Optic(name="Singlet")
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(index=1, thickness=0.5, radius=32.2526, is_stop=True, material="N-SF6")
    lens.surfaces.add(index=2, thickness=19.8532, radius=-31.9756)
    lens.surfaces.add(index=3)
    return _finish(lens)


def build_doublet():
    """Achromatic doublet: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 45."""
    from optiland import optic

    lens = optic.Optic(name="Doublet")
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(index=1, radius=12.38401, thickness=0.4340, is_stop=True, material="N-BAK1")
    lens.surfaces.add(index=2, radius=-7.94140, thickness=0.3210, material=("SF2", "schott"))
    lens.surfaces.add(index=3, radius=-48.44396, thickness=19.6059)
    lens.surfaces.add(index=4)
    return _finish(lens)


def _back_focus_spread_mm(lens) -> tuple[float, float, float]:
    """(F-line, C-line, spread) paraxial back focal distance, from real marginal rays.

    Traces one near-axial marginal ray per wavelength and extrapolates its
    intersection with the axis, so this never calls `aberrations` or `paraxial`.
    """
    focus = {}
    for wl in (LINE_F, LINE_D, LINE_C):
        rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=wl, num_rays=4)
        y = np.asarray(rays.y, dtype=float)
        m = np.asarray(rays.M, dtype=float)
        n = np.asarray(rays.N, dtype=float)
        # Use the outermost surviving ray; distance to the axis crossing is -y/(M/N).
        idx = int(np.argmax(np.abs(y)))
        focus[wl] = -y[idx] / (m[idx] / n[idx])
    return focus[LINE_F], focus[LINE_C], focus[LINE_F] - focus[LINE_C]


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis
    from optiland.materials import Material

    result = TutorialResult()
    singlet = build_singlet()
    doublet = build_doublet()

    lchc = {}
    for label, lens in (("singlet", singlet), ("doublet", doublet)):
        value = float(np.sum(np.asarray(lens.aberrations.LchC(), dtype=float)))
        lchc[label] = value
        result.record(**{f"{label}_lchc_mm": value})
        result.record(
            **{f"{label}_efl_mm": float(np.asarray(lens.paraxial.f2()).ravel()[0])}
        )

    result.check_close(
        "singlet_longitudinal_colour_matches_upstream_minus_0p789",
        "reference",
        lchc["singlet"],
        UPSTREAM_SINGLET_LCHC,
        abs_=5e-4,
    )
    result.check_close(
        "doublet_longitudinal_colour_matches_upstream_minus_0p015",
        "reference",
        lchc["doublet"],
        UPSTREAM_DOUBLET_LCHC,
        abs_=5e-4,
    )
    result.check_true(
        "the_doublet_is_an_order_of_magnitude_more_achromatic",
        "analytic",
        abs(lchc["doublet"]) < abs(lchc["singlet"]) / 10.0,
        f"|LchC| {abs(lchc['singlet']):.4f} mm (singlet) -> {abs(lchc['doublet']):.4f} mm "
        f"(doublet): {abs(lchc['singlet'] / lchc['doublet']):.0f}x better",
    )

    # -- independent confirmation from a real polychromatic trace ---------------
    spreads = {}
    for label, lens in (("singlet", singlet), ("doublet", doublet)):
        f_focus, c_focus, spread = _back_focus_spread_mm(lens)
        spreads[label] = spread
        result.record(
            **{
                f"{label}_traced_F_line_focus_mm": f_focus,
                f"{label}_traced_C_line_focus_mm": c_focus,
                f"{label}_traced_F_minus_C_focus_mm": spread,
            }
        )
    result.record(traced_spread_ratio=abs(spreads["singlet"] / spreads["doublet"]))
    result.check_true(
        "traced_focal_spread_confirms_the_aberration_coefficient",
        "analytic",
        abs(spreads["doublet"]) < abs(spreads["singlet"]) / 10.0,
        f"F-to-C axial focus spread measured from real traced rays: "
        f"{spreads['singlet']:.4f} mm (singlet) vs {spreads['doublet']:.4f} mm (doublet), a "
        f"{abs(spreads['singlet'] / spreads['doublet']):.0f}x collapse -- reproduced without "
        "calling aberrations at all",
    )
    result.check_true(
        "traced_spread_and_lchc_agree_in_magnitude_for_the_singlet",
        "analytic",
        0.3 < abs(spreads["singlet"] / lchc["singlet"]) < 3.0,
        f"traced F-C spread {spreads['singlet']:.4f} mm vs first-order LchC "
        f"{lchc['singlet']:.4f} mm (ratio "
        f"{abs(spreads['singlet'] / lchc['singlet']):.3f}); they are the same physical "
        "quantity to first order, differing by the higher-order colour the "
        "coefficient omits",
    )

    # -- tie the singlet's colour to catalogue dispersion ----------------------
    nsf6 = Material("N-SF6")
    n_d, n_f, n_c = (
        float(np.asarray(nsf6.n(w)).ravel()[0]) for w in (LINE_D, LINE_F, LINE_C)
    )
    abbe = (n_d - 1.0) / (n_f - n_c)
    efl = float(np.asarray(singlet.paraxial.f2()).ravel()[0])
    predicted = -efl / abbe
    result.record(
        nsf6_n_d=n_d,
        nsf6_abbe_number=abbe,
        singlet_predicted_lchc_from_thin_lens=predicted,
    )
    result.check_close(
        "singlet_colour_matches_the_thin_lens_minus_f_over_V_estimate",
        "analytic",
        lchc["singlet"],
        predicted,
        rel=0.1,
    )

    # -- the tutorial's plots -------------------------------------------------
    for label, lens in (("singlet", singlet), ("doublet", doublet)):
        lens.draw(wavelengths=[LINE_F, LINE_D, LINE_C], figsize=(16, 4), num_rays=3)
        plt.close("all")
        fan = analysis.RayFan(lens)
        fan.view()
        plt.close("all")
        result.check_true(
            f"{label}_draw_and_rayfan_render_headless",
            "qualitative",
            True,
            f"{label}.draw(wavelengths=[...]) and RayFan(...).view() completed",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
