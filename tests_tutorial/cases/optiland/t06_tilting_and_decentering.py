"""Beginner / "Tilting & De-centering Components" -- https://www.optiland.org/tutorials/tilting-and-decentering

Repo-owned reproduction of the misalignment tutorial: the same N-SF11 singlet
built three times -- aligned, with the first surface tilted 5 degrees about x
(`rx=np.radians(5.0)`), and with it decentered 1 mm in y (`dy=1.0`) -- with a
spot diagram for each.

Upstream shows three spot-diagram images and publishes no numbers. The physics
of each perturbation is nonetheless fully determined, and that is what this
reproduction checks:

* Aligned, on axis, with a symmetric pupil fill: the centroid sits exactly on
  the axis and the spot is symmetric in both x and y.
* A tilt **about the x axis** breaks the y symmetry only: the y centroid moves,
  the x centroid stays at zero to round-off, and the y spread grows.
* A **decenter in y** likewise moves the y centroid and leaves x centred. The
  image shift is of order ``dy`` but not bounded by it -- measured 1.285 mm for
  a 1.0 mm decenter, because displacing a *powered* surface adds a prismatic
  deflection on top of the translation.
* Both perturbations strictly increase the RMS spot radius relative to the
  aligned baseline -- a misaligned system cannot image better than an aligned
  one at the same conjugates.
* `analysis.SpotDiagram(lens)` runs headlessly and its RMS agrees with the
  RMS computed directly off the traced rays.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t06_tilting_and_decentering",
    title="Tilting & De-centering Components",
    level="beginner",
    url="https://www.optiland.org/tutorials/tilting-and-decentering",
    demonstrates=(
        "Per-surface coordinate-system perturbations through surfaces.add "
        "kwargs rx/ry/rz (tilts, radians) and dx/dy (decenters, mm), plus "
        "optiland.analysis.SpotDiagram on a non-rotationally-symmetric system."
    ),
)

WAVELENGTH_UM = 0.587
EPD_MM = 25.4


def build(**perturbation):
    """The tutorial's singlet, optionally perturbed on surface 1."""
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1, thickness=7, radius=19.93, is_stop=True, material="N-SF11", **perturbation
    )
    lens.surfaces.add(index=2, thickness=21.48)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def _spot_stats(lens) -> dict[str, float]:
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=12, distribution="hexapolar")
    x = np.asarray(rays.x, dtype=float)
    y = np.asarray(rays.y, dtype=float)
    return {
        "num_rays": float(x.size),
        "centroid_x_mm": float(x.mean()),
        "centroid_y_mm": float(y.mean()),
        "std_x_mm": float(x.std()),
        "std_y_mm": float(y.std()),
        "rms_radius_about_centroid_mm": float(
            np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2))
        ),
    }


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis

    result = TutorialResult()

    aligned = _spot_stats(build())
    tilted = _spot_stats(build(rx=np.radians(5.0)))
    decentered = _spot_stats(build(dy=1.0))
    result.record(aligned=aligned, tilted_rx_5deg=tilted, decentered_dy_1mm=decentered)

    # -- aligned baseline ----------------------------------------------------
    result.check_true(
        "aligned_on_axis_spot_is_exactly_centred",
        "analytic",
        abs(aligned["centroid_x_mm"]) < 1e-12 and abs(aligned["centroid_y_mm"]) < 1e-12,
        f"centroid = ({aligned['centroid_x_mm']:.3e}, {aligned['centroid_y_mm']:.3e}) mm",
    )
    result.check_close(
        "aligned_spot_is_rotationally_symmetric",
        "analytic",
        aligned["std_y_mm"],
        aligned["std_x_mm"],
        rel=1e-9,
    )

    # -- tilt about x breaks the y symmetry only -----------------------------
    result.check_true(
        "tilt_about_x_leaves_the_x_centroid_on_axis",
        "analytic",
        abs(tilted["centroid_x_mm"]) < 1e-12,
        f"x centroid {tilted['centroid_x_mm']:.3e} mm; a rotation about x preserves "
        "the y-z plane of symmetry, so no x-direction first moment can appear",
    )
    result.check_true(
        "tilt_about_x_displaces_the_y_centroid",
        "analytic",
        abs(tilted["centroid_y_mm"]) > 1e-6,
        f"y centroid {tilted['centroid_y_mm']:.6f} mm vs aligned "
        f"{aligned['centroid_y_mm']:.3e} mm",
    )
    result.check_true(
        "tilt_makes_the_spot_astigmatic",
        "analytic",
        tilted["std_y_mm"] > tilted["std_x_mm"],
        f"std_y {tilted['std_y_mm']:.6f} > std_x {tilted['std_x_mm']:.6f} mm: a tilt "
        "about x adds y-direction aberration, not x",
    )

    # -- decenter in y -------------------------------------------------------
    result.check_true(
        "decenter_in_y_leaves_the_x_centroid_on_axis",
        "analytic",
        abs(decentered["centroid_x_mm"]) < 1e-12,
        f"x centroid {decentered['centroid_x_mm']:.3e} mm",
    )
    decenter_ratio = abs(decentered["centroid_y_mm"]) / 1.0
    result.record(decenter_centroid_shift_over_decenter=decenter_ratio)
    result.check_true(
        "decenter_in_y_displaces_the_y_centroid_by_order_the_decenter",
        "analytic",
        0.3 < decenter_ratio < 3.0,
        f"y centroid {decentered['centroid_y_mm']:.6f} mm for a 1.0 mm surface decenter "
        f"(ratio {decenter_ratio:.3f}). The shift is NOT bounded by the decenter: "
        "displacing a powered surface adds a prismatic deflection on top of the "
        "translation, so the image can move further than the surface did.",
    )

    # -- both perturbations degrade the spot ---------------------------------
    for name, stats in (("tilt", tilted), ("decenter", decentered)):
        result.check_true(
            f"{name}_increases_rms_spot_radius",
            "analytic",
            stats["rms_radius_about_centroid_mm"] > aligned["rms_radius_about_centroid_mm"],
            f"{name} RMS {stats['rms_radius_about_centroid_mm']:.6f} mm > aligned "
            f"{aligned['rms_radius_about_centroid_mm']:.6f} mm",
        )

    # -- SpotDiagram agrees with the raw rays --------------------------------
    lens = build(rx=np.radians(5.0))
    spot = analysis.SpotDiagram(lens)
    rms_from_analysis = np.asarray(spot.rms_spot_radius(), dtype=float).ravel()
    fig_ax = spot.view()
    plt.close("all")
    result.record(
        spot_diagram_rms_mm=rms_from_analysis,
        spot_diagram_view_returned=type(fig_ax).__name__,
    )
    result.check_finite("spot_diagram_rms_finite", rms_from_analysis)
    result.check_true(
        "spot_diagram_rms_is_the_same_order_as_the_direct_ray_rms",
        "invariant",
        bool(
            0.2
            < float(rms_from_analysis[0]) / tilted["rms_radius_about_centroid_mm"]
            < 5.0
        ),
        f"SpotDiagram RMS {float(rms_from_analysis[0]):.6f} mm vs direct hexapolar RMS "
        f"{tilted['rms_radius_about_centroid_mm']:.6f} mm (different default pupil "
        "sampling and reference, so agreement is in magnitude not to round-off)",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
