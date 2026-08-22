"""Intermediate / "Raytracing Aspheres" -- https://www.optiland.org/tutorials/raytracing-aspheres

Repo-owned reproduction of the asphere tutorial: the same N-SF11 singlet with a
spherical first surface and with an even-aspheric first surface, compared by
spot diagram.

Upstream's claim is qualitative ("aspheric surfaces reduce spherical
aberration"). This reproduction turns it into two hard results:

* **The sag convention is verified against a closed form written out here.**
  Optiland's ``EvenAsphere`` evaluates

      z(r) = r^2 / (R * (1 + sqrt(1 - (1+k) r^2 / R^2)))  +  sum_i C_i * r^(2(i+1))

  and the loop index starts at ``i = 0``, so **``coefficients[0]`` multiplies
  r^2, not r^4**. That is *not* the Zemax/CODE V even-asphere convention, where
  the polynomial starts at r^4 because the r^2 term is degenerate with the base
  curvature. Transcribing a vendor prescription term-for-term into Optiland
  therefore shifts every coefficient by one order. The reproduction evaluates
  both conventions and asserts they disagree, then asserts Optiland matches the
  r^2-first one to float64 round-off.
* **The aspheric correction is quantified**: on-axis RMS spot radius drops from
  0.5731 mm to 0.0271 mm, a 21x reduction. Sharper than "the spot is smaller":
  halving the pupil shrinks the spherical surface's RMS spot by 9.7x -- the
  ``h^3`` signature of third-order spherical aberration, steepened a little by
  higher orders at this f/1.3 aperture -- but only 2.3x for the asphere. The
  third-order term has been cancelled, leaving a weaker higher-order residual.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t12_raytracing_aspheres",
    title="Raytracing Aspheres",
    level="intermediate",
    url="https://www.optiland.org/tutorials/raytracing-aspheres",
    demonstrates=(
        "surface_type='even_asphere' with conic= and coefficients= kwargs, "
        "Optic.update_paraxial, and the fact that Optiland's even-asphere "
        "polynomial starts at r^2 rather than the Zemax-conventional r^4."
    ),
)

R1_MM = 20.0
THICKNESS_MM = 7.0
BACK_MM = 21.56201105
EPD_MM = 20.0
WAVELENGTH_UM = 0.587
CONIC = 0.0
COEFFICIENTS = (-2.248851e-4, -4.690412e-6, -6.404376e-8)


def _base(lens):
    lens.surfaces.add(index=2, thickness=BACK_MM)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    lens.update_paraxial()
    return lens


def build_spherical():
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1, thickness=THICKNESS_MM, radius=R1_MM, is_stop=True, material="N-SF11"
    )
    return _base(lens)


def build_asphere():
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        thickness=THICKNESS_MM,
        radius=R1_MM,
        is_stop=True,
        material="N-SF11",
        surface_type="even_asphere",
        conic=CONIC,
        coefficients=list(COEFFICIENTS),
    )
    return _base(lens)


def sag_r2_first(r: float) -> float:
    """Even asphere with the polynomial starting at r^2 (Optiland's convention)."""
    r2 = float(r) ** 2
    base = r2 / (R1_MM * (1.0 + np.sqrt(1.0 - (1.0 + CONIC) * r2 / R1_MM**2)))
    return float(base + sum(c * r2 ** (i + 1) for i, c in enumerate(COEFFICIENTS)))


def sag_r4_first(r: float) -> float:
    """Even asphere with the polynomial starting at r^4 (Zemax/CODE V convention)."""
    r2 = float(r) ** 2
    base = r2 / (R1_MM * (1.0 + np.sqrt(1.0 - (1.0 + CONIC) * r2 / R1_MM**2)))
    return float(base + sum(c * r2 ** (i + 2) for i, c in enumerate(COEFFICIENTS)))


def _on_axis_rms(lens, num_rays: int = 24, epd_mm: float | None = None) -> float:
    if epd_mm is not None:
        lens.set_aperture(aperture_type="EPD", value=epd_mm)
        lens.update_paraxial()
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=num_rays)
    x = np.asarray(rays.x, dtype=float)
    y = np.asarray(rays.y, dtype=float)
    return float(np.sqrt(np.mean(x**2 + y**2)))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis

    result = TutorialResult()
    spherical = build_spherical()
    asphere = build_asphere()

    geom = asphere.surfaces.surfaces[1].geometry
    result.record(
        asphere_geometry_class=type(geom).__name__,
        asphere_conic=float(np.asarray(geom.k).ravel()[0]),
        asphere_coefficients=[float(np.asarray(c).ravel()[0]) for c in geom.coefficients],
    )
    result.check_true(
        "even_asphere_surface_type_builds_an_EvenAsphere_geometry",
        "invariant",
        type(geom).__name__ == "EvenAsphere",
        f"geometry class {type(geom).__name__}",
    )

    # -- sag convention -------------------------------------------------------
    radii = (0.0, 1.0, 2.0, 5.0, EPD_MM / 2.0)
    observed = [float(np.asarray(geom.sag(r, 0.0)).ravel()[0]) for r in radii]
    r2_first = [sag_r2_first(r) for r in radii]
    r4_first = [sag_r4_first(r) for r in radii]
    result.record(
        sag_sample_radii_mm=list(radii),
        sag_observed_mm=observed,
        sag_r2_first_convention_mm=r2_first,
        sag_r4_first_convention_mm=r4_first,
        max_abs_error_vs_r2_first=float(np.max(np.abs(np.array(observed) - r2_first))),
        max_abs_error_vs_r4_first=float(np.max(np.abs(np.array(observed) - r4_first))),
    )
    result.check_true(
        "sag_matches_the_r2_first_even_asphere_closed_form",
        "analytic",
        float(np.max(np.abs(np.array(observed) - r2_first))) < 1e-15,
        f"max |sag_optiland - sag_closed_form| = "
        f"{float(np.max(np.abs(np.array(observed) - r2_first))):.3e} mm over r = {radii}",
    )
    result.check_true(
        "the_two_coefficient_conventions_are_distinguishable",
        "analytic",
        float(np.max(np.abs(np.array(observed) - r4_first))) > 1e-4,
        f"the Zemax r^4-first reading of the same coefficient list differs by up to "
        f"{float(np.max(np.abs(np.array(observed) - r4_first))):.3e} mm -- so "
        "coefficients[0] multiplying r^2 is a real, testable convention, not a "
        "notational preference",
    )

    # -- the aberration reduction --------------------------------------------
    rms_spherical = _on_axis_rms(build_spherical())
    rms_asphere = _on_axis_rms(build_asphere())
    result.record(
        rms_spot_spherical_mm=rms_spherical,
        rms_spot_asphere_mm=rms_asphere,
        aspheric_improvement_factor=rms_spherical / rms_asphere,
    )
    result.check_true(
        "asphere_reduces_on_axis_rms_spot_by_more_than_an_order_of_magnitude",
        "analytic",
        rms_asphere < rms_spherical / 10.0,
        f"RMS spot radius {rms_spherical:.6f} mm (spherical) -> {rms_asphere:.6f} mm "
        f"(even asphere): {rms_spherical / rms_asphere:.1f}x better",
    )

    # Third-order spherical aberration scales as h^3; the corrected surface must
    # break that scaling, which is a sharper statement than "the spot is smaller".
    ratio_spherical = _on_axis_rms(build_spherical(), epd_mm=EPD_MM) / _on_axis_rms(
        build_spherical(), epd_mm=EPD_MM / 2.0
    )
    ratio_asphere = _on_axis_rms(build_asphere(), epd_mm=EPD_MM) / _on_axis_rms(
        build_asphere(), epd_mm=EPD_MM / 2.0
    )
    result.record(rms_pupil_scaling_spherical=ratio_spherical, rms_pupil_scaling_asphere=ratio_asphere)
    result.check_true(
        "spherical_surface_shows_h_cubed_scaling",
        "analytic",
        6.0 < ratio_spherical < 12.0,
        f"halving the pupil shrinks the RMS spot {ratio_spherical:.2f}x, bracketing the "
        "h^3 = 8x signature of third-order spherical aberration (steepened by higher "
        "orders at this f/1.3 aperture)",
    )
    result.check_true(
        "asphere_breaks_the_h_cubed_scaling",
        "analytic",
        ratio_asphere < 0.6 * ratio_spherical,
        f"halving the pupil changes the RMS spot by {ratio_spherical:.2f}x for the "
        f"spherical surface (the h^3 signature) but only {ratio_asphere:.2f}x for the "
        "asphere: the third-order term has been cancelled, so the residual grows far "
        "more slowly with pupil height",
    )

    for label, lens in (("spherical", build_spherical()), ("asphere", build_asphere())):
        spot = analysis.SpotDiagram(lens)
        rms = np.asarray(spot.rms_spot_radius(), dtype=float).ravel()
        spot.view()
        plt.close("all")
        result.record(**{f"spot_diagram_rms_{label}_mm": rms})
        result.check_finite(f"spot_diagram_rms_{label}_finite", rms)
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
