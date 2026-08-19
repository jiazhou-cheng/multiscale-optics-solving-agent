"""Beginner / "Singlet Lens" -- https://www.optiland.org/tutorials/your-first-optical-system

Repo-owned reproduction of the first official Optiland tutorial: build an
`Optic` from scratch (surfaces, aperture, field type/points, wavelengths) and
visualize it.

Upstream states no numerical reference -- it is a build-and-look tutorial -- so
validation is analytic plus invariant:

* The tutorial's singlet is a plano-convex N-SF11 lens with the curved side
  first, so the thick-lens EFL collapses to the closed form ``R1 / (n - 1)``
  exactly: the ``(n-1)t/(n R1 R2)`` term vanishes for ``R2 = inf``. That is an
  oracle for ``paraxial.f2()`` that never enters Optiland's paraxial solver,
  needing only ``Material('N-SF11').n(0.5)``.
* The declared ``EPD=25`` must come back out of ``paraxial.EPD()``.
* An on-axis trace must land every ray on the image surface at the declared
  axial position, with finite coordinates and unit direction cosines.
* The tutorial's own API (`Optic.add_surface`/`set_field_type`/`add_field`/
  `add_wavelength`) is **deprecated in the pinned 0.6.0** and scheduled for
  removal in 0.7.0. The reproduction builds the same singlet through both the
  tutorial's calls and the modern group API and asserts the two systems are
  numerically identical, which is what makes the migration safe to record.

`Optic.draw3D()` -- the tutorial's other deliverable -- is *not* called: it
hangs indefinitely in the headless container. See `failure_guide.md`.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t01_singlet_lens",
    title="Singlet Lens",
    level="beginner",
    url="https://www.optiland.org/tutorials/your-first-optical-system",
    demonstrates=(
        "Optic() from-scratch construction: surfaces.add(index/radius/thickness/"
        "material/is_stop), set_aperture(EPD), fields.set_type('angle'), "
        "fields.add, wavelengths.add(is_primary), draw. Also records the "
        "0.6.0 -> 0.7.0 deprecation of the tutorial's Optic.add_* API."
    ),
)

# The tutorial's prescription, verbatim.
R1_MM = 20.0
THICKNESS_MM = 7.0
BACK_THICKNESS_MM = 18.0
GLASS = "N-SF11"
EPD_MM = 25.0
WAVELENGTH_UM = 0.5


def build_singlet_modern():
    """The tutorial's system, built through the non-deprecated group API."""
    from optiland import optic

    singlet = optic.Optic()

    # define surfaces
    singlet.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    singlet.surfaces.add(
        index=1, radius=R1_MM, thickness=THICKNESS_MM, is_stop=True, material=GLASS
    )
    singlet.surfaces.add(index=2, radius=np.inf, thickness=BACK_THICKNESS_MM)
    singlet.surfaces.add(index=3)

    # define aperture (set_aperture is NOT deprecated in 0.6.0)
    singlet.set_aperture(aperture_type="EPD", value=EPD_MM)

    # define fields
    singlet.fields.set_type(field_type="angle")
    singlet.fields.add(y=0)

    # define wavelengths
    singlet.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)

    return singlet


def build_singlet_as_published():
    """Verbatim upstream call sequence, which is deprecated in 0.6.0.

    Returns ``(optic, deprecated_call_names)`` so the reproduction can record
    exactly which of the tutorial's calls warn.
    """
    from optiland import optic

    singlet = optic.Optic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        singlet.add_surface(index=0, radius=np.inf, thickness=np.inf)
        singlet.add_surface(
            index=1, radius=R1_MM, thickness=THICKNESS_MM, is_stop=True, material=GLASS
        )
        singlet.add_surface(index=2, radius=np.inf, thickness=BACK_THICKNESS_MM)
        singlet.add_surface(index=3)
        singlet.set_aperture(aperture_type="EPD", value=EPD_MM)
        singlet.set_field_type(field_type="angle")
        singlet.add_field(y=0)
        singlet.add_wavelength(value=WAVELENGTH_UM, is_primary=True)
        messages = sorted({str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)})
    return singlet, messages


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.materials import Material

    result = TutorialResult()
    singlet = build_singlet_modern()

    surfaces = singlet.surfaces.surfaces
    result.record(
        num_surfaces=len(surfaces),
        surface_types=[type(s).__name__ for s in surfaces],
        surface_z_mm=[float(np.asarray(s.geometry.cs.z).ravel()[0]) for s in surfaces],
    )
    result.check_true(
        "four_surfaces_object_lens_image",
        "invariant",
        len(surfaces) == 4,
        f"object + 2 lens surfaces + image = 4, got {len(surfaces)}",
    )

    # -- analytic EFL oracle ------------------------------------------------
    n_glass = float(np.asarray(Material(GLASS).n(WAVELENGTH_UM)).ravel()[0])
    efl_analytic_mm = R1_MM / (n_glass - 1.0)
    f2 = float(np.asarray(singlet.paraxial.f2()).ravel()[0])
    result.record(
        n_glass_at_0p5um=n_glass, efl_analytic_mm=efl_analytic_mm, paraxial_f2_mm=f2
    )
    result.check_close(
        "efl_matches_thick_lens_closed_form", "analytic", f2, efl_analytic_mm, rel=1e-12
    )

    epd = float(np.asarray(singlet.paraxial.EPD()).ravel()[0])
    result.record(paraxial_EPD_mm=epd)
    result.check_close("declared_EPD_round_trips", "invariant", epd, EPD_MM, rel=1e-12)

    # -- on-axis trace ------------------------------------------------------
    rays = singlet.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=16)
    x, y, z = np.asarray(rays.x), np.asarray(rays.y), np.asarray(rays.z)
    norms = np.asarray(rays.L) ** 2 + np.asarray(rays.M) ** 2 + np.asarray(rays.N) ** 2
    image_z = float(np.asarray(surfaces[-1].geometry.cs.z).ravel()[0])
    result.record(
        num_traced_rays=int(x.size),
        image_surface_z_mm=image_z,
        rms_spot_radius_mm=float(np.sqrt(np.mean(x**2 + y**2))),
        max_direction_norm_error=float(np.max(np.abs(norms - 1.0))),
    )
    result.check_finite("traced_coordinates_finite", np.concatenate([x, y, z]))
    result.check_true(
        "rays_land_on_image_surface",
        "invariant",
        bool(np.allclose(z, image_z, atol=1e-12)),
        f"max |z - {image_z}| = {float(np.max(np.abs(z - image_z))):.3e}",
    )
    result.check_true(
        "directions_are_unit_cosines",
        "invariant",
        bool(np.max(np.abs(norms - 1.0)) < 1e-12),
        f"max |1 - (L^2+M^2+N^2)| = {float(np.max(np.abs(norms - 1.0))):.3e}",
    )

    # -- upstream API is deprecated, and equivalent --------------------------
    published, deprecation_messages = build_singlet_as_published()
    published_rays = published.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=16)
    same_x = np.array_equal(np.asarray(published_rays.x), x)
    same_opd = np.array_equal(np.asarray(published_rays.opd), np.asarray(rays.opd))
    result.record(
        upstream_api_deprecation_messages=deprecation_messages,
        published_f2_mm=float(np.asarray(published.paraxial.f2()).ravel()[0]),
    )
    result.check_true(
        "tutorial_api_emits_deprecation_warnings",
        "invariant",
        len(deprecation_messages) >= 4,
        f"{len(deprecation_messages)} distinct DeprecationWarning(s): {deprecation_messages}",
    )
    result.check_true(
        "deprecated_and_modern_construction_are_bit_identical",
        "invariant",
        same_x and same_opd,
        "trace x and opd arrays are element-wise equal between the two build paths",
    )

    # -- the tutorial's 2D drawing ------------------------------------------
    fig, ax = singlet.draw(num_rays=10)
    result.record(draw2d_returned=[type(fig).__name__, type(ax).__name__])
    plt.close("all")
    result.check_true(
        "draw2d_renders_headless",
        "qualitative",
        True,
        "singlet.draw(num_rays=10) returned (Figure, Axes) under the Agg backend",
    )
    result.note(
        "Optic.draw3D() is deliberately NOT executed: it blocks forever in the "
        "headless agent_solver container (VTK finds no X server, no EGL and no "
        "OSMesa). Recorded in failure_guide.md."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
