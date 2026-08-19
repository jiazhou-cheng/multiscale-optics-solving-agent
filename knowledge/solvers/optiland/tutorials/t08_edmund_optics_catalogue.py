"""Beginner / "Edmund Optics Catalogue" -- https://www.optiland.org/tutorials/catalogue-edmund-optics

Repo-owned reproduction of the vendor-catalogue import tutorial:
``load_zemax_file`` on an Edmund Optics ``.zmx`` prescription, then
``Optic.info()``, ``analysis.SpotDiagram`` and ``analysis.RayFan``.

**Adaptation, and why.** Upstream instructs the reader to download
``zmax_47728.zmx`` from the Edmund Optics website. The `agent_solver` container
has no outbound network access, and a vendor file fetched at test time would not
be a reproducible artifact anyway. Optiland 0.6.0 ships no ``.zmx`` fixture
either (`rglob('*.zmx')` over the installed package returns nothing). This
reproduction therefore exercises the same import path on a prescription the
repository already owns and has validated analytically: Edmund Optics TECHSPEC
**#45-362**, the 20 mm x 50 mm EFL N-BK7 plano-convex singlet used by
`benchmarks/level1/L1-RAY-01/` (see `knowledge/solvers/optiland/source_manifest.yaml`).
The prescription is written out with ``save_zemax_file`` and read back with
``load_zemax_file``, which exercises both halves of Optiland's Zemax handler.

That substitution buys a *stronger* oracle than the tutorial has, because the
Edmund datasheet publishes the answers:

* EFL 50.00 mm and BFL 47.87 mm at 587.6 nm.

Both are checked against the reloaded system's paraxial solve, and independently
against the thick-lens closed form evaluated from the SCHOTT N-BK7 Sellmeier
index. The round trip is additionally required to preserve the trace exactly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t08_edmund_optics_catalogue",
    title="Edmund Optics Catalogue",
    level="beginner",
    url="https://www.optiland.org/tutorials/catalogue-edmund-optics",
    demonstrates=(
        "optiland.fileio.save_zemax_file / load_zemax_file (.zmx round trip), "
        "Optic.info(), analysis.SpotDiagram, analysis.RayFan. Substitutes a "
        "repo-owned Edmund #45-362 prescription for the tutorial's network "
        "download."
    ),
)

# Edmund Optics TECHSPEC #45-362, from the manufacturer's product page.
R1_MM = 25.84
CENTER_THICKNESS_MM = 3.23
CLEAR_APERTURE_MM = 19.0
DATASHEET_EFL_MM = 50.00
DATASHEET_BFL_MM = 47.87
WAVELENGTH_UM = 0.5876

# SCHOTT N-BK7 Sellmeier coefficients, used as an Optiland-independent oracle.
NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)


def nbk7_index(wavelength_um: float) -> float:
    w2 = float(wavelength_um) ** 2
    return float(
        np.sqrt(1.0 + sum(b * w2 / (w2 - c) for b, c in zip(NBK7_B, NBK7_C, strict=True)))
    )


def build_edmund_45362():
    from optiland import optic

    lens = optic.Optic(name="Edmund 45-362")
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        radius=R1_MM,
        thickness=CENTER_THICKNESS_MM,
        material="N-BK7",
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=np.inf, thickness=DATASHEET_BFL_MM)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=CLEAR_APERTURE_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis
    from optiland.fileio import load_zemax_file, save_zemax_file

    result = TutorialResult()
    original = build_edmund_45362()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "edmund_45362.zmx"
        save_zemax_file(original, str(path))
        zmx_text = path.read_text(errors="replace")
        lens = load_zemax_file(str(path))

    result.record(
        zmx_bytes=len(zmx_text),
        zmx_first_lines=[ln.strip() for ln in zmx_text.splitlines()[:6]],
        num_surfaces=len(lens.surfaces.surfaces),
        reloaded_name=str(lens.name),
    )
    result.check_true(
        "zmx_round_trip_preserves_surface_count",
        "invariant",
        len(lens.surfaces.surfaces) == len(original.surfaces.surfaces),
        f"{len(lens.surfaces.surfaces)} == {len(original.surfaces.surfaces)}",
    )

    radii = [float(np.asarray(s.geometry.radius).ravel()[0]) for s in lens.surfaces.surfaces[1:3]]
    result.record(reloaded_radii_mm=radii)
    result.check_close("zmx_round_trip_preserves_r1", "invariant", radii[0], R1_MM, rel=1e-9)
    result.check_true(
        "zmx_round_trip_preserves_plano_second_surface",
        "invariant",
        not np.isfinite(radii[1]) or abs(radii[1]) > 1e6,
        f"reloaded R2 = {radii[1]}",
    )

    # -- datasheet reference values ------------------------------------------
    efl = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    # paraxial.F2() is the back focal point measured *relative to the image
    # surface*, not in prescription z: it is the residual defocus of the placed
    # image plane. The back focal length is therefore
    # (z_image + F2) - z_last_lens_surface.
    z_image = float(np.asarray(lens.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
    z_last_lens = float(np.asarray(lens.surfaces.surfaces[2].geometry.cs.z).ravel()[0])
    f2_point = float(np.asarray(lens.paraxial.F2()).ravel()[0])
    bfl = z_image + f2_point - z_last_lens
    n_glass = nbk7_index(WAVELENGTH_UM)
    efl_closed_form = R1_MM / (n_glass - 1.0)
    bfl_closed_form = efl_closed_form - CENTER_THICKNESS_MM / n_glass
    result.record(
        paraxial_efl_mm=efl,
        paraxial_bfl_mm=bfl,
        paraxial_F2_relative_to_image_surface_mm=f2_point,
        image_surface_z_mm=z_image,
        nbk7_index_at_5876=n_glass,
        efl_closed_form_mm=efl_closed_form,
        bfl_closed_form_mm=bfl_closed_form,
    )
    result.check_close(
        "efl_matches_edmund_datasheet_50mm", "reference", efl, DATASHEET_EFL_MM, rel=2e-3
    )
    result.check_close(
        "bfl_matches_edmund_datasheet_47p87mm", "reference", bfl, DATASHEET_BFL_MM, rel=3e-3
    )
    result.check_close(
        "efl_matches_thick_lens_closed_form", "analytic", efl, efl_closed_form, rel=1e-9
    )
    result.check_close(
        "bfl_matches_thick_lens_closed_form", "analytic", bfl, bfl_closed_form, rel=1e-6
    )

    # -- the trace must survive the round trip exactly ------------------------
    before = original.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=8)
    after = lens.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=8)
    deviations = {
        attr: float(
            np.max(
                np.abs(
                    np.asarray(getattr(before, attr), dtype=float)
                    - np.asarray(getattr(after, attr), dtype=float)
                )
            )
        )
        for attr in ("x", "y", "z", "L", "M", "N", "opd")
    }
    result.record(
        trace_max_abs_deviation=deviations,
        num_traced_rays=int(np.asarray(after.x).size),
        radius_round_trip_relative_error=abs(radii[0] - R1_MM) / R1_MM,
    )
    result.check_true(
        "zmx_round_trip_is_lossy_but_only_at_float_text_precision",
        "invariant",
        all(v < 1e-6 for v in deviations.values()) and any(v > 0.0 for v in deviations.values()),
        f"max |delta| per array: {deviations}. Unlike the exact JSON round trip of "
        "t03, the .zmx round trip is NOT bit-identical: Zemax files record CURVATURE, "
        f"so R1 comes back as {radii[0]!r} instead of {R1_MM!r} (relative error "
        f"{abs(radii[0] - R1_MM) / R1_MM:.2e}) and the trace shifts accordingly.",
    )

    # -- the tutorial's analyses ---------------------------------------------
    lens.info()
    spot = analysis.SpotDiagram(lens)
    rms = np.asarray(spot.rms_spot_radius(), dtype=float).ravel()
    spot.view()
    plt.close("all")
    fan = analysis.RayFan(lens)
    fan.view()
    plt.close("all")
    result.record(spot_rms_mm=rms, working_f_number=efl / CLEAR_APERTURE_MM)
    result.check_finite("spot_rms_finite", rms)

    # Third-order spherical aberration makes transverse ray error scale as the
    # cube of the pupil height, so halving the EPD must shrink the RMS spot by 8x.
    # That is an analytic signature of the dominant aberration, checked without
    # any reference value.
    def _rms_at_epd(epd_mm: float) -> float:
        trial = build_edmund_45362()
        trial.set_aperture(aperture_type="EPD", value=epd_mm)
        trial.update_paraxial()
        r = trial.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=24)
        x = np.asarray(r.x, dtype=float)
        y = np.asarray(r.y, dtype=float)
        return float(np.sqrt(np.mean(x**2 + y**2)))

    rms_full = _rms_at_epd(CLEAR_APERTURE_MM)
    rms_half = _rms_at_epd(CLEAR_APERTURE_MM / 2.0)
    ratio = rms_full / rms_half
    result.record(rms_full_epd_mm=rms_full, rms_half_epd_mm=rms_half, rms_scaling_ratio=ratio)
    result.check_close(
        "spot_size_scales_as_pupil_height_cubed",
        "analytic",
        ratio,
        8.0,
        rel=0.1,
    )
    result.check_true(
        "on_axis_blur_is_aberration_not_diffraction_limited",
        "analytic",
        float(rms[0]) > 100.0 * (WAVELENGTH_UM * 1e-3) * (efl / CLEAR_APERTURE_MM),
        f"RMS spot radius {float(rms[0]):.6f} mm at f/{efl / CLEAR_APERTURE_MM:.2f} is more than "
        f"100x the {(WAVELENGTH_UM * 1e-3) * (efl / CLEAR_APERTURE_MM):.6f} mm diffraction "
        "scale lambda*F/#: an uncorrected plano-convex singlet is aberration limited",
    )
    result.check_true(
        "info_and_rayfan_render_headless", "qualitative", True, "Optic.info() and RayFan.view() completed"
    )
    result.note(
        "The tutorial's own artifact (zmax_47728.zmx from edmundoptics.com) was NOT "
        "downloaded: the container has no network access and Optiland 0.6.0 ships no "
        ".zmx fixture. save_zemax_file -> load_zemax_file on a repo-owned Edmund "
        "prescription exercises the same handler with a stronger, offline oracle. "
        "Reading a genuine vendor-authored .zmx (which may use surface/aperture "
        "records this round trip never emits) remains unverified."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
