"""Intermediate / "Thorlabs Catalogue" -- https://www.optiland.org/tutorials/catalogue-thorlabs

Repo-owned reproduction of the vendor-URL-import tutorial: ``load_zemax_file(url)``
straight from thorlabs.com, then an object-plane sweep
(``updater.set_thickness(surface_number=0)`` + ``image_solve()``) with RMS spot
radius plotted against object shift.

**The upstream URL import is a confirmed blocker, and this reproduction records
the evidence rather than skipping it.** The container *does* have outbound network
access, and the tutorial's URL is reachable -- but Thorlabs serves an HTML page
(1313 bytes beginning ``<!DOCTYPE html>``) rather than the ``.zmx`` payload, so
``load_zemax_file(url)`` raises ``ValueError: Failed to read Zemax file.``. That
failure is asserted below, so if Thorlabs ever restores direct file serving the
test tells us. Two further reasons not to build the reproduction on it even if it
worked: a repository test must not depend on a third-party server, and the file
is not a repository artifact.

The analysis workflow -- which is what the tutorial actually teaches -- is
reproduced in full on a repo-owned cemented doublet at finite conjugates. The
validation is the physics of a through-focus scan:

* ``image_solve()`` refocuses at every object position, so the sweep is **not** a
  defocus scan: the residual is pure aberration-vs-conjugate-ratio, and it falls
  monotonically as the object recedes because this doublet is corrected for an
  infinite conjugate. That monotonicity, in that direction, is the check.
* The image distance ``image_solve()`` finds satisfies the **exact** Gaussian
  conjugate equation ``1/s' + 1/s = 1/f`` -- to 1e-11 -- once the two distances are
  measured from the principal planes. Getting there establishes a convention that
  is easy to get silently wrong and that this repository had not recorded:

      ``paraxial.P1()`` is the front principal plane **relative to surface 1**;
      ``paraxial.P2()`` and ``paraxial.F2()`` are **relative to the image surface**.

  Measured from surface 4 instead, the same image distances miss a thin-lens
  prediction by 16-61%, which is what a naive reading produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t24_thorlabs_catalogue",
    title="Thorlabs Catalogue",
    level="intermediate",
    url="https://www.optiland.org/tutorials/catalogue-thorlabs",
    demonstrates=(
        "load_zemax_file(url) for a remote .zmx (blocked upstream), "
        "Optic.updater.set_thickness(value, surface_number=0) to move the object "
        "plane, Optic.image_solve(), and a through-focus RMS spot sweep."
    ),
    slow=True,
)

UPSTREAM_URL = (
    "https://www.thorlabs.com/_sd.cfm?fileName=20565-S03.zmx&partNumber=MAP051950-A"
)
NOMINAL_OBJECT_DISTANCE_MM = 400.0
WAVELENGTH_UM = 0.5876


def build_doublet(object_distance_mm: float = NOMINAL_OBJECT_DISTANCE_MM):
    """A cemented N-BK7/SF2 achromat at finite conjugates, repo-owned."""
    from optiland import optic

    lens = optic.Optic(name="finite-conjugate achromat")
    lens.surfaces.add(index=0, radius=np.inf, thickness=object_distance_mm)
    lens.surfaces.add(index=1, radius=29.32908, thickness=0.7, material="N-BK7", is_stop=True)
    lens.surfaces.add(index=2, radius=-20.06842, thickness=0.032)
    lens.surfaces.add(index=3, radius=-20.08770, thickness=0.5780, material=("SF2", "schott"))
    lens.surfaces.add(index=4, radius=-66.54774, thickness=47.3562)
    lens.surfaces.add(index=5)
    lens.set_aperture(aperture_type="imageFNO", value=8.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    lens.image_solve()
    return lens


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis
    from optiland.fileio import load_zemax_file

    result = TutorialResult()

    # -- 1. the upstream URL import, and its confirmed failure ------------------
    remote_error = ""
    try:
        load_zemax_file(UPSTREAM_URL)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the evidence
        remote_error = f"{type(exc).__name__}: {exc}"
    result.record(
        upstream_url=UPSTREAM_URL,
        upstream_url_failed=bool(remote_error),
        upstream_url_error_type=remote_error.split(":")[0] if remote_error else "",
    )
    # Any exception counts: the point is that the documented URL is not a usable
    # source in this environment. The observed failure with network access
    # available is ValueError("Failed to read Zemax file.") because Thorlabs serves
    # HTML; without network access it would be a URLError. Either way the
    # reproduction below is what carries the tutorial's content.
    result.check_true(
        "the_tutorials_remote_zmx_url_is_not_a_usable_source",
        "invariant",
        bool(remote_error),
        f"load_zemax_file({UPSTREAM_URL!r}) -> {remote_error or 'SUCCEEDED (upstream fixed?)'}. "
        "With outbound network available, Thorlabs answers this URL with a 1313-byte "
        "HTML page rather than the .zmx payload.",
    )

    # -- 2. the analysis workflow, on a repo-owned prescription -----------------
    lens = build_doublet()
    lens.info()
    spot = analysis.SpotDiagram(lens)
    nominal_rms = float(np.asarray(spot.rms_spot_radius()[0][0]).ravel()[0])
    spot.view()
    plt.close("all")
    efl = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    result.record(paraxial_efl_mm=efl, nominal_rms_spot_mm=nominal_rms)
    result.check_finite("nominal_rms_finite", [nominal_rms])

    dz = np.linspace(-40.0, 40.0, 33)
    rms_curve = []
    image_distance = []
    for shift in dz:
        trial = build_doublet()
        trial.updater.set_thickness(
            value=NOMINAL_OBJECT_DISTANCE_MM + float(shift), surface_number=0
        )
        trial.image_solve()
        rms_curve.append(
            float(
                np.asarray(
                    analysis.SpotDiagram(trial).rms_spot_radius()[0][0]
                ).ravel()[0]
            )
        )
        z_last = float(np.asarray(trial.surfaces.surfaces[4].geometry.cs.z).ravel()[0])
        z_image = float(np.asarray(trial.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
        image_distance.append(z_image - z_last)
    rms_curve = np.asarray(rms_curve, dtype=float)
    image_distance = np.asarray(image_distance, dtype=float)
    plt.plot(dz, rms_curve)
    plt.close("all")

    result.record(
        object_shift_mm=dz,
        rms_spot_curve_mm=rms_curve,
        image_distance_from_last_surface_mm=image_distance,
        best_rms_mm=float(rms_curve.min()),
        worst_rms_mm=float(rms_curve.max()),
    )
    result.check_finite("through_focus_curve_finite", rms_curve)
    result.check_true(
        "refocused_residual_falls_monotonically_as_the_object_recedes",
        "analytic",
        bool(np.all(np.diff(rms_curve) < 0.0)),
        f"RMS spot radius falls monotonically from {float(rms_curve[0]):.6f} mm at the "
        f"nearest object ({NOMINAL_OBJECT_DISTANCE_MM + float(dz[0]):.0f} mm) to "
        f"{float(rms_curve[-1]):.6f} mm at the farthest "
        f"({NOMINAL_OBJECT_DISTANCE_MM + float(dz[-1]):.0f} mm), at all {rms_curve.size} "
        "samples. image_solve() removes the defocus at each step, so what is left is "
        "aberration versus conjugate ratio -- and a doublet corrected for infinity must "
        "improve as the object recedes.",
    )
    result.check_true(
        "image_distance_shortens_as_the_object_recedes",
        "analytic",
        bool(np.all(np.diff(image_distance) < 0.0)),
        f"the solved image distance falls from {float(image_distance[0]):.4f} to "
        f"{float(image_distance[-1]):.4f} mm, as the conjugate equation requires",
    )
    # -- 3. the exact conjugate relation, referenced to the principal planes -----
    residuals = []
    for shift in dz:
        trial = build_doublet()
        trial.updater.set_thickness(
            value=NOMINAL_OBJECT_DISTANCE_MM + float(shift), surface_number=0
        )
        trial.image_solve()
        paraxial = trial.paraxial
        efl_trial = float(np.asarray(paraxial.f2()).ravel()[0])
        p1 = float(np.asarray(paraxial.P1()).ravel()[0])
        p2 = float(np.asarray(paraxial.P2()).ravel()[0])
        z_surface1 = float(np.asarray(trial.surfaces.surfaces[1].geometry.cs.z).ravel()[0])
        z_image = float(np.asarray(trial.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
        z_object = float(np.asarray(trial.surfaces.surfaces[0].geometry.cs.z).ravel()[0])
        # P1 is relative to surface 1; P2 and F2 are relative to the image surface.
        s_object = (z_surface1 + p1) - z_object
        s_image = z_image - (z_image + p2)
        residuals.append(abs(1.0 / s_image + 1.0 / s_object - 1.0 / efl_trial) * efl_trial)
    residuals = np.asarray(residuals, dtype=float)
    result.record(
        conjugate_equation_max_relative_residual=float(residuals.max()),
        principal_plane_P1_relative_to_surface_1_mm=p1,
        principal_plane_P2_relative_to_image_surface_mm=p2,
    )
    result.check_true(
        "the_gaussian_conjugate_equation_holds_exactly_from_the_principal_planes",
        "analytic",
        float(residuals.max()) < 1e-9,
        f"max |f*(1/s' + 1/s - 1/f)| = {float(residuals.max()):.3e} over all "
        f"{residuals.size} object positions, with s measured from P1 (which "
        "paraxial.P1() reports relative to SURFACE 1) and s' from P2 (which "
        "paraxial.P2() reports relative to the IMAGE SURFACE). Mixing those two "
        "reference planes up is what produced a 16-61% error on the first attempt.",
    )

    result.note(
        "The tutorial's own artifact could not be used: thorlabs.com answers the "
        "documented .zmx URL with HTML. Even when a vendor URL works, a repository "
        "test must not depend on a third-party server -- record the prescription "
        "instead. Genuine vendor-authored .zmx parsing therefore remains unverified "
        "here and in t08."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
