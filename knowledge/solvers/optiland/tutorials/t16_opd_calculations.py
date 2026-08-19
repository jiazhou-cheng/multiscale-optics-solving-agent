"""Intermediate / "OPD Calculations" -- https://www.optiland.org/tutorials/opd-calculations

Repo-owned reproduction of the wavefront tutorial: `wavefront.OPDFan` plus
`wavefront.OPD` maps at three field points on the bundled `EyepieceErfle`, viewed
in 2D and 3D.

Upstream is entirely visual. This reproduction reads the numbers behind the
pictures and checks the properties an OPD map must have:

* Optiland's ``OPD`` is **referenced to the chief ray and expressed in waves**,
  not millimetres: the pupil-centre value is zero to round-off and the map's
  peak-to-valley is a few waves for this eyepiece. That is the opposite reference
  convention from the raw ``RealRays.opd`` accumulator characterized in
  `conventions.md`, whose on-axis value for this system is thousands of waves --
  the two are compared here side by side so the distinction is recorded as
  evidence rather than prose.
* The on-axis map has no odd-in-y content -- the normalised first moment
  ``|<W*y>| / (<|W|> max|y|)`` is at round-off -- while at ``Hy = 1.0`` it is a
  substantial fraction, which is coma. (An ``x``-based azimuthal test cannot see
  this: OPD stays even in ``x`` for a ``y`` field, so the hexapolar grid's +/-x
  pairs agree to round-off at every field.)
* Aberration grows with field: peak-to-valley OPD at Hy = 1.0 exceeds Hy = 0.7
  which exceeds on axis.
* ``OPD.rms()`` is ``sqrt(mean(opd^2))`` over rays with ``intensity > 0`` and
  **leaves piston in**, so it is not the conventional piston-removed RMS wavefront
  error -- on axis it reads 0.1337 waves where the piston-removed value is 0.0664.
  ``WavefrontData.pupil_x/pupil_y`` are physical millimetres, not normalised.
  (Superseded description: it does not equal the intensity-weighted RMS about the mean recomputed here
  from ``OPD.get_data(field, wl).opd``.) ``OPD.data`` is a *dict* keyed by
  ``((Hx, Hy), wavelength)``, not a record -- the same shape as
  ``RmsWavefrontErrorVsField.data`` in t13.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t16_opd_calculations",
    title="OPD Calculations",
    level="intermediate",
    url="https://www.optiland.org/tutorials/opd-calculations",
    demonstrates=(
        "optiland.wavefront.OPDFan and wavefront.OPD(optic, field=(Hx,Hy), "
        "wavelength=...) with .view(projection='2d'|'3d', num_points=...), "
        ".rms(), and .data.opd -- chief-ray-referenced, in waves."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.5876
FIELDS = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import wavefront
    from optiland.samples.eyepieces import EyepieceErfle

    result = TutorialResult()
    lens = EyepieceErfle()
    result.record(num_surfaces=len(lens.surfaces.surfaces))

    opd_fan = wavefront.OPDFan(lens)
    opd_fan.view()
    plt.close("all")
    result.check_true(
        "opd_fan_renders_headless", "qualitative", True, "wavefront.OPDFan(lens).view() completed"
    )

    stats = {}
    for hx, hy in FIELDS:
        opd = wavefront.OPD(lens, field=(hx, hy), wavelength=WAVELENGTH_UM)
        # OPD.data is a dict keyed by ((Hx, Hy), wavelength); get_data() is the
        # supported accessor for the single WavefrontData record.
        data = opd.get_data((hx, hy), WAVELENGTH_UM)
        values = np.asarray(data.opd, dtype=float).ravel()
        intensity = np.asarray(data.intensity, dtype=float).ravel()
        px = np.asarray(data.pupil_x, dtype=float).ravel()
        py = np.asarray(data.pupil_y, dtype=float).ravel()
        radius = np.hypot(px, py)
        label = f"Hy_{hy:g}".replace(".", "p")
        stats[label] = {
            "num_pupil_points": int(values.size),
            "min_waves": float(values.min()),
            "max_waves": float(values.max()),
            "ptv_waves": float(values.max() - values.min()),
            "rms_accessor_waves": float(np.asarray(opd.rms()).ravel()[0]),
            # Optiland's rms() is sqrt(mean(opd^2)) over rays with intensity > 0.
            # It does NOT remove piston.
            "rms_recomputed_waves": float(np.sqrt(np.mean(values[intensity > 0] ** 2))),
            "rms_piston_removed_waves": float(np.std(values[intensity > 0])),
            "mean_waves": float(np.mean(values[intensity > 0])),
            "pupil_coordinate_max_abs_mm": float(np.max(np.abs(np.concatenate([px, py])))),
            "value_at_pupil_centre_waves": float(values[int(np.argmin(radius))]),
            "max_pupil_radius": float(radius.max()),
        }
        result.check_finite(f"opd_{label}_finite", values)
        opd.view(projection="3d" if hy == 1.0 else "2d", num_points=64)
        plt.close("all")
    result.record(opd_by_field=stats)

    on_axis = stats["Hy_0"]
    result.check_true(
        "opd_is_referenced_to_the_chief_ray_and_zero_at_the_pupil_centre",
        "analytic",
        abs(on_axis["value_at_pupil_centre_waves"]) < 1e-9,
        f"OPD at the pupil centre = {on_axis['value_at_pupil_centre_waves']:.3e} waves; "
        "a chief-ray-referenced wavefront has no piston by construction",
    )
    result.check_true(
        "opd_is_in_waves_not_millimetres",
        "analytic",
        0.01 < on_axis["ptv_waves"] < 1000.0,
        f"on-axis peak-to-valley {on_axis['ptv_waves']:.4f}: order-unity, so the unit is "
        "waves. The raw RealRays.opd accumulator for the same system is a "
        "millimetre-scale absolute path (see conventions.md), a different quantity "
        "with a different reference.",
    )
    for label, entry in stats.items():
        result.check_close(
            f"opd_rms_accessor_is_unweighted_sqrt_mean_square_{label}",
            "analytic",
            entry["rms_accessor_waves"],
            entry["rms_recomputed_waves"],
            rel=1e-9,
        )
    result.check_true(
        "opd_rms_includes_piston_and_is_not_the_conventional_rms_wavefront_error",
        "analytic",
        all(
            entry["rms_accessor_waves"] > entry["rms_piston_removed_waves"] * 1.02
            for entry in stats.values()
        ),
        "rms() vs piston-removed std per field: "
        + ", ".join(
            f"{label} {entry['rms_accessor_waves']:.6f} vs "
            f"{entry['rms_piston_removed_waves']:.6f}"
            for label, entry in stats.items()
        )
        + ". OPD.rms() is sqrt(mean(opd^2)) with the mean left in, so it is NOT the "
        "conventional piston-removed RMS wavefront error; a Marechal or Strehl "
        "estimate built on it would be wrong.",
    )
    result.check_true(
        "pupil_coordinates_are_in_millimetres_not_normalised",
        "invariant",
        stats["Hy_0"]["pupil_coordinate_max_abs_mm"] > 1.5,
        f"max |pupil coordinate| = {stats['Hy_0']['pupil_coordinate_max_abs_mm']:.4f}: "
        "WavefrontData.pupil_x/pupil_y are physical millimetres on the reference "
        "sphere, not normalised Px/Py",
    )

    # -- the absolute accumulator is a different quantity ----------------------
    raw = np.asarray(
        lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=6).opd, dtype=float
    )
    raw_waves = raw / (WAVELENGTH_UM * 1e-3)
    result.record(
        raw_realrays_opd_min_mm=float(raw.min()),
        raw_realrays_opd_max_mm=float(raw.max()),
        raw_realrays_opd_mean_waves=float(raw_waves.mean()),
    )
    result.check_true(
        "raw_realrays_opd_is_thousands_of_waves_where_the_wavefront_is_units",
        "analytic",
        float(raw_waves.mean()) > 1000.0 * on_axis["ptv_waves"],
        f"mean RealRays.opd = {float(raw_waves.mean()):.1f} waves absolute against a "
        f"{on_axis['ptv_waves']:.4f}-wave wavefront peak-to-valley: mixing the two would "
        "conjugate or piston-shift a coupler's input",
    )

    # -- rotational symmetry on axis, broken off axis --------------------------
    opd_axis = wavefront.OPD(lens, field=(0.0, 0.0), wavelength=WAVELENGTH_UM)
    opd_edge = wavefront.OPD(lens, field=(0.0, 1.0), wavelength=WAVELENGTH_UM)

    def _odd_in_y_fraction(opd_obj, field) -> float:
        """Normalised first moment of the OPD against pupil y: |<W*y>| / (<|W|>*max|y|).

        Zero for a wavefront even in y (rotational symmetry about the axis),
        nonzero once coma -- which is odd in the pupil coordinate -- appears. An
        x-based azimuthal test cannot see this, because OPD stays even in x for a
        y-field and the hexapolar grid's +/-x pairs agree to round-off at every field.
        """
        d = opd_obj.get_data(field, WAVELENGTH_UM)
        values = np.asarray(d.opd, dtype=float).ravel()
        py = np.asarray(d.pupil_y, dtype=float).ravel()
        scale = float(np.mean(np.abs(values))) * float(np.max(np.abs(py)))
        if scale == 0.0:
            return 0.0
        return float(abs(np.mean(values * py)) / scale)

    odd_axis = _odd_in_y_fraction(opd_axis, (0.0, 0.0))
    odd_edge = _odd_in_y_fraction(opd_edge, (0.0, 1.0))
    result.record(
        odd_in_y_fraction_on_axis=odd_axis,
        odd_in_y_fraction_edge_field=odd_edge,
    )
    result.check_true(
        "on_axis_wavefront_is_even_in_the_pupil_coordinate",
        "analytic",
        odd_axis < 1e-6,
        f"normalised |<W*y>| on axis = {odd_axis:.3e}: the on-axis wavefront of a "
        "rotationally symmetric system has no odd (coma-like) content",
    )
    result.check_true(
        "edge_field_wavefront_acquires_odd_coma_like_content",
        "analytic",
        odd_edge > 1e3 * max(odd_axis, 1e-12),
        f"normalised |<W*y>| = {odd_edge:.6f} at Hy=1.0 against {odd_axis:.3e} on axis: "
        "coma is odd in the pupil coordinate, and it is what the tutorial's 3D plot "
        "at the edge field is showing",
    )
    result.check_true(
        "aberration_grows_monotonically_with_field",
        "analytic",
        stats["Hy_1"]["ptv_waves"] > stats["Hy_0p7"]["ptv_waves"] > stats["Hy_0"]["ptv_waves"],
        "peak-to-valley OPD "
        f"{stats['Hy_0']['ptv_waves']:.4f} -> {stats['Hy_0p7']['ptv_waves']:.4f} -> "
        f"{stats['Hy_1']['ptv_waves']:.4f} waves for Hy = 0, 0.7, 1.0",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
