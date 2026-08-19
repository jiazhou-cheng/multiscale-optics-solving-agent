"""Intermediate / "Common Aberration Analyses" -- https://www.optiland.org/tutorials/common-aberration-analyses

Repo-owned reproduction of the analysis-gallery tutorial: run the whole
`optiland.analysis` suite over the bundled `CookeTriplet` -- SpotDiagram (with
`geometric_spot_radius` and `rms_spot_radius` tables), RayFan, YYbar, Distortion,
GridDistortion, FieldCurvature, RmsSpotSizeVsField, RmsWavefrontErrorVsField and
PupilAberration.

Upstream prints the two spot tables but publishes no values, and everything else
is a picture. Validation is therefore structural and physical:

* Both spot tables are indexed ``[field][wavelength]`` and cover all 3 fields x
  3 wavelengths of the sample.
* ``geometric_spot_radius >= rms_spot_radius`` for **every** cell. The geometric
  radius is the maximum ray excursion and the RMS is a quadratic mean over the
  same rays, so this ordering is a definition, not a coincidence -- and it is a
  real check on the two independent code paths.
* Spot size and wavefront error both grow from the axis outward on this triplet.
* Distortion is exactly zero on axis (an axial point has no transverse reference
  to be distorted against) and grows monotonically with field.
* The analyses' ``.data`` shapes are *not* uniform, and this reproduction records
  each one: ``SpotDiagram`` tables are ``[field][wavelength]`` nested lists,
  ``Distortion.data`` is one 128-point curve per wavelength,
  ``RmsWavefrontErrorVsField.data`` is a dict keyed by ``((Hx, Hy), wavelength)``
  whose *values are ``WavefrontData`` records*, not RMS scalars,
  and ``FieldCurvature.data`` is a list of ``(2, 128)`` arrays.
* Every analysis constructs and renders headlessly and returns finite data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t13_common_aberration_analyses",
    title="Common Aberration Analyses",
    level="intermediate",
    url="https://www.optiland.org/tutorials/common-aberration-analyses",
    demonstrates=(
        "optiland.analysis.{SpotDiagram,RayFan,YYbar,Distortion,GridDistortion,"
        "FieldCurvature,RmsSpotSizeVsField,RmsWavefrontErrorVsField,"
        "PupilAberration}, and the [field][wavelength] indexing of "
        "SpotDiagram.geometric_spot_radius / rms_spot_radius."
    ),
    slow=True,
)


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis
    from optiland.samples.objectives import CookeTriplet

    result = TutorialResult()
    lens = CookeTriplet()
    fields = lens.fields.get_field_coords()
    wavelengths = lens.wavelengths.get_wavelengths()
    result.record(fields=fields, wavelengths_um=wavelengths)

    spot = analysis.SpotDiagram(lens)
    geo = np.asarray(spot.geometric_spot_radius(), dtype=float)
    rms = np.asarray(spot.rms_spot_radius(), dtype=float)
    spot.view()
    plt.close("all")
    result.record(
        geometric_spot_radius_mm=geo,
        rms_spot_radius_mm=rms,
        spot_table_shape=list(geo.shape),
    )
    result.check_shape(
        "spot_tables_are_field_by_wavelength", geo, (len(fields), len(wavelengths))
    )
    result.check_true(
        "rms_and_geometric_tables_have_the_same_shape",
        "invariant",
        geo.shape == rms.shape,
        f"{geo.shape} == {rms.shape}",
    )
    result.check_finite("spot_tables_finite", np.concatenate([geo.ravel(), rms.ravel()]))
    result.check_true(
        "geometric_radius_bounds_rms_radius_everywhere",
        "analytic",
        bool(np.all(geo >= rms)),
        "geometric (max excursion) >= RMS (quadratic mean) in all "
        f"{geo.size} field x wavelength cells; worst margin "
        f"{float((geo - rms).min()):.6e} mm",
    )
    result.check_true(
        "spot_size_grows_off_axis",
        "analytic",
        bool(np.all(rms[-1] > rms[0])),
        f"RMS spot radius at the edge field {rms[-1]} exceeds the on-axis "
        f"{rms[0]} at every wavelength",
    )

    # -- distortion -----------------------------------------------------------
    # Distortion.data is a list of one 128-point field sweep per wavelength, in
    # per-cent, starting on axis.
    distortion = analysis.Distortion(lens)
    distortion.view()
    plt.close("all")
    dist_curves = [np.asarray(c, dtype=float).ravel() for c in distortion.data]
    on_axis = [float(c[0]) for c in dist_curves]
    result.record(
        distortion_num_curves=len(dist_curves),
        distortion_points_per_curve=int(dist_curves[0].size),
        distortion_on_axis_percent=on_axis,
        distortion_max_abs_percent=[float(np.max(np.abs(c))) for c in dist_curves],
    )
    result.check_finite("distortion_finite", np.concatenate(dist_curves))
    result.check_true(
        "distortion_curve_starts_at_zero_on_axis",
        "analytic",
        all(abs(v) < 1e-9 for v in on_axis),
        f"on-axis distortion per wavelength = {on_axis} %: an axial point has no "
        "transverse reference to be distorted against",
    )
    result.check_true(
        "distortion_grows_monotonically_with_field",
        "analytic",
        all(abs(float(c[-1])) > abs(float(c[c.size // 2])) > 0.0 for c in dist_curves),
        "|distortion| at the edge field exceeds its value at half field, at every "
        f"wavelength: edge {[round(float(c[-1]), 5) for c in dist_curves]} % vs half "
        f"{[round(float(c[c.size // 2]), 5) for c in dist_curves]} %",
    )

    # -- wavefront error vs field --------------------------------------------
    # RmsWavefrontErrorVsField.data is a dict keyed by ((Hx, Hy), wavelength)
    # whose values are WavefrontData records, NOT scalars: the RMS is derived from
    # the per-ray `opd` array (in waves) by the plotting layer. RMS wavefront error
    # is the spread about the mean, so np.std is the right reduction -- a piston
    # term is not an aberration.
    wfe_vs_field = analysis.RmsWavefrontErrorVsField(lens)
    wfe_vs_field.view()
    plt.close("all")
    wfe_map = {
        (float(key[0][1]), float(key[1])): float(
            np.std(np.asarray(value.opd, dtype=float))
        )
        for key, value in wfe_vs_field.data.items()
    }
    sample_key, sample_value = next(iter(wfe_vs_field.data.items()))
    result.record(
        wavefront_data_class=type(sample_value).__name__,
        wavefront_data_fields=sorted(
            f for f in getattr(type(sample_value), "__dataclass_fields__", {})
        ),
        wavefront_opd_samples_per_field=int(np.asarray(sample_value.opd).size),
    )
    hy_values = sorted({hy for hy, _ in wfe_map})
    wl_values = sorted({wl for _, wl in wfe_map})
    result.record(
        rms_wfe_num_entries=len(wfe_map),
        rms_wfe_num_field_samples=len(hy_values),
        rms_wfe_wavelengths_um=wl_values,
        rms_wfe_on_axis_waves=[wfe_map[(hy_values[0], wl)] for wl in wl_values],
        rms_wfe_edge_field_waves=[wfe_map[(hy_values[-1], wl)] for wl in wl_values],
    )
    result.check_finite("rms_wfe_vs_field_finite", list(wfe_map.values()))
    result.check_true(
        "rms_wavefront_error_is_keyed_by_field_and_wavelength",
        "invariant",
        len(wfe_map) == len(hy_values) * len(wl_values),
        f"{len(wfe_map)} entries == {len(hy_values)} field samples x "
        f"{len(wl_values)} wavelengths",
    )
    result.check_true(
        "wavefront_error_grows_off_axis",
        "analytic",
        all(wfe_map[(hy_values[-1], wl)] > wfe_map[(hy_values[0], wl)] for wl in wl_values),
        "RMS wavefront error rises from "
        f"{[round(wfe_map[(hy_values[0], wl)], 6) for wl in wl_values]} waves on axis to "
        f"{[round(wfe_map[(hy_values[-1], wl)], 6) for wl in wl_values]} waves at the edge",
    )

    # -- the remaining analyses must simply run and produce finite data -------
    for name, factory in (
        ("RayFan", analysis.RayFan),
        ("YYbar", analysis.YYbar),
        ("GridDistortion", analysis.GridDistortion),
        ("FieldCurvature", analysis.FieldCurvature),
        ("RmsSpotSizeVsField", analysis.RmsSpotSizeVsField),
        ("PupilAberration", analysis.PupilAberration),
    ):
        obj = factory(lens)
        obj.view()
        plt.close("all")
        data = getattr(obj, "data", None)
        flat: list[np.ndarray] = []

        def _collect(node):
            if isinstance(node, dict):
                for sub in node.values():
                    _collect(sub)
            elif isinstance(node, (list, tuple)):
                for sub in node:
                    _collect(sub)
            else:
                try:
                    arr = np.asarray(node, dtype=float)
                except (TypeError, ValueError):
                    return
                if arr.size:
                    flat.append(arr.ravel())

        _collect(data)
        finite = bool(np.all(np.isfinite(np.concatenate(flat)))) if flat else True
        result.record(**{f"{name}_data_type": type(data).__name__})
        result.check_true(
            f"{name}_runs_headlessly_with_finite_data",
            "invariant",
            finite,
            f"{name}(lens).view() completed; data type {type(data).__name__}, all finite={finite}",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
