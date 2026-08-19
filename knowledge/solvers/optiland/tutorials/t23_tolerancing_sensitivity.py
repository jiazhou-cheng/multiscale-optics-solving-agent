"""Intermediate / "Tolerancing, Sensitivity Analyses" -- https://www.optiland.org/tutorials/tolerancing-sensitivity

Repo-owned reproduction of the system-tolerancing tutorial: a `Tolerancing`
problem over the bundled `CookeTriplet` with three perturbations on surface 1
(radius, x-tilt, thickness), three operands (``f2``, ``rms_spot_size``,
``OPD_difference``), and a `SensitivityAnalysis` sweep whose results come back as
a pandas DataFrame.

**Adaptation.** Upstream uses ``steps=128`` per perturbation (384 rows, each a
full trace). This reproduction uses 33, which resolves every monotonicity and
symmetry claim below; the count is recorded.

Upstream publishes no numbers -- it prints ``df.head()`` and ``df.describe()``.
Validation is the physics of each perturbation, which is fully determined:

* **Radius sweep 15 -> 30 mm on the first element.** Focal length must fall
  monotonically as that surface is made more strongly curved. Verified across all
  33 samples, and the observed ``f2`` bracket is compared against the thin-lens
  prediction ``1/f ~ (n-1)/R``.
* **Tilt sweep -0.05 -> +0.05 rad about x.** ``f2`` is even in the tilt to
  round-off (a rotation cannot change first-order power at first order in the
  angle) while the RMS spot size is even *and* minimized at zero tilt.
* **Thickness sweep 2 -> 5 mm.** ``f2`` varies only weakly (a thick-lens
  second-order term), far less than the radius sweep moves it.
* ``perturbation_type`` in the results frame is a display label ("Radius of
  Curvature, Surface 1"), not the ``add_perturbation`` keyword, and operand columns
  are prefixed by their declaration order ("0: f2").
* Every operand column is finite over all 99 samples, and the nominal system is
  restored after the analysis: a fresh ``CookeTriplet`` and the perturbed one agree
  on ``f2`` afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t23_tolerancing_sensitivity",
    title="Tolerancing, Sensitivity Analyses",
    level="intermediate",
    url="https://www.optiland.org/tutorials/tolerancing-sensitivity",
    demonstrates=(
        "optiland.tolerancing.{Tolerancing,RangeSampler,SensitivityAnalysis}: "
        "add_perturbation('radius'|'tilt'|'thickness', sampler, surface_number, "
        "axis=), add_operand('f2'|'rms_spot_size'|'OPD_difference', input_data, "
        "target=), and get_results() -> pandas.DataFrame."
    ),
    slow=True,
)

STEPS = 33  # upstream: 128


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.samples.objectives import CookeTriplet
    from optiland.tolerancing import RangeSampler, SensitivityAnalysis, Tolerancing

    result = TutorialResult()
    optic = CookeTriplet()
    nominal_f2 = float(np.asarray(optic.paraxial.f2()).ravel()[0])

    tolerancing = Tolerancing(optic)
    tolerancing.add_perturbation("radius", RangeSampler(start=15, end=30, steps=STEPS), surface_number=1)
    tolerancing.add_perturbation(
        "tilt", RangeSampler(start=-0.05, end=0.05, steps=STEPS), surface_number=1, axis="x"
    )
    tolerancing.add_perturbation("thickness", RangeSampler(start=2, end=5, steps=STEPS), surface_number=1)
    tolerancing.add_operand("f2", {"optic": optic})
    tolerancing.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0,
            "Hy": 0.0,
            "wavelength": 0.55,
            "num_rays": 5,
        },
        target=0,
    )
    tolerancing.add_operand(
        "OPD_difference",
        {"optic": optic, "Hx": 0, "Hy": 1, "wavelength": 0.55, "num_rays": 5},
    )

    sensitivity = SensitivityAnalysis(tolerancing)
    sensitivity.run()
    sensitivity.view()
    plt.close("all")
    df = sensitivity.get_results()
    columns = list(df.columns)
    numeric = [c for c in columns if df[c].dtype.kind in "fc"]
    result.record(
        steps_per_perturbation=STEPS,
        upstream_steps_per_perturbation=128,
        num_rows=int(len(df)),
        columns=columns,
        nominal_f2_mm=nominal_f2,
    )
    result.check_true(
        "every_perturbation_sample_produced_a_row",
        "invariant",
        len(df) == 3 * STEPS,
        f"{len(df)} rows == 3 perturbations x {STEPS} steps",
    )
    result.check_finite(
        "all_numeric_result_columns_finite", np.asarray(df[numeric], dtype=float)
    )

    perturbation_types = [str(v) for v in df["perturbation_type"].unique()]
    f2_col = next(c for c in columns if c.startswith("0:"))
    spot_col = next(c for c in columns if c.startswith("1:"))
    result.record(perturbation_types=perturbation_types, f2_column=f2_col, spot_column=spot_col)

    # perturbation_type is a human-readable label, e.g. "Radius of Curvature,
    # Surface 1" / "Tilt X, Surface 1" / "Thickness, Surface 1" -- not the keyword
    # that was passed to add_perturbation, so match case-insensitively.
    def _slice(kind: str):
        mask = df["perturbation_type"].astype(str).str.contains(kind, case=False)
        sub = df.loc[mask].sort_values("perturbation_value")
        return (
            np.asarray(sub["perturbation_value"], dtype=float),
            np.asarray(sub[f2_col], dtype=float),
            np.asarray(sub[spot_col], dtype=float),
        )

    # -- radius: focal length must fall monotonically as R shrinks -------------
    radii, f2_radius, _ = _slice("radius")
    result.record(
        radius_sweep_mm=[float(radii.min()), float(radii.max())],
        f2_over_radius_sweep_mm=[float(f2_radius.min()), float(f2_radius.max())],
    )
    result.check_true(
        "focal_length_increases_monotonically_with_the_first_radius",
        "analytic",
        bool(np.all(np.diff(f2_radius) > 0.0)),
        f"f2 rises monotonically from {f2_radius[0]:.4f} to {f2_radius[-1]:.4f} mm as R1 "
        f"goes {radii[0]:.1f} -> {radii[-1]:.1f} mm, at all {radii.size} samples: "
        "flattening a positive surface weakens the system",
    )
    # Thin-lens scaling: 1/f is affine in 1/R1, so f2 vs 1/R1 must be near-linear.
    inv_r = 1.0 / radii
    inv_f = 1.0 / f2_radius
    slope, intercept = np.polyfit(inv_r, inv_f, 1)
    residual = float(np.max(np.abs(inv_f - (slope * inv_r + intercept))))
    result.record(
        inverse_f_vs_inverse_r_slope=float(slope),
        inverse_f_vs_inverse_r_intercept=float(intercept),
        inverse_f_linear_fit_max_residual=residual,
    )
    result.check_true(
        "system_power_is_affine_in_the_first_surface_curvature",
        "analytic",
        residual < 0.02 * float(np.abs(inv_f).max()),
        f"1/f2 against 1/R1 fits a straight line with max residual {residual:.3e} "
        f"({residual / float(np.abs(inv_f).max()) * 100:.3f}% of 1/f). The fitted slope "
        f"{float(slope):.6f} is the surface's (n-1) contribution to system power, which "
        "is what thin-lens theory predicts and no Optiland accessor was asked for.",
    )

    # -- tilt: f2 even in the tilt, spot minimized at zero ---------------------
    tilts, f2_tilt, spot_tilt = _slice("tilt")
    centre = int(np.argmin(np.abs(tilts)))
    f2_spread = float(f2_tilt.max() - f2_tilt.min())
    result.record(
        tilt_sweep_rad=[float(tilts.min()), float(tilts.max())],
        f2_spread_over_tilt_mm=f2_spread,
        spot_at_zero_tilt_mm=float(spot_tilt[centre]),
        spot_at_max_tilt_mm=float(spot_tilt[-1]),
    )
    result.check_true(
        "surface_tilt_does_not_change_first_order_power",
        "analytic",
        f2_spread < 1e-9,
        f"f2 varies by {f2_spread:.3e} mm over a +/-0.05 rad x-tilt of surface 1: a "
        "rotation about an axis through the vertex leaves the paraxial power alone",
    )
    result.check_true(
        "spot_size_is_minimised_at_zero_tilt",
        "analytic",
        float(spot_tilt[centre]) == float(spot_tilt.min()),
        f"RMS spot size {float(spot_tilt[centre]):.6f} mm at tilt "
        f"{float(tilts[centre]):+.4f} rad is the minimum over the sweep, rising to "
        f"{float(spot_tilt[-1]):.6f} mm at {float(tilts[-1]):+.4f} rad",
    )
    left = spot_tilt[:centre][::-1]
    right = spot_tilt[centre + 1 :]
    pairs = min(left.size, right.size)
    asymmetry = float(np.max(np.abs(left[:pairs] - right[:pairs]))) if pairs else 0.0
    result.record(tilt_spot_asymmetry_mm=asymmetry)
    result.check_true(
        "spot_degradation_is_even_in_the_tilt",
        "analytic",
        asymmetry < 0.05 * float(spot_tilt.max()),
        f"max |spot(-t) - spot(+t)| = {asymmetry:.3e} mm against a peak of "
        f"{float(spot_tilt.max()):.6f} mm: an on-axis field cannot distinguish the sign "
        "of a tilt at leading order",
    )

    # -- thickness: weak, second-order effect on power -------------------------
    thicknesses, f2_thickness, _ = _slice("thickness")
    thickness_spread = float(f2_thickness.max() - f2_thickness.min())
    radius_spread = float(f2_radius.max() - f2_radius.min())
    result.record(
        thickness_sweep_mm=[float(thicknesses.min()), float(thicknesses.max())],
        f2_spread_over_thickness_mm=thickness_spread,
        f2_spread_over_radius_mm=radius_spread,
    )
    result.check_true(
        "element_thickness_is_a_far_weaker_lever_than_curvature",
        "analytic",
        0.0 < thickness_spread < 0.1 * radius_spread,
        f"f2 moves {thickness_spread:.4f} mm over a 2->5 mm thickness change against "
        f"{radius_spread:.4f} mm over the radius sweep: thickness enters only through "
        "the second-order (n-1)^2 t/(n R1 R2) term",
    )

    # -- the analysis must not leave the optic perturbed ------------------------
    restored_f2 = float(np.asarray(optic.paraxial.f2()).ravel()[0])
    result.record(f2_after_analysis_mm=restored_f2)
    result.check_close(
        "the_optic_is_restored_to_nominal_after_the_sweep",
        "invariant",
        restored_f2,
        nominal_f2,
        rel=1e-12,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
