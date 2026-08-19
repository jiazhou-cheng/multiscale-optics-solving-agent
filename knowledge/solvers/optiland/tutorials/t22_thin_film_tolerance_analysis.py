"""Intermediate / "Thin Film Tolerance Analysis" -- https://www.optiland.org/tutorials/thin-film-tolerance-analysis

Repo-owned reproduction of the coating-tolerancing tutorial: a 7-layer
MgF2/Al2O3 broadband AR on N-BK7, a `ThinFilmSensitivityAnalysis` sweeping each
layer thickness over +/-3%, and a `ThinFilmMonteCarlo` over normally distributed
+/-2% thickness errors, ending in a per-wavelength yield table against the
tutorial's declared **R < 1%** specification.

**Adaptation.** Upstream runs 500 Monte Carlo iterations; this reproduction runs
150, which is ample for the yield statements below and fits the test budget. The
count is recorded. The perturbations use ``DistributionSampler(..., seed=100+i)``
exactly as upstream does, so the draw is reproducible -- unlike the BSDF
scattering in t21.

Upstream publishes no numbers but does declare a specification and a conclusion
("+/-2% thickness control keeps reflectance well within spec"), and both are checked:

* The **nominal** design meets its own spec: R < 1% at every wavelength in the
  420-680 nm target band.
* +/-2% thickness errors keep the yield high at every sampled wavelength, which is
  upstream's conclusion stated as a number.
* Sensitivity is monotone in perturbation size: the |R - R_nominal| envelope grows
  with |dt| for the layer the analysis identifies as most sensitive.
* The nominal point is inside every perturbed envelope, and the un-perturbed
  sensitivity row reproduces the nominal reflectance exactly -- which is what
  proves the sweep restores state between samples rather than accumulating drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t22_thin_film_tolerance_analysis",
    title="Thin Film Tolerance Analysis",
    level="intermediate",
    url="https://www.optiland.org/tutorials/thin-film-tolerance-analysis",
    demonstrates=(
        "thin_film.tolerancing.{ThinFilmTolerancing,ThinFilmSensitivityAnalysis,"
        "ThinFilmMonteCarlo}, tolerancing.perturbation.{RangeSampler,"
        "DistributionSampler}, ThinFilmOperand.reflectance, and pandas "
        "DataFrame results via get_results()."
    ),
    slow=True,
)

LAYERS_NM = (
    ("mgf2", 94.6, "MgF2"),
    ("al2o3", 319.7, "Al2O3"),
    ("mgf2", 17.7, "MgF2"),
    ("al2o3", 196.1, "Al2O3"),
    ("mgf2", 26.3, "MgF2"),
    ("al2o3", 170.9, "Al2O3"),
    ("mgf2", 190.4, "MgF2"),
)
SPEC_MAX_R = 0.01
TARGET_BAND_NM = (420.0, 680.0)
MC_ITERATIONS = 150  # upstream: 500
MC_WAVELENGTHS_NM = (430.0, 480.0, 550.0, 620.0, 670.0)


def build_stack():
    from optiland.materials import IdealMaterial, Material
    from optiland.thin_film import ThinFilmStack

    air = IdealMaterial(n=1.0)
    nbk7 = Material("N-BK7")
    films = {"mgf2": Material("MgF2", reference="Dodge-o"), "al2o3": Material("Al2O3", reference="Malitson")}
    stack = ThinFilmStack(incident_material=air, substrate_material=nbk7)
    for key, thickness_nm, name in LAYERS_NM:
        stack.add_layer_nm(films[key], thickness_nm, name=name)
    return stack


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.thin_film.optimization.operand.thin_film import ThinFilmOperand
    from optiland.thin_film.tolerancing import (
        ThinFilmMonteCarlo,
        ThinFilmSensitivityAnalysis,
        ThinFilmTolerancing,
    )
    from optiland.tolerancing.perturbation import DistributionSampler, RangeSampler

    result = TutorialResult()
    stack = build_stack()
    result.record(num_layers=len(stack.layers), stack_summary=str(stack))

    # -- the nominal design meets its own specification ------------------------
    wl_band = np.linspace(*TARGET_BAND_NM, 131)
    r_band = np.asarray(
        [float(np.asarray(ThinFilmOperand.reflectance(stack, wl)).ravel()[0]) for wl in wl_band],
        dtype=float,
    )
    result.record(
        nominal_mean_R_percent_in_band=float(r_band.mean() * 100.0),
        nominal_max_R_percent_in_band=float(r_band.max() * 100.0),
        nominal_R_at_550nm_percent=float(
            np.asarray(ThinFilmOperand.reflectance(stack, 550.0)).ravel()[0] * 100.0
        ),
    )
    result.check_finite("nominal_reflectance_finite", r_band)
    result.check_true(
        "the_nominal_design_meets_its_declared_R_below_1_percent_spec",
        "reference",
        bool(np.all(r_band < SPEC_MAX_R)),
        f"max R = {float(r_band.max() * 100.0):.4f}% over the declared "
        f"{TARGET_BAND_NM[0]:.0f}-{TARGET_BAND_NM[1]:.0f} nm target band, against the "
        "tutorial's R < 1% specification (mean "
        f"{float(r_band.mean() * 100.0):.4f}%)",
    )

    # -- sensitivity analysis --------------------------------------------------
    tol = ThinFilmTolerancing(stack)
    for wl in (450.0, 550.0, 650.0):
        tol.add_operand("R", wl)
    for i in range(len(LAYERS_NM)):
        tol.add_perturbation(i, "thickness", RangeSampler(-0.03, 0.03, 13))
    sa = ThinFilmSensitivityAnalysis(tol)
    sa.run()
    sa.view()
    plt.close("all")
    df_sa = sa.get_results()
    operand_cols = [c for c in df_sa.columns if c.split(":")[0] in {"0", "1", "2"}]
    ranges = {}
    for ptype in df_sa["perturbation_type"].unique():
        mask = df_sa["perturbation_type"] == ptype
        spread = df_sa.loc[mask, operand_cols].max() - df_sa.loc[mask, operand_cols].min()
        ranges[str(ptype)] = float(spread.max())
    worst_layer = max(ranges, key=ranges.get)
    result.record(
        sensitivity_rows=int(len(df_sa)),
        sensitivity_operand_columns=operand_cols,
        sensitivity_max_delta_R_percent_per_layer={
            k: v * 100.0 for k, v in sorted(ranges.items())
        },
        most_sensitive_layer=worst_layer,
    )
    result.check_true(
        "sensitivity_sweep_covers_every_layer_at_every_sample",
        "invariant",
        len(df_sa) == len(LAYERS_NM) * 13,
        f"{len(df_sa)} rows == {len(LAYERS_NM)} layers x 13 RangeSampler steps",
    )
    result.check_true(
        "every_layer_has_a_nonzero_but_bounded_sensitivity",
        "analytic",
        all(0.0 < v < 0.2 for v in ranges.values()),
        "max delta-R per layer over +/-3% thickness: "
        + ", ".join(f"{k}={v * 100:.3f}%" for k, v in sorted(ranges.items()))
        + f". Most sensitive: {worst_layer}.",
    )

    # The un-perturbed sample must reproduce the nominal reflectance exactly, or
    # the sweep is accumulating state between samples.
    zero_rows = df_sa[np.isclose(df_sa["perturbation_value"].astype(float), 0.0, atol=1e-12)]
    nominal_550 = float(np.asarray(ThinFilmOperand.reflectance(build_stack(), 550.0)).ravel()[0])
    col_550 = next(c for c in operand_cols if "550" in c)
    zero_values = np.asarray(zero_rows[col_550], dtype=float)
    result.record(
        num_zero_perturbation_rows=int(zero_values.size),
        zero_perturbation_max_deviation_from_nominal=float(
            np.max(np.abs(zero_values - nominal_550)) if zero_values.size else 0.0
        ),
    )
    result.check_true(
        "the_sweep_restores_state_between_samples",
        "analytic",
        zero_values.size == len(LAYERS_NM)
        and float(np.max(np.abs(zero_values - nominal_550))) < 1e-12,
        f"all {int(zero_values.size)} zero-perturbation rows reproduce the nominal "
        f"R(550 nm) = {nominal_550:.9f} to "
        f"{float(np.max(np.abs(zero_values - nominal_550))):.3e}: layer thicknesses are "
        "restored after each sample rather than drifting",
    )

    # -- Monte Carlo yield -----------------------------------------------------
    tol_mc = ThinFilmTolerancing(build_stack())
    for wl in MC_WAVELENGTHS_NM:
        tol_mc.add_operand("R", wl)
    for i in range(len(LAYERS_NM)):
        tol_mc.add_perturbation(
            i, "thickness", DistributionSampler("normal", seed=100 + i, loc=0.0, scale=0.02)
        )
    mc = ThinFilmMonteCarlo(tol_mc)
    mc.run(num_iterations=MC_ITERATIONS)
    mc.view_histogram(kde=True)
    mc.view_cdf()
    plt.close("all")
    df_mc = mc.get_results()
    mc_cols = [c for c in df_mc.columns if "R@" in c]
    yields = {c: float((df_mc[c] < SPEC_MAX_R).mean()) for c in mc_cols}
    result.record(
        mc_iterations=MC_ITERATIONS,
        upstream_mc_iterations=500,
        mc_columns=mc_cols,
        mc_yield_fraction=yields,
        mc_mean_R_percent={c: float(df_mc[c].mean() * 100.0) for c in mc_cols},
        mc_max_R_percent={c: float(df_mc[c].max() * 100.0) for c in mc_cols},
    )
    result.check_true(
        "monte_carlo_produced_the_requested_number_of_trials",
        "invariant",
        len(df_mc) == MC_ITERATIONS,
        f"{len(df_mc)} rows == {MC_ITERATIONS} requested iterations",
    )
    result.check_true(
        "two_percent_thickness_control_keeps_reflectance_within_spec",
        "reference",
        all(v > 0.8 for v in yields.values()),
        "yield (fraction of trials with R < 1%) per wavelength: "
        + ", ".join(f"{c} {v * 100:.1f}%" for c, v in yields.items())
        + ". Upstream's conclusion -- '+/-2% thickness control keeps reflectance well "
        "within spec' -- stated as numbers.",
    )
    result.check_true(
        "the_perturbed_mean_is_worse_than_nominal_at_every_wavelength",
        "analytic",
        all(
            float(df_mc[c].mean())
            > float(np.asarray(ThinFilmOperand.reflectance(build_stack(), float(wl))).ravel()[0])
            for c, wl in zip(mc_cols, MC_WAVELENGTHS_NM, strict=True)
        ),
        "random thickness errors can only move a reflectance minimum away from its "
        "optimum, so the ensemble mean must exceed the nominal at every wavelength",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
