"""Advanced / "Dichroic Mirror Optimization for Polarization Separation" -- https://www.optiland.org/tutorials/thin-film-optimization

Repo-owned reproduction of the polarization-beamsplitter tutorial: a 20-layer
TiO2/SiO2 quarter-wave stack referenced at 0.6 um / 45 degrees, then optimized with
`thin_film.optimization.ThinFilmOptimizer` under L-BFGS-B against a **custom
registered operand** that maximizes ``mean(Rs - Rp)`` over 595-605 nm at 45 degrees.

**Adaptation.** ``max_iterations`` is reduced from 1000 to 200; the recorded merit
shows the optimization has converged well before that. The count is recorded.

Upstream prints merit values but publishes none. Validation is the design intent
plus the physics of a polarizing stack:

* The custom operand's declared floor is ``min_val=0.99``, i.e.
  ``(1 + mean(Rs - Rp))/2 >= 0.99`` or ``mean(Rs - Rp) >= 0.98``. That is a real
  external target, and it is **not met**: L-BFGS-B converges to 0.9719 (mean
  ``Rs - Rp`` = 0.944) in 47 of the 200 allowed iterations, so it is a local optimum
  of the 20-variable problem rather than an iteration shortfall.
  ``add_operand(min_val=...)`` states a goal, not a guarantee. The design does improve
  substantially -- contrast 0.684 -> 0.972 -- so the method works.
* ``Rs > Rp`` at every wavelength in the 595-605 nm band after optimization: a
  polarization beamsplitter that reflects s and transmits p.
* The optimizer strictly reduces the residual-sum-of-squares, and every layer
  thickness ends inside its declared 30-300 nm bounds.
* Energy is conserved at both polarizations across the whole 500-700 nm plotting
  range (``R + T + A == 1`` to float64 round-off), before and after.
* ``optimizer.reset()`` restores the starting thicknesses exactly, which is what
  makes the before/after comparison in the tutorial's plot legitimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t31_dichroic_mirror_optimization",
    title="Dichroic Mirror Optimization for Polarization Separation",
    level="advanced",
    url="https://www.optiland.org/tutorials/thin-film-optimization",
    demonstrates=(
        "thin_film.optimization.ThinFilmOptimizer: add_variable(layer_index, "
        "min_nm, max_nm), register_operand(name, callable, overwrite), "
        "add_operand(property, min_val, input_data, label), rss(), "
        "optimize(method='L-BFGS-B', max_iterations), reset(), and "
        "ThinFilmStack.reflectance_nm_deg / Layer.update_thickness."
    ),
    slow=True,
)

REFERENCE_WL_UM = 0.6
REFERENCE_AOI_DEG = 45.0
NUM_PAIRS = 10
BAND_NM = (595.0, 605.0)
MIN_CONTRAST_OPERAND = 0.99
MAX_ITERATIONS = 200  # upstream: 1000
LAYER_BOUNDS_NM = (30.0, 300.0)


def build_stack():
    from optiland.materials import IdealMaterial, Material
    from optiland.thin_film import ThinFilmStack

    sio2 = Material("SiO2", reference="Gao")
    tio2 = Material("TiO2", reference="Zhukovsky")
    bk7 = Material("N-BK7", reference="SCHOTT")
    air = IdealMaterial(n=1.0)
    stack = ThinFilmStack(
        incident_material=air,
        substrate_material=bk7,
        reference_wl_um=REFERENCE_WL_UM,
        reference_AOI_deg=REFERENCE_AOI_DEG,
    )
    for _ in range(NUM_PAIRS):
        stack.add_layer_qwot(material=tio2, qwot_thickness=1.0, name="TiO2")
        stack.add_layer_qwot(material=sio2, qwot_thickness=1.0, name="SiO2")
    return stack


def polarization_contrast(stack, wavelength_nm, aoi_deg):
    """(1 + mean(Rs - Rp)) / 2 -- upstream's custom merit, in [0, 1]."""
    rs = stack.reflectance_nm_deg(wavelength_nm, aoi_deg, "s")
    rp = stack.reflectance_nm_deg(wavelength_nm, aoi_deg, "p")
    return (1 + np.mean(np.asarray(rs, dtype=float) - np.asarray(rp, dtype=float))) / 2


def _contrast(stack, wavelengths_nm) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(
        stack.reflectance_nm_deg(wavelengths_nm, REFERENCE_AOI_DEG, "s"), dtype=float
    ).ravel()
    rp = np.asarray(
        stack.reflectance_nm_deg(wavelengths_nm, REFERENCE_AOI_DEG, "p"), dtype=float
    ).ravel()
    return rs, rp


def _energy_residual(stack, wavelengths_um) -> float:
    worst = 0.0
    for polarization in ("s", "p"):
        out = stack.compute_rtRTA(
            wavelengths_um, aoi_rad=np.radians(REFERENCE_AOI_DEG), polarization=polarization
        )
        total = (
            np.asarray(out["R"], dtype=float)
            + np.asarray(out["T"], dtype=float)
            + np.asarray(out["A"], dtype=float)
        )
        worst = max(worst, float(np.abs(total - 1.0).max()))
    return worst


def run() -> TutorialResult:
    from optiland.thin_film.optimization import ThinFilmOptimizer

    result = TutorialResult()
    stack = build_stack()
    band_nm = np.linspace(*BAND_NM, 11)
    plot_nm = np.linspace(500.0, 700.0, 201)

    optimizer = ThinFilmOptimizer(stack)
    for layer_index in range(len(stack.layers)):
        optimizer.add_variable(
            layer_index=layer_index, min_nm=LAYER_BOUNDS_NM[0], max_nm=LAYER_BOUNDS_NM[1]
        )
    ThinFilmOptimizer.register_operand(
        "polarization_contrast", polarization_contrast, overwrite=True
    )
    optimizer.add_operand(
        property="polarization_contrast",
        min_val=MIN_CONTRAST_OPERAND,
        input_data={"wavelength_nm": band_nm, "aoi_deg": REFERENCE_AOI_DEG},
        label="Rs-Rp @ 595-605nm, 45deg",
    )
    optimizer.info()

    starting_thicknesses = [
        float(np.asarray(layer.thickness_um).ravel()[0]) for layer in stack.layers
    ]
    rs_before, rp_before = _contrast(stack, plot_nm)
    contrast_before = float(polarization_contrast(stack, band_nm, REFERENCE_AOI_DEG))
    rss_before = float(optimizer.rss())
    energy_before = _energy_residual(stack, plot_nm / 1000.0)

    outcome = optimizer.optimize(method="L-BFGS-B", max_iterations=MAX_ITERATIONS)
    contrast_after = float(polarization_contrast(stack, band_nm, REFERENCE_AOI_DEG))
    rss_after = float(optimizer.rss())
    rs_after, rp_after = _contrast(stack, plot_nm)
    final_thicknesses = [
        float(np.asarray(layer.thickness_um).ravel()[0]) for layer in stack.layers
    ]
    energy_after = _energy_residual(stack, plot_nm / 1000.0)
    in_band = (plot_nm >= BAND_NM[0]) & (plot_nm <= BAND_NM[1])

    result.record(
        num_layers=len(stack.layers),
        max_iterations=MAX_ITERATIONS,
        upstream_max_iterations=1000,
        optimize_result_keys=sorted(outcome.keys()) if isinstance(outcome, dict) else [],
        initial_merit=float(outcome["initial_merit"]) if isinstance(outcome, dict) else float("nan"),
        final_merit=float(outcome["final_merit"]) if isinstance(outcome, dict) else float("nan"),
        iterations=int(outcome["iterations"]) if isinstance(outcome, dict) else -1,
        rss_before=rss_before,
        rss_after=rss_after,
        contrast_operand_before=contrast_before,
        contrast_operand_after=contrast_after,
        mean_Rs_minus_Rp_in_band_before=float((rs_before - rp_before)[in_band].mean()),
        mean_Rs_minus_Rp_in_band_after=float((rs_after - rp_after)[in_band].mean()),
        starting_thickness_nm=[t * 1000.0 for t in starting_thicknesses],
        final_thickness_nm=[t * 1000.0 for t in final_thicknesses],
        max_energy_residual_before=energy_before,
        max_energy_residual_after=energy_after,
    )
    result.check_finite(
        "reflectance_curves_finite", np.concatenate([rs_after, rp_after, rs_before, rp_before])
    )
    result.check_true(
        "twenty_layers_became_twenty_variables",
        "invariant",
        len(optimizer.variables) == 2 * NUM_PAIRS,
        f"{len(optimizer.variables)} == 2 x {NUM_PAIRS} pairs",
    )
    result.check_true(
        "the_optimizer_reduces_the_residual_sum_of_squares",
        "invariant",
        rss_after < rss_before,
        f"RSS {rss_before:.6e} -> {rss_after:.6e} in "
        f"{int(outcome['iterations']) if isinstance(outcome, dict) else -1} iterations",
    )
    result.check_true(
        "the_declared_0p99_contrast_floor_is_approached_but_not_reached",
        "reference",
        0.95 <= contrast_after < MIN_CONTRAST_OPERAND,
        f"the custom operand (1 + mean(Rs - Rp))/2 reaches {contrast_after:.6f} from "
        f"{contrast_before:.6f}, i.e. mean(Rs - Rp) rises from {2 * contrast_before - 1:.6f} "
        f"to {2 * contrast_after - 1:.6f} over {BAND_NM[0]:.0f}-{BAND_NM[1]:.0f} nm at "
        f"{REFERENCE_AOI_DEG:.0f} degrees. The declared min_val of {MIN_CONTRAST_OPERAND} is "
        f"NOT met: L-BFGS-B converged in "
        f"{int(outcome['iterations']) if isinstance(outcome, dict) else -1} iterations, far "
        f"inside the {MAX_ITERATIONS} allowed, so this is a local optimum of the 20-variable "
        "problem rather than an iteration shortfall. add_operand(min_val=...) states a goal, "
        "not a guarantee.",
    )
    result.check_true(
        "s_reflects_more_than_p_across_the_whole_target_band",
        "analytic",
        bool(np.all((rs_after - rp_after)[in_band] > 0.0)),
        f"Rs - Rp in [{float((rs_after - rp_after)[in_band].min()):.6f}, "
        f"{float((rs_after - rp_after)[in_band].max()):.6f}] at every sampled wavelength in "
        "the band: the stack reflects s and transmits p, which is what a polarizing "
        "beamsplitter is",
    )
    violations = [
        t * 1000.0
        for t in final_thicknesses
        if not LAYER_BOUNDS_NM[0] - 1e-6 <= t * 1000.0 <= LAYER_BOUNDS_NM[1] + 1e-6
    ]
    result.check_true(
        "every_layer_thickness_respects_its_declared_bounds",
        "invariant",
        not violations,
        f"final thicknesses span "
        f"[{min(t * 1000.0 for t in final_thicknesses):.3f}, "
        f"{max(t * 1000.0 for t in final_thicknesses):.3f}] nm within "
        f"{LAYER_BOUNDS_NM} nm; violations {violations or 'none'}",
    )
    result.check_true(
        "energy_is_conserved_before_and_after_optimization",
        "analytic",
        energy_before < 1e-12 and energy_after < 1e-12,
        f"max |R + T + A - 1| over 500-700 nm at both polarizations: "
        f"{energy_before:.3e} before, {energy_after:.3e} after",
    )

    # -- reset() must restore the starting design exactly -----------------------
    optimizer.reset()
    reset_thicknesses = [
        float(np.asarray(layer.thickness_um).ravel()[0]) for layer in stack.layers
    ]
    result.record(
        max_reset_deviation_nm=float(
            np.max(np.abs(np.array(reset_thicknesses) - starting_thicknesses)) * 1000.0
        )
    )
    result.check_true(
        "reset_restores_the_starting_thicknesses_exactly",
        "analytic",
        float(np.max(np.abs(np.array(reset_thicknesses) - starting_thicknesses))) < 1e-15,
        f"max deviation "
        f"{float(np.max(np.abs(np.array(reset_thicknesses) - starting_thicknesses)) * 1000.0):.3e} nm. "
        "This is what makes the tutorial's before/after plot legitimate: reset() really "
        "does return the stack to the quarter-wave design.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
