"""Intermediate / "Multilayer Stack" -- https://www.optiland.org/tutorials/multilayer-stack

Repo-owned reproduction of the dielectric-mirror tutorial: a 20-layer
quarter-wave TiO2/SiO2 stack on N-BK7 built with ``add_layer_qwot``, plotted with
``ThinFilmStack.plot_structure``, then analysed with ``SpectralAnalyzer`` over
wavelength (at 45 degrees, s and p) and as a wavenumber-vs-angle map.

Upstream is purely visual. Everything checked here is either a closed form or a
conservation law:

* ``add_layer_qwot(material, qwot_thickness=1.0)`` must produce a physical
  thickness of ``reference_wl / (4 n(reference_wl))``. Verified per layer against
  the material's own index at the declared 0.6 um reference -- which is what pins
  the meaning of ``reference_wl_um`` and ``qwot_thickness``.
* A 10-pair quarter-wave stack is a **high reflector**: at the reference
  wavelength and normal incidence its reflectance must exceed 99%. Measured
  99.98%. This is the design property, not an incidental number.
* Away from the stop band the reflectance must fall back towards the bare
  substrate. Checked at 1.2 um.
* ``R + T + A == 1`` at every wavelength and both polarizations (to 1e-15).
* At normal incidence s and p agree to 1.8e-14; at 45 degrees they differ by up to
  0.996 in reflectance. At 0 deg there is no plane of incidence to distinguish them, and
  not, and s must reflect more than p (the s-polarized Fresnel coefficient is
  larger in magnitude at every interface).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t19_multilayer_stack",
    title="Multilayer Stack",
    level="intermediate",
    url="https://www.optiland.org/tutorials/multilayer-stack",
    demonstrates=(
        "ThinFilmStack(reference_wl_um=...), add_layer_qwot, plot_structure, "
        "and thin_film.SpectralAnalyzer.wavelength_view / map_view with "
        "wavelength_unit='nm'|'wavenumber' and polarization=['s','p']."
    ),
    slow=True,
)

REFERENCE_WL_UM = 0.6
NUM_PAIRS = 10


def build_stack():
    from optiland.materials import IdealMaterial, Material
    from optiland.thin_film import ThinFilmStack

    sio2 = Material("SiO2", reference="Gao")
    tio2 = Material("TiO2", reference="Zhukovsky")
    bk7 = Material("N-BK7", reference="SCHOTT")
    air = IdealMaterial(n=1.0)

    stack = ThinFilmStack(
        incident_material=air, substrate_material=bk7, reference_wl_um=REFERENCE_WL_UM
    )
    for _ in range(NUM_PAIRS):
        stack.add_layer_qwot(material=tio2, qwot_thickness=1.0, name="TiO2")
        stack.add_layer_qwot(material=sio2, qwot_thickness=1.0, name="SiO2")
    return stack, {"SiO2": sio2, "TiO2": tio2, "N-BK7": bk7, "air": air}


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    import optiland.backend as be
    from optiland.thin_film import SpectralAnalyzer

    result = TutorialResult()
    stack, materials = build_stack()

    result.record(
        num_layers=len(stack.layers),
        reference_wl_um=REFERENCE_WL_UM,
        stack_summary=str(stack),
    )
    result.check_true(
        "ten_pairs_give_twenty_layers",
        "invariant",
        len(stack.layers) == 2 * NUM_PAIRS,
        f"{len(stack.layers)} layers == 2 x {NUM_PAIRS} pairs",
    )

    # -- the quarter-wave optical thickness closed form ------------------------
    observed = []
    predicted = []
    for layer in stack.layers:
        thickness_um = float(np.asarray(layer.thickness_um).ravel()[0])
        n_ref = float(np.asarray(layer.material.n(REFERENCE_WL_UM)).ravel()[0])
        observed.append(thickness_um)
        predicted.append(REFERENCE_WL_UM / (4.0 * n_ref))
    result.record(
        layer_thickness_um=observed,
        qwot_closed_form_um=predicted,
        max_qwot_error_um=float(np.max(np.abs(np.array(observed) - predicted))),
    )
    result.check_true(
        "qwot_thickness_is_reference_wavelength_over_four_n",
        "analytic",
        float(np.max(np.abs(np.array(observed) - predicted))) < 1e-12,
        f"max |t_layer - lambda_ref/(4 n)| = "
        f"{float(np.max(np.abs(np.array(observed) - predicted))):.3e} um over all "
        f"{len(observed)} layers, with n taken at the declared {REFERENCE_WL_UM} um "
        "reference",
    )

    # -- the stack is a high reflector at the reference wavelength --------------
    at_ref = stack.compute_rtRTA(REFERENCE_WL_UM, aoi_rad=0.0, polarization="u")
    r_ref = float(np.asarray(at_ref["R"]).ravel()[0])
    off_band = stack.compute_rtRTA(1.2, aoi_rad=0.0, polarization="u")
    r_off = float(np.asarray(off_band["R"]).ravel()[0])
    n_sub = float(np.asarray(materials["N-BK7"].n(1.2)).ravel()[0])
    r_bare = ((1.0 - n_sub) / (1.0 + n_sub)) ** 2
    result.record(
        reflectance_at_reference=r_ref,
        reflectance_at_1p2um=r_off,
        bare_substrate_reflectance_at_1p2um=r_bare,
    )
    result.check_true(
        "quarter_wave_stack_is_a_high_reflector_at_the_reference_wavelength",
        "analytic",
        r_ref > 0.99,
        f"R = {r_ref * 100:.3f}% at {REFERENCE_WL_UM} um and normal incidence: the "
        f"defining property of a {NUM_PAIRS}-pair quarter-wave dielectric mirror",
    )
    result.check_true(
        "reflectance_collapses_outside_the_stop_band",
        "analytic",
        r_off < 0.5 * r_ref,
        f"R falls from {r_ref * 100:.3f}% at {REFERENCE_WL_UM} um to {r_off * 100:.3f}% at "
        f"1.2 um, on its way back towards the {r_bare * 100:.3f}% bare-substrate value",
    )

    # -- energy conservation over the whole analysed band ----------------------
    wl_nm = np.asarray(be.to_numpy(be.linspace(350, 1200, 1000)), dtype=float)
    wl_um = wl_nm / 1000.0
    residuals = {}
    reflectance = {}
    for polarization in ("s", "p"):
        out = stack.compute_rtRTA(wl_um, aoi_rad=np.radians(45.0), polarization=polarization)
        R = np.asarray(out["R"], dtype=float).ravel()
        T = np.asarray(out["T"], dtype=float).ravel()
        A = np.asarray(out["A"], dtype=float).ravel()
        residuals[polarization] = float(np.abs(R + T + A - 1.0).max())
        reflectance[polarization] = R
        result.record(
            **{
                f"mean_R_{polarization}_at_45deg": float(R.mean()),
                f"max_energy_residual_{polarization}": residuals[polarization],
            }
        )
        result.check_finite(f"rtRTA_{polarization}_finite", np.concatenate([R, T, A]))
    result.check_true(
        "energy_is_conserved_for_both_polarizations_at_45_degrees",
        "analytic",
        all(v < 1e-12 for v in residuals.values()),
        f"max |R + T + A - 1| over 1000 wavelengths: s {residuals['s']:.3e}, "
        f"p {residuals['p']:.3e}",
    )
    result.check_true(
        "s_reflects_more_than_p_at_oblique_incidence",
        "analytic",
        float(reflectance["s"].mean()) > float(reflectance["p"].mean()),
        f"mean R at 45 deg: s {reflectance['s'].mean() * 100:.3f}% > p "
        f"{reflectance['p'].mean() * 100:.3f}%, as the larger s-polarized Fresnel "
        "amplitude at every interface requires",
    )

    normal = {
        pol: np.asarray(
            stack.compute_rtRTA(wl_um, aoi_rad=0.0, polarization=pol)["R"], dtype=float
        ).ravel()
        for pol in ("s", "p")
    }
    result.record(
        max_s_minus_p_at_normal=float(np.abs(normal["s"] - normal["p"]).max()),
        max_s_minus_p_at_45deg=float(np.abs(reflectance["s"] - reflectance["p"]).max()),
    )
    result.check_true(
        "s_and_p_are_degenerate_at_normal_incidence",
        "analytic",
        float(np.abs(normal["s"] - normal["p"]).max()) < 1e-12
        and float(np.abs(reflectance["s"] - reflectance["p"]).max()) > 0.1,
        f"max |R_s - R_p| = {float(np.abs(normal['s'] - normal['p']).max()):.3e} at normal "
        f"incidence but {float(np.abs(reflectance['s'] - reflectance['p']).max()):.4f} at "
        "45 degrees: there is no plane of incidence to distinguish them at 0 deg",
    )

    # -- the tutorial's three figures -----------------------------------------
    stack.plot_structure()
    plt.close("all")
    analyzer = SpectralAnalyzer(stack=stack)
    analyzer.wavelength_view(
        be.linspace(350, 1200, 1000),
        wavelength_unit="nm",
        aoi=45,
        to_plot=["R"],
        polarization=["s", "p"],
    )
    plt.close("all")
    analyzer.map_view(
        wavelength_values=be.linspace(8000, 25000, 200),
        wavelength_unit="wavenumber",
        aoi_values=be.linspace(0, 80, 60),
        aoi_unit="deg",
        to_plot=["R"],
        polarization=["s", "p"],
        colormap="magma",
    )
    plt.close("all")
    result.check_true(
        "plot_structure_wavelength_view_and_map_view_render_headless",
        "qualitative",
        True,
        "all three SpectralAnalyzer/ThinFilmStack figures completed under Agg "
        "(map_view reduced from the tutorial's 1000x300 grid to 200x60 for runtime)",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
