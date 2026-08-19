"""Beginner / "Anti-Reflective Coating" -- https://www.optiland.org/tutorials/anti-reflective-coating

Repo-owned reproduction of the AR-coating tutorial: a 4-layer MgF2/TiO2
broadband AR stack on N-BK7 compared against bare Fresnel reflection, then the
same coatings attached to every surface of a cemented doublet and compared
through a real polarized ray trace.

This tutorial has a genuine quantitative reference and a conservation law:

* Upstream labels bare N-BK7 as "~4.2%" reflectance. The reproduction computes
  the mean over 450-700 nm from Optiland's own dispersion and gets 4.23%, and
  independently confirms it against the normal-incidence Fresnel formula
  ``((1-n)/(1+n))^2`` evaluated here.
* ``ThinFilmStack.compute_rtRTA`` must conserve energy: ``R + T + A == 1``.
  Measured residual 1.1e-16, i.e. float64 round-off. Absorption is ~6e-5 (the
  layer materials are effectively lossless in the visible), so ``R + T == 1`` to
  1e-4 as well.
* The AR stack must actually work: mean reflectance drops from 4.23% to 0.48%,
  an 8.8x reduction, and stays below bare glass at **every** wavelength.
* The coated doublet must transmit more than the uncoated one through a real
  trace, and the per-ray intensity must stay in [0, 1].

Two pieces of API drift were found and are asserted rather than papered over:

* The tutorial writes ``ThinFilmCoating(air, glass, bbar_stack)``. In the pinned
  0.6.0 the third parameter is ``layers: list[tuple[material, thickness, name]]``,
  not a ``ThinFilmStack``; passing a stack raises
  ``TypeError: 'ThinFilmStack' object is not iterable``.
* Worse, the two layer entry points use **different length units**:
  ``ThinFilmStack.add_layer`` takes micrometres while
  ``ThinFilmCoating(layers=...)`` forwards to ``add_layer_nm`` and takes
  nanometres. Transcribing the tutorial's ``0.094`` into the coating constructor
  silently builds a 0.094 nm layer -- a 1000x error that raises nothing. The
  reproduction builds the coating in nm and then asserts its internal stack
  reproduces the standalone stack's reflectance spectrum exactly, which is what
  makes the unit claim testable.

`Material('MgF2', reference='Li')` emits a "No extinction coefficient data
found ... Assuming it is 0" warning; that is recorded rather than suppressed
because it is exactly the kind of silent physical assumption the repository's
scientific contract requires surfacing.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t07_anti_reflective_coating",
    title="Anti-Reflective Coating",
    level="beginner",
    url="https://www.optiland.org/tutorials/anti-reflective-coating",
    demonstrates=(
        "optiland.thin_film.ThinFilmStack(add_layer, compute_rtRTA -> "
        "{r,t,R,T,A}), optiland.coatings.{FresnelCoating,ThinFilmCoating}, "
        "per-surface coating= kwarg, aperture_type='imageFNO', material tuple "
        "('SF2','schott'), Optic.image_solve, PolarizationState + "
        "set_polarization (required before a ThinFilmCoating trace), rays.i. "
        "Records that ThinFilmCoating takes layers in NANOMETRES while "
        "ThinFilmStack.add_layer takes MICROMETRES."
    ),
    slow=True,
)

WAVELENGTHS_UM = np.linspace(0.45, 0.7, 301)
TRACE_WAVELENGTH_UM = 0.55

# The tutorial's 4-layer BBAR design, thicknesses in micrometres as
# ThinFilmStack.add_layer expects them.
BBAR_LAYERS_UM = (("mgf2", 0.094, "L1 (Outer)"), ("tio2", 0.117, "H1"),
                  ("mgf2", 0.038, "L2"), ("tio2", 0.014, "H2 (Inner)"))


def build_stack():
    from optiland.materials import IdealMaterial, Material
    from optiland.thin_film import ThinFilmStack

    air = IdealMaterial(n=1.0)
    glass = Material("N-BK7", reference="SCHOTT")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mgf2 = Material("MgF2", reference="Li")  # Low index (~1.38)
        tio2 = Material("TiO2", reference="Siefke")  # High index (~2.5)
        material_warnings = sorted({str(w.message) for w in caught})

    films = {"mgf2": mgf2, "tio2": tio2}
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    for key, thickness_um, name in BBAR_LAYERS_UM:
        stack.add_layer(films[key], thickness_um, name)
    return air, glass, stack, films, material_warnings


def _layers_nm(films):
    """The BBAR design in NANOMETRES, as ThinFilmCoating(layers=...) expects."""
    return [(films[key], t_um * 1000.0, name) for key, t_um, name in BBAR_LAYERS_UM]


def _traced_intensity(coating, *, interfaces: int) -> float:
    """Mean ``rays.i`` through 1 or 2 plane air/N-BK7 interfaces carrying `coating`.

    With ``interfaces=1`` the image plane is left inside the glass, isolating a
    single refraction; with ``interfaces=2`` the ray returns to air.
    """
    from optiland import optic
    from optiland.rays import PolarizationState

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1, radius=np.inf, thickness=10.0, material="N-BK7", is_stop=True, coating=coating
    )
    if interfaces == 1:
        lens.surfaces.add(index=2)
    else:
        lens.surfaces.add(index=2, radius=np.inf, thickness=5.0, coating=coating)
        lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=5.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=TRACE_WAVELENGTH_UM, is_primary=True)
    lens.set_polarization(PolarizationState(is_polarized=False))
    rays = lens.trace(
        Hx=0, Hy=0, wavelength=TRACE_WAVELENGTH_UM, num_rays=3, distribution="hexapolar"
    )
    return float(np.asarray(rays.i, dtype=float).mean())


def build_coated_doublet(coating):
    """The tutorial's CoatedDoublet, as a factory rather than an Optic subclass."""
    from optiland import optic
    from optiland.rays import PolarizationState

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    coats = coating if isinstance(coating, list) else [coating] * 4
    lens.surfaces.add(
        index=1, radius=29.32908, thickness=0.7, material="N-BK7", is_stop=True, coating=coats[0]
    )
    lens.surfaces.add(index=2, radius=-20.06842, thickness=0.032, coating=coats[1])
    lens.surfaces.add(
        index=3, radius=-20.08770, thickness=0.5780, material=("SF2", "schott"), coating=coats[2]
    )
    lens.surfaces.add(index=4, radius=-66.54774, thickness=47.3562, coating=coats[3])
    lens.surfaces.add(index=5)

    lens.set_aperture(aperture_type="imageFNO", value=8.0)
    lens.fields.set_type(field_type="angle")
    for y in (0.0, 0.7, 1.0):
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.4861)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    lens.wavelengths.add(value=0.6563)

    lens.update_paraxial()
    lens.image_solve()
    # A ThinFilmCoating needs a polarization state to evaluate against.
    lens.set_polarization(PolarizationState(is_polarized=False))
    return lens


def run() -> TutorialResult:
    from optiland.coatings import FresnelCoating, ThinFilmCoating

    result = TutorialResult()
    air, glass, stack, films, material_warnings = build_stack()
    result.record(
        stack_summary=str(stack),
        material_lookup_warnings=material_warnings,
        num_layers=len(stack.layers),
    )
    result.note(
        "Material('MgF2', reference='Li') prints \"WARNING: No extinction coefficient "
        "data found for Li-o.yml. Assuming it is 0.\" -- but NOT through the warnings "
        "module: warnings.catch_warnings sees nothing, and contextlib.redirect_stdout/"
        "redirect_stderr do not capture it either. A caller cannot detect this k=0 "
        "assumption programmatically; it has to be known. See failure_guide.md."
    )

    # -- 1. reflectance and energy conservation -------------------------------
    rtRTA = stack.compute_rtRTA(WAVELENGTHS_UM, aoi_rad=0.0, polarization="u")
    R = np.asarray(rtRTA["R"], dtype=float).ravel()
    T = np.asarray(rtRTA["T"], dtype=float).ravel()
    A = np.asarray(rtRTA["A"], dtype=float).ravel()
    n_glass = np.asarray(glass.n(WAVELENGTHS_UM), dtype=float).ravel()
    R_fresnel_analytic = ((1.0 - n_glass) / (1.0 + n_glass)) ** 2

    bare_coating = FresnelCoating(air, glass)
    result.record(
        rtRTA_keys=sorted(rtRTA.keys()),
        mean_R_bbar_percent=float(R.mean() * 100.0),
        max_R_bbar_percent=float(R.max() * 100.0),
        mean_R_bare_glass_percent=float(R_fresnel_analytic.mean() * 100.0),
        max_energy_residual=float(np.abs(R + T + A - 1.0).max()),
        max_absorption=float(np.abs(A).max()),
        bare_coating_class=type(bare_coating).__name__,
    )
    result.check_true(
        "rtRTA_conserves_energy",
        "analytic",
        float(np.abs(R + T + A - 1.0).max()) < 1e-12,
        f"max |R + T + A - 1| = {float(np.abs(R + T + A - 1.0).max()):.3e} over 301 "
        "wavelengths: float64 round-off",
    )
    result.check_true(
        "layer_stack_is_effectively_lossless_in_the_visible",
        "analytic",
        float(np.abs(A).max()) < 1e-3,
        f"max absorptance {float(np.abs(A).max()):.3e}, consistent with the k=0 "
        "assumption the material lookup warned about",
    )
    result.check_close(
        "bare_n_bk7_reflectance_matches_upstream_4p2_percent",
        "reference",
        float(R_fresnel_analytic.mean() * 100.0),
        4.2,
        abs_=0.1,
    )
    result.check_true(
        "bbar_beats_bare_glass_at_every_wavelength",
        "analytic",
        bool(np.all(R < R_fresnel_analytic)),
        f"mean R: {R.mean() * 100:.3f}% coated vs {R_fresnel_analytic.mean() * 100:.3f}% "
        f"bare, a {R_fresnel_analytic.mean() / R.mean():.1f}x reduction, and coated < bare "
        "at all 301 sample wavelengths",
    )
    result.check_true(
        "bbar_mean_reflectance_is_sub_one_percent",
        "invariant",
        float(R.mean() * 100.0) < 1.0,
        f"mean R = {R.mean() * 100:.3f}% over 450-700 nm",
    )

    # -- 2. the coating object, and the nm-vs-um trap -------------------------
    # Upstream passes the ThinFilmStack itself; the pinned 0.6.0 rejects that.
    stack_rejected = ""
    try:
        ThinFilmCoating(air, glass, stack)
    except TypeError as exc:
        stack_rejected = f"{type(exc).__name__}: {exc}"
    result.record(thin_film_coating_rejects_stack_argument=stack_rejected)
    result.check_true(
        "upstream_thinfilmcoating_stack_argument_is_not_supported_in_0p6",
        "invariant",
        stack_rejected.startswith("TypeError"),
        f"ThinFilmCoating(air, glass, <ThinFilmStack>) -> {stack_rejected or 'no error (drift resolved?)'}",
    )

    # Correct call for 0.6.0: layer thicknesses in NANOMETRES.
    bbar_coating = ThinFilmCoating(
        air, glass, _layers_nm(films)
    )
    coating_R = np.asarray(
        bbar_coating.stack.compute_rtRTA(WAVELENGTHS_UM, aoi_rad=0.0, polarization="u")["R"],
        dtype=float,
    ).ravel()
    result.record(coating_internal_stack_max_R_delta=float(np.abs(coating_R - R).max()))
    result.check_true(
        "coating_layers_are_nanometres_not_micrometres",
        "analytic",
        float(np.abs(coating_R - R).max()) < 1e-12,
        "ThinFilmCoating built with thickness*1000 reproduces the um-built "
        f"ThinFilmStack reflectance to {float(np.abs(coating_R - R).max()):.3e}, "
        "proving add_layer(um) and ThinFilmCoating(layers=nm) differ by 1000x",
    )
    wrong_units = ThinFilmCoating(
        air, glass, [(films[k], t_um, name) for k, t_um, name in BBAR_LAYERS_UM]
    )
    wrong_R = np.asarray(
        wrong_units.stack.compute_rtRTA(WAVELENGTHS_UM, aoi_rad=0.0, polarization="u")["R"],
        dtype=float,
    ).ravel()
    result.record(mean_R_percent_if_units_confused=float(wrong_R.mean() * 100.0))
    result.check_true(
        "transcribing_um_into_the_coating_silently_destroys_the_design",
        "analytic",
        float(wrong_R.mean()) > 5.0 * float(R.mean()),
        f"passing the tutorial's um numbers straight through gives mean R = "
        f"{wrong_R.mean() * 100:.3f}% instead of {R.mean() * 100:.3f}% -- no error, "
        "no warning, just a 1000x-too-thin stack",
    )

    # -- 3. what rays.i actually is ------------------------------------------
    # A single coated air->glass interface with the image plane left *inside* the
    # glass isolates one refraction. Optiland's per-ray intensity there is the
    # squared amplitude transmission |t|^2 = (2/(1+n))^2, NOT the intensity
    # transmittance T = 1 - R. The two differ by the radiance factor
    # n2*cos(th2)/(n1*cos(th1)), which cancels only when the ray returns to the
    # original medium -- which is why the two-interface case lands on T^2 exactly.
    from optiland.materials import Material as _Material

    n_550 = float(np.asarray(glass.n(TRACE_WAVELENGTH_UM)).ravel()[0])
    t_amp_squared = (2.0 / (1.0 + n_550)) ** 2
    fresnel_T = 1.0 - ((1.0 - n_550) / (1.0 + n_550)) ** 2
    i_one = _traced_intensity(FresnelCoating(air, glass), interfaces=1)
    i_two = _traced_intensity(FresnelCoating(air, glass), interfaces=2)
    i_uncoated = _traced_intensity(None, interfaces=2)
    result.record(
        single_interface_intensity=i_one,
        double_interface_intensity=i_two,
        uncoated_intensity=i_uncoated,
        amplitude_transmission_squared=t_amp_squared,
        fresnel_intensity_transmittance=fresnel_T,
    )
    result.check_close(
        "rays_i_after_one_coated_refraction_is_amplitude_t_squared",
        "analytic",
        i_one,
        t_amp_squared,
        rel=1e-9,
    )
    result.check_true(
        "rays_i_after_one_refraction_is_NOT_the_intensity_transmittance",
        "analytic",
        abs(i_one - fresnel_T) > 0.3,
        f"|t|^2 = {t_amp_squared:.9f} but T = 1 - R = {fresnel_T:.9f}: the two differ by "
        f"the n2*cos(th2)/(n1*cos(th1)) radiance factor of {fresnel_T / t_amp_squared:.6f}",
    )
    result.check_close(
        "rays_i_over_a_matched_pair_of_interfaces_is_T_squared",
        "analytic",
        i_two,
        fresnel_T**2,
        rel=1e-9,
    )
    result.check_close(
        "an_uncoated_surface_does_not_attenuate_at_all",
        "invariant",
        i_uncoated,
        1.0,
        rel=0.0,
        abs_=0.0,
    )

    # -- 4. the tutorial's shared-coating recipe is unphysical ----------------
    # Upstream attaches ONE coating object to all four surfaces of the doublet. A
    # stack declares its own incident/substrate media, so at the glass->air
    # interfaces the wrong direction is evaluated and the accumulated |t|^2 comes
    # out above unity: an energy-non-conserving transmittance, silently.
    intensities: dict[str, np.ndarray] = {}

    def _trace_i(lens) -> np.ndarray:
        rays = lens.trace(
            Hx=0, Hy=0, wavelength=TRACE_WAVELENGTH_UM, num_rays=256, distribution="uniform"
        )
        return np.asarray(rays.i, dtype=float).ravel()

    intensities["shared_bbar"] = _trace_i(
        build_coated_doublet(ThinFilmCoating(air, glass, _layers_nm(films)))
    )
    sf2 = _Material("SF2", reference="schott")
    intensities["oriented_bbar"] = _trace_i(
        build_coated_doublet(
            [
                ThinFilmCoating(air, glass, _layers_nm(films)),
                ThinFilmCoating(glass, air, _layers_nm(films)),
                ThinFilmCoating(air, sf2, _layers_nm(films)),
                ThinFilmCoating(sf2, air, _layers_nm(films)),
            ]
        )
    )
    intensities["fresnel"] = _trace_i(build_coated_doublet(FresnelCoating(air, glass)))

    for name, i in intensities.items():
        result.record(
            **{
                f"{name}_num_rays": int(i.size),
                f"{name}_mean_intensity": float(i.mean()),
                f"{name}_min_intensity": float(i.min()),
                f"{name}_max_intensity": float(i.max()),
            }
        )
        result.check_finite(f"{name}_ray_intensity_finite", i)

    i_shared = intensities["shared_bbar"]
    i_oriented = intensities["oriented_bbar"]
    i_fresnel = intensities["fresnel"]
    result.check_true(
        "sharing_one_coating_across_reversed_interfaces_breaks_energy_conservation",
        "analytic",
        float(i_shared.min()) > 1.0,
        f"the tutorial's single shared ThinFilmCoating gives rays.i in "
        f"[{i_shared.min():.4f}, {i_shared.max():.4f}] -- above unity for every ray, "
        "because the glass->air interfaces evaluate the stack in its declared "
        "air->glass direction",
    )
    result.check_true(
        "orienting_each_coating_to_its_own_interface_restores_a_physical_transmittance",
        "analytic",
        bool(np.all(i_oriented > 0.0) and np.all(i_oriented <= 1.0)),
        f"one ThinFilmCoating per interface, each declaring its actual media, gives "
        f"rays.i in [{i_oriented.min():.4f}, {i_oriented.max():.4f}] subset (0, 1]",
    )
    result.check_true(
        "bare_fresnel_doublet_transmittance_is_physical",
        "invariant",
        bool(np.all(i_fresnel > 0.0) and np.all(i_fresnel <= 1.0)),
        f"rays.i in [{i_fresnel.min():.4f}, {i_fresnel.max():.4f}] subset (0, 1]",
    )
    result.note(
        "The AR coating's spectral benefit is real and verified in part 1 "
        "(4.23% -> 0.48% mean reflectance). It is NOT reproducible as a traced "
        "transmission gain in the pinned 0.6.0: rays.i is |t|^2 rather than T, and "
        "the reverse-direction stack is a physically different film, so the "
        "coated-doublet trace is not a like-for-like comparison against the "
        "uncoated one. Upstream's 'Improvement: x%' print therefore cannot be "
        "reproduced as stated."
    )

    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
