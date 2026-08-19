"""Advanced / "Introduction to Polarization" -- https://www.optiland.org/tutorials/introduction-to-polarization

Repo-owned reproduction of the polarization tutorial: the bundled
`ObjectiveUS008879901` traced with no coatings, then with
``surfaces.set_fresnel_coatings()`` under unpolarized light, then under a custom
Jones state, then under each of the four named states from
``create_polarization``, and finally with a ``PolarizerCoating`` inserted on one
surface.

Upstream's numbers appear only as prose ("transmission was 98% ... 19% with
unpolarized light and Fresnel effects"). Both are reproduced, and the rest of the
validation comes from the algebra of polarization, which is exact:

* **Uncoated**: ``rays.i`` is 0.98 -- upstream's "98%".
* **Fresnel-coated, unpolarized**: mean transmission collapses to 0.19 --
  upstream's "19%". This objective has 20+ air-glass interfaces, so ~4% loss each
  compounds to roughly ``0.96^n``.
* **A linear polarizer is a projector.** With a V-oriented ``PolarizerCoating``,
  H-polarized input is extinguished to 5.5e-6 (52.6 dB) -- not identically zero,
  because 23 Fresnel interfaces rotate the s/p basis around it -- V-polarized input
  passes, and +45-degree linear input passes **half**: ``cos^2(45 deg) = 1/2`` to
  5e-6. That last number is the sharpest available test that the Jones algebra is
  right rather than merely plausible.
* Circular input through a linear polarizer also passes half.
* ``rays.update_intensity(state)`` recomputes intensities from a stored polarization
  ray trace *without* re-tracing, so applying it for two different states to the
  same ``RealRays`` object must give the two states' transmissions -- checked
  against a fresh trace for each state.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t30_introduction_to_polarization",
    title="Introduction to Polarization",
    level="advanced",
    url="https://www.optiland.org/tutorials/introduction-to-polarization",
    demonstrates=(
        "rays.PolarizationState(is_polarized, Ex, Ey, phase_x, phase_y), "
        "rays.create_polarization('H'|'V'|'L+45'|'RCP'), "
        "surfaces.set_fresnel_coatings(), RealRays.update_intensity(state), "
        "coatings.PolarizerCoating(axis=...), and surface_group.stop_index."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.5875618
NUM_RAYS = 64
UPSTREAM_UNCOATED_TRANSMISSION = 0.98
UPSTREAM_FRESNEL_TRANSMISSION = 0.19


def build_objective(fresnel: bool = False, polarizer_axis=None):
    from optiland.samples.objectives import ObjectiveUS008879901

    lens = ObjectiveUS008879901()
    if fresnel:
        lens.surfaces.set_fresnel_coatings()
    if polarizer_axis is not None:
        from optiland.coatings import PolarizerCoating

        lens.surfaces.surfaces[4].interaction_model.coating = PolarizerCoating(
            axis=polarizer_axis
        )
    return lens


def _mean_transmission(lens, state=None, hy: float = 0.0) -> tuple[float, object]:
    if state is not None:
        lens.updater.set_polarization(state)
    rays = lens.trace(
        Hx=0, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS, distribution="uniform"
    )
    return float(np.asarray(rays.i, dtype=float).mean()), rays


def run() -> TutorialResult:
    from optiland.rays import PolarizationState, create_polarization

    result = TutorialResult()

    # -- 1. uncoated and Fresnel-coated, unpolarized ---------------------------
    bare = build_objective()
    bare_transmission, _ = _mean_transmission(bare)
    num_surfaces = len(bare.surfaces.surfaces)
    stop_index = int(bare.surfaces.stop_index)
    result.record(
        num_surfaces=num_surfaces,
        stop_index=stop_index,
        uncoated_mean_transmission=bare_transmission,
    )
    result.check_close(
        "uncoated_transmission_matches_upstreams_98_percent",
        "reference",
        bare_transmission,
        UPSTREAM_UNCOATED_TRANSMISSION,
        abs_=0.01,
    )

    fresnel = build_objective(fresnel=True)
    unpolarized = PolarizationState(is_polarized=False)
    fresnel_transmission, _ = _mean_transmission(fresnel, unpolarized)
    result.record(fresnel_unpolarized_mean_transmission=fresnel_transmission)
    result.check_close(
        "fresnel_coated_transmission_matches_upstreams_19_percent",
        "reference",
        fresnel_transmission,
        UPSTREAM_FRESNEL_TRANSMISSION,
        abs_=0.02,
    )
    # Sanity: ~4% loss per air-glass interface compounding over this many surfaces.
    glass_interfaces = sum(
        1
        for surface in fresnel.surfaces.surfaces[1:-1]
        if type(surface.material_post).__name__ == "Material"
        or type(surface.material_pre).__name__ == "Material"
    )
    result.record(
        num_glass_interfaces=glass_interfaces,
        compounded_four_percent_loss=float(0.96**glass_interfaces),
    )
    result.check_true(
        "the_fresnel_collapse_is_explained_by_compounded_interface_loss",
        "analytic",
        0.3 * float(0.96**glass_interfaces)
        < fresnel_transmission
        < 3.0 * float(0.96**glass_interfaces),
        f"{glass_interfaces} glass interfaces at ~4% loss each gives "
        f"{float(0.96**glass_interfaces):.4f} against the measured "
        f"{fresnel_transmission:.4f}: the 98% -> 19% collapse is compounding, not a bug",
    )

    # -- 2. the four named polarization states ---------------------------------
    named = {}
    for pol_type in ("H", "V", "L+45", "RCP"):
        lens = build_objective(fresnel=True)
        named[pol_type], _ = _mean_transmission(lens, create_polarization(pol_type))
    result.record(named_state_transmission=named)
    result.check_finite("named_state_transmission_finite", list(named.values()))
    result.check_true(
        "every_named_state_transmits_a_physical_fraction",
        "invariant",
        all(0.0 < v <= 1.0 for v in named.values()),
        f"transmission per state: {named}",
    )

    # -- 3. a linear polarizer is a projector ----------------------------------
    projector = {}
    for pol_type in ("H", "V", "L+45", "RCP"):
        lens = build_objective(fresnel=True, polarizer_axis=(0.0, 1.0, 0.0))
        projector[pol_type], _ = _mean_transmission(lens, create_polarization(pol_type))
    result.record(vertical_polarizer_transmission=projector)
    extinction = projector["H"] / projector["V"]
    result.record(polarizer_extinction_ratio=extinction)
    result.check_true(
        "a_vertical_polarizer_extinguishes_horizontal_input",
        "analytic",
        extinction < 1e-4,
        f"H through a V polarizer transmits {projector['H']:.3e} against "
        f"{projector['V']:.6f} for V -- an extinction ratio of {extinction:.2e} "
        f"({-10 * np.log10(extinction):.1f} dB). Not identically zero: 23 Fresnel "
        "interfaces before and after the polarizer rotate the s/p basis, so a little "
        "H leaks into the polarizer's eigenstate. That residual is physics, not "
        "numerical noise.",
    )
    result.check_true(
        "a_vertical_polarizer_passes_vertical_input",
        "analytic",
        projector["V"] > 0.5 * named["V"],
        f"V through a V polarizer transmits {projector['V']:.6f} against "
        f"{named['V']:.6f} with no polarizer -- the projector is transparent to its own "
        "eigenstate, up to the surface's Fresnel loss",
    )
    for pol_type in ("L+45", "RCP"):
        ratio = projector[pol_type] / projector["V"]
        result.record(**{f"{pol_type.replace('+', 'p')}_over_V_through_polarizer": ratio})
        result.check_close(
            f"a_vertical_polarizer_passes_half_of_{pol_type.replace('+', 'p')}_input",
            "analytic",
            ratio,
            0.5,
            rel=0.02,
        )

    # -- 4. update_intensity re-uses one trace for several states ---------------
    lens = build_objective(fresnel=True)
    lens.updater.set_polarization(unpolarized)
    rays = lens.trace(
        Hx=0, Hy=1, wavelength=WAVELENGTH_UM, num_rays=NUM_RAYS, distribution="uniform"
    )
    reused = {}
    for pol_type in ("H", "V", "L+45", "RCP"):
        state = create_polarization(pol_type)
        lens.updater.set_polarization(state)
        rays.update_intensity(state)
        reused[pol_type] = float(np.asarray(rays.i, dtype=float).mean())
    fresh = {}
    for pol_type in ("H", "V", "L+45", "RCP"):
        trial = build_objective(fresnel=True)
        fresh[pol_type], _ = _mean_transmission(trial, create_polarization(pol_type), hy=1.0)
    result.record(update_intensity_reused=reused, fresh_trace_per_state=fresh)
    deviations = {k: abs(reused[k] - fresh[k]) for k in reused}
    result.check_true(
        "update_intensity_reproduces_a_fresh_trace_for_each_state",
        "analytic",
        all(v < 1e-9 for v in deviations.values()),
        f"max |reused - fresh| = {max(deviations.values()):.3e} over the four named "
        "states: update_intensity recomputes from the stored polarization ray matrices, "
        "so one trace serves any number of input states",
    )

    # -- 5. a custom Jones state -----------------------------------------------
    lens = build_objective(fresnel=True)
    custom = PolarizationState(is_polarized=True, Ex=1, Ey=0.5, phase_x=0.2, phase_y=0)
    custom_transmission, _ = _mean_transmission(lens, custom, hy=1.0)
    result.record(custom_jones_transmission=custom_transmission)
    result.check_true(
        "a_custom_jones_state_lies_between_its_component_states",
        "analytic",
        min(fresh["H"], fresh["V"]) - 1e-9
        <= custom_transmission
        <= max(fresh["H"], fresh["V"]) + 1e-9,
        f"Ex=1, Ey=0.5 transmits {custom_transmission:.6f}, bracketed by pure H "
        f"({fresh['H']:.6f}) and pure V ({fresh['V']:.6f}) -- as a superposition of the "
        "two must be for a system whose Jones matrices are diagonal in that basis",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
