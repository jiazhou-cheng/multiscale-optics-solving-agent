"""Intermediate / "Introduction to Coatings" -- https://www.optiland.org/tutorials/introduction-to-coatings

Repo-owned reproduction of the coating-basics tutorial: the same cemented doublet
traced three ways -- uncoated, with four ``coatings.SimpleCoating`` surfaces of
declared transmittance, and with the string form ``coating="fresnel"`` plus an
unpolarized ``PolarizationState`` -- comparing mean ``rays.i``.

Upstream prints three averages but publishes none of them. Two of the three are
nonetheless exactly predictable, which makes this the sharpest available check on
what ``rays.i`` means:

* **Uncoated**: ``rays.i`` is 0.999784 on average -- close to, but *not* exactly,
  unity, with a ray-dependent spread of ~3e-5. Optiland does not apply Fresnel
  losses to an uncoated surface, but ``rays.i == 1`` is nonetheless not a safe
  uncoated baseline, so the uncoated trace is measured rather than assumed.
* **SimpleCoating**: the four surfaces declare transmittance 0.5, 0.6, 0.8, 0.9, and
  ``rays.i(coated) / rays.i(uncoated)`` equals the product ``0.216`` for **every**
  ray to 1e-16. That proves ``SimpleCoating.transmittance`` is a scalar *intensity*
  factor applied multiplicatively per surface, with no angle or wavelength
  dependence at all.
* **coating="fresnel"**: mean 0.8087, consistent with t07's finding that ``rays.i``
  accumulates ``|t|^2`` and that the factors cancel over the four
  air-glass/glass-air interfaces to give ``prod(1 - R_i)``. Checked against the
  normal-incidence Fresnel product computed here from the two glasses' indices.

The tutorial's ``self.updater.set_polarization(state)`` form is used: the
``Optic.set_polarization`` alias is deprecated for removal in 0.7.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t18_introduction_to_coatings",
    title="Introduction to Coatings",
    level="intermediate",
    url="https://www.optiland.org/tutorials/introduction-to-coatings",
    demonstrates=(
        "coatings.SimpleCoating(transmittance, reflectance), the string form "
        "coating='fresnel', Optic.updater.set_polarization, and rays.i as a "
        "multiplicative per-surface intensity factor."
    ),
    slow=True,
)

TRACE_WAVELENGTH_UM = 0.55
SIMPLE_TRANSMITTANCES = (0.5, 0.6, 0.8, 0.9)


def build_doublet(coatings_per_surface=None, polarized_state: bool = False):
    from optiland import optic
    from optiland.rays import PolarizationState

    coats = coatings_per_surface or [None] * 4
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
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
    lens.image_solve()
    if polarized_state:
        lens.updater.set_polarization(PolarizationState(is_polarized=False))
    return lens


def _mean_intensity(lens) -> np.ndarray:
    rays = lens.trace(
        Hx=0, Hy=0, wavelength=TRACE_WAVELENGTH_UM, num_rays=256, distribution="uniform"
    )
    return np.asarray(rays.i, dtype=float).ravel()


def run() -> TutorialResult:
    import warnings

    from optiland import coatings
    from optiland.materials import Material

    result = TutorialResult()

    uncoated = _mean_intensity(build_doublet())
    simple_coats = [
        coatings.SimpleCoating(transmittance=t, reflectance=1.0 - t)
        for t in SIMPLE_TRANSMITTANCES
    ]
    simple = _mean_intensity(build_doublet(simple_coats))
    fresnel = _mean_intensity(build_doublet(["fresnel"] * 4, polarized_state=True))

    for label, values in (("uncoated", uncoated), ("simple", simple), ("fresnel", fresnel)):
        result.record(
            **{
                f"{label}_num_rays": int(values.size),
                f"{label}_mean_intensity": float(values.mean()),
                f"{label}_min_intensity": float(values.min()),
                f"{label}_max_intensity": float(values.max()),
            }
        )
        result.check_finite(f"{label}_intensity_finite", values)

    result.record(uncoated_intensity_deficit=float(1.0 - uncoated.mean()))
    result.check_true(
        "an_uncoated_doublet_is_close_to_but_not_exactly_unit_transmittance",
        "invariant",
        0.0 < 1.0 - float(uncoated.mean()) < 1e-3 and float(uncoated.min()) > 0.99,
        f"rays.i = {float(uncoated.mean()):.9f} on average with NO coating anywhere, "
        f"spanning [{float(uncoated.min()):.9f}, {float(uncoated.max()):.9f}]. There is a "
        "small ray-dependent factor of order 2e-4 even on an uncoated system, so "
        "`rays.i == 1` is not a safe uncoated baseline -- the uncoated trace has to be "
        "measured, which is what the ratio test below does.",
    )

    expected_simple = float(np.prod(SIMPLE_TRANSMITTANCES))
    ratio = simple / uncoated
    result.record(
        expected_simple_product=expected_simple,
        simple_over_uncoated_ratio_min=float(ratio.min()),
        simple_over_uncoated_ratio_max=float(ratio.max()),
    )
    result.check_true(
        "simple_coating_transmittances_multiply_exactly_over_the_uncoated_baseline",
        "analytic",
        float(np.max(np.abs(ratio - expected_simple))) < 1e-12,
        f"rays.i(SimpleCoating) / rays.i(uncoated) = {float(ratio.mean()):.15f} for every "
        f"one of {ratio.size} rays, against the declared product "
        f"0.5*0.6*0.8*0.9 = {expected_simple}: max deviation "
        f"{float(np.max(np.abs(ratio - expected_simple))):.3e}",
    )
    result.check_true(
        "simple_coating_has_no_angle_or_wavelength_dependence",
        "analytic",
        float(ratio.max() - ratio.min()) < 1e-12,
        f"the ratio spans {float(ratio.max() - ratio.min()):.3e} across the whole pupil: "
        "SimpleCoating is a scalar intensity factor, not a physical model",
    )

    # -- the Fresnel case, against an independent normal-incidence product ------
    bk7 = Material("N-BK7")
    sf2 = Material("SF2", reference="schott")
    n_bk7 = float(np.asarray(bk7.n(TRACE_WAVELENGTH_UM)).ravel()[0])
    n_sf2 = float(np.asarray(sf2.n(TRACE_WAVELENGTH_UM)).ravel()[0])

    def _fresnel_T(n1: float, n2: float) -> float:
        return 1.0 - ((n1 - n2) / (n1 + n2)) ** 2

    predicted_fresnel = (
        _fresnel_T(1.0, n_bk7)
        * _fresnel_T(n_bk7, 1.0)
        * _fresnel_T(1.0, n_sf2)
        * _fresnel_T(n_sf2, 1.0)
    )
    result.record(
        n_bk7_at_550nm=n_bk7,
        n_sf2_at_550nm=n_sf2,
        predicted_fresnel_transmission=predicted_fresnel,
    )
    result.check_close(
        "fresnel_coated_transmission_matches_the_four_interface_product",
        "analytic",
        float(fresnel.mean()),
        predicted_fresnel,
        rel=0.02,
    )
    result.check_true(
        "fresnel_transmission_is_between_the_uncoated_and_simple_cases",
        "invariant",
        expected_simple < float(fresnel.mean()) < 1.0,
        f"{expected_simple:.4f} (declared SimpleCoating product) < "
        f"{float(fresnel.mean()):.4f} (real Fresnel) < 1.0 (uncoated)",
    )

    # -- the deprecated alias -------------------------------------------------
    lens = build_doublet(["fresnel"] * 4)
    from optiland.rays import PolarizationState

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        lens.set_polarization(PolarizationState(is_polarized=False))
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    result.record(set_polarization_deprecation=messages)
    result.check_true(
        "optic_set_polarization_is_deprecated_in_favour_of_updater",
        "invariant",
        any("updater" in m for m in messages),
        f"{messages or 'no DeprecationWarning'}",
    )
    alias_intensity = _mean_intensity(lens)
    result.check_close(
        "the_deprecated_alias_produces_identical_numbers",
        "invariant",
        float(alias_intensity.mean()),
        float(fresnel.mean()),
        rel=0.0,
        abs_=0.0,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
