"""Intermediate / "1st & 3rd Order Aberrations" -- https://www.optiland.org/tutorials/first-third-order-aberrations

Repo-owned reproduction of the Seidel-sum tutorial: read the five primary Seidel
sums and the twelve named third-order/chromatic coefficients off
`Optic.aberrations` for the bundled `TripletTelescopeObjective`.

Upstream prints all thirteen quantities and publishes none of them. What makes
them checkable is that third-order aberration theory fixes the *relationships*
between them, and none of those relationships is used to compute the others:

* ``TCC == 3 * CC`` -- tangential coma is exactly three times sagittal coma.
  Verified per surface to float64 round-off.
* Every longitudinal coefficient is its transverse partner divided by the final
  marginal ray angle: ``SC = -TSC/u'``, ``AC = -TAC/u'``, ``PC = -TPC/u'`` and
  ``LchC = -TAchC/u'``, with the single ``u'`` read from
  ``paraxial.marginal_ray()``. All four ratios come out equal to the same
  5.6001, per surface, across four physically unrelated aberration types --
  which is a much stronger statement than any one of them alone.
* The coefficients are **per surface** (6 entries for this 6-surface system), so
  the system aberration is their sum; the tutorial's ``for k, seidel in
  enumerate(...)`` loop is iterating surfaces for the Seidel sums but the named
  accessors return per-surface arrays too. That indexing is recorded explicitly.
* ``sum(PC) == -h_img^2 * P / 2``, where ``P = sum (n' - n)/(n n' R)`` is the
  Petzval sum read straight off the prescription here, without Optiland. This
  simultaneously pins what Optiland's ``PC`` *means* (edge-field longitudinal sag)
  and that its value is right.
* ``sum(LchC)`` is 0.4% of the focal length: this objective is achromatized by
  construction, so the residual must be *small* rather than a particular sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t14_first_third_order_aberrations",
    title="1st & 3rd Order Aberrations",
    level="intermediate",
    url="https://www.optiland.org/tutorials/first-third-order-aberrations",
    demonstrates=(
        "Optic.aberrations.seidels() and the named third-order coefficients "
        "TSC/SC/CC/TCC/TAC/AC/TPC/PC/DC/TAchC/LchC/TchC, all returned as "
        "per-surface arrays rather than system scalars."
    ),
)

NAMED = (
    "TSC", "SC", "CC", "TCC", "TAC", "AC", "TPC", "PC", "DC", "TAchC", "LchC", "TchC",
)


def run() -> TutorialResult:
    from optiland.samples.objectives import TripletTelescopeObjective

    result = TutorialResult()
    lens = TripletTelescopeObjective()
    ab = lens.aberrations
    num_surfaces = len(lens.surfaces.surfaces)

    seidels = np.asarray(ab.seidels(), dtype=float).ravel()
    coeffs = {name: np.asarray(getattr(ab, name)(), dtype=float).ravel() for name in NAMED}
    result.record(
        num_surfaces=num_surfaces,
        num_seidel_sums=int(seidels.size),
        seidels=seidels,
        **{f"coeff_{name}": arr for name, arr in coeffs.items()},
    )
    result.record(**{f"sum_{name}": float(arr.sum()) for name, arr in coeffs.items()})

    result.check_finite("seidels_finite", seidels)
    result.check_finite(
        "named_coefficients_finite", np.concatenate([arr for arr in coeffs.values()])
    )
    result.check_true(
        "there_are_five_primary_seidel_sums",
        "analytic",
        seidels.size == 5,
        f"seidels() returns {seidels.size} values (S1 spherical, S2 coma, S3 "
        "astigmatism, S4 Petzval, S5 distortion)",
    )
    lengths = {name: int(arr.size) for name, arr in coeffs.items()}
    result.record(coefficient_lengths=lengths)
    result.check_true(
        "named_coefficients_are_per_surface_not_per_system",
        "invariant",
        len(set(lengths.values())) == 1,
        f"every named accessor returns {next(iter(set(lengths.values())))} values for a "
        f"{num_surfaces}-surface system: {lengths}. The system aberration is their sum.",
    )

    # -- TCC = 3 * CC ---------------------------------------------------------
    tcc_over_cc = coeffs["TCC"] / coeffs["CC"]
    result.record(tcc_over_cc=tcc_over_cc)
    result.check_true(
        "tangential_coma_is_three_times_sagittal_coma",
        "analytic",
        bool(np.max(np.abs(tcc_over_cc - 3.0)) < 1e-12),
        f"TCC/CC = {tcc_over_cc} per surface; max deviation from 3 is "
        f"{float(np.max(np.abs(tcc_over_cc - 3.0))):.3e}",
    )

    # -- longitudinal = -transverse / u' --------------------------------------
    _, marginal_u = lens.paraxial.marginal_ray()
    u_final = float(np.asarray(marginal_u, dtype=float).ravel()[-1])
    result.record(final_marginal_ray_slope=u_final)
    pairs = (("SC", "TSC"), ("AC", "TAC"), ("PC", "TPC"), ("LchC", "TAchC"))
    ratios = {}
    for longitudinal, transverse in pairs:
        ratio = coeffs[longitudinal] / coeffs[transverse]
        ratios[f"{longitudinal}_over_{transverse}"] = ratio
        result.check_true(
            f"{longitudinal}_is_{transverse}_over_minus_final_marginal_slope",
            "analytic",
            bool(np.max(np.abs(ratio - (-1.0 / u_final))) < 1e-9),
            f"{longitudinal}/{transverse} = {ratio.mean():.6f} per surface vs "
            f"-1/u' = {-1.0 / u_final:.6f} (max deviation "
            f"{float(np.max(np.abs(ratio - (-1.0 / u_final)))):.3e})",
        )
    result.record(longitudinal_over_transverse_ratios=ratios)
    all_ratios = np.concatenate(list(ratios.values()))
    result.check_true(
        "the_same_conversion_factor_governs_four_unrelated_aberration_types",
        "analytic",
        bool(np.max(all_ratios) - np.min(all_ratios) < 1e-9),
        f"all {all_ratios.size} longitudinal/transverse ratios across spherical, "
        f"astigmatism, Petzval and axial colour agree to "
        f"{float(np.max(all_ratios) - np.min(all_ratios)):.3e}, at "
        f"{float(all_ratios.mean()):.6f}",
    )

    # -- Petzval sum, independently of Optiland -------------------------------
    # P = sum over surfaces of (n' - n) / (n n' R). Read the prescription only.
    primary_wl = float(np.asarray(lens.primary_wavelength).ravel()[0])
    petzval = 0.0
    terms = []
    for surface in lens.surfaces.surfaces[1:-1]:
        radius = float(np.asarray(surface.geometry.radius).ravel()[0])
        if not np.isfinite(radius):
            terms.append(0.0)
            continue
        n_pre = float(np.asarray(surface.material_pre.n(primary_wl)).ravel()[0])
        n_post = float(np.asarray(surface.material_post.n(primary_wl)).ravel()[0])
        term = (n_post - n_pre) / (n_pre * n_post * radius)
        terms.append(term)
        petzval += term
    efl = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    chief_y, _ = lens.paraxial.chief_ray()
    image_height = float(np.asarray(chief_y, dtype=float).ravel()[-1])
    # Optiland's longitudinal Petzval coefficient is the field sag at the edge
    # field: PC = -h_img^2 * P / 2, with P the prescription Petzval sum above.
    predicted_pc = -(image_height**2) * petzval / 2.0
    result.record(
        petzval_sum_per_surface=terms,
        petzval_sum_closed_form_per_mm=petzval,
        petzval_radius_mm=(1.0 / petzval) if petzval else float("inf"),
        efl_mm=efl,
        paraxial_image_height_mm=image_height,
        optiland_PC_sum=float(coeffs["PC"].sum()),
        predicted_PC_sum_from_petzval=predicted_pc,
    )
    result.check_close(
        "optiland_petzval_coefficient_matches_the_prescription_petzval_sum",
        "analytic",
        float(coeffs["PC"].sum()),
        predicted_pc,
        rel=1e-3,
    )
    result.check_true(
        "petzval_radius_is_positive_and_of_order_the_focal_length",
        "analytic",
        0.5 * efl < 1.0 / petzval < 5.0 * efl,
        f"Petzval radius 1/P = {1.0 / petzval:.3f} mm = {1.0 / (petzval * efl):.3f} x the "
        f"{efl:.3f} mm EFL. Positive, i.e. the field curves inward, as an all-positive-"
        "power uncorrected triplet must, and on the scale of the focal length rather "
        "than orders away from it.",
    )

    # -- axial colour: this objective is deliberately achromatized -------------
    lchc_sum = float(coeffs["LchC"].sum())
    result.record(lchc_sum_over_efl=lchc_sum / efl)
    result.check_true(
        "axial_colour_is_a_small_residual_of_the_focal_length",
        "analytic",
        abs(lchc_sum / efl) < 0.02,
        f"sum(LchC) = {lchc_sum:.6f} mm is {abs(lchc_sum / efl) * 100:.3f}% of the "
        f"{efl:.3f} mm focal length -- a telescope objective is achromatized by "
        "construction, so the residual must be small rather than a particular sign",
    )
    # A single uncorrected element of the same power must be far worse, which is
    # what makes the "small" above meaningful rather than an unfalsifiable claim.
    result.note(
        "The comparison against an uncorrected singlet of the same power lives in "
        "t15_chromatic_aberrations, where upstream publishes both numbers."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
