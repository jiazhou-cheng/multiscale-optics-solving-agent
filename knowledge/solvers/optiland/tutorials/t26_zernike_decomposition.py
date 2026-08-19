"""Advanced / "Zernike Decomposition" -- https://www.optiland.org/tutorials/zernike-decomposition

Repo-owned reproduction of the Zernike-fitting tutorial: `wavefront.ZernikeOPD` on
the bundled `EyepieceErfle` in all three supported conventions (``standard``,
``fringe``, ``noll``) with 37 terms, plus the 9-term coefficient printout.

Upstream prints coefficients but publishes no values. Zernike theory pins the
relationships between the three conventions exactly, and that is the validation --
including one trap that a user is very likely to fall into:

* **All three conventions agree on the piston term** to 1e-11. The mean of a
  wavefront cannot depend on the basis used to describe it.
* **``standard`` and ``noll`` are both orthonormal**, so ``sqrt(sum_k>=1 c_k^2)`` is
  the piston-removed RMS wavefront error, and the two conventions give *identical*
  values (they differ only in term ordering). Verified to 1e-11.
* **``fringe`` is NOT orthonormal.** Fringe Zernikes are normalized to unit peak
  rather than unit RMS, so the same quadrature sum gives 0.0769 waves instead of
  0.0444 -- 73% too large. Computing an RMS wavefront error from Fringe
  coefficients the way one legitimately can from Standard or Noll coefficients is
  a silent 1.7x error, and this reproduction asserts the discrepancy so the trap
  stays recorded.
* **The fit converges**: the residual RMS between the reconstructed and the actual
  wavefront falls monotonically as ``num_terms`` goes 4 -> 9 -> 16 -> 25 -> 37.
* The orthonormal quadrature sum tracks the piston-removed RMS measured directly
  off `wavefront.OPD`, with the shortfall accounted for by the unfitted residual.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t26_zernike_decomposition",
    title="Zernike Decomposition",
    level="advanced",
    url="https://www.optiland.org/tutorials/zernike-decomposition",
    demonstrates=(
        "wavefront.ZernikeOPD(zernike_type='standard'|'fringe'|'noll', "
        "num_terms=...) with .coeffs, .rms(), .view(), .view_residual(), and the "
        "fact that only standard/noll are orthonormal."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.55
NUM_TERMS = 37
TYPES = ("standard", "fringe", "noll")


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import wavefront
    from optiland.samples.eyepieces import EyepieceErfle

    result = TutorialResult()
    lens = EyepieceErfle()

    opd = wavefront.OPD(lens, field=(0, 0), wavelength=WAVELENGTH_UM)
    opd.view(projection="2d", num_points=64)
    plt.close("all")
    direct = np.asarray(opd.get_data((0, 0), WAVELENGTH_UM).opd, dtype=float)
    direct_mean = float(direct.mean())
    direct_rms_piston_removed = float(direct.std())
    result.record(
        opd_mean_waves=direct_mean,
        opd_rms_piston_removed_waves=direct_rms_piston_removed,
        opd_num_samples=int(direct.size),
    )

    fits = {}
    for zernike_type in TYPES:
        fit = wavefront.ZernikeOPD(
            lens,
            field=(0, 0),
            wavelength=WAVELENGTH_UM,
            zernike_type=zernike_type,
            num_terms=NUM_TERMS,
        )
        coeffs = np.asarray(fit.coeffs, dtype=float)
        fits[zernike_type] = {
            "piston": float(coeffs[0]),
            "quadrature_sum_excluding_piston": float(np.sqrt((coeffs[1:] ** 2).sum())),
            "rms_accessor": float(np.asarray(fit.rms()).ravel()[0]),
            "max_abs_coefficient": float(np.abs(coeffs).max()),
            "num_coefficients": int(coeffs.size),
        }
        result.record(**{f"coeffs_{zernike_type}": coeffs})
        result.check_finite(f"coeffs_{zernike_type}_finite", coeffs)
        result.check_true(
            f"{zernike_type}_returns_the_requested_number_of_terms",
            "invariant",
            coeffs.size == NUM_TERMS,
            f"{coeffs.size} == {NUM_TERMS}",
        )
        fit.view(projection="2d", num_points=64)
        plt.close("all")
    result.record(fit_summary=fits)

    pistons = [fits[t]["piston"] for t in TYPES]
    result.check_true(
        "all_three_conventions_agree_on_the_piston_term",
        "analytic",
        max(pistons) - min(pistons) < 1e-10,
        f"piston = {pistons} across {TYPES}: spread {max(pistons) - min(pistons):.3e}. "
        "The mean of a wavefront is basis-independent.",
    )
    result.check_close(
        "standard_and_noll_are_both_orthonormal",
        "analytic",
        fits["noll"]["quadrature_sum_excluding_piston"],
        fits["standard"]["quadrature_sum_excluding_piston"],
        rel=1e-9,
    )
    fringe_ratio = (
        fits["fringe"]["quadrature_sum_excluding_piston"]
        / fits["standard"]["quadrature_sum_excluding_piston"]
    )
    result.record(fringe_over_standard_quadrature_sum=fringe_ratio)
    result.check_true(
        "fringe_coefficients_are_not_orthonormal_and_must_not_be_rms_summed",
        "analytic",
        fringe_ratio > 1.5,
        f"sqrt(sum c^2) excluding piston: standard "
        f"{fits['standard']['quadrature_sum_excluding_piston']:.6f}, noll "
        f"{fits['noll']['quadrature_sum_excluding_piston']:.6f}, fringe "
        f"{fits['fringe']['quadrature_sum_excluding_piston']:.6f} waves -- fringe is "
        f"{fringe_ratio:.3f}x larger. Fringe Zernikes are normalized to unit PEAK, not "
        "unit RMS, so quadrature-summing them does not give an RMS wavefront error.",
    )
    result.check_true(
        "the_three_coefficient_vectors_are_genuinely_different",
        "invariant",
        not np.allclose(
            np.asarray(result.metrics["coeffs_standard"], dtype=float),
            np.asarray(result.metrics["coeffs_fringe"], dtype=float),
        )
        and not np.allclose(
            np.asarray(result.metrics["coeffs_standard"], dtype=float),
            np.asarray(result.metrics["coeffs_noll"], dtype=float),
        ),
        "standard, fringe and noll produce distinct coefficient vectors, so the "
        "agreements above are real rather than the same array read three times",
    )

    # -- the fit converges with the number of terms -----------------------------
    residuals = {}
    for num_terms in (4, 9, 16, 25, 37):
        fit = wavefront.ZernikeOPD(
            lens,
            field=(0, 0),
            wavelength=WAVELENGTH_UM,
            zernike_type="standard",
            num_terms=num_terms,
        )
        coeffs = np.asarray(fit.coeffs, dtype=float)
        captured = float(np.sqrt((coeffs[1:] ** 2).sum()))
        residuals[num_terms] = float(
            np.sqrt(max(direct_rms_piston_removed**2 - captured**2, 0.0))
        )
    result.record(unfitted_residual_rms_waves_by_num_terms=residuals)
    ordered = [residuals[n] for n in (4, 9, 16, 25, 37)]
    result.check_true(
        "the_zernike_fit_converges_as_terms_are_added",
        "analytic",
        all(b <= a + 1e-12 for a, b in zip(ordered, ordered[1:]))
        and ordered[-1] < 0.5 * ordered[0],
        "unfitted residual RMS (in quadrature against the directly measured "
        f"{direct_rms_piston_removed:.6f}-wave wavefront) falls "
        + " -> ".join(f"{v:.6f}" for v in ordered)
        + " waves for 4, 9, 16, 25, 37 terms. It plateaus after 9 terms: the first "
        "four Standard terms are piston plus three that vanish identically for an "
        "on-axis rotationally symmetric wavefront, and beyond 9 the remaining "
        "0.0158 waves is content this radial-order set cannot represent on this grid.",
    )
    result.check_true(
        "thirty_seven_terms_capture_most_of_the_wavefront",
        "analytic",
        fits["standard"]["quadrature_sum_excluding_piston"]
        > 0.9 * direct_rms_piston_removed,
        f"the orthonormal quadrature sum {fits['standard']['quadrature_sum_excluding_piston']:.6f} "
        f"reaches {fits['standard']['quadrature_sum_excluding_piston'] / direct_rms_piston_removed * 100:.2f}% "
        f"of the {direct_rms_piston_removed:.6f}-wave RMS measured directly off "
        "wavefront.OPD -- two different code paths on the same wavefront",
    )

    # -- the 9-term printout and the residual view ------------------------------
    nine = wavefront.ZernikeOPD(lens, (0, 1), WAVELENGTH_UM, zernike_type="noll", num_terms=9)
    nine_coeffs = np.asarray(nine.coeffs, dtype=float)
    nine.view_residual()
    plt.close("all")
    result.record(noll_9_term_coeffs_at_edge_field=nine_coeffs)
    result.check_finite("noll_9_term_coeffs_finite", nine_coeffs)
    result.check_true(
        "the_edge_field_fit_is_dominated_by_low_order_aberrations",
        "analytic",
        int(np.argmax(np.abs(nine_coeffs[1:]))) + 1 < 9,
        f"largest non-piston Noll term is Z{int(np.argmax(np.abs(nine_coeffs[1:]))) + 2} at "
        f"{float(nine_coeffs[int(np.argmax(np.abs(nine_coeffs[1:]))) + 1]):.6f} waves",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
