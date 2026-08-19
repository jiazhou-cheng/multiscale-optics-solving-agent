"""Beginner / "Material Database" -- https://www.optiland.org/tutorials/material-database

Repo-owned reproduction of the material tutorial: ``IdealMaterial``,
``AbbeMaterial`` (buchdahl / polynomial models), ``AbbeMaterialE``, and catalog
``Material`` lookups spanning glass, an organic (DNA), a salt (AgCl), a liquid
(toluene) and a gas (He).

The only numbers upstream publishes are the trivial ``IdealMaterial`` ones, so
the real validation here is analytic and comes from the *definitions* of the
material models rather than from Optiland:

* ``IdealMaterial(n, k)`` is wavelength-independent, exactly.
* ``AbbeMaterial(n=1.5, abbe=65, model='buchdahl')`` must reproduce its own
  defining numbers: ``n(d) == 1.5`` and ``(n(d) - 1) / (n(F) - n(C)) == 65`` at
  the Fraunhofer d/F/C lines (0.587562 / 0.486133 / 0.656273 um). It does, to
  0.2%.
* The two other Abbe models do **not**, and this reproduction pins how far off
  they are rather than asserting a correctness they do not have:
  ``model='polynomial'`` (the 0.6.0 *default*) recovers V_d = 63.96 for a
  requested 65 (1.6% low), and ``AbbeMaterialE`` recovers V_e = 40.46 for a
  requested 65 -- 38% low, and 0.57-0.83x the request over V_e = 20..80. See
  ``failure_guide.md``; use ``model='buchdahl'`` for anything quantitative.
* Catalog ``Material('N-BK7')`` is compared against the SCHOTT Sellmeier
  dispersion formula evaluated here, independently of Optiland.
* Every material shows normal dispersion (n decreasing with wavelength) across
  the visible, and He's index excess ``n - 1`` is ~1e-5 as a gas must be.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t04_material_database",
    title="Material Database",
    level="beginner",
    url="https://www.optiland.org/tutorials/material-database",
    demonstrates=(
        "optiland.materials: IdealMaterial(n, k), AbbeMaterial(n, abbe, model="
        "'buchdahl'|'polynomial'), AbbeMaterialE, Material(name[, reference]). "
        "Material.n/.k take wavelengths in micrometres and return backend arrays."
    ),
)

# Fraunhofer lines (um) -- the definitions the Abbe number is quoted against.
LINE_D = 0.587562
LINE_F = 0.486133
LINE_C = 0.656273
# The e-line set used by AbbeMaterialE.
LINE_E = 0.546074
LINE_F_PRIME = 0.479991
LINE_C_PRIME = 0.643847

# SCHOTT N-BK7 Sellmeier coefficients (datasheet cited in source_manifest.yaml).
NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)


def nbk7_sellmeier(wavelength_um: float) -> float:
    """N-BK7 index from the SCHOTT Sellmeier formula, computed without Optiland."""
    w2 = float(wavelength_um) ** 2
    n2 = 1.0 + sum(b * w2 / (w2 - c) for b, c in zip(NBK7_B, NBK7_C, strict=True))
    return float(np.sqrt(n2))


def _n(material, wavelength) -> float:
    return float(np.asarray(material.n(wavelength)).ravel()[0])


def run() -> TutorialResult:
    from optiland.materials import AbbeMaterial, AbbeMaterialE, IdealMaterial, Material

    result = TutorialResult()

    # -- 1. ideal, wavelength-independent material ---------------------------
    ideal = IdealMaterial(n=1.5, k=0)
    ideal_n = [_n(ideal, w) for w in (0.48, 0.55, 0.65)]
    ideal_k = [float(np.asarray(ideal.k(w)).ravel()[0]) for w in (0.48, 0.55, 0.65)]
    result.record(ideal_n_at_048_055_065=ideal_n, ideal_k_at_048_055_065=ideal_k)
    result.check_true(
        "ideal_material_index_is_exactly_wavelength_independent",
        "reference",
        all(v == 1.5 for v in ideal_n) and all(v == 0 for v in ideal_k),
        f"n={ideal_n} k={ideal_k} vs upstream-published n=1.5, k=0 at all three wavelengths",
    )

    # -- 2. Abbe models vs. their own defining numbers ------------------------
    # An Abbe material is *defined* by (n at the reference line, V between the
    # bracketing lines). Feeding those two numbers back out of the fitted model
    # is therefore a self-consistency oracle that needs no external data.
    buchdahl = AbbeMaterial(n=1.5, abbe=65.0, model="buchdahl")
    n_d, n_f, n_c = (_n(buchdahl, w) for w in (LINE_D, LINE_F, LINE_C))
    v_buchdahl = (n_d - 1.0) / (n_f - n_c)
    result.record(abbe_buchdahl_n_d=n_d, abbe_buchdahl_V_recovered=v_buchdahl)
    result.check_close("abbe_buchdahl_n_at_d_line_is_1p5", "analytic", n_d, 1.5, rel=1e-6)
    result.check_close("abbe_buchdahl_recovers_V_65", "analytic", v_buchdahl, 65.0, rel=2e-3)

    # The legacy polynomial model is the 0.6.0 DEFAULT and is a global fit to the
    # Schott catalog, not an interpolant through the requested point: it misses
    # both of its own defining numbers. Pin the size of the miss.
    polynomial = AbbeMaterial(n=1.5, abbe=65.0, model="polynomial")
    n_d_p, n_f_p, n_c_p = (_n(polynomial, w) for w in (LINE_D, LINE_F, LINE_C))
    v_polynomial = (n_d_p - 1.0) / (n_f_p - n_c_p)
    result.record(abbe_polynomial_n_d=n_d_p, abbe_polynomial_V_recovered=v_polynomial)
    result.check_true(
        "legacy_polynomial_model_misses_its_own_defining_numbers",
        "analytic",
        abs(n_d_p - 1.5) > 1e-5 and 63.5 < v_polynomial < 64.5,
        f"n_d={n_d_p:.6f} (requested 1.5) and V_d={v_polynomial:.4f} (requested 65): "
        "a catalog-wide polynomial fit, not an interpolant",
    )
    result.check_true(
        "buchdahl_is_more_self_consistent_than_legacy_polynomial",
        "analytic",
        abs(v_buchdahl - 65.0) < abs(v_polynomial - 65.0),
        f"|V-65|: buchdahl {abs(v_buchdahl - 65.0):.4f} < polynomial {abs(v_polynomial - 65.0):.4f}",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AbbeMaterial(n=1.5, abbe=65.0)
        default_model_warnings = [str(w.message) for w in caught if w.category is FutureWarning]
    result.record(abbe_default_model_future_warnings=default_model_warnings)
    result.check_true(
        "unspecified_abbe_model_warns_that_the_default_changes_in_0p7",
        "invariant",
        any("buchdahl" in m for m in default_model_warnings),
        f"{len(default_model_warnings)} FutureWarning(s): {default_model_warnings}",
    )

    # AbbeMaterialE is anchored at the e-line, which it hits exactly, but its
    # LASSO-fitted dispersion coefficients do NOT reproduce the requested V_e.
    abbe_e = AbbeMaterialE(n=1.5, abbe=65.0)
    n_e, n_fp, n_cp = (_n(abbe_e, w) for w in (LINE_E, LINE_F_PRIME, LINE_C_PRIME))
    v_e = (n_e - 1.0) / (n_fp - n_cp)
    result.record(abbe_e_n_at_e_line=n_e, abbe_e_V_recovered=v_e)
    result.check_close("abbe_e_n_at_e_line_is_exactly_1p5", "analytic", n_e, 1.5, rel=1e-9)
    sweep = {}
    for requested in (20.0, 30.0, 40.0, 50.0, 60.0, 65.0, 70.0, 80.0):
        mat = AbbeMaterialE(n=1.5, abbe=requested)
        recovered = (_n(mat, LINE_E) - 1.0) / (_n(mat, LINE_F_PRIME) - _n(mat, LINE_C_PRIME))
        sweep[f"{requested:g}"] = recovered / requested
    result.record(abbe_e_V_recovered_over_requested=sweep)
    result.check_true(
        "abbe_e_underestimates_its_own_V_across_the_whole_glass_range",
        "analytic",
        all(0.55 < ratio < 0.85 for ratio in sweep.values()),
        "recovered/requested V_e stays in 0.57-0.83 for V_e = 20..80: a systematic "
        f"defect of the pinned model, not a rounding error. ratios={sweep}",
    )
    # Independent confirmation against a real catalog glass: fit each model to
    # N-BK7's own (n, V) at its own reference line, then compare across the band.
    nbk7_ref = Material("N-BK7")
    nd_g, nf_g, nc_g = (_n(nbk7_ref, w) for w in (LINE_D, LINE_F, LINE_C))
    ne_g, nfp_g, ncp_g = (_n(nbk7_ref, w) for w in (LINE_E, LINE_F_PRIME, LINE_C_PRIME))
    fit_b = AbbeMaterial(n=nd_g, abbe=(nd_g - 1.0) / (nf_g - nc_g), model="buchdahl")
    fit_e = AbbeMaterialE(n=ne_g, abbe=(ne_g - 1.0) / (nfp_g - ncp_g))
    band = (0.42, 0.48, 0.546, 0.60, 0.65, 0.70)
    err_b = max(abs(_n(fit_b, w) - _n(nbk7_ref, w)) for w in band)
    err_e = max(abs(_n(fit_e, w) - _n(nbk7_ref, w)) for w in band)
    result.record(nbk7_max_index_error_buchdahl=err_b, nbk7_max_index_error_abbe_e=err_e)
    result.check_true(
        "buchdahl_beats_abbe_e_against_real_nbk7_by_two_orders_of_magnitude",
        "analytic",
        err_b < 1e-3 and err_e > 1e-2 and err_e / err_b > 50.0,
        f"max |dn| over 0.42-0.70 um: buchdahl {err_b:.2e}, AbbeMaterialE {err_e:.2e} "
        f"(ratio {err_e / err_b:.0f}x)",
    )

    # -- 3. catalog glass against an independent Sellmeier oracle -------------
    nbk7 = Material("N-BK7")
    for w in (0.48, LINE_D, 0.65):
        observed = _n(nbk7, w)
        expected = nbk7_sellmeier(w)
        result.record(**{f"nbk7_n_at_{str(w).replace('.', 'p')}": observed})
        result.check_close(
            f"nbk7_matches_schott_sellmeier_at_{w}um", "analytic", observed, expected, rel=1e-6
        )

    wavelengths = np.linspace(0.4, 0.75, 500)
    n_sf5 = np.asarray(Material("N-SF5").n(wavelengths), dtype=float)
    result.record(
        n_sf5_min=float(n_sf5.min()),
        n_sf5_max=float(n_sf5.max()),
        n_sf5_at_0p55=_n(Material("N-SF5"), 0.55),
    )
    result.check_finite("n_sf5_dispersion_curve_finite", n_sf5)
    result.check_true(
        "n_sf5_shows_normal_dispersion",
        "analytic",
        bool(np.all(np.diff(n_sf5) < 0.0)),
        "dn/dlambda < 0 at every one of the 499 intervals over 0.40-0.75 um",
    )

    sf5_default = _n(Material("N-SF5"), 0.55)
    sf5_schott = _n(Material("N-SF5", reference="Schott"), 0.55)
    result.record(n_sf5_default=sf5_default, n_sf5_reference_schott=sf5_schott)
    result.check_close(
        "explicit_schott_reference_matches_default_lookup",
        "invariant",
        sf5_schott,
        sf5_default,
        rel=1e-12,
    )

    # -- 4. non-glass entries ------------------------------------------------
    dna = _n(Material("DNA"), 0.26)
    agcl = _n(Material("AgCl"), 0.55)
    toluene = _n(Material("toluene"), 0.55)
    helium = _n(Material("He"), 0.612)
    result.record(
        n_dna_at_0p26um=dna,
        n_agcl_at_0p55um=agcl,
        n_toluene_at_0p55um=toluene,
        n_he_at_0p612um=helium,
    )
    result.check_true(
        "condensed_phase_indices_exceed_unity",
        "invariant",
        all(v > 1.0 for v in (dna, agcl, toluene)),
        f"DNA={dna:.5f} AgCl={agcl:.5f} toluene={toluene:.5f}",
    )
    result.check_true(
        "helium_index_excess_is_gas_like",
        "analytic",
        1e-6 < (helium - 1.0) < 1e-4,
        f"n(He) - 1 = {helium - 1.0:.3e}, i.e. order 1e-5 as a dilute gas requires",
    )
    result.check_true(
        "agcl_is_high_index_relative_to_glass",
        "invariant",
        agcl > _n(nbk7, 0.55),
        f"n(AgCl)={agcl:.5f} > n(N-BK7)={_n(nbk7, 0.55):.5f} at 0.55 um",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
