"""Characterize generic `Optic` construction for the PB5 canonical builder (CHE-56).

PB5 admits two construction paths this repository had never exercised -- an
even-aspheric surface and a diffractive grating surface -- plus catalog-material
resolution as a *declared* rather than incidental behaviour. Nothing here is
inferred from Optiland's documentation: every case is executed against the
pinned install and compared against a closed-form expectation or a falsifier.

Cases and their oracles
-----------------------
1. ``surface_kwargs_are_silently_filtered`` -- ``GeometryFactory.create``
   filters ``**kwargs`` down to the dataclass fields of the selected geometry
   config (``optiland/surfaces/factories/geometry_factory.py``), so a
   coefficient list handed to a ``standard`` surface is *dropped without
   error*. Falsifier for "Optiland will reject a malformed prescription": it
   will not, which is why the canonical builder must validate eagerly.
2. ``even_asphere_sag_matches_analytic`` -- the sag of the constructed geometry
   against the even-asphere series evaluated independently here. Establishes
   that ``coefficients[i]`` multiplies ``r**(2*(i+1))``, i.e. the series starts
   at ``r^2``.
3. ``even_asphere_reduces_to_sphere`` -- with all coefficients zero and
   ``conic = 0`` the sag must equal the spherical sag of the same radius.
4. ``grating_period_is_micrometres`` -- a plane transmission grating in air at
   normal incidence must diffract into ``sin(theta_m) = m * lambda / d``. The
   only way this holds is if ``grating_period`` shares units with
   ``RealRays.w``, which CHE-12 established is micrometres, while every length
   in the same prescription is millimetres. Three periods are swept so that a
   units error cannot be absorbed into a constant.
5. ``grating_order_zero_is_undeviated`` -- order 0 must pass straight through.
6. ``groove_orientation_is_radians`` -- ``PlaneGrating.grating_vector`` uses
   ``be.sin``/``be.cos`` of the stored angle, so the value is radians; at 0 the
   grating vector is +y and the deviation is in y, and at pi/2 it moves to x.
   A degrees reading would put pi/2 = 1.57 deg essentially back at 0.
7. ``catalog_names_resolve_to_one_exact_match`` -- each catalog glass named by
   the ``ReverseTelephoto`` prescription, showing how many rows survive
   Optiland's substring filter, how many are exact (Levenshtein 0), and which
   file wins. The bare name ``SK15`` resolves to HIKARI, not SCHOTT: a
   prescription that does not record the resolved file has not pinned its glass.
8. ``to_dict_is_a_usable_structural_oracle`` -- two independent builds of the
   same prescription produce equal ``Optic.to_dict()``, so the canonical
   builder can be checked against a sample-built system structurally rather
   than only through traced numbers.

Run inside the agent_solver container:
    ./run.sh python knowledge/solvers/optiland/probes/system_construction_probe.py
"""

from __future__ import annotations

import json

import numpy as np
import optiland.backend as be
from optiland.materials import IdealMaterial, Material
from optiland.optic import Optic
from optiland.rays.real_rays import RealRays

WAVELENGTH_UM = 0.55

# Radii/thicknesses are millimetres throughout (CHE-12).
_ASPHERE_RADIUS_MM = 10.0
_ASPHERE_CONIC = -0.5
_ASPHERE_COEFFICIENTS = (1.0e-3, -2.5e-5, 4.0e-7)


def _even_asphere_sag_analytic(r_mm, radius_mm, conic, coefficients):
    """Independent evaluation of the even-asphere series (not Optiland's code)."""
    r2 = r_mm**2
    curvature = 1.0 / radius_mm
    conic_term = curvature * r2 / (1.0 + np.sqrt(1.0 - (1.0 + conic) * curvature**2 * r2))
    series = sum(c * r2 ** (i + 1) for i, c in enumerate(coefficients))
    return conic_term + series


def _spherical_sag_analytic(r_mm, radius_mm):
    return radius_mm - np.sign(radius_mm) * np.sqrt(radius_mm**2 - r_mm**2)


def case_surface_kwargs_are_silently_filtered() -> dict:
    optic = Optic("kwarg-filter")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    # 'coefficients' is not a field of StandardConfig, and 'radius_of_curvature'
    # is not a field of anything: both are dropped with no error.
    optic.surfaces.add(
        index=1,
        radius=_ASPHERE_RADIUS_MM,
        thickness=1.0,
        material=IdealMaterial(n=1.5),
        coefficients=[1.0, 2.0, 3.0],
        radius_of_curvature=0.02,
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=0.0)
    geometry = optic.surfaces.surfaces[1].geometry
    return {
        "claim": "unknown/mismatched geometry kwargs are silently discarded, not rejected",
        "surface_type_requested": "standard (default)",
        "geometry_class": type(geometry).__name__,
        "geometry_has_coefficients_attribute": hasattr(geometry, "coefficients"),
        "radius_mm": float(be.to_numpy(geometry.radius)),
        "unknown_kwarg_raised": False,
        "implication": (
            "the canonical builder must reject unsupported prescription fields "
            "itself; reaching Optiland with them produces a silently different "
            "optical system"
        ),
    }


def _asphere_optic(coefficients, conic):
    optic = Optic("even-asphere")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        surface_type="even_asphere",
        radius=_ASPHERE_RADIUS_MM,
        conic=conic,
        coefficients=list(coefficients),
        thickness=2.0,
        material=IdealMaterial(n=1.5),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=10.0)
    optic.surfaces.add(index=3, radius=be.inf, thickness=0.0)
    optic.set_aperture(aperture_type="EPD", value=4.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return optic


def case_even_asphere_sag_matches_analytic() -> dict:
    optic = _asphere_optic(_ASPHERE_COEFFICIENTS, _ASPHERE_CONIC)
    geometry = optic.surfaces.surfaces[1].geometry
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    observed = np.asarray(be.to_numpy(geometry.sag(x=r, y=np.zeros_like(r))), dtype=np.float64)
    expected = _even_asphere_sag_analytic(
        r, _ASPHERE_RADIUS_MM, _ASPHERE_CONIC, _ASPHERE_COEFFICIENTS
    )
    # Falsifier: a series starting at r^4 instead of r^2.
    shifted = _even_asphere_sag_analytic(
        r, _ASPHERE_RADIUS_MM, _ASPHERE_CONIC, (0.0, *_ASPHERE_COEFFICIENTS)
    )
    return {
        "claim": "coefficients[i] multiplies r**(2*(i+1)); the series starts at r^2",
        "geometry_class": type(geometry).__name__,
        "radius_mm": _ASPHERE_RADIUS_MM,
        "conic": _ASPHERE_CONIC,
        "coefficients": list(_ASPHERE_COEFFICIENTS),
        "radial_positions_mm": r.tolist(),
        "observed_sag_mm": observed.tolist(),
        "analytic_sag_mm": expected.tolist(),
        "max_abs_error_mm": float(np.max(np.abs(observed - expected))),
        "max_abs_error_if_series_started_at_r4_mm": float(np.max(np.abs(observed - shifted))),
    }


def case_even_asphere_reduces_to_sphere() -> dict:
    optic = _asphere_optic((0.0, 0.0, 0.0), 0.0)
    geometry = optic.surfaces.surfaces[1].geometry
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    observed = np.asarray(be.to_numpy(geometry.sag(x=r, y=np.zeros_like(r))), dtype=np.float64)
    expected = _spherical_sag_analytic(r, _ASPHERE_RADIUS_MM)
    return {
        "claim": "a zero-coefficient, zero-conic even asphere is the sphere of the same radius",
        "geometry_class": type(geometry).__name__,
        "max_abs_error_mm": float(np.max(np.abs(observed - expected))),
        "observed_sag_mm": observed.tolist(),
        "analytic_sag_mm": expected.tolist(),
    }


def _grating_deviation(period_um, order, groove_angle_rad):
    """Diffract one on-axis, normally incident ray through a plane grating in air.

    Traced through `surfaces.trace` on a bare two-surface system so that the
    measured direction cosines are the grating's doing and nothing else's.
    """
    optic = Optic("plane-grating")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        surface_type="grating",
        radius=be.inf,
        grating_order=order,
        grating_period=period_um,
        groove_orientation_angle=groove_angle_rad,
        thickness=0.0,
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=0.0)
    rays = RealRays(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([-1.0]),
        L=np.array([0.0]),
        M=np.array([0.0]),
        N=np.array([1.0]),
        intensity=np.array([1.0]),
        wavelength=np.array([WAVELENGTH_UM]),
    )
    optic.surfaces.trace(rays, skip=1)
    return (
        float(np.asarray(be.to_numpy(rays.L))[0]),
        float(np.asarray(be.to_numpy(rays.M))[0]),
        float(np.asarray(be.to_numpy(rays.N))[0]),
    )


def case_grating_period_is_micrometres() -> dict:
    order = 1
    periods_um = [2.0, 4.0, 8.0]
    observed = []
    predicted = []
    for period_um in periods_um:
        _, m_dir, _ = _grating_deviation(period_um, order, 0.0)
        observed.append(m_dir)
        predicted.append(order * WAVELENGTH_UM / period_um)
    # Falsifier: if `grating_period` were millimetres, sin(theta) would be
    # 1000x smaller for the same number, i.e. 2.75e-4 rather than 0.275.
    return {
        "claim": (
            "grating_period shares units with RealRays.w (micrometres), not the "
            "millimetre geometry unit"
        ),
        "wavelength_um": WAVELENGTH_UM,
        "grating_order": order,
        "periods_um": periods_um,
        "observed_sin_theta": observed,
        "analytic_sin_theta": predicted,
        "max_abs_error": float(np.max(np.abs(np.array(observed) - np.array(predicted)))),
        "predicted_sin_theta_if_period_were_mm": [
            order * WAVELENGTH_UM * 1e-3 / p for p in periods_um
        ],
    }


def case_grating_order_zero_is_undeviated() -> dict:
    l_dir, m_dir, n_dir = _grating_deviation(2.0, 0, 0.0)
    return {
        "claim": "order 0 leaves a normally incident ray undeviated",
        "direction": [l_dir, m_dir, n_dir],
        "max_abs_transverse_direction": float(max(abs(l_dir), abs(m_dir))),
    }


def case_groove_orientation_is_radians() -> dict:
    period_um = 2.0
    order = 1
    expected_sin = order * WAVELENGTH_UM / period_um
    at_zero = _grating_deviation(period_um, order, 0.0)
    at_half_pi = _grating_deviation(period_um, order, np.pi / 2.0)
    return {
        "claim": (
            "groove_orientation_angle is radians; at 0 the grating vector is +y "
            "so the deviation is in y, and at pi/2 it is in x"
        ),
        "analytic_sin_theta": expected_sin,
        "direction_at_0_rad": list(at_zero),
        "direction_at_half_pi_rad": list(at_half_pi),
        "y_deviation_at_0": at_zero[1],
        "x_deviation_at_half_pi": at_half_pi[0],
        "note": (
            "at pi/2 the x component is -sin(theta): grating_vector = "
            "(-sin(angle), cos(angle), 0), so the dispersion axis rotates with "
            "the stored angle. Read as degrees, pi/2 = 1.57 deg would leave the "
            "deviation essentially still in y."
        ),
    }


def case_catalog_names_resolve_to_one_exact_match() -> dict:
    specs = [
        ("N-SK10", None),
        ("SK15", None),
        ("BASF2", None),
        ("FK3", None),
        ("SF15", "hikari"),
        ("N-LAK12", None),
    ]
    rows = {}
    for name, reference in specs:
        material = Material(name, reference)
        matches = material._find_material_matches(material._load_dataframe())
        scores = [int(s) for s in matches["similarity_score"].tolist()]
        rows[f"{name}|{reference}"] = {
            "substring_matches": len(scores),
            "exact_matches": sum(1 for s in scores if s == 0),
            "resolved_catalog_file": str(material.material_data["filename"]),
            "resolved_category": str(material.material_data["category_name"]),
            "n_at_0p5876_um": float(np.asarray(be.to_numpy(material.n(0.5876))).reshape(())),
        }
    return {
        "claim": (
            "every ReverseTelephoto glass has exactly one Levenshtein-0 match, so "
            "resolution is unambiguous -- but the surviving substring set is up to "
            "7 rows and the winning manufacturer is not implied by the name"
        ),
        "materials": rows,
        "implication": (
            "a canonical prescription must record the resolved catalog file so a "
            "database change is detected rather than silently traced"
        ),
    }


def case_to_dict_is_a_usable_structural_oracle() -> dict:
    from optiland.samples.objectives import ReverseTelephoto

    first = ReverseTelephoto().to_dict()
    second = ReverseTelephoto().to_dict()
    surface = first["surface_group"]["surfaces"][1]
    return {
        "claim": "Optic.to_dict() is stable across independent builds of the same prescription",
        "equal_across_builds": first == second,
        "top_level_keys": sorted(first),
        "surface_keys": sorted(surface),
        "surface_1_geometry": surface["geometry"],
        "surface_1_material_post_type": surface["material_post"]["type"],
        "aperture": first["aperture"],
        "wavelength_unit_declared_by_optiland": first["wavelengths"]["wavelengths"][0]["unit"],
        "field_definition": first["fields"]["field_definition"],
    }


def main() -> None:
    report = {
        "probe": "system_construction_probe",
        "issue": "CHE-56 (PB5)",
        "wavelength_um": WAVELENGTH_UM,
        "backend": "numpy (default; not switched)",
        "units": {
            "geometry": "millimetre (CHE-12)",
            "wavelength": "micrometre (CHE-12)",
            "grating_period": "micrometre (established by this probe)",
            "groove_orientation_angle": "radian (established by this probe)",
        },
        "cases": {
            "surface_kwargs_are_silently_filtered": case_surface_kwargs_are_silently_filtered(),
            "even_asphere_sag_matches_analytic": case_even_asphere_sag_matches_analytic(),
            "even_asphere_reduces_to_sphere": case_even_asphere_reduces_to_sphere(),
            "grating_period_is_micrometres": case_grating_period_is_micrometres(),
            "grating_order_zero_is_undeviated": case_grating_order_zero_is_undeviated(),
            "groove_orientation_is_radians": case_groove_orientation_is_radians(),
            "catalog_names_resolve_to_one_exact_match": (
                case_catalog_names_resolve_to_one_exact_match()
            ),
            "to_dict_is_a_usable_structural_oracle": case_to_dict_is_a_usable_structural_oracle(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
