"""Beginner / "Determining Lens Properties" -- https://www.optiland.org/tutorials/lens-properties

Repo-owned reproduction of the paraxial-property tutorial: read every cardinal
point, pupil, F-number, magnification, Lagrange invariant, marginal ray and
chief ray off `Optic.paraxial` for the bundled `CookeTriplet`.

Upstream prints these values but publishes none of them, so validation is a set
of closed-form paraxial identities that hold for *any* system in air. Each one
is computed from Optiland outputs that the identity does not itself produce, so
a bug in the paraxial solver has to conspire across several accessors to pass:

* ``f1 == -f2`` (equal object- and image-space index).
* ``F2 - P2 == f2`` and ``F1 - P1 == f1`` (definition of a principal plane).
* ``N1 == P1`` and ``N2 == P2`` (nodal points coincide with principal points in air).
* ``FNO == f2 / EPD``.
* Marginal-ray height at the object surface ``== EPD / 2``.
* Chief-ray slope at the object surface ``== tan(max field angle)``.
* The Lagrange invariant is conserved: ``|u_bar*y - u*y_bar|`` evaluated
  independently at the object surface and at the image surface (both in air)
  must agree with ``|paraxial.invariant()|``.
* ``magnification == 0`` for an object at infinity.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t02_lens_properties",
    title="Determining Lens Properties",
    level="beginner",
    url="https://www.optiland.org/tutorials/lens-properties",
    demonstrates=(
        "Optic.paraxial accessors: f1/f2, F1/F2, P1/P2, N1/N2, EPD/XPD, "
        "EPL/XPL, FNO, magnification, invariant, marginal_ray, chief_ray. "
        "All return backend arrays of shape (1,) or (num_surfaces, 1), not floats."
    ),
)


def _scalar(value) -> float:
    return float(np.asarray(value).ravel()[0])


def run() -> TutorialResult:
    from optiland.samples.objectives import CookeTriplet

    result = TutorialResult()
    lens = CookeTriplet()
    p = lens.paraxial

    props = {
        name: _scalar(getattr(p, name)())
        for name in (
            "f1", "f2", "F1", "F2", "P1", "P2", "N1", "N2",
            "EPD", "XPD", "EPL", "XPL", "FNO", "magnification", "invariant",
        )
    }
    result.record(**{f"paraxial_{k}": v for k, v in props.items()})

    ya, ua = (np.asarray(v, dtype=float) for v in p.marginal_ray())
    yb, ub = (np.asarray(v, dtype=float) for v in p.chief_ray())
    result.record(
        marginal_ray_shape=list(ya.shape),
        marginal_y=ya.ravel(),
        marginal_u=ua.ravel(),
        chief_y=yb.ravel(),
        chief_u=ub.ravel(),
        num_surfaces=len(lens.surfaces.surfaces),
    )
    result.check_shape("paraxial_rays_are_per_surface_columns", ya, (len(lens.surfaces.surfaces), 1))

    # -- closed-form paraxial identities -------------------------------------
    result.check_close("f1_equals_minus_f2_in_air", "analytic", props["f1"], -props["f2"], rel=1e-12)
    result.check_close(
        "back_focal_point_minus_back_principal_plane_is_efl",
        "analytic",
        props["F2"] - props["P2"],
        props["f2"],
        rel=1e-12,
    )
    result.check_close(
        "front_focal_point_minus_front_principal_plane_is_ffl",
        "analytic",
        props["F1"] - props["P1"],
        props["f1"],
        rel=1e-12,
    )
    result.check_close("front_nodal_equals_front_principal", "analytic", props["N1"], props["P1"], rel=1e-12)
    result.check_close("back_nodal_equals_back_principal", "analytic", props["N2"], props["P2"], rel=1e-12)
    result.check_close(
        "fno_equals_efl_over_epd", "analytic", props["FNO"], props["f2"] / props["EPD"], rel=1e-12
    )
    result.check_close(
        "marginal_ray_starts_at_semi_epd",
        "analytic",
        ya.ravel()[0],
        props["EPD"] / 2.0,
        rel=1e-12,
    )

    max_field_deg = _scalar(lens.fields.max_field)
    result.record(max_field_deg=max_field_deg)
    result.check_close(
        "chief_ray_slope_is_tan_max_field_angle",
        "analytic",
        ub.ravel()[0],
        math.tan(math.radians(max_field_deg)),
        rel=1e-9,
    )

    # Lagrange invariant, computed independently at the two surfaces in air.
    inv_object = ub.ravel()[0] * ya.ravel()[0] - ua.ravel()[0] * yb.ravel()[0]
    inv_image = ub.ravel()[-1] * ya.ravel()[-1] - ua.ravel()[-1] * yb.ravel()[-1]
    result.record(lagrange_invariant_object_space=inv_object, lagrange_invariant_image_space=inv_image)
    result.check_close(
        "lagrange_invariant_conserved_object_to_image",
        "analytic",
        inv_image,
        inv_object,
        rel=1e-6,
    )
    result.check_close(
        "lagrange_invariant_magnitude_matches_accessor",
        "analytic",
        abs(props["invariant"]),
        abs(inv_object),
        rel=1e-6,
    )
    result.note(
        "paraxial.invariant() returns the NEGATIVE of (u_bar*y - u*y_bar) for this "
        "system; only the magnitude is compared. The sign is a convention of "
        "Optiland's accessor, not a physical result."
    )

    result.check_close(
        "magnification_is_zero_for_infinite_object",
        "analytic",
        props["magnification"],
        0.0,
        abs_=1e-12,
    )
    result.check_true(
        "exit_pupil_is_virtual_and_behind_the_lens",
        "invariant",
        props["XPL"] < 0.0 < props["EPL"],
        f"EPL={props['EPL']:.4f} > 0 > XPL={props['XPL']:.4f}",
    )
    result.check_finite("all_paraxial_properties_finite", list(props.values()))
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
