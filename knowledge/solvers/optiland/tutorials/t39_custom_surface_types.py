"""Advanced / "Custom Surface Types" -- https://www.optiland.org/tutorials/extending-surfaces

Repo-owned reproduction of the surface-extension tutorial: subclass
`geometries.NewtonRaphsonGeometry` with a hand-written ``sag`` (a conic plus
``a*r + b*r^3 + c*sin(d*r^2)``) and a hand-written analytic ``_surface_normal``,
wrap it in a `surfaces.Surface`, and insert it into an `Optic` through
``surfaces.add(new_surface=...)``.

This tutorial's real content is that **the user must supply the surface normal by
hand**. The validation here is the check the tutorial does not make -- comparing the
analytic normal against a central finite difference of its own ``sag`` -- and that
check **fails on the published code**:

* **The tutorial's ``_surface_normal`` is mathematically wrong.** For the ``a*r``
  sag term, ``d(a*r)/dx = a*x/r``, but the tutorial writes ``a * x / r2`` -- off by
  a factor of ``r``. Measured disagreement against the finite-difference gradient is
  0.52 in direction cosine, i.e. the normal points somewhere else entirely. (The
  ``b*r^3`` and ``c*sin(d*r^2)`` derivatives in the same expression are correct.)
* A **corrected** normal, differing only in that one term, agrees with the finite
  difference to 1.3e-9 -- against the published version's 0.52 -- and refracts the
  rays differently enough to change the RMS spot radius. Both versions are traced so
  the diagnosis is falsifiable rather than asserted. The wrong normal is therefore a
  *silent physics error*, not a crash.
* Separately, **the vertex ray comes back NaN in both versions**: ``a*r`` is genuinely
  non-differentiable at ``r = 0`` (a conical cusp for ``a != 0``), so even the correct
  ``a*x/r`` is 0/0 there. The trace returns the NaN ray rather than raising.
* The normal is a unit vector everywhere (``nx^2 + ny^2 + nz^2 == 1``) in both
  versions -- which is why normalizing cannot rescue a wrong gradient, and why the
  unit-norm check alone is not a validation.
* The custom sag reduces to the base conic when ``a = b = c = 0``, checked against a
  concrete ``StandardGeometry`` of the same radius. (``NewtonRaphsonGeometry`` cannot
  be instantiated directly -- it declares ``sag`` and ``_surface_normal`` abstract,
  which is precisely why the tutorial subclasses it.)
* A real trace through the assembled lens produces finite coordinates, unit
  direction cosines, and every ray on the image plane.
* The custom geometry's own ``a*r`` term is singular at ``r = 0`` (``a*x/r^2``), and
  **how it fails depends on the input type**: Python floats raise
  ``ZeroDivisionError`` while numpy arrays return ``nan`` under the warning upstream
  suppresses. Both are recorded, because upstream's ``warnings.catch_warnings``
  only covers the array path.

``lens.draw3D()`` (upstream's final block) is not called: it hangs headlessly.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t39_custom_surface_types",
    title="Custom Surface Types",
    level="advanced",
    url="https://www.optiland.org/tutorials/extending-surfaces",
    demonstrates=(
        "subclassing geometries.NewtonRaphsonGeometry with custom sag() and "
        "_surface_normal(), assembling surfaces.Surface(geometry=..., "
        "material_post=..., is_stop=...), and Optic.surfaces.add(new_surface=...)."
    ),
)

RADIUS_MM = -513.7
A, B, C, D = -7.043e-01, 6.622e-05, 2.142e-01, -1.110e-02
SEMI_APERTURE_MM = 12.0
WAVELENGTH_UM = 0.55


def _make_geometry_class(corrected: bool = False):
    """The tutorial's geometry. With ``corrected=True``, the ``a*r`` derivative is fixed.

    Upstream writes ``self.a * x / r2`` for ``d(a*r)/dx``; the correct expression is
    ``self.a * x / r``. Everything else is identical, so any difference in behaviour
    between the two is attributable to that single term.
    """
    from optiland.geometries import NewtonRaphsonGeometry

    class NewGeometry(NewtonRaphsonGeometry):
        def __init__(self, coordinate_system, radius, conic, a, b, c, d):
            super().__init__(coordinate_system, radius, conic=0.0, tol=1e-10, max_iter=100)
            self.a, self.b, self.c, self.d = a, b, c, d

        def sag(self, x=0, y=0):
            r2 = x**2 + y**2
            z = r2 / (self.radius * (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
            r = np.sqrt(r2)
            z += self.a * r + self.b * r**3 + self.c * np.sin(self.d * r2)
            return z

        def _surface_normal(self, x, y):
            r2 = x**2 + y**2
            denom = self.radius * np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
            dfdx = x / denom
            dfdy = y / denom
            dfdz = -1
            r = np.sqrt(r2)
            first_order_denominator = r if corrected else r2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfdx += (
                    self.a * x / first_order_denominator
                    + 3 * self.b * x * r
                    + 2 * self.c * self.d * x * np.cos(self.d * r2)
                )
                dfdy += (
                    self.a * y / first_order_denominator
                    + 3 * self.b * y * r
                    + 2 * self.c * self.d * y * np.cos(self.d * r2)
                )
            mag = np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)
            return dfdx / mag, dfdy / mag, dfdz / mag

    return NewGeometry


def build_geometry(corrected: bool = False):
    from optiland.coordinate_system import CoordinateSystem

    NewGeometry = _make_geometry_class(corrected=corrected)
    cs = CoordinateSystem(x=0, y=0, z=0, rx=0, ry=0, rz=0, reference_cs=None)
    return NewGeometry(coordinate_system=cs, radius=RADIUS_MM, conic=0, a=A, b=B, c=C, d=D)


def build_lens(geometry):
    from optiland import optic
    from optiland.materials import Material
    from optiland.surfaces import Surface

    new_surface = Surface(
        geometry=geometry,
        previous_surface=None,
        material_post=Material(name="SF2"),
        is_stop=True,
    )
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(index=1, new_surface=new_surface, thickness=15)
    lens.surfaces.add(index=2, radius=-100, thickness=50)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries import StandardGeometry

    result = TutorialResult()
    geometry = build_geometry()

    # -- the custom sag reduces to the base conic when the extra terms vanish ----
    NewGeometry = _make_geometry_class()
    cs = CoordinateSystem(x=0, y=0, z=0, rx=0, ry=0, rz=0, reference_cs=None)
    plain = NewGeometry(coordinate_system=cs, radius=RADIUS_MM, conic=0, a=0, b=0, c=0, d=0)
    # NewtonRaphsonGeometry is abstract -- it declares sag and _surface_normal as
    # abstract methods, which is exactly why the tutorial subclasses it. Use the
    # concrete StandardGeometry as the base-conic oracle.
    reference = StandardGeometry(cs, RADIUS_MM, conic=0.0)
    probe_r = np.linspace(0.5, SEMI_APERTURE_MM, 25)
    plain_sag = np.asarray([float(plain.sag(r, 0.0)) for r in probe_r])
    reference_sag = np.asarray([float(np.asarray(reference.sag(r, 0.0)).ravel()[0]) for r in probe_r])
    result.record(
        max_abs_sag_difference_vs_base_conic=float(np.max(np.abs(plain_sag - reference_sag)))
    )
    result.check_true(
        "the_custom_sag_reduces_to_the_base_conic_when_its_terms_vanish",
        "analytic",
        float(np.max(np.abs(plain_sag - reference_sag))) < 1e-12,
        f"max |sag_custom(a=b=c=0) - sag_StandardGeometry| = "
        f"{float(np.max(np.abs(plain_sag - reference_sag))):.3e} mm over r = 0.5..12 mm",
    )

    # -- the hand-derived normal against a finite-difference gradient ------------
    # This is the check the tutorial does not make, and it is the one that matters.
    step = 1e-6
    samples = []
    for radius in (1.0, 3.0, 6.0, 9.0, 12.0):
        for angle in (0.0, 0.7, 1.9, 3.5, 5.1):
            samples.append((radius * np.cos(angle), radius * np.sin(angle)))

    def _normal_errors(geom):
        errors, unit_errors = [], []
        for x, y in samples:
            nx, ny, nz = (float(np.asarray(v).ravel()[0]) for v in geom._surface_normal(x, y))
            dzdx = (float(geom.sag(x + step, y)) - float(geom.sag(x - step, y))) / (2 * step)
            dzdy = (float(geom.sag(x, y + step)) - float(geom.sag(x, y - step))) / (2 * step)
            magnitude = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
            expected = (dzdx / magnitude, dzdy / magnitude, -1.0 / magnitude)
            errors.append(max(abs(nx - expected[0]), abs(ny - expected[1]), abs(nz - expected[2])))
            unit_errors.append(abs(nx**2 + ny**2 + nz**2 - 1.0))
        return float(max(errors)), float(max(unit_errors))

    published_error, published_unit_error = _normal_errors(geometry)
    corrected_geometry = build_geometry(corrected=True)
    corrected_error, corrected_unit_error = _normal_errors(corrected_geometry)
    result.record(
        num_normal_samples=len(samples),
        published_normal_vs_finite_difference_error=published_error,
        corrected_normal_vs_finite_difference_error=corrected_error,
        published_normal_unit_norm_error=published_unit_error,
        corrected_normal_unit_norm_error=corrected_unit_error,
    )
    result.check_true(
        "the_published_analytic_normal_disagrees_with_the_gradient_of_its_own_sag",
        "analytic",
        published_error > 0.1,
        f"max component-wise difference {published_error:.3e} over {len(samples)} aperture "
        "points between the tutorial's _surface_normal() and a central difference of its "
        "own sag(). d(a*r)/dx = a*x/r, but the tutorial writes a*x/r2. The b*r^3 and "
        "c*sin(d*r^2) derivatives beside it are correct, so this is one wrong term, not a "
        "wrong method.",
    )
    result.check_true(
        "correcting_only_that_one_term_makes_the_normal_agree_to_float_precision",
        "analytic",
        corrected_error < 1e-6,
        f"replacing a*x/r2 with a*x/r drops the disagreement from {published_error:.3e} to "
        f"{corrected_error:.3e}. Nothing else in the class changed, which is what makes "
        "the diagnosis falsifiable.",
    )
    result.check_true(
        "both_normals_are_unit_vectors_so_normalisation_cannot_catch_the_error",
        "analytic",
        published_unit_error < 1e-12 and corrected_unit_error < 1e-12,
        f"max |n|^2 - 1: published {published_unit_error:.3e}, corrected "
        f"{corrected_unit_error:.3e}. A wrong gradient normalizes to a unit vector just "
        "as happily as a right one, so a unit-norm assertion is not a validation of a "
        "hand-written surface normal.",
    )

    # -- the singular vertex ---------------------------------------------------
    on_axis_error = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            [float(np.asarray(v).ravel()[0]) for v in geometry._surface_normal(0.0, 0.0)]
        except Exception as exc:  # noqa: BLE001 - the failure mode is the evidence
            on_axis_error = f"{type(exc).__name__}: {exc}"
        array_normal = [
            float(np.asarray(v).ravel()[0])
            for v in geometry._surface_normal(np.array([0.0]), np.array([0.0]))
        ]
    result.record(
        on_axis_scalar_normal_error=on_axis_error,
        on_axis_array_normal=array_normal,
        on_axis_array_normal_is_finite=bool(np.all(np.isfinite(array_normal))),
    )
    result.check_true(
        "the_a_times_r_term_makes_the_normal_singular_at_the_vertex",
        "analytic",
        bool(on_axis_error) or not np.all(np.isfinite(array_normal)),
        f"_surface_normal(0.0, 0.0) with PYTHON FLOATS raises {on_axis_error!r}, while the "
        f"same call with numpy arrays returns {array_normal}. This is not only the wrong "
        "term: `a*r` is genuinely non-differentiable at r = 0 (the surface has a conical "
        "cusp there for a != 0), so even the corrected a*x/r is 0/0 at the vertex. "
        "Upstream's warnings.catch_warnings suppresses only the numpy path -- whether the "
        "failure is loud depends on the input TYPE.",
    )

    # -- a real trace, published vs corrected -----------------------------------
    traces = {}
    for label, corrected in (("published", False), ("corrected", True)):
        lens = build_lens(build_geometry(corrected=corrected))
        rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8)
        x = np.asarray(rays.x, dtype=float)
        y = np.asarray(rays.y, dtype=float)
        z = np.asarray(rays.z, dtype=float)
        norms = (
            np.asarray(rays.L, dtype=float) ** 2
            + np.asarray(rays.M, dtype=float) ** 2
            + np.asarray(rays.N, dtype=float) ** 2
        )
        image_z = float(np.asarray(lens.surfaces.surfaces[-1].geometry.cs.z).ravel()[0])
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(norms)
        traces[label] = {
            "num_rays": int(x.size),
            "all_finite": bool(finite.all()),
            "num_nonfinite_direction_cosines": int(np.count_nonzero(~np.isfinite(norms))),
            "image_surface_z_mm": image_z,
            # Reduced over the finite subset so the published (NaN) case still
            # records comparable numbers instead of nan everywhere.
            "rms_spot_radius_mm": float(np.sqrt(np.mean(x[finite] ** 2 + y[finite] ** 2)))
            if finite.any()
            else float("nan"),
            "max_direction_norm_error": float(np.max(np.abs(norms[finite] - 1.0)))
            if finite.any()
            else float("nan"),
            "max_abs_z_minus_image": float(np.max(np.abs(z[finite] - image_z)))
            if finite.any()
            else float("nan"),
        }
    result.record(traces=traces)
    result.check_true(
        "the_vertex_ray_comes_back_non_finite_in_both_versions",
        "analytic",
        traces["published"]["num_nonfinite_direction_cosines"] == 1
        and traces["corrected"]["num_nonfinite_direction_cosines"] == 1,
        f"exactly 1 of {traces['published']['num_rays']} rays has non-finite direction "
        "cosines in both the published and the corrected geometry: the r = 0 ray, killed "
        "by the conical cusp of the a*r term. The trace does NOT raise -- a NaN ray is "
        "returned alongside the good ones, so 'the script ran' is not evidence of "
        "anything. This failure is independent of the wrong derivative below.",
    )
    result.check_true(
        "the_corrected_geometry_traces_the_remaining_rays_correctly",
        "analytic",
        traces["corrected"]["max_direction_norm_error"] < 1e-12
        and traces["corrected"]["max_abs_z_minus_image"] < 1e-9,
        f"over the {traces['corrected']['num_rays'] - 1} finite rays: max unit-norm error "
        f"{traces['corrected']['max_direction_norm_error']:.3e}, all landing at z = "
        f"{traces['corrected']['image_surface_z_mm']}",
    )
    spot_ratio = (
        traces["published"]["rms_spot_radius_mm"] / traces["corrected"]["rms_spot_radius_mm"]
    )
    result.record(published_over_corrected_rms_spot=spot_ratio)
    result.check_true(
        "the_wrong_normal_changes_the_physics_not_just_the_finiteness",
        "analytic",
        abs(spot_ratio - 1.0) > 0.05,
        f"RMS spot radius over the finite rays: {traces['published']['rms_spot_radius_mm']:.6f} mm "
        f"published vs {traces['corrected']['rms_spot_radius_mm']:.6f} mm corrected "
        f"(ratio {spot_ratio:.4f}). The wrong a*x/r2 term refracts every ray in the wrong "
        "direction; the result is a different lens, silently.",
    )

    # -- the tutorial's 3D sag surface plot -------------------------------------
    grid = np.linspace(-SEMI_APERTURE_MM, SEMI_APERTURE_MM, 64)
    gx, gy = np.meshgrid(grid, grid)
    gz = geometry.sag(gx, gy)
    gz = np.asarray(gz, dtype=float).copy()
    gz[gx**2 + gy**2 > SEMI_APERTURE_MM**2] = np.nan
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(gx, gy, gz, cmap="viridis")
    plt.close("all")
    inside = np.isfinite(gz)
    result.record(
        sag_grid_min_mm=float(np.nanmin(gz)),
        sag_grid_max_mm=float(np.nanmax(gz)),
        sag_grid_finite_fraction=float(inside.mean()),
    )
    result.check_true(
        "the_sag_surface_is_finite_across_the_clear_aperture",
        "analytic",
        bool(np.all(np.isfinite(gz[inside]))) and float(inside.mean()) > 0.7,
        f"sag spans [{float(np.nanmin(gz)):.6f}, {float(np.nanmax(gz)):.6f}] mm over the "
        f"{float(inside.mean()) * 100:.1f}% of the 64x64 grid inside the 12 mm aperture",
    )
    build_lens(build_geometry(corrected=True)).draw(num_rays=4)
    plt.close("all")
    result.check_true(
        "the_custom_surface_lens_draws_headless",
        "qualitative",
        True,
        "lens.draw(num_rays=4) completed; lens.draw3D() deliberately skipped (hangs)",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
