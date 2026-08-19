"""Beginner / "Tracing and Analyzing Rays" -- https://www.optiland.org/tutorials/tracing-and-analyzing-rays

Repo-owned reproduction of the pupil-distribution and per-surface-array
tutorial: the four pupil distributions upstream shows
(`RandomDistribution`, `UniformDistribution`, `GaussianQuadrature`,
`HexagonalDistribution`), then three traces of `ReverseTelephoto` read back out
of the per-surface arrays (`x`, `y`, `opd`, `N`) rather than off the returned
`RealRays`.

Upstream publishes no numbers (every cell ends in a scatter plot), so validation
is invariant/analytic:

* Every distribution's normalized pupil points satisfy ``Px^2 + Py^2 <= 1``.
* ``HexagonalDistribution.generate_points(num_rings=6)`` produces the
  closed-form hexapolar count ``1 + 3*R*(R+1) = 127`` on ``R = 6`` rings, and
  exactly ``R + 1`` distinct pupil radii.
* ``GaussianQuadrature(is_symmetric=False).generate_points(num_rings=6)``
  produces a quadrature grid whose radii are strictly inside the pupil.
* ``RandomDistribution(seed=None)`` is **not reproducible** across instances,
  while ``RandomDistribution(seed=<int>)`` is. This matters beyond the
  distribution objects themselves: ``Optic.trace(distribution="random")`` -- the
  string form the tutorial uses for its scatter plots -- constructs an *unseeded*
  distribution internally, so two identical trace calls return different rays and
  no metric derived from them can be recorded as evidence. Passing a seeded
  ``RandomDistribution`` instance instead makes the same trace reproducible, and
  that is what this reproduction does -- after calling ``generate_points()`` on it,
  which ``Optic.trace`` does **not** do for a non-string distribution.
* The per-surface arrays are shaped ``(num_surfaces, num_rays)`` and their last
  row is element-wise identical to the returned ``RealRays`` -- i.e. the
  tutorial's `surface_group.x[num_surfaces-1, :]` idiom and `rays.x` are the
  same data. (`Optic.surface_group` is deprecated in 0.6.0; `Optic.surfaces` is
  used here and asserted to be the same object.)
* On axis with a *symmetric* (hexapolar) pupil fill the image-plane centroid is
  at the origin to float64 round-off; with the tutorial's *random* fill it is
  only zero to Monte Carlo sampling error, so that case is checked against the
  spot size instead. At the ``Hy=1`` edge field the centroid is displaced -- the
  asymmetry the tutorial's OPL map is showing.
* Image-plane optical path length is positive and finite, and the z direction
  cosine stays in ``(0, 1]`` (all rays still travelling forward).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t05_tracing_and_analyzing_rays",
    title="Tracing and Analyzing Rays",
    level="beginner",
    url="https://www.optiland.org/tutorials/tracing-and-analyzing-rays",
    demonstrates=(
        "optiland.distribution.{Random,Uniform,GaussianQuadrature,Hexagonal}"
        "Distribution.generate_points, Optic.trace(distribution=...), and the "
        "per-surface arrays Optic.surfaces.{x,y,opd,N} of shape "
        "(num_surfaces, num_rays)."
    ),
)

WAVELENGTH_UM = 0.55


def run() -> TutorialResult:
    from optiland import distribution
    from optiland.samples.objectives import ReverseTelephoto

    result = TutorialResult()
    lens = ReverseTelephoto()

    # -- 1. the four pupil distributions -------------------------------------
    dists = {}
    dist_rand = distribution.RandomDistribution(seed=12345)
    dist_rand.generate_points(num_points=100)
    dists["random"] = dist_rand
    dist_uniform = distribution.UniformDistribution()
    dist_uniform.generate_points(num_points=15)
    dists["uniform"] = dist_uniform
    dist_quad = distribution.GaussianQuadrature(is_symmetric=False)
    dist_quad.generate_points(num_rings=6)
    dists["gaussian_quadrature"] = dist_quad
    dist_hex = distribution.HexagonalDistribution()
    dist_hex.generate_points(num_rings=6)
    dists["hexagonal"] = dist_hex

    summary = {}
    for name, dist in dists.items():
        px = np.asarray(dist.x, dtype=float).ravel()
        py = np.asarray(dist.y, dtype=float).ravel()
        radius = np.hypot(px, py)
        summary[name] = {
            "num_points": int(px.size),
            "max_pupil_radius": float(radius.max()),
            "num_distinct_radii": int(np.unique(np.round(radius, 12)).size),
        }
        result.check_true(
            f"{name}_points_inside_unit_pupil",
            "invariant",
            bool(radius.max() <= 1.0 + 1e-12),
            f"max sqrt(Px^2+Py^2) = {radius.max():.12f} <= 1",
        )
        result.check_finite(f"{name}_points_finite", np.concatenate([px, py]))
    result.record(distributions=summary)

    num_rings = 6
    result.check_true(
        "hexapolar_count_matches_closed_form",
        "analytic",
        summary["hexagonal"]["num_points"] == 1 + 3 * num_rings * (num_rings + 1),
        f"{summary['hexagonal']['num_points']} == 1 + 3*{num_rings}*{num_rings + 1} "
        f"= {1 + 3 * num_rings * (num_rings + 1)}",
    )
    result.check_true(
        "hexapolar_has_one_radius_per_ring_plus_centre",
        "analytic",
        summary["hexagonal"]["num_distinct_radii"] == num_rings + 1,
        f"{summary['hexagonal']['num_distinct_radii']} distinct radii == {num_rings} rings + centre",
    )
    result.check_true(
        "gaussian_quadrature_radii_strictly_inside_pupil",
        "analytic",
        summary["gaussian_quadrature"]["max_pupil_radius"] < 1.0,
        "a Gauss-Legendre rule places no node on the integration boundary: "
        f"max radius {summary['gaussian_quadrature']['max_pupil_radius']:.6f} < 1",
    )

    # -- 2. RandomDistribution reproducibility -------------------------------
    a = distribution.RandomDistribution(seed=7)
    a.generate_points(num_points=32)
    b = distribution.RandomDistribution(seed=7)
    b.generate_points(num_points=32)
    seeded_identical = bool(np.array_equal(np.asarray(a.x), np.asarray(b.x)))
    c = distribution.RandomDistribution(seed=None)
    c.generate_points(num_points=32)
    d = distribution.RandomDistribution(seed=None)
    d.generate_points(num_points=32)
    unseeded_identical = bool(np.array_equal(np.asarray(c.x), np.asarray(d.x)))
    result.record(seeded_random_reproducible=seeded_identical, unseeded_random_reproducible=unseeded_identical)
    result.check_true(
        "random_distribution_is_reproducible_only_when_seeded",
        "invariant",
        seeded_identical and not unseeded_identical,
        f"seed=7 reproducible={seeded_identical}, seed=None reproducible={unseeded_identical} "
        "(the tutorial's seed=None must not be used in a repository probe)",
    )

    # -- 3. per-surface arrays ------------------------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_group = lens.surface_group
        surface_group_deprecated = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
    result.record(surface_group_deprecation=surface_group_deprecated)
    result.check_true(
        "surface_group_is_the_deprecated_alias_of_surfaces",
        "invariant",
        legacy_group is lens.surfaces and bool(surface_group_deprecated),
        f"same object, and access warns: {surface_group_deprecated}",
    )

    # A seeded instance, NOT distribution="random": see the module docstring.
    # generate_points() must be called by hand -- Optic.trace only calls it when
    # `distribution` is a STRING, so a bare instance raises
    # AttributeError: 'RandomDistribution' object has no attribute 'x'.
    unprepared = distribution.RandomDistribution(seed=20260819)
    unprepared_error = ""
    try:
        lens.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=1024, distribution=unprepared)
    except AttributeError as exc:
        unprepared_error = f"{type(exc).__name__}: {exc}"
    result.record(unprepared_distribution_instance_error=unprepared_error)
    result.check_true(
        "a_distribution_instance_must_have_generate_points_called_first",
        "invariant",
        unprepared_error.startswith("AttributeError"),
        f"trace(distribution=<fresh instance>) -> {unprepared_error or 'no error'}; "
        "num_rays is only forwarded to generate_points for the STRING form",
    )

    seeded_pupil = distribution.RandomDistribution(seed=20260819)
    seeded_pupil.generate_points(num_points=1024)
    rays = lens.trace(
        Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=1024, distribution=seeded_pupil
    )
    num_surfaces = lens.surfaces.num_surfaces
    x_all = np.asarray(lens.surfaces.x, dtype=float)
    x_image = x_all[num_surfaces - 1, :]
    y_image = np.asarray(lens.surfaces.y, dtype=float)[num_surfaces - 1, :]
    result.record(
        num_surfaces=int(num_surfaces),
        per_surface_array_shape=list(x_all.shape),
        on_axis_num_rays=int(x_image.size),
        on_axis_centroid_x_mm=float(x_image.mean()),
        on_axis_centroid_y_mm=float(y_image.mean()),
        on_axis_rms_spot_mm=float(np.sqrt(np.mean(x_image**2 + y_image**2))),
    )
    result.check_shape("per_surface_arrays_are_surface_by_ray", x_all, (num_surfaces, x_image.size))
    result.check_true(
        "last_surface_row_equals_returned_realrays",
        "invariant",
        bool(np.array_equal(x_image, np.asarray(rays.x, dtype=float))),
        "surfaces.x[num_surfaces-1, :] is element-wise equal to the returned rays.x",
    )
    # A *random* pupil fill makes the centroid a Monte Carlo estimate, so it is
    # only zero to sampling error -- compare it against the spot size, not zero.
    rms_on_axis = float(np.sqrt(np.mean(x_image**2 + y_image**2)))
    centroid_random = float(np.hypot(x_image.mean(), y_image.mean()))
    result.record(on_axis_centroid_over_rms_random=centroid_random / rms_on_axis)
    result.check_true(
        "random_pupil_on_axis_centroid_is_sampling_noise",
        "analytic",
        centroid_random < 0.05 * rms_on_axis,
        f"|centroid| = {centroid_random:.3e} mm is {centroid_random / rms_on_axis:.4f} of "
        f"the {rms_on_axis:.3e} mm RMS spot: consistent with 1/sqrt(1024) Monte Carlo "
        "residue, not with a real asymmetry",
    )
    # The exactly symmetric hexapolar fill must put the centroid at the origin.
    lens.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=12, distribution="hexapolar")
    x_hex = np.asarray(lens.surfaces.x, dtype=float)[num_surfaces - 1, :]
    y_hex = np.asarray(lens.surfaces.y, dtype=float)[num_surfaces - 1, :]
    result.record(
        on_axis_hexapolar_centroid_x_mm=float(x_hex.mean()),
        on_axis_hexapolar_centroid_y_mm=float(y_hex.mean()),
        on_axis_hexapolar_rms_spot_mm=float(np.sqrt(np.mean(x_hex**2 + y_hex**2))),
    )
    result.check_true(
        "symmetric_pupil_on_axis_centroid_is_at_the_origin",
        "analytic",
        abs(float(x_hex.mean())) < 1e-12 and abs(float(y_hex.mean())) < 1e-12,
        f"hexapolar centroid = ({x_hex.mean():.3e}, {y_hex.mean():.3e}) mm; a rotationally "
        "symmetric system on axis with a symmetric pupil fill has an exactly centred spot",
    )

    # The string form is unseeded, so the same call twice gives different rays.
    first = np.asarray(
        lens.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=64, distribution="random").x,
        dtype=float,
    )
    second = np.asarray(
        lens.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=64, distribution="random").x,
        dtype=float,
    )
    replay = distribution.RandomDistribution(seed=20260819)
    replay.generate_points(num_points=1024)
    seeded_again = np.asarray(
        lens.trace(
            Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=1024, distribution=replay
        ).x,
        dtype=float,
    )
    result.record(
        string_random_distribution_reproducible=bool(np.array_equal(first, second)),
        seeded_instance_distribution_reproducible=bool(np.array_equal(seeded_again, x_image)),
    )
    result.check_true(
        "trace_distribution_string_random_is_not_reproducible",
        "invariant",
        not np.array_equal(first, second) and np.array_equal(seeded_again, x_image),
        'Optic.trace(distribution="random") returns different rays on identical '
        "consecutive calls, while trace(distribution=RandomDistribution(seed=...)) "
        "reproduces its rays exactly. Only the latter is admissible as recorded evidence.",
    )

    # -- 4. edge-field OPL and direction cosines ------------------------------
    lens.trace(Hx=0, Hy=1, wavelength=WAVELENGTH_UM, num_rays=15, distribution="hexapolar")
    opd = np.asarray(lens.surfaces.opd, dtype=float)[num_surfaces - 1, :]
    y_edge = np.asarray(lens.surfaces.y, dtype=float)[num_surfaces - 1, :]
    result.record(
        edge_field_num_rays=int(opd.size),
        edge_field_opl_min_mm=float(opd.min()),
        edge_field_opl_max_mm=float(opd.max()),
        edge_field_opl_ptv_mm=float(opd.max() - opd.min()),
        edge_field_centroid_y_mm=float(y_edge.mean()),
    )
    result.check_finite("edge_field_opl_finite", opd)
    result.check_true(
        "opl_is_positive_and_non_degenerate",
        "invariant",
        bool(opd.min() > 0.0 and (opd.max() - opd.min()) > 0.0),
        f"OPL in [{opd.min():.6f}, {opd.max():.6f}] mm, peak-to-valley "
        f"{opd.max() - opd.min():.3e} mm",
    )
    result.check_true(
        "edge_field_centroid_is_displaced_from_axis",
        "analytic",
        abs(float(y_edge.mean())) > 1e-3,
        f"Hy=1 image centroid y = {y_edge.mean():.6f} mm, far from the on-axis 0 -- "
        "the field displacement the tutorial's OPL map is coloured by",
    )

    lens.trace(Hx=0.825, Hy=0.478, wavelength=0.567, num_rays=128, distribution="uniform")
    n_dir = np.asarray(lens.surfaces.N, dtype=float)[num_surfaces - 1, :]
    result.record(
        oblique_field_N_min=float(n_dir.min()),
        oblique_field_N_max=float(n_dir.max()),
        oblique_field_num_rays=int(n_dir.size),
    )
    result.check_true(
        "all_rays_still_travel_forward",
        "invariant",
        bool(np.all(n_dir > 0.0) and np.all(n_dir <= 1.0 + 1e-12)),
        f"z direction cosine in [{n_dir.min():.6f}, {n_dir.max():.6f}] subset (0, 1]",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
