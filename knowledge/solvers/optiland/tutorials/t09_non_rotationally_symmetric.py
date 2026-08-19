"""Intermediate / "Non-Rotationally Symmetric Systems" -- https://www.optiland.org/tutorials/non-rotationally-symmetric

Repo-owned reproduction of the folded-mirror tutorial: an 11-surface system with
four tilted mirrors and a relay doublet, built twice -- once with **absolute**
surface placement (`z=`, `y=`) and once with **relative** placement
(`thickness=`, `dy=`) -- then RMS-spot-optimized over the last mirror spacing.

The tutorial's point is that the two construction styles describe the same
system, and its output is three pictures. This reproduction makes the claim
testable and adds an optimization invariant:

* Both construction styles place every surface vertex at the same (y, z) and
  produce an element-wise identical ray trace. This is the actual assertion
  behind "two approaches", and it is not obvious: the relative form has to
  accumulate ``thickness`` along a path that reverses direction at each mirror.
* Exactly the four surfaces flagged ``material='mirror'`` deviate the beam by
  90 degrees (measured 90.000, 90.000, 90.000, 89.883), while the two refracting
  doublet surfaces deviate 1.17 and 2.28 degrees. Checked from the per-surface
  direction cosines, because ``material='mirror'`` produces **no** distinct
  surface or interaction class -- all 11 surfaces report
  ``RefractiveReflectiveModel``.
* The optimizer strictly reduces the RMS spot size it was given as an operand,
  and the variable it was allowed to move stays inside its declared bounds.
* Fields are asymmetric (-0.5, 0, +0.5 degrees) in a system with no rotational
  symmetry, so the on-axis field is *not* required to be centred; instead the
  reproduction checks that +/-0.5 deg land symmetrically about the 0 deg
  centroid, which the folded geometry must still preserve in the y-z plane.

``draw3D`` is not called (it hangs headlessly; see `failure_guide.md`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t09_non_rotationally_symmetric",
    title="Non-Rotationally Symmetric Systems",
    level="intermediate",
    url="https://www.optiland.org/tutorials/non-rotationally-symmetric",
    demonstrates=(
        "Absolute surface placement (z=, y= kwargs) vs relative placement "
        "(thickness=, dy=), material='mirror', per-surface rx tilts, "
        "analysis.SpotDiagram(num_rings, distribution), and "
        "optimization.OptimizationProblem + OptimizerGeneric with an "
        "rms_spot_size operand over a thickness variable."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.633
EPD_MM = 5.0
FIELDS_DEG = (-0.5, 0.0, 0.5)


def _configure(lens):
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    for y in FIELDS_DEG:
        lens.fields.add(y=y)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def build_absolute():
    """Approach 1: every surface vertex given in absolute (y, z)."""
    from optiland import optic

    lens = optic.Optic()
    _configure(lens)
    lens.surfaces.add(index=0, radius=np.inf, z=-np.inf)
    lens.surfaces.add(index=1, z=0)
    lens.surfaces.add(index=2, z=10, material="bk7", surface_type="standard")
    lens.surfaces.add(index=3, z=15, material="air", surface_type="standard")
    lens.surfaces.add(index=4, z=25, material="mirror", rx=-np.pi / 4, is_stop=True)
    lens.surfaces.add(index=5, z=25, y=-15, material="mirror", rx=-np.pi / 4)
    lens.surfaces.add(index=6, z=45, y=-15, material="mirror", rx=np.pi / 4)
    lens.surfaces.add(index=7, z=45, y=-10, radius=-30, material="bk7", rx=np.pi / 2)
    lens.surfaces.add(index=8, z=45, y=-5, radius=30, material="air", rx=np.pi / 2)
    lens.surfaces.add(index=9, z=45, y=10, material="mirror", rx=np.pi / 4)
    lens.surfaces.add(index=10, z=55, y=10)
    return lens


def build_relative():
    """Approach 2: the same system by accumulated thickness and decenter."""
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=-np.inf)
    lens.surfaces.add(index=1, thickness=10)
    lens.surfaces.add(index=2, thickness=5, material="bk7", surface_type="standard")
    lens.surfaces.add(index=3, thickness=10, material="air", surface_type="standard")
    lens.surfaces.add(index=4, thickness=0, material="mirror", rx=-np.pi / 4, is_stop=True)
    lens.surfaces.add(index=5, thickness=20, dy=-15, material="mirror", rx=-np.pi / 4)
    lens.surfaces.add(index=6, thickness=0, dy=-15, material="mirror", rx=np.pi / 4)
    lens.surfaces.add(index=7, thickness=0, dy=-10, radius=-30, material="bk7", rx=np.pi / 2)
    lens.surfaces.add(index=8, thickness=0, dy=-5, radius=30, material="air", rx=np.pi / 2)
    lens.surfaces.add(index=9, thickness=10, dy=10, material="mirror", rx=np.pi / 4)
    lens.surfaces.add(index=10, dy=10)
    _configure(lens)
    return lens


def _vertices(lens) -> np.ndarray:
    out = []
    for s in lens.surfaces.surfaces:
        cs = s.geometry.cs
        out.append(
            [float(np.asarray(cs.y).ravel()[0]), float(np.asarray(cs.z).ravel()[0])]
        )
    return np.asarray(out, dtype=float)


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import analysis, optimization

    result = TutorialResult()
    absolute = build_absolute()
    relative = build_relative()

    v_abs = _vertices(absolute)
    v_rel = _vertices(relative)
    finite = np.isfinite(v_abs).all(axis=1) & np.isfinite(v_rel).all(axis=1)
    result.record(
        num_surfaces=len(absolute.surfaces.surfaces),
        vertices_absolute_yz=v_abs,
        vertices_relative_yz=v_rel,
        max_vertex_deviation_mm=float(np.max(np.abs(v_abs[finite] - v_rel[finite]))),
    )
    result.check_true(
        "absolute_and_relative_construction_place_every_vertex_identically",
        "invariant",
        bool(np.max(np.abs(v_abs[finite] - v_rel[finite])) < 1e-12),
        f"max |delta(y, z)| over the {int(finite.sum())} finite surfaces = "
        f"{float(np.max(np.abs(v_abs[finite] - v_rel[finite]))):.3e} mm",
    )

    rays_abs = absolute.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=6)
    rays_rel = relative.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=6)
    identical = all(
        np.array_equal(
            np.asarray(getattr(rays_abs, a), dtype=float),
            np.asarray(getattr(rays_rel, a), dtype=float),
        )
        for a in ("x", "y", "z", "L", "M", "N", "opd")
    )
    result.record(num_traced_rays=int(np.asarray(rays_abs.x).size))
    result.check_true(
        "absolute_and_relative_construction_trace_identically",
        "invariant",
        identical,
        "x, y, z, L, M, N and opd all element-wise equal between the two builds",
    )

    # -- every mirror must deviate the beam by 90 degrees ---------------------
    # A 45-degree fold mirror turns the chief direction through 90 degrees. The
    # per-surface direction cosines make that checkable without trusting any
    # surface metadata -- which matters, because material='mirror' does NOT
    # produce a distinct surface or interaction class in 0.6.0: every surface
    # here reports interaction_model=RefractiveReflectiveModel and the only
    # difference is the post-surface material.
    absolute.trace(Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=6)
    dirs = np.stack(
        [
            np.asarray(absolute.surfaces.L, dtype=float),
            np.asarray(absolute.surfaces.M, dtype=float),
            np.asarray(absolute.surfaces.N, dtype=float),
        ]
    )
    deviations_deg = []
    for i in range(1, dirs.shape[1]):
        d0, d1 = dirs[:, i - 1, :], dirs[:, i, :]
        cos = (d0 * d1).sum(0) / (
            np.linalg.norm(d0, axis=0) * np.linalg.norm(d1, axis=0)
        )
        deviations_deg.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).mean()))
    folded = [i + 1 for i, dev in enumerate(deviations_deg) if dev > 60.0]
    interaction_models = sorted(
        {type(s.interaction_model).__name__ for s in absolute.surfaces.surfaces}
    )
    result.record(
        mean_deviation_per_surface_deg=deviations_deg,
        surfaces_deviating_more_than_60_deg=folded,
        interaction_model_classes=interaction_models,
    )
    result.check_true(
        "exactly_the_four_mirrors_deviate_the_beam_by_90_degrees",
        "analytic",
        folded == [4, 5, 6, 9]
        and all(abs(deviations_deg[i - 1] - 90.0) < 0.2 for i in folded),
        f"surfaces {folded} deviate "
        f"{[round(deviations_deg[i - 1], 4) for i in folded]} degrees; the two refracting "
        f"doublet surfaces deviate only "
        f"{[round(deviations_deg[i - 1], 4) for i in (7, 8)]} degrees",
    )
    result.check_true(
        "mirror_is_a_material_not_a_surface_or_interaction_type",
        "invariant",
        interaction_models == ["RefractiveReflectiveModel"],
        f"all 11 surfaces report interaction_model in {interaction_models}: "
        "material='mirror' is resolved by the material factory, so a builder cannot "
        "detect reflectivity from the surface or interaction class",
    )

    # -- field symmetry about the 0 deg field ---------------------------------
    centroids = {}
    for hy, label in ((-1.0, "minus_half_deg"), (0.0, "zero_deg"), (1.0, "plus_half_deg")):
        r = absolute.trace(Hx=0, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=6)
        centroids[label] = float(np.asarray(r.y, dtype=float).mean())
    result.record(field_centroids_y_mm=centroids)
    offset_minus = centroids["minus_half_deg"] - centroids["zero_deg"]
    offset_plus = centroids["plus_half_deg"] - centroids["zero_deg"]
    result.record(field_offset_minus_mm=offset_minus, field_offset_plus_mm=offset_plus)
    result.check_close(
        "opposite_fields_land_symmetrically_about_the_axial_field",
        "analytic",
        abs(offset_plus),
        abs(offset_minus),
        rel=0.05,
    )

    # -- optimization ---------------------------------------------------------
    lens = build_absolute()
    spot_before = np.asarray(
        analysis.SpotDiagram(lens, num_rings=10, distribution="hexapolar").rms_spot_radius(),
        dtype=float,
    )
    problem = optimization.OptimizationProblem()
    problem.add_variable(lens, "thickness", surface_number=9, min_val=10, max_val=20)
    for field in lens.fields.get_field_coords():
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=10,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": field[0],
                "Hy": field[1],
                "num_rays": 16,
                "wavelength": WAVELENGTH_UM,
                "distribution": "uniform",
            },
        )
    merit_before = float(problem.sum_squared())
    optimizer = optimization.OptimizerGeneric(problem)
    optimizer.optimize()
    merit_after = float(problem.sum_squared())
    thickness_after = float(
        np.asarray(lens.surfaces.surfaces[9].thickness).ravel()[0]
    )
    spot_after = np.asarray(
        analysis.SpotDiagram(lens, num_rings=10, distribution="hexapolar").rms_spot_radius(),
        dtype=float,
    )
    plt.close("all")
    result.record(
        merit_before=merit_before,
        merit_after=merit_after,
        optimized_thickness_mm=thickness_after,
        spot_rms_before_mm=spot_before.ravel(),
        spot_rms_after_mm=spot_after.ravel(),
    )
    result.check_true(
        "optimizer_reduces_the_merit_function",
        "invariant",
        merit_after < merit_before,
        f"sum of squared operand residuals {merit_before:.6e} -> {merit_after:.6e}",
    )
    result.check_true(
        "optimized_variable_respects_its_declared_bounds",
        "invariant",
        10.0 - 1e-9 <= thickness_after <= 20.0 + 1e-9,
        f"surface 9 thickness = {thickness_after:.6f} mm within [10, 20]",
    )
    result.check_true(
        "optimization_improves_the_mean_rms_spot_radius",
        "analytic",
        float(spot_after.mean()) < float(spot_before.mean()),
        f"mean RMS spot radius {float(spot_before.mean()):.6f} -> "
        f"{float(spot_after.mean()):.6f} mm over the three fields",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
