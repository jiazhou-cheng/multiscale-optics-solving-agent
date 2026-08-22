"""CHE-30 (M3.1): the established `RealRays.opd` convention, and its falsifiers.

M1 recorded `opd_reference` and `opd_sign` as `unverified`, so the coupler
contract layer refused a real Optiland trace as an optical path length. These
tests establish the convention against manufactured geometries with closed-form
answers, and -- equally important -- assert that each *competing* hypothesis is
detected. A characterization that cannot be made to fail proves nothing.

The established convention, all four parts:

1. **Quantity**: absolute accumulated *optical* path length, index-weighted by
   the medium preceding each surface. Not an OPD relative to a chief ray.
2. **Sign**: non-negative and non-decreasing under propagation and refraction,
   because `standard_surface.py` accumulates `be.abs(t * n_pre)`. Larger means
   longer path. Two interaction models subtract, so monotonicity is a property
   of the refractive path, not of `opd` in general (see the thin-lens test).
3. **Reference**: the ray launch state, where the accumulator is seeded to
   zero. For an object at infinity `Optic.trace` aims that plane at
   `positions[1] - (EPD - min(positions[1:-1]))`, so **the zero moves when the
   aperture changes**. For a finite object it is the object plane.
4. **Unit**: the lens geometry unit, millimetres for this project's
   prescriptions.

Evidence is recorded in
`benchmarks/probes/records/optiland/opd_convention_probe.json`, captured by
running `benchmarks/probes/optiland/opd_convention_probe.py` against the
pinned install.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_probe_expected

optiland_backend = pytest.importorskip("optiland.backend")
IdealMaterial = pytest.importorskip("optiland.materials").IdealMaterial
Optic = pytest.importorskip("optiland.optic").Optic
RealRays = pytest.importorskip("optiland.rays.real_rays").RealRays

pytestmark = pytest.mark.optiland

WAVELENGTH_UM = 0.55

# float64 round-off over a handful of accumulations; every case below is exact
# arithmetic in principle, so this is a dtype budget rather than a physics one.
ROUND_OFF_MM = 1e-12


def _rays(x, y, z, L, M, N):
    x, y, z, L, M, N = (np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (x, y, z, L, M, N))
    count = x.size
    return RealRays(
        x.copy(),
        y.copy(),
        z.copy(),
        L.copy(),
        M.copy(),
        N.copy(),
        np.ones(count, dtype=np.float64),
        np.full(count, WAVELENGTH_UM, dtype=np.float64),
    )


def _free_space_optic(distance_mm: float):
    optic = Optic("opd-convention-free-space")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.add(index=1, radius=optiland_backend.inf, z=distance_mm)
    return optic


def _infinite_object_optic(*, epd_mm: float, separation_mm: float):
    optic = Optic("opd-convention-launch-plane")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=optiland_backend.inf)
    optic.surfaces.add(index=1, radius=optiland_backend.inf, thickness=separation_mm, is_stop=True)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=epd_mm)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return optic


def test_accumulator_is_seeded_at_zero() -> None:
    """`opd` is measured from the launch state, not from a system plane."""
    rays = _rays(0.5, -0.25, 0.0, 0.0, 0.0, 1.0)
    assert float(np.max(np.abs(np.asarray(rays.opd)))) == 0.0


def test_opd_is_the_slant_path_not_the_axial_separation() -> None:
    """An oblique ray accumulates `d / N`, which pins 'absolute path' over 'plane offset'."""
    L = np.array([-0.03, 0.0, 0.02])
    M = np.array([0.01, -0.025, 0.015])
    N = np.sqrt(1.0 - L**2 - M**2)
    distance_mm = 10.0
    rays = _rays(np.zeros(3), np.zeros(3), np.zeros(3), L, M, N)
    _free_space_optic(distance_mm).surfaces[1].trace(rays)

    observed = np.asarray(rays.opd, dtype=np.float64)
    assert np.max(np.abs(observed - distance_mm / N)) <= ROUND_OFF_MM

    # Falsifier: the axial-separation hypothesis is wrong by the obliquity
    # excess, so this characterization is not trivially satisfiable.
    assert np.max(np.abs(observed - distance_mm)) > 1e-3


def test_opd_is_index_weighted_optical_path() -> None:
    """A glass slab contributes `n * t`, separating optical path from geometric path."""
    t_air_mm, t_glass_mm, n = 4.0, 6.0, 1.7
    rays = _rays(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    optic = Optic("opd-convention-index")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.add(
        index=1, radius=optiland_backend.inf, z=t_air_mm, material=IdealMaterial(n=n)
    )
    optic.surfaces.add(index=2, radius=optiland_backend.inf, z=t_air_mm + t_glass_mm)
    optic.surfaces[1].trace(rays)
    # The medium *preceding* each surface is what weights the segment, so the
    # glass index must not appear until the second surface is crossed.
    assert abs(float(np.asarray(rays.opd)[0]) - t_air_mm) <= ROUND_OFF_MM
    optic.surfaces[2].trace(rays)
    observed = float(np.asarray(rays.opd)[0])

    assert abs(observed - (t_air_mm + n * t_glass_mm)) <= ROUND_OFF_MM

    # Falsifier: a geometric-path accumulator would be short by (n-1)*t.
    assert abs(observed - (t_air_mm + t_glass_mm)) == pytest.approx(
        (n - 1.0) * t_glass_mm, rel=1e-9
    )


@pytest.mark.parametrize("epd_mm", [2.0, 4.0, 7.5])
def test_infinite_object_opl_zero_is_aperture_dependent(epd_mm: float) -> None:
    """The OPL zero sits EPD in front of the first surface, so it moves with the aperture.

    This is the fact that makes an *undeclared* Optiland OPL dangerous, and it
    is the explanation for M1's `opd = 12` at a 10 mm separation with EPD 2.0.
    """
    separation_mm = 10.0
    optic = _infinite_object_optic(epd_mm=epd_mm, separation_mm=separation_mm)
    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
    observed = np.asarray(rays.opd, dtype=np.float64)

    positions = np.asarray(
        optiland_backend.to_numpy(optic.surfaces.positions), dtype=np.float64
    ).ravel()
    offset_mm = epd_mm - float(np.min(positions[1:-1]))
    assert np.max(np.abs(observed - (offset_mm + separation_mm))) <= ROUND_OFF_MM

    # Falsifier: if the reference were the first surface, the error would be
    # zero. It is exactly EPD, and it tracks EPD across the sweep.
    assert float(np.max(np.abs(observed - separation_mm))) == pytest.approx(epd_mm, rel=1e-9)


def test_m1_opd_twelve_anomaly_is_fully_explained() -> None:
    """Reproduce M1's exact observation and account for every millimetre of it."""
    optic = _infinite_object_optic(epd_mm=2.0, separation_mm=10.0)
    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
    assert np.allclose(np.asarray(rays.opd, dtype=np.float64), 12.0, atol=ROUND_OFF_MM)


def test_finite_object_opl_zero_is_the_object_plane() -> None:
    """With a finite object no aperture offset applies; marginal rays are slant paths."""
    object_distance_mm, separation_mm = 50.0, 10.0
    optic = Optic("opd-convention-finite-object")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=object_distance_mm)
    optic.surfaces.add(index=1, radius=optiland_backend.inf, thickness=separation_mm, is_stop=True)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)

    rays = optic.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
    observed = np.asarray(rays.opd, dtype=np.float64)
    axial_span = object_distance_mm + separation_mm
    x = np.asarray(rays.x, dtype=np.float64)
    y = np.asarray(rays.y, dtype=np.float64)

    # Planar surfaces with air on both sides: no ray bends, so every ray's OPL
    # is one straight segment from the on-axis object point.
    predicted = np.sqrt(axial_span**2 + x**2 + y**2)
    assert np.max(np.abs(observed - predicted)) <= 1e-10

    # Falsifier: applying the infinite-object EPD offset here is wrong by EPD.
    assert abs(float(np.min(observed)) - (axial_span + 2.0)) == pytest.approx(2.0, rel=1e-9)


def test_opd_scales_with_the_geometry_unit() -> None:
    """Scaling the prescription scales `opd` exactly, so it carries the geometry unit."""

    def observe(distance_mm: float) -> np.ndarray:
        L = np.array([-0.03, 0.0, 0.02])
        M = np.array([0.01, -0.025, 0.015])
        N = np.sqrt(1.0 - L**2 - M**2)
        rays = _rays(np.zeros(3), np.zeros(3), np.zeros(3), L, M, N)
        _free_space_optic(distance_mm).surfaces[1].trace(rays)
        return np.asarray(rays.opd, dtype=np.float64)

    ratio = observe(100.0) / observe(10.0)
    assert np.allclose(ratio, 10.0, atol=1e-12)


def test_paraxial_surface_is_not_an_admissible_opl_source() -> None:
    """`surface_type="paraxial"` yields an OPL with a spurious -h^2/(2f) term.

    A plane wave through a perfect lens must reach the focus with equal optical
    path at every pupil height. The paraxial interaction model subtracts
    `(x^2+y^2)/(2f)` -- exactly the paraxial excess of `sqrt(f^2+h^2)` -- but it
    also sets `rays.N = copysign(1, N)` and leaves the direction un-normalized,
    so the following propagation adds the *axial* distance instead of the true
    Euclidean one. The intended cancellation therefore never happens and the
    subtraction survives in full.

    Consequence for M3.3: a diffraction-limited reference system must be built
    from real refractive surfaces. At f=50 mm and h=6 mm the artifact is
    0.36 mm, which is ~655 waves at 550 nm -- a defocus, not a rounding error.
    """
    focal_mm = 50.0
    heights = np.array([0.0, 2.0, 4.0, 6.0])
    zeros = np.zeros_like(heights)
    rays = _rays(heights, zeros, zeros, zeros, zeros, np.ones_like(heights))

    optic = Optic("opd-convention-paraxial")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=focal_mm, thickness=focal_mm, is_stop=True
    )
    optic.surfaces.add(index=2, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.trace(rays, skip=1)

    observed = np.asarray(rays.opd, dtype=np.float64)
    relative = observed - observed[0]

    # The departure from flat is the bare paraxial lens term, to round-off.
    assert np.max(np.abs(relative - (-(heights**2) / (2.0 * focal_mm)))) <= 1e-12

    # And it is emphatically not flat, which is the point of the test.
    assert float(np.max(np.abs(relative))) > 0.3


def test_real_singlet_opl_spread_is_physical_aberration() -> None:
    """Control for the previous test: real surfaces give an h^4 spread, not h^2."""
    n, radius_mm, thickness_mm = 1.5168, 25.0, 2.0
    efl_mm = radius_mm / (n - 1.0)
    bfl_mm = efl_mm - thickness_mm / n

    heights = np.array([2.0, 4.0, 6.0])
    all_heights = np.concatenate(([0.0], heights))
    zeros = np.zeros_like(all_heights)
    rays = _rays(all_heights, zeros, zeros, zeros, zeros, np.ones_like(all_heights))

    optic = Optic("opd-convention-real-singlet")
    optic.surfaces.add(index=0, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        radius=radius_mm,
        thickness=thickness_mm,
        material=IdealMaterial(n=n),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=optiland_backend.inf, thickness=bfl_mm)
    optic.surfaces.add(index=3, radius=optiland_backend.inf, thickness=0.0)
    optic.surfaces.trace(rays, skip=1)

    observed = np.asarray(rays.opd, dtype=np.float64)
    relative = np.abs(observed[1:] - observed[0])

    # Spherical aberration of a plano-convex singlet at its paraxial focus
    # scales as h^4: doubling the height multiplies the departure by ~16.
    assert relative[1] / relative[0] == pytest.approx(16.0, rel=0.15)
    assert relative[2] / relative[1] == pytest.approx((6.0 / 4.0) ** 4, rel=0.15)

    # A real surface stays far below the paraxial artifact at the same heights.
    assert float(np.max(relative)) < 0.05


def test_matches_recorded_probe_evidence() -> None:
    """Guard the recorded evidence file against silent drift in the pinned install."""
    expected = load_probe_expected("optiland", "opd_convention_probe")
    cases = expected["cases"]

    assert "be.abs(t * self.material_pre.n(rays.w))" in expected["accumulation_site"]
    assert cases["free_space_oblique"]["max_abs_error_slant_mm"] <= ROUND_OFF_MM
    assert cases["index_weighting"]["abs_error_optical_mm"] <= ROUND_OFF_MM
    assert cases["finite_object"]["max_abs_error_vs_per_ray_slant_mm"] <= 1e-10
    assert cases["geometry_scale"]["max_abs_ratio_error"] <= 1e-12
    for sweep in cases["trace_launch_plane"]["sweeps"]:
        assert sweep["max_abs_error_mm"] <= ROUND_OFF_MM
        assert sweep["abs_error_if_reference_were_first_surface_mm"] == pytest.approx(
            sweep["EPD_mm"], rel=1e-9
        )
    assert cases["paraxial_surface_breaks_opl"]["max_abs_departure_from_flat_mm"] > 0.3
    assert cases["wavefront_sign_convention"]["optiland_expression_present"] is True
