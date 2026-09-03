"""R11.2: the end-to-end PSF gate, and what the frozen L2-PSF-01 numbers say.

CHE-198. The first ticket where the whole chain runs --
`trace -> ray_to_scalar -> psf` -- and therefore the first place an error
anywhere in R04-R07 or R11.1 shows up as a single number. That is what makes it
valuable and what makes attribution hard, so almost everything here is an
attribution rather than a threshold.

**The primary gate does not close, and it is reported rather than widened.** The
frozen L2-PSF-01 bundle recorded `fft_oracle_intensity_relative_l2 = 1.0e-3`
against an observed `2.2072391812867093e-3` on the real traced M3-SINGLET-REF
singlet, with the disposition ATTRIBUTED AND UNMET: 94.8 % of the residual is an
Airy-scale offset that O1's own paraxial, aberration-free assumption cannot pin
on this system, and the 5.2 % it can speak about is inside the gate. This file
reaches the same verdict from a **completely independent implementation** -- a
different tracer call, a different coupler, a different measurement and an
oracle written fresh -- and adds one term the frozen record did not name.

What is standing on what
------------------------
This number is unattributable without the upstream gates, so they are named:

| stage | what it rests on |
| --- | --- |
| the prescription | R04, `tests/fixtures/systems.py`, transcribed surface by surface |
| the trace and its OPL reference | R05, R05.2 |
| the wavelet sum and its launch amplitude | R07, R07.1, R07.3 -- the `lambda R`
  plateau oracle is re-run here at the end of the chain |
| the ray-side advance | R09, cross-checked here against the tracer's own image-surface trace |
| the measurement | R11.1 |

The propagation leg is missing, on purpose
------------------------------------------
The ticket's chain reads `trace -> ray_to_scalar -> propagate -> psf`. **That
chain is refused by the tree's own contracts**, and the refusal is correct:
`ray_to_scalar` stamps `surface_only` on the field it emits -- the reconstruction
carries no `exp(i k r^2 / 2R)` curvature term -- and `backends.chromatix.propagate`
refuses a field carrying that flag. It is invisible in `|U|^2`, which is why the
intensity gate converges anyway, and it is not invisible to a caller who
propagates further. The frozen bundle's own caution #2 says the same thing in
prose; here it is executable, and pinned below.

So propagation happens on the **rays**, before the handoff, where R09 makes it
exact. `test_the_propagation_leg_happens_on_the_rays` runs the four-stage chain
that way and finds the two routes to the sensor plane agree to 4e-14.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from fixtures.systems import (
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    singlet_ref,
    singlet_source,
)
from oracles import (
    airy_first_null_radius_m,
    airy_psf_on_grid,
    disc_mask,
    measure_first_null_radius_m,
    numerical_aperture_from_geometry,
    peak_normalized_disc_relative_l2,
    pixels_per_airy_radius,
    radial_profile,
    relative_l2_intensity,
)
from ray_support import (
    FOCAL_M,
    WAVELENGTH_M,
    a_surface,
    converging_bundle,
    focal_peak_oracle,
    hexapolar_disc,
    plateau_radius_m,
)

from backends.chromatix import propagate
from backends.optiland import trace
from couplers import ray_to_scalar
from measurements import psf
from operators import propagate_rays
from representations import ContractError, RayBundle, ReferenceSurface

#: The frozen gate and the frozen observation, from
#: `pre-rewrite-2026-08-30:benchmarks/physics/L2-PSF-01/tolerances.yaml`.
#: **Neither is widened anywhere in this file.**
FROZEN_GATE = 1.0e-3
FROZEN_OBSERVED = 2.2072391812867093e-3

#: The two defensible NA declarations CHE-117 measured for this system, and the
#: best fit it found between them. Quoted, not recomputed.
FROZEN_NA_PARAXIAL = 0.0515667
FROZEN_NA_MAX_TRACED_COSINE = 0.0517163
FROZEN_NA_BEST_FIT = 0.0516457
FROZEN_RESIDUAL_AT_BEST_FIT = 7.021e-4

SEMI_APERTURE_M = SINGLET_ENTRANCE_PUPIL_DIAMETER_MM / 2.0 * 1e-3
SENSOR_PITCH_M = (1.0e-6, 1.0e-6)
SENSOR_SHAPE = (128, 128)

SYNTHETIC_RADIUS_M = 0.25e-3


def a_traced_bundle(*, rings: int = 64, reference_surface: str = "image_surface") -> RayBundle:
    """M3-SINGLET-REF, on axis, at the reference wavelength. The real thing."""
    return trace(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0), wavelength_um=0.55),
        sampling={"num_rings": rings, "reference_surface": reference_surface},
        execution={"device": "cpu", "precision": "fp64"},
    )


def reconstructed_intensity(
    rays: RayBundle,
    *,
    shape: tuple[int, int] = SENSOR_SHAPE,
    sample_pitch_m: tuple[float, float] = SENSOR_PITCH_M,
) -> Any:
    """The chain, in one line: rays -> field -> `|u|^2`, unscaled."""
    field, _ = ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=sample_pitch_m)
    return np.asarray(psf(field, normalization="raw").intensity)


def diffraction_limited_bundle(
    *,
    rings: int,
    radius_m: float = SYNTHETIC_RADIUS_M,
    focal_m: float = FOCAL_M,
    condition: str = "tangent",
) -> RayBundle:
    """A **perfect** spherical converging wavefront: constant optical path to the focus.

    `condition` is the pupil map, and the review of this ticket established that it
    is not a detail: it moves the residual against O1 by 7.4x at fixed NA.

    * `"tangent"` -- `rho = f tan(theta)`, which is what uniform area sampling of a
      flat pupil plus "point every ray at the focus" produces. The default, because
      it is what a naive construction gives.
    * `"sine"` -- `rho = f sin(theta)`, the aplanatic map an imaging system obeying
      the Abbe sine condition has, with its `sqrt(cos theta)` amplitude
      apodization. Much closer to what O1 assumes.

    The configuration in which O1's assumptions hold apart from paraxiality, and
    the one CHE-38 used to show that the coupler itself clears the gate. It is
    deliberately *not* `ray_support.converging_bundle`, whose optical path is
    `hypot(rho, R)` from a **flat** pupil-plane wavefront -- that is a defocused
    Fresnel configuration whose closed form is `focal_peak_oracle`, and comparing
    it to an Airy pattern gives a relative L2 of 5.0 and a first null 33 % short.
    The distinction is one line of optical path and it is the difference between
    an oracle test and nonsense; `test_the_flat_wavefront_bundle_is_not_the_airy_case`
    pins it so the next reader does not have to rediscover it.
    """
    rho, phi, area = hexapolar_disc(rings, radius_m)
    count = int(rho.size)
    if condition == "tangent":
        directions = np.column_stack(
            [-rho * np.cos(phi), -rho * np.sin(phi), np.full(count, focal_m)]
        )
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        amplitude = np.ones(count)
    elif condition == "sine":
        transverse = rho / focal_m
        axial = np.sqrt(1.0 - transverse**2)
        directions = np.column_stack(
            [-transverse * np.cos(phi), -transverse * np.sin(phi), axial]
        )
        amplitude = np.sqrt(axial)
    else:
        raise ValueError(f"condition={condition!r} is not 'tangent' or 'sine'")
    return RayBundle(
        positions_m=np.column_stack(
            [np.zeros(count), np.zeros(count), np.full(count, focal_m)]
        ),
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface("focus", z_m=focal_m),
        amplitude=amplitude.astype(np.complex128),
        optical_path_m=np.zeros(count),
        optical_path_reference="the converging spherical wavefront, constant to the focus",
        measure_weight=area,
        measure_kind="quadrature_area_m2",
    )


def synthetic_numerical_aperture(
    radius_m: float = SYNTHETIC_RADIUS_M, *, condition: str = "tangent"
) -> float:
    """The marginal ray's `sin(theta)` under each pupil map, which is what O1 wants."""
    if condition == "sine":
        return radius_m / FOCAL_M
    return numerical_aperture_from_geometry(semi_aperture_m=radius_m, distance_m=FOCAL_M)


def disc_coverage(intensity: Any, *, numerical_aperture: float, sample_pitch_m) -> float:
    """What fraction of the window the 5-Airy disc actually selects.

    **The disc is a cap on the compared region, not a guarantee of one.** On the
    traced-singlet grid it bites, selecting 20 % of the window. On the synthetic
    grids, which are sized to a few Airy radii so the ray sum stays cheap, the
    window is *inside* the disc and the comparison is over the whole window. That
    is a difference from the frozen bundle's region control and it is small --
    widening the aberration-free window until the disc does bite moves its metric
    from 8.8647e-4 to 8.8614e-4, 0.04 % -- but it is reported rather than left for
    a reader to discover, and every test below states which case it is in.
    """
    airy_radius = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    return float(
        disc_mask(
            shape=np.asarray(intensity).shape,
            sample_pitch_m=sample_pitch_m,
            radius_m=5.0 * airy_radius,
        ).mean()
    )


def gate_metric(intensity: Any, *, numerical_aperture: float, sample_pitch_m: tuple[float, float]):
    """The frozen metric: peak-normalized relative L2 over the 5-Airy-radius disc."""
    airy_radius = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    oracle = airy_psf_on_grid(
        shape=intensity.shape,
        sample_pitch_m=sample_pitch_m,
        numerical_aperture=numerical_aperture,
        wavelength_m=WAVELENGTH_M,
    )
    return peak_normalized_disc_relative_l2(
        intensity, oracle, sample_pitch_m=sample_pitch_m, radius_m=5.0 * airy_radius
    )


# ---------------------------------------------------------------------------
# 1. The oracle, checked before it is used to check anything
# ---------------------------------------------------------------------------


def test_the_airy_oracle_is_the_textbook_one() -> None:
    """An oracle nobody verified is an assumption with a docstring.

    Three independent handles: `J1` vanishes at the tabulated 3.8317, the classical
    `0.61 lambda / NA` follows, and the pattern is 1 at the origin and has its
    first zero exactly at the radius the formula names.
    """
    from oracles import AIRY_FIRST_NULL_V, airy_intensity_at_radius
    from scipy.optimize import brentq
    from scipy.special import j1

    assert brentq(j1, 3.0, 4.5) == pytest.approx(AIRY_FIRST_NULL_V, rel=1e-12)

    numerical_aperture = 0.05
    radius = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    assert radius == pytest.approx(0.6098 * WAVELENGTH_M / numerical_aperture, rel=1e-4)
    assert float(
        airy_intensity_at_radius(
            0.0, numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
        )
    ) == pytest.approx(1.0)
    assert float(
        airy_intensity_at_radius(
            radius, numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
        )
    ) < 1e-24
    # ...and the two landmarks past the first null, which a formula that had the
    # right first zero and the wrong shape would miss: the second zero sits at
    # `v = 7.0156` and the first ring peaks at `v = 5.1356` with **1.75 %** of the
    # central intensity. Both are handbook values for `(2 J1(v) / v)^2`.
    scale = WAVELENGTH_M / (2.0 * math.pi * numerical_aperture)
    assert float(
        airy_intensity_at_radius(
            7.015586669815619 * scale,
            numerical_aperture=numerical_aperture,
            wavelength_m=WAVELENGTH_M,
        )
    ) < 1e-24
    assert float(
        airy_intensity_at_radius(
            5.135622301840683 * scale,
            numerical_aperture=numerical_aperture,
            wavelength_m=WAVELENGTH_M,
        )
    ) == pytest.approx(0.017498, rel=1e-4)

    # The sampled grid carries the same landmarks, so `airy_psf_on_grid` places the
    # pattern on the `n // 2` origin rather than a half-pixel off it.
    pitch = (radius / 200.0, radius / 200.0)
    shape = (1601, 1601)
    grid = airy_psf_on_grid(
        shape=shape,
        sample_pitch_m=pitch,
        numerical_aperture=numerical_aperture,
        wavelength_m=WAVELENGTH_M,
    )
    assert grid[800, 800] == 1.0
    inside = disc_mask(shape=shape, sample_pitch_m=pitch, radius_m=radius)
    assert 0.85 < float(grid[inside].sum() / grid.sum()) < 0.88
    profile_r, profile_i = radial_profile(grid, sample_pitch_m=pitch, bin_width_m=pitch[0])
    assert profile_i[0] == pytest.approx(1.0, abs=2e-5)
    # To within one bin: `radial_profile` reports bin *centres*, so half a bin of
    # offset is the definition and not an error.
    assert profile_r[np.argmin(profile_i[: int(1.2 * radius / pitch[0])])] == pytest.approx(
        radius, abs=pitch[0]
    )


def test_the_peak_normalized_metric_cannot_see_a_scale_and_the_plain_one_can() -> None:
    """Why R11.2 criterion 2 exists at all.

    The frozen gate metric divides both inputs by their own peaks, so a global
    factor cancels exactly -- CHE-117 measured 1e-14 of movement under a `2^64`
    rescale. Restated here as an identity rather than a measurement, and set
    against `relative_l2_intensity`, which sees the same factor immediately.
    """
    numerical_aperture = synthetic_numerical_aperture()
    pitch = (0.3e-6, 0.3e-6)
    truth = airy_psf_on_grid(
        shape=(64, 64),
        sample_pitch_m=pitch,
        numerical_aperture=numerical_aperture,
        wavelength_m=WAVELENGTH_M,
    )
    rescaled = truth * 2.0**64
    assert peak_normalized_disc_relative_l2(
        rescaled, truth, sample_pitch_m=pitch, radius_m=1e-4
    ) == pytest.approx(0.0, abs=1e-14)
    assert relative_l2_intensity(rescaled, truth) == pytest.approx(2.0**64 - 1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Criterion 2 and 3: the un-normalized check, at the end of the chain
# ---------------------------------------------------------------------------


def test_the_lambda_r_plateau_oracle_survives_the_measurement() -> None:
    """Criteria 2 and 3. **The absolute scale, checked with nothing normalized.**

    R07.1's stationary-phase oracle is the one direct handle on the launch
    amplitude: for a uniform disc of radius `a` converging at `R` from a flat
    wavefront, the on-axis field is `lambda R |1 - exp(i pi a^2 / lambda R)|` in
    closed form, and at `a^2 = lambda R / 3` the truncation factor is exactly 1,
    so the answer is a bare `lambda R`.

    Re-run here at the *end* of the chain rather than on the field: the quantity
    compared is `PsfResult.raw_peak_intensity`, which is the number a caller
    checking R07's absolute scale actually reads. It reproduces
    `(lambda R)^2` to **4.0e-6 relative**.

    Peak-normalized comparisons cannot certify R07 -- that is this ticket's whole
    criterion 2 -- and this is the check that can. An omitted per-ray area weight
    is an exact constant factor; it moves this number and moves nothing in the
    gate metric.
    """
    radius = plateau_radius_m()
    rays, _ = converging_bundle(rings=200, radius_m=radius)
    field, _ = ray_to_scalar(rays, grid_shape=(8, 8), sample_pitch_m=(0.5e-6, 0.5e-6))
    measured = psf(field, normalization="raw")

    expected_amplitude = focal_peak_oracle(radius_m=radius)
    assert expected_amplitude == pytest.approx(WAVELENGTH_M * FOCAL_M, rel=1e-12)
    assert measured.raw_peak_intensity == pytest.approx(expected_amplitude**2, rel=1e-5)

    # ...and the same reconstruction with the quadrature measure stripped misses it
    # by orders of magnitude, which is what makes the check above load-bearing.
    unweighted, _ = converging_bundle(
        rings=200, radius_m=radius, measure_kind="importance_weight"
    )
    stripped, _ = ray_to_scalar(unweighted, grid_shape=(8, 8), sample_pitch_m=(0.5e-6, 0.5e-6))
    stripped_peak = psf(stripped, normalization="raw").raw_peak_intensity
    assert stripped_peak / expected_amplitude**2 > 1e6


# ---------------------------------------------------------------------------
# 3. The aberration-free arm: where O1's assumptions hold
# ---------------------------------------------------------------------------


def test_an_aberration_free_wavefront_lands_inside_the_frozen_gate() -> None:
    """The coupler and the measurement are not what the singlet residual is.

    A perfect spherical converging wavefront at the singlet's own NA, reconstructed
    and measured through the same code the real trace goes through, lands at
    **8.86e-4** against O1 -- inside the frozen 1.0e-3, on a window wide enough that
    the 5-Airy disc actually restricts the comparison.

    Inside the gate is not the same as the gate closing, and this file is about a
    gate that does not close. What this establishes is narrower and is the thing
    attribution needs: whatever the traced singlet's 2.1e-3 is, it is not the
    coupler and it is not the measurement, because those are the same code here.
    CHE-38's synthetic configuration reached 4.07e-4 by the same argument on a
    different aperture and ring count; the agreement claimed is of the
    *conclusion*, not of the number.
    """
    pitch = (0.6e-6, 0.6e-6)
    numerical_aperture = synthetic_numerical_aperture()
    intensity = reconstructed_intensity(
        diffraction_limited_bundle(rings=64), shape=(120, 120), sample_pitch_m=pitch
    )
    assert 0.5 < disc_coverage(
        intensity, numerical_aperture=numerical_aperture, sample_pitch_m=pitch
    ) < 0.8
    metric = gate_metric(
        intensity, numerical_aperture=numerical_aperture, sample_pitch_m=pitch
    )
    assert metric == pytest.approx(8.865e-4, rel=2e-3)
    assert metric < FROZEN_GATE


def test_the_flat_wavefront_bundle_is_not_the_airy_case() -> None:
    """`ray_support.converging_bundle` is deliberately defocused, and it matters.

    Its optical path is `hypot(rho, R)` measured from a **flat** pupil-plane
    wavefront, so the outer rays arrive with the full Fresnel quadratic phase --
    which is exactly why its closed form is `focal_peak_oracle` and not an Airy
    peak. Handing it to an Airy oracle gives a relative L2 above 1 and a first null
    a third short, and it looks like a broken coupler. It is not: it is the wrong
    oracle for that bundle.
    """
    rays, _ = converging_bundle(rings=64, radius_m=SYNTHETIC_RADIUS_M)
    pitch = (0.6e-6, 0.6e-6)
    intensity = reconstructed_intensity(rays, shape=(66, 66), sample_pitch_m=pitch)
    assert gate_metric(
        intensity, numerical_aperture=synthetic_numerical_aperture(), sample_pitch_m=pitch
    ) > 1.0


# ---------------------------------------------------------------------------
# 4. Criterion 1 and 6: the real traced singlet, and the gate that does not close
# ---------------------------------------------------------------------------


def test_the_traced_singlets_own_na_ambiguity_reproduces() -> None:
    """CHE-117's central measurement, from an independent tracer call.

    The largest transverse direction cosine this trace produces is
    **0.05171631827291936**; CHE-117 recorded 0.0517163 for the same system. Seven
    digits, through a different solver call, a different adapter and a different
    bundle type.

    That number is one of the system's two defensible NA declarations. The other
    is the paraxial geometric `a / sqrt(a^2 + R^2)`, and the point of quoting both
    is that they disagree by more than the gate they are used to decide.
    """
    rays = a_traced_bundle()
    largest_cosine = float(np.abs(np.asarray(rays.directions)[:, 0]).max())
    assert largest_cosine == pytest.approx(FROZEN_NA_MAX_TRACED_COSINE, abs=1e-7)

    paraxial = numerical_aperture_from_geometry(
        semi_aperture_m=SEMI_APERTURE_M, distance_m=SINGLET_EFFECTIVE_FOCAL_LENGTH_MM * 1e-3
    )
    assert abs(largest_cosine - paraxial) > 2e-4
    assert paraxial == pytest.approx(0.0514780, abs=1e-6)


def test_the_primary_gate_is_attributed_and_unmet() -> None:
    """**Criterion 6: the open gate is reported. Nothing here is widened.**

    The chain on the real traced singlet, measured against O1 over the 5-Airy
    disc, at each of the NA declarations the system leaves open:

    | NA | declaration | metric |
    | --- | --- | --- |
    | 0.0514780 | paraxial `a / sqrt(a^2 + R^2)` at the EFL | 5.13e-3 |
    | 0.0515667 | CHE-117's paraxial geometric | 2.57e-3 |
    | **0.0517163** | largest traced direction cosine | **2.13e-3** |
    | 0.0516457 | CHE-117's best fit | 7.79e-4 |

    The frozen observation is `2.2072391812867093e-3` at 512 rings; this reads
    2.13e-3 at 64 rings and 2.19e-3 at 128 (the slow test below), against a frozen
    record that declares the weighted arm "flat to 0.87 % from 49,537 to 3,148,801
    rays". 2.19e-3 is 0.83 % from the frozen value -- inside the flatness the
    record itself states, from an implementation that shares no code with it.

    **The gate stands unmet at 1.0e-3, and the spread across the rows above is
    why.** The choice of NA, which the system's own geometry does not settle,
    moves the metric by more than twice the gate. The best-fit row reproduces
    CHE-117's 7.021e-4 to 11 %, and it must **not** be read as the gate closing:
    fitting the oracle's scale to the field under test removes the independence
    that makes O1 admissible at all, and a scale-fitted Airy pattern cannot reject
    a wrong answer of the same shape.
    """
    intensity = reconstructed_intensity(a_traced_bundle())
    metrics = {
        name: gate_metric(
            intensity, numerical_aperture=value, sample_pitch_m=SENSOR_PITCH_M
        )
        for name, value in (
            ("paraxial_at_efl", 0.0514780),
            ("frozen_paraxial", FROZEN_NA_PARAXIAL),
            ("max_traced_cosine", FROZEN_NA_MAX_TRACED_COSINE),
            ("frozen_best_fit", FROZEN_NA_BEST_FIT),
        )
    }
    assert metrics["max_traced_cosine"] == pytest.approx(2.133e-3, rel=5e-3)
    assert metrics["frozen_paraxial"] == pytest.approx(2.565e-3, rel=5e-3)
    assert metrics["paraxial_at_efl"] == pytest.approx(5.133e-3, rel=5e-3)
    assert metrics["frozen_best_fit"] == pytest.approx(7.790e-4, rel=5e-3)

    # The gate, unmet, at every declaration the system's own geometry admits.
    assert metrics["max_traced_cosine"] > FROZEN_GATE
    assert metrics["frozen_paraxial"] > FROZEN_GATE
    # ...and the spread across those two alone exceeds a third of the gate.
    assert abs(metrics["frozen_paraxial"] - metrics["max_traced_cosine"]) > 0.3 * FROZEN_GATE
    # The best fit lands between them and inside the gate, which is the shape of
    # CHE-117's finding and is not a pass.
    assert FROZEN_NA_PARAXIAL < FROZEN_NA_BEST_FIT < FROZEN_NA_MAX_TRACED_COSINE
    # Characterization again: 11 % from CHE-117's 7.021e-4, on a different ray
    # count and a different implementation. It says the best-fit row lands where
    # CHE-117 put it, not that this reproduces that number.
    assert metrics["frozen_best_fit"] == pytest.approx(FROZEN_RESIDUAL_AT_BEST_FIT, rel=0.15)


def test_the_synthetic_floor_is_the_pupil_map_and_not_the_coupler() -> None:
    """The term the frozen attribution does not name -- and it is **O1's, not ours**.

    On an *exactly* spherical, aberration-free converging wavefront there is
    nothing left for a real oracle to find, and O1 still disagrees, because O1 is a
    function of one number and knows nothing about how the pupil maps radius to
    angle. At the singlet's NA and 64 rings:

    | pupil map | NA | metric vs O1 |
    | --- | --- | --- |
    | tangent, `rho = f tan(theta)` | 0.051611 | **8.86e-4** |
    | aplanatic, `rho = f sin(theta)` with `sqrt(cos theta)` | 0.051680 | **1.19e-4** |

    A factor of **7.4**, at the same NA, from one line of pupil geometry. So the
    first draft of this test was wrong in its attribution: it called the 8.9e-4 "O1's
    paraxiality" and read 89 % of the gate off it, when 87 % of that figure is the
    tangent-versus-sine map and only the remainder is paraxiality as such. **Nothing
    here measures which map M3-SINGLET-REF has**, so neither row transfers to the
    traced system, and this is *not* a sharper route to CHE-117's conclusion -- it
    is the same physics, the direction-cosine scale, reached on a configuration
    whose map was chosen rather than measured.

    What it does establish, and what attribution needs, is that the floor moves with
    a property of the *wavefront* and is therefore not a constant of the coupler,
    the quadrature or the measurement. The `NA^2` collapse below is the second half
    of that argument; the fitted exponent needs converged quadrature at the small
    apertures and is in the slow test.
    """
    pitch = (0.6e-6, 0.6e-6)
    by_map = {}
    for condition in ("tangent", "sine"):
        aperture = synthetic_numerical_aperture(condition=condition)
        by_map[condition] = gate_metric(
            reconstructed_intensity(
                diffraction_limited_bundle(rings=64, condition=condition),
                shape=(120, 120),
                sample_pitch_m=pitch,
            ),
            numerical_aperture=aperture,
            sample_pitch_m=pitch,
        )
    assert by_map["tangent"] == pytest.approx(8.865e-4, rel=5e-3)
    assert by_map["sine"] == pytest.approx(1.279e-4, rel=5e-3)
    assert by_map["tangent"] > 6.0 * by_map["sine"]
    # ...and the two apertures agree to a part in 1e-3, so this is the map and not
    # a difference in NA sneaking in.
    assert synthetic_numerical_aperture(condition="sine") == pytest.approx(
        synthetic_numerical_aperture(condition="tangent"), rel=2e-3
    )

def test_the_synthetic_floor_collapses_with_numerical_aperture() -> None:
    """The second half: whatever the floor is, it is `O(NA^2)` and so it is O1's.

    A constant of the coupler, of the quadrature or of the measurement would not
    care about the aperture. This one does, quadratically -- 8.9e-4 at NA 0.0516
    against 1.5e-4 at NA 0.0103 -- which is the signature of the leading correction
    to a paraxial oracle and of nothing under test.
    """
    pitch_scale = 20.0
    metrics = {}
    for radius in (0.25e-3, 0.05e-3):
        numerical_aperture = synthetic_numerical_aperture(radius)
        airy_radius = airy_first_null_radius_m(
            numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
        )
        pitch = (airy_radius / pitch_scale, airy_radius / pitch_scale)
        shape = (120, 120)
        intensity = reconstructed_intensity(
            diffraction_limited_bundle(rings=64, radius_m=radius),
            shape=shape,
            sample_pitch_m=pitch,
        )
        metrics[radius] = gate_metric(
            intensity, numerical_aperture=numerical_aperture, sample_pitch_m=pitch
        )
    # A 5x drop in NA must cost far more than 5x in residual. The observed ratio at
    # this ray count is 5.7 rather than the quadratic 20, and the difference is the
    # hexapolar quadrature: at 64 rings the small-aperture point still carries
    # 1.5e-4 of it, against a paraxial term of 4.1e-5. The slow test measures both
    # separately and fits the exponent at 2.0; what this one shows is that the
    # residual collapses with NA, which no constant of the coupler would do.
    assert metrics[0.25e-3] > 4.0 * metrics[0.05e-3]
    assert metrics[0.05e-3] < 2e-4
    assert metrics[0.25e-3] > 0.8 * FROZEN_GATE


# ---------------------------------------------------------------------------
# 5. Criterion 4: sampling, before physics
# ---------------------------------------------------------------------------


def test_the_shape_residual_does_not_move_with_the_grid() -> None:
    """Criterion 4, first half. Sampling contributes **nothing** to the metric.

    The same bundle on two pitches differing by 1.67x gives the same number to six
    significant figures. CHE-117 found the same thing over an 8x refinement and
    recorded "identical to ten significant figures"; the slow test below carries
    this to 21.6x, 2.4 -> 52.0 samples per Airy radius, and it still does not move.

    That is what makes the residual attributable at all: whatever it is, it is not
    the grid.
    """
    rays = diffraction_limited_bundle(rings=64)
    numerical_aperture = synthetic_numerical_aperture()
    coarse = gate_metric(
        reconstructed_intensity(rays, shape=(40, 40), sample_pitch_m=(1.0e-6, 1.0e-6)),
        numerical_aperture=numerical_aperture,
        sample_pitch_m=(1.0e-6, 1.0e-6),
    )
    fine = gate_metric(
        reconstructed_intensity(rays, shape=(66, 66), sample_pitch_m=(0.6e-6, 0.6e-6)),
        numerical_aperture=numerical_aperture,
        sample_pitch_m=(0.6e-6, 0.6e-6),
    )
    assert coarse == pytest.approx(fine, rel=1e-5)


def test_a_first_null_discrepancy_is_sampling_before_it_is_physics() -> None:
    """Criterion 4, second half, and the frozen finding it restates.

    L2-PSF-01 recorded an apparent **3.6 % first-null shift** that turned out to be
    **2.44 pixels per Airy radius** and not the quadrature weight. The same trap,
    reproduced on a pattern whose *shape* residual is provably grid-independent
    (the test above):

    | px per Airy radius | measured first null | error |
    | --- | --- | --- |
    | 2.41 | 17.27 um | **+166 %** |
    | 4.81 | 6.96 um | +7.07 % |
    | 10.83 | 6.58 um | +1.23 % |

    At 2.4 samples per Airy radius there is no minimum to interpolate and the
    number returned is an artifact of where the samples fell. **A first-null
    discrepancy is a statement about the grid until the grid is shown not to
    matter**, and the two halves of criterion 4 are what shows it.
    """
    rays = diffraction_limited_bundle(rings=64)
    numerical_aperture = synthetic_numerical_aperture()
    airy_radius = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    measured = []
    for pitch_um, shape in ((2.7, (16, 16)), (1.35, (30, 30)), (0.6, (66, 66))):
        pitch = (pitch_um * 1e-6, pitch_um * 1e-6)
        null_radius = measure_first_null_radius_m(
            reconstructed_intensity(rays, shape=shape, sample_pitch_m=pitch),
            sample_pitch_m=pitch,
        )
        measured.append(
            (
                pixels_per_airy_radius(sample_pitch_m=pitch, airy_radius_m=airy_radius),
                null_radius / airy_radius - 1.0,
            )
        )
    sampling = [round(value, 2) for value, _ in measured]
    errors = [error for _, error in measured]
    assert sampling == [2.41, 4.81, 10.83]
    assert errors[0] > 1.0, "the coarse grid must be visibly, not subtly, wrong"
    assert errors[1] == pytest.approx(0.0707, rel=0.05)
    assert errors[2] == pytest.approx(0.0123, rel=0.05)
    assert errors[0] > errors[1] > errors[2] > 0.0


# ---------------------------------------------------------------------------
# 6. The chain, and the leg of it the contracts refuse
# ---------------------------------------------------------------------------


def test_the_reconstructed_field_may_not_be_propagated() -> None:
    """`trace -> ray_to_scalar -> propagate -> psf` is refused, and correctly.

    The wavelet sum carries no `exp(i k r^2 / 2R)` curvature term, so the field it
    emits is valid **at its own surface and nowhere else** -- `ray_to_scalar`
    stamps `surface_only` and `backends.chromatix.propagate` refuses it. The frozen
    bundle's caution #2 says exactly this in prose: invisible in `|U|^2`, which is
    why the intensity gate above converges anyway; not invisible to a caller who
    propagates further.

    Pinned rather than worked around. A test that dropped the flag to make the
    four-stage chain run would be asserting the opposite of what R07 measured.
    """
    field, _ = ray_to_scalar(
        a_traced_bundle(rings=8), grid_shape=(32, 32), sample_pitch_m=SENSOR_PITCH_M
    )
    assert "surface_only" in field.validity
    with pytest.raises(ContractError) as raised:
        propagate(
            field,
            distance_m=1e-6,
            model={"method": "asm", "pad_width": 8, "target_surface": "downstream"},
        )
    assert raised.value.code == "REPRESENTATION_INCONSISTENT"
    assert raised.value.declaration == "validity"


def test_the_propagation_leg_happens_on_the_rays() -> None:
    """The four-stage chain, with the propagation where R09 makes it exact.

    `trace(exit_pupil) -> propagate_rays -> ray_to_scalar -> psf` against
    `trace(image_surface) -> ray_to_scalar -> psf`: two independent routes to the
    same plane, one advanced by this repository's operator over 4.8 mm and one
    carried there by the tracer itself. They agree to **4e-14 relative**, which is
    float64 round-off on a sum of 12,481 wavelets.

    That is a real end-to-end statement about R09 and R05 together, and it is the
    honest form of the ticket's chain: the advance is exact on the rays, and
    approximate on a field that has already thrown its curvature away.
    """
    traced = a_traced_bundle()
    at_pupil = a_traced_bundle(reference_surface="exit_pupil")
    assert at_pupil.reference_surface.z_m < traced.reference_surface.z_m

    advanced = propagate_rays(at_pupil, to=traced.reference_surface)
    direct = reconstructed_intensity(traced)
    via_pupil = reconstructed_intensity(advanced)
    assert float(np.linalg.norm(via_pupil - direct) / np.linalg.norm(direct)) < 1e-12
    assert via_pupil.max() == pytest.approx(direct.max(), rel=1e-12)


@pytest.mark.parametrize(("defocus_um", "expected_strehl"), [(20.0, 0.9868), (100.0, 0.7965)])
def test_a_defocused_sensor_loses_peak_the_way_the_geometry_says(
    defocus_um: float, expected_strehl: float
) -> None:
    """The same chain, off the focal plane, as a sanity check with an independent scale.

    The depth of focus of this f/9.7 system is about `lambda / (2 NA^2) = 103 um`,
    so a 100 um advance past the sensor should cost a fifth of the peak and a 20 um
    advance should cost almost nothing. It does. This is a plausibility check with
    a closed-form scale, not a gate.
    """
    traced = a_traced_bundle()
    target = ReferenceSurface(
        name="defocus",
        z_m=traced.reference_surface.z_m + defocus_um * 1e-6,
        medium_index=1.0,
    )
    focused = reconstructed_intensity(traced)
    defocused = reconstructed_intensity(propagate_rays(traced, to=target))
    assert defocused.max() / focused.max() == pytest.approx(expected_strehl, rel=2e-3)

    depth_of_focus_m = WAVELENGTH_M / (2.0 * FROZEN_NA_MAX_TRACED_COSINE**2)
    assert 90e-6 < depth_of_focus_m < 120e-6
    assert (defocus_um * 1e-6 < depth_of_focus_m) == (expected_strehl > 0.5)


def test_no_oracle_and_no_scipy_reaches_production() -> None:
    """Criterion 5, and the sharpest available form of it.

    The Airy pattern needs `scipy.special.j1`, so "is there an oracle in `src/`" has
    a one-token answer: **`src/` does not import scipy at all.** That is stronger
    than a name list and it cannot be evaded by renaming a function, and it stays
    true only as long as nobody puts an analytic reference into production -- which
    is the rule. `tests/` may import it freely; that is where evidence lives.
    """
    import ast as ast_module
    from pathlib import Path as PathType

    src = PathType(__file__).resolve().parents[2] / "src"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        for node in ast_module.walk(ast_module.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast_module.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast_module.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            offenders.extend(
                f"{path.name}: {name}" for name in names if name.split(".")[0] == "scipy"
            )
    assert offenders == [], f"an analytic oracle has reached production: {offenders}"
    assert len(list(src.rglob("*.py"))) > 10, "the walk found almost nothing"


# ---------------------------------------------------------------------------
# 7. The convergence evidence, at full size
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_shape_residual_is_flat_across_a_twentyfold_grid_refinement() -> None:
    """Criterion 4 at full range: 2.41 to 51.99 samples per Airy radius.

    The metric reads 8.1510e-4, 8.1505e-4, 8.1502e-4, 8.1499e-4, 8.1499e-4 -- a
    spread of 1.3e-4 relative across a 21.6x refinement -- while the measured null
    goes +165.8 %, +7.07 %, +1.23 %, +0.157 %, -0.017 % over the same points. The
    two are measured on the *same* five reconstructions, which is what makes the
    contrast an attribution rather than two separate observations.

    Windows here are `6 x` the Airy radius, so the 5-Airy disc does not restrict
    them; the comparison is over the whole window at every rung, which is the same
    convention at all five and is what the flatness claim is about.
    """
    rays = diffraction_limited_bundle(rings=200)
    numerical_aperture = synthetic_numerical_aperture()
    airy_radius = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    shape_metrics, null_errors = [], []
    for pitch_um in (2.7, 1.35, 0.6, 0.3, 0.125):
        pitch = (pitch_um * 1e-6, pitch_um * 1e-6)
        side = int(2 * math.ceil(6 * airy_radius / pitch[0] / 2))
        intensity = reconstructed_intensity(rays, shape=(side, side), sample_pitch_m=pitch)
        shape_metrics.append(
            gate_metric(
                intensity, numerical_aperture=numerical_aperture, sample_pitch_m=pitch
            )
        )
        null_errors.append(
            measure_first_null_radius_m(intensity, sample_pitch_m=pitch) / airy_radius - 1.0
        )
    assert max(shape_metrics) - min(shape_metrics) < 1.3e-7  # 1.3e-4 relative
    assert all(value == pytest.approx(8.150e-4, rel=1e-3) for value in shape_metrics)
    assert null_errors[0] > 1.0
    assert abs(null_errors[-1]) < 1e-3
    assert null_errors[0] > null_errors[1] > null_errors[2] > null_errors[3]


@pytest.mark.slow
def test_the_synthetic_residual_converges_in_ray_count_and_scales_as_na_squared() -> None:
    """The two other terms, so the 8.15e-4 floor is fully attributed.

    Quadrature: 1.184e-3, 8.861e-4, 8.259e-4, 8.121e-4 at 32, 64, 128, 256 rings --
    converging to about 8.1e-4, so the floor is not the ray sampling either.

    Aperture: 8.150e-4, 1.372e-4, 4.138e-5 at NA 0.05161, 0.02067, 0.01034, a
    fitted exponent of **2.0**. That is the floor, and `O(NA^2)` is the signature of
    a paraxial oracle meeting a real cone -- see
    `test_the_synthetic_floor_is_the_pupil_map_and_not_the_coupler` for which part
    of it is paraxiality proper and which is the tangent-condition pupil map this
    bundle happens to have. Both are O1's side of the comparison; neither is the
    coupler's.

    Both ladders are on windows *smaller* than the 5-Airy disc, so the comparison
    is over the whole window -- the ray sum is `O(N_rays x ny x nx)` and a window
    wide enough for the disc to bite at every rung is not affordable here. The
    default-gate test that does size its window for the disc measures 0.04 %
    between the two conventions.
    """
    numerical_aperture = synthetic_numerical_aperture()
    pitch = (0.3e-6, 0.3e-6)
    by_rings = [
        gate_metric(
            reconstructed_intensity(
                diffraction_limited_bundle(rings=rings), shape=(130, 130), sample_pitch_m=pitch
            ),
            numerical_aperture=numerical_aperture,
            sample_pitch_m=pitch,
        )
        for rings in (32, 64, 128, 256)
    ]
    assert by_rings[0] > by_rings[1] > by_rings[2] > by_rings[3]
    assert by_rings[-1] == pytest.approx(8.121e-4, rel=5e-3)
    assert abs(by_rings[-1] - by_rings[-2]) < 0.02 * by_rings[-1]

    apertures, residuals = [], []
    for radius in (0.25e-3, 0.10e-3, 0.05e-3):
        aperture_na = synthetic_numerical_aperture(radius)
        airy_radius = airy_first_null_radius_m(
            numerical_aperture=aperture_na, wavelength_m=WAVELENGTH_M
        )
        fine = (airy_radius / 20.0, airy_radius / 20.0)
        apertures.append(aperture_na)
        residuals.append(
            gate_metric(
                reconstructed_intensity(
                    diffraction_limited_bundle(rings=200, radius_m=radius),
                    shape=(120, 120),
                    sample_pitch_m=fine,
                ),
                numerical_aperture=aperture_na,
                sample_pitch_m=fine,
            )
        )
    exponent = float(
        np.polyfit(np.log(apertures), np.log(residuals), 1)[0]
    )
    assert exponent == pytest.approx(2.0, abs=0.15), f"O1's error should be O(NA^2), got {exponent}"


@pytest.mark.slow
def test_the_frozen_observation_reproduces_inside_its_declared_flatness() -> None:
    """Criterion 1, as precisely as an independent implementation can state it.

    At 128 rings (49,537 rays) the metric against O1 at the largest-traced-cosine
    NA reads **2.189e-3**. The frozen observation is `2.2072391812867093e-3` at 512
    rings, and the frozen record declares the weighted arm "flat to 0.87 % from
    49,537 to 3,148,801 rays". The difference is **0.83 %** -- inside the flatness
    the record states for exactly this ray count, from an implementation that
    shares no code with the one that produced it.

    Bit-identity is not claimed and is not available: this is a different tracer
    call, a different coupler, a different measurement and a fresh oracle. What
    reproduces is the number to three significant figures and, above, the whole
    attribution.
    """
    intensity = reconstructed_intensity(a_traced_bundle(rings=128))
    metric = gate_metric(
        intensity,
        numerical_aperture=FROZEN_NA_MAX_TRACED_COSINE,
        sample_pitch_m=SENSOR_PITCH_M,
    )
    assert metric == pytest.approx(2.189e-3, rel=2e-3)
    # **Characterization, not a gate.** 0.03 is a band for a cross-implementation
    # reproduction and is not the frozen record's 0.87 % ray-count flatness, which
    # is a convergence property of that implementation and is quoted above as
    # context for why 0.83 % is the *expected* size of the difference rather than
    # as the criterion. Tying the assertion to 0.0087 would make a numpy or Optiland
    # bump look like a physics regression, and the cheapest repair would look like
    # widening a frozen number.
    assert abs(metric / FROZEN_OBSERVED - 1.0) < 0.03
    assert metric > FROZEN_GATE
