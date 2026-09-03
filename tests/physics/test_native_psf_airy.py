"""R16.1's analytic gate: the native FFT PSF against the Airy pattern.

CHE-236, criterion 3. The one piece of evidence on this path that is **not** the
pinned solver agreeing with itself. `backends.optiland.psf` is a delegation, so
"the FFT PSF matches Optiland" is definitional and `AGENTS.md` forbids treating a
wrapper's agreement with the code it wraps as a correctness gate. What is left that
can decide anything is O1: the closed-form Airy intensity in
`tests/physics/oracles.py`, which shares no code with this path, with the pinned
solver, or with the `ray_to_scalar` chain R11.2 measured against the same oracle.

The gate metric and its threshold are the frozen ones and **neither is widened
here**: `oracles.peak_normalized_disc_relative_l2` over a 3-Airy-radius disc,
against the frozen `1.0e-3` of L2-PSF-01
(`pre-rewrite-2026-08-30:benchmarks/physics/L2-PSF-01/tolerances.yaml`).

The residual is pupil sampling, and that is measured
-----------------------------------------------------
The number that controls the residual is `num_rays` -- how finely the *pupil* is
sampled -- and not the image-plane grid. Measured on the R05 reference singlet,
peak-normalized L2 over the 3-Airy disc:

| pupil samples | grid | samples / Airy radius | L2 |
| --- | --- | --- | --- |
| 32 | 256 | 10.07 | 1.616e-2 |
| 64 | 512 | 9.91 | 5.661e-3 |
| 128 | 1024 | 9.83 | 1.825e-3 |
| **256** | **1024** | **4.90** | **8.182e-4** |
| 256 | 2048 | 9.80 | 8.185e-4 |

Refining the image grid at fixed pupil sampling moves it in the fourth
significant figure (the last two rows differ by 4e-7 relative across a 2x change
in pitch); doubling the pupil sampling divides it by about three. The likeliest
mechanism is the circular pupil mask -- a binary disc on a `num_rays x num_rays`
grid, whose staircase edge refines exactly with `num_rays` -- and it is the
**dominant** term rather than the whole of the residual: at fixed sampling,
stopping the singlet down to remove its spherical aberration takes 8.18e-4 to
5.63e-4, so about 31 % of the full-aperture residual is wavefront and the rest
converges away. Either way the gate closes at 256 pupil samples, and the
convergence sequence above is why the threshold did not have to move.

**The L2 gate is a shape and convergence gate, and it is blind to the absolute
scale. That is measured, not argued.** The FFT path's sample pitch is
`dx = lambda F/# (num_rays - 1) / grid_size` and the numerical aperture handed to
the oracle is `1 / (2 F/#)`, so the oracle's reduced radial variable is

    v = 2 pi NA r / lambda = pi i (num_rays - 1) / grid_size

for sample `i` -- **`lambda` and `F/#` cancel exactly**. Rescaling the working
F-number and the pitch together by 2 or by 10 moves the metric by 1e-13 relative,
which `test_the_shape_gate_cannot_see_the_absolute_scale` pins. So what this gate
decides is that the solver's pupil-to-image transform reproduces the Airy *shape*,
in units of the map's own sampling, and that the residual converges in pupil
sampling. It cannot reject a PSF that is the right shape at the wrong size, in the
same way `peak_normalized_disc_relative_l2` cannot reject one that is the right
shape at the wrong brightness -- and for the same structural reason.

The absolute scale is a separate check, from the fixture only
------------------------------------------------------------
`test_the_absolute_scale_is_the_fixtures_own_f_number` is the half the L2 metric
cannot make, and it uses **no number the solver reported**: the fixture's own
`SINGLET_EFFECTIVE_FOCAL_LENGTH_MM / SINGLET_ENTRANCE_PUPIL_DIAMETER_MM` is
exactly 9.7 by construction, so `lambda F/#` predicts both the reported pitch and
the analytic Airy radius independently. Measured: the reported pitch is 0.33 %
below that prediction, which is the *working* F-number (9.668128, from four traced
marginal rays) differing from the paraxial one -- 9.668128 / 9.7 = 0.99671, the
same 0.33 %, so the discrepancy is attributed rather than tolerated.

Why `NA = 1 / (2 F/#)` is admissible but not independent
--------------------------------------------------------
It is not a fit: `optiland.utils.get_working_FNO` is
`1 / (2 sqrt(mean((n sin theta)^2)))` over four real traced marginal rays, and
measured it reproduces CHE-117's "largest traced direction cosine" declaration
(0.0517163) to seven digits from a completely different measurement. Nothing here
tunes an NA to make a residual small, which `test_psf_verification.py` is explicit
would destroy the independence that makes O1 admissible at all. But it is also not
an independent *constraint on this comparison*, because it is the same F-number
that sets the pitch -- which is exactly what the cancellation above says.

What is deliberately NOT concluded
---------------------------------
R11.2 measured the project's own `trace -> ray_to_scalar -> psf` chain at 2.13e-3
against this same oracle on this same system and recorded the gate as ATTRIBUTED
AND UNMET, attributing 94.8 % of the residual to an Airy-scale offset. **8.18e-4
here is not evidence against that record and must not be read as any.** That
chain's image pitch comes from the reconstruction geometry, so its metric *is*
scale-sensitive; this one is not, by the cancellation above. The two numbers are
not the same comparison and the smaller one is not the better result.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from fixtures.systems import (
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    SINGLET_F_NUMBER,
    singlet_ref,
    singlet_ref_stopped_down,
    singlet_source,
)
from oracles import (
    airy_first_null_radius_m,
    airy_psf_on_grid,
    measure_first_null_radius_m,
    peak_normalized_disc_relative_l2,
    pixels_per_airy_radius,
)

from backends.optiland import psf
from backends.optiland.analysis import NativePsfAnalysis

EXECUTION: dict[str, str] = {"device": "cpu", "precision": "fp64"}

WAVELENGTH_M = 0.55e-06

#: The frozen L2-PSF-01 gate. Not widened anywhere in this file.
FROZEN_GATE = 1.0e-3

#: The disc the metric is taken over, in Airy radii. Three rather than the whole
#: window, for the reason `oracles.disc_mask` gives: the corners of a square window
#: are tens of Airy radii out, where both patterns are at the 1e-5 level and the
#: residual is dominated by window truncation. Measured, the choice between 3 and 5
#: moves the numbers below by under 1 %.
DISC_AIRY_RADII = 3.0

#: The configuration the gate is asserted at: 256 pupil samples on a 1024 grid.
#: Measured cost, 0.17 s.
GATE_NUM_RAYS = 256
GATE_GRID_SIZE = 1024

#: The measured convergence in pupil sampling at a fixed image grid, which is what
#: says the threshold above did not have to move. `(num_rays, grid_size, L2)`.
PUPIL_SAMPLING_SERIES = (
    (32, 256, 1.6163e-2),
    (64, 512, 5.6610e-3),
    (128, 1024, 1.8253e-3),
    (256, 1024, 8.1823e-4),
)


def _analysis(*, num_rays: int, grid_size: int) -> NativePsfAnalysis:
    """The native FFT PSF of M3-SINGLET-REF, on axis, at the reference wavelength."""
    return psf(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0), wavelength_um=0.55),
        method="fft",
        num_rays=num_rays,
        grid_size=grid_size,
        execution=EXECUTION,
    )


def _numerical_aperture(result: NativePsfAnalysis) -> float:
    """`1 / (2 F/#)` from the analysis's own working F-number.

    Optiland's `get_working_FNO` is `1 / (2 sqrt(mean(NA^2)))` over four real traced
    marginal rays, so inverting it recovers exactly the marginal-ray numerical
    aperture the trace produced: an admissible declaration, taken from the system
    rather than fitted to the oracle.

    **Admissible but not independent of the pitch**, which the same F-number sets --
    see the module docstring's cancellation. The absolute-scale claim is anchored on
    the prescription instead.
    """
    return 1.0 / (2.0 * result.working_f_number)


def _gate_metric(result: NativePsfAnalysis) -> tuple[float, float]:
    """`(metric, samples per Airy radius)` for one analysis against O1."""
    numerical_aperture = _numerical_aperture(result)
    airy_radius_m = airy_first_null_radius_m(
        numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
    )
    pitch = (result.pixel_pitch_m, result.pixel_pitch_m)
    oracle = airy_psf_on_grid(
        shape=result.image_shape,
        sample_pitch_m=pitch,
        numerical_aperture=numerical_aperture,
        wavelength_m=WAVELENGTH_M,
    )
    metric = peak_normalized_disc_relative_l2(
        result.intensity,
        oracle,
        sample_pitch_m=pitch,
        radius_m=DISC_AIRY_RADII * airy_radius_m,
    )
    return metric, pixels_per_airy_radius(sample_pitch_m=pitch, airy_radius_m=airy_radius_m)


def test_the_native_fft_psf_matches_the_airy_pattern_inside_the_frozen_gate() -> None:
    """**Criterion 3.** 8.18e-4 against the frozen 1.0e-3, at 4.90 samples per Airy radius.

    O1 only, at the numerical aperture the analysis itself reports. Nothing in this
    comparison is this repository's numerics on either side: the measured field
    comes from the pinned solver and the reference from a closed form.

    **A dimensionless statement.** Read the module docstring: `lambda` and `F/#`
    cancel out of this comparison, so what 8.18e-4 says is that the pattern has the
    Airy *shape* to that residual in units of its own sampling -- not that its size
    is right. The size is
    `test_the_absolute_scale_is_the_fixtures_own_f_number`'s job and it is checked
    against the fixture rather than against the solver.

    The sampling number is reported because the residual depends on it, and the
    convergence test below is what says 1.0e-3 did not have to move.
    """
    result = _analysis(num_rays=GATE_NUM_RAYS, grid_size=GATE_GRID_SIZE)
    metric, samples_per_airy_radius = _gate_metric(result)

    assert samples_per_airy_radius == pytest.approx(4.90, rel=0.01)
    assert metric == pytest.approx(8.182e-4, rel=0.01), metric
    assert metric < FROZEN_GATE

    # The premises: the configuration really is near diffraction limited under the
    # pinned solver's own normalization, and it really is the whole pupil.
    assert result.strehl_ratio > 0.998
    assert result.num_rays == GATE_NUM_RAYS


def test_the_declared_numerical_aperture_is_the_traced_one_and_not_a_fit() -> None:
    """Which NA the gate above rests on, and what changing it does.

    CHE-117 recorded 0.0517163 as the largest transverse direction cosine this
    system's trace produces. `1 / (2 * working_f_number)` reproduces it to seven
    digits from an entirely different measurement -- four marginal rays and a chief
    ray, inside the pinned solver -- so it is a declaration the system makes and
    not a number chosen to make a residual small.

    **What it is not is independent of the pitch**, since the same F-number sets
    both; that is the cancellation the module docstring derives. Changing the NA
    *alone* therefore reads as a scale mismatch and the metric does see it: at the
    paraxial geometric declaration (0.0514780, a 0.46 % smaller NA) it goes from
    8.18e-4 to 6.38e-3, and the slope of that -- 1.5 per unit fractional scale
    error, CHE-117's measurement -- is why this comparison is sensitive to a
    *relative* scale error while being blind to a common one.
    """
    result = _analysis(num_rays=GATE_NUM_RAYS, grid_size=GATE_GRID_SIZE)
    assert _numerical_aperture(result) == pytest.approx(0.0517163, abs=1e-7)

    # The same comparison at the paraxial declaration, to show the spread is real.
    paraxial = 0.0514780
    airy_radius_m = airy_first_null_radius_m(
        numerical_aperture=paraxial, wavelength_m=WAVELENGTH_M
    )
    pitch = (result.pixel_pitch_m, result.pixel_pitch_m)
    at_paraxial = peak_normalized_disc_relative_l2(
        result.intensity,
        airy_psf_on_grid(
            shape=result.image_shape,
            sample_pitch_m=pitch,
            numerical_aperture=paraxial,
            wavelength_m=WAVELENGTH_M,
        ),
        sample_pitch_m=pitch,
        radius_m=DISC_AIRY_RADII * airy_radius_m,
    )
    assert at_paraxial == pytest.approx(6.379e-3, rel=0.02), at_paraxial
    assert at_paraxial > FROZEN_GATE
    assert abs(at_paraxial - _gate_metric(result)[0]) > 5.0 * FROZEN_GATE


def test_the_shape_gate_cannot_see_the_absolute_scale() -> None:
    """The property the gate above has to be read with, pinned rather than argued.

    Rescaling the reported pitch and the oracle's numerical aperture *together*
    leaves the metric unchanged to float round-off: `v = 2 pi NA r / lambda` with
    `r` proportional to the pitch, so a common factor cancels. Measured at 2x and
    10x, the metric moves by under 1e-12 relative.

    Scaling the pitch alone does not cancel and the metric explodes to 1.37, which
    is what says this test is measuring an invariance and not a broken comparison.

    This is the same limitation `oracles.peak_normalized_disc_relative_l2` states
    about a global intensity scale, one axis over, and it exists here for the same
    structural reason: both quantities are divided out before the residual is taken.
    """
    result = _analysis(num_rays=GATE_NUM_RAYS, grid_size=GATE_GRID_SIZE)
    baseline, _ = _gate_metric(result)

    for factor in (2.0, 10.0):
        pitch = (result.pixel_pitch_m * factor, result.pixel_pitch_m * factor)
        numerical_aperture = _numerical_aperture(result) / factor
        airy_radius_m = airy_first_null_radius_m(
            numerical_aperture=numerical_aperture, wavelength_m=WAVELENGTH_M
        )
        rescaled = peak_normalized_disc_relative_l2(
            result.intensity,
            airy_psf_on_grid(
                shape=result.image_shape,
                sample_pitch_m=pitch,
                numerical_aperture=numerical_aperture,
                wavelength_m=WAVELENGTH_M,
            ),
            sample_pitch_m=pitch,
            radius_m=DISC_AIRY_RADII * airy_radius_m,
        )
        assert rescaled == pytest.approx(baseline, rel=1e-12), factor

    # The falsifier: the pitch alone is not invariant, so the comparison is live.
    pitch = (2.0 * result.pixel_pitch_m, 2.0 * result.pixel_pitch_m)
    airy_radius_m = airy_first_null_radius_m(
        numerical_aperture=_numerical_aperture(result), wavelength_m=WAVELENGTH_M
    )
    mismatched = peak_normalized_disc_relative_l2(
        result.intensity,
        airy_psf_on_grid(
            shape=result.image_shape,
            sample_pitch_m=pitch,
            numerical_aperture=_numerical_aperture(result),
            wavelength_m=WAVELENGTH_M,
        ),
        sample_pitch_m=pitch,
        radius_m=DISC_AIRY_RADII * airy_radius_m,
    )
    assert mismatched > 1.0, mismatched


def test_the_absolute_scale_is_the_fixtures_own_f_number() -> None:
    """The half the L2 gate cannot make, from the prescription and nothing else.

    `SINGLET_EFFECTIVE_FOCAL_LENGTH_MM / SINGLET_ENTRANCE_PUPIL_DIAMETER_MM` is
    **exactly 9.7** by construction -- the fixture derives the pupil from the focal
    length and that number -- so `lambda F/#` predicts the absolute size of
    everything on the returned record with no solver quantity involved. Three
    statements, none of which the peak-normalized metric can make:

    1. the **reported pitch** is `lambda F/# (num_rays - 1) / grid_size` to 0.33 %,
       and the 0.33 % is attributed: the solver's *working* F-number is 9.668128
       from four traced marginal rays, and 9.668128 / 9.7 = 0.99671;
    2. the **measured first null** is `1.22 lambda F/#` = 6.5087e-6 m to 1.23 %,
       measured 6.5885e-6 m -- and the two effects here partially *cancel* rather
       than adding: the estimator's own bias at 9.8 samples per Airy radius is
       +1.59 % against the traced-NA radius (6.4856e-6 m, which
       `test_the_pattern_is_centred_...` records as +1.6 %), and the traced radius is
       0.36 % *below* the nominal one because the working F-number is below the
       paraxial one. 1.23 % is 1.59 % minus that 0.36 %;
    3. stopping the pupil to a quarter moves the scale by **4.0117x** relative to
       the full-aperture scale claim 1 anchors, so the two configurations this file
       gates really are two different Airy scales and not one relabelled. This one
       is a ratio of two nulls each measured through its own reported pitch, so it
       inherits claim 1's anchoring rather than making a second absolute claim; the
       0.3 % excess over 4 is the working F-number again (38.79205 / 9.668128 =
       4.0124, matched to 0.02 %), because the real marginal-ray angle is not linear
       in pupil radius.
    """
    f_number = SINGLET_EFFECTIVE_FOCAL_LENGTH_MM / SINGLET_ENTRANCE_PUPIL_DIAMETER_MM
    assert f_number == pytest.approx(SINGLET_F_NUMBER, rel=1e-12)

    result = _analysis(num_rays=GATE_NUM_RAYS, grid_size=2048)
    predicted_pitch_m = WAVELENGTH_M * f_number * (result.num_rays - 1) / 2048
    assert result.pixel_pitch_m / predicted_pitch_m == pytest.approx(0.99671, rel=1e-4)
    assert result.working_f_number / f_number == pytest.approx(0.99671, rel=1e-4)

    nominal_airy_radius_m = 1.22 * WAVELENGTH_M * f_number
    measured_m = measure_first_null_radius_m(
        np.asarray(result.intensity),
        sample_pitch_m=(result.pixel_pitch_m, result.pixel_pitch_m),
    )
    assert measured_m / nominal_airy_radius_m == pytest.approx(1.0123, rel=1e-3)
    assert measured_m == pytest.approx(nominal_airy_radius_m, rel=2e-2)

    stopped = psf(
        singlet_ref_stopped_down(aperture_fraction=0.25),
        singlet_source(field_angle_deg=(0.0, 0.0), wavelength_um=0.55),
        method="fft",
        num_rays=GATE_NUM_RAYS,
        grid_size=2048,
        execution=EXECUTION,
    )
    stopped_null_m = measure_first_null_radius_m(
        np.asarray(stopped.intensity),
        sample_pitch_m=(stopped.pixel_pitch_m, stopped.pixel_pitch_m),
    )
    assert stopped_null_m / measured_m == pytest.approx(4.0117, rel=1e-3)


def test_the_residual_is_pupil_sampling_before_it_is_anything_else() -> None:
    """Why the threshold did not move: the residual converges in `num_rays`.

    Four configurations, each one a doubling of the pupil sampling, each measured
    against O1 at that configuration's own reported NA. The sequence divides by
    about three per doubling -- roughly `num_rays**-1.6` -- which is the behaviour of
    a staircased binary aperture edge and not of a physical discrepancy.

    A tolerance justified by a convergence sequence is a different claim from one
    justified by the number it has to pass, and this test is the difference.
    """
    metrics = []
    for num_rays, grid_size, expected in PUPIL_SAMPLING_SERIES:
        result = _analysis(num_rays=num_rays, grid_size=grid_size)
        assert result.num_rays == num_rays
        metric, _ = _gate_metric(result)
        assert metric == pytest.approx(expected, rel=0.02), (num_rays, metric)
        metrics.append(metric)

    assert metrics == sorted(metrics, reverse=True)
    for coarse, fine in itertools.pairwise(metrics):
        assert coarse / fine > 2.0
    # Only the last one is inside the gate, which is the honest reading of the
    # series: this is a converging comparison and not a threshold that happens to
    # be met.
    assert metrics[-1] < FROZEN_GATE <= metrics[-2]


def test_refining_the_image_grid_alone_does_not_move_the_residual() -> None:
    """The other half of the attribution: the image pitch is not what is limiting.

    Same pupil sampling, twice the image-plane resolution -- 4.90 against 9.80
    samples per Airy radius -- and the metric moves by under 1e-3 relative. If the
    residual were the grid, this is where it would show.
    """
    coarse = _analysis(num_rays=GATE_NUM_RAYS, grid_size=1024)
    fine = _analysis(num_rays=GATE_NUM_RAYS, grid_size=2048)

    coarse_metric, coarse_sampling = _gate_metric(coarse)
    fine_metric, fine_sampling = _gate_metric(fine)

    assert fine.pixel_pitch_m == pytest.approx(coarse.pixel_pitch_m / 2.0, rel=1e-12)
    assert fine_sampling == pytest.approx(2.0 * coarse_sampling, rel=1e-3)
    assert fine_metric == pytest.approx(coarse_metric, rel=1e-3)


def test_the_pattern_is_centred_and_its_first_null_is_the_analytic_one() -> None:
    """The two statements the L2 metric cannot make, since it peak-normalizes.

    A centred, diffraction-limited pattern: the peak sample is the grid origin on
    `representations.Frame`'s rule, and the first null along `+x` -- measured
    sub-pixel from the map, against a continuous analytic radius -- lands at
    6.589e-6 m against an analytic 6.486e-6 m, **+1.6 %**.

    The tolerance is 2 % and it is a statement about the measurement rather than
    about the physics: `oracles.measure_first_null_radius_m` interpolates a
    parabola through three samples around a broad minimum, and R11.2 measured that
    same estimator at +166 % on 2.4 samples per Airy radius. This runs at 9.8
    samples per Airy radius, where it is good to a couple of per cent, and the L2
    gate above is the shape statement that does not depend on it.

    Note that this comparison is **also** scale-free -- `analytic_m` comes from the
    solver's working F-number and `measured_m` from the pitch that same F-number
    sets, so a common error in both cancels here exactly as it does in the L2
    metric. The absolute statement is
    `test_the_absolute_scale_is_the_fixtures_own_f_number`, which compares the same
    measured null against `1.22 lambda F/#` built from the prescription alone.
    """
    result = _analysis(num_rays=GATE_NUM_RAYS, grid_size=2048)
    ny, nx = result.image_shape
    assert result.peak_index == (ny // 2, nx // 2)

    analytic_m = airy_first_null_radius_m(
        numerical_aperture=_numerical_aperture(result), wavelength_m=WAVELENGTH_M
    )
    measured_m = measure_first_null_radius_m(
        np.asarray(result.intensity),
        sample_pitch_m=(result.pixel_pitch_m, result.pixel_pitch_m),
    )
    assert measured_m == pytest.approx(6.5885e-06, rel=1e-3)
    assert measured_m == pytest.approx(analytic_m, rel=2e-2)

    # And `coordinates()` is the same grid the comparison was made on, so a consumer
    # plotting the record against the analytic radius sees the same picture.
    _, x = result.coordinates()
    assert x[nx // 2] == pytest.approx(0.0, abs=1e-18)
    assert float(x[-1]) > 3.0 * analytic_m


def test_a_stopped_down_singlet_is_more_diffraction_limited_and_still_matches() -> None:
    """The gate is not a property of one aberration content: it closes at f/38.8 too.

    Stopping the same singlet to a quarter of its entrance pupil removes essentially
    all of its spherical aberration -- the pinned solver reports Strehl 1.000000
    against 0.99899 at full aperture -- and the shape metric reads 5.63e-4 at the
    same pupil sampling. Two very different aberration contents, one threshold, and
    the residual is *smaller* where the aberration is smaller, which is the
    direction that says the metric is reading the wavefront and not only the grid.

    **This is not a second absolute scale**, even though the aperture changed: the
    samples-per-Airy-radius figure is 4.90 in both configurations, because the FFT
    path's pitch tracks the F-number that also sets the Airy radius. The absolute
    4x is asserted separately, in
    `test_the_absolute_scale_is_the_fixtures_own_f_number`.

    This is also the honest form of "aberration lowers the PSF": the *aberrated*
    direction is asserted in `tests/backends/test_optiland_psf.py` off axis, and
    here the near-perfect direction is asserted to still match a diffraction-limited
    closed form rather than merely to have a higher Strehl.
    """
    result = psf(
        singlet_ref_stopped_down(aperture_fraction=0.25),
        singlet_source(field_angle_deg=(0.0, 0.0), wavelength_um=0.55),
        method="fft",
        num_rays=GATE_NUM_RAYS,
        grid_size=GATE_GRID_SIZE,
        execution=EXECUTION,
    )

    assert result.working_f_number == pytest.approx(38.79205, rel=1e-4)
    assert result.strehl_ratio > 0.99999
    metric, samples_per_airy_radius = _gate_metric(result)
    assert metric == pytest.approx(5.63e-4, rel=0.02), metric
    assert metric < FROZEN_GATE
    # Less aberration, smaller residual, at the SAME sampling figure -- which is the
    # premise that makes the comparison between the two rows meaningful.
    full_aperture = _analysis(num_rays=GATE_NUM_RAYS, grid_size=GATE_GRID_SIZE)
    assert samples_per_airy_radius == pytest.approx(
        _gate_metric(full_aperture)[1], rel=1e-3
    )
    assert result.strehl_ratio > full_aperture.strehl_ratio
    assert metric < _gate_metric(full_aperture)[0]


def test_no_oracle_reaches_production() -> None:
    """The oracle is test-side, and `analysis.py` computes no Airy pattern.

    R11.2 makes the same assertion about its own chain and for the same reason: an
    oracle that leaked into `src/` would be comparing the tree against itself. The
    only import of `scipy.special` in this repository's PSF story is the one in
    `tests/physics/oracles.py`.
    """
    from pathlib import Path

    source = Path("src/backends/optiland/analysis.py").read_text()
    assert "scipy" not in source
    assert "bessel" not in source.lower()
    # `AIRY_FIRST_NULL_V`, the one constant every comparison in this file is scaled
    # by. Searched for as the number rather than the name, because the name could be
    # spelled differently and the number could not.
    assert "3.8317" not in source
    assert not math.isnan(FROZEN_GATE)
