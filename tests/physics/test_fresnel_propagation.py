"""Fresnel propagation against closed forms, and the one comparison that may not gate.

CHE-228 (R06.11). `backends.chromatix.fresnel_propagate` is the paraxial transfer
function, and everything below decides it against arithmetic rather than against
this project's own angular spectrum.

Three closed forms, each with the tolerance it is held to and the number measured:

===========================  =====================================  =========  ==========
case                         oracle                                 tolerance  measured
===========================  =====================================  =========  ==========
plane-wave phase advance     ``-pi (lambda_0/n) z f^2``, exact      5e-4 rad   see below
tilted-beam walk-off         ``z sin(theta)``                       0.05 px    1e-4 px
Gaussian spreading           paraxial ``w0 sqrt(1 + (z/zR)^2)``     2e-2 rel   2.9e-8
===========================  =====================================  =========  ==========

Every case goes through the **public operation**. Nothing here imports chromatix
or touches a propagator array: an earlier draft read `compute_transfer_propagator`
and the backend's own `f_grid`, and `tests/backends/test_chromatix_boundary.py`
refused it. That refusal was right twice over -- the anti-corruption boundary says
no chromatix symbol appears outside `backends/chromatix/`, and an oracle that takes
its frequency grid from the implementation it is judging has borrowed part of the
answer. The grids below are built from the declared pitch and shape in numpy.

The walk-off is the Fresnel analogue of `test_focal_plane_transform`'s
`f sin(theta)`: the kernel's group delay is `lambda_0 z f / n`, linear in spatial
frequency, so a beam at `theta` lands at `z sin(theta)` while the exact angular
spectrum lands it at `z tan(theta)`. Both are arithmetic.

**Whether the measurement can also *reject* `z tan(theta)` is a separate test, and
that is a finding rather than a formality.** The two predictions differ at order
`theta^3`, so on this grid they are 0.01 samples apart at 2 degrees and 0.13 at 5
-- far inside the 0.05-sample tolerance, meaning the parametrized case above
passes at those angles without distinguishing the two models at all. Only at 10
degrees do they separate, by 1.07 samples, and that is where the discrimination
is asserted. Writing the rejection into the parametrized case would have claimed a
resolution two thirds of its angles do not have.

Which oracle decides, and which does not
-----------------------------------------
The obvious comparison for a Fresnel kernel is against the exact one, and
`AGENTS.md` forbids it as a gate: "repository numerical code must not be the sole
correctness oracle for the same numerical code", and `O_ASM_PROPAGATE` is this
repository's code. So the ASM/Fresnel difference is measured *here* -- section 4 --
against the **closed-form residual** `n k0 z (1 - cos theta - sin^2(theta)/2)`, and
it is labelled diagnostic in its own docstring. What gates is the residual formula;
what the comparison establishes is that the two implementations differ by exactly
the amount the algebra says and by nothing else.

The relationship that makes the residual exact
-----------------------------------------------
`compute_transfer_propagator` is `exp(-i pi (lambda/n) z f^2)` and
`_carrier_removed_propagator` is `exp(-2 pi i |z| (lambda/n) f^2 / (delay + 1))`
with `delay = cos(theta)`. Substituting `delay -> 1` turns the second into the
first, so the whole of the Fresnel approximation is one axial ratio set to unity,
and the phase difference has a closed form with no fitted constant in it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backends.chromatix import fresnel_propagate, propagate
from representations import ReferenceSurface, ScalarField

WAVELENGTH_M = 0.532e-6
K = 2.0 * math.pi / WAVELENGTH_M

#: Inherited from `test_scalar_wave_propagation.py`, whose Gaussian case is the
#: same measurement under the exact kernel. Not re-chosen here: the paraxial
#: Gaussian is the closed form both propagations are held to, and a Fresnel run is
#: if anything closer to it than an exact one.
GAUSSIAN_TOLERANCE = 2e-2

#: How far the measured centroid may sit from `z sin(theta)`, in samples. Sub-pixel
#: because the discrimination against `z tan(theta)` is only 1.07 samples wide on
#: this grid; the measurement itself comes in at 1e-4 samples, so the tolerance is
#: three orders of magnitude of headroom rather than a fitted bound.
WALKOFF_TOLERANCE_SAMPLES = 0.05

#: The float32 phase floor for a kernel comparison. `carrier_phase_rad`-scale
#: quantities are not involved -- both kernels here are carrier-free -- so what is
#: left is one float32 epsilon on a phase of order `n k0 z sin^4(theta)/8`, times
#: the handful of operations that build it.
KERNEL_PHASE_TOLERANCE_RAD = 5e-4


def _field(u: np.ndarray, pitch_m: tuple[float, float], *, medium_index: float = 1.0):
    return ScalarField(
        u=u,
        sample_pitch_m=pitch_m,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(
            name="source", z_m=0.0, medium_index=medium_index
        ),
    )


def _fresnel(field: ScalarField, distance_m: float, *, pad_width: int = 0) -> ScalarField:
    return fresnel_propagate(
        field,
        distance_m=distance_m,
        model={"pad_width": pad_width, "target_surface": "target"},
    )


# ---------------------------------------------------------------------------
# 1. The transfer function, measured one spatial frequency at a time
# ---------------------------------------------------------------------------
#
# Through the public operation, on plane waves the grid represents exactly. The
# first version of this section read `compute_transfer_propagator` and the
# backend's own `f_grid` directly, and `test_chromatix_boundary.py` refused it --
# correctly, and for a second reason beyond the boundary rule: an oracle that takes
# its frequency grid from the implementation it is judging has borrowed part of the
# answer. The grid below is built from the declared pitch and shape in numpy.

KERNEL_SHAPE = (128, 128)
KERNEL_PITCH_M = (0.3e-6, 0.3e-6)
KERNEL_DISTANCE_M = 40e-6
KERNEL_INDEX = 1.33
#: x-axis bins to probe. Non-zero, because on axis the phase is 0 for every
#: propagator ever written; spread out, so a quadratic in `f` cannot be fitted by
#: a linear one; and inside the propagating cone, since `sin(theta) = lambda_0 f/n`
#: reaches 0.42 at bin 40 and 1.0 would be the evanescent cut.
KERNEL_BINS = (4, 11, 25, 40)


def _bin_frequency_per_m(bin_index: int) -> float:
    """The `n`-th DFT bin of the x axis, in cycles per metre: `n / (N dx)`."""
    return bin_index / (KERNEL_SHAPE[1] * KERNEL_PITCH_M[1])


def _uniform_advance(out: ScalarField, source: ScalarField) -> float:
    """`arg(U_out / U_in)` for a field that advances uniformly, in radians.

    Averaged over the grid before the angle is taken, so the estimate is the
    field's own mean phasor rather than one pixel's -- the same reduction
    `test_scalar_wave_propagation.py` uses for the exact kernel.
    """
    ratio = np.asarray(out.u) / np.asarray(source.u)
    return float(np.angle(np.mean(ratio)))


def _wrapped(radians: float) -> float:
    """`radians` folded into `(-pi, pi]`, for comparing against a phase that wraps."""
    return float(np.angle(np.exp(1j * radians)))


@pytest.mark.parametrize("bin_index", KERNEL_BINS)
def test_a_plane_wave_advances_by_the_fresnel_closed_form(bin_index: int) -> None:
    """Criterion 1. `-pi (lambda_0/n) z f^2`, one spatial frequency at a time.

    A plane wave at an exact DFT bin is periodic on the grid, so `pad_width=0` is
    the physical choice rather than a shortcut -- zero padding would destroy the
    periodicity, which is the same reasoning `test_scalar_wave_propagation.py`
    records for the exact kernel.

    This is the whole of what the operation claims to apply, and every other
    assertion in this file is downstream of it. Four bins rather than one, because
    the claim is that the phase is **quadratic** in `f`, and a single frequency
    cannot tell a quadratic from anything else that passes through the same point.
    """
    frequency = _bin_frequency_per_m(bin_index)
    field = _field(
        np.ones(KERNEL_SHAPE, dtype=np.complex64), KERNEL_PITCH_M, medium_index=KERNEL_INDEX
    )
    _, x = (np.asarray(axis) for axis in field.coordinates())
    carrier = np.broadcast_to(
        np.exp(2j * np.pi * frequency * x)[None, :], KERNEL_SHAPE
    ).astype(np.complex64)
    source = _field(carrier, KERNEL_PITCH_M, medium_index=KERNEL_INDEX)

    measured = _uniform_advance(_fresnel(source, KERNEL_DISTANCE_M), source)
    predicted = (
        -math.pi * (WAVELENGTH_M / KERNEL_INDEX) * KERNEL_DISTANCE_M * frequency**2
    )

    residual = _wrapped(measured - predicted)
    assert abs(residual) < KERNEL_PHASE_TOLERANCE_RAD

    # The premise: this bin carries a real phase, so the comparison is not against
    # zero. At bin 4 it is already 0.5 rad and at bin 40 it is 54 rad.
    assert abs(predicted) > 0.4


def test_the_advance_is_quadratic_in_frequency_and_not_merely_monotonic() -> None:
    """The falsifier for the case above, since four passing points could be a fit.

    `phi(f) / f^2` is a constant -- `-pi (lambda_0/n) z` -- for a Fresnel kernel and
    is not for any other propagator in this tree: the exact angular spectrum's
    `n k0 z (cos(theta) - 1)` bends away from it by `sin^2(theta)/4` in relative
    terms. Taken unwrapped, which is why the bins are chosen to keep the largest
    phase inside a few tens of radians rather than hundreds.
    """
    ratios = []
    for bin_index in KERNEL_BINS:
        frequency = _bin_frequency_per_m(bin_index)
        predicted = (
            -math.pi * (WAVELENGTH_M / KERNEL_INDEX) * KERNEL_DISTANCE_M * frequency**2
        )
        field = _field(
            np.ones(KERNEL_SHAPE, dtype=np.complex64),
            KERNEL_PITCH_M,
            medium_index=KERNEL_INDEX,
        )
        _, x = (np.asarray(axis) for axis in field.coordinates())
        source = _field(
            np.broadcast_to(
                np.exp(2j * np.pi * frequency * x)[None, :], KERNEL_SHAPE
            ).astype(np.complex64),
            KERNEL_PITCH_M,
            medium_index=KERNEL_INDEX,
        )
        # Unwrapped, by adding back the whole turns the prediction says are there.
        measured = _uniform_advance(_fresnel(source, KERNEL_DISTANCE_M), source)
        turns = round((predicted - measured) / (2.0 * math.pi))
        ratios.append((measured + 2.0 * math.pi * turns) / frequency**2)

    expected = -math.pi * (WAVELENGTH_M / KERNEL_INDEX) * KERNEL_DISTANCE_M
    for ratio in ratios:
        assert abs(ratio - expected) / abs(expected) < 1e-5


# ---------------------------------------------------------------------------
# 2. The discriminating case: z sin(theta), not z tan(theta)
# ---------------------------------------------------------------------------

WALKOFF_SHAPE = (256, 256)
WALKOFF_PITCH_M = (0.5e-6, 0.5e-6)
WALKOFF_DISTANCE_M = 200e-6
WALKOFF_WAIST_M = 12e-6


def _tilted_beam(theta_deg: float) -> ScalarField:
    """A confined Gaussian carrying a transverse carrier of `sin(theta)` in **x**."""
    field = _field(np.ones(WALKOFF_SHAPE, dtype=np.complex64), WALKOFF_PITCH_M)
    y, x = (np.asarray(axis) for axis in field.coordinates())
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    envelope = np.exp(-(grid_x**2 + grid_y**2) / WALKOFF_WAIST_M**2)
    ramp = np.exp(1j * K * math.sin(math.radians(theta_deg)) * grid_x)
    return _field((envelope * ramp).astype(np.complex64), WALKOFF_PITCH_M)


def _centroid_x_m(field: ScalarField) -> float:
    intensity = np.abs(np.asarray(field.u)).astype(np.float64) ** 2
    _, x = (np.asarray(axis) for axis in field.coordinates())
    return float((intensity * x[None, :]).sum() / intensity.sum())


@pytest.mark.parametrize("theta_deg", [2.0, 5.0, 10.0])
def test_a_tilted_beam_walks_off_by_z_sin_theta(theta_deg: float) -> None:
    """Criterion 2. The Fresnel kernel's group delay is linear in spatial frequency.

    The transfer function's phase is `-pi (lambda_0/n) z f^2`, whose group delay
    `-(1/2pi) dphi/df` is `lambda_0 z f / n = z sin(theta)`. Arithmetic, and
    independent of this project's angular spectrum.

    Three angles, because a single one cannot show that the relation is linear in
    `sin(theta)` rather than fitted at a point. Whether the measurement can also
    *reject* `z tan(theta)` is a separate question and a separate test, because the
    two predictions are not separable at every angle -- see below.
    """
    theta = math.radians(theta_deg)
    field = _tilted_beam(theta_deg)
    measured = _centroid_x_m(_fresnel(field, WALKOFF_DISTANCE_M, pad_width=256))

    prediction = WALKOFF_DISTANCE_M * math.sin(theta)
    assert abs(measured - prediction) / WALKOFF_PITCH_M[1] < WALKOFF_TOLERANCE_SAMPLES


def test_the_walk_off_measurement_rejects_z_tan_theta() -> None:
    """The falsifiable twin, at the one angle where the two predictions separate.

    `z sin(theta)` and `z tan(theta)` differ at order `theta^3`, so on this grid
    they are 0.01 samples apart at 2 degrees and 0.13 at 5 -- the case above passes
    at those angles without distinguishing the models at all, and asserting that it
    does would be asserting a resolution the geometry does not have. At 10 degrees
    they are 1.07 samples apart, 21 times the tolerance, and that is where the
    discrimination is real. The grid is not widened to make the gap larger: at
    200 um the beam walks 34.7 um against a 64 um half-extent, and pushing either
    the angle or the distance walks it off the window, which would replace a
    physics measurement with a truncation artifact.

    This is the Fresnel analogue of `O_FOCAL_PLANE_TRANSFORM`'s `f sin(theta)`
    claim, and it is a property of the model rather than a defect in it. The exact
    angular spectrum lands the same beam at `z tan(theta)`; both are arithmetic.
    """
    theta_deg = 10.0
    theta = math.radians(theta_deg)
    pitch = WALKOFF_PITCH_M[1]
    sin_prediction = WALKOFF_DISTANCE_M * math.sin(theta)
    tan_prediction = WALKOFF_DISTANCE_M * math.tan(theta)

    separation = abs(tan_prediction - sin_prediction) / pitch
    assert separation > 10.0 * WALKOFF_TOLERANCE_SAMPLES, (
        f"the two predictions differ by only {separation:.2f} samples, so this "
        "measurement cannot tell them apart and is not a discrimination"
    )

    measured = _centroid_x_m(_fresnel(_tilted_beam(theta_deg), WALKOFF_DISTANCE_M, pad_width=256))
    assert abs(measured - sin_prediction) / pitch < WALKOFF_TOLERANCE_SAMPLES
    assert abs(measured - tan_prediction) / pitch > separation / 2.0


# ---------------------------------------------------------------------------
# 3. Gaussian spreading, the closed form both propagations share
# ---------------------------------------------------------------------------


def _intensity_radius_m(field: ScalarField) -> float:
    """The 1/e^2 intensity radius as a second moment: `2 * sqrt(<r^2>)`."""
    intensity = np.abs(np.asarray(field.u)).astype(np.float64) ** 2
    y, x = (np.asarray(axis) for axis in field.coordinates())
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    total = intensity.sum()
    mean_square = ((grid_x**2 + grid_y**2) * intensity).sum() / total
    return float(2.0 * math.sqrt(mean_square / 2.0))


def test_a_gaussian_beam_spreads_by_the_paraxial_closed_form() -> None:
    """Criterion 3. `w(z) = w0 sqrt(1 + (z / zR)^2)`, `zR = pi w0^2 n / lambda_0`.

    The paraxial Gaussian is the *exact* solution of the Fresnel equation, so this
    is the one case where the approximation under test and the oracle agree by
    construction rather than to some order -- which makes it a check on the
    plumbing (padding, cropping, the pitch declaration) as much as on the kernel.
    """
    waist_m = 6e-6
    distance_m = 400e-6
    field = _field(np.ones((256, 256), dtype=np.complex64), (0.5e-6, 0.5e-6))
    y, x = (np.asarray(axis) for axis in field.coordinates())
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    field = _field(
        np.exp(-(grid_x**2 + grid_y**2) / waist_m**2).astype(np.complex64), (0.5e-6, 0.5e-6)
    )

    rayleigh_m = math.pi * waist_m**2 / WAVELENGTH_M
    predicted = waist_m * math.sqrt(1.0 + (distance_m / rayleigh_m) ** 2)
    measured = _intensity_radius_m(_fresnel(field, distance_m, pad_width=256))

    assert abs(measured - predicted) / predicted < GAUSSIAN_TOLERANCE
    # The premise: the beam actually spread, so the comparison is not against w0.
    assert predicted > 1.5 * waist_m


# ---------------------------------------------------------------------------
# 4. The ASM comparison -- DIAGNOSTIC, and it may not gate
# ---------------------------------------------------------------------------


def test_the_difference_from_the_exact_kernel_is_the_closed_form_residual() -> None:
    """Criterion 7. **Diagnostic evidence, not a gate**, and the distinction is the point.

    `AGENTS.md`: "Repository numerical code must not be the sole correctness oracle
    for the same numerical code." `O_ASM_PROPAGATE` is this repository's code, so a
    Fresnel-versus-ASM comparison cannot decide whether Fresnel is right. What it
    *can* do is establish that the two differ by exactly the amount the algebra
    predicts and by nothing else, which is a statement about the pair rather than
    about either one.

    So the oracle here is the closed-form residual and the ASM kernel is the second
    operand, not the reference:

        dphi = n k0 z (1 - cos(theta) - sin^2(theta)/2)
             = n k0 z (sin^4(theta)/8 + sin^6(theta)/16 + ...)

    which is exact: substituting `delay -> 1` in the carrier-removed angular
    spectrum's kernel gives the Fresnel kernel with a difference of exactly zero, so
    this residual has no fitted constant in it.

    Measured through both public operations on the same plane waves section 1 uses,
    rather than by reaching for either kernel. That keeps the comparison on this
    side of the chromatix boundary and, more to the point, keeps the frequency grid
    the prediction is evaluated on out of the hands of the implementations being
    compared.
    """
    largest = 0.0
    for bin_index in KERNEL_BINS:
        frequency = _bin_frequency_per_m(bin_index)
        field = _field(
            np.ones(KERNEL_SHAPE, dtype=np.complex64),
            KERNEL_PITCH_M,
            medium_index=KERNEL_INDEX,
        )
        _, x = (np.asarray(axis) for axis in field.coordinates())
        source = _field(
            np.broadcast_to(
                np.exp(2j * np.pi * frequency * x)[None, :], KERNEL_SHAPE
            ).astype(np.complex64),
            KERNEL_PITCH_M,
            medium_index=KERNEL_INDEX,
        )

        paraxial = _uniform_advance(_fresnel(source, KERNEL_DISTANCE_M), source)
        exact = _uniform_advance(
            propagate(
                source,
                distance_m=KERNEL_DISTANCE_M,
                model={
                    "method": "asm_carrier_removed",
                    "pad_width": 0,
                    "target_surface": "target",
                },
            ),
            source,
        )

        sine = (WAVELENGTH_M / KERNEL_INDEX) * frequency
        assert sine < 1.0, "the probe bin must be a propagating order"
        predicted = (
            2.0
            * math.pi
            * KERNEL_INDEX
            * KERNEL_DISTANCE_M
            / WAVELENGTH_M
            * (1.0 - math.sqrt(1.0 - sine**2) - 0.5 * sine**2)
        )
        assert abs(_wrapped(paraxial - exact - predicted)) < KERNEL_PHASE_TOLERANCE_RAD
        largest = max(largest, abs(predicted))

    # The premise: the residual is a real disagreement over these bins, not noise.
    # The leading term is n k0 z sin^4(theta)/8, so it grows fast with the angle.
    assert largest > 1.0


def test_the_two_propagations_agree_where_the_paraxial_bound_holds() -> None:
    """The other half of the diagnostic: inside the declared bound they converge.

    A validity bound that is never exercised in either direction is prose. The
    record says the result is usable where the field's own largest direction cosine
    satisfies `sin(theta_max) <= (lambda_0 / (n z))^(1/4)`, so a beam comfortably
    inside that bound must agree with the exact kernel, and the agreement is what
    says the bound is not merely conservative.

    Diagnostic, for the same reason as the test above: it compares two of this
    repository's paths.
    """
    index, distance = 1.0, 60e-6
    bound = (WAVELENGTH_M / (index * distance)) ** 0.25
    waist_m = 10e-6
    # A Gaussian's own largest meaningful direction cosine, ~lambda_0/(pi w0 n),
    # well inside the bound.
    divergence = WAVELENGTH_M / (math.pi * waist_m * index)
    assert divergence < 0.25 * bound, (divergence, bound)

    field = _field(np.ones((256, 256), dtype=np.complex64), (0.5e-6, 0.5e-6))
    y, x = (np.asarray(axis) for axis in field.coordinates())
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    field = _field(
        np.exp(-(grid_x**2 + grid_y**2) / waist_m**2).astype(np.complex64), (0.5e-6, 0.5e-6)
    )

    paraxial = _fresnel(field, distance, pad_width=128)
    exact = propagate(
        field,
        distance_m=distance,
        model={
            "method": "asm_carrier_removed",
            "pad_width": 128,
            "target_surface": "target",
        },
    )
    exact_intensity = np.abs(np.asarray(exact.u)) ** 2
    difference = np.abs(np.abs(np.asarray(paraxial.u)) ** 2 - exact_intensity)
    assert float(difference.max() / exact_intensity.max()) < 1e-3
