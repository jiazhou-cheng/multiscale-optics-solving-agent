"""The ideal coherent 4f relay, assembled from project primitives.

CHE-212 (R06.7). Run it:

    ./run.sh python -m benchmarks.systems.b4f_ideal

Two purposes, both required. It validates the physics of a **composition** --
per-operator tests cannot see a composition error, only a system can -- and it
demonstrates that a complete optical system is now expressible in this project's
public vocabulary rather than by calling a backend directly.

The system
----------
::

    source (R06.5)            normal-incidence plane wave
      -> complex transmission (R06.6)   the object, at the front focal plane
      -> focal-plane transform, f1 (R06.4)   -> Fourier plane, dx_F = lambda f1/(n N dx)
      -> complex transmission (R06.6)   the filter, at the Fourier plane
      -> focal-plane transform, f2 (R06.4)   -> image plane, dx_img = dx f2/f1

Run at `f1 == f2` and at `f1 != f2`, because several of the gates below are
trivially satisfied when the two are equal -- a magnification of -1 hides both the
`f1/f2` amplitude factor and the pitch change.

Four public functions in order, and no `System` class
-----------------------------------------------------
`operators/` may not import `solvers/` under the dependency allowlist, so nothing
in `src/` can hold this graph today. That is deliberate and it is not worked
around here: there is no `systems/` package, no composite-operator framework and
no `Pipeline`. The composition layer is R12/R13's question, and the fact that a
system currently reads as a script calling four functions is evidence for those
tickets rather than a defect this one should fix.

The one closed form that does most of the work
-----------------------------------------------
Two forward optical Fourier transforms compose into an exact statement. Each leg
carries the textbook `1/(i lambda f / n)` prefactor and the discrete relation
`DFT{DFT{U}}[k] = N U[-k]`, so with `dx_F = lambda f1 / (n N dx)`:

    U_img[k] = -(f1 / f2) * U_in[-k]        dx_img = dx * f2 / f1

Everything in that line is a separate claim the benchmark checks separately:

* the index mirroring is the **inversion** (criterion 1);
* `dx_img = dx f2/f1` with mirrored indices is the **magnification** `M = -f2/f1`
  in physical coordinates (criterion 2), and the pitch is an *independent*
  statement from it (criterion 4) -- a system can get the magnification right and
  the pitch wrong, and then every measured length is wrong by the same factor with
  nothing to reveal it;
* the `f1/f2` amplitude factor together with the pitch change conserves
  `discrete_power` exactly (criterion 7);
* the leading `-1` is `(1/i)^2`, a global pi that `|U|^2` cannot see. Both legs
  declare `carrier_removed_phase` for it, and the benchmark asserts the sign
  rather than comparing magnitudes.

Which oracles decide
--------------------
Every gate is closed-form Fourier optics: the discrete sampling relation, the
composition above, the Dirichlet kernel of a sampled boxcar, the Gaussian
transform pair, Jacobi--Anger's Bessel coefficients, and a pass/block predicate on
a stop radius. **No gate is another run of this repository's numerics.** The
`record.gate` entries carry `oracle_kind`, and a `diagnostic` entry may not decide
anything.

The direct `ifft2c(mask * fft2c(u))` NumPy model that CHE-144 used as a secondary
check is **not** run here, and that is stated rather than implied: the closed form
above is stronger than a differential check against another FFT, because it
predicts the amplitude factor, the sign and the pitch as well as the shape.
Adding a second numerical path would produce a number that agrees and decides
nothing.

Not covered, said plainly
--------------------------
The validity-envelope sweep over modulation frequency to the sampling limit --
CHE-144's most valuable output -- is explicitly optional in R06.7 and is **not
run**. Neither is a real or aberrated lens, ray-domain anything, partial
coherence, polarization, or a sensor model.

Cost: CPU, 2.8 s for both configurations on a 192 x 256 complex64 grid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import jv

from benchmarks.record import control, gate, write_record
from operators import (
    circular_aperture_amplitude,
    complex_transmission,
    numerical_aperture_radius_m,
)
from representations import ReferenceSurface, ScalarField
from solvers.chromatix import focal_plane_transform, fourier_plane_pitch_m
from sources import plane_wave

BENCHMARK_ID = "B-4F-IDEAL"
RECORDS = Path(__file__).resolve().parent / "records"

#: A NumPy array of unspecified dtype. An alias so the annotations below stay
#: readable; nothing in this module is generic over dtype.
Array = np.ndarray[Any, np.dtype[Any]]

WAVELENGTH_M = 0.532e-6
MEDIUM_INDEX = 1.0

#: 192 x 256 at 0.6 x 0.5 um. Asymmetric in **both** count and pitch: the two axes
#: then have different extents and different Fourier pitches, so a transposed
#: `(y, x)` cannot pass any gate below, and a mask built for the wrong axis is a
#: shape error rather than a plausible answer.
SHAPE = (192, 256)
PITCH_M = (0.6e-6, 0.5e-6)

#: The complex64 floor for a two-leg system, derived rather than fitted. float32
#: carries 1.19e-7 relative; each leg is one FFT pair whose rounding scale is set
#: by its largest term, so two legs accumulate a small multiple of that. Four
#: epsilons is 4.8e-7, and 5e-7 is the bound used.
#:
#: The worst residual actually measured against it is the grating relay's
#: `image_residual` at 3.17e-7 -- 2.7 epsilons, a margin of 1.6x, not the comfortable
#: factor the round number suggests. The reason it is kept anyway is that every
#: failure this gate exists to catch is O(1) and not marginal: the same comparison
#: without the index mirroring is 1.0 and without the global pi is 2.0, both
#: recorded beside the residual. A tolerance whose margin is 1.6x over noise and
#: 10^6 under every real error is doing its job; if that margin ever matters, the
#: fix is a float64 path and not a wider number.
COMPLEX64_FLOOR = 5e-7

#: The stop, as a numerical aperture rather than as a radius, so the radius comes
#: from `numerical_aperture_radius_m` and the cutoff frequency `NA/lambda` is
#: available as an independent statement. NA 0.1 at f1 = 20 mm is a 2.000 mm stop,
#: which spans 24.06 samples of this Fourier plane's own x pitch -- a *discretized*
#: aperture, and the record says how many samples it is rather than leaving a
#: reader to assume it is smooth.
NUMERICAL_APERTURE = 0.1

#: Object frequencies for the filtering gate, in bins of `1/(N dx_in)`. 15 is
#: inside the analytic cutoff (24.06 bins) and 35 is outside it, both by a wide
#: enough margin that the predicate is not a boundary case -- the boundary case
#: itself is gated in `tests/physics/test_thin_element_spectrum.py`.
PASSBAND_BIN = 15
STOPBAND_BIN = 35
PASSBAND_DEPTH = 0.4
STOPBAND_DEPTH = 0.3

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"name": "unit_magnification", "focal_length_1_m": 20e-3, "focal_length_2_m": 20e-3},
    {"name": "magnifying_relay", "focal_length_1_m": 20e-3, "focal_length_2_m": 40e-3},
)


# ---------------------------------------------------------------------------
# The composition path: four public calls, and the coordinates to read them on
# ---------------------------------------------------------------------------


def _surface(name: str) -> ReferenceSurface:
    return ReferenceSurface(name=name, z_m=0.0, medium_index=MEDIUM_INDEX)


def _illumination(transverse_wavevector: tuple[float, float] = (0.0, 0.0)) -> ScalarField:
    return plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=_surface("front_focal"),
        transverse_wavevector_rad_per_m=transverse_wavevector,
    )


def _leg(
    field: ScalarField, focal_length_m: float, target: str, *, direction: str = "forward"
) -> ScalarField:
    return focal_plane_transform(
        field,
        focal_length_m=focal_length_m,
        model={"target_surface": target, "direction": direction},
    )


def _axes(field: ScalarField) -> tuple[Array, Array]:
    y, x = field.coordinates()
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _mirror(array: Array) -> Array:
    """`a[-k]` on the `n // 2` origin, per axis.

    Index `j` maps to `2 * (n // 2) - j`, which for even `n` is `n - j` and so is
    `roll(flip(a), 1)`. Written once here because the whole inversion claim rests
    on it and `flip` alone -- `a[n - 1 - j]` -- is off by exactly one sample, a
    half-window phase ramp's worth of error that a symmetric input would not show.
    """
    mirrored: Array = np.roll(np.flip(array, axis=(0, 1)), (1, 1), axis=(0, 1))
    return mirrored


def _centroid_and_width(field: ScalarField) -> tuple[float, float, float, float]:
    """`(cy, cx, sigma_y, sigma_x)` of `|u|^2`, in metres, on the field's own axes."""
    intensity = np.abs(np.asarray(field.u)) ** 2
    y, x = _axes(field)
    total = float(intensity.sum())
    rows = intensity.sum(axis=1)
    columns = intensity.sum(axis=0)
    cy = float((rows * y).sum() / total)
    cx = float((columns * x).sum() / total)
    return (
        cy,
        cx,
        math.sqrt(float((rows * (y - cy) ** 2).sum() / total)),
        math.sqrt(float((columns * (x - cx) ** 2).sum() / total)),
    )


def _row_spectrum(field: ScalarField) -> Array:
    """`|DFT|` along `x` of the field's own centre row.

    For a field in a *position* domain -- the object or the image -- this is its
    line spectrum in bins. Bin indices survive both legs (leg two maps bin `j` to
    `-j`), so the same bin can be read at the object and at the image while the
    *physical* frequency it means changes with the pitch. That is the point of
    reading it this way, and it is why this must not be applied to a field that is
    already at the Fourier plane: `_fourier_row` is that case.
    """
    spectrum: Array = np.abs(np.fft.fft(np.asarray(field.u)[SHAPE[0] // 2]))
    return spectrum


def _fourier_row(field: ScalarField) -> Array:
    """`|u|` along the centre row of a field that is *already* at a Fourier plane.

    Read directly off the array, because the transform has happened: taking
    another DFT here would return the object and read as a spectrum that is
    exactly flat wherever the object was flat -- which is a plausible-looking wrong
    answer rather than an error.
    """
    row: Array = np.abs(np.asarray(field.u)[SHAPE[0] // 2])
    return row


# ---------------------------------------------------------------------------
# The objects, all built through `complex_transmission`
# ---------------------------------------------------------------------------


def _asymmetric_object() -> ScalarField:
    """Two unequal off-centre lobes under a two-axis phase carrier.

    Deliberately not a centred Gaussian: a symmetric input cannot detect an
    inversion failure and neither can a symmetric grid. The two lobes differ in
    position, width and weight, and the carrier makes the field complex, so
    `conj(U)` is a different field and the phasor control has something to break.
    """
    field = _illumination()
    y, x = _axes(field)
    waist_m = 12e-6
    amplitude = np.exp(
        -(((y[:, None] - 18e-6) ** 2 + (x[None, :] - 25e-6) ** 2) / waist_m**2)
    )
    amplitude += 0.45 * np.exp(
        -(((y[:, None] + 30e-6) ** 2 + (x[None, :] + 8e-6) ** 2) / (0.45 * waist_m) ** 2)
    )
    amplitude /= amplitude.max()
    phase = 3.0e5 * np.broadcast_to(x[None, :], SHAPE) + 1.0e5 * np.broadcast_to(
        y[:, None], SHAPE
    )
    return complex_transmission(
        field, amplitude=amplitude, phase_rad=phase.copy(), target_surface="object"
    )


def _gaussian_object(waist_m: float) -> ScalarField:
    field = _illumination()
    y, x = _axes(field)
    return complex_transmission(
        field,
        amplitude=np.exp(-((y[:, None] ** 2 + x[None, :] ** 2) / waist_m**2)),
        target_surface="object",
    )


def _slit_object(width_samples: int) -> ScalarField:
    """A boxcar `width_samples` wide in `x`, uniform in `y`.

    An integer sample count, and one that divides the grid: the sampled boxcar's
    transform is the Dirichlet kernel with **exact** zeros at bins that are
    multiples of `N / L`, which is a closed form rather than the continuous
    `sinc`'s approximation to it.
    """
    field = _illumination()
    mask = np.zeros(SHAPE[1], dtype=np.float64)
    origin = SHAPE[1] // 2
    mask[origin - width_samples // 2 : origin + width_samples // 2] = 1.0
    return complex_transmission(
        field,
        amplitude=np.broadcast_to(mask[None, :], SHAPE).copy(),
        target_surface="object",
    )


def _sinusoidal_grating_object(depth_rad: float, periods: int) -> ScalarField:
    field = _illumination()
    _, x = _axes(field)
    period_m = SHAPE[1] * PITCH_M[1] / periods
    profile = depth_rad * np.sin(2.0 * np.pi * x / period_m)
    return complex_transmission(
        field,
        phase_rad=np.broadcast_to(profile[None, :], SHAPE).copy(),
        target_surface="object",
    )


def _two_frequency_object() -> tuple[ScalarField, float]:
    """`(1 + a cos(2 pi f_a x) + b cos(2 pi f_b x)) / (1 + a + b)`, and its norm.

    An exact five-line spectrum on this grid, because both frequencies are integer
    bins of `1/(N dx)`. That is what makes the transmitted power fraction in
    criterion 7 an arithmetic statement rather than an integral.
    """
    field = _illumination()
    _, x = _axes(field)
    window_m = SHAPE[1] * PITCH_M[1]
    profile = (
        1.0
        + PASSBAND_DEPTH * np.cos(2.0 * np.pi * PASSBAND_BIN * x / window_m)
        + STOPBAND_DEPTH * np.cos(2.0 * np.pi * STOPBAND_BIN * x / window_m)
    )
    norm = 1.0 + PASSBAND_DEPTH + STOPBAND_DEPTH
    return (
        complex_transmission(
            field,
            amplitude=np.broadcast_to((profile / norm)[None, :], SHAPE).copy(),
            target_surface="object",
        ),
        norm,
    )


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def _sampling_gates(
    fourier: ScalarField, image: ScalarField, *, f1: float, f2: float
) -> list[dict[str, Any]]:
    """Criteria 3 and 4: the two sampling relations, as independent statements."""
    analytic_fourier = fourier_plane_pitch_m(
        PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    from_scratch = tuple(
        WAVELENGTH_M * f1 / (MEDIUM_INDEX * count * pitch)
        for count, pitch in zip(SHAPE, PITCH_M, strict=True)
    )
    analytic_image = tuple(pitch * f2 / f1 for pitch in PITCH_M)

    gates = [
        gate(
            "fourier_plane_pitch",
            oracle="dx_F = lambda f1 / (n N dx_in), per axis, recomputed here from the "
            "formula rather than by calling the same function the operator declares with",
            oracle_kind="closed_form",
            measured=list(fourier.sample_pitch_m),
            expected=list(from_scratch),
            tolerance=1e-15,
            tolerance_basis=(
                "float64 on both sides, at 1e-15 rather than exact equality because the "
                "recomputation groups the arithmetic differently from the library function "
                "-- (lambda f)/(n N dx) against (lambda f / n)/(N dx) -- and a 1-ulp "
                "associativity difference is not a sampling error. The separate exact "
                "equality against `fourier_plane_pitch_m` is kept beside it as what it "
                "actually is: the statement that the boundary carried the operator's own "
                "declaration through unchanged, which is a tautology about the formula and "
                "so cannot be the thing that decides"
            ),
            passed=(
                all(
                    abs(got - want) <= 1e-15 * want
                    for got, want in zip(fourier.sample_pitch_m, from_scratch, strict=True)
                )
                and fourier.sample_pitch_m == analytic_fourier
            ),
        ),
        gate(
            "image_plane_pitch",
            oracle="dx_img = dx_in f2 / f1, per axis",
            oracle_kind="closed_form",
            measured=list(image.sample_pitch_m),
            expected=list(analytic_image),
            tolerance=1e-15,
            tolerance_basis=(
                "float64 arithmetic on both sides; an independent claim from the "
                "magnification, because a system can invert correctly and still report "
                "every length wrong by f2/f1"
            ),
            passed=all(
                abs(got - want) <= 1e-15 * want
                for got, want in zip(image.sample_pitch_m, analytic_image, strict=True)
            ),
        ),
    ]
    return gates


def _frequency_axis_gate(f1: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Criterion 3, second half, plus the numbers the negative controls reuse.

    A single spatial frequency enters as a tilted plane wave and must land at the
    sample whose offset from the origin is `j`, i.e. at `x_F = j lambda f1 /
    (n N dx_in)`. Measured two ways -- the peak sample and the intensity centroid
    -- because the first is what a discretized reading gives and the second is what
    survives an off-grid carrier, and the controls need both.
    """
    window_m = SHAPE[1] * PITCH_M[1]
    bin_index = PASSBAND_BIN
    wavevector = 2.0 * math.pi * bin_index / window_m

    fourier = _leg(_illumination((0.0, wavevector)), f1, "fourier")
    intensity = np.abs(np.asarray(fourier.u)) ** 2
    peak = np.unravel_index(int(np.argmax(intensity)), SHAPE)
    _, x_axis = _axes(fourier)
    columns = intensity.sum(axis=0)
    centroid_m = float((columns * x_axis).sum() / columns.sum())
    concentration = float(intensity.max() / intensity.sum())

    analytic_m = bin_index * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_m)
    #: The mistake the control names: reading `k_x` in rad/m as if it were a
    #: spatial frequency in cycles/m puts the spot 2 pi times further out.
    two_pi_wrong_m = 2.0 * math.pi * analytic_m

    measured_m = float(x_axis[peak[1]])
    gates = [
        gate(
            "fourier_plane_frequency_axis",
            oracle="the sample at offset j is the spatial frequency j / (N dx_in), i.e. "
            "x_F = j lambda f1 / (n N dx_in)",
            oracle_kind="closed_form",
            measured={
                "peak_index_offset": [
                    int(peak[0]) - SHAPE[0] // 2,
                    int(peak[1]) - SHAPE[1] // 2,
                ],
                "peak_position_m": measured_m,
                "centroid_m": centroid_m,
                "peak_energy_fraction": concentration,
            },
            expected={"index_offset": [0, bin_index], "position_m": analytic_m},
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "the carrier sits on an integer DFT bin, so the whole plane wave lands on "
                "one output sample and the position is the declared pitch times an integer; "
                "the tolerance covers only the float32 storage of that pitch"
            ),
            passed=(
                int(peak[0]) - SHAPE[0] // 2 == 0
                and int(peak[1]) - SHAPE[1] // 2 == bin_index
                and abs(measured_m / analytic_m - 1.0) < COMPLEX64_FLOOR
                and concentration > 0.99
            ),
        )
    ]
    return gates, {
        "analytic_m": analytic_m,
        "measured_m": measured_m,
        "centroid_m": centroid_m,
        "concentration": concentration,
        "two_pi_wrong_m": two_pi_wrong_m,
        "wavevector": wavevector,
    }


def _composition_gates(f1: float, f2: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Criteria 1, 2 and 7 (open filter): the one exact closed form, three ways."""
    source = _asymmetric_object()
    fourier = _leg(source, f1, "fourier")
    #: The open filter is the *same* operator with both factors at their identity,
    #: which is R06.6's whole design: an open pupil is not a special case that has
    #: to be skipped.
    image = _leg(complex_transmission(fourier, amplitude=1.0), f2, "image")

    incoming = np.asarray(source.u)
    predicted = (-(f1 / f2) * _mirror(incoming)).astype(np.complex64)
    outgoing = np.asarray(image.u)
    scale = float(np.max(np.abs(predicted)))
    inversion_residual = float(np.max(np.abs(outgoing - predicted)) / scale)
    #: The falsifiable twin: the same comparison without the index mirroring. An
    #: upright image reads as 1.0 here, so the mirroring is the whole difference.
    upright_residual = float(
        np.max(np.abs(outgoing - (-(f1 / f2) * incoming).astype(np.complex64))) / scale
    )
    #: ...and without the global pi.
    unsigned_residual = float(
        np.max(np.abs(outgoing - ((f1 / f2) * _mirror(incoming)).astype(np.complex64))) / scale
    )

    before = _centroid_and_width(source)
    after = _centroid_and_width(image)
    magnification = -f2 / f1
    magnification_measured = [after[0] / before[0], after[1] / before[1]]
    width_ratio_measured = [after[2] / before[2], after[3] / before[3]]

    power_ratio = image.discrete_power() / source.discrete_power()

    gates = [
        gate(
            "image_is_minus_f1_over_f2_times_the_mirrored_input",
            oracle="U_img[k] = -(f1/f2) U_in[-k], from DFT{DFT{U}}[k] = N U[-k] and the "
            "1/(i lambda f / n) prefactor on each leg",
            oracle_kind="closed_form",
            measured={
                "residual": inversion_residual,
                "residual_without_mirroring": upright_residual,
                "residual_without_the_global_pi": unsigned_residual,
            },
            expected=0.0,
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "two FFT legs in complex64: four float32 epsilons is 4.8e-7. The two "
                "falsifiable twins land at O(1), so this is not a tolerance that could "
                "absorb a composition error"
            ),
            passed=(
                inversion_residual < COMPLEX64_FLOOR
                and upright_residual > 0.5
                and unsigned_residual > 0.5
            ),
        ),
        gate(
            "magnification",
            oracle="M = -f2/f1, as a feature's centroid and its second moment at both planes",
            oracle_kind="closed_form",
            measured={
                "centroid_ratio_yx": magnification_measured,
                "width_ratio_yx": width_ratio_measured,
            },
            expected={"centroid_ratio": magnification, "width_ratio": abs(magnification)},
            tolerance=1e-5,
            tolerance_basis=(
                "a ratio of two float32 second moments over the same window; 1e-5 is two "
                "orders above the 1.5e-7 measured and still far below any plausible "
                "magnification error, which would be a factor of f2/f1"
            ),
            passed=(
                all(abs(m / magnification - 1.0) < 1e-5 for m in magnification_measured)
                and all(abs(r / abs(magnification) - 1.0) < 1e-5 for r in width_ratio_measured)
            ),
        ),
        gate(
            "power_through_an_open_filter",
            oracle="the f1/f2 amplitude factor against the f2/f1 pitch change conserves "
            "sum |u|^2 dy dx exactly",
            oracle_kind="closed_form",
            measured=power_ratio,
            expected=1.0,
            tolerance=1e-6,
            tolerance_basis=(
                "a sum of squares in float32: two epsilons on the amplitude is ~2.4e-7 on "
                "the power, and the measured deviation is 2.36e-7 -- i.e. this gate runs at "
                "its derived floor with a factor of four in hand, not with orders. It is "
                "the gate that catches a normalization error nothing else here can see, "
                "which is why no comparison in this benchmark is normalized by its own peak"
            ),
            passed=abs(power_ratio - 1.0) < 1e-6,
        ),
        gate(
            "the_four_f_length_is_2f1_plus_2f2",
            oracle="each leg advances the declared surface by 2f",
            oracle_kind="closed_form",
            measured={
                "z_fourier_m": fourier.reference_surface.z_m,
                "z_image_m": image.reference_surface.z_m,
            },
            expected={"z_fourier_m": 2.0 * f1, "z_image_m": 2.0 * f1 + 2.0 * f2},
            tolerance=0.0,
            tolerance_basis="float64 addition of two declared focal lengths",
            passed=(
                fourier.reference_surface.z_m == 2.0 * f1
                and image.reference_surface.z_m == 2.0 * f1 + 2.0 * f2
            ),
        ),
    ]
    return gates, {"source": source, "fourier": fourier, "image": image}


def _transform_pair_gates(f1: float, f2: float) -> list[dict[str, Any]]:
    """Criterion 5: three known transform pairs, positions and amplitudes."""
    gates: list[dict[str, Any]] = []

    # -- a Gaussian returns a Gaussian of the analytically predicted waist -------
    #: 5 um: ten samples across the input waist, and 8.1 samples across the Fourier
    #: waist (`w_F / dx_F = N dx / (pi w0)`), so neither plane is undersampled.
    waist_m = 5e-6
    fourier = _leg(_gaussian_object(waist_m), f1, "fourier")
    predicted_waist = tuple(
        WAVELENGTH_M * f1 / (math.pi * MEDIUM_INDEX * waist_m) for _ in range(2)
    )
    measured = _centroid_and_width(fourier)
    # For an amplitude exp(-r^2/w^2) the intensity is exp(-2r^2/w^2), whose second
    # moment is w/2. The relation w_F = lambda f / (pi n w0) is the Fourier
    # transform pair of a Gaussian, not a fit.
    measured_waist = (2.0 * measured[2], 2.0 * measured[3])
    gates.append(
        gate(
            "gaussian_transforms_to_a_gaussian_of_the_predicted_waist",
            oracle="w_F = lambda f1 / (pi n w0); intensity second moment is w/2",
            oracle_kind="closed_form",
            measured=list(measured_waist),
            expected=list(predicted_waist),
            tolerance=2e-2,
            tolerance_basis=(
                "2e-2 is **borrowed**, not derived for this case: it is the threshold the "
                "B1-WAVE Gaussian-spreading family justified for the same second-moment "
                "estimator, where window truncation dominated. Here it does not -- the "
                "Fourier waist is 15.7 (x) and 13.0 (y) waists inside the window and the "
                "measured error is 4e-8 -- so this bound is loose-but-safe rather than "
                "tight. It is kept because it still rejects every plausible error in the "
                "relation (a missing pi is 3.14x, a missing n is 1.33x at index 1.33), and "
                "tightening it to the measurement would be fitting"
            ),
            passed=all(
                abs(got / want - 1.0) < 2e-2
                for got, want in zip(measured_waist, predicted_waist, strict=True)
            ),
        )
    )

    # -- a slit gives the sampled boxcar's Dirichlet kernel ----------------------
    width_samples = 16
    row = _fourier_row(_leg(_slit_object(width_samples), f1, "fourier"))
    origin = SHAPE[1] // 2
    peak = float(row[origin])
    zero_bin = SHAPE[1] // width_samples  # exact zeros at multiples of N / L
    nulls = [
        float(row[origin + k] / peak) for k in (zero_bin, 2 * zero_bin, 3 * zero_bin)
    ]
    checked = (1, 3, 5, 7)
    dirichlet = {
        k: abs(
            math.sin(math.pi * k * width_samples / SHAPE[1])
            / math.sin(math.pi * k / SHAPE[1])
        )
        / width_samples
        for k in checked
    }
    lobes = {k: float(row[origin + k] / peak) for k in checked}
    null_analytic_m = [
        m * WAVELENGTH_M * f1 / (MEDIUM_INDEX * width_samples * PITCH_M[1]) for m in (1, 2, 3)
    ]
    gates.append(
        gate(
            "slit_gives_the_sampled_boxcars_dirichlet_kernel",
            oracle="|D(k)| / L with D(k) = sin(pi k L / N) / sin(pi k / N); zeros at "
            "k = m N / L, i.e. x_F = m lambda f1 / (n w)",
            oracle_kind="closed_form",
            measured={
                "null_amplitudes": nulls,
                "null_positions_m": null_analytic_m,
                "lobe_amplitudes": {str(k): lobes[k] for k in checked},
            },
            expected={"nulls": 0.0, "lobes": {str(k): dirichlet[k] for k in checked}},
            tolerance=1e-6,
            tolerance_basis=(
                "the sampled boxcar's transform is the Dirichlet kernel exactly, so the "
                "only error is complex64 storage. The continuous sinc would need a percent-"
                "level tolerance covering a known discretization effect, which is why the "
                "sampled form is the oracle"
            ),
            passed=(
                all(value < 1e-6 for value in nulls)
                and all(abs(lobes[k] - dirichlet[k]) < 1e-6 for k in checked)
            ),
        )
    )

    # -- a sinusoidal phase grating gives Bessel orders, and the image reproduces it
    depth_rad = 1.5
    periods = 8
    grating = _sinusoidal_grating_object(depth_rad, periods)
    grating_fourier = _leg(grating, f1, "fourier")
    row = np.asarray(grating_fourier.u)[SHAPE[0] // 2]
    origin = SHAPE[1] // 2
    orders = {n: complex(row[origin + n * periods]) for n in range(-3, 4)}
    order_residuals = {
        str(n): abs(orders[n] / orders[0] - jv(n, depth_rad) / jv(0, depth_rad))
        for n in orders
    }
    grating_image = _leg(complex_transmission(grating_fourier, amplitude=1.0), f2, "image")
    image_residual = float(
        np.max(
            np.abs(
                np.asarray(grating_image.u)
                - (-(f1 / f2) * _mirror(np.asarray(grating.u))).astype(np.complex64)
            )
        )
        / (f1 / f2)
    )
    gates.append(
        gate(
            "sinusoidal_phase_grating_gives_bessel_orders_and_relays_end_to_end",
            oracle="Jacobi-Anger: A_n / A_0 = J_n(m) / J_0(m) at x_n = n lambda f1 / "
            "(n_med Lambda), signed; and the image reproduces the object",
            oracle_kind="closed_form",
            measured={
                "order_residuals": order_residuals,
                "image_residual": image_residual,
                "order_positions_m": [
                    n * WAVELENGTH_M * f1 / (MEDIUM_INDEX * SHAPE[1] * PITCH_M[1] / periods)
                    for n in (1, 2, 3)
                ],
            },
            expected=0.0,
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "the FFT's rounding scale is set by its largest term, so the *absolute* "
                "residual on A_n/A_0 is a few float32 epsilons however small A_n is -- "
                "which is why the comparison is absolute. The signed ratio is asserted, "
                "because J_{-n} = (-1)^n J_n and a magnitude-only check would pass a "
                "mirrored spectrum"
            ),
            passed=(
                all(value < COMPLEX64_FLOOR for value in order_residuals.values())
                and image_residual < COMPLEX64_FLOOR
            ),
        )
    )
    return gates


def _filtering_gates(f1: float, f2: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Criteria 6 and 7 (with a filter): the stop removes exactly what it should."""
    source, norm = _two_frequency_object()
    fourier = _leg(source, f1, "fourier")

    radius_m = numerical_aperture_radius_m(
        NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    cutoff_per_m = NUMERICAL_APERTURE / WAVELENGTH_M
    window_m = SHAPE[1] * PITCH_M[1]
    cutoff_bin = cutoff_per_m * window_m
    stop = circular_aperture_amplitude(
        SHAPE, sample_pitch_m=fourier.sample_pitch_m, radius_m=radius_m, edge="hard"
    )
    filtered = complex_transmission(fourier, amplitude=stop, target_surface="fourier_stopped")
    image = _leg(filtered, f2, "image")

    #: Which lines the analytic predicate keeps: |f| <= NA/lambda, i.e. |bin| <=
    #: cutoff_bin. Nothing about this reads the simulation.
    survives = {
        0: True,
        PASSBAND_BIN: cutoff_bin >= PASSBAND_BIN,
        STOPBAND_BIN: cutoff_bin >= STOPBAND_BIN,
    }
    spectrum = _row_spectrum(image)
    ratios = {
        str(bin_index): float(spectrum[bin_index] / spectrum[0])
        for bin_index in (PASSBAND_BIN, STOPBAND_BIN)
    }
    #: The Fourier coefficients of (1 + a cos + b cos)/norm are 1/norm at DC and
    #: a/(2 norm), b/(2 norm) at the sidebands, so the *ratio* to DC is a/2 and
    #: b/2 with the norm cancelling.
    expected_ratios = {
        str(PASSBAND_BIN): PASSBAND_DEPTH / 2.0 if survives[PASSBAND_BIN] else 0.0,
        str(STOPBAND_BIN): STOPBAND_DEPTH / 2.0 if survives[STOPBAND_BIN] else 0.0,
    }

    transmitted_fraction = image.discrete_power() / source.discrete_power()
    line_power = {
        0: 1.0,
        PASSBAND_BIN: 2.0 * (PASSBAND_DEPTH / 2.0) ** 2,
        STOPBAND_BIN: 2.0 * (STOPBAND_DEPTH / 2.0) ** 2,
    }
    expected_fraction = sum(
        power for bin_index, power in line_power.items() if survives[bin_index]
    ) / sum(line_power.values())

    gates = [
        gate(
            "the_stop_removes_exactly_the_frequencies_above_its_cutoff",
            oracle="pass iff |f| <= NA/lambda; the surviving sideband keeps its analytic "
            "amplitude ratio a/2 to the DC line",
            oracle_kind="closed_form",
            measured={
                "sideband_to_dc": ratios,
                "cutoff_bin": cutoff_bin,
                "stop_radius_m": radius_m,
                "stop_radius_in_fourier_samples": [
                    radius_m / fourier.sample_pitch_m[0],
                    radius_m / fourier.sample_pitch_m[1],
                ],
            },
            expected=expected_ratios,
            tolerance=1e-5,
            tolerance_basis=(
                "a ratio of two line amplitudes read off one complex64 DFT: a few float32 "
                "epsilons, with the blocked line at the arithmetic zero of an exact "
                "elementwise multiply by 0"
            ),
            passed=all(
                abs(ratios[key] - expected_ratios[key]) < 1e-5 for key in expected_ratios
            ),
        ),
        gate(
            "power_through_the_stop_is_the_analytically_transmitted_fraction",
            oracle="the transmitted fraction is the sum of |c_n|^2 over the surviving "
            "lines, over the sum over all of them",
            oracle_kind="closed_form",
            measured=transmitted_fraction,
            expected=expected_fraction,
            tolerance=1e-5,
            tolerance_basis=(
                "the object is an exact five-line spectrum on this grid, so the fraction is "
                "arithmetic on |c_n|^2; the residual is the same float32 power floor as the "
                "open-filter case, one order looser because two power sums are divided"
            ),
            passed=abs(transmitted_fraction / expected_fraction - 1.0) < 1e-5,
        ),
    ]
    return gates, {
        "source": source,
        "fourier": fourier,
        "stop": stop,
        "cutoff_bin": cutoff_bin,
        "norm": norm,
        "expected_ratios": expected_ratios,
    }


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


def _controls(f1: float, f2: float, filtering: dict[str, Any]) -> list[dict[str, Any]]:
    """Every one of CHE-144's controls, plus the one this boundary adds.

    Each is a *deliberately wrong* run whose only purpose is to break a gate that
    the correct run passes. A control that does not break its gate means the gate
    was not measuring what it claimed.
    """
    controls: list[dict[str, Any]] = []
    source = _asymmetric_object()
    incoming = np.asarray(source.u)
    predicted = (-(f1 / f2) * _mirror(incoming)).astype(np.complex64)
    scale = float(np.max(np.abs(predicted)))

    def relay(field: ScalarField, *, second_leg: str = "forward") -> Array:
        fourier = _leg(field, f1, "fourier")
        image = _leg(
            complex_transmission(fourier, amplitude=1.0), f2, "image", direction=second_leg
        )
        return np.asarray(image.u)

    # 1. Phasor-sign flip, expressed where it can be: the object's phase is
    #    negated, which is exactly conj(U) for a real-amplitude element on a
    #    normal-incidence illumination.
    y, x = _axes(source)
    phase = 3.0e5 * np.broadcast_to(x[None, :], SHAPE) + 1.0e5 * np.broadcast_to(
        y[:, None], SHAPE
    )
    conjugated = complex_transmission(
        _illumination(),
        amplitude=np.abs(incoming),
        phase_rad=-phase.copy(),
        target_surface="object",
    )
    residual = float(np.max(np.abs(relay(conjugated) - predicted)) / scale)
    controls.append(
        control(
            "phasor_sign_flip",
            changed="the object's phase negated, i.e. the conjugate phasor convention",
            breaks_gate="image_is_minus_f1_over_f2_times_the_mirrored_input",
            measured=residual,
            reference=COMPLEX64_FLOOR,
            broke=residual > 0.5,
        )
    )

    # 2. A transposed axis: the same single-frequency carrier put on `k_y` instead
    #    of `k_x`. The grid is asymmetric, so this is not a relabelling.
    window_x_m = SHAPE[1] * PITCH_M[1]
    wavevector = 2.0 * math.pi * PASSBAND_BIN / window_x_m
    analytic_m = PASSBAND_BIN * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_x_m)
    transposed = _leg(_illumination((wavevector, 0.0)), f1, "fourier")
    intensity = np.abs(np.asarray(transposed.u)) ** 2
    peak = np.unravel_index(int(np.argmax(intensity)), SHAPE)
    _, x_axis = _axes(transposed)
    measured_m = float(x_axis[peak[1]])
    controls.append(
        control(
            "transposed_axis",
            changed="the carrier moved from k_x to k_y",
            breaks_gate="fourier_plane_frequency_axis",
            measured={
                "peak_index_offset": [
                    int(peak[0]) - SHAPE[0] // 2,
                    int(peak[1]) - SHAPE[1] // 2,
                ],
                "x_position_m": measured_m,
            },
            reference={"index_offset": [0, PASSBAND_BIN], "x_position_m": analytic_m},
            broke=int(peak[1]) - SHAPE[1] // 2 != PASSBAND_BIN,
        )
    )

    # 3. The filter at the image plane instead of the Fourier plane. A stop there
    #    does not select spatial frequencies at all, so the out-of-band line
    #    survives -- and the resulting image is perfectly plausible.
    two_frequency = filtering["source"]
    unfiltered_image = _leg(
        complex_transmission(_leg(two_frequency, f1, "fourier"), amplitude=1.0), f2, "image"
    )
    misplaced = complex_transmission(
        unfiltered_image,
        amplitude=circular_aperture_amplitude(
            SHAPE,
            sample_pitch_m=unfiltered_image.sample_pitch_m,
            radius_m=numerical_aperture_radius_m(
                NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
            ),
            edge="hard",
        ),
    )
    spectrum = _row_spectrum(misplaced)
    stopband_ratio = float(spectrum[STOPBAND_BIN] / spectrum[0])
    controls.append(
        control(
            "filter_at_the_image_plane",
            changed="the same stop applied after leg two instead of between the legs",
            breaks_gate="the_stop_removes_exactly_the_frequencies_above_its_cutoff",
            measured={"stopband_to_dc": stopband_ratio},
            reference=filtering["expected_ratios"][str(STOPBAND_BIN)],
            broke=abs(stopband_ratio - filtering["expected_ratios"][str(STOPBAND_BIN)]) > 1e-5,
        )
    )

    # 4. A 2 pi scale error in the frequency grid: `k_x` in rad/m read as a spatial
    #    frequency in cycles/m.
    correct = _leg(_illumination((0.0, wavevector)), f1, "fourier")
    _, correct_x = _axes(correct)
    correct_m = float(
        correct_x[int(np.argmax(np.abs(np.asarray(correct.u)) ** 2) % SHAPE[1])]
    )
    wrong_prediction_m = 2.0 * math.pi * analytic_m
    controls.append(
        control(
            "two_pi_frequency_scale",
            changed="the Fourier-plane position predicted from k_x as if it were cycles/m",
            breaks_gate="fourier_plane_frequency_axis",
            measured={"position_m": correct_m},
            reference={"wrong_prediction_m": wrong_prediction_m},
            broke=abs(correct_m / wrong_prediction_m - 1.0) > 0.1,
        )
    )

    # 5. Grid-snapped vs continuous carrier placement. Not a wrong *model* -- a
    #    different case -- so what it breaks is the single-sample reading, while
    #    the centroid survives. R06.5 characterizes this; recording it here is what
    #    keeps R06.8's angle sweep from discovering it.
    off_grid = _leg(
        _illumination((0.0, 2.0 * math.pi * (PASSBAND_BIN + 0.5) / window_x_m)), f1, "fourier"
    )
    off_intensity = np.abs(np.asarray(off_grid.u)) ** 2
    concentration = float(off_intensity.max() / off_intensity.sum())
    _, off_x = _axes(off_grid)
    off_columns = off_intensity.sum(axis=0)
    off_centroid_m = float((off_columns * off_x).sum() / off_columns.sum())
    off_analytic_m = (PASSBAND_BIN + 0.5) * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_x_m)
    controls.append(
        control(
            "carrier_off_the_dft_grid",
            changed="the carrier moved half a bin off the DFT frequency grid",
            breaks_gate="fourier_plane_frequency_axis (its peak_energy_fraction > 0.99 part)",
            measured={
                "peak_energy_fraction": concentration,
                "centroid_m": off_centroid_m,
                "centroid_relative_error": off_centroid_m / off_analytic_m - 1.0,
            },
            reference={
                "peak_energy_fraction_on_grid": 1.0,
                "dirichlet_half_bin": (2.0 / math.pi) ** 2,
                "analytic_position_m": off_analytic_m,
            },
            broke=concentration < 0.99,
        )
    )

    # 6. The second leg run as an inverse transform. This is the control this
    #    project's boundary adds: it yields an upright image and a system that is
    #    not a 4f relay, and nothing in the artifact says so -- the pitch, the
    #    power and the declared surface all still look right.
    inverse_residual = float(
        np.max(np.abs(relay(source, second_leg="inverse") - predicted)) / scale
    )
    controls.append(
        control(
            "second_leg_as_an_inverse_transform",
            changed="model['direction']='inverse' on leg two",
            breaks_gate="image_is_minus_f1_over_f2_times_the_mirrored_input",
            measured=inverse_residual,
            reference=COMPLEX64_FLOOR,
            broke=inverse_residual > 0.5,
        )
    )
    return controls


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def run(configuration: dict[str, Any]) -> dict[str, Any]:
    """Run every gate and every control for one `(f1, f2)`, and build its record."""
    f1 = float(configuration["focal_length_1_m"])
    f2 = float(configuration["focal_length_2_m"])

    composition, artifacts = _composition_gates(f1, f2)
    frequency, _ = _frequency_axis_gate(f1)
    filtering, filtering_state = _filtering_gates(f1, f2)

    gates = [
        *_sampling_gates(artifacts["fourier"], artifacts["image"], f1=f1, f2=f2),
        *frequency,
        *composition,
        *_transform_pair_gates(f1, f2),
        *filtering,
    ]
    controls = _controls(f1, f2, filtering_state)

    return {
        "benchmark": BENCHMARK_ID,
        "ticket": "CHE-212",
        "configuration": configuration["name"],
        "produced_by": "benchmarks/systems/b4f_ideal.py",
        "composition": [
            "sources.plane_wave",
            "operators.complex_transmission",
            "solvers.chromatix.focal_plane_transform",
            "operators.complex_transmission",
            "solvers.chromatix.focal_plane_transform",
        ],
        "parameters": {
            "wavelength_m": WAVELENGTH_M,
            "medium_index": MEDIUM_INDEX,
            "shape": list(SHAPE),
            "sample_pitch_m": list(PITCH_M),
            "focal_length_1_m": f1,
            "focal_length_2_m": f2,
            "numerical_aperture": NUMERICAL_APERTURE,
            "stop_edge": "hard",
            "fourier_plane_pitch_m": list(artifacts["fourier"].sample_pitch_m),
            "image_plane_pitch_m": list(artifacts["image"].sample_pitch_m),
            "stop_radius_in_fourier_samples_yx": [
                numerical_aperture_radius_m(
                    NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
                )
                / pitch
                for pitch in artifacts["fourier"].sample_pitch_m
            ],
            "declared_validity_at_the_image_plane": sorted(artifacts["image"].validity),
        },
        "not_covered": [
            "the validity-envelope sweep over modulation frequency to the sampling limit "
            "(CHE-144's optional output; not run)",
            "a direct ifft2c(mask * fft2c(u)) NumPy cross-model (not run: the closed form "
            "above predicts the amplitude factor, the sign and the pitch, which a "
            "differential FFT check does not)",
            "real or aberrated lenses, ray-domain anything, partial coherence, "
            "polarization, noise and sensor models",
        ],
        "gates": gates,
        "negative_controls": controls,
    }


def main() -> int:
    """Run both configurations, write their records, and report."""
    failed = 0
    for configuration in CONFIGURATIONS:
        record = run(configuration)
        path = write_record(
            record, path=RECORDS / f"{BENCHMARK_ID}-{configuration['name']}.json"
        )
        print(f"\n=== {BENCHMARK_ID} / {configuration['name']} ===")
        for entry in record["gates"]:
            mark = "PASS" if entry["passed"] else "FAIL"
            print(f"  [{mark}] {entry['name']}  ({entry['oracle_kind']})")
            if not entry["passed"]:
                print(f"         measured {entry['measured']!r}")
                print(f"         expected {entry['expected']!r} +/- {entry['tolerance']!r}")
                failed += 1
        for entry in record["negative_controls"]:
            mark = "BROKE" if entry["broke_the_gate"] else "DID NOT BREAK"
            print(f"  [{mark}] control {entry['name']}")
            if not entry["broke_the_gate"]:
                print(f"         measured {entry['measured']!r} vs {entry['reference']!r}")
                failed += 1
        print(f"  record: {path.relative_to(Path(__file__).resolve().parents[2])}")

    if failed:
        print(f"\n{failed} gate(s) or control(s) did not hold.")
        return 1
    print("\nOK: every gate holds and every negative control breaks the gate it names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
