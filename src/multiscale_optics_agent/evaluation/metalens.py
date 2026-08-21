"""The 100x100 metalens configuration and its analytic oracle (CHE-70, Phase 22).

One frozen configuration produces **one** complex field, and both routes start
from it. That is the whole point of putting it here rather than in the benchmark
script: the direct-wave reference and the ray route cannot be tuned
independently, because neither of them owns the field.

The oracle is analytic, and why that is admissible
--------------------------------------------------
A stack of plane-parallel homogeneous layers is diagonal in the plane-wave basis.
The transverse wavevector is continuous across every interface, so a mode that
enters with spatial frequency ``f`` keeps it, and its phase advance through a
layer of thickness ``t`` and index ``n`` is exactly

    exp( i 2 pi t sqrt( (n/lambda)^2 - f^2 ) ).

There is no discretization and no approximation in that: for a field sampled on a
periodic grid, the finite sum over the grid's own plane waves **is** the exact
solution of the propagation problem the ray route is also solving. So this is an
analytic oracle in the strict sense the project requires -- not an independent
numerical implementation whose own errors would have to be bounded first.

Three routes to the same reference, for the single-layer case
------------------------------------------------------------
For a pure air gap the layered form must agree with two things that already
exist and were not written for this ticket:

* ``evaluation.asm_oracle.angular_spectrum_float64`` -- the repository's
  independent float64 angular-spectrum reference (CHE-40), written in the
  un-centred FFT convention rather than this module's centred one;
* Chromatix's ``asm_propagate`` -- a third-party package, M1-verified, with a
  genuinely different front end.

``tests/test_metalens_oracle.py`` holds all three against each other. The
agreement is what makes the layered form usable for the slab configuration, where
no prior repository reference exists.

The band limit is part of the configuration
-------------------------------------------
:func:`reference_field` takes the retained-mode mask and applies it. That is not
a convenience: the ray route cannot represent the modes the mask excludes (see
``couplers.streaming``), so a reference that kept them would be answering a
different question, and the difference would be charged to the coupler.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.core.optical_system import (
    ApertureSpec,
    FieldSpec,
    IdealMaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)
from multiscale_optics_agent.couplers.coherent_batch import (
    metres_to_micrometres,
    metres_to_millimetres,
)
from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    Frame,
    ReferencePlane,
)

__all__ = [
    "AIR_CONFIG",
    "CONFIGURATIONS",
    "SLAB_CONFIG",
    "Layer",
    "MetalensConfig",
    "PsfComparison",
    "centred_spectrum",
    "compare_psfs",
    "encircled_energy_radius_m",
    "layered_transfer",
    "metalens_field",
    "normalized_cross_correlation",
    "optical_system_spec",
    "reference_field",
    "retained_mode_mask",
]


@dataclass(frozen=True)
class Layer:
    """One homogeneous plane-parallel layer of the propagation stack."""

    thickness_m: float
    refractive_index: float
    name: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MetalensConfig:
    """Everything the benchmark's optics are, in SI, in one frozen object.

    The metalens is an ideal phase-only element: unit amplitude inside a circular
    aperture, zero outside, carrying the hyperbolic phase
    ``-k (sqrt(r^2 + f^2) - f)`` that focuses a normally-incident plane wave to a
    point at distance ``f`` **in air**. With a glass layer in the stack the focus
    moves, and the sensor plane is placed where the stack actually focuses rather
    than the metalens phase being retuned -- so the same phase profile serves both
    configurations and neither route sees a system built to flatter it.
    """

    name: str
    description: str
    grid: int
    sample_pitch_m: float
    wavelength_m: float
    aperture_radius_m: float
    design_focal_length_m: float
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        if self.grid <= 0 or self.grid % 2 != 0:
            raise ValueError(f"grid must be a positive even number, got {self.grid}")
        half_window = 0.5 * self.grid * self.sample_pitch_m
        if self.aperture_radius_m >= half_window:
            raise ValueError(
                f"aperture radius {self.aperture_radius_m} m does not fit inside the "
                f"{2 * half_window} m window"
            )

    # -- derived geometry --------------------------------------------------
    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid, self.grid)

    @property
    def pitch_pair(self) -> tuple[float, float]:
        return (self.sample_pitch_m, self.sample_pitch_m)

    @property
    def window_m(self) -> float:
        return self.grid * self.sample_pitch_m

    @property
    def sensor_distance_m(self) -> float:
        """Axial distance from the metalens plane to the sensor plane.

        For a stack containing glass the geometric focus shifts back by
        ``t (1 - 1/n)`` per layer -- the paraxial plane-parallel-plate shift. The
        sensor is put there. It is a *placement*, not a claim that the stack is
        aberration-free: a real plate adds spherical aberration, both routes see
        it, and the comparison is unaffected.
        """
        shift = sum(
            layer.thickness_m * (1.0 - 1.0 / layer.refractive_index)
            for layer in self.layers
            if layer.refractive_index != 1.0
        )
        return self.design_focal_length_m + shift

    @property
    def scaled_layers(self) -> tuple[Layer, ...]:
        """The layer stack with its air gap sized so the total reaches the sensor.

        Every layer except the last keeps its declared thickness; the last (air)
        layer absorbs the difference. That keeps "where is the sensor" a single
        derived quantity instead of two numbers that can disagree.
        """
        fixed = sum(layer.thickness_m for layer in self.layers[:-1])
        remainder = self.sensor_distance_m - fixed
        if remainder <= 0.0:
            raise ValueError(
                f"the fixed layers total {fixed} m but the sensor sits at "
                f"{self.sensor_distance_m} m; the trailing layer would be negative"
            )
        return (
            *self.layers[:-1],
            dataclasses.replace(self.layers[-1], thickness_m=remainder),
        )

    @property
    def numerical_aperture(self) -> float:
        radius, distance = self.aperture_radius_m, self.design_focal_length_m
        return radius / math.hypot(radius, distance)

    @property
    def airy_radius_m(self) -> float:
        return 0.61 * self.wavelength_m / self.numerical_aperture

    @property
    def grid_nyquist_direction_limit(self) -> float:
        """``lambda / (2 * pitch)`` -- the steepest ramp the output grid can hold."""
        return self.wavelength_m / (2.0 * self.sample_pitch_m)

    @property
    def max_lateral_travel_m(self) -> float:
        """How far the steepest *useful* ray moves laterally on the way to the sensor.

        Compared against the half window, this is whether the periodic grid's
        wraparound is physically relevant. Both routes solve the same periodic
        problem, so wraparound cannot break their agreement -- but it would make
        the PSF a wrapped artifact rather than a PSF, which is worth knowing.
        """
        tangent = self.aperture_radius_m / self.design_focal_length_m
        return self.sensor_distance_m * tangent

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "grid": self.grid,
            "grid_shape": list(self.grid_shape),
            "sample_pitch_m": self.sample_pitch_m,
            "window_m": self.window_m,
            "wavelength_m": self.wavelength_m,
            "aperture_radius_m": self.aperture_radius_m,
            "aperture_diameter_m": 2.0 * self.aperture_radius_m,
            "design_focal_length_m": self.design_focal_length_m,
            "sensor_distance_m": self.sensor_distance_m,
            "layers": [layer.as_dict() for layer in self.scaled_layers],
            "numerical_aperture": self.numerical_aperture,
            "airy_radius_m": self.airy_radius_m,
            "airy_radius_pixels": self.airy_radius_m / self.sample_pitch_m,
            "grid_nyquist_direction_limit": self.grid_nyquist_direction_limit,
            "max_lateral_travel_m": self.max_lateral_travel_m,
            "half_window_m": 0.5 * self.window_m,
            "phase_profile": "-k (sqrt(r^2 + f^2) - f), ideal hyperbolic metalens",
            "amplitude_profile": "1 inside the circular aperture, 0 outside",
            "illumination": "normally incident unit-amplitude plane wave",
            "coherence": "monochromatic, fully coherent",
            "units": "SI throughout; metres, radians",
        }


#: The primary configuration: the metalens phase and a pure air gap to the
#: sensor. Chosen so the oracle carries no assumption whatsoever beyond the
#: analytic angular spectrum -- no interface, no transmission coefficient, no
#: material model. 500 nm at a 250 nm (lambda/2) pitch is an ordinary
#: subwavelength metasurface sampling, and NA 0.196 puts the Airy radius at 6.2
#: pixels, so the PSF is resolved by the 100x100 sensor rather than sitting under
#: one sample.
AIR_CONFIG = MetalensConfig(
    name="METALENS-AIR-100",
    description=(
        "100x100 ideal hyperbolic metalens, 500 nm, 250 nm pitch, 20 um circular "
        "aperture, 50 um air gap to the sensor. NA 0.196."
    ),
    grid=100,
    sample_pitch_m=250e-9,
    wavelength_m=500e-9,
    aperture_radius_m=10e-6,
    design_focal_length_m=50e-6,
    layers=(Layer(thickness_m=50e-6, refractive_index=1.0, name="air"),),
)

#: The secondary configuration: the same metalens followed by a 10 um plate of
#: index 1.5, then air. Optiland must then actually refract at two interfaces and
#: accumulate an index-weighted path, so the OPL contract is exercised rather
#: than reduced to a geometric distance. The oracle gains exactly one declared
#: assumption -- ideal transmission at both interfaces, which is what the pinned
#: solver does with no coatings set -- and ``tests/test_metalens_oracle.py``
#: checks that assumption against the traced intensity instead of trusting it.
SLAB_CONFIG = MetalensConfig(
    name="METALENS-SLAB-100",
    description=(
        "100x100 ideal hyperbolic metalens, 500 nm, 250 nm pitch, 20 um circular "
        "aperture, then a 10 um n=1.5 plane-parallel plate and an air gap to the "
        "sensor at the plate-shifted focus."
    ),
    grid=100,
    sample_pitch_m=250e-9,
    wavelength_m=500e-9,
    aperture_radius_m=10e-6,
    design_focal_length_m=50e-6,
    layers=(
        Layer(thickness_m=10e-6, refractive_index=1.5, name="plate n=1.5"),
        Layer(thickness_m=40e-6, refractive_index=1.0, name="air"),
    ),
)

CONFIGURATIONS: dict[str, MetalensConfig] = {
    config.name: config for config in (AIR_CONFIG, SLAB_CONFIG)
}


def metalens_field(config: MetalensConfig) -> ComplexField:
    """The complex field just after the metalens. The single source for both routes.

    Built in complex128 on the host. Whatever precision a route computes in, it
    starts from this, and the downcast to complex64 is a bridge the caller plans
    and records rather than something that happens here.
    """
    coordinate = (np.arange(config.grid, dtype=np.float64) - config.grid // 2) * (
        config.sample_pitch_m
    )
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    radius = np.hypot(x, y)
    phase = -config.wavenumber * (
        np.sqrt(radius**2 + config.design_focal_length_m**2) - config.design_focal_length_m
    )
    u = np.where(radius <= config.aperture_radius_m, 1.0, 0.0) * np.exp(1j * phase)
    return ComplexField(
        u=u.astype(np.complex128),
        sample_pitch_m=config.pitch_pair,
        wavelength_m=config.wavelength_m,
        reference_plane=ReferencePlane(name="metalens_exit", z_m=0.0),
        frame=Frame(),
        normalization=(
            "u is complex amplitude; unit amplitude inside the aperture, so "
            "discrete power = (aperture pixel count) and carries no radiometric "
            "calibration"
        ),
        provenance={
            "source": "evaluation.metalens.metalens_field",
            "configuration": config.name,
            "aperture_pixel_count": int((radius <= config.aperture_radius_m).sum()),
            **config.as_dict(),
        },
    )


def centred_spectrum(u: np.ndarray) -> np.ndarray:
    """Modal amplitudes in the centred DFT convention ``wave_to_ray`` uses.

    ``fftshift(fft2(ifftshift(.)))/(ny*nx)``: zero frequency at index ``n // 2``,
    coefficients scaled so a plain sum of modes reproduces the field. Written out
    here rather than imported so the oracle does not depend on the coupler it is
    used to check.
    """
    array = np.asarray(u, dtype=np.complex128)
    ny, nx = array.shape
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array))) / (ny * nx)


def _direction_cosine_grids(config: MetalensConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(d_u, d_v)`` transverse direction cosines on the centred DFT grid."""
    axis = (
        np.fft.fftshift(np.fft.fftfreq(config.grid, d=config.sample_pitch_m))
        * config.wavelength_m
    )
    direction_v, direction_u = np.meshgrid(axis, axis, indexing="ij")
    return direction_u, direction_v


def retained_mode_mask(
    config: MetalensConfig, *, direction_cosine_floor: float
) -> np.ndarray:
    """The propagating-and-not-grazing mask, in the centred convention.

    Exactly the mask ``couplers.streaming.band_limit_spectrum`` builds: strictly
    propagating (``d_u^2 + d_v^2 < 1``) and axially steeper than the floor. Both
    are computed from the same two conditions so the oracle and the ray ensemble
    cannot end up with different mode sets, which is the failure this function
    exists to prevent.
    """
    direction_u, direction_v = _direction_cosine_grids(config)
    radial = direction_u**2 + direction_v**2
    axial = np.sqrt(np.clip(1.0 - radial, 0.0, None))
    return (radial < 1.0) & (axial >= direction_cosine_floor)


def layered_transfer(config: MetalensConfig) -> np.ndarray:
    """The exact plane-wave transfer function of the layer stack, centred convention.

    ``exp(i 2 pi t sqrt((n/lambda)^2 - f^2))`` per layer. A mode that is
    evanescent *in a layer* is zeroed there: over the distances here it would have
    decayed by many hundreds of e-foldings, and zeroing it is what the ray route
    does too (it has no direction to give such a mode).
    """
    direction_u, direction_v = _direction_cosine_grids(config)
    radial = direction_u**2 + direction_v**2
    transfer = np.ones(config.grid_shape, dtype=np.complex128)
    for layer in config.scaled_layers:
        index = layer.refractive_index
        argument = index**2 - radial
        propagating = argument > 0.0
        axial = np.sqrt(np.where(propagating, argument, 0.0))
        phase = config.wavenumber * layer.thickness_m * axial
        transfer = transfer * np.where(propagating, np.exp(1j * phase), 0.0)
    return transfer


def reference_field(
    config: MetalensConfig, *, direction_cosine_floor: float
) -> ComplexField:
    """The direct-wave field at the sensor plane. The gate's oracle.

    Analytic, float64, and restricted to exactly the modes the ray ensemble can
    carry. ``exp(i k n t)`` is **not** removed: the ray route accumulates the same
    absolute optical path, so keeping the carrier is what makes the two
    comparable in raw phase and not merely up to a piston.
    """
    field_in = metalens_field(config)
    spectrum = centred_spectrum(np.asarray(field_in.u))
    mask = retained_mode_mask(config, direction_cosine_floor=direction_cosine_floor)
    propagated = np.where(mask, spectrum * layered_transfer(config), 0.0)
    ny, nx = config.grid_shape
    u = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(propagated))) * (ny * nx)
    return ComplexField(
        u=u,
        sample_pitch_m=config.pitch_pair,
        wavelength_m=config.wavelength_m,
        reference_plane=ReferencePlane(name="sensor", z_m=config.sensor_distance_m),
        frame=Frame(),
        normalization="u is complex amplitude; same scale as the input field",
        provenance={
            "oracle": "analytic layered angular spectrum, float64",
            "oracle_kind": "analytic",
            "why_analytic": (
                "a plane-parallel homogeneous stack is diagonal in the plane-wave "
                "basis, so the finite sum over the grid's own modes is the exact "
                "solution rather than a discretization of one"
            ),
            "carrier": "retained; exp(i k n t) is not removed",
            "direction_cosine_floor": direction_cosine_floor,
            "retained_modes": int(mask.sum()),
            "layers": [layer.as_dict() for layer in config.scaled_layers],
            "configuration": config.name,
        },
    )


def optical_system_spec(config: MetalensConfig) -> OpticalSystemSpec:
    """The layer stack as a canonical Optiland prescription (CHE-56 / PB5).

    Expressed through ``OpticalSystemSpec`` and the generic builder rather than
    hand-assembled, so the surface types, materials and stop are validated before
    ``Optic`` is touched -- Optiland silently discards surface kwargs that do not
    belong to the geometry it selected, which is exactly what the schema exists to
    catch.

    Surface 0 is the metalens plane itself and carries the stop. It is a *plane in
    the material of the first layer*: the metalens is an ideal phase element whose
    phase is already in the field, so the ray side must not apply it again.
    """
    surfaces = []
    for position, layer in enumerate(config.scaled_layers):
        material = (
            IdealMaterialSpec(refractive_index=layer.refractive_index)
            if layer.refractive_index != 1.0
            else None
        )
        surfaces.append(
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=metres_to_millimetres(layer.thickness_m),
                is_stop=position == 0,
                **({"material": material} if material is not None else {}),
            )
        )
    return OpticalSystemSpec(
        name=config.name.replace("-", ""),
        description=config.description,
        object_distance_mm=None,
        surfaces=tuple(surfaces),
        aperture=ApertureSpec(
            value_mm=metres_to_millimetres(2.0 * config.aperture_radius_m)
        ),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(
            WavelengthSpec(
                value_um=metres_to_micrometres(config.wavelength_m), is_primary=True
            ),
        ),
    )


# --- metrics -----------------------------------------------------------------


def normalized_cross_correlation(test: np.ndarray, reference: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two same-shape real images.

    The mean is removed. Without that, two positive images that share a large
    pedestal correlate at 0.99 for reasons that have nothing to do with agreeing,
    and a PSF on a wide window is mostly pedestal. This is the stricter of the two
    conventions and the one the gate is stated in.
    """
    a = np.asarray(test, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    if denominator == 0.0:
        raise ValueError("a constant image has no normalized cross-correlation")
    return float(a @ b) / denominator


def _centroid(intensity: np.ndarray, pitch: float) -> tuple[float, float]:
    ny, nx = intensity.shape
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * pitch
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * pitch
    total = float(intensity.sum())
    if total <= 0.0:
        return (float("nan"), float("nan"))
    return (
        float((intensity.sum(axis=1) @ y) / total),
        float((intensity.sum(axis=0) @ x) / total),
    )


def _radial_grid(shape: tuple[int, int], pitch: float) -> np.ndarray:
    ny, nx = shape
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * pitch
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * pitch
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.hypot(xx, yy)


def encircled_energy_radius_m(
    intensity: np.ndarray, *, pitch: float, fraction: float = 0.5
) -> float:
    """Radius about the grid origin containing ``fraction`` of the window's energy.

    About the *origin*, not the centroid: the reference PSF is centred by
    construction, and using each image's own centroid would hide a lateral shift
    -- which is one of the errors this metric is supposed to be able to see.
    """
    radius = _radial_grid(intensity.shape, pitch).ravel()
    values = np.asarray(intensity, dtype=np.float64).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(values[order])
    total = cumulative[-1]
    if total <= 0.0:
        return float("nan")
    index = int(np.searchsorted(cumulative, fraction * total))
    return float(radius[order][min(index, radius.size - 1)])


def _fwhm_m(intensity: np.ndarray, *, pitch: float) -> float:
    """Full width at half maximum of the row through the peak, linearly interpolated."""
    peak_index = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    row = np.asarray(intensity[peak_index[0]], dtype=np.float64)
    peak = float(row.max())
    if peak <= 0.0:
        return float("nan")
    half, column = 0.5 * peak, int(np.argmax(row))

    def crossing(indices: range) -> float | None:
        previous = column
        for index in indices:
            if row[index] < half:
                span = row[previous] - row[index]
                if span == 0.0:
                    return float(index)
                return previous + (row[previous] - half) / span * (index - previous)
            previous = index
        return None

    left = crossing(range(column - 1, -1, -1))
    right = crossing(range(column + 1, row.size))
    if left is None or right is None:
        return float("nan")
    return abs(right - left) * pitch


@dataclass(frozen=True)
class PsfComparison:
    """Every number Phase 25/30 asks for, from one test/reference PSF pair.

    ``normalized_cross_correlation`` is the gate. The rest are here because NCC
    is blind to a global scale, so a run that reports only NCC cannot tell a
    converged estimator from one whose energy is twice the reference's -- which is
    exactly what a Monte Carlo field estimator does at small ray counts, and it is
    measured rather than assumed.
    """

    ncc: float
    normalized_mse: float
    reference_power: float
    test_power: float
    relative_power_error: float
    reference_peak: float
    test_peak: float
    relative_peak_error: float
    centroid_error_m: float
    reference_fwhm_m: float
    test_fwhm_m: float
    relative_fwhm_error: float
    reference_ee50_radius_m: float
    test_ee50_radius_m: float
    relative_ee50_error: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_psfs(test: np.ndarray, reference: np.ndarray, *, pitch: float) -> PsfComparison:
    """Compare two raw intensity images. Raw: normalization is reported, not applied."""
    a = np.asarray(test, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    reference_power = float(b.sum())
    test_power = float(a.sum())
    reference_peak = float(b.max())
    test_peak = float(a.max())
    # Normalized MSE is computed on peak-normalized images, so it measures shape
    # disagreement and the scale error is reported separately rather than folded in.
    scaled_a = a / test_peak if test_peak > 0.0 else a
    scaled_b = b / reference_peak if reference_peak > 0.0 else b
    denominator = float((scaled_b**2).sum())
    centroid_a = _centroid(a, pitch)
    centroid_b = _centroid(b, pitch)
    fwhm_a = _fwhm_m(a, pitch=pitch)
    fwhm_b = _fwhm_m(b, pitch=pitch)
    ee_a = encircled_energy_radius_m(a, pitch=pitch)
    ee_b = encircled_energy_radius_m(b, pitch=pitch)
    return PsfComparison(
        ncc=normalized_cross_correlation(a, b),
        normalized_mse=(
            float(((scaled_a - scaled_b) ** 2).sum() / denominator)
            if denominator > 0.0
            else float("nan")
        ),
        reference_power=reference_power,
        test_power=test_power,
        relative_power_error=(
            (test_power - reference_power) / reference_power
            if reference_power > 0.0
            else float("nan")
        ),
        reference_peak=reference_peak,
        test_peak=test_peak,
        relative_peak_error=(
            (test_peak - reference_peak) / reference_peak
            if reference_peak > 0.0
            else float("nan")
        ),
        centroid_error_m=float(
            math.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1])
        ),
        reference_fwhm_m=fwhm_b,
        test_fwhm_m=fwhm_a,
        relative_fwhm_error=(
            (fwhm_a - fwhm_b) / fwhm_b if fwhm_b and math.isfinite(fwhm_b) else float("nan")
        ),
        reference_ee50_radius_m=ee_b,
        test_ee50_radius_m=ee_a,
        relative_ee50_error=(
            (ee_a - ee_b) / ee_b if ee_b and math.isfinite(ee_b) else float("nan")
        ),
    )
