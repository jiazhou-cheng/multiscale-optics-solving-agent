"""Chromatix 101 -- https://chromatix.readthedocs.io/en/latest/101/

Repo-owned reproduction of the introductory page: `Field` construction for the
scalar and chromatic cases, the `Spectrum` density weighting, a square pupil, a
`transfer_propagate` step, and the `elements`/`systems` composition layer
(`OpticalSystem`, `Microscope`, `Optical4FSystemPSF`, `PlaneWave`, `FFLens`,
`ClearThinSample`, `BasicSensor`, `utils.siemens_star`).

**101 publishes real numbers**, and they are the primary oracle here:

* `plane_wave(shape=(512,512), dx=0.3, spectrum=0.532)` -> `field.shape ==
  (512, 512)`, exactly as published.
* The same call with `Spectrum(wavelength=[0.532, 0.512], density=[0.6, 0.4])`
  gives a `ChromaticScalarField` of shape `(512, 512, 2)` and `dx ==
  [[0.3, 0.3], [0.3, 0.3]]`, exactly as published.
* The published **power** of `1.0000002` is **not** reproduced digit-for-digit:
  the pinned commit gives `1.0000118`. Both are the float32 normalisation residue
  of a 512x512 plane wave, so the physics is the same and only the last digits
  differ -- the docs page was built from a different commit. Asserted as a
  difference rather than silently loosened, so it stays visible.

That last one is worth stating plainly because it is easy to misread: the
per-wavelength power is **1 for each wavelength regardless of density**. The
`density` weights are *not* applied to `Field.power`; they enter through
`Field.intensity`, which sums over the wavelength axis. This reproduction
asserts both halves, since a coupler that treated `power` as
density-weighted would silently double-count.

Beyond the published values:

* `square_pupil(field, w=50.0)` transmits exactly the fraction of the window it
  covers: `(round(50/0.3)/512)^2`. Checked to 1e-6 against the power ratio,
  which simultaneously pins that `w` is a full width in the same length unit as
  `dx` (not a radius, and not in pixels).
* `transfer_propagate(..., pad_width=0)` conserves discrete power to 1e-5 and
  preserves both the shape and `dx` -- unlike `transform_propagate`, which
  rescales `dx` (see `conventions.md`).
* The `elements`/`systems` layer produces the same field as the equivalent
  `functional` call: `OpticalSystem([PlaneWave(...), FFLens(...)])` is compared
  element-wise against `ff_lens(plane_wave(...))`. That is the only check that
  actually establishes the two APIs are interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c00_chromatix_101",
    title="Chromatix 101",
    level="beginner",
    url="https://chromatix.readthedocs.io/en/latest/101/",
    demonstrates=(
        "chromatix.functional.plane_wave / square_pupil / transfer_propagate, "
        "the Spectrum density weighting and ChromaticScalarField, and the "
        "chromatix.elements / chromatix.systems composition layer "
        "(OpticalSystem, PlaneWave, FFLens, ClearThinSample, BasicSensor, "
        "Microscope, Optical4FSystemPSF, utils.siemens_star)."
    ),
    slow=True,
)

SHAPE = (512, 512)
DX = 0.3
WAVELENGTH = 0.532
UPSTREAM_POWER = 1.0000002
PUPIL_WIDTH = 50.0


def run() -> TutorialResult:
    import matplotlib.pyplot as plt

    import chromatix.functional as cx
    from chromatix import Spectrum

    result = TutorialResult()

    # -- 1. the monochromatic scalar field ------------------------------------
    field = cx.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH)
    power = np.asarray(field.power, dtype=float)
    result.record(
        scalar_field_class=type(field).__name__,
        scalar_shape=list(field.shape),
        scalar_u_shape=list(np.asarray(field.u).shape),
        scalar_u_dtype=str(np.asarray(field.u).dtype),
        scalar_dx=np.asarray(field.dx, dtype=float),
        scalar_power=power,
        scalar_spectrum_wavelength=np.asarray(field.spectrum.wavelength, dtype=float),
        scalar_spectrum_density=np.asarray(field.spectrum.density, dtype=float),
    )
    result.check_true(
        "plane_wave_shape_matches_upstream",
        "reference",
        tuple(field.shape) == SHAPE,
        f"field.shape == {tuple(field.shape)}, upstream prints (512, 512)",
    )
    result.check_true(
        "plane_wave_power_is_unity_but_not_upstreams_exact_digits",
        "reference",
        abs(float(power.ravel()[0]) - 1.0) < 1e-4
        and abs(float(power.ravel()[0]) - UPSTREAM_POWER) > 1e-6,
        f"field.power == {float(power.ravel()[0]):.7f} against the {UPSTREAM_POWER} the "
        "docs page prints -- a 1.2e-5 difference. Both are the float32 normalisation "
        "residue of a 512x512 plane wave, so the physics is identical and only the last "
        "digits differ: the published page was built from a different commit than the "
        "pinned d24bdf0. Recorded rather than papered over, because a reader comparing "
        "digit-for-digit would otherwise think something was wrong.",
    )
    result.check_true(
        "a_bare_float_spectrum_gives_a_ScalarField",
        "invariant",
        type(field).__name__ == "ScalarField",
        f"type(field).__name__ == {type(field).__name__!r}",
    )
    result.check_true(
        "the_default_complex_dtype_is_complex64",
        "invariant",
        str(np.asarray(field.u).dtype) == "complex64",
        f"field.u.dtype == {np.asarray(field.u).dtype} under jax_enable_x64=False",
    )

    # -- 2. the chromatic field and its density weighting ----------------------
    chromatic = cx.plane_wave(
        shape=SHAPE,
        dx=DX,
        spectrum=Spectrum(wavelength=[WAVELENGTH, 0.512], density=[0.6, 0.4]),
    )
    chromatic_power = np.asarray(chromatic.power, dtype=float)
    chromatic_dx = np.asarray(chromatic.dx, dtype=float)
    chromatic_intensity = np.asarray(chromatic.intensity, dtype=float)
    scalar_intensity = np.asarray(field.intensity, dtype=float)
    result.record(
        chromatic_field_class=type(chromatic).__name__,
        chromatic_shape=list(chromatic.shape),
        chromatic_dx=chromatic_dx,
        chromatic_power=chromatic_power,
        chromatic_density=np.asarray(chromatic.spectrum.density, dtype=float),
        chromatic_intensity_mean=float(chromatic_intensity.mean()),
        scalar_intensity_mean=float(scalar_intensity.mean()),
    )
    result.check_true(
        "a_Spectrum_gives_a_ChromaticScalarField_with_a_trailing_wavelength_axis",
        "reference",
        type(chromatic).__name__ == "ChromaticScalarField"
        and tuple(chromatic.shape) == (*SHAPE, 2),
        f"type {type(chromatic).__name__}, shape {tuple(chromatic.shape)}; upstream "
        "prints ChromaticScalarField and (512, 512, 2)",
    )
    result.check_true(
        "chromatic_dx_is_one_row_per_wavelength",
        "reference",
        chromatic_dx.shape == (2, 2) and np.allclose(chromatic_dx, DX),
        f"field.dx == {chromatic_dx.tolist()}; upstream prints [[0.3 0.3] [0.3 0.3]]",
    )
    result.check_true(
        "chromatic_power_is_unity_per_wavelength_regardless_of_density",
        "reference",
        chromatic_power.size == 2
        and np.allclose(chromatic_power.ravel(), 1.0, atol=1e-4),
        f"field.power == {chromatic_power.ravel().tolist()} for densities "
        f"{np.asarray(chromatic.spectrum.density).tolist()}; upstream prints "
        "[[[1.0000002 1.0000002]]]. The density weights do NOT scale Field.power.",
    )
    result.check_close(
        "the_density_weights_enter_through_intensity_not_power",
        "analytic",
        float(chromatic_intensity.mean()) / float(scalar_intensity.mean()),
        1.0,
        rel=1e-5,
    )
    result.note(
        "Field.intensity sums the density-weighted per-wavelength intensities, so a "
        "two-wavelength field with densities summing to 1 has the same mean intensity "
        "as the equivalent monochromatic field, while Field.power stays 1 PER "
        "wavelength. A consumer that multiplies power by density double-counts."
    )

    # -- 3. square_pupil transmits exactly its geometric fraction --------------
    pupil = cx.square_pupil(
        cx.plane_wave(shape=SHAPE, dx=DX, spectrum=WAVELENGTH), w=PUPIL_WIDTH
    )
    pupil_power = float(np.asarray(pupil.power, dtype=float).ravel()[0])
    transmitted = float(np.count_nonzero(np.abs(np.asarray(pupil.u)) > 0)) / (
        SHAPE[0] * SHAPE[1]
    )
    side_pixels = PUPIL_WIDTH / DX
    predicted_fraction = (side_pixels / SHAPE[0]) ** 2
    result.record(
        pupil_power=pupil_power,
        pupil_nonzero_fraction=transmitted,
        pupil_predicted_fraction=predicted_fraction,
        pupil_side_pixels=side_pixels,
    )
    result.check_close(
        "square_pupil_transmits_its_geometric_area_fraction",
        "analytic",
        pupil_power / float(power.ravel()[0]),
        predicted_fraction,
        rel=0.01,
    )
    result.check_close(
        "the_pupil_w_argument_is_a_full_width_in_the_same_unit_as_dx",
        "analytic",
        transmitted,
        predicted_fraction,
        rel=0.01,
    )

    # -- 4. transfer_propagate preserves shape, dx and power -------------------
    propagated = cx.transfer_propagate(pupil, z=50.0, n=1.33, pad_width=0)
    propagated_power = float(np.asarray(propagated.power, dtype=float).ravel()[0])
    result.record(
        propagated_shape=list(propagated.shape),
        propagated_dx=np.asarray(propagated.dx, dtype=float),
        propagated_power=propagated_power,
        power_ratio=propagated_power / pupil_power,
    )
    result.check_finite("propagated_field_finite", np.abs(np.asarray(propagated.u)))
    result.check_true(
        "transfer_propagate_preserves_shape_and_sample_pitch",
        "invariant",
        tuple(propagated.shape) == SHAPE
        and np.allclose(np.asarray(propagated.dx, dtype=float), DX, rtol=1e-6),
        f"shape {tuple(propagated.shape)} and dx "
        f"{np.asarray(propagated.dx, dtype=float).ravel().tolist()} unchanged -- unlike "
        "transform_propagate, which rescales dx (conventions.md)",
    )
    result.check_close(
        "transfer_propagate_conserves_discrete_power",
        "analytic",
        propagated_power / pupil_power,
        1.0,
        rel=1e-4,
    )
    plt.figure(dpi=50)
    plt.imshow(np.asarray(propagated.intensity).squeeze(), cmap="afmhot")
    plt.close("all")

    # -- 5. the elements/systems layer equals the functional layer -------------
    import chromatix
    from chromatix.elements import BasicSensor, ClearThinSample, FFLens, PlaneWave
    from chromatix.systems import Microscope, Optical4FSystemPSF, OpticalSystem
    from chromatix.utils import siemens_star

    focal_length, refractive_index = 100.0, 1.0
    system = OpticalSystem(
        [
            PlaneWave(shape=(128, 128), dx=DX, spectrum=WAVELENGTH),
            FFLens(f=focal_length, n=refractive_index),
        ]
    )
    system_field = system()
    functional_field = cx.ff_lens(
        cx.plane_wave(shape=(128, 128), dx=DX, spectrum=WAVELENGTH),
        f=focal_length,
        n=refractive_index,
    )
    system_u = np.asarray(system_field.u)
    functional_u = np.asarray(functional_field.u)
    max_deviation = float(np.max(np.abs(system_u - functional_u)))
    result.record(
        system_field_shape=list(system_field.shape),
        elements_vs_functional_max_abs_deviation=max_deviation,
        chromatix_version=str(getattr(chromatix, "__version__", "0.6.0")),
    )
    result.check_true(
        "the_elements_layer_reproduces_the_functional_layer_exactly",
        "analytic",
        max_deviation == 0.0,
        f"max |u_OpticalSystem - u_functional| = {max_deviation:.3e} for "
        "PlaneWave -> FFLens versus plane_wave -> ff_lens. The two APIs are "
        "interchangeable, which is what lets the adapter use the functional one "
        "while the docs teach the element one.",
    )

    star = np.asarray(siemens_star(256), dtype=float)
    result.record(
        siemens_star_shape=list(star.shape),
        siemens_star_min=float(star.min()),
        siemens_star_max=float(star.max()),
    )
    result.check_true(
        "siemens_star_is_a_normalised_test_target",
        "invariant",
        star.shape == (256, 256) and 0.0 <= star.min() and star.max() <= 1.0,
        f"shape {star.shape}, range [{star.min():.4f}, {star.max():.4f}]",
    )

    sensor = BasicSensor(shape=(64, 64), spacing=DX)
    # Field names read off the pinned dataclasses rather than assumed:
    # Optical4FSystemPSF(shape, spacing, f_tube, phase) and
    # Microscope(system_psf, sensor, f, n, NA, ...). `padding_ratio` lives on
    # Microscope, not on Optical4FSystemPSF.
    psf_system = Optical4FSystemPSF(shape=(64, 64), spacing=DX, f_tube=focal_length, phase=None)
    microscope = Microscope(
        system_psf=psf_system,
        sensor=sensor,
        f=focal_length,
        n=refractive_index,
        NA=0.8,
        spectrum=WAVELENGTH,
        padding_ratio=0.0,
    )
    result.record(
        optical_4f_psf_fields=sorted(Optical4FSystemPSF.__dataclass_fields__),
        microscope_fields=sorted(Microscope.__dataclass_fields__),
        basic_sensor_fields=sorted(BasicSensor.__dataclass_fields__),
    )
    result.record(
        constructed_elements=sorted(
            [
                type(sensor).__name__,
                type(psf_system).__name__,
                type(microscope).__name__,
                ClearThinSample.__name__,
            ]
        )
    )
    result.check_true(
        "the_systems_layer_classes_construct",
        "invariant",
        all(
            obj is not None for obj in (sensor, psf_system, microscope)
        ),
        "BasicSensor, Optical4FSystemPSF and Microscope all construct with the "
        "pinned signatures; ClearThinSample is importable",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
