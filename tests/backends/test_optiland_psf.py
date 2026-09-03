"""`backends.optiland.psf`: the delegation, its three methods, its units, its refusals.

CHE-236 (R16.1). This path is a **delegation**, so almost nothing here is about the
arithmetic: reimplementing a reference sphere or a Huygens sum in order to check
one would be reimplementing the pipeline the module exists not to reimplement, and
`AGENTS.md` is explicit that a wrapper agreeing with the code it wraps is not a
correctness gate. The independent correctness evidence is
`tests/physics/test_native_psf_airy.py`, which holds the FFT method to the Airy
oracle -- code shared with neither this path nor the solver.

What is tested here is everything the delegation itself is responsible for:

* the pinned numbers, so a version bump or a wiring change is visible;
* **the units**, because the pinned classes disagree with *each other* about
  whether their own `pixel_pitch` is micrometres or millimetres, and a factor of
  1000 in a PSF pitch is entirely plausible-looking;
* that all three methods return one record with one set of field semantics;
* that FFT and MMDFT agree when configured to the same sampling, and that Huygens
  agrees with FFT at adequate sampling -- **reported as agreement of three paths
  through the same pupil, not as independent validation**, because that is all it
  is;
* the refusals -- an unknown method, a finite-conjugate source, a `RayBundle`
  handed in as a source, an unusable ray count, a sampling argument the selected
  method cannot use, a misspelled execution declaration;
* the boundary: no `Optic`, `Wavefront` or `WavefrontData` in the signature or on
  the record, and no public wavefront or pupil-field type in the diff at all.

Two tests are **characterization and say so**: the coarse-sampling disagreement
between Huygens and FFT (which is grid *centring*, measured, and not a difference
of physics), and what `remove_tilt` does on axis.

Cost. Huygens scales as `image_size**2 * num_rays**2` and the first call pays
numba's compilation (measured: 1.8 s once, then 0.05 s at 32 rays / 33 samples).
Every Huygens assertion below is at or below 33x33, and nothing here is a GPU
question.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.systems import (
    finite_conjugate_singlet,
    finite_conjugate_source,
    singlet_ref,
    singlet_source,
)

from backends.optiland import psf, trace
from backends.optiland.analysis import (
    NATIVE_PSF_ANALYSES,
    NATIVE_PSF_METHOD_DEFINITIONS,
    NATIVE_PSF_NORMALIZATION,
    NativePsfAnalysis,
)

EXECUTION: dict[str, str] = {"device": "cpu", "precision": "fp64"}

#: The frozen on-axis FFT analysis of the R05 reference singlet, `optiland 0.6.0`,
#: fp64 on the host. `num_rays=32` with no `grid_size` is the pinned class's own
#: OpticStudio emulation: 32 pupil samples, a 64-wide padded grid.
FFT_NUM_RAYS = 32
FFT_SHAPE = (64, 64)
FFT_PEAK = 99.90636958223962
FFT_PITCH_M = 2.5756498035505344e-06

#: `_get_working_FNO()` at this field and wavelength, against the fixture's nominal
#: F/9.7, and the radius of the chief-ray reference sphere (`WavefrontData.radius`,
#: 4.837461300309598 mm).
WORKING_F_NUMBER = 9.668128294852444
REFERENCE_SPHERE_RADIUS_M = 4.837461300309598e-03

#: The same singlet at 3 and 5 degrees, `num_rays=64` / `grid_size=256`. Off axis
#: this singlet is no longer diffraction limited and the peak says so.
OFF_AXIS_PEAKS = {0.0: 99.90147, 3.0: 98.66411, 5.0: 93.03898}


def _fft(**overrides: Any) -> NativePsfAnalysis:
    call: dict[str, Any] = {
        "method": "fft",
        "num_rays": FFT_NUM_RAYS,
        "execution": EXECUTION,
    }
    call.update(overrides)
    return psf(singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0)), **call)


def test_the_frozen_on_axis_fft_analysis_reproduces() -> None:
    """The pinned regression: shape, pitch, peak, Strehl, and the image-space geometry.

    Under delegation this is what "FFT matches Optiland" can honestly be -- there
    is no second implementation here to compare against, so the statement is that
    the wiring, the sampling and the unit conversion have not moved.
    """
    result = _fft()

    assert result.image_shape == FFT_SHAPE
    assert np.shape(result.intensity) == FFT_SHAPE
    assert result.peak_intensity == pytest.approx(FFT_PEAK, rel=1e-12)
    assert result.pixel_pitch_m == pytest.approx(FFT_PITCH_M, rel=1e-12)
    assert result.working_f_number == pytest.approx(WORKING_F_NUMBER, rel=1e-12)
    assert result.reference_sphere_radius_m == pytest.approx(
        REFERENCE_SPHERE_RADIUS_M, rel=1e-12
    )
    # The pupil sampling the solver actually used, which at 32 it does not reduce.
    assert result.num_rays == FFT_NUM_RAYS
    assert (result.fields_analyzed, result.wavelengths_analyzed) == (1, 1)
    # On axis the pattern is centred on the grid, which is `Frame`'s origin rule and
    # is what `coordinates()` assumes.
    assert result.peak_index == (FFT_SHAPE[0] // 2, FFT_SHAPE[1] // 2)


def test_the_fft_path_reports_the_pupil_sampling_it_used_and_not_the_one_requested() -> None:
    """The pinned class reduces `num_rays` itself, and the record says the truth.

    Measured on 0.6.0: with no `grid_size`, `calculate_grid_size` maps
    `num_rays -> floor(32 * 2**((log2(num_rays) - 5) / 2))` and pads to `2 *
    num_rays`, so 128 becomes 64 pupil samples on a 256 grid. Reporting the request
    would name a pupil sampling that never happened -- and pupil sampling is what
    the analytic gate's residual is controlled by
    (`tests/physics/test_native_psf_airy.py`), so it is the number that matters.
    """
    reduced = _fft(num_rays=128)
    assert reduced.num_rays == 64
    assert reduced.image_shape == (256, 256)

    # With an explicit `grid_size` nothing is reduced: the request is what ran.
    explicit = _fft(num_rays=64, grid_size=256)
    assert explicit.num_rays == 64
    assert explicit.image_shape == (256, 256)
    # ...and the two agree, which is what says the reduction is only a default and
    # not a second sampling rule.
    assert explicit.pixel_pitch_m == pytest.approx(reduced.pixel_pitch_m, rel=1e-12)


def test_the_normalization_is_strehl_percent_and_the_record_declares_it() -> None:
    """Criterion 4: `peak_intensity` ~ 100, and `strehl_ratio == peak / 100`.

    100.0 is the peak an unaberrated pupil of the *same aperture* would reach under
    the same propagation, so the number is a Strehl ratio times 100 and carries no
    radiometric scale at all. This is **not** `measurements.PsfNormalization`, and
    the record carries the declaration rather than a tag a consumer would have to
    look up.

    `strehl_ratio` is the solver's own call, and the two are not the same call on
    all three methods: `BasePSF.strehl_ratio` reads the centre sample and
    `MMDFTPSF` overrides it with the maximum. On a centred pattern they agree,
    which is what is asserted; the record carries both rather than choosing.
    """
    result = _fft()

    assert result.peak_intensity == pytest.approx(100.0, abs=0.1)
    assert result.strehl_ratio == pytest.approx(result.peak_intensity / 100.0, rel=1e-12)
    assert result.normalization == "strehl_percent"
    assert result.normalization_declaration == NATIVE_PSF_NORMALIZATION
    assert "peaks at 100.0" in result.normalization_declaration
    # And it is explicitly not this project's vocabulary for the other PSF path.
    assert result.normalization not in {"raw", "peak", "energy"}


def test_the_matrix_dft_honours_the_requested_image_size_and_pitch() -> None:
    """Criterion 5, in metres. The returned shape and pitch are the request.

    This is the argument that would be silently useless if the micrometre
    conversion were wrong in either direction: a pitch out by 1000 still returns a
    perfectly plausible-looking map.
    """
    requested_pitch_m = 1.5e-06
    result = psf(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        method="mmdft",
        num_rays=32,
        execution=EXECUTION,
        image_size=33,
        pixel_pitch_m=requested_pitch_m,
    )

    assert result.image_shape == (33, 33)
    assert result.pixel_pitch_m == pytest.approx(requested_pitch_m, rel=1e-12)
    # The extent follows, which is the number a consumer actually uses.
    y, x = result.coordinates()
    assert x[-1] - x[0] == pytest.approx(32 * requested_pitch_m, rel=1e-12)
    assert y[16] == pytest.approx(0.0, abs=1e-18)


def test_the_two_fourier_paths_agree_when_configured_to_the_same_sampling() -> None:
    """Criterion 6. **Two paths through the same pupil, not independent validation.**

    `MMDFTPSF._generate_pupil` is a copy of `ScalarFFTPSF._generate_pupils` (the
    pinned source says so) and both propagate the same complex pupil; setting the
    matrix DFT's output sampling to the FFT path's own `dx` and its size to the
    FFT grid makes them the same transform evaluated two ways. Agreement to 6e-13
    of the peak is therefore expected, and what it would catch is a real defect: a
    unit error on either side, or the pitch being reported for a grid the numbers
    did not come from.
    """
    fft = _fft()
    mmdft = psf(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        method="mmdft",
        num_rays=FFT_NUM_RAYS,
        execution=EXECUTION,
        image_size=FFT_SHAPE[0],
        pixel_pitch_m=fft.pixel_pitch_m,
    )

    assert mmdft.image_shape == fft.image_shape
    assert mmdft.pixel_pitch_m == pytest.approx(fft.pixel_pitch_m, rel=1e-12)
    assert mmdft.peak_intensity == pytest.approx(fft.peak_intensity, rel=1e-12)
    largest = float(np.max(np.abs(np.asarray(mmdft.intensity) - np.asarray(fft.intensity))))
    assert largest < 1e-12, largest


def test_huygens_agrees_with_fft_at_adequate_sampling() -> None:
    """Criterion 7, on the same low-NA fixture, at a sampling this test reports.

    The two are *not* the same computation -- Huygens sums spherical wavelets from
    the physical reference-sphere intersections to the actual image-surface
    geometry with the 1/R and obliquity factors, and the FFT path is a Fraunhofer
    transform of a pupil -- so this is a real comparison of two propagations of one
    pupil and the residual below is the difference between them.

    **The tolerance is not widened to pass and it is not a convergence claim.**
    Measured across pupil samplings 32, 64 and 128 and image pitches 2.62 and 1.31
    um, the peak-normalized L-infinity residual is 2.533e-3 to 2.559e-3 -- it does
    not move with either sampling, which is what says it is the propagation
    difference and not a grid artifact. 5e-3 is the stated tolerance and the
    measurement is 2.56e-3 at 32 pupil samples and 5.0 samples per Airy radius.

    The grid must be **odd** for this comparison to be about propagation at all;
    see the characterization test below for what an even one does.
    """
    fft = _fft(grid_size=128)
    huygens = psf(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        method="huygens",
        num_rays=FFT_NUM_RAYS,
        execution=EXECUTION,
        image_size=33,
        pixel_pitch_m=fft.pixel_pitch_m,
    )

    # The same physical grid, which is the premise of the comparison.
    assert huygens.pixel_pitch_m == pytest.approx(fft.pixel_pitch_m, rel=1e-12)
    # Measured: 99.90638500 against 99.90636958, a relative 1.5e-7. The peak is the
    # one number the two propagations are expected to agree on to the normalizer's
    # own precision, since both divide by the peak of the same ideal pupil.
    assert huygens.peak_intensity == pytest.approx(fft.peak_intensity, rel=1e-6)

    centre = fft.image_shape[0] // 2
    half = huygens.image_shape[0] // 2
    cropped = np.asarray(fft.intensity)[
        centre - half : centre + half + 1, centre - half : centre + half + 1
    ]
    measured = np.asarray(huygens.intensity) / huygens.peak_intensity
    residual = float(np.max(np.abs(measured - cropped / cropped.max())))
    assert residual == pytest.approx(2.56e-3, rel=0.05), residual
    assert residual < 5.0e-3


def test_characterization_an_even_huygens_grid_has_no_sample_at_the_peak() -> None:
    """**Characterization.** The 82.70-vs-99.91 disagreement, and its measured cause.

    R16.1 reports Huygens peaking at 82.696 where the FFT path peaks at 99.906 on
    this fixture, and names it a coarse-sampling discrepancy to be pinned so it is
    visible rather than surprising. Measured, it is **grid centring** and not the
    propagation: `ScalarHuygensPSF` lays its samples on
    `linspace(-extent, +extent, image_size)`, which for an even `image_size` has no
    sample at the centre of the pattern, so the reported peak is the largest sample
    *near* the peak. At `image_size=33` -- one more sample, the same pitch to 3 % --
    the same call peaks at 99.9064 and matches the FFT path to six digits.

    Nothing is corrected here. The record reports what the solver computed, on the
    grid the solver used, and this test is what makes the trap legible.

    **It is also the limit of `coordinates()`'s origin claim**, which is why the
    method's docstring states it: on an even Huygens grid, index `n // 2` is half a
    sample spacing off the centre of the pattern in each axis, so the coordinate
    the record hands back as `0.0` is not quite where the pattern is centred. For
    `fft`, `mmdft`, and odd-sized `huygens` it is.
    """
    setup, source = singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0))
    even = psf(
        setup, source, method="huygens", num_rays=32, execution=EXECUTION, image_size=32
    )
    odd = psf(
        setup, source, method="huygens", num_rays=32, execution=EXECUTION, image_size=33
    )

    assert even.peak_intensity == pytest.approx(82.6958, rel=1e-4)
    assert odd.peak_intensity == pytest.approx(99.9064, rel=1e-5)
    # Both put their largest sample at the grid centre; on the even grid that
    # sample is half a pitch off the pattern's centre in each axis, and that is the
    # whole of the 17 % difference.
    assert even.peak_index == (16, 16)
    assert odd.peak_index == (16, 16)
    # The auto-chosen pitches differ by only the sample count, so the two runs are
    # not at meaningfully different resolutions.
    assert odd.pixel_pitch_m == pytest.approx(even.pixel_pitch_m, rel=0.05)

    # And the origin sample is where the difference shows: on the odd grid it is the
    # pattern's peak, on the even grid it under-reads it by 17 %. Both records
    # report `coordinates()[n // 2] == 0`, and only one of them means it.
    for result in (even, odd):
        ny, nx = result.image_shape
        origin_sample = float(np.asarray(result.intensity)[ny // 2, nx // 2])
        assert origin_sample == pytest.approx(result.peak_intensity, rel=1e-12)
        assert float(result.coordinates()[0][ny // 2]) == 0.0
    assert even.peak_intensity / odd.peak_intensity == pytest.approx(0.8277, rel=1e-3)


def test_the_mirrored_huygens_image_size_default_is_still_the_pinned_one() -> None:
    """The one mirrored upstream default this module carries, read back off the source.

    `_native_pixel_pitch` has to know the output grid to undo the Huygens
    `linspace` off-by-one, and when `image_size` is not given that grid is the
    pinned class's own default. A mirrored default goes stale silently -- a version
    bump to 256 would mis-scale a requested pitch by 0.4 % and raise nothing -- so
    the mirror is checked against the signature it mirrors, and the branch that
    depends on it is exercised.

    `num_rays=8` keeps the 128x128 summation cheap (measured well under a second);
    the sampling is far too coarse to be a physics statement and nothing physical
    is asserted about it.
    """
    import inspect as inspect_module

    from optiland.psf import ScalarHuygensPSF

    from backends.optiland.analysis import _NATIVE_HUYGENS_IMAGE_SIZE

    pinned_default = inspect_module.signature(ScalarHuygensPSF).parameters["image_size"].default
    assert pinned_default == _NATIVE_HUYGENS_IMAGE_SIZE

    requested_pitch_m = 2.0e-06
    result = psf(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        method="huygens",
        num_rays=8,
        execution=EXECUTION,
        pixel_pitch_m=requested_pitch_m,
    )
    assert result.image_shape == (_NATIVE_HUYGENS_IMAGE_SIZE, _NATIVE_HUYGENS_IMAGE_SIZE)
    assert result.pixel_pitch_m == pytest.approx(requested_pitch_m, rel=1e-12)


def test_the_two_native_pixel_pitch_units_do_not_leak() -> None:
    """Criterion 9. One `pixel_pitch_m` in metres means one physical grid, on both.

    Measured on 0.6.0, `MMDFTPSF.pixel_pitch` is **micrometres** and
    `ScalarHuygensPSF.pixel_pitch` is **millimetres** -- a factor of 1000 between
    two attributes with the same name on two subclasses of the same base. The
    public argument is metres for both, and this is the test that says the
    asymmetry is handled in `analysis.py` and not left to a caller.

    It also covers the second half of the same conversion, which is not a unit: the
    Huygens attribute is *not* its own sample spacing (its `linspace` spans
    `image_size * pixel_pitch` across `image_size - 1` intervals), so a raw
    pass-through would describe a grid 3 % different from MMDFT's at these sizes.
    `coordinates()` is asserted equal rather than only the pitch, because that is
    the number a consumer plots against.
    """
    requested_pitch_m = 1.5e-06
    setup, source = singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0))
    common: dict[str, Any] = {
        "num_rays": 32,
        "execution": EXECUTION,
        "image_size": 33,
        "pixel_pitch_m": requested_pitch_m,
    }
    mmdft = psf(setup, source, method="mmdft", **common)
    huygens = psf(setup, source, method="huygens", **common)

    assert mmdft.pixel_pitch_m == pytest.approx(requested_pitch_m, rel=1e-12)
    assert huygens.pixel_pitch_m == pytest.approx(requested_pitch_m, rel=1e-12)
    for mine, theirs in zip(mmdft.coordinates(), huygens.coordinates(), strict=True):
        assert np.allclose(mine, theirs, rtol=1e-12, atol=0.0)
    # A metre-valued pitch is not a millimetre-valued one: the extent of a 33-sample
    # 1.5 um grid is 48 um, which is microns and not millimetres.
    assert 4.0e-05 < float(mmdft.coordinates()[1][-1] - mmdft.coordinates()[1][0]) < 5.0e-05


def test_every_method_returns_one_record_with_one_set_of_field_semantics() -> None:
    """Criterion 8. Three methods, one record type, one meaning per field.

    This is the property that makes `method` an argument rather than three
    operations: the return type does not vary with it, so one catalog record
    describes the callable.
    """
    setup, source = singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0))
    results = {
        "fft": psf(setup, source, method="fft", num_rays=32, execution=EXECUTION),
        "mmdft": psf(
            setup, source, method="mmdft", num_rays=32, execution=EXECUTION, image_size=33
        ),
        "huygens": psf(
            setup, source, method="huygens", num_rays=32, execution=EXECUTION, image_size=33
        ),
    }

    for method, result in results.items():
        assert isinstance(result, NativePsfAnalysis)
        assert result.method == method
        assert result.analysis == NATIVE_PSF_ANALYSES[method]
        assert result.method_definitions == NATIVE_PSF_METHOD_DEFINITIONS
        assert result.mode == "native"
        assert result.normalization == "strehl_percent"
        assert result.wavelength_m == pytest.approx(0.55e-6, rel=1e-12)
        assert result.strategy == "chief_ray"
        assert result.remove_tilt is False
        assert result.working_f_number == pytest.approx(WORKING_F_NUMBER, rel=1e-12)
        assert result.reference_sphere_radius_m == pytest.approx(
            REFERENCE_SPHERE_RADIUS_M, rel=1e-9
        )
        assert float(np.min(result.intensity)) >= 0.0

        # `coordinates()` reconstructs physical image coordinates for all three, on
        # the same origin rule, in metres.
        y, x = result.coordinates()
        ny, nx = result.image_shape
        assert (y.shape, x.shape) == ((ny,), (nx,))
        assert y[ny // 2] == pytest.approx(0.0, abs=1e-18)
        assert x[1] - x[0] == pytest.approx(result.pixel_pitch_m, rel=1e-12)
        # A PSF of an f/9.7 singlet at 550 nm is microns across, not millimetres and
        # not nanometres. The band is wide on purpose: this is a unit check.
        assert 1e-8 < float(x[-1]) < 1e-3


def test_the_field_analysed_is_the_one_declared_and_not_the_axis() -> None:
    """Criterion 11, off axis, where the mistake this catches is silent.

    `BasePSF` takes the field as a **normalized** `(Hx, Hy)` pair, so passing
    `(0, 0)` -- or normalizing against the wrong maximum -- analyses the axis and
    returns a perfectly plausible PSF of a different field point. Measured: the
    on-axis peak is 99.90 and the 5-degree peak is 93.04, so the two are
    distinguishable by 7 % of the normalization; at 3 degrees it is 98.66, and the
    monotone decrease is this singlet's off-axis aberration growing.

    That monotone decrease is also R16.1's "aberration lowers the PSF" case. The
    record does not carry an OPD span to report beside it -- no OPD crosses this
    boundary at all, by design -- so what is reported is the Strehl series itself.
    """
    peaks = {}
    for degrees, expected in OFF_AXIS_PEAKS.items():
        result = psf(
            singlet_ref(),
            singlet_source(field_angle_deg=(0.0, degrees)),
            method="fft",
            num_rays=64,
            execution=EXECUTION,
            grid_size=256,
        )
        assert result.field_angle_deg == (0.0, degrees)
        assert result.peak_intensity == pytest.approx(expected, rel=1e-5)
        peaks[degrees] = result.peak_intensity

    assert peaks[0.0] > peaks[3.0] > peaks[5.0]
    # The off-axis analysis is not the on-axis one under another label, which is the
    # failure a normalized-coordinate mistake produces.
    assert peaks[5.0] < 0.95 * peaks[0.0]


def test_characterization_remove_tilt_leaves_the_on_axis_peak() -> None:
    """**Characterization**, and the nearest thing this API can say about piston.

    R16.1 asks for "a uniform piston leaves the intensity unchanged". No OPD is
    injectable through this signature -- the pupil is produced inside the solver
    from a trace -- so that statement is not testable here and is not approximated.
    What is testable is the adjacent one the API does expose: on axis there is no
    tilt to remove, so `remove_tilt` moves the peak by less than 1e-9 relative.

    Off axis it would not be a no-op, and this test deliberately does not claim
    otherwise.
    """
    kept = _fft(remove_tilt=False)
    removed = _fft(remove_tilt=True)

    assert kept.remove_tilt is False
    assert removed.remove_tilt is True
    assert removed.peak_intensity == pytest.approx(kept.peak_intensity, rel=1e-9)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"method": "huygens", "grid_size": 64}, "does not apply"),
        ({"method": "mmdft", "grid_size": 64}, "does not apply"),
        ({"method": "fft", "image_size": 64}, "does not apply"),
        ({"method": "fft", "pixel_pitch_m": 1e-6}, "does not apply"),
        ({"method": "gerchberg"}, "is not one of"),
        ({"num_rays": 0}, "at least 1"),
        ({"method": "mmdft", "image_size": 33, "pixel_pitch_m": -1.0}, "positive length"),
        # A one-sample map has no adjacent samples, so it has no pitch to report.
        # Refused up front: `_native_sample_pitch_um`'s `(n - 1)` would otherwise
        # divide by zero in the translation, after the solver had done the work.
        ({"method": "huygens", "image_size": 1}, "at least 2"),
        ({"method": "mmdft", "image_size": 1, "pixel_pitch_m": 1e-6}, "at least 2"),
        ({"execution": {"device": "cpu"}}, "needs \\['precision'\\]"),
        (
            {"execution": {"device": "cpu", "precision": "fp64", "backend": "numpy"}},
            "does not take",
        ),
    ],
)
def test_an_unusable_request_is_refused_before_the_solver_runs(
    overrides: dict[str, Any], expected: str
) -> None:
    """Criterion 10, each refusal independently, and each of them a `ValueError`.

    A method-inapplicable sampling argument is refused rather than ignored, which
    is the same decision `Sampling.reference_surface` exists for: silently
    discarding a sampling argument computes a PSF at a sampling nobody asked for
    and reports no error.
    """
    call: dict[str, Any] = {"method": "fft", "num_rays": 32, "execution": EXECUTION}
    call.update(overrides)
    with pytest.raises(ValueError, match=expected):
        psf(singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0)), **call)


def test_a_finite_conjugate_source_is_refused_as_unsupported() -> None:
    """The same scope boundary and the same reason as `spot_diagram`'s.

    At a finite object distance `field_angle_deg` is a *position*
    (`problems.SourceSpec`), so handing it to the solver's `angle` field type would
    analyse a different object than the one declared. `NotImplementedError` and not
    `ValueError`: the source is well formed and this path does not support it.
    """
    with pytest.raises(NotImplementedError) as error:
        psf(
            finite_conjugate_singlet(),
            finite_conjugate_source(),
            method="fft",
            num_rays=32,
            execution=EXECUTION,
        )

    message = str(error.value)
    assert "infinite-conjugate" in message
    assert "object_distance" in message
    assert "spot_diagram" in message


def test_a_ray_bundle_is_not_a_source_for_this_path() -> None:
    """A caller holding rays cannot feed them to an analysis that generates its own.

    Unlike the spot case there is **no** project-side path that turns a `RayBundle`
    into a diffraction PSF at all, so the message says that rather than pointing at
    one: `measurements.psf` reduces a `ScalarField`.
    """
    rays = trace(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        sampling={"num_rings": 3, "reference_surface": "image_surface"},
        execution=EXECUTION,
    )
    with pytest.raises(TypeError) as error:
        psf(singlet_ref(), rays, method="fft", num_rays=32, execution=EXECUTION)  # type: ignore[arg-type]
    assert "SourceSpec" in str(error.value)
    assert "measurements.psf" in str(error.value)


def test_the_solver_refuses_a_sampling_it_cannot_execute() -> None:
    """The delegation working: an unusable sampling is the pinned solver's refusal.

    Two measured cases. Without a `grid_size` the FFT class requires at least 32
    pupil samples to emulate OpticStudio's grid at all; and the matrix DFT refuses
    an `image_size` beyond the pad size its own sampling admits. Neither is
    re-derived here, which is the same argument `spot_diagram` makes about
    `distribution`: a whitelist would be a second source of truth going stale at
    the next version bump.
    """
    setup, source = singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0))

    with pytest.raises(ValueError, match="num_rays must be at least 32"):
        psf(setup, source, method="fft", num_rays=16, execution=EXECUTION)
    with pytest.raises(ValueError, match="pad size"):
        psf(
            setup,
            source,
            method="mmdft",
            num_rays=32,
            execution=EXECUTION,
            image_size=4096,
            pixel_pitch_m=2.5e-06,
        )
    # And an unknown strategy is refused by the solver rather than by a whitelist here.
    with pytest.raises(ValueError, match="strategy"):
        psf(setup, source, method="fft", num_rays=32, execution=EXECUTION, strategy="nonsense")


def test_no_native_object_crosses_the_boundary() -> None:
    """Criterion 2, from both sides: the signature and the record.

    The AST walk in `test_optiland_boundary.py` proves no module outside this
    package imports optiland. This is the other half -- that the *values* and the
    *annotations* handed out are not native ones, which no import check can see.
    """
    result = _fft()
    for name, value in vars(result).items():
        assert isinstance(value, int | float | str | tuple | dict | bool | np.ndarray), (
            name,
            type(value),
        )
        assert type(value).__module__ in ("builtins", "numpy"), (name, type(value))

    # Whole words, because `OpticalSetup` legitimately contains `Optic` and a
    # substring test would report this project's own neutral record as a leak.
    # `RealRays` is deliberately not in this list even though it is the obvious
    # member: `test_optiland_boundary.py` forbids that name in any module outside
    # the package, including this one, so naming it here to check for it would
    # itself be the violation. The four below are the PSF path's native types.
    forbidden = ("Optic", "Wavefront", "WavefrontData", "BasePSF")
    pattern = re.compile(r"\b(" + "|".join(forbidden) + r")\b")
    signature = str(inspect.signature(psf))
    assert not pattern.search(signature), signature
    annotations = " ".join(
        str(annotation) for annotation in NativePsfAnalysis.__annotations__.values()
    )
    assert not pattern.search(annotations), annotations


def test_the_four_class_names_this_design_did_not_need_are_absent() -> None:
    """Criterion 1, as the absence R16.1's central decision is about.

    The rejected shape was a decomposition of the pinned solver's PSF pipeline into
    public wavefront, pupil-field, propagation-kernel and measurement nodes: four
    classes for graph flexibility no current consumer wants. `analysis.py` defines
    exactly the two result records and nothing else, and there is no second public
    PSF operation anywhere in the package.

    A budget records what exists and cannot record what was avoided, which is why
    this is a test -- the same arrangement R08.1's five absent names and R16's two
    have.
    """
    import backends.optiland as package
    import backends.optiland.analysis as module

    defined = {
        node.name
        for node in ast.walk(ast.parse(Path("src/backends/optiland/analysis.py").read_text()))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == {"NativePsfAnalysis", "NativeSpotAnalysis"}

    for absent in (
        "WavefrontNode",
        "PupilField",
        "PropagationKernel",
        "PsfMeasurement",
        # And no per-method public operation, which is the other half of criterion 1.
        "fft_psf",
        "mmdft_psf",
        "huygens_psf",
    ):
        assert not hasattr(module, absent), absent
        assert not hasattr(package, absent), absent

    assert package.OPERATIONS == ("psf", "spot_diagram", "trace", "trace_rays")
    assert sorted(name for name in package.__all__ if "psf" in name.lower()) == [
        "NATIVE_PSF_ANALYSES",
        "NATIVE_PSF_METHOD_DEFINITIONS",
        "NATIVE_PSF_NORMALIZATION",
        "NativePsfAnalysis",
        "PsfMethod",
        "psf",
    ]


def test_a_translation_error_would_not_survive_the_record() -> None:
    """`__post_init__`, on the failures a delegation actually produces.

    Thin on purpose -- re-deriving the PSF to check it would reimplement what this
    module wraps -- so what is checked is what a *wiring* mistake looks like: a
    declared shape that does not describe the map, a negative sample (an amplitude
    or a real part where an intensity belongs), a non-positive pitch, and more than
    one field or wavelength (a lens that did not come from `build_lens`).
    """
    import dataclasses

    good = _fft()
    for overrides, expected in (
        ({"image_shape": (32, 32)}, "does not describe the intensity map"),
        ({"intensity": np.asarray([[1.0, -1.0], [1.0, 1.0]]), "image_shape": (2, 2)},
         "negative sample"),
        ({"intensity": np.asarray([1.0, 2.0]), "image_shape": (2,)}, "one \\(ny, nx\\) map"),
        ({"pixel_pitch_m": 0.0}, "positive finite"),
        ({"wavelength_m": -1.0}, "positive finite"),
        ({"fields_analyzed": 2}, "exactly one field and one wavelength"),
    ):
        with pytest.raises(ValueError, match=expected):
            dataclasses.replace(good, **overrides)
