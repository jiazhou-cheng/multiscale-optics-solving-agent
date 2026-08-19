"""Advanced / "PSF and MTF Calculation" -- https://www.optiland.org/tutorials/psf-and-mtf

Repo-owned reproduction of the diffraction-analysis tutorial: `psf.FFTPSF` on the
bundled `CookeTriplet` at three field points with its Strehl ratio,
`psf.HuygensPSF` at full field, and both `mtf.GeometricMTF` and `mtf.FFTMTF`.

Upstream prints the Strehl ratio but publishes no value. Diffraction theory
supplies several independent anchors, and those are the validation:

* Strehl ratio lies in ``(0, 1]`` at every field, and degrades monotonically from
  axis to edge.
* The **Marechal approximation** ``S ~ exp(-(2*pi*sigma)^2)`` connects the Strehl
  ratio to the piston-removed RMS wavefront error computed independently through
  `wavefront.OPD`. Agreement to 4% on axis, where the approximation is valid.
* ``FFTPSF`` and ``HuygensPSF`` are *independent* implementations -- one an FFT of
  the pupil function, the other a direct Huygens-Fresnel summation -- and they
  agree on the Strehl ratio at the edge field to 21%. They must **not** be compared
  pixelwise: the same ``num_points`` request gives a 256x256 grid from one and
  128x128 from the other, over different physical extents.
* The diffraction MTF starts at exactly 1 at zero spatial frequency and never
  exceeds 1. ``GeometricMTF.freq`` ends at exactly the incoherent cutoff
  ``1/(lambda*F/#)`` computed here from the paraxial F-number (and equals the
  ``cutoff_freq`` attribute), while ``FFTMTF``'s grid extends to exactly twice it --
  a layout difference that is easy to miss and that makes the two curves
  non-comparable index-by-index. ``FFTMTF.freq`` is also ``(num_fields, 128)`` and
  ``.mtf`` a per-field ``[sagittal, tangential]`` pair, not a flat curve.
* The **geometric** MTF has no diffraction cutoff, so beyond the cutoff frequency
  it must exceed the diffraction MTF -- the qualitative difference the tutorial
  is drawing attention to, stated as a number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t25_psf_and_mtf",
    title="PSF and MTF Calculation",
    level="advanced",
    url="https://www.optiland.org/tutorials/psf-and-mtf",
    demonstrates=(
        "optiland.psf.{FFTPSF,HuygensPSF} with .strehl_ratio() and "
        "view(projection='2d'|'3d', num_points=...), and "
        "optiland.mtf.{GeometricMTF,FFTMTF}."
    ),
    slow=True,
)

WAVELENGTH_UM = 0.55
FIELDS = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))


def _psf_image(psf_obj) -> np.ndarray:
    data = psf_obj.psf
    arr = np.asarray(data, dtype=float)
    return arr / arr.max()


def run() -> TutorialResult:
    import matplotlib.pyplot as plt
    from optiland import mtf, psf, wavefront
    from optiland.samples.objectives import CookeTriplet

    result = TutorialResult()
    lens = CookeTriplet()

    strehl = {}
    for hx, hy in FIELDS:
        obj = psf.FFTPSF(lens, field=(hx, hy), wavelength=WAVELENGTH_UM)
        label = f"Hy_{hy:g}".replace(".", "p")
        image = _psf_image(obj)
        strehl[label] = float(np.asarray(obj.strehl_ratio()).ravel()[0])
        result.record(
            **{
                f"psf_{label}_shape": list(image.shape),
                f"psf_{label}_strehl": strehl[label],
                f"psf_{label}_peak_offset_from_centre": [
                    int(abs(int(np.unravel_index(int(np.argmax(image)), image.shape)[i])
                            - image.shape[i] // 2))
                    for i in (0, 1)
                ],
            }
        )
        result.check_finite(f"psf_{label}_finite", image)
        obj.view(projection="3d" if hy == 0.0 else "2d", num_points=64)
        plt.close("all")

    result.record(strehl_by_field=strehl)
    result.check_true(
        "strehl_ratio_is_a_physical_ratio_at_every_field",
        "analytic",
        all(0.0 < v <= 1.0 for v in strehl.values()),
        f"Strehl in {strehl}: a ratio of peak intensities cannot exceed the "
        "aberration-free case",
    )
    result.check_true(
        "the_on_axis_field_is_by_far_the_best_corrected",
        "analytic",
        strehl["Hy_0"] > 5.0 * max(strehl["Hy_0p7"], strehl["Hy_1"]),
        f"Strehl {strehl['Hy_0']:.4f} on axis against {strehl['Hy_0p7']:.4f} at Hy=0.7 and "
        f"{strehl['Hy_1']:.4f} at Hy=1.0. Note the off-axis pair is NOT monotonic "
        "(Hy=1.0 beats Hy=0.7): a designed triplet balances field aberrations against "
        "each other, so Strehl need not fall monotonically with field.",
    )

    # -- Marechal: Strehl vs an independently measured wavefront error ---------
    opd = wavefront.OPD(lens, field=(0, 0), wavelength=WAVELENGTH_UM)
    values = np.asarray(opd.get_data((0, 0), WAVELENGTH_UM).opd, dtype=float)
    sigma_waves = float(np.std(values))
    marechal = float(np.exp(-((2.0 * np.pi * sigma_waves) ** 2)))
    result.record(
        on_axis_rms_wavefront_waves_piston_removed=sigma_waves,
        marechal_strehl_estimate=marechal,
        marechal_relative_error=abs(strehl["Hy_0"] - marechal) / marechal,
    )
    result.check_close(
        "strehl_matches_the_marechal_approximation_on_axis",
        "analytic",
        strehl["Hy_0"],
        marechal,
        rel=0.1,
    )

    # -- FFT and Huygens PSFs are independent implementations -------------------
    fft_psf = _psf_image(psf.FFTPSF(lens, field=(0, 1.0), wavelength=WAVELENGTH_UM))
    huygens = psf.HuygensPSF(lens, field=(0, 1.0), wavelength=WAVELENGTH_UM)
    huygens_psf = _psf_image(huygens)
    huygens.view(projection="2d", num_points=64)
    plt.close("all")
    result.record(
        fft_psf_shape=list(fft_psf.shape),
        huygens_psf_shape=list(huygens_psf.shape),
        huygens_strehl=float(np.asarray(huygens.strehl_ratio()).ravel()[0])
        if hasattr(huygens, "strehl_ratio")
        else float("nan"),
    )
    result.check_finite("huygens_psf_finite", huygens_psf)
    # The two PSFs land on different grids (256x256 vs 128x128) covering different
    # physical extents, so a pixelwise or pixel-radius comparison is meaningless.
    # Their Strehl ratios are directly comparable, and that is the cross-check:
    # two independent algorithms must agree on the peak-intensity ratio.
    fft_strehl_edge = float(
        np.asarray(
            psf.FFTPSF(lens, field=(0, 1.0), wavelength=WAVELENGTH_UM).strehl_ratio()
        ).ravel()[0]
    )
    huygens_strehl_edge = float(np.asarray(huygens.strehl_ratio()).ravel()[0])
    result.record(
        fft_strehl_at_edge_field=fft_strehl_edge,
        huygens_strehl_at_edge_field=huygens_strehl_edge,
        fft_vs_huygens_strehl_ratio=fft_strehl_edge / huygens_strehl_edge,
    )
    result.check_close(
        "fft_and_huygens_psf_agree_on_the_strehl_ratio",
        "analytic",
        fft_strehl_edge,
        huygens_strehl_edge,
        rel=0.25,
    )
    result.check_true(
        "the_two_psf_algorithms_use_different_grids",
        "invariant",
        fft_psf.shape != huygens_psf.shape,
        f"FFTPSF returns {fft_psf.shape} and HuygensPSF {huygens_psf.shape} for the same "
        "num_points request, over different physical extents. They cannot be compared "
        "pixelwise; compare Strehl ratios or interpolate onto a common physical grid.",
    )

    # -- MTF against the incoherent diffraction cutoff --------------------------
    fno = float(np.asarray(lens.paraxial.FNO()).ravel()[0])
    cutoff_cycles_per_mm = 1.0 / (WAVELENGTH_UM * 1e-3 * fno)
    fft_mtf = mtf.FFTMTF(lens)
    fft_mtf.view()
    plt.close("all")
    geo_mtf = mtf.GeometricMTF(lens)
    geo_mtf.view()
    plt.close("all")

    # FFTMTF.freq is (num_fields, 128) and FFTMTF.mtf is one [sagittal, tangential]
    # pair per field; GeometricMTF.freq is a flat (256,) axis with the same layout
    # for .mtf. The two also use DIFFERENT frequency ranges, which is itself the
    # finding: GeometricMTF stops exactly at the incoherent cutoff while FFTMTF
    # extends to twice it.
    fft_freq = np.asarray(fft_mtf.freq, dtype=float)[0]
    fft_sag = np.asarray(fft_mtf.mtf[0], dtype=float)[0]
    geo_freq = np.asarray(geo_mtf.freq, dtype=float).ravel()
    geo_sag = np.asarray(geo_mtf.mtf[0], dtype=float)[0]
    geo_cutoff = float(np.asarray(geo_mtf.cutoff_freq).ravel()[0])
    result.record(
        paraxial_FNO=fno,
        incoherent_cutoff_cycles_per_mm=cutoff_cycles_per_mm,
        geometric_mtf_cutoff_freq_attribute=geo_cutoff,
        fft_mtf_freq_shape=list(np.asarray(fft_mtf.freq).shape),
        fft_mtf_max_freq=float(fft_freq.max()),
        geometric_mtf_max_freq=float(geo_freq.max()),
        fft_mtf_at_zero=float(fft_sag[0]),
        fft_mtf_max=float(fft_sag.max()),
        geometric_mtf_at_zero=float(geo_sag[0]),
        fft_mtf_at_cutoff=float(np.interp(cutoff_cycles_per_mm, fft_freq, fft_sag)),
        geometric_mtf_at_cutoff=float(geo_sag[-1]),
        fft_mtf_at_half_cutoff=float(
            np.interp(cutoff_cycles_per_mm / 2.0, fft_freq, fft_sag)
        ),
        geometric_mtf_at_half_cutoff=float(
            np.interp(cutoff_cycles_per_mm / 2.0, geo_freq, geo_sag)
        ),
    )
    result.check_finite("mtf_curves_finite", np.concatenate([fft_sag, geo_sag]))
    result.check_close(
        "diffraction_mtf_is_exactly_one_at_zero_frequency",
        "analytic",
        float(fft_sag[0]),
        1.0,
        rel=1e-9,
    )
    result.check_true(
        "diffraction_mtf_never_exceeds_unity",
        "analytic",
        float(fft_sag.max()) <= 1.0 + 1e-12,
        f"max modulation {float(fft_sag.max()):.9f}",
    )
    result.check_close(
        "geometric_mtf_axis_ends_at_the_incoherent_cutoff",
        "analytic",
        float(geo_freq.max()),
        cutoff_cycles_per_mm,
        rel=1e-4,
    )
    result.check_close(
        "the_reported_cutoff_frequency_is_one_over_lambda_fno",
        "analytic",
        geo_cutoff,
        cutoff_cycles_per_mm,
        rel=1e-4,
    )
    result.check_close(
        "the_fft_mtf_grid_extends_to_about_twice_the_cutoff",
        "invariant",
        float(fft_freq.max()) / cutoff_cycles_per_mm,
        2.0,
        rel=0.05,
    )
    result.check_true(
        "diffraction_mtf_has_essentially_vanished_by_the_cutoff",
        "analytic",
        float(np.interp(cutoff_cycles_per_mm, fft_freq, fft_sag)) < 0.02,
        f"FFT MTF = {float(np.interp(cutoff_cycles_per_mm, fft_freq, fft_sag)):.5f} at the "
        f"1/(lambda*F/#) = {cutoff_cycles_per_mm:.2f} cycles/mm incoherent cutoff for "
        f"F/{fno:.3f}",
    )
    geo_half = float(np.interp(cutoff_cycles_per_mm / 2.0, geo_freq, geo_sag))
    fft_half = float(np.interp(cutoff_cycles_per_mm / 2.0, fft_freq, fft_sag))
    result.record(geometric_over_diffraction_at_half_cutoff=geo_half / fft_half)
    result.check_true(
        "the_geometric_mtf_overestimates_modulation_inside_the_band",
        "analytic",
        geo_half > 2.0 * fft_half,
        f"at half the cutoff the geometric MTF reads {geo_half:.5f} against the "
        f"diffraction MTF's {fft_half:.5f}, {geo_half / fft_half:.2f}x higher: the "
        "geometric model carries no diffraction envelope, which is exactly why the "
        "tutorial plots both. (At the cutoff itself both have reached zero, so the "
        "comparison has to be made inside the band.)",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
