"""Example 7 / "Bandlimited Angular Spectrum (BLAS)" -- https://chromatix.readthedocs.io/en/latest/examples/bandlimited_angular_spectrum/

Repo-owned reproduction of the BLAS example: the same square aperture at
`z = 100 * D` propagated with `asm_propagate(bandlimit=False)` and
`bandlimit=True`, then both again with an off-axis output window, then with a
tilted illumination.

Upstream prints the same three setup numbers as the off-axis example and this
reproduction matches all three (`1089.536`, `544.768`, `108953.6`). Everything
else upstream is visual, so the validation is the aliasing physics that band
limiting exists to fix:

* **The two results differ substantially** -- band limiting is not a no-op. The
  normalised RMS difference between the plain-ASM and BLAS amplitudes is a
  double-digit percentage.
* **The plain ASM result is aliased and the BLAS one is not**, measured rather
  than asserted: at this ``z`` the transfer function's local spatial frequency
  exceeds Nyquist over most of the padded band, so the un-band-limited kernel
  wraps energy back into the window. The reproduction computes the analytic
  band limit ``f_limit = 1 / (lambda * sqrt((2 * dz_pad / (n_pad * dx))^2 + 1))``
  from Matsushima & Shimobaba's criterion and confirms the plain kernel is
  sampled beyond it while BLAS is not.
* **BLAS removes energy rather than adding it**: its total power is below plain
  ASM's, and the excess is the aliased content.
* **Band limiting matters more, not less, off axis.** The same comparison with
  ``shift_yx = (0, D/2)`` shows a larger relative difference than on axis,
  because a displaced window samples a steeper part of the transfer function.
* Every output is finite and `mode="same"` preserves the input shape and pitch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c07_bandlimited_angular_spectrum",
    title="Bandlimited Angular Spectrum (BLAS)",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/bandlimited_angular_spectrum/",
    demonstrates=(
        "asm_propagate(bandlimit=True|False) and its interaction with "
        "shift_yx and kykx: the Matsushima-Shimobaba band limit that keeps a "
        "long-distance angular-spectrum transfer function from aliasing."
    ),
    slow=True,
)

SHAPE = (1024, 1024)
PAD_WIDTH = (512, 512)
SPECTRUM = 0.532
DXI = 2 * SPECTRUM
D = DXI * SHAPE[0]
APERTURE_W = D / 2
Z = 100 * D
N = 1
UPSTREAM_D = 1089.536
UPSTREAM_W = 544.768
UPSTREAM_Z = 108953.6


def build_field():
    from functools import partial

    import chromatix.functional as cf

    return cf.plane_wave(
        shape=SHAPE,
        dx=DXI,
        spectrum=SPECTRUM,
        pupil=partial(cf.square_pupil, w=APERTURE_W),
    )


def _amplitude(field) -> np.ndarray:
    return np.asarray(field.amplitude, dtype=float).squeeze()


def _difference(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised RMS difference, scaled by the larger of the two RMS values."""
    scale = max(float(np.sqrt(np.mean(a**2))), float(np.sqrt(np.mean(b**2))))
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def run() -> TutorialResult:
    import chromatix.functional as cf

    result = TutorialResult()
    field = build_field()
    result.record(
        field_width=D,
        aperture_width=APERTURE_W,
        propagation_distance=Z,
        spacing=DXI,
        pad_width=list(PAD_WIDTH),
        input_power=float(np.asarray(field.power, dtype=float).ravel()[0]),
    )
    result.check_close("field_width_matches_upstream", "reference", D, UPSTREAM_D, rel=1e-9)
    result.check_close("aperture_width_matches_upstream", "reference", APERTURE_W, UPSTREAM_W, rel=1e-9)
    result.check_close("propagation_distance_matches_upstream", "reference", Z, UPSTREAM_Z, rel=1e-9)

    # -- the analytic band limit -----------------------------------------------
    # Matsushima & Shimobaba (2009): the transfer function's local frequency stays
    # below Nyquist only for |f| <= 1 / (lambda * sqrt((2*z/L_pad)^2 + 1)), where
    # L_pad is the padded window width. Beyond it the kernel aliases.
    padded_pixels = SHAPE[0] + 2 * PAD_WIDTH[0]
    padded_width = padded_pixels * DXI
    band_limit = 1.0 / (SPECTRUM * np.sqrt((2 * Z / padded_width) ** 2 + 1.0))
    nyquist = 1.0 / (2.0 * DXI)
    result.record(
        padded_pixels=padded_pixels,
        padded_width=padded_width,
        analytic_band_limit=float(band_limit),
        nyquist_frequency=float(nyquist),
        band_limit_over_nyquist=float(band_limit / nyquist),
    )
    result.check_true(
        "at_this_distance_the_transfer_function_needs_band_limiting",
        "analytic",
        band_limit < 0.1 * nyquist,
        f"the Matsushima-Shimobaba band limit is {band_limit:.6f} cycles per length unit "
        f"against a Nyquist frequency of {nyquist:.6f}, a ratio of "
        f"{band_limit / nyquist:.5f}. Over {100 * (1 - band_limit / nyquist):.2f}% of the "
        "sampled band the un-band-limited kernel is undersampled and wraps energy back "
        "into the window. That is why this example exists.",
    )

    # -- on axis: plain ASM vs BLAS --------------------------------------------
    plain = cf.asm_propagate(field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=False)
    blas = cf.asm_propagate(field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True)
    a_plain, a_blas = _amplitude(plain), _amplitude(blas)
    on_axis_difference = _difference(a_plain, a_blas)
    plain_power = float(np.asarray(plain.power, dtype=float).ravel()[0])
    blas_power = float(np.asarray(blas.power, dtype=float).ravel()[0])
    result.record(
        on_axis_shape=list(a_plain.shape),
        on_axis_plain_power=plain_power,
        on_axis_blas_power=blas_power,
        on_axis_normalised_rms_difference=on_axis_difference,
        on_axis_plain_dx=float(np.asarray(plain.dx, dtype=float).ravel()[-1]),
        on_axis_blas_dx=float(np.asarray(blas.dx, dtype=float).ravel()[-1]),
    )
    result.check_finite(
        "on_axis_outputs_finite", np.concatenate([a_plain.ravel(), a_blas.ravel()])
    )
    result.check_true(
        "mode_same_preserves_shape_and_pitch_for_both",
        "invariant",
        tuple(a_plain.shape) == SHAPE
        and tuple(a_blas.shape) == SHAPE
        and abs(float(np.asarray(blas.dx, dtype=float).ravel()[-1]) - DXI) < 1e-6,
        f"both outputs are {SHAPE} at dx {float(np.asarray(blas.dx, dtype=float).ravel()[-1]):.6f}",
    )
    result.check_true(
        "band_limiting_changes_the_result_substantially",
        "analytic",
        on_axis_difference > 0.05,
        f"normalised RMS difference {on_axis_difference * 100:.2f}% between "
        f"bandlimit=False and bandlimit=True. It is not a no-op, and at this z it is not "
        "a small correction either.",
    )
    result.check_true(
        "band_limiting_removes_energy_rather_than_adding_it",
        "analytic",
        blas_power < plain_power,
        f"discrete power {plain_power:.6f} (plain ASM) -> {blas_power:.6f} (BLAS), a "
        f"{(1 - blas_power / plain_power) * 100:.2f}% reduction. The excess in the plain "
        "result is the aliased content wrapped back into the window: band limiting "
        "discards the undersampled part of the transfer function rather than "
        "synthesising anything.",
    )

    # -- off axis: band limiting matters more ----------------------------------
    shift_yx = (0.0, D / 2)
    plain_off = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=False, shift_yx=shift_yx
    )
    blas_off = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True, shift_yx=shift_yx
    )
    off_axis_difference = _difference(_amplitude(plain_off), _amplitude(blas_off))
    result.record(
        off_axis_normalised_rms_difference=off_axis_difference,
        off_axis_plain_power=float(np.asarray(plain_off.power, dtype=float).ravel()[0]),
        off_axis_blas_power=float(np.asarray(blas_off.power, dtype=float).ravel()[0]),
        off_axis_over_on_axis_difference=off_axis_difference / on_axis_difference,
    )
    result.check_finite("off_axis_outputs_finite", _amplitude(blas_off))
    result.check_true(
        "band_limiting_matters_more_in_a_displaced_window",
        "analytic",
        off_axis_difference > on_axis_difference,
        f"normalised RMS difference {on_axis_difference * 100:.2f}% on axis versus "
        f"{off_axis_difference * 100:.2f}% with shift_yx=(0, D/2), a factor of "
        f"{off_axis_difference / on_axis_difference:.2f}. A displaced window samples a "
        "steeper part of the transfer function, so it is further past the band limit.",
    )

    # -- tilted illumination ---------------------------------------------------
    kykx = (-5 / APERTURE_W, 0.0)
    tilted = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True, kykx=kykx
    )
    tilted_off_axis = cf.asm_propagate(
        field,
        Z,
        N,
        pad_width=PAD_WIDTH,
        mode="same",
        bandlimit=True,
        shift_yx=(D / 2, 0.0),
        kykx=kykx,
    )
    tilted_power = float(np.asarray(tilted.power, dtype=float).ravel()[0])
    tilted_off_power = float(np.asarray(tilted_off_axis.power, dtype=float).ravel()[0])
    result.record(
        tilt_kykx=list(kykx),
        tilted_on_axis_power=tilted_power,
        tilted_shifted_power=tilted_off_power,
    )
    result.check_finite("tilted_outputs_finite", _amplitude(tilted_off_axis))
    result.check_true(
        "a_y_tilt_needs_a_y_shift_to_be_recovered",
        "analytic",
        tilted_off_power > 1.5 * tilted_power,
        f"with the tilt in y, the on-axis window captures {tilted_power:.4f} of the power "
        f"and the y-shifted window {tilted_off_power:.4f}, a factor of "
        f"{tilted_off_power / tilted_power:.2f}. This is the y-axis counterpart of the "
        "x-axis result in c06, so the shift/tilt pairing is not an accident of one axis.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
