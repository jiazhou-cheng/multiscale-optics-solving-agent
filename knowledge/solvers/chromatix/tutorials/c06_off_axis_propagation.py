"""Example 6 / "Off-Axis Propagation" -- https://chromatix.readthedocs.io/en/latest/examples/off_axis_propagation/

Repo-owned reproduction of the off-axis example: `shift_grid` for relabelling a
field's coordinates, and `asm_propagate(shift_yx=...)` for computing a
laterally displaced output window either through a modified transfer kernel or
through a chirp-z transform (`use_czt=True`), including the tilted-illumination
(`kykx=`) case.

Upstream prints five numbers and this reproduction matches all five exactly:

| upstream | quantity |
|---|---|
| `1089.536` | field width `D = 2 * lambda * 1024` |
| `544.768` | aperture width `w = D / 2` |
| `108953.6` | propagation distance `z = 100 * D` |
| `[(-544, 543), (-544, 543)]` | `field.spatial_limits` before shifting |
| `[(-544, 543), (0, 1088)]` | `shifted_field.spatial_limits` after `shift_yx=(0, D/2)` |

Two findings beyond the published values, both measured:

**1. `kykx` means two different things in two Chromatix functions, a factor of
2*pi apart.** Measured by sweeping `asm_propagate(kykx=(0, -m/w))` over
`m = 2.5, 5, 10` on an untilted field and reading the 90%-energy centroid off an
unclipped `mode="full"` window:

| `kykx_x` | `lambda * |k| * z` | `|k| / (2*pi/lambda) * z` | measured |
|---|---|---|---|
| -0.004589 | 266.00 | 42.34 | **265.83** |
| -0.009178 | 532.00 | 84.67 | **532.30** |
| -0.018356 | 1064.00 | 169.34 | **1063.15** |

So `asm_propagate`'s `kykx` enters as a **spatial frequency in cycles per length
unit** (`sin(theta) = lambda * kykx`), and the displacement is **opposite in sign**
to the parameter. `plane_wave`'s `kykx`, by contrast, is an **angular wavenumber in
radians per length unit** (`sin(theta) = kykx / (2*pi/lambda)`) -- verified
directly here at the same wavelength, and independently in
`c05_scalable_angular_spectrum` where the upstream example writes
`kx = 2*pi/lambda * sin(20 deg)` and gets a 20-degree beam. Reading either
convention for the other is a 2*pi position error.

**2. The two shifted-propagation implementations do NOT agree in amplitude.** The
modified-kernel path and the CZT path (`use_czt=True, output_dx=field.dx`)
correlate at r = 0.9984, so they compute the same structure, but the CZT output is
smaller by an amplitude factor of **1.2686** (power ratio 1.613 = 1.2686^2), with
a 4.6% residual even after that scale is removed. Nothing upstream or in the
docstrings mentions a normalisation difference. A consumer that treats the two as
interchangeable will be 27% off in amplitude.

Also established:

* **`shift_grid` relabels coordinates without touching the field.** The shifted
  field's `u` is bit-identical to the original's; only `spatial_limits` move. A
  consumer that expected `shift_grid` to *translate* the data would be wrong.
* **Off-axis propagation does what it is for.** With the beam tilted off the
  window, the on-axis output captures 49.9% of the power, the correctly shifted
  window captures 94.5% -- the same as the untilted case -- and a window shifted
  the wrong way captures 1.6%.
* Every output is finite and `mode="same"` returns the input shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c06_off_axis_propagation",
    title="Off-Axis Propagation",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/off_axis_propagation/",
    demonstrates=(
        "chromatix.shift_grid, Field.spatial_limits, and "
        "asm_propagate(shift_yx=..., bandlimit=..., kykx=..., use_czt=..., "
        "output_dx=..., mode='same') for a laterally displaced output window."
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
UPSTREAM_LIMITS = [(-544, 543), (-544, 543)]
UPSTREAM_SHIFTED_LIMITS = [(-544, 543), (0, 1088)]


def build_field():
    from functools import partial

    import chromatix.functional as cf

    return cf.plane_wave(
        shape=SHAPE,
        dx=DXI,
        spectrum=SPECTRUM,
        pupil=partial(cf.square_pupil, w=APERTURE_W),
    )


def _rounded_limits(field) -> list[tuple[int, int]]:
    return [(int(axis[0]), int(axis[1])) for axis in field.spatial_limits]


def _peak_x(field) -> float:
    intensity = np.asarray(field.intensity, dtype=float).squeeze()
    n = intensity.shape[-1]
    dx = float(np.asarray(field.dx, dtype=float).ravel()[-1])
    coordinate = (np.arange(n) - n // 2) * dx
    weights = intensity.sum(axis=0)
    order = np.argsort(weights)[::-1]
    core = order[np.cumsum(weights[order]) <= 0.9 * weights.sum()]
    return float((coordinate[core] * weights[core]).sum() / weights[core].sum())


def run() -> TutorialResult:
    import chromatix.functional as cf
    from chromatix import shift_grid

    result = TutorialResult()
    field = build_field()
    result.record(
        field_width=D,
        aperture_width=APERTURE_W,
        propagation_distance=Z,
        spacing=DXI,
        spatial_limits=_rounded_limits(field),
    )
    result.check_close("field_width_matches_upstream", "reference", D, UPSTREAM_D, rel=1e-9)
    result.check_close("aperture_width_matches_upstream", "reference", APERTURE_W, UPSTREAM_W, rel=1e-9)
    result.check_close("propagation_distance_matches_upstream", "reference", Z, UPSTREAM_Z, rel=1e-9)
    result.check_true(
        "spatial_limits_match_upstream",
        "reference",
        _rounded_limits(field) == UPSTREAM_LIMITS,
        f"field.spatial_limits rounds to {_rounded_limits(field)}; upstream prints "
        f"{UPSTREAM_LIMITS}",
    )

    # -- shift_grid relabels coordinates, it does not translate data ------------
    shift_yx = (0.0, D / 2)
    shifted = shift_grid(field, shift_yx=shift_yx)
    data_unchanged = bool(np.array_equal(np.asarray(shifted.u), np.asarray(field.u)))
    result.record(
        shift_yx=list(shift_yx),
        shifted_spatial_limits=_rounded_limits(shifted),
        shift_grid_leaves_u_unchanged=data_unchanged,
    )
    result.check_true(
        "shifted_spatial_limits_match_upstream",
        "reference",
        _rounded_limits(shifted) == UPSTREAM_SHIFTED_LIMITS,
        f"shifted_field.spatial_limits rounds to {_rounded_limits(shifted)}; upstream "
        f"prints {UPSTREAM_SHIFTED_LIMITS}",
    )
    result.check_true(
        "shift_grid_relabels_coordinates_without_touching_the_field",
        "analytic",
        data_unchanged,
        "shifted_field.u is bit-identical to field.u; only spatial_limits move. "
        "shift_grid declares where the window sits, it does not translate the data -- "
        "which is why the upstream cell plots field.amplitude with a shifted extent "
        "rather than plotting shifted_field.",
    )

    # -- band-limited on-axis propagation --------------------------------------
    on_axis = cf.asm_propagate(field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True)
    result.record(
        on_axis_shape=list(np.asarray(on_axis.u).shape),
        on_axis_power=float(np.asarray(on_axis.power, dtype=float).ravel()[0]),
        input_power=float(np.asarray(field.power, dtype=float).ravel()[0]),
    )
    result.check_finite("on_axis_output_finite", np.abs(np.asarray(on_axis.u)))
    result.check_true(
        "mode_same_returns_the_input_shape",
        "invariant",
        tuple(np.asarray(on_axis.u).shape)[-2:] == SHAPE,
        f"output shape {tuple(np.asarray(on_axis.u).shape)} with mode='same' against the "
        f"{SHAPE} input -- the crop happens inside Chromatix, unlike mode='full'",
    )

    # -- modified kernel vs chirp-z transform ----------------------------------
    kernel_shifted = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True, shift_yx=shift_yx
    )
    czt_shifted = cf.asm_propagate(
        field,
        Z,
        N,
        pad_width=PAD_WIDTH,
        mode="same",
        bandlimit=True,
        shift_yx=shift_yx,
        output_dx=field.dx,
        use_czt=True,
    )
    a_kernel = np.asarray(kernel_shifted.amplitude, dtype=float).squeeze()
    a_czt = np.asarray(czt_shifted.amplitude, dtype=float).squeeze()
    kernel_rms = float(np.sqrt(np.mean(a_kernel**2)))
    optimal_scale = float((a_kernel * a_czt).sum() / (a_czt * a_czt).sum())
    residual_after_scaling = float(
        np.sqrt(np.mean((a_kernel - optimal_scale * a_czt) ** 2)) / kernel_rms
    )
    correlation = float(np.corrcoef(a_kernel.ravel(), a_czt.ravel())[0, 1])
    kernel_power = float(np.asarray(kernel_shifted.power, dtype=float).ravel()[0])
    czt_power = float(np.asarray(czt_shifted.power, dtype=float).ravel()[0])
    result.record(
        kernel_shifted_shape=list(a_kernel.shape),
        czt_shifted_shape=list(a_czt.shape),
        kernel_amplitude_rms=kernel_rms,
        czt_amplitude_rms=float(np.sqrt(np.mean(a_czt**2))),
        kernel_vs_czt_correlation=correlation,
        kernel_vs_czt_optimal_amplitude_scale=optimal_scale,
        kernel_vs_czt_residual_after_scaling=residual_after_scaling,
        kernel_power=kernel_power,
        czt_power=czt_power,
        kernel_over_czt_power_ratio=kernel_power / czt_power,
    )
    result.check_finite(
        "shifted_outputs_finite", np.concatenate([a_kernel.ravel(), a_czt.ravel()])
    )
    result.check_true(
        "the_modified_kernel_and_the_czt_compute_the_same_structure",
        "analytic",
        correlation > 0.99,
        f"Pearson r = {correlation:.6f} between the modified-transfer-kernel and chirp-z "
        "amplitudes for the same displaced window: the two independent algorithms agree "
        "on where the light goes",
    )
    result.check_true(
        "but_they_disagree_on_amplitude_normalisation",
        "analytic",
        optimal_scale > 1.1 and residual_after_scaling < 0.1,
        f"the CZT output is smaller by an amplitude factor of {optimal_scale:.4f} (power "
        f"ratio {kernel_power / czt_power:.4f} = {optimal_scale:.4f}^2), and even after "
        f"removing that scale a {residual_after_scaling * 100:.1f}% RMS residual remains. "
        "Nothing upstream or in the docstrings mentions a normalisation difference, so a "
        "consumer that treats use_czt=True as a drop-in alternative will be 27% off in "
        "amplitude.",
    )

    # -- what kykx means in asm_propagate --------------------------------------
    wavenumber = 2 * np.pi / SPECTRUM
    sweep = {}
    for multiplier in (2.5, 5.0, 10.0):
        kykx = (0.0, -multiplier / APERTURE_W)
        out = cf.asm_propagate(
            field, Z, N, pad_width=(2048, 2048), mode="full", bandlimit=True, kykx=kykx
        )
        sweep[f"{multiplier:g}"] = {
            "kykx_x": float(kykx[1]),
            "predicted_spatial_frequency_reading": SPECTRUM * abs(kykx[1]) * Z,
            "predicted_angular_wavenumber_reading": abs(kykx[1]) / wavenumber * Z,
            "measured_shift": _peak_x(out),
            "power": float(np.asarray(out.power, dtype=float).ravel()[0]),
        }
    result.record(asm_kykx_sweep=sweep, wavenumber=wavenumber)
    frequency_errors = [
        abs(e["measured_shift"] - e["predicted_spatial_frequency_reading"])
        / e["predicted_spatial_frequency_reading"]
        for e in sweep.values()
    ]
    angular_errors = [
        abs(e["measured_shift"] - e["predicted_angular_wavenumber_reading"])
        / e["predicted_angular_wavenumber_reading"]
        for e in sweep.values()
    ]
    result.record(
        asm_kykx_spatial_frequency_max_relative_error=max(frequency_errors),
        asm_kykx_angular_wavenumber_max_relative_error=max(angular_errors),
    )
    result.check_true(
        "asm_propagate_kykx_is_a_spatial_frequency_in_cycles_per_length",
        "analytic",
        max(frequency_errors) < 0.01 and min(angular_errors) > 1.0,
        "measured displacement matches lambda*|kykx|*z to "
        f"{max(frequency_errors) * 100:.2f}% at all three sweep points, and misses "
        f"|kykx|/(2*pi/lambda)*z by at least {min(angular_errors) * 100:.0f}%. Measured "
        + ", ".join(f"{e['measured_shift']:.2f}" for e in sweep.values())
        + " against "
        + ", ".join(f"{e['predicted_spatial_frequency_reading']:.2f}" for e in sweep.values())
        + " (frequency reading) and "
        + ", ".join(f"{e['predicted_angular_wavenumber_reading']:.2f}" for e in sweep.values())
        + " (angular reading).",
    )
    result.check_true(
        "the_displacement_is_opposite_in_sign_to_kykx",
        "analytic",
        all(e["measured_shift"] > 0.0 for e in sweep.values()),
        "every sweep point used a NEGATIVE kykx_x and produced a POSITIVE displacement: "
        + ", ".join(
            f"kykx={e['kykx_x']:+.6f} -> {e['measured_shift']:+.2f}" for e in sweep.values()
        ),
    )

    # -- plane_wave's kykx uses the OTHER convention ---------------------------
    import chromatix.functional as _cf

    sin_theta_target = 0.005
    plane_kykx = (0.0, wavenumber * sin_theta_target)
    tilted_source = _cf.plane_wave(
        shape=SHAPE, dx=DXI, spectrum=SPECTRUM, kykx=list(plane_kykx)
    )
    tilted_source = _cf.square_pupil(tilted_source, w=APERTURE_W)
    propagated_source = cf.asm_propagate(
        tilted_source, Z, N, pad_width=(2048, 2048), mode="full", bandlimit=True
    )
    plane_measured = _peak_x(propagated_source)
    plane_predicted_angular = Z * float(
        np.tan(np.arcsin(min(plane_kykx[1] / wavenumber, 0.999)))
    )
    plane_predicted_frequency = SPECTRUM * plane_kykx[1] * Z
    result.record(
        plane_wave_kykx_x=float(plane_kykx[1]),
        plane_wave_measured_shift=plane_measured,
        plane_wave_predicted_angular_reading=plane_predicted_angular,
        plane_wave_predicted_frequency_reading=plane_predicted_frequency,
    )
    result.check_close(
        "plane_wave_kykx_is_an_angular_wavenumber_in_radians_per_length",
        "analytic",
        plane_measured,
        plane_predicted_angular,
        rel=0.02,
    )
    result.check_true(
        "the_two_kykx_conventions_differ_by_two_pi",
        "analytic",
        abs(plane_measured - plane_predicted_frequency)
        > 10.0 * abs(plane_measured - plane_predicted_angular),
        f"plane_wave(kykx={plane_kykx[1]:.4f}) at lambda={SPECTRUM} displaces the beam "
        f"{plane_measured:.2f} over z={Z:.1f}, matching the ANGULAR reading "
        f"{plane_predicted_angular:.2f} and not the frequency reading "
        f"{plane_predicted_frequency:.2f}. asm_propagate's kykx is the other way round. "
        "The two parameters share a name and differ by a factor of 2*pi/lambda vs "
        "lambda, i.e. 2*pi in the sin(theta) relation.",
    )

    # -- off-axis windowing is what the feature is for -------------------------
    tilt_kykx = (0.0, -5 / APERTURE_W)
    windows = {
        "untilted_on_axis": cf.asm_propagate(
            field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True
        ),
        "tilted_on_axis": cf.asm_propagate(
            field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True, kykx=tilt_kykx
        ),
        "tilted_shift_matching": cf.asm_propagate(
            field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True,
            kykx=tilt_kykx, shift_yx=(0.0, D / 2),
        ),
        "tilted_shift_opposite": cf.asm_propagate(
            field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True,
            kykx=tilt_kykx, shift_yx=(0.0, -D / 2),
        ),
    }
    captured = {
        name: float(np.asarray(out.power, dtype=float).ravel()[0])
        for name, out in windows.items()
    }
    result.record(power_captured_per_window=captured)
    for name, out in windows.items():
        result.check_finite(f"{name}_finite", np.abs(np.asarray(out.u)))
    result.check_true(
        "the_correctly_shifted_window_recovers_the_tilted_beam",
        "analytic",
        captured["tilted_on_axis"] < 0.6
        and captured["tilted_shift_matching"] > 0.9
        and captured["tilted_shift_opposite"] < 0.1,
        "power captured: untilted on-axis "
        f"{captured['untilted_on_axis']:.4f}, tilted on-axis "
        f"{captured['tilted_on_axis']:.4f}, tilted with the matching shift "
        f"{captured['tilted_shift_matching']:.4f}, tilted with the opposite shift "
        f"{captured['tilted_shift_opposite']:.4f}. The correctly shifted window recovers "
        "as much as the untilted case; the on-axis window loses half the beam and the "
        "wrongly shifted one loses almost all of it. That is what off-axis propagation is "
        "for, stated as numbers.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
