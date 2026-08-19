"""Example 8 / "Scaled and Shifted Free-Space Propagation" -- https://chromatix.readthedocs.io/en/latest/examples/rescaled_propagation/

Repo-owned reproduction of the rescaled-propagation example: a zoomed, laterally
shifted output window computed three ways -- a brute-force 4x-oversampled BLAS
propagation, the Yu et al. modified-kernel method (`output_dx=..., use_czt=False`)
and the chirp-z method (`use_czt=True`) -- plus the Fresnel counterparts through
`transfer_propagate(output_dx=..., shift_yx=...)`.

**This example publishes the number that explains the c06 anomaly.** Upstream's
own output records

    Yu et al.  ->  Output field norm : 3.1434343
    CZT        ->  Output field norm : 44.420246

i.e. the two methods differ in amplitude by a factor of **14.1** on upstream's own
machine, and the example then compares them only *after* normalising each by its
own norm, where they agree to a normalised RMSE of `7.2583052e-06`. So the
amplitude-scale disagreement `c06_off_axis_propagation` measured (1.27 there, at a
different zoom) is **upstream-known behaviour, not a pinned-environment
regression** -- and it is documented nowhere except in this cell's printed output.
A consumer must normalise before comparing, and must not read either norm as a
physical power.

Reproduced references:

| upstream | quantity |
|---|---|
| `1089.536` / `108.9536` / `10895.36` | `D`, `w = D/10`, `z = 10*D` |
| `float32` | `Field.amplitude.dtype` |
| `(1024, 1024)` | `Field.spatial_shape` with `mode="same"` |
| `(4096, 4096)` | the 4x-oversampled reference input shape |
| `7.2583052e-06` | normalised RMSE between the Yu et al. and CZT amplitudes |
| a `np.allclose(..., atol=1e-4)` assertion | passes |

Plus two things upstream does not check:

* **Both rescaled methods reproduce the brute-force reference.** Each is compared,
  after normalisation, against the corresponding crop of the 4096x4096 BLAS
  propagation. That is the check that establishes they are *correct* rather than
  merely mutually consistent.
* **`transfer_propagate` accepts the same `output_dx`/`shift_yx`** and produces a
  finite, shape-correct result, so the rescaling machinery is not ASM-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c08_rescaled_propagation",
    title="Scaled and Shifted Free-Space Propagation",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/rescaled_propagation/",
    demonstrates=(
        "asm_propagate(output_dx=..., shift_yx=..., use_czt=False|True) for a "
        "zoomed and shifted output window, the same arguments on "
        "transfer_propagate, and Field.spatial_shape / Field.spatial_limits / "
        "Field.extent."
    ),
    slow=True,
)

SHAPE = (1024, 1024)
PAD_WIDTH = (SHAPE[0] // 2, SHAPE[1] // 2)
SPECTRUM = 0.532
DXI = 2 * SPECTRUM
D = DXI * SHAPE[0]
APERTURE_W = D / 10
Z = 10 * D
N = 1
ZOOM = 4
UPSTREAM_D = 1089.536
UPSTREAM_W = 108.95360000000001
UPSTREAM_Z = 10895.36
UPSTREAM_YU_NORM = 3.1434343
UPSTREAM_CZT_NORM = 44.420246
UPSTREAM_NORMALISED_RMSE = 7.2583052e-06


def build_field(shape=SHAPE, spacing=None):
    from functools import partial

    import chromatix.functional as cf

    return cf.plane_wave(
        shape=shape,
        dx=DXI if spacing is None else spacing,
        spectrum=SPECTRUM,
        pupil=partial(cf.square_pupil, w=APERTURE_W),
    )


def _normalised(field) -> np.ndarray:
    amplitude = np.asarray(field.amplitude, dtype=float).squeeze()
    return amplitude / float(np.linalg.norm(amplitude))


def run() -> TutorialResult:
    import numpy as _np

    import chromatix.functional as cf

    result = TutorialResult()
    field = build_field()
    shift_yx = [0.0, APERTURE_W / 2]
    result.record(
        field_width=D,
        aperture_width=APERTURE_W,
        propagation_distance=Z,
        zoom_factor=ZOOM,
        shift_yx=list(shift_yx),
    )
    result.check_close("field_width_matches_upstream", "reference", D, UPSTREAM_D, rel=1e-9)
    result.check_close("aperture_width_matches_upstream", "reference", APERTURE_W, UPSTREAM_W, rel=1e-9)
    result.check_close("propagation_distance_matches_upstream", "reference", Z, UPSTREAM_Z, rel=1e-9)

    # -- baseline BLAS ---------------------------------------------------------
    baseline = cf.asm_propagate(field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True)
    result.record(
        baseline_amplitude_dtype=str(np.asarray(baseline.amplitude).dtype),
        baseline_spatial_shape=list(baseline.spatial_shape),
    )
    result.check_true(
        "the_output_amplitude_dtype_and_shape_match_upstream",
        "reference",
        str(np.asarray(baseline.amplitude).dtype) == "float32"
        and tuple(baseline.spatial_shape) == SHAPE,
        f"amplitude dtype {np.asarray(baseline.amplitude).dtype} and spatial_shape "
        f"{tuple(baseline.spatial_shape)}; upstream prints float32 and (1024, 1024)",
    )

    # -- the 4x oversampled brute-force reference ------------------------------
    hi_res_shape = (SHAPE[0] * ZOOM, SHAPE[1] * ZOOM)
    hi_res_spacing = _np.asarray(DXI / ZOOM)[..., _np.newaxis]
    hi_res_field = build_field(shape=hi_res_shape, spacing=hi_res_spacing)
    hi_res_pad = (hi_res_shape[0] // 2, hi_res_shape[1] // 2)
    result.record(hi_res_input_spatial_shape=list(hi_res_field.spatial_shape))
    result.check_true(
        "the_oversampled_reference_input_shape_matches_upstream",
        "reference",
        tuple(hi_res_field.spatial_shape) == hi_res_shape,
        f"{tuple(hi_res_field.spatial_shape)}; upstream prints (4096, 4096)",
    )
    hi_res = cf.asm_propagate(
        hi_res_field, Z, N, pad_width=hi_res_pad, mode="same", bandlimit=True
    )
    hi_res_amplitude = np.asarray(hi_res.amplitude, dtype=float).squeeze()

    # -- the two rescaled methods ---------------------------------------------
    yu = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True,
        output_dx=field.dx / ZOOM, shift_yx=shift_yx, use_czt=False,
    )
    czt = cf.asm_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same", bandlimit=True,
        output_dx=field.dx / ZOOM, shift_yx=shift_yx, use_czt=True,
    )
    yu_amplitude = np.asarray(yu.amplitude, dtype=float).squeeze()
    czt_amplitude = np.asarray(czt.amplitude, dtype=float).squeeze()
    yu_norm = float(np.linalg.norm(yu_amplitude))
    czt_norm = float(np.linalg.norm(czt_amplitude))
    normalised_rmse = float(np.sqrt(np.mean((_normalised(yu) - _normalised(czt)) ** 2)))
    allclose = bool(np.allclose(_normalised(yu), _normalised(czt), atol=1e-4))
    result.record(
        yu_spatial_shape=list(yu.spatial_shape),
        czt_spatial_shape=list(czt.spatial_shape),
        yu_amplitude_norm=yu_norm,
        czt_amplitude_norm=czt_norm,
        czt_over_yu_norm_ratio=czt_norm / yu_norm,
        normalised_rmse=normalised_rmse,
        normalised_allclose_atol_1e_4=allclose,
    )
    result.check_finite(
        "rescaled_outputs_finite", np.concatenate([yu_amplitude.ravel(), czt_amplitude.ravel()])
    )
    result.check_true(
        "both_rescaled_methods_return_the_input_shape",
        "reference",
        tuple(yu.spatial_shape) == SHAPE and tuple(czt.spatial_shape) == SHAPE,
        f"Yu et al. {tuple(yu.spatial_shape)}, CZT {tuple(czt.spatial_shape)}; upstream "
        "prints (1024, 1024) for both",
    )
    result.check_close(
        "the_yu_et_al_amplitude_norm_matches_upstream",
        "reference",
        yu_norm,
        UPSTREAM_YU_NORM,
        rel=0.05,
    )
    result.check_close(
        "the_czt_amplitude_norm_matches_upstream",
        "reference",
        czt_norm,
        UPSTREAM_CZT_NORM,
        rel=0.05,
    )
    result.check_true(
        "upstream_itself_records_the_amplitude_scale_disagreement",
        "reference",
        czt_norm / yu_norm > 5.0,
        f"the CZT amplitude norm is {czt_norm:.4f} against the Yu et al. {yu_norm:.4f}, a "
        f"factor of {czt_norm / yu_norm:.2f}. Upstream prints {UPSTREAM_CZT_NORM} and "
        f"{UPSTREAM_YU_NORM}, a factor of {UPSTREAM_CZT_NORM / UPSTREAM_YU_NORM:.2f}, so "
        "this is upstream-known behaviour and not a pinned-environment regression. It is "
        "documented nowhere except in this cell's printed output, and it is the same "
        "phenomenon c06_off_axis_propagation measured at a different zoom.",
    )
    result.check_close(
        "the_normalised_rmse_between_the_two_methods_matches_upstream",
        "reference",
        normalised_rmse,
        UPSTREAM_NORMALISED_RMSE,
        rel=1.0,
    )
    result.check_true(
        "the_two_methods_agree_after_normalisation_as_upstream_asserts",
        "reference",
        allclose and normalised_rmse < 1e-4,
        f"np.allclose(normalised, atol=1e-4) is {allclose} with a normalised RMSE of "
        f"{normalised_rmse:.6e} (upstream prints {UPSTREAM_NORMALISED_RMSE:.6e}). The "
        "structures agree; only the scales do not.",
    )

    # -- do they reproduce the brute-force reference? ---------------------------
    # The rescaled window covers `SHAPE` pixels at `DXI / ZOOM` centred on
    # `shift_yx`, which is a `SHAPE`-sized crop of the 4x-oversampled grid.
    centre = hi_res_shape[0] // 2
    offset_x = int(round(shift_yx[1] / (DXI / ZOOM)))
    offset_y = int(round(shift_yx[0] / (DXI / ZOOM)))
    half = SHAPE[0] // 2
    crop = hi_res_amplitude[
        centre + offset_y - half : centre + offset_y + half,
        centre + offset_x - half : centre + offset_x + half,
    ]
    crop_normalised = crop / float(np.linalg.norm(crop))
    reference_agreement = {}
    for name, normalised in (("yu", _normalised(yu)), ("czt", _normalised(czt))):
        rmse = float(np.sqrt(np.mean((normalised - crop_normalised) ** 2)))
        correlation = float(np.corrcoef(normalised.ravel(), crop_normalised.ravel())[0, 1])
        reference_agreement[name] = {"normalised_rmse": rmse, "correlation": correlation}
    result.record(
        brute_force_crop_shape=list(crop.shape),
        brute_force_crop_offsets=[offset_y, offset_x],
        agreement_with_brute_force=reference_agreement,
    )
    result.check_true(
        "both_rescaled_methods_reproduce_the_brute_force_oversampled_reference",
        "analytic",
        all(entry["correlation"] > 0.99 for entry in reference_agreement.values()),
        "against the corresponding crop of the 4096x4096 BLAS propagation: "
        + ", ".join(
            f"{k} r={v['correlation']:.6f} (normalised RMSE {v['normalised_rmse']:.3e})"
            for k, v in reference_agreement.items()
        )
        + ". Upstream only checks the two rescaled methods against EACH OTHER, which "
        "cannot distinguish two identically-wrong implementations. This check compares "
        "both against an independent brute-force computation.",
    )

    # -- the same rescaling on transfer_propagate ------------------------------
    fresnel = cf.transfer_propagate(field, Z, N, pad_width=PAD_WIDTH, mode="same")
    fresnel_scaled = cf.transfer_propagate(
        field, Z, N, pad_width=PAD_WIDTH, mode="same",
        output_dx=field.dx / ZOOM, shift_yx=shift_yx,
    )
    result.record(
        fresnel_spatial_shape=list(fresnel.spatial_shape),
        fresnel_scaled_spatial_shape=list(fresnel_scaled.spatial_shape),
        fresnel_scaled_dx=np.asarray(fresnel_scaled.dx, dtype=float),
        fresnel_scaled_norm=float(
            np.linalg.norm(np.asarray(fresnel_scaled.amplitude, dtype=float))
        ),
    )
    result.check_finite(
        "fresnel_rescaled_output_finite", np.asarray(fresnel_scaled.amplitude, dtype=float)
    )
    result.check_true(
        "output_dx_and_shift_yx_are_not_asm_specific",
        "invariant",
        tuple(fresnel_scaled.spatial_shape) == SHAPE
        and abs(
            float(np.asarray(fresnel_scaled.dx, dtype=float).ravel()[-1]) - DXI / ZOOM
        )
        < 1e-6,
        f"transfer_propagate accepts the same output_dx/shift_yx and returns "
        f"{tuple(fresnel_scaled.spatial_shape)} at dx "
        f"{float(np.asarray(fresnel_scaled.dx, dtype=float).ravel()[-1]):.6f} == "
        f"{DXI / ZOOM:.6f}",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
