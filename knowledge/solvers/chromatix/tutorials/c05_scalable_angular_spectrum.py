"""Example 5 / "Scalable Angular Spectrum" -- https://chromatix.readthedocs.io/en/latest/examples/sas/

Repo-owned reproduction of the SAS example: a square aperture illuminated by a
plane wave tilted 20 degrees, propagated a long distance three different ways --
`propagation.transform_propagate_sas` (scalable angular spectrum),
`transform_propagate` (single-FFT Fresnel) and `asm_propagate` (angular spectrum
with heavy padding).

Upstream is purely visual (four `imshow` calls, no printed numbers). The physics
of a tilted beam over a known distance is exact, though, and that is what turns
this into three independent cross-checks of the same answer:

* **Analytic displacement.** A beam tilted by ``theta`` travels ``z * tan(theta)``
  laterally over ``z``. At ``theta = 20 deg``, ``z = 1024`` that is ``372.71``.
  SAS and ASM both put the intensity peak at ``372.00`` and their 90%-energy
  centroid at ``373.6`` / ``373.2`` -- agreement to 0.3%, from two completely
  different discretisations.
* **`transform_propagate` puts it in the wrong place**, and by a diagnosable
  amount: ``350.00``, which is ``z * sin(theta) = 350.16`` to 0.05%. The
  single-FFT Fresnel propagator's output coordinate is the *direction-cosine*
  (Fourier) mapping ``x' = lambda z f_x``, i.e. it reports ``z * sin(theta)``
  rather than the geometric ``z * tan(theta)``. Correct in the paraxial limit; a
  **6.1% position error at 20 degrees**. That is not stated anywhere upstream and
  is exactly the kind of thing a coupler must know before chaining propagators.
* **SAS is the one that scales the output window**, by a factor of *exactly* the
  requested ``M_box = 8`` -- the whole point of the method, and not documented
  numerically upstream. ASM instead preserves the pitch and pays with a
  ``4608 x 4608`` grid.
* The **full-window** intensity centroid is biased low for all three (366 / 344 /
  351) because diffracted energy has left the window; it is recorded but not
  compared against the analytic value, since `Field.power` shows 4%, 3% and 0.1%
  of the energy already missing. The peak and the 90%-energy centroid are the
  robust measures and are what the checks use.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c05_scalable_angular_spectrum",
    title="Scalable Angular Spectrum",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/sas/",
    demonstrates=(
        "chromatix.functional.propagation.transform_propagate_sas versus "
        "transform_propagate and asm_propagate on the same tilted-aperture "
        "problem, plane_wave(kykx=...) for a tilted illumination, and the "
        "output-pitch rescaling each method applies."
    ),
    slow=True,
)

N_BOX = 512
L_BOX = 128.0
WAVELENGTH = 0.5
TILT_DEG = 20.0
M_BOX = 8
ASM_PAD = 2048


def _setup():
    import jax.numpy as jnp

    import chromatix.functional as cx

    shape = (N_BOX, N_BOX)
    kx = 2 * jnp.pi / WAVELENGTH * np.sin(TILT_DEG / 360 * 2 * np.pi)
    field = (
        cx.plane_wave(shape=shape, dx=L_BOX / N_BOX, spectrum=WAVELENGTH, kykx=[0, kx])
        / 0.0078125
    )
    aperture_width = L_BOX / 16
    field = cx.square_pupil(field, w=aperture_width)
    z = M_BOX / N_BOX / WAVELENGTH * L_BOX**2 * 2
    return field, z, aperture_width


def _profile(field) -> tuple[np.ndarray, np.ndarray]:
    """(x coordinate, x-marginal intensity) in the field's own length unit.

    The coordinate rule is Chromatix's own: index ``n // 2`` is zero, i.e.
    ``(arange(n) - n // 2) * dx`` (conventions.md, CHE-14).
    """
    intensity = np.asarray(field.intensity, dtype=float).squeeze()
    n = intensity.shape[-1]
    dx = float(np.asarray(field.dx, dtype=float).ravel()[-1])
    return (np.arange(n) - n // 2) * dx, intensity.sum(axis=0)


def _displacement_measures(field) -> dict[str, float]:
    """Peak, 90%-energy centroid and full centroid of the x-marginal, in length units.

    The full-window centroid is biased toward the window centre once energy has
    left the window, so the peak and the 90%-energy core are what the checks use.
    """
    x, weights = _profile(field)
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order])
    core = order[cumulative <= 0.9 * weights.sum()]
    return {
        "peak_x": float(x[int(np.argmax(weights))]),
        "core90_centroid_x": float((x[core] * weights[core]).sum() / weights[core].sum()),
        "full_centroid_x": float((x * weights).sum() / weights.sum()),
        "window_half_width": float(x.max()),
    }


def run() -> TutorialResult:
    import jax

    import chromatix.functional as cx

    result = TutorialResult()
    field, z, aperture_width = _setup()
    input_power = float(np.asarray(field.power, dtype=float).ravel()[0])
    input_dx = float(np.asarray(field.dx, dtype=float).ravel()[-1])
    predicted_shift = z * float(np.tan(np.deg2rad(TILT_DEG)))
    paraxial_shift = z * float(np.sin(np.deg2rad(TILT_DEG)))

    result.record(
        n_box=N_BOX,
        l_box=L_BOX,
        wavelength=WAVELENGTH,
        tilt_deg=TILT_DEG,
        magnification=M_BOX,
        aperture_width=aperture_width,
        propagation_distance=z,
        input_dx=input_dx,
        input_power=input_power,
        input_displacement=_displacement_measures(field),
        predicted_geometric_shift=predicted_shift,
        predicted_paraxial_shift=paraxial_shift,
    )
    result.check_close(
        "the_input_beam_starts_centred",
        "analytic",
        _displacement_measures(field)["core90_centroid_x"],
        0.0,
        abs_=input_dx,
    )

    fields = {
        "sas": jax.jit(cx.propagation.transform_propagate_sas)(field, z=z, n=1.0),
        "fresnel": jax.jit(cx.propagation.transform_propagate, static_argnums=3)(
            field, z=z, n=1.0, pad_width=256
        ),
        "asm": jax.jit(cx.propagation.asm_propagate, static_argnames=("pad_width", "mode"))(
            field, z=z, n=1.0, pad_width=ASM_PAD, mode="full"
        ),
    }
    outcomes = {
        name: {
            "shape": list(np.asarray(out.u).shape),
            "dx": float(np.asarray(out.dx, dtype=float).ravel()[-1]),
            "power": float(np.asarray(out.power, dtype=float).ravel()[0]),
            **_displacement_measures(out),
        }
        for name, out in fields.items()
    }
    result.record(propagators=outcomes)
    for name, out in fields.items():
        result.check_finite(f"{name}_output_finite", np.abs(np.asarray(out.u)))

    # -- SAS and ASM place the beam geometrically ------------------------------
    for name in ("sas", "asm"):
        result.check_close(
            f"{name}_peak_matches_z_tan_theta",
            "analytic",
            outcomes[name]["peak_x"],
            predicted_shift,
            rel=0.005,
        )
        result.check_close(
            f"{name}_core_centroid_matches_z_tan_theta",
            "analytic",
            outcomes[name]["core90_centroid_x"],
            predicted_shift,
            rel=0.005,
        )
    result.check_close(
        "sas_and_asm_agree_with_each_other_on_the_displacement",
        "analytic",
        outcomes["sas"]["core90_centroid_x"],
        outcomes["asm"]["core90_centroid_x"],
        rel=0.005,
    )

    # -- transform_propagate reports the paraxial position --------------------
    result.check_close(
        "fresnel_peak_matches_z_sin_theta_not_z_tan_theta",
        "analytic",
        outcomes["fresnel"]["peak_x"],
        paraxial_shift,
        rel=0.005,
    )
    fresnel_error = abs(outcomes["fresnel"]["peak_x"] - predicted_shift) / predicted_shift
    result.record(fresnel_geometric_position_error_fraction=fresnel_error)
    result.check_true(
        "the_single_fft_fresnel_propagator_mis_places_a_tilted_beam",
        "analytic",
        fresnel_error > 0.05,
        f"transform_propagate puts the peak at {outcomes['fresnel']['peak_x']:.3f}, which is "
        f"z*sin(theta) = {paraxial_shift:.3f} to 0.05%, not the geometric z*tan(theta) = "
        f"{predicted_shift:.3f}: a {fresnel_error * 100:.1f}% position error at "
        f"{TILT_DEG:.0f} degrees. Its output coordinate is the direction-cosine (Fourier) "
        "mapping x' = lambda*z*f_x, correct only in the paraxial limit. SAS and ASM both "
        "give the geometric position. A coupler chaining propagators must not mix them.",
    )

    # -- window scaling --------------------------------------------------------
    result.check_close(
        "sas_rescales_its_output_pitch_by_exactly_the_requested_magnification",
        "analytic",
        outcomes["sas"]["dx"] / input_dx,
        float(M_BOX),
        rel=1e-6,
    )
    result.check_true(
        "asm_preserves_its_sample_pitch_and_pads_instead",
        "invariant",
        abs(outcomes["asm"]["dx"] - input_dx) < 1e-6
        and outcomes["asm"]["shape"][0] > N_BOX,
        f"ASM dx {outcomes['asm']['dx']:.6f} == input {input_dx:.6f}, shape "
        f"{outcomes['asm']['shape']} vs input ({N_BOX}, {N_BOX}) -- the padded grid, as "
        f"conventions.md records. SAS instead reaches the same distance on a "
        f"{outcomes['sas']['shape']} grid with dx scaled by {M_BOX}.",
    )
    result.check_close(
        "asm_conserves_discrete_power",
        "analytic",
        outcomes["asm"]["power"] / input_power,
        1.0,
        rel=2e-3,
    )
    result.check_true(
        "the_scaling_methods_lose_a_few_percent_to_their_own_windowing",
        "analytic",
        all(
            0.5 < outcomes[name]["power"] / input_power <= 1.0 + 1e-6
            for name in ("sas", "fresnel")
        ),
        "power ratios: SAS "
        f"{outcomes['sas']['power'] / input_power:.6f}, Fresnel "
        f"{outcomes['fresnel']['power'] / input_power:.6f}, ASM "
        f"{outcomes['asm']['power'] / input_power:.6f}. Field.power is a sum over the "
        "sampled window only (conventions.md), so this is a truncation diagnostic and not "
        "an energy violation -- and it is why the FULL-window centroids "
        + ", ".join(f"{k} {v['full_centroid_x']:.1f}" for k, v in outcomes.items())
        + f" all sit below the analytic {predicted_shift:.1f} while the peaks do not.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
