"""CHE-37 (M3.8): verify the M3 slice against independent oracles.

Runs the SHIPPING path -- Optiland trace, ``C_RAY_TO_WAVE``, Chromatix ASM, PSF
measurement -- and compares it against two oracles, then perturbs it and reports
how far each perturbation moves relative to the oracle residual it has to beat.

Writes ``benchmarks/probes/records/m3_psf_verification.json``.

Configurations
--------------
``diffraction_limited``
    ``M3SingletRef`` on axis at the frozen grid. The analytic Airy vehicle.

``defocused``
    The same system observed away from best focus. This is M3.8's ABERRATED case,
    and it is not the one CHE-37 originally specified. ``ReverseTelephoto``
    off-axis was, and it cannot serve: this probe's companion measurements found
    it diffraction limited at *every* field point (P-V <= 0.078 waves against the
    0.25-wave Rayleigh limit, RMS <= 0.021 waves, geometric spot <= 0.18 Airy
    radii), so no field point of it produces a non-Airy PSF. Two further blockers
    stand independently: the frozen pitch is inadmissible beyond Hy ~ 0.25 because
    the chief-ray tilt raises the largest direction cosine, and the PSF lands
    0.42-1.13 mm off axis against a 0.23 mm window half-width. Defocus is used
    instead because it is a real, analytically exact aberration that needs no new
    prescription, no new handoff convention and no larger grid.

``off_axis``
    ``ReverseTelephoto`` at ``Hy = 0.2``: the asymmetry vehicle, not the aberrated
    one. Its purpose is the blind-spot audit. A circularly symmetric PSF centred
    on the axis cannot detect an x/y transpose, and both other configurations are
    exactly that. This one puts the PSF off centre along +y only, at a field where
    the frozen grid is still admissible (94.6% of the Nyquist pitch) and the PSF
    is still inside the observation window (209 um of 232 um).

    As M3.8 ran, this vehicle failed at its own purpose: the declared OPL carried
    none of the field's tilt, so its PSF formed on axis like the other two. CHE-41
    fixed the handoff (slice_protocol amendment A4), and the PSF now lands 114
    pixels off axis in y. The ``off_axis_handoff`` block below re-measures that
    through this probe's own path and retains M3.8's numbers as
    ``superseded_finding``. The transpose control still reads a margin of ~1.0 --
    for a second reason, documented there and owned by CHE-41.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmarks" / "slice_protocol.yaml"
RECORD_PATH = Path(__file__).resolve().parent / "records" / "m3_psf_verification.json"

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1e-6

#: Ray count. The frozen protocol criterion asks for one ray per Nyquist cell and
#: names 4096 as a starting point; CHE-38 owns the convergence study. Optiland's
#: num_rays is a hexapolar DENSITY, and 32 yields 3169 survivors.
NUM_RAYS = 32

SINGLET = {
    "sample": "M3SingletRef",
    "pupil_z_m": 0.06814345991561233e-3,
    "image_z_m": 4.90560476022521e-3,
    "pitch_m": 2.6587352810843895e-06,
    "grid_n": 188,
    "pad_width": 566,
    "na_frozen": 0.05171631827291936,
    "hy": 0.0,
}
REVERSE_TELEPHOTO = {
    "sample": "ReverseTelephoto",
    "pupil_z_m": 2.1547825721481666e-3,
    "image_z_m": 5.209361469999999e-3,
    "pitch_m": 1.8258157981959995e-06,
    "grid_n": 254,
    "pad_width": 762,
    "na_frozen": 0.07530880176185195,
    "hy": 0.2,
}

#: Axial shift for the aberrated case, in metres. Defocus wavefront error scales
#: as ``delta * NA^2 / (4 sqrt 3 lambda)`` in RMS waves. 120 um puts M3SingletRef
#: at ~0.092 waves RMS and ~0.31 waves peak-to-valley: past the 0.25-wave Rayleigh
#: quarter-wave criterion, so unambiguously not diffraction limited, while staying
#: just inside the Maréchal approximation's ~0.1-wave validity regime. A more
#: strongly aberrated case would leave that regime and make the Strehl cross-check
#: meaningless, which is the opposite of what M3.8 wants from it.
DEFOCUS_M = 120.0e-6


def _protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def _trace(geometry: dict[str, Any], directory: Path, *, hy: float | None = None):
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    return (
        get_adapter()
        .run(
            ModelRunRequest(
                run_id="che37",
                node_id="lens",
                config={
                    "sample": geometry["sample"],
                    "num_rays": NUM_RAYS,
                    "wavelength": WAVELENGTH_UM,
                    "Hx": 0.0,
                    "Hy": geometry["hy"] if hy is None else hy,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(directory),
                },
            )
        )
        .outputs["rays"]
    )


def _reconstruct(rays, geometry: dict[str, Any], directory: Path, **overrides: Any):
    from couplers.base import CouplerRunRequest
    from couplers.node import RayToWaveCoupler

    config = {
        "handoff_plane": "exit_pupil",
        "handoff_plane_z_m": geometry["pupil_z_m"],
        "grid_n": geometry["grid_n"],
        "target_sample_pitch_m": geometry["pitch_m"],
        "output_dir": str(directory),
    }
    config.update(overrides)
    return RayToWaveCoupler().transform(
        CouplerRunRequest(run_id="che37", edge_id="pupil", source=rays, config=config)
    )


def _propagate(field_record, geometry: dict[str, Any], directory: Path, **overrides: Any):
    from solvers.base import ModelRunRequest
    from solvers.chromatix.adapter import get_adapter

    config = {
        "propagation": "angular_spectrum",
        "propagation_method": "asm_carrier_removed",
        "target_plane_z_m": geometry["image_z_m"],
        "pad_width": geometry["pad_width"],
        "output_dir": str(directory),
    }
    config.update(overrides)
    return get_adapter().run(
        ModelRunRequest(
            run_id="che37",
            node_id="wave",
            inputs={"input_field": field_record},
            config=config,
        )
    )


def _shipping_psf(
    geometry: dict[str, Any],
    directory: Path,
    *,
    target_plane_z_m: float | None = None,
    coupler_overrides: dict[str, Any] | None = None,
    wave_overrides: dict[str, Any] | None = None,
    rays=None,
) -> dict[str, Any]:
    """One full pass of the shipping path, ending in a measured PSF."""
    from verification.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf_from_record,
    )

    rays = rays if rays is not None else _trace(geometry, directory / "rays")
    coupled = _reconstruct(rays, geometry, directory / "field", **(coupler_overrides or {}))
    if coupled.status.value != "succeeded":
        return {"status": "coupler_failed", "error": coupled.error_message}

    wave_config: dict[str, Any] = dict(wave_overrides or {})
    if target_plane_z_m is not None:
        wave_config["target_plane_z_m"] = target_plane_z_m
    result = _propagate(coupled.target, geometry, directory / "wave", **wave_config)
    if result.status.value != "succeeded":
        return {"status": "propagation_failed", "error": result.error_message}

    reported_pitch = result.diagnostics["output_sample_pitch_m"]
    measurement = measure_psf_from_record(
        result.outputs["output_field"],
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported_pitch[0]), float(reported_pitch[1])),
    )
    return {
        "status": "succeeded",
        "rays": rays,
        "pupil_field": coupled.target,
        "coupler_diagnostics": coupled.diagnostics,
        "wave_result": result,
        "measurement": measurement,
    }


def _aberration(rays, geometry: dict[str, Any], *, observation_z_m: float):
    """Wavefront error at the pupil, twice, because the two answer different things.

    ``at_observation_plane`` holds the reference sphere at the plane the slice
    actually propagates to. That is what the oracle must use: it is the aberration
    that forms the PSF *there*, and it includes any defocus between the declared
    plane and best focus.

    ``at_best_focus`` lets the sphere centre move. That characterizes the system's
    intrinsic aberration, and it is the number comparable to Optiland's own
    ``Wavefront``.

    Keeping them apart matters. Fitting the sphere for the deliberately defocused
    configuration optimized the defocus away and reported the aberrated case as
    having the same 0.0013 waves RMS as the in-focus one -- the fit is free to move
    the centre to exactly where the defocus put the focus.
    """
    from couplers.handoff import (
        DeclaredHandoffPlane,
        declare_coherent_bundle,
    )
    from verification.psf_oracles import pupil_aberration

    bundle = declare_coherent_bundle(
        rays, declared_plane=DeclaredHandoffPlane("exit_pupil", geometry["pupil_z_m"])
    ).bundle
    step = observation_z_m - geometry["pupil_z_m"]
    directions = np.asarray(bundle.directions)
    positions = np.asarray(bundle.positions_m)
    x_img = positions[:, 0] + directions[:, 0] * step / directions[:, 2]
    y_img = positions[:, 1] + directions[:, 1] * step / directions[:, 2]
    observation = (float(np.mean(x_img)), float(np.mean(y_img)), observation_z_m)
    at_plane = pupil_aberration(
        bundle, plane_z_m=geometry["pupil_z_m"], observation_point_m=observation, fit_sphere=False
    )
    at_focus = pupil_aberration(
        bundle, plane_z_m=geometry["pupil_z_m"], observation_point_m=observation, fit_sphere=True
    )
    return bundle, at_plane, at_focus


def _oracle_psf(aberration, geometry: dict[str, Any], *, distance_m: float, factor: int = 16):
    from verification.psf_oracles import fraunhofer_psf

    return fraunhofer_psf(
        aberration,
        pupil_pitch_m=geometry["pitch_m"],
        pupil_grid_n=geometry["grid_n"],
        fft_grid_n=factor * geometry["grid_n"],
        distance_m=distance_m,
    )


def _profile_residual(
    measured: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
    *,
    measured_pitch: tuple[float, float],
    reference_pitch: tuple[float, float],
    max_radius_m: float,
    measured_center_m: tuple[float, float] = (0.0, 0.0),
    reference_center_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Compare two PSFs sampled differently, on a common radial grid.

    Both are peak-normalized first, which is the frozen M3 oracle normalization,
    and both are azimuthally averaged about a STATED centre. An azimuthal average
    is only meaningful for a pattern that is rotationally symmetric *about that
    centre*, which is why the centre is a parameter rather than the grid origin:
    since CHE-41 the off-axis PSF forms 114 pixels off axis, and averaging it about
    the origin turns an Airy pattern into a smeared annulus whose profile is nearly
    invariant under anything a control can do to it. That is measurable: with the
    centre left at the origin, all six off-axis controls score a margin within 2.6x
    of 1.0, including an x/y transpose that visibly moves the peak.
    """
    from verification.psf_oracles import azimuthal_profile

    radii_m, profile_m = azimuthal_profile(
        measured / float(np.max(measured)),
        sample_pitch_m=measured_pitch,
        center_m=measured_center_m,
        max_radius_m=max_radius_m,
        radial_samples=400,
        azimuthal_samples=256,
    )
    radii_r, profile_r = azimuthal_profile(
        reference / float(np.max(reference)),
        sample_pitch_m=reference_pitch,
        center_m=reference_center_m,
        max_radius_m=max_radius_m,
        radial_samples=400,
        azimuthal_samples=256,
    )
    common = np.interp(radii_m, radii_r, profile_r)
    difference = profile_m - common
    denominator = float(np.linalg.norm(common))
    return {
        "max_abs_profile_residual": float(np.max(np.abs(difference))),
        "rms_profile_residual": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_profile_residual": (
            float(np.linalg.norm(difference) / denominator) if denominator else None
        ),
        "radial_samples": int(radii_m.size),
        "max_radius_m": float(max_radius_m),
    }


def _oracle_vs_shipping(
    measurement,
    oracle,
    *,
    max_radius_m: float,
) -> dict[str, Any]:
    """Compare the FFT oracle to the shipping PSF ON THE SHIPPING GRID.

    The two are sampled on different pitches by construction, and comparing their
    azimuthal profiles directly measures that difference: on the
    diffraction-limited case it read as a 13.5% disagreement while the oracle's own
    first null was correct to 0.14%. The oracle is finely sampled, so it is
    point-sampled at the shipping grid's pixel centres instead.
    """
    from verification.psf_oracles import resample_to_grid

    psf = measurement.intensity
    resampled = resample_to_grid(
        oracle.intensity,
        from_pitch_m=oracle.sample_pitch_m,
        to_pitch_m=measurement.sample_pitch_m,
        to_shape=psf.shape,
    )
    measured = psf / float(np.max(psf))
    reference = resampled / float(np.max(resampled))

    ny, nx = psf.shape
    dy, dx = measurement.sample_pitch_m
    yy = (np.arange(ny) - ny // 2) * dy
    xx = (np.arange(nx) - nx // 2) * dx
    inside = np.hypot(yy[:, None], xx[None, :]) <= max_radius_m
    difference = (measured - reference)[inside]
    denominator = float(np.linalg.norm(reference[inside]))
    return {
        "comparison_grid": "the shipping PSF's own grid; the oracle is point-sampled onto it",
        "compared_pixels": int(inside.sum()),
        "max_radius_m": float(max_radius_m),
        "max_abs_residual": float(np.max(np.abs(difference))),
        "rms_residual": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_residual": float(np.linalg.norm(difference) / denominator)
        if denominator
        else None,
    }


def _energy_ledger(passed: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    """traced ray power -> pupil power -> propagated power -> PSF integral."""
    from core.boundary import ComplexField

    rays = passed["rays"]
    arrays = dict(np.load(rays.uri))
    survived = arrays.get("survived")
    keep = np.ones(arrays["intensity"].shape, bool) if survived is None else survived.astype(bool)
    traced_power = float(np.sum(arrays["intensity"][keep]))

    pupil = ComplexField.from_artifact_record(passed["pupil_field"])
    pupil_power = pupil.discrete_power()

    diagnostics = passed["wave_result"].diagnostics
    measurement = passed["measurement"]

    return {
        "traced_ray_power": traced_power,
        "traced_ray_power_units": (
            "sum of Optiland per-ray intensity weights over survivors. A COUNT-like "
            "weight, not watts, and not an area integral"
        ),
        "reconstructed_pupil_power": pupil_power,
        "propagated_power_in": diagnostics["power_in"],
        "propagated_power_out": diagnostics["power_out"],
        "measured_psf_integral": measurement.raw_window_energy,
        "ratios": {
            "pupil_over_traced": pupil_power / traced_power if traced_power else None,
            "propagated_in_over_pupil": (
                diagnostics["power_in"] / pupil_power if pupil_power else None
            ),
            "propagated_out_over_in": diagnostics["power_conservation_ratio"],
            "psf_integral_over_propagated_out": (
                measurement.raw_window_energy / diagnostics["power_out"]
                if diagnostics["power_out"]
                else None
            ),
        },
        "attribution": {
            "pupil_over_traced": (
                "NOT a conservation law and not expected to be 1. The two sides "
                "measure different things: a sum of dimensionless ray weights "
                "versus sum(|u|^2)*dy*dx over a grid. The bundle declares "
                "reconstruction_normalization='none', so this ratio is the "
                "measure conversion of the wavelet sum, reported rather than "
                "compared to unity."
            ),
            "propagated_in_over_pupil": (
                "the complex64 cast at the Chromatix boundary plus Chromatix's own "
                "power bookkeeping. Bounded by the protocol's "
                "chromatix_complex64_truncation term."
            ),
            "propagated_out_over_in": (
                "window escape and padding. The ASM transfer function is "
                "unit-modulus wherever k_z is real and the grid carries no "
                "evanescent bins, so power over the INFINITE plane is conserved "
                "exactly; any deficit here is power that left the sampled window. "
                "On an UNPADDED run this reads 1.0 because wraparound recirculates "
                "it, which is why 1.0 is not evidence of correctness (CHE-35)."
            ),
            "psf_integral_over_propagated_out": (
                "must be 1 to float rounding: the measurement is |u|^2 on the same "
                "grid with the same pitch. A deviation is a measurement-layer "
                "defect, not physics."
            ),
        },
        "padding": {
            "pad_width": diagnostics.get("pad_width", geometry["pad_width"]),
            "output_edge_energy_fraction": diagnostics["output_edge_energy_fraction"],
            "input_edge_energy_fraction": diagnostics["input_edge_energy_fraction"],
            "edge_energy_is_a_weak_indicator": diagnostics[
                "edge_energy_is_a_weak_wraparound_indicator"
            ],
        },
        "psf_border_energy_fraction": measurement.border_energy_fraction,
    }


def _strehl(diffraction_limited: dict[str, Any], aberrated: dict[str, Any]) -> dict[str, Any]:
    """Measured Strehl from the retained RAW scale, against Maréchal.

    Strehl is peak intensity relative to the unaberrated peak *at equal total
    energy*, so it cannot be read off a peak-normalized PSF -- both are 1 by
    construction. This is what M3.7 kept ``raw_peak_intensity`` and
    ``raw_window_energy`` for.
    """
    dl = diffraction_limited["measurement"]
    ab = aberrated["measurement"]
    dl_ratio = dl.raw_peak_intensity / dl.raw_window_energy
    ab_ratio = ab.raw_peak_intensity / ab.raw_window_energy
    measured = ab_ratio / dl_ratio

    aberration = aberrated["aberration"]
    marechal = aberration.marechal_strehl
    return {
        "measured_strehl": measured,
        "measured_strehl_definition": (
            "(peak/energy)_aberrated / (peak/energy)_diffraction_limited, from the "
            "raw scale retained by the measurement. Energy normalization makes it "
            "independent of the uncalibrated ray-weight scale."
        ),
        "marechal_strehl": marechal,
        "rms_waves": aberration.rms_waves,
        "ratio_measured_over_marechal": measured / marechal if marechal else None,
        "marechal_valid_regime": aberration.marechal_is_in_regime,
        "validity": (
            "exp(-(2 pi sigma)^2) is a small-aberration expansion, conventionally "
            "trusted to a few percent below ~0.1 waves RMS. This is a sanity "
            "cross-check, NOT a tolerance gate: its purpose is to catch a gross "
            "error that the FFT oracle cannot, because that oracle shares the "
            "traced OPD map with the shipping path."
        ),
    }


def _perturbed_psf(
    rays,
    geometry: dict[str, Any],
    directory: Path,
    *,
    handoff_perturbation=None,
    core_perturbation=None,
    unit_amplitude: bool = False,
    target_plane_z_m: float | None = None,
) -> dict[str, Any]:
    """One pass with a single term perturbed, through the shipping functions.

    Not through ``RayToWaveCoupler.transform``: the graph node exposes no
    perturbation in its config, and adding one to ship a test would be the wrong
    trade. It calls ``declare_coherent_bundle`` and ``ray_to_wave``, which is what
    this function calls, and CHE-34 pinned the node's output as **bit-identical**
    to the direct call -- so perturbing here perturbs the code that ships. The
    unperturbed run below re-establishes that bit-identity at the PSF level rather
    than relying on the earlier claim.
    """
    from core.boundary import RayBundle
    from couplers.handoff import (
        DeclaredHandoffPlane,
        HandoffPerturbation,
        declare_coherent_bundle,
    )
    from couplers.ray_to_wave import Perturbation, ray_to_wave
    from verification.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf_from_record,
    )

    directory.mkdir(parents=True, exist_ok=True)
    bundle = declare_coherent_bundle(
        rays,
        declared_plane=DeclaredHandoffPlane("exit_pupil", geometry["pupil_z_m"]),
        perturbation=handoff_perturbation or HandoffPerturbation(),
    ).bundle

    if unit_amplitude:
        bundle = RayBundle(
            positions_m=bundle.positions_m,
            directions=bundle.directions,
            wavelength_m=bundle.wavelength_m,
            reference_plane=bundle.reference_plane,
            frame=bundle.frame,
            amplitude=np.ones_like(np.asarray(bundle.amplitude)),
            optical_path_length_m=bundle.optical_path_length_m,
            optical_path_length_reference=bundle.optical_path_length_reference,
            reconstruction_normalization=bundle.reconstruction_normalization,
            provenance=dict(bundle.provenance),
        )

    try:
        field, _ = ray_to_wave(
            bundle,
            grid_shape=(geometry["grid_n"], geometry["grid_n"]),
            sample_pitch_m=(geometry["pitch_m"], geometry["pitch_m"]),
            perturbation=core_perturbation or Perturbation(),
        )
    except Exception as exc:  # a refusal is a detection
        return {"status": "reconstruction_refused", "error": f"{type(exc).__name__}: {exc}"}

    record = field.to_artifact_record(
        artifact_id="pupil:perturbed", uri=directory / "pupil_field.npy"
    )
    record.metadata["z_m"] = geometry["pupil_z_m"]
    record.metadata["reference_plane"] = "exit_pupil"

    overrides: dict[str, Any] = {}
    if target_plane_z_m is not None:
        overrides["target_plane_z_m"] = target_plane_z_m
    try:
        result = _propagate(record, geometry, directory / "wave", **overrides)
    except Exception as exc:
        return {"status": "propagation_refused", "error": f"{type(exc).__name__}: {exc}"}
    if result.status.value != "succeeded":
        return {"status": "propagation_failed", "error": result.error_message}

    reported = result.diagnostics["output_sample_pitch_m"]
    measurement = measure_psf_from_record(
        result.outputs["output_field"],
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
    )
    return {"status": "succeeded", "measurement": measurement, "wave_result": result}


def _negative_controls(
    rays,
    geometry: dict[str, Any],
    directory: Path,
    *,
    baseline_psf: np.ndarray[Any, Any],
    max_radius_m: float,
    airy_na: float,
    psf_center_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Every perturbation through the shipping code, with a detection margin.

    The margin is what makes this more than "the result changed". Each perturbed
    PSF is scored by its residual against the analytic Airy pattern, divided by the
    residual the UNPERTURBED run already carries. A control whose margin is ~1 did
    not move the answer further than the oracle disagreement it has to beat, and is
    reported as undetected -- which is the M2 lesson: a control that cannot be made
    to fail validates nothing.

    ``psf_center_m`` is ``(y, x)`` and defaults to the grid origin, which is where
    the two on-axis configurations put their PSF. Added by CHE-41: once the
    off-axis handoff was fixed, the off-axis PSF moved 114 pixels off axis, and
    scoring it against an Airy pattern centred on the ORIGIN made all six controls
    read a margin within 2.6x of 1.0 -- the oracle was then measuring the offset
    rather than the perturbation. Placing both the reference and the azimuthal
    average at the traced image point restores the meaning of the margin. The
    default keeps the on-axis block bit-identical.
    """
    from couplers.handoff import HandoffPerturbation
    from couplers.ray_to_wave import Perturbation
    from verification.psf_oracles import airy_psf_on_grid

    def _score(passed: dict[str, Any]) -> dict[str, Any] | None:
        if passed.get("status") != "succeeded":
            return None
        measurement = passed["measurement"]
        psf = measurement.intensity
        analytic = airy_psf_on_grid(
            shape=psf.shape,
            sample_pitch_m=measurement.sample_pitch_m,
            wavelength_m=WAVELENGTH_M,
            numerical_aperture=airy_na,
            center_m=psf_center_m,
        )
        residual = _profile_residual(
            psf,
            analytic,
            measured_center_m=psf_center_m,
            reference_center_m=psf_center_m,
            measured_pitch=measurement.sample_pitch_m,
            reference_pitch=measurement.sample_pitch_m,
            max_radius_m=max_radius_m,
        )["relative_l2_profile_residual"]
        return {
            "relative_l2_vs_airy": residual,
            "peak_index": list(measurement.peak_index),
            "raw_peak_intensity": measurement.raw_peak_intensity,
            "raw_window_energy": measurement.raw_window_energy,
            "border_energy_fraction": measurement.border_energy_fraction,
            "max_abs_difference_vs_unperturbed": (
                float(np.max(np.abs(psf / psf.max() - baseline_psf / baseline_psf.max())))
                if psf.shape == baseline_psf.shape
                else None
            ),
        }

    control = _perturbed_psf(rays, geometry, directory / "control")
    control_score = _score(control)
    if control_score is None:
        return {"status": "unperturbed_control_failed", "detail": control.get("error")}
    reference = control_score["relative_l2_vs_airy"]

    perturbations = [
        (
            "opl_sign_flip",
            "HandoffPerturbation(opl_sign=-1): conjugates the wavefront, so the "
            "converging pupil field diverges instead.",
            {"handoff_perturbation": HandoffPerturbation(opl_sign=-1)},
            "",
        ),
        (
            "reconstruction_phase_sign_flip",
            "Perturbation(phase_sign=-1): the same conjugation, applied one stage "
            "later, inside the wavelet sum.",
            {"core_perturbation": Perturbation(phase_sign=-1)},
            "",
        ),
        (
            "axis_transpose",
            "Perturbation(transpose_axes=True): the output grid's axes swapped.",
            {"core_perturbation": Perturbation(transpose_axes=True)},
            "Expected INVISIBLE here: a circular pupil and an on-axis PSF are "
            "symmetric under transposition, and this entry exists to show the "
            "blind spot rather than to claim a pass. CHE-41 established that the "
            "off-axis configuration does not rescue it either, and why: this "
            "SCORE azimuthally averages about the grid centre, which cannot tell "
            "a peak at (114, 0) from one at (0, 114) for any configuration. See "
            "benchmarks/probes/records/m3_off_axis_handoff.json.",
        ),
        (
            "amplitude_weight_omitted",
            "Every ray's declared amplitude replaced by 1, through the RayBundle "
            "contract, leaving the OPL untouched.",
            {"unit_amplitude": True},
            "Expected WEAK: hexapolar weights on this configuration are near "
            "uniform, so the omission is close to an exact constant scale, and the "
            "frozen peak normalization cannot see a constant. The raw scale is what "
            "shows it -- compare raw_peak_intensity against the control.",
        ),
        (
            "oblique_ramp_omitted",
            "Perturbation(apply_oblique_ramp=False): off-axis rays deposit a piston "
            "instead of a ramp.",
            {"core_perturbation": Perturbation(apply_oblique_ramp=False)},
            "M2 found this inert for a single centred on-axis ray. Here the bundle "
            "fills the pupil, so it should bite.",
        ),
        (
            "propagation_distance_sign",
            "Propagate -z instead of +z: away from focus rather than to it.",
            {
                "target_plane_z_m": geometry["pupil_z_m"]
                - (geometry["image_z_m"] - geometry["pupil_z_m"])
            },
            "",
        ),
    ]

    results = []
    for name, description, kwargs, note in perturbations:
        passed = _perturbed_psf(rays, geometry, directory / name, **kwargs)
        score = _score(passed)
        if score is None:
            results.append(
                {
                    "control": name,
                    "description": description,
                    "outcome": "refused_before_a_psf_existed",
                    "detail": passed.get("error"),
                    "detected": True,
                    "detection_mechanism": "structured refusal",
                    "note": note,
                }
            )
            continue
        margin = score["relative_l2_vs_airy"] / reference if reference else None
        results.append(
            {
                "control": name,
                "description": description,
                "outcome": "ran",
                **score,
                "unperturbed_relative_l2_vs_airy": reference,
                "detection_margin": margin,
                "raw_peak_ratio_vs_control": (
                    score["raw_peak_intensity"] / control_score["raw_peak_intensity"]
                    if control_score["raw_peak_intensity"]
                    else None
                ),
                "detected": bool(margin is not None and margin > 10.0),
                "detection_mechanism": (
                    "analytic Airy profile residual, relative to the unperturbed "
                    "residual on the same configuration"
                ),
                "note": note,
            }
        )

    return {
        "status": "ran",
        "method": (
            "perturbations are applied through couplers.handoff."
            "HandoffPerturbation and couplers.ray_to_wave.Perturbation -- the hooks "
            "the shipping code already carries -- and the pupil field is built by "
            "the same ray_to_wave call the graph node wraps. CHE-34 pinned that "
            "node as bit-identical to this call."
        ),
        "detection_threshold": "detection_margin > 10x the unperturbed residual",
        "unperturbed_control": control_score,
        "controls": results,
    }


def _bundle_with(rays, geometry, *, opl_tilt_x: float = 0.0, unit_amplitude: bool = False):
    """The declared bundle, optionally with a known linear tilt added to its OPL."""
    from core.boundary import RayBundle
    from couplers.handoff import (
        DeclaredHandoffPlane,
        declare_coherent_bundle,
    )

    bundle = declare_coherent_bundle(
        rays, declared_plane=DeclaredHandoffPlane("exit_pupil", geometry["pupil_z_m"])
    ).bundle
    if opl_tilt_x == 0.0 and not unit_amplitude:
        return bundle
    positions = np.asarray(bundle.positions_m)
    opl = np.asarray(bundle.optical_path_length_m) + opl_tilt_x * positions[:, 0]
    amplitude = (
        np.ones_like(np.asarray(bundle.amplitude))
        if unit_amplitude
        else np.asarray(bundle.amplitude)
    )
    return RayBundle(
        positions_m=bundle.positions_m,
        directions=bundle.directions,
        wavelength_m=bundle.wavelength_m,
        reference_plane=bundle.reference_plane,
        frame=bundle.frame,
        amplitude=amplitude,
        optical_path_length_m=opl,
        optical_path_length_reference=bundle.optical_path_length_reference,
        reconstruction_normalization=bundle.reconstruction_normalization,
        provenance=dict(bundle.provenance),
    )


def _psf_from_bundle(bundle, geometry, directory: Path, *, core_perturbation=None):
    from couplers.ray_to_wave import Perturbation, ray_to_wave
    from verification.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf_from_record,
    )

    directory.mkdir(parents=True, exist_ok=True)
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(geometry["grid_n"], geometry["grid_n"]),
        sample_pitch_m=(geometry["pitch_m"], geometry["pitch_m"]),
        perturbation=core_perturbation or Perturbation(),
    )
    record = field.to_artifact_record(artifact_id="pupil:tilted", uri=directory / "pupil_field.npy")
    record.metadata["z_m"] = geometry["pupil_z_m"]
    record.metadata["reference_plane"] = "exit_pupil"
    result = _propagate(record, geometry, directory / "wave")
    if result.status.value != "succeeded":
        return None
    reported = result.diagnostics["output_sample_pitch_m"]
    return measure_psf_from_record(
        result.outputs["output_field"],
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
    )


def _orientation_control(rays, geometry: dict[str, Any], directory: Path) -> dict[str, Any]:
    """A transpose control that CAN fail, built from a synthetic known tilt.

    Neither frozen configuration can detect an x/y transpose. Both on-axis cases
    are rotationally symmetric, and as M3.8 ran, the off-axis case formed its PSF
    on axis too, so its symmetry was not broken either. A control that cannot be
    made to fail validates nothing -- M2's rule.

    CHE-41 fixed the off-axis handoff and the shipping PSF now lands 114 pixels off
    axis, so a real configuration can be made to fail. This synthetic control is
    kept rather than retired: it tests the x/y wiring against an ANALYTICALLY
    PREDICTED displacement, which the off-axis case does not, and M3.8's audit is a
    record of what was checked when.

    So the degree of freedom is excited directly: a known linear tilt is added to
    the declared OPL along x only. The PSF must move along +x by ``tilt * R``, an
    analytically predicted distance, and under ``transpose_axes=True`` the same
    displacement must appear along y instead. That tests the x/y wiring of the
    reconstruction with a case whose answer is known in advance.
    """
    from couplers.ray_to_wave import Perturbation

    tilt = 0.02
    distance = geometry["image_z_m"] - geometry["pupil_z_m"]
    predicted_shift_m = tilt * distance
    predicted_pixels = predicted_shift_m / geometry["pitch_m"]

    baseline = _psf_from_bundle(_bundle_with(rays, geometry), geometry, directory / "baseline")
    tilted = _psf_from_bundle(
        _bundle_with(rays, geometry, opl_tilt_x=tilt), geometry, directory / "tilted"
    )
    transposed = _psf_from_bundle(
        _bundle_with(rays, geometry, opl_tilt_x=tilt),
        geometry,
        directory / "tilted_transposed",
        core_perturbation=Perturbation(transpose_axes=True),
    )
    if baseline is None or tilted is None or transposed is None:
        return {"status": "a configuration failed to run"}

    def offset(measurement):
        return (
            measurement.peak_index[0] - measurement.intensity.shape[0] // 2,
            measurement.peak_index[1] - measurement.intensity.shape[1] // 2,
        )

    base_offset = offset(baseline)
    tilt_offset = offset(tilted)
    transpose_offset = offset(transposed)
    moved_x = tilt_offset[1] - base_offset[1]
    moved_y = tilt_offset[0] - base_offset[0]

    return {
        "status": "ran",
        "injected_opl_tilt_along_x": tilt,
        "predicted_shift_m": predicted_shift_m,
        "predicted_shift_pixels": predicted_pixels,
        "baseline_peak_offset_pixels": list(base_offset),
        "tilted_peak_offset_pixels": list(tilt_offset),
        "measured_shift_pixels_x": moved_x,
        "measured_shift_pixels_y": moved_y,
        "shift_ratio_measured_over_predicted": (
            abs(moved_x) / abs(predicted_pixels) if predicted_pixels else None
        ),
        "tilt_moves_the_psf_along_x_only": bool(abs(moved_x) > 1 and abs(moved_y) <= 1),
        "transposed_peak_offset_pixels": list(transpose_offset),
        "transpose_swaps_the_displacement_axis": bool(
            abs(transpose_offset[0] - base_offset[0]) > 1
            and abs(transpose_offset[1] - base_offset[1]) <= 1
        ),
        "transpose_is_detectable_here": bool(transpose_offset != tilt_offset and abs(moved_x) > 1),
        "why_this_exists": (
            "the frozen configurations cannot detect a transpose; this one can, and "
            "against an analytically predicted displacement rather than against "
            "another run of the same code"
        ),
    }


def _ray_count_convergence(
    geometry: dict[str, Any], directory: Path, *, distance_m: float, counts=(16, 24, 32, 48, 64, 96)
) -> dict[str, Any]:
    """Is the FFT-oracle residual dominated by ray sampling? Vary only the rays.

    The protocol's ``ray_sampling_error`` term is null with ``status:
    to_be_measured`` and owner CHE-38, while the ``fft_oracle_intensity_relative_l2``
    gate of 1.0e-3 explicitly requires that unmeasured term to fit inside it. If
    the measured residual exceeds the gate, the first question is whether this is
    the term responsible -- and the answer is a trend, not an opinion.

    CHE-35 already found the reconstruction far from ray-converged at these counts.
    """
    rows = []
    for count in counts:
        global NUM_RAYS
        previous = NUM_RAYS
        NUM_RAYS = count
        try:
            passed = _shipping_psf(geometry, directory / f"n{count}")
            if passed["status"] != "succeeded":
                rows.append({"num_rays_requested": count, "status": passed["status"]})
                continue
            _, aberration, _ = _aberration(
                passed["rays"], geometry, observation_z_m=geometry["image_z_m"]
            )
            oracle = _oracle_psf(aberration, geometry, distance_m=distance_m, factor=8)
            comparison = _oracle_vs_shipping(
                passed["measurement"], oracle, max_radius_m=5.0 * 6.4856e-6
            )
            peak_full = _strehl_against_airy(
                passed["measurement"], numerical_aperture=geometry["na_frozen"]
            )
            peak_windowed = _strehl_against_airy(
                passed["measurement"],
                numerical_aperture=geometry["na_frozen"],
                within_airy_radii=5.0,
            )
            rows.append(
                {
                    "num_rays_requested": count,
                    "traced_rays": int(np.asarray(aberration.positions_m).shape[0]),
                    "relative_l2_vs_fft_oracle": comparison["relative_l2_residual"],
                    "max_abs_residual": comparison["max_abs_residual"],
                    "airy_peak_deficit_full_window": peak_full["peak_intensity_relative_deficit"],
                    "airy_peak_deficit_within_5_airy_radii": peak_windowed[
                        "peak_intensity_relative_deficit"
                    ],
                    "psf_border_energy_fraction": passed["measurement"].border_energy_fraction,
                }
            )
        finally:
            NUM_RAYS = previous
    trend = [r.get("relative_l2_vs_fft_oracle") for r in rows if r.get("relative_l2_vs_fft_oracle")]
    return {
        "purpose": (
            "attribute the FFT-oracle residual. Only the ray count changes; the "
            "grid, padding, plane and system are the frozen ones."
        ),
        "rows": rows,
        "monotonically_falling": bool(len(trend) > 1 and all(b < a for a, b in pairwise(trend))),
        "first_over_last_ratio": (trend[0] / trend[-1] if len(trend) > 1 and trend[-1] else None),
        "protocol_term": "ray_sampling_error (value: null, status: to_be_measured, owner CHE-38)",
    }


def _strehl_against_airy(
    measurement, *, numerical_aperture: float, within_airy_radii: float | None = None
) -> dict[str, Any]:
    """Peak intensity relative to the analytic Airy at equal energy: the Strehl.

    This is the number the protocol's ``airy_peak_intensity_relative`` gate is
    about. It cannot be read off a peak-normalized PSF, where both peaks are 1 by
    construction, so it uses the raw scale the measurement retained.
    """
    from verification.psf_oracles import airy_psf_on_grid

    psf = measurement.intensity
    analytic = airy_psf_on_grid(
        shape=psf.shape,
        sample_pitch_m=measurement.sample_pitch_m,
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=numerical_aperture,
    )
    mask = np.ones_like(psf, dtype=bool)
    if within_airy_radii is not None:
        from verification.psf_oracles import airy_first_null_radius_m

        limit = within_airy_radii * airy_first_null_radius_m(WAVELENGTH_M, numerical_aperture)
        ny, nx = psf.shape
        dy, dx = measurement.sample_pitch_m
        yy = (np.arange(ny) - ny // 2) * dy
        xx = (np.arange(nx) - nx // 2) * dx
        mask = np.hypot(yy[:, None], xx[None, :]) <= limit

    measured_ratio = float(np.max(psf)) / float(np.sum(psf[mask]))
    analytic_ratio = float(np.max(analytic)) / float(np.sum(analytic[mask]))
    strehl = measured_ratio / analytic_ratio
    return {
        "strehl_against_analytic_airy": strehl,
        "peak_intensity_relative_deficit": 1.0 - strehl,
        "energy_window_airy_radii": within_airy_radii,
        "energy_pixels": int(mask.sum()),
        "definition": (
            "(peak / summed energy) of the measured PSF over the same ratio for the "
            "analytic Airy sampled on the same grid. Equal-energy normalization, so "
            "the uncalibrated ray-weight scale cancels. Reported over the full window "
            "and over a bounded one, because a ray-starved reconstruction puts energy "
            "in a broad pedestal and the full-window sum is dominated by it."
        ),
    }


def characterize() -> dict[str, Any]:
    from verification.psf_oracles import (
        AIRY_FIRST_NULL_COEFFICIENT_EXACT,
        AIRY_FIRST_NULL_COEFFICIENT_ROUNDED,
        airy_first_null_radius_m,
        airy_psf_on_grid,
        first_null_comparison,
        numerical_aperture_from_geometry,
    )

    protocol = _protocol()
    frozen_singlet = next(entry for entry in protocol["systems"] if entry["id"] == "M3-SINGLET-REF")
    out: dict[str, Any] = {
        "probe": "m3_psf_verification",
        "issue": "CHE-37 (M3.8)",
        "protocol_id": protocol["protocol_id"],
        "wavelength_m": WAVELENGTH_M,
        "num_rays_requested": NUM_RAYS,
    }

    workdir = Path(tempfile.mkdtemp(prefix="m3_psf_"))
    try:
        # ---------------------------------------------------------------
        # The Airy radius defect, stated in numbers before anything uses it
        # ---------------------------------------------------------------
        na = SINGLET["na_frozen"]
        radius_exact = airy_first_null_radius_m(WAVELENGTH_M, na)
        frozen_value_um = float(frozen_singlet["airy_radius_um"])
        out["airy_radius_protocol_defect"] = {
            "frozen_field": "systems.M3-SINGLET-REF.airy_radius_um",
            "frozen_value_um": frozen_value_um,
            "frozen_value_is": "1.22 * lambda / NA, the first-null DIAMETER",
            "true_first_null_radius_um": radius_exact * 1e6,
            "ratio_frozen_over_radius": frozen_value_um / (radius_exact * 1e6),
            "coefficient_used": AIRY_FIRST_NULL_COEFFICIENT_EXACT,
            "coefficient_protocol_quotes": AIRY_FIRST_NULL_COEFFICIENT_ROUNDED,
            "coefficient_difference_pct": 100.0
            * (AIRY_FIRST_NULL_COEFFICIENT_ROUNDED / AIRY_FIRST_NULL_COEFFICIENT_EXACT - 1.0),
            "frozen_pixels_field": "sampling.grids.M3-SINGLET-REF.airy_radius_in_pixels",
            "frozen_pixels_value": float(
                protocol["sampling"]["grids"]["M3-SINGLET-REF"]["airy_radius_in_pixels"]
            ),
            "true_radius_in_pixels": radius_exact / SINGLET["pitch_m"],
            "handling": (
                "the oracle computes the radius from the first zero of J1 and never "
                "reads the frozen field. The defect is recorded, not worked around, "
                "and no tolerance was widened."
            ),
        }
        out["numerical_aperture"] = {
            "frozen_protocol": na,
            "geometry_sine": numerical_aperture_from_geometry(
                0.4987073505473812e-3 / 2, SINGLET["image_z_m"] - SINGLET["pupil_z_m"]
            ),
            "paraxial_a_over_R": (0.4987073505473812e-3 / 2)
            / (SINGLET["image_z_m"] - SINGLET["pupil_z_m"]),
            "note": (
                "three definitions within 0.5% of each other. 0.5% of the Airy "
                "radius is 0.03 um, above the profile tolerance the residual is "
                "compared against, so each oracle records which one it used. The "
                "FFT oracle's image mapping x = lambda R f is paraxial, so it is "
                "compared against a/R."
            ),
        }

        # ---------------------------------------------------------------
        # 1. Diffraction-limited: shipping path vs analytic Airy
        # ---------------------------------------------------------------
        distance = SINGLET["image_z_m"] - SINGLET["pupil_z_m"]
        na_paraxial = (0.4987073505473812e-3 / 2) / distance
        dl = _shipping_psf(SINGLET, workdir / "dl")
        if dl["status"] != "succeeded":
            out["diffraction_limited"] = {"status": dl["status"], "error": dl.get("error")}
            return out

        rays_dl = dl["rays"]
        bundle_dl, aberration_dl, focus_dl = _aberration(
            rays_dl, SINGLET, observation_z_m=SINGLET["image_z_m"]
        )
        dl["aberration"] = aberration_dl

        measurement_dl = dl["measurement"]
        pitch_dl = measurement_dl.sample_pitch_m
        compare_radius = 5.0 * radius_exact
        analytic_on_grid = airy_psf_on_grid(
            shape=measurement_dl.intensity.shape,
            sample_pitch_m=pitch_dl,
            wavelength_m=WAVELENGTH_M,
            numerical_aperture=na,
        )
        out["diffraction_limited"] = {
            "configuration": {
                "sample": SINGLET["sample"],
                "hy": SINGLET["hy"],
                "grid_n": SINGLET["grid_n"],
                "pitch_m": SINGLET["pitch_m"],
                "pad_width": SINGLET["pad_width"],
                "propagation_distance_m": distance,
                "traced_rays": int(np.asarray(bundle_dl.positions_m).shape[0]),
            },
            "pupil_wavefront_at_observation_plane": aberration_dl.as_dict(),
            "pupil_wavefront_at_best_focus": focus_dl.as_dict(),
            "best_focus_offset_from_frozen_plane_m": focus_dl.sphere.shift_m[2],
            "psf": {
                "shape": list(measurement_dl.intensity.shape),
                "sample_pitch_m": list(pitch_dl),
                "peak_index": list(measurement_dl.peak_index),
                "peak_position_m": list(measurement_dl.peak_position_m),
                "grid_center_index": [
                    measurement_dl.intensity.shape[0] // 2,
                    measurement_dl.intensity.shape[1] // 2,
                ],
                "peak_is_on_axis": bool(
                    measurement_dl.peak_index
                    == (
                        measurement_dl.intensity.shape[0] // 2,
                        measurement_dl.intensity.shape[1] // 2,
                    )
                ),
                "border_energy_fraction": measurement_dl.border_energy_fraction,
                "normalization": measurement_dl.psf.normalization,
                "coherence_model": measurement_dl.psf.coherence_model,
            },
            "vs_analytic_airy": {
                **_profile_residual(
                    measurement_dl.intensity,
                    analytic_on_grid,
                    measured_pitch=pitch_dl,
                    reference_pitch=pitch_dl,
                    max_radius_m=compare_radius,
                ),
                "first_null": first_null_comparison(
                    measurement_dl.intensity,
                    sample_pitch_m=pitch_dl,
                    wavelength_m=WAVELENGTH_M,
                    numerical_aperture=na,
                ),
            },
            "energy_ledger": _energy_ledger(dl, SINGLET),
        }

        # The FFT oracle on the same traced data.
        oracle_dl = _oracle_psf(aberration_dl, SINGLET, distance_m=distance)
        out["diffraction_limited"]["vs_fft_oracle"] = {
            **oracle_dl.as_dict(),
            **_oracle_vs_shipping(measurement_dl, oracle_dl, max_radius_m=compare_radius),
            "profile_comparison_on_mismatched_grids": _profile_residual(
                measurement_dl.intensity,
                oracle_dl.intensity,
                measured_pitch=pitch_dl,
                reference_pitch=oracle_dl.sample_pitch_m,
                max_radius_m=compare_radius,
            ),
            "oracle_first_null": first_null_comparison(
                oracle_dl.intensity,
                sample_pitch_m=oracle_dl.sample_pitch_m,
                wavelength_m=WAVELENGTH_M,
                numerical_aperture=na_paraxial,
            ),
        }
        reference_residual = out["diffraction_limited"]["vs_analytic_airy"][
            "relative_l2_profile_residual"
        ]

        # ---------------------------------------------------------------
        # 2. Aberrated by defocus: shipping path vs the FFT oracle
        # ---------------------------------------------------------------
        defocused_plane = SINGLET["image_z_m"] + DEFOCUS_M
        defocused = _shipping_psf(
            SINGLET, workdir / "defocus", target_plane_z_m=defocused_plane, rays=rays_dl
        )
        if defocused["status"] == "succeeded":
            _, aberration_def, focus_def = _aberration(
                rays_dl, SINGLET, observation_z_m=defocused_plane
            )
            defocused["aberration"] = aberration_def
            measurement_def = defocused["measurement"]
            oracle_def = _oracle_psf(
                aberration_def, SINGLET, distance_m=defocused_plane - SINGLET["pupil_z_m"]
            )
            out["defocused"] = {
                "configuration": {
                    "sample": SINGLET["sample"],
                    "defocus_m": DEFOCUS_M,
                    "observation_plane_z_m": defocused_plane,
                    "why_not_reverse_telephoto_off_axis": (
                        "ReverseTelephoto is diffraction limited at every field "
                        "point; see reverse_telephoto_field_scan below."
                    ),
                },
                "pupil_wavefront_at_observation_plane": aberration_def.as_dict(),
                "pupil_wavefront_at_best_focus": focus_def.as_dict(),
                "defocus_is_visible_only_without_the_sphere_fit": (
                    "the fit is free to move the sphere centre to where the defocus "
                    "put the focus, so at_best_focus reports the intrinsic aberration "
                    "and the oracle uses at_observation_plane"
                ),
                "psf": {
                    "peak_index": list(measurement_def.peak_index),
                    "border_energy_fraction": measurement_def.border_energy_fraction,
                    "raw_peak_intensity": measurement_def.raw_peak_intensity,
                    "raw_window_energy": measurement_def.raw_window_energy,
                },
                "vs_fft_oracle": {
                    **oracle_def.as_dict(),
                    **_oracle_vs_shipping(measurement_def, oracle_def, max_radius_m=compare_radius),
                },
                "vs_analytic_airy_should_disagree": _profile_residual(
                    measurement_def.intensity,
                    airy_psf_on_grid(
                        shape=measurement_def.intensity.shape,
                        sample_pitch_m=measurement_def.sample_pitch_m,
                        wavelength_m=WAVELENGTH_M,
                        numerical_aperture=na,
                    ),
                    measured_pitch=measurement_def.sample_pitch_m,
                    reference_pitch=measurement_def.sample_pitch_m,
                    max_radius_m=compare_radius,
                ),
                "energy_ledger": _energy_ledger(defocused, SINGLET),
                "strehl": _strehl(dl, defocused),
            }
        else:
            out["defocused"] = {"status": defocused["status"], "error": defocused.get("error")}

        # ---------------------------------------------------------------
        # 3. Off-axis asymmetry vehicle
        # ---------------------------------------------------------------
        off = _shipping_psf(REVERSE_TELEPHOTO, workdir / "offaxis")
        if off["status"] == "succeeded":
            bundle_off, aberration_off, focus_off = _aberration(
                off["rays"], REVERSE_TELEPHOTO, observation_z_m=REVERSE_TELEPHOTO["image_z_m"]
            )
            measurement_off = off["measurement"]
            ny, nx = measurement_off.intensity.shape
            out["off_axis_asymmetry_vehicle"] = {
                "configuration": {
                    "sample": REVERSE_TELEPHOTO["sample"],
                    "hy": REVERSE_TELEPHOTO["hy"],
                    "grid_n": REVERSE_TELEPHOTO["grid_n"],
                    "pitch_m": REVERSE_TELEPHOTO["pitch_m"],
                    "purpose": (
                        "excite the x/y degree of freedom. A circular pupil with an "
                        "on-axis PSF cannot detect a transpose, and both other "
                        "configurations are exactly that. Since CHE-41 this vehicle "
                        "does put the PSF off centre in y only; see off_axis_handoff."
                    ),
                },
                "pupil_wavefront_at_observation_plane": aberration_off.as_dict(),
                "pupil_wavefront_at_best_focus": focus_off.as_dict(),
                "psf": {
                    "peak_index": list(measurement_off.peak_index),
                    "peak_position_m": list(measurement_off.peak_position_m),
                    "grid_center_index": [ny // 2, nx // 2],
                    "peak_is_off_axis_in_y_only": bool(
                        measurement_off.peak_index[0] != ny // 2
                        and measurement_off.peak_index[1] == nx // 2
                    ),
                    "border_energy_fraction": measurement_off.border_energy_fraction,
                },
                "energy_ledger": _energy_ledger(off, REVERSE_TELEPHOTO),
            }
            # The off-axis handoff, re-measured through this probe's own path.
            positions = np.asarray(bundle_off.positions_m)
            opl = np.asarray(bundle_off.optical_path_length_m)
            design = np.stack(
                [np.ones_like(positions[:, 1]), positions[:, 0], positions[:, 1]], axis=1
            )
            slope, *_ = np.linalg.lstsq(design, opl, rcond=None)
            step = REVERSE_TELEPHOTO["image_z_m"] - REVERSE_TELEPHOTO["pupil_z_m"]
            directions = np.asarray(bundle_off.directions)
            geometric_height = float(
                np.mean(positions[:, 1] + directions[:, 1] * step / directions[:, 2])
            )
            out["off_axis_handoff"] = {
                "verdict": (
                    "verified at Hy = 0.2 by CHE-41, and re-measured here through "
                    "M3.8's own path. The block this replaces recorded the defect; its "
                    "numbers are retained below under superseded_finding, because five "
                    "records were measured while they were true."
                ),
                "geometric_image_height_m": geometric_height,
                "geometric_image_height_pixels": geometric_height / REVERSE_TELEPHOTO["pitch_m"],
                "measured_psf_peak_offset_pixels": (measurement_off.peak_index[0] - ny // 2),
                "declared_opl_linear_slope_y": float(slope[2]),
                "opl_slope_required_to_reach_the_geometric_point": geometric_height / step,
                "slope_present_as_fraction_of_required": float(
                    slope[2] / (geometric_height / step)
                ),
                "wavefront_pv_waves_against_the_geometric_point": (
                    aberration_off.peak_to_valley_waves
                ),
                "wavefront_pv_waves_against_the_fitted_centre": focus_off.peak_to_valley_waves,
                "fitted_sphere_centre_y_m": focus_off.sphere.center_m[1],
                "fitted_centre_distance_from_axis_m": abs(focus_off.sphere.center_m[1]),
                "fitted_centre_distance_from_geometric_point_m": abs(
                    focus_off.sphere.center_m[1] - geometric_height
                ),
                "diagnosis": (
                    "the declared pupil OPL now carries the linear tilt the field "
                    "requires, and the shipping PSF lands at the traced chief-ray "
                    "intersection. The fitted least-squares slope reads ~0.19% above "
                    "the y_image / R the geometry names because a converging sphere's "
                    "slope is not constant across the pupil -- the sphere fit, not the "
                    "slope, is the oracle. CHE-41's own record carries that fit and a "
                    "reference-sphere-free geometric spot check."
                ),
                "cause_established_by_che41": (
                    "for an object at infinity Optiland seeds the OPD accumulator on a "
                    "plane perpendicular to z, not on the incoming wavefront. This "
                    "probe's guess at the difference, sin(theta) * y, was right in "
                    "mechanism and wrong in form: it is n_object * (d0 . r_launch), "
                    "evaluated at the LAUNCH coordinate, which is why no downstream "
                    "arithmetic could repair it and the ray adapter had to export it. "
                    "CHE-30 recorded the PISTON consequence of that launch plane; the "
                    "TILT consequence is invisible on axis, and CHE-30, CHE-32 and "
                    "CHE-33 all validated on axis only."
                ),
                "fixed_by": (
                    "CHE-41, under slice_protocol amendment A4: the declared off-axis "
                    "OPL reference is the incoming tilted wavefront. Evidence: "
                    "benchmarks/probes/records/m3_off_axis_handoff.json."
                ),
                "consequence_for_this_ticket": (
                    "M3.8's blind-spot audit found no frozen configuration could detect "
                    "an x/y transpose, and named this configuration's on-axis PSF as "
                    "the reason. That reason is gone -- the PSF is now 114 pixels off "
                    "axis in y only -- but the transpose control below STILL reads a "
                    "margin of ~1.0, because its metric azimuthally averages about the "
                    "grid centre and cannot tell (114, 0) from (0, 114). See CHE-41's "
                    "axis_transpose_control for the scoring that does detect it, and "
                    "orientation_control below for the synthetic-tilt control that was "
                    "M3.8's mitigation."
                ),
                "superseded_finding": {
                    "verdict": (
                        "the off-axis handoff is NOT verified, and this vehicle cannot "
                        "serve -- as measured by M3.8 before CHE-41"
                    ),
                    "declared_opl_linear_slope_y": 8.732361728171059e-05,
                    "slope_present_as_fraction_of_required": 0.0012767217717386374,
                    "fitted_sphere_centre_y_m": -1.013286092185631e-06,
                    "fitted_centre_distance_from_geometric_point_m": 0.00020993654920029492,
                    "wavefront_pv_waves_against_the_geometric_point": 57.01590386063586,
                    "wavefront_pv_waves_against_the_fitted_centre": 0.07187252771390978,
                    "measured_psf_peak_offset_pixels": -1,
                    "retained_because": (
                        "M3.8's report, the M3 protocol's open_structural_items entry "
                        "and CHE-41's own ticket all quote these numbers. They were "
                        "correct measurements of a wrong declaration."
                    ),
                },
            }
        else:
            out["off_axis_asymmetry_vehicle"] = {
                "status": off["status"],
                "error": off.get("error"),
            }

        # ---------------------------------------------------------------
        # 4. Negative controls
        # ---------------------------------------------------------------
        out["negative_controls"] = _negative_controls(
            rays_dl,
            SINGLET,
            workdir / "controls",
            baseline_psf=measurement_dl.intensity,
            max_radius_m=compare_radius,
            airy_na=na,
        )
        out["negative_controls"]["graph_node_residual_for_reference"] = reference_residual

        # The amplitude degree of freedom, measured rather than assumed.
        ray_arrays = dict(np.load(rays_dl.uri))
        intensity = ray_arrays["intensity"]
        survived = ray_arrays.get("survived")
        kept = intensity if survived is None else intensity[survived.astype(bool)]
        out["amplitude_degree_of_freedom"] = {
            "ray_intensity_min": float(kept.min()),
            "ray_intensity_max": float(kept.max()),
            "ray_intensity_spread": float(kept.max() - kept.min()),
            "relative_spread": float((kept.max() - kept.min()) / kept.mean()),
            "finding": (
                "Optiland's per-ray intensity weights on this configuration are "
                "uniform to ~1e-5 relative, so replacing every amplitude with 1 is "
                "an almost exact global scale -- and the frozen peak normalization "
                "cannot see a global scale. The amplitude path of the reconstruction "
                "is therefore NOT exercised by any M3 configuration. Exercising it "
                "needs a non-uniform weight: apodization, vignetting or a polarized "
                "Fresnel transmission, none of which M3 has."
            ),
            "consequence": (
                "the amplitude_weight_omitted control below is reported as "
                "undetected. That is the honest result, not a passing test."
            ),
        }

        # An orientation control that can actually fail.
        out["orientation_control"] = _orientation_control(rays_dl, SINGLET, workdir / "orientation")

        # Attribute the FFT-oracle residual.
        out["ray_count_convergence"] = _ray_count_convergence(
            SINGLET, workdir / "raycount", distance_m=distance
        )

        out["diffraction_limited"]["vs_analytic_airy"]["peak_intensity"] = _strehl_against_airy(
            measurement_dl, numerical_aperture=na
        )
        out["diffraction_limited"]["vs_analytic_airy"]["peak_intensity_within_5_airy_radii"] = (
            _strehl_against_airy(measurement_dl, numerical_aperture=na, within_airy_radii=5.0)
        )

        # ---------------------------------------------------------------
        # Gate verdicts, against the protocol's own frozen numbers
        # ---------------------------------------------------------------
        gates = protocol["tolerance_budget"]["gates"]
        airy_gate = float(gates["airy_peak_intensity_relative"]["value"])
        fft_gate = float(gates["fft_oracle_intensity_relative_l2"]["value"])
        energy_gate = float(gates["energy_accounting_unexplained_residual"]["value"])

        airy_measured = abs(
            out["diffraction_limited"]["vs_analytic_airy"]["peak_intensity_within_5_airy_radii"][
                "peak_intensity_relative_deficit"
            ]
        )
        airy_full_window = abs(
            out["diffraction_limited"]["vs_analytic_airy"]["peak_intensity"][
                "peak_intensity_relative_deficit"
            ]
        )
        sweep_residuals = [
            row.get("relative_l2_vs_fft_oracle")
            for row in out["ray_count_convergence"]["rows"]
            if row.get("relative_l2_vs_fft_oracle")
        ]
        sweep_best = min(sweep_residuals) if sweep_residuals else None
        sweep_peak = [
            abs(row["airy_peak_deficit_within_5_airy_radii"])
            for row in out["ray_count_convergence"]["rows"]
            if "airy_peak_deficit_within_5_airy_radii" in row
        ]
        fft_measured = out["diffraction_limited"]["vs_fft_oracle"]["relative_l2_residual"]
        ledger = out["diffraction_limited"]["energy_ledger"]["ratios"]
        psf_step = abs(1.0 - float(ledger["psf_integral_over_propagated_out"]))

        out["gates"] = {
            "airy_peak_intensity_relative": {
                "gate": airy_gate,
                "measured": airy_measured,
                "ratio_measured_over_gate": airy_measured / airy_gate,
                "verdict": "pass" if airy_measured <= airy_gate else "FAIL",
                "measured_metric": "peak/energy within 5 Airy radii, vs the same for analytic Airy",
                "full_window_variant": airy_full_window,
                "why_two_variants": (
                    "over the full 1320^2 window the analytic Airy's own wings, which "
                    "decay as 1/r^3, dominate its energy sum out to the grid corners at "
                    "+/-1.7 mm, while the measured PSF's are truncated. The bounded "
                    "window is the comparison that means something; the full-window "
                    "number is reported so the difference is visible rather than chosen."
                ),
                "best_over_ray_count_sweep": min(sweep_peak) if sweep_peak else None,
                "ray_sampling_dependence": (
                    "the full-window deficit collapses from 0.886 at 817 traced rays to "
                    "-0.007 at 12481, so this metric is dominated by the same "
                    "unmeasured ray_sampling_error term as the FFT-oracle gate: a "
                    "ray-starved reconstruction puts energy into a broad pedestal, "
                    "which is exactly what a peak/energy ratio penalizes."
                ),
                "note": (
                    "the gate composes 9.1e-4 of residual aberration + 3.5e-4 float32 "
                    "+ margin. The 9.1e-4 was derived at BEST FOCUS (0.00479 waves "
                    "RMS); the slice observes the frozen image plane, 7.2 um away, "
                    "where the wavefront is 0.00538 waves RMS and the Marechal "
                    "deficit is 1.14e-3. Still inside the gate, but the term was "
                    "derived at a plane the slice does not use."
                ),
            },
            "fft_oracle_intensity_relative_l2": {
                "gate": fft_gate,
                "measured": fft_measured,
                "ratio_measured_over_gate": fft_measured / fft_gate,
                "verdict": "pass" if fft_measured <= fft_gate else "FAIL",
                "best_residual_at_highest_ray_count": sweep_best,
                "best_over_gate": (sweep_best / fft_gate) if sweep_best else None,
                "ray_sampling_accounts_for": (
                    f"{fft_measured / sweep_best:.2g}x of the excess" if sweep_best else None
                ),
                "diagnosis": (
                    "FAILS. Most of it is attributed by measurement rather than by "
                    "argument: ray_count_convergence holds the grid, padding, plane "
                    "and system fixed and varies only the ray count, and the residual "
                    "falls monotonically. That is the protocol's own "
                    "ray_sampling_error term -- value null, status to_be_measured, "
                    "owner CHE-38 -- which this gate's text explicitly requires to fit "
                    "inside it, and CHE-35 independently found the reconstruction far "
                    "from ray-converged at these counts."
                ),
                "unresolved": (
                    "the trend FLATTENS above the gate. It does not extrapolate to "
                    "1.0e-3, so ray sampling is not the whole story and a residual "
                    "floor remains unattributed. Leading hypothesis, untested here: "
                    "the shipping reconstruction's pupil edge is set by where the rays "
                    "stop and is soft over roughly one ray spacing, while the oracle "
                    "applies a hard circular mask at the largest traced radius -- a "
                    "difference in the aperture function, which is what sets the ring "
                    "structure the residual is measuring. Reported as an open finding. "
                    "No tolerance was widened, and the gate is left as it stands."
                ),
            },
            "energy_accounting_unexplained_residual": {
                "gate": energy_gate,
                "measured_unattributed": psf_step,
                "verdict": "pass" if psf_step <= energy_gate else "FAIL",
                "note": (
                    "the only step in the ledger that must be unity is "
                    "psf_integral / propagated_power_out, because the measurement is "
                    "|u|^2 on the same grid. Every other step is attributed to a named "
                    "mechanism; the ray-weight-to-field-power step is a measure "
                    "conversion and is not expected to be 1."
                ),
            },
        }

        # The off-axis vehicle exists to catch what the on-axis controls cannot.
        # Since CHE-41 its PSF is 114 pixels off axis, so the oracle and the
        # azimuthal average are placed there; leaving them at the origin makes every
        # margin read ~1 and measures the offset instead of the perturbation.
        if out.get("off_axis_asymmetry_vehicle", {}).get("psf"):
            out["off_axis_negative_controls"] = _negative_controls(
                off["rays"],
                REVERSE_TELEPHOTO,
                workdir / "offaxis_controls",
                baseline_psf=off["measurement"].intensity,
                max_radius_m=compare_radius,
                airy_na=REVERSE_TELEPHOTO["na_frozen"],
                psf_center_m=(geometric_height, 0.0),
            )
            out["off_axis_negative_controls"]["scoring_centre_m_y_x"] = [
                geometric_height,
                0.0,
            ]
            out["off_axis_negative_controls"]["scoring_note"] = (
                "the analytic Airy reference and the azimuthal average are both "
                "centred on the traced geometric image point, not on the grid origin. "
                "Measured with them at the origin instead, all six controls score "
                "within 2.6x of 1.0 -- including an x/y transpose that visibly moves "
                "the peak from (1003, 889) to (889, 1003) -- because an azimuthal "
                "average about a point the PSF is not at is nearly invariant under "
                "anything a perturbation can do. CHE-41's own record scores the "
                "transpose a second way, with no azimuthal average at all."
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return out


def main() -> None:
    record = characterize()
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {RECORD_PATH.relative_to(ROOT)}")
    print(json.dumps(record, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
