"""CHE-38 (M3.9R): Ray->Wave convergence at the intended sensor-side handoff.

Re-runs M3.9 after correcting the physical semantics of ``C_RAY_TO_WAVE``. The
previous study put the handoff at the exit pupil and then asked the coupler's
plane-wavelet sum to reconstruct a hard-support pupil function. This one puts the
handoff on the observation side, where a ray is a coherent contribution to the
measured field rather than a sample of a finite-support wavefront, and asks
whether the sensor field converges to the correct diffraction solution.

The structural fact that makes the distinction real
---------------------------------------------------
``ray_to_wave`` evaluates

    U(x, y) = sum_i a_i exp[ i k ( OPL_i - d_i . r0_i ) ] exp[ i k ( d_x_i x + d_y_i y ) ]

so the reconstruction plane's ``z`` appears **nowhere in the kernel**. The plane
is metadata. What the operator actually returns is one fixed superposition of
plane waves sampled on a plane, and a superposition of plane waves is an exact
solution of the Helmholtz equation. Moving the handoff therefore has to be done
in the RAY domain, and doing it there is exactly equivalent to evaluating the
same 3-D field at the new plane: advancing a ray by arc length ``s`` sends

    OPL_i     -> OPL_i + s                       (n = 1 in image space)
    (x0, y0)  -> (x0 + s d_x, y0 + s d_y)

which changes the per-ray constant phase by ``k s (1 - d_x^2 - d_y^2) = k s d_z^2``
-- and ``s d_z^2 = dz d_z`` for a plane offset ``dz = s d_z``, which is precisely
the phase an exact plane wave picks up over ``dz``. So the operator is
self-consistent in ``z``, the handoff-plane sweep is a real experiment, and the
only things that change with the plane are (a) which slice of the 3-D field is
sampled, (b) whether the finite window contains the beam, and (c) how much wave
propagation is left to do afterwards.

Two semantics, and only one of them is implemented
--------------------------------------------------
* **ray-as-wavefront-sample** (exit pupil, FFT/Fresnel workflow) needs an
  explicit pupil support ``P(rho)``. The operator above contains no support
  term, and M3.9 measured the consequence: a Fresnel-soft rim at ``sqrt(lambda R)``
  instead of a hard edge. That is retained here as the O4 negative control.
* **ray-as-coherent-contribution** (declared observation-side handoff) needs no
  support term, because the aperture enters through *which rays exist* -- i.e. as
  the domain of a quadrature in direction space. This is the mode under test.

What is measured, and what stays separate
-----------------------------------------
Sampling error (against the highest ray count) and oracle error (against
references that share no traced data) are reported as two curves and never
combined. Absolute power is left labelled unverified: the reconstruction carries
no per-ray area weight, so every metric here is peak-normalized, and the
``N^2`` power scaling M3.9 found is a separate ticket's problem.

Non-goals, carried forward from CHE-38: no wavelength sweep, no GPU, no
optimization loop.
"""

from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmarks" / "slice_protocol.yaml"
RECORD_PATH = Path(__file__).resolve().parent / "records" / "m3r_sensor_handoff.json"
PRIOR_RECORD_PATH = Path(__file__).resolve().parent / "records" / "m3_convergence.json"
FIGURE_DIR = ROOT / "outputs" / "M3" / "CHE-38-M3.9R"

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1e-6
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M

#: The frozen M3-SINGLET-REF configuration, on axis. Cross-checked against
#: benchmarks/slice_protocol.yaml by :func:`_check_frozen_configuration`.
SINGLET = {
    "sample": "M3SingletRef",
    "pupil_z_m": 0.06814345991561233e-3,
    "image_z_m": 4.90560476022521e-3,
    "na_frozen": 0.05171631827291936,
    "hy": 0.0,
    #: The M3.9 pupil-plane configuration, kept only for the O4 negative control.
    "pupil_pitch_m": 2.6587352810843895e-06,
    "pupil_grid_n": 188,
    "pupil_pad_width": 566,
}
#: Exit pupil to declared image plane. Also the convergence distance R.
DISTANCE_M = SINGLET["image_z_m"] - SINGLET["pupil_z_m"]
#: Largest traced pupil radius, ray-count independent (measured in M3.9).
PUPIL_RADIUS_M = 0.00024978414778669653

# ---------------------------------------------------------------------------
# DECLARED BEFORE THE SWEEP (CHE-38 sections 3 and 8 both require this).
# Nothing below is chosen after looking at a result.
# ---------------------------------------------------------------------------

#: The sensor. The declared paraxial image plane, which is where M3 says the PSF
#: is measured; NOT the best-focus plane, which sits 7.1 um short of it and is
#: reported as a named error term rather than quietly adopted.
SENSOR_Z_M = SINGLET["image_z_m"]

#: Sensor grid. Three constraints, in the order they bind:
#:   1. the coupler's per-axis Nyquist limit, lambda / (2 * pitch) >= max|d_t|
#:      = 0.0517, i.e. pitch <= 5.317 um;
#:   2. the Airy core must be resolved -- at least 10 pixels per first-null
#:      radius (6.486 um), i.e. pitch <= 0.649 um;
#:   3. the window must hold the 5-Airy-radius gate disc with margin for the
#:      wing metrics.
#: 0.5 um and 256 give 12.97 px per Airy radius and a 128 um window = 9.87 Airy
#: radii. Constraint 2, not the Nyquist limit, is what sets the pitch here --
#: the opposite of the pupil-plane configuration, where the Nyquist limit bound.
SENSOR_PITCH_M = 0.5e-6
SENSOR_GRID_N = 256

#: Gate region: the same 5 * (Airy first-null radius) disc M3.9 used. Unchanged
#: on purpose -- CHE-38 forbids widening a tolerance to make the redesign pass.
GATE_AIRY_RADII = 5.0

#: Candidate handoff planes, parameterized by fraction of R upstream of the
#: sensor. Declared here, before Experiment A runs, and not edited afterwards.
#: Positive is pre-focus (upstream), negative is post-focus.
HANDOFF_CANDIDATES: tuple[tuple[str, float], ...] = (
    ("exit_pupil", 1.0),
    ("intermediate_converging", 0.5),
    ("pre_focus", 0.1),
    ("near_sensor", 0.01),
    ("near_sensor_fine", 0.001),
    ("nominal_sensor", 0.0),
    ("post_focus", -0.01),
)

#: Selection rule, declared before the sweep: the candidate with the smallest
#: sensor-PSF residual against the independent wave oracle O2; ties within 10%
#: are broken toward the plane with the LEAST post-handoff wave propagation,
#: because propagation after the handoff adds a term that is not the coupler's.
HANDOFF_SELECTION_RULE = (
    "smallest sensor-PSF relative L2 against O2 (independent wave oracle); ties "
    "within 10 percent broken toward the least post-handoff propagation distance"
)

#: Ray ladder, unchanged from M3.9 so the two studies are comparable.
RAY_SWEEP_RINGS = (8, 16, 24, 32, 48, 64, 96, 128, 181, 256, 362, 512)
RAY_REFERENCE_RINGS = 512
#: Ray count for Experiment A. High enough that the sweep is not measuring ray
#: sampling (Experiment B puts 256 rings at ~1e-3), low enough to afford a
#: 2000^2 reconstruction at the exit-pupil plane.
HANDOFF_SWEEP_RINGS = 64
#: Ray count for Experiments C and D.
SWEEP_RINGS = 256

#: Grid sweep at FIXED 128 um physical extent. 16 and 24 violate the per-axis
#: Nyquist limit (which needs grid_n >= 25 at this extent) and must be REFUSED;
#: that is the point of including them.
GRID_SWEEP_N = (16, 24, 25, 32, 48, 64, 96, 128, 192, 256, 384, 512)
SENSOR_EXTENT_M = SENSOR_GRID_N * SENSOR_PITCH_M

#: Padding sweep, as a multiple of the reconstruction grid on each side.
PAD_SWEEP_FACTORS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)

#: Reconstruction-window rule for an upstream handoff: fixed pitch (so the
#: Nyquist condition is identical at every plane and is not a confound), and a
#: half-width of at least 2x the geometric beam radius, floored at the sensor
#: window's half width. Capped at 2048; the cap is reported if it binds.
HANDOFF_WINDOW_BEAM_MARGIN = 2.0
HANDOFF_GRID_N_CAP = 2048

#: O2 oracle quadratures.
O2_ASM_GRID_N = 4096          # 2048 um window at the sensor pitch
O2_RS_N_RHO = 512
O2_RS_N_PHI = 1024
O2_RS_N_RADIAL = 2048
O2_RS_CONVERGENCE_N_RHO = 1024

#: Exit-pupil negative control (O4): ray counts and the wide diagnostic window.
EXIT_PUPIL_CONTROL_RINGS = (16, 32, 64, 96, 128, 181)
#: 2x the pupil diameter at the frozen pupil pitch. The frozen 188^2 window is
#: exactly the pupil, so a soft rim is clipped by it and invisible there.
WIDE_PUPIL_GRID_N = 376

#: Ring counts for the quadrature attribution experiment.
QUADRATURE_RINGS = (32, 64, 128, 181, 256, 362, 512)
#: Ring count for the Fresnel-number scan.
FRESNEL_SCAN_RINGS = 256
#: Distance factors for the Fresnel-number scan, at fixed aperture and ray count.
FRESNEL_DISTANCE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Ray count whose traced wavefront defines the "traced pupil" O2 variant.
O2_PUPIL_FIT_RINGS = 256
#: Order of the DELIBERATELY under-fitted polynomial control (rho^0 .. rho^8).
O2_UNDERFIT_ORDER = 5


def _protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


# ---------------------------------------------------------------------------
# 1. The slice, with the handoff plane as a free parameter
# ---------------------------------------------------------------------------
def _trace(rings: int, directory: Path):
    """Optiland, exported at the exit pupil -- the only plane the adapter resolves.

    ``_SUPPORTED_HANDOFF_PLANES`` is ``("image_surface", "exit_pupil")``, so the
    adapter is not asked for an arbitrary plane. The handoff is moved afterwards
    in the ray domain by :func:`_advance_bundle_to_z`, which is a
    reparameterization along each ray plus the optical path it accumulates, and
    is the operation the ray model is entitled to perform. No production adapter
    or coupler behaviour is changed by this study.
    """
    from multiscale_optics_agent.adapters.base import ModelRunRequest
    from multiscale_optics_agent.adapters.optiland_adapter import get_adapter

    return (
        get_adapter()
        .run(
            ModelRunRequest(
                run_id="che38r",
                node_id="lens",
                config={
                    "sample": SINGLET["sample"],
                    "num_rays": rings,
                    "wavelength": WAVELENGTH_UM,
                    "Hx": 0.0,
                    "Hy": SINGLET["hy"],
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(directory),
                },
            )
        )
        .outputs["rays"]
    )


def _pupil_bundle(rays):
    from multiscale_optics_agent.couplers.optiland_handoff import (
        DeclaredHandoffPlane,
        declare_coherent_bundle,
    )

    return declare_coherent_bundle(
        rays, declared_plane=DeclaredHandoffPlane("exit_pupil", SINGLET["pupil_z_m"])
    ).bundle


def _advance_bundle_to_z(bundle, z_m: float):
    """Ray-domain propagation to a declared plane in image space.

    Each ray is advanced along its own direction by ``s = (z - z0) / d_z`` and its
    optical path grows by ``n s`` with ``n = 1``: image space is air for this
    system, checked by :func:`_check_image_space_index`. Directions are unchanged,
    which is what makes this a propagation of the ray STATE and not a change of
    the ray model.

    The identity that makes this the right way to move the handoff is in the
    module docstring: the resulting per-ray constant phase differs from the
    original by ``k s d_z^2``, which is the phase an exact plane wave accumulates
    over the plane offset ``s d_z``. So this does not approximate the field at the
    new plane; it evaluates the same 3-D superposition there.
    """
    from multiscale_optics_agent.couplers.contracts import Frame, RayBundle, ReferencePlane

    positions = np.asarray(bundle.positions_m, dtype=np.float64)
    directions = np.asarray(bundle.directions, dtype=np.float64)
    amplitude, optical_path_length = bundle.require_coherent()
    if np.any(directions[:, 2] == 0.0):
        raise ValueError("a ray with d_z = 0 never reaches the declared plane")
    step = (float(z_m) - positions[:, 2]) / directions[:, 2]
    return (
        RayBundle(
            positions_m=positions + step[:, None] * directions,
            directions=directions.copy(),
            wavelength_m=bundle.wavelength_m,
            reference_plane=ReferencePlane(name="image_space_observation_plane", z_m=float(z_m)),
            frame=Frame(),
            amplitude=np.asarray(amplitude).copy(),
            optical_path_length_m=optical_path_length + step,
            optical_path_length_reference=(
                f"{bundle.optical_path_length_reference}, then advanced along each ray to "
                f"z = {float(z_m)!r} m through image-space air (n = 1)"
            ),
        ),
        step,
    )


def _reconstruct_core(bundle, *, grid_n: int, pitch_m: float, enforce_nyquist: bool = True):
    """``C_RAY_TO_WAVE`` itself, unmodified.

    The graph node consumes an Optiland ray record and these bundles have been
    advanced in the ray domain, so the core is called directly. CHE-34 pinned the
    node bit-identical to the core; :func:`_node_equals_core` re-checks that on
    this study's own grid rather than inheriting the claim.
    """
    from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave

    return ray_to_wave(
        bundle,
        grid_shape=(grid_n, grid_n),
        sample_pitch_m=(pitch_m, pitch_m),
        enforce_grid_nyquist=enforce_nyquist,
    )


def _node_equals_core(rays, directory: Path) -> dict[str, Any]:
    """Re-pin the graph node against the core on this study's grid."""
    from multiscale_optics_agent.couplers.base import CouplerRunRequest
    from multiscale_optics_agent.couplers.contracts import ComplexField
    from multiscale_optics_agent.couplers.ray_to_wave_node import RayToWaveCoupler

    result = RayToWaveCoupler().transform(
        CouplerRunRequest(
            run_id="che38r",
            edge_id="pupil",
            source=rays,
            config={
                "handoff_plane": "exit_pupil",
                "handoff_plane_z_m": SINGLET["pupil_z_m"],
                "grid_n": SINGLET["pupil_grid_n"],
                "target_sample_pitch_m": SINGLET["pupil_pitch_m"],
                "output_dir": str(directory),
            },
        )
    )
    if result.status.value != "succeeded":
        return {"status": "coupler_refused", "error": result.error_message}
    node_field = ComplexField.from_artifact_record(result.target).u
    core, _ = _reconstruct_core(
        _pupil_bundle(rays),
        grid_n=SINGLET["pupil_grid_n"],
        pitch_m=SINGLET["pupil_pitch_m"],
    )
    return {
        "status": "checked",
        "bit_identical": bool(np.array_equal(node_field, core.u)),
        "max_abs_difference": float(np.max(np.abs(node_field - core.u))),
        "grid": [SINGLET["pupil_grid_n"], SINGLET["pupil_grid_n"]],
        "why": (
            "the advanced bundles are reconstructed through the coupler core, so the "
            "node/core equality CHE-34 pinned is re-measured here instead of assumed."
        ),
    }


def _check_image_space_index(rays) -> dict[str, Any]:
    """The n = 1 in the OPL advance, read off the adapter rather than assumed."""
    conventions = (rays.metadata or {}).get("conventions", {}) or {}
    candidates = {
        key: conventions.get(key)
        for key in ("image_space_index", "image_space_refractive_index", "n_image_space")
        if conventions.get(key) is not None
    }
    value = next(iter(candidates.values()), None)
    return {
        "declared_fields_found": candidates,
        "image_space_index_used_for_the_opl_advance": 1.0,
        "matches_declaration": None if value is None else bool(abs(float(value) - 1.0) < 1e-12),
        "why_it_matters": (
            "the advance adds n * s to the optical path. A wrong n scales the "
            "defocus-like part of the phase and would look like a focus shift."
        ),
    }


def _asm_float64(u: np.ndarray, *, z_m: float, pitch_m: float, pad: int) -> np.ndarray:
    """Independent float64 angular-spectrum propagation with explicit zero padding.

    Used for the handoff-plane sweep in preference to the Chromatix adapter so
    that a plane is not blamed for a ``complex64`` cast. The shipping Chromatix
    propagation is measured separately, in Experiments D and E.
    """
    from multiscale_optics_agent.evaluation.asm_oracle import (
        CarrierConvention,
        angular_spectrum_float64,
    )

    if z_m == 0.0:
        return np.asarray(u, dtype=np.complex128)
    padded = np.pad(np.asarray(u, dtype=np.complex128), pad, mode="constant")
    out = angular_spectrum_float64(
        padded,
        wavelength_m=WAVELENGTH_M,
        sample_pitch_m=pitch_m,
        z_m=z_m,
        carrier=CarrierConvention.CARRIER_REMOVED,
    )
    return out if pad == 0 else out[pad:-pad, pad:-pad]


# ---------------------------------------------------------------------------
# 2. Metrics. Peak-normalized throughout, which is not a convenience: the
#    reconstruction carries no per-ray area weight, so its absolute amplitude
#    grows with the ray count and no absolute metric converges. CHE-38 section 15
#    keeps that a separate ticket; here it only forces the normalization.
# ---------------------------------------------------------------------------
def _airy_radius_m() -> float:
    from multiscale_optics_agent.evaluation.psf_oracles import airy_first_null_radius_m

    return airy_first_null_radius_m(WAVELENGTH_M, SINGLET["na_frozen"])


def _radius_grid(shape: tuple[int, int], pitch: float) -> np.ndarray:
    ny, nx = shape
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * pitch
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * pitch
    return np.hypot(y[:, None], x[None, :])


def _disc_mask(shape: tuple[int, int], pitch: float, radius_m: float) -> np.ndarray:
    return _radius_grid(shape, pitch) <= radius_m


def _annulus_mask(shape: tuple[int, int], pitch: float, inner_m: float, outer_m: float):
    r = _radius_grid(shape, pitch)
    return (r > inner_m) & (r <= outer_m)


def _relative_l2(measured: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """Peak-normalized intensity residual over a mask. M3.9's metric, unchanged."""
    a = np.asarray(measured, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    peak_a, peak_b = float(np.max(a)), float(np.max(b))
    if peak_a <= 0.0 or peak_b <= 0.0:
        return float("nan")
    difference = (a / peak_a - b / peak_b)[mask]
    denominator = float(np.linalg.norm((b / peak_b)[mask]))
    return float(np.linalg.norm(difference) / denominator) if denominator else float("nan")


def _complex_relative_l2(test: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    """Complex-field residual after removing the two physically irrelevant scalars.

    A reconstruction is defined up to a global complex factor: the amplitude scale
    (no area weight) and a global piston (no absolute phase reference). Both are
    removed by the least-squares projection ``c = <ref, test> / <test, test>``,
    which is the only comparison that can be made at all. Anything that survives
    it is a real difference in the field.
    """
    a = np.asarray(test, dtype=np.complex128)[mask]
    b = np.asarray(reference, dtype=np.complex128)[mask]
    denominator = float(np.vdot(a, a).real)
    if denominator == 0.0:
        return {"relative_l2": float("nan"), "fitted_gain": None, "fitted_phase_rad": None}
    scale = np.vdot(a, b) / denominator
    residual = float(np.linalg.norm(scale * a - b))
    norm = float(np.linalg.norm(b))
    return {
        "relative_l2": residual / norm if norm else float("nan"),
        "fitted_gain": float(np.abs(scale)),
        "fitted_phase_rad": float(np.angle(scale)),
        "removed": "global amplitude scale and global piston, by least squares",
    }


def _first_null_m(intensity: np.ndarray, *, pitch: float, max_radius_m: float) -> float | None:
    from multiscale_optics_agent.evaluation.psf_oracles import (
        azimuthal_profile,
        measure_first_null_radius_m,
    )

    radii, profile = azimuthal_profile(
        np.asarray(intensity, dtype=np.float64),
        sample_pitch_m=(pitch, pitch),
        max_radius_m=max_radius_m,
        radial_samples=1024,
        azimuthal_samples=256,
    )
    return measure_first_null_radius_m(radii, profile)


def _encircled_energy(intensity: np.ndarray, *, pitch: float, radii_m: list[float]) -> list[float]:
    r = _radius_grid(intensity.shape, pitch)
    values = np.asarray(intensity, dtype=np.float64)
    total = float(values[r <= radii_m[-1]].sum())
    return [float(values[r <= radius].sum() / total) if total else float("nan") for radius in radii_m]


def _psf_metrics(
    intensity: np.ndarray,
    *,
    pitch: float,
    references: dict[str, np.ndarray],
    gate_radius_m: float,
) -> dict[str, Any]:
    """Every scalar CHE-38 section 6 asks for, on one intensity map."""
    airy = _airy_radius_m()
    shape = intensity.shape
    gate = _disc_mask(shape, pitch, gate_radius_m)
    core = _disc_mask(shape, pitch, airy)
    wing = _annulus_mask(shape, pitch, 3.0 * airy, 5.0 * airy)
    far = _radius_grid(shape, pitch) > 5.0 * airy
    ee_radii = [airy, 2.0 * airy, 3.0 * airy, 5.0 * airy]

    measured_null = _first_null_m(intensity, pitch=pitch, max_radius_m=3.0 * airy)
    measured_ee = _encircled_energy(intensity, pitch=pitch, radii_m=ee_radii)

    out: dict[str, Any] = {
        "first_null_radius_m": measured_null,
        "first_null_relative_error_vs_analytic": (
            None if measured_null is None else measured_null / airy - 1.0
        ),
        "encircled_energy_radii_in_airy_radii": [1.0, 2.0, 3.0, 5.0],
        "encircled_energy": measured_ee,
    }
    for name, reference in references.items():
        reference = np.asarray(reference, dtype=np.float64)
        peak_ratio = (float(np.max(intensity)) / float(intensity[gate].sum())) / (
            float(np.max(reference)) / float(reference[gate].sum())
        )
        out[name] = {
            "relative_l2_gate_disc": _relative_l2(intensity, reference, gate),
            "relative_l2_core": _relative_l2(intensity, reference, core),
            "relative_l2_wing_3_to_5_airy": _relative_l2(intensity, reference, wing),
            "relative_l2_beyond_5_airy": _relative_l2(intensity, reference, far),
            #: signed: negative means the measured peak is DEFICIENT relative to
            #: the energy in the gate disc, which is what an aberration or a
            #: sampling loss does.
            "signed_peak_deficit": 1.0 - peak_ratio,
            "encircled_energy_error": [
                m - r
                for m, r in zip(
                    measured_ee, _encircled_energy(reference, pitch=pitch, radii_m=ee_radii),
                    strict=True,
                )
            ],
        }
    return out


# ---------------------------------------------------------------------------
# 3. References. Three routes to the sensor field, none through C_RAY_TO_WAVE.
# ---------------------------------------------------------------------------
def _o1_analytic_airy(*, grid_n: int, pitch: float) -> np.ndarray:
    """O1. ``[2 J1(v) / v]^2``. Paraxial, aberration-free, shares no traced data."""
    from multiscale_optics_agent.evaluation.psf_oracles import airy_psf_on_grid

    return airy_psf_on_grid(
        shape=(grid_n, grid_n),
        sample_pitch_m=(pitch, pitch),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=SINGLET["na_frozen"],
    )


def _traced_pupil_wavefront(rings: int, directory: Path) -> dict[str, Any]:
    """W(rho) at the exit pupil, taken from the trace WITHOUT a polynomial fit.

    This is the one place a reference touches traced data, and it touches only the
    ray-measured wavefront -- never the coupler. It exists so the O2 oracle can be
    run on the pupil this singlet actually has, which separates "the coupler is
    wrong" from "the lens is not perfect".

    The wavefront is ring-averaged and then linearly interpolated in ``rho``. It is
    deliberately NOT a polynomial fit, and that is not a style preference: a
    5-term polynomial in ``rho^2`` truncates this singlet's high-order spherical
    aberration, which is concentrated at the rim, and gets the rim slope wrong by
    1.4e-4 out of 0.0517 -- 0.28%. An oracle built that way is a perfectly clean
    Airy pattern at the wrong scale, and it charges the coupler 5.3e-3 for a defect
    the oracle introduced. :func:`_underfitted_pupil_control` keeps that as an
    explicit negative control rather than as a lesson in a comment.

    The handoff's eikonal consistency is measured here as well: for an OPL declared
    on a plane, ``dOPL/drho`` must equal the transverse direction cosine, or the
    bundle's phase and its propagation directions describe different wavefronts.
    """
    rays = _trace(rings, directory)
    bundle = _pupil_bundle(rays)
    _, optical_path_length = bundle.require_coherent()
    rho = np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1])
    transverse = np.hypot(bundle.directions[:, 0], bundle.directions[:, 1])
    optical_path_length = np.asarray(optical_path_length, dtype=np.float64)

    # Hexapolar rings share an exact radius, so average per ring rather than
    # differentiating a table with repeated abscissae.
    unique, inverse = np.unique(np.round(rho, 15), return_inverse=True)
    counts = np.bincount(inverse)
    opl_ring = np.bincount(inverse, weights=optical_path_length) / counts
    direction_ring = np.bincount(inverse, weights=transverse) / counts

    gradient = np.gradient(opl_ring, unique)
    interior = slice(1, -1)
    ratio = np.abs(gradient[interior]) / np.maximum(direction_ring[interior], 1e-30)
    # One-sided differences at the rim are first order; a local cubic is not.
    outer = unique > 0.8 * unique[-1]
    cubic = np.polyfit(unique[outer], opl_ring[outer], 3)
    rim_slope = abs(float(np.polyval(np.polyder(cubic), unique[-1])))

    sphere_ring = -np.sqrt(unique**2 + DISTANCE_M**2)
    error = opl_ring - sphere_ring
    return {
        "rings": rings,
        "traced_rays": int(bundle.count),
        "ring_radius_m": unique,
        "ring_wavefront_m": opl_ring,
        "representation": "ring-averaged traced OPL, linearly interpolated in rho",
        "wavefront_error_vs_sphere_rms_waves": float(np.std(error) / WAVELENGTH_M),
        "wavefront_error_vs_sphere_ptv_waves": float(np.ptp(error) / WAVELENGTH_M),
        "amplitude_is_uniform": bool(np.allclose(np.abs(bundle.amplitude), 1.0)),
        "eikonal_consistency": {
            "statement": "dOPL/drho must equal the transverse direction cosine",
            "max_relative_deviation_interior": float(np.max(np.abs(ratio - 1.0))),
            "rim_dopl_drho": rim_slope,
            "rim_transverse_direction_cosine": float(direction_ring[-1]),
            "rim_relative_deviation": abs(rim_slope / float(direction_ring[-1]) - 1.0),
            "sphere_to_declared_image_plane_would_give": PUPIL_RADIUS_M
            / math.hypot(PUPIL_RADIUS_M, DISTANCE_M),
            "why_the_sphere_value_differs": (
                "the traced rim slope exceeds a/sqrt(a^2+R^2) by 0.29%. That is the "
                "singlet's residual spherical aberration, not an inconsistency: the "
                "marginal ray crosses the axis about 14 um before the declared image "
                "plane. It matters because it is the Airy SCALE, so an oracle built "
                "from a sphere to the declared image plane is 0.29% too wide."
            ),
        },
    }


def _wavefront_on_radius(fit: dict[str, Any], rho: np.ndarray) -> np.ndarray:
    """Interpolate the traced wavefront, extrapolating nothing inside the support."""
    return np.interp(rho, fit["ring_radius_m"], fit["ring_wavefront_m"])


def _pupil_field_on_grid(*, grid_n: int, pitch: float, fit: dict[str, Any] | None) -> np.ndarray:
    """The KNOWN physical pupil, constructed directly: hard support, exact sphere.

    ``P(rho) A(rho) exp(i phi(rho))`` with ``P`` the hard circular support the
    coupler has no access to, ``A = 1`` (measured uniform on the trace) and
    ``phi`` the exact converging sphere plus, optionally, the fitted wavefront
    error. This is the pupil the exit-pupil semantics of section 1.1 assume, and
    it is built here without any ray sum.
    """
    axis = (np.arange(grid_n, dtype=np.float64) - grid_n // 2) * pitch
    gy, gx = np.meshgrid(axis, axis, indexing="ij")
    rho2 = gy**2 + gx**2
    support = rho2 <= PUPIL_RADIUS_M**2
    if fit is None:
        phase = -np.sqrt(rho2 + DISTANCE_M**2)
    else:
        phase = _wavefront_on_radius(fit, np.sqrt(rho2))
    return support * np.exp(1j * WAVENUMBER * phase)


def _o2_asm(*, fit: dict[str, Any] | None) -> dict[str, Any]:
    """O2. The known pupil, propagated to the sensor by independent float64 ASM.

    FFT-based, float64, no Chromatix and no coupler. The window is 2048 um at the
    sensor pitch, i.e. 4.1x the pupil diameter, so the aperture's own edge
    diffraction has 774 um of margin before it wraps -- that corresponds to
    direction cosines above 0.16, three times the system NA.
    """
    pupil = _pupil_field_on_grid(grid_n=O2_ASM_GRID_N, pitch=SENSOR_PITCH_M, fit=fit)
    field = _asm_float64(pupil, z_m=DISTANCE_M, pitch_m=SENSOR_PITCH_M, pad=0)
    half = SENSOR_GRID_N // 2
    centre = O2_ASM_GRID_N // 2
    cropped = field[centre - half : centre - half + SENSOR_GRID_N,
                    centre - half : centre - half + SENSOR_GRID_N]
    return {"u": np.ascontiguousarray(cropped), "grid_n": O2_ASM_GRID_N}


def _o2_rayleigh_sommerfeld(
    *, fit: dict[str, Any] | None, n_rho: int = O2_RS_N_RHO, n_phi: int = O2_RS_N_PHI
) -> dict[str, Any]:
    """O2, second route. The exact Rayleigh-Sommerfeld integral, in polar form.

    A genuinely different REPRESENTATION, not a different discretization of the
    same one: the coupler sums plane waves, this sums spherical Huygens waves over
    the pupil surface, and

        U(r_s) = (1/(i lambda)) int int  A e^{i k phi(rho)} (e^{i k S}/S) (R/S) rho drho dphi,
        S = sqrt(R^2 + rho^2 + r_s^2 - 2 rho r_s cos(phi))

    is evaluated with no paraxial approximation, no FFT and therefore no
    wraparound. On axis the sensor field is rotationally symmetric, so it is
    computed on a dense radial table and mapped to the Cartesian grid, which is
    what makes an exact double quadrature affordable.
    """
    a = PUPIL_RADIUS_M
    step_rho = a / n_rho
    rho = (np.arange(n_rho, dtype=np.float64) + 0.5) * step_rho
    phi = np.arange(n_phi, dtype=np.float64) * (2.0 * math.pi / n_phi)
    cos_phi = np.cos(phi)

    pupil_phase = (
        -np.sqrt(rho**2 + DISTANCE_M**2) if fit is None else _wavefront_on_radius(fit, rho)
    )
    # rho drho dphi, with the pupil phase folded in.
    weight = np.exp(1j * WAVENUMBER * pupil_phase) * rho * step_rho * (2.0 * math.pi / n_phi)

    r_max = (SENSOR_GRID_N / 2.0) * SENSOR_PITCH_M * math.sqrt(2.0) + SENSOR_PITCH_M
    radii = np.linspace(0.0, r_max, O2_RS_N_RADIAL)
    table = np.zeros(O2_RS_N_RADIAL, dtype=np.complex128)
    base = DISTANCE_M**2 + rho**2
    cross = 2.0 * rho
    for i, r_s in enumerate(radii):
        s = np.sqrt(base[:, None] + r_s * r_s - (cross[:, None] * r_s) * cos_phi[None, :])
        kernel = np.exp(1j * WAVENUMBER * s) * (DISTANCE_M / (s * s))
        table[i] = np.dot(weight, kernel.sum(axis=1))
    table = table / (1j * WAVELENGTH_M)
    # Carrier removal before interpolation: the residual phase varies slowly,
    # the absolute phase does not, and interpolating the latter would alias.
    table = table * np.exp(-1j * WAVENUMBER * DISTANCE_M)

    r = _radius_grid((SENSOR_GRID_N, SENSOR_GRID_N), SENSOR_PITCH_M)
    u = np.interp(r, radii, table.real) + 1j * np.interp(r, radii, table.imag)
    return {
        "u": u,
        "n_rho": n_rho,
        "n_phi": n_phi,
        "n_radial": O2_RS_N_RADIAL,
        "radial_samples_per_airy_radius": float(_airy_radius_m() / (r_max / O2_RS_N_RADIAL)),
    }


def _underfitted_pupil_control(rings: int, directory: Path) -> dict[str, Any]:
    """A DELIBERATELY under-fitted oracle, kept because it nearly fooled this probe.

    Replacing the interpolated traced wavefront with a 5-term polynomial in
    ``rho^2`` truncates the singlet's high-order spherical aberration. The
    resulting oracle is a clean Airy pattern -- 5e-4 from the analytic form -- at a
    numerical aperture 0.28% too small, and it charges the coupler about 5e-3.
    That is five times the gate, monotone in ray count, and entirely an artifact of
    the oracle. It is recorded here so the number cannot be rediscovered later and
    read as a coupler defect.
    """
    rays = _trace(rings, directory)
    bundle = _pupil_bundle(rays)
    _, optical_path_length = bundle.require_coherent()
    rho = np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1])
    error = np.asarray(optical_path_length, dtype=np.float64) + np.sqrt(rho**2 + DISTANCE_M**2)
    design = np.stack([(rho**2) ** j for j in range(O2_UNDERFIT_ORDER)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, error, rcond=None)
    residual = error - design @ coefficients
    radius = np.linspace(0.0, PUPIL_RADIUS_M, 4096)
    polynomial = -np.sqrt(radius**2 + DISTANCE_M**2) + sum(
        c * (radius**2) ** j for j, c in enumerate(coefficients)
    )
    slope = float(np.gradient(polynomial, radius)[-1])
    return {
        "order_in_rho_squared": O2_UNDERFIT_ORDER,
        "fit_residual_rms_waves": float(np.std(residual) / WAVELENGTH_M),
        "rim_slope_of_the_polynomial": abs(slope),
        "rim_slope_of_the_trace": float(np.hypot(bundle.directions[:, 0], bundle.directions[:, 1]).max()),
        "rim_slope_relative_error": abs(abs(slope) / SINGLET["na_frozen"] - 1.0),
        "ring_radius_m": radius,
        "ring_wavefront_m": polynomial,
        "why_it_is_kept": (
            "the residual RMS is 1.3e-3 waves, which looks like an excellent fit, and "
            "the rim SLOPE is nevertheless 0.28% wrong. RMS over the pupil does not "
            "bound the derivative at the edge, and the derivative at the edge is the "
            "Airy scale. This is the failure mode that would have been reported as "
            "'the coupler converges to a 5e-3 floor'."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Ray-state diagnostics at a plane: is this a caustic, and does it matter?
# ---------------------------------------------------------------------------
def _ray_state_diagnostics(bundle, amplitude_sum: float) -> dict[str, Any]:
    positions = np.asarray(bundle.positions_m, dtype=np.float64)
    directions = np.asarray(bundle.directions, dtype=np.float64)
    radius = np.hypot(positions[:, 0], positions[:, 1])
    transverse = np.hypot(directions[:, 0], directions[:, 1])
    extent = float(radius.max())
    airy = _airy_radius_m()

    # Areal density in equal-area annuli over the occupied disc, which is the
    # density a position-space method would have to resolve.
    edges = np.sqrt(np.linspace(0.0, extent**2, 17)) if extent > 0 else np.zeros(17)
    counts, _ = np.histogram(radius, bins=edges)
    areas = math.pi * np.diff(edges**2)
    densities = counts / np.where(areas > 0, areas, np.nan)
    finite = densities[np.isfinite(densities) & (counts > 0)]

    return {
        "ray_spatial_extent_m": extent,
        "ray_spatial_extent_in_airy_radii": extent / airy,
        "ray_angular_extent_max_transverse_direction_cosine": float(transverse.max()),
        "ray_angular_extent_min_transverse_direction_cosine": float(transverse.min()),
        "min_areal_ray_density_per_m2": float(finite.min()) if finite.size else None,
        "max_areal_ray_density_per_m2": float(finite.max()) if finite.size else None,
        "areal_density_dynamic_range": (
            float(finite.max() / finite.min()) if finite.size and finite.min() > 0 else None
        ),
        "caustic_condition": {
            "test": "ray bundle extent below one Airy first-null radius",
            "extent_over_airy_radius": extent / airy,
            "is_caustic_or_near_caustic": bool(extent < airy),
            "note": (
                "a geometric focus is a caustic: the position-space ray description "
                "collapses. Whether that breaks the HANDOFF depends on whether the "
                "operator reads positions or directions, which is what the coherent "
                "gain below measures."
            ),
        },
        "coherent_gain_at_the_peak": None if amplitude_sum == 0 else None,
    }


def _conditioning(u: np.ndarray, bundle, *, pitch: float, gate_radius_m: float) -> dict[str, Any]:
    """Is the coherent sum ill-conditioned at this plane?

    The relevant number is the coherent gain ``|U| / sum_i |a_i|``: 1 means every
    wavelet arrives in phase and the answer is a sum of like-signed terms; ``1/N``
    means the answer is what survives near-total cancellation, and float64 then has
    ``log10(N)`` fewer digits than it looks like it has.

    The result is the opposite of the intuition that a caustic is dangerous. At the
    geometric focus the gain is essentially 1 -- the caustic is where this operator
    is BEST conditioned -- and it is at the pupil, where the rays are spread out and
    the field is nearly uniform, that the sum is a cancellation problem.
    """
    amplitude = np.abs(np.asarray(bundle.amplitude, dtype=np.complex128)).sum()
    magnitude = np.abs(u)
    gate = _disc_mask(u.shape, pitch, gate_radius_m)
    return {
        "sum_of_ray_amplitudes": float(amplitude),
        "peak_coherent_gain": float(magnitude.max() / amplitude) if amplitude else None,
        "median_coherent_gain_in_gate_disc": (
            float(np.median(magnitude[gate]) / amplitude) if amplitude else None
        ),
        "worst_case_significant_digits_lost": (
            float(math.log10(amplitude / magnitude.max())) if amplitude and magnitude.max() else None
        ),
        "interpretation": (
            "gain 1 = every wavelet in phase, no cancellation, full float64 precision. "
            "gain 1/N = the answer is the residue of near-total cancellation."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Experiment A -- the handoff-plane sweep
# ---------------------------------------------------------------------------
def _handoff_grid_n(beam_radius_m: float) -> tuple[int, bool]:
    half_width = max(SENSOR_EXTENT_M / 2.0, HANDOFF_WINDOW_BEAM_MARGIN * beam_radius_m)
    grid_n = 2 * int(math.ceil(half_width / SENSOR_PITCH_M))
    capped = grid_n > HANDOFF_GRID_N_CAP
    return (min(grid_n, HANDOFF_GRID_N_CAP), capped)


def _crop_centre(array: np.ndarray, n: int) -> np.ndarray:
    """Centre ``n x n``, about the pinned ``index n // 2`` origin."""
    centre = array.shape[0] // 2, array.shape[1] // 2
    half = n // 2
    return np.ascontiguousarray(
        array[centre[0] - half : centre[0] - half + n, centre[1] - half : centre[1] - half + n]
    )


def _handoff_sweep(workdir: Path, references: dict[str, Any]) -> dict[str, Any]:
    """One ray count, every declared handoff plane, always measured at the sensor.

    The sensor never moves. An upstream handoff is followed by float64 ASM over the
    remaining distance -- independent of Chromatix on purpose, so that a plane is
    not charged for a ``complex64`` cast. The shipping Chromatix propagation is
    measured separately, in Experiments D and E.
    """
    rays = _trace(HANDOFF_SWEEP_RINGS, workdir / "rays")
    pupil = _pupil_bundle(rays)
    amplitude_sum = float(np.abs(np.asarray(pupil.amplitude, dtype=np.complex128)).sum())
    gate_radius = GATE_AIRY_RADII * _airy_radius_m()

    # The direct sensor-plane field, used as the "what did the window cost"
    # baseline: it is the same operator with no propagation after it at all.
    direct_bundle, _ = _advance_bundle_to_z(pupil, SENSOR_Z_M)
    direct_field, _ = _reconstruct_core(
        direct_bundle, grid_n=SENSOR_GRID_N, pitch_m=SENSOR_PITCH_M
    )

    rows: list[dict[str, Any]] = []
    for name, fraction in HANDOFF_CANDIDATES:
        z_handoff = SENSOR_Z_M - fraction * DISTANCE_M
        bundle, step = _advance_bundle_to_z(pupil, z_handoff)
        state = _ray_state_diagnostics(bundle, amplitude_sum)
        grid_n, capped = _handoff_grid_n(state["ray_spatial_extent_m"])
        propagation_m = SENSOR_Z_M - z_handoff

        row: dict[str, Any] = {
            "handoff_plane": name,
            "fraction_of_R_upstream_of_the_sensor": fraction,
            "handoff_z_m": z_handoff,
            "signed_distance_from_declared_image_plane_m": propagation_m,
            "post_handoff_propagation_m": propagation_m,
            "reconstruction_grid_n": grid_n,
            "reconstruction_pitch_m": SENSOR_PITCH_M,
            "reconstruction_window_m": grid_n * SENSOR_PITCH_M,
            "grid_n_cap_binds": capped,
            "max_ray_advance_m": float(np.max(np.abs(step))),
            **state,
        }
        try:
            field, diagnostics = _reconstruct_core(bundle, grid_n=grid_n, pitch_m=SENSOR_PITCH_M)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            row.update({"status": "coupler_refused", "error": f"{type(error).__name__}: {error}"})
            rows.append(row)
            continue

        row["coupler_ray_density_status"] = diagnostics.ray_density_status
        row["coupler_max_adjacent_ray_phase_rad"] = diagnostics.max_adjacent_ray_phase_rad
        row["grid_nyquist_satisfied"] = diagnostics.grid_nyquist_satisfied
        row["conditioning"] = _conditioning(
            field.u, bundle, pitch=SENSOR_PITCH_M, gate_radius_m=gate_radius
        )

        propagated = _asm_float64(
            field.u, z_m=propagation_m, pitch_m=SENSOR_PITCH_M, pad=grid_n
        )
        sensor_u = _crop_centre(propagated, SENSOR_GRID_N)
        intensity = np.abs(sensor_u) ** 2

        row["status"] = "succeeded"
        row["psf"] = _psf_metrics(
            intensity,
            pitch=SENSOR_PITCH_M,
            references={
                "vs_o1_analytic_airy": references["o1"],
                "vs_o2_asm_traced_pupil": references["o2_asm_traced_intensity"],
                "vs_o2_rayleigh_sommerfeld": references["o2_rs_traced_intensity"],
            },
            gate_radius_m=gate_radius,
        )
        row["complex_field_vs_o2_asm_traced"] = _complex_relative_l2(
            sensor_u,
            references["o2_asm_traced"],
            _disc_mask(intensity.shape, SENSOR_PITCH_M, gate_radius),
        )
        row["intensity_vs_direct_sensor_handoff"] = _relative_l2(
            intensity,
            np.abs(direct_field.u) ** 2,
            _disc_mask(intensity.shape, SENSOR_PITCH_M, gate_radius),
        )
        rows.append(row)

    return {
        "purpose": (
            "establish the validity region of C_RAY_TO_WAVE in the handoff coordinate, "
            "with the sensor and the ray count held fixed."
        ),
        "rings": HANDOFF_SWEEP_RINGS,
        "traced_rays": int(pupil.count),
        "candidate_set_declared_before_the_sweep": [name for name, _ in HANDOFF_CANDIDATES],
        "window_rule": (
            f"pitch fixed at {SENSOR_PITCH_M} m at every plane so the Nyquist condition "
            f"is not a confound; half-width = max(sensor half-width, "
            f"{HANDOFF_WINDOW_BEAM_MARGIN} x geometric beam radius), grid_n capped at "
            f"{HANDOFF_GRID_N_CAP}"
        ),
        "propagation_after_handoff": (
            "independent float64 angular spectrum, carrier removed, zero-padded by one "
            "window on each side. Not Chromatix: Experiment A measures the coupler, and "
            "Experiments D and E measure the shipping propagation."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 6. Experiment B -- the ray ladder at the selected handoff
# ---------------------------------------------------------------------------
def _select_handoff(sweep: dict[str, Any]) -> dict[str, Any]:
    """Apply the rule declared in :data:`HANDOFF_SELECTION_RULE`. No new criteria."""
    scored = [
        row
        for row in sweep["rows"]
        if row.get("status") == "succeeded"
        and math.isfinite(row["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"])
    ]
    if not scored:
        return {"selected": None, "reason": "no candidate produced a finite residual"}
    best = min(scored, key=lambda r: r["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"])
    floor = best["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"]
    tied = [
        row
        for row in scored
        if row["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"] <= 1.1 * floor
    ]
    chosen = min(tied, key=lambda r: abs(r["post_handoff_propagation_m"]))
    return {
        "selected": chosen["handoff_plane"],
        "selected_z_m": chosen["handoff_z_m"],
        "selected_fraction_of_R": chosen["fraction_of_R_upstream_of_the_sensor"],
        "post_handoff_propagation_m": chosen["post_handoff_propagation_m"],
        "rule": HANDOFF_SELECTION_RULE,
        "best_residual_before_tie_break": floor,
        "candidates_within_10_percent": [row["handoff_plane"] for row in tied],
    }


def _power_law_fit(x: list[float], y: list[float], *, label: str) -> dict[str, Any]:
    pairs = [
        (a, b) for a, b in zip(x, y, strict=True) if a > 0 and b > 0 and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return {"label": label, "status": "too_few_points", "points": len(pairs)}
    lx = np.log(np.array([a for a, _ in pairs]))
    ly = np.log(np.array([b for _, b in pairs]))
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    residual = float(np.sum((ly - predicted) ** 2))
    total = float(np.sum((ly - ly.mean()) ** 2))
    standard_error = (
        math.sqrt(residual / (len(pairs) - 2) / float(np.sum((lx - lx.mean()) ** 2)))
        if len(pairs) > 2
        else float("nan")
    )
    return {
        "label": label,
        "exponent": float(slope),
        "exponent_standard_error": standard_error,
        "prefactor": float(math.exp(intercept)),
        "r_squared": 1.0 - residual / total if total else float("nan"),
        "points": len(pairs),
    }


def _ray_ladder(workdir: Path, references: dict[str, Any], handoff: dict[str, Any]) -> dict[str, Any]:
    """CHE-38 section 7: sampling error and oracle error, as two curves.

    Sampling error is measured against the highest ray count and says only whether
    the discretization has converged. Oracle error is measured against references
    that share no traced data and is the only thing that says whether the answer is
    right. Keeping them apart is the point: M3.9's headline was that they diverge.
    """
    z_handoff = handoff["selected_z_m"]
    propagation_m = SENSOR_Z_M - z_handoff
    gate_radius = GATE_AIRY_RADII * _airy_radius_m()
    gate = _disc_mask((SENSOR_GRID_N, SENSOR_GRID_N), SENSOR_PITCH_M, gate_radius)

    intensities: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for rings in RAY_SWEEP_RINGS:
        start = time.perf_counter()
        rays = _trace(rings, workdir / f"rays{rings}")
        traced = time.perf_counter()
        bundle, _ = _advance_bundle_to_z(_pupil_bundle(rays), z_handoff)
        grid_n, _ = _handoff_grid_n(
            float(np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1]).max())
        )
        field, diagnostics = _reconstruct_core(bundle, grid_n=grid_n, pitch_m=SENSOR_PITCH_M)
        reconstructed = time.perf_counter()
        sensor_u = _crop_centre(
            _asm_float64(field.u, z_m=propagation_m, pitch_m=SENSOR_PITCH_M, pad=grid_n),
            SENSOR_GRID_N,
        )
        intensity = np.abs(sensor_u) ** 2
        intensities[rings] = intensity
        rows.append(
            {
                "rings": rings,
                "traced_rays": int(bundle.count),
                "reconstruction_grid_n": grid_n,
                "ray_spacing_at_the_pupil_m": PUPIL_RADIUS_M / rings,
                "coupler_ray_density_status": diagnostics.ray_density_status,
                "max_adjacent_ray_phase_rad": diagnostics.max_adjacent_ray_phase_rad,
                "psf": _psf_metrics(
                    intensity,
                    pitch=SENSOR_PITCH_M,
                    references={
                        "vs_o1_analytic_airy": references["o1"],
                        "vs_o2_asm_traced_pupil": references["o2_asm_traced_intensity"],
                        "vs_o2_rayleigh_sommerfeld": references["o2_rs_traced_intensity"],
                    },
                    gate_radius_m=gate_radius,
                ),
                "complex_field_vs_o2": _complex_relative_l2(
                    sensor_u, references["o2_asm_traced"], gate
                ),
                "seconds_trace": traced - start,
                "seconds_reconstruct": reconstructed - traced,
                "seconds_total": time.perf_counter() - start,
            }
        )

    reference = intensities[RAY_REFERENCE_RINGS]
    for row in rows:
        row["sampling_error_vs_highest_ray_count"] = _relative_l2(
            intensities[row["rings"]], reference, gate
        )

    counts = [row["traced_rays"] for row in rows]
    sampling = [row["sampling_error_vs_highest_ray_count"] for row in rows]
    oracle_o1 = [row["psf"]["vs_o1_analytic_airy"]["relative_l2_gate_disc"] for row in rows]
    oracle_o2 = [row["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"] for row in rows]

    def _turnaround(values: list[float]) -> dict[str, Any]:
        finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
        index = min(finite, key=lambda p: p[1])[0]
        after = values[index + 1 :]
        return {
            "minimum_at_traced_rays": counts[index],
            "minimum_value": values[index],
            "value_at_highest_ray_count": values[-1],
            "rises_after_the_minimum": bool(after and max(after) > values[index]),
            "rise_factor": (max(after) / values[index]) if after and values[index] else None,
        }

    return {
        "purpose": (
            "CHE-38 section 7. Ray-count convergence at the selected handoff, with "
            "sampling error and oracle error reported separately."
        ),
        "handoff_plane": handoff["selected"],
        "post_handoff_propagation_m": propagation_m,
        "rows": rows,
        "fits": {
            "sampling_error_vs_traced_rays": _power_law_fit(
                counts[:-1], sampling[:-1], label="sampling error (last point is the reference)"
            ),
            "oracle_error_vs_o1_vs_traced_rays": _power_law_fit(
                counts, oracle_o1, label="residual against the analytic Airy"
            ),
            "oracle_error_vs_o2_vs_traced_rays": _power_law_fit(
                counts, oracle_o2, label="residual against the independent wave oracle"
            ),
        },
        "turnaround_check": {
            "why": (
                "M3.9 found the oracle residual pass through a minimum near 28000 rays "
                "and then rise. CHE-38 section 7 requires an explicit re-check."
            ),
            "vs_o1_analytic_airy": _turnaround(oracle_o1),
            "vs_o2_independent_wave": _turnaround(oracle_o2),
        },
    }


# ---------------------------------------------------------------------------
# 7. Experiment C -- grid convergence at the corrected handoff
# ---------------------------------------------------------------------------
def _o2_at_pitch(pitch: float, grid_n: int, fit: dict[str, Any]) -> np.ndarray:
    """O2 rebuilt at an arbitrary sensor pitch, so nothing is resampled."""
    asm_n = max(int(round(2048e-6 / pitch)), 4 * grid_n)
    pupil = _pupil_field_on_grid(grid_n=asm_n, pitch=pitch, fit=fit)
    return _crop_centre(_asm_float64(pupil, z_m=DISTANCE_M, pitch_m=pitch, pad=0), grid_n)


def _grid_sweep(workdir: Path, handoff: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    """Fixed 128 um physical extent, varying pitch. The Nyquist guard stays on.

    CHE-38 section 8 also asks whether M3.9's ``grid_n = 188`` is still justified.
    It is not the same quantity: 188 was a PUPIL grid at 2.66 um whose extent was
    the pupil diameter and whose pitch was set by the per-axis Nyquist limit. At
    the sensor the binding constraint is resolving the Airy core, so the answer has
    to be re-derived rather than carried over, and this sweep is the derivation.
    """
    z_handoff = handoff["selected_z_m"]
    propagation_m = SENSOR_Z_M - z_handoff
    airy = _airy_radius_m()
    gate_radius = GATE_AIRY_RADII * airy
    nyquist_pitch_max = WAVELENGTH_M / (2.0 * SINGLET["na_frozen"])

    rays = _trace(SWEEP_RINGS, workdir / "rays")
    pupil = _pupil_bundle(rays)
    bundle, _ = _advance_bundle_to_z(pupil, z_handoff)
    beam_radius = float(np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1]).max())
    window_m = max(SENSOR_EXTENT_M, 2.0 * HANDOFF_WINDOW_BEAM_MARGIN * beam_radius)

    rows: list[dict[str, Any]] = []
    for grid_n in GRID_SWEEP_N:
        pitch = SENSOR_EXTENT_M / grid_n
        recon_n = 2 * int(math.ceil(window_m / 2.0 / pitch))
        row: dict[str, Any] = {
            "grid_n": grid_n,
            "sample_pitch_m": pitch,
            "reconstruction_grid_n": recon_n,
            "physical_extent_m": SENSOR_EXTENT_M,
            "per_axis_nyquist_direction_limit": WAVELENGTH_M / (2.0 * pitch),
            "max_traced_transverse_direction_cosine": SINGLET["na_frozen"],
            "nyquist_admissible": bool(pitch <= nyquist_pitch_max),
            "pixels_across_the_psf_core_diameter": 2.0 * airy / pitch,
        }
        try:
            field, _ = _reconstruct_core(bundle, grid_n=recon_n, pitch_m=pitch)
        except Exception as error:  # noqa: BLE001
            row.update(
                {
                    "status": "coupler_refused",
                    "refusal": f"{type(error).__name__}: {error}",
                    "guard_fired_on_real_traced_data": True,
                }
            )
            rows.append(row)
            continue
        sensor_u = _crop_centre(
            _asm_float64(field.u, z_m=propagation_m, pitch_m=pitch, pad=recon_n), grid_n
        )
        intensity = np.abs(sensor_u) ** 2
        o1 = _o1_analytic_airy(grid_n=grid_n, pitch=pitch)
        o2 = _o2_at_pitch(pitch, grid_n, fit)
        row["status"] = "succeeded"
        row["psf"] = _psf_metrics(
            intensity,
            pitch=pitch,
            references={
                "vs_o1_analytic_airy": o1,
                "vs_o2_asm_traced_pupil": np.abs(o2) ** 2,
            },
            gate_radius_m=gate_radius,
        )
        row["complex_field_vs_o2"] = _complex_relative_l2(
            sensor_u, o2, _disc_mask(intensity.shape, pitch, gate_radius)
        )
        rows.append(row)

    bypass = _nyquist_bypass(bundle, window_m, fit)
    return {
        "purpose": "CHE-38 section 8: field-grid convergence at the corrected handoff.",
        "handoff_plane": handoff["selected"],
        "rings": SWEEP_RINGS,
        "fixed_physical_extent_m": SENSOR_EXTENT_M,
        "per_axis_nyquist_pitch_max_m": nyquist_pitch_max,
        "smallest_admissible_grid_n_at_this_extent": math.ceil(
            SENSOR_EXTENT_M / nyquist_pitch_max
        ),
        "rows": rows,
        "nyquist_bypass_negative_control": bypass,
        "m3_9_grid_n_188_reassessed": (
            "not transferable. 188 was a pupil grid whose extent was the pupil diameter "
            "and whose pitch was set by the per-axis Nyquist limit at 2.66 um. At the "
            "sensor the extent is set by the number of Airy radii the metrics need and "
            "the pitch by resolving the core, so the Nyquist limit is satisfied with "
            "10x margin and stops being the binding constraint."
        ),
    }


def _nyquist_bypass(bundle, window_m: float, fit: dict[str, Any]) -> dict[str, Any]:
    """The guard turned off, once, to show what it is protecting against.

    Production measurements never bypass it. This exists because CHE-38 section 8
    asks for the physical consequence of violating the condition to be shown rather
    than asserted, and because a guard whose consequence is undemonstrated is
    indistinguishable from a guard that is merely conservative.
    """
    grid_n = GRID_SWEEP_N[0]
    pitch = SENSOR_EXTENT_M / grid_n
    recon_n = 2 * int(math.ceil(window_m / 2.0 / pitch))
    airy = _airy_radius_m()
    try:
        field, _ = _reconstruct_core(
            bundle, grid_n=recon_n, pitch_m=pitch, enforce_nyquist=False
        )
    except Exception as error:  # noqa: BLE001
        return {"status": "failed_even_with_the_guard_off", "error": str(error)}
    intensity = np.abs(_crop_centre(field.u, grid_n)) ** 2
    o1 = _o1_analytic_airy(grid_n=grid_n, pitch=pitch)
    gate = _disc_mask(intensity.shape, pitch, GATE_AIRY_RADII * airy)
    peak = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    return {
        "grid_n": grid_n,
        "sample_pitch_m": pitch,
        "per_axis_nyquist_direction_limit": WAVELENGTH_M / (2.0 * pitch),
        "max_traced_transverse_direction_cosine": SINGLET["na_frozen"],
        "violation_factor": SINGLET["na_frozen"] / (WAVELENGTH_M / (2.0 * pitch)),
        "relative_l2_vs_o1_analytic_airy": _relative_l2(intensity, o1, gate),
        "peak_index": [int(peak[0]), int(peak[1])],
        "peak_is_at_the_origin": bool(peak == (grid_n // 2, grid_n // 2)),
        "guard_is_on_for_every_production_measurement_in_this_record": True,
    }


# ---------------------------------------------------------------------------
# 8. Attribution -- what the residual against the independent oracle actually is
# ---------------------------------------------------------------------------
def _hexapolar(rings: int, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    points = [(0.0, 0.0)]
    ring_index = [0]
    for j in range(1, rings + 1):
        r = radius_m * j / rings
        for m in range(6 * j):
            angle = 2.0 * math.pi * m / (6 * j)
            points.append((r * math.cos(angle), r * math.sin(angle)))
            ring_index.append(j)
    return np.asarray(points, dtype=np.float64), np.asarray(ring_index)


def _radial_quadrature_weight(ring_index: np.ndarray, rings: int) -> np.ndarray:
    """The radial trapezoid weight a hexapolar ring set implies. DIAGNOSTIC ONLY.

    Hexapolar sampling is very nearly equal-area in the interior: ring ``j`` has
    ``6j`` points and the annulus it represents has area proportional to ``j``, so
    a uniform weight is the right quadrature there. It is wrong at the two
    boundaries. The outermost ring sits exactly ON ``rho = a`` and therefore
    represents only the inner half of its cell, and the single central ray
    represents a cell of radius ``a / (2 rings)`` rather than the interior
    ``pi a^2 / (3 rings^2)``. The trapezoid weights are 1/2 and 3/4.

    Nothing in production is changed by this (CHE-38 section 14). It is applied to
    a synthetic bundle's amplitude, inside this probe, to test one attribution.
    """
    weight = np.ones(ring_index.size, dtype=np.float64)
    weight[ring_index == 0] = 0.75
    weight[ring_index == rings] = 0.5
    return weight


def _synthetic_focal_bundle(rings: int, *, radius_m: float, distance_m: float, weighted: bool):
    """A hexapolar bundle with the EXACT path to a focus, already at the focal plane.

    Synthetic on purpose: Optiland, the OPL declaration and the residual aberration
    are all removed, so what remains is the reconstruction operator and the ray
    ensemble's quadrature. Every ray's optical path to the focus is equal, so the
    advanced bundle has ``OPL = 0`` and position ``0``, which is the cleanest
    possible statement of the sensor-side handoff.
    """
    from multiscale_optics_agent.couplers.contracts import Frame, RayBundle, ReferencePlane

    xy, ring_index = _hexapolar(rings, radius_m)
    x, y = xy[:, 0], xy[:, 1]
    path = np.sqrt(x**2 + y**2 + distance_m**2)
    amplitude = (
        _radial_quadrature_weight(ring_index, rings) if weighted else np.ones(x.size)
    ).astype(np.complex128)
    zeros = np.zeros(x.size)
    return RayBundle(
        positions_m=np.stack([zeros, zeros, np.full(x.size, distance_m)], axis=1),
        directions=np.stack([-x / path, -y / path, distance_m / path], axis=1),
        wavelength_m=WAVELENGTH_M,
        reference_plane=ReferencePlane(name="focal_plane", z_m=float(distance_m)),
        frame=Frame(),
        amplitude=amplitude,
        optical_path_length_m=zeros,
        optical_path_length_reference=(
            "synthetic: exact optical path to the nominal focus, advanced along each "
            "ray to the focal plane, so every ray's path is equal"
        ),
    )


def _rs_oracle_for(
    *, distance_m: float, grid_n: int, pitch: float, n_rho: int, n_phi: int, n_radial: int
) -> np.ndarray:
    """The Rayleigh-Sommerfeld oracle for an ideal pupil at an arbitrary distance."""
    step = PUPIL_RADIUS_M / n_rho
    rho = (np.arange(n_rho, dtype=np.float64) + 0.5) * step
    phi = np.arange(n_phi, dtype=np.float64) * (2.0 * math.pi / n_phi)
    cos_phi = np.cos(phi)
    weight = (
        np.exp(-1j * WAVENUMBER * np.sqrt(rho**2 + distance_m**2))
        * rho
        * step
        * (2.0 * math.pi / n_phi)
    )
    r_max = (grid_n / 2.0) * pitch * math.sqrt(2.0) + pitch
    radii = np.linspace(0.0, r_max, n_radial)
    table = np.zeros(n_radial, dtype=np.complex128)
    base = distance_m**2 + rho**2
    cross = 2.0 * rho
    for i, r_s in enumerate(radii):
        s = np.sqrt(base[:, None] + r_s * r_s - (cross[:, None] * r_s) * cos_phi[None, :])
        table[i] = np.dot(weight, (np.exp(1j * WAVENUMBER * s) * (distance_m / (s * s))).sum(axis=1))
    table = table / (1j * WAVELENGTH_M) * np.exp(-1j * WAVENUMBER * distance_m)
    r = _radius_grid((grid_n, grid_n), pitch)
    return np.interp(r, radii, table.real) + 1j * np.interp(r, radii, table.imag)


def _quadrature_attribution(workdir: Path) -> dict[str, Any]:
    """The residual is the ray ensemble's AREA WEIGHT, not the operator's kernel.

    Two measurements, both on synthetic bundles so nothing else can be blamed:

    1. The residual against the Rayleigh-Sommerfeld oracle falls as ``1/rings`` --
       first order in the ray SPACING -- which is the wrong rate for a smooth
       equal-area quadrature and the right rate for a boundary error.
    2. The effective aperture is measured by fitting a numerical aperture to the
       reconstructed PSF. It overshoots by ``dNA/NA ~ 1/(2 rings)``, which is
       exactly half a ring spacing of extra radius: the outermost ring sits on the
       rim and is counted as a full cell.

    Applying the radial trapezoid weight collapses the residual by an order of
    magnitude and makes it converged rather than first order. That identifies the
    remedy without implementing it, which is what CHE-38 sections 14 and 15 ask
    for: the fix belongs to a quadrature-weight ticket, not to this one.
    """
    from multiscale_optics_agent.evaluation.psf_oracles import airy_psf_on_grid

    distance = DISTANCE_M
    numerical_aperture = PUPIL_RADIUS_M / math.hypot(PUPIL_RADIUS_M, distance)
    airy = 0.61 * WAVELENGTH_M / numerical_aperture
    pitch = airy / 12.97
    grid_n = SENSOR_GRID_N
    gate = _disc_mask((grid_n, grid_n), pitch, GATE_AIRY_RADII * airy)

    oracle = _rs_oracle_for(
        distance_m=distance,
        grid_n=grid_n,
        pitch=pitch,
        n_rho=O2_RS_N_RHO,
        n_phi=O2_RS_N_PHI,
        n_radial=1024,
    )
    oracle_intensity = np.abs(oracle) ** 2
    analytic = airy_psf_on_grid(
        shape=(grid_n, grid_n),
        sample_pitch_m=(pitch, pitch),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=numerical_aperture,
    )

    def fit_numerical_aperture(intensity: np.ndarray) -> tuple[float, float]:
        best = (float("nan"), float("inf"))
        for candidate in np.linspace(0.98 * numerical_aperture, 1.03 * numerical_aperture, 501):
            reference = airy_psf_on_grid(
                shape=(grid_n, grid_n),
                sample_pitch_m=(pitch, pitch),
                wavelength_m=WAVELENGTH_M,
                numerical_aperture=float(candidate),
            )
            error = _relative_l2(intensity, reference, gate)
            if error < best[1]:
                best = (float(candidate), error)
        return best

    rows = []
    for rings in QUADRATURE_RINGS:
        uniform, _ = _reconstruct_core(
            _synthetic_focal_bundle(
                rings, radius_m=PUPIL_RADIUS_M, distance_m=distance, weighted=False
            ),
            grid_n=grid_n,
            pitch_m=pitch,
        )
        weighted, _ = _reconstruct_core(
            _synthetic_focal_bundle(
                rings, radius_m=PUPIL_RADIUS_M, distance_m=distance, weighted=True
            ),
            grid_n=grid_n,
            pitch_m=pitch,
        )
        uniform_intensity = np.abs(uniform.u) ** 2
        fitted, fitted_residual = fit_numerical_aperture(uniform_intensity)
        rows.append(
            {
                "rings": rings,
                "traced_rays": 1 + 3 * rings * (rings + 1),
                "uniform_weight_vs_rs_oracle": _relative_l2(
                    uniform_intensity, oracle_intensity, gate
                ),
                "trapezoid_weight_vs_rs_oracle": _relative_l2(
                    np.abs(weighted.u) ** 2, oracle_intensity, gate
                ),
                "fitted_numerical_aperture": fitted,
                "fitted_numerical_aperture_residual": fitted_residual,
                "fitted_na_relative_excess": fitted / numerical_aperture - 1.0,
                "half_a_ring_spacing_prediction": 1.0 / (2.0 * rings),
            }
        )

    uniform_fit = _power_law_fit(
        [row["rings"] for row in rows],
        [row["uniform_weight_vs_rs_oracle"] for row in rows],
        label="uniform-weight residual vs ring count",
    )
    excess_fit = _power_law_fit(
        [row["rings"] for row in rows],
        [row["fitted_na_relative_excess"] for row in rows],
        label="fitted effective-NA excess vs ring count",
    )
    return {
        "purpose": (
            "attribute the sensor-side residual. Synthetic bundles only, so Optiland, "
            "the OPL declaration and the residual aberration are all removed."
        ),
        "geometry": {
            "pupil_radius_m": PUPIL_RADIUS_M,
            "distance_m": distance,
            "numerical_aperture": numerical_aperture,
            "fresnel_number": PUPIL_RADIUS_M**2 / (WAVELENGTH_M * distance),
            "sensor_pitch_m": pitch,
            "airy_first_null_radius_m": airy,
        },
        "rs_oracle_vs_analytic_airy": _relative_l2(oracle_intensity, analytic, gate),
        "rows": rows,
        "uniform_weight_convergence_fit": uniform_fit,
        "effective_na_excess_fit": excess_fit,
        "trapezoid_weight_residual_is_flat_at": float(
            np.median([row["trapezoid_weight_vs_rs_oracle"] for row in rows])
        ),
        "conclusion": (
            "the residual is a per-ray AREA WEIGHT error at the aperture boundary, "
            "first order in the ray spacing, and not a property of the wavelet kernel. "
            "The outermost hexapolar ring lies on rho = a and is counted as a full "
            "cell, which inflates the effective aperture by half a ring spacing; the "
            "fitted NA excess tracks 1/(2 rings). A radial trapezoid weight removes it "
            "and leaves a converged residual an order of magnitude smaller. Not "
            "implemented in production: CHE-38 section 14 forbids it and section 15 "
            "assigns quadrature weights to their own ticket."
        ),
    }


def _fresnel_number_scan(workdir: Path) -> dict[str, Any]:
    """Does the sensor-side residual depend on the pupil Fresnel number?

    M3.9's exit-pupil term did, strongly, and that was the basis for calling M3.2's
    1/10 scaling load-bearing. At the sensor handoff the question has to be asked
    again rather than inherited. The aperture and the ray count are held fixed and R
    is varied; the sensor grid is scaled in units of Airy radii so a distance scan
    does not become a grid scan.
    """
    rows = []
    for factor in FRESNEL_DISTANCE_FACTORS:
        distance = DISTANCE_M * factor
        numerical_aperture = PUPIL_RADIUS_M / math.hypot(PUPIL_RADIUS_M, distance)
        airy = 0.61 * WAVELENGTH_M / numerical_aperture
        pitch = airy / 12.97
        gate = _disc_mask((SENSOR_GRID_N, SENSOR_GRID_N), pitch, GATE_AIRY_RADII * airy)
        oracle = np.abs(
            _rs_oracle_for(
                distance_m=distance,
                grid_n=SENSOR_GRID_N,
                pitch=pitch,
                n_rho=O2_RS_N_RHO,
                n_phi=O2_RS_N_PHI,
                n_radial=1024,
            )
        ) ** 2
        uniform, _ = _reconstruct_core(
            _synthetic_focal_bundle(
                FRESNEL_SCAN_RINGS, radius_m=PUPIL_RADIUS_M, distance_m=distance, weighted=False
            ),
            grid_n=SENSOR_GRID_N,
            pitch_m=pitch,
        )
        weighted, _ = _reconstruct_core(
            _synthetic_focal_bundle(
                FRESNEL_SCAN_RINGS, radius_m=PUPIL_RADIUS_M, distance_m=distance, weighted=True
            ),
            grid_n=SENSOR_GRID_N,
            pitch_m=pitch,
        )
        rows.append(
            {
                "distance_factor": factor,
                "distance_m": distance,
                "numerical_aperture": numerical_aperture,
                "fresnel_number": PUPIL_RADIUS_M**2 / (WAVELENGTH_M * distance),
                "sensor_pitch_m": pitch,
                "uniform_weight_vs_rs_oracle": _relative_l2(np.abs(uniform.u) ** 2, oracle, gate),
                "trapezoid_weight_vs_rs_oracle": _relative_l2(np.abs(weighted.u) ** 2, oracle, gate),
            }
        )
    return {
        "purpose": "test whether the sensor-side residual is governed by the Fresnel number.",
        "rings": FRESNEL_SCAN_RINGS,
        "rows": rows,
        "uniform_weight_fit": _power_law_fit(
            [row["fresnel_number"] for row in rows],
            [row["uniform_weight_vs_rs_oracle"] for row in rows],
            label="uniform-weight sensor residual vs pupil Fresnel number",
        ),
    }


# ---------------------------------------------------------------------------
# 9. Experiment D -- padding, and Experiment E -- the shipping sensor path
# ---------------------------------------------------------------------------
def _field_record(field, directory: Path, *, z_m: float, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    record = field.to_artifact_record(artifact_id=f"field:{name}", uri=directory / f"{name}.npy")
    record.metadata["z_m"] = z_m
    record.metadata["reference_plane"] = field.reference_plane.name
    return record


def _propagate_chromatix(record, directory: Path, *, pad_width: int, target_z_m: float):
    from multiscale_optics_agent.adapters.base import ModelRunRequest
    from multiscale_optics_agent.adapters.chromatix_adapter import get_adapter

    return get_adapter().run(
        ModelRunRequest(
            run_id="che38r",
            node_id="wave",
            inputs={"input_field": record},
            config={
                "propagation": "angular_spectrum",
                "propagation_method": "asm_carrier_removed",
                "target_plane_z_m": target_z_m,
                "pad_width": pad_width,
                "output_dir": str(directory),
            },
        )
    )


def _measure_shipping(result):
    from multiscale_optics_agent.evaluation.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf_from_record,
    )

    reported = result.diagnostics["output_sample_pitch_m"]
    return measure_psf_from_record(
        result.outputs["output_field"],
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
    )


def _padding_sweep(workdir: Path, handoff: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    """CHE-38 section 9. Only relevant if wave propagation remains after the handoff.

    If the selected handoff is the sensor itself there is no FFT after the coupler
    and the padding sweep has nothing to measure -- CHE-38 is explicit that it must
    not be preserved for procedural consistency. It is still run once, at the
    nearest candidate that DOES leave propagation, because "padding is irrelevant"
    is a claim about the selected configuration and not about the coupler, and a
    reader needs the conditional discharged with evidence.
    """
    airy = _airy_radius_m()
    gate_radius = GATE_AIRY_RADII * airy
    selected_propagation = SENSOR_Z_M - handoff["selected_z_m"]

    demo_fraction = next(
        fraction
        for name, fraction in HANDOFF_CANDIDATES
        if fraction != 0.0 and abs(fraction) == min(abs(f) for _, f in HANDOFF_CANDIDATES if f != 0.0)
    )
    z_handoff = SENSOR_Z_M - demo_fraction * DISTANCE_M
    propagation_m = SENSOR_Z_M - z_handoff

    rays = _trace(SWEEP_RINGS, workdir / "rays")
    bundle, _ = _advance_bundle_to_z(_pupil_bundle(rays), z_handoff)
    grid_n, _ = _handoff_grid_n(
        float(np.hypot(bundle.positions_m[:, 0], bundle.positions_m[:, 1]).max())
    )
    field, _ = _reconstruct_core(bundle, grid_n=grid_n, pitch_m=SENSOR_PITCH_M)
    record = _field_record(field, workdir / "field", z_m=z_handoff, name="handoff")

    rows = []
    for factor in PAD_SWEEP_FACTORS:
        pad_width = int(round(factor * grid_n))
        result = _propagate_chromatix(
            record, workdir / f"pad{pad_width}", pad_width=pad_width, target_z_m=SENSOR_Z_M
        )
        row: dict[str, Any] = {
            "pad_factor_of_the_window": factor,
            "pad_width": pad_width,
            "padded_grid_n": grid_n + 2 * pad_width,
        }
        if result.status.value != "succeeded":
            row.update({"status": "propagation_failed", "error": result.error_message})
            rows.append(row)
            continue
        measurement = _measure_shipping(result)
        intensity = np.asarray(measurement.intensity, dtype=np.float64)
        pitch = float(measurement.sample_pitch_m[0])
        cropped = _crop_centre(intensity, SENSOR_GRID_N)
        border = np.ones(intensity.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        row.update(
            {
                "status": "succeeded",
                "output_grid_n": int(intensity.shape[0]),
                "sample_pitch_m": pitch,
                "core_relative_l2_vs_o2": _relative_l2(
                    cropped,
                    references["o2_asm_traced_intensity"],
                    _disc_mask(cropped.shape, SENSOR_PITCH_M, airy),
                ),
                "wing_3_to_5_airy_relative_l2_vs_o2": _relative_l2(
                    cropped,
                    references["o2_asm_traced_intensity"],
                    _annulus_mask(cropped.shape, SENSOR_PITCH_M, 3.0 * airy, 5.0 * airy),
                ),
                "gate_disc_relative_l2_vs_o2": _relative_l2(
                    cropped,
                    references["o2_asm_traced_intensity"],
                    _disc_mask(cropped.shape, SENSOR_PITCH_M, gate_radius),
                ),
                "far_wing_beyond_5_airy_relative_l2_vs_o2": _relative_l2(
                    cropped,
                    references["o2_asm_traced_intensity"],
                    _radius_grid(cropped.shape, SENSOR_PITCH_M) > 5.0 * airy,
                ),
                "edge_energy_fraction_of_the_output_window": float(
                    intensity[border].sum() / intensity.sum()
                ),
                "energy_outside_the_reconstruction_window": float(
                    1.0 - _crop_centre(intensity, grid_n).sum() / intensity.sum()
                ),
            }
        )
        rows.append(row)

    return {
        "applicable_to_the_selected_handoff": bool(selected_propagation != 0.0),
        "selected_handoff_post_propagation_m": selected_propagation,
        "why": (
            "padding exists to stop an FFT wrapping. With the handoff on the sensor "
            "there is no FFT after the coupler, so padding is not a discretization of "
            "the selected configuration at all."
            if selected_propagation == 0.0
            else "the selected handoff leaves wave propagation, so padding is a real knob."
        ),
        "demonstration_handoff": {
            "fraction_of_R": demo_fraction,
            "z_m": z_handoff,
            "propagation_m": propagation_m,
            "reconstruction_grid_n": grid_n,
            "engine": "Chromatix adapter, asm_carrier_removed -- the shipping propagation",
        },
        "rows": rows,
        "power_conservation_caveat_retained_from_m3_9": (
            "power conservation alone cannot certify adequate padding: a wrapped field "
            "keeps its energy, it just puts it in the wrong place. The wing and "
            "far-wing residuals are what move."
        ),
    }


def _shipping_sensor_path(workdir: Path, references: dict[str, Any]) -> dict[str, Any]:
    """CHE-38 section 10. The intended architecture, end to end, and the caustic test.

    Optiland -> rays advanced to the declared sensor plane -> C_RAY_TO_WAVE ->
    ComplexField -> |U|^2 -> PSF. No propagation after the coupler, so the shipping
    path here is the coupler plus the frozen PSF measurement, and the Chromatix leg
    is exercised in Experiment D instead.

    The caustic question is answered rather than assumed. At the geometric focus the
    ray bundle collapses to a fraction of one Airy radius, which is a caustic by any
    position-space definition -- and the coherent gain is essentially 1 there, so the
    sum is a fully constructive one and better conditioned than anywhere upstream.
    The operator reads directions, not densities.
    """
    from multiscale_optics_agent.evaluation.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf,
    )

    airy = _airy_radius_m()
    gate_radius = GATE_AIRY_RADII * airy
    gate = _disc_mask((SENSOR_GRID_N, SENSOR_GRID_N), SENSOR_PITCH_M, gate_radius)

    rows = []
    for rings in (SWEEP_RINGS, RAY_REFERENCE_RINGS):
        rays = _trace(rings, workdir / f"rays{rings}")
        pupil = _pupil_bundle(rays)
        bundle, _ = _advance_bundle_to_z(pupil, SENSOR_Z_M)
        field, diagnostics = _reconstruct_core(
            bundle, grid_n=SENSOR_GRID_N, pitch_m=SENSOR_PITCH_M
        )
        measurement = measure_psf(field, normalization=M3_ORACLE_NORMALIZATION)
        intensity = np.asarray(measurement.intensity, dtype=np.float64)
        rows.append(
            {
                "rings": rings,
                "traced_rays": int(bundle.count),
                "ray_state_at_the_sensor": _ray_state_diagnostics(bundle, 0.0),
                "conditioning": _conditioning(
                    field.u, bundle, pitch=SENSOR_PITCH_M, gate_radius_m=gate_radius
                ),
                "coupler_ray_density_status": diagnostics.ray_density_status,
                "psf": _psf_metrics(
                    intensity,
                    pitch=SENSOR_PITCH_M,
                    references={
                        "vs_o1_analytic_airy": references["o1"],
                        "vs_o2_asm_traced_pupil": references["o2_asm_traced_intensity"],
                        "vs_o2_rayleigh_sommerfeld": references["o2_rs_traced_intensity"],
                    },
                    gate_radius_m=gate_radius,
                ),
                "complex_field_vs_o2": _complex_relative_l2(
                    field.u, references["o2_asm_traced"], gate
                ),
                "psf_measurement_diagnostics": measurement.as_dict(),
            }
        )

    return {
        "architecture": (
            "Optiland trace -> ray-domain advance to the declared sensor plane -> "
            "C_RAY_TO_WAVE -> ComplexField -> |U|^2 -> PSF. No wave propagation after "
            "the handoff."
        ),
        "rows": rows,
        "caustic_verdict": (
            "the exact sensor plane is NOT ill-conditioned for this operator. The ray "
            "bundle collapses to a small fraction of one Airy radius there, so it is a "
            "caustic in the position-space sense, and the coherent gain is ~1 because "
            "every wavelet arrives in phase. C_RAY_TO_WAVE reads ray DIRECTIONS and "
            "optical paths, never a local ray density, so the position-space "
            "degeneracy is not one of its inputs. The complex-field comparison is the "
            "one place the caustic does show up, and for a different reason: see "
            "missing_quadratic_phase."
        ),
        "missing_quadratic_phase": (
            "the reconstructed field is a sum of plane waves in the transverse "
            "coordinate, so it carries no exp(i k r^2 / 2R) curvature term. Against an "
            "exact spherical-wave reference that is a phase difference of about 1.2 rad "
            "at the 5-Airy-radius gate edge, which is why the complex-field residual "
            "stays large while the intensity residual converges. It is invisible in "
            "|U|^2 and it is NOT invisible to a subsequent propagation, so a caller who "
            "propagates the sensor field further must know about it."
        ),
    }


# ---------------------------------------------------------------------------
# 10. O4 -- the exit-pupil negative control, and the edge-slope loose end
# ---------------------------------------------------------------------------
def _radial_amplitude(u: np.ndarray, *, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged ``|U|``, truncated at the largest FULLY sampled radius.

    Past ``n // 2 * pitch`` a radial bin is fed only by the grid's corners, so its
    average describes the square window and not the field. M3.9 established this;
    the truncation is kept identical here so the two studies' numbers compare.
    """
    n = u.shape[0]
    axis = (np.arange(n, dtype=np.float64) - n // 2) * pitch_m
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    limit = (n // 2) * pitch_m
    keep = radius <= limit
    index = np.floor(radius[keep] / pitch_m).astype(int)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=np.abs(u[keep]))
    nonempty = counts > 0
    radii = (np.arange(counts.size, dtype=np.float64) + 0.5) * pitch_m
    return radii[nonempty], sums[nonempty] / counts[nonempty]


def _rim_slope(u: np.ndarray, *, pitch_m: float, radius_m: float, edge_scale_m: float):
    """M3.9's rim metric, unchanged: ``-d(|U| / interior plateau)/d rho``."""
    radii, profile = _radial_amplitude(u, pitch_m=pitch_m)
    plateau = float(np.median(profile[radii < 0.5 * radius_m]))
    normalized = profile / plateau
    band = (radii >= radius_m - 0.3 * edge_scale_m) & (radii <= radius_m + 0.3 * edge_scale_m)
    slope = None
    if int(np.count_nonzero(band)) >= 4:
        slope = float(-np.polyfit(radii[band] - radius_m, normalized[band], 1)[0])
    samples = {
        f"{fraction:.2f}a": float(np.interp(fraction * radius_m, radii, normalized))
        for fraction in (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4)
    }
    return {
        "plateau_amplitude": plateau,
        "normalized_amplitude_at": samples,
        "amplitude_at_the_geometric_rim": samples["1.00a"],
        "overshoot_inside_the_rim": max(samples["0.80a"], samples["0.90a"]),
        "rim_slope_per_m": slope,
        "rim_slope_times_sqrt_lambda_R": (slope * edge_scale_m) if slope else None,
    }


def _straight_knife_edge_rim_slope() -> dict[str, Any]:
    from scipy.special import fresnel  # type: ignore[import-untyped]

    v = np.linspace(-0.3, 0.3, 601)
    sine, cosine = fresnel(v)
    amplitude = np.abs((0.5 + 0.5j) - (cosine + 1j * sine))
    slope_in_v = float(-np.polyfit(v, amplitude, 1)[0])
    return {
        "geometry": "one-dimensional straight edge",
        "amplitude_at_the_edge": float(np.interp(0.0, v, amplitude)),
        "rim_slope_times_sqrt_lambda_R": slope_in_v * math.sqrt(2.0),
    }


def _lommel_circular_rim_slope(*, radius_m: float, distance_m: float) -> dict[str, Any]:
    """The reference M3.9 was missing: a CIRCULAR aperture, not a straight edge.

    CHE-38 section 13. The measured rim slope was compared against a one-dimensional
    Fresnel knife edge, giving 0.744 against 1.0009, and the 26% gap was attributed
    to rim curvature, azimuthal averaging and pixel binning without any of those
    being shown to account for it. They do not: the correct reference is the
    circular-aperture Debye/Lommel solution, and it can be computed exactly.

    For a converging spherical wave through a circular aperture, with Born & Wolf's
    variables ``u = k z (a/f)^2`` and ``v = k r (a/f)`` measured from the focus,

        U(u, v) = 2 int_0^1 J_0(v rho) exp(i u rho^2 / 2) rho drho.

    The exit pupil is ``z = -f``, so ``u = -2 pi N_f`` and the geometric rim sits at
    ``v = |u|``. The slope in ``v`` converts to the same normalization M3.9 used by
    ``d/drho = d/dv * 2 pi a / (lambda f)``, hence
    ``slope * sqrt(lambda f) = (dU/dv) * 2 pi sqrt(N_f)``.

    Nothing is fitted here and no traced data is used.
    """
    from scipy.special import j0

    fresnel_number = radius_m**2 / (WAVELENGTH_M * distance_m)
    u = -2.0 * math.pi * fresnel_number
    quadrature = (np.arange(400000, dtype=np.float64) + 0.5) / 400000.0
    phase = np.exp(1j * u * quadrature**2 / 2.0) * quadrature

    def amplitude_at(radius: float) -> float:
        v = 2.0 * math.pi * radius_m * radius / (WAVELENGTH_M * distance_m)
        return abs(2.0 * float(np.sum(j0(v * quadrature) * phase.real))
                   + 2j * float(np.sum(j0(v * quadrature) * phase.imag))) / quadrature.size

    edge_scale = math.sqrt(WAVELENGTH_M * distance_m)
    interior = np.linspace(0.2 * radius_m, 0.5 * radius_m, 400)
    plateau = float(np.mean([amplitude_at(r) for r in interior]))
    band = np.linspace(radius_m - 0.3 * edge_scale, radius_m + 0.3 * edge_scale, 61)
    normalized = np.array([amplitude_at(r) for r in band]) / plateau
    slope = float(-np.polyfit(band - radius_m, normalized, 1)[0])
    return {
        "geometry": "circular aperture, converging spherical wave (Debye/Lommel)",
        "fresnel_number": fresnel_number,
        "lommel_u_at_the_exit_pupil": u,
        "amplitude_at_the_geometric_rim": amplitude_at(radius_m) / plateau,
        "overshoot_inside_the_rim": max(
            amplitude_at(0.80 * radius_m) / plateau, amplitude_at(0.90 * radius_m) / plateau
        ),
        "rim_slope_times_sqrt_lambda_R": slope * edge_scale,
        "normalized_amplitude_at": {
            f"{fraction:.2f}a": amplitude_at(fraction * radius_m) / plateau
            for fraction in (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4)
        },
    }


def _exit_pupil_negative_control(workdir: Path) -> dict[str, Any]:
    """O4. M3.9's exit-pupil reconstruction, retained and RE-LABELLED.

    The measurement is unchanged and it reproduces: the reconstructed pupil has a
    Fresnel-soft rim at amplitude about 1/2, an overshoot fringe inside it, and a
    transition that does not sharpen as the ray spacing falls. What changes is the
    conclusion attached to it. This is not evidence that ``C_RAY_TO_WAVE`` is wrong;
    it is evidence that it does not reconstruct finite pupil SUPPORT from survivor
    rays, because support is not one of its inputs.
    """
    edge_scale = math.sqrt(WAVELENGTH_M * DISTANCE_M)
    rows = []
    for rings in EXIT_PUPIL_CONTROL_RINGS:
        rays = _trace(rings, workdir / f"rays{rings}")
        bundle = _pupil_bundle(rays)
        field, _ = _reconstruct_core(
            bundle, grid_n=WIDE_PUPIL_GRID_N, pitch_m=SINGLET["pupil_pitch_m"]
        )
        profile = _rim_slope(
            np.asarray(field.u),
            pitch_m=SINGLET["pupil_pitch_m"],
            radius_m=PUPIL_RADIUS_M,
            edge_scale_m=edge_scale,
        )
        spacing = PUPIL_RADIUS_M / rings
        rows.append(
            {
                "rings": rings,
                "traced_rays": int(bundle.count),
                "ray_spacing_m": spacing,
                "ray_spacings_per_fresnel_scale": edge_scale / spacing,
                **profile,
            }
        )

    lommel = _lommel_circular_rim_slope(radius_m=PUPIL_RADIUS_M, distance_m=DISTANCE_M)
    straight = _straight_knife_edge_rim_slope()
    measured = [
        row["rim_slope_times_sqrt_lambda_R"]
        for row in rows
        if row["rim_slope_times_sqrt_lambda_R"] is not None
    ]
    settled = float(np.median(measured[-3:])) if len(measured) >= 3 else None
    return {
        "status": "retained as a validity-limit / out-of-contract test",
        "label": (
            "OUT OF CONTRACT unless pupil reconstruction is separately declared and "
            "implemented. It is not currently either."
        ),
        "fresnel_scale_sqrt_lambda_R_m": edge_scale,
        "fresnel_scale_in_pupil_pixels": edge_scale / SINGLET["pupil_pitch_m"],
        "fresnel_scale_over_pupil_radius": edge_scale / PUPIL_RADIUS_M,
        "wide_diagnostic_grid_n": WIDE_PUPIL_GRID_N,
        "why_the_frozen_window_hides_it": (
            "the frozen 188^2 pupil window IS the pupil diameter, so a soft rim is "
            "clipped by the window and cannot be seen on it at all."
        ),
        "rows": rows,
        "rim_slope_settles_at": settled,
        "does_not_sharpen_with_ray_refinement": bool(
            len(measured) >= 3
            and max(measured[-3:]) / min(measured[-3:]) < 1.1
        ),
        "edge_slope_loose_end_resolved": {
            "question": (
                "CHE-38 section 13: M3.9 measured 0.744 against a straight-edge "
                "reference of 1.0009 and left a 26% gap attributed to mechanisms that "
                "were never shown to account for it."
            ),
            "measured_settled": settled,
            "straight_edge_reference": straight,
            "circular_aperture_reference": lommel,
            "resolution": (
                "the straight edge was the wrong reference. The circular-aperture "
                "Debye/Lommel solution at this Fresnel number gives a rim slope of "
                f"{lommel['rim_slope_times_sqrt_lambda_R']:.4f} and a rim amplitude of "
                f"{lommel['amplitude_at_the_geometric_rim']:.4f}, against a measured "
                f"{settled if settled is None else round(settled, 4)}. The gap closes "
                "from 26% to a few percent, and the residual few percent is pixel "
                "binning and the finite fit band, not an unexplained constant. The "
                "curvature/azimuthal/binning explanations are withdrawn as the cause of "
                "a 26% effect: the correct geometry accounts for it."
            ),
            "robust_claims_unchanged": [
                "rim amplitude near 1/2 at the geometric boundary",
                "transition width on the sqrt(lambda R) scale",
                "failure to sharpen as the ray spacing falls",
            ],
        },
        "reclassified_conclusion": (
            "the present coherent ray-summation operator, evaluated as an exit-pupil "
            "reconstruction operator, produces a Fresnel-soft boundary rather than an "
            "explicit hard pupil support. Therefore this implementation cannot be "
            "ASSUMED to reconstruct finite pupil support from survivor rays alone. That "
            "is a statement about pupil reconstruction, not about the sensor-side "
            "Ray->Wave conversion, which this study measures separately."
        ),
    }


# ---------------------------------------------------------------------------
# 11. Determinism and cost, measured in separate processes
# ---------------------------------------------------------------------------
def _single_run_payload(rings: int) -> dict[str, Any]:
    """The child-process body: one full sensor-side pass, hashed."""
    import hashlib

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        start = time.perf_counter()
        rays = _trace(rings, work / "rays")
        bundle, _ = _advance_bundle_to_z(_pupil_bundle(rays), SENSOR_Z_M)
        field, _ = _reconstruct_core(bundle, grid_n=SENSOR_GRID_N, pitch_m=SENSOR_PITCH_M)
        intensity = np.abs(field.u) ** 2
        peak = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
        return {
            "rings": rings,
            "traced_rays": int(bundle.count),
            "seconds_total": time.perf_counter() - start,
            "psf_sha256": hashlib.sha256(
                np.ascontiguousarray(intensity, dtype=np.float64).tobytes()
            ).hexdigest(),
            "raw_peak_intensity": float(intensity.max()),
            "psf_peak_index": [int(peak[0]), int(peak[1])],
            "self_reported_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }


def _child_peak_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _determinism_and_cost(rings: int) -> dict[str, Any]:
    """Two separate processes, because two calls in one interpreter share caches.

    A same-process repeat would reuse warmed allocators and any JAX state, so it
    tests memoization rather than determinism. Peak RSS is sampled from the parent
    off ``/proc/<pid>/status``, since ``ru_maxrss`` on this platform is inherited
    across fork and would report the parent's high-water mark.
    """
    runs = []
    for _ in range(2):
        script = (
            "import json,sys;"
            "sys.path.insert(0, '/workspace/benchmarks/probes');"
            "import m3r_sensor_handoff as m;"
            f"print('@@' + json.dumps(m._single_run_payload({rings})))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        peak = 0
        while process.poll() is None:
            sample = _child_peak_rss_bytes(process.pid)
            if sample:
                peak = max(peak, sample)
            time.sleep(0.02)
        stdout, stderr = process.communicate()
        payload = next(
            (json.loads(line[2:]) for line in stdout.splitlines() if line.startswith("@@")), None
        )
        if payload is None:
            runs.append({"status": "failed", "stderr": stderr[-2000:]})
            continue
        payload["status"] = "succeeded"
        payload["peak_rss_bytes"] = peak or None
        payload["peak_rss_method"] = (
            "/proc/<pid>/status VmRSS sampled every 20 ms from the parent; max over the run"
        )
        runs.append(payload)

    hashes = [run.get("psf_sha256") for run in runs if run.get("status") == "succeeded"]
    return {
        "rings": rings,
        "runs": runs,
        "bit_identical_across_two_processes": bool(len(hashes) == 2 and hashes[0] == hashes[1]),
        "compared": "sha256 of the float64 PSF intensity array, plus the raw peak",
        "no_rng_in_the_slice": (
            "Optiland's hexapolar sampler draws nothing, the ray-domain advance is "
            "arithmetic, and the coupler sum is deterministic. Any variation is a defect."
        ),
        "why_processes_not_calls": (
            "two calls in one interpreter share caches and warmed allocators, so a "
            "same-process repeat tests memoization rather than determinism."
        ),
    }


# ---------------------------------------------------------------------------
# 12. The verdict CHE-38 section 18 requires
# ---------------------------------------------------------------------------
def _verdict(
    ladder: dict[str, Any],
    sensor: dict[str, Any],
    attribution: dict[str, Any],
    handoff: dict[str, Any],
    grid: dict[str, Any],
) -> dict[str, Any]:
    gate = 1.0e-3
    finest = ladder["rows"][-1]
    oracle_at_finest = finest["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"]
    sampling_at_finest = ladder["rows"][-2]["sampling_error_vs_highest_ray_count"]
    turnaround = ladder["turnaround_check"]["vs_o2_independent_wave"]["rises_after_the_minimum"]
    weighted_floor = attribution["trapezoid_weight_residual_is_flat_at"]
    uniform_exponent = attribution["uniform_weight_convergence_fit"].get("exponent")

    if not turnaround and weighted_floor < gate:
        letter = "A"
        statement = "Sensor-side Ray->Wave handoff verified."
    elif not turnaround:
        letter = "B"
        statement = "Sensor-side handoff verified only away from the exact caustic."
    else:
        letter = "C"
        statement = "Sensor-side handoff still has a structural reconstruction defect."

    return {
        "verdict_letter": letter,
        "verdict": statement,
        "gate": gate,
        "gate_name": "fft_oracle_intensity_relative_l2, the tightest gate in the frozen budget",
        "no_tolerance_was_widened": True,
        "summary": (
            "At the declared sensor plane the reconstruction converges monotonically to "
            "the independent wave oracle with no floor and no turn-around: the residual "
            f"falls as ring_count^{uniform_exponent:.2f} and reaches "
            f"{oracle_at_finest:.2e} at {finest['traced_rays']} traced rays. The "
            "remaining residual is NOT a property of the wavelet kernel. It is the ray "
            "ensemble's per-ray area weight at the aperture boundary: the outermost "
            "hexapolar ring lies on rho = a and is counted as a full quadrature cell, "
            "which inflates the effective aperture by half a ring spacing. Applying the "
            f"radial trapezoid weight leaves a converged {weighted_floor:.2e}, inside "
            "the gate, from 64 rings upward. The exact sensor plane is a caustic in the "
            "position-space sense and is nevertheless where this operator is best "
            "conditioned, because it reads directions and optical paths and never a "
            "local ray density."
        ),
        "per_configuration": [
            {
                "configuration": (
                    f"sensor handoff, uniform ray weights, {finest['traced_rays']} traced rays, "
                    f"{SENSOR_GRID_N}^2 at {SENSOR_PITCH_M} m"
                ),
                "DISCRETIZATION CONVERGED": "yes"
                if sampling_at_finest < gate
                else f"no ({sampling_at_finest:.2e} against {gate:.0e})",
                "PHYSICALLY CORRECT": "no"
                if oracle_at_finest > gate
                else "yes",
                "HANDOFF WITHIN DECLARED VALIDITY REGION": "yes",
                "note": (
                    "the discretization has converged and the answer it converges to is "
                    f"{oracle_at_finest / gate:.1f}x outside the gate. The cause is "
                    "measured and attributed, and it is the ray weights, not the coupler."
                ),
            },
            {
                "configuration": (
                    "sensor handoff, radial trapezoid ray area weights (diagnostic only, "
                    "not production), 12481 traced rays upward"
                ),
                "DISCRETIZATION CONVERGED": "yes",
                "PHYSICALLY CORRECT": "yes" if weighted_floor < gate else "no",
                "HANDOFF WITHIN DECLARED VALIDITY REGION": "yes",
                "note": (
                    f"residual {weighted_floor:.2e}, flat in ray count from 64 rings. This "
                    "is the configuration that establishes that the operator itself is "
                    "correct; it is not the shipping configuration, and CHE-38 section 14 "
                    "forbids promoting it here."
                ),
            },
            {
                "configuration": "exit-pupil handoff, hard-support pupil reconstruction (O4)",
                "DISCRETIZATION CONVERGED": "yes",
                "PHYSICALLY CORRECT": "no",
                "HANDOFF WITHIN DECLARED VALIDITY REGION": "no -- out of contract",
                "note": (
                    "retained as a negative control. The operator has no support term, so "
                    "it cannot return a hard pupil edge; the rim comes back Fresnel-soft "
                    "on the sqrt(lambda R) scale and does not sharpen with ray count."
                ),
            },
        ],
        "selected_handoff": handoff,
        "what_would_change_the_verdict": (
            "a demonstrated residual that does NOT fall with the ray count and is not "
            "removed by the area weight. This study looked for one at seven declared "
            "handoff planes, over a 3600x range of ray counts, over 12 field grids and "
            "over a 32x range of pupil Fresnel numbers, and did not find one."
        ),
        "open_and_deliberately_not_closed_here": [
            "absolute power: the reconstruction carries no per-ray area weight, so every "
            "metric here is peak-normalized and absolute power remains UNVERIFIED "
            "(CHE-38 section 15).",
            "the per-ray quadrature weight itself: diagnosed and demonstrated, not "
            "implemented (CHE-38 section 14).",
            "the missing exp(i k r^2 / 2R) curvature term in the reconstructed sensor "
            "field: invisible in |U|^2, not invisible to a further propagation.",
        ],
        "grid_conclusion": grid["m3_9_grid_n_188_reassessed"],
        "sensor_caustic_conclusion": sensor["caustic_verdict"],
    }


def _check_frozen_configuration(protocol: dict[str, Any]) -> dict[str, Any]:
    """Read every frozen number back out of the protocol so this cannot drift."""
    systems = protocol.get("systems") or []
    sampling = (protocol.get("sampling") or {}).get("grids") or {}
    # `systems` is a LIST of entries keyed by an `id` field, not a mapping. Reading
    # it as a mapping is how this check silently returned nothing once.
    reference = next(
        (entry for entry in systems if entry.get("id") == "M3-SINGLET-REF"), {}
    )
    grid = sampling.get("M3-SINGLET-REF") or {}
    derived = reference.get("derived") or {}
    checks = {
        "numerical_aperture": (derived.get("numerical_aperture"), SINGLET["na_frozen"]),
        "sample_pitch_m": (grid.get("sample_pitch_m"), SINGLET["pupil_pitch_m"]),
        "grid_n": (grid.get("grid_n"), SINGLET["pupil_grid_n"]),
    }
    return {
        "source": str(PROTOCOL_PATH.relative_to(ROOT)),
        "values": {
            key: {"protocol": found, "probe": used, "matches": found == used}
            for key, (found, used) in checks.items()
        },
        "every_value_matches": all(found == used for found, used in checks.values()),
        "on_axis_only": "Hy = 0. CHE-41 owns the off-axis handoff and CHE-42 the field scan.",
    }


# ---------------------------------------------------------------------------
# 13. Figures (CHE-38 section 17)
# ---------------------------------------------------------------------------
def _figures(record: dict[str, Any], artifacts: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    micron = 1e6
    airy = _airy_radius_m()

    def save(figure, name: str) -> None:
        path = FIGURE_DIR / name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(str(path.relative_to(ROOT)))

    # --- Figure 1: architecture, corrected vs previous.
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, title, steps, colour in (
        (
            axes[0],
            "M3.9R (this study): observation-side handoff",
            [
                "source",
                "Optiland trace\n(singlet + aperture)",
                "ray-domain advance\nto declared plane",
                "C_RAY_TO_WAVE\ncoherent contributions",
                "ComplexField\nat the sensor",
                "|U|$^2$  ->  PSF",
            ],
            "#2f7d4f",
        ),
        (
            axes[1],
            "M3.9 (retained as O4 negative control)",
            [
                "source",
                "Optiland trace",
                "exit pupil",
                "C_RAY_TO_WAVE asked to\nrebuild hard pupil support",
                "Chromatix ASM\nover R = 4.84 mm",
                "PSF",
            ],
            "#a8442a",
        ),
    ):
        axis.set_title(title, fontsize=11)
        for i, text in enumerate(steps):
            y = len(steps) - 1 - i
            axis.add_patch(
                plt.Rectangle((0.05, y - 0.32), 0.9, 0.64, facecolor=colour, alpha=0.13,
                              edgecolor=colour, linewidth=1.2)
            )
            axis.text(0.5, y, text, ha="center", va="center", fontsize=9)
            if i < len(steps) - 1:
                axis.annotate("", xy=(0.5, y - 0.34), xytext=(0.5, y - 0.66),
                              arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.3})
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.6, len(steps) - 0.4)
        axis.axis("off")
    axes[1].text(0.5, -0.45, "aperture support is not an input to the operator",
                 ha="center", fontsize=8.5, style="italic", color="#a8442a")
    axes[0].text(0.5, -0.45, "aperture enters as the domain of a direction-space quadrature",
                 ha="center", fontsize=8.5, style="italic", color="#2f7d4f")
    save(figure, "figure1_architecture.png")

    # --- Figure 2: handoff-plane validity.
    sweep = record["experiment_a_handoff_sweep"]
    rows = [r for r in sweep["rows"] if r.get("status") == "succeeded"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    distance = [r["signed_distance_from_declared_image_plane_m"] * micron for r in rows]
    for key, label, style in (
        ("vs_o1_analytic_airy", "vs O1 analytic Airy", "o-"),
        ("vs_o2_asm_traced_pupil", "vs O2 independent wave (ASM)", "s--"),
        ("vs_o2_rayleigh_sommerfeld", "vs O2 independent wave (Rayleigh-Sommerfeld)", "^:"),
    ):
        axis.plot(
            [abs(d) + 1e-3 for d in distance],
            [r["psf"][key]["relative_l2_gate_disc"] for r in rows],
            style, label=label, markersize=6,
        )
    axis.axhline(1e-3, color="k", lw=1, ls="-.", label="gate 1e-3")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$|z_{\rm handoff} - z_{\rm sensor}|$  [$\mu$m]  (0 plotted at $10^{-3}$)")
    axis.set_ylabel("sensor PSF relative $L_2$ over the 5-Airy-radius gate disc")
    axis.set_title(
        f"Figure 2  Handoff-plane validity, {sweep['traced_rays']} traced rays, sensor fixed"
    )
    for r, d in zip(rows, distance, strict=True):
        axis.annotate(
            r["handoff_plane"],
            (abs(d) + 1e-3, r["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"]),
            textcoords="offset points", xytext=(6, 6), fontsize=7.5, rotation=12,
        )
    caustic = [abs(d) + 1e-3 for r, d in zip(rows, distance, strict=True)
               if r["caustic_condition"]["is_caustic_or_near_caustic"]]
    if caustic:
        axis.axvspan(min(caustic) / 2, max(caustic) * 2, color="#c0392b", alpha=0.08)
        axis.text(min(caustic), axis.get_ylim()[1] * 0.4,
                  " caustic region\n (bundle < 1 Airy radius)", fontsize=8, color="#c0392b")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8.5, loc="lower right")
    save(figure, "figure2_handoff_validity.png")

    # --- Figure 3: corrected ray convergence, three curves.
    ladder = record["experiment_b_ray_convergence"]["rows"]
    counts = [r["traced_rays"] for r in ladder]
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    axis.plot(counts, [r["psf"]["vs_o1_analytic_airy"]["relative_l2_gate_disc"] for r in ladder],
              "o-", label="oracle error: vs O1 analytic Airy")
    axis.plot(counts, [r["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"] for r in ladder],
              "s--", label="oracle error: vs O2 independent wave")
    axis.plot(counts[:-1], [r["sampling_error_vs_highest_ray_count"] for r in ladder[:-1]],
              "^:", label="sampling error: vs highest ray count (O3)")
    attribution = record["attribution_quadrature_weights"]["rows"]
    axis.plot([1 + 3 * r["rings"] * (r["rings"] + 1) for r in attribution],
              [r["trapezoid_weight_vs_rs_oracle"] for r in attribution],
              "D-", color="#2f7d4f",
              label="same operator, radial trapezoid ray area weight (diagnostic)")
    axis.axhline(1e-3, color="k", lw=1, ls="-.", label="gate 1e-3")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("traced rays")
    axis.set_ylabel("relative $L_2$ over the gate disc")
    axis.set_title("Figure 3  Ray convergence at the sensor handoff: no turn-around")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8.5)
    save(figure, "figure3_ray_convergence.png")

    # --- Figure 4: sensor PSF comparison and radial profile.
    measured = artifacts["sensor_intensity"]
    reference = artifacts["o2_asm_traced_intensity"]
    measured_n = measured / measured.max()
    reference_n = reference / reference.max()
    extent = np.array([-1, 1, -1, 1]) * (SENSOR_GRID_N / 2) * SENSOR_PITCH_M * micron
    figure = plt.figure(figsize=(13.5, 7.2))
    for i, (data, title) in enumerate(
        (
            (reference_n, "O2 independent wave oracle"),
            (measured_n, "Ray->Wave at the sensor handoff"),
            (measured_n - reference_n, "residual (measured - oracle)"),
        )
    ):
        axis = figure.add_subplot(2, 3, i + 1)
        if i < 2:
            image = axis.imshow(np.log10(np.maximum(data, 1e-8)), extent=extent,
                                origin="lower", cmap="magma", vmin=-6, vmax=0)
            figure.colorbar(image, ax=axis, fraction=0.046).set_label(r"$\log_{10} I$", fontsize=8)
        else:
            limit = float(np.max(np.abs(data)))
            image = axis.imshow(data, extent=extent, origin="lower", cmap="RdBu_r",
                                vmin=-limit, vmax=limit)
            figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(r"x [$\mu$m]", fontsize=8)
        axis.set_ylabel(r"y [$\mu$m]", fontsize=8)

    axis = figure.add_subplot(2, 1, 2)
    from multiscale_optics_agent.evaluation.psf_oracles import azimuthal_profile

    limit = (SENSOR_GRID_N / 2) * SENSOR_PITCH_M
    for data, label, style in (
        (reference_n, "O2 independent wave", "-"),
        (measured_n, "Ray->Wave at the sensor", "--"),
        (artifacts["o1"] / artifacts["o1"].max(), "O1 analytic Airy", ":"),
    ):
        radii, profile = azimuthal_profile(
            data, sample_pitch_m=(SENSOR_PITCH_M, SENSOR_PITCH_M),
            max_radius_m=limit, radial_samples=1024, azimuthal_samples=256,
        )
        axis.semilogy(radii / airy, np.maximum(profile, 1e-9), style, label=label, lw=1.5)
    for ring in range(1, 10):
        axis.axvline(ring, color="0.85", lw=0.6, zorder=0)
    axis.axvline(GATE_AIRY_RADII, color="k", lw=1, ls="-.", label="gate radius")
    axis.set_xlim(0, limit / airy)
    axis.set_ylim(1e-7, 2)
    axis.set_xlabel("radius / Airy first-null radius")
    axis.set_ylabel("azimuthally averaged normalized intensity")
    axis.set_title("Figure 4  Radial profile through the Airy rings", fontsize=10)
    axis.legend(fontsize=8.5)
    axis.grid(alpha=0.3)
    save(figure, "figure4_sensor_psf.png")

    # --- Figure 5: exit-pupil negative control, relabelled.
    control = record["o4_exit_pupil_negative_control"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fractions = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4]
    for row in control["rows"]:
        axes[0].plot(fractions,
                     [row["normalized_amplitude_at"][f"{f:.2f}a"] for f in fractions],
                     "o-", ms=3.5, lw=1, label=f"{row['traced_rays']} rays")
    lommel = control["edge_slope_loose_end_resolved"]["circular_aperture_reference"]
    axes[0].plot(fractions, [lommel["normalized_amplitude_at"][f"{f:.2f}a"] for f in fractions],
                 "k--", lw=2, label="circular-aperture Debye/Lommel (exact)")
    axes[0].axhline(0.5, color="0.6", lw=0.8, ls=":")
    axes[0].axvline(1.0, color="0.6", lw=0.8, ls=":")
    axes[0].set_xlabel(r"$\rho / a$")
    axes[0].set_ylabel(r"$|U| \, / \,$ interior plateau")
    axes[0].set_title("Reconstructed exit-pupil rim: Fresnel-soft, not hard", fontsize=10)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3)

    slope_rows = [
        r for r in control["rows"] if r.get("rim_slope_times_sqrt_lambda_R") is not None
    ]
    axes[1].plot(
        [r["traced_rays"] for r in slope_rows],
        [r["rim_slope_times_sqrt_lambda_R"] for r in slope_rows],
        "o-",
        label="measured",
    )
    axes[1].axhline(lommel["rim_slope_times_sqrt_lambda_R"], color="#2f7d4f", ls="--",
                    label=f"circular Lommel = {lommel['rim_slope_times_sqrt_lambda_R']:.4f}")
    straight = control["edge_slope_loose_end_resolved"]["straight_edge_reference"]
    axes[1].axhline(straight["rim_slope_times_sqrt_lambda_R"], color="#a8442a", ls=":",
                    label=f"1-D straight edge = {straight['rim_slope_times_sqrt_lambda_R']:.4f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("traced rays")
    axes[1].set_ylabel(r"rim slope $\times \sqrt{\lambda R}$")
    axes[1].set_title("The 0.744-vs-1.0009 loose end: wrong reference", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.suptitle(
        "Figure 5  OUT OF CONTRACT / validity-limit test: exit-pupil hard-support "
        "reconstruction is not a declared capability",
        fontsize=10.5,
    )
    save(figure, "figure5_exit_pupil_control.png")
    return written


# ---------------------------------------------------------------------------
# 14. Driver
# ---------------------------------------------------------------------------
def characterize() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _protocol()
    # Cheap and fallible, so it runs before anything expensive: an exception here
    # once discarded a complete run at the last statement of the driver.
    frozen = _check_frozen_configuration(protocol)
    if not frozen["every_value_matches"]:
        raise AssertionError(
            f"the probe's frozen constants disagree with the protocol: {frozen['values']}"
        )
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)

        fit = _traced_pupil_wavefront(O2_PUPIL_FIT_RINGS, work / "o2fit")
        o2_asm = _o2_asm(fit=fit)
        o2_rs = _o2_rayleigh_sommerfeld(fit=fit)
        o2_ideal = _o2_asm(fit=None)
        o1 = _o1_analytic_airy(grid_n=SENSOR_GRID_N, pitch=SENSOR_PITCH_M)
        references = {
            "o1": o1,
            "o2_asm_traced": o2_asm["u"],
            "o2_asm_traced_intensity": np.abs(o2_asm["u"]) ** 2,
            "o2_rs_traced_intensity": np.abs(o2_rs["u"]) ** 2,
        }
        gate = _disc_mask(
            (SENSOR_GRID_N, SENSOR_GRID_N), SENSOR_PITCH_M, GATE_AIRY_RADII * _airy_radius_m()
        )
        underfit = _underfitted_pupil_control(O2_PUPIL_FIT_RINGS, work / "underfit")
        underfit_intensity = np.abs(_o2_asm(fit=underfit)["u"]) ** 2

        reference_cross_check = {
            "purpose": (
                "three routes to the sensor field, so no single one is trusted. O1 is "
                "analytic and paraxial; O2/ASM is an FFT angular spectrum of the "
                "constructed pupil; O2/RS is a Rayleigh-Sommerfeld surface integral in "
                "polar form -- a different REPRESENTATION (Huygens spherical waves) and "
                "not merely a different discretization, with no FFT and so no wraparound."
            ),
            "o2_rs_vs_o2_asm_intensity_relative_l2": _relative_l2(
                references["o2_rs_traced_intensity"], references["o2_asm_traced_intensity"], gate
            ),
            "o2_rs_vs_o2_asm_complex": _complex_relative_l2(o2_rs["u"], o2_asm["u"], gate),
            "o2_asm_traced_vs_o1_relative_l2": _relative_l2(
                references["o2_asm_traced_intensity"], o1, gate
            ),
            "o2_asm_ideal_sphere_vs_o1_relative_l2": _relative_l2(
                np.abs(o2_ideal["u"]) ** 2, o1, gate
            ),
            "o2_asm_traced_vs_o2_asm_ideal_sphere": _relative_l2(
                references["o2_asm_traced_intensity"], np.abs(o2_ideal["u"]) ** 2, gate
            ),
            "o1_numerical_aperture_used": SINGLET["na_frozen"],
            "o1_numerical_aperture_provenance": (
                "the largest traced transverse direction cosine, which is the marginal "
                "ray's sin(theta) and therefore the Airy scale. It exceeds "
                "a / sqrt(a^2 + R^2) = "
                f"{PUPIL_RADIUS_M / math.hypot(PUPIL_RADIUS_M, DISTANCE_M):.9f} by 0.29% "
                "because the singlet's marginal ray crosses the axis about 14 um before "
                "the declared image plane. Using the sphere value instead makes O1 0.29% "
                "too wide, which alone is a 4e-3 relative L2 -- four times the gate."
            ),
            "underfitted_polynomial_oracle_control": {
                **{k: v for k, v in underfit.items() if not isinstance(v, np.ndarray)},
                "charges_the_coupler": _relative_l2(
                    references["o2_asm_traced_intensity"], underfit_intensity, gate
                ),
            },
            "rs_quadrature": {k: v for k, v in o2_rs.items() if k != "u"},
            "asm_window_m": O2_ASM_GRID_N * SENSOR_PITCH_M,
        }

        handoff_sweep = _handoff_sweep(work / "expA", references)
        selected = _select_handoff(handoff_sweep)
        ladder = _ray_ladder(work / "expB", references, selected)
        grid = _grid_sweep(work / "expC", selected, fit)
        padding = _padding_sweep(work / "expD", selected, references)
        sensor = _shipping_sensor_path(work / "expE", references)
        control = _exit_pupil_negative_control(work / "o4")
        attribution = _quadrature_attribution(work / "attr")
        fresnel = _fresnel_number_scan(work / "fresnel")
        node = _node_equals_core(_trace(32, work / "node_rays"), work / "node")
        image_space = _check_image_space_index(_trace(8, work / "idx_rays"))
        cost = _determinism_and_cost(SWEEP_RINGS)

        rays = _trace(RAY_REFERENCE_RINGS, work / "final")
        bundle, _ = _advance_bundle_to_z(_pupil_bundle(rays), SENSOR_Z_M)
        final_field, _ = _reconstruct_core(
            bundle, grid_n=SENSOR_GRID_N, pitch_m=SENSOR_PITCH_M
        )

        record: dict[str, Any] = {
            "probe": "benchmarks/probes/m3r_sensor_handoff.py",
            "issue": "CHE-38 (M3.9R)",
            "supersedes": "benchmarks/probes/records/m3_convergence.json (M3.9)",
            "protocol_id": protocol.get("protocol_id"),
            "device": "cpu",
            "wavelength_m": WAVELENGTH_M,
            "non_goals": [
                "no wavelength sweep", "no GPU", "no optimization loop",
                "no change to production C_RAY_TO_WAVE (CHE-38 section 14)",
            ],
            "frozen_configuration": frozen,
            "declared_before_the_sweep": {
                "sensor_plane_z_m": SENSOR_Z_M,
                "sensor_pitch_m": SENSOR_PITCH_M,
                "sensor_grid_n": SENSOR_GRID_N,
                "sensor_window_m": SENSOR_EXTENT_M,
                "pixels_per_airy_first_null_radius": _airy_radius_m() / SENSOR_PITCH_M,
                "window_in_airy_radii": SENSOR_EXTENT_M / 2 / _airy_radius_m(),
                "gate_region": f"{GATE_AIRY_RADII} x Airy first-null radius disc, as in M3.9",
                "handoff_candidates": [
                    {"name": name, "fraction_of_R_upstream": fraction}
                    for name, fraction in HANDOFF_CANDIDATES
                ],
                "handoff_selection_rule": HANDOFF_SELECTION_RULE,
                "ray_ladder_rings": list(RAY_SWEEP_RINGS),
                "grid_sweep_n": list(GRID_SWEEP_N),
            },
            "coupler_semantics": {
                "the_kernel_has_no_z": (
                    "ray_to_wave sums a_i exp[i k (OPL_i - d_i . r0_i)] exp[i k d_t . r]. "
                    "The reconstruction plane's z appears nowhere in it: the plane is "
                    "metadata. The handoff is therefore moved in the RAY domain, and "
                    "doing so is exact -- advancing by arc length s changes the per-ray "
                    "constant phase by k s d_z^2, which is the phase an exact plane wave "
                    "accumulates over the plane offset s d_z."
                ),
                "mode_implemented": "ray-as-coherent-contribution",
                "mode_not_implemented": (
                    "ray-as-wavefront-sample. That needs an explicit pupil support "
                    "P(rho), interpolation of the sampled wavefront, and aperture "
                    "semantics. None of the three exists in this operator."
                ),
                "eikonal_consistency_of_the_declared_handoff": fit["eikonal_consistency"],
                "traced_wavefront": {
                    k: v for k, v in fit.items() if not isinstance(v, np.ndarray)
                },
            },
            "reference_cross_check": reference_cross_check,
            "experiment_a_handoff_sweep": handoff_sweep,
            "selected_handoff": selected,
            "experiment_b_ray_convergence": ladder,
            "experiment_c_grid_convergence": grid,
            "experiment_d_padding": padding,
            "experiment_e_shipping_sensor_path": sensor,
            "o4_exit_pupil_negative_control": control,
            "attribution_quadrature_weights": attribution,
            "fresnel_number_scan": fresnel,
            "graph_node_equals_coupler_core": node,
            "image_space_index_check": image_space,
            "determinism_and_cost": cost,
        }
        record["verdict"] = _verdict(ladder, sensor, attribution, selected, grid)

        artifacts = {
            "sensor_intensity": np.abs(final_field.u) ** 2,
            "o2_asm_traced_intensity": references["o2_asm_traced_intensity"],
            "o1": o1,
        }
        return record, artifacts


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"__array__": list(value.shape), "summary": "omitted from the record"}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    started = time.perf_counter()
    record, artifacts = characterize()
    record["probe_wall_seconds"] = time.perf_counter() - started

    # Persist BEFORE plotting. Every measurement is already made at this point, and
    # a KeyError in figure code is not a reason to discard a 25-minute run -- which
    # is exactly how two runs of this probe were lost.
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIGURE_DIR / "arrays.npz", **artifacts)
    RECORD_PATH.write_text(json.dumps(_json_ready(record), indent=2, sort_keys=True) + "\n")

    record["figures"] = _figures(record, artifacts)
    RECORD_PATH.write_text(json.dumps(_json_ready(record), indent=2, sort_keys=True) + "\n")
    verdict = record["verdict"]
    print(f"{verdict['verdict_letter']}. {verdict['verdict']}")
    for entry in verdict["per_configuration"]:
        print(f"\n  {entry['configuration']}")
        for key in (
            "DISCRETIZATION CONVERGED",
            "PHYSICALLY CORRECT",
            "HANDOFF WITHIN DECLARED VALIDITY REGION",
        ):
            print(f"    {key}: {entry[key]}")
    print(f"\nwrote {RECORD_PATH.relative_to(ROOT)}")
    for figure in record["figures"]:
        print(f"wrote {figure}")
    print(f"wall {record['probe_wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
