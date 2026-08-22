"""CHE-35 (M3.6): propagate the reconstructed pupil field, and turn two warnings into numbers.

Run inside the container:

    ./run.sh python benchmarks/probes/chromatix/m3_pupil_to_focus.py \
        --write benchmarks/probes/records/chromatix/m3_pupil_to_focus.json

Four questions, each of which the adapter currently answers with a warning
string or not at all:

1. **What does Chromatix's ``complex64`` cast actually cost on the real pupil
   field?** Measured twice: the input truncation on its own, and end to end
   against the float64 reference (``verification/asm_oracle.py``), on both the
   absolute-carrier path and the carrier-conditioned path the protocol requires.

2. **Which phasor convention does Chromatix's ASM implement?** Settled by
   construction rather than by reading source: build an analytic converging
   spherical wave under this project's declared conventions
   (``exp(-i omega t)`` with ``exp(+i k z)``), propagate it by its own radius,
   and see whether it focuses. Its complex conjugate is propagated alongside as
   the control -- under the opposite convention that one would focus instead,
   and exactly one of the two must.

3. **What padding is needed, and what does the edge energy do?** Chromatix's own
   ``compute_padding_transfer`` estimate is compared against no padding and
   against the propagated field's edge-energy fraction, which is the observable
   wraparound indicator.

4. **What is conserved?** Power in versus power out, with the loss attributed.

Nothing here changes the M1-verified propagation. The probe selects between an
existing Chromatix call and CHE-40's carrier-removed kernel, both of which use
Chromatix's own FFTs, padding, and spatial-frequency grid.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from couplers.base import CouplerRunRequest
from couplers.node import RayToWaveCoupler
from solvers.base import ModelRunRequest
from solvers.optiland.adapter import get_adapter as get_ray_adapter
from verification.asm_oracle import (
    ASM_ORACLE_ID,
    CarrierConvention,
    angular_spectrum_float64,
    compare_fields,
    relative_phase_excursion_rad,
)

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml"

WAVELENGTH_UM = 0.55
WAVELENGTH_M = 5.5e-7
NUM_RAYS = 16
MM_PER_M = 1e-3
EPS32 = float(np.finfo(np.float32).eps)


def _system(system_id: str) -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    system = next(item for item in protocol["systems"] if item["id"] == system_id)
    grids = protocol["sampling"]["grids"][system_id]
    return {"derived": system["derived"], "grid": grids}


def build_pupil_field(sample: str, system_id: str, run_dir: Path) -> dict[str, Any]:
    """The real M3.5 output: a coupler-written ComplexField record."""
    derived = _system(system_id)["derived"]
    grid = _system(system_id)["grid"]

    rays = (
        get_ray_adapter()
        .run(
            ModelRunRequest(
                run_id="che35",
                node_id=sample,
                config={
                    "sample": sample,
                    "num_rays": NUM_RAYS,
                    "wavelength": WAVELENGTH_UM,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(run_dir / f"{sample}-rays"),
                },
            )
        )
        .outputs["rays"]
    )

    result = RayToWaveCoupler().transform(
        CouplerRunRequest(
            run_id="che35",
            edge_id=f"{sample}-pupil",
            source=rays,
            config={
                "handoff_plane": "exit_pupil",
                "handoff_plane_z_m": derived["exit_pupil_z_mm"] * MM_PER_M,
                "grid_n": int(grid["grid_n"]),
                "target_sample_pitch_m": float(grid["sample_pitch_m"]),
                "projection": "asm_consistent",
                "output_dir": str(run_dir / f"{sample}-field"),
            },
        )
    )
    assert result.status.value == "succeeded", result.error_message
    return {
        "record": result.target,
        "u": np.load(result.target.uri),
        "pitch_m": float(grid["sample_pitch_m"]),
        "distance_m": derived["propagation_distance_mm"] * MM_PER_M,
        "grid_n": int(grid["grid_n"]),
    }


def _chromatix_propagate(
    u: np.ndarray, *, pitch_m: float, z_m: float, pad_width: int, carrier_removed: bool
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from solvers.chromatix.carrier_removed_asm import (
        carrier_removed_asm_propagate,
        pin_wave_engine_precision,
    )

    pin_wave_engine_precision()
    import chromatix.functional as cf

    field = cf.Field.build(
        jnp.asarray(u, dtype=jnp.complex64), jnp.asarray([[pitch_m, pitch_m]]), WAVELENGTH_M
    )
    if carrier_removed:
        out = carrier_removed_asm_propagate(
            field, z_m=z_m, pad_width=pad_width, mode="same", wavelength_m=WAVELENGTH_M
        ).field
    else:
        out = cf.asm_propagate(field, z=z_m, n=1.0, pad_width=pad_width)
        if pad_width:
            from chromatix.functional.propagation import crop

            out = crop(out, pad_width)
    return np.asarray(jax.device_get(out.u)).reshape(u.shape)


def _edge_energy_fraction(u: np.ndarray) -> float:
    intensity = np.abs(u) ** 2
    total = float(intensity.sum())
    if total <= 0.0 or min(intensity.shape) < 3:
        return 0.0
    interior = float(intensity[1:-1, 1:-1].sum())
    return (total - interior) / total


def _discrete_power(u: np.ndarray, pitch_m: float) -> float:
    return float(np.sum(np.abs(u) ** 2) * pitch_m * pitch_m)


def _oracle(
    u: np.ndarray,
    *,
    pitch_m: float,
    z_m: float,
    carrier: CarrierConvention,
    pad_width: int = 0,
) -> np.ndarray:
    """The float64 reference, optionally with the same zero-padding Chromatix used.

    ``asm_oracle`` deliberately does not pad. Comparing a padded Chromatix run
    against an unpadded reference measures padding and dtype at once -- which is
    how the first version of this probe reported 6.1e-1 for a dtype effect
    predicted at 6.6e-3. Padding is applied here, identically on both sides, so
    the two can be separated.
    """
    if pad_width:
        u = np.pad(u, pad_width, mode="constant")
    out = angular_spectrum_float64(
        u, wavelength_m=WAVELENGTH_M, sample_pitch_m=pitch_m, z_m=z_m, carrier=carrier
    )
    if pad_width:
        out = out[pad_width:-pad_width, pad_width:-pad_width]
    return out


def measure_complex64_cost(pupil: dict[str, Any], pad_width: int) -> dict[str, Any]:
    """What the ``complex64`` cast costs, with padding held identical on both sides."""
    u = pupil["u"]
    pitch_m, z_m = pupil["pitch_m"], pupil["distance_m"]

    # 1. What the cast costs before anything propagates.
    input_only = compare_fields(u.astype(np.complex64).astype(np.complex128), u)

    # 2. End to end, both carrier conventions, at matched padding.
    paths = {}
    for name, carrier_removed, carrier in (
        ("absolute_carrier_path", False, CarrierConvention.ABSOLUTE),
        ("carrier_conditioned_path", True, CarrierConvention.CARRIER_REMOVED),
    ):
        reference = _oracle(u, pitch_m=pitch_m, z_m=z_m, carrier=carrier, pad_width=pad_width)
        test = _chromatix_propagate(
            u, pitch_m=pitch_m, z_m=z_m, pad_width=pad_width, carrier_removed=carrier_removed
        )
        paths[name] = compare_fields(test, reference).as_dict()

    carrier_phase_rad = 2.0 * np.pi * z_m / WAVELENGTH_M
    excursion_rad = relative_phase_excursion_rad(
        wavelength_m=WAVELENGTH_M, sample_pitch_m=pitch_m, z_m=z_m, shape=u.shape
    )
    return {
        "pad_width_used": pad_width,
        "padding_matched_on_both_sides": True,
        "input_cast_only": input_only.as_dict(),
        **paths,
        "represented_phase_rad": {
            "absolute_carrier_k_z": carrier_phase_rad,
            "relative_excursion_max": excursion_rad,
            "ratio": carrier_phase_rad / excursion_rad,
        },
        "predicted_from_eps32": {
            "absolute": EPS32 * carrier_phase_rad,
            "carrier_conditioned": EPS32 * excursion_rad,
        },
    }


def measure_padding_cost_in_float64(pupil: dict[str, Any], pad_width: int) -> dict[str, Any]:
    """Padding on its own, with no dtype involved: float64 padded vs float64 unpadded.

    This is the ``grid_truncation_and_padding`` budget term the protocol records
    as owed to this ticket. Measuring it in float64 is what makes it separable
    from the ``complex64`` term rather than tangled with it.
    """
    u, pitch_m, z_m = pupil["u"], pupil["pitch_m"], pupil["distance_m"]

    def run(pad: int) -> np.ndarray:
        return _oracle(
            u,
            pitch_m=pitch_m,
            z_m=z_m,
            carrier=CarrierConvention.CARRIER_REMOVED,
            pad_width=pad,
        )

    # Twice the selected padding, as the reference the sweep converges toward. It
    # is not itself converged -- the sequence below falls roughly like 1/pad --
    # which is why the budget term carries the next term's worth of headroom
    # rather than being read off the last row.
    converged_pad = int(2 * pad_width)
    converged = run(converged_pad)
    sweep = {}
    for pad in sorted({0, pad_width // 6, pad_width // 3, pad_width // 2, pad_width}):
        sweep[str(pad)] = {
            "padded_side": u.shape[0] + 2 * pad,
            **compare_fields(run(pad), converged).as_dict(),
        }
    return {
        "selected_pad_width": pad_width,
        "converged_pad_width": converged_pad,
        "sweep_against_converged": sweep,
        "selected_vs_converged": compare_fields(run(pad_width), converged).as_dict(),
        "unpadded_vs_converged": compare_fields(run(0), converged).as_dict(),
        "interpretation": (
            "unpadded_vs_converged is the error a padding-free run would incur and is "
            "the reason the slice pads; selected_vs_converged is what the DECLARED "
            "padding costs, and that is the number the tolerance budget takes. The "
            "sweep exists because edge-energy fraction turned out not to distinguish "
            "the two: it moves by 2x between a run with 1.4e-1 intensity error and a "
            "correct one, so it cannot be the wraparound gate on its own."
        ),
    }


def measure_padding_and_energy(pupil: dict[str, Any]) -> dict[str, Any]:
    from solvers.chromatix.carrier_removed_asm import (
        pin_wave_engine_precision,
    )

    pin_wave_engine_precision()
    from chromatix.functional.propagation import compute_padding_transfer

    u, pitch_m, z_m = pupil["u"], pupil["pitch_m"], pupil["distance_m"]
    automatic = int(compute_padding_transfer(u.shape[0], WAVELENGTH_M, pitch_m, z_m))

    cases = {}
    for name, pad_width in (("none", 0), ("automatic", automatic), ("half_grid", u.shape[0] // 2)):
        out = _chromatix_propagate(
            u, pitch_m=pitch_m, z_m=z_m, pad_width=pad_width, carrier_removed=True
        )
        power_in = _discrete_power(u, pitch_m)
        power_out = _discrete_power(out, pitch_m)
        cases[name] = {
            "pad_width": pad_width,
            "padded_side": u.shape[0] + 2 * pad_width,
            "output_edge_energy_fraction": _edge_energy_fraction(out),
            "power_in": power_in,
            "power_out": power_out,
            "power_ratio": power_out / power_in,
            "peak_intensity": float(np.max(np.abs(out) ** 2)),
        }
    return {
        "input_edge_energy_fraction": _edge_energy_fraction(u),
        "compute_padding_transfer": automatic,
        "cases": cases,
        "reference_free_note": (
            "power_ratio is measured on the CROPPED window, so it is not a "
            "conservation claim about the propagation: light leaving the window is a "
            "real loss of the observable, not of the field."
        ),
    }


def attribute_energy_loss(run_dir: Path, pad_width: int) -> dict[str, Any]:
    """Where does the power that never reaches the focus come from?

    Padding reveals a loss the unpadded run hides by wrapping it around, so the
    ratio has to be attributed rather than reported. The candidate is ray
    sampling: the reconstruction is a sum of discrete wavelets, and between them
    it carries structure the physical wavefront does not, which propagates to
    high angles and leaves the window.

    This is an ATTRIBUTION measurement, not the convergence study -- that is
    CHE-38's, and it owns the tolerance term. Three ray counts is the minimum
    that can distinguish "ray sampling" from "a fixed property of the geometry".
    """
    cases = []
    for num_rays in (8, 16, 24):
        rays = (
            get_ray_adapter()
            .run(
                ModelRunRequest(
                    run_id="che35-energy",
                    node_id=f"n{num_rays}",
                    config={
                        "sample": "M3SingletRef",
                        "num_rays": num_rays,
                        "wavelength": WAVELENGTH_UM,
                        "handoff_plane": "exit_pupil",
                        "output_directory": str(run_dir / f"energy-{num_rays}-rays"),
                    },
                )
            )
            .outputs["rays"]
        )
        derived = _system("M3-SINGLET-REF")["derived"]
        grid = _system("M3-SINGLET-REF")["grid"]
        result = RayToWaveCoupler().transform(
            CouplerRunRequest(
                run_id="che35-energy",
                edge_id=f"e{num_rays}",
                source=rays,
                config={
                    "handoff_plane": "exit_pupil",
                    "handoff_plane_z_m": derived["exit_pupil_z_mm"] * MM_PER_M,
                    "grid_n": int(grid["grid_n"]),
                    "target_sample_pitch_m": float(grid["sample_pitch_m"]),
                    "output_dir": str(run_dir / f"energy-{num_rays}-field"),
                },
            )
        )
        u = np.load(result.target.uri)
        pitch_m = float(grid["sample_pitch_m"])
        z_m = derived["propagation_distance_mm"] * MM_PER_M
        out = _chromatix_propagate(
            u, pitch_m=pitch_m, z_m=z_m, pad_width=pad_width, carrier_removed=True
        )
        intensity = np.abs(out) ** 2
        # Everything inside one Airy radius of the axis, as the "reached the
        # focus" fraction. 0.61*lambda/NA, which is the RADIUS -- see the
        # airy_radius_entry_is_a_diameter item in the protocol.
        airy_radius_m = 0.61 * WAVELENGTH_M / float(derived["numerical_aperture"])
        n = u.shape[0]
        axis = (np.arange(n) - n // 2) * pitch_m
        radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
        core = radius <= 3.0 * airy_radius_m
        reconstruction = result.diagnostics["reconstruction"]
        cases.append(
            {
                "requested_num_rays": num_rays,
                "traced_ray_count": reconstruction["ray_count"],
                "power_ratio_window": _discrete_power(out, pitch_m) / _discrete_power(u, pitch_m),
                "fraction_within_3_airy_radii": float(intensity[core].sum() / intensity.sum()),
                "ray_spacing_estimate_m": reconstruction["ray_spacing_estimate_m"],
                "max_adjacent_ray_phase_rad": reconstruction["max_adjacent_ray_phase_rad"],
                "ray_density_status": reconstruction["ray_density_status"],
            }
        )
    return {
        "airy_radius_definition": "0.61 * lambda / NA (the radius, not the diameter)",
        "cases": cases,
        "reading": (
            "if the retained fraction rises with ray count while the geometry is "
            "unchanged, the loss is ray sampling and belongs to CHE-38. If it is flat, "
            "it is a property of the window and belongs to the padding term."
        ),
    }


def settle_phasor_convention(pupil: dict[str, Any]) -> dict[str, Any]:
    """Which sign does Chromatix's ASM actually implement? Build a wave and look.

    Under this project's declared conventions -- ``exp(-i omega t)`` in time with
    ``exp(+i k z)`` in space -- a wave converging to a focus a distance ``R``
    downstream has pupil field ``exp(-i k sqrt(rho^2 + R^2))``, because the
    optical path still to travel is longest at the pupil edge. Its conjugate is
    the diverging wave.

    Propagate both by ``+R``. Exactly one must concentrate. Which one does is the
    measurement, and it is not inferable from Chromatix's documentation: the
    package declares no time convention.
    """
    pitch_m, z_m, n = pupil["pitch_m"], pupil["distance_m"], pupil["grid_n"]
    y = (np.arange(n) - n // 2) * pitch_m
    radius = np.hypot(*np.meshgrid(y, y, indexing="ij"))
    # A clear aperture at the frozen pupil radius, so the analytic answer is an
    # Airy pattern and the "did it focus" question has a quantitative form.
    aperture = radius <= (n // 2 - 2) * pitch_m
    path_to_focus = np.hypot(radius, z_m)

    converging = np.where(aperture, np.exp(-1j * 2.0 * np.pi * path_to_focus / WAVELENGTH_M), 0.0)
    diverging = np.conjugate(converging)

    out = {}
    for name, field in (("converging", converging), ("diverging", diverging)):
        propagated = _chromatix_propagate(
            field, pitch_m=pitch_m, z_m=z_m, pad_width=0, carrier_removed=True
        )
        intensity = np.abs(propagated) ** 2
        peak_index = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
        out[name] = {
            "peak_intensity": float(np.max(intensity)),
            "on_axis_intensity": float(intensity[n // 2, n // 2]),
            "peak_is_on_axis": tuple(int(i) for i in peak_index) == (n // 2, n // 2),
            "concentration_vs_input_peak": float(np.max(intensity) / np.max(np.abs(field) ** 2)),
        }

    # Quantitative, not just "the bigger one won": a clear circular aperture of
    # radius a focused at R gives an Airy peak of (pi a^2 / (lambda R))^2 relative
    # to unit pupil amplitude. Agreement pins the geometry as well as the sign.
    aperture_radius_m = (n // 2 - 2) * pitch_m
    predicted_peak = (np.pi * aperture_radius_m**2 / (WAVELENGTH_M * z_m)) ** 2
    ratio = out["converging"]["peak_intensity"] / out["diverging"]["peak_intensity"]
    return {
        "input_convention_tested": "exp(-i omega t) with exp(+i k z)",
        "converging_pupil_field": "exp(-i k sqrt(rho^2 + R^2))",
        "results": out,
        "analytic_airy_peak_check": {
            "aperture_radius_m": aperture_radius_m,
            "predicted_peak_intensity": predicted_peak,
            "measured_peak_intensity": out["converging"]["peak_intensity"],
            "ratio_measured_over_predicted": (out["converging"]["peak_intensity"] / predicted_peak),
        },
        "converging_over_diverging_peak_ratio": ratio,
        "verdict": (
            "chromatix asm_propagate implements exp(+i k_z z) for z > 0, which is the "
            "spatial factor this project declares alongside exp(-i omega t). A field "
            "written under the project convention focuses; its conjugate does not."
            if ratio > 10.0
            else "OPPOSITE convention -- the conjugate focused. This invalidates the "
            "project phasor declaration for this engine and must be resolved before "
            "any PSF claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        pupil = build_pupil_field("M3SingletRef", "M3-SINGLET-REF", run_dir)
        padding = measure_padding_and_energy(pupil)
        # Chromatix's own compute_padding_transfer value, declared explicitly
        # rather than left to a default. Half of it looked adequate on edge energy
        # and peak intensity and is 8x worse at the field level -- see the sweep.
        selected_pad_width = padding["compute_padding_transfer"]
        report = {
            "schema_version": 1,
            "issue": "CHE-35 (M3.6)",
            "probe": "m3_pupil_to_focus",
            "protocol_id": "M3-SLICE-CPU-V1",
            "reference_implementation": ASM_ORACLE_ID,
            "system": "M3-SINGLET-REF",
            "wavelength_m": WAVELENGTH_M,
            "grid_n": pupil["grid_n"],
            "sample_pitch_m": pupil["pitch_m"],
            "propagation_distance_m": pupil["distance_m"],
            "pupil_field_source": "M_RAY_OPTILAND -> C_RAY_TO_WAVE (CHE-33, CHE-34)",
            "selected_pad_width": selected_pad_width,
            "padding_and_energy": padding,
            "complex64_cost_unpadded": measure_complex64_cost(pupil, 0),
            "complex64_cost_at_selected_padding": measure_complex64_cost(pupil, selected_pad_width),
            "padding_cost_in_float64": measure_padding_cost_in_float64(pupil, selected_pad_width),
            "energy_loss_attribution": attribute_energy_loss(run_dir, selected_pad_width),
            "phasor_convention": settle_phasor_convention(pupil),
            "status": "passed",
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
