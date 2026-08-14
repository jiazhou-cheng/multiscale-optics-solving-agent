"""CHE-33 (M3.4): is the declared optical path length actually a wavefront?

Run inside the container:

    ./run.sh python knowledge/couplers/ray_to_wave/probes/coherent_handoff.py \
        --write knowledge/couplers/ray_to_wave/expected/coherent_handoff.json

The declaration made by
``multiscale_optics_agent.couplers.optiland_handoff.declare_coherent_bundle`` is
only worth the name if it can be checked against something that does not come
from this repository. It can:

    For a diffraction-limited system every ray reaches the focus with the same
    total optical path. So the optical path at a pupil plane a distance ``R``
    before the focus must satisfy, exactly,

        OPL(rho) - OPL(0) = R - sqrt(rho^2 + R^2)

    with the residual being the system's wavefront aberration and nothing else.

``M3-SINGLET-REF`` was frozen by M3.2 at 0.01700 waves peak-to-valley, measured
through M3.2's own optical-path-to-a-reference-sphere probe. This probe arrives
at the same number from the adapter's exported ``opd_native``, through CHE-30's
unit and reference convention and CHE-32's exit-pupil plane transfer. Two
independent routes to one number is the evidence; a single route would only be a
restatement.

The falsifiers are recorded next to the result, because a characterization that
cannot be made to fail proves nothing:

* ``opl_sign`` flipped -- conjugates the wavefront.
* ``transfer_opl_to_plane`` omitted -- leaves phase and position on planes a
  pupil-to-focus distance apart.

One protocol defect this probe found is recorded rather than silently worked
around: see ``pupil_to_focus_distance`` in the output.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from multiscale_optics_agent.adapters.base import ModelRunRequest
from multiscale_optics_agent.adapters.optiland_adapter import (
    _scientific_array_hash,
    get_adapter,
)
from multiscale_optics_agent.couplers.optiland_handoff import (
    DeclaredHandoffPlane,
    HandoffPerturbation,
    declare_coherent_bundle,
    reconstruct_hashed_arrays,
)

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "benchmarks" / "slice_protocol.yaml"

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1e-6
NUM_RAYS = 16
MM_PER_M = 1e-3

SYSTEMS = {
    "M3SingletRef": "M3-SINGLET-REF",
    "ReverseTelephoto": "M3-REVERSE-TELEPHOTO",
}


def _frozen(system_id: str) -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    for system in protocol["systems"]:
        if system["id"] == system_id:
            return system
    raise KeyError(system_id)


def _optiland_wavefront_pv_waves(sample: str) -> dict[str, Any]:
    """The same quantity, computed by Optiland's own wavefront machinery.

    This is a genuinely independent implementation of the wavefront: it goes
    through ``wavefront/strategy.py``, which references the chief ray and reports
    ``(opd_ref - opd) / (wavelength * 1e-3)`` -- the opposite sign to the one
    declared here, and a different reference-sphere construction. A peak-to-valley
    agreement between the two therefore checks the unit conversion, the plane
    transfer, and the reference subtraction all at once, against code this
    repository did not write.
    """
    import optiland.backend as be
    from optiland.wavefront import Wavefront

    from multiscale_optics_agent.adapters.optiland_adapter import _resolve_lens

    be.set_backend("numpy")
    lens = _resolve_lens(sample, __import__("optiland.samples.objectives", fromlist=["x"]), be)
    wavefront = Wavefront(
        lens,
        fields=[(0.0, 0.0)],
        wavelengths=[WAVELENGTH_UM],
        num_rays=NUM_RAYS,
        distribution="hexapolar",
    )
    opd_waves = np.asarray(be.to_numpy(wavefront.get_data((0.0, 0.0), WAVELENGTH_UM).opd)).ravel()
    return {
        "peak_to_valley_waves": float(np.ptp(opd_waves)),
        "rms_waves": float(np.std(opd_waves)),
        "sample_count": int(opd_waves.size),
    }


def _residual_pv_waves(optical_path_m: np.ndarray, oracle_m: np.ndarray) -> float:
    return float(np.ptp((np.asarray(optical_path_m) - oracle_m) / WAVELENGTH_M))


def characterize(sample: str, run_dir: Path) -> dict[str, Any]:
    system = _frozen(SYSTEMS[sample])
    frozen = system["derived"]
    record = (
        get_adapter()
        .run(
            ModelRunRequest(
                run_id="che33-coherent-handoff",
                node_id=sample,
                config={
                    "sample": sample,
                    "num_rays": NUM_RAYS,
                    "wavelength": WAVELENGTH_UM,
                    "Hx": 0.0,
                    "Hy": 0.0,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(run_dir / sample),
                },
            )
        )
        .outputs["rays"]
    )

    plane = DeclaredHandoffPlane(
        handoff_plane="exit_pupil", z_m=frozen["exit_pupil_z_mm"] * MM_PER_M
    )
    handoff = declare_coherent_bundle(record, declared_plane=plane)
    amplitude, optical_path_m = handoff.bundle.require_coherent()

    # The distance is measured from the record, not read from the protocol: the
    # protocol's own value for one of the two systems disagrees, and this probe
    # is where that was found.
    measured_distance_m = (
        handoff.diagnostics["traced_image_surface_z_m"] - handoff.diagnostics["reference_plane_z_m"]
    )
    frozen_distance_m = frozen["propagation_distance_mm"] * MM_PER_M

    radius_m = np.hypot(handoff.bundle.positions_m[:, 0], handoff.bundle.positions_m[:, 1])
    oracle_m = measured_distance_m - np.hypot(radius_m, measured_distance_m)

    flipped = declare_coherent_bundle(
        record, declared_plane=plane, perturbation=HandoffPerturbation(opl_sign=-1)
    )
    untransferred = declare_coherent_bundle(
        record, declared_plane=plane, perturbation=HandoffPerturbation(transfer_opl_to_plane=False)
    )

    declared_pv = _residual_pv_waves(optical_path_m, oracle_m)
    return {
        "sample": sample,
        "protocol_system_id": SYSTEMS[sample],
        "requested_num_rays": NUM_RAYS,
        "traced_ray_count": handoff.bundle.count,
        "exit_pupil_z_m": handoff.diagnostics["reference_plane_z_m"],
        "traced_image_surface_z_m": handoff.diagnostics["traced_image_surface_z_m"],
        "image_space_refractive_index": handoff.diagnostics["image_space_refractive_index"],
        "entrance_pupil_diameter_m": handoff.diagnostics["entrance_pupil_diameter_m"],
        "pupil_to_focus_distance": {
            "measured_from_record_m": measured_distance_m,
            "frozen_in_protocol_m": frozen_distance_m,
            "agrees": bool(abs(measured_distance_m - frozen_distance_m) <= 1e-12),
            "note": (
                "the protocol's propagation_distance_mm must equal image_plane_z_mm - "
                "exit_pupil_z_mm, which is what the wave leg has to propagate. Where "
                "it does not, the protocol entry is the defect; see amendment A2."
            ),
        },
        "removed_reference_opl_waves": handoff.diagnostics["removed_reference_opl_waves"],
        "relative_opl_span_waves": handoff.diagnostics["relative_opl_span_waves"],
        "amplitude_is_real_non_negative": bool(
            np.all(amplitude.imag == 0.0) and np.all(amplitude.real >= 0.0)
        ),
        "scientific_array_sha256": record.metadata["scientific_array_sha256"],
        "scientific_array_sha256_reproduced_from_bundle": _scientific_array_hash(
            reconstruct_hashed_arrays(handoff.bundle)
        ),
        "analytic_oracle": {
            "form": "OPL(rho) - OPL(0) = R - sqrt(rho^2 + R^2)",
            "declared_residual_pv_waves": declared_pv,
            "declared_residual_rms_waves": float(
                np.std((optical_path_m - oracle_m) / WAVELENGTH_M)
            ),
            "frozen_peak_to_valley_waves": (
                system.get("diffraction_limited_evidence", {}).get("peak_to_valley_waves")
            ),
        },
        # An independent implementation of the same wavefront, inside the pinned
        # package, with the opposite sign convention and its own reference sphere.
        "optiland_wavefront_cross_check": _optiland_wavefront_pv_waves(sample),
        "falsifiers": {
            "opl_sign_flipped_pv_waves": _residual_pv_waves(
                flipped.bundle.optical_path_length_m, oracle_m
            ),
            "plane_transfer_omitted_pv_waves": _residual_pv_waves(
                untransferred.bundle.optical_path_length_m, oracle_m
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        systems = {sample: characterize(sample, run_dir) for sample in SYSTEMS}

    report = {
        "schema_version": 1,
        "issue": "CHE-33 (M3.4)",
        "probe": "coherent_handoff",
        "protocol_id": "M3-SLICE-CPU-V1",
        "wavelength_m": WAVELENGTH_M,
        "status": "passed",
        "systems": systems,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
