"""CHE-35 (M3.6): the reconstructed pupil field, propagated to focus.

The wave leg already worked. What did not exist was a *number* for either of the
two losses the adapter used to describe in prose, or a check that the field
crossing into Chromatix comes from a real trace with nothing missing.

The pinned evidence is
``knowledge/solvers/chromatix/expected/m3_pupil_to_focus.json``, recorded by
``knowledge/solvers/chromatix/probes/m3_pupil_to_focus.py``. Tests here assert
the contract behaviour live and compare the measured physics against that
record, rather than re-deriving oracles that the probe already established.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_probe_expected

from multiscale_optics_agent.couplers.base import CouplerRunRequest
from multiscale_optics_agent.couplers.ray_to_wave_node import RayToWaveCoupler

pytest.importorskip("optiland")
pytest.importorskip("chromatix")

from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.adapters.chromatix_adapter import (
    _EXPECTED_PHASOR,
)
from multiscale_optics_agent.adapters.chromatix_adapter import (
    get_adapter as get_wave_adapter,
)
from multiscale_optics_agent.adapters.optiland_adapter import (
    get_adapter as get_ray_adapter,
)

EXPECTED = load_probe_expected("chromatix", "m3_pupil_to_focus")

PUPIL_Z_M = 0.06814345991561233e-3
IMAGE_Z_M = 4.90560476022521e-3
DISTANCE_M = IMAGE_Z_M - PUPIL_Z_M
PITCH_M = 2.6587352810843895e-06

# The frozen M3-SLICE-CPU-V1 configuration, not a cheaper stand-in. A smaller
# window makes the pupil overfill it, and the edge-energy diagnostic then fires
# correctly -- which would have made "no warning fired" a statement about the
# test grid rather than about the slice. The ray count is reduced instead, which
# does not change the window.
GRID_N = 188
PAD_WIDTH = 566


@pytest.fixture(scope="module")
def pupil_field(tmp_path_factory):
    """A real M3.5 output: Optiland trace -> C_RAY_TO_WAVE -> ComplexField record."""
    out = tmp_path_factory.mktemp("m36")
    rays = (
        get_ray_adapter()
        .run(
            ModelRunRequest(
                run_id="che35",
                node_id="lens",
                config={
                    "sample": "M3SingletRef",
                    "num_rays": 8,
                    "wavelength": 0.55,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(out / "rays"),
                },
            )
        )
        .outputs["rays"]
    )
    result = RayToWaveCoupler().transform(
        CouplerRunRequest(
            run_id="che35",
            edge_id="pupil",
            source=rays,
            config={
                "handoff_plane": "exit_pupil",
                "handoff_plane_z_m": PUPIL_Z_M,
                "grid_n": GRID_N,
                "target_sample_pitch_m": PITCH_M,
                "output_dir": str(out / "field"),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    return result.target


def _propagate(pupil_field, tmp_path, **config):
    base = {
        "propagation": "angular_spectrum",
        "propagation_method": "asm_carrier_removed",
        "target_plane_z_m": IMAGE_Z_M,
        "pad_width": PAD_WIDTH,
        "output_dir": str(tmp_path),
    }
    base.update(config)
    return get_wave_adapter().run(
        ModelRunRequest(
            run_id="che35", node_id="wave", inputs={"input_field": pupil_field}, config=base
        )
    )


# --- AC1: the coupler's record is accepted, with nothing missing and nothing warned


def test_the_coupler_record_is_accepted_with_no_metadata_gap_and_no_warning(pupil_field, tmp_path):
    """Every key `_run_asm_propagate` reads is populated by the coupler.

    "No warning fired" is the substantive half: both warnings this adapter used
    to emit on this path -- the complex128 cast and the phasor -- are now a
    measured number and an established convention respectively, so silence here
    means they were resolved rather than suppressed.
    """
    report = get_wave_adapter().validate_request(
        ModelRunRequest(
            run_id="che35",
            node_id="wave",
            inputs={"input_field": pupil_field},
            config={"propagation": "angular_spectrum", "target_plane_z_m": IMAGE_Z_M},
        )
    )
    assert report.valid, [issue.model_dump() for issue in report.errors]

    result = _propagate(pupil_field, tmp_path)
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert result.warnings == [], result.warnings
    assert pupil_field.metadata["phasor"] == _EXPECTED_PHASOR


# --- AC2: the complex64 cast is a number ------------------------------------


def test_the_complex64_cast_is_measured_not_warned_about(pupil_field, tmp_path):
    result = _propagate(pupil_field, tmp_path)
    truncation = result.diagnostics["complex64_input_truncation"]

    assert truncation is not None, "a complex128 input must be measured, not passed silently"
    assert truncation["relative_field_error"] > 0.0
    # The input cast alone is far below every gate; the transfer-function rounding
    # is the part that depends on the propagation method, and the diagnostic says so.
    assert truncation["relative_field_error"] < 1e-6
    assert "propagation_method" in truncation["scope"]


def test_the_pinned_measurement_stays_inside_the_protocol_budget():
    """The measured complex64 cost must sit under the term the protocol derived.

    The budget number was derived from `eps32 * max|z(k_z - k)|`, deliberately not
    read off a run, so this is a real comparison rather than a restatement. It is
    also the check that would fail if a later ticket lengthened the propagation or
    raised the NA without re-deriving the term.
    """
    from pathlib import Path

    import yaml

    protocol = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "benchmarks" / "slice_protocol.yaml").read_text()
    )
    terms = {term["name"]: term for term in protocol["tolerance_budget"]["terms"]}
    budget = float(terms["chromatix_complex64_truncation"]["value"])

    conditioned = EXPECTED["complex64_cost_at_selected_padding"]["carrier_conditioned_path"]
    assert conditioned["relative_intensity_l2_error"] < budget

    # And the conditioning has to be buying something, or the protocol's
    # requirement would be ceremony.
    absolute = EXPECTED["complex64_cost_at_selected_padding"]["absolute_carrier_path"]
    assert (
        absolute["relative_intensity_l2_error"] > 50.0 * conditioned["relative_intensity_l2_error"]
    )


# --- AC3: the phasor convention is established, and enforced -----------------


def test_chromatix_focuses_a_field_written_under_the_project_convention():
    """Settled by measurement, and quantitatively: the analytic Airy peak.

    A converging pupil field under `exp(-i omega t)` / `exp(+i k z)` reaches
    0.990 of `(pi a^2 / (lambda R))^2`. Its conjugate does not focus at all. The
    conjugate control is what makes this a measurement rather than an assertion:
    exactly one of the two must concentrate, and which one is the answer.
    """
    phasor = EXPECTED["phasor_convention"]
    assert phasor["results"]["converging"]["peak_is_on_axis"] is True
    assert phasor["results"]["diverging"]["peak_is_on_axis"] is False
    assert phasor["converging_over_diverging_peak_ratio"] > 100.0
    assert phasor["analytic_airy_peak_check"]["ratio_measured_over_predicted"] == pytest.approx(
        1.0, abs=0.02
    )


def test_a_mismatched_phasor_is_refused_rather_than_forwarded(pupil_field, tmp_path):
    """It used to be a warning. For a converging field it is focus versus defocus."""
    mismatched = pupil_field.model_copy(deep=True)
    mismatched.metadata["phasor"] = "exp(+i omega t)"

    report = get_wave_adapter().validate_request(
        ModelRunRequest(
            run_id="che35",
            node_id="wave",
            inputs={"input_field": mismatched},
            config={"propagation": "angular_spectrum", "target_plane_z_m": IMAGE_Z_M},
        )
    )
    assert not report.valid
    assert any(issue.code == "CHROMATIX_PHASOR_MISMATCH" for issue in report.errors)

    result = _propagate(mismatched, tmp_path)
    assert result.status is RunStatus.FAILED
    assert result.diagnostics["code"] == "CHROMATIX_PHASOR_MISMATCH"
    assert "output_field" not in result.outputs


# --- AC4: padding is declared and tested -------------------------------------


def test_the_declared_padding_is_the_measured_one_not_a_default():
    """`compute_padding_transfer`'s value is used, and the sweep says why.

    Half of it looked adequate on edge energy (0.01744 vs 0.01743) and on peak
    intensity (0.01%), and is 73x worse at the intensity level. That is the whole
    argument for not trusting the cheap indicators.
    """
    padding = EXPECTED["padding_cost_in_float64"]
    selected = padding["selected_pad_width"]
    assert selected == EXPECTED["padding_and_energy"]["compute_padding_transfer"]

    sweep = padding["sweep_against_converged"]
    assert sweep[str(selected)]["relative_intensity_l2_error"] < 2.0e-5
    assert sweep["0"]["relative_intensity_l2_error"] > 0.1
    # Monotone improvement with padding, so the selected value is on the converged
    # side of the knee rather than at a lucky point.
    errors = [sweep[key]["relative_intensity_l2_error"] for key in sorted(sweep, key=int)]
    assert errors == sorted(errors, reverse=True)


def test_edge_energy_is_reported_and_is_declared_a_weak_indicator(pupil_field, tmp_path):
    result = _propagate(pupil_field, tmp_path)
    diagnostics = result.diagnostics
    assert 0.0 <= diagnostics["input_edge_energy_fraction"] <= 1.0
    assert 0.0 <= diagnostics["output_edge_energy_fraction"] <= 1.0
    assert diagnostics["edge_energy_reporting_threshold"] == 0.05
    assert "not to certify padding" in diagnostics["edge_energy_is_a_weak_wraparound_indicator"]


# --- AC5: energy accounting ---------------------------------------------------


def test_power_accounting_states_what_is_conserved_and_what_left_the_window(pupil_field, tmp_path):
    result = _propagate(pupil_field, tmp_path)
    diagnostics = result.diagnostics
    assert diagnostics["power_conservation_ratio"] is not None
    assert "left the window" in diagnostics["power_accounting"]
    # The trap this sentence exists to name: an unpadded run reads 1.0 because
    # wraparound recirculates the light, so 1.0 is not evidence of correctness.
    unpadded = _propagate(pupil_field, tmp_path, pad_width=0)
    assert unpadded.diagnostics["power_conservation_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert (
        diagnostics["power_conservation_ratio"] < unpadded.diagnostics["power_conservation_ratio"]
    )


def test_the_window_loss_is_attributed_to_ray_sampling():
    """The deficit is not a propagation loss, and the probe says which one it is.

    The focused fraction rises 9x as the traced ray count goes 217 -> 1801 with
    the geometry untouched, while the adjacent-ray phase step falls 2.53 -> 0.28
    rad. That is ray sampling, so the number belongs to CHE-38's convergence study
    and must not be entered as a padding term here.
    """
    cases = EXPECTED["energy_loss_attribution"]["cases"]
    focused = [case["fraction_within_3_airy_radii"] for case in cases]
    phase_step = [case["max_adjacent_ray_phase_rad"] for case in cases]

    assert focused == sorted(focused), "focused fraction must rise with ray count"
    assert focused[-1] > 5.0 * focused[0]
    assert phase_step == sorted(phase_step, reverse=True)
    # Nowhere near converged at these counts, which is the point of saying so.
    assert focused[-1] < 0.5


# --- AC6: the distance is derived from two declared planes -------------------


def test_the_distance_is_derived_from_the_declared_planes(pupil_field, tmp_path):
    result = _propagate(pupil_field, tmp_path)
    assert result.outputs["output_field"].metadata["z_m"] == pytest.approx(DISTANCE_M, abs=1e-15)
    assert result.outputs["output_field"].metadata["source_plane_z_m"] == pytest.approx(
        PUPIL_Z_M, abs=1e-15
    )


def test_a_distance_that_disagrees_with_the_planes_is_a_structured_refusal(pupil_field, tmp_path):
    """The exact defect CHE-33 found in the frozen protocol, made unrepeatable.

    4.705605 mm (the back focal length) against a pupil-to-image distance of
    4.837461 mm is 0.311 waves of defocus. It produces a perfectly ordinary-looking
    PSF, so nothing but this check would catch it.
    """
    result = _propagate(pupil_field, tmp_path, z_m=4.70560476022521e-3)
    assert result.status is RunStatus.FAILED
    assert result.diagnostics["code"] == "CHROMATIX_PROPAGATION_DISTANCE_MISMATCH"
    assert "defocus, not a piston" in result.error_message


def test_a_bare_distance_still_works_for_callers_with_no_declared_target(pupil_field, tmp_path):
    """M1's callers pass z_m alone; the plane check is additive, not mandatory."""
    result = get_wave_adapter().run(
        ModelRunRequest(
            run_id="che35",
            node_id="wave",
            inputs={"input_field": pupil_field},
            config={
                "propagation": "angular_spectrum",
                "z_m": DISTANCE_M,
                "pad_width": PAD_WIDTH,
                "output_dir": str(tmp_path),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert result.outputs["output_field"].metadata["propagation_method"] == "asm_propagate"


# --- The carrier policy travels with the artifact ----------------------------


def test_the_carrier_removed_path_marks_its_absolute_phase_unphysical(pupil_field, tmp_path):
    """CHE-40's policy has to be readable off the record, not only in the caller."""
    conditioned = _propagate(pupil_field, tmp_path).outputs["output_field"].metadata
    assert conditioned["propagation_method"] == "asm_carrier_removed"
    assert conditioned["absolute_phase_is_physical"] is False
    assert conditioned["removed_carrier_phase_rad"] == pytest.approx(
        2.0 * np.pi * DISTANCE_M / 5.5e-7, rel=1e-12
    )

    absolute = (
        _propagate(pupil_field, tmp_path, propagation_method="asm_propagate")
        .outputs["output_field"]
        .metadata
    )
    assert absolute["absolute_phase_is_physical"] is True
    assert absolute["removed_carrier_phase_rad"] is None


def test_an_unknown_propagation_method_is_refused(pupil_field, tmp_path):
    result = _propagate(pupil_field, tmp_path, propagation_method="fresnel_ish")
    assert result.status is RunStatus.FAILED
    assert result.diagnostics["code"] == "CHROMATIX_UNSUPPORTED_PROPAGATION_METHOD"
