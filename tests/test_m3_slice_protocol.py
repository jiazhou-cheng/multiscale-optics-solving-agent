"""CHE-31 (M3.2) — the M3 slice protocol must be frozen before the slice is wired.

M2's L2 records that every M2 number sits at one configuration (16x16, 1 um,
500 nm) and that nothing there establishes grid-independence. So M3 has to
declare its own grid, its own plane, and its own tolerances, and the reason to
freeze them first is that the honest responses to a bad feasibility number --
reduce NA, move the plane, shrink the pupil -- are protocol decisions rather
than implementation details.

These tests pin the clauses that keep the protocol from being quietly relaxed
once real numbers arrive.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/slice_protocol.yaml"
DOCUMENT_PATH = ROOT / "benchmarks/M3_SLICE_PROTOCOL.md"
FEASIBILITY_PROBE = ROOT / "benchmarks/probes/m3_slice_feasibility.py"

WAVELENGTH_M = 5.5e-7


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def _system(protocol: dict, system_id: str) -> dict:
    for entry in protocol["systems"]:
        if entry["id"] == system_id:
            return entry
    raise AssertionError(f"system {system_id!r} is not declared in the protocol")


def test_protocol_extends_the_m2_contract() -> None:
    protocol = _protocol()
    assert protocol["protocol_id"] == "M3-SLICE-CPU-V1"
    assert protocol["extends"] == "M2-COUPLER-CPU-V1"
    assert protocol["scope"]["benchmark_ids"] == ["L2-PSF-01"]


def test_documents_and_probe_exist() -> None:
    assert DOCUMENT_PATH.exists()
    assert FEASIBILITY_PROBE.exists()
    assert "M3-SLICE-CPU-V1" in DOCUMENT_PATH.read_text()


def test_coupler_core_engine_ban_is_inherited_unchanged() -> None:
    """M3 runs both engines, but never inside the coupler core."""
    forbidden = _protocol()["scope"]["forbidden_import_prefixes"]["coupler_core"]
    assert sorted(forbidden) == ["chromatix", "optiland"]


def test_gradients_and_sensor_are_out_of_scope() -> None:
    """Forward-only. The PyTorch->JAX boundary stays forward_only until M4."""
    excluded = " ".join(_protocol()["scope"]["excluded_from_scope"]).lower()
    assert "gradient" in excluded
    assert "m_sensor_ideal" in excluded


def test_handoff_plane_is_read_from_the_system_not_constructed() -> None:
    plane = _protocol()["handoff_plane"]
    assert plane["constructed"] is False
    assert "XPL" in plane["source"] and "XPD" in plane["source"]
    # C_RAY_TO_WAVE accumulates onto a plane; a reference sphere is a different
    # operator and nothing in the repository implements one.
    assert plane["is_a_plane_not_a_sphere"] is True


def test_nyquist_rule_is_per_axis() -> None:
    """A direction-norm test both over- and under-constrains; M2 fixed this once."""
    sampling = _protocol()["sampling"]
    assert "direction_cosine_axis" in sampling["nyquist_rule"]
    assert "pitch_axis" in sampling["nyquist_rule"]
    assert "marginal rays" in sampling["nyquist_evaluated_over"]


@pytest.mark.parametrize("system_id", ["M3-SINGLET-REF", "M3-REVERSE-TELEPHOTO"])
def test_declared_pitch_satisfies_the_declared_nyquist_limit(system_id: str) -> None:
    """The frozen grid must actually obey the rule the protocol states."""
    protocol = _protocol()
    grid = protocol["sampling"]["grids"][system_id]
    numerical_aperture = float(_system(protocol, system_id)["derived"]["numerical_aperture"])

    limit_m = WAVELENGTH_M / (2.0 * numerical_aperture)
    assert grid["nyquist_pitch_max_m"] == pytest.approx(limit_m, rel=1e-6)
    assert grid["sample_pitch_m"] <= grid["nyquist_pitch_max_m"]

    oversampling = float(protocol["sampling"]["oversampling_factor"])
    assert grid["sample_pitch_m"] == pytest.approx(limit_m / oversampling, rel=1e-6)

    # The grid must still span the pupil, which is the constraint pulling the
    # other way: pitch down and extent fixed is what sets the point count.
    assert grid["grid_n"] >= grid["pupil_extent_m"] / grid["sample_pitch_m"] - 1.0


def test_reference_system_is_actually_diffraction_limited() -> None:
    """Rayleigh, with margin -- otherwise the Airy oracle measures the singlet."""
    evidence = _system(_protocol(), "M3-SINGLET-REF")["diffraction_limited_evidence"]
    assert evidence["peak_to_valley_waves"] <= 0.25
    # Taken well inside the limit on purpose; see the protocol document.
    assert evidence["peak_to_valley_waves"] <= 0.05
    assert evidence["marechal_strehl_estimate"] >= 0.99
    # Measured to a sphere centred on the focus, not to the image plane.
    assert "sphere" in evidence["method"]


def test_reference_system_uses_real_refractive_surfaces() -> None:
    """CHE-30: an ideal paraxial surface is not an admissible OPL source."""
    protocol = _protocol()
    assert _system(protocol, "M3-SINGLET-REF")["construction"]["surface_type"] == "standard"

    banned = {entry["id"] for entry in protocol["inadmissible_sources"]}
    assert "optiland_paraxial_surface" in banned
    assert "undeclared_optiland_opl" in banned


def test_aberrated_case_declares_that_no_analytic_oracle_exists() -> None:
    """Stating the absence is what keeps someone from comparing it to Airy."""
    telephoto = _system(_protocol(), "M3-REVERSE-TELEPHOTO")
    assert telephoto["analytic_oracle_available"] is False
    assert telephoto["analytic_oracle_absent_because"].strip()


def test_tolerance_budget_is_a_sum_of_sourced_terms() -> None:
    budget = _protocol()["tolerance_budget"]
    terms = {term["name"]: term for term in budget["terms"]}

    assert "coupler_float64_roundoff" in terms
    assert "chromatix_complex64_truncation" in terms
    assert "ray_sampling_error" in terms
    assert "grid_truncation_and_padding" in terms

    for term in budget["terms"]:
        # Either a measured value with a source, or explicitly owed by a ticket.
        if term["value"] is None:
            assert term["status"] == "to_be_measured"
            assert term["owner"].strip()
        else:
            assert term["source"].strip()


def test_chromatix_term_separates_three_precision_levels() -> None:
    """CHE-40: absolute phase, piston-aligned field, and intensity are distinct.

    M3.2 recorded two levels differing by 10x, because the common piston cancelled
    under squaring. The conditioned path removes that piston up front, so the gap
    does not carry over and the levels have to be stated separately rather than
    derived from one another.
    """
    terms = {t["name"]: t for t in _protocol()["tolerance_budget"]["terms"]}
    chromatix = terms["chromatix_complex64_truncation"]

    assert chromatix["level"].startswith("intensity")
    assert chromatix["applies_to_path"] == "carrier_conditioned"

    levels = chromatix["levels"]
    assert set(levels) == {"absolute_field_phase", "piston_aligned_field", "intensity"}

    # Absolute optical phase is not a number here; it is explicitly not preserved,
    # and stating it as a value would invite someone to claim it.
    assert levels["absolute_field_phase"]["value"] is None
    assert levels["absolute_field_phase"]["status"] == "not_preserved_by_the_required_path"
    # But its cost on the unconditioned path is recorded, so the trade is visible.
    assert levels["absolute_field_phase"]["cost_if_taken_from_the_absolute_path"] > 1e-3

    # The intensity term is not a tenth of the field term any more.
    assert levels["intensity"]["value"] == levels["piston_aligned_field"]["value"]
    assert chromatix["value"] == levels["intensity"]["value"]

    # M3.2's numbers survive as the fallback for an unconditioned path.
    superseded = chromatix["superseded_absolute_phase_path"]
    assert superseded["field"] > superseded["intensity"]


def test_required_conditioning_covers_both_sides_of_the_coupler() -> None:
    """CHE-40: form phase from a difference, on the wave side and the ray side.

    The ray side is recorded rather than implemented -- M3.4 owns the reference
    choice -- but it is recorded here so that M3.4 cannot reach the question
    without meeting the rule.
    """
    conditioning = _protocol()["required_conditioning"]

    wave = conditioning["wave_side"]
    assert "k_z + k" in wave["rule"], "the exact identity, not a subtraction"
    assert "paraxial approximation" in wave["is_not"]
    assert "no term is dropped" in wave["is_not"]
    assert wave["global_phase_policy"] == "retained_as_metadata_not_reapplied"
    # No consumer may quietly assume absolute phase off the propagated field.
    assert "No consumer may read absolute optical phase" in wave["global_phase_policy_detail"]

    ray = conditioning["ray_side"]
    assert "OPL_i - OPL_ref" in ray["rule"]
    assert "CHE-33" in ray["owner"]
    # Implemented by CHE-33. The two things that make the rule real rather than
    # aspirational are that a reference is named and that the removed piston is
    # kept rather than reapplied -- the same policy as the wave side.
    assert ray["status"] == "implemented"
    assert "chief ray" in ray["opl_reference_chosen"]
    assert ray["global_phase_policy"] == "retained_as_metadata_not_reapplied"
    # Conditioning has to be buying something: the removed piston must dominate
    # the retained signal, or the rule would be ceremony.
    for system_id, piston in ray["measured_piston_removed_waves"].items():
        assert piston > 50.0 * ray["measured_signal_retained_waves"][system_id]


def test_residual_aberration_is_scoped_to_the_airy_gate_only() -> None:
    """It is a physical property of the singlet, and the FFT oracle shares it."""
    terms = {t["name"]: t for t in _protocol()["tolerance_budget"]["terms"]}
    aberration = terms["reference_system_residual_aberration"]

    assert aberration["is_an_error"] is False
    assert "Airy" in aberration["applies_to"]

    gates = _protocol()["tolerance_budget"]["gates"]
    assert "aberration" in gates["airy_peak_intensity_relative"]["composition"]
    assert "no aberration term" in gates["fft_oracle_intensity_relative_l2"]["composition"]


def test_airy_gate_is_at_least_the_sum_of_its_named_parts() -> None:
    """A gate tighter than its own budget would fail for a declared reason."""
    protocol = _protocol()
    terms = {t["name"]: t for t in protocol["tolerance_budget"]["terms"]}
    floor = (
        terms["reference_system_residual_aberration"]["value"]
        + terms["chromatix_complex64_truncation"]["value"]
    )
    assert protocol["tolerance_budget"]["gates"]["airy_peak_intensity_relative"]["value"] >= floor


def test_no_gate_may_be_satisfied_by_widening() -> None:
    budget = _protocol()["tolerance_budget"]
    assert budget["no_gate_may_be_satisfied_by_widening"] is True
    assert "finding" in budget["widening_rule"]


def test_feasibility_verdict_records_the_superseded_constraint_and_the_rejection() -> None:
    """The rejected configuration is part of the evidence, not a deleted draft.

    CHE-40 reinstated it. Both facts have to stay legible: M3.2's measurement was
    right for the propagation it had, and the propagation changed.
    """
    feasibility = _protocol()["feasibility"]
    assert feasibility["verdict"] == "feasible_on_cpu_float64"
    assert "carrier_conditioned" in feasibility["binding_constraint"]
    # The superseded constraint is kept rather than overwritten.
    assert "float32" in feasibility["superseded_binding_constraint"]
    assert "CHE-40" in feasibility["binding_constraint_amended_by"]

    rejected = feasibility["rejected_configuration"]
    assert rejected["measured_float32_field_error"] > 1e-2
    assert rejected["propagation_distance_mm"] > 40.0
    assert rejected["status"] == "reinstated_as_admissible"
    # Reinstated on measured evidence, and only on a conditioned path.
    assert rejected["measured_carrier_conditioned_intensity_error"] < 1e-5
    assert rejected["measured_absolute_phase_intensity_error"] > 3.5e-4
    assert "cost, not precision" in rejected["not_selected_because"]


def test_the_one_tenth_scaling_is_a_fallback_not_a_requirement() -> None:
    """CHE-40 removed the numerical reason for it; the honest new reason is cost."""
    scale = _system(_protocol(), "M3-SINGLET-REF")["scale_status"]
    assert scale["value"] == "safe_fallback_configuration"
    assert "CHE-40" in scale["amended_by"]
    assert "required architecture constraint" in scale["was"]
    assert "cost choice" in scale["now"]
    # A later ticket may take the macroscopic system, but not unconditioned.
    assert "may not take it on an unconditioned path" in scale["what_would_change_it"]


def test_conditioned_complex64_term_covers_both_systems_represented_phase() -> None:
    """The budget term must bound `eps32 * max|z(k_z - k)|` on each system's own grid.

    Replaces the M3.2-era check that bounded `eps32 * k z` instead. That quantity
    is no longer what the required path represents, so guarding it would guard the
    wrong thing -- and would keep rejecting systems for a reason CHE-40 retired.
    The rule this enforces still bites: a later ticket that raises NA or lengthens
    the propagation without re-deriving the term fails here.
    """
    protocol = _protocol()
    eps32 = 1.1920929e-7
    terms = {t["name"]: t for t in protocol["tolerance_budget"]["terms"]}
    budget = float(terms["chromatix_complex64_truncation"]["value"])

    for system_id in ("M3-SINGLET-REF", "M3-REVERSE-TELEPHOTO"):
        distance_m = (
            float(_system(protocol, system_id)["derived"]["propagation_distance_mm"]) * 1e-3
        )
        pitch_m = float(protocol["sampling"]["grids"][system_id]["sample_pitch_m"])

        # Corner bin of the sampled band, where the relative phase excursion peaks.
        frequency_squared = 2.0 * (1.0 / (2.0 * pitch_m)) ** 2
        delay = math.sqrt(1.0 - WAVELENGTH_M**2 * frequency_squared)
        excursion = 2.0 * math.pi * distance_m * WAVELENGTH_M * frequency_squared / (delay + 1.0)
        projected = eps32 * excursion

        assert projected <= budget, (
            f"{system_id} represents {excursion:.4g} rad of relative phase, "
            f"projecting {projected:.3g} against a {budget:.3g} budget term"
        )
        # And the conditioning must actually be buying something at this distance.
        carrier_phase = 2.0 * math.pi * distance_m / WAVELENGTH_M
        assert carrier_phase / excursion > 10.0


def test_ray_count_is_a_tested_starting_point_not_a_frozen_answer() -> None:
    criterion = _protocol()["sampling"]["ray_count_criterion"]
    assert criterion["starting_value"] == 4096
    assert "CHE-38" in criterion["tested_by"]
    # Optiland's num_rays is a density request, not an output count.
    assert "density" in criterion["note"]
