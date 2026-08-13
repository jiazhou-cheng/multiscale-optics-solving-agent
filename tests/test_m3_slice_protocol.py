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


def test_chromatix_term_records_both_field_and_intensity_levels() -> None:
    """They differ by ~10x, and quoting only one hides a real cost."""
    terms = {t["name"]: t for t in _protocol()["tolerance_budget"]["terms"]}
    chromatix = terms["chromatix_complex64_truncation"]

    assert chromatix["level"].startswith("intensity")
    assert chromatix["field_level_value"] > chromatix["value"]
    assert "piston" in chromatix["field_versus_intensity_note"]


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


def test_feasibility_verdict_names_the_binding_constraint_and_the_rejection() -> None:
    """The rejected configuration is part of the evidence, not a deleted draft."""
    feasibility = _protocol()["feasibility"]
    assert feasibility["verdict"] == "feasible_on_cpu_float64"
    assert "float32" in feasibility["binding_constraint"]

    rejected = feasibility["rejected_configuration"]
    assert rejected["measured_float32_field_error"] > 1e-2
    assert rejected["propagation_distance_mm"] > 40.0


def test_propagation_distance_stays_inside_the_measured_float32_regime() -> None:
    """Both selected systems must sit where the float32 error is ~1e-3, not ~1e-1.

    Guards the decision the rejection was made for: if a later ticket swaps in a
    longer-focal-length system, this fails rather than silently degrading the PSF.
    """
    protocol = _protocol()
    eps32 = 1.1920929e-7
    field_budget = 1e-2

    for system_id in ("M3-SINGLET-REF", "M3-REVERSE-TELEPHOTO"):
        distance_m = (
            float(_system(protocol, system_id)["derived"]["propagation_distance_mm"]) * 1e-3
        )
        projected = eps32 * 2.0 * 3.141592653589793 * distance_m / WAVELENGTH_M
        assert projected < field_budget, f"{system_id} projected float32 error {projected:.3g}"


def test_ray_count_is_a_tested_starting_point_not_a_frozen_answer() -> None:
    criterion = _protocol()["sampling"]["ray_count_criterion"]
    assert criterion["starting_value"] == 4096
    assert "CHE-38" in criterion["tested_by"]
    # Optiland's num_rays is a density request, not an output count.
    assert "density" in criterion["note"]
