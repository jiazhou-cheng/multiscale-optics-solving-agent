"""CHE-21 — the M2 coupler protocol must be frozen before any coupler physics.

M1 baselines were analytic and used no RNG, so a single number could be
compared against a single oracle. C_WAVE_TO_RAY is a Monte Carlo estimator, so
"reproducible" and "accurate" become two separate claims. These tests pin the
clauses that keep those two claims separate, and that keep a stochastic result
from being repaired by re-rolling a seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/coupler_protocol.yaml"
DOCUMENT_PATH = ROOT / "benchmarks/M2_COUPLER_PROTOCOL.md"


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def test_coupler_protocol_extends_the_m1_contract() -> None:
    protocol = _protocol()

    assert protocol["protocol_id"] == "M2-COUPLER-CPU-V1"
    assert protocol["extends"] == "M1-BASELINE-CPU-V1"
    assert protocol["scope"]["benchmark_ids"] == ["L2-COUPLER-01"]
    assert protocol["execution"]["entrypoint"] == "./run.sh"
    assert protocol["execution"]["device"] == "cpu"
    assert protocol["execution"]["warmup_runs"] >= 1
    assert protocol["execution"]["measured_repeats"] >= 5
    assert protocol["execution"]["timing"]["primary_statistic"] == "median"
    assert protocol["result_sections"]["accuracy"]["separate_from_performance"]
    assert protocol["result_sections"]["performance"]["separate_from_accuracy"]


def test_coupler_core_may_not_import_either_engine() -> None:
    """M1 proved the engines are independently correct. If the coupler core
    could import one, a coupler defect could be misattributed to engine
    behavior and M1's evidence would stop bounding the search."""
    forbidden = _protocol()["scope"]["forbidden_import_prefixes"]["coupler_core"]

    assert set(forbidden) == {"optiland", "chromatix"}


def test_stochastic_evidence_cannot_be_satisfied_by_determinism_alone() -> None:
    stochastic = _protocol()["stochastic_estimator"]

    assert stochastic["seed_policy"]["explicit_seed_required"] is True
    # Bitwise reproducibility must be labelled as a reproducibility claim only.
    assert "never evidence" in stochastic["seed_policy"]["rule"]

    required = stochastic["required_evidence"]
    assert set(required) == {
        "exactness_limit",
        "unbiasedness",
        "convergence_order",
        "variance_comparison",
    }

    # The exactness limit removes sampling as an excuse before any stochastic
    # claim is made, and its tolerance is derived rather than chosen.
    assert required["exactness_limit"]["tolerance_basis"] == "dtype_roundoff_derived"

    # The unbiasedness tolerance IS the measured standard error.
    unbiasedness = required["unbiasedness"]
    assert unbiasedness["min_realizations"] >= 32
    assert unbiasedness["k_sigma"] == 3
    assert "measured standard error, not a chosen constant" in unbiasedness["rule"]

    # Convergence is gated on a fitted exponent, not on one point.
    convergence = required["convergence_order"]
    assert convergence["expected_exponent"] == -0.5
    assert 0 < convergence["exponent_tolerance"] <= 0.1
    assert convergence["min_sweep_points"] >= 5


def test_protocol_forbids_repairing_a_stochastic_result_by_reselection() -> None:
    forbidden = " ".join(_protocol()["stochastic_estimator"]["forbidden"]).lower()

    assert "best of several seeds" in forbidden
    assert "standard error" in forbidden
    assert "after seeing the metric" in forbidden


def test_differentiability_defaults_to_unverified_with_named_promotion_evidence() -> None:
    """SI S7.2 detaches the sampling density and holds sampled directions
    fixed, so the estimator is knowingly biased. The protocol must not let a
    passing optimization stand in for derivative evidence."""
    differentiability = _protocol()["differentiability"]

    assert differentiability["default_claim"] == "not_verified"
    evidence = " ".join(differentiability["required_evidence_before_any_claim"]).lower()
    assert "finite-difference" in evidence
    assert "bias magnitude" in evidence
    assert "omits" in evidence
    assert "named coupler direction" in differentiability["promotion_rule"]


def test_bundle_requires_stochastic_artifacts_and_a_physics_only_fingerprint() -> None:
    protocol = _protocol()

    assert set(protocol["artifacts"]["additional_required"]) == {
        "convergence.json",
        "ensemble_statistics.json",
    }

    excludes = " ".join(protocol["reproducibility"]["scientific_fingerprint"]["excludes"])
    # The M1.8 defect: per-case wall-clock leaked into the wave fingerprint and
    # made it track machine load instead of physics.
    assert "wall-clock" in excludes
    assert "peak memory" in excludes
    assert "process identifiers" in excludes
    assert "m1_bundle" in protocol["reproducibility"]["scientific_fingerprint"]["rule"]


def test_blocked_failed_and_unconverged_are_three_different_outcomes() -> None:
    failure = _protocol()["failure"]

    assert failure["status"] == "blocked"
    assert failure["solver_values_permitted"] is False
    distinction = failure["distinction"].lower()
    assert "no well-defined number exists" in distinction
    assert "disagree" in distinction
    assert "neither" in distinction


def test_schemas_accept_the_coupler_benchmark_and_its_new_sections() -> None:
    result_schema = json.loads((ROOT / "benchmarks/schemas/result.schema.json").read_text())
    provenance_schema = json.loads((ROOT / "benchmarks/schemas/provenance.schema.json").read_text())

    assert "L2-COUPLER-01" in result_schema["properties"]["benchmark_id"]["enum"]
    assert "L2-COUPLER-01" in provenance_schema["properties"]["benchmark_id"]["enum"]
    assert "M2-COUPLER-CPU-V1" in result_schema["properties"]["protocol_id"]["enum"]
    assert "M2-COUPLER-CPU-V1" in provenance_schema["properties"]["protocol_id"]["enum"]

    stochastic = result_schema["properties"]["stochastic"]
    assert set(stochastic["required"]) >= {
        "seed",
        "realizations",
        "exactness_limit",
        "unbiasedness",
        "convergence",
    }
    assert stochastic["properties"]["realizations"]["minimum"] >= 32
    assert (
        stochastic["properties"]["exactness_limit"]["properties"][
            "enumerated_all_propagating_bins"
        ]["const"]
        is True
    )

    differentiability = result_schema["properties"]["differentiability"]
    claims = differentiability["properties"]["claim"]["enum"]
    # CHE-28 added "characterized_unbiased_in_regime": the estimator turned out
    # not to be detectably biased in the regime measured, and forcing that
    # result into either "characterized_biased" or "verified" would have
    # misstated it. Every claim below "verified" must still be a
    # characterization, so exactly one promoting value may exist.
    assert set(claims) == {
        "not_verified",
        "characterized_biased",
        "characterized_unbiased_in_regime",
        "verified",
    }
    assert sum(claim == "verified" for claim in claims) == 1
    assert "omitted_terms" in differentiability["required"]

    assert "coupler" in result_schema["properties"]["reproducibility"]["properties"]["branch"]["enum"]


def test_protocol_document_states_the_boundary_contract() -> None:
    document = DOCUMENT_PATH.read_text()

    # The conventions M1 pinned are the coupler's contract; the document must
    # restate them rather than leave them to be rediscovered.
    for clause in (
        "M2-COUPLER-CPU-V1",
        "10.1021/acsphotonics.6c00818",
        "exp(-i ω t)",
        "exp(+i k z)",
        "⟨n̂, d̂⟩",
        "arcsin(D / 2R)",
        "opd_native",
        "not_verified",
    ):
        assert clause in document, clause

    # The two facts M1 handed forward as hazards must be carried, not dropped.
    assert "sign and reference plane are unverified" in document
    assert "padded arrays" in document
