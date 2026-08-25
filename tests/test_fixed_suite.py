"""FIXED-V1: what the suite has to prove about itself.

CHE-135 (M4.4). Fixed evaluation stays first-class -- regression, certification,
reproducibility, comparison between agent versions, paper results, held-out
final evaluation -- and what changed is that the instances are *chosen*. The
tests here are the review pass, mechanized:

* every instance argues why THIS setup rather than another point in the family,
  and a justification reducible to "it existed before" is rejected;
* every one of the seven verification statuses is the expected outcome of at
  least one instance, so the suite demonstrates the distinctions it claims;
* every component appears in a REQUIRED instance with an independent oracle,
  which is success metric S1;
* fingerprints reproduce, including across a process boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from core.capabilities import COMPONENT_CAPABILITIES
from core.paths import repository_root
from verification.families import BenchmarkCategory, family
from verification.fixed_suite import (
    FIXED_V1,
    HELDOUT,
    FixedInstance,
    Tier,
    components_without_required_coverage,
)
from verification.status import VerificationStatus

ROOT = repository_root()
MANIFEST = ROOT / "benchmarks" / "instances" / "FIXED-V1.yaml"

sys.path.insert(0, str(ROOT))

from scripts.generate_fixed_suite_manifest import render  # noqa: E402

# --------------------------------------------------------------------------- #
# Chosen, not inherited
# --------------------------------------------------------------------------- #

#: Shapes that are not justifications. "It already existed" is the one the
#: ticket names, and the others are its close relatives.
_NOT_A_JUSTIFICATION = (
    "it already exist",
    "it existed",
    "already in the repo",
    "for completeness",
    "for coverage",
    "historical",
    "we have it",
)


@pytest.mark.parametrize(
    "fixed", FIXED_V1.instances, ids=lambda f: f.instance.instance_id
)
def test_every_instance_argues_for_itself(fixed: FixedInstance) -> None:
    lowered = fixed.justification.lower()
    offending = [phrase for phrase in _NOT_A_JUSTIFICATION if phrase in lowered]
    assert not offending, (
        f"{fixed.instance.instance_id}: {offending} is not a justification. Say what "
        "this particular setup buys that another point in the same family would not."
    )
    assert len(fixed.justification.split()) >= 15


@pytest.mark.parametrize(
    "fixed", FIXED_V1.instances, ids=lambda f: f.instance.instance_id
)
def test_every_instance_is_held_out(fixed: FixedInstance) -> None:
    """M9 constructs the train/validation axes. This issue tags everything held
    out and stops."""
    assert fixed.instance.split_tag == HELDOUT


def test_instance_ids_are_unique() -> None:
    ids = [f.instance.instance_id for f in FIXED_V1.instances]
    assert len(ids) == len(set(ids))


def test_every_instance_belongs_to_a_registered_family() -> None:
    for fixed in FIXED_V1.instances:
        fam = family(fixed.instance.family_id)
        assert fixed.instance.family_version == fam.family_version, (
            "a version bump invalidates an instance's fingerprint; the suite must be "
            "rebuilt rather than carried across"
        )


# --------------------------------------------------------------------------- #
# The two tiers
# --------------------------------------------------------------------------- #


def test_the_two_tiers_are_both_populated_and_disjoint() -> None:
    required = FIXED_V1.tier(Tier.REQUIRED)
    extended = FIXED_V1.tier(Tier.EXTENDED)
    assert required and extended
    assert len(required) + len(extended) == len(FIXED_V1.instances)


def test_no_expensive_family_is_in_the_required_tier() -> None:
    """A required gate nobody can afford to run is not a gate.

    B3 and B4 are the composed and characterization families -- GPU hours and
    published-budget reproductions -- and none of them belongs in a gate that
    runs on every change.
    """
    for fixed in FIXED_V1.tier(Tier.REQUIRED):
        category = family(fixed.instance.family_id).category
        assert category not in (BenchmarkCategory.B3, BenchmarkCategory.B4), (
            f"{fixed.instance.instance_id} is {category.value} and tagged required"
        )


def test_every_required_instance_declares_a_cpu_execution_policy() -> None:
    """Deterministic, CPU, minutes. An instance whose family is CUDA-only cannot
    be in a tier that runs on a machine without a GPU."""
    from core.precision import DeviceKind

    for fixed in FIXED_V1.tier(Tier.REQUIRED):
        policy = family(fixed.instance.family_id).execution_policy
        assert DeviceKind.CPU in policy.devices, (
            f"{fixed.instance.instance_id}: required tier, and its family declares "
            f"{sorted(str(d) for d in policy.devices)}"
        )


def test_every_extended_instance_declares_a_runtime_envelope() -> None:
    """An expensive instance with no declared budget is one nobody can plan
    around."""
    for fixed in FIXED_V1.tier(Tier.EXTENDED):
        policy = family(fixed.instance.family_id).execution_policy
        assert policy.max_wall_seconds is not None


# --------------------------------------------------------------------------- #
# The suite proves the distinctions it claims
# --------------------------------------------------------------------------- #


def test_all_seven_verification_statuses_are_expected_somewhere() -> None:
    """A distinction nothing exercises is one the substrate cannot be trusted to
    make -- the same argument the agent harness's eight outcome codes make about
    themselves.
    """
    expected = {f.expected_status for f in FIXED_V1.instances if f.expected_status}
    missing = set(VerificationStatus) - expected
    assert not missing, f"no instance expects {sorted(s.value for s in missing)}"


def test_the_five_negative_outcomes_come_from_the_required_tier() -> None:
    """They are cheap -- a refusal costs nothing -- so there is no reason for any
    of them to be opt-in."""
    required = {
        f.expected_status for f in FIXED_V1.tier(Tier.REQUIRED) if f.expected_status
    }
    assert {
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.INVALID_CONFIGURATION,
        VerificationStatus.OUT_OF_VALIDITY,
        VerificationStatus.LOSSY_BUT_ALLOWED,
        VerificationStatus.BLOCKED,
    } <= required


def test_success_metric_s1_is_met() -> None:
    """Every component in the capability table appears in at least one REQUIRED
    instance whose family has an independent oracle."""
    gap = components_without_required_coverage()
    assert not gap, (
        f"these components have no required-tier instance with an independent oracle: "
        f"{sorted(gap)}"
    )
    assert set(COMPONENT_CAPABILITIES) == FIXED_V1.components_in(Tier.REQUIRED) | (
        FIXED_V1.components_in(Tier.REQUIRED)
    )


def test_every_registered_family_contributes_at_least_one_instance() -> None:
    """Not for coverage's sake -- a family with nothing frozen is a family whose
    evidence nothing regressions against."""
    from verification.families import FAMILIES

    registered = {f.family_id for f in FAMILIES}
    represented = set(FIXED_V1.families())
    assert registered == represented, f"unrepresented: {sorted(registered - represented)}"


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #


def test_every_fingerprint_reproduces_from_the_declaration() -> None:
    """Rebuild each instance from its own parameters and compare."""
    for fixed in FIXED_V1.instances:
        fam = family(fixed.instance.family_id)
        rebuilt = fam.instantiate(
            fixed.instance.instance_id,
            fixed.instance.parameters,
            seed=fixed.instance.seed,
            split_tag=HELDOUT,
            pinned_fingerprint=fixed.instance.fingerprint,
        )
        assert rebuilt.fingerprint == fixed.instance.fingerprint


def test_the_fingerprints_survive_a_process_boundary() -> None:
    """A committed fingerprint computed with a salted hash would be meaningless.

    Run under a different PYTHONHASHSEED and compare the whole manifest.
    """
    code = (
        "from verification.fixed_suite import FIXED_V1;"
        "print(','.join(f.instance.fingerprint for f in FIXED_V1.instances))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "7", "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=env,
    )
    here = ",".join(f.instance.fingerprint for f in FIXED_V1.instances)
    assert out.stdout.strip() == here


def test_an_infinite_parameter_still_fingerprints() -> None:
    """B2-EQUIV-01 declares an infinite substrate radius -- the planar case --
    and JSON has no literal for it.

    Mapped to a sentinel rather than dropped or coerced, so the planar instance
    keeps a stable fingerprint distinguishable from a very large finite radius.
    """
    import math

    planar = FIXED_V1.by_id("B2-EQUIV-01")
    assert math.isinf(float(planar.instance.parameters["substrate_radius_m"]))

    fam = family("B2-EQUIV")
    large = fam.instantiate(
        "large-but-finite",
        {**planar.instance.parameters, "substrate_radius_m": 1e12},
        split_tag=HELDOUT,
    )
    assert large.fingerprint != planar.instance.fingerprint


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_committed_manifest_matches_the_declaration() -> None:
    assert MANIFEST.is_file()
    assert MANIFEST.read_text(encoding="utf-8") == render(), (
        "benchmarks/instances/FIXED-V1.yaml no longer matches "
        "src/verification/fixed_suite.py. Regenerate it:\n"
        "    ./run.sh python scripts/generate_fixed_suite_manifest.py"
    )


def test_the_manifest_carries_the_fingerprint_and_the_justification() -> None:
    """It is the artifact somebody reads without opening Python."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["suite"] == "FIXED-V1"
    assert len(manifest["instances"]) == len(FIXED_V1.instances)
    for entry in manifest["instances"]:
        assert len(entry["fingerprint"]) == 64
        assert entry["justification"]
        assert entry["split_tag"] == HELDOUT
        assert entry["tier"] in {"required", "extended"}


def test_fixed_results_are_not_pooled_with_generated_ones() -> None:
    """There are no generated instances yet, and the suite is structured so that
    when there are, they cannot be mixed in by accident: FIXED_V1 holds only
    CANONICAL-origin instances, and a GENERATED one would fail this."""
    from verification.families import InstanceOrigin

    for fixed in FIXED_V1.instances:
        assert fixed.instance.origin is InstanceOrigin.CANONICAL


# --------------------------------------------------------------------------- #
# The non-generative declarations the ticket enumerates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family_id",
    ["B2-R2W-EXACT", "B3-DEMO2", "B0-UNITS", "B0-CONTRACT", "B3-DUALROUTE"],
)
def test_the_deliberately_non_generative_families_say_why(family_id: str) -> None:
    """Each of these is non-generative for a *different* reason, and the ticket
    enumerates them: the enumeration is the oracle; the published setup is what
    is certified; the measured wrong number is the artifact; a code needs an
    instance rather than a region; the non-independence finding is specific to
    two Optiland routes."""
    fam = family(family_id)
    assert fam.sampler is None
    assert fam.sampler_absent_reason is not None
    assert len(fam.sampler_absent_note.split()) >= 10


def test_no_family_in_the_suite_has_a_sampler_yet() -> None:
    """M9's non-goal, asserted so it stays one until then."""
    for family_id in FIXED_V1.families():
        assert family(family_id).sampler is None
