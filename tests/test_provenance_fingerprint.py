"""The projection that makes a scientific fingerprint reproducible.

`VOLATILE_KEYS` and `strip_volatile` decide what a benchmark's hash ignores.
They moved from `evaluation/m1_bundle.py` to `core/provenance.py` when the gen1
suite was archived, which turned them from an unguarded corner of a dead module
into public API that two live Level-2 benchmarks depend on.

The tests below are about the two ways the projection can be wrong, and they are
not symmetric:

* **Stripping too little** makes every run's fingerprint unique, so the hash
  answers "did I run this twice?" instead of "did the physics change?". Loud and
  self-correcting -- someone notices immediately.
* **Stripping too much** makes two genuinely different computations hash the
  same. Silent, and it makes the fingerprint actively misleading. That is why
  the second half of this file pins what must *survive* the projection.
"""

from __future__ import annotations

import json
import tempfile
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

from core.provenance import (
    RECORD_PROVENANCE_KEY,
    VOLATILE_KEYS,
    source_fingerprint,
    strip_volatile,
    verify_record_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def _fingerprint(payload: object) -> str:
    return json.dumps(strip_volatile(payload), sort_keys=True, default=float)


def test_every_volatile_key_is_removed_at_the_top_level() -> None:
    payload = {key: "x" for key in VOLATILE_KEYS} | {"metric": 1.0}
    assert strip_volatile(payload) == {"metric": 1.0}


def test_volatile_keys_are_removed_at_every_depth() -> None:
    """Per-case records carry their own timings, so a top-level filter is not enough."""
    payload = {
        "cases": [
            {"id": "a", "residual": 1e-9, "runtime_seconds": 3.1},
            {"id": "b", "residual": 2e-9, "runtime_seconds": 91.7},
        ],
        "summary": {"worst": 2e-9, "process_wall_seconds": 120.0},
    }
    assert strip_volatile(payload) == {
        "cases": [{"id": "a", "residual": 1e-9}, {"id": "b", "residual": 2e-9}],
        "summary": {"worst": 2e-9},
    }


def test_two_runs_that_differ_only_in_execution_detail_hash_the_same() -> None:
    """The property the whole projection exists for."""
    physics = {"cases": [{"id": "airy", "relative_l2": 4.07e-4}]}
    monday = physics | {
        "run_id": "run-2026-08-21-a",
        "timestamp_utc": "2026-08-21T04:11:02Z",
        "runtime_seconds": 18.4,
        "output_directory": "/workspace/outputs/monday",
    }
    tuesday = physics | {
        "run_id": "run-2026-08-22-z",
        "timestamp_utc": "2026-08-22T23:59:58Z",
        "runtime_seconds": 41.9,
        "output_directory": "/tmp/pytest-of-ci/scratch",
    }
    assert _fingerprint(monday) == _fingerprint(tuesday) == _fingerprint(physics)


def test_a_changed_measurement_changes_the_fingerprint() -> None:
    """The other half: the projection must not be so aggressive it hides physics."""
    before = {"cases": [{"id": "airy", "relative_l2": 4.07e-4}], "runtime_seconds": 1.0}
    after = {"cases": [{"id": "airy", "relative_l2": 4.08e-4}], "runtime_seconds": 1.0}
    assert _fingerprint(before) != _fingerprint(after)


def test_what_must_survive_the_projection() -> None:
    """Four things that change *what was computed* and are deliberately kept.

    Each would make the fingerprint claim reproducibility across a real change:
    a dirty tree is not the committed tree; a package bump can move a result;
    and device and dtype are the two axes this project separates most carefully
    (`core/precision.py`). None is a timing or an identity, so none belongs in
    VOLATILE_KEYS -- and a future addition to that tuple should have to argue
    past this test.
    """
    survivors = {
        "git_dirty": True,
        "packages": {"optiland": "0.6.0"},
        "device": "cuda:0",
        "dtype": "complex64",
    }
    assert strip_volatile(survivors) == survivors
    for key in survivors:
        assert key not in VOLATILE_KEYS


def test_scalars_and_empty_containers_pass_through_unchanged() -> None:
    for value in (1, 1.5, "text", None, True, [], {}):
        assert strip_volatile(value) == value


def test_the_projection_does_not_mutate_its_input() -> None:
    """A benchmark hashes the projection and then *writes the full record*.

    If `strip_volatile` mutated in place, the persisted result would silently
    lose its own timings -- the diagnostic information, kept out of the hash
    precisely so it can still be reported.
    """
    payload = {"run_id": "keep-me", "metric": 1.0, "nested": {"runtime_seconds": 2.0}}
    original = json.dumps(payload, sort_keys=True)
    strip_volatile(payload)
    assert json.dumps(payload, sort_keys=True) == original


# ---------------------------------------------------------------------------
# CHE-103: the committed records, checked against the tree that reads them
# ---------------------------------------------------------------------------
#
# The half of the guarantee that was missing. `strip_volatile` above lets two
# results be compared; nothing established that a committed result still
# describes the code in front of you. These tests are the standing gate: for
# every record under `benchmarks/probes/records/`, either its fingerprints still
# match, or it is explicitly declared as not yet enrolled with a reason.
#
# What this does NOT do is re-run the probes -- that is minutes of compute and
# does not belong in a required gate. It establishes that the probes, re-run,
# would produce the same thing. `test_a_record_that_drifts_from_its_probe_fails`
# below demonstrates the detection on a deliberately perturbed record, because a
# safety net nobody has dropped anything into is not evidence of a safety net.

RECORDS_DIR = ROOT / "benchmarks" / "probes" / "records"
REGISTER_PATH = RECORDS_DIR / "REGISTER.yaml"


def _register() -> dict[str, str]:
    return dict(yaml.safe_load(REGISTER_PATH.read_text())["unenrolled"])


def _all_records() -> list[Path]:
    return sorted(RECORDS_DIR.rglob("*.json"))


def _is_unenrolled(relative: str, register: dict[str, str]) -> bool:
    return any(fnmatch(relative, pattern) for pattern in register)


def test_every_committed_record_is_either_enrolled_or_declared() -> None:
    """A new record cannot be added without saying whether it is checked.

    This is the coverage guard. Without it the mechanism decays silently: each
    new unstamped record is individually reasonable and the set of checked
    records quietly stops growing with the set of records.
    """
    register = _register()
    undeclared = [
        path.relative_to(RECORDS_DIR).as_posix()
        for path in _all_records()
        if RECORD_PROVENANCE_KEY not in json.loads(path.read_text())
        and not _is_unenrolled(path.relative_to(RECORDS_DIR).as_posix(), register)
    ]
    assert not undeclared, (
        "these records carry no record_provenance block and are not declared in "
        f"{REGISTER_PATH.relative_to(ROOT)}: {undeclared}. Either stamp them (have "
        "the probe call core.provenance.record_provenance and regenerate) or add "
        "them to the register with the reason they are deferred."
    )


def test_the_register_names_only_records_that_exist() -> None:
    """A stale exemption is worse than none: it reads as coverage and is not."""
    existing = {path.relative_to(RECORDS_DIR).as_posix() for path in _all_records()}
    unmatched = [
        pattern for pattern in _register() if not any(fnmatch(name, pattern) for name in existing)
    ]
    assert not unmatched, (
        f"{REGISTER_PATH.relative_to(ROOT)} exempts records that no longer exist: "
        f"{unmatched}. Remove the entries."
    )


def test_the_register_gives_a_reason_for_every_exemption() -> None:
    empty = [pattern for pattern, reason in _register().items() if not str(reason).strip()]
    assert not empty, f"exempted without a reason: {empty}"


#: The enrolled records, and the probe that regenerates each. Used to make the
#: failure message actionable -- "this is stale" without "run this" is how a gate
#: gets skipped instead of satisfied.
ENROLLED_PROBES = {
    "m3_psf_verification": "psf_oracle_verification",
    "m3_convergence": "pupil_to_focus_convergence",
    "m3_off_axis_handoff": "off_axis_handoff",
    "m3_first_null_grid_convergence": "first_null_grid_convergence",
}


@pytest.mark.parametrize("name", sorted(ENROLLED_PROBES))
def test_the_enrolled_records_still_describe_this_tree(name: str) -> None:
    """The CHE-100 defect, as a failing test rather than a manual investigation.

    If this fails, the record no longer matches the code that produced it. The
    fix is to regenerate the record and say in the commit what changed -- not to
    delete the assertion. The failure message distinguishes a code change from an
    environment change, which is the distinction CHE-100 had to establish by
    checking out a second worktree by hand.
    """
    path = RECORDS_DIR / f"{name}.json"
    verdict = verify_record_provenance(
        json.loads(path.read_text()), root=ROOT, name=path.relative_to(ROOT).as_posix()
    )
    assert verdict.reproduces, (
        f"{verdict.explain()}. Regenerate with: "
        f"./run.sh python benchmarks/probes/{ENROLLED_PROBES[name]}.py"
    )


def test_a_record_that_drifts_from_its_probe_fails(tmp_path: Path) -> None:
    """Demonstrate the detection rather than asserting that it exists.

    A perturbed copy of the tree -- one changed digest, standing in for a source
    file someone edited under a committed record -- must be reported as stale,
    must be attributed to the CODE rather than the environment, and must name the
    file. All three matter: a bare "stale" would send the reader back to the
    bisection this mechanism exists to replace.
    """
    payload = json.loads((RECORDS_DIR / "m3_psf_verification.json").read_text())
    assert verify_record_provenance(payload, root=ROOT, name="unperturbed").reproduces, (
        "the demonstration needs a clean baseline to perturb, and this tree is "
        "already drifted -- see the enrolled-records failures above, which are the "
        "real finding here"
    )

    victim = "src/couplers/quadrature.py"
    files = payload[RECORD_PROVENANCE_KEY]["code_fingerprint"]["files"]
    assert victim in files, (
        f"{victim} should be in the psf probe's fingerprint -- it is the module "
        "CHE-103 attributed the record drift to"
    )
    files[victim] = "0" * 64

    verdict = verify_record_provenance(payload, root=ROOT, name="perturbed")
    assert not verdict.reproduces
    assert verdict.code_changed
    assert not verdict.environment_changed
    assert victim in verdict.changed_files
    assert victim in verdict.explain()
    assert "CODE changed" in verdict.explain()


def test_a_record_with_no_provenance_block_is_refused_rather_than_trusted() -> None:
    """Absence of evidence is reported as absence, not as a pass."""
    verdict = verify_record_provenance({"some": "result"}, root=ROOT, name="unstamped")

    assert not verdict.reproduces
    assert "carries no 'record_provenance' block" in verdict.explain()


def test_a_docstring_edit_does_not_invalidate_a_record() -> None:
    """The sensitivity that decides whether anyone keeps using this.

    Fingerprinting raw bytes would invalidate a record on a typo fix in a
    comment, and a mechanism that cries wolf gets routed around. The digest is
    taken over the AST with docstrings stripped, so prose is free and code is
    not.
    """
    body = "import os\n\n\ndef f(x):\n    return x + 1\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        target = root / "src" / "m.py"

        target.write_text(body)
        original = source_fingerprint([target], root=root)["combined_sha256"]

        target.write_text('"""A docstring."""\n\n' + body.replace("import os", "import os  # note"))
        assert source_fingerprint([target], root=root)["combined_sha256"] == original

        target.write_text(body.replace("x + 1", "x + 2"))
        assert source_fingerprint([target], root=root)["combined_sha256"] != original
