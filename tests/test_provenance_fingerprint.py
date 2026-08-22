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

from core.provenance import VOLATILE_KEYS, strip_volatile


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
