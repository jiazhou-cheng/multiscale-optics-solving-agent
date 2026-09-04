"""The prose that tells an agent where to look must point somewhere that exists.

CHE-203 (R14.3). Three trees, three answers, and — as with R14.1 and R14.2 — the
deletion had already happened. Measured, tag to now: `knowledge/` 40 files -> 3,
`benchmarks/` 360 -> 10, `docs/` 49 -> 2.

`knowledge/` was reduced, and the ticket said keep it
-----------------------------------------------------
The ticket's named risk is "deleting `knowledge/` by association", and that is what
happened: commit `78efe9f` ("Empty the rewrite tree") removed all forty files,
including `knowledge/couplers/ray_to_wave/conventions.md`, which the ticket calls
"the single most valuable file in the repository".

**Its content was carried, and that was verified item by item rather than assumed.**
Every load-bearing item of that file now lives in the new tree's own module
docstrings, with the same ticket citations:

| pack item | where it is now |
| -- | -- |
| H1, the traced path's sign and reference (CHE-30/CHE-41) | `backends/optiland/rays.py` |
| H2, `intensity` is a weight not an amplitude | `AMPLITUDE_MAPPING`, same module |
| H3, the wave backend returns padded arrays | `pad_width`/`padded` on `representations/scalar.py` |
| H4, near-grazing phase cancellation (CHE-70) | `grazing_floor_for_phase_budget` |
| launch amplitude carries the area element | `measure_weight` across eight modules |
| the `1/(i lambda z)` Kirchhoff omission | `SCALE_NOTE` in `couplers/ray_to_scalar.py` |
| the sign and orientation checklist | `PHASOR`/`SPATIAL_FACTOR` in `representations/` |
| the no-wavefront-curvature limitation (CHE-50) | `no_wavefront_curvature_term` in five modules |

So the pack is **not restored**, and the reason is the rewrite's own principle
rather than convenience: a prose card restating conventions that a 700-line module
docstring states beside the code would be the second source the pack's *own* old
README banned ("a third copy in prose could only ever drift").
`knowledge/README.md` now carries that rule in its current form — a record there is
the single source for the measurement it carries, and no package keeps a copy — and
`knowledge/capabilities/` is the one kind of content for which that is true in the
other direction.

What this module gates
----------------------
The part a deletion cannot do. `AGENTS.md` is the file every agent reads first, and
a path in it that resolves nowhere sends work at a tree that is gone. So every
repository path the *instructional* prose names must exist.

`docs/rewrite/reference_inventory.md` is exempt, and the exemption is the point
rather than a hole: citing the deleted tree is that document's entire function --
1,950 lines of "this algorithm was at `src/couplers/ray_to_wave.py:353`" — and it
is what makes the tag readable. A rule that forbade it would forbid the archive's
index.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from test_cutover import JUNK_DRAWER_NAMES

ROOT = Path(__file__).resolve().parents[2]

#: The prose an agent is told to act on, and therefore the prose whose paths must
#: resolve. `AGENTS.md` first because `CLAUDE.md` is one `@AGENTS.md` line.
INSTRUCTIONAL = (
    "AGENTS.md",
    "CLAUDE.md",
    "docs/architecture_principles.md",
    "knowledge/README.md",
    "benchmarks/README.md",
)

#: Exempt, because citing the deleted tree is what it is *for*.
HISTORICAL = ("docs/rewrite/reference_inventory.md",)

#: A path-shaped token: a slash, no spaces, and a file extension or a trailing
#: slash. Extracted from backticks only -- prose here uses them for every path, and
#: a bare `and/or` is not a directory.
_IN_BACKTICKS = re.compile(r"`([^`\n]+)`")
_PATH_LIKE = re.compile(r"^[A-Za-z0-9_./-]+(?:\.(?:md|py|json|yaml|yml|sh|toml)|/)$")

#: Tokens that look like paths and are not references to be resolved.
#:
#: `backends/<backend>/` is a template -- the angle brackets are the tell, and the
#: regex above already excludes them. The rest are named in order to say they must
#: **not** exist: `src/io/` because a top-level `io` package would shadow the
#: standard library's, and the junk-drawer names because a package that names no
#: domain accumulates whatever has no other home.
#:
#: `JUNK_DRAWER_NAMES` is imported from the cut-over gate rather than restated.
#: That gate is where the list is declared and argued; a second copy here would be
#: the two-source arrangement this whole rewrite is about, and it would drift the
#: first time one was added.
DELIBERATE_ABSENCES: frozenset[str] = frozenset(
    {"src/io/", "src/core/", "core/", "src/registry/"}
    | {f"{name}/" for name in JUNK_DRAWER_NAMES}
    | {f"src/{name}/" for name in JUNK_DRAWER_NAMES}
)

def _exists(candidate: str, *, document: str) -> bool:
    """Whether a path a document names resolves, against the three roots prose uses.

    * the **repository root**, for a fully qualified path;
    * `src/`, because the prose writes package names without the prefix -- the
      dependency allowlist in `AGENTS.md` is `numerics/ -> (nothing)`, and
      `docs/architecture_principles.md` follows it, since `src/` is a namespace root
      rather than a package and naming it would suggest otherwise;
    * the **document's own directory**, because a nested README writes relative
      paths: `benchmarks/README.md` says `systems/` about `benchmarks/systems/`.

    Three roots rather than one is a looser check than it looks only if a name
    collides across them, and a false *pass* here means a path that resolves
    somewhere -- which is the property being asserted.

    Falling back to the **frozen tag** if none of the three has it. Instructional
    prose does cite history, and legitimately: `knowledge/README.md` quotes the old
    pack's own rule ("`core/capabilities.py` owns those") in order to say the rule
    is superseded, and the probe-citation contract names `benchmarks/probes/`, which
    resolves only at the tag by design -- `tests/knowledge/test_capability_pack.py`
    runs `git cat-file` on every one.

    This is the rule R14.2 tried and abandoned for *tests*, and it works here for
    the reason it failed there: prose has no absence assertions except the handful
    listed above, and no synthetic fixture paths at all. A path that resolves
    neither now nor at the tag is unambiguously stale.
    """
    for base in (ROOT, ROOT / "src", (ROOT / document).parent):
        if (base / candidate).exists():
            return True
    return _resolves_at_the_tag(candidate)


_AT_THE_TAG: dict[str, bool] = {}


def _resolves_at_the_tag(candidate: str) -> bool:
    """Whether the path is reachable at `pre-rewrite-2026-08-30`. Memoized."""
    if candidate not in _AT_THE_TAG:
        stem = candidate.rstrip("/")
        _AT_THE_TAG[candidate] = any(
            subprocess.run(
                [
                    "git", "-C", str(ROOT), "cat-file", "-e",
                    f"pre-rewrite-2026-08-30:{prefix}{stem}",
                ],
                capture_output=True,
            ).returncode
            == 0
            for prefix in ("", "src/")
        )
    return _AT_THE_TAG[candidate]


def _paths_in(text: str) -> set[str]:
    found: set[str] = set()
    for candidate in _IN_BACKTICKS.findall(text):
        token = candidate.strip().rstrip(".,;:")
        if "/" not in token or " " in token:
            continue
        if _PATH_LIKE.match(token):
            found.add(token)
    return found


@pytest.mark.parametrize("document", INSTRUCTIONAL)
def test_every_path_the_instructional_prose_names_exists(document: str) -> None:
    """Acceptance criteria 1 and 4. A dangling path sends work at a deleted tree.

    Now, or at the frozen tag. That is the rule R14.2 found unworkable for *tests*
    and it works for prose -- see `_exists` on why the two are different in kind.
    What it catches is a path that resolves nowhere, which in instructional prose is
    worse than a path that never existed: an agent will open it.
    """
    path = ROOT / document
    assert path.is_file(), f"{document} is named as instructional prose and does not exist"
    dangling = sorted(
        candidate
        for candidate in _paths_in(path.read_text(encoding="utf-8"))
        if candidate not in DELIBERATE_ABSENCES and not _exists(candidate, document=document)
    )
    assert dangling == [], (
        f"{document} names paths that do not exist:\n  " + "\n  ".join(dangling)
    )


def test_the_extractor_finds_paths_and_not_prose() -> None:
    """The meta-check: an extractor that found nothing would pass everything."""
    found = _paths_in(
        "See `docs/architecture_principles.md` and `src/numerics/arrays.py`, or "
        "`and/or` which is not one, or `backends/<backend>/` which is a template."
    )
    assert found == {"docs/architecture_principles.md", "src/numerics/arrays.py"}
    # And the real files yield a useful number, so the parametrized test is not
    # asserting over an empty set.
    assert len(_paths_in((ROOT / "AGENTS.md").read_text(encoding="utf-8"))) > 5
    # Both resolution roots work, which is what the prose's convention needs.
    assert _exists("src/numerics/arrays.py", document="AGENTS.md")
    assert _exists("numerics/arrays.py", document="AGENTS.md")
    assert _exists("systems/", document="benchmarks/README.md"), "the relative root"
    assert not _exists("numerics/no_such_module.py", document="AGENTS.md")
    # The tag fallback, and that it is not a blanket pass.
    assert _resolves_at_the_tag("core/capabilities.py"), "the tag arm is untested"
    assert not _resolves_at_the_tag("core/no_such_module.py")


def test_the_archive_index_is_exempt_and_still_cites_the_deleted_tree() -> None:
    """The exemption, asserted from the other side.

    If the inventory stopped citing the deleted tree it would have stopped being the
    archive's index, and that is a louder failure than a dangling path: it is the
    only record of what was extracted before the deletion, including the tolerance
    derivations R14.1's risk turns on.
    """
    inventory = ROOT / HISTORICAL[0]
    assert inventory.is_file()
    text = inventory.read_text(encoding="utf-8")
    citations = _paths_in(text)
    into_deleted = {
        candidate
        for candidate in citations
        if candidate.startswith(("src/core/", "src/verification/", "src/couplers/ray_to_wave"))
    }
    assert into_deleted, "the inventory no longer cites the tree it indexes"
    assert len(citations) > 100, len(citations)


# ---------------------------------------------------------------------------
# The knowledge pack: taxonomy, agreement, and the content that was carried
# ---------------------------------------------------------------------------


def test_the_surviving_pack_uses_the_declared_operation_kinds() -> None:
    """Acceptance criterion 2. A pack describing an operator may not call it a coupler.

    The ticket's concern was that "three of the five coupler packs describe what are
    now **physical operators**, not couplers". Those five packs are gone, so what is
    left to check is that the surviving prose does not reintroduce the old taxonomy:
    `knowledge/README.md` may use the words, and where it does they must be the
    declared kinds used correctly -- it describes the five reference rows as
    "couplers and operators whose capability nobody had measured", which is the
    distinction, not a conflation.
    """
    from operations import OperationKind

    text = (ROOT / "knowledge" / "README.md").read_text(encoding="utf-8")
    kinds = {kind.value for kind in OperationKind}
    # The four PRIMITIVE kinds, which are the taxonomy a pack's prose can get
    # wrong. `composed` joined the enum on CHE-237 (R03.7) and is not one of them:
    # it names a fusion of primitives rather than a category of physical effect.
    assert kinds - {"composed"} == {"source", "coupler", "physical_operator", "measurement"}
    # No pack card in the surviving tree, so no card can miscategorize one.
    cards = list((ROOT / "knowledge").rglob("card.yaml"))
    assert cards == [], f"a pack card is back without a taxonomy review: {cards}"
    # And the words it does use are about capability rows, not about a taxonomy the
    # four kinds replaced.
    for retired in ("C_RAY_TO_WAVE", "C_WAVE_TO_RAY", "coupler taxonomy", "model card"):
        assert retired not in text, f"knowledge/README.md uses the retired term {retired!r}"


def test_the_pack_has_an_agreement_check() -> None:
    """Acceptance criterion 3, in the affirmative rather than as an excuse.

    The criterion allows either an agreement check or a plain statement that the
    packs are unenforced. The surviving pack has one:
    `tests/knowledge/test_capability_pack.py` loads every record through the same
    `ComponentCapabilities.__post_init__` the code uses, resolves every probe
    citation with `git cat-file` at the record's own tag, and pins the four measured
    facts. So the pack cannot drift free of the implementation -- which is precisely
    what the deleted `test_{coupler,solver}_knowledge_pack.py` did for the prose
    packs and what their deletion would otherwise have lost.
    """
    check = ROOT / "tests" / "knowledge" / "test_capability_pack.py"
    assert check.is_file()
    text = check.read_text(encoding="utf-8")
    for property_ in ("load_capabilities", "cat-file", "probe_tag", "capability_rows"):
        assert property_ in text, property_
    # And the refusal half is in the loader's own test, which is the file that
    # drives a deliberately widened record through `__post_init__`.
    loader = ROOT / "tests" / "knowledge" / "test_capability_loader.py"
    assert loader.is_file()
    assert "INVALID_CAPABILITY_DECLARATION" in loader.read_text(encoding="utf-8")


def test_the_carried_pack_items_are_in_the_code() -> None:
    """The verification behind not restoring the pack, as a test rather than a claim.

    Eight load-bearing items from `knowledge/couplers/ray_to_wave/conventions.md`,
    each checked to be present in the new tree's own source. This is what makes "the
    content was carried" a fact: if one of these disappeared, the pack's deletion
    would have cost something and this would say which.
    """
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in str(path)
    }
    items = {
        "H2 -- intensity is a weight, not an amplitude": "AMPLITUDE_MAPPING",
        "the omitted Kirchhoff prefactor": "Kirchhoff",
        # Cited by ticket rather than by the native attribute's name: that name is
        # confined to `backends/optiland/` by `test_optiland_boundary.py`, and
        # widening its exemption for a documentation test would be the wrong
        # direction. `CHE-41` is the ticket that characterized the reference
        # surface, and is just as distinctive.
        "H1 -- the traced path's reference surface (CHE-30/CHE-41)": "CHE-41",
        "H4 -- the near-grazing phase floor (CHE-70)": "grazing_floor",
        "the wavefront-curvature limitation (CHE-50)": "no_wavefront_curvature_term",
        "the phasor and spatial-factor conventions": "PHASOR",
        "the launch amplitude's area element": "measure_weight",
        "H3 -- padded arrays from the wave backend": "pad_width",
    }
    missing = [
        f"{label} ({token!r})"
        for label, token in items.items()
        if not any(token in text for text in sources.values())
    ]
    assert missing == [], (
        "an item the deleted knowledge pack carried is not in the new tree's source, "
        "so deleting the pack cost it:\n  " + "\n  ".join(missing)
    )


def test_the_deleted_vendored_data_has_an_analytic_replacement() -> None:
    """Acceptance criterion 5, justified rather than relocated.

    `benchmarks/probes/data/` held the paper's phase masks as `.npy` -- vendored,
    not regenerable, and `.gitignore` deliberately un-ignored them. They are gone
    with the reference tree, and nothing in `src/` or `tests/` reads them.

    The justification is that the replacement is **stronger**, not merely different.
    The tests that consumed a vendored mask were paper reproductions, checkable only
    against a recorded number; the new tree's diffractive tests build an analytic
    pi-phase binary grating whose diffraction orders have a closed form -- a
    pi-deep binary grating must put *zero* energy in every even order, which is a
    falsifiable statement about the physics rather than a comparison to a file.
    """
    # Narrowed by CHE-245 (T1) from `benchmarks/probes` to `benchmarks/probes/data`.
    # This test's subject is the **vendored `.npy` masks**, and that is what must
    # stay gone; `benchmarks/probes/` itself is the path the capability packs cite
    # for the measurement a declaration rests on, and T1 lands three records under
    # it from a committed driver. Asserting the whole tree away would have made the
    # deletion of un-regenerable data also a ban on recording a measurement, which
    # is not what R14.3 decided and not what the docstring above argues.
    assert not (ROOT / "benchmarks" / "probes" / "data").exists()
    consumers = [
        str(path.relative_to(ROOT))
        for tree in (ROOT / "src", ROOT / "tests")
        for path in tree.rglob("*.py")
        if "__pycache__" not in str(path)
        and path.resolve() != Path(__file__).resolve()
        and "probes/data" in path.read_text(encoding="utf-8")
    ]
    assert consumers == [], f"something still reads the vendored data: {consumers}"

    grating = ROOT / "tests" / "physics" / "test_diffractive_surface_full_field.py"
    text = grating.read_text(encoding="utf-8")
    assert "a_binary_phase_grating" in text, "the analytic replacement is gone"
    assert ".npy" not in text and "np.load" not in text, (
        "the diffractive tests read a file again, so the oracle is a recorded number "
        "rather than a closed form"
    )


def test_there_is_no_readme_to_keep_in_step() -> None:
    """Criterion 4's other half, and it is vacuous for a reason worth stating.

    The criterion asks that `AGENTS.md` **and `README.md`** reference only surviving
    paths. There is no root `README.md`: the reference tree's went with it, and
    `AGENTS.md` is the canonical shared context this project actually reads.
    Asserted rather than left unmentioned, so that adding one is a deliberate act
    that brings it under the rule above.
    """
    assert not (ROOT / "README.md").exists(), (
        "a root README.md appeared; add it to INSTRUCTIONAL so its paths are checked"
    )
    assert (ROOT / "AGENTS.md").is_file()
    assert "@AGENTS.md" in (ROOT / "CLAUDE.md").read_text(encoding="utf-8")


def test_the_tree_counts_are_what_the_report_claims() -> None:
    """Criterion 6, pinned so the reported numbers stay checkable.

    Not a ratchet -- a tree may legitimately grow -- but the *shape* is the claim:
    `knowledge/` is a capability pack and a README, `docs/` is two files, and
    `benchmarks/` is a driver plus its records. A tree that quietly reacquired forty
    files would be the pack coming back without the taxonomy review criterion 2 asks
    for.

    `benchmarks/` moved 10 -> 17 for CHE-238 through CHE-240, and the growth is the kind the
    docstring above sanctions rather than the kind it guards against. What arrived is
    one report (`reports/2026-09/`) and one verification harness
    (`verification/`, six modules) for the overnight ray/wave verification run --
    executable evidence and the document it writes into, not prose restating what a
    module already says. The distinction the count exists to catch is a *prose pack*
    reappearing; this is the opposite kind of file.

    `verification/` is deliberately not `systems/`. A benchmark there composes this
    project's public primitives and gates itself on closed-form optics, and
    `tests/benchmarks/test_records.py` enforces that for `systems/*.py` only. A
    verification harness reads a third-party prescription, runs the same
    configuration through this project's catalogued operations, and reports the
    difference -- which needs a direct `optiland` import and has no closed form to
    gate on. Two kinds of thing, two directories.

    `benchmarks/` moved 17 -> 23 for CHE-245 (T1), and it is a **third** kind of
    thing again: `probes/` is a driver, its provenance helper and three records of
    what the machine did -- where a buffer landed, how long a device took, what a
    DLPack bridge costs. Not a benchmark, because nothing decides it: every row is
    `BASELINE`, a recorded value with no oracle. Not a verification harness,
    because there is no third-party implementation on the other side of the
    comparison. `benchmarks/probes/__init__.py` states the three-way distinction,
    and `tests/unit/test_suite_shape.py` holds each of those records to naming a
    script in this tree that rewrites it -- which is the property the deleted
    152-record tree was missing and the reason a record tree is admissible here at
    all.

    Note for whoever next changes this number: it counts `git ls-files`, so a new
    file is invisible to this gate until it is staged. CHE-245's full-suite run
    was green with its six files untracked and this test failed on the very next
    run. Stage first, then run.
    """
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    counts = {
        tree: len([path for path in tracked if path.startswith(f"{tree}/")])
        for tree in ("knowledge", "benchmarks", "docs")
    }
    assert counts == {"knowledge": 3, "benchmarks": 23, "docs": 2}, counts
