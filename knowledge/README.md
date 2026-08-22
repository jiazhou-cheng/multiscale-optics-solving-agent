# Agent knowledge

Compact, versioned context disclosed to an agent. **Not** a mirror of upstream
documentation, and — since CHE-92 — not a test tree, an evidence database or a
paper archive either. Those three were 96% of this directory by line count and
they now live where they belong:

| Was here | Is now | Why |
| -- | -- | -- |
| `solvers/*/tutorials/` | `tests_tutorial/cases/<solver>/` | test code, run by a harness that used to `sys.path.insert` back into here |
| `solvers/*/probes/`, `couplers/*/probes/` | `benchmarks/probes/<component>/` | executable evidence, not disclosure |
| `solvers/*/expected/`, `couplers/*/expected/` | `benchmarks/probes/records/<component>/` | recorded outputs of the above |
| two 11.5 MB PDFs | out of git; `papers/raywave_tracing/README.md` keeps the DOI | a binary is not agent context |
| `source_manifest.yaml` | `benchmarks/roadmap_source_catalog.yaml` | a catalogue of ~30 packages this project will not integrate |

The evidence did not disappear, and neither did what it established. A claim on
a card names the record that backs it; the record is in `verification/`.

## Layout

```text
knowledge/
  solvers/{optiland,chromatix}/
    card.yaml            the one card for this component
    conventions.md       units, axes, sign, reference planes
    usage_notes.md       advisory guidance — how to use it correctly
    api_minimal_examples.md
    failure_guide.md     what goes wrong, and what the error means
  couplers/{ray_to_wave,wave_to_ray}/
    card.yaml
    conventions.md
    theory.md            the mathematics, from the paper
    failure_guide.md
    source_manifest.yaml
  papers/raywave_tracing/README.md
```

**One card per component.** There used to be two — a flat routing card in
`solver_cards/` and a deep card in the pack — and the deep one stated the
duplication was deliberate while nothing checked it. They had drifted three
ways, and every drift *understated* what was verified, which wastes effort
rather than causing errors. The deep card absorbed the flat one and
`solver_cards/` is gone.

**`usage_notes.md` was `usage_notes.md`.** "Capability" named two different
layers: `core/capabilities.py`, which is executable and authoritative, and prose
advice, which is neither. The rename leaves one meaning per word.

## What a card may and may not say

A card does **not** restate a device or dtype table. `core/capabilities.py` owns
those, `tests/test_registry_matches_capabilities.py` holds the registry to it,
and a third copy in prose could only ever drift — so a card carries the
*consequences* the table cannot express (that CUDA is torch-backend-only; that
the torch backend defaults to float32 while numpy defaults to float64; that
XLA:GPU silently computes complex64 in TF32) and points at the declaration for
the rest. `tests/test_solver_knowledge_pack.py` fails a card that restates one.

## The validation ladder

Deliberately narrower than scientific validity, and exactly three values:

- **`unvalidated`** — suitable for planning only.
- **`environment_verified`** — the exact package source and version, import,
  minimal CPU forward path, and recorded conventions passed in the supported
  container. This does **not** imply an analytic benchmark or a verified
  gradient.
- **`scientifically_validated`** — the issue-specific analytic or independent
  oracle, and the required convergence checks, also passed.

`validation_scope` is a separate field and carries the qualifier — "scalar
angular-spectrum, complex64, CPU and CUDA" — which is informative and used to be
fused into the status string, where nothing could check either half.

Before unattended execution: require at least `environment_verified`, read
`validation_scope` to see whether your task is inside it, and check the card's
`not_yet_probed` list. That list is a gate, so a stale entry is a defect —
entries are removed when they are cleared, not annotated as done.

## Ticket references

A ticket ID is legitimate in three places: a structured `verified_by:` or
`issue:` field, a historical report, and an attribution line where knowing *who
decided* is itself useful. It is not legitimate as the content of an
explanation.

The test is whether the sentence survives without the link. "The reference
sphere fit recovers the centre to 1e-12 (CHE-37)" does; "CHE-37 verified the
oracle" does not, because the reader now has to leave the repository to find out
what was verified and how well. Where a reference is the only index to a
measurement, replace it with the measurement: what was measured, on which
version and device, the observed value, the tolerance, the record path, and the
test that pins it.

This makes explanations longer, and that is the intended trade.

## Papers

Do not store copyrighted full papers here. One recorded exception: a paper this
project's own authors wrote may be stored in full, in its own subdirectory, with
a `README.md` naming the DOI and stating why the exception applies.
`papers/raywave_tracing/` is that case — and even there, CHE-92 moved the PDFs
out of git, because 11.5 MB of binary in a retrieval-only directory is a cost
every clone pays for something no agent reads.
