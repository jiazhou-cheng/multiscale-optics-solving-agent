# Knowledge-pack completeness audit

CHE-104 (M0.3). Written rather than generated, because the finding is about
*absence* and the interesting part is why each absence matters differently.
The mechanical half — "does each existing pack have all its files" — is
enforced by `tests/test_claim_ledger.py::test_an_existing_knowledge_pack_is_complete`.

A knowledge pack is what an agent reads before using a component. Its absence is
not a documentation debt; it is the component being unusable by the thing this
project is being built for.

## Result

| Component | Pack | Status |
| -- | -- | -- |
| `M_RAY_OPTILAND` | `knowledge/solvers/optiland/` | complete |
| `M_WAVE_CHROMATIX` | `knowledge/solvers/chromatix/` | complete |
| `C_RAY_TO_WAVE` | `knowledge/couplers/ray_to_wave/` | complete |
| `C_WAVE_TO_RAY` | `knowledge/couplers/wave_to_ray/` | complete |
| `C_PLANAR_DOE_STEP` | — | **absent** |
| `C_PATCH_WFT` | — | **absent** |

Required files, per `verification.claim_ledger.KNOWLEDGE_PACK_REQUIRED_FILES`:

* **solver** — `card.yaml`, `conventions.md`, `usage_notes.md`,
  `api_minimal_examples.md`, `failure_guide.md`
* **coupler** — `card.yaml`, `conventions.md`, `failure_guide.md`, `theory.md`

The two sets differ in one place, deliberately. A coupler needs `theory.md`
because it implements a specific equation from a specific paper and the mapping
from equation to code is the thing a reader has to check. A solver needs
`api_minimal_examples.md` and `usage_notes.md` because the hazard is different:
driving somebody else's library correctly.

## The finding

`C_PLANAR_DOE_STEP` and `C_PATCH_WFT` have **no pack at all** — no card, no
conventions, no failure guide, no theory. They are also the only two couplers
besides `C_RAY_TO_WAVE` that have executable graph nodes.

That combination is the whole point. `C_WAVE_TO_RAY` has a complete pack and no
graph node; the two couplers an agent could actually put in a graph are the two
it can read nothing about. A planner asked to compose a DOE step has the
registry entry, which states ports and devices, and nothing that states the
phasor sign, the order-truncation convention, the position-sampling contract, or
what the coupler does when the pad violates its clearance condition — which is
the list of things a coupler gets wrong.

Both are filed in `gap_list.md` at **high**: a claim the repository makes that no
executed evidence supports. Implementation is **M2.3 (CHE-111)**, which is
already scoped to benchmark both couplers and therefore has to establish the
conventions the packs would document. Writing the packs earlier would mean
writing them from the code rather than from evidence, which is how a pack comes
to describe what the code does instead of what is true.

`tests/test_claim_ledger.py::test_a_missing_knowledge_pack_is_filed_as_a_gap`
fails when either directory appears, as the prompt to move the component into
the completeness check and close its gap entry.

## What was not audited

Whether the *content* of the four complete packs is correct or current. This
audit checks that the files exist and that the ledger's claims cite evidence
that resolves; it does not re-derive the conventions each pack states. The one
content change made in this milestone was to
`knowledge/couplers/ray_to_wave/conventions.md`, under CHE-103, which corrected
a launch-amplitude normalization statement that was wrong by `(λz)²`. That
correction was found by review rather than by an audit like this one, which is
worth knowing about the limits of this table.
