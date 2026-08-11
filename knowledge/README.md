# Agent Knowledge Assets

This directory contains compact, versioned information disclosed to the planning and execution agents. It is not a bulk mirror of external documentation.

- `source_manifest.yaml`: authoritative links, roles, frameworks, and ingestion priority.
- `solver_cards/`: small routing and API-safety cards for the first-paper solver set.
- `papers/`: reserved for project-authored notes about related papers; do not store copyrighted full papers.

Validation status is intentionally narrower than scientific validity:

- `unvalidated`: suitable for planning only.
- `environment_verified`: the exact package source/version, import, minimal
  CPU forward path, and recorded conventions passed in the supported
  container. This does not imply an analytic benchmark or verified gradient.
- `scientifically_validated`: the issue-specific analytic or independent
  oracle and required convergence checks also passed.

Before unattended execution, require at least `environment_verified` and
check the card's explicit `not_yet_probed` list against the intended task.
