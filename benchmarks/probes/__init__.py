"""Probe records: one measurement of the environment, with no oracle to gate on.

Distinct from `benchmarks/systems/` and from `benchmarks/verification/`, and the
distinction is what the record is *for*. A system benchmark composes primitives
and is decided by a closed form. A verification row compares this project's
operation against the third-party tool it delegates to. A **probe** answers a
question about the machine -- where did this buffer land, how long did that take,
what does this library actually do -- and its rows carry no expected value at
all. `benchmarks/record.py`'s `gate`/`oracle_kind` vocabulary would make them
read as decided when nothing decided them, so the drivers here write
`benchmarks/verification/record.py`'s `Row` with `status="BASELINE"`, which is
that module's own name for a recorded value with no oracle.

`knowledge/capabilities/*.json` name their probes by a path under this tree, so
this is the established home for a measurement a capability declaration rests
on. Those particular paths (`benchmarks/probes/precision/...`) are pre-rewrite
and their scripts were deleted with the old source tree; the records they
produced live on in the packs, tagged `pre-rewrite-2026-08-30`.
"""
