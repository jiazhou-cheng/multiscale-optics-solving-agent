"""Runnable system-level benchmarks for the new tree.

First entry landed by CHE-212 (R06.7). See `benchmarks/README.md` for the layout,
the record format, and how a benchmark is invoked.

A benchmark composes the project's **public** vocabulary -- `sources`,
`solvers.chromatix`, `operators` -- and calls no backend function directly. The
composition happens here and not in `src/` because the dependency allowlist
forbids it there: `operators/` may not import `solvers/`, so no production
package can hold a graph that uses both. That is deliberate, and the mild
awkwardness of a script calling four public functions in order is real evidence
for the composition layer R12/R13 will design. This package is **not** that
layer: there is no `System` class, no pipeline, and no composite operator.
"""
