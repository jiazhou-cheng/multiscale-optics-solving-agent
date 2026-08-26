# Layer-C systems

CHE-141 (M2.5). The home for **layer-C** benchmark specs, drivers and records —
physically meaningful end-to-end optical systems, as opposed to the primitive
qualification (layer A) and numerical-realization characterization (layer B)
that support them. See [`docs/benchmark_design.md`](../../docs/benchmark_design.md)
for the layer axis and its three consistency rules.

A layer-C family declares a `topology` of at least three stages and at least two
distinct observables. Both are refused at construction if absent, because a
system claim whose chain is not written down cannot be checked against the
system it claims to model, and because no system collapses to a single
threshold.

## Nothing is here yet, and that is the correct state

The system ladder is authored from M2.7 onward — the ideal 4f relay, the
canonical refractive→wave boundary, the aberrated 4f relay, the terminal-DOE
hybrid, the embedded diffractive system, the conformal metasurface, and the SLM
relay. M2.5 installs the axis and the destination; it authors no system family.

## Existing layer-C evidence is re-homed by classification, not moved on disk

`B3-PSF-SINGLET`, `B3-DEMO2` and `B4-DEMO3` are layer C today. Their families
live in `src/verification/families/`, their drivers in `benchmarks/instances/`
and their records in `benchmarks/instances/records/`, and they **stay there**.

That is a deliberate decision, not an omission. Those records carry committed
scientific fingerprints; relocating 58 of them to express a taxonomy change
would invalidate every one for no scientific gain, and the classification is
already expressed by a field on the family. The generated layer view in
[`../INVENTORY.md`](../INVENTORY.md) and
[`../validation/coverage_matrix.md`](../validation/coverage_matrix.md) is what
makes "which evidence supports this system claim" answerable without reading
source.

So this directory is where *new* layer-C artifacts go. It is not a place that
existing evidence is migrating to.
