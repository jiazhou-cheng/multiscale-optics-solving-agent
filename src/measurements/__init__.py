"""Measurements: observables derived from physical state.

A measurement consumes a representation and produces an **observable**. It is not
a coupler -- it changes no representation -- and its output is not a
representation either, however well it serializes. The physical terminal state of
a simulation remains the `ScalarField` or the `RayBundle` it ended on.

One measurement has landed: `psf`, from CHE-197 (R11.1).

Not here, deliberately: `PSF` as a representation and `PSF.from_complex_field`
(an observable is not physical state at a boundary), `PsfMeasurement` as a second
class beside the result it wraps, `FraunhoferPsf`, `ReferenceSphere`,
`PupilAberration`, `MetricDefinition` and `AnalyticOracle`. The analytic oracles
are **evidence, not infrastructure**: they live under `tests/`, where a comparison
against them is a test rather than a capability the project ships.
"""

from measurements.psf import (
    COHERENCE_MODEL,
    NORMALIZATION_DECLARATIONS,
    PSF_INVARIANTS,
    PSF_NORMALIZATIONS,
    PsfNormalization,
    PsfResult,
    border_energy_fraction,
    psf,
)

#: The public callables in this package that are **semantic operations**, as
#: strings, one per `operations.catalog` record. CHE-221 (R03.4).
#:
#: Strings, and *this package does not import* `operations`: the dependency
#: allowlist gives no implementation package an edge to `operations/`, and an edge
#: would end the one property that package exists to provide -- listing what the
#: project can do would have loaded what it does it with.
#: `tests/operations/test_catalog.py` walks this tuple against the catalog in both
#: directions.
#:
#: Hand-maintained, and deliberately not derived from `__all__`:
#: `border_energy_fraction` is a reduction over an array that `PsfResult` already reports rather
#: than an operation over a representation, and the rest of `__all__` is the normalization
#: vocabulary.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = ("psf",)

__all__ = [
    "COHERENCE_MODEL",
    "NORMALIZATION_DECLARATIONS",
    "OPERATIONS",
    "PSF_INVARIANTS",
    "PSF_NORMALIZATIONS",
    "PsfNormalization",
    "PsfResult",
    "border_energy_fraction",
    "psf",
]
