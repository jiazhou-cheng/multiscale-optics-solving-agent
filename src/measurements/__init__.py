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

__all__ = [
    "COHERENCE_MODEL",
    "NORMALIZATION_DECLARATIONS",
    "PSF_INVARIANTS",
    "PSF_NORMALIZATIONS",
    "PsfNormalization",
    "PsfResult",
    "border_energy_fraction",
    "psf",
]
