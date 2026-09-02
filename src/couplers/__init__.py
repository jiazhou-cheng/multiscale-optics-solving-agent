"""Couplers: operations that change representation while preserving the state.

A coupler re-describes the *same* physical state at the *same* boundary in a
different representation. That is the line `docs/architecture_principles.md`
section 2 draws against a physical operator, which changes the state -- and heavy
numerics do not move an operation across it. `couplers.ray_to_scalar` is
`O(N_rays x N_pixels)` of complex exponentials and it still propagates nothing,
moves no surface and touches no `z_m`; the 7.1e-15 round trip its default
projection convention was chosen by is the evidence that it really preserves the
field rather than approximating it.

Representation bridging is central to this project, which is why couplers are
first-class assets here rather than glue. It is also why the package is small: one
module per direction, and a route through it is an *argument*, not a second
operation.

`couplers/` may import `representations/` and `numerics/`. It must not import a
solver, a problem or a backend -- the coupler core is the physics under test, and
if it could reach an engine a coupler defect could be misattributed to engine
behaviour.

Two modules, one per direction:

* `ray_to_scalar` (CHE-185/186/187/188, R07.1-R07.4) -- the coherent wavelet sum
  `RayBundle -> ScalarField`, with the projection convention, the sampling measure
  and the grazing-mode phase floor each stated and refusable rather than defaulted,
  and two numerical realizations of the same sum selected by an argument.
* `scalar_to_ray` (CHE-189/190, R08.1-R08.2) -- the angular-spectrum
  decomposition `ScalarField -> RayBundle`, emitting modes whose amplitude and
  measure the wavelet sum accepts, with three draw rules over the spectral axis
  and the variance each predicts.

**There is no round-trip operation, and there will not be one.** A ray -> wave ->
ray conversion with no physical transformation in between changes no state; it is
a representation-consistency check and its home is `tests/physics/`. Shipping it
would advertise a physical capability that is really a test fixture.

Not here either: `AngularSpectrum` as a public type. It is an intermediate, not a
boundary artifact -- nothing outside `scalar_to_ray` consumes one, and making it
public would add a third representation to a tree whose whole point is that there
are two. Also avoided on the scalar-to-ray side: `SamplingPerturbation`, `PositionPlan`,
`PatchPlan`, `Ensemble`, `PositionalAngularSampler`, `LaunchGeometry`,
`ChunkWorkItem`, `StreamingResult` and `StreamingReconstruction` -- nine classes
for what is a sampling function plus a diagnostics record. **And no chunking
framework**: if a workload needs chunking that is the executor's concern or the
caller's, not the coupler's.

Not here, deliberately: `RayToWaveCoupler` and the 533-LOC node wrapper around it,
`CoherentHandoff`, `DeclaredHandoffPlane`, `Perturbation`, `HandoffPerturbation`,
`StreamingReconstruction`, `StreamingResult`, `PositionalAngularSampler`,
`LaunchGeometry`, `BandLimit`, `ChunkWorkItem`, `CurvatureBudget`, a `Coupler`
protocol with its request/result pair, and `GradientProblem` /
`DifferentiabilityReport`. A function plus a descriptor is sufficient, and the
negative tests that `Perturbation` existed to serve perturb the *bundle* instead,
so they exercise the kernel that ships.
"""

from couplers.ray_to_scalar import (
    DEFAULT_KSPACE_OVERSAMPLE,
    DEFAULT_PHASE_BUDGET_RAD,
    REFUSALS,
    SCALE_NOTE,
    GrazingPolicy,
    Normalization,
    Projection,
    Reconstruction,
    ReconstructionDiagnostics,
    grazing_floor_for_phase_budget,
    grid_nyquist_direction_limit,
    ray_to_scalar,
)
from couplers.scalar_to_ray import (
    DRAW_RULES,
    SAMPLING_DENSITIES,
    DrawRule,
    SamplingDensity,
    SamplingDiagnostics,
    predicted_variance_ratio,
    scalar_to_ray,
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
#: `__all__` here has 20 names of which 2 are operations -- the other 18 are the
#: declaration tables, the diagnostics records, the enums, the sampling helpers and
#: this tuple itself -- and deriving coverage from it would demand a descriptor for
#: `DrawRule`. **The counted number is checked**, by
#: `tests/operations/test_catalog.py::test_the_counts_this_justification_rests_on`,
#: because the justification for the whole `OPERATIONS` mechanism rests on it and
#: adding a name to `__all__` is what makes it drift -- as this very tuple did.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = ("ray_to_scalar", "scalar_to_ray")

__all__ = [
    "DEFAULT_KSPACE_OVERSAMPLE",
    "DEFAULT_PHASE_BUDGET_RAD",
    "DRAW_RULES",
    "OPERATIONS",
    "REFUSALS",
    "SAMPLING_DENSITIES",
    "SCALE_NOTE",
    "DrawRule",
    "GrazingPolicy",
    "Normalization",
    "Projection",
    "Reconstruction",
    "ReconstructionDiagnostics",
    "SamplingDensity",
    "SamplingDiagnostics",
    "grazing_floor_for_phase_budget",
    "grid_nyquist_direction_limit",
    "predicted_variance_ratio",
    "ray_to_scalar",
    "scalar_to_ray",
]
