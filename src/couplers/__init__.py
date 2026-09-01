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
* `scalar_to_ray` (CHE-189, R08.1) -- the angular-spectrum decomposition
  `ScalarField -> RayBundle`, emitting modes whose amplitude and measure the
  wavelet sum accepts.

**There is no round-trip operation, and there will not be one.** A ray -> wave ->
ray conversion with no physical transformation in between changes no state; it is
a representation-consistency check and its home is `tests/physics/`. Shipping it
would advertise a physical capability that is really a test fixture.

Not here either: `AngularSpectrum` as a public type. It is an intermediate, not a
boundary artifact -- nothing outside `scalar_to_ray` consumes one, and making it
public would add a third representation to a tree whose whole point is that there
are two. Also avoided on the scalar-to-ray side: `SamplingPerturbation`,
`PositionPlan`, `PatchPlan` and `Ensemble`.

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
    SAMPLING_DENSITIES,
    SamplingDensity,
    SamplingDiagnostics,
    scalar_to_ray,
)

__all__ = [
    "DEFAULT_KSPACE_OVERSAMPLE",
    "DEFAULT_PHASE_BUDGET_RAD",
    "REFUSALS",
    "SAMPLING_DENSITIES",
    "SCALE_NOTE",
    "GrazingPolicy",
    "Normalization",
    "Projection",
    "Reconstruction",
    "ReconstructionDiagnostics",
    "SamplingDensity",
    "SamplingDiagnostics",
    "grazing_floor_for_phase_budget",
    "grid_nyquist_direction_limit",
    "ray_to_scalar",
    "scalar_to_ray",
]
