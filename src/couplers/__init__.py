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

One module, landed by CHE-185, CHE-186, CHE-187 and CHE-188 (R07.1-R07.4):

* `ray_to_scalar` -- the coherent wavelet sum `RayBundle -> ScalarField`, with the
  projection convention, the sampling measure and the grazing-mode phase floor each
  stated and refusable rather than defaulted, and two numerical realizations of the
  same sum selected by an argument.

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

__all__ = [
    "DEFAULT_KSPACE_OVERSAMPLE",
    "DEFAULT_PHASE_BUDGET_RAD",
    "REFUSALS",
    "SCALE_NOTE",
    "GrazingPolicy",
    "Normalization",
    "Projection",
    "Reconstruction",
    "ReconstructionDiagnostics",
    "grazing_floor_for_phase_budget",
    "grid_nyquist_direction_limit",
    "ray_to_scalar",
]
