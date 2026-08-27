"""Ray-wave couplers: representation transitions, diffractive interactions, propagation.

Three kinds of operation, and the package now says which is which (CHE-142).
The rule:

    **representation transition != diffractive physical interaction !=
    propagation.**

* :mod:`couplers.ray_to_wave` and :mod:`couplers.wave_to_ray` change what the
  light is *described by*. ``C_RAY_TO_WAVE``, ``C_WAVE_TO_RAY``.
* :mod:`couplers.interaction` is the diffractive interaction: incident coherent
  rays meet a diffractive surface, coherent rays come out. One operation with the
  granularity named explicitly -- ``FULL_FIELD`` (``couplers.cascade``),
  ``LOCAL_PATCH`` (``couplers.patch``), ``GENERALIZED_SNELL`` (CHE-143, not
  implemented). ``FULL_FIELD`` is the *shortcut* for ``LOCAL_PATCH`` and not its
  peer; SI S10 says so and :mod:`couplers.interaction` carries the statement.
* :mod:`couplers.propagation` moves a bundle between planes and changes neither
  the representation nor the physical content.

:mod:`couplers.ontology` is that partition as data, and
``tests/test_diffractive_interaction.py`` enforces it rather than trusting this
docstring.

Physics and provenance live in ``knowledge/couplers/``; the frozen execution
contract lives in ``benchmarks/protocols/m2_coupler_protocol.md``.
"""

from core.boundary import (
    AXIS_ORDER,
    ORIGIN_RULE,
    PHASOR,
    PSF,
    SPATIAL_FACTOR,
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
    WavefrontSamples,
)
from core.specs import CouplerRole
from couplers.base import (
    DEFAULT_SOURCE_PORT,
    Coupler,
    CouplerRunRequest,
    CouplerRunResult,
)
from couplers.cascade import CascadeDiagnostics, PrimarySampling, planar_doe_step
from couplers.doe_node import PlanarDoeStepCoupler
from couplers.generalized_snell import GeneralizedSnellDiagnostics, generalized_snell_step
from couplers.interaction import (
    INTERACTION_ID,
    CoverageBasis,
    DiffractiveInteractionResult,
    DiffractiveModel,
    DiffractiveSurface,
    FullFieldParameters,
    GeneralizedSnellParameters,
    LocalPatchParameters,
    PatchWindow,
    SamplingDensity,
    Substrate,
    diffractive_interaction,
)
from couplers.node import RayToWaveCoupler
from couplers.ontology import COUPLER_ROLES, OPERATION_ROLES
from couplers.patch_node import PatchWftCoupler
from couplers.propagation import advance_bundle_to_plane

# `diffractive_interaction` is the entry point a caller should reach for: one
# operation, model named, surface declared.
#
# `planar_doe_step` and the graph nodes remain exported for COMPATIBILITY, and
# here is the horizon rather than an open-ended "for now": they are what every
# shipped call site and every committed record use, so they stay until the M2
# system ladder (CHE-144 through CHE-150) has migrated its call sites to the
# entry point, and are then removed from this module's `__all__`. They are the
# models' implementations, not the long-term public ontology; a new caller that
# reaches for `planar_doe_step` is naming a granularity by its function name
# instead of declaring it, which is what CHE-142 removed the need for.
#
# `PatchWftCoupler`, `PlanarDoeStepCoupler` and `RayToWaveCoupler` are the three
# runnable graph nodes. Sorted rather than grouped by role, because a
# hand-grouped list drifts and ruff will not let it.
__all__ = [
    "AXIS_ORDER",
    "COUPLER_ROLES",
    "DEFAULT_SOURCE_PORT",
    "INTERACTION_ID",
    "OPERATION_ROLES",
    "ORIGIN_RULE",
    "PHASOR",
    "PSF",
    "SPATIAL_FACTOR",
    "CascadeDiagnostics",
    "ComplexField",
    "ContractCode",
    "ContractError",
    "Coupler",
    "CouplerRole",
    "CouplerRunRequest",
    "CouplerRunResult",
    "CoverageBasis",
    "DiffractiveInteractionResult",
    "DiffractiveModel",
    "DiffractiveSurface",
    "Frame",
    "FullFieldParameters",
    "GeneralizedSnellDiagnostics",
    "GeneralizedSnellParameters",
    "LocalPatchParameters",
    "PatchWftCoupler",
    "PatchWindow",
    "PlanarDoeStepCoupler",
    "PrimarySampling",
    "RayBundle",
    "RayToWaveCoupler",
    "ReferencePlane",
    "SamplingDensity",
    "Substrate",
    "WavefrontSamples",
    "advance_bundle_to_plane",
    "diffractive_interaction",
    "generalized_snell_step",
    "planar_doe_step",
]
