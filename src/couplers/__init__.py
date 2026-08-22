"""Bidirectional ray-wave couplers and their boundary artifact contracts.

Physics and provenance for these couplers live in ``knowledge/couplers/``; the
frozen execution contract lives in ``benchmarks/protocols/m2_coupler_protocol.md``.
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
from couplers.base import (
    DEFAULT_SOURCE_PORT,
    Coupler,
    CouplerRunRequest,
    CouplerRunResult,
)
from couplers.cascade import CascadeDiagnostics, PrimarySampling, planar_doe_step
from couplers.doe_node import PlanarDoeStepCoupler
from couplers.node import RayToWaveCoupler

# `PlanarDoeStepCoupler` and `RayToWaveCoupler` are the two runnable graph nodes;
# `planar_doe_step` and its two companions are the batched step as a library
# call, for a caller outside a graph. Sorted rather than grouped by role, because
# a hand-grouped list drifts and ruff will not let it.
__all__ = [
    "AXIS_ORDER",
    "DEFAULT_SOURCE_PORT",
    "ORIGIN_RULE",
    "PHASOR",
    "PSF",
    "SPATIAL_FACTOR",
    "CascadeDiagnostics",
    "ComplexField",
    "ContractCode",
    "ContractError",
    "Coupler",
    "CouplerRunRequest",
    "CouplerRunResult",
    "Frame",
    "PlanarDoeStepCoupler",
    "PrimarySampling",
    "RayBundle",
    "RayToWaveCoupler",
    "ReferencePlane",
    "WavefrontSamples",
    "planar_doe_step",
]
