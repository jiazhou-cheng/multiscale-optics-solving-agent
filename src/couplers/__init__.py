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

__all__ = [
    # The two runnable graph nodes.
    "PlanarDoeStepCoupler",
    "RayToWaveCoupler",
    # The batched planar step as a library call, for callers outside a graph.
    "CascadeDiagnostics",
    "PrimarySampling",
    "planar_doe_step",
    "AXIS_ORDER",
    "DEFAULT_SOURCE_PORT",
    "ORIGIN_RULE",
    "PHASOR",
    "PSF",
    "SPATIAL_FACTOR",
    "ComplexField",
    "ContractCode",
    "ContractError",
    "Coupler",
    "CouplerRunRequest",
    "CouplerRunResult",
    "Frame",
    "RayBundle",
    "ReferencePlane",
    "WavefrontSamples",
]
