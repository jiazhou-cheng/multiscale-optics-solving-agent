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
from couplers.node import RayToWaveCoupler

__all__ = [
    "RayToWaveCoupler",
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
