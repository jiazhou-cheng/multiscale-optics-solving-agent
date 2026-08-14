"""Bidirectional ray-wave couplers and their boundary artifact contracts.

Physics and provenance for these couplers live in ``knowledge/couplers/``; the
frozen execution contract lives in ``benchmarks/M2_COUPLER_PROTOCOL.md``.
"""

from multiscale_optics_agent.couplers.base import (
    DEFAULT_SOURCE_PORT,
    Coupler,
    CouplerRunRequest,
    CouplerRunResult,
)
from multiscale_optics_agent.couplers.contracts import (
    AXIS_ORDER,
    ORIGIN_RULE,
    PHASOR,
    SPATIAL_FACTOR,
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    PSF,
    RayBundle,
    ReferencePlane,
    WavefrontSamples,
)
from multiscale_optics_agent.couplers.ray_to_wave_node import RayToWaveCoupler

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
