"""Typed physics graphs for multi-scale optical simulation."""

from multiscale_optics_agent.core.graph import GraphValidator, ValidationReport
from multiscale_optics_agent.core.specs import CouplerSpec, GraphSpec, ModelSpec
from multiscale_optics_agent.registry.loader import Registry

__all__ = [
    "CouplerSpec",
    "GraphSpec",
    "GraphValidator",
    "ModelSpec",
    "Registry",
    "ValidationReport",
]

__version__ = "0.1.0"
