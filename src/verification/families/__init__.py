"""The benchmark family substrate (CHE-131, M0.5.2).

``schema`` defines what a family and an instance are and enforces the rules that
cannot be opted out of. ``predicates`` holds the executable validity bounds that
were already measured before this package existed. ``registry`` is the one place
that knows which families exist.
"""

# Importing the family modules is what registers them. Explicit rather than
# discovered, for the reason core/capabilities.py gives about its own table: a
# source of truth that is discovered agrees with itself by construction,
# including about which entries exist at all.
from verification.families import (  # noqa: F401
    b1_ray,
    b1_wave,
    b3_composed,
    b4_characterization,
)
from verification.families.registry import (
    FAMILIES,
    families_for_category,
    family,
    family_ids,
    register,
)
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkFamily,
    BenchmarkInstance,
    ExecutionParameter,
    ExecutionPolicy,
    FamilyOracle,
    InstanceOrigin,
    Invariant,
    Metric,
    NegativeControl,
    NegativeControlExpectation,
    NumericalParameter,
    Parameter,
    ParameterKind,
    PhysicalParameter,
    ProvenanceRule,
    RepresentationParameter,
    SamplerAbsentReason,
    StochasticEvidenceKind,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
    ValidityState,
    fingerprint_of,
)

__all__ = [
    "FAMILIES",
    "BenchmarkCategory",
    "BenchmarkFamily",
    "BenchmarkInstance",
    "ExecutionParameter",
    "ExecutionPolicy",
    "FamilyOracle",
    "InstanceOrigin",
    "Invariant",
    "Metric",
    "NegativeControl",
    "NegativeControlExpectation",
    "NumericalParameter",
    "Parameter",
    "ParameterKind",
    "PhysicalParameter",
    "ProvenanceRule",
    "RepresentationParameter",
    "SamplerAbsentReason",
    "StochasticEvidenceKind",
    "StochasticPolicy",
    "Tolerance",
    "ToleranceBasis",
    "ValidityBasis",
    "ValidityPredicate",
    "ValidityState",
    "families_for_category",
    "family",
    "family_ids",
    "fingerprint_of",
    "register",
]
