"""Project-specific exceptions."""


class MultiScaleOpticsError(Exception):
    """Base exception for the project."""


class RegistryError(MultiScaleOpticsError):
    """Raised when model or coupler registry data are invalid."""


class GraphCompilationError(MultiScaleOpticsError):
    """Raised when a graph cannot be compiled for execution."""


class AdapterError(MultiScaleOpticsError):
    """Base class for adapter-layer failures.

    Every adapter must raise one of the three subclasses below rather than a
    bare AdapterError, so callers can distinguish an environment problem
    (AdapterDependencyError), a request outside the adapter's declared
    capability (UnsupportedCapabilityError, raised before any solver call),
    and a solver that ran but failed on this input (SolverExecutionError).
    """


class AdapterDependencyError(AdapterError):
    """A required external package, or a sub-dependency of it, cannot be
    imported or is otherwise unusable in the current environment."""


class UnsupportedCapabilityError(AdapterError):
    """The request asks the adapter to do something it deliberately does not
    implement (e.g. a physics regime or gradient path known to be broken).
    Must be raised eagerly, before any solver call that could silently
    return a wrong or partial answer."""


class SolverExecutionError(AdapterError):
    """The underlying solver ran but failed for this specific input
    (non-convergence, NaN/Inf output, internal solver exception)."""


class AdapterNotFoundError(AdapterError):
    """No adapter implementation is registered for a requested model ID."""
