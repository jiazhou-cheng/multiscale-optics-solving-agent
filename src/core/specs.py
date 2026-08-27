"""Pydantic schemas for the model-coupler intermediate representation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects undeclared serialized fields."""

    model_config = ConfigDict(extra="forbid")


class Framework(StrEnum):
    JAX = "jax"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    INTERNAL = "internal"
    EXTERNAL = "external"


class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    EXTERNAL = "external"


class ApproximationClass(StrEnum):
    GEOMETRIC_OPTICS = "geometric_optics"
    EIKONAL = "eikonal"
    SCALAR_WAVE = "scalar_wave"
    VECTOR_WAVE = "vector_wave"
    TMM = "tmm"
    RCWA = "rcwa"
    FULL_WAVE_EM = "full_wave_em"
    EIGENMODE = "eigenmode"
    PHOTONIC_CIRCUIT = "photonic_circuit"
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    SENSOR = "sensor"
    TRANSFORMATION = "transformation"


class ArtifactKind(StrEnum):
    OPTICAL_SYSTEM = "optical_system"
    RAY_BUNDLE = "ray_bundle"
    WAVEFRONT_SAMPLES = "wavefront_samples"
    TRAVEL_TIME_FIELD = "travel_time_field"
    COMPLEX_FIELD = "complex_field"
    VECTOR_FIELD = "vector_field"
    PHASE_PROFILE = "phase_profile"
    PERIODIC_STRUCTURE = "periodic_structure"
    DIFFRACTION_CHANNELS = "diffraction_channels"
    EFFECTIVE_SURFACE = "effective_surface"
    MATERIAL_FIELD = "material_field"
    NEAR_FIELD_SURFACE = "near_field_surface"
    FAR_FIELD_ANGULAR_SPECTRUM = "far_field_angular_spectrum"
    MODE_BASIS = "mode_basis"
    MODAL_AMPLITUDES = "modal_amplitudes"
    S_PARAMETERS = "s_parameters"
    CIRCUIT_COMPONENT = "circuit_component"
    CIRCUIT_RESPONSE = "circuit_response"
    PSF = "psf"
    OTF = "otf"
    SENSOR_IMAGE = "sensor_image"
    ABSORBED_POWER_DENSITY = "absorbed_power_density"
    HEAT_SOURCE = "heat_source"
    TEMPERATURE_FIELD = "temperature_field"
    DISPLACEMENT_FIELD = "displacement_field"
    GEOMETRY_UPDATE = "geometry_update"


class DerivativeMode(StrEnum):
    NATIVE_AUTODIFF = "native_autodiff"
    CUSTOM_VJP = "custom_vjp"
    CUSTOM_JVP = "custom_jvp"
    ADJOINT = "adjoint"
    IMPLICIT = "implicit"
    ANALYTIC = "analytic"
    SURROGATE = "surrogate"
    FINITE_DIFFERENCE = "finite_difference"
    NONE = "none"


class Maturity(StrEnum):
    EXPERIMENTAL = "experimental"
    CHARACTERIZED = "characterized"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


class PortSpec(StrictModel):
    """A physical input/output port, including semantic metadata."""

    name: str = Field(min_length=1)
    artifact: ArtifactKind
    units: str | None = None
    requires_metadata: list[str] = Field(default_factory=list)
    provides_metadata: list[str] = Field(default_factory=list)
    optional: bool = False
    description: str = ""


class DerivativeSpec(StrictModel):
    mode: DerivativeMode
    verified: bool = False
    parameters: list[str] = Field(default_factory=list)
    notes: str = ""


class ValiditySpec(StrictModel):
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    hard_limits: dict[str, str] = Field(default_factory=dict)


class CostModelSpec(StrictModel):
    scaling: str = "unknown"
    memory_scaling: str = "unknown"
    estimator: str | None = None
    notes: str = ""


class SourceSpec(StrictModel):
    package: str
    docs_url: str
    repository_url: str | None = None
    pinned_version: str | None = None
    pinned_commit: str | None = None
    retrieved_at: str | None = None


class ModelSpec(StrictModel):
    id: str = Field(pattern=r"^M_[A-Z0-9_]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str
    framework: Framework
    approximation: ApproximationClass
    inputs: list[PortSpec] = Field(default_factory=list)
    outputs: list[PortSpec] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    derivative: DerivativeSpec
    validity: ValiditySpec = Field(default_factory=ValiditySpec)
    cost_model: CostModelSpec = Field(default_factory=CostModelSpec)
    source: SourceSpec | None = None
    maturity: Maturity = Maturity.EXPERIMENTAL
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ports(self) -> ModelSpec:
        for direction, ports in (("input", self.inputs), ("output", self.outputs)):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {direction} port in model {self.id}")
        return self

    def input_port(self, name: str) -> PortSpec | None:
        return next((port for port in self.inputs if port.name == name), None)

    def output_port(self, name: str) -> PortSpec | None:
        return next((port for port in self.outputs if port.name == name), None)


class CouplerRole(StrEnum):
    """Which of the three kinds of operation a registered coupler is (CHE-142).

    ``src/couplers/`` was using one word for three different things, so the
    registry could not say which of them a caller was selecting. The rule:
    **representation transition != diffractive physical interaction !=
    propagation.** ``src/couplers/ontology.py`` holds the package-side partition
    and the test that keeps the two in step.

    There is no default. A coupler row must say which it is, because the three
    answer different questions and a guess would be wrong silently: a caller
    reading ``C_PLANAR_DOE_STEP``'s ray-in/ray-out ports cannot tell from them
    whether the step re-described the light or diffracted it.
    """

    #: Changes what the light is described by. Rays <-> field.
    REPRESENTATION_TRANSITION = "representation_transition"
    #: Physics at a surface: coherent rays in, coherent rays out. Contains
    #: representation transitions; is not one.
    DIFFRACTIVE_INTERACTION = "diffractive_interaction"
    #: Moves a representation between planes and changes neither the
    #: representation nor the physical content.
    PROPAGATION = "propagation"


class InteractionSpec(StrictModel):
    """The shared physical identity a group of coupler rows computes (CHE-142).

    Two rows carrying the same ``id`` are the **same physical operation at
    different granularities**, not two peer claims -- which is what
    ``C_PLANAR_DOE_STEP`` and ``C_PATCH_WFT`` had been reading as. Each keeps its
    own capability row regardless: devices, dtypes and maturity genuinely differ
    between the models, and grouping them must not widen the narrower one.
    """

    #: The interaction's identity, shared across its models. Not a coupler id: an
    #: interaction is not separately runnable and has no capability of its own.
    id: str = Field(pattern=r"^I_[A-Z0-9_]+$")
    #: Which granularity this row implements.
    model: str = Field(min_length=1)
    #: How this model relates to the interaction's others, with its source. A
    #: group whose members do not say which is the shortcut for which is back to
    #: reading as peers.
    relation: str = Field(min_length=1)


class CouplerSpec(StrictModel):
    id: str = Field(pattern=r"^C_[A-Z0-9_]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str
    framework: Framework
    approximation: ApproximationClass = ApproximationClass.TRANSFORMATION
    #: Required, and deliberately not defaulted -- see :class:`CouplerRole`.
    role: CouplerRole
    #: Present only on a row that is one model of a shared interaction.
    interaction: InteractionSpec | None = None
    source: PortSpec
    target: PortSpec
    devices: list[Device] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    derivative: DerivativeSpec
    validity: ValiditySpec = Field(default_factory=ValiditySpec)
    cost_model: CostModelSpec = Field(default_factory=CostModelSpec)
    lossy: bool = False
    invariants: list[str] = Field(default_factory=list)
    source_info: SourceSpec | None = None
    maturity: Maturity = Maturity.EXPERIMENTAL
    tags: list[str] = Field(default_factory=list)


class NodeSpec(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    model: str = Field(pattern=r"^M_[A-Z0-9_]+$")
    config: dict[str, Any] = Field(default_factory=dict)


class PortRef(StrictModel):
    node: str
    port: str


class EdgeSpec(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    coupler: str = Field(pattern=r"^C_[A-Z0-9_]+$")
    source: PortRef
    target: PortRef
    config: dict[str, Any] = Field(default_factory=dict)


class DesignVariableSpec(StrictModel):
    name: str = Field(min_length=1)
    node: str
    parameter: str
    bounds: tuple[float, float] | None = None
    requires_gradient: bool = True

    @model_validator(mode="after")
    def ordered_bounds(self) -> DesignVariableSpec:
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"invalid bounds for {self.name}: {self.bounds}")
        return self


class ObjectiveSpec(StrictModel):
    name: str = Field(min_length=1)
    node: str
    port: str
    metric: str
    requires_gradient: bool = True


class VerificationSpec(StrictModel):
    id: str
    kind: str
    target: str
    tolerance: float | None = None
    required: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class GraphSpec(StrictModel):
    schema_version: str = "0.1.0"
    task_id: str | None = None
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = Field(default_factory=list)
    design_variables: list[DesignVariableSpec] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    verification: list[VerificationSpec] = Field(default_factory=list)
    require_verified_gradients: bool = False
    allow_cycles: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_graph_ids(self) -> GraphSpec:
        for label, ids in (
            ("node", [node.id for node in self.nodes]),
            ("edge", [edge.id for edge in self.edges]),
            ("design variable", [item.name for item in self.design_variables]),
            ("objective", [item.name for item in self.objectives]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} identifier")
        return self
