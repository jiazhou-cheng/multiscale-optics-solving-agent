"""The diffractive interaction, and the three models that compute it (CHE-142).

The architectural rule this module installs, because ``src/couplers/`` had been
using one word for three different kinds of thing:

    **representation transition != diffractive physical interaction !=
    propagation.**

* A **representation transition** changes what the light is *described by*.
  ``couplers/ray_to_wave.py`` (``C_RAY_TO_WAVE``) and
  ``couplers/wave_to_ray.py`` (``C_WAVE_TO_RAY``) are the two, and they are
  inverses of each other in the limit of complete sampling.
* A **diffractive interaction** is *physics at a surface*. One operation::

      incident coherent rays + diffractive surface -> outgoing coherent rays

  It happens to contain two representation transitions and a transmission, but
  that is its *implementation*, not its identity: a caller asking for it is
  asking what the surface does to the light, not for a change of description.
* **Propagation** moves an existing representation between planes and changes
  neither. ``couplers/propagation.py``.

``couplers/ontology.py`` states that partition as executable data, and
``tests/test_diffractive_interaction.py`` enforces it. The reason it needs
enforcing rather than documenting: ``C_PLANAR_DOE_STEP`` and ``C_PATCH_WFT``
read as two unrelated physics claims in the registry, and they are not two
claims at all -- they are one interaction at two granularities.

The models, and the relation between them
-----------------------------------------
:class:`DiffractiveModel` names the granularity, and it is **never inferred**.

``FULL_FIELD``
    Global angular-spectrum treatment: accumulate every incident ray coherently
    onto the one common plane, apply the complex transmission once, transform
    once, resample a fixed budget. SI Algorithm S1, implemented in
    ``couplers/cascade.py``. For a planar DOE or an SLM, where a single global
    field is affordable.
``LOCAL_PATCH``
    Local tangent-plane windowed-Fourier treatment: each patch of the surface is
    windowed out, transformed on its own, and the patches are summed coherently.
    SI eq S1 and S3-S5, implemented in ``couplers/patch.py``. For large or
    conformal surfaces, and for any surface whose global field does not fit in
    memory -- SI Table S2 records the 4032x4032 Grating-Lens DOE as OOM on a
    48 GB A6000 on the global route and complete in 4.982 s at 11.492 GB on the
    patch route.
``GENERALIZED_SNELL``
    The reduced-order model -- CHE-143 (M2.7). No field is formed at all: each
    ray is redirected by a local grating equation, evaluated at its own
    transverse position, from the same declared surface. Planar substrate only;
    a conformal one needs a per-ray local tangent frame this model does not
    accept and is refused. Implemented in ``couplers/generalized_snell.py``.

**They are not peers, and SI S10 says which is which.** Of the global
aggregation Algorithm S1 performs: *"For conformal DOEs, this global aggregation
before ray-DOE interaction is not applicable because rays intersect different
local tangent planes with position-dependent coordinate frames and surface
normals. We therefore retain the direct implementation."* The direct
implementation is ``LOCAL_PATCH``; ``FULL_FIELD`` is the **shortcut** available
when one common plane exists. Concretely, ``FULL_FIELD`` is ``LOCAL_PATCH`` at
one full-aperture patch, and ``tests/test_patch_wft.py`` measures that identity
at 1.4e-12 relative field error rather than asserting it.

The regimes therefore **overlap** rather than partition. On a planar substrate
both models are valid and the choice is a cost/variance one; on a conformal
substrate only ``LOCAL_PATCH`` is even applicable, and ``FULL_FIELD`` is refused
with :attr:`~core.boundary.ContractCode.MODEL_NOT_APPLICABLE` rather than
silently falling back -- a fallback would accumulate rays from different tangent
frames onto one plane and produce a plausible field that is wrong.

What this module does *not* do
------------------------------
It adds no physics and changes no number. Each model's parameters are handed
straight to the function that already implemented it, so a result obtained here
is bitwise the result obtained by calling that function directly, and
``tests/test_diffractive_interaction.py`` asserts exactly that for one
``FULL_FIELD`` and one ``LOCAL_PATCH`` case. The value it adds is that the model
is named in the API, the surface is one argument instead of four loose ones, and
a wrong pairing of the two is refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from core.boundary import (
    ComplexField,
    ContractCode,
    ContractError,
    RayBundle,
    ReferencePlane,
)
from couplers.cascade import CascadeDiagnostics, PrimarySampling, planar_doe_step
from couplers.generalized_snell import GeneralizedSnellDiagnostics, generalized_snell_step
from couplers.patch import (
    CoverageBasis,
    PatchDiagnostics,
    Substrate,
    patch_secondary_rays,
    plan_patches,
)
from couplers.wave_to_ray import SamplingDensity

# `CoverageBasis` and `Substrate` are re-exported rather than left in
# `couplers.patch`. They are vocabulary of the SURFACE and of the interaction's
# parameters, not of one model's implementation: a caller declaring a conformal
# substrate should not have to import it from the LOCAL_PATCH module, and before
# this the substrate was declared to the patch planner while `FULL_FIELD` never
# saw it at all. `couplers.patch` keeps defining them, so they are the same
# objects and every existing import still works.
__all__ = [
    "INTERACTION_ID",
    "MODEL_COUPLER_IDS",
    "CoverageBasis",
    "DiffractiveInteractionResult",
    "DiffractiveModel",
    "DiffractiveSurface",
    "FullFieldParameters",
    "GeneralizedSnellDiagnostics",
    "GeneralizedSnellParameters",
    "LocalPatchParameters",
    "PatchWindow",
    "SamplingDensity",
    "Substrate",
    "diffractive_interaction",
]

#: The shared identity the models declare in the registry. One interaction, three
#: models -- which is what stops ``C_PLANAR_DOE_STEP`` and ``C_PATCH_WFT`` from
#: reading as two unrelated physics claims. It is deliberately NOT a coupler id:
#: it is not separately runnable and has no capability row of its own, because
#: devices, dtypes and maturity genuinely differ per model.
INTERACTION_ID = "I_DIFFRACTIVE"


class DiffractiveModel(StrEnum):
    """Which granularity computes the interaction. Never inferred; see module doc."""

    #: Global angular-spectrum treatment. ``couplers/cascade.py``.
    FULL_FIELD = "full_field"
    #: Local tangent-plane windowed-Fourier treatment. ``couplers/patch.py``.
    LOCAL_PATCH = "local_patch"
    #: Reduced-order model, ``couplers/generalized_snell.py`` -- CHE-143 (M2.7).
    GENERALIZED_SNELL = "generalized_snell"


#: Which registry coupler row each model's capability lives in. ``None`` means the
#: model has no row yet because it has no implementation, which is a different
#: statement from having a row that claims nothing.
MODEL_COUPLER_IDS: dict[DiffractiveModel, str | None] = {
    DiffractiveModel.FULL_FIELD: "C_PLANAR_DOE_STEP",
    DiffractiveModel.LOCAL_PATCH: "C_PATCH_WFT",
    DiffractiveModel.GENERALIZED_SNELL: "C_GENERALIZED_SNELL",
}


class PatchWindow(StrEnum):
    """The patch window, declared because it is a model parameter with one value.

    Exactly one member, and that is the honest count rather than a placeholder.
    Any taper below 1 removes field that no other patch replaces, so the
    coherent patch sum stops converging to the full-surface response -- the
    partition-of-unity argument behind the SI S2 convergence relation is exactly
    what a taper breaks. A window is offered as a *declaration* so a caller can
    see that the rectangular choice was made, and a future taper would arrive as
    a new member with its own evidence rather than as a silent default change.
    """

    #: The patch indicator: 1 inside, 0 outside, no taper.
    RECTANGULAR = "rectangular"


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffractiveSurface:
    """The diffractive surface as one declared argument.

    Before this, "the diffractive surface" was four loose arguments repeated at
    every call site -- a transmission array, a grid shape, a sample pitch and a
    plane -- with the substrate and its radius passed separately to a *different*
    function on the patch route and not at all on the global one. A caller could
    therefore describe a curved surface to the planner and a planar one to the
    emitter without anything noticing.

    Conventions, stated because none of them is visible in an intensity:

    * ``transmission`` is a **complex amplitude transmission**, not an intensity
      and not a phase. A real array is refused: it is an amplitude mask with an
      undeclared phase.
    * the grid is ``(ny, nx)`` and coordinate zero sits at index ``n // 2``,
      matching :meth:`ComplexField.coordinates`. It is read off the array rather
      than declared separately, which removes the shape disagreement entirely.
    * ``plane`` is where the transmission lives. Neither model propagates the
      incident bundle to it -- the bundle must already be expressed there.
    * ``substrate`` and ``radius_m`` are the curvature declaration, and they must
      agree: planar means ``radius_m = inf``, and a finite radius on a declared
      planar substrate is two declarations contradicting each other rather than a
      harmless extra.
    """

    #: ``(ny, nx)`` complex amplitude transmission.
    transmission: np.ndarray[Any, Any]
    #: ``(dy, dx)`` sample pitch in metres.
    sample_pitch_m: tuple[float, float]
    #: Where the transmission lives.
    plane: ReferencePlane
    #: What the surface sits on. ``CONFORMAL`` is refused by both implemented
    #: models today, for different reasons -- see :func:`diffractive_interaction`.
    substrate: Substrate = Substrate.PLANAR
    #: Substrate radius of curvature in metres. ``inf`` for planar.
    radius_m: float = math.inf
    #: Refractive index on the incident side. ``FULL_FIELD`` and ``LOCAL_PATCH``
    #: only execute at 1.0: the transmission multiply and the angular-spectrum
    #: step both use the vacuum wavelength the incident bundle carries, so a
    #: declared index != 1 would need the in-medium wavelength and an interface
    #: factor neither has. ``GENERALIZED_SNELL`` is the one model that uses a
    #: declared index, in its own tangential-momentum-matching equation -- see
    #: :func:`diffractive_interaction`, which enforces the ``== 1.0`` restriction
    #: per model rather than here, since here the surface does not yet know
    #: which model will read it.
    n_incident: float = 1.0
    #: Refractive index on the transmitted side. Same rule as ``n_incident``.
    n_transmitted: float = 1.0

    def __post_init__(self) -> None:
        transmission = np.asarray(self.transmission)
        if transmission.ndim != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"the transmission must be a 2-D (ny, nx) grid, got shape "
                f"{transmission.shape}",
                declaration="transmission",
            )
        if not np.iscomplexobj(transmission):
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "the transmission must be complex; a real array is an amplitude "
                "mask with an undeclared phase, not a transmission",
                declaration="transmission",
                remedy=(
                    "Use DiffractiveSurface.from_phase() for a phase-only surface, "
                    "or supply exp(i phi) yourself."
                ),
            )
        pitch = tuple(float(p) for p in self.sample_pitch_m)
        if len(pitch) != 2 or not all(math.isfinite(p) and p > 0.0 for p in pitch):
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                f"sample_pitch_m must be two finite positive metres, got "
                f"{self.sample_pitch_m!r}",
                declaration="sample_pitch_m",
            )
        if self.substrate is Substrate.PLANAR and not math.isinf(float(self.radius_m)):
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                f"substrate={self.substrate.value!r} declares a flat surface but "
                f"radius_m={self.radius_m!r} declares a curved one",
                declaration="radius_m",
                remedy=(
                    "Leave radius_m at inf for a planar substrate, or declare "
                    "substrate='conformal' if the surface really is curved. The two "
                    "select different validity arguments -- a hard gate against a "
                    "bound -- so they must not disagree."
                ),
            )
        if self.substrate is Substrate.CONFORMAL and not (
            math.isfinite(float(self.radius_m)) and float(self.radius_m) > 0.0
        ):
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                f"substrate={self.substrate.value!r} needs a finite positive "
                f"radius_m, got {self.radius_m!r}: the only statement available on a "
                "curved substrate is the bound eps_curv <= arcsin(D / 2R), and it "
                "cannot be evaluated without R",
                declaration="radius_m",
            )
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "sample_pitch_m", pitch)
        object.__setattr__(self, "radius_m", float(self.radius_m))

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(ny, nx)``, read off the transmission rather than declared beside it."""
        shape = np.asarray(self.transmission).shape
        return (int(shape[0]), int(shape[1]))

    @classmethod
    def from_phase(
        cls,
        phase_rad: np.ndarray[Any, Any],
        *,
        sample_pitch_m: tuple[float, float],
        plane: ReferencePlane,
        substrate: Substrate = Substrate.PLANAR,
        radius_m: float = math.inf,
    ) -> DiffractiveSurface:
        """A lossless phase-only surface, ``t = exp(+i phi)``.

        The sign is the repository phasor convention (``core.boundary.PHASOR``),
        applied in one place rather than at each call site: a caller writing
        ``exp(-i phi)`` gets a conjugated surface, which is a real DOE that
        focuses on the wrong side and looks entirely plausible.
        """
        phase = np.asarray(phase_rad, dtype=np.float64)
        return cls(
            transmission=np.exp(1j * phase),
            sample_pitch_m=sample_pitch_m,
            plane=plane,
            substrate=substrate,
            radius_m=radius_m,
        )


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullFieldParameters:
    """Model parameters for ``FULL_FIELD``. Every default is what the function had.

    These are the knobs of :func:`couplers.cascade.planar_doe_step`, gathered so
    the model's configuration is one object rather than eight keyword arguments
    interleaved with the surface's.
    """

    model = DiffractiveModel.FULL_FIELD

    #: ``(P, 2)`` launch positions in metres, or ``None`` to have them sampled.
    #: Exactly one of this and ``primary_sampling``.
    launch_positions_xy_m: np.ndarray[Any, Any] | None = None
    #: How to draw the ``P`` launch positions, if they are not supplied.
    primary_sampling: PrimarySampling | None = None
    #: ``P``, required with ``primary_sampling``.
    primary_count: int | None = None
    #: ``S``. ``None`` enumerates every propagating bin -- the deterministic
    #: limit with no sampling error, and the gate the coupler protocol makes
    #: mandatory and first.
    secondary_count: int | None = None
    #: Which density the secondary wavevectors are drawn from.
    density_kind: SamplingDensity = SamplingDensity.UNIFORM
    #: Renormalize the transmitted field to the incident power. Off, and it
    #: should stay off: a lossy surface legitimately loses power.
    preserve_energy: bool = False
    #: Zero padding per side before the transform, in samples.
    pad_width: int = 0


@dataclass(frozen=True)
class LocalPatchParameters:
    """Model parameters for ``LOCAL_PATCH``, including the ones that were implicit.

    The patch route's configuration was split across
    :func:`couplers.patch.plan_patches` and
    :func:`couplers.patch.patch_secondary_rays`, and two of its scientific
    choices -- the window and the spectral density -- were not parameters at all.
    They are fields here so that a record shows what ran, and both refuse any
    value other than the one that executes rather than pretending to be dials.
    """

    model = DiffractiveModel.LOCAL_PATCH

    #: Patch side in samples. Odd: an even patch has no centre sample, so
    #: "centred on a ray" is undefined for it. No default -- patch size is the
    #: memory/accuracy dial this model exists to expose.
    patch_px: int
    #: Preference for the padded transform size, as a multiple of ``patch_px``.
    #: The realized ``pad_px`` is DERIVED and reported; see
    #: :func:`couplers.patch.resolve_pad_px`.
    pad_factor: int = 2
    #: Draw this many patch centres here. Exclusive with ``centers_xy_m``; both
    #: ``None`` selects the single full-aperture patch, which is the exactness
    #: anchor and the configuration in which ``FULL_FIELD`` is its special case.
    patch_count: int | None = None
    #: ``(P, 2)`` caller-supplied centres in metres -- the paper's actual
    #: configuration, where each incident ray defines a patch.
    centers_xy_m: np.ndarray[Any, Any] | None = None
    #: How supplied centres were drawn. Required with ``centers_xy_m``: the
    #: ``A_draw / A_patch`` correction is unbiased only for a known density, and
    #: the density is not recoverable from the positions.
    coverage_basis: CoverageBasis = CoverageBasis.UNKNOWN
    #: ``(P,)`` importance weights carrying ``lambda_c = P / (D pi_c)`` for a
    #: non-uniform centre density, or ``None`` for the uniform draw.
    center_weights: np.ndarray[Any, Any] | None = None
    #: Secondary rays per patch. ``None`` enumerates every propagating mode.
    secondary_count: int | None = None
    #: The curvature-error budget, in radians, that the patch size is checked
    #: against before anything is transformed.
    error_threshold_rad: float = 1e-3
    #: The patch window. One value executes; see :class:`PatchWindow`.
    window: PatchWindow = PatchWindow.RECTANGULAR
    #: The density the per-patch secondary wavevectors are drawn from. The
    #: emitter draws proportional to ``|U~|`` when sampling and uniformly when
    #: enumerating, and that is not currently switchable -- declared so a record
    #: says which estimator produced it, and refused if set to anything else.
    spectral_density: SamplingDensity = SamplingDensity.MAGNITUDE


@dataclass(frozen=True)
class GeneralizedSnellParameters:
    """Model parameters for ``GENERALIZED_SNELL``.

    ``patch_px`` is the one declared transverse scale, reused for two
    different purposes rather than doubled into two dials: it sizes the
    window ``single_order_dominance`` transforms (the same role it plays for
    ``LOCAL_PATCH``), and ``patch_px * sqrt(dy * dx)`` is the transverse scale
    the local-gradient-smoothness predicate checks the local phase curvature
    against. No caller-facing scale is inferred; this is the one that is
    declared.
    """

    model = DiffractiveModel.GENERALIZED_SNELL

    #: The diffraction order ``m``. Declared with a default of the physically
    #: usual value, not inferred: a caller who wants a different order says so.
    order: int = 1
    #: Patch side in samples, for the single-order-dominance measurement and
    #: the smoothness predicate's transverse scale. Odd, same rule and reason
    #: as :attr:`LocalPatchParameters.patch_px`.
    patch_px: int = 5

    def __post_init__(self) -> None:
        if int(self.patch_px) % 2 == 0:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"patch_px={self.patch_px} is even, so it has no centre sample "
                "to declare a transverse scale or a window from",
                declaration="patch_px",
                remedy=f"Use {int(self.patch_px) + 1} (or {int(self.patch_px) - 1}).",
            )


ModelParameters = FullFieldParameters | LocalPatchParameters | GeneralizedSnellParameters


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffractiveInteractionResult:
    """Outgoing rays, plus which model produced them and how.

    ``transmitted_field`` is present only for ``FULL_FIELD``, and is ``None``
    rather than a reconstruction for ``LOCAL_PATCH``. That asymmetry is real: the
    global route computes one transmitted field as an intermediate and returning
    it costs nothing, while the patch route never forms a global field at all --
    that is the whole reason it survives a surface the global route cannot hold.
    Synthesizing one here would be a second, unvalidated reconstruction dressed
    as an output of the step.
    """

    outgoing: RayBundle
    model: DiffractiveModel
    diagnostics: dict[str, Any] = field(default_factory=dict)
    transmitted_field: ComplexField | None = None
    #: The per-model diagnostics object, unwrapped, for a caller that wants the
    #: typed form rather than the dict.
    model_diagnostics: CascadeDiagnostics | PatchDiagnostics | GeneralizedSnellDiagnostics | None = None

    @property
    def interaction_id(self) -> str:
        return INTERACTION_ID


def _require_unit_indices(surface: DiffractiveSurface, *, model: str) -> None:
    """``FULL_FIELD`` and ``LOCAL_PATCH`` compute with the vacuum wavelength the
    incident bundle carries, so a declared index != 1 would need the in-medium
    wavelength and an interface factor that neither has.

    Checked here rather than in ``DiffractiveSurface.__post_init__``: the
    surface does not know which model will read it, and ``GENERALIZED_SNELL``
    uses a declared index in its own tangential-momentum-matching equation, so
    the restriction is a property of the other two models, not of the surface.
    """
    for name, value in (("n_incident", surface.n_incident), ("n_transmitted", surface.n_transmitted)):
        if float(value) != 1.0:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                f"{name}={value!r} is declared but not implemented for model={model!r}. "
                "It computes the transmission multiply and the angular spectrum with "
                "the vacuum wavelength the incident bundle carries; an index != 1 "
                "needs the in-medium wavelength and an interface factor that this "
                "model does not have",
                declaration=name,
                remedy=(
                    "Leave both indices at 1.0, or use model='generalized_snell', "
                    "which is the one model that uses a declared index. Refused rather "
                    "than ignored: a silently dropped index is a wavelength error of "
                    "n, which is a plausible-looking defocus rather than a visible "
                    "failure."
                ),
            )


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def diffractive_interaction(
    bundle: RayBundle,
    surface: DiffractiveSurface,
    *,
    model: DiffractiveModel,
    parameters: ModelParameters,
    rng: np.random.Generator | None = None,
) -> DiffractiveInteractionResult:
    """``incident coherent rays + diffractive surface -> outgoing coherent rays``.

    One physical operation, with the model named explicitly. ``model`` and
    ``parameters`` must agree, and neither is inferred from the other: inferring
    the model from the parameter type would make a caller who passed the wrong
    parameters silently get a different physics model than the one they named,
    which is the failure this signature exists to prevent.

    Refusals, all structured and none of them a fallback:

    * ``GENERALIZED_SNELL`` on a ``CONFORMAL`` substrate --
      ``MISSING_DECLARATION``. The model needs a per-ray local tangent frame on
      a curved surface and has no way to accept one declared.
    * ``FULL_FIELD`` or ``LOCAL_PATCH`` with a declared index other than 1.0 --
      ``MISSING_DECLARATION``. Only ``GENERALIZED_SNELL`` uses a declared index.
    * ``GENERALIZED_SNELL`` with an evanescent requested order, or a local phase
      that varies too fast for the gradient estimate to be trusted -- see
      :func:`couplers.generalized_snell.generalized_snell_step`.
    * ``FULL_FIELD`` on a ``CONFORMAL`` substrate --
      :attr:`~core.boundary.ContractCode.MODEL_NOT_APPLICABLE`. There is no one
      common plane to accumulate onto, so the model's central step has no
      meaning here. It would still *run*, which is exactly why it is refused:
      accumulating rays that struck different tangent frames onto a single plane
      produces a field that looks like a diffraction pattern.
    * ``LOCAL_PATCH`` on a ``CONFORMAL`` substrate -- also refused today, but for
      a different reason and with a different code. The model *is* the
      applicable one there (SI S10); what is missing is the implementation --
      Newton sag intersection, per-hit tangent frames, position-dependent
      normals. The refusal comes from
      :func:`couplers.patch.plan_patches` as ``MISSING_DECLARATION``, and the
      two codes must not be collapsed: one says "never this model", the other
      says "this model, once someone builds it".
    * ``model``/``parameters`` mismatch -- ``MISSING_DECLARATION``.

    No numerics live here. Each branch forwards to the function that already
    implemented the model, so the result is bitwise what that function returns.
    """
    if not isinstance(model, DiffractiveModel):
        model = DiffractiveModel(str(model))

    expected = MODEL_COUPLER_IDS[model]
    wanted = {
        DiffractiveModel.FULL_FIELD: "FullFieldParameters",
        DiffractiveModel.LOCAL_PATCH: "LocalPatchParameters",
        DiffractiveModel.GENERALIZED_SNELL: "GeneralizedSnellParameters",
    }[model]
    if getattr(parameters, "model", None) is not model:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"model={model.value!r} was requested with "
            f"{type(parameters).__name__}, which configures "
            f"{getattr(getattr(parameters, 'model', None), 'value', 'nothing')!r}",
            declaration="parameters",
            remedy=(
                f"Pass the parameters for the model you named. The pairing is not "
                f"inferred in either direction, because a caller who names one model "
                f"and configures another must be told rather than quietly given the "
                f"other one's physics. {model.value!r} takes {wanted}."
            ),
        )

    if model is DiffractiveModel.FULL_FIELD:
        if surface.substrate is not Substrate.PLANAR:
            raise ContractError(
                ContractCode.MODEL_NOT_APPLICABLE,
                f"model='full_field' cannot be applied to a "
                f"{surface.substrate.value!r} substrate. The model's central step is "
                "one coherent accumulation onto the ONE common Cartesian plane every "
                "incident ray crosses; on a curved substrate rays intersect different "
                "local tangent planes with position-dependent frames and normals, so "
                "there is no such plane (SI S10)",
                declaration="model",
                remedy=(
                    "Use model='local_patch', which SI S10 identifies as the direct "
                    "implementation for exactly this case -- note that its conformal "
                    "path is itself not implemented yet, so it will refuse too, but "
                    "with MISSING_DECLARATION rather than MODEL_NOT_APPLICABLE. This "
                    "is a refusal rather than a fallback because the accumulation "
                    "would still compute: it would fold rays from different tangent "
                    "frames into one field and return something that looks like a "
                    "diffraction pattern."
                ),
            )
        _require_unit_indices(surface, model=model.value)
        assert isinstance(parameters, FullFieldParameters)  # narrowed by the pairing check
        outgoing, transmitted, cascade = planar_doe_step(
            bundle,
            surface.transmission,
            grid_shape=surface.grid_shape,
            sample_pitch_m=surface.sample_pitch_m,
            plane=surface.plane,
            launch_positions_xy_m=parameters.launch_positions_xy_m,
            primary_sampling=parameters.primary_sampling,
            primary_count=parameters.primary_count,
            secondary_count=parameters.secondary_count,
            density_kind=parameters.density_kind,
            preserve_energy=parameters.preserve_energy,
            pad_width=parameters.pad_width,
            rng=rng,
        )
        return DiffractiveInteractionResult(
            outgoing=outgoing,
            model=model,
            transmitted_field=transmitted,
            model_diagnostics=cascade,
            diagnostics={
                "interaction": INTERACTION_ID,
                "model": model.value,
                "coupler": expected,
                "substrate": surface.substrate.value,
                "radius_m": surface.radius_m,
                "grid_shape": list(surface.grid_shape),
                **cascade.as_dict(),
            },
        )

    if model is DiffractiveModel.GENERALIZED_SNELL:
        if surface.substrate is not Substrate.PLANAR:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                f"model='generalized_snell' cannot be applied to a "
                f"{surface.substrate.value!r} substrate: the tangential-momentum "
                "equation is evaluated in the surface's own local tangent frame, "
                "and on a curved substrate that frame is position-dependent. This "
                "model has no way to accept one declared per ray",
                declaration="substrate",
                remedy=(
                    "Use a planar substrate, which is the only one this model "
                    "accepts. A per-ray local frame is a future declaration, not "
                    "an inferred one -- CHE-143 does not add automatic Newton-sag "
                    "intersection."
                ),
            )
        assert isinstance(parameters, GeneralizedSnellParameters)  # narrowed by the pairing check
        outgoing, snell = generalized_snell_step(
            bundle,
            surface.transmission,
            sample_pitch_m=surface.sample_pitch_m,
            plane=surface.plane,
            n_incident=surface.n_incident,
            n_transmitted=surface.n_transmitted,
            order=parameters.order,
            patch_px=parameters.patch_px,
        )
        return DiffractiveInteractionResult(
            outgoing=outgoing,
            model=model,
            transmitted_field=None,
            model_diagnostics=snell,
            diagnostics={
                "interaction": INTERACTION_ID,
                "model": model.value,
                "coupler": expected,
                "substrate": surface.substrate.value,
                "radius_m": surface.radius_m,
                "grid_shape": list(surface.grid_shape),
                **snell.as_dict(),
            },
        )

    # LOCAL_PATCH. `plan_patches` owns the conformal refusal and the curvature
    # precondition; it is called rather than pre-empted so there is exactly one
    # place that decides what a patch plan may be.
    _require_unit_indices(surface, model=model.value)
    assert isinstance(parameters, LocalPatchParameters)  # narrowed by the pairing check
    if parameters.window is not PatchWindow.RECTANGULAR:
        raise ContractError(
            ContractCode.MODEL_NOT_APPLICABLE,
            f"window={parameters.window!r} is not implemented. Any taper below 1 "
            "removes field that no other patch replaces, so the coherent patch sum "
            "stops converging to the full-surface response -- the partition-of-unity "
            "argument behind the SI S2 convergence relation is what a taper breaks",
            declaration="window",
            remedy=(
                "Leave the window rectangular. A taper would have to be declared as "
                "trading the exactness guarantee for a smoother spectrum, with its "
                "own evidence, rather than selected as an option."
            ),
        )
    if parameters.spectral_density is not SamplingDensity.MAGNITUDE:
        raise ContractError(
            ContractCode.MODEL_NOT_APPLICABLE,
            f"spectral_density={parameters.spectral_density.value!r} is not "
            "switchable on this model: the patch emitter draws proportional to "
            "|U~| when sampling and uniformly when enumerating, and no other "
            "density has an implementation or a measurement here",
            declaration="spectral_density",
            remedy=(
                "Leave it at MAGNITUDE, which is what the emitter does. It is a "
                "field rather than a hidden constant so a record says which "
                "estimator produced it; a value that does not execute is refused "
                "instead of silently ignored."
            ),
        )

    plan = plan_patches(
        grid_shape=surface.grid_shape,
        sample_pitch_m=surface.sample_pitch_m,
        patch_px=parameters.patch_px,
        pad_factor=parameters.pad_factor,
        patch_count=parameters.patch_count,
        centers_xy_m=parameters.centers_xy_m,
        coverage_basis=parameters.coverage_basis,
        substrate=surface.substrate,
        radius_m=surface.radius_m,
        error_threshold_rad=parameters.error_threshold_rad,
        rng=rng,
        center_weights=parameters.center_weights,
    )
    outgoing, patch = patch_secondary_rays(
        surface.transmission,
        plan=plan,
        sample_pitch_m=surface.sample_pitch_m,
        wavelength_m=float(bundle.wavelength_m),
        plane=surface.plane,
        secondary_count=parameters.secondary_count,
        rng=rng,
    )
    return DiffractiveInteractionResult(
        outgoing=outgoing,
        model=model,
        transmitted_field=None,
        model_diagnostics=patch,
        diagnostics={
            "interaction": INTERACTION_ID,
            "model": model.value,
            "coupler": expected,
            "substrate": surface.substrate.value,
            "radius_m": surface.radius_m,
            "grid_shape": list(surface.grid_shape),
            "window": parameters.window.value,
            "spectral_density": parameters.spectral_density.value,
            **patch.as_dict(),
        },
    )
