"""The coupler package's ontology, and the interaction that made it necessary.

CHE-142 (M2.6). ``src/couplers/`` was using one word for three kinds of thing:
``C_RAY_TO_WAVE``/``C_WAVE_TO_RAY`` change *representation*,
``C_PLANAR_DOE_STEP``/``C_PATCH_WFT`` are one *physical interaction* at two
granularities, and ``advance_bundle_to_plane`` is *propagation*. The rule
installed here:

    representation transition != diffractive physical interaction != propagation

Two halves, and the second is the one that actually protects anything.

**The ontology is enforced, not documented.** ``couplers/ontology.py`` states the
partition; these tests hold it against the package's exports and against the
registry, so a role edited in one place and not the other fails rather than
drifting. A comment claiming the distinction would be worth nothing -- the
distinction was already claimed in prose before this ticket and the registry
still read as two unrelated DOE steps.

**The refactor changed no number, and that is asserted bitwise.** The entry point
forwards to the functions that already implemented each model, so a
``FULL_FIELD`` result must be bitwise the ``planar_doe_step`` result and a
``LOCAL_PATCH`` result bitwise the ``plan_patches`` + ``patch_secondary_rays``
result. Bitwise rather than to a tolerance on purpose: a tolerance would pass if
the wrapper had quietly reordered an operation, and the claim being made is
identity rather than agreement.

Small and CPU-only throughout: 16x16 and 33x33 grids, full enumeration, seconds
in total. The convergence evidence for either model lives in
``tests/test_planar_doe_step.py``, ``tests/test_patch_wft.py`` and the benchmark
probes, and is not re-measured here -- this file is about the ontology.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from core.boundary import ContractCode, ContractError, ReferencePlane
from core.specs import CouplerRole
from couplers.cascade import PrimarySampling, planar_doe_step
from couplers.generalized_snell import (
    local_gradient_smoothness_margin,
    propagating_order_margin,
    single_order_dominance,
)
from couplers.interaction import (
    INTERACTION_ID,
    MODEL_COUPLER_IDS,
    DiffractiveModel,
    DiffractiveSurface,
    FullFieldParameters,
    GeneralizedSnellParameters,
    LocalPatchParameters,
    PatchWindow,
    diffractive_interaction,
)
from couplers.ontology import (
    COUPLER_ROLES,
    OPERATION_ROLES,
    diffractive_models_of,
    role_of_coupler,
    role_of_operation,
)
from couplers.patch import CoverageBasis, Substrate, patch_secondary_rays, plan_patches
from couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)
from registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.coupler


# --------------------------------------------------------------------------- #
# Fixtures. The FULL_FIELD anchor matches tests/test_planar_doe_step.py and the
# LOCAL_PATCH one matches tests/test_patch_wft.py, so a bitwise claim here is
# made in the configuration those files already gate.
# --------------------------------------------------------------------------- #

WAVELENGTH_M = 500e-9
FF_N = 16
FF_PITCH = (1e-6, 1e-6)
PLANE = ReferencePlane(name="doe", z_m=0.0)

LP_N = 33
LP_PITCH_M = 6.3e-6
LP_PITCH = (LP_PITCH_M, LP_PITCH_M)
LP_WAVELENGTH_M = 0.7e-6


def _full_field_bundle(seed: int = 20260822):
    from core.boundary import ComplexField

    rng = np.random.default_rng(seed)
    field = ComplexField(
        u=(rng.normal(size=(FF_N, FF_N)) + 1j * rng.normal(size=(FF_N, FF_N))).astype(
            np.complex128
        ),
        sample_pitch_m=FF_PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )
    spectrum = decompose(field)
    density = sampling_density(spectrum)
    return spectrum_to_rays(spectrum, enumerate_indices(density), density)


def _full_field_transmission(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(FF_N, FF_N)))


def _local_patch_transmission(seed: int = 20260822) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(LP_N, LP_N)) + 1j * rng.normal(size=(LP_N, LP_N))).astype(
        np.complex128
    )


def _local_patch_bundle():
    """A one-ray bundle. The patch route reads only the wavelength off it.

    Worth stating rather than hiding: ``patch_secondary_rays`` takes its geometry
    from the plan and the surface, and the incident bundle contributes the
    wavelength. The entry point therefore reads the wavelength off the bundle for
    both models, which is the same rule ``FULL_FIELD`` already followed through
    ``ray_to_wave``.
    """
    from core.boundary import RayBundle

    return RayBundle(
        positions_m=np.zeros((1, 3)),
        directions=np.array([[0.0, 0.0, 1.0]]),
        wavelength_m=LP_WAVELENGTH_M,
        reference_plane=PLANE,
    )


def _planar_surface(transmission: np.ndarray, pitch: tuple[float, float]):
    return DiffractiveSurface(
        transmission=transmission, sample_pitch_m=pitch, plane=PLANE
    )


def _assert_bitwise(actual, expected, what: str) -> None:
    """Bitwise, on every array a bundle carries. See the module docstring."""
    for name in (
        "positions_m",
        "directions",
        "amplitude",
        "optical_path_length_m",
    ):
        left = getattr(actual, name, None)
        right = getattr(expected, name, None)
        if left is None and right is None:
            continue
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype, f"{what}: {name} dtype {left.dtype} != {right.dtype}"
        assert left.shape == right.shape, f"{what}: {name} shape {left.shape} != {right.shape}"
        assert np.array_equal(left, right, equal_nan=True), (
            f"{what}: {name} is not bitwise identical to the pre-refactor path. "
            "CHE-142 declared no numerical change, so this is either a real "
            "numerics change or a reordering inside the wrapper -- both of which "
            "invalidate every committed record produced by this operator."
        )


# --------------------------------------------------------------------------- #
# The three-way distinction, enforced
# --------------------------------------------------------------------------- #


def test_every_public_operation_of_the_package_has_exactly_one_role() -> None:
    """The partition itself. A role table with a hole defaults nothing -- it fails."""
    assert set(OPERATION_ROLES.values()) == set(CouplerRole), (
        "a role with no operation means the enumeration is wider than the package, "
        "or an operation was classified into the wrong bucket"
    )
    for name in OPERATION_ROLES:
        assert isinstance(role_of_operation(name), CouplerRole)


def test_an_unclassified_operation_is_refused_rather_than_defaulted() -> None:
    """The failure mode this table exists to prevent.

    A lookup that defaulted would classify the next representation transition as
    whatever the table's most common value happens to be, silently -- which is
    exactly how ``advance_bundle_to_plane`` came to live inside a diffractive
    model.
    """
    with pytest.raises(KeyError, match="has no declared role"):
        role_of_operation("couplers.patch.extract_patch")


def test_the_roles_partition_the_operations_they_claim_to() -> None:
    """Each named operation resolves, is importable, and is what its role says.

    Importability matters: a role table naming a callable that moved is a
    statement about a package that no longer exists.
    """
    import importlib

    for qualified, role in OPERATION_ROLES.items():
        module_name, _, attribute = qualified.rpartition(".")
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), f"{qualified} does not exist"
        assert isinstance(role, CouplerRole)

    def named(role: CouplerRole) -> set[str]:
        return {k for k, v in OPERATION_ROLES.items() if v is role}

    transitions = named(CouplerRole.REPRESENTATION_TRANSITION)
    interactions = named(CouplerRole.DIFFRACTIVE_INTERACTION)
    propagations = named(CouplerRole.PROPAGATION)
    assert not (transitions & interactions) and not (interactions & propagations)
    # The specific confusions this ticket resolved, asserted by name rather than
    # left to the set algebra above.
    assert "couplers.propagation.advance_bundle_to_plane" in propagations
    assert "couplers.cascade.planar_doe_step" in interactions
    assert "couplers.ray_to_wave.ray_to_wave" in transitions


def test_propagation_is_reachable_from_its_own_module_and_still_from_the_old_one() -> None:
    """The compatibility alias, asserted rather than assumed.

    ``advance_bundle_to_plane`` moved out of ``couplers.patch``. Every existing
    caller imports it from there, so the alias must be the same object -- not a
    re-implementation.
    """
    from couplers import advance_bundle_to_plane as from_package
    from couplers.patch import advance_bundle_to_plane as from_patch
    from couplers.propagation import advance_bundle_to_plane as from_propagation

    assert from_patch is from_propagation is from_package


def test_the_surfaces_own_vocabulary_is_reachable_from_the_interaction() -> None:
    """``Substrate`` and ``CoverageBasis`` are the surface's words, not one model's.

    Before this the substrate was declared to the LOCAL_PATCH planner and
    ``FULL_FIELD`` never saw it at all, so "which substrate is this?" had no
    single place to be asked. Re-exported rather than redefined, asserted by
    identity: two enums with the same members and different identity is the
    failure this would otherwise introduce, and it compares equal by value.
    """
    import couplers
    import couplers.interaction as interaction_module
    import couplers.patch as patch_module

    for name in ("Substrate", "CoverageBasis"):
        canonical = getattr(patch_module, name)
        assert getattr(interaction_module, name) is canonical
        assert getattr(couplers, name) is canonical


def test_the_registry_agrees_with_the_package_on_every_role() -> None:
    """One ontology, two places that state it, and they must not drift."""
    couplers = Registry.from_package().couplers
    assert set(couplers) == set(COUPLER_ROLES), (
        "a registered coupler with no declared role, or a role for a row that no "
        "longer exists"
    )
    for coupler_id, spec in couplers.items():
        assert spec.role is role_of_coupler(coupler_id), (
            f"{coupler_id}: registry says {spec.role.value!r}, "
            f"couplers/ontology.py says {role_of_coupler(coupler_id).value!r}"
        )


def test_a_diffractive_row_declares_the_shared_interaction_and_a_transition_does_not() -> None:
    """What requirement 4 buys: one identity, still-separate capability rows."""
    couplers = Registry.from_package().couplers
    diffractive = [
        spec
        for spec in couplers.values()
        if spec.role is CouplerRole.DIFFRACTIVE_INTERACTION
    ]
    assert len(diffractive) == 3
    assert {spec.interaction.id for spec in diffractive} == {INTERACTION_ID}
    models = {spec.interaction.model for spec in diffractive}
    assert models == {
        DiffractiveModel.FULL_FIELD.value,
        DiffractiveModel.LOCAL_PATCH.value,
        DiffractiveModel.GENERALIZED_SNELL.value,
    }

    for spec in couplers.values():
        if spec.role is CouplerRole.REPRESENTATION_TRANSITION:
            assert spec.interaction is None, (
                f"{spec.id} is a representation transition and must not claim a "
                "diffractive interaction identity"
            )

    # Each model's registry row is the row `interaction.py` points at.
    for model, coupler_id in MODEL_COUPLER_IDS.items():
        assert coupler_id is not None, f"{model} has no registry row"
        assert couplers[coupler_id].interaction.model == model.value
        assert diffractive_models_of(coupler_id) == (model,)


def test_sharing_an_interaction_identity_does_not_widen_the_narrower_capability() -> None:
    """The specific hazard in grouping two rows: C_PATCH_WFT is CPU-only.

    Grouping is an ontology statement, not a capability one. If it silently
    widened devices or dtypes, the group would be advertising a CUDA patch route
    that has never executed.
    """
    from core.capabilities import capabilities_for
    from core.specs import Device

    couplers = Registry.from_package().couplers
    assert couplers["C_PATCH_WFT"].devices == [Device.CPU]
    assert couplers["C_PLANAR_DOE_STEP"].devices == [Device.CPU, Device.GPU]
    patch_caps = capabilities_for("C_PATCH_WFT")
    step_caps = capabilities_for("C_PLANAR_DOE_STEP")
    assert patch_caps.devices != step_caps.devices
    assert patch_caps.native_compute_dtypes != step_caps.native_compute_dtypes


def test_the_knowledge_cards_describe_the_interaction_and_its_models() -> None:
    """``knowledge/`` must not read as two unrelated DOE steps either.

    Held against the registry rather than checked for prose, so a card that
    renames a model or drops the shared identity fails instead of quietly
    describing a different ontology than the one that executes. ``role`` in a
    card is the registry ID -- the card schema has always used that key that way
    -- so the ontology lives under ``operation_role`` and ``interaction``.
    """
    import yaml

    couplers = Registry.from_package().couplers
    packs = {
        "C_PLANAR_DOE_STEP": ROOT / "knowledge/couplers/planar_doe_step/card.yaml",
        "C_PATCH_WFT": ROOT / "knowledge/couplers/patch_wft/card.yaml",
    }
    for coupler_id, path in packs.items():
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert card["role"] == coupler_id
        assert card["operation_role"] == CouplerRole.DIFFRACTIVE_INTERACTION.value
        interaction = card["interaction"]
        assert interaction["id"] == couplers[coupler_id].interaction.id
        assert interaction["model"] == couplers[coupler_id].interaction.model
        assert interaction["entry_point"] == (
            "couplers.interaction.diffractive_interaction"
        )
        # Every model of the interaction is accounted for by exactly one card
        # entry: its own `model`, plus one peer entry per other model.
        named = {interaction["model"]} | {
            peer["model"] for peer in interaction["peer_models"]
        }
        assert named == {model.value for model in DiffractiveModel}, (
            f"{path.name} names {sorted(named)}; the interaction has "
            f"{sorted(m.value for m in DiffractiveModel)}. A card that omits a "
            "model tells a reader the choice is narrower than it is"
        )
        for peer in interaction["peer_models"]:
            if peer["pack"] is not None:
                assert (ROOT / peer["pack"] / "card.yaml").is_file()

    # The README is where an agent lands first, so the relation has to be there.
    readme = (ROOT / "knowledge/couplers/README.md").read_text(encoding="utf-8")
    assert "representation transition" in readme and "propagation" in readme
    assert "I_DIFFRACTIVE" in readme
    assert "shortcut, not a peer" in readme


def test_the_si_s10_relation_is_stated_where_the_models_are_declared() -> None:
    """"Which is the shortcut for which" must be readable at the declaration.

    Not a style check. Two rows that share an identity and do not say how they
    relate are back to reading as peers, which is the defect CHE-142 opened on.
    """
    couplers = Registry.from_package().couplers
    full = couplers["C_PLANAR_DOE_STEP"].interaction.relation.lower()
    patch = couplers["C_PATCH_WFT"].interaction.relation.lower()
    assert "shortcut" in full and "s10" in full
    assert "direct implementation" in patch and "s10" in patch

    import couplers.interaction as interaction_module

    doc = (interaction_module.__doc__ or "").lower()
    assert "s10" in doc and "shortcut" in doc
    assert "overlap" in doc, (
        "the two models' regimes overlap on a planar substrate; a doc that "
        "implies a partition tells a caller the choice is forced when it is not"
    )


# --------------------------------------------------------------------------- #
# One entry point, model named explicitly, no numerical change
# --------------------------------------------------------------------------- #


def test_full_field_through_the_entry_point_is_bitwise_the_shipped_step() -> None:
    """Acceptance criterion: bitwise identity for at least one FULL_FIELD case."""
    bundle = _full_field_bundle()
    transmission = _full_field_transmission()
    launch = np.zeros((1, 2))

    expected, expected_field, expected_diagnostics = planar_doe_step(
        bundle,
        transmission,
        grid_shape=(FF_N, FF_N),
        sample_pitch_m=FF_PITCH,
        plane=PLANE,
        launch_positions_xy_m=launch,
        secondary_count=None,
    )
    result = diffractive_interaction(
        bundle,
        _planar_surface(transmission, FF_PITCH),
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(launch_positions_xy_m=launch),
    )

    _assert_bitwise(result.outgoing, expected, "FULL_FIELD")
    assert np.array_equal(result.transmitted_field.u, expected_field.u)
    assert result.model is DiffractiveModel.FULL_FIELD
    assert result.interaction_id == INTERACTION_ID
    assert result.diagnostics["coupler"] == "C_PLANAR_DOE_STEP"
    assert result.diagnostics["enumerated"] is expected_diagnostics.enumerated
    # The conventions the step declares survive into the unified diagnostics,
    # because a consumer reads the record and not this test.
    assert "reset to 0 at this plane" in result.diagnostics["opl_convention"]


def test_full_field_with_a_seeded_sampler_is_bitwise_the_shipped_step() -> None:
    """The stochastic configuration too: same seed, same generator, same draw.

    A wrapper that consumed one extra value from the generator before forwarding
    would pass the enumerated test above and fail here, which is why both are
    present.
    """
    bundle = _full_field_bundle()
    transmission = _full_field_transmission()
    kwargs = dict(
        primary_sampling=PrimarySampling.UNIFORM_ON_GRID,
        primary_count=4,
        secondary_count=8,
        density_kind=SamplingDensity.MAGNITUDE,
    )
    expected, _, _ = planar_doe_step(
        bundle,
        transmission,
        grid_shape=(FF_N, FF_N),
        sample_pitch_m=FF_PITCH,
        plane=PLANE,
        rng=np.random.default_rng(7),
        **kwargs,
    )
    result = diffractive_interaction(
        bundle,
        _planar_surface(transmission, FF_PITCH),
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(**kwargs),
        rng=np.random.default_rng(7),
    )
    _assert_bitwise(result.outgoing, expected, "FULL_FIELD sampled")


def test_local_patch_through_the_entry_point_is_bitwise_the_shipped_route() -> None:
    """Acceptance criterion: bitwise identity for at least one LOCAL_PATCH case.

    The full-aperture single patch, which is the exactness anchor and the
    configuration in which ``FULL_FIELD`` is this model's special case.
    """
    transmission = _local_patch_transmission()
    bundle = _local_patch_bundle()

    plan = plan_patches(
        grid_shape=(LP_N, LP_N),
        sample_pitch_m=LP_PITCH,
        patch_px=LP_N,
        pad_factor=2,
    )
    expected, expected_diagnostics = patch_secondary_rays(
        transmission,
        plan=plan,
        sample_pitch_m=LP_PITCH,
        wavelength_m=LP_WAVELENGTH_M,
        plane=PLANE,
        secondary_count=None,
    )
    result = diffractive_interaction(
        bundle,
        _planar_surface(transmission, LP_PITCH),
        model=DiffractiveModel.LOCAL_PATCH,
        parameters=LocalPatchParameters(patch_px=LP_N),
    )

    _assert_bitwise(result.outgoing, expected, "LOCAL_PATCH")
    assert result.model is DiffractiveModel.LOCAL_PATCH
    assert result.diagnostics["coupler"] == "C_PATCH_WFT"
    assert result.diagnostics["pad_px"] == expected_diagnostics.pad_px
    assert result.diagnostics["coverage"] == expected_diagnostics.coverage
    # No global field is synthesized. See DiffractiveInteractionResult.
    assert result.transmitted_field is None


def test_local_patch_with_drawn_centres_is_bitwise_the_shipped_route() -> None:
    """Many patches and a seeded draw, so the plan and the emitter share an rng."""
    transmission = _local_patch_transmission()
    bundle = _local_patch_bundle()

    plan = plan_patches(
        grid_shape=(LP_N, LP_N),
        sample_pitch_m=LP_PITCH,
        patch_px=11,
        pad_factor=2,
        patch_count=6,
        rng=np.random.default_rng(11),
    )
    expected, _ = patch_secondary_rays(
        transmission,
        plan=plan,
        sample_pitch_m=LP_PITCH,
        wavelength_m=LP_WAVELENGTH_M,
        plane=PLANE,
        secondary_count=None,
        rng=np.random.default_rng(11),
    )
    # One generator, both stages -- which is what the shipped graph node does.
    # Reproducing it needs the same object, so the reference above is built with
    # two generators at the same seed and the plan draw is the only consumer.
    result = diffractive_interaction(
        bundle,
        _planar_surface(transmission, LP_PITCH),
        model=DiffractiveModel.LOCAL_PATCH,
        parameters=LocalPatchParameters(patch_px=11, patch_count=6),
        rng=np.random.default_rng(11),
    )
    _assert_bitwise(result.outgoing, expected, "LOCAL_PATCH drawn")


# --------------------------------------------------------------------------- #
# Refusals: a wrong model for a declared surface, and never a default
# --------------------------------------------------------------------------- #


def test_full_field_on_a_conformal_substrate_is_refused_not_defaulted() -> None:
    """The acceptance criterion's named case.

    ``MODEL_NOT_APPLICABLE`` rather than ``MISSING_DECLARATION``: nothing is
    missing, and no further declaration would help. The remedy is the other
    model.
    """
    surface = DiffractiveSurface(
        transmission=_full_field_transmission(),
        sample_pitch_m=FF_PITCH,
        plane=PLANE,
        substrate=Substrate.CONFORMAL,
        radius_m=2e-3,
    )
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _full_field_bundle(),
            surface,
            model=DiffractiveModel.FULL_FIELD,
            parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2))),
        )
    error = excinfo.value
    assert error.code is ContractCode.MODEL_NOT_APPLICABLE
    assert error.declaration == "model"
    assert "local_patch" in (error.remedy or ""), "a refusal must name the way forward"
    assert "s10" in str(error).lower()


def test_local_patch_on_a_conformal_substrate_refuses_with_a_different_code() -> None:
    """The two conformal refusals must not collapse into one.

    ``FULL_FIELD`` there is never going to be right; ``LOCAL_PATCH`` is the right
    model and is missing an implementation. An agent that cannot tell those apart
    cannot tell "pick another model" from "wait for the work".
    """
    surface = DiffractiveSurface(
        transmission=_local_patch_transmission(),
        sample_pitch_m=LP_PITCH,
        plane=PLANE,
        substrate=Substrate.CONFORMAL,
        radius_m=5e-3,
    )
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.LOCAL_PATCH,
            parameters=LocalPatchParameters(patch_px=11),
        )
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "substrate"
    assert "not implemented" in str(excinfo.value)


def test_generalized_snell_on_a_conformal_substrate_is_refused() -> None:
    """The one refusal GENERALIZED_SNELL itself owns: no per-ray local frame.

    CHE-143 (M2.7) delivers the planar case. A conformal substrate needs a
    per-ray local tangent frame -- the surface normal is position-dependent --
    and this model has no way to accept one declared, so every conformal call
    is refused rather than approximated with the flat-plane frame.
    """
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            DiffractiveSurface(
                transmission=_local_patch_transmission(),
                sample_pitch_m=LP_PITCH,
                plane=PLANE,
                substrate=Substrate.CONFORMAL,
                radius_m=1e-2,
            ),
            model=DiffractiveModel.GENERALIZED_SNELL,
            parameters=GeneralizedSnellParameters(),
        )
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "substrate"


def test_the_model_and_its_parameters_are_not_inferred_from_each_other() -> None:
    """Naming one model and configuring another is refused in both directions.

    Inferring would hand a caller the *other* model's physics under the name they
    asked for, which no diagnostic downstream could recover.
    """
    surface = _planar_surface(_local_patch_transmission(), LP_PITCH)
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.FULL_FIELD,
            parameters=LocalPatchParameters(patch_px=11),
        )
    assert excinfo.value.declaration == "parameters"
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.LOCAL_PATCH,
            parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2))),
        )
    assert excinfo.value.declaration == "parameters"
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.GENERALIZED_SNELL,
            parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2))),
        )
    assert excinfo.value.declaration == "parameters"


# --------------------------------------------------------------------------- #
# The declared surface
# --------------------------------------------------------------------------- #


def test_a_real_transmission_is_refused_as_an_undeclared_phase() -> None:
    with pytest.raises(ContractError) as excinfo:
        DiffractiveSurface(
            transmission=np.ones((4, 4)), sample_pitch_m=FF_PITCH, plane=PLANE
        )
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "transmission"


def test_the_substrate_and_its_radius_may_not_disagree() -> None:
    """Two declarations of the same fact, and the pair selects the validity argument.

    Planar gets a hard gate (the tangent plane is exact everywhere); curved gets
    a bound. A row that says planar and carries a finite radius has not chosen.
    """
    with pytest.raises(ContractError) as excinfo:
        DiffractiveSurface(
            transmission=_full_field_transmission(),
            sample_pitch_m=FF_PITCH,
            plane=PLANE,
            radius_m=1e-3,
        )
    assert excinfo.value.declaration == "radius_m"

    with pytest.raises(ContractError) as excinfo:
        DiffractiveSurface(
            transmission=_full_field_transmission(),
            sample_pitch_m=FF_PITCH,
            plane=PLANE,
            substrate=Substrate.CONFORMAL,
        )
    assert excinfo.value.declaration == "radius_m"
    assert "arcsin" in str(excinfo.value), "the bound that needs R must be named"


def test_a_refractive_index_other_than_one_is_refused_for_full_field_and_local_patch() -> None:
    """Only ``GENERALIZED_SNELL`` implements a declared index.

    ``DiffractiveSurface`` accepts the declaration unconditionally -- it does
    not know which model will read it -- so the refusal is at dispatch, one
    per model that cannot use it. Silently dropping it would be a wavelength
    error of ``n``: a plausible-looking defocus rather than a visible failure.
    """
    for kwargs in ({"n_incident": 1.5}, {"n_transmitted": 1.5}):
        surface = DiffractiveSurface(
            transmission=_full_field_transmission(),
            sample_pitch_m=FF_PITCH,
            plane=PLANE,
            **kwargs,
        )
        with pytest.raises(ContractError) as excinfo:
            diffractive_interaction(
                _full_field_bundle(),
                surface,
                model=DiffractiveModel.FULL_FIELD,
                parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2))),
            )
        assert excinfo.value.declaration in {"n_incident", "n_transmitted"}
        assert "not implemented" in str(excinfo.value)

        local_surface = DiffractiveSurface(
            transmission=_local_patch_transmission(),
            sample_pitch_m=LP_PITCH,
            plane=PLANE,
            **kwargs,
        )
        with pytest.raises(ContractError) as excinfo:
            diffractive_interaction(
                _local_patch_bundle(),
                local_surface,
                model=DiffractiveModel.LOCAL_PATCH,
                parameters=LocalPatchParameters(patch_px=11),
            )
        assert excinfo.value.declaration in {"n_incident", "n_transmitted"}
        assert "not implemented" in str(excinfo.value)


def test_the_grid_shape_comes_off_the_transmission_rather_than_beside_it() -> None:
    """The shape disagreement the old signature made possible is now unreachable."""
    surface = _planar_surface(_full_field_transmission(), FF_PITCH)
    assert surface.grid_shape == (FF_N, FF_N)
    assert math.isinf(surface.radius_m)
    assert surface.substrate is Substrate.PLANAR


def test_from_phase_applies_the_repository_phasor_sign_in_one_place() -> None:
    """``t = exp(+i phi)``. A conjugated surface is a DOE that focuses the wrong way."""
    rng = np.random.default_rng(3)
    phase = rng.uniform(-np.pi, np.pi, size=(FF_N, FF_N))
    surface = DiffractiveSurface.from_phase(
        phase, sample_pitch_m=FF_PITCH, plane=PLANE
    )
    assert np.allclose(np.angle(surface.transmission), phase)
    assert np.allclose(np.abs(surface.transmission), 1.0)


def test_the_patch_model_parameters_that_do_not_execute_are_refused() -> None:
    """The window and the spectral density are declared, not dials.

    Both are fields so a record says which estimator ran; a value that does not
    execute is refused rather than silently ignored, because "I asked for a Hann
    window and got a rectangle" is invisible in the output.
    """
    surface = _planar_surface(_local_patch_transmission(), LP_PITCH)
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.LOCAL_PATCH,
            parameters=LocalPatchParameters(
                patch_px=11, spectral_density=SamplingDensity.UNIFORM
            ),
        )
    assert excinfo.value.code is ContractCode.MODEL_NOT_APPLICABLE
    assert excinfo.value.declaration == "spectral_density"
    assert list(PatchWindow) == [PatchWindow.RECTANGULAR], (
        "one member is the honest count; a taper would arrive with its own "
        "evidence rather than as a second option"
    )


def test_supplied_centres_still_need_their_coverage_basis() -> None:
    """The refusal that survives being moved behind a new entry point.

    ``A_draw / A_patch`` is unbiased only for a known density, and the density is
    not recoverable from the positions. Guessing scales the whole field by a
    constant that looks plausible.
    """
    surface = _planar_surface(_local_patch_transmission(), LP_PITCH)
    centres = np.zeros((3, 2))
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            _local_patch_bundle(),
            surface,
            model=DiffractiveModel.LOCAL_PATCH,
            parameters=LocalPatchParameters(patch_px=11, centers_xy_m=centres),
        )
    assert excinfo.value.declaration == "coverage_basis"

    # ...and it runs once declared, so the refusal is about the declaration and
    # not about the configuration being unsupported.
    result = diffractive_interaction(
        _local_patch_bundle(),
        surface,
        model=DiffractiveModel.LOCAL_PATCH,
        parameters=LocalPatchParameters(
            patch_px=11,
            centers_xy_m=centres,
            coverage_basis=CoverageBasis.UNIFORM_OVER_DILATED_APERTURE,
        ),
    )
    assert result.diagnostics["patch_count"] == 3


# --------------------------------------------------------------------------- #
# GENERALIZED_SNELL (CHE-143, M2.7): the reduced-order model
#
# Small and analytic throughout, in the same spirit as the FULL_FIELD /
# LOCAL_PATCH sections above: no field is ever formed here, so there is no
# convergence evidence to gate -- the claims are closed-form (the linear-ramp
# grating equation, ordinary Snell's law, undeflected propagation) or direct
# measurements of a declared margin, and every case is one or a few rays.
# --------------------------------------------------------------------------- #

GS_WAVELENGTH_M = 500e-9
GS_PLANE = ReferencePlane(name="doe", z_m=0.0)
GS_PITCH = (1e-6, 1e-6)
GS_N = 64


def _gs_bundle(directions, positions=None):
    directions = np.asarray(directions, dtype=np.float64)
    if positions is None:
        positions = np.zeros((directions.shape[0], 3))
    count = directions.shape[0]
    from core.boundary import RayBundle

    return RayBundle(
        positions_m=np.asarray(positions, dtype=np.float64),
        directions=directions,
        wavelength_m=GS_WAVELENGTH_M,
        reference_plane=GS_PLANE,
        amplitude=np.ones(count, dtype=np.complex128),
        optical_path_length_m=np.zeros(count),
        optical_path_length_reference="launch",
    )


def _linear_ramp_surface(period_m: float, *, pitch=GS_PITCH, n=GS_N, sign: float = 1.0):
    """A DOE whose phase is one global linear grating -- the exact closed form."""
    x = (np.arange(n) - n // 2) * pitch[1]
    phase = sign * 2.0 * np.pi * x / period_m
    phase2d = np.tile(phase, (n, 1))
    return DiffractiveSurface.from_phase(phase2d, sample_pitch_m=pitch, plane=GS_PLANE)


def test_a_linear_phase_ramp_deflects_to_the_exact_grating_angle() -> None:
    """``sin(theta_out) = (n_i sin(theta_in) + m lambda / Lambda) / n_t``, to
    round-off. The one configuration where GSL is not an approximation."""
    period_m = 5e-6
    surface = _linear_ramp_surface(period_m)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    expected_sin = GS_WAVELENGTH_M / period_m
    assert math.isclose(
        float(result.outgoing.directions[0, 0]), expected_sin, rel_tol=0.0, abs_tol=1e-12
    )
    assert math.isclose(float(result.outgoing.directions[0, 1]), 0.0, abs_tol=1e-12)


def test_a_linear_phase_ramp_at_oblique_incidence_matches_the_full_grating_equation() -> None:
    period_m = 6e-6
    surface = _linear_ramp_surface(period_m)
    theta_in = math.radians(15.0)
    bundle = _gs_bundle([[math.sin(theta_in), 0.0, math.cos(theta_in)]])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    expected_sin = math.sin(theta_in) + GS_WAVELENGTH_M / period_m
    assert math.isclose(
        float(result.outgoing.directions[0, 0]), expected_sin, rel_tol=0.0, abs_tol=1e-9
    )


def test_the_zero_phase_limit_is_ordinary_snells_law() -> None:
    """``phi = 0`` everywhere -- its gradient is zero, so GSL degenerates to the
    index-only term of its own equation, which is Snell's law."""
    flat = np.ones((GS_N, GS_N), dtype=np.complex128)
    surface = DiffractiveSurface(
        transmission=flat, sample_pitch_m=GS_PITCH, plane=GS_PLANE, n_transmitted=1.5
    )
    theta_in = math.radians(20.0)
    bundle = _gs_bundle([[math.sin(theta_in), 0.0, math.cos(theta_in)]])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    theta_out = math.asin(float(result.outgoing.directions[0, 0]))
    assert math.isclose(1.0 * math.sin(theta_in), 1.5 * math.sin(theta_out), rel_tol=1e-12)


def test_the_zero_gradient_limit_with_equal_indices_is_undeflected_propagation() -> None:
    """``phi = 0`` and ``n_i = n_t`` -- the ray must pass straight through."""
    flat = np.ones((GS_N, GS_N), dtype=np.complex128)
    surface = DiffractiveSurface(transmission=flat, sample_pitch_m=GS_PITCH, plane=GS_PLANE)
    direction = [math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0))]
    bundle = _gs_bundle([direction])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    assert np.allclose(result.outgoing.directions[0], direction, atol=1e-12)


def test_an_evanescent_requested_order_is_refused_not_returned_as_nonsense() -> None:
    period_m = 0.55e-6
    pitch = (0.05e-6, 0.05e-6)
    surface = _linear_ramp_surface(period_m, pitch=pitch)
    theta_in = math.asin(0.3)
    bundle = _gs_bundle([[math.sin(theta_in), 0.0, math.cos(theta_in)]])
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            bundle,
            surface,
            model=DiffractiveModel.GENERALIZED_SNELL,
            parameters=GeneralizedSnellParameters(order=1, patch_px=5),
        )
    assert excinfo.value.code is ContractCode.MODEL_NOT_APPLICABLE
    assert excinfo.value.declaration == "order"
    assert "evanescent" in str(excinfo.value)


def test_a_local_phase_discontinuity_is_refused_and_the_refusal_is_local() -> None:
    """A sharp jump breaks the gradient estimate for a ray that sits on it, and
    only for that ray -- a ray far from the jump is unaffected."""
    pitch = GS_PITCH
    n = 32
    x = (np.arange(n) - n // 2) * pitch[1]
    phase = 0.01 * x / pitch[1]
    phase2d = np.tile(phase, (n, 1)).astype(np.float64)
    jump_col = n // 2 + 2
    phase2d[:, jump_col:] += 3.0
    surface = DiffractiveSurface.from_phase(phase2d, sample_pitch_m=pitch, plane=GS_PLANE)

    ray_x = (jump_col - n // 2) * pitch[1]
    on_the_jump = _gs_bundle([[0.0, 0.0, 1.0]], positions=[[ray_x, 0.0, 0.0]])
    with pytest.raises(ContractError) as excinfo:
        diffractive_interaction(
            on_the_jump,
            surface,
            model=DiffractiveModel.GENERALIZED_SNELL,
            parameters=GeneralizedSnellParameters(order=1, patch_px=5),
        )
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "patch_px"

    far_from_it = _gs_bundle([[0.0, 0.0, 1.0]], positions=[[-10e-6, 0.0, 0.0]])
    result = diffractive_interaction(
        far_from_it,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    assert result.diagnostics["worst_local_gradient_smoothness_margin"] > 0.0


def test_an_edge_ray_reads_its_own_phase_and_amplitude_from_the_same_pixel() -> None:
    """A ray at the array's extreme edge column must read ITS OWN pixel's
    phase and amplitude, not the interior pixel the gradient stencil centres
    on to stay in bounds.

    The gradient/curvature stencil needs its centre clamped to the interior
    (``[2, n-3]``) to keep every tap in the array, but that clamped location
    must not leak into the ray's own phase (and hence OPL) or amplitude --
    those are properties of the ray's own transverse position, declared
    exactly by the additive OPL convention (``OPL_out = OPL_in + phi/k0``,
    ``amplitude_out = amplitude_in * |t|``) that
    ``knowledge/couplers/generalized_snell/conventions.md`` states.
    """
    pitch = GS_PITCH
    n = 32
    period_m = 50e-6  # smooth: no discontinuity anywhere, well inside validity
    x = (np.arange(n) - n // 2) * pitch[1]
    phase = 2.0 * np.pi * x / period_m
    magnitude = 1.0 + 0.01 * np.arange(n)  # position-dependent, so edge != interior
    transmission = (
        np.tile(magnitude, (n, 1)).astype(np.complex128) * np.exp(1j * np.tile(phase, (n, 1)))
    )
    surface = DiffractiveSurface(transmission=transmission, sample_pitch_m=pitch, plane=GS_PLANE)

    edge_col = 0
    edge_x = (edge_col - n // 2) * pitch[1]
    incident_opl = 1.23e-6
    bundle = _gs_bundle([[0.0, 0.0, 1.0]], positions=[[edge_x, 0.0, 0.0]])
    bundle = bundle.with_declared_optical_path_length(
        np.full(1, incident_opl), reference="test incident plane"
    )
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    k0 = 2.0 * math.pi / GS_WAVELENGTH_M
    assert math.isclose(
        float(result.outgoing.optical_path_length_m[0]),
        incident_opl + phase[edge_col] / k0,
        rel_tol=0.0,
        abs_tol=1e-18,
    )
    assert math.isclose(
        float(np.abs(result.outgoing.amplitude[0])), magnitude[edge_col], rel_tol=1e-12
    )
    # The regression this locks in: the interior-clamped stencil centre (used
    # only for the gradient/curvature taps) sits at column 2, not 0. If the
    # ray's own phase were ever read from that clamped location instead of its
    # own, this would silently pass with the WRONG expected value substituted
    # for the right one -- so assert the two columns actually differ.
    assert not math.isclose(phase[edge_col], phase[2], abs_tol=1e-9)
    assert not math.isclose(magnitude[edge_col], magnitude[2], abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Negative controls (CHE-143 Stage 3). Test-local reimplementations of a wrong
# formula, asserted to disagree with the exact analytic grating angle -- these
# are verification controls, not alternative production code.
# --------------------------------------------------------------------------- #


def test_control_a_phasor_sign_flip_conjugates_the_deflection() -> None:
    period_m = 5e-6
    plus = _linear_ramp_surface(period_m, sign=+1.0)
    minus = _linear_ramp_surface(period_m, sign=-1.0)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    result_plus = diffractive_interaction(
        bundle,
        plus,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    result_minus = diffractive_interaction(
        bundle,
        minus,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    dx_plus = float(result_plus.outgoing.directions[0, 0])
    dx_minus = float(result_minus.outgoing.directions[0, 0])
    assert math.isclose(dx_plus, -dx_minus, rel_tol=0.0, abs_tol=1e-12)
    assert abs(dx_plus) > 1e-6, "the control is void if the deflection itself is zero"


def test_control_an_order_sign_flip_conjugates_the_deflection() -> None:
    period_m = 5e-6
    surface = _linear_ramp_surface(period_m)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    result_plus = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    result_minus = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=-1, patch_px=5),
    )
    dx_plus = float(result_plus.outgoing.directions[0, 0])
    dx_minus = float(result_minus.outgoing.directions[0, 0])
    assert math.isclose(dx_plus, -dx_minus, rel_tol=0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# The order factor in the OPL rebasing -- CHE-148 (M2.12)
# ---------------------------------------------------------------------------
#
# `generalized_snell_step` returns `opl_out = opl_in + m phi / k0`. The `m` was
# missing until CHE-148's B3-DOE-INLINE-ORDER-MINUS1 instance reported a
# Strehl-like ratio of 3.2e-6 against a predicted 0.99967 on a system whose
# ray-side order position was correct to 9.3e-5: the rays were deflected as if
# the phase were `-phi` while the optical path carried `+phi`. Both tests below
# are code-independent identities rather than recorded numbers, which is why
# they are the regression and the diagnostic at the same time.


def test_negating_the_order_equals_conjugating_the_surface_in_every_field() -> None:
    """``exp(i (-1) phi)`` and ``exp(i (+1) (-phi))`` are one complex factor.

    So ``(order=-1, t)`` and ``(order=+1, conj(t))`` are one physical operation
    and must return one bundle -- bitwise, since no approximation separates them.
    With ``phi`` instead of ``m phi`` in the rebasing they agreed in DIRECTION and
    returned opposite optical paths, which is a contradiction rather than a
    tolerance.
    """
    period_m = 5e-6
    positions = np.array([[0.0, 0.0, 0.0], [1.3e-6, 0.0, 0.0], [-2.9e-6, 0.0, 0.0]])
    bundle = _gs_bundle([[0.0, 0.0, 1.0]] * 3, positions=positions)

    negated_order = diffractive_interaction(
        bundle,
        _linear_ramp_surface(period_m),
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=-1, patch_px=5),
    ).outgoing
    conjugated_surface = diffractive_interaction(
        bundle,
        _linear_ramp_surface(period_m, sign=-1.0),
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    ).outgoing

    assert np.array_equal(
        np.asarray(negated_order.directions), np.asarray(conjugated_surface.directions)
    )
    assert np.array_equal(
        np.asarray(negated_order.optical_path_length_m),
        np.asarray(conjugated_surface.optical_path_length_m),
    )
    # The control is void if the ramp put no phase on the rays in the first place.
    spread = np.ptp(np.asarray(negated_order.optical_path_length_m))
    assert spread > 0.1 * GS_WAVELENGTH_M


def test_the_zeroth_order_picks_up_no_surface_phase_at_all() -> None:
    """``order=0`` is the undiffracted transmission: no deflection, no ramp.

    With ``phi`` instead of ``m phi`` it was handed the whole ramp phase on a ray
    that had not been deflected -- a pupil phase with no matching momentum.
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.3e-6, 0.0, 0.0], [-2.9e-6, 0.0, 0.0]])
    bundle = _gs_bundle([[0.0, 0.0, 1.0]] * 3, positions=positions)
    result = diffractive_interaction(
        bundle,
        _linear_ramp_surface(5e-6),
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=0, patch_px=5),
    ).outgoing
    assert np.array_equal(np.asarray(result.directions), np.asarray(bundle.directions))
    assert np.array_equal(
        np.asarray(result.optical_path_length_m),
        np.asarray(bundle.optical_path_length_m),
    )


def test_the_first_order_rebasing_is_exactly_the_local_phase_over_k0() -> None:
    """``order=1`` must be bitwise what it was before the order factor existed.

    ``float(1) * phase`` is IEEE-exact, so every record and every test written
    before CHE-148 is unaffected. Pinned rather than assumed, because the fix
    would otherwise be a silent change to the default order.
    """
    period_m = 5e-6
    positions = np.array([[0.0, 0.0, 0.0], [1.3e-6, 0.0, 0.0], [-2.9e-6, 0.0, 0.0]])
    bundle = _gs_bundle([[0.0, 0.0, 1.0]] * 3, positions=positions)
    surface = _linear_ramp_surface(period_m)
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    ).outgoing

    pitch = GS_PITCH[1]
    snapped = np.round(positions[:, 0] / pitch) * pitch
    phase = np.arctan2(
        np.sin(2.0 * np.pi * snapped / period_m), np.cos(2.0 * np.pi * snapped / period_m)
    )
    k0 = 2.0 * math.pi / GS_WAVELENGTH_M
    expected = np.asarray(bundle.optical_path_length_m) + phase / k0
    assert np.allclose(
        np.asarray(result.optical_path_length_m), expected, rtol=0.0, atol=1e-18
    )


def test_control_omitting_2pi_from_the_gradient_gives_the_wrong_angle() -> None:
    """A caller who computed the gradient of ``phi / (2 pi)`` -- cycles, not
    radians -- and fed it straight into the k_t equation would be off by
    exactly ``2 pi``. Reimplemented here, test-local, to show it disagrees.
    """
    period_m = 5e-6
    n = GS_N
    pitch = GS_PITCH
    x = (np.arange(n) - n // 2) * pitch[1]
    cycles = x / period_m  # NOT multiplied by 2*pi -- the bug under test
    grad_wrong = np.gradient(cycles, pitch[1])[n // 2]  # rad/m, missing the 2*pi factor
    k0 = 2.0 * math.pi / GS_WAVELENGTH_M
    k_t_wrong = 1.0 * k0 * 0.0 + 1 * grad_wrong
    wrong_sin = k_t_wrong / k0
    correct_sin = GS_WAVELENGTH_M / period_m
    assert not math.isclose(wrong_sin, correct_sin, rel_tol=1e-6)

    # The production path, on the same surface, gets the correct answer.
    surface = _linear_ramp_surface(period_m)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    assert math.isclose(
        float(result.outgoing.directions[0, 0]), correct_sin, rel_tol=0.0, abs_tol=1e-12
    )


def test_control_a_gradient_in_pixels_instead_of_metres_gives_the_wrong_angle() -> None:
    """A caller who forgot to divide the per-sample phase step by the pitch
    would report a gradient in rad/sample, not rad/m -- wrong by a factor of
    the pitch itself, reimplemented here to show it disagrees.
    """
    period_m = 5e-6
    n = GS_N
    pitch = GS_PITCH
    grad_per_sample = 2.0 * math.pi * pitch[1] / period_m  # rad/sample (correct: rad/m)
    k0 = 2.0 * math.pi / GS_WAVELENGTH_M
    wrong_sin = grad_per_sample / k0  # missing the "/ pitch" that converts to rad/m
    correct_sin = GS_WAVELENGTH_M / period_m
    assert not math.isclose(wrong_sin, correct_sin, rel_tol=1e-6)

    surface = _linear_ramp_surface(period_m)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=5),
    )
    assert math.isclose(
        float(result.outgoing.directions[0, 0]), correct_sin, rel_tol=0.0, abs_tol=1e-12
    )


# --------------------------------------------------------------------------- #
# Validity predicates (CHE-143 Stage 4): signed normalized margins, evaluated
# directly rather than through the runtime refusal path.
# --------------------------------------------------------------------------- #


def test_propagating_order_margin_sign_convention() -> None:
    k0 = 2.0 * math.pi / GS_WAVELENGTH_M
    limit = k0  # n_transmitted = 1.0
    assert propagating_order_margin(np.array([0.0]), n_transmitted=1.0, k0=k0)[0] == 1.0
    assert propagating_order_margin(np.array([limit**2]), n_transmitted=1.0, k0=k0)[0] == 0.0
    inside = propagating_order_margin(np.array([(0.5 * limit) ** 2]), n_transmitted=1.0, k0=k0)[0]
    outside = propagating_order_margin(np.array([(1.5 * limit) ** 2]), n_transmitted=1.0, k0=k0)[0]
    assert inside > 0.0 > outside


def test_local_gradient_smoothness_margin_is_perfect_for_a_pure_ramp() -> None:
    """Zero curvature -- a linear ramp has none, to round-off -- means full
    headroom against the declared transverse scale."""
    margin = local_gradient_smoothness_margin(
        np.array([0.0]), np.array([0.0]), transverse_scale_m=5e-6
    )
    assert margin[0] == 1.0


def test_local_gradient_smoothness_margin_degrades_with_curvature() -> None:
    small = local_gradient_smoothness_margin(
        np.array([1e3]), np.array([0.0]), transverse_scale_m=5e-6
    )
    large = local_gradient_smoothness_margin(
        np.array([1e8]), np.array([0.0]), transverse_scale_m=5e-6
    )
    assert small[0] > large[0]


def test_single_order_dominance_is_high_for_a_well_resolved_grating() -> None:
    """A patch spanning several periods of a pure single-tone grating should
    show most of its local spectral power in the requested order.

    ``0.815 = 0.903**2`` is the well-known fraction of a truncated sinusoid's
    energy a rectangular window's mainlobe carries in each of two separable
    axes (Parseval on a Dirichlet kernel); ``0.7`` leaves headroom below that
    theoretical value rather than pinning it exactly.
    """
    period_m = 2e-6
    pitch = (0.2e-6, 0.2e-6)
    n = 64
    x = (np.arange(n) - n // 2) * pitch[1]
    phase = 2.0 * np.pi * x / period_m
    transmission = np.exp(1j * np.tile(phase, (n, 1)))
    target_dx = GS_WAVELENGTH_M / period_m
    dominance, margin = single_order_dominance(
        transmission,
        sample_pitch_m=pitch,
        center_xy_m=(0.0, 0.0),
        patch_px=33,
        wavelength_m=GS_WAVELENGTH_M,
        target_dir_xy=(target_dx, 0.0),
    )
    assert dominance > 0.7
    assert margin > 0.4


def test_single_order_dominance_is_low_for_a_two_tone_surface() -> None:
    """A surface splitting power between two spatial frequencies must show
    lower dominance in either single requested order than the pure grating
    above -- the model boundary this predicate exists to detect."""
    period_a, period_b = 2e-6, 3e-6
    pitch = (0.2e-6, 0.2e-6)
    n = 64
    x = (np.arange(n) - n // 2) * pitch[1]
    phase_a = np.exp(1j * 2.0 * np.pi * x / period_a)
    phase_b = np.exp(1j * 2.0 * np.pi * x / period_b)
    transmission = np.tile(phase_a + phase_b, (n, 1))
    target_dx = GS_WAVELENGTH_M / period_a
    dominance, _ = single_order_dominance(
        transmission,
        sample_pitch_m=pitch,
        center_xy_m=(0.0, 0.0),
        patch_px=33,
        wavelength_m=GS_WAVELENGTH_M,
        target_dir_xy=(target_dx, 0.0),
    )
    assert dominance < 0.9


# --------------------------------------------------------------------------- #
# The model boundary against FULL_FIELD (CHE-143 Stage 5): agreement in the
# smooth limit, disagreement outside it. Two cases, not a sweep -- M2.10 owns
# the characterization of where the boundary sits.
# --------------------------------------------------------------------------- #


def test_generalized_snell_agrees_with_full_field_for_a_smooth_blazed_grating() -> None:
    """The smooth limit: a coarse, well-sampled linear grating. Both models
    see one order, and the two outgoing directions should agree closely."""
    period_m = 10e-6
    pitch = (0.5e-6, 0.5e-6)
    n = 128
    x = (np.arange(n) - n // 2) * pitch[1]
    phase = 2.0 * np.pi * x / period_m
    transmission = np.exp(1j * np.tile(phase, (n, 1)))
    surface = DiffractiveSurface(transmission=transmission, sample_pitch_m=pitch, plane=GS_PLANE)

    bundle = _gs_bundle([[0.0, 0.0, 1.0]], positions=[[0.0, 0.0, 0.0]])
    snell_result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=65),
    )
    assert snell_result.diagnostics["single_order_dominance"] > 0.7

    # The same single incident ray, through FULL_FIELD: one ray accumulated
    # onto the plane forms a (weak) plane-wave field, and the mode whose
    # amplitude is largest in the enumerated outgoing spectrum is the dominant
    # diffracted order that route resolved -- the same physical question
    # GENERALIZED_SNELL answered directly. Two different incident rays would
    # interfere with each other in one accumulated field, which is not the
    # same question, so this stays a one-ray comparison.
    full_field_result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None),
    )
    outgoing_dx = full_field_result.outgoing.directions[:, 0]
    weights = np.abs(full_field_result.outgoing.amplitude)
    dominant_dx = float(outgoing_dx[np.argmax(weights)])
    snell_dx = float(snell_result.outgoing.directions[0, 0])
    assert math.isclose(snell_dx, dominant_dx, abs_tol=5e-3)


def test_generalized_snell_disagrees_outside_the_smooth_limit() -> None:
    """Outside the smooth limit -- a surface splitting power across two
    orders -- GENERALIZED_SNELL reports low dominance where FULL_FIELD, which
    forms the whole field, shows two comparable peaks. The two models' claims
    about "the" outgoing ray disagree, which is the boundary this pairing is
    for, not a bug in either. The two periods are far enough apart that the
    64-sample grid's own angular resolution (``lambda / (n pitch)``) actually
    separates them -- two orders closer than that would merge into what a
    coarse FULL_FIELD grid reads as one peak, which would test the grid's
    resolution rather than the claim under test."""
    pitch = (0.2e-6, 0.2e-6)
    n = 64
    x = (np.arange(n) - n // 2) * pitch[1]
    period_a, period_b = 2e-6, 3e-6
    transmission = np.tile(
        0.5 * np.exp(1j * 2.0 * np.pi * x / period_a) + 0.5 * np.exp(1j * 2.0 * np.pi * x / period_b),
        (n, 1),
    )
    surface = DiffractiveSurface(transmission=transmission, sample_pitch_m=pitch, plane=GS_PLANE)
    bundle = _gs_bundle([[0.0, 0.0, 1.0]])
    snell_result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(order=1, patch_px=33),
    )
    assert snell_result.diagnostics["single_order_dominance"] < 0.7

    full_field_result = diffractive_interaction(
        bundle,
        surface,
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None),
    )
    weights = np.abs(full_field_result.outgoing.amplitude)
    order_indices = np.argsort(weights)[::-1]
    top_two = weights[order_indices[:2]]
    # Two comparable peaks -- neither dominates -- is the FULL_FIELD-side
    # signature of the same "not a single order" condition GENERALIZED_SNELL's
    # dominance margin reports directly.
    assert top_two[1] / top_two[0] > 0.5
