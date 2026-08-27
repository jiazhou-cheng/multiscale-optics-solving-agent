"""What kind of thing each operation in this package is (CHE-142).

``src/couplers/`` was using one word -- coupler -- for three different kinds of
operation, so the package could not say which of the three a caller was asking
for. This module is the partition, as data:

============================== =================================================
role                           what changes
============================== =================================================
``REPRESENTATION_TRANSITION``  the description of the light. Rays become a field
                               or a field becomes rays; the physical content is
                               the same content, re-expressed and losing
                               whatever the target representation cannot hold.
``DIFFRACTIVE_INTERACTION``    the light itself, at a surface. Coherent rays in,
                               coherent rays out. It *contains* two
                               representation transitions and a transmission,
                               and that is its implementation rather than its
                               identity.
``PROPAGATION``                where the representation is evaluated. Neither
                               the representation nor the physical content
                               changes -- only the plane.
============================== =================================================

Why it has to be enforced rather than described. ``C_PLANAR_DOE_STEP`` and
``C_PATCH_WFT`` are both ``DIFFRACTIVE_INTERACTION``, and in the registry they
read as two unrelated physics claims -- while SI S10 says one is the shortcut for
the other. Meanwhile ``advance_bundle_to_plane`` sat inside the patch model,
which is where a reader would look for diffraction rather than for a plane
transfer. Both are the same mistake: the package had no vocabulary for the
distinction, so the distinction was invisible in exactly the places a caller
reads.

``tests/test_diffractive_interaction.py`` holds three things against this table:
that every public operation of the package has exactly one role, that the
registry's ``role`` field agrees with it, and that the two diffractive models
share an interaction identity while keeping their own capability rows.
"""

from __future__ import annotations

from core.specs import CouplerRole
from couplers.interaction import INTERACTION_ID, MODEL_COUPLER_IDS, DiffractiveModel

__all__ = [
    "COUPLER_ROLES",
    "INTERACTION_ID",
    "MODEL_COUPLER_IDS",
    "OPERATION_ROLES",
    "CouplerRole",
    "DiffractiveModel",
    "diffractive_models_of",
    "role_of_coupler",
    "role_of_operation",
]


#: Every public operation this package exposes, by role. Keyed by
#: ``module.callable`` so a reader can find it, and asserted exhaustive against
#: the package's own exports -- an operation added without a role fails the test
#: rather than defaulting to one.
OPERATION_ROLES: dict[str, CouplerRole] = {
    # --- representation transitions -------------------------------------
    "couplers.ray_to_wave.ray_to_wave": CouplerRole.REPRESENTATION_TRANSITION,
    "couplers.wave_to_ray.decompose": CouplerRole.REPRESENTATION_TRANSITION,
    "couplers.wave_to_ray.spectrum_to_rays": CouplerRole.REPRESENTATION_TRANSITION,
    "couplers.node.RayToWaveCoupler": CouplerRole.REPRESENTATION_TRANSITION,
    # --- one diffractive interaction, at three granularities -------------
    # `diffractive_interaction` is the entry point; the other three are the
    # per-model implementations it dispatches to, and they carry the same role
    # because they are the same physical operation computed differently.
    "couplers.interaction.diffractive_interaction": CouplerRole.DIFFRACTIVE_INTERACTION,
    "couplers.cascade.planar_doe_step": CouplerRole.DIFFRACTIVE_INTERACTION,
    "couplers.patch.patch_secondary_rays": CouplerRole.DIFFRACTIVE_INTERACTION,
    "couplers.generalized_snell.generalized_snell_step": CouplerRole.DIFFRACTIVE_INTERACTION,
    "couplers.doe_node.PlanarDoeStepCoupler": CouplerRole.DIFFRACTIVE_INTERACTION,
    "couplers.patch_node.PatchWftCoupler": CouplerRole.DIFFRACTIVE_INTERACTION,
    # --- propagation ------------------------------------------------------
    "couplers.propagation.advance_bundle_to_plane": CouplerRole.PROPAGATION,
}


#: The registry rows, by role. The source of truth is
#: ``src/registry/couplers.yaml``'s own ``role`` field -- this is the expectation
#: the test holds it to, so a row whose role is edited in one place and not the
#: other is a failure rather than a drift.
COUPLER_ROLES: dict[str, CouplerRole] = {
    "C_RAY_TO_WAVE": CouplerRole.REPRESENTATION_TRANSITION,
    "C_WAVE_TO_RAY": CouplerRole.REPRESENTATION_TRANSITION,
    "C_PLANAR_DOE_STEP": CouplerRole.DIFFRACTIVE_INTERACTION,
    "C_PATCH_WFT": CouplerRole.DIFFRACTIVE_INTERACTION,
    "C_GENERALIZED_SNELL": CouplerRole.DIFFRACTIVE_INTERACTION,
}


def role_of_operation(qualified_name: str) -> CouplerRole:
    """The role of one operation, or a failure naming what is classified.

    Refuses rather than defaulting. A default would silently classify a new
    representation transition as whatever the table's most common value happens
    to be, which is the ontology problem this module exists to fix.
    """
    try:
        return OPERATION_ROLES[qualified_name]
    except KeyError:
        raise KeyError(
            f"{qualified_name!r} has no declared role. Classified operations: "
            f"{sorted(OPERATION_ROLES)}. Add it to OPERATION_ROLES -- an "
            "operation in this package is a representation transition, a "
            "diffractive interaction, or a propagation, and which one it is is "
            "not inferable from its signature."
        ) from None


def role_of_coupler(coupler_id: str) -> CouplerRole:
    """The role of one registered coupler, or a failure naming what exists."""
    try:
        return COUPLER_ROLES[coupler_id]
    except KeyError:
        raise KeyError(
            f"no declared role for {coupler_id!r}; registered: "
            f"{sorted(COUPLER_ROLES)}"
        ) from None


def diffractive_models_of(coupler_id: str) -> tuple[DiffractiveModel, ...]:
    """Which diffractive model(s) a coupler row is the capability row for.

    Empty for a representation transition. The relation is many-to-one in
    principle and one-to-one today, which is why it is a tuple: a future model
    sharing a row would be a capability claim about two things at once and
    should be visible as such.
    """
    return tuple(
        model for model, cid in MODEL_COUPLER_IDS.items() if cid == coupler_id
    )
