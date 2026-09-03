"""Every `capabilities` citation in the catalog resolves to a measured record.

CHE-223 (R03.6), acceptance criterion 6. This is the half that `__post_init__` used
to do eagerly and no longer does: constructing a descriptor now validates the
**shape** of a component id and nothing else, so a well-formed id with no record
behind it is a claim nobody has measured and *this* is what catches it.

Separate from `tests/operations/test_descriptors.py` on purpose, and separate from
descriptor construction. The whole point of moving the check here is that
`operations/` no longer needs the concrete table to be importable -- see
`operations/descriptors.py` on why that asymmetry mattered -- so the module that
resolves the citations has to be a module that is allowed to read the pack.

It is the exact counterpart of `test_catalog_resolution.py`, which does the same
for `implementation`: both fields are references, both are checked for shape at
construction and for resolution here.
"""

from __future__ import annotations

import pytest

from numerics import capability_record_ids, load_capabilities
from operations import CATALOG

#: The records that cite a component, as `(operation_id, component_id)`.
CITATIONS = [
    (record.operation_id, record.capabilities)
    for record in CATALOG
    if record.capabilities is not None
]


def test_the_catalog_cites_something() -> None:
    """The meta-check: a parametrization over no citations proves nothing."""
    assert CITATIONS, "no catalog record cites a capability, so the tests below are vacuous"
    assert len(CITATIONS) == 7, [operation for operation, _ in CITATIONS]


@pytest.mark.parametrize(
    ("operation_id", "component"), CITATIONS, ids=[case[0] for case in CITATIONS]
)
def test_every_capability_citation_resolves_to_a_record(
    operation_id: str, component: str
) -> None:
    """Criterion 6, per citation. A stale or invented id fails here."""
    assert component in capability_record_ids(), (
        f"{operation_id} cites {component!r}, which has no record under "
        f"knowledge/capabilities/. Catalogued components: {list(capability_record_ids())}"
    )
    capability = load_capabilities(component)
    assert capability.component == component
    assert capability.probe.startswith("benchmarks/probes/")


def test_only_the_operations_that_drive_a_backend_cite_a_record() -> None:
    """Which citations exist, and the honest `None` for everything else.

    A coupler runs in whatever namespace the field it was handed carries, so it has
    no measured device/dtype row of its own -- citing the chromatix record would
    claim a measurement taken about something else. `None` is the citation, not a
    missing one, and `operations/descriptors.py` says so.
    """
    cited = dict(CITATIONS)
    assert cited == {
        "SO_RAY_LAUNCH_TRACE": "M_RAY_OPTILAND",
        "O_RAY_TRACE": "M_RAY_OPTILAND",
        # CHE-226 (R16). The native spot analysis executes in the same measured row:
        # it is the same package's sequential trace with a reduction on the end, and
        # the probe measured the package's device and dtype behaviour rather than one
        # semantic operation. `M_SPOT_DIAGRAM` cites nothing, correctly -- it drives
        # no backend and has no measured row of its own.
        "SOM_SPOT_DIAGRAM": "M_RAY_OPTILAND",
        # CHE-236 (R16.1). The native PSF analysis: the same package's sequential
        # trace with a diffraction propagation on the end, in the same measured
        # row. **The row does not mention that this operation needs numba at all**,
        # and it needs it for every method rather than only for Huygens: measured,
        # `optiland/psf/__init__.py` imports `huygens_fresnel_strategies`, which
        # does a module-level `from numba import njit, prange`. Probed working on
        # both namespaces at cpu/fp32 and cpu/fp64 (numba 0.66.0, torch
        # 2.13.0+cpu, both pinned in docker/requirements.txt), so the row's
        # device/precision claims are not falsified -- what is missing is a
        # dependency the pack has no field for. Recorded on the ticket rather than
        # papered over by widening the pack without a probe.
        "SOM_PSF": "M_RAY_OPTILAND",
        "O_ASM_PROPAGATE": "M_WAVE_CHROMATIX",
        # CHE-228 (R06.11). The Fresnel kernel runs on the same pinned build, in the
        # same complex64-only storage the row measured, so it cites the same record.
        # The measurement is about the package's device and dtype behaviour, and the
        # paraxial substitution changes neither.
        "O_FRESNEL_PROPAGATE": "M_WAVE_CHROMATIX",
        "O_FOCAL_PLANE_TRANSFORM": "M_WAVE_CHROMATIX",
    }
    for record in CATALOG:
        if record.capabilities is None:
            assert not record.implementation.startswith("backends."), record.operation_id


def test_several_descriptors_may_cite_one_record() -> None:
    """The pack rule, from the catalog's side.

    `SO_RAY_LAUNCH_TRACE` and `O_RAY_TRACE` both cite `M_RAY_OPTILAND` because the probe
    measured the *package's* device and dtype behaviour, not one semantic operation.
    Duplicating a component row per descriptor is the second source the knowledge
    pack removes, so this is pinned as intended rather than tolerated.

    The chromatix count went 3 -> 2 on CHE-224 (R15.1), which merged
    `S_WAVE_CHROMATIX` into `O_ASM_PROPAGATE`, and back to 3 on CHE-228 (R06.11)
    with `O_FRESNEL_PROPAGATE`; the optiland count went 2 -> 3 on CHE-226 (R16) with
    `SOM_SPOT_DIAGRAM`. Note that the *record* is untouched in every case: what
    changed is how many descriptors cite it, which is exactly the number this test
    exists to leave free.
    """
    per_component: dict[str, list[str]] = {}
    for operation_id, component in CITATIONS:
        per_component.setdefault(component, []).append(operation_id)
    assert sorted(per_component["M_RAY_OPTILAND"]) == [
        "O_RAY_TRACE",
        "SOM_PSF",
        "SOM_SPOT_DIAGRAM",
        "SO_RAY_LAUNCH_TRACE",
    ]
    assert len(per_component["M_WAVE_CHROMATIX"]) == 3
    # And there is exactly one record per component, not one per citation.
    assert len(capability_record_ids()) == 2


def test_an_unresolvable_citation_would_be_caught() -> None:
    """The falsifier, since the assertions above are all positive.

    Rebuilt on a copy of a real record so the test is known to discriminate: a
    well-formed id is accepted at construction -- that is the point of CHE-223 --
    and it is the resolution step here that refuses it.
    """
    import dataclasses

    record = next(r for r in CATALOG if r.capabilities == "M_RAY_OPTILAND")
    invented = dataclasses.replace(record, capabilities="M_RAY_INVENTED")
    assert invented.capabilities == "M_RAY_INVENTED", "construction must still accept it"
    assert invented.capabilities not in capability_record_ids()
    with pytest.raises(ValueError) as caught:
        load_capabilities(invented.capabilities)
    assert caught.value.code == "UNKNOWN_COMPONENT"
