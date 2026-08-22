"""The registry may not claim a device or dtype the capability model does not (CHE-61).

PB4b's rule is that the executable capability model in ``core/capabilities.py``
is the source of truth and the registry YAML is its reflection, updated only
after the executable tests pass. A rule with no test is a preference, so this is
the test.

It is deliberately an *equality* check per component rather than a subset check
in one direction. Both directions are failures, and they fail differently:

* registry wider than the capability model = a claim nothing has executed, which
  is what CHE-55's ``devices: [cpu, gpu, tpu]`` entries were and what PB4b exists
  to stop;
* registry narrower = a validated capability the graph planner will refuse to
  use, so the work of validating it was wasted.

Note what this does *not* check: that the capability model is itself true. That
is the job of ``tests/test_precision_execution_matrix.py`` (host) and
``tests/test_precision_gpu_pipeline.py`` (device), which execute the claims. This
file only keeps the two declarations from drifting apart.
"""

from __future__ import annotations

import pathlib

import pytest

from core.capabilities import (
    COMPONENT_CAPABILITIES,
    capability_matrix,
)
from core.precision import DeviceKind, DType
from core.specs import Device, Maturity

pytestmark = pytest.mark.coupler

#: Every declared component. The list used to be scoped to "the components PB4b
#: owns", which quietly exempted the FMMAX/FDTDX/JAX-FEM entries -- and those
#: were exactly the ones claiming `devices: [cpu, gpu, tpu]` with nothing
#: executed. CHE-87 deleted them, so the exemption has nothing left to protect
#: and the registry is now asserted to contain no entry outside this mapping
#: (see `test_registry_declares_no_component_without_a_capability` below).
_OWNED = sorted(COMPONENT_CAPABILITIES)


def _spec_for(registry, component: str):
    if component.startswith("C_"):
        return registry.couplers[component]
    return registry.models[component]


@pytest.mark.parametrize("component", _OWNED)
def test_declared_devices_match_the_capability_model(registry, component):
    spec = _spec_for(registry, component)
    capability = COMPONENT_CAPABILITIES[component]
    expected = {kind.to_spec_device_name() for kind in capability.devices}
    declared = {device.value for device in spec.devices}
    assert declared == expected, (
        f"{component}: registry declares devices {sorted(declared)} but the "
        f"capability model validates {sorted(expected)}. The capability model is "
        "the source of truth; update the registry from it, and only after the "
        "executable tests for the new device pass."
    )


@pytest.mark.parametrize("component", _OWNED)
def test_declared_dtypes_match_the_accepted_input_set(registry, component):
    """Registry ``dtypes`` means *accepted*, not *native* and not *ingestible*.

    Chromatix is the case that makes the distinction load-bearing: it will
    physically swallow a complex128 array and return complex64, so an
    "ingestible" reading would list complex128 and thereby claim a precision the
    package cannot compute in. That dtype lives in ``lossy_input_dtypes`` and is
    asserted to be absent here.
    """
    spec = _spec_for(registry, component)
    capability = COMPONENT_CAPABILITIES[component]
    expected = {str(dtype) for dtype in capability.accepted_input_dtypes}
    declared = set(spec.dtypes)
    assert declared == expected, (
        f"{component}: registry declares dtypes {sorted(declared)} but the "
        f"capability model accepts {sorted(expected)}."
    )
    lossy = {str(dtype) for dtype in capability.lossy_input_dtypes}
    assert declared.isdisjoint(lossy), (
        f"{component}: registry lists {sorted(declared & lossy)}, which the "
        "capability model marks as ingestible only through a recorded lossy "
        "conversion. Listing it here would advertise support for a precision the "
        "component cannot compute in."
    )


def test_no_component_claims_float16_anywhere():
    """float16 appears in no registry entry, and the reason is per-component.

    Optiland has no float16 mode at all; the couplers promote a float16 input to
    float32 and compute there, which is a promotion and not support. Both are
    reasons to leave it out, and neither is "we did not get round to it".
    """
    for capability in COMPONENT_CAPABILITIES.values():
        assert DType.FLOAT16 not in capability.accepted_input_dtypes, capability.component
        assert DType.FLOAT16 not in capability.native_compute_dtypes, capability.component


def test_the_documented_capability_table_matches_the_generated_one():
    """The doc's table is generated output, so it must still equal the generator.

    A capability matrix pasted into prose is a snapshot, and a snapshot of a
    capability is exactly what PB4b forbids everywhere else. Rendering it the same
    way the doc does and comparing row by row is what makes the documented table
    an assertion rather than a claim.
    """
    doc = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "precision"
        / "precision_device_policy.md"
    ).read_text()

    for row in capability_matrix():
        component = row["component"]
        rendered = [
            line
            for line in doc.splitlines()
            if line.startswith("| " + component) or line.startswith("| " + component + " ")
        ]
        assert rendered, f"{component} has no row in the documented capability table"
        cells = [cell.strip() for cell in rendered[0].strip("|").split("|")]
        # component, devices, precisions, input accepted, native, output, lossy,
        # namespaces, compute floor -- in the order the generator emits them.
        expected = [
            component,
            ", ".join(row["devices"]),
            ", ".join(row["precisions"]),
            ", ".join(row["accepted_input_dtypes"]),
            ", ".join(row["native_compute_dtypes"]),
            ", ".join(row["output_dtypes"]),
            ", ".join(row["lossy_input_dtypes"]) or "--",
            ", ".join(row["namespaces"]),
            row["minimum_compute_precision"],
        ]
        assert cells == expected, (
            f"{component}: the documented table row is stale.\n"
            f"  doc:       {cells}\n"
            f"  generated: {expected}\n"
            "Regenerate with benchmarks/probes/precision/capability_table.py."
        )


def test_registry_declares_no_component_without_a_capability(registry):
    """Every shipped registry entry has an executable capability declaration.

    This is the check the old ``_OWNED`` scoping made impossible. Declaring a
    component in the registry is what makes it addressable by a graph, so an
    entry with no capability declaration is a component the planner can select
    and the precision bridge cannot reason about -- which is strictly worse than
    the component not existing.

    The remedy is deliberately asymmetric. If the component is real, add its
    declaration to ``core/capabilities.py`` **with the probe evidence behind
    it**. If it is not, delete the registry entry. Adding a placeholder
    declaration to silence this test would manufacture the unvalidated claim the
    whole file exists to prevent.
    """
    declared = set(registry.models) | set(registry.couplers)
    undeclared = declared - set(COMPONENT_CAPABILITIES)
    assert not undeclared, (
        f"registry entries with no capability declaration: {sorted(undeclared)}. "
        "Either add one to core/capabilities.py backed by an executable probe, "
        "or delete the registry entry. Do not add a placeholder: a declaration "
        "with no probe behind it is the unvalidated claim this file forbids."
    )


def test_no_registry_entry_is_still_experimental(registry):
    """`maturity` has to carry information, and for 17 entries it carried none.

    All 17 entries were ``experimental`` before CHE-87, which makes the field a
    constant rather than a classification. The four that survived were set from
    what is actually measured: ``characterized`` for all of them, because each
    has an analytic or independent-implementation check and none has a certified
    gradient.

    A new entry arriving as ``experimental`` is not forbidden in principle -- it
    is forbidden here because the field only stays meaningful if setting it is a
    decision. If something genuinely is experimental, say so in this test with
    the reason.
    """
    for name, spec in {**registry.models, **registry.couplers}.items():
        assert spec.maturity != Maturity.EXPERIMENTAL, (
            f"{name} is still `maturity: experimental`. Set it from what has "
            "actually been measured -- characterized (checked against an "
            "analytic case, conservation law, convergence study or independent "
            "implementation), validated (that, plus a certified gradient), or "
            "deprecated."
        )


def test_no_component_claims_tpu():
    """`tpu` was in three CHE-55 entries and nothing has ever executed there."""
    assert all(
        DeviceKind.CUDA in capability.devices or DeviceKind.CPU in capability.devices
        for capability in COMPONENT_CAPABILITIES.values()
    )
    assert Device.TPU.value not in {
        kind.to_spec_device_name() for capability in COMPONENT_CAPABILITIES.values()
        for kind in capability.devices
    }
