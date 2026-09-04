"""Subject 5: the ray-to-wave chain on CUDA, gated against the host.

CHE-247 (T3) acceptance criteria 2 and 4. The record
(`benchmarks/systems/records/B-RAY-WAVE-CHAIN-*.json`) is where the chain's
placement is *measured*; this is where the two claims that need a **tolerance**
and an **inventory** are decided, because both belong here rather than there:

* `cells.tolerance_for` is this package's single tolerance derivation and
  `benchmarks/` deliberately cannot import it, so AC-2's comparison is gated
  here and the benchmark records the number as `diagnostic`;
* AC-4 asks that any remaining device-to-host round trip be named with its
  location. The benchmark names it in prose and times a buffer of its shape; a
  count of what actually crossed needs `backends.optiland.rays._host` to be
  instrumented, which a benchmark in `benchmarks/systems/` may not do -- it
  imports no backend, by an AST gate.

The chain itself is imported from the benchmark rather than restated. One
definition, two consumers: a second copy here would be a second optical system
the moment either was edited, which is the failure `AGENTS.md` puts first.

Nothing here is an oracle. The two legs are the same repository code, so their
agreement is characterization; what it can settle is whether the *device* changed
the answer, which is the whole question.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# `benchmarks/` is not an installed package -- `pyproject.toml` sets
# `package-dir = {"" = "src"}`, so only `src/` is on the path -- and this module
# lives in a package (`tests/parity/__init__.py`), which makes pytest insert
# `tests/` rather than the repository root. Inserted explicitly here, at the one
# module that needs it, rather than in `tests/conftest.py`, where it would give
# every test in the tree an import it has no business having.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.systems import b_ray_wave_chain as chain  # noqa: E402
from numerics import (  # noqa: E402
    ArrayNamespace,
    DeviceKind,
    DevicePlacement,
    DType,
    array_state,
)
from parity.cells import Cell, tolerance_for  # noqa: E402
from parity.conftest import unavailable_reason, verify_placement  # noqa: E402

#: The cell the chain's CUDA leg runs in. Constructed rather than taken from
#: `cells_for`, and the reason is a real distinction: this is not a cell set
#: derived from a component's capability pack, it is the one state the chain's
#: device leg can be in at all -- `jax_enable_x64` is off, so complex64 on the
#: device is the only option, and `backends.optiland.rays.trace_exit_state` is
#: what decides it. Held to that function in `test_optiland_exit_parity.py`.
CUDA_CELL = Cell(
    namespace=ArrayNamespace.JAX,
    device=DevicePlacement.parse(chain.CUDA_DEVICE),
    dtype=DType.COMPLEX64,
)

def _skip_unless_cuda() -> None:
    reason = unavailable_reason(CUDA_CELL)
    if reason is not None:
        pytest.skip(reason)


def _cuda_bundle() -> Any:
    """The chain's bundle after `O_PROPAGATE_RAYS`, from the benchmark's own plan."""
    from runtime import Executor

    with Executor() as executor:
        record = executor.execute(chain._plan(chain.CUDA_DEVICE)[:2])
        assert record.status == "completed", [
            (node.operation_id, node.status, node.diagnostics) for node in record.nodes
        ]
        return executor.result


def _tail(rays: Any) -> Any:
    """The chain's last three nodes on `rays`, as host float64 intensity."""
    from couplers import ray_to_scalar
    from measurements import psf
    from numerics import to_namespace
    from operators import complex_transmission

    field, _ = ray_to_scalar(
        rays,
        grid_shape=chain.GRID_SHAPE,
        sample_pitch_m=chain.SAMPLE_PITCH_M,
        grazing="band_limit",
    )
    result = psf(complex_transmission(field, amplitude=1.0), normalization="energy")
    host = to_namespace(result.intensity, namespace=ArrayNamespace.NUMPY)
    return np.asarray(host, dtype=np.float64)


@pytest.mark.gpu
def test_the_device_leg_agrees_with_the_host_at_the_same_precision() -> None:
    """AC-2, comparing the one thing a device comparison can compare.

    The comparison is against a host copy of the **same bundle with every dtype
    preserved**, not against the chain's `device="cpu"` leg, and that is not a
    convenience. The cpu leg runs the coupler at FP64: the Optiland host exit
    leaves `optical_path_m` and `measure_weight` at the host float64 they were
    declared in -- CHE-245 keeps that asymmetry deliberately, because it is what
    makes the host path bit-identical -- and
    `couplers.ray_to_scalar._compute_precision` takes the *maximum* precision over
    the arrays a bundle carries. The CUDA leg has no such option, since JAX cannot
    represent float64 here. So a CUDA-against-cpu number is a precision comparison
    wearing a device comparison's clothes, and the benchmark record measures the
    decomposition: device 3.7e-6, precision 3.5e-4, i.e. the naive number is ~99 %
    precision.

    Gating the naive number against `tolerance_for` would therefore fail by ~50x
    for a reason that is not the device, and the fix would look like a tolerance
    problem. `test_the_host_leg_reports_its_precision_drift` is what keeps that
    asymmetry from disappearing quietly instead.
    """
    _skip_unless_cuda()
    from numerics import to_namespace

    device_rays = _cuda_bundle()
    verify_placement(CUDA_CELL, device_rays.amplitude)

    host = DevicePlacement.parse("cpu")

    def moved(value: Any) -> Any:
        if value is None:
            return None
        return to_namespace(
            value,
            namespace=ArrayNamespace.NUMPY,
            device=host,
            dtype=array_state(value).dtype,
        )

    host_rays = dataclasses.replace(
        device_rays,
        positions_m=moved(device_rays.positions_m),
        directions=moved(device_rays.directions),
        amplitude=moved(device_rays.amplitude),
        optical_path_m=moved(device_rays.optical_path_m),
        measure_weight=moved(device_rays.measure_weight),
    )
    assert array_state(host_rays.positions_m).namespace is ArrayNamespace.NUMPY
    assert array_state(host_rays.positions_m).dtype is array_state(device_rays.positions_m).dtype

    device_intensity = _tail(device_rays)
    host_intensity = _tail(host_rays)

    # The ramp sum contracts over the ray count and is an einsum, so those are
    # the two facts `tolerance_for` needs; the squaring factor is the PSF's.
    # `squared=True` and not `2.0 * tolerance_for(...)`: the intensity's relative
    # error is twice the amplitude's, and that factor belongs inside the one
    # derivation point rather than beside the assertion, where it would be a
    # tolerance literal at a comparison site.
    bound = tolerance_for(
        CUDA_CELL, accumulation_length=int(device_rays.count), matmul=True, squared=True
    )
    # The benchmark's own metric, imported rather than re-implemented, so "the
    # record's number and the gated number are the same quantity" is structural.
    deviation = chain._peak_relative(device_intensity, host_intensity)
    assert deviation <= bound, (
        f"the CUDA leg differs from the same-precision host leg by {deviation:.3e}, past the "
        f"{bound:.3e} derived for a {device_rays.count}-term squared complex64 contraction. "
        "The margin here is about 4x, not the ~25x cells.py advertises for its own "
        "subject, because the measurand is max-abs-relative-to-peak of an intensity rather "
        "than a relative L2 of an amplitude. It still discriminates what it exists to "
        "catch: a lost matmul_precision_kwargs is ~2.6e-4, i.e. ~19x above this bound. Do "
        "not widen it -- the candidates at this margin are TF32 or a node that stopped "
        "being device-resident, and both are findings"
    )


@pytest.mark.gpu
def test_the_host_leg_reports_its_precision_drift() -> None:
    """The asymmetry above, held to being *reported* rather than merely known.

    The chain's `device="cpu"` leg asks for `fp32` and computes at `fp64`, and
    `NodeRecord.placement_disagreement` is what says so. That is the record doing
    its job, not a defect in the record -- but it is also the only thing standing
    between "the host leg is a different precision" and a future reader treating
    the CUDA-against-cpu number as a device measurement. So it is asserted:
    the drift is on `precision`, it is **not** on `device`, and it appears on the
    coupler, which is where `_compute_precision` takes its maximum.

    Marked `gpu` although it runs the host leg only, because it is a statement
    about the pair and belongs with the test above; running it in the default
    suite would give a claim about a comparison that suite never makes.
    """
    _skip_unless_cuda()
    from runtime import Executor

    with Executor() as executor:
        record = executor.execute(chain._plan("cpu"))
    assert record.status == "completed", [
        (node.operation_id, node.status, node.diagnostics) for node in record.nodes
    ]

    drifted = {
        node.operation_id: node.placement_disagreement
        for node in record.nodes
        if node.placement_disagreement
    }
    assert drifted, (
        "the host leg reported no placement disagreement at all. Either the precision "
        "asymmetry CHE-245 recorded has been resolved -- in which case the benchmark's "
        "chain_parity decomposition and this test are both stale -- or the record stopped "
        "observing precision, which is worse"
    )
    for operation_id, keys in drifted.items():
        assert tuple(keys) == ("precision",), (
            f"{operation_id} reports drift on {list(keys)}. Only `precision` is expected on "
            "the host leg; a `device` disagreement there would mean a cpu request did not "
            "land on the cpu"
        )
    assert "C_RAY_TO_SCALAR" in drifted, (
        "the coupler is where _compute_precision takes the maximum over a bundle's dtypes, "
        f"so it is where the FP32 request becomes FP64. Drift was reported at {list(drifted)}"
    )


@pytest.mark.gpu
def test_only_the_first_node_traverses_the_host_in_bulk() -> None:
    """AC-4, as an inventory rather than as prose.

    The benchmark names the one remaining bulk host traversal --
    `backends.optiland.rays.to_ray_bundle`'s nine-column read, kept by CHE-245
    because the declaration computed from those columns is host float64 by a
    recorded scientific decision -- and times a buffer of its shape. What it
    cannot do is *count* what crossed, because a benchmark in
    `benchmarks/systems/` imports no backend.

    The count is what makes the claim falsifiable, and it is taken twice: over
    node 1 alone, and over the whole five-node chain. Equal counts is the
    assertion, because it says nodes 2-5 add none -- which is stronger than any
    absolute number, and is exactly the failure mode CHE-247's Why describes ("a
    chain can be device-resident at every node and still round-trip through the
    host between two of them").

    Small reads are excluded and must be: `_scalar` reads the image plane's z and
    the refractive index, and `require_launch_surface` reads the surface
    positions. The gate is on bulk.

    **What is watched is the single funnel, and choosing it took a measurement.**
    Injecting a `rays._host` call into `operators.propagate_rays` does not produce
    a quiet host copy, it *raises*: `optiland.backend.utils.to_numpy` refuses a
    `jaxlib ArrayImpl` outright with `Unsupported object type`. So spying `_host`
    alone left the count comparing node 1's reads to themselves -- nodes 2-5
    cannot reach `_host` at all -- and the injection was caught only by the
    completion assertion, which is a weaker claim than this test makes.
    `numerics.arrays.to_host_numpy` is no better: it has no caller in `src/`, and
    patching the module attribute is invisible to `from numerics import
    to_host_numpy`, which is this tree's dominant import style.

    `numerics.arrays._to_numpy` is the one function every host copy in the project
    passes through -- `to_host_numpy` calls it and so does `to_namespace`'s NumPy
    branch -- so that is what is patched, and the count then has teeth: a later
    node that reached for the host by *any* route increments it.
    """
    _skip_unless_cuda()
    from backends.optiland import rays as rays_module
    from numerics import arrays as arrays_module
    from runtime import Executor

    ray_count = int(_cuda_bundle().count)
    original_host = rays_module._host
    original_to_numpy = arrays_module._to_numpy

    def crossings(plan: tuple[Any, ...], monkeypatch: Any) -> list[int]:
        seen: list[int] = []

        def host_spy(value: object) -> object:
            out = original_host(value)
            seen.append(int(np.size(out)))
            return out

        def to_numpy_spy(value: object, source: Any) -> Any:
            out = original_to_numpy(value, source)
            seen.append(int(np.size(out)))
            return out

        monkeypatch.setattr(rays_module, "_host", host_spy)
        monkeypatch.setattr(arrays_module, "_to_numpy", to_numpy_spy)
        with Executor() as executor:
            record = executor.execute(plan)
        assert record.status == "completed", [
            (node.operation_id, node.status, node.diagnostics) for node in record.nodes
        ]
        monkeypatch.undo()
        return [size for size in seen if size >= ray_count]

    patcher = pytest.MonkeyPatch()
    try:
        first_node = crossings(chain._plan(chain.CUDA_DEVICE)[:1], patcher)
        whole_chain = crossings(chain._plan(chain.CUDA_DEVICE), patcher)
    finally:
        patcher.undo()

    assert first_node, (
        "node 1 crossed the host with nothing bulk, so this gate is vacuous. "
        "to_ray_bundle reads nine columns to the host by design"
    )
    assert len(whole_chain) == len(first_node), (
        f"the whole chain made {len(whole_chain)} bulk host read(s) of sizes {whole_chain} "
        f"where node 1 alone makes {len(first_node)} of sizes {first_node}. Every read past "
        "the first node is a device-resident chain round-tripping through the host between "
        "two nodes, which is the failure CHE-247 exists to catch"
    )


@pytest.mark.gpu
def test_the_chains_device_cell_is_the_one_the_exit_state_declares() -> None:
    """`CUDA_CELL` above is not a hand-written triple; this is what pins it.

    `parity/cells.py`'s rule is that a cell set is derived from a declaration and
    never listed. This module constructs one cell, because the chain has exactly
    one device state available to it, and the declaration it must equal is
    `backends.optiland.rays.trace_exit_state` -- the production function CHE-245
    added -- widened to the complex counterpart the coupler emits.
    """
    _skip_unless_cuda()
    from backends.optiland.rays import trace_exit_state
    from numerics import Precision

    declared = trace_exit_state(
        device=DevicePlacement.parse(chain.CUDA_DEVICE), precision=Precision.FP32
    )
    assert declared.namespace is CUDA_CELL.namespace
    assert declared.device.kind is DeviceKind.CUDA
    assert declared.dtype.precision.complex_dtype is CUDA_CELL.dtype
