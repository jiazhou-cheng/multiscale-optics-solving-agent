"""Shared pytest fixtures and helpers for the test suite."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "solvers"


@functools.lru_cache(maxsize=1)
def cuda_unavailable_reason() -> str | None:
    """Why a CUDA device is not usable here, or ``None`` if one is.

    Called once per session and only when ``gpu``-marked tests were actually
    collected, because importing torch/jax costs seconds and the default CPU
    container has no GPU to find.

    Both frameworks are consulted because a container can be half-enabled in
    either direction: the default ``agent_solver`` image installs CPU-only
    builds of *both* (see docker/Dockerfile), while ``agent_solver_gpu``
    installs CUDA builds of both. Requiring both keeps a partially-provisioned
    image from reporting itself as GPU-capable and then failing mid-test.
    """
    reasons: list[str] = []

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is pinned in both images
        reasons.append("torch is not importable")
    else:
        if torch.version.cuda is None or "+cpu" in torch.__version__:
            reasons.append(f"torch is a CPU-only build ({torch.__version__})")
        elif not torch.cuda.is_available():
            reasons.append("torch.cuda.is_available() is False (no device attached)")

    try:
        import jax
    except ImportError:  # pragma: no cover - jax is pinned in both images
        reasons.append("jax is not importable")
    else:
        if not any(device.platform == "gpu" for device in jax.devices()):
            reasons.append(f"jax sees no gpu device (backend={jax.default_backend()!r})")

    if not reasons:
        return None
    return (
        "; ".join(reasons)
        + ". Run GPU tests in the CUDA container: `./run.sh --gpu pytest -q -m gpu`"
        + " (build it first with `./run.sh --gpu --rebuild pytest -q -m gpu`)."
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``gpu``-marked tests unless this session can actually run them.

    Two conditions must hold: the session must be dedicated to GPU tests, and a
    CUDA device must be reachable. Otherwise the GPU tests skip, which is what
    lets every documented tier command stay green unchanged -- Tier A does not
    filter out ``gpu``, so those tests are collected there and simply skip, on a
    CPU-only host *and* inside the GPU image.

    ``trylast=True`` matters: this hook must observe ``items`` after pytest's own
    ``-m`` deselection, because both decisions below depend on what will really
    run rather than on what happened to be collected.
    """
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return

    # On the GPU image, JAX puts *every* computation in the process on the GPU --
    # there is no per-test backend. So a mixed selection silently moves the
    # non-GPU tests onto a backend their tolerances were never derived for.
    # Measured on the GPU image (CHE-60): running Tier A there moved Chromatix
    # onto the GPU and broke two tolerance-sensitive tests in
    # test_m3_pupil_to_focus.py (archived by CHE-67; the measurement stands).
    # That is not evidence of a bug in Chromatix or in those tests -- they were
    # written against CPU float32 results -- but until dtype-appropriate GPU
    # tolerances exist, the established CPU results stay authoritative.
    #
    # So GPU tests run *only* as a dedicated selection and skip whenever anything
    # else is selected alongside them. This is also what keeps every documented
    # tier command green unchanged on both images: Tier A does not filter out
    # `gpu`, so those tests are collected there and simply skip.
    #
    # CHE-72 note: this guard used to be justified by SAX's klujax dependency
    # pinning `jax_platform_name='cpu'` process-globally at import, which the
    # harness then had to undo. SAX is gone and nothing is repaired here any
    # more; the reason above is what is left, and it is sufficient on its own.
    other_items = len(items) - len(gpu_items)
    if other_items:
        skip_not_dedicated = pytest.mark.skip(
            reason=(
                f"gpu tests need a dedicated session; {other_items} non-gpu test(s) are "
                "also selected. On the GPU image every test in the process computes on "
                "the GPU, and the non-gpu tolerances were derived on the CPU. "
                "Run: `MOA_GPUS=device=6 make test-gpu`. Not a bare "
                "`./run.sh --gpu pytest -q -m gpu`: since CHE-140 that inherits "
                "`-n 8 --dist loadfile` from addopts, and eight workers on one "
                "device is a resource-policy violation before it is a test failure."
            )
        )
        for item in gpu_items:
            item.add_marker(skip_not_dedicated)
        return

    reason = cuda_unavailable_reason()
    if reason is None:
        return

    skip_gpu = pytest.mark.skip(reason=f"no CUDA device available: {reason}")
    for item in gpu_items:
        item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.from_package()


#: Recorded probe evidence, moved out of `knowledge/` by CHE-92: that directory
#: is what gets disclosed to an agent, and a recorded array is not context.
#:
#: It lives under `benchmarks/probes/` rather than a top-level `verification/`,
#: which is where CHE-92 originally aimed it. Phase 5 had already created a
#: `src/verification/` package, and a repository-root directory of the same name
#: shadows it for anything run from the root -- `tests/test_flat_layout.py`
#: caught that. `benchmarks/probes/records/` is where the other recorded
#: evidence already was, so this is also the less surprising home.
RECORDS_ROOT = ROOT / "benchmarks" / "probes" / "records"


def load_probe_expected(component: str, probe: str) -> dict:
    """The recorded ground truth for one probe.

    Captured by running ``benchmarks/probes/<component>/<probe>.py`` against
    the real solver or coupler. Tests compare against it rather than re-deriving
    an oracle, which is the point: a test that re-derives its own reference is
    checking itself.
    """
    path = RECORDS_ROOT / component / f"{probe}.json"
    return json.loads(path.read_text())


#: Kept as a distinct name because the *claim* differs: a solver probe
#: characterizes a package, a coupler probe characterizes a transformation this
#: repository owns. They now read from one directory, and that is fine -- what
#: must not merge is the two kinds of claim.
load_coupler_probe_expected = load_probe_expected
