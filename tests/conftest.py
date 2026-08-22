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
                "Run: `./run.sh --gpu pytest -q -m gpu`"
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


def load_probe_expected(solver: str, probe: str) -> dict:
    """Load the recorded ground truth from knowledge/solvers/<solver>/expected/<probe>.json.

    This is evidence captured by running knowledge/solvers/<solver>/probes/<probe>.py
    against the real solver; adapter tests compare against it instead of
    re-deriving an oracle.
    """
    path = KNOWLEDGE_ROOT / solver / "expected" / f"{probe}.json"
    return json.loads(path.read_text())


COUPLER_KNOWLEDGE_ROOT = ROOT / "knowledge" / "couplers"


def load_coupler_probe_expected(coupler: str, probe: str) -> dict:
    """The coupler-side counterpart of :func:`load_probe_expected`.

    Coupler probes characterize a transformation rather than a package, so their
    evidence lives under ``knowledge/couplers/<coupler>/expected/`` beside the
    coupler card that cites it.
    """
    path = COUPLER_KNOWLEDGE_ROOT / coupler / "expected" / f"{probe}.json"
    return json.loads(path.read_text())
