"""Session-wide test configuration.

CHE-173 (R02.1) restores the `gpu`-marker gating that the greenfield deletion
removed along with the old suite -- `Makefile:test-gpu` names it as owed by
whichever ticket writes the first new GPU test, and R02.1 is that ticket. The
logic is re-derived from `pre-rewrite-2026-08-30:tests/conftest.py`; the fixtures
and probe-record loaders that shared that file are not, because the tests they
served are gone.
"""

from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def cuda_unavailable_reason() -> str | None:
    """Why a CUDA device is not usable here, or `None` if one is.

    Called once per session and only when `gpu`-marked tests were actually
    collected, because importing torch and jax costs seconds and the default CPU
    container has no device to find.

    Both frameworks are consulted because a container can be half-enabled in
    either direction: the default `agent_solver` image installs CPU-only builds
    of *both*, while `agent_solver_gpu` installs CUDA builds of both. Requiring
    both keeps a partially-provisioned image from reporting itself as GPU-capable
    and then failing mid-test.
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
        + ". Run GPU tests in the CUDA container: `MOA_GPUS=device=6 make test-gpu`"
        + " (build it first with `./run.sh --gpu --rebuild pytest -q -m gpu`)."
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip `gpu`-marked tests unless this session can actually run them.

    Two conditions must hold: the session must be dedicated to GPU tests, and a
    CUDA device must be reachable. Otherwise they skip, which is what lets the
    default gate stay green unchanged on a CPU host and inside the GPU image.

    The dedicated-session rule is not fastidiousness. On the GPU image JAX puts
    *every* computation in the process on the GPU -- there is no per-test
    backend -- so a mixed selection silently moves the non-GPU tests onto a
    backend their tolerances were never derived for. The reference implementation
    measured that (CHE-60: running the default tier on the GPU image moved
    Chromatix onto the device and broke two tolerance-sensitive tests written
    against CPU float32 results).

    Read that as being about the **default**, not about an inability to place.
    CHE-244 measured it on the GPU image: `jax.devices("cpu")` does expose a
    host device, `jax.device_put` onto it is honoured, and the placement is
    sticky through subsequent operations (`jnp.sin` of a host-placed array stays
    on the host). What JAX defaults to for a *newly created* array is the GPU,
    which is exactly what breaks a mixed selection, since none of the tests this
    rule protects place explicitly. One caveat worth knowing before relying on
    the above: `numerics.arrays.to_namespace` cannot reach that host device on
    the GPU image, because `_jax_device` selects from `jax.devices()` -- the
    default backend only, `[CudaDevice(id=0)]` there -- rather than from
    `jax.devices(platform)`, so it raises `DEVICE_NOT_AVAILABLE` for a host
    target. No landed command hits that combination (`make test` runs the CPU
    image and `make test-gpu` selects only `gpu`-marked tests), which is why
    CHE-244 recorded it rather than fixing it.

    `trylast=True` matters: this hook must observe `items` after pytest's own
    `-m` deselection, because both decisions depend on what will really run
    rather than on what happened to be collected.
    """
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return

    other_items = len(items) - len(gpu_items)
    if other_items:
        skip_not_dedicated = pytest.mark.skip(
            reason=(
                f"gpu tests need a dedicated session; {other_items} non-gpu test(s) are "
                "also selected. On the GPU image every test in the process computes on "
                "the GPU, and the non-gpu tolerances were derived on the CPU. "
                "Run: `MOA_GPUS=device=6 make test-gpu`."
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
