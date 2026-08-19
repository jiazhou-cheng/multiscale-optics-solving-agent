"""Phase 0 environment contract for CHE-60 (PB4a): can this container reach a GPU?

Every other GPU acceptance criterion in CHE-60 depends on this, so it is tested
directly instead of being assumed from the presence of a torch/jax install. The
parent issue CHE-55 deferred all GPU work on the grounds that "this container
has never had CUDA/GPU hardware available to probe against"; these tests are the
executable check of whether that is still true, rather than a claim in a
docstring.

All tests here are ``gpu``-marked, so they auto-skip in the default CPU
container (see ``cuda_unavailable_reason`` in conftest) and Tier A stays green
on a CPU-only host. Run them with::

    ./run.sh --gpu --rebuild pytest -q -m gpu    # first time (builds the image)
    ./run.sh --gpu pytest -q -m gpu              # subsequently

Note what these tests deliberately do *not* assert: they do not check that any
adapter or coupler uses the GPU. That is Phases 1-4. This file only pins down
that a CUDA device is reachable by both numerical backends, which is the
precondition, and records which device actually served the request (AGENTS.md:
"record the actual device used").
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _run_in_subprocess(body: str) -> str:
    """Execute `body` in a clean interpreter and return its stdout.

    Needed because the behavior under test is *process-global and
    irreversible-once-initialized* JAX state. Exercising it in-process would
    either pin this process to the CPU for every later test or be silently
    no-opped by an already-initialized backend, depending on ordering. A
    subprocess makes the characterization order-independent.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.torch
def test_torch_reaches_a_cuda_device() -> None:
    """torch must be a CUDA build *and* have a device attached.

    These are two independent failure modes and are asserted separately: the
    default image installs a `+cpu` build (no CUDA support compiled in at all),
    while a CUDA build launched without `--gpus` compiles fine but sees zero
    devices. Optiland's `backend.set_device('cuda')` needs both.
    """
    import torch

    assert torch.version.cuda is not None, (
        f"torch {torch.__version__} has no CUDA support compiled in. This is the "
        "default agent_solver image, which installs from the CPU-only wheel "
        "index; use ./run.sh --gpu."
    )
    assert "+cpu" not in torch.__version__, (
        f"torch {torch.__version__} is a CPU-only build; the GPU image should provide a +cu* build."
    )
    assert torch.cuda.is_available(), (
        "torch has CUDA support but no device is attached -- the container was "
        "started without --gpus."
    )
    assert torch.cuda.device_count() >= 1


@pytest.mark.jax
def test_jax_reaches_a_cuda_device() -> None:
    """jax must resolve a GPU backend, which requires the jax-cuda12 plugin.

    `jax==0.6.2` alone resolves a CPU-only jaxlib, so this fails in the default
    image even though `import jax` succeeds -- the failure is a silent
    `default_backend() == 'cpu'`, never an ImportError. That silence is exactly
    what Chromatix would inherit.
    """
    import jax

    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    assert gpu_devices, (
        f"jax sees no GPU device (backend={jax.default_backend()!r}, "
        f"devices={devices!r}). The jax-cuda12-plugin/pjrt packages in "
        "docker/requirements-gpu.txt provide it."
    )


@pytest.mark.torch
def test_gpu_visibility_honors_the_two_device_project_cap() -> None:
    """The container must not be handed more than 2 GPUs.

    AGENTS.md caps this project at 2 GPUs on a shared host and requires that the
    limit be enforced by container configuration, not host state. `run.sh`
    rejects `MOA_GPUS=all` and any selection above 2, so a compliant invocation
    can never expose more than 2 devices here. This asserts the policy from
    inside the container, where it actually matters -- if it fails, the caller
    bypassed run.sh (e.g. raw `docker run --gpus all`).
    """
    import torch

    visible = torch.cuda.device_count()
    assert visible <= 2, (
        f"{visible} GPUs are visible to this container, but AGENTS.md caps this "
        "project at 2. Start the container through ./run.sh --gpu (optionally "
        "MOA_GPUS=device=N) rather than passing --gpus all directly."
    )


@pytest.mark.torch
@pytest.mark.jax
def test_record_actual_devices_used(record_property: Callable[[str, object], None]) -> None:
    """Attach the concrete device identity to the test report as evidence.

    AGENTS.md requires recording the actual device used rather than the device
    requested. Emitting it through `record_property` puts it in the JUnit XML and
    `-rA` output, so a CHE-60 validation run carries its own provenance instead
    of relying on a hand-copied `nvidia-smi` line in a work log.
    """
    import jax
    import torch

    torch_devices = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    jax_devices = [str(device) for device in jax.devices()]

    record_property("torch_version", torch.__version__)
    record_property("torch_cuda_version", torch.version.cuda)
    record_property("torch_devices", ", ".join(torch_devices))
    record_property("torch_capability", str(torch.cuda.get_device_capability(0)))
    record_property("jax_version", jax.__version__)
    record_property("jax_backend", jax.default_backend())
    record_property("jax_devices", ", ".join(jax_devices))

    # Not a tautology given the assertions above: this catches the case where
    # torch and jax disagree about whether a GPU exists, which would make any
    # cross-framework handoff test in Phase 6 meaningless.
    assert torch_devices, "torch reported zero CUDA devices"
    assert jax.default_backend() == "gpu", (
        f"jax default backend is {jax.default_backend()!r} while torch sees "
        f"{torch_devices}; the two backends disagree about GPU availability."
    )


@pytest.mark.torch
def test_a_real_kernel_executes_on_the_gpu() -> None:
    """Smoke-test actual compute, not just device enumeration.

    Device enumeration can succeed while kernel launch fails -- the usual cause
    is a CUDA/driver mismatch (this image ships cu126 wheels against a 12.4
    driver, relying on CUDA minor-version compatibility). A matmul is the
    cheapest thing that forces real kernel execution and would surface that as a
    failure here rather than deep inside an Optiland trace in Phase 1.
    """
    import torch

    device = torch.device("cuda:0")
    left = torch.eye(64, device=device, dtype=torch.float32)
    right = torch.arange(64 * 64, device=device, dtype=torch.float32).reshape(64, 64)

    product = left @ right

    assert product.device.type == "cuda"
    # Multiplying by the identity must be exact even in float32, so this needs no
    # tolerance and any mismatch indicates a genuinely broken kernel path.
    assert torch.equal(product, right)


@pytest.mark.jax
def test_a_real_jax_computation_executes_on_the_gpu() -> None:
    """The jax counterpart: confirm a jitted kernel lands on the GPU.

    Chromatix goes through jax, so torch working proves nothing about it -- the
    two use entirely separate CUDA runtimes (torch's bundled libs versus the
    jax-cuda12 plugin's). Committed arrays report their device, which is what
    Phase 4 will need to assert about Chromatix fields.
    """
    import jax
    import jax.numpy as jnp

    values = jnp.arange(1024, dtype=jnp.float32)
    total = jax.jit(jnp.sum)(values)

    assert total.device.platform == "gpu", f"jitted computation ran on {total.device!r}, not a GPU"
    # 1023*1024/2, exactly representable in float32.
    assert float(total) == pytest.approx(523776.0)


# Deliberately not `sax`-marked despite exercising SAX: the marker is what the
# conftest conflict guard uses to detect "a SAX test will run in this process,"
# and these two run SAX in a *subprocess* precisely so they never pin this one.
@pytest.mark.jax
def test_importing_sax_silently_pins_jax_to_cpu() -> None:
    """Characterize a third-party hazard that can silently disable GPU execution.

    ``klujax.py:47`` runs ``jax.config.update(name="jax_platform_name",
    val="cpu")`` at import time, and klujax is a hard dependency of SAX. So
    importing SAX -- an out-of-milestone solver that a graph or test session may
    load for unrelated reasons -- forces every later JAX computation in that
    process onto the CPU.

    What makes this dangerous rather than merely annoying is that it is
    completely silent: no warning, no exception, and ``JAX_PLATFORMS`` is never
    set. Chromatix would keep producing correct-looking numbers at CPU speed
    while the caller believes it requested a GPU. This test exists so the
    behavior is pinned as *known* rather than rediscovered; if a future klujax
    drops the pin, this test fails and the workaround in
    ``conftest.undo_third_party_jax_platform_pin`` can be deleted.

    Consequence for CHE-60 Phase 4: the Chromatix adapter must report the device
    its output actually landed on, never the device that was requested.
    """
    backend = _run_in_subprocess(
        """
        import sax  # noqa: F401  -- imported for its side effect on jax config
        import jax
        print(jax.default_backend())
        """
    )

    assert backend == "cpu", (
        f"expected importing sax to pin jax to cpu, got {backend!r}. If klujax "
        "no longer pins jax_platform_name, remove the workaround in "
        "tests/conftest.py::undo_third_party_jax_platform_pin and this test."
    )


@pytest.mark.jax
def test_jax_platform_pin_is_reversible_only_before_backend_init() -> None:
    """Pin down *why* the conftest repair must run where it does.

    JAX builds its backend once and caches it, so clearing
    ``jax_platform_name`` helps only if nothing has initialized the backend yet.
    Both orders are checked in one subprocess:

      * reset, then first ``devices()`` call  -> GPU is recovered
      * ``devices()`` first, then reset       -> permanently stuck on CPU

    This is the constraint that forces the repair into
    ``pytest_collection_modifyitems`` (after collection's SAX import, before any
    test touches JAX) rather than into a fixture, which would run too late.
    """
    early, late = _run_in_subprocess(
        """
        import subprocess, sys, textwrap

        def backend_after(script):
            out = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                 capture_output=True, text=True, check=True)
            return out.stdout.strip()

        # Reset before the backend is ever built.
        print(backend_after('''
            import sax, jax
            jax.config.update("jax_platform_name", None)
            print(jax.default_backend())
        '''))
        # Reset after the backend has been built and cached.
        print(backend_after('''
            import sax, jax
            jax.devices()
            jax.config.update("jax_platform_name", None)
            print(jax.default_backend())
        '''))
        """
    ).splitlines()

    assert early == "gpu", (
        f"clearing jax_platform_name before backend init should recover the GPU, got {early!r}"
    )
    assert late == "cpu", (
        f"clearing jax_platform_name after backend init should NOT recover the "
        f"GPU (the backend is cached), got {late!r} -- if this now returns 'gpu', "
        "jax gained the ability to rebuild its backend and the conftest repair "
        "no longer needs to run during collection."
    )
