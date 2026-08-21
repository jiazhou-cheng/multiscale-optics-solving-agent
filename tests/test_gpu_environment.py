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

    Needed because the property under test is a fact about JAX's *initial*
    process-global state, and JAX builds its backend once and caches it. By the
    time any test in this session runs, that has already happened, so an
    in-process check would only re-read a decision made during collection. A
    subprocess makes the check order-independent.
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

    CHE-73 widened this from a 64x64 identity multiply to a 1024x1024 `randn`
    product with an explicit ``torch.cuda.synchronize()``. Both changes matter:
    the identity case can be satisfied by a cuBLAS path that never leaves the
    trivial branch, and CUDA launches are *asynchronous*, so without the
    synchronize a launch failure could surface later, in an unrelated test, as a
    sticky context error. Two assertions follow, because "on the device" and
    "numerically right" are different claims.
    """
    import torch

    assert torch.cuda.is_available()

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(0)
    left = torch.randn(1024, 1024, device=device, dtype=torch.float32, generator=generator)
    right = torch.randn(1024, 1024, device=device, dtype=torch.float32, generator=generator)

    product = left @ right
    torch.cuda.synchronize()

    assert product.device.type == "cuda"

    # An independent CPU evaluation of the same inputs, so this is a real oracle
    # rather than a self-comparison. rtol is loose because a 1024-term float32
    # dot product accumulates differently on the two backends (and XLA/cuBLAS may
    # use TF32 for the GPU one) -- the point here is "the numbers are the right
    # numbers", not a precision claim. See docs/precision/ for that.
    expected = left.cpu() @ right.cpu()
    torch.testing.assert_close(product.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.jax
def test_a_real_jax_computation_executes_on_the_gpu() -> None:
    """The jax counterpart: confirm a jitted kernel compiles, runs, and lands on the GPU.

    Chromatix goes through jax, so torch working proves nothing about it -- the
    two use entirely separate CUDA runtimes (torch's bundled libs versus the
    jax-cuda12 plugin's). Committed arrays report their device, which is what
    Phase 4 will need to assert about Chromatix fields.

    This is the test that catches Trap 1 above. ``jax.devices()`` needs only the
    driver, so it reports a `CudaDevice` on an image with no PTX compiler, and
    the failure then appears at the *first jitted call* as
    ``XlaRuntimeError: No PTX compilation provider is available``. Four separate
    things are therefore asserted (CHE-73), because each fails independently:

      * JIT **compilation** -- reaching the `x * x` fusion at all;
      * **execution** to completion, forced by ``block_until_ready()``, since JAX
        dispatch is async and an unforced error would surface elsewhere;
      * the process **backend** is the GPU, not a silent CPU fallback;
      * the **result array's own device**, which is the only one of the four that
        says where the answer physically is.

    Deliberately *not* sufficient, and the reason this test is shaped this way:
    ``jax.devices()`` or ``jax.devices()[0].platform == "gpu"``.
    """
    import jax
    import jax.numpy as jnp

    squared_sum = jax.jit(lambda x: jnp.sum(x * x))

    values = jnp.ones((1_000_000,), dtype=jnp.float32)
    total = squared_sum(values)
    total.block_until_ready()

    assert jax.default_backend() == "gpu", (
        f"jax default backend is {jax.default_backend()!r}; the computation fell "
        "back to the host instead of failing loudly"
    )
    assert total.device.platform == "gpu", (
        f"jitted computation produced its result on {total.device!r}, not a GPU"
    )
    # Exact, not approximate: 1e6 ones each squared to 1.0, and every partial sum
    # stays below 2**24, so float32 represents the whole reduction exactly
    # whatever order XLA reduces in. A tolerance here would hide a real error.
    assert float(total) == 1_000_000.0


#: The CUDA compiler/JIT trio XLA needs for the PTX stage. These three must be
#: the *same* version: `nvidia-cuda-nvcc-cu12` supplies ptxas/nvlink, and XLA
#: links the result through nvjitlink. Measured 12.6.85 across all three on the
#: agent_solver_gpu image, matching the family torch's cu126 wheels install.
_CUDA_TOOLCHAIN = ("nvidia-cuda-nvcc-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-nvjitlink-cu12")

#: CUDA *runtime* components, which track the CUDA release and so share its
#: 12.6.x prefix. Listed separately from the math libraries below because those
#: version independently of the toolkit (cufft is 11.x, cusolver 11.x) and
#: asserting "12.6" on them would be false.
_CUDA_RUNTIME = ("nvidia-cuda-runtime-cu12", "nvidia-cuda-cupti-cu12", "nvidia-nvtx-cu12")

_EXPECTED_CUDA_SERIES = "12.6"


@pytest.mark.torch
@pytest.mark.jax
def test_one_coherent_cuda_dependency_family(
    record_property: Callable[[str, object], None],
) -> None:
    """torch and JAX must share one CUDA 12.6 stack, not install two competing ones.

    This is the dependency-side guard for Trap 1, and it is what makes the
    minimal-toolchain choice checkable instead of merely documented. The GPU image
    deliberately does *not* use `jax[cuda12]` / `jax-cuda12-plugin[with-cuda]`,
    which would pull a second full `nvidia-*` set alongside torch's pinned 12.6.x
    wheels; pip installs exactly one version per distribution, so the loser is
    silently overwritten and whichever framework needed the other one breaks at
    runtime, far from the dependency change that caused it.

    So the failure this catches is a *dependency edit*, not a broken GPU: adding
    a CUDA extra, or bumping torch to a cu12x that no longer matches nvcc. The
    versions are recorded via `record_property` as well as asserted, so a run's
    JUnit XML carries the resolved stack rather than a hand-copied `pip list`.

    Asserted in three groups because they version on different schedules --
    treating them uniformly would mean either a false assertion or no assertion.
    See `docker/requirements-gpu.txt` for the pins and the reasoning.
    """
    from importlib.metadata import PackageNotFoundError, version

    import torch

    def installed(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    resolved = {
        name: installed(name)
        for name in (*_CUDA_TOOLCHAIN, *_CUDA_RUNTIME, "nvidia-cublas-cu12", "nvidia-cufft-cu12")
    }
    for name, found in resolved.items():
        record_property(name, found)
    record_property("torch_cuda_version", torch.version.cuda)

    # 1. The PTX toolchain is present at all. Its absence is Trap 1: enumeration
    #    keeps working and only the first jitted call dies.
    assert resolved["nvidia-cuda-nvcc-cu12"] is not None, (
        "nvidia-cuda-nvcc-cu12 is not installed, so XLA has no ptxas/nvlink and "
        "every jitted GPU computation will fail with 'No PTX compilation provider "
        "is available' even though jax.devices() reports a CUDA device."
    )

    # 2. Compiler and JIT-linker agree exactly. A split here is the concrete
    #    symptom of two CUDA stacks fighting over the same names.
    toolchain = {name: resolved[name] for name in _CUDA_TOOLCHAIN}
    assert len(set(toolchain.values())) == 1, (
        f"the CUDA compiler/JIT toolchain is not one version: {toolchain}. This is "
        "what installing a second nvidia-* set (e.g. via jax[cuda12]) looks like "
        "after pip picks one winner per package."
    )

    # 3. Everything that tracks the CUDA release tracks the *same* release, torch
    #    included -- torch is the reason 12.6 is the target rather than the latest.
    assert torch.version.cuda == _EXPECTED_CUDA_SERIES, (
        f"torch reports CUDA {torch.version.cuda}, not {_EXPECTED_CUDA_SERIES}. If "
        "this is a deliberate torch bump, re-pin nvidia-cuda-nvcc-cu12 to the "
        "matching family in docker/requirements-gpu.txt and update this test."
    )
    for name in (*_CUDA_TOOLCHAIN, *_CUDA_RUNTIME):
        found = resolved[name]
        assert found is not None and found.startswith(f"{_EXPECTED_CUDA_SERIES}."), (
            f"{name} is {found!r}, outside the CUDA {_EXPECTED_CUDA_SERIES} family "
            f"torch is built against. Resolved stack: {resolved}"
        )


@pytest.mark.jax
def test_a_clean_interpreter_reaches_the_gpu_with_no_platform_repair() -> None:
    """Nothing this project installs or imports may pin JAX off the GPU.

    This replaces two CHE-60 tests that characterized the opposite: SAX's
    ``klujax`` dependency ran ``jax.config.update("jax_platform_name", "cpu")``
    at *import* time, so importing SAX silently moved every later JAX
    computation in the process onto the host, and the pytest harness had to undo
    that during collection. CHE-72 removed SAX, so the hazard is gone at the
    source and the harness repairs nothing. What is worth guarding is the
    property that replaced it: a *fresh* interpreter must reach the GPU on its
    own.

    Asserted in a subprocess, and importing the project package rather than only
    ``jax``, because that is the thing that could regress -- a future dependency
    with the same import-time habit would be caught here. The in-process
    equivalent is worthless: this session's backend is already initialized by the
    time any test runs, and JAX builds its backend once and caches it.

    Three facts, because each fails independently (AGENTS.md: precision, dtype,
    device and namespace are separate concepts, and a requested device is never
    evidence of an actual one):

      * ``jax_platform_name`` is not pinned to ``'cpu'`` -- nothing repaired it;
      * the default backend is the GPU;
      * a jitted kernel's *output array* actually lands on a GPU device, which
        device enumeration alone does not establish (see Trap 1 in
        docs/testing/gpu_environment.md).
    """
    platform_name, backend, result_platform = _run_in_subprocess(
        """
        import multiscale_optics_agent  # noqa: F401  -- must not touch jax config
        import jax
        import jax.numpy as jnp

        print(jax.config.read("jax_platform_name") or "<unset>")
        print(jax.default_backend())
        print(jax.jit(jnp.sum)(jnp.arange(8, dtype=jnp.float32)).device.platform)
        """
    ).splitlines()

    assert platform_name != "cpu", (
        "a clean interpreter has jax_platform_name pinned to 'cpu'. Some import "
        "is setting it process-globally, which silently disables the GPU for "
        "everything downstream. Find the package that does it (CHE-72 removed "
        "the previous offender, SAX/klujax) rather than repairing the pin in the "
        "test harness."
    )
    assert backend == "gpu", (
        f"a clean interpreter reports backend {backend!r}, not 'gpu' "
        f"(jax_platform_name={platform_name!r})"
    )
    assert result_platform == "gpu", (
        f"a jitted kernel's output landed on {result_platform!r}, not a GPU, even "
        f"though the default backend is {backend!r}"
    )
