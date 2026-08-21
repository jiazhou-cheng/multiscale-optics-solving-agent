"""CHE-72 / CHE-73 GPU revalidation probe.

Run:

    ./run.sh --gpu --no-build python benchmarks/probes/gpu/che72_gpu_revalidation.py

Why this exists as a probe as well as a test suite: it answers the two tickets'
acceptance criteria in ONE process that pytest never touches. `tests/conftest.py`
used to repair a third-party `jax_platform_name='cpu'` pin during collection, so
"the GPU works under pytest" and "the GPU works in a clean interpreter" were
genuinely different claims. CHE-72 removed both SAX and that repair; this probe is
the standing check that the second claim now holds on its own.

The durable guards are `tests/test_gpu_environment.py` (8 tests, `gpu`-marked).
This probe is the human-readable snapshot whose output is quoted in
`docs/testing/gpu_environment.md`. Recorded output, agent_solver_gpu rebuilt
2026-08-20 on 1x RTX A6000, is in that document; re-run and diff rather than
trusting the transcript.
"""

import json
from importlib.metadata import PackageNotFoundError, distributions, version

record = {}

# --- SAX must be entirely absent from the image ---
installed = {(d.metadata["Name"] or "").lower() for d in distributions()}
record["sax_family_installed"] = sorted(
    installed & {"sax", "klujax", "jaxellip", "lark", "natsort", "orjson", "scikit-rf"}
)
for mod in ("sax", "klujax"):
    try:
        __import__(mod)
        record[f"{mod}_importable"] = True
    except ImportError:
        record[f"{mod}_importable"] = False

# --- no platform pin is set or repaired anywhere ---
# E402 suppressed deliberately: importing jax only AFTER the absence checks
# above is the point of this probe -- reordering it would make the platform
# read meaningless, since the backend is built once and cached.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

record["jax_platform_name"] = jax.config.read("jax_platform_name") or "<unset>"
record["jax_version"] = jax.__version__
record["jax_devices"] = [str(d) for d in jax.devices()]
record["jax_default_backend"] = jax.default_backend()

# --- JAX: jit compile + execute + synchronize + result device (CHE-73 shape) ---
f = jax.jit(lambda x: jnp.sum(x * x))
x = jnp.ones((1_000_000,), dtype=jnp.float32)
y = f(x)
y.block_until_ready()
record["jax_jit_result"] = float(y)
record["jax_jit_result_exact"] = float(y) == 1_000_000.0
record["jax_result_device"] = str(y.device)
record["jax_result_platform"] = y.device.platform

# --- torch: real CUDA matmul + synchronize (CHE-73 shape) ---
import torch  # noqa: E402

record["torch_version"] = torch.__version__
record["torch_cuda_version"] = torch.version.cuda
record["torch_cuda_available"] = torch.cuda.is_available()
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b
torch.cuda.synchronize()
record["torch_result_device"] = c.device.type
record["torch_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

# --- one coherent CUDA family ---
cuda_pkgs = {}
for name in sorted(n for n in installed if n.startswith("nvidia-")):
    try:
        cuda_pkgs[name] = version(name)
    except PackageNotFoundError:
        cuda_pkgs[name] = None
record["nvidia_packages"] = cuda_pkgs

print(json.dumps(record, indent=2))
