"""Shared pytest fixtures and helpers for the test suite."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from multiscale_optics_agent.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "solvers"

# Records any repair of a third-party global JAX platform pin so it is reported
# in the terminal summary rather than applied invisibly -- mutating another
# package's global config is exactly the kind of thing that should be stated out
# loud in the run it affects.
_JAX_PLATFORM_REPAIR: pytest.StashKey[list[str]] = pytest.StashKey()


def undo_third_party_jax_platform_pin() -> str | None:
    """Undo klujax's process-global pinning of JAX to CPU. Returns what it did.

    Measured on the GPU image (CHE-60): ``klujax.py:47`` executes

        jax.config.update(name="jax_platform_name", val="cpu")

    at *import* time. klujax is a hard dependency of SAX, so merely importing
    SAX -- which pytest does during collection, for every run, because
    ``tests/test_sax_adapter.py`` imports it at module level -- silently forces
    every subsequent JAX computation in the process onto the CPU. No warning is
    emitted and ``JAX_PLATFORMS`` is never set, so the only visible symptom is
    that ``jax.default_backend()`` returns ``'cpu'`` on a machine with a working
    GPU. This is why ``-m gpu`` skipped everything while running
    ``tests/test_gpu_environment.py`` alone passed.

    The pin is reversible, but only before JAX initializes its backend: the
    backend is built once and cached, so a reset after the first
    ``jax.devices()`` call has no effect (verified both ways). This function is
    therefore called from ``pytest_collection_modifyitems`` -- after collection
    has done the offending imports, but before any test has touched JAX.

    Note this is a *test-harness* repair, not a fix for the underlying hazard:
    any production process that imports SAX before Chromatix inherits the same
    silent CPU downgrade. CHE-60 Phase 4 must verify Chromatix's actual output
    device rather than trusting the requested one.
    """
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is pinned in both images
        return None

    # `jax.config.read(...)`, not `jax.config.jax_platform_name`: jax 0.6.2
    # exposes `jax_platform_name` as a settable option but not as a readable
    # attribute (only the newer `jax_platforms` is). Unset reads as `''`.
    if jax.config.read("jax_platform_name") != "cpu":  # type: ignore[no-untyped-call]
        return None

    jax.config.update("jax_platform_name", None)  # type: ignore[no-untyped-call]
    if jax.default_backend() == "cpu":
        return (
            "jax_platform_name was pinned to 'cpu' (klujax, imported via SAX) and "
            "could not be undone -- JAX had already initialized its CPU backend"
        )
    return "undid klujax's jax_platform_name='cpu' pin (imported via SAX)"


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
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
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

    # Enabling the GPU requires mutating process-global JAX state (see
    # undo_third_party_jax_platform_pin), which changes the backend every other
    # test in the session computes on. That is only acceptable in a session whose
    # tests all asked for the GPU, so GPU tests run *only* in a dedicated
    # selection and skip whenever anything else is selected alongside them.
    #
    # Both directions of this were measured on the GPU image, and neither is
    # hypothetical:
    #   * with SAX selected, repairing the pin broke
    #     test_mzi_circuit_matches_analytic_oracle_and_probe_evidence (klujax
    #     needs JAX on the CPU);
    #   * with Tier A selected, it moved Chromatix onto the GPU and broke two
    #     tolerance-sensitive tests in test_m3_pupil_to_focus.py.
    # Skipping the GPU tests is the safe direction in both: the established CPU
    # results stay authoritative and the GPU tests remain available on their own.
    other_items = len(items) - len(gpu_items)
    if other_items:
        skip_not_dedicated = pytest.mark.skip(
            reason=(
                f"gpu tests need a dedicated session; {other_items} non-gpu test(s) are "
                "also selected. Enabling the GPU mutates process-global jax state "
                "(klujax pins jax_platform_name='cpu' at import) and would change what "
                "those tests compute on. Run: `./run.sh --gpu pytest -q -m gpu`"
            )
        )
        for item in gpu_items:
            item.add_marker(skip_not_dedicated)
        return

    # Must run before cuda_unavailable_reason(), which initializes the JAX
    # backend and thereby freezes whatever platform is configured at that moment.
    repair = undo_third_party_jax_platform_pin()
    if repair is not None:
        config.stash.setdefault(_JAX_PLATFORM_REPAIR, []).append(repair)

    reason = cuda_unavailable_reason()
    if reason is None:
        return

    skip_gpu = pytest.mark.skip(reason=f"no CUDA device available: {reason}")
    for item in gpu_items:
        item.add_marker(skip_gpu)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Surface any global JAX platform repair this run had to perform."""
    repairs = terminalreporter.config.stash.get(_JAX_PLATFORM_REPAIR, [])
    for repair in repairs:
        terminalreporter.write_line(f"jax platform repair: {repair}")


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
