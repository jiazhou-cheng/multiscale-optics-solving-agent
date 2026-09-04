"""Provenance for a probe record, and the one directory they are written to.

`benchmarks/verification/record.py` already owns the record contract -- `Row`,
`STATUSES`, `finish`, and the git/environment fingerprint CHE-238 specified --
and this module deliberately adds only the one thing that contract gets wrong
here. Its `provenance()` embeds `device_execution()`, which is the *constant*
`{"device": "cpu", "precision": "fp64"}`, justified there by CHE-238 §3: the CPU
image's jaxlib and torch are CPU-only builds, so naming a device would have been
a claim that environment could not honour.

A probe under this tree runs on the CUDA image on purpose, and writing "device:
cpu" into a record of a GPU measurement would be exactly the invention
`AGENTS.md` forbids. So `probe_provenance` takes the same fingerprint and
replaces that one row with what the process can actually *see*: which frameworks
report a device, and which one. Observed, not requested -- a probe that trusted
its own `--gpu` flag could not detect the case where the flag was accepted and no
device was attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.verification.record import provenance

__all__ = ["RECORDS", "observed_devices", "probe_provenance"]

#: Where every probe record in this ticket family is written. Beside the driver,
#: on `benchmarks/README.md`'s rule that a record lives with the script that
#: wrote it, and under a per-backend directory because the capability packs
#: already cite that shape (`benchmarks/probes/<area>/...`).
RECORDS = Path(__file__).resolve().parent / "records" / "optiland"


def observed_devices() -> dict[str, Any]:
    """What each framework reports about the devices attached to this process.

    Every field is read from the framework rather than from an argument. A
    missing framework is recorded as such instead of being omitted, because
    "torch could not be imported" and "torch saw no GPU" send a reader to
    different problems.
    """
    report: dict[str, Any] = {}
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
            ],
        }
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        report["torch"] = {"error": f"not importable ({exc})"}
    try:
        import jax

        report["jax"] = {
            "version": jax.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "enable_x64": bool(jax.config.read("jax_enable_x64")),
        }
    except ImportError as exc:  # pragma: no cover - jax is a hard dependency
        report["jax"] = {"error": f"not importable ({exc})"}
    return report


def probe_provenance() -> dict[str, Any]:
    """CHE-238's fingerprint with its constant device row replaced by an observed one.

    The replaced row also carried `precision`, and it is not reinstated here: a
    probe measures more than one precision per run, so a single process-level
    precision would be a claim about the wrong scope. It moves onto the row
    instead, in `configuration`, where p1 and p2 carry it. P3 carries none
    because a DLPack bridge has no precision -- it has a dtype, which its own
    `dlpack_result_state` reports as observed.

    Note that the container decides *which* physical GPU an ordinal refers to:
    `MOA_GPUS=device=6` makes that GPU the process's `cuda:0`, so the ordinal in
    a record is the container's and the device *name* is what identifies the
    hardware. Stated here because a record that named `cuda:0` alone would read
    as GPU 0 of the host, which is not what it measured.
    """
    record = provenance()
    # Asserted rather than assigned over. A blind write would leave a record
    # carrying *both* claims -- the inherited constant under a renamed key and the
    # observed report under this one -- if the upstream contract moved, and a
    # record that asserts two devices is worse than one that fails to build.
    assert "device" in record["environment"], (
        "benchmarks/verification/record.py::provenance no longer carries "
        "environment['device']; this module replaces that row and cannot silently "
        "leave the inherited cpu/fp64 constant in place under another name"
    )
    record["environment"]["device"] = {
        "declaration": (
            "observed, not requested. The container's visible device set decides which "
            "physical GPU is cuda:0; see MOA_GPUS in run.sh."
        ),
        **observed_devices(),
    }
    return record
