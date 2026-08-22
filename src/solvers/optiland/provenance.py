"""What ran, on what, and a hash of what it produced.

Three things the adapter records rather than computes: the packaged
``ModelSpec``, the host CPU's name, and a content hash over the traced arrays.

The hash is the load-bearing one. It is taken over the arrays themselves in a
fixed key order with dtype and shape mixed in, so it separates a float32 trace
from a float64 one carrying the same values -- which a hash of a printed summary
would not.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.specs import ModelSpec
from registry.loader import Registry
from solvers.optiland.constants import (
    MODEL_ID,
)


@lru_cache(maxsize=1)
def _load_spec() -> ModelSpec:
    return Registry.from_package().models[MODEL_ID]


def _cpu_device_name() -> str:
    """Return an observable CPU description without claiming core isolation."""
    model = platform.processor().strip()
    if not model:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return model or platform.machine() or "cpu"


def _scientific_array_hash(arrays: Mapping[str, Any]) -> str:
    """Hash names, dtype, shape, and contiguous bytes independent of NPZ metadata."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
