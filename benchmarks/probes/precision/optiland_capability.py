"""What Optiland's device/precision API actually accepts (CHE-61, PB4b).

Answers three questions that the project had been guessing at:

* which precisions `set_precision` accepts, and what it does with anything else;
* whether `set_device` works off the torch backend (it does not);
* where an Optiland array physically lands once both are set.

Run with `./run.sh --gpu python benchmarks/probes/precision/optiland_capability.py`.
"""

import inspect
import json

import optiland.backend as be
import torch

out = {}
out["torch"] = {
    "version": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
out["optiland"] = {
    "set_precision_sig": str(inspect.signature(be.set_precision)),
    "set_device_sig": str(inspect.signature(be.set_device)),
    "get_device": "numpy backend raises BackendCapabilityError",
    "has_get_precision": hasattr(be, "get_precision"),
}
try:
    out["optiland"]["precision_src"] = inspect.getsource(be.set_precision)
except Exception as e:
    out["optiland"]["precision_src"] = repr(e)
try:
    out["optiland"]["device_src"] = inspect.getsource(be.set_device)
except Exception as e:
    out["optiland"]["device_src"] = repr(e)

# torch backend + cuda + float32
be.set_backend("torch")
try:
    be.set_device("cuda")
    be.set_precision("float32")
    a = be.array([1.0, 2.0, 3.0])
    out["optiland"]["cuda_f32"] = {
        "type": type(a).__name__,
        "dtype": str(a.dtype),
        "device": str(a.device),
    }
except Exception as e:
    out["optiland"]["cuda_f32"] = repr(e)
try:
    be.set_precision("float64")
    a = be.array([1.0, 2.0, 3.0])
    out["optiland"]["cuda_f64"] = {"dtype": str(a.dtype), "device": str(a.device)}
except Exception as e:
    out["optiland"]["cuda_f64"] = repr(e)
try:
    be.set_precision("float16")
    out["optiland"]["fp16"] = "ACCEPTED"
except Exception as e:
    out["optiland"]["fp16"] = f"{type(e).__name__}: {e}"
try:
    be.set_precision("float64")
    be.set_device("cpu")
    be.set_backend("numpy")
    be.set_device("cuda")
    out["optiland"]["numpy_cuda"] = "ACCEPTED"
except Exception as e:
    out["optiland"]["numpy_cuda"] = f"{type(e).__name__}: {e}"

print(json.dumps(out, indent=2))
