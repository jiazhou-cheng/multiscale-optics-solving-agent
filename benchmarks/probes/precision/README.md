# Precision / dtype / device capability probes (CHE-61, PB4b)

Every device and dtype claim in `core/capabilities.py`, `registry/models.yaml`,
`registry/couplers.yaml` and `docs/precision/precision_device_policy.md` was
measured by one of these, against the pinned installs. They are committed so the
evidence is reproducible rather than asserted; the docs and the capability
declarations cite them by name.

Run each through the container. The GPU probes need the CUDA image.

| Probe | Answers | Image |
| --- | --- | --- |
| `optiland_capability.py` | What `set_precision` / `set_device` accept and refuse, and where an Optiland array actually lands | `--gpu` |
| `chromatix_capability.py` | Whether Chromatix has any complex128 path (it does not), where `asm_propagate` output lands, and whether torch→JAX DLPack stays on the device | `--gpu` |
| `default_precision.py` | Optiland's *default* precision per backend — the finding that the torch backend defaults to float32 while numpy defaults to float64 | either |
| `grad_precision.py` | The gradient path at Optiland's default vs a declared precision, quantifying the difference against the recorded gradient probe | either |
| `gpu_matmul.py` | Whether a complex64 `einsum` on this GPU is actually complex64 — it is TF32 by default, ~1500x less accurate | `--gpu` |
| `tolerance.py` | Host coupler error at float64/float32 in NumPy and JAX, and the round-trip error at each precision | either |
| `tolerance_gpu.py` | The same measurements on the device | `--gpu` |
| `capability_table.py` | Renders the capability matrix that `docs/precision/precision_device_policy.md` embeds | either |

```bash
./run.sh       python benchmarks/probes/precision/tolerance.py
./run.sh --gpu python benchmarks/probes/precision/gpu_matmul.py
```

`capability_table.py` is the generator for the documented table;
`tests/test_registry_matches_capabilities.py` fails if the doc no longer matches
it, so the table cannot go stale silently.

These write nothing and record nothing under `knowledge/`: they are capability
*measurements* used to justify a declaration, not the recorded solver fixtures
that `knowledge/solvers/*/expected/` holds. Keeping them separate is deliberate —
a capability claim is checked by `tests/test_precision_execution_matrix.py` and
`tests/test_precision_gpu_pipeline.py`, which execute it, not by comparing
against a stored number.
