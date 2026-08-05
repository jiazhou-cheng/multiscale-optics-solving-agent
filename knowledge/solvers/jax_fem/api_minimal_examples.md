# Minimal JAX-FEM examples (validated against pinned version `0.0.12`)

Every snippet below was actually executed inside the `agent_solver`
container against the pinned install in `docker/requirements.txt`. Output
values shown are real, captured on 2026-07-30, not illustrative. **There is
no working end-to-end forward-solve example in this file** because that
path is blocked -- see `capability_notes.md`.

## 1. Import / initialization

```python
import jax_fem  # prints an ASCII-art banner to stdout, harmless
import jax_fem.problem as p          # Problem, Mesh, FiniteElement
import jax_fem.generate_mesh as gm   # rectangle_mesh, box_mesh, ...
```

Full probe: `probes/import_probe.py`; captured output:
`expected/import_probe.json`.

## 2. Minimal mesh generation

```python
import jax_fem.generate_mesh as gm

mesh = gm.rectangle_mesh(Nx=4, Ny=4, domain_x=1.0, domain_y=1.0)
# type(mesh) == meshio._mesh.Mesh
# mesh.points.shape == (25, 2)          -- 5x5 nodes for a 4x4-cell grid
# mesh.cells_dict["quad"].shape == (16, 4)
```

Full probe: `probes/mesh_probe.py`; captured output:
`expected/mesh_probe.json`.

## 3. Batched / vectorized example

Not applicable / not reached -- would require the blocked solve path.

## 4. Gradient example

Not reached -- would require the blocked solve path. Do not fabricate one.

## 5. Serialization / export

Not reached. The Quickstart describes `save_sol()` exporting to VTU for
ParaView, but that function lives downstream of a solve and was not probed.

## 6. Common error signatures and repairs

See `failure_guide.md`. The single most important one:

```python
>>> import jax_fem.solver
ModuleNotFoundError: No module named 'petsc4py'
```

Full probe (captures this as an expected, regression-checked failure):
`probes/solver_failure_probe.py`; captured output:
`expected/solver_failure_probe.json`.
