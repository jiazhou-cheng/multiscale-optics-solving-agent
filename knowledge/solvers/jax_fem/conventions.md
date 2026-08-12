# JAX-FEM conventions (pinned version `0.0.12`)

**Read this first:** the conventions below only cover the parts of jax-fem
that actually import in this environment (mesh generation, `Problem`
class shape). The solve step itself (`jax_fem.solver`) is blocked -- see
`solver_card.yaml` `blocker` and `failure_guide.md`. Nothing about units,
discretization convergence, or derivative behavior of an actual FEM solve
has been verified, because there is currently no way to run one here.

## Mesh representation

`jax_fem.generate_mesh.rectangle_mesh(Nx, Ny, domain_x, domain_y)` returns
a plain `meshio.Mesh` object (confirmed via `probes/mesh_probe.py`):

- `mesh.points`: shape `(num_nodes, spatial_dim)`, e.g. `(25, 2)` for a
  4x4-cell rectangle (5x5 = 25 nodes).
- `mesh.cells_dict`: a dict keyed by cell type string (e.g. `"quad"`),
  each value an `(num_cells, nodes_per_cell)` connectivity array, e.g.
  `(16, 4)` for 16 quad cells.

This is meshio's own convention, not something jax-fem invents -- any
coupler feeding a mesh into jax-fem, or consuming one from it, should treat
it as a `meshio.Mesh` and rely on meshio's documented conventions, not
jax-fem-specific ones.

## Units

Not established. The official Quickstart's Poisson example uses
dimensionless unit-square domains; no probe in this repository has
exercised a physically-dimensioned (SI) case, because doing so would
require the blocked `solver()` path. **Do not assume jax-fem enforces SI
units** -- treat any unit conversion as the adapter's responsibility and
test it explicitly once the solver path is unblocked.

## Import side effects

`import jax_fem` unconditionally prints an ASCII-art banner to stdout
(observed real behavior, see `probes/import_probe.py` docstring). This is
harmless but will pollute structured logs or JSON-formatted stdout capture
in an automated pipeline -- redirect or filter it if you need clean stdout.

## Differentiation

Not established. `jax_fem.solver` -- the module that would presumably wire
up the implicit-differentiation path implied by the project's registry
entry (`derivative.mode: implicit` for `M_THERMAL_JAX_FEM`) -- cannot be
imported without petsc4py. No gradient claim can be made for this solver
until that blocker is resolved and a real directional-derivative probe
passes under the repository gradient-verification policy.
