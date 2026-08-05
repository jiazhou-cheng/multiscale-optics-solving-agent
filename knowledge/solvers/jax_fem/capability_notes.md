# JAX-FEM capability notes

**Bottom line: as pinned in this project's `agent_solver` image, JAX-FEM
can build meshes and define `Problem` subclasses, but CANNOT run an actual
finite-element solve.** See `solver_card.yaml` `blocker` for the full
explanation (missing petsc4py, no prebuilt wheel). Everything below
reflects that reality -- do not read this as "JAX-FEM is usable for
M_THERMAL_JAX_FEM," it currently is not, end-to-end, in this environment.

## What is confirmed to work

- `import jax_fem.problem` -- exposes `Problem`, `Mesh`, `FiniteElement`
  classes (the subclassing contract shown in the official Quickstart).
- `import jax_fem.generate_mesh` -- real mesh generation via `rectangle_mesh`,
  `box_mesh`, `box_mesh_gmsh`, `cylinder_mesh_gmsh`; returns a `meshio.Mesh`.
  Verified with a real 4x4 rectangle -> 25 nodes, 16 quad cells.
- `jax_fem.mma` module exists (method of moving asymptotes -- used for
  topology optimization in the upstream project), not yet probed.

## What is confirmed NOT to work

- `import jax_fem.solver` -- raises `ModuleNotFoundError: No module named
  'petsc4py'` immediately, because the module does an unconditional
  top-level `from petsc4py import PETSc`. This means the documented
  `solver()` entry point -- the actual linear/nonlinear FEM solve -- cannot
  be reached at all.
- By extension: nothing that depends on `solver()` can be exercised --
  no forward solve, no analytic comparison, no mesh-convergence study, no
  gradient probe. All four `required_probes` in
  `knowledge/solver_cards/jax_fem.yaml` are blocked.

## Do not assume (per CLAUDE.md section 3, rule 1 and rule 5)

- That `pip install jax-fem` gives you a working environment -- it
  declares zero dependencies and needs a hand-assembled dependency set
  (see `docker/requirements.txt` and `failure_guide.md`).
- That the PyPI license classifier ("BSD") is correct -- the actual repo
  LICENSE file is GPLv3.
- That this solver is ready to route a benchmark task to
  `M_THERMAL_JAX_FEM` -- it is not, until the petsc4py blocker is resolved.
- That a workaround linear solver written to route around petsc4py would
  count as "using jax-fem" -- it would not; that is a silent fidelity
  substitution (CLAUDE.md section 3, rule 4) and must not be represented as
  the real jax_fem.solver path.

## Follow-up required before this solver can be used

See `solver_card.yaml` `blocker.follow_up_options`. In short: either build
PETSc/petsc4py from source in the Docker image, switch to a conda-based
image with conda-forge's prebuilt petsc4py, or wait for/check for an
upstream release that makes petsc4py optional.
