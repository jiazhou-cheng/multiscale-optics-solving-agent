# JAX-FEM failure guide

Real errors hit while building this knowledge pack (2026-07-30), with
repairs. Add to this file rather than silently working around a new one.

## `pip install jax-fem` gives you a package with no dependencies

**Symptom:** `pip install jax-fem` succeeds, `import jax_fem` even works
(it only needs the standard library plus its own `pyfiglet`/logging bits),
but any real usage (`import jax_fem.problem`, `import jax_fem.generate_mesh`)
immediately fails with `ModuleNotFoundError` for numpy/scipy/meshio/etc.

**Cause:** the published wheel's METADATA has zero `Requires-Dist` lines.
Confirmed by downloading the wheel directly
(`pip download --no-deps jax-fem`) and reading its `METADATA` file, and by
checking the upstream `pyproject.toml`
(https://raw.githubusercontent.com/deepmodeling/jax-fem/main/pyproject.toml),
which has no `dependencies = [...]` field at all.

**Fix:** install the real dependency set by hand. This project's
`docker/requirements.txt` does this, sourced from jax-fem's own
`environment.yml`: `numpy, scipy, matplotlib, meshio==5.3.5,
gmsh==4.15.2, fenics-basix==0.11.0, pyfiglet==1.0.4` (jax itself is already
pinned for the other JAX-ecosystem solvers in the same image).

## `OSError: libGLU.so.1: cannot open shared object file`

**Symptom:** raised on `import jax_fem.generate_mesh` (which imports
`gmsh` transitively via `jax_fem.utils`), even though the task at hand is
purely headless mesh generation with no rendering involved.

**Cause:** gmsh's Python bindings `dlopen` a real OpenGL/GLU shared library
at import time regardless of whether you use any visualization feature.

**Fix:** install `libglu1-mesa` (and, empirically, also needed:
`libgl1 libxrender1 libxcursor1 libxinerama1 libxft2`) via apt. Already
done in `docker/Dockerfile`.

## `OSError: libgomp.so.1: cannot open shared object file`

**Symptom:** same import path as above, appears *after* fixing the GLU
error -- gmsh (or a dependency compiled with OpenMP) also needs the GNU
OpenMP runtime.

**Fix:** install `libgomp1` via apt. Already done in `docker/Dockerfile`.

## `ModuleNotFoundError: No module named 'petsc4py'` -- the big one

**Symptom:** `import jax_fem.solver` fails immediately.

**Cause:** `jax_fem/solver.py` has an unconditional top-level
`from petsc4py import PETSc`. petsc4py has no prebuilt PyPI wheel (source
distribution only), and building it requires compiling PETSc itself
(needs a Fortran compiler, BLAS/LAPACK, typically MPI) -- a substantial,
slow addition deliberately not made to this lightweight pip-based image.

**Not a fix:** `apt-get install python3-petsc4py` -- Debian does package
this, but it installs into the system Debian Python's site-packages, not
the from-source-built `python:3.12-slim` interpreter this image's pip uses.
Verified this does not make `import petsc4py` succeed for the pip-managed
interpreter.

**Real fixes (not yet implemented, tracked in solver_card.yaml):**
1. Build PETSc + petsc4py from source (gfortran, libopenmpi-dev,
   libblas-dev, liblapack-dev, then `pip install petsc petsc4py`) -- slow.
2. Switch this solver to a conda-based image and use conda-forge's
   prebuilt `petsc4py` -- matches jax-fem's own documented install path
   (`conda env create -f environment.yml`).
3. Re-check on refresh whether a future jax-fem release makes this
   optional/lazy.

## PyPI license classifier is wrong

**Symptom:** none at runtime -- a documentation trap. PyPI's `info.license`
field for this package says "BSD License."

**Cause:** the actual repository LICENSE file
(https://raw.githubusercontent.com/deepmodeling/jax-fem/main/LICENSE) is
the full GNU GPLv3 text. The PyPI classifier is simply incorrect.

**Fix:** always fetch and read the actual LICENSE file for this package
before making any redistribution/licensing decision; do not trust the PyPI
classifier for jax-fem specifically.

## Unconditional ASCII-art banner on import

**Symptom:** `import jax_fem` prints a multi-line pyfiglet banner to
stdout, which will corrupt anything expecting clean JSON/structured stdout
from a script that imports it.

**Fix:** redirect/filter stdout around the import if you need clean output
(this knowledge pack's own probe scripts had their captured `expected/*.json`
files hand-stripped of the banner -- see the top of each probe file).
