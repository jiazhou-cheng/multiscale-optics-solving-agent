# `tests_tutorial/` — the on-demand tutorial reproduction suite (CHE-67)

These are the repo-owned reproductions of upstream Optiland and Chromatix
tutorials. They are **active, maintained tests** — not archived — but they do not
run in the default suite.

## Run them

```bash
./run.sh pytest -q tests_tutorial          # the whole suite
make test-tutorial                         # same command, via the Makefile
./run.sh pytest -q tests_tutorial -m "not slow"        # the cheap half only
./run.sh pytest -q tests_tutorial/test_optiland_tutorials.py -k t28  # one reproduction
```

Expect **~33 minutes** and peaks near **6.2 GiB RSS** for the full suite (60
tests, 2003 s measured in CHE-64). `tests_tutorial/test_chromatix_tutorials.py`
alone is 1020 s; `test_optiland_tutorials.py` is 983 s.

## Why they are not in the default suite

They are 76% of the old suite's runtime while answering a different question from
the rest of the suite: *has the pinned third-party solver changed?*, not *is this
repository's physics right?* (CHE-64, `docs/archive/2026-08-testing/test_runtime_audit.md`). That
makes them a dependency-pin gate, so the useful cadence is a pin change — not
every commit.

**Run them when:**

- `docker/Dockerfile`, `docker/requirements.txt`, or a pin in `pyproject.toml`
  changes (this is the case they exist for);
- before or after a substantial Optiland/Chromatix integration change;
- as a periodic sweep, roughly weekly.

## How the exclusion works

`pyproject.toml` sets `testpaths = ["tests"]`, so a bare `pytest` — and every
documented tier command, which all omit an explicit path — never looks in this
directory. Naming the directory is the only way in, which is what makes the suite
on-demand rather than marker-gated: there is no `-m "not tutorial"` to remember,
so no forgotten flag can cost anyone 33 minutes. `norecursedirs` covers the one
gap `testpaths` leaves -- `pytest .`, where an explicit path argument would
otherwise override `testpaths` and sweep this directory up (CHE-67 measured
exactly that). It does not affect naming this directory yourself, since
`norecursedirs` is not applied to command-line paths.

The `tutorial` marker is still on every test here (module-level `pytestmark`), so
`-m tutorial` selects exactly this suite once you have pointed pytest at the
directory. `tests/test_suite_layout.py` fails if a tutorial test reappears under
`tests/` or if `testpaths` widens.

Refresh the recorded evidence these tests compare against with:

```bash
./run.sh python tests_tutorial/cases/optiland/run_all.py  --write-expected
./run.sh python tests_tutorial/cases/chromatix/run_all.py --write-expected
```

Known flake: `test_optiland_tutorials.py::test_tutorial_reproduction[t21_surface_roughness_scattering]`
fails roughly 1 run in 11 on an unseedable numba RNG (CHE-64 F1,
`docs/archive/2026-08-testing/test_runtime_audit.md`). It is a known-flaky reproduction, not a
regression signal on its own.
