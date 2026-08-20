"""Repository-level tooling. Not part of the installed package.

`pyproject.toml` discovers packages under `src/` only, so this file exists purely
so `pytest -p scripts.pytest_resource_profile` can import the profiler plugin by
module path (CHE-64).
"""
