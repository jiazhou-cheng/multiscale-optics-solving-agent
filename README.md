# Ray/Wave Agent Context Pack

This pack replaces duplicated, always-loaded agent documentation with:

- one canonical `AGENTS.md` for repository-wide rules;
- a one-line `CLAUDE.md` importing `AGENTS.md`;
- a machine-readable context manifest;
- concise, task-linkable current-scope documents;
- the live Linear project as the task source of truth;
- a static synchronization check and a reference-only historical archive.

## Execution is container-only

`./run.sh` is the only supported entry point for executing project code. It runs
the command inside the `agent_solver` Docker image with the repository mounted at
`/workspace`, which is also the container working directory, so source edits are
visible immediately to the editable install.

```bash
./run.sh                              # existing image, then interactive shell
./run.sh pytest -q                    # existing image, then test in container
./run.sh python path/to/probe.py      # existing image, then run probe in container
./run.sh --no-build pytest -q         # do not rebuild the image (the default)
./run.sh --rebuild pytest -q          # rebuild the image (cached), then test
```

Do **not** run project commands such as `python`, `pip`, `pytest`, `ruff`, or
`mypy` directly on the host. Host-side work is limited to editing files,
Git/Linear operations, and invoking Docker through `run.sh`. Rebuild with
`--rebuild` after changing `docker/Dockerfile` or dependency files, or when the
image does not exist. If a required check cannot run through `./run.sh`, report a
structured environment/setup failure instead of falling back to the host.

`run.sh` allocates a pseudo-TTY only for interactive callers. CI jobs and agent
shells can invoke it directly. The container runs with the invoking host user's
UID and GID so files written through the workspace mount remain host-writable.

## Checking the shared agent context

```bash
./run.sh python scripts/check_context_sync.py
./run.sh pytest -q tests/test_context_sync.py
```

The check enforces that `AGENTS.md` is the canonical static context, that
`CLAUDE.md` only imports it, that every path named in `CONTEXT_MANIFEST.yaml`
exists, and that the container-only rule and its documented `./run.sh` flags
match `run.sh`.

Work is tracked in Linear; the repository-wide rules live in `AGENTS.md`.
