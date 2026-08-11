# Ray/Wave Agent Context Pack

This pack replaces duplicated, always-loaded agent documentation with:

- one canonical `AGENTS.md` for repository-wide rules;
- a one-line `CLAUDE.md` importing `AGENTS.md`;
- a machine-readable context manifest;
- concise, task-linkable current-scope documents;
- the live Linear project as the task source of truth;
- a static synchronization check and a reference-only historical archive.

Copy the files into the repository root, preserving paths. Review names and test commands against the actual codebase before merging.

Run:

```bash
python scripts/check_context_sync.py
```

Then create the Linear project and begin with `RW-001`, the read-only repository audit.
