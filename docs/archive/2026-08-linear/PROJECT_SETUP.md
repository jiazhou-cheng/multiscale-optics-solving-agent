# Linear Project Setup — Ray/Wave Vertical Slice

## Project

- Name: `Ray-Wave Vertical Slice`
- Goal: one verified Optiland -> project ray-wave coupler -> Chromatix forward pipeline.
- Success: reproducible PSF demo with explicit conventions, characterization tests, and honest gradient boundaries.

## Workflow States

- Backlog: idea not yet scoped.
- Ready: acceptance criteria and non-goals are complete.
- In Progress: one implementer owns the issue and worktree.
- In Review: PR and evidence are linked.
- Done: acceptance criteria pass and follow-ups are split into new issues.
- Blocked: dependency or scientific ambiguity is documented.

## Labels

- `area:context`
- `area:artifact-contract`
- `area:optiland`
- `area:ray-wave`
- `area:chromatix`
- `area:verification`
- `type:audit`
- `type:probe`
- `type:implementation`
- `type:test`
- `risk:convention`
- `risk:gradient`
- `risk:dependency`

## Agent Coordination

- Linear issue is the dynamic task contract for both agents.
- Assign one implementer per issue. Do not let Codex and Claude Code edit the same branch/worktree concurrently.
- Use the second agent as reviewer or as implementer of a different, non-overlapping issue.
- Every agent must read the issue before editing and post evidence back to the issue.
- One issue maps to one PR unless the issue is explicitly an audit with no code changes.

## MCP Setup

Claude Code:

```bash
claude mcp add --transport http linear-server https://mcp.linear.app/mcp
```

Then open Claude Code and run `/mcp` to authenticate.

Codex:

```bash
codex mcp add linear --url https://mcp.linear.app/mcp
codex mcp login linear
```

## Dispatch Prompt

Use the same message for either agent:

```text
Open Linear issue <ISSUE-ID>. Treat it as the task scope.
Read AGENTS.md and only the files linked by the issue.
Restate acceptance criteria and non-goals before editing.
Inspect git status and current tests.
Implement only the issue, run the specified checks, review the diff,
and post a concise result with evidence and follow-up issues back to Linear.
```

## Review Prompt

```text
Review the PR only against Linear issue <ISSUE-ID> and AGENTS.md.
Report: Must fix before merge, Should fix soon, Safe to merge.
Focus on acceptance-criteria gaps, scientific-contract errors, convention or gradient mistakes,
broken data flow, missing tests, and unrelated scope expansion.
```
