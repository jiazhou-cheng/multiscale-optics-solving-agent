# Module Granularity — Working Rules

Do not use one meaning of “module” for graph design, adapter design, and Python file layout.

## 1. Physics-Graph Node

A graph node should represent a meaningful approximation or solver execution boundary.

Create a node when:

- It solves a distinct physical approximation.
- It can be run and verified independently.
- It consumes and produces stable typed artifacts.
- Its validity, cost, or derivative semantics differ from adjacent work.

Do not create a node merely because an object exists in the lens system.

### Current Decision

- One full sequential Optiland trace to a declared pupil/reference plane is one node.
- Each lens surface remains internal to that node.
- Chromatix propagation is a second node.
- Ray-to-wave reconstruction is a coupler edge.

### When a Surface Becomes a Node or Boundary

Promote a surface only when at least one is true:

- It is simulated by another model, such as RCWA/FDTD for a metasurface.
- It has a reusable input/output artifact contract.
- It introduces a separately measurable approximation or information-loss boundary.
- It must be scheduled, cached, differentiated, or verified independently.

Otherwise, surface-as-node is too fine-grained and exposes solver internals as architecture.

## 2. Adapter Capability

An adapter should expose a narrow, stable operation rather than mirror an entire external package.

Initial Optiland capabilities:

- Build or load one deterministic lens system.
- Trace a ray batch to a named plane.
- Export ray/pupil data into project-owned artifacts.

Initial Chromatix capabilities:

- Construct a field from a project-owned `ComplexField` artifact.
- Run one selected propagation method.
- Produce field and PSF outputs with diagnostics.

Do not wrap every analysis, surface class, optimizer, or visualization API.

## 3. Python Module or Class

Organize code around one cohesive scientific responsibility.

Good reasons to split:

- Different external dependencies.
- Different artifact contracts.
- Different validation or failure behavior.
- A unit cannot be tested without unrelated setup.
- Changes repeatedly cause conflicts between independent workstreams.

Bad reasons to split:

- One class per noun.
- One file per optical surface.
- An arbitrary line-count target without a contract boundary.

## 4. Issue and PR Size

A Linear issue should yield one independently reviewable result, such as:

- An executable package probe.
- A typed artifact contract.
- A characterized legacy coupler.
- One adapter capability.
- One verification case.
- One end-to-end forward benchmark.

Do not combine repository cleanup, architecture redesign, adapter implementation, and benchmark completion in one issue.

## 5. Granularity Review

After the first end-to-end demo, review boundaries using evidence:

- Which modules changed together?
- Which artifacts were reused?
- Which tests were expensive or difficult to isolate?
- Which boundaries created conversion noise without scientific value?
- Which internal components now need independent fidelity, caching, or verification?

Record the result as an architecture decision, not as an implicit refactor.
