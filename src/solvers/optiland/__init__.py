"""The Optiland integration: adapter, system builder, coherent trace.

Three layers of `docs/architecture/solver_layering.md`'s tier live here --
`adapter.py` is the graph-facing `ModelAdapter`, `builder.py` turns a canonical
`OpticalSystemSpec` into an Optiland system, and `coherent_trace.py` traces a
caller-supplied coherent ray population rather than being a second ray-trace
entry point.

Nothing here may import from `couplers/`. The boundary artifacts both sides
exchange live in `core/`.
"""
