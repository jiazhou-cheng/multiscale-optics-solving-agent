"""The Chromatix integration: adapter and the carrier-removed ASM.

`adapter.py` is the graph-facing `ModelAdapter`; `carrier_removed_asm.py` is the
propagation variant that removes the carrier phase before transferring, named
for what the physics is rather than for the milestone that added it.

Nothing here may import from `couplers/`. The boundary artifacts both sides
exchange live in `core/`.
"""
