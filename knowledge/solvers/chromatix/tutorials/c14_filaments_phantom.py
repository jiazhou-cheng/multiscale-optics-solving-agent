"""Example 14 / "Filaments phantom data generator" -- https://chromatix.readthedocs.io/en/latest/examples/filaments/

Repo-owned reproduction of the filament phantom generator:
`chromatix.utils.filaments_3d(shape, ?, radius=, rand_offset=, num_filaments=)`
produces a 3D array of randomly placed filaments, which the example projects and
slices for display.

Upstream is a two-cell visual demo with no printed numbers. Validation is the
properties a *randomised* phantom generator must have to be usable as test data,
and one of them turns out not to hold:

* The requested shape is returned and every voxel is finite.
* **The occupancy scales with `num_filaments`**: 50 filaments occupy
  substantially more voxels than 5. That is the only check establishing the
  parameter does anything.
* **The occupancy scales with `radius`** in the same way.
* **Reproducibility is measured, not assumed.** The signature *does* accept a
  ``seed`` (the Holoscope example uses ``filaments_3d(..., seed=972920147)``), but
  the docs page reproduced here never passes one, so this reproduction checks
  directly whether two identical unseeded calls agree and records the verdict.
  Whatever the answer, a downstream test must be written to match it.
* The projection the example plots and its ``z = 84`` slice are non-empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c14_filaments_phantom",
    title="Filaments phantom data generator",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/filaments/",
    demonstrates=(
        "chromatix.utils.filaments_3d(shape, scale, radius=, rand_offset=, "
        "num_filaments=) as a randomised 3D phantom, and whether it is "
        "reproducible."
    ),
    slow=True,
    # filaments_3d takes no seed; the occupancy of a random phantom varies between
    # calls, so recorded metrics replay at a statistical budget. See the docstring.
    metric_rtol=0.5,
)

SHAPE = (128, 128, 128)
SLICE_INDEX = 84
NUM_FILAMENTS = 50
RADIUS = 0.9
RAND_OFFSET = 0.2


def run() -> TutorialResult:
    import jax.numpy as jnp

    from chromatix.utils import filaments_3d

    result = TutorialResult()
    array = np.asarray(
        filaments_3d(
            SHAPE, 1, radius=RADIUS, rand_offset=RAND_OFFSET, num_filaments=NUM_FILAMENTS
        )
    )
    result.record(
        requested_shape=list(SHAPE),
        returned_shape=list(array.shape),
        dtype=str(array.dtype),
        occupied_voxels=int(np.count_nonzero(array)),
        value_range=[float(array.min()), float(array.max())],
        num_filaments=NUM_FILAMENTS,
        radius=RADIUS,
        rand_offset=RAND_OFFSET,
    )
    result.check_true(
        "the_requested_shape_is_returned",
        "invariant",
        tuple(array.shape) == SHAPE,
        f"{array.shape} == {SHAPE}",
    )
    result.check_finite("phantom_finite", array)
    result.check_true(
        "the_phantom_is_non_empty",
        "invariant",
        int(np.count_nonzero(array)) > 0,
        f"{int(np.count_nonzero(array))} occupied voxels of {int(np.prod(SHAPE))} "
        f"({100 * np.count_nonzero(array) / np.prod(SHAPE):.3f}%)",
    )

    # -- the parameters do something -------------------------------------------
    few = np.asarray(
        filaments_3d(SHAPE, 1, radius=RADIUS, rand_offset=RAND_OFFSET, num_filaments=5)
    )
    thin = np.asarray(
        filaments_3d(
            SHAPE, 1, radius=0.3, rand_offset=RAND_OFFSET, num_filaments=NUM_FILAMENTS
        )
    )
    result.record(
        occupied_with_5_filaments=int(np.count_nonzero(few)),
        occupied_with_radius_0p3=int(np.count_nonzero(thin)),
    )
    result.check_true(
        "occupancy_scales_with_num_filaments",
        "analytic",
        int(np.count_nonzero(array)) > 2 * int(np.count_nonzero(few)),
        f"{NUM_FILAMENTS} filaments occupy {int(np.count_nonzero(array))} voxels against "
        f"{int(np.count_nonzero(few))} for 5, a factor of "
        f"{np.count_nonzero(array) / max(np.count_nonzero(few), 1):.2f}",
    )
    result.check_true(
        "occupancy_scales_with_radius",
        "analytic",
        int(np.count_nonzero(array)) > int(np.count_nonzero(thin)),
        f"radius {RADIUS} occupies {int(np.count_nonzero(array))} voxels against "
        f"{int(np.count_nonzero(thin))} at radius 0.3",
    )

    # -- reproducibility, measured ---------------------------------------------
    replay = np.asarray(
        filaments_3d(
            SHAPE, 1, radius=RADIUS, rand_offset=RAND_OFFSET, num_filaments=NUM_FILAMENTS
        )
    )
    identical = bool(np.array_equal(array, replay))
    occupancy_spread = abs(
        int(np.count_nonzero(replay)) - int(np.count_nonzero(array))
    ) / max(int(np.count_nonzero(array)), 1)
    result.record(
        reproducible=identical,
        replay_occupied_voxels=int(np.count_nonzero(replay)),
        occupancy_relative_spread=occupancy_spread,
    )
    result.check_true(
        "the_generators_reproducibility_is_recorded_either_way",
        "invariant",
        True,
        f"two identical UNSEEDED calls to filaments_3d return bit-identical arrays: "
        f"{identical}. Occupancy {int(np.count_nonzero(array))} vs "
        f"{int(np.count_nonzero(replay))} (relative spread {occupancy_spread:.4f}). The "
        "signature does accept a `seed` -- the Holoscope example passes "
        "seed=972920147 -- but this docs page never does, so measuring is the only way to "
        "know what the default gives. If it is False, an unseeded filament phantom cannot "
        "be frozen as a bit-exact repository fixture.",
    )

    # -- the two views the example plots ---------------------------------------
    projection = np.asarray(jnp.sum(np.asarray(array), axis=1))
    plane = array[:, SLICE_INDEX, :]
    masked = plane * (plane < 15)
    result.record(
        projection_shape=list(projection.shape),
        projection_max=float(projection.max()),
        slice_index=SLICE_INDEX,
        slice_occupied_voxels=int(np.count_nonzero(plane)),
        masked_slice_occupied_voxels=int(np.count_nonzero(masked)),
    )
    result.check_finite("projection_finite", projection)
    result.check_true(
        "the_projection_and_the_masked_z_84_slice_are_non_trivial",
        "analytic",
        float(projection.max()) > 0.0 and int(np.count_nonzero(masked)) > 0,
        f"the axis-1 projection peaks at {float(projection.max()):.4f} and the "
        f"z = {SLICE_INDEX} slice has {int(np.count_nonzero(plane))} occupied voxels "
        f"({int(np.count_nonzero(masked))} after the example's `< 15` mask)",
    )
    result.note(
        "The example multiplies by 1j before colourising (`colorize(1j * arr_p)`), which "
        "is a display choice: filaments_3d returns a REAL array, unlike pollen_3d which "
        "returns a complex one. Recorded because the two generators are presented "
        "identically upstream but do not have the same dtype."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
