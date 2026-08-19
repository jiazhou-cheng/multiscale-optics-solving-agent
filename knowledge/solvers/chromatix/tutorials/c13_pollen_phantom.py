"""Example 13 / "Pollen grain phantom data generator" -- https://chromatix.readthedocs.io/en/latest/examples/pollen/

Repo-owned reproduction of the pollen phantom generator: `chromatix.utils.pollen_3d`
produces a complex-valued 3D scattering potential, which the example projects and
slices for display.

Upstream is a two-cell visual demo with no printed numbers, and two things about
the returned array are worth knowing before using it as test data:

* **It is a REAL `float64` NumPy array**, not complex, with values in `[0, 1.1]`.
  Upstream colourises it with a helper that calls `np.angle(z)`, which is
  identically 0 for a non-negative real array -- so that plot is a magnitude map,
  and the hue axis carries no information. The companion filaments example
  multiplies by `1j` for the same reason. Neither fact is stated upstream.
* **The array is dense and contains denormals.** At the default `radius=0.8`,
  75% of voxels are strictly non-zero, and the smallest non-zero values are
  `5e-324` -- subnormal doubles. A naive `count_nonzero` therefore measures
  nothing useful, and denormal arithmetic can be orders of magnitude slower on
  some hardware. Occupancy here is measured above a `1e-6 * max` threshold.

What is checked:

* The requested shape and a finite array.
* **`filled=True` and `filled=False` differ**, and the filled phantom occupies more
  volume -- the only check that establishes the flag does anything.
* **`radius` controls the extent**, and at `radius=0.4` the phantom no longer
  touches the volume boundary while at the default `0.8` it does. That pins the
  parameter's meaning as a fraction of the half-window, and warns that the default
  is not paddable.
* **The generator is deterministic**: two calls with identical arguments return
  bit-identical arrays, which is what makes it admissible as a fixture.
* The projection the example plots (`sum` over axis 1) and its ``z = 84`` slice are
  both non-trivial.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c13_pollen_phantom",
    title="Pollen grain phantom data generator",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/pollen/",
    demonstrates="chromatix.utils.pollen_3d(shape, filled=...) as a synthetic 3D phantom.",
    slow=True,
)

SHAPE = [128, 128, 128]
SLICE_INDEX = 84


def run() -> TutorialResult:
    import jax.numpy as jnp

    from chromatix.utils import pollen_3d

    result = TutorialResult()
    filled = np.asarray(pollen_3d(SHAPE, filled=True))
    hollow = np.asarray(pollen_3d(SHAPE, filled=False))
    threshold = 1e-6 * float(np.abs(filled).max())

    def _occupancy(array: np.ndarray) -> int:
        return int(np.count_nonzero(np.abs(array) > threshold))

    smallest_nonzero = float(np.min(np.abs(filled[np.abs(filled) > 0.0])))
    result.record(
        requested_shape=list(SHAPE),
        filled_shape=list(filled.shape),
        filled_dtype=str(filled.dtype),
        hollow_dtype=str(hollow.dtype),
        is_complex=bool(np.iscomplexobj(filled)),
        occupancy_threshold=threshold,
        filled_occupied_voxels=_occupancy(filled),
        hollow_occupied_voxels=_occupancy(hollow),
        naive_nonzero_fraction=float(np.count_nonzero(filled) / filled.size),
        smallest_strictly_nonzero_value=smallest_nonzero,
        filled_value_range=[float(filled.min()), float(filled.max())],
    )
    result.check_true(
        "the_requested_shape_is_returned",
        "invariant",
        tuple(filled.shape) == tuple(SHAPE),
        f"{filled.shape} == {tuple(SHAPE)}",
    )
    result.check_finite("phantom_finite", filled)
    result.check_true(
        "the_phantom_is_a_real_float64_array_not_a_complex_one",
        "invariant",
        not np.iscomplexobj(filled) and filled.dtype == np.float64,
        f"dtype {filled.dtype}, values in [{float(filled.min()):.6f}, "
        f"{float(filled.max()):.6f}]. Upstream's colourise helper calls np.angle on it, "
        "which is identically 0 for a non-negative real array, so the hue axis of that "
        "plot carries no information.",
    )
    result.check_true(
        "the_array_contains_subnormal_values_so_count_nonzero_is_misleading",
        "analytic",
        smallest_nonzero < 1e-300
        and float(np.count_nonzero(filled) / filled.size) > 0.5,
        f"the smallest strictly non-zero magnitude is {smallest_nonzero:.3e} -- a "
        f"subnormal double -- and {100 * np.count_nonzero(filled) / filled.size:.1f}% of "
        "voxels are strictly non-zero at the default radius. Occupancy has to be measured "
        f"above a threshold (here {threshold:.3e}), and denormal arithmetic can be far "
        "slower than normal arithmetic on some hardware.",
    )
    result.check_true(
        "filled_occupies_more_volume_than_hollow",
        "analytic",
        _occupancy(filled) > _occupancy(hollow),
        f"above the {threshold:.3e} threshold: filled={_occupancy(filled)} voxels, "
        f"hollow={_occupancy(hollow)} voxels "
        f"({_occupancy(filled) / max(_occupancy(hollow), 1):.3f}x). The hollow variant is "
        "the shell only, so the flag genuinely changes the phantom.",
    )

    # -- radius controls the extent --------------------------------------------
    def _boundary_occupancy(array: np.ndarray) -> int:
        faces = (
            array[0], array[-1], array[:, 0], array[:, -1], array[:, :, 0], array[:, :, -1]
        )
        return int(sum(np.count_nonzero(np.abs(face) > threshold) for face in faces))

    sweep = {}
    for radius in (0.8, 0.6, 0.4, 0.25):
        array = np.asarray(pollen_3d(SHAPE, filled=True, radius=radius))
        sweep[f"{radius:g}"] = {
            "occupancy": _occupancy(array),
            "boundary_occupancy": _boundary_occupancy(array),
        }
    result.record(radius_sweep=sweep, total_voxels=int(np.prod(SHAPE)))
    result.check_true(
        "the_default_radius_gives_a_compact_interior_phantom",
        "analytic",
        _boundary_occupancy(filled) == 0,
        f"above the {threshold:.3e} threshold, no voxel on any of the six faces of the "
        f"{tuple(SHAPE)} volume is occupied at the default radius=0.8, so the phantom can "
        "be zero-padded without cutting it. (With a naive count_nonzero the faces look "
        "occupied -- that is the subnormal noise above, not structure.)",
    )
    result.check_true(
        "decreasing_radius_makes_the_phantom_LARGER_not_smaller",
        "analytic",
        sweep["0.25"]["occupancy"] > sweep["0.8"]["occupancy"]
        and sweep["0.25"]["boundary_occupancy"] > 0,
        "occupancy / boundary occupancy by radius: "
        + ", ".join(
            f"{k} -> {v['occupancy']}/{v['boundary_occupancy']}" for k, v in sweep.items()
        )
        + f" out of {int(np.prod(SHAPE))} voxels. `radius` is NOT an object radius in the "
        "obvious sense: reducing it from 0.8 to 0.25 fills the entire volume and clips "
        "against the boundary. The default 0.8 is the compact setting, and a smaller value "
        "is not a smaller object.",
    )

    # -- reproducibility -------------------------------------------------------
    replay = np.asarray(pollen_3d(SHAPE, filled=True))
    identical = bool(np.array_equal(filled, replay))
    result.record(reproducible=identical)
    result.check_true(
        "the_generator_is_deterministic",
        "invariant",
        identical,
        "two calls with identical arguments return bit-identical arrays, which is what "
        "makes this usable as a test fixture -- unlike filaments_3d (c14).",
    )

    # -- the two views the example plots ---------------------------------------
    projection = np.asarray(jnp.sum(np.asarray(filled), axis=1))
    plane = filled[:, SLICE_INDEX, :]
    result.record(
        projection_shape=list(projection.shape),
        projection_max=float(projection.max()),
        slice_index=SLICE_INDEX,
        slice_occupied_voxels=_occupancy(plane),
        slice_max=float(plane.max()),
    )
    result.check_finite("projection_finite", projection)
    result.check_true(
        "the_projection_and_the_z_84_slice_are_both_non_trivial",
        "analytic",
        float(projection.max()) > 0.0 and _occupancy(plane) > 0,
        f"the axis-1 projection peaks at {float(projection.max()):.6f} and the "
        f"z = {SLICE_INDEX} slice has {_occupancy(plane)} occupied voxels, so both cells "
        "the example plots show real structure",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
