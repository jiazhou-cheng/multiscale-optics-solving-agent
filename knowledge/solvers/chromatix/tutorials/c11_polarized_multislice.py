"""Example 11 / "Scattering through 3D birefringent samples" -- https://chromatix.readthedocs.io/en/latest/examples/polarized_multislice/

Repo-owned reproduction of the birefringent multislice example: four uniaxial
beads with different crystal orientations in a 70 x 180 x 180 volume, illuminated
by x-polarized light and propagated with
`cf.polarized_multislice_thick_sample`.

Upstream prints one number -- `Sample shape: (70, 180, 180, 3, 3)` -- which is
reproduced. The rest of the validation is the vector-optics content, and it pins
two conventions this repository's `capability_notes.md` records but had not
executed:

* **`VectorField.u`'s last axis is ordered `(E_z, E_y, E_x)`**, the reverse of this
  project's `(E_x, E_y, E_z)`. Established here from upstream's own code -- it
  normalises by `field.u[90, 90, 2]` to set the *x* amplitude and labels
  `amplitude[..., 2]` as "Amplitude Ex" -- and then verified by measurement: the
  x-polarized input carries essentially all of its energy in component index 2 and
  none in 0 or 1.
* **Birefringence generates cross-polarization.** The input has `E_y = E_z = 0`, and
  after the sample both are non-zero -- that is the physics the example exists to
  show, and it is what distinguishes a genuinely tensorial propagation from a
  scalar one applied three times. Quantified as the fraction of output energy in
  the components the input did not have.
* **The four beads differ.** Their crystal orientations are four different rotation
  matrices, and the cross-polarized signal is not spatially uniform: measuring
  `|E_y|^2` in each bead's quadrant gives four different values, so the orientation
  really enters the calculation.
* The potential is Hermitian-symmetric per voxel (a physical permittivity tensor
  must be), the output is finite, and the field shape survives the sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _chromatix_harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c11_polarized_multislice",
    title="Scattering through 3D birefringent samples",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/polarized_multislice/",
    demonstrates=(
        "cf.polarized_multislice_thick_sample with a per-voxel 3x3 permittivity "
        "tensor, cf.linear / plane_wave(scalar=False) for a VectorField, "
        "chromatix.utils.utils.sigmoid_taper, and the (E_z, E_y, E_x) component "
        "order of VectorField.u."
    ),
    slow=True,
)

SIZE = (4.55, 11.7, 11.7)
SPACING = 0.065
WAVELENGTH = 0.405
N_BACKGROUND = 1.33
N_BEAD = (1.44, 1.40, 1.37)  # z, y, x
BEAD_RADIUS = 1.5
NA = 0.8
UPSTREAM_SAMPLE_SHAPE = (70, 180, 180, 3, 3)


def _rotations():
    import jax.numpy as jnp

    def rx(theta):
        return jnp.array(
            [
                [jnp.cos(theta), jnp.sin(theta), 0],
                [-jnp.sin(theta), jnp.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def ry(theta):
        return jnp.array(
            [
                [jnp.cos(theta), 0, -jnp.sin(theta)],
                [0, 1, 0],
                [jnp.sin(theta), 0, jnp.cos(theta)],
            ]
        )

    def rz(theta):
        return jnp.array(
            [
                [1, 0, 0],
                [0, jnp.cos(theta), jnp.sin(theta)],
                [0, -jnp.sin(theta), jnp.cos(theta)],
            ]
        )

    def rotation(theta_z, theta_y, theta_x):
        return rz(theta_z) @ ry(theta_y) @ rx(theta_x)

    return rotation


def build_potential():
    """Upstream's `paper_sample()`: four differently-oriented uniaxial beads."""
    import jax.numpy as jnp

    rotation = _rotations()
    k0 = 2 * jnp.pi / WAVELENGTH
    shape = np.around(np.array(SIZE) / SPACING).astype(int)
    z = jnp.linspace(SPACING / 2, SIZE[0] - SPACING / 2, shape[0])
    y = jnp.linspace(SIZE[1] - SPACING / 2, SPACING / 2, shape[1])
    x = jnp.linspace(SPACING / 2, SIZE[2] - SPACING / 2, shape[2])
    grid = jnp.stack(jnp.meshgrid(z, y, x, indexing="ij"), axis=-1)

    bead_positions = jnp.array(
        [
            [SIZE[0] / 2, 8.85, 2.85],
            [SIZE[0] / 2, 8.85, 8.85],
            [SIZE[0] / 2, 2.85, 2.85],
            [SIZE[0] / 2, 2.85, 8.85],
        ]
    )
    orientations = jnp.array(
        [
            [0.0, jnp.pi / 2, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, jnp.pi / 2],
            [jnp.pi / 4, jnp.pi / 4, jnp.pi / 4],
        ]
    )
    permittivity = jnp.zeros((*shape, 3, 3))
    background = jnp.eye(3) * N_BACKGROUND**2
    for position, orientation in zip(bead_positions, orientations, strict=True):
        bead = rotation(*orientation).T @ jnp.diag(jnp.array(N_BEAD) ** 2) @ rotation(*orientation)
        mask = jnp.sum((grid - position) ** 2, axis=-1) < BEAD_RADIUS**2
        permittivity += k0**2 * jnp.where(
            mask[..., None, None], background - bead, jnp.zeros((3, 3))
        )
    return permittivity, bead_positions


def run() -> TutorialResult:
    import jax.numpy as jnp

    import chromatix.functional as cf
    from chromatix.utils.utils import sigmoid_taper

    result = TutorialResult()
    potential, bead_positions = build_potential()
    potential_np = np.asarray(potential)
    result.record(
        sample_shape=list(potential_np.shape),
        sample_dtype=str(potential_np.dtype),
        n_bead_zyx=list(N_BEAD),
        n_background=N_BACKGROUND,
        bead_radius=BEAD_RADIUS,
        wavelength=WAVELENGTH,
        spacing=SPACING,
    )
    result.check_true(
        "the_sample_shape_matches_upstream",
        "reference",
        tuple(potential_np.shape) == UPSTREAM_SAMPLE_SHAPE,
        f"{tuple(potential_np.shape)}; upstream prints {UPSTREAM_SAMPLE_SHAPE}",
    )
    result.check_finite("potential_finite", potential_np)

    # -- a permittivity tensor must be symmetric per voxel ----------------------
    transpose_error = float(
        np.max(np.abs(potential_np - np.swapaxes(potential_np, -1, -2)))
    )
    scale = float(np.max(np.abs(potential_np)))
    result.record(
        potential_max_abs=scale,
        potential_symmetry_error=transpose_error,
        potential_relative_symmetry_error=transpose_error / scale,
    )
    result.check_true(
        "the_permittivity_tensor_is_symmetric_at_every_voxel",
        "analytic",
        transpose_error / scale < 1e-6,
        f"max |eps - eps^T| = {transpose_error:.3e} against a tensor scale of {scale:.3e} "
        f"(relative {transpose_error / scale:.3e}). R^T diag(n^2) R is symmetric for any "
        "rotation R, so this checks the rotation composition as well as the assembly.",
    )
    off_diagonal = float(
        np.max(np.abs(potential_np - np.einsum("...ij,ij->...ij", potential_np, np.eye(3))))
    )
    result.record(potential_max_off_diagonal=off_diagonal)
    result.check_true(
        "the_rotated_beads_produce_genuinely_off_diagonal_tensors",
        "analytic",
        off_diagonal / scale > 0.01,
        f"max off-diagonal magnitude {off_diagonal:.3e}, i.e. "
        f"{off_diagonal / scale * 100:.2f}% of the tensor scale. A diagonal-only potential "
        "would be three independent scalar problems; the rotated beads are not.",
    )

    # -- the x-polarized VectorField ------------------------------------------
    field = cf.plane_wave(
        (potential_np.shape[1], potential_np.shape[2]),
        SPACING,
        WAVELENGTH,
        amplitude=cf.linear(0),
        scalar=False,
    )
    field = field.replace(u=field.u / field.u[90, 90, 2])
    field = field.replace(
        u=field.u * sigmoid_taper(field.spatial_shape, 4)[..., jnp.newaxis]
    )
    input_u = np.asarray(field.u)
    component_energy_in = [
        float(np.sum(np.abs(input_u[..., i]) ** 2)) for i in range(3)
    ]
    total_in = sum(component_energy_in)
    result.record(
        field_class=type(field).__name__,
        field_u_shape=list(input_u.shape),
        input_component_energy_fraction=[e / total_in for e in component_energy_in],
    )
    result.check_true(
        "the_field_is_a_VectorField_with_a_trailing_three_component_axis",
        "invariant",
        type(field).__name__.startswith("Vector") and input_u.shape[-1] == 3,
        f"type {type(field).__name__}, u shape {input_u.shape}",
    )
    result.check_true(
        "vectorfield_u_is_ordered_E_z_E_y_E_x",
        "analytic",
        component_energy_in[2] / total_in > 0.999
        and component_energy_in[0] / total_in < 1e-6
        and component_energy_in[1] / total_in < 1e-6,
        "for cf.linear(0) -- x-polarized light -- the energy fractions per component index "
        f"are {[round(e / total_in, 9) for e in component_energy_in]}: essentially all of "
        "it sits at index 2. Combined with upstream's own code (it normalises by "
        "field.u[90, 90, 2] to set the x amplitude and labels amplitude[..., 2] as "
        "'Amplitude Ex'), the last axis is (E_z, E_y, E_x) -- the REVERSE of this "
        "project's (E_x, E_y, E_z) convention. Any coupler must transpose it.",
    )

    # -- the propagation -------------------------------------------------------
    out_field = cf.polarized_multislice_thick_sample(
        field, potential, N_BACKGROUND, SPACING, NA=NA
    )
    amplitude = np.asarray(out_field.amplitude, dtype=float)
    phase = np.asarray(out_field.phase, dtype=float)
    output_u = np.asarray(out_field.u)
    component_energy_out = [
        float(np.sum(np.abs(output_u[..., i]) ** 2)) for i in range(3)
    ]
    total_out = sum(component_energy_out)
    result.record(
        output_u_shape=list(output_u.shape),
        output_amplitude_range=[float(amplitude.min()), float(amplitude.max())],
        output_phase_range=[float(phase.min()), float(phase.max())],
        output_component_energy_fraction=[e / total_out for e in component_energy_out],
        output_over_input_total_energy=total_out / total_in,
    )
    result.check_finite("output_finite", np.concatenate([amplitude.ravel(), phase.ravel()]))
    result.check_true(
        "the_field_shape_survives_the_sample",
        "invariant",
        output_u.shape == input_u.shape,
        f"{output_u.shape} == {input_u.shape}",
    )
    result.check_true(
        "the_phase_is_wrapped_into_minus_pi_to_pi",
        "invariant",
        -np.pi - 1e-6 <= float(phase.min()) and float(phase.max()) <= np.pi + 1e-6,
        f"phase in [{float(phase.min()):.6f}, {float(phase.max()):.6f}], which is why "
        "upstream plots it with vmin=-pi, vmax=pi",
    )
    cross_polarized_fraction = (
        component_energy_out[0] + component_energy_out[1]
    ) / total_out
    result.record(cross_polarized_energy_fraction=cross_polarized_fraction)
    result.check_true(
        "the_birefringent_sample_generates_cross_polarization",
        "analytic",
        cross_polarized_fraction > 1e-4,
        f"the input carried {(component_energy_in[0] + component_energy_in[1]) / total_in:.3e} "
        f"of its energy outside E_x; the output carries {cross_polarized_fraction:.6f}. "
        "That conversion is the physics this example exists to show, and it is what "
        "distinguishes a genuinely tensorial propagation from a scalar one applied three "
        "times.",
    )

    # -- the four beads are not equivalent -------------------------------------
    height, width = output_u.shape[0], output_u.shape[1]
    quadrants = {
        "top_left": (slice(0, height // 2), slice(0, width // 2)),
        "top_right": (slice(0, height // 2), slice(width // 2, width)),
        "bottom_left": (slice(height // 2, height), slice(0, width // 2)),
        "bottom_right": (slice(height // 2, height), slice(width // 2, width)),
    }
    per_bead = {
        name: float(np.mean(np.abs(output_u[rows, columns, 1]) ** 2))
        for name, (rows, columns) in quadrants.items()
    }
    values = list(per_bead.values())
    result.record(mean_Ey_intensity_per_quadrant=per_bead)
    result.check_true(
        "the_four_crystal_orientations_give_four_different_responses",
        "analytic",
        max(values) > 2.0 * min(values),
        "mean |E_y|^2 per quadrant: "
        + ", ".join(f"{k} {v:.6e}" for k, v in per_bead.items())
        + f" -- a spread of {max(values) / max(min(values), 1e-30):.1f}x. The four beads "
        "differ only in their rotation matrices, so the orientation really enters the "
        "calculation rather than being averaged away.",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
