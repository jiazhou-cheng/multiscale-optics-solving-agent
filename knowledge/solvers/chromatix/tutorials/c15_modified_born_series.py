"""Example 15 / "Modified Born Series" -- https://chromatix.readthedocs.io/en/latest/examples/modified_born/

Repo-owned reproduction of the modified-Born-series example from
`chromatix.experimental.modified_born_series`: a 2D dielectric disc
(`n = 1.5` in `n = 1.0` background) at `lambda / 8` sampling with absorbing
boundaries, solved by the fixed-point `solve()`.

**This is the only full-wave solver in Chromatix**, and it lives under
`chromatix.experimental`. Upstream's page truncates after the sample-construction
cell, so the solver call here is written from the pinned signature
(`solve(sample, source, initial_E=None, maxiter=1000, tol=1e-3,
implicit_diff=True, use_bicgstab=False)`) rather than copied.

Validation is analytic, because a full-wave solver on a *homogeneous* domain has a
closed-form answer:

* **The empty-sample control.** With `permittivity = n_background^2` everywhere and
  a plane-wave source, the solution must be a plane wave: comparing the solved
  field's phase gradient along the propagation axis against `k0 * n_background`
  gives a direct check of the dispersion relation, with no free parameters.
* **The sample is assembled correctly.** The real permittivity is exactly
  `n_background^2` outside the disc and `n_material^2` inside, and the imaginary
  part is non-zero **only** outside `Sample.ROI` -- so the boundary condition does
  not contaminate the region of interest. (`add_absorbing_bc` *pads* the sample:
  a `[256, 341, 1]` grid becomes `(320, 405, 1)` at `thickness=2.0`, and
  `Sample.ROI` is the tuple of slices that recovers the original region. Indexing
  the padded array with an un-padded mask raises an `IndexError`.)
* **The disc scatters.** The solved field inside the disc differs substantially
  from the empty-sample solution, and the difference is concentrated on the disc
  rather than spread over the domain.
* **The absorbing boundary works**: field amplitude at the domain edge is far below
  the amplitude in the interior, so the solution is not dominated by wrap-around.
* **`solve()`'s docstring is wrong about its own layout.** It says "the first
  (left-most) axis the polarization vector"; the return is actually
  `(*spatial, 3)`, component-last -- and the *input* current density must be
  component-last too, or the solver raises a broadcasting `TypeError`.
* **Both solver paths agree.** The memory-efficient fixed-point iteration
  (`use_bicgstab=False`) and BiCGStab (`True`) are different algorithms for the same
  linear system and must converge to the same field; measured agreement is
  recorded. That is the strongest available check with no external oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, pin_jax_precision, standalone_main

pin_jax_precision()

TUTORIAL = TutorialMeta(
    slug="c15_modified_born_series",
    title="Modified Born Series",
    level="advanced",
    url="https://chromatix.readthedocs.io/en/latest/examples/modified_born/",
    demonstrates=(
        "chromatix.experimental.modified_born_series: EmptySample, Source, "
        "add_absorbing_bc(sample, axis, thickness, max_extinction) and "
        "solve(sample, source, maxiter, tol, use_bicgstab) -- the only full-wave "
        "solver in Chromatix."
    ),
    slow=True,
)

WAVELENGTH = 0.5
MATERIAL_INDEX = 1.5
N_BACKGROUND = 1.0
GRID_SHAPE = [256, 256 * 4 // 3, 1]
BC_THICKNESS = 2.0
MAX_EXTINCTION = 0.25


def build_samples():
    """Returns (sample with the disc, empty control sample, disc mask, radius)."""
    import jax.numpy as jnp

    from chromatix.experimental.modified_born_series.sample import (
        EmptySample,
        add_absorbing_bc,
    )

    spacing = WAVELENGTH / 8
    base = EmptySample(GRID_SHAPE, spacing)
    contrast = MATERIAL_INDEX - N_BACKGROUND
    radius = (base.extent[0] / 2) / 2
    mask = jnp.sum(base.grid**2, axis=-1) < radius**2
    refractive_index = N_BACKGROUND + contrast * mask
    with_disc = add_absorbing_bc(
        base.replace(permittivity=refractive_index**2),
        axis=(0, 1),
        thickness=BC_THICKNESS,
        max_extinction=MAX_EXTINCTION,
    )
    empty = add_absorbing_bc(
        EmptySample(GRID_SHAPE, spacing).replace(
            permittivity=jnp.full(mask.shape, N_BACKGROUND**2)
        ),
        axis=(0, 1),
        thickness=BC_THICKNESS,
        max_extinction=MAX_EXTINCTION,
    )
    return with_disc, empty, np.asarray(mask), float(radius), float(spacing)


def build_source(sample):
    """A plane-wave line source just inside the entrance face, along axis 0.

    Upstream's page truncates before any source cell, so this is this repository's
    construction: a polarization-first ``(3, *spatial)`` array on the PADDED shape,
    with a uniform x-polarized line placed a few voxels inside ``Sample.ROI``.

    Two API details, both found by trying the obvious thing first:

    * ``Source`` takes a **current density**, not a field. It stores
      ``field = -1j/k0 * 1e-6 * c * mu_0 * current_density``, and passing
      ``field=`` raises ``TypeError``.
    * The current density must be **component-last** ``(*spatial, 3)``. The
      solver's ``split_trans_long_ft`` multiplies it by a ``(*spatial, 3)``
      k-grid, so a polarization-first array raises a broadcasting ``TypeError``
      -- even though ``solve()``'s docstring describes its *return* as
      polarization-first.
    """
    import jax.numpy as jnp

    from chromatix.experimental.modified_born_series.sample import Source

    k0 = 2 * jnp.pi / WAVELENGTH
    # The solver's internal split_trans_long_ft multiplies by a (*spatial, 3)
    # k-grid, so the source current density must be COMPONENT-LAST -- despite
    # solve()'s docstring describing its RETURN as polarization-first.
    current_density = jnp.zeros((*sample.permittivity.shape, 3), dtype=jnp.complex64)
    entrance = sample.ROI[0].start + 4
    current_density = current_density.at[entrance, sample.ROI[1], :, 2].set(1.0 + 0.0j)
    # Source takes a CURRENT DENSITY, not a field: it stores
    # field = -1j/k0 * 1e-6 * c * mu_0 * current_density. Passing `field=` raises.
    return Source(current_density=current_density, k0=k0)


def run() -> TutorialResult:
    import jax.numpy as jnp

    from chromatix.experimental.modified_born_series.solver import solve

    result = TutorialResult()
    with_disc, empty, mask, radius, spacing = build_samples()
    permittivity = np.asarray(with_disc.permittivity)
    roi = with_disc.ROI
    permittivity_roi = permittivity[roi]

    result.record(
        wavelength=WAVELENGTH,
        material_refractive_index=MATERIAL_INDEX,
        n_background=N_BACKGROUND,
        grid_shape=list(GRID_SHAPE),
        spacing=spacing,
        object_radius=radius,
        permittivity_shape=list(permittivity.shape),
        roi_shape=list(permittivity_roi.shape),
        roi_slices=[[sl.start, sl.stop] for sl in roi],
        permittivity_dtype=str(permittivity.dtype),
        disc_voxel_fraction=float(mask.mean()),
        bc_thickness=BC_THICKNESS,
        max_extinction=MAX_EXTINCTION,
    )
    result.check_finite("permittivity_finite", np.abs(permittivity))
    result.check_true(
        "add_absorbing_bc_pads_the_sample_and_records_the_region_of_interest",
        "invariant",
        permittivity.shape[0] > GRID_SHAPE[0]
        and tuple(permittivity_roi.shape) == tuple(mask.shape),
        f"the requested {tuple(GRID_SHAPE)} grid becomes {permittivity.shape} after "
        f"add_absorbing_bc(thickness={BC_THICKNESS}), and Sample.ROI = "
        f"{[(sl.start, sl.stop) for sl in roi]} recovers the original "
        f"{tuple(permittivity_roi.shape)}. Indexing the padded array with an un-padded "
        "mask raises IndexError, which is the mistake this check exists to prevent.",
    )

    # -- the sample is assembled correctly -------------------------------------
    real_inside = permittivity_roi.real[mask]
    real_outside = permittivity_roi.real[~mask]
    result.record(
        permittivity_real_inside_disc=[float(real_inside.min()), float(real_inside.max())],
        permittivity_real_outside_disc=[float(real_outside.min()), float(real_outside.max())],
        expected_inside=MATERIAL_INDEX**2,
        expected_outside=N_BACKGROUND**2,
    )
    result.check_true(
        "the_real_permittivity_is_exactly_n_squared_on_both_sides",
        "analytic",
        abs(float(real_inside.min()) - MATERIAL_INDEX**2) < 1e-5
        and abs(float(real_inside.max()) - MATERIAL_INDEX**2) < 1e-5
        and abs(float(real_outside.min()) - N_BACKGROUND**2) < 1e-5
        and abs(float(real_outside.max()) - N_BACKGROUND**2) < 1e-5,
        f"inside the disc Re(eps) in [{float(real_inside.min()):.6f}, "
        f"{float(real_inside.max()):.6f}] against n^2 = {MATERIAL_INDEX**2}; outside, "
        f"[{float(real_outside.min()):.6f}, {float(real_outside.max()):.6f}] against "
        f"{N_BACKGROUND**2}",
    )
    imaginary = permittivity.imag
    imaginary_roi = permittivity_roi.imag
    result.record(
        max_imag_in_roi=float(np.max(np.abs(imaginary_roi))),
        max_imag_overall=float(np.max(np.abs(imaginary))),
    )
    result.check_true(
        "the_absorbing_boundary_does_not_reach_the_region_of_interest",
        "analytic",
        float(np.max(np.abs(imaginary_roi))) < 1e-9
        and float(np.max(np.abs(imaginary))) > 1e-3,
        f"Im(eps) is at most {float(np.max(np.abs(imaginary_roi))):.3e} inside Sample.ROI "
        f"and reaches {float(np.max(np.abs(imaginary))):.6f} in the pad: the extinction is "
        "confined to the padding, so it cannot contaminate the physics.",
    )

    # -- the empty-sample control has a closed-form answer ----------------------
    source = build_source(empty)
    empty_solution = np.asarray(solve(empty, source, maxiter=400, tol=1e-4))
    result.record(
        solution_shape=list(empty_solution.shape),
        solution_dtype=str(empty_solution.dtype),
    )
    result.check_finite("empty_solution_finite", np.abs(empty_solution))
    component_energy = [
        float(np.sum(np.abs(empty_solution[..., i]) ** 2)) for i in range(3)
    ]
    result.record(solution_component_energy=component_energy)
    result.check_true(
        "the_solver_returns_a_component_LAST_field_contradicting_its_docstring",
        "invariant",
        empty_solution.shape[-1] == 3
        and tuple(empty_solution.shape[:-1]) == tuple(permittivity.shape),
        f"solve() returns {empty_solution.shape}, i.e. the spatial dimensions "
        f"{tuple(permittivity.shape)} followed by a trailing 3-component axis. Its own "
        "docstring says \"the first (left-most) axis the polarization vector\" -- that is "
        "wrong, and so is the input layout: the source current density must ALSO be "
        "component-last or the solver raises a broadcasting TypeError.",
    )
    result.check_true(
        "an_x_polarized_source_produces_energy_only_in_component_index_2",
        "analytic",
        component_energy[2] > 0.0
        and component_energy[0] == 0.0
        and component_energy[1] == 0.0,
        f"per-component energies {component_energy}: the same (E_z, E_y, E_x) ordering "
        "c11 and c12 establish for VectorField.u also holds for this solver's output, so "
        "the two halves of Chromatix at least agree with each other.",
    )

    # Phase gradient along the propagation axis must be k0 * n_background.
    component = empty_solution[:, :, 0, 2]
    entrance = roi[0].start + 12
    exit_plane = roi[0].stop - 12
    centre = (roi[1].start + roi[1].stop) // 2
    line = component[entrance:exit_plane, centre]
    phase = np.unwrap(np.angle(line))
    measured_k = float(np.polyfit(np.arange(line.size) * spacing, phase, 1)[0])
    expected_k = 2 * np.pi / WAVELENGTH * N_BACKGROUND
    result.record(
        measured_axial_wavenumber=abs(measured_k),
        expected_axial_wavenumber=expected_k,
        axial_wavenumber_relative_error=abs(abs(measured_k) - expected_k) / expected_k,
        phase_fit_samples=int(line.size),
    )
    result.check_close(
        "the_empty_sample_solution_propagates_at_k0_times_n_background",
        "analytic",
        abs(measured_k),
        expected_k,
        rel=0.02,
    )

    # -- the disc scatters -----------------------------------------------------
    disc_solution = np.asarray(solve(with_disc, build_source(with_disc), maxiter=400, tol=1e-4))
    result.check_finite("disc_solution_finite", np.abs(disc_solution))
    difference = (
        np.abs(disc_solution[roi][:, :, 0, 2]) - np.abs(empty_solution[roi][:, :, 0, 2])
    )
    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    inside_change = float(np.mean(np.abs(difference[mask_2d])))
    outside_change = float(np.mean(np.abs(difference[~mask_2d])))
    reference_amplitude = float(np.mean(np.abs(empty_solution[roi][..., 2])))
    result.record(
        mean_abs_amplitude_change_inside_disc=inside_change,
        mean_abs_amplitude_change_outside_disc=outside_change,
        interior_reference_amplitude=reference_amplitude,
        inside_over_outside_change=inside_change / max(outside_change, 1e-30),
    )
    result.check_true(
        "the_dielectric_disc_changes_the_field_and_does_so_locally",
        "analytic",
        inside_change > 0.05 * reference_amplitude
        and inside_change > 1.5 * outside_change,
        f"mean |amplitude change| relative to the empty control is {inside_change:.6f} "
        f"inside the disc and {outside_change:.6f} outside, against an interior reference "
        f"amplitude of {reference_amplitude:.6f}. The scattering is real and concentrated "
        "on the object rather than spread over the domain.",
    )

    # -- the absorbing boundary suppresses the edge ----------------------------
    edge = float(np.mean(np.abs(disc_solution[:2, :, 0, 2])))  # outermost pad rows
    result.record(
        mean_edge_amplitude=edge,
        edge_over_interior=edge / max(reference_amplitude, 1e-30),
    )
    result.check_true(
        "the_absorbing_boundary_suppresses_the_domain_edge",
        "analytic",
        edge < 0.5 * reference_amplitude,
        f"mean amplitude on the first two rows of the domain is {edge:.6e} against an "
        f"interior mean of {reference_amplitude:.6e}, a ratio of "
        f"{edge / max(reference_amplitude, 1e-30):.4f}: the boundary condition is "
        "absorbing, so the solution is not dominated by wrap-around",
    )

    # -- the two solver algorithms must agree ----------------------------------
    bicgstab = np.asarray(
        solve(with_disc, build_source(with_disc), maxiter=400, tol=1e-4, use_bicgstab=True)
    )
    a = disc_solution[roi][..., 2]
    b = bicgstab[roi][..., 2]
    scale = float(np.sqrt(np.mean(np.abs(a) ** 2)))
    relative_difference = float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / scale)
    correlation = float(
        np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    )
    result.record(
        fixed_point_vs_bicgstab_relative_rms=relative_difference,
        fixed_point_vs_bicgstab_overlap=correlation,
    )
    result.check_finite("bicgstab_solution_finite", np.abs(bicgstab))
    result.check_true(
        "the_fixed_point_and_bicgstab_solvers_agree",
        "analytic",
        correlation > 0.99,
        f"complex overlap {correlation:.6f} and relative RMS difference "
        f"{relative_difference:.3e} between the memory-efficient "
        "fixed-point iteration and BiCGStab over Sample.ROI. These are different algorithms for the same "
        "linear system, so agreement is the strongest check available without an external "
        "oracle -- and neither upstream nor the docstring asserts it.",
    )
    result.note(
        "chromatix.experimental.modified_born_series is the only full-wave solver in "
        "Chromatix and it lives under `experimental`. Upstream's example page truncates "
        "after the sample-construction cell, so the solve() call in this reproduction is "
        "written from the pinned signature rather than copied. The source construction "
        "(a polarization-first (3, *spatial) array with a line source at the entrance "
        "face) is this repository's choice; upstream shows no source cell."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
