"""A supplied `RayBundle` through an Optiland system, with what it carries intact.

CHE-217 (R05.6). The bug this file exists for is invisible: `to_ray_bundle`
builds its amplitude as `sqrt(intensity)`, which is correct on the
solver-generated path and *only* there, because the pinned solver seeds
`intensity = ones_like` without apodization and clips by zeroing the row rather
than removing it. Reuse that output path for a bundle a caller supplied and the
per-ray coefficient `a_i` is silently replaced -- by `|a_i|` if the modulus were
bridged in as an intensity, dropping every radian of phase, or by `1` if it were
not. No error, no diagnostic, and no downstream intensity check that can see it.
`measure_weight` has the same shape of problem from the other direction: a
regenerated hexapolar pupil area element substituted for a caller's importance
weights rescales every reconstruction downstream by a factor nothing observes.

So every assertion about the amplitude and the measure below is written to be
**falsifiable against those two specific wrong implementations**, and not merely
against a crash:

* the moduli are distinct per ray and none is 1, so a `sqrt(intensity)`
  implementation passes every geometric check and fails only the amplitude
  identity;
* the phases are non-zero and distinct, so a modulus-only implementation fails
  even where the moduli happen to agree -- and `test_a_modulus_only_result_is_a_
  distinguishable_failure` states that as an explicit control, so the test is
  known to discriminate rather than assumed to;
* the weights are O(1) importance weights, while a pupil area element on this
  system is of order 1e-9 square metres, so a substituted quadrature is off by
  nine orders of magnitude rather than by a subtlety.

Lengths, and why they are literals here
---------------------------------------
This module is **not** exempt from `test_optiland_boundary.py`'s
millimetre rule, so it cannot name the prescription's own units, and every length
below is SI with its provenance in a comment. Same convention as
`tests/physics/ray_support.py`'s `WAVELENGTH_M` / `FOCAL_M`. Each literal is an
order-of-magnitude fact about M3-SINGLET-REF rather than a coupling to one of its
numbers: "comfortably inside the pupil" and "beyond the front face's radius of
curvature, so the surface cannot be intersected at all".

What is not asserted here
-------------------------
No physics oracle for the composed route. That the optical path convention is the
one it claims is `test_optiland_opl_convention.py`'s, already established, and an
oracle for `scalar_to_ray -> trace -> ray_to_scalar` is R07/R08 territory. What
this file establishes about the path is that it is a **composition**: the incoming
path is added rather than replaced, the reference says so, and a unit error of the
size that scales `k * OPL` by a thousand is caught by the ratio bound.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest
from fixtures.systems import singlet_ref
from test_optiland_boundary import code_tokens

from couplers import ray_to_scalar, scalar_to_ray
from representations import UNVERIFIED, ContractError, RayBundle, ReferenceSurface, ScalarField
from solvers.optiland import trace_rays
from solvers.optiland.rays import (
    COMPOSED_OPL_REFERENCE_VERSION,
    require_declared_optical_path,
)

CPU64 = {"device": "cpu", "precision": "fp64"}

#: The M3 reference wavelength, in metres. Same value `ray_support.WAVELENGTH_M`
#: carries, restated rather than imported so this file does not pull the coupler
#: test support into a solver test.
WAVELENGTH_M = 0.55e-6

#: Radii, in metres, comfortably inside M3-SINGLET-REF's entrance pupil, whose
#: semi-diameter is about 2.5e-4 m. Nothing here depends on being *at* the rim:
#: the fixture declares no physical clear aperture, and a surface aperture model
#: is R05.9's, so what these have to be is on the lens and near enough to the
#: axis that the trace is well conditioned.
INSIDE_RADII_M = (0.0, 5.0e-5, 1.0e-4, 1.5e-4, 2.0e-4)

#: Radii, in metres, **beyond the front face's radius of curvature** (2.5e-3 m).
#: A ray travelling parallel to the axis at a larger radius cannot intersect that
#: sphere at all, so the trace returns non-finite geometry for it. This is the
#: negative control for survival, and it is a geometric fact about a convex
#: surface rather than an aperture rule.
MISSING_RADII_M = (4.0e-3, 8.0e-3)


def a_bundle(
    *,
    radii_m: tuple[float, ...] = INSIDE_RADII_M,
    z_m: float = 0.0,
    medium_index: float = 1.0,
    optical_path_offset_m: float = 0.0,
    amplitude_scale: complex = 1.0,
) -> RayBundle:
    """A supplied bundle on the singlet's first surface, hand-built and declared.

    Deliberately nonuniform in every quantity this ticket is about. The moduli
    span 0.3 to 2.1 -- distinct per ray, none equal to 1 -- and the phases span
    two radians, so neither `sqrt(intensity)` nor `abs(amplitude)` can reproduce
    them. The weights span 1 to 5, which is an importance-weight scale and not an
    area element.

    `optical_path_offset_m` shifts the incoming path by a constant, which is what
    makes the composition testable as an addition rather than as a value.
    `amplitude_scale` multiplies it by a complex constant, which is what makes the
    phase testable independently of the modulus.
    """
    radii = np.asarray(radii_m, dtype=np.float64)
    count = int(radii.size)
    moduli = np.linspace(0.3, 2.1, count)
    phases = np.linspace(0.2, 2.2, count)
    return RayBundle(
        positions_m=np.column_stack([radii, np.zeros(count), np.full(count, z_m)]),
        directions=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(
            name="emitting surface", z_m=z_m, medium_index=medium_index
        ),
        amplitude=amplitude_scale * moduli * np.exp(1j * phases),
        optical_path_m=np.full(count, optical_path_offset_m),
        optical_path_reference="zero at the emitting surface; the accumulated path restarts here",
        measure_weight=np.linspace(1.0, 5.0, count),
        measure_kind="importance_weight",
    )


def a_coupled_bundle() -> RayBundle:
    """What `couplers.scalar_to_ray` produces, on the singlet's first surface.

    The grid pitch is 1e-5 m so that the largest propagating transverse direction
    cosine is about `lambda / (2 dx)`, i.e. 0.028: every mode of the enumeration is
    near-axis and the whole ensemble reaches the image surface. That keeps this a
    bookkeeping test rather than a study of what a grazing mode does to a lens.

    The launch points are a few 5e-5 m offsets, well inside the pupil, so the
    ensemble is `modes x launch points` rays emitted from real positions rather
    than all from the origin.
    """
    rng = np.random.default_rng(7)
    shape = (8, 8)
    spectrum = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    field = ScalarField(
        u=spectrum.astype(np.complex128),
        sample_pitch_m=(1.0e-5, 1.0e-5),
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="emitting surface", z_m=0.0, medium_index=1.0),
    )
    launch = np.array(
        [[0.0, 0.0], [5.0e-5, 0.0], [0.0, 5.0e-5], [-5.0e-5, -5.0e-5]], dtype=np.float64
    )
    rays, _ = scalar_to_ray(field, launch_positions_xy_m=launch)
    return rays


# ---------------------------------------------------------------------------
# 1. Criterion 1 -- a coupler's own output traces, and nothing regenerates it
# ---------------------------------------------------------------------------


def test_a_coupler_bundle_traces_and_returns_a_bundle() -> None:
    """Criterion 1: `scalar_to_ray` output in, `RayBundle` out."""
    rays = a_coupled_bundle()
    assert rays.count == 8 * 8 * 4
    assert rays.measure_kind == "importance_weight"

    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    assert isinstance(traced, RayBundle)
    # The row count is preserved, which is the whole correspondence claim: the
    # caller still holds the ensemble it handed over.
    assert traced.count == rays.count
    assert traced.wavelength_m == rays.wavelength_m
    assert traced.reference_surface.name == "image_surface"
    # Every array is a plain host buffer in the state the caller handed over, so
    # the untouched amplitude and the evolved geometry are one artifact.
    for name in ("positions_m", "directions", "amplitude", "optical_path_m", "measure_weight"):
        assert isinstance(getattr(traced, name), np.ndarray)


def test_the_bundle_path_needs_no_source_declaration_at_all() -> None:
    """CHE-218 (R05.7) acceptance criterion 2, stated as two absences.

    Before the split this path had to be handed a record that *required* a field
    angle and an object distance, so a caller holding only a `RayBundle` had to
    invent both for a lens to be built. Now: the setup carries no illumination
    field to fill in, and `trace_rays` has no source parameter to pass one
    through. Together those are the whole of "without being converted back into
    source parameters" -- there is nowhere for such a value to go.
    """
    import dataclasses as dc
    import inspect

    from problems import OpticalSetup, SourceSpec

    # Disjoint field sets: no name belongs to both records, so there is no
    # illumination value a setup could be asked for. Read off the dataclasses
    # rather than spelled out, which also keeps this file inside the boundary
    # gate's rule on prescription units.
    setup_fields = {f.name for f in dc.fields(OpticalSetup)}
    source_fields = {f.name for f in dc.fields(SourceSpec)}
    assert setup_fields and source_fields
    assert not (setup_fields & source_fields), (
        f"a field belongs to both records: {sorted(setup_fields & source_fields)}"
    )
    assert "field_angle_deg" not in setup_fields
    assert set(inspect.signature(trace_rays).parameters) == {"setup", "rays", "execution"}

    # And it runs on the coupler's own artifact, which has none of those to give.
    rays = a_coupled_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)
    assert traced.count == rays.count


#: Fragments naming a source specification the supplied-bundle path may not
#: construct. Matched as substrings of code tokens, the way
#: `test_optiland_boundary.py` matches its own rule, so a suffixed spelling
#: cannot slip past an exact-name list.
SOURCE_SPECIFICATION_FRAGMENTS = (
    "Hx",
    "Hy",
    "num_ray",
    "num_ring",
    "field_deg",
    "hexapolar",
    "distribution",
    "generate_rays",
    "object_distance",
    "entrance_pupil",
    "sqrt",
)


def test_no_source_specification_is_constructed_in_the_call_path() -> None:
    """Criterion 1's absence half, checked structurally rather than by inspection.

    A field angle, an object distance or a pupil sampling density appearing
    anywhere in this call path would mean the trace had regenerated the rays from
    a higher-level specification instead of consuming the representation it was
    given -- and `sqrt` would mean the amplitude had come back off the intensity.
    Docstrings are outside the rule for the same reason
    `test_optiland_boundary.py` exempts them: this module's own explanation of
    what it forbids is not a use of it.
    """
    from solvers.optiland import rays as rays_module
    from solvers.optiland import solver as solver_module

    functions = (
        solver_module.trace_rays,
        rays_module.to_native_rays,
        rays_module.to_traced_ray_bundle,
        rays_module.compose_optical_path_m,
        rays_module.require_launch_surface,
        rays_module.surface_positions_m,
    )
    offenders = [
        f"{function.__name__}: {token!r}"
        for function in functions
        for token in sorted(code_tokens(inspect.getsource(function)))
        if any(fragment in token for fragment in SOURCE_SPECIFICATION_FRAGMENTS)
    ]
    assert offenders == [], (
        "the supplied-bundle path names a source specification or reads the "
        "amplitude off the intensity:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. Criterion 2 -- the amplitude survives, modulus and phase
# ---------------------------------------------------------------------------


def test_a_nonuniform_complex_amplitude_survives_ray_for_ray() -> None:
    """Criterion 2, exactly and ray for ray."""
    rays = a_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    # Exact equality, not a tolerance: the amplitude never crossed into the
    # solver, so there is no arithmetic for it to have been rounded by.
    assert np.array_equal(traced.amplitude, rays.amplitude)
    assert traced.amplitude.dtype == rays.amplitude.dtype

    # The premises that make the assertion above falsifiable rather than vacuous.
    moduli = np.abs(rays.amplitude)
    assert len(set(moduli.tolist())) == rays.count, "the moduli must be distinct per ray"
    assert np.all(moduli != 1.0), "a modulus of 1 is where sqrt(intensity) is accidentally right"
    assert float(np.ptp(np.angle(rays.amplitude))) > 1.0, "the phase spread must be substantial"


def test_a_modulus_only_result_is_a_distinguishable_failure() -> None:
    """The control that proves the test above discriminates.

    Both wrong implementations are constructed explicitly and shown to differ
    from what the boundary returned. Without this, "the amplitude is equal to the
    input" proves nothing about the defect it exists for -- the two could have
    agreed on this fixture by accident.
    """
    rays = a_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    # `sqrt(intensity)`, i.e. the modulus, phase discarded.
    modulus_only = np.sqrt(np.abs(rays.amplitude) ** 2)
    assert not np.allclose(traced.amplitude, modulus_only)
    # `intensity = ones_like`, i.e. the amplitude replaced by 1.
    unit = np.ones_like(rays.amplitude)
    assert not np.allclose(traced.amplitude, unit)


def test_a_complex_scale_on_the_input_appears_exactly_on_the_output() -> None:
    """The phase, isolated from the modulus.

    Scaling the incoming amplitude by `c` must scale the outgoing one by `c`. A
    modulus-only implementation scales by `|c|` and passes every intensity check
    while doing it.
    """
    scale = 2.5 * math.cos(0.7) + 2.5j * math.sin(0.7)
    plain = trace_rays(singlet_ref(), a_bundle(), execution=CPU64)
    scaled = trace_rays(singlet_ref(), a_bundle(amplitude_scale=scale), execution=CPU64)

    assert np.allclose(scaled.amplitude, scale * plain.amplitude, rtol=0.0, atol=1e-15)
    # ...and the geometry did not notice, because an amplitude is not a ray.
    assert np.array_equal(scaled.positions_m, plain.positions_m)
    assert np.array_equal(scaled.optical_path_m, plain.optical_path_m)


# ---------------------------------------------------------------------------
# 3. Criterion 3 -- the measure survives, and is not a pupil quadrature
# ---------------------------------------------------------------------------


def test_a_nontrivial_measure_survives_unchanged() -> None:
    """Criterion 3: the weights and the kind are the caller's, both untouched."""
    rays = a_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    assert np.array_equal(traced.measure_weight, rays.measure_weight)
    assert traced.measure_kind == "importance_weight"
    assert traced.measure_kind != "undeclared"
    # Distinct per ray, so a uniform substitution is visible.
    assert len(set(traced.measure_weight.tolist())) == rays.count
    # And emphatically not an area element. A hexapolar cell on this system is
    # `pi a^2 / (3 n^2)` with `a` about 2.5e-4 m, i.e. of order 1e-9; these are
    # dimensionless O(1) importance weights and the two cannot be confused by
    # anything but a substitution.
    assert float(np.min(traced.measure_weight)) > 1.0e-3


def test_the_coupler_measure_crosses_the_trace_unchanged() -> None:
    """Criterion 3 on the artifact the ticket is actually about."""
    rays = a_coupled_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    assert np.array_equal(traced.measure_weight, rays.measure_weight)
    assert traced.measure_kind == rays.measure_kind == "importance_weight"


# ---------------------------------------------------------------------------
# 4. Criterion 4 -- survival, and nothing else
# ---------------------------------------------------------------------------


def test_only_the_rays_that_miss_a_surface_are_marked() -> None:
    """Criterion 4's negative control, with the surviving rays held to identity."""
    radii = INSIDE_RADII_M + MISSING_RADII_M
    rays = a_bundle(radii_m=radii)
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    assert traced.count == rays.count, "the row count is preserved, not filtered"
    marked = traced.amplitude == 0.0
    expected = np.array([radius in MISSING_RADII_M for radius in radii])
    assert np.array_equal(marked, expected), (
        "exactly the rays beyond the front face's radius of curvature must be marked"
    )

    # Every other ray is untouched in both quantities.
    survived = ~expected
    assert np.array_equal(traced.amplitude[survived], rays.amplitude[survived])
    assert np.array_equal(traced.measure_weight[survived], rays.measure_weight[survived])
    # The measure is a property of how the caller sampled, so a marked row keeps
    # its weight too: nothing about the trace restates it.
    assert np.array_equal(traced.measure_weight, rays.measure_weight)
    # A marked row is unreadable rather than merely wrong: the zeroed amplitude is
    # what makes its placeholder geometry contribute nothing to any sum.
    assert np.all(np.isfinite(traced.positions_m))
    assert np.all(np.isfinite(traced.optical_path_m))


def test_a_bundle_with_no_surviving_ray_is_refused() -> None:
    """Criterion 4's other end: an empty result is a refusal, not an empty bundle."""
    rays = a_bundle(radii_m=MISSING_RADII_M)
    with pytest.raises(ContractError) as raised:
        trace_rays(singlet_ref(), rays, execution=CPU64)
    assert raised.value.code == "EMPTY_ENSEMBLE"


# ---------------------------------------------------------------------------
# 5. Criterion 5 -- the launch surface, checked and not assumed
# ---------------------------------------------------------------------------


def test_a_launch_surface_that_is_not_where_the_trace_starts_is_refused() -> None:
    """Criterion 5. Getting this wrong is a different optical system, not a crash."""
    # One micrometre off the first traced surface: a thousand times the tolerance
    # and still small enough that no geometric check would notice.
    rays = a_bundle(z_m=1.0e-6)
    with pytest.raises(ContractError) as raised:
        trace_rays(singlet_ref(), rays, execution=CPU64)
    assert raised.value.code == "FRAME_MISMATCH"
    assert raised.value.declaration == "reference_surface"


def test_a_launch_medium_the_prescription_disagrees_with_is_refused() -> None:
    """The other half of "where the rays are": which medium they are in.

    The composed path is `incoming + n-weighted accumulator`, so the two halves
    have to be measured in one medium. A caller declaring glass at a surface the
    prescription puts in air gets a path wrong by the index ratio over the first
    transfer, and nothing downstream can attribute it back here.
    """
    rays = a_bundle(medium_index=1.5)
    with pytest.raises(ContractError) as raised:
        trace_rays(singlet_ref(), rays, execution=CPU64)
    assert raised.value.code == "MISSING_DECLARATION"


# ---------------------------------------------------------------------------
# 6. Criterion 6 -- the optical path is composed, and says so
# ---------------------------------------------------------------------------


def test_the_output_path_is_the_incoming_path_plus_this_trace() -> None:
    """Criterion 6: composition, asserted as an addition rather than as a value."""
    offset = 3.0e-3
    plain = trace_rays(singlet_ref(), a_bundle(), execution=CPU64)
    shifted = trace_rays(
        singlet_ref(), a_bundle(optical_path_offset_m=offset), execution=CPU64
    )

    # The incoming path is added, not replaced: shifting it by a constant shifts
    # the output by exactly that constant, and the increment this trace
    # contributed does not depend on the incoming declaration at all.
    assert np.allclose(shifted.optical_path_m - plain.optical_path_m, offset, atol=1e-15)

    # The increment is a physical optical path and is in metres. The geometric
    # axial distance from the launch surface to the image surface is the lower
    # bound -- the ray crosses glass, so its optical path is strictly longer --
    # and the ratio bound is what catches the unit error that scales `k * OPL` by
    # a thousand.
    increment = plain.optical_path_m
    geometric = plain.reference_surface.z_m - 0.0
    assert geometric > 0.0
    ratio = increment / geometric
    assert np.all(ratio > 1.0), "the path through glass exceeds the geometric distance"
    assert np.all(ratio < 2.0), "an increment in the wrong length unit would be 1000x this"


def test_the_composed_reference_names_the_incoming_one_and_passes_the_gate() -> None:
    """Criterion 6's declaration half, and the vocabulary extension that admits it."""
    rays = a_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)

    reference = traced.optical_path_reference
    assert reference is not None
    assert reference.startswith(COMPOSED_OPL_REFERENCE_VERSION)
    # It quotes the incoming reference rather than discarding it, so a consumer
    # can see whose zero the composed path is measured from.
    assert rays.optical_path_reference in reference
    # And the gate admits it, which is the extension this ticket made.
    require_declared_optical_path(traced)
    # `require_coherent()` is what a ray-to-wave coupler passes through, and it
    # must accept a composed path: the sign is the incoming bundle's, declared.
    traced.require_coherent()


def test_the_gate_still_refuses_a_bare_accumulator() -> None:
    """Criterion 6's other half: the vocabulary was extended, not loosened."""
    traced = trace_rays(singlet_ref(), a_bundle(), execution=CPU64)

    for reference in (
        "accumulated optical path from the trace",
        "optiland-composed",  # a near-miss on the new prefix
        UNVERIFIED,
    ):
        disguised = dataclasses.replace(traced, optical_path_reference=reference)
        with pytest.raises(ContractError) as raised:
            require_declared_optical_path(disguised)
        assert raised.value.code == "OPL_REFERENCE_UNVERIFIED"


# ---------------------------------------------------------------------------
# 7. What the entry point refuses before it traces
# ---------------------------------------------------------------------------


def test_a_bundle_that_is_not_coherent_is_refused() -> None:
    """No amplitude to carry across, or no path to compose onto."""
    rays = a_bundle()
    for name in ("amplitude", "optical_path_m"):
        incomplete = dataclasses.replace(
            rays,
            **{name: None},
            **({"optical_path_reference": None} if name == "optical_path_m" else {}),
        )
        with pytest.raises(ContractError) as raised:
            trace_rays(singlet_ref(), incomplete, execution=CPU64)
        assert raised.value.code == "COHERENT_STATE_INCOMPLETE"


def test_an_unverified_incoming_reference_is_refused_rather_than_laundered() -> None:
    """The refusal that matters most, because the alternative is not an error.

    Composing onto a path whose sign is not established and then labelling the sum
    with this package's own version prefix would turn an unverified path into an
    admissible one. A wrong sign conjugates the wavefront, and no intensity check
    downstream can tell.
    """
    rays = dataclasses.replace(a_bundle(), optical_path_reference=UNVERIFIED)
    with pytest.raises(ContractError) as raised:
        trace_rays(singlet_ref(), rays, execution=CPU64)
    assert raised.value.code == "OPL_REFERENCE_UNVERIFIED"


def test_the_second_argument_must_be_a_bundle() -> None:
    with pytest.raises(TypeError, match="RayBundle"):
        trace_rays(singlet_ref(), singlet_ref(), execution=CPU64)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "execution",
    [{"device": "cpu"}, {"device": "cpu", "precision": "fp64", "backend": "numpy"}],
)
def test_an_incomplete_execution_request_is_refused(execution: dict[str, str]) -> None:
    """The same argument contract `trace` has, on the same checker."""
    with pytest.raises(ValueError, match="execution="):
        trace_rays(singlet_ref(), a_bundle(), execution=execution)  # type: ignore[arg-type]


def test_an_inadmissible_precision_is_refused_before_the_solver_runs() -> None:
    with pytest.raises(ValueError) as raised:
        trace_rays(
            singlet_ref(), a_bundle(), execution={"device": "cpu", "precision": "fp16"}
        )
    assert getattr(raised.value, "code", None) == "UNSUPPORTED_DTYPE"


# ---------------------------------------------------------------------------
# 8. The composed node -- scalar_to_ray -> trace_rays -> ray_to_scalar
# ---------------------------------------------------------------------------


def test_the_composed_route_is_not_silently_rescaled() -> None:
    """`scalar_to_ray -> trace_rays -> ray_to_scalar`, end to end.

    The assertion is an **invariance**, not an oracle: the reconstruction is exactly
    linear in the incoming amplitude. That is what a rescaling breaks and what a
    physics oracle for this route -- R07/R08's -- is not needed to state.

    Both wrong implementations fail it, and for different reasons. `sqrt(intensity)`
    scales the reconstruction by `|c|` rather than `c`, so the phase is gone. A
    regenerated pupil quadrature replaces O(1) importance weights with area
    elements of order 1e-9, so the whole field is rescaled by nine orders of
    magnitude -- and, because it is a *constant* factor, it would survive every
    check that normalizes the peak.
    """
    scale = 2.5 * math.cos(0.7) + 2.5j * math.sin(0.7)
    grid = {"grid_shape": (16, 16), "sample_pitch_m": (1.0e-6, 1.0e-6)}

    rays = a_coupled_bundle()
    traced = trace_rays(singlet_ref(), rays, execution=CPU64)
    # The identity the whole ticket is about, on the coupler's own artifact.
    assert np.array_equal(traced.amplitude, rays.amplitude)
    assert np.array_equal(traced.measure_weight, rays.measure_weight)

    scaled_rays = dataclasses.replace(rays, amplitude=scale * rays.amplitude)
    scaled_traced = trace_rays(singlet_ref(), scaled_rays, execution=CPU64)
    assert np.array_equal(scaled_traced.positions_m, traced.positions_m)
    assert np.array_equal(scaled_traced.optical_path_m, traced.optical_path_m)

    field, _ = ray_to_scalar(traced, **grid)  # type: ignore[arg-type]
    scaled_field, _ = ray_to_scalar(scaled_traced, **grid)  # type: ignore[arg-type]

    reference = np.max(np.abs(field.u))
    assert reference > 0.0, "the reconstruction must carry light"
    assert np.allclose(scaled_field.u, scale * field.u, rtol=1e-12, atol=1e-12 * reference)
    # Falsifiable: a modulus-only trace would have given |c| times the field, and
    # |c| times is a different field because c is not real.
    assert not np.allclose(scaled_field.u, abs(scale) * field.u, atol=1e-3 * reference)
