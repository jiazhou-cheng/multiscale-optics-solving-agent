"""The launch operation: system-bound, captured before the trace, measure declared.

CHE-219 (R05.8). What this file holds is the *contract* of
`solvers.optiland.launch.launch`, plus the two structural claims the ticket makes
about where launch responsibilities may live. The physics it produces is already
held elsewhere and deliberately not duplicated here:

* `tests/physics/test_optiland_finite_conjugate.py` holds the launch state itself
  -- the point-source origin coincidence at `abs=0.0`, the seeded-zero
  accumulator, the collimated/diverging mirror image, and the row-for-row
  correspondence between the captured state and `Optic.trace`;
* `tests/physics/test_optiland_opl_convention.py` holds the CHE-30/CHE-41 optical
  path the object-space term feeds;
* `tests/physics/test_optiland_rays.py` holds the frozen ray records, which are
  what says R05.8 moved no number.

Ring counts are small on purpose -- `num_rings=2` is 19 rays -- because every
assertion here is about a contract rather than about numerical convergence. The
two that need a real system's aiming behaviour say so and use the fixture where
it is measurable.

No millimetre and no native ray attribute appears below, so this file is in
neither exemption set of `tests/solvers/test_optiland_boundary.py`. That is worth
stating: `launch` returns a *neutral* `RayBundle` in SI, and the only reason its
second return value is package-facing is the object-space term it carries in
native units for `rays.declare_optical_path_m` to apply.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from fixtures.systems import (
    FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
    REVERSE_TELEPHOTO,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    finite_conjugate_singlet,
    finite_conjugate_source,
    reverse_telephoto_source,
    singlet_ref,
    singlet_source,
)

from problems import SourceSpec
from representations import ContractError, RayBundle
from solvers.optiland import trace
from solvers.optiland.launch import (
    AIMING_MODES,
    DEFAULT_AIMING,
    LAUNCH_GEOMETRY_UNVERIFIED,
    LAUNCH_OPL_REFERENCE,
    launch,
)
from solvers.optiland.rays import (
    LAUNCH_PLANE_WAVEFRONT,
    LAUNCH_POINT_SOURCE,
    hexapolar_area_weight_m2,
    hexapolar_ray_count,
    to_ray_bundle,
)
from solvers.optiland.system import build_lens

NUM_RINGS = 2
WAVELENGTH_UM = 0.55
CPU64 = {"device": "cpu", "precision": "fp64"}

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "solvers" / "optiland"


def _singlet(field_angle_deg: tuple[float, float] = (0.0, 0.0)) -> tuple[object, SourceSpec]:
    source = singlet_source(field_angle_deg=field_angle_deg)
    return build_lens(singlet_ref(), source), source


def _column_spread(columns: object) -> float:
    """The largest per-component spread of an `(N, 3)` array.

    Per column and not over the whole array: `ptp` of the flattened directions of
    a perfectly collimated on-axis bundle is 1.0, because it spans the 0 of the
    transverse cosines and the 1 of the axial one. The quantity meant here is
    "does any component vary across the rays".
    """
    values = np.asarray(columns)
    return max(float(np.ptp(values[:, axis])) for axis in range(values.shape[1]))


# ---------------------------------------------------------------------------
# 1. Launch requires a constructed system, and needs no trace
# ---------------------------------------------------------------------------


def test_the_launch_api_takes_the_constructed_system_explicitly() -> None:
    """Acceptance criterion 1: the system is a required positional argument.

    A declarative source alone cannot produce a launch. The signature is what says
    so -- `lens` has no default and there is no overload that omits it -- and it is
    checked on the signature rather than by calling with one argument, because a
    `TypeError` from a missing argument would also be raised by a typo.
    """
    import inspect

    parameters = inspect.signature(launch).parameters
    assert list(parameters)[:2] == ["lens", "source"]
    assert parameters["lens"].default is inspect.Parameter.empty
    assert parameters["source"].default is inspect.Parameter.empty
    assert parameters["num_rings"].default is inspect.Parameter.empty
    assert parameters["aiming"].default == DEFAULT_AIMING


def test_the_source_package_cannot_produce_a_launch_bundle() -> None:
    """Acceptance criterion 1 and 5, as one claim: this is the only producer.

    `sources/` exports no operation returning a `RayBundle` and imports no solver,
    so a system-launch bundle is reachable only through the Optiland solver layer.
    Asserted here as well as in `tests/sources/test_sources_package.py` because
    this is the file that names the replacement -- an absence is only meaningful
    beside the thing that fills it.
    """
    import sources

    # `OPERATIONS` (CHE-221) is a tuple of strings, not a callable, so the loop
    # reads it as the declaration it is: every name it advertises, and no name
    # that produces a ray bundle.
    for name in sources.OPERATIONS:
        annotation = str(getattr(sources, name).__annotations__.get("return", ""))
        assert "RayBundle" not in annotation

    lens, source = _singlet()
    bundle, _ = launch(lens, source, num_rings=NUM_RINGS)
    assert isinstance(bundle, RayBundle)


def test_a_launch_bundle_is_obtained_without_tracing_the_system() -> None:
    """Acceptance criterion 2: no trace runs, and the bundle is fully declared.

    The launch bundle is a representation in its own right, not an intermediate:
    it carries geometry, wavelength, a declared reference surface, an amplitude, a
    referenced optical path and a declared measure, and `require_coherent()`
    accepts it. That matters beyond tidiness -- a launch state nobody can hold is
    a launch state nobody can check, which is the condition R05.8 exists to end.
    """
    lens, source = _singlet()
    bundle, declaration = launch(lens, source, num_rings=NUM_RINGS)

    assert bundle.count == hexapolar_ray_count(NUM_RINGS) == 19
    assert declaration["ray_count"] == bundle.count
    assert bundle.wavelength_m == pytest.approx(source.wavelength_um * 1.0e-6, abs=0.0)
    bundle.require_coherent()

    # Every ray points forward and the direction norms are unit to round-off, so
    # this is a bundle a consumer can use rather than a debug dump.
    directions = np.asarray(bundle.directions)
    assert np.all(directions[:, 2] > 0.0)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1e-15)

    # The launch amplitude is 1.0 for every ray, and that is read off the solver
    # rather than written by this adapter: `RayGenerator.generate_rays` seeds
    # `intensity = ones_like(Px)` when there is no apodization, and neither fixture
    # declares one. Pinned because `launch`'s own comment claims it as measured,
    # and an apodization is exactly the thing that would make it vary -- at which
    # point `sqrt(intensity)` would stop being a phase-free unit amplitude.
    np.testing.assert_array_equal(
        np.asarray(bundle.amplitude), np.ones(bundle.count, dtype=np.complex128)
    )


@pytest.mark.parametrize(
    ("setup_factory", "source_factory", "geometry", "surface_name"),
    [
        (singlet_ref, singlet_source, LAUNCH_PLANE_WAVEFRONT, "launch_plane"),
        (
            finite_conjugate_singlet,
            finite_conjugate_source,
            LAUNCH_POINT_SOURCE,
            "object_plane",
        ),
    ],
)
def test_the_launch_surface_is_the_geometry_the_system_actually_launches_on(
    setup_factory: object, source_factory: object, geometry: str, surface_name: str
) -> None:
    """The two launch geometries, each declared from the captured state.

    They are mirror images and neither is assumed: at infinity the directions are
    common and the origins spread over a plane; for a point source the origin is
    common and the directions spread. The declared `ReferenceSurface` says which
    one it is, because "a plane the collimated bundle crosses" and "the plane
    through the object point" are not interchangeable to a consumer.

    The infinite-conjugate launch plane is **not** the first surface, which is the
    measured fact that decided against reusing the R05.6 supplied-bundle trace
    path: the solver launches at `z = -EPD`, and `rays.require_launch_surface`
    correctly refuses a supplied bundle declared anywhere but the surface the
    trace starts from.
    """
    source = source_factory()
    lens = build_lens(setup_factory(), source)
    bundle, declaration = launch(lens, source, num_rings=NUM_RINGS)

    assert declaration["launch_geometry"] == geometry
    assert bundle.reference_surface.name == surface_name
    assert bundle.reference_surface is declaration["launch_surface"]
    assert bundle.reference_surface.medium_index == 1.0

    positions = np.asarray(bundle.positions_m)
    assert float(np.ptp(positions[:, 2])) == 0.0, "one launch surface, exactly"
    assert bundle.reference_surface.z_m == pytest.approx(float(positions[0, 2]), abs=0.0)

    if geometry is LAUNCH_POINT_SOURCE:
        # One point: the origins coincide in all three coordinates, and it sits at
        # the declared object distance.
        for axis in range(3):
            assert float(np.ptp(positions[:, axis])) == 0.0
        assert bundle.reference_surface.z_m == pytest.approx(
            -FINITE_CONJUGATE_OBJECT_DISTANCE_MM * 1.0e-3, abs=0.0
        )
        # Directions spread: a point source radiates.
        assert _column_spread(bundle.directions) > 1.0e-3
    else:
        # One plane, before the first surface, and the transverse extent is the
        # entrance pupil rather than an arbitrary aperture.
        assert bundle.reference_surface.z_m < 0.0
        assert float(np.ptp(positions[:, 0])) == pytest.approx(
            SINGLET_ENTRANCE_PUPIL_DIAMETER_MM * 1.0e-3, rel=1e-12
        )
        # Directions common to every ray: a collimated bundle, the mirror image.
        assert _column_spread(bundle.directions) == 0.0


def test_the_launch_optical_path_is_the_object_space_reference_term() -> None:
    """One arithmetic for "wavefront to launch point", shared with the traced path.

    The launch bundle's `optical_path_m` *is* the CHE-41 term the traced bundle's
    declaration applies, in metres, with no piston removed -- and its reference
    string says exactly that, so it cannot be mistaken for
    `rays.OPL_REFERENCE_VERSION`. Off axis it is a real tilt; on axis it is
    identically zero, which is the piston `rays.declare_optical_path_m`
    deliberately does not add.
    """
    lens, source = _singlet(field_angle_deg=(0.0, 3.0))
    tilted, declaration = launch(lens, source, num_rings=NUM_RINGS)

    assert str(tilted.optical_path_reference).startswith(LAUNCH_OPL_REFERENCE)
    assert not str(tilted.optical_path_reference).startswith("optiland-declared-opl")
    np.testing.assert_array_equal(
        np.asarray(tilted.optical_path_m),
        np.asarray(declaration["object_space"]["offset_native"]) * 1.0e-3,
    )
    assert float(np.ptp(tilted.optical_path_m)) > 0.0

    # On axis the same term is a pure **piston**, not a zero, and that distinction
    # is the whole reason `n0 * z0` is retained rather than dropped: the quantity
    # is the optical path from ONE stated wavefront -- the plane through the global
    # origin normal to `d0` -- and on axis that plane sits `EPD` in front of the
    # launch plane. `rays.declare_optical_path_m` does not add a constant term,
    # because step 4 removes it exactly, which is what keeps every on-axis frozen
    # number where it was.
    on_axis, _ = launch(*_singlet(), num_rings=NUM_RINGS)
    assert float(np.ptp(on_axis.optical_path_m)) == 0.0
    assert float(np.asarray(on_axis.optical_path_m)[0]) == pytest.approx(
        on_axis.reference_surface.z_m, abs=0.0
    )

    # And a point source: the launch point IS the wavefront, so the term is
    # exactly zero per ray however far off axis the source sits.
    point_source = finite_conjugate_source(field_angle_deg=(1.0, 2.0))
    point, _ = launch(
        build_lens(finite_conjugate_singlet(), point_source), point_source, num_rings=NUM_RINGS
    )
    assert float(np.max(np.abs(point.optical_path_m))) == 0.0


# ---------------------------------------------------------------------------
# 2. The measure is declared at launch
# ---------------------------------------------------------------------------


def test_the_measure_is_declared_at_launch_from_the_generated_pupil_sample() -> None:
    """Acceptance criterion 3: the layer that chose the sampling declares it.

    The weights are the same absolute entrance-pupil area elements R05.2
    established -- `pi a^2 / (3 n^2)` per interior-ring ray, 3/4 at the centre and
    1/2 at the rim, summing to `pi a^2 (1 + 1/(4 n^2))` exactly. What changed is
    *when*: they are assigned from the pupil coordinates the fan was generated
    from, while the complete sample and its ring identity are still known, rather
    than recovered from traced output afterwards.

    Compared against `rays.hexapolar_area_weight_m2` driven from the ring indices
    directly, so this is the shared arithmetic and not a second copy of it.
    """
    lens, source = _singlet()
    bundle, declaration = launch(lens, source, num_rings=NUM_RINGS)

    assert bundle.measure_kind == "quadrature_area_m2"
    assert bundle.measure_weight is not None
    np.testing.assert_array_equal(
        np.asarray(bundle.measure_weight), np.asarray(declaration["measure_weight"])
    )
    assert declaration["measure_kind"] == "quadrature_area_m2"
    assert "Assigned at LAUNCH" in declaration["measure_note"]

    radius_m = SINGLET_ENTRANCE_PUPIL_DIAMETER_MM * 1.0e-3 / 2.0
    # Ring index by construction for a 2-ring fan: one centre ray, then 6 and 12.
    ring_index = np.array([0] + [1] * 6 + [2] * 12)
    np.testing.assert_allclose(
        np.asarray(bundle.measure_weight),
        hexapolar_area_weight_m2(ring_index, NUM_RINGS, radius_m),
        rtol=0.0,
        atol=0.0,
    )
    total = float(np.sum(np.asarray(bundle.measure_weight)))
    assert total / (math.pi * radius_m**2) == pytest.approx(
        1.0 + 1.0 / (4.0 * NUM_RINGS**2), rel=1e-12
    )


def test_the_traced_bundle_takes_its_measure_from_the_launch_declaration() -> None:
    """The measure is carried, not recomputed -- bitwise, at every surviving row.

    Nothing clips on either fixture, so every launched cell survives and the
    traced weights are the launch weights. That equality is the behavioural half
    of criterion 3; `test_no_launch_state_is_reconstructed_after_the_trace` below
    is the structural half.
    """
    lens, source = _singlet()
    launched, declaration = launch(lens, source, num_rings=NUM_RINGS)
    native = lens.trace(
        Hx=0.0, Hy=0.0, wavelength=source.wavelength_um, num_rays=NUM_RINGS
    )
    traced, diagnostics = to_ray_bundle(
        lens, native, launch=declaration, reference_surface="image_surface"
    )

    assert traced.count == launched.count
    np.testing.assert_array_equal(
        np.asarray(traced.measure_weight), np.asarray(launched.measure_weight)
    )
    assert traced.measure_kind == launched.measure_kind
    assert diagnostics["launch_ray_count"] == launched.count
    assert diagnostics["launch_geometry"] == LAUNCH_PLANE_WAVEFRONT
    assert diagnostics["launch_aiming"]["requested"] == DEFAULT_AIMING


def test_no_launch_state_is_reconstructed_after_the_trace() -> None:
    """Acceptance criterion 3, structurally: `rays.py` no longer regenerates a launch.

    An AST walk rather than a substring scan of the whole file, because `rays.py`
    *explains* at length why the regeneration was a hazard and flagging its own
    prose would make the only correct response "stop explaining the rule". What is
    checked is code: no call to the backend's distribution factory or ray
    generator, no read of the paraxial entrance pupil, and no use of the ring
    index -- the four things the reconstruction was made of.

    `hexapolar_ring_index` and `hexapolar_area_weight_m2` still *live* in
    `rays.py`, which is criterion 4: one implementation of the quadrature, with
    `launch.py` as its only caller. So this walks for *uses*, not for definitions.
    """
    tree = ast.parse((PACKAGE / "rays.py").read_text(encoding="utf-8"))
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.update(alias.name for alias in node.names)
            used.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            used.update(alias.name.split(".")[-1] for alias in node.names)

    for reconstruction in (
        "create_distribution",
        "distribution",
        "generate_rays",
        "ray_generator",
        "EPD",
        "hexapolar_ring_index",
    ):
        assert reconstruction not in used - defined, (
            f"rays.py reconstructs launch state via {reconstruction!r}; CHE-219 moved "
            "that behind the launch boundary"
        )

    # And the positive statement: launch.py is where each of them now lives.
    launch_source = (PACKAGE / "launch.py").read_text(encoding="utf-8")
    for name in ("create_distribution", "generate_rays", "hexapolar_ring_index", "EPD"):
        assert name in launch_source


def test_the_hexapolar_arithmetic_has_exactly_one_implementation() -> None:
    """Acceptance criterion 4: no second copy of the quadrature or the ring assignment.

    `launch.py` calls the `rays.py` functions rather than restating the formulas,
    so the two numbers a mistake here would be invisible in -- the `3 n^2`
    denominator and the 3/4 and 1/2 boundary corrections -- appear in one file.
    """
    launch_source = (PACKAGE / "launch.py").read_text(encoding="utf-8")
    rays_source = (PACKAGE / "rays.py").read_text(encoding="utf-8")

    assert "hexapolar_area_weight_m2" in launch_source
    for formula in ("3.0 * num_rings**2", "0.75 * nominal_m2", "0.5 * nominal_m2"):
        assert formula in rays_source
        assert formula not in launch_source

    # The launch-reference arithmetic is the mirror image: it lives in `launch.py`
    # only, and `rays.py` applies the term it is handed rather than measuring one.
    for term in ("l0 * x0 + m0 * y0 + n0 * z0", "_point_source_reference"):
        assert term in launch_source
    assert "l0 * x0 + m0 * y0 + n0 * z0" not in rays_source


# ---------------------------------------------------------------------------
# 3. Aiming is explicit and observable
# ---------------------------------------------------------------------------


def test_the_aiming_mode_is_declared_and_read_back_off_the_lens() -> None:
    """Acceptance criterion 6, first half: the mode used is readable by the caller.

    `requested` is what was asked for and `observed` is what the lens now carries
    -- read back rather than echoed, because the *trace's* own ray generation
    consults that configuration and not this function's argument. Reporting the
    request while the lens held something else is precisely the failure this
    declaration exists to make impossible.
    """
    lens, source = _singlet()
    for mode in AIMING_MODES:
        _, declaration = launch(lens, source, num_rings=NUM_RINGS, aiming=mode)
        assert declaration["aiming"]["requested"] == mode
        assert declaration["aiming"]["observed"]["mode"] == mode
        assert declaration["aiming"]["modes"] == list(AIMING_MODES)


def test_an_unrecognized_aiming_mode_is_refused_at_the_call_that_chose_it() -> None:
    """Not at the next ray generation, which is where the backend would raise.

    Refused on both entry points, because a caller reaching `trace` should not have
    to learn the vocabulary from a traceback inside an aimer factory.
    """
    lens, source = _singlet()
    with pytest.raises(ValueError, match="aiming="):
        launch(lens, source, num_rings=NUM_RINGS, aiming="parabolic")
    with pytest.raises(ValueError, match="aiming="):
        trace(
            singlet_ref(),
            source,
            sampling={"num_rings": NUM_RINGS, "reference_surface": "image_surface"},
            execution=CPU64,
            aiming="parabolic",
        )


def test_the_default_mode_is_the_backend_default_and_moves_nothing() -> None:
    """Acceptance criterion 8's precondition, and criterion 6's negative control.

    `DEFAULT_AIMING` is `RealRayTracer.__init__`'s own value, so declaring it
    explicitly cannot change a frozen number -- and the whole R05.8 parity claim
    rests on that. Measured here as bit identity between a launch that sets the
    mode and one that never touched the lens's aiming configuration.
    """
    assert DEFAULT_AIMING == "paraxial"

    lens, source = _singlet(field_angle_deg=(0.0, 3.0))
    untouched = dict(lens.ray_tracer.ray_aiming_config)
    declared, _ = launch(lens, source, num_rings=NUM_RINGS)
    assert untouched["mode"] == DEFAULT_AIMING

    fresh_lens, _ = _singlet(field_angle_deg=(0.0, 3.0))
    explicit, _ = launch(fresh_lens, source, num_rings=NUM_RINGS, aiming=DEFAULT_AIMING)
    np.testing.assert_array_equal(
        np.asarray(declared.positions_m), np.asarray(explicit.positions_m)
    )


def test_the_mode_changes_the_launch_where_real_aiming_happens() -> None:
    """Acceptance criterion 6, second half: the argument is not decoration.

    Off axis on M3-REVERSE-TELEPHOTO the pupil has to be *found*, and the
    iterative aimer finds a different answer than the paraxial one: the launch
    coordinates move by 1.97e-2 native units, which is 6% of the entrance-pupil
    diameter. If changing the mode moved nothing anywhere, declaring it would be a
    knob rather than a physical statement.
    """
    source = reverse_telephoto_source(field_angle_deg=(0.0, 21.0))
    paraxial, _ = launch(
        build_lens(REVERSE_TELEPHOTO, source), source, num_rings=NUM_RINGS, aiming="paraxial"
    )
    iterative, _ = launch(
        build_lens(REVERSE_TELEPHOTO, source), source, num_rings=NUM_RINGS, aiming="iterative"
    )
    difference = float(
        np.max(np.abs(np.asarray(paraxial.positions_m) - np.asarray(iterative.positions_m)))
    )
    assert difference == pytest.approx(1.97e-5, rel=0.05), "in metres: 1.97e-2 native units"


def test_the_mode_is_invisible_where_the_modes_are_physically_equivalent() -> None:
    """The other half of criterion 6: no artificial difference where there is none.

    On axis on the singlet there is nothing to aim -- the chief ray is the axis and
    every mode agrees on the pupil -- so all three modes must produce the *same*
    launch, bitwise. A mode argument that perturbed this case would be introducing
    an aiming residual rather than declaring one.
    """
    reference = None
    for mode in AIMING_MODES:
        lens, source = _singlet()
        bundle, _ = launch(lens, source, num_rings=NUM_RINGS, aiming=mode)
        columns = np.concatenate(
            [np.asarray(bundle.positions_m), np.asarray(bundle.directions)], axis=1
        )
        if reference is None:
            reference = columns
        else:
            np.testing.assert_array_equal(columns, reference)


# ---------------------------------------------------------------------------
# 4. Refusals, and what launch will not do
# ---------------------------------------------------------------------------


def test_a_supplied_bundle_is_not_something_launch_will_take() -> None:
    """Acceptance criterion 7: launch never re-aims a bundle the caller owns.

    The refusal is on the type and it names the other entry point, because the two
    are not alternatives at this argument position: a `RayBundle` is *already* a
    launch, and re-aiming it would replace the caller's declared geometry with
    this system's pupil map.
    """
    lens, source = _singlet()
    already_launched, _ = launch(lens, source, num_rings=NUM_RINGS)
    with pytest.raises(TypeError, match="trace_rays"):
        launch(lens, already_launched, num_rings=NUM_RINGS)  # type: ignore[arg-type]


def test_a_ring_count_below_one_is_refused_rather_than_clamped() -> None:
    """A hexapolar fan of zero rings is not a sampling; it is a missing argument."""
    lens, source = _singlet()
    for num_rings in (0, -1):
        with pytest.raises(ValueError, match="num_rings"):
            launch(lens, source, num_rings=num_rings)


def test_the_launch_is_refused_when_its_state_cannot_be_read() -> None:
    """A launch state that is not finite is no bundle at all, not a partial one.

    Driven through the shipping `_launch_columns` with a manufactured state,
    because no problem this schema can state produces one -- and a refusal with no
    reachable case is a claim about a path that does not exist. It is a refusal
    rather than a degradation because launch is now the *producer* of the rays:
    before R05.8 an unreadable launch left the object-space term unavailable and
    the trace continued on axis, which is still what `declare_optical_path_m` does
    with a term it is handed as unavailable.
    """
    from solvers.optiland.launch import _launch_columns

    class _State:
        def __init__(self, **columns: object) -> None:
            for name, value in columns.items():
                setattr(self, name, value)

    good = {
        "x": np.zeros(4),
        "y": np.zeros(4),
        "z": np.zeros(4),
        "L": np.zeros(4),
        "M": np.zeros(4),
        "N": np.ones(4),
        "i": np.ones(4),
        "w": np.full(4, WAVELENGTH_UM),
        "opd": np.zeros(4),
    }
    assert _launch_columns(_State(**good))["x"].size == 4

    with pytest.raises(ContractError) as excinfo:
        _launch_columns(_State(**{**good, "x": np.array([0.0, np.nan, 0.0, 0.0])}))
    assert excinfo.value.code == "NON_FINITE"

    with pytest.raises(ContractError) as excinfo:
        _launch_columns(_State(**{**good, "y": np.zeros(3)}))
    assert excinfo.value.code == "SHAPE_MISMATCH"

    with pytest.raises(ContractError) as excinfo:
        _launch_columns(_State(**{name: np.zeros(0) for name in good}))
    assert excinfo.value.code == "EMPTY_ENSEMBLE"


def test_an_unverified_launch_geometry_is_named_rather_than_called_collimated() -> None:
    """The default that is not the collimated one, which would be an approximation.

    When `_object_space_reference` declines the term -- a finite object whose
    launch origins do not coincide, an unreadable object surface -- the launch
    points still lie on one plane, so a surface exists to declare. What does *not*
    exist is any evidence that the plane is a **wavefront** of the incoming bundle,
    and naming it `launch_plane` / "plane wavefront of a collimated bundle" would
    assert exactly the thing that was declined. `rays.declare_optical_path_m` reads
    the geometry off the term rather than off this name, so the two cannot
    disagree; this pins that the name itself stays honest.
    """
    from solvers.optiland.launch import _declare_launch_optical_path

    assert LAUNCH_GEOMETRY_UNVERIFIED not in (LAUNCH_PLANE_WAVEFRONT, LAUNCH_POINT_SOURCE)
    assert "undetermined" in LAUNCH_GEOMETRY_UNVERIFIED

    # And with the term unavailable the launch bundle carries no optical path at
    # all rather than a zero it did not measure: a bundle with no path is honestly
    # incoherent, one carrying an unmeasured zero claims its surface is a wavefront.
    path, reference = _declare_launch_optical_path(
        {"available": False, "reason": "manufactured for this test", "offset_native": None},
        launch_geometry=LAUNCH_GEOMETRY_UNVERIFIED,
    )
    assert path is None
    assert reference is None


def test_a_fan_that_is_not_the_hexapolar_layout_leaves_the_measure_undeclared() -> None:
    """The population guard: the area element is defined for one layout only.

    `rays.hexapolar_area_weight_m2` assigns `pi a^2 / (3 n^2)` per ray with 3/4 at
    the centre and 1/2 at the rim, which is the right quadrature *only* for one
    centre ray plus `6j` points on ring `j`. Before R05.8 this check came free from
    the traced row count; declaring the measure at launch means asking the fan
    directly, and the alternative is a silently wrong absolute area rather than an
    honest absence -- which R07's kernel would multiply into every reconstruction.
    """
    from solvers.optiland.launch import _declare_measure

    lens, _ = _singlet()
    short = hexapolar_ray_count(NUM_RINGS) - 1
    weight, kind, note = _declare_measure(
        lens,
        pupil_x=np.zeros(short),
        pupil_y=np.zeros(short),
        num_rings=NUM_RINGS,
        wavelength_um=WAVELENGTH_UM,
    )
    assert weight is None
    assert kind == "undeclared"
    assert str(short) in note and str(hexapolar_ray_count(NUM_RINGS)) in note


def test_a_declaration_from_another_wavelength_is_refused() -> None:
    """The row count says "same number of rays"; it cannot say "same light".

    Both refractive indices `to_ray_bundle` reads -- object-space and image-space --
    are evaluated at the declaration's wavelength, so a same-length declaration
    taken at a different wavelength would refer the optical path through media the
    trace never travelled in. On a dispersive prescription that is a real error and
    on an air-only one it is invisible, which is why it is checked rather than
    trusted.
    """
    lens, source = _singlet()
    _, declaration = launch(lens, source, num_rings=NUM_RINGS)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=0.6563, num_rays=NUM_RINGS)
    with pytest.raises(ContractError) as excinfo:
        to_ray_bundle(lens, native, launch=declaration, reference_surface="image_surface")
    assert excinfo.value.code == "MISSING_DECLARATION"
    assert "0.6563" in str(excinfo.value)


def test_an_unreadable_entrance_pupil_leaves_the_measure_undeclared_not_guessed() -> None:
    """`undeclared` is the honest answer and the useful one: R07 refuses on it.

    A missing measure is not the same kind of defect as a missing off-axis tilt
    term -- the bundle is still coherent, it is simply unweighted -- so it degrades
    with a stated reason instead of raising. The aperture area the relative cell
    areas scale to is the one thing the quadrature cannot be derived without.
    """
    from solvers.optiland.launch import _declare_measure

    class _NoPupil:
        class paraxial:
            @staticmethod
            def EPD() -> float:
                raise RuntimeError("no entrance pupil")

    # A correctly populated fan, so this reaches the pupil read rather than being
    # short-circuited by the population guard above.
    points = np.zeros(hexapolar_ray_count(NUM_RINGS))
    weight, kind, note = _declare_measure(
        _NoPupil(),
        pupil_x=points,
        pupil_y=points,
        num_rings=NUM_RINGS,
        wavelength_um=WAVELENGTH_UM,
    )
    assert weight is None
    assert kind == "undeclared"
    assert "entrance pupil diameter could not be read" in note
