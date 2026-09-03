"""`backends.optiland.spot_diagram`: the delegation, its boundary, and its refusals.

CHE-226 (R16). This path is a *delegation*, so the tests are not about the
arithmetic -- reimplementing the metrics here in order to check them would be
reimplementing the analysis the module exists not to reimplement. What is tested
is everything the delegation is responsible for:

* the pinned numbers, so a version bump or a wiring change is visible
  (`optiland 0.6.0`, recorded on the result itself);
* the units, because Optiland is millimetres and this project is metres, and a
  factor of 1000 in a spot radius is entirely plausible-looking;
* an **independent** physical check of the off-axis centroid against the paraxial
  image height `f tan(theta)`, which is not shared code with anything here;
* the boundary: no `Optic`, no `RealRays`, no `SpotData` in the result;
* the refusals -- a finite-conjugate source, a `RayBundle` handed in as a source,
  an unusable ring count, a misspelled execution declaration;
* that no rendering is reachable from this path.

One test is **characterization and not a gate**, and it says so: the native and
the project-owned spot metrics agree on the same system, because for an unapodized
launch fan every ray has intensity 1 and the intensity-weighted definitions reduce
to the unweighted ones. That agreement is evidence about a coincidence of
definitions. The correctness oracle for `measurements.spot_diagram` is the closed
form in `tests/physics/test_spot_diagram.py`, per `AGENTS.md`: this project's
numerics must not be gated on shared-input agreement with the code they wrap.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.systems import (
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    finite_conjugate_singlet,
    finite_conjugate_source,
    singlet_ref,
    singlet_source,
)

from backends.optiland import spot_diagram, trace
from backends.optiland.analysis import NATIVE_ANALYSIS, NATIVE_SPOT_METRIC_DEFINITIONS
from measurements import spot_diagram as measure_spot

EXECUTION: dict[str, str] = {"device": "cpu", "precision": "fp64"}

#: The frozen on-axis analysis of the R05 reference singlet, `optiland 0.6.0`,
#: fp64 on the host, 4 hexapolar rings. `1 + 3n(n + 1) = 61` rays.
ON_AXIS_RING_COUNT = 4
ON_AXIS_RAYS = 61
ON_AXIS_RMS_M = 4.868145748216336e-07
ON_AXIS_GEOMETRIC_M = 7.267701944603559e-07

#: The same at 5 degrees with 3 rings (`1 + 3*3*4 = 37` rays).
OFF_AXIS_FIELD_DEG = 5.0
OFF_AXIS_RING_COUNT = 3
OFF_AXIS_RAYS = 37
OFF_AXIS_RMS_M = 2.6637460500087058e-06
OFF_AXIS_CENTROID_Y_M = 4.229826147671514e-04


def _on_axis() -> Any:
    return spot_diagram(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        num_rings=ON_AXIS_RING_COUNT,
        execution=EXECUTION,
    )


def test_the_frozen_on_axis_analysis_reproduces() -> None:
    """The pinned regression, and the ray count that says the sampling reached the solver."""
    result = _on_axis()

    assert result.x_m.shape == (ON_AXIS_RAYS,)
    assert result.rms_radius_m == pytest.approx(ON_AXIS_RMS_M, rel=1e-12)
    assert result.geometric_radius_m == pytest.approx(ON_AXIS_GEOMETRIC_M, rel=1e-12)
    # An on-axis spot is centred on the axis, to round-off.
    assert result.centroid_m[0] == pytest.approx(0.0, abs=1e-15)
    assert result.centroid_m[1] == pytest.approx(0.0, abs=1e-15)
    assert result.num_rings == ON_AXIS_RING_COUNT
    assert (result.fields_analyzed, result.wavelengths_analyzed) == (1, 1)


def test_the_result_is_in_metres_and_not_the_solver_s_native_length_unit() -> None:
    """The unit conversion, checked by scale rather than by reading the source.

    A sub-micron RMS spot for an f/9.7 singlet is metres; the same number left in the
    solver's native length unit would be a thousand times larger, i.e. 5e-4 m, which
    is a plausible-looking spot radius for a bad lens. The band is wide on purpose --
    this is a unit check, not a second copy of the frozen value above.

    (The native unit is not named in this file: `test_optiland_boundary.py` forbids
    that outside the package, and this test is not one of its exemptions.)
    """
    result = _on_axis()
    assert 1e-9 < result.rms_radius_m < 1e-4
    assert result.wavelength_m == pytest.approx(0.55e-6, rel=1e-12)


def test_the_off_axis_centroid_is_the_paraxial_image_height() -> None:
    """`f tan(theta)`, an independent physical check of the units AND the sign.

    Not shared code with anything in this project: the effective focal length comes
    from the fixture's own thin-lens arithmetic and the tangent from `math`. The
    tolerance is 0.2%, which is real-ray distortion plus the thin-lens
    approximation of the fixture's EFL -- it is not a fit, and it is far tighter
    than the factor this test exists to catch.
    """
    result = spot_diagram(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, OFF_AXIS_FIELD_DEG)),
        num_rings=OFF_AXIS_RING_COUNT,
        execution=EXECUTION,
    )

    paraxial_m = (
        SINGLET_EFFECTIVE_FOCAL_LENGTH_MM * math.tan(math.radians(OFF_AXIS_FIELD_DEG)) * 1e-3
    )
    assert result.centroid_m[1] == pytest.approx(paraxial_m, rel=2e-3)
    assert result.centroid_m[1] > 0.0
    assert result.centroid_m[0] == pytest.approx(0.0, abs=1e-15)
    assert result.x_m.shape == (OFF_AXIS_RAYS,)
    assert result.rms_radius_m == pytest.approx(OFF_AXIS_RMS_M, rel=1e-12)
    assert result.centroid_m[1] == pytest.approx(OFF_AXIS_CENTROID_Y_M, rel=1e-12)


def test_the_provenance_of_the_numbers_is_on_the_record() -> None:
    """The fixture metadata R16 asks for, on the artifact rather than in a test file."""
    import optiland

    result = _on_axis()
    assert result.analysis == NATIVE_ANALYSIS == "optiland.analysis.SpotDiagram"
    assert result.mode == "native"
    assert result.optiland_version == str(optiland.__version__)
    assert result.metric_definitions == NATIVE_SPOT_METRIC_DEFINITIONS
    # The native definitions are *different* from this project's, and the record
    # carries them for exactly that reason: both are called "RMS spot radius".
    assert "unweighted" in result.metric_definitions["rms_radius_m"]
    assert "NOT necessarily about centroid_m" in result.metric_definitions["rms_radius_m"]


def test_no_native_object_crosses_the_boundary() -> None:
    """Every field is a number, a string, a tuple, a host array or a mapping.

    The AST walk in `test_optiland_boundary.py` proves no module outside this
    package imports optiland. This is the other half: that the *values* handed out
    are not native ones, which no import check can see.
    """
    result = _on_axis()
    for name, value in vars(result).items():
        assert isinstance(value, int | float | str | tuple | dict | np.ndarray), (name, type(value))
        assert type(value).__module__ in ("builtins", "numpy"), (name, type(value))


def test_a_finite_conjugate_source_is_refused_as_unsupported() -> None:
    """The scope boundary of this ticket, and it is a refusal rather than a reading.

    At a finite object distance `field_angle_deg` is a *position*
    (`problems.SourceSpec`), so handing it to the solver's `angle` field type would
    analyse a different object than the one declared -- silently, with a plausible
    spot. `NotImplementedError` and not `ValueError`: the source is well formed and
    this path does not support it.
    """
    with pytest.raises(NotImplementedError) as error:
        spot_diagram(
            finite_conjugate_singlet(),
            finite_conjugate_source(),
            num_rings=3,
            execution=EXECUTION,
        )

    message = str(error.value)
    assert "infinite-conjugate" in message
    assert "object_distance" in message
    # And it points at the path that *can* take it.
    assert "measurements.spot_diagram" in message


def test_a_ray_bundle_is_not_a_source_for_this_path() -> None:
    """The other half of the two-paths rule, as a type refusal.

    A caller holding rays must not be able to feed them to an analysis that
    generates its own -- the rays would be silently discarded and the spot would be
    of a fan the caller never asked for.
    """
    rays = trace(
        singlet_ref(),
        singlet_source(field_angle_deg=(0.0, 0.0)),
        sampling={"num_rings": 3, "reference_surface": "image_surface"},
        execution=EXECUTION,
    )
    with pytest.raises(TypeError) as error:
        spot_diagram(singlet_ref(), rays, num_rings=3, execution=EXECUTION)  # type: ignore[arg-type]
    assert "measurements.spot_diagram" in str(error.value)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"num_rings": 0}, "at least 1"),
        ({"execution": {"device": "cpu"}}, "needs ['precision']"),
        (
            {"execution": {"device": "cpu", "precision": "fp64", "backend": "numpy"}},
            "does not take",
        ),
    ],
)
def test_an_unusable_request_is_refused_before_the_solver_runs(
    kwargs: dict[str, Any], expected: str
) -> None:
    """The same argument checks a trace makes, reached through the same helpers."""
    call: dict[str, Any] = {"num_rings": 3, "execution": EXECUTION}
    call.update(kwargs)
    with pytest.raises(ValueError, match=expected.replace("[", r"\[").replace("]", r"\]")):
        spot_diagram(
            singlet_ref(), singlet_source(field_angle_deg=(0.0, 0.0)), **call
        )


def test_the_pass_through_arguments_reach_the_solver_and_are_validated_by_it() -> None:
    """`distribution` changes what is traced; an unknown value is the solver's refusal.

    With `distribution="random"` the solver reads the ring count as a ray count, so
    4 comes back as 4 rays rather than 61 -- which is the cheapest proof that the
    argument is not being dropped. And the vocabulary is not re-listed here: an
    unknown distribution or coordinate system raises from the pinned solver, which
    is the delegation working rather than a whitelist going stale.
    """
    setup = singlet_ref()
    source = singlet_source(field_angle_deg=(0.0, 0.0))

    random = spot_diagram(
        setup, source, num_rings=4, execution=EXECUTION, distribution="random"
    )
    assert random.x_m.shape == (4,)
    assert random.distribution == "random"

    with pytest.raises(ValueError, match="distribution"):
        spot_diagram(setup, source, num_rings=4, execution=EXECUTION, distribution="nonsense")
    with pytest.raises(ValueError, match=r"[Cc]oordinates"):
        spot_diagram(setup, source, num_rings=4, execution=EXECUTION, coordinates="nonsense")


def test_nothing_on_this_path_renders_anything() -> None:
    """`view()` is never called, and this module names no plotting library.

    Optiland itself depends on matplotlib, so a `sys.modules` check would report on
    the dependency rather than on this path. The AST walk is the claim that can be
    made: this module calls no `view` and imports no plotting.
    """
    source = Path("src/backends/optiland/analysis.py")
    tree = ast.parse(source.read_text())

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "view" not in called

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"matplotlib", "pyplot", "seaborn", "plotly"}
    # The measurement package is not reachable from here either: `backends` has no
    # allowlist edge to `measurements`, which is why the two result records exist.
    assert "measurements" not in imported


def test_the_two_class_names_this_module_did_not_need_are_absent() -> None:
    """The +1 class this package budgeted, and the two it did not.

    `scripts/class_budget.py` names both in the raise note: `AnalysisRequest` (an
    argument bag over a signature that already types its four arguments) and
    `AnalysisType` as an enum with a dispatch table -- one callable per analysis is
    what the catalog's per-record signature gate can describe, so the enum would
    have cost a class and bought nothing.
    """
    import backends.optiland.analysis as module

    defined = {
        node.name
        for node in ast.walk(ast.parse(Path("src/backends/optiland/analysis.py").read_text()))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == {"NativeSpotAnalysis"}
    assert not hasattr(module, "AnalysisRequest")
    assert not hasattr(module, "AnalysisType")


def test_characterization_only_the_two_paths_agree_on_an_unapodized_fan() -> None:
    """**Characterization, not a correctness gate.** Read the module docstring.

    Every ray of an unapodized Optiland launch fan has intensity 1, so this
    project's intensity-weighted centroid and RMS reduce exactly to the pinned
    solver's unweighted ones, and both read the same traced intersections. The
    agreement is therefore expected and is evidence about a coincidence of
    *definitions* -- it is not what makes either implementation correct, and
    `AGENTS.md` forbids treating it as such: `tests/physics/test_spot_diagram.py`
    holds the project-owned metrics to closed form instead.

    What this *would* catch is a real defect: a unit error on either side, a
    reference-surface mix-up, or the two paths silently measuring different
    populations.
    """
    setup = singlet_ref()
    source = singlet_source(field_angle_deg=(0.0, OFF_AXIS_FIELD_DEG))

    native = spot_diagram(
        setup,
        source,
        num_rings=OFF_AXIS_RING_COUNT,
        execution=EXECUTION,
        # Like for like: this project's radii are about the centroid, and the
        # solver's default reference is the CHIEF RAY. Comparing the two without
        # this argument compares two different definitions -- see the second half
        # of this test, which pins that they disagree.
        reference="centroid",
    )
    rays = trace(
        setup,
        source,
        sampling={"num_rings": OFF_AXIS_RING_COUNT, "reference_surface": "image_surface"},
        execution=EXECUTION,
    )
    assert rays.ray_splitting == "unsplit"
    project = measure_spot(rays)

    assert project.included_count == native.x_m.size
    assert np.allclose(np.abs(rays.amplitude) ** 2, 1.0)
    assert project.rms_radius_m == pytest.approx(native.rms_radius_m, rel=1e-12)
    assert project.geometric_radius_m == pytest.approx(native.geometric_radius_m, rel=1e-12)
    assert project.centroid_m[1] == pytest.approx(native.centroid_m[1], rel=1e-12)

    # And the trap, pinned: with the solver's own default the radii are about the
    # chief ray, so they do NOT agree off axis -- by 0.3% on the RMS and 5% on the
    # geometric radius. Small, plausible, and the reason
    # `NATIVE_SPOT_METRIC_DEFINITIONS` exists and is carried on the record.
    chief_ray = spot_diagram(
        setup, source, num_rings=OFF_AXIS_RING_COUNT, execution=EXECUTION
    )
    assert chief_ray.reference == "chief_ray"
    assert chief_ray.rms_radius_m > native.rms_radius_m
    assert chief_ray.geometric_radius_m > native.geometric_radius_m
    assert chief_ray.rms_radius_m != pytest.approx(project.rms_radius_m, rel=1e-6)
    # The reported centroid is the same either way: `centroid()` does not consult
    # `reference`, which is exactly what makes the radii-about-centroid reading wrong.
    assert chief_ray.centroid_m == pytest.approx(native.centroid_m, abs=1e-18)
