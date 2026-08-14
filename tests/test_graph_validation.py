from pathlib import Path

from multiscale_optics_agent.core.graph import GraphValidator, Severity
from multiscale_optics_agent.core.specs import GraphSpec
from multiscale_optics_agent.registry.loader import Registry


ROOT = Path(__file__).resolve().parents[1]


def _gradient_requiring_graph() -> GraphSpec:
    """A ray->wave graph that asks for a gradient it cannot have.

    CHE-31 reconciled `examples/graphs/ray_to_wave.yaml` with what M3 actually
    executes, which meant removing its gradient-requiring design variable and
    objective: a declared design variable with `requires_gradient: true` reads as
    a capability claim, and no gradient through either coupler is verified. The
    two gradient-policy tests below used to borrow that file as a fixture, so
    they need their own. Declaring it here also states *why* the gradient is
    wanted, which the borrowed file never did.
    """
    return GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "lens", "model": "M_RAY_OPTILAND"},
                {"id": "wave", "model": "M_WAVE_CHROMATIX"},
            ],
            "edges": [
                {
                    "id": "pupil_reconstruction",
                    "coupler": "C_RAY_TO_WAVE",
                    # CHE-34 moved C_RAY_TO_WAVE's source port to `rays`: the
                    # wavelet sum needs a direction per sample and
                    # wavefront_samples has none.
                    "source": {"node": "lens", "port": "rays"},
                    "target": {"node": "wave", "port": "input_field"},
                }
            ],
            "design_variables": [
                {
                    "name": "asphere_coefficient",
                    "node": "lens",
                    "parameter": "surfaces[2].asphere.a4",
                    "bounds": [-0.01, 0.01],
                    "requires_gradient": True,
                }
            ],
            "objectives": [
                {
                    "name": "focal_peak",
                    "node": "wave",
                    "port": "output_field",
                    "metric": "normalized_peak_intensity",
                    "requires_gradient": True,
                }
            ],
            "require_verified_gradients": False,
        }
    )


def test_ray_to_wave_example_is_structurally_valid() -> None:
    """The shipped M3 example graph must validate and claim no gradient."""
    registry = Registry.from_package()
    graph = Registry.load_graph(ROOT / "examples" / "graphs" / "ray_to_wave.yaml")
    report = GraphValidator(registry).validate(graph)

    assert report.valid, [issue.model_dump() for issue in report.errors]
    # Forward-only in M3: nothing in the file may request a gradient, so the
    # validator has no gradient path to warn about.
    assert not graph.design_variables
    assert all(not objective.requires_gradient for objective in graph.objectives)
    assert not any(issue.code == "GRADIENT_PATH_NOT_FULLY_VERIFIED" for issue in report.issues)


def test_unverified_gradient_path_warns() -> None:
    """Asking for a gradient across C_RAY_TO_WAVE must warn: nothing verifies it."""
    registry = Registry.from_package()
    report = GraphValidator(registry).validate(_gradient_requiring_graph())

    assert report.valid, [issue.model_dump() for issue in report.errors]
    assert any(issue.severity is Severity.WARNING for issue in report.issues)
    assert any(issue.code == "GRADIENT_PATH_NOT_FULLY_VERIFIED" for issue in report.issues)


def test_unknown_model_fails() -> None:
    registry = Registry.from_package()
    graph = GraphSpec.model_validate(
        {
            "nodes": [{"id": "bad", "model": "M_DOES_NOT_EXIST"}],
        }
    )
    report = GraphValidator(registry).validate(graph)

    assert not report.valid
    assert any(issue.code == "UNKNOWN_MODEL" for issue in report.errors)


def test_artifact_type_mismatch_fails() -> None:
    """Both ends wrong at once, so neither check can hide behind the other.

    CHE-36 (M3.7) repointed this fixture. It used to name C_FIELD_TO_PSF, which is
    no longer a registered coupler -- PSF extraction is a measurement on the
    terminal field, not an edge. C_ABSORPTION_TO_HEAT replaces it because it has
    the same property this test needs and nothing else: its declared source
    (`absorbed_power_density`) and target (`heat_source`) both differ from the
    ports wired below, so one report must carry both mismatch codes.
    """
    registry = Registry.from_package()
    graph = GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "em", "model": "M_EM_FDTDX"},
                {"id": "wave", "model": "M_WAVE_CHROMATIX"},
            ],
            "edges": [
                {
                    "id": "wrong",
                    "coupler": "C_ABSORPTION_TO_HEAT",
                    # near_field is `near_field_surface`, not absorbed_power_density;
                    # input_field is `complex_field`, not heat_source. M_EM_FDTDX does
                    # expose an `absorbed_power` port, so wiring `near_field` here is
                    # the deliberate half of the error, not an accident of the model.
                    "source": {"node": "em", "port": "near_field"},
                    "target": {"node": "wave", "port": "input_field"},
                }
            ],
        }
    )
    report = GraphValidator(registry).validate(graph)

    assert not report.valid
    codes = {issue.code for issue in report.errors}
    assert "COUPLER_SOURCE_TYPE_MISMATCH" in codes
    assert "COUPLER_TARGET_TYPE_MISMATCH" in codes


def test_field_to_psf_is_not_a_registered_coupler() -> None:
    """CHE-36 (M3.7): the retirement is pinned, so it cannot drift back in.

    `ComplexField -> |U|^2` is a measurement of the terminal simulated field, not
    a cross-representation handoff, so it is not a coupler. The registry is the
    architectural statement of what a coupler is, and a graph that names the
    retired id must fail rather than validate.
    """
    registry = Registry.from_package()
    assert "C_FIELD_TO_PSF" not in registry.couplers

    graph = GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "wave", "model": "M_WAVE_CHROMATIX"},
                {"id": "sensor", "model": "M_SENSOR_IDEAL"},
            ],
            "edges": [
                {
                    "id": "measurement_as_an_edge",
                    "coupler": "C_FIELD_TO_PSF",
                    "source": {"node": "wave", "port": "output_field"},
                    "target": {"node": "sensor", "port": "psf"},
                }
            ],
        }
    )
    report = GraphValidator(registry).validate(graph)

    assert not report.valid
    assert any(issue.code == "UNKNOWN_COUPLER" for issue in report.errors), [
        issue.model_dump() for issue in report.errors
    ]


def test_cycle_is_rejected() -> None:
    registry = Registry.from_package()
    graph = GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "a", "model": "M_WAVE_CHROMATIX"},
                {"id": "b", "model": "M_WAVE_CHROMATIX"},
            ],
            "edges": [
                {
                    "id": "a_to_b",
                    "coupler": "C_RAY_TO_WAVE",
                    "source": {"node": "a", "port": "output_field"},
                    "target": {"node": "b", "port": "input_field"},
                },
                {
                    "id": "b_to_a",
                    "coupler": "C_RAY_TO_WAVE",
                    "source": {"node": "b", "port": "output_field"},
                    "target": {"node": "a", "port": "input_field"},
                },
            ],
            "allow_cycles": False,
        }
    )
    report = GraphValidator(registry).validate(graph)

    assert not report.valid
    assert any(issue.code == "CYCLE_NOT_ALLOWED" for issue in report.errors)


def test_strict_verified_gradient_policy_rejects_unverified_path() -> None:
    registry = Registry.from_package()
    strict_graph = _gradient_requiring_graph().model_copy(
        update={"require_verified_gradients": True}
    )
    report = GraphValidator(registry).validate(strict_graph)

    assert not report.valid
    assert any(issue.code == "NO_DIFFERENTIABLE_PATH" for issue in report.errors)
