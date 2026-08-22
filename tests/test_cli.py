"""The packaged console script, which had no test at all.

`multiscale-optics` is a `[project.scripts]` entry point, so it is part of what
this repository ships. It was the only shipped module at 0% coverage that also
had no test file anywhere -- `make validate` invoked it, and nothing checked
what it printed or what it exited with.

The three tests below are deliberately about the *contract* of each command
rather than its formatting: which rows appear, what the exit code is, and what
happens on the failure path. Asserting on Rich's table layout would fail on a
terminal-width change and teach the next person to loosen the assertion.

One consequence worth stating: `list-models` and `list-couplers` read the
**packaged** registry, so these tests are also a check that the shipped YAML
loads through `importlib.resources`. That is the path `Registry.from_package`
takes in a wheel, and it is not the same path as reading the file from the
source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "examples" / "graphs"


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    # A wide terminal so Rich does not truncate an id mid-word and turn a
    # content assertion into a layout assertion.
    return CliRunner(env={"COLUMNS": "200"})


def test_list_models_names_every_registered_model(runner: CliRunner) -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0, result.output
    assert "M_RAY_OPTILAND" in result.output
    assert "M_WAVE_CHROMATIX" in result.output
    # The maturity column is the one CHE-87 made carry information; a table that
    # silently dropped it would still look fine.
    assert "characterized" in result.output


def test_list_couplers_reports_the_lossy_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["list-couplers"])
    assert result.exit_code == 0, result.output
    assert "C_RAY_TO_WAVE" in result.output
    assert "C_WAVE_TO_RAY" in result.output
    # Both couplers are declared lossy, and that declaration is the reason a
    # graph planner may not treat either as a free change of representation.
    assert "True" in result.output


def test_validate_accepts_the_shipped_example_graph(runner: CliRunner) -> None:
    result = runner.invoke(app, ["validate", str(GRAPHS / "ray_to_wave.yaml")])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_validate_exits_nonzero_on_an_unknown_coupler(tmp_path: Path, runner: CliRunner) -> None:
    """The failure path, which is the half `make validate` never exercises.

    A validator that prints its errors and exits 0 is worse than no validator:
    every caller that checks the exit code reads it as a pass.
    """
    graph = tmp_path / "broken.yaml"
    graph.write_text(
        "nodes:\n"
        "  - {id: lens, model: M_RAY_OPTILAND}\n"
        "  - {id: wave, model: M_WAVE_CHROMATIX}\n"
        "edges:\n"
        "  - id: bad\n"
        "    coupler: C_NOT_A_COUPLER\n"
        "    source: {node: lens, port: rays}\n"
        "    target: {node: wave, port: input_field}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(graph)])
    assert result.exit_code == 1, result.output
    assert "UNKNOWN_COUPLER" in result.output


def test_validate_reports_a_malformed_graph_rather_than_tracing_back(
    tmp_path: Path, runner: CliRunner
) -> None:
    """A YAML that is not a graph must produce a RegistryError, not a KeyError.

    `Registry.load_graph` wraps pydantic's failure so the message names the file.
    Losing that wrapping is a silent regression: the command still fails, but the
    operator gets a schema traceback instead of "invalid graph YAML at <path>".
    """
    graph = tmp_path / "not_a_graph.yaml"
    graph.write_text("just: a mapping\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(graph)])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "not_a_graph.yaml" in str(result.exception)
