"""Command-line interface for registry inspection and graph validation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.graph import GraphValidator, Severity
from registry.loader import Registry

app = typer.Typer(no_args_is_help=True, help="Validate and inspect typed optical physics graphs.")
console = Console()


@app.command("list-models")
def list_models() -> None:
    """List models in the packaged registry."""

    registry = Registry.from_package()
    table = Table("ID", "Approximation", "Framework", "Derivative", "Maturity")
    for model in sorted(registry.models.values(), key=lambda item: item.id):
        table.add_row(
            model.id,
            model.approximation.value,
            model.framework.value,
            model.derivative.mode.value,
            model.maturity.value,
        )
    console.print(table)


@app.command("list-couplers")
def list_couplers() -> None:
    """List couplers in the packaged registry."""

    registry = Registry.from_package()
    table = Table("ID", "Source", "Target", "Derivative", "Lossy")
    for coupler in sorted(registry.couplers.values(), key=lambda item: item.id):
        table.add_row(
            coupler.id,
            coupler.source.artifact.value,
            coupler.target.artifact.value,
            coupler.derivative.mode.value,
            str(coupler.lossy),
        )
    console.print(table)


@app.command()
def validate(graph: Path) -> None:
    """Validate a graph YAML against the packaged model/coupler registry."""

    registry = Registry.from_package()
    graph_spec = Registry.load_graph(graph)
    report = GraphValidator(registry).validate(graph_spec)

    table = Table("Severity", "Code", "Location", "Message")
    for issue in report.issues:
        table.add_row(
            issue.severity.value,
            issue.code,
            issue.location or "—",
            issue.message,
        )
    console.print(table)
    if not report.valid:
        raise typer.Exit(code=1)
    if any(issue.severity is Severity.WARNING for issue in report.issues):
        console.print("[yellow]Graph is structurally valid with scientific qualifications.[/yellow]")
    else:
        console.print("[green]Graph is valid.[/green]")


if __name__ == "__main__":
    app()
