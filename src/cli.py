"""Command-line interface for registry inspection, graph validation and execution."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.execution import RunStatus
from core.graph import GraphValidator, Severity
from discovery import (
    check_connection,
    describe_component,
    route_capability,
)
from registry.loader import Registry
from runtime.executor import GraphExecutor, InMemoryCache

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


#: Module-level singletons: ruff's B008 is right that a call in a default is a
#: trap, and typer needs the option objects to exist somewhere.
_OUTPUT_OPTION = typer.Option(None, help="Where to write the execution record JSON.")
_SEED_OPTION = typer.Option(None, help="Seed threaded through every node.")
_CACHE_OPTION = typer.Option(False, help="Reuse node results within this process.")
_POLICY_OPTION = typer.Option("cold", help="Context policy: cold, warm or guided.")


@app.command()
def run(
    graph: Path,
    output: Path | None = _OUTPUT_OPTION,
    seed: int | None = _SEED_OPTION,
    cache: bool = _CACHE_OPTION,
) -> None:
    """Execute a validated graph and write its execution record.

    The record says what happened. It does not say whether the physics was
    right: that is `verification.verifier.verify`, which reads this record
    against a benchmark family. Keeping the two apart is what lets one run be
    judged against different criteria later without re-running it.
    """

    registry = Registry.from_package()
    graph_spec = Registry.load_graph(graph)
    executor = GraphExecutor(registry, cache=InMemoryCache() if cache else None)
    record = executor.run(graph_spec, seed=seed)

    table = Table("Node", "Component", "Outcome", "Wall (s)", "Codes")
    for node in record.nodes:
        table.add_row(
            node.node_id,
            node.component,
            node.outcome.value,
            f"{node.cost.wall_seconds:.3f}" if node.cost else "—",
            ", ".join(node.contract_codes) or "—",
        )
    console.print(table)

    if record.refusal is not None:
        console.print(f"[red]{record.refusal.kind.value}[/red]: {record.refusal.detail}")
        if record.refusal.remedy:
            console.print(f"  remedy: {record.refusal.remedy}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        console.print(f"wrote {output}")

    console.print(f"status: {record.status.value}")
    if record.status is not RunStatus.SUCCEEDED:
        raise typer.Exit(code=1)


@app.command()
def describe(component: str, policy: str = _POLICY_OPTION) -> None:
    """Everything one query can say about one component.

    Can, should, and what happens if you get it wrong -- from one call. Every
    field is derived from the capability table, the registry, the validator, the
    ledger, the family registry or the refusal catalogue; none is written here.
    """

    description = describe_component(component, policy=policy)
    console.print(f"[bold]{description.component}[/bold] {description.version} "
                  f"({description.kind}, {description.maturity})")
    console.print(description.description)

    if description.derivative_warning:
        console.print(f"[red]{description.derivative_warning}[/red]")

    ports = Table("Port", "Direction", "Artifact", "Requires", "Provides")
    for port in [*description.inputs, *description.outputs]:
        ports.add_row(
            port.name,
            port.direction,
            port.artifact,
            ", ".join(port.requires_metadata) or "—",
            ", ".join(port.provides_metadata) or "—",
        )
    console.print(ports)

    console.print(
        f"devices {description.devices}  native dtypes "
        f"{description.native_compute_dtypes}  lossy-in {description.lossy_input_dtypes}"
    )

    families = Table("Family", "Category", "Decides?", "Gate")
    for family in description.families:
        families.add_row(
            family.family_id,
            family.category,
            "yes" if family.gate_deciding else "[yellow]no[/yellow]",
            family.gate_status,
        )
    console.print(families)

    refusals = Table("Refusal", "Outcome", "Remedy")
    for refusal in description.refusals:
        refusals.add_row(refusal.code, refusal.status, refusal.remedy)
    console.print(refusals)


@app.command("check-connection")
def check_connection_command(source: str, target: str) -> None:
    """`check-connection M_RAY_OPTILAND.rays M_WAVE_CHROMATIX.input_field`.

    Answers with the mediating coupler and the declarations the edge must carry
    -- which are otherwise discoverable only by being refused.
    """

    try:
        source_component, source_port = source.split(".", 1)
        target_component, target_port = target.split(".", 1)
    except ValueError:
        raise typer.BadParameter("use <COMPONENT>.<port> for both arguments") from None

    report = check_connection(source_component, source_port, target_component, target_port)
    verdict = "[green]compatible[/green]" if report.compatible else "[red]incompatible[/red]"
    console.print(f"{source} -> {target}: {verdict}")
    if report.coupler:
        console.print(f"  coupler: {report.coupler}")
    if report.required_edge_declarations:
        console.print(f"  the edge must declare: {report.required_edge_declarations}")
    for issue in report.issues:
        console.print(f"  [{issue['code']}] {issue['message']}")
    if not report.compatible:
        raise typer.Exit(code=1)


@app.command("route-capability")
def route_capability_command(components: list[str]) -> None:
    """At what device and precision can this ordered route execute?"""

    answer = route_capability(components)
    console.print(f"route: {' -> '.join(answer.route)}")
    console.print(
        f"  crossable: {answer.feasible}   devices: {answer.devices or '—'}"
    )
    console.print(
        f"  one precision for the whole route: {answer.uniform_precision_available} "
        f"{answer.uniform_compute_dtypes or ''}"
    )
    for handover in answer.handovers:
        kind = "lossy" if handover.lossy else "exact"
        dtypes = handover.lossy_dtypes if handover.lossy else handover.exact_dtypes
        console.print(f"  {handover.producer} -> {handover.consumer}: {kind} {dtypes}")
    if answer.blocking_pair:
        console.print(f"  [yellow]blocking pair[/yellow]: {answer.blocking_pair}")
    if answer.reason:
        console.print(f"  {answer.reason}")
    if not answer.feasible:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
