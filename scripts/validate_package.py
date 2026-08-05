"""Fast repository consistency checks that do not import optional physics solvers."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from multiscale_optics_agent.core.graph import GraphValidator
from multiscale_optics_agent.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")


def validate_yaml_files() -> None:
    for path in ROOT.rglob("*.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)


def validate_graph_examples(registry: Registry) -> None:
    validator = GraphValidator(registry)
    for path in (ROOT / "examples" / "graphs").glob("*.yaml"):
        report = validator.validate(Registry.load_graph(path))
        if not report.valid:
            details = "\n".join(f"{item.code}: {item.message}" for item in report.errors)
            raise RuntimeError(f"Invalid example graph {path}:\n{details}")


def validate_markdown_links_are_well_formed() -> None:
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for url in MARKDOWN_LINK.findall(text):
            if any(char.isspace() for char in url):
                raise RuntimeError(f"Whitespace in Markdown URL {url!r} at {path}")


def main() -> None:
    registry = Registry.from_package()
    validate_yaml_files()
    validate_graph_examples(registry)
    validate_markdown_links_are_well_formed()
    print(
        f"Validated {len(registry.models)} models, {len(registry.couplers)} couplers, "
        "all YAML files, and all example graphs."
    )


if __name__ == "__main__":
    main()
