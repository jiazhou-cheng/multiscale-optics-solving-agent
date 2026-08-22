"""Render the component capability matrix that the precision policy doc embeds.

Generated output, not a hand-maintained table:
``tests/test_registry_matches_capabilities.py`` re-renders these rows and fails if
``docs/precision/precision_device_policy.md`` no longer matches, so the
documented matrix cannot go stale while the capabilities move.

    ./run.sh python benchmarks/probes/precision/capability_table.py
"""

from typing import Any

from core.capabilities import capability_matrix

COLUMNS = (
    ("Component", "component"),
    ("Devices", "devices"),
    ("Precisions", "precisions"),
    ("Input accepted", "accepted_input_dtypes"),
    ("Native compute", "native_compute_dtypes"),
    ("Output", "output_dtypes"),
    ("Ingestible but lossy", "lossy_input_dtypes"),
    ("Namespaces", "namespaces"),
    ("Compute floor", "minimum_compute_precision"),
)


def render(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(value) or "--"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {', '.join(item)}" for key, item in value.items())
    return str(value)


def main() -> None:
    rows = capability_matrix()
    headers = [header for header, _ in COLUMNS]
    widths = [
        max(len(header), *(len(render(row[key])) for row in rows))
        for header, key in COLUMNS
    ]

    def line(cells: list[str]) -> str:
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=True)]
        return "| " + " | ".join(padded) + " |"

    print(line(headers))
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for row in rows:
        print(line([render(row[key]) for _, key in COLUMNS]))

    print()
    for row in rows:
        print(f"### {row['component']}\n")
        print(f"- device namespaces: {render(row['device_namespaces'])}")
        print(f"- evidence: {row['evidence']}")
        print(f"- notes: {row['notes']}\n")


if __name__ == "__main__":
    main()
