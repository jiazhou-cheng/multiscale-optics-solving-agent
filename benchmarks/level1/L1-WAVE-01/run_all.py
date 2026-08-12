"""Run the complete independent M1 Chromatix branch bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiscale_optics_agent.evaluation.m1_bundle import WAVE_SPEC, build_branch_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/M1/wave"))
    args = parser.parse_args()
    return build_branch_bundle(WAVE_SPEC, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
