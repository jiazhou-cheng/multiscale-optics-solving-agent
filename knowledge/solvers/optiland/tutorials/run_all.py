"""Run every repo-owned Optiland tutorial reproduction, one at a time (CHE-57 / PB6).

    ./run.sh python knowledge/solvers/optiland/tutorials/run_all.py
    ./run.sh python knowledge/solvers/optiland/tutorials/run_all.py --write-expected
    ./run.sh python knowledge/solvers/optiland/tutorials/run_all.py --only t04 t10

Reproductions execute sequentially in slug order, never concurrently: the
Optiland backend is process-global mutable state (`conventions.md`) and the
repository's GPU/resource policy forbids parallel solver execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _optiland_harness import emit, iter_tutorial_modules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="slug prefixes to run")
    args = parser.parse_args()

    failures: list[str] = []
    for module in iter_tutorial_modules():
        meta = module.TUTORIAL
        if args.only and not any(meta.slug.startswith(prefix) for prefix in args.only):
            continue
        print(f"\n===== {meta.slug}: {meta.title} ({meta.level}) =====", flush=True)
        code = emit(meta, module.run(), write_expected=args.write_expected)
        if code:
            failures.append(meta.slug)
    print(f"\n{len(failures)} failing reproduction(s): {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
