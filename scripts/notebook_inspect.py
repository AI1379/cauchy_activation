#!/usr/bin/env python3
"""Inspect code/markdown cells in one or more Jupyter notebooks.

Usage:
  python scripts/notebook_inspect.py notebooks/cauchy_res_mixer/*.ipynb
  python scripts/notebook_inspect.py --defs-only notebooks/foo.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_notebook_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.exists():
            paths.append(p)
            continue
        paths.extend(sorted(Path().glob(raw)))
    return paths


def inspect_notebook(path: Path, defs_only: bool) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\n===== {path} =====")
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")

    for idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))

        if defs_only and not any(token in source for token in ("class ", "def ")):
            continue

        first_line = source.strip().splitlines()[0] if source.strip() else "<empty>"
        print(f"- Cell {idx:>3}: {ctype:>8} | {first_line[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect notebook cell summaries.")
    parser.add_argument("paths", nargs="+", help="Notebook paths or glob patterns.")
    parser.add_argument(
        "--defs-only",
        action="store_true",
        help="Only print code cells containing class/def definitions.",
    )
    args = parser.parse_args()

    paths = iter_notebook_paths(args.paths)
    if not paths:
        raise SystemExit("No notebook files found.")

    for path in paths:
        inspect_notebook(path, args.defs_only)


if __name__ == "__main__":
    main()
