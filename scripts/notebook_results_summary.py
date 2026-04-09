#!/usr/bin/env python3
"""Summarize final metrics from notebook outputs.

This script scans `execute_result` and `stream` outputs for lines containing
keywords such as `test_acc`, `accuracy`, `mse`, and `rel_l2`.

Usage:
  python scripts/notebook_results_summary.py notebooks/cauchy_res_mixer/*.ipynb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

KEYWORDS = ("test_acc", "accuracy", "mse", "rel_l2", "max_abs_error")


def iter_notebook_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.exists():
            paths.append(p)
            continue
        paths.extend(sorted(Path().glob(raw)))
    return paths


def normalize_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def has_keyword(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in KEYWORDS)


def summarize_notebook(path: Path, tail_cells: int) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    cells = cells[-tail_cells:] if tail_cells > 0 else cells

    print(f"\n===== {path} =====")

    hits = 0
    for idx, cell in enumerate(cells, start=max(1, len(nb.get("cells", [])) - len(cells) + 1)):
        if cell.get("cell_type") != "code":
            continue

        for out in cell.get("outputs", []):
            chunks: list[str] = []

            if out.get("output_type") == "stream":
                chunks.append(normalize_text(out.get("text", "")))
            elif out.get("output_type") in {"execute_result", "display_data"}:
                data = out.get("data", {})
                if "text/plain" in data:
                    chunks.append(normalize_text(data["text/plain"]))

            for chunk in chunks:
                lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
                lines = [ln for ln in lines if has_keyword(ln)]
                if not lines:
                    continue

                hits += 1
                print(f"- Cell {idx}:")
                for ln in lines[:8]:
                    print(f"  {ln[:180]}")

    if hits == 0:
        print("(No metric-like output found in scanned cells.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metric-like lines from notebook outputs.")
    parser.add_argument("paths", nargs="+", help="Notebook paths or glob patterns.")
    parser.add_argument(
        "--tail-cells",
        type=int,
        default=12,
        help="Only inspect the last N cells of each notebook (default: 12). Use 0 for all cells.",
    )
    args = parser.parse_args()

    paths = iter_notebook_paths(args.paths)
    if not paths:
        raise SystemExit("No notebook files found.")

    for path in paths:
        summarize_notebook(path, args.tail_cells)


if __name__ == "__main__":
    main()
