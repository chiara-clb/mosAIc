"""Build a Jupyter notebook from a plain-text cell source.

Source format: cells are separated by marker lines.
    #%% md      a markdown cell
    #%% code    a code cell that needs the network, a GPU or heavy libraries
    #%% core    a code cell with pure logic; tests run these cells offline

Usage:  python notebooks/build.py notebooks/tot_validation/notebook.src notebooks/tot_validation/checkpoint4_tot_validation.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KINDS = {"md", "code", "core"}


def parse_cells(text: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("#%% "):
            kind = line[4:].strip()
            if kind not in KINDS:
                raise ValueError(f"unknown cell kind {kind!r}")
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            raise ValueError("text before the first cell marker")
    return [(k, "\n".join(lines).strip("\n")) for k, lines in cells]


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            meta = {"tags": ["core"]} if kind == "core" else {}
            out.append({"cell_type": "code", "metadata": meta, "source": lines,
                        "execution_count": None, "outputs": []})
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(src: str | Path, dst: str | Path) -> dict:
    nb = to_notebook(parse_cells(Path(src).read_text()))
    Path(dst).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return nb


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
