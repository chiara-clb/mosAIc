"""Run a notebook's `core` cells offline, so their logic can be tested without Colab."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))

from build import parse_cells  # noqa: E402

SRC = ROOT / "notebooks" / "tot_validation" / "notebook.src"
BASELINE_SRC = ROOT / "notebooks" / "baseline_comparison" / "notebook.src"


def load_core(src: Path = SRC, module_name: str = "tot_core", **overrides) -> dict:
    """Execute every core cell of `src` in order and return the namespace.

    `overrides` are injected after the settings cell (the first core cell) runs.
    """
    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod          # dataclasses look up their module here
    ns = mod.__dict__
    first = True
    for kind, cell in parse_cells(Path(src).read_text()):
        if kind != "core":
            continue
        exec(compile(cell, f"<{module_name} core cell>", "exec"), ns)
        if first:
            ns.update(overrides)
            first = False
    return ns


def load_baseline_core(**overrides) -> dict:
    return load_core(BASELINE_SRC, "baseline_core", **overrides)
