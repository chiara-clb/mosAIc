"""Run the notebook's `core` cells offline, so their logic can be tested without Colab."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))

from build import parse_cells  # noqa: E402

SRC = ROOT / "notebooks" / "tot_validation" / "notebook.src"


def load_core(**overrides) -> dict:
    """Execute every core cell in order and return the namespace.

    `overrides` are injected after the settings cell runs, e.g. load_core(BEAM_WIDTH=1).
    """
    import types
    mod = types.ModuleType("tot_core")
    sys.modules["tot_core"] = mod          # dataclasses look up their module here
    ns = mod.__dict__
    first = True
    for kind, src in parse_cells(SRC.read_text()):
        if kind != "core":
            continue
        exec(compile(src, f"<core cell>", "exec"), ns)
        if first:
            ns.update(overrides)
            first = False
    return ns
