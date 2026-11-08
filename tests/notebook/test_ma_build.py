"""Checkpoint 5 notebook: optional cells and the built notebook."""

import json

from tests.notebook.core import MULTI_AGENT_SRC, ROOT, load_multi_agent_core
from build import parse_cells, to_notebook

NS = load_multi_agent_core()
BUILT = ROOT / "notebooks" / "multi_agent" / "checkpoint5_multi_agent.ipynb"


def test_crewai_arguments_come_from_the_role_cards():
    kw = NS["crewai_kwargs"]("research")
    assert kw["role"] == "Research analyst" and kw["allow_delegation"] is False
    assert set(kw) == {"role", "goal", "backstory", "allow_delegation", "verbose"}
    assert NS["USE_CREWAI"] is False


def test_scripted_by_default():
    assert NS["MODE"] == "scripted" and NS["CHAT"] is NS["scripted_chat"]


def test_every_section_is_present():
    heads = [c.splitlines()[0] for k, c in parse_cells(MULTI_AGENT_SRC.read_text()) if k == "md" and c.startswith("## ")]
    assert [h.split(".")[0] for h in heads] == [f"## {i}" for i in range(1, 12)]


def test_committed_notebook_is_up_to_date():
    fresh = to_notebook(parse_cells(MULTI_AGENT_SRC.read_text()))
    assert json.loads(BUILT.read_text()) == json.loads(json.dumps(fresh)), \
        "run: python notebooks/build.py notebooks/multi_agent/notebook.src notebooks/multi_agent/checkpoint5_multi_agent.ipynb"
