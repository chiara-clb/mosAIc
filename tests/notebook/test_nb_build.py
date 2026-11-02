import json

import pytest

from tests.notebook.core import ROOT, SRC, load_core
from build import build, parse_cells, to_notebook


def test_parse_cells_kinds_and_text():
    cells = parse_cells("#%% md\n# Title\n#%% core\nx = 1\n\n#%% code\n!pip install x\n")
    assert cells == [("md", "# Title"), ("core", "x = 1"), ("code", "!pip install x")]


def test_unknown_marker_and_stray_text_rejected():
    with pytest.raises(ValueError):
        parse_cells("#%% python\nx = 1")
    with pytest.raises(ValueError):
        parse_cells("x = 1\n#%% core\ny = 2")


def test_core_cells_are_tagged():
    nb = to_notebook([("core", "x = 1"), ("code", "y = 2"), ("md", "hi")])
    assert nb["cells"][0]["metadata"] == {"tags": ["core"]}
    assert nb["cells"][1]["metadata"] == {}
    assert nb["cells"][2]["cell_type"] == "markdown"


def test_built_notebook_matches_source(tmp_path):
    out = tmp_path / "nb.ipynb"
    nb = build(SRC, out)
    loaded = json.loads(out.read_text())
    assert loaded["nbformat"] == 4
    assert len(loaded["cells"]) == len(parse_cells(SRC.read_text())) == len(nb["cells"])


def test_committed_notebook_is_up_to_date():
    committed = ROOT / "notebooks" / "tot_validation" / "checkpoint4_tot_validation.ipynb"
    fresh = to_notebook(parse_cells(SRC.read_text()))
    assert json.loads(committed.read_text()) == json.loads(json.dumps(fresh)), \
        "run: python notebooks/build.py notebooks/tot_validation/notebook.src notebooks/tot_validation/checkpoint4_tot_validation.ipynb"


def test_settings_match_the_design():
    ns = load_core()
    assert (ns["BEAM_WIDTH"], ns["PRUNE_MARGIN"], ns["MAX_DEPTH"], ns["MAX_MODEL_CALLS"]) == (3, 0.3, 5, 25)
    assert ns["STRENGTH_POINTS"] == {"small": 3, "medium": 6, "large": 9}
    assert ns["MAX_TOTAL_SHIFT"] == 15
    assert (ns["K_CANDIDATES"], ns["K_FINAL"], ns["MMR_LAMBDA"]) == (20, 5, 0.7)
    assert ns["MIN_SCORE"] == 0.20 and ns["FILING_MIN_SCORE"] < ns["MIN_SCORE"]   # lowered after the scenario run
    assert len(ns["CASE_TICKERS"]) == 30 and set(ns["TICKERS"]) <= set(ns["CASE_TICKERS"])
    assert abs(sum(ns["WEIGHTS"].values()) - 1) < 1e-9


def test_overrides_apply_after_settings():
    assert load_core(BEAM_WIDTH=1)["BEAM_WIDTH"] == 1
