import json

from scipy.stats import norm

from tests.notebook.core import BASELINE_SRC, ROOT, load_baseline_core
from build import parse_cells, to_notebook

ns = load_baseline_core()


def test_committed_notebook_is_up_to_date():
    committed = ROOT / "notebooks" / "baseline_comparison" / "baseline_comparison.ipynb"
    fresh = to_notebook(parse_cells(BASELINE_SRC.read_text()))
    assert json.loads(committed.read_text()) == json.loads(json.dumps(fresh)), \
        "run: python notebooks/build.py notebooks/baseline_comparison/notebook.src notebooks/baseline_comparison/baseline_comparison.ipynb"


def test_settings_match_the_design():
    assert (ns["HORIZON"], ns["SIGMA_LOOKBACK"], ns["BUCKET_K"]) == (21, 252, 0.5)
    assert len(ns["TICKERS"]) == 30
    assert ns["VERSIONS"] == ("A", "B", "C", "D", "Fixed guess")


def test_fixed_guess_is_the_average_month():
    tail = 100 * norm.cdf(-0.5)
    assert ns["FIXED_GUESS"]["fall"] == round(tail, 2) == ns["FIXED_GUESS"]["gain"]
    assert abs(sum(ns["FIXED_GUESS"].values()) - 100) < 0.01


def test_calm_bins_cover_every_day_without_overlap():
    days = [d for lo, hi in ns["CALM_BINS"] for d in range(lo, min(hi, 200) + 1)]
    assert days == list(range(1, 201))
