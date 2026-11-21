from datetime import UTC, datetime

from mosaic.prefetch import prefetch
from mosaic.world import make_demo_world


def test_prefetch_writes_a_coverage_row_per_stock_and_date(tmp_path):
    world = make_demo_world()
    dates = [world.as_of]
    out = tmp_path / "coverage.csv"
    seen = []
    table = prefetch(["ACME"], dates, world_factory=lambda t, d: (world, 0),
                     answer_key=lambda t, d: {"fall": 30, "flat": 40, "gain": 30}, out=out, log=seen.append)
    row = table.iloc[0]
    assert row["news_stories"] == len(world.news)
    assert row["filings_30d"] >= 1 and row["filing_chunks_30d"] >= row["filings_30d"]
    assert row["answer_key"] == "ok" and row["gdelt_errors"] == 0
    assert out.exists() and seen[0].startswith("ACME 2025-06-02")


def test_prefetch_keeps_going_after_a_failure(tmp_path):
    def broken(t, d):
        raise ConnectionError("SEC down")

    table = prefetch(["AAPL", "MSFT"], [datetime(2026, 7, 15, tzinfo=UTC)], world_factory=broken, log=lambda m: None)
    assert len(table) == 2 and table["error"].str.contains("SEC down").all()
