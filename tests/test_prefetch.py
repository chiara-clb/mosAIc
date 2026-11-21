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


def test_prefetch_can_run_just_a_few_chosen_dates():
    from mosaic.prefetch import parse_pairs
    pairs = parse_pairs("aapl:2026-08-05, INTC:2026-07-29")
    assert pairs == [("AAPL", datetime(2026, 8, 5, tzinfo=UTC)), ("INTC", datetime(2026, 7, 29, tzinfo=UTC))]
    world = make_demo_world()
    seen = []
    table = prefetch(["MSFT"], [datetime(2026, 7, 1, tzinfo=UTC)], world_factory=lambda t, d: (seen.append((t, d)), (world, 0))[1],
                     pairs=pairs, log=lambda m: None)
    assert seen == pairs and list(table["ticker"]) == ["AAPL", "INTC"]
