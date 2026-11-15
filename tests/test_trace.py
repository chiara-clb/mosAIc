from datetime import datetime, timedelta, timezone

from mosaic.trace import Trace, leakage_audit

AS_OF = datetime(2026, 7, 15, tzinfo=timezone.utc)


def make_trace():
    t = Trace(run_id="AAPL-2026-07-15", as_of=AS_OF)
    t.add("coordinator", "start", ticker="AAPL")
    t.add("research", "tool_call", tool="search_documents", status="ok")
    t.used("research", "8-K-0001", AS_OF - timedelta(days=3))
    return t


def test_clean_run_passes_the_audit():
    assert leakage_audit(make_trace()) == []


def test_planted_late_source_is_caught():
    t = make_trace()
    t.used("research", "8-K-LATE", AS_OF + timedelta(hours=1))
    t.used("research", "8-K-SAME-DAY", AS_OF)              # the start of the day itself is already too late
    found = leakage_audit(t)
    assert [f["source"] for f in found] == ["8-K-LATE", "8-K-SAME-DAY"]
    assert found[0]["node"] == "research" and found[0]["step"] == 3


def test_naive_timestamps_are_read_as_utc():
    t = make_trace()
    t.used("research", "news-1", datetime(2026, 7, 15, 9, 0))
    assert [f["source"] for f in leakage_audit(t)] == ["news-1"]


def test_events_are_numbered_and_filterable():
    t = make_trace()
    assert [e["step"] for e in t.events] == [0, 1, 2]
    assert [e["tool"] for e in t.of_kind("tool_call")] == ["search_documents"]


def test_jsonl_round_trip(tmp_path):
    t = make_trace()
    t.used("research", "8-K-LATE", AS_OF + timedelta(days=1))
    path = t.write_jsonl(tmp_path / "runs" / "trace.jsonl")
    back = Trace.read_jsonl(path)
    assert back.run_id == t.run_id and back.as_of == AS_OF
    assert len(back.events) == 4
    assert [f["source"] for f in leakage_audit(back)] == ["8-K-LATE"]
