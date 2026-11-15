import json

from mosaic.cli import main
from mosaic.trace import Trace, leakage_audit


def test_demo_writes_report_state_and_trace(tmp_path, capsys):
    assert main(["--demo", "--out", str(tmp_path)]) == 0
    folder = tmp_path / "1_normal" / "ACME-2025-06-02"
    report = (folder / "report.md").read_text()
    assert "# Acme Robotics Inc. (ACME), forecast as of 2025-06-02" in report
    assert "| F1 | 2025-05-28 | spread | small |" in report and "Leakage audit: clean" in report
    assert "| S2 | 2025-05-22 | gain | small |" in report
    state = json.loads((folder / "state.json").read_text())
    assert state["final"]["review"] == "publish"
    assert leakage_audit(Trace.read_jsonl(folder / "trace.jsonl")) == []
    assert "1 normal" in capsys.readouterr().out


def test_all_scenarios(tmp_path, capsys):
    assert main(["--demo", "--scenario", "all", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count("ACME-2025-06-02") == 9 and "hold" in out
    held = (tmp_path / "9_rare_event_big_move" / "ACME-2025-06-02" / "report.md").read_text()
    assert "high impact" in held and "thin evidence" in held


def test_baseline_report_says_why(tmp_path):
    main(["--demo", "--scenario", "6 tool down", "--out", str(tmp_path)])
    report = (tmp_path / "6_tool_down" / "ACME-2025-06-02" / "report.md").read_text()
    assert "Why the baseline was reported" in report and "No adjustments" in report


def test_bad_requests_are_refused(capsys):
    assert main(["--demo", "--scenario", "nope"]) == 2
    assert main(["ZZZZ", "--as-of", "2026-07-15"]) == 2
    assert "Request refused" in capsys.readouterr().out
    assert main([]) == 2


def test_live_mode_says_when_it_is_not_available(capsys):
    assert main(["AAPL", "--as-of", "2026-07-15"]) == 3
    assert "Use --demo" in capsys.readouterr().out
