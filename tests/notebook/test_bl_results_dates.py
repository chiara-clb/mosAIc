from datetime import datetime, timedelta, timezone

from tests.notebook.core import load_baseline_core

ns = load_baseline_core()
UTC = timezone.utc


def d(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC, hour=21)


def test_one_results_per_quarter_skips_interim_updates():
    # Chevron-style: interim update early in the month, real results at the end, 10-Q a few weeks later
    r202 = [d("2025-04-08"), d("2025-05-02"), d("2025-07-09"), d("2025-08-01")]
    reports = [d("2025-05-07"), d("2025-08-06")]
    assert ns["main_results"](r202, reports) == [d("2025-05-02"), d("2025-08-01")]


def test_results_after_the_latest_report_are_kept():
    r202 = [d("2025-05-02"), d("2025-07-09"), d("2025-08-01")]
    reports = [d("2025-05-07")]
    assert ns["main_results"](r202, reports) == [d("2025-05-02"), d("2025-08-01")]


def test_reports_without_results_and_duplicates():
    r202 = [d("2025-05-02"), d("2025-05-02")]
    reports = [d("2025-02-20"), d("2025-05-07")]          # nothing in the 75 days before February's report
    assert ns["main_results"](r202, reports) == [d("2025-05-02")]


def test_results_too_long_before_the_report_are_not_matched():
    r202 = [d("2025-01-02")]
    reports = [d("2025-05-07")]
    assert ns["main_results"](r202, reports) == []


def test_no_reports_keeps_the_latest_only():
    assert ns["main_results"]([d("2025-01-02"), d("2025-04-02")], []) == [d("2025-04-02")]
