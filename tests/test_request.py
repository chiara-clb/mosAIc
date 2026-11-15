from datetime import datetime, timezone

import pytest

from mosaic.request import RequestRefused, check_request, require

NOW = datetime(2026, 11, 15, tzinfo=timezone.utc)
AS_OF = datetime(2026, 7, 15, tzinfo=timezone.utc)


def test_a_valid_request_passes():
    assert check_request("AAPL", AS_OF, price_days_before=400, probes={"SEC EDGAR": lambda: True}, now=NOW) == []


@pytest.mark.parametrize("kw,word", [
    ({"ticker": "ZZZZ"}, "not one of the 30"),
    ({"as_of": datetime(2027, 1, 4, tzinfo=timezone.utc)}, "in the future"),
    ({"price_days_before": 100}, "only 100 days"),
    ({"probes": {"GDELT": lambda: False}}, "GDELT cannot be reached"),
    ({"probes": {"GDELT": lambda: 1 / 0}}, "GDELT cannot be reached"),
])
def test_each_reason_to_refuse(kw, word):
    args = {"ticker": "AAPL", "as_of": AS_OF, "price_days_before": 400, "now": NOW, **kw}
    reasons = check_request(args.pop("ticker"), args.pop("as_of"), **args)
    assert len(reasons) == 1 and word in reasons[0], reasons


def test_naive_dates_are_read_as_utc_and_the_demo_universe_can_be_passed():
    assert check_request("ACME", datetime(2025, 6, 2), universe={"ACME"}, now=NOW) == []


def test_require_raises_with_all_reasons():
    with pytest.raises(RequestRefused) as e:
        require("ZZZZ", datetime(2027, 1, 4, tzinfo=timezone.utc), now=NOW)
    assert len(e.value.reasons) == 2
