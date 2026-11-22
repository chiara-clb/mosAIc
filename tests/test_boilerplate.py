"""The boilerplate check: every kind found in the 47-forecast run is caught, and real news is not."""

import pytest

from mosaic.boilerplate import is_boilerplate, numeric_share

BOILERPLATE = [
    "are filed as Exhibit 1.1 and Exhibit 4.1, respectively, and incorporated herein by reference.",
    "The foregoing description of the Notes is qualified in its entirety by reference to the full text",
    "Refer to Note 14 – Contingencies of the Notes to Financial Statements",
    "JPMorgan Chase & Co. (Registrant)",
    "The sale of the Notes was registered under the Company’s registration statement on Form S-3 filed on February",
    "ITEM 2.03 CREATION OF A DIRECT FINANCIAL OBLIGATION OR AN OBLIGATION UNDER AN OFF-BALANCE SHEET ARRANGEMENT OF",
    "John L. Hennessy 10,516,289,201 1,833,243,807 20,895,056",
    "Number of shares Against 1,399,727,580",
    "Legal risk is the risk of loss primarily caused by the actual or alleged failure to meet legal obligations",
    "Free cash flow $ 24,461 $ 24,551 $ 10,116 $ (5,855) $ 53,273",
    # found on Amazon, 1 July 2026, after the check was added
    "Term Loan Agreement, dated as of June 8 2026, among",
    "Credit Agreement, dated June 8, 2026, by and between",
    "pursuant to an Underwriting Agreement dated June 8, 2026 (the “Underwriting Agreement”) among the Company and the several underwriters named therein",
    "The information set forth in Item 1.01 of this Current Report on Form 8-K is hereby incorporated by reference into this Item 2.03.",
]

NEWS = [
    "Microsoft to cut 4 , 800 jobs , latest AI - driven tech layoffs",
    "Apple to spend over $30bn with Broadcom to produce 15 billion US - made chips",
    "the Firm’s Board of Directors has authorized a new common share repurchase program of $50 billion",
    "Mr. Parker will join the Company after 26 years at Microsoft Corporation",
    "Google Cloud saw a meaningful acceleration in growth as revenues increased 82% to $24.8 billion",
    "Revenue is expected to be $108.0 billion, plus or minus 2%.",
    "C$4,750,000,000 aggregate principal amount of its 5.000% notes due 2056",
    "Apple iPhone Ultra Release : Folding iPhone Tipped For Delays And Supply Issues",
    "closed the sale of C$1,250,000,000 aggregate principal amount of its 3.400% notes due 2029",
    "Amazon signs agreement to buy a stake in a chip designer",
    "Microsoft to cut 4 , 800 jobs , latest AI - driven tech layoffs",
    "Emerges As A Top - Three Global Chip Business Hidden Inside Its AI Buildout",
]


@pytest.mark.parametrize("quote", BOILERPLATE)
def test_boilerplate_found_in_the_run_is_caught(quote):
    assert is_boilerplate(quote)


@pytest.mark.parametrize("quote", NEWS)
def test_real_news_and_results_are_not_flagged(quote):
    assert not is_boilerplate(quote)


def test_empty_and_odd_input():
    assert not is_boilerplate("") and not is_boilerplate(None) and not is_boilerplate("   ")
    assert numeric_share("1,000 2,000 shares") == pytest.approx(2 / 3)
