"""All design values in one place.

Each value cites the document it comes from, so the code can be checked against the design.
Change a value with `dataclasses.replace(SETTINGS, name=value)` rather than editing it in other modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# The fixed universe (full design §1): about 3 large-cap US stocks per sector,
# with liquid options and regular 10-K / 10-Q filings. AAPL, NKE and INTC are kept
# from the Checkpoint 3 runs so results stay comparable.
UNIVERSE: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "INTC": "Technology",
    "GOOGL": "Communication services", "META": "Communication services", "NFLX": "Communication services",
    "AMZN": "Consumer discretionary", "TSLA": "Consumer discretionary", "NKE": "Consumer discretionary",
    "JPM": "Financials", "GS": "Financials", "V": "Financials",
    "JNJ": "Health care", "UNH": "Health care", "LLY": "Health care",
    "PG": "Consumer staples", "KO": "Consumer staples", "WMT": "Consumer staples",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "BA": "Industrials", "LMT": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities",
    "PLD": "Real estate", "AMT": "Real estate",
    "LIN": "Materials", "FCX": "Materials",
}

OUTCOMES: tuple[str, str, str] = ("fall", "flat", "gain")


@dataclass(frozen=True)
class Settings:
    # --- Forecast definition (full design §1) ---
    horizon_days: int = 21            # trading days ahead
    sigma_lookback_days: int = 252    # one year of daily returns for sigma
    bucket_k: float = 0.5             # fall < -k*sigma, gain > +k*sigma

    # --- Data (Checkpoint 3) ---
    forms: tuple[str, ...] = ("10-Q", "10-K", "8-K")
    chunk_words: int = 150
    overlap_words: int = 30
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval (Checkpoint 3, presentation plan) ---
    fresh_days: int = 30              # search and critic only use filings from the last 30 days
    k_candidates: int = 20
    k_final: int = 5
    mmr_lambda: float = 0.7
    min_score: float = 0.20           # searches: "nothing relevant found" below this (0.30 lost most evidence; design review 5.2)
    filing_min_score: float = 0.10    # every filing in the window is a candidate; only near-empty passages fall below
    research_extract_words: int = 40  # design review 3.3: research agent sees header + short cleaned extract

    # --- Coordinator and baseline (full design §4, notebook) ---
    earnings_gap_days: int = 91       # next results estimated this long after the last (design review 3.7)
    results_max_gap_days: int = 75    # results = last item-2.02 filing within 75 days before each 10-Q / 10-K
    after_close_utc_hour: int = 20    # filed at or after this hour moves the stock the next trading day
    baseline_version: str = "C"       # design review 5.1: results days left out, blended with the past year
    ewma_decay: float = 0.94
    blend_recent: float = 0.5         # weight on recent moves; the rest on the past year
    results_jump_last: int = 4        # results-day moves averaged for the jump term (version E: 8, root mean square)
    use_vol_ratio: bool = False       # frozen implied-to-realised ratio -- design review 3.2, open until option data

    # --- Agent (Checkpoint 2, notebook) ---
    max_research_steps: int = 4
    critic_retries: int = 2
    strength_points: dict[str, int] = field(default_factory=lambda: {"small": 3, "medium": 6, "large": 9})
    max_total_shift: float = 15.0     # points, per outcome
    min_probability: float = 1.0      # percent; every outcome stays possible

    # --- Novelty (design review 5.2) ---
    novelty_sim_range: tuple[float, float] = (0.5, 0.95)   # similarity to the previous filing: new .. repeated

    # --- Market information (design review 5.3) ---
    market_ticker: str = "SPY"                       # S&P 500 fund, to separate company news from market moves
    reaction_edges: tuple[float, ...] = (-2.0, -0.5, 0.5, 2.0)   # in typical daily moves since the filing
    reaction_labels: tuple[str, ...] = ("big fall", "fall", "small", "rise", "big rise")
    prior_strength: int = 20                         # pseudo-events pulling small samples toward all events
    base_rate_scale: float = 0.02                    # how fast a base-rate score moves away from 0.5
    news_days: int = 14                              # GDELT headlines before the forecast date
    tone_base_days: int = 60
    news_story_sim: float = 0.6
    news_min_articles: int = 2
    news_max_stories: int = 2                        # news items per forecast; filings fill the rest up to 5
    news_link_days: int = 3                          # same-story rule: a story this close to a filing...
    news_link_sim: float = 0.35                      # ...and this similar is coverage of it
    news_novelty: float = 0.5
    gdelt_spacing_seconds: float = 5.0

    # --- Tree of Thought (Checkpoint 4) ---
    beam_width: int = 3
    readings_per_item: int = 3
    max_depth: int = 5
    max_model_calls: int = 25
    prune_margin: float = 0.3         # drop nodes more than this below the best
    keep_no_effect_min_beam: int = 2  # the all-"no effect" node is guaranteed only with 2+ slots

    # --- Risk gate (Checkpoint 6): no trigger -> publish, one -> flag, two or more -> hold ---
    risk_high_impact: float = 10.0    # an outcome moved this many points from the baseline
    risk_single_source: float = 5.0   # a move this large resting on one ledger line
    risk_rare_cases: int = 30         # fewer past cases than this behind a line: thin evidence
    risk_low_confidence: float = 0.5  # the model's own probability that a chosen reading is right

    # --- Evaluation (design review 3.4) ---
    main_metric: str = "log_loss_implied"
    tuning_metric: str = "log_loss_implied"
    scenarios: tuple[str, ...] = ("event", "after results", "results coming", "mid-quarter", "random")   # design review 5.4

    # --- LLM (pluggable; API default) ---
    llm_provider: str = field(default_factory=lambda: os.environ.get("MOSAIC_LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: os.environ.get("MOSAIC_LLM_MODEL", "claude-sonnet-5-5"))
    llm_temperature: float = 0.0
    run_model_budget: int = 40        # model calls for a whole forecast, retries included
    seed: int = 0

    # --- Files ---
    data_dir: str = field(default_factory=lambda: os.environ.get("MOSAIC_DATA_DIR", "data"))
    runs_dir: str = field(default_factory=lambda: os.environ.get("MOSAIC_RUNS_DIR", "runs"))
    sec_user_agent: str = field(default_factory=lambda: os.environ.get("MOSAIC_SEC_USER_AGENT", ""))


SETTINGS = Settings()
