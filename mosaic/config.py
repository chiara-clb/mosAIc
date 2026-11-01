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
    min_score: float = 0.30           # below this best cosine score: "nothing relevant found"
    research_extract_words: int = 40  # design review 3.3: research agent sees header + short cleaned extract

    # --- Coordinator and baseline (full design §4, notebook) ---
    earnings_gap_days: int = 91       # next results estimated this long after the last (design review 3.7)
    ewma_decay: float = 0.94
    vol_model: str = "ewma"           # "ewma" or "garch" -- design review 3.2, still open
    use_vol_ratio: bool = False       # frozen implied-to-realised ratio -- design review 3.2, still open

    # --- Agent (Checkpoint 2, notebook) ---
    max_research_steps: int = 4
    critic_retries: int = 2
    strength_points: dict[str, int] = field(default_factory=lambda: {"small": 3, "medium": 6, "large": 9})
    max_total_shift: float = 15.0     # points, per outcome
    min_probability: float = 1.0      # percent; every outcome stays possible

    # --- Tree of Thought (Checkpoint 4) ---
    beam_width: int = 3
    readings_per_item: int = 3
    max_depth: int = 5
    max_model_calls: int = 25
    prune_margin: float = 0.3         # drop nodes more than this below the best

    # --- Evaluation (design review 3.4) ---
    main_metric: str = "log_loss_implied"
    tuning_metric: str = "log_loss_implied"

    # --- LLM (pluggable; API default) ---
    llm_provider: str = field(default_factory=lambda: os.environ.get("MOSAIC_LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: os.environ.get("MOSAIC_LLM_MODEL", "claude-sonnet-5-5"))
    llm_temperature: float = 0.0
    seed: int = 0

    # --- Files ---
    data_dir: str = field(default_factory=lambda: os.environ.get("MOSAIC_DATA_DIR", "data"))
    runs_dir: str = field(default_factory=lambda: os.environ.get("MOSAIC_RUNS_DIR", "runs"))
    sec_user_agent: str = field(default_factory=lambda: os.environ.get("MOSAIC_SEC_USER_AGENT", ""))


SETTINGS = Settings()
