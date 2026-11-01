from dataclasses import replace

from mosaic.config import OUTCOMES, SETTINGS, UNIVERSE


def test_universe_has_30_unique_tickers():
    assert len(UNIVERSE) == 30
    assert len(set(UNIVERSE)) == 30


def test_universe_covers_every_sector_with_at_most_four_stocks():
    sectors = {}
    for sector in UNIVERSE.values():
        sectors[sector] = sectors.get(sector, 0) + 1
    assert len(sectors) == 11
    assert all(1 <= n <= 4 for n in sectors.values())


def test_checkpoint_stocks_kept():
    assert {"AAPL", "NKE", "INTC"} <= set(UNIVERSE)


def test_forecast_definition_matches_design():
    assert SETTINGS.horizon_days == 21
    assert SETTINGS.sigma_lookback_days == 252
    assert SETTINGS.bucket_k == 0.5
    assert OUTCOMES == ("fall", "flat", "gain")


def test_retrieval_values_match_checkpoint_3():
    s = SETTINGS
    assert (s.chunk_words, s.overlap_words) == (150, 30)
    assert s.embed_model.endswith("all-MiniLM-L6-v2")
    assert (s.fresh_days, s.k_candidates, s.k_final) == (30, 20, 5)
    assert s.mmr_lambda == 0.7
    assert s.min_score == 0.30


def test_ledger_and_tot_values_match_design():
    s = SETTINGS
    assert s.strength_points == {"small": 3, "medium": 6, "large": 9}
    assert s.max_total_shift == 15
    assert (s.beam_width, s.readings_per_item, s.max_depth, s.max_model_calls) == (3, 3, 5, 25)
    assert s.prune_margin == 0.3
    assert s.critic_retries == 2


def test_decided_options():
    assert SETTINGS.research_extract_words == 40          # design review 3.3
    assert SETTINGS.main_metric == SETTINGS.tuning_metric == "log_loss_implied"  # 3.4
    assert SETTINGS.llm_temperature == 0.0


def test_settings_are_frozen_and_replaceable():
    changed = replace(SETTINGS, mmr_lambda=0.5)
    assert changed.mmr_lambda == 0.5 and SETTINGS.mmr_lambda == 0.7
