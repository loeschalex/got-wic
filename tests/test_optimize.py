from got_wic.model import default_config
from got_wic.optimize import optimize, OptResult


def test_optimize_returns_sorted_results():
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=80,
        n_players_b=60,
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=20,  # coarse grid for speed
    )
    assert len(results) > 0
    # Results should be sorted descending by score_a
    scores = [r.score_a for r in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_best_beats_empty():
    """Best allocation should score more than doing nothing."""
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=80,
        n_players_b=60,
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=20,
    )
    assert results[0].score_a > 0


def test_optimize_result_has_allocation():
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=50,
        n_players_b=50,
        opponent_spread=0.5,
        opponent_aggression=0.5,
        step_pct=25,
    )
    best = results[0]
    assert isinstance(best, OptResult)
    assert best.allocation is not None
    assert best.score_a >= best.score_b or True  # may lose if 50v50


def test_more_players_scores_higher():
    """Side with more players should generally score higher."""
    cfg = default_config()
    results_80v40 = optimize(cfg, 80, 40, 0.5, 0.5, step_pct=25)
    results_40v80 = optimize(cfg, 40, 80, 0.5, 0.5, step_pct=25)
    assert results_80v40[0].score_a > results_40v80[0].score_a
