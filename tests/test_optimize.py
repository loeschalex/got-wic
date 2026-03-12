from got_wic.model import default_config, AllianceProfile, PlayerTier
from got_wic.optimize import optimize, OptResult


def _make_profile(n):
    return AllianceProfile(
        tiers=[PlayerTier("minnow", 8.0, 100)],
        counts=[n],
    )


def test_optimize_returns_results_with_mc_stats():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
    )
    best = results[0]
    assert isinstance(best, OptResult)
    assert hasattr(best, "win_rate")
    assert hasattr(best, "mean_score_a")
    assert hasattr(best, "std_score_a")


def test_optimize_ranking_aggressive():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="aggressive",
    )
    # Should be sorted by mean_score_a descending
    for i in range(len(results) - 1):
        assert results[i].mean_score_a >= results[i + 1].mean_score_a


def test_optimize_ranking_conservative():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="conservative",
    )
    # Should be sorted by p25 descending
    for i in range(len(results) - 1):
        assert results[i].p25_score_a >= results[i + 1].p25_score_a


def test_optimize_ranking_win_focused():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="win_focused",
    )
    # Should be sorted by win_rate descending
    for i in range(len(results) - 1):
        assert results[i].win_rate >= results[i + 1].win_rate
