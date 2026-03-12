import numpy as np
from got_wic.model import default_config, Allocation, PlayerTier, AllianceProfile
from got_wic.montecarlo import run_monte_carlo, MonteCarloResult, save_results, load_results


def _make_alloc(total, assignments):
    return Allocation(assignments=assignments, total_armies=total)


def _make_profile(n):
    return AllianceProfile(
        tiers=[PlayerTier("minnow", 8.0, 100)],
        counts=[n],
    )


def test_monte_carlo_returns_result():
    cfg = default_config()
    a = _make_alloc(30, {"phase1": {"Stark Outpost": 10}, "phase2": {"Stark Outpost": 10}, "phase3": {"Stark Outpost": 10}})
    b = _make_alloc(15, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = run_monte_carlo(cfg, a, b, _make_profile(10), _make_profile(5), n_trials=20, noise_scale=0.1)
    assert isinstance(result, MonteCarloResult)
    assert result.n_trials == 20
    assert len(result.score_distribution_a) == 20


def test_monte_carlo_win_rate_dominant():
    """Dominant side should win most trials."""
    cfg = default_config()
    a = _make_alloc(90, {
        "phase1": {"Stark Outpost": 30, "Armory": 30, "Hot Spring": 30},
        "phase2": {"Stark Outpost": 30, "Armory": 30, "Hot Spring": 30},
        "phase3": {"Stark Outpost": 20, "Armory": 20, "Hot Spring": 20, "Stronghold": 15, "dragon": 15},
    })
    b = _make_alloc(15, {"phase1": {"Stark Outpost": 5}, "phase2": {"Stark Outpost": 5}, "phase3": {"Stark Outpost": 5}})
    result = run_monte_carlo(cfg, a, b, _make_profile(30), _make_profile(5), n_trials=50, noise_scale=0.1)
    assert result.win_rate > 0.8


def test_monte_carlo_has_percentiles():
    cfg = default_config()
    a = _make_alloc(30, {"phase1": {"Stark Outpost": 10}, "phase2": {"Stark Outpost": 10}, "phase3": {"Stark Outpost": 10}})
    b = _make_alloc(15, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = run_monte_carlo(cfg, a, b, _make_profile(10), _make_profile(5), n_trials=20, noise_scale=0.1)
    assert 5 in result.percentiles
    assert 50 in result.percentiles
    assert 95 in result.percentiles
    assert result.percentiles[5] <= result.percentiles[50] <= result.percentiles[95]


def test_monte_carlo_hopeless_detection():
    """Massively outnumbered side should be flagged hopeless."""
    cfg = default_config()
    a = _make_alloc(9, {"phase1": {"Stark Outpost": 3}, "phase2": {"Stark Outpost": 3}, "phase3": {"Stark Outpost": 3}})
    b = _make_alloc(300, {
        "phase1": {"Stark Outpost": 50, "Armory": 50, "Greyjoy Outpost": 50},
        "phase2": {"Stark Outpost": 50, "Armory": 50, "Greyjoy Outpost": 50},
        "phase3": {"Stark Outpost": 30, "Armory": 30, "Greyjoy Outpost": 30, "Stronghold": 30, "dragon": 30},
    })
    result = run_monte_carlo(cfg, a, b, _make_profile(3), _make_profile(100), n_trials=30, noise_scale=0.1)
    assert result.hopeless is True


def test_save_and_load_results(tmp_path):
    cfg = default_config()
    a = _make_alloc(30, {"phase1": {"Stark Outpost": 10}, "phase2": {"Stark Outpost": 10}, "phase3": {"Stark Outpost": 10}})
    b = _make_alloc(15, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = run_monte_carlo(cfg, a, b, _make_profile(10), _make_profile(5), n_trials=20, noise_scale=0.1)

    path = tmp_path / "test_result.npz"
    save_results(result, path)
    assert path.exists()

    loaded = load_results(path)
    assert loaded.n_trials == result.n_trials
    assert loaded.win_rate == result.win_rate
    assert np.allclose(loaded.score_distribution_a, result.score_distribution_a)
