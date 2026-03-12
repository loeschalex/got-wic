
import numpy as np

from got_wic import default_config, optimize, default_alliance_profile
from got_wic.montecarlo import run_monte_carlo, save_results, load_results


def test_full_pipeline_dominant_side():
    """80 vs 40 players: dominant side should win consistently."""
    cfg = default_config()
    pa = default_alliance_profile(80)
    pb = default_alliance_profile(40)
    results = optimize(
        cfg, pa, pb,
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=20,
    )
    best = results[0]
    assert best.mean_score_a > 0
    assert best.win_rate > 0.5


def test_full_pipeline_save_load(tmp_path):
    """Run MC, save to npz, load back, verify data integrity."""
    cfg = default_config()
    from got_wic.model import Allocation, AllianceProfile, PlayerTier
    pa = AllianceProfile(tiers=[PlayerTier("minnow", 8.0, 100)], counts=[20])
    pb = AllianceProfile(tiers=[PlayerTier("minnow", 8.0, 100)], counts=[20])
    a = Allocation(
        assignments={"phase1": {"Stark Outpost": 20}, "phase2": {"Stark Outpost": 20}, "phase3": {"Stark Outpost": 20}},
        total_armies=60,
    )
    b = Allocation(
        assignments={"phase1": {"Greyjoy Outpost": 20}, "phase2": {"Greyjoy Outpost": 20}, "phase3": {"Greyjoy Outpost": 20}},
        total_armies=60,
    )
    result = run_monte_carlo(cfg, a, b, pa, pb, n_trials=30, noise_scale=0.1)
    path = tmp_path / "test.npz"
    save_results(result, path)
    loaded = load_results(path)
    assert loaded.n_trials == 30
    assert np.allclose(loaded.score_distribution_a, result.score_distribution_a)


def test_hopeless_scenario():
    """3 vs 100 players should be flagged as hopeless."""
    cfg = default_config()
    pa = default_alliance_profile(3)
    pb = default_alliance_profile(100)
    results = optimize(
        cfg, pa, pb,
        opponent_spread=0.5,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=20,
    )
    best = results[0]
    assert best.hopeless is True
