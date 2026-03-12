from got_wic import default_config, optimize


def test_full_pipeline_80v60():
    """End-to-end: optimize for 80v60, check we get actionable results."""
    cfg = default_config()
    results = optimize(cfg, 80, 60, opponent_spread=0.7, opponent_aggression=0.5, step_pct=25)
    best = results[0]

    # Should produce a positive score
    assert best.score_a > 0
    # Should beat the opponent (80 > 60 players)
    assert best.score_a > best.score_b
    # Breakdown should sum to total
    total = sum(best.breakdown_a.values())
    assert total == best.score_a


def test_full_pipeline_50v50_close_game():
    """50v50 should be a close game."""
    cfg = default_config()
    results = optimize(cfg, 50, 50, opponent_spread=0.5, opponent_aggression=0.5, step_pct=25)
    best = results[0]
    assert best.score_a > 0
    # Score difference should be relatively small compared to total
    diff = abs(best.score_a - best.score_b)
    assert diff < best.score_a  # not a blowout
