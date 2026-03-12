from got_wic.model import default_config, Allocation, PlayerTier, AllianceProfile
from got_wic.simulate import simulate


def _make_alloc(total: int, phase_assignments: dict[str, dict[str, int]]) -> Allocation:
    return Allocation(assignments=phase_assignments, total_armies=total)


def _make_profile(n_players: int) -> AllianceProfile:
    """Simple profile: all minnows for testing predictability."""
    return AllianceProfile(
        tiers=[PlayerTier("minnow", 8.0, 100)],  # high healing so no attrition in basic tests
        counts=[n_players],
    )


def test_empty_allocation_scores_zero():
    cfg = default_config()
    a = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    pa = _make_profile(20)
    pb = _make_profile(20)
    result = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    assert result.score_a == 0
    assert result.score_b == 0


def test_dominant_side_wins_building():
    """Side A sends 30 armies to Stark Outpost, Side B sends 5. A should win."""
    cfg = default_config()
    a = _make_alloc(90, {
        "phase1": {"Stark Outpost": 30},
        "phase2": {"Stark Outpost": 30},
        "phase3": {"Stark Outpost": 30},
    })
    b = _make_alloc(15, {
        "phase1": {"Stark Outpost": 5},
        "phase2": {"Stark Outpost": 5},
        "phase3": {"Stark Outpost": 5},
    })
    pa = _make_profile(30)
    pb = _make_profile(5)
    result = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    # A should capture and hold -- positive score
    assert result.score_a > 0
    assert result.breakdown_a["first_capture"] > 0
    assert result.breakdown_a["hold"] > 0


def test_result_has_timeline():
    """SimResult should include per-tick power levels."""
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(30, {"phase1": {}, "phase2": {}, "phase3": {}})
    pa = _make_profile(10)
    pb = _make_profile(10)
    result = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    assert len(result.timeline) == cfg.match_duration


def test_result_has_healing_and_casualty_stats():
    """SimResult should track healing spent and casualties."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    b = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    pa = _make_profile(20)
    pb = _make_profile(20)
    result = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    assert hasattr(result, "healing_spent_a")


def test_deterministic_without_noise():
    """noise_scale=0 should produce identical results on repeated runs."""
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(15, {
        "phase1": {"Stark Outpost": 5},
        "phase2": {"Stark Outpost": 5},
        "phase3": {"Stark Outpost": 5},
    })
    pa = _make_profile(10)
    pb = _make_profile(5)
    r1 = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    r2 = simulate(cfg, a, b, pa, pb, noise_scale=0.0)
    assert r1.score_a == r2.score_a
    assert r1.score_b == r2.score_b
