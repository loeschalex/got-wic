from got_wic.model import default_config, Allocation
from got_wic.simulate import simulate


def _make_alloc(total: int, phase_assignments: dict[str, dict[str, int]]) -> Allocation:
    return Allocation(assignments=phase_assignments, total_armies=total)


def test_empty_allocation_scores_zero():
    cfg = default_config()
    a = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 0
    assert result.score_b == 0


def test_uncontested_outpost_scores_hold_points():
    """Side A sends 10 armies to Stark Outpost, Side B sends 0.
    Outpost opens at minute 0, holds for 60 minutes.
    2 outposts * 80 pts/min * 60 min = 9600, plus first capture 2*200 = 400.
    Total = 10000."""
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(30, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000


def test_contested_objective_winner_takes_points():
    """Side A sends 20 to Stark Outpost, Side B sends 10.
    Side A wins majority, gets all points."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    b = _make_alloc(60, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000
    assert result.score_b == 0


def test_strongholds_only_score_after_minute_12():
    """Strongholds open at minute 12. Hold for 48 minutes.
    4 * 180 * 48 = 34560, plus first capture 4 * 600 = 2400. Total = 36960."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {},
        "phase2": {},
        "phase3": {"Stronghold": 30},
    })
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 36_960


def test_dragon_escort_scores_points():
    """Side A sends armies to dragon, Side B sends 0. Should get escort pts."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {},
        "phase2": {},
        "phase3": {"dragon": 30},
    })
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a >= 6000  # 2 dragons * 3000


def test_result_has_breakdown():
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(30, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.breakdown_a["first_capture"] == 400
    assert result.breakdown_a["hold"] == 9600
    assert result.breakdown_a["dragon"] == 0


def test_both_sides_score_different_objectives():
    """A holds Stark Outposts, B holds Greyjoy Outposts. Both score."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    b = _make_alloc(60, {
        "phase1": {"Greyjoy Outpost": 20},
        "phase2": {"Greyjoy Outpost": 20},
        "phase3": {"Greyjoy Outpost": 20},
    })
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000
    assert result.score_b == 10_000
