from got_wic.model import default_config
from got_wic.opponent import generate_opponent


def test_even_spread_distributes_equally():
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=0.5)
    # With full spread, armies should be distributed across all available objectives per phase
    phase3_armies = alloc.armies_used("phase3")
    assert phase3_armies > 0
    assert phase3_armies <= 300


def test_zero_spread_concentrates():
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=0.0, aggression=0.5)
    # With zero spread, all armies go to highest-value objective
    phase3 = alloc.assignments["phase3"]
    counts = list(phase3.values())
    assert max(counts) == sum(counts)  # all in one objective


def test_defensive_stays_own_side():
    """aggression=0 means opponent (Side B = Greyjoy) stays on own side."""
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=0.0)
    phase1 = alloc.assignments["phase1"]
    # Should not send armies to Stark Outpost (enemy territory)
    assert phase1.get("Stark Outpost", 0) == 0


def test_aggressive_pushes_enemy_side():
    """aggression=1 means opponent pushes into enemy territory."""
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=1.0)
    phase1 = alloc.assignments["phase1"]
    # Should send armies to Stark Outpost (enemy territory)
    assert phase1.get("Stark Outpost", 0) > 0


def test_allocation_never_exceeds_total():
    cfg = default_config()
    for spread in [0.0, 0.5, 1.0]:
        for aggression in [0.0, 0.5, 1.0]:
            alloc = generate_opponent(cfg, total_armies=300, spread=spread, aggression=aggression)
            for phase in ["phase1", "phase2", "phase3"]:
                assert alloc.is_valid(phase), f"Overcommit at spread={spread}, agg={aggression}, {phase}"
