from got_wic.model import (
    Objective,
    Dragon,
    TreasureConfig,
    GameConfig,
    Allocation,
    default_config,
)


def test_objective_creation():
    obj = Objective(
        name="Stark Outpost",
        count=2,
        first_capture=200,
        hold_pts_min=80,
        opens_at=0,
        zone="near_stark",
    )
    assert obj.name == "Stark Outpost"
    assert obj.count == 2
    assert obj.hold_pts_min == 80


def test_dragon_creation():
    d = Dragon(name="Winter Ice", escort_pts=3000, spawns_at=12)
    assert d.escort_pts == 3000


def test_default_config_has_correct_objectives():
    cfg = default_config()
    names = [o.name for o in cfg.objectives]
    assert "Stark Outpost" in names
    assert "Stronghold" in names
    assert "Armory" in names
    assert "Hot Spring" in names
    assert "Greyjoy Outpost" in names
    assert len(cfg.objectives) == 5  # Winterfell excluded (not capturable)


def test_default_config_has_dragons():
    cfg = default_config()
    assert len(cfg.dragons) == 2
    assert cfg.dragons[0].escort_pts == 3000


def test_default_config_phases():
    cfg = default_config()
    assert cfg.phase_boundaries == [0, 8, 12]
    assert cfg.match_duration == 60


def test_allocation_total_armies():
    alloc = Allocation(
        assignments={"phase1": {"Stark Outpost": 40, "Armory": 20}},
        total_armies=180,
    )
    assert alloc.armies_used("phase1") == 60
    assert alloc.armies_unused("phase1") == 120


def test_allocation_rejects_overcommit():
    alloc = Allocation(
        assignments={"phase1": {"Stark Outpost": 100, "Armory": 100}},
        total_armies=180,
    )
    assert not alloc.is_valid("phase1")


def test_default_config_total_objectives_count():
    """Sum of all objective counts should be 10 (excludes Winterfell)."""
    cfg = default_config()
    total = sum(o.count for o in cfg.objectives)
    assert total == 10
