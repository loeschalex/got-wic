from got_wic.model import (
    Allocation,
    AllianceProfile,
    Dragon,
    Objective,
    PlayerTier,
    default_alliance_profile,
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


# --- PlayerTier / AllianceProfile tests ---


def test_player_tier_creation():
    tier = PlayerTier(name="whale", combat_power=100.0, healing_capacity=-1)
    assert tier.name == "whale"
    assert tier.combat_power == 100.0
    assert tier.healing_capacity == -1  # unlimited


def test_alliance_profile_total_players():
    profile = AllianceProfile(
        tiers=[
            PlayerTier("whale", 100.0, -1),
            PlayerTier("dolphin", 30.0, 8),
            PlayerTier("minnow", 8.0, 4),
            PlayerTier("alt", 1.0, 2),
        ],
        counts=[2, 15, 25, 38],
    )
    assert profile.total_players == 80
    assert profile.total_power == 2 * 100 + 15 * 30 + 25 * 8 + 38 * 1


def test_alliance_profile_validation_mismatched_lengths():
    """tiers and counts must have same length."""
    import pytest
    with pytest.raises(ValueError):
        AllianceProfile(
            tiers=[PlayerTier("whale", 100.0, -1)],
            counts=[2, 15],
        )


def test_default_alliance_profile():
    profile = default_alliance_profile(total_players=80)
    assert profile.total_players == 80
    assert len(profile.tiers) == 4
    assert profile.tiers[0].name == "whale"
    assert profile.tiers[3].name == "alt"
