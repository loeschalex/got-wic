import numpy as np
from got_wic.combat import resolve_tick, CombatState, BuildingFight, apply_attrition
from got_wic.model import PlayerTier


def test_equal_forces_both_take_losses():
    """Equal power on both sides -> both take roughly equal losses."""
    state = CombatState(power_a=100.0, power_b=100.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    # Both should lose power, roughly equally
    assert result.power_a < 100.0
    assert result.power_b < 100.0
    assert abs(result.power_a - result.power_b) < 1e-6


def test_dominant_side_takes_fewer_losses():
    """Side A has 3x power -> Side B loses much more."""
    state = CombatState(power_a=300.0, power_b=100.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0, dt=0.1)
    loss_a = 300.0 - result.power_a
    loss_b = 100.0 - result.power_b
    assert loss_b > loss_a  # weaker side loses more


def test_zero_power_side_takes_no_action():
    """If one side has 0 power, the other takes no losses."""
    state = CombatState(power_a=100.0, power_b=0.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    assert result.power_a == 100.0
    assert result.power_b == 0.0


def test_power_never_goes_negative():
    """After a tick, power should be clamped at 0."""
    state = CombatState(power_a=1.0, power_b=10000.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    assert result.power_a >= 0.0
    assert result.power_b >= 0.0


def test_noise_produces_variance():
    """With noise, repeated ticks from same state should produce different results."""
    rng = np.random.default_rng(42)
    state = CombatState(power_a=100.0, power_b=100.0)
    results = [resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.3, rng=rng) for _ in range(50)]
    powers_a = [r.power_a for r in results]
    assert max(powers_a) - min(powers_a) > 0.1  # should vary


def test_deterministic_without_noise():
    """Without noise, same inputs should produce same outputs."""
    state = CombatState(power_a=100.0, power_b=80.0)
    r1 = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    r2 = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    assert r1.power_a == r2.power_a
    assert r1.power_b == r2.power_b


# --- Attrition / Healing tests ---


def test_attrition_removes_weakest_first():
    """Losses should deplete alts before whales."""
    tiers = [
        PlayerTier("whale", 100.0, -1),
        PlayerTier("alt", 1.0, 2),
    ]
    fight = BuildingFight(
        tiers=tiers,
        counts=[1, 10],
        healing_remaining=[0, 0],  # no healing left for alts
    )
    updated = apply_attrition(fight, losses=8.0)
    assert updated.counts[1] < 10  # alts reduced
    assert updated.counts[0] == 1  # whale untouched


def test_healing_restores_power():
    """Players with healing budget can recover from losses."""
    tiers = [PlayerTier("minnow", 8.0, 4)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[10],
        healing_remaining=[40],
    )
    updated = apply_attrition(fight, losses=16.0)
    assert updated.total_power >= fight.total_power - 16.0


def test_no_healing_permanent_loss():
    """Players with 0 healing remaining are gone forever."""
    tiers = [PlayerTier("alt", 1.0, 2)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[10],
        healing_remaining=[0],
    )
    updated = apply_attrition(fight, losses=5.0)
    assert updated.counts[0] == 5
    assert updated.total_power == 5.0


def test_whale_unlimited_healing():
    """Whales (healing_capacity=-1) always heal back."""
    tiers = [PlayerTier("whale", 100.0, -1)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[2],
        healing_remaining=[0],
    )
    updated = apply_attrition(fight, losses=100.0)
    assert updated.counts[0] == 2
