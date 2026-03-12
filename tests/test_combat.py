import numpy as np
from got_wic.combat import resolve_tick, CombatState


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
