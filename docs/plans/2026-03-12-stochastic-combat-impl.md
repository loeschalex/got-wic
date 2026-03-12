# Stochastic Combat Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace the deterministic majority-wins combat model with a stochastic Lanchester-based simulator, adding player composition tiers, Monte Carlo orchestration, and risk-adjusted optimizer output.

**Architecture:** Four-module layered approach — `combat.py` (Lanchester fight resolution) → updated `simulate.py` (uses combat.py) → `montecarlo.py` (N-trial orchestration + `.npz` export) → updated `optimize.py` (ranks by MC stats). New `AllianceProfile` and `PlayerTier` dataclasses in `model.py`.

**Tech Stack:** Python 3.13+, numpy, pytest, marimo

---

### Task 0: Add PlayerTier and AllianceProfile to model.py

**Files:**
- Modify: `src/got_wic/model.py`
- Test: `tests/test_model.py`

**Step 1: Write the failing tests**

Add to `tests/test_model.py`:

```python
from got_wic.model import PlayerTier, AllianceProfile, default_alliance_profile


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v -k "tier or alliance or profile"`
Expected: FAIL — `ImportError: cannot import name 'PlayerTier'`

**Step 3: Write minimal implementation**

Add to `src/got_wic/model.py`:

```python
@dataclass(frozen=True)
class PlayerTier:
    name: str
    combat_power: float
    healing_capacity: int  # -1 = unlimited


@dataclass(frozen=True)
class AllianceProfile:
    tiers: list[PlayerTier]
    counts: list[int]

    def __post_init__(self):
        if len(self.tiers) != len(self.counts):
            raise ValueError(
                f"tiers ({len(self.tiers)}) and counts ({len(self.counts)}) must have same length"
            )

    @property
    def total_players(self) -> int:
        return sum(self.counts)

    @property
    def total_power(self) -> float:
        return sum(t.combat_power * c for t, c in zip(self.tiers, self.counts))


def default_alliance_profile(total_players: int) -> AllianceProfile:
    """Create a default 4-tier alliance profile for a given player count.

    Distributes players roughly: 3% whale, 17% dolphin, 30% minnow, 50% alt.
    """
    import math
    n_whale = max(1, round(total_players * 0.03))
    n_dolphin = round(total_players * 0.17)
    n_minnow = round(total_players * 0.30)
    n_alt = total_players - n_whale - n_dolphin - n_minnow
    tiers = [
        PlayerTier("whale", 100.0, -1),
        PlayerTier("dolphin", 30.0, 8),
        PlayerTier("minnow", 8.0, 4),
        PlayerTier("alt", 1.0, 2),
    ]
    return AllianceProfile(tiers=tiers, counts=[n_whale, n_dolphin, n_minnow, n_alt])
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All PASS

**Step 5: Update `__init__.py` exports**

Add `PlayerTier`, `AllianceProfile`, `default_alliance_profile` to `src/got_wic/__init__.py` imports and `__all__`.

**Step 6: Commit**

```bash
git add src/got_wic/model.py src/got_wic/__init__.py tests/test_model.py
git commit -m "feat: add PlayerTier and AllianceProfile dataclasses"
```

---

### Task 1: Implement Lanchester combat resolution

**Files:**
- Create: `src/got_wic/combat.py`
- Test: `tests/test_combat.py`

**Step 1: Write the failing tests**

Create `tests/test_combat.py`:

```python
import numpy as np
from got_wic.combat import resolve_tick, CombatState


def test_equal_forces_both_take_losses():
    """Equal power on both sides → both take roughly equal losses."""
    state = CombatState(power_a=100.0, power_b=100.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
    # Both should lose power, roughly equally
    assert result.power_a < 100.0
    assert result.power_b < 100.0
    assert abs(result.power_a - result.power_b) < 1e-6


def test_dominant_side_takes_fewer_losses():
    """Side A has 3x power → Side B loses much more."""
    state = CombatState(power_a=300.0, power_b=100.0)
    result = resolve_tick(state, alpha=1.0, beta=1.0, noise_scale=0.0)
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_combat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'got_wic.combat'`

**Step 3: Write minimal implementation**

Create `src/got_wic/combat.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CombatState:
    power_a: float
    power_b: float


def resolve_tick(
    state: CombatState,
    alpha: float = 1.0,
    beta: float = 1.0,
    noise_scale: float = 0.0,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> CombatState:
    """Resolve one tick of Lanchester combat.

    loss_a = beta * power_b * dt (+ noise)
    loss_b = alpha * power_a * dt (+ noise)
    """
    if state.power_a <= 0 or state.power_b <= 0:
        return CombatState(
            power_a=max(0.0, state.power_a),
            power_b=max(0.0, state.power_b),
        )

    eff_alpha = alpha
    eff_beta = beta

    if noise_scale > 0 and rng is not None:
        eff_alpha *= 1.0 + rng.normal(0, noise_scale)
        eff_beta *= 1.0 + rng.normal(0, noise_scale)
        eff_alpha = max(0.0, eff_alpha)
        eff_beta = max(0.0, eff_beta)

    loss_a = eff_beta * state.power_b * dt
    loss_b = eff_alpha * state.power_a * dt

    return CombatState(
        power_a=max(0.0, state.power_a - loss_a),
        power_b=max(0.0, state.power_b - loss_b),
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_combat.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/got_wic/combat.py tests/test_combat.py
git commit -m "feat: add Lanchester combat resolution module"
```

---

### Task 2: Add healing and attrition to combat

**Files:**
- Modify: `src/got_wic/combat.py`
- Test: `tests/test_combat.py`

**Step 1: Write the failing tests**

Add to `tests/test_combat.py`:

```python
from got_wic.model import PlayerTier, AllianceProfile
from got_wic.combat import BuildingFight, apply_attrition


def test_attrition_removes_weakest_first():
    """Losses should deplete alts before minnows."""
    tiers = [
        PlayerTier("whale", 100.0, -1),
        PlayerTier("alt", 1.0, 2),
    ]
    fight = BuildingFight(
        tiers=tiers,
        counts=[1, 10],
        healing_remaining=[0, 20],  # whales: unlimited tracked as 0 (sentinel); alts: 20 heals
    )
    # Apply 8 power worth of losses (should wipe 8 alts)
    updated = apply_attrition(fight, losses=8.0)
    assert updated.counts[1] < 10  # alts reduced
    assert updated.counts[0] == 1  # whale untouched


def test_healing_restores_power():
    """Players with healing budget can recover from losses."""
    tiers = [PlayerTier("minnow", 8.0, 4)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[10],
        healing_remaining=[40],  # 10 players * 4 heals each
    )
    updated = apply_attrition(fight, losses=16.0)  # kill 2 minnows
    # With healing, those 2 minnows should come back (if healing available)
    assert updated.total_power >= fight.total_power - 16.0


def test_no_healing_permanent_loss():
    """Players with 0 healing remaining are gone forever."""
    tiers = [PlayerTier("alt", 1.0, 2)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[10],
        healing_remaining=[0],  # no healing left
    )
    updated = apply_attrition(fight, losses=5.0)
    assert updated.counts[0] == 5  # 5 alts permanently gone
    assert updated.total_power == 5.0


def test_whale_unlimited_healing():
    """Whales (healing_capacity=-1) always heal back."""
    tiers = [PlayerTier("whale", 100.0, -1)]
    fight = BuildingFight(
        tiers=tiers,
        counts=[2],
        healing_remaining=[0],  # sentinel for unlimited
    )
    updated = apply_attrition(fight, losses=100.0)
    # Whale should heal back
    assert updated.counts[0] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_combat.py -v -k "attrition or healing"`
Expected: FAIL — `ImportError: cannot import name 'BuildingFight'`

**Step 3: Write minimal implementation**

Add to `src/got_wic/combat.py`:

```python
@dataclass
class BuildingFight:
    tiers: list[PlayerTier]
    counts: list[int]  # current player counts per tier
    healing_remaining: list[int]  # remaining heals per tier (aggregate)

    @property
    def total_power(self) -> float:
        return sum(t.combat_power * c for t, c in zip(self.tiers, self.counts))


def apply_attrition(fight: BuildingFight, losses: float) -> BuildingFight:
    """Apply losses bottom-up through tiers. Players with healing come back."""
    new_counts = list(fight.counts)
    new_healing = list(fight.healing_remaining)
    remaining_loss = losses

    # Apply losses from weakest tier upward
    for i in range(len(fight.tiers) - 1, -1, -1):
        tier = fight.tiers[i]
        if remaining_loss <= 0 or new_counts[i] == 0:
            continue

        # How many players can this loss kill?
        kills = min(new_counts[i], int(remaining_loss / tier.combat_power))
        if kills == 0 and remaining_loss >= tier.combat_power:
            kills = 1
        kills = min(kills, new_counts[i])
        remaining_loss -= kills * tier.combat_power

        # Healing: unlimited (-1) → always come back, otherwise spend from budget
        if tier.healing_capacity == -1:
            # Whales always heal — counts stay the same
            pass
        elif new_healing[i] >= kills:
            new_healing[i] -= kills
            # Players heal back — counts stay the same
        else:
            # Partial healing
            healed = new_healing[i]
            new_healing[i] = 0
            new_counts[i] -= (kills - healed)

    return BuildingFight(
        tiers=fight.tiers,
        counts=new_counts,
        healing_remaining=new_healing,
    )
```

Add import at top of `combat.py`:

```python
from got_wic.model import PlayerTier
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_combat.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/got_wic/combat.py tests/test_combat.py
git commit -m "feat: add healing and tier-based attrition to combat"
```

---

### Task 3: Update simulate.py to use Lanchester combat

**Files:**
- Modify: `src/got_wic/simulate.py`
- Modify: `tests/test_simulate.py`

**Step 1: Write the failing tests**

Replace `tests/test_simulate.py` with tests that use `AllianceProfile`:

```python
import numpy as np
from got_wic.model import default_config, Allocation, PlayerTier, AllianceProfile
from got_wic.simulate import simulate, SimResult


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
    # A should capture and hold — positive score
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
    assert "healing_spent_a" in result.breakdown_a or hasattr(result, "healing_spent_a")


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulate.py -v`
Expected: FAIL — `simulate() got unexpected keyword argument 'noise_scale'` or similar

**Step 3: Rewrite simulate.py**

Rewrite `src/got_wic/simulate.py` to use `combat.py`. The updated `simulate()` function signature:

```python
def simulate(
    cfg: GameConfig,
    alloc_a: Allocation,
    alloc_b: Allocation,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    noise_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SimResult:
```

Key changes:
- Accept `AllianceProfile` for both sides
- Distribute players from profile to buildings based on allocation percentages
- Use `resolve_tick()` + `apply_attrition()` per building per minute
- Track per-tick power at each building in timeline
- Track healing spent and casualties in result
- `rng` parameter for reproducible stochastic runs

Updated `SimResult`:

```python
@dataclass
class SimResult:
    score_a: int
    score_b: int
    breakdown_a: dict[str, int | float]
    breakdown_b: dict[str, int | float]
    timeline: list[dict]
    healing_spent_a: int
    healing_spent_b: int
    casualties_a: dict[str, int]  # tier_name -> count lost
    casualties_b: dict[str, int]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulate.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/got_wic/simulate.py tests/test_simulate.py
git commit -m "feat: update simulation engine with Lanchester combat"
```

---

### Task 4: Implement Monte Carlo orchestration

**Files:**
- Create: `src/got_wic/montecarlo.py`
- Test: `tests/test_montecarlo.py`

**Step 1: Write the failing tests**

Create `tests/test_montecarlo.py`:

```python
import numpy as np
from pathlib import Path
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_montecarlo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'got_wic.montecarlo'`

**Step 3: Write minimal implementation**

Create `src/got_wic/montecarlo.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from got_wic.model import GameConfig, Allocation, AllianceProfile
from got_wic.simulate import simulate


@dataclass
class MonteCarloResult:
    n_trials: int
    mean_score_a: float
    std_score_a: float
    mean_score_b: float
    std_score_b: float
    win_rate: float
    score_distribution_a: np.ndarray
    score_distribution_b: np.ndarray
    percentiles: dict[int, float]
    hopeless: bool
    alloc_a: Allocation
    alloc_b: Allocation
    profile_a: AllianceProfile
    profile_b: AllianceProfile
    config: GameConfig
    timestamp: str


def run_monte_carlo(
    cfg: GameConfig,
    alloc_a: Allocation,
    alloc_b: Allocation,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    n_trials: int = 500,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    scores_a = np.empty(n_trials)
    scores_b = np.empty(n_trials)

    for i in range(n_trials):
        result = simulate(cfg, alloc_a, alloc_b, profile_a, profile_b, noise_scale=noise_scale, rng=rng)
        scores_a[i] = result.score_a
        scores_b[i] = result.score_b

    wins = np.sum(scores_a > scores_b)
    pcts = {p: float(np.percentile(scores_a, p)) for p in [5, 25, 50, 75, 95]}

    return MonteCarloResult(
        n_trials=n_trials,
        mean_score_a=float(np.mean(scores_a)),
        std_score_a=float(np.std(scores_a)),
        mean_score_b=float(np.mean(scores_b)),
        std_score_b=float(np.std(scores_b)),
        win_rate=float(wins / n_trials),
        score_distribution_a=scores_a,
        score_distribution_b=scores_b,
        percentiles=pcts,
        hopeless=float(wins / n_trials) < 0.05,
        alloc_a=alloc_a,
        alloc_b=alloc_b,
        profile_a=profile_a,
        profile_b=profile_b,
        config=cfg,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_results(result: MonteCarloResult, path: Path) -> None:
    metadata = {
        "n_trials": result.n_trials,
        "mean_score_a": result.mean_score_a,
        "std_score_a": result.std_score_a,
        "mean_score_b": result.mean_score_b,
        "std_score_b": result.std_score_b,
        "win_rate": result.win_rate,
        "hopeless": result.hopeless,
        "percentiles": result.percentiles,
        "timestamp": result.timestamp,
    }
    np.savez_compressed(
        path,
        score_distribution_a=result.score_distribution_a,
        score_distribution_b=result.score_distribution_b,
        metadata=json.dumps(metadata),
    )


def load_results(path: Path) -> MonteCarloResult:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return MonteCarloResult(
        n_trials=metadata["n_trials"],
        mean_score_a=metadata["mean_score_a"],
        std_score_a=metadata["std_score_a"],
        mean_score_b=metadata["mean_score_b"],
        std_score_b=metadata["std_score_b"],
        win_rate=metadata["win_rate"],
        score_distribution_a=data["score_distribution_a"],
        score_distribution_b=data["score_distribution_b"],
        percentiles={int(k): v for k, v in metadata["percentiles"].items()},
        hopeless=metadata["hopeless"],
        alloc_a=None,  # not serialized in v1
        alloc_b=None,
        profile_a=None,
        profile_b=None,
        config=None,
        timestamp=metadata["timestamp"],
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_montecarlo.py -v`
Expected: All PASS

**Step 5: Update `__init__.py` exports**

Add `MonteCarloResult`, `run_monte_carlo`, `save_results`, `load_results` to exports.

**Step 6: Commit**

```bash
git add src/got_wic/montecarlo.py tests/test_montecarlo.py src/got_wic/__init__.py
git commit -m "feat: add Monte Carlo orchestration with npz persistence"
```

---

### Task 5: Update optimizer to use Monte Carlo

**Files:**
- Modify: `src/got_wic/optimize.py`
- Modify: `tests/test_optimize.py`

**Step 1: Write the failing tests**

Replace `tests/test_optimize.py`:

```python
from got_wic.model import default_config, AllianceProfile, PlayerTier, default_alliance_profile
from got_wic.optimize import optimize, OptResult


def _make_profile(n):
    return AllianceProfile(
        tiers=[PlayerTier("minnow", 8.0, 100)],
        counts=[n],
    )


def test_optimize_returns_results_with_mc_stats():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
    )
    best = results[0]
    assert isinstance(best, OptResult)
    assert hasattr(best, "win_rate")
    assert hasattr(best, "mean_score_a")
    assert hasattr(best, "std_score_a")


def test_optimize_ranking_aggressive():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="aggressive",
    )
    # Should be sorted by mean_score_a descending
    for i in range(len(results) - 1):
        assert results[i].mean_score_a >= results[i + 1].mean_score_a


def test_optimize_ranking_conservative():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="conservative",
    )
    # Should be sorted by p25 descending
    for i in range(len(results) - 1):
        assert results[i].p25_score_a >= results[i + 1].p25_score_a


def test_optimize_ranking_win_focused():
    cfg = default_config()
    results = optimize(
        cfg,
        profile_a=_make_profile(30),
        profile_b=_make_profile(20),
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=50,
        n_trials=10,
        ranking="win_focused",
    )
    # Should be sorted by win_rate descending
    for i in range(len(results) - 1):
        assert results[i].win_rate >= results[i + 1].win_rate
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimize.py -v`
Expected: FAIL

**Step 3: Rewrite optimize.py**

Update `src/got_wic/optimize.py`:

Updated `OptResult`:

```python
@dataclass
class OptResult:
    allocation: Allocation
    mean_score_a: float
    std_score_a: float
    mean_score_b: float
    win_rate: float
    p25_score_a: float
    breakdown_a: dict[str, float]
    breakdown_b: dict[str, float]
    hopeless: bool
```

Updated `optimize()` signature:

```python
def optimize(
    cfg: GameConfig,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    opponent_spread: float,
    opponent_aggression: float,
    step_pct: int = 10,
    n_trials: int = 100,
    noise_scale: float = 0.1,
    ranking: str = "aggressive",  # "aggressive" | "conservative" | "win_focused"
) -> list[OptResult]:
```

Ranking logic:
- `"aggressive"`: sort by `mean_score_a` descending
- `"conservative"`: sort by `p25_score_a` descending
- `"win_focused"`: sort by `win_rate` descending

Uses `total_armies = profile_a.total_players * 3` for allocation generation and `profile_b.total_players * 3` for opponent.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimize.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/got_wic/optimize.py tests/test_optimize.py
git commit -m "feat: update optimizer with Monte Carlo ranking modes"
```

---

### Task 6: Update integration tests

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Write updated integration tests**

Replace `tests/test_integration.py`:

```python
from pathlib import Path

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
```

**Step 2: Run tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: update integration tests for stochastic simulator"
```

---

### Task 7: Update opponent.py to use AllianceProfile

**Files:**
- Modify: `src/got_wic/opponent.py`
- Modify: `tests/test_opponent.py`

**Step 1: Update opponent to accept AllianceProfile**

The `generate_opponent` function currently takes `total_armies: int`. Update it to also accept an `AllianceProfile` so total armies can be derived from `profile.total_players * 3`. Keep backward compatibility by making `profile` optional.

```python
def generate_opponent(
    cfg: GameConfig,
    total_armies: int,
    spread: float,
    aggression: float,
) -> Allocation:
```

No signature change needed — the caller (`optimize.py`) already computes `total_armies` from the profile. The existing tests should still pass.

**Step 2: Run existing tests**

Run: `pytest tests/test_opponent.py -v`
Expected: All PASS (no changes needed)

**Step 3: Commit (only if changes were made)**

---

### Task 8: Update notebook UI

**Files:**
- Modify: `notebooks/sow-7th-ann-se.py`

**Step 1: Update the marimo notebook**

Add new UI sections:
1. Alliance profile builder (4-tier sliders for count, power, healing per side)
2. Monte Carlo controls (n_trials, noise_scale)
3. Score histogram (matplotlib/plotly distribution plot)
4. Win probability display
5. Percentile table
6. Hopeless warning banner
7. Optimizer results with ranking mode toggle
8. Timeline visualization (per-building power curves)

Import from the library instead of embedding duplicated code (WASM no longer required).

**Step 2: Run the notebook manually**

Run: `marimo run notebooks/sow-7th-ann-se.py`
Verify: All sections render, sliders work, optimizer produces results.

**Step 3: Commit**

```bash
git add notebooks/sow-7th-ann-se.py
git commit -m "feat: update notebook UI for stochastic simulator"
```

---

### Task 9: Run full test suite and final verification

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 3: Run type checker**

Run: `ty check src/`
Expected: No errors (or known pre-existing ones only)

**Step 4: Final commit if any cleanup needed**

```bash
git commit -m "chore: final cleanup for stochastic combat simulator"
```
