from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from got_wic.model import PlayerTier


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

        # Healing: unlimited (-1) -> always come back, otherwise spend from budget
        if tier.healing_capacity == -1:
            # Whales always heal -- counts stay the same
            pass
        elif new_healing[i] >= kills:
            new_healing[i] -= kills
            # Players heal back -- counts stay the same
        else:
            # Partial healing
            healed = new_healing[i]
            new_healing[i] = 0
            new_counts[i] -= kills - healed

    return BuildingFight(
        tiers=fight.tiers,
        counts=new_counts,
        healing_remaining=new_healing,
    )
