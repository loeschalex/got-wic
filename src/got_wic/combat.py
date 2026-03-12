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
