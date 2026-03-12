from __future__ import annotations

import numpy as np

from got_wic.model import GameConfig, Allocation, Objective


# Objectives the opponent (Greyjoy/Side B) considers "own side"
_OWN_ZONES = {"near_greyjoy"}
_ENEMY_ZONES = {"near_stark"}
_NEUTRAL_ZONES = {"center", "mid"}


def _available_objectives(cfg: GameConfig, phase: str) -> list[Objective]:
    """Return objectives available in a given phase."""
    phase_idx = int(phase.replace("phase", "")) - 1
    boundaries = cfg.phase_boundaries
    phase_start = boundaries[phase_idx] if phase_idx < len(boundaries) else 0
    return [o for o in cfg.objectives if o.opens_at <= phase_start]


def _weight_objective(obj: Objective, aggression: float) -> float:
    """Weight an objective by its value and zone affinity."""
    base = obj.hold_pts_min * obj.count
    if obj.zone in _OWN_ZONES:
        zone_mult = 1.0 + (1.0 - aggression)  # defensive bonus
    elif obj.zone in _ENEMY_ZONES:
        if aggression < 1e-9:
            return 0.0  # fully defensive: never go to enemy side
        zone_mult = aggression
    else:
        zone_mult = 1.0
    return base * zone_mult


def generate_opponent(
    cfg: GameConfig,
    total_armies: int,
    spread: float,
    aggression: float,
) -> Allocation:
    """Generate an opponent allocation from spread and aggression parameters.

    spread: 0 = all-in on best objective, 1 = even across all.
    aggression: 0 = defensive (own side), 1 = push enemy territory.
    """
    assignments: dict[str, dict[str, int]] = {}

    for phase in ["phase1", "phase2", "phase3"]:
        available = _available_objectives(cfg, phase)
        if not available:
            assignments[phase] = {}
            continue

        # Compute weights
        weights = np.array([_weight_objective(o, aggression) for o in available], dtype=float)

        # Filter out zero-weight objectives
        mask = weights > 0
        available_filtered = [o for o, m in zip(available, mask) if m]
        weights = weights[mask]

        if len(available_filtered) == 0:
            assignments[phase] = {}
            continue

        if spread < 1e-9:
            # All-in on highest weight
            idx = int(np.argmax(weights))
            assignments[phase] = {available_filtered[idx].name: total_armies}
            continue

        # Interpolate between concentrated (softmax with low temp) and uniform
        # spread=0 → very peaked, spread=1 → uniform
        temperature = 0.1 + spread * 10.0  # range [0.1, 10.1]
        scaled = weights / (weights.max() + 1e-9) * (1.0 / temperature)
        exp_w = np.exp(scaled - scaled.max())
        probs = exp_w / exp_w.sum()

        # Distribute armies proportionally, rounding to ints
        raw = probs * total_armies
        counts = np.floor(raw).astype(int)
        remainder = total_armies - counts.sum()
        # Distribute remainder to highest fractional parts
        fracs = raw - counts
        for _ in range(int(remainder)):
            idx = int(np.argmax(fracs))
            counts[idx] += 1
            fracs[idx] = -1

        phase_alloc = {}
        for obj, count in zip(available_filtered, counts):
            if count > 0:
                phase_alloc[obj.name] = int(count)

        # Add dragon/treasure armies in phase3
        if phase == "phase3" and cfg.dragons:
            # Divert some armies to dragons based on aggression
            dragon_share = int(total_armies * 0.1 * (0.5 + aggression * 0.5))
            # Reduce from existing proportionally
            used = sum(phase_alloc.values())
            if used + dragon_share > total_armies:
                scale = (total_armies - dragon_share) / max(used, 1)
                phase_alloc = {k: max(1, int(v * scale)) for k, v in phase_alloc.items()}
            phase_alloc["dragon"] = dragon_share

        assignments[phase] = phase_alloc

    return Allocation(assignments=assignments, total_armies=total_armies)
