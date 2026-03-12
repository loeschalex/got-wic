from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from got_wic.combat import BuildingFight, apply_attrition, resolve_combat_minute
from got_wic.model import AllianceProfile, Allocation, GameConfig


@dataclass
class SimResult:
    score_a: int
    score_b: int
    breakdown_a: dict[str, int | float] = field(default_factory=dict)
    breakdown_b: dict[str, int | float] = field(default_factory=dict)
    timeline: list[dict] = field(default_factory=list)
    healing_spent_a: int = 0
    healing_spent_b: int = 0
    casualties_a: dict[str, int] = field(default_factory=dict)
    casualties_b: dict[str, int] = field(default_factory=dict)


def _phase_for_minute(minute: int, boundaries: list[int]) -> str:
    """Return the phase name for a given minute."""
    for i in range(len(boundaries) - 1, -1, -1):
        if minute >= boundaries[i]:
            return f"phase{i + 1}"
    return "phase1"


def _armies_at(alloc: Allocation, phase: str, objective: str) -> int:
    return alloc.assignments.get(phase, {}).get(objective, 0)


def _make_building_fight(
    profile: AllianceProfile,
    armies_here: int,
    total_armies: int,
) -> BuildingFight:
    """Create a BuildingFight by distributing profile tiers proportionally."""
    if total_armies == 0 or armies_here == 0:
        return BuildingFight(
            tiers=list(profile.tiers),
            counts=[0] * len(profile.tiers),
            healing_remaining=[0] * len(profile.tiers),
        )

    fraction = min(1.0, armies_here / total_armies)
    counts = []
    healing = []
    for tier, tier_count in zip(profile.tiers, profile.counts):
        n = max(0, round(tier_count * fraction))
        counts.append(n)
        if tier.healing_capacity == -1:
            healing.append(0)  # sentinel for unlimited
        else:
            healing.append(n * tier.healing_capacity)

    return BuildingFight(tiers=list(profile.tiers), counts=counts, healing_remaining=healing)


def simulate(
    cfg: GameConfig,
    alloc_a: Allocation,
    alloc_b: Allocation,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    noise_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SimResult:
    fc_a = fc_b = 0
    hold_a = hold_b = 0
    dragon_a = dragon_b = 0
    treasure_a = treasure_b = 0
    total_healing_a = 0
    total_healing_b = 0

    captured_by: dict[str, str | None] = {}

    dragon_minutes_a = 0
    dragon_minutes_b = 0
    dragon_available_minutes = (
        max(0, cfg.match_duration - cfg.dragons[0].spawns_at) if cfg.dragons else 0
    )

    # Per-building fight state: refreshed each phase change
    fights_a: dict[str, BuildingFight] = {}
    fights_b: dict[str, BuildingFight] = {}
    current_phase = ""

    timeline: list[dict] = []

    # Combat sub-steps per minute (5 = dt of 0.2)
    n_combat_steps = 5

    for t in range(cfg.match_duration):
        phase = _phase_for_minute(t, cfg.phase_boundaries)

        # Rebuild fights when phase changes
        if phase != current_phase:
            current_phase = phase
            fights_a.clear()
            fights_b.clear()
            for obj in cfg.objectives:
                armies_a = _armies_at(alloc_a, phase, obj.name)
                armies_b = _armies_at(alloc_b, phase, obj.name)
                fights_a[obj.name] = _make_building_fight(
                    profile_a, armies_a, alloc_a.total_armies
                )
                fights_b[obj.name] = _make_building_fight(
                    profile_b, armies_b, alloc_b.total_armies
                )

        tick_data: dict[str, dict] = {}

        # --- Building objectives ---
        for obj in cfg.objectives:
            if t < obj.opens_at:
                continue

            fa = fights_a[obj.name]
            fb = fights_b[obj.name]
            power_a = fa.total_power
            power_b = fb.total_power

            # Run combat resolution for this minute
            if power_a > 0 and power_b > 0:
                new_pa, new_pb = resolve_combat_minute(
                    power_a, power_b, noise_scale, n_combat_steps, rng
                )

                # Apply losses as attrition
                loss_a = power_a - new_pa
                loss_b = power_b - new_pb

                heal_before_a = sum(fa.healing_remaining)
                heal_before_b = sum(fb.healing_remaining)

                if loss_a > 0:
                    fights_a[obj.name] = apply_attrition(fa, loss_a)
                if loss_b > 0:
                    fights_b[obj.name] = apply_attrition(fb, loss_b)

                fa = fights_a[obj.name]
                fb = fights_b[obj.name]

                total_healing_a += heal_before_a - sum(fa.healing_remaining)
                total_healing_b += heal_before_b - sum(fb.healing_remaining)

                power_a = fa.total_power
                power_b = fb.total_power

            # Determine holder
            holder = None
            if power_a > power_b:
                holder = "a"
            elif power_b > power_a:
                holder = "b"

            if holder is not None:
                cap_key = f"{obj.name}_{holder}"
                if cap_key not in captured_by:
                    captured_by[cap_key] = holder
                    bonus = obj.first_capture * obj.count
                    if holder == "a":
                        fc_a += bonus
                    else:
                        fc_b += bonus

                pts = obj.hold_pts_min * obj.count
                if holder == "a":
                    hold_a += pts
                else:
                    hold_b += pts

            tick_data[obj.name] = {"power_a": power_a, "power_b": power_b}

        # --- Dragons ---
        if cfg.dragons and t >= cfg.dragons[0].spawns_at:
            da = _armies_at(alloc_a, phase, "dragon")
            db = _armies_at(alloc_b, phase, "dragon")
            if da > db:
                dragon_minutes_a += 1
            elif db > da:
                dragon_minutes_b += 1

        # --- Treasure ---
        if t >= cfg.treasure.starts_at:
            ta = _armies_at(alloc_a, phase, "treasure")
            tb = _armies_at(alloc_b, phase, "treasure")
            avg_pts = (cfg.treasure.normal_pts + cfg.treasure.rare_pts) // 2
            treasure_a += ta * avg_pts // 10
            treasure_b += tb * avg_pts // 10

        timeline.append(tick_data)

    # Dragon scoring
    total_dragon_pts = sum(d.escort_pts for d in cfg.dragons)
    if dragon_available_minutes > 0:
        if dragon_minutes_a > dragon_minutes_b:
            dragon_a = total_dragon_pts
        elif dragon_minutes_b > dragon_minutes_a:
            dragon_b = total_dragon_pts

    score_a = fc_a + hold_a + dragon_a + treasure_a
    score_b = fc_b + hold_b + dragon_b + treasure_b

    # Compute casualties
    casualties_a: dict[str, int] = {}
    casualties_b: dict[str, int] = {}
    for obj_name, fa in fights_a.items():
        for tier, orig_count, cur_count in zip(
            fa.tiers,
            _make_building_fight(profile_a, _armies_at(alloc_a, current_phase, obj_name), alloc_a.total_armies).counts,
            fa.counts,
        ):
            lost = orig_count - cur_count
            if lost > 0:
                casualties_a[tier.name] = casualties_a.get(tier.name, 0) + lost
    for obj_name, fb in fights_b.items():
        for tier, orig_count, cur_count in zip(
            fb.tiers,
            _make_building_fight(profile_b, _armies_at(alloc_b, current_phase, obj_name), alloc_b.total_armies).counts,
            fb.counts,
        ):
            lost = orig_count - cur_count
            if lost > 0:
                casualties_b[tier.name] = casualties_b.get(tier.name, 0) + lost

    return SimResult(
        score_a=score_a,
        score_b=score_b,
        breakdown_a={"first_capture": fc_a, "hold": hold_a, "dragon": dragon_a, "treasure": treasure_a},
        breakdown_b={"first_capture": fc_b, "hold": hold_b, "dragon": dragon_b, "treasure": treasure_b},
        timeline=timeline,
        healing_spent_a=total_healing_a,
        healing_spent_b=total_healing_b,
        casualties_a=casualties_a,
        casualties_b=casualties_b,
    )
