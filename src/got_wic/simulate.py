from __future__ import annotations

from dataclasses import dataclass, field

from got_wic.model import GameConfig, Allocation


@dataclass
class SimResult:
    score_a: int
    score_b: int
    breakdown_a: dict[str, int] = field(default_factory=dict)
    breakdown_b: dict[str, int] = field(default_factory=dict)
    timeline: list[dict] = field(default_factory=list)


def _phase_for_minute(minute: int, boundaries: list[int]) -> str:
    """Return the phase name for a given minute."""
    # boundaries = [0, 8, 12] → phase1=0-7, phase2=8-11, phase3=12+
    for i in range(len(boundaries) - 1, -1, -1):
        if minute >= boundaries[i]:
            return f"phase{i + 1}"
    return "phase1"


def _armies_at(alloc: Allocation, phase: str, objective: str) -> int:
    return alloc.assignments.get(phase, {}).get(objective, 0)


def simulate(cfg: GameConfig, alloc_a: Allocation, alloc_b: Allocation) -> SimResult:
    fc_a = fc_b = 0
    hold_a = hold_b = 0
    dragon_a = dragon_b = 0
    treasure_a = treasure_b = 0

    # Track which side captured each objective first (for first-capture bonus)
    captured_by: dict[str, str | None] = {}  # obj_name -> "a" | "b" | None

    # Dragon escort: simplified — if you have majority at "dragon" for enough
    # cumulative minutes, you get the escort points. Model as: side with majority
    # over the dragon period gets the points, proportional to time held.
    dragon_minutes_a = 0
    dragon_minutes_b = 0
    dragon_available_minutes = max(0, cfg.match_duration - cfg.dragons[0].spawns_at) if cfg.dragons else 0

    for t in range(cfg.match_duration):
        phase = _phase_for_minute(t, cfg.phase_boundaries)

        # --- Building objectives ---
        for obj in cfg.objectives:
            if t < obj.opens_at:
                continue

            armies_a = _armies_at(alloc_a, phase, obj.name)
            armies_b = _armies_at(alloc_b, phase, obj.name)

            holder = None
            if armies_a > armies_b:
                holder = "a"
            elif armies_b > armies_a:
                holder = "b"
            # tie: no one captures (simplification for v1)

            if holder is not None:
                # First capture bonus (once per objective per side)
                cap_key = f"{obj.name}_{holder}"
                if cap_key not in captured_by:
                    captured_by[cap_key] = holder
                    bonus = obj.first_capture * obj.count
                    if holder == "a":
                        fc_a += bonus
                    else:
                        fc_b += bonus

                # Hold points
                pts = obj.hold_pts_min * obj.count
                if holder == "a":
                    hold_a += pts
                else:
                    hold_b += pts

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
            # Simplified: points proportional to armies sent, average of normal/rare
            avg_pts = (cfg.treasure.normal_pts + cfg.treasure.rare_pts) // 2
            treasure_a += ta * avg_pts // 10  # scaled down — 1 army ≈ 1/10 treasure/min
            treasure_b += tb * avg_pts // 10

    # --- Dragon scoring ---
    # Side with majority of dragon-minutes gets the escort points
    total_dragon_pts = sum(d.escort_pts for d in cfg.dragons)
    if dragon_available_minutes > 0:
        if dragon_minutes_a > dragon_minutes_b:
            dragon_a = total_dragon_pts
        elif dragon_minutes_b > dragon_minutes_a:
            dragon_b = total_dragon_pts
        # tie: no one gets dragon points

    score_a = fc_a + hold_a + dragon_a + treasure_a
    score_b = fc_b + hold_b + dragon_b + treasure_b

    return SimResult(
        score_a=score_a,
        score_b=score_b,
        breakdown_a={"first_capture": fc_a, "hold": hold_a, "dragon": dragon_a, "treasure": treasure_a},
        breakdown_b={"first_capture": fc_b, "hold": hold_b, "dragon": dragon_b, "treasure": treasure_b},
    )
