from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from got_wic.model import GameConfig, Allocation
from got_wic.opponent import generate_opponent
from got_wic.simulate import simulate


@dataclass
class OptResult:
    allocation: Allocation
    score_a: int
    score_b: int
    breakdown_a: dict[str, int]
    breakdown_b: dict[str, int]


def _generate_allocations(
    cfg: GameConfig,
    total_armies: int,
    step_pct: int,
) -> list[Allocation]:
    """Generate candidate allocations by distributing armies across objectives in % steps."""
    # Allocate % to groups: own_outposts, center_armory, center_hotspring, strongholds, dragon, enemy_outposts
    groups = {
        "own_outposts": "Stark Outpost",
        "center_armory": "Armory",
        "center_hotspring": "Hot Spring",
        "strongholds": "Stronghold",
        "enemy_outposts": "Greyjoy Outpost",
        "dragon": "dragon",
    }

    # Phase 1 groups: own_outposts, center_armory, center_hotspring, enemy_outposts
    p1_keys = ["own_outposts", "center_armory", "center_hotspring", "enemy_outposts"]
    # Phase 3 groups: all
    p3_keys = list(groups.keys())

    steps = list(range(0, 101, step_pct))
    allocations = []

    for p1_combo in _feasible_combos(steps, len(p1_keys)):
        p1_alloc = {}
        for key, pct in zip(p1_keys, p1_combo):
            armies = int(total_armies * pct / 100)
            if armies > 0:
                p1_alloc[groups[key]] = armies

        for p3_combo in _feasible_combos(steps, len(p3_keys)):
            p3_alloc = {}
            for key, pct in zip(p3_keys, p3_combo):
                armies = int(total_armies * pct / 100)
                if armies > 0:
                    p3_alloc[groups[key]] = armies

            alloc = Allocation(
                assignments={
                    "phase1": p1_alloc,
                    "phase2": p1_alloc,  # same as phase1 for simplicity
                    "phase3": p3_alloc,
                },
                total_armies=total_armies,
            )
            if alloc.is_valid("phase1") and alloc.is_valid("phase3"):
                allocations.append(alloc)

    return allocations


def _feasible_combos(steps: list[int], n: int) -> list[tuple[int, ...]]:
    """Generate all n-tuples from steps that sum to <= 100."""
    if n == 0:
        return [()]
    if n == 1:
        return [(s,) for s in steps]
    results = []
    for combo in product(steps, repeat=n):
        if sum(combo) <= 100:
            results.append(combo)
    return results


def optimize(
    cfg: GameConfig,
    n_players_a: int,
    n_players_b: int,
    opponent_spread: float,
    opponent_aggression: float,
    step_pct: int = 10,
) -> list[OptResult]:
    total_a = n_players_a * 3
    total_b = n_players_b * 3

    opponent_alloc = generate_opponent(cfg, total_b, opponent_spread, opponent_aggression)
    candidates = _generate_allocations(cfg, total_a, step_pct)

    results = []
    for alloc in candidates:
        sim = simulate(cfg, alloc, opponent_alloc)
        results.append(OptResult(
            allocation=alloc,
            score_a=sim.score_a,
            score_b=sim.score_b,
            breakdown_a=sim.breakdown_a,
            breakdown_b=sim.breakdown_b,
        ))

    results.sort(key=lambda r: -r.score_a)
    return results
