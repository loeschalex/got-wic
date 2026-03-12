from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from got_wic.model import AllianceProfile, Allocation, GameConfig
from got_wic.montecarlo import run_monte_carlo
from got_wic.opponent import generate_opponent


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


def _generate_allocations(
    cfg: GameConfig,
    total_armies: int,
    step_pct: int,
) -> list[Allocation]:
    """Generate candidate allocations by distributing armies across objectives in % steps."""
    groups = {
        "own_outposts": "Stark Outpost",
        "center_armory": "Armory",
        "center_hotspring": "Hot Spring",
        "strongholds": "Stronghold",
        "enemy_outposts": "Greyjoy Outpost",
        "dragon": "dragon",
    }

    p1_keys = ["own_outposts", "center_armory", "center_hotspring", "enemy_outposts"]
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
                    "phase2": p1_alloc,
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
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    opponent_spread: float,
    opponent_aggression: float,
    step_pct: int = 10,
    n_trials: int = 100,
    noise_scale: float = 0.1,
    ranking: str = "aggressive",
) -> list[OptResult]:
    total_a = profile_a.total_players * 3
    total_b = profile_b.total_players * 3

    opponent_alloc = generate_opponent(cfg, total_b, opponent_spread, opponent_aggression)
    candidates = _generate_allocations(cfg, total_a, step_pct)

    results = []
    for alloc in candidates:
        mc = run_monte_carlo(
            cfg, alloc, opponent_alloc, profile_a, profile_b,
            n_trials=n_trials, noise_scale=noise_scale,
        )
        # Compute average breakdowns from a single deterministic run for display
        from got_wic.simulate import simulate
        det = simulate(cfg, alloc, opponent_alloc, profile_a, profile_b, noise_scale=0.0)

        results.append(OptResult(
            allocation=alloc,
            mean_score_a=mc.mean_score_a,
            std_score_a=mc.std_score_a,
            mean_score_b=mc.mean_score_b,
            win_rate=mc.win_rate,
            p25_score_a=mc.percentiles.get(25, 0.0),
            breakdown_a={k: float(v) for k, v in det.breakdown_a.items()},
            breakdown_b={k: float(v) for k, v in det.breakdown_b.items()},
            hopeless=mc.hopeless,
        ))

    # Sort by ranking mode
    if ranking == "conservative":
        results.sort(key=lambda r: -r.p25_score_a)
    elif ranking == "win_focused":
        results.sort(key=lambda r: -r.win_rate)
    else:  # aggressive (default)
        results.sort(key=lambda r: -r.mean_score_a)

    return results
