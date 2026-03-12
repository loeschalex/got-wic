from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from got_wic.combat import resolve_combat_minute_batch
from got_wic.model import Allocation, AllianceProfile, GameConfig
from got_wic.simulate import _armies_at, _make_building_fight, _phase_for_minute, simulate


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
    alloc_a: Allocation | None
    alloc_b: Allocation | None
    profile_a: AllianceProfile | None
    profile_b: AllianceProfile | None
    config: GameConfig | None
    timestamp: str


def _simulate_batch(
    cfg: GameConfig,
    alloc_a: Allocation,
    alloc_b: Allocation,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    n_trials: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run N trials simultaneously using vectorized combat.

    Returns (scores_a, scores_b) arrays of shape (n_trials,).

    Uses simplified power-decay model (no per-tier attrition tracking)
    for speed. Tier composition affects initial power distribution.
    """
    n_obj = len(cfg.objectives)
    n_combat_steps = 5

    # Pre-compute initial power at each building for each phase
    phases = ["phase1", "phase2", "phase3"]
    phase_power_a: dict[str, list[float]] = {}
    phase_power_b: dict[str, list[float]] = {}
    for phase in phases:
        pa_list = []
        pb_list = []
        for obj in cfg.objectives:
            fa = _make_building_fight(profile_a, _armies_at(alloc_a, phase, obj.name), alloc_a.total_armies)
            fb = _make_building_fight(profile_b, _armies_at(alloc_b, phase, obj.name), alloc_b.total_armies)
            pa_list.append(fa.total_power)
            pb_list.append(fb.total_power)
        phase_power_a[phase] = pa_list
        phase_power_b[phase] = pb_list

    # Pre-compute dragon/treasure armies per phase
    dragon_a_phase = {}
    dragon_b_phase = {}
    treasure_a_phase = {}
    treasure_b_phase = {}
    for phase in phases:
        dragon_a_phase[phase] = _armies_at(alloc_a, phase, "dragon")
        dragon_b_phase[phase] = _armies_at(alloc_b, phase, "dragon")
        treasure_a_phase[phase] = _armies_at(alloc_a, phase, "treasure")
        treasure_b_phase[phase] = _armies_at(alloc_b, phase, "treasure")

    # Per-trial state: power at each building (n_trials, n_obj)
    power_a = np.zeros((n_trials, n_obj))
    power_b = np.zeros((n_trials, n_obj))

    # Scoring accumulators (n_trials,)
    scores_a = np.zeros(n_trials)
    scores_b = np.zeros(n_trials)

    # First-capture tracking: (n_trials, n_obj, 2) — captured_a, captured_b
    fc_done_a = np.zeros((n_trials, n_obj), dtype=bool)
    fc_done_b = np.zeros((n_trials, n_obj), dtype=bool)

    # Dragon minutes
    dragon_mins_a = np.zeros(n_trials)
    dragon_mins_b = np.zeros(n_trials)

    current_phase = ""
    avg_treasure_pts = (cfg.treasure.normal_pts + cfg.treasure.rare_pts) // 2

    for t in range(cfg.match_duration):
        phase = _phase_for_minute(t, cfg.phase_boundaries)

        # Reset power on phase change
        if phase != current_phase:
            current_phase = phase
            for j in range(n_obj):
                power_a[:, j] = phase_power_a[phase][j]
                power_b[:, j] = phase_power_b[phase][j]

        # Combat at each building
        for j, obj in enumerate(cfg.objectives):
            if t < obj.opens_at:
                continue

            pa_col = power_a[:, j]
            pb_col = power_b[:, j]

            # Only run combat where both sides have power
            contested = (pa_col > 0) & (pb_col > 0)
            if contested.any():
                ca = pa_col.copy()
                cb = pb_col.copy()
                ca[~contested] = 0.0
                cb[~contested] = 0.0
                new_a, new_b = resolve_combat_minute_batch(
                    ca, cb, noise_scale, n_combat_steps, rng
                )
                power_a[:, j] = np.where(contested, new_a, pa_col)
                power_b[:, j] = np.where(contested, new_b, pb_col)

            # Determine holder per trial
            pa_j = power_a[:, j]
            pb_j = power_b[:, j]
            a_holds = pa_j > pb_j
            b_holds = pb_j > pa_j

            # First capture
            new_fc_a = a_holds & ~fc_done_a[:, j]
            new_fc_b = b_holds & ~fc_done_b[:, j]
            fc_bonus = obj.first_capture * obj.count
            scores_a += new_fc_a * fc_bonus
            scores_b += new_fc_b * fc_bonus
            fc_done_a[:, j] |= new_fc_a
            fc_done_b[:, j] |= new_fc_b

            # Hold points
            hold_pts = obj.hold_pts_min * obj.count
            scores_a += a_holds * hold_pts
            scores_b += b_holds * hold_pts

        # Dragons
        if cfg.dragons and t >= cfg.dragons[0].spawns_at:
            da = dragon_a_phase[phase]
            db = dragon_b_phase[phase]
            if da > db:
                dragon_mins_a += 1
            elif db > da:
                dragon_mins_b += 1

        # Treasure
        if t >= cfg.treasure.starts_at:
            ta = treasure_a_phase[phase]
            tb = treasure_b_phase[phase]
            scores_a += ta * avg_treasure_pts // 10
            scores_b += tb * avg_treasure_pts // 10

    # Dragon scoring
    total_dragon_pts = sum(d.escort_pts for d in cfg.dragons)
    if cfg.dragons:
        scores_a += np.where(dragon_mins_a > dragon_mins_b, total_dragon_pts, 0)
        scores_b += np.where(dragon_mins_b > dragon_mins_a, total_dragon_pts, 0)

    return scores_a, scores_b


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

    if noise_scale > 0 and n_trials > 1:
        # Use vectorized batch simulation
        scores_a, scores_b = _simulate_batch(
            cfg, alloc_a, alloc_b, profile_a, profile_b,
            n_trials, noise_scale, rng,
        )
    else:
        # Deterministic or single trial: use full simulation
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
        alloc_a=None,
        alloc_b=None,
        profile_a=None,
        profile_b=None,
        config=None,
        timestamp=metadata["timestamp"],
    )
