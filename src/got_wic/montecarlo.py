from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from got_wic.model import Allocation, AllianceProfile, GameConfig
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
    alloc_a: Allocation | None
    alloc_b: Allocation | None
    profile_a: AllianceProfile | None
    profile_b: AllianceProfile | None
    config: GameConfig | None
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
        alloc_a=None,
        alloc_b=None,
        profile_a=None,
        profile_b=None,
        config=None,
        timestamp=metadata["timestamp"],
    )
