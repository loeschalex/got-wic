# Stochastic Combat Simulator — Design Document

**Date:** 2026-03-12
**Status:** Approved
**Builds on:** [SoW Optimizer v1](2026-03-12-sow-optimizer-design.md)

## Goal

Replace the deterministic "majority wins" combat model with a stochastic Lanchester-based simulation. The tool should produce win probabilities, score distributions, and risk-adjusted recommendations — and honestly tell the user when a scenario is unwinnable.

## Architecture: Two-Layer Approach

Four modules, cleanly separated:

```
combat.py          → single-tick fight resolution (Lanchester)
simulate.py        → updated 60-min match using combat.py
montecarlo.py      → N-trial orchestration + aggregation
optimize.py        → updated to rank by Monte Carlo stats
```

Notebook remains the primary UI. Deployment stays local (`marimo run`), with a future path to a web API (FastAPI layer on top).

---

## 1. Player Composition Model

### PlayerTier

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | "whale", "dolphin", "minnow", "alt" |
| `combat_power` | float | Relative power per player |
| `healing_capacity` | int | Full army rebuilds available. -1 = unlimited |

### AllianceProfile

| Field | Type | Description |
|-------|------|-------------|
| `tiers` | list[PlayerTier] | The 4 tiers |
| `counts` | list[int] | Players per tier |

Derived: `total_players = sum(counts)`, `total_power = sum(tier.power * count)`.

### Default 4-tier preset

| Tier | Power | Healing | Typical count |
|------|------:|--------:|:--------------|
| Whale | 100 | unlimited (-1) | 1–5 |
| Dolphin | 30 | 8 | 10–20 |
| Minnow | 8 | 4 | 20–30 |
| Alt | 1 | 2 | remaining |

Power values are relative and user-tunable. The global player population follows a Pareto distribution, but any given alliance is a small sample — so the user defines their own composition directly.

---

## 2. Combat Resolution — Lanchester Model

`combat.py` resolves per-building fights using discrete Lanchester's Square Law.

### Per-tick attrition

```
loss_a = β · power_b · dt
loss_b = α · power_a · dt
```

- **α, β**: attrition coefficients (tunable, default 1.0 each)
- Losses applied bottom-up through tiers: alts → minnows → dolphins → whales
- Small random chance per tick that a higher-tier player gets targeted (models being focused/unlucky)
- Players heal to restore power until healing budget exhausted, then permanently removed
- When one side hits 0 power at a building, the other holds uncontested

### Stochastic layer

Per-tick noise applied to attrition coefficients — small random perturbation models battlefield chaos. Close fights become genuinely uncertain; dominant fights stay predictable.

### Properties

- Naturally produces tipping-point behavior: below a critical power ratio, defeat is inevitable
- Attrition snowballs: losses → less power → faster losses (square law)
- Healing depletion means early losses compound into late-game weakness

---

## 3. Updated Simulation Engine

`simulate.py` updated to use Lanchester combat.

### Inputs

- `GameConfig` (unchanged)
- `Allocation` for both sides (unchanged)
- `AllianceProfile` for both sides (new)

### Per-tick loop (minutes 0–59)

```
For each open building:
  1. Apply Lanchester attrition (both sides lose power)
  2. Heal bottom-up (if budget remains)
  3. Remove depleted players
  4. Determine holder (side with remaining power)
  5. Award hold points + first capture if applicable

For each active dragon:
  Same resolution as buildings

If treasure phase (≥ minute 8):
  Award proportional to power sent
```

### Updated SimResult

- Per-tick power levels at each building (for timeline visualization)
- Total healing spent per side
- Per-tier casualty counts
- Score breakdowns (first capture / hold / dragon / treasure) as before

---

## 4. Monte Carlo Orchestration

New `montecarlo.py` module.

### API

```python
def run_monte_carlo(
    cfg: GameConfig,
    alloc_a: Allocation,
    alloc_b: Allocation,
    profile_a: AllianceProfile,
    profile_b: AllianceProfile,
    n_trials: int = 500,
    noise_scale: float = 0.1,
) -> MonteCarloResult
```

### MonteCarloResult

| Field | Type | Description |
|-------|------|-------------|
| `n_trials` | int | Number of trials run |
| `mean_score_a` | float | Average score for side A |
| `std_score_a` | float | Standard deviation |
| `mean_score_b` | float | Average score for side B |
| `win_rate` | float | Fraction of trials where A > B |
| `score_distribution_a` | array | All trial scores (for histograms) |
| `score_distribution_b` | array | All trial scores |
| `percentiles` | dict | {5, 25, 50, 75, 95} → score |
| `hopeless` | bool | True if win_rate < 5% |
| `alloc_a, alloc_b` | Allocation | For reproducibility |
| `profile_a, profile_b` | AllianceProfile | For reproducibility |
| `config` | GameConfig | For reproducibility |
| `timestamp` | str | When the run was executed |

### Result persistence

Results serialized to **NumPy `.npz`** (compressed). Score arrays stored natively; metadata (config, profiles, allocations) embedded as JSON string. Downstream analysis reads from saved files — no re-run required.

---

## 5. Updated Optimizer

`optimize.py` uses Monte Carlo instead of deterministic simulation.

### Ranking modes (user-selectable)

1. **Aggressive** — rank by mean score
2. **Conservative** — rank by 25th percentile (worst-likely-case)
3. **Win-focused** — rank by win rate

### Performance mitigations

- Coarse grid first (step_pct=25%), refine around top candidates
- Configurable `n_trials` (fewer for exploration, more for final answer)
- Parallelize across candidates with `multiprocessing`

### Output

Top N allocations with: mean, std, win rate, percentiles, hopeless flag. All backed by saved `.npz` files.

---

## 6. Notebook UI Updates

Runs locally via `marimo run` (no WASM).

### New UI elements

- **Alliance profile builder**: 4-tier table with sliders for count, power, healing per tier. One per side. Quick presets ("whale-heavy", "balanced", "casual").
- **Monte Carlo controls**: n_trials, noise_scale sliders
- **Score histogram**: distribution from all trials
- **Win probability**: prominent display
- **Percentile table**: 5th / 25th / 50th / 75th / 95th
- **Hopeless warning**: banner when win_rate < 5%
- **Optimizer results**: mean, std, win rate columns + ranking mode toggle
- **Timeline visualization**: per-building power curves over 60 minutes showing attrition and tipping points
- **Scenario comparison**: load multiple saved `.npz` results, compare side by side

---

## Scope exclusions (future work)

- Nash equilibrium / game-theoretic opponent modeling
- Troop type rock-paper-scissors mechanics
- March time modeling
- Dynamic mid-match reallocation
- Web API deployment (approach C)
