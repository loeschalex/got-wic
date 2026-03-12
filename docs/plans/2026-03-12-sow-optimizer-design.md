# Siege of Winterfell — Strategy Optimizer Design

## Goal

Build a **strategic insight engine** for the Siege of Winterfell (7th Anniversary SE)
event in Game of Thrones: Winter is Coming. Given player counts on each side and
configurable game mechanics, answer: "Which objectives should we prioritise, and how
does that change as the player ratio shifts?"

This is not a live battle planner — it generates general strategic patterns and
tipping-point analysis.

## Game Model

### Objectives (11 contestable locations)

| Objective        | Count | First Capture | Hold pts/min | Opens at | Zone          |
|------------------|------:|-------------:|-------------:|---------:|---------------|
| Stark Outpost    |     2 |          200 |           80 |     0:00 | near_stark    |
| Greyjoy Outpost  |     2 |          200 |           80 |     0:00 | near_greyjoy  |
| Armory           |     1 |          400 |          120 |     0:00 | center        |
| Hot Spring       |     1 |          400 |          120 |     0:00 | center        |
| Stronghold       |     4 |          600 |          180 |    12:00 | mid           |
| Winterfell       |     1 |            — |            — |    12:00 | center        |

Armory grants +50% ATK/DEF/HP; Hot Spring grants +100% Heal Speed.
Winterfell is not capturable — it spawns resource chests at 12:00.

### Dragons (spawn at 12:00)

| Dragon       | Escort Points | Arrival Buff       |
|--------------|-------------:|--------------------|
| Winter Ice   |        3,000 | +50% Heal Speed    |
| Flaming Sun  |        3,000 | +Movement Speed    |

- Majority-based tug-of-war escort toward your spawn.
- Speed scales with troop count advantage.
- Drops score packages during contested escort (after 10% progress).
- "Catch-up" mechanic: faster speed when regaining lost progress.

### Treasure Hunt (starts at 8:00)

- Spawns at random locations at irregular intervals.
- Normal treasure: 80 pts + 1 Blessing Point.
- Rare treasure: 120 pts + 2 Blessing Points.

### Phases

| Phase | Minutes | Opens                                        |
|-------|---------|----------------------------------------------|
|     1 |    0–7  | 4 Outposts + Armory + Hot Spring             |
|     2 |   8–11  | + Treasure Hunt                              |
|     3 |  12–60  | + 4 Strongholds + 2 Dragons + Chests         |

### Armies

- 2 sides, up to 100 players each, 3 armies per player (max 300 per side).
- Contest resolution: **binary majority** — more armies at an objective = you hold it.
  Ties go to current holder (configurable).

### Approximate march times (from spawn)

| Destination      | Time   |
|------------------|-------:|
| Own Outposts     | ~1 min |
| Armory/Hot Spring| ~2 min |
| Strongholds      | ~2.5–3 min |
| Enemy Outposts   | ~3–4 min |

March times are not modelled as movement in v1 — they inform which phase an
objective can realistically be contested from.

## Optimizer

### Decision variable

An **allocation** assigns armies to objectives per phase:

```python
allocation = {
    "phase1": {"Stark Outpost": 40, "Greyjoy Outpost": 20, "Armory": 15, ...},
    "phase2": { ... + "treasure": 10 },
    "phase3": { ... + "Stronghold": 60, "dragon": 30, "chest_escort": 15 },
}
```

Constraint: armies assigned per phase <= total armies (n_players * 3).

### Opponent model (parametric)

Two parameters generate the opponent's allocation:

- **spread** (0–1): 0 = all-in on one objective, 1 = even distribution.
- **aggression** (0–1): 0 = defensive (hold own side), 1 = push into enemy territory.

### Search

For each (n_a, n_b, opponent_spread, opponent_aggression):

1. Enumerate candidate allocations (discretised in steps of ~10% of total armies).
2. Simulate 60 minutes, accumulate score.
3. Rank by total points.

### Outputs

1. **Best allocation** for a given opponent profile.
2. **Player ratio sweep**: how optimal allocation shifts from 50v50 to 100v30.
3. **Sensitivity heatmap**: score impact when opponent switches strategy.
4. **Tipping points**: "Below 60 players, abandon enemy outposts."

## Architecture

```
src/got_wic/
    __init__.py
    model.py        # Objective, Dragon, GameConfig dataclasses
    simulate.py     # minute-by-minute scorer
    opponent.py     # parametric opponent strategy generator
    optimize.py     # grid search, ranking

notebooks/
    sow-7th-ann-se.py  # marimo UI (replaces current notebook)
```

### Key types

- `Objective` — name, count, first_capture, hold_pts_min, opens_at, zone
- `Dragon` — name, escort_pts, spawns_at
- `TreasureConfig` — starts_at, normal_pts, rare_pts
- `GameConfig` — objectives, dragons, treasure, match_duration, phase_boundaries
- `Allocation` — dict of phase -> {objective -> army_count}

### Notebook UI

1. Config panel — player counts + point values (pre-filled, editable).
2. Opponent profile — spread/aggression sliders.
3. Single simulation — score breakdown for a manual allocation.
4. Optimizer results — top allocations + recommendation.
5. Insight sweeps — player ratio, opponent sensitivity, tipping points.

## Scope exclusions (v1)

- No combat / troop strength simulation.
- No march-time transitions between phases.
- No progress bar / capture timer modelling.
- No Monte Carlo / stochastic runs.
- No Nash equilibrium (planned for v2).
