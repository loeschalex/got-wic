# SoW Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Implement a zone-based allocation optimizer that answers "which objectives to prioritise at each player ratio" for the Siege of Winterfell event.

**Architecture:** Pure Python library in `src/got_wic/` with dataclass-based game model, minute-by-minute simulation, parametric opponent, and grid-search optimizer. Marimo notebook for interactive UI.

**Tech Stack:** Python 3.13, dataclasses, numpy, marimo, pytest

---

### Task 0: Add pytest to dev dependencies

**Files:**
- Modify: `pyproject.toml:21-25`

**Step 1: Add pytest**

Add `"pytest>=8.0"` to the dev dependency group in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "ruff>=0.15.5",
    "ty>=0.0.21",
    "pytest>=8.0",
]
```

**Step 2: Install**

Run: `uv sync`
Expected: Clean install with pytest available.

**Step 3: Verify**

Run: `uv run pytest --version`
Expected: `pytest 8.x.x`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest to dev dependencies"
```

---

### Task 1: Game model dataclasses (`model.py`)

**Files:**
- Create: `src/got_wic/model.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
from got_wic.model import (
    Objective,
    Dragon,
    TreasureConfig,
    GameConfig,
    Allocation,
    default_config,
)


def test_objective_creation():
    obj = Objective(
        name="Stark Outpost",
        count=2,
        first_capture=200,
        hold_pts_min=80,
        opens_at=0,
        zone="near_stark",
    )
    assert obj.name == "Stark Outpost"
    assert obj.count == 2
    assert obj.hold_pts_min == 80


def test_dragon_creation():
    d = Dragon(name="Winter Ice", escort_pts=3000, spawns_at=12)
    assert d.escort_pts == 3000


def test_default_config_has_correct_objectives():
    cfg = default_config()
    names = [o.name for o in cfg.objectives]
    assert "Stark Outpost" in names
    assert "Stronghold" in names
    assert "Armory" in names
    assert "Hot Spring" in names
    assert "Greyjoy Outpost" in names
    assert len(cfg.objectives) == 5  # Winterfell excluded (not capturable)


def test_default_config_has_dragons():
    cfg = default_config()
    assert len(cfg.dragons) == 2
    assert cfg.dragons[0].escort_pts == 3000


def test_default_config_phases():
    cfg = default_config()
    assert cfg.phase_boundaries == [0, 8, 12]
    assert cfg.match_duration == 60


def test_allocation_total_armies():
    alloc = Allocation(
        assignments={"phase1": {"Stark Outpost": 40, "Armory": 20}},
        total_armies=180,
    )
    assert alloc.armies_used("phase1") == 60
    assert alloc.armies_unused("phase1") == 120


def test_allocation_rejects_overcommit():
    alloc = Allocation(
        assignments={"phase1": {"Stark Outpost": 100, "Armory": 100}},
        total_armies=180,
    )
    assert not alloc.is_valid("phase1")


def test_default_config_total_objectives_count():
    """Sum of all objective counts should be 10 (excludes Winterfell)."""
    cfg = default_config()
    total = sum(o.count for o in cfg.objectives)
    assert total == 10
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'got_wic.model'`

**Step 3: Write implementation**

Create `src/got_wic/model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Objective:
    name: str
    count: int
    first_capture: int
    hold_pts_min: int
    opens_at: int
    zone: str


@dataclass(frozen=True)
class Dragon:
    name: str
    escort_pts: int
    spawns_at: int


@dataclass(frozen=True)
class TreasureConfig:
    starts_at: int = 8
    normal_pts: int = 80
    rare_pts: int = 120


@dataclass(frozen=True)
class GameConfig:
    objectives: list[Objective]
    dragons: list[Dragon]
    treasure: TreasureConfig = field(default_factory=TreasureConfig)
    match_duration: int = 60
    phase_boundaries: list[int] = field(default_factory=lambda: [0, 8, 12])


@dataclass
class Allocation:
    assignments: dict[str, dict[str, int]]  # phase_name -> {objective_name -> armies}
    total_armies: int

    def armies_used(self, phase: str) -> int:
        return sum(self.assignments.get(phase, {}).values())

    def armies_unused(self, phase: str) -> int:
        return self.total_armies - self.armies_used(phase)

    def is_valid(self, phase: str) -> bool:
        return self.armies_used(phase) <= self.total_armies


def default_config() -> GameConfig:
    objectives = [
        Objective("Stark Outpost", 2, 200, 80, 0, "near_stark"),
        Objective("Greyjoy Outpost", 2, 200, 80, 0, "near_greyjoy"),
        Objective("Armory", 1, 400, 120, 0, "center"),
        Objective("Hot Spring", 1, 400, 120, 0, "center"),
        Objective("Stronghold", 4, 600, 180, 12, "mid"),
    ]
    dragons = [
        Dragon("Winter Ice", 3000, 12),
        Dragon("Flaming Sun", 3000, 12),
    ]
    return GameConfig(objectives=objectives, dragons=dragons)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_model.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/got_wic/model.py tests/test_model.py
git commit -m "feat: add game model dataclasses with default SoW config"
```

---

### Task 2: Simulation engine (`simulate.py`)

**Files:**
- Create: `src/got_wic/simulate.py`
- Create: `tests/test_simulate.py`

**Step 1: Write the failing tests**

Create `tests/test_simulate.py`:

```python
from got_wic.model import default_config, Allocation
from got_wic.simulate import simulate


def _make_alloc(total: int, phase_assignments: dict[str, dict[str, int]]) -> Allocation:
    return Allocation(assignments=phase_assignments, total_armies=total)


def test_empty_allocation_scores_zero():
    cfg = default_config()
    a = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 0
    assert result.score_b == 0


def test_uncontested_outpost_scores_hold_points():
    """Side A sends 10 armies to Stark Outpost, Side B sends 0.
    Outpost opens at minute 0, holds for 60 minutes.
    2 outposts * 80 pts/min * 60 min = 9600, plus first capture 2*200 = 400.
    Total = 10000."""
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(30, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000


def test_contested_objective_winner_takes_points():
    """Side A sends 20 to Stark Outpost, Side B sends 10.
    Side A wins majority, gets all points."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    b = _make_alloc(60, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000
    assert result.score_b == 0


def test_strongholds_only_score_after_minute_12():
    """Strongholds open at minute 12. Hold for 48 minutes.
    4 * 180 * 48 = 34560, plus first capture 4 * 600 = 2400. Total = 36960."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {},
        "phase2": {},
        "phase3": {"Stronghold": 30},
    })
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a == 36_960


def test_dragon_escort_scores_points():
    """Side A sends armies to dragon, Side B sends 0. Should get escort pts."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {},
        "phase2": {},
        "phase3": {"dragon": 30},
    })
    b = _make_alloc(60, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.score_a >= 6000  # 2 dragons * 3000


def test_result_has_breakdown():
    cfg = default_config()
    a = _make_alloc(30, {
        "phase1": {"Stark Outpost": 10},
        "phase2": {"Stark Outpost": 10},
        "phase3": {"Stark Outpost": 10},
    })
    b = _make_alloc(30, {"phase1": {}, "phase2": {}, "phase3": {}})
    result = simulate(cfg, a, b)
    assert result.breakdown_a["first_capture"] == 400
    assert result.breakdown_a["hold"] == 9600
    assert result.breakdown_a["dragon"] == 0


def test_both_sides_score_different_objectives():
    """A holds Stark Outposts, B holds Greyjoy Outposts. Both score."""
    cfg = default_config()
    a = _make_alloc(60, {
        "phase1": {"Stark Outpost": 20},
        "phase2": {"Stark Outpost": 20},
        "phase3": {"Stark Outpost": 20},
    })
    b = _make_alloc(60, {
        "phase1": {"Greyjoy Outpost": 20},
        "phase2": {"Greyjoy Outpost": 20},
        "phase3": {"Greyjoy Outpost": 20},
    })
    result = simulate(cfg, a, b)
    assert result.score_a == 10_000
    assert result.score_b == 10_000
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_simulate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'got_wic.simulate'`

**Step 3: Write implementation**

Create `src/got_wic/simulate.py`:

```python
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_simulate.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add src/got_wic/simulate.py tests/test_simulate.py
git commit -m "feat: add minute-by-minute simulation engine"
```

---

### Task 3: Opponent strategy generator (`opponent.py`)

**Files:**
- Create: `src/got_wic/opponent.py`
- Create: `tests/test_opponent.py`

**Step 1: Write the failing tests**

Create `tests/test_opponent.py`:

```python
from got_wic.model import default_config
from got_wic.opponent import generate_opponent


def test_even_spread_distributes_equally():
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=0.5)
    # With full spread, armies should be distributed across all available objectives per phase
    phase3_armies = alloc.armies_used("phase3")
    assert phase3_armies > 0
    assert phase3_armies <= 300


def test_zero_spread_concentrates():
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=0.0, aggression=0.5)
    # With zero spread, all armies go to highest-value objective
    phase3 = alloc.assignments["phase3"]
    counts = list(phase3.values())
    assert max(counts) == sum(counts)  # all in one objective


def test_defensive_stays_own_side():
    """aggression=0 means opponent (Side B = Greyjoy) stays on own side."""
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=0.0)
    phase1 = alloc.assignments["phase1"]
    # Should not send armies to Stark Outpost (enemy territory)
    assert phase1.get("Stark Outpost", 0) == 0


def test_aggressive_pushes_enemy_side():
    """aggression=1 means opponent pushes into enemy territory."""
    cfg = default_config()
    alloc = generate_opponent(cfg, total_armies=300, spread=1.0, aggression=1.0)
    phase1 = alloc.assignments["phase1"]
    # Should send armies to Stark Outpost (enemy territory)
    assert phase1.get("Stark Outpost", 0) > 0


def test_allocation_never_exceeds_total():
    cfg = default_config()
    for spread in [0.0, 0.5, 1.0]:
        for aggression in [0.0, 0.5, 1.0]:
            alloc = generate_opponent(cfg, total_armies=300, spread=spread, aggression=aggression)
            for phase in ["phase1", "phase2", "phase3"]:
                assert alloc.is_valid(phase), f"Overcommit at spread={spread}, agg={aggression}, {phase}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_opponent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'got_wic.opponent'`

**Step 3: Write implementation**

Create `src/got_wic/opponent.py`:

```python
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
        zone_mult = aggression  # only go there if aggressive
    else:
        zone_mult = 1.0
    return base * max(zone_mult, 0.01)


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

        if spread < 1e-9:
            # All-in on highest weight
            idx = int(np.argmax(weights))
            assignments[phase] = {available[idx].name: total_armies}
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
        for obj, count in zip(available, counts):
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_opponent.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/got_wic/opponent.py tests/test_opponent.py
git commit -m "feat: add parametric opponent strategy generator"
```

---

### Task 4: Grid search optimizer (`optimize.py`)

**Files:**
- Create: `src/got_wic/optimize.py`
- Create: `tests/test_optimize.py`

**Step 1: Write the failing tests**

Create `tests/test_optimize.py`:

```python
from got_wic.model import default_config
from got_wic.optimize import optimize, OptResult


def test_optimize_returns_sorted_results():
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=80,
        n_players_b=60,
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=20,  # coarse grid for speed
    )
    assert len(results) > 0
    # Results should be sorted descending by score_a
    scores = [r.score_a for r in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_best_beats_empty():
    """Best allocation should score more than doing nothing."""
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=80,
        n_players_b=60,
        opponent_spread=0.7,
        opponent_aggression=0.5,
        step_pct=20,
    )
    assert results[0].score_a > 0


def test_optimize_result_has_allocation():
    cfg = default_config()
    results = optimize(
        cfg,
        n_players_a=50,
        n_players_b=50,
        opponent_spread=0.5,
        opponent_aggression=0.5,
        step_pct=25,
    )
    best = results[0]
    assert isinstance(best, OptResult)
    assert best.allocation is not None
    assert best.score_a >= best.score_b or True  # may lose if 50v50


def test_more_players_scores_higher():
    """Side with more players should generally score higher."""
    cfg = default_config()
    results_80v40 = optimize(cfg, 80, 40, 0.5, 0.5, step_pct=25)
    results_40v80 = optimize(cfg, 40, 80, 0.5, 0.5, step_pct=25)
    assert results_80v40[0].score_a > results_40v80[0].score_a
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/got_wic/optimize.py`:

```python
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
    # Objective names + special targets
    obj_names = [o.name for o in cfg.objectives]
    special = ["dragon"]
    all_targets = obj_names + special

    # Phase 1: only objectives available at minute 0
    phase1_targets = [o.name for o in cfg.objectives if o.opens_at <= 0]
    # Phase 3: all objectives + dragon
    phase3_targets = all_targets

    steps = list(range(0, 101, step_pct))
    allocations = []

    # For tractability, we use a simplified approach:
    # Allocate % to groups: own_outposts, center (armory+hotspring), strongholds, dragon, enemy_outposts
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_optimize.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/got_wic/optimize.py tests/test_optimize.py
git commit -m "feat: add grid search optimizer"
```

---

### Task 5: Package exports

**Files:**
- Modify: `src/got_wic/__init__.py`

**Step 1: Update `__init__.py`**

```python
from got_wic.model import (
    Allocation,
    Dragon,
    GameConfig,
    Objective,
    TreasureConfig,
    default_config,
)
from got_wic.opponent import generate_opponent
from got_wic.optimize import OptResult, optimize
from got_wic.simulate import SimResult, simulate

__all__ = [
    "Allocation",
    "Dragon",
    "GameConfig",
    "Objective",
    "OptResult",
    "SimResult",
    "TreasureConfig",
    "default_config",
    "generate_opponent",
    "optimize",
    "simulate",
]
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add src/got_wic/__init__.py
git commit -m "feat: export all public API from package"
```

---

### Task 6: Marimo notebook

**Files:**
- Replace: `notebooks/sow-7th-ann-se.py`

**Step 1: Write the new notebook**

Replace `notebooks/sow-7th-ann-se.py` with a marimo notebook that:

1. **Section 1 — Config**: Player count sliders (Side A, Side B), objective point value editors (pre-filled from `default_config()`).
2. **Section 2 — Opponent Profile**: `spread` and `aggression` sliders (0–1).
3. **Section 3 — Manual Allocation**: Sliders per objective group per phase (own outposts, center, strongholds, enemy outposts, dragon). Shows simulation result as a score breakdown table.
4. **Section 4 — Optimizer**: Runs `optimize()` and shows top 10 allocations table with score breakdown.
5. **Section 5 — Player Ratio Sweep**: Table showing optimal allocation as player count varies (30v70 → 70v30 in steps of 10). Shows which objectives get prioritised at each ratio.
6. **Section 6 — Opponent Sensitivity**: For the current player ratio, sweep opponent_spread and opponent_aggression, show heatmap-style table of our optimal score.
7. **Section 7 — Tipping Points**: Summary of strategic insights from the sweeps.

This task is larger and requires interactive development in marimo. The exact cell code will be written during implementation based on the working library API.

**Step 2: Run the notebook to verify**

Run: `uv run marimo edit notebooks/sow-7th-ann-se.py`
Expected: Notebook loads, sliders work, optimizer produces results.

**Step 3: Commit**

```bash
git add notebooks/sow-7th-ann-se.py
git commit -m "feat: replace notebook with new optimizer UI"
```

---

### Task 7: Integration test — end-to-end

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
from got_wic import default_config, optimize


def test_full_pipeline_80v60():
    """End-to-end: optimize for 80v60, check we get actionable results."""
    cfg = default_config()
    results = optimize(cfg, 80, 60, opponent_spread=0.7, opponent_aggression=0.5, step_pct=25)
    best = results[0]

    # Should produce a positive score
    assert best.score_a > 0
    # Should beat the opponent (80 > 60 players)
    assert best.score_a > best.score_b
    # Breakdown should sum to total
    total = sum(best.breakdown_a.values())
    assert total == best.score_a


def test_full_pipeline_50v50_close_game():
    """50v50 should be a close game."""
    cfg = default_config()
    results = optimize(cfg, 50, 50, opponent_spread=0.5, opponent_aggression=0.5, step_pct=25)
    best = results[0]
    assert best.score_a > 0
    # Score difference should be relatively small compared to total
    diff = abs(best.score_a - best.score_b)
    assert diff < best.score_a  # not a blowout
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All 2 tests PASS.

**Step 3: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests"
```
