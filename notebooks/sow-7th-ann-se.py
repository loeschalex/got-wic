# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "numpy>=2.2.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from __future__ import annotations
    from dataclasses import dataclass, field
    from itertools import product

    import marimo as mo
    import numpy as np

    return dataclass, field, mo, np, product


@app.cell(hide_code=True)
def _(dataclass, field):
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
        assignments: dict[str, dict[str, int]]
        total_armies: int

        def armies_used(self, phase: str) -> int:
            return sum(self.assignments.get(phase, {}).values())

        def armies_unused(self, phase: str) -> int:
            return self.total_armies - self.armies_used(phase)

        def is_valid(self, phase: str) -> bool:
            return self.armies_used(phase) <= self.total_armies

    return Allocation, Dragon, GameConfig, Objective


@app.cell(hide_code=True)
def _(Dragon, GameConfig, Objective):
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

    return (default_config,)


@app.cell(hide_code=True)
def _(Allocation, GameConfig, dataclass, field):
    @dataclass
    class SimResult:
        score_a: int
        score_b: int
        breakdown_a: dict[str, int] = field(default_factory=dict)
        breakdown_b: dict[str, int] = field(default_factory=dict)

    def _phase_for_minute(minute: int, boundaries: list[int]) -> str:
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
        captured_by: dict[str, str | None] = {}
        dragon_minutes_a = 0
        dragon_minutes_b = 0

        for t in range(cfg.match_duration):
            phase = _phase_for_minute(t, cfg.phase_boundaries)

            for obj in cfg.objectives:
                if t < obj.opens_at:
                    continue
                aa = _armies_at(alloc_a, phase, obj.name)
                ab = _armies_at(alloc_b, phase, obj.name)
                holder = None
                if aa > ab:
                    holder = "a"
                elif ab > aa:
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

            if cfg.dragons and t >= cfg.dragons[0].spawns_at:
                da = _armies_at(alloc_a, phase, "dragon")
                db = _armies_at(alloc_b, phase, "dragon")
                if da > db:
                    dragon_minutes_a += 1
                elif db > da:
                    dragon_minutes_b += 1

            if t >= cfg.treasure.starts_at:
                ta = _armies_at(alloc_a, phase, "treasure")
                tb = _armies_at(alloc_b, phase, "treasure")
                avg_pts = (cfg.treasure.normal_pts + cfg.treasure.rare_pts) // 2
                treasure_a += ta * avg_pts // 10
                treasure_b += tb * avg_pts // 10

        total_dragon_pts = sum(d.escort_pts for d in cfg.dragons)
        if cfg.dragons and (cfg.match_duration > cfg.dragons[0].spawns_at):
            if dragon_minutes_a > dragon_minutes_b:
                dragon_a = total_dragon_pts
            elif dragon_minutes_b > dragon_minutes_a:
                dragon_b = total_dragon_pts

        score_a = fc_a + hold_a + dragon_a + treasure_a
        score_b = fc_b + hold_b + dragon_b + treasure_b
        return SimResult(
            score_a=score_a,
            score_b=score_b,
            breakdown_a={"first_capture": fc_a, "hold": hold_a, "dragon": dragon_a, "treasure": treasure_a},
            breakdown_b={"first_capture": fc_b, "hold": hold_b, "dragon": dragon_b, "treasure": treasure_b},
        )

    return (simulate,)


@app.cell(hide_code=True)
def _(Allocation, GameConfig, Objective, np):
    _OWN_ZONES = {"near_greyjoy"}
    _ENEMY_ZONES = {"near_stark"}

    def _available_objectives(cfg: GameConfig, phase: str) -> list[Objective]:
        phase_idx = int(phase.replace("phase", "")) - 1
        boundaries = cfg.phase_boundaries
        phase_start = boundaries[phase_idx] if phase_idx < len(boundaries) else 0
        return [o for o in cfg.objectives if o.opens_at <= phase_start]

    def _weight_objective(obj: Objective, aggression: float) -> float:
        base = obj.hold_pts_min * obj.count
        if obj.zone in _OWN_ZONES:
            zone_mult = 1.0 + (1.0 - aggression)
        elif obj.zone in _ENEMY_ZONES:
            if aggression < 1e-9:
                return 0.0
            zone_mult = aggression
        else:
            zone_mult = 1.0
        return base * zone_mult

    def generate_opponent(cfg: GameConfig, total_armies: int, spread: float, aggression: float) -> Allocation:
        assignments: dict[str, dict[str, int]] = {}
        for phase in ["phase1", "phase2", "phase3"]:
            available = _available_objectives(cfg, phase)
            if not available:
                assignments[phase] = {}
                continue
            weights = np.array([_weight_objective(o, aggression) for o in available], dtype=float)
            mask = weights > 0
            avail_f = [o for o, m in zip(available, mask) if m]
            weights = weights[mask]
            if len(avail_f) == 0:
                assignments[phase] = {}
                continue
            if spread < 1e-9:
                idx = int(np.argmax(weights))
                assignments[phase] = {avail_f[idx].name: total_armies}
                continue
            temperature = 0.1 + spread * 10.0
            scaled = weights / (weights.max() + 1e-9) * (1.0 / temperature)
            exp_w = np.exp(scaled - scaled.max())
            probs = exp_w / exp_w.sum()
            raw = probs * total_armies
            counts = np.floor(raw).astype(int)
            remainder = total_armies - counts.sum()
            fracs = raw - counts
            for _ in range(int(remainder)):
                idx = int(np.argmax(fracs))
                counts[idx] += 1
                fracs[idx] = -1
            phase_alloc = {}
            for obj, count in zip(avail_f, counts):
                if count > 0:
                    phase_alloc[obj.name] = int(count)
            if phase == "phase3" and cfg.dragons:
                dragon_share = int(total_armies * 0.1 * (0.5 + aggression * 0.5))
                used = sum(phase_alloc.values())
                if used + dragon_share > total_armies:
                    sc = (total_armies - dragon_share) / max(used, 1)
                    phase_alloc = {k: max(1, int(v * sc)) for k, v in phase_alloc.items()}
                phase_alloc["dragon"] = dragon_share
            assignments[phase] = phase_alloc
        return Allocation(assignments=assignments, total_armies=total_armies)

    return (generate_opponent,)


@app.cell(hide_code=True)
def _(Allocation, GameConfig, dataclass, generate_opponent, product, simulate):
    @dataclass
    class OptResult:
        allocation: Allocation
        score_a: int
        score_b: int
        breakdown_a: dict[str, int]
        breakdown_b: dict[str, int]

    def _feasible_combos(steps: list[int], n: int) -> list[tuple[int, ...]]:
        if n == 0:
            return [()]
        if n == 1:
            return [(s,) for s in steps]
        return [combo for combo in product(steps, repeat=n) if sum(combo) <= 100]

    def _generate_allocations(cfg: GameConfig, total_armies: int, step_pct: int) -> list[Allocation]:
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
                    assignments={"phase1": p1_alloc, "phase2": p1_alloc, "phase3": p3_alloc},
                    total_armies=total_armies,
                )
                if alloc.is_valid("phase1") and alloc.is_valid("phase3"):
                    allocations.append(alloc)
        return allocations

    def optimize(cfg, n_players_a, n_players_b, opponent_spread, opponent_aggression, step_pct=10):
        total_a = n_players_a * 3
        total_b = n_players_b * 3
        opponent_alloc = generate_opponent(cfg, total_b, opponent_spread, opponent_aggression)
        candidates = _generate_allocations(cfg, total_a, step_pct)
        results = []
        for alloc in candidates:
            sim = simulate(cfg, alloc, opponent_alloc)
            results.append(OptResult(
                allocation=alloc, score_a=sim.score_a, score_b=sim.score_b,
                breakdown_a=sim.breakdown_a, breakdown_b=sim.breakdown_b,
            ))
        results.sort(key=lambda r: -r.score_a)
        return results

    return (optimize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Siege of Winterfell — 7th Anniversary Strategy Optimizer

    This notebook models the **Siege of Winterfell (7th Anniversary Special Edition)** event.
    It simulates a 60-minute battle and grid-searches over deployment allocations to find
    the strategy that maximises total alliance points.

    ## Scoring channels

    1. **First capture** — one-time points when your alliance first occupies a building.
    2. **Occupation** — points per minute for each building you control.
    3. **Treasure digging** — treasures spawn at random locations from minute 8. Normal = 80 pts, Rare = 120 pts.
    4. **Dragon escort** — Twin dragons spawn at minute 12. Each worth 3,000 pts. Majority controls escort direction.

    ## Map timeline

    | Period | Minutes | Events |
    |--------|---------|--------|
    | 1 | 0–8 | Outposts, Armory, Hot Spring available |
    | 2 | 8–12 | Treasure spawning begins |
    | 3 | 12–60 | Strongholds unlock, Dragons spawn |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Config
    """)
    return


@app.cell(hide_code=True)
def _(default_config, mo):
    cfg = default_config()
    mo.ui.table(
        [
            {
                "Building": o.name,
                "Count": o.count,
                "First Capture": o.first_capture,
                "Hold pts/min": o.hold_pts_min,
                "Opens at": f"{o.opens_at} min",
                "Zone": o.zone,
            }
            for o in cfg.objectives
        ],
        label="Building parameters (from default config)",
    )
    return (cfg,)


@app.cell
def _(mo):
    players_a = mo.ui.slider(10, 100, value=80, step=5, label="Side A players")
    players_b = mo.ui.slider(10, 100, value=60, step=5, label="Side B players")
    mo.vstack([players_a, players_b])
    return players_a, players_b


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Opponent Profile
    """)
    return


@app.cell
def _(mo):
    opp_spread = mo.ui.slider(0.0, 1.0, value=0.7, step=0.05, label="Opponent spread (0=concentrated, 1=even)")
    opp_aggression = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="Opponent aggression (0=defensive, 1=aggressive)")
    mo.vstack([opp_spread, opp_aggression])
    return opp_aggression, opp_spread


@app.cell(hide_code=True)
def _(cfg, generate_opponent, mo, opp_aggression, opp_spread, players_b):
    opp_alloc = generate_opponent(cfg, players_b.value * 3, opp_spread.value, opp_aggression.value)
    opp_rows = []
    for _phase in ["phase1", "phase2", "phase3"]:
        _assigns = opp_alloc.assignments.get(_phase, {})
        for _name, _armies in sorted(_assigns.items()):
            opp_rows.append({"Phase": _phase, "Target": _name, "Armies": _armies})
    mo.vstack([
        mo.md("### Opponent allocation preview"),
        mo.ui.table(opp_rows, label=f"Total opponent armies: {players_b.value * 3}"),
    ])
    return (opp_alloc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Manual Allocation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Set the percentage of your total armies for each objective group per phase.
    Phase 1–2 share the same allocation. Phase 3 has its own (strongholds + dragon unlock).
    """)
    return


@app.cell
def _(mo):
    p1_stark = mo.ui.slider(0, 100, value=30, step=5, label="Phase 1-2: Stark Outpost %")
    p1_armory = mo.ui.slider(0, 100, value=20, step=5, label="Phase 1-2: Armory %")
    p1_hotspring = mo.ui.slider(0, 100, value=20, step=5, label="Phase 1-2: Hot Spring %")
    p1_greyjoy = mo.ui.slider(0, 100, value=10, step=5, label="Phase 1-2: Greyjoy Outpost %")
    mo.vstack([p1_stark, p1_armory, p1_hotspring, p1_greyjoy])
    return p1_armory, p1_greyjoy, p1_hotspring, p1_stark


@app.cell
def _(mo):
    p3_stark = mo.ui.slider(0, 100, value=10, step=5, label="Phase 3: Stark Outpost %")
    p3_armory = mo.ui.slider(0, 100, value=10, step=5, label="Phase 3: Armory %")
    p3_hotspring = mo.ui.slider(0, 100, value=10, step=5, label="Phase 3: Hot Spring %")
    p3_stronghold = mo.ui.slider(0, 100, value=30, step=5, label="Phase 3: Stronghold %")
    p3_greyjoy = mo.ui.slider(0, 100, value=10, step=5, label="Phase 3: Greyjoy Outpost %")
    p3_dragon = mo.ui.slider(0, 100, value=20, step=5, label="Phase 3: Dragon %")
    mo.vstack([p3_stark, p3_armory, p3_hotspring, p3_stronghold, p3_greyjoy, p3_dragon])
    return (
        p3_armory,
        p3_dragon,
        p3_greyjoy,
        p3_hotspring,
        p3_stark,
        p3_stronghold,
    )


@app.cell(hide_code=True)
def _(
    Allocation,
    cfg,
    mo,
    opp_alloc,
    p1_armory,
    p1_greyjoy,
    p1_hotspring,
    p1_stark,
    p3_armory,
    p3_dragon,
    p3_greyjoy,
    p3_hotspring,
    p3_stark,
    p3_stronghold,
    players_a,
    simulate,
):
    total_a = players_a.value * 3

    def _pct_to_armies(pct, total):
        return int(total * pct / 100)

    p1_assign = {}
    for _name, _slider in [
        ("Stark Outpost", p1_stark),
        ("Armory", p1_armory),
        ("Hot Spring", p1_hotspring),
        ("Greyjoy Outpost", p1_greyjoy),
    ]:
        _v = _pct_to_armies(_slider.value, total_a)
        if _v > 0:
            p1_assign[_name] = _v

    p3_assign = {}
    for _name, _slider in [
        ("Stark Outpost", p3_stark),
        ("Armory", p3_armory),
        ("Hot Spring", p3_hotspring),
        ("Stronghold", p3_stronghold),
        ("Greyjoy Outpost", p3_greyjoy),
        ("dragon", p3_dragon),
    ]:
        _v = _pct_to_armies(_slider.value, total_a)
        if _v > 0:
            p3_assign[_name] = _v

    manual_alloc = Allocation(
        assignments={"phase1": p1_assign, "phase2": p1_assign, "phase3": p3_assign},
        total_armies=total_a,
    )
    manual_sim = simulate(cfg, manual_alloc, opp_alloc)

    p1_total_pct = p1_stark.value + p1_armory.value + p1_hotspring.value + p1_greyjoy.value
    p3_total_pct = p3_stark.value + p3_armory.value + p3_hotspring.value + p3_stronghold.value + p3_greyjoy.value + p3_dragon.value

    mo.md(f"""
    ### Manual Allocation Result

    | Metric | Side A | Side B |
    |--------|-------:|-------:|
    | **Total Score** | **{manual_sim.score_a:,}** | **{manual_sim.score_b:,}** |
    | First capture | {manual_sim.breakdown_a["first_capture"]:,} | {manual_sim.breakdown_b["first_capture"]:,} |
    | Hold | {manual_sim.breakdown_a["hold"]:,} | {manual_sim.breakdown_b["hold"]:,} |
    | Dragon | {manual_sim.breakdown_a["dragon"]:,} | {manual_sim.breakdown_b["dragon"]:,} |
    | Treasure | {manual_sim.breakdown_a["treasure"]:,} | {manual_sim.breakdown_b["treasure"]:,} |
    | | | |
    | Phase 1-2 used | {p1_total_pct}% | |
    | Phase 3 used | {p3_total_pct}% | |
    | **Margin** | **{manual_sim.score_a - manual_sim.score_b:+,}** | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Optimizer
    """)
    return


@app.cell
def _(mo):
    step_pct = mo.ui.slider(10, 50, value=25, step=5, label="Grid step % (smaller = slower but more precise)")
    step_pct
    return (step_pct,)


@app.cell(hide_code=True)
def _(
    cfg,
    mo,
    opp_aggression,
    opp_spread,
    optimize,
    players_a,
    players_b,
    step_pct,
):
    opt_results = optimize(
        cfg,
        players_a.value,
        players_b.value,
        opp_spread.value,
        opp_aggression.value,
        step_pct=step_pct.value,
    )
    top10 = opt_results[:10]

    opt_rows = []
    for _rank, _r in enumerate(top10, 1):
        _p1 = _r.allocation.assignments.get("phase1", {})
        _p3 = _r.allocation.assignments.get("phase3", {})
        opt_rows.append({
            "#": _rank,
            "Score A": f"{_r.score_a:,}",
            "Score B": f"{_r.score_b:,}",
            "Margin": f"{_r.score_a - _r.score_b:+,}",
            "P1 Stark": _p1.get("Stark Outpost", 0),
            "P1 Armory": _p1.get("Armory", 0),
            "P1 HotSpr": _p1.get("Hot Spring", 0),
            "P1 Greyjoy": _p1.get("Greyjoy Outpost", 0),
            "P3 Stark": _p3.get("Stark Outpost", 0),
            "P3 Armory": _p3.get("Armory", 0),
            "P3 HotSpr": _p3.get("Hot Spring", 0),
            "P3 Strong": _p3.get("Stronghold", 0),
            "P3 Greyjoy": _p3.get("Greyjoy Outpost", 0),
            "P3 Dragon": _p3.get("dragon", 0),
        })

    opt_best = top10[0]
    mo.vstack([
        mo.md(f"""
        ### Top 10 Allocations ({players_a.value}v{players_b.value})

        **Best score: {opt_best.score_a:,}** (margin: {opt_best.score_a - opt_best.score_b:+,})
        """),
        mo.ui.table(opt_rows, label="Armies per objective"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Player Ratio Sweep
    """)
    return


@app.cell(hide_code=True)
def _(cfg, mo, opp_aggression, opp_spread, optimize):
    sweep_rows = []
    for _a in range(30, 80, 10):
        _b = 100 - _a
        _results = optimize(cfg, _a, _b, opp_spread.value, opp_aggression.value, step_pct=25)
        _best = _results[0]
        _p1 = _best.allocation.assignments.get("phase1", {})
        _p3 = _best.allocation.assignments.get("phase3", {})
        sweep_rows.append({
            "Ratio": f"{_a}v{_b}",
            "Score A": f"{_best.score_a:,}",
            "Score B": f"{_best.score_b:,}",
            "Margin": f"{_best.score_a - _best.score_b:+,}",
            "P1 Stark": _p1.get("Stark Outpost", 0),
            "P1 Armory": _p1.get("Armory", 0),
            "P3 Strong": _p3.get("Stronghold", 0),
            "P3 Dragon": _p3.get("dragon", 0),
            "P3 Greyjoy": _p3.get("Greyjoy Outpost", 0),
        })

    mo.vstack([
        mo.md("### Optimal allocation as player ratio varies"),
        mo.ui.table(sweep_rows, label="Best allocation per ratio"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Opponent Sensitivity
    """)
    return


@app.cell(hide_code=True)
def _(cfg, mo, optimize, players_a, players_b):
    sensitivity_rows = []
    for _spread in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for _agg in [0.0, 0.3, 0.5, 0.7, 1.0]:
            _results = optimize(cfg, players_a.value, players_b.value, _spread, _agg, step_pct=25)
            _best = _results[0]
            sensitivity_rows.append({
                "Spread": _spread,
                "Aggression": _agg,
                "Our Score": f"{_best.score_a:,}",
                "Their Score": f"{_best.score_b:,}",
                "Margin": f"{_best.score_a - _best.score_b:+,}",
            })

    mo.vstack([
        mo.md(f"### Our optimal score vs opponent profile ({players_a.value}v{players_b.value})"),
        mo.ui.table(sensitivity_rows, label="Sweep over opponent spread x aggression"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Strategic Insights

    **Key takeaways from the sweeps above:**

    1. **Dragon escort is high-value** — 6,000 pts (2x3,000) from minute 12 onwards. Always contest if you have the numbers.
    2. **Strongholds dominate late-game** — 4x180 = 720 pts/min from minute 12. The first-capture bonus (4x600 = 2,400) is also the largest.
    3. **Outposts are early-game anchors** — cheap to hold, steady 80 pts/min income from minute 0.
    4. **Armory + Hot Spring** — moderate value (120 pts/min each) but also grant combat buffs (50% stats / 100% heal speed).
    5. **Player advantage compounds** — even a small numbers advantage (e.g. 60v40) lets you contest more objectives simultaneously.
    6. **Against defensive opponents** — push center and strongholds harder; they won't contest your outposts.
    7. **Against aggressive opponents** — defend your outposts and let them overextend.
    """)
    return


if __name__ == "__main__":
    app.run()
