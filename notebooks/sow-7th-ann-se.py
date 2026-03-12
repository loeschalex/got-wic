# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "numpy>=2.2.0",
#     "matplotlib>=3.9.0",
#     "got-wic",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from __future__ import annotations

    import marimo as mo
    import numpy as np

    from got_wic.model import (
        Allocation,
        AllianceProfile,
        GameConfig,
        PlayerTier,
        default_alliance_profile,
        default_config,
    )
    from got_wic.combat import CombatState, resolve_tick, BuildingFight, apply_attrition
    from got_wic.simulate import simulate, SimResult
    from got_wic.montecarlo import run_monte_carlo, MonteCarloResult, save_results, load_results
    from got_wic.opponent import generate_opponent
    from got_wic.optimize import optimize, OptResult

    return (
        AllianceProfile,
        Allocation,
        PlayerTier,
        default_config,
        generate_opponent,
        mo,
        np,
        optimize,
        run_monte_carlo,
        simulate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Siege of Winterfell — 7th Anniversary Strategy Optimizer

    This notebook models the **Siege of Winterfell (7th Anniversary Special Edition)** event
    using a **stochastic Lanchester combat model** with Monte Carlo simulation.

    ## Scoring channels

    1. **First capture** — one-time points when your alliance first occupies a building.
    2. **Occupation** — points per minute for each building you control.
    3. **Treasure digging** — treasures spawn at random locations from minute 8. Normal = 80 pts, Rare = 120 pts.
    4. **Dragon escort** — Twin dragons spawn at minute 12. Each worth 3,000 pts. Majority controls escort direction.

    ## Combat model

    - **Lanchester attrition** — power losses proportional to opponent strength each tick
    - **Player tiers** — whales, dolphins, minnows, alts with different combat power and healing
    - **Healing** — players can recover from losses until their healing budget runs out
    - **Noise** — stochastic variance models real-game uncertainty
    - **Monte Carlo** — run N trials to get win probability, score distributions, and percentiles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Game Config
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
        label="Building parameters",
    )
    return (cfg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Alliance Profiles
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Side A (your alliance)
    """)
    return


@app.cell
def _(mo):
    a_whale_n = mo.ui.number(value=2, start=0, label="Whales (100 power, unlimited heal)")
    a_dolphin_n = mo.ui.number(value=14, start=0, label="Dolphins (30 power, 8 heals)")
    a_minnow_n = mo.ui.number(value=24, start=0, label="Minnows (8 power, 4 heals)")
    a_alt_n = mo.ui.number(value=40, start=0, label="Alts (1 power, 2 heals)")
    mo.vstack([a_whale_n, a_dolphin_n, a_minnow_n, a_alt_n])
    return a_alt_n, a_dolphin_n, a_minnow_n, a_whale_n


@app.cell
def _(
    AllianceProfile,
    PlayerTier,
    a_alt_n,
    a_dolphin_n,
    a_minnow_n,
    a_whale_n,
    mo,
):
    profile_a = AllianceProfile(
        tiers=[
            PlayerTier("whale", 100.0, -1),
            PlayerTier("dolphin", 30.0, 8),
            PlayerTier("minnow", 8.0, 4),
            PlayerTier("alt", 1.0, 2),
        ],
        counts=[a_whale_n.value, a_dolphin_n.value, a_minnow_n.value, a_alt_n.value],
    )
    mo.md(f"**Side A:** {profile_a.total_players} players, {profile_a.total_power:.0f} total power")
    return (profile_a,)


@app.cell
def _(mo):
    mo.md("""
    ### Side B (opponent alliance)
    """)
    return


@app.cell
def _(mo):
    b_whale_n = mo.ui.number(value=1, start=0, label="Whales")
    b_dolphin_n = mo.ui.number(value=10, start=0, label="Dolphins")
    b_minnow_n = mo.ui.number(value=20, start=0, label="Minnows")
    b_alt_n = mo.ui.number(value=30, start=0, label="Alts")
    mo.vstack([b_whale_n, b_dolphin_n, b_minnow_n, b_alt_n])
    return b_alt_n, b_dolphin_n, b_minnow_n, b_whale_n


@app.cell
def _(
    AllianceProfile,
    PlayerTier,
    b_alt_n,
    b_dolphin_n,
    b_minnow_n,
    b_whale_n,
    mo,
):
    profile_b = AllianceProfile(
        tiers=[
            PlayerTier("whale", 100.0, -1),
            PlayerTier("dolphin", 30.0, 8),
            PlayerTier("minnow", 8.0, 4),
            PlayerTier("alt", 1.0, 2),
        ],
        counts=[b_whale_n.value, b_dolphin_n.value, b_minnow_n.value, b_alt_n.value],
    )
    mo.md(f"**Side B:** {profile_b.total_players} players, {profile_b.total_power:.0f} total power")
    return (profile_b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Opponent Behavior
    """)
    return


@app.cell
def _(mo):
    opp_spread = mo.ui.slider(0.0, 1.0, value=0.7, step=0.05, label="Opponent spread (0=concentrated, 1=even)")
    opp_aggression = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="Opponent aggression (0=defensive, 1=aggressive)")
    mo.vstack([opp_spread, opp_aggression])
    return opp_aggression, opp_spread


@app.cell(hide_code=True)
def _(cfg, generate_opponent, mo, opp_aggression, opp_spread, profile_b):
    opp_alloc = generate_opponent(cfg, profile_b.total_players * 3, opp_spread.value, opp_aggression.value)
    opp_rows = []
    for _phase in ["phase1", "phase2", "phase3"]:
        _assigns = opp_alloc.assignments.get(_phase, {})
        for _name, _armies in sorted(_assigns.items()):
            opp_rows.append({"Phase": _phase, "Target": _name, "Armies": _armies})
    mo.vstack([
        mo.md("### Opponent allocation preview"),
        mo.ui.table(opp_rows, label=f"Total opponent armies: {profile_b.total_players * 3}"),
    ])
    return (opp_alloc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Monte Carlo Controls
    """)
    return


@app.cell
def _(mo):
    mc_n_trials = mo.ui.slider(10, 500, value=50, step=10, label="Number of MC trials")
    mc_noise = mo.ui.slider(0.0, 0.5, value=0.1, step=0.01, label="Noise scale (0=deterministic)")
    mo.vstack([mc_n_trials, mc_noise])
    return mc_n_trials, mc_noise


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Manual Allocation
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
    mc_n_trials,
    mc_noise,
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
    profile_a,
    profile_b,
    run_monte_carlo,
    simulate,
):
    total_a = profile_a.total_players * 3

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

    # Run Monte Carlo on manual allocation
    manual_mc = run_monte_carlo(
        cfg, manual_alloc, opp_alloc, profile_a, profile_b,
        n_trials=mc_n_trials.value, noise_scale=mc_noise.value,
    )

    # Deterministic run for breakdown
    manual_det = simulate(cfg, manual_alloc, opp_alloc, profile_a, profile_b, noise_scale=0.0)

    p1_total_pct = p1_stark.value + p1_armory.value + p1_hotspring.value + p1_greyjoy.value
    p3_total_pct = p3_stark.value + p3_armory.value + p3_hotspring.value + p3_stronghold.value + p3_greyjoy.value + p3_dragon.value

    _hopeless_banner = ""
    if manual_mc.hopeless:
        _hopeless_banner = "\n> **HOPELESS** — win rate below 5%. Consider adjusting your alliance composition or strategy.\n"

    mo.md(f"""
    ### Manual Allocation — Monte Carlo Results ({mc_n_trials.value} trials, noise={mc_noise.value})
    {_hopeless_banner}

    | Metric | Value |
    |--------|------:|
    | **Win Rate** | **{manual_mc.win_rate:.1%}** |
    | Mean Score A | {manual_mc.mean_score_a:,.0f} ± {manual_mc.std_score_a:,.0f} |
    | Mean Score B | {manual_mc.mean_score_b:,.0f} ± {manual_mc.std_score_b:,.0f} |
    | P5 (worst case) | {manual_mc.percentiles[5]:,.0f} |
    | P25 (conservative) | {manual_mc.percentiles[25]:,.0f} |
    | **P50 (median)** | **{manual_mc.percentiles[50]:,.0f}** |
    | P75 | {manual_mc.percentiles[75]:,.0f} |
    | P95 (best case) | {manual_mc.percentiles[95]:,.0f} |

    **Deterministic breakdown:**

    | Channel | Side A | Side B |
    |---------|-------:|-------:|
    | First capture | {manual_det.breakdown_a["first_capture"]:,} | {manual_det.breakdown_b["first_capture"]:,} |
    | Hold | {manual_det.breakdown_a["hold"]:,} | {manual_det.breakdown_b["hold"]:,} |
    | Dragon | {manual_det.breakdown_a["dragon"]:,} | {manual_det.breakdown_b["dragon"]:,} |
    | Treasure | {manual_det.breakdown_a["treasure"]:,} | {manual_det.breakdown_b["treasure"]:,} |
    | Healing spent | {manual_det.healing_spent_a} | {manual_det.healing_spent_b} |
    | | | |
    | Phase 1-2 used | {p1_total_pct}% | |
    | Phase 3 used | {p3_total_pct}% | |
    """)
    return (manual_mc,)


@app.cell(hide_code=True)
def _(manual_mc, mo, np):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(manual_mc.score_distribution_a, bins=20, alpha=0.7, label="Side A", color="#2563eb")
    ax.hist(manual_mc.score_distribution_b, bins=20, alpha=0.7, label="Side B", color="#dc2626")
    ax.axvline(np.median(manual_mc.score_distribution_a), color="#2563eb", linestyle="--", label=f"A median: {np.median(manual_mc.score_distribution_a):,.0f}")
    ax.axvline(np.median(manual_mc.score_distribution_b), color="#dc2626", linestyle="--", label=f"B median: {np.median(manual_mc.score_distribution_b):,.0f}")
    ax.set_xlabel("Total Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Score Distribution (Monte Carlo)")
    ax.legend()
    fig.tight_layout()

    mo.vstack([
        mo.md("### Score Histogram"),
        mo.as_html(fig),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Optimizer
    """)
    return


@app.cell
def _(mo):
    step_pct = mo.ui.slider(10, 50, value=25, step=5, label="Grid step % (smaller = slower but more precise)")
    ranking_mode = mo.ui.dropdown(
        options=["aggressive", "conservative", "win_focused"],
        value="aggressive",
        label="Ranking mode",
    )
    mo.vstack([step_pct, ranking_mode])
    return ranking_mode, step_pct


@app.cell(hide_code=True)
def _(
    cfg,
    mc_n_trials,
    mc_noise,
    mo,
    opp_aggression,
    opp_spread,
    optimize,
    profile_a,
    profile_b,
    ranking_mode,
    step_pct,
):
    opt_results = optimize(
        cfg,
        profile_a=profile_a,
        profile_b=profile_b,
        opponent_spread=opp_spread.value,
        opponent_aggression=opp_aggression.value,
        step_pct=step_pct.value,
        n_trials=mc_n_trials.value,
        noise_scale=mc_noise.value,
        ranking=ranking_mode.value,
    )
    top10 = opt_results[:10]

    opt_rows = []
    for _rank, _r in enumerate(top10, 1):
        _p1 = _r.allocation.assignments.get("phase1", {})
        _p3 = _r.allocation.assignments.get("phase3", {})
        opt_rows.append({
            "#": _rank,
            "Win%": f"{_r.win_rate:.0%}",
            "Mean A": f"{_r.mean_score_a:,.0f}",
            "±σ": f"{_r.std_score_a:,.0f}",
            "P25": f"{_r.p25_score_a:,.0f}",
            "Mean B": f"{_r.mean_score_b:,.0f}",
            "Hopeless": "⚠" if _r.hopeless else "",
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
    _hopeless_opt = ""
    if opt_best.hopeless:
        _hopeless_opt = "\n> **HOPELESS** — even the best allocation has <5% win rate.\n"

    mo.vstack([
        mo.md(f"""
        ### Top 10 Allocations — {ranking_mode.value} ranking
        {_hopeless_opt}
        **Best:** win rate {opt_best.win_rate:.0%}, mean score {opt_best.mean_score_a:,.0f} ± {opt_best.std_score_a:,.0f}
        """),
        mo.ui.table(opt_rows, label="Armies per objective"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Strategic Insights

    **Key takeaways:**

    1. **Dragon escort is high-value** — 6,000 pts (2×3,000) from minute 12 onwards. Always contest if you have the numbers.
    2. **Strongholds dominate late-game** — 4×180 = 720 pts/min from minute 12. The first-capture bonus (4×600 = 2,400) is also the largest.
    3. **Outposts are early-game anchors** — cheap to hold, steady 80 pts/min income from minute 0.
    4. **Whales are force multipliers** — a single whale (100 power, unlimited heal) can hold a building against many alts.
    5. **Healing budgets matter** — once healing runs out, losses become permanent. Focus fire exhausts enemy heals.
    6. **Noise favors the underdog** — with stochastic variance, weaker sides occasionally win buildings they'd lose deterministically.
    7. **Use conservative ranking** when you need a reliable floor score (tournament settings).
    8. **Use win_focused ranking** when the only thing that matters is winning (head-to-head matches).
    """)
    return


if __name__ == "__main__":
    app.run()
