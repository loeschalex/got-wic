import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Siege of Winterfell — Strategy Simulator

    This notebook models the point-accumulation mechanics of the **Siege of Winterfell**
    event in *Game of Thrones: Winter is Coming* (GTArcade).  It simulates a 60-minute
    battle under configurable assumptions and grid-searches over deployment allocations
    to find the strategy that maximises total alliance points.

    ## Scoring channels

    1. **First capture** — one-time points when your alliance first occupies a building.
    2. **Holding** — points per minute for each building you control.
    3. **Chest escort** — chests spawn at Winterfell starting at minute 12.  The first
       yields 4 000 pts; each subsequent chest adds 1 000.  A new chest appears
       5 min after successful delivery.  Escorting takes time and ties up deployments.

    ## Map timeline

    | Period | Minutes | Available |
    |--------|---------|-----------|
    | 1 | 0–12 | All buildings |
    | 2 | 12–60 | All buildings; chests begin spawning |

    ## Deployment model

    Each player contributes 3 deployments.  Deployments are split across three roles:
    **building capture/hold**, **chest escort**, and **reserve**.  The number of buildings
    you can hold is bounded by `min(available_buildings, deployments / cost_per_building)`,
    further attenuated by the **opponent strength** parameter.  Chest throughput is
    constrained by the game rule that only one chest at a time comes from Winterfell
    (next spawns after delivery + delay).

    > **Point values below are placeholders.**  Replace them with the actual in-game
    > values for calibrated results.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Building Parameters
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Each building type: **first-capture points**, **hold points/min**,
    **count** on the map, **capture time**, and **deployment cost** (how many
    deployments needed to capture and garrison one instance).
    """)
    return


@app.cell
def _(mo):
    outpost_first = mo.ui.slider(
        100, 2000, value=500, step=100, label="Outpost — first capture pts"
    )
    outpost_hold = mo.ui.slider(
        10, 500, value=50, step=10, label="Outpost — hold pts/min"
    )
    outpost_count = mo.ui.slider(
        1, 4, value=1, step=1, label="Outposts you can contest"
    )
    outpost_cap_time = mo.ui.slider(
        0.5, 5.0, value=1.0, step=0.5, label="Outpost — capture time (min)"
    )
    outpost_dep_cost = mo.ui.slider(
        1, 6, value=2, step=1, label="Outpost — deployments to hold one"
    )
    mo.vstack(
        [outpost_first, outpost_hold, outpost_count, outpost_cap_time, outpost_dep_cost]
    )
    return (
        outpost_cap_time,
        outpost_count,
        outpost_dep_cost,
        outpost_first,
        outpost_hold,
    )


@app.cell
def _(mo):
    stronghold_first = mo.ui.slider(
        500, 10000, value=3000, step=500, label="Stronghold — first capture pts"
    )
    stronghold_hold = mo.ui.slider(
        50, 1000, value=200, step=50, label="Stronghold — hold pts/min"
    )
    stronghold_count = mo.ui.slider(
        1, 4, value=4, step=1, label="Strongholds (total contestable)"
    )
    stronghold_cap_time = mo.ui.slider(
        0.5, 5.0, value=2.0, step=0.5, label="Stronghold — capture time (min)"
    )
    stronghold_dep_cost = mo.ui.slider(
        1, 10, value=4, step=1, label="Stronghold — deployments to hold one"
    )
    mo.vstack(
        [
            stronghold_first,
            stronghold_hold,
            stronghold_count,
            stronghold_cap_time,
            stronghold_dep_cost,
        ]
    )
    return (
        stronghold_cap_time,
        stronghold_count,
        stronghold_dep_cost,
        stronghold_first,
        stronghold_hold,
    )


@app.cell
def _(mo):
    special_first = mo.ui.slider(
        500, 5000, value=1500, step=500, label="Special bldg — first capture pts"
    )
    special_hold = mo.ui.slider(
        20, 500, value=100, step=20, label="Special bldg — hold pts/min"
    )
    special_count = mo.ui.slider(
        1, 4, value=2, step=1, label="Special bldgs (Hot Spring + Armory)"
    )
    special_cap_time = mo.ui.slider(
        0.5, 5.0, value=1.5, step=0.5, label="Special bldg — capture time (min)"
    )
    special_dep_cost = mo.ui.slider(
        1, 8, value=3, step=1, label="Special bldg — deployments to hold one"
    )
    mo.vstack(
        [special_first, special_hold, special_count, special_cap_time, special_dep_cost]
    )
    return (
        special_cap_time,
        special_count,
        special_dep_cost,
        special_first,
        special_hold,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Chest / Escort Parameters
    """)
    return


@app.cell
def _(mo):
    chest_base_pts = mo.ui.slider(
        1000, 8000, value=4000, step=500, label="First chest points"
    )
    chest_increment = mo.ui.slider(
        500, 3000, value=1000, step=500, label="Point increment per chest"
    )
    chest_spawn_delay = mo.ui.slider(
        1, 10, value=5, step=1, label="Respawn delay after delivery (min)"
    )
    escort_time = mo.ui.slider(
        1, 10, value=3, step=1, label="Escort round-trip time (min)"
    )
    escort_dep_cost = mo.ui.slider(
        1, 6, value=3, step=1, label="Deployments tied up per escort run"
    )
    mo.vstack(
        [
            chest_base_pts,
            chest_increment,
            chest_spawn_delay,
            escort_time,
            escort_dep_cost,
        ]
    )
    return (
        chest_base_pts,
        chest_increment,
        chest_spawn_delay,
        escort_dep_cost,
        escort_time,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Force & Opponent Parameters
    """)
    return


@app.cell
def _(mo):
    n_players = mo.ui.slider(5, 50, value=20, step=1, label="Alliance players in event")
    opp_strength = mo.ui.slider(
        0.0, 1.0, value=0.7, step=0.05, label="Hold-success rate (vs opponent)"
    )
    mo.vstack([n_players, opp_strength])
    return n_players, opp_strength


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Manual Strategy Allocation

    Set the percentage of your total deployment budget for each role.
    Reserve = 100% − buildings% − chests% (clamped to ≥ 0).
    """)
    return


@app.cell
def _(mo):
    pct_buildings = mo.ui.slider(
        0, 100, value=60, step=5, label="% deployments → buildings"
    )
    pct_chests = mo.ui.slider(
        0, 100, value=30, step=5, label="% deployments → chest escort"
    )
    mo.vstack([pct_buildings, pct_chests])
    return pct_buildings, pct_chests


@app.cell
def _(mo):
    mo.md("""
    ## 5 · Simulation Results
    """)
    return


@app.cell
def _(
    chest_base_pts,
    chest_increment,
    chest_spawn_delay,
    escort_dep_cost,
    escort_time,
    n_players,
    np,
    opp_strength,
    outpost_cap_time,
    outpost_count,
    outpost_dep_cost,
    outpost_first,
    outpost_hold,
    special_cap_time,
    special_count,
    special_dep_cost,
    special_first,
    special_hold,
    stronghold_cap_time,
    stronghold_count,
    stronghold_dep_cost,
    stronghold_first,
    stronghold_hold,
):
    MATCH_DURATION = 60
    PHASE2_START = 12

    _BLDG_CATALOG = [
        (
            "Outpost",
            outpost_count.value,
            outpost_first.value,
            outpost_hold.value,
            0,
            outpost_cap_time.value,
            outpost_dep_cost.value,
        ),
        (
            "Stronghold",
            stronghold_count.value,
            stronghold_first.value,
            stronghold_hold.value,
            PHASE2_START,
            stronghold_cap_time.value,
            stronghold_dep_cost.value,
        ),
        (
            "Special",
            special_count.value,
            special_first.value,
            special_hold.value,
            PHASE2_START,
            special_cap_time.value,
            special_dep_cost.value,
        ),
    ]
    _CHEST_BASE = chest_base_pts.value
    _CHEST_INC = chest_increment.value
    _CHEST_DELAY = chest_spawn_delay.value
    _ESCORT_TIME = escort_time.value
    _ESCORT_DEP = escort_dep_cost.value
    _HOLD_RATE = opp_strength.value
    _N_PLAYERS = n_players.value

    def simulate(pct_bld: int, pct_ch: int) -> dict:
        total_dep = _N_PLAYERS * 3
        dep_bld = int(total_dep * pct_bld / 100)
        dep_ch = int(total_dep * pct_ch / 100)
        dep_res = total_dep - dep_bld - dep_ch

        # ── Greedy building allocation (by hold-value per deployment) ──
        sorted_bldgs = sorted(
            _BLDG_CATALOG,
            key=lambda b: b[3] / max(b[6], 1),
            reverse=True,
        )
        remaining_dep = dep_bld
        held = {}
        for bname, bmax, bfirst, bhold, bavail, bcap, bcost in sorted_bldgs:
            if remaining_dep <= 0 or bcost <= 0:
                held[bname] = (0, bavail + bcap, bfirst, bhold)
                continue
            can_afford = remaining_dep // bcost
            n_raw = min(can_afford, bmax)
            n_held = max(0, int(np.floor(n_raw * _HOLD_RATE)))
            remaining_dep -= n_held * bcost
            held[bname] = (n_held, bavail + bcap, bfirst, bhold)

        # ── Can we escort? ──
        can_escort = dep_ch >= _ESCORT_DEP

        # ── Minute-by-minute ──
        cum = 0
        fc_total = 0
        h_total = 0
        ch_total = 0
        timeline = []
        next_chest_avail = float(PHASE2_START)
        chests_delivered = 0
        escort_busy_until = -1  # minute when current escort finishes

        for t in range(MATCH_DURATION):
            mf = mh = mc = 0

            for bname, (nh, cmn, bfirst, bhold) in held.items():
                cap_t = int(np.ceil(cmn))
                if t == cap_t and nh > 0:
                    mf += bfirst * nh
                if t >= cap_t:
                    mh += bhold * nh

            # Chest logic
            if can_escort:
                # Check delivery
                if escort_busy_until >= 0 and t >= escort_busy_until:
                    pts = _CHEST_BASE + chests_delivered * _CHEST_INC
                    mc += pts
                    chests_delivered += 1
                    next_chest_avail = t + _CHEST_DELAY
                    escort_busy_until = -1

                # Launch new escort
                if escort_busy_until < 0 and t >= next_chest_avail:
                    escort_busy_until = t + _ESCORT_TIME

            fc_total += mf
            h_total += mh
            ch_total += mc
            cum += mf + mh + mc
            timeline.append(
                {
                    "minute": t,
                    "first_capture": mf,
                    "hold": mh,
                    "chest": mc,
                    "total_this_min": mf + mh + mc,
                    "cumulative": cum,
                }
            )

        return {
            "total": cum,
            "first_capture": fc_total,
            "hold": h_total,
            "chest": ch_total,
            "chests_delivered": chests_delivered,
            "pct_bld": pct_bld,
            "pct_ch": pct_ch,
            "pct_res": 100 - pct_bld - pct_ch,
            "dep_bld": dep_bld,
            "dep_ch": dep_ch,
            "dep_res": dep_res,
            "dep_total": total_dep,
            "buildings_held": {k: v[0] for k, v in held.items()},
            "timeline": timeline,
        }

    return (simulate,)


@app.cell
def _(pct_buildings, pct_chests, simulate):
    _pb = min(pct_buildings.value, 100)
    _pc = min(pct_chests.value, 100 - _pb)
    manual_result = simulate(_pb, _pc)
    return (manual_result,)


@app.cell
def _(manual_result, mo):
    _r = manual_result
    _bh = _r["buildings_held"]
    mo.md(
        f"""
        ### Manual Allocation Result

        | Metric | Value |
        |--------|------:|
        | **Total points** | **{_r["total"]:,}** |
        | First-capture pts | {_r["first_capture"]:,} |
        | Holding pts | {_r["hold"]:,} |
        | Chest-escort pts | {_r["chest"]:,} |
        | Chests delivered | {_r["chests_delivered"]} |
        | | |
        | Deployments → bldg | {_r["dep_bld"]} / {_r["dep_total"]} |
        | Deployments → chest | {_r["dep_ch"]} / {_r["dep_total"]} |
        | Deployments → reserve | {_r["dep_res"]} / {_r["dep_total"]} |
        | | |
        | Outposts held | {_bh.get("Outpost", 0)} |
        | Strongholds held | {_bh.get("Stronghold", 0)} |
        | Special bldgs held | {_bh.get("Special", 0)} |
        """
    )
    return


@app.cell
def _(manual_result, mo):
    _r = manual_result
    _t = max(_r["total"], 1)
    mo.md(
        f"""
        ### Point Breakdown

        | Source | Points | Share |
        |--------|-------:|------:|
        | First capture | {_r["first_capture"]:,} | {_r["first_capture"] / _t * 100:.1f}% |
        | Holding | {_r["hold"]:,} | {_r["hold"] / _t * 100:.1f}% |
        | Chest escort | {_r["chest"]:,} | {_r["chest"] / _t * 100:.1f}% |
        | **Total** | **{_r["total"]:,}** | 100% |
        """
    )
    return


@app.cell
def _(manual_result, mo):
    _tl = manual_result["timeline"]
    _rows = [e for e in _tl if e["minute"] % 5 == 0 or e["minute"] == 59]
    _lines = []
    for _e in _rows:
        _lines.append(
            f"| {_e['minute']:>3} | {_e['first_capture']:>8,} "
            f"| {_e['hold']:>8,} | {_e['chest']:>8,} "
            f"| {_e['total_this_min']:>8,} | {_e['cumulative']:>10,} |"
        )
    _hdr = (
        "| Min | 1st Cap  |   Hold   |  Chest   |  Minute  | Cumulative |\n"
        "|----:|:--------:|:--------:|:--------:|:--------:|:----------:|"
    )
    mo.md(f"### Timeline (every 5 min)\n\n{_hdr}\n" + "\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 · Allocation Optimiser

    Grid search over `(% buildings, % chests)` in 5 % steps.
    """)
    return


@app.cell
def _(simulate):
    _all = []
    for _pb in range(0, 105, 5):
        for _pc in range(0, 105 - _pb, 5):
            _all.append(simulate(_pb, _pc))
    _all.sort(key=lambda x: -x["total"])
    opt_results = _all
    opt_best = _all[0]
    return opt_best, opt_results


@app.cell
def _(mo, opt_best, opt_results):
    _top = opt_results[:10]
    _lines = []
    for _i, _r in enumerate(_top):
        _tag = " ← best" if _i == 0 else ""
        _lines.append(
            f"| {_r['pct_bld']:>3}% | {_r['pct_ch']:>3}% | {_r['pct_res']:>3}% "
            f"| {_r['dep_bld']:>4} | {_r['dep_ch']:>4} "
            f"| {_r['total']:>10,} | {_r['chests_delivered']:>2} "
            f"| {_r['first_capture']:>8,} | {_r['hold']:>8,} | {_r['chest']:>8,} |{_tag}"
        )
    _hdr = (
        "| Bldg | Chest | Res | #DepB | #DepC |      Total | Ch | 1st Cap  |   Hold   |  Chest   |\n"
        "|-----:|------:|----:|------:|------:|-----------:|---:|---------:|---------:|---------:|"
    )
    _b = opt_best
    mo.md(
        f"""
        ### Top 10 Allocations

        {_hdr}
        {"".join(chr(10) + l for l in _lines)}

        **Recommendation**: **{_b["pct_bld"]}%** buildings, **{_b["pct_ch"]}%** chests,
        **{_b["pct_res"]}%** reserve → **{_b["total"]:,}** projected points.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7 · Sensitivity: Hold-Success Sweep

    The table below re-runs the optimiser for hold-success rates from 0.3 to 1.0,
    showing how the optimal allocation shifts with opponent pressure.
    """)
    return


@app.cell
def _(
    chest_base_pts,
    chest_increment,
    chest_spawn_delay,
    escort_dep_cost,
    escort_time,
    n_players,
    np,
    outpost_cap_time,
    outpost_count,
    outpost_dep_cost,
    outpost_first,
    outpost_hold,
    special_cap_time,
    special_count,
    special_dep_cost,
    special_first,
    special_hold,
    stronghold_cap_time,
    stronghold_count,
    stronghold_dep_cost,
    stronghold_first,
    stronghold_hold,
):
    MATCH = 60
    P2 = 12
    _cat = [
        (
            "Outpost",
            outpost_count.value,
            outpost_first.value,
            outpost_hold.value,
            0,
            outpost_cap_time.value,
            outpost_dep_cost.value,
        ),
        (
            "Stronghold",
            stronghold_count.value,
            stronghold_first.value,
            stronghold_hold.value,
            P2,
            stronghold_cap_time.value,
            stronghold_dep_cost.value,
        ),
        (
            "Special",
            special_count.value,
            special_first.value,
            special_hold.value,
            P2,
            special_cap_time.value,
            special_dep_cost.value,
        ),
    ]

    def _sim_hr(pct_bld, pct_ch, hr):
        td = n_players.value * 3
        db = int(td * pct_bld / 100)
        dc = int(td * pct_ch / 100)
        sb = sorted(_cat, key=lambda b: b[3] / max(b[6], 1), reverse=True)
        rem = db
        held = {}
        for bn, bm, bf, bh, ba, bc, bdc in sb:
            if rem <= 0 or bdc <= 0:
                held[bn] = (0, ba + bc, bf, bh)
                continue
            ca = rem // bdc
            nr = min(ca, bm)
            nh = max(0, int(np.floor(nr * hr)))
            rem -= nh * bdc
            held[bn] = (nh, ba + bc, bf, bh)
        can_e = dc >= escort_dep_cost.value
        cum = 0
        nca = float(P2)
        cd = 0
        ebu = -1
        for t in range(MATCH):
            mf = mh = mc = 0
            for bn, (nh, cmn, bf, bh) in held.items():
                ct = int(np.ceil(cmn))
                if t == ct and nh > 0:
                    mf += bf * nh
                if t >= ct:
                    mh += bh * nh
            if can_e:
                if ebu >= 0 and t >= ebu:
                    mc += chest_base_pts.value + cd * chest_increment.value
                    cd += 1
                    nca = t + chest_spawn_delay.value
                    ebu = -1
                if ebu < 0 and t >= nca:
                    ebu = t + escort_time.value
            cum += mf + mh + mc
        return cum

    sweep_rows = []
    for _hr_pct in range(30, 105, 10):
        _hr = _hr_pct / 100
        _best_t = 0
        _best_pb = 0
        _best_pc = 0
        for _pb in range(0, 105, 5):
            for _pc in range(0, 105 - _pb, 5):
                _t = _sim_hr(_pb, _pc, _hr)
                if _t > _best_t:
                    _best_t = _t
                    _best_pb = _pb
                    _best_pc = _pc
        sweep_rows.append((_hr, _best_pb, _best_pc, 100 - _best_pb - _best_pc, _best_t))
    return (sweep_rows,)


@app.cell
def _(mo, sweep_rows):
    _lines = []
    for _hr, _pb, _pc, _pr, _tot in sweep_rows:
        _lines.append(f"| {_hr:.1f} | {_pb:>3}% | {_pc:>3}% | {_pr:>3}% | {_tot:>10,} |")
    _hdr = (
        "| Hold rate | Bldg | Chest | Res |      Total |\n"
        "|---------:|-----:|------:|----:|-----------:|"
    )
    mo.md(f"### Optimal allocation by hold-success rate\n\n{_hdr}\n" + "\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8 · Notes & Limitations

    1. The simulation is **deterministic**. A stochastic variant (Beta-distributed
       hold-success per minute, Monte Carlo over uncertain point values) would
       produce distributional outputs rather than point estimates.

    2. **Only one chest at a time** can be in transit from Winterfell per the game
       rules.  The model enforces this.

    3. The model does **not** account for combat buffs from Hot Spring / Armory,
       troop strength decay, or healing mechanics.

    4. **Temporal reallocation** (shifting from buildings-heavy in Period 1 to
       chests-heavy in Period 2) is not modelled; the current version uses a
       static allocation for the full 60 minutes.

    5. Building point values are **placeholders**.  Update them from in-game
       screenshots for calibrated recommendations.
    """)
    return


if __name__ == "__main__":
    app.run()
