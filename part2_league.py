"""
Part 2: Leicester against the league.

All measures use 11 v 11 play (each match cut at the first red card), per match.

  Ball:        share of open-play passes (the possession measure) and possessions per
               match. Time on the ball is not used: StatsBomb timestamps include
               dead-ball time, which stretched some possessions (Leicester's average
               possession lasted 20 seconds, the median 9).
  Threat:      xT added per match and per completed pass or carry (x 1000 in the table).
  Chances:     open-play xG per match (penalties and possessions started by a corner,
               free kick or throw-in excluded), and per possession.
  Where possessions started: average distance from own goal of the first action, and
               the share of open-play possessions starting in the opponent half and in
               the final third. 'Starting value' = the xT of the zone where each
               possession began, summed per match: danger a team had before moving the
               ball at all, which xT added does not count.
  Conceded:    the same threat and chance measures for opponents.

Output: results/team_profile.csv, results/possessions.csv.gz, printed table
"""

import numpy as np
import pandas as pd

from common import RED, SET_PIECES, TEAM, clock, load_actions, load_matches
from part1_xt import NX, NY, zone

OPEN_PAT = ["Regular Play", "From Counter", "From Goal Kick", "From Keeper", "From Kick Off"]
RESTART_PAT = ["From Corner", "From Free Kick", "From Throw In"]


def cut_11v11(df, A):
    reds = A[A.card.isin(list(RED))].assign(c=lambda d: clock(d)).groupby("match_id").c.min()
    c = clock(df)
    lim = df.match_id.map(reds).fillna(999)
    return df[(c < lim) & (df.period <= 2)]


def possessions(A, xt):
    """One row per possession: team, pattern, start location, duration, xG."""
    a = A.dropna(subset=["x"])
    a = a[a.team == a.possession_team]
    g = a.groupby(["match_id", "period", "possession"])
    P = g.agg(team=("possession_team", "first"), pattern=("play_pattern", "first"),
              x=("x", "first"), y=("y", "first"), t0=("t", "min"), t1=("t", "max"),
              minute=("minute", "first"), second=("second", "first")).reset_index()
    P["duration"] = P.t1 - P.t0
    P["start_xt"] = xt[zone(P.x.values, P.y.values)]
    sh = A[(A.type == "Shot") & (A.pass_type != "Penalty")]
    P = P.merge(sh.groupby(["match_id", "period", "possession"]).xg.sum().rename("xg"),
                on=["match_id", "period", "possession"], how="left")
    P["xg"] = P.xg.fillna(0)
    return P


def main():
    A0 = load_actions()
    M = load_matches()
    xt = pd.read_csv("results/xt_grid.csv").values.ravel()   # index = ix * NY + iy
    A = cut_11v11(A0, A0)
    mv = cut_11v11(pd.read_csv("results/moves_xt.csv.gz"), A0)
    P = cut_11v11(possessions(A, xt), A0)
    P.to_csv("results/possessions.csv.gz", index=False, compression="gzip")

    passes = A[(A.type == "Pass") & ~A.pass_type.isin(SET_PIECES)]
    tot = passes.groupby("match_id").size()
    ps = passes.groupby(["match_id", "team"]).size().rename("passes").reset_index()
    ps["share"] = ps.passes / ps.match_id.map(tot)

    opP = P[P.pattern.isin(OPEN_PAT)]
    rows = []
    teams = sorted(A.team.unique())
    opp_of = {}
    for r in M.itertuples():
        opp_of[(r.match_id, r.home)] = r.away
        opp_of[(r.match_id, r.away)] = r.home
    for team in teams:
        ms = A[A.team == team].match_id.unique()
        n = len(ms)
        mine = mv[mv.team == team]
        theirs = mv[mv.match_id.isin(ms) & (mv.team != team)]
        pm = P[P.team == team]
        po = opP[opP.team == team]
        po_opp = opP[opP.match_id.isin(ms) & (opP.team != team)]
        rows.append({
            "team": team,
            "pass_share": ps[ps.team == team].share.mean(),
            "possessions_pm": len(pm) / n,
            "xt_pm": mine.xt_added.sum() / n,
            "xt_per_action_x1000": mine.xt_added.mean() * 1000,
            "op_xg_pm": po.xg.sum() / n,
            "op_xg_per_possession": po.xg.mean(),
            "start_x": po.x.mean(),
            "start_opp_half": (po.x >= 60).mean(),
            "start_final_third": (po.x >= 80).mean(),
            "start_value_pm": po.start_xt.sum() / n,
            "conceded_xt_pm": theirs.xt_added.sum() / n,
            "conceded_op_xg_pm": po_opp.xg.sum() / n,
        })
    T = pd.DataFrame(rows).set_index("team")
    T.to_csv("results/team_profile.csv")
    R = T.rank(ascending=False).astype(int)
    for c in ["conceded_xt_pm", "conceded_op_xg_pm"]:
        R[c] = T[c].rank().astype(int)   # 1 = conceded least
    pd.set_option("display.width", 200)
    lei = pd.DataFrame({"Leicester": T.loc[TEAM], "rank (of 20)": R.loc[TEAM],
                        "league median": T.median(), "best": T.max()})
    lei.loc[["conceded_xt_pm", "conceded_op_xg_pm"], "best"] = \
        T[["conceded_xt_pm", "conceded_op_xg_pm"]].min()
    print("Leicester against the league (11 v 11, per match). Ranks: 1 = highest, except "
          "conceded where 1 = least.\n")
    print(lei.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nTop 5 by starting value per match:")
    print(T.start_value_pm.sort_values(ascending=False).head(5).round(3).to_string())
    print("\nTop 5 by open-play xG per possession:")
    print(T.op_xg_per_possession.sort_values(ascending=False).head(5).round(4).to_string())


if __name__ == "__main__":
    main()
