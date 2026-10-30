"""
Part 3: How Leicester moved the ball.

11 v 11 only. For every team, then Leicester's value, rank (1 = highest) and the
league median.

  Passes or carries: share of a team's xT that came from passes, and xT per pass and
      per carry.
  Speed: seconds from the start of the possession to each pass or carry. Share of xT
      added in the first 10 seconds of open-play possessions, and open-play xG from
      possessions whose first shot came within 10 seconds of the start ('fast').
  Long balls: open-play passes of 35+ yards: share of passes, completion, share of xT.
      'Over the top': completed passes from the team's own half into the final third.
  Where: share of xT added by moves ending in the wide channels (within 18 yards of a
      touchline) and in the penalty box.

Output: results/how_profile.csv, printed table
"""

import numpy as np
import pandas as pd

from common import SET_PIECES, TEAM, load_actions
from part2_league import OPEN_PAT, cut_11v11

FAST = 10
LONG = 35


def in_box(x, y):
    return (x >= 102) & (y >= 18) & (y <= 62)


def main():
    A0 = load_actions()
    A = cut_11v11(A0, A0)
    mv = cut_11v11(pd.read_csv("results/moves_xt.csv.gz"), A0)
    P = pd.read_csv("results/possessions.csv.gz")
    key = ["match_id", "period", "possession"]
    mv = mv.merge(P[key + ["t0", "pattern"]], on=key, how="left")
    mv["since_start"] = mv.t - mv.t0
    passes = A[(A.type == "Pass") & ~A.pass_type.isin(SET_PIECES)].copy()
    passes["long"] = passes.length >= LONG
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN_PAT)]
    sh = sh.merge(P[key + ["t0"]], on=key, how="left")
    first = sh.sort_values("t").groupby(key).first()
    fast_poss = first[(first.t - first.t0) <= FAST].index
    sh_poss = sh.groupby(key).xg.sum()

    rows = []
    for team in sorted(A.team.unique()):
        n = A[A.team == team].match_id.nunique()
        m = mv[mv.team == team]
        op = m[m.pattern.isin(OPEN_PAT)]
        p = passes[passes.team == team]
        lg = p[p.long]
        sp = sh_poss[sh_poss.index.isin(sh[sh.team == team].set_index(key).index.unique())]
        fast_xg = sp[sp.index.isin(fast_poss)].sum()
        ott = m[(m.type == "Pass") & (m.x < 60) & (m.end_x >= 80)]
        rows.append({
            "team": team,
            "xt_share_passes": m[m.type == "Pass"].xt_added.sum() / m.xt_added.sum(),
            "xt_per_pass_x1000": m[m.type == "Pass"].xt_added.mean() * 1000,
            "xt_per_carry_x1000": m[m.type == "Carry"].xt_added.mean() * 1000,
            "xt_first10s_share": op[op.since_start <= FAST].xt_added.sum() / op.xt_added.sum(),
            "fast_op_xg_pm": fast_xg / n,
            "fast_share_of_op_xg": fast_xg / sp.sum(),
            "long_share": p.long.mean(),
            "long_completion": lg.outcome.isna().mean(),
            "xt_share_long": m[(m.type == "Pass") & (m.length >= LONG)].xt_added.sum()
            / m.xt_added.sum(),
            "over_top_pm": len(ott) / n,
            "xt_share_wide_end": m[(m.end_y < 18) | (m.end_y > 62)].xt_added.sum()
            / m.xt_added.sum(),
            "xt_share_box_end": m[in_box(m.end_x, m.end_y) & ~in_box(m.x, m.y)].xt_added.sum()
            / m.xt_added.sum(),
        })
    T = pd.DataFrame(rows).set_index("team")
    T.to_csv("results/how_profile.csv")
    R = T.rank(ascending=False, method="min").astype(int)
    out = pd.DataFrame({"Leicester": T.loc[TEAM], "rank": R.loc[TEAM],
                        "median": T.median(), "top team": T.idxmax(), "top value": T.max()})
    pd.set_option("display.width", 200)
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
