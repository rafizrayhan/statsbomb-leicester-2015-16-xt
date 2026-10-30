"""
Part 4: The players.

11 v 11 only. Minutes come from the starting XI and substitutions, each match cut at
the first red card. Players with 900+ minutes are ranked against everyone in the league
with 900+ minutes in the same role group (from their most common position).

  Threat added per 90: xT added by a player's completed passes and his carries.
  Threat received per 90: xT added by completed passes he received (credits the
      runner, so a forward who gets on the end of passes is counted).
  Ball wins per 90: successful ball recoveries, interceptions won and tackles won.
  Possessions started: the player whose action began a possession (for Leicester),
      and the open-play xG those possessions produced, in total and within 10 seconds.
      A ball win logged in the 3 seconds before the possession began counts as the start.
  Fast chances by shooter: open-play xG per 90 from shots whose possession began no
      more than 10 seconds earlier (forwards, attacking midfielders and wingers).

Output: results/players.csv, results/leicester_starters.csv, printed tables
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from common import LINEUPS, RED, TEAM, clean_names, clock, load_actions
from part2_league import OPEN_PAT, cut_11v11

MIN90 = 900
ROLE = {
    "Goalkeeper": "GK",
    "Right Center Back": "CB", "Left Center Back": "CB", "Center Back": "CB",
    "Right Back": "FB", "Left Back": "FB", "Right Wing Back": "FB", "Left Wing Back": "FB",
    "Center Defensive Midfield": "CM", "Right Defensive Midfield": "CM",
    "Left Defensive Midfield": "CM", "Right Center Midfield": "CM",
    "Left Center Midfield": "CM", "Center Midfield": "CM",
    "Center Attacking Midfield": "AM/W", "Right Attacking Midfield": "AM/W",
    "Left Attacking Midfield": "AM/W", "Right Wing": "AM/W", "Left Wing": "AM/W",
    "Right Midfield": "AM/W", "Left Midfield": "AM/W",
    "Center Forward": "FW", "Right Center Forward": "FW", "Left Center Forward": "FW",
    "Secondary Striker": "FW",
}
ROLE_NAME = {"GK": "goalkeepers", "CB": "centre-backs", "FB": "full-backs",
             "CM": "central midfielders", "AM/W": "attacking midfielders and wingers",
             "FW": "forwards"}


def minutes(A0):
    """Minutes per player in the 11 v 11 window of each match."""
    L = clean_names(pd.read_csv(LINEUPS), cols=("player",))
    c = clock(A0)
    end_all = A0[A0.period <= 2].assign(c=c).groupby("match_id").c.max()
    reds = A0[A0.card.isin(list(RED))].assign(c=c).groupby("match_id").c.min()
    end = end_all.combine(reds, lambda a, b: min(a, b) if pd.notna(b) else a)
    subs = A0[(A0.type == "Substitution") & (A0.period <= 2)].assign(c=c)
    mins = defaultdict(float)
    team_of = {}
    for mid, g in L.groupby("match_id"):
        e = end[mid]
        on = {p: 0.0 for p in g.player}
        for p, t in zip(g.player, g.team):
            team_of[p] = t
        off = {}
        for r in subs[subs.match_id == mid].itertuples():
            if r.c < e:
                off[r.player] = r.c
                on[r.recipient] = r.c
                team_of[r.recipient] = r.team
        for p, s in on.items():
            mins[p] += max(min(off.get(p, e), e) - s, 0)
    return mins, team_of


def main():
    A0 = load_actions()
    A = cut_11v11(A0, A0)
    mv = cut_11v11(clean_names(pd.read_csv("results/moves_xt.csv.gz")), A0)
    mins, team_of = minutes(A0)
    pos = A.dropna(subset=["player", "position"]).groupby("player").position.agg(
        lambda s: s.value_counts().index[0])

    made = mv.groupby("player").xt_added.sum()
    by_pass = mv[mv.type == "Pass"].groupby("player").xt_added.sum()
    by_carry = mv[mv.type == "Carry"].groupby("player").xt_added.sum()
    recv = mv[mv.type == "Pass"].groupby("recipient").xt_added.sum()
    wins = A[((A.type == "Ball Recovery") & (A.outcome == "Success"))
             | ((A.type == "Interception") & A.outcome.isin(["Won", "Success In Play",
                                                              "Success Out"]))
             | ((A.type == "Duel") & (A.pass_type == "Tackle")
                & A.outcome.isin(["Won", "Success In Play", "Success Out"]))]
    win_n = wins.groupby("player").size()

    rows = []
    for p, m in mins.items():
        if m <= 0 or p not in pos.index:
            continue
        rows.append({"player": p, "team": team_of.get(p), "minutes": m,
                     "position": pos[p], "role": ROLE.get(pos[p], "AM/W"),
                     "xt_p90": made.get(p, 0) / m * 90,
                     "xt_pass_p90": by_pass.get(p, 0) / m * 90,
                     "xt_carry_p90": by_carry.get(p, 0) / m * 90,
                     "xt_received_p90": recv.get(p, 0) / m * 90,
                     "ball_wins_p90": win_n.get(p, 0) / m * 90})
    P = pd.DataFrame(rows)
    Q = P[P.minutes >= MIN90].copy()
    for c in ["xt_p90", "xt_pass_p90", "xt_carry_p90", "xt_received_p90", "ball_wins_p90"]:
        Q[c + "_rank"] = Q.groupby("role")[c].rank(ascending=False, method="min").astype(int)
    Q["role_n"] = Q.groupby("role").player.transform("size")
    P = P.merge(Q[["player"] + [c for c in Q.columns if c.endswith("_rank")] + ["role_n"]],
                on="player", how="left")
    P.to_csv("results/players.csv", index=False)

    # who started Leicester's possessions, and what they produced
    Pz = pd.read_csv("results/possessions.csv.gz")
    key = ["match_id", "period", "possession"]
    a = A[(A.team == A.possession_team) & A.player.notna()].dropna(subset=["x"])
    starter = a.sort_values("idx").groupby(key).agg(starter=("player", "first"),
                                                    start_type=("type", "first"),
                                                    start_t=("t", "first"),
                                                    team=("team", "first"))
    # A ball win is often logged as the last event of the opponent's possession; if a
    # player of the new team won the ball in the 3 seconds before, he gets the credit.
    won = A[((A.type == "Ball Recovery") & (A.outcome == "Success"))
            | ((A.type == "Interception") & A.outcome.isin(["Won", "Success In Play",
                                                             "Success Out"]))
            | ((A.type == "Duel") & (A.pass_type == "Tackle")
               & A.outcome.isin(["Won", "Success In Play", "Success Out"]))]
    won = won[won.team != won.possession_team][["match_id", "period", "t", "team", "player"]]
    st = starter.reset_index().sort_values("start_t")
    st = pd.merge_asof(st, won.sort_values("t").rename(columns={"t": "won_t",
                                                                "player": "winner"}),
                       left_on="start_t", right_on="won_t", by=["match_id", "period", "team"],
                       direction="backward", tolerance=3.0)
    st["starter"] = st.winner.fillna(st.starter)
    starter = st.set_index(key)[["starter", "start_type"]]
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN_PAT)]
    sh = sh.merge(Pz[key + ["t0"]], on=key)
    first = sh.sort_values("t").groupby(key).agg(t=("t", "first"), t0=("t0", "first"))
    first["xg"] = sh.groupby(key).xg.sum()
    first["fast"] = (first.t - first.t0) <= 10
    lei = Pz[(Pz.team == TEAM) & Pz.pattern.isin(OPEN_PAT)].set_index(key)
    lei = lei.join(starter).join(first[["xg", "fast"]], rsuffix="_shot")
    lei["xg_shot"] = lei.xg_shot.fillna(0)
    lei["fast"] = lei.fast.fillna(False).astype(bool)
    S = lei.groupby("starter").agg(possessions=("start_type", "size"),
                                   op_xg=("xg_shot", "sum"),
                                   fast_xg=("xg_shot", lambda v: v[lei.loc[v.index, "fast"]].sum()))
    S["share_of_fast_xg"] = S.fast_xg / S.fast_xg.sum()
    S = S.sort_values("fast_xg", ascending=False)
    S.to_csv("results/leicester_starters.csv")

    sh2 = sh.copy()
    sh2["fast"] = (sh2.t - sh2.t0) <= 10
    g = sh2.groupby("player").agg(op_xg=("xg", "sum"),
                                  fast_xg=("xg", lambda v: v[sh2.loc[v.index, "fast"]].sum()))
    F = P[(P.minutes >= MIN90) & P.role.isin(["FW", "AM/W"])].set_index("player").join(g)
    F[["op_xg", "fast_xg"]] = F[["op_xg", "fast_xg"]].fillna(0)
    F["op_xg_p90"] = F.op_xg / F.minutes * 90
    F["fast_xg_p90"] = F.fast_xg / F.minutes * 90
    F["fast_share"] = F.fast_xg / F.op_xg
    F = F.sort_values("fast_xg_p90", ascending=False)
    F[["team", "minutes", "op_xg_p90", "fast_xg_p90", "fast_share"]].to_csv(
        "results/fast_shooters.csv")

    pd.set_option("display.width", 220)
    L = P[(P.team == TEAM) & (P.minutes >= MIN90)].sort_values("xt_p90", ascending=False)
    print("Leicester players (900+ minutes, 11 v 11). Rank within role among league players "
          "with 900+ minutes.\n")
    print(L[["player", "minutes", "role", "role_n", "xt_p90", "xt_p90_rank", "xt_pass_p90",
             "xt_carry_p90", "xt_carry_p90_rank", "xt_received_p90", "xt_received_p90_rank",
             "ball_wins_p90", "ball_wins_p90_rank"]].round(3).to_string(index=False))
    print("\nWho started Leicester's open-play possessions, and the xG they produced:\n")
    print(S.head(8).round(3).to_string())
    print("\nFast open-play xG per 90 by shooter (forwards, attacking midfielders, wingers; "
          "900+ minutes):")
    print(F.head(6)[["team", "op_xg_p90", "fast_xg_p90", "fast_share"]].round(3).to_string())
    print(f"Vardy: {int(F.fast_xg_p90.rank(ascending=False)['Jamie Vardy'])} of {len(F)} for "
          f"fast xG per 90; rank "
          f"{int(F.op_xg_p90.rank(ascending=False)['Jamie Vardy'])} for all open-play xG per 90")
    for role in ["FW", "AM/W", "CM"]:
        q = P[(P.role == role) & (P.minutes >= MIN90)]
        print(f"\nTop 5 {ROLE_NAME[role]} by threat received per 90:")
        print(q.sort_values("xt_received_p90", ascending=False).head(5)[
            ["player", "team", "xt_received_p90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
