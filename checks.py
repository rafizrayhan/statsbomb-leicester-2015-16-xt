"""
Checks for the Leicester ball-progression project.

  1. Results: all 380 scores rebuilt from events (see part 0).
  2. xT model sensitivity: Leicester's ranks with a coarser (12 x 8) and a finer (24 x 16)
     grid, and with the model trained on all four complete 2015/16 leagues (Premier
     League, La Liga, Serie A, Ligue 1; 1,517 matches) instead of the Premier League only.
  3. Speed threshold: Leicester's rank for fast open-play xG at 5, 10, 15 and 20 seconds.
  4. Game state: fast open-play xG per 90 minutes played while the score was level, and
     while leading / trailing, so Leicester's speed is not just counter-attacking with a
     lead.
  5. Match by match: in how many matches Leicester beat the league median for fast xG,
     and how much of their season total came from their best three matches.

Output: results/checks.txt
"""

import concurrent.futures as cf
import io
import os

import numpy as np
import pandas as pd

from common import BASE, TEAM, clean_names, clock, compact, get_json, load_actions
from part1_xt import build, value_moves
from part2_league import OPEN_PAT, cut_11v11

OTHER = [("La Liga", "2015/2016"), ("Serie A", "2015/2016"), ("Ligue 1", "2015/2016")]
OTHER_FILE = "data/other_leagues.csv.gz"


def other_leagues():
    """Passes, carries and shots from La Liga, Serie A and Ligue 1 2015/16 (cached)."""
    if os.path.exists(OTHER_FILE):
        return pd.read_csv(OTHER_FILE, low_memory=False)
    comps = get_json(f"{BASE}/competitions.json")
    ids = []
    for name, season in OTHER:
        c = [x for x in comps if x["competition_name"] == name and x["season_name"] == season][0]
        ids += [m["match_id"] for m in
                get_json(f"{BASE}/matches/{c['competition_id']}/{c['season_id']}.json")]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (r, _) in enumerate(ex.map(compact, ids)):
            rows += [x for x in r if x["type"] in ("Pass", "Carry", "Shot")]
            if (i + 1) % 200 == 0:
                print(f"  {i + 1} of {len(ids)} other-league matches")
    D = pd.DataFrame(rows)
    D.to_csv(OTHER_FILE, index=False, compression="gzip")
    return D


def leicester_ranks(A, mv):
    """Ranks for xT per match, xT per action and open-play xG per unit of xT."""
    A0 = A
    m = cut_11v11(mv, A0)
    n = A.groupby("team").match_id.nunique()
    xt_pm = m.groupby("team").xt_added.sum() / n
    xt_pa = m.groupby("team").xt_added.mean()
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN_PAT)]
    ratio = sh.groupby("team").xg.sum() / mv.groupby("team").xt_added.sum()
    r = lambda s: int(s.rank(ascending=False, method="min")[TEAM])
    return r(xt_pm), r(xt_pa), r(ratio)


def score_state(A):
    """For every action, the score difference from the acting team's view before it."""
    g = A[((A.type == "Shot") & (A.outcome == "Goal")) | (A.type == "Own Goal For")]
    M = pd.read_csv("data/matches.csv")
    teams = {r.match_id: (r.home, r.away) for r in M.itertuples()}
    goals = {mid: list(zip(d.idx, d.team)) for mid, d in g.groupby("match_id")}
    return goals, teams


def main():
    out = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=out)

    A = load_actions()
    M = pd.read_csv("data/matches.csv")
    gl = A[((A.type == "Shot") & (A.outcome == "Goal")) | (A.type == "Own Goal For")]
    g = gl.groupby(["match_id", "team"]).size()
    ok = sum(g.get((r.match_id, r.home), 0) == r.home_score
             and g.get((r.match_id, r.away), 0) == r.away_score for r in M.itertuples())
    say(f"1. Results: {ok} of {len(M)} scores rebuilt exactly from events.")

    say("\n2. xT model sensitivity (Leicester rank of 20: xT per match / xT per pass or "
        "carry / open-play xG per unit of xT)")
    for name, nx, ny, data in [("16 x 12, Premier League (main)", 16, 12, A),
                               ("12 x 8, Premier League", 12, 8, A),
                               ("24 x 16, Premier League", 24, 16, A)]:
        xt, _ = build(data, nx, ny)
        mv = value_moves(A, xt, nx, ny)
        say(f"   {name}: " + " / ".join(map(str, leicester_ranks(A, mv))))
    O = clean_names(other_leagues())
    both = pd.concat([A[A.type.isin(["Pass", "Carry", "Shot"])], O], ignore_index=True)
    xt4, _ = build(both, 16, 12)
    mv4 = value_moves(A, xt4, 16, 12)
    say(f"   16 x 12, trained on 4 leagues ({both.match_id.nunique():,} matches): "
        + " / ".join(map(str, leicester_ranks(A, mv4))))
    xt1, _ = build(A, 16, 12)
    say(f"   correlation of zone values, PL-only vs 4-league model: "
        f"{np.corrcoef(xt1, xt4)[0, 1]:.3f}")

    A11 = cut_11v11(A, A)
    P = pd.read_csv("results/possessions.csv.gz")
    key = ["match_id", "period", "possession"]
    sh = A11[(A11.type == "Shot") & (A11.pass_type == "Open Play")
             & A11.play_pattern.isin(OPEN_PAT)].merge(P[key + ["t0"]], on=key)
    first = sh.sort_values("t").groupby(key).agg(team=("team", "first"), t=("t", "first"),
                                                 t0=("t0", "first"), idx=("idx", "first"),
                                                 minute=("minute", "first"),
                                                 second=("second", "first"))
    first["xg"] = sh.groupby(key).xg.sum()
    first["sec"] = first.t - first.t0
    first = first.reset_index()
    n = A11.groupby("team").match_id.nunique()

    say("\n3. Speed threshold: Leicester's rank for open-play xG per match from shots within "
        "N seconds")
    for s in [5, 10, 15, 20]:
        v = first[first.sec <= s].groupby("team").xg.sum() / n
        say(f"   {s:>2}s: rank {int(v.rank(ascending=False)[TEAM])}, Leicester {v[TEAM]:.3f}, "
            f"next best {v.drop(TEAM).max():.3f}, median {v.median():.3f}")

    say("\n4. Game state: fast (10s) open-play xG per 90 minutes in each state")
    goals = {mid: sorted(zip(d.minute + d.second / 60, d.team))
             for mid, d in gl.groupby("match_id")}
    ends = A11.assign(c=clock(A11)).groupby("match_id").c.max()
    mins = {}
    for r in M.itertuples():
        e = ends.get(r.match_id, 90)
        tl = [0.0] + [c for c, _ in goals.get(r.match_id, []) if c < e] + [e]
        hs = as_ = 0
        evs = [x for x in goals.get(r.match_id, []) if x[0] < e]
        for i in range(len(tl) - 1):
            d = tl[i + 1] - tl[i]
            for team, diff in [(r.home, hs - as_), (r.away, as_ - hs)]:
                st = "level" if diff == 0 else ("leading" if diff > 0 else "trailing")
                mins[(team, st)] = mins.get((team, st), 0) + d
            if i < len(evs):
                if evs[i][1] == r.home:
                    hs += 1
                else:
                    as_ += 1

    def state_at(row):
        c = row.minute + row.second / 60
        hs = as_ = 0
        home, away = M.set_index("match_id").loc[row.match_id, ["home", "away"]]
        for gc, team in goals.get(row.match_id, []):
            if gc < c:
                if team == home:
                    hs += 1
                else:
                    as_ += 1
        diff = (hs - as_) if row.team == home else (as_ - hs)
        return "level" if diff == 0 else ("leading" if diff > 0 else "trailing")

    fast = first[first.sec <= 10].copy()
    fast["state"] = fast.apply(state_at, axis=1)
    for st in ["level", "leading", "trailing"]:
        v = fast[fast.state == st].groupby("team").xg.sum()
        m90 = pd.Series({t: mins.get((t, st), 0) for t in n.index})
        rate = (v.reindex(n.index).fillna(0) / m90 * 90)
        say(f"   {st:<8}: Leicester {rate[TEAM]:.3f} per 90 ({mins.get((TEAM, st), 0):.0f} "
            f"minutes), rank {int(rate.rank(ascending=False)[TEAM])}, median {rate.median():.3f}")

    say("\n5. Match by match (fast open-play xG)")
    pm = fast.groupby(["team", "match_id"]).xg.sum()
    league_med = (pm.groupby("team").sum() / n).median()
    lei = pm.loc[TEAM].reindex(A[A.team == TEAM].match_id.unique()).fillna(0)
    say(f"   Leicester beat the league-median team average ({league_med:.2f}) in "
        f"{(lei > league_med).sum()} of {len(lei)} matches")
    top3 = lei.sort_values(ascending=False).head(3).sum() / lei.sum()
    say(f"   their best 3 matches gave {top3:.0%} of the season's fast xG; without them, "
        f"{(lei.sum() - lei.sort_values().tail(3).sum()) / (len(lei) - 3):.3f} per match "
        f"(league median {league_med:.3f})")
    open("results/checks.txt", "w").write(out.getvalue())


if __name__ == "__main__":
    main()
