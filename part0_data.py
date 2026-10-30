"""
Part 0: Download and audit the 2015/16 Premier League.

  - Downloads all 380 matches' events and keeps only the fields this project needs
    (data/actions.csv.gz, data/lineups.csv, data/matches.csv).
  - Rebuilds every score from the events (shots that scored + own goals) and checks it
    against the match file.
  - Rebuilds the league table and shows Leicester's season.
  - Counts the passes and carries the xT model will learn from.

Output: data/*, results/league_table.csv, printed audit
"""

import concurrent.futures as cf
import os

import pandas as pd

from common import (ACTIONS, BASE, COMP, LINEUPS, MATCHES, SEASON, TEAM, compact,
                    get_json, load_actions)


def download():
    ms = get_json(f"{BASE}/matches/{COMP}/{SEASON}.json")
    M = pd.DataFrame([{"match_id": m["match_id"], "date": m["match_date"],
                       "week": m.get("match_week"),
                       "home": m["home_team"]["home_team_name"],
                       "away": m["away_team"]["away_team_name"],
                       "home_score": m["home_score"], "away_score": m["away_score"]}
                      for m in ms]).sort_values(["date", "match_id"])
    rows, lineups = [], []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (r, lu) in enumerate(ex.map(compact, M.match_id)):
            rows += r
            lineups += lu
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} of {len(M)} matches")
    os.makedirs("data", exist_ok=True)
    M.to_csv(MATCHES, index=False)
    pd.DataFrame(lineups).to_csv(LINEUPS, index=False)
    pd.DataFrame(rows).to_csv(ACTIONS, index=False, compression="gzip")


def main():
    if not os.path.exists(ACTIONS):
        download()
    M = pd.read_csv(MATCHES)
    A = load_actions()
    os.makedirs("results", exist_ok=True)

    goals = A[((A.type == "Shot") & (A.outcome == "Goal")) | (A.type == "Own Goal For")]
    g = goals.groupby(["match_id", "team"]).size()
    M["home_ev"] = [g.get((m, h), 0) for m, h in zip(M.match_id, M.home)]
    M["away_ev"] = [g.get((m, a), 0) for m, a in zip(M.match_id, M.away)]
    ok = ((M.home_ev == M.home_score) & (M.away_ev == M.away_score)).sum()
    print(f"{len(M)} matches, {A.match_id.nunique()} with events, {len(A):,} actions kept")
    print(f"scores rebuilt from events: {ok} of {len(M)} match exactly")
    bad = M[(M.home_ev != M.home_score) | (M.away_ev != M.away_score)]
    if len(bad):
        print(bad[["date", "home", "away", "home_score", "away_score", "home_ev", "away_ev"]])

    rows = []
    for r in M.itertuples():
        for team, gf, ga in [(r.home, r.home_score, r.away_score),
                             (r.away, r.away_score, r.home_score)]:
            rows.append({"team": team, "w": gf > ga, "d": gf == ga, "l": gf < ga,
                         "gf": gf, "ga": ga})
    T = pd.DataFrame(rows).groupby("team").sum()
    T["played"] = T.w + T.d + T.l
    T["pts"] = 3 * T.w + T.d
    T["gd"] = T.gf - T.ga
    T = T.sort_values(["pts", "gd", "gf"], ascending=False)
    T.insert(0, "pos", range(1, len(T) + 1))
    T[["pos", "played", "w", "d", "l", "gf", "ga", "gd", "pts"]].to_csv(
        "results/league_table.csv")
    print("\nLeague table rebuilt from the match files (top 6 and Leicester)")
    print(T[["pos", "played", "w", "d", "l", "gf", "ga", "gd", "pts"]].head(6).to_string())

    p = A[(A.type == "Pass")]
    c = A[(A.type == "Carry")]
    op = p[~p.pass_type.isin(["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"])]
    print(f"\nPasses: {len(p):,} (open play {len(op):,}, completed open play "
          f"{op.outcome.isna().sum():,}); carries: {len(c):,}; shots: "
          f"{(A.type == 'Shot').sum():,}")
    lei = A[A.team == TEAM]
    print(f"Leicester: {lei.match_id.nunique()} matches, {(lei.type == 'Pass').sum():,} passes, "
          f"{(lei.type == 'Carry').sum():,} carries, {(lei.type == 'Shot').sum()} shots")
    print(f"Matches with a red card: "
          f"{A[A.card.isin(['Red Card', 'Second Yellow'])].match_id.nunique()}")


if __name__ == "__main__":
    main()
