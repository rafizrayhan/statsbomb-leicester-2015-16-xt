"""
Part 3c: Inside Leicester's fast attacks.

Fast attack = an open-play possession (not started by a corner, free kick or throw-in)
whose first shot came within 10 seconds of its first action. 11 v 11 only.

  - Number of fast attacks per match and xG per fast attack, for every team.
  - For Leicester: passes and carries before the first shot, passes per fast attack,
    and the passing links that repeated.

Too few passes repeat for a reliable passing network (most fast attacks had one pass),
so the links are reported as counts only.

Output: results/fast_attacks_teams.csv, results/fast_attack_links.csv, printed summary
"""

import pandas as pd

from common import TEAM, load_actions
from part2_league import OPEN_PAT, cut_11v11

FAST = 10


def main():
    A0 = load_actions()
    A = cut_11v11(A0, A0)
    P = pd.read_csv("results/possessions.csv.gz")
    key = ["match_id", "period", "possession"]
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN_PAT)]
    sh = sh.merge(P[key + ["t0"]], on=key)
    first = sh.sort_values("t").groupby(key).agg(team=("team", "first"), t=("t", "first"),
                                                 t0=("t0", "first"))
    first["xg"] = sh.groupby(key).xg.sum()
    fast = first[(first.t - first.t0) <= FAST]
    n = A.groupby("team").match_id.nunique()
    T = pd.DataFrame({"fast_attacks_pm": fast.groupby("team").size() / n,
                      "xg_per_fast_attack": fast.groupby("team").xg.mean()})
    for c in list(T):
        T[c + "_rank"] = T[c].rank(ascending=False, method="min").astype(int)
    T.to_csv("results/fast_attacks_teams.csv")

    lf = fast[fast.team == TEAM]
    a = A[A.team == TEAM].merge(lf[["t"]].rename(columns={"t": "t_shot"}).reset_index(),
                                on=key)
    a = a[a.t <= a.t_shot]
    passes = a[a.type == "Pass"]
    done = passes[passes.outcome.isna() & passes.recipient.notna()]
    links = done.groupby(["player", "recipient"]).size().sort_values(ascending=False)
    links.rename("passes").to_csv("results/fast_attack_links.csv")
    per = passes.groupby(key).size().reindex(lf.index).fillna(0)

    print(f"Leicester fast attacks: {len(lf)} ({T.loc[TEAM, 'fast_attacks_pm']:.2f} per match, "
          f"rank {T.loc[TEAM, 'fast_attacks_pm_rank']}; median {T.fast_attacks_pm.median():.2f})")
    print(f"xG per fast attack: {T.loc[TEAM, 'xg_per_fast_attack']:.3f} (rank "
          f"{T.loc[TEAM, 'xg_per_fast_attack_rank']}; median {T.xg_per_fast_attack.median():.3f}; "
          f"best {T.xg_per_fast_attack.idxmax()} {T.xg_per_fast_attack.max():.3f})")
    for t in ["Liverpool", "Tottenham Hotspur"]:
        print(f"  {t}: {T.loc[t, 'fast_attacks_pm']:.2f} per match, "
              f"{T.loc[t, 'xg_per_fast_attack']:.3f} xG each")
    print(f"Before the first shot: {len(passes)} passes ({len(done)} completed), "
          f"{(a.type == 'Carry').sum()} carries; median passes per fast attack {per.median():.0f}")
    print(f"Distinct passing pairs {len(links)}; 5+ passes: {(links >= 5).sum()}")
    print("Most repeated links:")
    print(links.head(6).to_string())


if __name__ == "__main__":
    main()
