"""
Part 1: Build an expected threat (xT) model from the whole 2015/16 Premier League.

Method (Karun Singh's xT, a standard approach):
  - Split the pitch into a 16 x 12 grid (each zone 7.5 x 6.67 yards).
  - For each zone, from every open-play action that starts there:
        shot share   s = shots / (shots + moves)
        move share   m = moves / (shots + moves)
        goal rate    g = goals / shots            (open-play shots, no penalties)
        transitions  T[z -> z'] = successful moves from z ending in z' / all moves from z
    Moves are open-play passes (completed or not) and carries. A failed pass counts as
    a move that loses the ball, so it lowers the value of risky zones.
  - Solve xT(z) = s*g + m * sum over z' of T[z -> z'] * xT(z'), by repeating until it
    stops changing. xT(z) = the chance that having the ball in zone z leads to a goal
    within the next few actions.
  - Every completed open-play pass or carry gets xT added = xT(end zone) - xT(start zone).
    Failed passes get 0 (they are already priced into the zone values).

Validation: does team xT track team xG and goals, match by match and over the season?

Output: results/xt_grid.csv, results/moves_xt.csv.gz, printed checks
"""

import numpy as np
import pandas as pd

from common import SET_PIECES, load_actions, load_matches

NX, NY = 16, 12


def zone(x, y, nx=NX, ny=NY):
    ix = np.clip((x / 120 * nx).astype(int), 0, nx - 1)
    iy = np.clip((y / 80 * ny).astype(int), 0, ny - 1)
    return ix * ny + iy


def build(A, nx=NX, ny=NY, tol=1e-6):
    passes = A[(A.type == "Pass") & ~A.pass_type.isin(SET_PIECES)].dropna(subset=["x"])
    carries = A[A.type == "Carry"].dropna(subset=["x", "end_x"])
    shots = A[(A.type == "Shot") & (A.pass_type == "Open Play")].dropna(subset=["x"])
    nz = nx * ny
    n_shot = np.bincount(zone(shots.x.values, shots.y.values, nx, ny), minlength=nz)
    n_goal = np.bincount(zone(shots.x.values, shots.y.values, nx, ny),
                         weights=(shots.outcome == "Goal").values, minlength=nz)
    moves = pd.concat([passes, carries])
    z0 = zone(moves.x.values, moves.y.values, nx, ny)
    n_move = np.bincount(z0, minlength=nz)
    ok = moves.outcome.isna().values & moves.end_x.notna().values
    z1 = zone(moves.end_x.values[ok], moves.end_y.values[ok], nx, ny)
    T = np.zeros((nz, nz))
    np.add.at(T, (z0[ok], z1), 1)
    T = T / np.maximum(n_move, 1)[:, None]
    tot = np.maximum(n_shot + n_move, 1)
    s, m = n_shot / tot, n_move / tot
    g = np.where(n_shot > 0, n_goal / np.maximum(n_shot, 1), 0)
    xt = np.zeros(nz)
    for it in range(200):
        new = s * g + m * (T @ xt)
        if np.abs(new - xt).max() < tol:
            break
        xt = new
    return new, it + 1


def value_moves(A, xt, nx=NX, ny=NY):
    mv = A[((A.type == "Pass") & ~A.pass_type.isin(SET_PIECES) & A.outcome.isna())
           | (A.type == "Carry")].dropna(subset=["x", "end_x"]).copy()
    mv["xt_start"] = xt[zone(mv.x.values, mv.y.values, nx, ny)]
    mv["xt_end"] = xt[zone(mv.end_x.values, mv.end_y.values, nx, ny)]
    mv["xt_added"] = mv.xt_end - mv.xt_start
    return mv


def main():
    A = load_actions()
    xt, iters = build(A)
    grid = xt.reshape(NX, NY)
    pd.DataFrame(grid).to_csv("results/xt_grid.csv", index=False)
    print(f"xT solved in {iters} iterations")
    print(f"zone values: min {xt.min():.4f}, median {np.median(xt):.4f}, max {xt.max():.3f}")
    print(f"own half average {grid[:NX // 2].mean():.4f}; final third average "
          f"{grid[int(NX * 2 / 3):].mean():.4f}; zone in front of goal "
          f"{grid[-1, NY // 2]:.3f}")

    mv = value_moves(A, xt)
    mv[["match_id", "idx", "period", "minute", "second", "t", "type", "possession",
        "play_pattern", "team", "player", "position", "x", "y", "end_x", "end_y",
        "pass_type", "height", "cross", "recipient", "length", "under_pressure",
        "xt_start", "xt_end", "xt_added"]].to_csv("results/moves_xt.csv.gz", index=False,
                                                  compression="gzip")
    print(f"{len(mv):,} completed open-play passes and carries valued; "
          f"total xT added {mv.xt_added.sum():.1f}, positive part {mv.xt_added.clip(0).sum():.1f}")

    # validation
    shots = A[(A.type == "Shot") & (A.pass_type != "Penalty")]
    npxg = shots.groupby(["match_id", "team"]).xg.sum()
    goals = A[((A.type == "Shot") & (A.outcome == "Goal") & (A.pass_type != "Penalty"))
              | (A.type == "Own Goal For")].groupby(["match_id", "team"]).size()
    tm = mv.groupby(["match_id", "team"]).xt_added.agg(
        net="sum", pos=lambda v: v.clip(lower=0).sum())
    tm["npxg"] = npxg.reindex(tm.index).fillna(0)
    tm["goals"] = goals.reindex(tm.index).fillna(0)
    print("\nValidation, 760 team-matches (correlation):")
    print(f"  xT added (net) vs np xG {tm.net.corr(tm.npxg):.2f}; vs non-penalty goals "
          f"{tm.net.corr(tm.goals):.2f}")
    print(f"  xT added (positive only) vs np xG {tm.pos.corr(tm.npxg):.2f}")
    se = tm.groupby("team").sum()
    print("Season totals, 20 teams (correlation):")
    print(f"  xT added vs np xG {se.net.corr(se.npxg):.2f}; vs non-penalty goals "
          f"{se.net.corr(se.goals):.2f}")
    print(f"  rank correlation xT vs np xG {se.net.rank().corr(se.npxg.rank()):.2f}")
    print("\nTeam season totals (sorted by xT added)")
    print(se.sort_values("net", ascending=False)[["net", "npxg", "goals"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
