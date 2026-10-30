"""
Part 3 chart: how Leicester got their chances.
  Left: open-play xG per match, built up by how quickly the first shot came after the
        possession started. One line per team; Leicester in blue.
  Right: four tiles on how they moved the ball (carries, long balls, where).

Input:  data/actions.csv.gz, results/possessions.csv.gz, results/how_profile.csv
Output: figures/part3_how.png, results/speed_curve.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from common import TEAM, load_actions
from part2_league import OPEN_PAT, cut_11v11

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREY, TRACK = "#0053A0", "#d3d2cc", "#ecebe7"
SECS = np.arange(0, 61, 1)


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def speed_curve():
    A0 = load_actions()
    A = cut_11v11(A0, A0)
    P = pd.read_csv("results/possessions.csv.gz")
    key = ["match_id", "period", "possession"]
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN_PAT)]
    sh = sh.merge(P[key + ["t0"]], on=key)
    first = sh.sort_values("t").groupby(key).agg(team=("team", "first"), t=("t", "first"),
                                                 t0=("t0", "first"))
    first["xg"] = sh.groupby(key).xg.sum()
    first["sec"] = first.t - first.t0
    n = A.groupby("team").match_id.nunique()
    C = pd.DataFrame({s: first[first.sec <= s].groupby("team").xg.sum() / n for s in SECS})
    C.to_csv("results/speed_curve.csv")
    return C, first.groupby("team").xg.sum() / n


def main():
    os.makedirs("figures", exist_ok=True)
    for f in font_manager.findSystemFonts():
        if "Inter" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Inter", "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    C, total = speed_curve()
    H = pd.read_csv("results/how_profile.csv", index_col=0)
    R = H.rank(ascending=False, method="min").astype(int)

    fig = plt.figure(figsize=(15, 7.6))
    ax = fig.add_axes([0.06, 0.14, 0.5, 0.6])
    for team, row in C.iterrows():
        if team != TEAM:
            ax.plot(SECS, row.values, color=GREY, lw=1.1, zorder=1)
    ax.plot(SECS, C.loc[TEAM].values, color=BLUE, lw=3.2, zorder=3)
    med = C.median()
    ax.plot(SECS, med.values, color=INK_2, lw=1.6, ls="--", zorder=2)
    ax.axvline(10, color=INK_3, lw=0.8, ls=":")
    v10, m10 = C.loc[TEAM, 10], med[10]
    ax.annotate(f"Within 10 seconds:\nLeicester {v10:.2f} xG a match,\nleague median {m10:.2f}",
                (10, v10), xytext=(22, 0.13), fontsize=10.5, color=BLUE,
                fontweight="semibold", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d8e3f2"), zorder=5)
    ax.text(60.5, C.loc[TEAM, 60], "Leicester", color=BLUE, fontsize=11.5, fontweight="bold",
            va="center")
    ax.text(60.5, med[60], "league median", color=INK_2, fontsize=10, va="center")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Seconds from the start of the possession to its first shot", fontsize=10.5,
                  color=INK_2)
    ax.set_ylabel("Open-play xG per match (cumulative)", fontsize=10.5, color=INK_2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#d8d7d2")
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(colors=INK_3, labelsize=9.5)
    ax.grid(axis="y", color=TRACK, lw=0.8)
    fig.text(0.04, 0.8, "Leicester's chances came early", fontsize=13.5, fontweight="bold")
    fig.text(0.04, 0.765, "Open-play xG, counted by how fast the first shot came. Grey = the "
             "other 19 teams.", fontsize=10, color=INK_2)

    tiles = [
        (f"{H.loc[TEAM, 'fast_share_of_op_xg']:.0%}",
         "of Leicester's open-play xG\ncame within 10 seconds",
         f"{ordinal(R.loc[TEAM, 'fast_share_of_op_xg'])} · median "
         f"{H.fast_share_of_op_xg.median():.0%}"),
        (f"{H.loc[TEAM, 'xt_per_carry_x1000']:.2f}",
         "threat per carry (xT x 1000):\nthe most dangerous running with\nthe ball",
         f"{ordinal(R.loc[TEAM, 'xt_per_carry_x1000'])} · median "
         f"{H.xt_per_carry_x1000.median():.2f}"),
        (f"{H.loc[TEAM, 'long_share']:.0%}",
         "of passes were long (35+ yards),\nbut only "
         f"{H.loc[TEAM, 'long_completion']:.0%} found a teammate",
         f"{ordinal(R.loc[TEAM, 'long_share'])} most · "
         f"{ordinal(R.loc[TEAM, 'long_completion'])} for completion"),
        (f"{H.loc[TEAM, 'xt_share_wide_end']:.0%}",
         "of their threat came from\nmoves ending out wide",
         f"{ordinal(R.loc[TEAM, 'xt_share_wide_end'])} · median "
         f"{H.xt_share_wide_end.median():.0%}"),
    ]
    for i, (big, a, b) in enumerate(tiles):
        x0 = 0.64 + (i % 2) * 0.175
        y0 = 0.45 - (i // 2) * 0.31
        fig.add_artist(FancyBboxPatch((x0, y0), 0.16, 0.27, transform=fig.transFigure,
                                      boxstyle="round,pad=0,rounding_size=0.012",
                                      facecolor="white", edgecolor="#e3e2dd", lw=1.2))
        fig.text(x0 + 0.012, y0 + 0.2, big, fontsize=26, fontweight="bold", color=BLUE)
        fig.text(x0 + 0.012, y0 + 0.17, a, fontsize=9.6, va="top", linespacing=1.3)
        fig.text(x0 + 0.012, y0 + 0.03, b, fontsize=8.6, color=INK_2, va="bottom",
                 wrap=True)
    fig.text(0.64, 0.8, "How they moved the ball", fontsize=13.5, fontweight="bold")
    fig.text(0.64, 0.765, "Leicester, with rank among 20 teams", fontsize=10, color=INK_2)

    fig.text(0.03, 0.965, "Leicester's edge was speed: the best chances in seconds",
             fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.92, "Premier League 2015/16, 11 v 11. Possessions not started by a "
             "corner, free kick or throw-in; penalties excluded.", fontsize=11, color=INK_2,
             va="top")
    fig.text(0.03, 0.02, "Data: StatsBomb open data. Seconds measured from the first action "
             "of the possession to its first shot. Leicester rank 1st for chances within 5, 10, "
             "15 and 20 seconds.", fontsize=8.5, color=INK_3)
    fig.savefig("figures/part3_how.png", dpi=200)
    print("Saved figures/part3_how.png")


if __name__ == "__main__":
    main()
