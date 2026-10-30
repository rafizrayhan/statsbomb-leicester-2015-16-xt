"""
Part 2 chart: Leicester against the other 19 teams, one row per measure.
Each dot is a team (per match, 11 v 11). Leicester in blue with its rank.

Input:  results/team_profile.csv
Output: figures/part2_league.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from common import TEAM

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREY, TRACK = "#0053A0", "#c9c8c2", "#ecebe7"

ROWS = [
    ("WITH THE BALL", None, None, False),
    ("Share of the passing", "pass_share", "{:.0%}", False),
    ("Threat added per match (xT)", "xt_pm", "{:.2f}", False),
    ("Threat per pass or carry (xT x 1000)", "xt_per_action_x1000", "{:.2f}", False),
    ("Open-play chances per match (xG)", "op_xg_pm", "{:.2f}", False),
    ("Chance quality per possession (xG)", "op_xg_per_possession", "{:.3f}", False),
    ("WHERE POSSESSIONS STARTED", None, None, False),
    ("Share starting in the opponent half", "start_opp_half", "{:.0%}", False),
    ("Starting danger per match (xT of start zone)", "start_value_pm", "{:.2f}", False),
    ("WITHOUT THE BALL", None, None, False),
    ("Threat conceded per match (xT, right = least)", "conceded_xt_pm", "{:.2f}", True),
    ("Chances conceded (xG, right = least)", "conceded_op_xg_pm", "{:.2f}", True),
]


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main():
    os.makedirs("figures", exist_ok=True)
    for f in font_manager.findSystemFonts():
        if "Inter" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Inter", "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    T = pd.read_csv("results/team_profile.csv", index_col=0)
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    n = len(ROWS)
    ax.set_xlim(-0.75, 1.18)
    ax.set_ylim(n - 0.3, -0.7)
    ax.axis("off")
    for i, (lab, key, fmt, low_good) in enumerate(ROWS):
        if key is None:
            ax.text(-0.75, i + 0.15, lab, fontsize=10.5, fontweight="bold", color=INK_3)
            continue
        v = T[key]
        lo, hi = v.min(), v.max()
        pos = (v - lo) / (hi - lo)
        if low_good:
            pos = 1 - pos
        ax.text(-0.75, i, lab, fontsize=12, va="center")
        ax.hlines(i, 0, 1, color=TRACK, lw=6, zorder=1)
        ax.scatter(pos.drop(TEAM), [i] * (len(v) - 1), s=55, color=GREY, zorder=2,
                   edgecolor="white", lw=0.8)
        rank = int((v.rank(ascending=low_good, method="min"))[TEAM])
        ax.scatter(pos[TEAM], i, s=190, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
        ax.text(pos[TEAM], i - 0.38, fmt.format(v[TEAM]), ha="center", fontsize=10,
                color=BLUE, fontweight="bold")
        ax.text(1.05, i, ordinal(rank), fontsize=13, fontweight="bold", va="center",
                color=BLUE if rank <= 5 else (INK if rank <= 15 else INK_3))
    ax.text(0, -0.55, "lowest", fontsize=9.5, color=INK_3, ha="left")
    ax.text(1, -0.55, "highest", fontsize=9.5, color=INK_3, ha="right")
    ax.text(1.05, -0.55, "Leicester\nrank of 20", fontsize=9.5, color=INK_3, va="center")
    fig.text(0.03, 0.965, "Less of the ball, more purpose with it", fontsize=20,
             fontweight="bold", va="top")
    fig.text(0.03, 0.922, "Leicester against the other 19 Premier League teams, 2015/16, per "
             "match. Each dot is a team; Leicester in blue.", fontsize=11, color=INK_2,
             va="top")
    fig.text(0.03, 0.015, "Data: StatsBomb open data. 11 v 11 only (cut at the first red card). "
             "Open-play xG excludes penalties and possessions started by a corner, free kick "
             "or throw-in.\nShare of the passing: open-play passes, completed or not. Ranks: 1st = "
             "highest, except conceded rows where 1st = least conceded.", fontsize=8.5, color=INK_3)
    fig.savefig("figures/part2_league.png", dpi=200)
    print("Saved figures/part2_league.png")


if __name__ == "__main__":
    main()
