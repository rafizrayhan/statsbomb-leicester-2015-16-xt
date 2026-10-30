"""
Part 1 chart.
  Left: the xT danger map learned from the 2015/16 Premier League.
  Right: each team's threat from passes and carries (xT) against the quality of its
         open-play chances (xG from possessions not started by a corner, free kick or
         throw-in). Leicester highlighted.

Input:  results/xt_grid.csv, results/moves_xt.csv.gz, data/actions.csv.gz
Output: figures/part1_xt_model.png, results/team_xt_vs_xg.csv
"""

import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch

from common import TEAM, load_actions

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREY, LINE = "#0053A0", "#c9c8c2", "#9a9994"
OPEN = ["Regular Play", "From Counter", "From Goal Kick", "From Keeper", "From Kick Off"]
SHORTN = {"Manchester City": "Man City", "Manchester United": "Man Utd",
          "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
          "West Bromwich Albion": "West Brom", "AFC Bournemouth": "Bournemouth",
          "Newcastle United": "Newcastle", "Crystal Palace": "Palace",
          "Leicester City": "Leicester", "Swansea City": "Swansea", "Stoke City": "Stoke",
          "Norwich City": "Norwich", "Aston Villa": "Aston Villa"}


def team_table():
    A = load_actions()
    mv = pd.read_csv("results/moves_xt.csv.gz")
    sh = A[(A.type == "Shot") & (A.pass_type == "Open Play") & A.play_pattern.isin(OPEN)]
    T = pd.DataFrame({"xt": mv.groupby("team").xt_added.sum(),
                      "open_play_xg": sh.groupby("team").xg.sum()})
    T["xg_per_xt"] = T.open_play_xg / T.xt
    T.to_csv("results/team_xt_vs_xg.csv")
    return T


def main():
    os.makedirs("figures", exist_ok=True)
    for f in font_manager.findSystemFonts():
        if "Inter" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Inter", "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    G = pd.read_csv("results/xt_grid.csv").values  # shape (16, 12): [x zone, y zone]
    T = team_table()

    fig = plt.figure(figsize=(15, 7.4))
    ax = fig.add_axes([0.03, 0.12, 0.5, 0.62])
    pitch = Pitch(pitch_type="statsbomb", line_color=LINE, pitch_color=SURFACE, linewidth=1,
                  line_zorder=3)
    pitch.draw(ax=ax)
    cmap = LinearSegmentedColormap.from_list("xt", ["#f3f6fb", "#b9cde8", "#5a8ac6",
                                                    "#0053A0", "#0b2a5c"])
    ax.imshow(np.sqrt(G.T), extent=(0, 120, 80, 0), cmap=cmap, zorder=1, aspect="auto",
              interpolation="nearest")
    for ix, iy, lab in [(15, 6, None), (12, 6, None), (8, 6, None), (3, 6, None)]:
        v = G[ix, iy]
        ax.text((ix + 0.5) * 7.5, (iy + 0.5) * 80 / 12, f"{v:.3f}" if v < 0.1 else f"{v:.2f}",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if v > 0.03 else INK, zorder=4,
                path_effects=[pe.withStroke(linewidth=2,
                                            foreground="#0b2a5c" if v > 0.03 else SURFACE)])
    ax.annotate("", xy=(112, -4), xytext=(70, -4), annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2))
    ax.text(68, -4, "attacking direction", ha="right", va="center", fontsize=10,
            color=INK_2)
    fig.text(0.03, 0.8, "How dangerous is each zone?", fontsize=13.5, fontweight="bold")
    fig.text(0.03, 0.765, "Chance that having the ball there leads to a goal within the "
             "next few actions", fontsize=10, color=INK_2)

    bx = fig.add_axes([0.61, 0.14, 0.36, 0.6])
    for team, r in T.iterrows():
        lei = team == TEAM
        bx.scatter(r.xt, r.open_play_xg, s=170 if lei else 60, color=BLUE if lei else GREY,
                   zorder=3, edgecolor="white", lw=1)
        dx, dy, ha = {"Manchester United": (0.6, 0.6, "left"),
                      "Southampton": (0.5, -0.9, "left")}.get(team, (0.5, 0.25, "left"))
        if lei:
            dy = 0.6
        bx.text(r.xt + dx, r.open_play_xg + dy, SHORTN.get(team, team),
                fontsize=11.5 if lei else 8.5, color=BLUE if lei else INK_3,
                fontweight="bold" if lei else "normal", zorder=4, ha=ha)
    k = T.open_play_xg.sum() / T.xt.sum()
    xs = np.array([28, 60])
    bx.plot(xs, k * xs, color=INK_3, ls="--", lw=1)
    bx.text(59.5, k * 59.5 - 1.8, "league average\nratio", fontsize=8.5, color=INK_3,
            ha="right", va="top")
    for sp in ["top", "right"]:
        bx.spines[sp].set_visible(False)
    bx.spines["left"].set_color("#d8d7d2")
    bx.spines["bottom"].set_color("#d8d7d2")
    bx.tick_params(colors=INK_3, labelsize=9.5)
    bx.set_xlabel("Threat added by passes and carries (xT, season)", fontsize=10.5,
                  color=INK_2)
    bx.set_ylabel("Open-play chances (xG, season)", fontsize=10.5, color=INK_2)
    fig.text(0.58, 0.8, "Moving the ball against creating chances", fontsize=13.5,
             fontweight="bold")
    lt = T.loc[TEAM]
    fig.text(0.58, 0.765, f"Leicester: {T.xt.rank(ascending=False)[TEAM]:.0f}th for xT, "
             f"{T.open_play_xg.rank(ascending=False)[TEAM]:.0f}nd for open-play xG",
             fontsize=10, color=BLUE, fontweight="semibold")

    fig.text(0.03, 0.965, "Leicester created far better chances than their passing and "
             "carrying suggests", fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.92, "An expected threat (xT) model built from all 380 Premier League "
             "matches of 2015/16.", fontsize=11, color=INK_2, va="top")
    fig.text(0.03, 0.02, "Data: StatsBomb open data. xT: 16 x 12 grid, open-play passes and "
             "carries, open-play shots (Karun Singh's method). Open-play xG excludes "
             "penalties and possessions started by a corner, free kick or throw-in.",
             fontsize=8.5, color=INK_3)
    fig.savefig("figures/part1_xt_model.png", dpi=200)
    print("Saved figures/part1_xt_model.png")
    print(T.sort_values("xg_per_xt", ascending=False).round(2).head(5).to_string())


if __name__ == "__main__":
    main()
