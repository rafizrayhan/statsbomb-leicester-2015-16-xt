"""
Part 4 chart: the players behind Leicester's fast attacks.
  Left:   who started the possessions that became fast chances (share of Leicester's
          open-play xG from shots within 10 seconds of the start).
  Middle: who moved the ball into danger (xT added per 90), with league rank by role.
  Right:  who finished fast attacks (fast open-play xG per 90), league top 6.

Input:  results/leicester_starters.csv, results/players.csv, results/fast_shooters.csv
Output: figures/part4_players.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from common import TEAM

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, LIGHT, GREY = "#0053A0", "#a9c3e4", "#cfcec8"
SHORT = {"Riyad Mahrez": "Mahrez", "Jamie Vardy": "Vardy", "Kasper Schmeichel": "Schmeichel",
         "Danny Drinkwater": "Drinkwater", "Christian Fuchs": "Fuchs",
         "N'Golo Kanté": "Kanté", "Robert Huth": "Huth", "Marc Albrighton": "Albrighton",
         "Jeffrey Schlupp": "Schlupp", "Shinji Okazaki": "Okazaki", "Wes Morgan": "Morgan",
         "Danny Simpson": "Simpson", "Christian Benteke Liolo": "Benteke",
         "Daniel Andre Sturridge": "Sturridge",
         "Roberto Firmino Barbosa de Oliveira": "Firmino",
         "Gerard Deulofeu Lázaro": "Deulofeu", "Demarai Gray": "Gray"}
ROLE_SHORT = {"AM/W": "wide/attacking", "FB": "full-backs", "FW": "forwards",
              "CM": "central mids", "GK": "goalkeepers", "CB": "centre-backs"}


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main():
    os.makedirs("figures", exist_ok=True)
    for f in font_manager.findSystemFonts():
        if "Inter" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "Inter", "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    S = pd.read_csv("results/leicester_starters.csv", index_col=0)
    P = pd.read_csv("results/players.csv")
    F = pd.read_csv("results/fast_shooters.csv", index_col=0)

    fig = plt.figure(figsize=(15.5, 7.8))
    # left: starters
    ax = fig.add_axes([0.1, 0.13, 0.2, 0.6])
    s = S[S.possessions >= 20].head(7)
    for i, (p, r) in enumerate(s.iterrows()):
        hi = p in ("Riyad Mahrez", "Jamie Vardy")
        ax.barh(i, r.share_of_fast_xg, color=BLUE if hi else LIGHT, height=0.66)
        ax.text(-0.006, i, SHORT.get(p, p), ha="right", va="center", fontsize=11,
                fontweight="bold" if hi else "normal")
        ax.text(r.share_of_fast_xg + 0.006, i, f"{r.share_of_fast_xg:.0%}", va="center",
                fontsize=10.5, color=BLUE if hi else INK_2,
                fontweight="bold" if hi else "normal")
        ax.text(0.265, i, f"{int(r.possessions)}", va="center", ha="right", fontsize=9.5,
                color=INK_3)
    ax.text(0.265, -0.8, "possessions\nstarted", fontsize=8.5, color=INK_3, ha="right",
            va="bottom")
    ax.set_xlim(0, 0.27)
    ax.set_ylim(len(s) - 0.4, -0.8)
    ax.axis("off")
    fig.text(0.03, 0.8, "Who started them", fontsize=13.5, fontweight="bold")
    fig.text(0.03, 0.765, "Share of Leicester's fast open-play xG", fontsize=10, color=INK_2)
    k = P[P.player == "N'Golo Kanté"].iloc[0]
    fig.text(0.03, 0.06, f"Kanté won the ball most in the league ({k.ball_wins_p90:.1f} per 90),\n"
             "but few of those possessions became fast chances.", fontsize=9.5, color=INK_2,
             linespacing=1.35)

    # middle: xT added per 90, Leicester players
    bx = fig.add_axes([0.45, 0.13, 0.17, 0.6])
    L = P[(P.team == TEAM) & (P.minutes >= 900) & (P.role != "GK")].sort_values(
        "xt_p90", ascending=False).head(8)
    for i, r in enumerate(L.itertuples()):
        hi = r.xt_p90_rank <= 6
        bx.barh(i, r.xt_p90, color=BLUE if hi else LIGHT, height=0.66)
        bx.text(-0.004, i, SHORT.get(r.player, r.player), ha="right", va="center",
                fontsize=11, fontweight="bold" if hi else "normal")
        bx.text(r.xt_p90 + 0.005, i, f"{r.xt_p90:.2f}", va="center", fontsize=10.5,
                color=BLUE if hi else INK_2, fontweight="bold" if hi else "normal")
        bx.text(0.36, i, f"{ordinal(r.xt_p90_rank)} of {int(r.role_n)} {ROLE_SHORT[r.role]}",
                va="center", fontsize=9, color=INK_2)
    bx.set_xlim(0, 0.32)
    bx.set_ylim(len(L) - 0.4, -0.8)
    bx.axis("off")
    fig.text(0.38, 0.8, "Who moved the ball into danger", fontsize=13.5, fontweight="bold")
    fig.text(0.38, 0.765, "Threat added per 90 (xT), rank in the league by role",
             fontsize=10, color=INK_2)

    # right: fast shooters
    cx = fig.add_axes([0.82, 0.13, 0.13, 0.6])
    f = F.head(6)
    for i, (p, r) in enumerate(f.iterrows()):
        lei = r.team == TEAM
        cx.barh(i, r.fast_xg_p90, color=BLUE if lei else GREY, height=0.66)
        cx.text(-0.004, i, f"{SHORT.get(p, p)}", ha="right", va="center", fontsize=11,
                fontweight="bold" if lei else "normal")
        cx.text(r.fast_xg_p90 + 0.004, i, f"{r.fast_xg_p90:.2f}", va="center", fontsize=10.5,
                color=BLUE if lei else INK_2, fontweight="bold" if lei else "normal")
    cx.set_xlim(0, 0.24)
    cx.set_ylim(len(f) - 0.4, -0.8)
    cx.axis("off")
    fig.text(0.735, 0.8, "Who finished them", fontsize=13.5, fontweight="bold")
    fig.text(0.735, 0.765, "Fast open-play xG per 90, league top 6", fontsize=10,
             color=INK_2)
    v = F.loc["Jamie Vardy"]
    fig.text(0.735, 0.06, f"Vardy: 1st of {len(F)} attackers for fast xG and for all\n"
             f"open-play xG per 90 ({v.op_xg_p90:.2f}).", fontsize=9.5, color=INK_2,
             linespacing=1.35)

    fig.text(0.03, 0.965, "Mahrez and Vardy started the fast attacks, and Vardy finished them",
             fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.92, "Fast = the first shot came within 10 seconds of the possession "
             "starting. Players with 900+ minutes, 11 v 11.", fontsize=11, color=INK_2,
             va="top")
    fig.text(0.03, 0.015, "Data: StatsBomb open data. Open play excludes penalties and "
             "possessions started by a corner, free kick or throw-in. A ball win in the 3 "
             "seconds before a possession counts as its start.", fontsize=8.5, color=INK_3)
    fig.savefig("figures/part4_players.png", dpi=200)
    print("Saved figures/part4_players.png")


if __name__ == "__main__":
    main()
