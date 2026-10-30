"""
Shared data handling for the Leicester ball-progression project.

Premier League 2015/16, all 380 matches, from StatsBomb open data.
Each match's events are downloaded once and reduced to a compact table of the
fields this project needs (data/actions.csv.gz). Raw files are not kept.
"""

import json
import os
import time
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMP, SEASON = 2, 27            # Premier League, 2015/2016
TEAM = "Leicester City"
ACTIONS = "data/actions.csv.gz"
LINEUPS = "data/lineups.csv"
MATCHES = "data/matches.csv"
RED = {"Red Card", "Second Yellow"}
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
KEEP = {"Pass", "Carry", "Shot", "Dribble", "Ball Recovery", "Interception", "Duel",
        "Own Goal For", "Substitution", "Foul Committed", "Bad Behaviour", "Clearance",
        "Block", "Miscontrol", "Dispossessed", "Ball Receipt*", "Pressure",
        "Goal Keeper"}


def get_json(url, retries=6):
    """Download JSON, waiting and retrying if GitHub rate-limits (HTTP 429)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


def ts(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def card(e):
    for k in ("foul_committed", "bad_behaviour"):
        c = e.get(k, {}).get("card", {}).get("name")
        if c:
            return c
    return None


def compact(match_id):
    """Reduce one match's events to the fields used in this project."""
    ev = get_json(f"{BASE}/events/{match_id}.json")
    rows, lineup = [], []
    for i, e in enumerate(ev):
        t = e["type"]["name"]
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                lineup.append({"match_id": match_id, "team": e["team"]["name"],
                               "player": p["player"]["name"],
                               "position": p["position"]["name"]})
            continue
        if t not in KEEP:
            continue
        loc = e.get("location") or [None, None]
        end = [None, None]
        r = {"match_id": match_id, "idx": i, "period": e["period"], "minute": e["minute"],
             "second": e["second"], "t": ts(e), "type": t, "possession": e["possession"],
             "possession_team": e["possession_team"]["name"],
             "play_pattern": e["play_pattern"]["name"], "team": e["team"]["name"],
             "player": e.get("player", {}).get("name"),
             "position": e.get("position", {}).get("name"),
             "x": loc[0], "y": loc[1], "under_pressure": bool(e.get("under_pressure"))}
        if t == "Pass":
            p = e["pass"]
            end = p["end_location"]
            r.update(pass_type=p.get("type", {}).get("name"),
                     outcome=p.get("outcome", {}).get("name"),
                     height=p.get("height", {}).get("name"),
                     cross=bool(p.get("cross")), recipient=p.get("recipient", {}).get("name"),
                     length=p.get("length"))
        elif t == "Carry":
            end = e["carry"]["end_location"]
        elif t == "Shot":
            s = e["shot"]
            end = s.get("end_location", [None, None])[:2]
            r.update(xg=s.get("statsbomb_xg"), outcome=s["outcome"]["name"],
                     pass_type=s.get("type", {}).get("name"))
        elif t == "Dribble":
            r.update(outcome=e["dribble"]["outcome"]["name"])
        elif t == "Duel":
            r.update(pass_type=e["duel"].get("type", {}).get("name"),
                     outcome=e["duel"].get("outcome", {}).get("name"))
        elif t == "Interception":
            r.update(outcome=e["interception"]["outcome"]["name"])
        elif t == "Ball Recovery":
            r.update(outcome="Failure" if e.get("ball_recovery", {}).get("recovery_failure")
                     else "Success")
        elif t == "Ball Receipt*":
            r.update(outcome=e.get("ball_receipt", {}).get("outcome", {}).get("name"))
        elif t == "Substitution":
            r.update(recipient=e["substitution"]["replacement"]["name"])
        elif t == "Goal Keeper":
            r.update(pass_type=e["goalkeeper"].get("type", {}).get("name"),
                     outcome=e["goalkeeper"].get("outcome", {}).get("name"))
        c = card(e)
        if c:
            r["card"] = c
        r["end_x"], r["end_y"] = (end + [None, None])[:2]
        rows.append(r)
    return rows, lineup


def clean_names(df, cols=("player", "recipient")):
    """The data spells some names two ways (for example N''Golo and N'Golo Kanté)."""
    for c in cols:
        if c in df:
            df[c] = df[c].str.replace("''", "'", regex=False)
    return df


def load_actions():
    return clean_names(pd.read_csv(ACTIONS, low_memory=False))


def load_matches():
    return pd.read_csv(MATCHES)


def clock(df):
    """Match clock in minutes (2nd-half minutes already start at 45)."""
    return df.minute + df.second / 60
