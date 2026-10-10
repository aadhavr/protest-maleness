"""Validate the hand-built exam-shock panel and add the shock week.

Input : raw/exam_shocks.csv   (see templates/exam_shocks.csv)
Output: interim/exam_shocks.csv
Every row must carry a source_url. The script stops if one is missing.
"""
import pandas as pd
from config import RAW, INT, MAIN_SHOCKS, PLACEBO_SHOCKS, OTHER_SHOCKS, EVENT_DATE
from lib import week_start, require_cols

COLS = ["exam_id", "exam_name", "conducting_body", "level", "state",
        "exam_date", "shock_date", "shock_type", "source_url"]

df = pd.read_csv(RAW / "exam_shocks.csv", dtype=str)
require_cols(df, COLS, "exam_shocks.csv")
df = df.dropna(how="all")

problems = []
if df.exam_id.duplicated().any():
    problems.append(f"duplicate exam_id: {df.exam_id[df.exam_id.duplicated()].tolist()}")
if df.source_url.isna().any():
    problems.append(f"no source_url: {df.exam_id[df.source_url.isna()].tolist()}")
bad_type = ~df.shock_type.isin(MAIN_SHOCKS + PLACEBO_SHOCKS + OTHER_SHOCKS)
if bad_type.any():
    problems.append(f"unknown shock_type: {df.loc[bad_type, 'shock_type'].unique().tolist()}")
bad_level = ~df.level.isin(["national", "state"])
if bad_level.any():
    problems.append(f"level must be national/state: {df.exam_id[bad_level].tolist()}")
state_missing = (df.level == "state") & df.state.isna()
if state_missing.any():
    problems.append(f"state exams with no state: {df.exam_id[state_missing].tolist()}")

for c in ["exam_date", "shock_date", "leak_reported_date"]:
    if c in df:
        parsed = pd.to_datetime(df[c], errors="coerce")
        bad = df[c].notna() & parsed.isna()
        if bad.any():
            problems.append(f"unparseable {c}: {df.exam_id[bad].tolist()}")
        df[c] = parsed

if problems:
    raise SystemExit("exam_shocks.csv failed checks:\n  " + "\n  ".join(problems))

if EVENT_DATE not in df:
    raise SystemExit(f"EVENT_DATE column {EVENT_DATE} not in exam_shocks.csv")
miss = df[EVENT_DATE].isna()
if miss.any():
    print(f"[01] {miss.sum()} shocks have no {EVENT_DATE}; they use shock_date: {df.exam_id[miss].tolist()}")
df["event_date_used"] = df[EVENT_DATE].fillna(df.shock_date)
import os
excl = [x.strip() for x in os.environ.get("PM_EXCLUDE", "").split(",") if x.strip()]
if excl:
    print(f"[01] excluded for this run (PM_EXCLUDE): {excl}")
    df = df[~df.exam_id.isin(excl)].copy()
nodate = df.event_date_used.isna()
if nodate.any():
    print(f"[01] no usable date, left out of this run: {df.exam_id[nodate].tolist()}")
    df = df[~nodate].copy()
df["shock_week"] = week_start(df.event_date_used)
print(f"[01] week 0 = week of {EVENT_DATE}")
df["is_main"] = df.shock_type.isin(MAIN_SHOCKS)

# Shocks in the same week share a window; flag them so you can decide
# whether to merge them into one event.
same_week = df[df.is_main].groupby("shock_week").exam_id.transform("count") > 1
df["shares_week"] = False
df.loc[same_week[same_week].index, "shares_week"] = True

df.sort_values("shock_date").to_csv(INT / "exam_shocks.csv", index=False)
print(df.groupby(["shock_type", "level"]).size().rename("n").to_string())
print(f"[01] wrote {len(df)} shocks -> {INT/'exam_shocks.csv'}")
if df.shares_week.any():
    print("[01] note: some main shocks share a week:",
          df.loc[df.shares_week, "exam_id"].tolist())
