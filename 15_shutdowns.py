"""Exploratory: exam-day internet shutdowns as shocks to online exposure.

Only valid if exam shutdowns are separate from protest shutdowns. The
script checks that first: it tests whether exam shutdowns follow
recent protest (they should not).
  y_{s,t} = sum_{j=-2..4} b_j shutdown_exam_{s,t-j} + a_s + g_t + e      (TWFE, clusters = states)
Outcomes: trends_index (online) and n_exam (street), from state_week.csv.
Input: raw/shutdowns.csv (state, start_date, end_date, reason = exam|protest|other, source_url)
"""
import numpy as np
import pandas as pd
from config import INT, OUT
from lib import demean, ols_cluster

sw = pd.read_csv(INT / "state_week.csv", parse_dates=["week"])
if "shutdown_exam" not in sw:
    raise SystemExit("no shutdown data; add raw/shutdowns.csv and re-run 08_state_panel.py")
sw = sw.sort_values(["state", "week"])
for c in [c for c in sw.columns if c.startswith(("n_", "shutdown_"))]:
    sw[c] = sw[c].fillna(0)
sw["exam_sd"] = (sw.shutdown_exam > 0).astype(int)

# balance check: do exam shutdowns follow recent protest?
sw["protest_lag1"] = sw.groupby("state").n_events.shift(1)
d = sw.dropna(subset=["protest_lag1"])
Z = demean(np.column_stack([d.exam_sd, d.protest_lag1]), [d.state, d.week])
t = ols_cluster(Z[:, 0], Z[:, 1:], d.state, ["protest_lag1"])["table"]
print("[15] balance: exam shutdown on last week's protest count (want ~0):")
print(t.round(4).to_string(index=False))

lags = range(-2, 5)
for j in lags:
    sw[f"sd_{j}"] = sw.groupby("state").exam_sd.shift(j)
for y in [c for c in ["trends_index", "n_exam"] if c in sw]:
    d = sw.dropna(subset=[y] + [f"sd_{j}" for j in lags])
    X = d[[f"sd_{j}" for j in lags]].to_numpy()
    Z = demean(np.column_stack([d[y], X]), [d.state, d.week])
    t = ols_cluster(Z[:, 0], Z[:, 1:], d.state, [f"b_{j}" for j in lags])["table"]
    t.to_csv(OUT / f"shutdowns_{y}.csv", index=False)
    print(f"[15] {y} (j < 0 are leads; want ~0):")
    print(t[["term", "coef", "se", "p"]].round(4).to_string(index=False))
