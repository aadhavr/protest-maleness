"""Average repeated Google Trends pulls and divide by the anchor term.

Input : raw/trends/pull*_*.csv (from collect/c08), raw/crosswalks/trends_states.csv (geo, state)
Output: interim/trends_state_week.csv  (state, week, term, ratio_mean, ratio_sd, n_pulls)
        interim/trends_index.csv       (state, week, trends_index = mean log ratio over terms)
ratio = term / anchor inside the SAME request, so it compares across states.
Weeks where the anchor is 0 are dropped (the ratio is undefined).
"""
import numpy as np
import pandas as pd
from config import RAW, INT, TRENDS_ANCHOR, TRENDS_GROUPS, TRENDS_PRIMARY
from lib import week_start

files = sorted((RAW / "trends").glob("pull*_*.csv"))
if not files:
    raise SystemExit("no Trends pulls; run collect/c08_trends_pull.py")
raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
if "isPartial" in raw:
    raw = raw[~raw.isPartial.astype(str).str.lower().eq("true")]
xw = pd.read_csv(RAW / "crosswalks" / "trends_states.csv")
terms = [c for c in raw.columns if c not in
         {"date", "isPartial", "state", "pull", "batch", "pulled_at", TRENDS_ANCHOR}]
long = raw.melt(id_vars=["date", "state", "pull", "batch", TRENDS_ANCHOR], value_vars=terms,
                var_name="term", value_name="value").dropna(subset=["value"])
long = long[long[TRENDS_ANCHOR] > 0]
long["ratio"] = long.value / long[TRENDS_ANCHOR]
long["week"] = week_start(long.date)
agg = (long.groupby(["state", "week", "term"]).ratio
       .agg(ratio_mean="mean", ratio_sd="std", n_pulls="count").reset_index())
agg = agg.rename(columns={"state": "geo"}).merge(xw, on="geo", how="left")
if agg.state.isna().any():
    print(f"[a07] geo codes missing from trends_states.csv: {agg.geo[agg.state.isna()].unique().tolist()}")
# several Trends regions can map to one 2011 state (Telangana + Andhra Pradesh): average them
agg = (agg.dropna(subset=["state"]).groupby(["state", "week", "term"])
       .agg(ratio_mean=("ratio_mean", "mean"), ratio_sd=("ratio_sd", "mean"), n_pulls=("n_pulls", "sum"))
       .reset_index())
agg.to_csv(INT / "trends_state_week.csv", index=False)

# indices: mean over terms of log(ratio + small constant), per state-week.
# trends_index = the original primary groups; trends_<group> = each group separately.
eps = agg.ratio_mean[agg.ratio_mean > 0].min() / 2
agg["lr"] = np.log(agg.ratio_mean + eps)
grp = {t_: g for g, ts in TRENDS_GROUPS.items() for t_ in ts}
agg["group"] = agg.term.map(grp)
prim = agg[agg.group.isin(TRENDS_PRIMARY)].groupby(["state", "week"]).lr.mean().rename("trends_index")
by_g = agg.dropna(subset=["group"]).pivot_table(index=["state", "week"], columns="group", values="lr", aggfunc="mean")
by_g.columns = [f"trends_{c}" for c in by_g.columns]
idx = pd.concat([prim, by_g], axis=1).reset_index()
idx.to_csv(INT / "trends_index.csv", index=False)
print(f"[a07] indices: {[c for c in idx.columns if c.startswith('trends')]}")
cv = (agg.ratio_sd / agg.ratio_mean).replace(np.inf, np.nan).median()
print(f"[a07] {agg.term.nunique()} terms x {agg.state.nunique()} states; "
      f"median across-pull CV = {cv:.2f} (high values mean noisy series)")
