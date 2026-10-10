"""Randomisation inference for the main estimate (few treated states).

Statistic: b in  y = b * (A_tot x post) + FE(stack x district) + FE(stack x week),
on stacks built the same way as 04 (state exams + zero-exposure control districts).
Null distribution: each shock keeps its exposure pattern but gets a random
false week 0, drawn from weeks whose full window lies inside the protest data
and does not overlap a real shock window in the same state.
p = share of |b_fake| >= |b_real|. Valid with few treated states, unlike CRV1.
Usage: python 05b_randomization.py [outcome] [n_reps]
Output: output/ri_{outcome}.csv, output/figures/ri_{outcome}.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (RAW, INT, OUT, FIG, START_DATE, END_DATE, WINDOW_PRE, WINDOW_POST,
                    MAIN_SHOCKS, STATE_EXAM_CONTROLS, EVENT_DATE)
from lib import demean, ols_cluster, blog_style

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rng = np.random.default_rng(20241)

sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"], dtype={"exam_id": str})
sh = sh[sh.shock_type.isin(MAIN_SHOCKS)]
ex = pd.read_csv(INT / "exposure.csv", dtype={"exam_id": str, "district_id": str})
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})[["district_id", "state"]]
if y not in pr:
    raise SystemExit(f"{y} not in the protest panel")
yv = pr.set_index(["district_id", "week"])[y]

# exposure per shock, with zero-exposure controls for state exams
groups = {}
for s in sh.itertuples():
    d = ex[ex.exam_id == s.exam_id][["district_id", "state", "A_tot"]]
    if s.level == "state":
        d = d[d.state == s.state]
        if STATE_EXAM_CONTROLS and len(d):
            d = pd.concat([d, dist[dist.state != s.state].assign(A_tot=0.0)], ignore_index=True)
    if len(d) and d.A_tot.std() > 1e-9 * max(d.A_tot.abs().max(), 1e-12):
        groups[s.exam_id] = (d.reset_index(drop=True), s.shock_week, s.state if s.level == "state" else None)
if not groups:
    raise SystemExit("no shock with exposure variation")
print(f"[05b] shocks used: {list(groups)}")

lo_w = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())
all_weeks = pd.date_range(lo_w, pd.Timestamp(END_DATE), freq="7D")
ks = np.arange(-WINDOW_PRE, WINDOW_POST + 1)


def build(weeks0):
    parts = []
    for e, (d, _, _) in groups.items():
        T = weeks0[e]
        g = d.loc[d.index.repeat(len(ks))].copy()
        g["k"] = np.tile(ks, len(d))
        g["week"] = T + pd.to_timedelta(g.k * 7, unit="D")
        g["stack_id"] = e
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p = p[(p.week >= lo_w) & (p.week <= pd.Timestamp(END_DATE))]
    p["y"] = yv.reindex(pd.MultiIndex.from_arrays([p.district_id, p.week])).fillna(0).to_numpy()
    return p


def stat(p):
    x = (p.A_tot * (p.k >= 0)).to_numpy()
    Z = demean(np.column_stack([p.y, x]), [p.stack_id + "|" + p.district_id,
                                           p.stack_id + "|" + p.week.dt.strftime("%Y%m%d")])
    return ols_cluster(Z[:, 0], Z[:, 1:], p.state, ["x"])["table"].iloc[0].coef


real = {e: v[1] for e, v in groups.items()}
b_real = stat(build(real))

# eligible false weeks per shock
span = pd.Timedelta(weeks=WINDOW_PRE + WINDOW_POST)
elig = {}
for e, (d, T, st) in groups.items():
    same = [v[1] for v in groups.values() if v[2] == st] if st else [v[1] for v in groups.values()]
    ok = [w for w in all_weeks
          if w - pd.Timedelta(weeks=WINDOW_PRE) >= lo_w and w + pd.Timedelta(weeks=WINDOW_POST) <= pd.Timestamp(END_DATE)
          and all(abs(w - t) > span for t in same)]
    elig[e] = ok
    print(f"[05b] {e}: {len(ok)} eligible false weeks")

import os
MODE = os.environ.get("PM_RI", "shift")
fake = []
if MODE == "shift":
    # shift ALL shock dates by the same number of weeks (circular inside the data period):
    # keeps the real spacing and overlap between shocks
    n_w = len(all_weeks)
    sp = WINDOW_PRE + WINDOW_POST
    shift_rows = []
    for o in [o for o in range(1, n_w) if sp < o < n_w - sp]:
        wk = {e: all_weeks[(int((real[e] - lo_w).days // 7) + o) % n_w] for e in groups}
        fake.append(stat(build(wk)))
        shift_rows.append({"offset_weeks": o, "b_fake": fake[-1],
                           **{f"wk0_{e}": wk[e].date() for e in groups}})
else:
    for r in range(reps):
        wk = {e: elig[e][rng.integers(len(elig[e]))] for e in groups}
        fake.append(stat(build(wk)))
reps = len(fake)
fake = np.array(fake)
p_ri = (np.sum(np.abs(fake) >= abs(b_real)) + 1) / (reps + 1)
pd.DataFrame({"b_fake": fake}).assign(b_real=b_real).to_csv(OUT / f"ri_{y}.csv", index=False)
if MODE == "shift":
    sr = pd.DataFrame(shift_rows).assign(b_real=b_real)
    sr["abs_b"] = sr.b_fake.abs()
    sr.sort_values("abs_b", ascending=False).to_csv(OUT / f"ri_shifts_{y}.csv", index=False)
    top = sr.sort_values("abs_b", ascending=False).head(8)
    print(f"[05b] the {len(top)} largest false-date estimates (are there unlisted leaks in these windows?):")
    print(top.drop(columns=["abs_b", "b_real"]).to_string(index=False))
print(f"[05b] week 0 = {EVENT_DATE}; b_real = {b_real:.6f}; randomisation p = {p_ri:.3f} ({reps} false-date draws)")
print(f"[05b] b_real is above {np.mean(fake < b_real):.1%} of false-date estimates")

blog_style()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(fake, bins=40, alpha=0.7)
ax.axvline(b_real, color="k", lw=2)
ax.set_xlabel(r"$\hat b$ with false exam dates")
ax.set_ylabel("Draws")
ax.set_title(f"Real estimate (line) vs {reps} false dates; p = {p_ri:.3f}", fontsize=9, loc="left")
fig.savefig(FIG / f"ri_{y}.png")
