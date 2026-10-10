"""Descriptive figure: weekly exam protests in the exposed state vs the rest
of India, around each state exam. Rates per 10 million residents aged 15-29.
Lines: exam date (solid) and cancellation announcement (dashed).
Usage: python 05c_descriptive.py [outcome] [weeks_each_side]
Output: output/figures/descriptive_{outcome}.png, output/descriptive_{outcome}.csv
"""
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW, INT, OUT, FIG, START_DATE, END_DATE, MAIN_SHOCKS
from lib import week_start, blog_style

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
sh = pd.read_csv(RAW / "exam_shocks.csv", parse_dates=["exam_date", "shock_date"])
sh = sh[sh.shock_type.isin(MAIN_SHOCKS) & (sh.level == "state")]
ex = pd.read_csv(INT / "exposure.csv", dtype={"exam_id": str})
sh = sh[sh.exam_id.isin(ex.exam_id)]
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
pr = pr.merge(dist[["district_id", "state"]], on="district_id")
pop = dist.groupby("state").pop_15_29.sum()

rows = []
for s in sh.itertuples():
    T0 = week_start(pd.Series([s.exam_date if pd.notna(s.exam_date) else s.shock_date])).iloc[0]
    wks = pd.date_range(T0 - pd.Timedelta(weeks=W), T0 + pd.Timedelta(weeks=W), freq="7D")
    wks = wks[(wks >= pd.Timestamp(START_DATE) - pd.Timedelta(days=6)) & (wks <= pd.Timestamp(END_DATE))]
    sub = pr[pr.week.isin(wks)]
    inside = sub[sub.state == s.state].groupby("week")[y].sum().reindex(wks, fill_value=0)
    rest = sub[sub.state != s.state].groupby("week")[y].sum().reindex(wks, fill_value=0)
    k = ((wks - T0).days // 7).astype(int)
    rows.append(pd.DataFrame({"exam_id": s.exam_id, "state": s.state, "week": wks, "k": k,
                              "exposed": inside.values / pop[s.state] * 1e7,
                              "rest": rest.values / (pop.sum() - pop[s.state]) * 1e7,
                              "cancel_k": ((week_start(pd.Series([s.shock_date])).iloc[0] - T0).days // 7)}))
df = pd.concat(rows, ignore_index=True)
df.to_csv(OUT / f"descriptive_{y}.csv", index=False)

blog_style()
n = df.exam_id.nunique()
cols = min(n, 2)
fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(6 * cols, 3.2 * math.ceil(n / cols)), squeeze=False)
for ax, (e, d) in zip(axes.flat, df.groupby("exam_id", sort=False)):
    ax.plot(d.k, d.exposed, marker="o", ms=3, label=f"{d.state.iloc[0]}")
    ax.plot(d.k, d.rest, marker="o", ms=3, label="Rest of India")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(d.cancel_k.iloc[0], color="k", lw=1, ls="--")
    ax.set_title(e, fontsize=10, loc="left")
    ax.set_xlabel("Weeks from exam (solid) ; dashed = cancellation")
    ax.set_ylabel("Exam protests per 10M youth")
    ax.legend(frameon=False, fontsize=8)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / f"descriptive_{y}.png")
print(f"[05c] {n} state shocks -> {FIG / f'descriptive_{y}.png'}")
