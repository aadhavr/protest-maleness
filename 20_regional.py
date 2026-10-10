"""Contrast cases: Bangladesh (quota ruling) and Nepal (youth protests), one dated shock each.

Weekly national series from ACLED: all events, student events, and the share of reports
that mention women. Statistic: mean(weeks 0..+8) - mean(weeks -8..-1).
Randomisation: the same statistic at every other week whose window fits the data
(a single-shock case study: the p-value says how unusual this window is, nothing more).
Shocks (verify the dates): Bangladesh 2024-06-05 (High Court restores the quotas);
Nepal 2025-09-08 (start of the youth protests; only 3 post weeks exist in the data).
Output: output/regional_case.csv, figures/regional_case.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW, OUT, FIG
from lib import week_start, blog_style

WOMEN = r"\b(?:women|woman|female|girls?|mahila|ladies|nari)\b"
CASES = {"bangladesh": "2024-06-05", "nepal": "2025-09-08"}
PRE, POST = 8, 8
rows, series = [], {}
for cty, d0 in CASES.items():
    files = sorted((RAW / f"acled_{cty}").glob("*.csv"))
    if not files:
        print(f"[20] no ACLED data for {cty}")
        continue
    r = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True).drop_duplicates("event_id_cnty")
    r["week"] = week_start(r.event_date)
    act = r[[c for c in ["assoc_actor_1", "assoc_actor_2", "actor1", "actor2"] if c in r]].fillna("").agg(" ".join, axis=1)
    r["students"] = act.str.contains("Students", regex=True)
    r["women"] = r.notes.fillna("").str.contains(WOMEN, case=False, regex=True)
    w = r.groupby("week").agg(events=("event_id_cnty", "size"), students=("students", "sum"),
                              women_n=("women", "sum"))
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="7D")).fillna(0)
    w["women_share"] = w.women_n / w.events.where(w.events > 0)
    series[cty] = w
    T = week_start(pd.Series([pd.Timestamp(d0)])).iloc[0]

    def stat(t, col):
        pre_m = (w.index >= t - pd.Timedelta(weeks=PRE)) & (w.index < t)
        post_m = (w.index >= t) & (w.index <= t + pd.Timedelta(weeks=POST))
        if pre_m.sum() < 4 or post_m.sum() < 3:
            return np.nan, 0
        if col == "women_share":       # event-weighted: share of ALL events in the window
            a, b = w[pre_m], w[post_m]
            if a.events.sum() == 0 or b.events.sum() == 0:
                return np.nan, 0
            return b.women_n.sum() / b.events.sum() - a.women_n.sum() / a.events.sum(), int(post_m.sum())
        return w.loc[post_m, col].mean() - w.loc[pre_m, col].mean(), int(post_m.sum())

    for col in ["events", "students", "women_share"]:
        real, npost = stat(T, col)
        others = [stat(t, col)[0] for t in w.index if abs((t - T).days) > 7 * (PRE + POST)]
        others = np.array([x for x in others if not np.isnan(x)])
        p = (np.sum(np.abs(others) >= abs(real)) + 1) / (len(others) + 1) if len(others) else np.nan
        rows.append({"country": cty, "outcome": col, "post_minus_pre": real, "post_weeks": npost,
                     "p_vs_other_windows": p, "n_windows": len(others)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "regional_case.csv", index=False)
print(res.round(4).to_string(index=False))

if series:
    blog_style()
    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 3.6), squeeze=False)
    for ax, (cty, w) in zip(axes.flat, series.items()):
        T = week_start(pd.Series([pd.Timestamp(CASES[cty])])).iloc[0]
        s = w[(w.index >= T - pd.Timedelta(weeks=16)) & (w.index <= T + pd.Timedelta(weeks=16))]
        ax.plot(s.index, s.events, label="All events")
        ax.plot(s.index, s.students, label="Student events")
        ax2 = ax.twinx()
        ax2.plot(s.index, s.women_share, color="grey", ls=":", label="Share mentioning women")
        ax.axvline(T, color="k", lw=1)
        ax.set_title(cty.title(), loc="left", fontsize=10)
        ax.set_ylabel("Events per week")
        ax2.set_ylabel("Share mentioning women")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.savefig(FIG / "regional_case.png")
