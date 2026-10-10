"""State-level event study + randomisation inference (courts, Trends, protest).

  y_st = sum_k b_k * A_se * 1[t - T_e = k] + a_{stack x state} + g_{stack x week} + e_st
A_se = applicants per 1,000 youth in the exposed state (0 elsewhere).
Treatment varies across only a few states, so the reported inference is the
RANDOMISATION p-value: every shock gets false week-0 dates (same exposure), the
DiD estimate is recomputed, and p = share of |false| >= |real|.
Missing weeks: counts -> 0; trends_index -> dropped (not observed).
Usage: python 09_state_event_study.py <outcome> [--placebo] [n_reps]
Output: output/es_state_{tag}_{outcome}*.{csv,txt}, figures/es_state_{tag}_{outcome}.png, figures/ri_state_...
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import INT, OUT, FIG, REF_K, REF_MODE, EVENT_DATE, START_DATE, END_DATE, WINDOW_PRE, WINDOW_POST
from lib import event_study, plot_event_study, demean, ols_cluster, blog_style

args = [a for a in sys.argv[1:] if not a.startswith("--")]
y = args[0] if args else "n_orders"
reps = int(args[1]) if len(args) > 1 else 500
tag = "placebo" if "--placebo" in sys.argv else "main"
df = pd.read_csv(INT / f"stacked_state_{tag}.csv", dtype={"stack_id": str}, parse_dates=["week"])
if y not in df:
    raise SystemExit(f"{y} not in stacked_state_{tag}.csv; available: "
                     f"{[c for c in df.columns if c.startswith(('n_', 'trends', 'shutdown'))]}")
is_count = not y.startswith("trends")
df = df.assign(**{y: df[y].fillna(0)}) if is_count else df.dropna(subset=[y])

fit, tab, txt = event_study(df, y, ["fe_se", "fe_wk"], "state", REF_K, controls=(), sexes=("tot",), mode=REF_MODE)
txt = f"sample=state_{tag}  window -{WINDOW_PRE}..+{WINDOW_POST}  week 0 = {EVENT_DATE}  stacks: {', '.join(df.stack_id.unique())}\n" + txt

# ---- randomisation inference on the simple DiD ----
sw = pd.read_csv(INT / "state_week.csv", parse_dates=["week"]).set_index(["state", "week"])[y]
expo = df.drop_duplicates(["stack_id", "state"])[["stack_id", "state", "A_tot"]]
real_w = df[df.k == 0].drop_duplicates("stack_id").set_index("stack_id").week
treated = expo[expo.A_tot > 0].groupby("stack_id").state.apply(set)
lo_w = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())
hi_w = pd.Timestamp(END_DATE)
ks = np.arange(-WINDOW_PRE, WINDOW_POST + 1)


def did(weeks0):
    parts = []
    for e, T in weeks0.items():
        g = expo[expo.stack_id == e]
        g = g.loc[g.index.repeat(len(ks))].copy()
        g["k"] = np.tile(ks, len(g) // len(ks))
        g["week"] = T + pd.to_timedelta(g.k * 7, unit="D")
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p = p[(p.week >= lo_w) & (p.week <= hi_w)]
    p["yv"] = sw.reindex(pd.MultiIndex.from_arrays([p.state, p.week])).to_numpy()
    p["yv"] = p.yv.fillna(0) if is_count else p.yv
    p = p.dropna(subset=["yv"])
    x = (p.A_tot * (p.k >= 0)).to_numpy()
    Z = demean(np.column_stack([p.yv, x]), [p.stack_id + "|" + p.state, p.stack_id + "|" + p.week.dt.strftime("%Y%m%d")])
    return ols_cluster(Z[:, 0], Z[:, 1:], p.state, ["x"])["table"].iloc[0].coef


b_real = did(real_w.to_dict())
span = pd.Timedelta(weeks=WINDOW_PRE + WINDOW_POST)
all_w = pd.date_range(lo_w, hi_w, freq="7D")
obs_w = set(sw.dropna().index.get_level_values("week")) if not is_count else set(all_w)
elig = {}
for e, T in real_w.items():
    same = [real_w[o] for o in real_w.index if treated.get(o, set()) & treated.get(e, set())]
    elig[e] = [w for w in all_w
               if w - pd.Timedelta(weeks=WINDOW_PRE) >= lo_w and w + pd.Timedelta(weeks=WINDOW_POST) <= hi_w
               and w in obs_w and all(abs(w - t) > span for t in same)]
rng = np.random.default_rng(7)
import os
if os.environ.get("PM_RI", "shift") == "shift":
    # shift ALL shock dates by the same number of weeks (circular): keeps spacing and overlap
    n_w, sp = len(all_w), WINDOW_PRE + WINDOW_POST
    offs = [o for o in range(1, n_w) if sp < o < n_w - sp]
    fake = np.array([did({e: all_w[(int((T - lo_w).days // 7) + o) % n_w] for e, T in real_w.items()}) for o in offs])
    reps = len(fake)
else:
    fake = np.array([did({e: elig[e][rng.integers(len(elig[e]))] for e in real_w.index}) for _ in range(reps)])
p_ri = (np.sum(np.abs(fake) >= abs(b_real)) + 1) / (reps + 1)
txt += (f"\nRANDOMISATION: DiD b = {b_real:.6f}; p = {p_ri:.3f} ({reps} false-date draws; "
        f"b above {np.mean(fake < b_real):.1%} of false estimates)  <- use this p-value")
txt += "\nUnits: change in the outcome per state-week per 1 applicant per 1,000 youth."

stem = f"state_{tag}_{y}" + (f"_post{WINDOW_POST}" if (WINDOW_PRE, WINDOW_POST) != (8, 8) else "")
tab.to_csv(OUT / f"es_{stem}.csv", index=False)
(OUT / f"es_{stem}_tests.txt").write_text(txt)
pd.DataFrame({"b_fake": fake}).assign(b_real=b_real).to_csv(OUT / f"ri_{stem}.csv", index=False)
print(txt)
plot_event_study(tab, FIG / f"es_{stem}.png", f"Effect on {y} per state-week", REF_K,
                 xlabel=f"Weeks relative to {EVENT_DATE.replace('_', ' ')}",
                 title=f"95% CIs (state clusters, indicative only); randomisation p = {p_ri:.3f}")
blog_style()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(fake, bins=40, alpha=0.7)
ax.axvline(b_real, color="k", lw=2)
ax.set_xlabel(r"$\hat b$ with false exam dates")
ax.set_ylabel("Draws")
ax.set_title(f"{y}: real (line) vs {reps} false dates; p = {p_ri:.3f}", fontsize=9, loc="left")
fig.savefig(FIG / f"ri_{stem}.png")
plt.close(fig)
