"""Translate the main estimate into plain terms, for each state shock.

b (from the same stacked DiD as 05, re-estimated here) is extra events per district-week
per applicant per 1,000 youth. For the exposed districts of each shock:
  extra events per week in the state  = b * sum_d A_d
  extra events over weeks 0..+POST    = that * (POST + 1)
  baseline per week                   = mean weekly events in the exposed districts, weeks -PRE..-1
  % increase                          = extra per week / baseline per week
Usage: python 24_effect_size.py [outcome]      (default n_exam_strict_nonparty)
Output: output/effect_size_{outcome}.csv
"""
import sys
import numpy as np
import pandas as pd
from config import INT, OUT, WINDOW_POST
from lib import demean, ols_cluster

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict_nonparty"
df = pd.read_csv(INT / "stacked_main.csv", dtype={"district_id": str, "stack_id": str})
if y not in df:
    raise SystemExit(f"{y} not in stacked_main.csv")
X = np.column_stack([df.A_tot * df.post, df.other_post_exp])
Z = demean(np.column_stack([df[y], X]), [df.fe_de, df.fe_stw])
t = ols_cluster(Z[:, 0], Z[:, 1:], df.state, ["A_x_post", "other"])["table"].iloc[0]
b, lo, hi = t.coef, t.ci_lo, t.ci_hi
rows = []
flat = df.groupby("stack_id").apply(lambda d: d.loc[d.A_tot > 0, "state"].nunique() > 1 and (d.loc[d.A_tot > 0].A_tot / d.loc[d.A_tot > 0].A_tot.mean()).std() < 1e-6)
for e, d in df[(df.A_tot > 0) & ~df.stack_id.isin(flat[flat].index)].groupby("stack_id"):
    A = d.drop_duplicates("district_id").A_tot.sum()
    pre = d[d.k < 0].groupby("k")[y].sum().mean()
    per_week = b * A
    rows.append({"exam_id": e, "state": d.state.iloc[0] if d.state.nunique() == 1 else "several",
                 "districts": d.district_id.nunique(), "applicants_per_1000_youth_mean": d.A_tot.mean(),
                 "extra_per_week": per_week, "extra_per_week_ci": f"{lo * A:.2f} to {hi * A:.2f}",
                 f"extra_weeks_0_{WINDOW_POST}": per_week * (WINDOW_POST + 1),
                 "baseline_per_week": pre,
                 "pct_increase": 100 * per_week / pre if pre > 0 else np.nan})
r = pd.DataFrame(rows).sort_values("extra_per_week", ascending=False)
r.to_csv(OUT / f"effect_size_{y}.csv", index=False)
pd.set_option("display.width", 170)
print(f"[24] b = {b:.6f} (95% CI {lo:.6f} to {hi:.6f}, state clusters), outcome {y}")
print(r.round(2).to_string(index=False))
print(f"[24] median extra events per week in the exposed state: {r.extra_per_week.median():.2f}")
print("[24] % increases are unstable where the pre-period baseline is near zero; report counts, not %.")
print("[24] The b is an average over all shocks; per-shock numbers apply that average, not shock-specific effects.")
