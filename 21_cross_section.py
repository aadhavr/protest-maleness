"""Which districts protest about exams? A within-state cross-section (associations only).

Outcome: exam protests per 10 million residents aged 15-29 over the whole window,
as log(1 + rate); also any exam protest (0/1, linear probability).
Regressors: standardised Census 2011 traits (literacy, urban share, SC/ST share, non-worker
share, youth share, female work rate, cohort sex ratio). State fixed effects: each district is
compared only with districts in its own state, so state institutions, politics and exam
administration are held fixed. Errors clustered by state. Not causal.
Usage: python 21_cross_section.py [outcome]
Output: output/cross_section_{outcome}.csv
"""
import sys
import numpy as np
import pandas as pd
from config import RAW, INT, OUT
from lib import demean, ols_cluster, district_traits

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
pr = pd.read_csv(INT / "acled_district_week.csv", dtype={"district_id": str})
Z, cols = district_traits(RAW)
if not cols:
    raise SystemExit("no district traits: needs raw/district_covariates.csv and/or Census PCA files")
tot = pr.groupby("district_id")[y].sum().rename("count")
d = Z.merge(tot, on="district_id", how="left").fillna({"count": 0}).dropna(subset=cols + ["pop_15_29"])
d["rate"] = d["count"] / d.pop_15_29 * 1e7
d["log_rate"] = np.log1p(d.rate)
d["any"] = (d["count"] > 0).astype(float)
print(f"[21] {len(d)} districts; {int((d['count'] > 0).sum())} with any {y}; traits: {cols}")
out = []
for dep in ["log_rate", "any"]:
    M = demean(np.column_stack([d[dep]] + [d[c] for c in cols]), [d.state])
    t = ols_cluster(M[:, 0], M[:, 1:], d.state, cols)["table"].assign(outcome=dep)
    out.append(t)
    print(f"[21] {dep} (state FE, state clusters; per 1 SD of each trait):")
    print(t[["term", "coef", "se", "p"]].round(4).to_string(index=False))
pd.concat(out).to_csv(OUT / f"cross_section_{y}.csv", index=False)
print("[21] Associations only: districts differ in many unmeasured ways even within a state.")
