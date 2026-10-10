"""Three extensions of the Part 1 design (run 04_stack.py first).

(1) Exam type: the DiD separately for police, clerical/administrative and teacher exams.
    A rough proxy for pool composition while sex ratios are missing. Descriptive (1-4 shocks each).
(2) Media attention: articles about each leak in GDELT (URL slugs, weeks -1..+4 around the
    exam) against that exam's own DiD estimate. Descriptive (7 points; Spearman correlation).
(3) Pre-period-only trends: for each stack, the exposed-state slope is estimated on the PRE
    weeks only, removed from all weeks, and the DiD re-estimated. This avoids the post-fade bias
    of a trend fitted over the whole window (S2 in 17_robustness.py).
Usage: python 19_extensions.py [outcome]
Output: output/ext_exam_type.csv, output/ext_media.csv, output/ext_pretrend.txt
"""
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
from config import RAW, INT, OUT
from lib import demean, ols_cluster

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
df = pd.read_csv(INT / "stacked_main.csv", dtype={"district_id": str, "stack_id": str}, parse_dates=["week"])
df = df[df.groupby("stack_id").A_tot.transform("std") > 0]


def did(d, yy=None):
    yy = d[y] if yy is None else yy
    X = np.column_stack([d.A_tot * d.post, d.other_post_exp])
    Z = demean(np.column_stack([yy, X]), [d.fe_de, d.fe_stw])
    return ols_cluster(Z[:, 0], Z[:, 1:], d.state, ["A_x_post", "other"])["table"].iloc[0]


# (1) exam type
TYPES = {"UP_CONST_2024": "police", "BIHAR_CONST_2023": "police", "BPSC_TRE3_2024": "teacher",
         "GPSSB_JRCLERK_2023": "clerical", "JSSC_CGL_2024": "clerical", "UPPSC_ROARO_2024": "clerical",
         "UKPSC_PATWARI_2023": "clerical"}
df["etype"] = df.stack_id.map(TYPES).fillna("other")
rows = []
for t, d in df.groupby("etype"):
    r = did(d)
    rows.append({"group": t, "stacks": d.stack_id.nunique(), "b": r.coef, "se_state": r.se, "p_state": r.p})
per = []
for e, d in df.groupby("stack_id"):
    r = did(d)
    per.append({"exam_id": e, "type": TYPES.get(e, "other"), "b": r.coef, "se_state": r.se})
per = pd.DataFrame(per)
out1 = pd.concat([pd.DataFrame(rows), per.assign(group="exam: " + per.exam_id)], ignore_index=True)
out1.to_csv(OUT / "ext_exam_type.csv", index=False)
print("[19] (1) DiD by exam type (state clusters; very few shocks per group: descriptive):")
print(pd.DataFrame(rows).round(6).to_string(index=False))

# (2) media attention
KEYS = {"UP_CONST_2024": r"constable.*(up|uttar)|(up|uttar).*constable|upprpb",
        "BIHAR_CONST_2023": r"constable.*bihar|bihar.*constable|csbc",
        "BPSC_TRE3_2024": r"bpsc.*(tre|teacher)|(tre|teacher).*bpsc",
        "UPPSC_ROARO_2024": r"ro ?aro|uppsc",
        "JSSC_CGL_2024": r"jssc",
        "GPSSB_JRCLERK_2023": r"junior clerk|gpssb|gujarat.*clerk",
        "UKPSC_PATWARI_2023": r"patwari|lekhpal|ukpsc"}
mf = RAW / "candidates" / "gdelt_bulk_matches.csv"
if mf.exists():
    m = pd.read_csv(mf, dtype=str).drop_duplicates("url")
    m["date"] = pd.to_datetime(m.file_day, format="%Y%m%d")
    sh = pd.read_csv(RAW / "exam_shocks.csv", parse_dates=["exam_date", "shock_date"])
    cov = []
    for e, pat in KEYS.items():
        s = sh[sh.exam_id == e]
        if s.empty:
            continue
        t0 = s.exam_date.iloc[0] if pd.notna(s.exam_date.iloc[0]) else s.shock_date.iloc[0]
        win = m[(m.date >= t0 - pd.Timedelta(weeks=1)) & (m.date <= t0 + pd.Timedelta(weeks=4))]
        cov.append({"exam_id": e, "articles": int(win.slug.str.contains(pat, case=False, regex=True).sum())})
    if not cov:
        print("[19] (2) none of the shocks in KEYS is in exam_shocks.csv")
        cov = pd.DataFrame(columns=["exam_id", "articles"])
    cov = pd.DataFrame(cov).merge(per[["exam_id", "b"]], on="exam_id", how="left")
    cov.to_csv(OUT / "ext_media.csv", index=False)
    print("[19] (2) GDELT articles per leak vs the exam's own DiD:")
    print(cov.to_string(index=False))
    ok = cov.dropna()
    if len(ok) >= 4:
        rho, p = stats.spearmanr(ok.articles, ok.b)
        print(f"[19] (2) Spearman rho = {rho:.2f} (p = {p:.2f}, n = {len(ok)}): descriptive only")
else:
    print("[19] (2) no GDELT matches file; run collect/c02b_gdelt_bulk.py")

# (3) pre-period-only trends
adj = df[y].astype(float).copy()
slopes = {}
for e, d in df.groupby("stack_id"):
    pre = d[d.k < 0]
    x = (pre.A_tot * pre.k).to_numpy()
    Z = demean(np.column_stack([pre[y], x]), [pre.fe_de, pre.fe_stw])
    if np.allclose(Z[:, 1], 0):
        continue
    s = float(np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0][0])
    slopes[e] = s
    adj.loc[d.index] = d[y] - s * d.A_tot * d.k
base, detr = did(df), did(df, adj)
txt = (f"(3) pre-period-only trends, outcome {y}\n"
       f"baseline DiD   b = {base.coef:.6f} (state-cluster p = {base.p:.3f})\n"
       f"pre-trend-adj  b = {detr.coef:.6f} (state-cluster p = {detr.p:.3f})\n"
       f"slopes per stack (per applicant per 1,000 youth per week): "
       + ", ".join(f"{k}: {v:.2e}" for k, v in slopes.items()))
(OUT / "ext_pretrend.txt").write_text(txt)
print("[19] " + txt)
