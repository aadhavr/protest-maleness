"""Mechanisms: does the response grow with marriage-market pressure or
with weak outside options for women?

  y = b (post x A_tot) + sum_Z g_Z (post x A_tot x Z_std)
      + sum_Z h_Z (post x Z_std) + c other_post_exp + FE(de) + FE(stw) + e
Z: cohort_sex_ratio (men per woman, cohort now ~20-34), female_work_rate.
Z is standardised (mean 0, sd 1) across districts in the panel.
Input: raw/district_covariates.csv
"""
import sys
import numpy as np
import pandas as pd
from config import RAW, INT, OUT
from lib import demean, ols_cluster

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
df = pd.read_csv(INT / "stacked_main.csv", dtype={"district_id": str, "stack_id": str})
cov = pd.read_csv(RAW / "district_covariates.csv", dtype={"district_id": str})
zs = [z for z in ["cohort_sex_ratio", "female_work_rate"] if z in cov]
df = df.merge(cov[["district_id"] + zs], on="district_id", how="inner").dropna(subset=zs)
# sex splits exist for too few exams, so A_M and A_F are not separately identified: use A_tot
names, cols = ["post_x_Atot"], [df.post * df.A_tot]
for z in zs:
    d = df.drop_duplicates("district_id")[z]
    zz = (df[z] - d.mean()) / d.std()
    names += [f"post_x_Atot_x_{z}", f"post_x_{z}"]
    cols += [df.post * df.A_tot * zz, df.post * zz]
names.append("other")
cols.append(df.other_post_exp)
Z = demean(np.column_stack([df[y]] + cols), [df.fe_de, df.fe_stw])
t = ols_cluster(Z[:, 0], Z[:, 1:], df.state, names)["table"]   # treatment is state-level: cluster by state
t.to_csv(OUT / f"mechanisms_{y}.csv", index=False)
print(t.round(5).to_string(index=False))
print("[13] read g_Z as heterogeneity, not causal effects of Z: Z is not randomly assigned.")
