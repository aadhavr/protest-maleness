"""Layer 1: does the protest response scale with the pool's male share?

Step 1: for each stack e, estimate  y = b_e * (A_tot x post) + c * other_post_exp + FE.
Step 2: WLS of b_e on the pool male share, weights 1 / se_e^2.
With few exams, read step 2 as descriptive. Report the number of exams.
Usage: python 06_dose_response.py [outcome]
Output: output/dose_response_{outcome}.csv, output/figures/dose_response_{outcome}.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import INT, OUT, FIG
from lib import demean, ols_cluster, blog_style

y_name = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
df = pd.read_csv(INT / "stacked_main.csv", dtype={"district_id": str, "stack_id": str})
pool = pd.read_csv(INT / "pool_shares.csv", dtype={"exam_id": str})

rows = []
for e, d in df.groupby("stack_id"):
    if d.district_id.nunique() < 20:
        print(f"[06] skip {e}: <20 districts")
        continue
    x = (d.A_tot * d.post).to_numpy()
    Z = demean(np.column_stack([d[y_name], x, d.other_post_exp]), [d.fe_de, d.fe_stw])
    if np.allclose(Z[:, 1], 0):
        print(f"[06] skip {e}: no exposure variation after FEs")
        continue
    f = ols_cluster(Z[:, 0], Z[:, 1:], d.district_id, ["exp_post", "other"])["table"].iloc[0]
    rows.append({"exam_id": e, "b": f.coef, "se": f.se, "G": d.district_id.nunique()})
est = pd.DataFrame(rows).merge(pool[["exam_id", "male_share", "n_applicants"]], on="exam_id")
est = est.dropna(subset=["male_share"])
est.to_csv(OUT / f"dose_response_{y_name}.csv", index=False)

if len(est) < 3:
    raise SystemExit(f"[06] only {len(est)} exams with estimates; need >=3 for a slope")

w = 1 / est.se**2
Xw = np.column_stack([np.ones(len(est)), est.male_share])
W = np.diag(w)
beta = np.linalg.solve(Xw.T @ W @ Xw, Xw.T @ W @ est.b)
resid = est.b - Xw @ beta
# HC1 on the WLS second stage (first-stage noise is in the weights, not the SE)
meat = Xw.T @ W @ np.diag(resid**2) @ W @ Xw
bread = np.linalg.inv(Xw.T @ W @ Xw)
n, k = Xw.shape
V = n / (n - k) * bread @ meat @ bread
se = np.sqrt(np.diag(V))
p = 2 * stats.t.sf(abs(beta[1] / se[1]), df=n - k)
print(f"[06] exams={n}  slope on male share = {beta[1]:.5f} (se {se[1]:.5f}, p={p:.3f})")

blog_style()
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(est.male_share, est.b, s=30 + 400 * w / w.max(), alpha=0.7)
xs = np.linspace(est.male_share.min(), est.male_share.max(), 50)
ax.plot(xs, beta[0] + beta[1] * xs, lw=1.5)
for _, r in est.iterrows():
    ax.annotate(r.exam_id, (r.male_share, r.b), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, lw=0.8, color="k")
ax.set_xlabel("Male share of applicant pool")
ax.set_ylabel(r"$\hat b_e$: protest response per unit exposure")
ax.set_title(f"{n} exams; point size = 1/se²", fontsize=9, loc="left")
fig.savefig(FIG / f"dose_response_{y_name}.png")
