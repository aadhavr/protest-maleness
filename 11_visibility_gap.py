"""Layer 4: how much more male does a protest LOOK than the pool it represents?

Input : output/photo_coding_sheet.csv (hand-coded; keep = 1 rows used)
        interim/pool_shares.csv
Per exam: photo male share = sum n_men / sum (n_men + n_women), people pooled
across photos; 95% CI by bootstrap over photos (photos are the sampling unit).
gap = photo male share - applicant-pool male share.
If a photo has two coders, their agreement is reported and their counts averaged.
Output: output/visibility_gap.csv, output/figures/visibility_gap.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import INT, OUT, FIG
from lib import blog_style

B = 2000
rng = np.random.default_rng(42)
ph = pd.read_csv(OUT / "photo_coding_sheet.csv", dtype={"exam_id": str})
ph = ph[pd.to_numeric(ph.keep, errors="coerce") == 1].copy()
for c in ["n_men", "n_women", "n_unclear"]:
    ph[c] = pd.to_numeric(ph[c], errors="coerce").fillna(0)

# inter-coder agreement where a photo was coded twice
dup = ph.groupby("url").filter(lambda d: d.coder.nunique() >= 2)
if len(dup):
    s = dup.assign(ms=dup.n_men / (dup.n_men + dup.n_women).replace(0, np.nan))
    w = s.pivot_table(index="url", columns="coder", values="ms").dropna()
    if w.shape[1] >= 2:
        a, b = w.iloc[:, 0], w.iloc[:, 1]
        print(f"[11] double-coded photos: {len(w)}; r = {np.corrcoef(a, b)[0, 1]:.3f}; "
              f"mean |diff| in male share = {np.mean(np.abs(a - b)):.3f}")
ph = ph.groupby(["url", "exam_id"], as_index=False)[["n_men", "n_women", "n_unclear"]].mean()

pool = pd.read_csv(INT / "pool_shares.csv", dtype={"exam_id": str}).set_index("exam_id")
rows = []
for e, d in ph.groupby("exam_id"):
    m, f = d.n_men.to_numpy(), d.n_women.to_numpy()
    if (m + f).sum() == 0:
        continue
    est = m.sum() / (m + f).sum()
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        den = (m[i] + f[i]).sum()
        if den > 0:
            bs.append(m[i].sum() / den)
    ps = pool.male_share.get(e, np.nan)
    rows.append({"exam_id": e, "n_photos": len(d), "n_people": int((m + f).sum()),
                 "unclear_share": d.n_unclear.sum() / (m + f + d.n_unclear).sum(),
                 "photo_male_share": est, "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975),
                 "pool_male_share": ps, "gap": est - ps,
                 "gap_ci_lo": np.quantile(bs, .025) - ps, "gap_ci_hi": np.quantile(bs, .975) - ps})
res = pd.DataFrame(rows).sort_values("gap")
res.to_csv(OUT / "visibility_gap.csv", index=False)
print(res.round(3).to_string(index=False))
print("[11] the pool share is treated as fixed; its sampling error is small next to the photo sample.")

blog_style()
fig, ax = plt.subplots(figsize=(6, 0.45 * len(res) + 1.2))
yy = np.arange(len(res))
ax.hlines(yy, res.pool_male_share, res.photo_male_share, color="grey", lw=1)
ax.scatter(res.pool_male_share, yy, label="Applicant pool", zorder=3)
ax.errorbar(res.photo_male_share, yy, xerr=[res.photo_male_share - res.ci_lo, res.ci_hi - res.photo_male_share],
            fmt="o", capsize=2, label="People in protest photos", zorder=3)
ax.set_yticks(yy, res.exam_id)
ax.axvline(0.5, lw=0.8, ls="--", color="grey")
ax.set_xlabel("Male share")
ax.legend(frameon=False, loc="lower right")
fig.savefig(FIG / "visibility_gap.png")
