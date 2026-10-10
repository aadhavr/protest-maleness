"""Blind hand-validation of LLM labels.

python 14_validate_labels.py make comments 300    -> output/handlabel_comments.csv
python 14_validate_labels.py score comments       -> precision/recall/F1/kappa
(same for "petitioners").
The sheet hides the model's labels. Sampling is stratified: half from
items the model called positive, half from the rest, so rare positives
get enough checks; scores are reweighted back to the full population.
"""
import sys
import numpy as np
import pandas as pd
from config import INT, OUT

mode, task = sys.argv[1], sys.argv[2]
cfg = {"comments": ("comments_labeled.csv", ["gendered_blame", "manosphere_frame"], "text_in"),
       "petitioners": ("court_orders_labeled.csv", ["petitioner_sex"], "text_in")}[task]
df = pd.read_csv(INT / cfg[0], low_memory=False)
sheet = OUT / f"handlabel_{task}.csv"

if mode == "make":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    key = cfg[1][0]
    pos = (df[key] == 1) if task == "comments" else (df[key] == "female")
    rng = np.random.default_rng(7)
    a = df[pos].sample(min(n // 2, pos.sum()), random_state=rng.integers(1e9))
    b = df[~pos].sample(min(n - len(a), (~pos).sum()), random_state=rng.integers(1e9))
    s = pd.concat([a.assign(stratum="pos"), b.assign(stratum="neg")]).sample(frac=1, random_state=1)
    s[["id", "stratum", cfg[2]] ].assign(**{f"hand_{k}": "" for k in cfg[1]}).drop(columns="stratum").to_csv(sheet, index=False)
    s[["id", "stratum"]].to_csv(OUT / f"handlabel_{task}_strata.csv", index=False)
    print(f"[14] {len(s)} items -> {sheet} (fill hand_* columns; do not look at model labels)")
else:
    h = pd.read_csv(sheet).merge(pd.read_csv(OUT / f"handlabel_{task}_strata.csv"), on="id")
    m = h.merge(df[["id"] + cfg[1]], on="id")
    key0 = cfg[1][0]
    pos_all = ((df[key0] == 1) if task == "comments" else (df[key0] == "female")).mean()
    m["w"] = np.where(m.stratum == "pos", pos_all / (m.stratum == "pos").mean(),
                      (1 - pos_all) / (m.stratum == "neg").mean())
    for k in cfg[1]:
        d = m.dropna(subset=[f"hand_{k}"])
        def to01(x):
            num = pd.to_numeric(x, errors="coerce")
            txt = x.astype(str).str.strip().str.lower()
            return ((num == 1) | txt.isin(["female", "true", "yes"])).astype(int)
        yh, yp = to01(d[f"hand_{k}"]), to01(d[k])
        print(f"[14] {k}: hand positives {int(yh.sum())}, model positives {int(yp.sum())} (in the sample)")
        if yh.sum() == 0 or yp.sum() == 0:
            print(f"[14] {k}: no positives on one side; precision/recall are not defined. "
                  "Check the hand labels (1 = yes, 0 = no).")
        w = d.w.to_numpy()
        tp, fp = (w * (yp & yh)).sum(), (w * (yp & ~yh.astype(bool))).sum()
        fn = (w * (~yp.astype(bool) & yh.astype(bool))).sum()
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        po = (w * (yp == yh)).sum() / w.sum()
        p1, q1 = (w * yp).sum() / w.sum(), (w * yh).sum() / w.sum()
        pe = p1 * q1 + (1 - p1) * (1 - q1)
        print(f"[14] {k}: n={len(d)}  precision={prec:.3f}  recall={rec:.3f}  "
              f"F1={2 * prec * rec / (prec + rec):.3f}  kappa={(po - pe) / (1 - pe):.3f}  (population-weighted)")
