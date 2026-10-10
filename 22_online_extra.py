"""Part 3 additions: what do commenters blame, and how large an effect could the Trends test see?

(a) Blame targets: distribution of the classifier's blame_target in labeled comments
    (women, gender_quota, caste_quota, merit_system, government, exam_body, other, none),
    with 95% Wilson intervals. Shows where blame goes, not only that it is not gendered.
(b) Minimum detectable effect (MDE) of each Trends test, from its randomisation distribution:
    an effect smaller than the 95th percentile of |b_fake| cannot be told apart from noise.
    Converted to log points (and %) of the index at the mean exposure of the exposed states.
Run after 09_state_event_study.py for each trends_* outcome.
Output: output/online_blame_targets.csv, output/trends_mde.csv
"""
import numpy as np
import pandas as pd
from config import INT, OUT
from lib import wilson

f = INT / "comments_labeled.csv"
if f.exists():
    c = pd.read_csv(f, low_memory=False)
    if "blame_target" in c:
        c = c.dropna(subset=["blame_target"])
        n = len(c)
        rows = []
        for t, k in c.blame_target.value_counts().items():
            p, lo, hi = wilson(int(k), n)
            rows.append({"blame_target": t, "n": int(k), "share": p, "ci_lo": lo, "ci_hi": hi})
        bt = pd.DataFrame(rows)
        bt.to_csv(OUT / "online_blame_targets.csv", index=False)
        print(f"[22] (a) blame targets in {n:,} labeled comments:")
        print(bt.round(4).to_string(index=False))

st = INT / "stacked_state_main.csv"
mean_A = None
if st.exists():
    s = pd.read_csv(st, usecols=["stack_id", "state", "A_tot"]).drop_duplicates(["stack_id", "state"])
    mean_A = s[s.A_tot > 0].A_tot.mean()
rows = []
for rf in sorted(OUT.glob("ri_state_main_trends*.csv")):
    r = pd.read_csv(rf)
    mde = np.quantile(np.abs(r.b_fake), 0.95)
    row = {"outcome": rf.stem.replace("ri_state_main_", ""), "b_real": r.b_real.iloc[0], "mde_b": mde,
           "draws": len(r)}
    if mean_A:
        row["mde_log_points_at_mean_exposure"] = mde * mean_A
        row["mde_percent_at_mean_exposure"] = 100 * (np.exp(mde * mean_A) - 1)
        row["real_percent_at_mean_exposure"] = 100 * (np.exp(row["b_real"] * mean_A) - 1)
    rows.append(row)
if rows:
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "trends_mde.csv", index=False)
    print(f"[22] (b) minimum detectable effects (mean exposure in exposed states: {mean_A:.1f} applicants per 1,000 youth):")
    print(m.round(4).to_string(index=False))
    print("[22] An MDE of X% means: a rise in searches smaller than about X% after a leak would not be detected.")
else:
    print("[22] (b) no ri_state_main_trends*.csv files: run 09_state_event_study.py for the trends outcomes first")
