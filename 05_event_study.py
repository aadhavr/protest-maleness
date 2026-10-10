"""District stacked event study with separate male and female exposure.

  y_dt = sum_{k != -1} ( b^M_k A^M_de + b^F_k A^F_de ) 1[t - T_e = k]
         + c * other_post_exp_dt + a_{de} + g_{s(d),t,e} + e_dt

Errors clustered by district (CRV1, t with G-1 df).
Usage: python 05_event_study.py [outcome] [--placebo] [--total]
  --total: one series on A_tot (all applicants). Use it while sex data are missing.
Output: output/es_{tag}_{outcome}.csv, _tests.txt, figures/es_{tag}_{outcome}.png
"""
import sys
import pandas as pd
from config import INT, OUT, FIG, REF_K, REF_MODE, CLUSTER, EVENT_DATE
from lib import event_study, plot_event_study

args = [a for a in sys.argv[1:] if not a.startswith("--")]
y_name = args[0] if args else "n_exam_strict"
tag = "placebo" if "--placebo" in sys.argv else "main"
total = "--total" in sys.argv

df = pd.read_csv(INT / f"stacked_{tag}.csv", dtype={"district_id": str, "stack_id": str})
if y_name not in df:
    raise SystemExit(f"[05] {y_name} not in the panel (run the classifier and 02_acled.py). "
                     f"Available: {[c for c in df.columns if c.startswith('n_')]}")
if not total:
    has_sex = df.groupby("stack_id")[["A_M", "A_F"]].sum().sum(axis=1) > 0
    print(f"[05] stacks with sex data: {int(has_sex.sum())} of {len(has_sex)}")
    if has_sex.sum() < 4:
        print("[05] WARNING: with fewer than 4 exams with sex data, b^M and b^F are weakly identified. "
              "Use --total for the main estimate.")
fit, tab, txt = event_study(df, y_name, ["fe_de", "fe_stw"], "state" if CLUSTER == "state" else "district_id",
                            REF_K, sexes=("tot",) if total else ("M", "F"), mode=REF_MODE)
tag = tag + ("_total" if total else "")
txt = f"sample={tag}  week 0 = {EVENT_DATE}\n" + txt + "\nUnits: extra events per district-week per 1 applicant per 1,000 youth."
tab.to_csv(OUT / f"es_{tag}_{y_name}.csv", index=False)
(OUT / f"es_{tag}_{y_name}_tests.txt").write_text(txt)
print(txt)
plot_event_study(tab, FIG / f"es_{tag}_{y_name}.png", f"Effect on {y_name} per district-week", REF_K,
                 xlabel=f"Weeks relative to {EVENT_DATE.replace('_', ' ')}",
                 title=f"95% CIs, clustered by {CLUSTER}; re-centred on pre-period mean")
