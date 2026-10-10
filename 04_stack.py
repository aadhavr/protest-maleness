"""Build the stacked event-study panel (one stack per shock).

Stack e holds every district with exposure to exam e (for state exams,
only districts in that state), for weeks T_e + k, k in [-PRE, POST].
Usage: python 04_stack.py            -> main shocks
       python 04_stack.py --placebo  -> placebo shocks
Output: interim/stacked_{main|placebo}.csv
"""
import sys
import pandas as pd
from config import (START_DATE, END_DATE, RAW, INT, WINDOW_PRE, WINDOW_POST, DROP_OVERLAP,
                    MAIN_SHOCKS, PLACEBO_SHOCKS, FE_TIME, STATE_EXAM_CONTROLS)

placebo = "--placebo" in sys.argv
types = PLACEBO_SHOCKS if placebo else MAIN_SHOCKS
tag = "placebo" if placebo else "main"

sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"], dtype={"exam_id": str})
ex = pd.read_csv(INT / "exposure.csv", dtype={"exam_id": str, "district_id": str})
import os
pr = pd.read_csv(INT / os.environ.get("PM_ACLED_PANEL", "acled_district_week.csv"), parse_dates=["week"], dtype={"district_id": str})
dist_all = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})[["district_id", "state"]]
outcomes = [c for c in pr.columns if c.startswith("n_")]

main_weeks = sh[sh.shock_type.isin(MAIN_SHOCKS)].set_index("exam_id").shock_week
exposed = ex[ex.A_tot > 0].groupby("district_id").exam_id.apply(set)
lo, hi = pd.Timedelta(weeks=WINDOW_PRE), pd.Timedelta(weeks=WINDOW_POST)

# Exposure to every main shock in its own post period, by district-week.
# Used as a control: other_post_exp = this total minus the stack's own shock.
ms = sh[sh.shock_type.isin(MAIN_SHOCKS)][["exam_id", "shock_week"]]
ep = ex.merge(ms, on="exam_id")[["district_id", "exam_id", "A_tot", "shock_week"]]
ep = ep.loc[ep.index.repeat(WINDOW_POST + 1)].copy()
ep["week"] = ep.shock_week + pd.to_timedelta(ep.groupby(level=0).cumcount() * 7, unit="D")
post_exp = ep.groupby(["district_id", "week"]).A_tot.sum().rename("all_post_exp").reset_index()

stacks, log = [], []
for _, s in sh[sh.shock_type.isin(types)].iterrows():
    e, T = s.exam_id, s.shock_week
    d = ex[ex.exam_id == e]
    if s.level == "state":
        d = d[d.state == s.state]
        if STATE_EXAM_CONTROLS and len(d):
            # add every other district with zero exposure as a control group, so the
            # stack compares the exam's state (or region) with the rest of India
            ctrl = dist_all[~dist_all.district_id.isin(d.district_id) & (dist_all.state != s.state)]
            d = pd.concat([d, ctrl.assign(exam_id=e, A_M=0.0, A_F=0.0, A_tot=0.0, imputed=False)],
                          ignore_index=True)
    n0 = int((d.A_tot > 0).sum())
    if DROP_OVERLAP:
        others = main_weeks.drop(e, errors="ignore")
        clash = set(others[(others >= T - lo) & (others <= T + hi)].index)
        if clash:
            bad = [dd for dd in d.district_id if exposed.get(dd, set()) & clash]
            # (controls exposed to another shock are dropped too)
            d = d[~d.district_id.isin(bad)]
    kept = int((d.A_tot > 0).sum())
    log.append({"exam_id": e, "districts_exposed": n0, "districts_kept": kept})
    if kept == 0:
        continue
    ks = range(-WINDOW_PRE, WINDOW_POST + 1)
    grid = pd.MultiIndex.from_product([d.district_id.unique(), ks], names=["district_id", "k"]).to_frame(index=False)
    grid["week"] = T + pd.to_timedelta(grid.k * 7, unit="D")
    grid = grid.merge(d[["district_id", "state", "A_M", "A_F", "A_tot", "imputed"]], on="district_id")
    grid["stack_id"] = e
    stacks.append(grid)

log = pd.DataFrame(log)
print(log.to_string(index=False))
if (log.districts_kept < 0.5 * log.districts_exposed).any():
    print("[04] warning: overlap rule dropped >50% of districts in some stacks. "
          "Consider DROP_OVERLAP=False with other-shock exposure as a control.")
if not stacks:
    raise SystemExit("[04] no stacks built")

panel = pd.concat(stacks, ignore_index=True)
# weeks outside the protest data are unknown, not zero: drop them
lo_w = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())
in_cov = (panel.week >= lo_w) & (panel.week <= pd.Timestamp(END_DATE))
if (~in_cov).any():
    cut = panel.loc[~in_cov].groupby("stack_id").k.agg(["min", "max"])
    print(f"[04] dropped weeks outside the protest data (start {START_DATE}, end {END_DATE}):")
    print(cut.to_string())
panel = panel[in_cov]
panel = panel.merge(pr, on=["district_id", "week"], how="left")
panel[outcomes] = panel[outcomes].fillna(0)
panel = panel.merge(post_exp, on=["district_id", "week"], how="left")
panel["all_post_exp"] = panel.all_post_exp.fillna(0)
own = panel.A_tot * ((panel.k >= 0) & (panel.k <= WINDOW_POST)) if not placebo else 0
panel["other_post_exp"] = (panel.all_post_exp - own).clip(lower=0)
panel["post"] = (panel.k >= 0).astype(int)
panel["fe_de"] = panel.stack_id + "|" + panel.district_id
wk = panel.week.dt.strftime("%Y-%m-%d")
panel["fe_stw"] = (panel.stack_id + "|" + panel.state + "|" + wk) if FE_TIME == "state_week" \
    else (panel.stack_id + "|" + wk)
if FE_TIME == "state_week" and "imputed" in panel and panel.imputed.any():
    print("[04] warning: some exposure is imputed from state totals; with state-week FEs "
          "those stacks carry almost no identifying variation. Consider FE_TIME='week'.")
panel.to_csv(INT / f"stacked_{tag}.csv", index=False)
var = panel.groupby(["stack_id", "fe_stw"]).A_tot.std().fillna(0).groupby("stack_id").max()
for e_ in var.index[var < 1e-12]:
    print(f"[04] WARNING {e_}: exposure does not vary inside its fixed-effect cells, so this exam "
          "identifies nothing. Needs state- or district-level applicant counts.")
print(f"[04] {tag}: {panel.stack_id.nunique()} stacks, {len(panel):,} rows -> stacked_{tag}.csv")
