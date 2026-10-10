"""Robustness of the Part 1 main effect to time-varying confounders (Joshi checks).

All specs: stacked district panel (state exams + zero-exposure control districts),
y = b * (A_tot x post) + controls + FE(stack x district) + FE(stack x week).
  S0  baseline
  S1  + state election window: 1 if the district's state polls within the next
        ELECTION_LEAD weeks (raw/state_elections.csv)
  S2  + a linear trend in event time for the exposed state, one per stack
  S3  + post x district characteristics (standardised): literacy, urban share,
        SC/ST share, non-worker share, youth share, female work rate, cohort sex ratio
  S4  all of S1-S3
Inference: CRV1 by state (indicative) and randomisation over calendar shifts of all
shock dates together (the controls move with the calendar where they should).
Usage: python 17_robustness.py [outcome]
Output: output/robustness_{outcome}.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from config import (RAW, INT, OUT, START_DATE, END_DATE, WINDOW_PRE, WINDOW_POST,
                    MAIN_SHOCKS, STATE_EXAM_CONTROLS, EVENT_DATE)
from lib import demean, ols_cluster

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
LEAD = int(os.environ.get("PM_ELECTION_LEAD", 8))
sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"], dtype={"exam_id": str})
sh = sh[sh.shock_type.isin(MAIN_SHOCKS)]
ex = pd.read_csv(INT / "exposure.csv", dtype={"exam_id": str, "district_id": str})
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
yv = pr.set_index(["district_id", "week"])[y]
lo_w = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())
hi_w = pd.Timestamp(END_DATE)
all_weeks = pd.date_range(lo_w, hi_w, freq="7D")
ks = np.arange(-WINDOW_PRE, WINDOW_POST + 1)

# ---- district characteristics (S3) ----
Z = dist[["district_id", "state", "pop_15_29"]].copy()
cov_f = RAW / "district_covariates.csv"
if cov_f.exists():
    Z = Z.merge(pd.read_csv(cov_f, dtype={"district_id": str}), on="district_id", how="left")
pca = [f for f in (RAW / "census_pca").glob("*") if f.suffix.lower() in (".csv", ".xls", ".xlsx")] \
    if (RAW / "census_pca").exists() else []
if pca:
    p = pd.concat([pd.read_csv(f, dtype=str) if f.suffix.lower() == ".csv" else pd.read_excel(f, dtype=str) for f in pca])
    p.columns = [c.strip() for c in p.columns]
    p = p[p.Level.str.upper() == "DISTRICT"].copy()
    p["district_id"] = p.State.str.strip().str.zfill(2) + p.District.str.strip().str.zfill(3)
    num = lambda c: pd.to_numeric(p[c], errors="coerce") if c in p else np.nan
    p = p.assign(TOT_P=num("TOT_P"), P_LIT=num("P_LIT"), P_SC=num("P_SC"), P_ST=num("P_ST"), NON_WORK_P=num("NON_WORK_P"))
    tot = p[p.TRU.str.title() == "Total"].drop_duplicates("district_id").set_index("district_id")
    urb = p[p.TRU.str.title() == "Urban"].drop_duplicates("district_id").set_index("district_id")
    pz = pd.DataFrame({"literacy": tot.P_LIT / tot.TOT_P, "sc_st_share": (tot.P_SC + tot.P_ST) / tot.TOT_P,
                       "nonworker_share": tot.NON_WORK_P / tot.TOT_P,
                       "urban_share": urb.TOT_P.reindex(tot.index).fillna(0) / tot.TOT_P,
                       "tot_pop": tot.TOT_P})
    Z = Z.merge(pz.reset_index(), on="district_id", how="left")
    Z["youth_share"] = Z.pop_15_29 / Z.tot_pop
zcols = [c for c in ["literacy", "urban_share", "sc_st_share", "nonworker_share", "youth_share",
                     "female_work_rate", "cohort_sex_ratio"] if c in Z and Z[c].notna().mean() > 0.9]
for c in zcols:
    Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
Z[zcols] = Z[zcols].fillna(0)
print(f"[17] S3 characteristics: {zcols if zcols else 'none found (needs PCA / covariates)'}")

# ---- election windows (S1) ----
el_f = RAW / "state_elections.csv"
elect = set()
if el_f.exists():
    el = pd.read_csv(el_f, parse_dates=["poll_date"])
    for st, g in el.groupby("state"):
        a, b = g.poll_date.min() - pd.Timedelta(weeks=LEAD), g.poll_date.max()
        for w in pd.date_range(a - pd.Timedelta(days=a.weekday()), b, freq="7D"):
            elect.add((st, w))
    print(f"[17] election windows: {el.state.nunique()} states, {LEAD} weeks before polling to the last poll")
else:
    print("[17] no raw/state_elections.csv: S1 and S4 skip the election control")

# ---- stacks ----
groups = {}
for s in sh.itertuples():
    d = ex[ex.exam_id == s.exam_id][["district_id", "state", "A_tot"]]
    if s.level == "state":
        d = d[d.state == s.state]
        if STATE_EXAM_CONTROLS and len(d):
            d = pd.concat([d, dist[dist.state != s.state][["district_id", "state"]].assign(A_tot=0.0)], ignore_index=True)
    if len(d) and d.A_tot.std() > 1e-9 * max(d.A_tot.abs().max(), 1e-12):
        groups[s.exam_id] = (d.reset_index(drop=True), s.shock_week)
print(f"[17] stacks: {list(groups)}")


def build(weeks0):
    parts = []
    for e, (d, _) in groups.items():
        g = d.loc[d.index.repeat(len(ks))].copy()
        g["k"] = np.tile(ks, len(d))
        g["week"] = weeks0[e] + pd.to_timedelta(g.k * 7, unit="D")
        g["stack_id"] = e
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p = p[(p.week >= lo_w) & (p.week <= hi_w)].copy()
    p["y"] = yv.reindex(pd.MultiIndex.from_arrays([p.district_id, p.week])).fillna(0).to_numpy()
    p["post"] = (p.k >= 0).astype(int)
    p["elect"] = [int((s, w) in elect) for s, w in zip(p.state, p.week)]
    return p.merge(Z[["district_id"] + zcols], on="district_id", how="left")


SPECS = {"S0 baseline": (False, False, False), "S1 + election window": (True, False, False),
         "S2 + exposed-state trends": (False, True, False), "S3 + post x district traits": (False, False, True),
         "S4 all controls": (True, True, True)}


def fit(p, spec, full=False):
    use_el, use_tr, use_z = spec
    names, cols = ["A_x_post"], [p.A_tot * p.post]
    if use_el and elect:
        names.append("elect"); cols.append(p.elect)
    if use_tr:
        tr = (p.A_tot > 0).astype(float) * p.k
        for e in p.stack_id.unique():
            names.append(f"trend_{e}"); cols.append(tr * (p.stack_id == e))
    if use_z:
        for c in zcols:
            names.append(f"post_x_{c}"); cols.append(p.post * p[c])
    X = np.column_stack(cols).astype(float)
    keep = X.std(axis=0) > 0
    keep[0] = True
    Zm = demean(np.column_stack([p.y.to_numpy(float), X[:, keep]]),
                [p.stack_id + "|" + p.district_id, p.stack_id + "|" + p.week.dt.strftime("%Y%m%d")])
    t = ols_cluster(Zm[:, 0], Zm[:, 1:], p.state, [n for n, k_ in zip(names, keep) if k_])["table"]
    return t if full else t.iloc[0]


real = {e: v[1] for e, v in groups.items()}
p_real = build(real)
n_w, sp = len(all_weeks), WINDOW_PRE + WINDOW_POST
offs = [o for o in range(1, n_w) if sp < o < n_w - sp]
shift_panels = [build({e: all_weeks[(int((T - lo_w).days // 7) + o) % n_w] for e, T in real.items()}) for o in offs]
rows = []
for name, spec in SPECS.items():
    r = fit(p_real, spec)
    fake = np.array([fit(pp, spec).coef for pp in shift_panels])
    p_ri = (np.sum(np.abs(fake) >= abs(r.coef)) + 1) / (len(fake) + 1)
    rows.append({"spec": name, "b": r.coef, "se_state": r.se, "p_state": r.p, "p_randomisation": p_ri,
                 "n_shifts": len(fake)})
    print(f"[17] {name:30s} b = {r.coef:.6f}  (state-cluster p = {r.p:.3f})  randomisation p = {p_ri:.3f}")
res = pd.DataFrame(rows)
res.to_csv(OUT / f"robustness_{y}.csv", index=False)
full = fit(p_real, SPECS["S4 all controls"], full=True)
full.to_csv(OUT / f"robustness_{y}_S4_coefficients.csv", index=False)
print(f"[17] week 0 = {EVENT_DATE}. Stable b across S0-S4 = the effect is not explained by these confounders.")
