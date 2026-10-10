"""State-week panel and state-level stacks (courts, Trends, protest by state).

Outcomes (state x week):
  n_events, n_exam, ...        ACLED district counts summed to state
  n_orders, n_orders_female,   court orders that match exam keywords (collect/c04, c09 petitioners),
  n_orders_male                mapped to states with raw/crosswalks/court_state.csv
  n_orders_body[_female|_male] stricter: the respondent is an exam/recruitment body
  trends_index                 from 07_trends_build.py (if present)
  shutdown_exam                exam-day internet shutdown days (raw/shutdowns.csv, if present)
Exposure: applicants per 1,000 youth in the exposed state(s); 0 in all other states.
Output: interim/state_week.csv, interim/stacked_state_main.csv, interim/stacked_state_placebo.csv
"""
import pandas as pd
from config import (RAW, INT, WINDOW_PRE, WINDOW_POST, MAIN_SHOCKS, PLACEBO_SHOCKS, EXPOSURE_SCALE,
                    START_DATE, END_DATE, COURT_DATE, EXAM_BODY_REGEX)
from lib import week_start

dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
pr = pr.merge(dist[["district_id", "state"]], on="district_id")
out_cols = [c for c in pr.columns if c.startswith("n_")]
sw = pr.groupby(["state", "week"])[out_cols].sum().reset_index()

f = INT / "court_orders_labeled.csv"
if f.exists():
    import re
    co = pd.read_csv(f, low_memory=False, dtype={"court_code": str})
    co["court_code"] = co.court_code.map(lambda x: re.sub(r"[^0-9]+", "_", str(x)).strip("_"))
    cs = pd.read_csv(RAW / "crosswalks" / "court_state.csv", dtype={"court_code": str})
    cs = cs[cs.state.notna() & (cs.state != "")]
    co = co.merge(cs, on="court_code", how="inner")    # SC orders have no state; excluded here
    dcol = COURT_DATE if COURT_DATE in co else "decision_date"
    dt = pd.to_datetime(co[dcol], errors="coerce", dayfirst=(dcol != "decision_date"))
    if dcol != "decision_date":
        dt = dt.fillna(pd.to_datetime(co.decision_date, errors="coerce"))
    co["week"] = week_start(dt)
    print(f"[08] court orders dated by {dcol}; {co.week.notna().mean():.1%} have a usable date")
    resp = co.get("respondent", pd.Series("", index=co.index)).fillna("").astype(str)
    co["body"] = resp.str.contains(EXAM_BODY_REGEX, case=False, regex=True)
    sex = co.get("petitioner_sex", pd.Series("", index=co.index)).fillna("")
    co["fem"], co["male"] = sex.eq("female"), sex.eq("male")
    co["body_fem"], co["body_male"] = co.body & co.fem, co.body & co.male
    co["one"] = 1
    cw = (co.dropna(subset=["week"]).groupby(["state", "week"])[["one", "fem", "male", "body", "body_fem", "body_male"]]
          .sum().rename(columns={"one": "n_orders", "fem": "n_orders_female", "male": "n_orders_male",
                                 "body": "n_orders_body", "body_fem": "n_orders_body_female",
                                 "body_male": "n_orders_body_male"}).reset_index())
    print(f"[08] orders with an exam body as respondent: {int(co.body.sum())} of {len(co)}")
    sw = sw.merge(cw, on=["state", "week"], how="outer")
else:
    print("[08] no court_orders_labeled.csv; court outcomes skipped")

f = INT / "trends_index.csv"
if f.exists():
    sw = sw.merge(pd.read_csv(f, parse_dates=["week"]), on=["state", "week"], how="outer")

f = RAW / "shutdowns.csv"
if f.exists():
    sd = pd.read_csv(f, parse_dates=["start_date", "end_date"])
    sd["end_date"] = sd.end_date.fillna(sd.start_date)
    days = [(r.state, d, r.reason) for r in sd.itertuples()
            for d in pd.date_range(r.start_date, r.end_date)]
    dd = pd.DataFrame(days, columns=["state", "date", "reason"])
    dd["week"] = week_start(dd.date)
    for reason in ["exam", "protest"]:
        c = dd[dd.reason == reason].groupby(["state", "week"]).size().rename(f"shutdown_{reason}")
        sw = sw.merge(c.reset_index(), on=["state", "week"], how="left")
sw.to_csv(INT / "state_week.csv", index=False)

# ---- state exposure and stacks (same design as Part 1, at state level) ----
# Each stack holds ALL states. The exposed state(s) get applicants per 1,000 youth;
# every other state gets 0. National exams enter only if exposure varies by state.
ex = pd.read_csv(INT / "exposure.csv", dtype={"exam_id": str, "district_id": str})
pop = dist.groupby("state").pop_15_29.sum()
se = ex.groupby(["exam_id", "state"])[["M", "F", "O"]].sum().reset_index()
for c, cols in [("A_M", ["M"]), ("A_F", ["F"]), ("A_tot", ["M", "F", "O"])]:
    se[c] = se[cols].sum(axis=1) / se.state.map(pop) * EXPOSURE_SCALE
sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"], dtype={"exam_id": str})
states = sorted(pop.index)
lo_w = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())


def build_stacks(types):
    out = []
    for s in sh[sh.shock_type.isin(types)].itertuples():
        d = pd.DataFrame({"state": states}).merge(se[se.exam_id == s.exam_id][["state", "A_M", "A_F", "A_tot"]],
                                                  on="state", how="left").fillna(0.0)
        if d.A_tot.std() < 1e-9 * max(d.A_tot.max(), 1e-12):
            print(f"[08] {s.exam_id}: no exposure variation across states; left out")
            continue
        g = pd.MultiIndex.from_product([states, range(-WINDOW_PRE, WINDOW_POST + 1)],
                                       names=["state", "k"]).to_frame(index=False)
        g["week"] = s.shock_week + pd.to_timedelta(g.k * 7, unit="D")
        g = g[(g.week >= lo_w) & (g.week <= pd.Timestamp(END_DATE))]
        out.append(g.merge(d, on="state").assign(stack_id=s.exam_id))
    if not out:
        return None
    st = pd.concat(out, ignore_index=True).merge(sw, on=["state", "week"], how="left")
    st["post"] = (st.k >= 0).astype(int)
    st["fe_se"] = st.stack_id + "|" + st.state
    st["fe_wk"] = st.stack_id + "|" + st.week.dt.strftime("%Y-%m-%d")
    return st


for tag, types in [("main", MAIN_SHOCKS), ("placebo", PLACEBO_SHOCKS)]:
    st = build_stacks(types)
    if st is None:
        print(f"[08] {tag}: no usable stacks")
        continue
    st.to_csv(INT / f"stacked_state_{tag}.csv", index=False)
    print(f"[08] {tag}: {st.stack_id.nunique()} stacks ({', '.join(st.stack_id.unique())}), {len(st):,} rows")
print(f"[08] state-week panel: {len(sw):,} rows; outcomes: {[c for c in sw.columns if c.startswith(('n_', 'trends', 'shutdown'))]}")
