"""Build district x exam exposure by sex and each exam's pool sex ratio.

Input : raw/applicants/*.csv  (exam_id, district_id, state, sex, applicants, source_url,
                               geo_level = "district" | "state" | "region" | "national" | "national_rest")
        national_rest = the national total minus the states given as state rows for that exam
        sex = M / F, or "total" when no sex split exists (counts in A_tot only)
        region rows: state column holds the region name from raw/regions.csv
        raw/districts.csv     (district_id, district_name, state, pop_15_29)
Output: interim/exposure.csv, interim/pool_shares.csv

District-by-sex applicant counts are rarely published. Where only STATE
totals by sex exist (geo_level = "state"), the script spreads them over
the state's districts by youth population and marks imputed = True.
Imputed exposure is constant per youth inside a state, so it identifies
effects only ACROSS states: set FE_TIME = "week" in config.py for those
estimates, and report them separately.
"""
import pandas as pd
from config import RAW, INT, EXPOSURE_SCALE
from lib import require_cols

files = sorted((RAW / "applicants").glob("*.csv"))
if not files:
    raise SystemExit("put applicant files in raw/applicants/")
ap = pd.concat([pd.read_csv(f, dtype={"district_id": str, "exam_id": str}) for f in files], ignore_index=True)
require_cols(ap, ["exam_id", "sex", "applicants", "source_url"], "applicants")
if "geo_level" not in ap:
    ap["geo_level"] = "district"
ap["geo_level"] = ap.geo_level.fillna("district")
if ap.source_url.isna().any():
    raise SystemExit(f"applicant rows without source_url: {ap.exam_id[ap.source_url.isna()].unique().tolist()}")
ap["sex"] = ap.sex.astype(str).str.upper().str[0].map({"M": "M", "F": "F"}).fillna("O")

dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
require_cols(dist, ["district_id", "state", "pop_15_29"], "districts.csv")

# --- national totals: spread over all districts by youth population ---
na = ap[ap.geo_level == "national"]
if len(na):
    w = dist.assign(w=dist.pop_15_29 / dist.pop_15_29.sum())[["district_id", "w"]]
    na = na.drop(columns=["district_id"], errors="ignore").assign(_k=1).merge(w.assign(_k=1), on="_k").drop(columns="_k")
    na["applicants"] = na.applicants * na.w
    na["imputed"] = True
# --- national_rest: national remainder after the states given separately for that exam ---
nr = ap[ap.geo_level == "national_rest"]
if len(nr):
    parts = []
    for e, g in nr.groupby("exam_id"):
        covered = set(ap[(ap.exam_id == e) & (ap.geo_level == "state")].state)
        dd_ = dist[~dist.state.isin(covered)]
        w = dd_.assign(w=dd_.pop_15_29 / dd_.pop_15_29.sum())[["district_id", "w"]]
        parts.append(g.drop(columns=["district_id"], errors="ignore").assign(_k=1)
                     .merge(w.assign(_k=1), on="_k").drop(columns="_k"))
    nr = pd.concat(parts, ignore_index=True)
    nr["applicants"] = nr.applicants * nr.w
    nr["imputed"] = True
# --- regions (e.g. Telangana inside 2011 Andhra Pradesh): raw/regions.csv (region, district_id) ---
rg = ap[ap.geo_level == "region"]
if len(rg):
    regs = pd.read_csv(RAW / "regions.csv", dtype={"district_id": str})
    if "district_id" not in regs and "district_name" in regs:      # match by name inside 2011 state
        norm = lambda x: "".join(ch for ch in str(x).lower() if ch.isalnum())
        key = dist.assign(_n=dist.district_name.map(norm), _s=dist.state.map(norm))
        regs = regs.assign(_n=regs.district_name.map(norm), _s=regs.state_2011.map(norm)).merge(
            key[["_n", "_s", "district_id"]], on=["_n", "_s"], how="left")
        bad = regs[regs.district_id.isna()]
        if len(bad):
            print(f"[03] regions.csv names not found in districts.csv: {bad.district_name.tolist()} - fix the spelling")
        regs = regs.dropna(subset=["district_id"])
    rw = regs.merge(dist[["district_id", "pop_15_29"]], on="district_id")
    rw["w"] = rw.pop_15_29 / rw.groupby("region").pop_15_29.transform("sum")
    rg = rg.drop(columns=["district_id"], errors="ignore").merge(rw[["region", "district_id", "w"]],
                                                               left_on="state", right_on="region", how="inner")
    rg["applicants"] = rg.applicants * rg.w
    rg["imputed"] = True
# --- spread state totals over districts by youth population ---
st = ap[ap.geo_level == "state"]
if len(st):
    require_cols(st, ["state"], "state-level applicant rows")
    share = dist.assign(w=dist.pop_15_29 / dist.groupby("state").pop_15_29.transform("sum"))
    st = st.drop(columns=["district_id"], errors="ignore").merge(
        share[["district_id", "state", "w"]], on="state", how="inner")
    st["applicants"] = st.applicants * st.w
    st["imputed"] = True
dd = ap[ap.geo_level == "district"].assign(imputed=False)
ap = pd.concat([dd, st] + [x for x in (na, nr, rg) if len(x)], ignore_index=True)

unknown = set(ap.district_id) - set(dist.district_id)
if unknown:
    print(f"[03] warning: {len(unknown)} district_ids not in districts.csv, e.g. {list(unknown)[:5]}")

wide = (ap.pivot_table(index=["exam_id", "district_id", "imputed"], columns="sex",
                       values="applicants", aggfunc="sum", fill_value=0)
        .reindex(columns=["M", "F", "O"], fill_value=0).reset_index())
wide = wide.merge(dist[["district_id", "state", "pop_15_29"]], on="district_id", how="inner")
for s in ["M", "F"]:
    wide[f"A_{s}"] = wide[s] / wide.pop_15_29 * EXPOSURE_SCALE
wide["A_tot"] = (wide.M + wide.F + wide.O) / wide.pop_15_29 * EXPOSURE_SCALE

pool = wide.groupby("exam_id").agg(M=("M", "sum"), F=("F", "sum"), O=("O", "sum"),
                                   imputed=("imputed", "any"))
pool["n_applicants"] = pool.M + pool.F + pool.O
pool["male_share"] = pool.M / (pool.M + pool.F)
no_sex = pool.index[(pool.M + pool.F) == 0].tolist()
if no_sex:
    print(f"[03] warning: no sex breakdown for exams {no_sex}; they cannot enter the M/F model")

wide[["exam_id", "district_id", "state", "imputed", "M", "F", "O", "A_M", "A_F", "A_tot"]].to_csv(
    INT / "exposure.csv", index=False)
pool.reset_index().to_csv(INT / "pool_shares.csv", index=False)
print(pool[["n_applicants", "male_share", "imputed"]].round(3).to_string())
print(f"[03] exposure for {wide.exam_id.nunique()} exams x {wide.district_id.nunique()} districts")
