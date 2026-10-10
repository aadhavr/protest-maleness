"""Build district population and covariates from Census 2011 tables.

Download by hand (census portal, one file per state) and save:
  raw/census_c13/*.xls[x]  C-13 "Single year age returns by residence and sex"
  raw/census_pca/*.csv     Primary Census Abstract (optional; for female work rate)

The C-13 parser assumes the standard column order:
  table, state_code, district_code, area_name, age,
  total_persons, total_males, total_females, (rural P M F), (urban P M F)
It prints the first rows and checks that single ages sum to "All ages".
If your files differ, change C13_COLS below. Check before you trust it.

Outputs:
  raw/districts.csv            district_id, district_name, state, pop_15_29, pop_source
  raw/district_covariates.csv  district_id, cohort_sex_ratio, female_work_rate
district_id = 2-digit state code + 3-digit district code (Census 2011).
"""
import re
import pandas as pd
from _common import *  # noqa: F401,F403 (sets import path)
from config import RAW, YOUTH_AGES_2011, MARRIAGE_AGES_2011

C13_COLS = ["table", "state_code", "district_code", "area_name", "age", "P", "M", "F"]

files = sorted((RAW / "census_c13").glob("*.xls*"))
if not files:
    raise SystemExit("put Census 2011 C-13 files in raw/census_c13/")
frames = []
for f in files:
    for _, sh in pd.read_excel(f, sheet_name=None, header=None, dtype=str).items():
        sh = sh.iloc[:, :8]
        sh.columns = C13_COLS
        sh = sh[sh.state_code.fillna("").str.strip().str.fullmatch(r"\d{1,2}")]
        frames.append(sh)
c = pd.concat(frames, ignore_index=True)
for col in ["P", "M", "F"]:
    c[col] = pd.to_numeric(c[col].str.replace(",", ""), errors="coerce")
c["state_code"] = c.state_code.str.strip().str.zfill(2)
c["district_code"] = c.district_code.str.strip().str.zfill(3)
c["age_num"] = pd.to_numeric(c.age.str.extract(r"^(\d+)")[0], errors="coerce")
c["all_ages"] = c.age.str.contains("All", case=False, na=False)
print(c.head(8).to_string())

states = (c[(c.district_code == "000") & c.all_ages]
          .assign(state=lambda d: d.area_name.str.replace(r"^State\s*-\s*|\s*\(\d+\)\s*$", "", regex=True).str.strip().str.title())
          [["state_code", "state"]].drop_duplicates("state_code"))
d = c[c.district_code != "000"].copy()
d["district_id"] = d.state_code + d.district_code

# check: single ages (and "not stated") sum to All ages
tot = d[d.all_ages].set_index("district_id").P
parts = d[~d.all_ages].groupby("district_id").P.sum()
gap = ((parts - tot).abs() / tot).dropna()
if (gap > 0.001).any():
    print(f"[c03] warning: single ages do not sum to All ages in {int((gap > 0.001).sum())} districts; check C13_COLS")


def cohort(lo, hi, col):
    m = d.age_num.between(lo, hi)
    return d[m].groupby("district_id")[col].sum()


names = (d[d.all_ages].assign(district_name=lambda x: x.area_name.str.replace(
    r"^District\s*-\s*|\s*\(\d+\)\s*$", "", regex=True).str.strip().str.title())
    .drop_duplicates("district_id").set_index("district_id")[["state_code", "district_name"]])
out = names.join(cohort(*YOUTH_AGES_2011, "P").rename("pop_15_29")).reset_index()
out = out.merge(states, on="state_code", how="left")
out["pop_source"] = f"Census 2011 C-13, ages {YOUTH_AGES_2011[0]}-{YOUTH_AGES_2011[1]} in 2011"
out[["district_id", "district_name", "state", "pop_15_29", "pop_source"]].to_csv(RAW / "districts.csv", index=False)

cov = pd.DataFrame({"cohort_sex_ratio": cohort(*MARRIAGE_AGES_2011, "M") / cohort(*MARRIAGE_AGES_2011, "F")})

pca_files = sorted(f for f in (RAW / "census_pca").glob("*") if f.suffix.lower() in (".csv", ".xls", ".xlsx"))
if pca_files:
    p = pd.concat([pd.read_csv(f, dtype=str) if f.suffix.lower() == ".csv" else pd.read_excel(f, dtype=str)
                   for f in pca_files])
    p.columns = [x.strip() for x in p.columns]
    p = p[(p.Level.str.upper() == "DISTRICT") & (p.TRU.str.title() == "Total")]
    p["district_id"] = p.State.str.strip().str.zfill(2) + p.District.str.strip().str.zfill(3)
    p = p.drop_duplicates("district_id").set_index("district_id")
    print(f"[c03] PCA: {len(p)} districts with female work data")
    cov["female_work_rate"] = pd.to_numeric(p.TOT_WORK_F) / pd.to_numeric(p.TOT_F)
else:
    print("[c03] no PCA files; female_work_rate left out")
cov.reset_index().to_csv(RAW / "district_covariates.csv", index=False)
print(f"[c03] {len(out)} districts -> raw/districts.csv; covariates -> raw/district_covariates.csv")
