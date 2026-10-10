"""Survey evidence on the "male resentment" premise and on who protests (descriptive, not causal).

WVS Wave 7, India (fieldwork 2023)
  (a) Ever attended a peaceful demonstration, by sex and age (Q209: 1 have done, 2 might do, 3 never)
  (b) "When jobs are scarce, men should have more right to a job than women" (Q33: 1-5, 1-2 = agree)
Pew Research Center, India survey (late 2019 - early 2020, about 30,000 adults)
  (c) The same jobs item, by sex, age, employment and education, with state fixed effects

Main test of the premise: is agreement higher among YOUNG, NOT-EMPLOYED MEN than among
comparable women (and older men)? Linear probability models, survey weights, state/region FE,
errors clustered by state/region.

Input  : data/raw/surveys/  any .sav (needs pyreadstat), .dta or .csv file per survey, named with
         "wvs" or "pew" in the file name. Variables are found automatically:
         WVS by its standard names (Q33, Q209, Q260 sex, Q262 age, Q279 employment, Q275 education,
         N_REGION_ISO region, W_WEIGHT, B_COUNTRY_ALPHA); Pew by searching variable LABELS.
         If a variable is not found, the script stops and lists candidates; write the right names
         into data/raw/surveys/mapping.json, e.g. {"pew": {"jobs": "Q47c", "sex": "GENDER", ...}}.
Output : output/survey_shares.csv, output/survey_models.csv
"""
import json
import re
import numpy as np
import pandas as pd
from config import RAW, OUT
from lib import demean, ols_cluster, wilson

SDIR = RAW / "surveys"
MAP_F = SDIR / "mapping.json"
MAPPING = json.loads(MAP_F.read_text()) if MAP_F.exists() else {}
VALUE_LABELS = {}


def load(tag):
    files = [f for f in SDIR.glob("*") if tag in f.name.lower() and f.suffix.lower() in (".sav", ".dta", ".csv")]
    if not files:
        return None, {}
    f = sorted(files)[0]
    labels = {}
    if f.suffix.lower() == ".sav":
        import pyreadstat
        df, meta = pyreadstat.read_sav(f, apply_value_formats=False)
        labels = dict(zip(meta.column_names, meta.column_labels))
        VALUE_LABELS[tag] = meta.variable_value_labels
    elif f.suffix.lower() == ".dta":
        with pd.io.stata.StataReader(f) as r:
            labels = r.variable_labels()
        df = pd.read_stata(f, convert_categoricals=False)
    else:
        df = pd.read_csv(f, low_memory=False)
    print(f"[23] {tag}: {f.name}, {len(df):,} rows, {df.shape[1]} columns")
    return df, labels


def find(df, labels, key, patterns, tag, required=True):
    if MAPPING.get(tag, {}).get(key):
        return MAPPING[tag][key]
    for p in patterns:                      # exact names first, then label search
        if p in df.columns:
            return p
    hits = [c for c, lab in labels.items() if lab and any(re.search(p, str(lab), re.I) for p in patterns)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1 and not required:
        return hits[0]
    if required:
        cand = hits or [c for c, lab in labels.items() if lab and re.search(patterns[-1].split()[0], str(lab), re.I)][:10]
        raise SystemExit(f"[23] {tag}: cannot identify '{key}'. Candidates: "
                         f"{[(c, labels.get(c)) for c in cand][:10]}\n      Add it to {MAP_F}.")
    return None


def wshare(y, w):
    ok = y.notna() & w.notna()
    y, w = y[ok].astype(float), w[ok]
    n_eff = w.sum() ** 2 / (w ** 2).sum()           # Kish effective sample size
    p = float((y * w).sum() / w.sum())
    _, lo, hi = wilson(p * n_eff, n_eff)
    return p, lo, hi, int(ok.sum())


def lpm(d, yname, xcols, fe, cl, w):
    """Weighted LPM with one set of fixed effects: weighted within-transform (subtract weighted
    group means), then scale by sqrt(w). CRV1 by `cl`; prints the number of clusters."""
    d = d.dropna(subset=[yname, fe, cl, w] + xcols)
    fe_s = d[fe].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    d = d[(d[w] > 0) & ~fe_s.isin(["nan", "none", "", "998", "999", "98", "99"])]
    if len(d) == 0:
        raise SystemExit(f"[23] no complete rows for {yname}: check the weight and the codes in mapping.json")
    M = np.column_stack([d[yname]] + [d[c] for c in xcols]).astype(float)
    wv = d[w].to_numpy(float)
    g = pd.factorize(d[fe])[0]
    sw = np.zeros(g.max() + 1); np.add.at(sw, g, wv)
    for j in range(M.shape[1]):
        s_ = np.zeros(g.max() + 1); np.add.at(s_, g, wv * M[:, j])
        M[:, j] -= (s_ / sw)[g]
    M *= np.sqrt(wv)[:, None]
    fit = ols_cluster(M[:, 0], M[:, 1:], d[cl].astype(str), xcols)
    G = fit["G"]
    if G < 20:      # too few clusters for CRV1: fall back to heteroskedasticity-robust errors
        fit = ols_cluster(M[:, 0], M[:, 1:], np.arange(len(d)), xcols)
        print(f"[23] {yname}: n = {len(d):,}; only {G} {cl} clusters, so robust (HC1-type) errors are used")
    else:
        print(f"[23] {yname}: n = {len(d):,}, clusters ({cl}) = {G}")
    return fit["table"].assign(n=len(d), clusters=G)


shares, models = [], []

# ---------------- WVS ----------------
wvs, wl = load("wvs")
if wvs is not None:
    if "B_COUNTRY_ALPHA" in wvs:
        wvs = wvs[wvs.B_COUNTRY_ALPHA.astype(str).str.upper() == "IND"]
    elif "B_COUNTRY" in wvs:
        wvs = wvs[pd.to_numeric(wvs.B_COUNTRY, errors="coerce") == 356]
    print(f"[23] WVS India rows: {len(wvs):,}")
    v = {k: find(wvs, wl, k, p, "wvs", req) for k, p, req in [
        ("jobs", ["Q33", r"jobs are scarce"], True), ("demo", ["Q209", r"peaceful demonstration"], True),
        ("sex", ["Q260", r"^sex"], True), ("age", ["Q262", r"^age"], True),
        ("emp", ["Q279", r"employment status"], True), ("edu", ["Q275", r"education"], False),
        ("region", ["N_REGION_ISO", r"region"], True), ("weight", ["W_WEIGHT", r"weight"], False)]}
    d = pd.DataFrame({k: pd.to_numeric(wvs[c], errors="coerce") if k != "region" else wvs[c].astype(str)
                      for k, c in v.items() if c})
    d = d[(d.sex.isin([1, 2])) & (d.age.between(18, 100))]
    d["female"] = (d.sex == 2).astype(int)
    d["young"] = (d.age <= 29).astype(int)
    d["unemployed"] = (d.emp == 7).astype(float).where(d.emp > 0)
    d["agree_jobs"] = d.jobs.where(d.jobs.between(1, 5)).le(2).astype(float).where(d.jobs.between(1, 5))
    d["demo_done"] = d.demo.where(d.demo.between(1, 3)).eq(1).astype(float).where(d.demo.between(1, 3))
    d["w"] = d.weight if "weight" in d else 1.0
    for yv, lab in [("demo_done", "WVS: ever attended a peaceful demonstration"),
                    ("agree_jobs", "WVS: men should have more right to a job (agree)")]:
        for grp, sub in [("men", d[d.female == 0]), ("women", d[d.female == 1]),
                         ("men 18-29", d[(d.female == 0) & (d.young == 1)]),
                         ("women 18-29", d[(d.female == 1) & (d.young == 1)])]:
            p, lo, hi, n = wshare(sub[yv], sub.w)
            shares.append({"measure": lab, "group": grp, "share": p, "ci_lo": lo, "ci_hi": hi, "n": n})
    d["female_x_young"] = d.female * d.young
    for yv in ["demo_done", "agree_jobs"]:
        t = lpm(d, yv, ["female", "young", "female_x_young"], "region", "region", "w")
        models.append(t.assign(survey="WVS 2023", outcome=yv))

# ---------------- Pew ----------------
pew, pl = load("pew")
if pew is not None:
    p = {k: find(pew, pl, k, pats, "pew", req) for k, pats, req in [
        ("jobs", [r"few jobs|jobs are scarce|more right.{0,10}job"], True),
        ("sex", [r"\bsex\b|\bgender\b|male or female"], True),
        ("age", [r"^age\b|age in .{0,5}categor|\bage of respondent|how old"], True),
        ("emp", [r"employment status|are you (currently )?(employed|working)"], False),
        ("edu", [r"education"], False),
        ("state", [r"\bstate\b"], True), ("weight", [r"weight"], False)]}
    print(f"[23] Pew variables used: {p}")
    vl = VALUE_LABELS.get("pew", {})
    for k in ["jobs", "sex", "age", "emp", "edu"]:
        if p.get(k) and p[k] in vl:
            print(f"[23] codes for {k} ({p[k]}): {vl[p[k]]}")
    q = pd.DataFrame({"jobs": pd.to_numeric(pew[p["jobs"]], errors="coerce"),
                      "sex": pd.to_numeric(pew[p["sex"]], errors="coerce"),
                      "age": pd.to_numeric(pew[p["age"]], errors="coerce"),
                      "state": pew[p["state"]].astype(str),
                      "w": pd.to_numeric(pew[p["weight"]], errors="coerce") if p["weight"] else 1.0})
    if p["weight"] and (q.w.notna().mean() < 0.9 or (q.w <= 0).mean() > 0.1):
        print(f"[23] WARNING: '{p['weight']}' does not look like a survey weight "
              f"({q.w.notna().mean():.0%} non-missing). Using UNWEIGHTED estimates. "
              "Set the right weight in mapping.json (\"weight\": \"NAME\").")
        q["w"] = 1.0
    print("[23] Pew value counts (check the codes):")
    print("  jobs:", q.jobs.value_counts().sort_index().to_dict())
    print("  sex :", q.sex.value_counts().sort_index().to_dict())
    agree = MAPPING.get("pew", {}).get("jobs_agree_codes", [1, 2])
    female = MAPPING.get("pew", {}).get("female_code", 2)
    young_codes = MAPPING.get("pew", {}).get("age_young_codes")
    if young_codes:                     # categorical age (e.g. QAGErec: 1 = 18-25)
        q = q[q.sex.notna() & q.age.notna()]
        q["young"] = q.age.isin(young_codes).astype(int)
    else:
        q = q[q.age.between(18, 100) & q.sex.notna()]
        q["young"] = (q.age <= 29).astype(int)
    q["female"] = (q.sex == female).astype(int)
    valid = q.jobs.isin(MAPPING.get("pew", {}).get("jobs_valid_codes", [1, 2, 3, 4]))
    q["agree_jobs"] = q.jobs.isin(agree).astype(float).where(valid)
    if p.get("emp"):
        notwork = MAPPING.get("pew", {}).get("not_working_codes")
        if notwork:
            q["not_working"] = pd.to_numeric(pew[p["emp"]], errors="coerce").isin(notwork).astype(float)
    for grp, sub in [("men", q[q.female == 0]), ("women", q[q.female == 1]),
                     ("young men", q[(q.female == 0) & (q.young == 1)]),
                     ("young women", q[(q.female == 1) & (q.young == 1)])]:
        pr, lo, hi, n = wshare(sub.agree_jobs, sub.w)
        shares.append({"measure": "Pew: men should have more right to a job (agree)", "group": grp,
                       "share": pr, "ci_lo": lo, "ci_hi": hi, "n": n})
    q["female_x_young"] = q.female * q.young
    xs = ["female", "young", "female_x_young"]
    if "not_working" in q:
        q["male_young_notwork"] = (1 - q.female) * q.young * q.not_working
        xs += ["not_working", "male_young_notwork"]
    else:
        print("[23] Pew employment codes not set: add 'not_working_codes' to mapping.json for the key test")
    models.append(lpm(q, "agree_jobs", xs, "state", "state", "w").assign(survey="Pew 2019-20", outcome="agree_jobs"))

if not shares:
    raise SystemExit(f"[23] no survey files in {SDIR} (file names must contain 'wvs' or 'pew')")
s = pd.DataFrame(shares)
s.to_csv(OUT / "survey_shares.csv", index=False)
pd.set_option("display.width", 160)
print(s.round(3).to_string(index=False))
if models:
    m = pd.concat(models, ignore_index=True)
    m.to_csv(OUT / "survey_models.csv", index=False)
    print(m[["survey", "outcome", "term", "coef", "se", "p", "n"]].round(4).to_string(index=False))
print("[23] Descriptive only. Weighted shares use Kish effective n for the intervals.")
