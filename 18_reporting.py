"""Why are women almost absent from exam-protest reports? Four descriptive checks on ACLED.

(a) Police response: share of events with intervention / excessive force, exam vs other protests.
(b) Wording: share of reports that name protesters only with sex-neutral nouns
    (students, aspirants, candidates, youth ...) and no gendered noun.
(c) Same source: women-mention gap between exam and other protests WITHIN the same
    ACLED source (weighted by exam events per source), and within source scale.
    If the gap survives inside sources, it is not an artifact of which outlets cover exams.
(d) Crowd size (ACLED tags): exam vs other protests.
(e) Region: women-mention and Women-tag shares in Bangladesh and Nepal (all protests and
    student protests), same measures, same window.
All shares with 95% Wilson intervals. Descriptive only.
Output: output/reporting_checks.csv
"""
import re
import numpy as np
import pandas as pd
from config import RAW, INT, OUT
from lib import wilson

WOMEN = r"\b(?:women|woman|female|girls?|mahila|ladies)\b"
GENDERED = r"\b(?:women|woman|female|girls?|mahila|ladies|men|man|male|boys?)\b"
NEUTRAL = r"\b(?:students?|aspirants?|candidates?|youths?|job ?seekers?|unemployed|applicants?|trainees?|teachers?)\b"
INTERVENE = ["Protest with intervention", "Excessive force against protesters"]

ev = pd.read_csv(INT / "acled_events.csv", low_memory=False)
if "exam_strict" not in ev:
    raise SystemExit("acled_events.csv has no exam_strict column: run the classifier and 02_acled.py")
ev["exam"] = ev.exam_strict.astype(str).str.lower().eq("true")
notes = ev.notes.fillna("")
ev["women_text"] = notes.str.contains(WOMEN, case=False, regex=True)
ev["neutral_only"] = notes.str.contains(NEUTRAL, case=False, regex=True) & ~notes.str.contains(GENDERED, case=False, regex=True)
ev["any_person_noun"] = notes.str.contains(NEUTRAL + "|" + GENDERED, case=False, regex=True)
ev["intervened"] = ev.sub_event_type.isin(INTERVENE)
ev["is_protest"] = ev.event_type.eq("Protests")
rows = []


def share(label, group, col, d):
    k, n = int(d[col].sum()), len(d)
    p, lo, hi = wilson(k, n)
    rows.append({"check": label, "group": group, "share": p, "ci_lo": lo, "ci_hi": hi, "n": n})


# (a) police response, protests only (riots are violent by definition)
pr = ev[ev.is_protest]
share("a police intervention / excessive force", "exam protests", "intervened", pr[pr.exam])
share("a police intervention / excessive force", "other protests", "intervened", pr[~pr.exam])
share("a riot (not protest) share", "exam events", "is_protest", ev[ev.exam])
rows[-1]["share"] = 1 - rows[-1]["share"]; rows[-1]["ci_lo"], rows[-1]["ci_hi"] = 1 - rows[-1]["ci_hi"], 1 - rows[-1]["ci_lo"]
share("a riot (not protest) share", "other events", "is_protest", ev[~ev.exam])
rows[-1]["share"] = 1 - rows[-1]["share"]; rows[-1]["ci_lo"], rows[-1]["ci_hi"] = 1 - rows[-1]["ci_hi"], 1 - rows[-1]["ci_lo"]

# (b) wording
share("b sex-neutral nouns only", "exam events", "neutral_only", ev[ev.exam])
share("b sex-neutral nouns only", "other events", "neutral_only", ev[~ev.exam])
pp = ev[ev.any_person_noun]
share("b sex-neutral only | report names any person noun", "exam events", "neutral_only", pp[pp.exam])
share("b sex-neutral only | report names any person noun", "other events", "neutral_only", pp[~pp.exam])

# (c) same source
if "source" in ev:
    s = ev.assign(src=ev.source.fillna("").str.split(";")).explode("src")
    s["src"] = s.src.str.strip()
    s = s[s.src != ""]
    g = s.groupby(["src", "exam"]).women_text.agg(["mean", "size"]).unstack("exam")
    g.columns = [f"{a}_{'exam' if b else 'other'}" for a, b in g.columns]
    g = g.dropna()
    g = g[(g.size_exam >= 10) & (g.size_other >= 10)]
    if len(g):
        g["diff"] = g.mean_exam - g.mean_other
        w = g.size_exam / g.size_exam.sum()
        wd = float((w * g["diff"]).sum())
        rng = np.random.default_rng(1)
        boot = [float((w.iloc[i] / w.iloc[i].sum() * g["diff"].iloc[i]).sum())
                for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
        rows.append({"check": "c within-source gap: women mention, exam minus other", "group": f"{len(g)} sources",
                     "share": wd, "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                     "n": int(g.size_exam.sum())})
        g.sort_values("size_exam", ascending=False).to_csv(OUT / "reporting_by_source.csv")
        print(f"[18] (c) {len(g)} sources with >= 10 exam and >= 10 other events; per-source table -> reporting_by_source.csv")
    else:
        print("[18] (c) no source has >= 10 exam and >= 10 other events")
    if "source_scale" in ev:
        for sc, d in ev.groupby("source_scale"):
            if d.exam.sum() >= 20:
                share("c women mention within source scale", f"{sc}: exam", "women_text", d[d.exam])
                share("c women mention within source scale", f"{sc}: other", "women_text", d[~d.exam])

# (a2-b2) the same checks for party-led and non-party exam protests
PARTY = (r"\b(?:INC|Indian National Congress|BJP|Bharatiya Janata|SP|Samajwadi|AAP|Aam Aadmi|BSP|Bahujan Samaj|"
         r"RJD|Rashtriya Janata|JD\(U\)|Janata Dal|CPI|CPI\(M\)|CPI-M|Communist Party|TMC|AITC|Trinamool|DMK|AIADMK|"
         r"Shiv Sena|SS-UBT|NCP|BRS|YSRCP|TDP|Telugu Desam|BJD|Biju Janata|JMM|Jharkhand Mukti|INLD|JJP|RLD|"
         r"IYC|Indian Youth Congress|BJYM|DYFI|Yuva Morcha)\b")
acts = ev[[c for c in ["actor1", "assoc_actor_1", "actor2", "assoc_actor_2"] if c in ev]].fillna("").agg(" ".join, axis=1)
ev["party"] = acts.str.contains(PARTY, regex=True)
ex = ev[ev.exam]
for lab, d in [("non-party exam events", ex[~ex.party]), ("party-led exam events", ex[ex.party])]:
    share("p women mentioned", lab, "women_text", d)
    share("p sex-neutral nouns only", lab, "neutral_only", d)
    share("p police intervention / excessive force (protests)", lab, "intervened", d[d.is_protest])

# (d) crowd size
if "tags" in ev:
    ev["crowd"] = ev.tags.fillna("").str.extract(r"crowd size=([^;]+)")[0].str.strip().str.lower()
    ct = pd.crosstab(ev.exam.map({True: "exam", False: "other"}), ev.crowd.fillna("not given"), normalize="index")
    ct.to_csv(OUT / "reporting_crowd_size.csv")
    print("[18] (d) crowd size distribution (row shares):")
    print(ct.round(3).to_string())

# (e) region
for cty in ["bangladesh", "nepal"]:
    files = sorted((RAW / f"acled_{cty}").glob("*.csv"))
    if not files:
        continue
    r = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True).drop_duplicates("event_id_cnty")
    r["women_text"] = r.notes.fillna("").str.contains(WOMEN, case=False, regex=True)
    act = r[[c for c in ["assoc_actor_1", "assoc_actor_2", "actor1", "actor2"] if c in r]].fillna("").agg(" ".join, axis=1)
    r["women_tagged"] = act.str.contains(r"\bWomen\b", regex=True)
    r["students"] = act.str.contains("Students", regex=True)
    for lab, d in [("all events", r), ("student events", r[r.students])]:
        share("e women mentioned", f"{cty.title()}: {lab}", "women_text", d)
        share("e Women actor tag", f"{cty.title()}: {lab}", "women_tagged", d)
india_students = ev[ev[[c for c in ["assoc_actor_1", "assoc_actor_2"] if c in ev]].fillna("").agg(" ".join, axis=1)
                    .str.contains("Students", regex=True)]
share("e women mentioned", "India: student events", "women_text", india_students)
share("e women mentioned", "India: all events", "women_text", ev)

res = pd.DataFrame(rows)
res.to_csv(OUT / "reporting_checks.csv", index=False)
pd.set_option("display.width", 170)
print(res.round(3).to_string(index=False))
