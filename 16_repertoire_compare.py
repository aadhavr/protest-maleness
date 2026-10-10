"""Layer 2, descriptive: do women appear more in courts than in street protest reports?

Courts : female share of petitioners (persons with a known sex) in exam cases,
         (a) all keyword-matched cases, (b) cases with an exam body as respondent.
Street : share of ACLED exam protests (classifier label) that
         (c) carry a "Women" actor tag, (d) mention women in the event description.
Each also for all cases / all protests, and for the 7 exposed states only.
95% Wilson intervals. CAUTION: these are DIFFERENT measures (petitioner sex vs a
protest report); compare the patterns, not the levels, and say so in the post.
Output: output/repertoire_compare.csv, output/figures/repertoire_compare.png
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW, INT, OUT, FIG, EXAM_BODY_REGEX, MAIN_SHOCKS
from lib import blog_style


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, c - h, c + h


sh = pd.read_csv(RAW / "exam_shocks.csv")
exposed = set(sh[sh.shock_type.isin(MAIN_SHOCKS) & (sh.level == "state")].state)
rows = []

# ---- courts ----
co = pd.read_csv(INT / "court_orders_labeled.csv", low_memory=False, dtype={"court_code": str})
co["court_code"] = co.court_code.map(lambda x: re.sub(r"[^0-9]+", "_", str(x)).strip("_"))
cs = pd.read_csv(RAW / "crosswalks" / "court_state.csv", dtype={"court_code": str})
co = co.merge(cs[["court_code", "state"]], on="court_code", how="left")
co["body"] = co.get("respondent", pd.Series("", index=co.index)).fillna("").astype(str).str.contains(
    EXAM_BODY_REGEX, case=False, regex=True)
known = co[co.petitioner_sex.isin(["female", "male"])]
for scope, d in [("all states", known), ("exposed states", known[known.state.isin(exposed)])]:
    for label, sub in [("Court: all exam cases", d), ("Court: exam body is respondent", d[d.body])]:
        k, n = int((sub.petitioner_sex == "female").sum()), len(sub)
        p, lo, hi = wilson(k, n)
        rows.append({"measure": label, "scope": scope, "female_share": p, "ci_lo": lo, "ci_hi": hi,
                     "n": n, "definition": "female petitioners / petitioners with known sex"})
print(f"[16] courts: {len(co)} cases, {len(known)} with a known petitioner sex "
      f"({(co.petitioner_sex == 'unknown').mean():.0%} unknown: organisations, groups, unclear names)")

# ---- street ----
ev = pd.read_csv(INT / "acled_events.csv", low_memory=False)
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})[["district_id", "state"]]
ev["district_id"] = ev.district_id.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
ev = ev.merge(dist, on="district_id", how="left")
if "exam_strict" not in ev:
    raise SystemExit("acled_events.csv has no exam_strict column: run the classifier and 02_acled.py")
ev["women_text"] = ev.notes.fillna("").str.contains(r"\b(?:women|woman|female|girls?|mahila)\b",
                                                    case=False, regex=True)
for scope, d in [("all states", ev), ("exposed states", ev[ev.state.isin(exposed)])]:
    for label, sub in [("Street: exam protests", d[d.exam_strict == True]), ("Street: all protests", d)]:
        for col, how in [("women_tagged", "Women actor tag"), ("women_text", "women mentioned in description")]:
            k, n = int(sub[col].sum()), len(sub)
            p, lo, hi = wilson(k, n)
            rows.append({"measure": f"{label} ({how})", "scope": scope, "female_share": p, "ci_lo": lo,
                         "ci_hi": hi, "n": n, "definition": f"events with {how} / events"})

res = pd.DataFrame(rows)
res.to_csv(OUT / "repertoire_compare.csv", index=False)
pd.set_option("display.width", 160)
print(res[["measure", "scope", "female_share", "ci_lo", "ci_hi", "n"]].round(3).to_string(index=False))

blog_style()
r = res[res.scope == "all states"].iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(r) + 1.2))
yy = np.arange(len(r))
ax.errorbar(r.female_share, yy, xerr=[r.female_share - r.ci_lo, r.ci_hi - r.female_share], fmt="o", capsize=3)
ax.set_yticks(yy, r.measure, fontsize=8)
ax.set_xlabel("Share (95% Wilson interval)")
ax.set_title("Women in courts vs in street-protest reports (different measures; compare patterns)",
             fontsize=9, loc="left")
fig.savefig(FIG / "repertoire_compare.png")
print(f"[16] -> {FIG / 'repertoire_compare.png'}")
