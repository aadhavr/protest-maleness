"""Find CANDIDATE exam leaks/cancellations in news (GDELT DOC 2.0 API).

The output is a review sheet, not data. For each true shock, add a row
to raw/exam_shocks.csv with an official notice or two independent
reports as source_url. Record rejected candidates too (column verdict),
so the search is documented.

Note: some documentation describes the DOC API as a rolling ~3-month
window. The script queries month by month and logs empty months, so you
can see where coverage stops. For older periods, search official
notices (NTA, state PSC and selection-board sites) and news archives by hand.
Output: raw/candidates/gdelt_articles.csv, output/exam_candidates_review.csv
"""
import re
import time
import pandas as pd
from _common import get
from config import RAW, OUT, START_DATE, END_DATE, GDELT_QUERIES

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
out = RAW / "candidates"
out.mkdir(parents=True, exist_ok=True)

rows, log = [], []
for q in GDELT_QUERIES:
    for m in pd.period_range(START_DATE, END_DATE, freq="M"):
        p = {"query": q, "mode": "ArtList", "format": "json", "maxrecords": 250, "sort": "DateAsc",
             "startdatetime": m.start_time.strftime("%Y%m%d%H%M%S"),
             "enddatetime": min(m.end_time, pd.Timestamp(END_DATE)).strftime("%Y%m%d%H%M%S")}
        r = get(URL, params=p)
        time.sleep(5.5)                           # GDELT asks for about 1 call / 5 s
        try:
            arts = r.json().get("articles", [])
        except ValueError:                        # GDELT returns plain-text errors
            arts = []
            log.append({"query": q, "month": str(m), "n": 0, "msg": r.text[:200]})
            continue
        log.append({"query": q, "month": str(m), "n": len(arts), "msg": ""})
        if len(arts) == 250:
            print(f"[c02] {m}: hit 250 cap; split this month into weeks if needed")
        for a in arts:
            rows.append({"query": q, "seendate": a.get("seendate"), "title": a.get("title"),
                         "url": a.get("url"), "domain": a.get("domain"),
                         "language": a.get("language")})
    print(f"[c02] done: {q[:60]}...")

pd.DataFrame(log).to_csv(out / "gdelt_query_log.csv", index=False)
arts = pd.DataFrame(rows).drop_duplicates("url")
arts["date"] = pd.to_datetime(arts.seendate, format="%Y%m%dT%H%M%SZ", errors="coerce").dt.date
arts.to_csv(out / "gdelt_articles.csv", index=False)

# Group headlines into rough candidates: same week + shared exam keyword.
EXAMS = r"(NEET|JEE|CUET|UGC[- ]NET|SSC|CGL|RRB|NTPC|UPSC|[A-Z]{2,5}PSC|constable|TET|CTET|REET|patwari|lekhpal|board exam|recruitment)"
arts["exam_kw"] = arts.title.fillna("").str.extract(EXAMS, flags=re.I)[0].str.upper()
arts["week"] = pd.to_datetime(arts.date) - pd.to_timedelta(pd.to_datetime(arts.date).dt.weekday, unit="D")
cand = (arts.dropna(subset=["exam_kw"]).groupby(["week", "exam_kw"])
        .agg(n_articles=("url", "size"), example_title=("title", "first"), example_url=("url", "first"))
        .reset_index().sort_values(["week", "n_articles"], ascending=[True, False]))
cand["verdict"] = ""        # fill: shock / not_shock / duplicate
cand["exam_id"] = ""
cand.to_csv(OUT / "exam_candidates_review.csv", index=False)
print(f"[c02] {len(arts)} articles, {len(cand)} candidate (week, exam) groups -> exam_candidates_review.csv")
