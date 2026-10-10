"""Candidate exam shocks from GDELT 1.0 daily event files (bulk, no API).

Why not the DOC API: it rate-limits hard and, per some documentation,
searches only a rolling ~3-month window, which is older than our study
period. The daily bulk files go back to 2013 and have no rate limit.

Source: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
(tab-separated, no header, one row per coded EVENT, not per article).
Limits (state these in the methods):
  * Rows have no article text. The filter matches the SOURCEURL slug,
    e.g. ".../neet-paper-leak-cbi-arrests-...". Articles without a
    descriptive URL are missed.
  * One article can make several event rows; the script keeps one row per URL.
  * Output is CANDIDATES for hand verification, never data.

Each day is streamed, filtered and discarded (about 5-10 GB in total over
the full period; nothing large is kept on disk). Resumable: finished days
are logged in raw/candidates/gdelt_bulk_done.txt.
Output: raw/candidates/gdelt_bulk_matches.csv, output/exam_candidates_review.csv
"""
import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from _common import get
from config import RAW, OUT, START_DATE, END_DATE

BASE = "http://data.gdeltproject.org/events/{d}.export.CSV.zip"
COL_DATE, COL_A1GEO, COL_A2GEO, COL_ACTGEO, COL_URL = 1, 37, 44, 51, 57
N_COLS = 58

# URL-slug terms (hyphens and underscores become spaces before matching)
SLUG = re.compile(
    r"paper leak|question paper|exam (?:cancel|scrap|postpon|void|re ?conduct|irregular|leak|row|scam)|"
    r"(?:cancel|scrap|postpon)\w* (?:\w+ ){0,3}exam|re ?exam|re ?test|neet|ugc net|cuet|"
    r"\bssc\b|\b[a-z]{1,4}psc\b|\bupsc\b|\brrb\b|recruitment (?:exam|scam|test)|"
    r"constable (?:exam|recruitment)|\btet\b|\bctet\b|\breet\b|patwari|lekhpal|"
    r"aspirants|job seekers?|cheating racket|dummy candidate", re.I)
INDIA_DOMAIN = re.compile(r"\.in/|\.co\.in|timesofindia|hindustantimes|indianexpress|thehindu|ndtv|"
                          r"news18|indiatoday|livemint|deccanherald|tribuneindia|newindianexpress|"
                          r"theprint|scroll\.in|thewire|zeenews|abplive|jagran|bhaskar|amarujala", re.I)
EXAMS = (r"(NEET|JEE|CUET|UGC[- ]NET|CSIR[- ]NET|SSC|CGL|CHSL|RRB|NTPC|UPSC|[A-Z]{2,5}PSC|"
         r"constable|TET|CTET|REET|STET|patwari|lekhpal|teacher recruitment|police recruitment|"
         r"board exam|recruitment)")

cdir = RAW / "candidates"
cdir.mkdir(parents=True, exist_ok=True)
done_f, match_f = cdir / "gdelt_bulk_done.txt", cdir / "gdelt_bulk_matches.csv"
done = set(done_f.read_text().split()) if done_f.exists() else set()
days = [d.strftime("%Y%m%d") for d in pd.date_range(START_DATE, END_DATE)]
todo = [d for d in days if d not in done]
print(f"[c02b] {len(days)} days in window, {len(todo)} to process")


def process(day):
    r = get(BASE.format(d=day), timeout=180, tries=4)
    if r.status_code != 200:
        return day, None, f"HTTP {r.status_code}"
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    except Exception as ex:
        return day, None, f"bad zip: {ex}"
    rows = []
    for line in raw.splitlines():
        f = line.split("\t")
        if len(f) < N_COLS:
            continue
        url = f[COL_URL]
        slug = re.sub(r"[-_/.+]+", " ", url.split("://")[-1])
        if not SLUG.search(slug):
            continue
        india = "IN" in (f[COL_A1GEO], f[COL_A2GEO], f[COL_ACTGEO]) or INDIA_DOMAIN.search(url)
        if india:
            rows.append((day, f[COL_DATE], url, slug[:300]))
    return day, rows, ""


workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_done = 0
with ThreadPoolExecutor(max_workers=workers) as pool, done_f.open("a") as dfh:
    futs = {pool.submit(process, d): d for d in todo}
    for fut in as_completed(futs):
        try:
            day, rows, err = fut.result()
        except Exception as ex:
            print(f"  {futs[fut]}: failed ({str(ex)[:80]}); re-run to retry")
            continue
        if rows is None:
            print(f"  {day}: {err}; re-run to retry")
            continue
        if rows:
            pd.DataFrame(rows, columns=["file_day", "event_date", "url", "slug"]).to_csv(
                match_f, mode="a", header=not match_f.exists(), index=False)
        dfh.write(day + "\n")
        dfh.flush()
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(todo)} days")

# ---- candidates for review ----
if not match_f.exists():
    raise SystemExit("[c02b] no matches")
m = pd.read_csv(match_f, dtype=str).drop_duplicates("url")
m["date"] = pd.to_datetime(m.file_day, format="%Y%m%d")
m["week"] = m.date - pd.to_timedelta(m.date.dt.weekday, unit="D")
m["exam_kw"] = m.slug.str.extract(EXAMS, flags=re.I)[0].str.upper()
m["leak_or_cancel"] = m.slug.str.contains(r"leak|cancel|scrap|postpon|void|re ?exam|re ?test|irregular|scam",
                                          case=False, regex=True)
cand = (m.dropna(subset=["exam_kw"]).groupby(["week", "exam_kw"])
        .agg(n_urls=("url", "size"), n_leak_cancel=("leak_or_cancel", "sum"),
             example_slug=("slug", "first"), example_url=("url", "first"))
        .reset_index())
cand = cand[cand.n_leak_cancel > 0].sort_values(["week", "n_leak_cancel"], ascending=[True, False])
cand["verdict"] = ""        # fill: shock / not_shock / duplicate
cand["exam_id"] = ""
cand.to_csv(OUT / "exam_candidates_review.csv", index=False)
print(f"[c02b] {len(m)} unique matching URLs; {len(cand)} (week, exam) candidates with leak/cancel terms "
      f"-> output/exam_candidates_review.csv")
missing = set(days) - set(done_f.read_text().split())
if missing:
    print(f"[c02b] {len(missing)} days not processed yet; run again to retry them")
