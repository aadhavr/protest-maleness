"""Protest photos from Wikimedia Commons -> hand-coding sheet (Layer 4).

Commons gives free licences and clear dates, so the sample is reproducible.
Add wire photos (AP, Reuters) as extra rows by hand: URL and date only;
never download or republish them.
Output: raw/photos/commons_candidates.csv
        output/photo_coding_sheet.csv  (fill: keep, exam_id, n_men, n_women, n_unclear, coder)
Coding rule: count only people whose face or body is clearly visible and in
the protesting group (not police, not press). If you cannot judge, count
as unclear. Two coders on a 20% overlap sample; report agreement.
"""
import pandas as pd
from _common import get
from config import RAW, OUT, COMMONS_QUERIES

API = "https://commons.wikimedia.org/w/api.php"
rows = []
for q in COMMONS_QUERIES:
    cont = {}
    while True:
        p = {"action": "query", "format": "json", "generator": "search", "gsrsearch": q,
             "gsrnamespace": 6, "gsrlimit": 50, "prop": "imageinfo",
             "iiprop": "url|timestamp|extmetadata", **cont}
        j = get(API, params=p).json()
        for pg in j.get("query", {}).get("pages", {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {})
            val = lambda k: (md.get(k) or {}).get("value", "")
            rows.append({"query": q, "title": pg.get("title"), "url": ii.get("url"),
                         "page": ii.get("descriptionurl"), "date_original": val("DateTimeOriginal"),
                         "license": val("LicenseShortName"), "artist": val("Artist"),
                         "description": val("ImageDescription")[:500]})
        if "continue" not in j or len(rows) > 2000:
            break
        cont = j["continue"]
    print(f"[c05] {q}: {len(rows)} total so far")

(RAW / "photos").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows).drop_duplicates("url")
df.to_csv(RAW / "photos" / "commons_candidates.csv", index=False)
sheet = df[["title", "url", "page", "date_original", "license", "description"]].assign(
    source="commons", keep="", exam_id="", n_men="", n_women="", n_unclear="", coder="", notes="")
f = OUT / "photo_coding_sheet.csv"
if f.exists():
    old = pd.read_csv(f)
    sheet = pd.concat([old, sheet[~sheet.url.isin(old.url)]], ignore_index=True)
sheet.to_csv(f, index=False)
print(f"[c05] {len(df)} candidate photos; coding sheet -> {f}")
