"""Find and download the Census 2011 Primary Census Abstract (PCA) at district level.

The Census Digital Library has many PCA tables (total, SC, ST, houseless, ...).
This script searches the catalog, keeps the TOTAL-population PCA candidates,
downloads their Excel files, and keeps a file only if it has the columns that
c03 needs (Level, TRU, State, District, TOT_F, TOT_WORK_F) with district rows.
Usage: python3 c00b_census_pca_download.py [--insecure]
Output: raw/census_pca/*.xls[x], raw/census_pca/search_log.csv
If nothing passes the check, open search_log.csv, pick the right catalog page,
download the file by hand into raw/census_pca/, and run c03_census.py.
"""
import io
import re
import sys
import time
from urllib.parse import urljoin
import pandas as pd
from _common import get
from config import RAW

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
VERIFY = "--insecure" not in sys.argv
if not VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://censusindia.gov.in/nada/index.php/catalog"
QUERIES = ["primary census abstract district 2011", "PCA total India States District 2011",
           "primary census abstract data tables 2011"]
SKIP = re.compile(r"\b(SC|ST|scheduled|houseless|HS|others|OTH|town|village|C\.?D\.?|1991|2001|1981)\b", re.I)
NEED = {"level", "tru", "state", "district", "tot_f", "tot_work_f"}
out = RAW / "census_pca"
out.mkdir(parents=True, exist_ok=True)

cands = {}
for q in QUERIES:
    for page in (1, 2, 3):
        try:
            r = get(BASE, params={"sk": q, "ps": 50, "page": page}, timeout=60, verify=VERIFY)
        except RuntimeError as ex:
            print(f"[c00b] search failed: {ex}")
            continue
        for href, title in re.findall(r'<a[^>]+href="([^"]*catalog/\d+)"[^>]*title="([^"]+)"', r.text):
            if "2011" in title and re.search(r"primary census abstract|\bPCA\b", title, re.I) and not SKIP.search(title):
                cands[urljoin(BASE, href)] = title
        time.sleep(1)
print(f"[c00b] {len(cands)} candidate catalog pages")

log = []
for url, title in cands.items():
    try:
        page = get(url, timeout=60, verify=VERIFY).text
    except RuntimeError as ex:
        log.append({"url": url, "title": title, "status": f"page failed: {ex}"})
        continue
    links = [urljoin(url, h.replace("&amp;", "&")) for h in re.findall(r'href="([^"]+)"', page)]
    links = [l for l in links if re.search(r"\.xlsx?($|\?)|/download/", l, re.I)]
    status = "no excel link"
    for l in links:
        try:
            d = get(l, timeout=180, verify=VERIFY)
        except RuntimeError:
            continue
        if d.content[:4] not in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"):
            continue
        try:
            x = pd.read_excel(io.BytesIO(d.content), nrows=50)
        except Exception:
            continue
        cols = {str(c).strip().lower() for c in x.columns}
        if NEED <= cols and x.astype(str).apply(lambda s: s.str.upper()).isin(["DISTRICT"]).any().any():
            name = re.sub(r"[^\w.-]+", "_", l.split("/")[-1])[:80] or "pca.xlsx"
            if not name.lower().endswith((".xls", ".xlsx")):
                name += ".xlsx" if d.content[:2] == b"PK" else ".xls"
            (out / name).write_bytes(d.content)
            status = f"OK -> {name}"
            break
        status = f"excel without needed columns ({sorted(cols)[:6]}...)"
    log.append({"url": url, "title": title, "status": status})
    print(f"[c00b] {status} | {title[:80]}")
    time.sleep(1)
pd.DataFrame(log).to_csv(out / "search_log.csv", index=False)
ok = [r for r in log if r["status"].startswith("OK")]
print(f"[c00b] {len(ok)} usable PCA file(s) in {out}. Next: python3 c03_census.py")
