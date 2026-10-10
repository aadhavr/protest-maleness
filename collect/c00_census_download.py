"""Download the 35 state/UT Census 2011 C-13 tables (total population, not SC/ST).

For each state the script tries, in order:
  1. the Census Digital Library catalog page: it finds a link to the
     DDW-XX00C-13.xls file (XX = 2011 state code) and downloads it;
  2. the old static URL  .../2011census/C-series/DDW-XX00C-13.xls
It keeps a file only if it is a real Excel file (checks the first bytes),
so an HTML error page is never saved as data.
Output: raw/census_c13/DDW-XX00C-13.xls, raw/census_c13/download_log.csv
Then run c03_census.py.  Needs: requests, truststore (and xlrd for c03).
Usage: python3 c00_census_download.py [--insecure]
"""
import re
import sys
import time
from urllib.parse import urljoin
import pandas as pd
from _common import get
from config import RAW

# 2011 state code -> (name, Census Digital Library catalog id) for C-13 (total)
STATES = {
    "01": ("Jammu and Kashmir", 1451), "02": ("Himachal Pradesh", 1450), "03": ("Punjab", 1464),
    "04": ("Chandigarh", 1442), "05": ("Uttarakhand", 1470), "06": ("Haryana", 1449),
    "07": ("NCT of Delhi", 1446), "08": ("Rajasthan", 1465), "09": ("Uttar Pradesh", 1469),
    "10": ("Bihar", 1441), "11": ("Sikkim", 1466), "12": ("Arunachal Pradesh", 1439),
    "13": ("Nagaland", 1461), "14": ("Manipur", 1458), "15": ("Mizoram", 1460),
    "16": ("Tripura", 1468), "17": ("Meghalaya", 1459), "18": ("Assam", 1440),
    "19": ("West Bengal", 1471), "20": ("Jharkhand", 1452), "21": ("Odisha", 1462),
    "22": ("Chhattisgarh", 1443), "23": ("Madhya Pradesh", 1456), "24": ("Gujarat", 1448),
    "25": ("Daman & Diu", 1445), "26": ("Dadra & Nagar Haveli", 1444), "27": ("Maharashtra", 1457),
    "28": ("Andhra Pradesh", 1438), "29": ("Karnataka", 1453), "30": ("Goa", 1447),
    "31": ("Lakshadweep", 1455), "32": ("Kerala", 1454), "33": ("Tamil Nadu", 1467),
    "34": ("Puducherry", 1463), "35": ("Andaman and Nicobar Islands", 1437),
}
CAT = "https://censusindia.gov.in/nada/index.php/catalog/{id}"
OLD = ["https://censusindia.gov.in/2011census/C-series/DDW-{c}00C-13.xls",
       "http://www.censusindia.gov.in/2011census/C-series/DDW-{c}00C-13.xls"]
XLS_MAGIC = (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")   # old .xls (OLE2) or .xlsx (zip)

# censusindia.gov.in often does not send its intermediate certificate, so
# Python cannot verify it (browsers can, because they fetch the missing
# certificate). Fix 1: use the operating system's trust store (truststore).
# Fix 2 (--insecure): skip verification for this host only. That is
# acceptable here because every file is checked for Excel bytes and is
# public data, but use it only if fix 1 fails.
try:
    import truststore
    truststore.inject_into_ssl()
    print("[c00] using the system trust store (truststore)")
except ImportError:
    print("[c00] tip: pip install truststore  (fixes most certificate errors)")
VERIFY = "--insecure" not in sys.argv
if not VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("[c00] WARNING: certificate verification is OFF for this run")

out = RAW / "census_c13"
out.mkdir(parents=True, exist_ok=True)


def is_excel(b: bytes) -> bool:
    return b[:4] in XLS_MAGIC


def candidate_links(page_url, html):
    """Links on a catalog page that may lead to the C-13 file (not SC/ST)."""
    hrefs = re.findall(r'href="([^"]+)"', html)
    good = []
    for h in hrefs:
        u = urljoin(page_url, h.replace("&amp;", "&"))
        low = u.lower()
        if "sc" in low.split("c-13")[-1][:3] or "st.xls" in low:
            continue
        if low.endswith((".xls", ".xlsx")) or "/download/" in low or "related-materials" in low:
            good.append(u)
    return list(dict.fromkeys(good))


log = []
for code, (name, cid) in STATES.items():
    f = out / f"DDW-{code}00C-13.xls"
    if f.exists() and is_excel(f.read_bytes()[:4]):
        log.append({"code": code, "state": name, "status": "already", "url": ""})
        continue
    got, tried = None, []
    # 1) catalog page, then its related-materials page
    for page in [CAT.format(id=cid), CAT.format(id=cid) + "/related-materials"]:
        try:
            r = get(page, timeout=60, verify=VERIFY)
        except RuntimeError:
            continue
        if r.status_code != 200:
            continue
        for u in candidate_links(page, r.text):
            if u in tried:
                continue
            tried.append(u)
            try:
                d = get(u, timeout=120, verify=VERIFY)
            except RuntimeError:
                continue
            if d.status_code == 200 and is_excel(d.content):
                got = (u, d.content)
                break
            time.sleep(1)
        if got:
            break
    # 2) old static URLs
    if not got:
        for pat in OLD:
            u = pat.format(c=code)
            tried.append(u)
            try:
                d = get(u, timeout=120, verify=VERIFY)
            except RuntimeError:
                continue
            if d.status_code == 200 and is_excel(d.content):
                got = (u, d.content)
                break
    if got:
        f.write_bytes(got[1])
        log.append({"code": code, "state": name, "status": "ok", "url": got[0]})
        print(f"[c00] {code} {name}: ok ({len(got[1]) / 1e6:.1f} MB)")
    else:
        log.append({"code": code, "state": name, "status": "FAILED", "url": " | ".join(tried[:6])})
        print(f"[c00] {code} {name}: FAILED - download by hand from {CAT.format(id=cid)}")
    time.sleep(2)

pd.DataFrame(log).to_csv(out / "download_log.csv", index=False)
n_ok = sum(x["status"] in ("ok", "already") for x in log)
print(f"[c00] {n_ok}/{len(STATES)} files in {out}")
if n_ok < len(STATES):
    print("[c00] For FAILED rows: open the catalog page, click Download, and save the file "
          "as DDW-XX00C-13.xls in raw/census_c13/ (XX = the state code in the log).")
