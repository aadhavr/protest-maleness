"""Court orders about exam leaks/cancellations (Layer 2: voice through courts).

Source: Dattam Labs open datasets on AWS (CC-BY-4.0), public S3, no account:
  s3://indian-high-court-judgments     metadata/parquet/year=Y/court=C/bench=B/metadata.parquet
  s3://indian-supreme-court-judgments  metadata/parquet/year=Y/...
These are judgments and ORDERS with decision dates, not filings. Orders in
writ petitions often come fast, but treat timing as lagged voice.
Cite: "Indian High Court Judgments was accessed on DATE from
https://registry.opendata.aws/indian-high-court-judgments."

Output: raw/courts/{bucket}_{year}_{court}_{bench}.parquet  (downloaded metadata)
        interim/court_orders_exam.csv  (keyword hits, one row per order)
Needs: pyarrow
"""
import re
import xml.etree.ElementTree as ET
import pandas as pd
from _common import get
from config import RAW, INT, COURT_REGEX, COURT_YEARS

BUCKETS = {"hc": "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com",
           "sc": "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"}
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
out = RAW / "courts"
out.mkdir(parents=True, exist_ok=True)


def list_keys(base, prefix):
    keys, token = [], None
    while True:
        p = {"list-type": "2", "prefix": prefix}
        if token:
            p["continuation-token"] = token
        root = ET.fromstring(get(base + "/", params=p).content)
        keys += [k.text for k in root.findall(".//s3:Contents/s3:Key", NS)]
        nxt = root.find("s3:NextContinuationToken", NS)
        if nxt is None:
            return keys
        token = nxt.text


hits = []
for tag, base in BUCKETS.items():
    for y in COURT_YEARS:
        keys = [k for k in list_keys(base, f"metadata/parquet/year={y}/") if k.endswith(".parquet")]
        print(f"[c04] {tag} {y}: {len(keys)} parquet files")
        for k in keys:
            local = out / (f"{tag}_" + re.sub(r"[^\w=.-]+", "_", k.split("metadata/parquet/")[1]))
            if not local.exists():
                local.write_bytes(get(f"{base}/{k}").content)
            df = pd.read_parquet(local)
            text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
            m = pd.Series(False, index=df.index)
            for c in text_cols:          # column by column: fast, and safe with missing values
                m |= df[c].fillna("").astype(str).str.contains(COURT_REGEX, case=False, regex=True)
            if m.any():
                h = df[m].copy()
                h["source_bucket"], h["source_key"] = tag, k
                ck = re.search(r"court=([^/]+)", k)
                h["court_code"] = ck.group(1) if ck else "SC"
                hits.append(h)

res = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame()
date_col = next((c for c in res.columns if "decision_date" in c.lower()), None)
if date_col:
    res["decision_date"] = pd.to_datetime(res[date_col], errors="coerce", dayfirst=True)
title_col = next((c for c in res.columns if c.lower() in ("title", "case_title", "parties")), None)
res["title_text"] = res[title_col] if title_col else ""
res.to_csv(INT / "court_orders_exam.csv", index=False)
print(f"[c04] {len(res)} keyword-matched orders -> interim/court_orders_exam.csv")
print(f"[c04] columns available: {list(res.columns)[:25]}")
print("[c04] next: map court_code -> state in raw/crosswalks/court_state.csv "
      "(court codes: github.com/vanga/indian-high-court-judgments opendata/docs/high_courts.csv)")
