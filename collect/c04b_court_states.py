"""Map High Court codes (from c04) to 2011 Census state names.

Reads the court list published with the dataset (opendata/docs/high_courts.csv in
github.com/vanga/indian-high-court-judgments), matches court names to states with
the table below, and writes raw/crosswalks/court_state.csv (court_code, court_name, state).
Courts that serve several states are mapped to their main state (see NOTE column).
Telangana and Andhra Pradesh High Courts both map to 2011 "Andhra Pradesh".
If the download fails, the script writes the court codes found in your data with an
empty state column: fill it by hand.
Run after c04_courts_s3.py.
"""
import io
import re
import pandas as pd
from _common import get
from config import RAW, INT

URL = "https://raw.githubusercontent.com/vanga/indian-high-court-judgments/main/opendata/docs/high_courts.csv"
# keyword in court name -> (2011 state, note)
KEY = [("allahabad", "Uttar Pradesh", ""), ("patna", "Bihar", ""), ("jharkhand", "Jharkhand", ""),
       ("gujarat", "Gujarat", ""), ("uttarakhand", "Uttarakhand", ""), ("orissa", "Odisha", ""),
       ("odisha", "Odisha", ""), ("telangana", "Andhra Pradesh", "Telangana (2011: Andhra Pradesh)"),
       ("andhra", "Andhra Pradesh", ""), ("bombay", "Maharashtra", "also Goa, Dadra & Nagar Haveli, Daman & Diu"),
       ("calcutta", "West Bengal", "also Andaman & Nicobar"), ("madras", "Tamil Nadu", "also Puducherry"),
       ("karnataka", "Karnataka", ""), ("kerala", "Kerala", "also Lakshadweep"), ("rajasthan", "Rajasthan", ""),
       ("madhya", "Madhya Pradesh", ""), ("chhattisgarh", "Chhattisgarh", ""),
       ("punjab", "Punjab", "also Haryana and Chandigarh: ambiguous"), ("delhi", "Delhi", ""),
       ("himachal", "Himachal Pradesh", ""), ("jammu", "Jammu", "Jammu & Kashmir"), ("kashmir", "Jammu", ""),
       ("gauhati", "Assam", "also Nagaland, Mizoram, Arunachal Pradesh"), ("tripura", "Tripura", ""),
       ("manipur", "Manipur", ""), ("meghalaya", "Meghalaya", ""), ("sikkim", "Sikkim", "")]
norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
ncode = lambda x: re.sub(r"[^0-9]+", "_", str(x)).strip("_")

states = pd.read_csv(RAW / "districts.csv").state.unique()
def to_census(name):
    hits = [s for s in states if norm(name) in norm(s) or norm(s) in norm(name)]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else "")

co = pd.read_csv(INT / "court_orders_exam.csv", low_memory=False, dtype={"court_code": str})
codes = sorted(set(co.court_code.dropna().map(ncode)) - {"SC", ""})
rows = []
if "court" in co.columns:                       # court names are in the data itself
    names = co.dropna(subset=["court_code"]).assign(code_n=co.court_code.map(ncode)).groupby("code_n").court.first()
    for code, nm in names.items():
        low = str(nm).lower()
        hit = next(((st, note) for k, st, note in KEY if k in low), ("", ""))
        rows.append({"court_code": code, "court_name": nm, "state": to_census(hit[0]) if hit[0] else "", "note": hit[1]})
    print(f"[c04b] mapped from court names in your data: {sum(r['state'] != '' for r in rows)} of {len(rows)}")
if rows:
    pass
else:
  try:
    r = get(URL, timeout=60)
    hc = pd.read_csv(io.StringIO(r.text), dtype=str)
    print(f"[c04b] court list columns: {list(hc.columns)}")
    code_col = next(c for c in hc.columns if "code" in c.lower())
    name_col = next(c for c in hc.columns if "name" in c.lower() or "court" in c.lower() and c != code_col)
    hc["code_n"] = hc[code_col].map(ncode)
    for _, h in hc.iterrows():
        nm = str(h[name_col]).lower()
        hit = next(((st, note) for k, st, note in KEY if k in nm), ("", ""))
        rows.append({"court_code": h.code_n, "court_name": h[name_col], "state": to_census(hit[0]) if hit[0] else "",
                     "note": hit[1]})
  except Exception as ex:
    print(f"[c04b] could not read the court list ({str(ex)[:120]}); writing codes for hand mapping")
xw = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["court_code", "court_name", "state", "note"])
missing = [c for c in codes if c not in set(xw.court_code)]
xw = pd.concat([xw, pd.DataFrame({"court_code": missing, "court_name": "", "state": "", "note": "not in court list"})])
out = RAW / "crosswalks" / "court_state.csv"
out.parent.mkdir(parents=True, exist_ok=True)
xw.to_csv(out, index=False)
unm = xw[(xw.state == "") & xw.court_code.isin(codes)]
print(f"[c04b] {len(codes)} court codes in your data; {len(unm)} without a state -> fill by hand in {out}")
if len(unm):
    print(unm.to_string(index=False))
