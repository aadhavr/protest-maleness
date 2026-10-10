"""Weekly Google Trends by Indian state: anchor term in every request, repeated pulls.

pytrends is archived and fails with HTTP 429, so this script calls the two
endpoints that the Trends website uses, with a cookie warm-up:
  1. /trends/api/explore            -> widget token for the time series
  2. /trends/api/widgetdata/multiline -> the weekly values
Both responses start with an anti-XSSI prefix that is removed before parsing.
These are internal endpoints: Google can change them. If every request fails,
use Google's official Trends API (gated alpha) or the website's CSV download.

Why the anchor: Trends rescales each request to 0-100, so values are comparable
across states only as term / anchor inside the same request.
Why repeated pulls: Trends samples searches; averaging pulls reduces noise.
Resumable: finished files are skipped. Slow by design (waits to avoid blocks).
Output: raw/trends/pull{i}_{geo}_{batch}.csv, raw/crosswalks/trends_states.csv
Then run 07_trends_build.py.
"""
import json
import random
import re
import time
import pandas as pd
import requests
from _common import UA
from config import (RAW, START_DATE, END_DATE, TRENDS_TERMS, TRENDS_ANCHOR, TRENDS_TIMEFRAME,
                    TRENDS_PULLS, TRENDS_SLEEP_S, TRENDS_GROUPS)

if not TRENDS_TERMS or not TRENDS_ANCHOR:
    raise SystemExit("set TRENDS_TERMS and TRENDS_ANCHOR in config.py")
TF = TRENDS_TIMEFRAME or f"{START_DATE} {END_DATE}"
GEOS = {"IN-AP": "Andhra Pradesh", "IN-TG": "Andhra Pradesh", "IN-AS": "Assam", "IN-BR": "Bihar",
        "IN-CT": "Chhattisgarh", "IN-DL": "Delhi", "IN-GA": "Goa", "IN-GJ": "Gujarat", "IN-HR": "Haryana",
        "IN-HP": "Himachal Pradesh", "IN-JH": "Jharkhand", "IN-JK": "Jammu", "IN-KA": "Karnataka",
        "IN-KL": "Kerala", "IN-MP": "Madhya Pradesh", "IN-MH": "Maharashtra", "IN-OR": "Odisha",
        "IN-PB": "Punjab", "IN-RJ": "Rajasthan", "IN-TN": "Tamil Nadu", "IN-UP": "Uttar Pradesh",
        "IN-UT": "Uttarakhand", "IN-WB": "West Bengal"}

# crosswalk geo -> 2011 state name as written in districts.csv
norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
states = pd.read_csv(RAW / "districts.csv").state.unique()
def census(name):
    hits = [s for s in states if norm(name) in norm(s) or norm(s) in norm(name)]
    return hits[0] if hits else ""
xw = pd.DataFrame({"geo": list(GEOS), "state": [census(v) for v in GEOS.values()]})
(RAW / "crosswalks").mkdir(parents=True, exist_ok=True)
xw.to_csv(RAW / "crosswalks" / "trends_states.csv", index=False)
if (xw.state == "").any():
    print(f"[c08] geo codes without a census state (fix trends_states.csv): {xw[xw.state == ''].geo.tolist()}")

out = RAW / "trends"
out.mkdir(parents=True, exist_ok=True)
# one request per term group (max 4 terms + anchor). Files are named by group; files from the
# first run (named by batch number 0, 1, 2) count as done for the first three groups.
batches = []
for gname, terms in TRENDS_GROUPS.items():
    for j in range(0, len(terms), 4):
        batches.append((gname if j == 0 else f"{gname}{j // 4}", terms[j:j + 4]))
OLD = {name: k for k, name in enumerate(list(TRENDS_GROUPS)[:3])}
HL, TZ = "en-US", "-330"


def strip(txt):
    return json.loads(txt[txt.index("{"):] if txt.lstrip().startswith(")") else txt)


def new_session():
    s = requests.Session()
    s.headers.update({**UA, "Accept-Language": "en-US,en;q=0.9"})
    try:
        s.get("https://trends.google.com/trends/explore?geo=IN", timeout=30)   # cookie warm-up
    except requests.RequestException:
        pass
    return s


def fetch(s, terms, geo):
    req = {"comparisonItem": [{"keyword": k, "geo": geo, "time": TF} for k in terms],
           "category": 0, "property": ""}
    r = s.get("https://trends.google.com/trends/api/explore",
              params={"hl": HL, "tz": TZ, "req": json.dumps(req)}, timeout=60)
    if r.status_code != 200:
        return None, r.status_code
    w = next(w for w in strip(r.text)["widgets"] if w["id"] == "TIMESERIES")
    r2 = s.get("https://trends.google.com/trends/api/widgetdata/multiline",
               params={"hl": HL, "tz": TZ, "req": json.dumps(w["request"]), "token": w["token"]}, timeout=60)
    if r2.status_code != 200:
        return None, r2.status_code
    rows = strip(r2.text)["default"]["timelineData"]
    df = pd.DataFrame({"date": pd.to_datetime([int(x["time"]) for x in rows], unit="s"),
                       "isPartial": [bool(x.get("isPartial", False)) for x in rows]})
    for j, k in enumerate(terms):
        df[k] = [x["value"][j] for x in rows]
    return df, 200


s = new_session()
def done(i, g, b):
    return (out / f"pull{i}_{g}_{b}.csv").exists() or (b in OLD and (out / f"pull{i}_{g}_{OLD[b]}.csv").exists())


todo = [(i, g, b, t) for i in range(TRENDS_PULLS) for g in GEOS for b, t in batches if not done(i, g, b)]
print(f"[c08] {len(todo)} requests to do ({TRENDS_PULLS} pulls x {len(GEOS)} states x {len(batches)} batches); "
      f"about {len(todo) * (TRENDS_SLEEP_S + 5) / 3600:.1f} h at the current pace")
fails = 0
for n, (i, g, b, terms) in enumerate(todo, 1):
    for attempt in range(4):
        try:
            df, code = fetch(s, terms + [TRENDS_ANCHOR], g)
        except (requests.RequestException, StopIteration, KeyError, ValueError) as ex:
            df, code = None, str(ex)[:60]
        if df is not None:
            break
        wait = 60 * (attempt + 1) * (5 if code == 429 else 1)
        print(f"  {g} batch {b} pull {i}: {code}; new session, waiting {wait} s")
        time.sleep(wait)
        s = new_session()
    if df is None:
        fails += 1
        print(f"  {g} batch {b} pull {i}: FAILED (re-run later to retry)")
        if fails >= 5 and fails == n:
            raise SystemExit("[c08] every request fails: Google is blocking this IP or changed the endpoints")
        continue
    df.assign(state=g, pull=i, batch=b, pulled_at=pd.Timestamp.now("UTC")).to_csv(out / f"pull{i}_{g}_{b}.csv", index=False)
    if n % 10 == 0:
        print(f"  {n}/{len(todo)} done")
    time.sleep(TRENDS_SLEEP_S + random.uniform(0, 10))
print("[c08] finished. Run 07_trends_build.py")
