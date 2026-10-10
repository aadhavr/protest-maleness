"""Download ACLED events for India (protests + riots) through the OAuth API.

Auth: POST https://acleddata.com/oauth/token (password grant, client_id=acled);
data: GET https://acleddata.com/api/acled/read with a Bearer token.
Pulls one calendar month at a time and pages through each month,
so no single call hits the 5,000-row default.
Output: raw/acled/acled_{YYYY-MM}.csv  (skips months already on disk;
        re-run with --refresh to re-pull, because ACLED revises events)
"""
import io
import sys
import pandas as pd
import requests
from _common import get, need_env
from config import RAW, START_DATE, END_DATE, ACLED_COUNTRIES, ACLED_EVENT_TYPES

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
PAGE = 5000

user, pw = need_env("ACLED_USERNAME", "ACLED_PASSWORD")
tok = requests.post(TOKEN_URL, data={"username": user, "password": pw, "grant_type": "password",
                                     "client_id": "acled", "scope": "authenticated"}, timeout=60)
if tok.status_code != 200:
    raise SystemExit(f"ACLED token request failed: {tok.status_code} {tok.text[:300]}")
auth = {"Authorization": f"Bearer {tok.json()['access_token']}"}

import os
# Comparison countries (descriptive only): PM_ACLED_COUNTRY=Bangladesh python3 c01_acled_api.py
_cty = os.environ.get("PM_ACLED_COUNTRY")
if _cty:
    ACLED_COUNTRIES = [_cty]
out = RAW / ("acled" if not _cty else f"acled_{_cty.lower().replace(' ', '_')}")
out.mkdir(parents=True, exist_ok=True)
refresh = "--refresh" in sys.argv
months = pd.period_range(START_DATE, END_DATE, freq="M")
for m in months:
    f = out / f"acled_{m}.csv"
    if f.exists() and not refresh:
        continue
    lo, hi = m.start_time.date(), min(m.end_time.date(), pd.Timestamp(END_DATE).date())
    frames = []
    # One request series per country and event type. The ":OR:" syntax was
    # dropped in testing (June 2024 returned Riots only), so do not use it.
    for country in ACLED_COUNTRIES:
        for etype in ACLED_EVENT_TYPES:
            page = 1
            while True:
                params = {
                    "_format": "csv", "country": country, "event_type": etype,
                    "event_date": f"{lo}|{hi}", "event_date_where": "BETWEEN",
                    "limit": PAGE, "page": page,
                }
                r = get(READ_URL, params=params, headers=auth)
                if r.status_code != 200:
                    raise SystemExit(f"ACLED read failed {m} {etype} p{page}: {r.status_code} {r.text[:300]}")
                txt = r.text.lstrip("\ufeff").strip()
                if not txt:
                    break
                try:
                    d = pd.read_csv(io.StringIO(txt))
                except pd.errors.EmptyDataError:
                    break
                if d.empty:
                    break
                frames.append(d)
                if len(d) < PAGE:
                    break
                page += 1
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out_df):
        out_df = out_df.drop_duplicates("event_id_cnty")
        counts = out_df.event_type.value_counts().to_dict()
        missing = [e for e in ACLED_EVENT_TYPES if e not in counts]
        if missing:
            print(f"[c01] WARNING {m}: no rows for {missing}; check the request")
    else:
        # Do not save an empty month: a later run (for example after an
        # access upgrade) must download it again instead of skipping it.
        print(f"[c01] {m}: 0 events returned; not saved. If later months are also "
              "empty, your access level probably has a time lag.")
        continue
    out_df.to_csv(f, index=False)
    print(f"[c01] {m}: {len(out_df)} events {counts}")
