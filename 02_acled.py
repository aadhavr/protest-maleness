"""Clean ACLED exports into a district-week protest panel.

Input : raw/acled/*.csv                       (ACLED export, any number of files)
        raw/crosswalks/acled_districts.csv    (acled_admin1, acled_admin2, district_id)
Output: interim/acled_events.csv              (event level, with flags)
        interim/acled_district_week.csv       (district x week counts)
        output/acled_unmatched.csv            (names to add to the crosswalk)
        output/acled_exam_flag_audit.csv      (sample to hand-check the regex)
"""
import difflib
import re
import pandas as pd
from config import (RAW, INT, OUT, ACLED_COUNTRIES, ACLED_EVENT_TYPES,
                    EXAM_REGEX, CLASH_SUBTYPES)
from lib import week_start, require_cols

files = sorted((RAW / "acled").glob("*.csv"))
if not files:
    raise SystemExit("put ACLED export CSVs in raw/acled/")
ev = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
require_cols(ev, ["event_id_cnty", "event_date", "event_type", "sub_event_type",
                  "country", "admin1", "admin2", "notes"], "ACLED export")
ev = ev.drop_duplicates("event_id_cnty")
ev = ev[ev.country.isin(ACLED_COUNTRIES) & ev.event_type.isin(ACLED_EVENT_TYPES)].copy()

ev["week"] = week_start(ev.event_date)
notes = ev.notes.fillna("")
ev["exam_related"] = notes.str.contains(EXAM_REGEX, case=False, regex=True)
ev["peaceful"] = ev.sub_event_type.eq("Peaceful protest")
ev["clash"] = ev.sub_event_type.isin(CLASH_SUBTYPES)
actor_cols = [c for c in ["assoc_actor_1", "assoc_actor_2", "actor1", "actor2"] if c in ev]
actors = ev[actor_cols].fillna("").agg(" ".join, axis=1)
ev["women_tagged"] = actors.str.contains(r"\bWomen\b", case=True, regex=True)
ev["students_tagged"] = actors.str.contains(r"Students", case=True, regex=True)

# ---- district assignment ----
# Preferred: event-level 2011 districts from coordinates (collect/c10_acled_spatial_xwalk.py).
# Fallback: (admin1, admin2) name crosswalk.
ev_xw_f = RAW / "crosswalks" / "acled_event_district.csv"
if ev_xw_f.exists():
    exw = pd.read_csv(ev_xw_f, dtype={"district_id": str})
    ev = ev.merge(exw[["event_id_cnty", "district_id", "method"]], on="event_id_cnty", how="left")
    print(f"[02] district from coordinates: {ev.district_id.notna().mean():.1%} of events assigned "
          f"{ev.method.value_counts().to_dict()}")
    new_ids = ev.district_id.isna() & ~ev.event_id_cnty.isin(exw.event_id_cnty)
    if new_ids.any():
        print(f"[02] {new_ids.sum()} events are newer than the crosswalk; re-run collect/c10_acled_spatial_xwalk.py")
else:
    # ---- district crosswalk ----
    xw_f = RAW / "crosswalks" / "acled_districts.csv"
    if not xw_f.exists():
        # First run: write a starter crosswalk to fill, with suggestions when
        # raw/districts.csv (from collect/c03_census.py) already exists.
        xw_f.parent.mkdir(parents=True, exist_ok=True)
        units = (ev.groupby(["admin1", "admin2"], dropna=False).size().rename("n_events")
                 .reset_index().sort_values("n_events", ascending=False))
        units["district_id"] = ""
        units["suggested_district_id"] = ""
        units["suggested_name"] = ""
        units["match"] = ""
        dfile = RAW / "districts.csv"
        if dfile.exists():
            dist = pd.read_csv(dfile, dtype={"district_id": str})
            norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
            for i, r in units.iterrows():
                cand = dist[dist.state.map(norm) == norm(r.admin1)]
                if cand.empty:
                    cand = dist
                names = cand.district_name.map(norm).tolist()
                exact = cand[cand.district_name.map(norm) == norm(r.admin2)]
                if len(exact) == 1:
                    units.loc[i, ["suggested_district_id", "suggested_name", "match"]] = \
                        [exact.district_id.iloc[0], exact.district_name.iloc[0], "exact"]
                else:
                    close = difflib.get_close_matches(norm(r.admin2), names, n=1, cutoff=0.75)
                    if close:
                        row = cand[cand.district_name.map(norm) == close[0]].iloc[0]
                        units.loc[i, ["suggested_district_id", "suggested_name", "match"]] = \
                            [row.district_id, row.district_name, "fuzzy - CHECK"]
            print(f"[02] suggestions: {(units.match == 'exact').sum()} exact, "
                  f"{units.match.str.startswith('fuzzy').sum()} fuzzy, {(units.match == '').sum()} none")
        else:
            print("[02] raw/districts.csv not found; run collect/c03_census.py first to get suggestions")
        todo = xw_f.with_name("acled_districts_TODO.csv")
        units.rename(columns={"admin1": "acled_admin1", "admin2": "acled_admin2"}).to_csv(todo, index=False)
        ev.to_csv(INT / "acled_events.csv", index=False)
        raise SystemExit(
            f"[02] no crosswalk yet. Wrote {todo.name} ({len(units)} ACLED districts).\n"
            "     Fill district_id (copy exact suggestions; check fuzzy ones), save the file\n"
            "     as acled_districts.csv in the same folder, then run 02_acled.py again.")

    xw = pd.read_csv(xw_f, dtype=str)
    require_cols(xw, ["acled_admin1", "acled_admin2", "district_id"], "acled_districts.csv")
    xw = xw[["acled_admin1", "acled_admin2", "district_id"]].dropna(subset=["district_id"])
    xw = xw[xw.district_id.str.strip() != ""]
    ev = ev.merge(xw, left_on=["admin1", "admin2"], right_on=["acled_admin1", "acled_admin2"],
                  how="left")
    um = (ev[ev.district_id.isna()].groupby(["admin1", "admin2"], dropna=False)
          .size().rename("n_events").reset_index().sort_values("n_events", ascending=False))
    if len(um):
        known = xw.acled_admin2.dropna().unique().tolist()
        um["closest_in_xwalk"] = um.admin2.fillna("").map(
            lambda s: "; ".join(difflib.get_close_matches(s, known, n=3, cutoff=0.6)))
        um.to_csv(OUT / "acled_unmatched.csv", index=False)
        share = um.n_events.sum() / len(ev)
        print(f"[02] {len(um)} admin units unmatched ({share:.1%} of events) -> acled_unmatched.csv")


# ---- regex audit sample: 100 hits and 100 non-hits that mention students ----
aud = pd.concat([
    ev[ev.exam_related].sample(min(100, ev.exam_related.sum()), random_state=1),
    ev[~ev.exam_related & ev.students_tagged].sample(
        min(100, (~ev.exam_related & ev.students_tagged).sum()), random_state=1),
])
aud_f = OUT / "acled_exam_flag_audit.csv"
if not aud_f.exists():          # never overwrite: you may have labeled it
    aud[["event_id_cnty", "event_date", "admin2", "exam_related", "notes"]].assign(
        hand_label="").to_csv(aud_f, index=False)

# ---- classifier labels (collect/c09_classify_llm.py acled), if present ----
lab_f = INT / "acled_exam_labeled.csv"
if lab_f.exists():
    lab = pd.read_csv(lab_f, usecols=["event_id_cnty", "exam_label", "category", "exam_name"])
    ev = ev.drop(columns=[c for c in ["exam_label", "category", "exam_name"] if c in ev]).merge(
        lab, on="event_id_cnty", how="left")
    ev["exam_strict"] = ev.exam_related & (pd.to_numeric(ev.exam_label, errors="coerce") == 1)
    ev["exam_entrance"] = ev.exam_strict & (ev.category == "entrance_exam")
    ev["exam_recruit"] = ev.exam_strict & (ev.category == "recruitment_exam")
    unl = ev.exam_related & ev.exam_label.isna()
    print(f"[02] classifier: {ev.exam_strict.sum()} of {ev.exam_related.sum()} flagged events are exam protests"
          + (f"; {unl.sum()} flagged events not yet labeled (re-run c09 acled)" if unl.any() else ""))
else:
    print("[02] no classifier labels yet; n_exam_strict not built (run collect/c09_classify_llm.py acled)")

import os
DROPPED = bool(os.environ.get("PM_DROP_GEO3")) and "geo_precision" in ev
if os.environ.get("PM_DROP_GEO3") and not DROPPED:
    print("[02] PM_DROP_GEO3 set but there is no geo_precision column; nothing dropped")
if DROPPED:
    n0 = len(ev)
    ev = ev[pd.to_numeric(ev.geo_precision, errors="coerce") != 3]
    print(f"[02] PM_DROP_GEO3: dropped {n0 - len(ev)} approximate-location events; panel -> acled_district_week_geo12.csv")
else:
    ev.to_csv(INT / "acled_events.csv", index=False)
ev = ev.dropna(subset=["district_id"])
# Party-organised events: protests led by political parties or their adult youth wings.
# Student unions (NSUI, ABVP, SFI, AISA, AISF, AIDSO, KSU, ...) are NOT counted as parties.
PARTY = (r"\b(?:INC|Indian National Congress|BJP|Bharatiya Janata|SP|Samajwadi|AAP|Aam Aadmi|BSP|Bahujan Samaj|"
         r"RJD|Rashtriya Janata|JD\(U\)|Janata Dal|CPI|CPI\(M\)|CPI-M|Communist Party|TMC|AITC|Trinamool|DMK|AIADMK|"
         r"Shiv Sena|SS-UBT|NCP|BRS|YSRCP|TDP|Telugu Desam|BJD|Biju Janata|JMM|Jharkhand Mukti|INLD|JJP|RLD|"
         r"IYC|Indian Youth Congress|BJYM|DYFI|Yuva Morcha)\b")
act_cols = [c for c in ["actor1", "assoc_actor_1", "actor2", "assoc_actor_2"] if c in ev]
_acts = ev[act_cols].fillna("").agg(" ".join, axis=1)
ev["party_led"] = _acts.str.contains(PARTY, regex=True)
# Sensitivity: also count party-affiliated STUDENT wings as party-led (broad definition)
STUDENT_WINGS = r"\b(?:NSUI|ABVP|SFI|AISA|AISF|AIDSO|KSU|TMCP|CYSS|SAD Student|INSO|MSF)\b"
ev["party_led_broad"] = ev.party_led | _acts.str.contains(STUDENT_WINGS, regex=True)

flags = {"n_events": None, "n_exam": ev.exam_related, "n_peaceful": ev.peaceful, "n_clash": ev.clash,
         "n_women_tagged": ev.women_tagged,
         "n_exam_peaceful": ev.exam_related & ev.peaceful, "n_exam_clash": ev.exam_related & ev.clash}
if "exam_strict" in ev:
    flags.update({"n_exam_strict": ev.exam_strict,
                  "n_exam_strict_peaceful": ev.exam_strict & ev.peaceful,
                  "n_exam_strict_clash": ev.exam_strict & ev.clash,
                  "n_exam_entrance": ev.exam_entrance, "n_exam_recruit": ev.exam_recruit,
                  "n_exam_strict_nonparty": ev.exam_strict & ~ev.party_led,
                  "n_exam_strict_party": ev.exam_strict & ev.party_led,
                  "n_exam_strict_nonparty_broad": ev.exam_strict & ~ev.party_led_broad})
    print(f"[02] exam protests led by parties or their youth wings: {int((ev.exam_strict & ev.party_led).sum())} "
          f"of {int(ev.exam_strict.sum())}; adding party student wings: {int((ev.exam_strict & ev.party_led_broad).sum())}")
cnt = pd.DataFrame({k: (1 if v is None else v.astype(int)) for k, v in flags.items()}, index=ev.index)
cnt[["district_id", "week"]] = ev[["district_id", "week"]]
panel = cnt.groupby(["district_id", "week"]).sum().reset_index()
panel.to_csv(INT / ("acled_district_week_geo12.csv" if DROPPED else "acled_district_week.csv"), index=False)
print(f"[02] {len(ev)} matched events, {len(panel)} district-weeks with >=1 event")
