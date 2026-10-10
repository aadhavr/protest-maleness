"""FAKE data for a smoke test only. Nothing here is real.
Writes to tests/fake_data/raw. Run the pipeline with
PM_DATA_ROOT=tests/fake_data so outputs never mix with real ones.
A known effect is planted: male exposure raises exam protests, female does not."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(__file__).parent / "fake_data" / "raw"
for sub in ["acled", "applicants", "crosswalks"]:
    (root / sub).mkdir(parents=True, exist_ok=True)

states = [f"S{i}" for i in range(6)]
dist = pd.DataFrame({"district_id": [f"D{i:03d}" for i in range(240)]})
dist["district_name"] = "Fake " + dist.district_id
dist["state"] = [states[i % 6] for i in range(240)]
dist["pop_15_29"] = rng.integers(100_000, 900_000, 240)
dist["pop_source"] = "FAKE"
dist.to_csv(root / "districts.csv", index=False)

weeks = pd.date_range("2023-01-02", "2026-06-29", freq="W-MON")
exams = []
for j in range(14):
    lvl = "national" if j < 8 else "state"
    T = weeks[10 + j * 12]
    exams.append(dict(exam_id=f"E{j:02d}", exam_name=f"Fake exam {j}", conducting_body="FAKE",
                      level=lvl, state="" if lvl == "national" else states[j % 6],
                      exam_date=str((T - pd.Timedelta(days=20)).date()), leak_reported_date="",
                      shock_date=str((T + pd.Timedelta(days=2)).date()),
                      shock_type="cancelled" if j < 12 else "leak_no_cancel",
                      source_url="https://example.invalid/fake", notes="FAKE",
                      male_share=rng.uniform(0.25, 0.85), T=T))
pd.DataFrame(exams).drop(columns=["male_share", "T"]).to_csv(root / "exam_shocks.csv", index=False)

rows, lam = [], {}
for e in exams:
    ds = dist if e["level"] == "national" else dist[dist.state == e["state"]]
    for d in ds.itertuples():
        tot = int(rng.integers(200, 6000))
        m = int(rng.binomial(tot, e["male_share"]))
        rows += [(e["exam_id"], d.district_id, "M", m, "https://example.invalid/fake"),
                 (e["exam_id"], d.district_id, "F", tot - m, "https://example.invalid/fake")]
        if e["shock_type"] == "cancelled":
            for k in range(4):
                w = e["T"] + pd.Timedelta(weeks=k)
                lam[(d.district_id, w)] = lam.get((d.district_id, w), 0) + 40 * m / d.pop_15_29
pd.DataFrame(rows, columns=["exam_id", "district_id", "sex", "applicants", "source_url"]).to_csv(
    root / "applicants" / "fake.csv", index=False)

ev = []
for d in dist.itertuples():
    for w in weeks:
        for _ in range(rng.poisson(0.05 + lam.get((d.district_id, w), 0))):
            ev.append(dict(event_id_cnty=f"IND{len(ev)}",
                           event_date=str((w + pd.Timedelta(days=int(rng.integers(0, 7)))).date()),
                           event_type="Protests",
                           sub_event_type=rng.choice(["Peaceful protest", "Protest with intervention"]),
                           country="India", admin1=d.state, admin2=d.district_name,
                           assoc_actor_1=rng.choice(["Students (India)", "Women (India)", ""]),
                           notes=rng.choice(["Students protested the paper leak and exam cancellation.",
                                             "Farmers protested crop prices."])))
pd.DataFrame(ev).to_csv(root / "acled" / "fake.csv", index=False)
dist.assign(acled_admin1=dist.state, acled_admin2=dist.district_name)[
    ["acled_admin1", "acled_admin2", "district_id"]].to_csv(root / "crosswalks" / "acled_districts.csv", index=False)
print(f"FAKE data written to {root}: {len(ev)} events")

# ---------------- extra FAKE inputs for Layers 2-5 and mechanisms ----------------
interim = root.parent / "interim"
output = root.parent / "output"
for p in (interim, output, root / "trends"):
    p.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"district_id": dist.district_id,
              "cohort_sex_ratio": rng.normal(1.08, 0.04, 240),
              "female_work_rate": rng.uniform(0.1, 0.4, 240)}).to_csv(root / "district_covariates.csv", index=False)
pd.DataFrame({"district_id": dist.district_id.iloc[:12], "hub_name": "FAKE"}).to_csv(root / "hubs.csv", index=False)

# photos: pools' male share + a planted +0.15 visibility gap
ph = []
for e in exams[:12]:
    for i in range(15):
        n = int(rng.integers(5, 40))
        m = int(rng.binomial(n, min(0.97, e["male_share"] + 0.15)))
        ph.append(dict(url=f"https://example.invalid/{e['exam_id']}_{i}.jpg", keep=1, exam_id=e["exam_id"],
                       n_men=m, n_women=n - m, n_unclear=int(rng.integers(0, 4)), coder="A"))
        if i < 3:
            ph.append({**ph[-1], "coder": "B", "n_men": max(0, m + int(rng.integers(-1, 2)))})
pd.DataFrame(ph).to_csv(output / "photo_coding_sheet.csv", index=False)

# comments: blame rises after male-heavy shocks
com = []
for e in exams[:8]:
    for k in range(-8, 9):
        w = e["T"] + pd.Timedelta(weeks=k)
        pb = 0.05 + (0.12 * (e["male_share"] - 0.3) if 0 <= k <= 3 else 0)
        for i in range(60):
            com.append(dict(id=f"c{len(com)}", created=str(w + pd.Timedelta(days=int(rng.integers(0, 7)))),
                            text_in="FAKE", gendered_blame=int(rng.random() < pb),
                            manosphere_frame=int(rng.random() < 0.03)))
pd.DataFrame(com).to_csv(interim / "comments_labeled.csv", index=False)

# trends pulls: 3 pulls, anchor + 2 terms, 6 fake state geos
geos = [f"IN-F{i}" for i in range(6)]
pd.DataFrame({"geo": geos, "state": states}).to_csv(root / "crosswalks" / "trends_states.csv", index=False)
for i in range(3):
    for g in geos:
        n = len(weeks)
        pd.DataFrame({"date": weeks, "termA": rng.integers(0, 100, n), "termB": rng.integers(0, 100, n),
                      "ANCHOR": rng.integers(20, 100, n), "isPartial": False, "state": g, "pull": i,
                      "batch": 0, "pulled_at": "FAKE"}).to_csv(root / "trends" / f"pull{i}_{g}_0.csv", index=False)

# court orders with petitioner sex
co = pd.DataFrame({"court_code": rng.choice([f"C{i}" for i in range(6)], 800),
                   "decision_date": [str(pd.Timestamp(d).date()) for d in rng.choice(weeks, 800)],
                   "title_text": "FAKE vs FAKE",
                   "petitioner_sex": rng.choice(["female", "male", "unknown"], 800)})
co.to_csv(interim / "court_orders_labeled.csv", index=False)
pd.DataFrame({"court_code": [f"C{i}" for i in range(6)], "state": states}).to_csv(
    root / "crosswalks" / "court_state.csv", index=False)

pd.DataFrame([dict(state=states[i % 6], start_date=str(weeks[5 + 7 * i].date()), end_date=str((weeks[5 + 7 * i] + pd.Timedelta(days=1)).date()),
                   reason="exam" if i % 3 else "protest", source_url="https://example.invalid/fake") for i in range(20)]
             ).to_csv(root / "shutdowns.csv", index=False)
print("FAKE extras written")
