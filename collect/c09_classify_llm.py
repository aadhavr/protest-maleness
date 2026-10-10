"""LLM labels for (a) online comments, (b) court-order petitioner sex,
(c) ACLED events: is the protest about an exam?

Usage: python c09_classify_llm.py comments
       python c09_classify_llm.py petitioners
       python c09_classify_llm.py acled     (validates against raw/validation/acled_exam_audit_labeled.csv)
Results are cached (interim/llm_cache.jsonl) so re-runs cost nothing.
The labels are measurements with error. Before any use, hand-label a
random sample (a14_validate_labels.py makes the sheet and scores it) and
report precision and recall. Petitioner sex from names is error-prone:
keep "unknown", and never publish names.
Output: interim/comments_labeled.csv or interim/court_orders_labeled.csv
"""
import hashlib
import json
import re
import sys
import time
import pandas as pd
import requests
from _common import need_env
from config import RAW, INT, OUT, LLM_MODEL, LLM_BATCH

task = sys.argv[1] if len(sys.argv) > 1 else "comments"
(api_key,) = need_env("ANTHROPIC_API_KEY")

SYSTEM = {
"comments": """You label social-media comments about Indian exams, jobs and protests.
Comments may be English, Hindi (Devanagari or romanised) or Hinglish.
Return ONLY a JSON array, one object per input, with keys:
  "id": the input id (copy exactly)
  "gendered_blame": 1 if the comment blames women, women's quotas/reservation, or women's
      employment for men's lost jobs or opportunities; else 0
  "blame_target": one of "women", "gender_quota", "caste_quota", "merit_system",
      "government", "exam_body", "other", "none"
  "manosphere_frame": 1 if the comment uses male-grievance ideology (men as victims of
      women, feminism or laws that favour women; "simp", "alpha/beta", claims that
      dowry or harassment laws are misused against men); else 0
  "language": one of "en", "hi", "hinglish", "other"
Label only what the text says. If unsure, use 0 and "none".""",
"petitioners": """You receive the petitioner field of Indian court cases (one or more names), or a
case title. Classify the FIRST named petitioner/appellant (for a title: the party before "vs",
"v.", "versus").
Return ONLY a JSON array, one object per input, with keys:
  "id": the input id (copy exactly)
  "petitioner_type": "person", "group_of_persons", "organisation", "state", "unknown"
  "petitioner_sex": "female", "male", "mixed", "unknown"  (use "unknown" unless the given
      name makes it clear; do not guess from surnames alone)
Do not return any names.""",
"acled": """You label protest events in India from ACLED event descriptions.
Return ONLY a JSON array, one object per input, with keys:
  "id": the input id (copy exactly)
  "exam_label": 1 if the protest is about a SPECIFIC competitive recruitment exam, entrance or
      licensing exam (NEET, JEE, CUET, NExT, PSC/SSC/board exams, ...), or a SPECIFIC government
      recruitment process for job aspirants: paper leaks, cancellations, irregularities, nepotism
      or scams in a recruitment process, results delays, exam pattern or schedule, appointment
      of selected candidates, waiting lists, eligibility-test holders demanding recruitment.
      Party or student-group protests count. A multi-issue protest counts if one of its named
      demands is such an exam or recruitment-process issue (e.g. "prices, toll tax and NEET
      irregularities" = 1; a protest against a governor that includes the anti-NEET bill = 1).
      Label 0 for: general unemployment, prices or jobs-creation demands; demands to fill
      vacancies or change recruitment POLICY with no specific exam or process at issue (district-
      wise recruitment, contractual hiring system, "recruit teachers permanently"); demands by
      people who ALREADY hold jobs, including contract or guest staff (regularisation, pay, job
      security), even if an organisation name contains "Recruitment Board"; promises of jobs
      as compensation (land, eviction); university, college, semester, school or board exams
      and admissions.
  "category": one of "entrance_exam", "recruitment_exam", "general_unemployment",
      "employee_demands", "university_school_exam", "other"
      (use "entrance_exam" or "recruitment_exam" only when exam_label = 1)
  "exam_name": the specific exam or recruitment body if named (e.g. "NEET-UG 2024",
      "UP Police Constable", "TSPSC Group-1", "JSSC CGL"), else ""
Judge only from the text."""}[task]

cache_f = INT / "llm_cache.jsonl"
cache = {}
if cache_f.exists():
    for line in cache_f.read_text().splitlines():
        rec = json.loads(line)
        cache[rec["key"]] = rec["label"]


PROMPT_V = hashlib.sha1(SYSTEM.encode()).hexdigest()[:8]   # a changed rule = new labels


def key_of(i, text):
    return hashlib.sha1(f"{task}|{LLM_MODEL}|{PROMPT_V}|{i}|{text}".encode()).hexdigest()


def call(batch):
    body = {"model": LLM_MODEL, "max_tokens": 4000, "system": SYSTEM,
            "messages": [{"role": "user", "content": json.dumps(batch, ensure_ascii=False)}]}
    for i in range(5):
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=120,
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"})
        if r.status_code in (429, 500, 529):
            time.sleep(10 * (i + 1))
            continue
        if r.status_code >= 400:          # show the API's reason (e.g. low credit balance, bad request)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        txt = "".join(b.get("text", "") for b in r.json()["content"])
        txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
        return json.loads(txt)
    raise RuntimeError("API failed 5 times")


if task == "comments":
    parts = [pd.read_csv(f) for f in (RAW / "online").glob("*_comments.csv")] + \
            [pd.read_csv(f) for f in (RAW / "online").glob("reddit_items.csv")]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    df["text_in"] = df.text.fillna("").str.slice(0, 1500)
    import os
    if os.environ.get("PM_COMMENT_WINDOWS"):
        # label only comments posted inside the event windows of the shocks (cheaper, and
        # these are the comments the timing test uses)
        from config import WINDOW_PRE, WINDOW_POST
        sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"])
        created = pd.to_datetime(df.created, utc=True, errors="coerce").dt.tz_localize(None)
        keep = pd.Series(False, index=df.index)
        for t0 in sh.shock_week.dropna().unique():
            t0 = pd.Timestamp(t0)
            keep |= created.between(t0 - pd.Timedelta(weeks=WINDOW_PRE), t0 + pd.Timedelta(weeks=WINDOW_POST + 1))
        print(f"[c09] PM_COMMENT_WINDOWS: {int(keep.sum()):,} of {len(df):,} comments fall inside shock windows")
        df = df[keep]
    out_f = INT / "comments_labeled.csv"
elif task == "acled":
    ev = pd.read_csv(INT / "acled_events.csv", low_memory=False,
                     usecols=["event_id_cnty", "exam_related", "notes"])
    val_f = RAW / "validation" / "acled_exam_audit_labeled.csv"
    val_ids = set(pd.read_csv(val_f).event_id_cnty) if val_f.exists() else set()
    if not val_ids:
        print("[c09] no validation file at raw/validation/acled_exam_audit_labeled.csv; no score")
    df = ev[(ev.exam_related == True) | ev.event_id_cnty.isin(val_ids)].copy()
    df["id"] = df.event_id_cnty
    df["text_in"] = df.notes.fillna("").str.slice(0, 1500)
    out_f = INT / "acled_exam_labeled.csv"
else:
    df = pd.read_csv(INT / "court_orders_exam.csv", low_memory=False)
    df["id"] = "co_" + df.index.astype(str)
    src = "petitioner" if "petitioner" in df and df.petitioner.notna().mean() > 0.5 else "title_text"
    print(f"[c09] petitioners: using column {src}")
    df["text_in"] = df[src].fillna("").astype(str).str.slice(0, 300)
    out_f = INT / "court_orders_labeled.csv"

df["key"] = [key_of(i, t) for i, t in zip(df.id, df.text_in)]
todo = df[~df.key.isin(cache) & df.text_in.str.strip().ne("")]
print(f"[c09] {task}: {len(df)} items, {len(todo)} to label")
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

WORKERS = int(__import__("os").environ.get("PM_LLM_WORKERS", "6"))
batches = [todo.iloc[s:s + LLM_BATCH] for s in range(0, len(todo), LLM_BATCH)]
lock = threading.Lock()
done_n, t0 = 0, time.time()


def run_batch(b):
    return b, call([{"id": i, "text": t} for i, t in zip(b.id, b.text_in)])


with cache_f.open("a") as fh, ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futs = [pool.submit(run_batch, b) for b in batches]
    for fut in as_completed(futs):
        try:
            b, out = fut.result()
            labels = {x["id"]: x for x in out}
        except Exception as ex:          # bad JSON or API failure: skip, re-run later
            print(f"  a batch failed ({str(ex)[:100]}); re-run later to fill it")
            continue
        with lock:
            for i, k in zip(b.id, b.key):
                if i in labels:
                    cache[k] = labels[i]
                    fh.write(json.dumps({"key": k, "label": labels[i]}, ensure_ascii=False) + "\n")
            fh.flush()                    # saved now: stopping the script loses nothing
            done_n += len(b)
            rate = done_n / max(time.time() - t0, 1e-9)
            print(f"  {done_n}/{len(todo)}  ({rate:.1f} items/s, about "
                  f"{(len(todo) - done_n) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)

lab = pd.DataFrame([{**cache[k], "key": k} for k in df.key if k in cache]).drop(columns="id", errors="ignore")
res = df.merge(lab, on="key", how="left").drop(columns=["key"])
res.to_csv(out_f, index=False)
print(f"[c09] -> {out_f}  (unlabeled: {res.iloc[:, -1].isna().sum()})")

# ---- validation for the ACLED task ----
if task == "acled" and val_ids:
    v = pd.read_csv(val_f)[["event_id_cnty", "hand_label"]].merge(
        res[["event_id_cnty", "exam_label", "category", "exam_name", "text_in"]], on="event_id_cnty")
    v = v.dropna(subset=["hand_label", "exam_label"])
    h, p = v.hand_label.astype(int), v.exam_label.astype(int)
    tp, fp, fn, tn = ((p == 1) & (h == 1)).sum(), ((p == 1) & (h == 0)).sum(), ((p == 0) & (h == 1)).sum(), ((p == 0) & (h == 0)).sum()
    n = len(v)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    print(f"[c09] validation on {n} hand-labeled events: accuracy {po:.3f}, precision {tp / max(tp + fp, 1):.3f}, "
          f"recall {tp / max(tp + fn, 1):.3f}, kappa {(po - pe) / (1 - pe):.3f}")
    print(f"      confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    dis = v[h != p]
    dis.to_csv(OUT / "acled_classifier_disagreements.csv", index=False)
    print(f"      {len(dis)} disagreements -> output/acled_classifier_disagreements.csv (read them: "
          "some may be errors in the hand labels)")
