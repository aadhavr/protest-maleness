"""Shared settings for the protest-maleness pipeline.

Set PM_DATA_ROOT to point the whole pipeline at another data folder
(the smoke test uses this so fake data never mixes with real data).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PM_DATA_ROOT", ROOT / "data"))
RAW = DATA / "raw"          # source files: collectors (collect/) and you write here
INT = DATA / "interim"      # built by scripts
OUT = DATA / "output"
FIG = OUT / "figures"
for p in (INT, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

# ---- Event-study window (weeks relative to the shock week) ----
WINDOW_PRE = int(os.environ.get("PM_WINDOW_PRE", 8))
WINDOW_POST = int(os.environ.get("PM_WINDOW_POST", 8))
REF_K = -1                  # omitted event time in the regression
# Reported coefficients are re-centred on the MEAN of the pre-period
# (k = -WINDOW_PRE .. -1), so one unusual week cannot move all of them.
REF_MODE = "pre_mean"       # or "single" (relative to REF_K only)
# Treatment varies by state when applicant counts are state totals: cluster by state.
CLUSTER = os.environ.get("PM_CLUSTER", "state")
# Which date defines week 0. "shock_date" = official cancellation announcement.
# Protests often START before the cancellation (they demand it), so also run
# with "exam_date" (or "leak_reported_date"), whose timing is not driven by protest.
EVENT_DATE = os.environ.get("PM_EVENT_DATE", "shock_date")

# Shock types used in the main estimates vs. placebo estimates.
# These must match the shock_type column in raw/exam_shocks.csv.
MAIN_SHOCKS = ["cancelled", "voided"]
PLACEBO_SHOCKS = ["leak_no_cancel", "result_no_leak"]
# Precautionary postponements (e.g. NEET-PG, CSIR-NET June 2024). Not in the main
# estimates; use them in a robustness check by adding "postponed" to MAIN_SHOCKS.
OTHER_SHOCKS = ["postponed"]

# Other shocks often fall inside a stack's window (national exams expose
# almost every district). Default: keep all districts and control for
# exposure to OTHER main shocks in their own post period (other_post_exp).
# Strict option: drop a district from stack e if another main shock that
# exposed it falls inside stack e's window. Report both in the post.
DROP_OVERLAP = False

# Time fixed effects. "state_week" compares districts within a state
# (needs district-level applicant data). "week" compares across states
# (use when applicants are only known by state; see 03_exposure.py).
FE_TIME = "week"
# State exams: add all other districts (zero exposure) as controls. Needed when
# applicant counts exist only at state level: then the exam's state is compared
# with the rest of India, which requires FE_TIME = "week".
STATE_EXAM_CONTROLS = True

# Exposure = applicants per EXPOSURE_SCALE residents aged 15-29.
EXPOSURE_SCALE = 1_000

# ---- ACLED ----
ACLED_COUNTRIES = ["India"]
ACLED_EVENT_TYPES = ["Protests", "Riots"]
# Regex terms that flag an ACLED event as exam/recruitment related.
# These are search terms, not findings. Hand-check a sample of hits
# and misses (02_acled.py writes one) before you trust the flag.
EXAM_REGEX = (
    r"\bexam|paper leak|question paper|\bNEET\b|\bNTA\b|recruitment|"
    r"\bSSC\b|aspirant|entrance test|competitive test|job seeker|unemploy"
)
CLASH_SUBTYPES = [
    "Protest with intervention",
    "Excessive force against protesters",
    "Violent demonstration",
    "Mob violence",
]

# ---- Google Trends (07_trends_pull.py) ----
# Fill after validation. Max 4 terms per batch (the anchor takes slot 5).
# Trends terms by group. "manosphere", "mra" and "gender_jobs" were fixed before any result.
# "hindi" was added AFTER the first (null) result: report it as a second, added test.
# "control" is a POSITIVE CONTROL: exam-leak searches should rise in the exposed state;
# if they do not, the Trends design cannot detect state-level responses at all.
TRENDS_GROUPS: dict[str, list[str]] = {
    "manosphere": ["andrew tate", "sigma male", "red pill", "alpha male"],
    "mra": ["498a", "dowry case", "false dowry case", "men's rights"],
    "gender_jobs": ["women reservation", "mahila aarakshan", "feminism"],
    "hindi": ["पुरुष आयोग", "दहेज कानून", "महिला आरक्षण", "फर्जी दहेज केस"],
    "control": ["paper leak", "पेपर लीक", "exam cancelled", "sarkari naukri"],
}
TRENDS_TERMS: list[str] = [t for g in TRENDS_GROUPS.values() for t in g]
TRENDS_PRIMARY = ["manosphere", "mra", "gender_jobs"]     # groups in the original trends_index
TRENDS_ANCHOR = os.environ.get("PM_TRENDS_ANCHOR") or "weather"   # an empty variable falls back to "weather"          # one stable, high-volume term used in every request
TRENDS_TIMEFRAME = None     # None = START_DATE to END_DATE (weekly values)
TRENDS_PULLS = 3            # independent pulls to average (Trends samples)
TRENDS_SLEEP_S = 20

# ======================================================================
# Collection settings (collect/ scripts). Secrets come from environment
# variables only; never write them into this file.
#   ACLED_USERNAME, ACLED_PASSWORD         (myACLED account)
#   YOUTUBE_API_KEY                        (Google Cloud, YouTube Data API v3)
#   REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
#   ANTHROPIC_API_KEY                      (classifier)
# ======================================================================
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"   # ACLED data received by email to 2 Oct 2026

# News search for CANDIDATE exam shocks (GDELT). Candidates are not data:
# every candidate must be checked against an official notice or two
# independent news reports before it enters raw/exam_shocks.csv.
# Do NOT add shocks that you found through protest data (ACLED): that
# selects shocks on the outcome.
GDELT_QUERIES = [
    '("paper leak" OR "question paper leaked") (exam OR recruitment) sourcecountry:india',
    '("exam cancelled" OR "exam canceled" OR "exam postponed") (leak OR irregularities) sourcecountry:india',
    '("recruitment exam" OR "entrance exam") (cancelled OR scrapped OR voided) sourcecountry:india',
]

# Court orders: keyword filter on metadata text (case-insensitive regex).
COURT_REGEX = (
    r"paper leak|question paper|\bNEET\b|National Testing Agency|recruitment exam|"
    r"examination.{0,40}cancel|re-?exam|re-?test|public service commission|"
    r"staff selection|recruitment board"
)
COURT_YEARS = [2023, 2024, 2025, 2026]
# Date that places a court case in time. Filing (registration) is closer to the
# moment of "voice" than the decision date. Falls back to decision_date if missing.
COURT_DATE = "date_of_registration"
# Stricter court outcome: the RESPONDENT is an exam or recruitment body.
EXAM_BODY_REGEX = (r"public service commission|\bPSC\b|staff selection|selection (?:board|commission)|"
                   r"recruitment (?:board|commission)|National Testing Agency|\bNTA\b|"
                   r"examination (?:board|council)|\b[A-Z]{2,6}SSC\b|\bCSBC\b|\bUPPRPB\b|\bJSSC\b|\bGPSSB\b")

# Online framing. Search inputs only; check each subreddit exists.
YOUTUBE_QUERIES = ["paper leak protest", "NEET protest", "cockroach janta party",
                   "exam cancelled students protest", "berozgari protest",
                   # added for Part 3 (Hindi and exam-specific)
                   "पेपर लीक प्रदर्शन", "भर्ती परीक्षा रद्द", "पुलिस भर्ती पेपर लीक", "शिक्षक भर्ती प्रदर्शन",
                   "RO ARO paper leak", "UP police constable paper leak", "BPSC TRE paper leak",
                   "JSSC CGL protest", "Bihar constable paper leak", "Gujarat junior clerk paper leak"]
YOUTUBE_MAX_VIDEOS_PER_QUERY = 50          # search.list costs 100 quota units/call
YOUTUBE_MAX_COMMENTS_PER_VIDEO = 500
REDDIT_SUBREDDITS = ["india", "IndiaSpeaks", "indiadiscussion", "JEENEETards",
                     "IndianMenRights", "unitedstatesofindia"]
REDDIT_QUERIES = ["paper leak", "exam cancelled", "NEET", "cockroach", "unemployment",
                  "reservation women", "government job"]

# Photos: Wikimedia Commons only (free licences, clear provenance).
# Wire-agency photos (AP, Reuters) must be coded by hand from their sites
# and never redistributed.
COMMONS_QUERIES = ["paper leak protest India", "NEET protest", "Cockroach Janta Party",
                   "Jantar Mantar protest 2026", "student protest India 2024",
                   "Bangladesh quota reform movement"]

# LLM classifier (c09). Validate against your hand labels (a14) before use.
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_BATCH = 20

# Coaching hubs (Layer 3). Map each to your district_id in raw/hubs.csv.
HUB_NAMES = ["Kota", "Prayagraj", "Patna", "Mukherjee Nagar (Delhi)", "Indore",
             "Jaipur", "Lucknow", "Hyderabad"]

# Mechanism covariates (from Census 2011 C-13, see collect/c03).
# Cohort ages IN 2011 that correspond to ages 15-29 / 20-34 in 2026.
YOUTH_AGES_2011 = (0, 14)        # aged ~15-29 in 2026 (no mortality/migration adj.)
MARRIAGE_AGES_2011 = (5, 19)     # aged ~20-34 in 2026
