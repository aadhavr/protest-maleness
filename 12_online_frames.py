"""Layer 5: does gendered blame in online comments rise after exam leaks?

National weekly series (YouTube comments; Reddit if collected): share of labeled
comments with gendered_blame = 1 and with manosphere_frame = 1.
Sex data exist for too few shocks, so this version asks a simpler question:
across ALL main shocks (state and national), is the share higher in the weeks
after the exam than before?  Statistic: mean over shocks of
[mean share, weeks 0..+POST] - [mean share, weeks -PRE..-1].
Inference: randomisation over calendar shifts of all shock dates together (the comment series has no
geography, so only time variation exists). Weeks with < MIN_N comments are dropped.
CAUTION: comments come from videos found by protest searches, so comment volume
itself clusters around protests; the share (not the count) is the outcome.
Output: output/online_weekly.csv, output/online_event.csv, figures/online_event.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import INT, OUT, FIG, WINDOW_PRE, WINDOW_POST, MAIN_SHOCKS, START_DATE, END_DATE
from lib import week_start, blog_style

MIN_N, REPS = 20, 2000
c = pd.read_csv(INT / "comments_labeled.csv", low_memory=False)
for col in ["gendered_blame", "manosphere_frame"]:
    c[col] = pd.to_numeric(c[col], errors="coerce")
c = c.dropna(subset=["gendered_blame"])
c["week"] = week_start(pd.to_datetime(c.created, utc=True, errors="coerce").dt.tz_localize(None))
wk = c.groupby("week").agg(n=("id", "size"), blame=("gendered_blame", "mean"), mano=("manosphere_frame", "mean"))
wk = wk[(wk.n >= MIN_N) & (wk.index >= pd.Timestamp(START_DATE) - pd.Timedelta(days=6)) & (wk.index <= pd.Timestamp(END_DATE))]
wk.to_csv(OUT / "online_weekly.csv")
print(f"[12] {len(c):,} labeled comments; {len(wk)} weeks with >= {MIN_N} comments; "
      f"overall blame share {c.gendered_blame.mean():.3%}, manosphere share {c.manosphere_frame.mean():.3%}")

sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"])
T = sorted(sh[sh.shock_type.isin(MAIN_SHOCKS)].shock_week.dropna().unique())
ks = np.arange(-WINDOW_PRE, WINDOW_POST + 1)


def stat(weeks0, y):
    vals = []
    for t0 in weeks0:
        s = wk[y].reindex([pd.Timestamp(t0) + pd.Timedelta(weeks=int(k)) for k in ks])
        s.index = ks
        pre, post = s[s.index < 0].dropna(), s[s.index >= 0].dropna()
        if len(pre) >= 3 and len(post) >= 3:
            vals.append(post.mean() - pre.mean())
    return np.mean(vals) if vals else np.nan, len(vals)


# False dates: shift ALL shock dates by the same random number of weeks (circularly inside
# the data period). This keeps the real spacing (and overlap) between shocks, so the false
# estimates have the same dependence structure as the real one.
lo = pd.Timestamp(START_DATE) - pd.Timedelta(days=pd.Timestamp(START_DATE).weekday())
n_weeks = int((pd.Timestamp(END_DATE) - lo).days // 7) + 1
span = WINDOW_PRE + WINDOW_POST
offsets = [o for o in range(1, n_weeks) if span < o < n_weeks - span]


def shifted(o):
    return [lo + pd.Timedelta(weeks=int(((pd.Timestamp(t) - lo).days // 7 + o) % n_weeks)) for t in T]


rng = np.random.default_rng(3)
for y in ["blame", "mano"]:
    real, n_used = stat(T, y)
    if np.isnan(real):
        print(f"[12] {y}: too few weeks with comments around the shocks")
        continue
    fake = np.array([stat(shifted(o), y)[0] for o in offsets])     # every eligible shift once
    fake = fake[~np.isnan(fake)]
    p = (np.sum(np.abs(fake) >= abs(real)) + 1) / (len(fake) + 1)
    print(f"[12] {y}: post - pre share = {real:+.4f} over {n_used} shocks; randomisation p = {p:.3f} "
          f"({len(fake)} calendar shifts)")

# event-time profile (all shocks pooled)
rows = []
for t0 in T:
    for k in ks:
        w = pd.Timestamp(t0) + pd.Timedelta(weeks=int(k))
        if w in wk.index:
            rows.append({"k": k, "blame": wk.loc[w, "blame"], "mano": wk.loc[w, "mano"], "n": wk.loc[w, "n"]})
ev = pd.DataFrame(rows)
if len(ev):
    prof = ev.groupby("k")[["blame", "mano"]].mean()
    prof.to_csv(OUT / "online_event.csv")
    blog_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prof.index, prof.blame, marker="o", ms=3, label="Gendered blame")
    ax.plot(prof.index, prof.mano, marker="o", ms=3, label="Manosphere frame")
    ax.axvline(-0.5, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Weeks relative to exam date (all shocks pooled)")
    ax.set_ylabel("Share of comments")
    ax.legend(frameon=False)
    fig.savefig(FIG / "online_event.png")
