"""Reddit posts and comments (Layer 5). Needs: pip install praw
Create a "script" app at reddit.com/prefs/apps for the credentials.
Usernames are NOT stored. Reddit search returns a limited number of
results per query, so coverage of older periods is partial; say so.
Output: raw/online/reddit_items.csv
"""
import pandas as pd
import praw
from _common import need_env
from config import RAW, START_DATE, REDDIT_SUBREDDITS, REDDIT_QUERIES

cid, sec, ua = need_env("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
r = praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)
t0 = pd.Timestamp(START_DATE, tz="UTC").timestamp()
out = RAW / "online" / "reddit_items.csv"
out.parent.mkdir(parents=True, exist_ok=True)
done = set(pd.read_csv(out).parent) if out.exists() else set()

for sub in REDDIT_SUBREDDITS:
    for q in REDDIT_QUERIES:
        try:
            posts = list(r.subreddit(sub).search(q, sort="new", time_filter="all", limit=None))
        except Exception as ex:
            print(f"[c07] r/{sub} '{q}': {ex}")
            continue
        rows = []
        for s in posts:
            if s.created_utc < t0 or f"t3_{s.id}" in done:
                continue
            done.add(f"t3_{s.id}")
            ts = lambda x: pd.Timestamp(x, unit="s", tz="UTC").isoformat()
            rows.append({"id": f"rd_t3_{s.id}", "platform": "reddit", "parent": f"t3_{s.id}",
                         "subreddit": sub, "query": q, "created": ts(s.created_utc),
                         "text": f"{s.title}\n{s.selftext}"[:4000], "likes": s.score})
            s.comments.replace_more(limit=0)
            for c in s.comments.list():
                rows.append({"id": f"rd_t1_{c.id}", "platform": "reddit", "parent": f"t3_{s.id}",
                             "subreddit": sub, "query": q, "created": ts(c.created_utc),
                             "text": c.body[:4000], "likes": c.score})
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[c07] r/{sub} '{q}': {len(rows)} items")
