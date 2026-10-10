"""YouTube comments on protest-related videos (Layer 5).

API: YouTube Data API v3. search.list = 100 quota units per call,
commentThreads.list = 1 unit. Default daily quota is 10,000 units.
Only public comments. Author names and channel ids are NOT stored.
Output: raw/online/youtube_videos.csv, raw/online/youtube_comments.csv
"""
import pandas as pd
from _common import get, need_env
from config import RAW, START_DATE, END_DATE, YOUTUBE_QUERIES, \
    YOUTUBE_MAX_VIDEOS_PER_QUERY, YOUTUBE_MAX_COMMENTS_PER_VIDEO

(key,) = need_env("YOUTUBE_API_KEY")
B = "https://www.googleapis.com/youtube/v3"
out = RAW / "online"
out.mkdir(parents=True, exist_ok=True)
vf, cf = out / "youtube_videos.csv", out / "youtube_comments.csv"
seen = set(pd.read_csv(vf).video_id) if vf.exists() else set()

vids = []
for q in YOUTUBE_QUERIES:
    tok, n = None, 0
    while n < YOUTUBE_MAX_VIDEOS_PER_QUERY:
        p = {"part": "snippet", "q": q, "type": "video", "regionCode": "IN", "maxResults": 50,
             "publishedAfter": f"{START_DATE}T00:00:00Z", "publishedBefore": f"{END_DATE}T23:59:59Z",
             "key": key, **({"pageToken": tok} if tok else {})}
        j = get(f"{B}/search", params=p).json()
        if "error" in j:
            raise SystemExit(f"YouTube error: {j['error'].get('message')}")
        for it in j.get("items", []):
            vids.append({"video_id": it["id"]["videoId"], "query": q,
                         "published": it["snippet"]["publishedAt"], "title": it["snippet"]["title"]})
            n += 1
        tok = j.get("nextPageToken")
        if not tok:
            break
v = pd.DataFrame(vids).drop_duplicates("video_id")
v = v[~v.video_id.isin(seen)]
v.to_csv(vf, mode="a", header=not vf.exists(), index=False)
print(f"[c06] {len(v)} new videos")

for vid in v.video_id:
    rows, tok = [], None
    while len(rows) < YOUTUBE_MAX_COMMENTS_PER_VIDEO:
        p = {"part": "snippet", "videoId": vid, "maxResults": 100, "textFormat": "plainText",
             "key": key, **({"pageToken": tok} if tok else {})}
        j = get(f"{B}/commentThreads", params=p).json()
        if "error" in j:                          # comments disabled, etc.
            break
        for it in j.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            rows.append({"id": "yt_" + it["id"], "platform": "youtube", "parent": vid,
                         "created": s["publishedAt"], "text": s["textDisplay"],
                         "likes": s.get("likeCount", 0)})
        tok = j.get("nextPageToken")
        if not tok:
            break
    if rows:
        pd.DataFrame(rows).to_csv(cf, mode="a", header=not cf.exists(), index=False)
print("[c06] comments ->", cf)
