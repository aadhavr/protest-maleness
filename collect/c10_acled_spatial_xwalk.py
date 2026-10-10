"""Assign each ACLED event to its 2011 Census district by its coordinates.

Why: ACLED uses today's districts (about 780); the Census has 640 (2011).
Name matching fails for districts created after 2011 and cannot split an
event between two 2011 parents. A point-in-polygon join solves both.

Boundaries: DataMeet "maps" repository, Districts/Census_2011/2011_Dist.shp
(CC BY 4.0; cite "District boundaries: DataMeet India community").
Needs: pip install geopandas   (run 02_acled.py once first: it writes interim/acled_events.csv)

Output:
  raw/crosswalks/acled_event_district.csv   event_id_cnty -> district_id (used first by 02)
  raw/crosswalks/acled_districts.csv        (admin1, admin2) -> modal 2011 district (fallback)
  output/xwalk_check.csv                    name pairs whose events split across 2011 districts
"""
import sys
from pathlib import Path
import pandas as pd
from _common import get
from config import RAW, INT, OUT

import geopandas as gpd

BASE = "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist"
shp_dir = RAW / "shapes" / "census2011_districts"
shp_dir.mkdir(parents=True, exist_ok=True)
for ext in ["shp", "shx", "dbf", "prj"]:
    f = shp_dir / f"2011_Dist.{ext}"
    if not f.exists():
        r = get(f"{BASE}.{ext}", timeout=120)
        if r.status_code != 200:
            raise SystemExit(f"download failed: {BASE}.{ext} ({r.status_code})")
        f.write_bytes(r.content)
g = gpd.read_file(shp_dir / "2011_Dist.shp")
if g.crs is None:
    g = g.set_crs(4326)
g = g.to_crs(4326)
print(f"[c10] shapefile: {len(g)} polygons; columns: {list(g.columns)}")

# ---- find the columns that rebuild our district_id (2-digit state + 3-digit district) ----
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
ids = set(dist.district_id)
num = lambda s: pd.to_numeric(s, errors="coerce")
cols = [c for c in g.columns if c != "geometry"]
best = (0, None)
for dc in cols:
    d = num(g[dc])
    if d.isna().mean() > 0.2:
        continue
    for sc in cols:
        s = num(g[sc])
        if s.isna().mean() > 0.2 or sc == dc:
            continue
        cand = s.fillna(-1).astype(int).astype(str).str.zfill(2) + d.fillna(-1).astype(int).astype(str).str.zfill(3)
        hit = cand.isin(ids).mean()
        if hit > best[0]:
            best = (hit, (sc, dc, cand))
if best[0] < 0.9:
    raise SystemExit(f"[c10] could not rebuild district_id from the shapefile (best match {best[0]:.0%}). "
                     f"Columns: {cols}. Send me this message.")
sc, dc, cand = best[1]
g["district_id"] = cand
print(f"[c10] district_id = {sc} (state) + {dc} (district); {best[0]:.1%} of polygons match districts.csv")
g = g[g.district_id.isin(ids)][["district_id", "geometry"]]

# ---- spatial join ----
ev = pd.read_csv(INT / "acled_events.csv", low_memory=False,
                 usecols=lambda c: c in {"event_id_cnty", "admin1", "admin2", "latitude", "longitude", "geo_precision"})
pts = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.longitude, ev.latitude), crs=4326)
j = gpd.sjoin(pts, g, how="left", predicate="within").drop_duplicates("event_id_cnty")
j["method"] = "within"
miss = j.district_id.isna()
if miss.any():
    # points just outside a polygon (coast, rivers, border simplification): nearest within 10 km
    m_crs = 7755                        # WGS 84 / India NSF LCC (metres)
    near = gpd.sjoin_nearest(pts[pts.event_id_cnty.isin(j.loc[miss, "event_id_cnty"])].to_crs(m_crs),
                             g.to_crs(m_crs), how="left", max_distance=10_000).drop_duplicates("event_id_cnty")
    near = near.set_index("event_id_cnty").district_id
    j.loc[miss, "district_id"] = j.loc[miss, "event_id_cnty"].map(near)
    j.loc[miss & j.district_id.notna(), "method"] = "nearest<=10km"
j.loc[j.district_id.isna(), "method"] = "unassigned"
print("[c10] assignment:", j.method.value_counts().to_dict())
if "geo_precision" in j:
    print(f"[c10] geo_precision 3 (approximate location): {(j.geo_precision == 3).mean():.1%} of events. "
          "Run the main results with and without them.")

xw_dir = RAW / "crosswalks"
xw_dir.mkdir(parents=True, exist_ok=True)
j[["event_id_cnty", "district_id", "method", "geo_precision"]].to_csv(xw_dir / "acled_event_district.csv", index=False)

# ---- name-pair fallback crosswalk + split report ----
ok = j.dropna(subset=["district_id"])
pair = (ok.groupby(["admin1", "admin2", "district_id"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False))
pair["share"] = pair.n / pair.groupby(["admin1", "admin2"]).n.transform("sum")
modal = pair.drop_duplicates(["admin1", "admin2"])
modal.rename(columns={"admin1": "acled_admin1", "admin2": "acled_admin2"})[
    ["acled_admin1", "acled_admin2", "district_id", "n", "share"]].to_csv(xw_dir / "acled_districts.csv", index=False)
split = modal[modal.share < 0.9].sort_values("n", ascending=False)
split.to_csv(OUT / "xwalk_check.csv", index=False)
print(f"[c10] {len(modal)} ACLED districts mapped; {len(split)} split across 2011 districts "
      "(expected for districts carved from two parents) -> output/xwalk_check.csv")

# ---- check against the exact name matches from 02 ----
todo = xw_dir / "acled_districts_TODO.csv"
if todo.exists():
    t = pd.read_csv(todo, dtype=str)
    t = t[t.match == "exact"].drop(columns=["district_id"], errors="ignore").merge(modal.rename(columns={"admin1": "acled_admin1", "admin2": "acled_admin2"}),
                                    on=["acled_admin1", "acled_admin2"], how="inner")
    agree = (t.suggested_district_id == t.district_id).mean()
    print(f"[c10] check: spatial = name match for {agree:.1%} of the {len(t)} exact-name districts "
          "(expect > 95%; lower means a code problem)")
print("[c10] done. Now run 02_acled.py again from the project root.")
