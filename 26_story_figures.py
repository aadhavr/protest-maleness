"""Story figures for policy readers that a political scientist can still audit.

Builds output/story/story.html (all figures, one page) and one HTML file per figure:
  spread    Animated map zoomed to the exposed state (dots per district) + a live line, exposed vs rest
  rise      Fixed 2.5D map of India: each state's shape rises and falls with its weekly protest rate
  tiles     Fallback for the atlas when geopandas or the shapefile is missing: state tiles placed like India
  atlas     Real map of India (one shape per state): all leaks, or one leak's rise by state; click a state for its weekly line
  national  NEET: state exposure vs change in protest
  docket    Court cases stacked by filing month, coloured by petitioner sex, leaks marked
  calendar  (left out by default, see SKIP) Calendar heatmap of exam protests across India; hover names the top state that week
  eventstudy The estimate week by week with 95% intervals; party-led and NEET-UG 2024 placebo series on the legend
  unusual   How unusual is the real effect? The real estimate against every false-date estimate
  speccurve Specification curve: estimates across outcomes x dropped shock x trend adjustment
  courts    Street now, courts later: the protest response (weeks 0-8) and the court response (to 26)
  women     Where are the women? Out of every 50: applicants, past protesters, petitioners, and protest reports
  reports   How exam protests get reported: women mentioned, sex-neutral wording, police force, India vs neighbours
  surveys   Survey answers by sex (WVS, Pew): protest participation and 'men first for jobs'
  search    Search-box styled Trends results with a detectable-change meter and event-time line
  comments  400 YouTube-style comment cards coloured by blame target (waffle), clickable legend
Each figure is skipped if its inputs are missing. Plotly.js from a CDN; no other dependencies in the page.
Run after: 02, 04, 05 (--total, non-party), 05b (non-party), 05c, 09 (n_orders_body, PM_WINDOW_POST=26),
16, 22, 23. The atlas needs geopandas and the DataMeet shapefile from collect/c10.
"""
import json
import numpy as np
import pandas as pd
from config import RAW, INT, OUT, WINDOW_PRE, WINDOW_POST, START_DATE, END_DATE
from lib import demean, ols_cluster

D = OUT / "story"
D.mkdir(parents=True, exist_ok=True)
Y = "n_exam_strict_nonparty"
CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
INK, MUTED, ACCENT, HOT, COOL = "#1d1d2b", "#6b7280", "#c8553d", "#f4a259", "#1f6f8b"
FONT = {"family": "IBM Plex Sans, Helvetica, Arial, sans-serif", "size": 13, "color": INK}
CSS = """
*{box-sizing:border-box}
body{font-family:"IBM Plex Sans",Helvetica,Arial,sans-serif;margin:0;color:#1d1d2b;background:#fbfaf7}
.wrap{max-width:1040px;margin:0 auto;padding:18px}
h2{font-size:22px;margin:34px 0 4px} .kicker{color:#6b7280;margin:0 0 10px;font-size:14px;line-height:1.45}
.note{font-size:12px;color:#6b7280;margin:6px 0 0;line-height:1.4}
.card{background:#fff;border:1px solid #e7e5e0;border-radius:10px;padding:14px 14px 10px;margin-bottom:26px}
select{font:inherit;padding:4px 8px;border-radius:6px;border:1px solid #ccc;background:#fff}
.grid{display:grid;grid-template-columns:repeat(9,1fr);gap:6px}
.tile{border-radius:7px;padding:4px 5px 0;cursor:pointer;border:1px solid #e7e5e0;min-height:74px;position:relative}
.tile:hover{outline:2px solid #1d1d2b}
.tile .nm{font-size:11px;font-weight:600;line-height:1.1}
.tile .val{font-size:10px;color:#6b7280}
.tgrid{grid-template-columns:repeat(10,1fr);max-width:820px;margin:0 auto}.tgrid .tile{aspect-ratio:1;min-height:0;background:#fff;padding:5px;display:flex;flex-direction:column;justify-content:space-between}
.tgrid .tile.hot{background:#fbe3d9;border-color:#e9b8a6}.tgrid .tile.sel{outline:2px solid #1d1d2b}
.tile .lk{font-weight:400;font-size:9px;color:#9a3f2c}
@media(max-width:600px){.tgrid{gap:3px}.tgrid .tile{padding:2px}.tile .lk{display:none}}
.legend{display:flex;gap:12px;font-size:12px;color:#6b7280;margin:6px 0}

.ctl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
.btn{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid #1d1d2b;background:#1d1d2b;color:#fff;cursor:pointer}
.big{font-size:15px;color:#1d1d2b}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:820px){.mgrid{grid-template-columns:repeat(2,1fr)}}
.mcell{border:1px solid #eee9e0;border-radius:8px;padding:6px 6px 0}
.msvg{display:block;width:100%;height:auto;margin:4px 0 6px}
.mleg{display:flex;align-items:center;gap:8px;font-size:12px;color:#6b7280;margin:0 0 10px;flex-wrap:wrap}
.mbar{display:inline-block;width:120px;height:10px;border-radius:3px;background:linear-gradient(90deg,#f1efea,#f4d6a6 15%,#f4a259 50%,#c8553d)}
.rs-tip{position:absolute;display:none;pointer-events:none;background:#1d1d2b;color:#fff;font-size:12px;padding:5px 8px;border-radius:6px;white-space:nowrap}
.u-grid{display:flex;flex-direction:column;gap:14px}
.u-row{display:grid;grid-template-columns:230px 1fr 120px;gap:14px;align-items:center}
.u-row.rep{border-top:1px dashed #d9d4ca;padding-top:14px}
.u-lab{font-size:14px;font-weight:600;line-height:1.25}.u-lab span{display:block;font-weight:400;font-size:11px;color:#6b7280;margin-top:2px}
.u-icons{display:grid;grid-template-columns:repeat(25,1fr);gap:2px 3px}
.u-ic svg{display:block;width:100%;height:auto;fill:#d9d4ca}.u-ic.on svg{fill:#8a3fd1}
.u-row.rep .u-ic svg{fill:#e4e0d8}.u-row.rep .u-ic.on svg{fill:#8a3fd1}
.u-val b{font-size:24px;color:#8a3fd1;display:block;line-height:1}.u-val span{font-size:11px;color:#6b7280}
@media(max-width:700px){.u-row{grid-template-columns:1fr}}
.rb-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px 28px}
@media(max-width:700px){.rb-grid{grid-template-columns:1fr}}
.rb-t{font-weight:600;font-size:14px;margin-bottom:2px}.rb-s{font-size:11px;color:#6b7280;margin-bottom:4px}
.rb-row{display:grid;grid-template-columns:110px 1fr 84px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.rb-track{position:relative;height:14px;background:#f3f1ec;border-radius:3px}
.rb-bar{position:absolute;left:0;top:0;bottom:0;background:#b9b4aa;border-radius:3px}.rb-bar.hi{background:#3d3d4f}
.rb-ci{position:absolute;top:6px;height:2px;background:#1d1d2b;opacity:.55}
.rb-v{font-variant-numeric:tabular-nums;font-weight:600}
.rb-track.dumb{background:transparent;border-bottom:1px solid #e7e5e0;height:16px}
.db-line{position:absolute;top:7px;height:2px;background:#cfc9bf}
.db-dot{position:absolute;top:2px;width:12px;height:12px;margin-left:-6px;border-radius:50%;border:2px solid #fff}
.part{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#8a3fd1;margin:40px 0 -8px}
.at-wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:18px;align-items:start}
@media(max-width:820px){.at-wrap{grid-template-columns:1fr}}
.at-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.at-map svg{display:block;width:100%;height:auto;max-height:560px}.at-map path{cursor:pointer;transition:fill .3s}.at-map path:hover{opacity:.75}
.mout{display:inline-block;width:14px;height:10px;border:2px solid #1d1d2b;border-radius:2px;margin-left:10px}
.mt{font-weight:600;font-size:13px}.ms{font-size:11px;color:#6b7280}
.yt{color:#0f0f0f}
.yt-head{display:flex;gap:24px;align-items:center;font-size:16px;margin-bottom:8px}.yt-sort{font-size:14px;color:#606060}
.yt-chips{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 8px}
.yt-chip{font-size:13px;font-family:inherit;border:0;border-radius:8px;padding:6px 11px;background:#f2f2f2;color:#0f0f0f;cursor:pointer}
.yt-chip i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.yt-chip span{color:#606060;margin-left:4px}
.yt-chip.on{background:#0f0f0f;color:#fff}.yt-chip.on span{color:#ddd}
.yt-note{font-size:13px;color:#3c3c3c;min-height:18px;margin-bottom:8px}
.yt-list{height:460px;overflow-y:auto;border-top:1px solid #eee;padding-right:6px}
.yt-c{display:flex;gap:12px;padding:10px 0;transition:opacity .25s}.yt-c.dim{opacity:.12}
.yt-av{width:36px;height:36px;border-radius:50%;flex:none}
.yt-body{flex:1;min-width:0}.yt-meta{display:flex;gap:8px;align-items:center;font-size:12px;color:#606060;margin-bottom:6px}
.yt-name{display:inline-block;width:84px;height:9px;border-radius:5px;background:#d9d9d9}
.yt-tag{border:1px solid;border-radius:10px;padding:0 7px;font-size:11px}
.yt-line{height:9px;border-radius:5px;background:#e9e9e9;margin:5px 0}
.yt-act{font-size:12px;color:#606060;margin-top:6px}

.vp{max-width:820px}
.vp-screen{position:relative;background:#0f0f0f;color:#fff;border-radius:12px 12px 0 0;aspect-ratio:16/7;display:flex;flex-direction:column;justify-content:center;align-items:center;transition:box-shadow .3s}
.vp-kicker{position:absolute;top:14px;left:18px;font-size:13px;color:#aaa}
.vp-now{font-size:30px;font-weight:600;text-align:center;padding:0 20px}.vp-share{font-size:15px;color:#ccc;margin-top:6px}
.vp-play{position:absolute;bottom:14px;left:16px;width:42px;height:42px;border-radius:50%;border:0;background:#fff;color:#0f0f0f;font-size:16px;cursor:pointer}
.vp-bar{position:relative;height:10px;background:#3a3a3a;cursor:pointer}
.vp-seg{position:absolute;top:2px;height:6px;border-radius:2px;cursor:pointer}.vp-seg:hover{top:0;height:10px}
.vp-head{position:absolute;top:-4px;width:16px;height:16px;margin-left:-8px;border-radius:50%;background:#ff3d3d;box-shadow:0 0 0 2px #fff;pointer-events:none;z-index:2}
.vp-ctrl{background:#0f0f0f;color:#ddd;font-size:13px;padding:8px 14px;border-radius:0 0 12px 12px;display:flex;gap:8px}
.vp-title{font-size:17px;font-weight:600;margin:14px 0 8px}
.vp-list{display:flex;flex-direction:column;gap:2px}
.vp-row{display:grid;grid-template-columns:14px 52px 1fr auto;gap:10px;align-items:center;text-align:left;border:0;background:transparent;padding:7px 8px;border-radius:8px;cursor:pointer;font-family:inherit;font-size:14px;color:#1d1d2b}
.vp-row:hover,.vp-row.on{background:#f2f0eb}.vp-dot{width:12px;height:12px;border-radius:3px}.vp-ts{color:#1f6f8b;font-variant-numeric:tabular-nums}.vp-len{color:#6b7280;font-size:13px}

.dk-head{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:4px}
.dk-tabs{display:flex;gap:6px}.dk-tab{font-family:inherit;font-size:13px;border:1px solid #d9d6cf;background:#fff;border-radius:16px;padding:5px 12px;cursor:pointer}
.dk-tab.on{background:#1d1d2b;color:#fff;border-color:#1d1d2b}.dk-stat{font-size:14px}

.cb-legend{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.cb-chip{font-family:inherit;font-size:13px;border:1px solid #e2dfd8;background:#fff;border-radius:16px;padding:4px 10px;cursor:pointer;color:#1d1d2b}
.cb-chip b{font-weight:600;margin-left:3px}.cb-chip.on{border-color:#1d1d2b;background:#f4f2ee}
.cb-sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cb-head{display:flex;gap:28px;align-items:center;font-size:17px;color:#0f0f0f;margin:4px 0 12px}
.cb-head span{font-size:14px;color:#606060}
.cb-grid{display:grid;grid-template-columns:repeat(16,1fr);gap:4px}
@media(max-width:900px){.cb-grid{grid-template-columns:repeat(14,1fr)}}
@media(max-width:600px){.cb-grid{grid-template-columns:repeat(8,1fr)}}
.cb{display:flex;gap:5px;height:30px;padding:5px 5px 3px;border-radius:4px;background:var(--t);transition:opacity .2s;overflow:hidden}
.cb .av{flex:none;width:13px;height:13px;border-radius:50%;background:var(--c)}
.cb .tx{flex:1;min-width:0;display:flex;flex-direction:column;gap:4px;padding-top:1px}
.cb .tx b{display:block;height:4px;width:50%;border-radius:2px;background:#3f3f46;opacity:.7}
.cb .tx b em{display:none}
.cb .tx i{display:block;height:3px;width:85%;border-radius:2px;background:#bdb8ae}
.cb .ft{display:flex;gap:3px;align-items:center;margin-top:1px}
.cb .ft svg{width:5px;height:5px;fill:#a19d95}
.cb.dim{opacity:.12}
.cb-note{font-size:14px;margin-top:12px}
.gq{font-family:inherit;color:#202124}
.gq-bar{display:flex;align-items:center;gap:12px;border:1px solid #dfe1e5;border-radius:24px;padding:11px 18px;box-shadow:0 1px 6px rgba(32,33,36,.18);font-size:16px;max-width:640px}
.gq-caret{color:#4285F4;animation:blink 1s step-end infinite}@keyframes blink{50%{opacity:0}}
.gq-chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.gq-chip{font-size:13px;font-family:inherit;border:1px solid #dadce0;border-radius:16px;padding:6px 12px;background:#fff;color:#3c4043;cursor:pointer}
.gq-chip i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--c);margin-right:7px}
.gq-chip.on{border-color:var(--c);background:#f8f9fa;font-weight:500}
.gq-card{border:1px solid #dadce0;border-radius:8px;padding:16px}
.gq-row{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap}.gq-card{overflow:hidden}
.gq-title{font-size:18px}.gq-sub{font-size:13px;color:#5f6368}
.gq-num{text-align:right}.gq-num div:first-child{font-size:34px;font-weight:500}.gq-small{font-size:12px;color:#5f6368}
.gq-meter{margin-top:14px}.gq-track{position:relative;height:14px;background:#f1f3f4;border-radius:7px;overflow:hidden}
.gq-mde{position:absolute;top:0;bottom:0;background:#dadce0}.gq-val{position:absolute;top:3px;bottom:3px;border-radius:4px}
.gq-legend{display:flex;gap:16px;font-size:12px;color:#5f6368;margin-top:6px}.sw{display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:5px}
.sw.val{background:#34A853}.sw.mde{background:#dadce0}
"""
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">')
EMBED_CSS = "body{background:#fff}.wrap{max-width:none;padding:2px 2px 6px}.card{border:0;padding:0;margin:0;border-radius:0}h2{margin-top:0}"
RESIZE_JS = """<script>(function(){function s(){parent.postMessage({wsFig:'NAME',h:document.documentElement.scrollHeight},'*');}
window.addEventListener('load',s);window.addEventListener('resize',s);if(window.ResizeObserver)new ResizeObserver(s).observe(document.body);setTimeout(s,800);})();</script>"""
CITE = "Data: ACLED (acleddata.com), Census 2011; author's calculations."
blocks = {}


SKIP = {"calendar"}   # left out as redundant: the atlas shows the same weekly series with leaks marked


def emit(name, title, kicker, body_html, note):
    if name in SKIP:
        print(f"[26] {name} left out (in SKIP)")
        return
    html = (f'<div class="card" id="{name}"><h2>{title}</h2><p class="kicker">{kicker}</p>'
            f'{body_html}<p class="note">{note}</p></div>')
    blocks[name] = html
    # per-figure page: made to sit in an iframe in the blog post (white, no card frame); it reports its
    # height to the parent page so the iframe can fit it without a scroll bar
    page = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title>{FONT_LINK}<script src="{CDN}"></script><style>{CSS}{EMBED_CSS}</style></head>'
            f'<body><div class="wrap">{html}</div>{RESIZE_JS.replace("NAME", name)}</body></html>')
    (D / f"{name}.html").write_text(page, encoding="utf-8")
    print(f"[26] {name}.html")


def lay(**kw):
    base = {"font": FONT, "paper_bgcolor": "white", "plot_bgcolor": "white",
            "margin": {"l": 56, "r": 16, "t": 16, "b": 46}, "hoverlabel": {"font": {"family": FONT["family"]}}}
    base.update(kw)
    return base


J = lambda o: json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x))

# ---------------- common data ----------------
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
if Y not in pr:
    raise SystemExit(f"{Y} not in the panel: run 02_acled.py with classifier labels first")
pr = pr.merge(dist[["district_id", "state"]], on="district_id", how="left")
pop_st = dist.groupby("state").pop_15_29.sum()
raw_sh = pd.read_csv(RAW / "exam_shocks.csv", parse_dates=["exam_date", "shock_date"])
raw_sh = raw_sh[raw_sh.shock_type.isin(["cancelled", "voided"])].copy()
raw_sh["t0"] = raw_sh.exam_date.fillna(raw_sh.shock_date)
raw_sh = raw_sh.dropna(subset=["t0"])
raw_sh["wk0"] = raw_sh.t0 - pd.to_timedelta(raw_sh.t0.dt.weekday, unit="D")
LABELS = {"UKPSC_PATWARI_2023": "Uttarakhand patwari exam, Jan 2023", "GPSSB_JRCLERK_2023": "Gujarat junior clerk, Jan 2023",
          "BIHAR_CONST_2023": "Bihar police constable, Oct 2023", "JSSC_CGL_2024": "Jharkhand CGL, Jan 2024",
          "UPPSC_ROARO_2024": "UP RO/ARO, Feb 2024", "UP_CONST_2024": "UP police constable, Feb 2024",
          "BPSC_TRE3_2024": "Bihar teacher exam, Mar 2024", "UKSSSC_GRAD_2025": "Uttarakhand graduate-level, Sep 2025",
          "NEETUG_2026": "NEET-UG, May 2026", "NEETUG_2024": "NEET-UG, May 2024", "UGCNET_2024": "UGC-NET, Jun 2024"}
nice = lambda e: LABELS.get(e, e.replace("_", " ").title())
# keep only the leaks that enter the analysis: exposure varies across districts
_exf = INT / "exposure.csv"
if _exf.exists():
    _ex = pd.read_csv(_exf, dtype={"district_id": str})
    # an exam identifies something if exposure differs across India's districts: a state exam
    # (exposed districts > 0, all others 0) or a national exam with state-level counts. A national
    # exam with one number for everyone (UGC-NET) does not.
    _nd = dist.district_id.nunique()
    def _varies(a):
        a = a[a > 0]
        return len(a) > 0 and (len(a) < _nd or a.std() > 1e-12 * a.abs().max())
    _v = _ex.groupby("exam_id").A_tot.apply(_varies)
    ANALYSIS = set(_v[_v].index)
    raw_sh = raw_sh[raw_sh.exam_id.isin(ANALYSIS)].copy()
    print(f"[26] leaks shown: {sorted(raw_sh.exam_id)}")

# ---------------- spread: zoomed animated map + live line ----------------
try:
    import geopandas as gpd
    shp = RAW / "shapes" / "census2011_districts" / "2011_Dist.shp"
    g = gpd.read_file(shp).to_crs(4326)
    ids = set(dist.district_id)
    best = None
    for sc in g.columns:
        for dc in g.columns:
            if "geometry" in (sc, dc) or sc == dc:
                continue
            a, b = pd.to_numeric(g[sc], errors="coerce"), pd.to_numeric(g[dc], errors="coerce")
            if a.isna().mean() > .2 or b.isna().mean() > .2:
                continue
            cand = a.fillna(-1).astype(int).astype(str).str.zfill(2) + b.fillna(-1).astype(int).astype(str).str.zfill(3)
            if best is None or cand.isin(ids).mean() > best[0]:
                best = (cand.isin(ids).mean(), cand)
    g["district_id"] = best[1]
    g = g[g.district_id.isin(ids)].merge(dist[["district_id", "state", "district_name"]], on="district_id")
    pts = g.geometry.representative_point()
    g["lon"], g["lat"] = pts.x.round(3), pts.y.round(3)
    st = g.dissolve(by="state")[["geometry"]].reset_index()
    bounds = {s_: [round(v, 2) for v in geom.bounds] for s_, geom in zip(st.state, st.geometry)}
    from shapely.geometry import mapping
    _u = st.geometry.union_all() if hasattr(st.geometry, "union_all") else st.geometry.unary_union
    gi = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "India", "properties": {},
                                                      "geometry": mapping(_u.simplify(0.01, preserve_topology=True))}]}
    st["geometry"] = st.geometry.simplify(0.01, preserve_topology=True)
    gs = json.loads(st.to_json())
    for ft in gs["features"]:
        ft["id"] = ft["properties"]["state"]
    pv = pr.pivot_table(index="district_id", columns="week", values=Y, aggfunc="sum", fill_value=0)
    order = g.district_id.tolist()
    dname = (g.district_name + ", " + g.state).tolist()
    dpop = g[["district_id"]].merge(dist[["district_id", "pop_15_29"]], on="district_id", how="left").pop_15_29.fillna(0).round().astype(int).tolist()
    st_w = pr.groupby(["state", "week"])[Y].sum().unstack(fill_value=0)
    SH = {}
    for r in raw_sh.itertuples():
        nat = r.level == "national"
        exp_state = None if nat else r.state
        if not nat and exp_state not in st_w.index:
            continue
        frames = []
        for k in range(-4, 9):
            w = r.wk0 + pd.Timedelta(weeks=k)
            z = pv[w].reindex(order).fillna(0).astype(int).tolist() if w in pv.columns else [0] * len(order)
            if w in st_w.columns:
                col = st_w[w]
                if nat:
                    exp_rate = float(col.sum() / pop_st.sum() * 1e7)
                    rest_rate = None
                    n_exp = int(col.sum())
                else:
                    exp_rate = float(col.get(exp_state, 0) / pop_st[exp_state] * 1e7)
                    others = col.drop(exp_state, errors="ignore")
                    rest_rate = float(others.sum() / pop_st.reindex(others.index).sum() * 1e7)
                    n_exp = int(col.get(exp_state, 0))
            else:
                exp_rate, rest_rate, n_exp = 0.0, (None if nat else 0.0), 0
            frames.append({"k": k, "z": z, "date": w.strftime("%d %b %Y"), "e": round(exp_rate, 2),
                           "r": None if rest_rate is None else round(rest_rate, 3), "n": n_exp})
        if nat:
            box = [68, 7, 98, 37]
        else:
            b0 = bounds[exp_state]
            pad = max(2.5, 0.6 * max(b0[2] - b0[0], b0[3] - b0[1]))
            box = [b0[0] - pad, b0[1] - pad, b0[2] + pad, b0[3] + pad]
        SH[r.exam_id] = {"label": nice(r.exam_id) + (" (national)" if nat else f" ({exp_state})"),
                         "state": exp_state or "", "box": box, "frames": frames}
    if SH:
        body = ('<div class="ctl"><select id="spSel"></select><button id="spPlay" class="btn">&#9654; Play</button>'
                '<span id="spBig" class="big"></span></div><div id="spFig" style="height:600px"></div>')
        script = f"""
<script>
(function(){{
const GS={J(gs)}, GI={J(gi)}, IDS={J(order)}, NM={J(dname)}, LON={J(g.lon.tolist())}, LAT={J(g.lat.tolist())}, POP={J(dpop)}, SH={J(SH)};
const sel=document.getElementById('spSel'), big=document.getElementById('spBig');
Object.entries(SH).forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.text=v.label;sel.appendChild(o)}});
const KS=[...Array(13).keys()].map(i=>i-4);
const rate=(n,i)=>POP[i]>0?n/POP[i]*1e7:0;
function bubbles(z,scale){{const idx=z.map((v,i)=>v>0&&POP[i]>0?i:-1).filter(i=>i>=0);
  return {{lon:idx.map(i=>LON[i]),lat:idx.map(i=>LAT[i]),
    text:idx.map(i=>NM[i]+': '+z[i]+(z[i]==1?' protest':' protests')+', '+rate(z[i],i).toFixed(1)+' per 10M youth'),
    size:idx.map(i=>Math.min(70,5+scale*Math.sqrt(rate(z[i],i))))}};}}
function cum(frames,j){{const out=frames[0].z.map(()=>0);for(let t=0;t<=j;t++){{if(frames[t].k<0)continue;frames[t].z.forEach((v,i)=>out[i]+=v)}};return out;}}
function headline(s,f){{const where=s.state||'India';
  return 'Week '+(f.k>=0?'+':'')+f.k+' ('+f.date+'): <b>'+f.n+'</b> non-party exam '+(f.n==1?'protest':'protests')+' in '+where;}}
// the map changes look with the stage of the leak: before / exam weeks / after
const stage=k=>k<0?0:(k<=2?1:2);
const SZ=[0.33,1,0.66], SLC=['{INK}','{ACCENT}','#9a3f2c'], SLW=[2,3.6,2.6];
const BADGE=[['BEFORE THE EXAM','#6b7280'],['EXAM WEEKS: LEAK SURFACES','{ACCENT}'],['AFTER THE EXAM','#9a3f2c']];
function stData(s,k){{const t=stage(k), ex=id=>!s.state||id===s.state;
  return {{z:GS.features.map(f=>ex(f.id)?SZ[t]:0),
    marker:{{line:{{color:GS.features.map(f=>ex(f.id)&&s.state?SLC[t]:'#c9c5bc'),width:GS.features.map(f=>ex(f.id)&&s.state?SLW[t]:0.6)}}}}}};}}
function ann(f){{const t=stage(f.k);
  return [{{xref:'x',yref:'paper',x:-0.4,y:0.9,text:'exam',showarrow:false,xanchor:'left',font:{{size:11,color:'{MUTED}'}}}},
    {{xref:'paper',yref:'paper',x:0.005,y:0.995,xanchor:'left',yanchor:'top',showarrow:false,borderpad:6,bgcolor:BADGE[t][1],
      text:'<b>'+BADGE[t][0]+'</b>, week '+(f.k>=0?'+':'')+f.k,font:{{color:'white',size:13}}}}];}}
function frameData(s,j){{const f=s.frames[j], cb=bubbles(cum(s.frames,j),2.4), nb=bubbles(f.z,3.2);
  const ks=s.frames.slice(0,j+1).map(x=>x.k);
  return [stData(s,f.k),
    {{lon:cb.lon,lat:cb.lat,text:cb.text.map(t=>t+' since the exam'),marker:{{size:cb.size}}}},
    {{lon:nb.lon,lat:nb.lat,text:nb.text,marker:{{size:nb.size}}}},
    {{x:ks,y:s.frames.slice(0,j+1).map(x=>x.e)}},
    {{x:ks,y:s.frames.slice(0,j+1).map(x=>x.r)}}];}}
function draw(key){{
  const s=SH[key]; const ymax=Math.max(1,...s.frames.map(f=>Math.max(f.e,f.r||0)))*1.15;
  const d0=frameData(s,0);
  const states={{type:'choropleth',geojson:GS,locations:GS.features.map(f=>f.id),...d0[0],zmin:0,zmax:1,showscale:false,
    hoverinfo:'location',colorscale:[[0,'#f3f1ec'],[0.33,'#e2dacb'],[0.66,'#efbca6'],[1,'#e68f72']],geo:'geo'}};
  const outline={{type:'choropleth',geojson:GI,locations:['India'],z:[0],showscale:false,hoverinfo:'skip',geo:'geo',
    colorscale:[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],marker:{{line:{{color:'#8d8a83',width:1.2}}}}}};
  const traces=[states,
    {{type:'scattergeo',mode:'markers',geo:'geo',...d0[1],hovertemplate:'%{{text}}<extra></extra>',name:'Since the exam',
      marker:{{...d0[1].marker,color:'rgba(200,85,61,0.18)',line:{{color:'rgba(200,85,61,0.55)',width:1}}}}}},
    {{type:'scattergeo',mode:'markers',geo:'geo',...d0[2],hovertemplate:'%{{text}}<extra></extra>',name:'This week',
      marker:{{...d0[2].marker,color:'{ACCENT}',opacity:0.9,line:{{color:'white',width:1.5}}}}}},
    {{type:'scatter',mode:'lines+markers',xaxis:'x',yaxis:'y',...d0[3],name:s.state||'All India',
      line:{{color:'{ACCENT}',width:3}},marker:{{size:8}},hovertemplate:'week %{{x}}: %{{y:.1f}} per 10M youth<extra></extra>'}},
    {{type:'scatter',mode:'lines',xaxis:'x',yaxis:'y',...d0[4],name:'Rest of India',line:{{color:'{MUTED}',width:2,dash:'dot'}},
      hovertemplate:'week %{{x}}: %{{y:.2f}} per 10M youth<extra></extra>'}}, outline];
  const frames=s.frames.map((f,j)=>({{name:String(f.k),data:frameData(s,j),traces:[0,1,2,3,4],layout:{{annotations:ann(f)}}}}));
  const layout={{font:{J(FONT)},paper_bgcolor:'white',plot_bgcolor:'white',margin:{{l:10,r:20,t:10,b:40}},showlegend:true,
    legend:{{orientation:'h',x:0.66,y:-0.08,font:{{size:11}}}},
    geo:{{domain:{{x:[0,0.6],y:[0,1]}},projection:{{type:'mercator'}},resolution:50,showframe:false,
      showcountries:false,showsubunits:false,showlakes:false,showland:true,landcolor:'#faf8f3',showcoastlines:true,
      coastlinecolor:'#c9c5bc',showocean:true,oceancolor:'#eef3f5',
      lonaxis:{{range:[s.box[0],s.box[2]]}},lataxis:{{range:[s.box[1],s.box[3]]}},bgcolor:'white'}},
    xaxis:{{domain:[0.68,1],range:[-4.5,8.5],title:'Weeks from exam',zeroline:false,gridcolor:'#eee',dtick:2}},
    yaxis:{{domain:[0.12,0.88],range:[0,ymax],title:'Exam protests per 10M youth',gridcolor:'#eee'}},
    shapes:[{{type:'rect',xref:'x',yref:'paper',x0:-0.5,x1:2.5,y0:0.12,y1:0.88,fillcolor:'rgba(200,85,61,0.07)',line:{{width:0}}}},
            {{type:'line',xref:'x',yref:'paper',x0:-0.5,x1:-0.5,y0:0.12,y1:0.88,line:{{color:'{INK}',width:1,dash:'dash'}}}}],
    annotations:ann(s.frames[0]),
    sliders:[{{active:0,x:0.68,len:0.32,y:0.02,currentvalue:{{visible:false}},pad:{{t:0}},
      steps:s.frames.map(f=>({{label:String(f.k),method:'animate',args:[[String(f.k)],{{mode:'immediate',frame:{{duration:500,redraw:true}},transition:{{duration:0}}}}]}}))}}]}};
  big.innerHTML=headline(s,s.frames[0]);
  Plotly.newPlot('spFig',traces,layout,{{responsive:true,displaylogo:false}}).then(gd=>{{
    Plotly.addFrames('spFig',frames);
    gd.on('plotly_animatingframe',ev=>{{const f=s.frames.find(x=>String(x.k)===ev.name);if(f)big.innerHTML=headline(s,f);}});
    gd.on('plotly_sliderchange',ev=>{{const f=s.frames.find(x=>String(x.k)===ev.step.label);if(f)big.innerHTML=headline(s,f);}});
  }});
}}
sel.onchange=()=>draw(sel.value);
document.getElementById('spPlay').onclick=()=>Plotly.animate('spFig',null,{{frame:{{duration:700,redraw:true}},transition:{{duration:0}},mode:'immediate'}});
const first=Object.keys(SH).find(k=>k.indexOf('UKSSSC')===0)||sel.value; sel.value=first; draw(first);
}})();
</script>"""
        emit("spread", "Watch a leak spread",
             "Pick a leak and press Play. Dots are districts with exam protests that week.",
             body + script,
             "Non-party exam protests per 10M residents aged 15-29; dot area grows with the rate. Shaded: weeks 0-2. ACLED; Census 2011; DataMeet.")
except Exception as ex:
    print(f"[26] spread skipped ({str(ex)[:160]})")

# ---------------- state shapes (plain SVG, true proportions) for the atlas ----------------
PATHS, MP, CEN, RINGS = None, [], None, None
st_w = pr.groupby(["state", "week"])[Y].sum().unstack(fill_value=0)
rate2 = st_w.div(pop_st, axis=0) * 1e7
big_states = pop_st[pop_st >= 5e6].index
try:
    import geopandas as gpd
    import math
    shp = RAW / "shapes" / "census2011_districts" / "2011_Dist.shp"
    gm = gpd.read_file(shp).to_crs(4326)
    ids = set(dist.district_id)
    best = None
    for sc in gm.columns:
        for dc in gm.columns:
            if "geometry" in (sc, dc) or sc == dc:
                continue
            a, b = pd.to_numeric(gm[sc], errors="coerce"), pd.to_numeric(gm[dc], errors="coerce")
            if a.isna().mean() > .2 or b.isna().mean() > .2:
                continue
            cand = a.fillna(-1).astype(int).astype(str).str.zfill(2) + b.fillna(-1).astype(int).astype(str).str.zfill(3)
            if best is None or cand.isin(ids).mean() > best[0]:
                best = (cand.isin(ids).mean(), cand)
    gm["district_id"] = best[1]
    gm = gm[gm.district_id.isin(ids)].merge(dist[["district_id", "state"]], on="district_id")
    stg = gm.dissolve(by="state")[["geometry"]].reset_index()
    stg["geometry"] = stg.geometry.simplify(0.02, preserve_topology=True)
    # equirectangular projection scaled by cos(23 deg): India keeps its real shape
    KX, S = math.cos(math.radians(23)), 20.0
    x0, y1 = stg.total_bounds[0], stg.total_bounds[3]
    def _ring(coords):
        pts = [f"{(lo - x0) * KX * S:.1f},{(y1 - la) * S:.1f}" for lo, la in coords]
        return "M" + "L".join(pts) + "Z"
    def _path(geom):
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        return "".join(_ring(p.exterior.coords) for p in polys if p.area > 0.002 or len(polys) == 1)
    PATHS = {s_: _path(gg) for s_, gg in zip(stg.state, stg.geometry)}
    b = stg.total_bounds
    VB = [0, 0, round((b[2] - b[0]) * KX * S, 1), round((b[3] - b[1]) * S, 1)]
    # the same projection for the 3D spike map: outlines as one line (None between rings), north up
    y0 = b[1]
    OX, OY = [], []
    for gg in stg.geometry:
        for p_ in (list(gg.geoms) if gg.geom_type == "MultiPolygon" else [gg]):
            if p_.area < 0.002 and gg.geom_type == "MultiPolygon":
                continue
            for lo, la in p_.exterior.coords:
                OX.append(round((lo - x0) * KX * S, 1)); OY.append(round((la - y0) * S, 1))
            OX.append(None); OY.append(None)
    _rp = stg.geometry.representative_point()
    CEN = {s_: [round((pt.x - x0) * KX * S, 1), round((pt.y - y0) * S, 1)] for s_, pt in zip(stg.state, _rp)}
    # rings in SVG coordinates (y down) and label points, for the fixed 2.5D map
    RINGS, LP = {}, {}
    for s_, gg in zip(stg.state, stg.geometry):
        g2 = gg.simplify(0.03, preserve_topology=True)
        rr = []
        for p_ in (list(g2.geoms) if g2.geom_type == "MultiPolygon" else [g2]):
            if p_.area < 0.01 and g2.geom_type == "MultiPolygon":
                continue
            rr.append([v for lo, la in p_.exterior.coords[:-1] for v in (round((lo - x0) * KX * S, 1), round((y1 - la) * S, 1))])
        RINGS[s_] = rr
    for s_, pt in zip(stg.state, _rp):
        LP[s_] = [round((pt.x - x0) * KX * S, 1), round((y1 - pt.y) * S, 1)]
    MP = []
    for r in raw_sh[raw_sh.level == "state"].itertuples():
        pre_w = [r.wk0 + pd.Timedelta(weeks=k) for k in range(-8, 0)]
        post_w = [r.wk0 + pd.Timedelta(weeks=k) for k in range(0, 3)]
        if sum(w in rate2.columns for w in post_w) < 2:
            continue
        ch = rate2.reindex(columns=post_w).mean(axis=1) - rate2.reindex(columns=pre_w).mean(axis=1)
        post = rate2.loc[rate2.index.intersection(big_states)].reindex(columns=post_w).mean(axis=1)
        rank = int((post > post.get(r.state, np.nan)).sum()) + 1 if r.state in post.index else None
        MP.append({"label": nice(r.exam_id), "state": r.state, "rank": rank, "n": int(post.notna().sum()),
                   "z": {s_: round(float(v), 2) for s_, v in ch.items() if s_ in PATHS and pd.notna(v)}})
    allv = [v for m_ in MP for v in m_["z"].values() if v > 0]
    ZMAX_MAP = float(np.quantile(allv, 0.97)) if allv else 1.0
except Exception as ex:
    PATHS = None
    print(f"[26] state shapes not built; tile grid used instead ({str(ex)[:160]})")

# ---------------- tiles: every state, every week, laid out like India ----------------
import re as _re
_TILE = {  # code: (column, row, key words that match the state name); square tiles, India's real proportions
    "JK": (2, 0, ["jammuandkashmir", "jammukashmir"]), "LA": (3, 0, ["ladakh"]),
    "PB": (2, 1, ["punjab"]), "HP": (3, 1, ["himachalpradesh"]), "UK": (4, 1, ["uttarakhand", "uttaranchal"]),
    "AR": (9, 1, ["arunachalpradesh"]),
    "CH": (1, 2, ["chandigarh"]), "HR": (2, 2, ["haryana"]), "DL": (3, 2, ["delhi", "nctofdelhi"]),
    "UP": (4, 2, ["uttarpradesh"]), "BR": (5, 2, ["bihar"]), "SK": (6, 2, ["sikkim"]),
    "AS": (8, 2, ["assam"]), "NL": (9, 2, ["nagaland"]),
    "RJ": (2, 3, ["rajasthan"]), "MP": (3, 3, ["madhyapradesh"]), "CG": (4, 3, ["chhattisgarh", "chattisgarh"]),
    "JH": (5, 3, ["jharkhand"]), "WB": (6, 3, ["westbengal"]), "ML": (8, 3, ["meghalaya"]), "MN": (9, 3, ["manipur"]),
    "GJ": (1, 4, ["gujarat"]),
    "DN": (2, 4, ["dadraandnagarhaveli", "dadranagarhaveli", "dadraandnagarhavelianddamananddiu"]),
    "MH": (3, 4, ["maharashtra"]), "TS": (4, 4, ["telangana"]), "OD": (5, 4, ["odisha", "orissa"]),
    "TR": (8, 4, ["tripura"]), "MZ": (9, 4, ["mizoram"]),
    "DD": (1, 5, ["damananddiu", "damandiu"]), "GA": (2, 5, ["goa"]), "KA": (3, 5, ["karnataka"]),
    "AP": (4, 5, ["andhrapradesh"]),
    "LD": (1, 6, ["lakshadweep"]), "KL": (3, 6, ["kerala"]), "TN": (4, 6, ["tamilnadu"]),
    "PY": (5, 6, ["puducherry", "pondicherry"]),
    "AN": (8, 6, ["andamanandnicobarislands", "andamannicobarislands", "andamanandnicobar"]),
}
_key = lambda x: _re.sub(r"[^a-z]", "", str(x).lower().replace("&", "and"))
_lookup = {k: code for code, (_, _, ks) in _TILE.items() for k in ks}
rate_all = (st_w.div(pop_st, axis=0) * 1e7).dropna(how="all").sort_index(axis=1)
rate_all = rate_all.T.rolling(4, min_periods=1).mean().T          # 4-week average
india = (st_w.sum() / pop_st.sum() * 1e7).sort_index().rolling(4, min_periods=1).mean()
wks = [w.strftime("%Y-%m-%d") for w in rate_all.columns]
leak_d = lambda r: r.wk0.strftime("%Y-%m-%d")
NAT = [{"d": leak_d(r), "label": nice(r.exam_id)} for r in raw_sh[raw_sh.level == "national"].itertuples()]
OWN = {}
for r in raw_sh[raw_sh.level == "state"].itertuples():
    OWN.setdefault(r.state, []).append({"d": leak_d(r), "label": nice(r.exam_id)})
TL, used, spare = [], set(), 0
for s_ in rate_all.index:
    code = _lookup.get(_key(s_))
    if code is None or code in used:
        col, row, code = spare % 10, 7 + spare // 10, (code or s_[:3].upper())
        spare += 1
        print(f"[26] tiles: no map position for '{s_}'; put in the bottom row")
    else:
        col, row = _TILE[code][0], _TILE[code][1]
        used.add(code)
    TL.append({"s": s_, "code": code, "c": col + 1, "r": row + 1,
               "y": [round(float(v), 2) if pd.notna(v) else None for v in rate_all.loc[s_]],
               "leaks": OWN.get(s_, [])})
_pos = [v for t in TL for v in t["y"] if v and v > 0]
TCAP = float(np.quantile(_pos, 0.98)) if _pos else 1.0
if TL and PATHS and MP:
    first = max(TL, key=lambda t: (len(t["leaks"]), t["s"] == "Uttar Pradesh"))["s"]
    SER = {t["s"]: {"y": t["y"], "leaks": t["leaks"]} for t in TL}
    body = ('<div class="at-chips" id="atChips"></div><div class="at-wrap"><div class="at-map">'
            '<svg id="atSvg" viewBox="' + " ".join(str(v) for v in VB) + '"></svg><div class="mleg" id="atLeg"></div></div>'
            '<div class="at-side"><div id="atNote" class="big"></div><div id="atFig" style="height:380px"></div></div></div>')
    script = f"""
<script>
(function(){{
const P={J(PATHS)}, SER={J(SER)}, W={J(wks)}, IN={J([round(float(v), 3) for v in india.reindex(rate_all.columns).fillna(0)])},
  NAT={J(NAT)}, LK={J(MP)}, ZMAX={ZMAX_MAP:.3f};
const svg=document.getElementById('atSvg'), chips=document.getElementById('atChips'), leg=document.getElementById('atLeg'), note=document.getElementById('atNote');
const STOPS=[[0,[241,239,234]],[0.15,[244,214,166]],[0.5,[244,162,89]],[1,[200,85,61]]];
function col(v){{if(v===undefined)return '#ffffff'; let t=Math.max(0,Math.min(1,v/ZMAX));
  for(let i=1;i<STOPS.length;i++){{if(t<=STOPS[i][0]){{const [a,ca]=STOPS[i-1],[b,cb]=STOPS[i],u=(t-a)/(b-a);
    return 'rgb('+ca.map((c,j)=>Math.round(c+(cb[j]-c)*u)).join(',')+')';}}}} return 'rgb(200,85,61)';}}
const own=s=>(SER[s]||{{leaks:[]}}).leaks;
const NS='http://www.w3.org/2000/svg', EL={{}};
Object.keys(P).forEach(s=>{{const p=document.createElementNS(NS,'path'); p.setAttribute('d',P[s]); p.setAttribute('vector-effect','non-scaling-stroke');
  const t=document.createElementNS(NS,'title'); p.appendChild(t); p.addEventListener('click',()=>pick(s)); svg.appendChild(p); EL[s]=p;}});
let mode=-1, sel={J(first)};
function paint(){{
  const L=mode>=0?LK[mode]:null;
  Object.entries(EL).forEach(([s,p])=>{{
    let fill, tip;
    if(L){{const v=L.z[s]; fill=col(v); tip=s+': '+(v===undefined?'no data':(v>=0?'+':'')+v.toFixed(2)+' exam protests per 10M youth per week after this leak');}}
    else{{const n=own(s).length; fill=n>=2?'#e68f72':(n==1?'#f3c9b8':'#f6f4ef'); tip=s+(n?': '+own(s).map(l=>l.label).join('; '):': no state exam leak in our list');}}
    const hot=L?s===L.state:false;
    p.setAttribute('fill',fill); p.setAttribute('stroke',s===sel?'{INK}':(hot?'{INK}':'#cfc9bf'));
    p.setAttribute('stroke-width',s===sel?2.4:(hot?2:0.6)); p.firstChild.textContent=tip+' (click for weekly line)';
  }});
  if(L&&EL[L.state])svg.appendChild(EL[L.state]); if(EL[sel])svg.appendChild(EL[sel]);
  leg.innerHTML=L?'<span>no rise</span><span class="mbar"></span><span>+'+ZMAX.toFixed(1)+' or more per 10M youth per week</span><span class="mout"></span><span>ran the exam</span>'
    :'<span class="sw" style="background:#f3c9b8"></span><span>1 leaked exam</span><span class="sw" style="background:#e68f72"></span><span>2 leaked exams</span><span class="mout"></span><span>selected</span>';
}}
function chart(){{
  const t=SER[sel]||{{y:W.map(()=>null),leaks:[]}}, L=mode>=0?LK[mode]:null, sh=[], an=[];
  t.leaks.forEach((l,i)=>{{sh.push({{type:'line',xref:'x',yref:'paper',x0:l.d,x1:l.d,y0:0,y1:1,line:{{color:'{ACCENT}',width:2}}}});
    an.push({{xref:'x',yref:'paper',x:l.d,y:1.03+0.09*i,text:l.label,showarrow:false,xanchor:'left',font:{{size:11,color:'{ACCENT}'}}}});}});
  NAT.forEach((l,i)=>{{sh.push({{type:'line',xref:'x',yref:'paper',x0:l.d,x1:l.d,y0:0,y1:1,line:{{color:'{COOL}',width:1.5,dash:'dash'}}}});
    an.push({{xref:'x',yref:'paper',x:l.d,y:1.03+0.09*i,text:l.label,showarrow:false,xanchor:'right',font:{{size:11,color:'{COOL}'}}}});}});
  const head='<b>'+sel+'</b>: non-party exam protests per 10M youth, 4-week average';
  note.innerHTML=L?head+'<br><span class="ms">'+(L.rank?L.state+' ranks <b>#'+L.rank+' of '+L.n+'</b> large states in the 2 weeks after '+L.label:L.label)+'</span>':head;
  Plotly.react('atFig',[
    {{x:W,y:IN,type:'scatter',mode:'lines',name:'India average',line:{{color:'{MUTED}',width:1.5,dash:'dot'}},hovertemplate:'India: %{{y:.2f}}<extra></extra>'}},
    {{x:W,y:t.y,type:'scatter',mode:'lines',name:sel,line:{{color:'{INK}',width:2.2}},hovertemplate:sel+': %{{y:.2f}}<extra></extra>'}}],
    {{font:{J(FONT)},paper_bgcolor:'white',plot_bgcolor:'white',margin:{{l:50,r:10,t:30+26*Math.max(t.leaks.length,NAT.length,1),b:30}},hovermode:'x unified',
     legend:{{orientation:'h',x:0,y:-0.12}},xaxis:{{gridcolor:'#f0eee9'}},yaxis:{{title:'per 10M youth',rangemode:'tozero',gridcolor:'#f0eee9'}},
     shapes:sh,annotations:an}},{{responsive:true,displaylogo:false}});
}}
function pick(s){{sel=s; paint(); chart();}}
function setMode(i){{mode=i; [...chips.children].forEach((c,j)=>c.classList.toggle('on',j===i+1)); if(i>=0)sel=LK[i].state; paint(); chart();}}
[{{label:'All leaks'}},...LK].forEach((l,j)=>{{const c=document.createElement('button'); c.className='cb-chip'; c.textContent=l.label;
  c.onclick=()=>setMode(j-1); chips.appendChild(c);}});
setMode(-1);
}})();
</script>"""
    emit("atlas", "Every state, every leak",
         "Pick a leak to colour states by how much protest rose after it. Click a state for its weekly line.",
         body + script,
         "Colour: change in weekly non-party exam protests per 10M residents aged 15-29, weeks 0-2 vs weeks -8 to -1. Rank: among states with at least 5M aged 15-29. ACLED; Census 2011; DataMeet.")
    ATLAS = True
else:
    ATLAS = False
if TL and not ATLAS:
    first = max(TL, key=lambda t: (len(t["leaks"]), t["s"] == "Uttar Pradesh"))["s"]
    body = ('<div class="legend"><span><span class="sw" style="background:#fbe3d9;border:1px solid #e9b8a6"></span>'
            'state ran a leaked exam</span><span><span style="color:' + ACCENT + '">&#9474;</span> that state\'s leak</span>'
            '<span><span style="color:' + COOL + '">&#9478;</span> national leak</span>'
            '<span>Click a state to open it below.</span></div>'
            '<div id="tlGrid" class="grid tgrid"></div><div class="ctl" style="margin-top:12px"><span id="tlBig" class="big"></span></div>'
            '<div id="tlFig" style="height:330px"></div>')
    script = f"""
<script>
(function(){{
const T={J(TL)}, W={J(wks)}, IN={J([round(float(v), 3) for v in india.reindex(rate_all.columns).fillna(0)])}, NAT={J(NAT)}, CAP={TCAP:.3f};
const grid=document.getElementById('tlGrid'), big=document.getElementById('tlBig');
const n=W.length, xi=d=>{{let i=W.findIndex(w=>w>=d);return i<0?n-1:i;}};
const SW=100, SH=34, X=i=>(i/(n-1)*SW).toFixed(1), Yv=v=>(SH-Math.min(v,CAP)/CAP*(SH-3)).toFixed(1);
function spark(t){{
  let p='', on=false; t.y.forEach((v,i)=>{{if(v===null){{on=false;return;}} p+=(on?'L':'M')+X(i)+' '+Yv(v); on=true;}});
  const nat=NAT.map(l=>'<line x1="'+X(xi(l.d))+'" x2="'+X(xi(l.d))+'" y1="0" y2="'+SH+'" stroke="{COOL}" stroke-width="1" stroke-dasharray="2 2"/>').join('');
  const own=t.leaks.map(l=>'<line x1="'+X(xi(l.d))+'" x2="'+X(xi(l.d))+'" y1="0" y2="'+SH+'" stroke="{ACCENT}" stroke-width="2"/>').join('');
  return '<svg viewBox="0 0 '+SW+' '+SH+'" preserveAspectRatio="none" style="width:100%;height:60%;display:block">'+nat+own+
    '<path d="'+p+'" fill="none" stroke="{INK}" stroke-width="1.2" vector-effect="non-scaling-stroke"/></svg>';
}}
T.forEach(t=>{{const d=document.createElement('div'); d.className='tile'+(t.leaks.length?' hot':''); d.dataset.s=t.s;
  d.style.gridColumn=t.c; d.style.gridRow=t.r;
  d.title=t.s+(t.leaks.length?': '+t.leaks.map(l=>l.label).join('; '):'');
  d.innerHTML='<div class="nm">'+t.code+(t.leaks.length?' <span class="lk">'+t.leaks.length+(t.leaks.length>1?' leaks':' leak')+'</span>':'')+'</div>'+spark(t);
  d.onclick=()=>show(t.s); grid.appendChild(d);}});
function show(s){{
  const t=T.find(x=>x.s===s); [...grid.children].forEach(c=>c.classList.toggle('sel',c.dataset.s===s));
  big.innerHTML='<b>'+t.s+'</b>: non-party exam protests per 10M youth, 4-week average'+(t.leaks.length?'. Leaks: '+t.leaks.map(l=>l.label).join('; '):'. No state exam leak in our list');
  const sh=[], an=[];
  t.leaks.forEach((l,i)=>{{sh.push({{type:'line',xref:'x',yref:'paper',x0:l.d,x1:l.d,y0:0,y1:1,line:{{color:'{ACCENT}',width:2}}}});
    an.push({{xref:'x',yref:'paper',x:l.d,y:1.03+0.09*i,text:l.label,showarrow:false,xanchor:'left',font:{{size:11,color:'{ACCENT}'}}}});}});
  NAT.forEach((l,i)=>{{sh.push({{type:'line',xref:'x',yref:'paper',x0:l.d,x1:l.d,y0:0,y1:1,line:{{color:'{COOL}',width:1.5,dash:'dash'}}}});
    an.push({{xref:'x',yref:'paper',x:l.d,y:1.03+0.09*i,text:l.label,showarrow:false,xanchor:'right',font:{{size:11,color:'{COOL}'}}}});}});
  Plotly.react('tlFig',[
    {{x:W,y:IN,type:'scatter',mode:'lines',name:'India average',line:{{color:'{MUTED}',width:1.5,dash:'dot'}},hovertemplate:'India: %{{y:.2f}}<extra></extra>'}},
    {{x:W,y:t.y,type:'scatter',mode:'lines',name:t.s,line:{{color:'{INK}',width:2.2}},hovertemplate:t.s+': %{{y:.2f}}<extra></extra>'}}],
    {{font:{J(FONT)},paper_bgcolor:'white',plot_bgcolor:'white',margin:{{l:50,r:10,t:30+26*Math.max(t.leaks.length,NAT.length),b:30}},hovermode:'x unified',
     legend:{{orientation:'h',x:0,y:-0.12}},xaxis:{{gridcolor:'#f0eee9'}},yaxis:{{title:'per 10M youth',rangemode:'tozero',gridcolor:'#f0eee9'}},
     shapes:sh,annotations:an}},{{responsive:true,displaylogo:false}});
}}
show({J(first)});
}})();
</script>"""
    emit("tiles", "Every state, every week",
         "Each tile is a state. Red marks: its own leaks; blue: national leaks. Click a tile.",
         body + script,
         "Non-party exam protests per 10M residents aged 15-29, 4-week average. ACLED; Census 2011.")

# ---------------- rise: fixed 2.5D map, each state's own shape rises and falls week by week ----------------
if RINGS and TL:
    SP = [t for t in TL if t["s"] in RINGS]
    names = [t["s"] for t in SP]
    cap3 = float(np.quantile([v for t in SP for v in t["y"] if v and v > 0], 0.99)) if SP else 1.0
    nW = len(wks)
    def _win(d):
        i = next((j for j, w in enumerate(wks) if w >= d), None)
        return set(range(i, min(i + 4, nW))) if i is not None else set()
    natw = set().union(*[_win(l["d"]) for l in NAT]) if NAT else set()
    ownw = {t["s"]: set().union(*[_win(l["d"]) for l in t["leaks"]]) if t["leaks"] else set() for t in SP}
    HH = [[round(min(t["y"][j] or 0.0, cap3) / cap3, 3) for t in SP] for j in range(nW)]
    CC = ["".join("2" if j in ownw[t["s"]] else ("1" if j in natw else "0") for t in SP) for j in range(nW)]
    RAW3 = [[t["y"][j] for t in SP] for j in range(nW)]
    LEAKW = {}
    for t in SP:
        for l in t["leaks"]:
            for j in _win(l["d"]):
                LEAKW.setdefault(j, []).append(l["label"])
    for l in NAT:
        for j in _win(l["d"]):
            LEAKW.setdefault(j, []).append(l["label"] + " (national)")
    body = ('<div class="ctl"><button id="rsPlay" class="btn">&#9654; Play</button>'
            '<input id="rsSlider" type="range" min="0" value="0" style="flex:1;min-width:160px">'
            '</div><div id="rsBig" class="big" style="min-height:22px"></div>'
            '<div style="position:relative"><canvas id="rsCanvas" style="width:100%;display:block"></canvas>'
            '<div id="rsTip" class="rs-tip"></div></div>'
            '<div class="legend"><span><span class="sw" style="background:#cfc8bb"></span>height = exam protests</span>'
            '<span><span class="sw" style="background:' + ACCENT + '"></span>4 weeks after its own leak</span>'
            '<span><span class="sw" style="background:#7fb0c4"></span>4 weeks after a national leak</span></div>')
    script = f"""
<script>
(function(){{
const RG={J([RINGS[n_] for n_ in names])}, LP={J([LP[n_] for n_ in names])}, NM={J(names)}, W={J(wks)}, H={J(HH)}, C={J(CC)},
  RAW={J(RAW3)}, LW={J({str(k): v for k, v in LEAKW.items()})};
const TOP=['#e4dfd5','#a9cbd8','{ACCENT}'], WALL=['#b9b2a5','#6f9cb0','#8e3826'];
const cv=document.getElementById('rsCanvas'), ctx=cv.getContext('2d'), tip=document.getElementById('rsTip');
const sl=document.getElementById('rsSlider'), big=document.getElementById('rsBig'), btn=document.getElementById('rsPlay');
sl.max=W.length-1;
const TILT=0.58;
let xs=[],ys=[]; RG.forEach(rs=>rs.forEach(r=>{{for(let i=0;i<r.length;i+=2){{xs.push(r[i]);ys.push(r[i+1]);}}}}));
const X0=Math.min(...xs),X1=Math.max(...xs),Y0=Math.min(...ys),Y1=Math.max(...ys);
// orientation of each ring (screen y down): sign of the shoelace sum
const OR=RG.map(rs=>rs.map(r=>{{let a=0;for(let i=0;i<r.length;i+=2){{const j=(i+2)%r.length;a+=r[i]*r[j+1]-r[j]*r[i+1];}}return a>0?1:-1;}}));
// draw far (north) states first
const ORDER=RG.map((rs,n)=>[n,Math.max(...rs.flatMap(r=>r.filter((v,i)=>i%2)))]).sort((a,b)=>a[1]-b[1]).map(x=>x[0]);
let k=1,ox=0,oy=0,HMAX=100,cssW=0,paths=[];
function fit(){{cssW=cv.clientWidth||900; const cssH=Math.min(680,Math.round(cssW*0.78)); const dpr=window.devicePixelRatio||1;
  cv.style.height=cssH+'px'; cv.width=Math.round(cssW*dpr); cv.height=Math.round(cssH*dpr); ctx.setTransform(dpr,0,0,dpr,0,0);
  HMAX=cssH*0.3; k=Math.min((cssW-20)/(X1-X0),(cssH-HMAX-20)/((Y1-Y0)*TILT)); ox=(cssW-(X1-X0)*k)/2-X0*k; oy=HMAX+10-Y0*k*TILT;}}
const px=x=>ox+x*k, py=y=>oy+y*k*TILT;
function draw(h,c){{
  ctx.clearRect(0,0,cv.width,cv.height); paths=[];
  ctx.fillStyle='#f3f0ea'; ctx.strokeStyle='#ddd7cc'; ctx.lineWidth=0.6;
  RG.forEach(rs=>rs.forEach(r=>{{ctx.beginPath();ctx.moveTo(px(r[0]),py(r[1]));for(let i=2;i<r.length;i+=2)ctx.lineTo(px(r[i]),py(r[i+1]));ctx.closePath();ctx.fill();ctx.stroke();}}));
  ORDER.forEach(n=>{{const e=h[n]*HMAX, col=+c[n];
    RG[n].forEach((r,ri)=>{{const o=OR[n][ri];
      if(e>0.5){{ctx.fillStyle=WALL[col]; ctx.strokeStyle=WALL[col]; ctx.lineWidth=0.5;
        for(let i=0;i<r.length;i+=2){{const j=(i+2)%r.length, dx=r[j]-r[i], dy=r[j+1]-r[i+1], ny=o>0?-dx:dx;
          if(ny<=0)continue; const ax=px(r[i]),ay=py(r[i+1]),bx=px(r[j]),by=py(r[j+1]);
          ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.lineTo(bx,by-e);ctx.lineTo(ax,ay-e);ctx.closePath();ctx.fill();ctx.stroke();}}}}
    }});
    const p=new Path2D(); RG[n].forEach(r=>{{p.moveTo(px(r[0]),py(r[1])-e);for(let i=2;i<r.length;i+=2)p.lineTo(px(r[i]),py(r[i+1])-e);p.closePath();}});
    ctx.fillStyle=TOP[col]; ctx.fill(p); ctx.strokeStyle='#ffffff'; ctx.lineWidth=0.8; ctx.stroke(p); paths[n]=p;
    if(col==2){{ctx.font='600 12px IBM Plex Sans, Helvetica, Arial, sans-serif'; ctx.fillStyle='{INK}'; ctx.textAlign='center';
      ctx.fillText(NM[n],px(LP[n][0]),py(LP[n][1])-e-6);}}
  }});
}}
function label(j){{const r=RAW[j].map((v,n)=>[v||0,NM[n]]).sort((a,b)=>b[0]-a[0])[0], lk=LW[String(j)], d=new Date(W[j]+'T00:00:00');
  return '<b>'+d.toLocaleDateString('en-GB',{{day:'numeric',month:'short',year:'numeric'}})+'</b>. Highest: '+r[1]+' ('+r[0].toFixed(1)+' per 10M youth)'+
    (lk?'. <span style="color:{ACCENT}">Leak: '+lk.join('; ')+'</span>':'');}}
let cur=0, hNow=H[0].slice(), timer=null, raf=null;
function go(j,anim){{const from=hNow.slice(), to=H[j], t0=performance.now(), dur=anim?200:0; cur=j; sl.value=j; big.innerHTML=label(j);
  cancelAnimationFrame(raf);
  const step=t=>{{const u=dur?Math.min(1,(t-t0)/dur):1; hNow=from.map((v,n)=>v+(to[n]-v)*u); draw(hNow,C[j]); if(u<1)raf=requestAnimationFrame(step);}};
  raf=requestAnimationFrame(step);}}
function stop(){{clearInterval(timer);timer=null;btn.innerHTML='&#9654; Play';}}
btn.onclick=()=>{{if(timer){{stop();return;}} if(cur>=W.length-1)cur=-1; btn.innerHTML='&#10073;&#10073; Pause';
  timer=setInterval(()=>{{if(cur>=W.length-1){{stop();return;}} go(cur+1,true);}},240);}};
sl.oninput=()=>{{stop();go(+sl.value,false);}};
cv.addEventListener('mousemove',ev=>{{const b=cv.getBoundingClientRect(), x=ev.clientX-b.left, y=ev.clientY-b.top; let hit=-1;
  for(let q=ORDER.length-1;q>=0;q--){{const n=ORDER[q]; if(paths[n]&&ctx.isPointInPath(paths[n],x*(window.devicePixelRatio||1),y*(window.devicePixelRatio||1))){{hit=n;break;}}}}
  if(hit<0){{tip.style.display='none';return;}} const v=RAW[cur][hit];
  tip.innerHTML='<b>'+NM[hit]+'</b><br>'+(v===null?'no data':v.toFixed(2)+' per 10M youth'); tip.style.display='block';
  tip.style.left=(x+12)+'px'; tip.style.top=(y+12)+'px';}});
cv.addEventListener('mouseleave',()=>{{tip.style.display='none';}});
window.addEventListener('resize',()=>{{fit();draw(hNow,C[cur]);}});
fit(); go(0,false);
}})();
</script>"""
    emit("rise", "India, week by week",
         "Press Play. Each state rises with its exam protests; red marks the four weeks after its own leak.",
         body + script,
         "Height: non-party exam protests per 10M residents aged 15-29, 4-week average, capped at the 99th percentile. "
         "ACLED; Census 2011; DataMeet.")

# ---------------- national leaks: exposure vs response by state ----------------
exf = INT / "exposure.csv"
all_sh = pd.read_csv(RAW / "exam_shocks.csv", parse_dates=["exam_date", "shock_date"])
all_sh["t0"] = all_sh.exam_date.fillna(all_sh.shock_date)
nat_sh = all_sh[(all_sh.level == "national") & all_sh.t0.notna()].copy()
if exf.exists() and len(nat_sh):
    ex = pd.read_csv(exf, dtype={"district_id": str})
    st_w = pr.groupby(["state", "week"])[Y].sum().unstack(fill_value=0)
    panels, labels = [], []
    for r in nat_sh.itertuples():
        e = ex[ex.exam_id == r.exam_id]
        if e.empty:
            print(f"[26] national: {r.exam_id} has no exposure rows (run 03_exposure.py)")
            continue
        expo = e.groupby("state").A_tot.mean()
        if expo.std() < 1e-9 * max(expo.abs().max(), 1e-12):
            print(f"[26] national: {r.exam_id} has no state variation in exposure; left out")
            continue
        w0 = r.t0 - pd.Timedelta(days=r.t0.weekday())
        pre_w = [w for w in (w0 + pd.Timedelta(weeks=k) for k in range(-WINDOW_PRE, 0)) if w in st_w.columns]
        post_w = [w for w in (w0 + pd.Timedelta(weeks=k) for k in range(0, WINDOW_POST + 1)) if w in st_w.columns]
        if len(pre_w) < 4 or len(post_w) < 3:
            continue
        rate = lambda cols: st_w[cols].mean(axis=1) / pop_st.reindex(st_w.index) * 1e7
        dd = pd.DataFrame({"expo": expo, "change": rate(post_w) - rate(pre_w), "pop": pop_st}).dropna()
        dd = dd[dd["pop"] > 1e6]
        if len(dd) < 5:
            continue
        wts = dd["pop"] / dd["pop"].sum()
        xm, ym = (wts * dd.expo).sum(), (wts * dd.change).sum()
        slope = (wts * (dd.expo - xm) * (dd.change - ym)).sum() / (wts * (dd.expo - xm) ** 2).sum()
        xs = np.linspace(dd.expo.min(), dd.expo.max(), 20)
        placebo = r.shock_type != "cancelled"
        panels.append([
            {"type": "scatter", "mode": "markers+text", "x": dd.expo.round(2).tolist(), "y": dd.change.round(3).tolist(),
             "text": [s_ if p_ > 8e6 else "" for s_, p_ in zip(dd.index, dd["pop"])], "textposition": "top center",
             "textfont": {"size": 10, "color": MUTED}, "customdata": dd.index.tolist(),
             "marker": {"size": (np.sqrt(dd["pop"] / 1e6) * 4).round(1).tolist(), "color": MUTED if placebo else ACCENT,
                        "opacity": 0.75, "line": {"color": INK, "width": 0.6}},
             "hovertemplate": "%{customdata}<br>%{x:.1f} candidates per 1,000 youth"
                              "<br>change: %{y:+.2f} per 10M youth per week<extra></extra>", "name": "States"},
            {"type": "scatter", "mode": "lines", "x": xs.round(2).tolist(), "y": (ym + slope * (xs - xm)).round(3).tolist(),
             "line": {"color": INK, "dash": "dash", "width": 1.5}, "name": f"Population-weighted fit (slope {slope:+.3f})",
             "hoverinfo": "skip"}])
        labels.append(f"{nice(r.exam_id)}: " + ("leaked, not cancelled (placebo)" if placebo else "cancelled"))
    if panels:
        order_i = sorted(range(len(labels)), key=lambda i: "placebo" in labels[i])   # cancelled first
        panels = [panels[i] for i in order_i]
        labels = [labels[i] for i in order_i]
        tr = []
        for i, p_ in enumerate(panels):
            for t in p_:
                t["visible"] = i == 0
                tr.append(t)
        btns = [{"label": lab, "method": "update", "args": [{"visible": [j // 2 == i for j in range(len(tr))]}]}
                for i, lab in enumerate(labels)]
        L = lay(xaxis={"title": "Candidates per 1,000 residents aged 15-29 (state)", "gridcolor": "#eee"},
                yaxis={"title": "Change in non-party exam protests<br>per 10M youth per week",
                       "zeroline": True, "zerolinecolor": INK, "gridcolor": "#eee"},
                legend={"orientation": "h", "y": -0.22}, hovermode="closest", height=480,
                margin={"l": 70, "r": 16, "t": 50, "b": 46},
                updatemenus=[{"buttons": btns, "x": 0, "y": 1.14, "xanchor": "left", "direction": "down"}])
        emit("national", "NEET-UG: did states with more candidates protest more?",
             "Each bubble is a state. Switch to NEET-UG 2024 (leaked, not cancelled) to compare.",
             '<div id="nat" style="height:480px"></div>'
             + f"<script>Plotly.newPlot('nat',{J(tr)},{J(L)},{{responsive:true,displaylogo:false}});</script>",
             "Change: weekly non-party exam protests per 10M aged 15-29, weeks 0-8 minus weeks -8 to -1. States under 1M aged 15-29 omitted. Candidates: NTA. ACLED; Census 2011.")

# ---------------- calendar ----------------
wk = pr.groupby("week")[Y].sum()
wk = wk.reindex(pd.date_range(wk.index.min(), wk.index.max(), freq="7D"), fill_value=0)
top = pr.groupby(["week", "state"])[Y].sum().reset_index().sort_values(Y, ascending=False).drop_duplicates("week").set_index("week")
cal = pd.DataFrame({"n": wk})
cal["year"] = cal.index.year
cal["wk"] = cal.index.isocalendar().week.astype(int)
cal["top"] = [f"{top.state.get(d, '')} ({int(top[Y].get(d, 0))})" if d in top.index and top[Y].get(d, 0) > 0 else "none"
              for d in cal.index]
years = sorted(cal.year.unique())
Z = [[None] * 53 for _ in years]
TXT = [[""] * 53 for _ in years]
for d, r in cal.iterrows():
    yi, wi = years.index(r.year), min(int(r.wk), 53) - 1
    Z[yi][wi] = int(r.n)
    TXT[yi][wi] = f"Week of {d:%d %b %Y}<br>{int(r.n)} non-party exam protests<br>Top state: {r.top}"
# triangles as a scatter trace on numeric rows (category axes misplace annotations)
lx, ly, lt = [], [], []
for rr in raw_sh.itertuples():
    yr = rr.wk0.year
    if yr in years:
        lx.append(min(int(rr.wk0.isocalendar()[1]), 53)); ly.append(years.index(yr)); lt.append(nice(rr.exam_id))
MONTHS = {1: "Jan", 5: "Feb", 9: "Mar", 14: "Apr", 18: "May", 22: "Jun", 27: "Jul", 31: "Aug", 35: "Sep",
          40: "Oct", 44: "Nov", 48: "Dec"}
tr = [{"type": "heatmap", "z": Z, "x": list(range(1, 54)), "y": list(range(len(years))), "text": TXT,
       "hovertemplate": "%{text}<extra></extra>", "xgap": 3, "ygap": 6, "hoverongaps": False,
       "colorscale": [[0, "#f3f1ec"], [0.08, "#f4d6a6"], [0.35, HOT], [1, ACCENT]],
       "colorbar": {"thickness": 10, "title": {"text": "Events"}, "len": 0.9}},
      {"type": "scatter", "mode": "markers", "x": lx, "y": ly, "text": lt,
       "marker": {"symbol": "triangle-down", "size": 11, "color": INK, "line": {"color": "white", "width": 1.2}},
       "hovertemplate": "Leak: %{text}<extra></extra>", "showlegend": False}]
cal_layout = lay(xaxis={"tickvals": list(MONTHS), "ticktext": list(MONTHS.values()), "range": [0.5, 53.5],
                        "showgrid": False, "zeroline": False},
                 yaxis={"tickvals": list(range(len(years))), "ticktext": [str(y) for y in years],
                        "autorange": "reversed", "showgrid": False, "zeroline": False},
                 margin={"l": 56, "r": 16, "t": 10, "b": 36}, hovermode="closest")
body = f'<div id="calFig" style="height:{70 + 58 * len(years)}px"></div>'
script = f"<script>Plotly.newPlot('calFig',{J(tr)},{J(cal_layout)},{{responsive:true,displaylogo:false}});</script>"
emit("calendar", "A calendar of discontent",
     "Each row is a year and each square a week. Darker squares are weeks with more non-party exam protests across "
     "India. Black markers sit on the exam weeks of the leaks. Hover on a square to see which state led that week.",
     body + script, CITE)

# ---------------- event study: the estimate itself, week by week, with intervals ----------------
ES_SER = [(OUT / f"es_main_total_{Y}.csv", "Non-party exam protests (main)", ACCENT, 0.0),
          (OUT / "es_main_total_n_exam_strict_party.csv", "Party-led exam protests", "#8d8a83", -0.18),
          (OUT / f"es_placebo_total_{Y}.csv", "Placebo: NEET-UG 2024 (leaked, not cancelled)", COOL, 0.18)]
tr = []
for f_, nm_, col_, off_ in ES_SER:
    if not f_.exists():
        continue
    t_ = pd.read_csv(f_)
    if "sex" in t_:
        t_ = t_[t_.sex == "tot"]
    t_ = t_.sort_values("k")
    sc_ = 1000
    tr.append({"type": "scatter", "mode": "markers", "name": nm_, "x": (t_.k + off_).round(2).tolist(),
               "y": (t_.coef * sc_).round(4).tolist(),
               "error_y": {"type": "data", "symmetric": False, "array": ((t_.ci_hi - t_.coef) * sc_).round(4).tolist(),
                           "arrayminus": ((t_.coef - t_.ci_lo) * sc_).round(4).tolist(), "color": col_, "thickness": 1.5, "width": 0},
               "marker": {"color": col_, "size": 8, "line": {"color": "white", "width": 1}},
               "customdata": t_.k.tolist(), "hovertemplate": nm_ + "<br>week %{customdata}: %{y:.3f}<extra></extra>",
               "visible": True if off_ == 0 else "legendonly"})
if tr:
    L = lay(xaxis={"title": "Weeks from exam date", "dtick": 2, "zeroline": False, "gridcolor": "#f0eee9"},
            yaxis={"title": "Extra protests per 1,000 district-weeks", "zeroline": True, "zerolinecolor": INK, "gridcolor": "#f0eee9"},
            legend={"orientation": "h", "y": -0.22}, hovermode="closest",
            shapes=[{"type": "rect", "x0": -0.5, "x1": 2.5, "yref": "paper", "y0": 0, "y1": 1, "fillcolor": "rgba(200,85,61,0.07)", "line": {"width": 0}},
                    {"type": "line", "x0": -0.5, "x1": -0.5, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": INK, "width": 1, "dash": "dash"}}])
    emit("eventstudy", "The effect, week by week",
         "Extra exam protests in a district for each extra applicant per 1,000 young residents. Click the legend to add the party-led and placebo series.",
         '<div id="es" style="height:440px"></div>' + f"<script>Plotly.newPlot('es',{J(tr)},{J(L)},{{responsive:true,displaylogo:false}});</script>",
         "Stacked event study, district and week fixed effects per shock, re-centred on the pre-period mean; bars: 95% "
         "intervals clustered by state (indicative; headline p from randomisation). ACLED; Census 2011.")

# ---------------- unusual (randomisation inference) ----------------
rf = OUT / f"ri_shifts_{Y}.csv"
if rf.exists():
    r = pd.read_csv(rf)
    b = float(r.b_real.iloc[0])
    p = (np.sum(np.abs(r.b_fake) >= abs(b)) + 1) / (len(r) + 1)
    wk0_cols = [c for c in r.columns if c.startswith("wk0_")]
    hover = ["Shift of %d weeks<br>estimate %.5f" % (o, v) for o, v in zip(r.offset_weeks, r.b_fake)]
    jitter = np.random.default_rng(1).uniform(-0.35, 0.35, len(r)).round(3).tolist()
    tr = [{"type": "scatter", "mode": "markers", "x": r.b_fake.round(6).tolist(), "y": jitter, "text": hover,
           "hovertemplate": "%{text}<extra></extra>", "marker": {"size": 8, "color": MUTED, "opacity": 0.55},
           "name": f"{len(r)} false exam dates"},
          {"type": "scatter", "mode": "markers+text", "x": [b], "y": [0], "text": ["real dates"], "textposition": "top center",
           "marker": {"size": 18, "color": ACCENT, "symbol": "diamond", "line": {"color": INK, "width": 1}},
           "name": "Real exam dates", "hovertemplate": "Real estimate %{x:.5f}<extra></extra>"}]
    L = lay(xaxis={"title": "Estimated effect (extra events per district-week per applicant per 1,000 youth)",
                   "zeroline": True, "zerolinecolor": INK, "gridcolor": "#eee"},
            yaxis={"visible": False, "range": [-0.8, 0.8]}, legend={"orientation": "h", "y": -0.3}, height=330,
            annotations=[{"x": b, "y": 0.62, "text": f"Only {np.mean(np.abs(r.b_fake) >= abs(b)) * 100:.0f}% of false dates "
                          f"give an effect this large (p = {p:.3f})", "showarrow": False, "font": {"color": ACCENT}}])
    body = '<div id="ri" style="height:330px"></div>'
    emit("unusual", "Could the result be luck?",
         "Grey: the same analysis with every exam date moved. Diamond: the real dates.",
         body + f"<script>Plotly.newPlot('ri',{J(tr)},{J(L)},{{responsive:true,displaylogo:false}});</script>",
         "All nine dates moved together by the same number of weeks. ACLED; Census 2011.")

# ---------------- spec curve ----------------
sp = INT / "stacked_main.csv"
if sp.exists():
    df = pd.read_csv(sp, dtype={"district_id": str, "stack_id": str})
    df = df[df.groupby("stack_id").A_tot.transform("std") > 0]
    outs = [c for c in ["n_exam_strict_nonparty", "n_exam_strict_nonparty_broad", "n_exam_strict", "n_exam"] if c in df]
    lab_out = {"n_exam_strict_nonparty": "Non-party (main)", "n_exam_strict_nonparty_broad": "Non-party, broad party list",
               "n_exam_strict": "All exam protests", "n_exam": "Keyword measure"}
    stacks = ["none"] + sorted(df.stack_id.unique())
    rows = []

    def did(d, yy):
        Z_ = demean(np.column_stack([yy, d.A_tot * d.post, d.other_post_exp]), [d.fe_de, d.fe_stw])
        t = ols_cluster(Z_[:, 0], Z_[:, 1:], d.state, ["x", "o"])["table"].iloc[0]
        return t.coef, t.ci_lo, t.ci_hi

    for o in outs:
        for drop in stacks:
            d = df if drop == "none" else df[df.stack_id != drop]
            for trend in [False, True]:
                yy = d[o].astype(float).copy()
                if trend:
                    for e, gg in d.groupby("stack_id"):
                        pre = gg[gg.k < 0]
                        Zp = demean(np.column_stack([pre[o], pre.A_tot * pre.k]), [pre.fe_de, pre.fe_stw])
                        if not np.allclose(Zp[:, 1], 0):
                            s_ = float(np.linalg.lstsq(Zp[:, 1:], Zp[:, 0], rcond=None)[0][0])
                            yy.loc[gg.index] = gg[o] - s_ * gg.A_tot * gg.k
                c_, lo_, hi_ = did(d, yy)
                rows.append({"outcome": lab_out[o], "dropped": "none" if drop == "none" else nice(drop),
                             "trend": "pre-period trend removed" if trend else "no trend adjustment",
                             "b": c_, "lo": lo_, "hi": hi_})
    s = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
    s.to_csv(OUT / "spec_curve.csv", index=False)
    s["i"] = range(len(s))
    main = (s.outcome == "Non-party (main)") & (s.dropped == "none") & (s.trend == "no trend adjustment")
    hov = (s.outcome + "<br>dropped: " + s.dropped + "<br>" + s.trend + "<br>b = " + s.b.map("{:.5f}".format)).tolist()
    colors = [ACCENT if m_ else (COOL if lo_ > 0 else MUTED) for m_, lo_ in zip(main, s.lo)]
    tr = [{"type": "scatter", "mode": "markers", "x": s.i.tolist(), "y": s.b.tolist(), "text": hov,
           "hovertemplate": "%{text}<extra></extra>", "marker": {"color": colors, "size": [11 if m_ else 6 for m_ in main]},
           "error_y": {"type": "data", "symmetric": False, "array": (s.hi - s.b).tolist(), "arrayminus": (s.b - s.lo).tolist(),
                       "color": "#c9c9c9", "thickness": 1, "width": 0}, "xaxis": "x", "yaxis": "y", "showlegend": False}]
    rows_lab = [("Outcome", o) for o in s.outcome.unique()] + [("Trend", t) for t in s.trend.unique()] + \
               [("Dropped", d_) for d_ in ["none"] + sorted(set(s.dropped) - {"none"})]
    for j, (kind, val) in enumerate(rows_lab):
        col = {"Outcome": "outcome", "Trend": "trend", "Dropped": "dropped"}[kind]
        m_ = s[col] == val
        tr.append({"type": "scatter", "mode": "markers", "x": s.i[m_].tolist(), "y": [j] * int(m_.sum()),
                   "marker": {"symbol": "square", "size": 5, "color": INK}, "hoverinfo": "skip", "xaxis": "x", "yaxis": "y2",
                   "showlegend": False})
    L = lay(grid={"rows": 2, "columns": 1}, height=640, margin={"l": 210, "r": 16, "t": 16, "b": 30},
            yaxis={"domain": [0.48, 1], "title": "Estimate (95% CI)", "zeroline": True, "zerolinecolor": INK, "gridcolor": "#eee"},
            yaxis2={"domain": [0, 0.44], "tickvals": list(range(len(rows_lab))), "ticktext": [f"{k}: {v}" for k, v in rows_lab],
                    "autorange": "reversed", "tickfont": {"size": 10}},
            xaxis={"visible": False})
    share_pos = (s.lo > 0).mean()
    body = '<div id="spec" style="height:640px"></div>'
    emit("speccurve", f"The result across {len(s)} ways to run it",
         "Each dot is one version of the analysis with its 95% interval; the panel below shows the choices. Orange: the main version.",
         body + f"<script>Plotly.newPlot('spec',{J(tr)},{J(L)},{{responsive:true,displaylogo:false}});</script>",
         "State-clustered intervals, indicative only. ACLED; Census 2011.")

# ---------------- courts ----------------
st_es = OUT / f"es_main_total_{Y}.csv"
ct_es = next(iter(sorted(OUT.glob("es_state_main_n_orders_body_post*.csv"))), None)
if st_es.exists() and ct_es is not None:
    a, c = pd.read_csv(st_es).sort_values("k"), pd.read_csv(ct_es).sort_values("k")
    sc = lambda t: (t.coef / max(t.coef.abs().max(), 1e-12))
    tr = [{"type": "scatter", "mode": "lines+markers", "x": a.k.tolist(), "y": sc(a).round(3).tolist(), "name": "Street: exam protests",
           "line": {"color": ACCENT, "width": 3}, "hovertemplate": "week %{x}<br>street response (scaled) %{y:.2f}<extra></extra>"},
          {"type": "scatter", "mode": "lines+markers", "x": c.k.tolist(), "y": sc(c).round(3).tolist(), "name": "Courts: cases against exam bodies",
           "line": {"color": COOL, "width": 3, "dash": "dot"}, "hovertemplate": "week %{x}<br>court response (scaled) %{y:.2f}<extra></extra>"}]
    L = lay(xaxis={"title": "Weeks from exam date", "zeroline": True, "zerolinecolor": INK},
            yaxis={"title": "Response, scaled to its own peak", "gridcolor": "#eee", "zeroline": True},
            legend={"orientation": "h", "y": -0.25}, hovermode="x unified",
            shapes=[{"type": "rect", "x0": 0, "x1": 2, "yref": "paper", "y0": 0, "y1": 1, "fillcolor": "rgba(200,85,61,0.08)", "line": {"width": 0}},
                    {"type": "rect", "x0": 10, "x1": 25, "yref": "paper", "y0": 0, "y1": 1, "fillcolor": "rgba(31,111,139,0.08)", "line": {"width": 0}}])
    emit("courts", "Protest comes in weeks, court cases in months",
         "Each line is scaled to its own peak.",
         '<div id="courts" style="height:400px"></div>' + f"<script>Plotly.newPlot('courts',{J(tr)},{J(L)},{{responsive:true,displaylogo:false}});</script>",
         "Street: district event study. Courts: state event study, 26 weeks. Dattam Labs (CC BY 4.0); ACLED.")

# ---------------- docket: every case against an exam body, as a stack of case files ----------------
cf = INT / "court_orders_labeled.csv"
csf = RAW / "crosswalks" / "court_state.csv"
if cf.exists():
    import re as _re
    from config import EXAM_BODY_REGEX, COURT_DATE
    co = pd.read_csv(cf, low_memory=False, dtype={"court_code": str})
    co["court_code"] = co.court_code.map(lambda x: _re.sub(r"[^0-9]+", "_", str(x)).strip("_"))
    if csf.exists():
        cs = pd.read_csv(csf, dtype={"court_code": str})[["court_code", "state"]]
        co = co.merge(cs, on="court_code", how="left")
    else:
        co["state"] = ""
    dcol = COURT_DATE if COURT_DATE in co else "decision_date"
    dt = pd.to_datetime(co[dcol], errors="coerce", dayfirst=(dcol != "decision_date"))
    if dcol != "decision_date" and "decision_date" in co:
        dt = dt.fillna(pd.to_datetime(co.decision_date, errors="coerce"))
    co["month"] = dt.dt.to_period("M").dt.to_timestamp()
    resp = co.get("respondent", pd.Series("", index=co.index)).fillna("").astype(str)
    co["body"] = resp.str.contains(EXAM_BODY_REGEX, case=False, regex=True)
    sex = co.get("petitioner_sex", pd.Series("", index=co.index)).fillna("unknown").str.lower()
    co["sex"] = np.where(sex.eq("female"), "Woman", np.where(sex.eq("male"), "Man", "Not identifiable"))
    lo, hi = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
    sets = {}
    for key, sub in [("body", co[co.body]), ("all", co)]:
        sub = sub.dropna(subset=["month"])
        sub = sub[(sub.month >= lo) & (sub.month <= hi)]
        order_s = {"Woman": 0, "Man": 1, "Not identifiable": 2}
        sub = sub.assign(o=sub.sex.map(order_s)).sort_values(["month", "o"])
        unit = 1 if len(sub) <= 1500 else int(np.ceil(len(sub) / 1500))
        pts = {"x": [], "y": [], "s": [], "t": []}
        for m, gm in sub.groupby("month"):
            cnt = gm.sex.value_counts()
            yi = 0
            for sx in ["Woman", "Man", "Not identifiable"]:
                k_ = int(round(cnt.get(sx, 0) / unit))
                for _ in range(k_):
                    pts["x"].append(m.strftime("%Y-%m-15")); pts["y"].append(yi); pts["s"].append(sx)
                    pts["t"].append(f"{m:%b %Y}: {len(gm)} cases filed, {sx.lower()} petitioner")
                    yi += 1
        known = sub[sub.sex != "Not identifiable"]
        fem = float((known.sex == "Woman").mean()) if len(known) else float("nan")
        sets[key] = {"pts": pts, "unit": unit, "n": int(len(sub)), "fem": round(100 * fem, 1)}
    leaks = [{"x": r.t0.strftime("%Y-%m-%d"), "label": nice(r.exam_id)} for r in raw_sh.itertuples()
             if lo <= r.t0 <= hi]
    body = ('<div class="dk-head"><div class="dk-tabs" id="dkTabs"></div><div class="dk-stat" id="dkStat"></div></div>'
            '<div id="dkFig" style="height:430px"></div>')
    script = f"""
<script>
(function(){{
const S={J(sets)}, L={J(leaks)}; const COL={{'Woman':'#8a3fd1','Man':'#1a7fb0','Not identifiable':'#cfcac0'}};
const tabs=document.getElementById('dkTabs'), stat=document.getElementById('dkStat');
function draw(key){{
  const s=S[key]; [...tabs.children].forEach(b=>b.classList.toggle('on',b.dataset.k===key));
  stat.innerHTML='<b>'+(isFinite(s.fem)?s.fem.toFixed(0)+'%':'n/a')+'</b> of petitioners whose sex can be identified are women. '+s.n.toLocaleString()+' cases; '+
    (s.unit>1?'each square = '+s.unit+' cases':'each square = 1 case');
  const tr=['Woman','Man','Not identifiable'].map(sx=>{{const idx=s.pts.s.map((v,i)=>v===sx?i:-1).filter(i=>i>=0);
    return {{type:'scatter',mode:'markers',x:idx.map(i=>s.pts.x[i]),y:idx.map(i=>s.pts.y[i]),text:idx.map(i=>s.pts.t[i]),name:sx,
      marker:{{symbol:'square',size:key==='body'?7:4,color:COL[sx],line:{{width:key==='body'?1:0,color:'white'}}}},
      hovertemplate:'%{{text}}<extra></extra>'}};}});
  const ymax=Math.max(5,...s.pts.y)+3;
  const shapes=L.map(l=>({{type:'line',x0:l.x,x1:l.x,yref:'paper',y0:0,y1:0.94,line:{{color:'#1d1d2b',width:1,dash:'dot'}}}}));
  const ann=L.map((l,i)=>({{x:l.x,yref:'paper',y:0.95+(i%2)*0.04,text:l.label.split(',')[0],showarrow:false,font:{{size:10,color:'#6b7280'}},xanchor:'left'}}));
  Plotly.react('dkFig',tr,{{font:{J(FONT)},paper_bgcolor:'white',plot_bgcolor:'white',margin:{{l:44,r:10,t:40,b:40}},hovermode:'closest',
    xaxis:{{type:'date',gridcolor:'#f0eee9'}},yaxis:{{title:(s.unit>1?'cases ('+s.unit+' per square)':'cases filed that month'),range:[-1,ymax],gridcolor:'#f0eee9',zeroline:false}},
    legend:{{orientation:'h',y:-0.12}},shapes:shapes,annotations:ann}},{{responsive:true,displaylogo:false}});
}}
[['body','Cases against an exam body'],['all','All exam-related cases']].forEach(([k,l])=>{{const b=document.createElement('button');
  b.className='dk-tab';b.dataset.k=k;b.textContent=l;b.onclick=()=>draw(k);tabs.appendChild(b);}});
draw('body');
}})();
</script>"""
    emit("docket", "Who takes exam disputes to court?",
         "Each square is a case, by filing month and the first petitioner's sex.",
         body + script,
         "High Court and Supreme Court judgments (Dattam Labs, CC BY 4.0). Sex inferred from names by a language model; no names shown.")

# ---------------- women 1: out of every 50 -- applicants, protesters, petitioners, and reports ----------------
WOM, MEN_C = "#8a3fd1", "#1a7fb0"
U_rows = []
pool = INT / "pool_shares.csv"
if pool.exists():
    pp = pd.read_csv(pool, dtype={"exam_id": str})
    for e, lab in [("UP_CONST_2024", "UP police constable applicants, 2024"), ("NEETUG_2024", "NEET-UG candidates, 2024")]:
        r_ = pp[pp.exam_id == e]
        if len(r_) and pd.notna(r_.male_share.iloc[0]):
            U_rows.append({"lab": lab, "v": 100 * (1 - float(r_.male_share.iloc[0])), "kind": "person",
                           "src": "Official applicant counts", "what": "women among applicants"})
ss = OUT / "survey_shares.csv"
if ss.exists():
    q = pd.read_csv(ss)
    q = q[q.measure.str.contains("demonstration")].set_index("group").share
    if {"men", "women"} <= set(q.index):
        U_rows.append({"lab": "Adults who have joined a demonstration", "v": 100 * q["women"] / (q["men"] + q["women"]),
                       "kind": "person", "src": "World Values Survey 2023; any demonstration; assumes equal numbers of men and women",
                       "what": "women among people who have protested"})
rc = OUT / "repertoire_compare.csv"
RCQ = None
if rc.exists():
    RCQ = pd.read_csv(rc)
    q = RCQ[RCQ.scope == "all states"].set_index("measure")
    if "Court: all exam cases" in q.index:
        U_rows.append({"lab": "Petitioners in exam court cases", "v": 100 * float(q.loc["Court: all exam cases", "female_share"]),
                       "kind": "person", "src": "High Court and Supreme Court judgments; petitioners whose sex is identifiable",
                       "what": "women among petitioners"})
    k_ = [m_ for m_ in q.index if m_.startswith("Street: exam protests (women mentioned")]
    if k_:
        U_rows.append({"lab": "Exam-protest reports", "v": 100 * float(q.loc[k_[0], "female_share"]), "kind": "report",
                       "src": "ACLED event descriptions", "what": "reports that mention any woman"})
if U_rows:
    body = '<div id="uGrid" class="u-grid"></div>'
    script = f"""
<script>
(function(){{
const R={J(U_rows)}, g=document.getElementById('uGrid');
const PERSON='<svg viewBox="0 0 10 20"><circle cx="5" cy="3.4" r="3"/><path d="M1 8.5q0-1.8 1.8-1.8h4.4Q9 6.7 9 8.5V14H7.2v6H2.8v-6H1z"/></svg>';
const DOC='<svg viewBox="0 0 14 18"><path d="M1 1h8l4 4v12H1z"/></svg>';
R.forEach(r=>{{const n=Math.round(r.v/2); let icons='';
  for(let i=0;i<50;i++)icons+='<span class="u-ic'+(i<n?' on':'')+'">'+(r.kind==='person'?PERSON:DOC)+'</span>';
  const d=document.createElement('div'); d.className='u-row'+(r.kind==='report'?' rep':'');
  d.innerHTML='<div class="u-lab">'+r.lab+'<span>'+r.src+'</span></div><div class="u-icons" title="'+r.v.toFixed(1)+'% '+r.what+'">'+icons+
    '</div><div class="u-val"><b>'+r.v.toFixed(r.v<10?1:0)+'%</b><span>'+r.what+'</span></div>';
  g.appendChild(d);}});
}})();
</script>"""
    emit("women", "Where are the women?",
         "Out of every 50 people, purple are women. The last row counts reports, not people.",
         body + script,
         "Each row is a different source and measure. The last row is the share of exam-protest reports that mention any "
         "woman, which is not the share of protesters who are women.")

# ---------------- women 2: how exam protests are reported ----------------
RB = []
rcx = OUT / "reporting_checks.csv"
if RCQ is not None:
    q = RCQ[RCQ.scope == "all states"].set_index("measure")
    a_ = [m_ for m_ in q.index if m_.startswith("Street: exam protests (women mentioned")]
    b_ = [m_ for m_ in q.index if m_.startswith("Street: all protests (women mentioned")]
    if a_ and b_:
        RB.append({"t": "Report mentions a woman", "bars": [
            {"g": "Exam protests", "v": float(q.loc[a_[0], "female_share"]), "lo": float(q.loc[a_[0], "ci_lo"]), "hi": float(q.loc[a_[0], "ci_hi"]), "n": int(q.loc[a_[0], "n"])},
            {"g": "All protests", "v": float(q.loc[b_[0], "female_share"]), "lo": float(q.loc[b_[0], "ci_lo"]), "hi": float(q.loc[b_[0], "ci_hi"]), "n": int(q.loc[b_[0], "n"])}]})
if rcx.exists():
    q = pd.read_csv(rcx)
    def _bars(check, groups, names):
        out = []
        for g_, nm in zip(groups, names):
            r_ = q[(q.check == check) & (q.group == g_)]
            if len(r_):
                r_ = r_.iloc[0]
                out.append({"g": nm, "v": float(r_.share), "lo": float(r_.ci_lo), "hi": float(r_.ci_hi), "n": int(r_.n)})
        return out
    b = _bars("b sex-neutral nouns only", ["exam events", "other events"], ["Exam protests", "Other events"])
    if len(b) == 2:
        RB.append({"t": "Protesters described only as students, aspirants or youth", "bars": b})
    b = _bars("a police intervention / excessive force", ["exam protests", "other protests"], ["Exam protests", "Other protests"])
    if len(b) == 2:
        RB.append({"t": "Police intervened or used force", "bars": b})
    b = _bars("e women mentioned", ["India: student events", "Bangladesh: student events", "Nepal: student events"],
              ["India", "Bangladesh", "Nepal"])
    if len(b) >= 2:
        RB.append({"t": "Student protest reports that mention a woman", "bars": b})
if RB:
    body = '<div id="rbGrid" class="rb-grid"></div>'
    script = f"""
<script>
(function(){{
const P={J(RB)}, g=document.getElementById('rbGrid'); const pct=v=>(100*v).toFixed(v<0.1?1:0)+'%';
P.forEach(p=>{{const mx=Math.max(...p.bars.map(b=>b.hi))*1.1;
  const d=document.createElement('div'); d.className='rb-panel';
  d.innerHTML='<div class="rb-t">'+p.t+'</div>'+p.bars.map((b,i)=>'<div class="rb-row" title="'+pct(b.v)+' (95% CI '+pct(b.lo)+' to '+pct(b.hi)+'), n = '+b.n.toLocaleString()+'">'+
    '<span class="rb-g">'+b.g+'</span><span class="rb-track"><span class="rb-bar'+(i===0?' hi':'')+'" style="width:'+(100*b.v/mx)+'%"></span>'+
    '<span class="rb-ci" style="left:'+(100*b.lo/mx)+'%;width:'+(100*(b.hi-b.lo)/mx)+'%"></span></span><span class="rb-v">'+pct(b.v)+'</span></div>').join('');
  g.appendChild(d);}});
}})();
</script>"""
    emit("reports", "How exam protests get reported",
         "Exam protests against other events in the same reports. Hover on a bar for its interval.",
         body + script,
         "Shares of ACLED events, India 2023-2026 unless named; thin line: 95% interval. Regional row: student events in "
         "each country's ACLED data, same window.")

# ---------------- women 3: survey answers by sex ----------------
if ss.exists():
    q = pd.read_csv(ss)
    SV = []
    for meas, title_, sub in [
        ("WVS: ever attended a peaceful demonstration", "Has attended a peaceful demonstration", "World Values Survey, 2023"),
        ("WVS: men should have more right to a job (agree)", "Agrees: when jobs are scarce, men should get them first", "World Values Survey, 2023"),
        ("Pew: men should have more right to a job (agree)", "Agrees: when jobs are scarce, men should get them first", "Pew Research Center, 2019-20")]:
        d = q[q.measure == meas].set_index("group")
        rows_ = []
        for lab, gm, gw in [("All adults", "men", "women"), ("Young adults", "men 18-29", "women 18-29"),
                            ("Young adults", "young men", "young women")]:
            if gm in d.index and gw in d.index:
                rows_.append({"lab": lab, "m": float(d.loc[gm, "share"]), "w": float(d.loc[gw, "share"]),
                              "mlo": float(d.loc[gm, "ci_lo"]), "mhi": float(d.loc[gm, "ci_hi"]),
                              "wlo": float(d.loc[gw, "ci_lo"]), "whi": float(d.loc[gw, "ci_hi"])})
        if rows_:
            SV.append({"t": title_, "sub": sub, "rows": rows_})
    if SV:
        body = ('<div class="legend"><span><span class="sw" style="background:' + MEN_C + ';border-radius:50%"></span>men</span>'
                '<span><span class="sw" style="background:' + WOM + ';border-radius:50%"></span>women</span></div>'
                '<div id="svGrid" class="rb-grid"></div>')
        script = f"""
<script>
(function(){{
const P={J(SV)}, g=document.getElementById('svGrid'); const pct=v=>(100*v).toFixed(0)+'%';
P.forEach(p=>{{const mx=Math.min(1,Math.max(...p.rows.flatMap(r=>[r.mhi,r.whi]))*1.15);
  const d=document.createElement('div'); d.className='rb-panel';
  d.innerHTML='<div class="rb-t">'+p.t+'</div><div class="rb-s">'+p.sub+'</div>'+p.rows.map(r=>{{
    const a=Math.min(r.m,r.w)/mx*100, b=Math.max(r.m,r.w)/mx*100;
    return '<div class="rb-row" title="men '+pct(r.m)+' ('+pct(r.mlo)+' to '+pct(r.mhi)+'); women '+pct(r.w)+' ('+pct(r.wlo)+' to '+pct(r.whi)+')">'+
      '<span class="rb-g">'+r.lab+'</span><span class="rb-track dumb"><span class="db-line" style="left:'+a+'%;width:'+(b-a)+'%"></span>'+
      '<span class="db-dot" style="left:'+(r.m/mx*100)+'%;background:{MEN_C}"></span><span class="db-dot" style="left:'+(r.w/mx*100)+'%;background:{WOM}"></span></span>'+
      '<span class="rb-v"><span style="color:{MEN_C}">'+pct(r.m)+'</span> / <span style="color:{WOM}">'+pct(r.w)+'</span></span></div>';}}).join('');
  g.appendChild(d);}});
}})();
</script>"""
        emit("surveys", "Men and women, in their own words",
             "Survey answers by sex. Hover on a row for 95% intervals.",
             body + script,
             "Weighted shares. Young adults: aged 18-29 (WVS) or the youngest Pew age band. All protests, not only exam protests.")

# ---------------- search: a search box for each group of terms ----------------
mf = OUT / "trends_mde.csv"
tf = INT / "trends_index.csv"
if mf.exists():
    try:
        from config import TRENDS_GROUPS
    except ImportError:
        TRENDS_GROUPS = {}
    m_ = pd.read_csv(mf).set_index("outcome")
    groups = [("trends_control", "control", "Searches about the leak itself", "the control: these should rise"),
              ("trends_manosphere", "manosphere", "Manosphere searches", ""),
              ("trends_mra", "mra", "Men's-rights searches", ""),
              ("trends_gender_jobs", "gender_jobs", "Searches about women's quotas", "")]
    curves = {}
    if tf.exists():
        td = pd.read_csv(tf, parse_dates=["week"])
        td["week"] = td.week - pd.to_timedelta(td.week.dt.weekday, unit="D")
        for col, *_ in groups:
            if col not in td:
                continue
            P = td.pivot_table(index="week", columns="state", values=col)
            rows = []
            for r in raw_sh[raw_sh.level == "state"].itertuples():
                if r.state not in P.columns:
                    continue
                ks = range(-8, 9)
                ws = [r.wk0 + pd.Timedelta(weeks=k) for k in ks]
                if sum(w in P.index for w in ws) < 12:
                    continue
                sub = P.reindex(ws)
                diff = (sub[r.state] - sub.drop(columns=r.state).mean(axis=1)).to_numpy()
                pre = np.nanmean(diff[:8])
                rows.append(diff - pre)
            if rows:
                avg = np.nanmean(np.vstack(rows), axis=0)
                curves[col] = {"k": list(range(-8, 9)), "y": [None if np.isnan(v) else round(100 * (np.exp(v) - 1), 2) for v in avg],
                               "n": len(rows)}
    series = {}
    if tf.exists():
        leak_states = sorted(raw_sh[raw_sh.level == "state"].state.unique())
        for col, *_ in groups:
            if col not in td:
                continue
            P = td.pivot_table(index="week", columns="state", values=col).sort_index()
            to_pct = lambda x: (100 * (np.exp(x - np.nanmean(x)) - 1)).round(1)
            ind = P.mean(axis=1).to_numpy()
            sts = {st_: [None if np.isnan(v) else float(v) for v in to_pct(P[st_].to_numpy())]
                   for st_ in leak_states if st_ in P.columns}
            if sts:
                series[col] = {"w": [d.strftime("%Y-%m-%d") for d in P.index],
                               "india": [None if np.isnan(v) else float(v) for v in to_pct(ind)], "st": sts}
        LEAKS_BY_STATE = {st_: [{"x": r.wk0.strftime("%Y-%m-%d"), "l": nice(r.exam_id)}
                                for r in raw_sh[raw_sh.state == st_].itertuples()] for st_ in leak_states}
    else:
        LEAKS_BY_STATE = {}
    cards = []
    for col, key, title, sub in groups:
        if col not in m_.index:
            continue
        terms = TRENDS_GROUPS.get(key, [])
        cards.append({"col": col, "title": title, "sub": sub, "terms": terms,
                      "real": round(float(m_.loc[col, "real_percent_at_mean_exposure"]), 1),
                      "mde": round(float(m_.loc[col, "mde_percent_at_mean_exposure"]), 1),
                      "curve": curves.get(col), "series": series.get(col)})
    if cards:
        body = """
<div class="gq">
  <div class="gq-bar"><svg viewBox="0 0 24 24" width="20" height="20"><circle cx="10" cy="10" r="6.5" fill="none" stroke="#5f6368" stroke-width="2"/><line x1="15" y1="15" x2="21" y2="21" stroke="#5f6368" stroke-width="2.2" stroke-linecap="round"/></svg>
    <span id="gqText"></span><span class="gq-caret">|</span></div>
  <div id="gqChips" class="gq-chips"></div>
  <div class="gq-card">
    <div class="gq-row"><div><div id="gqTitle" class="gq-title"></div><div id="gqSub" class="gq-sub"></div></div>
      <div class="gq-num"><div id="gqBig"></div><div class="gq-small">change in the exposed state after a leak</div></div></div>
    <div class="gq-meter"><div class="gq-track"><div id="gqMde" class="gq-mde"></div><div id="gqVal" class="gq-val"></div></div>
      <div class="gq-legend"><span><i class="sw val"></i>measured change</span><span><i class="sw mde"></i>smallest change this test could detect</span></div></div>
    <div class="gq-sub" style="margin-top:12px">Interest over time: exposed state minus the rest of India, weeks from the exam. Weekly values are noisy; the grey band is the range before the exam.</div>
    <div id="gqLine" style="height:220px"></div>
    <div class="gq-sub" style="margin-top:14px">Interest over time, 2023 to 2025, in one state against the India average (each relative to its own average). Red lines: that state's leaks.</div>
    <div id="gqStates" class="gq-chips" style="margin:8px 0 0"></div>
    <div id="gqTS" style="height:240px"></div>
  </div>
</div>"""
        script = f"""
<script>
(function(){{
const C={J(cards)}; const chips=document.getElementById('gqChips'); const txt=document.getElementById('gqText');
const GC=['#4285F4','#EA4335','#FBBC05','#34A853'];
let timer=null;
function type(s){{clearInterval(timer);txt.textContent='';let i=0;timer=setInterval(()=>{{txt.textContent=s.slice(0,++i);if(i>=s.length)clearInterval(timer);}},35);}}
function show(i){{
  const c=C[i]; [...chips.children].forEach((b,j)=>b.classList.toggle('on',j===i));
  type(c.terms.slice(0,4).join(', ')||c.title);
  document.getElementById('gqTitle').textContent=c.title;
  document.getElementById('gqSub').textContent=c.sub||'did these rise where the exam leaked?';
  const big=document.getElementById('gqBig'); const sig=Math.abs(c.real)>c.mde;
  big.textContent=(c.real>0?'+':'')+c.real.toFixed(1)+'%'; big.style.color=sig?'#188038':'#5f6368';
  const lim=Math.max(6,Math.abs(c.real)*1.2,c.mde*1.2), pct=v=>50+50*v/lim;
  const mde=document.getElementById('gqMde'); mde.style.left=pct(-c.mde)+'%'; mde.style.width=(pct(c.mde)-pct(-c.mde))+'%';
  const val=document.getElementById('gqVal'); val.style.left=Math.min(pct(0),pct(c.real))+'%'; val.style.width=Math.abs(pct(c.real)-pct(0))+'%';
  val.style.background=sig?'#34A853':'#9aa0a6';
  const cv=c.curve;
  if(cv){{Plotly.react('gqLine',[{{x:cv.k,y:cv.y,type:'scatter',mode:'lines',line:{{color:'#4285F4',width:2.5,shape:'spline'}},
      hovertemplate:'week %{{x}}: %{{y:+.1f}}%<extra></extra>',fill:'tozeroy',fillcolor:'rgba(66,133,244,0.08)'}}],
    {{font:{J(FONT)},margin:{{l:44,r:10,t:6,b:30}},paper_bgcolor:'white',plot_bgcolor:'white',
      xaxis:{{zeroline:false,gridcolor:'#f1f3f4',dtick:2}},yaxis:{{ticksuffix:'%',zeroline:true,zerolinecolor:'#9aa0a6',gridcolor:'#f1f3f4'}},
      shapes:[{{type:'rect',xref:'paper',yref:'y',x0:0,x1:1,y0:Math.min(...cv.y.slice(0,8).filter(v=>v!==null)),
                y1:Math.max(...cv.y.slice(0,8).filter(v=>v!==null)),fillcolor:'rgba(154,160,166,0.13)',line:{{width:0}},layer:'below'}},
              {{type:'line',xref:'x',yref:'paper',x0:-0.5,x1:-0.5,y0:0,y1:1,line:{{color:'#9aa0a6',dash:'dot'}}}}],
      annotations:[{{xref:'paper',yref:'y',x:0.01,y:Math.max(...cv.y.slice(0,8).filter(v=>v!==null)),yanchor:'bottom',xanchor:'left',
                showarrow:false,text:'range before the exam',font:{{size:11,color:'#5f6368'}}}}]}},
    {{responsive:true,displaylogo:false,displayModeBar:false}});}}
  else document.getElementById('gqLine').innerHTML='<p class="gq-sub">Weekly series not available for this group.</p>';
  CUR=i; drawTS();
}}
const LB={J(LEAKS_BY_STATE)}; let CUR=0, ST=null;
const stBox=document.getElementById('gqStates');
function drawTS(){{
  const sr=C[CUR].series; const el=document.getElementById('gqTS');
  if(!sr){{el.innerHTML='<p class="gq-sub">Weekly series not available for this group.</p>';stBox.innerHTML='';return;}}
  const names=Object.keys(sr.st); if(!ST||!names.includes(ST)) ST=names.includes('Uttar Pradesh')?'Uttar Pradesh':names[0];
  stBox.innerHTML='';
  names.forEach(n=>{{const b=document.createElement('button');b.className='gq-chip'+(n===ST?' on':'');b.style.setProperty('--c','#4285F4');
    b.textContent=n;b.onclick=()=>{{ST=n;drawTS();}};stBox.appendChild(b);}});
  const lk=LB[ST]||[];
  Plotly.react('gqTS',[
    {{x:sr.w,y:sr.india,type:'scatter',mode:'lines',name:'India average',line:{{color:'#9aa0a6',width:1.5}},hovertemplate:'India %{{x|%d %b %Y}}: %{{y:+.0f}}%<extra></extra>'}},
    {{x:sr.w,y:sr.st[ST],type:'scatter',mode:'lines',name:ST,line:{{color:'#4285F4',width:2}},hovertemplate:ST+' %{{x|%d %b %Y}}: %{{y:+.0f}}%<extra></extra>'}}],
    {{font:{J(FONT)},margin:{{l:44,r:10,t:22,b:30}},paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'x unified',
      legend:{{orientation:'h',y:1.12,x:1,xanchor:'right'}},xaxis:{{type:'date',gridcolor:'#f1f3f4'}},
      yaxis:{{ticksuffix:'%',gridcolor:'#f1f3f4',zeroline:true,zerolinecolor:'#dadce0'}},
      shapes:lk.map(l=>({{type:'line',x0:l.x,x1:l.x,yref:'paper',y0:0,y1:1,line:{{color:'#EA4335',width:1.4,dash:'dot'}}}})),
      annotations:lk.map((l,j)=>({{x:l.x,yref:'paper',y:1.0-0.08*j,text:l.l.split(',')[0],showarrow:false,xanchor:'left',font:{{size:10,color:'#EA4335'}}}}))}},
    {{responsive:true,displaylogo:false,displayModeBar:false}});
}}
C.forEach((c,i)=>{{const b=document.createElement('button');b.className='gq-chip';b.style.setProperty('--c',GC[i%4]);
  b.innerHTML='<i></i>'+c.title;b.onclick=()=>show(i);chips.appendChild(b);}});
show(0); setTimeout(()=>show(0),10);
}})();
</script>"""
        emit("search", "What people search for after a leak",
             "Pick a group of searches. Grey band: too small to tell from noise.",
             body + script,
             "Google Trends by state, Jan 2023-Sep 2025, against a fixed anchor term. Not affiliated with Google.")

# ---------------- bubbles: 400 comment bubbles, coloured by who gets blamed ----------------
bf = OUT / "online_blame_targets.csv"
if bf.exists():
    b_ = pd.read_csv(bf)
    total = int(b_.n.sum())
    grp = {"none": "No one in particular", "exam_body": "The exam body", "government": "The government",
           "women": "Women or women's quotas", "gender_quota": "Women or women's quotas",
           "caste_quota": "Caste reservation", "merit_system": "The merit system"}
    b_["g"] = b_.blame_target.map(grp).fillna("Something else")
    cats = b_.groupby("g", as_index=False).n.sum().sort_values("n", ascending=False)
    order_g = ["The exam body", "The government", "Women or women's quotas", "Caste reservation", "The merit system",
               "Something else", "No one in particular"]
    cats["o"] = cats.g.map({g_: i for i, g_ in enumerate(order_g)}).fillna(99)
    cats = cats.sort_values("o")
    COL = {"The exam body": "#c8553d", "The government": "#d99a1e", "Caste reservation": "#1a7fb0",
           "Women or women's quotas": "#8a3fd1", "The merit system": "#8d8a83", "Something else": "#b9b4aa",
           "No one in particular": "#e4e1da"}
    N = 400
    exact = cats.n / total * N
    alloc = np.floor(exact).astype(int)
    for i in (exact - alloc).sort_values(ascending=False).index[: N - alloc.sum()]:
        alloc[i] += 1
    data = [{"g": r_.g, "n": int(r_.n), "share": round(100 * r_.n / total, 2), "cells": int(a_), "c": COL.get(r_.g, "#b9b4aa")}
            for r_, a_ in zip(cats.itertuples(), alloc)]
    per = round(total / N)
    body = (f'<div class="cb-head"><b>{total:,} Comments</b><span>&#8801; Sort by: who gets blamed</span></div>'
            '<div class="cb-legend" id="cbLeg"></div><div class="cb-grid" id="cbGrid"></div>'
            '<div class="cb-note" id="cbNote"></div>')
    script = f"""
<script>
(function(){{
const D={J(data)}, PER={per}; const grid=document.getElementById('cbGrid'), leg=document.getElementById('cbLeg'), note=document.getElementById('cbNote');
const fmt=v=>v<1?v.toFixed(2):v.toFixed(1);
const tint=h=>{{const n=parseInt(h.slice(1),16),r=n>>16,g=(n>>8)&255,b=n&255,m=c=>Math.round(c+(255-c)*0.82);return 'rgb('+m(r)+','+m(g)+','+m(b)+')';}};
const THUMB='<svg viewBox="0 0 24 24"><path d="M2 21h4V9H2v12zM23 10c0-1.1-.9-2-2-2h-6.3l1-4.6v-.3c0-.4-.2-.8-.4-1.1L14.2 1 7.6 7.6C7.2 8 7 8.5 7 9v10c0 1.1.9 2 2 2h9c.8 0 1.5-.5 1.8-1.2l3-7.1c.1-.2.2-.5.2-.7v-2z"/></svg>';
D.forEach(d=>{{for(let i=0;i<d.cells;i++){{const b=document.createElement('div');b.className='cb';b.dataset.g=d.g;
  b.style.setProperty('--c',d.c);b.style.setProperty('--t',tint(d.c));b.title=d.g+': '+fmt(d.share)+'% of comments ('+d.n.toLocaleString()+')';
  b.innerHTML='<span class="av"></span><span class="tx"><b></b><i></i><span class="ft">'+THUMB+'</span></span>';grid.appendChild(b);}}}});
function pick(g){{
  [...leg.children].forEach(c=>c.classList.toggle('on',c.dataset.g===g));
  [...grid.children].forEach(b=>b.classList.toggle('dim',!!g&&b.dataset.g!==g));
  const d=D.find(x=>x.g===g);
  note.innerHTML=d?'<b>'+fmt(d.share)+'%</b> of comments blame <b>'+d.g.toLowerCase()+'</b>: '+d.n.toLocaleString()+' comments, '+d.cells+' of 400 comment cards':
    'Each card stands for about <b>'+PER+'</b> real comments. Click a label to highlight it.';
}}
D.forEach(d=>{{const c=document.createElement('button');c.className='cb-chip';c.dataset.g=d.g;
  c.innerHTML='<span class="cb-sw" style="background:'+d.c+'"></span>'+d.g+' <b>'+fmt(d.share)+'%</b>';
  c.onclick=()=>pick(c.classList.contains('on')?null:d.g);leg.appendChild(c);}});
pick(null);
}})();
</script>"""
    emit("comments", "Who do YouTube commenters blame?",
         f"{total:,} comments; each card stands for about {per}. Avatar colour: who the comment blames.",
         body + script,
         "Classified by a language model; the women category is an upper bound. No comment text or usernames shown. YouTube Data API.")

# ---------------- page ----------------
# page order: the women angle gets its own, longest part; figures flagged as needing explanation go last
PARTS = [("Part 1: Do leaks bring people onto the street?", ["rise", "spread", "atlas", "tiles", "eventstudy", "unusual", "national"]),
         ("Part 2: Who shows up, and who gets seen?", ["women", "reports", "surveys", "docket"]),
         ("Part 3: Does the anger turn against women online?", ["comments", "search"]),
         ("Appendix: figures that need a sentence of explanation", ["speccurve", "courts", "calendar"])]
if blocks:
    parts_html, done = [], set()
    for head, names_ in PARTS:
        got = [blocks[n_] for n_ in names_ if n_ in blocks]
        done.update(n_ for n_ in names_ if n_ in blocks)
        if got:
            parts_html.append(f'<div class="part">{head}</div>' + "".join(got))
    parts_html += [h_ for n_, h_ in blocks.items() if n_ not in done]
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Who Shows Up: interactive figures</title>{FONT_LINK}<script src="{CDN}"></script><style>{CSS}</style></head>'
            f'<body><div class="wrap">{"".join(parts_html)}</div></body></html>')
    (D / "story.html").write_text(html, encoding="utf-8")
    print(f"[26] story.html with {len(blocks)} figures -> {D}")
