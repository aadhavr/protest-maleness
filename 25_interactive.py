"""Interactive figures for the blog post (Plotly.js, one HTML file per figure + one combined page).

Figures (each made only if its inputs exist):
  1 timeline   Weekly exam protests in India, 2023-2026, non-party vs party-led, with leak dates
  2 shocks     Shock explorer: exposed state vs rest of India around each exam (dropdown)
  3 event      Event-study coefficients with 95% CIs (dropdown: all / non-party / party-led)
  4 map        District map of exam protests per 10 million youth (dropdown: measure), hubs outlined
  5 reporting  Exam vs other protests: women mentioned, sex-neutral wording, police force
  6 trends     Google Trends: change after a leak vs smallest detectable change
  7 surveys    WVS and Pew shares by sex and age
Inputs: interim/acled_district_week.csv, raw/districts.csv, interim/exam_shocks.csv,
        output/es_main_total_<y>.csv (run 05 with --total), output/descriptive_<y>.csv (run 05c),
        output/reporting_checks.csv, output/trends_mde.csv, output/survey_shares.csv,
        raw/shapes/census2011_districts/2011_Dist.shp (map; needs geopandas), raw/hubs.csv
Output: output/interactive/*.html  (self-contained except plotly.js from a CDN)
Embed a figure in the blog with an <iframe> to its HTML file.
"""
import json
import numpy as np
import pandas as pd
from config import RAW, INT, OUT

D = OUT / "interactive"
D.mkdir(parents=True, exist_ok=True)
CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
FONT = {"family": "IBM Plex Sans, Helvetica, Arial, sans-serif", "size": 13}
C = {"np": "#1f6f8b", "p": "#c8553d", "ctl": "#2a9d8f", "grey": "#8d99ae", "dark": "#22223b"}
CITE = "Source: ACLED (acleddata.com); author's calculations."
figs = []


def layout(title, **kw):
    base = {"title": {"text": title, "x": 0, "xanchor": "left", "font": {"size": 17}}, "font": FONT,
            "margin": {"l": 60, "r": 20, "t": 70, "b": 60}, "plot_bgcolor": "white", "paper_bgcolor": "white",
            "hovermode": "x unified", "legend": {"orientation": "h", "y": -0.18}}
    base.update(kw)
    return base


def page(name, title, traces, lay, note="", config=None):
    cfg = {"responsive": True, "displaylogo": False, **(config or {})}
    div = f"fig_{name}"
    block = (f'<div class="fig"><div id="{div}" style="width:100%;height:520px"></div>'
             f'<p class="note">{note}</p></div>\n'
             f'<script>Plotly.newPlot("{div}", {json.dumps(traces)}, {json.dumps(lay)}, {json.dumps(cfg)});</script>')
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><script src="{CDN}"></script>'
            '<style>body{font-family:"IBM Plex Sans",Helvetica,Arial,sans-serif;margin:0;padding:12px;background:#fff;color:#22223b}'
            '.note{font-size:12px;color:#5c677d;margin:4px 0 0}.fig{max-width:980px;margin:0 auto 28px}</style></head>'
            f'<body>{block}</body></html>')
    (D / f"{name}.html").write_text(html, encoding="utf-8")
    figs.append(block)
    print(f"[25] {name}.html")


def dropdown(labels, n_per, x=0, y=1.13):
    """Buttons that show one group of n_per traces at a time."""
    total = len(labels) * n_per
    btns = []
    for i, lab in enumerate(labels):
        vis = [False] * total
        for j in range(n_per):
            vis[i * n_per + j] = True
        btns.append({"label": lab, "method": "update", "args": [{"visible": vis}]})
    return [{"buttons": btns, "direction": "down", "x": x, "xanchor": "left", "y": y, "yanchor": "top",
             "showactive": True}]


sh = pd.read_csv(INT / "exam_shocks.csv", parse_dates=["shock_week"]) if (INT / "exam_shocks.csv").exists() else None
raw_sh = pd.read_csv(RAW / "exam_shocks.csv", parse_dates=["exam_date", "shock_date"]) if (RAW / "exam_shocks.csv").exists() else None
pr = pd.read_csv(INT / "acled_district_week.csv", parse_dates=["week"], dtype={"district_id": str})
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})

# ---------- 1 timeline ----------
cols = [c for c in ["n_exam_strict_nonparty", "n_exam_strict_party"] if c in pr]
if cols:
    w = pr.groupby("week")[cols].sum().sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="7D"), fill_value=0)
    x = [d.strftime("%Y-%m-%d") for d in w.index]
    tr = [{"type": "scatter", "mode": "lines", "x": x, "y": w[c].tolist(),
           "name": "Non-party (aspirants, students, others)" if c.endswith("nonparty") else "Party-led",
           "line": {"color": C["np"] if c.endswith("nonparty") else C["p"], "width": 2},
           "stackgroup": "one"} for c in cols]
    shapes, ann = [], []
    if raw_sh is not None:
        for r in raw_sh[raw_sh.shock_type.isin(["cancelled", "voided"])].itertuples():
            d0 = r.exam_date if pd.notna(r.exam_date) else r.shock_date
            if pd.isna(d0):
                continue
            shapes.append({"type": "line", "x0": d0.strftime("%Y-%m-%d"), "x1": d0.strftime("%Y-%m-%d"),
                           "yref": "paper", "y0": 0, "y1": 1, "line": {"color": C["grey"], "dash": "dot", "width": 1}})
            ann.append({"x": d0.strftime("%Y-%m-%d"), "yref": "paper", "y": 1.0, "text": r.exam_id.replace("_", " "),
                        "showarrow": False, "textangle": -90, "font": {"size": 9, "color": C["grey"]},
                        "xanchor": "right", "yanchor": "top"})
    lay = layout("Exam protests in India, by week",
                 xaxis={"rangeslider": {"visible": True}, "type": "date"},
                 yaxis={"title": "Exam protest events per week", "gridcolor": "#eee"}, shapes=shapes, annotations=ann)
    page("timeline", "Exam protests by week", tr, lay,
         "Dotted lines: exam dates of the leaks in the analysis. Drag the slider to zoom. " + CITE)

# ---------- 2 shock explorer ----------
for y in ["n_exam_strict_nonparty", "n_exam_strict"]:
    f = OUT / f"descriptive_{y}.csv"
    if f.exists():
        d = pd.read_csv(f)
        labels, tr = [], []
        for i, (e, g) in enumerate(d.groupby("exam_id", sort=False)):
            g = g.sort_values("k")
            vis = i == 0
            st = g.state.iloc[0]
            tr += [{"type": "scatter", "mode": "lines+markers", "x": g.k.tolist(), "y": g.exposed.round(3).tolist(),
                    "name": st, "line": {"color": C["np"], "width": 3}, "visible": vis,
                    "hovertemplate": f"{st}<br>week %{{x}}: %{{y:.2f}} per 10M youth<extra></extra>"},
                   {"type": "scatter", "mode": "lines", "x": g.k.tolist(), "y": g.rest.round(3).tolist(),
                    "name": "Rest of India", "line": {"color": C["grey"], "width": 2}, "visible": vis,
                    "hovertemplate": "Rest of India<br>week %{x}: %{y:.2f}<extra></extra>"}]
            labels.append(e.replace("_", " "))
        lay = layout("A leak, state by state", updatemenus=dropdown(labels, 2),
                     xaxis={"title": "Weeks from exam date", "zeroline": True, "zerolinecolor": C["dark"]},
                     yaxis={"title": "Exam protests per 10 million aged 15-29", "gridcolor": "#eee"},
                     hovermode="closest")
        page("shocks", "Shock explorer", tr, lay,
             "Choose a leak. Week 0 is the exam. " + CITE)
        break

# ---------- 3 event study ----------
ys = [("n_exam_strict_nonparty", "Non-party exam protests"), ("n_exam_strict", "All exam protests"),
      ("n_exam_strict_party", "Party-led exam protests")]
tr, labels = [], []
for i, (y, lab) in enumerate([t for t in ys if (OUT / f"es_main_total_{t[0]}.csv").exists()]):
    t = pd.read_csv(OUT / f"es_main_total_{y}.csv").sort_values("k")
    vis = i == 0
    tr += [{"type": "scatter", "mode": "markers", "x": t.k.tolist(), "y": t.coef.tolist(), "name": lab,
            "marker": {"color": C["np"] if "nonparty" in y else C["p"] if "party" in y else C["dark"], "size": 8},
            "error_y": {"type": "data", "symmetric": False, "array": (t.ci_hi - t.coef).tolist(),
                        "arrayminus": (t.coef - t.ci_lo).tolist(), "thickness": 1.2},
            "visible": vis, "hovertemplate": "week %{x}<br>effect %{y:.5f}<extra></extra>"}]
    labels.append(lab)
if tr:
    lay = layout("Effect of a leak on exam protests, week by week", updatemenus=dropdown(labels, 1),
                 xaxis={"title": "Weeks from exam date"},
                 yaxis={"title": "Extra events per district-week per applicant per 1,000 youth",
                        "zeroline": True, "zerolinecolor": C["dark"], "gridcolor": "#eee"},
                 shapes=[{"type": "line", "x0": -0.5, "x1": -0.5, "yref": "paper", "y0": 0, "y1": 1,
                          "line": {"color": C["grey"], "dash": "dash"}}], hovermode="closest")
    page("event", "Event study", tr, lay,
         "95% intervals (state clusters, indicative). Coefficients re-centred on the pre-exam mean. " + CITE)

# ---------- 4 map ----------
shp = RAW / "shapes" / "census2011_districts" / "2011_Dist.shp"
try:
    import geopandas as gpd
    if not shp.exists():
        raise FileNotFoundError(shp)
    g = gpd.read_file(shp).to_crs(4326)
    ids = set(dist.district_id)
    best = None
    for sc in g.columns:
        for dc in g.columns:
            if sc == dc or sc == "geometry" or dc == "geometry":
                continue
            a = pd.to_numeric(g[sc], errors="coerce")
            b = pd.to_numeric(g[dc], errors="coerce")
            if a.isna().mean() > .2 or b.isna().mean() > .2:
                continue
            cand = a.fillna(-1).astype(int).astype(str).str.zfill(2) + b.fillna(-1).astype(int).astype(str).str.zfill(3)
            hit = cand.isin(ids).mean()
            if best is None or hit > best[0]:
                best = (hit, cand)
    g["district_id"] = best[1]
    g = g[g.district_id.isin(ids)][["district_id", "geometry"]]
    g["geometry"] = g.geometry.simplify(0.02, preserve_topology=True)
    gj = json.loads(g.to_json())
    for ft in gj["features"]:
        ft["id"] = ft["properties"]["district_id"]
    tot = pr.groupby("district_id")[[c for c in ["n_exam_strict_nonparty", "n_exam_strict", "n_events"] if c in pr]].sum()
    m = dist.set_index("district_id")[["district_name", "state", "pop_15_29"]].join(tot).fillna(0)
    hubs = set()
    if (RAW / "hubs.csv").exists():
        h = pd.read_csv(RAW / "hubs.csv")
        nm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
        key = {(nm(a), nm(b)): i for i, a, b in zip(m.index, m.district_name, m.state)}
        hubs = {key.get((nm(a), nm(b))) for a, b in zip(h.district_name, h.state)} - {None}
    measures = [(c, lab) for c, lab in [("n_exam_strict_nonparty", "Non-party exam protests"),
                                        ("n_exam_strict", "All exam protests"), ("n_events", "All protests")] if c in m]
    tr, labels = [], []
    for i, (c, lab) in enumerate(measures):
        rate = (m[c] / m.pop_15_29 * 1e7)
        z = np.log10(rate + 1).round(3)
        tr.append({"type": "choropleth", "geojson": gj, "locations": m.index.tolist(), "z": z.tolist(),
                   "customdata": np.column_stack([m.district_name, m.state, rate.round(1), m[c].astype(int)]).tolist(),
                   "hovertemplate": "%{customdata[0]}, %{customdata[1]}<br>%{customdata[3]} events"
                                    "<br>%{customdata[2]} per 10M youth<extra></extra>",
                   "colorscale": "Blues" if "exam" in c else "Greys", "marker": {"line": {"width": 0.2, "color": "#fff"}},
                   "colorbar": {"title": "log10(1 + per 10M youth)", "thickness": 12}, "visible": i == 0, "name": lab})
        labels.append(lab)
    if hubs:
        hub_ft = {"type": "FeatureCollection", "features": [ft for ft in gj["features"] if ft["id"] in hubs]}
        tr.append({"type": "choropleth", "geojson": hub_ft, "locations": sorted(hubs), "z": [1] * len(hubs),
                   "colorscale": [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], "showscale": False,
                   "marker": {"line": {"width": 2.2, "color": C["p"]}}, "hoverinfo": "skip", "name": "Coaching hubs"})
    lay = layout("Where exam protests happen, 2023-2026",
                 geo={"fitbounds": "locations", "visible": False, "projection": {"type": "mercator"}},
                 margin={"l": 0, "r": 0, "t": 70, "b": 0}, hovermode="closest")
    btns = []
    for i, lab in enumerate(labels):
        vis = [j == i for j in range(len(labels))] + ([True] if hubs else [])
        btns.append({"label": lab, "method": "update", "args": [{"visible": vis}]})
    lay["updatemenus"] = [{"buttons": btns, "direction": "down", "x": 0, "y": 1.1, "xanchor": "left"}]
    page("map", "District map", tr, lay,
         "Per 10 million residents aged 15-29 (Census 2011), log scale. Red outlines: coaching-hub districts. "
         "District boundaries: DataMeet (2011). " + CITE, config={"scrollZoom": True})
except Exception as ex:
    print(f"[25] map skipped ({str(ex)[:120]})")

# ---------- 5 reporting ----------
f = OUT / "reporting_checks.csv"
if f.exists():
    r = pd.read_csv(f)
    rows = [("a police intervention / excessive force", "Police intervention or force", "exam protests", "other protests"),
            ("b sex-neutral only | report names any person noun", "Only sex-neutral nouns (when people are named)", "exam events", "other events")]
    cats, ex_v, ot_v, ex_e, ot_e = [], [], [], [], []
    for chk, lab, ge, go in rows:
        a = r[(r.check == chk) & (r.group == ge)]
        b = r[(r.check == chk) & (r.group == go)]
        if len(a) and len(b):
            cats.append(lab)
            ex_v.append(100 * a.share.iloc[0]); ot_v.append(100 * b.share.iloc[0])
            ex_e.append([100 * (a.share.iloc[0] - a.ci_lo.iloc[0]), 100 * (a.ci_hi.iloc[0] - a.share.iloc[0])])
            ot_e.append([100 * (b.share.iloc[0] - b.ci_lo.iloc[0]), 100 * (b.ci_hi.iloc[0] - b.share.iloc[0])])
    rc = OUT / "repertoire_compare.csv"
    if rc.exists():
        q = pd.read_csv(rc)
        q = q[q.scope == "all states"]
        a = q[q.measure.str.contains(r"Street: exam protests \(women mentioned")]
        b = q[q.measure.str.contains(r"Street: all protests \(women mentioned")]
        if len(a) and len(b):
            cats.insert(0, "Report mentions women")
            ex_v.insert(0, 100 * a.female_share.iloc[0]); ot_v.insert(0, 100 * b.female_share.iloc[0])
            ex_e.insert(0, [100 * (a.female_share.iloc[0] - a.ci_lo.iloc[0]), 100 * (a.ci_hi.iloc[0] - a.female_share.iloc[0])])
            ot_e.insert(0, [100 * (b.female_share.iloc[0] - b.ci_lo.iloc[0]), 100 * (b.ci_hi.iloc[0] - b.female_share.iloc[0])])
    if cats:
        tr = [{"type": "bar", "x": cats, "y": np.round(ex_v, 1).tolist(), "name": "Exam protests", "marker": {"color": C["np"]},
               "error_y": {"type": "data", "symmetric": False, "array": [e[1] for e in ex_e], "arrayminus": [e[0] for e in ex_e]},
               "hovertemplate": "%{x}<br>Exam protests: %{y:.1f}%<extra></extra>"},
              {"type": "bar", "x": cats, "y": np.round(ot_v, 1).tolist(), "name": "Other protests", "marker": {"color": C["grey"]},
               "error_y": {"type": "data", "symmetric": False, "array": [e[1] for e in ot_e], "arrayminus": [e[0] for e in ot_e]},
               "hovertemplate": "%{x}<br>Other protests: %{y:.1f}%<extra></extra>"}]
        lay = layout("How exam protests are recorded", barmode="group",
                     yaxis={"title": "% of events", "gridcolor": "#eee"}, hovermode="closest")
        page("reporting", "Reporting", tr, lay, "95% intervals. Text measures use ACLED's event summaries. " + CITE)

# ---------- 6 trends ----------
f = OUT / "trends_mde.csv"
if f.exists():
    t = pd.read_csv(f)
    names = {"trends_control": "Exam-leak terms (control)", "trends_index": "Main index (fixed in advance)",
             "trends_manosphere": "Manosphere terms", "trends_gender_jobs": "Gendered-jobs terms",
             "trends_mra": "Men's-rights terms"}
    t = t[t.outcome.isin(names)].copy()
    t["label"] = t.outcome.map(names)
    t = t.set_index("outcome").loc[[k for k in names if k in set(t.outcome)]].reset_index()
    tr = [{"type": "bar", "orientation": "h", "y": t.label.tolist(), "x": t.real_percent_at_mean_exposure.round(2).tolist(),
           "name": "Change after a leak", "marker": {"color": [C["ctl"] if o == "trends_control" else C["np"] for o in t.outcome]},
           "hovertemplate": "%{y}<br>change: %{x:.2f}%<extra></extra>"},
          {"type": "scatter", "mode": "markers", "y": t.label.tolist(), "x": t.mde_percent_at_mean_exposure.round(2).tolist(),
           "name": "Smallest detectable change", "marker": {"symbol": "line-ns-open", "size": 22, "color": C["dark"]},
           "hovertemplate": "%{y}<br>detectable: %{x:.2f}%<extra></extra>"}]
    lay = layout("Searches after a leak: the control moves, resentment terms do not",
                 xaxis={"title": "% change in searches at average exposure", "zeroline": True, "gridcolor": "#eee"},
                 yaxis={"automargin": True}, hovermode="closest")
    page("trends", "Google Trends", tr, lay, "Google Trends, by state, 2023-2025; author's calculations.")

# ---------- 7 surveys ----------
f = OUT / "survey_shares.csv"
if f.exists():
    s = pd.read_csv(f)
    meas = s.measure.unique().tolist()
    groups = ["men", "women", "men 18-29", "women 18-29", "young men", "young women"]
    tr, labels = [], []
    for i, mname in enumerate(meas):
        q = s[s.measure == mname].set_index("group").reindex([g_ for g_ in groups if g_ in set(s[s.measure == mname].group)])
        tr.append({"type": "bar", "x": q.index.tolist(), "y": (100 * q.share).round(1).tolist(),
                   "error_y": {"type": "data", "symmetric": False, "array": (100 * (q.ci_hi - q.share)).round(1).tolist(),
                               "arrayminus": (100 * (q.share - q.ci_lo)).round(1).tolist()},
                   "marker": {"color": [C["np"] if "women" not in g_ else C["p"] for g_ in q.index]},
                   "visible": i == 0, "name": mname, "hovertemplate": "%{x}: %{y:.1f}%<extra></extra>"})
        labels.append(mname.replace("WVS: ", "WVS 2023: ").replace("Pew: ", "Pew 2019-20: "))
    lay = layout("What surveys say", updatemenus=dropdown(labels, 1, y=1.15),
                 yaxis={"title": "% of respondents", "range": [0, 100], "gridcolor": "#eee"}, hovermode="closest")
    page("surveys", "Surveys", tr, lay, "Weighted shares, 95% intervals. World Values Survey Wave 7 (India, 2023); "
         "Pew Research Center India survey (2019-20).")

# ---------- combined page ----------
if figs:
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Who Shows Up: figures</title><script src="{CDN}"></script>'
            '<style>body{font-family:"IBM Plex Sans",Helvetica,Arial,sans-serif;margin:0;padding:16px;color:#22223b}'
            '.note{font-size:12px;color:#5c677d;margin:4px 0 0}.fig{max-width:980px;margin:0 auto 36px}</style></head><body>'
            + "\n".join(figs) + "</body></html>")
    (D / "all_figures.html").write_text(html, encoding="utf-8")
    print(f"[25] all_figures.html ({len(figs)} figures) -> {D}")
