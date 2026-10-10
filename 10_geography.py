"""Layer 3: inside the exposed state, do coaching-hub districts protest more than
their share of the state's young people predicts?

Only stacks whose EXPOSED districts include a hub are used (state exams in a state
with a hub; national exams use all districts). Control districts in other states
are dropped, so every comparison is hub vs non-hub inside the same exposed state.
(a) Descriptive: hub share of the state's post-shock exam protests vs hub share of
    the state's youth (applicant counts are state totals spread by youth population,
    so the youth share is the only honest "expected" share).
(b) Regression on exposed districts only:
    y = d * (post x hub) + b * (post x A_tot) + c * other_post_exp + FE(de) + FE(stw) + e
    d > 0: hubs gain more protest after a shock than other districts of the same state.
Input: raw/hubs.csv (district_id, or district_name + state)
"""
import sys
import numpy as np
import pandas as pd
from config import RAW, INT, OUT, WINDOW_POST
from lib import demean, ols_cluster

y = sys.argv[1] if len(sys.argv) > 1 else "n_exam_strict"
dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
hubs = pd.read_csv(RAW / "hubs.csv", dtype={"district_id": str})
if "district_id" not in hubs:
    nm = lambda x: "".join(ch for ch in str(x).lower() if ch.isalnum())
    hubs = hubs.assign(_n=hubs.district_name.map(nm), _s=hubs.state.map(nm)).merge(
        dist.assign(_n=dist.district_name.map(nm), _s=dist.state.map(nm))[["_n", "_s", "district_id"]],
        on=["_n", "_s"], how="left")
    if hubs.district_id.isna().any():
        print(f"[10] hub names not found in districts.csv: {hubs[hubs.district_id.isna()].district_name.tolist()}")
    hubs = hubs.dropna(subset=["district_id"])
hub_ids = set(hubs.district_id)

df = pd.read_csv(INT / "stacked_main.csv", dtype={"district_id": str, "stack_id": str})
df = df[df.A_tot > 0].copy()                                  # exposed districts only
# national exams with exposure spread by youth population carry no applicant geography: leave out
flat = df.groupby("stack_id").apply(lambda d: (d.A_tot / d.A_tot.mean()).std() < 1e-6 and d.state.nunique() > 1)
if flat.any():
    print(f"[10] left out (national, no applicant geography): {list(flat[flat].index)}")
    df = df[~df.stack_id.isin(flat[flat].index)]
df["hub"] = df.district_id.isin(hub_ids).astype(int)
has_hub = df.groupby("stack_id").hub.max()
keep = has_hub[has_hub == 1].index
print(f"[10] stacks with a hub among exposed districts: {list(keep)}")
print(f"[10] left out (no hub in the exposed state): {list(has_hub[has_hub == 0].index)}")
df = df[df.stack_id.isin(keep)]
if df.empty:
    raise SystemExit("[10] no stack has a hub among its exposed districts")

youth = dist.set_index("district_id").pop_15_29
rows = []
for e, d in df.groupby("stack_id"):
    ds = d.drop_duplicates("district_id")
    yshare = youth.reindex(ds[ds.hub == 1].district_id).sum() / youth.reindex(ds.district_id).sum()
    pre, post = d[d.k < 0], d[d.k >= 0]
    sh = lambda x: x.loc[x.hub == 1, y].sum() / x[y].sum() if x[y].sum() else np.nan
    rows.append({"stack_id": e, "hub_share_youth": yshare,
                 "pre_protests": int(pre[y].sum()), "hub_share_pre": sh(pre),
                 "post_protests": int(post[y].sum()), "hub_share_post": sh(post)})
desc = pd.DataFrame(rows).set_index("stack_id")
desc["ratio_post"] = desc.hub_share_post / desc.hub_share_youth
desc.to_csv(OUT / f"geography_desc_{y}.csv")
print(desc.round(3).to_string())

X = np.column_stack([df.post * df.hub, df.post * df.A_tot, df.other_post_exp])
Z = demean(np.column_stack([df[y], X]), [df.fe_de, df.fe_stw])
t = ols_cluster(Z[:, 0], Z[:, 1:], df.district_id, ["post_x_hub", "post_x_Atot", "other"])["table"]
t.to_csv(OUT / f"geography_reg_{y}.csv", index=False)
print(t.round(5).to_string(index=False))
print(f"[10] exposed districts: {df.district_id.nunique()}, of which hubs: {df.loc[df.hub == 1, 'district_id'].nunique()}")
print("[10] ratio_post > 1: hubs have more than their youth share of post-shock exam protests.")
print("[10] Compare hub_share_post with hub_share_pre: concentration that exists before the shock is not caused by it.")
print("[10] Small counts (post_protests < 20) are anecdotes, not estimates.")
