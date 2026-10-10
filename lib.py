"""Small, dependency-light helpers: weeks, fixed-effect demeaning,
OLS with cluster-robust variance, Wald tests, and plot style."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def week_start(s: pd.Series) -> pd.Series:
    """Monday of the ISO week that contains each date."""
    d = pd.to_datetime(s)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def _codes(x) -> np.ndarray:
    return pd.factorize(pd.Series(x), sort=False)[0]


def demean(M: np.ndarray, fes: list, tol: float = 1e-10, maxit: int = 5_000) -> np.ndarray:
    """Remove several sets of fixed effects by alternating projections
    (method of alternating projections; converges to the within transform)."""
    M = np.asarray(M, dtype=float).copy()
    if M.ndim == 1:
        M = M[:, None]
    groups = [_codes(g) for g in fes]
    counts = [np.bincount(g) for g in groups]
    for _ in range(maxit):
        delta = 0.0
        for g, c in zip(groups, counts):
            sums = np.zeros((c.size, M.shape[1]))
            np.add.at(sums, g, M)
            means = sums / c[:, None]
            M -= means[g]
            delta = max(delta, np.abs(means).max())
        if delta < tol:
            return M
    raise RuntimeError("demean did not converge; check for disconnected FE sets")


def ols_cluster(y, X, cluster, names) -> dict:
    """OLS on already-demeaned data with CRV1 variance clustered on `cluster`.
    Small-sample factor G/(G-1) * (N-1)/(N-k), as in Stata/pyfixest CRV1."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g = _codes(cluster)
    G = g.max() + 1
    scores = np.zeros((G, k))
    np.add.at(scores, g, X * u[:, None])
    c = G / (G - 1) * (n - 1) / (n - k) if G > 1 and n > k else np.nan
    V = c * XtX_inv @ (scores.T @ scores) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tab = pd.DataFrame({"term": names, "coef": b, "se": se})
    tab["t"] = tab.coef / tab.se
    # t with G-1 df: the usual choice when clusters, not rows, drive inference
    tab["p"] = 2 * stats.t.sf(np.abs(tab.t), df=G - 1)
    crit = stats.t.ppf(0.975, df=G - 1)
    tab["ci_lo"] = tab.coef - crit * tab.se
    tab["ci_hi"] = tab.coef + crit * tab.se
    return {"table": tab, "V": pd.DataFrame(V, index=names, columns=names),
            "n": n, "G": G, "resid": u}


def wald(fit: dict, R: np.ndarray, r: np.ndarray | None = None) -> dict:
    """Wald test of R b = r. F form with (q, G-1) df."""
    b = fit["table"].coef.to_numpy()
    V = fit["V"].to_numpy()
    R = np.atleast_2d(R)
    r = np.zeros(R.shape[0]) if r is None else r
    d = R @ b - r
    W = float(d @ np.linalg.pinv(R @ V @ R.T) @ d)
    q = R.shape[0]
    F = W / q
    return {"W": W, "q": q, "F": F, "p": float(stats.f.sf(F, q, fit["G"] - 1))}


def blog_style():
    """Match the blog font (IBM Plex Sans) when it is installed."""
    import matplotlib as mpl
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    mpl.rcParams.update({
        "font.family": "IBM Plex Sans" if "IBM Plex Sans" in have else "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
        "savefig.bbox": "tight",
    })
    if "IBM Plex Sans" not in have:
        print("[style] IBM Plex Sans not found; using default sans-serif")


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# ---------------------------------------------------------------------
# Event-study helpers shared by 05 (district) and 09 (state) scripts
# ---------------------------------------------------------------------
def event_study(df, y_name, fe_cols, cluster_col, ref_k=-1, controls=("other_post_exp",), sexes=("M", "F"),
                mode="pre_mean"):
    """Event study with one exposure series per entry in `sexes` (A_M, A_F) or ("tot",) for A_tot.
    Returns (fit, table, tests_text)."""
    ks = sorted(k for k in df.k.unique() if k != ref_k)
    names, cols = [], []
    for sex in sexes:
        for k in ks:
            names.append(f"{sex}_k{k}")
            cols.append(df[f"A_{sex}"].to_numpy() * (df.k.to_numpy() == k))
    for c in controls:
        if c in df:
            names.append(c)
            cols.append(df[c].to_numpy())
    Z = demean(np.column_stack([df[y_name].to_numpy()] + cols), [df[c] for c in fe_cols])
    fit = ols_cluster(Z[:, 0], Z[:, 1:], df[cluster_col], names)
    tab = fit["table"]
    tab = tab[tab.term.str.match(r"^(M|F|tot)_k-?\d+$")].copy()
    tab["sex"] = tab.term.str.split("_k").str[0]
    tab["k"] = tab.term.str.split("_k").str[1].astype(int)

    idx = {t: i for i, t in enumerate(names)}
    pre, post = [k for k in ks if k < 0], [k for k in ks if k >= 0]
    pre_all = sorted(pre + [ref_k])                     # pre weeks incl. the omitted one (b = 0)
    V = fit["V"].to_numpy()
    b = fit["table"].coef.to_numpy()
    G = fit["G"]
    crit = stats.t.ppf(0.975, df=G - 1)

    def contrast(weights):                              # weights: {name: w}; returns (est, se, p)
        a = np.zeros(len(names))
        for n_, w_ in weights.items():
            if n_ in idx:
                a[idx[n_]] += w_
        est = float(a @ b)
        se = float(np.sqrt(max(a @ V @ a, 0)))
        p = float(2 * stats.t.sf(abs(est / se), df=G - 1)) if se > 0 else np.nan
        return est, se, p

    lines = [f"outcome={y_name} N={fit['n']:,} G={G} (clusters: {cluster_col})"]
    rows = []
    for sex in sexes:
        mean_pre = {f"{sex}_k{j}": -1 / len(pre_all) for j in pre_all}
        if mode == "pre_mean":
            for k in ks + [ref_k]:
                w = dict(mean_pre)
                w[f"{sex}_k{k}"] = w.get(f"{sex}_k{k}", 0) + 1
                est, se, p = contrast(w)
                rows.append({"term": f"{sex}_k{k}", "sex": sex, "k": k, "coef": est, "se": se, "p": p,
                             "ci_lo": est - crit * se, "ci_hi": est + crit * se})
        # pre-trend: linear slope across pre weeks
        kk = np.array(pre_all, float)
        sl = (kk - kk.mean()) / ((kk - kk.mean()) ** 2).sum()
        est, se, p = contrast({f"{sex}_k{j}": sl[i] for i, j in enumerate(pre_all)})
        lines.append(f"pre-trend slope {sex}: {est:.6f} per week (se {se:.6f}), p = {p:.3f}")
        R = np.zeros((len(pre), len(names)))
        for i, k in enumerate(pre):
            R[i, idx[f"{sex}_k{k}"]] = 1
        w_ = wald(fit, R)
        lines.append(f"pre-weeks jointly = week {ref_k} {sex}: F({w_['q']},{G-1}) = {w_['F']:.3f}, p = {w_['p']:.3f}"
                     " (sensitive to one unusual week)")
        did = dict(mean_pre)
        for k in post:
            did[f"{sex}_k{k}"] = did.get(f"{sex}_k{k}", 0) + 1 / len(post)
        est, se, p = contrast(did)
        lines.append(f"MAIN {sex}: mean(post) - mean(pre) = {est:.6f} (se {se:.6f}), p = {p:.3f}")
    if tuple(sexes) == ("M", "F"):
        w = {}
        for k in post:
            w[f"M_k{k}"] = w.get(f"M_k{k}", 0) + 1 / len(post)
            w[f"F_k{k}"] = w.get(f"F_k{k}", 0) - 1 / len(post)
        for j in pre_all:
            w[f"M_k{j}"] = w.get(f"M_k{j}", 0) - 1 / len(pre_all)
            w[f"F_k{j}"] = w.get(f"F_k{j}", 0) + 1 / len(pre_all)
        est, se, p = contrast(w)
        lines.append(f"M - F difference in (post - pre): {est:.6f} (se {se:.6f}), p = {p:.3f}")
    if mode == "pre_mean":
        tab = pd.DataFrame(rows).sort_values(["sex", "k"])
        lines.append("Coefficients are re-centred on the pre-period mean (weeks "
                     f"{min(pre_all)}..{max(pre_all)}).")
    if G < 40:
        lines.append(f"CAUTION: only {G} clusters; CRV1 p-values can over-reject. "
                     "Use the randomisation p-value (05b) for inference.")
    return fit, tab, "\n".join(lines)


def plot_event_study(tab, path, ylabel, ref_k=-1, xlabel="Weeks relative to week 0", title=None):
    import matplotlib.pyplot as plt
    blog_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    series = [("M", -0.12, r"$\hat\beta^{M}_k$ (male applicants)"),
              ("F", 0.12, r"$\hat\beta^{F}_k$ (female applicants)"),
              ("tot", 0.0, r"$\hat\beta_k$ (all applicants)")]
    for sex, off, lab in series:
        if sex not in set(tab.sex):
            continue
        t = tab[tab.sex == sex][["k", "coef", "ci_lo", "ci_hi"]]
        if ref_k not in set(t.k):
            t = pd.concat([t, pd.DataFrame({"k": [ref_k], "coef": [0.0], "ci_lo": [0.0], "ci_hi": [0.0]})])
        t = t.sort_values("k")
        ax.errorbar(t.k + off, t.coef, yerr=[t.coef - t.ci_lo, t.ci_hi - t.coef],
                    fmt="o", ms=4, capsize=2, label=lab)
    ax.axhline(0, lw=0.8, color="k")
    ax.axvline(ref_k + 0.5, lw=0.8, ls="--", color="grey")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.set_title(title or "95% CIs (t, G-1 df), errors clustered", fontsize=9, loc="left")
    fig.savefig(path)
    plt.close(fig)


def district_traits(RAW, standardise=True):
    """District characteristics from Census 2011 (districts.csv, district_covariates.csv, PCA files).
    Returns (DataFrame with district_id, state + traits, list of trait columns)."""
    dist = pd.read_csv(RAW / "districts.csv", dtype={"district_id": str})
    Z = dist[["district_id", "state", "pop_15_29"]].copy()
    cov_f = RAW / "district_covariates.csv"
    if cov_f.exists():
        Z = Z.merge(pd.read_csv(cov_f, dtype={"district_id": str}), on="district_id", how="left")
    d = RAW / "census_pca"
    files = [f for f in d.glob("*") if f.suffix.lower() in (".csv", ".xls", ".xlsx")] if d.exists() else []
    if files:
        p = pd.concat([pd.read_csv(f, dtype=str) if f.suffix.lower() == ".csv" else pd.read_excel(f, dtype=str)
                       for f in files])
        p.columns = [c.strip() for c in p.columns]
        p = p[p.Level.str.upper() == "DISTRICT"].copy()
        p["district_id"] = p.State.str.strip().str.zfill(2) + p.District.str.strip().str.zfill(3)
        for c in ["TOT_P", "P_LIT", "P_SC", "P_ST", "NON_WORK_P"]:
            p[c] = pd.to_numeric(p[c], errors="coerce") if c in p else np.nan
        tot = p[p.TRU.str.title() == "Total"].drop_duplicates("district_id").set_index("district_id")
        urb = p[p.TRU.str.title() == "Urban"].drop_duplicates("district_id").set_index("district_id")
        pz = pd.DataFrame({"literacy": tot.P_LIT / tot.TOT_P, "sc_st_share": (tot.P_SC + tot.P_ST) / tot.TOT_P,
                           "nonworker_share": tot.NON_WORK_P / tot.TOT_P,
                           "urban_share": urb.TOT_P.reindex(tot.index).fillna(0) / tot.TOT_P,
                           "tot_pop": tot.TOT_P})
        Z = Z.merge(pz.reset_index(), on="district_id", how="left")
        Z["youth_share"] = Z.pop_15_29 / Z.tot_pop
    cols = [c for c in ["literacy", "urban_share", "sc_st_share", "nonworker_share", "youth_share",
                        "female_work_rate", "cohort_sex_ratio"] if c in Z and Z[c].notna().mean() > 0.9]
    if standardise:
        for c in cols:
            Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
    return Z, cols


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, c - h, c + h
