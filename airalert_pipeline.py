#!/usr/bin/env python3
"""
Hourly forecast: will a NEW air-raid alert start in region R within the next hour?

Source : volunteer_data_en.csv (oblast-level, current to the day, has `naive` flag)
Run    : python airalert_pipeline.py --data data/volunteer_data_en.csv

LEAKAGE RULES (enforced by construction + checked by truncation_test):
  * Decision time t = an hour boundary (UTC). Target y = 1 if an alert STARTS in [t, t+1h).
  * Features may use only: alert starts < t, and alert ends <= t.
    For an alert still running at t we use only the fact that it is active, never its end time.
  * Calendar features are deterministic functions of t.
  * Neighbour regions are selected from data strictly BEFORE the first CV training cutoff.
  * Splits are chronological with an embargo gap; models/encoders are fit on train rows only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
EXCLUDE_REGIONS = {"Luhanska oblast"}          # permanent siren: target undefined
MODEL_START = pd.Timestamp("2025-01-01", tz="UTC")  # rows before this only feed lookbacks
TEST_DAYS = 42            # final hold-out
VAL_DAYS = 28             # length of each rolling-origin validation window
N_FOLDS = 5
GAP_HOURS = 24            # embargo between train end and validation/test start
MAX_ALERT_MIN = 24 * 60   # longer alerts are treated as missing end signals
IMPUTED_MIN = 30          # dataset's own rule for missing ends
TOP_K = 3                 # data-driven neighbours per region
CAP_H = 720.0             # cap / fill for "hours since" features
LOCAL_TZ = "Europe/Kyiv"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
NON_FEATURE = {"t", "region", "y", "y_count", "active", "hour_local", "eligible"}


def to_min(x) -> np.ndarray:
    """Timestamps -> float minutes since epoch (unit-safe across pandas versions)."""
    return np.atleast_1d(np.asarray((x - EPOCH) / pd.Timedelta(minutes=1), dtype="float64"))


# ------------------------------------------------------------------ STEP 1: load & clean
def load_clean(path: Path):
    df = pd.read_csv(path)
    for c in ("started_at", "finished_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    n0 = len(df)
    df = df[~df["region"].isin(EXCLUDE_REGIONS)].copy()
    df = df.drop_duplicates(["region", "started_at"])

    dur = (df["finished_at"] - df["started_at"]) / pd.Timedelta(minutes=1)
    if (dur < 0).any():
        raise ValueError("Negative durations found")
    bad = dur > MAX_ALERT_MIN
    df.loc[bad, "finished_at"] = df.loc[bad, "started_at"] + pd.Timedelta(minutes=IMPUTED_MIN)
    df["imputed_end"] = df["naive"].astype(bool) | bad

    df = df.sort_values(["region", "started_at"]).reset_index(drop=True)
    prev_end = df.groupby("region")["finished_at"].shift()
    n_overlap = int((df["started_at"] < prev_end).sum())
    if n_overlap:
        raise ValueError(f"{n_overlap} overlapping alerts within a region; merge them first")

    data_end = max(df["started_at"].max(), df.loc[~df["imputed_end"], "finished_at"].max())
    print(f"[load] rows {n0} -> {len(df)} | regions {df['region'].nunique()} | "
          f"over-24h fixed: {int(bad.sum())} | imputed ends: {df['imputed_end'].mean():.3f}")
    print(f"[load] last observed event: {data_end}")
    return df, data_end


# ------------------------------------------------------------------ STEP 2: split boundaries
def split_bounds(label_end: pd.Timestamp):
    """All boundaries are timestamps; a row's t is in a window iff lo <= t < hi."""
    test_start = label_end + pd.Timedelta(hours=1) - pd.Timedelta(days=TEST_DAYS)
    dev_end = test_start - pd.Timedelta(hours=GAP_HOURS)
    folds = []
    for k in range(N_FOLDS):
        val_hi = dev_end - pd.Timedelta(days=VAL_DAYS * (N_FOLDS - 1 - k))
        val_lo = val_hi - pd.Timedelta(days=VAL_DAYS)
        train_hi = val_lo - pd.Timedelta(hours=GAP_HOURS)
        folds.append((train_hi, val_lo, val_hi))
    return folds, dev_end, test_start


# ------------------------------------------------------------------ STEP 3: neighbours (train period only)
def start_matrix(groups, regions, g0m: float, n: int) -> np.ndarray:
    cols = []
    for r in regions:
        s = to_min(groups[r]["started_at"]) if r in groups else np.array([])
        h = np.floor((s - g0m) / 60).astype(int)
        h = h[(h >= 0) & (h < n)]
        cols.append(np.bincount(h, minlength=n).astype(float))
    return np.column_stack(cols)


def select_neighbors(alerts, regions, g0, sel_end, k=TOP_K):
    """Top-k regions whose starts in hour h-1 best predict region r's starts in hour h.
    Uses ONLY hours with MODEL_START <= t < sel_end."""
    grid = pd.date_range(g0, sel_end - pd.Timedelta(hours=1), freq="h")
    groups = dict(tuple(alerts.groupby("region")))
    g0m = to_min(g0)[0]
    S = start_matrix(groups, regions, g0m, len(grid)) > 0
    i0 = int(np.searchsorted(grid, MODEL_START))
    Y = S[i0:].astype(float)
    X = S[i0 - 1:-1].astype(float)           # lag-1 indicators
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    C = (Xc.T @ Yc) / (len(X) * np.outer(Xc.std(0), Yc.std(0)) + 1e-12)  # C[c, r]
    np.fill_diagonal(C, -np.inf)
    nb = {r: [regions[c] for c in np.argsort(-C[:, j])[:k]] for j, r in enumerate(regions)}
    return nb


# ------------------------------------------------------------------ STEP 4: backward-looking features
def region_arrays(s: np.ndarray, e: np.ndarray, g0m: float, n: int):
    grid = g0m + 60.0 * np.arange(n)
    nan = np.full(n, np.nan)
    if len(s) == 0:
        f = {"y_count": np.zeros(n), "h_since_start": nan, "h_since_end": nan}
        zero = pd.Series(np.zeros(n))
        return f, zero, np.zeros(n, dtype=bool), zero

    h = np.floor((s - g0m) / 60).astype(int)
    ok = (h >= 0) & (h < n)
    starts = np.bincount(h[ok], minlength=n).astype(float)

    mins = np.zeros(n)                       # alert-minutes inside each hour
    for si, ei in zip(s, e):
        a, hh = si, int((si - g0m) // 60)
        while a < ei and 0 <= hh < n:
            b = min(ei, g0m + (hh + 1) * 60.0)
            mins[hh] += b - a
            a, hh = b, hh + 1

    idx = np.searchsorted(s, grid, side="left") - 1      # last start strictly < t
    has = idx >= 0
    j = np.clip(idx, 0, None)
    last_s = np.where(has, s[j], np.nan)
    last_e = np.where(has, e[j], np.nan)
    active = has & (last_e > grid)                       # running at t (end only compared, never used)
    h_since_start = (grid - last_s) / 60.0
    h_since_end = np.where(has & ~active, (grid - last_e) / 60.0, np.where(has, 0.0, np.nan))

    f = {"y_count": starts, "h_since_start": h_since_start, "h_since_end": h_since_end}
    return f, pd.Series(starts), active, pd.Series(mins)


def build_features(alerts, regions, neighbors, g0, t_end) -> pd.DataFrame:
    """Long frame (region, t). Every feature at row t uses information strictly before t."""
    grid = pd.date_range(g0, t_end, freq="h")
    n, R = len(grid), len(regions)
    g0m = to_min(g0)[0]
    groups = dict(tuple(alerts.groupby("region")))

    per, S_cols, A_cols = {}, [], []
    for r in regions:
        g = groups.get(r)
        s = to_min(g["started_at"]) if g is not None else np.array([])
        e = to_min(g["finished_at"]) if g is not None else np.array([])
        f, st, active, ms = region_arrays(s, e, g0m, n)
        for k in (1, 2, 3, 24):
            f[f"lag{k}"] = st.shift(k).to_numpy()
        sh = st.shift(1)                              # starts in the PREVIOUS hour onward
        for w in (3, 6, 24, 168):
            f[f"cnt_{w}h"] = sh.rolling(w, min_periods=1).sum().to_numpy()
        msh = ms.shift(1)
        for w in (6, 24):
            f[f"alert_min_{w}h"] = msh.rolling(w, min_periods=1).sum().to_numpy()
        f["h_since_start"] = np.nan_to_num(f["h_since_start"], nan=CAP_H).clip(max=CAP_H)
        f["h_since_end"] = np.nan_to_num(f["h_since_end"], nan=CAP_H).clip(max=CAP_H)
        per[r] = (f, active)
        S_cols.append(st.to_numpy())
        A_cols.append(active.astype(float))

    S, A = np.column_stack(S_cols), np.column_stack(A_cols)
    S1 = np.vstack([np.zeros((1, R)), S[:-1]])                           # previous hour's starts
    S3 = pd.DataFrame(S1).rolling(3, min_periods=1).sum().to_numpy()    # previous 3 hours
    local = grid.tz_convert(LOCAL_TZ)
    hr, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    cal = {"hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
           "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7)}
    keep = np.asarray(grid >= MODEL_START)

    frames = []
    for ri, r in enumerate(regions):
        f, active = per[r]
        nbi = [regions.index(x) for x in neighbors[r]]
        feats = dict(f)
        feats.update(
            nbr_starts_1h=S1[:, nbi].sum(1), nbr_starts_3h=S3[:, nbi].sum(1),
            nbr_active=A[:, nbi].sum(1),
            nat_active_others=A.sum(1) - A[:, ri],
            nat_starts_1h_others=S1.sum(1) - S1[:, ri],
            nat_starts_3h_others=S3.sum(1) - S3[:, ri], **cal)
        d = pd.DataFrame({k: np.asarray(v)[keep] for k, v in feats.items()})
        d.insert(0, "t", grid[keep])
        d.insert(1, "region", r)
        d["active"] = active[keep]
        d["hour_local"] = hr[keep]
        d["y"] = (d.pop("y_count") > 0).astype(float)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["region"] = pd.Categorical(out["region"], categories=regions)
    fl = [c for c in out.columns if c not in NON_FEATURE and out[c].dtype == "float64"]
    out[fl] = out[fl].astype("float32")
    return out


# ------------------------------------------------------------------ STEP 5: leakage test
def truncation_test(alerts, regions, neighbors, g0, full, feat_cols, n_checks=5, seed=0):
    """Rebuild features using ONLY data known at T (ongoing alerts get an unknown end) and
    require them to equal the full-data features at T. Any future peek makes this fail."""
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(full.loc[full["y"].notna() & (full["t"] >= MODEL_START + pd.Timedelta(days=30)),
                                   "t"].unique())
    cols = feat_cols + ["active"]
    for i in rng.choice(len(ts), n_checks, replace=False):
        T = ts[i]
        tr = alerts[alerts["started_at"] < T].copy()
        ongoing = tr["finished_at"] > T
        tr.loc[ongoing, "finished_at"] = T + pd.Timedelta(days=36500)   # end unknown at T
        a = build_features(tr, regions, neighbors, g0, T)
        a = a[a["t"] == T].set_index("region").loc[regions, cols].astype(float)
        b = full[full["t"] == T].set_index("region").loc[regions, cols].astype(float)
        bad = [c for c in cols if not np.allclose(a[c], b[c], equal_nan=True, rtol=1e-5, atol=1e-5)]
        if bad:
            raise AssertionError(f"LEAKAGE at {T}: features differ when future data is removed: {bad}")
    print(f"[leak-test] PASSED: {n_checks} random timestamps, all features identical with future removed")


# ------------------------------------------------------------------ STEP 6: models & metrics
def metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y)
    both = 0 < y.sum() < len(y)
    pr = average_precision_score(y, p) if both else np.nan
    return {"n": len(y), "pos_rate": y.mean(), "pr_auc": pr,
            "pr_lift": pr / y.mean() if both else np.nan,
            "roc_auc": roc_auc_score(y, p) if both else np.nan,
            "brier": brier_score_loss(y, np.clip(p, 0, 1))}


def rate_model(train, keys):
    """Baseline: empirical start rate by `keys`, estimated on TRAIN rows only."""
    tab = train.groupby(keys, observed=True)["y"].mean()
    glob = train["y"].mean()

    def predict(d):
        k = d[keys].copy()
        k["region"] = k["region"].astype(str)
        idx = pd.MultiIndex.from_frame(k)
        t2 = tab.copy()
        t2.index = pd.MultiIndex.from_frame(t2.index.to_frame(index=False).assign(
            region=lambda x: x["region"].astype(str)))
        return t2.reindex(idx).fillna(glob).to_numpy()
    return predict


def fit_predict_all(train, test, x_cols):
    from xgboost import XGBClassifier
    tr, te = train.copy(), test.copy()
    for d in (tr, te):
        d["lag1_pos"] = (d["lag1"] > 0).astype(int)
    preds = {
        "base_region": rate_model(tr, ["region"])(te),
        "base_region_hour": rate_model(tr, ["region", "hour_local"])(te),
        "persistence": rate_model(tr, ["region", "lag1_pos"])(te),
    }
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                      enable_categorical=True, eval_metric="logloss", n_jobs=-1, random_state=0)
    m.fit(tr[x_cols], tr["y"].astype(int))
    preds["xgboost"] = m.predict_proba(te[x_cols])[:, 1]
    return preds, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/volunteer_data_en.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    alerts, data_end = load_clean(Path(args.data))
    regions = sorted(alerts["region"].unique())
    g0 = alerts["started_at"].min().floor("h")
    origin = data_end.floor("h")                       # first hour NOT fully observed
    label_end = origin - pd.Timedelta(hours=1)         # last hour with a complete label
    folds, dev_end, test_start = split_bounds(label_end)
    print(f"[time] live forecast origin {origin} | last labelled t {label_end}")
    print(f"[time] dev < {dev_end} | test {test_start} .. {label_end}")

    nb = select_neighbors(alerts, regions, g0, folds[0][0])   # data before first train cutoff only
    print("[neighbors] (chosen from data before", folds[0][0], ")")
    for r in regions:
        print(f"   {r:26s} <- {', '.join(nb[r])}")

    df = build_features(alerts, regions, nb, g0, origin)
    df.loc[df["t"] > label_end, "y"] = np.nan          # origin hour has no complete label
    df["eligible"] = (~df["active"]) & df["y"].notna()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE and df[c].dtype != "category"]
    x_cols = feat_cols + ["region"]
    print(f"[data] rows {len(df)} | features {len(feat_cols)} (+region) | "
          f"in-alert rows excluded: {(df['active'] & df['y'].notna()).mean():.1%}")
    act = df[df["active"] & df["y"].notna()]
    print(f"[data] excluded in-alert rows where a new alert still started within the hour: "
          f"{act['y'].mean():.2%} (these cases cannot be forecast by this model)")
    el = df[df["eligible"]]
    print("[data] monthly start rate among eligible rows (check for drift):")
    print(el.groupby(el["t"].dt.strftime("%Y-%m"))["y"].mean().tail(12).round(4).to_string())

    truncation_test(alerts, regions, nb, g0, df, feat_cols)

    # ---- rolling-origin CV (expanding window)
    rows = []
    for k, (train_hi, val_lo, val_hi) in enumerate(folds):
        tr = df[df["eligible"] & (df["t"] < train_hi)]
        va = df[df["eligible"] & (df["t"] >= val_lo) & (df["t"] < val_hi)]
        preds, _ = fit_predict_all(tr, va, x_cols)
        for name, p in preds.items():
            rows.append({"fold": k, "model": name, "val": f"{val_lo:%Y-%m-%d}..{val_hi:%Y-%m-%d}",
                         **metrics(va["y"], p)})
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_results.csv", index=False)
    print("\n[CV] mean over folds (pooled across regions):")
    print(cv.groupby("model")[["pr_auc", "pr_lift", "roc_auc", "brier"]].mean().round(4).to_string())

    # ---- final train on all dev data, evaluate once on the untouched test window
    tr = df[df["eligible"] & (df["t"] < dev_end)]
    te = df[df["eligible"] & (df["t"] >= test_start) & (df["t"] <= label_end)]
    preds, model = fit_predict_all(tr, te, x_cols)
    res = pd.DataFrame([{"model": n, **metrics(te["y"], p)} for n, p in preds.items()])
    res.to_csv(out / "test_results.csv", index=False)
    print("\n[TEST] pooled:")
    print(res.round(4).to_string(index=False))

    per = []
    for r in regions:
        m = (te["region"] == r).to_numpy()
        for n, p in preds.items():
            per.append({"region": r, "model": n, **metrics(te.loc[m, "y"], p[m])})
    per = pd.DataFrame(per)
    per.to_csv(out / "test_per_region.csv", index=False)
    piv = per.pivot(index="region", columns="model", values="pr_auc")[["base_region", "xgboost"]]
    piv["gain"] = piv["xgboost"] - piv["base_region"]
    print("\n[TEST] PR-AUC per region (xgboost vs region base rate):")
    print(piv.round(3).sort_values("gain").to_string())

    imp = pd.Series(model.feature_importances_, index=x_cols).sort_values(ascending=False)
    print("\n[xgboost] top features:\n", imp.head(10).round(3).to_string())

    # ---- calibration
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = preds["xgboost"]
    cal = pd.DataFrame({"p": p, "y": te["y"].to_numpy()})
    cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
    g = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    print("\n[TEST] calibration (xgboost):\n", g.round(4).to_string())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, g.values[:, :2].max()], [0, g.values[:, :2].max()], "k--", lw=1)
    ax.plot(g["pred"], g["obs"], "o-")
    ax.set(xlabel="predicted P(alert starts in next hour)", ylabel="observed frequency",
           title="Calibration, test window")
    fig.tight_layout()
    fig.savefig(out / "calibration_test.png", dpi=130)

    live = df[(df["t"] == origin)]
    print(f"\n[live] feature rows for forecast origin {origin}: {len(live)} (y unknown by definition)")
    print(f"[done] results written to {out.resolve()}")


if __name__ == "__main__":
    main()
