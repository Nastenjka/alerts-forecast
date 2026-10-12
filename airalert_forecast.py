#!/usr/bin/env python3
"""
Approximate alert-start probabilities per region for the next HOUR and the next DAY.

    python airalert_forecast.py --refresh              # download latest data, then forecast
    python airalert_forecast.py --refresh --evaluate   # also backtest both horizons on the last 6 weeks

Output per region:
    p_1h   P(a NEW alert starts in [origin, origin+1h))      (not defined if the region is in alert at origin)
    p_24h  P(at least one NEW alert starts in [origin, origin+24h))

Needs airalert_pipeline.py in the same folder. This is a rough statistical estimate built from past alert
timing only -- it is NOT a warning system. Always follow official alerts.

Leakage rules (same as the pipeline):
  * features at `origin` use only events before `origin`;
  * a horizon-H model is trained only on rows whose whole label window [t, t+H) ended before `origin`
    (for the backtest: before the test window starts).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import airalert_pipeline as pl

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
CSV_URL = f"https://raw.githubusercontent.com/{REPO}/main/datasets/volunteer_data_en.csv"
COMMIT_API = f"https://api.github.com/repos/{REPO}/commits?path=datasets/volunteer_data_en.csv&per_page=1"
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, tree_method="hist", enable_categorical=True,
              eval_metric="logloss", n_jobs=-1, random_state=0)
STALE_HOURS = 2.0


# ------------------------------------------------------------------ data refresh
def refresh(path: Path) -> None:
    """Download the newest CSV; also try to record when the repo last updated it (= how far the data reaches).
    The commit-time lookup uses the unauthenticated GitHub API (rate-limited); if it fails we still keep the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = {"User-Agent": "airalert-forecast"}
    with urllib.request.urlopen(urllib.request.Request(CSV_URL, headers=hdr), timeout=120) as r:
        path.write_bytes(r.read())
    meta = path.with_suffix(".meta.json")
    meta.unlink(missing_ok=True)
    print(f"[refresh] downloaded {path} ({path.stat().st_size / 1e6:.1f} MB)")
    try:
        with urllib.request.urlopen(urllib.request.Request(COMMIT_API, headers=hdr), timeout=30) as r:
            commit_time = json.load(r)[0]["commit"]["committer"]["date"]
        meta.write_text(json.dumps({"commit_time": commit_time,
                                    "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat()}))
        print(f"[refresh] repo last updated this file at {commit_time}")
    except Exception as e:
        print(f"[refresh] could not read the commit time ({e}); data coverage is assumed to end at the last event")


def coverage_end(data_end: pd.Timestamp, path: Path) -> pd.Timestamp:
    """The data cannot be assumed to reach beyond the last event unless the repo's commit time says so."""
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        ct = pd.Timestamp(json.loads(meta.read_text())["commit_time"])
        return max(data_end, ct)
    return data_end


# ------------------------------------------------------------------ targets
def add_target(df: pd.DataFrame, alerts: pd.DataFrame, H: int, origin: pd.Timestamp) -> np.ndarray:
    """y_H = 1 if any alert STARTS in [t, t+H hours). NaN where the window is not fully observed."""
    s_by = {r: np.sort(pl.to_min(g["started_at"])) for r, g in alerts.groupby("region")}
    tm = pl.to_min(df["t"])
    reg = df["region"].astype(str).to_numpy()
    y = np.full(len(df), np.nan)
    for r, s in s_by.items():
        m = reg == r
        lo = np.searchsorted(s, tm[m], side="left")
        hi = np.searchsorted(s, tm[m] + 60.0 * H, side="left")
        y[m] = (hi > lo).astype(float)
    y[(df["t"] + pd.Timedelta(hours=H) > origin).to_numpy()] = np.nan
    return y


def horizon_setup(df, H, feat):
    """H=1 follows the validated design (rows already in alert are excluded: a new start is ~impossible).
    H>1 keeps all rows and gives the model `active_now` as a feature."""
    if H == 1:
        return ~df["active"].to_numpy(), feat + ["region"]
    return np.ones(len(df), dtype=bool), feat + ["active_now", "region"]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/volunteer_data_en.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--refresh", action="store_true", help="download the latest CSV first")
    ap.add_argument("--evaluate", action="store_true", help="backtest both horizons on the last 6 weeks")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 24])
    a = ap.parse_args()
    data_path, out = Path(a.data), Path(a.out)
    out.mkdir(exist_ok=True)

    if a.refresh:
        try:
            refresh(data_path)
        except Exception as e:                                   # network problems must not kill the run
            print(f"[refresh] FAILED ({e}); using the existing file")
    alerts, data_end = pl.load_clean(data_path)
    regions = sorted(alerts["region"].unique())
    g0 = alerts["started_at"].min().floor("h")
    cov = coverage_end(data_end, data_path)
    origin = cov.floor("h")
    folds, _, _ = pl.split_bounds(origin - pd.Timedelta(hours=1))

    nb = pl.select_neighbors(alerts, regions, g0, folds[0][0])   # data before the first CV cutoff only
    df = pl.build_features(alerts, regions, nb, g0, origin)
    df["active_now"] = df["active"].astype("float32")
    feat = [c for c in df.columns if c not in pl.NON_FEATURE and c != "active_now" and df[c].dtype != "category"]

    # consistency check: the new target code must reproduce the validated 1-hour label exactly
    y1 = add_target(df, alerts, 1, origin)
    ref = df["y"].where(df["t"] <= origin - pd.Timedelta(hours=1)).to_numpy()
    ok = ~np.isnan(y1) & ~np.isnan(ref)
    assert (y1[ok] == ref[ok]).all(), "target mismatch vs validated 1-hour label"

    now = pd.Timestamp.now(tz="UTC")
    behind = (now - cov).total_seconds() / 3600
    print(f"[time] data observed to {data_end:%Y-%m-%d %H:%M} UTC | assumed coverage {cov:%Y-%m-%d %H:%M} UTC | "
          f"forecast origin {origin:%Y-%m-%d %H:%M} UTC")
    if behind > STALE_HOURS:
        print(f"\n*** WARNING: the data is ~{behind:.0f} h behind the clock ({now:%H:%M} UTC). The forecast windows "
              f"below start at {origin:%H:%M} UTC, i.e. in the past. Run with --refresh later for a current "
              f"forecast. ***\n")

    live = df[df["t"] == origin].set_index("region").loc[regions].reset_index()
    calm = ~live["active"].to_numpy()
    res = pd.DataFrame({"region": live["region"].astype(str),
                        "status": np.where(calm, "calm", "IN ALERT")})
    bt = []

    for H in a.horizons:
        y = add_target(df, alerts, H, origin)
        use, x_cols = horizon_setup(df, H, feat)
        lab = use & ~np.isnan(y)                                  # label window fully observed before origin
        train = df[lab].assign(y=y[lab])
        model = XGBClassifier(**PARAMS).fit(train[x_cols], train["y"].astype(int))
        name = "p_1h" if H == 1 else f"p_{H}h"
        p = np.full(len(live), np.nan)
        rows = calm if H == 1 else np.ones(len(live), dtype=bool)
        p[rows] = model.predict_proba(live.loc[rows, x_cols])[:, 1]
        res[name] = p
        recent = train[train["t"] > origin - pd.Timedelta(days=28)].groupby("region", observed=True)["y"].mean()
        res[f"base28_{H}h"] = res["region"].map(recent.rename(index=str))
        print(f"[H={H}h] trained on {len(train)} rows, positive rate {train['y'].mean():.3f}")

        if a.evaluate:
            test_start = origin - pd.Timedelta(days=pl.TEST_DAYS)
            tr_m = use & ~np.isnan(y) & (df["t"] + pd.Timedelta(hours=H) <= test_start).to_numpy()
            te_m = use & ~np.isnan(y) & (df["t"] >= test_start).to_numpy()
            tr, te = df[tr_m].assign(y=y[tr_m]), df[te_m].assign(y=y[te_m])
            m2 = XGBClassifier(**PARAMS).fit(tr[x_cols], tr["y"].astype(int))
            pe = m2.predict_proba(te[x_cols])[:, 1]
            pb = pl.rate_model(tr, ["region"])(te)
            bt.append({"horizon_h": H, "model": "xgboost", **pl.metrics(te["y"], pe)})
            bt.append({"horizon_h": H, "model": "region_base_rate", **pl.metrics(te["y"], pb)})
            cal = pd.DataFrame({"p": pe, "y": te["y"].to_numpy()})
            cal["bin"] = pd.qcut(cal["p"], 5, duplicates="drop")
            print(f"[backtest H={H}h] calibration, test {test_start:%m-%d}..{origin - pd.Timedelta(hours=H):%m-%d}:")
            print(cal.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                                         n=("y", "size")).round(3).to_string())

    if "p_1h" in res and "p_24h" in res:                         # an alert within 1h implies one within 24h
        both = res["p_1h"].notna()
        res.loc[both, "p_24h"] = np.maximum(res.loc[both, "p_24h"], res.loc[both, "p_1h"])

    sort_col = "p_24h" if "p_24h" in res else res.columns[2]
    res = res.sort_values(sort_col, ascending=False)
    res.to_csv(out / "forecast.csv", index=False)

    print(f"\nFORECAST from {origin:%Y-%m-%d %H:%M} UTC  (Kyiv time: {origin.tz_convert('Europe/Kyiv'):%H:%M})\n")
    show = res.copy()
    for c in [c for c in show.columns if c.startswith(("p_", "base28"))]:
        show[c] = show[c].map(lambda v: "   n/a" if pd.isna(v) else f"{v:5.0%}")
    print(show.to_string(index=False))
    print("\nbase28_* = how often that happened in the same region over the last 28 days (for context).")
    print("p_1h is n/a for regions already in alert. Rough estimate only; follow official alerts.")

    if bt:
        b = pd.DataFrame(bt)
        b.to_csv(out / "backtest.csv", index=False)
        print("\n[backtest] last 6 weeks, pooled over regions:")
        print(b[["horizon_h", "model", "pos_rate", "pr_auc", "roc_auc", "brier"]].round(3).to_string(index=False))
    print(f"\n[done] {out.resolve() / 'forecast.csv'}")


if __name__ == "__main__":
    main()