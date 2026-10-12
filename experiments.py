#!/usr/bin/env python3
"""
Experiments on top of airalert_pipeline.py (keep both files in the same folder).

  python airalert_experiments.py --stage start                    # MODEL_START sensitivity (CV only)
  python airalert_experiments.py --stage neighbors --start 2025-01-01   # none / data / geo / both (CV only)
  python airalert_experiments.py --stage tune --start 2025-01-01 --neighbors both   # XGBoost grid (CV only)
  python airalert_experiments.py --stage final --start ... --neighbors ... --params NAME   # test, run ONCE

Selection uses rolling-origin CV folds only. The test window is touched only by --stage final.
Extra dependency for geographic neighbours:  pip install shapely
"""
from __future__ import annotations

import argparse
import itertools
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import airalert_pipeline as pl

# =============================================================== EDIT HERE (1/3): start dates to compare
STARTS = ["2023-01-01", "2024-01-01", "2025-01-01", "2025-06-01", "2025-11-01"]

# =============================================================== EDIT HERE (2/3): XGBoost grid
BASE = dict(tree_method="hist", enable_categorical=True, eval_metric="logloss",
            n_jobs=-1, random_state=0, subsample=0.8, colsample_bytree=0.8)
PARAM_GRID = {
    "default":   dict(BASE, n_estimators=300, max_depth=5, learning_rate=0.05, min_child_weight=5),
    "shallow":   dict(BASE, n_estimators=400, max_depth=3, learning_rate=0.05, min_child_weight=20, reg_lambda=5),
    "very_shallow": dict(BASE, n_estimators=600, max_depth=2, learning_rate=0.05, min_child_weight=20, reg_lambda=5),
    "mid_reg":   dict(BASE, n_estimators=300, max_depth=4, learning_rate=0.03, min_child_weight=50, reg_lambda=10, gamma=1.0),
    "deep_reg":  dict(BASE, n_estimators=400, max_depth=6, learning_rate=0.03, min_child_weight=100, reg_lambda=20, gamma=1.0),
    "slow_shallow": dict(BASE, n_estimators=800, max_depth=3, learning_rate=0.02, min_child_weight=30, reg_lambda=10),
}

# =============================================================== geography (EDIT HERE 3/3 only if the URL changes)
GEO_URL = "https://raw.githubusercontent.com/EugeneBorshch/ukraine_geojson/master/UA_FULL_Ukraine.geojson"
ISO2REGION = {
    "UA-71": "Cherkaska oblast", "UA-74": "Chernihivska oblast", "UA-77": "Chernivetska oblast",
    "UA-12": "Dnipropetrovska oblast", "UA-14": "Donetska oblast", "UA-26": "Ivano-Frankivska oblast",
    "UA-63": "Kharkivska oblast", "UA-65": "Khersonska oblast", "UA-68": "Khmelnytska oblast",
    "UA-32": "Kyivska oblast", "UA-35": "Kirovohradska oblast", "UA-46": "Lvivska oblast",
    "UA-48": "Mykolaivska oblast", "UA-51": "Odeska oblast", "UA-53": "Poltavska oblast",
    "UA-56": "Rivnenska oblast", "UA-59": "Sumska oblast", "UA-61": "Ternopilska oblast",
    "UA-05": "Vinnytska oblast", "UA-07": "Volynska oblast", "UA-21": "Zakarpatska oblast",
    "UA-23": "Zaporizka oblast", "UA-18": "Zhytomyrska oblast",
    "UA-43": "Crimea (not modelled)", "UA-09": "Luhanska oblast",
}


def geo_neighbors(regions, path="data/UA_FULL_Ukraine.geojson", min_border_deg=0.05):
    """Land-border neighbours computed from OSM boundaries (shared border longer than ~5 km)."""
    import shapely
    from shapely.geometry import shape
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GEO_URL, p)
    feats = json.loads(p.read_text(encoding="utf-8"))["features"]
    geoms = {}
    for f in feats:
        code = f["properties"]["iso3166-2"]
        geoms[ISO2REGION[code]] = shapely.make_valid(shape(f["geometry"]))   # 3 southern oblasts are invalid
    adj = {r: set() for r in regions}
    for a, b in itertools.combinations(geoms, 2):
        if a in adj and b in adj:
            length = geoms[a].buffer(0.01).intersection(geoms[b].boundary).length
            if length > min_border_deg:
                adj[a].add(b)
                adj[b].add(a)
    # Kyiv City is an enclave of Kyivska oblast and is not a separate polygon in this file
    adj["Kyiv City"] = {"Kyivska oblast"}
    adj["Kyivska oblast"].add("Kyiv City")
    empty = [r for r, v in adj.items() if not v]
    if empty:
        raise ValueError(f"No geographic neighbours found for {empty}")
    return {r: sorted(v) for r, v in adj.items()}


def compare_neighbors(data_nb, geo_nb):
    rows = []
    for r in data_nb:
        d, g = set(data_nb[r]), set(geo_nb[r])
        rows.append({"region": r, "data_top3": ", ".join(x.replace(" oblast", "") for x in data_nb[r]),
                     "n_geo": len(g), "data_in_geo": len(d & g),
                     "jaccard": round(len(d & g) / len(d | g), 2)})
    t = pd.DataFrame(rows)
    chance = np.mean([len(geo_nb[r]) / (len(geo_nb) - 1) for r in geo_nb])
    print(t.to_string(index=False))
    print(f"\nShare of data-picked neighbours that are geographic neighbours: "
          f"{t['data_in_geo'].sum() / (pl.TOP_K * len(t)):.1%} (random picks would give ~{chance:.1%})")
    return t


# =============================================================== shared helpers
def setup():
    alerts, data_end = pl.load_clean(Path(ARGS.data))
    regions = sorted(alerts["region"].unique())
    g0 = alerts["started_at"].min().floor("h")
    origin = data_end.floor("h")
    label_end = origin - pd.Timedelta(hours=1)
    folds, dev_end, test_start = pl.split_bounds(label_end)
    return alerts, regions, g0, origin, label_end, folds, dev_end, test_start


def data_neighbors(alerts, regions, g0, first_train_hi, start):
    old = pl.MODEL_START
    pl.MODEL_START = pd.Timestamp(start, tz="UTC")
    nb = pl.select_neighbors(alerts, regions, g0, first_train_hi)
    pl.MODEL_START = old
    return nb


def make_neighbors(mode, alerts, regions, g0, first_train_hi, start):
    if mode == "none":
        return {r: [] for r in regions}
    data = data_neighbors(alerts, regions, g0, first_train_hi, start)
    if mode == "data":
        return data
    geo = geo_neighbors(regions)
    if mode == "geo":
        return geo
    return {r: sorted(set(data[r]) | set(geo[r])) for r in regions}      # both


def build(alerts, regions, g0, origin, label_end, nb, build_start):
    pl.MODEL_START = pd.Timestamp(build_start, tz="UTC")
    df = pl.build_features(alerts, regions, nb, g0, origin)
    df.loc[df["t"] > label_end, "y"] = np.nan
    df["eligible"] = (~df["active"]) & df["y"].notna()
    feat = [c for c in df.columns if c not in pl.NON_FEATURE and df[c].dtype != "category"]
    return df, feat + ["region"]


def run_cv(df, x_cols, params, folds):
    from xgboost import XGBClassifier
    rows = []
    for k, (train_hi, val_lo, val_hi) in enumerate(folds):
        tr = df[df["eligible"] & (df["t"] < train_hi)]
        va = df[df["eligible"] & (df["t"] >= val_lo) & (df["t"] < val_hi)]
        m = XGBClassifier(**params)
        m.fit(tr[x_cols], tr["y"].astype(int), eval_set=[(va[x_cols], va["y"].astype(int))], verbose=False)
        p = m.predict_proba(va[x_cols])[:, 1]
        ll = np.array(m.evals_result()["validation_0"]["logloss"])
        mt = pl.metrics(va["y"], p)
        bm = pl.metrics(va["y"], pl.rate_model(tr, ["region"])(va))
        rows.append({"fold": k, "n_train": len(tr), "pr_auc": mt["pr_auc"], "roc_auc": mt["roc_auc"],
                     "brier": mt["brier"], "base_pr_auc": bm["pr_auc"], "base_brier": bm["brier"],
                     "ll_last": ll[-1], "ll_min": ll.min(), "best_iter": int(ll.argmin()) + 1})
    return pd.DataFrame(rows)


def summarize(cv):
    return pd.Series({"pr_auc": cv["pr_auc"].mean(), "pr_auc_sd": cv["pr_auc"].std(),
                      "roc_auc": cv["roc_auc"].mean(), "brier": cv["brier"].mean(),
                      "d_pr_auc_vs_base": (cv["pr_auc"] - cv["base_pr_auc"]).mean(),
                      "d_brier_vs_base": (cv["brier"] - cv["base_brier"]).mean(),
                      "ll_last": cv["ll_last"].mean(), "ll_min": cv["ll_min"].mean(),
                      "best_iter_med": cv["best_iter"].median(), "n_train_fold0": cv["n_train"].iloc[0]})


# =============================================================== stages
def stage_start():
    alerts, regions, g0, origin, label_end, folds, *_ = setup()
    # neighbours fixed (chosen from 2025-01-01..first train cutoff) so ONLY the training window varies
    nb = data_neighbors(alerts, regions, g0, folds[0][0], "2025-01-01")
    df, x_cols = build(alerts, regions, g0, origin, label_end, nb, min(STARTS))
    out = {}
    for s in STARTS:
        sub = df[df["t"] >= pd.Timestamp(s, tz="UTC")]
        out[s] = summarize(run_cv(sub, x_cols, PARAM_GRID["default"], folds))
        print(f"  done {s}")
    res = pd.DataFrame(out).T
    print("\n[start sensitivity] CV, identical validation windows, default XGBoost params")
    print(res.round(4).to_string())
    res.to_csv(Path(ARGS.out) / "sens_start.csv")


def stage_neighbors():
    alerts, regions, g0, origin, label_end, folds, *_ = setup()
    data_nb = data_neighbors(alerts, regions, g0, folds[0][0], ARGS.start)
    geo_nb = geo_neighbors(regions)
    print("[neighbour comparison] data-derived top-3 vs geographic land borders")
    compare_neighbors(data_nb, geo_nb).to_csv(Path(ARGS.out) / "neighbors_compare.csv", index=False)
    out = {}
    for mode in ("none", "data", "geo", "both"):
        nb = make_neighbors(mode, alerts, regions, g0, folds[0][0], ARGS.start)
        df, x_cols = build(alerts, regions, g0, origin, label_end, nb, ARGS.start)
        out[mode] = summarize(run_cv(df, x_cols, PARAM_GRID["default"], folds))
        print(f"  done {mode}")
    res = pd.DataFrame(out).T
    print(f"\n[neighbour modes] CV, start={ARGS.start}, default XGBoost params")
    print(res.round(4).to_string())
    res.to_csv(Path(ARGS.out) / "sens_neighbors.csv")


def stage_tune():
    alerts, regions, g0, origin, label_end, folds, *_ = setup()
    nb = make_neighbors(ARGS.neighbors, alerts, regions, g0, folds[0][0], ARGS.start)
    df, x_cols = build(alerts, regions, g0, origin, label_end, nb, ARGS.start)
    out = {}
    for name, prm in PARAM_GRID.items():
        out[name] = summarize(run_cv(df, x_cols, prm, folds))
        print(f"  done {name}")
    res = pd.DataFrame(out).T.sort_values("brier")
    print(f"\n[tuning] CV mean over {len(folds)} folds, start={ARGS.start}, neighbours={ARGS.neighbors}")
    print("(ll_last vs ll_min / best_iter_med: a big gap means too many trees -> overfitting)")
    print(res.round(4).to_string())
    res.to_csv(Path(ARGS.out) / "tuning_cv.csv")


def stage_final():
    from xgboost import XGBClassifier
    alerts, regions, g0, origin, label_end, folds, dev_end, test_start = setup()
    nb = make_neighbors(ARGS.neighbors, alerts, regions, g0, folds[0][0], ARGS.start)
    df, x_cols = build(alerts, regions, g0, origin, label_end, nb, ARGS.start)
    tr = df[df["eligible"] & (df["t"] < dev_end)]
    te = df[df["eligible"] & (df["t"] >= test_start) & (df["t"] <= label_end)]
    m = XGBClassifier(**PARAM_GRID[ARGS.params])
    m.fit(tr[x_cols], tr["y"].astype(int))
    p = m.predict_proba(te[x_cols])[:, 1]
    rows = {"base_region": pl.rate_model(tr, ["region"])(te), "xgboost_tuned": p}
    res = pd.DataFrame({n: pl.metrics(te["y"], v) for n, v in rows.items()}).T
    print(f"[FINAL TEST] start={ARGS.start} neighbours={ARGS.neighbors} params={ARGS.params}")
    print(res.round(4).to_string())
    res.to_csv(Path(ARGS.out) / "final_test.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["start", "neighbors", "tune", "final"])
    ap.add_argument("--data", default="data/volunteer_data_en.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--neighbors", default="data", choices=["none", "data", "geo", "both"])
    ap.add_argument("--params", default="default", choices=list(PARAM_GRID))
    ARGS = ap.parse_args()
    Path(ARGS.out).mkdir(exist_ok=True)
    {"start": stage_start, "neighbors": stage_neighbors, "tune": stage_tune, "final": stage_final}[ARGS.stage]()