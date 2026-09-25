#!/usr/bin/env python3
"""fast_qstats_g1.py — add QUANTILES stats (q01/q10/q50/q90/q99) to a v3.0 LeRobot dataset's meta/stats.json
for observation.state + action ONLY (MolmoAct2 normalization_mapping: STATE=QUANTILES, ACTION=QUANTILES,
VISUAL=IDENTITY ⇒ images need NO quantiles). Faithful to lerobot.scripts.augment_dataset_quantile_stats
(per-episode get_feature_stats + aggregate_stats) but reads state/action from parquet ⇒ NO video decode
(the official one is ~5.5s/ep decoding all frames = ~4.7h for 3094 ep; this is ~1-2 min).

Usage: python fast_qstats_g1.py --root /root/g1_datasets/merged
"""
import argparse, glob, json
import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import get_feature_stats, aggregate_stats

QUANTILES = [0.01, 0.1, 0.5, 0.9, 0.99]
FEATURES = ["observation.state", "action"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = a.root
    stats_path = f"{root}/meta/stats.json"
    stats = json.load(open(stats_path))

    # read all v3.0 data parquets
    files = sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))
    assert files, f"no data parquets under {root}/data"
    df = pd.concat([pd.read_parquet(f, columns=FEATURES + ["episode_index"]) for f in files], ignore_index=True)
    n_ep = df["episode_index"].nunique()
    print(f"[qstats] {len(df)} frames, {n_ep} episodes across {len(files)} parquet file(s)")

    # per-episode stats (faithful to official: get_feature_stats per episode, then aggregate_stats)
    per_ep = []
    for ep, g in df.groupby("episode_index", sort=True):
        est = {}
        for feat in FEATURES:
            arr = np.stack(g[feat].to_numpy()).astype(np.float32)  # (ep_len, dim)
            est[feat] = get_feature_stats(arr, axis=0, keepdims=(arr.ndim == 1), quantile_list=QUANTILES)
        per_ep.append(est)
    agg = aggregate_stats(per_ep)  # {feat: {min,max,mean,std,count,q01,q10,q50,q90,q99}}

    # merge quantiles (and refreshed scalar stats) into stats.json for state+action; keep image stats as-is
    for feat in FEATURES:
        merged = dict(stats.get(feat, {}))
        for k, v in agg[feat].items():
            merged[k] = v.tolist() if isinstance(v, np.ndarray) else v
        stats[feat] = merged
        qk = [k for k in stats[feat] if k.startswith("q")]
        print(f"[qstats] {feat}: keys now {list(stats[feat].keys())}  q01={np.round(np.asarray(stats[feat]['q01']),4)}")

    json.dump(stats, open(stats_path, "w"))
    print(f"[qstats] wrote {stats_path} — state+action now QUANTILES-ready (images untouched, IDENTITY)")


if __name__ == "__main__":
    main()
